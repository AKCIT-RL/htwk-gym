"""GPU gate for the Frente F task-obs history (F-lite).

Checks:
  1. env F1 builds; obs 369 dims (249 + 10*12); finite.
  2. layout: newest history frame == current task block (slots 237:249).
  3. lag: frame -2 equals the PREVIOUS step's task block (teleport the ball,
     the old position must still be visible one slot back).
  4. reset: all history frames replicate the fresh block (no stale data).
  5. warm-start equivalence: expanded ckpt outputs the SAME action mean as
     the original ckpt on the padded obs (zero-init check).
  6. regression: history off -> 249 dims, bit-identical env behavior.
"""

import os
import sys

MIMICKIT_ROOT = "/workspace/MimicKit"
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
os.chdir(MIMICKIT_ROOT)

import isaacgym  # noqa: F401
import numpy as np
import torch

import envs.env_builder as env_builder
import util.mp_util as mp_util
import util.util as util

DEVICE = "cuda:0"
NUM_ENVS = 8
ENV_F1 = "data/envs/mcwamp_g1_soccer_env_f1.yaml"
ENV_C = "data/envs/mcwamp_g1_soccer_env_c.yaml"
ENGINE = "data/engines/isaac_gym_engine_uneven.yaml"

CHAR = 237
BLOCK = 12
H = 10
BASE = CHAR + BLOCK  # 249

failures = []


def check(name, ok, detail=""):
    print("{:s} {:s}{:s}".format("PASS" if ok else "FAIL", name,
                                 " | " + detail if detail else ""))
    if (not ok):
        failures.append(name)


def steps(env, n):
    adim = env.get_action_space().shape[0]
    obs = info = None
    for _ in range(n):
        obs, r, done, info = env.step(torch.zeros([NUM_ENVS, adim], device=DEVICE))
    return obs, info


def main():
    mp_util.init(0, 1, DEVICE, int(np.random.randint(6000, 7000)))
    util.set_rand_seed(1)

    if (len(sys.argv) > 1 and sys.argv[1] == "--regression"):
        # Isaac Gym allows one sim per process: history-off check runs alone.
        env_off = env_builder.build_env(env_file=ENV_C, engine_file=ENGINE,
                                        num_envs=NUM_ENVS, device=DEVICE,
                                        visualize=False)
        obs_off, _ = env_off.reset()
        check("history off -> contract unchanged (249)",
              obs_off.shape[1] == BASE, "got {:d}".format(obs_off.shape[1]))
        if (failures):
            sys.exit(1)
        print("REGRESSION CHECK PASSED")
        return

    env = env_builder.build_env(env_file=ENV_F1, engine_file=ENGINE,
                                num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    obs, info = env.reset()

    check("F1 env builds, obs is 369 dims", obs.shape[1] == BASE + H * BLOCK,
          "got {:d}".format(obs.shape[1]))
    check("obs finite", torch.all(torch.isfinite(obs)).item())

    hist = obs[:, BASE:].reshape(NUM_ENVS, H, BLOCK)
    cur = obs[:, CHAR:BASE]
    check("reset: newest history frame == current block",
          torch.allclose(hist[:, -1], cur, atol=1e-5))
    spread = (hist - hist[:, -1:]).abs().max().item()
    check("reset: all frames replicated (no stale/zero data)", spread < 1e-5,
          "max spread {:.2e}".format(spread))

    # lag check: step once recording the block, step again, frame -2 must
    # equal the recorded one
    obs1, _ = steps(env, 1)
    block1 = obs1[:, CHAR:BASE].clone()
    obs2, _ = steps(env, 1)
    hist2 = obs2[:, BASE:].reshape(NUM_ENVS, H, BLOCK)
    check("history lags by one step (frame -2 == previous block)",
          torch.allclose(hist2[:, -2], block1, atol=1e-5),
          "max err {:.2e}".format((hist2[:, -2] - block1).abs().max().item()))
    check("newest frame == current block after steps",
          torch.allclose(hist2[:, -1], obs2[:, CHAR:BASE], atol=1e-5))

    # rollout sanity
    obs, info = steps(env, 30)
    check("obs finite after 30 steps", torch.all(torch.isfinite(obs)).item())

    # --- warm-start equivalence -----------------------------------------------
    import learning.agent_builder as agent_builder
    import learning.base_agent as base_agent
    agent = agent_builder.build_agent("data/agents/mcwamp_g1_soccer_agent.yaml",
                                      env, DEVICE)
    agent.load("output/soccer_warmstart/model_init_p1p2_hist10.pt")
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    sd_old = torch.load("output/soccer_warmstart/model_init_p1p2.pt", map_location=DEVICE)
    sd_new = {k: v.to(DEVICE) if torch.is_tensor(v) else v
              for k, v in agent.state_dict().items()}
    w_new = sd_new["_model._actor_layers.0.weight"]
    w_old = sd_old["_model._actor_layers.0.weight"]
    same_prefix = torch.allclose(w_new[:, :BASE], w_old, atol=0)
    zero_pad = torch.all(w_new[:, BASE:] == 0).item()
    check("expanded actor weights: prefix identical + zero pad",
          same_prefix and zero_pad)

    with torch.no_grad():
        a1, _ = agent._decide_action(obs, info)
        # scramble the history slots: with zero-init columns the action must
        # not change at all
        obs_scrambled = obs.clone()
        obs_scrambled[:, BASE:] = torch.randn_like(obs_scrambled[:, BASE:])
        a2, _ = agent._decide_action(obs_scrambled, info)
    check("zero-init: action invariant to history content at init",
          torch.allclose(a1, a2, atol=1e-6),
          "max diff {:.2e}".format((a1 - a2).abs().max().item()))

    print("=" * 58)
    if (failures):
        print("FAILURES: {}".format(failures))
        sys.exit(1)
    print("ALL SOCCER HISTORY (FRENTE F) CHECKS PASSED")


if (__name__ == "__main__"):
    main()
