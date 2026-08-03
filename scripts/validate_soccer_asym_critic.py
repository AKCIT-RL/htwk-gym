"""GPU gate for the asymmetric privileged critic (Frente F3).

Env f2 (virtual perception + history) must publish a critic_obs in the info
with the TRUE ball state while the actor obs keeps the perceived one, and the
MCWAMP agent must route the critics to it with a seeded normalizer.
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
ENV_F2 = "data/envs/mcwamp_g1_soccer_env_f2.yaml"
ENGINE = "data/engines/isaac_gym_engine_uneven.yaml"

CHAR = 237
BLOCK = 12
BASE = CHAR + BLOCK

failures = []


def check(name, ok, detail=""):
    print("{:s} {:s}{:s}".format("PASS" if ok else "FAIL", name,
                                 " | " + detail if detail else ""))
    if (not ok):
        failures.append(name)


def main():
    mp_util.init(0, 1, DEVICE, int(np.random.randint(6000, 7000)))
    util.set_rand_seed(1)

    env = env_builder.build_env(env_file=ENV_F2, engine_file=ENGINE,
                                num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    obs, info = env.reset()
    H = env._task_hist_steps

    check("env publishes critic_obs on reset", "critic_obs" in info)
    cobs = info["critic_obs"]
    check("critic_obs shape == obs shape", cobs.shape == obs.shape,
          "{} vs {}".format(tuple(cobs.shape), tuple(obs.shape)))
    check("critic_obs finite", torch.all(torch.isfinite(cobs)).item())
    check("char prefix identical (only ball slots may differ)",
          torch.allclose(cobs[:, :CHAR], obs[:, :CHAR], atol=1e-5))
    check("critic mask is always 1 (true ball never drops)",
          torch.all(cobs[:, BASE - 1] == 1.0).item())

    adim = env.get_action_space().shape[0]
    diffs = []
    mask_col = []
    for _ in range(60):
        obs, r, done, info = env.step(torch.zeros([NUM_ENVS, adim], device=DEVICE))
        cobs = info["critic_obs"]
        diffs.append((cobs[:, CHAR:BASE - 1] - obs[:, CHAR:BASE - 1]).abs().max().item())
        mask_col.append(cobs[:, BASE - 1].min().item())
    check("critic sees true ball: task block diverges from perceived obs",
          max(diffs) > 1e-3, "max abs diff {:.3f}".format(max(diffs)))
    check("critic mask stays 1 during rollout", min(mask_col) == 1.0)
    check("critic_obs finite after 60 steps", torch.all(torch.isfinite(cobs)).item())

    chist = cobs[:, BASE:].reshape(NUM_ENVS, H, BLOCK)
    check("critic history newest frame == privileged block",
          torch.allclose(chist[:, -1],
                         env._compute_task_block(privileged=True), atol=1e-5))

    # --- agent side ------------------------------------------------------------
    import learning.agent_builder as agent_builder
    agent = agent_builder.build_agent("data/agents/mcwamp_g1_soccer_agent.yaml",
                                      env, DEVICE)
    check("agent enables asymmetric critic path", agent._use_critic_obs)
    check("agent owns a critic_obs normalizer", hasattr(agent, "_critic_obs_norm"))

    agent.load("output/soccer_warmstart/model_init_p1p2_hist10.pt")
    same = torch.allclose(agent._critic_obs_norm._mean, agent._obs_norm._mean) \
        and torch.allclose(agent._critic_obs_norm._std, agent._obs_norm._std)
    check("old ckpt load seeds critic norm from obs norm", same)

    # one training iteration end-to-end (buffer keys, shapes, losses finite)
    import learning.base_agent as base_agent
    agent.set_mode(base_agent.AgentMode.TRAIN)
    agent._init_train()
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    with torch.no_grad():
        agent._rollout_train(agent._steps_per_iter)
    data_info = agent._build_train_data()
    train_info = agent._update_model()
    critic_loss = float(train_info.get("critic_loss", float("nan")))
    check("one train iter runs with critic_obs in the buffer",
          np.isfinite(critic_loss), "critic_loss {:.3f}".format(critic_loss))
    check("buffer holds critic_obs and next_critic_obs",
          "critic_obs" in agent._exp_buffer._buffers
          and "next_critic_obs" in agent._exp_buffer._buffers
          if hasattr(agent._exp_buffer, "_buffers") else True)

    print("=" * 58)
    if (failures):
        print("FAILURES: {}".format(failures))
        sys.exit(1)
    print("ALL ASYMMETRIC CRITIC (FRENTE F3) CHECKS PASSED")


if (__name__ == "__main__"):
    main()
