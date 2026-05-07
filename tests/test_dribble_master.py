#!/usr/bin/env python3
"""Validation tests for DribbleMaster – run inside Docker before full training.

Usage (inside Docker container):
  python3 tests/test_dribble_master.py

Each test isolates one layer of the stack so failures point to the exact problem.
Tests are ordered from cheapest to most expensive (CPU-only → GPU sim).
"""

import sys
import os
import math
import traceback

# Isaac Gym MUST be imported before torch – do it once at the top
try:
    import isaacgym  # noqa: F401
    _HAS_ISAACGYM = True
except ImportError:
    _HAS_ISAACGYM = False

import torch  # safe to import after isaacgym

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Helpers ──────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def _run(name, fn):
    global _passed, _failed
    print(f"  [{name}] ", end="", flush=True)
    try:
        fn()
        print("✓")
        _passed += 1
    except Exception as e:
        print(f"✗  →  {e}")
        traceback.print_exc()
        _failed += 1


def _header(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. YAML Config Parsing (CPU-only, no Isaac Gym)
# ══════════════════════════════════════════════════════════════════════════════
def test_yaml_configs():
    _header("1. YAML Config Parsing")
    import yaml

    for stage in ("Stage1", "Stage2"):
        path = f"envs/T1/Dribble_Master_{stage}.yaml"

        def _parse(p=path, s=stage):
            with open(p) as f:
                cfg = yaml.safe_load(f)
            assert cfg["env"]["num_observations"] == 39, f"Expected 39 obs, got {cfg['env']['num_observations']}"
            assert cfg["env"]["num_actions"] == 14, f"Expected 14 actions, got {cfg['env']['num_actions']}"
            assert cfg["env"]["num_privileged_obs"] == 20, f"Expected 20 priv obs"
            assert "virtual_camera" in cfg, "Missing virtual_camera section"
            assert "ball" in cfg, "Missing ball section"
            assert "init_distance_range" in cfg["ball"], "Missing ball.init_distance_range"
            assert "model" in cfg, "Missing model section"
            assert cfg["model"]["actor_hidden_dims"] == [512, 256, 128], f"Wrong actor dims"
            assert cfg["model"]["critic_hidden_dims"] == [768, 256, 128], f"Wrong critic dims"
            # Stage-specific checks
            if s == "Stage1":
                assert cfg["virtual_camera"]["hfov_deg"] > 100, "Stage1 should have wide FOV"
                assert cfg["rewards"]["scales"].get("ball_velocity_tracking", 0) == 0, \
                    "Stage1 should have ball_velocity_tracking=0"
            else:
                assert cfg["virtual_camera"]["hfov_deg"] < 100, "Stage2 should have real FOV"
                assert cfg["rewards"]["scales"].get("ball_velocity_tracking", 0) > 0, \
                    "Stage2 should have ball_velocity_tracking>0"

        _run(f"Parse {stage} YAML", _parse)

    def _check_reward_functions_exist():
        """Verify every reward name in the YAML has a matching _reward_<name> method.
        Uses AST parsing to avoid import issues with isaacgym."""
        import yaml
        import ast
        with open(os.path.join(project_root, "envs", "T1", "dribble_master.py")) as f:
            tree = ast.parse(f.read())
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DribbleMaster":
                class_def = node
                break
        assert class_def is not None, "DribbleMaster class not found"
        methods = {n.name for n in class_def.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for stage in ("Stage1", "Stage2"):
            with open(f"envs/T1/Dribble_Master_{stage}.yaml") as f:
                cfg = yaml.safe_load(f)
            for name in cfg["rewards"]["scales"]:
                method = f"_reward_{name}"
                assert method in methods, \
                    f"{stage}: reward '{name}' has no method DribbleMaster.{method}"

    _run("All reward methods exist", _check_reward_functions_exist)


# ══════════════════════════════════════════════════════════════════════════════
# 2. URDF Validation (CPU-only)
# ══════════════════════════════════════════════════════════════════════════════
def test_urdf():
    _header("2. URDF Validation")
    import xml.etree.ElementTree as ET

    def _check_urdf():
        tree = ET.parse("resources/T1/T1_dribble.urdf")
        root = tree.getroot()
        joints = {j.get("name"): j.get("type") for j in root.findall(".//joint")}

        # Head joints must be revolute (not fixed)
        assert joints.get("AAHead_yaw") == "revolute", \
            f"AAHead_yaw should be revolute, got {joints.get('AAHead_yaw')}"
        assert joints.get("Head_pitch") == "revolute", \
            f"Head_pitch should be revolute, got {joints.get('Head_pitch')}"

        # Count revolute joints (should be 14: 12 leg + 2 head)
        revolute = [n for n, t in joints.items() if t == "revolute"]
        assert len(revolute) == 14, f"Expected 14 revolute joints, got {len(revolute)}: {revolute}"

        # Arm joints should still be fixed
        for name, typ in joints.items():
            if "Arm" in name or "Elbow" in name or "Wrist" in name:
                assert typ == "fixed", f"Arm joint {name} should be fixed, got {typ}"

    _run("T1_dribble.urdf structure", _check_urdf)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Architecture (CPU-only, needs torch)
# ══════════════════════════════════════════════════════════════════════════════
def test_model():
    _header("3. Model Architecture (CPU)")
    import torch

    def _check_model_default():
        from utils.model import ActorCritic
        m = ActorCritic(num_act=14, num_obs=39, num_privileged_obs=20)
        # Default dims [256,128,128] / [256,256,128]
        obs = torch.randn(4, 39)
        priv = torch.randn(4, 20)
        dist = m.act(obs)
        assert dist.sample().shape == (4, 14), f"Wrong action shape: {dist.sample().shape}"
        val = m.est_value(obs, priv)
        assert val.shape == (4,), f"Wrong value shape: {val.shape}"

    def _check_model_custom():
        from utils.model import ActorCritic
        m = ActorCritic(
            num_act=14, num_obs=39, num_privileged_obs=20,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[768, 256, 128],
        )
        obs = torch.randn(8, 39)
        priv = torch.randn(8, 20)
        dist = m.act(obs)
        assert dist.sample().shape == (8, 14)
        val = m.est_value(obs, priv)
        assert val.shape == (8,)

        # Verify layer sizes
        # Actor: 39→512→256→128→14
        assert m.actor[0].in_features == 39
        assert m.actor[0].out_features == 512
        assert m.actor[2].out_features == 256
        assert m.actor[4].out_features == 128
        assert m.actor[6].out_features == 14
        # Critic: 59→768→256→128→1
        assert m.critic[0].in_features == 59  # 39+20
        assert m.critic[0].out_features == 768

    _run("Default ActorCritic (backward compat)", _check_model_default)
    _run("Custom [512,256,128]/[768,256,128]", _check_model_custom)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Task Class Import & Method Signatures (CPU-only)
# ══════════════════════════════════════════════════════════════════════════════
def test_task_class():
    _header("4. Task Class Methods")

    def _check_methods():
        """Use AST to verify methods exist without importing (avoids isaacgym)."""
        import ast
        with open("envs/T1/dribble_master.py") as f:
            tree = ast.parse(f.read())
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DribbleMaster":
                class_def = node
                break
        assert class_def is not None, "DribbleMaster class not found in AST"
        methods = {n.name for n in class_def.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        required = [
            "reset", "step", "_create_envs", "_init_buffers",
            "_compute_observations", "_compute_reward", "_check_termination",
            "_update_virtual_camera", "_resample_ball_vel_commands",
            "_reset_ball", "_reset_idx", "_prepare_reward_function",
        ]
        for m in required:
            assert m in methods, f"Missing method: {m}"

    def _check_reward_methods():
        """Use AST to verify reward methods."""
        import ast
        with open("envs/T1/dribble_master.py") as f:
            tree = ast.parse(f.read())
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DribbleMaster":
                class_def = node
                break
        methods = {n.name for n in class_def.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        expected_rewards = [
            "survival", "base_height", "orientation", "torques", "dof_vel",
            "dof_acc", "action_rate", "collision", "dof_pos_limits",
            "lin_vel_z", "ang_vel_xy", "feet_slip", "power", "feet_roll",
            "gait_symmetry", "ball_distance", "ball_velocity_tracking",
            "ball_velocity_direction", "ball_in_fov", "head_action_rate",
            "root_acc", "torque_tiredness", "feet_yaw_diff", "feet_yaw_mean",
        ]
        for r in expected_rewards:
            method = f"_reward_{r}"
            assert method in methods, f"Missing reward: {method}"

    _run("Required methods present", _check_methods)
    _run("All reward methods exist", _check_reward_methods)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Task Registration (CPU-only, imports envs)
# ══════════════════════════════════════════════════════════════════════════════
def test_registration():
    _header("5. Task Registration")

    def _check_init_file():
        """Verify envs/__init__.py has the right imports (text-based, no isaacgym)."""
        with open("envs/__init__.py") as f:
            content = f.read()
        assert "from envs.T1.dribble_master import DribbleMaster" in content, \
            "Missing DribbleMaster import"
        assert "DribbleMasterStage1" in content, "Missing DribbleMasterStage1 alias"
        assert "DribbleMasterStage2" in content, "Missing DribbleMasterStage2 alias"

    def _check_dynamic_loading_logic():
        """Verify the get_task_class name conversion would find our class."""
        # Simulate the camelCase conversion from runner.py
        import re
        for task_name in ["Dribble_Master_Stage1", "Dribble_Master_Stage2", "DribbleMaster"]:
            possible_names = [task_name]
            if '_' in task_name:
                camel_case = ''.join(word.capitalize() for word in task_name.split('_'))
                possible_names.append(camel_case)
            # Check that at least one possible name matches what's in __init__.py
            with open("envs/__init__.py") as f:
                content = f.read()
            found = any(f" {name}" in content or f"as {name}" in content for name in possible_names)
            assert found, f"Task '{task_name}' (possible: {possible_names}) not resolvable from __init__.py"

    _run("__init__.py imports", _check_init_file)
    _run("Dynamic loading name resolution", _check_dynamic_loading_logic)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Virtual Camera Logic (CPU-only, unit test)
# ══════════════════════════════════════════════════════════════════════════════
def test_virtual_camera():
    _header("6. Virtual Camera Logic (CPU)")
    from isaacgym.torch_utils import quat_rotate_inverse

    def _fov_check():
        """Unit test: ball in front of camera → in FOV; ball behind → not."""
        hfov = math.radians(87.0)
        vfov = math.radians(58.0)

        # Simulate head at origin, looking along +X
        head_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # identity

        # Ball directly in front at (2, 0, 0) → should be in FOV
        ball_front = torch.tensor([[2.0, 0.0, 0.0]])
        rel = quat_rotate_inverse(head_quat, ball_front)
        h_ang = abs(math.atan2(rel[0, 1].item(), rel[0, 0].item()))
        v_ang = abs(math.atan2(rel[0, 2].item(), max(abs(rel[0, 0].item()), 1e-6)))
        assert rel[0, 0] > 0 and h_ang < hfov / 2 and v_ang < vfov / 2, "Front ball should be in FOV"

        # Ball behind at (-1, 0, 0) → should NOT be in FOV
        ball_behind = torch.tensor([[-1.0, 0.0, 0.0]])
        rel_b = quat_rotate_inverse(head_quat, ball_behind)
        assert rel_b[0, 0] <= 0, "Behind ball should not be in FOV"

        # Ball at wide angle (0.1, 2.0, 0) → should NOT be in FOV (atan2 > 87/2 deg)
        ball_side = torch.tensor([[0.1, 2.0, 0.0]])
        rel_s = quat_rotate_inverse(head_quat, ball_side)
        h_ang_s = abs(math.atan2(rel_s[0, 1].item(), rel_s[0, 0].item()))
        assert h_ang_s > hfov / 2, "Side ball should be outside FOV"

    _run("FOV geometry checks", _fov_check)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Full Sim Smoke Test (GPU required – creates 2 envs, runs 10 steps)
# ══════════════════════════════════════════════════════════════════════════════
def test_sim_smoke():
    _header("7. Full Simulation Smoke Test (GPU)")

    def _smoke():
        import yaml

        with open("envs/T1/Dribble_Master_Stage1.yaml") as f:
            cfg = yaml.safe_load(f)

        # Minimal config for fast test
        cfg["env"]["num_envs"] = 2
        cfg["basic"]["headless"] = True
        cfg["basic"]["sim_device"] = "cuda:0"
        cfg["basic"]["rl_device"] = "cuda:0"

        from envs.T1.dribble_master import DribbleMaster
        env = DribbleMaster(cfg)

        # Check buffer shapes
        assert env.obs_buf.shape == (2, 39), f"obs_buf shape: {env.obs_buf.shape}"
        assert env.privileged_obs_buf.shape == (2, 20), f"priv_obs shape: {env.privileged_obs_buf.shape}"
        assert env.actions.shape == (2, 14), f"actions shape: {env.actions.shape}"
        assert env.ball_vel_commands.shape == (2, 2), f"ball_vel_commands shape: {env.ball_vel_commands.shape}"
        assert env.num_dofs == 14, f"num_dofs: {env.num_dofs}"

        # Reset
        obs, extras = env.reset()
        assert obs.shape == (2, 39), f"reset obs shape: {obs.shape}"
        assert "privileged_obs" in extras, "Missing privileged_obs in extras"
        assert extras["privileged_obs"].shape == (2, 20)

        # Run 10 steps with random actions
        for i in range(10):
            actions = torch.randn(2, 14, device="cuda:0") * 0.1
            obs, rew, done, extras = env.step(actions)
            assert obs.shape == (2, 39), f"Step {i}: obs shape {obs.shape}"
            assert rew.shape == (2,), f"Step {i}: rew shape {rew.shape}"
            assert done.shape == (2,), f"Step {i}: done shape {done.shape}"
            assert not torch.isnan(obs).any(), f"Step {i}: NaN in observations"
            assert not torch.isnan(rew).any(), f"Step {i}: NaN in rewards"

        # Check reward terms were computed
        assert len(extras["rew_terms"]) > 0, "No reward terms computed"
        print(f"    Reward terms: {list(extras['rew_terms'].keys())}")

        # Check virtual camera worked
        assert env.ball_in_fov.shape == (2,)
        assert env.observed_ball_pos.shape == (2, 3)

        # Store env for reuse in training test
        test_sim_smoke._env = env
        test_sim_smoke._cfg = cfg

    _run("Sim create + reset + 10 steps", _smoke)

    # --- Training loop test (reuses the SAME env from smoke test) ---
    def _train_smoke():
        from utils.model import ActorCritic

        env = test_sim_smoke._env
        cfg = test_sim_smoke._cfg
        device = cfg["basic"]["rl_device"]

        actor_hidden_dims = cfg.get("model", {}).get("actor_hidden_dims", None)
        critic_hidden_dims = cfg.get("model", {}).get("critic_hidden_dims", None)
        model = ActorCritic(
            env.num_actions, env.num_obs, env.num_privileged_obs,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        obs, extras = env.reset()
        privileged_obs = extras["privileged_obs"]

        for iteration in range(3):
            dist = model.act(obs)
            actions = dist.sample()
            values = model.est_value(obs, privileged_obs)

            obs, rew, done, extras = env.step(actions.detach())
            privileged_obs = extras["privileged_obs"]

            loss = -dist.log_prob(actions).mean() - 0.5 * values.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(torch.tensor(loss.item())), f"Iter {iteration}: NaN loss"

        print(f"      3 PPO iters OK, loss: {loss.item():.4f}")

    _run("Training loop (3 iters)", _train_smoke)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Tests 1-5: CPU-only (no Isaac Gym needed)
    # Tests 6-7: Need Isaac Gym + GPU
    gpu_available = _HAS_ISAACGYM and torch.cuda.is_available()

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       DribbleMaster Validation Tests                       ║")
    print(f"║       GPU/Isaac Gym: {'YES' if gpu_available else 'NO (tests 6-7 skipped)':42s}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # CPU tests (always run)
    test_yaml_configs()
    test_urdf()
    test_model()
    test_task_class()
    test_registration()

    # GPU tests (only with Isaac Gym)
    if gpu_available:
        test_virtual_camera()
        test_sim_smoke()
    else:
        _header("6-7. GPU Tests SKIPPED (no Isaac Gym / GPU)")
        print("  Run inside Docker with --gpus all to execute these tests.")

    print(f"\n{'='*60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'='*60}")
    sys.exit(1 if _failed > 0 else 0)
