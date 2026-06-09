"""Policy inference for the Kicking policy on the real Booster T1 robot.

Observation layout — 48 dimensions (must match training exactly):
    0: 3   projected gravity (robot frame)
    3: 6   base angular velocity (robot frame)
    6: 8   ball position XY in robot frame
    8:10   kick target direction [cos θ, sin θ] in robot frame
   10:11   target z relative to ball z
   11:12   target distance (ball-to-target, metres)
   12:24   dof_pos − default_dof_pos
   24:36   dof_vel
   36:48   last actions

The ball position and kick target direction must be provided externally
(e.g. from a vision system or hardcoded for testing).
"""

import numpy as np
import torch


class PolicyKicking:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = torch.jit.load(cfg["policy"]["policy_path"])
        self.policy.eval()
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)
        self.dof_targets = np.copy(self.default_dof_pos)

        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def _compute_target_dir_robot(self, ball_x, ball_y, target_angle_rad):
        """Compute [cos θ, sin θ] of kick target direction in robot frame."""
        return np.cos(target_angle_rad), np.sin(target_angle_rad)

    def inference(
        self,
        dof_pos,
        dof_vel,
        base_ang_vel,
        projected_gravity,
        ball_robot_xy,
        target_angle_rad=0.0,
        target_z_rel=0.0,
    ):
        norm = self.cfg["policy"]["normalization"]
        ball_x, ball_y = float(ball_robot_xy[0]), float(ball_robot_xy[1])

        # 0:3  gravity
        self.obs[0:3] = projected_gravity * norm["gravity"]

        # 3:6  angular velocity
        self.obs[3:6] = base_ang_vel * norm["ang_vel"]

        # 6:8  ball XY in robot frame
        self.obs[6] = ball_x * norm["ball_pos"]
        self.obs[7] = ball_y * norm["ball_pos"]

        # 8:10  target direction [cos θ, sin θ] in robot frame
        tcos, tsin = self._compute_target_dir_robot(ball_x, ball_y, target_angle_rad)
        self.obs[8] = tcos * norm["target_dir"]
        self.obs[9] = tsin * norm["target_dir"]

        # 10:11  target z relative to ball z
        self.obs[10] = target_z_rel * norm["target_z"]

        # 11:12  target distance (ball to target, metres)
        self.obs[11] = self.cfg["policy"].get("kick_target_ref_distance", 8.0) * norm["target_distance"]

        # 12:24  dof_pos − default (legs only, indices 11:23)
        self.obs[12:24] = (dof_pos[11:] - self.default_dof_pos[11:]) * norm["dof_pos"]

        # 24:36  dof_vel (legs only)
        self.obs[24:36] = dof_vel[11:] * norm["dof_vel"]

        # 36:48  last actions
        self.obs[36:48] = self.actions

        # Inference
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.actions[:] = self.policy(obs_tensor).detach().numpy()
        self.actions[:] = np.clip(self.actions, -norm["clip_actions"], norm["clip_actions"])

        # Build full joint targets
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
