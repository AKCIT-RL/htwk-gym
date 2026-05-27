"""Policy inference for the Kicking Bikinha (toe-kick) policy on the real robot.

Observation layout — 52 dimensions (must match training exactly):
    0: 3   projected gravity (robot frame)
    3: 6   base angular velocity (robot frame)
    6: 9   walk commands [vx, vy, wz] — auto-computed toward ball, zero when near
    9:11   gait cos/sin — always zero for kicking
   11:13   ball position XY in robot frame
   13:15   kick target direction [cos θ, sin θ] in robot frame
   15:16   target z relative to ball z
   16:28   dof_pos − default_dof_pos
   28:40   dof_vel
   40:52   last actions

The ball position and kick target direction must be provided externally
(e.g. from a vision system or hardcoded for testing).
"""

import numpy as np
import torch


class PolicyKickingBikinha:
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

        # Approach command parameters (match training)
        self.approach_stop_dist = self.cfg["policy"].get("approach_stop_dist", 0.5)
        self.approach_max_speed = self.cfg["policy"].get("approach_max_speed", 1.0)
        self.approach_yaw_gain = self.cfg["policy"].get("approach_yaw_gain", 0.3)

    def _compute_approach_commands(self, ball_x, ball_y):
        """Mimic _update_commands_toward_ball from training env.

        Given ball position in robot-local XY frame, compute walk commands
        [vx, vy, wz] that steer the robot toward the ball. Commands go to
        zero when the robot is within `approach_stop_dist` of the ball.
        """
        dist = np.sqrt(ball_x ** 2 + ball_y ** 2 + 1e-12)
        speed = min(dist * (self.approach_max_speed / self.approach_stop_dist),
                    self.approach_max_speed)

        if dist < self.approach_stop_dist:
            return 0.0, 0.0, 0.0

        cmd_vx = (ball_x / dist) * speed
        cmd_vy = (ball_y / dist) * speed
        cmd_wz = np.arctan2(ball_y, ball_x) * self.approach_yaw_gain

        return cmd_vx, cmd_vy, cmd_wz

    def _compute_target_dir_robot(self, ball_x, ball_y, target_angle_rad):
        """Compute [cos θ, sin θ] of kick target direction in robot frame.

        On the real robot the ball→target direction in the world is given
        by `target_angle_rad` (0 = straight ahead in world). Since ball_x/y
        are already in the robot frame and the robot's yaw is implicitly
        identity for those, the direction is simply:
            cos(target_angle_rad), sin(target_angle_rad)
        rotated into the robot frame. For simplicity when the caller already
        expresses the target in the robot frame, `target_angle_rad` is
        the angle FROM the ball TO the target in the robot frame.
        """
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
        """Run one policy step.

        Args:
            dof_pos: joint positions (23,) — only indices 11: used.
            dof_vel: joint velocities (23,) — only indices 11: used.
            base_ang_vel: IMU gyro (3,).
            projected_gravity: gravity in robot frame (3,).
            ball_robot_xy: ball [x, y] in robot-local frame (metres).
            target_angle_rad: kick direction in robot frame (rad, 0 = forward).
            target_z_rel: target z minus ball z (metres, usually ~0).

        Returns:
            dof_targets (23,) — full joint target array for the PD controller.
        """
        norm = self.cfg["policy"]["normalization"]
        ball_x, ball_y = float(ball_robot_xy[0]), float(ball_robot_xy[1])

        # 0:3  gravity
        self.obs[0:3] = projected_gravity * norm["gravity"]

        # 3:6  angular velocity
        self.obs[3:6] = base_ang_vel * norm["ang_vel"]

        # 6:9  approach walk commands (auto-computed)
        cmd_vx, cmd_vy, cmd_wz = self._compute_approach_commands(ball_x, ball_y)
        self.obs[6] = cmd_vx * norm["lin_vel"]
        self.obs[7] = cmd_vy * norm["lin_vel"]
        self.obs[8] = cmd_wz * norm["ang_vel"]

        # 9:11  gait cos/sin — always zero for kicking
        self.obs[9] = 0.0
        self.obs[10] = 0.0

        # 11:13  ball XY in robot frame
        self.obs[11] = ball_x * norm["ball_pos"]
        self.obs[12] = ball_y * norm["ball_pos"]

        # 13:15  target direction [cos θ, sin θ] in robot frame
        tcos, tsin = self._compute_target_dir_robot(ball_x, ball_y, target_angle_rad)
        self.obs[13] = tcos * norm["target_dir"]
        self.obs[14] = tsin * norm["target_dir"]

        # 15:16  target z relative to ball z
        self.obs[15] = target_z_rel * norm["target_z"]

        # 16:28  dof_pos − default (legs only, indices 11:23)
        self.obs[16:28] = (dof_pos[11:] - self.default_dof_pos[11:]) * norm["dof_pos"]

        # 28:40  dof_vel (legs only)
        self.obs[28:40] = dof_vel[11:] * norm["dof_vel"]

        # 40:52  last actions
        self.obs[40:52] = self.actions

        # Inference
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.actions[:] = self.policy(obs_tensor).detach().numpy()
        self.actions[:] = np.clip(self.actions, -norm["clip_actions"], norm["clip_actions"])

        # Build full joint targets
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
