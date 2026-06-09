"""Policy inference for the Base Walk Extended policy on the real Booster T1 robot.

Observation layout — 53 dimensions (must match training exactly):
    0: 3   projected gravity (robot frame)
    3: 6   base angular velocity (robot frame)
    6: 9   walk commands [vx, vy, wz]
    9:11   gait cos/sin
   11:17   zeros — ball/target slots (unused during base walk)
   17:29   dof_pos − default_dof_pos
   29:41   dof_vel
   41:53   last actions
"""

import numpy as np
import torch


class PolicyBaseWalkExtended:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        norm = self.cfg["policy"]["normalization"]

        # 0:3  gravity
        self.obs[0:3] = projected_gravity * norm["gravity"]
        # 3:6  angular velocity
        self.obs[3:6] = base_ang_vel * norm["ang_vel"]
        # 6:9  walk commands
        self.obs[6] = self.smoothed_commands[0] * norm["lin_vel"] * (self.gait_frequency > 1.0e-8)
        self.obs[7] = self.smoothed_commands[1] * norm["lin_vel"] * (self.gait_frequency > 1.0e-8)
        self.obs[8] = self.smoothed_commands[2] * norm["ang_vel"] * (self.gait_frequency > 1.0e-8)
        # 9:11  gait cos/sin
        self.obs[9]  = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        # 11:17  ball/target slots — zeros (already zero from init)
        # 17:29  dof_pos (legs only, indices 11:23)
        self.obs[17:29] = (dof_pos[11:] - self.default_dof_pos[11:]) * norm["dof_pos"]
        # 29:41  dof_vel (legs only)
        self.obs[29:41] = dof_vel[11:] * norm["dof_vel"]
        # 41:53  last actions
        self.obs[41:53] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(self.actions, -norm["clip_actions"], norm["clip_actions"])

        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
