"""DribbleMaster – Two-stage curriculum RL for humanoid soccer dribbling.

Implements the method from:
  "Dribble Master: Learning Agile Humanoid Dribbling through Legged Locomotion"
  (Wang, Zhou, Wu – arXiv:2505.12679v3, ICRA 2026)

Key features vs existing tasks:
  - 14-DOF action space (12 legs + 2 head joints for active sensing)
  - Ball velocity commands (vx, vy in global frame) instead of robot velocity
  - Virtual camera model simulating RealSense D455 FOV
  - Active sensing: robot learns to move head to track ball
  - Self-contained two-stage curriculum (Stage 1 & 2 differ only by YAML config)
"""

import os
import math

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
    quat_mul,
)

assert gymtorch

import torch
import numpy as np
from envs.base_task import BaseTask
from utils.utils import apply_randomization


class DribbleMaster(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------
    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        # Load ball asset
        ball_cfg = self.cfg["ball"]
        ball_root = os.path.dirname(ball_cfg["file"])
        ball_file = os.path.basename(ball_cfg["file"])

        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = ball_cfg["density"]
        ball_asset_options.angular_damping = 0.0
        ball_asset_options.linear_damping = 0.0
        ball_asset_options.max_angular_velocity = 1000.0
        ball_asset_options.max_linear_velocity = 1000.0
        ball_asset_options.disable_gravity = False
        ball_asset_options.replace_cylinder_with_capsule = False
        ball_asset_options.thickness = ball_cfg["thickness"]

        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, ball_asset_options)

        self.ball_radius = 0.05
        self.ball_init_pos = to_torch(ball_cfg["init_pos"], device=self.device)
        self.ball_init_rot = to_torch(ball_cfg["init_rot"], device=self.device)
        self.ball_init_lin_vel = to_torch(ball_cfg["init_lin_vel"], device=self.device)
        self.ball_init_ang_vel = to_torch(ball_cfg["init_ang_vel"], device=self.device)

        # Robot DOF info
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        # Head body index for virtual camera
        head_link_name = asset_cfg.get("head_link", "H2")
        self.head_body_index = self.gym.find_asset_rigid_body_index(robot_asset, head_link_name)

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"]
            + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(-5, 0.0, -5)
        env_upper = gymapi.Vec3(5, 5, 5)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            # Ball actor
            ball_pose = gymapi.Transform()
            ball_pose.p = gymapi.Vec3(*self.env_origins[i])
            ball_pose.p += gymapi.Vec3(self.ball_init_pos[0], self.ball_init_pos[1], self.ball_init_pos[2])
            ball_pose.r = gymapi.Quat(self.ball_init_rot[0], self.ball_init_rot[1], self.ball_init_rot[2], self.ball_init_rot[3])

            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, ball_cfg["name"], i)
            ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
            ball_shape_props[0].restitution = ball_cfg["restitution"]
            ball_shape_props[0].friction = ball_cfg["friction"]
            ball_shape_props[0].rolling_friction = ball_cfg["rolling_friction"]
            self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.ball_handles.append(ball_handle)

        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ball_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True)
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------
    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # Gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = root_states.view(self.num_envs, 2, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies + 1, 13)

        self.base_pos = self.root_states[:, 0, 0:3]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.ball_pos = self.root_states[:, 1, 0:3]
        self.ball_rot = self.root_states[:, 1, 3:7]
        self.ball_lin_vel = self.body_states[:, -1, 7:10]
        self.ball_ang_vel = self.body_states[:, -1, 10:13]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        # Ball velocity commands (vx, vy in global frame) — the paper's "commands"
        self.ball_vel_commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Gait clock
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()

        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        # --- Virtual Camera Model ---
        cam_cfg = self.cfg.get("virtual_camera", {})
        self.cam_hfov = math.radians(cam_cfg.get("hfov_deg", 87.0))
        self.cam_vfov = math.radians(cam_cfg.get("vfov_deg", 58.0))
        self.cam_half_hfov = self.cam_hfov / 2.0
        self.cam_half_vfov = self.cam_vfov / 2.0
        self.ball_obs_history_s = cam_cfg.get("history_s", 0.3)
        self.ball_obs_history_steps = max(1, int(self.ball_obs_history_s / self.dt))

        # Observed ball position (may be stale if out of FOV)
        self.observed_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_in_fov = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.steps_since_ball_seen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.last_ball_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Compatibility with Runner (curriculum stats expected by recorder)
        self.curriculum_prob = torch.ones(1, dtype=torch.float, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0

    def _prepare_reward_function(self):
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

        # Reset virtual camera state
        self.observed_ball_pos[env_ids] = self.ball_pos[env_ids]
        self.ball_in_fov[env_ids] = 1.0
        self.steps_since_ball_seen[env_ids] = 0
        self.last_ball_lin_vel_world[env_ids] = 0.0

        # Resample ball velocity commands
        self._resample_ball_vel_commands(env_ids)

        # Reset gait
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0],
            self.cfg["commands"]["gait_frequency"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.gait_process[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = (2 * env_ids).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 0, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, 0, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("init_base_ang"),
            ),
        )
        self.root_states[env_ids, 0, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )

        # Reset ball at configurable distance from robot
        self._reset_ball(env_ids)

        robot_actor_indices = 2 * env_ids
        ball_actor_indices = 2 * env_ids + 1
        actor_indices = torch.stack((robot_actor_indices, ball_actor_indices), dim=-1).view(-1).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(actor_indices), actor_indices.shape[0]
        )

    def _reset_ball(self, env_ids):
        """Reset ball at a distance from the robot, configurable per stage."""
        robot_pos = self.root_states[env_ids, 0, 0:3]
        robot_quat = self.root_states[env_ids, 0, 3:7]

        dist_range = self.cfg["ball"].get("init_distance_range", [0.3, 1.5])
        dist = torch_rand_float(dist_range[0], dist_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        angle = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(1)

        # Ball position relative to robot in world frame
        dx = dist * torch.cos(angle)
        dy = dist * torch.sin(angle)

        self.root_states[env_ids, 1, 0] = robot_pos[:, 0] + dx
        self.root_states[env_ids, 1, 1] = robot_pos[:, 1] + dy
        self.root_states[env_ids, 1, 2] = self.ball_radius

        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        self.root_states[env_ids, 1, 3:7] = identity_quat
        self.root_states[env_ids, 1, 7:13] = 0.0

    def _resample_ball_vel_commands(self, env_ids):
        """Resample target ball velocity commands (vx, vy in global frame)."""
        cmd_cfg = self.cfg["commands"]
        vel_range = cmd_cfg.get("ball_vel_range", [-1.5, 1.5])
        self.ball_vel_commands[env_ids, 0] = torch_rand_float(
            vel_range[0], vel_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ball_vel_commands[env_ids, 1] = torch_rand_float(
            vel_range[0], vel_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        resampling_time = cmd_cfg.get("resampling_time_s", [3.0, 5.0])
        self.cmd_resample_time[env_ids] += torch.randint(
            int(resampling_time[0] / self.dt), int(resampling_time[1] / self.dt),
            (len(env_ids),), device=self.device,
        )

    # ------------------------------------------------------------------
    # Virtual Camera Model
    # ------------------------------------------------------------------
    def _update_virtual_camera(self):
        """Determine if ball is in the head camera's FOV and update observed ball position."""
        # Head body state: position + quaternion
        head_pos = self.body_states[:, self.head_body_index, 0:3]
        head_quat = self.body_states[:, self.head_body_index, 3:7]

        # Ball position relative to head in head-local frame
        ball_to_head_world = self.ball_pos - head_pos
        ball_in_head_frame = quat_rotate_inverse(head_quat, ball_to_head_world)

        # Camera looks along +X in head frame, Y is left, Z is up
        # Compute horizontal angle (atan2(y, x)) and vertical angle (atan2(z, x))
        dist_xy = torch.sqrt(ball_in_head_frame[:, 0] ** 2 + ball_in_head_frame[:, 1] ** 2).clamp(min=1e-6)
        h_angle = torch.abs(torch.atan2(ball_in_head_frame[:, 1], ball_in_head_frame[:, 0]))
        v_angle = torch.abs(torch.atan2(ball_in_head_frame[:, 2], dist_xy))

        # Ball must be in front of camera (x > 0) and within FOV
        in_fov = (ball_in_head_frame[:, 0] > 0) & (h_angle < self.cam_half_hfov) & (v_angle < self.cam_half_vfov)

        self.ball_in_fov[:] = in_fov.float()

        # Update observed ball position only when in FOV
        visible_mask = in_fov
        self.observed_ball_pos[visible_mask] = self.ball_pos[visible_mask]
        # Add perception noise when visible
        self.observed_ball_pos[visible_mask] = apply_randomization(
            self.observed_ball_pos[visible_mask], self.cfg["noise"].get("ball_pos"))

        # Track steps since last seen
        self.steps_since_ball_seen[visible_mask] = 0
        self.steps_since_ball_seen[~visible_mask] += 1

        # After history timeout, freeze to last known (already frozen by not updating)
        # The policy sees stale ball_pos + ball_in_fov=0 indicator

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions):
        # Pre physics
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # Physics step with decimation
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # Post physics
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, -1, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, -1, 10:13]
        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = (
            self.base_lin_vel * self.cfg["normalization"]["filter_weight"]
            + self.filtered_lin_vel * (1.0 - self.cfg["normalization"]["filter_weight"])
        )
        self.filtered_ang_vel[:] = (
            self.base_ang_vel * self.cfg["normalization"]["filter_weight"]
            + self.filtered_ang_vel * (1.0 - self.cfg["normalization"]["filter_weight"])
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        # Update virtual camera
        self._update_virtual_camera()

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        # Store ball vel before reset
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]

        # Reset terminated envs
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self.last_ball_lin_vel_world[env_ids] = 0.0

        # Resample ball velocity commands periodically
        resample_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_ball_vel_commands(resample_ids)

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 0, 10:13] = apply_randomization(self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]), self.cfg["randomization"].get("push_force"))
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]), self.cfg["randomization"].get("push_torque"))
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques), gymapi.LOCAL_SPACE)

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0).unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]),
            dim=2,
        )

    def _check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        self.reset_buf |= self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _compute_observations(self):
        """Observation layout (39 dims):
          0: 2  ball velocity commands (vx, vy) in global frame
          2: 5  body orientation (yaw, roll, pitch)
          5: 7  clock signal (sin, cos)
         7:10  observed ball position relative to robot (xyz) — frozen when out of FOV
           10  ball-in-FOV indicator
        11:25  joint positions (14 DOF)
        25:39  joint velocities (14 DOF)
        """
        _, base_roll, base_yaw = get_euler_xyz(self.base_quat)
        base_pitch = torch.asin((-self.projected_gravity[:, 0]).clamp(-1, 1))

        # Ball position relative to robot in robot frame (use observed, not ground truth)
        ball_rel_world = self.observed_ball_pos - self.base_pos
        ball_rel_robot = quat_rotate_inverse(self.base_quat, ball_rel_world)

        cmd_scale = self.cfg["normalization"].get("ball_vel_cmd", 1.0)

        self.obs_buf = torch.cat((
            # 0:2 ball velocity commands
            self.ball_vel_commands * cmd_scale,
            # 2:5 body orientation
            apply_randomization(
                torch.stack([base_yaw, base_roll, base_pitch], dim=-1),
                self.cfg["noise"].get("orientation")
            ),
            # 5:7 clock signal
            (torch.sin(2 * math.pi * self.gait_process) * (self.gait_frequency > 1e-8).float()).unsqueeze(-1),
            (torch.cos(2 * math.pi * self.gait_process) * (self.gait_frequency > 1e-8).float()).unsqueeze(-1),
            # 7:10 observed ball position (robot frame)
            apply_randomization(ball_rel_robot, self.cfg["noise"].get("ball_pos")) * self.cfg["normalization"].get("ball_pos", 1.0),
            # 10 ball-in-FOV indicator
            self.ball_in_fov.unsqueeze(-1),
            # 11:25 joint positions
            apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
            # 25:39 joint velocities
            apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            # Base mass scaling (4)
            self.base_mass_scaled,
            # Base linear velocity (3)
            apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
            # Base height (1)
            apply_randomization(
                self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos),
                self.cfg["noise"].get("height")
            ).unsqueeze(-1),
            # Ground truth ball velocity (3)
            self.ball_lin_vel,
            # Ground truth ball position relative to robot (3)
            ball_rel_robot,
            # Pushing forces/torques (6)
            self.pushing_forces[:, 0, :] * self.cfg["normalization"].get("push_force", 0.1),
            self.pushing_torques[:, 0, :] * self.cfg["normalization"].get("push_torque", 0.5),
        ), dim=-1)

        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _reward_survival(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_base_height(self):
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_dof_pos_limits(self):
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_lin_vel_z(self):
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_feet_slip(self):
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            ) * (self.episode_length_buf > 1).float()
        )

    def _reward_power(self):
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    # --- Gait / clock-based rewards ---
    def _reward_gait_symmetry(self):
        """Reward for matching clock-based reference for hip pitch joints.
        sin(phase) → left hip pitch, -sin(phase) → right hip pitch (paper Sec III.A)."""
        ref_amplitude = self.cfg["rewards"].get("gait_ref_amplitude", 0.2)
        phase = 2 * math.pi * self.gait_process
        left_ref = ref_amplitude * torch.sin(phase)
        right_ref = -ref_amplitude * torch.sin(phase)

        # Find hip pitch DOF indices (assuming naming convention contains "Hip_Pitch")
        left_hip_idx = None
        right_hip_idx = None
        for i, name in enumerate(self.dof_names):
            if "Left_Hip_Pitch" in name:
                left_hip_idx = i
            elif "Right_Hip_Pitch" in name:
                right_hip_idx = i

        if left_hip_idx is None or right_hip_idx is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        left_err = torch.square(self.dof_pos[:, left_hip_idx] - self.default_dof_pos[0, left_hip_idx] - left_ref)
        right_err = torch.square(self.dof_pos[:, right_hip_idx] - self.default_dof_pos[0, right_hip_idx] - right_ref)
        sigma = self.cfg["rewards"].get("gait_tracking_sigma", 0.1)
        return torch.exp(-(left_err + right_err) / sigma) * (self.gait_frequency > 1e-8).float()

    # --- Ball interaction rewards ---
    def _reward_ball_distance(self):
        """Penalty proportional to distance between robot and ball (encourages chasing)."""
        dist = torch.norm(self.ball_pos[:, :2] - self.base_pos[:, :2], dim=-1)
        return dist

    def _reward_ball_velocity_tracking(self):
        """Reward for ball velocity matching the commanded target (global frame)."""
        ball_vel_xy = self.root_states[:, 1, 7:9]  # ball linear vel x,y in world frame
        vel_error = torch.norm(ball_vel_xy - self.ball_vel_commands, dim=-1)
        sigma = self.cfg["rewards"].get("ball_vel_tracking_sigma", 0.5)
        return torch.exp(-vel_error ** 2 / sigma)

    def _reward_ball_velocity_direction(self):
        """Reward for ball velocity being aligned with commanded direction."""
        ball_vel_xy = self.root_states[:, 1, 7:9]
        cmd_norm = torch.norm(self.ball_vel_commands, dim=-1, keepdim=True).clamp(min=0.1)
        cmd_dir = self.ball_vel_commands / cmd_norm
        vel_proj = (ball_vel_xy * cmd_dir).sum(dim=-1)
        return vel_proj.clamp(min=0.0)

    # --- Active sensing rewards ---
    def _reward_ball_in_fov(self):
        """Reward for keeping the ball in the camera's field of view."""
        return self.ball_in_fov

    def _reward_head_action_rate(self):
        """Penalty for rapid head joint movements (smoothness)."""
        # Head DOFs are the first 2 in our 14-DOF action space (indices 0,1 assuming
        # head joints come first in URDF). Find actual indices.
        head_action_indices = []
        for i, name in enumerate(self.dof_names):
            if "Head" in name or "AAHead" in name:
                head_action_indices.append(i)
        if not head_action_indices:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        idx = torch.tensor(head_action_indices, device=self.device, dtype=torch.long)
        return torch.sum(torch.square(self.last_actions[:, idx] - self.actions[:, idx]), dim=-1)

    def _reward_root_acc(self):
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_torque_tiredness(self):
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)
