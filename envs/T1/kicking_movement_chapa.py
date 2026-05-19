"""Chapa (instep) kick task.

Variant of `KickingMovementBica` that biases the policy toward kicking with the
flat top of the foot ("dorso/chapa") instead of the toe ("biquinha"). The
robot model has a single foot link, so we cannot differentiate toe vs instep
contact geometrically; instead we shape the reward so that, at the moment of
impact, the ball is centred over the foot in the foot's local X axis and the
foot is roughly horizontal (pitch ~ 0).

Three reward terms (all configurable in `Kicking_Movement_Chapa.yaml`):

- ``chapa_foot_pitch_alignment`` — continuous during approach (ball stationary
  and closest foot near the ball). Gaussian on `feet_pitch` of the closest
  foot. Encourages the foot to be flat in the approach pose.
- ``chapa_contact_centering`` — continuous during approach. Gaussian on the
  ball position in the closest foot's local frame: prefers ``x_local`` near
  the centre of the foot box (≈ 0.01 m offset from the foot link origin).
- ``chapa_impact_pose`` — single-step pulse on the step where
  ``kick_just_detected`` flips True. Combines pitch-at-impact and
  ``x_local``-at-impact gaussians. Reads buffers populated by the parent class.

The approach reward inherited from the parent is also rescaled by a pose
factor so that approaching with a toe-down pose is worth less than the same
approach in a flat-foot pose.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from envs.T1.kicking_movement_bica import KickingMovementBica


class KickingMovementChapa(KickingMovementBica):
    """Instep-kick variant of `KickingMovementBica`."""

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _chapa_closest_foot_idx(self) -> torch.Tensor:
        """Per-env index (0=left, 1=right) of the foot currently closest to the ball."""
        d_left = torch.norm(self.feet_pos[:, 0, :] - self.ball_pos, dim=-1)
        d_right = torch.norm(self.feet_pos[:, 1, :] - self.ball_pos, dim=-1)
        return (d_right < d_left).long()

    def _chapa_ball_in_foot_frame(self, foot_idx: torch.Tensor) -> torch.Tensor:
        """Ball position in the local frame of the selected foot (shape [N, 3])."""
        rows = torch.arange(self.num_envs, device=self.device)
        foot_pos = self.feet_pos[rows, foot_idx]
        foot_quat = self.feet_quat[rows, foot_idx]
        return quat_rotate_inverse(foot_quat, self.ball_pos - foot_pos)

    def _chapa_near_ball_mask(self, foot_idx: torch.Tensor, radius: float) -> torch.Tensor:
        """True for envs where the chosen foot is within `radius` of the ball AND the ball is stationary."""
        rows = torch.arange(self.num_envs, device=self.device)
        foot_pos = self.feet_pos[rows, foot_idx]
        dist = torch.norm(foot_pos - self.ball_pos, dim=-1)
        ball_speed_thresh = self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)
        ball_still = torch.norm(self.ball_lin_vel, dim=-1) < ball_speed_thresh
        return (dist < radius) & ball_still

    # ----------------------------------------------------------------------
    # New rewards specific to Chapa
    # ----------------------------------------------------------------------
    def _reward_chapa_foot_pitch_alignment(self):
        """Continuous: gaussian on closest-foot pitch while approaching a stationary ball."""
        cfg = self.cfg["rewards"]
        radius = cfg.get("chapa_pose_radius", 0.25)
        sigma = cfg.get("chapa_pitch_sigma", 0.15)
        target = cfg.get("chapa_pose_target_pitch", 0.0)

        foot_idx = self._chapa_closest_foot_idx()
        rows = torch.arange(self.num_envs, device=self.device)
        pitch = self.feet_pitch[rows, foot_idx]
        gate = self._chapa_near_ball_mask(foot_idx, radius).float()

        reward = torch.exp(-torch.square(pitch - target) / (sigma * sigma + 1e-8))
        return reward * gate

    def _reward_chapa_contact_centering(self):
        """Continuous: gaussian on ball X position in the closest foot's frame, near ball, ball still."""
        cfg = self.cfg["rewards"]
        radius = cfg.get("chapa_pose_radius", 0.25)
        sigma = cfg.get("chapa_center_sigma", 0.04)
        # Foot collision box is centred at x_local = +0.01 m (URDF offset).
        target_x = cfg.get("chapa_center_target_x", 0.01)
        # Soft penalty: when |x_local| crosses the toe threshold, decay quickly.
        toe_threshold = cfg.get("chapa_toe_threshold", 0.08)

        foot_idx = self._chapa_closest_foot_idx()
        ball_local = self._chapa_ball_in_foot_frame(foot_idx)
        x_local = ball_local[:, 0]

        centred = torch.exp(-torch.square(x_local - target_x) / (sigma * sigma + 1e-8))
        toe_excess = torch.clamp(torch.abs(x_local) - toe_threshold, min=0.0)
        toe_penalty = torch.exp(-toe_excess / 0.02)  # decays fast once past the toe

        gate = self._chapa_near_ball_mask(foot_idx, radius).float()
        return centred * toe_penalty * gate

    def _reward_chapa_impact_pose(self):
        """Pulse: fires on the single step `kick_just_detected` is True.

        Combines two gaussians using the pose snapshot taken by the parent at
        impact: foot pitch and ball X position in the kicking foot's frame.
        Result is divided by dt so the YAML scale is in "per-event units"
        (the framework multiplies every scale by dt internally).
        """
        cfg = self.cfg["rewards"]
        pitch_sigma = cfg.get("chapa_impact_pitch_sigma", 0.10)
        center_sigma = cfg.get("chapa_impact_center_sigma", 0.03)
        target_pitch = cfg.get("chapa_pose_target_pitch", 0.0)
        target_x = cfg.get("chapa_center_target_x", 0.01)

        pitch_score = torch.exp(
            -torch.square(self.kick_foot_pitch_at_impact - target_pitch)
            / (pitch_sigma * pitch_sigma + 1e-8)
        )
        x_score = torch.exp(
            -torch.square(self.kick_ball_local_at_impact[:, 0] - target_x)
            / (center_sigma * center_sigma + 1e-8)
        )

        pulse = self.kick_just_detected.float()
        return pitch_score * x_score * pulse / self.dt

    # ----------------------------------------------------------------------
    # Override the dominant post-kick reward to gate it by the pose-at-impact.
    # This is the key signal that pushes the policy from toe-kick to chapa:
    # the *ball velocity toward target* reward — which dominates the budget
    # for ~2 s after impact — is multiplied by a gaussian on the pitch the
    # foot had at the impact step. With weight=1.0 the policy can only
    # collect that reward by having approached with the foot horizontal.
    # Gated by `chapa_velocity_pose_weight` (default 0.0 = identical to the
    # ungated toe-kick baseline), so this override is fully opt-in.
    # ----------------------------------------------------------------------
    def _reward_ball_velocity_target_direction(self):
        base = super()._reward_ball_velocity_target_direction()

        cfg = self.cfg["rewards"]
        weight = cfg.get("chapa_velocity_pose_weight", 0.0)
        if weight <= 0.0:
            return base

        sigma = cfg.get("chapa_impact_pitch_sigma", 0.25)
        target_pitch = cfg.get("chapa_pose_target_pitch", 0.0)
        pose_factor = torch.exp(
            -torch.square(self.kick_foot_pitch_at_impact - target_pitch)
            / (sigma * sigma + 1e-8)
        )

        w = max(0.0, min(1.0, weight))
        return base * ((1.0 - w) + w * pose_factor)

    # ----------------------------------------------------------------------
    # Override approach reward to weight by pose factor
    # ----------------------------------------------------------------------
    def _reward_kicking_foot_approach_ball_stationary(self):
        """Inherited approach reward, multiplied by a pose factor that favours flat-foot approach."""
        base = super()._reward_kicking_foot_approach_ball_stationary()

        cfg = self.cfg["rewards"]
        pose_weight = cfg.get("chapa_approach_pose_weight", 0.0)
        if pose_weight <= 0.0:
            return base

        sigma = cfg.get("chapa_pitch_sigma", 0.15)
        target_pitch = cfg.get("chapa_pose_target_pitch", 0.0)
        foot_idx = self._chapa_closest_foot_idx()
        rows = torch.arange(self.num_envs, device=self.device)
        pitch = self.feet_pitch[rows, foot_idx]
        pose_factor = torch.exp(-torch.square(pitch - target_pitch) / (sigma * sigma + 1e-8))

        # Blend: pose_weight=0 -> identical to parent; pose_weight=1 -> fully pose-gated.
        weight = max(0.0, min(1.0, pose_weight))
        return base * ((1.0 - weight) + weight * pose_factor)
