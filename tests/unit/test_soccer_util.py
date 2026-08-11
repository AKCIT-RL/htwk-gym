import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from envs import soccer_util


def _yaw_quat(yaw, n=1):
    """Quaternion (x, y, z, w) for a rotation of `yaw` around z."""
    half = 0.5 * yaw
    q = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])
    return q.repeat(n, 1)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

def test_ball_steer_command_direction_and_speed():
    root_pos = torch.tensor([[0.0, 0.0, 0.8],
                             [0.0, 0.0, 0.8],
                             [0.0, 0.0, 0.8]])
    ball_pos = torch.tensor([[3.0, 4.0, 0.11],    # dist 5 -> speed capped
                             [0.0, 0.9, 0.11],    # dist 0.9 -> ramp
                             [0.1, 0.0, 0.11]])   # dist 0.1 < stop -> zero
    cmd = soccer_util.compute_ball_steer_command(root_pos, ball_pos, 0.45, 1.5)

    assert cmd.shape == (3, 3)
    # unit direction toward the ball
    assert torch.allclose(cmd[0, 0:2], torch.tensor([0.6, 0.8]), atol=1e-5)
    assert torch.allclose(cmd[1, 0:2], torch.tensor([0.0, 1.0]), atol=1e-5)
    # speed = clamp(dist - stop_dist, 0, max)
    assert cmd[0, 2] == pytest.approx(1.5)
    assert cmd[1, 2] == pytest.approx(0.45, abs=1e-5)
    assert cmd[2, 2] == pytest.approx(0.0, abs=1e-6)


def test_ball_steer_command_degenerate_overlap():
    root_pos = torch.tensor([[1.0, 2.0, 0.8]])
    ball_pos = torch.tensor([[1.0, 2.0, 0.11]])
    cmd = soccer_util.compute_ball_steer_command(root_pos, ball_pos, 0.45, 1.5)
    assert torch.all(torch.isfinite(cmd))
    assert cmd[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_kick_direction_reward_projection_and_threshold():
    ball_pos = torch.tensor([[0.0, 0.0, 0.11]] * 4)
    goal_pos = torch.tensor([[5.0, 0.0]] * 4)  # goal along +x
    ball_vel = torch.tensor([[5.0, 0.0, 0.0],    # straight at goal, fast
                             [0.0, 5.0, 0.0],    # purely lateral
                             [-5.0, 0.0, 0.0],   # backward
                             [0.5, 0.0, 0.0]])   # toward goal but below min_vel
    t_moving = torch.zeros(4)
    r = soccer_util.compute_kick_direction_reward(ball_pos, ball_vel, goal_pos,
                                                  1.0, 0.2, t_moving, 30.0)
    assert r[0] == pytest.approx(4.0, abs=1e-5)   # 5 - 1, no decay at t=0
    assert r[1] == pytest.approx(0.0, abs=1e-6)
    assert r[2] == pytest.approx(0.0, abs=1e-6)
    assert r[3] == pytest.approx(0.0, abs=1e-6)


def test_kick_direction_reward_decay_and_cap():
    ball_pos = torch.tensor([[0.0, 0.0, 0.11]] * 3)
    goal_pos = torch.tensor([[5.0, 0.0]] * 3)
    ball_vel = torch.tensor([[5.0, 0.0, 0.0],
                             [5.0, 0.0, 0.0],
                             [100.0, 0.0, 0.0]])
    t_moving = torch.tensor([0.0, 0.2, 0.0])
    r = soccer_util.compute_kick_direction_reward(ball_pos, ball_vel, goal_pos,
                                                  1.0, 0.2, t_moving, 30.0)
    assert r[1] == pytest.approx(4.0 * math.exp(-1.0), abs=1e-5)
    assert r[2] == pytest.approx(30.0)  # capped


def test_kick_direction_reward_degenerate_ball_on_goal():
    ball_pos = torch.tensor([[5.0, 0.0, 0.11]])
    goal_pos = torch.tensor([[5.0, 0.0]])
    ball_vel = torch.tensor([[3.0, 0.0, 0.0]])
    r = soccer_util.compute_kick_direction_reward(ball_pos, ball_vel, goal_pos,
                                                  1.0, 0.2, torch.zeros(1), 30.0)
    assert torch.all(torch.isfinite(r))


def test_observations_identity_heading():
    root_pos = torch.tensor([[1.0, 2.0, 0.8]])
    root_rot = _yaw_quat(0.0)
    ball_pos = torch.tensor([[3.0, 2.0, 0.11]])
    goal_pos = torch.tensor([[7.0, 2.0]])
    goal_dir = torch.tensor([[-1.0, 0.0]])

    obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                  goal_pos, goal_dir)
    assert obs.shape == (1, 6)
    expected = torch.tensor([[2.0, 0.0, 6.0, 0.0, -1.0, 0.0]])
    assert torch.allclose(obs, expected, atol=1e-5)


def test_observations_rotated_heading():
    # robot facing +y (yaw 90deg): ball straight ahead in world +y appears at
    # local +x.
    root_pos = torch.tensor([[0.0, 0.0, 0.8]])
    root_rot = _yaw_quat(math.pi / 2)
    ball_pos = torch.tensor([[0.0, 2.0, 0.11]])
    goal_pos = torch.tensor([[1.0, 0.0]])
    goal_dir = torch.tensor([[-1.0, 0.0]])

    obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                  goal_pos, goal_dir)
    expected = torch.tensor([[2.0, 0.0,    # ball straight ahead
                              0.0, -1.0,   # goal 1m to the robot's right
                              0.0, 1.0]])  # goal dir (-1,0) world -> local (0,1)
    assert torch.allclose(obs, expected, atol=1e-5)


def test_observations_invariant_to_roll_pitch():
    # heading extraction should ignore roll/pitch components
    root_pos = torch.zeros(1, 3)
    yaw = 0.7
    q_yaw = _yaw_quat(yaw)[0]
    # compose with a pitch rotation (around y): q_total = q_yaw * q_pitch
    half_p = 0.3
    q_pitch = torch.tensor([0.0, math.sin(half_p), 0.0, math.cos(half_p)])
    from util import torch_util
    q_total = torch_util.quat_mul(q_yaw.unsqueeze(0), q_pitch.unsqueeze(0))

    ball_pos = torch.tensor([[1.5, -0.5, 0.11]])
    goal_pos = torch.tensor([[5.0, 0.0]])
    goal_dir = torch.tensor([[-1.0, 0.0]])

    obs_yaw = soccer_util.compute_soccer_observations(root_pos, _yaw_quat(yaw),
                                                      ball_pos, goal_pos, goal_dir)
    obs_full = soccer_util.compute_soccer_observations(root_pos, q_total,
                                                       ball_pos, goal_pos, goal_dir)
    assert torch.allclose(obs_yaw, obs_full, atol=1e-4)


# ---------------------------------------------------------------------------
# potential-based rewards
# ---------------------------------------------------------------------------

def test_ball_approach_positive_when_closing():
    prev_root = torch.tensor([[0.0, 0.0, 0.8]])
    root = torch.tensor([[1.0, 0.0, 0.8]])
    ball = torch.tensor([[3.0, 0.0, 0.11]])

    r = soccer_util.compute_ball_approach_reward(root, prev_root, ball, ball)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-5)

    r_away = soccer_util.compute_ball_approach_reward(prev_root, root, ball, ball)
    assert torch.allclose(r_away, torch.tensor([-1.0]), atol=1e-5)


def test_goal_progress_telescopes():
    # sum of shaping increments equals total potential drop over a trajectory
    goal = torch.tensor([[7.0, 0.0]])
    traj = torch.tensor([[0.0, 0.0, 0.11],
                         [1.0, 0.5, 0.11],
                         [3.0, -0.2, 0.11],
                         [6.0, 0.1, 0.11]])
    total = torch.zeros(1)
    for i in range(1, traj.shape[0]):
        total += soccer_util.compute_goal_progress_reward(
            traj[i].unsqueeze(0), traj[i - 1].unsqueeze(0), goal)

    d0 = torch.linalg.norm(goal[0] - traj[0, 0:2])
    dT = torch.linalg.norm(goal[0] - traj[-1, 0:2])
    assert torch.allclose(total, (d0 - dT).unsqueeze(0), atol=1e-5)


def test_progress_ignores_z():
    goal = torch.tensor([[7.0, 0.0]])
    flat = soccer_util.compute_goal_progress_reward(
        torch.tensor([[2.0, 0.0, 0.11]]), torch.tensor([[1.0, 0.0, 0.11]]), goal)
    lofted = soccer_util.compute_goal_progress_reward(
        torch.tensor([[2.0, 0.0, 1.5]]), torch.tensor([[1.0, 0.0, 0.11]]), goal)
    assert torch.allclose(flat, lofted, atol=1e-5)


# ---------------------------------------------------------------------------
# goal / out-of-bounds detection
# ---------------------------------------------------------------------------

def test_goal_scored_inside_mouth():
    goal_pos = torch.tensor([[7.0, 0.0]])
    goal_dir = torch.tensor([[-1.0, 0.0]])  # points into the field
    width = 2.6
    radius = 0.11

    inside = torch.tensor([[7.2, 0.5, 0.11]])   # 0.2m past the line, in mouth
    flags = soccer_util.compute_goal_scored_flags(inside, goal_pos, goal_dir,
                                                  width, radius)
    assert flags.tolist() == [True]

    on_line = torch.tensor([[7.05, 0.0, 0.11]])  # not fully across (depth 0.05 < r)
    flags = soccer_util.compute_goal_scored_flags(on_line, goal_pos, goal_dir,
                                                  width, radius)
    assert flags.tolist() == [False]

    wide = torch.tensor([[7.2, 1.4, 0.11]])  # past the line but outside the mouth
    flags = soccer_util.compute_goal_scored_flags(wide, goal_pos, goal_dir,
                                                  width, radius)
    assert flags.tolist() == [False]

    in_front = torch.tensor([[6.0, 0.0, 0.11]])
    flags = soccer_util.compute_goal_scored_flags(in_front, goal_pos, goal_dir,
                                                  width, radius)
    assert flags.tolist() == [False]


def test_goal_scored_rotated_goal():
    # goal on the -x side, facing +x
    goal_pos = torch.tensor([[-7.0, 0.0]])
    goal_dir = torch.tensor([[1.0, 0.0]])
    scored = torch.tensor([[-7.3, -0.8, 0.11]])
    flags = soccer_util.compute_goal_scored_flags(scored, goal_pos, goal_dir,
                                                  2.6, 0.11)
    assert flags.tolist() == [True]


def test_out_of_bounds():
    balls = torch.tensor([[0.0, 0.0, 0.11],
                          [7.1, 0.0, 0.11],
                          [-7.1, 0.0, 0.11],
                          [0.0, 4.6, 0.11],
                          [6.9, 4.4, 0.11]])
    flags = soccer_util.compute_out_of_bounds_flags(balls, 14.0, 9.0)
    assert flags.tolist() == [False, True, True, True, False]


def test_ball_out_exempts_goal_corridor():
    goal_pos = torch.tensor([[7.0, 0.0]])
    goal_dir = torch.tensor([[-1.0, 0.0]])
    args = (14.0, 9.0, goal_pos, goal_dir, 2.6, 0.11)

    # crossing the line inside the mouth: NOT out (goal detection owns it)
    crossing = torch.tensor([[7.05, 0.5, 0.11]])
    assert soccer_util.compute_ball_out_flags(crossing, *args).tolist() == [False]

    # crossing the line outside the mouth: out
    wide = torch.tensor([[7.05, 1.5, 0.11]])
    assert soccer_util.compute_ball_out_flags(wide, *args).tolist() == [True]

    # far behind the goal line (past the crossing band): out
    behind = torch.tensor([[7.3, 0.0, 0.11]])
    assert soccer_util.compute_ball_out_flags(behind, *args).tolist() == [True]

    # sanity: goal fires before the corridor exemption expires
    goal_flag = soccer_util.compute_goal_scored_flags(
        torch.tensor([[7.12, 0.0, 0.11]]), goal_pos, goal_dir, 2.6, 0.11)
    assert goal_flag.tolist() == [True]

    # ordinary sideline out unchanged
    side = torch.tensor([[0.0, 4.6, 0.11]])
    assert soccer_util.compute_ball_out_flags(side, *args).tolist() == [True]


# ---------------------------------------------------------------------------
# auxiliary shaping
# ---------------------------------------------------------------------------

def test_stagnation_flags():
    root = torch.tensor([[0.05, 0.0, 0.8], [1.0, 1.0, 0.8]])
    window = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
    flags = soccer_util.compute_stagnation_flags(root, window, 0.1)
    assert flags.tolist() == [True, False]


def test_kick_components_sideways_vs_forward():
    root_rot = _yaw_quat(0.0, n=3)
    foot_vel = torch.tensor([[0.0, 2.0, 0.0],    # pure sideways
                             [2.0, 0.0, 0.0],    # pure forward
                             [-2.0, 0.0, 0.0]])  # backward: forward comp clamps to 0
    contact = torch.tensor([True, True, True])

    comps = soccer_util.compute_kick_components(root_rot, foot_vel, contact)
    assert comps.shape == (3, 2)
    expected = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    assert torch.allclose(comps, expected, atol=1e-5)


def test_kick_components_zero_without_contact():
    root_rot = _yaw_quat(0.0)
    foot_vel = torch.tensor([[1.0, 3.0, 0.5]])
    comps = soccer_util.compute_kick_components(root_rot, foot_vel,
                                                torch.tensor([False]))
    assert torch.allclose(comps, torch.zeros(1, 2))


def test_kick_components_follow_heading():
    # robot facing +y: world +x velocity is sideways in the heading frame
    root_rot = _yaw_quat(math.pi / 2)
    foot_vel = torch.tensor([[2.0, 0.0, 0.0]])
    comps = soccer_util.compute_kick_components(root_rot, foot_vel,
                                                torch.tensor([True]))
    assert torch.allclose(comps, torch.tensor([[2.0, 0.0]]), atol=1e-5)


def test_foot_proximity_penalty():
    left = torch.tensor([[0.0, 0.1, 0.05], [0.0, 0.5, 0.05]])
    right = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]])
    pen = soccer_util.compute_foot_proximity_penalty(left, right, 0.2)
    assert torch.allclose(pen, torch.tensor([0.1, 0.0]), atol=1e-5)


def test_action_rate_penalty():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    prev = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    pen = soccer_util.compute_action_rate_penalty(a, prev)
    assert torch.allclose(pen, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_joint_limit_penalty():
    low = torch.tensor([-1.0, -1.0])
    high = torch.tensor([1.0, 1.0])
    dof = torch.tensor([[0.0, 0.5], [1.2, -1.3]])
    pen = soccer_util.compute_joint_limit_penalty(dof, low, high)
    assert torch.allclose(pen, torch.tensor([0.0, 0.5]), atol=1e-6)


def test_ball_contact_flags():
    foot = torch.tensor([[0.0, 0.0, 0.05], [1.0, 0.0, 0.05]])
    ball = torch.tensor([[0.1, 0.0, 0.11], [0.0, 0.0, 0.11]])
    contact = soccer_util.compute_ball_contact_flags(foot, ball, 0.25)
    assert contact.tolist() == [True, False]


def test_base_accel_penalty():
    vel = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dt = 0.5
    pen = soccer_util.compute_base_accel_penalty(vel, prev, dt)
    # env 0: a = (1 - 0) / 0.5 = 2 -> ||a||^2 = 4; env 1: 0
    assert torch.allclose(pen, torch.tensor([4.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

def test_batch_shapes_and_finiteness():
    torch.manual_seed(0)
    n = 64
    root_pos = torch.randn(n, 3)
    root_rot = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    ball_pos = torch.randn(n, 3)
    goal_pos = torch.randn(n, 2)
    goal_dir = torch.nn.functional.normalize(torch.randn(n, 2), dim=-1)

    obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                  goal_pos, goal_dir)
    assert obs.shape == (n, 6)
    assert torch.isfinite(obs).all()

    r1 = soccer_util.compute_ball_approach_reward(root_pos, root_pos + 0.1,
                                                  ball_pos, ball_pos)
    r2 = soccer_util.compute_goal_progress_reward(ball_pos, ball_pos + 0.1, goal_pos)
    assert r1.shape == (n,) and r2.shape == (n,)
    assert torch.isfinite(r1).all() and torch.isfinite(r2).all()


def test_determinism():
    torch.manual_seed(3)
    n = 16
    root_pos = torch.randn(n, 3)
    root_rot = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    ball_pos = torch.randn(n, 3)
    goal_pos = torch.randn(n, 2)
    goal_dir = torch.nn.functional.normalize(torch.randn(n, 2), dim=-1)

    a = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                goal_pos, goal_dir)
    b = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                goal_pos, goal_dir)
    assert torch.equal(a, b)


def test_field_line_segments_geometry():
    import numpy as np

    length, width, goal_w = 14.0, 9.0, 2.6
    starts, ends, cols = soccer_util.build_field_line_segments(length, width, goal_w)

    assert starts.shape == ends.shape
    assert starts.shape[1] == 3 and cols.shape == (starts.shape[0], 4)
    assert starts.dtype == np.float32 and cols.dtype == np.float32
    assert np.isfinite(starts).all() and np.isfinite(ends).all()

    # everything stays inside the field footprint
    pts = np.concatenate([starts, ends], axis=0)
    assert np.abs(pts[:, 0]).max() <= 0.5 * length + 1e-5
    assert np.abs(pts[:, 1]).max() <= 0.5 * width + 1e-5

    # ground markings sit at a single small z; only goal posts rise above
    ground = pts[pts[:, 2] < 0.1]
    assert np.allclose(ground[:, 2], ground[0, 2])
    assert pts[:, 2].max() > 1.0  # posts exist

    # goal mouth: green segments on the +x goal line span exactly goal_width
    green = np.abs(cols[:, 1] - 0.9) < 1e-6
    gpts = np.concatenate([starts[green], ends[green]], axis=0)
    assert np.allclose(gpts[:, 0], 0.5 * length)
    assert np.isclose(gpts[:, 1].max() - gpts[:, 1].min(), goal_w)

    # left/right symmetry of the white ground markings
    white = ~green
    wpts = np.concatenate([starts[white], ends[white]], axis=0)
    assert np.allclose(np.sort(wpts[:, 1]), np.sort(-wpts[:, 1]), atol=1e-5)


# ---------------------------------------------------------------------------
# field grid layout / uneven-ground extent
# ---------------------------------------------------------------------------

def test_field_offsets_grid_layout():
    # 5 envs -> 3 cols x 2 rows, pitch = field + 2*sep
    offsets = soccer_util.compute_field_offsets(5, 14.0, 9.0, 2.0)
    assert offsets.shape == (5, 2)
    assert offsets.dtype == np.float32
    px, py = 18.0, 13.0
    expected = np.array([[-px, -0.5 * py], [0.0, -0.5 * py], [px, -0.5 * py],
                         [-px, 0.5 * py], [0.0, 0.5 * py]], dtype=np.float32)
    assert np.allclose(offsets, expected)
    # grid is centered
    assert abs(offsets[:, 0].max() + offsets[:, 0].min()) < 1e-4

    # no two fields overlap: distinct offsets at least one pitch apart per axis
    d = np.abs(offsets[:, None, :] - offsets[None, :, :])
    far = (d[..., 0] > px - 1e-4) | (d[..., 1] > py - 1e-4)
    np.fill_diagonal(far, True)
    assert far.all()


def test_field_grid_extent_covers_all_fields():
    for n in (1, 2, 5, 8, 33):
        offsets = soccer_util.compute_field_offsets(n, 14.0, 9.0, 2.0)
        size_x, size_y = soccer_util.compute_field_grid_extent(n, 14.0, 9.0, 2.0)
        # every field (center +- half pitch) inside the extent
        assert np.abs(offsets[:, 0]).max() + 0.5 * 18.0 <= 0.5 * size_x + 1e-4
        assert np.abs(offsets[:, 1]).max() + 0.5 * 13.0 <= 0.5 * size_y + 1e-4


# ---------------------------------------------------------------------------
# steering anneal schedule
# ---------------------------------------------------------------------------

def test_anneal_scale_disabled_and_linear():
    # disabled: start < 0
    assert soccer_util.compute_anneal_scale(10_000_000, -1.0, -1.0) == 1.0
    # before start
    assert soccer_util.compute_anneal_scale(4_999_999, 5e6, 15e6) == 1.0
    # linear midpoint
    assert soccer_util.compute_anneal_scale(10e6, 5e6, 15e6) == pytest.approx(0.5)
    # end and beyond
    assert soccer_util.compute_anneal_scale(15e6, 5e6, 15e6) == 0.0
    assert soccer_util.compute_anneal_scale(30e6, 5e6, 15e6) == 0.0


def test_anneal_scale_step_schedule_for_eval():
    # start == end == 0: zero from the very first sample (eval configs)
    assert soccer_util.compute_anneal_scale(0, 0.0, 0.0) == 0.0
    assert soccer_util.compute_anneal_scale(1e9, 0.0, 0.0) == 0.0
    # degenerate end <= start behaves as a step at start
    assert soccer_util.compute_anneal_scale(4e6, 5e6, 5e6) == 1.0
    assert soccer_util.compute_anneal_scale(5e6, 5e6, 5e6) == 0.0


# ---------------------------------------------------------------------------
# virtual perception (Frente E, paper section 9)
# ---------------------------------------------------------------------------

def test_perception_noise_std_formula():
    dist = torch.tensor([0.0, 1.0, 7.0])
    std = soccer_util.compute_perception_noise_std(dist, 0.124, 0.149)
    assert torch.allclose(std, torch.tensor([0.149, 0.273, 1.017]), atol=1e-6)


def test_detection_prob_plateau_decay_and_fov():
    dist = torch.tensor([1.0, 7.0, 8.5, 10.0, 12.0])
    in_fov = torch.ones(5, dtype=torch.bool)
    p = soccer_util.compute_ball_detection_prob(dist, in_fov, 0.9, 7.0, 3.0)
    assert torch.allclose(p, torch.tensor([0.9, 0.9, 0.45, 0.0, 0.0]), atol=1e-6)
    # outside the FOV the probability is zero regardless of distance
    p_out = soccer_util.compute_ball_detection_prob(dist, torch.zeros(5, dtype=torch.bool),
                                                    0.9, 7.0, 3.0)
    assert torch.all(p_out == 0.0)


def test_ball_in_fov_bearing():
    root_pos = torch.zeros([4, 3])
    root_rot = _yaw_quat(0.0, 4)  # heading +x
    ball_pos = torch.tensor([[2.0, 0.0, 0.11],    # dead ahead
                             [0.0, 2.0, 0.11],    # 90 deg left
                             [-2.0, 0.0, 0.11],   # behind
                             [2.0, 1.0, 0.11]])   # ~26.6 deg left
    half = math.radians(60.0)  # 120 deg full FOV
    in_fov = soccer_util.compute_ball_in_fov(root_pos, root_rot, ball_pos, half)
    assert in_fov.tolist() == [True, False, False, True]
    # fov <= 0 disables the check
    all_in = soccer_util.compute_ball_in_fov(root_pos, root_rot, ball_pos, 0.0)
    assert torch.all(all_in)


def test_ball_in_fov_follows_heading():
    # robot rotated 90 deg left: a ball at +y is now dead ahead
    root_pos = torch.zeros([1, 3])
    root_rot = _yaw_quat(math.pi / 2.0, 1)
    ball_pos = torch.tensor([[0.0, 3.0, 0.11]])
    in_fov = soccer_util.compute_ball_in_fov(root_pos, root_rot, ball_pos,
                                             math.radians(60.0))
    assert bool(in_fov[0])


# ---------------------------------------------------------------------------
# episode termination on ball events (paper 4.1)
# ---------------------------------------------------------------------------

NULL, FAIL, SUCC, TIME = 0, 1, 2, 3


def test_ball_event_dones_goal_succ_oob_fail():
    done = torch.tensor([NULL, NULL, NULL], dtype=torch.long)
    goal = torch.tensor([True, False, False])
    oob = torch.tensor([False, True, False])
    new_done, soft = soccer_util.apply_ball_event_dones(done, goal, oob,
                                                        NULL, SUCC, FAIL)
    assert new_done.tolist() == [SUCC, FAIL, NULL]
    assert soft.tolist() == [True, True, False]
    # input must not be mutated in place (caller writes back explicitly)
    assert done.tolist() == [NULL, NULL, NULL]


def test_ball_event_dones_fall_and_timeout_take_precedence():
    # a fall (FAIL) or timeout (TIME) already decided by the caller wins:
    # the env gets a FULL reset, never the soft ball-only one
    done = torch.tensor([FAIL, TIME, FAIL, TIME], dtype=torch.long)
    goal = torch.tensor([True, True, False, False])
    oob = torch.tensor([False, False, True, True])
    new_done, soft = soccer_util.apply_ball_event_dones(done, goal, oob,
                                                        NULL, SUCC, FAIL)
    assert new_done.tolist() == [FAIL, TIME, FAIL, TIME]
    assert not soft.any()


def test_ball_event_dones_no_events_noop():
    done = torch.tensor([NULL, TIME, FAIL], dtype=torch.long)
    goal = torch.zeros(3, dtype=torch.bool)
    oob = torch.zeros(3, dtype=torch.bool)
    new_done, soft = soccer_util.apply_ball_event_dones(done, goal, oob,
                                                        NULL, SUCC, FAIL)
    assert new_done.tolist() == [NULL, TIME, FAIL]
    assert not soft.any()


def test_ball_event_dones_matches_env_flag_convention():
    # env guarantees oob &= ~goal; the goal branch must win when both are
    # passed anyway (goal is applied to its own mask, oob to its own)
    done = torch.tensor([NULL], dtype=torch.long)
    goal = torch.tensor([True])
    oob = torch.tensor([False])
    new_done, soft = soccer_util.apply_ball_event_dones(done, goal, oob,
                                                        NULL, SUCC, FAIL)
    assert new_done.tolist() == [SUCC]
    assert soft.tolist() == [True]
