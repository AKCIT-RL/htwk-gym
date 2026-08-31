"""Mirror operator tests.

The dangerous failure mode here is silent: a wrong index or sign produces no
crash, only a policy that learns the wrong symmetry over 2 GPU-hours. The
involution test alone does NOT catch a wrong sign on a symmetric pair, so the
load-bearing test in this file is `test_mirror_commutes_with_the_real_obs_code`,
which checks the operator against char_env.compute_char_obs on random states:

    M_obs( obs(state) )  ==  obs( mirror(state) )

Everything else is a cheaper check that localises a failure once that one trips.
"""

import sys
from pathlib import Path

import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

import learning.mirror_util as mirror_util  # noqa: E402
from anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from envs.char_env import compute_char_obs  # noqa: E402

G1_ASSET = MIMICKIT_ROOT / "data" / "assets" / "g1" / "g1.xml"

# from data/envs/wamp_g1_steering_env.yaml
KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "head_link",
              "left_wrist_yaw_link", "right_wrist_yaw_link"]
STEERING_OBS_SIZE = 242


def _char_model():
    m = MJCFCharModel(device="cpu")
    m.load(str(G1_ASSET))
    return m


def _obs_mirror(model):
    return mirror_util.build_obs_mirror(
        task_key="TaskSteeringEnv", char_model=model, key_body_names=KEY_BODIES,
        root_height_obs=True, obs_size=STEERING_OBS_SIZE)


# forward_kinematics mixes in the model's own float32 buffers, so anything that
# flows through it has to be float32 too.
#
# Measured residual on 64 random states, per observation block:
#   root_h, root_vel, root_ang_vel, joint_rot, dof_vel   0.0 exactly
#   root_rot                                             4.8e-07
#   key_pos                                              1.0e-05
# Only key_pos carries error, and only because it comes out of a ~30-link
# float32 quaternion chain in forward_kinematics - the arithmetic, not the map.
# For scale, test_a_single_wrong_sign_is_detected below measures what an actual
# bug looks like: 6.9, i.e. ~7e5 times larger. The tolerance sits between them
# with five orders of magnitude to spare.
DT = torch.float32
TOL = 1e-4


def _rand_state(model, n, seed):
    """A physically arbitrary, deliberately asymmetric state."""
    g = torch.Generator().manual_seed(seed)
    dof = model.get_dof_size()

    root_pos = torch.randn(n, 3, generator=g, dtype=DT)
    root_pos[:, 2] = 0.8 + 0.1 * torch.randn(n, generator=g, dtype=DT)

    q = torch.randn(n, 4, generator=g, dtype=DT)
    root_rot = q / torch.linalg.norm(q, dim=-1, keepdim=True)   # (x, y, z, w)

    root_vel = torch.randn(n, 3, generator=g, dtype=DT)
    root_ang_vel = torch.randn(n, 3, generator=g, dtype=DT)
    dof_pos = 0.5 * torch.randn(n, dof, generator=g, dtype=DT)
    dof_vel = torch.randn(n, dof, generator=g, dtype=DT)
    return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel


def _mirror_state(state, dof_perm, dof_signs):
    """Reflect a full character state across the sagittal (XZ) plane.

    Independent of mirror_util's observation map: this is written from the
    physics, so agreeing with it is evidence, not tautology. For a rotation by
    theta about axis a, conjugating by M = diag(1,-1,1) gives a rotation by
    -theta about M a, i.e. (x, y, z, w) -> (-x, y, -z, w).
    """
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = state
    flip_vec = torch.tensor([1.0, -1.0, 1.0], dtype=DT)
    flip_pseudo = torch.tensor([-1.0, 1.0, -1.0], dtype=DT)
    flip_quat = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=DT)

    perm = torch.as_tensor(dof_perm, dtype=torch.long)
    signs = torch.as_tensor(dof_signs, dtype=DT)

    return (root_pos * flip_vec,
            root_rot * flip_quat,
            root_vel * flip_vec,
            root_ang_vel * flip_pseudo,
            dof_pos[:, perm] * signs,
            dof_vel[:, perm] * signs)


def _obs_of(model, state):
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = state
    joint_rot = model.dof_to_rot(dof_pos)
    body_pos, _ = model.forward_kinematics(root_pos, root_rot, joint_rot)
    key_ids = [model.get_body_id(n) for n in KEY_BODIES]
    key_pos = body_pos[:, key_ids, :]
    return compute_char_obs(root_pos=root_pos, root_rot=root_rot, root_vel=root_vel,
                            root_ang_vel=root_ang_vel, joint_rot=joint_rot,
                            dof_vel=dof_vel, key_pos=key_pos,
                            global_obs=False, root_height_obs=True)


# --------------------------------------------------------------------------
# the load-bearing test
# --------------------------------------------------------------------------

def test_mirror_commutes_with_the_real_obs_code():
    """M(obs(s)) == obs(mirror(s)) on arbitrary states.

    This is the only test that ties the operator to the observation the policy
    actually receives. A swapped index, a missed sign, or a block boundary off
    by one all break it.
    """
    model = _char_model()
    obs_perm, obs_signs = _obs_mirror(model)
    dof_perm, dof_signs = mirror_util.build_dof_mirror(model)

    perm = torch.as_tensor(obs_perm, dtype=torch.long)
    signs = torch.as_tensor(obs_signs, dtype=DT)

    state = _rand_state(model, n=32, seed=0)
    # the steering task block is not produced by compute_char_obs, so compare
    # only the character block, which is what this operator derives
    char_dims = STEERING_OBS_SIZE - len(mirror_util.TASK_OBS_MIRROR["TaskSteeringEnv"])

    obs = _obs_of(model, state)
    obs_of_mirrored = _obs_of(model, _mirror_state(state, dof_perm, dof_signs))
    mirrored_obs = obs[:, perm[:char_dims]] * signs[:char_dims]

    err = torch.max(torch.abs(mirrored_obs - obs_of_mirrored)).item()
    assert err < TOL, "mirror disagrees with the real obs code by {:.3e}".format(err)


def test_a_single_wrong_sign_is_detected():
    """Negative control: corrupt one sign, the commutation test must fail.

    Without this, a tolerance loose enough to absorb float32 noise could also
    be absorbing a real defect. It pins the separation between the two.
    """
    model = _char_model()
    obs_perm, obs_signs = _obs_mirror(model)
    dof_perm, dof_signs = mirror_util.build_dof_mirror(model)

    perm = torch.as_tensor(obs_perm, dtype=torch.long)
    signs = torch.as_tensor(obs_signs, dtype=DT)
    char_dims = STEERING_OBS_SIZE - len(mirror_util.TASK_OBS_MIRROR["TaskSteeringEnv"])

    state = _rand_state(model, n=32, seed=0)
    obs = _obs_of(model, state)
    obs_of_mirrored = _obs_of(model, _mirror_state(state, dof_perm, dof_signs))

    corrupted = signs.clone()
    corrupted[8] = -corrupted[8]                      # one slot of root_rot
    err = torch.max(torch.abs(obs[:, perm[:char_dims]] * corrupted[:char_dims]
                              - obs_of_mirrored)).item()
    assert err > 1e3 * TOL, \
        "a wrong sign produced only {:.3e}; the tolerance is not discriminating".format(err)


def test_symmetric_pose_is_a_fixed_point():
    """A left-right symmetric state must be its own mirror.

    Cheaper than the test above and it isolates sign errors specifically: an
    index-only bug can still leave the symmetric pose invariant, a sign bug
    cannot.
    """
    model = _char_model()
    obs_perm, obs_signs = _obs_mirror(model)
    dof_perm, dof_signs = mirror_util.build_dof_mirror(model)

    perm = torch.as_tensor(obs_perm, dtype=torch.long)
    signs = torch.as_tensor(obs_signs, dtype=DT)
    dperm = torch.as_tensor(dof_perm, dtype=torch.long)
    dsigns = torch.as_tensor(dof_signs, dtype=DT)

    g = torch.Generator().manual_seed(7)
    n, dof = 8, model.get_dof_size()

    # symmetrise an arbitrary pose: p_sym = (p + mirror(p)) / 2 is a fixed point
    # of the DOF mirror by construction
    raw = 0.4 * torch.randn(n, dof, generator=g, dtype=DT)
    dof_pos = 0.5 * (raw + raw[:, dperm] * dsigns)
    raw_v = 0.4 * torch.randn(n, dof, generator=g, dtype=DT)
    dof_vel = 0.5 * (raw_v + raw_v[:, dperm] * dsigns)

    root_pos = torch.zeros(n, 3, dtype=DT)
    root_pos[:, 2] = 0.8
    root_rot = torch.zeros(n, 4, dtype=DT)
    root_rot[:, 3] = 1.0                                   # identity, facing +x
    root_vel = torch.zeros(n, 3, dtype=DT)
    root_vel[:, 0] = 1.2                                   # forward only
    root_ang_vel = torch.zeros(n, 3, dtype=DT)
    root_ang_vel[:, 1] = 0.3                               # pitch only

    state = (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
    char_dims = STEERING_OBS_SIZE - len(mirror_util.TASK_OBS_MIRROR["TaskSteeringEnv"])

    obs = _obs_of(model, state)
    mirrored = obs[:, perm[:char_dims]] * signs[:char_dims]
    err = torch.max(torch.abs(mirrored - obs)).item()
    assert err < TOL, "symmetric pose is not a fixed point (err {:.3e})".format(err)


# --------------------------------------------------------------------------
# structural properties
# --------------------------------------------------------------------------

def test_obs_mirror_is_an_exact_involution():
    model = _char_model()
    obs_perm, obs_signs = _obs_mirror(model)
    assert mirror_util.check_involution(obs_perm, obs_signs)

    perm = torch.as_tensor(obs_perm, dtype=torch.long)
    signs = torch.as_tensor(obs_signs, dtype=torch.float64)
    x = torch.randn(4, STEERING_OBS_SIZE, dtype=torch.float64)
    twice = mirror_util.mirror(mirror_util.mirror(x, perm, signs), perm, signs)
    assert torch.equal(twice, x)


def test_action_mirror_is_an_exact_involution():
    model = _char_model()
    perm_l, signs_l = mirror_util.build_dof_mirror(model)
    assert mirror_util.check_involution(perm_l, signs_l)

    perm = torch.as_tensor(perm_l, dtype=torch.long)
    signs = torch.as_tensor(signs_l, dtype=torch.float64)
    x = torch.randn(4, model.get_dof_size(), dtype=torch.float64)
    assert torch.equal(mirror_util.mirror(mirror_util.mirror(x, perm, signs), perm, signs), x)


def test_mirror_preserves_norm():
    model = _char_model()
    obs_perm, obs_signs = _obs_mirror(model)
    perm = torch.as_tensor(obs_perm, dtype=torch.long)
    signs = torch.as_tensor(obs_signs, dtype=torch.float64)

    x = torch.randn(16, STEERING_OBS_SIZE, dtype=torch.float64)
    a = torch.linalg.norm(x, dim=-1)
    b = torch.linalg.norm(mirror_util.mirror(x, perm, signs), dim=-1)
    assert torch.max(torch.abs(a - b)).item() < 1e-12


def test_dof_mirror_pairs_limbs_and_signs_axes():
    model = _char_model()
    perm, signs = mirror_util.build_dof_mirror(model)

    names = [model.get_joint(j).name for j in range(1, model.get_num_joints())
             if model.get_joint(j).get_dof_dim() > 0]
    for i, name in enumerate(names):
        partner = names[perm[i]]
        if name.startswith("left_"):
            assert partner == "right_" + name[5:]
        elif name.startswith("right_"):
            assert partner == "left_" + name[6:]
        else:
            assert partner == name, "{} should map to itself".format(name)

        expected = 1.0 if name.endswith("pitch_joint") or name.endswith("knee_joint") \
            or name.endswith("elbow_joint") else -1.0
        assert signs[i] == expected, "{} got sign {}".format(name, signs[i])


def test_joint_rot_block_covers_zero_dof_joints():
    """The G1 has a FIXED head_link among its joints.

    dof_to_rot emits one rotation per non-root joint, DOF-less ones included,
    so the joint_rot block is wider than the DOF block and needs its own
    permutation. Conflating the two shifts every arm slot by one.
    """
    model = _char_model()
    jr_perm, jr_signs = mirror_util.build_joint_rot_mirror(model)
    dof_perm, _ = mirror_util.build_dof_mirror(model)

    assert len(jr_perm) == 6 * (model.get_num_joints() - 1)
    assert len(dof_perm) == model.get_dof_size()
    assert len(jr_perm) // 6 != len(dof_perm), \
        "this asset no longer has a DOF-less joint; revisit the test's premise"
    assert mirror_util.check_involution(jr_perm, jr_signs)


# --------------------------------------------------------------------------
# guards that must fire
# --------------------------------------------------------------------------

def test_obs_size_mismatch_is_rejected():
    model = _char_model()
    with pytest.raises(ValueError, match="layout changed"):
        mirror_util.build_obs_mirror(task_key="TaskSteeringEnv", char_model=model,
                                     key_body_names=KEY_BODIES, root_height_obs=True,
                                     obs_size=STEERING_OBS_SIZE + 1)


def test_unregistered_task_is_rejected():
    model = _char_model()
    with pytest.raises(KeyError, match="TaskSoccerEnv"):
        mirror_util.build_obs_mirror(task_key="TaskSoccerEnv", char_model=model,
                                     key_body_names=KEY_BODIES, root_height_obs=True,
                                     obs_size=STEERING_OBS_SIZE)


def test_unpaired_key_body_is_rejected():
    model = _char_model()
    with pytest.raises(ValueError, match="no mirror partner"):
        mirror_util.build_obs_mirror(task_key="TaskSteeringEnv", char_model=model,
                                     key_body_names=["left_ankle_roll_link"],
                                     root_height_obs=True, obs_size=STEERING_OBS_SIZE)


def test_non_principal_hinge_axis_is_rejected():
    with pytest.raises(ValueError, match="not aligned"):
        mirror_util._hinge_sign([0.7, 0.7, 0.0])


# --------------------------------------------------------------------------
# normalizer equivariance (the precondition for mirroring in normalized space)
# --------------------------------------------------------------------------

def test_g1_action_normalizer_commutes_with_the_mirror():
    """The action normalizer's mean/std come from the MJCF joint limits.

    Mirroring actions in normalized space is only valid when mirror(mean)==mean
    and std[perm]==std. It holds for the G1 because its limits are exact mirror
    images; this test pins that so a change of asset fails loudly.
    """
    import xml.etree.ElementTree as ET

    model = _char_model()
    perm, signs = mirror_util.build_dof_mirror(model)

    root = ET.parse(str(G1_ASSET)).getroot()
    ranges = {}
    for j in root.findall("./worldbody//joint"):
        rng = j.get("range")
        if rng is not None:
            lo, hi = (float(v) for v in rng.split())
            ranges[j.get("name")] = (lo, hi)

    names = [model.get_joint(j).name for j in range(1, model.get_num_joints())
             if model.get_joint(j).get_dof_dim() > 0]
    lows = torch.tensor([ranges[n][0] for n in names], dtype=torch.float64)
    highs = torch.tensor([ranges[n][1] for n in names], dtype=torch.float64)

    mean = 0.5 * (highs + lows)
    std = 0.5 * (highs - lows)

    ok, mean_err, std_err = mirror_util.check_normalizer_equivariance(mean, std, perm, signs)
    assert ok, "mean err {:.3e}, std err {:.3e}".format(mean_err, std_err)
