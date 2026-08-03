import sys
from pathlib import Path

import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from learning.multi_critic_util import compute_multi_critic_adv


def _make_adv(t=16, b=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(t, b, 2)


def _full_mask(t=16, b=8):
    return torch.ones(t * b, dtype=torch.bool)


def test_shapes_dtype_and_finite():
    adv = _make_adv()
    out, info = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)

    assert out.shape == (16, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    for key in ("adv0_mean", "adv0_std", "adv1_mean", "adv1_std", "adv_mean", "adv_std"):
        assert torch.isfinite(info[key]).all()


def test_per_stream_scale_invariance():
    # Rescaling the raw magnitude of either stream must not change the
    # combined advantage: this is the property that prevents one reward
    # stream from drowning the other (the failure mode of the single critic).
    adv = _make_adv(seed=1)
    scaled = adv * torch.tensor([1000.0, 1e-3])

    out_a, _ = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)
    out_b, _ = compute_multi_critic_adv(scaled, [2.0, 1.0], _full_mask(), 4.0)

    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_single_stream_weights_recover_stream():
    # With weight only on one stream, output equals that stream standardized.
    adv = _make_adv(seed=2)
    mask = _full_mask()

    for s, weights in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
        out, _ = compute_multi_critic_adv(adv, weights, mask, 10.0)
        stream = adv[..., s]
        std, mean = torch.std_mean(stream.flatten())
        expected = (stream - mean) / std
        # Re-standardization of an already standardized stream is idempotent
        # up to numerical noise.
        assert torch.allclose(out, expected, atol=1e-4)


def test_flat_stream_collapse_keeps_other_gradient():
    # Regression for the style-collapse failure: one stream constant
    # (zero variance) must yield finite output identical to using the live
    # stream alone -- the dead stream contributes nothing but breaks nothing.
    adv = _make_adv(seed=3)
    adv[..., 1] = 0.7  # flat aux stream

    out, info = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)
    ref, _ = compute_multi_critic_adv(adv, [2.0, 0.0], _full_mask(), 4.0)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=1e-4)
    assert info["adv1_std"] == pytest.approx(0.0, abs=1e-6)


def test_weight_ratio_controls_stream_influence():
    # With independent unit streams and weights [2, 1], the combined
    # (pre-clip) advantage must covary with the goal stream twice as much as
    # with the aux stream.
    torch.manual_seed(4)
    adv = torch.randn(500, 64, 2)
    out, _ = compute_multi_critic_adv(adv, [2.0, 1.0], torch.ones(500 * 64, dtype=torch.bool), 100.0)

    flat = out.flatten()
    g = adv[..., 0].flatten()
    a = adv[..., 1].flatten()
    cov_g = torch.mean((flat - flat.mean()) * (g - g.mean()))
    cov_a = torch.mean((flat - flat.mean()) * (a - a.mean()))

    assert cov_g / cov_a == pytest.approx(2.0, rel=0.05)


def test_mask_excludes_samples_from_stats():
    # Stats must come only from masked samples: a huge outlier in an
    # unmasked slot cannot change the output at masked positions.
    adv = _make_adv(seed=5)
    mask = _full_mask()
    mask[0] = False

    out_clean, _ = compute_multi_critic_adv(adv, [2.0, 1.0], mask, 4.0)

    poisoned = adv.clone()
    poisoned.view(-1, 2)[0] = torch.tensor([1e6, -1e6])
    out_poisoned, _ = compute_multi_critic_adv(poisoned, [2.0, 1.0], mask, 4.0)

    flat_clean = out_clean.flatten()[mask]
    flat_poisoned = out_poisoned.flatten()[mask]
    assert torch.allclose(flat_clean, flat_poisoned, atol=1e-5)


def test_clip_bounds_output():
    adv = _make_adv(seed=6)
    adv.view(-1, 2)[3] = torch.tensor([1e4, 1e4])  # in-mask outlier

    out, _ = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)
    assert out.abs().max() <= 4.0 + 1e-6


def test_deterministic():
    adv = _make_adv(seed=7)
    out_a, _ = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)
    out_b, _ = compute_multi_critic_adv(adv, [2.0, 1.0], _full_mask(), 4.0)
    assert torch.equal(out_a, out_b)


def test_wrong_weight_count_raises():
    adv = _make_adv(seed=8)
    with pytest.raises(AssertionError):
        compute_multi_critic_adv(adv, [1.0], _full_mask(), 4.0)
