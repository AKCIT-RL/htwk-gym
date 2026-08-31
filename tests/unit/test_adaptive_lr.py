import math
import sys
from pathlib import Path

import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from learning.adaptive_lr_util import adapt_lr, compute_approx_kl
from learning.distribution_gaussian_diag import DistributionGaussianDiag

DESIRED_KL = 0.01
LR0 = 1e-4


def _make_dist(mean_val, std, dim):
    mean = torch.full((1, dim), mean_val, dtype=torch.float64)
    logstd = torch.full((1, dim), math.log(std), dtype=torch.float64)
    return DistributionGaussianDiag(mean=mean, logstd=logstd)


def _sample_logps(old, new, n, seed):
    """Draw n samples from `old` and return (new_logp, old_logp) on them."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, old.mean.shape[-1], generator=gen, dtype=torch.float64)
    x = old.mean + old.stddev * noise
    return new.log_prob(x), old.log_prob(x)


# --------------------------------------------------------------------------
# adapt_lr: the three branches of the rsl_rl rule
# --------------------------------------------------------------------------

def test_adapt_lr_shrinks_above_upper_band():
    # kl > 2 * desired -> divide by factor
    assert adapt_lr(LR0, 0.03, DESIRED_KL) == pytest.approx(LR0 / 1.5, rel=1e-12)


def test_adapt_lr_grows_below_lower_band():
    # kl < desired / 2 -> multiply by factor
    assert adapt_lr(LR0, 0.001, DESIRED_KL) == pytest.approx(LR0 * 1.5, rel=1e-12)


def test_adapt_lr_dead_band_leaves_lr_untouched():
    # anything inside [desired/2, 2*desired] must not move the lr at all
    for kl in [DESIRED_KL / 2.0, 0.007, DESIRED_KL, 0.015, 2.0 * DESIRED_KL]:
        assert adapt_lr(LR0, kl, DESIRED_KL) == LR0


def test_adapt_lr_boundaries_are_not_strict():
    # exactly at the edges the rule is a no-op; only strictly outside acts
    assert adapt_lr(LR0, 2.0 * DESIRED_KL, DESIRED_KL) == LR0
    assert adapt_lr(LR0, DESIRED_KL / 2.0, DESIRED_KL) == LR0
    assert adapt_lr(LR0, 2.0 * DESIRED_KL + 1e-9, DESIRED_KL) < LR0
    assert adapt_lr(LR0, DESIRED_KL / 2.0 - 1e-9, DESIRED_KL) > LR0


# --------------------------------------------------------------------------
# adapt_lr: clamps hold under sustained pressure
# --------------------------------------------------------------------------

def test_adapt_lr_clamps_at_lr_min_under_sustained_high_kl():
    lr = LR0
    for _ in range(200):
        lr = adapt_lr(lr, 1.0, DESIRED_KL, lr_min=1e-5, lr_max=1e-2)
    assert lr == pytest.approx(1e-5, rel=1e-12)


def test_adapt_lr_clamps_at_lr_max_under_sustained_low_kl():
    lr = LR0
    for _ in range(200):
        lr = adapt_lr(lr, 0.0, DESIRED_KL, lr_min=1e-5, lr_max=1e-2)
    assert lr == pytest.approx(1e-2, rel=1e-12)


def test_adapt_lr_recovers_from_a_clamp():
    # a clamp must be a floor, not a trap: once the KL falls back the lr climbs
    lr = 1e-5
    for _ in range(5):
        lr = adapt_lr(lr, 0.0, DESIRED_KL)
    assert lr == pytest.approx(1e-5 * 1.5 ** 5, rel=1e-12)


def test_adapt_lr_never_leaves_the_clamp_range():
    lr = LR0
    kls = [0.0, 1.0, 0.01, 0.5, 1e-6, 0.02, 0.03]
    for i in range(300):
        lr = adapt_lr(lr, kls[i % len(kls)], DESIRED_KL, lr_min=1e-5, lr_max=1e-2)
        assert 1e-5 <= lr <= 1e-2


def test_adapt_lr_input_already_outside_range_is_pulled_back():
    assert adapt_lr(1.0, DESIRED_KL, DESIRED_KL, lr_min=1e-5, lr_max=1e-2) == 1e-2
    assert adapt_lr(1e-9, DESIRED_KL, DESIRED_KL, lr_min=1e-5, lr_max=1e-2) == 1e-5


def test_adapt_lr_is_deterministic():
    a = [adapt_lr(LR0, kl, DESIRED_KL) for kl in [0.03, 0.001, 0.01]]
    b = [adapt_lr(LR0, kl, DESIRED_KL) for kl in [0.03, 0.001, 0.01]]
    assert a == b


# --------------------------------------------------------------------------
# compute_approx_kl
# --------------------------------------------------------------------------

def test_approx_kl_is_zero_for_identical_logps():
    torch.manual_seed(0)
    logp = torch.randn(1024, dtype=torch.float64)
    assert compute_approx_kl(logp, logp).item() == pytest.approx(0.0, abs=1e-12)


def test_approx_kl_is_non_negative_on_arbitrary_logps():
    # k3 is non-negative per sample by construction: (r-1) - log r >= 0 for r > 0.
    # This is the property E[-log r] does not have, and the reason for using k3.
    torch.manual_seed(0)
    for _ in range(20):
        a_logp = torch.randn(4096, dtype=torch.float64) * 3.0
        old_a_logp = torch.randn(4096, dtype=torch.float64) * 3.0
        assert compute_approx_kl(a_logp, old_a_logp).item() >= 0.0


def test_approx_kl_matches_closed_form_at_the_operating_point():
    # KL ~= 0.01 with action_std 0.05, i.e. the regime the controller targets.
    dim = 8
    std = 0.05
    delta = math.sqrt(2.0 * DESIRED_KL * std * std / dim)

    old = _make_dist(0.0, std, dim)
    new = _make_dist(delta, std, dim)

    exact = old.kl(new).item()
    assert exact == pytest.approx(DESIRED_KL, rel=1e-9)

    a_logp, old_a_logp = _sample_logps(old, new, n=400000, seed=0)
    est = compute_approx_kl(a_logp, old_a_logp).item()
    assert est == pytest.approx(exact, rel=0.05)


def test_approx_kl_matches_closed_form_away_from_the_operating_point():
    # tracks the true KL across two orders of magnitude, not just at 0.01
    dim = 8
    std = 0.05
    for target in [0.001, 0.1]:
        delta = math.sqrt(2.0 * target * std * std / dim)
        old = _make_dist(0.0, std, dim)
        new = _make_dist(delta, std, dim)

        exact = old.kl(new).item()
        a_logp, old_a_logp = _sample_logps(old, new, n=400000, seed=1)
        est = compute_approx_kl(a_logp, old_a_logp).item()
        assert est == pytest.approx(exact, rel=0.06)


def test_approx_kl_detects_a_std_change_not_only_a_mean_shift():
    # a pure variance change has zero mean shift; the estimator must still see it
    dim = 8
    old = _make_dist(0.0, 0.05, dim)
    new = _make_dist(0.0, 0.056, dim)

    exact = old.kl(new).item()
    assert exact > 0.0

    a_logp, old_a_logp = _sample_logps(old, new, n=400000, seed=2)
    est = compute_approx_kl(a_logp, old_a_logp).item()
    assert est == pytest.approx(exact, rel=0.06)


def test_approx_kl_averaging_converges_on_the_true_value():
    # k3 is unbiased, so averaging independent estimates tightens on the exact
    # KL. This is what makes a single-minibatch estimate usable as a controller
    # input despite its variance.
    dim = 8
    std = 0.05
    delta = math.sqrt(2.0 * DESIRED_KL * std * std / dim)
    old = _make_dist(0.0, std, dim)
    new = _make_dist(delta, std, dim)
    exact = old.kl(new).item()

    ests = [compute_approx_kl(*_sample_logps(old, new, n=8192, seed=s)).item()
            for s in range(64)]
    assert sum(ests) / len(ests) == pytest.approx(exact, rel=0.05)


def test_approx_kl_is_deterministic():
    torch.manual_seed(0)
    a_logp = torch.randn(4096, dtype=torch.float64)
    old_a_logp = torch.randn(4096, dtype=torch.float64)
    first = compute_approx_kl(a_logp, old_a_logp).item()
    second = compute_approx_kl(a_logp, old_a_logp).item()
    assert first == second


# --------------------------------------------------------------------------
# the two pieces working together
# --------------------------------------------------------------------------

def test_controller_drives_lr_down_when_the_policy_moves_too_far():
    # end-to-end on the real estimator: a policy step 5x past the target KL
    # must reduce the lr.
    dim = 8
    std = 0.05
    delta = math.sqrt(2.0 * (5.0 * DESIRED_KL) * std * std / dim)
    old = _make_dist(0.0, std, dim)
    new = _make_dist(delta, std, dim)

    kl = compute_approx_kl(*_sample_logps(old, new, n=65536, seed=3)).item()
    assert adapt_lr(LR0, kl, DESIRED_KL) < LR0


def test_controller_leaves_lr_alone_when_the_policy_is_on_target():
    dim = 8
    std = 0.05
    delta = math.sqrt(2.0 * DESIRED_KL * std * std / dim)
    old = _make_dist(0.0, std, dim)
    new = _make_dist(delta, std, dim)

    kl = compute_approx_kl(*_sample_logps(old, new, n=65536, seed=4)).item()
    assert adapt_lr(LR0, kl, DESIRED_KL) == LR0
