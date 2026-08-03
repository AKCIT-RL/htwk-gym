import math
import sys
from pathlib import Path

import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from learning.wgan_util import (
    compute_interp_grad_penalty,
    compute_wamp_disc_rewards,
    compute_wasserstein_disc_loss,
    normalize_scores,
    update_score_stats_ema,
)


def test_disc_loss_prefers_separated_scores():
    # Perfectly separated (demo high, agent low) must score lower loss than collapsed.
    demo = torch.full((64,), 10.0)
    agent = torch.full((64,), -10.0)
    loss_sep, info = compute_wasserstein_disc_loss(demo, agent, 0.4)

    collapsed = torch.zeros(64)
    loss_col, _ = compute_wasserstein_disc_loss(collapsed, collapsed, 0.4)

    assert loss_sep < loss_col
    expected = -2.0 * math.tanh(0.4 * 10.0)  # exact: -tanh(4) - tanh(4)
    assert loss_sep == pytest.approx(expected, abs=1e-6)
    assert loss_col == pytest.approx(0.0, abs=1e-6)
    assert info["disc_w_gap"] == pytest.approx(-expected, abs=1e-6)


def test_disc_loss_bounded_and_finite():
    torch.manual_seed(0)
    demo = torch.randn(128) * 1e4
    agent = torch.randn(128) * 1e4
    loss, info = compute_wasserstein_disc_loss(demo, agent, 0.4)
    assert torch.isfinite(loss)
    assert -2.0 <= loss.item() <= 2.0
    for v in info.values():
        assert torch.isfinite(v)


def test_disc_loss_gradient_alive_when_separated():
    # Unlike a saturated BCE classifier, the agent-side gradient must not vanish
    # for moderately separated scores.
    agent = torch.full((32,), -1.0, requires_grad=True)
    demo = torch.full((32,), 1.0)
    loss, _ = compute_wasserstein_disc_loss(demo, agent, 0.4)
    loss.backward()
    assert torch.all(agent.grad.abs() > 1e-3)


def test_grad_penalty_zero_for_unit_linear_critic():
    # D(x) = w.x with ||w|| = 1 has grad norm exactly 1 -> penalty 0.
    torch.manual_seed(1)
    dim = 16
    w = torch.randn(dim)
    w = w / torch.linalg.norm(w)

    def disc_fn(x):
        return (x @ w).unsqueeze(-1)

    demo = torch.randn(32, dim)
    agent = torch.randn(32, dim)
    penalty, grad_norm = compute_interp_grad_penalty(disc_fn, demo, agent)
    assert penalty.item() == pytest.approx(0.0, abs=1e-10)
    assert grad_norm.item() == pytest.approx(1.0, abs=1e-6)


def test_grad_penalty_matches_constant_grad_norm():
    # D(x) = k * w.x with ||w|| = 1 -> grad norm k -> penalty (k-1)^2.
    torch.manual_seed(2)
    dim = 8
    k = 3.0
    w = torch.randn(dim)
    w = w / torch.linalg.norm(w)

    def disc_fn(x):
        return k * (x @ w)

    demo = torch.randn(16, dim)
    agent = torch.randn(16, dim)
    penalty, grad_norm = compute_interp_grad_penalty(disc_fn, demo, agent)
    assert penalty.item() == pytest.approx((k - 1.0) ** 2, rel=1e-5)
    assert grad_norm.item() == pytest.approx(k, rel=1e-5)


def test_grad_penalty_is_differentiable():
    dim = 4
    w = torch.randn(dim, requires_grad=True)

    def disc_fn(x):
        return x @ w

    demo = torch.randn(8, dim)
    agent = torch.randn(8, dim)
    penalty, _ = compute_interp_grad_penalty(disc_fn, demo, agent)
    penalty.backward()
    assert w.grad is not None
    assert torch.all(torch.isfinite(w.grad))


def test_grad_penalty_rejects_mismatched_batches():
    def disc_fn(x):
        return x.sum(dim=-1)

    with pytest.raises(AssertionError):
        compute_interp_grad_penalty(disc_fn, torch.randn(8, 4), torch.randn(4, 4))


def test_rewards_bounded_and_monotone():
    scores = torch.linspace(-100.0, 100.0, 1001)
    r = compute_wamp_disc_rewards(scores, 0.4, 1.0)
    assert torch.all(r >= 0.0)
    assert torch.all(r <= 1.0)
    diffs = r[1:] - r[:-1]
    assert torch.all(diffs >= 0.0)  # increasing in score
    assert r[0].item() == pytest.approx(0.0, abs=1e-6)
    assert r[-1].item() == pytest.approx(1.0, abs=1e-6)
    mid = compute_wamp_disc_rewards(torch.zeros(1), 0.4, 1.0)
    assert mid.item() == pytest.approx(0.5, abs=1e-6)


def test_rewards_scale():
    scores = torch.zeros(5)
    r = compute_wamp_disc_rewards(scores, 0.4, 2.0)
    assert torch.allclose(r, torch.full((5,), 1.0))


def test_score_norm_revives_reward_spread_for_pinned_scores():
    # Regression for the observed failure: agent logits drifted to ~-20 and the
    # raw tanh reward collapsed to a constant ~0. With standardization the
    # reward must keep a usable spread regardless of the absolute offset.
    torch.manual_seed(0)
    scores = -20.0 + 0.5 * torch.randn(4096)

    raw_r = compute_wamp_disc_rewards(scores, 0.4, 1.0)
    assert torch.std(raw_r).item() < 1e-5  # dead signal without normalization

    mean, var = update_score_stats_ema(torch.zeros(1), torch.ones(1), scores, alpha=1.0)
    z = normalize_scores(scores, mean, var)
    norm_r = compute_wamp_disc_rewards(z, 0.4, 1.0)
    assert torch.std(norm_r).item() > 0.1
    assert torch.all(norm_r >= 0.0) and torch.all(norm_r <= 1.0)


def test_score_norm_invariant_to_affine_drift():
    # Standardized rewards must be identical when scores suffer an affine drift,
    # provided the stats track the drifted distribution.
    torch.manual_seed(1)
    base = torch.randn(2048)
    drifted = -17.0 + 3.0 * base

    m0, v0 = update_score_stats_ema(torch.zeros(1), torch.ones(1), base, alpha=1.0)
    m1, v1 = update_score_stats_ema(torch.zeros(1), torch.ones(1), drifted, alpha=1.0)

    r0 = compute_wamp_disc_rewards(normalize_scores(base, m0, v0), 0.4, 1.0)
    r1 = compute_wamp_disc_rewards(normalize_scores(drifted, m1, v1), 0.4, 1.0)
    assert torch.allclose(r0, r1, atol=1e-5)


def test_score_norm_preserves_ranking():
    torch.manual_seed(2)
    scores = -20.0 + torch.randn(512)
    m, v = update_score_stats_ema(torch.zeros(1), torch.ones(1), scores, alpha=1.0)
    r = compute_wamp_disc_rewards(normalize_scores(scores, m, v), 0.4, 1.0)
    order_scores = torch.argsort(scores)
    order_rewards = torch.argsort(r)
    assert torch.equal(order_scores, order_rewards)


def test_ema_update_math_and_tracking():
    # alpha=1 adopts batch stats exactly.
    batch = torch.tensor([1.0, 3.0])
    m, v = update_score_stats_ema(torch.zeros(1), torch.ones(1), batch, alpha=1.0)
    assert m.item() == pytest.approx(2.0)
    assert v.item() == pytest.approx(1.0)  # population variance of [1, 3]

    # Partial alpha blends old and new.
    m2, v2 = update_score_stats_ema(m, v, torch.tensor([4.0, 4.0]), alpha=0.5)
    assert m2.item() == pytest.approx(0.5 * 2.0 + 0.5 * 4.0)
    assert v2.item() == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)

    # Repeated updates converge to a drifted stationary distribution.
    m3, v3 = torch.zeros(1), torch.ones(1)
    torch.manual_seed(3)
    for _ in range(200):
        m3, v3 = update_score_stats_ema(m3, v3, -20.0 + 0.5 * torch.randn(4096), alpha=0.05)
    assert m3.item() == pytest.approx(-20.0, abs=0.1)
    assert v3.item() == pytest.approx(0.25, abs=0.05)


def test_score_norm_degenerate_std_no_nan():
    # Identical scores -> zero variance; min_std clamp must avoid NaN/Inf and
    # z-clip must bound the output.
    scores = torch.full((128,), -20.0)
    m, v = update_score_stats_ema(torch.zeros(1), torch.ones(1), scores, alpha=1.0)
    z = normalize_scores(scores, m, v)
    assert torch.all(torch.isfinite(z))
    assert torch.all(z.abs() <= 4.0)
    r = compute_wamp_disc_rewards(z, 0.4, 1.0)
    assert torch.all(torch.isfinite(r))


def test_disc_update_converges_on_separable_data():
    # End-to-end sanity: a small MLP critic trained with the WGAN loss + GP
    # must open a positive Wasserstein gap on linearly separable data.
    torch.manual_seed(3)
    dim = 6
    critic = torch.nn.Sequential(
        torch.nn.Linear(dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    demo = torch.randn(256, dim) + 2.0
    agent = torch.randn(256, dim) - 2.0

    # GP 50 deliberately slows the critic (Lipschitz constraint), so give it
    # enough steps; the gap must open monotonically in trend.
    for _ in range(800):
        opt.zero_grad()
        loss, info = compute_wasserstein_disc_loss(
            critic(demo).squeeze(-1), critic(agent).squeeze(-1), 0.4)
        penalty, _ = compute_interp_grad_penalty(critic, demo, agent)
        total = loss + 50.0 * penalty
        total.backward()
        opt.step()

    _, info = compute_wasserstein_disc_loss(
        critic(demo).squeeze(-1), critic(agent).squeeze(-1), 0.4)
    assert info["disc_w_gap"].item() > 0.3

    # And rewards must rank demo-like samples above agent-like ones.
    r_demo = compute_wamp_disc_rewards(critic(demo).squeeze(-1), 0.4, 1.0)
    r_agent = compute_wamp_disc_rewards(critic(agent).squeeze(-1), 0.4, 1.0)
    assert r_demo.mean() > r_agent.mean() + 0.15
