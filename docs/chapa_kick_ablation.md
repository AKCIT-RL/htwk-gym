# Chapa Kick Ablation Notes

Working notes for the *Kicking Movement Chapa* (instep kick) ablation,
including everything you need to continue the work on another machine.

- **Branch:** `kicking_movement`
- **Task:** `T1/Kicking_Movement_Chapa`
- **Env class:** `KickingMovementChapa` (extends `KickingMovementBica`)
- **Best run shipped:** `logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/`
- **Best checkpoint:** `nn/model_4000.pth`
- **JIT export:** [`deploy/models/kicking_chapa.pt`](../deploy/models/kicking_chapa.pt)
- **Deploy config snapshot:** [`deploy/configs/Kicking_Chapa.yaml`](../deploy/configs/Kicking_Chapa.yaml)

---

## 1. What we set out to do

The previously-shipped *Kicking Bikinha* (toe-kick) policy hits the ball
with the foot pitched ~53° down — contact is on the toe tip. We wanted
a separate policy that actually kicks **de chapa** (instep / top of the
foot, foot held horizontal at impact), without losing too much of the
existing accuracy.

Constraint: only **one** mechanism change vs. the toe-kick baseline,
so we could attribute any effect (good or bad) to that one knob.

## 2. Why earlier attempts (V1–V4) failed

V1–V4 all added standalone chapa-shaping rewards on top of the bica
reward set (pose-error gaussians, impact pulses on pitch + ball-x-local,
weighting of approach reward). All four converged to the **same**
toe-kick attractor:

| variant | foot pitch @ impact | flat_pose_rate | instep_contact_rate | hit rate |
|---------|---------------------:|---------------:|--------------------:|---------:|
| V1      | ~0.92 rad (~53°)    | 3.4 %          | 1.6 %               | 70.8 %   |
| V2      | ~0.92 rad           | ~3 %           | ~2 %                | 70.8 %   |
| V3      | ~0.90 rad           | ~4 %           | ~2 %                | 68 %     |
| V4      | ~0.90 rad           | ~4 %           | ~2 %                | 67 %     |

Diagnosis: the dominant reward term in the bica budget is
`ball_velocity_target_direction` (~30 pts/s, sustained ~2 s after
impact). It pays the same whether the ball was launched with the toe
or with the instep. Standalone pose rewards we added were dwarfed by
that signal, so the policy maximised whip speed (= toe whip) and
ignored the pose-shaping bonuses.

## 3. The single change in V5

V5 adds **exactly one** thing on top of V2's config: a multiplicative
gate on the dominant post-kick reward, controlled by the new config
key `chapa_velocity_pose_weight`.

In code ([`envs/T1/kicking_movement_chapa.py`](../envs/T1/kicking_movement_chapa.py)):

```python
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
```

In config ([`envs/T1/Kicking_Movement_Chapa.yaml`](../envs/T1/Kicking_Movement_Chapa.yaml)):

```yaml
chapa_pose_target_pitch: 0.0          # rad, target = foot horizontal
chapa_impact_pitch_sigma: 0.25        # rad gaussian width
chapa_velocity_pose_weight: 1.0       # 0.0 = ungated baseline
```

Mechanically: at impact, `self.kick_foot_pitch_at_impact` is snapshotted
(parent class). The post-kick reward is then multiplied by
`exp(-pitch²/σ²)`. With `weight=1.0` and pitch ≈ 1 rad (toe-down), the
pose factor is ~0.02 — the policy loses 98 % of the dominant reward
unless it brings the foot flat. With pitch ≈ 0, factor ≈ 1, so the
policy keeps everything it had.

`weight=0.0` makes the override a no-op (returns `super()` as-is), so
the override is fully opt-in and the toe-kick baseline is exactly
reproducible.

## 4. V5 results

### 4.1 Training-time wandb metrics (`run-20260519_163927-8exjm0p0`)

Read straight from
`logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/wandb/run-*/files/wandb-summary.json`:

| metric                              | toe-kick baseline | V5 (chapa)         |
|-------------------------------------|------------------:|-------------------:|
| `chapa/foot_pitch_at_impact_abs_mean` | 0.92 rad (53°)  | **0.058 rad (3.3°)** |
| `chapa/flat_pose_rate_0.10rad`      |             3.4 %|         **82.4 %** |
| `chapa/instep_contact_rate`         |             1.6 %|             9.6 %  |
| `chapa/contact_x_local_abs_mean`    |              n/a |          0.190 m   |
| `chapa/foot_roll_at_impact_abs_mean`|              n/a |          0.090 rad |
| `episode/chapa_impact_pose`         |              n/a |          0.0       |
| `episode/ball_velocity_target_direction` |  ~30 (V2)   |          6.77      |
| `kick/success_rate_20deg`           |              ~95 %|          95.2 %   |
| `kick/angular_error_deg_mean`       |             ~13°  |         10.86°    |

Read: the **pose** target was hit (pitch ~3°, flat_pose 82 %), the
**accuracy** held up (95 % success, 11° mean angular error), but the
**ball speed** collapsed (`ball_velocity_target_direction` dropped from
~30 to ~7), and the **contact x-local** is ~19 cm — i.e. the ball is
contacted ~19 cm in front of the foot frame origin. With a foot box of
~20 cm, that is **past the toe edge**: the policy keeps the foot flat
but kicks the ball with the **leading edge** of the horizontal foot
(shovel / pá), not with the instep.

That explains why `instep_contact_rate` is only 9.6 %: instep is
defined as `|x_local| < 0.08 m` AND flat pose. Pose is fine, contact
position is not.

### 4.2 Offline eval (`eval_results/T1_Kicking_Movement_Chapa_V5/20260519_180045/`)

`evaluate_kick.py`, 6 scenarios × 60 envs (360 attempts):

| scenario     | n  | hit% |
|--------------|----|-----:|
| angles       | 60 |  73 %|
| ball_pos     | 60 |  42 %|
| robot_yaw    | 60 |  55 %|
| ball_vel     | 60 |  65 %|
| distance     | 60 |  80 %|
| disturb_push | 60 |  90 %|
| **aggregate**|360 | **67.5 %** |

- Mean angular error: 12.4°
- Mean lateral error: 0.51 m
- Mean ball speed: 6.86 m/s
- Falls: 22 / 360 (6 %)

Compared to the toe-kick baseline (V2): hit rate −3 pp, ball speed
slightly lower, no falls regression.

## 5. Observed regression at deploy

Running the JIT-exported `deploy/models/kicking_chapa.pt` in Isaac and
MuJoCo via a custom inference script, the robot visually kicks with
the **foot pointing down** (classic bica posture), not flat.

Hypotheses, in decreasing probability order:

1. **Observation layout mismatch in the custom inference script.**
   The chapa policy expects a 52-dim observation vector with the
   layout documented in `KickingMovementBica._compute_observations()`
   (see [`envs/T1/kicking_movement_bica.py`](../envs/T1/kicking_movement_bica.py)
   around line 1190). Critical real-valued slots:

   ```
   0:3   gravity
   3:6   ang_vel
   6:9   commands              ← from _update_commands_toward_ball
   9:11  gait cos/sin          ← zeros for kicking task
   11:13 ball_pos_xy           ← REAL ball XY in robot frame
   13:15 target_dir [cosθ,sinθ]← REAL target dir in robot frame
   15:16 target_z relative     ← REAL (target_z - ball_z)
   16:28 dof_pos (12 leg DOFs)
   28:40 dof_vel (12 leg DOFs)
   40:52 last actions          (12)
   ```

   The existing utility [`deploy/utils/policy_shoot.py`](../deploy/utils/policy_shoot.py)
   builds a **44-dim** vector with constants at obs[6]=0.43, obs[7]=0.205
   and no ball/target slots at all. If the user's script is based on it
   (or otherwise omits the ball/target slots), the policy is effectively
   "blind" to the ball and reverts to a default leg posture — which can
   easily look like a toe-down stance.

2. **Pose-OK / contact-bad artifact, exposed at deploy.** Even with
   correct observations, the V5 policy contacts the ball ~19 cm in
   front of the foot frame. In sim with shadows and viewer angle, that
   may visually look like a downward toe poke. Two ways to check:
   the foot **orientation** at impact (chapa = visibly horizontal,
   bica = visibly pointed down), and the contact point on the foot
   mesh (chapa = top, bica = tip).

3. **Wrong checkpoint exported.** Unlikely — only one V5 directory has
   non-empty `nn/` (`2026-05-19-16-39-23/`); the sibling `…-16-39-05`
   is an empty failed start.

4. **Pitch sign / wrap-around bug in the snapshot.** `feet_pitch` is
   `(pitch + π) mod 2π − π`. Numerically possible that an impact with
   the foot flipped 180° around its pitch axis would log as ~0 rad
   even though the foot is upside-down. Worth verifying with a few
   raw rollouts, but the visual chapa pose in the wandb video should
   already rule this out.

## 6. Plan to resolve the deploy regression

1. **Run the raw `.pth` in-repo with the official tooling** (no custom
   inference script). This isolates the policy from the deploy plumbing:

   ```sh
   # Isaac Gym viewer (or headless if no X)
   python3 play.py --task T1/Kicking_Movement_Chapa \
     --checkpoint logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/nn/model_4000.pth \
     --num_envs 1

   # MuJoCo viewer
   python3 play_mujoco_kick.py --task T1/Kicking_Movement_Chapa \
     --checkpoint logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/nn/model_4000.pth
   ```

   - If the foot is visibly **flat** at impact → policy is fine,
     hypothesis (1). Fix the deploy script's obs assembly.
   - If the foot is visibly **toe-down** at impact → hypothesis (2):
     pose metric is real but training rollouts kick with foot flat
     while deterministic rollouts somehow don't. Train V6 with an
     additional gate (see §7).

2. **If hypothesis (1):** add a `deploy/utils/policy_kick.py` mirroring
   `policy_shoot.py` but assembling the 52-dim layout, plus a thin
   `deploy/deploy_kick.py` runner. Validate by comparing actions
   element-wise with `play_mujoco_kick.py` on identical inputs.

3. **If hypothesis (2):** add a second multiplicative gate to
   `_reward_ball_velocity_target_direction` on `contact_x_local`:

   ```python
   x = self.kick_ball_local_at_impact[:, 0]
   x_factor = torch.exp(-torch.square(x - target_x) / (sigma_x**2 + 1e-8))
   return base * pose_factor * x_factor  # both gates multiplicative
   ```

   with `target_x = 0.0` (foot centre) and `sigma_x ≈ 0.05` m. Suggested
   yaml additions:

   ```yaml
   chapa_velocity_centre_weight: 1.0
   chapa_velocity_centre_sigma:  0.05
   chapa_velocity_centre_target: 0.0
   ```

   Train V6 with the same docker recipe (4096 envs, 4000 iters,
   cuda:0). Acceptance: `chapa/foot_pitch_at_impact_abs_mean < 0.1 rad`
   **and** `chapa/instep_contact_rate > 0.5` while keeping
   `kick/success_rate_20deg > 0.6`.

## 7. How to continue this ablation on another machine

You only need a clone of this branch — all checkpoints and eval
artifacts for V5 are committed under `logs/…/_V5/` and
`eval_results/…_V5/` (force-added past `.gitignore`).

```sh
git clone <repo> && cd htwk-gym && git checkout kicking_movement

# Build the container (same image used for the original V5 training)
docker build -t htwk-gym .

# Quick sanity: reproduce the offline eval on the shipped checkpoint
docker run --rm --gpus all --network host \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/eval_results:/app/eval_results \
  htwk-gym \
  python3 evaluate_kick.py \
    --task T1/Kicking_Movement_Chapa \
    --checkpoint logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/nn/model_4000.pth \
    --scenarios all --num_envs 60 --headless True \
    --sim_device cuda:0 --rl_device cuda:0

# Resume / fine-tune from any V5 checkpoint
docker run -it --rm --gpus all --network host \
  -e WANDB_API_KEY=<your_key> \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.gymtorch_cache:/root/.cache/torch_extensions \
  htwk-gym \
  python3 train.py --task T1/Kicking_Movement_Chapa \
    --checkpoint logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/nn/model_4000.pth \
    --num_envs 4096 --headless True \
    --sim_device cuda:0 --rl_device cuda:0

# Train a fresh V6 (e.g. with the additional contact-x gate from §6.3)
# 1. Edit envs/T1/kicking_movement_chapa.py to add the x_factor multiplicative gate.
# 2. Edit envs/T1/Kicking_Movement_Chapa.yaml to add the chapa_velocity_centre_* keys.
# 3. Re-export model after training:
#       python3 export_model.py --task T1/Kicking_Movement_Chapa \
#         --checkpoint logs/T1/T1/Kicking_Movement_Chapa/<timestamp>/nn/model_4000.pth
#       cp logs/.../model_4000.pt deploy/models/kicking_chapa.pt
#       cp envs/T1/Kicking_Movement_Chapa.yaml deploy/configs/Kicking_Chapa.yaml
```

## 8. Inventory of committed artifacts

```
logs/T1/T1/Kicking_Movement_Chapa_V5/2026-05-19-16-39-23/
  config.yaml                                  # training config snapshot
  nn/model_{100,200,…,4000}.pth                # 40 checkpoints × 2.2 MB
  summaries/                                   # tensorboard event files
  wandb/                                       # full wandb run (summary + binary db)

eval_results/T1_Kicking_Movement_Chapa_V5/20260519_180045/
  all_scenarios_summary.json                   # aggregate + per-scenario metrics
  {angles,ball_pos,ball_vel,distance,disturb_push,robot_yaw}/
    attempts.csv                               # per-attempt raw metrics
    summary.json                               # scenario aggregate

deploy/models/kicking_chapa.pt                 # JIT-scripted actor (model_4000.pth)
deploy/configs/Kicking_Chapa.yaml              # 52-dim obs contract for the .pt
```

## 9. Commit trail

```
2b716ea chore(ablation): add Chapa V5 training run + offline eval artifacts
7b34ba4 docs(README): document Kicking Chapa (instep) policy and eval results
c9b6295 feat(deploy): add Kicking Chapa (instep) policy
3c17895 feat(chapa): enable pose-gated velocity reward in canonical config
1680647 feat(chapa): pose-gate post-kick velocity reward for instep technique
```
