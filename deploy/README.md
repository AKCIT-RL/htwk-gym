# Deploy on Booster Robot

This directory contains scripts and utilities for deploying trained policies on the Booster robot, including support for real-time parameter editing.

## Quickstart: Setup & Run Deployment Script

Follow these steps to set up your environment and deploy a policy on the robot:

1. **Copy the `deploy/` folder to your robot (Intel Board recommended):**
   ```sh
   $ scp -r deploy/ <username>@<robot_ip>:/<destination>/
   ```

2. **SSH into the robot and set up your environment:**
   ```sh
   $ ssh <username>@<robot_ip>
   $ cd /<destination>/deploy
   $ python3 -m venv venv
   $ source venv/bin/activate
   $ pip install -r requirements.txt
   ```
   - **Install the Booster Robotics SDK:**  
     Follow the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f) and complete the [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5CInSf) section.


3. **Prepare the robot:**
   - Power on the robot.
   - Switch robot to **PREP Mode**.
   - Place the robot in a stable standing position in an open area.

4. **Deploy the policy:**
   - **For basic walking:**
     ```sh
     $ python deploy_base_walk.py --config=Base_Walk.yaml --net=127.0.0.1
     ```
   - **For base walk extended (53-obs, transfer-learning backbone):**
     ```sh
     $ python deploy_base_walk_extended.py --config=Base_Walk_Extended.yaml --net=127.0.0.1
     ```
   - **For parameterized walking (with real-time editing):**
     ```sh
     $ python deploy_parameter_walk.py --config=Parameter_Walk.yaml --net=127.0.0.1
     $ streamlit run streamlit_observation_editor.py
     ```
     - Open your browser at `http://<robot_ip>:8501` to access the web-based control interface.
     - Use interface sliders to adjust gait parameters and commands in real time.
   - **For kicking (main policy — 48 obs):**
     ```sh
     $ python deploy_kicking.py --config=Kicking.yaml --net=127.0.0.1 \
         --ball_x 0.35 --ball_y 0.0 --kick_angle_deg 0 --target_z 0.0
     ```
     See [Kicking Deployment](#kicking-deployment) below for details.
   - **For kicking (bikinha / toe-kick — 53 obs):**
     ```sh
     $ python deploy_kicking_bikinha.py --config=Kicking_Bikinha.yaml --net=127.0.0.1 \
         --ball_x 0.35 --ball_y 0.0 --kick_angle_deg 0 --target_z 0.0
     ```
     See [Kicking Bikinha Deployment](#kicking-bikinha-deployment) below for details.

5. **Exit Safely:**
   - Press `Ctrl+C` to stop deployment scripts.
   - Switch robot back to **PREP Mode** before turning off or moving the robot.

---

### Kicking Deployment

The **Kicking** policy (48 obs) is the main kicking policy. It receives ball position and a kick target, and executes the full kicking motion without an explicit gait signal.

#### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Config file in `configs/`. Use `Kicking.yaml`. |
| `--net` | `127.0.0.1` | Network interface for SDK. Use `127.0.0.1` on the Intel Board. |
| `--ball_x` | `0.35` | Ball X position in robot frame (metres, positive = forward). |
| `--ball_y` | `0.0` | Ball Y position in robot frame (metres, positive = left). |
| `--kick_angle_deg` | `0` | Kick direction in robot frame (degrees, 0 = straight ahead). |
| `--target_z` | `0.0` | Target z height relative to ball (metres, 0 = ground kick). |

#### Setup for a quick test

1. Place the ball approximately **35 cm in front** of the robot's feet, centred.
2. Run:
   ```sh
   $ python deploy_kicking.py --config=Kicking.yaml --net=127.0.0.1
   ```
3. The robot will kick the ball straight ahead.

#### Using with vision (future)

To plug in a ball detector, override `_get_ball_position()` in `deploy_kicking.py`.

---

### Kicking Bikinha Deployment

The **Kicking Bikinha** policy (53 obs) performs a toe-kick ("biquinha") on a stationary ball. The robot walks toward the ball, aligns, and kicks it toward a target direction.

#### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Config file in `configs/`. Use `Kicking_Bikinha.yaml`. |
| `--net` | `127.0.0.1` | Network interface for SDK. Use `127.0.0.1` on the Intel Board. |
| `--ball_x` | `0.35` | Ball X position in robot frame (metres, positive = forward). |
| `--ball_y` | `0.0` | Ball Y position in robot frame (metres, positive = left). |
| `--kick_angle_deg` | `0` | Kick direction in robot frame (degrees, 0 = straight ahead). |
| `--target_z` | `0.0` | Target z height relative to ball (metres, 0 = ground kick). |

#### Setup for a quick test

1. Place the ball approximately **35 cm in front** of the robot's feet, centred (this is `--ball_x 0.35 --ball_y 0.0`).
2. Run:
   ```sh
   $ python deploy_kicking_bikinha.py --config=Kicking_Bikinha.yaml --net=127.0.0.1
   ```
3. The robot will approach the ball and kick it straight ahead.

#### Adjusting kick direction

Use `--kick_angle_deg` to aim. Positive = left, negative = right (in the robot's frame):

```sh
# Kick 15° to the left
$ python deploy_kicking_bikinha.py --config=Kicking_Bikinha.yaml --kick_angle_deg 15

# Kick 10° to the right
$ python deploy_kicking_bikinha.py --config=Kicking_Bikinha.yaml --kick_angle_deg -10
```

> **Note:** The policy was trained on angles in the ±15° range. Larger angles may work but accuracy degrades.

#### Adjusting ball position

If the ball is not directly in front, specify its position in the robot's frame:

```sh
# Ball 40 cm ahead and 10 cm to the left
$ python deploy_kicking_bikinha.py --config=Kicking_Bikinha.yaml --ball_x 0.40 --ball_y 0.10
```

> **Note:** The policy works best with ball positions in the range `dx ∈ [0.20, 0.45]`, `dy ∈ [-0.15, 0.15]`.

#### Using with vision (future)

To plug in a ball detector, override `_get_ball_position()` in `deploy_kicking_bikinha.py`. It should return `np.array([x, y])` in the robot's local frame, updated each control step.

#### Known limitations

- **Left-foot bias:** The policy almost always kicks with the left foot.
- **Yaw sensitivity:** If the robot starts misaligned (>15° from the ball), hit accuracy drops significantly.
- **Target z range:** The policy was trained mostly with `target_z ≈ 0` (ground kicks). Non-zero values are accepted via `--target_z` but accuracy is untested.

---

### Notes

- **Configuration files:**  
  All config files are in `configs/` (`Base_Walk.yaml`, `Base_Walk_Extended.yaml`, `Parameter_Walk.yaml`, `Kicking.yaml`, `Kicking_Bikinha.yaml`). Each contains model paths, control gains, normalization, and limits.

- **Observation layouts:**

  | Policy | Script | Obs | Model |
  |--------|--------|-----|-------|
  | Base Walk | `deploy_base_walk.py` | 47 | `models/base_walk.pt` |
  | Base Walk Extended | `deploy_base_walk_extended.py` | 53 | `models/base_walk_extended.pt` |
  | Parameter Walk | `deploy_parameter_walk.py` | 44 | `models/parameter_walk.pt` |
  | Kicking | `deploy_kicking.py` | 48 | `models/kicking.pt` |
  | Kicking Bikinha | `deploy_kicking_bikinha.py` | 53 | `models/kicking_bikinha.pt` |

- **Real-Time Observation Controls:**  
  The Streamlit interface lets you adjust gait frequency, foot yaw, body pitch/roll, feet offset, and walk commands on the fly (requires `deploy_parameter_walk.py`).

- **Network interface (`--net`):**  
  Use `127.0.0.1` if running on the Intel Board. Otherwise, specify the proper FastDDS/network address if deploying remotely or in simulation.

- **SDK & Policy Troubles:**  
  Ensure the Booster SDK is installed correctly and that model files exist in `models/`. For Streamlit issues, make sure `live_observation_values.json` is being created.

- **Robot Safety:**  
  Always enter/exit PREP Mode carefully and check surroundings before starting motion.

---
