"""Deploy the Kicking policy on the real Booster T1 robot.

The policy expects ball position and kick target to be provided. Two modes:

  1. **Fixed-target** (default): ball position and kick direction are passed
     via CLI flags (``--ball_x``, ``--ball_y``, ``--kick_angle_deg``). Useful
     for quick tests — place the ball at the specified position in front of
     the robot, run the script, and the robot will approach and kick.

  2. **Vision (future)**: plug in a ball-detection module that updates
     ``ball_robot_xy`` every control step. The hook is ``_get_ball_position``
     — override it in a subclass or modify this file.

Usage:
    python deploy_kicking.py \\
        --config Kicking.yaml --net 127.0.0.1 \\
        --ball_x 0.35 --ball_y 0.0 --kick_angle_deg 0
"""

import numpy as np
import time
import yaml
import logging
import threading
import argparse
import signal
import sys
import os

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy_kicking import PolicyKicking


class KickingController:
    def __init__(self, cfg_file, ball_x, ball_y, kick_angle_deg, target_z=0.0) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.remoteControlService = RemoteControlService()
        self.policy = PolicyKicking(cfg=self.cfg)

        # Ball and target parameters
        self.ball_robot_xy = np.array([ball_x, ball_y], dtype=np.float32)
        self.kick_angle_rad = np.deg2rad(kick_angle_deg)
        self.target_z_rel = target_z

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True
        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()
            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU rpy values too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def _get_ball_position(self):
        """Return ball [x, y] in robot-local frame.

        Override this method to plug in a vision-based ball detector.
        Default implementation returns the fixed value from CLI args.
        """
        return self.ball_robot_xy

    def cleanup(self) -> None:
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and self.publish_runner is not None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        self.client.ChangeMode(RobotMode.kCustom)

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")
        print(f"[kick] ball=({self.ball_robot_xy[0]:.2f}, {self.ball_robot_xy[1]:.2f})  "
              f"angle={np.rad2deg(self.kick_angle_rad):.1f}°  target_z={self.target_z_rel:.2f}m")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.next_inference_time += self.policy.get_policy_interval()

        ball_xy = self._get_ball_position()

        self.dof_target[:] = self.policy.inference(
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            ball_robot_xy=ball_xy,
            target_angle_rad=self.kick_angle_rad,
            target_z_rel=self.target_z_rel,
        )
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]

            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                self.low_cmd.motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                self.low_cmd.motor_cmd[i].kp = 0.0

            self._send_cmd(self.low_cmd)
            time.sleep(0.001)

    def __enter__(self) -> "KickingController":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Deploy Kicking on T1 robot.")
    parser.add_argument("--config", required=True, type=str,
                        help="Config file name in configs/ (e.g. Kicking.yaml)")
    parser.add_argument("--net", type=str, default="127.0.0.1",
                        help="Network interface for SDK communication.")
    parser.add_argument("--ball_x", type=float, default=0.35,
                        help="Ball X position in robot frame (metres, forward). Default: 0.35")
    parser.add_argument("--ball_y", type=float, default=0.0,
                        help="Ball Y position in robot frame (metres, left+). Default: 0.0")
    parser.add_argument("--kick_angle_deg", type=float, default=0.0,
                        help="Kick direction in robot frame (degrees, 0=straight). Default: 0")
    parser.add_argument("--target_z", type=float, default=0.0,
                        help="Target z height relative to ball (metres). Default: 0.0 (ground kick)")
    args = parser.parse_args()

    cfg_file = os.path.join("configs", args.config)
    print(f"Starting Kicking controller, connecting to {args.net} ...")
    print(f"  ball_x={args.ball_x:.2f} m  ball_y={args.ball_y:.2f} m  "
          f"kick_angle={args.kick_angle_deg:.1f}°  target_z={args.target_z:.2f} m")

    ChannelFactory.Instance().Init(0, args.net)

    with KickingController(cfg_file, args.ball_x, args.ball_y, args.kick_angle_deg, args.target_z) as controller:
        time.sleep(2)
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()
