#!/usr/bin/env python3
"""Day4 Gymnasium environment based on the copied Day3-v5 grasp model."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import yaml

from visual_grasp_v5 import TestFailure, VisualGrasp


class Day4Backend(VisualGrasp):
    """Reuse Day3 control without using its result writer."""

    def __init__(self, config: dict[str, Any], root: Path) -> None:
        self.episode_max_block_z = -math.inf
        super().__init__(config, root, "day4")

    def on_odom(self, msg) -> None:
        super().on_odom(msg)

        if self.block_z is not None:
            self.episode_max_block_z = max(
                self.episode_max_block_z,
                float(self.block_z),
            )


class Day4GraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Observation:
      [ex, ey, ez, q1, q2, q3, q4, q5, sin(delta_yaw), cos(delta_yaw)]

    Action:
      [delta_x, delta_y, delta_z, delta_yaw]
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()

        self.config_path = Path(config_path).expanduser().resolve()
        self.root = Path(__file__).resolve().parent

        self.config = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        )

        self.d4 = self.config["day4"]

        self._owns_rclpy = not rclpy.ok()

        if self._owns_rclpy:
            rclpy.init(args=[])

        self.backend = Day4Backend(
            self.config,
            self.root,
        )

        action_m = float(
            self.d4["action_limit_m"]
        )

        action_yaw = math.radians(
            float(self.d4["yaw_action_limit_deg"])
        )

        self.action_space = spaces.Box(
            low=np.asarray(
                [
                    -action_m,
                    -action_m,
                    -action_m,
                    -action_yaw,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    action_m,
                    action_m,
                    action_m,
                    action_yaw,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        joint_low = [
            self.backend.joint_limits[name][0] - 0.05
            for name in self.backend.arm_joints
        ]

        joint_high = [
            self.backend.joint_limits[name][1] + 0.05
            for name in self.backend.arm_joints
        ]

        self.observation_space = spaces.Box(
            low=np.asarray(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    *joint_low,
                    -1.0,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    1.0,
                    1.0,
                    1.0,
                    *joint_high,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._interfaces_checked = False
        self._episode = 0
        self._steps = 0
        self._invalid_actions = 0
        self._closed = False
        self._last_seed: int | None = None

        self.visual: dict[str, Any] | None = None
        self.selected: dict[str, Any] | None = None

        self.target_tcp = np.zeros(
            3,
            dtype=float,
        )

        self.target_yaw = 0.0

        self.tcp_offset = np.zeros(
            3,
            dtype=float,
        )

        self.block_world_start = np.zeros(
            3,
            dtype=float,
        )

        self.previous_distance = math.inf

        self.reset_metrics: dict[str, float] = {}

    @staticmethod
    def _line_angle_error(
        target: float,
        current: float,
    ) -> float:
        """
        Treat the gripper closing direction as a line.

        The returned error lies in [-pi/2, pi/2].
        """
        delta = target - current

        return 0.5 * math.atan2(
            math.sin(2.0 * delta),
            math.cos(2.0 * delta),
        )

    @staticmethod
    def _yaw_rotation(delta: float) -> np.ndarray:
        c = math.cos(delta)
        s = math.sin(delta)

        return np.asarray(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _tcp_pose(
        self,
        q: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if q is None:
            joints = self.backend.arm_q()
        else:
            joints = list(map(float, q))

        transform = self.backend.urdf.transform(
            self.config["base_frame"],
            self.config["tcp_link"],
            dict(
                zip(
                    self.backend.arm_joints,
                    joints,
                )
            ),
        )

        return (
            np.asarray(
                transform[:3, 3],
                dtype=float,
            ),
            np.asarray(
                transform[:3, :3],
                dtype=float,
            ),
        )

    @staticmethod
    def _current_yaw(rotation: np.ndarray) -> float:
        closing_axis = rotation[:, 1]

        return math.atan2(
            float(closing_axis[1]),
            float(closing_axis[0]),
        )

    def _get_obs(self) -> np.ndarray:
        q = self.backend.arm_q()

        tcp, rotation = self._tcp_pose(q)

        position_error = (
            self.target_tcp - tcp
        )

        yaw_error = self._line_angle_error(
            self.target_yaw,
            self._current_yaw(rotation),
        )

        observation = np.asarray(
            [
                *position_error.tolist(),
                *q,
                math.sin(yaw_error),
                math.cos(yaw_error),
            ],
            dtype=np.float32,
        )

        if observation.shape != (10,):
            raise RuntimeError(
                f"观测维数错误：{observation.shape}"
            )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "观测超出observation_space："
                f"{observation}"
            )

        return observation

    def _sample_block_pose(
        self,
    ) -> tuple[np.ndarray, float]:
        if 'white_area' in self.config.get('day8', {}):
            from white_area import sample_position
            side = float(self.config['day8']['nominal']['side_m'])
            yaw = math.radians(float(self.np_random.uniform(*self.d4['sampling']['yaw_deg'])))
            xy = sample_position(self.config, self.np_random, side, yaw)
            spawn = self.config['robot_spawn_world']
            return np.array([xy[0]+spawn[0], xy[1]+spawn[1],
                             self.config['table_top_world_z']+side/2]), yaw
        x_bounds = self.d4["sampling"]["base_x_m"]
        y_bounds = self.d4["sampling"]["base_y_m"]
        yaw_bounds = self.d4["sampling"]["yaw_deg"]

        base_x = float(
            self.np_random.uniform(
                float(x_bounds[0]),
                float(x_bounds[1]),
            )
        )

        base_y = float(
            self.np_random.uniform(
                float(y_bounds[0]),
                float(y_bounds[1]),
            )
        )

        yaw = math.radians(
            float(
                self.np_random.uniform(
                    float(yaw_bounds[0]),
                    float(yaw_bounds[1]),
                )
            )
        )

        robot_spawn = np.asarray(
            self.config["robot_spawn_world"],
            dtype=float,
        )

        block_world = np.asarray(
            [
                robot_spawn[0] + base_x,
                robot_spawn[1] + base_y,
                float(
                    self.config[
                        "block_reset_world"
                    ][2]
                ),
            ],
            dtype=float,
        )

        return block_world, yaw

    def _set_block_pose(
        self,
        xyz: np.ndarray,
        yaw: float,
    ) -> None:
        quaternion_z = math.sin(
            yaw / 2.0
        )

        quaternion_w = math.cos(
            yaw / 2.0
        )

        request = (
            f'name: "{self.config["block_name"]}", '
            f'position: {{'
            f'x: {xyz[0]:.9f}, '
            f'y: {xyz[1]:.9f}, '
            f'z: {xyz[2]:.9f}'
            f'}}, '
            f'orientation: {{'
            f'x: 0.0, '
            f'y: 0.0, '
            f'z: {quaternion_z:.9f}, '
            f'w: {quaternion_w:.9f}'
            f'}}'
        )

        process = subprocess.run(
            [
                "ign",
                "service",
                "-s",
                (
                    f'/world/'
                    f'{self.config["world_name"]}'
                    f'/set_pose'
                ),
                "--reqtype",
                "ignition.msgs.Pose",
                "--reptype",
                "ignition.msgs.Boolean",
                "--timeout",
                "3000",
                "--req",
                request,
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

        output = (
            process.stdout
            + "\n"
            + process.stderr
        ).strip().lower()

        service_ok = (
            process.returncode == 0
            and (
                "true" in output
                or "data: 1" in output
            )
        )

        if not service_ok:
            raise TestFailure(
                "RESET_FAIL",
                "ign set_pose did not return true",
            )

    def _arm_speed(self) -> float:
        values = [
            abs(
                float(
                    self.backend.joints[name][1]
                )
            )
            for name in self.backend.arm_joints
            if (
                name in self.backend.joints
                and math.isfinite(
                    float(
                        self.backend.joints[name][1]
                    )
                )
            )
        ]

        return max(
            values,
            default=math.inf,
        )

    def _wait_reset_settle(
        self,
        arm_target: list[float],
        block_target: np.ndarray,
    ) -> dict[str, float]:
        stable_start: float | None = None

        deadline = (
            time.monotonic()
            + float(
                self.config[
                    "timeouts"
                ]["reset_wall_sec"]
            )
        )

        while time.monotonic() < deadline:
            self.backend.spin_once()

            arm_speed = self._arm_speed()

            arm_error = max(
                abs(
                    float(
                        self.backend.joints[name][0]
                    )
                    - float(
                        arm_target[index]
                    )
                )
                for index, name
                in enumerate(
                    self.backend.arm_joints
                )
            )

            if self.backend.block_xyz is None:
                block_error = math.inf
            else:
                block_error = float(
                    np.linalg.norm(
                        np.asarray(
                            self.backend.block_xyz,
                            dtype=float,
                        )
                        - block_target
                    )
                )

            settled = (
                arm_error
                <= float(
                    self.config[
                        "reset"
                    ][
                        "arm_settle_joint_error_rad"
                    ]
                )
                and arm_speed
                <= float(
                    self.config[
                        "reset"
                    ][
                        "arm_settle_velocity_rad_sec"
                    ]
                )
                and block_error
                <= float(
                    self.config[
                        "reset"
                    ]["pose_tolerance_m"]
                )
                and self.backend.block_linear
                <= float(
                    self.config[
                        "reset"
                    ][
                        "linear_speed_max_m_sec"
                    ]
                )
                and self.backend.block_angular
                <= float(
                    self.config[
                        "reset"
                    ][
                        "angular_speed_max_rad_sec"
                    ]
                )
            )

            if settled:
                if stable_start is None:
                    stable_start = (
                        self.backend.sim_time
                    )

                stable_time = (
                    self.backend.sim_time
                    - stable_start
                )

                if stable_time >= float(
                    self.config[
                        "reset"
                    ][
                        "arm_settle_sim_sec"
                    ]
                ):
                    self.backend.clear_contacts()

                    return {
                        "reset_arm_max_velocity":
                            arm_speed,
                        "reset_block_linear_velocity":
                            float(
                                self.backend.block_linear
                            ),
                        "reset_block_angular_velocity":
                            float(
                                self.backend.block_angular
                            ),
                    }
            else:
                stable_start = None

        raise TestFailure(
            "RESET_FAIL",
            "robot or block did not settle",
        )

    def _clear_perception_cache(self) -> None:
        self.backend.target_msg = None
        self.backend.surface_msg = None
        self.backend.footprint_msg = None

        self.backend.detected = False
        self.backend.confidence = 0.0

        self.backend.target_received_wall = 0.0
        self.backend.surface_received_wall = 0.0
        self.backend.footprint_received_wall = 0.0
        self.backend.detected_received_wall = 0.0
        self.backend.confidence_received_wall = 0.0

    def _publish_step_target(
        self,
        q: list[float],
    ) -> dict[str, float]:
        """等待整条轨迹执行完，再用位置误差和速度判定完成。"""
        duration = max(
            1.0,
            float(self.d4["step_duration_sec"]),
        )

        minimum_execution = (
            duration
            * float(
                self.d4.get(
                    "step_min_execution_fraction",
                    1.0,
                )
            )
        )

        self.backend.spin_once()

        start_sim = float(
            self.backend.sim_time
        )

        initial_q = (
            self.backend.arm_q()
        )

        initial_error = max(
            abs(a - b)
            for a, b
            in zip(initial_q, q)
        )

        message = (
            self.backend.arm_message(
                q,
                duration,
            )
        )

        self.backend.arm_pub.publish(
            message
        )

        stable_start = None

        deadline = (
            time.monotonic()
            + float(
                self.d4[
                    "step_timeout_wall_sec"
                ]
            )
        )

        elapsed = 0.0
        error = math.inf
        speed = math.inf

        while time.monotonic() < deadline:
            self.backend.spin_once()

            elapsed = max(
                0.0,
                float(
                    self.backend.sim_time
                )
                - start_sim,
            )

            error = max(
                abs(
                    float(
                        self.backend
                        .joints[name][0]
                    )
                    - float(q[index])
                )
                for index, name
                in enumerate(
                    self.backend.arm_joints
                )
            )

            speed = self._arm_speed()

            valid = (
                elapsed
                >= minimum_execution

                and error
                <= float(
                    self.d4[
                        "step_settle_joint_error_rad"
                    ]
                )

                and speed
                <= float(
                    self.d4[
                        "step_settle_velocity_rad_sec"
                    ]
                )
            )

            if valid:
                if stable_start is None:
                    stable_start = float(
                        self.backend.sim_time
                    )

                stable_elapsed = (
                    float(
                        self.backend.sim_time
                    )
                    - stable_start
                )

                if stable_elapsed >= float(
                    self.d4[
                        "step_settle_sim_sec"
                    ]
                ):
                    print(
                        "DAY4_MOTION_DONE=1 "
                        f"ELAPSED_SIM={elapsed:.3f} "
                        f"INITIAL_ERR={initial_error:.5f} "
                        f"FINAL_ERR={error:.5f} "
                        f"ARM_V={speed:.5f}",
                        flush=True,
                    )

                    return {
                        "step_elapsed_sim_sec":
                            elapsed,
                        "step_initial_joint_error_rad":
                            initial_error,
                        "step_final_joint_error_rad":
                            error,
                        "step_arm_max_velocity":
                            speed,
                    }

            else:
                stable_start = None

        raise TestFailure(
            "DAY4_STEP_TIMEOUT",
            (
                f"elapsed_sim={elapsed:.3f}, "
                f"required_sim="
                f"{minimum_execution:.3f}, "
                f"joint_error={error:.6f}, "
                f"arm_velocity={speed:.6f}"
            ),
        )

    def _reset_once(
        self,
        fixed_center: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.backend.current_trial_id = (
            self._episode
        )

        self.backend.open_gripper()
        print(
            "DAY4_RESET_GRIPPER=OPEN "
            f"ANGLE_RAD={self.backend.r_position():.6f} "
            f"EFFORT_NM={self.backend.current_gripper_effort:.6f}",
            flush=True,
        )

        initial = list(
            map(
                float,
                self.config["initial_pose"],
            )
        )

        self.backend.move_arm(
            initial,
            float(
                self.config[
                    "motion"
                ]["initial_duration_sec"]
            ),
            "RESET_FAIL",
        )
        self.backend.open_gripper()
        print(
            "DAY4_RESET_GRIPPER=AFTER_INITIAL_MOVE "
            f"ANGLE_RAD={self.backend.r_position():.6f} "
            f"EFFORT_NM={self.backend.current_gripper_effort:.6f}",
            flush=True,
        )

        if fixed_center:
            block_world = np.asarray(self.config["block_reset_world"], dtype=float)
            block_yaw = 0.0
        else:
            block_world, block_yaw = self._sample_block_pose()

        self.backend.closing_started = False

        self._set_block_pose(
            block_world,
            block_yaw,
        )

        self._wait_reset_settle(
            initial,
            block_world,
        )

        self._clear_perception_cache()

        self.visual = (
            self.backend.collect_visual()
        )

        self.selected = (
            self.backend.solve_candidates(
                self.visual
            )
        )

        self.target_tcp = np.asarray(
            self.selected["descent_ee_xyz"],
            dtype=float,
        )

        self.target_yaw = math.radians(
            float(
                self.selected["offset_deg"]
            )
        )

        self.tcp_offset = np.asarray(
            self.selected["tcp_offset_m"],
            dtype=float,
        )

        self.block_world_start = (
            block_world.copy()
        )

        pregrasp, descent, _ = (
            self.backend.geometry_targets(
                list(
                    map(
                        float,
                        self.visual["surface"],
                    )
                )
            )
        )

        self.backend.move_arm(
            list(
                map(
                    float,
                    self.selected[
                        "pregrasp_joints"
                    ],
                )
            ),
            float(
                self.config[
                    "motion"
                ][
                    "pregrasp_duration_sec"
                ]
            ),
            "RESET_FAIL",
        )

        clearance = min(
            float(
                self.d4[
                    "start_clearance_m"
                ]
            ),
            float(
                self.config[
                    "motion"
                ][
                    "pregrasp_clearance_m"
                ]
            ),
        )

        ratio = (
            clearance
            / max(
                float(
                    self.config[
                        "motion"
                    ][
                        "pregrasp_clearance_m"
                    ]
                ),
                1e-9,
            )
        )

        start_center = (
            np.asarray(
                descent,
                dtype=float,
            )
            + ratio
            * (
                np.asarray(
                    pregrasp,
                    dtype=float,
                )
                - np.asarray(
                    descent,
                    dtype=float,
                )
            )
        ).tolist()

        start_path, _ = (
            self.backend.contour_cartesian_path(
                pregrasp,
                start_center,
                self.target_yaw,
                self.tcp_offset.tolist(),
                list(
                    map(
                        float,
                        self.selected[
                            "pregrasp_joints"
                        ],
                    )
                ),
                "RESET_FAIL",
            )
        )

        self.backend.move_arm_path(
            start_path,
            float(
                self.d4[
                    "start_duration_sec"
                ]
            ),
            "RESET_FAIL",
        )

        start_q = list(
            map(
                float,
                start_path[-1],
            )
        )

        self.reset_metrics = (
            self._wait_reset_settle(
                start_q,
                block_world,
            )
        )

        if self.backend.block_z is None:
            self.backend.episode_max_block_z = (
                float(block_world[2])
            )
        else:
            self.backend.episode_max_block_z = (
                float(self.backend.block_z)
            )

        self._steps = 0
        self._invalid_actions = 0

        observation = self._get_obs()

        self.previous_distance = float(
            np.linalg.norm(
                observation[:3]
            )
        )

        info: dict[str, Any] = {
            **self.reset_metrics,
            "reset_ok": True,
            "block_world_xyz":
                block_world.tolist(),
            "target_tcp_xyz":
                self.target_tcp.tolist(),
            "seed": self._last_seed,
        }

        return observation, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._last_seed = seed
        self._episode += 1

        if not self._interfaces_checked:
            self.backend.check_interfaces()
            self._interfaces_checked = True

        attempts = int(
            self.d4["reset_attempts"]
        )
        fixed_center = bool((options or {}).get("fixed_center", False))
        last_error: TestFailure | None = None

        for attempt in range(
            1,
            attempts + 1,
        ):
            try:
                return self._reset_once(fixed_center=fixed_center)

            except TestFailure as exception:
                last_error = exception
                if attempt >= attempts:
                    raise TestFailure(
                        "RESET_FAIL",
                        "Day4 reset failed after "
                        f"{attempts} attempts: {exception}",
                    ) from exception

                print(
                    f"DAY4_RESET_RETRY={attempt} "
                    f"GRIPPER_ANGLE_RAD={self.backend.r_position():.6f} "
                    f"GRIPPER_EFFORT_NM={self.backend.current_gripper_effort:.6f} "
                    f"REASON={exception}",
                    flush=True,
                )

                reset_ok, reset_detail = self.backend.safe_recover()
                if not reset_ok:
                    print(f"DAY4_RESET_RECOVERY={reset_detail}", flush=True)

        raise RuntimeError(f"unreachable reset state: {last_error}")

    def _solve_increment(
        self,
        action: np.ndarray,
    ) -> tuple[list[float] | None, bool]:
        current_q = self.backend.arm_q()

        current_tcp, current_rotation = (
            self._tcp_pose(current_q)
        )

        current_yaw = self._current_yaw(
            current_rotation
        )

        desired_tcp = (
            current_tcp
            + np.asarray(
                action[:3],
                dtype=float,
            )
        )

        desired_yaw = (
            current_yaw
            + float(action[3])
        )

        guessed_rotation = (
            self._yaw_rotation(
                float(action[3])
            )
            @ current_rotation
        )

        desired_center = (
            desired_tcp
            + guessed_rotation
            @ self.tcp_offset
        )

        allow_contact = (
            self.previous_distance
            <= float(
                self.d4[
                    "position_trigger_m"
                ]
            )
            + 0.015
        )

        try:
            q, _ = (
                self.backend.solve_contour_ik(
                    desired_center.tolist(),
                    desired_yaw,
                    self.tcp_offset.tolist(),
                    [current_q],
                    "DAY4_STEP_IK",
                    allow_contact,
                )
            )

            solved_tcp, _ = (
                self._tcp_pose(q)
            )

            corrected_center = (
                desired_center
                + (
                    desired_tcp
                    - solved_tcp
                )
            )

            q, _ = (
                self.backend.solve_contour_ik(
                    corrected_center.tolist(),
                    desired_yaw,
                    self.tcp_offset.tolist(),
                    [q],
                    "DAY4_STEP_IK",
                    allow_contact,
                )
            )

        except TestFailure:
            return None, False

        maximum_joint_change = max(
            abs(a - b)
            for a, b
            in zip(current_q, q)
        )

        if maximum_joint_change > float(
            self.config[
                "motion"
            ]["maximum_joint_step_rad"]
        ):
            return None, False

        try:
            segment = (
                self.backend
                .interpolate_joint_segment(
                    current_q,
                    q,
                    float(
                        self.config[
                            "motion"
                        ][
                            "state_validity_joint_step_rad"
                        ]
                    ),
                )
            )

            for state in segment:
                valid, _ = (
                    self.backend.state_is_valid(
                        state,
                        allow_grasp_contacts=
                            allow_contact,
                    )
                )

                if not valid:
                    return None, True

        except TestFailure:
            return None, False

        return q, False

    def _block_displaced(self) -> bool:
        if self.backend.block_xyz is None:
            return False

        current = np.asarray(
            self.backend.block_xyz,
            dtype=float,
        )

        xy_displacement = float(
            np.linalg.norm(
                current[:2]
                - self.block_world_start[:2]
            )
        )

        z_displacement = abs(
            float(
                current[2]
                - self.block_world_start[2]
            )
        )

        return (
            xy_displacement
            > float(
                self.config[
                    "verification"
                ][
                    "pregrip_xy_disturbance_m"
                ]
            )
            or z_displacement
            > float(
                self.config[
                    "verification"
                ][
                    "pregrip_z_disturbance_m"
                ]
            )
        )

    @staticmethod
    def _grasp_result() -> dict[str, Any]:
        return {
            "left_contact_seen": False,
            "right_contact_seen": False,
            "dual_contact": False,
            "stall_detected": False,
            "stall_peak_stable_sim_sec":
                0.0,
            "maximum_penetration_m":
                0.0,
            "stable_hold": False,
            "max_relative_slip_m": 0.0,
        }

    def _attempt_grasp(
        self,
    ) -> tuple[bool, bool, bool, bool, float]:
        if (
            self.visual is None
            or self.selected is None
        ):
            raise RuntimeError(
                "grasp data missing"
            )

        result = self._grasp_result()

        if self.backend.block_z is None:
            block_z_before = float(
                self.block_world_start[2]
            )
        else:
            block_z_before = float(
                self.backend.block_z
            )

        self.backend.episode_max_block_z = (
            block_z_before
        )

        success = False
        dropped = False
        collision = False
        stable_hold = False
        max_relative_slip = 0.0

        try:
            print(
                "DAY4_GRASP_ATTEMPT=1",
                flush=True,
            )

            _, hold_target = (
                self.backend
                .dual_contact_search(
                    result,
                    self.visual,
                    self.selected,
                )
            )

            self.backend.wait_stall(
                hold_target,
                result,
            )

            current_q = (
                self.backend.arm_q()
            )

            current_center, _ = (
                self.backend.urdf_grasp_pose(
                    current_q,
                    self.tcp_offset.tolist(),
                )
            )

            _, _, lift_center = (
                self.backend.geometry_targets(
                    list(
                        map(
                            float,
                            self.visual["surface"],
                        )
                    )
                )
            )

            lift_path, _ = (
                self.backend
                .contour_cartesian_path(
                    current_center
                    .astype(float)
                    .tolist(),
                    lift_center,
                    self.target_yaw,
                    self.tcp_offset.tolist(),
                    current_q,
                    "DAY4_LIFT",
                )
            )

            self.backend.set_phase(
                "LIFT_80_MM"
            )

            lifted_z = (
                self.backend.lift_and_verify(
                    lift_path,
                    block_z_before,
                )
            )

            _, _, _, stable_hold, max_relative_slip = self.backend.hold(
                block_z_before,
                lifted_z,
            )
            result["stable_hold"] = stable_hold
            result["max_relative_slip_m"] = max_relative_slip

            # Keep the existing gripper servo active while returning the
            # block to the table.  Only release after the backend's existing
            # height/contact criterion has accepted the placement.
            self.backend.set_phase("PLACE_DOWN")
            self.backend.place_down(
                current_q,
                block_z_before,
            )

            self.backend.set_phase("OPEN_GRIPPER")
            self.backend.open_gripper()

            self.backend.set_phase("RETREAT_PREGRASP")
            self.backend.move_arm(
                list(
                    map(
                        float,
                        self.selected[
                            "pregrasp_joints"
                        ],
                    )
                ),
                float(
                    self.config[
                        "motion"
                    ][
                        "retreat_duration_sec"
                    ]
                ),
                "RESET_FAIL",
            )

            if stable_hold:
                self.backend.set_phase("GRASP_SUCCESS")
                print(
                    f"TRIAL={self.backend.current_trial_id:03d} "
                    "DAY5_PLACE_RELEASE=PASS",
                    flush=True,
                )
                print("DAY4_GRASP_RESULT=SUCCESS", flush=True)
                success = True
            else:
                self.backend.set_phase("UNSTABLE_HOLD")
                print(
                    f"DAY4_GRASP_RESULT=UNSTABLE MAX_RELATIVE_SLIP_M={max_relative_slip:.6f}",
                    flush=True,
                )

        except TestFailure as exception:
            self.backend.stop_gripper()
            max_relative_slip = float(
                self.backend.max_relative_slip
            )
            stable_hold = False
            print(
                f"DAY4_GRASP_EXCEPTION={exception}",
                flush=True,
            )

            reason = str(
                getattr(
                    exception,
                    "reason",
                    "",
                )
            )

            penetration_collision = (
                self.backend.max_penetration
                > float(
                    self.config[
                        "gripper"
                    ][
                        "penetration_fail_m"
                    ]
                )
            )

            collision = (
                penetration_collision
                or "COLLISION" in reason
            )

            maximum_lift = (
                self.backend
                .episode_max_block_z
                - block_z_before
            )

            if self.backend.block_z is None:
                current_lift = -math.inf
            else:
                current_lift = (
                    float(
                        self.backend.block_z
                    )
                    - block_z_before
                )

            dropped = (
                not collision
                and maximum_lift >= 0.010
                and (
                    current_lift
                    < maximum_lift - 0.010
                    or current_lift
                    < float(
                        self.config[
                            "verification"
                        ][
                            "minimum_hold_height_m"
                        ]
                    )
                    or self.backend.table_contact()
                )
            )

        return (
            success,
            dropped,
            collision,
            stable_hold,
            max_relative_slip,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if (
            self.selected is None
            or self.visual is None
        ):
            raise RuntimeError(
                "必须先调用reset()"
            )

        action_array = np.asarray(
            action,
            dtype=np.float32,
        )

        if not np.isfinite(action_array).all():
            raise ValueError("action contains NaN or Inf")

        if action_array.shape != (4,):
            raise ValueError(
                "动作必须是4维，当前为："
                f"{action_array.shape}"
            )

        action_array = np.clip(
            action_array,
            self.action_space.low,
            self.action_space.high,
        )

        self._steps += 1

        tcp_before, _ = self._tcp_pose()

        requested_tcp_move_m = float(
            np.linalg.norm(
                action_array[:3]
            )
        )

        executed_tcp_move_m = 0.0

        motion_metrics = {
            "step_elapsed_sim_sec": 0.0,
            "step_initial_joint_error_rad": 0.0,
            "step_final_joint_error_rad": 0.0,
            "step_arm_max_velocity": 0.0,
        }

        success = False
        dropped = False
        collision = False
        attempted = False
        stable_hold = False
        max_relative_slip = 0.0

        q, planned_collision = (
            self._solve_increment(
                action_array
            )
        )

        if planned_collision:
            collision = True

        elif q is None:
            self._invalid_actions += 1

        else:
            try:
                motion_metrics = (
                    self._publish_step_target(q)
                )

                tcp_after, _ = (
                    self._tcp_pose()
                )

                executed_tcp_move_m = float(
                    np.linalg.norm(
                        tcp_after
                        - tcp_before
                    )
                )

                valid, _ = (
                    self.backend
                    .state_is_valid(
                        self.backend.arm_q(),
                        allow_grasp_contacts=True,
                    )
                )

                collision = (
                    not valid
                    or self._block_displaced()
                )

                if not collision:
                    self._invalid_actions = 0

            except TestFailure as exception:
                print(
                    f"DAY4_STEP_EXCEPTION={exception}",
                    flush=True,
                )

                valid, _ = (
                    self.backend
                    .state_is_valid(
                        self.backend.arm_q(),
                        allow_grasp_contacts=True,
                    )
                )

                collision = (
                    not valid
                    or self._block_displaced()
                )

                if not collision:
                    self._invalid_actions += 1

        observation = self._get_obs()

        new_distance = float(
            np.linalg.norm(
                observation[:3]
            )
        )

        yaw_error = abs(
            math.atan2(
                float(observation[8]),
                float(observation[9]),
            )
        )

        approach_reward = (
            float(
                self.d4[
                    "reward"
                ]["approach_scale"]
            )
            * (
                self.previous_distance
                - new_distance
            )
        )

        self.previous_distance = (
            new_distance
        )

        if (
            not collision
            and new_distance
            <= float(
                self.d4[
                    "position_trigger_m"
                ]
            )
            and yaw_error
            <= math.radians(
                float(
                    self.d4[
                        "yaw_trigger_deg"
                    ]
                )
            )
        ):
            attempted = True

            (
                success,
                dropped,
                grasp_collision,
                stable_hold,
                max_relative_slip,
            ) = self._attempt_grasp()

            collision = (
                collision
                or grasp_collision
            )

        if success:
            success_reward = float(
                self.d4[
                    "reward"
                ]["success"]
            )
        else:
            success_reward = 0.0

        if dropped:
            drop_penalty = float(
                self.d4[
                    "reward"
                ]["drop"]
            )
        else:
            drop_penalty = 0.0

        if collision:
            collision_penalty = float(
                self.d4[
                    "reward"
                ]["collision"]
            )
        else:
            collision_penalty = 0.0

        reward = (
            approach_reward
            + success_reward
            + drop_penalty
            + collision_penalty
        )

        terminated = bool(
            success
            or dropped
            or collision
            or attempted
        )

        truncated = bool(
            not terminated
            and (
                self._steps
                >= int(
                    self.d4["max_steps"]
                )
                or self._invalid_actions
                >= int(
                    self.d4[
                        "invalid_action_limit"
                    ]
                )
            )
        )

        if terminated:
            observation = self._get_obs()

        # Preserve task semantics while exposing the actual failure to Day9.
        info: dict[str, Any] = {
            "success": success,
            "stable_hold": stable_hold,
            "max_relative_slip_m": max_relative_slip,
            "grasp_attempted": attempted,
            "distance_m": new_distance,
            "yaw_error_rad": yaw_error,
            "step_count": self._steps,
            "dropped": dropped,
            "collision": collision,
            "invalid_actions": self._invalid_actions,
            "termination_reason": (
                "success" if success else "drop" if dropped else
                "collision" if collision else "grasp_failed" if attempted else
                "invalid_action_limit" if self._invalid_actions >= int(self.d4["invalid_action_limit"]) else
                "max_steps" if truncated else ""
            ),

            "approach_reward":
                approach_reward,

            "success_reward":
                success_reward,

            "drop_penalty":
                drop_penalty,

            "collision_penalty":
                collision_penalty,

            "requested_tcp_move_m":
                requested_tcp_move_m,

            "executed_tcp_move_m":
                executed_tcp_move_m,

            **motion_metrics,
            **self.reset_metrics,
        }

        print(
            "DAY4_STEP=1 "
            f"EPISODE={self._episode:03d} "
            f"STEP={self._steps:02d} "
            f"DIST={new_distance:.6f} "
            f"YAW_ERR_DEG="
            f"{math.degrees(yaw_error):.3f} "
            f"REQUESTED_MOVE="
            f"{requested_tcp_move_m:.6f} "
            f"EXECUTED_MOVE="
            f"{executed_tcp_move_m:.6f} "
            f"ATTEMPT={int(attempted)} "
            f"SUCCESS={int(success)} "
            f"TERMINATED={int(terminated)} "
            f"TRUNCATED={int(truncated)}",
            flush=True,
        )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            if rclpy.ok():
                self.backend.safe_recover()
        except Exception:
            pass

        try:
            self.backend.destroy_node()
        finally:
            if (
                self._owns_rclpy
                and rclpy.ok()
            ):
                rclpy.shutdown()
