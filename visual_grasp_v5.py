#!/usr/bin/env python3
"""Day3-v5 deterministic RGB-D, multi-orientation physical grasp runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Callable

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PointStamped, PolygonStamped, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from ros_gz_interfaces.msg import Contacts
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml

import numpy as np
from scipy.optimize import least_squares

from grasp_geometry_v5 import (
    URDFGeometry, grasp_rotation, matrix_to_quaternion, ordered_rectangle,
    pose_error, quaternion_to_matrix, square_circular_mean,
)


PASSIVE_JOINTS = ["l_joint", "l_in_joint", "l_out_joint", "r_in_joint", "r_out_joint"]
FAILURE_REASONS = [
    "PERCEPTION_FAIL", "PERCEPTION_UNSTABLE", "OUT_OF_WORKSPACE",
    "OUT_OF_REACHABLE_REGION", "IK_FAIL", "COLLISION_PLAN_FAIL",
    "TRAJECTORY_PLAN_FAIL", "ORIENTATION_IK_FAIL", "PREGRASP_EXEC_FAIL", "DESCENT_EXEC_FAIL",
    "NO_LEFT_CONTACT", "NO_RIGHT_CONTACT", "NO_DUAL_CONTACT", "NO_STALL",
    "CORNER_COLLISION", "NO_VALID_GRASP",
    "BLOCK_NOT_LIFTED", "BLOCK_DROPPED_DURING_LIFT",
    "BLOCK_DROPPED_DURING_HOLD", "PLACE_DOWN_FAIL", "RESET_FAIL",
    "SIMULATION_FAILURE",
]
RESULT_FIELDS = [
    "trial_id", "case_id", "grid_row", "grid_col", "repeat", "planned_base_xyz",
    "planned_world_xyz", "config_sha256", "start_wall_time", "end_wall_time", "cycle_wall_sec",
    "success", "failure_reason", "failure_detail", "visual_detected", "visual_center_x",
    "visual_center_y", "visual_center_z", "visual_std_x", "visual_std_y", "visual_std_z",
    "visual_surface_x", "visual_surface_y", "visual_surface_z", "visual_confidence",
    "visual_contour_angle_deg", "visual_contour_std_deg", "visual_footprint_m",
    "orientation_candidates", "selected_orientation_deg", "selected_quaternion_xyzw",
    "selected_tilt_deg", "selected_lean_sign", "selected_square_symmetry",
    "selected_tcp_offset_m", "expected_contact_angle", "contact_gap_m",
    "contact_geometry_valid", "contact_geometry_detail",
    "selected_fk_position_error_m", "selected_fk_orientation_error_rad", "selected_joint_motion_rad",
    "selected_pregrasp_fk_position_error_m", "selected_pregrasp_fk_orientation_error_rad",
    "selected_descent_fk_position_error_m", "selected_descent_fk_orientation_error_rad",
    "selected_lift_fk_position_error_m", "selected_lift_fk_orientation_error_rad",
    "selected_predicted_finger_lift_m", "selected_predicted_finger_horizontal_drift_m",
    "pregrasp_ee_xyz", "descent_ee_xyz", "lift_ee_xyz",
    "pregrasp_joints", "descent_joints", "lift_joints", "trajectory_waypoint_count",
    "minimum_joint_margin_rad", "minimum_collision_clearance_m", "contact_angle", "hold_target",
    "left_contact_seen", "right_contact_seen", "dual_contact", "stall_detected",
    "stall_min_error_rad", "stall_max_error_rad", "stall_final_error_rad",
    "stall_peak_stable_sim_sec", "stall_min_velocity_rad_sec", "stall_max_velocity_rad_sec",
    "maximum_penetration_m", "block_z_before", "block_z_after", "lift_m",
    "minimum_hold_z", "hold_sim_sec", "hold_wall_sec", "stable_hold",
    "max_relative_slip_m", "reset_success", "sim_restart_index",
]


class TestFailure(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class SimulationFailure(TestFailure):
    def __init__(self, detail: str) -> None:
        super().__init__("SIMULATION_FAILURE", detail)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def blank_result(trial_id: int, config_hash: str, restart: int) -> dict[str, object]:
    result: dict[str, object] = {name: None for name in RESULT_FIELDS}
    result.update({
        "trial_id": trial_id,
        "config_sha256": config_hash,
        "start_wall_time": now_text(),
        "success": False,
        "failure_reason": "",
        "failure_detail": "",
        "visual_detected": False,
        "left_contact_seen": False,
        "right_contact_seen": False,
        "dual_contact": False,
        "stall_detected": False,
        "stall_peak_stable_sim_sec": 0.0,
        "maximum_penetration_m": 0.0,
        "hold_sim_sec": 0.0,
        "hold_wall_sec": 0.0,
        "stable_hold": False,
        "max_relative_slip_m": 0.0,
        "reset_success": False,
        "sim_restart_index": restart,
    })
    return result


def rebuild_tables(results_dir: Path) -> None:
    rows = []
    for path in sorted(results_dir.glob("trial_[0-9][0-9][0-9].json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_text(results_dir / "trials.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary = results_dir / "trials.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for source in rows:
            row = {}
            for name in RESULT_FIELDS:
                value = source.get(name)
                row[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            writer.writerow(row)
    temporary.replace(results_dir / "trials.csv")


class VisualGrasp(Node):
    def __init__(self, config: dict, root: Path, config_hash: str) -> None:
        super().__init__(
            "day3_v5_visual_grasp",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.c = config
        self.root = root
        self.config_hash = config_hash
        self.topics = config["topics"]
        self.g = config["gripper"]
        self.m = config["motion"]
        self.v = config["verification"]
        self.rst = config["reset"]
        self.t = config["timeouts"]
        self.geometry = config["geometry"]
        self.orientation_config = config["orientation"]
        self.arm_joints = list(config["arm_joints"])
        self.urdf = URDFGeometry(root / self.geometry["urdf"])
        self.joint_limits = self.urdf.joint_limits(self.arm_joints)
        self.phase = "INITIALIZING"
        self.current_trial_id = 0
        self.sim_time = 0.0
        self.last_clock_wall = time.monotonic()
        self.have_clock = False
        self.joints: dict[str, tuple[float, float]] = {}
        self.joint_efforts: dict[str, float] = {}
        self.gazebo_joints: dict[str, tuple[float, float]] = {}
        self.block_xyz: list[float] | None = None
        self.block_z: float | None = None
        self.block_linear = math.inf
        self.block_angular = math.inf
        self.block_contact_names: set[str] = set()
        self.left_last_wood = -math.inf
        self.right_last_wood = -math.inf
        self.left_depth = 0.0
        self.right_depth = 0.0
        self.left_contact_samples: list[dict[str, object]] = []
        self.right_contact_samples: list[dict[str, object]] = []
        self.max_penetration = 0.0
        self.current_gripper_target = float(self.g["open_target"])
        self.current_gripper_effort = 0.0
        self.gripper_mode = "stopped"
        self.gripper_servo_active = False
        self.gripper_servo_integral = 0.0
        self.gripper_servo_last_sim = 0.0
        self.first_contact_angle: float | None = None
        self.last_gripper_publish = 0.0
        self.last_audit_sim = -math.inf
        self.hold_reference_relative_z: float | None = None
        self.max_relative_slip = 0.0
        self.gripper_audit_rows: list[dict[str, object]] = []
        self.closing_started = False
        self.target_msg: PointStamped | None = None
        self.target_received_wall = 0.0
        self.surface_msg: PointStamped | None = None
        self.surface_received_wall = 0.0
        self.footprint_msg: PolygonStamped | None = None
        self.footprint_received_wall = 0.0
        self.detected = False
        self.detected_received_wall = 0.0
        self.confidence = 0.0
        self.confidence_received_wall = 0.0
        self.status_last_wall: dict[str, float] = {}
        self.scene_initialized = False

        qos = 30
        self.create_subscription(Clock, "/clock", self.on_clock, qos)
        self.create_subscription(JointState, self.topics["joint_states"], self.on_joints, qos)
        self.create_subscription(JointState, self.topics["gazebo_joint_states"], self.on_gazebo_joints, qos)
        self.create_subscription(Odometry, self.topics["block_odometry"], self.on_odom, qos)
        self.create_subscription(Contacts, self.topics["left_contacts"], self.on_left_contacts, qos)
        self.create_subscription(Contacts, self.topics["right_contacts"], self.on_right_contacts, qos)
        self.create_subscription(Contacts, self.topics["block_contacts"], self.on_block_contacts, qos)
        self.create_subscription(PointStamped, self.topics["target"], self.on_target, 20)
        self.create_subscription(PointStamped, self.topics["surface"], self.on_surface, 20)
        self.create_subscription(PolygonStamped, self.topics["footprint"], self.on_footprint, 20)
        self.create_subscription(Bool, self.topics["detected"], self.on_detected, 20)
        self.create_subscription(Float32, self.topics["confidence"], self.on_confidence, 20)
        self.gripper_pub = self.create_publisher(Float64MultiArray, self.topics["gripper_command"], 10)
        self.arm_pub = self.create_publisher(JointTrajectory, self.topics["arm_command"], 10)
        self.ik = self.create_client(GetPositionIK, self.topics["ik_service"])
        self.fk = self.create_client(GetPositionFK, self.topics["fk_service"])
        self.state_validity = self.create_client(GetStateValidity, self.topics["state_validity_service"])
        self.apply_scene = self.create_client(ApplyPlanningScene, self.topics["planning_scene_service"])

    @staticmethod
    def normalize_joint(name: str) -> str:
        return name.rsplit("::", 1)[-1]

    def on_clock(self, msg: Clock) -> None:
        value = msg.clock.sec + msg.clock.nanosec * 1e-9
        if value > self.sim_time + 1e-9:
            self.last_clock_wall = time.monotonic()
        self.sim_time = value
        self.have_clock = True

    def on_joints(self, msg: JointState) -> None:
        for index, raw in enumerate(msg.name):
            if index >= len(msg.position):
                continue
            velocity = msg.velocity[index] if index < len(msg.velocity) else float("nan")
            name = self.normalize_joint(raw)
            self.joints[name] = (float(msg.position[index]), float(velocity))
            effort = msg.effort[index] if index < len(msg.effort) else float("nan")
            self.joint_efforts[name] = float(effort)

    def on_gazebo_joints(self, msg: JointState) -> None:
        for index, raw in enumerate(msg.name):
            if index < len(msg.position):
                velocity = msg.velocity[index] if index < len(msg.velocity) else float("nan")
                self.gazebo_joints[self.normalize_joint(raw)] = (float(msg.position[index]), float(velocity))

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.block_xyz = [float(p.x), float(p.y), float(p.z)]
        self.block_z = float(p.z)
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        self.block_linear = math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
        self.block_angular = math.sqrt(angular.x ** 2 + angular.y ** 2 + angular.z ** 2)

    @staticmethod
    def contact_values(msg: Contacts) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for contact in msg.contacts:
            depth = max((float(value) for value in contact.depths), default=0.0)
            count = max(len(contact.positions), len(contact.normals), len(contact.depths), 1)
            for index in range(count):
                position = contact.positions[min(index, len(contact.positions) - 1)] if contact.positions else None
                normal = contact.normals[min(index, len(contact.normals) - 1)] if contact.normals else None
                values.append({
                    "collision1": contact.collision1.name,
                    "collision2": contact.collision2.name,
                    "depth": depth,
                    "position_world": None if position is None else [float(position.x), float(position.y), float(position.z)],
                    "normal_world": None if normal is None else [float(normal.x), float(normal.y), float(normal.z)],
                })
        return values

    def finger_contacts(self, msg: Contacts, collision: str, side: str) -> None:
        recent: list[dict[str, object]] = []
        for value in self.contact_values(msg):
            first, second, depth = value["collision1"], value["collision2"], float(value["depth"])
            joined = f"{first} {second}"
            if "wood_block" not in joined or collision not in joined:
                continue
            recent.append(value)
            self.max_penetration = max(self.max_penetration, depth)
            if side == "left":
                self.left_last_wood = self.sim_time
                self.left_depth = depth
            else:
                self.right_last_wood = self.sim_time
                self.right_depth = depth
        if recent:
            if side == "left":
                self.left_contact_samples = recent
            else:
                self.right_contact_samples = recent

    def on_left_contacts(self, msg: Contacts) -> None:
        self.finger_contacts(msg, "l_out_link_box_collision", "left")

    def on_right_contacts(self, msg: Contacts) -> None:
        self.finger_contacts(msg, "r_out_link_box_collision", "right")

    def on_block_contacts(self, msg: Contacts) -> None:
        names: set[str] = set()
        for value in self.contact_values(msg):
            names.update((str(value["collision1"]), str(value["collision2"])))
        self.block_contact_names = names

    def on_target(self, msg: PointStamped) -> None:
        self.target_msg = msg
        self.target_received_wall = time.monotonic()

    def on_surface(self, msg: PointStamped) -> None:
        self.surface_msg = msg
        self.surface_received_wall = time.monotonic()

    def on_footprint(self, msg: PolygonStamped) -> None:
        self.footprint_msg = msg
        self.footprint_received_wall = time.monotonic()

    def on_detected(self, msg: Bool) -> None:
        self.detected = bool(msg.data)
        self.detected_received_wall = time.monotonic()

    def on_confidence(self, msg: Float32) -> None:
        self.confidence = float(msg.data)
        self.confidence_received_wall = time.monotonic()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        print(f"TRIAL={self.current_trial_id:03d} PHASE={phase}", flush=True)

    def status_print(self, key: str, message: str, period_wall_sec: float = 1.0) -> None:
        now = time.monotonic()
        if now - self.status_last_wall.get(key, -math.inf) >= period_wall_sec:
            self.status_last_wall[key] = now
            print(message, flush=True)

    def left_contact(self) -> bool:
        return self.sim_time - self.left_last_wood <= float(self.g["contact_dropout_sim_sec"])

    def right_contact(self) -> bool:
        return self.sim_time - self.right_last_wood <= float(self.g["contact_dropout_sim_sec"])

    def table_contact(self) -> bool:
        names = " ".join(self.block_contact_names)
        return "table" in names or "top_collision" in names

    def r_position(self) -> float:
        return self.joints.get("r_joint", (float("nan"), float("nan")))[0]

    def r_velocity(self) -> float:
        return self.joints.get("r_joint", (float("nan"), float("nan")))[1]

    def r_effort(self) -> float:
        return self.joint_efforts.get("r_joint", float("nan"))

    def tcp_world_z(self) -> float:
        return float(self.tcp_world_xyz()[2])

    def tcp_world_xyz(self) -> np.ndarray:
        if not all(name in self.joints for name in self.arm_joints):
            return np.full(3, float("nan"))
        positions = {name: self.joints[name][0] for name in self.arm_joints}
        transform = self.urdf.transform(self.c["base_frame"], self.c["tcp_link"], positions)
        return (
            np.asarray(self.c["robot_spawn_world"], dtype=float)
            + np.asarray(transform[:3, 3], dtype=float)
        )

    def relative_block_tcp_xyz(self) -> np.ndarray:
        tcp_xyz = self.tcp_world_xyz()
        if self.block_xyz is None or not np.all(np.isfinite(tcp_xyz)):
            return np.full(3, float("nan"))
        return np.asarray(self.block_xyz, dtype=float) - tcp_xyz

    def relative_block_tcp_z(self) -> float:
        tcp_z = self.tcp_world_z()
        if self.block_z is None or not math.isfinite(tcp_z):
            return float("nan")
        return float(self.block_z) - tcp_z

    def analytic_contact_closure(self) -> float:
        angle = self.r_position()
        if self.first_contact_angle is None or not math.isfinite(angle):
            return 0.0
        first_gap = float(self.urdf.finger_geometry(self.first_contact_angle)["gap_m"])
        current_gap = float(self.urdf.finger_geometry(angle)["gap_m"])
        return max(0.0, first_gap - current_gap)

    def record_gripper_audit(self) -> None:
        audited = {"CLOSE_SEARCH", "WAIT_DUAL_CONTACT", "WAIT_STALL", "LIFT_80_MM", "HOLD"}
        if self.phase not in audited or self.sim_time - self.last_audit_sim < 0.05:
            return
        self.last_audit_sim = self.sim_time
        actual = self.r_position()
        relative_z = self.relative_block_tcp_z()
        if self.hold_reference_relative_z is not None and math.isfinite(relative_z):
            self.max_relative_slip = max(
                self.max_relative_slip, self.hold_reference_relative_z - relative_z)
        measured = self.r_effort()
        limit = float(self.g["effort_limit_nm"])
        effort_valid = (
            math.isfinite(measured)
            and abs(measured) <= 1.5 * limit
            and (
                abs(self.current_gripper_effort) <= 1e-6
                or abs(measured) >= 0.1 * abs(self.current_gripper_effort)
            )
        )
        tcp_xyz = self.tcp_world_xyz()
        relative_xyz = self.relative_block_tcp_xyz()
        self.gripper_audit_rows.append({
            "trial_id": self.current_trial_id,
            "phase": self.phase,
            "sim_time_sec": self.sim_time,
            "gripper_target_rad": self.current_gripper_target,
            "gripper_actual_rad": actual,
            "gripper_error_rad": actual - self.current_gripper_target,
            "gripper_velocity_rad_sec": self.r_velocity(),
            "command_effort_nm": self.current_gripper_effort,
            "measured_effort_nm": measured,
            "measured_effort_valid": effort_valid,
            "left_contact": self.left_contact(),
            "right_contact": self.right_contact(),
            "block_z_m": self.block_z,
            "tcp_x_m": tcp_xyz[0],
            "tcp_y_m": tcp_xyz[1],
            "tcp_z_m": tcp_xyz[2],
            "block_relative_tcp_x_m": relative_xyz[0],
            "block_relative_tcp_y_m": relative_xyz[1],
            "block_relative_tcp_z_m": relative_xyz[2],
            "max_relative_slip_m": self.max_relative_slip,
            "analytic_contact_closure_m": self.analytic_contact_closure(),
        })

    def drain_gripper_audit(self) -> list[dict[str, object]]:
        rows, self.gripper_audit_rows = self.gripper_audit_rows, []
        return rows

    def publish_gripper(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_gripper_publish < 1.0 / float(self.g["command_rate_hz"]):
            return
        self.gripper_pub.publish(Float64MultiArray(data=[float(self.current_gripper_effort)]))
        self.last_gripper_publish = now

    def command_gripper(self, effort: float, target: float | None = None) -> None:
        self.gripper_servo_active = False
        if target is not None:
            self.current_gripper_target = max(0.0, min(1.57, float(target)))
        limit = float(self.g["effort_limit_nm"])
        self.current_gripper_effort = max(-limit, min(limit, float(effort)))
        self.publish_gripper(force=True)

    def stop_gripper(self) -> None:
        self.gripper_servo_active = False
        self.gripper_mode = "stopped"
        self.current_gripper_effort = 0.0
        self.publish_gripper(force=True)

    def maintain_open_gripper(self) -> None:
        if self.gripper_mode != "open":
            return
        angle = self.r_position()
        if not math.isfinite(angle):
            return
        if angle < self.current_gripper_target - 0.015:
            self.current_gripper_effort = float(self.g["effort_limit_nm"])
        else:
            self.current_gripper_effort = float(self.g["open_hold_effort_nm"])

    def start_gripper_servo(self, target: float) -> None:
        bounded_target = max(0.0, min(1.57, float(target)))
        if not self.gripper_servo_active or abs(bounded_target - self.current_gripper_target) > 1e-9:
            velocity = self.r_velocity()
            preload = (
                -float(self.g["velocity_proportional_gain_nm_sec_per_rad"])
                * velocity
                if math.isfinite(velocity)
                else 0.0
            )
            limit = float(self.g["hold_effort_limit_nm"])
            self.gripper_servo_integral = max(
                -limit,
                min(limit, preload - float(self.g["hold_effort_nm"])),
            )
        self.current_gripper_target = bounded_target
        self.gripper_mode = "closing"
        self.gripper_servo_active = True
        self.gripper_servo_last_sim = self.sim_time
        self.update_gripper_servo()
        self.publish_gripper(force=True)

    def update_gripper_servo(self) -> None:
        if not self.gripper_servo_active:
            return
        angle = self.r_position()
        velocity = self.r_velocity()
        if not math.isfinite(angle) or not math.isfinite(velocity):
            return
        maximum_velocity = float(self.g["compensation_velocity_rad_sec"])
        desired_velocity = max(
            -maximum_velocity,
            min(
                maximum_velocity,
                float(self.g["position_velocity_gain"])
                * (self.current_gripper_target - angle),
            ),
        )
        velocity_error = desired_velocity - velocity
        elapsed = max(0.0, min(0.1, self.sim_time - self.gripper_servo_last_sim))
        self.gripper_servo_last_sim = self.sim_time
        limit = float(self.g["hold_effort_limit_nm"])
        self.gripper_servo_integral = max(
            -limit,
            min(
                limit,
                self.gripper_servo_integral
                + float(self.g["velocity_integral_gain_nm_per_rad"])
                * velocity_error
                * elapsed,
            ),
        )
        effort = (
            self.gripper_servo_integral
            + float(self.g["velocity_proportional_gain_nm_sec_per_rad"])
            * velocity_error
        )
        self.current_gripper_effort = max(-limit, min(limit, effort))

    def check_closing_safety(self, reason: str) -> None:
        angle = self.r_position()
        closure = self.analytic_contact_closure()
        self.max_penetration = max(self.max_penetration, closure)
        if ((math.isfinite(angle) and angle <= float(self.g["close_target"]) + 1e-3)
                or closure > float(self.g["penetration_fail_m"])):
            self.stop_gripper()
            raise TestFailure(
                reason,
                f"gripper safety limit: angle={angle:.6f} target={self.current_gripper_target:.6f} "
                f"analytic_contact_closure={closure:.6f}",
            )

    def set_gripper(self, target: float) -> None:
        """Compatibility helper for the effort-mode close path."""
        self.current_gripper_target = max(0.0, min(1.57, float(target)))
        self.command_gripper(-float(self.g["search_effort_nm"]))

    def spin_once(self, timeout: float = 0.04, check_clock: bool = True) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)
        self.maintain_open_gripper()
        self.update_gripper_servo()
        self.publish_gripper()
        self.record_gripper_audit()
        if check_clock and self.have_clock and time.monotonic() - self.last_clock_wall > float(self.t["clock_stall_wall_sec"]):
            raise SimulationFailure("/clock stopped advancing")

    def wait_until(self, predicate: Callable[[], bool], timeout: float, reason: str) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.spin_once()
            if predicate():
                return
        raise TestFailure(reason)

    @staticmethod
    def control_command(arguments: list[str], timeout: float = 20.0) -> tuple[int, str]:
        process = subprocess.run(["ros2", "control", *arguments], text=True, capture_output=True,
                                 timeout=timeout, check=False)
        return process.returncode, (process.stdout + "\n" + process.stderr).strip()

    def check_interfaces(self) -> None:
        self.set_phase("CHECK_INTERFACES")
        deadline = time.monotonic() + float(self.t["interfaces_wall_sec"])
        detail = ""
        while time.monotonic() < deadline:
            for _ in range(10):
                self.spin_once(check_clock=False)
            rc1, controllers = self.control_command(["list_controllers"])
            rc2, interfaces = self.control_command(["list_hardware_interfaces"])
            if rc1 or rc2:
                detail = controllers + " " + interfaces
                continue
            active = all(
                name in controllers and "active" in controllers.split(name, 1)[1].splitlines()[0]
                for name in ("joint_state_broadcaster", "arm_controller", "gripper_controller")
            )
            command_section = interfaces.split("state interfaces", 1)[0]
            correct = "r_joint/effort" in command_section and "r_joint/position" not in command_section
            passive = any(f"{name}/" in command_section for name in PASSIVE_JOINTS)
            joints = all(name in self.joints for name in self.arm_joints + ["r_joint"])
            services = (self.ik.service_is_ready() and self.fk.service_is_ready() and
                        self.state_validity.service_is_ready() and self.apply_scene.service_is_ready())
            if active and correct and not passive and joints and self.block_z is not None and services and self.have_clock:
                return
            detail = f"active={active} r_effort_only={correct} passive_command={passive} joints={joints} odom={self.block_z is not None} ik_fk={services} clock={self.have_clock}"
        if "active=False" in detail:
            raise TestFailure("CONTROLLER_NOT_ACTIVE", detail)
        raise TestFailure("INTERFACE_NOT_READY", detail)

    def arm_q(self) -> list[float]:
        if not all(name in self.joints for name in self.arm_joints):
            raise TestFailure("INTERFACE_NOT_READY", "arm joints unavailable")
        return [self.joints[name][0] for name in self.arm_joints]

    def robot_state(self, q: list[float]) -> RobotState:
        values = {name: pair[0] for name, pair in self.joints.items() if math.isfinite(pair[0])}
        values.update(dict(zip(self.arm_joints, map(float, q))))
        state = RobotState()
        state.joint_state.name = list(values)
        state.joint_state.position = list(values.values())
        state.is_diff = False
        return state

    def wait_future(self, future, timeout: float, reason: str):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not future.done():
            self.spin_once()
        if not future.done() or future.exception() is not None or future.result() is None:
            raise TestFailure(reason, "service timeout or exception")
        return future.result()

    def fk_poses(self, q: list[float], links: list[str]) -> dict[str, dict[str, list[float]]]:
        request = GetPositionFK.Request()
        request.header.frame_id = self.c["base_frame"]
        request.fk_link_names = links
        request.robot_state = self.robot_state(q)
        response = self.wait_future(self.fk.call_async(request), float(self.m["service_timeout_wall_sec"]), "INTERFACE_NOT_READY")
        if response.error_code.val != 1 or len(response.pose_stamped) != len(links):
            raise TestFailure("INTERFACE_NOT_READY", f"FK error {response.error_code.val}")
        output = {}
        for link, stamped in zip(links, response.pose_stamped):
            p, o = stamped.pose.position, stamped.pose.orientation
            output[link] = {"xyz": [float(p.x), float(p.y), float(p.z)],
                            "quaternion_xyzw": [float(o.x), float(o.y), float(o.z), float(o.w)]}
        return output

    def solve_tcp_ik(self, xyz: list[float], orientation: list[float], seed: list[float], reason: str) -> list[float]:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.c["move_group"]
        ik.robot_state = self.robot_state(seed)
        ik.avoid_collisions = True
        ik.ik_link_name = self.c["tcp_link"]
        ik.pose_stamped.header.frame_id = self.c["base_frame"]
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = map(float, xyz)
        (ik.pose_stamped.pose.orientation.x, ik.pose_stamped.pose.orientation.y,
         ik.pose_stamped.pose.orientation.z, ik.pose_stamped.pose.orientation.w) = map(float, orientation)
        ik.timeout = Duration(seconds=float(self.m["ik_timeout_sec"])).to_msg()
        response = self.wait_future(self.ik.call_async(request), float(self.m["service_timeout_wall_sec"]), reason)
        if response.error_code.val != 1:
            raise TestFailure(reason, f"MoveIt error {response.error_code.val}")
        values = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if any(name not in values for name in self.arm_joints):
            raise TestFailure(reason, "IK result missing arm joints")
        q = [float(values[name]) for name in self.arm_joints]
        margin = float(self.m["joint_limit_margin_rad"])
        outside = [name for name, value in zip(self.arm_joints, q)
                   if not self.joint_limits[name][0] + margin <= value <= self.joint_limits[name][1] - margin]
        if outside:
            raise TestFailure(reason, f"joint limit/margin violated: {outside} q={q}")
        return q

    def state_is_valid(self, q: list[float], allow_grasp_contacts: bool = False) -> tuple[bool, str]:
        request = GetStateValidity.Request()
        request.robot_state = self.robot_state(q)
        request.group_name = self.c["move_group"]
        response = self.wait_future(
            self.state_validity.call_async(request), float(self.m["service_timeout_wall_sec"]),
            "COLLISION_PLAN_FAIL")
        contacts = [(item.contact_body_1, item.contact_body_2) for item in response.contacts]
        if not response.valid and allow_grasp_contacts and contacts:
            permitted_links = {str(self.geometry["left_finger_link"]), str(self.geometry["right_finger_link"])}
            expected = all("day3_v5_visual_block" in {first, second} and
                           bool(permitted_links.intersection({first, second}))
                           for first, second in contacts)
            if expected:
                return True, "expected finger/block contact"
        text_contacts = [f"{first}<->{second}" for first, second in contacts]
        return bool(response.valid), ",".join(text_contacts[:5])

    def validate_tcp_pose(self, q: list[float], desired_xyz: list[float], desired_q: list[float],
                          reason: str, allow_grasp_contacts: bool = False) -> dict[str, object]:
        actual = self.fk_poses(q, [self.c["tcp_link"]])[self.c["tcp_link"]]
        position_error, orientation_error = pose_error(
            actual["xyz"], actual["quaternion_xyzw"], desired_xyz, desired_q)
        if (position_error > float(self.m["tcp_position_tolerance_m"]) or
                orientation_error > float(self.orientation_config["maximum_orientation_error_rad"])):
            raise TestFailure(reason, f"IK/FK error position={position_error:.6f} orientation={orientation_error:.6f}")
        valid, detail = self.state_is_valid(q, allow_grasp_contacts)
        if not valid:
            raise TestFailure("COLLISION_PLAN_FAIL", detail or "invalid robot state")
        return {"xyz": actual["xyz"], "quaternion_xyzw": actual["quaternion_xyzw"],
                "position_m": position_error, "orientation_rad": orientation_error}

    @staticmethod
    def interpolate_joint_segment(first: list[float], second: list[float], maximum_step: float) -> list[list[float]]:
        count = max(1, int(math.ceil(max(abs(b - a) for a, b in zip(first, second)) / maximum_step)))
        return [[a + (b - a) * step / count for a, b in zip(first, second)]
                for step in range(1, count + 1)]

    def check_joint_path(self, start: list[float], path: list[list[float]], reason: str) -> None:
        previous = list(start)
        for target in path:
            for state in self.interpolate_joint_segment(
                    previous, target, float(self.m["state_validity_joint_step_rad"])):
                valid, detail = self.state_is_valid(state)
                if not valid:
                    raise TestFailure(reason, detail or "collision/invalid intermediate state")
            if max(abs(a - b) for a, b in zip(previous, target)) > float(self.m["maximum_joint_step_rad"]):
                raise TestFailure("TRAJECTORY_PLAN_FAIL", "discontinuous Cartesian IK branch")
            previous = target

    def cartesian_ik_path(self, start_grasp: list[float], end_grasp: list[float], rotation: np.ndarray,
                          tcp_offset: list[float], seed: list[float], reason: str) -> tuple[list[list[float]], list[dict[str, object]]]:
        distance = self.vector_distance(start_grasp, end_grasp)
        steps = max(1, int(math.ceil(distance / float(self.m["cartesian_step_m"]))))
        quaternion = matrix_to_quaternion(rotation)
        previous = list(seed)
        path: list[list[float]] = []
        errors: list[dict[str, object]] = []
        offset_world = rotation @ np.asarray(tcp_offset, dtype=float)
        for index in range(1, steps + 1):
            alpha = index / steps
            grasp = np.asarray(start_grasp) + alpha * (np.asarray(end_grasp) - np.asarray(start_grasp))
            tcp = (grasp - offset_world).astype(float).tolist()
            q = self.solve_tcp_ik(tcp, quaternion, previous, reason)
            if max(abs(a - b) for a, b in zip(previous, q)) > float(self.m["maximum_joint_step_rad"]):
                raise TestFailure("TRAJECTORY_PLAN_FAIL", "Cartesian IK branch jump")
            actual = self.validate_tcp_pose(q, tcp, quaternion, reason, allow_grasp_contacts=True)
            actual_center = np.asarray(actual["xyz"]) + quaternion_to_matrix(actual["quaternion_xyzw"]) @ np.asarray(tcp_offset)
            center_error = float(np.linalg.norm(actual_center - grasp))
            if center_error > float(self.m["grasp_center_tolerance_m"]):
                raise TestFailure(reason, f"TCP-corrected grasp-center FK error={center_error:.6f}")
            actual["grasp_center_error_m"] = center_error
            path.append(q)
            errors.append(actual)
            previous = q
        return path, errors

    def arm_message(self, positions: list[float], seconds: float) -> JointTrajectory:
        msg = JointTrajectory()
        msg.joint_names = self.arm_joints
        point = JointTrajectoryPoint()
        point.positions = list(map(float, positions))
        seconds = max(1.0, float(seconds))
        point.time_from_start = DurationMsg(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
        msg.points = [point]
        return msg

    def arm_path_message(self, path: list[list[float]], seconds: float) -> JointTrajectory:
        if not path:
            raise TestFailure("TRAJECTORY_PLAN_FAIL", "empty path")
        msg = JointTrajectory()
        msg.joint_names = self.arm_joints
        seconds = max(1.0, float(seconds))
        for index, positions in enumerate(path, 1):
            elapsed = seconds * index / len(path)
            point = JointTrajectoryPoint()
            point.positions = list(map(float, positions))
            point.time_from_start = DurationMsg(sec=int(elapsed), nanosec=int((elapsed % 1.0) * 1e9))
            msg.points.append(point)
        return msg

    def arm_reached(self, target: list[float]) -> bool:
        tolerance = float(self.m["arm_joint_tolerance_rad"])
        return all(name in self.joints and abs(self.joints[name][0] - target[i]) <= tolerance
                   for i, name in enumerate(self.arm_joints))

    def arm_tightly_reached(self, target: list[float]) -> bool:
        if not all(name in self.joints for name in self.arm_joints):
            return False
        error = max(abs(self.joints[name][0] - target[i]) for i, name in enumerate(self.arm_joints))
        velocities = [abs(self.joints[name][1]) for name in self.arm_joints
                      if math.isfinite(self.joints[name][1])]
        return (error <= float(self.m["grasp_settle_joint_error_rad"]) and velocities and
                max(velocities) <= float(self.m["grasp_settle_velocity_rad_sec"]))

    def move_arm(self, target: list[float], duration_sec: float, reason: str) -> None:
        msg = self.arm_message(target, duration_sec)
        self.arm_pub.publish(msg)
        for _ in range(5):
            self.spin_once()
        if not self.arm_reached(target):
            self.arm_pub.publish(msg)
        self.wait_until(lambda: self.arm_reached(target), float(self.t["arm_wall_sec"]), reason)

    def move_arm_path(self, path: list[list[float]], duration_sec: float, reason: str) -> None:
        target = path[-1]
        msg = self.arm_path_message(path, duration_sec)
        self.arm_pub.publish(msg)
        for _ in range(5):
            self.spin_once()
        if not self.arm_reached(target):
            self.arm_pub.publish(msg)
        self.wait_until(lambda: self.arm_reached(target), float(self.t["arm_wall_sec"]), reason)

    def wait_arm_settle(self, target: list[float], reason: str) -> None:
        stable_start: float | None = None
        deadline = time.monotonic() + float(self.t["arm_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            error = max(abs(self.joints[name][0] - target[i]) for i, name in enumerate(self.arm_joints))
            velocities = [abs(self.joints[name][1]) for name in self.arm_joints if math.isfinite(self.joints[name][1])]
            valid = error <= float(self.m["grasp_settle_joint_error_rad"]) and velocities and max(velocities) <= float(self.m["grasp_settle_velocity_rad_sec"])
            if valid:
                stable_start = self.sim_time if stable_start is None else stable_start
                if self.sim_time - stable_start >= float(self.m["grasp_settle_sim_sec"]):
                    return
            else:
                stable_start = None
        raise TestFailure(reason, "arm did not settle at grasp target")

    def set_block_pose(self, xyz: list[float]) -> None:
        if self.closing_started:
            raise TestFailure("RESET_FAIL", "set_pose forbidden after closing begins")
        request = (f'name: "{self.c["block_name"]}", position: {{x: {xyz[0]}, y: {xyz[1]}, z: {xyz[2]}}}, '
                   "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}")
        process = subprocess.run([
            "ign", "service", "-s", f'/world/{self.c["world_name"]}/set_pose',
            "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
            "--timeout", "3000", "--req", request,
        ], text=True, capture_output=True, timeout=30, check=False)
        output = (process.stdout + "\n" + process.stderr).strip().lower()
        if process.returncode or not ("true" in output or "data: 1" in output):
            raise TestFailure("RESET_FAIL", output)

    @staticmethod
    def box_object(object_id: str, center: list[float], size: list[float], yaw: float = 0.0) -> CollisionObject:
        item = CollisionObject()
        item.header.frame_id = "base_link"
        item.id = object_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(map(float, size))
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, center)
        pose.orientation.z = math.sin(float(yaw) / 2.0)
        pose.orientation.w = math.cos(float(yaw) / 2.0)
        item.primitives = [primitive]
        item.primitive_poses = [pose]
        item.operation = CollisionObject.ADD
        return item

    def update_planning_scene(self, visual: dict[str, object]) -> None:
        if self.scene_initialized:
            return
        center = list(map(float, visual["xyz"]))
        angle = float(visual.get("angle_rad", 0.0))
        dimensions = list(map(float, visual.get("dimensions_m", self.c["block_size_m"][:2])))
        size_xy = [max(float(self.c["block_size_m"][0]), dimensions[0]),
                   max(float(self.c["block_size_m"][1]), dimensions[1])]
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            self.box_object("day3_v5_table", self.geometry["table_center_base_m"], self.geometry["table_size_m"]),
            self.box_object("day3_v5_pedestal", self.geometry["pedestal_center_base_m"], self.geometry["pedestal_size_m"]),
        ]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self.wait_future(self.apply_scene.call_async(request),
                                    float(self.m["service_timeout_wall_sec"]), "COLLISION_PLAN_FAIL")
        if not response.success:
            raise TestFailure("COLLISION_PLAN_FAIL", "MoveIt rejected planning-scene update")
        self.scene_initialized = True

    def reset_block(self, xyz: list[float] | None = None) -> None:
        expected = list(map(float, xyz if xyz is not None else self.c["block_reset_world"]))
        self.set_block_pose(expected)
        stable_start: float | None = None
        deadline = time.monotonic() + float(self.t["reset_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            position_ok = self.block_xyz is not None and math.sqrt(sum((self.block_xyz[i] - expected[i]) ** 2 for i in range(3))) <= float(self.rst["pose_tolerance_m"])
            speed_ok = self.block_linear <= float(self.rst["linear_speed_max_m_sec"]) and self.block_angular <= float(self.rst["angular_speed_max_rad_sec"])
            if position_ok and speed_ok:
                stable_start = self.sim_time if stable_start is None else stable_start
                if self.sim_time - stable_start >= float(self.rst["stable_sim_sec"]):
                    self.clear_contacts()
                    return
            else:
                stable_start = None
        raise TestFailure("RESET_FAIL", "block did not settle at standard pose")

    def clear_contacts(self) -> None:
        self.left_last_wood = self.right_last_wood = -math.inf
        self.left_depth = self.right_depth = 0.0
        self.left_contact_samples = []
        self.right_contact_samples = []
        self.max_penetration = 0.0
        self.first_contact_angle = None
        self.hold_reference_relative_z = None
        self.max_relative_slip = 0.0
        self.block_contact_names.clear()

    def open_gripper(self) -> None:
        target = float(self.g["open_target"])
        self.command_gripper(float(self.g["effort_limit_nm"]), target)
        deadline = time.monotonic() + float(self.t["open_wall_sec"])
        stable_start: float | None = None
        while time.monotonic() < deadline:
            self.spin_once()
            opened = (
                math.isfinite(self.r_position())
                and self.r_position() >= target - 0.01
                and abs(self.r_velocity()) <= float(self.g["stall_velocity_rad_sec"])
            )
            if opened:
                stable_start = self.sim_time if stable_start is None else stable_start
                if self.sim_time - stable_start >= float(self.rst["stable_sim_sec"]):
                    break
            else:
                stable_start = None
        else:
            self.stop_gripper()
            raise TestFailure("INTERFACE_NOT_READY", "gripper open timeout")
        self.gripper_mode = "open"
        self.current_gripper_effort = float(self.g["open_hold_effort_nm"])
        self.publish_gripper(force=True)
        if self.r_position() < float(self.g["open_minimum"]):
            raise TestFailure("INTERFACE_NOT_READY", f"gripper did not open: angle={self.r_position():.6f}")
        self.closing_started = False

    def camera_topics_ready(self) -> bool:
        return all(self.get_publishers_info_by_topic(self.topics[name]) for name in ("rgb", "depth", "camera_info"))

    def collect_visual(self) -> dict[str, object]:
        deadline = time.monotonic() + float(self.t["camera_topics_wall_sec"])
        while time.monotonic() < deadline and not self.camera_topics_ready():
            self.spin_once()
        if not self.camera_topics_ready():
            raise TestFailure("PERCEPTION_FAIL", "camera topic missing")
        samples: list[dict[str, object]] = []
        last_stamp = None
        deadline = time.monotonic() + float(self.t["visual_wall_sec"])
        needed = int(self.c["perception"]["samples"])
        last_evaluation: dict[str, object] | None = None
        while time.monotonic() < deadline:
            self.spin_once()
            msg = self.target_msg
            surface_msg = self.surface_msg
            footprint_msg = self.footprint_msg
            now = time.monotonic()
            if msg is None or surface_msg is None or footprint_msg is None or not self.detected:
                continue
            if max(now - self.target_received_wall, now - self.detected_received_wall,
                   now - self.surface_received_wall, now - self.footprint_received_wall,
                   now - self.confidence_received_wall) > 0.30:
                continue
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            surface_stamp = (surface_msg.header.stamp.sec, surface_msg.header.stamp.nanosec)
            footprint_stamp = (footprint_msg.header.stamp.sec, footprint_msg.header.stamp.nanosec)
            if stamp != surface_stamp or stamp != footprint_stamp:
                continue
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            xyz = [float(msg.point.x), float(msg.point.y), float(msg.point.z)]
            surface = [float(surface_msg.point.x), float(surface_msg.point.y), float(surface_msg.point.z)]
            points = np.asarray([[float(point.x), float(point.y)] for point in footprint_msg.polygon.points], dtype=float)
            if msg.header.frame_id and msg.header.frame_id != self.c["base_frame"]:
                raise TestFailure("PERCEPTION_UNSTABLE", f"frame={msg.header.frame_id}")
            if len(points) != 4:
                continue
            try:
                corners, angle, dimensions = ordered_rectangle(points)
            except ValueError:
                continue
            if all(math.isfinite(value) for value in xyz + surface + points.ravel().tolist()) and math.isfinite(self.confidence):
                samples.append({"xyz": xyz, "surface": surface, "confidence": self.confidence,
                                "corners": corners.tolist(), "angle": angle, "dimensions": dimensions})
                if len(samples) > needed:
                    samples = samples[-needed:]
                if len(samples) == needed:
                    xyz = [statistics.median([sample["xyz"][i] for sample in samples]) for i in range(3)]
                    std = [statistics.pstdev([sample["xyz"][i] for sample in samples]) for i in range(3)]
                    surface = [statistics.median([sample["surface"][i] for sample in samples]) for i in range(3)]
                    confidence = statistics.median([float(sample["confidence"]) for sample in samples])
                    angle, angle_std = square_circular_mean([float(sample["angle"]) for sample in samples])
                    dimensions = [statistics.median([float(sample["dimensions"][i]) for sample in samples]) for i in range(2)]
                    # The target is a square: perspective/arm occlusion can shorten one
                    # fitted side.  Preserve the contour angle modulo 90 degrees and use
                    # the longer observed side for both symmetric contact dimensions.
                    square_side = max(dimensions)
                    dimensions = [square_side, square_side]
                    corners = np.median(np.asarray([sample["corners"] for sample in samples]), axis=0).tolist()
                    last_evaluation = {"xyz": xyz, "std": std, "surface": surface,
                                       "confidence": confidence, "angle": angle, "angle_std": angle_std,
                                       "dimensions": dimensions, "corners": corners}
                    stable = (confidence >= float(self.c["perception"]["confidence_minimum"]) and
                              max(std) <= float(self.c["perception"]["maximum_std_m"]) and
                              math.degrees(angle_std) <= float(self.c["perception"]["maximum_axis_std_deg"]))
                    if stable:
                        for axis, value, bounds in zip("xyz", xyz, (self.c["perception"]["workspace_x_m"], self.c["perception"]["workspace_y_m"], self.c["perception"]["workspace_z_m"])):
                            if not float(bounds[0]) <= value <= float(bounds[1]):
                                raise TestFailure("OUT_OF_WORKSPACE", f"{axis}={value:.5f} bounds={bounds}")
                        expected_delta = float(self.c["block_size_m"][2]) / 2.0
                        actual_delta = surface[2] - xyz[2]
                        if abs(actual_delta - expected_delta) > float(self.c["perception"]["center_surface_delta_tolerance_m"]):
                            raise TestFailure("PERCEPTION_UNSTABLE", f"surface-center z delta={actual_delta:.5f}")
                        side_bounds = self.c["perception"]["object_side_m"]
                        if any(not float(side_bounds[0]) <= side <= float(side_bounds[1]) for side in dimensions):
                            raise TestFailure("PERCEPTION_UNSTABLE", f"footprint dimensions={dimensions}")
                        print(f"TRIAL={self.current_trial_id:03d} VISUAL_CENTER={xyz} SURFACE={surface} "
                              f"CONTOUR_DEG={math.degrees(angle):.2f} AXIS_STD_DEG={math.degrees(angle_std):.2f} "
                              f"FOOTPRINT={dimensions} CONFIDENCE={confidence:.4f}", flush=True)
                        return {"xyz": xyz, "surface": surface, "std": std, "confidence": confidence,
                                "angle_rad": angle, "angle_std_rad": angle_std,
                                "dimensions_m": dimensions, "corners_xy": corners}
        if last_evaluation is None:
            raise TestFailure("PERCEPTION_FAIL", f"valid samples {len(samples)}/{needed}")
        raise TestFailure("PERCEPTION_UNSTABLE", json.dumps(last_evaluation, ensure_ascii=False))

    def prepare_initial(self, test_world: list[float]) -> None:
        self.check_interfaces()
        self.set_phase("RESET_BLOCK")
        self.reset_block(test_world)
        self.set_phase("OPEN_GRIPPER")
        self.open_gripper()
        self.set_phase("MOVE_INITIAL_VIEW")
        self.move_arm(list(map(float, self.c["initial_pose"])), float(self.m["initial_duration_sec"]), "INTERFACE_NOT_READY")
        self.set_phase("WAIT_SETTLE")
        target = list(map(float, self.c["initial_pose"]))
        stable_start: float | None = None
        deadline = time.monotonic() + float(self.t["arm_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            error = max(abs(self.joints[name][0] - target[i]) for i, name in enumerate(self.arm_joints))
            velocities = [abs(self.joints[name][1]) for name in self.arm_joints if math.isfinite(self.joints[name][1])]
            settled = error <= float(self.rst["arm_settle_joint_error_rad"]) and velocities and max(velocities) <= float(self.rst["arm_settle_velocity_rad_sec"])
            if settled:
                stable_start = self.sim_time if stable_start is None else stable_start
                if self.sim_time - stable_start >= float(self.rst["arm_settle_sim_sec"]):
                    return
            else:
                stable_start = None
        raise TestFailure("INTERFACE_NOT_READY", "initial-view pose did not settle")

    @staticmethod
    def rotation_to_quaternion(rotation: list[list[float]]) -> list[float]:
        r = rotation
        trace = r[0][0] + r[1][1] + r[2][2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s,
                 (r[1][0] - r[0][1]) / s, 0.25 * s]
        elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
            s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
            q = [0.25 * s, (r[0][1] + r[1][0]) / s,
                 (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s]
        elif r[1][1] > r[2][2]:
            s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
            q = [(r[0][1] + r[1][0]) / s, 0.25 * s,
                 (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s]
        else:
            s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
            q = [(r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s,
                 0.25 * s, (r[1][0] - r[0][1]) / s]
        norm = math.sqrt(sum(value * value for value in q))
        return [value / norm for value in q]

    @staticmethod
    def rotate(rotation: list[list[float]], vector: list[float]) -> list[float]:
        return [sum(rotation[row][col] * vector[col] for col in range(3)) for row in range(3)]

    @staticmethod
    def quaternion_error(first: list[float], second: list[float]) -> float:
        dot = abs(sum(a * b for a, b in zip(first, second)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    @staticmethod
    def vector_distance(first: list[float], second: list[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    def geometry_targets(self, surface: list[float]) -> tuple[list[float], list[float], list[float]]:
        normal = list(map(float, self.geometry["table_normal_base"]))
        # DAY8 uses a fixed signed surface offset derived from the URDF closing
        # sweep. The visual top surface, not episode ground-truth size, sets z.
        descent = [surface[i] - normal[i] * float(self.geometry["contact_center_insertion_m"])
                   for i in range(3)]
        pregrasp = [descent[i] + normal[i] * float(self.m["pregrasp_clearance_m"]) for i in range(3)]
        lift = [descent[i] + normal[i] * float(self.m["lift_command_m"]) for i in range(3)]
        return pregrasp, descent, lift

    def seed_bank(self, current: list[float]) -> list[list[float]]:
        """Point-independent deterministic seeds spanning the actual URDF limits."""
        seeds = [list(current)]
        rng = random.Random(int(self.orientation_config["seed_random_seed"]))
        for _ in range(int(self.orientation_config["deterministic_seed_count"])):
            seeds.append([rng.uniform(*self.joint_limits[name]) for name in self.arm_joints])
        return seeds

    def joint_margin(self, path: list[list[float]]) -> float:
        return min(min(value - self.joint_limits[name][0], self.joint_limits[name][1] - value)
                   for q in path for name, value in zip(self.arm_joints, q))

    def urdf_grasp_pose(self, q: list[float], tcp_offset: list[float]) -> tuple[np.ndarray, np.ndarray]:
        transform = self.urdf.transform(
            self.c["base_frame"], self.c["tcp_link"], dict(zip(self.arm_joints, q)))
        center = transform[:3, 3] + transform[:3, :3] @ np.asarray(tcp_offset, dtype=float)
        return center, transform[:3, :3]

    def solve_contour_ik(self, grasp_center: list[float], axis_angle: float,
                         tcp_offset: list[float], seeds: list[list[float]], reason: str,
                         allow_grasp_contacts: bool) -> tuple[list[float], dict[str, object]]:
        """Solve all five active joints while yaw/pitch/tilt remain decision variables.

        A five-DOF arm cannot satisfy an arbitrary six-DOF quaternion.  The physically
        relevant constraints are the 3-D contact-center position plus the two degrees
        that keep the closing axis horizontal and parallel to an OpenCV rectangle side.
        The remaining approach tilt is searched by the optimizer and checked afterward.
        """
        target = np.asarray(grasp_center, dtype=float)
        desired = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0], dtype=float)
        perpendicular = np.array([-desired[1], desired[0], 0.0], dtype=float)
        margin = float(self.m["joint_limit_margin_rad"])
        lower = np.asarray([self.joint_limits[name][0] + margin for name in self.arm_joints])
        upper = np.asarray([self.joint_limits[name][1] - margin for name in self.arm_joints])
        candidates: list[tuple[float, list[float], dict[str, object]]] = []
        rejections: list[str] = []

        def residual(values: np.ndarray) -> np.ndarray:
            center, rotation = self.urdf_grasp_pose(values.tolist(), tcp_offset)
            closing = rotation[:, 1]
            return np.concatenate(((center - target) / 0.003,
                                   [float(np.dot(closing, perpendicular)) / 0.02,
                                    float(closing[2]) / 0.30]))

        for seed in seeds:
            start = np.clip(np.asarray(seed, dtype=float), lower, upper)
            solution = least_squares(residual, start, bounds=(lower, upper),
                                     xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=250)
            q = solution.x.astype(float).tolist()
            center, rotation = self.urdf_grasp_pose(q, tcp_offset)
            closing = rotation[:, 1]
            position_error = float(np.linalg.norm(center - target))
            line_error = min(float(np.linalg.norm(closing - desired)),
                             float(np.linalg.norm(closing + desired)))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(rotation[:, 2], [0.0, 0.0, -1.0]))))))
            if (position_error > float(self.m["grasp_center_tolerance_m"]) or line_error > 0.06 or
                    abs(float(closing[2])) > 0.50 or
                    tilt < float(self.orientation_config["tilt_min_deg"]) - 1e-6 or
                    tilt > float(self.orientation_config["tilt_max_deg"]) + 1e-6):
                rejections.append(f"geometry pos={position_error:.5f} line={line_error:.5f} tilt={tilt:.2f}")
                continue
            quaternion = matrix_to_quaternion(rotation)
            tcp_xyz = transform_xyz = (center - rotation @ np.asarray(tcp_offset)).tolist()
            try:
                moveit_fk = self.validate_tcp_pose(q, tcp_xyz, quaternion, reason, allow_grasp_contacts)
            except TestFailure as exc:
                rejections.append(str(exc))
                continue
            score = position_error * 200.0 + line_error * 2.0 + self.vector_distance(seed, q) * 0.02
            candidates.append((score, q, {
                "tcp_xyz": transform_xyz, "quaternion_xyzw": quaternion,
                "position_m": position_error, "orientation_rad": float(moveit_fk["orientation_rad"]),
                "grasp_center_error_m": position_error, "closing_axis_line_error": line_error,
                "tilt_deg": tilt,
            }))
        if not candidates:
            raise TestFailure(reason, f"no contour-constrained IK at center={grasp_center} "
                              f"axis={math.degrees(axis_angle):.2f}; samples={rejections[:3]}")
        _, q, details = min(candidates, key=lambda item: item[0])
        return q, details

    def contour_cartesian_path(self, start: list[float], end: list[float], axis_angle: float,
                               tcp_offset: list[float], seed: list[float], reason: str) -> tuple[list[list[float]], list[dict[str, object]]]:
        count = max(1, int(math.ceil(self.vector_distance(start, end) / float(self.m["cartesian_step_m"]))))
        path, errors, previous = [], [], list(seed)
        for index in range(1, count + 1):
            alpha = index / count
            center = (np.asarray(start) + alpha * (np.asarray(end) - np.asarray(start))).tolist()
            q, detail = self.solve_contour_ik(center, axis_angle, tcp_offset, [previous], reason, True)
            jump = max(abs(a - b) for a, b in zip(previous, q))
            if jump > float(self.m["maximum_joint_step_rad"]):
                raise TestFailure("TRAJECTORY_PLAN_FAIL", f"contour IK branch jump={jump:.6f}")
            for state in self.interpolate_joint_segment(
                    previous, q, float(self.m["state_validity_joint_step_rad"])):
                valid, collision = self.state_is_valid(state, allow_grasp_contacts=True)
                if not valid:
                    raise TestFailure("COLLISION_PLAN_FAIL", collision or "invalid Cartesian intermediate state")
            path.append(q)
            errors.append(detail)
            previous = q
        return path, errors

    def analytic_gripper_clearance(self, q: list[float], visual: dict[str, object]) -> float:
        """Check the intentional target gap and table clearance from URDF boxes."""
        positions = dict(zip(self.arm_joints, q))
        width = statistics.median(list(map(float, visual["dimensions_m"])))
        key = (width, float(self.g["overclose_offset"]), float(self.g["open_target"]))
        cache = getattr(self, "_day8_finger_sweeps", {})
        if key not in cache:
            contact_angle, _ = self.urdf.contact_angle_for_width(width)
            lower = max(self.urdf.joints["r_joint"].lower,
                        contact_angle - float(self.g["overclose_offset"]))
            vertices = []
            # Fixed finger vertices in link5; only link5-to-base changes along a path.
            for angle in np.linspace(lower, float(self.g["open_target"]), 17):
                for link, collision in ((self.geometry["left_finger_link"], self.geometry["left_finger_collision"]),
                                        (self.geometry["right_finger_link"], self.geometry["right_finger_collision"])):
                    box = self.urdf.boxes[(str(link), str(collision))]
                    transform = self.urdf.box_transform("link5", box, {"r_joint": float(angle)})
                    for sx in (-.5, .5):
                        for sy in (-.5, .5):
                            for sz in (-.5, .5):
                                vertices.append((transform @ np.r_[np.array([sx,sy,sz])*box.size, 1.])[:3])
            cache[key] = np.asarray(vertices)
            self._day8_finger_sweeps = cache
        transform = self.urdf.transform(self.c["base_frame"], "link5", positions)
        minimum_z = float(np.min(cache[key] @ transform[2,:3] + transform[2,3]))
        table_clearance = minimum_z
        open_geometry = self.urdf.finger_geometry(float(self.g["open_target"]))
        gap_clearance = float(open_geometry["gap_m"]) - statistics.median(
            list(map(float, visual["dimensions_m"])))
        if table_clearance < float(self.geometry["minimum_table_clearance_m"]):
            raise TestFailure("COLLISION_PLAN_FAIL", f"finger/table clearance={table_clearance:.6f}")
        if gap_clearance < 0.0005:
            raise TestFailure("COLLISION_PLAN_FAIL", f"open finger/block gap clearance={gap_clearance:.6f}")
        return min(table_clearance, gap_clearance)

    def solve_candidates(self, visual: dict[str, object]) -> dict[str, object]:
        center = list(map(float, visual["xyz"]))
        surface = list(map(float, visual["surface"]))
        current = self.arm_q()
        self.update_planning_scene(visual)
        width = statistics.median(list(map(float, visual["dimensions_m"])))
        expected_contact_angle, tcp_offset = self.urdf.tcp_offset_for_width(width, self.c["tcp_link"])
        pregrasp, descent, lift = self.geometry_targets(surface)
        records: list[dict[str, object]] = []
        best: dict[str, object] | None = None
        solved_lines: set[int] = set()
        for symmetry in self.orientation_config["square_symmetries"]:
            line_index = int(symmetry) % 2
            axis_angle = float(visual["angle_rad"]) + line_index * math.pi / 2.0
            if line_index in solved_lines:
                records.append({"contour_yaw_deg": math.degrees(float(visual["angle_rad"])),
                                "symmetry": int(symmetry), "feasible": True,
                                "equivalent_to_symmetry": line_index})
                continue
            solved_lines.add(line_index)
            candidate: dict[str, object] | None = None
            failures: list[str] = []
            try:
                pre_q, pre_error = self.solve_contour_ik(
                    pregrasp, axis_angle, tcp_offset, self.seed_bank(current), "IK_FAIL", False)
                for state in self.interpolate_joint_segment(
                        current, pre_q, float(self.m["state_validity_joint_step_rad"])):
                    valid, detail = self.state_is_valid(state)
                    if not valid:
                        raise TestFailure("COLLISION_PLAN_FAIL", detail or "current-to-pregrasp collision")
                descent_path, descent_errors = self.contour_cartesian_path(
                    pregrasp, descent, axis_angle, tcp_offset, pre_q, "IK_FAIL")
                descent_q = descent_path[-1]
                lift_path, lift_errors = self.contour_cartesian_path(
                    descent, lift, axis_angle, tcp_offset, descent_q, "IK_FAIL")
                lift_q = lift_path[-1]
                full_path = [pre_q] + descent_path + lift_path
                collision_clearance = min(self.analytic_gripper_clearance(q, visual)
                                          for q in descent_path + lift_path)
                margin = self.joint_margin(full_path)
                joint_motion = sum(self.vector_distance(a, b) for a, b in zip([current] + full_path, full_path))
                errors = [pre_error] + descent_errors + lift_errors
                position_error = max(float(value["grasp_center_error_m"]) for value in errors)
                orientation_error = max(float(value["orientation_rad"]) for value in errors)
                score = (float(self.orientation_config["position_weight"]) * position_error +
                         float(self.orientation_config["orientation_weight"]) * orientation_error +
                         float(self.orientation_config["joint_motion_weight"]) * joint_motion -
                         float(self.orientation_config["joint_margin_weight"]) * margin)
                final_rotation = self.urdf_grasp_pose(descent_q, tcp_offset)[1]
                tool_z = final_rotation[:, 2]
                radial = np.asarray([center[0], center[1], 0.0])
                radial /= max(float(np.linalg.norm(radial)), 1e-9)
                lean_sign = 1 if float(np.dot(tool_z, radial)) >= 0.0 else -1
                candidate = {
                    "offset_deg": math.degrees(axis_angle), "tilt_deg": descent_errors[-1]["tilt_deg"],
                    "lean_sign": lean_sign, "square_symmetry": int(symmetry),
                    "quaternion_xyzw": descent_errors[-1]["quaternion_xyzw"],
                    "tcp_offset_m": tcp_offset, "expected_contact_angle": expected_contact_angle,
                    "pregrasp_ee_xyz": pre_error["tcp_xyz"], "descent_ee_xyz": descent_errors[-1]["tcp_xyz"],
                    "lift_ee_xyz": lift_errors[-1]["tcp_xyz"],
                    "pregrasp_joints": pre_q, "descent_joints": descent_q, "lift_joints": lift_q,
                    "pregrasp_path": [pre_q], "descent_path": descent_path, "lift_path": lift_path,
                    "fk_position_error_m": position_error, "fk_orientation_error_rad": orientation_error,
                    "pose_fk_errors": {
                        "pregrasp": {"position_m": pre_error["grasp_center_error_m"], "orientation_rad": pre_error["orientation_rad"]},
                        "descent": {"position_m": descent_errors[-1]["grasp_center_error_m"], "orientation_rad": descent_errors[-1]["orientation_rad"]},
                        "lift": {"position_m": lift_errors[-1]["grasp_center_error_m"], "orientation_rad": lift_errors[-1]["orientation_rad"]},
                    },
                    "predicted_finger_lift_m": float(self.m["lift_command_m"]),
                    "predicted_finger_horizontal_drift_m": 0.0, "joint_motion_rad": joint_motion,
                    "minimum_joint_margin_rad": margin, "trajectory_waypoint_count": len(full_path), "score": score,
                    "minimum_collision_clearance_m": collision_clearance,
                }
            except TestFailure as exc:
                failures.append(str(exc))
            summary = {"contour_yaw_deg": math.degrees(float(visual["angle_rad"])),
                       "symmetry": int(symmetry), "feasible": candidate is not None,
                       "failure_samples": failures[:3]}
            if candidate is not None:
                summary.update({key: candidate[key] for key in
                                ("tilt_deg", "fk_position_error_m", "fk_orientation_error_rad",
                                 "joint_motion_rad", "minimum_joint_margin_rad", "score")})
                if best is None or float(candidate["score"]) < float(best["score"]):
                    best = candidate
            records.append(summary)
        if best is None:
            raise TestFailure("IK_FAIL", json.dumps(records, ensure_ascii=False))
        best["orientation_candidates"] = records
        print(f"TRIAL={self.current_trial_id:03d} SELECTED_CONTOUR_ORIENTATION={best['offset_deg']:+.2f}deg "
              f"TILT={best['tilt_deg']:.1f} LEAN={best['lean_sign']:+d} SYMMETRY={best['square_symmetry']} "
              f"IK_FK_POS_ERROR={best['fk_position_error_m']:.6f}m "
              f"IK_FK_ORIENTATION_ERROR={best['fk_orientation_error_rad']:.6f}rad "
              f"FINGER_FK_LIFT={best['predicted_finger_lift_m']:.6f}m "
              f"FINGER_FK_DRIFT={best['predicted_finger_horizontal_drift_m']:.6f}m "
              f"JOINT_MOTION={best['joint_motion_rad']:.6f}rad", flush=True)
        return best

    def validate_contact_geometry(self, result: dict[str, object], visual: dict[str, object],
                                  selected: dict[str, object], angle: float) -> None:
        width = statistics.median(list(map(float, visual["dimensions_m"])))
        expected = float(selected["expected_contact_angle"])
        finger = self.urdf.finger_geometry(angle)
        gap = float(finger["gap_m"])
        result["expected_contact_angle"] = expected
        result["contact_gap_m"] = gap
        early = (angle > expected + float(self.g["early_contact_angle_margin_rad"]) or
                 gap > width + float(self.g["early_contact_gap_margin_m"]))
        if early:
            detail = f"early contact angle={angle:.5f} expected={expected:.5f} gap={gap:.5f} width={width:.5f}"
            result["contact_geometry_valid"] = False
            result["contact_geometry_detail"] = detail
            raise TestFailure("CORNER_COLLISION", detail)

        def usable(samples: list[dict[str, object]]) -> list[tuple[np.ndarray, np.ndarray]]:
            output = []
            spawn = np.asarray(self.c["robot_spawn_world"], dtype=float)
            for sample in samples:
                if sample["position_world"] is None or sample["normal_world"] is None:
                    continue
                output.append((np.asarray(sample["position_world"], dtype=float) - spawn,
                               np.asarray(sample["normal_world"], dtype=float)))
            return output

        left, right = usable(self.left_contact_samples), usable(self.right_contact_samples)
        if not left or not right:
            # This Gazebo bridge may publish collision names and depths with empty
            # position/normal arrays.  In that case retain the strict aperture test,
            # visually corrected contact center, inner-finger sensor identity, and
            # require the subsequent physical lift+hold test instead of inventing data.
            detail = (f"contact vectors unavailable left={len(left)} right={len(right)}; "
                      f"aperture geometry accepted provisionally gap={gap:.5f} width={width:.5f}; "
                      "lift_and_hold_required")
            result["contact_geometry_valid"] = True
            result["contact_geometry_detail"] = detail
            return
        rotation = quaternion_to_matrix(list(map(float, selected["quaternion_xyzw"])))
        center = np.asarray(visual["xyz"], dtype=float)
        closing = rotation[:, 1]
        left_p = np.mean([item[0] for item in left], axis=0)
        right_p = np.mean([item[0] for item in right], axis=0)
        left_n = np.mean([item[1] for item in left], axis=0)
        right_n = np.mean([item[1] for item in right], axis=0)
        left_local = rotation.T @ (left_p - center)
        right_local = rotation.T @ (right_p - center)
        normal_alignment = min(abs(float(np.dot(left_n, closing))) / max(float(np.linalg.norm(left_n)), 1e-9),
                               abs(float(np.dot(right_n, closing))) / max(float(np.linalg.norm(right_n)), 1e-9))
        margin = float(self.g["contact_inside_margin_m"])
        length = max(map(float, visual["dimensions_m"]))
        between = left_local[1] * right_local[1] < 0.0
        inside_faces = (max(abs(float(left_local[0])), abs(float(right_local[0]))) <= length / 2.0 + margin and
                        max(abs(float(left_local[2])), abs(float(right_local[2]))) <=
                        float(self.c["block_size_m"][2]) / 2.0 + margin)
        opposed = float(np.dot(left_n, right_n)) < 0.0
        valid = (between and inside_faces and opposed and
                 normal_alignment >= float(self.g["contact_normal_alignment_min"]))
        detail = json.dumps({
            "gap_m": gap, "width_m": width, "left_local_m": left_local.tolist(),
            "right_local_m": right_local.tolist(), "center_between": bool(between),
            "inside_faces": bool(inside_faces), "opposed_normals": bool(opposed),
            "normal_alignment": normal_alignment,
        }, ensure_ascii=False)
        result["contact_geometry_valid"] = valid
        result["contact_geometry_detail"] = detail
        if not valid:
            corner = not between or not inside_faces
            raise TestFailure("CORNER_COLLISION" if corner else "NO_VALID_GRASP", detail)

    def dual_contact_search(self, result: dict[str, object], visual: dict[str, object],
                            selected: dict[str, object]) -> tuple[float, float]:
        self.set_phase("CLOSE_SEARCH")
        self.clear_contacts()
        self.closing_started = True
        self.gripper_mode = "closing"
        self.command_gripper(-float(self.g["search_effort_nm"]), float(self.g["close_target"]))
        stable_start: float | None = None
        deadline = time.monotonic() + float(self.t["dual_contact_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            result["left_contact_seen"] = bool(result["left_contact_seen"] or self.left_contact())
            result["right_contact_seen"] = bool(result["right_contact_seen"] or self.right_contact())
            self.status_print("contact", f"TRIAL={self.current_trial_id:03d} LEFT_CONTACT={int(self.left_contact())} "
                              f"RIGHT_CONTACT={int(self.right_contact())} STALL=0")
            if self.max_penetration > float(self.g["penetration_fail_m"]):
                self.stop_gripper()
                raise TestFailure("NO_DUAL_CONTACT", f"excessive penetration={self.max_penetration:.6f}")
            if self.left_contact() and self.right_contact():
                if stable_start is None:
                    self.set_phase("WAIT_DUAL_CONTACT")
                    stable_start = self.sim_time
                    self.first_contact_angle = float(self.r_position())
                    hold_target = max(
                        float(self.g["close_target"]),
                        self.first_contact_angle - float(self.g["overclose_offset"]),
                    )
                    self.start_gripper_servo(hold_target)
                if self.sim_time - stable_start >= float(self.g["dual_contact_stable_sim_sec"]):
                    first_angle = float(self.first_contact_angle)
                    result["contact_angle"] = float(first_angle)
                    try:
                        self.validate_contact_geometry(result, visual, selected, float(first_angle))
                    except TestFailure:
                        self.stop_gripper()
                        raise
                    hold_target = max(float(self.g["close_target"]),
                                      first_angle - float(self.g["overclose_offset"]))
                    self.start_gripper_servo(hold_target)
                    result["dual_contact"] = True
                    return float(first_angle), float(self.current_gripper_target)
            else:
                stable_start = None
                self.first_contact_angle = None
                self.command_gripper(-float(self.g["search_effort_nm"]), float(self.g["close_target"]))
        self.stop_gripper()
        if not result["left_contact_seen"]:
            raise TestFailure("NO_LEFT_CONTACT")
        if not result["right_contact_seen"]:
            raise TestFailure("NO_RIGHT_CONTACT")
        raise TestFailure("NO_DUAL_CONTACT")

    def wait_stall(self, hold_target: float, result: dict[str, object]) -> None:
        self.set_phase("WAIT_STALL")
        self.start_gripper_servo(hold_target)
        stable_start: float | None = None
        peak_stable = 0.0
        deadline = time.monotonic() + float(self.t["stall_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            velocity = abs(self.r_velocity())
            error = self.r_position() - self.current_gripper_target
            if math.isfinite(error):
                old_min = result.get("stall_min_error_rad")
                old_max = result.get("stall_max_error_rad")
                result["stall_min_error_rad"] = error if old_min is None else min(float(old_min), error)
                result["stall_max_error_rad"] = error if old_max is None else max(float(old_max), error)
                result["stall_final_error_rad"] = error
            if math.isfinite(velocity):
                old_min_v = result.get("stall_min_velocity_rad_sec")
                old_max_v = result.get("stall_max_velocity_rad_sec")
                result["stall_min_velocity_rad_sec"] = velocity if old_min_v is None else min(float(old_min_v), velocity)
                result["stall_max_velocity_rad_sec"] = velocity if old_max_v is None else max(float(old_max_v), velocity)
            valid = (
                self.left_contact()
                and self.right_contact()
                and math.isfinite(velocity)
                and velocity <= float(self.g["stall_velocity_rad_sec"])
                and error >= -0.005
                and self.analytic_contact_closure()
                >= float(self.g["minimum_contact_closure_m"])
            )
            self.status_print("stall", f"TRIAL={self.current_trial_id:03d} LEFT_CONTACT={int(self.left_contact())} "
                              f"RIGHT_CONTACT={int(self.right_contact())} STALL={int(valid)} "
                              f"GRIPPER_ERROR={error:.5f} VELOCITY={velocity:.5f} "
                              f"CLOSURE_M={self.analytic_contact_closure():.5f}")
            self.check_closing_safety("NO_STALL")
            if valid:
                stable_start = self.sim_time if stable_start is None else stable_start
                peak_stable = max(peak_stable, self.sim_time - stable_start)
                result["stall_peak_stable_sim_sec"] = peak_stable
                if peak_stable >= float(self.g["stall_duration_sim_sec"]):
                    return
            else:
                stable_start = None
        self.stop_gripper()
        raise TestFailure("NO_STALL")

    def lift_and_verify(self, path: list[list[float]], block_z_before: float) -> float:
        q = path[-1]
        msg = self.arm_path_message(path, float(self.m["lift_duration_sec"]))
        self.arm_pub.publish(msg)
        deadline = time.monotonic() + float(self.t["lift_wall_sec"])
        drop_start: float | None = None
        reached_without_lift_start: float | None = None
        maximum_z = block_z_before
        while time.monotonic() < deadline:
            self.spin_once()
            self.check_closing_safety("BLOCK_DROPPED_DURING_LIFT")
            if self.block_z is not None:
                maximum_z = max(maximum_z, self.block_z)
            actual_lift = maximum_z - block_z_before
            current_lift = -math.inf if self.block_z is None else self.block_z - block_z_before
            self.status_print("lift", f"TRIAL={self.current_trial_id:03d} ACTUAL_LIFT_M={current_lift:.5f} "
                              f"LEFT_CONTACT={int(self.left_contact())} RIGHT_CONTACT={int(self.right_contact())}")
            dropped = (actual_lift >= float(self.v["minimum_lift_m"]) and
                       (current_lift < float(self.v["minimum_hold_height_m"]) or self.table_contact()))
            partial_drop = (actual_lift >= 0.010 and current_lift < actual_lift - 0.010)
            if dropped or partial_drop:
                drop_start = self.sim_time if drop_start is None else drop_start
                if self.sim_time - drop_start >= float(self.v["drop_stable_sim_sec"]):
                    self.stop_gripper()
                    raise TestFailure("BLOCK_DROPPED_DURING_LIFT", f"current lift={current_lift:.6f}")
            else:
                drop_start = None
            if self.arm_tightly_reached(q) and current_lift < float(self.v["minimum_lift_m"]):
                reached_without_lift_start = (self.sim_time if reached_without_lift_start is None
                                              else reached_without_lift_start)
                if self.sim_time - reached_without_lift_start >= float(self.v["drop_stable_sim_sec"]):
                    self.stop_gripper()
                    raise TestFailure("BLOCK_NOT_LIFTED", f"arm reached lift pose; maximum lift={actual_lift:.6f}")
            else:
                reached_without_lift_start = None
            if (self.arm_tightly_reached(q) and current_lift >= float(self.v["minimum_lift_m"])
                    and not self.table_contact()):
                return maximum_z
        self.stop_gripper()
        raise TestFailure("BLOCK_NOT_LIFTED", f"maximum lift={maximum_z - block_z_before:.6f}")

    def hold(self, block_z_before: float, lifted_z: float) -> tuple[float, float, float, bool, float]:
        self.set_phase("HOLD")
        start_sim = self.sim_time
        start_wall = time.monotonic()
        deadline = time.monotonic() + float(self.t["hold_wall_sec"])
        minimum_z = lifted_z
        lost_start: float | None = None
        reference_relative_z = self.relative_block_tcp_z()
        self.hold_reference_relative_z = (
            reference_relative_z if math.isfinite(reference_relative_z) else None)
        self.max_relative_slip = 0.0
        stable_hold = True
        while time.monotonic() < deadline:
            self.spin_once()
            self.check_closing_safety("BLOCK_DROPPED_DURING_HOLD")
            if self.block_z is not None:
                minimum_z = min(minimum_z, self.block_z)
            relative_z = self.relative_block_tcp_z()
            if self.hold_reference_relative_z is not None and math.isfinite(relative_z):
                self.max_relative_slip = max(
                    self.max_relative_slip, self.hold_reference_relative_z - relative_z)
            if self.max_relative_slip > float(self.v["maximum_relative_slip_m"]):
                stable_hold = False
            below = (self.block_z is None or
                     self.block_z - block_z_before < float(self.v["minimum_hold_height_m"]) or
                     self.table_contact())
            if below:
                lost_start = self.sim_time if lost_start is None else lost_start
                if self.sim_time - lost_start >= float(self.v["drop_stable_sim_sec"]):
                    self.stop_gripper()
                    raise TestFailure("BLOCK_DROPPED_DURING_HOLD", "height/table criterion failed")
            else:
                lost_start = None
            elapsed_sim = self.sim_time - start_sim
            elapsed_wall = time.monotonic() - start_wall
            current_lift = -math.inf if self.block_z is None else self.block_z - block_z_before
            self.status_print("hold", f"TRIAL={self.current_trial_id:03d} HOLD_SIM_SEC={elapsed_sim:.3f} "
                              f"HOLD_WALL_SEC={elapsed_wall:.3f} ACTUAL_LIFT_M={current_lift:.5f} "
                              f"RELATIVE_SLIP_M={self.max_relative_slip:.5f} "
                              f"LEFT_CONTACT={int(self.left_contact())} RIGHT_CONTACT={int(self.right_contact())}")
            if (elapsed_sim >= float(self.v["hold_sim_sec"]) and
                    elapsed_wall >= float(self.v["hold_wall_sec"])):
                return minimum_z, elapsed_sim, elapsed_wall, stable_hold, self.max_relative_slip
        self.stop_gripper()
        raise TestFailure("BLOCK_DROPPED_DURING_HOLD", "hold timeout")

    def place_down(self, q: list[float], block_z_before: float) -> None:
        msg = self.arm_message(q, float(self.m["place_duration_sec"]))
        self.arm_pub.publish(msg)
        deadline = time.monotonic() + float(self.t["place_wall_sec"])
        while time.monotonic() < deadline:
            self.spin_once()
            height_ok = self.block_z is not None and abs(self.block_z - block_z_before) <= float(self.v["place_height_tolerance_m"])
            if self.arm_reached(q) and height_ok and self.table_contact():
                return
        raise TestFailure("PLACE_DOWN_FAIL", f"z={self.block_z} table_contact={self.table_contact()}")

    def safe_recover(self) -> tuple[bool, str]:
        details = []
        try:
            self.stop_gripper()
            self.open_gripper()
        except Exception as exc:
            details.append(f"open={exc}")
        self.closing_started = False
        try:
            self.move_arm(list(map(float, self.c["initial_pose"])), float(self.m["safe_duration_sec"]), "RESET_FAIL")
        except Exception as exc:
            details.append(f"arm={exc}")
        try:
            self.reset_block()
        except Exception as exc:
            details.append(f"block={exc}")
        return not details, "; ".join(details)

    def trial(self, trial_id: int, restart: int, test_case: dict[str, object], formal: bool) -> dict[str, object]:
        self.current_trial_id = trial_id
        result = blank_result(trial_id, self.config_hash, restart)
        start_monotonic = time.monotonic()
        started = False
        results_dir = self.root / "results"
        planned_world = list(map(float, test_case["planned_world_xyz"]))
        result.update({key: test_case[key] for key in
                       ("case_id", "grid_row", "grid_col", "repeat", "planned_base_xyz", "planned_world_xyz")})
        try:
            self.prepare_initial(planned_world)
            result["start_wall_time"] = now_text()
            start_monotonic = time.monotonic()
            started = True
            if formal:
                atomic_text(results_dir / "in_progress.json", json.dumps({
                    "trial_id": trial_id, "case_id": test_case["case_id"],
                    "config_sha256": self.config_hash,
                    "start_wall_time": result["start_wall_time"], "sim_restart_index": restart,
                }, indent=2) + "\n")
            self.set_phase("VISUAL_DETECTION")
            visual = self.collect_visual()
            result.update({
                "visual_detected": True,
                "visual_center_x": visual["xyz"][0], "visual_center_y": visual["xyz"][1],
                "visual_center_z": visual["xyz"][2], "visual_std_x": visual["std"][0],
                "visual_std_y": visual["std"][1], "visual_std_z": visual["std"][2],
                "visual_surface_x": visual["surface"][0], "visual_surface_y": visual["surface"][1],
                "visual_surface_z": visual["surface"][2], "visual_confidence": visual["confidence"],
                "visual_contour_angle_deg": math.degrees(float(visual["angle_rad"])),
                "visual_contour_std_deg": math.degrees(float(visual["angle_std_rad"])),
                "visual_footprint_m": visual["dimensions_m"],
            })
            self.set_phase("GENERATE_TOP_GRASP_CANDIDATES")
            selected = self.solve_candidates(visual)
            result.update({
                "orientation_candidates": selected["orientation_candidates"],
                "selected_orientation_deg": selected["offset_deg"],
                "selected_quaternion_xyzw": selected["quaternion_xyzw"],
                "selected_tilt_deg": selected["tilt_deg"], "selected_lean_sign": selected["lean_sign"],
                "selected_square_symmetry": selected["square_symmetry"],
                "selected_tcp_offset_m": selected["tcp_offset_m"],
                "expected_contact_angle": selected["expected_contact_angle"],
                "selected_fk_position_error_m": selected["fk_position_error_m"],
                "selected_fk_orientation_error_rad": selected["fk_orientation_error_rad"],
                "selected_joint_motion_rad": selected["joint_motion_rad"],
                "selected_pregrasp_fk_position_error_m": selected["pose_fk_errors"]["pregrasp"]["position_m"],
                "selected_pregrasp_fk_orientation_error_rad": selected["pose_fk_errors"]["pregrasp"]["orientation_rad"],
                "selected_descent_fk_position_error_m": selected["pose_fk_errors"]["descent"]["position_m"],
                "selected_descent_fk_orientation_error_rad": selected["pose_fk_errors"]["descent"]["orientation_rad"],
                "selected_lift_fk_position_error_m": selected["pose_fk_errors"]["lift"]["position_m"],
                "selected_lift_fk_orientation_error_rad": selected["pose_fk_errors"]["lift"]["orientation_rad"],
                "selected_predicted_finger_lift_m": selected["predicted_finger_lift_m"],
                "selected_predicted_finger_horizontal_drift_m": selected["predicted_finger_horizontal_drift_m"],
                "pregrasp_ee_xyz": selected["pregrasp_ee_xyz"],
                "descent_ee_xyz": selected["descent_ee_xyz"], "lift_ee_xyz": selected["lift_ee_xyz"],
                "pregrasp_joints": selected["pregrasp_joints"],
                "descent_joints": selected["descent_joints"], "lift_joints": selected["lift_joints"],
                "trajectory_waypoint_count": selected["trajectory_waypoint_count"],
                "minimum_joint_margin_rad": selected["minimum_joint_margin_rad"],
                "minimum_collision_clearance_m": selected["minimum_collision_clearance_m"],
            })
            pre_q = list(map(float, selected["pregrasp_joints"]))
            descent_q = list(map(float, selected["descent_joints"]))
            lift_q = list(map(float, selected["lift_joints"]))
            self.set_phase("PREGRASP")
            self.move_arm(pre_q, float(self.m["pregrasp_duration_sec"]), "PREGRASP_EXEC_FAIL")
            self.wait_arm_settle(pre_q, "PREGRASP_EXEC_FAIL")
            self.set_phase("DESCENT")
            self.move_arm_path(selected["descent_path"], float(self.m["descent_duration_sec"]), "DESCENT_EXEC_FAIL")
            self.wait_arm_settle(descent_q, "DESCENT_EXEC_FAIL")
            if self.block_xyz is None or self.block_z is None:
                raise SimulationFailure("block odometry missing")
            xy = math.hypot(self.block_xyz[0] - planned_world[0], self.block_xyz[1] - planned_world[1])
            zerr = abs(self.block_xyz[2] - planned_world[2])
            if xy > float(self.v["pregrip_xy_disturbance_m"]) or zerr > float(self.v["pregrip_z_disturbance_m"]):
                raise TestFailure("DESCENT_EXEC_FAIL", f"block disturbed before grip xy={xy:.5f} zerr={zerr:.5f}")
            block_z_before = float(self.block_z)
            result["block_z_before"] = block_z_before
            angle, hold_target = self.dual_contact_search(result, visual, selected)
            result["contact_angle"], result["hold_target"] = angle, hold_target
            self.wait_stall(hold_target, result)
            result["stall_detected"] = True
            self.set_phase("LIFT_80_MM")
            lifted_z = self.lift_and_verify(selected["lift_path"], block_z_before)
            result["block_z_after"] = lifted_z
            result["lift_m"] = lifted_z - block_z_before
            minimum_z, held_sim, held_wall, stable_hold, max_slip = self.hold(block_z_before, lifted_z)
            result["minimum_hold_z"] = minimum_z
            result["hold_sim_sec"] = held_sim
            result["hold_wall_sec"] = held_wall
            result["stable_hold"] = stable_hold
            result["max_relative_slip_m"] = max_slip
            self.set_phase("PLACE_DOWN")
            self.place_down(descent_q, block_z_before)
            self.set_phase("OPEN_GRIPPER")
            self.open_gripper()
            self.set_phase("RETREAT_PREGRASP")
            self.move_arm(pre_q, float(self.m["retreat_duration_sec"]), "RESET_FAIL")
            self.set_phase("ARM_HOME")
            self.move_arm(list(map(float, self.c["initial_pose"])), float(self.m["initial_duration_sec"]), "RESET_FAIL")
            self.set_phase("BLOCK_RESET")
            self.reset_block()
            result["reset_success"] = True
            result["success"] = stable_hold
            if stable_hold:
                self.set_phase("SUCCESS")
                print(f"TRIAL={trial_id:03d} RESULT=SUCCESS", flush=True)
            else:
                result["failure_reason"] = "NO_VALID_GRASP"
                result["failure_detail"] = (
                    f"relative slip {max_slip:.6f} exceeds "
                    f"{float(self.v['maximum_relative_slip_m']):.6f}")
                self.set_phase("UNSTABLE_HOLD")
                print(f"TRIAL={trial_id:03d} RESULT=FAIL REASON=NO_VALID_GRASP "
                      f"DETAIL={result['failure_detail']}", flush=True)
        except TestFailure as exc:
            if not started and formal:
                raise
            reason = exc.reason if exc.reason in FAILURE_REASONS else "SIMULATION_FAILURE"
            result["failure_reason"] = reason
            result["failure_detail"] = exc.detail or (exc.reason if reason != exc.reason else "")
            reset_ok, reset_detail = self.safe_recover()
            result["reset_success"] = reset_ok
            if reset_detail:
                result["failure_detail"] = (str(result["failure_detail"]) + "; recovery=" + reset_detail).strip("; ")
            print(f"TRIAL={trial_id:03d} RESULT=FAIL REASON={reason} DETAIL={result['failure_detail']}", flush=True)
        except Exception as exc:
            if not started and formal:
                raise
            result["failure_reason"] = "SIMULATION_FAILURE"
            result["failure_detail"] = repr(exc)
            reset_ok, reset_detail = self.safe_recover()
            result["reset_success"] = reset_ok
            if reset_detail:
                result["failure_detail"] += "; recovery=" + reset_detail
            print(f"TRIAL={trial_id:03d} RESULT=FAIL REASON=SIMULATION_FAILURE DETAIL={exc!r}", flush=True)
        result["maximum_penetration_m"] = self.max_penetration
        result["end_wall_time"] = now_text()
        result["cycle_wall_sec"] = time.monotonic() - start_monotonic
        return result


def stratified_candidates(config: dict):
    """Yield deterministic position-only samples, cycling through all strata."""
    sampling = config["sampling"]
    rng = random.Random(int(sampling["shuffle_seed"]))
    x0, x1 = map(float, sampling["base_x_m"])
    y0, y1 = map(float, sampling["base_y_m"])
    rows, columns = int(sampling["rows"]), int(sampling["columns"])
    strata = [(row, col) for row in range(rows) for col in range(columns)]
    attempt = 0
    while attempt < int(sampling["maximum_preflight_attempts"]):
        order = list(strata)
        rng.shuffle(order)
        for row, col in order:
            attempt += 1
            x = x0 + (row + rng.random()) * (x1 - x0) / rows
            y = y0 + (col + rng.random()) * (y1 - y0) / columns
            yield attempt, row + 1, col + 1, [x, y, float(config["block_size_m"][2]) / 2.0]
            if attempt >= int(sampling["maximum_preflight_attempts"]):
                return


def case_from_base(config: dict, base: list[float], row: int, col: int,
                   repeat: int, case_id: str) -> dict[str, object]:
    spawn = list(map(float, config["robot_spawn_world"]))
    z_world = float(config["table_top_world_z"]) + float(config["block_size_m"][2]) / 2.0
    return {"case_id": case_id, "grid_row": row, "grid_col": col, "repeat": repeat,
            "planned_base_xyz": list(map(float, base)),
            "planned_world_xyz": [spawn[0] + base[0], spawn[1] + base[1], z_world]}


def write_test_cases(root: Path, config: dict, cases: list[dict[str, object]],
                     rejections: list[dict[str, object]]) -> None:
    for trial_id, case in enumerate(cases, 1):
        case["trial_id"] = trial_id
    payload = {
        "schema_version": 2, "selection": "stratified_complete_pose_preflight",
        "shuffle_seed": int(config["sampling"]["shuffle_seed"]),
        "block_yaw_deg": float(config["sampling"]["block_yaw_deg"]),
        "case_count": len(cases), "cases": cases,
    }
    path = root / "test_cases.json"
    if path.exists() and (root / "frozen_config.yaml").exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("frozen test_cases.json differs from preflight output")
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    atomic_text(root / "preflight_rejections.jsonl",
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejections))


def nominal_visual(base: list[float], config: dict) -> dict[str, object]:
    side = list(map(float, config["block_size_m"][:2]))
    return {"xyz": list(base), "surface": [base[0], base[1], float(config["block_size_m"][2])],
            "std": [0.0, 0.0, 0.0], "confidence": 1.0, "angle_rad": 0.0,
            "angle_std_rad": 0.0, "dimensions_m": side,
            "corners_xy": []}


def load_test_cases(root: Path) -> list[dict[str, object]]:
    path = root / "test_cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 50 or [int(case["trial_id"]) for case in cases] != list(range(1, 51)):
        raise RuntimeError("frozen test_cases.json must contain trial IDs 1..50")
    if len({str(case["case_id"]) for case in cases}) != 50:
        raise RuntimeError("test case IDs are not unique")
    return cases


def run_preflight(node: VisualGrasp, root: Path, config: dict) -> list[dict[str, object]]:
    node.check_interfaces()
    node.open_gripper()
    node.move_arm(list(map(float, config["initial_pose"])),
                  float(config["motion"]["initial_duration_sec"]), "INTERFACE_NOT_READY")
    needed_per = int(config["sampling"]["samples_per_stratum"])
    accepted_counts: dict[tuple[int, int], int] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for attempt, row, col, base in stratified_candidates(config):
        if accepted_counts.get((row, col), 0) >= needed_per:
            continue
        try:
            selected = node.solve_candidates(nominal_visual(base, config))
            repeat = accepted_counts.get((row, col), 0) + 1
            accepted_counts[(row, col)] = repeat
            case = case_from_base(config, base, row, col, repeat,
                                  f"r{row}c{col}_sample{repeat}")
            case["preflight"] = {
                "status": "complete_pose_reachable", "tilt_deg": selected["tilt_deg"],
                "contour_orientation_deg": selected["offset_deg"],
                "minimum_joint_margin_rad": selected["minimum_joint_margin_rad"],
                "trajectory_waypoint_count": selected["trajectory_waypoint_count"],
            }
            accepted.append(case)
            print(f"PREFLIGHT=ACCEPT attempt={attempt} stratum={row},{col} base={base}", flush=True)
        except TestFailure as exc:
            rejected.append({"attempt": attempt, "grid_row": row, "grid_col": col,
                             "planned_base_xyz": base, "failure_reason": exc.reason,
                             "failure_detail": exc.detail})
            print(f"PREFLIGHT=REJECT attempt={attempt} stratum={row},{col} reason={exc.reason} "
                  f"detail={exc.detail[:800]}", flush=True)
        if len(accepted) == int(config["sampling"]["rows"]) * int(config["sampling"]["columns"]) * needed_per:
            break
    expected = int(config["sampling"]["rows"]) * int(config["sampling"]["columns"]) * needed_per
    if len(accepted) != expected:
        atomic_text(root / "preflight_rejections.jsonl",
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejected))
        raise TestFailure("OUT_OF_REACHABLE_REGION", f"accepted {len(accepted)}/{expected}")
    random.Random(int(config["sampling"]["shuffle_seed"])).shuffle(accepted)
    write_test_cases(root, config, accepted, rejected)
    atomic_text(root / "preflight_summary.json", json.dumps({
        "accepted": len(accepted), "rejected": len(rejected),
        "failure_counts": {reason: sum(item["failure_reason"] == reason for item in rejected)
                           for reason in sorted({item["failure_reason"] for item in rejected})},
    }, indent=2, ensure_ascii=False) + "\n")
    return accepted


def representative_cases(config: dict) -> list[dict[str, object]]:
    output = []
    for index, (name, base) in enumerate(config["representative_points"].items(), 1):
        output.append(case_from_base(config, list(map(float, base)), index, 1, 0, f"representative_{name}"))
    return output


def representative_gate(root: Path) -> tuple[bool, dict[str, object]]:
    reports = sorted((root / "results/representative").glob("run_*/report.json"))
    if not reports:
        return False, {"reason": "no representative report"}
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    results = report["results"]
    planned = all(item.get("pregrasp_joints") and item.get("descent_joints") and item.get("lift_joints")
                  for item in results)
    by_name = {str(item["case_id"]).removeprefix("representative_"): item for item in results}
    lateral = all(bool(by_name.get(name, {}).get("success")) for name in ("left", "right"))
    return planned and lateral, {"fully_planned_all_five": planned,
                                 "left_valid_lift_hold": bool(by_name.get("left", {}).get("success")),
                                 "right_valid_lift_hold": bool(by_name.get("right", {}).get("success")),
                                 "report": str(reports[-1].relative_to(root))}


def immutable_artifacts(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for path in root.iterdir():
        if path.is_file() and (path.suffix in {".py", ".sh", ".yaml"} or path.name in {"gui.config", "test_cases.json"}):
            if path.name not in {"frozen_config.yaml", "frozen_config.sha256", "frozen_manifest.sha256"}:
                paths.add(path)
    for directory in (root / "simulations", root / "generated", root / "moveit_v5"):
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def freeze(root: Path, config: dict) -> str:
    path = root / "frozen_config.yaml"
    hash_path = root / "frozen_config.sha256"
    manifest_path = root / "frozen_manifest.sha256"
    if path.exists() or hash_path.exists() or manifest_path.exists():
        raise RuntimeError("frozen configuration already exists; refusing overwrite")
    frozen = json.loads(json.dumps(config))
    frozen["frozen_test_cases_sha256"] = sha256(root / "test_cases.json")
    atomic_text(path, yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True))
    digest = sha256(path)
    atomic_text(hash_path, f"{digest}  frozen_config.yaml\n")
    lines = [f"{sha256(artifact)}  {artifact.relative_to(root).as_posix()}"
             for artifact in immutable_artifacts(root)]
    atomic_text(manifest_path, "\n".join(lines) + "\n")
    return digest


def verify_frozen(root: Path, config_path: Path, config: dict) -> str:
    expected_file = root / "frozen_config.sha256"
    manifest = root / "frozen_manifest.sha256"
    if (config_path.resolve() != (root / "frozen_config.yaml").resolve()
            or not expected_file.is_file() or not manifest.is_file()):
        raise SimulationFailure("batch requires frozen config and manifest")
    actual = sha256(config_path)
    expected = expected_file.read_text(encoding="utf-8").split()[0]
    if actual != expected:
        raise SimulationFailure("frozen config SHA256 mismatch")
    if sha256(root / "test_cases.json") != str(config.get("frozen_test_cases_sha256", "")):
        raise SimulationFailure("frozen test cases changed")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        artifact = root / name.strip()
        if not artifact.is_file() or sha256(artifact) != digest:
            raise SimulationFailure(f"frozen artifact changed: {name.strip()}")
    return actual


def save_formal_result(root: Path, result: dict) -> None:
    results = root / "results"
    trial_path = results / f"trial_{int(result['trial_id']):03d}.json"
    if trial_path.exists():
        raise RuntimeError(f"refusing to overwrite {trial_path}")
    atomic_text(trial_path, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    marker = results / "in_progress.json"
    if marker.exists():
        marker.unlink()
    rebuild_tables(results)


def recover_stale_marker(root: Path, config_hash: str, restart: int) -> None:
    marker_path = root / "results/in_progress.json"
    if not marker_path.is_file():
        return
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    trial_id = int(marker["trial_id"])
    trial_path = root / "results" / f"trial_{trial_id:03d}.json"
    if trial_path.exists():
        marker_path.unlink()
        rebuild_tables(root / "results")
        return
    archive = root / "results/interrupted"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    marker_path.replace(archive / f"trial_{trial_id:03d}_{stamp}.json")


def next_trial_id(root: Path) -> int:
    ids = [int(path.stem.split("_")[1]) for path in (root / "results").glob("trial_[0-9][0-9][0-9].json")]
    return next((trial_id for trial_id in range(1, 51) if trial_id not in ids), 51)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "representative", "pilot", "freeze", "batch"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sim-restart-index", type=int, default=0)
    parser.add_argument("--representative-name", choices=["center", "left", "right", "near", "far"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    if args.mode == "freeze":
        gate, detail = representative_gate(root)
        if not gate:
            raise RuntimeError(f"representative gate failed: {detail}")
        load_test_cases(root)
        digest = freeze(root, config)
        print(f"DAY3_V5_FROZEN_CONFIG_SHA256={digest} GATE={json.dumps(detail, ensure_ascii=False)}")
        return 0
    if args.mode == "batch":
        config_hash = verify_frozen(root, config_path, config)
        if not args.resume and any((root / "results").glob("trial_[0-9][0-9][0-9].json")):
            raise RuntimeError("formal results exist; use --resume")
        recover_stale_marker(root, config_hash, args.sim_restart_index)
    rclpy.init()
    node = VisualGrasp(config, root, config_hash)
    try:
        if args.mode == "preflight":
            cases = run_preflight(node, root, config)
            print(f"DAY3_V5_PREFLIGHT_ACCEPTED={len(cases)} SHA256={sha256(root / 'test_cases.json')}")
            return 0
        if args.mode in {"representative", "pilot"}:
            run_dir = root / "results/representative" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=False)
            results = []
            selected_cases = representative_cases(config)
            if args.representative_name:
                selected_cases = [case for case in selected_cases
                                  if case["case_id"] == f"representative_{args.representative_name}"]
            for trial_id, case in enumerate(selected_cases, 1):
                result = node.trial(trial_id, args.sim_restart_index, case, formal=False)
                results.append(result)
                atomic_text(run_dir / f"{case['case_id']}.json",
                            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            atomic_text(run_dir / "report.json", json.dumps({"results": results}, indent=2,
                                                             ensure_ascii=False) + "\n")
            gate, detail = representative_gate(root)
            print(f"DAY3_V5_REPRESENTATIVE_GATE={'PASS' if gate else 'FAIL'} {json.dumps(detail, ensure_ascii=False)}")
            return 0 if gate else 1
        cases = load_test_cases(root)
        while next_trial_id(root) <= args.trials:
            trial_id = next_trial_id(root)
            verify_frozen(root, config_path, config)
            result = node.trial(trial_id, args.sim_restart_index, cases[trial_id - 1], formal=True)
            save_formal_result(root, result)
            print(f"DAY3_V5_TRIAL_COMPLETED={trial_id} SUCCESS={int(bool(result['success']))}", flush=True)
        print(f"DAY3_V5_TRIALS_COMPLETED={args.trials}", flush=True)
        return 0
    except TestFailure as exc:
        print(f"DAY3_V5_RUN=FAIL: {exc.reason}: {exc.detail}", file=sys.stderr, flush=True)
        return 75 if exc.reason in {"SIMULATION_FAILURE", "INTERFACE_NOT_READY", "CONTROLLER_NOT_ACTIVE"} else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
