#!/usr/bin/env python3
"""Geometry and URDF helpers shared by the Day3-v5 grasp pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def normalize_square_angle(angle_rad: float) -> float:
    """Return a square edge angle in [-pi/4, pi/4), modulo 90 degrees."""
    period = math.pi / 2.0
    return (float(angle_rad) + period / 2.0) % period - period / 2.0


def square_angle_distance(first: float, second: float) -> float:
    return abs(normalize_square_angle(float(first) - float(second)))


def square_circular_mean(angles: list[float]) -> tuple[float, float]:
    """Mean and circular standard deviation for an orientation modulo pi/2."""
    if not angles:
        raise ValueError("at least one square angle is required")
    values = np.asarray(angles, dtype=float) * 4.0
    sine = float(np.mean(np.sin(values)))
    cosine = float(np.mean(np.cos(values)))
    resultant = max(1e-12, min(1.0, math.hypot(sine, cosine)))
    mean = normalize_square_angle(math.atan2(sine, cosine) / 4.0)
    std = math.sqrt(max(0.0, -2.0 * math.log(resultant))) / 4.0
    return mean, std


def ordered_rectangle(points_xy: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Fit an OpenCV rectangle and return ordered corners and square-symmetric yaw."""
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("at least four points are required")
    rectangle = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rectangle).astype(float)
    center = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))
    corners = corners[order]
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    angle = normalize_square_angle(math.atan2(float(edge[1]), float(edge[0])))
    dimensions = tuple(sorted((float(rectangle[1][0]), float(rectangle[1][1])), reverse=True))
    return corners, angle, dimensions


def matrix_to_quaternion(rotation: np.ndarray) -> list[float]:
    return Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat().astype(float).tolist()


def quaternion_to_matrix(quaternion_xyzw: list[float]) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=float)).as_matrix()


def pose_error(actual_xyz: list[float], actual_q: list[float], desired_xyz: list[float],
               desired_q: list[float]) -> tuple[float, float]:
    position = float(np.linalg.norm(np.asarray(actual_xyz) - np.asarray(desired_xyz)))
    dot = abs(float(np.dot(np.asarray(actual_q), np.asarray(desired_q))))
    orientation = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    return position, orientation


def grasp_rotation(center_xy: list[float], contour_angle: float, tilt_rad: float,
                   lean_sign: int, symmetry_index: int) -> np.ndarray:
    """Construct a reachable 5-DOF top-grasp orientation.

    The arm approach plane may be radial, but the projected finger closing axis is
    aligned to an OpenCV contour edge.  Target position azimuth never selects the
    closing direction.
    """
    azimuth = math.atan2(float(center_xy[1]), float(center_xy[0]))
    radial = np.array([math.cos(azimuth), math.sin(azimuth), 0.0], dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    tool_z = math.cos(float(tilt_rad)) * down + int(lean_sign) * math.sin(float(tilt_rad)) * radial
    tool_z /= np.linalg.norm(tool_z)
    axis_angle = float(contour_angle) + int(symmetry_index) * math.pi / 2.0
    desired_closing = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0], dtype=float)
    tool_y = desired_closing - float(np.dot(desired_closing, tool_z)) * tool_z
    norm = float(np.linalg.norm(tool_y))
    if norm < 1e-9:
        raise ValueError("closing axis is parallel to approach axis")
    tool_y /= norm
    tool_x = np.cross(tool_y, tool_z)
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    return np.column_stack((tool_x, tool_y, tool_z))


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic: tuple[str, float, float] | None


@dataclass(frozen=True)
class BoxCollision:
    link: str
    name: str
    origin: np.ndarray
    size: np.ndarray


def _origin(node: ET.Element | None) -> np.ndarray:
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if node is not None:
        xyz = [float(x) for x in node.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in node.get("rpy", "0 0 0").split()]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


class URDFGeometry:
    """Small deterministic FK reader; it never changes the robot description."""

    def __init__(self, path: Path):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self.joints: dict[str, Joint] = {}
        self.by_child: dict[str, Joint] = {}
        self.boxes: dict[tuple[str, str], BoxCollision] = {}
        for node in root.findall("joint"):
            name = str(node.get("name"))
            parent = str(node.find("parent").get("link"))
            child = str(node.find("child").get("link"))
            axis_node = node.find("axis")
            axis = np.asarray([float(x) for x in axis_node.get("xyz", "0 0 0").split()]
                              if axis_node is not None else [0.0, 0.0, 0.0], dtype=float)
            limit = node.find("limit")
            lower = float(limit.get("lower", "-inf")) if limit is not None else -math.inf
            upper = float(limit.get("upper", "inf")) if limit is not None else math.inf
            mimic_node = node.find("mimic")
            mimic = None
            if mimic_node is not None:
                mimic = (str(mimic_node.get("joint")), float(mimic_node.get("multiplier", "1")),
                         float(mimic_node.get("offset", "0")))
            joint = Joint(name, parent, child, str(node.get("type")), _origin(node.find("origin")),
                          axis, lower, upper, mimic)
            self.joints[name] = joint
            self.by_child[child] = joint
        for link in root.findall("link"):
            link_name = str(link.get("name"))
            for collision in link.findall("collision"):
                box = collision.find("./geometry/box")
                if box is None:
                    continue
                size = np.asarray([float(x) for x in box.get("size", "").split()], dtype=float)
                name = str(collision.get("name", ""))
                self.boxes[(link_name, name)] = BoxCollision(
                    link_name, name, _origin(collision.find("origin")), size)

    def joint_limits(self, names: list[str]) -> dict[str, tuple[float, float]]:
        return {name: (self.joints[name].lower, self.joints[name].upper) for name in names}

    def _value(self, joint: Joint, positions: dict[str, float]) -> float:
        if joint.mimic is None:
            return float(positions.get(joint.name, 0.0))
        master, multiplier, offset = joint.mimic
        return multiplier * float(positions.get(master, 0.0)) + offset

    def transform(self, root_link: str, target_link: str, positions: dict[str, float]) -> np.ndarray:
        path: list[Joint] = []
        link = target_link
        while link != root_link:
            if link not in self.by_child:
                raise ValueError(f"{target_link} is not below {root_link}")
            joint = self.by_child[link]
            path.append(joint)
            link = joint.parent
        result = np.eye(4)
        for joint in reversed(path):
            result = result @ joint.origin
            if joint.kind in {"revolute", "continuous"}:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(
                    joint.axis / max(float(np.linalg.norm(joint.axis)), 1e-12)
                    * self._value(joint, positions)).as_matrix()
                result = result @ rotation
            elif joint.kind == "prismatic":
                translation = np.eye(4)
                translation[:3, 3] = joint.axis * self._value(joint, positions)
                result = result @ translation
        return result

    def box_transform(self, root_link: str, box: BoxCollision,
                      positions: dict[str, float]) -> np.ndarray:
        return self.transform(root_link, box.link, positions) @ box.origin

    def finger_geometry(self, r_position: float, root_link: str = "link5") -> dict[str, object]:
        positions = {"r_joint": float(r_position)}
        left = self.boxes[("l_out_link", "l_out_link_box_collision")]
        right = self.boxes[("r_out_link", "r_out_link_box_collision")]
        left_t = self.box_transform(root_link, left, positions)
        right_t = self.box_transform(root_link, right, positions)
        left_center = left_t[:3, 3]
        right_center = right_t[:3, 3]
        axis = left_center - right_center
        axis /= np.linalg.norm(axis)

        def support(transform: np.ndarray, size: np.ndarray) -> float:
            return float(np.sum(np.abs(transform[:3, :3].T @ axis) * size * 0.5))

        left_radius = support(left_t, left.size)
        right_radius = support(right_t, right.size)
        left_inner = left_center - axis * left_radius
        right_inner = right_center + axis * right_radius
        gap = float(np.dot(left_inner - right_inner, axis))
        midpoint = (left_inner + right_inner) * 0.5
        return {
            "gap_m": gap,
            "midpoint_link5_m": midpoint.astype(float).tolist(),
            "closing_axis_link5": axis.astype(float).tolist(),
            "left_inner_link5_m": left_inner.astype(float).tolist(),
            "right_inner_link5_m": right_inner.astype(float).tolist(),
        }

    def contact_angle_for_width(self, width_m: float) -> tuple[float, dict[str, object]]:
        lower, upper = self.joints["r_joint"].lower, self.joints["r_joint"].upper
        target = float(width_m)

        def residual(value: float) -> float:
            return float(self.finger_geometry(value)["gap_m"]) - target

        low_value, high_value = residual(lower), residual(upper)
        if low_value == 0.0:
            angle = lower
        elif high_value == 0.0:
            angle = upper
        elif low_value * high_value < 0.0:
            angle = float(brentq(residual, lower, upper, xtol=1e-10))
        else:
            angle = min((lower, upper), key=lambda value: abs(residual(value)))
        return angle, self.finger_geometry(angle)

    def tcp_offset_for_width(self, width_m: float, tcp_link: str = "end_effector_link") -> tuple[float, list[float]]:
        angle, geometry = self.contact_angle_for_width(width_m)
        link5_to_tcp = self.transform("link5", tcp_link, {})
        tcp_to_link5 = np.linalg.inv(link5_to_tcp)
        midpoint = np.ones(4)
        midpoint[:3] = np.asarray(geometry["midpoint_link5_m"], dtype=float)
        offset = (tcp_to_link5 @ midpoint)[:3]
        return angle, offset.astype(float).tolist()

