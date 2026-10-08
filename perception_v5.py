#!/usr/bin/env python3
"""Day3-v5 OpenCV RGB-D red-block center estimator in base_link."""

import json
import threading
from functools import wraps
from rcl_interfaces.msg import SetParametersResult
from day8_env import PerceptionNoise

import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32
from tf2_ros import Buffer, TransformException, TransformListener

from grasp_geometry_v5 import ordered_rectangle


def domain_locked(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self.domain_lock:
            return function(self, *args, **kwargs)
    return wrapped


class Perception(Node):

    def __init__(self):
        super().__init__('perception')

        self.domain_lock = threading.RLock()
        self.domain_noise = PerceptionNoise()
        self.domain_cutoff_ns = -1
        defaults = {
            'day8_domain': '',
            'rgb_topic': '/depth_cam/rgbd/image',
            'depth_topic': '/depth_cam/rgbd/depth_image',
            'camera_info_topic': '/depth_cam/rgbd/camera_info',
            'base_frame': 'base_link',
            'camera_frame': 'depth_cam_frame',
            'camera_frame_is_optical': False,
            'optical_to_camera_rotation': [
                0.0, -1.0, 0.0,
                0.0, 0.0, -1.0,
                1.0, 0.0, 0.0,
            ],
            'target_topic': '/grasp/target_point',
            'surface_topic': '/grasp/target_surface',
            'footprint_topic': '/grasp/target_footprint',
            'confidence_topic': '/grasp/target_confidence',
            'detected_topic': '/grasp/detected',
            'debug_topic': '/grasp/debug_image',
            'object_height_m': 0.03,
            'red_low_1': [0, 120, 60],
            'red_high_1': [8, 255, 255],
            'red_low_2': [170, 120, 60],
            'red_high_2': [179, 255, 255],
            'morphology_kernel_px': 5,
            'min_contour_area_px': 100.0,
            'max_contour_area_ratio': 0.20,
            'depth_patch_radius_px': 6,
            'min_depth_pixels': 12,
            'min_depth_m': 0.10,
            'max_depth_m': 1.50,
            'sync_slop_sec': 0.12,
            'tf_timeout_sec': 0.20,
            'temporal_scale_m': 0.02,
            'min_object_side_m': 0.025,
            'max_object_side_m': 0.040,
            'publish_debug': True,
            'validation_frames': 0,
            'report_path': str(
                Path(__file__).resolve().parent
                / 'results/perception_validation.txt'
            ),
            'expected_center_base_m': [0.32, 0.0, 0.015],
            'center_tolerance_m': [0.04, 0.04, 0.025],
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.add_on_set_parameters_callback(self.update_domain)
        initial_domain = self.get_parameter('day8_domain')
        result = self.update_domain([initial_domain])
        if not result.successful:
            raise ValueError(result.reason)

        p = lambda name: self.get_parameter(name).value

        self.rgb_topic = str(p('rgb_topic'))
        self.depth_topic = str(p('depth_topic'))
        self.info_topic = str(p('camera_info_topic'))
        self.base_frame = str(p('base_frame'))
        self.camera_frame = str(p('camera_frame'))

        self.camera_frame_is_optical = bool(
            p('camera_frame_is_optical')
        )

        rotation_values = np.asarray(
            p('optical_to_camera_rotation'),
            dtype=float,
        )

        if rotation_values.size != 9:
            raise ValueError(
                'optical_to_camera_rotation '
                'must contain 9 values'
            )

        self.optical_to_camera_rotation = (
            rotation_values.reshape(3, 3)
        )

        rotation_check = (
            self.optical_to_camera_rotation.T
            @ self.optical_to_camera_rotation
        )

        determinant = float(
            np.linalg.det(
                self.optical_to_camera_rotation
            )
        )

        if not np.allclose(
            rotation_check,
            np.eye(3),
            atol=1e-6,
        ):
            raise ValueError(
                'optical_to_camera_rotation '
                'is not orthonormal'
            )

        if not np.isclose(
            determinant,
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                'optical_to_camera_rotation '
                'must have determinant +1'
            )

        self.object_height = float(p('object_height_m'))

        self.red_low_1 = np.asarray(p('red_low_1'), np.uint8)
        self.red_high_1 = np.asarray(p('red_high_1'), np.uint8)
        self.red_low_2 = np.asarray(p('red_low_2'), np.uint8)
        self.red_high_2 = np.asarray(p('red_high_2'), np.uint8)

        self.kernel_px = max(1, int(p('morphology_kernel_px')))
        if self.kernel_px % 2 == 0:
            self.kernel_px += 1

        self.min_area = float(p('min_contour_area_px'))
        self.max_area_ratio = float(p('max_contour_area_ratio'))
        self.patch_radius = max(1, int(p('depth_patch_radius_px')))
        self.min_depth_pixels = max(1, int(p('min_depth_pixels')))
        self.min_depth = float(p('min_depth_m'))
        self.max_depth = float(p('max_depth_m'))
        self.tf_timeout = float(p('tf_timeout_sec'))
        self.temporal_scale = max(
            1e-6,
            float(p('temporal_scale_m')),
        )
        self.min_object_side = float(p('min_object_side_m'))
        self.max_object_side = float(p('max_object_side_m'))

        self.publish_debug = bool(p('publish_debug'))
        self.validation_frames = max(
            0,
            int(p('validation_frames')),
        )

        self.report_path = Path(
            str(p('report_path'))
        ).expanduser()

        self.expected_center = np.asarray(
            p('expected_center_base_m'),
            float,
        )

        self.center_tolerance = np.asarray(
            p('center_tolerance_m'),
            float,
        )

        self.bridge = CvBridge()
        self.info = None
        self.previous_center = None

        self.start_wall = time.monotonic()
        self.last_log_wall = 0.0

        self.done = False
        self.shutdown_timer = None

        self.total = 0
        self.detected = 0
        self.tf_failures = 0
        self.tf_latest_fallbacks = 0

        self.centers = []
        self.confidences = []

        self.target_pub = self.create_publisher(
            PointStamped,
            str(p('target_topic')),
            10,
        )

        self.surface_pub = self.create_publisher(
            PointStamped,
            str(p('surface_topic')),
            10,
        )

        self.footprint_pub = self.create_publisher(
            PolygonStamped,
            str(p('footprint_topic')),
            10,
        )

        self.conf_pub = self.create_publisher(
            Float32,
            str(p('confidence_topic')),
            10,
        )

        self.detected_pub = self.create_publisher(
            Bool,
            str(p('detected_topic')),
            10,
        )

        self.debug_pub = self.create_publisher(
            Image,
            str(p('debug_topic')),
            2,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.info_cb,
            qos_profile_sensor_data,
        )

        self.rgb_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=float(p('sync_slop_sec')),
        )

        self.sync.registerCallback(self.rgbd_cb)

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=5.0)
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.get_logger().info(
            f'Day 2 perception started: '
            f'{self.rgb_topic} + {self.depth_topic}; '
            f'{self.camera_frame} -> {self.base_frame}'
        )

        if self.validation_frames:
            self.get_logger().info(
                f'Automatic validation: '
                f'{self.validation_frames} synchronized frames'
            )

    def info_cb(self, msg):
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            return

        first = self.info is None
        self.info = msg

        if first:
            self.get_logger().info(
                f'CameraInfo: {msg.width}x{msg.height}, '
                f'fx={msg.k[0]:.2f}, '
                f'fy={msg.k[4]:.2f}, '
                f'cx={msg.k[2]:.2f}, '
                f'cy={msg.k[5]:.2f}'
            )

    def update_domain(self, parameters):
        for parameter in parameters:
            if parameter.name != 'day8_domain':
                continue
            try:
                payload = json.loads(parameter.value) if parameter.value else None
                if payload:
                    json.dumps(payload, allow_nan=False)
                    for key in ('brightness', 'contrast', 'pixel_std', 'depth_std_m',
                                'invalid_fraction', 'translation_m', 'rotation_rad', 'noise_seed', 'enabled'):
                        if key not in payload:
                            raise ValueError('missing ' + key)
                    if not 0 <= payload['invalid_fraction'] <= 1:
                        raise ValueError('invalid pixel fraction')
                    for key in ('translation_m', 'rotation_rad'):
                        if np.asarray(payload[key]).shape != (3,):
                            raise ValueError('invalid extrinsic vector')
                noise = PerceptionNoise(payload)
                with self.domain_lock:
                    self.domain_noise = noise
                    self.previous_center = None
                    # Reject queued pre-update images without touching synchronizer
                    # internals or acquiring its lock in the opposite order.
                    self.domain_cutoff_ns = self.get_clock().now().nanoseconds
                return SetParametersResult(successful=True)
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    @domain_locked
    def rgbd_cb(self, rgb_msg, depth_msg):
        if self.done or self.info is None:
            return
        if Time.from_msg(rgb_msg.header.stamp).nanoseconds <= self.domain_cutoff_ns:
            return

        self.total += 1

        try:
            bgr = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                'bgr8',
            )

            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                'passthrough',
            )

        except Exception as exc:
            self.miss()
            self.get_logger().error(
                f'Image conversion failed: {exc}'
            )
            self.finish_if_ready()
            return

        encoding = depth_msg.encoding.upper()

        if encoding == '16UC1':
            depth_m = (
                np.asarray(depth, np.float32) * 0.001
            )

        elif encoding == '32FC1':
            depth_m = np.asarray(
                depth,
                np.float32,
            )

        else:
            self.miss()
            self.get_logger().error(
                f'Unsupported depth encoding: '
                f'{depth_msg.encoding}'
            )
            self.finish_if_ready()
            return

        bgr, depth_m = self.domain_noise.images(bgr, depth_m)

        result = self.detect_red(
            bgr,
            depth_m,
        )

        if result is None:
            self.miss()

            self.annotate(
                bgr,
                'RED BLOCK NOT DETECTED',
                False,
            )

            self.publish_debug_image(
                bgr,
                rgb_msg,
            )

            self.periodic_log(
                'NOT DETECTED'
            )

            self.finish_if_ready()
            return

        (
            contour,
            mask,
            u,
            v,
            depth_value,
            depth_ratio,
            solidity,
        ) = result

        optical_xyz = self.back_project(
            u,
            v,
            depth_value,
            rgb_msg.width,
            rgb_msg.height,
        )

        if self.camera_frame_is_optical:
            camera_xyz = optical_xyz

        else:
            # Fixed extrinsic rotation for the current
            # Gazebo robot_cam mounting:
            #
            # x_depth = -y_optical
            # y_depth = -z_optical
            # z_depth =  x_optical
            #
            # This is mount-specific and is not the
            # generic ROS camera_link conversion.
            camera_xyz = (
                self.optical_to_camera_rotation
                @ optical_xyz
            )

        surface = self.to_base(
            camera_xyz,
            rgb_msg,
        )

        # The v3 detector back-projected the 2-D contour centroid.  With an
        # oblique camera that centroid includes a visible side face and biases
        # x by roughly a centimetre.  Keep the same OpenCV colour segmentation,
        # but reconstruct all valid red pixels and estimate the horizontal top
        # plane in base_link.  No simulator pose is used here.
        cloud_geometry = self.top_geometry_from_mask(mask, depth_m, rgb_msg)
        if cloud_geometry is not None:
            surface, footprint, contour_angle, contour_dimensions, orientation_quality = cloud_geometry
        else:
            footprint = None
            contour_angle = float('nan')
            contour_dimensions = (float('nan'), float('nan'))
            orientation_quality = 0.0

        if surface is None or footprint is None:
            self.tf_failures += 1
            self.miss()

            self.annotate(
                bgr,
                'TF UNAVAILABLE',
                False,
            )

            self.publish_debug_image(
                bgr,
                rgb_msg,
            )

            self.finish_if_ready()
            return

        center = surface.copy()
        center[2] -= self.object_height / 2.0

        if self.previous_center is None:
            temporal = 0.5
        else:
            displacement = float(
                np.linalg.norm(
                    center - self.previous_center
                )
            )

            temporal = math.exp(
                -displacement / self.temporal_scale
            )

        self.previous_center = center.copy()

        confidence = float(
            np.clip(
                0.30 * solidity
                + 0.30 * depth_ratio
                + 0.20 * temporal
                + 0.20 * orientation_quality,
                0.0,
                1.0,
            )
        )

        self.publish_point(
            self.surface_pub,
            surface,
            rgb_msg,
        )

        footprint_msg = PolygonStamped()
        footprint_msg.header.stamp = rgb_msg.header.stamp
        footprint_msg.header.frame_id = self.base_frame
        footprint_msg.polygon.points = [
            Point32(x=float(point[0]), y=float(point[1]), z=float(surface[2]))
            for point in footprint
        ]
        self.footprint_pub.publish(footprint_msg)

        self.publish_point(
            self.target_pub,
            center,
            rgb_msg,
        )

        self.conf_pub.publish(
            Float32(data=confidence)
        )

        self.detected_pub.publish(
            Bool(data=True)
        )

        self.detected += 1
        self.centers.append(center.copy())
        self.confidences.append(confidence)

        cv2.drawContours(
            bgr,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

        cv2.circle(
            bgr,
            (u, v),
            5,
            (255, 255, 255),
            -1,
        )

        cv2.circle(
            bgr,
            (u, v),
            self.patch_radius,
            (255, 255, 0),
            1,
        )

        self.annotate(
            bgr,
            (
                f'center=('
                f'{center[0]:.3f},'
                f'{center[1]:.3f},'
                f'{center[2]:.3f}) m '
                f'edge={math.degrees(contour_angle):.1f}deg '
                f'conf={confidence:.2f}'
            ),
            True,
        )

        self.publish_debug_image(
            bgr,
            rgb_msg,
        )

        rate = self.total / max(
            time.monotonic() - self.start_wall,
            1e-6,
        )

        self.periodic_log(
            f'DETECTED '
            f'pixel=({u},{v}) '
            f'depth={depth_value:.3f} m '
            f'surface=('
            f'{surface[0]:.3f},'
            f'{surface[1]:.3f},'
            f'{surface[2]:.3f}) m '
            f'center=('
            f'{center[0]:.3f},'
            f'{center[1]:.3f},'
            f'{center[2]:.3f}) m '
            f'conf={confidence:.2f} '
            f'edge={math.degrees(contour_angle):.1f}deg '
            f'sides=({contour_dimensions[0]:.3f},{contour_dimensions[1]:.3f})m '
            f'sync_rate={rate:.2f} Hz '
            f'mask_px={int(np.count_nonzero(mask))}'
        )

        self.finish_if_ready()

    def detect_red(self, bgr, depth_m):
        hsv = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.bitwise_or(
            cv2.inRange(
                hsv,
                self.red_low_1,
                self.red_high_1,
            ),
            cv2.inRange(
                hsv,
                self.red_low_2,
                self.red_high_2,
            ),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.kernel_px,
                self.kernel_px,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        area = float(
            cv2.contourArea(contour)
        )

        image_area = (
            bgr.shape[0] * bgr.shape[1]
        )

        if (
            area < self.min_area
            or area
            > self.max_area_ratio * image_area
        ):
            return None

        moments = cv2.moments(contour)

        if abs(moments['m00']) < 1e-9:
            return None

        u = int(
            round(
                moments['m10']
                / moments['m00']
            )
        )

        v = int(
            round(
                moments['m01']
                / moments['m00']
            )
        )

        hull_area = float(
            cv2.contourArea(
                cv2.convexHull(contour)
            )
        )

        if hull_area:
            solidity = float(
                np.clip(
                    area / hull_area,
                    0.0,
                    1.0,
                )
            )
        else:
            solidity = 0.0

        object_mask = np.zeros(
            mask.shape,
            np.uint8,
        )

        cv2.drawContours(
            object_mask,
            [contour],
            -1,
            255,
            -1,
        )

        object_mask = cv2.erode(
            object_mask,
            np.ones(
                (3, 3),
                np.uint8,
            ),
            iterations=1,
        )

        yy, xx = np.ogrid[
            :mask.shape[0],
            :mask.shape[1],
        ]

        patch = (
            (xx - u) ** 2
            + (yy - v) ** 2
            <= self.patch_radius ** 2
        )

        candidate = (
            patch
            & (object_mask > 0)
        )

        valid = (
            candidate
            & np.isfinite(depth_m)
            & (depth_m >= self.min_depth)
            & (depth_m <= self.max_depth)
        )

        if (
            np.count_nonzero(valid)
            < self.min_depth_pixels
        ):
            candidate = object_mask > 0

            valid = (
                candidate
                & np.isfinite(depth_m)
                & (depth_m >= self.min_depth)
                & (depth_m <= self.max_depth)
            )

        valid_count = int(
            np.count_nonzero(valid)
        )

        candidate_count = int(
            np.count_nonzero(candidate)
        )

        if (
            valid_count < self.min_depth_pixels
            or candidate_count == 0
        ):
            return None

        z = float(
            np.median(depth_m[valid])
        )

        depth_ratio = float(
            np.clip(
                valid_count / candidate_count,
                0.0,
                1.0,
            )
        )

        return (
            contour,
            mask,
            u,
            v,
            z,
            depth_ratio,
            solidity,
        )

    def back_project(
        self,
        u,
        v,
        z,
        width,
        height,
    ):
        if self.info.width:
            sx = width / self.info.width
        else:
            sx = 1.0

        if self.info.height:
            sy = height / self.info.height
        else:
            sy = 1.0

        fx = self.info.k[0] * sx
        fy = self.info.k[4] * sy
        cx = self.info.k[2] * sx
        cy = self.info.k[5] * sy

        return np.array(
            [
                (u - cx) * z / fx,
                (v - cy) * z / fy,
                z,
            ],
            float,
        )

    def top_geometry_from_mask(self, mask, depth_m, image_msg):
        valid = (
            (mask > 0)
            & np.isfinite(depth_m)
            & (depth_m >= self.min_depth)
            & (depth_m <= self.max_depth)
        )
        vv, uu = np.nonzero(valid)
        if uu.size < self.min_depth_pixels:
            return None
        # Bound CPU cost deterministically without changing the estimator.
        stride = max(1, int(math.ceil(uu.size / 6000.0)))
        uu = uu[::stride].astype(float)
        vv = vv[::stride].astype(float)
        zz = depth_m[valid][::stride].astype(float)
        width = float(image_msg.width)
        height = float(image_msg.height)
        sx = width / self.info.width if self.info.width else 1.0
        sy = height / self.info.height if self.info.height else 1.0
        fx = self.info.k[0] * sx
        fy = self.info.k[4] * sy
        cx = self.info.k[2] * sx
        cy = self.info.k[5] * sy
        optical = np.column_stack(((uu - cx) * zz / fx, (vv - cy) * zz / fy, zz))
        if self.camera_frame_is_optical:
            camera = optical
        else:
            camera = optical @ self.optical_to_camera_rotation.T
        base = self.to_base(camera, image_msg)
        if base is None or len(base) < self.min_depth_pixels:
            return None
        z_reference = float(np.percentile(base[:, 2], 90.0))
        top = base[base[:, 2] >= z_reference - 0.003]
        if len(top) < self.min_depth_pixels:
            return None
        # An oblique view does not sample the square top uniformly.  The robust
        # midpoint of its reconstructed extents estimates the geometric centre
        # without using simulator truth; median z estimates the top plane.
        try:
            corners, angle, dimensions = ordered_rectangle(top[:, :2])
        except (ValueError, cv2.error):
            return None
        long_side, short_side = dimensions
        if not (self.min_object_side <= short_side <= self.max_object_side and
                self.min_object_side <= long_side <= self.max_object_side):
            return None
        surface = np.array([float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])),
                            float(np.median(top[:, 2]))], dtype=float)
        # For a square, the two axes are physically equivalent.  Quality therefore
        # measures edge support and plausible dimensions, not a misleading PCA ratio.
        hull = cv2.convexHull(top[:, :2].astype(np.float32))
        hull_area = max(float(cv2.contourArea(hull)), 1e-9)
        rectangularity = float(np.clip(hull_area / max(long_side * short_side, 1e-9), 0.0, 1.0))
        size_quality = math.exp(-abs((long_side + short_side) * 0.5 - self.object_height) / 0.015)
        quality = float(np.clip(0.65 * rectangularity + 0.35 * size_quality, 0.0, 1.0))
        return surface, corners, angle, dimensions, quality

    def to_base(
        self,
        camera_xyz,
        image_msg,
    ):
        source = (
            self.camera_frame
            or image_msg.header.frame_id
        )

        timeout = Duration(
            seconds=self.tf_timeout
        )

        try:
            transform = (
                self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source,
                    Time.from_msg(
                        image_msg.header.stamp
                    ),
                    timeout=timeout,
                )
            )

        except TransformException:
            try:
                transform = (
                    self.tf_buffer.lookup_transform(
                        self.base_frame,
                        source,
                        Time(),
                        timeout=timeout,
                    )
                )

                self.tf_latest_fallbacks += 1

            except TransformException as exc:
                self.periodic_log(
                    f'Cannot transform '
                    f'{source} -> '
                    f'{self.base_frame}: '
                    f'{exc}'
                )
                return None

        q = transform.transform.rotation

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        norm = math.sqrt(
            x * x
            + y * y
            + z * z
            + w * w
        )

        if norm < 1e-12:
            return None

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        rotation = np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ]
        )

        t = transform.transform.translation

        translation = np.array(
            [
                t.x,
                t.y,
                t.z,
            ]
        )

        rotation, translation = self.domain_noise.extrinsics(rotation, translation)
        points = np.asarray(camera_xyz, dtype=float)
        if points.ndim == 1:
            return rotation @ points + translation
        return points @ rotation.T + translation

    def publish_point(
        self,
        publisher,
        xyz,
        source_msg,
    ):
        msg = PointStamped()

        msg.header.stamp = (
            source_msg.header.stamp
        )

        msg.header.frame_id = (
            self.base_frame
        )

        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])

        publisher.publish(msg)

    def miss(self):
        self.detected_pub.publish(
            Bool(data=False)
        )

        self.conf_pub.publish(
            Float32(data=0.0)
        )

        self.previous_center = None

    @staticmethod
    def annotate(
        image,
        text,
        success,
    ):
        cv2.rectangle(
            image,
            (4, 4),
            (
                image.shape[1] - 5,
                36,
            ),
            (0, 0, 0),
            -1,
        )

        if success:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(
            image,
            text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    def publish_debug_image(
        self,
        image,
        source_msg,
    ):
        if not self.publish_debug:
            return

        msg = self.bridge.cv2_to_imgmsg(
            image,
            'bgr8',
        )

        msg.header = source_msg.header

        self.debug_pub.publish(msg)

    def periodic_log(self, text):
        now = time.monotonic()

        if now - self.last_log_wall >= 1.0:
            self.get_logger().info(text)
            self.last_log_wall = now

    def finish_if_ready(self):
        if (
            self.validation_frames <= 0
            or self.total
            < self.validation_frames
        ):
            return

        self.done = True
        self.write_report()

        self.shutdown_timer = self.create_timer(
            0.2,
            self.shutdown_after_report,
        )

    def write_report(self):
        elapsed = max(
            time.monotonic()
            - self.start_wall,
            1e-6,
        )

        detection_rate = (
            self.detected
            / max(self.total, 1)
        )

        if self.centers:
            centers = np.vstack(
                self.centers
            )

            median = np.median(
                centers,
                axis=0,
            )

            std = np.std(
                centers,
                axis=0,
            )

            median_conf = float(
                np.median(
                    self.confidences
                )
            )

            error = np.abs(
                median
                - self.expected_center
            )

            center_ok = bool(
                np.all(
                    error
                    <= self.center_tolerance
                )
            )

        else:
            median = np.full(
                3,
                np.nan,
            )

            std = np.full(
                3,
                np.nan,
            )

            error = np.full(
                3,
                np.inf,
            )

            median_conf = 0.0
            center_ok = False

        detection_ok = (
            detection_rate >= 0.95
        )

        confidence_ok = (
            median_conf >= 0.70
        )

        if (
            detection_ok
            and confidence_ok
            and center_ok
        ):
            final = 'PASS'
        else:
            final = 'NOT_PASS'

        def vec(values):
            return ','.join(
                f'{value:.6f}'
                for value in values
            )

        lines = [
            (
                'DAY 2 RGB-D RED-BLOCK '
                'PERCEPTION REPORT'
            ),
            (
                'timestamp_local='
                + time.strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
            ),
            (
                f'opencv_version='
                f'{cv2.__version__}'
            ),
            (
                f'numpy_version='
                f'{np.__version__}'
            ),
            (
                f'total_synchronized_frames='
                f'{self.total}'
            ),
            (
                f'detected_frames='
                f'{self.detected}'
            ),
            (
                f'detection_rate='
                f'{detection_rate:.6f}'
            ),
            (
                f'synchronized_wall_rate_hz='
                f'{self.total / elapsed:.6f}'
            ),
            (
                f'tf_failures='
                f'{self.tf_failures}'
            ),
            (
                f'tf_latest_fallbacks='
                f'{self.tf_latest_fallbacks}'
            ),
            (
                f'median_center_base_m='
                f'{vec(median)}'
            ),
            (
                f'std_center_base_m='
                f'{vec(std)}'
            ),
            (
                f'expected_center_base_m='
                f'{vec(self.expected_center)}'
            ),
            (
                f'absolute_center_error_m='
                f'{vec(error)}'
            ),
            (
                f'center_tolerance_m='
                f'{vec(self.center_tolerance)}'
            ),
            (
                f'median_confidence='
                f'{median_conf:.6f}'
            ),
            (
                'criterion_detection_rate_'
                f'ge_0.95={detection_ok}'
            ),
            (
                'criterion_median_confidence_'
                f'ge_0.70={confidence_ok}'
            ),
            (
                'criterion_center_within_'
                f'tolerance={center_ok}'
            ),
            (
                f'DAY2_FINAL_RESULT={final}'
            ),
        ]

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_path.write_text(
            '\n'.join(lines) + '\n',
            encoding='utf-8',
        )

        self.get_logger().info(
            '\n' + '\n'.join(lines)
        )

        self.get_logger().info(
            f'Report written to '
            f'{self.report_path}'
        )

    def shutdown_after_report(self):
        self.shutdown_timer.cancel()

        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = Perception()

    executor = MultiThreadedExecutor(
        num_threads=2
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
