#!/usr/bin/env python3
"""Complete Gazebo world launch for the Day3-v5 visual grasp experiment."""
from pathlib import Path
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    root = Path(__file__).resolve().parent
    world = root / "simulations/robot_gazebo/worlds/grasp_table.sdf"
    urdf = root / "generated/day3_v5_robot.urdf"
    sdf = root / "generated/day3_v5_robot.sdf"
    for path in (world, urdf, sdf):
        if not path.is_file():
            raise RuntimeError(f"required Day3-v5 artifact is missing: {path}")

    robot_description = urdf.read_text(encoding="utf-8")
    gui_value = LaunchConfiguration("gui").perform(context).strip().lower()
    if gui_value not in ("true", "false"):
        raise RuntimeError("gui must be true or false")
    ign_args = f"-r {world}" if gui_value == "true" else f"-r -s {world}"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_ign_gazebo"),
                "launch",
                "ign_gazebo.launch.py",
            )
        ),
        launch_arguments={"ign_args": ign_args}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="day3_v5_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/depth_cam/rgbd/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/depth_cam/rgbd/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/depth_cam/rgbd/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/world/robot_world/model/robot/link/l_out_link/sensor/day3_v5_left_contact/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
            "/world/robot_world/model/robot/link/r_out_link/sensor/day3_v5_right_contact/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
            "/world/robot_world/model/wood_block/link/block_link/sensor/day3_v5_block_contact/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
            "/day3_v5/block/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        ],
        remappings=[
            (
                "/world/robot_world/model/robot/link/l_out_link/sensor/day3_v5_left_contact/contact",
                "/day3_v5/left_finger/contacts",
            ),
            (
                "/world/robot_world/model/robot/link/r_out_link/sensor/day3_v5_right_contact/contact",
                "/day3_v5/right_finger/contacts",
            ),
            (
                "/world/robot_world/model/wood_block/link/block_link/sensor/day3_v5_block_contact/contact",
                "/day3_v5/block/contacts",
            ),
        ],
        parameters=[{"use_sim_time": True}],
    )

    gazebo_joint_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="day3_v5_gazebo_joint_bridge",
        output="screen",
        arguments=[
            "/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
        ],
        remappings=[("/joint_states", "/day3_v5/gazebo_joint_states")],
        parameters=[{"use_sim_time": True}],
    )

    spawn = Node(
        package="ros_ign_gazebo",
        executable="create",
        name="day3_v5_spawn_robot",
        output="screen",
        arguments=[
            "-file", str(sdf),
            "-name", "robot",
            "-allow_renaming", "false",
            "-x", "-0.20",
            "-y", "0.0",
            "-z", "0.75",
        ],
        parameters=[{"use_sim_time": True}],
    )

    joint_state = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "180", "--switch-timeout", "30"],
        output="screen",
    )
    arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager-timeout", "180", "--switch-timeout", "30"],
        output="screen",
    )
    gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager-timeout", "180", "--switch-timeout", "30"],
        output="screen",
    )

    return [
        # Ogre2 CreateLogger reads HOME directly, ignoring IGN_HOMEDIR.
        # Scope its application home to this launch context, preserving the shell
        # home and the existing user-installed policy dependencies.
        SetEnvironmentVariable('HOME', str(root / 'runtime/home')),
        gazebo,
        robot_state_publisher,
        bridge,
        gazebo_joint_bridge,
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[joint_state])),
        RegisterEventHandler(OnProcessExit(target_action=joint_state, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[gripper])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start the complete Gazebo GUI; use false for headless checks.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
