#!/usr/bin/env python3
"""MoveIt with entity-copied DAY8 parameters and no source-package fallback."""
from pathlib import Path
import yaml
from launch_param_builder import load_xacro
from moveit_configs_utils.moveit_configs_builder import MoveItConfigs
from moveit_configs_utils.launches import generate_move_group_launch


def build_moveit_config():
    root=Path(__file__).resolve().parent
    config=root/'moveit_v5'
    def load(name):
        return yaml.safe_load((config/name).read_text())
    result=MoveItConfigs(package_path=config)
    for name,value in load('launch_defaults.yaml').items():
        setattr(result,name,value)
    result.robot_description={'robot_description':load_xacro(root/'generated/day3_v5_robot.urdf')}
    result.robot_description_semantic={'robot_description_semantic':load_xacro(config/'jetarm_6dof.srdf')}
    result.robot_description_kinematics={'robot_description_kinematics':load('kinematics.yaml')}
    result.joint_limits={'robot_description_planning':load('joint_limits.yaml')}
    result.pilz_cartesian_limits={'robot_description_planning':load('pilz_cartesian_limits.yaml')}
    result.trajectory_execution.update(load('moveit_controllers.yaml'))
    sensors=load('sensors_3d.yaml')
    result.sensors_3d=sensors if sensors.get('sensors') else {}
    return result


def generate_launch_description():
    return generate_move_group_launch(build_moveit_config())
