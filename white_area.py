"""Planar JetArm recognition rectangle; coordinates never enter visual observations."""
import math
import numpy as np


def center(config):
    return list(config['day8']['white_area']['center_base_m'][:2])


def rotation(config):
    a = config['day8']['white_area']['yaw_rad']
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def half_extents(config, side, block_yaw=0.):
    area = config['day8']['white_area']
    relative = block_yaw - area['yaw_rad']
    margin = side / 2 * (abs(math.cos(relative)) + abs(math.sin(relative)))
    return np.array(area['size_xy_m']) / 2 - margin


def contains(config, xy, side, block_yaw=0.):
    local = (np.asarray(xy) - center(config)) @ rotation(config)
    return bool(np.all(np.abs(local) <= half_extents(config, side, block_yaw) + 1e-12))


def sample_position(config, rng, side, block_yaw=0., requested=None):
    half = half_extents(config, side, block_yaw)
    if np.any(half <= 0):
        raise ValueError('block does not fit inside white recognition area')
    if requested is not None:
        if not contains(config, requested, side, block_yaw):
            raise ValueError('whole block must remain inside white recognition area')
        return list(map(float, requested))
    return (np.asarray(center(config)) + rotation(config) @ rng.uniform(-half, half)).tolist()


def corners(config):
    half = np.asarray(config['day8']['white_area']['size_xy_m']) / 2
    return [(np.asarray(center(config)) + rotation(config) @ (half * p)).tolist()
            for p in ([1, 1], [-1, 1], [-1, -1], [1, -1])]


def smoke_points(config):
    # Maximum cube size keeps these same points inside the region at every level.
    side = config['day8']['nominal']['side_m'] * (1 + max(
        x['size_fraction'] for x in config['day8']['levels'].values()))
    half = half_extents(config, side) - .001
    return [center(config)] + [(np.asarray(center(config)) + rotation(config) @ p).tolist()
        for p in ([half[0], 0], [-half[0], 0], [0, half[1]], [0, -half[1]])]
