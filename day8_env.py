#!/usr/bin/env python3
"""Episode-scoped task/domain curriculum over the unchanged grasp executor."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import threading
from collections import deque
from functools import wraps
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import yaml


def local_path(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT)
    return path


def atomic_json(path, payload):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


class CurriculumManager:
    def __init__(self, config, mode=None, level=None):
        self.config = config
        self.mode = mode or config['mode']
        if self.mode not in ('off', 'fixed', 'auto'):
            raise ValueError('invalid curriculum mode')
        self.level = int((level or config['initial_level'])[1:])
        if not 0 <= self.level <= 3:
            raise ValueError('invalid level')
        self.window = deque(maxlen=int(config['window']))
        self.dwell = 0
        self.valid_total = 0

    @property
    def label(self):
        return f'L{self.level}'

    def update(self, success, technical_failure=False):
        before = self.label
        if technical_failure:
            return before, before
        self.valid_total += 1
        self.dwell += 1
        self.window.append(bool(success))
        if (self.mode == 'auto' and len(self.window) == self.window.maxlen
                and self.dwell >= self.config['minimum_dwell']):
            rate = sum(self.window) / len(self.window)
            if rate >= self.config['promote_rate']:
                self.level = min(3, self.level + 1)
            elif rate <= self.config['demote_rate']:
                self.level = max(0, self.level - 1)
            if self.label != before:
                self.dwell = 0
        return before, self.label

    def state(self):
        return dict(mode=self.mode, level=self.label, window=list(self.window),
                    dwell=self.dwell, valid_total=self.valid_total)

    def restore(self, state):
        if state['mode'] != self.mode:
            raise ValueError('state mode mismatch')
        level = int(state['level'][1:])
        if not 0 <= level <= 3 or len(state['window']) > self.window.maxlen:
            raise ValueError('invalid curriculum state')
        self.level = level
        self.window.extend(state['window'])
        self.dwell = int(state['dwell'])
        self.valid_total = int(state['valid_total'])


class DomainRandomizer:
    def __init__(self, config):
        self.config = config
        self.d7 = config['day8']
        for label, limits in self.d7['levels'].items():
            if not 0 < float(limits['position_fraction']) <= 1:
                raise ValueError(f'{label}: position rectangle exceeds the baseline safety region')
            if any(not math.isfinite(float(v)) or float(v) < 0 for v in limits.values()):
                raise ValueError(f'{label}: invalid domain range')

    def sample(self, seed, level, mode='fixed', position_base=None):
        rng = np.random.default_rng(seed)
        limits = self.d7['levels'][level]
        nominal = self.d7['nominal']
        enabled = mode != 'off'
        span = limits['position_fraction'] if enabled else 1.0
        def bound(key):
            return float(limits[key]) if enabled else 0.0
        def scale(name, key):
            return float(nominal[name] * (1 + rng.uniform(-bound(key), bound(key))))
        side = scale('side_m', 'size_fraction')
        yaw = float(rng.uniform(*self.config['day4']['sampling']['yaw_deg']))
        if 'white_area' in self.d7:
            from white_area import sample_position
            point = sample_position(self.config, rng, side, math.radians(yaw), position_base)
        elif position_base is not None and level == 'L0' and list(position_base) == [.32, 0.]:
            point = list(position_base)  # Dedicated nominal check before the grid is frozen.
        else:
            grid = json.loads(local_path(self.d7['position_grid_path']).read_text())
            if grid['status'] != 'PASS':
                raise ValueError('position grid has not passed geometric validation')
            points = grid['levels'][level if enabled else 'L3']
            point = points[int(rng.integers(len(points)))]
            if position_base is not None:
                if list(position_base) not in points:
                    raise ValueError('requested point outside this curriculum grid')
                point = list(position_base)
        position = [float(point[0] + self.config['robot_spawn_world'][0]),
                    float(point[1] + self.config['robot_spawn_world'][1]),
                    float(self.config['table_top_world_z'] + side / 2)]
        # Uniform radius bounds the vector norm, rather than independently expanding axes.
        def vector(magnitude):
            direction = rng.normal(size=3)
            return (direction / np.linalg.norm(direction) * rng.uniform(0, magnitude)).tolist()
        return dict(seed=int(seed), level=level, enabled=enabled, position_world_m=position,
                    position_fraction=span, side_m=side, yaw_rad=math.radians(yaw),
                    mass_kg=scale('mass_kg', 'mass_fraction'),
                    mu=scale('mu', 'friction_fraction'), mu2=scale('mu2', 'friction_fraction'),
                    brightness=scale('brightness', 'rgb_fraction'),
                    contrast=scale('contrast', 'rgb_fraction'), pixel_std=bound('pixel_std'),
                    depth_std_m=bound('depth_std_mm') / 1000,
                    invalid_fraction=bound('invalid_fraction'),
                    translation_m=vector(bound('translation_mm') / 1000),
                    rotation_rad=vector(math.radians(bound('rotation_deg'))),
                    delay_sec=float(rng.uniform(0, bound('delay_ms'))) / 1000,
                    noise_seed=int(np.random.SeedSequence(seed).generate_state(1)[0]))

    def write_sdf(self, params):
        world = ET.parse(local_path(self.d7['world_sdf']))
        model = copy.deepcopy(world.find(".//model[@name='wood_block']"))
        if model is None:
            raise ValueError('wood_block missing')
        link = model.find("link[@name='block_link']")
        mass = params['mass_kg']
        link.find('inertial/mass').text = repr(mass)
        x = y = z = params['side_m']
        for node in link.findall('collision/geometry/box/size') + link.findall('visual/geometry/box/size'):
            node.text = ' '.join(map(repr, (x, y, z)))
        for name, value in dict(ixx=mass*(y*y+z*z)/12, iyy=mass*(x*x+z*z)/12,
                                izz=mass*(x*x+y*y)/12, ixy=0, ixz=0, iyz=0).items():
            link.find('inertial/inertia/' + name).text = repr(value)
        for ode in link.findall('collision/surface/friction/ode'):
            ode.find('mu').text = repr(params['mu'])
            ode.find('mu2').text = repr(params['mu2'])
        model.find('pose').text = ' '.join(map(str, [*params['position_world_m'], 0, 0, params['yaw_rad']]))
        sdf = ET.Element('sdf', version='1.6')
        sdf.append(model)
        path = local_path(self.d7['episode_sdf'])
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(sdf).write(path, encoding='utf-8', xml_declaration=True)
        return path


class PerceptionNoise:
    """Deterministic frame streams; caller locks for an entire RGB-D callback."""
    def __init__(self, params=None):
        self.params = params
        streams = np.random.SeedSequence((params or {}).get('noise_seed', 0)).spawn(2)
        self.rgb_rng, self.depth_rng = [np.random.default_rng(s) for s in streams]

    def images(self, bgr, depth):
        p = self.params
        if not p or not p['enabled']:
            return bgr, depth
        if p['brightness'] != 1 or p['contrast'] != 1 or p['pixel_std']:
            bgr = np.clip(((bgr.astype(float) - 127.5) * p['contrast'] + 127.5)
                          * p['brightness'] + self.rgb_rng.normal(0, p['pixel_std'], bgr.shape),
                          0, 255).astype(np.uint8)
        if p['depth_std_m'] or p['invalid_fraction']:
            depth = depth.copy()
            valid = np.isfinite(depth) & (depth > 0)
            noise = self.depth_rng.normal(0, p['depth_std_m'], depth.shape)
            depth[valid] += noise[valid]
            depth[self.depth_rng.random(depth.shape) < p['invalid_fraction']] = 0.0
        return bgr, depth

    def extrinsics(self, rotation, translation):
        p = self.params
        if not p or not p['enabled']:
            return rotation, translation
        vector = np.asarray(p['rotation_rad'])
        theta = np.linalg.norm(vector)
        delta = np.eye(3)
        if theta:
            x, y, z = vector / theta
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            delta += math.sin(theta)*skew + (1-math.cos(theta))*(skew @ skew)
        return delta @ rotation, translation + np.asarray(p['translation_m'])


class TechnicalFailure(RuntimeError):
    pass


def technical_guard(operation):
    """Promote transport failures before the original grasp handlers can catch them."""
    @wraps(operation)
    def guarded(*args, **kwargs):
        from visual_grasp_v5 import TestFailure
        try:
            return operation(*args, **kwargs)
        except TestFailure as exc:
            if any(word in (exc.reason + ' ' + exc.detail).upper()
                   for word in ('INTERFACE', 'SIMULATION_FAILURE', 'CLOCK', 'SERVICE',
                                'CONTROLLER_NOT_ACTIVE', 'TIMEOUT')):
                raise TechnicalFailure(str(exc)) from exc
            raise
    return guarded


class DelayedPublisher:
    """FIFO driven only by backend /clock. Wall polling never advances delay."""
    def __init__(self, publisher, owner):
        self.publisher, self.owner = publisher, owner
        self.pending = deque()
        self.lock = threading.RLock()
        self.last_sent_sim = None

    def clock(self):
        return getattr(self.owner, 'delay_sim_time', self.owner.env.base_env.backend.sim_time)

    def publish(self, message):
        ready = getattr(self.owner, 'delay_ready', None)
        if ready is not None and not ready.wait(timeout=10.):
            raise TechnicalFailure('delay dispatcher has no valid simulation clock')
        with self.lock:
            now = self.clock()
            delay = (self.owner.params or {}).get('delay_sec', 0)
            if delay == 0:
                self.publisher.publish(message)
                self.last_sent_sim = now
                self.owner.delay_records.append(dict(requested_sec=0., actual_sec=0., queued_sim=now, sent_sim=now))
            else:
                self.pending.append((now, delay, copy.deepcopy(message)))
            self.flush()

    def flush(self):
        with self.lock:
            now = self.clock()
            while self.pending and now + 1e-12 >= self.pending[0][0] + self.pending[0][1]:
                queued, delay, message = self.pending.popleft()
                self.publisher.publish(message)
                self.last_sent_sim = now
                self.owner.delay_records.append(dict(requested_sec=delay, actual_sec=now-queued,
                                                      queued_sim=queued, sent_sim=now))

    def clear(self):
        with self.lock:
            self.pending.clear()

    def close(self):
        self.clear()

    def __getattr__(self, name):
        return getattr(self.publisher, name)


# ROS and policy dependencies are imported only when constructing the live wrapper.
class Day8GraspEnv:
    def __init__(self, config_path=ROOT / 'config.yaml', *, controller=None, require_model=True):
        from residual_env import ResidualGraspEnv
        from day4_env import Day4GraspEnv
        owner = self
        class EpisodeBase(Day4GraspEnv):
            @technical_guard
            def _publish_step_target(self, q):
                return super()._publish_step_target(q)

            def _sample_block_pose(self):
                return np.array(owner.params['position_world_m']), owner.params['yaw_rad']

            def _set_block_pose(self, xyz, yaw):
                # Called after the existing safe open/initial-arm sequence.
                owner.rebuild_block()
                super()._set_block_pose(xyz, yaw)

        self.config = yaml.safe_load(Path(config_path).read_text())
        self.d7 = self.config['day8']
        self.randomizer = DomainRandomizer(self.config)
        self.params = None
        self.env = ResidualGraspEnv(config_path, controller=controller, base_env=EpisodeBase(config_path))
        if require_model and not self.env.controller.model_loaded:
            self.env.close()
            raise TechnicalFailure(self.env.controller.model_load_error)
        assert self.env.controller.base_gain == 0.8
        self.delay_records = []
        backend = self.env.base_env.backend
        for method in ('spin_once', 'wait_future', 'fk_poses', 'control_command', 'check_interfaces'):
            setattr(backend, method, technical_guard(getattr(backend, method)))
        backend.arm_pub = DelayedPublisher(backend.arm_pub, self)
        backend.gripper_pub = DelayedPublisher(backend.gripper_pub, self)
        # An independent clock executor also runs while the main thread is inside
        # synchronous IK / Ignition calls. Wall time never releases a command.
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        self.delay_sim_time = backend.sim_time
        self.delay_ready = threading.Event()
        self.delay_node = Node('day8_delay_clock')
        def on_delay_clock(message):
            self.delay_sim_time = message.clock.sec + message.clock.nanosec * 1e-9
            self.delay_ready.set()
            backend.arm_pub.flush()
            backend.gripper_pub.flush()
        self.delay_node.create_subscription(Clock, '/clock', on_delay_clock,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.delay_executor = SingleThreadedExecutor()
        self.delay_executor.add_node(self.delay_node)
        self.delay_stop = threading.Event()
        def dispatch_clock():
            while not self.delay_stop.is_set() and rclpy.ok():
                self.delay_executor.spin_once(timeout_sec=.02)
        self.delay_thread = threading.Thread(target=dispatch_clock, name='day8_sim_delay', daemon=True)
        self.delay_thread.start()
        self.rebuild_count = 0

    def service(self, suffix, reqtype, request):
        seconds = float(self.d7['service_timeout_sec'])
        p = subprocess.run(['ign', 'service', '-s', f"/world/{self.config['world_name']}/{suffix}",
                            '--reqtype', 'ignition.msgs.' + reqtype,
                            '--reptype', 'ignition.msgs.Boolean', '--timeout', str(int(seconds*1000)),
                            '--req', request], capture_output=True, text=True, timeout=seconds+2)
        if p.returncode or not ('data: true' in p.stdout or 'data: 1' in p.stdout):
            raise TechnicalFailure(f'Ignition {suffix}: {p.stdout} {p.stderr}')

    def rebuild_block(self):
        path = self.randomizer.write_sdf(self.params)
        self.service('remove', 'Entity', 'name: "wood_block", type: MODEL')
        self.service('create', 'EntityFactory', f'sdf_filename: "{path}", name: "wood_block", allow_renaming: false')
        base = self.env.base_env
        backend = base.backend
        # Drain queued old samples before invalidating caches. Existing reset waits
        # subsequently require fresh odometry, pose and velocity stability.
        for _ in range(10):
            backend.spin_once(timeout=0.01)
        backend.block_xyz = backend.block_z = None
        backend.block_linear = backend.block_angular = math.inf
        backend.episode_max_block_z = -math.inf
        backend.clear_contacts()
        base._clear_perception_cache()
        self.rebuild_count += 1
        self.params['sdf_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.params['model_rebuild_acknowledged'] = True
        self.verify_physics()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def verify_physics(self):
        timeout = float(self.d7['service_timeout_sec'])
        result = subprocess.run([
            'ign', 'service', '-s', f"/world/{self.config['world_name']}/generate_world_sdf",
            '--reqtype', 'ignition.msgs.SdfGeneratorConfig',
            '--reptype', 'ignition.msgs.StringMsg', '--timeout', str(int(timeout*1000)),
            '--req', 'global_entity_gen_config: {expand_include_tags: {data: true}}'],
            capture_output=True, text=True, timeout=timeout+2)
        if result.returncode or 'data:' not in result.stdout:
            raise TechnicalFailure('world SDF readback service failed')
        text = ast.literal_eval(result.stdout.split('data:', 1)[1].strip())
        (ROOT / 'runtime/world_readback.sdf').write_text(text)
        models = ET.fromstring(text).findall(".//model[@name='wood_block']")
        if len(models) != 1:
            raise TechnicalFailure('wood_block readback missing or duplicated')
        model = models[0]
        values = {'mass_kg': float(model.findtext('link/inertial/mass'))}
        for key in ('mu', 'mu2'):
            values[key] = float(model.findtext('link/collision/surface/friction/ode/' + key))
        digits = int(self.d7['sdf_readback_significant_digits'])
        def serialized_equal(actual, expected):
            # Fortress serializes math::Inertial through a six-significant-digit
            # stream. Friction retains full precision; accept either representation.
            return any(math.isclose(actual, candidate, rel_tol=1e-12, abs_tol=1e-15)
                       for candidate in (expected, float(format(expected, f'.{digits}g'))))
        for key, value in values.items():
            if not serialized_equal(value, self.params[key]):
                raise TechnicalFailure(f'physics readback mismatch: {key} actual={value} expected={self.params[key]}')
        for kind in ('collision', 'visual'):
            actual = list(map(float, model.findtext('link/'+kind+'/geometry/box/size').split()))
            # Gazebo geometry vectors use six decimal places, unlike inertial scalars.
            expected = self.params['side_m']
            fixed = float(format(expected, '.6f'))
            if any(not (serialized_equal(v, expected) or math.isclose(v, fixed, rel_tol=1e-12, abs_tol=1e-15)) for v in actual):
                raise TechnicalFailure(f'size readback mismatch: {kind} actual={actual} expected={expected}')
            values[kind+'_size_m'] = actual
        if values['collision_size_m'] != values['visual_size_m']:
            raise TechnicalFailure('readback visual/collision size disagreement')
        requested = ET.parse(local_path(self.d7['episode_sdf'])).find('model')
        for key in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'):
            node = 'link/inertial/inertia/' + key
            if not serialized_equal(float(model.findtext(node)), float(requested.findtext(node))):
                raise TechnicalFailure('inertia readback mismatch: ' + key)
            values.setdefault('inertia_kg_m2', {})[key] = float(model.findtext(node))
        applied = ROOT / 'runtime/wood_block_applied.sdf'
        container = ET.Element('sdf', version='1.9')
        container.append(model)
        ET.ElementTree(container).write(applied, encoding='utf-8', xml_declaration=True)
        self.params['physics_readback'] = values

    def update_perception(self):
        import rclpy
        from rcl_interfaces.srv import SetParametersAtomically
        from rclpy.parameter import Parameter
        node = self.env.base_env.backend
        client = node.create_client(SetParametersAtomically, self.d7['perception_node'] + '/set_parameters_atomically')
        try:
            timeout = float(self.d7['service_timeout_sec'])
            if not client.wait_for_service(timeout_sec=timeout):
                raise TechnicalFailure('perception parameter service unavailable')
            keys = ('enabled', 'brightness', 'contrast', 'pixel_std', 'depth_std_m',
                    'invalid_fraction', 'translation_m', 'rotation_rad', 'noise_seed')
            value = json.dumps({k: self.params[k] for k in keys}, allow_nan=False)
            request = SetParametersAtomically.Request()
            request.parameters = [Parameter('day8_domain', value=value).to_parameter_msg()]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            if not future.done() or future.result() is None or not future.result().result.successful:
                raise TechnicalFailure('perception parameter update not acknowledged')
            self.env.base_env._clear_perception_cache()
        finally:
            node.destroy_client(client)

    def reset(self, seed, level, mode, position_base=None):
        for publisher in (self.env.base_env.backend.arm_pub, self.env.base_env.backend.gripper_pub):
            publisher.clear()
        self.delay_records = []
        self.params = self.randomizer.sample(seed, level, mode, position_base)
        # Reset/recovery only: do not change the fixed visual height prior.
        self.env.base_env.config['block_reset_world'] = list(self.params['position_world_m'])
        self.env.base_env.backend.c['block_reset_world'] = list(self.params['position_world_m'])
        self.update_perception()
        # Reset failures are separate technical attempts, never silently hidden
        # by the inherited reset retry loop. Safety and settling checks are identical.
        attempts = self.env.base_env.d4['reset_attempts']
        self.env.base_env.d4['reset_attempts'] = 1
        try:
            return self.env.reset(seed=seed)
        finally:
            self.env.base_env.d4['reset_attempts'] = attempts

    def control_step(self):
        return self.env.control_step()

    def close(self):
        self.delay_stop.set()
        self.delay_thread.join(timeout=2.)
        self.delay_executor.shutdown()
        self.delay_node.destroy_node()
        backend = self.env.base_env.backend
        for name in ('arm_pub','gripper_pub'):
            publisher = getattr(backend,name)
            publisher.close()
            setattr(backend,name,publisher.publisher)
        self.env.close()


def run_episodes(mode, level, episodes, seed, *, resume=False, tag='manual'):
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    d7 = config['day8']
    manager = CurriculumManager(d7, mode, level)
    state_path = local_path(d7['state_path'])
    attempt = 0
    if mode == 'auto' and resume and state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved['seed'] != seed:
            raise ValueError('resume seed mismatch')
        manager.restore(saved)
        attempt = saved['next_attempt']
    output = local_path(d7['results_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = None
    run_id = os.environ.get('DAY8_CHECK_RUN_ID', f'manual_{time.time_ns()}')
    records = []
    failures = 0
    valid = 0
    def alarm_handler(*_):
        raise TechnicalFailure('episode wall timeout')
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    try:
        while valid < episodes:
            episode_seed = int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
            record = dict(episode=manager.valid_total+1, attempt=attempt, seed=episode_seed,
                          run_seed=seed, tag=tag, run_id=run_id, mode=mode, level_before=manager.label,
                          level_after=manager.label, parameters={}, **{'return': 0.0}, steps=0,
                          success=False, terminated=False, truncated=False,
                          technical_failure=False, technical_reason='', actions=[])
            signal.alarm(int(d7['episode_timeout_sec']))
            try:
                if runtime is None:
                    runtime = Day8GraspEnv()
                obs, _ = runtime.reset(episode_seed, manager.label, mode)
                record['parameters'] = copy.deepcopy(runtime.params)
                while not (record['terminated'] or record['truncated']):
                    obs, reward, terminated, truncated, info = runtime.control_step()
                    if obs.shape != (10,) or not np.isfinite(obs).all() or not math.isfinite(reward):
                        raise TechnicalFailure('non-finite observation/reward')
                    decision = runtime.env.last_decision
                    if not decision.residual_valid:
                        raise TechnicalFailure(decision.fallback_reason)
                    if not runtime.env.action_space.contains(decision.final_action):
                        raise TechnicalFailure('action outside limits')
                    record['actions'].append(decision.as_dict())
                    record['return'] += float(reward)
                    record['steps'] += 1
                    record.update(terminated=bool(terminated), truncated=bool(truncated), success=bool(info['success']))
                    if record['steps'] > config['day4']['max_steps']:
                        raise TechnicalFailure('step budget exceeded')
                valid += 1
                failures = 0
            except Exception as exc:
                record.update(technical_failure=True, technical_reason=f'{type(exc).__name__}: {exc}')
                if runtime and runtime.params:
                    record['parameters'] = copy.deepcopy(runtime.params)
                failures += 1
            finally:
                signal.alarm(0)
            if runtime:
                record['actual_delays'] = copy.deepcopy(runtime.delay_records)
            record['level_before'], record['level_after'] = manager.update(record['success'], record['technical_failure'])
            with output.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            records.append(record)
            attempt += 1
            if mode == 'auto':
                atomic_json(state_path, dict(**manager.state(), seed=seed, next_attempt=attempt))
            print(f"DAY8_EPISODE={attempt} LEVEL={record['level_before']} VALID={valid}/{episodes} TECHNICAL={record['technical_failure']} SUCCESS={record['success']} REASON={record['technical_reason']}", flush=True)
            if failures > d7['technical_retries']:
                break
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if runtime:
            runtime.close()
    return records


def main():
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())['day8']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['off', 'fixed', 'auto'], default=config['mode'])
    parser.add_argument('--level', choices=['L0', 'L1', 'L2', 'L3'])
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error('episodes must be positive and seed nonnegative')
    records = run_episodes(args.mode, args.level, args.episodes, args.seed, resume=args.resume)
    return 0 if sum(not r['technical_failure'] for r in records) == args.episodes else 1


if __name__ == '__main__':
    raise SystemExit(main())
