#!/usr/bin/env python3
"""Day9: common physical task, externally supplied B/F actions, bounded auditing."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
import numpy as np
import gymnasium as gym
import yaml
from day8_env import Day8GraspEnv, CurriculumManager, TechnicalFailure
from residual_env import ResidualActionController, ActionGuardError
from visual_grasp_v5 import TestFailure


class RunInterrupted(BaseException):
    """A stop request bypasses ordinary per-scene technical retry handlers."""


def local(path):
    p = (ROOT / path).resolve()
    p.relative_to(ROOT)
    return p


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    p = local(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


class Trace:
    def __init__(self, run_id):
        self.run_id = run_id
        self.path = local(f'results/{run_id}/events.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', buffering=1)
        self.totals = defaultdict(lambda: dict(wall=0., exclusive=0., sim=0., count=0))
        self.stack = []
        self.backend = None
        self.episode = 0
        self.step = 0
        self.failures = []

    def emit(self, kind, **values):
        self.stream.write(json.dumps(clean(dict(run=self.run_id, episode=self.episode,
            step=self.step, kind=kind, monotonic=time.monotonic(), **values)),
            ensure_ascii=False, allow_nan=False)+'\n')

    def sim(self):
        return float(self.backend.sim_time) if self.backend and self.backend.have_clock else None

    def account(self, name, wall, sim=0., child=0.):
        t = self.totals[name]
        t['wall'] += wall; t['exclusive'] += max(0., wall-child)
        t['sim'] += max(0., sim or 0.); t['count'] += 1
        if self.stack: self.stack[-1]['child'] += wall

    @contextmanager
    def span(self, name):
        t, s = time.monotonic(), self.sim()
        frame = {'child': 0.}; self.stack.append(frame)
        error = None
        try:
            yield
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.failures.append(dict(stage=name, reason=error))
            raise
        finally:
            self.stack.pop()
            end_sim = self.sim()
            wall = time.monotonic()-t
            sim = None if s is None or end_sim is None else end_sim-s
            self.account(name, wall, sim, frame['child'])
            self.emit('timing', stage=name, wall_sec=wall, exclusive_sec=max(0., wall-frame['child']),
                      sim_sec=sim, rtf=sim/wall if wall and sim is not None else None, error=error)

    def wrap(self, obj, method, label):
        original = getattr(obj, method)
        def measured(*args, **kwargs):
            with self.span(label): return original(*args, **kwargs)
        setattr(obj, method, measured)

    def snapshot(self): return copy.deepcopy(dict(self.totals))

    def close(self):
        if not self.stream.closed: self.stream.close()


def pause_simulation():
    result = subprocess.run(['ign', 'service', '-s', '/world/robot_world/control',
        '--reqtype', 'ignition.msgs.WorldControl', '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '3000', '--req', 'pause: true'], capture_output=True, text=True, timeout=5)
    return dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def install_execution_audit(runtime, trace, profile):
    """Profile switches change only named optimization candidates; safety is common."""
    base = runtime.env.base_env; b = base.backend
    trace.backend = b
    flags = {
        'baseline': set(), 'open_servo': {'open_servo'},
        'reset_once': {'reset_once'}, 'no_resend': {'no_resend'},
        'combined': set(runtime.config.get('day9', {}).get('accepted_optimizations', [])),
    }[profile]
    if profile == 'combined' and not flags:
        raise ValueError('combined profile requires recorded accepted_optimizations')
    b.day9_open_active = False
    b.day9_open_integral = 0.
    b.day9_search_active = False
    b.day9_search_integral = 0.
    b.day9_recovery_done = False
    b.day9_place_q = None
    b.day9_recovery_failed = False
    b.day9_skip_close_recovery = False

    def sent_elapsed():
        publisher = b.arm_pub
        pending = getattr(publisher, 'pending', ())
        sent = getattr(publisher, 'last_sent_sim', None)
        if pending: return -math.inf
        if sent is None: sent = getattr(b, 'day9_arm_start', b.sim_time)
        return b.sim_time - sent

    def wait_motion(target, duration, reason):
        stable = None
        deadline = time.monotonic() + float(b.t['arm_wall_sec'])
        while time.monotonic() < deadline:
            b.spin_once()
            valid = sent_elapsed() >= max(1., duration) and b.arm_tightly_reached(target)
            if valid:
                stable = b.sim_time if stable is None else stable
                if b.sim_time-stable >= float(b.m['grasp_settle_sim_sec']): return
            else: stable = None
        raise TestFailure(reason, 'Day9 trajectory completion/feedback timeout')

    def move(target, duration, reason, path=None):
        b.day9_recovery_done = False
        message = b.arm_message(target, duration) if path is None else b.arm_path_message(path, duration)
        b.day9_arm_start = b.sim_time
        b.arm_pub.publish(message)
        if 'no_resend' not in flags:
            for _ in range(5): b.spin_once()
            if not b.arm_reached(target): b.arm_pub.publish(message)
        wait_motion(target, duration, reason)

    b.move_arm = lambda q, duration, reason: move(q, duration, reason)
    b.move_arm_path = lambda path, duration, reason: move(path[-1], duration, reason, path)

    # Closed-hold gains/guards remain unchanged; search and its handoff are audited
    # below. A separate opening servo
    # shares documented gains, but has its own integral, velocity and effort caps.
    original_maintain = b.maintain_open_gripper
    def maintain_open():
        if not b.day9_open_active:
            original_maintain(); return
        q, v = b.r_position(), b.r_velocity()
        if not (math.isfinite(q) and math.isfinite(v)): return
        dt = max(0., min(.1, b.sim_time-b.day9_open_last))
        b.day9_open_last = b.sim_time
        target_v = float(np.clip(float(b.g['position_velocity_gain'])*(b.current_gripper_target-q), -.10, .10))
        error = target_v-v
        kp = float(b.g['velocity_proportional_gain_nm_sec_per_rad'])
        ki = float(b.g['velocity_integral_gain_nm_per_rad'])
        limit = float(b.g['effort_limit_nm'])
        proposed = float(np.clip(b.day9_open_integral+ki*error*dt, -limit, limit))
        if abs(proposed+kp*error) <= limit or error*(proposed+kp*error) < 0:
            b.day9_open_integral = proposed
        b.current_gripper_effort = float(np.clip(b.day9_open_integral+kp*error, -limit, limit))
    b.maintain_open_gripper = maintain_open
    original_command, original_start, original_stop = b.command_gripper, b.start_gripper_servo, b.stop_gripper
    def command(*args, **kwargs):
        b.day9_open_active = False
        effort = float(args[0])
        target = args[1] if len(args)>1 else kwargs.get('target')
        searching = effort < 0 and target == float(b.g['close_target']) and b.phase in ('CLOSE_SEARCH','WAIT_DUAL_CONTACT')
        if searching:
            if not b.day9_search_active:
                b.day9_search_integral = 0.
                b.day9_search_last = b.sim_time
                trace.emit('closing_search_control',velocity_reference_rad_sec=-.10,actual_hard_bound_verified=False,
                    previous_control='constant -0.02 Nm; measured driver outran passive joints',
                    kp_velocity=b.g['velocity_proportional_gain_nm_sec_per_rad'],ki_velocity=b.g['velocity_integral_gain_nm_per_rad'])
            b.day9_search_active = True
            b.gripper_servo_active = False
            b.current_gripper_target = float(target)
            b.gripper_mode = 'closing'
            update_search();b.publish_gripper(force=True)
            return
        b.day9_search_active = False
        return original_command(*args, **kwargs)
    def start_servo(*args, **kwargs):
        b.day9_open_active = False
        from_search = b.day9_search_active
        if from_search:
            # Preserve the gravity/load integral learned by velocity search.
            # Day8's velocity-only preload discarded it and caused an overshoot
            # beyond the unchanged 3 mm guard in the first Day9 regression.
            target = float(args[0] if args else kwargs['target'])
            limit = float(b.g['hold_effort_limit_nm'])
            b.gripper_servo_integral = float(np.clip(b.day9_search_integral,-limit,limit))
            b.current_gripper_target = max(0.,min(1.57,target))
            b.gripper_servo_active = True
            trace.emit('search_to_hold_integral',integral=b.gripper_servo_integral,
                       actual_velocity=b.r_velocity(),actual_angle=b.r_position(),target=target)
        b.day9_search_active = False
        return original_start(*args, **kwargs)
    def stop():
        b.day9_open_active = False
        b.day9_search_active = False
        return original_stop()
    b.command_gripper, b.start_gripper_servo, b.stop_gripper = command, start_servo, stop
    original_update = b.update_gripper_servo
    def update_search():
        if not b.day9_search_active:
            return original_update()
        v = b.r_velocity()
        if not math.isfinite(v): return
        dt = max(0., min(.1, b.sim_time-b.day9_search_last))
        b.day9_search_last = b.sim_time
        error = -.10-v
        kp = float(b.g['velocity_proportional_gain_nm_sec_per_rad'])
        ki = float(b.g['velocity_integral_gain_nm_per_rad'])
        limit = float(b.g['hold_effort_limit_nm'])
        proposed = float(np.clip(b.day9_search_integral+ki*error*dt,-limit,limit))
        if abs(proposed+kp*error)<=limit or error*(proposed+kp*error)<0:
            b.day9_search_integral=proposed
        b.current_gripper_effort=float(np.clip(b.day9_search_integral+kp*error,-limit,limit))
    b.update_gripper_servo=update_search

    reset_open_count = 0
    in_reset = False
    def open_gripper():
        nonlocal reset_open_count
        if in_reset:
            reset_open_count += 1
            if 'reset_once' in flags and reset_open_count == 2:
                if b.r_position() >= float(b.g['open_minimum']) and abs(b.r_velocity()) <= float(b.g['stall_velocity_rad_sec']):
                    trace.emit('reset_open_reused', actual=b.r_position(), velocity=b.r_velocity())
                    return
        target = float(b.g['open_target'])
        b.day9_recovery_done = False
        if 'open_servo' in flags:
            b.stop_gripper()
            b.current_gripper_target = target
            b.gripper_mode = 'open'
            b.day9_open_active = True
            b.day9_open_integral = 0.
            b.day9_open_last = b.sim_time
            maintain_open(); b.publish_gripper(force=True)
        else:
            b.command_gripper(float(b.g['effort_limit_nm']), target)
        stable = None; motion = settle = 0.
        deadline = time.monotonic() + float(b.t['open_wall_sec'])
        try:
            while time.monotonic() < deadline:
                tick = time.monotonic(); b.spin_once(); dt = time.monotonic()-tick
                angle = b.r_position(); velocity = abs(b.r_velocity())
                position_ok = abs(angle-target) <= .01 if 'open_servo' in flags else angle >= target-.01
                opened = math.isfinite(angle) and position_ok and velocity <= float(b.g['stall_velocity_rad_sec'])
                if opened:
                    settle += dt; stable = b.sim_time if stable is None else stable
                    if b.sim_time-stable >= float(b.rst['stable_sim_sec']): break
                else:
                    motion += dt; stable = None
            else:
                b.stop_gripper(); raise TestFailure('INTERFACE_NOT_READY', 'Day9 gripper open timeout')
            b.gripper_mode = 'open'
            if 'open_servo' not in flags:
                b.current_gripper_effort = float(b.g['open_hold_effort_nm'])
            b.publish_gripper(force=True)
            if b.r_position() < float(b.g['open_minimum']):
                raise TestFailure('INTERFACE_NOT_READY', 'open minimum not reached')
            b.closing_started = False
        finally:
            trace.account('open_motion', motion)
            trace.account('open_settle', settle)
            trace.emit('open_result', target_rad=target, actual_rad=b.r_position(),
                       velocity_rad_sec=b.r_velocity(), effort_nm=b.current_gripper_effort,
                       movement_wall_sec=motion, settle_wall_sec=settle, servo='open_servo' in flags)
    b.open_gripper = open_gripper

    old_reset = base._reset_once
    def reset_once(*args, **kwargs):
        nonlocal reset_open_count, in_reset
        reset_open_count = 0; in_reset = True; b.day9_recovery_done = False
        try: return old_reset(*args, **kwargs)
        finally: in_reset = False
    base._reset_once = reset_once

    original_lift = b.lift_and_verify
    def lift(path, before):
        b.day9_place_q = b.arm_q()
        b.day9_arm_start = b.sim_time
        value = original_lift(path, before)
        wait_motion(path[-1], float(b.m['lift_duration_sec']), 'BLOCK_NOT_LIFTED')
        return value
    b.lift_and_verify = lift
    original_place = b.place_down
    def place(q, before):
        b.day9_arm_start = b.sim_time
        original_place(q, before)
        wait_motion(q, float(b.m['place_duration_sec']), 'PLACE_DOWN_FAIL')
    b.place_down = place
    # Keep exact step settle thresholds; add dispatch-aware completion below.
    old_publish = base._publish_step_target
    def step_publish(q):
        result = old_publish(q)
        duration = max(1., float(base.d4['step_duration_sec']))
        if sent_elapsed() < duration:
            wait_motion(q, duration, 'DAY4_STEP_TIMEOUT')
        return result
    base._publish_step_target = step_publish

    def recover():
        if b.day9_skip_close_recovery: return True, 'already safely closed by Day9'
        if b.day9_recovery_failed: return False, 'simulation already paused; no further commands'
        with trace.span('recovery'):
            try:
                if not b.have_clock or time.monotonic()-b.last_clock_wall > float(b.t['clock_stall_wall_sec']):
                    raise TechnicalFailure('no advancing clock for recovery')
                ground = float(b.c['block_reset_world'][2])
                above = b.block_z is not None and b.block_z > ground+float(b.v['place_height_tolerance_m'])
                if above:
                    if b.day9_place_q is None:
                        raise TechnicalFailure('object elevated without a validated placement target')
                    b.set_phase('PLACE_DOWN')
                    b.place_down(b.day9_place_q, ground)
                    trace.emit('recovery_placement', table_contact=b.table_contact(), block_z=b.block_z)
                b.set_phase('OPEN_GRIPPER'); b.open_gripper()
                b.closing_started = False
                b.set_phase('RETURN_OBSERVATION')
                b.move_arm(list(map(float, b.c['initial_pose'])), float(b.m['safe_duration_sec']), 'RESET_FAIL')
                b.day9_recovery_done = True
                return True, ''
            except Exception as exc:
                b.day9_recovery_failed = True
                try: paused = pause_simulation()
                except Exception as pause_error: paused = {'error': repr(pause_error)}
                trace.emit('recovery_failed', error=repr(exc), pause=paused)
                return False, str(exc)
    b.safe_recover = recover

    original_contact = b.finger_contacts
    last_contact = {'left': -math.inf, 'right': -math.inf}
    def contacts(msg, collision, side):
        original_contact(msg, collision, side)
        if b.sim_time-last_contact[side] >= .05:
            values = [r for r in b.contact_values(msg) if 'wood_block' in str(r) and collision in str(r)]
            if values:
                last_contact[side] = b.sim_time
                trace.emit('contact', side=side, sim_time=b.sim_time, contacts=values,
                           phase=b.phase,
                           payload_fields=[dict(collision1=c.collision1.name, collision2=c.collision2.name,
                                position_count=len(c.positions), normal_count=len(c.normals), depth_count=len(c.depths))
                                for c in msg.contacts],
                           actual_rad=b.r_position(), target_rad=b.current_gripper_target,
                           gazebo_joints=b.gazebo_joints)
    b.finger_contacts = contacts
    original_geometry = b.validate_contact_geometry
    def geometry(result, *args, **kwargs):
        try: return original_geometry(result, *args, **kwargs)
        finally:
            trace.emit('contact_geometry', result=result,
                gazebo_joints=b.gazebo_joints,
                vectors_present=bool(b.left_contact_samples and b.right_contact_samples and
                    all(x['position_world'] is not None and x['normal_world'] is not None
                        for x in b.left_contact_samples+b.right_contact_samples)))
    b.validate_contact_geometry = geometry
    for obj, name, label in [
        (base, '_reset_once', 'reset'), (b, 'collect_visual', 'vision'),
        (b, 'solve_candidates', 'ik'), (b, 'contour_cartesian_path', 'ik_path'),
        (base, '_solve_increment', 'ik_step'), (base, '_publish_step_target', 'action_execution'),
        (b, 'dual_contact_search', 'closing'), (b, 'wait_stall', 'grip_settle'),
        (b, 'lift_and_verify', 'lift'), (b, 'hold', 'hold'), (b, 'place_down', 'place'),
        (b, 'open_gripper', 'open'), (runtime, 'rebuild_block', 'randomization_physics'),
        (runtime, 'update_perception', 'randomization_perception'),
    ]: trace.wrap(obj, name, label)
    old_move = b.move_arm
    def timed_move(q, duration, reason):
        label = ('return_observation' if np.allclose(q, b.c['initial_pose']) else
                 'retreat' if b.phase == 'RETREAT_PREGRASP' else 'arm_transfer')
        with trace.span(label): return old_move(q, duration, reason)
    b.move_arm = timed_move
    trace.wrap(b, 'move_arm_path', 'arm_path')


class Day9Env(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, method, config_path, trace, *, phase='diagnostic', profile='baseline', seed=20260908):
        super().__init__()
        if method not in ('A', 'B', 'F'): raise ValueError('Only A/B/F are implemented')
        self.method, self.phase, self.trace = method, phase, trace
        self.config_path = local(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.controller = ResidualActionController(self.config_path, load_model=False)
        self.runtime = Day8GraspEnv(self.config_path, controller=self.controller, require_model=False)
        self.base = self.runtime.env.base_env
        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space
        self.manager = CurriculumManager(self.config['day8'], 'auto' if method == 'F' and phase == 'train' else 'off', 'L0')
        self.run_seed = seed; self.attempt = 0; self.obs = None; self.closed = False
        self.fixed_scene = None
        install_execution_audit(self.runtime, trace, profile)

    def set_scene(self, scene): self.fixed_scene = copy.deepcopy(scene)

    def log_delays(self):
        # Training/validation retain actual sim-clock dispatch delays. Historical
        # diagnostic traces without this field are not retroactively fabricated.
        if self.phase not in ('train','validation'):return
        end=len(self.runtime.delay_records)
        rows=self.runtime.delay_records[getattr(self,'delay_logged',0):end]
        self.delay_logged=end
        if rows:self.trace.emit('command_delays',rows=rows)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        backend = self.base.backend
        if backend.day9_recovery_failed:
            raise TechnicalFailure('unsafe recovery already paused this simulation')
        if self.obs is not None and backend.block_z is not None:
            previous_ground = float(backend.c['block_reset_world'][2])
            if backend.block_z > previous_ground+float(backend.v['place_height_tolerance_m']):
                ok, detail = backend.safe_recover()
                if not ok: raise TechnicalFailure('placement required before next reset: '+detail)
        self.trace.episode += 1; self.trace.step = 0
        if self.fixed_scene:
            scene = self.fixed_scene
            actual_seed, level, mode = scene['seed'], scene['level'], scene['mode']
            xy = scene.get('position_base')
        else:
            actual_seed = int(np.random.SeedSequence([self.run_seed, self.attempt]).generate_state(1)[0])
            level = self.manager.label
            mode = 'fixed' if self.method == 'F' and self.phase == 'train' else 'off'
            xy = None
        self.attempt += 1
        self.delay_logged=0
        self.controller.reset_episode()
        self.base.backend.day9_recovery_done = False
        with self.trace.span('reset_total'):
            try:
                self.obs, info = self.runtime.reset(actual_seed, level, mode, xy)
            except Exception as exc:
                technical = isinstance(exc, TechnicalFailure) or any(x in str(exc).upper() for x in
                    ('INTERFACE', 'CLOCK', 'SERVICE', 'READBACK', 'CONTROLLER_NOT_ACTIVE', 'SIMULATION_FAILURE'))
                before, after = self.manager.update(False, technical) if self.phase == 'train' else (self.manager.label, self.manager.label)
                self.trace.emit('reset_failure', technical=technical, reason=repr(exc),
                                parameters=self.runtime.params, level_before=before, level_after=after,
                                env_steps=0, replay_transitions=0)
                raise
        self.trace.emit('reset_result', parameters=self.runtime.params, observation=self.obs, info=info)
        self.log_delays()
        return self.obs.copy(), info

    def step(self, action):
        if self.obs is None: raise RuntimeError('reset required')
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (4,) or not np.isfinite(a).all(): raise ActionGuardError('nonfinite/malformed policy action')
        self.trace.step += 1
        self.runtime.env._validate_workspace()
        if self.method == 'A':
            executed = self.controller.compute_base_action(self.obs)
            decision = dict(base_action=executed, final_action=executed)
        elif self.method == 'B':
            executed = np.clip(a, self.action_space.low, self.action_space.high)
            decision = dict(base_action=np.zeros(4), raw_policy_action=a, final_action=executed)
        else:
            d = self.controller.decide(self.obs, raw_action=a)
            executed, decision = d.final_action, d.as_dict()
        if not self.action_space.contains(executed): raise ActionGuardError('action bounds')
        old_obs = self.obs.copy()
        with self.trace.span('environment_step'):
            self.obs, reward, terminated, truncated, info = self.base.step(executed)
        if not np.isfinite(self.obs).all() or not math.isfinite(reward): raise TechnicalFailure('nonfinite transition')
        info = dict(info, day9_action=clean(decision), is_success=bool(info['success']))
        if terminated or truncated:
            if self.phase == 'train': self.manager.update(info['success'], False)
            info['curriculum'] = self.manager.state()
        self.trace.emit('transition', observation=old_obs, policy_action=a, action=decision,
                        next_observation=self.obs, reward=reward, terminated=terminated,
                        truncated=truncated, info=info)
        rows = self.base.backend.drain_gripper_audit()
        if rows: self.trace.emit('gripper_audit', rows=rows)
        self.log_delays()
        return self.obs.copy(), reward, terminated, truncated, info

    def close(self):
        if self.closed: return
        self.closed = True
        b = self.base.backend
        ok, detail = b.safe_recover()
        self.log_delays()
        self.trace.emit('safe_close', ok=ok, detail=detail)
        b.day9_skip_close_recovery = True
        if ok: b.stop_gripper()
        self.runtime.close()
        if not ok:
            raise TechnicalFailure('Day9 final recovery failed: '+detail)
