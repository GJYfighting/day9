#!/usr/bin/env python3
"""Start/stop only the processes registered by this isolated Day9 session."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / 'runtime/processes.json'


def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


def owned(record):
    try:
        env = Path(f"/proc/{record['pid']}/environ").read_bytes().split(b'\0')
        return (identity(record['pid']) == record['start_ticks'] and
                ('DAY9_ROOT=' + str(ROOT)).encode() in env and
                b'IGN_PARTITION=day9_ubuntu' in env)
    except OSError:
        return False


def main(action):
    if os.environ.get('DAY9_ROOT') != str(ROOT) or os.environ.get('ROS_DOMAIN_ID') != '91' or os.environ.get('IGN_PARTITION') != 'day9_ubuntu':
        raise SystemExit('Source Day9 session_env.sh first (domain 91, day9_ubuntu).')
    if action == 'serve':
        stopping = False
        def stop_signal(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGINT, stop_signal)
        signal.signal(signal.SIGTERM, stop_signal)
        main('start')
        try:
            while not stopping:
                time.sleep(.5)
        finally:
            main('stop')
        return
    records = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    active = [r for r in records if owned(r)]
    if action == 'status':
        print(json.dumps(active, indent=2)); return
    if action == 'stop':
        for r in active:
            os.killpg(r['pid'], signal.SIGINT)
        end = time.monotonic() + 12
        while any(owned(r) for r in active) and time.monotonic() < end:
            time.sleep(.2)
        for r in active:
            if owned(r): os.killpg(r['pid'], signal.SIGTERM)
        return
    if active:
        if len(active) != 3: raise SystemExit('Partial Day9 stack: inspect logs, then stop before restart.')
        print('Existing Day9 stack retained.'); return
    run = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + str(time.time_ns())
    jobs = [
        ('world', ['ros2', 'launch', str(ROOT/'world.launch.py'), 'gui:=false']),
        ('moveit', ['ros2', 'launch', str(ROOT/'moveit.launch.py')]),
        ('perception', ['/usr/bin/python3', str(ROOT/'perception_v5.py'), '--ros-args', '--params-file', str(ROOT/'perception_v5.yaml')]),
    ]
    current = []
    for name, args in jobs:
        path = ROOT/'runtime/logs'/f'{run}_{name}.log'
        with path.open('x') as stream:
            child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        record = dict(name=name, pid=child.pid, start_ticks=identity(child.pid), log=str(path), args=args)
        current.append(record)
        REGISTRY.write_text(json.dumps(current, indent=2)+'\n')
    print(json.dumps(current, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status', 'serve'])
    main(parser.parse_args().action)
