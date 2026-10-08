#!/usr/bin/env python3
"""One bounded, zero-update checkpoint/log throughput measurement."""
import json
import time
from day9_env import ROOT, write_json, clean
from run_day9 import model_hash, budget_state
from stable_baselines3 import SAC


def main():
    destination=ROOT/'results/io_probe.json'
    if destination.exists():
        print('Existing I/O evidence retained: '+str(destination));return
    declaration=dict(purpose='checkpoint and representative JSON logging cost only',
        environment_steps=0,updates=0,model_roundtrips=2,log_records=100,
        wall_limit_sec=30,retries=0,stop='deadline, nonfinite data, model/replay roundtrip mismatch')
    write_json('results/io_probe_declaration.json',declaration);print(json.dumps(declaration),flush=True)
    deadline=min(time.monotonic()+30,budget_state()['stop_measurement_monotonic'])
    results=[];directory=ROOT/f'models/diagnostic/io_{time.time_ns()}';directory.mkdir()
    for method in ['B','F']:
        if time.monotonic()>=deadline:raise TimeoutError('I/O budget')
        model=SAC.load(ROOT/f'models/diagnostic/probe_{method}.zip',device='cpu')
        model.load_replay_buffer(ROOT/f'models/diagnostic/probe_{method}_replay.pkl')
        start=time.monotonic();model.save(directory/method);model_sec=time.monotonic()-start
        start=time.monotonic();model.save_replay_buffer(directory/(method+'_replay.pkl'));replay_sec=time.monotonic()-start
        start=time.monotonic();loaded=SAC.load(directory/(method+'.zip'),device='cpu')
        loaded.load_replay_buffer(directory/(method+'_replay.pkl'));load_sec=time.monotonic()-start
        if model_hash(loaded)!=model_hash(model) or loaded.replay_buffer.size()!=model.replay_buffer.size():
            raise AssertionError('I/O roundtrip mismatch')
        results.append(dict(method=method,model_save_sec=model_sec,replay_save_sec=replay_sec,
            load_sec=load_sec,model_bytes=(directory/(method+'.zip')).stat().st_size,
            replay_bytes=(directory/(method+'_replay.pkl')).stat().st_size,updates_added=0))
    examples=[]
    for p in sorted((ROOT/'results').glob('*_paired_*/events.jsonl')):
        for line in p.open():
            if len(examples)>=100:break
            e=json.loads(line)
            if e['kind'] in ('transition','contact'):examples.append(e)
        if len(examples)>=100:break
    start=time.monotonic()
    path=ROOT/f'results/io_log_probe_{time.time_ns()}.jsonl'
    with path.open('x',buffering=1) as stream:
        for e in examples:
            if time.monotonic()>=deadline:raise TimeoutError('I/O log budget')
            stream.write(json.dumps(clean(e),ensure_ascii=False,allow_nan=False)+'\n')
    logging=dict(records=len(examples),wall_sec=time.monotonic()-start,bytes=path.stat().st_size)
    write_json(destination,dict(status='PASS',checkpoint=results,logging=logging,
        note='buffered OS writes, no fsync durability timing; includes full allocated replay arrays; no network/update-frequency changes'))
    print(destination.read_text())


if __name__=='__main__':main()
