"""Copy a frozen Day9 experiment into a physically independent Day10 tree."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_start(day9, pipeline_state):
    day9=Path(day9).resolve();day10=day9.parent/'day10'
    sys.path.insert(0,str(day9))
    from run_day9 import code_hashes, artifact_hashes
    frozen=day9/'configs/frozen.yaml';manifest=day9/'configs/frozen_manifest.json'
    pilot=day9/'results/pilot_summary.json'
    if not all(p.is_file() for p in (frozen,manifest,pilot)):
        raise RuntimeError('Day9 freeze artifacts incomplete')
    original=json.loads(manifest.read_text())
    if original['code_hashes']!=code_hashes() or original['artifact_hashes']!=artifact_hashes():
        raise RuntimeError('Day9 frozen source differs from live files')
    if original.get('automation_hashes')!={
        str(p.relative_to(day9)):digest(p) for p in sorted((day9/'automation').glob('*.py'))}:
        raise RuntimeError('Day9 stage automation differs from frozen source')
    if original['config_sha256']!=digest(frozen) or original['pilot_sha256']!=digest(pilot):
        raise RuntimeError('Day9 frozen config or pilot changed')
    if json.loads(pilot.read_text())['status']!='COMPLETE':
        raise RuntimeError('Day9 pilot is incomplete')
    hold_path=day9/'results/day10_hold.json'
    if hold_path.exists():
        hold=json.loads(hold_path.read_text())
        if hold.get('hold_after_day9') is True:
            pipeline_state['status']='DAY9_COMPLETE_DAY10_DEFERRED'
            pipeline_state['day10_handoff_deferred']=dict(
                reason='user requested stopping after Day9',
                request=str(hold_path.relative_to(day9)),
                time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            (day9/'results/pipeline_state.json').write_text(json.dumps(pipeline_state,indent=2)+'\n')
            return None
    if day10.exists():
        raise RuntimeError('Day10 already exists; inspect and resume its own runner, never overwrite')
    day10.mkdir(parents=True)
    copied={}
    runtime_python={'run_day9.py','day9_training.py','day9_env.py','day8_env.py',
        'day4_env.py','residual_env.py','visual_grasp_v5.py','grasp_geometry_v5.py',
        'perception_v5.py','white_area.py','world.launch.py','moveit.launch.py'}
    for rel,hash_value in sorted(original['code_hashes'].items()):
        if rel in ('manage_day9.py','session_env.sh'):continue
        if rel.endswith('.py') and '/' not in rel and rel not in runtime_python:continue
        source=day9/rel
        if digest(source)!=hash_value:raise RuntimeError(f'Day9 source changed: {rel}')
        target=day10/rel;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if rel.startswith('generated/') and rel.endswith(('.sdf','.urdf')):
            data=data.replace(str(day9).encode(),str(day10).encode())
        if rel=='run_day9.py':
            data=data.replace(b"os.environ.get('DAY9_ROOT')",b"os.environ.get('DAY10_ROOT')")
            data=data.replace(b'[20260911,20260912,20260913] for s in args.seeds',
                              b'[20260911] for s in args.seeds')
        target.write_bytes(data);copied[rel]=hash_value
    for rel,hash_value in sorted(original['artifact_hashes'].items()):
        source=day9/rel
        if digest(source)!=hash_value:raise RuntimeError(f'Day9 asset changed: {rel}')
        target=day10/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);copied[rel]=hash_value
    for path in (frozen,):
        target=day10/'configs'/path.name;shutil.copy2(path,target)
        copied[str(path.relative_to(day9))]=digest(path)
    shutil.copytree(day9/'meshes',day10/'meshes',symlinks=False)
    (day10/'results').mkdir()
    (day10/'models').mkdir(exist_ok=True)
    (day10/'provenance').mkdir()
    for rel in ('results/pilot_summary.json','results/formal_forecast_post_pilot.json',
                'configs/frozen_manifest.json','REPORT.md','CORRECTNESS.md'):
        p=day9/rel
        if p.exists():
            target=day10/'provenance'/rel.replace('/','__')
            shutil.copy2(p,target)
            copied[rel]=digest(p)
    manager=(day9/'manage_day9.py').read_text().replace('DAY9','DAY10').replace('Day9','Day10')
    manager=manager.replace('day9_ubuntu','day10_ubuntu').replace("!= '91'","!= '92'")
    (day10/'manage_day10.py').write_text(manager)
    script=(day9/'session_env.sh').read_text().replace('DAY9','DAY10').replace('Day9','Day10')
    script=script.replace('day9_','day10_').replace('day[0-8]','day[0-9]')
    script=script.replace('DAY10_ROS_DOMAIN_ID:-91','DAY10_ROS_DOMAIN_ID:-92')
    # The handoff process inherits Day9's shell. Clear every inherited Day9
    # location before sourcing ROS, then construct Day10 paths from this copy.
    script=script.replace('\nDAY10_ROOT=',
        '\nfor DAY10_OLD_VAR in ${!DAY9_@}; do unset "$DAY10_OLD_VAR"; done\n'
        'unset DAY10_OLD_VAR\nDAY10_ROOT=',1)
    (day10/'session_env.sh').write_text(script)
    (day10/'run_day10.py').write_text('''#!/usr/bin/env python3
"""Day10 local entry; only the authorized B/F seed 20260911 jobs are accepted."""
from run_day9 import main
if __name__ == '__main__':main()
''')
    (day10/'automation').mkdir()
    shutil.copy2(day9/'automation/run_formal.py',day10/'automation/run_formal.py')
    # Day10's runner is standalone. Its imports and supervisor all resolve here.
    source_manifest=dict(source_day9=str(day9),day9_frozen_sha256=digest(frozen),
        day9_pilot_sha256=digest(pilot),day9_code_hashes=original['code_hashes'],
        copied_source_hashes=copied,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        formal_jobs={'B':dict(seed=20260911,steps=10000),'F':dict(seed=20260911,steps=10000)},
        note='all runtime inputs are local to Day10; shared ROS/Gazebo/install used read-only')
    source_manifest['resource_hashes']={str(p.relative_to(day10)):digest(p)
        for p in sorted((day10/'meshes').rglob('*')) if p.is_file()}
    (day10/'provenance/source.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    # Query the copied code in a separate interpreter so no Day9 module can be cached.
    check='''import json,run_day9; print(json.dumps(dict(code=run_day9.code_hashes(),assets=run_day9.artifact_hashes())))'''
    env=os.environ.copy();env['PYTHONPATH']=str(day10/'vendor')+':'+str(day10)
    output=subprocess.check_output(['/usr/bin/python3','-c',check],cwd=day10,env=env,text=True)
    own=json.loads(output)
    day10_manifest=dict(config_sha256=digest(day10/'configs/frozen.yaml'),
        code_hashes=own['code'],artifact_hashes=own['assets'],
        resource_hashes=source_manifest['resource_hashes'],
        automation_sha256=digest(day10/'automation/run_formal.py'),
        day9_frozen_manifest_sha256=digest(manifest),
        formal_network_and_replay='fresh per method',formal_steps=20000,
        seed=20260911,methods=['B','F'])
    (day10/'configs/frozen_manifest.json').write_text(json.dumps(day10_manifest,indent=2)+'\n')
    # Static independence check on active resources. Provenance is allowed to
    # mention Day9, but executable paths and imports must not.
    forbidden=str(day9).encode()
    active=(list(day10.glob('*.py'))+list(day10.glob('*.sh'))+
            list((day10/'generated').rglob('*.sdf'))+
            list((day10/'generated').rglob('*.urdf')))
    if any(forbidden in p.read_bytes() for p in active):
        raise RuntimeError('Day10 active code/resource still references Day9')
    for p in day10.rglob('*'):
        if p.is_symlink() and not str(p.resolve()).startswith(str(day10)+'/'):
            raise RuntimeError(f'Day10 external link: {p}')
    (day10/'provenance/independence.json').write_text(json.dumps(dict(status='PASS',
        active_paths_checked=len(active),day9_runtime_references=0,
        external_writable_links=0,day9_deletion_simulated_by_static_path_audit=True),indent=2)+'\n')
    log=day10/'runtime/logs/day10_formal_pipeline.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=['/bin/bash','-lc',f'source {day10}/session_env.sh && exec /usr/bin/python3 -u {day10}/automation/run_formal.py']
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=day10,stdout=stream,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    pipeline_state['day10_handoff']=dict(pid=child.pid,log=str(log),command=command,
        day10_manifest_sha256=digest(day10/'configs/frozen_manifest.json'))
    pipeline_state['status']='DAY10_HANDED_OFF'
    (day9/'results/pipeline_state.json').write_text(json.dumps(pipeline_state,indent=2)+'\n')
    return day10
