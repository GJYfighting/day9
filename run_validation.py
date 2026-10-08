#!/usr/bin/env python3
"""Run the pending checks after geometric grid validation completes."""
import json
import subprocess
import time
import argparse
import yaml
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--prepare',action='store_true',help='capture empty-scene RGB-D and rebuild the grid first')
parser.add_argument('--start-level',choices=['L0','L1','L2','L3'],help='resume smoke checks from this level, retaining previous checks')
parser.add_argument('--complete-missing',action='store_true',help='only append a fixed full-point supplemental wave if a level has fewer than five complete episodes')
options=parser.parse_args()
white_mode='white_area' in yaml.safe_load((root/'config.yaml').read_text())['day8']
if options.prepare:
    for name,args in [('camera_reference',['--capture-reference']),('grid',['--grid'])]:
        with (root/'runtime/logs'/f'{name}_final.log').open('w') as stream:
            subprocess.run(['/usr/bin/python3',str(root/'day8_check.py'),*args],cwd=root,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    if white_mode:
        subprocess.run(['/usr/bin/python3',str(root/'white_area_check.py'),'--view'],cwd=root,check=True)
        break
    path=root/'results/position_grid.json'
    if path.exists():
        grid=json.loads(path.read_text())
        if grid.get('status')=='PASS' and grid.get('visibility_status')=='PASS':break
    time.sleep(5)
else:
    raise SystemExit('grid validation timeout')
stages=[] if options.start_level or options.complete_missing else [('offline',['--offline']),('nominal',['--nominal'])]
levels=['L0','L1','L2','L3']
for level in ([] if options.complete_missing else levels[levels.index(options.start_level or 'L0'):]):
    stages.append(('smoke_'+level,['--smoke','--level',level]))
for name,args in stages:
    with (root/'runtime/logs'/f'{name}.log').open('w') as stream:
        code=subprocess.call(['/usr/bin/python3',str(root/'day8_check.py'),*args],
                             cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    print(name,'exit',code,flush=True)
    if code:
        raise SystemExit(code)

# One predeclared complete five-point wave; never remove difficult locations or
# retain only successful seeds. All attempts from both waves remain in the report.
for level in levels:
    key='smoke_'+level
    report=json.loads((root/'results/day8_check.json').read_text())
    previous=report['checks'].get(key,{})
    if previous.get('completed',0)>=5:
        continue
    path=root/'results'/f'{key}.json'
    primary=json.loads(path.read_text()) if path.exists() else []
    (root/'results'/f'{key}_primary.json').write_text(json.dumps(primary,indent=2))
    with (root/'runtime/logs'/f'{key}_supplement_1000.log').open('w') as stream:
        code=subprocess.call(['/usr/bin/python3',str(root/'day8_check.py'),'--smoke','--level',level,'--seed-offset','1000'],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    if code:raise SystemExit(code)
    supplemental=json.loads(path.read_text())
    (root/'results'/f'{key}_supplement_1000.json').write_text(json.dumps(supplemental,indent=2))
    combined=primary+supplemental
    path.write_text(json.dumps(combined,indent=2))
    report=json.loads((root/'results/day8_check.json').read_text())
    requested=previous.get('requested',5)+5
    completed=sum(r['completed'] for r in combined)
    successes=sum(r['success'] for r in combined)
    execution_ok=completed>=requested and all(r['returned_to_table'] for r in combined)
    report['checks'][key]=dict(status='PASS' if execution_ok and successes>=requested else 'PARTIAL' if execution_ok else 'FAIL',
        requested=requested,minimum_complete_required=5,minimum_complete_met=completed>=5,
        attempts=len(combined),completed=completed,successes=successes,
        detected=sum(r['detected'] for r in combined),resets=sum(r['reset'] for r in combined),
        returned=sum(r['returned_to_table'] for r in combined),records='results/'+key+'.json',
        supplemental_wave=dict(seed_offset=1000,all_five_positions_repeated=True,primary='results/'+key+'_primary.json',supplemental='results/'+key+'_supplement_1000.json'))
    (root/'results/day8_check.json').write_text(json.dumps(report,indent=2))
    print(key,'supplemented; complete=',completed,'all attempts=',len(combined),flush=True)
