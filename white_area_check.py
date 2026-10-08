"""Checks for the continuous white-area revision; no training or point filtering."""
import argparse
import ast
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from day8_env import ROOT, DomainRandomizer, atomic_json
from white_area import center, corners, contains, rotation, half_extents
from day8_check import report_check


def check():
    c = yaml.safe_load((ROOT/'config.yaml').read_text())
    r = DomainRandomizer(c)
    assert np.allclose(c['initial_pose'], np.deg2rad([0, 9.6, -67.2, -108, 0]))
    counts = {}
    for level in c['day8']['levels']:
        points = []
        for seed in range(1000):
            p = r.sample(seed, level)
            assert p == r.sample(seed, level)
            xy = np.array(p['position_world_m'][:2]) - c['robot_spawn_world'][:2]
            assert contains(c, xy, p['side_m'], p['yaw_rad'])
            points.append((xy-center(c)) @ rotation(c) / half_extents(c,p['side_m']))
        points=np.asarray(points)
        assert np.max(np.abs(points.mean(axis=0))) < .07
        assert np.all(points.min(axis=0)<-.98) and np.all(points.max(axis=0)>.98)
        counts[level]=dict(samples=1000,normalized_min=points.min(axis=0).tolist(),
                           normalized_max=points.max(axis=0).tolist())
    nominal=r.sample(0,'L0',position_base=center(c))
    assert np.allclose(nominal['position_world_m'],c['block_reset_world'])
    try: r.sample(0,'L0',position_base=corners(c)[0])
    except ValueError: pass
    else: raise AssertionError('center on outer corner should reject protruding cube')
    world=ET.parse(ROOT/c['day8']['world_sdf'])
    marker=world.find(".//model[@name='white_recognition_area']")
    assert marker is not None and len(marker.findall('.//visual'))==4
    assert not marker.findall('.//collision')
    assert np.allclose(list(map(float,world.findtext(".//model[@name='wood_block']/pose").split()))[:3],c['block_reset_world'])
    for path in ROOT.glob('*.py'):ast.parse(path.read_text())
    for path in ROOT.glob('*.yaml'):yaml.safe_load(path.read_text())
    report_check('white_area',dict(status='PASS',sampling='continuous_uniform',
        frame=c['base_frame'],center_base_m=c['day8']['white_area']['center_base_m'],
        rectangle_size_xy_m=c['day8']['white_area']['size_xy_m'],corners_base_xy_m=corners(c),
        initial_pose_rad=c['initial_pose'],initial_pose_deg=np.rad2deg(c['initial_pose']).tolist(),
        nominal_block_center_base_m=[*center(c),.015],levels=counts,
        full_footprint_containment=True,visual_outline_no_collision=True,
        does_not_claim_entire_area_visible_or_graspable=True))
    print('WHITE_AREA_CHECK=PASS (4000 continuous samples)',flush=True)


def view_check():
    from grasp_geometry_v5 import URDFGeometry
    from day8_check import renderer_visibility
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    fingerprint=ROOT/'results/camera_reference/model_sha256.txt'
    assert fingerprint.exists() and fingerprint.read_text().strip()==hashlib.sha256((ROOT/c['geometry']['urdf']).read_bytes()).hexdigest(), 'stale camera reference model'
    actual=json.loads((ROOT/'results/camera_reference/joints.json').read_text())
    assert np.max(np.abs(np.array([actual[j] for j in c['arm_joints']])-c['initial_pose']))<.04, 'stale camera reference pose'
    camera=yaml.safe_load((ROOT/'results/camera_reference/camera_info.yaml').read_text())
    depth=np.load(ROOT/'results/camera_reference/depth.npy')
    geometry=URDFGeometry(ROOT/c['geometry']['urdf'])
    transform=geometry.transform('base_link','depth_cam_link',actual)
    half=half_extents(c,.0315)-.001
    rows=[]
    for x in np.linspace(-half[0],half[0],5):
        for y in np.linspace(-half[1],half[1],5):
            xy=(center(c)+rotation(c)@np.array([x,y])).tolist()
            result=renderer_visibility(*xy,.0315,camera,transform,depth)
            rows.append(dict(xy=xy,**result))
    detail=dict(status='PASS' if all(p['clear'] for p in rows) else 'FAIL',
        clear=sum(p['clear'] for p in rows),tested=len(rows),points=rows,
        interpretation='Diagnostic samples only; no failed position is removed from the sampler.')
    report_check('white_area_view',detail)
    print(json.dumps(detail),flush=True)


def finalize_revision():
    report=json.loads((ROOT/'results/day8_check.json').read_text())
    before=json.loads((ROOT/'results/source_before.json').read_text())
    from pathlib import Path
    changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    protected=[ROOT.parent/('day'+str(i)) for i in range(1,8)]+[ROOT.parent/'src/simulations']
    added=[str(p) for directory in protected if directory.exists() for p in directory.rglob('*')
           if p.is_file() and str(p) not in before]
    report['checks']['isolation']=dict(status='PASS' if not changed and not added else 'FAIL',
                                     files=len(before),modified=changed,added=added)
    required=['white_area','offline','camera_reference','white_area_view','nominal','isolation']
    if 'camera_mount' in report['checks']:
        required += ['camera_mount','live_camera_mount','camera_visual_rays','noncamera_configuration','smoke_L0']
        report['camera_repair_status']='PASS' if all(report['checks'].get(k,{}).get('status')=='PASS'
            for k in ['camera_mount','live_camera_mount','camera_visual_rays','camera_reference']) else 'FAIL'
    report['unpassed']=[k for k in required if report['checks'].get(k,{}).get('status')!='PASS']
    report['status']='PASS' if not report['unpassed'] else 'FAIL'
    report['formal_sac_training']=False
    artifacts=list(ROOT.glob('*.py'))+list(ROOT.glob('*.yaml'))+[ROOT/'README.md',ROOT/'session_env.sh']
    for directory in ['generated','meshes','models','moveit_v5','simulations','vendor']:
        artifacts += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    artifacts += [p for p in (ROOT/'results').rglob('*') if p.is_file()
                  and p.name not in ['day8_check.json','frozen_manifest.json','cleanup_manifest.json']
                  and 'history' not in p.parts]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    atomic_json('results/frozen_manifest.json',dict(status=report['status'],sha256=hashes,
        unpassed=report['unpassed'],scope='white-area revision snapshot; acceptance limitations in day8_check.json'))
    report['freeze']=dict(fully_accepted=report['status']=='PASS',manifest='results/frozen_manifest.json',files=len(hashes))
    report['scope']='Camera repair, white-area visibility, nominal and L0 center/edge smoke; previous multi-level results are historical.'
    atomic_json('results/day8_check.json',report)
    print(json.dumps(dict(status=report['status'],unpassed=report['unpassed'],modified_source_files=changed)))


def report_path():
    return yaml.safe_load((ROOT/'config.yaml').read_text())['day8']['world_sdf']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--view',action='store_true');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.view:view_check()
    elif a.finalize:finalize_revision()
    else:check()
