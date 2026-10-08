#!/usr/bin/env python3
"""DAY8 checks; inference only, never trains SAC."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import time
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml
from day8_env import ROOT, atomic_json
from white_area import center as white_center, contains, smoke_points


def report_check(name, detail):
    p=ROOT/'results/day8_check.json'
    report=json.loads(p.read_text()) if p.exists() else dict(status='INCOMPLETE', formal_sac_training=False, checks={})
    report['checks'][name]=detail
    atomic_json('results/day8_check.json',report)




def capture_reference():
    """Capture object-free RGB-D at the nominal initial robot pose, then restore block."""
    import rclpy
    import cv2
    from sensor_msgs.msg import Image,CameraInfo
    from cv_bridge import CvBridge
    from types import SimpleNamespace
    from visual_grasp_v5 import VisualGrasp,nominal_visual
    from day8_env import DomainRandomizer,Day8GraspEnv
    c=yaml.safe_load((ROOT/'config.yaml').read_text());rclpy.init()
    b=VisualGrasp(c,ROOT,'camera_reference');frames={};removed=False
    owner=SimpleNamespace(config=c,d7=c['day8'])
    dr=DomainRandomizer(c);model=dr.write_sdf(dr.sample(0,'L0',position_base=white_center(c)))
    b.create_subscription(Image,c['topics']['rgb'],lambda m:frames.update(rgb=m),2)
    b.create_subscription(Image,c['topics']['depth'],lambda m:frames.update(depth=m),2)
    b.create_subscription(CameraInfo,c['topics']['camera_info'],lambda m:frames.update(info=m),2)
    try:
        b.check_interfaces()
        b.update_planning_scene(nominal_visual([*white_center(c),.015],c))
        start_q=b.arm_q()
        b.check_joint_path(start_q,b.interpolate_joint_segment(
            start_q,c['initial_pose'],float(c['motion']['maximum_joint_step_rad'])/2), 'RESET_FAIL')
        b.prepare_initial(c['block_reset_world'])
        Day8GraspEnv.service(owner,'remove','Entity','name: "wood_block", type: MODEL');removed=True
        start=b.sim_time;deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            b.spin_once()
            if set(frames)!={'rgb','depth','info'}:continue
            rgb,depth=frames['rgb'],frames['depth']
            rs=rgb.header.stamp.sec+rgb.header.stamp.nanosec*1e-9
            ds=depth.header.stamp.sec+depth.header.stamp.nanosec*1e-9
            if rs<start+.5 or abs(rs-ds)>.01:continue
            directory=ROOT/'results/camera_reference';directory.mkdir(exist_ok=True)
            bridge=CvBridge();depth_array=bridge.imgmsg_to_cv2(depth,'passthrough')
            finite=int(np.isfinite(depth_array).sum())
            if finite<.05*depth_array.size:raise RuntimeError('initial view is occluded: insufficient finite depth')
            cv2.imwrite(str(directory/'rgb.png'),bridge.imgmsg_to_cv2(rgb,'bgr8'))
            np.save(directory/'depth.npy',depth_array)
            (directory/'joints.json').write_text(json.dumps({k:float(v[0]) for k,v in b.joints.items()}))
            info=frames['info'];camera=dict(width=int(info.width),height=int(info.height),
                k=list(map(float,info.k)),d=list(map(float,info.d)),frame_id=info.header.frame_id)
            (directory/'camera_info.yaml').write_text(yaml.safe_dump(camera,sort_keys=False))
            (directory/'model_sha256.txt').write_text(hashlib.sha256((ROOT/c['geometry']['urdf']).read_bytes()).hexdigest()+'\n')
            report_check('camera_reference',dict(status='PASS',initial_pose=c['initial_pose'],
                finite_depth_pixels=finite,total_pixels=depth_array.size,object_removed_during_capture=True,
                original_tf_chain_preserved=True))
            print('object-free camera reference saved',flush=True)
            break
        else:raise RuntimeError('camera reference capture timeout')
    finally:
        if removed:Day8GraspEnv.service(owner,'create','EntityFactory',f'sdf_filename: "{model}", name: "wood_block", allow_renaming: false')
        b.destroy_node();rclpy.shutdown()


def renderer_visibility(x,y,side,camera,transform,depth):
    """Full box projection against an object-free rendered initial-view depth map.

    Invalid near-depth pixels count as occluded: RGB can still show the robot there.
    Ray/box slab intersections give the expected optical z at each covered pixel.
    """
    optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    vertices=np.array([[x+dx,y+dy,z] for dx,dy,z in itertools.product(
        [-side/2,side/2],[-side/2,side/2],[0,side])])
    cam=(vertices-transform[:3,3])@transform[:3,:3]@optical
    uv=cam[:,:2]/cam[:,2:]*[camera['k'][0],camera['k'][4]]+[camera['k'][2],camera['k'][5]]
    lo=np.floor(uv.min(axis=0)).astype(int);hi=np.ceil(uv.max(axis=0)).astype(int)
    if np.any(lo<0) or hi[0]>=camera['width'] or hi[1]>=camera['height']:
        return dict(clear=False,reason='projection_outside_image')
    uu,vv=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
    ray=np.column_stack([(uu.ravel()-camera['k'][2])/camera['k'][0],
                         (vv.ravel()-camera['k'][5])/camera['k'][4],np.ones(uu.size)])
    ray=ray@optical.T@transform[:3,:3].T
    bounds=np.array([[x-side/2,y-side/2,0],[x+side/2,y+side/2,side]])
    with np.errstate(divide='ignore',invalid='ignore'):
        times=(bounds[:,None,:]-transform[:3,3])/ray[None,:,:]
    near=np.min(times,axis=0).max(axis=1);far=np.max(times,axis=0).min(axis=1)
    hit=(near<=far)&(near>0)
    observed=depth[vv.ravel()[hit],uu.ravel()[hit]]
    invalid=~np.isfinite(observed)|(observed<=0)
    blocked=np.isfinite(observed)&(observed<near[hit]-.002)
    return dict(clear=bool(len(observed)>0 and not np.any(invalid|blocked)),
                projected_pixels=len(observed),invalid_pixels=int(invalid.sum()),occluded_pixels=int(blocked.sum()),
                reason='rendered_depth_visibility')




def offline_check():
    from day8_env import DomainRandomizer, DelayedPublisher, CurriculumManager, PerceptionNoise
    from types import SimpleNamespace
    from residual_env import ResidualActionController
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    dr=DomainRandomizer(copy.deepcopy(c))
    dr.d7['episode_sdf']='runtime/offline_sample.sdf'
    grid=None if 'white_area' in c['day8'] else json.loads((ROOT/'results/position_grid.json').read_text())
    samples={};world=ET.Element('sdf',version='1.6');w=ET.SubElement(world,'world',name='sample_validation')
    for i in range(4):
        level=f'L{i}';v=c['day8']['levels'][level]; rows=[]
        for seed in range(100):
            p=dr.sample(seed,level);assert p==dr.sample(seed,level)
            xy=np.asarray(p['position_world_m'][:2])-np.asarray(c['robot_spawn_world'][:2])
            if 'white_area' in c['day8']:
                assert contains(c,xy,p['side_m'],p['yaw_rad'])
            else:
                assert any(np.allclose(xy,point,rtol=0,atol=1e-12) for point in grid['levels'][level])
            assert not any('gain' in k for k in p)
            for key,nom,frac in [('side_m',.03,v['size_fraction']),('mass_kg',.03,v['mass_fraction']),
                                 ('mu',6.,v['friction_fraction']),('mu2',6.,v['friction_fraction']),
                                 ('brightness',1.,v['rgb_fraction']),('contrast',1.,v['rgb_fraction'])]:
                assert nom*(1-frac)-1e-12<=p[key]<=nom*(1+frac)+1e-12
            assert p['pixel_std']==[0,2,5,8][i]
            assert p['depth_std_m']==[0,.0001,.0002,.0003][i]
            assert p['invalid_fraction']==[0,.01,.03,.05][i]
            assert np.linalg.norm(p['translation_m'])<=v['translation_mm']/1000+1e-12
            assert np.linalg.norm(p['rotation_rad'])<=math.radians(v['rotation_deg'])+1e-12
            assert 0<=p['delay_sec']<=v['delay_ms']/1000
            assert abs(p['position_world_m'][2]-p['side_m']/2-.75)<1e-12
            path=dr.write_sdf(p);m=ET.parse(path).find('model')
            assert m.findtext('link/collision/geometry/box/size')==m.findtext('link/visual/geometry/box/size')
            assert np.allclose(list(map(float,m.findtext('link/visual/geometry/box/size').split())),[p['side_m']]*3,rtol=0,atol=1e-15)
            for key in ['ixx','iyy','izz']:
                assert math.isclose(float(m.findtext('link/inertial/inertia/'+key)),p['mass_kg']*p['side_m']**2/6,rel_tol=1e-12)
            for key in ['ixy','ixz','iyz']:assert float(m.findtext('link/inertial/inertia/'+key))==0
            m.set('name',f'{level}_{seed}');w.append(m);rows.append(p)
        samples[level]=rows
    sample_path=ROOT/'runtime/sampled_models.sdf';ET.ElementTree(world).write(sample_path)
    proc=subprocess.run(['ign','sdf','-k',str(sample_path)],capture_output=True,text=True,timeout=120)
    assert proc.returncode==0 and 'Valid' in proc.stdout+proc.stderr,proc.stdout+proc.stderr
    atomic_json('results/parameter_samples.json',samples)
    controller=ResidualActionController(ROOT/'config.yaml')
    assert controller.model_loaded and controller.base_gain==.8
    assert controller.model.observation_space.shape==(10,) and controller.model.action_space.shape==(4,)
    assert not any('gain' in k for d in c['day8']['levels'].values() for k in d)
    assert not any('gain' in k for k in c['day8']['nominal'])
    base=SimpleNamespace(sim_time=1.);sent=[]
    owner=SimpleNamespace(params={'delay_sec':.1},env=SimpleNamespace(base_env=SimpleNamespace(backend=base)),delay_records=[])
    publisher=DelayedPublisher(SimpleNamespace(publish=sent.append),owner)
    publisher.publish('command');time.sleep(.12);publisher.flush();assert sent==[]
    base.sim_time=1.05;publisher.flush();assert sent==[]
    base.sim_time=1.12;publisher.flush();assert sent==['command']
    assert math.isclose(owner.delay_records[0]['actual_sec'],.12)
    import threading
    owner.delay_ready=threading.Event();owner.delay_sim_time=0.
    waiting=threading.Thread(target=publisher.publish,args=('initialized_command',))
    waiting.start();time.sleep(.03)
    assert sent==['command'] and not publisher.pending
    owner.delay_sim_time=100.;owner.delay_ready.set();waiting.join(timeout=1.)
    assert not waiting.is_alive() and publisher.pending[0][0]==100.
    time.sleep(.03);publisher.flush();assert sent==['command']
    owner.delay_sim_time=100.11;publisher.flush()
    assert sent==['command','initialized_command']
    assert math.isclose(owner.delay_records[-1]['actual_sec'],.11)
    for i in range(4):
        p=samples[f'L{i}'][0];a=PerceptionNoise(p);b=PerceptionNoise(p)
        rgb=np.full((64,64,3),127,np.uint8);depth=np.full((64,64),.4,np.float32)
        for _ in range(5):
            aa=a.images(rgb,depth);bb=b.images(rgb,depth)
            assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
            if i==0:assert np.array_equal(aa[0],rgb) and np.array_equal(aa[1],depth)
            else:
                assert abs(float(aa[0].std())-p['pixel_std'])<.4
                valid=aa[1]>0
                assert abs(float((aa[1][valid]-depth[valid]).std())/p['depth_std_m']-1)<.1
                expected=p['invalid_fraction'];tolerance=6*math.sqrt(expected*(1-expected)/depth.size)
                assert abs(float((~valid).mean())-expected)<=tolerance
    manager=CurriculumManager(c['day8'],'auto','L0')
    for _ in range(100):manager.update(True)
    assert manager.label=='L1'
    for _ in range(100):manager.update(False)
    assert manager.label=='L0'
    report_check('offline',dict(status='PASS',samples_per_level=100,total=400,seed_reproducible=True,
        all_sdf_valid=True,size_visual_collision_equal=True,inertia_correct=True,bottom_matches_table=True,
        gain_fixed=.8,delay_uses_simulation_clock=True,paused_clock_does_not_release=True,clock_initialization_gate_checked=True,
        deterministic_rgbd=True,rgb_depth_noise_units_empirically_checked=True,unchanged_observation_action_shapes=[10,4],curriculum_transition_checked=True))




def verify_freeze():
    frozen=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    changed=[name for name,sha in frozen['sha256'].items() if not (ROOT/name).is_file() or
             hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    if changed:raise RuntimeError('frozen artifacts changed: '+str(changed))
    print(json.dumps(dict(frozen_hashes_match=True,acceptance_status=frozen['status'],unpassed=frozen['unpassed'])))


def smoke_check(nominal=False, level=None, seed_offset=0, technical_retries=None, point_index=None):
    from day8_env import Day8GraspEnv
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    grid=json.loads((ROOT/'results/position_grid.json').read_text()) if not nominal and 'white_area' not in c['day8'] else None
    levels=['L0'] if nominal else ([level] if level else ['L0','L1','L2','L3'])
    for lev in levels:
        key='nominal' if nominal else 'smoke_'+lev
        previous=ROOT/'results'/(key+'.json')
        if previous.exists():
            history=ROOT/'results/history';history.mkdir(exist_ok=True)
            (history/(key+'_'+str(time.time_ns())+'.json')).write_bytes(previous.read_bytes())
        points=[white_center(c)] if nominal else (smoke_points(c) if 'white_area' in c['day8'] else grid['levels'][lev])
        if not nominal:
            # Center plus four directional extremes of each cumulative grid.
            center=points[0]
            chosen=[center]+[max(points,key=lambda p:(sign*p[axis],sum((p[k]-center[k])**2 for k in [0,1])))
                              for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]]
        else: chosen=points
        records=[]
        tasks=[(i,p) for i,p in enumerate(chosen) if point_index is None or i==point_index]
        requested_count=len(tasks)
        if not requested_count:raise ValueError('point index does not exist in this check')
        attempts_by_index={}
        for attempt,(index,point) in enumerate(tasks):
            attempts_by_index[index]=attempts_by_index.get(index,0)+1
            runtime=None;record=dict(level=lev,index=index,position_base=point,nominal=nominal,
                attempt=attempt,seed_offset=seed_offset,
                detected=False,reset=False,grasp_attempted=False,success=False,completed=False,
                returned_to_table=False,failures=[],actions=[])
            start=time.time()
            try:
                runtime=Day8GraspEnv()
                b=runtime.env.base_env.backend
                base=runtime.env.base_env
                original_settle=base._wait_reset_settle
                def settle_audit(arm_target,block_target):
                    try:return original_settle(arm_target,block_target)
                    except Exception:
                        record['reset_failure_state']=dict(arm_target=list(arm_target),arm_actual=b.arm_q(),
                            arm_speed=base._arm_speed(),block_target=list(block_target),block_actual=b.block_xyz,
                            block_linear=b.block_linear,block_angular=b.block_angular)
                        raise
                base._wait_reset_settle=settle_audit
                original_collect=b.collect_visual
                def collect():
                    value=original_collect();record['detected']=True;record['visual']=value;return value
                b.collect_visual=collect
                original_path=b.contour_cartesian_path
                def path_audit(*args,**kwargs):
                    try:return original_path(*args,**kwargs)
                    except Exception as exc:
                        if 'DAY4_LIFT' in args:
                            record['failures'].append('closed_gripper_lift_plan: '+str(exc))
                        raise
                b.contour_cartesian_path=path_audit
                # Record failure reason without changing rewards or success predicates.
                for name in ['dual_contact_search','wait_stall','lift_and_verify','hold','place_down']:
                    original=getattr(b,name)
                    def audited(*args,_fn=original,_name=name,**kwargs):
                        try:
                            result=_fn(*args,**kwargs)
                            record[_name]='PASS'
                            return result
                        except Exception as exc:
                            record['failures'].append(_name+': '+str(exc));raise
                    setattr(b,name,audited)
                obs,info=runtime.reset(20260913+int(lev[1])*100+index+seed_offset,lev,'fixed',point)
                record['reset']=True
                for step in range(c['day4']['max_steps']):
                    obs,reward,terminated,truncated,info=runtime.control_step()
                    assert obs.shape==(10,) and np.isfinite(obs).all()
                    record['actions'].append(runtime.env.last_decision.as_dict())
                    record['grasp_attempted'] |= bool(info.get('grasp_attempted',False))
                    if terminated or truncated:
                        record.update(completed=True,success=bool(info['success']),last_info=info)
                        break
                if not record['success'] and not record['failures']:
                    record['failures'].append('episode terminated/truncated without successful grasp')
            except Exception as exc:
                record['failures'].append(type(exc).__name__+': '+str(exc))
            finally:
                if runtime:
                    record['parameters']=copy.deepcopy(runtime.params)
                    b=runtime.env.base_env.backend
                    ok,detail=b.safe_recover()
                    record['returned_to_table']=bool(ok and b.block_z is not None and
                        abs(b.block_z-(.75+runtime.params['side_m']/2))<=c['reset']['pose_tolerance_m'])
                    record['recovery_detail']=detail
                    record['recovery_pose_world_m']=b.block_xyz
                    record['placed_with_gripper']=record.get('place_down')=='PASS'
                    record['actual_delays']=runtime.delay_records.copy()
                    runtime.close()
                record['elapsed_wall_sec']=time.time()-start
                records.append(record)
                key='nominal' if nominal else 'smoke_'+lev
                atomic_json('results/'+key+'.json',records)
                report_check(key,dict(status='RUNNING',requested=requested_count,recorded=len(records)))
                print('DAY8_SMOKE',lev,index,json.dumps({k:record[k] for k in ['completed','success','returned_to_table','failures']}),flush=True)
                retry_limit=c['day8']['technical_retries'] if technical_retries is None else technical_retries
                if not record['completed'] and attempts_by_index[index]<=retry_limit:
                    # Retry exactly the same point and parameter seed, never filter difficult points.
                    tasks.append((index,point))
        completed=sum(r['completed'] for r in records)
        execution_ok=completed==requested_count and all(r['returned_to_table'] for r in records)
        grasp_ok=sum(r['success'] for r in records)==requested_count
        report_check('nominal' if nominal else 'smoke_'+lev,dict(
            status='PASS' if execution_ok and grasp_ok else 'PARTIAL' if execution_ok else 'FAIL',
            execution_status='PASS' if execution_ok else 'FAIL',grasp_status='PASS' if grasp_ok else 'FAIL',
            requested=requested_count,attempts=len(records),completed=completed,successes=sum(r['success'] for r in records),
            detected=sum(r['detected'] for r in records),resets=sum(r['reset'] for r in records),
            returned=sum(r['returned_to_table'] for r in records),records='results/'+('nominal' if nominal else 'smoke_'+lev)+'.json'))




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--grid',action='store_true')
    parser.add_argument('--capture-reference',action='store_true')
    parser.add_argument('--visibility',action='store_true')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--nominal',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--level',choices=['L0','L1','L2','L3'])
    parser.add_argument('--seed-offset',type=int,default=0)
    parser.add_argument('--technical-retries',type=int,choices=range(6),default=None,
                        help='Validation-only retry bound; leaves environment curriculum configuration unchanged')
    parser.add_argument('--point-index',type=int,choices=range(5),default=None,
                        help='Replay a recorded technical failure at exactly the original point and seed')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--verify-freeze',action='store_true')
    args=parser.parse_args()
    if args.capture_reference:
        try: capture_reference()
        except Exception as exc:
            report_check('camera_reference',dict(status='FAIL',reason=str(exc)))
            raise
    if args.grid:
        from white_area_check import check
        check()
    if args.visibility:
        from white_area_check import view_check
        view_check()
    if args.offline: offline_check()
    if args.nominal: smoke_check(nominal=True,technical_retries=args.technical_retries)
    if args.smoke: smoke_check(level=args.level,seed_offset=args.seed_offset,technical_retries=args.technical_retries,point_index=args.point_index)
    if args.finalize:
        from white_area_check import finalize_revision
        finalize_revision()
    if args.verify_freeze: verify_freeze()
