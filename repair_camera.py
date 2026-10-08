"""Rebuild only the camera frames/housing from the archived DAY8 model."""
import copy
import ast
import argparse
import json
import math
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from day8_env import ROOT, atomic_json
from grasp_geometry_v5 import URDFGeometry


def fixed(root, name, parent, child, xyz, rpy):
    ET.SubElement(root,'link',name=child)
    j=ET.SubElement(root,'joint',name=name,type='fixed')
    ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=child)
    ET.SubElement(j,'origin',xyz=' '.join(map(str,xyz)),rpy=' '.join(map(str,rpy)))


def main():
    baseline=ROOT/'generated/camera_baseline/day3_v5_robot.urdf'
    tree=ET.parse(baseline);root=tree.getroot()
    housing=root.find("link[@name='depth_cam_link']");housing.set('name','camera_housing_link')
    j=root.find("joint[@name='depth_cam_joint']");j.set('name','camera_housing_joint')
    j.find('child').set('link','camera_housing_link')
    # CAD housing stays on the wrist; moving the optical origin must not drag it.
    housing.find('collision/origin').attrib=dict(housing.find('visual/origin').attrib)
    white=root.find("link[@name='depth_cam_frame']/visual/origin")
    white.attrib=dict(housing.find('visual/origin').attrib)
    j=root.find("joint[@name='depth_cam_joint_sim']")
    j.find('parent').set('link','camera_housing_link')
    j.find('origin').attrib=dict(xyz='0 0 0',rpy='0 0 0')

    # Real kinematics hand axes in the simulation wrist frame:
    # hand +X = wrist +Z (forward); hand +Y = wrist +Y;
    # hand +Z = wrist -X (toward the camera housing, verified against CAD).
    # Its reference origin is at link3+tool_link from the real kinematics, not
    # the simulation's existing 80 mm grasp TCP. Do not move the grasp TCP.
    hand_z=.05945583202+.112
    fixed(root,'camera_calibration_hand_joint','link4','camera_calibration_hand',
          [0,0,hand_z],[0,-math.pi/2,0])
    hand_to_optical=np.array([[0,0,1,-.101],[-1,0,0,0],[0,-1,0,.045],[0,0,0,1.]])
    fixed(root,'depth_cam_optical_joint','camera_calibration_hand','depth_cam_optical_frame',
          hand_to_optical[:3,3],Rotation.from_matrix(hand_to_optical[:3,:3]).as_euler('xyz'))
    optical_to_camera=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    fixed(root,'depth_cam_joint','depth_cam_optical_frame','depth_cam_link',
          [0,0,0],Rotation.from_matrix(optical_to_camera.T).as_euler('xyz'))
    path=ROOT/'generated/day3_v5_robot.urdf';ET.indent(tree);tree.write(path,encoding='unicode',xml_declaration=True)
    sdf=subprocess.run(['ign','sdf','-p',str(path)],capture_output=True,text=True,check=True)
    start=sdf.stdout.index('<sdf');end=sdf.stdout.rindex('</sdf>')+len('</sdf>')
    converted=ET.fromstring(sdf.stdout[start:end])
    # DAY7/DAY8's validated SDF intentionally uses only finger contact boxes.
    # Full URDF conversion reintroduces expensive arm mesh collisions and renames
    # the finger collisions, breaking the existing contact sensors. Preserve all
    # original physics/contact/controller elements and transplant camera visuals,
    # frames and sensor pose only.
    sdf_tree=ET.parse(ROOT/'generated/camera_baseline/day3_v5_robot.sdf')
    model=sdf_tree.find('model');wrist=model.find("link[@name='link4']")
    new_wrist=converted.find(".//link[@name='link4']")
    for visual in list(wrist.findall('visual')):
        uri=visual.findtext('geometry/mesh/uri','')
        if 'LINK4_GEMINI' in uri:
            match=next(v for v in new_wrist.findall('visual') if v.findtext('geometry/mesh/uri')==uri)
            index=list(wrist).index(visual);wrist.remove(visual);wrist.insert(index,copy.deepcopy(match))
    wrist.find("sensor[@name='robot_cam']/pose").text=new_wrist.findtext("sensor[@name='robot_cam']/pose")
    camera_frames={'camera_housing_link','camera_housing_joint','camera_calibration_hand',
        'camera_calibration_hand_joint','depth_cam_optical_frame','depth_cam_optical_joint',
        'depth_cam_frame','depth_cam_joint_sim','depth_cam_link','depth_cam_joint'}
    for frame in list(model.findall('frame')):
        if frame.get('name') in camera_frames:model.remove(frame)
    for frame in converted.findall('.//model/frame'):
        if frame.get('name') in camera_frames:model.append(copy.deepcopy(frame))
    sdf_tree.write(ROOT/'generated/day3_v5_robot.sdf',encoding='unicode',xml_declaration=True)
    subprocess.run(['ign','sdf','-k',str(ROOT/'generated/day3_v5_robot.sdf')],check=True)
    geometry=URDFGeometry(path);old=URDFGeometry(baseline)
    c=yaml.safe_load((ROOT/'config.yaml').read_text());q=dict(zip(c['arm_joints'],c['initial_pose']));q['r_joint']=1.3
    t=geometry.transform('link4','depth_cam_link',q)
    assert np.allclose(t[:3,3],[-.045,0,.07045583202],atol=1e-12)
    assert np.allclose(geometry.transform('camera_calibration_hand','depth_cam_optical_frame',q),hand_to_optical,atol=1e-12)
    assert np.allclose(geometry.transform('base_link','end_effector_link',q),old.transform('base_link','end_effector_link',q))
    # Sensor frames must change, but all actuated joints and grasp geometry stay.
    for joint in root.findall('joint'):
        if joint.get('type')!='fixed':
            original=ET.parse(baseline).find("joint[@name='%s']"%joint.get('name'))
            signature=lambda node:[(x.tag,x.attrib,(x.text or '').strip()) for x in node.iter()]
            assert signature(joint)==signature(original)
    sensor=ET.parse(ROOT/'generated/day3_v5_robot.sdf').find('.//sensor[@name="robot_cam"]')
    pose=list(map(float,sensor.findtext('pose').split()))
    assert np.allclose(pose[:3],t[:3,3],atol=1e-6)
    assert np.allclose(Rotation.from_euler('xyz',pose[3:]).as_matrix(),t[:3,:3],atol=1e-5)
    original_sdf=ET.parse(ROOT/'generated/camera_baseline/day3_v5_robot.sdf')
    for tag in ['collision','joint','plugin']:
        assert [ET.tostring(x) for x in original_sdf.findall('.//'+tag)]==[ET.tostring(x) for x in sdf_tree.findall('.//'+tag)]
    detail=dict(status='PASS',hand_to_optical=hand_to_optical.tolist(),link4_to_camera=t.tolist(),
        calibration_hand_reference_z_m=hand_z,grasp_tcp_unchanged=True,actuated_joints_unchanged=True,
        sensor_urdf_sdf_consistent=True,all_sdf_collisions_joints_plugins_preserved=True,source_commit='70502c8',
        interpretation='Source hand-eye prior transferred via CAD axes; not a new real-camera calibration.')
    atomic_json('results/camera_repair_geometry.json',detail)
    c['day8']['camera_mount']=dict(reference='camera_calibration_hand',hand_reference_z_m=hand_z,
        link4_translation_m=t[:3,3].tolist(),source_hand_to_optical=hand_to_optical.tolist(),
        derivation='results/camera_repair_geometry.json')
    (ROOT/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    print(json.dumps(detail),flush=True)


def verify_live():
    from day8_check import report_check
    p=subprocess.run(['ign','service','-s','/world/robot_world/generate_world_sdf',
        '--reqtype','ignition.msgs.SdfGeneratorConfig','--reptype','ignition.msgs.StringMsg',
        '--timeout','30000','--req','global_entity_gen_config: {expand_include_tags: {data: true}}'],
        capture_output=True,text=True,timeout=32,check=True)
    text=ast.literal_eval(p.stdout.split('data:',1)[1].strip())
    (ROOT/'runtime/camera_final_world_readback.sdf').write_text(text)
    live=ET.fromstring(text).find('.//model[@name="robot"]')
    target=ET.parse(ROOT/'generated/day3_v5_robot.sdf').find('model')
    sensors=live.findall('.//sensor[@name="robot_cam"]');assert len(sensors)==1
    pose=list(map(float,sensors[0].findtext('pose').split()))
    expected=list(map(float,target.findtext('.//sensor[@name="robot_cam"]/pose').split()))
    assert np.allclose(pose[:3],expected[:3],atol=1e-6)
    assert np.allclose(Rotation.from_euler('xyz',pose[3:]).as_matrix(),Rotation.from_euler('xyz',expected[3:]).as_matrix(),atol=1e-5)
    names=lambda model:sorted(x.get('name') for x in model.findall('link/collision'))
    assert names(live)==names(target), (names(live),names(target))
    report_check('live_camera_mount',dict(status='PASS',sensor_count=1,sensor_pose_in_link4=pose,
        collision_names=names(live),actual_model_readback=True))
    print('LIVE_CAMERA_AND_CONTACT_GEOMETRY=PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-live',action='store_true');args=parser.parse_args()
    if args.verify_live:verify_live()
    else:main()
