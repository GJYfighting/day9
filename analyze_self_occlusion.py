import sys,json,struct,collections
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from grasp_geometry_v5 import URDFGeometry
R=Path(__file__).resolve().parent;g=URDFGeometry(R/'generated/day3_v5_robot.urdf');q=json.loads((R/'results/camera_reference/joints.json').read_text());root=ET.parse(R/'generated/day3_v5_robot.urdf')
tri=[];labels=[]
for l in root.findall('link'):
 for v in l.findall('visual'):
  mesh=v.find('geometry/mesh')
  if mesh is None:continue
  p=Path(mesh.get('filename').removeprefix('file://'));b=p.read_bytes();n=struct.unpack('<I',b[80:84])[0]
  if 84+n*50!=len(b):raise RuntimeError('nonbinary '+str(p))
  dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);vertices=np.frombuffer(b,dtype=dtype,offset=84,count=n)['v'].astype(float)*np.array(list(map(float,mesh.get('scale','1 1 1').split())))
  o=v.find('origin');t=np.eye(4)
  if o is not None:t[:3,3]=list(map(float,o.get('xyz','0 0 0').split()));t[:3,:3]=Rotation.from_euler('xyz',list(map(float,o.get('rpy','0 0 0').split()))).as_matrix()
  t=g.transform('base_link',l.get('name'),q)@t;tri.append(vertices@t[:3,:3].T+t[:3,3]);labels += [l.get('name')+'/'+p.name]*n
tri=np.concatenate(tri);a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a
cam=g.transform('base_link','depth_cam_link',q);origin=cam[:3,3]
def hit(direction,minimum=1e-5):
 h=np.cross(direction,e2);det=(e1*h).sum(1);valid=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[valid]=1/det[valid];s=origin-a;u=(s*h).sum(1)*inv;z=np.cross(s,e1);v=(z*direction).sum(1)*inv;d=(e2*z).sum(1)*inv;valid &= (u>=0)&(v>=0)&(u+v<=1)&(d>minimum)&(d<1)
 idx=np.flatnonzero(valid)
 if not len(idx):return None
 idx=idx[np.argmin(d[idx])];return dict(mesh=labels[idx],distance_scale=float(d[idx]))
counts=collections.Counter();examples={}
for y in np.linspace(5,195,11):
 for x in np.linspace(5,315,17):
  direction=cam[:3,:3]@np.array([1,-(x-160)/277.191356,-(y-100)/277.191356]);h=hit(direction)
  label=h['mesh'] if h else 'no_robot_intersection';counts[label]+=1
  if h:examples.setdefault(label,[]).append(h['distance_scale'])
point=np.array([.18459222659243593,-.012438901168749173,.03]);targethit=hit(point-origin)
relative=np.linalg.inv(g.transform('base_link','end_effector_link',q))@cam
out=dict(camera_to_base=cam.tolist(),camera_relative_to_tcp=relative.tolist(),ray_count=sum(counts.values()),nearest_visual_mesh_counts=dict(counts),optical_depth_range_m={k:[min(v),max(v)] for k,v in examples.items()},center_target_hit=targethit)
out['clipping_diagnostics']={}
for minimum in [.001,.01,.1]:
 counts=collections.Counter()
 for y in np.linspace(5,195,11):
  for x in np.linspace(5,315,17):
   direction=cam[:3,:3]@np.array([1,-(x-160)/277.191356,-(y-100)/277.191356]);h=hit(direction,minimum)
   counts[h['mesh'] if h else 'no_robot_intersection']+=1
 out['clipping_diagnostics'][str(minimum)]=dict(counts)
out['camera_in_link3']=(np.linalg.inv(g.transform('base_link','link3',q))@cam)[:3,3].tolist()
(R/'results/self_occlusion_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
