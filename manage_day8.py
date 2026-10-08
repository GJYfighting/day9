import os,signal,time,subprocess,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
if os.environ.get('IGN_PARTITION')!='day8_ubuntu' or os.environ.get('ROS_DOMAIN_ID')!='90':
 raise SystemExit('Source DAY8 session_env.sh with the default isolated domain before using this helper.')
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'cmdline').read_bytes();args=raw.split(b'\0');env=(p/'environ').read_bytes().split(b'\0');cmd=raw.replace(b'\0',b' ').decode()
  if b'IGN_PARTITION=day8_ubuntu' not in env:continue
  server=cmd.startswith('ign gazebo ') and str(root) in cmd
  launch=b'launch' in args and any(str(root/x).encode() in args for x in ['world.launch.py','moveit.launch.py'])
  perception=len(args)>1 and args[1] in [b'perception_v5.py',str(root/'perception_v5.py').encode()]
  if server or launch or perception:found.append(int(p.name))
 except OSError:pass
for pid in found:
 try:os.kill(pid,signal.SIGINT)
 except ProcessLookupError:pass
time.sleep(6)
for pid in found:
 try:
  cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if cmd.startswith('ign gazebo ') and str(root) in cmd:os.kill(pid,signal.SIGTERM)
 except OSError:pass
time.sleep(2)
started={}
if sys.argv[1]=='restart':
 for label,args in [('world',['ros2','launch',str(root/'world.launch.py'),'gui:=false']),('moveit',['ros2','launch',str(root/'moveit.launch.py')]),('perception',['/usr/bin/python3',str(root/'perception_v5.py'),'--ros-args','--params-file',str(root/'perception_v5.yaml')])]:
  stream=(root/'runtime/logs'/('camera_final_'+label+'.log')).open('w')
  child=subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);started[label]=child.pid;stream.close()
(root/'runtime/camera_final_processes.json').write_text(json.dumps(dict(stopped=found,started=started),indent=2));print(started)
