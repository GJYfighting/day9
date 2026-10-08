"""Deterministic action/replay tests; no ROS commands or synthetic learning claims."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import signal
import numpy as np
from day9_env import Day9Env, ROOT
from residual_env import ResidualActionController
from run_day9 import SpaceOnly, new_model, validate_candidate
from day9_training import add_transition
from day8_env import CurriculumManager
import yaml


class ContractTests(unittest.TestCase):
    def make_env(self, method):
        e=Day9Env.__new__(Day9Env)
        e.method=method;e.phase='diagnostic'
        e.action_space=SpaceOnly().action_space
        e.controller=ResidualActionController(ROOT/'configs/candidate.yaml',load_model=False)
        e.obs=np.zeros(10,np.float32);e.obs[9]=1
        e.trace=SimpleNamespace(step=0,span=lambda _:nullcontext(),emit=lambda *a,**kw:None)
        e.runtime=SimpleNamespace(env=SimpleNamespace(_validate_workspace=lambda:None))
        e.received=[]
        def step(a):
            e.received.append(a.copy())
            return e.obs.copy(),0.,False,False,dict(success=False)
        e.base=SimpleNamespace(step=step,backend=SimpleNamespace(drain_gripper_audit=lambda:[]))
        return e

    def test_B_no_base_or_fusion(self):
        e=self.make_env('B')
        def forbidden(*a,**kw):raise AssertionError('base/fusion called for B')
        e.controller.compute_base_action=forbidden;e.controller.decide=forbidden
        a=np.array([.003,-.002,.001,.05],np.float32)
        e.step(a);np.testing.assert_array_equal(e.received[0],a)

    def test_F_uses_external_action_no_predict(self):
        e=self.make_env('F');e.obs[0]=.002
        def forbidden(*a,**kw):raise AssertionError('internal policy inference')
        e.controller._infer_raw_residual=forbidden
        a=np.array([.005,-.005,.005,.08],np.float32)
        e.step(a)
        expected=np.array([.0016,0,0,0],np.float32)+a*.2*.25
        np.testing.assert_allclose(e.received[0],expected,atol=1e-8)

    def test_replay_stores_policy_not_fused_action_and_timeout_mask(self):
        model=new_model(SpaceOnly(),20260908)
        o=np.zeros(10,np.float32);o[9]=1
        a=np.array([.005,-.005,0,.08726646],np.float32)
        add_transition(model,o,a,o,1.,False,True)
        np.testing.assert_allclose(model.replay_buffer.actions[0,0], [1,-1,0,1],atol=1e-6)
        self.assertEqual(model.replay_buffer.timeouts[0,0],1.)
        self.assertEqual(model.replay_buffer.sample(1).dones.item(),0.)
        self.assertEqual(model._n_updates,0)

    def test_task_failure_advances_curriculum_technical_does_not(self):
        c=yaml.safe_load((ROOT/'configs/candidate.yaml').read_text())
        m=CurriculumManager(c['day8'],'auto','L0')
        m.update(False,False);self.assertEqual(list(m.window),[False])
        m.update(False,True);self.assertEqual(m.valid_total,1)
        for _ in range(99):m.update(True,False)
        self.assertEqual(m.label,'L1')

    def test_invalid_nan_rejected_before_backend(self):
        e=self.make_env('B')
        with self.assertRaises(Exception):e.step(np.full(4,np.nan,np.float32))
        self.assertEqual(e.received,[])

    def test_changed_descriptive_P1_cannot_silently_run_fixed_code(self):
        c=yaml.safe_load((ROOT/'configs/candidate.yaml').read_text())
        validate_candidate(c)
        c['day9']['P1']['batch_size']=256
        with self.assertRaises(ValueError):validate_candidate(c)

    def test_failed_safe_close_propagates_and_does_not_resume_commands(self):
        e=Day9Env.__new__(Day9Env);e.closed=False;e.phase='diagnostic'
        called=[]
        b=SimpleNamespace(safe_recover=lambda:(False,'paused'),stop_gripper=lambda:called.append('stop'))
        e.base=SimpleNamespace(backend=b)
        e.runtime=SimpleNamespace(close=lambda:called.append('close'))
        e.trace=SimpleNamespace(emit=lambda *a,**kw:None)
        with self.assertRaises(Exception):e.close()
        self.assertEqual(called,['close'])
        self.assertTrue(b.day9_skip_close_recovery)

    def test_interrupt_during_reset_is_not_retried_as_scene_failure(self):
        from day9_training import train_one
        calls=dict(reset=0,recover=0,close=0)
        class FakeEnv:
            attempt=0
            def __init__(self,*a,**kw):
                def recover():calls['recover']+=1;return True,''
                self.base=SimpleNamespace(backend=SimpleNamespace(safe_recover=recover))
            def reset(self):
                calls['reset']+=1;self.attempt+=1
                signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            def close(self):calls['close']+=1
        class FakeTrace:
            def __init__(self,*a):pass
            def emit(self,*a,**kw):pass
            def snapshot(self):return {}
            def close(self):pass
        model=SimpleNamespace(num_timesteps=0,_n_updates=0,set_logger=lambda _:None)
        with patch.multiple('day9_training',Day9Env=FakeEnv,Trace=FakeTrace,
                checkpoint=lambda *a,**kw:{},write_json=lambda *a,**kw:None,configure=lambda *a:None), \
             patch.multiple('run_day9',new_model=lambda *a:model,resolved_model=lambda _: {},model_hash=lambda _:'unchanged'), \
             patch('builtins.print'):
            result=train_one('B','configs/candidate.yaml',20260909,3000,300,stage='unit_interrupt')
        self.assertEqual(calls,dict(reset=1,recover=0,close=1))
        self.assertEqual(result['status'],'INTERRUPTED')
        self.assertEqual(result['steps'],0)


if __name__=='__main__':unittest.main()
