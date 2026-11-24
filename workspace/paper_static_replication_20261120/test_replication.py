import ast
import inspect
from pathlib import Path
import textwrap
import unittest
import numpy as np
import run_replication as r


class ReplicationTests(unittest.TestCase):
    def test_original_learning_methods_are_unchanged(self):
        for name in ('ApproximatedSarsaLambdaAgent','HAApproximatedSarsaLambdaAgent'):
            src=(r.ROOT/'author_direct_comparison_20261120/author_agents'/f'{name}.py').read_text()
            cls=r.Plain if name.startswith('Approximated') else r.HA
            expected=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name=='learn')
            expected.lineno=1
            actual=ast.parse(textwrap.dedent(inspect.getsource(cls.learn))).body[0]
            self.assertEqual(ast.dump(expected),ast.dump(actual))
        old=(r.ROOT/'author_direct_comparison_20261120/author_agents/pa_original.py').read_text()
        new=(r.HERE/'paper_author/pa_source.py').read_text()
        self.assertEqual(old,new)

    def test_exact_historical_step_reset_and_seed_bodies(self):
        raw=ast.parse((r.HERE/'upstream/gym_v0105_mountain_car.py').read_text())
        cls=next(n for n in raw.body if isinstance(n,ast.ClassDef))
        for name in ('step','reset','seed'):
            wanted=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            got=ast.parse(textwrap.dedent(inspect.getsource(getattr(r.MountainCarEnv,name)))).body[0]
            self.assertEqual(ast.dump(wanted),ast.dump(got))
        a=r.Environment('historical',91);b=r.Environment('historical',91)
        np.testing.assert_array_equal(a.reset(),b.reset())
        self.assertEqual(a.reset().dtype,np.dtype('float64'))
        self.assertEqual(a.low.dtype,np.dtype('float32'))

    def test_environment_goal_reward_ambiguity_is_explicit(self):
        a=r.Environment('historical',92);a.reset();a.base.state=(.51,.01)
        _,reward,done,_=a.step(2);self.assertTrue(done);self.assertEqual(reward,-1.)
        b=r.Environment('paper_text',92);b.reset();b.base.state=(.59,.02)
        _,reward,done,_=b.step(2);self.assertTrue(done);self.assertEqual(reward,1.)

    def test_fast_projection_equal_to_original_old_bounds(self):
        env,agent,raw,features,rng=r.setup('historical',93,'plain')
        obs=np.random.RandomState(1).uniform([-1.2,-.07],[.6,.07],(300,2))
        for s in obs:
            for a in range(3):np.testing.assert_array_equal(features(s)[a],raw[a](s))

    def test_static_setter_only_transfers_tau_and_coders(self):
        _,teacher,raw,_,_=r.setup('historical',94,'donor')
        tau=np.random.RandomState(3).uniform(.1,.5,teacher.pheromone_trace.shape)
        payload=dict(tau=tau,coders=raw)
        env,agent,_,_,rng=r.setup('historical',95,'static','frozen',payload)
        r.train_episodes(env,agent,rng,2)
        np.testing.assert_array_equal(agent.pheromone_trace,tau)
        self.assertEqual(len(env.episodes),2)
        self.assertFalse(agent.best_trajectory['trajectory'])

    def test_author_statistics_zero_index_and_excluded_best_episode(self):
        result=r.find_best_results([[1.,2.,3.]],[[5.,4.,4.]])
        self.assertEqual(result,dict(values=[4.],indexes=[1],times=[1.],execution_times=[6.]))


if __name__=='__main__':unittest.main()
