import unittest
import numpy as np
from study import update, Memory, Historic, Config, environment

class Mechanics(unittest.TestCase):
    def test_canonical_dense_reference(self):
        rng=np.random.default_rng(8642);w=np.zeros((3,13));z=w.copy();wr=w.copy();zr=w.copy()
        for _ in range(60):
            a=int(rng.integers(3));idx=rng.choice(13,4,replace=False);delta,dq=rng.normal(size=2)
            x=np.zeros_like(w);x[a,idx]=1.;dot=float((zr*x).sum())
            zr=.99*.9*zr+(1-.04*.99*.9*dot)*x
            wr+=.04*(delta+dq)*zr-.04*dq*x
            update(w,z,idx,a,.04,.99,.9,delta,dq,True)
            np.testing.assert_allclose(w,wr,rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(z,zr,rtol=1e-13,atol=1e-14)
    def test_source_dense_reference(self):
        rng=np.random.default_rng(8643);w=np.zeros((3,13));z=w.copy();wr=w.copy();zr=w.copy()
        for _ in range(60):
            a=int(rng.integers(3));idx=rng.choice(13,4,replace=False);delta,dq=rng.normal(size=2)
            zr[a,idx]+=1-.04*.99*.9*zr[a,idx].sum()
            wr+=.04*delta*zr+.04*dq*zr;wr[a,idx]-=.04*dq;zr*=.99*.9
            update(w,z,idx,a,.04,.99,.9,delta,dq,False)
            np.testing.assert_allclose(w,wr,rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(z,zr,rtol=1e-13,atol=1e-14)
    def test_one_trajectory_one_count(self):
        m=Memory(10);m.add([2,2,2,4],[-1]*4,1.,True)
        self.assertEqual(m.count[2],1);self.assertEqual(m.mean(-200)[2],-2)
        self.assertEqual(m.count[4],1);self.assertEqual(m.mean(-200)[4],-1)
        self.assertEqual(m.mean(-200)[8],-200)
        m.add([2,2],[-1]*2,1.,False);self.assertEqual(m.count[2],1)
    def test_state_layouts(self):
        for key in ('mountain_car','u_detour','four_rooms','slippery_rooms'):
            e=environment(key,9182,Config());s=e.reset();self.assertTrue(0<=e.count_bin(s)<e.n_count)
    def test_dutch_implementations_differ(self):
        z=np.ones((2,4));w=np.zeros_like(z);zc=z.copy();wc=w.copy()
        update(w,z,np.array([0,2]),0,.05,1.,.9,-1.,0.,False)
        update(wc,zc,np.array([0,2]),0,.05,1.,.9,-1.,0.,True)
        self.assertGreater(float(np.max(abs(w-wc))),0.)

if __name__=='__main__':unittest.main()
