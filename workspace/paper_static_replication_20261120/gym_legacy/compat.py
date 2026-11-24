"""Only the old Gym surfaces needed by the unchanged MountainCar environment."""
from types import SimpleNamespace
import numpy as np


class Discrete:
    def __init__(self,n):self.n=n
    def sample(self):return spaces.np_random.randint(self.n)
    def contains(self,x):return isinstance(x,(int,np.integer)) and 0<=x<self.n


class Box:
    def __init__(self,low,high):
        # Gymv0.10.5 autodetects float32; the resulting bounds feed tile coding.
        self.low=low.astype(np.float32);self.high=high.astype(np.float32)


gym=SimpleNamespace(Env=object)
spaces=SimpleNamespace(Discrete=Discrete,Box=Box,np_random=np.random.RandomState(0))
