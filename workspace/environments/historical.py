import numpy as np
from gym_legacy.mountain_car import MountainCarEnv
from navigation import GridEnv


class Environment:
    def __init__(self,kind,seed,record_steps=False):
        self.kind=kind;self.key='mountain_car';self.cap=self._max_episode_steps=1000
        self.gamma=1.;self.learning_rate=.5
        self.base=MountainCarEnv();self.base.seed(int(seed));self.base.reset()
        if kind=='paper_text':self.base.goal_position=.6
        self.low=self.base.observation_space.low;self.high=self.base.observation_space.high
        self.actions=self.base.action_space;self.action_space=self.actions
        self.record_steps=record_steps;self.data=[];self.episodes=[];self.total_steps=0
        self.last_success=False;self._phase=0
    def reset(self,position=None):
        obs=self.base.reset()
        if position is not None:self.base.state=np.array([position,0.],dtype=float);obs=self.base.state.copy()
        self.t=0;self.last_success=False;self.reward_total=0.;self.current_obs=obs.copy();self.start_obs=obs.copy()
        return obs
    def step(self,action):
        obs,reward,terminated,info=self.base.step(action)
        if self.kind=='paper_text' and terminated:reward=1.
        self.t+=1;self.total_steps+=1;self.last_success=terminated;self.reward_total+=reward
        done=bool(terminated or self.t>=self.cap)
        if self.record_steps:self.data.append((self._phase,self.current_obs.copy(),int(action),float(reward),obs.copy(),done))
        self.current_obs=obs.copy()
        if done:self.episodes.append(dict(length=self.t,success=bool(terminated),original_return=float(self.reward_total),
            cumulative_steps=self.total_steps,start_observation=self.start_obs.tolist()))
        return obs,reward,done,info


class Historic(Environment):
    def __init__(self,seed):
        super().__init__('historical',seed)
        self.n_count=900
        self._max_episode_steps=self.cap
    def count_bin(self,obs):
        x=np.minimum(29,np.maximum(0,((obs-self.low)/(self.high-self.low)*30).astype(int)))
        return int(x[0]*30+x[1])

def environment(key,seed,cfg):
    return Historic(seed) if key=='mountain_car' else GridEnv(key,seed,cfg.grid_cap)

