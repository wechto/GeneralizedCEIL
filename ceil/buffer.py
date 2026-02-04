import numpy as np
import random

from imitation.data.rollout import rollout_stats

from typing import Sequence
from imitation.data import types

from typing import (
    Optional,
)

class TrajectoryBuffer:
    
    _trajectories: np.ndarray
    capacity: int
    dim: int
    length: int
    max_length: int
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        length: Optional[int] = 128,
        max_length: Optional[int] = 1001,
        is_demo = False,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dim = obs_dim + act_dim + 1
        self.length = length
        self.max_length = max_length
        self.capacity = capacity
        
        self._trajectories = np.zeros((self.capacity, self.max_length, self.dim))
        
        self._index = 0
        self._index_index = np.array([0]*self.capacity)
        self._index_max = np.array([0]*self.capacity)
        
        self._stats = None
        self.demo_cap = 0
        
        self.is_demo = is_demo
        self.norm = [np.array([-0.001]*self.dim, dtype="float32"), np.array([0.001]*self.dim, dtype="float32")]
        
    def sample(self, n_trajs: int, nintraj = 64) -> Sequence[types.TrajectoryWithRew]:
        """Sample n_trajs trajectories. 16 ---> 64

        Args:
            n_samples: The number of samples.

        Returns:
            A Transitions named tuple containing n_samples transitions.
        """
        assert self._index > 0

        n_trajs_ = n_trajs//nintraj
        assert n_trajs == (n_trajs_ * nintraj)

        # n_index = np.random.randint(0, min(self._index, self.capacity), n_trajs)
        ind_m = min(self._index, self.capacity)
        ind_p = np.exp(self._index_max[:ind_m]/max(self._index_max)*1.0)
        # ind_p = self._index_max[:ind_m]
        ind_p = ind_p/(ind_p).sum()
        n_index = np.random.choice(ind_m, size=n_trajs_, p=ind_p)
        
        ret = None
        ts = []
        td = []
        for n in n_index:
            s = np.random.randint(0, self._index_index[n], (nintraj,1))
            # s = (s // (self.length-1)) * (self.length-1)
            ss = s + np.arange(self.length)
            if ret is None:
                ret = self._trajectories[n][ss]
                ts = ss+1
                td = self._index_index[n] - ss[:,0]
            else:
                ret = np.concatenate([ret, self._trajectories[n][ss]], 0)
                ts = np.concatenate([ts, ss+1], 0)
                td = np.concatenate([td, self._index_index[n] - ss[:,0]], 0)
                
            # ret = ret + list(self._trajectories[n][ss])
            # ts = ts + list(ss+1)
            # tv = tv + [self.length,] * nintraj
            # td = td + list(self._index_index[n] - ss[:,0])
            assert ss.max() <= self._index_max[n]

        # for n in n_index:
        #     s = np.random.randint(0, self._index_index[n])
        #     # temp_sp = np.exp(np.arange(1, self._index_index[n]+1)[::-1] / self.max_length * 1.0 )
        #     # s = np.random.choice(self._index_index[n], p=temp_sp/np.sum(temp_sp))
        #     # if not self.is_demo:
        #     s = int((s // (self.length-1)) * (self.length-1))
        #     # temp_p = np.arange(1,self._index_index[n]+1)
        #     # temp_p = temp_p/(temp_p).sum()
        #     # s = np.random.choice(self._index_index[n], p=temp_p)
        #     # s = np.random.randint(0, self.max_length-self.length)
        #     # s = min(s, self._index_index[n]-1)

        #     # s = np.random.randint(self._index_index[n]//2, self._index_index[n])
        #     ret.append(self._trajectories[n][s:s+self.length].copy())
        #     ts.append(np.arange(s, self.length+s) + 1)
        #     tv.append(self.length)
        #     td.append(self._index_index[n] - s)
        #     if s+self.length > self._index_max[n]:
        #         ts[-1][-(s+self.length-self._index_max[n]): ] = 0 #\ ts[-1][-(s+self.length-self._index_max[n]) -1 ]
        #         tv[-1] = tv[-1] - (s+self.length-self._index_max[n])
        #         assert tv[-1] > 0
        #         assert self.is_demo == False 
        #         assert False
            
            

        ret = np.array(ret, dtype="float32")
        ts = np.array(ts, dtype="int32")
        td = np.array(td, dtype="float32")
        return ret, ts, td # batch_size * length * dim ,
    
    def update(self, ):
        ind = min(self._index, self.capacity)
        ind_p = self.max_length - self._index_max[:ind]
        ind_p = ind_p/(ind_p).sum()
        n_index = np.random.choice(ind, size=10, p=ind_p)
        for n in n_index:
            typesize = self._trajectories[n].strides[-1]
            viewT = np.lib.stride_tricks.as_strided(self._trajectories[n].copy(), 
                                            shape=(self._index_index[n]-self.length+1, self.length, self.dim),
                                            strides=(self.dim*typesize, self.dim*typesize, typesize),
                                            )
        pass 

    
        
    def store(self, trajectories: Sequence[types.TrajectoryWithRew], is_demo = False) -> None:
        """Store trajectories """

        trajectories = trajectories[:self.capacity]
        self._stats = rollout_stats(trajectories)
        
        trajs, num, num_max, size_max, norm_m = self.padding_trajs(trajectories)
        if num == 0:
            return
        assert num <= self.capacity
        
        if self._index < self.capacity: 
            ind = np.array([*range(num)]) + self._index 
            ind = ind % self.capacity
            ind = ind.tolist()
            # if self._index + num >= self.capacity:
            #     ind = [(i + (self.demo_cap if i<self.demo_cap else 0) ) for i in ind]
            # if is_demo: self.demo_cap = len(trajectories)
        else:
            # ind = np.argpartition(self._index_max, num)[:num]
            ind_m = min(self._index, self.capacity)
            ind_p = max(self._index_max) + 1 - self._index_max[:ind_m]
            ind_p = ind_p/((ind_p).sum())
            ind = np.random.choice(ind_m, size=num, p=ind_p, replace=False).tolist()

        self._trajectories[ind] = trajs.copy()
        self._index += num
        self._index_index[ind] = num_max.copy()
        self._index_max[ind] = size_max.copy()
        
        # if (self._index % 10 == 0) and (self._index > self.capacity):
        #     temp = [*range(self.capacity)]
        #     random.shuffle(temp)
        #     self._trajectories = self._trajectories[temp]
        #     self._index_index = self._index_index[temp]
        #     self._index_max = self._index_max[temp]
        
        if num > 0:
            self.norm[0] = np.min(np.concatenate(([self.norm[0]], np.array(norm_m[0], dtype="float32")), 0), 0)
            self.norm[1] = np.max(np.concatenate(([self.norm[1]], np.array(norm_m[1], dtype="float32")), 0), 0)
            
    
    def padding_trajs(self, trajectories: Sequence[types.TrajectoryWithRew]) -> Sequence[np.ndarray]:
        trajs = []
        num = 0; num_index_max = []; temp_size = []; norm_m = [[], []];
        for traj in trajectories:
            if traj.obs.shape[0] <= self.length:
                continue
            num += 1
            assert traj.obs.shape[0] == traj.acts.shape[0] + 1
            traj_terminal = np.ones((traj.obs.shape[0],1))
            traj_terminal[-1:,0] = (1-traj.terminal) * 1.
            temp = np.concatenate((traj.obs,
                                   np.concatenate((traj.acts, np.zeros((1,self.act_dim))), 0),
                                   traj_terminal,
                                   ), axis=1)[:self.max_length]
            # if not self.is_demo:
            #     temp = temp[-self.length*5:]
            num_index_max.append(max(min(temp.shape[0]-self.length, self.max_length-self.length), 1) ) 
            temp_size += [temp.shape[0]]
            if temp.shape[0] < self.max_length:
                temp_ = np.zeros((self.max_length,)+temp.shape[1:])
                temp_[:temp.shape[0]] = temp
                # temp_[temp.shape[0]:] = temp[-1]
                temp = temp_.copy()
            trajs.append(temp)
            
            norm_m[0].append(np.min(temp, 0))
            norm_m[1].append(np.max(temp, 0))
        print("\n", temp_size)
        return trajs, num, num_index_max, temp_size, norm_m
    
    # def padding_vec(self, vec: np.ndarray) -> np.ndarray:
    #     vec_shape = vec.shape
    #     if vec_shape[0] < self.length:
    #         temp = np.zeros((self.length,)+vec_shape[1:])
    #         temp[:vec_shape[0]] = vec
    #         vec = temp.copy()
    #     return vec 
    
        
    def stats(self):
        Num = sum(self._index_max > max(self._index_max) - 10)
        return self._stats, {'Num': Num, 
                            'Max': max(self._index_max), 
                            'Mean': np.mean(self._index_max[:self._index]),
                            'Nbuffer': self._index} 

        