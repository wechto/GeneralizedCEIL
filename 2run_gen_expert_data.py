import gym
import numpy as np

# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.util.util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import torch 

from imitation.data.types import TrajectoryWithRew 

import pickle
import os.path as osp, os



NUM_DEMO = 20

env_name="Hopper1-v2" # 3234
env_name="HalfCheetah1-v2" # 12135
env_name="Walker2d1-v2" # 4592
env_name="Ant1-v2" 

env_name="HalfCheetah-v2" # 

data = torch.load("_data/experts-new/"+env_name+"/params.pkl")
policy =  data["evaluation/policy"]
env = data["evaluation/env"]
# env = gym.make(env_name)

set_gpu_mode(True)
policy.cuda()

demonstrations = []
demo_dir = "_data/demo-new/"

try:
    print("Loading...")
    with open(osp.join(demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl"), "rb") as savepkl:
        demonstrations = pickle.load(savepkl)
        print("Load demo from", demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl")
except:
    print("Load fails...")

    i = 0 
    for _ in range(1000):
        # dict_keys(['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'dones', 
        # 'agent_infos', 'env_infos', 'full_observations', 'full_next_observations'])
        env.reset() 
        traj = rollout(
            env,
            policy,
            max_path_length=1000,
            render=False,
        )
        print(_, i, traj['observations'].shape[0])
        if traj['observations'].shape[0] < 1000:
            continue 

        traj = TrajectoryWithRew(obs=np.concatenate((traj['observations'], traj['next_observations'][-1:]), 0), 
                                acts=traj['actions'], 
                                infos=None,
                                terminal=traj['terminals'][-1],
                                rews=traj['rewards'][:, 0])
        if "Hopper" in env_name: 
            if traj.rews.sum() < 3000:
                print(traj.rews.sum())
                continue 
        if "HalfCheetah" in env_name: 
            if traj.rews.sum() < 7000:#(12000 if "1-" not in env_name else 5000):
                print(traj.rews.sum())
                continue 
        if "Ant" in env_name: 
            if traj.rews.sum() < (5000 if "1-" not in env_name else 3000):
                print(traj.rews.sum())
                continue 
        i += 1 
        demonstrations.append(traj)
        if len(demonstrations) >= NUM_DEMO:
            break 

    with open(osp.join(demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl"), "wb") as savepkl:
        pickle.dump(demonstrations, savepkl)
        print("Save demo to", demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl")

assert len(demonstrations) == NUM_DEMO

from imitation.data import rollout 
demo_stats = rollout.rollout_stats(demonstrations)
print(env_name+"_"+str(NUM_DEMO))
print("return:")
print("rollout/ep_rew_min", demo_stats["return_min"])
print("rollout/ep_rew_mean", demo_stats["return_mean"])
print("rollout/ep_rew_std", demo_stats["return_std"])
print("len:")
print("rollout/ep_len_mean", demo_stats["len_mean"])
print("rollout/ep_len_std", demo_stats["len_std"])


print()
print('### '*20)
print('### '*20)
print()

# for env_name in ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2", 
#                  "Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Ant-v3", 
#                  "Hopper1-v2", "HalfCheetah1-v2", "Walker2d1-v2", "Ant1-v2", ]:
#     print(env_name, ":")
#     with open(osp.join(demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl"), "rb") as savepkl:
#         demonstrations = pickle.load(savepkl)
#         print("Load demo from", demo_dir, env_name+"_"+str(NUM_DEMO)+".pkl")

#         from imitation.data import rollout 
#         demo_stats = rollout.rollout_stats(demonstrations)
#         print(env_name+"_"+str(NUM_DEMO))
#         print("return:")
#         print("rollout/ep_rew_min", demo_stats["return_min"])
#         print("rollout/ep_rew_mean", demo_stats["return_mean"])
#         print("rollout/ep_rew_std", demo_stats["return_std"])
#         print("len:")
#         print("rollout/ep_len_mean", demo_stats["len_mean"])
#         print("rollout/ep_len_std", demo_stats["len_std"])
#         print()

# # env_name="Hopper-v3" # 3234
# # env_name="HalfCheetah-v3" # 12135
# # env_name="Walker2d-v3" # 4592
# # env_name="Ant-v3" 

