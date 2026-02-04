import argparse

import random 
import numpy as np 
import torch 


from imitation.rewards.reward_nets import BasicRewardNet

from ceil.actor import ContextualActor
# from ceil.encoder import TrajEncoder
# from ceil.encoder_ import TrajEncoder
from ceil.encoder_mlp import TrajEncoder
from ceil.ceil import CEIL

import pickle
import os.path as osp, time, atexit, os

import datetime

def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = torch.float32
    torch.set_default_dtype(dtype)

def get_args():
    parser = argparse.ArgumentParser(description='CEIL arguments')
    # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    parser.add_argument('--source', default="Hopper-v2", 
                        help='training interaction/offline-data environment')
    parser.add_argument('--target', default="Hopper-v2", 
                        help='testing/demo environment')
    parser.add_argument('--num', type=int, default=20)
    parser.add_argument('--mode', default="online", help='online or offline-m or offline-mr or offline-me or offline-e')
    parser.add_argument('--demo', default="lfd", help='lfd or lfo')
    parser.add_argument('--seed', type=int, default=2)
    
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--context_triplet_margin', type=float, default=0.0)
    
    parser.add_argument('--context_size', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=2)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    setup(args.seed)
    
    assert args.source in ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    assert args.target in ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2", 
                 "Hopper1-v2", "HalfCheetah1-v2", "Walker2d1-v2", "Ant1-v2", ]
    assert args.mode in ["online", "offline-m", "offline-mr", "offline-me", "offline-e"]
    assert args.demo in ["lfd", "lfo"]
    
    with open(osp.join("_data/demo/", args.target+"_"+str(20)+".pkl"), "rb") as savepkl: 
        demonstrations = pickle.load(savepkl) # TrajectoryWithRew 

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger_folder=str.format(
                "ceil/logs/{}_{}_{}_{}_{}/window{}_context{}_seed{}_{}", 
                args.source, args.target, args.mode, args.demo, args.num,
                args.window_size, args.context_size, args.seed, time_str
            )

    ceil_trainer = CEIL(
        source_env_name=args.source,
        target_env_name=args.target,
        mode=args.mode,
        demo=args.demo,
        demonstrations=demonstrations[:args.num],
        
        actor_cls=ContextualActor,
        encoder_cls=TrajEncoder,
        window_size=args.window_size,
        context_size=args.context_size,
        disc_cls=BasicRewardNet,
        cdmi_cls=BasicRewardNet,
        alpha=args.alpha,

        n_steps=1,
        n_context_updates_per_round=0,
        n_HIM_updates_per_round=15,
        n_disc_updates_per_round=0,
        n_cdmi_updates_per_round=2,
        traj_batch_size=1024*2*2//args.window_size,
        context_triplet_margin=args.context_triplet_margin,
        
        seed=args.seed,
        device="cuda",

        c_logger_folder=logger_folder,
        c_normalize_states=True,
    )
    ceil_trainer.train(int(1e6))
    
    print('Done!')




# hopper 2 0 



 
