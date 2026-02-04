from typing import Sequence 
from imitation.data import types
import numpy as np
import pickle

def load_demo(
    env_name,
    num_demo, 
    rng
)->Sequence[types.TrajectoryWithRew]:
    print("Loading...")
    path = "logs/demo/"+env_name+".pkl"
    with open(path, "rb") as savepkl:
        demonstrations = pickle.load(savepkl)
        print("Load demo from", path)
    
    assert len(demonstrations)>=num_demo, 'demonstration size less than required (n_demo)'

    # Sample random `num_trajectories` experts.
    perm = np.arange(len(demonstrations))
    perm = rng.permutation(perm)
    
    return np.array(demonstrations)[perm[:num_demo]]