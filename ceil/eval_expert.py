import gym
from stable_baselines3 import SAC

from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import seals


import sys

print("loading expert model at", sys.argv[1])

expert = SAC.load(sys.argv[1])

env_name = sys.argv[1].split('/')[1]
rng = np.random.default_rng()
venv = make_vec_env(env_name, n_envs=10, rng=rng)

expert_rewards, _ = evaluate_policy(
    expert, venv, 10, return_episode_rewards=True
)

print("mean episoder reward (10 episode)", np.mean(expert_rewards))

