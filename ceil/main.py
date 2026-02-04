from stable_baselines3 import PPO
from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np


expert = PPO.load("expert")


rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "CartPole-v0",
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=500, min_episodes=10),
    rng=rng,
)

"""rollouts dim: [n_traj * Trajectory]"""
print("n_traj:", len(rollouts))
print("traj length:", len(rollouts[0]))
# print("...:", len(rollouts[0][0]))

# print(rollouts[0].acts) 
# print(rollouts[0].obs)
# print(rollouts[0].rews)