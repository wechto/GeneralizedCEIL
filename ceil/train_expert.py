import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals  # needed to load environments

env = gym.make('HalfCheetah-v2')
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
    verbose=1,
    tensorboard_log="logs/expert/HalfCheetah",
)
expert.learn(
    int(1e6),
    # log_interval = 100,
)  # Note: set to 100000 to train a proficient expert

expert.save("HalfCheetah_expert")