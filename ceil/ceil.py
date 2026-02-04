"""CEIL"""

from typing import Callable, Iterable, Iterator, Optional, Type, overload, Union

import copy
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import base_class, vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, get_action_dim

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.gail import RewardNetFromDiscriminatorLogit
from imitation.util.util import make_vec_env
from imitation.data.rollout import generate_trajectories 
from imitation.data.rollout import rollout_stats 
from imitation.data import rollout, types, wrappers
from imitation.util import logger, networks, util
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger

from ceil.buffer import TrajectoryBuffer

import datetime
import tqdm
import logging


import gym
import dataclasses
import torch.utils.tensorboard as thboard

class EMA():
    '''
        empirical moving average 
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model, beta=None):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight, beta=beta)

    def update_average(self, old, new, beta=None):
        if old is None:
            return new
        if beta is None:
            beta = self.beta
        return old * beta + (1 - beta) * new


class CEIL(base.DemonstrationAlgorithm[types.Transitions]):
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    # venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    _demo_data_loader: Optional[Iterable[base.TransitionMapping]]
    _endless_expert_iterator: Optional[Iterator[base.TransitionMapping]]

    venv_wrapped: vec_env.VecEnvWrapper

    def __init__(
        self,
        *,
        source_env_name:str,
        target_env_name:str,
        mode:str,
        demo:str,
        demonstrations: base.AnyTransitions,
    
        actor_cls,
        encoder_cls,
        window_size: int,
        context_size: int,
        disc_cls,
        cdmi_cls,
        alpha,
        
        n_steps: int,
        n_cdmi_updates_per_round: int,
        n_context_updates_per_round: int,
        n_HIM_updates_per_round: int,
        n_disc_updates_per_round:int,
        traj_batch_size: int,
        context_triplet_margin: float,
        
        seed: int,
        c_logger_folder: str,
        c_normalize_states: bool=False,
        ema_decay: float=0.0,

        replay_buffer_capacity: int= 200, # 100 lfd; 500 lfo;
        
        device: Union[th.device, str] = "cuda",
    ):
        """Builds Trainer."""

        c_logger= imit_logger.configure(
            folder=c_logger_folder,
            format_strs=["stdout", "log", "csv", "tensorboard"]
        )
        self.c_normalize_states = c_normalize_states
        
        rng = np.random.default_rng(seed)
        self.source_env_name = source_env_name
        self.target_env_name = target_env_name
        # TODO 
        source_venv = make_vec_env(source_env_name, n_envs=10, rng=rng, max_episode_steps=1000 if "heetah" not in self.source_env_name else 200)
        self.target_venv = make_vec_env(target_env_name, n_envs=5, rng=rng) #, max_episode_steps=1000)
        
        self.mode = mode #["online", "offline-m", "offline-mr", "offline-me", "offline-e"]
        self.demo = demo #["lfd", "lfo"]
        
        # length for context inference
        self.window_size = window_size
        self.context_size = context_size
        
        self.venv = source_venv
        self.venv_train = wrappers.BufferingWrapper(self.venv)
        self.venv_eval = wrappers.BufferingWrapper(self.target_venv)
        self.venv_eval_source = wrappers.BufferingWrapper(make_vec_env(source_env_name, n_envs=3, rng=rng))
        # set up for env rollout
        self._last_obs = self.venv_train.reset()
        self.n_envs = self._last_obs.shape[0]
        self._last_ts = np.array((0,) * self.n_envs)
        
        self.obs_dim = get_flattened_obs_dim(self.venv.observation_space)
        self.action_dim = get_action_dim(self.venv.action_space)

        self.replay_buffer = None
        
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=c_logger,
            allow_variable_horizon=True,
        )  
        
        self.alpha = alpha

        self._global_step = 0
        self._context_step = 0
        self._hindsight_step = 0
        self._disc_step=0
        
        self.n_steps= n_steps
        self.n_context_updates_per_round = n_context_updates_per_round
        self.n_HIM_updates_per_round = n_HIM_updates_per_round
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.n_cdmi_updates_per_round = n_cdmi_updates_per_round
        self.traj_batch_size = traj_batch_size
        
        self.device = device

        if self.replay_buffer is None: 
            if "heetah" in self.source_env_name: 
                replay_buffer_capacity = 1000
            self.replay_buffer=TrajectoryBuffer(replay_buffer_capacity, 
                                                self.obs_dim, self.action_dim, 
                                                length=self.window_size)
        
        
        self.triplet_loss = th.nn.TripletMarginLoss(margin=context_triplet_margin)
        
        self.set_random_seed(seed)
        self.ema = EMA(ema_decay)
        
        self._setup_model(
            actor_cls, 
            encoder_cls, 
            disc_cls, 
            cdmi_cls,
        )

        self.stats_len_mean = 0
        

        
    def set_demonstrations(self, demo_trajs: Iterable[types.Trajectory]) -> None:
        n_demos = len(demo_trajs)
        self.demo_buffer = TrajectoryBuffer(n_demos, self.obs_dim, self.action_dim, length=self.window_size, is_demo=True)
        self.demo_buffer.store(demo_trajs)
        
        if self.mode in ["offline-m", "offline-mr", "offline-me", "offline-e"]:
            import d4rl
            from imitation.data.types import TrajectoryWithRew 
            # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
            temp_d4rl_name = {"Hopper-v2": "hopper-", 
                              "HalfCheetah-v2": "halfcheetah-", 
                              "Walker2d-v2": "walker2d-", 
                              "Ant-v2": "ant-"}[self.source_env_name]
            temp_d4rl_name += {"offline-m": "medium-v2", 
                               "offline-mr": "medium-replay-v2", 
                               "offline-me": "medium-expert-v2", 
                               "offline-e": "expert-v2"}[self.mode]
            temp_env = gym.make(temp_d4rl_name)
            temp_dataset = temp_env.get_dataset()
            temp_N = temp_dataset['observations'].shape[0]
            temp_trajs = []
            temp_traj = {"obs":[], "acts":[], "rews":[], "dones":[]} 

            use_timeouts = False
            if 'timeouts' in temp_dataset:
                use_timeouts = True 

            episode_step = 0
            
            for i in range(temp_N):
                temp_traj["obs"] += [temp_dataset['observations'][i]]
                temp_traj["acts"] += [temp_dataset['actions'][i]]
                temp_traj["rews"] += [temp_dataset['rewards'][i]]

                temp_done_bool = bool(temp_dataset['terminals'][i])
                if use_timeouts:
                    final_timestep = temp_dataset['timeouts'][i]
                else:
                    final_timestep = (episode_step == temp_env._max_episode_steps-1)

                episode_step += 1
                if  temp_done_bool or final_timestep:
                    temp_traj["obs"] += [temp_dataset['next_observations'][i]]
                    assert np.array(temp_traj["obs"]).shape[1] > 2
                    temp_traj = TrajectoryWithRew(obs=np.array(temp_traj["obs"]),
                                                    acts=np.array(temp_traj["acts"]),
                                                    infos=None,
                                                    terminal=True,
                                                    rews=np.array(temp_traj["rews"]))
                    temp_trajs.append(temp_traj)
                    temp_traj = {"obs":[], "acts":[], "rews":[], "dones":[]}
                    episode_step = 0

            self.replay_buffer=TrajectoryBuffer(len(temp_trajs), 
                                                self.obs_dim, self.action_dim, 
                                                length=self.window_size)
            self.replay_buffer.store(temp_trajs, is_demo=False)
        
        self.temp_norm_min = self.demo_buffer.norm[0]
        self.temp_norm_max = self.demo_buffer.norm[1]
    
    def _setup_model(
        self,
        actor_cls, 
        encoder_cls, 
        disc_cls,
        cdmi_cls
    )-> None:
        
        actor_args={
            "obs_dim": self.obs_dim,
            "context_size": self.context_size,
            "action_dim": self.action_dim,
            "output_max": self.venv.action_space.high[0],
        }
        
        encoder_args={
            "horizon": self.window_size,
            "transition_dim": self.obs_dim, # int(self.obs_dim + self.action_dim * (self.demo in ["lfd"])),
            "context_dim": self.context_size,
            "obs_dim": self.obs_dim,
            "act_dim": self.action_dim,
        }
        # encoder_args={
        #     "num_inputs": int(self.obs_dim + self.action_dim * (self.demo in ["lfd"])),
        #     "num_outputs": self.context_size,
        # }
        
        disc_args={
            "observation_space":self.venv.observation_space,
            "action_space":self.venv.action_space,
            "use_action":False, # True if self.demo in ["lfd"] else False,
            "use_next_state":True, 
            # "normalize_input_layer":RunningNorm,
        }
        
        cdmi_args={
            "observation_space":gym.spaces.Box(np.array([-1]*(self.context_size+2*1)), np.array([1]*(self.context_size+2*1))),
            "action_space":self.venv.action_space,
            "use_action":False,
            "use_next_state":False,
            "hid_sizes": (128, 128), 
            # "dropout_prob": 0.1, 
            # "normalize_input_layer":RunningNorm,
        }
        
        self.actor = actor_cls(**actor_args)
        self.encoder = encoder_cls(**encoder_args)
        self.disc=disc_cls(**disc_args)
        self.cdmi=cdmi_cls(**cdmi_args)
        
        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=3e-4)# 1e-3 3e-4 2e-4 1e-4 
        self.disc_opt = th.optim.Adam(self.disc.parameters(), lr=3e-4)
        self.cdmi_opt = th.optim.Adam(self.cdmi.parameters(), lr=3e-4)

        self.actor_lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.actor_opt, 
                                                                                    T_0 = 1000,
                                                                                    T_mult=1,
                                                                                    eta_min=1e-5) # 1e-4
        self.encoder_lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.encoder_opt, 
                                                                                    T_0 = 1000,
                                                                                    T_mult=1,
                                                                                    eta_min=1e-5) # 1e-4
        
        self.actor.to(self.device)
        self.encoder.to(self.device)
        self.disc.to(self.device)
        self.cdmi.to(self.device)
        
        self.context = th.nn.Parameter(self.init_context(), requires_grad=True)
        self.context_opt = th.optim.Adam([self.context], lr=3e-4)
        self.context.to(self.device)

        self.old_context = copy.deepcopy(self.context)
        self.context_buffer = copy.deepcopy(self.context)
        
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.old_actor = copy.deepcopy(self.actor)
        
    
    def set_random_seed(self, seed) -> None:
        set_random_seed(seed, using_cuda=True)
        self.venv.seed(seed)
        self.venv_train.seed(seed)
        self.venv_eval.seed(seed)
        self.venv_eval_source.seed(seed)

        self.venv.action_space.seed(seed)
        self.venv_train.action_space.seed(seed)
        self.venv_eval.action_space.seed(seed)
        self.venv_eval_source.action_space.seed(seed)

        
    def init_context(self, ) -> th.Tensor:
        self.mean, self.std = 0., 0.
        with th.no_grad():
            demo_samples, demo_ts, demo_td = self.demo_buffer.sample(self.traj_batch_size)
            self._set_norm_obs_mean_std(demo_samples[:,:,:self.obs_dim], demo_samples[:,:,:self.obs_dim], w=0.)

            demo_samples = th.as_tensor(demo_samples).to(self.device)
            demo_ts = th.as_tensor(demo_ts).to(self.device).long()
            demo_embedding = self.encoder(demo_samples, demo_ts, 
                                          conp=True, obsp=False, actp=False, transip=False)[0].mean(0).detach().cpu()
        return demo_embedding

    @property
    def policy(self):
        return self.actor, self.context
    
    def get_action(self, obs):
        if len(obs.shape) == 1:
            obs = np.array([obs])
        obs_tensor = obs_as_tensor(obs, self.device)
        obs_tensor = self._norm_obs(obs_tensor)
        return self.actor(obs_tensor, self.context.to(self.device))[0]
        
    def policy_rollout(self, n_rollout_steps: int, n_rollout)->None:
        # Switch to eval mode (this affects batch norm / dropout)
        # self.actor.set_training_mode(False)
        # self.encoder.set_traning_mode(False)
        self.actor.eval()

        assert self.mode in ["online"] 
        
        if self.mode in ["online"]: 
            n_steps = 0
            while n_steps < n_rollout_steps:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device).to(th.float32)
                    obs_tensor = self._norm_obs(obs_tensor)
                    ts_tensor = obs_as_tensor(np.array(self._last_ts, dtype="int32"), self.device).long()
                    # obs_tensor = th.cat((obs_tensor, ts), 1)
                    alpha_cr = 1.0 #0.5 + 0.5 * np.random.rand()
                    # if np.random.rand() <= 0.001:
                    #     alpha_cr = 0.5 + 0.5 * np.random.rand()
                    context_run = self.context * alpha_cr + \
                                    self.context_buffer * (1 - alpha_cr)

                    actor_actions_mean, actor_actions = self.actor(obs_tensor, context_run.to(self.device))
                
                actions = actor_actions.cpu().numpy()
                if np.random.rand() <= 0.01:
                    actions += np.random.randn(*actions.shape) * 0.01

                if n_rollout < 1:
                    actions = np.array([self.venv_train.action_space.sample() for i in range(self.n_envs)])

                clipped_actions = actions # Rescale and perform action
                if isinstance(self.venv_train.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.venv_train.action_space.low, self.venv_train.action_space.high)

                new_obs, _, dones, _ = self.venv_train.step(clipped_actions)
                n_steps += 1
                self._last_obs = new_obs
                self._last_ts = (1 - dones) * (self._last_ts + 1) % self.window_size

            # # self._global_step += 1
            trajs, _ = self.venv_train.pop_finished_trajectories()
            if len(trajs) > 0:
                self.replay_buffer.store(trajs)
                # self.eval_buffer.store(trajs)
                
            stats, info = self.replay_buffer.stats()
            # stats = self.eval_buffer.stats()
        if stats is None: return; 
        self.logger.record("rollout/number", len(trajs))
        self.logger.record("rollout/ep_rew_mean", stats["return_mean"])
        self.logger.record("rollout/ep_rew_std", stats["return_std"])
        self.logger.record("rollout/ep_len_mean", stats["len_mean"])
        self.logger.record("rollout/ep_len_std", stats["len_std"])
        self.logger.record("buffer/Num", info['Num'])
        self.logger.record("buffer/Max", info['Max'])
        self.logger.record("buffer/Mean", info['Mean'])
        self.logger.record("buffer/Nbuffer", info['Nbuffer'])

        self.stats_len_mean = stats["len_mean"]

        if "heetah" in self.source_env_name:
            self.policy_evaluate(1001, n_rollout)
    
    def policy_evaluate(self, n_rollout_steps: int, n_rollout: int)->None:
        if n_rollout % 1000 == 0:
            self.actor.eval()
            last_obs = self.venv_eval.reset()
            n_steps = 0
            while n_steps < n_rollout_steps:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(last_obs, self.device).to(th.float32)
                    obs_tensor = self._norm_obs(obs_tensor)
                    actor_actions_mean, _ = self.actor(obs_tensor, self.context.to(self.device))
                actions = actor_actions_mean.cpu().numpy()
                clipped_actions = actions
                if isinstance(self.venv_eval.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.venv_eval.action_space.low, self.venv_eval.action_space.high)

                new_obs, _, dones, _ = self.venv_eval.step(clipped_actions)
                last_obs = new_obs
                n_steps += 1
            trajs, _ = self.venv_eval.pop_finished_trajectories()
            self.evaluate_number = len(trajs)
            self.evaluate_stats = rollout_stats(trajs)

        self.logger.record("evaluate/number", self.evaluate_number)
        self.logger.record("evaluate/ep_rew_mean", self.evaluate_stats["return_mean"])
        self.logger.record("evaluate/ep_rew_std", self.evaluate_stats["return_std"])
        self.logger.record("evaluate/ep_len_mean", self.evaluate_stats["len_mean"])
        self.logger.record("evaluate/ep_len_std", self.evaluate_stats["len_std"])

        if self.source_env_name != self.target_env_name:
            if n_rollout % 1000 == 0:
                self.actor.eval()
                last_obs = self.venv_eval_source.reset()
                n_steps = 0
                while n_steps < n_rollout_steps:
                    with th.no_grad():
                        obs_tensor = obs_as_tensor(last_obs, self.device).to(th.float32)
                        obs_tensor = self._norm_obs(obs_tensor)
                        actor_actions_mean, _ = self.actor(obs_tensor, self.context.to(self.device))
                    actions = actor_actions_mean.cpu().numpy()
                    clipped_actions = actions
                    if isinstance(self.venv_eval_source.action_space, gym.spaces.Box):
                        clipped_actions = np.clip(actions, self.venv_eval_source.action_space.low, self.venv_eval_source.action_space.high)

                    new_obs, _, dones, _ = self.venv_eval_source.step(clipped_actions)
                    last_obs = new_obs
                    n_steps += 1
                trajs, _ = self.venv_eval_source.pop_finished_trajectories()
                self.evaluate_number_source = len(trajs)
                self.evaluate_stats_source = rollout_stats(trajs)
            self.logger.record("evaluate_source/number", self.evaluate_number_source)
            self.logger.record("evaluate_source/ep_rew_mean", self.evaluate_stats_source["return_mean"])
            self.logger.record("evaluate_source/ep_rew_std", self.evaluate_stats_source["return_std"])
            self.logger.record("evaluate_source/ep_len_mean", self.evaluate_stats_source["len_mean"])
            self.logger.record("evaluate_source/ep_len_std", self.evaluate_stats_source["len_std"])




    def train_cdmi(self, demo_samples, buffer_samples, demo_ts, buffer_ts,):
        self.encoder.train()
        self.cdmi.train()

        with self.logger.accumulate_means("cdmi"):
            # compute loss
            self.cdmi_opt.zero_grad()
            self.encoder_opt.zero_grad()
            
            demo_samples = th.as_tensor(demo_samples).to(self.device)
            # buffer_samples = th.as_tensor(buffer_samples).to(self.device)

            noised_samples = demo_samples.detach()
            # noised_samples = buffer_samples.detach()
            noised_samples = noised_samples + (th.rand(noised_samples.shape).to(self.device) - 0.5) * 0.2
            
            demo_ts = th.as_tensor(demo_ts).to(self.device).long()


            demo_embedding = self.encoder(demo_samples, demo_ts, 
                                                       conp=True, obsp=False, actp=False, transip=False)[0]#.detach()
            noised_embedding  = self.encoder(noised_samples, demo_ts, 
                                                       conp=True, obsp=False, actp=False, transip=False)[0]#.detach()
            
            # 10: demo 
            # 01: noised 
            n_demo_embedding = th.Tensor([1., 0.]*1).to(self.device).repeat(self.traj_batch_size,1)
            n_noised_embedding = th.Tensor([0., 1.]*1).to(self.device).repeat(self.traj_batch_size,1)
            
            tiled_demo_ = th.cat([demo_embedding, noised_embedding,], dim=0)
            tiled_n_demo_ = th.cat([n_demo_embedding, n_noised_embedding,], dim=0)
            idx_ = th.randperm(self.traj_batch_size*2)
            shuffled_tiled_n_demo_ = tiled_n_demo_[idx_]
            
            temp_inputs = th.cat([th.cat([tiled_demo_, tiled_demo_],dim=0), 
                                  th.cat([tiled_n_demo_, shuffled_tiled_n_demo_],dim=0)], 
                                 dim=1)
            logits = self.cdmi(temp_inputs, None, None, None)
            pred_joint_ = logits[: self.traj_batch_size*2]
            pred_marginal_ = logits[self.traj_batch_size*2 :]
            loss = - (th.mean(pred_joint_) - th.log(th.mean(th.exp(pred_marginal_))))
            
            loss.backward()
            self.cdmi_opt.step()
            self.encoder_opt.step()
            
            self.logger.record("loss", float(loss.detach().cpu()))
            

            
            


    def train_HIM(self, update_num, demo_samples, buffer_samples, demo_ts, buffer_ts, demo_td, buffer_td,):
        self.encoder.train()
        self.actor.train()
        self.ema_encoder.eval()

        with self.logger.accumulate_means("HIM"):
            
            self.encoder_opt.zero_grad()
            self.actor_opt.zero_grad()

            NN = 1 
            window_size_NN = int(self.window_size*NN)
            
            demo_samples = th.as_tensor(demo_samples).to(self.device)
            buffer_samples = th.as_tensor(buffer_samples).to(self.device)
            demo_ts = th.as_tensor(demo_ts).to(self.device).long()
            buffer_ts = th.as_tensor(buffer_ts).to(self.device).long()
            demo_td = th.as_tensor(demo_td).to(self.device).unsqueeze(1)
            buffer_td = th.as_tensor(buffer_td).to(self.device).unsqueeze(1)

            
            
            demo_embedding, demo_loss_vq, demo_pre_obs, demo_pre_act, demo_transi, _  = self.encoder(demo_samples, demo_ts, 
                                        conp=True, obsp=True, actp=(False if self.demo == 'lfd' else True), transip=False)
            # demo_embedding = demo_embedding.mean(0).repeat(self.traj_batch_size, 1)
            buffer_embedding, buffer_loss_vq, buffer_pre_obs, buffer_pre_act, buffer_transi, _ = self.encoder(buffer_samples, buffer_ts, 
                                        conp=True, obsp=True, actp=(False if self.demo == 'lfd' else True), transip=False)
            
            
            range_obs_dim_terminal = [*range(self.obs_dim),-1]

            loss_dec_demo_obs = F.mse_loss(demo_pre_obs[:,:-1,:],# + demo_samples[:,:1,range_obs_dim_terminal].detach(), 
                                           demo_samples[:,1:,range_obs_dim_terminal].detach()) * 100. 
            loss_dec_buffer_obs = F.mse_loss(buffer_pre_obs[:,:-1,:],# + buffer_samples[:,:1,range_obs_dim_terminal].detach(), 
                                             buffer_samples[:,1:,range_obs_dim_terminal].detach()) * 100.
            

            loss_pre_buffer_act = 0. if self.demo == 'lfd' else \
                F.mse_loss(buffer_pre_act, buffer_samples[:, :-1, -self.action_dim-1:-1])
            


            loss_positive_mean = F.mse_loss(demo_embedding, 
                                            demo_embedding.mean(0, keepdim=True).repeat(self.traj_batch_size,1).detach()) * 10000.
                                            # self.context.to(self.device).unsqueeze(0).repeat(self.traj_batch_size,1).detach()) * 10000.
            
            
            
            # no no       de no
            # no de ok    de de
            # demo_embedding = demo_embedding + (self.context.to(self.device) - demo_embedding).detach()
            demo_embedding_ = demo_embedding.unsqueeze(1).repeat(1,window_size_NN,1)#.detach()
            # demo_embedding_ = demo_embedding.mean(0, keepdim=True).unsqueeze(1).repeat(self.traj_batch_size,window_size_NN,1)#.detach()
            buffer_embedding_ = buffer_embedding.unsqueeze(1).repeat(1,window_size_NN,1)#.detach()
            
            assert demo_samples.shape[2] == self.obs_dim+self.action_dim+1
            demo_samples_ = demo_samples[:, :window_size_NN, :self.obs_dim+self.action_dim]
            buffer_samples_ = buffer_samples[:, :window_size_NN, :self.obs_dim+self.action_dim]
            demo_ts_ = demo_ts[:, :window_size_NN]#.reshape(-1, self.obs_dim+self.action_dim)
            buffer_ts_ = buffer_ts[:, :window_size_NN]#.reshape(-1, self.obs_dim+self.action_dim)
            
            # noise = th.rand(self.traj_batch_size*2*window_size_NN, self.obs_dim).to(self.device) * 0.0002 - 0.0001
            samples_cat_ = th.cat([demo_samples_[:,:, :self.obs_dim], buffer_samples_[:,:, :self.obs_dim]],0).detach()
            ts_cat_ = th.cat([demo_ts_, buffer_ts_],0).detach()
            predicted_actions_, _ = self.actor(samples_cat_, 
                                    th.cat([demo_embedding_, buffer_embedding_],0), )
            
            # TODO: lfd lfo 
            loss_demo_act = F.mse_loss(predicted_actions_[:self.traj_batch_size,:-1], 
                                       demo_samples_[:,:-1,-self.action_dim:].detach() if self.demo == 'lfd' else \
                                       demo_pre_act.detach()
                                       ) * 2.71828
            loss_buffer_act = F.mse_loss(predicted_actions_[self.traj_batch_size:,:-1], 
                                         buffer_samples_[:,:-1,-self.action_dim:].detach(), 
                                         reduction='none').mean(-1)
            with th.no_grad():
                self_context_unsqueeze = self.context.to(self.device).unsqueeze(0).unsqueeze(0).repeat(self.traj_batch_size,window_size_NN,1)
                anchor_negative = th.norm(self_context_unsqueeze - buffer_embedding_, dim=-1)[:, :-1] + 1e-6
            loss_buffer_act = th.mean(loss_buffer_act \
                                    * th.exp((1. - anchor_negative / anchor_negative.max(0,keepdim=True).values) * 1.).detach()\
                                    )# 
            
            loss_negative_mean = F.mse_loss(buffer_embedding - self.context.to(self.device).detach(), 
                                            th.clip(buffer_embedding - self.context.to(self.device), -1, 1).detach()) * 1.
            loss_positive_mean += loss_negative_mean

            

            loss = loss_positive_mean * 10. \
                    + loss_dec_buffer_obs * 100. + loss_dec_demo_obs * 100. \
                    + (demo_loss_vq + buffer_loss_vq) * 1. \
                    + loss_buffer_act * 100. + loss_demo_act  * 100. \
                    + loss_pre_buffer_act * 100
                    
            loss.backward()
            
            self.encoder_opt.step()
            self.actor_opt.step()
            
            self.encoder_lr_scheduler.step()
            self.actor_lr_scheduler.step()

            # print(self.encoder_opt.param_groups[0]["lr"], self.actor_opt.param_groups[0]["lr"])

            self._hindsight_step += 1
            
            self.logger.record("loss_act_demo", float(loss_demo_act.detach().cpu()))
            # self.logger.record("loss_act_demo_(t)", float(loss_temp_demo_act.detach().cpu()))
            self.logger.record("loss_act_buffer", float(loss_buffer_act.detach().cpu()))
            self.logger.record("loss_dec_demo_obs", float(loss_dec_demo_obs.detach().cpu()))
            self.logger.record("loss_dec_buffer_obs", float(loss_dec_buffer_obs.detach().cpu()))
            
            self.logger.record("loss_vq_demo", float(demo_loss_vq.detach().cpu()))
            self.logger.record("loss_vq_buffer", float(buffer_loss_vq.detach().cpu()))

            self.logger.record("loss_pre_buffer_act", float(0.) if self.demo == 'lfd' else \
                               float(loss_pre_buffer_act.detach().cpu()))

            

            # self.logger.dump(self._hindsight_step)

        with th.no_grad():
            self.ema_encoder.eval()
            self.ema.update_model_average(self.ema_encoder, self.encoder)

            self.context = self.context*0.5 + 0.5*self.ema_encoder(demo_samples, demo_ts, 
                                            conp=True, obsp=False, actp=False, transip=False)[0].mean(0).cpu()
            self.context_buffer = self.ema_encoder(buffer_samples, buffer_ts, 
                                                   conp=True, obsp=False, actp=False, transip=False)[0][:2].mean(0).cpu()
        
        return self.context

        
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        
        n_rounds = total_timesteps // (self.n_steps*self.venv.num_envs)
        if self.mode in ["offline-m", "offline-mr", "offline-me", "offline-e"]: 
             n_rounds = total_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.HIM_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            
            "rollout the policy, store the trajectory in the buffer"
            if self.mode in ["online"]:
                self.policy_rollout(self.n_steps if round > 0 else 1001, round)
                # self.policy_evaluate(1001, round)

            # if self.mode in ["offline-m", "offline-mr", "offline-me", "offline-e"]: 
            #     self.policy_evaluate(1001, round)
            self.policy_evaluate(1001, round)
            
            self._global_step += 1

            
            for i in range(max(self.n_disc_updates_per_round, 
                               self.n_HIM_updates_per_round, 
                               self.n_context_updates_per_round,
                               self.n_cdmi_updates_per_round,
                               )):
                
                demo_samples, demo_ts, demo_td = self.demo_buffer.sample(self.traj_batch_size)
                buffer_samples, buffer_ts, buffer_td = self.replay_buffer.sample(self.traj_batch_size)

                self._set_norm_obs_mean_std(demo_samples[:, :, :self.obs_dim], buffer_samples[:, :, :self.obs_dim], 
                                            w=(0.50 if round < 10 else 0.90), )
                demo_samples[:, :, :self.obs_dim] = self._norm_obs(demo_samples[:, :, :self.obs_dim])
                buffer_samples[:, :, :self.obs_dim] = self._norm_obs(buffer_samples[:, :, :self.obs_dim])
                
                
                if i < self.n_HIM_updates_per_round:
                    # with networks.training(self.encoder):
                    old_context = self.train_HIM(i, demo_samples, buffer_samples, demo_ts, buffer_ts, demo_td, buffer_td, )
                    
                
                if i < self.n_cdmi_updates_per_round:
                    self.train_cdmi(demo_samples, buffer_samples, demo_ts, buffer_ts,)
            

            
            
            with th.no_grad():
                self.ema_encoder.eval()
                self.old_context = old_context
                self.ema.update_model_average(self.old_actor, self.actor, beta=0.)
                
                demo_embedding = self.ema_encoder(th.as_tensor(demo_samples).to(self.device), 
                                                  th.as_tensor(demo_ts).to(self.device).long(),
                                                  conp=True, obsp=False, actp=False, transip=False)[0]
                buffer_embedding = self.ema_encoder(th.as_tensor(buffer_samples).to(self.device), 
                                                    th.as_tensor(buffer_ts).to(self.device).long(),
                                                    conp=True, obsp=False, actp=False, transip=False)[0]
                

                anchor_positive = th.norm(self.context.to(self.device).unsqueeze(0).repeat(self.traj_batch_size,1) - demo_embedding, dim=-1).mean()
                anchor_negative = th.norm(self.context.to(self.device).repeat(self.traj_batch_size,1) - buffer_embedding, dim=-1).mean()
                

                anchor_mean = F.mse_loss(self.context.to(self.device), demo_embedding.mean(0))
            
            self.logger.record("anchor/anchor_mean", float(anchor_mean.cpu()))
            self.logger.record("anchor/anchor_positive", float(anchor_positive.cpu()))
            self.logger.record("anchor/anchor_negative", float(anchor_negative.cpu()))
            # self.logger.record("anchor/anchor_aug", float(anchor_aug.cpu()))
            # self.logger.record("anchor/positive_mean", float(positive_mean.cpu()))
            # self.logger.record("anchor/positive_negative", float(positive_negative.cpu()))

            # self.logger.record("anchor/z_context", self.context.detach().cpu().numpy().mean())
            self.logger.record("anchor/z_context(max)", self.context.detach().cpu().numpy().max())
            self.logger.record("anchor/z_context(min)", self.context.detach().cpu().numpy().min())
            
            self.logger.dump(self._global_step)

    def _make_HIM_dict(self, traj:types.TrajectoryWithRew):
        
        traj_in_o_a = np.concatenate([traj.obs[:-1],traj.acts],-1)
        
        HIM_dict={
            "traj_window": self._make_traj_window(traj_in_o_a),
            "obs": np.array(traj.obs[:-1], dtype="float32"),
            "acts":np.array(traj.acts, dtype="float32"),
        }
        
        HIM_dict_th = {k: th.as_tensor(v).to(self.device) for k, v in HIM_dict.items()}
        
        return HIM_dict_th
    
    
    def _make_disc_train_dict(self, samples):
        disc_train_dict={
            "obs": samples[:, :-1, :self.obs_dim].reshape(-1, self.obs_dim), # :
            "acts": None, # samples[:, :-1, self.obs_dim:].reshape(-1, self.action_dim) if self.demo in ["lfd"] else None, 
            "next_obs": samples[:, 1:, :self.obs_dim].reshape(-1, self.obs_dim),
        }
        disc_train_dict_th = {k: th.as_tensor(v).to(self.device) if v is not None else None for k, v in disc_train_dict.items()}
        
        return disc_train_dict_th
    
    def _set_norm_obs_mean_std(self, a, b, w=0.90):
        x = np.concatenate((a,b), 0).reshape(-1, self.obs_dim)
        mean, std = x.mean(0), x.std(0)
        self.mean = w*self.mean + (1-w)*mean
        self.std = w*self.std + (1-w)*std


    def _norm_obs(self, obs):
        # HalfCheetah 
        if "heetah" in self.source_env_name or self.c_normalize_states:
            self.temp_norm_min = np.min(np.concatenate([[self.demo_buffer.norm[0]], 
                                                    [self.replay_buffer.norm[0]]], 0), 0)[:self.obs_dim]
            self.temp_norm_max = np.max(np.concatenate([[self.demo_buffer.norm[1]], 
                                                    [self.replay_buffer.norm[1]]], 0), 0)[:self.obs_dim]
            temp1 = (self.temp_norm_max + self.temp_norm_min)/2.
            temp1 = self.temp_norm_min
            temp2 = self.temp_norm_max - self.temp_norm_min
            if type(obs) == th.Tensor:
                temp1 = th.Tensor(temp1).to(self.device)
                temp2 = th.Tensor(temp2).to(self.device)
            return (obs - temp1[:self.obs_dim])/temp2[:self.obs_dim] * 2.  - 1.
        # mean, std = self.mean, self.std
        # if type(obs) == th.Tensor:
        #     mean = th.Tensor(mean).to(self.device)
        #     std = th.Tensor(std).to(self.device)
        # return (obs - mean)/(std + 1e-6)
        return obs
    
# 200 500 800 1000 