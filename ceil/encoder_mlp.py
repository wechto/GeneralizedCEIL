
import torch
import torch.nn as nn
from torch.nn import functional as F

import einops
from einops.layers.torch import Rearrange
import math

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class TrajEncoder(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        context_dim,
        obs_dim,
        act_dim,
        embt_dim = 5,
        num_embeddings = 4096,
        hidden_size = 512,
        ensemble_num = 1,
    ):
        super().__init__()

        self.horizon = horizon
        self.transition_dim = transition_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.context_dim = context_dim
        self.embt_dim = embt_dim
        self.num_embeddings = num_embeddings * (horizon-1)

        # self.enc_t = nn.Embedding(self.horizon+1, self.embt_dim)
        # # self.enc_obs = nn.Linear(self.obs_dim+1, self.emb_dim)
        # # self.enc_obst = nn.Linear(self.emb_dim*2, self.emb_dim)

        self.enc_1 = nn.Linear(self.horizon*(self.obs_dim+1), hidden_size)
        self.enc_2 = nn.Linear(hidden_size, hidden_size)
        self.enc_3 = nn.Linear(hidden_size, hidden_size)
        self.enc_4 = nn.Linear(hidden_size, self.context_dim)

        # self.con = nn.Linear(self.context_dim, self.emb_dim)
        # self.con0 = nn.Linear(self.emb_dim, hidden_size)

        self.dec_t = nn.Embedding(self.horizon+1, self.embt_dim)
        self.dec_obs_1 = nn.Linear(self.context_dim+self.embt_dim+self.obs_dim+1, hidden_size)
        self.dec_obs_2 = nn.Linear(hidden_size, hidden_size)
        self.dec_obs_3 = nn.Linear(hidden_size, hidden_size)
        self.dec_obs_4 = nn.Linear(hidden_size, self.obs_dim+1)
        
        self.act_1 = nn.Linear(self.obs_dim*2, hidden_size)
        self.act_2 = nn.Linear(hidden_size, hidden_size)
        self.act_3 = nn.Linear(hidden_size, hidden_size)
        self.act_4 = nn.Linear(hidden_size, self.act_dim)


        self.ensemble_num = ensemble_num
        self.transi_models = nn.ModuleList([])
        for _ in range(ensemble_num):
            temp = nn.Sequential(nn.Linear(self.obs_dim+self.act_dim+1, hidden_size), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, self.obs_dim+1)
                                 )
            self.transi_models.append(temp)

        # # initialize embeddings
        # self.embeddings = nn.Embedding(self.num_embeddings, self.context_dim)

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.context_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)

        decay = 0.99
        self.epsilon = 1e-5
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)


    def forward(self, inputx, inputts, 
                conp = False, obsp = False, 
                actp = False, transip = False):
        '''
            x : [ batch x horizon x transition ]
        '''
        con, loss, enc_x = None, 0., None
        if conp:
            # ts = self.enc_t(torch.max(inputts - inputts[:, :1] + 1, torch.Tensor([0]).to(inputts.device).long()).detach())
            # # xp = self.enc_obs(inputx[:,:,[*range(self.obs_dim),-1]])
            # # xpt = self.enc_obst(torch.cat([xp, ts], -1))
            # xpt = torch.cat([ts, 
            #                  inputx[:,:,[*range(self.obs_dim),-1]]], -1)
            xpt = inputx[:,:,[*range(self.obs_dim),-1]]

            x = einops.rearrange(xpt, 'b h t -> b (h t)')

            x = F.relu(self.enc_1(x))
            x = F.relu(self.enc_2(x))
            x = F.relu(self.enc_3(x))
            enc_x = self.enc_4(x)
            # enc_x = F.clip(enc_x)
            # enc_x = torch.clip(enc_x, -1., 1.)

            encoding_indices = self.get_code_indices(enc_x)
            quantized = self.quantize(encoding_indices)


            # update embeddings with EMA
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
                updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
                n = torch.sum(updated_ema_cluster_size)
                updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
                dw = torch.matmul(encodings.t(), enc_x) # sum encoding vectors of each cluster
                updated_ema_dw = self.ema_dw(dw)
                normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
                self.embeddings.data = normalised_updated_ema_w


            # # embedding loss: move the embeddings towards the encoder's output
            # q_latent_loss = F.mse_loss(quantized, enc_x.detach())

            # commitment loss
            e_latent_loss = F.mse_loss(enc_x, quantized.detach())
            loss = e_latent_loss

            # Straight Through Estimator
            con = enc_x + (quantized - enc_x).detach()
            

        obs = None
        if obsp:
            assert conp 
            x = con.unsqueeze(1).repeat(1, self.horizon, 1)
            ts = self.dec_t(torch.max(inputts - inputts[:, :1] + 1, torch.Tensor([0]).to(inputts.device).long()).detach())
            xi0 = inputx[:, :1, [*range(self.obs_dim),-1]].repeat(1, self.horizon, 1)
            x = torch.cat([x, ts, xi0], -1)
            x = F.relu(self.dec_obs_1(x))
            x = F.relu(self.dec_obs_2(x))
            x = F.relu(self.dec_obs_3(x))
            obs = self.dec_obs_4(x)
        

        act = None
        if actp:
            act_x = torch.cat([inputx[:, :-1, :self.obs_dim], inputx[:, 1:, :self.obs_dim]], -1)
            act_x = F.relu(self.act_1(act_x))
            act_x = F.relu(self.act_2(act_x))
            act_x = F.relu(self.act_3(act_x))
            act = self.act_4(act_x)


        transiss = []
        if transip:
            for transi_model in self.transi_models:
                transiss.append(transi_model(inputx))

        return con, loss, obs, act, transiss, enc_x

    # def get_code_indices(self, flat_x):
    #     # compute L2 distance
    #     distances = (
    #         torch.sum(flat_x ** 2, dim=1, keepdim=True) +
    #         torch.sum(self.embeddings.weight ** 2, dim=1) -
    #         2. * torch.matmul(flat_x, self.embeddings.weight.t())
    #     ) # [N, M]
    #     encoding_indices = torch.argmin(distances, dim=1) # [N,]
    #     return encoding_indices

    # def quantize(self, encoding_indices):
    #     """Returns embedding tensor for a batch of indices."""
    #     return self.embeddings(encoding_indices)
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
        

