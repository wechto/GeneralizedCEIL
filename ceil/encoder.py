
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            # nn.Mish(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)    


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x inp_channels x horizon ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TrajEncoder(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=64,
        dim_mults=(2, 2, 4, 8),#dim_mults=(1, 2, 4, 8),
        context_dim=16,
        t_dim = 6,
    ):
        super().__init__()
        self.t_dim = t_dim
        
        transition_dim = transition_dim + t_dim

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2
            # if not is_last:
            #     horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            # nn.Mish(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_dim, fc_dim // 2),
            # nn.Mish(),
            nn.ReLU(),
            nn.Linear(fc_dim // 2, context_dim),
        )
        self.final_block_t = nn.Sequential(
            nn.Linear(context_dim, fc_dim // 2),
            # nn.Mish(),
            # nn.ReLU(),
            nn.Linear(fc_dim // 2, 1),
        )
        
        self.apply(weights_init_)

    def forward(self, x, t, *args):
        '''
            x : [ batch x horizon x transition ]
        '''
        t = einops.repeat(t, 'b h -> b h r',r=self.t_dim)
        x = torch.cat([x, t], -1)
        x = einops.rearrange(x, 'b h t -> b t h')
        
        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x)
            x = resnet2(x)
            x = downsample(x)

        ##
        x = self.mid_block1(x)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        x = self.final_block(x)
        x = x + torch.clamp(torch.randn_like(x), -2, 2) * 0.001 
        out = torch.sigmoid(x)
        t = self.final_block_t(x)
        t = torch.tanh(t) * 2. 
        return out, t 
