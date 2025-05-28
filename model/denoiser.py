import torch as t
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial





class UnetMLP(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 1), resnet_block_groups=8, time_dim=128, nb_mod=1):
        super().__init__()
        self.nb_mod = nb_mod
        init_dim = init_dim or dim
        self.init_lin = nn.Linear(dim, init_dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_mod, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                nn.Linear(dim_in, dim_out) if is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                nn.Linear(dim_out, dim_in) if not is_last else nn.Linear(dim_out, dim_in)
            ]))

        self.out_dim = out_dim or dim
        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_lin = nn.Sequential(
            nn.GroupNorm(resnet_block_groups, init_dim),
            nn.SiLU(),
            nn.Linear(init_dim, self.out_dim)
        )

    def forward(self, x, t, std=None):
        t = t.reshape((t.size(0), self.nb_mod))
        x = self.init_lin(x)
        r = x.clone()
        t = self.time_mlp(t).squeeze()
        h = []

        for block1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_lin(x) / std if std is not None else self.final_lin(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups, shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups, shift_scale=shift_scale)
        self.lin_layer = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, t=scale_shift)
        h = self.block2(h)
        return h + self.lin_layer(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, dim)
        self.shift_scale = shift_scale

    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if t is not None:
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t
        return x
    

class Denoiser(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, n_layers=3, emb_size=64):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim   
        input_dim = x_dim + y_dim
        hidden_dim = 64 if input_dim <= 10 else 128 if input_dim <= 50 else 256
        self.unet = UnetMLP(dim=input_dim, 
                            init_dim=hidden_dim, 
                            dim_mults=[], 
                            time_dim=hidden_dim, 
                            nb_mod=1,
                            out_dim=x_dim)
            
    def forward(self, x, logsnr, y=None):
        if y is None:
            y = t.zeros(x.shape[0], self.y_dim, device=x.device)
        input_tensor = t.cat([x.flatten(1), y.flatten(1)], dim=1)
        return self.unet(input_tensor, logsnr)