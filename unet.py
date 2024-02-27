import torch as th
from torch import nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Optional, Tuple, Union, List
import math
from einops import rearrange

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class TimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
    
    def forward(self, t: th.Tensor):
        '''
        t: 1D Tensor
        '''
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(-emb * th.arange(0, half_dim, device=device))
        emb = t[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ResidualBlock(TimestepBlock):

    def __init__(self, dim_in: int, dim_out: int, dim_time: int, 
                    n_groups: int = 32, dropout: float = 0.1):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(n_groups, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_time, dim_out),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), padding=(1, 1)),
        )

        if dim_in == dim_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=(1, 1))
    
    def forward(self, x: th.Tensor, emb: th.Tensor):
        '''
        x: [b, c, h, w]
        emb: [b, c']
        '''
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]
        # broadcast
        h += emb_out
        h = self.out_layers(h)
        return h + self.shortcut(x)
    
class AttentionBlock(nn.Module):

    def __init__(self, dim: int, heads: int = 1, dim_head: int = None, n_groups: int = 32):
        super().__init__()

        if dim_head is None:
            dim_head = dim

        self.heads = heads
        self.dim_head = dim_head
        self.proj = nn.Linear(dim, heads * dim_head * 3)
        self.scale = dim_head ** -0.5
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x: th.Tensor):
        width = x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        qkv = self.proj(x)
        qkv = rearrange(qkv, 'b s (h d) -> b s h d', h = self.heads)
        q, k, v = th.chunk(qkv, 3, dim=-1)
        weight = th.einsum('b s h d, b t h d -> b s t h', q, k) * self.scale
        weight = weight.softmax(dim=2)
        res = th.einsum('b s t h, b t h d -> b s h d', weight, v)
        res = rearrange(res, 'b s h d -> b s (h d)')
        h = self.proj_out(res) + x
        return rearrange(h, 'b (h w) c -> b c h w', w = width)

class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, (3,3), (2,2), (1,1))

    def forward(self, x: th.Tensor):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, kernel_size=(3,3), padding=(1,1))

    def forward(self, x: th.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)    

class UNet(nn.Module):

    def __init__(self, channels: int = 3, dim: int = 64,
                 dim_mult = (1, 2, 4, 8), is_attn = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()

        time_embed_dim = dim * 4
        # PosEmb + MLP
        self.time_embed = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.down_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(channels, dim, kernel_size=(3,3), padding=(1,1)),
                ),
            ]
        )

        self.up_blocks = nn.ModuleList([])

        ch = dim
        down_block_dims = []
        # (Res + Attn) * n + (Down)
        for level, mult in enumerate(dim_mult):
            for _ in range(n_blocks):
                down_block_dims.append(ch)
                layers = [
                    ResidualBlock(ch, mult * dim, time_embed_dim),
                ]
                ch = mult * dim
                if is_attn[level]:
                    layers.append(AttentionBlock(ch))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
            if level != len(dim_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(
                    Downsample(ch)
                ))
            down_block_dims.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim),
        )

        # ((Res + Attn) * n + Up)
        for level, mult in list(enumerate(dim_mult))[::-1]:
            for i in range(n_blocks + 1):
                down_dim = down_block_dims.pop()
                layers = [
                    ResidualBlock(ch + down_dim, down_dim, time_embed_dim)
                ]
                ch = down_dim
                if is_attn[level]:
                    layers.append(AttentionBlock(ch))
                if level and i == n_blocks:
                    layers.append(Upsample(ch))
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, channels, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x: th.Tensor, t: th.Tensor):
        t = self.time_embed(t)
        h = x
        hs = []
        for module in self.down_blocks:
            h = module(h, t)
            hs.append(h)

        h = self.middle_block(h, t)

        for module in self.up_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t)

        return self.out(h) 




# block = ResidualBlock(64, 128, 64)
# x = th.randn(2, 3, 32, 32)
# emb = TimeEmbedding(64)
# t = th.ones(2)
# t = emb(t)
# out = block(x, t)
# block = AttentionBlock(64, 8, 32)
# out = block(x)
# model = UNet(is_attn=(False, False, False, False))
# out = model(x, t)
# print(out.shape)
