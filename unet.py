import torch as th
from torch import nn

from typing import Optional, Tuple, Union, List
import math
from einops import rearrange

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
    
class ResidualBlock(nn.Module):

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

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.proj = nn.Linear(dim, heads * dim_head * 3)

    def forward(self, x: th.Tensor, t: Optional[th.Tensor] = None):
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj(x)
        qkv = rearrange(x, 'b s (h d n) -> b s h (d n)', d = self.dim_head, n = 3)
        q, k, v = th.chunk(qkv, 3, dim=-1)

# block = ResidualBlock(64, 128, 64)
# x = th.randn(2, 64, 32, 32)
# emb = TimeEmbedding(64)
# t = th.ones(2)
# t = emb(t)
# out = block(x, t)
# block = AttentionBlock(64, 8, 32)
# out = block(x)
