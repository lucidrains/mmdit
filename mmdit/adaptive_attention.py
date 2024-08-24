from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, reduce, rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def softclamp(t, value):
    return (t / value).tanh() * value

Linear = partial(nn.Linear, bias = False)

# class

class AdaptiveAttention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        num_adaptive_weights = 1, # 1 becomes regular self attention with no gating
        softclamp = False,
        softclamp_value = 50.,
    ):
        """
        this idea was inspired by adaptive convs from gigagan https://arxiv.org/abs/2303.05511

        ein notation:
        b - batch
        n - sequence
        h - heads
        d - feature dimension
        w - adaptive weight
        """

        super().__init__()
        assert num_adaptive_weights >= 1

        has_gating = num_adaptive_weights > 1
        self.has_gating = has_gating
        self.num_adaptive_weights = num_adaptive_weights

        dim_inner = dim_head * heads
        scale = dim_head ** -0.5

        self.scale = scale
        self.softclamp = softclamp
        self.softclamp_value = softclamp_value

        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * num_adaptive_weights * 3),
            Rearrange('b n (qkv h d w) -> qkv b h n d w', qkv = 3, h = heads, w = num_adaptive_weights)
        )

        if has_gating:
            self.to_gates = nn.Sequential(
                Linear(dim, num_adaptive_weights * heads),
                Rearrange('b n (h w) -> b h n 1 w', w = num_adaptive_weights),
                nn.Softmax(dim = -1)
            )

        self.to_out_weights = nn.Parameter(torch.randn(heads, dim_head, dim * num_adaptive_weights))

    def forward(
        self,
        x,
        mask = None
    ):
        has_gating = self.has_gating

        qkv = self.to_qkv(x)

        # token dependent choosing of which weight

        if has_gating:
            gates = self.to_gates(x)
            qkv = reduce(qkv * gates, '... w -> ...', 'sum')
        else:
            qkv = rearrange(qkv, '... 1 -> ...')

        # usual self attention logic

        q, k, v = qkv

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if self.softclamp:
            sim = softclamp(sim, self.softclamp_value)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # again, adaptive weight on the outward projection
        # with gates from above

        out = einsum(out, self.to_out_weights, 'b h n d, h d e -> b h n e')

        if has_gating:
            out = rearrange(out, '... (d w) -> ... d w', w = self.num_adaptive_weights)
            out = reduce(out * gates, '... w -> ...', 'sum')
        else:
            out = rearrange(out, 'b h n d -> b n (h d)')

        return out

# example

if __name__ == '__main__':
    adaptive_attn = AdaptiveAttention(
        dim = 512,
        num_adaptive_weights = 4
    )

    text_tokens = torch.randn(1, 256, 512)
    image_tokens = torch.randn(1, 1024, 512)
    audio_tokens = torch.randn(1, 128, 512)

    tokens = torch.cat((text_tokens, image_tokens, audio_tokens), dim = -2)

    out = adaptive_attn(tokens)
