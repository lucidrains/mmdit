from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    RMSNorm,
    FeedForward
)

from mmdit.mmdit_pytorch import (
    JointAttention
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# adaptive layernorm
# aim for clarity in generalized version

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)

            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            x = x * gamma + beta

        return x

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_modalities: Tuple[int, ...],
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        softclamp = False,
        softclamp_value = 50.,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)

            self.to_post_branch_gammas = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias, 1.)

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.joint_attn = JointAttention(
            dim = dim_joint_attn,
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn,
            softclamp = softclamp,
            softclamp_value = softclamp_value,
        )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):
        assert len(modality_tokens) == self.num_modalities
        assert not (exists(time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        ln_kwargs = dict()

        if self.has_cond:
            ln_kwargs = dict(cond = time_cond)

            gammas = self.to_post_branch_gammas(time_cond)
            attn_gammas, ff_gammas = gammas.chunk(2, dim = -1)

        # attention layernorms

        modality_tokens_residual = modality_tokens

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

        # attention

        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # post attention gammas

        if self.has_cond:
            attn_gammas = attn_gammas.split(self.dim_modalities, dim = -1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, attn_gammas)]

        # add attention residual

        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # handle feedforward adaptive layernorm

        modality_tokens_residual = modality_tokens

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas

        if self.has_cond:
            ff_gammas = ff_gammas.split(self.dim_modalities, dim = -1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, ff_gammas)]

        # add feedforward residual

        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # returns

        return modality_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities,
        final_norms = True,
        **block_kwargs
    ):
        super().__init__()
        blocks = [MMDiTBlock(dim_modalities = dim_modalities, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):
        for block in self.blocks:
            modality_tokens = block(
                time_cond = time_cond,
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        return tuple(modality_tokens)
