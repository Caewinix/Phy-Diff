#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
import natten
try:
    from natten.functional import na2d_qk_with_bias
except: pass
from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from .. import flops
from .basic_core import scale_for_cosine_sim, checkpoint
from ..modules import AdaRMSNorm, Linear, AxialRoPE, FeedForwardBlock, apply_wd, zero_init, apply_rotary_emb_


def _pad_if_small(x, window_size, height, width):
    x = x.permute(0, 2, 3, 1)
    # Pad if the input is small than the minimum supported size
    padded_height, padded_width = height, width
    padding_h = padding_w = 0
    if height < window_size or width < window_size:
        padding_h = max(0, window_size - padded_height)
        padding_w = max(0, window_size - padded_width)
        x = pad(x, (0, 0, 0, padding_w, 0, padding_h))
        _, padded_height, padded_width, _ = x.shape
    return x.permute(0, 3, 1, 2), padded_height, padded_width


class NeighborhoodConvAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim_channels: int,
        dim_head: int | None = None,
        num_heads: int | None = None,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_groups: int | None = 32
    ):
        super().__init__()
        self.group_norm = nn.Identity() if num_groups is None or num_groups == 0 else nn.GroupNorm(num_groups, dim_channels)
        
        if num_heads is None:
            num_heads = 1
            if dim_head is None:
                dim_head = dim_channels
        else:
            if dim_head is None:
                dim_head = dim_channels // num_heads
        dim_attn = dim_head * num_heads
        
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = qk_scale or self.dim_head ** -0.5
        
        self.proj_q = nn.Conv2d(dim_channels, dim_attn, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(dim_channels, dim_attn, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(dim_channels, dim_attn, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(dim_attn, dim_channels, 1, stride=1, padding=0)
        
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.group_norm(x)
        
        batch_size, _, height, width= x.shape
        x, padded_height, padded_width = _pad_if_small(x, self.window_size, height, width)
        
        q = self.proj_q(x).view(batch_size, self.num_heads, self.dim_head, padded_height, padded_width).permute(0, 1, 3, 4, 2)
        k = self.proj_k(x).view(batch_size, self.num_heads, self.dim_head, padded_height, padded_width).permute(0, 1, 3, 4, 2)
        v = self.proj_v(x).view(batch_size, self.num_heads, self.dim_head, padded_height, padded_width).permute(0, 1, 3, 4, 2)
        
        q = q * self.scale
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.reshape(batch_size, padded_height, padded_width, -1)

        # Remove padding, if added any
        if padded_height or padded_width:
            x = x[:, :height, :width, :]
        x = x.permute(0, 3, 1, 2)
        
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
        qk = natten.functional.na2d_qk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1).to(v.dtype)
        x = natten.functional.na2d_av(a, v, self.kernel_size, 1)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x