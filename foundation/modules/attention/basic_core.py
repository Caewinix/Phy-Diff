import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from typing import Optional
from functools import reduce
from .. import flags

class SingleHeadAttentionNormal(nn.Module):
    def __init__(self, dim_channels):
        super().__init__()
        self.dim_channels = dim_channels
        self.scale = dim_channels ** (-0.5)

    def forward(self, x_shape, q, k, v):
        batch_size, _, height, width = x_shape
        q = q.permute(0, 2, 3, 1).view(batch_size, -1, self.dim_channels)
        k = k.view(batch_size, self.dim_channels, -1)
        w = torch.bmm(q, k) * self.scale
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(batch_size, -1, self.dim_channels)
        h = torch.bmm(w, v)
        assert list(h.shape) == [batch_size, height * width, self.dim_channels]
        h = h.view(batch_size, height, width, self.dim_channels).permute(0, 3, 1, 2)
        
        return h


class MultiHeadAttentionNormal(nn.Module):
    def __init__(self, num_heads, dim_head):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** (-0.5)

    def forward(self, x_shape, q, k, v):
        batch_size, _, height, width = x_shape
        q = q.permute(0, 2, 3, 1).view(batch_size, -1, self.num_heads, self.dim_head)
        k = k.permute(0, 2, 3, 1).view(batch_size, -1, self.num_heads, self.dim_head)
        v = v.permute(0, 2, 3, 1).view(batch_size, -1, self.num_heads, self.dim_head)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        
        out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, height, width)
        return out

# class FlashAttention(nn.Module):
#     def __init__(self, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
        
#     def forward(self, x_shape, q, k, v):
#         batch_size, _, height, width = x_shape
#         seq_length = height * width
#         q = q.permute(0, 2, 3, 1).view(batch_size, seq_length, self.num_heads, -1).contiguous()
#         k = k.permute(0, 2, 3, 1).view(batch_size, seq_length, self.num_heads, -1).contiguous()
#         v = v.permute(0, 2, 3, 1).view(batch_size, seq_length, self.num_heads, -1).contiguous()
#         h = F.scaled_dot_product_attention(q, k, v)
#         h = h.permute(0, 1, 3, 2).view(batch_size, -1, height, width)
#         return h

class AttnBlock(nn.Module):
    def __init__(
        self,
        dim_channels: int,
        dim_head: int | None = None,
        num_heads: int | None = None,
        dim_cond: int | None = None,
        num_groups: int | None = 32
    ):
        super().__init__()
        self.group_norm = nn.Identity() if num_groups is None or num_groups == 0 else nn.GroupNorm(num_groups, dim_channels)
        
        if dim_cond is None:
            dim_cond = dim_channels
        
        if num_heads is None:
            num_heads = 1
            if dim_head is None:
                dim_head = dim_channels
        else:
            if dim_head is None:
                dim_head = dim_channels // num_heads
        dim_attn = dim_head * num_heads
        
        self.proj_q = nn.Conv2d(dim_channels, dim_attn, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(dim_cond, dim_attn, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(dim_cond, dim_attn, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(dim_attn, dim_channels, 1, stride=1, padding=0)
        
        self.attn_mechanism = (#FlashAttention(num_heads) if parse_version(torch.__version__) >= parse_version("2.0") else
                               (MultiHeadAttentionNormal(num_heads, dim_head)
                                     if num_heads > 1
                                     else SingleHeadAttentionNormal(dim_channels)))
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        h = self.group_norm(x)
        if cond is None:
            cond = h
        q = self.proj_q(h)
        k = self.proj_k(cond)
        v = self.proj_v(cond)
        h = self.attn_mechanism(torch.as_tensor(x.shape), q, k, v)
        h = self.proj(h)
        return x + h


@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)