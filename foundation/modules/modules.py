import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from einops import rearrange
from enum import Enum, auto
from typing import overload, Callable
from functools import reduce
from . import flops, flags

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ContinuityEmbedding(nn.Module):
    def __init__(self, total, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(total).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [total, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [total, d_model // 2, 2]
        emb = emb.view(total, d_model)

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        return emb


class DiscretenessEmbedding(nn.Module):
    def __init__(self, total, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=total + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        emb = self.embedding(x)
        return emb


class FlattenSequential(nn.Sequential):
    def __init__(self, sequence, start_dim: int = 1, end_dim: int = -1):
        super().__init__(*sequence)
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, input):
        return super().forward(input.flatten(self.start_dim, self.end_dim))


class VectorEmbedding(nn.Module):
    def __init__(self, vector_length: int, d_model: int, dim: int | None, dropout = 0, input_dim: int = 1):
        super().__init__()
        increment = (d_model - input_dim) // 3
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim + increment),
            nn.ReLU(),
            nn.Linear(input_dim + increment, input_dim + increment * 2),
            nn.ReLU(),
            nn.Linear(input_dim + increment * 2, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.is_scalar = vector_length <= 1
        if not self.is_scalar:
            position_weight = torch.zeros(vector_length, d_model)
            position = torch.arange(0, vector_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *  # 1 * d_model
                                -(math.log(10000.0) / d_model))
            position_weight[:, 0::2] = torch.sin(position * div_term) # max_len * d_model
            position_weight[:, 1::2] = torch.cos(position * div_term)
            position_weight = position_weight.unsqueeze(0)
            self.register_buffer('position_weight', position_weight)
        
        if dim is None:
            self.final = nn.Identity()
        else:
            self.final = FlattenSequential(
                [nn.Linear(d_model * vector_length, dim),
                 Swish(),
                 nn.Linear(dim, dim)],
                1
            )
        
        self.initialize()
    
    def initialize(self):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)
        x = self.layers(x)
        if not self.is_scalar:
            x = x + self.position_weight[:, :x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return self.final(x)


def slog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


class RealEmbedding(VectorEmbedding):
    def __init__(self, vector_length: int, d_model: int, dim: int, dropout=0, input_dim: int = 1):
        super().__init__(vector_length, d_model, dim, dropout, input_dim)
    def forward(self, x: torch.Tensor):
        return super().forward(slog(x))


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.c1.weight)
        init.zeros_(self.c1.bias)
        init.xavier_uniform_(self.c2.weight)
        init.zeros_(self.c2.bias)

    def forward(self, x):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=1, padding=2)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.c1.weight)
        init.zeros_(self.c1.bias)

    def forward(self, x, size):
        x = F.interpolate(
            x, size=size, mode='nearest')
        x = self.c1(x) + self.c2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float | int):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.out_block = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        init.xavier_uniform_(self.out_block[-1].weight, gain=1e-5)
    
    def middle_block(self, x, *args):
        return x
    
    def final(self, x, *args):
        return x

    def forward(self, x, *args):
        h = self.in_block(x)
        h = self.middle_block(h, *args)
        h = self.out_block(h)

        h = h + self.shortcut(x)
        h = self.final(h, *args)
        return h


class DiffResBlock(ResBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 middle_dim,
                 dropout,
                 tailing_module: nn.Module | None = None,
                 is_tailing_module_prior: bool = True,
                 num_tailing_guidence_modules = 0,
                 num_guidence_modules = 0):
        super().__init__(in_channels, out_channels, dropout)
        self.t_emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(middle_dim, out_channels),
        )
        self.guidence_modules_projs = nn.ModuleList([
            nn.Sequential(
                Swish(),
                nn.Linear(middle_dim, out_channels),
            ) for _ in range(num_guidence_modules)
        ])
        self.num_tailing_guidence_modules = num_tailing_guidence_modules
        self.is_tailing_module_prior = is_tailing_module_prior
        self.tailing_guidence_modules_projs = nn.ModuleList([
            nn.Sequential(
                Swish(),
                nn.Linear(middle_dim, out_channels),
            ) for _ in range(num_tailing_guidence_modules)
        ])
        self.tailing_module = tailing_module
        self.initialize()
    
    def middle_block(self, h, time_embedded, *args):
        h += self.t_emb_proj(time_embedded)[:, :, None, None]
        guidence_args_start = len(args) - len(self.guidence_modules_projs)
        self.__tailing_args_length = guidence_args_start - self.num_tailing_guidence_modules
        for i, guidence_embedded in enumerate(args[guidence_args_start:]):
            h += self.guidence_modules_projs[i](guidence_embedded)[:, :, None, None]
        if (self.tailing_module is None) or (not self.is_tailing_module_prior):
            for i, tailing_guidence_modules_proj in enumerate(self.tailing_guidence_modules_projs):
                h += tailing_guidence_modules_proj(args[self.__tailing_args_length + i])[:, :, None, None]
        return h

    def final(self, x, time_embedded, *args):
        if self.tailing_module is None:
            return x
        else:
            return self.tailing_module(x, *args[:self.__tailing_args_length])
    
    @overload
    def forward(self, x, time_embedded, guidence_embedded, *args):
        ...
    
    @overload
    def forward(self, x, time_embedded, tailing, guidence_embedded, *args):
        ...
    
    @overload
    def forward(self, x, time_embedded, tailing, tailing_guidence_embedded, guidence_embedded, *args):
        ...
    
    def forward(self, x, t_emb, *args):
        return super().forward(x, t_emb, *args)


class ResBlockType(Enum):
    down = auto()
    middle = auto()
    up = auto()


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


# Param tags
def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


# Kernels
@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class Linear(nn.Linear):
    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


# Token merging and splitting
class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


class Identity(nn.Identity):
    def forward(self, input, *args, **kwargs):
        return super().forward(input)

def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# class LatentEncoder(nn.Module):
#     def __init__(self,
#                  original_channels: int,
#                  latent_channels: int,
#                  middle_channels: int = 128,
#                  channel_multipliers: Iterable[int] = [1, 2, 3, 4],
#                  num_res_blocks: int = 2,
#                  attention_levels: list[int] = [2],
#                  dropout: float | int = 0.15):
#         super().__init__()
#         self.__num_res_blocks = num_res_blocks

#         self.head = nn.Conv2d(original_channels, middle_channels, kernel_size=3, stride=1, padding=1)
#         self.down_blocks = nn.ModuleList()
#         now_ch = middle_channels
#         final_index = len(channel_multipliers) - 1
#         for i, mult in enumerate(channel_multipliers):
#             out_ch = middle_channels * mult
#             for _ in range(num_res_blocks):
#                 if i in attention_levels:
#                     self.down_blocks.append(
#                         nn.Sequential(
#                             ResBlock(in_ch=now_ch, out_ch=out_ch, dropout=dropout),
#                             AttnBlock(out_ch)
#                         )
#                     )
#                 else:
#                     self.down_blocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, dropout=dropout))
#                 now_ch = out_ch
#             if i != final_index:
#                 self.down_blocks.append(DownSample(now_ch))

#         self.middle_blocks = nn.Sequential(
#             ResBlock(in_ch=now_ch, out_ch=now_ch, dropout=dropout),
#             AttnBlock(now_ch),
#             ResBlock(in_ch=now_ch, out_ch=now_ch, dropout=dropout),
#         )
        
#         self.tail = nn.Sequential(
#             nn.GroupNorm(num_groups=32, num_channels=now_ch, eps=1e-6),
#             Swish(),
#             nn.Conv2d(now_ch, latent_channels, 3, stride=1, padding=1)
#         )
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.head.weight)
#         init.zeros_(self.head.bias)
#         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
#         init.zeros_(self.tail[-1].bias)

#     def forward(self, x):
#         # Downsampling
#         x = self.head(x)
#         judgement_divisor = self.__num_res_blocks + 1
#         for i, layer in enumerate(self.down_blocks):
#             if (i - self.__num_res_blocks) % judgement_divisor != 0:
#                 x = layer(x)
#             else:
#                 x = layer(x)
#         # Middle
#         x = self.middle_blocks(x)
#         x = self.tail(x)
#         return x