"""k-diffusion transformer diffusion models, version 2."""
from typing import Sequence, Optional, Callable, Dict, overload
from dataclasses import dataclass
from functools import reduce
from typing import Union
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch._dynamo
from utils.args_context import ArgsContext
from ... import flags
from . import layers
from ...attention.axial_rope import make_axial_pos
from ...modules import apply_wd, zero_init, RMSNorm, LinearGEGLU, Linear, TokenMerge, TokenSplit, TokenSplitWithoutSkip, Identity


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers
def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Rotary position embeddings
@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


# Mapping network
class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class HDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        concat_channels: Optional[int] = None,
        patch_size: int = [4, 4],
        widths: Sequence[int] = [128, 256],
        middle_width: int = 512,
        depths: Sequence[int] = [2, 2],
        middle_depth: int = 4,
        layer_block_builders: Dict[int, Callable[[ArgsContext], nn.Module]] = {},
        mapping_width: int = 256,
        mapping_depth: int = 2,
        mapping_dim_feed_forward: Optional[int] = None,
        mapping_dropout: float = 0.,
        is_layer_cond_module_prior: bool = True,
        layer_cond_init_modules: Union[nn.Module, Sequence[nn.Module], None] = None,
        layer_cond_guidence_modules: Union[nn.Module, Sequence[nn.Module], None] = None,
        guidence_modules: Union[nn.Module, Sequence[nn.Module], None] = None
    ):
        super().__init__()
        
        self.layer_block_builders = layer_block_builders
        
        if concat_channels is None:
            self.concat_num = 0
        else:
            self.concat_num = concat_channels

        self.patch_in = TokenMerge(in_channels + self.concat_num, widths[0], patch_size)

        self.time_embedding = layers.FourierFeatures(1, mapping_width)
        self.time_in_proj = Linear(mapping_width, mapping_width, bias=False)
        
        self.is_layer_cond_module_prior = is_layer_cond_module_prior
        if layer_cond_init_modules is None:
            layer_cond_init_modules = []
        elif not isinstance(layer_cond_init_modules, Sequence):
            layer_cond_init_modules = [layer_cond_init_modules]
        if layer_cond_guidence_modules is None:
            layer_cond_guidence_modules = []
        elif not isinstance(layer_cond_guidence_modules, Sequence):
            layer_cond_guidence_modules = [layer_cond_guidence_modules]
        if guidence_modules is None:
            guidence_modules = []
        elif not isinstance(guidence_modules, Sequence):
            guidence_modules = [guidence_modules]
        self.layer_cond_init_modules = nn.ModuleList(layer_cond_init_modules)
        self.layer_cond_guidence_modules = nn.ModuleList(layer_cond_guidence_modules)
        self.guidence_modules = nn.ModuleList(guidence_modules)
        
        self.guidence_modules_projs = nn.ModuleList([
            Linear(mapping_width, mapping_width, bias=False) for _ in range(len(self.guidence_modules))
        ])
        self.is_layer_cond_module_prior = is_layer_cond_module_prior
        self.layer_cond_guidence_modules_projs = nn.ModuleList([
            Linear(mapping_width, mapping_width, bias=False) for _ in range(len(self.layer_cond_guidence_modules))
        ])
        
        self.mapping = MappingNetwork(mapping_depth, mapping_width, mapping_dim_feed_forward, dropout=mapping_dropout)

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        num_half_levels = len(widths)
        self.num_total_levels = 2 * num_half_levels + 1
        for i, (width, depth) in enumerate(zip(widths, depths)):
            self.down_levels.append(Level([self.build_layer_block(width, j, i) for j in range(depth)]))
            self.up_levels.append(Level([self.build_layer_block(width, j + depth, self.num_total_levels - 1 - i) for j in range(depth)]))
        self.mid_level = Level([self.build_layer_block(middle_width, j, num_half_levels) for j in range(middle_depth)])
        
        no_middle_widths = widths
        no_first_widths = np.concatenate((widths[1:],[middle_width]))

        self.merges = nn.ModuleList([TokenMerge(width_1, width_2) for width_1, width_2 in zip(no_middle_widths, no_first_widths)])
        self.splits = nn.ModuleList([TokenSplit(width_2, width_1) for width_1, width_2 in zip(no_middle_widths, no_first_widths)])

        self.out_norm = RMSNorm(widths[0])
        self.patch_out = TokenSplitWithoutSkip(widths[0], out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)
        
        del self.layer_block_builders
    
    def build_layer_block(self, width: int, layer_index: int, level_index: int):
        args_context = ArgsContext(
            width=width,
            layer_index=layer_index,
            level_index=level_index
        )
        if level_index in self.layer_block_builders.keys():
            return self.layer_block_builders[level_index](args_context)
        else:
            return Identity()

    def layer_block_parameters(self, time, *args):
        layer_cond_args_length = len(args) - len(self.guidence_modules)
        # Timestep embedding
        # time_embedded = self.time_embedding(time)
        # c_noise = torch.log(time) / 4
        time_embedded = self.time_embedding(time[..., None])
        block_mapping = self.time_in_proj(time_embedded)
        
        guidence_embedded = [guidence_module(args[layer_cond_args_length + i]) for i, guidence_module in enumerate(self.guidence_modules)]
        for i, guidence_emb in enumerate(guidence_embedded):
            block_mapping += self.guidence_modules_projs[i](guidence_emb)
        
        layer_cond_args = list(args[:layer_cond_args_length])
        layer_cond_initialized = [init_module(layer_cond_args[i]) for i, init_module in enumerate(self.layer_cond_init_modules)]
        layer_cond_initialized += layer_cond_args[len(layer_cond_initialized):]
        layer_cond_guidence_embedded = [layer_cond_guidence_module(layer_cond_initialized[i]) for i, layer_cond_guidence_module in enumerate(self.layer_cond_guidence_modules)]
        if not self.is_layer_cond_module_prior:
            for i, layer_cond_guidence_modules_proj in enumerate(self.layer_cond_guidence_modules_projs):
                block_mapping += layer_cond_guidence_modules_proj(layer_cond_guidence_embedded[i])
        
        return block_mapping, layer_cond_initialized
    
    def add_residual(self, h, residual: Optional[torch.Tensor]):
        if residual is None:
            return h
        else:
            return h + residual.to(h.dtype)
    
    def prepare_additional_residuals(self, additional_residuals: Optional[Sequence[torch.Tensor]]):
        all_levels_length = self.num_total_levels
        additional_residuals_list = additional_residuals
        additional_residuals = np.empty(all_levels_length, dtype=object)
        if additional_residuals_list is not None:
            additional_residuals_list = np.fromiter(additional_residuals_list, dtype=object)
            additional_residuals_length = len(additional_residuals_list)
            if additional_residuals_length < all_levels_length:
                additional_residuals[:additional_residuals_length] = additional_residuals_list
            else:
                additional_residuals[:] = additional_residuals_list[:additional_residuals_length]
        return additional_residuals
    
    def split_additional_residuals(self, additional_residuals):
        num_half_levels = len(self.down_levels)
        return additional_residuals[:num_half_levels], additional_residuals[num_half_levels:-num_half_levels], additional_residuals[-num_half_levels:]
    
    def split_skip_residuals(self, skip_residuals):
        num_half_levels = len(self.down_levels)
        if skip_residuals is Sequence:
            if len(skip_residuals) > num_half_levels:
                return skip_residuals[:-num_half_levels], skip_residuals[-num_half_levels:]
            else:
                return None, skip_residuals
        else:
            return None, None
    
    def pop_first_residual(self, residuals: np.ndarray):
        first_item = residuals[0]
        if len(residuals) > 1:
            other_items = residuals[1:]
        else:
            other_items = residuals[0:0]
        return first_item, other_items
    
    def concat_channels(self, x: torch.Tensor, *args):
        if self.concat_num > 0:
            x = torch.cat((x,) + args[:self.concat_num], dim=1)
            args = args[self.concat_num:]
        return x, args
    
    def patch_positionally(self, x: torch.Tensor):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)
        return x, pos
    
    def make_block_mapping(self, time: torch.Tensor, *args):
        # Mapping Network
        block_mapping, layer_cond = self.layer_block_parameters(time, *args)
        block_mapping = self.mapping(block_mapping)
        return block_mapping, layer_cond
    
    def down_sample(self, x, additional_residuals, pos, block_mapping, layer_cond):
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            first_item, additional_residuals = self.pop_first_residual(additional_residuals)
            x = self.add_residual(down_level(x, pos, block_mapping, *layer_cond), first_item)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)
        return x, pos, skips, poses
    
    def middle_sample(self, x, additional_residuals, pos, block_mapping, layer_cond):
        first_item, additional_residuals = self.pop_first_residual(additional_residuals)
        x = self.add_residual(self.mid_level(x, pos, block_mapping, *layer_cond), first_item)
        return x
    
    def middle_skip_connect(self, x, skip_residuals):
        if skip_residuals is Sequence:
            skip_first_item, _ = self.pop_first_residual(skip_residuals)
            x = self.add_residual(x, skip_first_item)
        return x
    
    def up_sample(self, x, additional_residuals, skips, poses, block_mapping, layer_cond, *, skip_residuals):
        if skip_residuals is Sequence:
            def split_skip(split, x, skip):
                skip_first_item, skip_residuals = self.pop_first_residual(skip_residuals)
                return split(x, skip + skip_first_item)
        else:
            def split_skip(split, x, skip):
                return split(x, skip)
                
        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            first_item, additional_residuals = self.pop_first_residual(additional_residuals)
            x = split_skip(split, x, skip)
            x = self.add_residual(up_level(x, pos, block_mapping, *layer_cond), first_item)
        return x
    
    def unpatch(self, x: torch.Tensor):
        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)
        return x

    @overload
    def forward(self, x, t, guidence, *args, additional_residuals: Optional[Sequence[torch.Tensor]] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        ...
    
    @overload
    def forward(self, x, t, layer_cond, guidence, *args, additional_residuals: Optional[Sequence[torch.Tensor]] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        ...

    @overload
    def forward(self, x, t, concatenation, guidence, *args, additional_residuals: Optional[Sequence[torch.Tensor]] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        ...
    
    @overload
    def forward(self, x, t, concatenation, layer_cond, guidence, *args, additional_residuals: Optional[Sequence[torch.Tensor]] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        ...

    def forward(self, x: torch.Tensor, timestep, *args, additional_residuals: Optional[Sequence] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        additional_residuals = self.prepare_additional_residuals(additional_residuals)
        down_additional_residuals, middle_additional_residuals, up_additional_residuals = self.split_additional_residuals(additional_residuals)
        middle_skip_residuals, up_skip_residuals = self.split_skip_residuals(skip_residuals)
        
        x, args = self.concat_channels(x, *args)
        x, pos = self.patch_positionally(x)
        
        block_mapping, layer_cond = self.make_block_mapping(timestep, *args)

        # Hourglass Transformer
        x, pos, skips, poses = self.down_sample(x, down_additional_residuals, pos, block_mapping, layer_cond)
        x = self.middle_sample(x, middle_additional_residuals, pos, block_mapping, layer_cond)
        x = self.middle_skip_connect(x, middle_skip_residuals)
        x = self.up_sample(x, up_additional_residuals, skips, poses, block_mapping, layer_cond, skip_residuals=up_skip_residuals)

        x = self.unpatch(x)
        return x
