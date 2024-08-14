from typing import Sequence, Optional, Union, Dict, Callable
import einops
import torch
import torch as th
import torch.nn as nn
from .hdit import HDiT, downscale_pos
from ...modules import zero_module, TokenMerge
from utils.args_context import ArgsContext
from einops import rearrange, repeat
from torchvision.utils import make_grid

class ControlledHDiTModel(HDiT):
    def forward(self, x: torch.Tensor, timestep, *args, additional_residuals: Optional[Sequence] = None, skip_residuals: Optional[Sequence[torch.Tensor]] = None):
        additional_residuals = self.prepare_additional_residuals(additional_residuals)
        down_additional_residuals, middle_additional_residuals, up_additional_residuals = self.split_additional_residuals(additional_residuals)
        middle_skip_residuals, up_skip_residuals = self.split_skip_residuals(skip_residuals)
        
        x, args = self.concat_channels(x, *args)
        with torch.no_grad():
            x, pos = self.patch_positionally(x)
            block_mapping, layer_cond = self.make_block_mapping(timestep, *args)

            # Hourglass Transformer
            x, pos, skips, poses = self.down_sample(x, down_additional_residuals, pos, block_mapping, layer_cond)
            x = self.middle_sample(x, middle_additional_residuals, pos, block_mapping, layer_cond)
        x = self.middle_skip_connect(x, middle_skip_residuals)
        x = self.up_sample(x, up_additional_residuals, skips, poses, block_mapping, layer_cond, skip_residuals=up_skip_residuals)

        x = self.unpatch(x)
        return x


class ZeroConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
    )-> None:
        super().__init__()
        self.zero_conv = zero_module(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    
    def forward(self, x: torch.Tensor):
        x = x.movedim(-1, -3)
        x = self.zero_conv(x)
        x = x.movedim(-3, -1)
        return x


class ControlHDiT(HDiT):
    def __init__(
        self,
        in_channels: int,
        hint_channels: int,
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
        super().__init__(
            in_channels,
            1,
            concat_channels,
            patch_size,
            widths,
            middle_width,
            depths,
            middle_depth,
            layer_block_builders,
            mapping_width,
            mapping_depth,
            mapping_dim_feed_forward,
            mapping_dropout,
            is_layer_cond_module_prior,
            layer_cond_init_modules,
            layer_cond_guidence_modules,
            guidence_modules
        )
        del self.up_levels, self.splits, self.out_norm, self.patch_out
        
        self.input_hint_patch_in = TokenMerge(hint_channels, 256, patch_size)
        self.input_hint_zero_conv = ZeroConv(256, widths[0], 3, padding=1)
        
        self.zero_convs = nn.ModuleList([self.make_zero_conv(widths[0])])
        for width in widths:
            self.zero_convs.append(self.make_zero_conv(width))
        
        self.middle_zero_conv = self.make_zero_conv(middle_width)
    
    def hint_patch(self, hint: torch.Tensor):
        hint = hint.movedim(-3, -1)
        hint = self.input_hint_patch_in(hint)
        hint = self.input_hint_zero_conv(hint)
        return hint
    
    def make_zero_conv(self, channels):
        return ZeroConv(channels, channels, 1, padding=0)
    
    def forward(self, x: torch.Tensor, timestep, hint, *args, additional_residuals: Optional[Sequence] = None):
        additional_residuals = self.prepare_additional_residuals(additional_residuals)
        
        x, args = self.concat_channels(x, *args)
        x, pos = self.patch_positionally(x)
        
        block_mapping, layer_cond = self.make_block_mapping(timestep, *args)
        
        guided_hint = self.hint_patch(hint)
        
        outs = []
        for down_level, merge, zero_conv in zip(self.down_levels, self.merges, self.zero_convs):
            first_item, additional_residuals = self.pop_first_residual(additional_residuals)
            x = self.add_residual(down_level(x, pos, block_mapping, *layer_cond), first_item)
            x = merge(x)
            pos = downscale_pos(pos)
            if guided_hint is not None:
                x = x + guided_hint
                guided_hint = None
            outs.append(zero_conv(x))
        
        x = self.middle_sample(x, None, pos, block_mapping, layer_cond)
        outs.append(self.middle_zero_conv(x))

        return outs


class ControlDiffusionModel(nn.Module):
    def __init__(self, control_model: nn.Module, controlled_backbone: nn.Module):
        super().__init__()
        self.control_model = control_model
        self.controlled_backbone = controlled_backbone
    
    def apply_model(self, x_noisy, t, control: Union[Sequence, torch.Tensor, None], *args, additional_residuals: Optional[Sequence] = None):
        if control is Sequence:
            control = torch.cat(control, 1)
        controlled = self.control_model(x_noisy, t, control, *args, additional_residuals=additional_residuals)
        eps = self.controlled_backbone(x_noisy, t, *args, additional_residuals=additional_residuals, skip_residuals=controlled)
        return eps
    
    def forward(self, x_noisy, t, control: Union[Sequence, torch.Tensor, None], *args, additional_residuals: Optional[Sequence] = None):
        return self.apply_model(x_noisy, t, control, *args, additional_residuals=additional_residuals)