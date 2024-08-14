import torch
from torch import nn
from torch.nn import init
import numpy as np
from typing import Sequence, Callable, Optional, Union, Dict
from collections import OrderedDict
from ...attention import AttnBlock
from ...modules import *
from utils.args_context import ArgsContext


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        middle_channels: int = 128,
        channel_multipliers: Sequence[int] = [1, 2, 3, 4],
        num_res_blocks: int = 2
    ):
        super().__init__()
        self.__num_res_blocks = num_res_blocks
        self.head = nn.Conv2d(input_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        
        self.__half_blocks_length = len(channel_multipliers)
        self.__middle_blocks_length = 2
        
        self.downblocks = nn.ModuleList()
        chs = [middle_channels] # record output channel when dowmsample for upsample
        now_ch = middle_channels
        final_index = self.__half_blocks_length - 1
        for i, mult in enumerate(channel_multipliers):
            out_ch = middle_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(self.build_res_block(now_ch, out_ch, ResBlockType.down, i))
                now_ch = out_ch
                chs.append(now_ch)
            if i != final_index:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            self.build_res_block(now_ch, now_ch, ResBlockType.middle, i) for i in range(self.__middle_blocks_length)
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = middle_channels * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(self.build_res_block(chs.pop() + now_ch, out_ch, ResBlockType.up, i))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, output_channels, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
    
    def build_res_block(self, in_ch: int, out_ch: int, res_type: ResBlockType, level_index: int):
        return ResBlock(in_ch=in_ch, out_ch=out_ch, dropout=0.15)
    
    def res_block_parameters(self, *args):
        return tuple()
    
    def down_sample(self, hs: list, x, *res_block_param, additional_residuals: np.ndarray):
        judgement_divisor = self.__num_res_blocks + 1
        
        h = self.head(x)
        hs.append((h, x.shape[-2:]))
        for i, layer in enumerate(self.downblocks):
            last_shape = h.shape[-2:]
            if (i - self.__num_res_blocks) % judgement_divisor != 0:
                first_item, additional_residuals = self.pop_first_residual(additional_residuals)
                h = self.add_residual(layer(h, *res_block_param), first_item)
            else:
                h = layer(h)
            hs.append((h, last_shape))
        return h
    
    def middle_sample(self, h, *res_block_param, additional_residuals: np.ndarray):
        for layer in self.middleblocks:
            first_item, additional_residuals = self.pop_first_residual(additional_residuals)
            h = self.add_residual(layer(h, *res_block_param), first_item)
        return h
    
    def up_sample(self, hs: list, h, *res_block_param, additional_residuals: np.ndarray):
        judgement_difference = self.__num_res_blocks + 1
        judgement_divisor = judgement_difference + 1
        
        for i, layer in enumerate(self.upblocks):
            if (i - judgement_difference) % judgement_divisor != 0:
                first_item, additional_residuals = self.pop_first_residual(additional_residuals)
                top_hs = hs.pop()
                h = torch.cat([h, top_hs[0]], dim=1)
                h = self.add_residual(layer(h, *res_block_param), first_item)
            else:
                h = layer(h, top_hs[1])
        return h
    
    def add_residual(self, h, residual: Optional[torch.Tensor]):
        if residual is None:
            return h
        else:
            return h + residual
    
    def prepare_additional_residuals(self, additional_residuals: Optional[Sequence[torch.Tensor]]):
        all_levels_length = self.__half_blocks_length * 2 + self.__middle_blocks_length
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
    
    def pop_first_residual(self, residuals: np.ndarray):
        first_item = residuals[0]
        if len(residuals) > 1:
            other_items = residuals[1:]
        else:
            other_items = residuals[0:0]
        return first_item, other_items
    
    def forward(self, x, *args, additional_residuals: Optional[Sequence[torch.Tensor]] = None):
        additional_residuals = self.prepare_additional_residuals(additional_residuals)
        
        res_block_param = self.res_block_parameters(*args)
        
        hs = []
        h = self.down_sample(hs, x, *res_block_param, additional_residuals=additional_residuals[:self.__half_blocks_length])
        h = self.middle_sample(h, *res_block_param, additional_residuals=additional_residuals[self.__half_blocks_length:-self.__half_blocks_length])
        h = self.up_sample(hs, h, *res_block_param, additional_residuals=additional_residuals[-self.__half_blocks_length:])
        h = self.tail(h)

        assert len(hs) == 0
        return h


class DiffUNet(UNet):
    def __init__(
        self,
        total_time: int,
        original_channels: int,
        middle_channels: int = 128,
        concat_channels: Optional[int] = None,
        channel_multipliers: Sequence[int] = [1, 2, 3, 4],
        num_res_blocks: int = 2,
        dropout: Union[float, int] = 0.15,
        res_tailing_builders: Dict[int, Callable[[ArgsContext[DiffResBlock]], nn.Module]] | None = {
            2: lambda context: AttnBlock(context.out_channels)
        },
        is_tailing_module_prior: bool = True,
        tailing_init_modules: Union[nn.Module, Sequence[nn.Module], None] = None,
        tailing_guidence_modules: Union[nn.Module, Sequence[nn.Module], None] = None,
        guidence_modules: Union[nn.Module, Sequence[nn.Module], None] = None
    ):
        # assert all([i < len(channel_multipliers) + 2 for i in res_tailing_builders.keys()]), '`res_tailing_layers` index out of bound.'
        if all([i < len(channel_multipliers) for i in res_tailing_builders.keys()]):
            self.down_res_tailing_builders = res_tailing_builders
            self.middle_res_tailing_builders = {}
            self.up_res_tailing_builders = res_tailing_builders
        else:
            len_half_blocks = len(channel_multipliers)
            len_middle_blocks = 2
            len_total_blocks = len_half_blocks * 2 + len_middle_blocks
            self.down_res_tailing_builders = OrderedDict()
            self.middle_res_tailing_builders = OrderedDict()
            self.up_res_tailing_builders = OrderedDict()
            for layer_level, tailing_builder in res_tailing_builders.items():
                if layer_level < len_half_blocks:
                    self.down_res_tailing_builders.setdefault(layer_level, tailing_builder)
                elif layer_level >= len_half_blocks + 2:
                    self.up_res_tailing_builders.setdefault(len_total_blocks - layer_level, tailing_builder)
                else:
                    self.middle_res_tailing_builders.setdefault(layer_level - len_half_blocks, tailing_builder)
        self.num_half_levels = len(channel_multipliers)
        self.res_tailing_builders = res_tailing_builders
        self.tdim = middle_channels * 4
        self.dropout = dropout
        self.is_tailing_module_prior = is_tailing_module_prior
        if tailing_init_modules is None:
            tailing_init_modules = []
        elif not isinstance(tailing_init_modules, Sequence):
            tailing_init_modules = [tailing_init_modules]
        if guidence_modules is None:
            guidence_modules = []
        elif not isinstance(guidence_modules, Sequence):
            guidence_modules = [guidence_modules]
        self.num_guidence_modules = len(guidence_modules)
        if tailing_guidence_modules is None:
            tailing_guidence_modules = []
        elif not isinstance(tailing_guidence_modules, Sequence):
            tailing_guidence_modules = [tailing_guidence_modules]
        self.num_tailing_guidence_modules = len(tailing_guidence_modules)
        
        if concat_channels is None:
            self.concat_num = 0
        else:
            self.concat_num = concat_channels
        
        super().__init__(
                original_channels + self.concat_num,
                original_channels,
                middle_channels,
                channel_multipliers,
                num_res_blocks
            )
        self.time_embedding = ContinuityEmbedding(total_time, middle_channels, self.tdim)
        self.tailing_init_modules = nn.ModuleList(tailing_init_modules)
        self.tailing_guidence_modules = nn.ModuleList(tailing_guidence_modules)
        self.guidence_modules = nn.ModuleList(guidence_modules)
        
        del self.num_half_levels
        del self.res_tailing_builders
        del self.tdim
        del self.dropout
        del self.num_guidence_modules
        del self.is_tailing_module_prior
        del self.num_tailing_guidence_modules
        del self.down_res_tailing_builders
        del self.middle_res_tailing_builders
        del self.up_res_tailing_builders
    
    def build_res_block(self, in_ch: int, out_ch: int, res_type: ResBlockType, level_index: int):
        res_args_context = ArgsContext[DiffResBlock, 'tailing_module'](
            in_channels=in_ch,
            out_channels=out_ch,
            middle_dim=self.tdim,
            dropout=self.dropout,
            is_tailing_module_prior=self.is_tailing_module_prior,
            num_tailing_guidence_modules=self.num_tailing_guidence_modules,
            num_guidence_modules=self.num_guidence_modules
        )
        if res_type == ResBlockType.down:
            tailing_module = (None if not (level_index in self.down_res_tailing_builders.keys()) else
                              self.down_res_tailing_builders[level_index](res_args_context))
        elif res_type == ResBlockType.up:
            tailing_module = (None if not (level_index in self.up_res_tailing_builders.keys()) else
                              self.up_res_tailing_builders[level_index](res_args_context))
        else:
            if level_index in self.middle_res_tailing_builders.keys():
                tailing_module = self.middle_res_tailing_builders[level_index](res_args_context)
            elif bool(level_index):
                tailing_module = None
            else:
                tailing_module = AttnBlock(out_ch)
        return DiffResBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                middle_dim=self.tdim,
                dropout=self.dropout,
                tailing_module=tailing_module,
                is_tailing_module_prior=self.is_tailing_module_prior,
                num_tailing_guidence_modules=self.num_tailing_guidence_modules,
                num_guidence_modules=self.num_guidence_modules
            )
    
    def res_block_parameters(self, t, *args):
        tailing_args_length = len(args) - len(self.guidence_modules)
        # Timestep embedding
        time_embedded = self.time_embedding(t)
        guidence_embedded = [guidence_module(args[tailing_args_length + i]) for i, guidence_module in enumerate(self.guidence_modules)]
        tailing_args = list(args[:tailing_args_length])
        tailing_initialized = [init_module(tailing_args[i]) for i, init_module in enumerate(self.tailing_init_modules)]
        tailing_initialized += tailing_args[len(tailing_initialized):]
        tailing_guidence_embedded = [tailing_guidence_module(tailing_initialized[i]) for i, tailing_guidence_module in enumerate(self.tailing_guidence_modules)]
        return time_embedded, *tailing_initialized, *tailing_guidence_embedded, *guidence_embedded

    @overload
    def forward(self, x, t, guidence, *args, additional_residuals: Optional[list] = None):
        ...
    
    @overload
    def forward(self, x, t, tailing, guidence, *args, additional_residuals: Optional[list] = None):
        ...

    @overload
    def forward(self, x, t, concatenation, guidence, *args, additional_residuals: Optional[list] = None):
        ...
    
    @overload
    def forward(self, x, t, concatenation, tailing, guidence, *args, additional_residuals: Optional[list] = None):
        ...
   
    def forward(self, x, t, *args, additional_residuals: Optional[list] = None):
        if self.concat_num > 0:
            x = torch.cat((x,) + args[:self.concat_num], dim=1)
            args = args[self.concat_num:]
        return super().forward(x, t, *args, additional_residuals=additional_residuals)