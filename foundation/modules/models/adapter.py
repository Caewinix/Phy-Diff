import torch
from torch import Tensor
from torch import nn
import numpy as np
from typing import List, Tuple, Sequence


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class SimpleResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, need_downsample, kernel_size=3, skep=False, use_conv=True):
        super().__init__()
        ps = kernel_size // 2
        if in_channels != out_channels or skep == False:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, ps)
        if skep == False:
            self.skep = nn.Conv2d(in_channels, out_channels, kernel_size, 1, ps)
        else:
            self.skep = None

        self.down = need_downsample
        if self.down == True:
            self.down_opt = Downsample(in_channels, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None: # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = [128, 256, 384, 512],
        num_resnet_block: int = 3,
        kernel_size: int = 3,
        downscale_factor: int = 8,
        skep: bool = True,
        use_conv: bool = False
    ):
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.num_resnet_block = num_resnet_block
        self.body = []
        for i in range(len(channels)):
            for j in range(num_resnet_block):
                if (i != 0) and (j == 0):
                    self.body.append(
                        SimpleResBlock(channels[i - 1], channels[i], True, kernel_size, skep, use_conv=use_conv))
                else:
                    self.body.append(
                        SimpleResBlock(channels[i], channels[i], False, kernel_size, skep, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(in_channels * downscale_factor * downscale_factor, channels[0], 3, 1, 1)

    def forward(self, x: torch.Tensor, *, need_tokens: bool = False):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        channels_len = len(self.channels)
        features = np.empty(channels_len, dtype=object)
        x = self.conv_in(x)
        for i in range(channels_len):
            for j in range(self.num_resnet_block):
                idx = i * self.num_resnet_block + j
                x = self.body[idx](x)
            if need_tokens:
                features[i] = x.movedim(-3, -1)
            else:
                features[i] = x

        return list(features)