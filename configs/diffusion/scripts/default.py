import torch
import numpy as np
import torch_ext as te
from foundation.modules.modules import *
from foundation.modules.attention import NeighborhoodTransformerLayer, GlobalTransformerLayer
from foundation.running.util.normalization import normalize_across_origin, normalize_positive

print(__name__)

class SelectiveNeighborhoodTransformerLayer(NeighborhoodTransformerLayer):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0, cond_index=None):
        super().__init__(d_model, d_ff, d_head, cond_features, kernel_size, dropout)
        self.cond_index = cond_index
        
    def forward(self, x, pos, cond, *args):
        if self.cond_index is None:
            return super().forward(x, pos, cond)
        return super().forward(x, pos, cond, args[self.cond_index])

class SelectiveGlobalTransformerLayer(GlobalTransformerLayer):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0, cond_index=None):
        super().__init__(d_model, d_ff, d_head, cond_features, dropout)
        self.cond_index = cond_index
    
    def forward(self, x, pos, cond, *args):
        if self.cond_index is None:
            return super().forward(x, pos, cond)
        return super().forward(x, pos, cond, args[self.cond_index])

def layer_block_builders(config):
    mapping_width = config.model.backbone.parameters.mapping_width
    return {
        # d_ff = context.width * 3
        0: lambda context: SelectiveNeighborhoodTransformerLayer(context.width, context.width, 64, mapping_width, 7, 0., cond_index=None),
        1: lambda context: SelectiveNeighborhoodTransformerLayer(context.width, context.width, 64, mapping_width, 7, 0., cond_index=None),
        2: lambda context: SelectiveGlobalTransformerLayer(context.width, context.width, 64, mapping_width, 0., cond_index=None),
        3: lambda context: SelectiveNeighborhoodTransformerLayer(context.width, context.width, 64, mapping_width, 7, 0., cond_index=None),
        4: lambda context: SelectiveNeighborhoodTransformerLayer(context.width, context.width, 64, mapping_width, 7, 0., cond_index=None),
    }

def guidence_modules(config):
    mapping_width = config.model.backbone.parameters.mapping_width
    return [VectorEmbedding(4, mapping_width, mapping_width),
            ContinuityEmbedding(110, mapping_width, mapping_width),
            RealEmbedding(1, mapping_width, mapping_width)]

def train_data_getter(config):
    def wrapper(data: tuple, device: torch.device):
        images = data[0].to(device)
        base_images = data[1].to(device)
        adc_atlas = data[-2].to(device)
        b_vectors = data[3].to(device)
        b_values = data[4].to(device).view((-1,) + (1,) * 3)
        position_index = data[2].to(device)
        structures = torch.exp(-2 * b_values * adc_atlas)
        return images, structures, base_images, b_vectors, position_index, b_values - 5000
    return wrapper

def test_data_getter(config):
    def wrapper(data: tuple, device: torch.device):
        base_images = data[1].to(device)
        adc_atlas = data[-2].to(device)
        b_vectors = data[3].to(device)
        b_values = data[4].to(device).view((-1,) + (1,) * 3)
        position_index = data[2].to(device)
        structures = torch.exp(-2 * b_values * adc_atlas)
        return structures, base_images, b_vectors, position_index, b_values - 5000
    return wrapper

def test_comparison_getter(config):
    def wrapper(data: tuple, device: torch.device):
        return data[0].to(device)
    return wrapper

def test_initial_state_getter(config):
    def wrapper(data: tuple, device: torch.device):
        return data[1].to(device)
    return wrapper

def sampled_images_transform(config):
    def wrapper(img):
        return img
    return wrapper

def transform(tensor: torch.Tensor):
    dim = tuple(np.arange(1, len(tensor.shape)))
    min_value = te.min_dims(tensor, dims=dim, keepdim=True)
    max_value = te.max_dims(tensor, dims=dim, keepdim=True)
    difference = max_value - min_value
    difference[difference == 0] = 1
    tensor = (tensor - min_value) / difference
    _, H, W = tensor.shape
    padding_left = (256 - W) // 2
    padding_right = 256 - W - padding_left
    padding_top = (256 - H) // 2
    padding_bottom = 256 - H - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    tensor = F.pad(tensor, padding, "constant", value=0)
    return tensor * 2 - 1

def adc_atlases_transform(tensor: torch.Tensor):
    dim = tuple(np.arange(1, len(tensor.shape)))
    min_value = te.min_dims(tensor, dims=dim, keepdim=True)
    max_value = te.max_dims(tensor, dims=dim, keepdim=True)
    difference = max_value - min_value
    difference[difference == 0] = 1
    tensor = (tensor - min_value) / difference
    _, H, W = tensor.shape
    padding_left = (256 - W) // 2
    padding_right = 256 - W - padding_left
    padding_top = (256 - H) // 2
    padding_bottom = 256 - H - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    tensor = F.pad(tensor, padding, "constant", value=0)
    return tensor * (max_value - min_value) + min_value
vector_transform = normalize_across_origin

# adapter=Adapter(in_channels=1,kernel_size=1, channels=[128,256,512], downscale_factor=4)
# a,b,c=adapter(torch.rand((1,1,256,256)), need_tokens=True)