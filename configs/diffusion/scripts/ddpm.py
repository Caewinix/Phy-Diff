import torch
import numpy as np
import torch_ext as te
from foundation.modules.modules import *
from foundation.modules.attention import NeighborhoodTransformerLayer, GlobalTransformerLayer
from foundation.running.utils.normalization import normalize_across_origin, normalize_positive

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

class MLP(nn.Module):
    def __init__(self, vector_length: int, d_model: int, dim: int, dropout = 0, input_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.final = FlattenSequential(
            [nn.Linear(d_model * vector_length, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)],
            1
        )
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)
        x = self.layers(x)
        x = self.dropout(x)
        return self.final(x)

def guidence_modules(config):
    mapping_width = config.model.backbone.parameters.mapping_width
    return [MLP(4, mapping_width, mapping_width)]

def train_data_getter(config):
    def wrapper(data: tuple, device: torch.device):
        images = data[0].to(device)
        base_images = data[1].to(device)
        b_vectors = data[3].to(device)
        return images, base_images, b_vectors
    return wrapper

def test_data_getter(config):
    def wrapper(data: tuple, device: torch.device):
        base_images = data[1].to(device)
        b_vectors = data[3].to(device)
        return base_images, b_vectors
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
vector_transform = normalize_across_origin
