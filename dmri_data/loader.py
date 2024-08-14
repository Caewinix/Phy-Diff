from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy_ext as npe


def load_dmri_npk(path: str):
    # Load preprocessed image from a npk file.
    npk_data = npe.loadk(path, mmap_mode='r')
    return npk_data['images'], npk_data['b_vectors'], npk_data['b_values']


def load_dmri_adc_atlas_npk(path: str):
    npk_data = npe.loadk(path, mmap_mode='r')
    return npk_data['images'], npk_data['b_values']


class DmriNpkDataset(Dataset):
    def __init__(self, diffusion_path: str, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        self.diffusion_images, self.b_vectors, self.b_values = load_dmri_npk(diffusion_path)
        self.b_sequence_num, slice_position_num, height, width = self.diffusion_images.shape
        self.diffusion_images = self.diffusion_images.reshape(self.b_sequence_num * slice_position_num, 1, height, width)
    
    def __len__(self):
        return self.diffusion_images.shape[0]
    
    def __getitem__(self, index):
        item = torch.from_numpy(np.asarray(self.diffusion_images[index].copy()))
        if self.transform is not None:
            item = self.transform(item)
        return item.float()
    

class BaseMriNpyDataset(Dataset):
    def __init__(self, base_path: str, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        self.base_images = np.load(base_path, mmap_mode='r')
    
    def __len__(self):
        return self.base_images.shape[0]
    
    def __getitem__(self, index):
        item = torch.from_numpy(np.asarray(self.base_images[index].copy())).unsqueeze_(0)
        if self.transform is not None:
            item = self.transform(item)
        return item.float()

class AdcAtlasesNpkDataset(Dataset):
    def __init__(self, adc_atlases_path: str, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        self.adc_atlas_images, self.unique_b_values = load_dmri_adc_atlas_npk(adc_atlases_path)
        self.b_sequence_num, slice_position_num, height, width = self.adc_atlas_images.shape
        self.adc_atlas_images = self.adc_atlas_images.reshape(self.b_sequence_num * slice_position_num, 1, height, width)
    
    def __len__(self):
        return self.adc_atlas_images.shape[0]
    
    def __getitem__(self, index):
        item = torch.from_numpy(np.asarray(self.adc_atlas_images[index].copy()))
        min_value = torch.min(item).view(-1, 1, 1)
        max_value = torch.max(item).view(-1, 1, 1)
        if self.transform is not None:
            item = self.transform(item)
        return item.float(), min_value.float(), max_value.float()