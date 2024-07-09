import numpy as np
import os
from typing import Optional, Callable, Sequence, Union
from torchvision.utils import _log_api_usage_once
from torch.utils.data import Dataset
import torch
from sortedcontainers import SortedDict
from monai.transforms import DivisiblePad
from .loader import load_dmri_adc_atlas_npk, load_dmri_npk
from foundation.running.utils.normalization import normalize_positive

class ToNumpyImage:
    # A Transform that may be used.
    def __init__(self, mode=None):
        _log_api_usage_once(self)
        self.mode = mode

    def __call__(self, pic):
        arr = np.array(pic)
        if len(arr.shape) < 3:
            arr = np.expand_dims(arr, axis=2)
        return arr

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += f"mode={self.mode}"
        format_string += ")"
        return format_string
    

class DmriDataset(Dataset):
    def __init__(
            self,
            diffusion_dir: str,
            base_dir: str,
            adc_atlases_path: str,
            xtract_atlases_path: Optional[str] = None,
            restricted_b_value: Union[float, Sequence[float], None] = None,
            restricted_slice_range: Union[range, Sequence[int], None] = None,
            restricted_tolerance: float = 10+1e-6,
            transform: Optional[Callable] = None,
            adc_atlases_transform: Optional[Callable] = None,
            vector_transform: Optional[Callable] = None,
        ):
        self.transform = transform
        self.adc_atlases_transform = adc_atlases_transform
        self.vector_transform = vector_transform

        diffusion_filenames = sorted(os.listdir(diffusion_dir))
        diffusion_filenames = np.array([filename for filename in diffusion_filenames if filename.endswith('.npk')])
        self.__diffusion_images, self.__b_vectors, self.__b_values = np.vectorize(lambda filename: load_dmri_npk(os.path.join(diffusion_dir, filename)), otypes=[object, object, object])(diffusion_filenames)
        base_filenames = sorted(os.listdir(base_dir))
        base_filenames = np.array([filename for filename in base_filenames if filename.endswith('.npy')])
        self.__base_images = np.vectorize(lambda filename: np.load(os.path.join(base_dir, filename), mmap_mode='r'), otypes=[object])(base_filenames)
        self.__adc_atlases_images, self.unique_b_values = load_dmri_adc_atlas_npk(adc_atlases_path)
        self.__xtract_atlases = None if xtract_atlases_path is None else torch.from_numpy(np.load(xtract_atlases_path)).float()
        
        if restricted_slice_range is not None:
            if restricted_slice_range is range:
                start = restricted_slice_range.start
                stop = restricted_slice_range.stop
            else:
                start = restricted_slice_range[0]
                stop = restricted_slice_range[1]
            self.__diffusion_images = np.vectorize(lambda subject_images: subject_images[:, start:stop, :, :], otypes=[object])(self.__diffusion_images)
            self.__base_images = np.vectorize(lambda subject_images: subject_images[start:stop, :, :], otypes=[object])(self.__base_images)
            self.__adc_atlases_images = self.__adc_atlases_images[:, start:stop, :, :]
            self.__xtract_atlases = self.__xtract_atlases[start:stop, :, :, :]
        
        self.__subject_based_shapes = np.vectorize(lambda images: images.shape[:2], otypes=[object])(self.__diffusion_images)
        self.__subject_based_length = np.vectorize(lambda images: images.shape[0] * images.shape[1], otypes=[int])(self.__diffusion_images)
        self.__subject_based_length = np.cumsum(self.__subject_based_length)

        self.__adc_b_indexes = SortedDict()
        
        b_index = 0
        def place_adc_b_index(b_value: int):
            nonlocal b_index
            self.__adc_b_indexes.setdefault(b_value, b_index)
            b_index += 1
        np.vectorize(place_adc_b_index, otypes=[])(self.unique_b_values)
        self.unique_b_values = torch.from_numpy(np.asarray(self.unique_b_values.copy()))
        
        if restricted_b_value is None:
            self.__indices = np.arange(self.__subject_based_length[-1])
        else:
            if not isinstance(restricted_b_value, Sequence):
                restricted_b_value = [restricted_b_value]
            self.__indices = np.empty(0, dtype=int)
            stack_b_values = self.__b_values.copy()
            for subject_index, subject_shape in enumerate(self.__subject_based_shapes):
                stack_b_values[subject_index] = np.broadcast_to(stack_b_values[subject_index], subject_shape)
            stack_b_values = np.vstack(stack_b_values).reshape(-1,)
            for b_value in restricted_b_value:
               self.__indices = np.concatenate([self.__indices, np.where(np.abs(stack_b_values - b_value) <= restricted_tolerance)[0]])
            self.__indices = np.unique(self.__indices)
    
    def subject_bin_index(self, full_index):
        subject_index = np.searchsorted(self.__subject_based_length, full_index, 'right')
        if subject_index > 0:
            bin_index = full_index - self.__subject_based_length[subject_index - 1]
        else:
            bin_index = full_index
        return subject_index, bin_index
    
    def b_position_index(self, subject_index, bin_index):
        return np.unravel_index(bin_index, self.__subject_based_shapes[subject_index])
    
    def __len__(self):
        return len(self.__indices)
 
    def __get_item(self, index):
        subject_id, bin_index = self.subject_bin_index(index)
        b_sequence_index, position_index = self.b_position_index(subject_id, bin_index)
        image, b_vector, b_value = (np.asarray(self.__diffusion_images[subject_id][b_sequence_index, position_index].copy()),
                                    np.asarray(self.__b_vectors[subject_id][b_sequence_index].copy()),
                                    np.asarray(self.__b_values[subject_id][b_sequence_index].copy()))
        
        image = torch.from_numpy(image).unsqueeze_(0)
        base_image = torch.from_numpy(
            np.asarray(self.__base_images[subject_id][position_index]).copy()).unsqueeze_(0)
        adc_atlas = (torch.from_numpy(np.asarray(
                self.__adc_atlases_images[self.__adc_b_indexes[b_value.item()], position_index]).copy()).unsqueeze_(0)
                     if b_value != 0 else torch.full(image.shape, 1.))
        
        b_value = torch.from_numpy(b_value)
        b_vector = torch.from_numpy(b_vector)
        b_vector = torch.concat((b_value, b_vector), dim=0)
        # b_label = torch.tensor(torch.where(b_value == self.unique_b_values)[0].item())
        original_base_image = base_image
        if self.transform is not None:
            image = self.transform(image)
            base_image = self.transform(base_image)
        if self.adc_atlases_transform is not None:
            adc_atlas = self.adc_atlases_transform(adc_atlas)
        if self.vector_transform is not None:
            b_vector = self.vector_transform(b_vector)
        
        basic_return_args = image.float(), base_image.float(), torch.tensor(position_index), b_vector.float(), b_value.float(), adc_atlas.float()
        
        if self.__xtract_atlases is None:
            return basic_return_args
        else:
            xtract_atlases = self.__xtract_atlases[position_index]
            channel_size, _, _ = xtract_atlases.shape
            non_zero_mask = xtract_atlases.view(channel_size, -1).any(-1)
            xtract_atlases[torch.logical_not(non_zero_mask)] = (torch.sum(xtract_atlases[non_zero_mask], dim=0) + 0.2) * normalize_positive(original_base_image)
            if self.transform is not None:
                xtract_atlases = self.transform(xtract_atlases)
            return *basic_return_args, xtract_atlases
    
    def __getitem__(self, index):
        return self.__get_item(self.__indices[index])