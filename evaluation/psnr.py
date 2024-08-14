import torch
from torch import nn
import numpy as np
from torch_ext import map_range

_20_mul_log_10_255 = 20 * torch.log10(torch.tensor(255))

def psnr_core(image_1, image_2):
    dim = tuple(np.arange(1, len(image_1.shape)))
    image_1 = map_range(image_1, (0, 255), dim=dim)
    image_2 = map_range(image_2, (0, 255), dim=dim)
    return image_1, image_2

class PSNRMetric(nn.Module):
    def forward(self, image_1, image_2) -> torch.Tensor:
        # image_1, image_2 = psnr_core(image_1, image_2)
        # mse = torch.nn.functional.mse_loss(image_1, image_2)
        # return _20_mul_log_10_255 - 10 * torch.log10(mse)
        return PSNRMetric.value_from(image_1, image_2)
    
    @staticmethod
    def value_from(image_1, image_2) -> torch.Tensor:
        image_1, image_2 = psnr_core(image_1, image_2)
        mse = torch.mean(torch.square(image_1 - image_2), dim=(1, 2, 3))
        return (_20_mul_log_10_255 - 10 * torch.log10(mse)).unsqueeze_(1)