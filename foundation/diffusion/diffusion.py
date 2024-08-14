import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
import torch_ext as te
import numpy as np
from .functions import *


class GaussianDiffuser(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_1: float = 1.e-4,
        beta_T: float = 0.02,
        T: int = 1000
    ):
        super().__init__()

        self.model = model
        self.T = T

        self.betas = te.buffer(torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = te.buffer(torch.sqrt(alphas_bar))
        self.sqrt_one_minus_alphas_bar = te.buffer(torch.sqrt(1. - alphas_bar))
    
    @staticmethod
    def diffuse(x_0, noise, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar):
        # q_sample
        mean = extract(sqrt_alphas_bar, t, x_0.shape) * x_0
        std = extract(sqrt_one_minus_alphas_bar, t, x_0.shape)
        x_t = mean + std * noise
        return x_t
    
    def forward(self, x_0, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=x_0.shape[:1], device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = GaussianDiffuser.diffuse(x_0, noise, t, self.sqrt_alphas_bar, self.sqrt_one_minus_alphas_bar)
        loss = F.mse_loss(self.model(x_t, t, *cond, additional_residuals=additional_residuals), noise, reduction='none')
        return loss


class StructureDiffuser(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_1: float = 1.e-4,
        beta_T: float = 0.02,
        T: int = 1000,
        alpha_gap: float = 1e-8
    ):
        super().__init__()

        self.model = model
        self.T = T
        
        self.alpha_1 = 1 - beta_1
        self.alpha_T = 1 - beta_T
        self.alpha_gap = alpha_gap
    
    @staticmethod
    def diffuse(x_0, struct, noise, t, alpha_1, alpha_T, T, alpha_gap):
        # q_sample
        start, stop = te.map_ranges(struct, [(alpha_1 - alpha_gap, alpha_1), (alpha_T - alpha_gap, alpha_T)], dim=tuple(np.arange(1, len(struct.shape))), dtype=torch.float64) #1e-3
        # start, stop = te.invert(start), te.invert(stop)
        alphas_bar = te.linspace_cumprod_cover(t, start, stop, T, dtype=start.dtype)
        mean = torch.sqrt(alphas_bar) * x_0
        std = torch.sqrt(1. - alphas_bar)
        x_t = (mean + std * noise).to(x_0.dtype)
        return x_t
    
    def forward(self, x_0, struct, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=x_0.shape[:1], device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = StructureDiffuser.diffuse(x_0, struct, noise, t, self.alpha_1, self.alpha_T, self.T, self.alpha_gap)
        loss = F.mse_loss(self.model(x_t, t, *cond, additional_residuals=additional_residuals), noise, reduction='none')
        return loss