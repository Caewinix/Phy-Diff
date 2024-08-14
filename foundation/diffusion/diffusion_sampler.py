from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
import torch_ext as te
import numpy as np
from tqdm import tqdm
from .diffusion import GaussianDiffuser, StructureDiffuser
from .functions import *
from interface import *


class DiffusionSampler(Interface[nn.Module]):
    @abstractmethod
    def sample(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        pass


class DiffusionModelSampler(nn.Module):
    def __init__(self, model, sampler: DiffusionSampler):
        super().__init__()

        self.model = model
        self.sampler = sampler

    def forward(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        return self.sampler(x_T, *cond, additional_residuals=additional_residuals, initial_state=initial_state, strength=strength, repeat_noise=repeat_noise)


class DDPMSampler(nn.Module, metaclass=DiffusionSampler):
    def __init__(self, model: nn.Module, beta_1: float, beta_T: float, T: int):
        super().__init__()

        self.model = model
        self.T = T

        self.betas = te.buffer(torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.coeff1 = te.buffer(torch.sqrt(1. / alphas))
        self.coeff2 = te.buffer(self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.posterior_var = te.buffer(self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.alphas_bar = te.buffer(alphas_bar)

    def __predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def __p_mean_variance(self, x_t, t, cond, additional_residuals):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, *cond, additional_residuals=additional_residuals)
        xt_prev_mean = self.__predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def sample(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        # p_sample
        """
        Algorithm 2.
        """
        if initial_state is None or strength >= 1.:
            t_start = self.T - 1
            x_t = x_T
        else:
            t_start = int((self.T - 1) * strength)
            strengthened_t = torch.full((x_T.shape[0],), t_start, dtype=torch.long, device=x_T.device)
            x_t = GaussianDiffuser.diffuse(initial_state, x_T, strengthened_t, torch.sqrt(self.alphas_bar), torch.sqrt(1. - self.alphas_bar))
        for timestep in tqdm(range(t_start, -1, -1), leave=False):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * timestep
            mean, var= self.__p_mean_variance(x_t=x_t, t=t, cond=cond, additional_residuals=additional_residuals)
            # no noise when t == 0
            if timestep == 0:
                noise = 0
            elif repeat_noise:
                noise = torch.randn((1, *x_t.shape[1:]))
            else:
                noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def forward(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        return self.sample(x_T, *cond, additional_residuals=additional_residuals, initial_state=initial_state, strength=strength, repeat_noise=repeat_noise)


class StructureDDPMSampler(nn.Module, metaclass=DiffusionSampler):
    def __init__(self, model: nn.Module, alpha_gap: float, beta_1: float, beta_T: float, T: int):
        super().__init__()

        self.model = model
        self.alpha_gap = alpha_gap
        self.T = T
        self.alpha_1 = 1 - beta_1
        self.alpha_T = 1 - beta_T

    def __predict_xt_prev_mean_from_eps(self, x_t, alphas, alphas_bar, eps):
        assert x_t.shape == eps.shape
        coeff1 = torch.sqrt(1. / alphas)
        coeff2 = coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar)
        return coeff1 * x_t - coeff2 * eps

    def __p_mean_variance(self, x_t, t, cond, additional_residuals, start, stop):
        alphas = te.linspace_cover(t, start, stop, self.T, dtype=x_t.dtype)
        alphas_bar = te.linspace_cumprod_cover(t, start, stop, self.T, dtype=x_t.dtype)
        # below: only log_variance is used in the KL computations
        var = 1 - alphas
        t_0_index = torch.where(t == 0)[0]
        if len(t_0_index) > 0:
            t_0 = x_t.new_zeros([x_t.shape[0], ], dtype=torch.long)
            alphas_bar_for_var = te.linspace_cumprod_cover(t_0 + 1, start, stop, self.T, dtype=x_t.dtype)
            alphas_bar_prev_for_var = te.linspace_cumprod_cover(t_0, start, stop, self.T, dtype=x_t.dtype)
            var[t_0_index] = (1 - alphas_bar_for_var[t_0_index]) * (1. - alphas_bar_prev_for_var[t_0_index]) / (1. - alphas_bar_for_var[t_0_index])
        eps = self.model(x_t, t, *cond, additional_residuals=additional_residuals)
        xt_prev_mean = self.__predict_xt_prev_mean_from_eps(x_t, alphas, alphas_bar, eps)

        return xt_prev_mean, var

    def sample(self, x_T: torch.Tensor, struct: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        # p_sample
        """
        Algorithm 2.
        """
        if initial_state is None or strength >= 1.:
            t_start = self.T - 1
            x_t = x_T
        else:
            t_start = int((self.T - 1) * strength)
            strengthened_t = torch.full((x_T.shape[0],), t_start, dtype=torch.long, device=x_T.device)
            x_t = StructureDiffuser.diffuse(initial_state, struct, x_T, strengthened_t, self.alpha_1, self.alpha_T, self.T, self.alpha_gap)
        start, stop = te.map_ranges(struct, [(self.alpha_1 - self.alpha_gap, self.alpha_1), (self.alpha_T - self.alpha_gap, self.alpha_T)], dim=tuple(np.arange(1, len(struct.shape))))
        # start, stop = te.invert(start), te.invert(stop)
        for timestep in tqdm(range(t_start, -1, -1), leave=False):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * timestep
            mean, var = self.__p_mean_variance(x_t, t, cond, additional_residuals, start, stop)
            # no noise when t == 0
            if timestep == 0:
                noise = 0
            elif repeat_noise:
                noise = torch.randn((1, *x_t.shape[1:]))
            else:
                noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def forward(self, x_T: torch.Tensor, struct: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        return self.sample(x_T, struct, *cond, additional_residuals=additional_residuals, initial_state=initial_state, strength=strength, repeat_noise=repeat_noise)


class DDIMSampler(nn.Module, metaclass=DiffusionSampler):
    class Method(Enum):
        uniform = auto()
        quad = auto()
    
    def __init__(
        self,
        model: nn.Module,
        beta_1: float,
        beta_T: float,
        T: int,
        expected_timesteps: int = 50,
        method: Method = Method.uniform,
        eta: float = 0.
    ):
        super().__init__()

        self.model = model
        self.T = T

        if method == DDIMSampler.Method.uniform:
            c = T // expected_timesteps
            self.ddim_timesteps = np.asarray(list(range(0, T, c)))
        elif method == DDIMSampler.Method.quad:
            self.ddim_timesteps = ((np.linspace(0, np.sqrt(T * .8), expected_timesteps)) ** 2).astype(int)
        self.ddim_prev_timesteps = np.concatenate([[0], self.ddim_timesteps[:-1]])
        
        self.ddim_eta = eta

        self.betas = te.buffer(torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar = te.buffer(alphas_bar)

    def sample(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        # p_sample
        """
        Algorithm 2.
        """
        flip_ddim_timesteps = np.flip(self.ddim_timesteps)
        flip_ddim_prev_timesteps = np.flip(self.ddim_prev_timesteps)
        if initial_state is None or strength >= 1.:
            skip_steps = 0
            x_t = x_T
            ddim_timesteps = flip_ddim_timesteps
            ddim_prev_timesteps = flip_ddim_prev_timesteps
        else:
            t_start = int((self.T - 1) * strength)
            strengthened_t = torch.full((x_T.shape[0],), t_start, dtype=torch.long, device=x_T.device)
            x_t = GaussianDiffuser.diffuse(initial_state, x_T, strengthened_t, torch.sqrt(self.alphas_bar), torch.sqrt(1. - self.alphas_bar))
            skip_steps = np.argmin(np.abs(flip_ddim_timesteps - t_start))
            if t_start >= flip_ddim_timesteps[skip_steps]:
                skip_steps += 1
            ddim_timesteps = flip_ddim_timesteps[skip_steps:]
            ddim_prev_timesteps = flip_ddim_prev_timesteps[skip_steps:]
        length = len(ddim_timesteps)
        for timestep, prev_timestep in tqdm(zip(ddim_timesteps, ddim_prev_timesteps), leave=False, total=length):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * timestep
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_timestep
            ddim_alpha = extract(self.alphas_bar, t, x_t.shape)
            ddim_alpha_prev = extract(self.alphas_bar, prev_t, x_t.shape)
            ddim_sqrt_one_minus_alpha = torch.sqrt(1. - ddim_alpha)
            
            e_t = self.model(x_t, t, *cond, additional_residuals=additional_residuals)
            pred_x0 = (x_t - ddim_sqrt_one_minus_alpha * e_t) / torch.sqrt(ddim_alpha)
            if self.ddim_eta == 0:
                ddim_sigma = 0
                sigma_noise = 0
            else:
                ddim_sigma = self.ddim_eta * torch.sqrt((1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev))
                noise = noise_like(x_t.shape, x_t.device, repeat_noise)
                sigma_noise = ddim_sigma * noise
            dir_xt = torch.sqrt(1. - ddim_alpha_prev - ddim_sigma ** 2) * e_t
            x_t = torch.sqrt(ddim_alpha_prev) * pred_x0 + dir_xt + sigma_noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def forward(self, x_T: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        return self.sample(x_T, *cond, additional_residuals=additional_residuals, initial_state=initial_state, strength=strength, repeat_noise=repeat_noise)


class StructureDDIMSampler(nn.Module, metaclass=DiffusionSampler):
    def __init__(
        self,
        model: nn.Module,
        alpha_gap: float,
        beta_1: float,
        beta_T: float,
        T: int,
        expected_timesteps: int = 50,
        method: DDIMSampler.Method = DDIMSampler.Method.uniform,
        eta: float = 0.
    ):
        super().__init__()

        self.model = model
        self.T = T
        self.alpha_gap = alpha_gap
        self.alpha_1 = 1 - beta_1
        self.alpha_T = 1 - beta_T
        
        if method == DDIMSampler.Method.uniform:
            c = T // expected_timesteps
            self.ddim_timesteps = np.asarray(list(range(0, T, c)))
        elif method == DDIMSampler.Method.quad:
            self.ddim_timesteps = ((np.linspace(0, np.sqrt(T * .8), expected_timesteps)) ** 2).astype(int)
        self.ddim_prev_timesteps = np.concatenate([[0], self.ddim_timesteps[:-1]])
        
        self.ddim_eta = eta

    def sample(self, x_T: torch.Tensor, struct: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        # p_sample
        """
        Algorithm 2.
        """
        start, stop = te.map_ranges(struct, [(self.alpha_1 - self.alpha_gap, self.alpha_1), (self.alpha_T - self.alpha_gap, self.alpha_T)], dim=tuple(np.arange(1, len(struct.shape))))
        
        flip_ddim_timesteps = np.flip(self.ddim_timesteps)
        flip_ddim_prev_timesteps = np.flip(self.ddim_prev_timesteps)
        if initial_state is None or strength >= 1.:
            skip_steps = 0
            x_t = x_T
            ddim_timesteps = flip_ddim_timesteps
            ddim_prev_timesteps = flip_ddim_prev_timesteps
        else:
            t_start = int((self.T - 1) * strength)
            strengthened_t = torch.full((x_T.shape[0],), t_start, dtype=torch.long, device=x_T.device)
            x_t = StructureDiffuser.diffuse(initial_state, struct, x_T, strengthened_t, self.alpha_1, self.alpha_T, self.T, self.alpha_gap)
            skip_steps = np.argmin(np.abs(flip_ddim_timesteps - t_start))
            if t_start >= flip_ddim_timesteps[skip_steps]:
                skip_steps += 1
            ddim_timesteps = flip_ddim_timesteps[skip_steps:]
            ddim_prev_timesteps = flip_ddim_prev_timesteps[skip_steps:]
        length = len(ddim_timesteps)
        for timestep, prev_timestep in tqdm(zip(ddim_timesteps, ddim_prev_timesteps), leave=False, total=length):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * timestep
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * prev_timestep
            ddim_alpha = te.linspace_cumprod_cover(t, start, stop, self.T, dtype=x_t.dtype)
            ddim_alpha_prev = te.linspace_cumprod_cover(prev_t, start, stop, self.T, dtype=x_t.dtype)
            ddim_sqrt_one_minus_alpha = torch.sqrt(1. - ddim_alpha)
            
            e_t = self.model(x_t, t, *cond, additional_residuals=additional_residuals)
            pred_x0 = (x_t - ddim_sqrt_one_minus_alpha * e_t) / torch.sqrt(ddim_alpha)
            if self.ddim_eta == 0:
                ddim_sigma = 0
                sigma_noise = 0
            else:
                ddim_sigma = self.ddim_eta * torch.sqrt((1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev))
                noise = noise_like(x_t.shape, x_t.device, repeat_noise)
                sigma_noise = ddim_sigma * noise
            dir_xt = torch.sqrt(1. - ddim_alpha_prev - ddim_sigma ** 2) * e_t
            x_t = torch.sqrt(ddim_alpha_prev) * pred_x0 + dir_xt + sigma_noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def forward(self, x_T: torch.Tensor, struct: torch.Tensor, *cond, additional_residuals: Optional[Sequence[torch.Tensor]] = None, initial_state: Optional[torch.Tensor] = None, strength: float = 0.8, repeat_noise: bool = False):
        return self.sample(x_T, struct, *cond, additional_residuals=additional_residuals, initial_state=initial_state, strength=strength, repeat_noise=repeat_noise)