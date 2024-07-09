import copy
import itertools
import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module = self.module
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long))
        self.decay = decay

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
        )
        self_param_detached = []
        model_param_detached = []
        for p_averaged, p_model in zip(self_param, model_param):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            for avg_model_param, model_param in zip(self_param_detached, model_param_detached):
                avg_model_param.detach().mul_(self.decay).add_(model_param, alpha=1 - self.decay)

        self.n_averaged += 1