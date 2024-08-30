import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class Bias(Metric):
    residual: list[Tensor]

    is_differentiable = False
    higher_is_better = None
    full_state_update = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('residual', default=[], dist_reduce_fx='cat')

    def update(self, residual: Tensor) -> None:
        self.residual.append(residual)

    def compute(self):
        residual = dim_zero_cat(self.residual)
        return residual.mean()


class AbsBias(Bias):

    higher_is_better = False

    def compute(self):
        residual = dim_zero_cat(self.residual)
        return residual.mean().abs()


class Resolution(Metric):
    residual: list[Tensor]

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('residual', default=[], dist_reduce_fx='cat')

    def update(self, residual: Tensor) -> None:
        self.residual.append(residual)

    def compute(self):
        residual = dim_zero_cat(self.residual)
        if len(residual):
            q16 = torch.quantile(residual, q=0.16, dim=0)
            q84 = torch.quantile(residual, q=0.84, dim=0)
            resolution = (q84 - q16) / 2
        else:
            resolution = torch.nan
        return resolution
