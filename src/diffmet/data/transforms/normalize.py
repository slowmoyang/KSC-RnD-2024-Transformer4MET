import torch
from torch import nn
from torch import Tensor


class Normalize(nn.Module):
    offset: Tensor
    scale: Tensor

    def __init__(self, offset, scale) -> None:
        super().__init__()

        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float))

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor: # type: ignore
        return (input - self.offset) / self.scale

    @torch.no_grad()
    def inverse(self, input: Tensor) -> Tensor: # type: ignore
        return input * self.scale + self.offset
