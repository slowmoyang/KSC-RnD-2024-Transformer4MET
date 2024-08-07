import torch
from torch import nn
from torch import Tensor


class SignLog1pAbs(nn.Module):

    def __init__(self, idx: list[int]) -> None:
        """
        """
        super().__init__()
        self.idx = idx

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor: # type: ignore
        output = input.clone()
        x = output[..., self.idx]
        x = x.sign() * x.abs().log1p()
        output[..., self.idx] = x
        return output

    @torch.no_grad()
    def inverse(self, input: Tensor) -> Tensor: # type: ignore
        output = input.clone()
        x = output[..., self.idx]
        x = x.sign() * x.abs().expm1()
        output[..., self.idx] = x
        return output
