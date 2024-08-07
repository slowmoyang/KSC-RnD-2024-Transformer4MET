import torch
from torch import Tensor
from torch.nn import Sequential


class SequentialBijection(Sequential):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    @torch.no_grad()
    def inverse(self, input: Tensor) -> Tensor:
        output = input
        for layer in reversed(self):
            output = layer.inverse(output)
        return output

