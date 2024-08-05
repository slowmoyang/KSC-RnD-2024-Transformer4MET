import torch
from torch import Tensor
from torch.nn import Sequential


class TransformSequential(Sequential):
    """
    one-to-one
    """

    def __init__(self, transform_list: list) -> None:
        super().__init__(*transform_list)

    @torch.no_grad()
    def inverse(self, input: Tensor) -> Tensor:
        output = input
        for layer in reversed(self):
            output = layer.inverse(output)
        return output

