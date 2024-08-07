import math
import torch
from torch import nn
from torch import Tensor
from tensordict import TensorDict


def rotate(vector: Tensor, rotation_matrix: Tensor):
    shape = vector.shape
    rank = len(shape)
    assert shape[-1] == 2

    if rank >= 3:
        repeats = math.prod(shape[1:-1])
        rotation_matrix = rotation_matrix.repeat_interleave(
            repeats=repeats,
            dim=0
        )
        vector = vector.reshape(-1, 2)

    vector = torch.vmap(torch.mv)(rotation_matrix, vector)

    if rank >= 3:
        vector = vector.reshape(*shape)
    return vector


class EventRotation(nn.Module):

    def forward(self,
                input: TensorDict,
    ) -> TensorDict:
        batch_size = len(input)

        # TODO double for accuracy?
        phi = 2 * torch.pi * torch.rand(batch_size, device=input.device)

        s = torch.sin(phi)
        c = torch.cos(phi)
        # rotation matrix
        # rot: (batch_size, 2, 2)
        rotation_matrix = torch.stack(
            tensors=[
                torch.stack([c, -s], dim=1),
                torch.stack([s, c], dim=1)
            ],
            dim=1
        )
        for key in ['track', 'tower', 'gen_met']:
            self.rotate(input, key, rotation_matrix)
        return input

    @staticmethod
    def rotate(input: TensorDict, key: str, rotation_matrix: Tensor):
        # select px and py
        momentum = input[key][..., :2]
        momentum = rotate(momentum, rotation_matrix)
        input[key] = torch.cat(
            tensors=[
                momentum,
                input[key][..., 2:]
            ],
            dim=-1
        )
