import abc
from torch import nn
from tensordict.nn import TensorDictModule


def init_modules(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Model(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 projection,
                 backbone,
                 regression_head
    ) -> None:
        super().__init__()

        self.projection = projection
        self.backbone = backbone
        self.regression_head = regression_head

        self.reset_parameters()

    @abc.abstractmethod
    def to_tensor_dict_module(self) -> TensorDictModule:
        ...

    def reset_parameters(self):
        self.apply(init_modules)
