from torch import Tensor
from tensordict.nn.common import NestedKey
from tensordict.nn import TensorDictModule
from .base import Model


class DelphesModel(Model):

    def forward(self, # type: ignore[override]
                track: Tensor,
                track_data_mask,
                tower: Tensor,
                tower_data_mask,
    ) -> Tensor:
        x, x_data_mask = self.projection(
            track,
            track_data_mask,
            tower,
            tower_data_mask,
        )

        z = self.backbone(x, x_data_mask)
        y = self.regression_head(z)
        return y

    def to_tensor_dict_module(self):
        in_keys: list[NestedKey] = [
            'track',
            ('masks', 'track'),
            'tower',
            ('masks', 'tower'),
        ]

        out_keys: list[NestedKey] = [
            'rec_met'
        ]

        return TensorDictModule(
            module=self,
            in_keys=in_keys,
            out_keys=out_keys,
        )
