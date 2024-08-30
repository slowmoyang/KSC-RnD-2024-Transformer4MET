import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
from .attention import CrossAttentionModule
from .attention import SelfAttentionModule


class PerceiverEncoder(nn.Module):
    latent: nn.Parameter

    def __init__(self,
                 latent_len: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 activation: str,
                 widening_factor: int,
                 dropout_p: float,
                 weight_sharing: bool,
    ) -> None:
        """
        Args:
            latent_len:
            embed_dim:
            num_heads:
            num_layers:
            dropout_p:
        """
        super().__init__()
        self.register_parameter(
            name='latent',
            param=self.init_latent(latent_len, embed_dim)
        )

        self.cross_attn = CrossAttentionModule(
            embed_dim=embed_dim,
            num_heads=1,
            activation=activation,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

        block = SelfAttentionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation=activation,
            widening_factor=widening_factor,
            dropout_p=dropout_p
        )

        if weight_sharing:
            self.block_list = nn.ModuleList([block] * num_layers)
        else:
            self.block_list = _get_clones(block, num_layers)

    @staticmethod
    def init_latent(length: int,
                    num_features: int,
                    std: float = 0.02
    ) -> nn.Parameter:
        latent = torch.empty(length, num_features)
        nn.init.trunc_normal_(latent, std=std)
        return nn.Parameter(latent) # type: ignore

    def forward(self, input: Tensor, data_mask: Tensor) -> Tensor:
        latent = self.latent.unsqueeze(0).repeat(input.size(0), 1, 1)
        z = self.cross_attn(
            target=latent,
            source=input,
            source_data_mask=data_mask)

        for block in self.block_list:
            z = block(input=z, data_mask=None)
        # FIXME
        return z



class Perceiver(nn.Module):

    def __init__(self,
                 latent_len: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: float,
                 num_layers: int,
                 activation: str,
                 widening_factor: int,
                 weight_sharing: bool,
    ):
        super().__init__()

        self.encoder = PerceiverEncoder(
            latent_len=latent_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            num_layers=num_layers,
            activation=activation,
            widening_factor=widening_factor,
            weight_sharing=weight_sharing,
        )

    def forward(self,
                input: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        output = self.encoder(input=input, data_mask=data_mask)
        output = output.mean(dim=1)
        return output

