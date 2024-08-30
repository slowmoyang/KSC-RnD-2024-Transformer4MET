from typing import Final
from lightning import LightningModule
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules import ModuleDict
from tensordict import TensorDict
from torchmetrics import MetricCollection
from ..metrics import Bias, AbsBias, Resolution
from ..utils.math import rectify_phi, to_polar
from ..models import L1PFModel, DelphesModel
from .utils import get_class
from ..optim import configure_optimizers
from ..data.transforms import SequentialBijection


DEFAULT_PT_BINNING: Final[list[list[float]]] = [
    [0, 45],
    [45, 70],
    [70, 100],
    [100, 165],
    [165, float('inf')],
]

class LitModel(LightningModule):

    def __init__(self,
                 augmentation,
                 preprocessing,
                 model: L1PFModel | DelphesModel,
                 criterion,
                 optimizer_class_path: str = 'torch.optim.AdamW',
                 lr: float = 3.0e-4,
                 optimizer_init_args: dict = {},
                 pt_binning: list[list[float]] = DEFAULT_PT_BINNING,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("lr")
        self.augmentation = nn.Sequential(*augmentation)

        self.preprocessing = ModuleDict({
            key: SequentialBijection(*value)
            for key, value in preprocessing.items()
        })

        self.model = model.to_tensor_dict_module()
        self.criterion = criterion

        self.optimizer_class = get_class(optimizer_class_path)
        self.optimizer_init_args = optimizer_init_args

        self.pt_binning = pt_binning
        self.val_metrics = self.build_metrics(self.pt_binning, 'val')
        self.test_metrics = self.build_metrics(self.pt_binning, 'test')

    def build_metrics(self,
                      pt_bins: list[list[float]],
                      stage: str,
    ) -> ModuleDict:
        """

        >>> eval_metrics["pt-0-45"]["pt"]
        """
        # FIXME gen met pt binning
        metrics = MetricCollection({
            'bias': Bias(),
            'absbias': AbsBias(),
            'res': Resolution(),
        })

        eval_metrics = ModuleDict()
        for low, up in pt_bins:
            pt_key = f'pt-{low:.0f}-{up:.0f}'

            eval_metrics[pt_key] = ModuleDict({
                key: metrics.clone(f'{stage}_{pt_key}_{key}_')
                for key in ['px', 'py', 'pt', 'phi']
            })
        return eval_metrics

    def training_step(self, # type: ignore
                      input: TensorDict,
    ) -> Tensor:
        input = self.augmentation(input)
        for key, value in self.preprocessing.items():
            input[key] = value(input[key])


        output = self.model(input)

        loss = self.criterion(input=output['rec_met'], target=output['gen_met'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _eval_step(self, # type: ignore
                  input: TensorDict,
                  metrics: ModuleDict,
                  stage: str,
    ) -> None:
        for key, value in self.preprocessing.items():
            input[key] = value(input[key])

        output = self.model(input)
        loss = self.criterion(input=output['rec_met'], target=output['gen_met'])

        gen_met = output['gen_met']
        rec_met = output['rec_met']
        if 'gen_met' in self.preprocessing.keys():
            met_preprocessing = self.preprocessing['gen_met']
            # undo normalisation
            gen_met: Tensor = met_preprocessing.inverse(gen_met) # type: ignore
            rec_met: Tensor = met_preprocessing.inverse(rec_met) # type: ignore

        # (px, py) to (pt, phi)
        gen_met_polar = to_polar(gen_met)
        rec_met_polar = to_polar(rec_met)

        gen_met_pt = gen_met_polar[:, 0]

        residual = rec_met - gen_met
        residual_polar = rec_met_polar - gen_met_polar
        residual_polar[:, 1] = rectify_phi(residual_polar[:, 1])

        for low, up in self.pt_binning:
            pt_key = f'pt-{low:.0f}-{up:.0f}'

            pt_mask = torch.logical_and(gen_met_pt > low, gen_met_pt < up)

            masked_residual = residual[pt_mask]
            masked_residual_polar = residual_polar[pt_mask]

            metrics[pt_key]['px'].update(masked_residual[:, 0]) # type: ignore
            metrics[pt_key]['py'].update(masked_residual[:, 1]) # type: ignore
            metrics[pt_key]['pt'].update(masked_residual_polar[:, 0]) # type: ignore
            metrics[pt_key]['phi'].update(masked_residual_polar[:, 1]) # type: ignore

        self.log(f'{stage}_loss', loss, prog_bar=True)

        return output

    def _on_eval_epoch_end(self, metrics: ModuleDict, stage: str):
        log_dict = {}
        for component_dict in metrics.values():
            for each in component_dict.values():
                log_dict |= each.compute() # type: ignore

        # val_pt-300-350_phi_bias
        for low, up in self.pt_binning:
            pt_key = f'pt-{low:.0f}-{up:.0f}'
            prefix = f'{stage}_{pt_key}'
            log_dict[f'{prefix}_sum-px-absbias-py-absbias'] = log_dict[f'{prefix}_px_absbias'] + log_dict[f'{prefix}_py_absbias']

        self.log_dict(log_dict, prog_bar=True)


    def validation_step(self, # type: ignore
                        input: TensorDict,
    ) -> None:
        return self._eval_step(input=input, metrics=self.val_metrics,
                               stage='val')

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end(metrics=self.val_metrics, stage='val')

    def test_step(self, # type: ignore
                  input: TensorDict,
    ) -> None:
        return self._eval_step(input=input, metrics=self.test_metrics,
                               stage='test')

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end(metrics=self.test_metrics, stage='test')

    def predict_step(self, # type: ignore[override]
                     input
    ):
        for key, value in self.preprocessing.items():
            input[key] = value(input[key])

        output = self.model(input)
        rec_met = output['rec_met']

        if 'gen_met' in self.preprocessing.keys():
            met_preprocessing = self.preprocessing['gen_met']
            rec_met = met_preprocessing.inverse(rec_met)
        return rec_met

    def configure_optimizers(self):
        return configure_optimizers(
            model=self,
            optimizer_class=self.optimizer_class,
            lr=self.hparams['lr'],
            **self.optimizer_init_args
        )
