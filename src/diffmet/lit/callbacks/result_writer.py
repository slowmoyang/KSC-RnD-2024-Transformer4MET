from pathlib import Path
from typing import cast
import uproot.writing
from hist.hist import Hist
from hist.axis import Regular
from hist.storage import Double
from tensordict import TensorDict
import vector
import numpy as np
from lightning.pytorch.callbacks import Callback
from ...analysis import analyse_result


class ResultWriter(Callback):
    file: uproot.WritableDirectory

    def __init__(self) -> None:
        super().__init__()

        axis_dict = {
            'pt': Regular(1000, 0, 1000),
            'phi': Regular(1000, -np.pi, +np.pi),
            'px': Regular(1000, -1000, 1000),
            'py': Regular(1000, -1000, 1000),
        }

        self.met_dict: dict[str, Hist] = {
            f'{algo}_{component}': Hist(axis_dict[component], storage=Double())
            for algo in ['gen', 'rec', 'pf', 'puppi']
            for component in ['pt', 'phi', 'px', 'py']
        }

        residual_axis_dict = {
            'pt': Regular(1000, -1000, 1000),
            'phi': Regular(1000, -np.pi, +np.pi),
            'px': Regular(1000, -1000, 1000),
            'py': Regular(1000, -1000, 1000),
        }

        gen_met_pt_axis = Regular(1000, 0, 1000, name='gen_met_pt', label=r'Generated $p_{T}^{miss}\ [GeV]$')

        self.residual_dict: dict[str, Hist] = {
            f'{algo}_{component}': Hist(gen_met_pt_axis, residual_axis_dict[component], storage=Double())
            for algo in ['rec', 'pf', 'puppi']
            for component in ['pt', 'phi', 'px', 'py']
        }

    def on_test_start(self, trainer, pl_module) -> None:
        if trainer.log_dir is None:
            raise RuntimeError
        log_dir = Path(trainer.log_dir)
        path = log_dir / 'output.root'
        self.file = uproot.writing.create(path)

        # FIXME
        branch_names = [f'{algo}_met_{component}'
                        for algo in ['gen', 'rec', 'puppi', 'pf']
                        for component in ['pt', 'phi']]

        branch_types = {each: np.float32 for each in branch_names}
        self.file.mktree(name='tree', branch_types=branch_types)


    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_idx,
                          dataloader_idx=0
    ):
        outputs = cast(TensorDict, outputs)

        met_dict = {key: outputs[f'{key}_met']
                    for key in ['gen', 'rec', 'pf', 'puppi']}

        ########################################################################
        # undo preprocessing
        ########################################################################
        if 'gen_met' in pl_module.preprocessing.keys():
            preprocessing = pl_module.preprocessing['gen_met']
            # undo normalisation
            for key in ['gen', 'rec']:
                met_dict[key] = preprocessing.inverse(met_dict[key]) # type: ignore

        ########################################################################
        # torch.Tensor to vector.MomentumNumpy2D
        ########################################################################
        # tensor to numpy ndarray
        met_dict = {key: value.cpu().numpy() for key, value in met_dict.items()}

        # numpy to MomentumNumpy2D
        met_dict = {
            key: vector.MomentumNumpy2D({each: value[:, idx]
                                         for idx, each
                                         in enumerate(['px', 'py'])})
            for key, value in met_dict.items()
        }

        ########################################################################
        # fill tree
        ########################################################################
        chunk = {
            f'{algo}_met_{component}': getattr(met_dict[algo], component)
            for algo in ['gen', 'rec', 'puppi', 'pf']
            for component in ['pt', 'phi']
        }

        self.file['tree'].extend(chunk)


        ########################################################################
        #
        #######################################################################
        for key in self.met_dict.keys():
            algo, component = key.split('_')
            self.met_dict[key].fill(getattr(met_dict[algo], component))

        ########################################################################
        #
        #######################################################################
        residual_dict = {}
        for prefix in ['rec', 'pf', 'puppi']:
            for suffix in ['pt', 'px', 'py']:
                residual_dict[f'{prefix}_{suffix}'] = getattr(met_dict[prefix], suffix) - getattr(met_dict['gen'], suffix)
            residual_dict[f'{prefix}_phi'] = met_dict[prefix].deltaphi(met_dict['gen'])

        for key in self.residual_dict.keys():
            algo, component = key.split('_')
            self.residual_dict[key].fill(met_dict['gen'].pt, residual_dict[key])

    def on_test_end(self,
                    trainer,
                    pl_module
    ) -> None:
        if trainer.log_dir is None:
            raise RuntimeError
        log_dir = Path(trainer.log_dir)

        for key, value in self.met_dict.items():
            algo, component = key.split('_')
            self.file[f'met/{algo}/{component}'] = value

        for key, value in self.residual_dict.items():
            algo, component = key.split('_')
            self.file[f'residual/{algo}/{component}'] = value

        self.file.close()

        analyse_result(log_dir)
