from pathlib import Path
from typing import Final, cast
import uproot.writing
from tensordict import TensorDict
import vector
import numpy as np
from lightning.pytorch.callbacks import Callback
from ...analysis import analyse_result


class ResultWriter(Callback):
    file: uproot.WritableDirectory

    ALGO_LIST: Final[list[str]] = ['gen', 'rec', 'puppi', 'pf']

    def __init__(self) -> None:
        super().__init__()

    def on_test_start(self, trainer, pl_module) -> None:
        if trainer.log_dir is None:
            raise RuntimeError
        log_dir = Path(trainer.log_dir)
        path = log_dir / 'output.root'
        self.file = uproot.writing.create(path)

        # FIXME
        branch_names = [f'{algo}_met_{component}'
                        for algo in self.ALGO_LIST
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
                    for key in self.ALGO_LIST}

        ########################################################################
        # undo preprocessing
        ########################################################################
        if 'gen_met' in pl_module.preprocessing.keys():
            preprocessing = pl_module.preprocessing['gen_met']
            for key in ['gen', 'rec']:
                met_dict[key] = preprocessing.inverse(met_dict[key]) # type: ignore

        ########################################################################
        # torch.Tensor to vector.MomentumNumpy2D
        ########################################################################
        # tensor to numpy ndarray
        met_dict = {key: value.cpu().numpy()
                    for key, value in met_dict.items()}

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
            for algo in self.ALGO_LIST
            for component in ['pt', 'phi']
        }

        self.file['tree'].extend(chunk)

    def on_test_end(self,
                    trainer,
                    pl_module
    ) -> None:
        if trainer.log_dir is None:
            raise RuntimeError
        log_dir = Path(trainer.log_dir)
        self.file.close()

        input_path = log_dir / 'output.root'
        tree = uproot.open({input_path: 'tree'})
        data = tree.arrays(library='np')

        output_dir = log_dir / 'result'
        output_dir.mkdir()

        analyse_result(data, output_dir)
