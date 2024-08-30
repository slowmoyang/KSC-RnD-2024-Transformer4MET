from pathlib import Path
from typing import Literal
from lightning.pytorch.callbacks import Callback
from ...utils.learningcurve import make_learning_curves




class LearningCurvePlotter(Callback):

    def __init__(self,
                 metric: str,
                 mode: str,
                 delimiter: str = '__',
    ) -> None:
        """
        Args:
            metric: a metric to find a best checkpoint
            mode:
        """
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.delimiter = delimiter

    def on_test_end(self, trainer, pl_module) -> None:
        if trainer.log_dir is None:
            raise RuntimeError('log_dir is None')
        log_dir = Path(trainer.log_dir)
        make_learning_curves(
            log_dir=log_dir,
            metric=self.metric,
            mode=self.mode,
            delimiter=self.delimiter,
        )
