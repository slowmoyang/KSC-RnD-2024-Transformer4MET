from pathlib import Path
from lightning.pytorch.callbacks import Callback
from ...utils.learningcurve import make_learning_curves

class LearningCurvePlotter(Callback):

    def on_test_end(self, trainer, pl_module) -> None:
        if trainer.log_dir is None:
            raise RuntimeError('log_dir is None')
        log_dir = Path(trainer.log_dir)
        make_learning_curves(log_dir)
