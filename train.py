#!/usr/bin/env python
import json
from pathlib import Path
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner.tuning import Tuner
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as mh
from diffmet.lit.model import LitModel
from diffmet.lit.datamodule import DataModule
mpl.use('agg')
mh.style.use(mh.styles.CMS)


def run_lr_find(trainer, model, datamodule):
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    assert lr_finder is not None

    assert trainer.log_dir is not None
    log_dir = Path(trainer.log_dir)

    output_dir = log_dir / 'lr-find'
    output_dir.mkdir()

    with open(output_dir / 'results.json', 'w') as stream:
        output = {
            'suggestion': lr_finder.suggestion(),
            'results': lr_finder.results,
        }
        json.dump(output, stream, indent=2)

    fig = lr_finder.plot(suggest=True)
    assert fig is not None

    output_path = output_dir / 'plot'
    for suffix in ['.png', '.pdf']:
        fig.savefig(output_path.with_suffix(suffix))
    plt.close(fig)


def run(trainer: Trainer,
        model: LitModel,
        datamodule: DataModule,
):
    torch.set_num_threads(1)

    run_lr_find(trainer=trainer, model=model, datamodule=datamodule)

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)



def main():
    cli = LightningCLI(
        model_class=LitModel,
        datamodule_class=DataModule,
        seed_everything_default=1234,
        run=False, # used to de-activate automatic fitting.
        trainer_defaults={
            'max_epochs': 10,
            'accelerator': 'gpu',
            'devices': [0],
            'log_every_n_steps': 1,
        },
        save_config_kwargs={
            'overwrite': True
        },
        parser_kwargs={
            'parser_mode': 'omegaconf'
        }
    )

    run(
        trainer=cli.trainer,
        model=cli.model,
        datamodule=cli.datamodule
    )



if __name__ == '__main__':
    main()
