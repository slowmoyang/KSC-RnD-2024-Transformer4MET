#!/usr/bin/env python
import sys
from pathlib import Path
import argparse
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import matplotlib as mpl
import mplhep as mh
from diffmet.lit.model import LitModel
from diffmet.lit.datamodule import DataModule
from diffmet.lit.utils import run_lr_find
mpl.use('agg')
mh.style.use(mh.styles.CMS)


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--ckpt', dest='ckpt_path', type=Path,
                            help='checkpoint')
        parser.add_argument('--lr-find', action='store_true',
                            help='run learning rate finding')
        parser.add_argument('--validate-before-fit', action=argparse.BooleanOptionalAction,
                            default=True,
                            help='validate before fit')


def run(trainer: Trainer,
        model: LitModel,
        datamodule: DataModule,
        config,
):
    torch.set_num_threads(1)

    ckpt_path = config['ckpt_path']

    if config['lr_find']:
        print('🚀 LEARNING RATE FINDING')
        run_lr_find(trainer=trainer, model=model, datamodule=datamodule,
                    ckpt_path=ckpt_path)

    if config['validate_before_fit']:
        print('🚀 VALIDATION')
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    print('🚀 FIT')
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    print('🚀 TEST')
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')


def main():
    cmd = ' '.join(sys.argv)
    print(f'🤖 parsing "{cmd}"')

    cli = LitCLI(
        model_class=LitModel,
        datamodule_class=DataModule,
        seed_everything_default=1337,
        run=False, # used to de-activate automatic fitting.
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
        datamodule=cli.datamodule,
        config=cli.config,
    )



if __name__ == '__main__':
    main()
