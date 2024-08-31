#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path
from typing import Any
from socket import gethostname
from htcondor.htcondor import Submit
from htcondor.htcondor import Schedd
from coolname import generate_slug
from diffmet.lit.utils import get_next_version_name


PROJECT_PREFIX = Path(__file__).parent
CONFIG_DIR = PROJECT_PREFIX / 'config'


def run(experiment: str,
        task: str | None,
        version: str | None,
        config: Path | None,
        data: Path | None,
        sampler: Path | None,
        augmentation: Path | None,
        preprocessing: Path | None,
        model: Path | None,
        loss: Path | None,
        optimizer: Path | None,
        trainer: Path | None,
        lr_scheduler: Path | None,
        memory: str
):
    log_dir = PROJECT_PREFIX / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    save_dir = log_dir / experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    task = task or generate_slug(2)
    task_dir = save_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    version = version or get_next_version_name(task_dir)
    version_dir = task_dir / version
    version_dir.mkdir()

    executable = PROJECT_PREFIX / 'train.py'

    condor_log_dir = version_dir / 'condor'
    condor_log_dir.mkdir()

    config_dir = version_dir / 'config'
    config_dir.mkdir()

    config_dict = dict(
        data=data,
        sampler=sampler,
        augmentation=augmentation,
        preprocessing=preprocessing,
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        trainer=trainer,
        config=config,
    )

    # ordered
    config_to_flag = {
        'config': 'config',
        'data': 'data',
        'sampler': 'data',
        'augmentation': 'model',
        'preprocessing': 'model',
        'model': 'model',
        'loss': 'model',
        'optimizer': 'model',
        'trainer': 'trainer',
        'lr_scheduler': 'lr_scheduler',
    }

    arg_list = []
    for key, value in config_dict.items():
        if value is not None:
            value = value.resolve()
            if not value.exists():
                raise FileNotFoundError(value)
            tmp_config = str(config_dir / f'{key}.yaml')
            shutil.copyfile(str(value), tmp_config)
            flag = config_to_flag[key]
            arg_list.append(f'--{flag} {tmp_config}')

    arg_list += [
        f'--trainer.logger.class_path lightning.pytorch.loggers.CSVLogger',
        f'--trainer.logger.init_args.save_dir {save_dir}',
        f'--trainer.logger.init_args.name {task}',
        f'--trainer.logger.init_args.version {version}',
    ]

    arguments = ' '.join(arg_list)

    job_batch_name = f'diffmet.training.{experiment}.{task}.{version}'

    raw_submit: dict[str, Any] = {
        'universe': 'vanilla',
        'getenv': 'True',
        # job
        'executable': executable,
        'arguments': arguments,
        #
        'should_transfer_files': 'YES',
        'when_to_transfer_output': 'ON_EXIT',
        # resources
        'request_cpus': 1,
        'request_GPUs': 1,
        'request_memory': memory,
        #
        'JobBatchName': job_batch_name,
        'output': condor_log_dir / 'output.log',
        'error': condor_log_dir / 'error.log',
        'log': condor_log_dir / 'log.log',
        'priority': 100,
    }

    submit: dict[str, str] = {key: str(value)
                              for key, value in raw_submit.items()}

    submit = Submit(submit)
    schedd = Schedd()
    schedd.submit(submit)

    print(f'🚀🚀🚀 submit {job_batch_name}')

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    hostname = gethostname()
    if hostname == 'gate.sscc.uos.ac.kr':
        host_alias = 'uos'
    elif hostname.endswith('.knu.ac.kr'):
        host_alias = 'knu'
    else:
        raise RuntimeError(hostname)

    config_default_dict = {
        'config': None,
        'data': CONFIG_DIR / 'data' / 'delphes' / host_alias / 'enhanched-high-met.yaml',
        'sampler': CONFIG_DIR / 'sampler' / 'base.yaml',
        'augmentation': CONFIG_DIR / 'augmentation' / 'delphes' / 'base.yaml',
        'preprocessing': CONFIG_DIR / 'preprocessing' / 'delphes' / 'base.yaml',
        'model': CONFIG_DIR / 'model' / 'delphes' / 'base.yaml',
        'loss': CONFIG_DIR / 'loss' / 'MSE.yaml',
        'optimizer': CONFIG_DIR / 'optimizer' / 'AdamW.yaml',
        'trainer': CONFIG_DIR / 'trainer' / 'base.yaml',
        'lr_scheduler': None,
    }

    config_name_list = list(config_default_dict.keys())

    for name in config_name_list:
        flag = f'--{name}'
        help = f'{name} config'
        default = config_default_dict[name]

        parser.add_argument(flag, type=Path, default=default, help=help)

    parser.add_argument('-e', '--experiment', type=str, required=True,
                        help='experiment name like sanity-check or data-imbalance-mitigation')
    parser.add_argument('-t', '--task', type=str, help='task name')
    parser.add_argument('-v', '--version', type=str, help='version')
    parser.add_argument('-m', '--memory', type=str, default='50GB', help='request memory')

    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
