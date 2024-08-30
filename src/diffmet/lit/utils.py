import json
from pathlib import Path
from typing import Type
import torch
from lightning.pytorch.tuner.tuning import Tuner
import matplotlib.pyplot as plt

# FIXME importlib?
def get_class(class_path: str) -> Type:
    """
    adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/2.2.1/src/lightning/pytorch/cli.py#L730-L747
    """
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)



def run_lr_find(trainer, model, datamodule, ckpt_path: Path | None):
    if ckpt_path is not None:
        print(f'🤖 load {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        del ckpt

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


def get_next_version(root: Path) -> int:
    """
    adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/loggers/csv_logs.py#L171-L189
    """
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    existing_versions: list[int] = []
    for each in root.glob('version_*'):
        if each.is_dir():
            dir_ver = each.name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
        # else: warn

    return max(existing_versions) + 1 if len(existing_versions) > 0 else 0

def get_next_version_name(root: Path) -> str:
    version_num = get_next_version(root)
    return f'version_{version_num:d}'
