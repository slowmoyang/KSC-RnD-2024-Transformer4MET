from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Final
import json
import uproot
import numpy as np
import vector
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


MARKER_DICT = {
    'rec': 's',
    'puppi': '^',
    'pf': 'o',
}

LABEL_DICT = {
    'rec': 'Deep Learning',
    'puppi': 'PUPPI',
    'pf': 'ParticleFlow (PF)'
}

COLOR_DICT = {
    'pf': 'tab:blue',
    'puppi': 'tab:green',
    'rec': 'tab:orange',
}


COMPONENT_LABEL_DICT = {
    'px': r'$p_{x}^{miss}$',
    'py': r'$p_{y}^{miss}$',
    'pt': r'$p_{T}^{miss}$',
    'phi': r'$\phi(\vec{p}_{T}^{miss})$',
}

COMPONENT_UNIT_DICT = {
    'px': 'GeV',
    'py': 'GeV',
    'pt': 'GeV',
    'phi': 'rad',
}

RANGE_DICT: Final[dict[str, tuple[float, float]]] = {
    'pt': (0, 400),
    'phi': (-np.pi, +np.pi),
    'px': (-400, 400),
    'py': (-400, 400),
}


def compute_resolution(y):
    return(np.percentile(y, 84)-np.percentile(y, 16))/2.0

@dataclass
class BinnedStatistic:
    x: np.ndarray
    xerr: np.ndarray
    mean: np.ndarray
    bias: np.ndarray
    resolution: np.ndarray
    range: tuple[float, float]

    @property
    def xmin(self):
        return self.x - self.xerr

    @property
    def xmax(self):
        return self.x + self.xerr

    @classmethod
    def from_arrays(cls,
                    gen: vector.MomentumNumpy2D,
                    rec: vector.MomentumNumpy2D,
                    component: str,
                    bins: int,
                    range: tuple[float, float],
    ):
        gen_val = getattr(gen, component)
        rec_val = getattr(rec, component)
        if component == 'phi':
            residual = rec.deltaphi(gen)
        else:
            residual = rec_val - gen_val

        kwargs = {'bins': bins, 'range': range}

        mean = binned_statistic(gen_val, rec_val, statistic='mean', **kwargs)
        bias = binned_statistic(gen_val, residual, statistic='mean', **kwargs)
        resolution = binned_statistic(gen_val, residual, statistic=compute_resolution, **kwargs)

        bin_centers = (mean.bin_edges[:-1] + mean.bin_edges[1:]) / 2
        bin_half_widths = (mean.bin_edges[1:] - mean.bin_edges[:-1]) / 2

        return cls(
            x=bin_centers,
            xerr=bin_half_widths,
            mean=mean.statistic,
            bias=bias.statistic,
            resolution=resolution.statistic,
            range=range,
        )

    def hlines(self, y, ax=None, **kwargs):
        ax = ax or plt.gca()
        return ax.hlines(y=y, xmin=self.xmin,  xmax=self.xmax, **kwargs)

    def plot_bias(self, ax=None, **kwargs):
        kwargs.setdefault('lw', 3)
        return self.hlines(y=self.bias, ax=ax, **kwargs)

    def plot_resolution(self, ax=None, **kwargs):
        kwargs.setdefault('lw', 3)
        return self.hlines(y=self.resolution, ax=ax, **kwargs)

    def plot_rec_vs_gen(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        return ax.errorbar(x=self.x, y=self.mean, xerr=self.xerr, yerr=self.resolution, **kwargs)



def plot_bias(stat_dict,
              component: str,
              label_dict: dict[str, str] = LABEL_DICT,
              color_dict: dict = COLOR_DICT,
              baseline_key: str = 'pf'
):
    fig, (ax_main, ax_ratio) = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1], sharex=True)
    fig.subplots_adjust(hspace=0)

    component_label = COMPONENT_LABEL_DICT[component]
    component_unit = COMPONENT_UNIT_DICT[component]

    ax_main.set_ylabel(rf'{component_label} bias, $b$ [{component_unit}]')

    ax_ratio.set_xlabel(f'Generated {component_label} [{component_unit}]')
    ax_ratio.set_ylabel(r'$|b| - |b_{PF}|$')

    for key, stat in stat_dict.items():
        label = label_dict[key]
        color = color_dict[key]
        stat.plot_bias(ax=ax_main, label=label, color=color)
    ax_main.axhline(0, ls=':', lw=3, color='gray')
    ax_main.legend()

    for key, stat in stat_dict.items():
        if key == baseline_key:
            continue
        y = np.abs(stat.bias) - np.abs(stat_dict[baseline_key].bias)
        stat.hlines(y=y, label=label_dict[key], color=color_dict[key], lw=2)
    ax_ratio.axhline(0, ls=':', lw=3, color=color_dict[baseline_key])
    return fig


def plot_resolution(stat_dict,
                    component: str,
                    label_dict: dict[str, str] = LABEL_DICT,
                    color_dict: dict = COLOR_DICT,
                    baseline_key: str = 'pf'
):
    fig, (ax_main, ax_ratio) = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1], sharex=True)
    fig.subplots_adjust(hspace=0)

    component_label = COMPONENT_LABEL_DICT[component]
    component_unit = COMPONENT_UNIT_DICT[component]

    ax_ratio.set_xlabel(f'Generated {component_label} [{component_unit}]')
    ax_main.set_ylabel(rf'{component_label} resolution, $\sigma$ [{component_unit}]')
    ax_ratio.set_ylabel(r'$\sigma_{PF} - \sigma$')

    for key, stat in stat_dict.items():
        stat.plot_resolution(
            ax=ax_main,
            label=label_dict[key],
            color=color_dict[key],
            lw=3)
    ax_main.legend()
    ax_main.grid()

    for key, stat in stat_dict.items():
        if key == baseline_key:
            continue
        y = stat.resolution - stat_dict['pf'].resolution
        stat.hlines(ax=ax_ratio, y=y, label=LABEL_DICT[key], color=COLOR_DICT[key], lw=2)

    ax_ratio.grid()
    return fig


def plot_metric(stat_dict, component, metric):
    if metric == 'bias':
        return plot_bias(stat_dict, component)
    elif metric == 'resolution':
        return plot_resolution(stat_dict, component)
    else:
        raise ValueError(f'{metric=}')


def analyse_result(log_dir: Path, treepath: str = 'tree'):
    input_path = log_dir / 'output.root'
    tree = uproot.open({input_path: treepath})
    data = tree.arrays(library='np')

    data = {algo: vector.MomentumNumpy2D({component: data[f'{algo}_met_{component}'] for component in ['pt', 'phi']})
            for algo in ['gen', 'rec', 'puppi', 'pf']}

    output_dir = log_dir / 'result'
    output_dir.mkdir()
    for component in ['pt', 'phi', 'px', 'py']:
        stat_dict = {}
        for algo in ['rec', 'puppi', 'pf']:
            stat_dict[algo] = BinnedStatistic.from_arrays(
                gen=data['gen'],
                rec=data[algo],
                component=component,
                bins=20,
                range=RANGE_DICT[component],
            )

            np.savez(output_dir / f'{algo}_{component}.npz',
                     **asdict(stat_dict[algo]))

        for metric in ['bias', 'resolution']:
            fig = plot_metric(stat_dict, component, metric=metric)
            output_path = output_dir / f'{component}_{metric}'
            for suffix in ['.png', '.pdf']:
                fig.savefig(output_path.with_suffix(suffix))
