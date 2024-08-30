from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Final, cast
import json
from matplotlib.figure import Figure
import uproot
import numpy as np
import vector
from hist.hist import Hist
from hist.axis import Regular
from hist.storage import Double
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from vector import MomentumNumpy2D


MARKER_DICT = {
    'rec': 's',
    'puppi': '^',
    'pf': 'o',
}

LABEL_DICT = {
    'gen': 'Generated',
    'rec': 'Deep Learning',
    'puppi': 'PUPPI',
    'pf': 'ParticleFlow (PF)'
}

COLOR_DICT = {
    'gen': 'black',
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


def compute_bias(residual_arr):
    return np.mean(residual_arr)


def compute_resolution(residual_arr):
    p84 = np.percentile(residual_arr, 84)
    p16 = np.percentile(residual_arr, 16)
    return (p84 - p16) / 2

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
                    gen: MomentumNumpy2D,
                    rec: MomentumNumpy2D,
                    component: str,
                    x_component: str | None = None,
                    bins: int = 20,
                    range: tuple[float, float] | None = None,
    ):
        x_component = x_component or component

        range = range or RANGE_DICT[x_component]

        x = getattr(gen, x_component)
        gen_val = getattr(gen, component)
        rec_val = getattr(rec, component)
        if component == 'phi':
            residual = rec.deltaphi(gen)
        else:
            residual = rec_val - gen_val

        kwargs = {
            'x': x,
            'bins': bins,
            'range': range,
        }

        mean = binned_statistic(values=rec_val, statistic='mean', **kwargs)
        bias = binned_statistic(values=residual, statistic='mean', **kwargs)
        resolution = binned_statistic(values=residual, statistic=compute_resolution, **kwargs)

        bin_centers = (mean.bin_edges[:-1] + mean.bin_edges[1:]) / 2
        bin_half_widths = np.diff(mean.bin_edges) / 2

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

    def plot_abs_bias(self, ax=None, **kwargs):
        kwargs.setdefault('lw', 3)
        return self.hlines(y=np.abs(self.bias), ax=ax, **kwargs)

    def plot_resolution(self, ax=None, **kwargs):
        kwargs.setdefault('lw', 3)
        return self.hlines(y=self.resolution, ax=ax, **kwargs)

    def plot_rec_vs_gen(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        return ax.errorbar(x=self.x, y=self.mean, xerr=self.xerr, yerr=self.resolution, **kwargs)



def plot_bias(stat_dict,
              component: str,
              x_component: str | None = None,
              label_dict: dict[str, str] = LABEL_DICT,
              color_dict: dict = COLOR_DICT,
              baseline_key: str = 'pf'
):
    x_component = x_component or component

    fig, (ax_main, ax_ratio) = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1], sharex=True)
    fig.subplots_adjust(hspace=0)

    component_label = COMPONENT_LABEL_DICT[component]
    component_unit = COMPONENT_UNIT_DICT[component]

    x_component_label = COMPONENT_LABEL_DICT[x_component]
    x_component_unit = COMPONENT_UNIT_DICT[x_component]

    ax_main.set_ylabel(rf'{component_label} bias, $b$ [{component_unit}]')

    ax_ratio.set_xlabel(f'Generated {x_component_label} [{x_component_unit}]')
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
                    x_component: str | None = None,
                    label_dict: dict[str, str] = LABEL_DICT,
                    color_dict: dict = COLOR_DICT,
                    baseline_key: str = 'pf'
):
    x_component = x_component or component

    fig, (ax_main, ax_ratio) = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1], sharex=True)
    fig.subplots_adjust(hspace=0)

    component_label = COMPONENT_LABEL_DICT[component]
    component_unit = COMPONENT_UNIT_DICT[component]

    x_component_label = COMPONENT_LABEL_DICT[x_component]
    x_component_unit = COMPONENT_UNIT_DICT[x_component]

    ax_ratio.set_xlabel(f'Generated {x_component_label} [{x_component_unit}]')
    ax_main.set_ylabel(rf'{component_label} resolution, $\sigma$ [{component_unit}]')
    ax_ratio.set_ylabel(r'$\sigma - \sigma_{PF}$')

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


def plot_metric(stat_dict, component, x_component, metric) -> plt.Figure:
    kwargs = dict(
        stat_dict=stat_dict,
        component=component,
        x_component=x_component,
    )

    if metric == 'bias':
        return plot_bias(**kwargs)
    elif metric == 'resolution':
        return plot_resolution(**kwargs)
    else:
        raise ValueError(f'{metric=}')


def plot_component(met_dict: dict[str, MomentumNumpy2D], component: str) -> plt.Figure:
    fig, ax = plt.subplots()
    fig = cast(plt.Figure, fig)
    ax = cast(plt.Axes, ax)

    key_list = list(reversed(met_dict.keys()))

    xlow, xup = RANGE_DICT[component]
    hist_axis = Regular(20, xlow, xup)

    for key, met in met_dict.items(): # FIXME
        hist = Hist(hist_axis, storage=Double()) # type: ignore
        hist.fill(getattr(met_dict[key], component))

        label = LABEL_DICT[key]
        color = COLOR_DICT[key]

        hist.plot(ax=ax, label=label, color=color, lw=2, histtype='step',
                  density=True, yerr=True)

    ax.legend()
    ax.grid()

    xlabel = f'{COMPONENT_LABEL_DICT[component]} [{COMPONENT_UNIT_DICT[component]}]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('AU')
    return fig


def save_fig(fig: Figure, output_path: Path) -> None:
    #fig.tight_layout()
    for suffix in ['.png', '.pdf']:
        fig.savefig(output_path.with_suffix(suffix))


def analyse_result(data: dict[str, np.ndarray],
                   output_dir: Path,
) -> None:
    met_dict: dict[str, MomentumNumpy2D] = { # type: ignore
        algo: MomentumNumpy2D(dict(pt=data[f'{algo}_met_pt'],
                                   phi=data[f'{algo}_met_phi']))
            for algo in ['gen', 'rec', 'puppi', 'pf']
    }

    for component in ['pt', 'phi', 'px', 'py']:

        fig = plot_component(met_dict, component)
        save_fig(fig, output_dir / component)

        stat_dict = {}
        for algo in ['rec', 'puppi', 'pf']:
            stat_dict[algo] = BinnedStatistic.from_arrays(
                gen=met_dict['gen'],
                rec=met_dict[algo],
                component=component,
                x_component=component,
                bins=20,
            )

            np.savez(output_dir / f'{algo}_{component}.npz',
                     **asdict(stat_dict[algo]))

        for metric in ['bias', 'resolution']:
            fig = plot_metric(stat_dict, component=component, x_component=component, metric=metric)
            save_fig(fig, output_dir / f'{component}_{metric}_vs_{component}')

        #######################################################################
        #
        #######################################################################
        if component != 'pt':
            stat_dict = {}
            for algo in ['rec', 'puppi', 'pf']:
                stat_dict[algo] = BinnedStatistic.from_arrays(
                    gen=met_dict['gen'],
                    rec=met_dict[algo],
                    component=component,
                    x_component='pt',
                    bins=20,
                )

                np.savez(output_dir / f'{algo}_{component}.npz',
                         **asdict(stat_dict[algo]))

            for metric in ['bias', 'resolution']:
                fig = plot_metric(stat_dict, component, x_component=component, metric=metric)
                save_fig(fig, output_dir / f'{component}_{metric}_vs_pt')
