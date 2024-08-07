import warnings
import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler


class FlatGenMETpTRandomSampler(WeightedRandomSampler):
    """
    This sampler aims at sampling events so that generated MET pT distribution
    is flat.
    """

    def __init__(self,
                 dataset,
                 boundaries: Tensor | None = None,
    ):
        """
        Args:
            dataset: dataset, where gen met are taken
            boundaries: boundaries of a gen met pt histogram
        """
        boundaries = boundaries or torch.linspace(0, 400, 41)

        # gen_met is supposed to be (px, py)
        gen_met_pt = [each['gen_met'].norm(p=2) for each in dataset]
        gen_met_pt = torch.stack(gen_met_pt)
        gen_met_pt.clip_(boundaries[0], boundaries[-1])

        hist, _ = torch.histogram(input=gen_met_pt, bins=boundaries)
        if hist.eq(0).any():
            warnings.warn(message='found empty bin')
            hist.clamp_(1, None) # FIXME warning

        # FIXME
        pdf = hist / len(hist)

        bins = torch.bucketize(gen_met_pt, boundaries=boundaries, right=False) - 1

        # sampling weights
        weights = 1 / pdf[bins]

        super().__init__(
            weights=weights.tolist(),
            num_samples=len(weights)
        )
