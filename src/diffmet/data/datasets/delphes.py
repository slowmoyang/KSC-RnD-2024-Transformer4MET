import time
import os.path
import glob
import numpy as np
import awkward as ak
from tensordict import TensorDict
import uproot
import vector
import h5py as h5
import torch
from torch import Tensor
from .base import TensorDictListDataset
from .utils import convert_ak_to_tensor
vector.register_awkward()

class DelphesDataset(TensorDictListDataset):

    @classmethod
    def _from_root(cls,
                  path: str,
                  treepath: str = 'tree',
                  entry_stop: int | None = None
    ):

        tree = uproot.open(f'{path}:{treepath}')

        expressions: list[str] = [
            # track
            'track_px',
            'track_py',
            'track_eta',
            'track_charge',
            'track_is_electron',
            'track_is_muon',
            'track_is_hadron',
            'track_is_reco_pu',
            # tower
            'tower_px',
            'tower_py',
            'tower_eta',
            'tower_is_hadron',
            # genMet
            'gen_met_pt',
            'gen_met_phi',
            # puppi
            'puppi_met_pt',
            'puppi_met_phi',
            # pf
            'pf_met_pt',
            'pf_met_phi',
        ]

        data = tree.arrays( # type: ignore
            expressions=expressions,
            entry_stop=entry_stop,
        )

        gen_met_chunk = ak.Array(
            data={
                'pt': data.gen_met_pt,
                'phi': data.gen_met_phi,
            },
            with_name='Momentum2D'
        )

        puppi_chunk = ak.Array(
            data={
                'pt': data.puppi_met_pt,
                'phi': data.puppi_met_phi,
            },
            with_name='Momentum2D'
        )

        pf_chunk = ak.Array(
            data={
                'pt': data.pf_met_pt,
                'phi': data.pf_met_phi,
            },
            with_name='Momentum2D'
        )

        track_chunk = zip(
            data.track_px,
            data.track_py,
            data.track_eta,
            data.track_charge,
            data.track_is_electron,
            data.track_is_muon,
            data.track_is_hadron,
            data.track_is_reco_pu,
        )

        track_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1)).float()
            for each in track_chunk
        ]

        tower_chunk = zip(
            data.tower_px,
            data.tower_py,
            data.tower_eta,
            data.tower_is_hadron,
        )

        tower_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1)).float()
            for each in tower_chunk
        ]

        gen_met_chunk = np.stack(
            arrays=[
                gen_met_chunk.px, # type: ignore
                gen_met_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        gen_met_chunk = convert_ak_to_tensor(gen_met_chunk).float()

        # puppi
        puppi_chunk = np.stack(
            arrays=[
                puppi_chunk.px, # type: ignore
                puppi_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        puppi_chunk = convert_ak_to_tensor(puppi_chunk).float()

        # pf
        pf_chunk = np.stack(
            arrays=[
                pf_chunk.px, # type: ignore
                pf_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        pf_chunk = convert_ak_to_tensor(pf_chunk).float()

        example_list = [
            TensorDict(
                source={
                    'track': track,
                    'tower': tower,
                    'gen_met': gen_met,
                    'puppi_met': puppi,
                    'pf_met': pf,
                },
                batch_size=[]
            )
            for track, tower, gen_met, puppi, pf
            in zip(track_chunk, tower_chunk, gen_met_chunk, puppi_chunk, pf_chunk)
        ]

        return cls(example_list)

    @classmethod
    def _from_h5(cls,
                  path: str,
    ):
        track_keys = [
            'track_px',
            'track_py',
            'track_eta',
            'track_charge',
            'track_is_electron',
            'track_is_muon',
            'track_is_hadron',
            'track_is_reco_pu',
        ]

        tower_keys = [
            'tower_px',
            'tower_py',
            'tower_eta',
            'tower_is_hadron',
        ]


        with h5.File(path, 'r') as file:
            def read_pf(key_list):
                pf = [file[key][:] for key in key_list] # type: ignore
                pf = [torch.from_numpy(np.stack(each, axis=1).astype(np.float32))
                      for each in zip(*pf)]
                return pf

            def read_met(prefix):
                pt: Tensor = torch.from_numpy(file[f'{prefix}_met_pt'][:]) # type: ignore
                phi: Tensor = torch.from_numpy(file[f'{prefix}_met_phi'][:]) # type: ignore
                px = pt * torch.cos(phi)
                py = pt * torch.sin(phi)
                met = torch.stack([px, py], dim=1)
                return met

            data = dict(
                track=read_pf(track_keys),
                tower=read_pf(tower_keys),
                gen_met=read_met('gen'),
                puppi_met=read_met('puppi'),
                pf_met=read_met('pf'),
            )

        data = [TensorDict(source=dict(zip(data.keys(), each)), batch_size=[])
                for each in zip(*data.values())]
        return cls(data)


    @classmethod
    def load(cls, path_list: list[str]):
        """
        Args:
            path_list: a list of paths, which can contain a glob pattern
        Returns:
            DelphesDataset
        """
        _, suffix = os.path.splitext(path_list[0])
        if suffix == '.root':
            method = cls._from_root
        elif suffix in ('.h5', '.hdf5'):
            method = cls._from_h5
        else:
            raise RuntimeError(f'{suffix=}')

        # FIXME
        pattern_list = path_list

        dataset = cls([])
        for pattern in pattern_list:
            path_list = glob.glob(pattern)
            if len(path_list) == 0:
                raise RuntimeError(f'globbing {pattern} gives an empty list')

            for path in path_list:
                print(f'loading {path}', end='')
                start = time.time()
                dataset += method(path=path)
                elapsed_time = time.time() - start
                print(f' ({elapsed_time:.1f} s)')
        return dataset
