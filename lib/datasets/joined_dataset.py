__all__ = ['JoinedDataset']

from typing import Iterable, List, Optional

from torch.utils.data import ConcatDataset, Dataset

from lib import datasets


class JoinedDataset(ConcatDataset):
    def __init__(self,
                 concatenated_datasets: Iterable[dict],
                 weighted_sampling: bool = True,
                 weights: Optional[Iterable[float]] = None  # Weights for each dataset; overrides weighted sampling
                 ):
        dsets: List[Dataset] = []
        for dset_config in concatenated_datasets:
            dset_type = dset_config.pop('type')
            dsets.append(getattr(datasets, dset_type)(**dset_config))
        super().__init__(dsets)
        dsets_length: List[int] = [len(dset) for dset in dsets]
        if weights is not None:
            assert len(weights) == len(dsets), "Please specify one weight per dataset"
        elif weighted_sampling:
            weights = [1] * len(dsets_length)
        self.sampling_weights = self._get_sampling_weights(dsets_length, weights)

    @staticmethod
    def _get_sampling_weights(ds_lengths: Iterable[int],
                              ds_weights: Optional[Iterable[float]],
                              ) -> List[float]:
        if ds_weights is not None:
            weights = []
            for cur_length, ds_weight in zip(ds_lengths, ds_weights):
                weights.extend([ds_weight / cur_length] * cur_length)
            return weights
        else:
            return [1] * sum(ds_lengths)
