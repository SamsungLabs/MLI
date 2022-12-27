import collections.abc as container_abcs
import functools
import logging
from typing import Optional

import numpy as np
import torch.multiprocessing as torchmp
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
try:
    import horovod.torch as hvd
except ImportError:
    pass

from lib import datasets
from lib.datasets.joined_dataset import JoinedDataset
from lib.datasets.samplers import DistributedSamplerWrapper
from lib.utils.base import seed_freeze

logger = logging.getLogger(__name__)


def _collate_without_seqs(batch):
    """
    Do not apply default_collate recursively to the inputs warped in sequences
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, container_abcs.Mapping):
        return {key: _collate_without_seqs([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_collate_without_seqs(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        return functools.reduce(lambda x, y: x+y, batch, type(elem)())
    else:
        return default_collate(batch)


def collate_fn(mode: Optional[str] = None):
    if mode is None:
        return default_collate
    elif mode == 'without_seqs':
        return _collate_without_seqs
    else:
        raise ValueError(f'Unknown mode = {mode}')


def get_dataloader_from_params(params: dict, 
                               name: str, 
                               base_seed: Optional[int] = None,
                               use_horovod: bool = False,
                               ):
    """ Construct dataloader based on config. """
    dataloader_config = params['dataloaders']
    if name in dataloader_config:
        dataset_params = dataloader_config[name]['dataset']
        dataloader_params = dataloader_config[name]['params']
        dataset_type = dataset_params.pop('type')
        fix_seed = dataset_params.pop('fix_seed', False)
        dset = getattr(datasets, dataset_type)(**dataset_params)
        if 'subset' in dataloader_config[name]:
            dset = Subset(dset,
                          indices=np.random.choice(len(dset),
                                                   size=dataloader_config[name]['subset']['n_samples'],
                                                   replace=False,
                                                   ),
                          )
        collate_fn_type = dataloader_params.pop('collate_fn_type', None)

        if isinstance(dset, JoinedDataset) and dataloader_params['shuffle']:
            dataloader_params['sampler'] = WeightedRandomSampler(weights=dset.sampling_weights,
                                                                 num_samples=len(dset),
                                                                 replacement=True)
            dataloader_params['shuffle'] = False
        horovod_kwargs = {}
        if use_horovod:
            num_replicas = hvd.size()
            rank = hvd.rank()
            shuffle = dataloader_params.get('shuffle', True)
            if 'sampler' not in dataloader_params:
                dataloader_params['sampler'] = DistributedSampler(dset,
                                                                  num_replicas=num_replicas,
                                                                  rank=rank,
                                                                  shuffle=shuffle,
                                                                  )
            else:
                dataloader_params['sampler'] = DistributedSamplerWrapper(dataloader_params['sampler'],
                                                                         num_replicas=num_replicas,
                                                                         rank=rank,
                                                                         shuffle=shuffle,
                                                                         )
            if (dataloader_params.get('num_workers', 0) > 0
                and hasattr(torchmp, '_supports_context')
                and torchmp._supports_context
                and 'forkserver' in torchmp.get_all_start_methods()
            ):
                horovod_kwargs['multiprocessing_context'] = 'forkserver'

        local_seed_freeze = functools.partial(seed_freeze, base_seed=base_seed, total_fix_seed=fix_seed)
        dloader = DataLoader(dset,
                             worker_init_fn=local_seed_freeze,
                             collate_fn=collate_fn(collate_fn_type),
                             **dataloader_params,
                             **horovod_kwargs,
                             )
        logger.debug(f'Created {name} dataloader with {dataset_type}. '
                     f'Length = {len(dset)} objects')
        return dloader
    return None
