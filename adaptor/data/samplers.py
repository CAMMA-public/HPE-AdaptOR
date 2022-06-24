# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

import itertools
import torch
from torch.utils.data.sampler import Sampler
from typing import Optional
from detectron2.utils import comm

# modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py
__all__ = ["WeightedTrainingSampler"]
class WeightedTrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """
    def __init__(
        self,
        size: int,
        weights: torch.Tensor,
        replacement: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self.weights = weights
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                #yield from torch.randperm(self._size, generator=g)
                yield from torch.multinomial(self.weights, self.num_samples, self.replacement, generator=g)
            else:
                yield from torch.arange(self._size)