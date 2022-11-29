from __future__ import absolute_import

from typing import Iterable, List

import numpy as np
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators import DataIterator


@DataIterator.register("random")
class RandomSampleIterator(DataIterator):
    @overrides
    def _create_batches(
        self, instances: List[Instance], shuffle: bool
    ) -> Iterable[Batch]:
        """
        Assumptions: All of the data is in memory
        """
        num_batches = self.get_num_batches(instances)
        for bix in range(num_batches):
            indices = self._generate_indices(bix, len(instances), shuffle)
            batch_instances = [instances[index] for index in indices]
            batch = Batch(batch_instances)
            yield batch

    def _generate_indices(self, curr_ix: int, total: int, shuffle: bool) -> List[int]:
        if shuffle:
            indices = np.random.randint(total, size=(self._batch_size)).tolist()
        else:
            indices = list(
                range(
                    curr_ix * self._batch_size,
                    min((curr_ix + 1) * self._batch_size, total),
                )
            )
        return indices
