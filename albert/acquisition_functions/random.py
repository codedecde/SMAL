from __future__ import absolute_import

import datetime
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from overrides import overrides

from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.acquisition_functions.util import setup
from allennlp.common import JsonDict
from allennlp.data import Instance

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("random")
class RandomAcquisitionFunction(BaseAcquisitionFunction):
    @setup
    @overrides
    def select_new_data(
        self,
        train_data: List[Instance],
        unlabeled_data: List[Instance],
        file_path: Optional[str] = None,
    ) -> Tuple[List[Instance], List[Instance]]:
        """Randomly selects data for acquisition. This is predominantly used as a baseline.

        Parameters
        ----------
        train_data : List[Instance]
            The training data to add instances to
        unlabeled_data : List[Instance]
            The unlabeled data to add instances from
        file_path : Optional[str], optional
            If specified, also writes the acquired instances.
            This is primarily useful for debugging purposes. By default None

        Returns
        -------
        Tuple[List[Instance], List[Instance]]
            The training data, with the newly acquired instances added, and the unlabeled data
        """
        # pylint: disable=unused-argument
        index_list = np.random.permutation(len(unlabeled_data)).tolist()
        new_train, new_unlabeled = self.select_data_from_indices(
            unlabeled_data, index_list
        )
        metadata: List[JsonDict] = []
        for instance in new_train:
            instance_str = " ".join([token.text for token in instance["tokens"].tokens])
            metadata.append({"tokens": instance_str})
            train_data.append(instance)
        unlabeled_data = new_unlabeled

        if file_path:
            with open(file_path, "w") as outfile:
                for elem in metadata:
                    outfile.write(elem["tokens"] + "\n")

        return train_data, unlabeled_data
