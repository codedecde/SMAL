from __future__ import absolute_import

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

import allennlp.nn.util as nn_util
from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.acquisition_functions.util import setup
from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("sequence_tagging_simple_total_token_entropy")
class SequenceTaggingSimpleTotalTokenEntropy(BaseAcquisitionFunction):
    @overrides
    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        with torch.no_grad():
            batch = Batch(instances)
            batch.index_instances(self.model.vocab)
            tensor_dict = batch.as_tensor_dict()
            if "tags" in tensor_dict:
                tensor_dict.pop("tags")
            tensor_dict = nn_util.move_to_device(tensor_dict, self.get_cuda_device())
            output_dict = self.model(**tensor_dict)
            # shape: (batch, seq_len, num_classes)
            logits = output_dict["logits"]
            # shape: (batch, seq_len)
            mask = output_dict["mask"]
            # shape: (batch, seq_len, num_classes)
            probs = logits.softmax(-1)
            # shape: (batch, seq_len, num_classes)
            log_probs = F.log_softmax(logits, -1)
            # shape: (batch, seq_len)
            token_entropy = -(probs * log_probs).sum(-1)
            # shape: (batch,)
            sequence_entropy = (token_entropy * mask.float()).sum(-1)
            # Higher entropy -> more uncertain. Hence we add a -ve,
            # and sort ascending later to collect more uncertain examples
            scores = -sequence_entropy
        return scores

    @setup
    @overrides
    def select_new_data(
        self,
        train_data: List[Instance],
        unlabeled_data: List[Instance],
        file_path: Optional[str] = None,
    ) -> Tuple[List[Instance], List[Instance]]:
        scores: torch.Tensor = self.score_unlabeled_data(unlabeled_data)
        _, indices = torch.sort(scores)
        new_train_data, unlabeled_data = self.select_data_from_indices(
            unlabeled_data, indices
        )
        metadata: List[JsonDict] = [
            {"tokens": " ".join(token.text for token in instance["tokens"].tokens)}
            for instance in new_train_data
        ]
        if file_path:
            with open(file_path, "w") as outfile:
                for elem in metadata:
                    outfile.write(elem["tokens"] + "\n")

        train_data.extend(new_train_data)
        return train_data, unlabeled_data
