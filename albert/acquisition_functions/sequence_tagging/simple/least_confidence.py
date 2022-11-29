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
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("sequence_tagging_simple_least_confidence")
class SequenceTaggingSimpleLeastConfidence(BaseAcquisitionFunction):
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
            # shape: (batch, seq_len, num_tags), (batch, seq_len)
            log_probs, _ = F.log_softmax(logits, -1).max(-1)
            # shape: (batch,)
            confidence_scores = (log_probs * mask.float()).sum(-1)
        return confidence_scores

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
        new_train_data, new_unlabeled_data = self.select_data_from_indices(
            unlabeled_data, indices
        )

        metadata: List[JsonDict] = []
        for instance in new_train_data:
            instance_str = " ".join([token.text for token in instance["tokens"]])
            metadata.append({"tokens": instance_str})

        if file_path:
            with open(file_path, "w") as outfile:
                for elem in metadata:
                    outfile.write(elem["tokens"] + "\n")

        train_data.extend(new_train_data)
        unlabeled_data = new_unlabeled_data
        return train_data, unlabeled_data
