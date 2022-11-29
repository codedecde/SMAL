from __future__ import absolute_import

import json
import logging
from typing import List, Optional, Tuple

import torch
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


@BaseAcquisitionFunction.register("sequence_tagging_simple_mnlp")
class SequenceTaggingSimpleMNLP(BaseAcquisitionFunction):
    @overrides
    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        """Implements the maximum normalized log probability acquisition function
        as presented in [Deep Active Learning for Named Entity Recognition](https://arxiv.org/pdf/1707.05928.pdf)
        However, compared to the CRF version, this one simply makes the local assumption that each token prediction
        is independent of the others.

        Parameters
        ----------
        instances : List[Instance]
            The instances to score

        Returns
        -------
        torch.Tensor
            The scores for each instance
        """
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
            # shape: (batch, seq_len)
            _, preds = logits.max(-1)
            # shape: (batch,)
            normalized_neg_log_probs = nn_util.sequence_cross_entropy_with_logits(
                logits, preds, mask, average=None
            )
            # Since the cross entropy loss computes the negative loss, we
            # combine the - in the end. Note that these scores are all negative and
            # averaged across tokens with more negative indicating a higher level of
            # uncertainty
            scores = -normalized_neg_log_probs
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
        metadata_list: List[JsonDict] = []
        for instance in new_train_data:
            instance_str = " ".join([token.text for token in instance["tokens"].tokens])
            metadata = {"tokens": instance_str}
            if "metadata" in instance and "lang" in instance["metadata"]:
                metadata["lang"] = instance["metadata"]["lang"]
            metadata_list.append(metadata)

        if file_path:
            with open(file_path, "w") as outfile:
                for elem in metadata_list:
                    outfile.write(json.dumps(elem) + "\n")

        train_data.extend(new_train_data)
        return train_data, unlabeled_data
