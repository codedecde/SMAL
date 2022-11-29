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


@BaseAcquisitionFunction.register("sequence_tagging_simple_multilingual_mnlp")
class SequenceTaggingSimpleMultilingualMNLP(BaseAcquisitionFunction):
    @overrides
    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        """
        This model generates a probability at a per language level, and is trained in a
        multi-headed fashion. The scoring is also done at a per language level, based on the
        lang_mask passed.
        """
        with torch.no_grad():
            batch = Batch(instances)
            batch.index_instances(self.model.vocab)
            tensor_dict = batch.as_tensor_dict()
            if "label" in tensor_dict:
                # we don't have label while making the active learning prediction
                tensor_dict.pop("label")
            tensor_dict = nn_util.move_to_device(tensor_dict, self.get_cuda_device())
            output_dict = self.model(**tensor_dict)
            normalized_neg_log_probs = 0
            lang_mask = output_dict["lang_mask"].float()
            for lang_index in range(self.model.num_langs):
                mask = lang_mask[:, lang_index]
                # shape: (batch, seq_len, num_classes)
                logits = output_dict["logits"][lang_index]
                # shape: (batch, seq_len)
                sequence_mask = output_dict["mask"]
                # shape: (batch, seq_len)
                _, preds = logits.max(-1)
                # shape: (batch,)
                lang_normalized_neg_log_probs = (
                    nn_util.sequence_cross_entropy_with_logits(
                        logits, preds, sequence_mask, average=None
                    )
                )
                normalized_neg_log_probs += mask * lang_normalized_neg_log_probs
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
