from __future__ import absolute_import

import logging
from typing import List, Optional, Tuple

import torch
from overrides import overrides

import allennlp.nn.util as nn_util
from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.acquisition_functions.util import setup
from albert.nn.util import pad_sequences_to_length
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("sequence_tagging_crf_mnlp")
class SequenceTaggingCRFMNLP(BaseAcquisitionFunction):
    r"""Implements the maximum normalized log probability acquisition function
    as presented in [Deep Active Learning for Named Entity Recognition](https://arxiv.org/pdf/1707.05928.pdf)

    .. note:
        score = \frac{1}{n} argmax_{y_1 \cdots y_n} \mathbb{P}(y_1 \cdots y_n | x_1 \cdots x_n)
    """

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
            # shape: (batch, seq_len)
            predicted_tags: List[List[int]] = output_dict["tags"]
            predicted_tags = pad_sequences_to_length(predicted_tags, mask.size(1))
            predicted_tags_tensor = torch.LongTensor(predicted_tags)
            if self.get_cuda_device() >= 0:
                predicted_tags_tensor = predicted_tags_tensor.cuda(
                    self.get_cuda_device()
                )
            # shape: (batch,)
            log_numerator = self.model.crf._joint_likelihood(
                logits, predicted_tags_tensor, mask
            )  # pylint: disable=protected-access
            # shape: (batch,)
            log_denominator = (
                self.model.crf._input_likelihood(  # pylint: disable=protected-access
                    logits, mask
                )
            )

            normalized_log_prob = (log_numerator - log_denominator) / mask.sum(-1)
            # Note that these scores are all negative and
            # averaged across tokens with more negative indicating a higher level of uncertainty
            scores = normalized_log_prob
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
