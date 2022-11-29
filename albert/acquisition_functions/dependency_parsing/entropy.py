from __future__ import absolute_import

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

import allennlp.nn.util as nn_util
from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.acquisition_functions.util import setup
from albert.budgets import Budget
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.models import Model

logger = logging.getLogger(__name__)

ALLOWED_DECODING = set(["greedy", "mst"])


@BaseAcquisitionFunction.register("dependency_parsing_entropy")
class DependencyParsingEntropy(BaseAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        budget: Budget,
        decode: Optional[str] = "greedy",
        batch_size: Optional[int] = None,
    ) -> None:
        super(DependencyParsingEntropy, self).__init__(
            model=model, budget=budget, batch_size=batch_size
        )
        self.decode = decode
        assert self.decode in ALLOWED_DECODING

    def _extract_features(
        self, tokens: Dict[str, torch.Tensor], pos_tags: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        embedded_text_input, mask = self.model.embeddings_from_tokens_and_pos(
            tokens, pos_tags
        )
        encoded_text, mask = self.model.encode_text(embedded_text_input, mask)
        (
            head_tag_representation,
            child_tag_representation,
        ) = self.model.generate_head_tag_representations(encoded_text)
        attended_arcs = self.model.generate_arc_scores(encoded_text, mask)
        return head_tag_representation, child_tag_representation, attended_arcs, mask

    @overrides
    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        with torch.no_grad():
            batch = Batch(instances)
            batch.index_instances(self.model.vocab)
            tensor_dict = batch.as_tensor_dict()

            if "head_indices" in tensor_dict:
                tensor_dict.pop("head_indices")
            if "head_tags" in tensor_dict:
                tensor_dict.pop("head_tags")

            tensor_dict = nn_util.move_to_device(tensor_dict, self.get_cuda_device())
            tokens = tensor_dict["tokens"]
            pos = tensor_dict["pos_tags"]
            (
                head_tag_representation,
                child_tag_representation,
                attended_arcs,
                mask,
            ) = self._extract_features(tokens, pos)

            confidence_scores = self.model.decoder.entropy(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=attended_arcs,
                mask=mask,
                decoding=self.decode,
            )

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
