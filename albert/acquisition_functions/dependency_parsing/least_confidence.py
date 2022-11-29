from __future__ import absolute_import

import logging
from typing import Dict, List, Optional, Tuple

import torch
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
ALLOWED_NORMALIZATION = set(["tokens", "arcs"])


@BaseAcquisitionFunction.register("dependency_parsing_least_confidence")
class DependencyParsingLeastConfidence(BaseAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        budget: Budget,
        decode: Optional[str] = "greedy",
        normalize: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super(DependencyParsingLeastConfidence, self).__init__(
            model=model, budget=budget, batch_size=batch_size
        )
        self.decode = decode
        if normalize:
            assert normalize in ALLOWED_NORMALIZATION
        self.normalize = normalize
        assert self.decode in ALLOWED_DECODING

    def normalize_confidence(
        self, confidence: torch.Tensor, lengths: torch.LongTensor
    ) -> torch.Tensor:
        if self.normalize:
            if self.normalize == "tokens":
                return confidence / lengths.float()
            if self.normalize == "arcs":
                return confidence * 2 / (lengths * (lengths + 1)).float()
        return confidence

    def _extract_features(
        self, tokens: Dict[str, torch.Tensor], pos_tags: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        # shape: (batch, seq_len, embed_dim + pos_dim), shape: (batch, seq_len)
        embedded_text_input, mask = self.model.embeddings_from_tokens_and_pos(
            tokens, pos_tags
        )
        # shape: (batch, seq_len + 1, encoding_dim), shape: (batch, seq_len + 1)
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

            decoding = "mst" if self.model.use_mst_decoding_for_validation else "greedy"

            predicted_heads, predicted_head_tags = self.model.decoder.infer_tree(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=attended_arcs,
                mask=mask,
                decoding=decoding,
            )
            confidence_scores = self.model.decoder.likelihood(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=attended_arcs,
                mask=mask,
                head_indices=predicted_heads,
                head_tags=predicted_head_tags,
                decoding=self.decode,
            )
        lengths = mask.sum(-1).long() - 1
        return self.normalize_confidence(confidence_scores, lengths)

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


@BaseAcquisitionFunction.register("dependency_parsing_least_confidence_greedy")
class GreedyDependencyParsingLeastConfidence(DependencyParsingLeastConfidence):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(GreedyDependencyParsingLeastConfidence, self).__init__(
            model=model, budget=budget, decode="greedy", batch_size=batch_size
        )


@BaseAcquisitionFunction.register(
    "token_normalized_dependency_parsing_least_confidence_greedy"
)
class TokenNormalizedGreedyDependencyParsingLeastConfidence(
    DependencyParsingLeastConfidence
):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(TokenNormalizedGreedyDependencyParsingLeastConfidence, self).__init__(
            model=model,
            budget=budget,
            decode="greedy",
            normalize="tokens",
            batch_size=batch_size,
        )


@BaseAcquisitionFunction.register(
    "arcs_normalized_dependency_parsing_least_confidence_greedy"
)
class ArcsNormalizedGreedyDependencyParsingLeastConfidence(
    DependencyParsingLeastConfidence
):
    def __init__(
        self,
        model: Model,
        budget: Budget,
        normalize="tokens",
        batch_size: Optional[int] = None,
    ) -> None:
        super(ArcsNormalizedGreedyDependencyParsingLeastConfidence, self).__init__(
            model=model,
            budget=budget,
            decode="greedy",
            normalize="arcs",
            batch_size=batch_size,
        )


@BaseAcquisitionFunction.register("dependency_parsing_least_confidence_mst")
class MSTDependencyParsingLeastConfidence(DependencyParsingLeastConfidence):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(MSTDependencyParsingLeastConfidence, self).__init__(
            model=model, budget=budget, decode="mst", batch_size=batch_size
        )


@BaseAcquisitionFunction.register(
    "token_normalized_dependency_parsing_least_confidence_mst"
)
class TokenNormalizedMSTDependencyParsingLeastConfidence(
    DependencyParsingLeastConfidence
):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(TokenNormalizedMSTDependencyParsingLeastConfidence, self).__init__(
            model=model,
            budget=budget,
            decode="mst",
            normalize="tokens",
            batch_size=batch_size,
        )


@BaseAcquisitionFunction.register(
    "arcs_normalized_dependency_parsing_least_confidence_mst"
)
class ArcsNormalizedMSTDependencyParsingLeastConfidence(
    DependencyParsingLeastConfidence
):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(ArcsNormalizedMSTDependencyParsingLeastConfidence, self).__init__(
            model=model,
            budget=budget,
            decode="mst",
            normalize="arcs",
            batch_size=batch_size,
        )
