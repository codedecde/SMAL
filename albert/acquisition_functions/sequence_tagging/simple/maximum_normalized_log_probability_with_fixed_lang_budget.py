from __future__ import absolute_import

import json
import logging
from typing import List, Optional, Tuple

import torch
from overrides import overrides

from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.acquisition_functions.sequence_tagging.simple.mnlp import (
    SequenceTaggingSimpleMNLP,
)
from albert.acquisition_functions.util import setup, sort_scores_by_lang
from allennlp.common import JsonDict
from allennlp.data import Instance

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("simple_mnlp_with_fixed_lang_budget")
class SimpleMaximumNormalizedLogProbabilityWithFixedLangBudget(
    SequenceTaggingSimpleMNLP
):
    @setup
    @overrides
    def select_new_data(
        self,
        train_data: List[Instance],
        unlabeled_data: List[Instance],
        file_path: Optional[str] = None,
    ) -> Tuple[List[Instance], List[Instance]]:
        scores: torch.Tensor = self.score_unlabeled_data(unlabeled_data)
        lang_to_unlabeled, lang_to_indices = sort_scores_by_lang(scores, unlabeled_data)
        new_train_data, unlabeled_data = self.equal_budget_select_data_from_indices(
            lang_to_unlabeled, lang_to_indices
        )

        metadata_list: List[JsonDict] = []
        for instance in new_train_data:
            instance_str = " ".join([token.text for token in instance["tokens"].tokens])
            metadata = {"tokens": instance_str}
            metadata["lang"] = instance["metadata"]["lang"]
            metadata_list.append(metadata)

        if file_path:
            with open(file_path, "w") as outfile:
                for elem in metadata_list:
                    outfile.write(json.dumps(elem) + "\n")

        train_data.extend(new_train_data)
        return train_data, unlabeled_data
