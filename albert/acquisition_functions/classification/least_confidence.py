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
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@BaseAcquisitionFunction.register("classification_least_confidence")
class ClassificationLeastConfidence(BaseAcquisitionFunction):
    @overrides
    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        """Scores instances by their least confidence.

        Parameters
        ----------
        instances : List[Instance]
            The list of instances to score.

        Returns
        -------
        torch.Tensor
            The scores for each instance.
        """
        loss = torch.nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            batch = Batch(instances)
            batch.index_instances(self.model.vocab)
            tensor_dict = batch.as_tensor_dict()
            if "label" in tensor_dict:
                # we don't have label while making the active learning prediction
                tensor_dict.pop("label")
            tensor_dict = nn_util.move_to_device(tensor_dict, self.get_cuda_device())
            output_dict = self.model(**tensor_dict)
            # shape: (batch, num_classes)
            logits = output_dict["logits"]
            # shape: (batch,)
            _, preds = logits.max(-1)
            # shape: (batch,)
            scores = -(loss(logits, preds))
        return scores

    @setup
    @overrides
    def select_new_data(
        self,
        train_data: List[Instance],
        unlabeled_data: List[Instance],
        file_path: Optional[str] = None,
    ) -> Tuple[List[Instance], List[Instance]]:
        """Selects instances that the model is least confident on.

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
