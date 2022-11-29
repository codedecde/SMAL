from __future__ import absolute_import

import logging
from typing import Dict, List, Optional, Tuple

import torch

import allennlp.nn.util as nn_util
from albert.acquisition_functions.util import IndexList, setup
from albert.budgets import Budget
from allennlp.common import Registrable
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.models import Model

logger = logging.getLogger(__name__)


class BaseAcquisitionFunction(Registrable):
    def __init__(
        self, model: Model, budget: Budget, batch_size: Optional[int] = None
    ) -> None:
        super(BaseAcquisitionFunction, self).__init__()
        self.model = model
        self.budget = budget
        self._batch_size = batch_size or 32

    def get_cuda_device(self) -> int:
        return self.model._get_prediction_device()  # pylint: disable=protected-access

    @setup
    def select_new_data(
        self,
        train_data: List[Instance],
        unlabeled_data: List[Instance],
        file_path: Optional[str] = None,
    ) -> Tuple[List[Instance], List[Instance]]:
        raise NotImplementedError("Child class implements this")

    def select_data_from_indices(
        self, unlabeled_data: List[Instance], indices: IndexList
    ) -> Tuple[List[Instance], List[Instance]]:
        """Select data based on indices. Indices here refers to the indices in the
        unlabeled data to be added until the budget runs out. Everything else is then
        returned as unlabeled data

        Parameters
        ----------
        unlabeled_data : List[Instance]
            The unlabeled data
        indices : IndexList
            The indices to be selected until the budget is met.

        Returns
        -------
        Tuple[List[Instance], List[Instance]]
            The selected indices as well as the unlabeled data not selected for annotation
        """
        new_train: List[Instance] = []
        new_unlabeled: List[Instance] = []
        self.budget.reset()
        for index in indices:
            instance = unlabeled_data[index]
            if self.budget.can_add_instance(instance):
                new_train.append(instance)
            else:
                new_unlabeled.append(instance)
        return new_train, new_unlabeled

    def equal_budget_select_data_from_indices(
        self,
        lang_to_unlabeled: Dict[str, List[Instance]],
        lang_to_indices: Dict[str, IndexList],
    ) -> Tuple[List[Instance], List[Instance]]:
        """Select data based in indices *per language*.
        The assumption here is that each language scores its instances separately,
        and we split the budget equally between the languages for selection purposes.
        We then select for each language until the language budget is met, and finally
        combine all instances to return the selected data.

        Parameters
        ----------
        lang_to_unlabeled : Dict[str, List[Instance]]
            The unlabeled data, separated by lang. The indices of
            lang_to_indices should correspond to their respective
            unlabeled data points
        lang_to_indices : Dict[str, IndexList]
            The mapping of indices to be selected for each language
            to the actual indices. These indices should be sorted so that
            the top corresponds to the unlabeled data to be selected

        Returns
        -------
        Tuple[List[Instance], List[Instance]]
            The selected indices as well as the unlabeled data not selected for
            annotation
        """
        # First get the total budget
        # so that we can compute the per language budget
        total_budget = self.budget.get_budget_size()
        num_langs = len(lang_to_indices)
        per_lang_budget = [total_budget // num_langs] * num_langs
        # if we have some excess left over, arbitrarily assign it to the last language
        new_train_data = []
        new_unlabeled_data = []
        per_lang_budget[-1] += total_budget % num_langs
        for lang_budget, lang in zip(per_lang_budget, lang_to_indices):
            indices = lang_to_indices[lang]
            unlabeled_data = lang_to_unlabeled[lang]
            self.budget.set_budget_size(lang_budget)
            lang_train_data, lang_unlabeled_data = self.select_data_from_indices(
                unlabeled_data, indices
            )
            new_train_data.extend(lang_train_data)
            new_unlabeled_data.extend(lang_unlabeled_data)
        self.budget.set_budget_size(total_budget)
        return new_train_data, new_unlabeled_data

    def score_instance(self, instance: Instance) -> torch.Tensor:
        return self.score_instances([instance])[0]

    def score_instances(self, instances: List[Instance]) -> torch.Tensor:
        raise NotImplementedError("Child class implements this")

    def score_unlabeled_data(self, unlabeled_data: List[Instance]) -> torch.Tensor:
        scores: torch.Tensor = torch.zeros(len(unlabeled_data))
        scores = nn_util.move_to_device(scores, self.get_cuda_device())
        batch_size = self._batch_size
        for index in Tqdm.tqdm(range(0, len(unlabeled_data), batch_size)):
            batch_instances = unlabeled_data[index : index + batch_size]
            batch_scores = self.score_instances(batch_instances)
            scores[index : index + batch_scores.size(0)] = self.score_instances(
                batch_instances
            )
        return scores
