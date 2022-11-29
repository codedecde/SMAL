from __future__ import absolute_import

import datetime
import functools
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from allennlp.data import Instance

logger = logging.getLogger(__name__)
IndexList = Union[List[int], torch.LongTensor]


def setup(func) -> Callable[[Any], Tuple[List[Instance], List[Instance]]]:
    """A decorator for acquisition functions, that measures the amount of time
    taken for acquisition, and also logs basic statistics abount how many instances
    were acquired, and how many samples are left in the un-annotated pool. This also
    sets the model to eval model

    Args:
        func (Callable[[Any], Any]):
            An acquisition function. This assumes the first argument is a
            `self' argument, that has a model instance, which is used for scoring


    Returns:
        Callable[[Any], Tuple[List[Instance], List[Instance]]]:
            The wrapped function for logging statistics
    """

    @functools.wraps(func)
    def wrapper_function(self, *args, **kwargs):
        self.model.eval()
        logger.info("Starting selection")
        start_time = time.time()
        train_data, unlabeled_data = func(self, *args, **kwargs)
        time_elapsed = time.time() - start_time
        formatted_time = str(datetime.timedelta(seconds=int(time_elapsed)))
        logger.info(f"Selection Finished. Took {formatted_time}s")
        logger.info(f"Training  Instances: {len(train_data)}")
        logger.info(f"Unlabeled Instances: {len(unlabeled_data)}")
        logger.info(
            f"Frac Selected: {len(train_data) / (len(train_data) + len(unlabeled_data)):.2f}"
        )
        return train_data, unlabeled_data

    return wrapper_function


def sort_scores_by_lang(
    scores: torch.Tensor, unlabeled_data: List[Instance]
) -> Tuple[Dict[str, List[Instance]], Dict[str, IndexList]]:
    """Given scores for instances, and the instances themselves,
    partitions the instances by language, sorts them and returns
    the sorted indices for each language.

    Args:
        scores (torch.Tensor):
            The acquisition scores for each instance
        unlabeled_data (List[Instance]):
            The corresponding instances

    Returns:
        Tuple[Dict[str, List[Instance]], Dict[str, IndexList]]:
            A dictionary mapping the languages to the subset of instances
            as well as language mapping to sorted indices

    """
    lang2unlabeled = defaultdict(list)
    lang2indices = defaultdict(list)
    lang2unlabeled_indices = defaultdict(list)
    for index, instance in enumerate(unlabeled_data):
        assert "metadata" in instance and "lang" in instance["metadata"]
        lang = instance["metadata"]["lang"]
        lang2unlabeled[lang].append(instance)
        lang2unlabeled_indices[lang].append(index)
    for lang, index_list in lang2unlabeled_indices.items():
        sub_scores = scores[index_list]
        _, indices = torch.sort(sub_scores)
        lang2indices[lang] = indices
    return lang2unlabeled, lang2indices
