from __future__ import absolute_import

import argparse

import albert  # pylint: disable=unused-import
import albert.baseline_active_learning_main as trainer_module
import allennlp
from albert.utils import get_train_arguments


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("The Run script")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    trainer_module.train(args)
