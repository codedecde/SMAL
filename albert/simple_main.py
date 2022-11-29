from __future__ import absolute_import

import argparse
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict

from albert.utils.common_utils import set_random_seed, setup_logger, setup_output_dir
from allennlp.common import Params
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.util import evaluate

logger = logging.getLogger(__name__)


def get_train_arguments() -> argparse.Namespace:
    """Get arguments necessary for training.
    We support two types: either start from scratch (in which case you need to specify
    the config file) or from an existing director (in which case we simply carry on from
    where we left off)

    """
    parser = argparse.ArgumentParser()
    # Basic indicators about logging
    parser.add_argument(
        "-s",
        "--seed",
        action="store",
        dest="seed",
        type=int,
        default=-1,
        help="use fixed random seed",
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store",
        dest="loglevel",
        type=str,
        default="INFO",
        help="Logging Level",
        required=False,
    )
    # Now either we specify a config file
    parser.add_argument(
        "-cf",
        "--config_file",
        action="store",
        dest="config_file",
        type=str,
        default=None,
        help="path to config file",
        required=False,
    )
    # Or we specify an existing directory to start from
    parser.add_argument(
        "-ed",
        "--exp_dir",
        action="store",
        dest="exp_dir",
        type=str,
        help="serialization directory to load",
        default=None,
    )
    # and the overrides config path
    parser.add_argument(
        "-o",
        "--overrides",
        action="store",
        dest="overrides",
        type=str,
        help="the overrides config",
        default=None,
    )
    all_args, _ = parser.parse_known_args(sys.argv[1:])
    if all_args.config_file is None and all_args.exp_dir is None:
        raise RuntimeError(
            "We need either the config_file or the serialized dir to work with"
        )
    return all_args


def train(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    """This is the main training loop. We support
    two kinds of utilities: either start from scratch (in
    which case, we will create a new serialization dir etc),
    or start from an existing run (in which case we need the
    serialization dir ``the run-X`` folder) to be specified)
    """
    if args.seed > 0:
        set_random_seed(args.seed)

    if args.config_file:
        config = Params.from_file(args.config_file)

        # Setup the experiment directory
        serialization_dir = setup_output_dir(config, args.loglevel)
        pieces = TrainerPieces.from_params(
            config, serialization_dir=serialization_dir  # pylint: disable=no-member
        )
        trainer = Trainer.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.train_dataset,
            validation_data=pieces.validation_dataset,
            params=pieces.params,
        )
    else:
        experiment_dir = args.exp_dir
        overrides_file = args.overrides
        if overrides_file:
            overrides_params_str = json.dumps(
                Params.from_file(overrides_file).as_ordered_dict()
            )
        else:
            overrides_params_str = ""
        assert os.path.exists(
            experiment_dir
        ), f"Experiment dir {experiment_dir} not found"
        # we now create a new folder in order to log the continued experiment
        # the old experiment dir would usually be of the form {PATH}/run-XX.
        # We want the new directory to be of the form {PATH}/run-XX-continued
        base_dir, run_dir = os.path.split(experiment_dir)

        new_experiment_dir = os.path.join(base_dir, f"{run_dir}-continued")
        os.makedirs(new_experiment_dir, exist_ok=True)

        # setup logging
        logger_file = os.path.join(new_experiment_dir, "logfile.log")
        setup_logger(logger_file, args.loglevel)

        config_file = os.path.join(experiment_dir, "config.json")
        config = Params.from_file(config_file, params_overrides=overrides_params_str)

        config.to_file(os.path.join(new_experiment_dir, "config.json"))

        # setup the new serialization dir
        serialization_dir = os.path.join(experiment_dir, "models")
        new_serialization_dir = os.path.join(new_experiment_dir, "models")
        os.makedirs(new_serialization_dir, exist_ok=True)

        # Copy over the old serialization model
        # note that we just copy the latest file, not the best one
        checkpointer = Checkpointer(serialization_dir=serialization_dir)
        model_path, training_state_path = checkpointer.find_latest_checkpoint()
        _, model_path_name = os.path.split(model_path)
        _, training_state_path_name = os.path.split(training_state_path)
        new_model_path = os.path.join(new_serialization_dir, model_path_name)
        new_training_state_path = os.path.join(
            new_serialization_dir, training_state_path_name
        )
        shutil.copyfile(src=model_path, dst=new_model_path)
        shutil.copyfile(src=training_state_path, dst=new_training_state_path)

        # copy the vocabulary
        vocab_path = os.path.join(experiment_dir, "models", "vocabulary")
        new_vocab_path = os.path.join(new_serialization_dir, "vocabulary")
        shutil.copytree(src=vocab_path, dst=new_vocab_path)

        pieces = TrainerPieces.from_params(
            params=config,  # pylint: disable=no-member
            serialization_dir=new_serialization_dir,
            recover=True,
        )

        trainer = Trainer.from_params(
            model=pieces.model,
            serialization_dir=new_serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.train_dataset,
            validation_data=pieces.validation_dataset,
            params=pieces.params,
        )
    # start training
    trainer.train()
    # this is the trained model
    model = trainer.model
    if pieces.test_dataset is not None:
        logger.info("Computing Test Accuracy")
        data_iterator = BasicIterator()
        data_iterator.index_with(model.vocab)
        test_metrics = evaluate(
            model=model,
            instances=pieces.test_dataset,
            data_iterator=data_iterator,
            cuda_device=model._get_prediction_device(),  # pylint: disable=protected-access
            batch_weight_key=None,
        )
        for key in test_metrics:
            logger.info("Testing: {0} - {1:2.2f}".format(key, test_metrics[key]))
    logger.info("Done.")


if __name__ == "__main__":
    train_args = get_train_arguments()
    train(train_args)
