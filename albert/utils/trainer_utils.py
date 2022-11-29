"""
General training utilities
"""
from __future__ import absolute_import

import argparse
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

import allennlp.nn.util as nn_util
from albert.utils.common_utils import (
    read_from_config_file,
    set_random_seed,
    setup_output_dir,
)
from allennlp.commands.fine_tune import fine_tune_model
from allennlp.common import Params, Tqdm
from allennlp.common.util import sanitize
from allennlp.data import DataIterator
from allennlp.models import archive_model, load_archive
from allennlp.training import Trainer
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_pieces import TrainerPieces

logger = logging.getLogger(__name__)


def create_optimizer(
    optimizer_params: Params, model: nn.Module
) -> torch.optim.Optimizer:
    optimizer_params = optimizer_params.duplicate() if optimizer_params else None
    model_parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = Optimizer.from_params(model_parameters, optimizer_params)
    return optimizer


def sample_from_iterator(
    iterator: DataIterator, cuda_device: int
) -> Dict[str, torch.Tensor]:
    tensor_dict = next(iterator)
    tensor_dict = nn_util.move_to_device(tensor_dict, cuda_device)
    return tensor_dict


def get_train_arguments() -> Dict[str, Any]:
    """Get arguments necessary for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--config_file",
        action="store",
        dest="config_file",
        type=str,
        help="path to config file",
        required=False,
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
        "-d",
        "--serialization_dir",
        action="store",
        dest="serialization_dir",
        type=str,
        help="serialization directory to load",
    )
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        dest="recover",
        default=False,
        help="Recover previous model",
    )
    all_args = parser.parse_known_args(sys.argv[1:])
    return vars(all_args[0])


def get_fine_tune_args() -> Dict[str, Any]:
    """
    Gets the arguments for fine-tuning
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--config_file",
        action="store",
        dest="config_file",
        type=str,
        help="path to config file",
        required=False,
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
        "-md",
        "--model_dir",
        action="store",
        dest="model-dir",
        type=str,
        help="model directory to load",
    )
    all_args = parser.parse_known_args(sys.argv[1:])
    return vars(all_args[0])


def train(args: Dict[str, Any]) -> None:
    """This is the main training function. Usually, for most models,
    this will not change

    Parameters:
        args (Dict): A dictionary of the following format:
            {
                config_file (str): the path to the configuration file
                seed (int): The random seed
                loglevel (str): The loglevel (Default: INFO)
            }
    """
    config = read_from_config_file(args["config_file"])
    if args["seed"] > 0:
        set_random_seed(args["seed"])
    # Setup the experiment directory
    if args["serialization_dir"] is None:
        serialization_dir = setup_output_dir(config, args["loglevel"])
        args["recover"] = False
    else:
        serialization_dir = args["serialization_dir"]
    training_pieces = TrainerPieces.from_params(
        config, serialization_dir=serialization_dir, recover=args["recover"]
    )
    trainer = Trainer.from_params(
        model=training_pieces.model,
        serialization_dir=serialization_dir,
        iterator=training_pieces.iterator,
        train_data=training_pieces.train_dataset,
        validation_data=training_pieces.validation_dataset,
        params=training_pieces.params,
    )
    trainer.train()
    logger.info("Done.")


def fine_tune(args: Dict[str, Any]) -> None:
    """
    To fine tune a trained model on different data
    Params:
          Dictionary of the type:
          {
              config_file: Path to the JSONNET file.
              seed       : Random seed.
              loglevel   : Logging level
              model-dir  : Model directory.
         }
    """
    # Make the model into an archive
    archive_model(args["model-dir"])  # Usually of the form /..../run-xx/models
    # Get the running directory
    run_dir, _ = os.path.split(args["model-dir"])

    # Read the archive
    model = load_archive(os.path.join(args["model-dir"], "model.tar.gz"))

    # Read config for fine-tuning
    params = read_from_config_file(args["config_file"])

    # Insert the finetuned at the end.
    # This turns run-30 to run-30-finetuned
    new_dir = f"{run_dir}-finetuned"

    # AllenNLP finetuner serializes models into one folder.
    serialization_dir = os.path.join(new_dir, "models")

    logger.info("Fine-Tuning Model from {0}".format(args["model-dir"]))
    fine_tune_model(model.model, params, serialization_dir)

    # copy the config into the parent folder to conform with trainer output
    shutil.copy(os.path.join(serialization_dir, "config.json"), new_dir)
