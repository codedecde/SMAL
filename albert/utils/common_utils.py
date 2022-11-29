from __future__ import absolute_import

import argparse
import io
import json
import logging
import os
import shutil
import sys
from subprocess import PIPE, Popen
from typing import Optional

import numpy as np
import torch

import allennlp.nn.util as nn_util
from albert.utils.path_utils import append_expdir_to_paths
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Vocabulary
from allennlp.models import Model
from allennlp.models.model import remove_pretrained_embedding_params
from allennlp.predictors import Predictor

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def bool_flag(s: str) -> bool:
    """
    Parse boolean arguments from the command line.
    ..note::
    Usage in argparse:
        parser.add_argument(
            "--cuda", type=bool_flag, default=True, help="Run on GPU")
    """
    if s.lower() in ["off", "false", "0"]:
        return False
    if s.lower() in ["on", "true", "1"]:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def setup_logger(logfile: str = "", loglevel: str = "INFO") -> logging.RootLogger:
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )
    fmt = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    if logfile != "":
        logfile_handle = logging.FileHandler(logfile, "w")
        logfile_handle.setFormatter(fmt)
        logger.addHandler(logfile_handle)
    return logger


def setup_output_dir(config: Params, loglevel: str) -> str:
    """Setup the Experiment Folder
    Note that the output_dir stores each run as run-1, ....
    Makes the next run directory. This also sets up the logger
    A run directory has the following structure
    - run-1
        - models
                * modelname*.tar.gz
                - vocabulary
                    * namespace_1.txt
                    * namespace_2.txt ...
        * config.json
        * githash.log of current run
        * gitdiff.log of current run
        * logfile.log (the log of the current run)
    Arguments:
        config (``allennlp.common.Params``): The experiment parameters
        loglevel (str): The logger mode [INFO/DEBUG/ERROR]
    Returns
        str, allennlp.common.Params: The filename, and the modified config
    """
    output_dir = config.get("base_output_dir", "./Outputs")
    make_directory(output_dir)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith("run-"):
            last_run = max(last_run, int(dirname.split("-")[1]))
    new_dirname = os.path.join(output_dir, "run-%d" % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, "models")
    make_directory(best_model_dirname)
    vocab_dirname = os.path.join(best_model_dirname, "vocabulary")
    make_directory(vocab_dirname)

    config_file = os.path.join(new_dirname, "config.json")
    write_config_to_file(config_file, config)

    # Save the git hash
    process = Popen('git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    stdout = stdout.decode("ascii").strip("\n").strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)

    # Save the git diff
    process = Popen("git diff".split(), stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode("ascii", errors="ignore")
        fp.write(stdout)

    # Set up the logger
    logfile = os.path.join(new_dirname, "logfile.log")
    setup_logger(logfile, loglevel)
    return best_model_dirname


def make_directory(dirname: str) -> None:
    """Constructs a directory with name dirname, if
    it doesn't exist. Can also take in a path, and recursively
    apply it.
    """
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        raise RuntimeError(
            "Race condition. " "Two methods trying to create to same place"
        )


def read_from_config_file(filepath: str, params_overrides: str = "") -> Params:
    """Read Parameters from a config file
    Arguments:
        filepath (str): The file to read the
            config from
        params_overrides (str): Overriding the config
            Can potentially be used for command line args
            e.g. '{"model.embedding_dim": 10}'
    Returns:
        allennlp.common.Params: The parameters
    """
    return Params.from_file(params_file=filepath, params_overrides=params_overrides)


def write_config_to_file(filepath: str, config: Params) -> None:
    """Writes the config to a json file, specifed by filepath"""
    with io.open(filepath, "w", encoding="utf-8", errors="ignore") as fd:
        json.dump(
            fp=fd,
            obj=config.as_dict(quiet=True),
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )


def get_model_from_baseline_runs(
    model_path: str,
    config: Params,
    iteration: Optional[int],
    cuda_device: Optional[float] = -1,
) -> Model:
    vocab_path = os.path.join(model_path, "vocabulary")
    vocab_params = config.get("vocabulary", Params({}))
    vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
    vocab = Vocabulary.by_name(vocab_choice).from_files(
        vocab_path,
        vocab_params.get("padding_token", None),
        vocab_params.get("oov_token", None),
    )

    model_params = config.get("model")
    remove_pretrained_embedding_params(model_params)
    model = Model.from_params(vocab=vocab, params=model_params)
    model.extend_embedder_vocab()
    weights_file = os.path.join(model_path, f"iteration-{iteration}", "best.th")

    model_state = torch.load(
        weights_file, map_location=nn_util.device_mapping(cuda_device)
    )
    model_weights = model_state["model"]
    model.load_state_dict(model_weights)
    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()
    return model


def get_predictor_from_dir(
    base_dir: str,
    predictor_params: Optional[Params] = None,
    cuda_device: int = -1,
    iteration: int = -1,
    code_dir: Optional[str] = None,
) -> Predictor:
    assert os.path.exists(base_dir), f"{base_dir} not found"
    config = read_from_config_file(os.path.join(base_dir, "config.json"))
    if code_dir:
        config = append_expdir_to_paths(config, code_dir)
    model_path = os.path.join(base_dir, "models")
    if iteration >= -1:
        model = get_model_from_baseline_runs(model_path, config, iteration, cuda_device)
    else:
        model = Model.load(config.duplicate(), model_path, cuda_device=cuda_device)

    model.eval()
    reader = DatasetReader.from_params(config.pop("dataset_reader"))
    if iteration >= -1:
        return Predictor(model=model, dataset_reader=reader)
    predictor_params = predictor_params or Params({"type": config["model"]["type"]})
    try:
        predictor = Predictor.from_params(
            params=predictor_params.duplicate(), model=model, dataset_reader=reader
        )
    except ConfigurationError as e:  # pylint: disable=unused-variable
        predictor_name = predictor_params["type"]
        logger.warning(f"{predictor_name} not found in list of registered classes")
        predictor = Predictor(model=model, dataset_reader=reader)
    return predictor
