from __future__ import absolute_import

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

import albert.trainers.util as trainer_util
from albert.trainers import BaselineActiveLearningTrainer
from albert.utils import (
    bool_flag,
    fine_tune,
    get_fine_tune_args,
    get_train_arguments,
    setup_logger,
)
from albert.utils.common_utils import (
    get_predictor_from_dir,
    set_random_seed,
    setup_output_dir,
)
from allennlp.common import Params, Tqdm
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.models import Model

logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser(description="Trainer/Tester")
    parser.add_argument(
        "-t", "--train", action="store", dest="train", type=bool_flag, default=True
    )
    parser.add_argument(
        "-f", "--fine_tune", action="store_true", dest="fine-tune", default=False
    )
    all_args = parser.parse_known_args(sys.argv[1:])
    return vars(all_args[0])


def get_test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bd", "--base_dir", action="store", dest="base_dir", required=True
    )
    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        dest="infile",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-o", "--outfile", action="store", dest="outfile", default=None, required=False
    )
    parser.add_argument(
        "-c", "--cuda", action="store", dest="cuda", type=bool_flag, default=False
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
    config = Params.from_file(args["config_file"])
    if args["seed"] > 0:
        set_random_seed(args["seed"])
    # Setup the experiment directory
    if args["serialization_dir"] is None:
        serialization_dir = setup_output_dir(config, args["loglevel"])
        args["recover"] = False
    else:
        serialization_dir = args["serialization_dir"]

    # we load the dataset reader
    reader = DatasetReader.from_params(config.pop("dataset_reader"))
    # The reason we want two readers is that in case of truncations and large tokens
    # we want them to be accounted for during testing, but not for training and
    # validation purposes.
    test_reader_params = config.pop("test_dataset_reader", None)
    test_reader = (
        DatasetReader.from_params(test_reader_params) if test_reader_params else reader
    )

    train_file = config.pop("train_data_path")
    validation_file = config.pop("validation_data_path")
    test_param = config.pop("test_data_path", None)
    unlabeled_file = config.pop("unlabeled_data_path")

    train_data = reader.read(train_file)
    validation_data = reader.read(validation_file)
    unlabeled_data = reader.read(unlabeled_file)
    test_data = trainer_util.get_test_data(test_param, test_reader)

    vocab_params = config.pop("vocabulary", {})

    vocab = Vocabulary.from_params(
        params=vocab_params, instances=train_data + validation_data + unlabeled_data
    )
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    model = Model.from_params(vocab=vocab, params=config.pop("model"))

    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    test_iterator_params = config.pop("test_iterator", None)
    test_iterator = (
        DataIterator.from_params(test_iterator_params)
        if test_iterator_params
        else iterator
    )
    test_iterator.index_with(model.vocab)

    trainer_params = config.pop("trainer_params")
    config.assert_empty("Config")

    trainer = BaselineActiveLearningTrainer.from_params(
        model=model,
        serialization_dir=serialization_dir,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        unlabeled_data=unlabeled_data,
        batcher=iterator,
        test_batcher=test_iterator,
        trainer_params=trainer_params,
    )
    trainer.train()
    logger.info("Done.")


def test(args: Dict[str, Any]):
    # pylint: disable=protected-access
    """This is the testing function. Usually, different models
    may have different testing behaviour, depending on what they
    are trying to achieve

    Parameters:
        args (Dict[str, Any]): A dictionary consisting of the following
            keys:
                * base_dir (str): The base model directory to load
                    the model from
                * infile (str): The file to load the test data from
    """
    setup_logger()
    base_dir = args["base_dir"]

    device = 0 if args["cuda"] else -1
    predictor = get_predictor_from_dir(base_dir, device)
    reader, model = predictor._dataset_reader, predictor._model

    if args["infile"]:
        instances = reader.read(args["infile"])

        batch_size = 32
        predictions = []
        for ix in Tqdm.tqdm(range(0, len(instances), batch_size)):
            batch = instances[ix : min(len(instances), ix + batch_size)]
            prediction = predictor.predict_batch_instance(batch)
            predictions += prediction
        metrics = model.get_metrics(reset=True)
        for metric in metrics:
            metric_val = metrics[metric]
            logger.info(f"{metric}: {metric_val:0.2f}")

    outfile = args["outfile"]
    if outfile is not None:
        assert args["infile"], "Need an input file to generate outputs"
        with open(outfile, "w") as f:
            json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    common_args = get_arguments()
    if common_args["fine-tune"]:
        fine_tune_args = get_fine_tune_args()
        fine_tune(fine_tune_args)
    elif common_args["train"]:
        train_args = get_train_arguments()
        train(train_args)
    else:
        test_args = get_test_arguments()
        test(test_args)
