from __future__ import absolute_import

import argparse
import logging
import sys

import allennlp.training.util as training_util
from albert.utils.common_utils import bool_flag, get_predictor_from_dir, setup_logger
from allennlp.data.iterators import BasicIterator

logger = logging.getLogger(__name__)


def get_evaluation_arguments():
    """Get arguments necessary for evaluating.


    arguments:

        exp_dir: str
            The expriment directory to load the model from
        infile: str
            The file to compute metrics for
        cuda: bool, optional (default = ``False``)
            If true, use the GPU

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ed", "--exp_dir", action="store", dest="exp_dir", required=True
    )
    parser.add_argument(
        "--iteration",
        action="store",
        dest="iteration",
        type=int,
        required=False,
        default=-1,
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
        "-c", "--cuda", action="store", dest="cuda", type=bool_flag, default=False
    )
    all_args, _ = parser.parse_known_args(sys.argv[1:])
    return all_args


def evaluate(args: argparse.Namespace):
    # pylint: disable=protected-access
    """This is the testing function"""
    setup_logger()
    exp_dir = args.exp_dir

    device = 0 if args.cuda else -1
    predictor = get_predictor_from_dir(
        base_dir=exp_dir, iteration=args.iteration, cuda_device=device
    )
    reader, model = (
        predictor._dataset_reader,
        predictor._model,
    )  # pylint: disable=protected-access

    if args.infile:
        instances = reader.read(args.infile)
        data_iterator = BasicIterator()
        data_iterator.index_with(model.vocab)
        test_metrics = training_util.evaluate(
            model=model,
            instances=instances,
            data_iterator=data_iterator,
            cuda_device=device,
            batch_weight_key=None,
        )
        for key in test_metrics:
            logger.info("Testing: {0} - {1:2.4f}".format(key, test_metrics[key]))


if __name__ == "__main__":
    args = get_evaluation_arguments()
    evaluate(args)
