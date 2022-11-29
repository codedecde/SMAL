from __future__ import absolute_import

import argparse
import glob
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple

import conllu
import tqdm

logger = logging.getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert Dependency Parser Dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        dest="input_dir",
        required=True,
        help="The root directory with the different dependency parses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        dest="output_dir",
        required=True,
        help="The root directory to write the parsed info to",
    )
    args = parser.parse_args()
    return args


def parse_conllu_annotation(annotation: conllu.models.TokenList, lang: str) -> None:
    for token_info in annotation:
        if token_info["misc"] is None:
            token_info["misc"] = OrderedDict({"lang": lang})
        else:
            assert isinstance(token_info["misc"], OrderedDict)
            token_info["misc"]["lang"] = lang
    return annotation


def get_lang_and_dataset_name(dirname: str) -> Tuple[str, str]:
    lang2code = {
        "english": "en",
        "dutch": "nl",
        "german": "de",
        "spanish": "es",
        "japanese": "ja",
    }
    match = re.search(r"UD_(?P<lang>[a-zA-Z]+)-(?P<dataset>.*)", dirname)
    lang = match["lang"]
    dataset = match["dataset"]
    return lang2code[lang.lower()], dataset


def process_conllu_file(infile: str, outfile: str, lang: str):
    with open(infile, "r") as conllu_infile:
        with open(outfile, "w") as conllu_outfile:
            for annotation in conllu.parse_incr(conllu_infile):
                annotation = parse_conllu_annotation(annotation, lang)
                conllu_outfile.write(annotation.serialize())


def process_data(args: argparse.Namespace):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for dirpath in tqdm.tqdm(glob.glob(os.path.join(input_dir, "UD_*"))):
        if os.path.isdir(dirpath):
            _, dirname = os.path.split(dirpath)
            lang, dataset = get_lang_and_dataset_name(dirname)
            if dataset == "PUD":
                # This dataset has no training data. We ignore this type
                continue
            outdir = os.path.join(output_dir, dirname)
            os.makedirs(outdir, exist_ok=True)
            logger.info(f"Processing {dirname}")
            for fpath in glob.glob(os.path.join(input_dir, dirname, "*.conllu")):
                _, fname = os.path.split(fpath)
                infile = os.path.join(input_dir, dirname, fname)
                outfile = os.path.join(output_dir, dirname, fname)
                process_conllu_file(infile, outfile, lang)


if __name__ == "__main__":
    args = get_arguments()
    process_data(args)
