from __future__ import absolute_import

import argparse
import os


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert Amazon Reviews dataset")
    parser.add_argument("-r", "--root", dest="root", type=str, required=True)
    args = parser.parse_args()
    return args


def reformat_file(infile: str, outfile: str, lang: str) -> None:
    with open(infile, "r") as fin:
        with open(outfile, "w") as fout:
            for line in fin:
                line = line.strip()
                if line != "":
                    label, review = line.split("\t", 1)
                    fout.write("\t".join([label, lang, review]) + "\n")


def reformat(args: argparse.Namespace):
    root_dir = args.root
    assert os.path.exists(root_dir)
    for lang in os.listdir(root_dir):
        lang_dir = os.path.join(root_dir, lang)
        reformatted_dir = os.path.join(lang_dir, "reformatted")
        os.makedirs(reformatted_dir, exist_ok=True)
        for file in ["train.tsv", "test.tsv"]:
            infile = os.path.join(lang_dir, file)
            outfile = os.path.join(reformatted_dir, file)
            reformat_file(infile, outfile, lang)


if __name__ == "__main__":
    args = get_arguments()
    reformat(args)
