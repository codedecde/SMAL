from __future__ import absolute_import

import argparse
import io
from typing import List, Tuple

import numpy as np
import tqdm


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--words_file", type=str, dest="words_file")
    parser.add_argument(
        "-e", "--embed_file", type=str, dest="embed_file", required=True
    )
    parser.add_argument("-o", "--out_file", type=str, dest="out_file", required=True)
    args = parser.parse_args()
    return args


def read_embeddings(words_file: str, embed_file: str) -> Tuple[List[str], np.array]:
    embeddings = np.load(embed_file)["embeddings"]
    words = []
    with io.open(
        words_file, "r", encoding="utf-8", newline="\n", errors="ignore"
    ) as infile:
        for word in infile:
            words.append(word.rstrip())
    return words, embeddings


def write_embeddings(words_list: List[str], embeddings: np.array, outfile: str) -> None:
    with io.open(outfile, "w", encoding="utf-8") as outfile:
        for word, embedding in tqdm.tqdm(zip(words_list, embeddings)):
            outfile.write("%s %s\n" % (word, " ".join("%.5f" % x for x in embedding)))


if __name__ == "__main__":
    args = get_arguments()
    words_list, embeddings = read_embeddings(args.words_file, args.embed_file)
    write_embeddings(words_list, embeddings, args.out_file)
