import argparse
import itertools
import os
from typing import Callable, List, Optional

import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer


def get_wordpiece_tokenizer(bert_model_str: str) -> Callable[[str], List[str]]:
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_str, do_lower_case=False)
    wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize
    return wordpiece_tokenizer


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    return False


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        dest="mode",
        choices=["file", "folder", "test"],
        required=False,
        help="Main command arguments",
    )
    parser.add_argument("-b", "--bert_path", type=str, dest="bert_path")
    args, _ = parser.parse_known_args()
    return args


def get_folder_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--indir", type=str, dest="indir", required=True)
    parser.add_argument(
        "-t", "--threshold", type=int, dest="threshold", required=False, default=256
    )
    parser.add_argument(
        "-e", "--encoding", type=str, dest="encoding", required=False, default="iob1"
    )
    parser.add_argument(
        "-l",
        "--langs",
        dest="langs",
        nargs="+",
        default=["en", "es", "de", "nl"],
        required=False,
    )
    args, _ = parser.parse_known_args()
    return args


def get_file_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--infile", type=str, dest="infile", required=True)
    parser.add_argument("-o", "--outfile", type=str, dest="outfile", required=True)
    parser.add_argument(
        "-t", "--threshold", type=int, dest="threshold", required=False, default=256
    )
    args, _ = parser.parse_known_args()
    return args


def get_wordpieces(
    tokens: List[str], wordpiece_tokenizer: Callable[[str], List[str]]
) -> List[str]:

    tokenized = [
        wordpiece for token in tokens for wordpiece in wordpiece_tokenizer(token)
    ]
    return tokenized


def filter_file(
    infile: str,
    wordpiece_tokenizer: Callable[[str], List[str]],
    outfile: str,
    threshold: Optional[int] = 256,
) -> None:
    outlines = []
    with open(infile, "r", encoding="ISO-8859-1") as data_file:
        for is_divider, lines in tqdm.tqdm(itertools.groupby(data_file, _is_divider)):
            if not is_divider:
                lines_list = [line for line in lines]
                tokens = [line.split()[0] for line in lines_list]
                wordpieces = get_wordpieces(tokens, wordpiece_tokenizer)
                if len(wordpieces) < threshold:
                    outlines.append("".join(lines_list))
    with open(outfile, "w", encoding="ISO-8859-1") as outfile:
        outfile.write("\n".join(outlines))


if __name__ == "__main__":

    global_args = get_arguments()
    wordpiece_tokenizer = get_wordpiece_tokenizer(global_args.bert_path)
    if global_args.mode == "test":
        tokens = [
            "The",
            "quick",
            "brown",
            "fox",
            "jumped",
            "over",
            "the",
            "@allennlpisawesome@",
            "lazy",
            "dog",
        ]
        wordpieces = get_wordpieces(tokens, wordpiece_tokenizer)
        print(len(wordpieces), len(tokens))
    elif global_args.mode == "file":
        args = get_file_args()
        filter_file(args.infile, wordpiece_tokenizer, args.outfile, args.threshold)
    elif global_args.mode == "folder":
        args = get_folder_args()
        encoding = args.encoding
        for lang in args.langs:
            dir_path = os.path.join(args.indir, lang, encoding)
            assert os.path.exists(dir_path)
            # First the training file
            train_file = os.path.join(dir_path, f"{lang}.train")
            assert os.path.exists(train_file)
            out_train_file = os.path.join(dir_path, f"{lang}.filtered.train")
            filter_file(train_file, wordpiece_tokenizer, out_train_file, args.threshold)
            # The the dev file
            valid_file = os.path.join(dir_path, f"{lang}.testa")
            out_valid_file = os.path.join(dir_path, f"{lang}.filtered.testa")
            filter_file(valid_file, wordpiece_tokenizer, out_valid_file, args.threshold)
