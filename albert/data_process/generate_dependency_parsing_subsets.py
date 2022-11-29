from __future__ import absolute_import

import argparse
import glob
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# split_ratio = 0.
SEED = 53  # Number of episodes in The Good Place

INPUT_DIRNAME = "processed_treebanks"
OUTPUT_DIRNAME = "split_treebanks"
LANGS = ["Dutch", "German", "English", "Japanese", "Spanish"]
LANGS_WITH_ALL = LANGS + ["all4", "all5"]


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert Dependency Parser Dataset")
    parser.add_argument(
        "--parent_path",
        type=str,
        dest="parent_path",
        required=True,
        help="The root directory with the data",
    )
    parser.add_argument(
        "--token_frac",
        type=float,
        dest="token_frac",
        required=False,
        default=1.0,
        help="The fraction of number of tokens to be placed in the seed vs rest",
    )
    args = parser.parse_args()
    return args


def count_file_dups(fname):
    lines = []
    with open(fname) as f:
        for l in f:
            if l.startswith("# text = "):
                lines.append(l)

    return len(lines), len(lines) - len(set(lines))


# Step 1: Join train data per language
def get_merged_data(data_type, langs, input_path):
    lang_file_mapping = {l: [] for l in langs}

    for dirpath in glob.glob(os.path.join(input_path, "UD_*")):
        if os.path.isdir(dirpath):
            _, dirname = os.path.split(dirpath)
            for fpath in glob.glob(os.path.join(input_path, dirname, "*.conllu")):
                _, fname = os.path.split(fpath)
                if data_type in fname:
                    is_done = False
                    for lang in langs:
                        if lang in dirname:
                            assert not is_done
                            lang_file_mapping[lang].append(fpath)
                            is_done = True
                    assert is_done

    train_contents = {lang: [] for lang in langs}

    for lang in langs:
        for f in lang_file_mapping[lang]:
            with open(f) as f_ptr:
                train_contents[lang] += f_ptr.read().split("\n\n")
        if train_contents[lang][-1].strip() == "":
            train_contents[lang] = train_contents[lang][:-1]

    print("\tPre-dedup Word stats:")
    for lang in langs:
        print(
            "\t\t",
            lang,
            ":",
            sum(
                1
                for l in ("\n".join(train_contents[lang])).split("\n")
                if l.strip() and not l.strip().startswith("#")
            ),
        )

    return train_contents


# Step 2: De-dup data
def get_deduped_data(langs, train_contents, get_all=False):

    train_contents_unique_intermed = {lang: {} for lang in langs}

    for lang in langs:
        for line in train_contents[lang]:
            key = "\n".join(line.split("\n# ")[-1].split("\n")[1:])
            if key not in train_contents_unique_intermed:
                train_contents_unique_intermed[lang][key] = []
            train_contents_unique_intermed[lang][key].append(line)

    train_contents_unique = {
        lang: [
            train_contents_unique_intermed[lang][key][0]
            for key in train_contents_unique_intermed[lang]
        ]
        for lang in langs
    }

    if get_all:
        train_contents_unique["all5"] = [
            e for lang in langs for e in train_contents_unique[lang]
        ]
        langs4 = [lang for lang in langs if lang != "Japanese"]
        train_contents_unique["all4"] = [
            e for lang in langs4 for e in train_contents_unique[lang]
        ]

    return train_contents_unique


# Step 3: Shuffle data deterministically
def get_shuffled_data(langs, seed, train_contents_unique):
    np.random.seed(SEED)

    print("\tPre-shuffle, en: ", train_contents_unique["English"][0])

    for lang in langs:
        np.random.shuffle(train_contents_unique[lang])

    print("\tPost-shuffle, en: ", train_contents_unique["English"][0])

    print("\tPost-dedup Word stats:")
    for lang in langs:
        print(
            "\t\t",
            lang,
            ":",
            sum(
                1
                for l in ("\n".join(train_contents_unique[lang])).split("\n")
                if l.strip() and not l.strip().startswith("#")
            ),
        )

    return train_contents_unique


# Step 4: Split data
def get_split_data(langs, shuffled_contents, token_count):
    seed_out = {lang: [] for lang in langs}
    rest_out = {lang: [] for lang in langs}

    count_so_far = {lang: 0 for lang in langs}

    if token_count == -1:
        return shuffled_contents, shuffled_contents

    for lang in langs:
        for line in shuffled_contents[lang]:
            toks_in_line = len(line.strip().split("\n# ")[-1].split("\n")[1:])
            if toks_in_line + count_so_far[lang] <= token_count:
                count_so_far[lang] += toks_in_line
                seed_out[lang].append(line)
            else:
                rest_out[lang].append(line)

    return seed_out, rest_out


# Step 5: Write the output
def write_to_output(data_type, langs, out_path, seed_out, rest_out):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for lang in langs:

        seed_out_file_path = os.path.join(
            output_path, lang + "-" + data_type + "-seed.conllu"
        )
        with open(seed_out_file_path, "w") as f:
            f.write("\n\n".join(seed_out[lang]) + "\n\n")

        rest_out_file_path = os.path.join(
            output_path, lang + "-" + data_type + "-rest.conllu"
        )
        with open(rest_out_file_path, "w") as f:
            f.write("\n\n".join(rest_out[lang]) + "\n\n")


if __name__ == "__main__":
    args = get_arguments()

    token_frac = args.token_frac
    parent_path = args.parent_path
    input_path = os.path.join(parent_path, INPUT_DIRNAME)
    output_path = os.path.join(parent_path, OUTPUT_DIRNAME)

    for data_type, tot_toks in zip(["train", "dev"], [430470, 67688]):
        if token_frac == 1.0:
            token_count = -1
        else:
            token_count = token_frac * tot_toks

        print("\n\nGenerating file:", data_type)
        train_contents = get_merged_data(data_type, LANGS, input_path)
        train_contents_unique = get_deduped_data(LANGS, train_contents, get_all=True)

        tmp1 = {k: len(train_contents[k]) for k in train_contents}
        tmp2 = {k: len(train_contents_unique[k]) for k in train_contents}
        print("\tContents: ", tmp1)
        print("\tUnique Contents: ", tmp2)

        shuffled_contents = get_shuffled_data(
            LANGS_WITH_ALL, SEED, train_contents_unique
        )
        data_seed, data_rest = get_split_data(
            LANGS_WITH_ALL, shuffled_contents, token_count
        )
        write_to_output(data_type, LANGS_WITH_ALL, output_path, data_seed, data_rest)
