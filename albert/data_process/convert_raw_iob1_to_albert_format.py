from __future__ import absolute_import

import argparse
import itertools
from typing import List


def _is_divider(line: str) -> bool:
    if line.strip() == "":
        return True
    first_token = line.split()[0]
    if first_token == "-DOCSTART-":
        return True
    return False


def convert(lines: List[str], lang: str = "en") -> List[str]:
    new_lines = []
    for line in lines:
        token, pos, chunk, tag = line.split()
        new_lines.append(" ".join([token, lang, tag]))
    return new_lines


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert Conll2003 files")
    parser.add_argument("-i", "--infile", dest="infile", type=str, required=True)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str, required=True)
    parser.add_argument(
        "-l", "--lang", dest="lang", type=str, required=False, default="en"
    )
    args = parser.parse_args()
    return args


def convert_file(infile: str, outfile: str, lang: str) -> None:
    data = []
    with open(infile, "r") as infile_handle:
        for is_divider, lines in itertools.groupby(infile_handle, _is_divider):
            if not is_divider:
                data.append(convert(lines, lang))
    with open(outfile, "w") as outfile_handle:
        outfile_handle.write("\n\n".join("\n".join(lines) for lines in data))


if __name__ == "__main__":
    args = get_args()
    convert_file(args.infile, args.outfile, args.lang)
