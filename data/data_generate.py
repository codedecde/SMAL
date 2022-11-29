LANGS = ["en"]
SEED = 236  # episodes in Friends

OUTFILES = ["seed.txt", "rest.txt"]
SEED_SIZE = 5.0  # As a percent
MODE = "mbert"  # how to count when creating seed: lines vs tokens
DEDUP = True  # whether duplicates are to be removed
SKIP_LENGTH = (
    175  # skip lines longer than this in train, to avoid out of GPU mem issues
)
EN_SEED_SIZE = 10181

assert MODE in ["lines", "tokens", "mbert"]

if len(LANGS) == 1:
    lang_str = LANGS[0]
elif len(LANGS) == 4:
    lang_str = "all"
else:
    raise NotImplementedError("LANGS must be one or all")

dedup_str = "_dedup" if DEDUP else ""

OUTDIR = "conll2003/%s/%s/al%s/al_%s_seed236_line%s%s" % (
    lang_str,
    MODE,
    dedup_str,
    lang_str,
    str(SEED_SIZE / 100.0),
    dedup_str,
)

import os

import numpy as np

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

lines = []
curr_sent = []
num_toks = 0
max_len = 0.0

for lang in LANGS:
    if lang == "de":
        file_encoding = "latin-1"
    else:
        file_encoding = None
    with open("conll2003/%s/iob1/%s.train" % (lang, lang), encoding=file_encoding) as f:
        for l in f:
            if l.strip() == "" or l.startswith("-DOCSTART-"):
                if len(curr_sent) > 0:
                    lines.append(curr_sent)
                    num_toks += len(curr_sent)
                    max_len = max(max_len, len(curr_sent))
                    curr_sent = []
                continue

            l_toks = l.strip().split()
            curr_sent.append(l_toks[0] + " " + lang + " " + l_toks[-1])

num_lines = len(lines)

np.random.seed(SEED)
np.random.shuffle(lines)
print(
    "Total lines: %d, Total tokens: %d, Max length: %d" % (num_lines, num_toks, max_len)
)

if DEDUP:
    seen = (
        set()
    )  # http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
    all_sents = ["\t".join(l) for l in lines]
    all_unique_sents = [s for s in all_sents if not (s in seen or seen.add(s))]
    lines = [s.split("\t") for s in all_unique_sents]
    distinct_toks = sum([len(s) for s in lines])
    print("Total de-duped lines: %d, Total tokens: %d" % (len(lines), distinct_toks))

print("First sentence post shuffle: ", lines[0])

seeded_toks = 0
seeded_lines = 0
is_done = False

with open(os.path.join(OUTDIR, OUTFILES[0]), "w") as f_seed, open(
    os.path.join(OUTDIR, OUTFILES[1]), "w"
) as f_al:
    for i, l in enumerate(lines):
        if len(l) >= SKIP_LENGTH:
            continue
        if not is_done and (
            (MODE == "tokens" and (seeded_toks + len(l)) < SEED_SIZE * num_toks / 100.0)
            or (MODE == "mbert" and (seeded_toks + len(l)) < EN_SEED_SIZE)
            or (MODE == "lines" and (seeded_lines + 1) < SEED_SIZE * num_lines / 100.0)
        ):
            f_seed.write("\n".join(l))
            f_seed.write("\n\n")
            seeded_toks += len(l)
            seeded_lines += 1
        else:
            # If we're done with seed, we're done. This ensures that a biased seed set is not created by sampling
            # smaller sentences to "pad up" the difference between the seed budget and the seed set we've generated
            is_done = True
            f_al.write("\n".join(l))
            f_al.write("\n\n")

print(
    "\n\nDone.\nLines in seed: %d of %d\nTokens in seed: %d of %d"
    % (seeded_lines, num_lines, seeded_toks, num_toks)
)
