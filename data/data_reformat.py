LANGS = ["en", "de", "es", "nl"]

import os

import numpy as np

for lang in LANGS:
    OUTDIR = "conll2003/" + lang + "/reformatted/"
    if not os.path.exists(os.path.join(OUTDIR)):
        os.makedirs(os.path.join(OUTDIR))
    for fname in os.listdir("conll2003/%s/iob1/" % (lang)):
        docstart = False
        with open("conll2003/%s/iob1/%s" % (lang, fname)) as f, open(
            os.path.join(OUTDIR, fname), "w"
        ) as f_out:
            for i, l in enumerate(f):
                if l.startswith("-DOCSTART-"):
                    docstart = True
                    continue
                elif l.strip() == "" and docstart:
                    docstart = False
                    continue
                elif l.strip() == "":
                    f_out.write(l)
                    continue

                l_toks = l.strip().split()
                f_out.write(l_toks[0] + " " + lang + " " + l_toks[-1] + "\n")

        print(
            "Done converting %s into %s"
            % ("conll2003/%s/iob1/%s" % (lang, fname), os.path.join(OUTDIR, fname))
        )
