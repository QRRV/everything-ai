# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz: two treebanks of Universal Dependencies 2.19
(15 November 2026), UD_English-EWT and UD_Dutch-Alpino, both CC-BY-SA-4.0, at their tag r2.19.

What the course keeps of each sentence: its sent_id, its text, which part of the treebank it is in (a comment
`# split = train`, `dev` or `test`), and per word the columns ID, FORM, LEMMA, UPOS, HEAD and DEPREL. The other four
columns (XPOS, FEATS, DEPS, MISC) become `_`, and the lines of multiword tokens (such as "don't" over "do" and "n't")
and of empty nodes go, so every line of a word is a syntactic word with a whole number as its ID. The result is still
CoNLL-U. From EWT the course leaves out the 4,899 sentences from emails of Enron (sent_id email-enronsent...): they
hold the real names and addresses of the people who wrote them. Download the six files at the tag below into one
folder and run, from the course folder:

    uv run python data/tools/make_ud.py <folder>
"""
import gzip
import hashlib
import sys
from pathlib import Path

from everything_ai import paths

TREEBANKS = {
    "ud-en-ewt.conllu.gz": {
        "source": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.19/en_ewt-ud-{split}.conllu",
        "sha256": {
            "train": "fa6b2c9fc048273ee52f9609bcdc5880df366b6b72847e01a353259ea78c2a37",
            "dev": "ebd876f58675824910a380e1c28145739856a0d595fc44851d577d0e7047f356",
            "test": "5b5df91ca23677345b941dc58d81617753e0d2d8bee12094e2a25cead3c18973",
        },
        "leave_out": "email-enronsent",
    },
    "ud-nl-alpino.conllu.gz": {
        "source": "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/r2.19/nl_alpino-ud-{split}.conllu",
        "sha256": {
            "train": "73920030b143f2320974403e798f45b9b7170aa9268fdf1f2a954be4733288ca",
            "dev": "58252957e848bcf5da5e632b35ef988a9b18cfea0f337ea91757458ae8ae262b",
            "test": "3634dc8f5ea58f7a4921477051cc3427d53737143906501c5d59348f18579cc9",
        },
        "leave_out": None,
    },
}


def sentences(text: str):
    """The sentences of a CoNLL-U file, each as its comment lines and its word lines."""
    comments, words = [], []
    for line in text.splitlines():
        if not line:
            if words:
                yield comments, words
            comments, words = [], []
        elif line.startswith("#"):
            comments.append(line)
        else:
            words.append(line.split("\t"))
    if words:
        yield comments, words


def comment(comments: list, key: str) -> str:
    for line in comments:
        if line.startswith(f"# {key} = "):
            return line[len(f"# {key} = "):]
    raise ValueError(key)


def main(folder: str) -> None:
    for out, tb in TREEBANKS.items():
        lines, kept, left, n_words = [], 0, 0, 0
        for split in ("train", "dev", "test"):
            path = Path(folder) / tb["source"].rsplit("/", 1)[1].format(split=split)
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == tb["sha256"][split], path
            for comments, words in sentences(data.decode("utf-8")):
                sent_id = comment(comments, "sent_id")
                if tb["leave_out"] and sent_id.startswith(tb["leave_out"]):
                    left += 1
                    continue
                lines += [f"# sent_id = {sent_id}", f"# split = {split}", f"# text = {comment(comments, 'text')}"]
                for cols in words:
                    if not cols[0].isdigit():   # a multiword token (3-4) or an empty node (8.1)
                        continue
                    wid, form, lemma, upos, _, _, head, deprel, _, _ = cols
                    lines.append("\t".join([wid, form, lemma, upos, "_", "_", head, deprel, "_", "_"]))
                    n_words += 1
                lines.append("")
                kept += 1
        text = "\n".join(lines) + "\n"
        with open(paths.ROOT / "data" / out, "wb") as f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
                g.write(text.encode("utf-8"))
        size = (paths.ROOT / "data" / out).stat().st_size
        print(out, kept, "sentences,", n_words, "words,", left, "left out,", size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
