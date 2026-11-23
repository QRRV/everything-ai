# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/snli-sample.tsv.gz: pairs of sentences of SNLI 1.0, the Stanford Natural Language Inference corpus
(Bowman, Angeli, Potts and Manning, EMNLP 2015; CC-BY-SA-4.0), for the lesson of domain taal about natural language
inference (read with everything_ai.snli).

Each pair of SNLI is a premise, the caption of a photo (nearly always of Flickr30k; about 4,000 training pairs have a
caption of VisualGenome), and a hypothesis that a crowd worker wrote for it, with the label people gave the pair:
entailment, neutral or contradiction. The course keeps the pairs of 10,000 premises of the training set, all pairs with
a label of each of them (36,788 pairs), and the 9,824 pairs of the test set that have a label. The 176 test pairs for
which no label got three of the five votes (label -1) go. The premises of the training set are taken in the order of the
SHA-256 of their text, so that the choice does not depend on a random generator. `shown` is 0 for the pairs that a
lesson does not show as an example: those with a crude, sexual or violent word, a slur, nudity, blood, a weapon, drink
or smoking (WORDS). They stay in the file; no such filter is complete. The sentences are unchanged.

SNLI comes from Hugging Face as parquet files at a fixed commit of the dataset stanfordnlp/snli. Download the two files
below into one folder and run, from the course folder:

    uv run --with pyarrow python data/tools/make_snli_sample.py <folder>
"""
import gzip
import hashlib
import re
import sys
from pathlib import Path

COMMIT = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
SOURCE = f"https://huggingface.co/datasets/stanfordnlp/snli/resolve/{COMMIT}/plain_text/{{split}}-00000-of-00001.parquet"
SHA256 = {
    "train": "ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620",
    "test": "4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c",
}
LABELS = ("entailment", "neutral", "contradiction")   # labels 0, 1 and 2 of the parquet files
PREMISES = 10_000
OUT = Path(__file__).resolve().parents[2] / "data" / "snli-sample.tsv.gz"
# words that keep a pair from being shown as an example
WORDS = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|damn\w*|ass|asses|asshole\w*|bastard\w*|crap\w*|dick|dicks|piss\w*|"
                   r"cunt\w*|whore\w*|slut\w*|fag\w*|nigg\w*|retard\w*|sex|sexy|sexual\w*|porn\w*|nude\w*|naked|"
                   r"topless|bikini\w*|underwear|rape\w*|kill|kills|killed|killing\w*|killer\w*|murder\w*|dead|"
                   r"death\w*|die|died|dies|dying|blood\w*|gun|guns|rifle\w*|weapon\w*|knife|knives|bomb\w*|"
                   r"terror\w*|suicid\w*|torture\w*|corpse\w*|nazi\w*|drug|drugs|cocaine|heroin|drunk\w*|beer\w*|"
                   r"wine|vodka|whiskey|alcohol\w*|cigar\w*|smok\w*)\b", re.IGNORECASE)


def _rows(folder: Path, split: str) -> list:
    """(premise, hypothesis, label) for every pair of one parquet file, in its order; label is -1 without consensus."""
    import pyarrow.parquet as pq

    path = folder / f"{split}-00000-of-00001.parquet"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256[split], f"{path} is not the file of commit {COMMIT}"
    table = pq.read_table(path).to_pydict()
    return list(zip(table["premise"], table["hypothesis"], table["label"]))


def pairs(folder: Path) -> list:
    """(split, shown, label, premise, hypothesis) for every pair the course keeps: train first, then test."""
    out = []
    train = [r for r in _rows(folder, "train") if r[2] >= 0]
    premises = sorted({r[0] for r in train}, key=lambda p: hashlib.sha256(p.encode("utf-8")).hexdigest())
    keep = set(premises[:PREMISES])
    for split, rows in (("train", [r for r in train if r[0] in keep]),
                        ("test", [r for r in _rows(folder, "test") if r[2] >= 0])):
        for premise, hypothesis, label in rows:
            assert "\t" not in premise + hypothesis and "\n" not in premise + hypothesis
            shown = "0" if WORDS.search(premise + " " + hypothesis) else "1"
            out.append((split, shown, LABELS[label], premise, hypothesis))
    return out


def main(folder: Path) -> None:
    found = pairs(folder)
    lines = ["split\tshown\tlabel\tpremise\thypothesis"] + ["\t".join(row) for row in found]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    OUT.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    n = {s: sum(r[0] == s for r in found) for s in ("train", "test")}
    print(f"{OUT}: {n['train']} train pairs, {n['test']} test pairs, {sum(r[1] == '0' for r in found)} not shown, "
          f"SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
