# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/stsb-captions.tsv.gz and data/stsb-captions-granite.npz: the 1,750 pairs of image captions of the STS
Benchmark, in English and in Dutch, each with the score people gave it for how alike the two sentences are in meaning,
from 0 (on different topics) to 5 (they mean the same thing), and the cosine similarity Granite Embedding gives each
pair in both languages. Lesson nlp-sts chooses a threshold on those cosine similarities with them.

The STS Benchmark (Cer, Diab, Agirre, Lopez-Gazpio and Specia, SemEval 2017) collects 8,628 English pairs of the STS
tasks of 2012 to 2017, in train, dev and test parts. Its scores are CC BY-SA 4.0; its sentences have the license of
their own source, and only two sources let the course pass them on with a short credit: `images`, captions of PASCAL VOC-2008
pictures (Rashtchian and others 2010, "released under a CreativeCommons Attribution-ShareAlike license"), and
`track5.en-en`, sentences of SNLI (Bowman and others 2015, CC BY-SA 4.0). The other sources (MSRpar and MSRvid of
Microsoft Research, headlines, DEFT, the forums) are left out. The Dutch sentences are the machine translation by
DeepL of stsb-multi-mt (Philip May, 2021), "distributed under the same license as the original dataset"; the scores
are the English ones.

Download, and check, the two sources:

- https://dl.fbaipublicfiles.com/glue/data/STS-B.zip, the STS Benchmark as GLUE ships it, with the original files in
  STS-B/original/ (SHA-256 below);
- data/stsb-nl-train.csv, stsb-nl-dev.csv and stsb-nl-test.csv of https://github.com/PhilipMay/stsb-multi-mt at
  commit 30de0dec4ee199b7f42351d3f1a0b19592955385, into one folder.

Then download Granite Embedding once in a lesson, or with `uv run python -c "from everything_ai import embed;
embed.download()"`, and run, from the course folder:

    uv run python data/tools/make_sts_captions.py <STS-B.zip> <the folder with the Dutch files>

What changes: nothing in the words or the scores. The lines of the Dutch files follow the lines of the English ones;
the script checks that the English sentences and scores of stsb-multi-mt are those of the STS Benchmark.

What the files hold. stsb-captions.tsv.gz: a header line
`part<TAB>source<TAB>score<TAB>shown<TAB>en1<TAB>en2<TAB>nl1<TAB>nl2`, then the 1,750 pairs in the order of the STS
Benchmark's train, dev and test files (1,000, 375 and 375 pairs); `source` is `images` or `track5`, and `score` the
score people gave the pair, as the benchmark gives it. `shown` is 0 for the pairs a lesson does not show as an example
(nudity, blood, weapons, death, smoking and drinking, found with the word list below in the English sentences); they
stay in the file, so that counts over the set stay the counts of the published set. stsb-captions-
granite.npz: `en` and `nl`, two arrays of 1,750 numbers in float32, the cosine similarity of Granite's embeddings of
the two sentences of each pair, in the order of the lines. Both files are written with a fixed date, so the same
sources give the same bytes.
"""
import csv
import gzip
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

from everything_ai import embed, paths

GLUE_SHA256 = "e60a6393de5a8b5b9bac5020a1554b54e3691f9d600b775bd131e613ac179c85"
DUTCH_SHA256 = {
    "train": "4d60f75ec8e7a51c79f2fd7207747cf567bb16812f12f798031b119a14018f7c",
    "dev": "d2b2a895824d6bd60a28dbf9c22248df6e65603a389c08943255b7c2c694f8f2",
    "test": "a0ee96b6ee9ed0181664e687f6b42f6273a87c885f34833a1e5a06368759cb35",
}
SOURCES = {"images": "images", "track5.en-en": "track5"}   # the sources whose sentences the course may pass on
# words in the English sentences that keep a pair from being shown as an example
NOT_SHOWN = re.compile(r"\b(naked|nude\w*|bikini\w*|bloody|blood|kill\w*|dead|death|gun\w*|weapon\w*|sex\w*|drunk|"
                       r"beer|wine|lager|cigarette\w*|smok\w*)\b", re.IGNORECASE)
TSV = paths.ROOT / "data" / "stsb-captions.tsv.gz"
NPZ = paths.ROOT / "data" / "stsb-captions-granite.npz"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pairs(glue_zip: Path, dutch: Path) -> list:
    """(part, source, score, shown, en1, en2, nl1, nl2) for every pair of the two sources, in the order of the files."""
    data = glue_zip.read_bytes()
    if _sha256(data) != GLUE_SHA256:
        sys.exit(f"{glue_zip} is not the STS-B.zip the course pins")
    found = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for part in ("train", "dev", "test"):
            english = [line.split("\t") for line in z.read(f"STS-B/original/sts-{part}.tsv").decode("utf-8").split("\n")
                       if line.strip()]
            raw = (dutch / f"stsb-nl-{part}.csv").read_bytes()
            if _sha256(raw) != DUTCH_SHA256[part]:
                sys.exit(f"stsb-nl-{part}.csv is not the file the course pins")
            translated = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
            if len(english) != len(translated):
                sys.exit(f"{part}: {len(english)} English pairs and {len(translated)} Dutch ones")
            for row, nl in zip(english, translated):
                file, score, en1, en2 = row[1], row[4], row[5], row[6]
                if abs(float(score) - float(nl[2])) > 1e-9:
                    sys.exit(f"{part}: the scores of the two files differ at {en1!r}")
                if file in SOURCES:
                    shown = "0" if NOT_SHOWN.search(en1 + " " + en2) else "1"
                    found.append((part, SOURCES[file], score, shown, en1, en2, nl[0], nl[1]))
    return found


def write_tsv(found: list) -> None:
    lines = ["part\tsource\tscore\tshown\ten1\ten2\tnl1\tnl2"] + ["\t".join(row) for row in found]
    for row in found:
        if any("\t" in x or "\n" in x for x in row):
            sys.exit(f"a tab or a line break in {row!r}")
    with open(TSV, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(lines) + "\n").encode("utf-8"))


def cosines(first: list, second: list) -> np.ndarray:
    a, b = embed.embed(first).numpy(), embed.embed(second).numpy()   # each of length 1
    return np.sum(a * b, axis=1).astype(np.float32)


def write_npz(found: list) -> dict:
    arrays = {"en": cosines([r[4] for r in found], [r[5] for r in found]),
              "nl": cosines([r[6] for r in found], [r[7] for r in found])}
    with zipfile.ZipFile(NPZ, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    return arrays


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    found = pairs(Path(sys.argv[1]), Path(sys.argv[2]))
    write_tsv(found)
    arrays = write_npz(found)
    print(f"{TSV.name}: {len(found)} pairs; {NPZ.name}: en {arrays['en'].shape}, nl {arrays['nl'].shape}")


if __name__ == "__main__":
    main()
