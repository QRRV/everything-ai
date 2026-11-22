# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/uci-sentiment.tsv.gz: the 3,000 sentences of "Sentiment Labelled Sentences" (UCI Machine Learning
Repository, dataset 331), short sentences from reviews on Amazon, IMDb and Yelp, each labelled positive (1) or
negative (0), 500 of each for each website. The course uses them for its first lessons that turn texts into vectors
and sort them.

Dimitrios Kotzias made the set for the paper "From Group to Individual Labels using Deep Features" (Kotzias, Denil,
de Freitas and Smyth, KDD 2015); UCI publishes it under CC BY 4.0 (https://doi.org/10.24432/C57604). The zip has no
versions, so the course pins it by its SHA-256. Download
https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip and run, from the course folder:

    uv run python data/tools/make_uci_sentiment.py <the zip>

What changes: nothing in the words. Each line of the three files `amazon_cells_labelled.txt`, `imdb_labelled.txt` and
`yelp_labelled.txt` is cut at its last tab into the sentence and the label. The spaces around a sentence go (every line
of the IMDb file has two spaces before the tab). Eight characters in the IMDb file are control characters U+0085,
U+0096 and U+0097: the bytes of the ellipsis, the en dash and the em dash in Windows-1252, read with the wrong code page
at some point, so they become those three characters again. The files are read line by line: a reader that treats
double quotes as quoting, such as Python's csv module with its default settings, finds only 748 lines in the IMDb file.

What the file holds: a header line `source<TAB>line<TAB>label<TAB>shown<TAB>text`, then the 3,000 sentences in the
order of the zip: amazon, imdb, yelp, each by its line number in its file (from 1), so that a sentence can be found
again. `shown` is 0 for the sentences a lesson does not show as an example: swear words, insults, slurs, and crude
or sexual remarks, found with a word list and read by eye. They stay in the file, so that counts over the set stay the
counts of the published set.
"""
import gzip
import hashlib
import sys
import zipfile
from pathlib import Path

from everything_ai import paths

SHA256 = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
FOLDER = "sentiment labelled sentences/"
SOURCES = {"amazon": "amazon_cells_labelled.txt", "imdb": "imdb_labelled.txt", "yelp": "yelp_labelled.txt"}
# the three control characters, as the Windows-1252 characters their bytes stand for
CONTROLS = {"\x85": "…", "\x96": "–", "\x97": "—"}
# line numbers of the sentences that are not shown as examples
NOT_SHOWN = {
    "amazon": {113, 329, 351, 358, 380, 386, 407, 503, 545, 570, 615, 622, 654, 895, 908},
    "imdb": {35, 81, 95, 111, 118, 144, 146, 160, 165, 178, 187, 231, 251, 252, 253, 254, 255, 283, 310, 318, 351, 374,
             376, 380, 381, 417, 425, 429, 456, 515, 517, 537, 560, 603, 604, 707, 708, 857, 871, 872, 874,
             875, 880, 881, 882, 883, 914, 989},
    "yelp": {6, 38, 40, 47, 75, 92, 126, 187, 221, 234, 403, 415, 428, 597, 611, 685, 718, 801, 839, 953, 968, 975},
}


def main(zip_path: str) -> None:
    data = Path(zip_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256, "not the pinned zip"
    rows = ["source\tline\tlabel\tshown\ttext"]
    with zipfile.ZipFile(zip_path) as z:
        for source, name in SOURCES.items():
            text = z.read(FOLDER + name).decode("utf-8")
            lines = text.split("\n")
            assert lines[-1] == "" and len(lines) == 1001, name
            labels = {"0": 0, "1": 0}
            for number, line in enumerate(lines[:-1], start=1):
                sentence, label = line.rsplit("\t", 1)
                assert label in labels, (name, number)
                labels[label] += 1
                for bad, good in CONTROLS.items():
                    sentence = sentence.replace(bad, good)
                sentence = sentence.strip()
                assert sentence and "\t" not in sentence
                assert not any("\x80" <= ch <= "\x9f" for ch in sentence), (name, number)
                shown = 0 if number in NOT_SHOWN[source] else 1
                rows.append(f"{source}\t{number}\t{label}\t{shown}\t{sentence}")
            assert labels == {"0": 500, "1": 500}, (name, labels)
            print(f"{source}: 1,000 sentences, {len(NOT_SHOWN[source])} not shown")
    out = paths.ROOT / "data" / "uci-sentiment.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(rows) + "\n").encode("utf-8"))
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
