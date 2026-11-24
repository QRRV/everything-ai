# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/pattern-sentiment.tsv.gz: the sentiment lexicons of Pattern for English and Dutch, one row per sense of
a word, with its polarity, subjectivity and intensity. Lesson nlp-sentiment-lexicon compares them with the lexicon a
learner builds from labeled review sentences.

Pattern (CLiPS, University of Antwerp) is by Tom De Smedt and Walter Daelemans; its code is BSD-3-Clause, and the two
lexicon files say license="PDDL" in their root element. The course took them from the repository clips/pattern at
commit af754685cca3713db0abc4f020f2e94467c19d85 (the head of master), and pins each file by the SHA-256 below:

    https://raw.githubusercontent.com/clips/pattern/<commit>/pattern/text/en/en-sentiment.xml
    https://raw.githubusercontent.com/clips/pattern/<commit>/pattern/text/nl/nl-sentiment.xml

Download both into a folder and run, from the course folder:

    uv run python data/tools/make_pattern_lexicons.py <folder>

What the file holds: a header line, then one sense per line, English first and then Dutch, each in the order of the
lexicon file, with the columns lang (en or nl), form (the word as the file gives it, also a few phrases such as
"in good taste"), pos (the Penn Treebank tag, JJ for an adjective; empty for 6 English and 11 Dutch rows), sense
(the file's short gloss; empty for 129 English and 221 Dutch rows), polarity (-1.0 to +1.0), subjectivity (0.0 to
1.0), intensity, confidence (the file's confidence, or its reliability for the 221 English rows without one),
wordnet_id (a WordNet 3.0 synset, such as a-02203763; empty for 134 English and 859 Dutch rows) and shown, separated by tabs. The numbers stay as the file
writes them.

shown is 0 for the rows the course does not show in a lesson: the 12 English rows the file labels "profanity", crude
or insulting words the file does not label, and words for a group of people or for sex, which the lexicon gives a
polarity like any other adjective. They stay in the file and count, as in Pattern.
"""
import gzip
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from everything_ai import paths

COMMIT = "af754685cca3713db0abc4f020f2e94467c19d85"
SHA256 = {"en-sentiment.xml": "0ca603db55570bab9b2865716a3696ede01e3da02b570c47628a9dbcc3b37d1e",
          "nl-sentiment.xml": "79284a8f68d496428dea494cd982490700beb83eb16e51d06a550894e8ef8d23"}
HIDDEN = {   # forms the course does not show, besides the rows labeled "profanity"
    "en": {"bloody", "gay", "jackass", "jewish", "naked", "retard", "retarded", "retards", "sexual", "sexy"},
    "nl": {"gay", "godverdoms", "homoseksueel", "klote", "kloten", "lullig", "pissig", "sexy", "verkloten", "verwijfd"},
}
COLUMNS = ["lang", "form", "pos", "sense", "polarity", "subjectivity", "intensity", "confidence", "wordnet_id", "shown"]


def main(folder: str) -> None:
    rows, counts = [], {}
    for lang in ("en", "nl"):
        name = f"{lang}-sentiment.xml"
        data = (Path(folder) / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == SHA256[name], name
        root = ET.fromstring(data)
        assert root.tag == "sentiment" and root.get("language") == lang and root.get("license") == "PDDL"
        n = hidden = 0
        for w in root.iter("word"):
            shown = 0 if w.get("label") == "profanity" or w.get("form").lower() in HIDDEN[lang] else 1
            cells = [lang, w.get("form"), w.get("pos", ""), w.get("sense", ""), w.get("polarity"), w.get("subjectivity"),
                     w.get("intensity"), w.get("confidence", w.get("reliability", "")), w.get("wordnet_id", ""), str(shown)]
            assert all("\t" not in c and "\n" not in c for c in cells), cells
            float(cells[4])
            rows.append("\t".join(cells))
            n += 1
            hidden += 1 - shown
        counts[lang] = (n, hidden)
    out = paths.ROOT / "data" / "pattern-sentiment.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\t".join(COLUMNS) + "\n" + "\n".join(rows) + "\n").encode("utf-8"))
    print(", ".join(f"{lang}: {n:,} senses, {h} not shown" for lang, (n, h) in counts.items()))
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
