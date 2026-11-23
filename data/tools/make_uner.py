# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/uner-en-ewt.tsv.gz: the names of people, places and organizations in UNER_English-EWT of Universal NER
v1 (Mayhew and others, NAACL 2024; CC-BY-SA-4.0), at a fixed commit, for the lessons of domain taal about sequence
labeling, names and span F1 (read with everything_ai.uner).

UNER_English-EWT gives every word of the sentences of UD_English-EWT (at commit c491d62a of that treebank) one tag in
IOB2: B-PER, B-LOC or B-ORG on the first word of a name, I-PER, I-LOC or I-ORG on the words after it, and O on every
other word. What the course keeps of each sentence: its sent_id, which part of the dataset it is in (train, dev or
test), its words and their tags. The column that names who checked a sentence goes. As in data/tools/make_ud.py, the
course leaves out the 4,899 sentences from emails of Enron (sent_id email-...): they hold the real names and addresses
of the people who wrote them. `shown` is 0 for the sentences that a lesson does not show as an example: those with an
email address, a web address, a phone number, a postal code or a street address (CONTACT, STREET), and those with a
crude, sexual or violent word or a slur (WORDS). They stay in the file; no such filter is complete. The words of
11,702 of the 11,722 kept sentences are the same as in data/ud-en-ewt.conllu.gz (Universal Dependencies 2.19), with
the same sent_id and split. Download the three files at the commit below into one folder and run, from the course
folder:

    uv run python data/tools/make_uner.py <folder>
"""
import gzip
import hashlib
import re
import sys
from pathlib import Path

from everything_ai import paths

COMMIT = "8ed072de2fd24022cc62458997fe96a8fe191ea4"
SOURCE = f"https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/{COMMIT}/en_ewt-ud-{{split}}.iob2"
SHA256 = {
    "train": "647b296624433d2b0f304f81974404fb1d5ed1520cc05649b4646f5777a5497d",
    "dev": "0cafd881cf31d39897b6d588420fab8fbbe4423d271725c59921ac9dd49c0713",
    "test": "821ad826d5f503ac03b4edc3fbc2174f21fbb005ec25863caf2051c434616d6e",
}
LEAVE_OUT = "email"
TAGS = {"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"}
OUT = paths.ROOT / "data" / "uner-en-ewt.tsv.gz"
# words that keep a sentence from being shown as an example: crude, sexual and violent words and slurs
WORDS = re.compile(r"^(fuck\w*|shit\w*|bitch\w*|damn\w*|ass|asses|asshole\w*|bastard\w*|crap\w*|dick|dicks|piss\w*|"
                   r"cunt\w*|whore\w*|slut\w*|fag\w*|nigg\w*|retard\w*|sex|sexy|sexual\w*|porn\w*|nude\w*|naked|"
                   r"rape\w*|kill|kills|killed|killing\w*|killer\w*|murder\w*|dead|death\w*|die|died|dies|dying|"
                   r"bomb\w*|terror\w*|suicid\w*|torture\w*|massacre\w*|execut\w*|behead\w*|assassinat\w*|corpse\w*|"
                   r"nazi\w*|hitler|drug|drugs|cocaine|heroin)$", re.IGNORECASE)
# tokens of an email address, a web address, a phone number or a postal code of five digits
CONTACT = re.compile(r"@|^\d{5}(-\d{4})?$|\d{3}[-.)]\d{3}|^P\.?O\.?$|^www\.|^https?:|\.(com|net|org)$", re.IGNORECASE)
# a word of a street address, which counts when the sentence also holds a number
STREET = re.compile(r"^(St|Ave|Avenue|Rd|Road|Street|Blvd|Box|Suite|Court|Drive|Lane)\.?$")


def shown(forms: list) -> bool:
    """Whether a lesson may show this sentence as an example."""
    if any(CONTACT.search(f) or WORDS.match(f) for f in forms):
        return False
    return not (any(STREET.match(f) for f in forms) and any(f.isdigit() for f in forms))


def sentences(text: str):
    """The sentences of an IOB2 file of Universal NER, each as its sent_id and its lines of words."""
    sent_id, words = None, []
    for line in text.splitlines():
        if not line:
            if words:
                yield sent_id, words
            sent_id, words = None, []
        elif line.startswith("# sent_id = "):
            sent_id = line[len("# sent_id = "):]
        elif not line.startswith("#"):
            words.append(line.split("\t"))
    if words:
        yield sent_id, words


def main(folder: str) -> None:
    rows, left, n_words, hidden = ["sent_id\tsplit\tshown\twords\ttags"], 0, 0, 0
    for split in ("train", "dev", "test"):
        path = Path(folder) / SOURCE.rsplit("/", 1)[1].format(split=split)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == SHA256[split], path
        for sent_id, words in sentences(data.decode("utf-8")):
            if sent_id.startswith(LEAVE_OUT):
                left += 1
                continue
            assert [w[0] for w in words] == [str(i) for i in range(1, len(words) + 1)], sent_id
            forms, tags = [w[1] for w in words], [w[2] for w in words]
            assert all(f and " " not in f and "\t" not in f for f in forms), sent_id
            assert set(tags) <= TAGS, sent_id
            show = shown(forms)
            rows.append("\t".join([sent_id, split, str(int(show)), " ".join(forms), " ".join(tags)]))
            n_words += len(words)
            hidden += not show
    text = "\n".join(rows) + "\n"
    with open(OUT, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(text.encode("utf-8"))
    print(OUT.name, len(rows) - 1, "sentences,", n_words, "words,", left, "left out,", hidden, "not shown,",
          OUT.stat().st_size, "bytes,", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
