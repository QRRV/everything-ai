# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/massive-en-nl.tsv.gz: the English (en-US) and Dutch (nl-NL) commands of MASSIVE 1.1, short commands to a
voice assistant, each with its intent (one of 60, such as alarm_set or play_music), its scenario (one of 18, such as
alarm or music) and its slots. The course uses them for micro and macro averages, where the intents are far from equally
common, and later for filling slots.

MASSIVE (FitzGerald and others, ACL 2023) was made by having professional translators localize the English commands of
SLURP into 50 languages; Amazon publishes it under CC BY 4.0, and SLURP, whose English text it includes, is CC BY 4.0
too. The course pins the tarball of version 1.1 by its SHA-256. Download
https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz and run, from the course folder:

    uv run python data/tools/make_massive.py <the tarball>

What changes: nothing in the words. Of each record the file keeps the id, the partition (train, dev or test), the
scenario, the intent, the command (`utt`) and the command with its slots marked (`annot_utt`, such as
`wake me up at [time : five am]`). It leaves out the ids of the workers, the Dutch judgments of the translations and the
way each slot was translated. The English and the Dutch file have the same ids, partitions, scenarios and intents, so
each line holds a command in both languages.

What the file holds: a header line `id<TAB>partition<TAB>scenario<TAB>intent<TAB>shown<TAB>en<TAB>nl<TAB>en_slots<TAB>
nl_slots`, then the 16,521 commands in the order of the files. `shown` is 0 for the commands a lesson does not show as an
example: swear words, insults, sexual remarks, death and violence, and politicians by name, found with the word lists
below in either language. They stay in the file, so that counts over the set stay the counts of the published set.
"""
import gzip
import hashlib
import json
import re
import sys
import tarfile

from everything_ai import paths

SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
FILES = {"en": "1.1/data/en-US.jsonl", "nl": "1.1/data/nl-NL.jsonl"}
KEPT = ("id", "partition", "scenario", "intent")
# words, in either language, that keep a command from being shown as an example
NOT_SHOWN = {
    "en": r"\b(fuck\w*|shit\w*|bitch\w*|damn|ass|asshole|bastard\w*|crap|dick|piss\w*|sex|sexy|sexiest|porn\w*|nude\w*|"
          r"naked|kill\w*|die|dies|died|dead|deadly|death|bleed|drug|drugs|terror\w*|nazi\w*|hate|gun|guns|religion|god|"
          r"trump\w*|obama\w*|putin\w*|dictator\w*|suicid\w*)\b",
    "nl": r"\b(kut\w*|klote\w*|shit\w*|lul|kanker\w*|godverdomme|verdomme|neuk\w*|hoer\w*|seks\w*|sex|porno\w*|dood|"
          r"doden|zelfmoord|haat|terror\w*|nazi\w*|drugs|wapen\w*|religie|god|trump\w*|obama\w*|poetin\w*|dictator\w*)\b",
}


def _records(tar: tarfile.TarFile, name: str) -> list:
    data = tar.extractfile(name).read().decode("utf-8")
    return [json.loads(line) for line in data.split("\n") if line]


def main(tar_path: str) -> None:
    with open(tar_path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == SHA256, "not the pinned tarball"
    with tarfile.open(tar_path, "r:gz") as tar:
        records = {lang: _records(tar, name) for lang, name in FILES.items()}
    en, nl = records["en"], records["nl"]
    assert len(en) == len(nl) == 16521
    rows = ["id\tpartition\tscenario\tintent\tshown\ten\tnl\ten_slots\tnl_slots"]
    hidden = 0
    for a, b in zip(en, nl):
        assert all(a[k] == b[k] for k in KEPT), a["id"]
        assert a["locale"] == "en-US" and b["locale"] == "nl-NL"
        texts = [a["utt"], b["utt"], a["annot_utt"], b["annot_utt"]]
        assert all(t == t.strip() and t and "\t" not in t and "\n" not in t for t in texts), a["id"]
        shown = 0 if re.search(NOT_SHOWN["en"], a["utt"].lower()) or re.search(NOT_SHOWN["nl"], b["utt"].lower()) else 1
        hidden += 1 - shown
        rows.append("\t".join([a["id"], a["partition"], a["scenario"], a["intent"], str(shown)] + texts))
    out = paths.ROOT / "data" / "massive-en-nl.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(rows) + "\n").encode("utf-8"))
    print(f"{len(rows) - 1} commands, {hidden} not shown")
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
