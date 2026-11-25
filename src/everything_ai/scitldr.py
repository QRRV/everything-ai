# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The test set of SciTLDR-A, by Isabel Cachola, Kyle Lo, Arman Cohan and Daniel S. Weld (2020): 618 papers in computer
science from OpenReview, each with its title, its abstract split into sentences, and one or more TLDRs, a summary of
one sentence: the first by the authors, the others written by experts from the peer reviews. Each abstract marks its
oracle sentence: the sentence with the highest ROUGE-1 against the TLDRs. The set is under Apache-2.0 at
github.com/allenai/scitldr (course/route/bronnen.yaml, scitldr): everything_ai.hub downloads the file once, 1.2 MB, from
a fixed commit, and keeps it only when its SHA-256 is the pinned one. `papers` reads it as text, one JSON object per
line: nothing in it is ever run."""
import functools
import json

from everything_ai import hub

FILE = ("github.com/allenai/scitldr", "test.jsonl")
_MOST = 3_000_000   # bytes the course reads at most; the file has 1,204,107


def ready() -> bool:
    """Whether the file is downloaded."""
    return hub.pinned(*FILE) is not None


@functools.cache
def _papers() -> tuple:
    path = hub.pinned(*FILE)
    if path is None:
        raise FileNotFoundError("SciTLDR is not downloaded yet: everything_ai.hub.download(*everything_ai.scitldr.FILE)")
    if path.stat().st_size > _MOST:
        raise ValueError("the SciTLDR file is larger than the course expects")
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            sentences = tuple(str(s) for s in row["source"])
            labels = list(row["source_labels"])
            out.append((str(row["paper_id"]), str(row["title"]), sentences, tuple(str(t) for t in row["target"]),
                        labels.index(1) if 1 in labels else -1))
    return tuple(out)


def papers() -> list:
    """Every paper of the test set, in the order of the file, as a new list of dictionaries with "id", "title",
    "sentences" (a list of the sentences of the abstract), "tldrs" (a list, the authors' TLDR first) and "oracle" (the
    place of the oracle sentence, counted from 0). FileNotFoundError when the file is not downloaded yet:
    hub.download(*FILE) gets it."""
    return [{"id": i, "title": t, "sentences": list(s), "tldrs": list(d), "oracle": o} for i, t, s, d, o in _papers()]
