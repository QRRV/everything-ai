# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-tm-matches.tsv: for each of the 3,707 sentences of nlp-bleu, from which nlp-post-editing takes
its sentences, the Tatoeba id of its best match in the lesson's translation memory, by the fuzzy match of their Dutch
words (one minus the edit distance divided by the longer length), and of equal matches the first in the memory. The
lesson's own search finds them (checks.search). A header line `id<TAB>match`, then one sentence per line, in the
order of nlp-bleu. Run from the course folder:

    uv run python data/tools/make_course_tm_matches.py
"""
import importlib.util

from everything_ai import paths

OUT = paths.ROOT / "data" / "course-tm-matches.tsv"


def main() -> None:
    spec = importlib.util.spec_from_file_location("nlp_post_editing", paths.LESSONS / "nlp-post-editing" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    memory = C._memory()
    lines = ["id\tmatch"]
    for sid, text, _ in C._all_sentences():
        (score, i), = C.search(C.dutch_words(text), 1)
        lines.append(f"{sid}\t{memory[i][0]}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines) - 1, "sentences,", len(memory), "in the memory")


if __name__ == "__main__":
    main()
