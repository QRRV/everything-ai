# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson tok-pre-tokenization: chunk_pair_counts, the pairs of neighboring ids inside every
piece of a text and never across two pieces; and train_pre_bpe, byte-level BPE on the pieces that a regular expression
cuts a text into. Both are saved in the learner's project (work/projects/llm/chunk_pairs.py and pre_bpe.py) when right;
the demo compares the learner's tokens with those of byte-level BPE without pre-tokenization."""
import re
from functools import cache

from everything_ai import draw, paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

# train_pre_bpe imports merge_ids from the learner's byte_bpe.py of tok-byte-level, which imports pair_counts from their
# bpe.py of tok-bpe, which imports their merge; and chunk_pair_counts of exercise 2. Where one is missing or broken, the
# course's version stands in (template 1.3).
projects.builds_on(("llm", "merge", "tok-bpe", "merge"), ("llm", "bpe", "tok-bpe", "train-bpe"),
                   ("llm", "byte_bpe", "tok-byte-level", "train-byte-bpe"),
                   ("llm", "chunk_pairs", "tok-pre-tokenization", "chunk-pair-counts"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
TRAIN_CHARS, MERGES = 20_000, 100
PATTERN = r" ?\w+| ?[^\w\s]+|\s+"
LINE = "To be, or not to be, that is the question"


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def _merge_ids(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out


def _counts(chunks):
    counts = {}
    for ids in chunks:
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _train(chunks, n):
    merges = {}
    for step in range(n):
        counts = _counts(chunks)
        best = None
        for pair in counts:
            if best is None or counts[pair] > counts[best]:
                best = pair
        merges[best] = 256 + step
        chunks = [_merge_ids(ids, best, 256 + step) for ids in chunks]
    return merges


def pieces(text_: str) -> list:
    return re.findall(PATTERN, text_)


def encode(text_, merges, merge=_merge_ids):
    """The ids of a text with pre-tokenization: every piece encoded on its own, the ids after each other."""
    out = []
    for piece in pieces(text_):
        ids = list(piece.encode("utf-8"))
        for pair in merges:
            ids = merge(ids, pair, merges[pair])
        out = out + ids
    return out


def vocab(merges) -> dict:
    v = {i: bytes([i]) for i in range(256)}
    for pair in merges:
        v[merges[pair]] = v[pair[0]] + v[pair[1]]
    return v


def token_text(merges, i) -> str:
    return vocab(merges)[i].decode("utf-8", errors="replace")


@cache
def course_merges() -> tuple:
    """The course's byte-level BPE with pre-tokenization on the first 20,000 characters, as (pair, id) pairs."""
    return tuple(_train([list(p.encode("utf-8")) for p in pieces(text()[:TRAIN_CHARS])], MERGES).items())


@cache
def plain_merges() -> tuple:
    """The same BPE without pre-tokenization, as in tok-byte-level: the whole text is one piece."""
    return tuple(_train([list(text()[:TRAIN_CHARS].encode("utf-8"))], MERGES).items())


def crossing(token: str) -> bool:
    """A token that holds a white space after other characters: the end of one word and what comes after it."""
    return re.search(r"\S\s", token) is not None


def pieces_picture(labels: dict):
    """The line of Hamlet cut into pieces, and then into tokens by the course's 100 merge rules with pre-tokenization.
    Ink only: tokens are input."""
    merges = dict(course_merges())
    line_pieces = pieces(LINE)
    ids = encode(LINE, merges)
    rows = [(labels["pieces"].format(k=len(line_pieces)), [(p, None) for p in line_pieces]),
            (labels["tokens"].format(k=len(ids)), [(token_text(merges, i), None) for i in ids])]
    return draw.token_rows(rows, title=labels["title"], alt=labels["alt"])


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


def shown(token: str) -> str:
    """A token for the demo text, with a space as ␣ and a new line as ↵, so both can be seen."""
    return token.replace(" ", "␣").replace("\n", "↵")


# The lesson is done when these are correct. "findall" is the choice question in the lesson.
REQUIRED = ["findall", "chunk-pair-counts", "train-pre-bpe"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"

# --- exercise 2: chunk_pair_counts --------------------------------------------------------------------------------------------

_C_HEAD = "def chunk_pair_counts(chunks):\n    counts = {}\n    for ids in chunks:\n"
_C_START = _C_HEAD + "        # every pair of neighbors inside this piece, counted in counts\n        ...\n    return counts\n"
_C_LOOP = ("        for i in range(len(ids) - 1):\n            pair = (ids[i], ids[i + 1])\n            if pair in counts:\n"
           "                counts[pair] = counts[pair] + 1\n            else:\n                counts[pair] = 1\n")


def _c(loop: str = _C_LOOP, head: str = _C_HEAD) -> str:
    return head + loop + "    return counts\n"


# --- exercise 3: train_pre_bpe -----------------------------------------------------------------------------------------------

_T_HEAD = ("import re\nfrom llm.byte_bpe import merge_ids\nfrom llm.chunk_pairs import chunk_pair_counts\n\n"
           "PATTERN = r\" ?\\w+| ?[^\\w\\s]+|\\s+\"\n\n\ndef train_pre_bpe(text, n):\n")
_T_CHUNKS = "    chunks = [list(piece.encode(\"utf-8\")) for piece in re.findall(PATTERN, text)]\n"
_T_LOOP = ("    merges = {}\n    for step in range(n):\n        counts = chunk_pair_counts(chunks)\n        best = None\n"
           "        most = 0\n        for pair in counts:\n            if counts[pair] > most:\n                best = pair\n                most = counts[pair]\n"
           "        new_id = 256 + step\n        merges[best] = new_id\n")
_T_MERGE = "        chunks = [merge_ids(ids, best, new_id) for ids in chunks]\n"
_T_START = (_T_HEAD + "    # the pieces of the text, each as a list of its bytes in UTF-8\n    chunks = ...\n" + _T_LOOP
            + "        # the pair merged inside every piece\n        chunks = ...\n    return merges\n")


def _t(chunks: str = _T_CHUNKS, merge: str = _T_MERGE) -> str:
    return _T_HEAD + chunks + _T_LOOP + merge + "    return merges\n"


EXERCISES = {
    "chunk-pair-counts": FunctionBox(
        name="chunk-pair-counts",
        function="chunk_pair_counts",
        start=_C_START,
        solution=_c(),
        visible=[([[104, 105], [32, 104, 105], [33]],)],
        hidden=[([],), ([[7]],), ([[1, 1, 1]],), ([[1, 2], [2, 1]],), ([[5, 6, 5, 6], [6, 5]],)],
        python_values=True,
        save_as=("llm", "chunk_pairs"),
        first=[_DOTS],
        mistakes=[
            Mistake("new-key", error=lambda e: isinstance(e, KeyError),
                    en="The first time a pair comes by, it is not in `counts` yet, so adding 1 to it stops with an error. Look first with `if pair in counts:`; if it is not there, start it at 1.",
                    nl="De eerste keer dat een paar langskomt, staat het nog niet in `counts`, dus als je er 1 bij optelt, stopt Python met een fout. Kijk eerst met `if pair in counts:`; staat het er niet in, begin het dan bij 1."),
            Mistake("list-pair", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="A list cannot be a key of a dictionary. Make the pair a tuple, with round brackets: `(ids[i], ids[i + 1])`.",
                    nl="Een lijst kan geen key van een dictionary zijn. Maak van het paar een tuple, met ronde haakjes: `(ids[i], ids[i + 1])`."),
            Mistake("across", wrong=_c(head="def chunk_pair_counts(chunks):\n    counts = {}\n    all_ids = []\n    for ids in chunks:\n        all_ids = all_ids + ids\n    for ids in [all_ids]:\n"),
                    en=_CALL + ": you also count the pairs across two pieces, such as the last id of one and the first of the next. Count inside every piece on its own: the loop over the pairs goes inside the loop over the pieces.",
                    nl=_CALL_NL + ": je telt ook de paren over twee stukken heen, zoals het laatste id van het ene en het eerste van het volgende. Tel binnen elk stuk apart: de loop over de paren hoort binnen de loop over de stukken."),
            Mistake("last-only", wrong=_c(head="def chunk_pair_counts(chunks):\n    for ids in chunks:\n        counts = {}\n"),
                    en=_CALL + ": only the pairs of the last piece are left: every piece starts the counts again. Start `counts` once, before the loop over the pieces, as it was given.",
                    nl=_CALL_NL + ": alleen de paren van het laatste stuk blijven over: elk stuk begint de tellingen opnieuw. Begin `counts` één keer, vóór de loop over de stukken, zoals het gegeven was."),
            Mistake("from-zero", wrong=_c(loop=_C_LOOP.replace("counts[pair] = 1\n", "counts[pair] = 0\n")),
                    en=_CALL + ": every count is 1 too low. When a new pair comes by, it has been seen once already: start it at 1.",
                    nl=_CALL_NL + ": elke telling is 1 te laag. Komt er een nieuw paar langs, dan is het al één keer gezien: begin het bij 1."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with a loop over the pairs of neighbors in `ids`, each counted in `counts`: one more when it is there already, else a new key that starts at 1.",
                      "nl": "Vervang de `...` door een loop over de paren buren in `ids`, elk geteld in `counts`: één erbij als het er al staat, anders een nieuwe key die bij 1 begint."},
            "wrong": {"en": _CALL + ". Inside the loop over the pieces: a loop over `range(len(ids) - 1)`, the pair `(ids[i], ids[i + 1])`, and its count one higher. Step 3 of the recipe?",
                      "nl": _CALL_NL + ". Binnen de loop over de stukken: een loop over `range(len(ids) - 1)`, het paar `(ids[i], ids[i + 1])`, en zijn telling één hoger. Stap 3 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function work for any pieces, also a piece of one id, and no pieces at all?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Werkt je functie voor alle stukken, ook een stuk van één id, en helemaal geen stukken?"},
            "correct": {"en": "Your function counts only the pairs inside a piece: \"hi\" and \" hi\" give (104, 105) twice, and the pair from \"hi\" into \" hi\" is not counted. It is in your project now, as `{saved}`.",
                        "nl": "Je functie telt alleen de paren binnen een stuk: \"hi\" en \" hi\" geven (104, 105) twee keer, en het paar van \"hi\" naar \" hi\" wordt niet geteld. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `chunk_pair_counts` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `chunk_pair_counts` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "train-pre-bpe": FunctionBox(
        name="train-pre-bpe",
        function="train_pre_bpe",
        start=_T_START,
        solution=_t(),
        visible=[("the cat, the hat", 3)],
        hidden=[("low lower lowest", 3), ("aa aa", 2), ("hi hi hi", 2), ("To be, or not to be", 4), ("Één ijsje, één!", 3)],
        python_values=True,
        save_as=("llm", "pre_bpe"),
        first=[_DOTS],
        mistakes=[
            Mistake("one-piece", wrong=_t(chunks="    chunks = [list(text.encode(\"utf-8\"))]\n"),
                    en=_CALL + ": your text is one piece, so merges still cross from one word into the next. Cut it first: `re.findall(PATTERN, text)`, and every piece as its bytes.",
                    nl=_CALL_NL + ": je tekst is één stuk, dus merges gaan nog steeds van het ene woord naar het volgende. Knip hem eerst: `re.findall(PATTERN, text)`, en elk stuk als zijn bytes."),
            Mistake("chars", wrong=_t(chunks="    chunks = [list(piece) for piece in re.findall(PATTERN, text)]\n"),
                    en=_CALL + ": your pieces are lists of characters, not of bytes. Every piece as its bytes in UTF-8: `list(piece.encode(\"utf-8\"))`.",
                    nl=_CALL_NL + ": je stukken zijn lijsten van tekens, niet van bytes. Elk stuk als zijn bytes in UTF-8: `list(piece.encode(\"utf-8\"))`."),
            Mistake("code-points", wrong=_t(chunks="    chunks = [[ord(c) for c in piece] for piece in re.findall(PATTERN, text)]\n"),
                    en=_CALL + ": your ids are code points, such as 233 for é, not bytes. Every piece as its bytes in UTF-8: `list(piece.encode(\"utf-8\"))`.",
                    nl=_CALL_NL + ": je ids zijn code points, zoals 233 voor é, geen bytes. Elk stuk als zijn bytes in UTF-8: `list(piece.encode(\"utf-8\"))`."),
            Mistake("bytes-cut", error=lambda e: isinstance(e, TypeError) and "string pattern on a bytes-like object" in str(e),
                    en="`re.findall` with `PATTERN` cuts a text, not bytes. Cut the text first, `re.findall(PATTERN, text)`, and then turn every piece into its bytes.",
                    nl="`re.findall` met `PATTERN` knipt een tekst, geen bytes. Knip eerst de tekst, `re.findall(PATTERN, text)`, en maak dan van elk stuk zijn bytes."),
            Mistake("not-merged", wrong=_t(merge="        chunks = chunks\n"),
                    en=_CALL + ": every round finds the same pair. Merge it inside every piece, `[merge_ids(ids, best, new_id) for ids in chunks]`, and keep that as `chunks`.",
                    nl=_CALL_NL + ": elke ronde vindt hetzelfde paar. Voeg het samen binnen elk stuk, `[merge_ids(ids, best, new_id) for ids in chunks]`, en bewaar dat als `chunks`."),
            Mistake("first-piece", wrong=_t(merge="        chunks = [merge_ids(chunks[0], best, new_id)] + chunks[1:]\n"),
                    en=_CALL + ": you merge the pair in the first piece only. Merge it in every piece: a comprehension over `chunks`.",
                    nl=_CALL_NL + ": je voegt het paar alleen in het eerste stuk samen. Voeg het samen in elk stuk: een comprehension over `chunks`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the pieces of `re.findall(PATTERN, text)`, each as a list of its bytes; then every piece with the pair merged, by `merge_ids`.",
                      "nl": "Vervang de twee `...`: de stukken van `re.findall(PATTERN, text)`, elk als lijst van zijn bytes; dan elk stuk met het paar samengevoegd, door `merge_ids`."},
            "wrong": {"en": _CALL + ". The pieces of `re.findall(PATTERN, text)` as lists of bytes, and every round the pair merged in every piece: steps 1, 2 and 4 of the recipe?",
                      "nl": _CALL_NL + ". De stukken van `re.findall(PATTERN, text)` als lijsten van bytes, en elke ronde het paar samengevoegd in elk stuk: stap 1, 2 en 4 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `text`, `n` and `PATTERN`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `text`, `n` en `PATTERN`, en geen vaste getallen?"},
            "correct": {"en": "Your function learns merge rules inside the pieces: in \"the cat, the hat\", `t` and `h` become 256, then `the` 257, and `a` and `t` 258; no rule holds a comma and a space. It is in your project now, as `{saved}`.",
                        "nl": "Je functie leert merge rules binnen de stukken: in \"the cat, the hat\" worden `t` en `h` 256, dan `the` 257, en `a` en `t` 258; geen regel bevat een komma en een spatie. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `train_pre_bpe` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `train_pre_bpe` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: the learner's BPE with pre-tokenization against the same BPE without ------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when chunk_pairs.py or pre_bpe.py is not in the project yet; "bpe" when they stop with an
    error; "slow" when training takes too long; else a dict with the learner's first new tokens, the tokens of LINE,
    how many of their 100 new tokens cross from one word into the next and how many of the course's tokens without
    pre-tokenization do, the bytes per id on 5,000 characters it did not learn from, and the modules the course's
    versions stood in for."""
    if not (_saved("chunk_pairs") and _saved("pre_bpe")):
        return None
    run = run_code("from llm.pre_bpe import train_pre_bpe\nfrom llm.byte_bpe import merge_ids")
    if run.error is not None:
        return "bpe"
    train, merge = run.names["train_pre_bpe"], run.names["merge_ids"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        merges = dict(train(text()[:TRAIN_CHARS], MERGES))
        new = [token_text(merges, i) for i in merges.values()]
        plain = dict(plain_merges())
        test = text()[TRAIN_CHARS:TRAIN_CHARS + 5_000]
        return {"first": new[:12], "line": [token_text(merges, i) for i in encode(LINE, merges, merge)],
                "crossing": sum(crossing(t) for t in new),
                "plain_crossing": sum(crossing(token_text(plain, i)) for i in plain.values()),
                "per_id": len(test.encode("utf-8")) / len(encode(test, merges, merge))}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "bpe"
    out["stand_ins"] = stand_ins
    return out


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("chunk-pair-counts", _C_START, "empty", None),
    ("chunk-pair-counts", _c(), "correct", None),
    ("chunk-pair-counts", _c(loop="        for a, b in zip(ids, ids[1:]):\n            counts[(a, b)] = counts.get((a, b), 0) + 1\n"), "correct", None),
    ("chunk-pair-counts", _c(loop="        for i in range(len(ids) - 1):\n            pair = (ids[i], ids[i + 1])\n            counts[pair] = counts[pair] + 1\n"), "error", "new-key"),
    ("chunk-pair-counts", _c(loop=_C_LOOP.replace("pair = (ids[i], ids[i + 1])", "pair = [ids[i], ids[i + 1]]")), "error", "list-pair"),
    ("chunk-pair-counts", _c(head="def chunk_pair_counts(chunks):\n    counts = {}\n    all_ids = []\n    for ids in chunks:\n        all_ids = all_ids + ids\n    for ids in [all_ids]:\n"), "wrong", "across"),
    ("chunk-pair-counts", _c(head="def chunk_pair_counts(chunks):\n    for ids in chunks:\n        counts = {}\n"), "wrong", "last-only"),
    ("chunk-pair-counts", _c(loop=_C_LOOP.replace("counts[pair] = 1\n", "counts[pair] = 0\n")), "wrong", "from-zero"),
    ("chunk-pair-counts", _c(loop=_C_LOOP.replace("range(len(ids) - 1)", "range(len(ids) - 1)").replace("            pair = (ids[i], ids[i + 1])\n", "            pair = (ids[i], ids[i + 1])\n            if ids[i] == 1:\n                continue\n")), "almost", None),
    ("train-pre-bpe", _T_START, "empty", None),
    ("train-pre-bpe", _t(), "correct", None),
    ("train-pre-bpe", _t(chunks="    chunks = []\n    for piece in re.findall(PATTERN, text):\n        chunks.append(list(piece.encode(\"utf-8\")))\n",
                         merge="        new_chunks = []\n        for ids in chunks:\n            new_chunks.append(merge_ids(ids, best, new_id))\n        chunks = new_chunks\n"), "correct", None),
    ("train-pre-bpe", _t(chunks="    chunks = [list(text.encode(\"utf-8\"))]\n"), "wrong", "one-piece"),
    ("train-pre-bpe", _t(chunks="    chunks = [list(piece) for piece in re.findall(PATTERN, text)]\n"), "wrong", "chars"),
    ("train-pre-bpe", _t(chunks="    chunks = [[ord(c) for c in piece] for piece in re.findall(PATTERN, text)]\n"), "wrong", "code-points"),
    ("train-pre-bpe", _t(chunks="    chunks = [list(piece) for piece in re.findall(PATTERN, text.encode(\"utf-8\"))]\n"), "error", "bytes-cut"),
    ("train-pre-bpe", _t(merge="        chunks = chunks\n"), "wrong", "not-merged"),
    ("train-pre-bpe", _t(merge="        chunks = [merge_ids(chunks[0], best, new_id)] + chunks[1:]\n"), "wrong", "first-piece"),
    ("train-pre-bpe", _t().replace("for step in range(n):", "for step in range(3):"), "almost", None),
]
