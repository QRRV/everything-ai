# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson tok-bpe: merge, which merges one pair of tokens everywhere in a list of tokens, and
train_bpe, which learns merge rules from a text with byte pair encoding. Both are saved in the learner's project
(work/projects/llm/merge.py and bpe.py) when right; the demo trains the learner's BPE on Tiny Shakespeare."""
import re
from functools import cache

from everything_ai import draw, paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

# train_bpe imports the merge of exercise 2; until that is saved, the course's version stands in (template 1.3).
projects.builds_on(("llm", "merge", "tok-bpe", "merge"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
TRAIN_CHARS, MERGES, TEST_CHARS = 20_000, 100, 5_000
LINE = "To be, or not to be, that is the question"


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def _merge(tokens, pair):
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(tokens[i] + tokens[i + 1])
            i = i + 2
        else:
            out.append(tokens[i])
            i = i + 1
    return out


def _pair_counts(tokens):
    counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _train(text_, n):
    tokens = list(text_)
    merges = []
    for _ in range(n):
        counts = _pair_counts(tokens)
        best = None
        for pair in counts:
            if best is None or counts[pair] > counts[best]:
                best = pair
        merges.append(best)
        tokens = _merge(tokens, best)
    return merges


def encode(text_, merges, merge=_merge):
    """The tokens of a text: its characters, with every merge rule applied in the order it was learned."""
    tokens = list(text_)
    for pair in merges:
        tokens = merge(tokens, pair)
    return tokens


@cache
def course_merges() -> tuple:
    return tuple(_train(text()[:TRAIN_CHARS], MERGES))


@cache
def line_counts() -> tuple:
    """How many tokens LINE has after the first n merge rules of the course, for every n from 0 to MERGES."""
    merges = course_merges()
    return tuple(len(encode(LINE, merges[:n])) for n in range(MERGES + 1))


def line_tokens(n: int) -> list[str]:
    """LINE after the first n merge rules of the course."""
    return encode(LINE, course_merges()[:n])


def rule(n: int) -> tuple[str, str]:
    """Merge rule n of the course, counting from 1."""
    return course_merges()[n - 1]


_VIEWBOX = re.compile(r'viewBox="0 0 (\d+) (\d+)"')


@cache
def _rows_height() -> int:
    """How tall the picture of the line's tokens is at its tallest stop (with few rules it wraps onto more lines), so
    that every stop is drawn that tall and the chart under it stays in place while you slide or drag."""
    return max(int(_VIEWBOX.search(draw.token_rows([("", [(t, None) for t in line_tokens(n)])], title="t").text)[2])
               for n in range(MERGES + 1))


def line_frame(n: int, labels: dict) -> str:
    """The line of Hamlet after the first n merge rules of the course's BPE, learned from the first 20,000 characters
    of Tiny Shakespeare, and under it how many tokens the line has for every n from 0 to 100. Drawn as SVG text, once
    for every stop of the slider (L.explore); dragging across the chart sets n. Ink only: tokens are input."""
    from everything_ai import chart
    tokens = line_tokens(n)
    rows = draw.token_rows([(labels["row"].format(n=n, k=len(tokens)), [(t, None) for t in tokens])],
                           title=labels["title"], alt=labels["alt"]).text
    rows = _VIEWBOX.sub(lambda m: f'viewBox="0 0 {m[1]} {_rows_height()}"', rows, count=1)
    counts = line_counts()
    p = chart.Panel((0, MERGES), (0, 45), title=labels["count_title"], xlabel=labels["x"], ylabel=labels["y"],
                    yticks=[0, 10, 20, 30, 40])
    p.line([n, n], [0, 45], "ink-3", 1.2)
    p.line(list(range(MERGES + 1)), list(counts), "ink", 2)
    p.points([n], [counts[n]], "ink", 4.5)
    return rows + chart.figure([p], alt=labels["count_alt"], width=440, height=220, drag=0)


# The lesson is done when these are correct. "order" is the choice question in the lesson.
REQUIRED = ["order", "merge", "train-bpe"]

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

# --- exercise 2: merge -----------------------------------------------------------------------------------------------------

_M_HEAD = ("def merge(tokens, pair):\n    out = []\n    i = 0\n    while i < len(tokens):\n"
           "        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:\n")
_M_START = (_M_HEAD + "            # the two tokens as one new token, and on past both\n            ...\n"
            "        else:\n            # this token as it is, and on to the next\n            ...\n    return out\n")
_M_YES = "            out.append(tokens[i] + tokens[i + 1])\n            i = i + 2\n"
_M_NO = "            out.append(tokens[i])\n            i = i + 1\n"


def _m(yes: str = _M_YES, no: str = _M_NO, head: str = _M_HEAD) -> str:
    return head + yes + "        else:\n" + no + "    return out\n"


# --- exercise 3: train_bpe -------------------------------------------------------------------------------------------------

_T_HEAD = ("from llm.merge import merge\n\n\ndef pair_counts(tokens):\n    counts = {}\n    for i in range(len(tokens) - 1):\n"
           "        pair = (tokens[i], tokens[i + 1])\n        if pair in counts:\n            counts[pair] = counts[pair] + 1\n"
           "        else:\n            counts[pair] = 1\n    return counts\n\n\n"
           "def train_bpe(text, n):\n    tokens = list(text)\n    merges = []\n    for step in range(n):\n"
           "        counts = pair_counts(tokens)\n")
_T_START = (_T_HEAD + "        # the most frequent pair: the first one with the highest count\n        ...\n"
            "        # keep it as a merge rule, and merge it everywhere in the tokens\n        ...\n    return merges\n")
_T_BEST = ("        best = None\n        most = 0\n        for pair in counts:\n            if counts[pair] > most:\n"
           "                best = pair\n                most = counts[pair]\n")
_T_KEEP = "        merges.append(best)\n        tokens = merge(tokens, best)\n"


def _t(best: str = _T_BEST, keep: str = _T_KEEP, head: str = _T_HEAD) -> str:
    return head + best + keep + "    return merges\n"


EXERCISES = {
    "merge": FunctionBox(
        name="merge",
        function="merge",
        start=_M_START,
        solution=_m(),
        visible=[(["l", "o", "w", " ", "l", "o", "w", "e", "r"], ("l", "o"))],
        hidden=[(["a", "a", "a"], ("a", "a")), (["a", "a", "a", "a"], ("a", "a")), (["th", "e", " ", "th", "e"], ("th", "e")),
                (["a", "b"], ("b", "a")), (["x", "y"], ("x", "y")), ([], ("x", "y"))],
        python_values=True,
        save_as=("llm", "merge"),
        first=[_DOTS],
        mistakes=[
            Mistake("end", error=lambda e: isinstance(e, IndexError),
                    en="Your function stops with an IndexError: at the last token, `tokens[i + 1]` does not exist. Look at `tokens[i + 1]` only where the given `if` has checked `i < len(tokens) - 1` first; under `else`, the token is `tokens[i]` alone.",
                    nl="Je functie stopt met een IndexError: bij het laatste token bestaat `tokens[i + 1]` niet. Kijk alleen naar `tokens[i + 1]` waar de gegeven `if` eerst `i < len(tokens) - 1` heeft gecontroleerd; onder `else` is het token alleen `tokens[i]`."),
            Mistake("one-step", wrong=_m(yes="            out.append(tokens[i] + tokens[i + 1])\n            i = i + 1\n"),
                    en=_CALL + ": after a merge, your loop goes on at the second token of the pair, so it shows up twice. A merged pair is used up: go on past both, `i = i + 2`.",
                    nl=_CALL_NL + ": na een merge gaat je loop verder bij het tweede token van het paar, dus dat komt twee keer terug. Een samengevoegd paar is opgebruikt: ga verder voorbij allebei, `i = i + 2`."),
            Mistake("pair-kept", wrong=_m(yes="            out.append(pair)\n            i = i + 2\n"),
                    en=_CALL + ": the new token is a pair, not a piece of text. The new token is the two strings put together: `tokens[i] + tokens[i + 1]`.",
                    nl=_CALL_NL + ": het nieuwe token is een paar, geen stuk tekst. Het nieuwe token is de twee strings aan elkaar: `tokens[i] + tokens[i + 1]`."),
            Mistake("first-only", wrong=_m(yes="            out.append(tokens[i] + tokens[i + 1])\n            return out + tokens[i + 2:]\n"),
                    en=_CALL + ": only the first time the pair comes by is merged. Merge it everywhere: after a merge, the loop goes on.",
                    nl=_CALL_NL + ": alleen de eerste keer dat het paar langskomt, wordt samengevoegd. Voeg het overal samen: na een merge gaat de loop verder."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: under the `if`, the new token added to `out` and `i` two further; under `else`, the token as it is and `i` one further.",
                      "nl": "Vervang de twee `...`: onder de `if` het nieuwe token toegevoegd aan `out` en `i` twee verder; onder `else` het token zoals het is en `i` één verder."},
            "wrong": {"en": _CALL + ". Under the `if`, the two tokens go into `out` as one string, and `i` moves past both; under `else`, the token goes into `out` as it is, and `i` moves on by one. Step 3 of the recipe?",
                      "nl": _CALL_NL + ". Onder de `if` gaan de twee tokens als één string in `out`, en gaat `i` voorbij allebei; onder `else` gaat het token zoals het is in `out`, en gaat `i` één verder. Stap 3 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function work for any pair, also when it overlaps itself, as in `[\"a\", \"a\", \"a\"]`?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Werkt je functie voor elk paar, ook als het zichzelf overlapt, zoals in `[\"a\", \"a\", \"a\"]`?"},
            "correct": {"en": "Your function merges a pair everywhere, from left to right, and never twice over the same token: `[\"a\", \"a\", \"a\"]` gives `[\"aa\", \"a\"]`. It is in your project now, as `{saved}`.",
                        "nl": "Je functie voegt een paar overal samen, van links naar rechts, en nooit twee keer over hetzelfde token: `[\"a\", \"a\", \"a\"]` geeft `[\"aa\", \"a\"]`. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `merge` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `merge` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "train-bpe": FunctionBox(
        name="train-bpe",
        function="train_bpe",
        start=_T_START,
        solution=_t(),
        visible=[("low lower lowest", 3)],
        hidden=[("hello hello help", 4), ("xyzxyzxy", 4), ("aaaa", 2), ("abababcab", 3), ("To be, or not to be", 5)],
        python_values=True,
        save_as=("llm", "bpe"),
        first=[_DOTS],
        mistakes=[
            Mistake("same-pair", wrong=_t(keep="        merges.append(best)\n"),
                    en=_CALL + ": every round merges the same pair. Merge it in the tokens and keep the result, `tokens = merge(tokens, best)`, so the next round counts the pairs of the new tokens.",
                    nl=_CALL_NL + ": elke ronde voegt hetzelfde paar samen. Voeg het samen in de tokens en bewaar de uitkomst, `tokens = merge(tokens, best)`, zodat de volgende ronde de paren van de nieuwe tokens telt."),
            Mistake("last-max", wrong=_t(best=_T_BEST.replace("counts[pair] > most", "counts[pair] >= most")),
                    en=_CALL + ": when two pairs have the same highest count, your loop takes the last one. Take the first, as the comment says: only a higher count replaces `best`, so `>` and not `>=`.",
                    nl=_CALL_NL + ": als twee paren dezelfde hoogste telling hebben, neemt je loop de laatste. Neem de eerste, zoals het commentaar zegt: alleen een hogere telling vervangt `best`, dus `>` en niet `>=`."),
            Mistake("last-pair", wrong=_t(best=_T_BEST.replace("                most = counts[pair]\n", "")),
                    en=_CALL + ": your loop ends with the last pair, whatever its count. A pair may only replace `best` when its count is higher than the highest so far, and then its count becomes the new highest, as `_most` in the worked example.",
                    nl=_CALL_NL + ": je loop eindigt bij het laatste paar, wat zijn telling ook is. Een paar mag `best` alleen vervangen als zijn telling hoger is dan de hoogste tot nu toe, en dan wordt zijn telling de nieuwe hoogste, zoals `_most` in het uitgewerkte voorbeeld."),
            Mistake("new-token", wrong=_t(keep="        merges.append(best[0] + best[1])\n        tokens = merge(tokens, best)\n"),
                    en=_CALL + ": you keep the new token, not the merge rule. A merge rule is the pair itself, `best`: a tokenizer needs both halves to know what to merge in a new text.",
                    nl=_CALL_NL + ": je bewaart het nieuwe token, niet de merge rule. Een merge rule is het paar zelf, `best`: een tokenizer heeft beide helften nodig om te weten wat hij in een nieuwe tekst moet samenvoegen."),
            Mistake("count", wrong=_t(keep="        merges.append(counts[best])\n        tokens = merge(tokens, best)\n"),
                    en=_CALL + ": you keep how often the pair came by, not the pair. Add `best` itself to `merges`.",
                    nl=_CALL_NL + ": je bewaart hoe vaak het paar langskwam, niet het paar. Voeg `best` zelf toe aan `merges`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the pair with the highest count in `counts`, the first one if there is a tie; then that pair added to `merges`, and merged in `tokens`.",
                      "nl": "Vervang de twee `...`: het paar met de hoogste telling in `counts`, het eerste bij gelijke stand; dan dat paar toegevoegd aan `merges`, en samengevoegd in `tokens`."},
            "wrong": {"en": _CALL + ". Every round: the first pair with the highest count, added to `merges` and merged in the tokens, so that the next round counts the new tokens. Steps 2 and 3 of the recipe, again and again?",
                      "nl": _CALL_NL + ". Elke ronde: het eerste paar met de hoogste telling, toegevoegd aan `merges` en samengevoegd in de tokens, zodat de volgende ronde de nieuwe tokens telt. Stap 2 en 3 van het recept, keer op keer?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `n` and `text`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `n` en `text`, en geen vaste getallen?"},
            "correct": {"en": "Your function learns merge rules: \"low lower lowest\" first gives `l` and `o`, then `lo` and `w`, then a space and `low`. It is in your project now, as `{saved}`.",
                        "nl": "Je functie leert merge rules: \"low lower lowest\" geeft eerst `l` en `o`, dan `lo` en `w`, dan een spatie en `low`. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `train_bpe` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `train_bpe` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: the learner's BPE on Tiny Shakespeare -----------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when merge.py or bpe.py is not in the project yet; "bpe" when they stop with an error; "slow"
    when training takes too long; else a dict with the merge rules and the merge it used, the
    characters per token on 5,000 characters it did not learn from, and the modules the course's versions stood in
    for."""
    if not (_saved("merge") and _saved("bpe")):
        return None
    run = run_code("from llm.merge import merge\nfrom llm.bpe import train_bpe")
    if run.error is not None:
        return "bpe"
    merge, train_bpe = run.names["merge"], run.names["train_bpe"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        merges = list(train_bpe(text()[:TRAIN_CHARS], MERGES))
        test = text()[TRAIN_CHARS:TRAIN_CHARS + TEST_CHARS]
        return {"merges": merges, "merge": merge,
                "per_token": len(test) / len(encode(test, merges, merge))}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "bpe"
    out["stand_ins"] = stand_ins
    return out


def tokenizer(result: dict):
    """For the Tokens block of the demo: a text to the ids of its tokens, with the merge rules and the merge of the
    demo, and an id back to its token. The ids are those of the vocabulary: the characters of the text BPE learned
    from, then one token per merge rule; a character it never saw gets the next free id."""
    merges, merge = result["merges"], result["merge"]
    vocab = sorted(set(text()[:TRAIN_CHARS])) + [a + b for a, b in merges]
    index = {}
    for i, token in enumerate(vocab):
        index.setdefault(token, i)

    def ids(s: str) -> list[int]:
        out = []
        for token in encode(s, merges, merge):
            if token not in index:
                index[token] = len(vocab)
                vocab.append(token)
            out.append(index[token])
        return out

    return ids, lambda some: "".join(vocab[i] for i in some)


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


def shown(token: str) -> str:
    """A token for the demo text, with a space as ␣ and a new line as ↵, so both can be seen."""
    return token.replace(" ", "␣").replace("\n", "↵")


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("merge", _M_START, "empty", None),
    ("merge", _m(), "correct", None),
    ("merge", _m(yes="            out.append(pair[0] + pair[1])\n            i += 2\n", no="            out.append(tokens[i])\n            i += 1\n"), "correct", None),
    ("merge", _m(head=_M_HEAD.replace("i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]",
                                      "tokens[i] == pair[0] and tokens[i + 1] == pair[1]")), "error", "end"),
    ("merge", _m(yes="            out.append(tokens[i] + tokens[i + 1])\n            i = i + 1\n"), "wrong", "one-step"),
    ("merge", _m(yes="            out.append(pair)\n            i = i + 2\n"), "wrong", "pair-kept"),
    ("merge", _m(yes="            out.append(tokens[i] + tokens[i + 1])\n            return out + tokens[i + 2:]\n"), "wrong", "first-only"),
    ("merge", _m(yes="            out.append(\"lo\")\n            i = i + 2\n"), "almost", None),
    ("train-bpe", _T_START, "empty", None),
    ("train-bpe", _t(), "correct", None),
    ("train-bpe", _t(best="        best = max(counts, key=counts.get)\n"), "correct", None),
    ("train-bpe", _t(keep="        merges.append(best)\n"), "wrong", "same-pair"),
    ("train-bpe", _t(keep="        merges.append(best)\n        merge(tokens, best)\n"), "wrong", "same-pair"),
    ("train-bpe", _t(best="        best = None\n        for pair in counts:\n            if best is None or counts[pair] > counts[best]:\n"
                          "                best = pair\n"), "correct", None),
    ("train-bpe", _t(best=_T_BEST.replace("counts[pair] > most", "counts[pair] >= most")), "wrong", "last-max"),
    ("train-bpe", _t(best=_T_BEST.replace("                most = counts[pair]\n", "")), "wrong", "last-pair"),
    ("train-bpe", _t(best="        for pair in counts:\n            best = pair\n"), "wrong", "last-pair"),
    ("merge", _m(no="            out.append(tokens[i] + tokens[i + 1])\n            i = i + 1\n"), "error", "end"),
    ("train-bpe", _t(keep="        merges.append(best[0] + best[1])\n        tokens = merge(tokens, best)\n"), "wrong", "new-token"),
    ("train-bpe", _t(keep="        merges.append(counts[best])\n        tokens = merge(tokens, best)\n"), "wrong", "count"),
    ("train-bpe", _t(head=_T_HEAD.replace("for step in range(n):", "for step in range(3):")), "almost", None),
]
