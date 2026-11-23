# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-wordnet: a sort of pairs of words of a small lexicon by their path similarity, every
hypernym of a sense with its fewest steps up by breadth-first search (ancestors), and the path similarity of two senses
through the hypernym they share (path_similarity).

WordNet is Open English WordNet 2025, which the course ships (data/wordnet-en.tsv.gz, read by everything_ai.wordnet),
with the Dutch words of Open Dutch WordNet beside it; the cosine similarities come from GloVe
(everything_ai.wordvectors). The kitchen lexicon and the small dicts of the cases are the course's own; every number in
the text and the pictures is computed by the course (tests/test_nlp_wordnet.py computes them again)."""
import contextlib
import functools
import html
import io
import math
from collections import deque

import numpy as np

from everything_ai import wordnet, wordvectors, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

_KEY = "nlp-wordnet"

# --- the course's own functions ---------------------------------------------------------------------------------------


def ancestors(name, hypernyms) -> dict:
    """A dict from `name` and every sense above it to its fewest steps up from `name`: breadth-first search."""
    steps = {name: 0}
    queue = deque([name])
    while len(queue) > 0:
        node = queue.popleft()
        for up in hypernyms[node]:
            if up not in steps:
                steps[up] = steps[node] + 1
                queue.append(up)
    return steps


def path_similarity(a, b, hypernyms):
    """1 / (steps + 1) for the shortest path from a up to a hypernym the two share and down to b; None without one."""
    up_a = ancestors(a, hypernyms)
    up_b = ancestors(b, hypernyms)
    steps = []
    for name in up_a:
        if name in up_b:
            steps.append(up_a[name] + up_b[name])
    if len(steps) == 0:
        return None
    return 1 / (min(steps) + 1)


# --- the data -----------------------------------------------------------------------------------------------------------


def hypernyms() -> dict:
    """A new dict from every synset of WordNet to the list of its hypernyms: 107,425 synsets, with [] for a top, an
    adjective and an adverb."""
    return {name: list(s.hypernyms) for name, s in wordnet.synsets().items()}


HYPERNYMS = hypernyms()

# The kitchen lexicon of the lesson: every word with its one hypernym, in English and in Dutch.
KITCHEN = {
    "en": {"thing": [], "utensil": ["thing"], "food": ["thing"], "cutlery": ["utensil"], "pan": ["utensil"],
           "drink": ["food"], "soup": ["food"], "fork": ["cutlery"], "spoon": ["cutlery"], "knife": ["cutlery"],
           "tea": ["drink"], "coffee": ["drink"]},
    "nl": {"ding": [], "keukengerei": ["ding"], "eten": ["ding"], "bestek": ["keukengerei"], "pan": ["keukengerei"],
           "drank": ["eten"], "soep": ["eten"], "vork": ["bestek"], "lepel": ["bestek"], "mes": ["bestek"],
           "thee": ["drank"], "koffie": ["drank"]},
}
EXAMPLE = {"en": ("coffee", "soup"), "nl": ("koffie", "soep")}   # the two words of the worked example


@functools.lru_cache(maxsize=8192)
def _up(name: str) -> tuple:
    """The senses above a sense of WordNet with their fewest steps, as pairs; kept for the explore."""
    return tuple(ancestors(name, HYPERNYMS).items())


def up_paths(name, hypernyms) -> dict:
    """For `name` and every sense above it: the path from `name` up to it with the fewest steps, the first that
    breadth-first search finds. A hypernym that is not a key of `hypernyms` counts as a top."""
    paths = {name: [name]}
    queue = deque([name])
    while len(queue) > 0:
        node = queue.popleft()
        for up in hypernyms.get(node, []):
            if up not in paths:
                paths[up] = paths[node] + [up]
                queue.append(up)
    return paths


def shortest_path(a, b, hypernyms):
    """(the shared hypernym, the path from a up to it, the path from b up to it) for the shortest path between a and b,
    the first of the shortest in the order of the senses above a; None when they share no hypernym."""
    pa, pb = up_paths(a, hypernyms), up_paths(b, hypernyms)
    best = None
    for name in pa:
        if name in pb:
            n = len(pa[name]) + len(pb[name]) - 2
            if best is None or n < best[0]:
                best = (n, name)
    if best is None:
        return None
    return best[1], pa[best[1]], pb[best[1]]


def _glove() -> tuple:
    def make():
        words, vectors = wordvectors.glove()
        return {w: i for i, w in enumerate(words)}, vectors
    return kept(f"{_KEY}:glove", make)


def cosine(a: str, b: str):
    """The cosine similarity of two words in GloVe, in lowercase; None when GloVe lacks one of them."""
    row, vectors = _glove()
    a, b = a.lower(), b.lower()
    if a not in row or b not in row:
        return None
    x, y = vectors[row[a]], vectors[row[b]]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


# --- the look of the pictures -------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_SMALL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"


def _marked(text: str) -> str:
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code, as
    in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{html.escape(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _plain_word(word) -> str:
    """A typed word for a note: without asterisks and backticks, which the note reads as italics and code."""
    return str(word).replace("*", "").replace("`", "")


def _int(n: int, lang: str) -> str:
    """A whole number with a separator per thousand: 107,425 in English, 107.425 in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x + 0.0:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = (), shade: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; the rows in
    `shade` get a gray band."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join(
        f'<tr style="{"background:var(--ea-data-wash)" if i in shade else ""}">'
        + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>' for j, c in enumerate(row))
        + "</tr>" for i, row in enumerate(rows_))
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _bar(value: float, style: str, scale: float = 140) -> str:
    width = max(0.0, min(1.0, value)) * scale
    return f'<span style="{_BAR};width:{width:.0f}px;background:{style}"></span>'


def _short(value, limit: int = 60) -> str:
    try:
        text = repr(value)
    except Exception:   # noqa: BLE001   a value whose repr stops with an error
        text = "?"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _text(value) -> str:
    """A name as a picture shows it: a text as it is, anything else as its short repr."""
    return value if isinstance(value, str) else _short(value, 40)


def _dutch(name: str) -> str:
    """The first Dutch words of a sense of WordNet, in italics, or nothing."""
    s = wordnet.synsets().get(name)
    if s is None or not s.dutch:
        return ""
    return f' <i style="{_SMALL}">{html.escape(", ".join(s.dutch[:2]))}</i>'


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v) -> bool:
    """A number a picture can draw: not inf or nan, and not an int too large for a float."""
    try:
        return _is_number(v) and math.isfinite(float(v))
    except (OverflowError, ValueError, TypeError):
        return False


def _whole(v) -> bool:
    return _finite(v) and float(v).is_integer()


def _same(got, want) -> bool:
    """A similarity as the check compares it: None and None, or two numbers that differ by less than 1e-9."""
    if want is None:
        return got is None
    return _finite(got) and abs(got - want) < 1e-9


# --- the path between two senses, as two ladders that meet at the top ---------------------------------------------------

_PATH_WORDS = {
    "en": {"steps": "{n} steps: {x} up from {a} and {y} up from {b} to {meet}. Path similarity: 1/{d} = {s}.",
           "same": "{a} and {b} are the same sense: 0 steps, path similarity 1.",
           "none": "{a} and {b} share no hypernym: no path joins them, and the path similarity is None."},
    "nl": {"steps": "{n} stappen: {x} omhoog vanaf {a} en {y} omhoog vanaf {b} naar {meet}. Path similarity: 1/{d} = {s}.",
           "same": "{a} en {b} zijn dezelfde betekenis: 0 stappen, path similarity 1.",
           "none": "{a} en {b} delen geen hyperoniem: er is geen pad, en de path similarity is None."},
}


def path_table(a, b, hypernyms, lang: str, style: str = _CHIP_DATA, wordnet_names: bool = True) -> tuple:
    """The shortest path between a and b as a table of two columns, from the shared hypernym at the top, on a gray
    band, down to a and to b; and a line with its steps and its path similarity. Gives (html, line)."""
    w = _PATH_WORDS[lang]
    found = shortest_path(a, b, hypernyms)
    if found is None:
        line = w["none"].format(a=a, b=b)
        return _label(line), line
    meet, pa, pb = found
    n = len(pa) + len(pb) - 2
    if n == 0:
        line = w["same"].format(a=a, b=b)
        return _label(line), line

    def rung(name):
        return _chip(str(name), style) + (_dutch(name) if wordnet_names else "")
    left, right = pa[::-1][1:], pb[::-1][1:]
    tall = max(len(left), len(right))
    rows = [[rung(meet), ""]]
    for i in range(tall):
        rows.append([rung(left[i]) if i < len(left) else "", rung(right[i]) if i < len(right) else ""])
    line = w["steps"].format(n=n, x=len(pa) - 1, y=len(pb) - 1, a=a, b=b, meet=meet, d=n + 1,
                             s=_dec(1 / (n + 1), lang))
    return _table(rows, [str(a), str(b)], shade=(0,)) + _label(line), line


# --- the idea's pictures: two paths in WordNet, and the kitchen lexicon --------------------------------------------------

IDEA_PAIRS = [("coffee.n.01", "tea.n.01"), ("coffee.n.01", "cup.n.01")]

_IDEA_WORDS = {
    "en": {"title": "Two paths in WordNet: coffee and tea, coffee and cup",
           "legend": "Gray: the senses of WordNet. Each table is one shortest path: from the hypernym the two senses "
                     "share, on the gray band at the top, down to each of them. Beside each name, in italics: its "
                     "Dutch words."},
    "nl": {"title": "Twee paden in WordNet: koffie en thee, koffie en kop",
           "legend": "Grijs: de betekenissen van WordNet. Elke tabel is één kortste pad: van het hyperoniem dat de "
                     "twee betekenissen delen, op de grijze band bovenaan, omlaag naar elk van de twee. Naast elke "
                     "naam, schuin gedrukt: de Nederlandse woorden."},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    body, lines = "", []
    for a, b in IDEA_PAIRS:
        table, line = path_table(a, b, HYPERNYMS, lang)
        body += table
        lines.append(line)
    return yours.figure(w["title"], body + _note(w["legend"]), alt=f'{w["title"]}. ' + " ".join(lines))


_TREE_WORDS = {
    "en": {"also": "also under {names}",
           "loose": "Not below any top, because they are above themselves: {names}.",
           "cut": "And {n} more lines.",
           "kitchen": "The kitchen lexicon of this lesson, from its top down",
           "kitchen_legend": "Gray: the lexicon's words. Each word stands one step in from its hypernym: *thing* is "
                             "the top, and *coffee* is a *drink*, a *food* and a *thing*."},
    "nl": {"also": "ook onder {names}",
           "loose": "Niet onder een top, omdat de woorden boven zichzelf staan: {names}.",
           "cut": "En nog {n} regels.",
           "kitchen": "Het keukenlexicon van deze les, van de top omlaag",
           "kitchen_legend": "Grijs: de woorden van het lexicon. Elk woord staat één stap ingesprongen onder het "
                             "hyperoniem: *ding* is de top, en *koffie* is een *drank*, *eten* en een *ding*."},
}
TREE_LINES = 200   # lines of a tree that a picture draws
TREE_VISITS = 5000   # places in a tree that a picture counts: a lexicon with many words of several hypernyms has many


def tree(lexicon: dict, lang: str, style: str = _CHIP_DATA) -> str:
    """A lexicon from its tops down: each word one step in under each of its hypernyms. A word with more than one
    hypernym stands under each, with a note; a word above itself is listed apart. The lexicon is checked before."""
    w = _TREE_WORDS[lang]
    below = {}
    for word, ups in lexicon.items():
        for up in ups:
            below.setdefault(up, []).append(word)
    tops = [word for word, ups in lexicon.items() if not [u for u in ups if u in lexicon]]
    lines, seen, total = [], set(), [0]

    def down(word, depth, path):
        total[0] += 1
        seen.add(word)
        if len(lines) >= TREE_LINES:
            if total[0] < TREE_VISITS:
                for child in below.get(word, []):
                    if child not in path and child != word and total[0] < TREE_VISITS:
                        down(child, depth + 1, path + [word])
            return
        others = [u for u in lexicon.get(word, []) if u in lexicon and (not path or u != path[-1])]
        also = f' <span style="{_SMALL}">{html.escape(w["also"].format(names=", ".join(map(str, others))))}</span>' if others else ""
        lines.append(f'<div style="padding-left:{1.2 * depth:.1f}rem">{_chip(str(word), style)}{also}</div>')
        for child in below.get(word, []):
            if child not in path and child != word and total[0] < TREE_VISITS:
                down(child, depth + 1, path + [word])

    for top in tops:
        down(top, 0, [])
    body = "".join(lines)
    if total[0] > len(lines):
        more = _int(total[0] - len(lines), lang) + ("+" if total[0] >= TREE_VISITS else "")
        body += _note(w["cut"].format(n=more))
    loose = [str(word) for word in lexicon if word not in seen]
    if loose:
        body += _note(w["loose"].format(names=", ".join(loose[:20])))
    return body


def kitchen_picture(lang: str) -> str:
    w = _TREE_WORDS[lang]
    words = KITCHEN[lang]
    return yours.figure(w["kitchen"], tree(words, lang) + _note(w["kitchen_legend"]),
                        alt=f'{w["kitchen"]}: ' + "; ".join(f"{k}: {', '.join(v) or '-'}" for k, v in words.items()))


# --- running the learner's function in a picture ------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 1   # what the learner's function gets per call in a picture


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit, on shallow copies (the lists inside are copied too, for a
    dict of lists); _STOPPED when it stops with an error, and the Timeout passes on."""
    def copy(a):
        if isinstance(a, dict):
            return {k: list(v) if isinstance(v, list) else v for k, v in a.items()}
        return a
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


# --- the picture under exercise 2: every sense above dog and house cat, by your ancestors -------------------------------

ANCESTOR_SENSES = ["dog.n.01", "domestic_cat.n.01"]
CHIPS = 30   # senses a row of the picture shows

_ANCESTOR_WORDS = {
    "en": {"title": "Every sense above dog and house cat, by your function",
           "head": ["Steps up", "Senses"],
           "missing": "Missing from your dict: {names}.",
           "more": "and {n} more",
           "share": "The two dicts share {n} senses. The nearest: {name}, {x} up from dog.n.01 and {y} up from "
                    "domestic_cat.n.01: {s} steps together.",
           "share_none": "The two dicts share no sense.",
           "legend": "Each table is what your function gives with `C.HYPERNYMS`, grouped by the number of steps. "
                     "Blue: a sense at the same number of steps as in the course's dict; purple: a sense at another "
                     "number, or one that is not above the sense at all. Beside each name, in italics: its Dutch words.",
           "error": "Your function stops with an error on `ancestors(\"{name}\", C.HYPERNYMS)`.",
           "slow": "Your function is still running after {s} second with `ancestors(\"{name}\", C.HYPERNYMS)`.",
           "odd": "With `ancestors(\"{name}\", C.HYPERNYMS)`, your function gives back {got}, and not a dict."},
    "nl": {"title": "Elke betekenis boven hond en huiskat, door je functie",
           "head": ["Stappen omhoog", "Betekenissen"],
           "missing": "Ontbreekt in je dict: {names}.",
           "more": "en nog {n}",
           "share": "De twee dicts delen {n} betekenissen. De dichtstbijzijnde: {name}, {x} omhoog vanaf dog.n.01 "
                    "en {y} omhoog vanaf domestic_cat.n.01: samen {s} stappen.",
           "share_none": "De twee dicts delen geen betekenis.",
           "legend": "Elke tabel is wat je functie met `C.HYPERNYMS` geeft, gegroepeerd naar het aantal stappen. "
                     "Blauw: een betekenis op hetzelfde aantal stappen als in de dict van de course; paars: een "
                     "betekenis op een ander aantal, of een betekenis die er helemaal niet boven staat. Naast elke "
                     "naam, schuin gedrukt: de Nederlandse woorden.",
           "error": "Je functie stopt met een fout op `ancestors(\"{name}\", C.HYPERNYMS)`.",
           "slow": "Je functie is na {s} seconde nog bezig met `ancestors(\"{name}\", C.HYPERNYMS)`.",
           "odd": "Met `ancestors(\"{name}\", C.HYPERNYMS)` geeft je functie {got} terug, en geen dict."},
}


def ancestors_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _ANCESTOR_WORDS[lang]
    body, notes, alts, dicts = "", [], [], []
    for name in ANCESTOR_SENSES:
        try:
            got = _call(fn, name, HYPERNYMS)
        except Timeout:
            notes.append(w["slow"].format(name=name, s=PICTURE_SECONDS))
            break
        if got is _STOPPED:
            notes.append(w["error"].format(name=name))
            break
        if not isinstance(got, dict):
            notes.append(w["odd"].format(name=name, got=_short(got)))
            break
        course = ancestors(name, HYPERNYMS)
        groups, odd = {}, []
        for k, v in got.items():
            if _whole(v) and 0 <= v <= 99:
                groups.setdefault(int(v), []).append(k)
            else:
                odd.append(k)
        rows = []
        for step in sorted(groups):
            ks = sorted(groups[step], key=_short)
            chips = "".join(_chip(_text(k), _CHIP_PRED if course.get(k) == step else _CHIP_PURPLE)
                            + (_dutch(k) if isinstance(k, str) else "") for k in ks[:CHIPS])
            if len(ks) > CHIPS:
                chips += f' <span style="{_SMALL}">{html.escape(w["more"].format(n=_int(len(ks) - CHIPS, lang)))}</span>'
            rows.append([str(step), chips])
        if odd:
            rows.append(["?", "".join(_chip(f"{_short(k, 40)}: {_short(got[k], 20)}", _CHIP_PURPLE) for k in odd[:12])])
        body += _label(name) + _table(rows, w["head"], numbers=(0,))
        missing = [k for k in course if k not in got]
        if missing:
            body += _note(w["missing"].format(names=", ".join(missing[:20]) + (", …" if len(missing) > 20 else "")))
        alts.append(f"{name}: " + "; ".join(f"{s}: {', '.join(_text(k) for k in sorted(groups[s], key=_short)[:CHIPS])}"
                                            for s in sorted(groups)[:20]))
        dicts.append({k: v for k, v in got.items() if _finite(v)})
    if len(dicts) == 2:
        shared = sorted(((dicts[0][k] + dicts[1][k], _text(k), k) for k in dicts[0] if k in dicts[1]),
                        key=lambda t: (t[0], t[1]))
        if shared:
            s, name, key = shared[0]
            x = dicts[0][key]
            line = w["share"].format(n=len(shared), name=name, x=_num(x, lang), y=_num(s - x, lang), s=_num(s, lang))
        else:
            line = w["share_none"]
        body += _label(line)
        alts.append(line)
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}. ' + " ".join(alts) + " " + _bare(text))


def _num(x, lang: str) -> str:
    """A number of steps as a picture writes it: whole numbers without decimals."""
    if _whole(x):
        return _int(int(x), lang)
    return _dec(x, lang) if _finite(x) else _short(x, 20)


# --- the picture under exercise 3: pairs that are alike and pairs that are related --------------------------------------

PAIRS = [
    ("alike", [("car.n.01", "truck.n.01"), ("coffee.n.01", "tea.n.01"), ("dog.n.01", "wolf.n.01"),
               ("doctor.n.01", "nurse.n.01")]),
    ("related", [("car.n.01", "driver.n.01"), ("teacher.n.01", "school.n.01"), ("doctor.n.01", "hospital.n.01"),
                 ("coffee.n.01", "cup.n.01")]),
]

_PAIR_WORDS = {
    "en": {"title": "Alike or related: your path similarity beside GloVe's cosine similarity",
           "head": ["Senses", "Your path similarity", "Cosine similarity in GloVe"],
           "groups": {"alike": "Alike: one is a kind of thing like the other",
                      "related": "Related: they go together, but are different kinds of things"},
           "legend": "Blue: what your function gives with `C.HYPERNYMS`; purple where it differs from the course's "
                     "path similarity. Gray: the cosine similarity of the two words in GloVe, for its one vector per "
                     "word, whatever the sense. On the gray bands: the two kinds of pairs.",
           "error": "Your function stops with an error on `path_similarity(\"{a}\", \"{b}\", C.HYPERNYMS)`.",
           "slow": "Your function is still running after {s} second with "
                   "`path_similarity(\"{a}\", \"{b}\", C.HYPERNYMS)`."},
    "nl": {"title": "Gelijkend of verwant: je path similarity naast de cosine similarity van GloVe",
           "head": ["Betekenissen", "Jouw path similarity", "Cosine similarity in GloVe"],
           "groups": {"alike": "Gelijkend: het ene is een soort ding zoals het andere",
                      "related": "Verwant: de twee horen bij elkaar, maar zijn verschillende soorten dingen"},
           "legend": "Blauw: wat je functie met `C.HYPERNYMS` geeft; paars waar dat verschilt van de path similarity "
                     "van de course. Grijs: de cosine similarity van de twee woorden in GloVe, voor de ene vector "
                     "per woord, welke betekenis ook bedoeld is. Op de grijze banden: de twee soorten paren.",
           "error": "Je functie stopt met een fout op `path_similarity(\"{a}\", \"{b}\", C.HYPERNYMS)`.",
           "slow": "Je functie is na {s} seconde nog bezig met `path_similarity(\"{a}\", \"{b}\", C.HYPERNYMS)`."},
}


def _word_of(name: str) -> str:
    return name.split(".")[0]


def similarity_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _PAIR_WORDS[lang]
    rows, alts, notes, group_rows = [], [], [], []
    stopped = False
    for group, pairs in PAIRS:
        group_rows.append(len(rows))
        rows.append([f"<b>{html.escape(w['groups'][group])}</b>", "", ""])
        for a, b in pairs:
            try:
                got = _call(fn, a, b, HYPERNYMS)
            except Timeout:
                notes.append(w["slow"].format(a=a, b=b, s=PICTURE_SECONDS))
                stopped = True
                break
            if got is _STOPPED:
                notes.append(w["error"].format(a=a, b=b))
                stopped = True
                break
            want = path_similarity(a, b, HYPERNYMS)
            color = "var(--ea-pred)" if _same(got, want) else "var(--ea-loss)"
            if _finite(got):
                mine = f'{_bar(got, color)} <span style="font-variant-numeric:tabular-nums">{_dec(got, lang)}</span>'
            else:
                mine = _chip(_short(got, 30), _CHIP_PURPLE)
            c = cosine(_word_of(a), _word_of(b))
            theirs = (f'{_bar(c, "var(--ea-data)")} <span style="font-variant-numeric:tabular-nums">{_dec(c, lang)}</span>'
                      if c is not None else "")
            rows.append([f"<code>{html.escape(a)}</code> · <code>{html.escape(b)}</code>", mine, theirs])
            alts.append(f"{a} and {b}: {_short(got, 30)}, GloVe {_dec(c, lang) if c is not None else '-'}")
        if stopped:
            break
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], _table(rows, w["head"], shade=tuple(group_rows)) + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- try it: a lexicon of your own --------------------------------------------------------------------------------------

LARGEST = 500   # words of a lexicon that the picture takes

_LEXICON_WORDS = {
    "en": {"title": "Your lexicon, from its tops down",
           "path": "The shortest path between {a} and {b}",
           "legend": "Blue: the words of your lexicon, each one step in from its hypernym. In the path, on the "
                     "gray band at the top: the word where the two meet.",
           "not_dict": "`my_lexicon` is {got}, and not a dict. Make it a dict from each word to the list of its "
                       "hypernyms.",
           "not_list": "The hypernyms of *{word}* are {got}, and not a list of words: write them between [ and ].",
           "unknown": "*{word}* has the hypernym *{up}*, which is not a word of the lexicon. Add it, such as "
                      "`\"{up}\": []` for a new top.",
           "loop": "*{word}* is above itself, through its hypernyms: the picture leaves it out of the paths.",
           "big": "The lexicon has {n} words; the picture takes the first {k}.",
           "empty": "The lexicon is empty. Add a word, such as `\"thing\": []`.",
           "choose": "Choose two words of the lexicon."},
    "nl": {"title": "Je lexicon, van de toppen omlaag",
           "path": "Het kortste pad tussen {a} en {b}",
           "legend": "Blauw: de woorden van je lexicon, elk één stap ingesprongen onder het hyperoniem. In het "
                     "pad, op de grijze band bovenaan: het woord waar de twee samenkomen.",
           "not_dict": "`my_lexicon` is {got}, en geen dict. Maak er een dict van, van elk woord naar de lijst met "
                       "hyperoniemen.",
           "not_list": "De hyperoniemen van *{word}* zijn {got}, en geen lijst met woorden: zet de hyperoniemen tussen [ en ].",
           "unknown": "*{word}* heeft het hyperoniem *{up}*, en dat is geen woord van het lexicon. Voeg het toe, "
                      "zoals `\"{up}\": []` voor een nieuwe top.",
           "loop": "*{word}* staat boven zichzelf, via de hyperoniemen: de afbeelding laat het woord weg uit de paden.",
           "big": "Het lexicon heeft {n} woorden; de afbeelding neemt de eerste {k}.",
           "empty": "Het lexicon is leeg. Voeg een woord toe, zoals `\"ding\": []`.",
           "choose": "Kies twee woorden van het lexicon."},
}


def clean_lexicon(lexicon, lang: str) -> tuple:
    """(a dict from each word to the list of its hypernyms that are words of the lexicon, the notes on what was left
    out), or (None, notes) for what is no dict at all."""
    w = _LEXICON_WORDS[lang]
    if not isinstance(lexicon, dict):
        return None, [w["not_dict"].format(got=_plain_word(_short(lexicon, 40)))]
    notes, out = [], {}
    items = list(lexicon.items())
    if len(items) > LARGEST:
        notes.append(w["big"].format(n=_int(len(items), lang), k=_int(LARGEST, lang)))
        items = items[:LARGEST]
    keys = [k for k, _ in items]
    for word, ups in items:
        if not isinstance(ups, (list, tuple)) or not all(isinstance(u, (str, int, float)) for u in ups):
            notes.append(w["not_list"].format(word=_plain_word(word), got=_plain_word(_short(ups, 40))))
            out[word] = []
            continue
        kept_ups = []
        for up in ups:
            if up in keys:
                if up not in kept_ups:
                    kept_ups.append(up)
            else:
                notes.append(w["unknown"].format(word=_plain_word(word), up=_plain_word(up)))
        out[word] = kept_ups
    for word in out:
        above = ancestors(word, out)
        if any(word in out[x] for x in above if x != word) or word in out[word]:
            notes.append(w["loop"].format(word=_plain_word(word)))
    return out, notes


def lexicon_words(lexicon) -> list:
    """The words of a lexicon for the two menus, as text, in the lexicon's order; [] for what is no dict."""
    if not isinstance(lexicon, dict):
        return []
    return list(dict.fromkeys(str(k) for k in list(lexicon)[:LARGEST]))


def first_choice(words: list, lang: str):
    """The word a menu starts with: the first word of the worked example where the lexicon has it."""
    a, _ = EXAMPLE[lang]
    return a if a in words else (words[0] if words else None)


def second_choice(words: list, lang: str):
    """The word the other menu starts with: the second word of the worked example where the lexicon has it."""
    _, b = EXAMPLE[lang]
    return b if b in words else (words[-1] if words else None)


def lexicon_picture(lexicon, first, second, lang: str) -> str:
    w = _LEXICON_WORDS[lang]
    words, notes = clean_lexicon(lexicon, lang)
    if words is None:
        return yours.figure(w["title"], _note(notes[0]), alt=_bare(notes[0]))
    if not words:
        return yours.figure(w["title"], _note(w["empty"]), alt=_bare(w["empty"]))
    by_text = {str(k): k for k in words}
    looped = {k for k in words if any(k in words[x] for x in ancestors(k, words))}
    body = tree(words, lang, _CHIP_PRED)
    alt = [f"{k}: {', '.join(map(str, v)) or '-'}" for k, v in list(words.items())[:40]]
    if first in by_text and second in by_text and by_text[first] not in looped and by_text[second] not in looped:
        a, b = by_text[first], by_text[second]
        table, line = path_table(a, b, words, lang, style=_CHIP_PRED, wordnet_names=False)
        body += _label(w["path"].format(a=a, b=b)) + table
        alt.append(line)
    elif first not in by_text or second not in by_text:
        notes.append(w["choose"])
    text = " ".join(notes[:8] + [w["legend"]])
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}: ' + "; ".join(alt) + ". " + _bare(text))


# --- try it: two English words in WordNet -------------------------------------------------------------------------------

LONGEST = 60    # characters of a typed word that count
SHOWN = 5       # pairs of senses in the table

_WORDS_WORDS = {
    "en": {"title": "{a} and {b} in WordNet",
           "head": ["Sense of the first word", "Sense of the second word", "Steps", "Path similarity"],
           "best": "The pair of senses most alike, and its path",
           "pairs": "{n} pairs of senses with the same part of speech; the {k} most alike:",
           "cosine": "Cosine similarity of *{a}* and *{b}* in GloVe: {c}.",
           "no_cosine": "GloVe lacks *{a}* or *{b}*, so it gives no cosine similarity.",
           "none_word": "WordNet has no English word *{word}*. Try another form, such as one without -s.",
           "no_pairs": "*{a}* and *{b}* have no nouns or verbs in common to compare: adjectives and adverbs have no "
                       "hypernyms in WordNet.",
           "no_path": "No pair of senses shares a hypernym: two verbs below different tops have no path.",
           "empty": "Type two words.",
           "legend": "Gray: what WordNet and GloVe hold. The path similarity compares two senses; the explore tries "
                     "every noun of the first word with every noun of the second, and every verb with every verb. In the "
                     "path, on the gray band at the top: the hypernym the two senses share."},
    "nl": {"title": "{a} en {b} in WordNet",
           "head": ["Betekenis van het eerste woord", "Betekenis van het tweede woord", "Stappen", "Path similarity"],
           "best": "Het paar betekenissen dat het meest op elkaar lijkt, en het pad ertussen",
           "pairs": "{n} paren betekenissen met dezelfde woordsoort; de {k} die het meest op elkaar lijken:",
           "cosine": "Cosine similarity van *{a}* en *{b}* in GloVe: {c}.",
           "no_cosine": "GloVe heeft *{a}* of *{b}* niet, dus er is geen cosine similarity.",
           "none_word": "WordNet heeft geen Engels woord *{word}*. Probeer een andere vorm, zoals een zonder -s.",
           "no_pairs": "*{a}* en *{b}* hebben geen zelfstandige naamwoorden of werkwoorden om te vergelijken: "
                       "bijvoeglijke naamwoorden en bijwoorden hebben in WordNet geen hyperoniemen.",
           "no_path": "Geen enkel paar betekenissen deelt een hyperoniem: twee werkwoorden onder verschillende toppen "
                      "hebben geen pad.",
           "empty": "Typ twee woorden.",
           "legend": "Grijs: wat WordNet en GloVe bevatten. De path similarity vergelijkt twee betekenissen; de "
                     "verkenning probeert elk zelfstandig naamwoord van het eerste woord met elk zelfstandig naamwoord "
                     "van het tweede, en elk werkwoord met elk werkwoord. In het pad, op de grijze band bovenaan: het "
                     "hyperoniem dat de twee betekenissen delen."},
}


def _english_senses(word: str) -> list:
    return wordnet.senses(word) or wordnet.senses(word.lower())


def sense_pairs_of(a: str, b: str) -> list:
    """(similarity, steps, sense of a, sense of b) for every noun of a with every noun of b and every verb with every
    verb that share a hypernym, most alike first; pairs alike as much in the order of the senses."""
    out = []
    sa, sb = _english_senses(a), _english_senses(b)
    for i, x in enumerate(sa):
        for j, y in enumerate(sb):
            if wordnet.synset(x).pos != wordnet.synset(y).pos or wordnet.synset(x).pos not in ("n", "v"):
                continue
            up_x, up_y = dict(_up(x)), dict(_up(y))
            sums = [up_x[n] + up_y[n] for n in up_x if n in up_y]
            if sums:
                out.append((1 / (min(sums) + 1), min(sums), i, j, x, y))
            else:
                out.append((None, None, i, j, x, y))
    found = sorted([p for p in out if p[0] is not None], key=lambda p: (-p[0], p[2], p[3]))
    return [(s, n, x, y) for s, n, _, _, x, y in found] + [(None, None, x, y) for s, n, _, _, x, y in out if s is None]


def words_picture(a, b, lang: str) -> str:
    w = _WORDS_WORDS[lang]
    a = " ".join(str(a)[:LONGEST].split())
    b = " ".join(str(b)[:LONGEST].split())
    title = w["title"].format(a=a or "…", b=b or "…")
    if not a or not b:
        return yours.figure(title, _note(w["empty"]), alt=w["empty"])
    for word in (a, b):
        if not _english_senses(word):
            text = w["none_word"].format(word=_plain_word(word))
            return yours.figure(title, _note(text), alt=_bare(text))
    pairs = sense_pairs_of(a, b)
    c = cosine(a, b)
    cos_line = (w["cosine"].format(a=_plain_word(a), b=_plain_word(b), c=_dec(c, lang)) if c is not None
                else w["no_cosine"].format(a=_plain_word(a), b=_plain_word(b)))
    if not pairs:
        text = w["no_pairs"].format(a=_plain_word(a), b=_plain_word(b)) + " " + cos_line
        return yours.figure(title, _note(text), alt=_bare(text))
    found = [p for p in pairs if p[0] is not None]
    if not found:
        text = w["no_path"] + " " + cos_line + " " + w["legend"]
        return yours.figure(title, _note(text), alt=_bare(text))
    rows = [[f"<code>{html.escape(x)}</code>", f"<code>{html.escape(y)}</code>", str(n),
             f'{_bar(s, "var(--ea-data)")} <span style="font-variant-numeric:tabular-nums">{_dec(s, lang)}</span>']
            for s, n, x, y in found[:SHOWN]]
    body = _label(w["pairs"].format(n=_int(len(pairs), lang), k=min(SHOWN, len(found))))
    body += _table(rows, w["head"], numbers=(2,))
    s, n, x, y = found[0]
    table, line = path_table(x, y, HYPERNYMS, lang)
    body += _label(w["best"]) + table
    text = cos_line + " " + w["legend"]
    return yours.figure(title, body + _note(text),
                        alt=f"{title}. " + "; ".join(f"{x} and {y}: {_dec(s, lang)}" for s, n, x, y in found[:SHOWN])
                        + ". " + line + " " + _bare(text))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["similarity", "ancestors", "path_similarity"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name, as the lesson's code writes it.
    The learner's function gets a copy, with the lists inside copied too."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return {k: list(v) for k, v in self.value.items()}


def _code(v) -> str:
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_code(k)}: {_code(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_code(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_code(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


def _shown(v):
    """What a function gave, as a message shows it: a small dict in full, as code; a large one by its size."""
    if isinstance(v, dict) and len(v) > 20:
        return {"en": f"a dict with {len(v):,} keys", "nl": f"een dict met {_int(len(v), 'nl')} keys"}
    return _code(v)


def _same_dict(got, want) -> bool:
    return isinstance(got, dict) and dict(got) == want


def _return_in_loop(got, text) -> bool:
    """A `return` indented under a `for` or `while` line: the loop then ends in its first round."""
    lines = [line.split("#")[0].rstrip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        head = line.lstrip()
        if head.startswith(("for ", "while ")) and head.endswith(":"):
            depth = len(line) - len(head)
            for after in lines[i + 1:]:
                body = after.lstrip()
                if not body:
                    continue
                if len(after) - len(body) <= depth:
                    break
                if body == "return" or body.startswith(("return ", "return(")):
                    return True
    return False


def _a_list(got, text) -> bool:
    return isinstance(got, (list, tuple, set))


def _empty_min(e) -> bool:
    return isinstance(e, ValueError)


# exercise 2: ancestors. The visible cases; then a sense whose second hypernym is also above its first, two paths of
# different lengths to one sense, a sense with two hypernyms that meet higher up, and three senses of WordNet.
SMALL = {"poodle": ["dog"], "dog": ["animal"], "cat": ["animal"], "animal": []}
_TWO_WAYS = {"x": ["y", "z"], "y": [], "z": ["y"]}
_DIAMOND = {"a": ["b", "c"], "b": ["d"], "c": ["e"], "e": ["d"], "d": []}
PETS = {"puppy": ["dog"], "dog": ["canine", "pet"], "canine": ["animal"], "pet": ["animal"], "kitten": ["cat"],
        "cat": ["feline", "pet"], "feline": ["animal"], "animal": []}
_HYPERNYMS_NAMED = Named("C.HYPERNYMS", HYPERNYMS)
ANCESTOR_CASES = [
    ("poodle", SMALL),
    ("animal", SMALL),
    ("x", _TWO_WAYS),
    ("a", _DIAMOND),
    ("puppy", PETS),
    ("dog.n.01", _HYPERNYMS_NAMED),
    ("cup.n.01", _HYPERNYMS_NAMED),
    ("run.v.01", _HYPERNYMS_NAMED),
]

_AN_HEAD = "from collections import deque\n\n\ndef ancestors(name, hypernyms):\n"
_AN_TOP = (_AN_HEAD + "    steps = {name: 0}\n    queue = deque([name])\n    while len(queue) > 0:\n"
           "        node = queue.popleft()\n")
_AN_START = _AN_TOP + "        ...\n    return steps\n"


def _an(body: str, take: str = "queue.popleft()") -> str:
    """ancestors with these lines in the loop, after the line that takes the next sense from the queue."""
    return _AN_TOP.replace("queue.popleft()", take) + body + "    return steps\n"


_AN = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
          "                steps[up] = steps[node] + 1\n                queue.append(up)\n")
_AN_STACK = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
                "                steps[up] = steps[node] + 1\n                queue.append(up)\n", take="queue.pop()")
_AN_OVERWRITE = _an("        for up in hypernyms[node]:\n            steps[up] = steps[node] + 1\n"
                    "            queue.append(up)\n")
_AN_ONE = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
              "                steps[up] = 1\n                queue.append(up)\n")
_AN_SAME = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
               "                steps[up] = steps[node]\n                queue.append(up)\n")
_AN_FIRST = _an("        for up in hypernyms[node][:1]:\n            if up not in steps:\n"
                "                steps[up] = steps[node] + 1\n                queue.append(up)\n")
_AN_NOT_QUEUED = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
                     "                steps[up] = steps[node] + 1\n")
_AN_RETURN = _an("        for up in hypernyms[node]:\n            if up not in steps:\n"
                 "                steps[up] = steps[node] + 1\n                queue.append(up)\n        return steps\n")
_AN_LIST = _AN.replace("    return steps\n", "    return list(steps)\n")

# exercise 3: path_similarity. The visible cases; then a pair whose first shared hypernym is not on the shortest
# path, two senses with no hypernym in common, two pets with two shared hypernyms, and four pairs of WordNet.
_FAR = {"a": ["x", "p"], "p": ["y"], "y": ["q"], "q": ["r"], "r": ["x"], "b": ["y"], "x": []}
_APART = {"run": ["move"], "move": [], "know": []}
SIMILARITY_CASES = [
    ("poodle", "cat", SMALL),
    ("cat", "cat", SMALL),
    ("a", "b", _FAR),
    ("run", "know", _APART),
    ("puppy", "kitten", PETS),
    ("coffee.n.01", "tea.n.01", _HYPERNYMS_NAMED),
    ("coffee.n.01", "cup.n.01", _HYPERNYMS_NAMED),
    ("dog.n.01", "domestic_cat.n.01", _HYPERNYMS_NAMED),
    ("run.v.01", "sleep.v.01", _HYPERNYMS_NAMED),
]

_PS_HEAD = "def path_similarity(a, b, hypernyms):\n"
_PS_TOP = (_PS_HEAD + "    up_a = ancestors(a, hypernyms)\n    up_b = ancestors(b, hypernyms)\n    steps = []\n")
_PS_START = _PS_TOP + "    ...\n"
_PS_LOOP = "    for name in up_a:\n        if name in up_b:\n            steps.append(up_a[name] + up_b[name])\n"
_PS_NONE = "    if len(steps) == 0:\n        return None\n"
_PS = _PS_TOP + _PS_LOOP + _PS_NONE + "    return 1 / (min(steps) + 1)\n"
_PS_STEPS = _PS_TOP + _PS_LOOP + _PS_NONE + "    return min(steps)\n"
_PS_NO_PLUS = _PS_TOP + _PS_LOOP + _PS_NONE + "    return 1 / min(steps)\n"
_PS_ONE_SIDE = _PS.replace("steps.append(up_a[name] + up_b[name])", "steps.append(up_a[name])")
_PS_OTHER_SIDE = _PS.replace("steps.append(up_a[name] + up_b[name])", "steps.append(up_b[name])")
_PS_PRECEDENCE = _PS.replace("1 / (min(steps) + 1)", "1 / min(steps) + 1")
_PS_MAX = _PS.replace("min(steps)", "max(steps)")
_PS_ZERO = _PS.replace("        return None\n", "        return 0\n")
_PS_NO_NONE = _PS_TOP + _PS_LOOP + "    return 1 / (min(steps) + 1)\n"
_PS_FIRST = (_PS_TOP + "    for name in up_a:\n        if name in up_b:\n"
             "            return 1 / (up_a[name] + up_b[name] + 1)\n    return None\n")

EXERCISES = {
    "similarity": Sort(
        name="similarity",
        groups={
            "half": {"en": "1/2", "nl": "1/2"},
            "third": {"en": "1/3", "nl": "1/3"},
            "quarter": {"en": "1/4", "nl": "1/4"},
            "fifth": {"en": "1/5", "nl": "1/5"},
        },
        items={
            "fork": {"en": "*fork* and *cutlery*", "nl": "*vork* en *bestek*"},
            "knife": {"en": "*knife* and *pan*", "nl": "*mes* en *pan*"},
            "spoon": {"en": "*spoon* and *knife*", "nl": "*lepel* en *mes*"},
            "pan_soup": {"en": "*pan* and *soup*", "nl": "*pan* en *soep*"},
            "tea": {"en": "*tea* and *drink*", "nl": "*thee* en *drank*"},
            "drink": {"en": "*drink* and *utensil*", "nl": "*drank* en *keukengerei*"},
            "pan": {"en": "*pan* and *cutlery*", "nl": "*pan* en *bestek*"},
            "fork_food": {"en": "*fork* and *food*", "nl": "*vork* en *eten*"},
        },
        solution={"fork": "half", "tea": "half", "spoon": "third", "pan": "third", "knife": "quarter",
                  "drink": "quarter", "pan_soup": "fifth", "fork_food": "fifth"},
        nudge={
            "fork": {"en": "Is *cutlery* the hypernym of *fork* itself? How many steps is that?",
                     "nl": "Is *bestek* zelf het hyperoniem van *vork*? Hoeveel stappen is dat?"},
            "knife": {"en": "Which hypernym do *knife* and *pan* share first, and how many steps is it up from each?",
                      "nl": "Welk hyperoniem delen *mes* en *pan* als eerste, en hoeveel stappen ligt dat boven elk?"},
            "spoon": {"en": "*spoon* and *knife* have the same hypernym. How many steps up and down is that?",
                      "nl": "*lepel* en *mes* hebben hetzelfde hyperoniem. Hoeveel stappen omhoog en omlaag is dat?"},
            "pan_soup": {"en": "Do *pan* and *soup* share any hypernym below the top?",
                         "nl": "Delen *pan* en *soep* een hyperoniem onder de top?"},
            "tea": {"en": "Is *drink* the hypernym of *tea* itself? How many steps is that?",
                    "nl": "Is *drank* zelf het hyperoniem van *thee*? Hoeveel stappen is dat?"},
            "drink": {"en": "Where do *drink* and *utensil* meet? Count the steps from each.",
                      "nl": "Waar komen *drank* en *keukengerei* samen? Tel de stappen vanaf elk."},
            "pan": {"en": "*pan* and *cutlery* hang from the same word. How many steps is the path?",
                    "nl": "*pan* en *bestek* hangen onder hetzelfde woord. Hoeveel stappen telt het pad?"},
            "fork_food": {"en": "How many steps is it from *fork* up to the top, and from *food*?",
                          "nl": "Hoeveel stappen is het van *vork* omhoog naar de top, en vanaf *eten*?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each pair under its path similarity: count the steps of the shortest path between the two words.",
                      "nl": "Zet elk paar onder de path similarity: tel de stappen van het kortste pad tussen de twee woorden."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The pair {item} belongs under another path similarity. How many steps does the shortest path between the two have?",
                      "nl": "Het paar {item} hoort bij een andere path similarity. Hoeveel stappen telt het kortste pad tussen de twee?"},
            "correct": {"en": "All {n} are right. A word and its own hypernym are 1 step apart, two words under one hypernym 2, and the further up two words meet, the lower their path similarity.",
                        "nl": "Alle {n} kloppen. Een woord en het eigen hyperoniem liggen 1 stap uit elkaar, twee woorden onder één hyperoniem 2, en hoe hoger twee woorden samenkomen, hoe lager de path similarity."},
        },
    ),
    "ancestors": FunctionBox(
        name="ancestors",
        picture=ancestors_picture,
        function="ancestors",
        start=_AN_START,
        solution=_AN,
        visible=ANCESTOR_CASES[:2],
        hidden=ANCESTOR_CASES[2:],
        compare=_same_dict,
        show=_shown,
        first=[_DOTS],
        time_limit=1.0,
        mistakes=[
            Mistake("return-in-loop", matches=_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: `return steps` stands inside the loop, so your function gives back the dict after the first round. How far should `return steps` be indented, so that it runs after the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `return steps` staat in de loop, dus je functie geeft de dict al na de eerste ronde terug. Hoe ver moet `return steps` inspringen, zodat de regel na de loop komt?"),
            Mistake("not-queued", wrong=_AN_NOT_QUEUED,
                    en="`{call}` gives {got}, and it should give {want}: your dict stops one step up, so the search never goes on from a hypernym it found. Does each hypernym go into the queue, and does the loop look at the hypernyms of `node`, the sense it just took from the queue?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je dict stopt één stap omhoog, dus het zoeken gaat nooit verder vanaf een hyperoniem dat het vond. Gaat elk hyperoniem de wachtrij in, en kijkt de loop naar de hyperoniemen van `node`, de betekenis die de loop net uit de wachtrij nam?"),
            Mistake("first-only", wrong=_AN_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function follows only the first hypernym of each sense, as `chain` did. A sense can have more than one: does your loop go over all of them?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie volgt alleen het eerste hyperoniem van elke betekenis, zoals `chain` deed. Een betekenis kan er meer hebben: gaat je loop alle hyperoniemen langs?"),
            Mistake("overwrite", wrong=_AN_OVERWRITE,
                    en="`{call}` gives {got}, and it should give {want}: a sense that the search reaches a second time, along a longer path, gets the longer number of steps. Its first number was already the fewest: when should a hypernym get its steps?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een betekenis die het zoeken een tweede keer bereikt, langs een langer pad, krijgt het langere aantal stappen. Het eerste getal was al het kleinste: wanneer moet een hyperoniem de stappen krijgen?"),
            Mistake("stack", wrong=_AN_STACK,
                    en="`{call}` gives {got}, and it should give {want}: some sense gets more steps than its fewest. The box takes the next sense with `queue.popleft()`, from the front, as breadth-first search does; does your code still take it from the front?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een betekenis krijgt meer stappen dan het kleinste aantal. Het vak neemt de volgende betekenis met `queue.popleft()`, van voren, zoals breadth-first search doet; neemt je code de betekenis nog steeds van voren?"),
            Mistake("all-one", wrong=_AN_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every sense above `name` gets 1 step. A hypernym of `node` is one step further up than `node` itself: where do you find how far up `node` is?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke betekenis boven `name` krijgt 1 stap. Een hyperoniem van `node` ligt één stap hoger dan `node` zelf: waar vind je hoe hoog `node` ligt?"),
            Mistake("no-step", wrong=_AN_SAME,
                    en="`{call}` gives {got}, and it should give {want}: every sense gets 0 steps. A hypernym of `node` lies one step further up than `node`: what should you add?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke betekenis krijgt 0 stappen. Een hyperoniem van `node` ligt één stap hoger dan `node`: wat moet je erbij optellen?"),
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back only the senses, and not the dict from each sense to its steps. What comes after `return`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft alleen de betekenissen terug, en niet de dict van elke betekenis naar de stappen. Wat komt er na `return`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give every hypernym of `node` its number of steps, and put it in the queue.",
                      "nl": "Vervang de `...`: geef elk hyperoniem van `node` het aantal stappen, en zet het in de wachtrij."},
            "slow": {"en": "Your function is still running after 1 second on a case of the check: the loop never stops. Does every round take a sense out of `queue`, and does a sense go into the queue only once?",
                     "nl": "Je functie is na 1 seconde nog bezig met een geval van de check: de loop stopt nooit. Haalt elke ronde een betekenis uit `queue`, en gaat een betekenis maar één keer de wachtrij in?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each hypernym of `node` get its steps, one more than `node`, and go into the queue?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Krijgt elk hyperoniem van `node` de stappen, één meer dan `node`, en gaat het de wachtrij in?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a sense with two hypernyms too, and for a sense that two paths reach?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een betekenis met twee hyperoniemen, en voor een betekenis die twee paden bereiken?"},
            "correct": {"en": "Your `ancestors` is right in all {n} cases, also for three senses of WordNet. Look at the picture: which senses lie above both dog and house cat?",
                        "nl": "Je `ancestors` klopt in alle {n} gevallen, ook voor drie betekenissen van WordNet. Kijk naar de afbeelding: welke betekenissen liggen boven zowel hond als huiskat?"},
        },
    ),
    "path_similarity": FunctionBox(
        name="path_similarity",
        picture=similarity_picture,
        function="path_similarity",
        start=_PS_START,
        solution=_PS,
        visible=SIMILARITY_CASES[:2],
        hidden=SIMILARITY_CASES[2:],
        compare=_same,
        show=_code,
        first=[_DOTS],
        names={"ancestors": ancestors},
        time_limit=1.0,
        mistakes=[
            Mistake("empty-steps", error=_empty_min,
                    en="With `{call}`, your function stops with a ValueError: the two senses share no hypernym, so `steps` stays empty, and Python cannot take the smallest number of an empty list. What should the function give back when `steps` is empty?",
                    nl="Met `{call}` stopt je functie met een ValueError: de twee betekenissen delen geen hyperoniem, dus `steps` blijft leeg, en Python kan van een lege lijst niet het kleinste getal nemen. Wat moet de functie teruggeven als `steps` leeg is?"),
            Mistake("return-in-loop", matches=_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back a similarity inside the loop, for the first hypernym the two senses share. That need not be the shortest path. How far should the `return` be indented, so that it runs after the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft al in de loop een gelijkenis terug, voor het eerste hyperoniem dat de twee betekenissen delen. Dat hoeft niet het kortste pad te zijn. Hoe ver moet de `return` inspringen, zodat de regel na de loop komt?"),
            Mistake("steps", wrong=_PS_STEPS,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the number of steps of the shortest path. The path similarity turns that number into 1 divided by the steps plus 1: what comes after `return`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het aantal stappen van het kortste pad terug. De path similarity maakt van dat getal 1 gedeeld door het aantal stappen plus 1: wat komt er na `return`?"),
            Mistake("no-plus-one", wrong=_PS_NO_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function divides 1 by the steps alone. Then a sense and itself, 0 steps apart, cannot get 1. What does the path similarity add to the steps?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt 1 door alleen het aantal stappen. Dan kan een betekenis met zichzelf, 0 stappen uit elkaar, geen 1 krijgen. Wat telt de path similarity bij het aantal stappen op?"),
            Mistake("one-side", wrong=_PS_ONE_SIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function counts only the steps from `a` up to each shared hypernym. A path goes up from `a` and then down to `b`: which steps are missing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt alleen de stappen van `a` omhoog naar elk gedeeld hyperoniem. Een pad gaat omhoog vanaf `a` en dan omlaag naar `b`: welke stappen ontbreken?"),
            Mistake("other-side", wrong=_PS_OTHER_SIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function counts only the steps from `b` up to each shared hypernym. A path goes up from `a` and then down to `b`: which steps are missing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt alleen de stappen van `b` omhoog naar elk gedeeld hyperoniem. Een pad gaat omhoog vanaf `a` en dan omlaag naar `b`: welke stappen ontbreken?"),
            Mistake("precedence", wrong=_PS_PRECEDENCE,
                    en="`{call}` gives {got}, and it should give {want}: Python divides before it adds, so `1 / x + 1` is `(1 / x) + 1`. Where do the parentheses go?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: Python deelt voordat het optelt, dus `1 / x + 1` is `(1 / x) + 1`. Waar horen de haakjes?"),
            Mistake("longest", wrong=_PS_MAX,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the longest path through a shared hypernym. Which of the numbers in `steps` belongs to the shortest path?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het langste pad via een gedeeld hyperoniem. Welk getal in `steps` hoort bij het kortste pad?"),
            Mistake("zero", wrong=_PS_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: for two senses that share no hypernym, your function gives back 0. Then no path joins them, and they get no similarity: what does Python write for nothing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor twee betekenissen die geen hyperoniem delen, geeft je functie 0 terug. Dan is er helemaal geen pad, en geen gelijkenis: wat schrijft Python voor niets?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: put the steps of the path through each shared hypernym in `steps`, and give back the path similarity of the shortest, or `None`.",
                      "nl": "Vervang de `...`: zet het aantal stappen van het pad via elk gedeeld hyperoniem in `steps`, en geef de path similarity van het kortste terug, of `None`."},
            "slow": {"en": "Your function is still running after 1 second on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 1 seconde nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each hypernym the two senses share, count the steps of the path through it: is the path similarity 1 / (steps + 1) for the shortest of those paths, and `None` when the two share none?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel voor elk hyperoniem dat de twee betekenissen delen de stappen van het pad erlangs: is de path similarity 1 / (stappen + 1) voor het kortste van die paden, en `None` als de twee geen hyperoniem delen?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function find the shortest path when two hypernyms are shared, and give `None` when none is?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Vindt je functie het kortste pad als twee hyperoniemen gedeeld worden, en geeft je functie `None` als er geen gedeeld wordt?"},
            "correct": {"en": "Your `path_similarity` is right in all {n} cases, also for four pairs of WordNet. Look at the picture: which pairs get a high path similarity, and which a high cosine similarity in GloVe?",
                        "nl": "Je `path_similarity` klopt in alle {n} gevallen, ook voor vier paren uit WordNet. Kijk naar de afbeelding: welke paren krijgen een hoge path similarity, en welke een hoge cosine similarity in GloVe?"},
        },
    ),
}

CASES = [
    ("similarity", {}, "empty", None),
    ("similarity", {"fork": "half", "tea": "half", "spoon": "third", "pan": "third", "knife": "quarter",
                    "drink": "quarter", "pan_soup": "fifth", "fork_food": "fifth"}, "correct", None),
    ("similarity", {"fork": "half", "tea": "half"}, "wrong", "unfinished"),
    ("similarity", {"fork": "third"}, "wrong", "wrong:fork"),
    ("similarity", {"tea": "third"}, "wrong", "wrong:tea"),
    ("similarity", {"spoon": "half"}, "wrong", "wrong:spoon"),
    ("similarity", {"pan": "quarter"}, "wrong", "wrong:pan"),
    ("similarity", {"knife": "third"}, "wrong", "wrong:knife"),
    ("similarity", {"drink": "fifth"}, "wrong", "wrong:drink"),
    ("similarity", {"pan_soup": "quarter"}, "wrong", "wrong:pan_soup"),
    ("similarity", {"fork_food": "quarter"}, "wrong", "wrong:fork_food"),
    ("ancestors", _AN_START, "empty", None),
    ("ancestors", _AN, "correct", None),
    ("ancestors", _an("        for up in hypernyms[node]:\n            if up in steps:\n                continue\n"
                      "            steps[up] = steps[node] + 1\n            queue.append(up)\n"), "correct", None),
    ("ancestors", _an("        k = steps[node]\n        for up in hypernyms[node]:\n            if not up in steps:\n"
                      "                steps[up] = k + 1\n                queue.append(up)\n"), "correct", None),
    ("ancestors", _AN_RETURN, "wrong", "return-in-loop"),
    ("ancestors", _AN_NOT_QUEUED, "wrong", "not-queued"),
    ("ancestors", _AN_FIRST, "wrong", "first-only"),
    ("ancestors", _AN_OVERWRITE, "wrong", "overwrite"),
    ("ancestors", _AN_STACK, "wrong", "stack"),
    ("ancestors", _AN_ONE, "wrong", "all-one"),
    ("ancestors", _AN_SAME, "wrong", "no-step"),
    ("ancestors", _AN_LIST, "wrong", "a-list"),
    ("ancestors", _an("        queue.append(node)\n        for up in hypernyms[node]:\n            if up not in steps:\n"
                      "                steps[up] = steps[node] + 1\n                queue.append(up)\n"), "slow", None),
    ("ancestors", _AN_TOP + "        ...\n    return {name: 0}\n", "wrong", "dots-left"),
    ("path_similarity", _PS_START, "empty", None),
    ("path_similarity", _PS, "correct", None),
    ("path_similarity", _PS_TOP + "    for name in up_b:\n        if name in up_a:\n"
                        "            steps.append(up_b[name] + up_a[name])\n    if steps == []:\n        return None\n"
                        "    return 1 / (1 + min(steps))\n", "correct", None),
    ("path_similarity", _PS_TOP + "    best = None\n    for name in up_a:\n        if name in up_b:\n"
                        "            n = up_a[name] + up_b[name]\n            if best == None or n < best:\n"
                        "                best = n\n    if best == None:\n        return None\n"
                        "    return 1 / (best + 1)\n", "correct", None),
    ("path_similarity", _PS_NO_PLUS, "wrong", "no-plus-one"),
    ("path_similarity", _PS_NO_NONE, "error", "empty-steps"),
    ("path_similarity", _PS_FIRST, "wrong", "return-in-loop"),
    ("path_similarity", _PS_STEPS, "wrong", "steps"),
    ("path_similarity", _PS_ONE_SIDE, "wrong", "one-side"),
    ("path_similarity", _PS_OTHER_SIDE, "wrong", "other-side"),
    ("path_similarity", _PS_PRECEDENCE, "wrong", "precedence"),
    ("path_similarity", _PS_MAX, "wrong", "longest"),
    ("path_similarity", _PS_ZERO, "wrong", "zero"),
    ("path_similarity", _PS_TOP + "    ...\n    return None\n", "wrong", "dots-left"),
]
