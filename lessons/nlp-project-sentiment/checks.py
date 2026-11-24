# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-project-sentiment, the second step of the project `text`: a label for each sentence from a
lexicon, with no label for a sum of exactly 0 (lexicon_labels, saved as text/lexicon.py beside the course's version of
lexicon and score), the macro F1 of two labels when a label may be missing (macro_f1, saved as text/scores.py), and
the sentences that two classifiers get wrong, together and each alone (compare, saved as text/compare.py); a demo that
trains the classifier of the project, with the learner's features and step, on the review sentences in English and in
Dutch; and a choice about what shared mistakes say.

The sentences are "Sentiment Labelled Sentences" (Kotzias, Denil, de Freitas and Smyth, KDD 2015; UCI Machine Learning
Repository, CC BY 4.0, https://doi.org/10.24432/C57604), in data/uci-sentiment.tsv.gz, and the course's Dutch
translation of them in data/uci-sentiment-nl.tsv.gz, split as in the lessons before: every fifth sentence is held out.
Pattern's lists are the sentiment lexicons of Pattern (Tom De Smedt and Walter Daelemans, CLiPS; PDDL), in
data/pattern-sentiment.tsv.gz. Sentences marked as not shown are counted but never shown. Every number in the text and
the pictures is computed by the course (tests/test_nlp_project_sentiment.py computes them again)."""
import contextlib
import html
import io
import math
import re
import time
import warnings

import numpy as np

from everything_ai import books, chart, live, projects, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, run_code, with_time_limit

_KEY = "nlp-project-sentiment"

# the project modules this lesson uses, in the order they build on each other (template 1.3)
projects.builds_on(("text", "features", "nlp-project-spam", "features"), ("text", "train", "nlp-project-spam", "step"),
                   ("text", "lexicon", "nlp-project-sentiment", "lexicon_labels"),
                   ("text", "scores", "nlp-project-sentiment", "macro_f1"),
                   ("text", "compare", "nlp-project-sentiment", "compare"))

# --- the sentences -------------------------------------------------------------------------------------------------------

HELD_OUT = 5     # every fifth sentence is held out, as in the lessons before
STEPS = 300      # steps of gradient descent, as in Logistic regression on words
RATE = 1.0       # its learning rate
THRESHOLD = 0.5  # a sentence is positive from this probability on
AFTER = 2        # a negation turns the score of the next two words around
LANGS = ("en", "nl")
LABELS = {"en": {1: "positive", 0: "negative", None: "no label"}, "nl": {1: "positief", 0: "negatief", None: "geen label"}}
NEGATIONS = {   # the negation words of A list of good and bad words
    "en": ["not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "aren't", "can't", "couldn't", "didn't",
           "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mustn't", "shouldn't", "wasn't", "weren't",
           "won't", "wouldn't"],
    "nl": ["niet", "geen", "nooit", "niets", "niemand", "nergens"],
}
LANGUAGE = {"en": {"en": "English", "nl": "Dutch"}, "nl": {"en": "Engels", "nl": "Nederlands"}}


def words(text: str) -> list:
    """The words of a text in small letters: runs of letters, with an apostrophe inside a word as in *don't*."""
    return books.words(text.lower())


def _all(lang: str) -> tuple:
    return books.review_sentences() if lang == "en" else books.dutch_review_sentences()


def _split(lang: str, held: bool) -> list:
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all(lang))
            if (k % HELD_OUT == HELD_OUT - 1) == held]


def training(lang: str) -> list:
    """The 2,400 training sentences in one language, as a new list of [words, label], words first as in the project:
    label 1 positive, 0 negative."""
    return [[words(text), label] for label, text, _ in _split(lang, False)]


def held_out(lang: str) -> list:
    """The 600 held-out sentences in one language, as a new list of [words, label]."""
    return [[words(text), label] for label, text, _ in _split(lang, True)]


def held_texts(lang: str) -> list:
    """The 600 held-out sentences as (label, text, shown), in the same order."""
    return _split(lang, True)


def gold(lang: str) -> list:
    """The labels of the 600 held-out sentences."""
    return [label for label, _, _ in _split(lang, True)]


# --- the course's versions of the functions of the project ----------------------------------------------------------------

def document_frequencies(sentences: list) -> dict:
    df = {}
    for ws, _ in sentences:
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    return df


def features(ws: list, df: dict, n: int) -> dict:
    row = {}
    for w in ws:
        if w in df:
            row[w] = row.get(w, 0.0) + math.log10(n / df[w])
    return row


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _model(lang: str) -> dict:
    """The classifier of the project in one language, as the course trains it with NumPy: the TF-IDF rows of the
    training sentences, then STEPS steps of gradient descent with learning rate RATE from weight 0 and bias 0, as the
    project's step does. {"df", "weights", "bias", "p": the probability of positive for every held-out sentence}."""
    def make():
        train = training(lang)
        n = len(train)
        df = document_frequencies(train)
        vocab = sorted(df)
        col = {w: k for k, w in enumerate(vocab)}

        def rows(sentences):
            x = np.zeros((len(sentences), len(vocab)))
            for i, (ws, _) in enumerate(sentences):
                for w, v in features(ws, df, n).items():
                    x[i, col[w]] = v
            return x
        x = rows(train)
        y = np.array([label for _, label in train], dtype=float)
        wv, b = np.zeros(len(vocab)), 0.0
        for _ in range(STEPS):
            p = 1 / (1 + np.exp(-(x @ wv + b)))
            wv = wv - RATE * ((p - y) @ x / n)
            b = b - RATE * float((p - y).mean())
        ph = 1 / (1 + np.exp(-(rows(held_out(lang)) @ wv + b)))
        return {"df": df, "weights": {w: float(wv[k]) for k, w in enumerate(vocab)}, "bias": b,
                "p": [float(q) for q in ph]}
    return kept(f"{_KEY}:model:{lang}", make)


def model(lang: str) -> dict:
    m = _model(lang)
    return {"df": dict(m["df"]), "weights": dict(m["weights"]), "bias": m["bias"], "p": list(m["p"])}


def classifier_labels(lang: str) -> list:
    """The labels the course's classifier gives the held-out sentences: 1 from THRESHOLD on, else 0."""
    return [int(p >= THRESHOLD) for p in _model(lang)["p"]]


def lexicon(sentences: list) -> dict:
    """The course's version of the lexicon of A list of good and bad words, from [words, label] pairs."""
    counts = {}
    for ws, label in sentences:
        for w in ws:
            counts.setdefault(w, [0, 0])[label] += 1
    return {w: math.log10((c[1] + 1) / (c[0] + 1)) for w, c in counts.items()}


def score(ws: list, scores: dict, negations: list, after: int = AFTER) -> float:
    """The sum of the scores of the words the lexicon has, with the `after` words after a negation turned around."""
    total, left = 0.0, 0
    for w in ws:
        if w in scores:
            total += -scores[w] if left > 0 else scores[w]
        left -= 1
        if w in negations:
            left = after
    return total


def label_of(total: float):
    return 1 if total > 0 else 0 if total < 0 else None


def _lexicon(lang: str) -> dict:
    return kept(f"{_KEY}:lexicon:{lang}", lambda: lexicon(training(lang)))


def course_lexicon(lang: str) -> dict:
    """The lexicon from the 2,400 training sentences of one language."""
    return dict(_lexicon(lang))


def _pattern(lang: str) -> dict:
    def make():
        by = {}
        for lang_, form, pos, _, p, _, _, _ in books.pattern_sentiment():
            if lang_ == lang:
                by.setdefault(form.lower(), {}).setdefault(pos, []).append(p)
        # as Pattern does for a word without a tag: the mean of its senses per tag, then the mean over its tags
        return {w: sum(sum(v) / len(v) for v in tags.values()) / len(tags) for w, tags in by.items()}
    return kept(f"{_KEY}:pattern:{lang}", make)


def pattern(lang: str) -> dict:
    """Pattern's list in one language, as A list of good and bad words takes it."""
    return dict(_pattern(lang))


def lexicon_labels_of(lang: str, which: str = "lexicon") -> list:
    """The labels a lexicon gives the held-out sentences, with negation: "lexicon" for the list from the training
    sentences, "pattern" for Pattern's list."""
    scores = _lexicon(lang) if which == "lexicon" else _pattern(lang)
    return [label_of(score(ws, scores, NEGATIONS[lang])) for ws, _ in held_out(lang)]


def f1_of(gold_: list, said: list, label: int) -> float:
    tp = sum(g == label and s == label for g, s in zip(gold_, said))
    fp = sum(g != label and s == label for g, s in zip(gold_, said))
    fn = sum(g == label and s != label for g, s in zip(gold_, said))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def macro_of(gold_: list, said: list) -> float:
    return (f1_of(gold_, said, 1) + f1_of(gold_, said, 0)) / 2


def compare_of(gold_: list, mine: list, theirs: list) -> tuple:
    both = [i for i in range(len(gold_)) if mine[i] != gold_[i] and theirs[i] != gold_[i]]
    only_mine = [i for i in range(len(gold_)) if mine[i] != gold_[i] and theirs[i] == gold_[i]]
    only_theirs = [i for i in range(len(gold_)) if mine[i] == gold_[i] and theirs[i] != gold_[i]]
    return both, only_mine, only_theirs


def numbers(lang: str) -> dict:
    """The numbers the text gives for one language."""
    def make():
        g = gold(lang)
        out = {"training": len(training(lang)), "held_out": len(g), "positive": sum(g)}
        train = training(lang)
        vocab = document_frequencies(train)
        counts = {}
        for ws, _ in train:
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
        out["vocab"] = len(vocab)
        out["once"] = sum(1 for c in counts.values() if c == 1)
        for name, said in (("classifier", classifier_labels(lang)), ("lexicon", lexicon_labels_of(lang)),
                           ("pattern", lexicon_labels_of(lang, "pattern"))):
            out[name] = {"right": sum(s == x for s, x in zip(said, g)), "none": said.count(None),
                         "f1_pos": f1_of(g, said, 1), "f1_neg": f1_of(g, said, 0), "macro": macro_of(g, said)}
        both, only_c, only_l = compare_of(g, classifier_labels(lang), lexicon_labels_of(lang))
        out["both"], out["only_classifier"], out["only_lexicon"] = len(both), len(only_c), len(only_l)
        wc, wl = len(g) - out["classifier"]["right"], len(g) - out["lexicon"]["right"]
        out["chance"] = wc * wl / len(g)
        return out
    return kept(f"{_KEY}:numbers:{lang}", make)


# --- the sentences the lesson shows -------------------------------------------------------------------------------------

IDEA = 358       # Predictable, but not a bad watch. (positive): the classifier gets it wrong, the lexicon right
WATCHED = [0, 230, 358]   # three held-out sentences the demo follows while it trains
EXAMPLES = {"both": [70], "classifier": [358, 234], "lexicon": [230]}   # the sentences named in the text, per group
SHOWN_PER_GROUP = 4   # sentences per group in the project's picture

_HIDDEN = {"en": "(a sentence the course does not show)", "nl": "(een zin die de course niet laat zien)"}
# a sentence of the set that says he or she (in Dutch also ze, hun, zijn and the like) stays out of a picture, as the
# course's own writing does
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


def _shows(text: str, shown: int, lang: str) -> bool:
    return bool(shown) and _PRONOUN[lang].search(text) is None


def sentence(i: int, lang: str) -> str:
    """Held-out sentence i as the lesson may show it."""
    _, text, shown = held_texts(lang)[i]
    return text if _shows(text, shown, lang) else _HIDDEN[lang]


def showable(i: int, lang: str) -> bool:
    _, text, shown = held_texts(lang)[i]
    return _shows(text, shown, lang)


# --- the small examples of the exercises ------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `small_texts`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


SMALL_SCORES = {"good": 0.5, "bad": -0.7, "not": -0.6}
SMALL_TEXTS = [["good", "food"], ["not", "good"], ["not", "bad"], ["fine"]]
SMALL_NEGATIONS = ["not"]
SMALL_GOLD = [1, 1, 1, 0, 0, 0]
SMALL_SAID = [1, 1, 0, 0, None, 1]
SMALL_MINE = [1, 0, 1, 1, 0, 0]
SMALL_THEIRS = [1, 0, 0, 0, None, 0]
REAL = 5   # how many held-out sentences the real case of exercise 1 takes

_ST = _Val("small_texts", SMALL_TEXTS)
_SS = _Val("small_scores", SMALL_SCORES)
_SG = _Val("small_gold", SMALL_GOLD)
_SD = _Val("small_said", SMALL_SAID)
_SM = _Val("small_mine", SMALL_MINE)
_STH = _Val("small_theirs", SMALL_THEIRS)


def _real_texts() -> list:
    return [ws for ws, _ in held_out("en")[:REAL]]


# --- the code of the exercises -------------------------------------------------------------------------------------------

_LEX_HEAD = (
    "import numpy as np\n\n\n"
    "def lexicon(sentences):\n"
    "    # the course's version of your lexicon of A list of good and bad words: for every word, log10 of\n"
    "    # its count in positive sentences plus 1 over its count in negative sentences plus 1\n"
    "    counts = {}\n"
    "    for words, label in sentences:\n"
    "        for word in words:\n"
    "            if word in counts:\n"
    "                counts[word][label] = counts[word][label] + 1\n"
    "            else:\n"
    "                counts[word] = [0, 0]\n"
    "                counts[word][label] = 1\n"
    "    scores = {}\n"
    "    for word in counts:\n"
    "        scores[word] = np.log10((counts[word][1] + 1) / (counts[word][0] + 1))\n"
    "    return scores\n\n\n"
    "def score(words, scores, negations):\n"
    "    # the course's version of your score: the sum of the scores of the words the lexicon has,\n"
    "    # with the two words after a negation counted the other way around\n"
    "    total = 0\n"
    "    left = 0\n"
    "    for word in words:\n"
    "        if word in scores:\n"
    "            if left > 0:\n"
    "                total = total - scores[word]\n"
    "            else:\n"
    "                total = total + scores[word]\n"
    "        left = left - 1\n"
    "        if word in negations:\n"
    "            left = 2\n"
    "    return total\n\n\n"
    "def lexicon_labels(texts, scores, negations):\n"
    "    # a label for the words of each sentence: 1 for a sum above 0, 0 for a sum below 0,\n"
    "    # and None for a sum of exactly 0, when the lexicon says nothing\n"
    "    labels = []\n"
    "    for words in texts:\n")
_LEX_START = _LEX_HEAD + "        ...\n    return labels\n"


def _lx(call: str = "score(words, scores, negations)", start: str = "None", up: str = "1", down: str = "0",
        add: str = "label", end: str = "    return labels\n") -> str:
    return (_LEX_HEAD + f"        total = {call}\n        label = {start}\n        if total > 0:\n            label = {up}\n"
            f"        if total < 0:\n            label = {down}\n        labels.append({add})\n" + end)


_LEX = _lx()

_F1_HEAD = ("def macro_f1(gold, said):\n"
            "    # the mean of the F1 scores of the labels 1 and 0; a None in `said` is a label for neither\n"
            "    total = 0\n"
            "    for label in [1, 0]:\n")
_F1_START = _F1_HEAD + "        ...\n    return total / 2\n"


def _f1(tp: str = "gold[i] == label and said[i] == label", fp: str = "gold[i] != label and said[i] == label",
        fn: str = "gold[i] == label and said[i] != label", add: str = "2 * tp / (2 * tp + fp + fn)",
        end: str = "    return total / 2\n", labels: str = "") -> str:
    head = _F1_HEAD if not labels else _F1_HEAD.replace("[1, 0]", labels)
    return (head + "        tp = 0\n        fp = 0\n        fn = 0\n        for i in range(len(gold)):\n"
            f"            if {tp}:\n                tp = tp + 1\n            if {fp}:\n                fp = fp + 1\n"
            f"            if {fn}:\n                fn = fn + 1\n        total = total + {add}\n" + end)


_F1 = _f1()

_CMP_HEAD = ("def compare(gold, mine, theirs):\n"
             "    # the places of the sentences that both get wrong, that only `mine` gets wrong,\n"
             "    # and that only `theirs` gets wrong\n"
             "    both = []\n"
             "    only_mine = []\n"
             "    only_theirs = []\n"
             "    for i in range(len(gold)):\n")
_CMP_START = _CMP_HEAD + "        ...\n    return both, only_mine, only_theirs\n"


def _cmp(both: str = "mine[i] != gold[i] and theirs[i] != gold[i]",
         mine: str = "mine[i] != gold[i] and theirs[i] == gold[i]",
         theirs: str = "mine[i] == gold[i] and theirs[i] != gold[i]", add: str = "i",
         end: str = "    return both, only_mine, only_theirs\n") -> str:
    return (_CMP_HEAD + f"        if {both}:\n            both.append({add})\n        if {mine}:\n"
            f"            only_mine.append({add})\n        if {theirs}:\n            only_theirs.append({add})\n" + end)


_CMP = _cmp()


def _course(code: str, name: str):
    space: dict = {}
    exec(compile(code, f"<course {name}>", "exec"), space)
    return space[name]


lexicon_labels = _course(_LEX, "lexicon_labels")
macro_f1 = _course(_F1, "macro_f1")
compare = _course(_CMP, "compare")


# --- the pictures: a few helpers ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_FADED = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"
_CHIP_PARAM = _CHIP + ";background:var(--ea-param-wash);color:var(--ea-param-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_TURNED = "font-style:italic;text-decoration:underline dotted"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"
_FAINT = "color:var(--ea-ink-3)"


def _marked(text: str) -> str:
    parts_ = html.escape(text).split("*")
    return "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts_))


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,400 in English, 2.400 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2, sign: bool = False) -> str:
    """A number with `digits` decimals, with − for minus (and + when `sign`): 0.82 in English, 0,82 in Dutch. A
    number that rounds to 0 gets no sign."""
    x = round(float(x), digits) + 0.0
    sign = sign and x != 0
    text = (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 8.0) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _number(v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 5   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _left(deadline: float) -> float:
    """The seconds a picture has left until `deadline`, a time.monotonic()."""
    return deadline - time.monotonic()


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text = w[key].format(s=PICTURE_SECONDS)
    return yours.figure(w["title"], _note(text), alt=text)


def _same(a, b) -> bool:
    return _number(a) and _number(b) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _value_chip(mine, want, lang: str, digits: int = 3, style: str = _CHIP_PRED) -> str:
    """Your number in blue when it is right; else in purple, with the right one dashed beside it."""
    if _same(mine, want):
        return _chip(_dec(mine, lang, digits), style)
    if _number(mine):   # enough decimals to show that the two differ, as 0.8183 against 0.8180
        while digits < 6 and _dec(mine, lang, digits) == _dec(want, lang, digits):
            digits += 1
    got = _chip(_dec(mine, lang, digits) if _number(mine) else str(mine), _CHIP_PURPLE)
    return got + _chip(_dec(want, lang, digits), _CHIP_MISSING)


def _count_chip(mine, want, lang: str) -> str:
    if isinstance(mine, (int, np.integer)) and not isinstance(mine, bool) and int(mine) == want:
        return _chip(_num(mine, lang), _CHIP_PRED)
    return _chip(str(mine), _CHIP_PURPLE) + _chip(_num(want, lang), _CHIP_MISSING)


def _label_chip(said, gold_: int, lang: str) -> str:
    text = LABELS[lang].get(said, str(said)) if said in (0, 1, None) else str(said)
    return _chip(text, _CHIP_DATA if said == gold_ else _CHIP_PURPLE)


def _sentence(i: int, lang: str, longest: int = 0) -> str:
    text = sentence(i, lang)
    if longest and len(text) > longest:
        text = text[:longest - 1].rstrip() + "…"
    return f'<span style="{_TEXT}">{html.escape(text)}</span>'


# --- the picture in The idea: one sentence in your classifier and in your lexicon --------------------------------------

_IDEA_TEXT = {
    "en": {"title": "One held-out sentence in the classifier and in the lexicon",
           "classifier": "The classifier, after training", "lexicon": "The lexicon from the training sentences",
           "head_c": ["Word", "TF-IDF value", "Weight", "Value × weight"],
           "head_l": ["Word", "Score", "Counts as"],
           "bias": "bias", "z": "z, the sum", "p": "p, the probability of positive",
           "sum": "Sum, with the two words after *not* turned around",
           "label": "Label", "gold": "The sentence's own label: {g}.",
           "note": "Gray: the words in the classifier's table, and a label that is the sentence's own. In the color of "
                   "the text: the TF-IDF values, value × weight and z, and in the lexicon the words, their scores and "
                   "the sum. Orange: the weights and the bias the classifier learned. Blue: the probability of "
                   "positive. Dotted and in italics: a score turned around, in the two words after a negation. Purple: "
                   "a label other than the sentence's own.",
           "alt": "{title}: {sentence} {gold} Classifier: {crow}; bias {bias}, z {z}, p {p}, label {cl}. Lexicon: "
                  "{lrow}; sum {sum}, label {ll}. {note}"},
    "nl": {"title": "Eén achtergehouden zin in de classifier en in het lexicon",
           "classifier": "De classifier, na de training", "lexicon": "Het lexicon uit de trainingszinnen",
           "head_c": ["Woord", "TF-IDF-waarde", "Weight", "Waarde × weight"],
           "head_l": ["Woord", "Score", "Telt als"],
           "bias": "bias", "z": "z, de som", "p": "p, de kans op positief",
           "sum": "Som, met de twee woorden na *niet* omgedraaid",
           "label": "Label", "gold": "Het eigen label van de zin: {g}.",
           "note": "Grijs: de woorden in de tabel van de classifier, en een label dat het eigen label van de zin is. "
                   "In de kleur van de tekst: de TF-IDF-waarden, waarde × weight en z, en in het lexicon de woorden, "
                   "de scores en de som. Oranje: de weights en de bias die de classifier leerde. Blauw: de kans op "
                   "positief. Gestippeld en schuin: een omgedraaide score, in de twee woorden na een ontkenning. "
                   "Paars: een ander label dan het eigen label van de zin.",
           "alt": "{title}: {sentence} {gold} Classifier: {crow}; bias {bias}, z {z}, p {p}, label {cl}. Lexicon: "
                  "{lrow}; som {sum}, label {ll}. {note}"},
}


def _turned(ws: list, negations: list) -> list:
    out, left = [], 0
    for w in ws:
        out.append(left > 0)
        left -= 1
        if w in negations:
            left = AFTER
    return out


def idea_numbers(lang: str) -> dict:
    """What the idea's picture shows: the classifier's z and p, and the lexicon's sum, for sentence IDEA."""
    m = _model(lang)
    ws, label = held_out(lang)[IDEA]
    row = features(ws, m["df"], len(training(lang)))
    z = m["bias"] + sum(v * m["weights"][w] for w, v in row.items())
    lx = _lexicon(lang)
    return {"words": ws, "label": label, "row": row, "z": z, "p": _sigmoid(z),
            "sum": score(ws, lx, NEGATIONS[lang]), "turned": _turned(ws, NEGATIONS[lang])}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    m, n = _model(lang), idea_numbers(lang)
    lx = _lexicon(lang)
    c_rows, c_alt = [], []
    for word in dict.fromkeys(n["words"]):
        v, wt = n["row"][word], m["weights"][word]
        c_rows.append([_chip(word), _dec(v, lang, 3), _chip(_dec(wt, lang, 3), _CHIP_PARAM), _dec(v * wt, lang, 3)])
        c_alt.append(f"{word} {_dec(v, lang, 3)} × {_dec(wt, lang, 3)}")
    c_rows.append([_chip(w["bias"]), "", _chip(_dec(m["bias"], lang, 3), _CHIP_PARAM), _dec(m["bias"], lang, 3)])
    c_rows.append([f'<b style="{_TEXT}">{html.escape(w["z"])}</b>', "", "", f"<b>{_dec(n['z'], lang, 3)}</b>"])
    cl = int(n["p"] >= THRESHOLD)
    classifier = (_label(w["classifier"]) + _table(c_rows, w["head_c"], (1, 2, 3))
                  + f'<div style="{_LABEL}">{html.escape(w["p"])}: {_chip(_dec(n["p"], lang, 3), _CHIP_PRED)} '
                    f'{_bar(n["p"], 1.0, "var(--ea-pred)", 6.0)} {_label_chip(cl, n["label"], lang)}</div>')
    l_rows, l_alt = [], []
    for word, t in zip(n["words"], n["turned"]):
        if word in lx:
            s = lx[word]
            counted = -s if t else s
            cell = (f'<span style="{_TURNED}">{_dec(s, lang, 2, True)} → {_dec(counted, lang, 2, True)}</span>' if t
                    else _dec(s, lang, 2, True))
            l_rows.append([f"<i>{html.escape(word)}</i>", _dec(s, lang, 2, True), cell])
            l_alt.append(f"{word} {_dec(counted, lang, 2, True)}")
    ll = label_of(n["sum"])
    lexicon_ = (_label(w["lexicon"]) + _table(l_rows, w["head_l"], (1, 2))
                + f'<div style="{_LABEL}">{_marked(w["sum"])}: <b>{_dec(n["sum"], lang, 2, True)}</b> '
                  f'{_label_chip(ll, n["label"], lang)}</div>')
    g = LABELS[lang][n["label"]]
    body = (f'<div style="{_TEXT};margin-bottom:0.2rem"><i>{html.escape(sentence(IDEA, lang))}</i></div>'
            + f'<div style="{_UI}">{html.escape(w["gold"].format(g=g))}</div>'
            + classifier + lexicon_ + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], sentence=sentence(IDEA, lang), gold=w["gold"].format(g=g), crow="; ".join(c_alt),
        bias=_dec(m["bias"], lang, 3), z=_dec(n["z"], lang, 3), p=_dec(n["p"], lang, 3), cl=LABELS[lang][cl],
        lrow="; ".join(l_alt), sum=_dec(n["sum"], lang, 2, True), ll=LABELS[lang][ll], note=w["note"]))


# --- exercise 1: the labels of your lexicon on the held-out sentences ---------------------------------------------------

_LEX_TEXT = {
    "en": {"title": "Your labels on the 600 held-out sentences",
           "head": ["Lexicon", "Positive", "Negative", "No label", "Right"],
           "rows": {"lexicon": "From the training sentences", "pattern": "Pattern's list"},
           "none": "Without a label in Pattern's list, for example:",
           "note": "Blue: what your lexicon_labels gives, counted by label, and how many labels are the sentence's "
                   "own. Purple: a number other than the course's, which then stands dashed beside it. In the color of "
                   "the text: the sentences.",
           "odd": "Your function gave something other than a list with a label for every sentence, so the picture "
                  "stays empty.",
           "stopped": "Your function stopped with an error on the held-out sentences, so the picture stays empty.",
           "slow": "Your function took longer than {s:g} seconds for the held-out sentences, so the picture stays "
                   "empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je labels op de 600 achtergehouden zinnen",
           "head": ["Lexicon", "Positief", "Negatief", "Geen label", "Goed"],
           "rows": {"lexicon": "Uit de trainingszinnen", "pattern": "Lijst van Pattern"},
           "none": "Zonder label in de lijst van Pattern, bijvoorbeeld:",
           "note": "Blauw: wat je lexicon_labels geeft, geteld per label, en hoeveel labels het eigen label van de zin "
                   "zijn. Paars: een ander getal dan dat van de course, dat er dan gestreept naast staat. In de kleur "
                   "van de tekst: de zinnen.",
           "odd": "Je functie gaf iets anders dan een lijst met een label voor elke zin, dus de afbeelding blijft "
                  "leeg.",
           "stopped": "Je functie stopte met een fout op de achtergehouden zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de achtergehouden zinnen, dus de "
                   "afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}
NONE_SHOWN = 2   # sentences without a label in Pattern's list that the picture shows


def _label_counts(said: list, gold_: list) -> list:
    return [said.count(1), said.count(0), said.count(None), sum(s == g for s, g in zip(said, gold_))]


def lexicon_picture(seen) -> str:
    lang = seen.lang
    w = _LEX_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    texts = [ws for ws, _ in held_out(lang)]
    g = gold(lang)
    lists = {"lexicon": _lexicon(lang), "pattern": _pattern(lang)}
    got = {}
    deadline = time.monotonic() + PICTURE_SECONDS
    for name, scores in lists.items():
        out = _call(lambda: fn([list(t) for t in texts], dict(scores), list(NEGATIONS[lang])), _left(deadline))
        if out is _SLOW or out is _STOPPED:
            return _empty(w, "slow" if out is _SLOW else "stopped")
        if not (isinstance(out, list) and len(out) == len(texts) and all(x in (0, 1, None) for x in out)
                and not any(isinstance(x, bool) for x in out)):
            return _empty(w, "odd")
        got[name] = out
    rows_, alts = [], []
    for name in lists:
        mine = _label_counts(got[name], g)
        want = _label_counts(lexicon_labels_of(lang, name), g)
        rows_.append([html.escape(w["rows"][name])] + [_count_chip(a, b, lang) for a, b in zip(mine, want)])
        alts.append(f"{w['rows'][name]}: " + ", ".join(f"{h} {_num(a, lang)}" for h, a in zip(w["head"][1:], mine)))
    nones = [i for i, x in enumerate(got["pattern"]) if x is None and showable(i, lang)][:NONE_SHOWN]
    body = _table(rows_, w["head"], (1, 2, 3, 4))
    if nones:
        body += _label(w["none"]) + "".join(f'<div style="{_TEXT}"><i>{html.escape(sentence(i, lang))}</i></div>'
                                             for i in nones)
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- exercise 2: your macro F1 for the three classifiers in the two languages -----------------------------------------

_F1_TEXT = {
    "en": {"title": "Your macro F1 on the 600 held-out sentences",
           "head": ["Classifier", "English", "Dutch"],
           "rows": {"classifier": "Logistic regression, the course's", "lexicon": "Lexicon from the training sentences",
                    "pattern": "Pattern's list"},
           "note": "Blue: what your macro_f1 gives for the labels of each classifier against the sentences' own. "
                   "Purple: a number other than the course's, which then stands dashed beside it.",
           "odd": "Your function gave something other than a number from 0 to 1, so the picture stays empty.",
           "stopped": "Your function stopped with an error on the held-out sentences, so the picture stays empty.",
           "slow": "Your function took longer than {s:g} seconds for the six lists of labels, so the picture stays "
                   "empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je macro F1 op de 600 achtergehouden zinnen",
           "head": ["Classifier", "Engels", "Nederlands"],
           "rows": {"classifier": "Logistische regressie, van de course", "lexicon": "Lexicon uit de trainingszinnen",
                    "pattern": "Lijst van Pattern"},
           "note": "Blauw: wat je macro_f1 geeft voor de labels van elke classifier tegen de eigen labels van de "
                   "zinnen. Paars: een ander getal dan dat van de course, dat er dan gestreept naast staat.",
           "odd": "Je functie gaf iets anders dan een getal van 0 tot 1, dus de afbeelding blijft leeg.",
           "stopped": "Je functie stopte met een fout op de achtergehouden zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de zes lijsten met labels, dus de "
                   "afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def _labels_of(name: str, lang: str) -> list:
    return classifier_labels(lang) if name == "classifier" else lexicon_labels_of(lang, name)


def macro_picture(seen) -> str:
    lang = seen.lang
    w = _F1_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    rows_, alts = [], []
    labels = {(name, lg): _labels_of(name, lg) for name in ("classifier", "lexicon", "pattern") for lg in LANGS}
    deadline = time.monotonic() + PICTURE_SECONDS
    for name in ("classifier", "lexicon", "pattern"):
        cells, said_ = [], []
        for lg in LANGS:
            g = gold(lg)
            said = labels[name, lg]
            got = _call(lambda: fn(list(g), list(said)), _left(deadline))
            if got is _SLOW or got is _STOPPED:
                return _empty(w, "slow" if got is _SLOW else "stopped")
            if not (_number(got) and 0 <= float(got) <= 1):
                return _empty(w, "odd")
            cells.append(_value_chip(got, numbers(lg)[name]["macro"], lang))
            said_.append(f"{LANGUAGE[lang][lg]} {_dec(got, lang, 3)}")
        rows_.append([html.escape(w["rows"][name])] + cells)
        alts.append(f"{w['rows'][name]}: " + ", ".join(said_))
    body = _table(rows_, w["head"], (1, 2)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- exercise 3: the mistakes the classifier and the lexicon share -----------------------------------------------------

_CMP_TEXT = {
    "en": {"title": "The mistakes of the classifier and the lexicon",
           "head": ["", "English", "Dutch"],
           "rows": ["Wrong in both", "Wrong only in the classifier", "Wrong only in the lexicon"],
           "chance": "If their mistakes had nothing to do with each other, about {en} English and {nl} Dutch sentences "
                     "would be wrong in both.",
           "note": "Blue: how many places your compare gives in each list, for the course's classifier and the "
                   "lexicon from the training sentences. Purple: a number other than the course's, which then stands "
                   "dashed beside it, with a purple bar.",
           "odd": "Your function gave something other than three lists of places, so the picture stays empty.",
           "stopped": "Your function stopped with an error on the held-out sentences, so the picture stays empty.",
           "slow": "Your function took longer than {s:g} seconds for the held-out sentences, so the picture stays "
                   "empty.",
           "alt": "{title}. {rows}. {chance} {note}"},
    "nl": {"title": "De fouten van de classifier en het lexicon",
           "head": ["", "Engels", "Nederlands"],
           "rows": ["Fout in allebei", "Alleen fout in de classifier", "Alleen fout in het lexicon"],
           "chance": "Als de fouten niets met elkaar te maken hadden, zouden ongeveer {en} Engelse en {nl} Nederlandse "
                     "zinnen fout zijn in allebei.",
           "note": "Blauw: hoeveel plaatsen je compare in elke lijst geeft, voor de classifier van de course en het "
                   "lexicon uit de trainingszinnen. Paars: een ander getal dan dat van de course, dat er dan gestreept "
                   "naast staat, met een paarse balk.",
           "odd": "Je functie gaf iets anders dan drie lijsten met plaatsen, dus de afbeelding blijft leeg.",
           "stopped": "Je functie stopte met een fout op de achtergehouden zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de achtergehouden zinnen, dus de "
                   "afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {chance} {note}"},
}


def _is_places(out, n: int) -> bool:
    return (isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, list) for x in out)
            and all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 0 <= i < n for x in out for i in x))


def compare_picture(seen) -> str:
    lang = seen.lang
    w = _CMP_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    cols = {}
    labels = {lg: (classifier_labels(lg), lexicon_labels_of(lg)) for lg in LANGS}   # outside your function's time
    deadline = time.monotonic() + PICTURE_SECONDS
    for lg in LANGS:
        g = gold(lg)
        out = _call(lambda: fn(list(g), list(labels[lg][0]), list(labels[lg][1])), _left(deadline))
        if out is _SLOW or out is _STOPPED:
            return _empty(w, "slow" if out is _SLOW else "stopped")
        if not _is_places(out, len(g)):
            return _empty(w, "odd")
        n = numbers(lg)
        cols[lg] = [(len(x), want) for x, want in zip(out, (n["both"], n["only_classifier"], n["only_lexicon"]))]
    rows_, alts = [], []
    for k, row_name in enumerate(w["rows"]):
        rows_.append([html.escape(row_name)] + [
            _count_chip(cols[lg][k][0], cols[lg][k][1], lang)
            + _bar(cols[lg][k][0], 120, "var(--ea-pred)" if cols[lg][k][0] == cols[lg][k][1] else "var(--ea-loss)", 4.0)
            for lg in LANGS])
        alts.append(f"{row_name}: " + ", ".join(f"{LANGUAGE[lang][lg]} {cols[lg][k][0]}" for lg in LANGS))
    chance = w["chance"].format(en=_num(numbers("en")["chance"], lang), nl=_num(numbers("nl")["chance"], lang))
    body = _table(rows_, w["head"], (1, 2)) + _note(chance) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), chance=chance,
                                                              note=w["note"]))


# --- the demo: your classifier trains on the review sentences in English and in Dutch -----------------------------------

DEMO_LIMIT = 180.0   # seconds the demo's training gets, for both languages
LOOKS = [1, 2, 5] + list(range(10, STEPS + 1, 10))   # the steps the rewind stops at


def project_functions() -> "dict | str":
    """The functions of your project that the demo trains with: document_frequencies and features from
    text/features.py, step from text/train.py; the course's version for a file that is missing or does not pass
    (builds_on). The name of a module that cannot be imported or lacks its function."""
    fns = {}
    for module, names in (("features", ("document_frequencies", "features")), ("train", ("step",))):
        run = run_code(f"from text.{module} import {', '.join(names)}")
        if run.error is not None or not all(callable(run.names.get(n)) for n in names):
            return module
        for n in names:
            fns[n] = run.names[n]
    fns["yours"] = [module for module in ("features", "train") if projects.state("text", module) == "yours"]
    return fns


class _Broken(Exception):
    def __init__(self, module: str):
        super().__init__(module)
        self.module = module


def _p_of(r: dict, ws: dict, b: float) -> float:
    return _sigmoid(b + sum(v * ws.get(w, 0.0) for w, v in r.items()))


def _mean_loss(rows: list, ws: dict, b: float) -> float:
    total = 0.0
    for r, label in rows:
        p = min(max(_p_of(r, ws, b), 1e-15), 1 - 1e-15)
        total -= math.log(p) if label == 1 else math.log(1 - p)
    return total / len(rows)


def _right(rows: list, ws: dict, b: float) -> int:
    return sum((_p_of(r, ws, b) >= THRESHOLD) == (label == 1) for r, label in rows)


def _train(fns: dict, lang: str, show=None) -> dict:
    train, held = training(lang), held_out(lang)
    n = len(train)
    try:
        df = fns["document_frequencies"]([[list(ws), label] for ws, label in train])
        rows = [(fns["features"](list(ws), df, n), label) for ws, label in train]
        held_rows = [(fns["features"](list(ws), df, n), label) for ws, label in held]
    except Timeout:
        raise
    except Exception as e:   # noqa: BLE001
        raise _Broken("features") from e
    if not all(isinstance(r, dict) and all(_number(v) for v in r.values()) for r, _ in rows + held_rows):
        raise _Broken("features")
    clean = [({w: float(v) for w, v in r.items()}, label) for r, label in rows]
    held_clean = [({w: float(v) for w, v in r.items()}, label) for r, label in held_rows]
    ws = {}
    for r, _ in clean:
        for word in r:
            ws[word] = 0.0
    b = 0.0
    curve, looks = [], {}
    for s in range(1, STEPS + 1):
        try:
            out = fns["step"]([(dict(r), label) for r, label in clean], dict(ws), b, RATE)
        except Timeout:
            raise
        except Exception as e:   # noqa: BLE001
            raise _Broken("train") from e
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict) and _number(out[1])
                and all(_number(out[0].get(word)) for word in ws)):
            raise _Broken("train")
        ws = {word: float(out[0][word]) for word in ws}
        b = float(out[1])
        loss = _mean_loss(clean, ws, b)
        curve.append((s, loss))
        if s in LOOKS:
            looks[s] = (dict(ws), b)
        if show is not None:
            show(lang, s, loss, ws, b, held_clean)
    return {"lang": lang, "curve": curve, "looks": looks, "weights": ws, "bias": b, "held_rows": held_clean,
            "p": [_p_of(r, ws, b) for r, _ in held_clean]}


def demo(show=None) -> "dict | str":
    """The demo's run within DEMO_LIMIT seconds: your features and step (or the course's) on the review sentences, in
    English and then in Dutch, with `show(lang, step, loss, weights, bias, held_rows)` after every step. {"en": run,
    "nl": run, "yours": [...]}, or a string when it cannot: "slow", or the module that went wrong."""
    fns = project_functions()
    if isinstance(fns, str):
        return fns

    def both():
        return {lang: _train(fns, lang, show) for lang in LANGS}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runs = with_time_limit(both, DEMO_LIMIT)
    except Timeout:
        return "slow"
    except _Broken as e:
        return e.module
    runs["yours"] = fns["yours"]
    return runs


_LIVE_TEXT = {
    "en": {"title": "Your classifier in {language}", "x": "step", "y": "log loss",
           "alt": "The mean log loss of the 2,400 {language} training sentences after each step of gradient descent.",
           "strong": "The strongest words now", "up": "Toward positive", "down": "Toward negative",
           "watched": "Three held-out sentences now", "sentence": "Sentence", "label": "Label", "p": "p",
           "verdict": "At 0.5", "after": "The strongest words after step {s}",
           "watched_after": "Three held-out sentences after step {s}",
           "right": "All held-out sentences at 0.5: {r} of the 600 get their own label.",
           "progress": "**Training in {language}** · step {s} of {n} · log loss {loss}",
           "note": "Gray: the words and the sentences' own labels. In the color of the text: the sentences. Orange: "
                   "the weights. Blue: the probability of positive. Purple: a label at 0.5 other than the sentence's "
                   "own."},
    "nl": {"title": "Je classifier in het {language}", "x": "stap", "y": "log loss",
           "alt": "De gemiddelde log loss van de 2.400 {language} trainingszinnen na elke stap gradient descent.",
           "strong": "De sterkste woorden nu", "up": "Richting positief", "down": "Richting negatief",
           "watched": "Drie achtergehouden zinnen nu", "sentence": "Zin", "label": "Label", "p": "p",
           "verdict": "Bij 0,5", "after": "De sterkste woorden na stap {s}",
           "watched_after": "Drie achtergehouden zinnen na stap {s}",
           "right": "Alle achtergehouden zinnen bij 0,5: {r} van de 600 krijgen het eigen label.",
           "progress": "**Training in het {language}** · stap {s} van {n} · log loss {loss}",
           "note": "Grijs: de woorden en de eigen labels van de zinnen. In de kleur van de tekst: de zinnen. Oranje: "
                   "de weights. Blauw: de kans op positief. Paars: een label bij 0,5 dat niet het eigen label van de "
                   "zin is."},
}
_ADJECTIVE = {"en": {"en": "English", "nl": "Dutch"}, "nl": {"en": "Engelse", "nl": "Nederlandse"}}


def live_words(lang: str, trained: str) -> dict:
    """The words of the live chart for a run in the language `trained`, in the lesson's language."""
    w = dict(_LIVE_TEXT[lang])
    w["title"] = w["title"].format(language=LANGUAGE[lang][trained])
    w["alt"] = w["alt"].format(language=_ADJECTIVE[lang][trained])
    w["progress"] = w["progress"].replace("{language}", LANGUAGE[lang][trained])
    return w


def _strongest(ws: dict, k: int = 5) -> tuple:
    ranked = sorted(ws.items(), key=lambda t: (t[1], t[0]))
    return [w for w, _ in ranked[::-1][:k]], [w for w, _ in ranked[:k]]


def _strong_table(ws: dict, lang: str, title: str) -> str:
    w = _LIVE_TEXT[lang]
    up, down = _strongest(ws)
    most = max([abs(ws[x]) for x in up + down] + [1e-9])
    rows_ = []
    for a, d in zip(up, down):
        rows_.append([_chip(a), _chip(_dec(ws[a], lang, 2), _CHIP_PARAM), _bar(ws[a], most, "var(--ea-param)", 3.5),
                      _chip(d), _chip(_dec(ws[d], lang, 2), _CHIP_PARAM), _bar(-ws[d], most, "var(--ea-param)", 3.5)])
    return _label(title) + _table(rows_, [w["up"], "", "", w["down"], "", ""], numbers_=(1, 4))


def now_parts(held_rows: list, ws: dict, b: float, lang: str, trained: str, step_: int | None = None) -> list:
    """What the demo shows beside its chart: the five strongest words each way, three held-out sentences with their
    probability of positive, and how many held-out sentences get their own label at 0.5, now (or after `step_`)."""
    w = _LIVE_TEXT[lang]
    strong = w["strong"] if step_ is None else w["after"].format(s=step_)
    watched = w["watched"] if step_ is None else w["watched_after"].format(s=step_)
    rows_ = []
    for i in WATCHED:
        r, label = held_rows[i]
        p = _p_of(r, ws, b)
        said = int(p >= THRESHOLD)
        rows_.append([_sentence(i, trained, 46), _chip(LABELS[lang][label]), _chip(_dec(p, lang, 3), _CHIP_PRED),
                      _bar(p, 1.0, "var(--ea-pred)", 4.0), _label_chip(said, label, lang)])
    return [_strong_table(ws, lang, strong),
            _label(watched) + _table(rows_, [w["sentence"], w["label"], w["p"], "", w["verdict"]], numbers_=(2,))
            + _note(w["right"].format(r=_right(held_rows, ws, b))) + _note(w["note"])]


def curve_picture(curve: list, lang: str, trained: str, at: int | None = None) -> str:
    """The log loss after each step of a run; with `at`, a line at that step, for the rewind."""
    w = live_words(lang, trained)
    top = 0.7
    p = chart.Panel((0, STEPS), (0, top), title=w["title"], xlabel=w["x"], ylabel=w["y"])
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    p.line([x for x, _ in curve], [min(y, top) for _, y in curve], "loss", 2.2)
    svg = chart.figure([p], alt=w["alt"], width=live.CHART_WIDTH, height=240)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_frame(run: dict, step_: int, lang: str) -> str:
    """The rewind at a step: the curve with a line at that step, and beside it what the classifier made then."""
    ws, b = run["looks"][step_]
    return live.rewound(curve_picture(run["curve"], lang, run["lang"], at=step_),
                        now_parts(run["held_rows"], ws, b, lang, run["lang"], step_))


def rewind_right(run: dict, step_: int) -> int:
    ws, b = run["looks"][step_]
    return _right(run["held_rows"], ws, b)


def run_labels(run: dict) -> list:
    return [int(p >= THRESHOLD) for p in run["p"]]


# --- what your project can do now: the two classifiers side by side -------------------------------------------------------

_PROJECT_TEXT = {
    "en": {"title": "Your classifier and your lexicon on the held-out sentences",
           "head": ["", "English", "Dutch"],
           "rows": ["Macro F1 of your classifier", "Macro F1 of your lexicon"],
           "groups": {"both": "Wrong in both, {n} sentences", "classifier": "Wrong only in your classifier, {n}",
                      "lexicon": "Wrong only in your lexicon, {n}"},
           "head_s": ["Sentence", "Label", "p", "Lexicon"],
           "note": "Blue: your macro F1 from your macro_f1, and the probability of positive of your classifier. Gray: "
                   "the sentences' own labels. In the color of the text: the groups your compare gives, with their "
                   "sizes, the sentences, and the lexicon's sum beside its label. Purple: a label other than the "
                   "sentence's own.",
           "course": "Where a file of yours is not in your project yet or does not pass, the course's version "
                     "counted here: {files}.",
           "alt": "{title}. {rows}. {groups} {note}"},
    "nl": {"title": "Je classifier en je lexicon op de achtergehouden zinnen",
           "head": ["", "Engels", "Nederlands"],
           "rows": ["Macro F1 van je classifier", "Macro F1 van je lexicon"],
           "groups": {"both": "Fout in allebei, {n} zinnen", "classifier": "Alleen fout in je classifier, {n}",
                      "lexicon": "Alleen fout in je lexicon, {n}"},
           "head_s": ["Zin", "Label", "p", "Lexicon"],
           "note": "Blauw: je macro F1 uit je macro_f1, en de kans op positief van je classifier. Grijs: de eigen "
                   "labels van de zinnen. In de kleur van de tekst: de groepen die je compare geeft, met de grootte, de "
                   "zinnen, en de som van het lexicon naast het label. Paars: een label dat niet het eigen label van "
                   "de zin is.",
           "course": "Waar een bestand van jou nog niet in je project staat of niet klopt, telde hier de versie "
                     "van de course: {files}.",
           "alt": "{title}. {rows}. {groups} {note}"},
}


def project_modules() -> dict:
    """lexicon_labels, macro_f1 and compare from your project, each the course's version when yours is missing or does
    not pass (builds_on); "course" lists the files that are the course's here."""
    out = {"course": []}
    for module, name in (("lexicon", "lexicon_labels"), ("scores", "macro_f1"), ("compare", "compare")):
        run = run_code(f"from text.{module} import {name}")
        fn = run.names.get(name) if run.error is None else None
        if not callable(fn) or projects.state("text", module) != "yours":
            fn = {"lexicon": lexicon_labels, "scores": macro_f1, "compare": compare}[module]
            out["course"].append(f"text/{module}.py")
        out[name] = fn
    return out


def project_results(runs: dict) -> dict:
    """Per language: the labels of your classifier and your lexicon, their macro F1 and the three lists of compare,
    with your project's functions (the course's where yours do not pass)."""
    fns = project_modules()
    out = {"course": fns["course"]}
    for lg in LANGS:
        g = gold(lg)
        mine = run_labels(runs[lg])
        texts = [ws for ws, _ in held_out(lg)]

        def labels_now():
            return fns["lexicon_labels"]([list(t) for t in texts], course_lexicon(lg), list(NEGATIONS[lg]))
        theirs = _call(labels_now, 20)
        if theirs is _STOPPED or not (isinstance(theirs, list) and len(theirs) == len(texts)):
            theirs = lexicon_labels_of(lg)
        m1 = _call(lambda: fns["macro_f1"](list(g), list(mine)), 20)
        m2 = _call(lambda: fns["macro_f1"](list(g), list(theirs)), 20)
        groups = _call(lambda: fns["compare"](list(g), list(mine), list(theirs)), 20)
        if not _number(m1) or not _number(m2):
            m1, m2 = macro_of(g, mine), macro_of(g, theirs)
        if not _is_places(groups, len(g)):
            groups = compare_of(g, mine, theirs)
        out[lg] = {"mine": mine, "theirs": theirs, "macro": (float(m1), float(m2)), "groups": groups}
    return out


def project_picture(runs: dict, lang: str) -> str:
    w = _PROJECT_TEXT[lang]
    res = project_results(runs)
    rows_ = [[html.escape(w["rows"][k])] + [_chip(_dec(res[lg]["macro"][k], lang, 3), _CHIP_PRED) for lg in LANGS]
             for k in range(2)]
    alts = [f"{w['rows'][k]}: " + ", ".join(f"{LANGUAGE[lang][lg]} {_dec(res[lg]['macro'][k], lang, 3)}" for lg in LANGS)
            for k in range(2)]
    body = _table(rows_, w["head"], (1, 2))
    r = res[lang]
    lx = _lexicon(lang)
    g = gold(lang)
    group_alts = []
    for key, places in zip(("both", "classifier", "lexicon"), r["groups"]):
        chosen = [i for i in EXAMPLES[key] if i in places and showable(i, lang)]
        # then the shortest sentences of the group that the lesson may show, which read quickest
        rest = sorted((i for i in places if i not in chosen and showable(i, lang)), key=lambda i: (len(sentence(i, lang)), i))
        chosen += rest[:SHOWN_PER_GROUP - len(chosen)]
        s_rows = []
        for i in chosen:
            total = score(held_out(lang)[i][0], lx, NEGATIONS[lang])
            p = runs[lang]["p"][i]
            s_rows.append([_sentence(i, lang, 60), _chip(LABELS[lang][g[i]]),
                           _chip(_dec(p, lang, 3), _CHIP_PRED) + _label_chip(int(p >= THRESHOLD), g[i], lang),
                           f'<span style="{_TEXT}">{_dec(total, lang, 2, True)}</span> '
                           + _label_chip(r["theirs"][i], g[i], lang)])
        title = w["groups"][key].format(n=_num(len(places), lang))
        body += _label(title) + _table(s_rows, w["head_s"], (2, 3))
        group_alts.append(title + ": " + "; ".join(sentence(i, lang) for i in chosen))
    body += _note(w["note"])
    if res["course"]:
        body += _note(w["course"].format(files=", ".join(res["course"])))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts),
                                                              groups=" ".join(group_alts), note=w["note"]))


def broken(name: str, lang: str) -> str:
    """What the demo says when it cannot run: it is too slow, or a module of your project goes wrong."""
    texts = {"en": {"slow": "*Training took longer than three minutes, so the demo stopped. Is your `step` doing more than one step of gradient descent?*",
                    "features": "*Your `work/projects/text/features.py` stops with an error on the review sentences, or gives something other than a dictionary of numbers. Check exercise 1 of *Project: a spam filter with TF-IDF* again.*",
                    "train": "*Your `work/projects/text/train.py` stops with an error on the review sentences, or `step` gives something other than a weight for every word and a bias. Check exercise 2 of *Project: a spam filter with TF-IDF* again.*"},
             "nl": {"slow": "*De training duurde langer dan drie minuten, dus de demo stopte. Doet je `step` meer dan één stap gradient descent?*",
                    "features": "*Je `work/projects/text/features.py` stopt met een fout op de zinnen uit de recensies, of geeft iets anders dan een dictionary met getallen. Check oefening 1 van *Project: een spamfilter met TF-IDF* opnieuw.*",
                    "train": "*Je `work/projects/text/train.py` stopt met een fout op de zinnen uit de recensies, of `step` geeft iets anders dan een weight voor elk woord en een bias. Check oefening 2 van *Project: een spamfilter met TF-IDF* opnieuw.*"}}
    return texts[lang].get(name, texts[lang]["features"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


def _show(v):
    """A value for a message, the way code writes it: a number with at most three decimals."""
    if isinstance(v, (bool, np.bool_)) or v is None:
        return repr(bool(v)) if isinstance(v, np.bool_) else repr(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = round(float(v), 3) + 0.0
        return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:g}"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_show(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return repr(v)


def _not_list(got, text) -> bool:
    return not isinstance(got, list)


def _not_number(got, text) -> bool:
    return not _number(got)


def _not_triple(got, text) -> bool:
    return not (isinstance(got, tuple) and len(got) == 3)


def _counts_not_lists(got, text) -> bool:
    return isinstance(got, tuple) and len(got) == 3 and all(isinstance(x, (int, np.integer)) for x in got)


REQUIRED = ["lexicon_labels", "macro_f1", "compare", "shared-mistakes"]


class _Lazy(_Val):
    """A test value made only when a check needs it, such as the held-out sentences; written by its name."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.make(), memo)


_HT = _Lazy("first_five", _real_texts)
_CL = _Lazy("course_lexicon", lambda: _lexicon("en"))
_NEG = _Lazy("negations", lambda: NEGATIONS["en"])
_GOLD_EN = _Lazy("gold_en", lambda: gold("en"))
_CLF_EN = _Lazy("classifier_en", lambda: classifier_labels("en"))
_PAT_NL = _Lazy("pattern_nl", lambda: lexicon_labels_of("nl", "pattern"))
_GOLD_NL = _Lazy("gold_nl", lambda: gold("nl"))
_LEX_EN = _Lazy("lexicon_en", lambda: lexicon_labels_of("en"))

EXERCISES = {
    "lexicon_labels": FunctionBox(
        name="lexicon_labels",
        picture=lexicon_picture,
        function="lexicon_labels",
        start=_LEX_START,
        solution=_LEX,
        show=_show,
        save_as=("text", "lexicon"),
        visible=[(_ST, _SS, SMALL_NEGATIONS)],
        hidden=[([], _SS, SMALL_NEGATIONS), ([["not", "fine", "bad"]], _SS, SMALL_NEGATIONS),
                ([["bad", "not"]], _SS, SMALL_NEGATIONS), ([["good", "bad"]], {"good": 0.5, "bad": -0.5}, ["not"]),
                ([["niet", "goed"], ["geen", "slechte", "service"]], {"goed": 0.5, "slechte": -0.8, "service": 0.1,
                                                                    "niet": -0.6}, ["niet", "geen"]),
                (_HT, _CL, _NEG)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list with a label for every sentence. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn met een label voor elke zin. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("tie-positive", wrong=_lx(start="1"),
                    en="`{call}` gives {got}, and it should give {want}: a sentence whose sum is exactly 0, such as *fine* in `small_texts`, gets 1 from you. With a sum of 0, the lexicon says nothing about the sentence. Which label means that it has no label?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een zin met een som van precies 0, zoals *fine* in `small_texts`, krijgt bij jou 1. Bij een som van 0 zegt het lexicon niets over de zin. Welk label betekent dat de zin geen label heeft?"),
            Mistake("tie-negative", wrong=_lx(start="0"),
                    en="`{call}` gives {got}, and it should give {want}: a sentence whose sum is exactly 0, such as *fine* in `small_texts`, gets 0 from you, which means negative. With a sum of 0, the lexicon says nothing about the sentence. Which label means that it has no label?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een zin met een som van precies 0, zoals *fine* in `small_texts`, krijgt bij jou 0, en dat betekent negatief. Bij een som van 0 zegt het lexicon niets over de zin. Welk label betekent dat de zin geen label heeft?"),
            Mistake("swapped", wrong=_lx(up="0", down="1"),
                    en="`{call}` gives {got}, and it should give {want}: positive and negative are the other way around. Which label belongs to a sum above 0?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: positief en negatief zijn omgedraaid. Welk label hoort bij een som boven 0?"),
            Mistake("no-negation", wrong=_lx(call="score(words, scores, [])"),
                    en="`{call}` gives {got}, and it should give {want}: *not bad* in `small_texts` counts as negative with you, so no score is turned around after *not*. What does your call of `score` get as its third argument?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *not bad* in `small_texts` telt bij jou als negatief, dus na *not* wordt geen score omgedraaid. Wat krijgt je aanroep van `score` als derde argument?"),
            Mistake("sums", wrong=_lx(add="total"),
                    en="`{call}` gives {got}, and it should give {want}: the sums themselves, not the labels. What should go into `labels` for a sum above 0?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de sommen zelf, niet de labels. Wat hoort er in `labels` bij een som boven 0?"),
            Mistake("words", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="`{call}` stops with a TypeError, *unhashable type: 'list'*: a list is looked up in the dictionary `scores`, where a single word belongs. Which list does your call of `score` get?",
                    nl="`{call}` stopt met een TypeError, *unhashable type: 'list'*: er wordt een lijst opgezocht in de dictionary `scores`, waar één woord hoort. Welke lijst krijgt je aanroep van `score`?"),
            Mistake("return-in-loop", wrong=_lx(end="        return labels\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the label of the first sentence. Your `return` stands inside the loop, so the function stops after one sentence. Where does `return labels` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het label van de eerste zin. Je `return` staat binnen de loop, dus de functie stopt na één zin. Waar hoort `return labels`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the lines of the loop: the sum of each sentence from `score`, then its label, added to `labels`.",
                      "nl": "Vervang de `...` door de regels van de loop: de som van elke zin uit `score`, dan het label, toegevoegd aan `labels`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which label does each sum get?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welk label krijgt elke som?"},
            "almost": {"en": "Right for `lexicon_labels(small_texts, small_scores, [\"not\"])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `scores` and the `negations` it gets?",
                       "nl": "Goed voor `lexicon_labels(small_texts, small_scores, [\"not\"])`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `scores` en de `negations` uit de aanroep?"},
            "correct": {"en": "Your `lexicon_labels` is right in all {n} cases. It is now in your project, in `{saved}`, together with the course's `lexicon` and `score`.",
                        "nl": "Je `lexicon_labels` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`, samen met `lexicon` en `score` van de course."},
            "kept": _kept("lexicon_labels"),
        },
    ),
    "macro_f1": FunctionBox(
        name="macro_f1",
        picture=macro_picture,
        function="macro_f1",
        start=_F1_START,
        solution=_F1,
        show=_show,
        save_as=("text", "scores"),
        visible=[(_SG, _SD)],
        hidden=[([1, 0, 1, 0], [1, 0, 1, 0]), ([1, 1, 0, 0, 0], [1, 1, 1, 1, 1]), ([1, 0], [None, None]),
                ([0, 0, 1], [1, 1, 0]), (_GOLD_EN, _CLF_EN), (_GOLD_NL, _PAT_NL)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give one number: the mean of the two F1 scores. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en het moet één getal zijn: het gemiddelde van de twee F1 scores. Wat geeft je `return`-regel terug?"),
            Mistake("zero-division", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="`{call}` stops with a ZeroDivisionError: for a label, `2 * tp + fp + fn` is 0. That happens when your counts stay 0. Do your three `if` lines count for every place `i`?",
                    nl="`{call}` stopt met een ZeroDivisionError: voor een label is `2 * tp + fp + fn` 0. Dat gebeurt als je aantallen 0 blijven. Tellen je drie `if`-regels voor elke plaats `i`?"),
            Mistake("accuracy", wrong=_f1(add="0", end="    right = 0\n    for i in range(len(gold)):\n        if gold[i] == said[i]:\n            right = right + 1\n    return right / len(gold)\n"),
                    en="`{call}` gives {got}, and it should give {want}: the share of the sentences that get their own label. Macro F1 is the mean of the F1 score of each label. Which counts does the F1 score of a label need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aandeel van de zinnen dat het eigen label krijgt. Macro F1 is het gemiddelde van de F1 score van elk label. Welke aantallen heeft de F1 score van een label nodig?"),
            Mistake("one-label", wrong=_f1(labels="[1]"),
                    en="`{call}` gives {got}, and it should give {want}: half of the F1 score of label 1, and nothing for label 0. Does your function get to label 0 before it gives back its answer?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de helft van de F1 score van label 1, en niets voor label 0. Komt label 0 aan de beurt vóór je functie het antwoord teruggeeft?"),
            Mistake("none-negative", wrong=_f1(fp="gold[i] != label and (said[i] == label or (said[i] is None and label == 0))",
                                               tp="gold[i] == label and (said[i] == label or (said[i] is None and label == 0))",
                                               fn="gold[i] == label and said[i] != label and not (said[i] is None and label == 0)"),
                    en="`{call}` gives {got}, and it should give {want}: a None in `said` counts as label 0 with you. None is a label for neither: for which of the three counts can a None count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een None in `said` telt bij jou als label 0. None is voor geen van beide een label: bij welke van de drie aantallen kan een None meetellen?"),
            Mistake("precision", wrong=_f1(add="tp / (tp + fp)"),
                    en="`{call}` gives {got}, and it should give {want}: the mean of the precision of the labels. The F1 score also counts the false negatives. What does it divide `2 * tp` by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het gemiddelde van de precision van de labels. De F1 score telt ook de false negatives mee. Waardoor deelt de F1 score `2 * tp`?"),
            Mistake("recall", wrong=_f1(add="tp / (tp + fn)"),
                    en="`{call}` gives {got}, and it should give {want}: the mean of the recall of the labels. The F1 score also counts the false positives. What does it divide `2 * tp` by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het gemiddelde van de recall van de labels. De F1 score telt ook de false positives mee. Waardoor deelt de F1 score `2 * tp`?"),
            Mistake("no-two", wrong=_f1(add="tp / (tp + fp + fn)"),
                    en="`{call}` gives {got}, and it should give {want}: each F1 score is too small. The F1 score of a label is $2\\,TP / (2\\,TP + FP + FN)$: where does the 2 come in with you?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke F1 score is te klein. De F1 score van een label is $2\\,TP / (2\\,TP + FP + FN)$: waar komt de 2 bij jou terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the counts of one label, a loop over the places `i` with three `if` lines, and the line that adds its F1 score to `total`.",
                      "nl": "Vervang de `...` door de aantallen van één label, een loop over de plaatsen `i` met drie `if`-regels, en de regel die de F1 score optelt bij `total`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What are TP, FP and FN of each label?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wat zijn TP, FP en FN van elk label?"},
            "almost": {"en": "Right for `macro_f1(small_gold, small_said)`. But `{call}` gives {got}, and it should give {want}. Does your function work for lists of another length, and for a list without a single right label?",
                       "nl": "Goed voor `macro_f1(small_gold, small_said)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor lijsten met een andere lengte, en voor een lijst zonder één goed label?"},
            "correct": {"en": "Your `macro_f1` is right in all {n} cases. It is now in your project, in `{saved}`.",
                        "nl": "Je `macro_f1` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`."},
            "kept": _kept("macro_f1"),
        },
    ),
    "compare": FunctionBox(
        name="compare",
        picture=compare_picture,
        function="compare",
        start=_CMP_START,
        solution=_CMP,
        show=_show,
        save_as=("text", "compare"),
        visible=[(_SG, _SM, _STH)],
        hidden=[([], [], []), ([1, 0, 1], [1, 0, 1], [1, 0, 1]), ([1, 0], [0, 1], [0, None]), ([1, 0, 0], [1, 1, 0], [0, 0, 1]),
                (_GOLD_EN, _CLF_EN, _LEX_EN)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-triple", matches=_not_triple,
                    en="`{call}` gives {got}, and it should give three lists. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet drie lijsten zijn. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("counts", matches=_counts_not_lists,
                    en="`{call}` gives {got}: how many places each list has. The picture under *What your project can do now* needs the places themselves, to show the sentences. What goes into each list?",
                    nl="`{call}` geeft {got}: hoeveel plaatsen elke lijst heeft. De afbeelding onder *Wat je project nu kan* heeft de plaatsen zelf nodig, om de zinnen te laten zien. Wat komt er in elke lijst?"),
            Mistake("right", wrong=_cmp(both="mine[i] == gold[i] and theirs[i] == gold[i]",
                                        mine="mine[i] == gold[i] and theirs[i] != gold[i]",
                                        theirs="mine[i] != gold[i] and theirs[i] == gold[i]"),
                    en="`{call}` gives {got}, and it should give {want}: your lists hold the places where the labels are right. Which places does a list of mistakes hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je lijsten bevatten de plaatsen waar de labels goed zijn. Welke plaatsen hoort een lijst met fouten te bevatten?"),
            Mistake("swapped", wrong=_cmp(mine="mine[i] == gold[i] and theirs[i] != gold[i]",
                                          theirs="mine[i] != gold[i] and theirs[i] == gold[i]"),
                    en="`{call}` gives {got}, and it should give {want}: the two lists of mistakes made by only one are the other way around. In which list does a place go where only `mine` is wrong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de twee lijsten met fouten van maar één van beide zijn omgedraaid. In welke lijst komt een plaats waar alleen `mine` fout is?"),
            Mistake("not-only", wrong=_cmp(mine="mine[i] != gold[i]", theirs="theirs[i] != gold[i]"),
                    en="`{call}` gives {got}, and it should give {want}: a place that both get wrong is also in the lists of each alone. What should be true of `theirs[i]` for a mistake that only `mine` makes?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een plaats die allebei fout hebben, staat ook in de lijsten van elk apart. Wat moet er gelden voor `theirs[i]` bij een fout die alleen `mine` maakt?"),
            Mistake("none-right", wrong=_cmp(both="mine[i] != gold[i] and theirs[i] != gold[i] and theirs[i] is not None",
                                             mine="mine[i] != gold[i] and (theirs[i] == gold[i] or theirs[i] is None)",
                                             theirs="mine[i] == gold[i] and theirs[i] != gold[i] and theirs[i] is not None"),
                    en="`{call}` gives {got}, and it should give {want}: a None in `theirs` counts as right with you. A sentence without a label did not get its own label: is it right or wrong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een None in `theirs` telt bij jou als goed. Een zin zonder label kreeg het eigen label niet: is dat goed of fout?"),
            Mistake("labels", wrong=_cmp(add="gold[i]"),
                    en="`{call}` gives {got}, and it should give {want}: your lists hold the labels of the sentences, not their places. What should go into a list with `append`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je lijsten bevatten de labels van de zinnen, niet de plaatsen. Wat hoort er met `append` in een lijst?"),
            Mistake("return-in-loop", wrong=_cmp(end="        return both, only_mine, only_theirs\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first place is looked at. Your `return` stands inside the loop, so the function stops after one sentence. Where does the `return` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen naar de eerste plaats wordt gekeken. Je `return` staat binnen de loop, dus de functie stopt na één zin. Waar hoort de `return`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with three `if` lines, one for each list, that compare `mine[i]` and `theirs[i]` with `gold[i]`.",
                      "nl": "Vervang de `...` door drie `if`-regels, één voor elke lijst, die `mine[i]` en `theirs[i]` vergelijken met `gold[i]`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which places are wrong in both, and which in only one?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke plaatsen zijn fout in allebei, en welke in maar één?"},
            "almost": {"en": "Right for `compare(small_gold, small_mine, small_theirs)`. But `{call}` gives {got}, and it should give {want}. Does your function go over every place of `gold`, also in lists of another length?",
                       "nl": "Goed voor `compare(small_gold, small_mine, small_theirs)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gaat je functie over elke plaats van `gold`, ook bij lijsten met een andere lengte?"},
            "correct": {"en": "Your `compare` is right in all {n} cases. It is now in your project, in `{saved}`.",
                        "nl": "Je `compare` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`."},
            "kept": _kept("compare"),
        },
    ),
}

CASES = [
    ("lexicon_labels", _LEX_START, "empty", None),
    ("lexicon_labels", _LEX, "correct", None),
    ("lexicon_labels", _LEX_HEAD + "        labels.append(1)\n    return labels\n", "wrong", None),
    ("lexicon_labels", _LEX_HEAD + "        labels.append(1)\n    return len(labels)\n", "wrong", "not-list"),
    ("lexicon_labels", _lx(start="1"), "wrong", "tie-positive"),
    ("lexicon_labels", _lx(start="0"), "wrong", "tie-negative"),
    ("lexicon_labels", _lx(up="0", down="1"), "wrong", "swapped"),
    ("lexicon_labels", _lx(call="score(words, scores, [])"), "wrong", "no-negation"),
    ("lexicon_labels", _lx(add="total"), "wrong", "sums"),
    ("lexicon_labels", _lx(call="score(texts, scores, negations)"), "error", "words"),
    ("lexicon_labels", _lx(end="        return labels\n"), "wrong", "return-in-loop"),
    ("lexicon_labels", _lx(call="score(words, scores, ['not'])"), "almost", None),
    ("lexicon_labels", _LEX_HEAD + "        total = score(words, scores, negations)\n        if total > 0:\n"
     "            labels.append(1)\n        elif total < 0:\n            labels.append(0)\n        else:\n"
     "            labels.append(None)\n    return labels\n", "correct", None),
    ("macro_f1", _F1_START, "empty", None),
    ("macro_f1", _F1, "correct", None),
    ("macro_f1", _F1_HEAD + "        total = total + 1\n    return total / 2\n", "wrong", None),
    ("macro_f1", _f1(end="    return [total]\n"), "wrong", "not-number"),
    ("macro_f1", _f1(tp="False", fp="False", fn="False"), "error", "zero-division"),
    ("macro_f1", _f1(add="0", end="    right = 0\n    for i in range(len(gold)):\n        if gold[i] == said[i]:\n"
                               "            right = right + 1\n    return right / len(gold)\n"), "wrong", "accuracy"),
    ("macro_f1", _f1(labels="[1]"), "wrong", "one-label"),
    ("macro_f1", _f1(fp="gold[i] != label and (said[i] == label or (said[i] is None and label == 0))",
                     tp="gold[i] == label and (said[i] == label or (said[i] is None and label == 0))",
                     fn="gold[i] == label and said[i] != label and not (said[i] is None and label == 0)"),
     "wrong", "none-negative"),
    ("macro_f1", _f1(add="tp / (tp + fp)"), "wrong", "precision"),
    ("macro_f1", _f1(add="tp / (tp + fn)"), "wrong", "recall"),
    ("macro_f1", _f1(add="tp / (tp + fp + fn)"), "wrong", "no-two"),
    ("macro_f1", _f1(end="        return total / 2\n"), "wrong", "one-label"),
    ("macro_f1", _f1(tp="gold[i] == label and said[i] == label and i < 6"), "almost", None),
    ("macro_f1", "def macro_f1(gold, said):\n    scores = []\n    for label in [1, 0]:\n"
     "        tp = sum(1 for g, s in zip(gold, said) if g == label and s == label)\n"
     "        fp = sum(1 for g, s in zip(gold, said) if g != label and s == label)\n"
     "        fn = sum(1 for g, s in zip(gold, said) if g == label and s != label)\n"
     "        scores.append(2 * tp / (2 * tp + fp + fn))\n    return sum(scores) / len(scores)\n", "correct", None),
    ("compare", _CMP_START, "empty", None),
    ("compare", _CMP, "correct", None),
    ("compare", _CMP_HEAD + "        both.append(i)\n    return both, only_mine, only_theirs\n", "wrong", None),
    ("compare", _cmp(end="    return both\n"), "wrong", "not-triple"),
    ("compare", _cmp(end="    return len(both), len(only_mine), len(only_theirs)\n"), "wrong", "counts"),
    ("compare", _cmp(both="mine[i] == gold[i] and theirs[i] == gold[i]", mine="mine[i] == gold[i] and theirs[i] != gold[i]",
                     theirs="mine[i] != gold[i] and theirs[i] == gold[i]"), "wrong", "right"),
    ("compare", _cmp(mine="mine[i] == gold[i] and theirs[i] != gold[i]", theirs="mine[i] != gold[i] and theirs[i] == gold[i]"),
     "wrong", "swapped"),
    ("compare", _cmp(mine="mine[i] != gold[i]", theirs="theirs[i] != gold[i]"), "wrong", "not-only"),
    ("compare", _cmp(both="mine[i] != gold[i] and theirs[i] != gold[i] and theirs[i] is not None",
                     mine="mine[i] != gold[i] and (theirs[i] == gold[i] or theirs[i] is None)",
                     theirs="mine[i] == gold[i] and theirs[i] != gold[i] and theirs[i] is not None"), "wrong", "none-right"),
    ("compare", _cmp(add="gold[i]"), "wrong", "labels"),
    ("compare", _cmp(end="        return both, only_mine, only_theirs\n"), "wrong", "return-in-loop"),
    ("compare", _cmp(both="mine[i] != gold[i] and theirs[i] != gold[i] and i < 6"), "almost", None),
    ("compare", _CMP_HEAD + "        mine_wrong = mine[i] != gold[i]\n        theirs_wrong = theirs[i] != gold[i]\n"
     "        if mine_wrong and theirs_wrong:\n            both.append(i)\n        elif mine_wrong:\n"
     "            only_mine.append(i)\n        elif theirs_wrong:\n            only_theirs.append(i)\n"
     "    return both, only_mine, only_theirs\n", "correct", None),
]
