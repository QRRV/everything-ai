# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-micro-macro: the true positives, false positives and false negatives of one label among
many (counts), the macro F1, the plain mean of the F1 scores of the labels (macro_f1), and the micro F1, the F1 of the
counts added up over the labels (micro_f1); and sort situations by the average that fits them.

The commands are the English commands of MASSIVE 1.1 (FitzGerald and others, ACL 2023; Amazon, CC BY 4.0,
https://doi.org/10.18653/v1/2023.acl-long.235), in data/massive-en-nl.tsv.gz, made by data/tools/make_massive.py; the
Dutch commands of the same file give the Dutch numbers in the lesson's text. Commands marked as not shown are counted
but never shown. The confusion matrix of the worked example is that of the example with three labels in Jurafsky and
Martin, Speech and Language Processing, 3rd edition draft, section 4.9.1, with a row for each true label. The other
texts, the cases of the checks and the pictures are the course's own."""
import collections
import contextlib
import copy
import functools
import html
import io
import math
import re

import numpy as np

from everything_ai import books, chart, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, run_code, with_time_limit

# --- the commands ------------------------------------------------------------------------------------------------------

K = 0.1   # the add-k smoothing of naive Bayes
KNOWN = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]   # the stops of the demo: how many intents the classifier knows
_COLUMN = {"en": 5, "nl": 6}   # where books.assistant_commands() keeps the English and the Dutch command


def words(text: str) -> list:
    """The words of a command in small letters: runs of letters, with an apostrophe inside a word as in *what's*."""
    return books.words(text.lower())


@functools.cache
def _part(part: str, lang: str) -> tuple:
    col = _COLUMN[lang]
    return tuple((r[3], r[col], r[4]) for r in books.assistant_commands() if r[1] == part)


def training(lang: str = "en") -> list:
    """The 11,514 training commands of MASSIVE, as (intent, text, shown)."""
    return list(_part("train", lang))


def test(lang: str = "en") -> list:
    """The 2,974 test commands of MASSIVE, as (intent, text, shown)."""
    return list(_part("test", lang))


@functools.cache
def _gold() -> tuple:
    return tuple(intent for intent, _, _ in _part("test", "en"))


def gold() -> list:
    """The true intent of each test command, in order."""
    return list(_gold())


def labels() -> list:
    """The 59 intents of the test commands, in the order of the alphabet. cooking_query is only in the training
    commands, with 4."""
    return sorted(set(_gold()))


def train_counts() -> dict:
    """How many training commands each of the 60 intents has."""
    return dict(collections.Counter(intent for intent, _, _ in _part("train", "en")))


def test_counts() -> dict:
    return dict(collections.Counter(_gold()))


@functools.cache
def _by_count() -> tuple:
    tc = train_counts()
    return tuple(sorted(tc, key=lambda intent: (-tc[intent], intent)))


def by_count() -> list:
    """The 60 intents of the training commands, from the most training commands to the fewest; the same number in the
    order of the alphabet."""
    return list(_by_count())


def labels_by_count() -> list:
    """The 59 intents of the test commands in the order of by_count(): the order of the bars in the pictures."""
    test_ = set(_gold())
    return [intent for intent in _by_count() if intent in test_]


def _scores(lang: str, prior: bool) -> np.ndarray:
    """For each test command, the log score of naive Bayes for each of the 60 intents, in the order of by_count():
    log P(intent), when `prior`, plus log P(word | intent) for each word of the command the training commands have,
    with add-k smoothing (k = K) over the vocabulary of all training commands."""
    def make():
        order = by_count()
        row = {intent: k for k, intent in enumerate(order)}
        train = training(lang)
        vocab = sorted({w for _, text, _ in train for w in words(text)})
        col = {w: k for k, w in enumerate(vocab)}
        table = np.zeros((len(order), len(vocab)))
        n = np.zeros(len(order))
        for intent, text, _ in train:
            n[row[intent]] += 1
            for w in words(text):
                table[row[intent], col[w]] += 1
        logp = np.log((table + K) / (table.sum(axis=1, keepdims=True) + K * len(vocab)))
        start = np.log(n / n.sum()) if prior else np.zeros(len(order))
        out = np.zeros((len(test(lang)), len(order)))
        for k, (_, text, _) in enumerate(test(lang)):
            idx = [col[w] for w in words(text) if w in col]
            out[k] = start + logp[:, idx].sum(axis=1)
        return out
    return kept(f"nlp-micro-macro:scores:{lang}:{prior}", make)


def said_known(k: int, lang: str = "en", prior: bool = True) -> list:
    """For each test command, the intent naive Bayes says when it knows only the k most common intents of by_count():
    the one with the highest score among them, the first in that order on a tie. It learns each of them from the same
    training commands, so knowing fewer intents only takes the others away."""
    order = _by_count()
    best = np.argmax(_scores(lang, prior)[:, :k], axis=1)
    return [order[int(i)] for i in best]


def said(lang: str = "en") -> list:
    """For each test command, the intent naive Bayes says: 2,437 of the 2,974 English commands get their own."""
    return said_known(len(_by_count()), lang)


def said_flat() -> list:
    """The same without log P(intent), as if every intent were equally common: 2,434 right."""
    return said_known(len(_by_count()), "en", prior=False)


# --- the course's own answers ------------------------------------------------------------------------------------------

def counts(gold, said, label):
    tp = fp = fn = 0
    for g, s in zip(gold, said):
        if g == label and s == label:
            tp += 1
        elif s == label:
            fp += 1
        elif g == label:
            fn += 1
    return tp, fp, fn


def f1(tp, fp, fn) -> float:
    """2 TP / (2 TP + FP + FN), and 0 for a label with no counts at all."""
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def all_counts(gold_, said_, labels_) -> dict:
    """{label: (tp, fp, fn)} for every label, in one pass over the commands."""
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for g, s in zip(gold_, said_):
        if g == s:
            tp[g] += 1
        else:
            fp[s] += 1
            fn[g] += 1
    return {label: (tp[label], fp[label], fn[label]) for label in labels_}


def macro_f1(gold_, said_, labels_) -> float:
    c = all_counts(gold_, said_, labels_)
    return sum(f1(*c[label]) for label in labels_) / len(labels_)


def micro_f1(gold_, said_, labels_) -> float:
    c = all_counts(gold_, said_, labels_)
    tp, fp, fn = (sum(v[j] for v in c.values()) for j in range(3))
    return f1(tp, fp, fn)


def scores(said_=None) -> dict:
    """{"macro", "micro", "right", "f1": {label: F1}} of a list of said intents on the test commands, said() if none."""
    said_ = said() if said_ is None else said_
    g, labels_ = _gold(), labels()
    c = all_counts(g, said_, labels_)
    return {"macro": macro_f1(g, said_, labels_), "micro": micro_f1(g, said_, labels_),
            "right": sum(1 for a, b in zip(g, said_) if a == b), "counts": c,
            "f1": {label: f1(*c[label]) for label in labels_}}


def dutch_scores() -> dict:
    """The same for the Dutch commands, which the Dutch lesson names in a note: micro F1 0.799, macro F1 0.730."""
    g = [intent for intent, _, _ in test("nl")]
    s = said("nl")
    c = all_counts(g, s, labels())
    return {"macro": macro_f1(g, s, labels()), "micro": micro_f1(g, s, labels()),
            "zero": sorted(label for label in labels() if c[label][0] == 0)}


def known_scores(k: int) -> dict:
    s, g, labels_ = said_known(k), _gold(), labels()
    known = set(_by_count()[:k])
    return {"macro": macro_f1(g, s, labels_), "micro": micro_f1(g, s, labels_),
            "covered": sum(1 for x in g if x in known)}


def example(k: int) -> tuple:
    """(text, shown, true intent, said intent) of test command k."""
    intent, text, shown = test()[k]
    return text, shown, intent, said()[k]


# --- the small examples ------------------------------------------------------------------------------------------------

SMALL_GOLD = ["play_music", "play_music", "alarm_set", "play_music", "play_music", "play_music"]
SMALL_SAID = ["play_music", "play_music", "play_music", "play_music", "play_music", "play_music"]
SMALL_LABELS = ["alarm_set", "play_music"]
MORE_GOLD = ["weather_query", "weather_query", "weather_query", "alarm_set", "alarm_set", "play_music", "play_music",
             "play_music"]
MORE_SAID = ["weather_query", "weather_query", "alarm_set", "alarm_set", "play_music", "play_music", "play_music",
             "weather_query"]
MORE_LABELS = ["alarm_set", "play_music", "weather_query"]

# The example with three labels of Jurafsky and Martin, section 4.9.1: a row for each true label (urgent, normal,
# spam) and a column for each label the system gave, as in the lesson Beyond accuracy (their figure has the system's
# labels in the rows).
SLP_NAMES = ["urgent", "normal", "spam"]
SLP_MATRIX = [[8, 5, 3], [10, 60, 30], [1, 50, 200]]

COUNT_INTENTS = ["calendar_set", "play_music", "general_quirky", "weather_query", "audio_volume_other",
                 "music_dislikeness", "iot_hue_lighton", "general_greet"]   # the eight the picture under exercise 1 shows


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"


def _marked(text: str) -> str:
    parts_ = html.escape(text).split("*")
    return "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts_))


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,974 in English, 2.974 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 3) -> str:
    """A number with `digits` decimals: 0.752 in English, 0,752 in Dutch."""
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 8.0) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


_SAY = {
    "en": {"stopped": "Your function stopped with an error on the test commands, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the test commands, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op de testopdrachten, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de testopdrachten, dus de afbeelding blijft leeg."},
}


# --- a chart of the F1 score of every intent ---------------------------------------------------------------------------

_CHART_TEXT = {
    "en": {"x": "the 59 intents, from most to fewest training commands", "y": "F1 score",
           "macro": "macro {v}", "micro": "micro {v}", "unknown": "not known"},
    "nl": {"x": "de 59 intents, van de meeste naar de minste trainingsopdrachten", "y": "F1 score",
           "macro": "macro {v}", "micro": "micro {v}", "unknown": "niet bekend"},
}


def _f1_panel(f1s: dict, lang: str, macro=None, micro=None, known: int | None = None, macro_role: str = "ink",
              right_macro=None, title: str = "") -> chart.Panel:
    """A bar for the F1 score of each of the 59 intents, gray, in the order of labels_by_count(); a solid line for the
    macro F1 and a dashed one for the micro F1, when given; with `known`, a purple wash over the intents the classifier
    does not know; with `right_macro`, the right macro F1 dashed in purple beside a wrong one."""
    w = _CHART_TEXT[lang]
    order = labels_by_count()
    n = len(order)
    # room at the right of the bars for the words of the lines, so that they never sit on a bar
    p = chart.Panel(x=(0, n + 18), y=(0, 1), title=title, xlabel=w["x"], ylabel=w["y"],
                    xticks=[0.5, 9.5, 19.5, 29.5, 39.5, 49.5, n - 0.5],
                    xtick_labels=["1", "10", "20", "30", "40", "50", str(n)], yticks=[0, 0.25, 0.5, 0.75, 1])
    if known is not None and known < n:
        p.rect(known, 0, n, 1, role="loss")
        if n - known >= 12:
            p.text(known + (n - known) / 2, 0.95, w["unknown"], role="loss")
    for k, label in enumerate(order):
        v = f1s[label]
        if v > 0:
            p.rect(k + 0.12, 0, k + 0.88, v, role="data", solid=True)
    labels_at = []
    if micro is not None:
        p.line([0, n], [micro, micro], role="ink", width=1.6, dashed=True)
        labels_at.append((micro, w["micro"].format(v=_dec(micro, lang)), "ink"))
    if right_macro is not None:
        p.line([0, n], [right_macro, right_macro], role="loss", width=1.4, dashed=True)
    if macro is not None:
        p.line([0, n], [macro, macro], role=macro_role, width=1.8)
        labels_at.append((macro, w["macro"].format(v=_dec(macro, lang)), macro_role))
    # the words of the lines right of the bars, each at the height of its line; two lines close together get their
    # words moved apart, the higher one's up and the lower one's down
    labels_at.sort(key=lambda t: -t[0])
    ys = [v for v, _, _ in labels_at]
    if len(ys) == 2 and ys[0] - ys[1] < 0.08:
        middle = (ys[0] + ys[1]) / 2
        ys = [middle + 0.04, middle - 0.04]
    ys = [min(max(y, 0.06), 0.98) for y in ys]
    if len(ys) == 2 and ys[0] - ys[1] < 0.08:
        ys = [ys[1] + 0.08, ys[1]] if ys[1] < 0.5 else [ys[0], ys[0] - 0.08]
    for (_, text, role), y in zip(labels_at, ys):
        p.text(n + 0.6, y - 0.02, text, role=role, anchor="start")
    return p


def _alt_f1s(f1s: dict, lang: str) -> str:
    order = labels_by_count()
    return "; ".join(f"{label} {_dec(f1s[label], lang, 2)}" for label in order)


# --- the idea's picture ------------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "The F1 score of each intent, and the two averages",
           "note": "Naive Bayes on the 2,974 test commands. Gray: the F1 score of one intent. Solid line: the macro "
                   "F1, the plain mean of the 59 bars. Dashed line: the micro F1, from the counts of all intents "
                   "added up.",
           "alt": "{title}. Bars from left to right: {rows}. {note}"},
    "nl": {"title": "De F1 score van elke intent, en de twee gemiddelden",
           "note": "Naive Bayes op de 2.974 testopdrachten. Grijs: de F1 score van één intent. Doorgetrokken lijn: "
                   "de macro F1, het gewone gemiddelde van de 59 balken. Gestreepte lijn: de micro F1, uit de "
                   "aantallen van alle intents bij elkaar opgeteld.",
           "alt": "{title}. Balken van links naar rechts: {rows}. {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    s = scores()
    p = _f1_panel(s["f1"], lang, macro=s["macro"], micro=s["micro"])
    svg = chart.figure([p], alt=w["alt"].format(title=w["title"], rows=_alt_f1s(s["f1"], lang), note=w["note"]),
                       width=460, height=290)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return (f'<div style="{_LABEL};font-weight:600;color:var(--ea-ink)">{html.escape(w["title"])}</div>'
            + svg + _note(w["note"]))


# --- the picture under exercise 1: your counts for eight intents -------------------------------------------------------

_COUNT_TEXT = {
    "en": {"title": "Your counts for eight intents, on the 2,974 test commands",
           "head": ["Intent", "Test commands", "TP", "FP", "FN"],
           "common": "The four most common in the test commands:",
           "rare": "Four of the least common:",
           "note": "TP: commands of the intent that naive Bayes gives this intent. FP: commands of another intent that "
                   "get this intent. FN: commands of the intent that get another one. Gray: your counts. Purple: a "
                   "count that should be another one; dashed: the right one.",
           "odd": "Your function gave something other than three whole numbers, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je aantallen voor acht intents, op de 2.974 testopdrachten",
           "head": ["Intent", "Testopdrachten", "TP", "FP", "FN"],
           "common": "De vier die het vaakst voorkomen in de testopdrachten:",
           "rare": "Vier van de intents die het minst voorkomen:",
           "note": "TP: opdrachten van de intent die van naive Bayes deze intent krijgen. FP: opdrachten van een "
                   "andere intent die deze intent krijgen. FN: opdrachten van de intent die een andere intent krijgen. "
                   "Grijs: je aantallen. Paars: een aantal dat niet klopt; met een gestreepte rand: het goede aantal.",
           "odd": "Je functie gaf iets anders dan drie hele getallen, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def _three(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) == 3 and all(_number(x) and float(x) == int(x) for x in v)


def counts_picture(seen) -> str:
    lang = seen.lang
    w = _COUNT_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    g, s = gold(), said()
    got, said_ = _run_or_say(lambda: [fn(list(g), list(s), label) for label in COUNT_INTENTS], w,
                             lambda out: all(_three(v) for v in out))
    if said_:
        return said_
    right = all_counts(g, s, labels())
    tcounts = test_counts()

    def cell(mine_, want):
        if int(mine_) == want:
            return _chip(_num(mine_, lang))
        return _chip(_num(mine_, lang), _CHIP_PURPLE) + _chip(_num(want, lang), _CHIP_MISSING)
    rows_, alts = [], []
    for label, mine_ in zip(COUNT_INTENTS, got):
        rows_.append([_chip(label), _num(tcounts[label], lang)] + [cell(m, r) for m, r in zip(mine_, right[label])])
        alts.append(f"{label}, {_num(tcounts[label], lang)}: " + ", ".join(_num(m, lang) for m in mine_))
    body = (_label(w["common"]) + _table(rows_[:4], w["head"], numbers=(1, 2, 3, 4)) + _label(w["rare"])
            + _table(rows_[4:], w["head"], numbers=(1, 2, 3, 4)) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 2: your macro F1 ------------------------------------------------------------------------

_MACRO_TEXT = {
    "en": {"title": "Your macro F1 on the 2,974 test commands",
           "yours": "Your macro F1: {v}, the mean of the 59 bars.",
           "yours_wrong": "Your macro F1: {v}.",
           "note": "Gray: the F1 score of each intent, from the counts of naive Bayes. Solid line: your macro F1; "
                   "purple when it should be another number, which then stands dashed beside it.",
           "odd": "Your function gave something other than a number from 0 to 1, so the picture stays empty.",
           "alt": "{title}. {yours} Bars from left to right: {rows}. {note}"},
    "nl": {"title": "Je macro F1 op de 2.974 testopdrachten",
           "yours": "Je macro F1: {v}, het gemiddelde van de 59 balken.",
           "yours_wrong": "Je macro F1: {v}.",
           "note": "Grijs: de F1 score van elke intent, uit de aantallen van naive Bayes. Doorgetrokken lijn: je "
                   "macro F1; paars als het een ander getal moet zijn, dat er dan gestreept naast staat.",
           "odd": "Je functie gaf iets anders dan een getal van 0 tot 1, dus de afbeelding blijft leeg.",
           "alt": "{title}. {yours} Balken van links naar rechts: {rows}. {note}"},
}


def macro_picture(seen) -> str:
    lang = seen.lang
    w = _MACRO_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got, said_ = _run_or_say(lambda: fn(gold(), said(), labels()), w, lambda v: _number(v) and 0 <= float(v) <= 1)
    if said_:
        return said_
    s = scores()
    mine_ = float(got)
    ok = math.isclose(mine_, s["macro"], rel_tol=1e-6, abs_tol=1e-9)
    p = _f1_panel(s["f1"], lang, macro=mine_, macro_role="ink" if ok else "loss",
                  right_macro=None if ok else s["macro"])
    yours_text = w["yours" if ok else "yours_wrong"].format(v=_dec(mine_, lang, 4))
    svg = chart.figure([p], alt=w["alt"].format(title=w["title"], yours=yours_text, rows=_alt_f1s(s["f1"], lang),
                                                note=w["note"]), width=460, height=290)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    head, tail = w["yours" if ok else "yours_wrong"].split("{v}")
    v = _chip(_dec(mine_, lang, 4)) if ok else _chip(_dec(mine_, lang, 4), _CHIP_PURPLE) + _chip(_dec(s["macro"], lang, 4), _CHIP_MISSING)
    yours_html = f'<div style="{_LABEL}">{html.escape(head)}{v}{html.escape(tail)}</div>'
    return yours.figure(w["title"], yours_html + svg + _note(w["note"]), alt="")


# --- the picture under exercise 3: your micro F1, against the share right ---------------------------------------------

_MICRO_TEXT = {
    "en": {"title": "Your micro F1 on the 2,974 test commands",
           "pooled": "Added up over the 59 intents: TP {tp}, FP {fp}, FN {fn}.",
           "head": ["Number", "Value", ""],
           "rows": ["Your micro F1", "Share of the test commands that get their own intent", "Macro F1"],
           "note": "Gray: your micro F1, and the course's numbers for the other two. Purple: your micro F1 when it "
                   "should be another number, which then stands dashed beside it.",
           "odd": "Your function gave something other than a number from 0 to 1, so the picture stays empty.",
           "alt": "{title}. {pooled} {rows}. {note}"},
    "nl": {"title": "Je micro F1 op de 2.974 testopdrachten",
           "pooled": "Opgeteld over de 59 intents: TP {tp}, FP {fp}, FN {fn}.",
           "head": ["Getal", "Waarde", ""],
           "rows": ["Je micro F1", "Aandeel van de testopdrachten dat de eigen intent krijgt", "Macro F1"],
           "note": "Grijs: je micro F1, en de getallen van de course voor de andere twee. Paars: je micro F1 als het "
                   "een ander getal moet zijn, dat er dan gestreept naast staat.",
           "odd": "Je functie gaf iets anders dan een getal van 0 tot 1, dus de afbeelding blijft leeg.",
           "alt": "{title}. {pooled} {rows}. {note}"},
}


def micro_picture(seen) -> str:
    lang = seen.lang
    w = _MICRO_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got, said_ = _run_or_say(lambda: fn(gold(), said(), labels()), w, lambda v: _number(v) and 0 <= float(v) <= 1)
    if said_:
        return said_
    s = scores()
    tp, fp, fn_ = (sum(v[j] for v in s["counts"].values()) for j in range(3))
    mine_ = float(got)
    ok = math.isclose(mine_, s["micro"], rel_tol=1e-6, abs_tol=1e-9)
    share = s["right"] / len(gold())
    first = _chip(_dec(mine_, lang, 4)) if ok else _chip(_dec(mine_, lang, 4), _CHIP_PURPLE) + _chip(_dec(s["micro"], lang, 4), _CHIP_MISSING)
    values = [mine_, share, s["macro"]]
    shown = [first, _chip(_dec(share, lang, 4)), _chip(_dec(s["macro"], lang, 4))]
    rows_ = [[html.escape(name), v, _bar(x, 1.0, "var(--ea-data)", 10.0)]
             for name, v, x in zip(w["rows"], shown, values)]
    pooled = w["pooled"].format(tp=_num(tp, lang), fp=_num(fp, lang), fn=_num(fn_, lang))
    body = _label(pooled) + _table(rows_, w["head"], numbers=(1,)) + _note(w["note"])
    alt_rows = "; ".join(f"{name}: {_dec(x, lang, 4)}" for name, x in zip(w["rows"], values))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], pooled=pooled, rows=alt_rows,
                                                              note=w["note"]))


# --- the demo: a classifier that knows fewer intents -------------------------------------------------------------------

_DEMO_TEXT = {
    "en": {"title": "Naive Bayes that knows only the {k} most common intents",
           "alt": "{title}. Bars from left to right: {rows}. Macro F1 {macro}, micro F1 {micro}.",
           "said": "Knowing the {k} most common intents, whose commands are {cov} of the 2,974 test commands: micro F1 "
                   "{micro}, macro F1 {macro}.{who}",
           "who": " Your `micro_f1` and `macro_f1` worked out the two averages.",
           "title_one": "Naive Bayes that knows only the most common intent",
           "said_one": "Knowing only the most common intent, whose commands are {cov} of the 2,974 test commands: micro "
                       "F1 {micro}, macro F1 {macro}.{who}",
           "title_all": "Naive Bayes that knows all 60 intents",
           "said_all": "Knowing all 60 intents, as before: micro F1 {micro}, macro F1 {macro}.{who}"},
    "nl": {"title": "Naive Bayes die alleen de {k} meest voorkomende intents kent",
           "alt": "{title}. Balken van links naar rechts: {rows}. Macro F1 {macro}, micro F1 {micro}.",
           "said": "Met de {k} meest voorkomende intents, met samen {cov} van de 2.974 testopdrachten: micro F1 "
                   "{micro}, macro F1 {macro}.{who}",
           "who": " Je `micro_f1` en `macro_f1` rekenden de twee gemiddelden uit.",
           "title_one": "Naive Bayes die alleen de meest voorkomende intent kent",
           "said_one": "Met alleen de meest voorkomende intent, met {cov} van de 2.974 testopdrachten: micro F1 "
                       "{micro}, macro F1 {macro}.{who}",
           "title_all": "Naive Bayes die alle 60 intents kent",
           "said_all": "Met alle 60 intents, zoals eerst: micro F1 {micro}, macro F1 {macro}.{who}"},
}


def mine(box_text, correct: bool, name: str, names: dict | None = None):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text, dict(names or {}))
    fn = run.names.get(name) if run.error is None else None
    return fn if callable(fn) else None


def demo_values(macro_fn=None, micro_fn=None) -> dict:
    """{k: (macro F1, micro F1)} for every stop of KNOWN, from the learner's functions when given and they work on
    every stop in time, else from the course's; and "yours": whether the learner's did it."""
    def course_values():
        return {k: (known_scores(k)["macro"], known_scores(k)["micro"]) for k in KNOWN}
    if macro_fn is None or micro_fn is None:
        return {"values": course_values(), "yours": False}

    def theirs():
        return {k: (macro_fn(gold(), said_known(k), labels()), micro_fn(gold(), said_known(k), labels())) for k in KNOWN}
    try:
        got = _call(theirs, 30.0)
    except Timeout:
        got = _STOPPED
    if got is _STOPPED or not all(_number(a) and _number(b) for a, b in got.values()):
        return {"values": course_values(), "yours": False}
    return {"values": {k: (float(a), float(b)) for k, (a, b) in got.items()}, "yours": True}


def demo_frame(k: int, values: dict, lang: str) -> tuple:
    """The picture and the sentence of the demo at k known intents."""
    w = _DEMO_TEXT[lang]
    k = int(k)
    macro, micro = values["values"][k]
    s = said_known(k)
    c = all_counts(gold(), s, labels())
    f1s = {label: f1(*c[label]) for label in labels()}
    known = min(k, len(labels_by_count()))
    end = "_one" if k == 1 else "_all" if k >= len(by_count()) else ""
    title = w["title" + end].format(k=k)
    p = _f1_panel(f1s, lang, macro=macro, micro=micro, known=known)
    svg = chart.figure([p], alt=w["alt"].format(title=title, rows=_alt_f1s(f1s, lang), macro=_dec(macro, lang),
                                                micro=_dec(micro, lang)), width=460, height=290)
    head = f'<div style="{_LABEL};font-weight:600;color:var(--ea-ink)">{html.escape(title)}</div>'
    cov = known_scores(k)["covered"]
    said_text = w["said" + end].format(k=k, cov=_num(cov, lang), micro=_dec(micro, lang), macro=_dec(macro, lang),
                                 who=w["who"] if values["yours"] else "")
    return head + svg, said_text


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["counts", "macro", "micro", "prior", "which-average"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, numbers rounded to six decimals."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value(p) for p in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, (float, np.floating)):
        return repr(round(float(v), 6) + 0.0)
    if isinstance(v, np.integer):
        return repr(int(v))
    return repr(v)


def _shown(v):
    """A value for a message: a long list as its length and its first items, in each language."""
    if isinstance(v, (list, tuple)) and len(v) > 12:
        first = ", ".join(_value(x) for x in v[:3])
        return {"en": f"a list of {len(v):,} values, starting with {first}",
                "nl": f"een lijst van {_num(len(v), 'nl')} waarden, die begint met {first}"}
    return _value(v)


class Named:
    """A value of the check's own that a message shows under a name, such as the test commands of the lesson."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return list(self.value) if isinstance(self.value, list) else copy.deepcopy(self.value, memo)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _index_error(e) -> bool:
    """A label used as a place in a list: `for i in gold:` gives the labels themselves, not their places."""
    return isinstance(e, TypeError) and "list indices must be integers" in str(e)


def _zero(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _sums_in_loop(got, text) -> bool:
    """The three sums set to 0 inside the loop over the labels, so that only the last label counts."""
    code = "\n".join(line.split("#")[0] for line in str(text).splitlines())
    return re.search(r"^[ \t]{8,}all_(tp|fp|fn)\s*=\s*(all_\w+\s*=\s*)*0\b", code, flags=re.MULTILINE) is not None


_SMALL = (Named("small_gold", SMALL_GOLD), Named("small_said", SMALL_SAID))
_MORE = (Named("more_gold", MORE_GOLD), Named("more_said", MORE_SAID))
_TEST = (Named("gold", gold()), Named("said", said()))
_NAMES = {}
_NAMES_C = {"counts": counts}

# exercise 1: counts. The two labels of the small example, then: the three labels of the other small example, a label
# that is in neither list, and three intents of the test commands
COUNT_CASES = [
    _SMALL + ("play_music",),
    _SMALL + ("alarm_set",),
    _MORE + ("weather_query",),
    _MORE + ("alarm_set",),
    _MORE + ("play_music",),
    _SMALL + ("weather_query",),
    _TEST + ("calendar_set",),
    _TEST + ("general_quirky",),
    _TEST + ("music_dislikeness",),
]

_C_START = "def counts(gold, said, label):\n    ...\n"
_C = ("def counts(gold, said, label):\n"
      "    tp = 0\n"
      "    fp = 0\n"
      "    fn = 0\n"
      "    for i in range(len(gold)):\n"
      "        if gold[i] == label and said[i] == label:\n"
      "            tp = tp + 1\n"
      "        if gold[i] != label and said[i] == label:\n"
      "            fp = fp + 1\n"
      "        if gold[i] == label and said[i] != label:\n"
      "            fn = fn + 1\n"
      "    return tp, fp, fn\n")
_C_SWAP = _C.replace("    return tp, fp, fn\n", "    return tp, fn, fp\n")
_C_TP_ANY = _C.replace("if gold[i] == label and said[i] == label:", "if gold[i] == said[i]:")
_C_FP_ALL = _C.replace("if gold[i] != label and said[i] == label:", "if said[i] == label:")
_C_FN_ALL = _C.replace("if gold[i] == label and said[i] != label:", "if gold[i] == label:")
_C_FP_SAME = _C.replace("if gold[i] != label and said[i] == label:", "if gold[i] == label and said[i] == label:")
_C_TN = _C.replace("if gold[i] != label and said[i] == label:", "if gold[i] != label and said[i] != label:")
_C_RETURN_IN = _C.replace("    return tp, fp, fn\n", "        return tp, fp, fn\n")
_C_LIST = _C.replace("    return tp, fp, fn\n", "    return [tp, fp, fn]\n")

# exercise 2: macro_f1. The two small examples, then: every label right, the second example with a shorter list of
# labels, the test commands, the test commands without the prior, and with a classifier that knows only the 10 most
# common intents
MACRO_CASES = [
    _SMALL + (Named("small_labels", SMALL_LABELS),),
    _MORE + (Named("more_labels", MORE_LABELS),),
    (_lit(["alarm_set", "play_music"]), _lit(["alarm_set", "play_music"]), _lit(["alarm_set", "play_music"])),
    _MORE + (_lit(["alarm_set", "weather_query"]),),
    _TEST + (Named("labels", labels()),),
    (Named("gold", gold()), Named("C.said_flat()", said_flat()), Named("labels", labels())),
    (Named("gold", gold()), Named("C.said_known(10)", said_known(10)), Named("labels", labels())),
]
MICRO_CASES = MACRO_CASES

_M_START = "def macro_f1(gold, said, labels):\n    ...\n"
_M = ("def macro_f1(gold, said, labels):\n"
      "    total = 0\n"
      "    for label in labels:\n"
      "        tp, fp, fn = counts(gold, said, label)\n"
      "        total = total + 2 * tp / (2 * tp + fp + fn)\n"
      "    return total / len(labels)\n")
_M_GOLD_N = _M.replace("total / len(labels)", "total / len(gold)")
_M_NO_MEAN = _M.replace("    return total / len(labels)\n", "    return total\n")
_M_HALF = _M.replace("2 * tp / (2 * tp + fp + fn)", "2 * tp / (tp + fp + fn)")
_M_RESET = ("def macro_f1(gold, said, labels):\n"
            "    for label in labels:\n"
            "        total = 0\n"
            "        tp, fp, fn = counts(gold, said, label)\n"
            "        total = total + 2 * tp / (2 * tp + fp + fn)\n"
            "    return total / len(labels)\n")
_M_RETURN_IN = _M.replace("    return total / len(labels)\n", "        return total / len(labels)\n")
_M_WEIGHTED = _M.replace("for label in labels:", "for label in gold:").replace("total / len(labels)", "total / len(gold)")
_M_PRECISION = _M.replace("2 * tp / (2 * tp + fp + fn)", "tp / (tp + fp)")
_M_PR_MEANS = ("def macro_f1(gold, said, labels):\n"
               "    p_total = 0\n"
               "    r_total = 0\n"
               "    for label in labels:\n"
               "        tp, fp, fn = counts(gold, said, label)\n"
               "        if tp + fp > 0:\n"
               "            p_total = p_total + tp / (tp + fp)\n"
               "        r_total = r_total + tp / (tp + fn)\n"
               "    p = p_total / len(labels)\n"
               "    r = r_total / len(labels)\n"
               "    return 2 * p * r / (p + r)\n")

_MI_START = "def micro_f1(gold, said, labels):\n    ...\n"
_MI = ("def micro_f1(gold, said, labels):\n"
       "    all_tp = 0\n"
       "    all_fp = 0\n"
       "    all_fn = 0\n"
       "    for label in labels:\n"
       "        tp, fp, fn = counts(gold, said, label)\n"
       "        all_tp = all_tp + tp\n"
       "        all_fp = all_fp + fp\n"
       "        all_fn = all_fn + fn\n"
       "    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n")
_MI_MEAN = _M.replace("def macro_f1(", "def micro_f1(")
_MI_HALF = _MI.replace("2 * all_tp / (2 * all_tp + all_fp + all_fn)", "2 * all_tp / (all_tp + all_fp + all_fn)")
_MI_RESET = ("def micro_f1(gold, said, labels):\n"
             "    for label in labels:\n"
             "        all_tp = 0\n"
             "        all_fp = 0\n"
             "        all_fn = 0\n"
             "        tp, fp, fn = counts(gold, said, label)\n"
             "        all_tp = all_tp + tp\n"
             "        all_fp = all_fp + fp\n"
             "        all_fn = all_fn + fn\n"
             "    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n")
_MI_RETURN_IN = _MI.replace("    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n",
                            "        return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n")
_MI_PER_LABEL = _MI.replace("    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n",
                            "    return 2 * all_tp / (2 * all_tp + all_fp + all_fn) / len(labels)\n")
_MI_ACCURACY = ("def micro_f1(gold, said, labels):\n"
                "    right = 0\n"
                "    for i in range(len(gold)):\n"
                "        if gold[i] == said[i]:\n"
                "            right = right + 1\n"
                "    return right / len(gold)\n")
_MI_LAST = _MI.replace("    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n",
                       "    return 2 * tp / (2 * tp + fp + fn)\n")

EXERCISES = {
    "counts": FunctionBox(
        name="counts",
        picture=counts_picture,
        function="counts",
        start=_C_START,
        solution=_C,
        names=_NAMES,
        visible=COUNT_CASES[:2],
        hidden=COUNT_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function stops with a `TypeError`: it uses a label as a place in a list. `for i in gold:` gives the labels themselves; `for i in range(len(gold)):` gives the places 0, 1, 2 and so on, so that `gold[i]` and `said[i]` belong to the same command.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: je functie gebruikt een label als plek in een lijst. `for i in gold:` geeft de labels zelf; `for i in range(len(gold)):` geeft de plekken 0, 1, 2 enzovoort, zodat `gold[i]` en `said[i]` bij dezelfde opdracht horen."),
            Mistake("swapped", wrong=_C_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: the false positives and the false negatives are swapped. A false positive is a command the classifier gives `label` while its true intent is another; a false negative is a command of `label` that gets another intent. Give them back in the order `tp, fp, fn`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de false positives en de false negatives zijn omgewisseld. Een false positive is een opdracht die de classifier `label` geeft terwijl de echte intent een andere is; een false negative is een opdracht van `label` die een andere intent krijgt. Geef de aantallen terug in de volgorde `tp, fp, fn`."),
            Mistake("any-right", wrong=_C_TP_ANY,
                    en="`{call}` gives {got}, and it should give {want}: your function counts every command that gets its own intent as a true positive, whatever the intent. A true positive for `label` is a command of `label` that gets `label`: both `gold[i]` and `said[i]` must be `label`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt elke opdracht die de eigen intent krijgt als true positive, welke intent het ook is. Een true positive voor `label` is een opdracht van `label` die `label` krijgt: `gold[i]` en `said[i]` moeten allebei `label` zijn."),
            Mistake("fp-all", wrong=_C_FP_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your false positives also hold the true positives. A command the classifier gives `label` is only a false positive when its own intent is another one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je false positives zitten de true positives ook. Een opdracht die de classifier `label` geeft, is alleen een false positive als de eigen intent een andere is."),
            Mistake("fn-all", wrong=_C_FN_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your false negatives also hold the true positives. A command of `label` is only a false negative when the classifier gives it another intent.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je false negatives zitten de true positives ook. Een opdracht van `label` is alleen een false negative als de classifier er een andere intent aan geeft."),
            Mistake("fp-same", wrong=_C_FP_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your false positives count the same commands as your true positives. A false positive gets `label` from the classifier, while its own intent is another one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je false positives tellen dezelfde opdrachten als je true positives. Een false positive krijgt `label` van de classifier, terwijl de eigen intent een andere is."),
            Mistake("true-negatives", wrong=_C_TN,
                    en="`{call}` gives {got}, and it should give {want}: in the middle place your function counts the commands where neither the true intent nor the one the classifier gives is `label`: the true negatives. A false positive gets `label` from the classifier, while its own intent is another one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: op de middelste plek telt je functie de opdrachten waarbij de echte intent en de intent van de classifier allebei geen `label` zijn: de true negatives. Een false positive krijgt `label` van de classifier, terwijl de eigen intent een andere is."),
            Mistake("return-in-loop", wrong=_C_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first command counts. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de eerste opdracht telt. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
            Mistake("list", wrong=_C_LIST,
                    en="`{call}` gives {got}, and it should give {want}: the right three numbers, in a list. Give them back as a tuple, with `return tp, fp, fn`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede drie getallen, in een lijst. Geef de getallen terug als tuple, met `return tp, fp, fn`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start three counts at 0, go over the places of the commands, add 1 to the right count for each command, and give back `tp, fp, fn`.",
                      "nl": "Vervang de `...`: zet drie aantallen eerst op 0, ga de plekken van de opdrachten langs, tel bij elke opdracht 1 op bij het goede aantal, en geef `tp, fp, fn` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count, for each command, whether its true intent is `label`, and whether the classifier gives it `label`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie bij elke opdracht of de echte intent `label` is, en of de classifier er `label` aan geeft?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for any lists and any label: another small example, a label that is in neither list, the 2,974 test commands?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle lijsten en elk label: een ander klein voorbeeld, een label dat in geen van beide lijsten staat, de 2.974 testopdrachten?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give three numbers. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet drie getallen zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `counts` is right in all {n} cases. Look at the picture: your counts for the four most common intents of the test commands and four of the least common.",
                        "nl": "Je `counts` klopt in alle {n} gevallen. Kijk naar de afbeelding: je aantallen voor de vier intents die het vaakst voorkomen in de testopdrachten, en vier van de intents die het minst voorkomen."},
        },
    ),
    "macro": FunctionBox(
        name="macro",
        picture=macro_picture,
        function="macro_f1",
        start=_M_START,
        solution=_M,
        names=_NAMES_C,
        visible=MACRO_CASES[:2],
        hidden=MACRO_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("precision", error=_zero,
                    en="With `{call}`, your function divides by 0. A label the classifier never gives, such as alarm_set in the small example, has no precision: 0 true positives divided by 0 commands. The F1 score from the counts, `2 * tp / (2 * tp + fp + fn)`, is 0 then, and that is what this exercise needs.",
                    nl="Met `{call}` deelt je functie door 0. Een label dat de classifier nooit geeft, zoals alarm_set in het kleine voorbeeld, heeft geen precision: 0 true positives gedeeld door 0 opdrachten. De F1 score uit de aantallen, `2 * tp / (2 * tp + fp + fn)`, is dan 0, en dat getal heeft deze oefening nodig."),
            Mistake("pr-means", wrong=_M_PR_MEANS,
                    en="`{call}` gives {got}, and it should give {want}: the F1 score of the mean precision and the mean recall. That is the other macro F1 of *Watch out* under *The idea*; this lesson takes the plain mean of the F1 scores of the labels.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de F1 score van de gemiddelde precision en de gemiddelde recall. Dat is de andere macro F1 uit *Let op* bij *Het idee*; deze les neemt het gewone gemiddelde van de F1 scores van de labels."),
            Mistake("n-of-gold", wrong=_M_GOLD_N,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of commands. The mean is over the labels: divide by the number of labels.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal opdrachten. Het gemiddelde gaat over de labels: deel door het aantal labels."),
            Mistake("weighted", wrong=_M_WEIGHTED,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes over the commands, so each label counts as often as it occurs, and a common label weighs more. Go over the labels, `for label in labels:`, so that each counts once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop gaat over de opdrachten, dus elk label telt zo vaak mee als het voorkomt, en een label dat vaak voorkomt weegt zwaarder. Ga de labels langs, `for label in labels:`, zodat elk label één keer telt."),
            Mistake("no-mean", wrong=_M_NO_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: the sum of the F1 scores. The macro F1 is their mean: divide the sum by the number of labels.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som van de F1 scores. De macro F1 is het gemiddelde van de F1 scores: deel de som door het aantal labels."),
            Mistake("half", wrong=_M_HALF,
                    en="`{call}` gives {got}, and it should give {want}: in your F1 score the denominator has the true positives once. The F1 score is `2 * tp / (2 * tp + fp + fn)`: twice in both places.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je F1 score staan de true positives één keer onder de deelstreep. De F1 score is `2 * tp / (2 * tp + fp + fn)`: twee keer, boven en onder."),
            Mistake("pooled", wrong=_MI.replace("def micro_f1(", "def macro_f1("),
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the counts of all labels and works out one F1 score from them. That is the micro F1, of exercise 3. The macro F1 works out an F1 score for each label, and takes their mean.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de aantallen van alle labels op en rekent daar één F1 score uit. Dat is de micro F1, van oefening 3. De macro F1 rekent voor elk label een F1 score uit, en neemt daar het gemiddelde van."),
            Mistake("reset", wrong=_M_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the last label counts. Your `total` starts at 0 again in each round; set it once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het laatste label telt. Je `total` begint in elke ronde weer bij 0; zet `total` één keer, vóór de loop."),
            Mistake("return-in-loop", wrong=_M_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first label counts. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste label telt. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each label, get its counts from `counts`, work out its F1 score, add them up, and give back their plain mean over the labels.",
                      "nl": "Vervang de `...`: haal voor elk label de aantallen op met `counts`, reken de F1 score uit, tel de scores op, en geef het gewone gemiddelde over de labels terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function work out the F1 score of each label from its own counts, and take the plain mean over the labels?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Rekent je functie de F1 score van elk label uit met de eigen aantallen, en neemt je functie het gewone gemiddelde over de labels?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for any lists: every label right, a list of labels that leaves some intents out, the 2,974 test commands with the 59 intents? Does it use the labels of `labels`, and only those?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle lijsten: elk label goed, een lijst met labels waarin intents ontbreken, de 2.974 testopdrachten met de 59 intents? Gebruikt je functie de labels uit `labels`, en alleen die labels?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `macro_f1` is right in all {n} cases. Look at the picture: your macro F1 on the test commands, as a line over the F1 scores of the 59 intents.",
                        "nl": "Je `macro_f1` klopt in alle {n} gevallen. Kijk naar de afbeelding: je macro F1 op de testopdrachten, als lijn over de F1 scores van de 59 intents."},
        },
    ),
    "micro": FunctionBox(
        name="micro",
        picture=micro_picture,
        function="micro_f1",
        start=_MI_START,
        solution=_MI,
        names=_NAMES_C,
        visible=MICRO_CASES[:2],
        hidden=MICRO_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("mean", wrong=_MI_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the mean of the F1 scores of the labels. That is the macro F1, of exercise 2. The micro F1 first adds up the counts of all labels, and works out one F1 score from the sums.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gemiddelde van de F1 scores van de labels. Dat is de macro F1, van oefening 2. De micro F1 telt eerst de aantallen van alle labels op, en rekent met de sommen één F1 score uit."),
            Mistake("accuracy", wrong=_MI_ACCURACY,
                    en="`{call}` gives {got}, and it should give {want}: the share of the commands that get their own intent. That equals the micro F1 only when every intent in the two lists is in `labels`. A mistake with an intent outside `labels` is a false positive without a false negative, or the other way round, or neither; and a right command of such an intent is not a true positive. Add up the counts that `counts` gives for the labels in `labels`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aandeel van de opdrachten dat de eigen intent krijgt. Dat is alleen gelijk aan de micro F1 als elke intent in de twee lijsten ook in `labels` staat. Een fout met een intent buiten `labels` is een false positive zonder false negative, of andersom, of geen van beide; en een goede opdracht van zo'n intent is geen true positive. Tel de aantallen op die `counts` geeft voor de labels in `labels`."),
            Mistake("half", wrong=_MI_HALF,
                    en="`{call}` gives {got}, and it should give {want}: in your F1 score the denominator has the true positives once. The F1 score is 2 TP divided by 2 TP + FP + FN: twice in both places.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je F1 score staan de true positives één keer onder de deelstreep. De F1 score is 2 TP gedeeld door 2 TP + FP + FN: twee keer, boven en onder."),
            Mistake("per-label", wrong=_MI_PER_LABEL,
                    en="`{call}` gives {got}, and it should give {want}: your function divides the F1 score by the number of labels. The micro F1 is one F1 score of the summed counts; there is no mean to take.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt de F1 score door het aantal labels. De micro F1 is één F1 score van de opgetelde aantallen; een gemiddelde hoort er niet bij."),
            Mistake("reset", matches=_sums_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: only the last label counts. Your sums start at 0 again in each round; set them once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het laatste label telt. Je sommen beginnen in elke ronde weer bij 0; zet de sommen één keer, vóór de loop."),
            Mistake("last-label", wrong=_MI_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the F1 score of the last label alone. After the loop, work out the F1 score with the sums `all_tp`, `all_fp` and `all_fn`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de F1 score van alleen het laatste label. Reken na de loop de F1 score uit met de sommen `all_tp`, `all_fp` en `all_fn`."),
            Mistake("return-in-loop", wrong=_MI_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first label counts. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste label telt. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start three sums at 0, add the counts of each label to them, and give back one F1 score of the three sums.",
                      "nl": "Vervang de `...`: zet drie sommen eerst op 0, tel de aantallen van elk label erbij op, en geef één F1 score van de drie sommen terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add up the true positives, false positives and false negatives of all labels first, and work out the F1 score once, from those sums?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie eerst de true positives, false positives en false negatives van alle labels op, en rekent je functie daarna één keer de F1 score uit, met die sommen?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for any lists: every label right, a list of labels that leaves some intents out, the 2,974 test commands with the 59 intents? Does it use the labels of `labels`, and only those?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle lijsten: elk label goed, een lijst met labels waarin intents ontbreken, de 2.974 testopdrachten met de 59 intents? Gebruikt je functie de labels uit `labels`, en alleen die labels?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `micro_f1` is right in all {n} cases. Look at the picture: your micro F1 on the test commands, next to the share of the commands that get their own intent.",
                        "nl": "Je `micro_f1` klopt in alle {n} gevallen. Kijk naar de afbeelding: je micro F1 op de testopdrachten, naast het aandeel van de opdrachten dat de eigen intent krijgt."},
        },
    ),
    "which-average": Sort(
        name="which-average",
        groups={
            "micro": {"en": "Micro F1 fits best", "nl": "Micro F1 past het best"},
            "macro": {"en": "Macro F1 fits best", "nl": "Macro F1 past het best"},
        },
        items={
            "helpdesk": {"en": "A helpdesk sends each incoming message to one of 12 teams, and wants as many of all messages as possible to reach the right team.",
                         "nl": "Een helpdesk stuurt elk binnenkomend bericht naar een van 12 teams, en wil dat zoveel mogelijk van alle berichten bij het goede team komen."},
            "lights": {"en": "An assistant should understand every kind of request, also rare ones such as turning on the lights in one room.",
                       "nl": "Een assistent moet elk soort verzoek begrijpen, ook zeldzame, zoals het licht aandoen in één kamer."},
            "report": {"en": "A yearly report gives one number for an assistant: the share of all commands it understood.",
                       "nl": "Een jaarverslag geeft één getal voor een assistent: het aandeel van alle opdrachten dat de assistent begreep."},
            "notes": {"en": "A system sorts notes of doctors into 30 diagnoses, and missing a rare diagnosis is as bad as missing a common one.",
                      "nl": "Een systeem deelt aantekeningen van artsen in bij 30 diagnoses, en een zeldzame diagnose missen is even erg als een veelvoorkomende missen."},
            "folders": {"en": "A mail filter sorts 10,000 mails a day into 5 folders, and every mail in the wrong folder costs the same.",
                        "nl": "Een mailfilter deelt 10.000 mails per dag in bij 5 mappen, en elke mail in de verkeerde map kost evenveel."},
            "languages": {"en": "A tool that tells which of 40 languages a text is in should do well on small languages such as Frisian too.",
                          "nl": "Een programma dat zegt in welke van 40 talen een tekst staat, moet het ook goed doen bij kleine talen zoals het Fries."},
        },
        solution={"helpdesk": "micro", "lights": "macro", "report": "micro", "notes": "macro", "folders": "micro",
                  "languages": "macro"},
        nudge={
            "helpdesk": {"en": "Does every message count the same here, or every team?",
                         "nl": "Telt hier elk bericht even zwaar, of elk team?"},
            "lights": {"en": "A rare request has few commands. Which average lets its score count as much as that of a common request?",
                       "nl": "Een zeldzaam verzoek heeft weinig opdrachten. Bij welk gemiddelde telt de score ervan even zwaar als de score van een veelvoorkomend verzoek?"},
            "report": {"en": "With one label per command, which of the two averages is the share of commands that come out right?",
                       "nl": "Met één label per opdracht: welk van de twee gemiddelden is het aandeel van de opdrachten dat goed gaat?"},
            "notes": {"en": "Should a diagnosis with few notes weigh as much as one with many?",
                      "nl": "Moet een diagnose met weinig aantekeningen even zwaar wegen als een diagnose met veel?"},
            "folders": {"en": "When each mistake costs the same, what do you want to count: mails or folders?",
                        "nl": "Als elke fout evenveel kost, wat wil je dan tellen: mails of mappen?"},
            "languages": {"en": "Frisian has far fewer texts than English. Which average would hardly notice when Frisian goes wrong?",
                          "nl": "Van het Fries zijn er veel minder teksten dan van het Engels. Welk gemiddelde merkt het nauwelijks als het Fries misgaat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the situations again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de situaties opnieuw uit."},
            "empty": {"en": "Put each situation under the average that fits it best.",
                      "nl": "Zet elke situatie bij het gemiddelde dat er het best bij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this situation goes somewhere else.",
                      "nl": "{item}: deze situatie hoort ergens anders."},
            "correct": {"en": "All {n} are right. Where every message, command or mail counts the same, as for the helpdesk, the report and the mail filter, the micro F1 says what you want to know: with one label each, it is the share that comes out right. Where every label should work, also a rare one, as for the lights, the diagnoses and Frisian, the macro F1 shows when a small label goes wrong, which the micro F1 hides.",
                        "nl": "Alle {n} kloppen. Waar elk bericht, elke opdracht of elke mail even zwaar telt, zoals bij de helpdesk, het jaarverslag en het mailfilter, zegt de micro F1 wat je wilt weten: met één label per stuk is dat het aandeel dat goed gaat. Waar elk label moet werken, ook een zeldzaam label, zoals bij het licht, de diagnoses en het Fries, laat de macro F1 zien wanneer een klein label misgaat, en de micro F1 verbergt dat."},
        },
    ),
}

CASES = [
    ("counts", _C_START, "empty", None),
    ("counts", _C, "correct", None),
    ("counts", "def counts(gold, said, label):\n    tp = fp = fn = 0\n    for g, s in zip(gold, said):\n        if g == label and s == label:\n            tp += 1\n        elif s == label:\n            fp += 1\n        elif g == label:\n            fn += 1\n    return tp, fp, fn\n", "correct", None),
    ("counts", "def counts(gold, said, label):\n    tp = 0\n    fp = 0\n    fn = 0\n    for i in range(len(gold)):\n        if said[i] == label:\n            if gold[i] == label:\n                tp = tp + 1\n            else:\n                fp = fp + 1\n        elif gold[i] == label:\n            fn = fn + 1\n    return (tp, fp, fn)\n", "correct", None),
    ("counts", _C_SWAP, "wrong", "swapped"),
    ("counts", _C_TP_ANY, "wrong", "any-right"),
    ("counts", _C_FP_ALL, "wrong", "fp-all"),
    ("counts", _C_FN_ALL, "wrong", "fn-all"),
    ("counts", _C_TN, "wrong", "true-negatives"),
    ("counts", _C_RETURN_IN, "wrong", "return-in-loop"),
    ("counts", _C_LIST, "wrong", "list"),
    ("counts", _C_FP_SAME, "wrong", "fp-same"),
    ("counts", _C.replace("for i in range(len(gold)):", "for i in gold:"), "error", "index"),
    ("counts", _C.replace("    return tp, fp, fn\n", "    print(tp, fp, fn)\n"), "wrong", "prints"),
    ("counts", _C.replace("    return tp, fp, fn\n", ""), "wrong", "no-return"),
    ("counts", "def counts(gold, said, label):\n    if label == \"play_music\":\n        return 5, 1, 0\n    return 0, 0, 1\n", "almost", None),
    ("macro", _M_START, "empty", None),
    ("macro", _M, "correct", None),
    ("macro", "def macro_f1(gold, said, labels):\n    scores = []\n    for label in labels:\n        tp, fp, fn = counts(gold, said, label)\n        scores.append(2 * tp / (2 * tp + fp + fn))\n    return sum(scores) / len(scores)\n", "correct", None),
    ("macro", "def macro_f1(gold, said, labels):\n    total = 0\n    for label in labels:\n        tp, fp, fn = counts(gold, said, label)\n        if tp == 0:\n            continue\n        p = tp / (tp + fp)\n        r = tp / (tp + fn)\n        total += 2 * p * r / (p + r)\n    return total / len(labels)\n", "correct", None),
    ("macro", _M_PRECISION, "error", "precision"),
    ("macro", _M_GOLD_N, "wrong", "n-of-gold"),
    ("macro", _M_WEIGHTED, "wrong", "weighted"),
    ("macro", _M_NO_MEAN, "wrong", "no-mean"),
    ("macro", _M_HALF, "wrong", "half"),
    ("macro", _MI.replace("def micro_f1(", "def macro_f1("), "wrong", "pooled"),
    ("macro", _M_RESET, "wrong", "reset"),
    ("macro", _M_RETURN_IN, "wrong", "return-in-loop"),
    ("macro", _M_PR_MEANS, "wrong", "pr-means"),
    ("macro", _M.replace("total / len(labels)", "total / len(set(gold))"), "almost", None),
    ("macro", _M.replace("for label in labels:", "for label in set(gold):"), "almost", None),
    ("macro", _M.replace("    return total / len(labels)\n", ""), "wrong", "no-return"),
    ("macro", "def macro_f1(gold, said, labels):\n    if len(labels) == 2:\n        return 0.45454545454545453\n    return 0.6111111111111112\n", "almost", None),
    ("micro", _MI_START, "empty", None),
    ("micro", _MI, "correct", None),
    ("micro", _MI_ACCURACY, "wrong", "accuracy"),
    ("micro", "def micro_f1(gold, said, labels):\n    tp = fp = fn = 0\n    for label in labels:\n        a, b, c = counts(gold, said, label)\n        tp += a\n        fp += b\n        fn += c\n    return 2 * tp / (2 * tp + fp + fn)\n", "correct", None),
    ("micro", _MI_MEAN, "wrong", "mean"),
    ("micro", _MI_HALF, "wrong", "half"),
    ("micro", _MI_PER_LABEL, "wrong", "per-label"),
    ("micro", _MI_LAST, "wrong", "last-label"),
    ("micro", _MI_RESET, "wrong", "reset"),
    ("micro", _MI.replace("    for label in labels:\n", "    for label in labels:\n        all_tp = all_fp = all_fn = 0\n"), "wrong", "reset"),
    ("micro", _MI.replace("for label in labels:", "for label in set(gold):"), "almost", None),
    ("micro", _MI_RETURN_IN, "wrong", "return-in-loop"),
    ("micro", _MI.replace("    return 2 * all_tp / (2 * all_tp + all_fp + all_fn)\n", ""), "wrong", "no-return"),
    ("micro", "def micro_f1(gold, said, labels):\n    if len(labels) == 2:\n        return 0.8333333333333334\n    return 0.625\n", "almost", None),
    ("which-average", {}, "empty", None),
    ("which-average", {"helpdesk": "micro", "lights": "macro", "report": "micro", "notes": "macro", "folders": "micro",
                       "languages": "macro"}, "correct", None),
    ("which-average", {"helpdesk": "micro", "lights": "macro"}, "wrong", "unfinished"),
    ("which-average", {"report": "macro"}, "wrong", "wrong:report"),
    ("which-average", {"helpdesk": "micro", "lights": "macro", "report": "micro", "notes": "macro", "folders": "micro",
                       "languages": "micro"}, "wrong", "wrong:languages"),
]
