# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-sentiment-lexicon: a sentiment lexicon built from labeled sentences (lexicon), the score of
a sentence with a lexicon, with the score turned around after a negation (score); and sort words by what the two
lexicons say of them; with the pictures of what each function gives on the review sentences, a picture of the idea and
a box for a sentence of your own.

The sentences are "Sentiment Labelled Sentences" (Kotzias, Denil, de Freitas and Smyth, KDD 2015; UCI Machine Learning
Repository, CC BY 4.0, https://doi.org/10.24432/C57604), in data/uci-sentiment.tsv.gz, and the course's Dutch
translation of them in data/uci-sentiment-nl.tsv.gz, split as in the lessons before: every fifth sentence is held out.
The lists of Pattern are the sentiment lexicons of Pattern for English and Dutch (Tom De Smedt and Walter Daelemans,
CLiPS; PDDL), in data/pattern-sentiment.tsv.gz. Sentences and entries marked as not shown are counted but never shown.
Every number in the text and the pictures is computed by the course (tests/test_nlp_sentiment_lexicon.py computes them
again)."""
import ast
import contextlib
import html
import io
import math
import re

import numpy as np

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

_KEY = "nlp-sentiment-lexicon"

# --- the sentences and the two lexicons --------------------------------------------------------------------------------

HELD_OUT = 5   # every fifth sentence is held out, as in the lessons before
AFTER = 2      # a negation turns the score of the next two words around
LABELS = {"en": {1: "positive", 0: "negative"}, "nl": {1: "positief", 0: "negatief"}}
NEGATIONS = {   # the words that turn the score of the next two words around
    "en": ["not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "aren't", "can't", "couldn't", "didn't",
           "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mustn't", "shouldn't", "wasn't", "weren't",
           "won't", "wouldn't"],
    "nl": ["niet", "geen", "nooit", "niets", "niemand", "nergens"],
}


def words(text: str) -> list:
    """The words of a text in small letters: runs of letters, with an apostrophe inside a word as in *don't*."""
    return books.words(text.lower())


def _all(lang: str) -> tuple:
    return books.review_sentences() if lang == "en" else books.dutch_review_sentences()


def _split(lang: str, held: bool) -> list:
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all(lang))
            if (k % HELD_OUT == HELD_OUT - 1) == held]


def training(lang: str) -> list:
    """The 2,400 training sentences in one language, as a new list of [label, words]: label 1 positive, 0 negative."""
    return [[label, words(text)] for label, text, _ in _split(lang, False)]


def held_out(lang: str) -> list:
    """The 600 held-out sentences in one language, every fifth sentence of the set, as (label, text, shown)."""
    return _split(lang, True)


def _lexicon(lang: str) -> tuple:
    def make():
        counts_ = {}
        for label, ws in training(lang):
            for w in ws:
                counts_.setdefault(w, [0, 0])[label] += 1
        return {w: math.log10((c[1] + 1) / (c[0] + 1)) for w, c in counts_.items()}, counts_
    return kept(f"{_KEY}:lexicon:{lang}", make)


def lexicon(lang: str) -> dict:
    """The course's lexicon from the 2,400 training sentences: for each word, log10 of (its count in positive
    sentences + 1) / (its count in negative sentences + 1)."""
    return dict(_lexicon(lang)[0])


def counts(lang: str) -> dict:
    """For each word of the training sentences: [its count in negative sentences, its count in positive ones]."""
    return {w: list(c) for w, c in _lexicon(lang)[1].items()}


def _pattern(lang: str) -> tuple:
    def make():
        by, shown = {}, {}
        for lang_, form, pos, _, p, _, _, sh in books.pattern_sentiment():
            if lang_ == lang:
                w = form.lower()
                by.setdefault(w, {}).setdefault(pos, []).append(p)
                shown[w] = shown.get(w, 1) and sh
        # as Pattern does for a word without a tag: the mean of its senses per tag, then the mean over its tags
        return {w: sum(sum(v) / len(v) for v in tags.values()) / len(tags) for w, tags in by.items()}, shown
    return kept(f"{_KEY}:pattern:{lang}", make)


def pattern(lang: str) -> dict:
    """Pattern's list in one language: for each word, its polarity from -1 to +1, the mean over its senses."""
    return dict(_pattern(lang)[0])


def pattern_senses(word: str, lang: str) -> list:
    """The senses of a word in Pattern's list, as (pos, sense, polarity)."""
    return [(pos, sense, p) for lang_, form, pos, sense, p, *_ in books.pattern_sentiment()
            if lang_ == lang and form.lower() == word]


def score(ws: list, lexicon_: dict, negations: list, after: int = AFTER) -> float:
    """The course's score of a list of words: the sum of the scores of the words in the lexicon, with the score of the
    `after` words that follow a negation turned around (after=0: no negation)."""
    total, left = 0.0, 0
    for w in ws:
        if w in lexicon_:
            total += -lexicon_[w] if left > 0 else lexicon_[w]
        left -= 1
        if after and w in negations:
            left = after
    return total


def label_of(s: float):
    """1 for a score above 0, 0 for one below 0, None for 0: the lexicon says nothing."""
    return 1 if s > 0 else 0 if s < 0 else None


def right_count(lang: str, lexicon_: dict, after: int = AFTER) -> int:
    """How many held-out sentences a lexicon gives their own label."""
    return sum(label_of(score(words(t), lexicon_, NEGATIONS[lang], after)) == label for label, t, _ in held_out(lang))


def _shown_words(lang: str) -> set:
    def make():
        out = set()
        for _, text, shown in _split(lang, False):
            if shown:
                out.update(words(text))
        return out
    return kept(f"{_KEY}:shown:{lang}", make)


def showable(word: str, lang: str) -> bool:
    """Whether a picture may show a word of a lexicon: it stands in a training sentence the course shows, and Pattern's
    list does not mark it as not shown."""
    return word in _shown_words(lang) and _pattern(lang)[1].get(word, 1) == 1


IDEA_SENTENCE = 519   # a held-out sentence, the same in both languages: "Not good for the money."
EXAMPLE_WORDS = {"en": ["excellent", "not", "the", "money"], "nl": ["uitstekend", "niet", "het", "geld"]}   # worked example
OFTEN = 10            # the word lists in the pictures take words that occur at least this often in training


def numbers(lang: str) -> dict:
    """Every number the lesson names for one language."""
    def make():
        lx, pl, neg = lexicon(lang), pattern(lang), NEGATIONS[lang]
        held = held_out(lang)
        out = {"vocab": len(lx), "pattern_words": len(pl),
               "learned": right_count(lang, lx, 0), "learned_neg": right_count(lang, lx),
               "pattern": right_count(lang, pl, 0), "pattern_neg": right_count(lang, pl),
               "pattern_none": sum(1 for _, t, _ in held if not any(w in pl for w in words(t))),
               "pattern_zero": sum(1 for _, t, _ in held if label_of(score(words(t), pl, neg, 0)) is None),
               "with_negation": sum(1 for _, t, _ in held if any(w in neg for w in words(t))),
               "pattern_senses": sum(1 for r in books.pattern_sentiment() if r[0] == lang)}
        out["pattern_spoke"] = len(held) - out["pattern_zero"]
        return out
    return dict(kept(f"{_KEY}:numbers:{lang}", make))


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-ui);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_INK = _CHIP + ";border:1px solid var(--ea-ink-3);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_WRONG = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_TURNED = "font-style:italic;text-decoration:underline;text-decoration-style:dotted"   # a score turned around
_FAINT = "color:var(--ea-ink-3)"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"
_BAR = "display:inline-block;height:0.62rem;border-radius:2px;vertical-align:middle"
_HIDDEN = {"en": "(a sentence the course does not show)", "nl": "(een zin die de course niet laat zien)"}
# A sentence of the set that says he or she (in Dutch also ze, hun, zijn and the like) stays out of a picture, as the
# course's own text has none: the picture then shows _HIDDEN in its place.
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


def _shows(text: str, shown: int, lang: str) -> bool:
    return bool(shown) and _PRONOUN[lang].search(text) is None


def _sentence(text: str, shown: int, lang: str) -> str:
    if not _shows(text, shown, lang):
        return f'<span style="{_TEXT};font-style:italic;color:var(--ea-ink-2)">{html.escape(_HIDDEN[lang])}</span>'
    return f'<span style="{_TEXT}">{html.escape(text)}</span>'


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
    """A number with `digits` decimals, with − for minus (and + when `sign`): 0.45 in English, 0,45 in Dutch. A number
    that rounds to 0 gets no sign."""
    x = round(x, digits) + 0.0
    sign = sign and x != 0
    text = (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _bar(x: float, top: float, colour: str, wrong: bool = False, width: float = 5.0) -> str:
    """A bar for a score: to the left of the middle line for a negative score, to the right for a positive one, as long
    as `width` for `top`."""
    w = max(0.1, width * min(abs(x) / top, 1.0))
    edge = "border:1px dashed var(--ea-loss);" if wrong else ""
    inner = (f'<span style="{_BAR};position:absolute;top:0;{"right" if x < 0 else "left"}:{width:.2f}rem;width:{w:.2f}rem;'
             f'background:{colour};{edge}box-sizing:border-box"></span>')
    return (f'<span style="position:relative;display:inline-block;width:{2 * width:.2f}rem;height:0.62rem;'
            f'vertical-align:middle">{inner}<span style="position:absolute;left:{width:.2f}rem;top:-0.15rem;width:1px;'
            f'height:0.92rem;background:var(--ea-ink-3)"></span></span>')


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of a picture


def _call(make, seconds: float = PICTURE_SECONDS):
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
    """(what make() gives, None), or (None, a figure that says why the picture stays empty): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_SECONDS)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


def _is_number(x) -> bool:
    return (isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
            and math.isfinite(float(x)))


# --- the idea's picture: one sentence, word by word, in the two lists ---------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "One sentence in two lexicons",
           "head": ["Word", "Pattern's list", "From the reviews"],
           "sum": "Sum",
           "sum_neg": "Sum, with the score after a negation turned around",
           "own": "Label of the sentence:",
           "none": "–",
           "note": "Gray: the label the sentence has in the set, and a score before it is turned around. In the color of the text: the score of each word in "
                   "each lexicon, and the label its sum gives; – for a word the lexicon does not have. Dotted and in "
                   "italics: a score turned around, in the two words after *{neg}*. Purple: a label other than the "
                   "sentence's own.",
           "alt": "{title}. {sentence} {rows}. {own} {label}. {note}"},
    "nl": {"title": "Eén zin in twee lexicons",
           "head": ["Woord", "Lijst van Pattern", "Uit de recensies"],
           "sum": "Som",
           "sum_neg": "Som, met de score na een ontkenning omgedraaid",
           "own": "Label van de zin:",
           "none": "–",
           "note": "Grijs: het label dat de zin in de set heeft, en een score vóór het omdraaien. In de kleur van de tekst: de score van elk woord in "
                   "elk lexicon, en het label dat de som geeft; – voor een woord dat het lexicon niet heeft. Gestippeld "
                   "en schuin: een omgedraaide score, in de twee woorden na *{neg}*. Paars: een ander label dan het "
                   "label van de zin.",
           "alt": "{title}. {sentence} {rows}. {own} {label}. {note}"},
}


def _turned(ws: list, negations: list, after: int = AFTER) -> list:
    """For each word, whether its score is turned around: it is one of the `after` words after a negation."""
    out, left = [], 0
    for w in ws:
        out.append(left > 0)
        left -= 1
        if w in negations:
            left = after
    return out


def _value(w: str, lexicon_: dict, turned: bool, lang: str, none: str) -> str:
    if w not in lexicon_:
        return f'<span style="{_FAINT}">{none}</span>'
    x = lexicon_[w]
    if turned:
        return (f'<span style="{_FAINT}">{_dec(x, lang, sign=True)}</span> → '
                f'<span style="{_TURNED}">{_dec(-x, lang, sign=True)}</span>')
    return _dec(x, lang, sign=True)


def _label_chip(s: float, label: int, lang: str) -> str:
    said = label_of(s)
    if said is None:
        return _chip("0", _CHIP_WRONG)
    return _chip(LABELS[lang][said], _CHIP_WRONG if said != label else _CHIP_INK)


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    label, text, _ = held_out(lang)[IDEA_SENTENCE]
    ws, neg = words(text), NEGATIONS[lang]
    lists = (pattern(lang), lexicon(lang))
    turned = _turned(ws, neg)
    rows_, alts = [], []
    for word, t in zip(ws, turned):
        rows_.append([f"<i>{html.escape(word)}</i>"] + [_value(word, lx, t, lang, w["none"]) for lx in lists])
        alts.append(f"{word}: " + ", ".join(_dec(lx[word], lang, sign=True) if word in lx else w["none"] for lx in lists))
    sums = [score(ws, lx, neg, 0) for lx in lists]
    sums_neg = [score(ws, lx, neg) for lx in lists]
    rows_.append([f"<b>{html.escape(w['sum'])}</b>"]
                 + [f"{_dec(s, lang, sign=True)} {_label_chip(s, label, lang)}" for s in sums])
    rows_.append([f"<b>{html.escape(w['sum_neg'])}</b>"]
                 + [f"{_dec(s, lang, sign=True)} {_label_chip(s, label, lang)}" for s in sums_neg])
    note = w["note"].format(neg=ws[0])
    own_html = f'<div style="{_LABEL}">{html.escape(w["own"])} {_chip(LABELS[lang][label])}</div>'
    body = f'<div style="{_LABEL}">{_marked("*" + text + "*")}</div>' + _table(rows_, w["head"], (1, 2)) + own_html + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], sentence=text, rows="; ".join(alts), own=w["own"], label=LABELS[lang][label],
        note=note.replace("*", "")))


# --- the picture under Exercise 1: your lexicon from the 2,400 training sentences ---------------------------------------

_LEX_TEXT = {
    "en": {"title": "Your lexicon from the 2,400 training sentences",
           "head_pos": ["Most positive", "Score", ""],
           "head_neg": ["Most negative", "Score", ""],
           "count": "Your lexicon has {n} words. Adding up its scores gives {right} of the 600 held-out sentences their "
                    "own label; the course's lexicon gives {course}, the classifier of *Logistic regression on words* "
                    "{classifier}.",
           "differ": "{d} words have another score than in the course's lexicon, such as *{word}*: {mine} instead of "
                     "{want}.",
           "differ_one": "One word has another score than in the course's lexicon: *{word}*, with {mine} instead of "
                         "{want}.",
           "differ_count": "{d} words have another score than in the course's lexicon.",
           "differ_count_one": "One word has another score than in the course's lexicon.",
           "same": "Every word has the score it has in the course's lexicon.",
           "note": "The words that occur at least {often} times in the training sentences, with the highest and the "
                   "lowest scores of your lexicon. Blue: your score, as a bar from the middle line, to the right for "
                   "positive and to the left for negative. Purple, dashed: a score other than the course's.",
           "stopped": "Your function stopped with an error on the training sentences, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the 2,400 training sentences, so the picture stays "
                   "empty.",
           "odd": "Your function does not give a dict with a number for each word, so the picture stays empty.",
           "alt": "{title}. {pos}. {neg}. {count} {differ} {note}"},
    "nl": {"title": "Jouw lexicon uit de 2.400 trainingszinnen",
           "head_pos": ["Meest positief", "Score", ""],
           "head_neg": ["Meest negatief", "Score", ""],
           "count": "Jouw lexicon heeft {n} woorden. De scores optellen geeft {right} van de 600 achtergehouden zinnen "
                    "het eigen label; het lexicon van de course geeft er {course}, de classifier van *Logistische "
                    "regressie op woorden* {classifier}.",
           "differ": "{d} woorden hebben een andere score dan in het lexicon van de course, zoals *{word}*: {mine} in "
                     "plaats van {want}.",
           "differ_one": "Eén woord heeft een andere score dan in het lexicon van de course: *{word}*, met {mine} in "
                         "plaats van {want}.",
           "differ_count": "{d} woorden hebben een andere score dan in het lexicon van de course.",
           "differ_count_one": "Eén woord heeft een andere score dan in het lexicon van de course.",
           "same": "Elk woord heeft de score die het in het lexicon van de course heeft.",
           "note": "De woorden die minstens {often} keer in de trainingszinnen staan, met de hoogste en de laagste "
                   "scores van jouw lexicon. Blauw: jouw score, als balk vanaf de middelste streep, naar rechts voor "
                   "positief en naar links voor negatief. Paars, gestreept: een andere score dan in het lexicon van de course.",
           "stopped": "Je functie stopte met een fout op de trainingszinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de 2.400 trainingszinnen, dus de afbeelding "
                   "blijft leeg.",
           "odd": "Je functie geeft geen dict met een getal voor elk woord, dus de afbeelding blijft leeg.",
           "alt": "{title}. {pos}. {neg}. {count} {differ} {note}"},
}
CLASSIFIER = {"en": 491, "nl": 480}   # held-out sentences right for the classifier of nlp-logistic-text (nlp-sentiment)
SHOWN_WORDS = 8                       # words at each end of the lists in the picture


def _is_lexicon(v) -> bool:
    return isinstance(v, dict) and all(isinstance(k, str) and _is_number(x) for k, x in v.items())


def lexicon_picture(seen) -> str:
    lang = seen.lang
    w = _LEX_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    mine, failed = _run_or_say(lambda: fn(training(lang)), w, lambda v: _is_lexicon(v) and len(v) > 0)
    if failed:
        return failed
    mine = {k: float(x) for k, x in mine.items()}
    course, cn = lexicon(lang), counts(lang)
    often = [x for x in cn if sum(cn[x]) >= OFTEN and showable(x, lang) and x in mine]
    ranked = sorted(often, key=lambda x: (mine[x], x))
    top = max([abs(mine[x]) for x in ranked] + [1e-9])

    def rows_of(chosen):
        return [[f"<i>{html.escape(x)}</i>", _dec(mine[x], lang, sign=True),
                 _bar(mine[x], top, "var(--ea-pred)", wrong=abs(mine[x] - course.get(x, math.inf)) > 1e-6)]
                for x in chosen]
    pos = list(reversed(ranked[-SHOWN_WORDS:]))
    neg = ranked[:SHOWN_WORDS]
    right = right_count(lang, mine, 0)
    count = w["count"].format(n=_num(len(mine), lang), right=right, course=numbers(lang)["learned"],
                              classifier=CLASSIFIER[lang])
    differ_words = sorted((x for x in set(mine) | set(course)
                           if abs(mine.get(x, math.inf) - course.get(x, math.inf)) > 1e-6 or (x in mine) != (x in course)),
                          key=lambda x: (-sum(cn.get(x, [0, 0])), x))
    shown_differ = [x for x in differ_words if x in mine and x in course and showable(x, lang)]
    if differ_words and shown_differ:
        x = shown_differ[0]
        one = "_one" if len(differ_words) == 1 else ""
        differ = w["differ" + one].format(d=_num(len(differ_words), lang), word=x, mine=_dec(mine[x], lang, 3, True),
                                          want=_dec(course[x], lang, 3, True))
    elif differ_words:
        one = "_one" if len(differ_words) == 1 else ""
        differ = w["differ_count" + one].format(d=_num(len(differ_words), lang))
    else:
        differ = w["same"]
    note = w["note"].format(often=OFTEN)
    body = (_table(rows_of(pos), w["head_pos"], (1,)) + _table(rows_of(neg), w["head_neg"], (1,))
            + _label(count) + _label(differ) + _note(note))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], pos=", ".join(f"{x} {_dec(mine[x], lang, sign=True)}" for x in pos),
        neg=", ".join(f"{x} {_dec(mine[x], lang, sign=True)}" for x in neg), count=count.replace("*", ""),
        differ=differ.replace("*", ""), note=note))


# --- the picture under Exercise 2: your score on the held-out sentences, with both lexicons -----------------------------

_SCORE_TEXT = {
    "en": {"title": "Your score on the 600 held-out sentences",
           "head": ["Lexicon", "Without negation", "With your score"],
           "pattern": "Pattern's list",
           "learned": "From the reviews",
           "changed": "Sentences with a negation whose label your score changes, with Pattern's list:",
           "none": "With Pattern's list, your score changes the label of no sentence.",
           "head2": ["Sentence", "Label", "Without", "With yours"],
           "nothing": "nothing",
           "note": "The number of held-out sentences that each lexicon gives their own label: without negation, and "
                   "with your score, in blue where it is the count of the course's score, purple and dashed where it is "
                   "not. Below, gray: the label of the sentence; in the color of the text: the label without negation, "
                   "where it is right; blue: the label your score gives, where it is right; purple, dashed: a label "
                   "that is wrong, or *nothing* where the score is 0 and the lexicon says nothing.",
           "stopped": "Your function stopped with an error on a held-out sentence, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the held-out sentences, so the picture stays empty.",
           "odd": "Your function does not give one number for every sentence, so the picture stays empty.",
           "alt": "{title}. {pattern}: {p0} without, {p1} with yours. {learned}: {l0} without, {l1} with yours. "
                  "{changed} {rows}. {note}"},
    "nl": {"title": "Jouw score op de 600 achtergehouden zinnen",
           "head": ["Lexicon", "Zonder ontkenning", "Met jouw score"],
           "pattern": "Lijst van Pattern",
           "learned": "Uit de recensies",
           "changed": "Zinnen met een ontkenning waarvan jouw score het label verandert, met de lijst van Pattern:",
           "none": "Met de lijst van Pattern verandert jouw score het label van geen enkele zin.",
           "head2": ["Zin", "Label", "Zonder", "Met de jouwe"],
           "nothing": "niets",
           "note": "Het aantal achtergehouden zinnen dat elk lexicon het eigen label geeft: zonder ontkenning, en met "
                   "jouw score, blauw waar het het aantal van de score van de course is, paars en gestreept waar niet. "
                   "Daaronder, grijs: het label van de zin; in de kleur van de tekst: het label zonder ontkenning, waar "
                   "het klopt; blauw: het label dat jouw score geeft, waar het klopt; paars, gestreept: een label dat "
                   "niet klopt, of *niets* waar de score 0 is en het lexicon niets zegt.",
           "stopped": "Je functie stopte met een fout op een achtergehouden zin, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de achtergehouden zinnen, dus de afbeelding "
                   "blijft leeg.",
           "odd": "Je functie geeft niet voor elke zin één getal, dus de afbeelding blijft leeg.",
           "alt": "{title}. {pattern}: {p0} zonder, {p1} met de jouwe. {learned}: {l0} zonder, {l1} met de jouwe. "
                  "{changed} {rows}. {note}"},
}
CHANGED_SHOWN = 5   # sentences in the second table


def score_rows(fn, lang: str) -> dict:
    """What fn gives each held-out sentence with Pattern's list and with the course's lexicon."""
    neg = NEGATIONS[lang]
    pl, lx = pattern(lang), lexicon(lang)
    sents = [words(t) for _, t, _ in held_out(lang)]
    return {"pattern": [fn(list(ws), pl, list(neg)) for ws in sents],
            "learned": [fn(list(ws), lx, list(neg)) for ws in sents]}


def _said(s: float, label: int, lang: str, nothing: str, mine: bool) -> str:
    got = label_of(s)
    if got is None:
        return _chip(nothing, _CHIP_WRONG)
    if got != label:
        return _chip(LABELS[lang][got], _CHIP_WRONG)
    return _chip(LABELS[lang][got], _CHIP_PRED if mine else _CHIP_INK)


def score_picture(seen) -> str:
    lang = seen.lang
    w = _SCORE_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    held = held_out(lang)
    got, failed = _run_or_say(lambda: score_rows(fn, lang), w,
                              lambda g: all(len(g[k]) == len(held) and all(_is_number(x) for x in g[k]) for k in g))
    if failed:
        return failed
    n = numbers(lang)
    right = {k: sum(label_of(float(s)) == label for s, (label, _, _) in zip(got[k], held)) for k in got}
    p_chip = _chip(str(right["pattern"]), _CHIP_PRED if right["pattern"] == n["pattern_neg"] else _CHIP_WRONG)
    l_chip = _chip(str(right["learned"]), _CHIP_PRED if right["learned"] == n["learned_neg"] else _CHIP_WRONG)
    rows_ = [[html.escape(w["pattern"]), str(n["pattern"]), p_chip], [html.escape(w["learned"]), str(n["learned"]), l_chip]]
    pl, neg = pattern(lang), NEGATIONS[lang]
    changed, alts = [], []
    for (label, text, shown), s in zip(held, got["pattern"]):
        s0 = score(words(text), pl, neg, 0)
        if any(x in neg for x in words(text)) and label_of(s0) != label_of(float(s)) and len(changed) < CHANGED_SHOWN:
            changed.append([_sentence(text, shown, lang), _chip(LABELS[lang][label]),
                            _said(s0, label, lang, w["nothing"], False), _said(float(s), label, lang, w["nothing"], True)])
            alts.append(f"{text if _shows(text, shown, lang) else _HIDDEN[lang]} ({LABELS[lang][label]}): "
                        f"{_dec(s0, lang, sign=True)} → {_dec(float(s), lang, sign=True)}")
    head_line = w["changed"] if changed else w["none"]
    body = (_table(rows_, w["head"], (1, 2)) + _label(head_line)
            + (_table(changed, w["head2"]) if changed else "") + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], pattern=w["pattern"], p0=n["pattern"], p1=right["pattern"], learned=w["learned"],
        l0=n["learned"], l1=right["learned"], changed=head_line, rows="; ".join(alts), note=w["note"].replace("*", "")))


# --- a sentence of your own -----------------------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Your sentence in the two lexicons",
           "head": ["Word", "Pattern's list", "From the reviews"],
           "sum": "Sum",
           "sum_neg": "Sum, with the score after a negation turned around",
           "none": "–",
           "nothing": "nothing",
           "empty": "Type a sentence in the box, and the picture shows the score of each word in both lexicons.",
           "more": "and {n} more words",
           "note": "In the color of the text: the score of each word in each lexicon, and the label its sum gives; – "
                   "for a word the lexicon does not have, and *nothing* for a sum of 0. Dotted and in italics: a score "
                   "turned around, in the two words after a negation.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw zin in de twee lexicons",
           "head": ["Woord", "Lijst van Pattern", "Uit de recensies"],
           "sum": "Som",
           "sum_neg": "Som, met de score na een ontkenning omgedraaid",
           "none": "–",
           "nothing": "niets",
           "empty": "Typ een zin in het vak, en de afbeelding laat de score van elk woord in beide lexicons zien.",
           "more": "en nog {n} woorden",
           "note": "In de kleur van de tekst: de score van elk woord in elk lexicon, en het label dat de som geeft; – "
                   "voor een woord dat het lexicon niet heeft, en *niets* voor een som van 0. Gestippeld en schuin: een "
                   "omgedraaide score, in de twee woorden na een ontkenning.",
           "alt": "{title}. {rows}. {note}"},
}
TRY_LONGEST = 400   # characters of a sentence of your own that the picture reads
TRY_ROWS = 30       # words the table shows
TRY_START = {   # the sentence in the box at the start: negative without negation in the list from the reviews
    "en": "Not bad at all, and the screen is great.",
    "nl": "Niet slecht, en het scherm is geweldig.",
}


def _sum_cell(s: float, lang: str, nothing: str) -> str:
    said = label_of(s)
    chip = _chip(nothing, _CHIP_INK) if said is None else _chip(LABELS[lang][said], _CHIP_INK)
    return f"{_dec(s, lang, sign=True)} {chip}"


def try_picture(text: str, lang: str) -> str:
    """The words of a sentence someone typed, each with its score in Pattern's list and in the list from the reviews,
    and the sums without and with negation."""
    w = _TRY_TEXT[lang]
    ws = words((text or "")[:TRY_LONGEST])
    if not ws:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    neg = NEGATIONS[lang]
    lists = (pattern(lang), lexicon(lang))
    turned = _turned(ws, neg)
    rows_, alts = [], []
    for word, t in list(zip(ws, turned))[:TRY_ROWS]:
        rows_.append([f"<i>{html.escape(word)}</i>"] + [_value(word, lx, t, lang, w["none"]) for lx in lists])
        alts.append(f"{word}: " + ", ".join(_dec(lx[word], lang, sign=True) if word in lx else w["none"] for lx in lists))
    if len(ws) > TRY_ROWS:
        more = w["more"].format(n=len(ws) - TRY_ROWS)
        rows_.append([f'<span style="{_FAINT}">{html.escape(more)}</span>', "", ""])
        alts.append(more)
    for key, after in (("sum", 0), ("sum_neg", AFTER)):
        sums = [score(ws, lx, neg, after) for lx in lists]
        rows_.append([f"<b>{html.escape(w[key])}</b>"] + [_sum_cell(s, lang, w["nothing"]) for s in sums])
        alts.append(f"{w[key]}: " + ", ".join(_dec(s, lang, sign=True) for s in sums))
    note = w["note"]
    body = _table(rows_, w["head"], (1, 2)) + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts),
                                                              note=note.replace("*", "")))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["lexicon", "score", "words-lists"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _dots_left(got, text) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_NAMES = {"np": np, "math": math}


def _code(v) -> str:
    """A value for a message, the way code writes it: text in double quotes, a number with at most three decimals."""
    v = v.item() if isinstance(v, np.generic) else v
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            return str(float(v))
        text = f"{round(float(v), 3) + 0.0:.3f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text
    if isinstance(v, list):
        return "[" + ", ".join(_code(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_code(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_code(k)}: {_code(x)}" for k, x in v.items()) + "}"
    return repr(v)


# Exercise 1: lexicon

LEX_CASES = [
    ([[1, ["great", "food"]], [0, ["bad", "food"]], [1, ["great", "service"]]],),
    ([[0, ["bad", "bad", "food"]], [1, ["good", "food"]]],),
    ([],),
    ([[1, []]],),
    ([[0, ["slow"]], [0, ["slow"]], [0, ["slow"]]],),
    ([[1, ["nice", "nice", "nice"]]],),
    ([[0, ["koud", "eten"]], [1, ["warm", "eten"]], [1, ["warm", "brood"]]],),
]
_LEX_HEAD = "def lexicon(sentences):\n"
_LEX_START = _LEX_HEAD + "    ...\n"
_LEX_COUNT = ("    counts = {}\n"
              "    for label, words in sentences:\n"
              "        for word in words:\n"
              "            if word in counts:\n"
              "                counts[word][label] = counts[word][label] + 1\n"
              "            else:\n"
              "                counts[word] = [0, 0]\n"
              "                counts[word][label] = 1\n")
_LEX_SCORE = ("    scores = {}\n"
              "    for word in counts:\n"
              "        negative = counts[word][0]\n"
              "        positive = counts[word][1]\n"
              "        scores[word] = np.log10((positive + 1) / (negative + 1))\n"
              "    return scores\n")
_LEX = _LEX_HEAD + _LEX_COUNT + _LEX_SCORE
_LEX_NO_ONE = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "np.log10(positive / negative)")
_LEX_LOGS = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "np.log10(positive) - np.log10(negative)")
_LEX_TOP = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "np.log10(positive / (negative + 1))")
_LEX_SWAPPED = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "np.log10((negative + 1) / (positive + 1))")
_LEX_LN = _LEX.replace("np.log10(", "np.log(")
_LEX_DIFFERENCE = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "positive - negative")
_LEX_RATIO = _LEX.replace("np.log10((positive + 1) / (negative + 1))", "(positive + 1) / (negative + 1)")
_LEX_COUNTS = _LEX_HEAD + _LEX_COUNT + "    return counts\n"
_LEX_ONCE = _LEX.replace("        for word in words:\n", "        for word in set(words):\n")
_LEX_RESET = (_LEX_HEAD + "    for label, words in sentences:\n        counts = {}\n"
              + _LEX_COUNT.split("\n", 2)[2] + _LEX_SCORE)
_LEX_FIRST = _LEX.replace("                counts[word] = [0, 0]\n                counts[word][label] = 1\n",
                          "                counts[word] = [0, 0]\n")
_LEX_RETURN_IN = _LEX_HEAD + _LEX_COUNT + "".join("    " + line + "\n" for line in _LEX_SCORE.splitlines())


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _log_of_zero(e) -> bool:
    return isinstance(e, ValueError) and "domain" in str(e)


def _not_dict(got, text) -> bool:
    return got is not None and not isinstance(got, dict)


# Exercise 2: score

SCORE_CASES = [
    (["not", "good", "for", "the", "money"], {"not": -0.6, "good": 0.5, "money": -0.7}, ["not", "no", "never"]),
    (["never", "bad", "and", "not", "slow"], {"bad": -0.7, "slow": -0.5, "good": 0.5}, ["not", "never"]),
    ([], {"good": 0.5}, ["not"]),
    (["good"], {"good": 0.5}, ["not"]),
    (["not", "very", "good"], {"very": 0.2, "good": 0.5}, ["not"]),
    (["no", "good", "food", "great", "service"], {"good": 0.5, "food": 0.1, "great": 1.5, "service": 0.1}, ["no"]),
    (["not", "bad"], {"not": -0.6, "bad": -0.7}, ["not"]),
    (["not", "not", "good"], {"good": 0.5}, ["not"]),
    (["niet", "lekker", "en", "geen", "goede", "service"], {"lekker": 0.9, "goede": 0.6, "service": 0.1}, ["niet", "geen"]),
    (["good", "not"], {"good": 0.5}, ["not"]),
]
_SCORE_HEAD = "def score(words, lexicon, negations):\n"
_SCORE_START = _SCORE_HEAD + "    ...\n"
_SCORE = (_SCORE_HEAD
          + "    total = 0\n"
          + "    left = 0\n"
          + "    for word in words:\n"
          + "        if word in lexicon:\n"
          + "            if left > 0:\n"
          + "                total = total - lexicon[word]\n"
          + "            else:\n"
          + "                total = total + lexicon[word]\n"
          + "        left = left - 1\n"
          + "        if word in negations:\n"
          + "            left = 2\n"
          + "    return total\n")
_SCORE_ONE = _SCORE.replace("            left = 2\n", "            left = 1\n")
_SCORE_THREE = _SCORE.replace("            left = 2\n", "            left = 3\n")
_SCORE_REST = _SCORE.replace("        left = left - 1\n", "")
_SCORE_ITSELF = (_SCORE.replace("        left = left - 1\n        if word in negations:\n            left = 2\n",
                                "        left = left - 1\n")
                 .replace("    for word in words:\n", "    for word in words:\n        if word in negations:\n            left = 2\n"))
_SCORE_IGNORED = (_SCORE_HEAD + "    total = 0\n    for word in words:\n        if word in lexicon:\n"
                  "            total = total + lexicon[word]\n    return total\n")
_SCORE_HALF = _SCORE.replace("total = total - lexicon[word]", "total = total - 0.5 * lexicon[word]")
_SCORE_FIXED = _SCORE.replace("        if word in negations:\n", '        if word == "not":\n')
_SCORE_RETURN_IN = _SCORE.replace("            left = 2\n    return total\n", "            left = 2\n        return total\n")
_SCORE_LABEL = _SCORE.replace("    return total\n", "    if total > 0:\n        return 1\n    return 0\n")


def _not_number(got, text) -> bool:
    return got is not None and not _is_number(got)


# Exercise 3: what the two lists say of a word. The same eight words in both languages (tests/test_nlp_sentiment_lexicon.py
# checks each group against the two lexicons, and the numbers in the correct message).

SORT_WORDS = {"great": {"en": "great", "nl": "geweldig"}, "terrible": {"en": "terrible", "nl": "verschrikkelijk"},
              "boring": {"en": "boring", "nl": "saai"}, "minutes": {"en": "minutes", "nl": "minuten"},
              "price": {"en": "price", "nl": "prijs"}, "plot": {"en": "plot", "nl": "plot"},
              "small": {"en": "small", "nl": "klein"}, "thin": {"en": "thin", "nl": "dun"}}
SORT_SOLUTION = {"great": "agree", "terrible": "agree", "boring": "agree", "minutes": "reviews", "price": "reviews",
                 "plot": "reviews", "small": "disagree", "thin": "disagree"}


def sort_group(word: str, lang: str) -> str:
    """The group of a word: both lists give it a score with the same sign, only the list from the reviews has it, or the
    two give it scores with opposite signs."""
    lx, pl = lexicon(lang), pattern(lang)
    if word not in pl:
        return "reviews"
    return "agree" if (lx[word] > 0) == (pl[word] > 0) else "disagree"


EXERCISES = {
    "lexicon": FunctionBox(
        name="lexicon",
        picture=lexicon_picture,
        function="lexicon",
        start=_LEX_START,
        solution=_LEX,
        names=_NAMES,
        visible=LEX_CASES[:2],
        hidden=LEX_CASES[2:],
        first=[_DOTS],
        show=_code,
        mistakes=[
            Mistake("no-plus-one", error=_zero_division,
                    en="With `{call}`, your function stops with a `ZeroDivisionError`: a word that never occurs in a negative sentence has a count of 0 there, and your function divides by it. What does the recipe do to both counts before it divides?",
                    nl="Met `{call}` stopt je functie met een `ZeroDivisionError`: een woord dat nooit in een negatieve zin staat, heeft daar een aantal van 0, en je functie deelt erdoor. Wat doet het recept met beide aantallen vóór het deelt?"),
            Mistake("log-of-zero", error=_log_of_zero,
                    en="With `{call}`, your function stops with a `ValueError`: it takes the log of 0, for a word that never occurs in a positive sentence. What does the recipe do to both counts before it divides?",
                    nl="Met `{call}` stopt je functie met een `ValueError`: je functie neemt de logaritme van 0, voor een woord dat nooit in een positieve zin staat. Wat doet het recept met beide aantallen vóór het deelt?"),
            Mistake("logs", wrong=_LEX_LOGS,
                    en="`{call}` gives {got}, and it should give {want}: a word that occurs on one side only gets `inf` or `-inf`, a score without end. What does the recipe do to both counts first?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat maar aan één kant voorkomt, krijgt `inf` of `-inf`, een score zonder einde. Wat doet het recept eerst met beide aantallen?"),
            Mistake("top-only", wrong=_LEX_TOP,
                    en="`{call}` gives {got}, and it should give {want}: a word that never occurs in a positive sentence gets `-inf`, and every other score is too low. Which counts does the recipe add 1 to?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat nooit in een positieve zin staat, krijgt `-inf`, en elke andere score is te laag. Bij welke aantallen telt het recept 1 op?"),
            Mistake("swapped", wrong=_LEX_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: a word from the positive sentences gets a negative score. Which count goes on top of the fraction?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord uit de positieve zinnen krijgt een negatieve score. Welk aantal hoort boven de deelstreep?"),
            Mistake("natural-log", wrong=_LEX_LN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the natural log. Which log does the recipe take?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt de natuurlijke logaritme. Welke logaritme neemt het recept?"),
            Mistake("difference", wrong=_LEX_DIFFERENCE,
                    en="`{call}` gives {got}, and it should give {want}: your scores are differences of counts, so a word that occurs often gets a large score. How does the recipe make one number of the two counts?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je scores zijn verschillen van aantallen, dus een woord dat vaak voorkomt, krijgt een grote score. Hoe maakt het recept één getal van de twee aantallen?"),
            Mistake("no-log", wrong=_LEX_RATIO,
                    en="`{call}` gives {got}, and it should give {want}: a word in as many positive as negative sentences gets 1, and every score is above 0. What does the recipe do with the fraction last?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat in evenveel positieve als negatieve zinnen staat, krijgt 1, en elke score is groter dan 0. Wat doet het recept als laatste met de breuk?"),
            Mistake("counts", wrong=_LEX_COUNTS,
                    en="`{call}` gives {got}, and it should give {want}: the two counts of each word. What should each word get in the end?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de twee aantallen van elk woord. Wat moet elk woord uiteindelijk krijgen?"),
            Mistake("once", wrong=_LEX_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: your function counts a word once in a sentence, also when it stands there twice. How often should a word count that stands twice in one sentence?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt een woord één keer per zin, ook als het er twee keer staat. Hoe vaak moet een woord tellen dat twee keer in één zin staat?"),
            Mistake("reset", wrong=_LEX_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the words of the last sentence. Where does your function make its dictionary of counts?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de woorden van de laatste zin. Waar maakt je functie de dictionary met aantallen?"),
            Mistake("first-lost", wrong=_LEX_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: the first time your function meets a word, it does not count it. What should the first sentence with a word add to its counts?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste keer dat je functie een woord tegenkomt, telt je functie het niet. Wat moet de eerste zin met een woord bij de aantallen optellen?"),
            Mistake("return-in-loop", wrong=_LEX_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the words of the first sentence. Where does your `return` stand?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de woorden van de eerste zin. Waar staat je `return`?"),
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give {want}: a dictionary with a score for each word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een dictionary met een score voor elk woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count how often each word occurs in negative and in positive sentences, and give each word log10 of the positive count plus 1 over the negative count plus 1.",
                      "nl": "Vervang de `...`: tel hoe vaak elk woord in negatieve en in positieve zinnen voorkomt, en geef elk woord log10 van het positieve aantal plus 1 gedeeld door het negatieve aantal plus 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What does your function count for each word, and what does it make of the two counts?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wat telt je functie voor elk woord, en wat maakt je functie van de twee aantallen?"},
            "almost": {"en": "Right for the examples. But `{call}` gives {got}, and it should give {want}. Does your function work for no sentences at all, a sentence without words, a word in negative sentences only, a word three times in one sentence, and a first sentence that is negative?",
                       "nl": "Goed voor de voorbeelden. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor helemaal geen zinnen, een zin zonder woorden, een woord in alleen negatieve zinnen, een woord drie keer in één zin, en een eerste zin die negatief is?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a dictionary. Where does your function give back its scores?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een dictionary zijn. Waar geeft je functie de scores terug?"},
            "correct": {"en": "Your `lexicon` is right in all {n} cases. Look at the picture: the lexicon your function builds from the 2,400 training sentences, and how many held-out sentences it gets right.",
                        "nl": "Je `lexicon` klopt in alle {n} gevallen. Kijk naar de afbeelding: het lexicon dat je functie uit de 2.400 trainingszinnen bouwt, en hoeveel achtergehouden zinnen het goed heeft."},
        },
    ),
    "score": FunctionBox(
        name="score",
        picture=score_picture,
        function="score",
        start=_SCORE_START,
        solution=_SCORE,
        visible=SCORE_CASES[:2],
        hidden=SCORE_CASES[2:],
        first=[_DOTS],
        show=_code,
        mistakes=[
            Mistake("one-word", wrong=_SCORE_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function turns around the score of one word after a negation. How many words after a negation does the recipe turn around?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie draait de score van één woord na een ontkenning om. Hoeveel woorden na een ontkenning draait het recept om?"),
            Mistake("three-words", wrong=_SCORE_THREE,
                    en="`{call}` gives {got}, and it should give {want}: your function turns around the scores of three words after a negation. How many words after a negation does the recipe turn around?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie draait de scores van drie woorden na een ontkenning om. Hoeveel woorden na een ontkenning draait het recept om?"),
            Mistake("rest", wrong=_SCORE_REST,
                    en="`{call}` gives {got}, and it should give {want}: after a negation, your function turns around the score of every word up to the end. Where should the turning around stop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na een ontkenning draait je functie de score van elk woord tot het eind om. Waar moet het omdraaien ophouden?"),
            Mistake("itself", wrong=_SCORE_ITSELF,
                    en="`{call}` gives {got}, and it should give {want}: your function also turns around the score of the negation itself, and of one word after it. Which words should get their score turned around?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie draait ook de score van de ontkenning zelf om, en van één woord erna. Van welke woorden moet de score om?"),
            Mistake("ignored", wrong=_SCORE_IGNORED,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up every score as it is, also right after a negation. What should happen to the score of a word right after *not*?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt elke score op zoals de score in het lexicon staat, ook vlak na een ontkenning. Wat moet er gebeuren met de score van een woord vlak na *niet*?"),
            Mistake("half", wrong=_SCORE_HALF,
                    en="`{call}` gives {got}, and it should give {want}: after a negation, your function takes −0.5 times the score, as Pattern does. What does the recipe of this lesson do with it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na een ontkenning neemt je functie −0,5 keer de score, zoals Pattern doet. Wat doet het recept van deze les ermee?"),
            Mistake("fixed-word", wrong=_SCORE_FIXED,
                    en="`{call}` gives {got}, and it should give {want}: your function turns scores around after *not* only. Which words should it look for?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie draait alleen na *not* scores om. Naar welke woorden moet je functie kijken?"),
            Mistake("return-in-loop", wrong=_SCORE_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first word only. Where does your `return` stand?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar het eerste woord. Waar staat je `return`?"),
            Mistake("label", wrong=_SCORE_LABEL,
                    en="`{call}` gives {got}, and it should give {want}: a label, 1 or 0. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een label, 1 of 0. Wat moet je functie teruggeven?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, the sum of the scores. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één getal, de som van de scores. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up the scores of the words that the lexicon has, and take the score of the two words after a negation with the opposite sign.",
                      "nl": "Vervang de `...`: tel de scores op van de woorden die het lexicon heeft, en neem de score van de twee woorden na een ontkenning met het omgekeerde teken."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which scores does your function add up, and which does it turn around?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke scores telt je functie op, en welke draait je functie om?"},
            "almost": {"en": "Right for the examples. But `{call}` gives {got}, and it should give {want}. Does your function work for an empty list, a sentence without a negation, a word two places after a negation, a word three places after one, a negation with a score of its own, two negations in a row, Dutch negations, and a negation at the end?",
                       "nl": "Goed voor de voorbeelden. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor een lege lijst, een zin zonder ontkenning, een woord twee plaatsen na een ontkenning, een woord drie plaatsen erna, een ontkenning met een eigen score, twee ontkenningen achter elkaar, Nederlandse ontkenningen, en een ontkenning aan het eind?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Where does your function give back its total?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Waar geeft je functie het totaal terug?"},
            "correct": {"en": "Your `score` is right in all {n} cases. Look at the picture: how many held-out sentences each lexicon gets right without negation and with your score, and sentences whose label your score changes.",
                        "nl": "Je `score` klopt in alle {n} gevallen. Kijk naar de afbeelding: hoeveel achtergehouden zinnen elk lexicon goed heeft zonder ontkenning en met jouw score, en zinnen waarvan jouw score het label verandert."},
        },
    ),
    "words-lists": Sort(
        name="words-lists",
        groups={
            "agree": {"en": "The two lists agree", "nl": "De twee lijsten zijn het eens"},
            "reviews": {"en": "Only the list from the reviews has it", "nl": "Alleen de lijst uit de recensies heeft het"},
            "disagree": {"en": "The two lists disagree", "nl": "De twee lijsten zijn het oneens"},
        },
        items={k: {lang: f"*{v[lang]}*" for lang in ("en", "nl")} for k, v in SORT_WORDS.items()},
        solution=SORT_SOLUTION,
        nudge={
            "great": {"en": "Is *great* praise or a complaint, in a dictionary and in a review?",
                      "nl": "Is *geweldig* lof of een klacht, in een woordenboek en in een recensie?"},
            "terrible": {"en": "Is there a review where *terrible* is good news?",
                         "nl": "Is er een recensie waarin *verschrikkelijk* goed nieuws is?"},
            "boring": {"en": "Does anyone praise a film or a book for being *boring*?",
                       "nl": "Prijst iemand een film of een boek omdat het *saai* is?"},
            "minutes": {"en": "Pattern's list holds mostly adjectives. Is *minutes* one? And where do minutes come up in a review of a phone or a film?",
                        "nl": "De lijst van Pattern bevat vooral bijvoeglijke naamwoorden. Is *minuten* er een? En waar komen minuten voor in een recensie van een telefoon of een film?"},
            "price": {"en": "Pattern's list holds mostly adjectives. Is *price* one? And which reviews talk about the price?",
                      "nl": "De lijst van Pattern bevat vooral bijvoeglijke naamwoorden. Is *prijs* er een? En welke recensies hebben het over de prijs?"},
            "plot": {"en": "Pattern's list holds mostly adjectives. Is *plot* one? And which reviews of a film talk about the plot?",
                     "nl": "De lijst van Pattern bevat vooral bijvoeglijke naamwoorden. Is *plot* er een? En welke recensies van een film hebben het over het plot?"},
            "small": {"en": "In a dictionary, *small* often means below average. Is a small phone good news or bad news?",
                      "nl": "In een woordenboek betekent *klein* vaak iets minder goeds. Is een kleine telefoon goed of slecht nieuws?"},
            "thin": {"en": "In Pattern's list, *thin* means lacking excess flesh. Is a thin phone or a thin pizza good news or bad news?",
                     "nl": "In de lijst van Pattern betekent *dun* mager, plat of vloeibaar. Is een dunne telefoon of een dunne crêpe goed of slecht nieuws?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each word in a group: what do the two lexicons say of it?",
                      "nl": "Zet elk woord in een groep: wat zeggen de twee lexicons erover?"},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The word {item} belongs in another group.",
                      "nl": "Het woord {item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. *thin* gets −0.40 in Pattern's list, which reads it as lacking excess flesh, and +0.22 from the reviews, where it stands in sentences such as *You'll love how thin it is.* *plot* is not in Pattern's list, and gets −0.78 from the reviews: 23 times in negative sentences, such as *No plot whatsoever!*, and 3 times in positive ones.",
                        "nl": "Alle {n} kloppen. *dun* krijgt −0,17 in de lijst van Pattern, die het leest als mager, plat of vloeibaar, en +0,37 uit de recensies, waar het staat in zinnen als *Deze telefoon is dun en licht en het scherm is prachtig.* *plot* staat niet in de lijst van Pattern, en krijgt −0,80 uit de recensies: 18 keer in negatieve zinnen, zoals *Totaal geen plot!*, en 2 keer in positieve."},
        },
    ),
}


CASES = [
    ("lexicon", _LEX_START, "empty", None),
    ("lexicon", _LEX, "correct", None),
    ("lexicon", _LEX.replace("np.log10(", "math.log10("), "correct", None),
    ("lexicon", _LEX_HEAD + "    positive = {}\n    negative = {}\n    for label, words in sentences:\n"
                "        for word in words:\n            positive[word] = positive.get(word, 0) + label\n"
                "            negative[word] = negative.get(word, 0) + 1 - label\n"
                "    return {w: np.log10((positive[w] + 1) / (negative[w] + 1)) for w in positive}\n", "correct", None),
    ("lexicon", _LEX_NO_ONE, "error", "no-plus-one"),
    ("lexicon", _LEX_NO_ONE.replace("np.log10(", "math.log10("), "error", "no-plus-one"),
    ("lexicon", _LEX_TOP.replace("np.log10(", "math.log10("), "error", "log-of-zero"),
    ("lexicon", _LEX_LOGS, "wrong", "logs"),
    ("lexicon", _LEX_TOP, "wrong", "top-only"),
    ("lexicon", _LEX_SWAPPED, "wrong", "swapped"),
    ("lexicon", _LEX_LN, "wrong", "natural-log"),
    ("lexicon", _LEX_LN.replace("np.log(", "math.log("), "wrong", "natural-log"),
    ("lexicon", _LEX_DIFFERENCE, "wrong", "difference"),
    ("lexicon", _LEX_RATIO, "wrong", "no-log"),
    ("lexicon", _LEX_COUNTS, "wrong", "counts"),
    ("lexicon", _LEX_ONCE, "wrong", "once"),
    ("lexicon", _LEX_RESET, "wrong", "reset"),
    ("lexicon", _LEX_FIRST, "wrong", "first-lost"),
    ("lexicon", _LEX_RETURN_IN, "wrong", "return-in-loop"),
    ("lexicon", _LEX.replace("    return scores\n", "    return list(scores.values())\n"), "wrong", "not-dict"),
    ("lexicon", _LEX.replace("    return scores\n", "    print(scores)\n"), "wrong", "prints"),
    ("lexicon", _LEX.replace("    scores = {}\n", "    scores = {}\n    if len(sentences) == 0:\n        return {\"\": 0}\n"),
     "almost", None),
    ("lexicon", _LEX_HEAD + "    ...\n    return {}\n", "wrong", "dots-left"),
    ("score", _SCORE_START, "empty", None),
    ("score", _SCORE, "correct", None),
    ("score", _SCORE_HEAD + "    total = 0\n    last = -10\n    for i in range(len(words)):\n"
              "        if words[i] in lexicon:\n            if i - last <= 2:\n"
              "                total = total - lexicon[words[i]]\n            else:\n"
              "                total = total + lexicon[words[i]]\n        if words[i] in negations:\n"
              "            last = i\n    return total\n", "correct", None),
    ("score", _SCORE_ONE, "wrong", "one-word"),
    ("score", _SCORE_THREE, "wrong", "three-words"),
    ("score", _SCORE_REST, "wrong", "rest"),
    ("score", _SCORE_ITSELF, "wrong", "itself"),
    ("score", _SCORE_IGNORED, "wrong", "ignored"),
    ("score", _SCORE_HALF, "wrong", "half"),
    ("score", _SCORE_FIXED, "wrong", "fixed-word"),
    ("score", _SCORE_RETURN_IN, "wrong", "return-in-loop"),
    ("score", _SCORE_LABEL, "wrong", "label"),
    ("score", _SCORE.replace("    return total\n", "    return [total]\n"), "wrong", "not-number"),
    ("score", _SCORE.replace("    return total\n", "    print(total)\n"), "wrong", "prints"),
    ("score", _SCORE.replace("    total = 0\n", "    total = 0\n    if len(words) == 0:\n        return 1\n"), "almost", None),
    ("score", _SCORE_HEAD + "    ...\n    return 0\n", "wrong", "dots-left"),
    ("words-lists", {}, "empty", None),
    ("words-lists", dict(SORT_SOLUTION), "correct", None),
    ("words-lists", {"great": "agree", "minutes": "reviews"}, "wrong", "unfinished"),
    ("words-lists", {"great": "disagree"}, "wrong", "wrong:great"),
    ("words-lists", {"terrible": "reviews"}, "wrong", "wrong:terrible"),
    ("words-lists", {"boring": "disagree"}, "wrong", "wrong:boring"),
    ("words-lists", {"minutes": "agree"}, "wrong", "wrong:minutes"),
    ("words-lists", {"price": "disagree"}, "wrong", "wrong:price"),
    ("words-lists", {"plot": "agree"}, "wrong", "wrong:plot"),
    ("words-lists", {"small": "agree"}, "wrong", "wrong:small"),
    ("words-lists", {"thin": "reviews"}, "wrong", "wrong:thin"),
]
