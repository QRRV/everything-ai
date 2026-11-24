# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-sentiment: the label of a whole review from the labels of its sentences (majority), and
the weight of each word around the last "but" (scales); and sort sentences with "but" by where their judgment lies;
with the pictures of what each function gives for the course's reviews and for the held-out sentences with "but", a
picture of the idea and a box for a review of your own.

The sentences are "Sentiment Labelled Sentences" (Kotzias, Denil, de Freitas and Smyth, KDD 2015; UCI Machine Learning
Repository, CC BY 4.0, https://doi.org/10.24432/C57604), in data/uci-sentiment.tsv.gz, and the course's Dutch
translation of them in data/uci-sentiment-nl.tsv.gz; the 18 short reviews are the course's own, in
data/course-reviews.tsv. In each language, the classifier is the logistic regression on TF-IDF values of the lesson
nlp-logistic-text, trained here again in the same way on the same 2,400 sentences (every fifth sentence is held out).
Sentences marked as not shown are counted but never shown. Every number in the text and the pictures is computed by
the course (tests/test_nlp_sentiment.py computes them again)."""
import ast
import contextlib
import html
import io
import math
import re

import numpy as np

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

_KEY = "nlp-sentiment"

# --- the sentences and the classifier ----------------------------------------------------------------------------------

HELD_OUT = 5   # every fifth sentence is held out, as in the lessons A document as a vector and Logistic regression
STEPS = 300    # steps of gradient descent, as in Logistic regression on words
RATE = 1.0     # its learning rate
HALF = 0.5     # the weight of a word before the last "but"
BUT = {"en": "but", "nl": "maar"}
LABELS = {"en": {1: "positive", 0: "negative"}, "nl": {1: "positief", 0: "negatief"}}


def words(text: str) -> list:
    """The words of a text in small letters: runs of letters, with an apostrophe inside a word as in *don't*."""
    return books.words(text.lower())


def _all(lang: str) -> tuple:
    return books.review_sentences() if lang == "en" else books.dutch_review_sentences()


def training(lang: str) -> list:
    """The 2,400 training sentences in one language, as (label, text, shown)."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all(lang))
            if k % HELD_OUT != HELD_OUT - 1]


def held_out(lang: str) -> list:
    """The 600 held-out sentences in one language, every fifth sentence of the set, as (label, text, shown)."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all(lang))
            if k % HELD_OUT == HELD_OUT - 1]


def _model(lang: str) -> dict:
    """The classifier of Logistic regression on words in one language, with NumPy: the TF-IDF values over the training
    sentences, then STEPS steps of gradient descent with learning rate RATE from weight 0 for every word and bias 0.
    {"idf": {word: log10(2,400 / df)}, "weights": {word: weight}, "bias": b}."""
    def make():
        train = training(lang)
        df = {}
        for _, text, _ in train:
            for w in set(words(text)):
                df[w] = df.get(w, 0) + 1
        vocab = sorted(df)
        cols = {w: k for k, w in enumerate(vocab)}
        idf = {w: math.log10(len(train) / df[w]) for w in vocab}
        x = np.zeros((len(train), len(vocab)))
        for i, (_, text, _) in enumerate(train):
            for w in words(text):
                x[i, cols[w]] += idf[w]
        y = np.array([label for label, _, _ in train], dtype=float)
        wv, b = np.zeros(len(vocab)), 0.0
        for _ in range(STEPS):
            p = 1 / (1 + np.exp(-(x @ wv + b)))
            wv = wv - RATE * ((p - y) @ x / len(train))
            b = b - RATE * float((p - y).mean())
        return {"idf": idf, "weights": {w: float(wv[k]) for k, w in enumerate(vocab)}, "bias": b}
    return kept(f"{_KEY}:model:{lang}", make)


def logit(text: str, lang: str, scales: list | None = None) -> float:
    """z of the classifier for a text: the bias plus, for each word of the text, its log10(2,400 / df) times its weight,
    times its scale when `scales` is given (one number for each word, in order). A word that no training sentence has
    adds nothing. Without scales, this is the bias plus TF-IDF value times weight, as in Logistic regression on words."""
    m = _model(lang)
    ws = words(text)
    scales = [1.0] * len(ws) if scales is None else scales
    z = m["bias"]
    for w, s in zip(ws, scales):
        if w in m["idf"]:
            z += m["idf"][w] * m["weights"][w] * s
    return z


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def probability(text: str, lang: str, scales: list | None = None) -> float:
    """The probability that the classifier in `lang` gives that a text is positive."""
    return _sigmoid(logit(text, lang, scales))


def label_of(text: str, lang: str, scales: list | None = None) -> int:
    """1 when the classifier gives the text a probability of positive of 0.5 or more, else 0."""
    return 1 if logit(text, lang, scales) >= 0 else 0


def weight(word: str, lang: str) -> float:
    return _model(lang)["weights"].get(word, 0.0)


def bias(lang: str) -> float:
    return _model(lang)["bias"]


def held_right(lang: str, scales_of=None) -> list:
    """For each held-out sentence, whether the classifier gives it its own label; with `scales_of(words)` giving the
    scale of each word."""
    out = []
    for label, text, _ in held_out(lang):
        scales = None if scales_of is None else scales_of(words(text))
        out.append(label_of(text, lang, scales) == label)
    return out


def with_but(lang: str) -> list:
    """The held-out sentences that have the word "but" (in Dutch "maar"), as (label, text, shown)."""
    return [(label, text, shown) for label, text, shown in held_out(lang) if BUT[lang] in words(text)]


# --- the course's reviews ----------------------------------------------------------------------------------------------

KINDS = {"en": {"product": "product", "film": "film", "restaurant": "restaurant"},
         "nl": {"product": "product", "film": "film", "restaurant": "restaurant"}}


def reviews(lang: str) -> list:
    """The course's 18 reviews in one language, as a new list of (label, sentences), with each sentence as (label,
    text): label 1 positive, 0 negative."""
    k = 1 if lang == "en" else 2
    return [(label, [(s[0], s[k]) for s in sentences]) for _, label, sentences in books.course_reviews()]


def review_kinds() -> list:
    return [kind for kind, _, _ in books.course_reviews()]


def whole(sentences: list) -> str:
    """The text of a review: its sentences, joined with a space."""
    return " ".join(text for _, text in sentences)


def sentence_labels(sentences: list, lang: str) -> list:
    """The label the classifier gives each sentence of a review."""
    return [label_of(text, lang) for _, text in sentences]


def _drop(scales_: list) -> list:
    """The scales of the guess: 0 instead of HALF, so the words before the last "but" add nothing."""
    return [0.0 if x == HALF else x for x in scales_]


def numbers(lang: str) -> dict:
    """The numbers the lesson states for one language."""
    def make():
        but = BUT[lang]
        revs = reviews(lang)
        out = {"held": len(held_out(lang)), "held_right": sum(held_right(lang)),
               "held_half": sum(held_right(lang, lambda ws: scales(ws, but))),
               "held_drop": sum(held_right(lang, lambda ws: _drop(scales(ws, but))))}
        wb = with_but(lang)
        out["but"] = len(wb)
        for key, of in (("but_right", None), ("but_half", lambda ws: scales(ws, but)),
                        ("but_drop", lambda ws: _drop(scales(ws, but)))):
            out[key] = sum(label_of(t, lang, None if of is None else of(words(t))) == label for label, t, _ in wb)
        out["fixed"] = [t for label, t, _ in wb
                        if label_of(t, lang) != label and label_of(t, lang, scales(words(t), but)) == label]
        out["broken"] = [t for label, t, _ in wb
                         if label_of(t, lang) == label and label_of(t, lang, scales(words(t), but)) != label]
        out["sentences"] = sum(len(s) for _, s in revs)
        out["sentences_right"] = sum(label_of(t, lang) == sl for _, s in revs for sl, t in s)
        out["majority_right"] = sum(majority(sentence_labels(s, lang)) == label for label, s in revs)
        out["whole_right"] = sum(label_of(whole(s), lang) == label for label, s in revs)
        out["whole_half"] = sum(label_of(whole(s), lang, scales(words(whole(s)), but)) == label for label, s in revs)
        out["majority_wrong"] = [k for k, (label, s) in enumerate(revs) if majority(sentence_labels(s, lang)) != label]
        out["whole_wrong"] = [k for k, (label, s) in enumerate(revs) if label_of(whole(s), lang) != label]
        out["half_wrong"] = [k for k, (label, s) in enumerate(revs)
                             if label_of(whole(s), lang, scales(words(whole(s)), but)) != label]
        label, s = revs[IDEA_REVIEW]
        b = bias(lang)
        out["idea_p"] = [probability(t, lang) for _, t in s]
        out["idea_parts"] = [logit(t, lang) - b for _, t in s]
        out["idea_whole"] = probability(whole(s), lang)
        out["bias"] = b
        out["vocab"] = len(_model(lang)["idf"])
        return out
    return kept(f"{_KEY}:numbers:{lang}", make)


def split_sentences(text: str) -> list:
    """A text cut into sentences at the spaces after a full stop, a question mark or an exclamation mark (or after one
    of them and a closing quote or bracket), and at line breaks; empty pieces left out."""
    pieces = re.split(r"(?<=[.!?…])\s+|(?<=[.!?…][\"'”’)\]])\s+|\n+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


# --- the course's solutions --------------------------------------------------------------------------------------------

def majority(labels):
    """The course's solution of Exercise 1: 1 when more sentences are positive than negative, 0 when fewer, and on a
    tie the label of the last sentence."""
    positive = 0
    negative = 0
    for label in labels:
        if label == 1:
            positive = positive + 1
        else:
            negative = negative + 1
    if positive > negative:
        return 1
    if negative > positive:
        return 0
    return labels[-1]


def scales(words, word):
    """The course's solution of Exercise 2: HALF for each word before the last `word`, 1.0 for that word and every
    word after it; 1.0 for every word when `word` is not there."""
    last = -1
    for i in range(len(words)):
        if words[i] == word:
            last = i
    out = []
    for i in range(len(words)):
        if i < last:
            out.append(0.5)
        else:
            out.append(1.0)
    return out


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
_WORD = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"
_WORD_HALF = _WORD + ";color:var(--ea-ink-3)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"


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


def _dec(x: float, lang: str, digits: int = 3) -> str:
    """A number with `digits` decimals: 0.450 in English, 0,450 in Dutch."""
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _p_bar(p: float, width: float = 6.0) -> str:
    """A bar for a probability, as long as `width` for 1, in the color of the text, with a thin line at 0.5."""
    w = max(0.12, width * min(max(p, 0.0), 1.0))
    return (f'<span style="position:relative;display:inline-block;width:{width:.2f}rem;height:0.6rem;'
            f'vertical-align:middle;background:var(--ea-line);border-radius:2px">'
            f'<span style="{_BAR};position:absolute;left:0;top:0;width:{w:.2f}rem;background:var(--ea-ink-2)"></span>'
            f'<span style="position:absolute;left:{width / 2:.2f}rem;top:-0.15rem;width:1px;height:0.9rem;'
            f'background:var(--ea-page)"></span></span>')


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; no head when
    `head` is empty."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


_HIDDEN = {"en": "(a sentence the course does not show)", "nl": "(een zin die de course niet laat zien)"}


# A sentence of the set that says he or she (in Dutch also ze, hun, zijn and the like) stays out of a picture, as the
# course's own text has none: the picture then shows _HIDDEN in its place.
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


def _shows(text: str, shown: int, lang: str) -> bool:
    """Whether a picture shows a sentence of the set: the set lets it be shown, and it has no he or she."""
    return bool(shown) and _PRONOUN[lang].search(text) is None


def _sentence(text: str, shown: int, lang: str) -> str:
    if not _shows(text, shown, lang):
        return f'<span style="{_TEXT};font-style:italic;color:var(--ea-ink-2)">{html.escape(_HIDDEN[lang])}</span>'
    return f'<span style="{_TEXT}">{html.escape(text)}</span>'


def _halved(text: str, sc: list) -> str:
    """A text with each word in the color of the text, and a word at less than its full weight faint."""
    ws = words(text)
    out, k, pos = [], 0, 0
    low = text.lower()
    for w, s in zip(ws, sc):
        at = low.find(w, pos)
        if at < 0:
            break
        out.append(html.escape(text[pos:at]))
        style = _WORD_HALF if s < 1 else _WORD
        out.append(f'<span style="{style}">{html.escape(text[at:at + len(w)])}</span>')
        pos = at + len(w)
        k += 1
    out.append(html.escape(text[pos:]))
    return f'<span style="{_TEXT}">{"".join(out)}</span>'


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


def _sign(label: int) -> str:
    return "+" if label == 1 else "−"


# --- the idea's picture: one review, per sentence and as a whole --------------------------------------------------------

IDEA_REVIEW = 1   # the kettle: three sentences of praise, then it leaks

_IDEA_TEXT = {
    "en": {"title": "One review, per sentence and as a whole",
           "head": ["Sentence", "Label", "p", "", "Classifier"],
           "majority": "Majority of the sentences: {pos} positive, {neg} negative",
           "whole": "The whole review as one text",
           "own": "Label of the review:",
           "note": "Gray: the labels the course gave each sentence and the review. In the color of the text: what the "
                   "classifier says, with p, the probability of positive, as a bar; the thin line in each bar is 0.5. "
                   "Purple: a label of the majority or of the whole text other than the review's own.",
           "alt": "{title}. {rows}. {majority}: {m}. {whole}: p = {p}, {w}. {own} {note}"},
    "nl": {"title": "Eén recensie, per zin en als geheel",
           "head": ["Zin", "Label", "p", "", "Classifier"],
           "majority": "Meerderheid van de zinnen: {pos} positief, {neg} negatief",
           "whole": "De hele recensie als één tekst",
           "own": "Label van de recensie:",
           "note": "Grijs: de labels die de course elke zin en de recensie gaf. In de kleur van de tekst: wat de "
                   "classifier zegt, met p, de kans op positief, als balk; het dunne streepje in elke balk is 0,5. "
                   "Paars: een label van de meerderheid of van de hele tekst dat afwijkt van het label van de recensie.",
           "alt": "{title}. {rows}. {majority}: {m}. {whole}: p = {p}, {w}. {own} {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    label, sentences = reviews(lang)[IDEA_REVIEW]
    rows_, alts = [], []
    for own, text in sentences:
        p = probability(text, lang)
        said = 1 if p >= 0.5 else 0
        rows_.append([_sentence(text, 1, lang), _chip(LABELS[lang][own]), _dec(p, lang), _p_bar(p),
                      _chip(LABELS[lang][said], _CHIP_INK)])
        alts.append(f"{text} ({LABELS[lang][own]}): p = {_dec(p, lang)}, {LABELS[lang][said]}")
    said = sentence_labels(sentences, lang)
    m = majority(said)
    pos = sum(said)
    m_text = w["majority"].format(pos=pos, neg=len(said) - pos)
    rows_.append([f"<b>{html.escape(m_text)}</b>", "", "", "",
                  _chip(LABELS[lang][m], _CHIP_WRONG if m != label else _CHIP_INK)])
    p = probability(whole(sentences), lang)
    wl = 1 if p >= 0.5 else 0
    rows_.append([f"<b>{html.escape(w['whole'])}</b>", "", _dec(p, lang), _p_bar(p),
                  _chip(LABELS[lang][wl], _CHIP_WRONG if wl != label else _CHIP_INK)])
    own = f'{w["own"]} {LABELS[lang][label]}.'
    own_html = f'<div style="{_LABEL}">{html.escape(w["own"])} {_chip(LABELS[lang][label])}</div>'
    body = _table(rows_, w["head"], numbers=(2,)) + own_html + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], rows="; ".join(alts), majority=m_text, m=LABELS[lang][m], whole=w["whole"],
        p=_dec(p, lang), w=LABELS[lang][wl], own=own, note=w["note"]))


# --- the picture under Exercise 1: your majority vote on the 18 reviews -------------------------------------------------

_MAJ_TEXT = {
    "en": {"title": "Your majority vote on the 18 reviews",
           "head": ["", "Sentences", "Your majority", "Whole text", "Label"],
           "count": "Your majority vote gives {m} of the 18 reviews their own label; the whole review as one text, {w}.",
           "note": "In the color of the text: what the classifier says, + for positive and − for negative, for each "
                   "sentence and for the whole review as one text. Blue: what your majority gives. Gray: the label the "
                   "course gave the review. Purple: a label other than the review's own.",
           "stopped": "Your function stopped with an error on the labels of a review, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the 18 reviews, so the picture stays empty.",
           "odd": "Your function does not give 1 or 0 for every review, so the picture stays empty.",
           "alt": "{title}. {rows}. {count} {note}"},
    "nl": {"title": "Jouw meerderheid op de 18 recensies",
           "head": ["", "Zinnen", "Jouw meerderheid", "Hele tekst", "Label"],
           "count": "Jouw meerderheid geeft {m} van de 18 recensies het eigen label; de hele recensie als één tekst {w}.",
           "note": "In de kleur van de tekst: wat de classifier zegt, + voor positief en − voor negatief, voor elke zin "
                   "en voor de hele recensie als één tekst. Blauw: wat jouw meerderheid geeft. Grijs: het label dat de "
                   "course de recensie gaf. Paars: een ander label dan het label van de recensie.",
           "stopped": "Je functie stopte met een fout op de labels van een recensie, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de 18 recensies, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elke recensie 1 of 0, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {count} {note}"},
}


def _is_label(v) -> bool:
    """1 or 0, also as 1.0 or 0.0, which the check counts as right too; never True or False."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    return float(v) in (0.0, 1.0)


def majority_rows(fn, lang: str) -> list:
    """For each of the 18 reviews: (label, the classifier's sentence labels, what fn gives them, the whole text's
    label)."""
    out = []
    for label, sentences in reviews(lang):
        said = sentence_labels(sentences, lang)
        out.append((label, said, fn(list(said)), label_of(whole(sentences), lang)))
    return out


def majority_picture(seen) -> str:
    lang = seen.lang
    w = _MAJ_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    got, failed = _run_or_say(lambda: majority_rows(fn, lang), w, lambda rs: all(_is_label(r[2]) for r in rs))
    if failed:
        return failed
    rows_, alts = [], []
    kinds = review_kinds()
    for k, (label, said, mine, wl) in enumerate(got, 1):
        mine = int(mine)
        sents = "".join(_chip(_sign(s), _CHIP_INK) for s in said)
        rows_.append([f"{k}. {html.escape(KINDS[lang][kinds[k - 1]])}", sents,
                      _chip(LABELS[lang][mine], _CHIP_WRONG if mine != label else _CHIP_PRED),
                      _chip(LABELS[lang][wl], _CHIP_WRONG if wl != label else _CHIP_INK),
                      _chip(LABELS[lang][label])])
        alts.append(f"{k}: {' '.join(_sign(s) for s in said)}, {LABELS[lang][mine]}, {LABELS[lang][wl]}, "
                    f"{LABELS[lang][label]}")
    count = w["count"].format(m=sum(int(r[2]) == r[0] for r in got), w=sum(r[3] == r[0] for r in got))
    body = _table(rows_, w["head"]) + _label(count) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), count=count,
                                                              note=w["note"]))


# --- the picture under Exercise 2: your scales on the held-out sentences with "but" -------------------------------------

_SCALES_TEXT = {
    "en": {"title": "Your scales on the held-out sentences",
           "count": "Of the {n} held-out sentences with *{but}*, the classifier gives {plain} their own label without "
                    "scales, and {mine} with yours; of all 600, {plain_all} and {mine_all}. The whole reviews of the "
                    "course with your scales: {rev} of 18, against {rev_plain} without.",
           "changed": "The sentences whose label changes with your scales:",
           "none": "With your scales, no sentence gets another label.",
           "head": ["Sentence", "Label", "p without", "p with yours"],
           "note": "Faint: a word that your scales give less than its full weight. Gray: the label of the sentence. "
                   "Blue: p with your scales, where the label is now right. Purple: where it is now wrong.",
           "stopped": "Your function stopped with an error on the words of a held-out sentence, so the picture stays "
                      "empty.",
           "slow": "Your function took more than {s:g} seconds for the held-out sentences, so the picture stays empty.",
           "odd": "Your function does not give one number for each word of every sentence, so the picture stays empty.",
           "alt": "{title}. {count} {changed} {rows}. {note}"},
    "nl": {"title": "Jouw scales op de achtergehouden zinnen",
           "count": "Van de {n} achtergehouden zinnen met *{but}* geeft de classifier er {plain} het eigen label zonder "
                    "scales, en {mine} met de jouwe; van alle 600 {plain_all} en {mine_all}. De hele recensies van de "
                    "course met jouw scales: {rev} van de 18, tegen {rev_plain} zonder.",
           "changed": "De zinnen die met jouw scales een ander label krijgen:",
           "none": "Met jouw scales krijgt geen enkele zin een ander label.",
           "head": ["Zin", "Label", "p zonder", "p met de jouwe"],
           "note": "Vaag: een woord dat jouw scales minder dan de volle weight geven. Grijs: het label van de zin. "
                   "Blauw: p met jouw scales, waar het label nu goed is. Paars: waar het label nu fout is.",
           "stopped": "Je functie stopte met een fout op de woorden van een achtergehouden zin, dus de afbeelding blijft "
                      "leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de achtergehouden zinnen, dus de afbeelding "
                   "blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord van elke zin één getal, dus de afbeelding blijft leeg.",
           "alt": "{title}. {count} {changed} {rows}. {note}"},
}


def _is_scales(v, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(
        isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
        and math.isfinite(float(x)) for x in v)


def scales_rows(fn, lang: str) -> dict:
    """What fn gives the words of every held-out sentence and of every course review, with the word "but" of the
    language: {"held": [scales of each held-out sentence], "reviews": [scales of each whole review]}."""
    held = [fn(words(text), BUT[lang]) for _, text, _ in held_out(lang)]
    revs = [fn(words(whole(sentences)), BUT[lang]) for _, sentences in reviews(lang)]
    return {"held": held, "reviews": revs}


def scales_picture(seen) -> str:
    lang = seen.lang
    w = _SCALES_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    sents = held_out(lang)
    revs = reviews(lang)

    def ok(got):
        return (all(_is_scales(s, len(words(t))) for s, (_, t, _) in zip(got["held"], sents))
                and all(_is_scales(s, len(words(whole(r)))) for s, (_, r) in zip(got["reviews"], revs)))
    got, failed = _run_or_say(lambda: scales_rows(fn, lang), w, ok)
    if failed:
        return failed
    but = BUT[lang]
    n = plain = mine = plain_all = mine_all = 0
    rows_, alts = [], []
    for (label, text, shown), sc in zip(sents, got["held"]):
        sc = [float(x) for x in sc]
        p0, p1 = probability(text, lang), probability(text, lang, sc)
        r0, r1 = (p0 >= 0.5) == (label == 1), (p1 >= 0.5) == (label == 1)
        plain_all += r0
        mine_all += r1
        if but in words(text):
            n += 1
            plain += r0
            mine += r1
        if (p0 >= 0.5) != (p1 >= 0.5):
            seen_ = _shows(text, shown, lang)
            shown_text = _halved(text, sc) if seen_ else _sentence(text, 0, lang)
            rows_.append([shown_text, _chip(LABELS[lang][label]), _dec(p0, lang),
                          _chip(_dec(p1, lang), _CHIP_PRED if r1 else _CHIP_WRONG)])
            alts.append(f"{text if seen_ else _HIDDEN[lang]} ({LABELS[lang][label]}): {_dec(p0, lang)} → "
                        f"{_dec(p1, lang)}")
    rev = sum(label_of(whole(s), lang, [float(x) for x in sc]) == label for (label, s), sc in zip(revs, got["reviews"]))
    rev_plain = sum(label_of(whole(s), lang) == label for label, s in revs)
    count = w["count"].format(n=n, but=but, plain=plain, mine=mine, plain_all=plain_all, mine_all=mine_all, rev=rev,
                              rev_plain=rev_plain)
    changed = w["changed"] if rows_ else w["none"]
    body = _label(count) + _label(changed) + (_table(rows_, w["head"], numbers=(2, 3)) if rows_ else "") + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], count=count.replace("*", ""),
                                                              changed=changed, rows="; ".join(alts), note=w["note"]))


# --- Try it: a review of your own ---------------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Your review",
           "head": ["Sentence", "p", "", "Classifier"],
           "majority": "Majority of the sentences: {pos} positive, {neg} negative",
           "whole": "The whole review as one text",
           "but": "The whole review, with half weight before the last *but*",
           "empty": "Type a review of a few sentences in the box.",
           "note": "In the color of the text: what the classifier says, with p, the probability of positive, as a bar; "
                   "the thin line in each bar is 0.5. Faint: a word at half weight, before the last *but*. The "
                   "classifier knows the {n} words of the training sentences; a word it does not know adds nothing.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw recensie",
           "head": ["Zin", "p", "", "Classifier"],
           "majority": "Meerderheid van de zinnen: {pos} positief, {neg} negatief",
           "whole": "De hele recensie als één tekst",
           "but": "De hele recensie, met halve weight vóór de laatste *maar*",
           "empty": "Typ een recensie van een paar zinnen in het vak.",
           "note": "In de kleur van de tekst: wat de classifier zegt, met p, de kans op positief, als balk; het dunne "
                   "streepje in elke balk is 0,5. Vaag: een woord met halve weight, vóór de laatste *maar*. De "
                   "classifier kent de {n} woorden van de trainingszinnen; een woord dat de classifier niet kent, telt "
                   "niet mee.",
           "alt": "{title}. {rows}. {note}"},
}

TRY_LONGEST = 2000   # characters of a review of your own that the picture reads
TRY_START = {   # the review in the box at the start: the whole text is positive, and negative with half weight
    "en": "The screen is bright and the sound is clear. The battery lasts all day. But after a month it stopped working, "
          "and support never answered.",
    "nl": "Het scherm is mooi en het geluid is helder. De accu gaat de hele dag mee. Maar na een maand ging de oplader "
          "kapot, en ik heb alles teruggestuurd.",
}


def try_picture(text: str, lang: str) -> str:
    """The sentences of a review someone typed, each with its probability of positive, then the majority, the whole
    text and the whole text with half weight before the last "but" (in Dutch "maar")."""
    w = _TRY_TEXT[lang]
    text = (text or "")[:TRY_LONGEST]
    sentences = split_sentences(text)
    if not sentences:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    rows_, alts = [], []
    said = []
    for s in sentences:
        p = probability(s, lang)
        said.append(1 if p >= 0.5 else 0)
        rows_.append([_sentence(s, 1, lang), _dec(p, lang), _p_bar(p), _chip(LABELS[lang][said[-1]], _CHIP_INK)])
        alts.append(f"{s}: p = {_dec(p, lang)}, {LABELS[lang][said[-1]]}")
    m = majority(said)
    pos = sum(said)
    m_text = w["majority"].format(pos=pos, neg=len(said) - pos)
    rows_.append([f"<b>{html.escape(m_text)}</b>", "", "", _chip(LABELS[lang][m], _CHIP_INK)])
    alts.append(f"{m_text}: {LABELS[lang][m]}")
    full = " ".join(sentences)
    p = probability(full, lang)
    rows_.append([f"<b>{html.escape(w['whole'])}</b>", _dec(p, lang), _p_bar(p),
                  _chip(LABELS[lang][1 if p >= 0.5 else 0], _CHIP_INK)])
    alts.append(f"{w['whole']}: p = {_dec(p, lang)}")
    sc = scales(words(full), BUT[lang])
    if any(s != 1 for s in sc):
        pb = probability(full, lang, sc)
        rows_.append([f"<b>{_marked(w['but'])}</b><br>{_halved(full, sc)}", _dec(pb, lang), _p_bar(pb),
                      _chip(LABELS[lang][1 if pb >= 0.5 else 0], _CHIP_INK)])
        alts.append(f"{w['but'].replace('*', '')}: p = {_dec(pb, lang)}")
    note = w["note"].format(n=_num(len(_model(lang)["idf"]), lang))
    body = _table(rows_, w["head"], numbers=(1,)) + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts),
                                                              note=note.replace("*", "")))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["majority", "scales", "where-judgment"]

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


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


# Exercise 1: majority

MAJ_CASES = [
    ([1, 1, 1, 0],),
    ([0, 1],),
    ([1, 0],),
    ([1],),
    ([0],),
    ([0, 0, 1, 1],),
    ([1, 1, 0, 0],),
    ([0, 1, 0, 1, 0],),
    ([1, 0, 1, 0, 1, 1],),
    ([0, 0, 0, 1],),
]
_MAJ_HEAD = "def majority(labels):\n"
_MAJ_START = _MAJ_HEAD + "    ...\n"
_MAJ = (_MAJ_HEAD
        + "    positive = 0\n"
        + "    negative = 0\n"
        + "    for label in labels:\n"
        + "        if label == 1:\n"
        + "            positive = positive + 1\n"
        + "        else:\n"
        + "            negative = negative + 1\n"
        + "    if positive > negative:\n"
        + "        return 1\n"
        + "    if negative > positive:\n"
        + "        return 0\n"
        + "    return labels[-1]\n")
_MAJ_TIE_FIRST = _MAJ.replace("    return labels[-1]\n", "    return labels[0]\n")
_MAJ_TIE_POSITIVE = _MAJ.replace("    return labels[-1]\n", "    return 1\n")
_MAJ_TIE_NEGATIVE = _MAJ.replace("    return labels[-1]\n", "    return 0\n")
_MAJ_COUNT = (_MAJ_HEAD
              + "    positive = 0\n"
              + "    for label in labels:\n"
              + "        if label == 1:\n"
              + "            positive = positive + 1\n"
              + "    return positive\n")
_MAJ_SHARE = _MAJ_COUNT.replace("    return positive\n", "    return positive / len(labels)\n")
_MAJ_LAST = _MAJ_HEAD + "    return labels[-1]\n"
_MAJ_SWAPPED = (_MAJ.replace("        return 1\n    if negative", "        return 0\n    if negative")
                .replace("        return 0\n    return labels", "        return 1\n    return labels"))
_MAJ_EMPTY_LIST = (_MAJ_HEAD
                   + "    positive = 0\n"
                   + "    for label in labels:\n"
                   + "        positive = positive + label\n"
                   + "    if positive > len(labels) - positive:\n"
                   + "        return 1\n"
                   + "    return 0\n")


def _not_label(got, text) -> bool:
    return got is not None and not _is_label(got)


# Exercise 2: scales

SC_CASES = [
    (["good", "food", "but", "slow", "service"], "but"),
    (["great", "food"], "but"),
    (["but", "the", "pizza", "is", "great"], "but"),
    (["nice", "but", "loud", "but", "worth", "it"], "but"),
    (["it", "works", "but"], "but"),
    (["lekker", "eten", "maar", "trage", "bediening"], "maar"),
    (["good", "but", "slow"], "maar"),
    ([], "but"),
    (["this", "item", "worked", "great", "but", "it", "broke", "after", "months", "of", "use"], "but"),
]
_SC_HEAD = "def scales(words, word):\n"
_SC_START = _SC_HEAD + "    ...\n"
_SC = (_SC_HEAD
       + "    last = -1\n"
       + "    for i in range(len(words)):\n"
       + "        if words[i] == word:\n"
       + "            last = i\n"
       + "    out = []\n"
       + "    for i in range(len(words)):\n"
       + "        if i < last:\n"
       + "            out.append(0.5)\n"
       + "        else:\n"
       + "            out.append(1.0)\n"
       + "    return out\n")
_SC_FIRST = (_SC_HEAD
             + "    last = -1\n"
             + "    for i in range(len(words)):\n"
             + "        if words[i] == word and last == -1:\n"
             + "            last = i\n"
             + "    out = []\n"
             + "    for i in range(len(words)):\n"
             + "        if i < last:\n"
             + "            out.append(0.5)\n"
             + "        else:\n"
             + "            out.append(1.0)\n"
             + "    return out\n")
_SC_WORD_HALF = _SC.replace("        if i < last:\n", "        if i <= last:\n")
_SC_AFTER = _SC.replace("        if i < last:\n", "        if i > last and last >= 0:\n")
_SC_DROP = _SC.replace("            out.append(0.5)\n", "            out.append(0.0)\n")
_SC_DOUBLED = (_SC.replace("            out.append(0.5)\n", "            out.append(1.0)\n")
               .replace("        else:\n            out.append(1.0)\n", "        else:\n            out.append(2.0)\n"))
_SC_ABSENT = _SC.replace("    last = -1\n", "    last = len(words)\n")
_SC_LAST = _SC_HEAD + "    last = -1\n    for i in range(len(words)):\n        if words[i] == word:\n            last = i\n    return last\n"


def _not_list(got, text) -> bool:
    return got is not None and not isinstance(got, (list, tuple))


# Exercise 3: where is the judgment? Eight held-out sentences with "but" (in Dutch "maar"), all shown, by their place in
# the held-out set (tests/test_nlp_sentiment.py checks the texts and the probabilities in the correct message).

SORT_SENTENCES = {"scene": 371, "broke": 16, "banana": 411, "spoilers": 289, "microphone": 71, "full": 432,
                  "promise": 528, "scared": 294}
_SORT_LABEL = {"scene": 1, "broke": 0, "banana": 0, "spoilers": 1, "microphone": 1, "full": 1, "promise": 0, "scared": 0}
_LABELED = {"en": {1: " (positive)", 0: " (negative)"}, "nl": {1: " (positief)", 0: " (negatief)"}}   # after each sentence



EXERCISES = {
    "majority": FunctionBox(
        name="majority",
        picture=majority_picture,
        function="majority",
        start=_MAJ_START,
        solution=_MAJ,
        visible=MAJ_CASES[:2],
        hidden=MAJ_CASES[2:],
        first=[_DOTS],
        mistakes=[
            Mistake("tie-first", wrong=_MAJ_TIE_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: on a tie, your function gives the label of the first sentence. Which sentence decides a tie in this lesson?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een gelijke stand geeft je functie het label van de eerste zin. Welke zin beslist een gelijke stand in deze les?"),
            Mistake("tie-positive", wrong=_MAJ_TIE_POSITIVE,
                    en="`{call}` gives {got}, and it should give {want}: on a tie, your function always says positive. Which sentence should decide a tie?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een gelijke stand zegt je functie altijd positief. Welke zin moet een gelijke stand beslissen?"),
            Mistake("tie-negative", wrong=_MAJ_TIE_NEGATIVE,
                    en="`{call}` gives {got}, and it should give {want}: on a tie, your function always says negative. Which sentence should decide a tie?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een gelijke stand zegt je functie altijd negatief. Welke zin moet een gelijke stand beslissen?"),
            Mistake("last-only", wrong=_MAJ_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the label of the last sentence, also when most sentences say otherwise. When should the last sentence decide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het label van de laatste zin, ook als de meeste zinnen iets anders zeggen. Wanneer moet de laatste zin beslissen?"),
            Mistake("swapped", wrong=_MAJ_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function says negative when most sentences are positive, and positive when most are negative. Which label goes with a majority of positive sentences?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zegt negatief als de meeste zinnen positief zijn, en positief als de meeste negatief zijn. Welk label hoort bij een meerderheid van positieve zinnen?"),
            Mistake("count", wrong=_MAJ_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of positive sentences. What should your function give back: a count, or the label of the review?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal positieve zinnen. Wat moet je functie teruggeven: een aantal, of het label van de recensie?"),
            Mistake("share", wrong=_MAJ_SHARE,
                    en="`{call}` gives {got}, and it should give {want}: the share of positive sentences. What should your function give back: a share, or the label of the review?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de zinnen dat positief is. Wat moet je functie teruggeven: een deel, of het label van de recensie?"),
            Mistake("not-label", matches=_not_label,
                    en="`{call}` gives {got}, and it should give {want}: the label of the review is 1 or 0. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het label van de recensie is 1 of 0. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the positive and the negative sentences, and give back the label of the review.",
                      "nl": "Vervang de `...`: tel de positieve en de negatieve zinnen, en geef het label van de recensie terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What does your function count, and what does it give back on a tie?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wat telt je functie, en wat geeft je functie terug bij een gelijke stand?"},
            "almost": {"en": "Right for the examples. But `{call}` gives {got}, and it should give {want}. Does your function work for one sentence, for a tie in either order, and for more positive or more negative sentences?",
                       "nl": "Goed voor de voorbeelden. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor één zin, voor een gelijke stand in beide volgordes, en voor meer positieve of meer negatieve zinnen?"},
            "correct": {"en": "Your `majority` is right in all {n} cases. Look at the picture: your majority vote on the 18 reviews of the course, beside the whole review as one text.",
                        "nl": "Je `majority` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw meerderheid op de 18 recensies van de course, naast de hele recensie als één tekst."},
        },
    ),
    "scales": FunctionBox(
        name="scales",
        picture=scales_picture,
        function="scales",
        start=_SC_START,
        solution=_SC,
        visible=SC_CASES[:1],
        hidden=SC_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("first", wrong=_SC_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first `word`. Which one decides where the half weight stops when `word` occurs more than once?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt naar het eerste `word`. Welk `word` bepaalt waar de halve weight ophoudt als het woord vaker voorkomt?"),
            Mistake("word-half", wrong=_SC_WORD_HALF,
                    en="`{call}` gives {got}, and it should give {want}: your function also gives `word` itself half its weight. Which words get `0.5`: those before `word`, or `word` too?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft ook `word` zelf de halve weight. Welke woorden krijgen `0.5`: de woorden vóór `word`, of `word` ook?"),
            Mistake("after", wrong=_SC_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your function gives half weight to the words after `word`. Which part of a sentence carries the judgment?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de woorden na `word` de halve weight. Welk deel van een zin draagt het oordeel?"),
            Mistake("drop", wrong=_SC_DROP,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the words before `word` no weight at all. How much weight should they keep?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de woorden vóór `word` helemaal geen weight. Hoeveel weight moeten de woorden vóór `word` houden?"),
            Mistake("doubled", wrong=_SC_DOUBLED,
                    en="`{call}` gives {got}, and it should give {want}: your function doubles the words from `word` on and leaves the rest. Which words should change, and to what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie verdubbelt de woorden vanaf `word` en laat de rest zoals het was. Welke woorden moeten veranderen, en in wat?"),
            Mistake("absent", wrong=_SC_ABSENT,
                    en="`{call}` gives {got}, and it should give {want}: when `word` is not in the list, your function gives every word half its weight. What should a word get when no `word` comes after it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als `word` niet in de lijst staat, geeft je functie elk woord de halve weight. Wat moet een woord krijgen als er geen `word` na komt?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list with one number for each word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met één getal voor elk woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: find where the last `word` stands, and give back one number for each word: `0.5` before it, `1.0` from there on.",
                      "nl": "Vervang de `...`: zoek waar het laatste `word` staat, en geef één getal voor elk woord terug: `0.5` ervoor, `1.0` vanaf daar."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words does your function give `0.5`, and where does that stop?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke woorden geeft je functie `0.5`, en waar houdt dat op?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work without `word`, with `word` first or last, with `word` twice, for an empty list, and for another `word`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie zonder `word`, met `word` vooraan of achteraan, met twee keer `word`, voor een lege lijst, en voor een ander `word`?"},
            "correct": {"en": "Your `scales` is right in all {n} cases. Look at the picture: the held-out sentences whose label changes with your scales, and the counts with and without.",
                        "nl": "Je `scales` klopt in alle {n} gevallen. Kijk naar de afbeelding: de achtergehouden zinnen die met jouw scales een ander label krijgen, en de aantallen met en zonder."},
        },
    ),
    "where-judgment": Sort(
        name="where-judgment",
        groups={
            "after": {"en": "The judgment comes after *but*", "nl": "Het oordeel komt na *maar*"},
            "before": {"en": "The judgment comes before *but*", "nl": "Het oordeel komt vóór *maar*"},
            "both": {"en": "*But* turns nothing around", "nl": "*Maar* draait niets om"},
        },
        items={
            "scene": {"en": "The best scene in the movie is at the end, but I won't spoil it." + _LABELED["en"][_SORT_LABEL["scene"]],
                      "nl": "De beste scène van de film zit aan het eind, maar ik verklap niets." + _LABELED["nl"][_SORT_LABEL["scene"]]},
            "broke": {"en": "This item worked great, but it broke after 6 months of use." + _LABELED["en"][_SORT_LABEL["broke"]],
                      "nl": "Dit product werkte geweldig, maar na 6 maanden gebruik ging het kapot." + _LABELED["nl"][_SORT_LABEL["broke"]]},
            "banana": {"en": "The ripped banana was not only ripped, but petrified and tasteless." + _LABELED["en"][_SORT_LABEL["banana"]],
                       "nl": "De rijpe banaan was niet alleen rijp, maar ook versteend en smakeloos." + _LABELED["nl"][_SORT_LABEL["banana"]]},
            "spoilers": {"en": "I won't say any more - I don't like spoilers, so I don't want to be one, but I believe this film is worth your time." + _LABELED["en"][_SORT_LABEL["spoilers"]],
                         "nl": "Meer zeg ik niet - ik hou niet van spoilers, dus ik wil er zelf ook geen geven, maar ik vind deze film je tijd waard." + _LABELED["nl"][_SORT_LABEL["spoilers"]]},
            "microphone": {"en": "The microphone also works well, but (according to people I have called) it applifies everything." + _LABELED["en"][_SORT_LABEL["microphone"]],
                           "nl": "De microfoon werkt ook goed, maar (volgens mensen die ik heb gebeld) wordt alles versterkt." + _LABELED["nl"][_SORT_LABEL["microphone"]]},
            "full": {"en": "Restaurant is always full but never a wait." + _LABELED["en"][_SORT_LABEL["full"]],
                     "nl": "Het restaurant zit altijd vol, maar je hoeft nooit te wachten." + _LABELED["nl"][_SORT_LABEL["full"]]},
            "promise": {"en": "This place has a lot of promise but fails to deliver." + _LABELED["en"][_SORT_LABEL["promise"]],
                        "nl": "Deze zaak belooft veel, maar maakt het niet waar." + _LABELED["nl"][_SORT_LABEL["promise"]]},
            "scared": {"en": "I was very uneasy with how bad this movie was, but not scared at all." + _LABELED["en"][_SORT_LABEL["scared"]],
                       "nl": "Ik voelde me erg ongemakkelijk bij hoe slecht deze film was, maar bang was ik helemaal niet." + _LABELED["nl"][_SORT_LABEL["scared"]]},
        },
        solution={"scene": "before", "broke": "after", "banana": "both", "spoilers": "after", "microphone": "before",
                  "full": "both", "promise": "after", "scared": "both"},
        nudge={
            "scene": {"en": "What do the words after *but* say about the film: something good, something bad, or nothing?",
                      "nl": "Wat zeggen de woorden na *maar* over de film: iets goeds, iets slechts of niets?"},
            "broke": {"en": "Which part says how the item is doing in the end?",
                      "nl": "Welk deel zegt hoe het product het uiteindelijk doet?"},
            "banana": {"en": "Is the part after *but* a turn, or more of the same?",
                       "nl": "Is het deel na *maar* een omslag, of meer van hetzelfde?"},
            "spoilers": {"en": "Where does the writer say something about the film itself?",
                         "nl": "Waar zegt de schrijver iets over de film zelf?"},
            "microphone": {"en": "Given the label, is amplifying everything praise here, or a complaint?",
                           "nl": "Is alles versterken hier lof of een klacht, gezien het label?"},
            "full": {"en": "Is a restaurant that is always full good news or bad news here? And never having to wait?",
                     "nl": "Is een restaurant dat altijd vol zit hier goed of slecht nieuws? En nooit hoeven wachten?"},
            "promise": {"en": "A promise is good news. Does the place keep it, and which part says so?",
                        "nl": "Een belofte is goed nieuws. Maakt de zaak de belofte waar, en welk deel zegt dat?"},
            "scared": {"en": "What does the writer think of the film before *but*, and what after it?",
                       "nl": "Wat vindt de schrijver van de film vóór *maar*, en wat erna?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each sentence in a group: where does its judgment, positive or negative, lie?",
                      "nl": "Zet elke zin in een groep: waar staat het oordeel, positief of negatief?"},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The sentence *{item}* belongs in another group.",
                      "nl": "De zin *{item}* hoort in een andere groep."},
            "correct": {"en": "All {n} are right. Half weight before the last *but* helps only where the judgment comes after it: *This place has a lot of promise but fails to deliver.* goes from 0.539 to 0.350, and is now right. Where the judgment comes before it, the scales pull the wrong way: *The best scene in the movie is at the end, but I won't spoil it.* goes from 0.777 to 0.639. Where *but* turns nothing around, a rule that leans on it is right only by luck.",
                        "nl": "Alle {n} kloppen. Halve weight vóór de laatste *maar* helpt alleen als het oordeel erna komt: *Dit product werkte geweldig, maar na 6 maanden gebruik ging het kapot.* gaat van 0,491 naar 0,215: nog duidelijker negatief. Komt het oordeel ervóór, dan trekken de scales de verkeerde kant op: *De beste scène van de film zit aan het eind, maar ik verklap niets.* gaat van 0,869 naar 0,582. Draait *maar* niets om, dan zit een regel die op *maar* leunt er alleen bij toeval goed."},
        },
    ),
}


CASES = [
    ("majority", _MAJ_START, "empty", None),
    ("majority", _MAJ, "correct", None),
    ("majority", _MAJ_HEAD + "    positive = sum(labels)\n    negative = len(labels) - positive\n"
                 "    if positive > negative:\n        return 1\n    if positive < negative:\n        return 0\n"
                 "    return labels[len(labels) - 1]\n", "correct", None),
    ("majority", _MAJ_HEAD + "    positive = labels.count(1)\n    negative = labels.count(0)\n    if positive == negative:\n"
                 "        return labels[-1]\n    if positive > negative:\n        return 1\n    return 0\n", "correct", None),
    ("majority", _MAJ_TIE_FIRST, "wrong", "tie-first"),
    ("majority", _MAJ_TIE_POSITIVE, "wrong", "tie-positive"),
    ("majority", _MAJ_TIE_NEGATIVE, "wrong", "tie-negative"),
    ("majority", _MAJ_EMPTY_LIST, "wrong", "tie-negative"),
    ("majority", _MAJ_LAST, "wrong", "last-only"),
    ("majority", _MAJ_SWAPPED, "wrong", "swapped"),
    ("majority", _MAJ_COUNT, "wrong", "count"),
    ("majority", _MAJ_SHARE, "wrong", "share"),
    ("majority", _MAJ_HEAD + '    return "positive"\n', "wrong", "not-label"),
    ("majority", _MAJ.replace("    return labels[-1]\n", "    return labels[-1] == 1\n"), "wrong", "not-label"),
    ("majority", _MAJ.replace("return 1\n", "print(1)\n").replace("return 0\n", "print(0)\n")
                     .replace("return labels[-1]\n", "print(labels[-1])\n"), "wrong", "prints"),
    ("majority", _MAJ_HEAD + "    if len(labels) == 1:\n        return 0\n" + _MAJ[len(_MAJ_HEAD):], "almost", None),
    ("majority", _MAJ_HEAD + "    ...\n    return labels[-1]\n", "wrong", "dots-left"),
    ("scales", _SC_START, "empty", None),
    ("scales", _SC, "correct", None),
    ("scales", _SC_HEAD + "    last = 0\n    for i in range(len(words)):\n        if words[i] == word:\n            last = i\n"
               "    return [0.5] * last + [1.0] * (len(words) - last)\n", "correct", None),
    ("scales", _SC.replace("out.append(0.5)", "out.append(1 / 2)").replace("out.append(1.0)", "out.append(1)"), "correct",
     None),
    ("scales", _SC_FIRST, "wrong", "first"),
    ("scales", _SC_WORD_HALF, "wrong", "word-half"),
    ("scales", _SC_AFTER, "wrong", "after"),
    ("scales", _SC_DROP, "wrong", "drop"),
    ("scales", _SC_DOUBLED, "wrong", "doubled"),
    ("scales", _SC_ABSENT, "wrong", "absent"),
    ("scales", _SC_LAST, "wrong", "not-list"),
    ("scales", _SC.replace("    return out\n", "    print(out)\n"), "wrong", "prints"),
    ("scales", _SC.replace("words[i] == word", 'words[i] == "but"'), "almost", None),
    ("scales", _SC_HEAD + "    ...\n    return [1.0] * len(words)\n", "wrong", "dots-left"),
    ("where-judgment", {}, "empty", None),
    ("where-judgment", {"scene": "before", "broke": "after", "banana": "both", "spoilers": "after",
                        "microphone": "before", "full": "both", "promise": "after", "scared": "both"}, "correct", None),
    ("where-judgment", {"scene": "before", "broke": "after"}, "wrong", "unfinished"),
    ("where-judgment", {"scene": "after"}, "wrong", "wrong:scene"),
    ("where-judgment", {"broke": "before"}, "wrong", "wrong:broke"),
    ("where-judgment", {"banana": "after"}, "wrong", "wrong:banana"),
    ("where-judgment", {"spoilers": "before"}, "wrong", "wrong:spoilers"),
    ("where-judgment", {"microphone": "after"}, "wrong", "wrong:microphone"),
    ("where-judgment", {"full": "after"}, "wrong", "wrong:full"),
    ("where-judgment", {"promise": "both"}, "wrong", "wrong:promise"),
    ("where-judgment", {"scared": "after"}, "wrong", "wrong:scared"),
]
