# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-logistic-text: the probability that logistic regression gives a sentence from the TF-IDF
values of its words (probability), the slopes of the log loss for every word and for the bias (gradients), and the
share of sentences labeled right (accuracy); and sort words by the weight they get.

The sentences are "Sentiment Labelled Sentences" (Kotzias, Denil, de Freitas and Smyth, KDD 2015; UCI Machine Learning
Repository, CC BY 4.0, https://doi.org/10.24432/C57604), in data/uci-sentiment.tsv.gz, made by
data/tools/make_uci_sentiment.py, held out as in the lesson A document as a vector. Sentences marked as not shown are
counted but never shown. The other texts, the cases of the checks and the pictures are the course's own."""
import contextlib
import copy
import html
import io
import math
import re

import numpy as np

from everything_ai import books, chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, run_code, with_time_limit

# --- the sentences -----------------------------------------------------------------------------------------------------

HELD_OUT = 5   # every fifth sentence is held out, as in the lesson A document as a vector
LABELS = {"en": {1: "positive", 0: "negative"}, "nl": {1: "positief", 0: "negatief"}}
STEPS = 300    # steps of gradient descent in the demo
RATE = 1.0     # its learning rate


def words(text: str) -> list:
    """The words of a text in small letters: runs of letters, with an apostrophe inside a word as in *don't*."""
    return books.words(text.lower())


def _all() -> tuple:
    return books.review_sentences()


def training() -> list:
    """The 2,400 training sentences, as (label, text, shown): all but every fifth sentence of the set."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all()) if k % HELD_OUT != HELD_OUT - 1]


def held_out() -> list:
    """The 600 held-out sentences, every fifth sentence of the set, as (label, text, shown)."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all()) if k % HELD_OUT == HELD_OUT - 1]


def find(source: str, line: int) -> tuple:
    """(label, text, shown) of the sentence on `line` of the file of `source` (amazon, imdb or yelp)."""
    for s, n, label, shown, text in _all():
        if s == source and n == line:
            return label, text, shown
    raise KeyError((source, line))


def _df() -> dict:
    """The document frequency of every word of the training sentences: in how many of the 2,400 it occurs."""
    def make():
        df = {}
        for _, text, _ in training():
            for w in set(words(text)):
                df[w] = df.get(w, 0) + 1
        return df
    return kept("nlp-logistic-text:df", make)


def document_frequencies() -> dict:
    """The document frequencies of the 4,518 words of the training sentences, as a new dict."""
    return dict(_df())


def vocabulary() -> list:
    """The words of the training sentences in alphabetical order, as a new list."""
    return sorted(_df())


def row(text: str) -> dict:
    """The TF-IDF row of a text: a dict from each of its words to its count times log10(2,400 / df), the document
    frequency over the training sentences, in the order the words first occur. A word that no training sentence has
    is left out."""
    df, n = _df(), len(training())
    counts = {}
    for w in words(text):
        if w in df:
            counts[w] = counts.get(w, 0) + 1
    return {w: c * math.log10(n / df[w]) for w, c in counts.items()}


def _rows() -> tuple:
    return kept("nlp-logistic-text:rows", lambda: ([row(t) for _, t, _ in training()],
                                                   [row(t) for _, t, _ in held_out()]))


def training_examples() -> list:
    """The 2,400 training sentences as (row, label), a new list with new rows each time."""
    return [(dict(r), label) for r, (label, _, _) in zip(_rows()[0], training())]


def held_out_examples() -> list:
    """The 600 held-out sentences as (row, label), a new list with new rows each time."""
    return [(dict(r), label) for r, (label, _, _) in zip(_rows()[1], held_out())]


# --- the course's solutions --------------------------------------------------------------------------------------------

def probability(row, weights, bias):
    """The course's solution of Exercise 1: the sigmoid of the bias plus, for each word of the row, its TF-IDF value
    times its weight."""
    z = bias
    for word in row:
        z = z + row[word] * weights[word]
    return 1 / (1 + np.exp(-z))


def gradients(examples, weights, bias):
    """The course's solution of Exercise 2: the slope of the mean log loss for every weight and for the bias. Each
    example adds its error, the probability minus the label, times the value of each of its words, divided by n."""
    n = len(examples)
    grad = {word: 0.0 for word in weights}
    grad_bias = 0.0
    for row, label in examples:
        error = probability(row, weights, bias) - label
        for word in row:
            grad[word] = grad[word] + error * row[word] / n
        grad_bias = grad_bias + error / n
    return grad, grad_bias


def accuracy(examples, weights, bias):
    """The course's solution of Exercise 3: the share of the examples whose label is 1 exactly when the probability is
    0.5 or more."""
    right = 0
    for row, label in examples:
        guess = 0
        if probability(row, weights, bias) >= 0.5:
            guess = 1
        if guess == label:
            right = right + 1
    return right / len(examples)


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _p(r: dict, weights: dict, bias: float) -> float:
    return _sigmoid(bias + sum(x * weights.get(w, 0.0) for w, x in r.items()))


def _loss(examples: list, weights: dict, bias: float) -> float:
    """The mean log loss of the examples."""
    total = 0.0
    for r, label in examples:
        p = min(max(_p(r, weights, bias), 1e-12), 1 - 1e-12)
        total -= math.log(p) if label == 1 else math.log(1 - p)
    return total / len(examples)


# --- training ----------------------------------------------------------------------------------------------------------

LOOKS = [1, 2, 5] + list(range(10, STEPS + 1, 10))   # the steps the rewind stops at


def _course() -> dict:
    """The course's run, with NumPy, from weight 0 for every word and bias 0: STEPS steps with learning rate RATE, the
    same steps as gradients() with the course's probability. {"curve": [(step, loss)], "looks": {step: (weights,
    bias)}, "weights", "bias"}; the loss is that of the weights after the step."""
    def make():
        vocab = vocabulary()
        cols = {w: k for k, w in enumerate(vocab)}
        rows = _rows()[0]
        x = np.zeros((len(rows), len(vocab)))
        for i, r in enumerate(rows):
            for w, v in r.items():
                x[i, cols[w]] = v
        y = np.array([label for label, _, _ in training()], dtype=float)
        w, b = np.zeros(len(vocab)), 0.0
        curve, looks = [], {}
        for step in range(1, STEPS + 1):
            p = 1 / (1 + np.exp(-(x @ w + b)))
            w = w - RATE * ((p - y) @ x / len(rows))
            b = b - RATE * float((p - y).mean())
            p = np.clip(1 / (1 + np.exp(-(x @ w + b))), 1e-12, 1 - 1e-12)
            curve.append((step, float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))))
            if step in LOOKS:
                looks[step] = ({v: float(w[k]) for k, v in enumerate(vocab)}, b)
        return {"curve": curve, "looks": looks, "weights": looks[STEPS][0], "bias": looks[STEPS][1]}
    return kept("nlp-logistic-text:course", make)


def weights() -> dict:
    """The weights after the course's STEPS steps, as a new dict from word to weight."""
    return dict(_course()["weights"])


def bias() -> float:
    """The bias after the course's STEPS steps."""
    return _course()["bias"]


def weight(word: str) -> float:
    return _course()["weights"][word]


def parts(r: dict) -> list:
    """(word, value, weight) for each word of a row, with the weights of the course's run."""
    return [(w, v, weight(w)) for w, v in r.items()]


def held_right(ws: dict | None = None, b: float | None = None) -> list:
    """For each held-out sentence, whether the weights (the course's by default) give it its own label."""
    ws = _course()["weights"] if ws is None else ws
    b = bias() if b is None else b
    return [(_p(r, ws, b) >= 0.5) == (label == 1) for r, (label, _, _) in zip(_rows()[1], held_out())]


def training_right() -> int:
    return sum((_p(r, _course()["weights"], bias()) >= 0.5) == (label == 1)
               for r, (label, _, _) in zip(_rows()[0], training()))


def strongest(ws: dict, k: int = 6) -> tuple:
    """The k words with the highest weights, highest first, and the k with the lowest, lowest first."""
    ranked = sorted(ws.items(), key=lambda t: (t[1], t[0]))
    return [w for w, _ in ranked[::-1][:k]], [w for w, _ in ranked[:k]]


def _class_counts() -> dict:
    def make():
        out = {}
        for label, text, _ in training():
            for w in set(words(text)):
                pos, neg = out.get(w, (0, 0))
                out[w] = (pos + 1, neg) if label == 1 else (pos, neg + 1)
        return out
    return kept("nlp-logistic-text:class-counts", make)


def class_counts(word: str) -> tuple:
    """(positive, negative): in how many positive and how many negative training sentences the word occurs."""
    return _class_counts().get(word, (0, 0))


def naive_bayes() -> list:
    """For each held-out sentence, whether naive Bayes on the counts, with Laplace smoothing, gives it its own label:
    the same method and the same 497 as in the lesson A document as a vector."""
    def make():
        vocab = vocabulary()
        cols = {w: k for k, w in enumerate(vocab)}
        n = len(vocab)
        labels = [label for label, _, _ in training()]
        counts = {0: [0] * n, 1: [0] * n}
        for label, (_, text, _) in zip(labels, training()):
            for w in words(text):
                counts[label][cols[w]] += 1
        total = {c: sum(counts[c]) for c in counts}
        out = []
        for truth, text, _ in held_out():
            row_ = {}
            for w in words(text):
                if w in cols:
                    row_[cols[w]] = row_.get(cols[w], 0) + 1
            scores = {}
            for c in (0, 1):
                s = math.log(labels.count(c) / len(labels))
                for k, x in sorted(row_.items()):
                    s += x * math.log((counts[c][k] + 1) / (total[c] + n))
                scores[c] = s
            out.append(max((1, 0), key=lambda c: scores[c]) == truth)
        return out
    return list(kept("nlp-logistic-text:naive-bayes", make))


NEAREST_RIGHT = 397   # the most similar training sentence, in A document as a vector: 66.2% of 600


def mine(box_text, correct: bool, name: str, names: dict | None = None):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text, dict(names or {}))
    fn = run.names.get(name) if run.error is None else None
    return fn if callable(fn) else None


def train(grads=None, show=None, steps: int = STEPS) -> dict:
    """Gradient descent on the training sentences with dictionaries, from weight 0 for every word and bias 0: each step
    takes the slopes from `grads` (the learner's gradients, or the course's) and moves each weight by minus RATE times
    its slope. `show(step, loss, weights, bias)` hears every step. Gives back {"curve", "looks", "weights", "bias"}
    as _course() does; a ValueError when `grads` gives something else than a dict with a number for every word and a
    number for the bias."""
    grads = grads or gradients
    examples = training_examples()
    ws = {w: 0.0 for w in vocabulary()}
    b = 0.0
    curve, looks = [], {}
    for step in range(1, steps + 1):
        out = grads([(dict(r), label) for r, label in examples], dict(ws), b)
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict) and _number(out[1])):
            raise ValueError("gradients")
        g, gb = out
        for w in ws:
            v = g.get(w)
            if not _number(v):
                raise ValueError("gradients")
            ws[w] = ws[w] - RATE * float(v)
        b = b - RATE * float(gb)
        loss = _loss(examples, ws, b)
        curve.append((step, loss))
        if step in LOOKS:
            looks[step] = (dict(ws), b)
        if show is not None:
            show(step, loss, ws, b)
    return {"curve": curve, "looks": looks, "weights": dict(ws), "bias": b}


DEMO_LIMIT = 120.0   # seconds the demo's training gets


def demo(grads=None, show=None):
    """The demo's run, within DEMO_LIMIT seconds: train(grads, show), or "slow" when it takes longer, or "gradients"
    when the learner's function stops with an error or gives something else than slopes."""
    try:
        return with_time_limit(lambda: train(grads, show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except Exception:   # noqa: BLE001   the learner's function; the demo says what went wrong
        return "gradients"


# --- the small examples of the exercises -------------------------------------------------------------------------------

SMALL_WEIGHTS = {"great": 2.21, "good": 1.33, "food": -0.16, "not": -1.78, "bad": -1.49}
SMALL_BIAS = -0.18
START_WEIGHTS = {w: 0.0 for w in SMALL_WEIGHTS}
SMALL_EXAMPLES = [({"great": 1.15, "food": 1.363}, 1), ({"not": 1.04, "good": 1.13}, 0), ({"bad": 1.511, "food": 1.363}, 0)]
MORE_EXAMPLES = [({"good": 1.13, "food": 1.363}, 1), ({"not": 1.04, "bad": 1.511}, 1), ({"great": 1.15}, 0),
                 ({"bad": 1.511}, 0)]
EXAMPLE_TEXT = "I will not be eating there again."   # yelp, line 610, held out, labeled negative
IDEA_SENTENCE = ("yelp", 310)   # "I did not expect this to be so good!", held out, labeled positive
WATCHED = [("yelp", 240), ("yelp", 310), ("imdb", 670)]   # three held-out sentences the demo watches


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
_CHIP_PARAM = _CHIP + ";background:var(--ea-param-wash);color:var(--ea-param-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
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
    """A whole number with a separator for thousands: 4,518 in English, 4.518 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.52 in English, 0,52 in Dutch; a minus sign for a negative number."""
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _pct(x: float, lang: str, digits: int = 1) -> str:
    return _dec(x, lang, digits) + "%"


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 8.0) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _signed(x: float, most: float, color: str = "var(--ea-param)", width: float = 5.0) -> str:
    """A bar to the left of a middle line for a negative number and to the right for a positive one."""
    left = _bar(-x, most, color, width) if x < 0 else ""
    right = _bar(x, most, color, width) if x > 0 else ""
    return (f'<span style="display:inline-flex;align-items:center;white-space:nowrap">'
            f'<span style="display:inline-flex;justify-content:flex-end;width:{width:.2f}rem">{left}</span>'
            f'<span style="display:inline-block;width:1px;height:0.9rem;background:var(--ea-ink-3)"></span>'
            f'<span style="display:inline-flex;width:{width:.2f}rem">{right}</span></span>')


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


def _sentence(text: str, shown: int, lang: str, longest: int = 64) -> str:
    """A sentence of the set, cut off after `longest` characters, or a placeholder for one the course does not show."""
    if not shown:
        return f'<span style="{_TEXT};color:var(--ea-ink-3)">{html.escape(_HIDDEN[lang])}</span>'
    cut = text if len(text) <= longest else text[:longest - 1] + "…"
    return f'<span style="{_TEXT}">{html.escape(cut)}</span>'


def _plain(text: str, shown: int, lang: str) -> str:
    return text if shown else _HIDDEN[lang]


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
    "en": {"stopped": "Your function stopped with an error on the review sentences, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the review sentences, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op de zinnen uit reviews, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de zinnen uit reviews, dus de afbeelding blijft leeg."},
}


# --- the idea's picture: what each word of one sentence adds ------------------------------------------------------------

_PARTS_TEXT = {
    "en": {"title": "What each word adds: {text}",
           "head": ["Word", "TF-IDF value x", "Weight w", "x × w", ""],
           "bias": "bias b",
           "sum": "z = b + the sum of x × w",
           "p": "p = σ(z), the probability of positive",
           "label": "Label: {label}. Logistic regression says {said}.",
           "note": "Gray: the words of the sentence. Orange: the weights after training, and what each word adds to "
                   "z, to the left when it pushes towards negative and to the right when it pushes towards positive. "
                   "Blue: the probability that follows.",
           "purple": "Purple: a probability that gives the sentence a label other than its own.",
           "alt": "{title}. {rows}. Bias {bias}. z = {z}, p = {p}. {label} {note}"},
    "nl": {"title": "Wat elk woord bijdraagt: {text}",
           "head": ["Woord", "TF-IDF-waarde x", "Weight w", "x × w", ""],
           "bias": "bias b",
           "sum": "z = b + de som van x × w",
           "p": "p = σ(z), de kans op positief",
           "label": "Label: {label}. Logistische regressie zegt {said}.",
           "note": "Grijs: de woorden van de zin. Oranje: de weights na de training, en wat elk woord bijdraagt aan "
                   "z, naar links als het woord z kleiner maakt en naar rechts als het woord z groter maakt. Blauw: de "
                   "kans die daaruit volgt.",
           "purple": "Paars: een kans die de zin een ander label geeft dan het eigen label.",
           "alt": "{title}. {rows}. Bias {bias}. z = {z}, p = {p}. {label} {note}"},
}


def parts_picture(text: str, label: int | None, lang: str, title: str | None = None, ws: dict | None = None,
                  b: float | None = None, after: str = "") -> str:
    """For one text: each word with its TF-IDF value, its weight and their product as a bar, then the bias, z and p;
    with the course's weights unless `ws` and `b` are given. `label` is the sentence's own label, or None; `after` is a
    line under the note."""
    w = _PARTS_TEXT[lang]
    ws = _course()["weights"] if ws is None else ws
    b = bias() if b is None else b
    r = row(text)
    items = [(word, v, ws.get(word, 0.0)) for word, v in r.items()]
    most = max([abs(v * x) for _, v, x in items] + [abs(b), 0.5])
    rows_, alts = [], []
    for word, v, x in items:
        rows_.append([_chip(word), _dec(v, lang, 3), _chip(_dec(x, lang, 3), _CHIP_PARAM),
                      _dec(v * x, lang, 3), _signed(v * x, most)])
        alts.append(f"{word} {_dec(v, lang, 3)} × {_dec(x, lang, 3)} = {_dec(v * x, lang, 3)}")
    z = b + sum(v * x for _, v, x in items)
    p = _sigmoid(z)
    rows_.append([html.escape(w["bias"]), "", _chip(_dec(b, lang, 3), _CHIP_PARAM), _dec(b, lang, 3), _signed(b, most)])
    rows_.append([html.escape(w["sum"]), "", "", f"<b>{_dec(z, lang, 3)}</b>", _signed(z, max(most, abs(z)))])
    said_label = 1 if p >= 0.5 else 0
    wrong = label is not None and said_label != label
    p_chip = _chip(_dec(p, lang, 3), _CHIP_PURPLE if wrong else _CHIP_PRED)
    rows_.append([html.escape(w["p"]), "", "", p_chip, _bar(p, 1.0, "var(--ea-pred)", 5.0)])
    body = _table(rows_, w["head"], numbers=(1, 2, 3))
    said = LABELS[lang][said_label]
    label_text = w["label"].format(label=LABELS[lang][label], said=said) if label is not None else ""
    if label_text:
        body += _label(label_text)
    note = w["note"] + (" " + w["purple"] if label is not None else "")
    body += _note(note) + (_note(after) if after else "")
    head = title or w["title"].format(text=text)
    return yours.figure(head, body, alt=w["alt"].format(title=head, rows="; ".join(alts), bias=_dec(b, lang, 3),
                                                        z=_dec(z, lang, 3), p=_dec(p, lang, 3), label=label_text,
                                                        note=note) + (" " + after if after else ""))


def idea_picture(lang: str) -> str:
    label, text, _ = find(*IDEA_SENTENCE)
    return parts_picture(text, label, lang)


# --- the picture under exercise 1: your probabilities for some held-out sentences --------------------------------------

PROBABILITY_SENTENCES = [("yelp", 240), ("imdb", 410), ("amazon", 805), ("yelp", 310), ("imdb", 850), ("yelp", 610),
                         ("amazon", 845)]

_PROB_TEXT = {
    "en": {"title": "Your probabilities for seven held-out sentences",
           "head": ["Sentence", "Label", "Your p", ""],
           "note": "With the weights after training. Gray: the sentence and its own label. Blue: the probability of "
                   "positive that your function gives, as a bar from 0 to 1; 0.5 or more says positive. Purple: a "
                   "probability that should be another one; dashed: the right one.",
           "odd": "Your function gave something other than a number from 0 to 1, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je kansen voor zeven achtergehouden zinnen",
           "head": ["Zin", "Label", "Je p", ""],
           "note": "Met de weights na de training. Grijs: de zin en het eigen label. Blauw: de kans op positief die je "
                   "functie geeft, als balk van 0 tot 1; 0,5 of meer zegt positief. Paars: een kans die niet klopt; met "
                   "een gestreepte rand: de goede kans.",
           "odd": "Je functie gaf iets anders dan een getal van 0 tot 1, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def probability_picture(seen) -> str:
    lang = seen.lang
    w = _PROB_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    found = [find(s, n) for s, n in PROBABILITY_SENTENCES]
    ws, b = _course()["weights"], bias()

    def make():
        return [fn(row(t), dict(ws), b) for _, t, _ in found]
    got, said = _run_or_say(make, w, lambda g: all(_number(v) and 0 <= float(v) <= 1 for v in g))
    if said:
        return said
    rows_, alts = [], []
    for (label, text, shown), mine_ in zip(found, got):
        want = _p(row(text), ws, b)
        mine_ = float(mine_)
        if math.isclose(mine_, want, rel_tol=1e-6, abs_tol=1e-9):
            shown_p = _chip(_dec(mine_, lang, 3), _CHIP_PRED)
        else:
            shown_p = _chip(_dec(mine_, lang, 3), _CHIP_PURPLE) + _chip(_dec(want, lang, 3), _CHIP_MISSING)
        rows_.append([_sentence(text, shown, lang), _chip(LABELS[lang][label]), shown_p,
                      _bar(mine_, 1.0, "var(--ea-pred)", 5.0)])
        alts.append(f"{_plain(text, shown, lang)} ({LABELS[lang][label]}): {_dec(mine_, lang, 3)}")
    body = _table(rows_, w["head"], numbers=(2,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 2: your slopes at the start ------------------------------------------------------------

_GRAD_TEXT = {
    "en": {"title": "Your slopes at the start, over the 2,400 training sentences",
           "head": ["Word", "Positive", "Negative", "Your slope", ""],
           "down": "The six words with the most negative slope, whose weight the first step raises most:",
           "up": "The six words with the most positive slope, whose weight the first step lowers most:",
           "bias": "Your slope for the bias: {g}. 1,209 of the 2,400 training sentences are positive, a little more "
                   "than half, so at the start the errors of the bias almost cancel out.",
           "note": "With every weight and the bias at 0, so that every sentence gets p = 0.5. Gray: in how many positive "
                   "and negative training sentences the word occurs. Orange: the slope your function gives, to the "
                   "left when it is negative. Purple: a slope that should be another one; dashed: the right one.",
           "odd": "Your function gave something other than a dictionary with a number for every word and a number for "
                  "the bias, so the picture stays empty.",
           "alt": "{title}. {down} {rows_down}. {up} {rows_up}. {bias} {note}"},
    "nl": {"title": "Je hellingen aan het begin, over de 2.400 trainingszinnen",
           "head": ["Woord", "Positief", "Negatief", "Je helling", ""],
           "down": "De zes woorden met de meest negatieve helling, waarvan de eerste stap de weight het meest verhoogt:",
           "up": "De zes woorden met de meest positieve helling, waarvan de eerste stap de weight het meest verlaagt:",
           "bias": "Je helling voor de bias: {g}. 1.209 van de 2.400 trainingszinnen zijn positief, iets meer dan de "
                   "helft, dus aan het begin vallen de fouten voor de bias bijna tegen elkaar weg.",
           "note": "Met elke weight en de bias op 0, zodat elke zin p = 0,5 krijgt. Grijs: in hoeveel positieve en "
                   "negatieve trainingszinnen het woord voorkomt. Oranje: de helling die je functie geeft, naar links "
                   "als de helling negatief is. Paars: een helling die niet klopt; met een gestreepte rand: de goede "
                   "helling.",
           "odd": "Je functie gaf iets anders dan een dictionary met een getal voor elk woord en een getal voor de "
                  "bias, dus de afbeelding blijft leeg.",
           "alt": "{title}. {down} {rows_down}. {up} {rows_up}. {bias} {note}"},
}


def _start_slopes() -> tuple:
    return kept("nlp-logistic-text:start", lambda: gradients(training_examples(), {w: 0.0 for w in vocabulary()}, 0.0))


def gradients_picture(seen) -> str:
    lang = seen.lang
    w = _GRAD_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    vocab = vocabulary()

    def ok(g):
        return (isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], dict) and _number(g[1])
                and all(_number(g[0].get(word)) for word in vocab))
    got, said = _run_or_say(lambda: fn(training_examples(), {word: 0.0 for word in vocab}, 0.0), w, ok)
    if said:
        return said
    right, right_b = _start_slopes()
    ranked = sorted(right.items(), key=lambda t: (t[1], t[0]))
    down, up = [word for word, _ in ranked[:6]], [word for word, _ in ranked[::-1][:6]]
    most = max(abs(right[x]) for x in down + up)

    def table(chosen):
        rows_, alts = [], []
        for word in chosen:
            mine_, want = float(got[0][word]), right[word]
            pos, neg = class_counts(word)
            if math.isclose(mine_, want, rel_tol=1e-6, abs_tol=1e-9):
                shown = _chip(_dec(mine_, lang, 4), _CHIP_PARAM)
            else:
                shown = _chip(_dec(mine_, lang, 4), _CHIP_PURPLE) + _chip(_dec(want, lang, 4), _CHIP_MISSING)
            rows_.append([_chip(word), _num(pos, lang), _num(neg, lang), shown, _signed(mine_, most)])
            alts.append(f"{word} {_num(pos, lang)}/{_num(neg, lang)}: {_dec(mine_, lang, 4)}")
        return _table(rows_, w["head"], numbers=(1, 2, 3)), "; ".join(alts)
    t_down, a_down = table(down)
    t_up, a_up = table(up)
    g_b = float(got[1])
    b_text = (_chip(_dec(g_b, lang, 4), _CHIP_PARAM) if math.isclose(g_b, right_b, abs_tol=1e-9)
              else _chip(_dec(g_b, lang, 4), _CHIP_PURPLE) + _chip(_dec(right_b, lang, 4), _CHIP_MISSING))
    bias_line = w["bias"].format(g=_dec(g_b, lang, 4))
    bias_html = f'<div style="{_LABEL}">{html.escape(w["bias"].split("{g}")[0])}{b_text}{html.escape(w["bias"].split("{g}")[1])}</div>'
    body = _label(w["down"]) + t_down + _label(w["up"]) + t_up + bias_html + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], down=w["down"], rows_down=a_down,
                                                              up=w["up"], rows_up=a_up, bias=bias_line, note=w["note"]))


# --- the demo: training with your gradients, and what the model makes meanwhile ---------------------------------------

_LIVE_TEXT = {
    "en": {"title": "Training on the 2,400 sentences", "x": "step", "y": "log loss",
           "alt": "The mean log loss of the 2,400 training sentences after each step of gradient descent.",
           "strong": "The strongest words now", "towards_pos": "Towards positive", "towards_neg": "Towards negative",
           "watched": "Three held-out sentences now", "sentence": "Sentence", "label": "Label", "p": "p",
           "after": "The strongest words after step {s}", "watched_after": "Three held-out sentences after step {s}",
           "progress": "**Training** · step {s} of {n} · log loss {loss}",
           "note": "Gray: the words, the sentences and their own labels. Orange: the weights. Blue: the probability "
                   "of positive. Purple: a probability that gives a sentence a label other than its own."},
    "nl": {"title": "Training op de 2.400 zinnen", "x": "stap", "y": "log loss",
           "alt": "De gemiddelde log loss van de 2.400 trainingszinnen na elke stap gradient descent.",
           "strong": "De sterkste woorden nu", "towards_pos": "Richting positief", "towards_neg": "Richting negatief",
           "watched": "Drie achtergehouden zinnen nu", "sentence": "Zin", "label": "Label", "p": "p",
           "after": "De sterkste woorden na stap {s}", "watched_after": "Drie achtergehouden zinnen na stap {s}",
           "progress": "**Training** · stap {s} van {n} · log loss {loss}",
           "note": "Grijs: de woorden, de zinnen en de eigen labels. Oranje: de weights. Blauw: de kans op positief. "
                   "Paars: een kans die een zin een ander label geeft dan het eigen label."},
}


def live_words(lang: str) -> dict:
    return dict(_LIVE_TEXT[lang])


def _strong_table(ws: dict, lang: str, title: str) -> str:
    w = _LIVE_TEXT[lang]
    up, down = strongest(ws, 6)
    most = max([abs(ws[x]) for x in up + down] + [1e-9])
    rows_ = []
    for a, d in zip(up, down):
        rows_.append([_chip(a), _chip(_dec(ws[a], lang, 2), _CHIP_PARAM), _bar(ws[a], most, "var(--ea-param)", 4.0),
                      _chip(d), _chip(_dec(ws[d], lang, 2), _CHIP_PARAM), _bar(-ws[d], most, "var(--ea-param)", 4.0)])
    return (_label(title) + _table(rows_, [w["towards_pos"], "", "", w["towards_neg"], "", ""], numbers=(1, 4)))


def _watched_table(ws: dict, b: float, lang: str, title: str) -> str:
    w = _LIVE_TEXT[lang]
    rows_ = []
    for s, n in WATCHED:
        label, text, shown = find(s, n)
        p = _p(row(text), ws, b)
        wrong = (p >= 0.5) != (label == 1)
        rows_.append([_sentence(text, shown, lang, 46), _chip(LABELS[lang][label]),
                      _chip(_dec(p, lang, 2), _CHIP_PURPLE if wrong else _CHIP_PRED), _bar(p, 1.0, "var(--ea-pred)", 3.5)])
    return _label(title) + _table(rows_, [w["sentence"], w["label"], w["p"], ""], numbers=(2,))


def now_parts(ws: dict, b: float, lang: str, step: int | None = None) -> list:
    """What the demo shows beside its chart: the six strongest words each way, and three held-out sentences with
    their probability now (or after `step`, for the rewind)."""
    w = _LIVE_TEXT[lang]
    strong = w["strong"] if step is None else w["after"].format(s=step)
    watched = w["watched"] if step is None else w["watched_after"].format(s=step)
    return [_strong_table(ws, lang, strong), _watched_table(ws, b, lang, watched) + _note(w["note"])]


def curve_picture(curve: list, lang: str, at: int | None = None) -> str:
    """The log loss after each step of the run; with `at`, a line at that step, for the rewind."""
    w = _LIVE_TEXT[lang]
    top = max(0.7, max(y for _, y in curve))
    p = chart.Panel((0, STEPS), (0, top), title=w["title"], xlabel=w["x"], ylabel=w["y"])
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    p.line([x for x, _ in curve], [y for _, y in curve], "loss", 2.2)
    svg = chart.figure([p], alt=w["alt"], width=live.CHART_WIDTH, height=240)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_frame(run: dict, step: int, lang: str) -> str:
    """The rewind at a step: the curve with a line at that step, and beside it the strongest words and the three
    sentences after it."""
    ws, b = run["looks"][step]
    return live.rewound(curve_picture(run["curve"], lang, at=step), now_parts(ws, b, lang, step))


def rewind_right(run: dict, step: int) -> int:
    ws, b = run["looks"][step]
    return sum(held_right(ws, b))


# --- the second demo: a sentence of your own ---------------------------------------------------------------------------

DEMO_TEXT = "The food was not great, but the staff were lovely."
DEMO_LONGEST = 300   # characters of the text that the program looks at

_TRY_TEXT = {
    "en": {"title": "What logistic regression makes of your sentence",
           "empty": "Type a sentence with at least one word that the training sentences have.",
           "outside": "Left out, since no training sentence has them: {words}."},
    "nl": {"title": "Wat logistische regressie van je zin maakt",
           "empty": "Typ een zin met minstens één woord dat in de trainingszinnen staat.",
           "outside": "Weggelaten, omdat geen enkele trainingszin deze woorden heeft: {words}."},
}


def try_picture(text, lang: str) -> str:
    w = _TRY_TEXT[lang]
    text = str(text or "")[:DEMO_LONGEST]
    r = row(text)
    if not r:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    outside = []
    for x in words(text):
        if x not in _df() and x not in outside:
            outside.append(x)
    line = w["outside"].format(words=", ".join(outside[:12]) + (" …" if len(outside) > 12 else "")) if outside else ""
    return parts_picture(text, None, lang, title=w["title"], after=line)


# --- the picture under exercise 3: your accuracy, against naive Bayes ---------------------------------------------------

_ACC_TEXT = {
    "en": {"title": "Your accuracy on the 600 held-out sentences",
           "yours": "Your accuracy for logistic regression: {a}, that is {k} of 600.",
           "methods": ["Logistic regression on TF-IDF", "Naive Bayes on the counts", "The most similar sentence"],
           "head": ["Method", "Right", "Share", ""],
           "grid": "Each square one held-out sentence, in order; purple where the method gets it wrong:",
           "both": "Both get {both} sentences wrong; only logistic regression {lr}, only naive Bayes {nb}.",
           "note": "Gray: your accuracy, and the course's numbers for the three methods, the last two from the "
                   "lesson A document as a vector. Purple: your accuracy when it is not the right one, which then "
                   "stands dashed beside it, and a sentence a method labels wrong.",
           "odd": "Your function gave something other than a number from 0 to 1, so the picture stays empty.",
           "alt": "{title}. {yours} {rows}. {both} {note}"},
    "nl": {"title": "Je accuracy op de 600 achtergehouden zinnen",
           "yours": "Je accuracy voor logistische regressie: {a}, dus {k} van de 600.",
           "methods": ["Logistische regressie op TF-IDF", "Naive Bayes op de aantallen", "De meest gelijkende zin"],
           "head": ["Methode", "Goed", "Aandeel", ""],
           "grid": "Elk vakje één achtergehouden zin, op volgorde; paars waar de methode de zin fout heeft:",
           "both": "Beide hebben {both} zinnen fout; alleen logistische regressie {lr}, alleen naive Bayes {nb}.",
           "note": "Grijs: je accuracy, en de getallen van de course voor de drie methoden, de laatste twee uit de "
                   "les Een document als vector. Paars: je accuracy als je accuracy niet klopt, met de goede accuracy "
                   "ernaast met een gestreepte rand, en een zin die een methode fout labelt.",
           "odd": "Je functie gaf iets anders dan een getal van 0 tot 1, dus de afbeelding blijft leeg.",
           "alt": "{title}. {yours} {rows}. {both} {note}"},
}


def _grid(right: list) -> str:
    cells = "".join(f'<span style="display:inline-block;width:6px;height:6px;margin:0 1px 1px 0;border-radius:1px;'
                    f'background:{"var(--ea-line)" if ok else "var(--ea-loss)"}"></span>' for ok in right)
    return f'<div style="max-width:100%;line-height:0;margin:0.25rem 0 0.35rem">{cells}</div>'


def accuracy_picture(seen) -> str:
    lang = seen.lang
    w = _ACC_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    ws, b = _course()["weights"], bias()
    got, said = _run_or_say(lambda: fn(held_out_examples(), dict(ws), b), w,
                            lambda g: _number(g) and 0 <= float(g) <= 1)
    if said:
        return said
    lr, nb = held_right(), naive_bayes()
    want = sum(lr) / len(lr)
    mine_ = float(got)
    a = _chip(_dec(mine_, lang, 4))
    if not math.isclose(mine_, want, rel_tol=1e-6, abs_tol=1e-9):
        a = _chip(_dec(mine_, lang, 4), _CHIP_PURPLE) + _chip(_dec(want, lang, 4), _CHIP_MISSING)
    yours_text = w["yours"].format(a=_dec(mine_, lang, 4), k=_num(round(mine_ * 600), lang))
    head, tail = w["yours"].split("{a}")
    yours_html = (f'<div style="{_LABEL}">{html.escape(head)}{a}'
                  f'{html.escape(tail.format(k=_num(round(mine_ * 600), lang)))}</div>')
    counts = [sum(lr), sum(nb), NEAREST_RIGHT]
    rows_ = [[html.escape(name), _num(k, lang), _pct(100 * k / 600, lang), _bar(k, 600, "var(--ea-data)", 8.0)]
             for name, k in zip(w["methods"], counts)]
    both = sum(1 for x, y in zip(lr, nb) if not x and not y)
    only_lr = sum(1 for x, y in zip(lr, nb) if not x and y)
    only_nb = sum(1 for x, y in zip(lr, nb) if x and not y)
    both_text = w["both"].format(both=_num(both, lang), lr=_num(only_lr, lang), nb=_num(only_nb, lang))
    body = (yours_html + _table(rows_, w["head"], numbers=(1, 2)) + _label(w["grid"])
            + _label(w["methods"][0]) + _grid(lr) + _label(w["methods"][1]) + _grid(nb)
            + _label(both_text) + _note(w["note"]))
    alt_rows = "; ".join(f"{name}: {_num(k, lang)}" for name, k in zip(w["methods"], counts))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], yours=yours_text, rows=alt_rows,
                                                              both=both_text, note=w["note"]))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["probability", "gradients", "accuracy", "what-weight"]

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

PROBES = ["great", "good", "not", "bad", "the", "food"]   # words a long dictionary is shown by


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, numbers rounded to six decimals."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value(p) for p in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (float, np.floating)):
        return repr(round(float(v), 6) + 0.0)
    if isinstance(v, np.integer):
        return repr(int(v))
    return repr(v)


def _long(v: dict) -> dict:
    shown = [k for k in PROBES if k in v] or sorted(v, key=str)[:4]
    listed = ", ".join(f"{_value(k)}: {_value(v[k])}" for k in shown)
    return {"en": f"a dictionary of {len(v):,} words, with {listed}",
            "nl": f"een dictionary van {_num(len(v), 'nl')} woorden, met {listed}"}


def _shown(v):
    """A value for a message: a long dictionary as its length and a few of its words, in each language; a pair of a
    dictionary and a number with the dictionary shown so."""
    if isinstance(v, dict) and len(v) > 12:
        return _long(v)
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], dict) and len(v[0]) > 12:
        d = _long(v[0])
        return {lang: f"({d[lang]}, {_value(v[1])})" for lang in ("en", "nl")}
    return _value(v)


class Named:
    """A value of the check's own that a message shows under a name, such as the training sentences of the lesson."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Examples(Named):
    """A list of (row, label), copied fast: a new list with a new dict for each row."""

    def __deepcopy__(self, memo):
        return [(dict(r), label) for r, label in self.value]


class _Weights(Named):
    def __deepcopy__(self, memo):
        return dict(self.value)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _unpack(e) -> bool:
    """An example used whole where its row or its label belongs: each example is a pair (row, label)."""
    text = str(e)
    return (isinstance(e, ValueError) and "unpack" in text
            or isinstance(e, TypeError) and ("tuple indices" in text or "unhashable type: 'dict'" in text)
            or isinstance(e, NameError) and "'label'" in text)


def _no_name(name: str):
    return lambda e: isinstance(e, NameError) and f"'{name}'" in str(e)


_NAMES = {"np": np, "math": math}
_NAMES_P = {"np": np, "math": math, "probability": probability}

# exercise 1: probability. The two examples of the lesson, then: no words, a word with a count of two, another bias, a
# strongly positive row, and the example sentence of the lesson with the weights after training
PROB_CASES = [
    (_lit({"great": 1.15, "food": 1.363}), Named("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit({"not": 1.04, "good": 1.13}), Named("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit({}), Named("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit({"bad": 3.022}), Named("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit({"food": 1.363, "good": 1.13}), Named("small_weights", SMALL_WEIGHTS), 0.75),
    (_lit({"great": 2.3, "good": 1.13}), Named("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (Named(f"row({_value(EXAMPLE_TEXT)})", row(EXAMPLE_TEXT)), _Weights("weights", weights()), Named("bias", bias())),
]

_P_START = "def probability(row, weights, bias):\n    ...\n"
_P = ("def probability(row, weights, bias):\n"
      "    z = bias\n"
      "    for word in row:\n"
      "        z = z + row[word] * weights[word]\n"
      "    return 1 / (1 + np.exp(-z))\n")
_P_NO_BIAS = _P.replace("    z = bias\n", "    z = 0\n")
_P_NO_SIGMOID = _P.replace("    return 1 / (1 + np.exp(-z))\n", "    return z\n")
_P_SIGN = _P.replace("np.exp(-z)", "np.exp(z)")
_P_VALUE = _P.replace("z = z + row[word] * weights[word]", "z = z + row[word]")
_P_WEIGHT = _P.replace("z = z + row[word] * weights[word]", "z = z + weights[word]")
_P_EXP_ONLY = _P.replace("    return 1 / (1 + np.exp(-z))\n", "    return np.exp(-z)\n")
_P_BIAS_EACH = _P.replace("z = z + row[word] * weights[word]", "z = z + bias + row[word] * weights[word]")
_P_RESET = ("def probability(row, weights, bias):\n"
            "    for word in row:\n"
            "        z = bias\n"
            "        z = z + row[word] * weights[word]\n"
            "    return 1 / (1 + np.exp(-z))\n")
_P_RETURN_IN = ("def probability(row, weights, bias):\n"
                "    z = bias\n"
                "    for word in row:\n"
                "        z = z + row[word] * weights[word]\n"
                "        return 1 / (1 + np.exp(-z))\n")

# exercise 2: gradients. The two examples of the lesson, then: one example, a word in examples with both labels, another
# bias, and the 2,400 training sentences at the start and with the weights after training
GRAD_CASES = [
    (_Examples("small_examples", SMALL_EXAMPLES), _Weights("start_weights", START_WEIGHTS), 0.0),
    (_Examples("small_examples", SMALL_EXAMPLES), _Weights("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit([({"good": 1.13}, 1)]), _Weights("start_weights", START_WEIGHTS), 0.0),
    (_Examples("more_examples", MORE_EXAMPLES), _Weights("small_weights", SMALL_WEIGHTS), 0.4),
    (_Examples("examples", training_examples()), _Weights("zero_weights", {w: 0.0 for w in vocabulary()}), 0.0),
    (_Examples("examples", training_examples()), _Weights("weights", weights()), Named("bias", bias())),
]

_G_START = "def gradients(examples, weights, bias):\n    ...\n"
_G = ("def gradients(examples, weights, bias):\n"
      "    n = len(examples)\n"
      "    grad = {word: 0.0 for word in weights}\n"
      "    grad_bias = 0.0\n"
      "    for row, label in examples:\n"
      "        error = probability(row, weights, bias) - label\n"
      "        for word in row:\n"
      "            grad[word] = grad[word] + error * row[word] / n\n"
      "        grad_bias = grad_bias + error / n\n"
      "    return grad, grad_bias\n")
_G_SIGN = _G.replace("error = probability(row, weights, bias) - label", "error = label - probability(row, weights, bias)")
_G_NO_N = _G.replace("error * row[word] / n", "error * row[word]").replace("error / n", "error")
_G_NO_X = _G.replace("error * row[word] / n", "error / n")
_G_ONLY_X = _G.replace("error * row[word] / n", "row[word] / n")
_G_ROUNDED = _G.replace("error = probability(row, weights, bias) - label",
                        "error = (probability(row, weights, bias) >= 0.5) - label")
_G_BIAS_IN = _G.replace("            grad[word] = grad[word] + error * row[word] / n\n        grad_bias = grad_bias + error / n\n",
                        "            grad[word] = grad[word] + error * row[word] / n\n            grad_bias = grad_bias + error / n\n")
_G_ONLY_ROWS = ("def gradients(examples, weights, bias):\n"
                "    n = len(examples)\n"
                "    grad = {}\n"
                "    grad_bias = 0.0\n"
                "    for row, label in examples:\n"
                "        error = probability(row, weights, bias) - label\n"
                "        for word in row:\n"
                "            if word in grad:\n"
                "                grad[word] = grad[word] + error * row[word] / n\n"
                "            else:\n"
                "                grad[word] = error * row[word] / n\n"
                "        grad_bias = grad_bias + error / n\n"
                "    return grad, grad_bias\n")
_G_EMPTY = _G.replace("grad = {word: 0.0 for word in weights}", "grad = {}")
_G_RETURN_IN = _G.replace("    return grad, grad_bias\n", "        return grad, grad_bias\n")
_G_NO_BIAS = _G.replace("    return grad, grad_bias\n", "    return grad\n")
_G_WEIGHTS_N = _G.replace("    n = len(examples)\n", "    n = len(weights)\n")

# exercise 3: accuracy. The two examples of the lesson, then: one example labeled wrong, a bias that makes every
# sentence positive, and the 600 held-out sentences with the weights after training
ACC_CASES = [
    (_Examples("small_examples", SMALL_EXAMPLES), _Weights("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_Examples("more_examples", MORE_EXAMPLES), _Weights("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_lit([({"bad": 1.511}, 1)]), _Weights("small_weights", SMALL_WEIGHTS), SMALL_BIAS),
    (_Examples("more_examples", MORE_EXAMPLES), _Weights("small_weights", SMALL_WEIGHTS), 9.0),
    (_Examples("held", held_out_examples()), _Weights("weights", weights()), Named("bias", bias())),
]

_A_START = "def accuracy(examples, weights, bias):\n    ...\n"
_A = ("def accuracy(examples, weights, bias):\n"
      "    right = 0\n"
      "    for row, label in examples:\n"
      "        guess = 0\n"
      "        if probability(row, weights, bias) >= 0.5:\n"
      "            guess = 1\n"
      "        if guess == label:\n"
      "            right = right + 1\n"
      "    return right / len(examples)\n")
_A_COUNT = _A.replace("    return right / len(examples)\n", "    return right\n")
_A_PERCENT = _A.replace("    return right / len(examples)\n", "    return 100 * right / len(examples)\n")
_A_FLIPPED = _A.replace(">= 0.5", "< 0.5")
_A_KEPT = ("def accuracy(examples, weights, bias):\n"
           "    right = 0\n"
           "    guess = 0\n"
           "    for row, label in examples:\n"
           "        if probability(row, weights, bias) >= 0.5:\n"
           "            guess = 1\n"
           "        if guess == label:\n"
           "            right = right + 1\n"
           "    return right / len(examples)\n")
_A_WRONG = _A.replace("if guess == label:", "if guess != label:")
_A_RETURN_IN = ("def accuracy(examples, weights, bias):\n"
                "    right = 0\n"
                "    for row, label in examples:\n"
                "        guess = 0\n"
                "        if probability(row, weights, bias) >= 0.5:\n"
                "            guess = 1\n"
                "        if guess == label:\n"
                "            right = right + 1\n"
                "        return right / len(examples)\n")
_A_WEIGHTS_N = _A.replace("right / len(examples)", "right / len(weights)")
def _other_line(got, text) -> bool:
    """A comparison of the probability with a number from 0 to 1 other than 0.5, such as `>= 0.6`."""
    code = "\n".join(line.split("#")[0] for line in str(text).splitlines())
    found = [float(x) for x in re.findall(r"[<>]=?\s*(\d*\.\d+|\d+)", code)]
    return any(0 < x < 1 and abs(x - 0.5) > 1e-9 for x in found)


_A_P_LABEL = _A.replace("        guess = 0\n        if probability(row, weights, bias) >= 0.5:\n            guess = 1\n"
                        "        if guess == label:\n",
                        "        if probability(row, weights, bias) == label:\n")

EXERCISES = {
    "probability": FunctionBox(
        name="probability",
        picture=probability_picture,
        function="probability",
        start=_P_START,
        solution=_P,
        names=_NAMES,
        visible=PROB_CASES[:2],
        hidden=PROB_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("over-weights", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it looks up a word in `row` that the row does not have. Your loop goes over the words of `weights`; go over the words of the row instead, since a word the row lacks adds nothing.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie zoekt in `row` een woord op dat niet in de rij staat. Je loop gaat over de woorden van `weights`; ga in plaats daarvan over de woorden van de rij, want een woord dat niet in de rij staat, telt niet mee."),
            Mistake("no-sigmoid", wrong=_P_NO_SIGMOID,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the logit $z$ itself. The probability is the sigmoid of it, `1 / (1 + np.exp(-z))`, a number between 0 and 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de logit $z$ zelf terug. De kans is de sigmoid daarvan, `1 / (1 + np.exp(-z))`, een getal tussen 0 en 1."),
            Mistake("sign", wrong=_P_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: the probability of negative, 1 minus the right one. The sigmoid has a minus sign in the exponent: `np.exp(-z)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kans op negatief, 1 min de goede kans. De sigmoid heeft een minteken in de exponent: `np.exp(-z)`."),
            Mistake("exp-only", wrong=_P_EXP_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back `np.exp(-z)` alone. The sigmoid divides 1 by 1 plus that.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft alleen `np.exp(-z)` terug. De sigmoid deelt 1 door 1 plus dat getal."),
            Mistake("no-bias", wrong=_P_NO_BIAS,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out the bias. The sum starts at the bias, and each word adds its value times its weight.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat de bias weg. De som begint bij de bias, en elk woord telt de waarde maal de weight erbij op."),
            Mistake("bias-each", wrong=_P_BIAS_EACH,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the bias again for every word. The bias counts once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de bias voor elk woord opnieuw op. De bias telt één keer."),
            Mistake("value-only", wrong=_P_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the TF-IDF values, and leaves out the weights. Multiply each value by the weight of its word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de TF-IDF-waarden op, en laat de weights weg. Vermenigvuldig elke waarde met de weight van het woord."),
            Mistake("weight-only", wrong=_P_WEIGHT,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the weights, and leaves out the TF-IDF values. Multiply each weight by the value of its word in the row.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de weights op, en laat de TF-IDF-waarden weg. Vermenigvuldig elke weight met de waarde van het woord in de rij."),
            Mistake("reset", wrong=_P_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the last word counts. Your `z` starts at the bias again in each round; set it once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het laatste woord telt. Je `z` begint in elke ronde weer bij de bias; zet `z` één keer, vóór de loop."),
            Mistake("return-in-loop", wrong=_P_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first word counts. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste woord telt. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start a sum at the bias, add for each word of the row its value times its weight, and give back the sigmoid of the sum.",
                      "nl": "Vervang de `...`: begin een som bij de bias, tel voor elk woord van de rij de waarde maal de weight op, en geef de sigmoid van de som terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function start at the bias, add each value times its weight, and take the sigmoid at the end?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begint je functie bij de bias, telt je functie elke waarde maal de weight op, en neemt je functie aan het eind de sigmoid?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every row: no words at all, another bias, a sentence of the set?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke rij: helemaal geen woorden, een andere bias, een zin uit de set?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `probability` is right in all {n} cases. Look at the picture: the probabilities your function gives seven held-out sentences, with the weights after training.",
                        "nl": "Je `probability` klopt in alle {n} gevallen. Kijk naar de afbeelding: de kansen die je functie zeven achtergehouden zinnen geeft, met de weights na de training."},
        },
    ),
    "gradients": FunctionBox(
        name="gradients",
        picture=gradients_picture,
        function="gradients",
        start=_G_START,
        solution=_G,
        names=_NAMES_P,
        visible=GRAD_CASES[:2],
        hidden=GRAD_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it adds to the slope of a word that `grad` does not hold yet. Start `grad` with a slope of 0 for every word of `weights`.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie telt iets op bij de helling van een woord dat nog niet in `grad` staat. Begin `grad` met helling 0 voor elk woord van `weights`."),
            Mistake("unpack", error=_unpack,
                    en="With `{call}`, your function stops with an error: each example is a pair of a row and a label. Take the two apart in the loop: `for row, label in examples:`.",
                    nl="Met `{call}` stopt je functie met een fout: elk voorbeeld is een paar van een rij en een label. Haal de twee uit elkaar in de loop: `for row, label in examples:`."),
            Mistake("sign", wrong=_G_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: every slope has the wrong sign. The error is the probability minus the label, $p - y$; with the label minus the probability, gradient descent walks uphill.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke helling heeft het verkeerde teken. De fout is de kans min het label, $p - y$; met het label min de kans loopt gradient descent bergop."),
            Mistake("rounded", wrong=_G_ROUNDED,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the label it would give, 0 or 1, minus the label. The error uses the probability itself, so that a sentence that is almost right pushes only a little.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het label dat de functie zou geven, 0 of 1, min het label. De fout gebruikt de kans zelf, zodat een zin die bijna goed is de weights maar een beetje verandert."),
            Mistake("no-mean", wrong=_G_NO_N,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the slopes of the examples, and does not divide by their number. The slopes are of the mean log loss: divide each part by `n`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de hellingen van de voorbeelden op, en deelt niet door het aantal. Het gaat om de hellingen van de gemiddelde log loss: deel elk deel door `n`."),
            Mistake("no-value", wrong=_G_NO_X,
                    en="`{call}` gives {got}, and it should give {want}: each word gets the error alone. The slope for a weight is the error times the value of that word in the row: a word with a high TF-IDF value pushes harder.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt alleen de fout. De helling voor een weight is de fout maal de waarde van dat woord in de rij: een woord met een hoge TF-IDF-waarde telt zwaarder mee."),
            Mistake("no-error", wrong=_G_ONLY_X,
                    en="`{call}` gives {got}, and it should give {want}: each word gets its value alone. Multiply the value by the error, the probability minus the label, so that a sentence the model already gets right pushes little.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt alleen de eigen waarde. Vermenigvuldig de waarde met de fout, de kans min het label, zodat een zin die het model al goed heeft de weights weinig verandert."),
            Mistake("bias-per-word", wrong=_G_BIAS_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the error to the slope of the bias once for every word. The bias is one number for the whole sentence: add its part once per example, outside the loop over the words.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de fout voor elk woord bij de helling van de bias op. De bias is één getal voor de hele zin: tel het deel ervan één keer per voorbeeld op, buiten de loop over de woorden."),
            Mistake("missing-words", wrong=_G_ONLY_ROWS,
                    en="`{call}` gives {got}, and it should give {want}: your dictionary lacks the words of `weights` that no example has. Such a word gets no push, so its slope is 0; give it 0, so that every weight has a slope.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je dictionary ontbreken de woorden van `weights` die in geen enkel voorbeeld staan. Zo'n woord krijgt van geen enkele zin een bijdrage, dus de helling is 0; geef het woord 0, zodat elke weight een helling heeft."),
            Mistake("n-of-weights", wrong=_G_WEIGHTS_N,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of words in `weights`. The mean is over the examples: divide by `len(examples)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal woorden in `weights`. Het gemiddelde gaat over de voorbeelden: deel door `len(examples)`."),
            Mistake("no-bias", wrong=_G_NO_BIAS,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the slopes of the weights only. Give back two things, the dictionary and the slope of the bias: `return grad, grad_bias`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft alleen de hellingen van de weights terug. Geef twee dingen terug, de dictionary en de helling van de bias: `return grad, grad_bias`."),
            Mistake("return-in-loop", wrong=_G_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first example counts. The `return` stands inside the loop over the examples; move it back, under the first `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste voorbeeld telt. De `return` staat in de loop over de voorbeelden; zet de `return` terug, onder de eerste `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with a slope of 0 for every word of `weights` and for the bias, add for each example its error times the value of each of its words, divided by `n`, and its error divided by `n` for the bias, and give back both.",
                      "nl": "Vervang de `...`: begin met helling 0 voor elk woord van `weights` en voor de bias, tel voor elk voorbeeld de fout maal de waarde van elk van de woorden op, gedeeld door `n`, en de fout gedeeld door `n` voor de bias, en geef beide terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each example add its error, the probability minus the label, times the value of each of its words, divided by the number of examples?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt elk voorbeeld de fout, de kans min het label, maal de waarde van elk van de woorden op, gedeeld door het aantal voorbeelden?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for any examples: one example, another bias, all 2,400 training sentences?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle voorbeelden: één voorbeeld, een andere bias, alle 2.400 trainingszinnen?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a dictionary and a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een dictionary en een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `gradients` is right in all {n} cases. Look at the picture: the slopes your function gives at the start, over the 2,400 training sentences. The demo below trains with your function.",
                        "nl": "Je `gradients` klopt in alle {n} gevallen. Kijk naar de afbeelding: de hellingen die je functie aan het begin geeft, over de 2.400 trainingszinnen. De demo hieronder traint met je functie."},
        },
    ),
    "accuracy": FunctionBox(
        name="accuracy",
        picture=accuracy_picture,
        function="accuracy",
        start=_A_START,
        solution=_A,
        names=_NAMES_P,
        visible=ACC_CASES[:2],
        hidden=ACC_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("count", wrong=_A_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of examples labeled right. The accuracy is the share: divide that number by the number of examples.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal voorbeelden met het goede label. De accuracy is het aandeel: deel dat aantal door het aantal voorbeelden."),
            Mistake("percent", wrong=_A_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: a percentage. Give the share as a number from 0 to 1, without the 100.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een percentage. Geef het aandeel als getal van 0 tot 1, zonder de 100."),
            Mistake("flipped", wrong=_A_FLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: 1 minus the right share, the share of examples labeled wrong. The probability is that of positive, so a probability of 0.5 or more gives the label 1; count the examples whose own label is the one your function gives.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: 1 min het goede aandeel, het aandeel voorbeelden met een fout label. Het gaat om de kans op positief, dus een kans van 0,5 of meer geeft het label 1; tel de voorbeelden waarvan het eigen label het label is dat je functie geeft."),
            Mistake("kept", wrong=_A_KEPT,
                    en="`{call}` gives {got}, and it should give {want}: once an example gets 1, every later example gets 1 too. Set the label your function would give back to 0 at the start of each round, inside the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zodra een voorbeeld 1 krijgt, krijgt elk volgend voorbeeld ook 1. Zet het label dat je functie zou geven aan het begin van elke ronde terug op 0, in de loop."),
            Mistake("p-is-label", wrong=_A_P_LABEL,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the probability itself with the label, and a probability is almost never exactly 0 or 1. Turn it into a label first: 1 for a probability of 0.5 or more, else 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt de kans zelf met het label, en een kans is bijna nooit precies 0 of 1. Maak er eerst een label van: 1 voor een kans van 0,5 of meer, anders 0."),
            Mistake("other-line", matches=_other_line,
                    en="`{call}` gives {got}, and it should give {want}: your function draws the line at another probability. As the exercise says, a probability of 0.5 or more gives the label 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie legt de grens bij een andere kans. Zoals de oefening zegt, geeft een kans van 0,5 of meer het label 1."),
            Mistake("n-of-weights", wrong=_A_WEIGHTS_N,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of words in `weights`. The share is over the examples: divide by `len(examples)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal woorden in `weights`. Het aandeel gaat over de voorbeelden: deel door `len(examples)`."),
            Mistake("return-in-loop", wrong=_A_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: only the first example counts. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste voorbeeld telt. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the examples that get their own label, 1 for a probability of 0.5 or more and else 0, and give back that count divided by the number of examples.",
                      "nl": "Vervang de `...`: tel de voorbeelden die het eigen label krijgen, 1 voor een kans van 0,5 of meer en anders 0, en geef dat aantal gedeeld door het aantal voorbeelden terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give each example the label 1 for a probability of 0.5 or more and else 0, and count the examples where that is their own label?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie elk voorbeeld het label 1 bij een kans van 0,5 of meer en anders 0, en telt je functie de voorbeelden waarbij dat het eigen label is?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for any examples: one example, another bias, the 600 held-out sentences?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle voorbeelden: één voorbeeld, een andere bias, de 600 achtergehouden zinnen?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `accuracy` is right in all {n} cases. Look at the picture: your accuracy on the 600 held-out sentences, next to naive Bayes.",
                        "nl": "Je `accuracy` klopt in alle {n} gevallen. Kijk naar de afbeelding: je accuracy op de 600 achtergehouden zinnen, naast naive Bayes."},
        },
    ),
    "what-weight": Sort(
        name="what-weight",
        groups={
            "up": {"en": "A clear positive weight (more than 0.15)", "nl": "Een duidelijk positieve weight (meer dan 0,15)"},
            "zero": {"en": "About 0 (between −0.15 and 0.15)", "nl": "Ongeveer 0 (tussen −0,15 en 0,15)"},
            "down": {"en": "A clear negative weight (less than −0.15)", "nl": "Een duidelijk negatieve weight (minder dan −0,15)"},
        },
        items={
            "the": {"en": "*the*: in 538 positive and 518 negative training sentences",
                    "nl": "*the*: in 538 positieve en 518 negatieve trainingszinnen"},
            "money": {"en": "*money*: in 3 positive and 21 negative training sentences",
                      "nl": "*money*: in 3 positieve en 21 negatieve trainingszinnen"},
            "is": {"en": "*is*: in 295 positive and 239 negative training sentences",
                   "nl": "*is*: in 295 positieve en 239 negatieve trainingszinnen"},
            "place": {"en": "*place*: in 43 positive and 43 negative training sentences",
                      "nl": "*place*: in 43 positieve en 43 negatieve trainingszinnen"},
            "film": {"en": "*film*: in 81 positive and 43 negative training sentences",
                     "nl": "*film*: in 81 positieve en 43 negatieve trainingszinnen"},
            "at": {"en": "*at*: in 37 positive and 61 negative training sentences",
                   "nl": "*at*: in 37 positieve en 61 negatieve trainingszinnen"},
        },
        solution={"the": "zero", "money": "down", "is": "up", "place": "zero", "film": "up", "at": "down"},
        nudge={
            "the": {"en": "Is this word in clearly more sentences of one kind than of the other?",
                    "nl": "Staat dit woord in duidelijk meer zinnen van de ene soort dan van de andere?"},
            "money": {"en": "In which kind of sentence is this word far more often?",
                      "nl": "In welke soort zin staat dit woord veel vaker?"},
            "is": {"en": "Of every hundred sentences with this word, about 55 are of one kind. Which kind, and which way do they push?",
                   "nl": "Van elke honderd zinnen met dit woord zijn er ongeveer 55 van één soort. Welke soort, en welke kant trekken die zinnen de weight op?"},
            "place": {"en": "When a word is in as many sentences of each kind, do their pushes add up or cancel out?",
                      "nl": "Als een woord in evenveel zinnen van elke soort staat, werken de zinnen dan samen, of heffen de zinnen elkaar op?"},
            "film": {"en": "Does it matter for the weight what the word means, or in which sentences it occurs?",
                     "nl": "Doet het voor de weight ertoe wat het woord betekent, of in welke zinnen het woord voorkomt?"},
            "at": {"en": "Of every hundred sentences with this word, about 62 are of one kind. Which kind?",
                   "nl": "Van elke honderd zinnen met dit woord zijn er ongeveer 62 van één soort. Welke soort?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Put each word under the weight it gets after training on the 2,400 sentences.",
                      "nl": "Zet elk woord bij de weight die het krijgt na de training op de 2.400 zinnen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this word goes somewhere else.",
                      "nl": "{item}: dit woord hoort ergens anders."},
            "correct": {"en": "All {n} are right. *the* gets −0.083 and *place* −0.023: their sentences push both ways about equally. *money* gets −0.344: 21 of its 24 sentences are complaints, many about wasted money. *film* gets +0.595, *is* +0.249 and *at* −0.496, though none of the three is positive or negative in itself: a weight follows the sentences a word is in, not what it means. *at* is often in *at all*, as in *Avoid at all costs.*: 13 negative training sentences have *at all*, and 3 positive ones.",
                        "nl": "Alle {n} kloppen. *the* krijgt −0,083 en *place* −0,023: de zinnen met deze twee woorden trekken ongeveer even hard naar beide kanten. *money* krijgt −0,344: 21 van de 24 zinnen zijn klachten, vaak over weggegooid geld. *film* krijgt +0,595, *is* +0,249 en *at* −0,496, hoewel geen van de drie op zichzelf positief of negatief is: een weight volgt de zinnen waarin een woord staat, niet wat het woord betekent. *at* staat vaak in *at all*, zoals in *Avoid at all costs.* (*Koste wat kost vermijden.*): 13 negatieve trainingszinnen hebben *at all*, en 3 positieve."},
        },
    ),
}

SORT_WORDS = {"the": (538, 518), "money": (3, 21), "is": (295, 239), "place": (43, 43), "film": (81, 43),
              "at": (37, 61)}

CASES = [
    ("probability", _P_START, "empty", None),
    ("probability", _P, "correct", None),
    ("probability", _P.replace("np.exp", "math.exp"), "correct", None),
    ("probability", "def probability(row, weights, bias):\n    z = bias\n    for word, value in row.items():\n        z += value * weights[word]\n    return 1 / (1 + np.exp(-z))\n", "correct", None),
    ("probability", "def probability(row, weights, bias):\n    z = bias + sum(row[w] * weights[w] for w in row)\n    return 1 / (1 + math.exp(-z))\n", "correct", None),
    ("probability", _P_NO_SIGMOID, "wrong", "no-sigmoid"),
    ("probability", _P_SIGN, "wrong", "sign"),
    ("probability", _P_EXP_ONLY, "wrong", "exp-only"),
    ("probability", _P_NO_BIAS, "wrong", "no-bias"),
    ("probability", _P_BIAS_EACH, "wrong", "bias-each"),
    ("probability", _P_VALUE, "wrong", "value-only"),
    ("probability", _P_WEIGHT, "wrong", "weight-only"),
    ("probability", _P_RESET, "wrong", "reset"),
    ("probability", _P_RETURN_IN, "wrong", "return-in-loop"),
    ("probability", _P.replace("    return 1 / (1 + np.exp(-z))\n", "    print(1 / (1 + np.exp(-z)))\n"), "wrong", "prints"),
    ("probability", "def probability(row, weights, bias):\n    if \"great\" in row:\n        return 0.8950522979810875\n    return 0.3709134732442586\n", "almost", None),
    ("gradients", _G_START, "empty", None),
    ("gradients", _G, "correct", None),
    ("gradients", "def gradients(examples, weights, bias):\n    n = len(examples)\n    grad = {}\n    for word in weights:\n        grad[word] = 0\n    grad_bias = 0\n    for row, label in examples:\n        p = probability(row, weights, bias)\n        for word in row:\n            grad[word] += (p - label) * row[word] / n\n        grad_bias += (p - label) / n\n    return grad, grad_bias\n", "correct", None),
    ("gradients", "def gradients(examples, weights, bias):\n    grad = {word: 0.0 for word in weights}\n    grad_bias = 0.0\n    for row, label in examples:\n        error = probability(row, weights, bias) - label\n        for word in row:\n            grad[word] = grad[word] + error * row[word]\n        grad_bias = grad_bias + error\n    for word in grad:\n        grad[word] = grad[word] / len(examples)\n    return grad, grad_bias / len(examples)\n", "correct", None),
    ("gradients", _G_EMPTY, "error", "lookup"),
    ("gradients", _G.replace("for row, label in examples:", "for row in examples:"), "error", "unpack"),
    ("gradients", _G_SIGN, "wrong", "sign"),
    ("gradients", _G_ROUNDED, "wrong", "rounded"),
    ("gradients", _G_NO_N, "wrong", "no-mean"),
    ("gradients", _G_NO_X, "wrong", "no-value"),
    ("gradients", _G_ONLY_X, "wrong", "no-error"),
    ("gradients", _G_BIAS_IN, "wrong", "bias-per-word"),
    ("gradients", _G_ONLY_ROWS, "wrong", "missing-words"),
    ("gradients", _G_WEIGHTS_N, "wrong", "n-of-weights"),
    ("gradients", _G_NO_BIAS, "wrong", "no-bias"),
    ("gradients", _G_RETURN_IN, "wrong", "return-in-loop"),
    ("gradients", _G.replace("    return grad, grad_bias\n", ""), "wrong", "no-return"),
    ("accuracy", _A_START, "empty", None),
    ("accuracy", _A, "correct", None),
    ("accuracy", "def accuracy(examples, weights, bias):\n    right = 0\n    for row, label in examples:\n        if (probability(row, weights, bias) >= 0.5) == (label == 1):\n            right += 1\n    return right / len(examples)\n", "correct", None),
    ("accuracy", _A.replace(">= 0.5", "> 0.5"), "correct", None),
    ("accuracy", _A_COUNT, "wrong", "count"),
    ("accuracy", _A_PERCENT, "wrong", "percent"),
    ("accuracy", _A_FLIPPED, "wrong", "flipped"),
    ("accuracy", _A_WRONG, "wrong", "flipped"),
    ("accuracy", _A_KEPT, "wrong", "kept"),
    ("accuracy", _A_P_LABEL, "wrong", "p-is-label"),
    ("accuracy", _A_WEIGHTS_N, "wrong", "n-of-weights"),
    ("accuracy", _A_RETURN_IN, "wrong", "return-in-loop"),
    ("accuracy", _A.replace("    return right / len(examples)\n", ""), "wrong", "no-return"),
    ("accuracy", "def accuracy(examples, weights, bias):\n    if len(examples) == 3:\n        return 1.0\n    return 0.5\n", "almost", None),
    ("probability", _P.replace("for word in row:", "for word in weights:"), "error", "over-weights"),
    ("accuracy", _A.replace(">= 0.5", ">= 0.6"), "wrong", "other-line"),
    ("accuracy", _A.replace(">= 0.5", "> 0.5"), "correct", None),
    ("what-weight", {}, "empty", None),
    ("what-weight", {"the": "zero", "money": "down", "is": "up", "place": "zero", "film": "up", "at": "down"},
     "correct", None),
    ("what-weight", {"the": "zero", "money": "down"}, "wrong", "unfinished"),
    ("what-weight", {"is": "zero"}, "wrong", "wrong:is"),
    ("what-weight", {"the": "zero", "money": "down", "is": "up", "place": "zero", "film": "zero", "at": "down"},
     "wrong", "wrong:film"),
]
