# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-project-spam, the first step of the project `text`: the TF-IDF row of a message, with no
column for a word the training messages lack (features, saved as text/features.py), one step of gradient descent on
those rows (step, with train, saved as text/train.py), and how many spam messages a filter lets through and how many
other messages it blocks at a threshold (mistakes, saved as text/judge.py); and a choice between two filters.

The messages are those of the SMS Spam Collection v.1 (Almeida and Gómez Hidalgo, 2011; UCI Machine Learning
Repository, dataset 228, CC BY 4.0, https://doi.org/10.24432/C5CC84), which the learner downloads: everything_ai.sms.
The lesson shows only the messages it chose (WATCHED, IDEA_LINE, BANK_LINE, NB_LINES), with numbers masked. The small
messages of the exercises, the texts, the cases of the checks and the pictures are the course's own."""
import contextlib
import errno
import functools
import html
import io
import math
import re
import urllib.error
import warnings

import numpy as np

from everything_ai import chart, hub, live, projects, sms, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, run_code, with_time_limit
from everything_ai.lesson import download_card, download_size

STEPS = 200    # steps of gradient descent in the demo
RATE = 1.0     # its learning rate
THRESHOLD = 0.5   # the threshold of the demo, as in Logistic regression on words
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]   # the stops of the threshold explore: 0.05 to 0.95
LOOKS = [1, 2, 5] + list(range(10, STEPS + 1, 10))   # the steps the rewind stops at

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The SMS Spam Collection is downloaded once from the UCI Machine Learning Repository, {download_size(sms.FILE, 'en')}, when you click its button in the lesson; the worked example and the demos need it, the exercises work without it.",
    "nl": f"Werkt op elke computer. De SMS Spam Collection wordt één keer gedownload van de UCI Machine Learning Repository, {download_size(sms.FILE, 'nl')}, als je in de les op de knop ervoor klikt; het uitgewerkte voorbeeld en de demo's hebben de set nodig, de oefeningen werken zonder.",
}


def _size(lang: str) -> str:
    return download_size(sms.FILE, lang)


def sms_or_stop(button, texts: dict):
    """everything_ai.sms for the lesson's cells, once the zip is downloaded. When it is not there yet, the cell stops
    with `texts["ask"]` and the button; a click downloads it, with the progress in the cell. `texts` has "ask", "busy"
    ({done} and {total} in KB), "no_net", "not_pinned", "no_room" and "failed" ({why}): why it did not work; and
    "here"."""
    import marimo as mo
    failed = None
    if not sms.ready() and button.value:
        total = round(hub.PINNED[sms.FILE]["size"] / 1e3)

        def progress(done, size):
            mo.output.replace(mo.md(texts["busy"].format(done=round(done / 1e3), total=total)))
        try:
            hub.download(*sms.FILE, progress=progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not sms.ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return sms


# --- the messages the lesson shows -------------------------------------------------------------------------------------

WATCHED = [4298, 1197, 2953]   # three test messages the demo watches, by their line in the file: spam, spam, ham
IDEA_LINE = 1197               # the test message of the picture in The idea
BANK_LINE = 3825               # the ham test message with the highest probability of spam after training: a bank's warning
NB_LINES = [1705, 4601, 2174]  # three ham test messages that naive Bayes blocks
_MASK = {"en": "[number]", "nl": "[nummer]"}


def shown(line: int, lang: str, longest: int = 0) -> str:
    """A message the lesson chose, as it shows it: every number of five digits or more masked, and cut off after
    `longest` characters when that is given."""
    text = re.sub(r"\d{5,}", _MASK[lang], sms.lines()[line - 1][1])
    return text if not longest or len(text) <= longest else text[:longest - 1] + "…"


@functools.cache
def _test_index(line: int) -> int:
    """Where a message the lesson chose stands among the test messages: the first test message with its words."""
    want = sms.words(sms.lines()[line - 1][1])
    return next(i for i, (w, _) in enumerate(sms.test()) if w == want)


# --- the course's own functions: the solutions, and logistic.py of the project ---------------------------------------

def probability(row, weights, bias):
    z = bias
    for word in row:
        z = z + row[word] * weights[word]
    return 1 / (1 + np.exp(-z))


def gradients(examples, weights, bias):
    n = len(examples)
    grad = {word: 0.0 for word in weights}
    grad_bias = 0.0
    for row, label in examples:
        error = probability(row, weights, bias) - label
        for word in row:
            grad[word] = grad[word] + error * row[word] / n
        grad_bias = grad_bias + error / n
    return grad, grad_bias


def document_frequencies(messages):
    df = {}
    for words, label in messages:
        for word in set(words):
            if word not in df:
                df[word] = 0
            df[word] = df[word] + 1
    return df


def features(words, df, n):
    row = {}
    for word in words:
        if word in df:
            if word not in row:
                row[word] = 0.0
            row[word] = row[word] + np.log10(n / df[word])
    return row


def step(examples, weights, bias, rate):
    grad, grad_bias = gradients(examples, weights, bias)
    new_weights = {}
    for word in weights:
        new_weights[word] = weights[word] - rate * grad[word]
    new_bias = bias - rate * grad_bias
    return new_weights, new_bias


def mistakes(examples, weights, bias, threshold):
    missed = 0
    blocked = 0
    for row, label in examples:
        p = probability(row, weights, bias)
        if label == 1 and p < threshold:
            missed = missed + 1
        if label == 0 and p >= threshold:
            blocked = blocked + 1
    return missed, blocked


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


# --- the course's run, with NumPy: the numbers of the text, checked by the tests --------------------------------------

def course_run() -> dict:
    """The course's filter on the real messages, with NumPy: TF-IDF rows with the document frequencies of the training
    messages, STEPS steps of gradient descent with learning rate RATE from weight 0 and bias 0, the same steps as
    step() with the course's gradients. Gives {"vocab", "df", "weights" (an array over vocab), "bias", "curve" (the mean
    log loss after each step), "start" (before the first step), "test_p", "test_y", "train_p", "train_y"}."""
    def make():
        training, test = sms.training(), sms.test()
        n = len(training)
        df = document_frequencies(training)
        vocab = sorted(df)
        cols = {w: k for k, w in enumerate(vocab)}

        def sparse(messages):
            rows_, cols_, vals = [], [], []
            for i, (words, _) in enumerate(messages):
                for w, v in features(words, df, n).items():
                    rows_.append(i)
                    cols_.append(cols[w])
                    vals.append(float(v))
            return (np.array(rows_, dtype=int), np.array(cols_, dtype=int), np.array(vals),
                    np.array([label for _, label in messages], dtype=float))

        A, B = sparse(training), sparse(test)

        def p_of(S, w, b, m):
            z = np.bincount(S[0], weights=S[2] * w[S[1]], minlength=m) + b
            return 1 / (1 + np.exp(-z))

        def loss(p, y):
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        w, b = np.zeros(len(vocab)), 0.0
        start = loss(p_of(A, w, b, n), A[3])
        curve = []
        for _ in range(STEPS):
            p = p_of(A, w, b, n)
            w = w - RATE * np.bincount(A[1], weights=A[2] * (p - A[3])[A[0]], minlength=len(vocab)) / n
            b = b - RATE * float((p - A[3]).mean())
            curve.append(loss(p_of(A, w, b, n), A[3]))
        return {"vocab": vocab, "df": df, "weights": w, "bias": b, "curve": curve, "start": start,
                "test_p": p_of(B, w, b, len(test)), "test_y": B[3], "train_p": p_of(A, w, b, n), "train_y": A[3]}
    return kept("nlp-project-spam/course-run", make)


def course_mistakes(threshold: float) -> tuple:
    """(missed, blocked) of the course's filter on the test messages at a threshold."""
    run = course_run()
    p, y = run["test_p"], run["test_y"]
    return int(((p < threshold) & (y == 1)).sum()), int(((p >= threshold) & (y == 0)).sum())


def naive_bayes() -> list:
    """What naive Bayes of *Spam or not: naive Bayes* says of every test message, 1 for spam: counted on the training
    messages, with add-1 smoothing over the vocabulary of the training messages, and the share of each label as its
    prior; a word that no training message has is left out."""
    def make():
        training = sms.training()
        counts = {0: {}, 1: {}}
        for words, label in training:
            for w in words:
                counts[label][w] = counts[label].get(w, 0) + 1
        vocab = set(counts[0]) | set(counts[1])
        total = {c: sum(counts[c].values()) for c in counts}
        prior = {c: sum(1 for _, label in training if label == c) / len(training) for c in counts}

        def score(words, c):
            return math.log(prior[c]) + sum(math.log((counts[c].get(w, 0) + 1) / (total[c] + len(vocab)))
                                            for w in words if w in vocab)
        return [1 if score(words, 1) > score(words, 0) else 0 for words, _ in sms.test()]
    return kept("nlp-project-spam/naive-bayes", make)


# The numbers of the text, from course_run() and naive_bayes() on the real messages (tests/test_nlp_project_spam.py).
TRAINING, TEST, SPAM_TRAINING, SPAM_TEST = 4137, 1034, 514, 139
LINES, SPAM_LINES, COPIES, DIFFERENT = 5574, 747, 403, 5171
VOCAB = 6957          # different words in the training messages: weights of the filter
UNKNOWN_MESSAGES = 493   # test messages with at least one word that no training message has
MISSED, BLOCKED = 15, 0          # the course's filter at 0.5 on the test messages
NB_MISSED, NB_BLOCKED = 10, 5    # naive Bayes on the test messages
LOSS_END = 0.031      # the mean log loss of the training messages after STEPS steps
TRAIN_RIGHT = 4114    # training messages the filter gets right at 0.5
BIAS_END = -3.344
BY_THRESHOLD = {0.05: (1, 53), 0.1: (2, 16), 0.2: (8, 1), 0.3: (10, 1), 0.35: (11, 0), 0.4: (11, 0), 0.5: (15, 0),
                0.6: (19, 0), 0.7: (22, 0), 0.8: (31, 0), 0.9: (43, 0)}
SPAM_WORDS = [("call", 0.94), ("p", 0.93), ("txt", 0.73), ("free", 0.71), ("text", 0.71), ("stop", 0.66)]
HAM_WORDS = [("i", -0.81), ("my", -0.50), ("gt", -0.42), ("lt", -0.42), ("me", -0.41), ("ok", -0.38)]
WORD_DF = {"call": (367, 225), "free": (163, 119), "i": (1289, 34), "p": (121, 114), "lt": (181, 0)}   # df, of them spam
WATCHED_P = {4298: 0.712, 1197: 0.253, 2953: 0.114}   # after STEPS steps
BANK_P = 0.329

# The picture in The idea: the test message IDEA_LINE after training, rounded to three decimals.
IDEA_ROW = {"you": 0.537, "have": 1.009, "new": 1.648, "please": 1.634, "call": 1.052}
IDEA_WEIGHTS = {"you": -0.084, "have": 0.203, "new": 0.413, "please": 0.263, "call": 0.943}
IDEA_WORDS = ["you", "have", "new", "voicemail", "please", "call"]   # its words; "voicemail" is in no training message


def idea_z() -> float:
    return BIAS_END + sum(IDEA_ROW[w] * IDEA_WEIGHTS[w] for w in IDEA_ROW)


# --- the small examples of the exercises -------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `small_df`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


SMALL_MESSAGES = [
    (["win", "a", "free", "prize", "call", "now"], 1),
    (["call", "me", "when", "you", "are", "free"], 0),
    (["txt", "stop", "to", "end", "free", "msgs"], 1),
    (["are", "you", "home", "i", "will", "call", "later"], 0),
    (["claim", "your", "prize", "txt", "claim", "now"], 1),
    (["ok", "see", "you", "at", "home"], 0),
    (["your", "mobile", "won", "a", "prize"], 1),
    (["i", "am", "free", "now"], 0),
]
SMALL_DF = document_frequencies(SMALL_MESSAGES)
SMALL_WORDS = ["free", "prize", "free", "call", "tonight"]   # "tonight" is in no small message
UNKNOWN = ("tonight", "maybe")   # the words of the cases that their df does not know
REAL_DF = {"call": 367, "free": 163, "i": 1289}   # three document frequencies of the real training messages

SMALL_EXAMPLES = [({"free": 0.602, "prize": 0.426}, 1), ({"call": 0.426, "free": 0.301}, 0),
                  ({"prize": 0.426, "txt": 0.602}, 1)]
START_WEIGHTS = {"free": 0.0, "prize": 0.0, "call": 0.0, "txt": 0.0}
SMALL_WEIGHTS = {"free": 1.2, "prize": 2.0, "call": -0.8, "txt": 1.5}
SMALL_BIAS = -0.6
MORE_EXAMPLES = SMALL_EXAMPLES + [({"call": 0.426}, 1), ({"free": 0.301, "txt": 0.602}, 0),
                                  ({"txt": 0.301, "call": 0.426}, 1)]
EMPTY_EXAMPLES = [({}, 0), ({}, 1)]   # two messages without a word the filter knows: both get p = 0.5 at bias 0

_SW = _Val("small_words", SMALL_WORDS)
_SDF = _Val("small_df", SMALL_DF)
_SE = _Val("small_examples", SMALL_EXAMPLES)
_ME = _Val("more_examples", MORE_EXAMPLES)
_ST = _Val("start_weights", START_WEIGHTS)
_SWT = _Val("small_weights", SMALL_WEIGHTS)

# --- the exercises -----------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


def _no_module(name: str) -> Mistake:
    return Mistake("no-logistic", error=lambda e: isinstance(e, ImportError) and "text" in str(e),
                   en=f"Python cannot find `text.logistic` with `{name}` in it: the file `work/projects/text/logistic.py` that the course put in your project is gone, or changed. The course puts it back when you delete the folder `work/projects/text/` and open the lesson again; your other files there are then gone too.",
                   nl=f"Python kan `text.logistic` met `{name}` erin niet vinden: het bestand `work/projects/text/logistic.py` dat de course in je project zette, is weg of veranderd. De course zet het terug als je de map `work/projects/text/` verwijdert en de les opnieuw opent; je andere bestanden daar zijn dan ook weg.")


_F_HEAD = ("import numpy as np\n\n\ndef document_frequencies(messages):\n"
           "    # for every word: in how many of the messages it is, as in the worked example\n"
           "    df = {}\n    for words, label in messages:\n        for word in set(words):\n"
           "            if word not in df:\n                df[word] = 0\n            df[word] = df[word] + 1\n"
           "    return df\n\n\ndef features(words, df, n):\n"
           "    # the TF-IDF row of one message: for every word of `words` that `df` knows,\n"
           "    # how often it is in the message times its inverse document frequency\n")
_F_START = _F_HEAD + "    ...\n"


def _f(add: str = "row[word] + np.log10(n / df[word])", skip: bool = True, first: bool = True) -> str:
    lines = ["    row = {}", "    for word in words:"]
    pad = "        "
    if skip:
        lines.append("        if word in df:")
        pad = "            "
    if first:
        lines += [pad + "if word not in row:", pad + "    row[word] = 0.0"]
    lines.append(pad + f"row[word] = {add}")
    return _F_HEAD + "\n".join(lines) + "\n    return row\n"


_F = _f()

_S_HEAD = ("from text.logistic import gradients\n\n\ndef step(examples, weights, bias, rate):\n"
           "    # one step of gradient descent for every weight and for the bias, from their slopes;\n"
           "    # gives back the new weights and the new bias\n"
           "    grad, grad_bias = gradients(examples, weights, bias)\n    new_weights = {}\n")
_S_TAIL = ("    return new_weights, new_bias\n\n\ndef train(examples, steps, rate):\n"
           "    # weight 0 for every word of the examples, and bias 0; then `steps` steps\n"
           "    weights = {}\n    for row, label in examples:\n        for word in row:\n            weights[word] = 0.0\n"
           "    bias = 0.0\n    for s in range(steps):\n        weights, bias = step(examples, weights, bias, rate)\n"
           "    return weights, bias\n")
_S_START = _S_HEAD + "    ...\n" + _S_TAIL


def _s(weight: str = "weights[word] - rate * grad[word]", bias: str = "bias - rate * grad_bias", rows: bool = False) -> str:
    loop = ("    for row, label in examples:\n        for word in row:\n            new_weights[word] = " + weight + "\n"
            if rows else "    for word in weights:\n        new_weights[word] = " + weight + "\n")
    return _S_HEAD + loop + f"    new_bias = {bias}\n" + _S_TAIL


_S = _s()

_M_HEAD = ("from text.logistic import probability\n\n\ndef mistakes(examples, weights, bias, threshold):\n"
           "    # how many spam messages the filter lets through, and how many other messages it blocks;\n"
           "    # it blocks a message whose probability of spam is threshold or more\n"
           "    missed = 0\n    blocked = 0\n    for row, label in examples:\n"
           "        p = probability(row, weights, bias)\n")
_M_START = _M_HEAD + "        ...\n    return missed, blocked\n"


def _m(missed: str = "label == 1 and p < threshold", blocked: str = "label == 0 and p >= threshold",
       end: str = "    return missed, blocked\n") -> str:
    return (_M_HEAD + f"        if {missed}:\n            missed = missed + 1\n"
            f"        if {blocked}:\n            blocked = blocked + 1\n" + end)


_M = _m()


def _not_dict(got, text) -> bool:
    return not isinstance(got, dict)


def _not_pair(got, text) -> bool:
    return not (isinstance(got, tuple) and len(got) == 2)


def _not_counts(got, text) -> bool:
    return isinstance(got, tuple) and len(got) == 2 and not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in got)


def _show_row(v):
    """A row as the check's messages write it: every number with three decimals."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {float(x):.3f}" if _number(x) else f"{k!r}: {x!r}" for k, x in v.items()) + "}"
    return repr(v)


def _show_step(v):
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], dict) and _number(v[1]):
        return f"({_show_row(v[0])}, {float(v[1]):.3f})"
    return _show_row(v)


# --- the pictures under the checks -------------------------------------------------------------------------------------

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
    """A whole number with a separator for thousands: 4,137 in English, 4.137 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.52 in English, 0,52 in Dutch; a minus sign for a negative number."""
    text = f"{float(x) + 0.0:.{digits}f}".replace("-", "−")
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
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(make, seconds)
    except Exception:   # noqa: BLE001   also Timeout; the check above says what went wrong
        return _STOPPED


def _same(a, b) -> bool:
    return _number(a) and _number(b) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _value_chip(mine, want, lang: str, style: str, digits: int = 3) -> str:
    """Your number in its role's colour when it is right; else in purple, with the right one dashed beside it."""
    if _same(mine, want):
        return _chip(_dec(mine, lang, digits), style)
    got = _chip(_dec(mine, lang, digits) if _number(mine) else str(mine), _CHIP_PURPLE)
    return got + _chip(_dec(want, lang, digits), _CHIP_MISSING)


_SAY = {
    "en": {"stopped": "Your function stopped with an error on this case, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op dit geval, dus de afbeelding blijft leeg."},
}

_FEAT_TEXT = {
    "en": {"title": "Your TF-IDF rows",
           "small": "`features(small_words, small_df, 8)`: the message *free prize free call tonight*, with the eight small messages",
           "real": "`features([\"call\", \"free\", \"call\"], real_df, 4137)`: three words of the real training messages",
           "head": ["Word", "Count", "df", "Your value"],
           "none": "no column", "extra": "a column it should not have", "missing": "missing",
           "note": "Gray: the words, how often each is in the message, in how many messages it is, and a value that is "
                   "right. Faded: a word that df does not know; it gets no column. Purple: a value that should be "
                   "another one, or a column that should not be there; dashed purple: what should be there instead.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je TF-IDF-rijen",
           "small": "`features(small_words, small_df, 8)`: het bericht *free prize free call tonight*, met de acht kleine berichten",
           "real": "`features([\"call\", \"free\", \"call\"], real_df, 4137)`: drie woorden van de echte trainingsberichten",
           "head": ["Woord", "Aantal", "df", "Je waarde"],
           "none": "geen kolom", "extra": "een kolom die er niet hoort", "missing": "ontbreekt",
           "note": "Grijs: de woorden, hoe vaak elk in het bericht staat, in hoeveel berichten het staat, en een waarde "
                   "die klopt. Vaag: een woord dat df niet kent; het krijgt geen kolom. Paars: een waarde die anders "
                   "moet, of een kolom die er niet hoort; paars gestreept: wat er in plaats daarvan hoort.",
           "alt": "{title}. {rows}. {note}"},
}


def _row_table(words: list, df: dict, n: int, got, lang: str) -> tuple:
    w = _FEAT_TEXT[lang]
    want = features(words, df, n)
    rows_, alts = [], []
    for word in dict.fromkeys(words):
        count = words.count(word)
        known = word in df
        if known and word in got:
            cell = _value_chip(got[word], want[word], lang, _CHIP_DATA)
        elif known:
            cell = _chip(w["missing"], _CHIP_PURPLE) + _chip(_dec(want[word], lang, 3), _CHIP_MISSING)
        elif word in got:
            cell = _chip(_dec(got[word], lang, 3) if _number(got[word]) else str(got[word]), _CHIP_PURPLE) + \
                _chip(w["extra"], _CHIP_MISSING)
        else:
            cell = _chip(w["none"], _CHIP_FADED)
        rows_.append([_chip(word, _CHIP_DATA if known else _CHIP_FADED), _num(count, lang),
                      _num(df[word], lang) if known else "–", cell])
        mine = got.get(word)
        alts.append(f"{word} ×{count}, df {df[word] if known else '–'}: "
                    + (_dec(mine, lang, 3) if _number(mine) else w["none"] if not known else w["missing"]))
    for word in got:
        if word not in words:
            rows_.append([_chip(word, _CHIP_PURPLE), "0", "–", _chip(w["extra"], _CHIP_MISSING)])
            alts.append(f"{word}: {w['extra']}")
    return _table(rows_, w["head"], numbers=(1, 2, 3)), "; ".join(alts)


def features_picture(seen) -> str:
    lang = seen.lang
    w = _FEAT_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    parts_, alts = [], []
    for key, words, df, n in (("small", SMALL_WORDS, SMALL_DF, 8), ("real", ["call", "free", "call"], REAL_DF, 4137)):
        got = _call(lambda: fn(list(words), dict(df), n))
        if got is _STOPPED or not isinstance(got, dict):
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"]) if got is _STOPPED else ""
        table, alt = _row_table(words, df, n, got, lang)
        parts_.append(_label(w[key].replace("`", "")) + table)
        alts.append(f"{w[key].replace('`', '')}: {alt}")
    return yours.figure(w["title"], "".join(parts_) + _note(w["note"]),
                        alt=w["alt"].format(title=w["title"], rows=". ".join(alts), note=w["note"]))


_STEP_TEXT = {
    "en": {"title": "Your step on the small examples",
           "first": "`step(small_examples, start_weights, 0.0, 1.0)`: from weight 0",
           "second": "`step(small_examples, small_weights, -0.6, 0.5)`: from the small weights, learning rate 0.5",
           "head": ["", "Before", "Slope", "Your new value"],
           "bias": "bias",
           "note": "Orange: the weights and the bias, before and after your step. Gray: the slopes of gradients. "
                   "Purple: a value that should be another one; dashed: the right value.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je stap op de kleine voorbeelden",
           "first": "`step(small_examples, start_weights, 0.0, 1.0)`: vanaf weight 0",
           "second": "`step(small_examples, small_weights, -0.6, 0.5)`: vanaf de kleine weights, learning rate 0,5",
           "head": ["", "Ervoor", "Helling", "Je nieuwe waarde"],
           "bias": "bias",
           "note": "Oranje: de weights en de bias, voor en na je stap. Grijs: de hellingen van gradients. Paars: een "
                   "waarde die anders moet; gestreept: de goede waarde.",
           "alt": "{title}. {rows}. {note}"},
}


def step_picture(seen) -> str:
    lang = seen.lang
    w = _STEP_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    parts_, alts = [], []
    for key, ws, b, rate in (("first", START_WEIGHTS, 0.0, 1.0), ("second", SMALL_WEIGHTS, SMALL_BIAS, 0.5)):
        examples = [(dict(r), label) for r, label in SMALL_EXAMPLES]
        got = _call(lambda: fn(examples, dict(ws), b, rate))
        if got is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        if not (isinstance(got, tuple) and len(got) == 2 and isinstance(got[0], dict)):
            return ""
        grad, grad_b = gradients(SMALL_EXAMPLES, ws, b)
        want, want_b = step(SMALL_EXAMPLES, ws, b, rate)
        rows_, said = [], []
        for word in ws:
            mine = got[0].get(word, "–")
            rows_.append([_chip(word), _chip(_dec(ws[word], lang, 3), _CHIP_PARAM), _dec(grad[word], lang, 3),
                          _value_chip(mine, want[word], lang, _CHIP_PARAM)])
            said.append(f"{word} {_dec(ws[word], lang, 3)} → {_dec(mine, lang, 3) if _number(mine) else mine}")
        rows_.append([_chip(w["bias"]), _chip(_dec(b, lang, 3), _CHIP_PARAM), _dec(grad_b, lang, 3),
                      _value_chip(got[1], want_b, lang, _CHIP_PARAM)])
        said.append(f"{w['bias']} {_dec(b, lang, 3)} → {_dec(got[1], lang, 3) if _number(got[1]) else got[1]}")
        parts_.append(_label(w[key].replace("`", "")) + _table(rows_, w["head"], numbers=(1, 2, 3)))
        alts.append(f"{w[key].replace('`', '')}: " + ", ".join(said))
    return yours.figure(w["title"], "".join(parts_) + _note(w["note"]),
                        alt=w["alt"].format(title=w["title"], rows=". ".join(alts), note=w["note"]))


_JUDGE_TEXT = {
    "en": {"title": "Your mistakes on the six examples",
           "panel": "more_examples, small_weights, bias −0.6", "x": "probability of spam", "spam": "spam", "ham": "not spam",
           "line": "threshold {t}",
           "head": ["Call", "Yours", "Right"],
           "note": "A dot per example, at its probability of spam; the dashed line is the threshold 0.5. Gray: a "
                   "message the filter treats right. Purple: spam to the left of the line, let through, and a message "
                   "that is not spam on the line or to its right, blocked. In the table, purple: an answer that should "
                   "be another one; dashed: the right one.",
           "alt": "{title}. Probabilities: {dots}. {rows}. {note}"},
    "nl": {"title": "Je mistakes op de zes voorbeelden",
           "panel": "more_examples, small_weights, bias −0,6", "x": "kans op spam", "spam": "spam", "ham": "geen spam",
           "line": "threshold {t}",
           "head": ["Aanroep", "Van jou", "Goed"],
           "note": "Een stip per voorbeeld, bij de kans op spam; de gestreepte lijn is de threshold 0,5. Grijs: een "
                   "bericht dat het filter goed behandelt. Paars: spam links van de lijn, doorgelaten, en een bericht "
                   "dat geen spam is op de lijn of rechts ervan, tegengehouden. In de tabel, paars: een antwoord dat "
                   "anders moet; gestreept: het goede.",
           "alt": "{title}. Kansen: {dots}. {rows}. {note}"},
}

_JUDGE_CASES = [("more_examples, small_weights, -0.6, 0.5", MORE_EXAMPLES, SMALL_WEIGHTS, SMALL_BIAS, 0.5),
                ("more_examples, small_weights, -0.6, 0.3", MORE_EXAMPLES, SMALL_WEIGHTS, SMALL_BIAS, 0.3),
                ("more_examples, small_weights, -0.6, 0.8", MORE_EXAMPLES, SMALL_WEIGHTS, SMALL_BIAS, 0.8),
                ("[({}, 0), ({}, 1)], small_weights, 0.0, 0.5", EMPTY_EXAMPLES, SMALL_WEIGHTS, 0.0, 0.5)]


def _strip(points: list, threshold: float, lang: str, title: str, width: int = 520, height: int = 200) -> str:
    """Messages as dots on two rows, spam above and not spam below, at their probability of spam, with the threshold
    as a dashed line: purple for a mistake at that threshold, gray for the others. `points` holds (p, label, y), y a
    small shift within the row so that dots with almost the same probability stay apart."""
    w = _JUDGE_TEXT[lang]
    panel = chart.Panel((0, 1), (-0.6, 1.9), title=title, xlabel=w["x"], yticks=[])
    for role, wrong in (("data", False), ("loss", True)):
        xs, ys, seen = [], [], set()
        for p, label, y in points:
            mistake = (label == 1 and p < threshold) or (label == 0 and p >= threshold)
            spot = (round(p * 500), round((label + y) * 40))   # dots that would fall on one another are drawn once
            if mistake == wrong and spot not in seen:
                seen.add(spot)
                xs.append(p)
                ys.append(label + y)
        if xs:
            panel.points(xs, ys, role, r=3.2)
    panel.line([threshold, threshold], [-0.6, 1.9], "ink-2", 1.4, dashed=True)
    panel.text(0.01, 1.62, w["spam"], "ink-2", anchor="start")
    panel.text(0.01, 0.62, w["ham"], "ink-2", anchor="start")
    svg = chart.figure([panel], alt=title, width=width, height=height)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def mistakes_picture(seen) -> str:
    lang = seen.lang
    w = _JUDGE_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    rows_, alts = [], []
    for call, examples, ws, b, t in _JUDGE_CASES:
        got = _call(lambda: fn([(dict(r), label) for r, label in examples], dict(ws), b, t))
        if got is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        want = mistakes(examples, ws, b, t)
        ok = isinstance(got, tuple) and len(got) == 2 and all(_number(v) for v in got) and \
            all(int(g) == v and float(g) == v for g, v in zip(got, want))
        shown_got = f"({', '.join(str(int(v)) if _number(v) and float(v) == int(v) else str(v) for v in got)})" \
            if isinstance(got, tuple) else repr(got)[:40]
        call_text = call.replace("-0.6", "−0,6" if lang == "nl" else "−0.6")
        if lang == "nl":
            call_text = call_text.replace("0.5", "0,5").replace("0.3", "0,3").replace("0.8", "0,8").replace("0.0", "0,0")
        rows_.append([f'<span style="{_TEXT};font-family:var(--ea-font-code)">mistakes({html.escape(call_text)})</span>',
                      _chip(shown_got, _CHIP_DATA if ok else _CHIP_PURPLE),
                      _chip(f"({want[0]}, {want[1]})", _CHIP_DATA if ok else _CHIP_MISSING)])
        alts.append(f"mistakes({call_text}): {shown_got}, {want}")
    points = [(float(probability(r, SMALL_WEIGHTS, SMALL_BIAS)), label, 0.18 * ((k % 3) - 1))
              for k, (r, label) in enumerate(MORE_EXAMPLES)]
    strip = _strip(points, 0.5, lang, w["panel"], width=520, height=190)
    dots = ", ".join(f"{_dec(p, lang, 3)} ({w['spam'] if label else w['ham']})" for p, label, _ in points)
    return yours.figure(w["title"], strip + _table(rows_, w["head"]) + _note(w["note"]),
                        alt=w["alt"].format(title=w["title"], dots=dots, rows="; ".join(alts), note=w["note"]))


# --- the exercises -----------------------------------------------------------------------------------------------------

_SPAM_BLOCKED = {   # one comparison of mistakes turned round, which the example at 0.5 does not show
    "en": "`{call}` gives {got}: your first number counts the spam your filter blocks, not the spam it lets through. Spam gets through when its `p` is below the threshold.",
    "nl": "`{call}` geeft {got}: je eerste getal telt de spam die je filter tegenhoudt, niet de spam die het doorlaat. Spam komt erdoor als `p` onder de threshold ligt.",
}
_HAM_THROUGH = {
    "en": "`{call}` gives {got}: your second number counts the other messages your filter lets through, not the ones it blocks. Which messages that are not spam does the filter block?",
    "nl": "`{call}` geeft {got}: je tweede getal telt de andere berichten die je filter doorlaat, niet de berichten die het tegenhoudt. Welke berichten die geen spam zijn, houdt het filter tegen?",
}


REQUIRED = ["features", "step", "mistakes", "phone"]

EXERCISES = {
    "features": FunctionBox(
        name="features",
        picture=features_picture,
        function="features",
        start=_F_START,
        solution=_F,
        show=_show_row,
        save_as=("text", "features"),
        visible=[(_SW, _SDF, 8)],
        hidden=[(["tonight", "maybe"], _SDF, 8), (["call", "free", "call"], _Val("real_df", REAL_DF), 4137), ([], _SDF, 8)],
        first=[_DOTS],
        mistakes=[
            Mistake("unknown", error=lambda e: isinstance(e, KeyError) and e.args[:1] in [(u,) for u in UNKNOWN],
                    en="`{call}` stops with a KeyError: a word of the message, such as `tonight` in `small_words`, is in no message that `df` counted, so `df` does not know it. Skip such a word with an `if`: the filter has no weight for it, so it gets no column.",
                    nl="`{call}` stopt met een KeyError: een woord van het bericht, zoals `tonight` in `small_words`, staat in geen enkel bericht dat `df` telde, dus `df` kent het niet. Sla zo'n woord over met een `if`: het filter heeft er geen weight voor, dus het krijgt geen kolom."),
            Mistake("no-math", error=lambda e: isinstance(e, NameError) and "'math'" in str(e),
                    en="`{call}` stops: the box does not import `math`. Add `import math` at the top, or use NumPy, which the box imports as `np`.",
                    nl="`{call}` stopt: het vak importeert `math` niet. Zet `import math` bovenaan, of gebruik NumPy, dat het vak importeert als `np`."),
            Mistake("first-time", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a KeyError on `row`: the first time a word comes along, `row` does not have it yet, so `row[word]` on the right of `=` does not exist. Give the word the value `0.0` first, when it is not in `row` yet.",
                    nl="`{call}` stopt met een KeyError op `row`: de eerste keer dat een woord langskomt, heeft `row` het nog niet, dus `row[word]` rechts van de `=` bestaat niet. Geef het woord eerst de waarde `0.0`, als het nog niet in `row` staat."),
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give a dictionary: for every word of the message that `df` knows, its TF-IDF value. Keep `return row` as the last line, with `row` a dictionary.",
                    nl="`{call}` geeft {got}, en het moet een dictionary zijn: voor elk woord van het bericht dat `df` kent, de TF-IDF-waarde. Houd `return row` als laatste regel, met `row` een dictionary."),
            Mistake("counts", wrong=_f("row[word] + 1"),
                    en="`{call}` gives {got}: how often each word is in the message. That is only the first of the two factors. Each time the word comes along, add its inverse document frequency instead of 1, so that a word in many messages counts for less.",
                    nl="`{call}` geeft {got}: hoe vaak elk woord in het bericht staat. Dat is alleen de eerste van de twee factoren. Tel elke keer dat het woord langskomt de inverse document frequency op in plaats van 1, dan telt een woord dat in veel berichten staat minder."),
            Mistake("once", wrong=_f("np.log10(n / df[word])"),
                    en="`{call}` gives {got}: *free* is twice in `small_words` and gets the same value as a word that is there once. Each time the word comes along, add to what `row[word]` already has, so that the count is in it.",
                    nl="`{call}` geeft {got}: *free* staat twee keer in `small_words` en krijgt dezelfde waarde als een woord dat er één keer staat. Tel elke keer dat het woord langskomt op bij wat `row[word]` al heeft, dan zit het aantal erin."),
            Mistake("ln", wrong=_f("row[word] + np.log(n / df[word])"),
                    en="`{call}` gives {got}: values that are 2.3 times too large. `np.log` is the natural logarithm; TF-IDF in this course takes the logarithm with base 10, `np.log10`.",
                    nl="`{call}` geeft {got}: waarden die 2,3 keer te groot zijn. `np.log` is de natuurlijke logaritme; TF-IDF neemt in deze course de logaritme met grondtal 10, `np.log10`."),
            Mistake("upside-down", wrong=_f("row[word] + np.log10(df[word] / n)"),
                    en="`{call}` gives {got}: values below 0. The inverse document frequency grows when fewer messages have the word: which of the two numbers goes on top?",
                    nl="`{call}` geeft {got}: waarden onder 0. De inverse document frequency wordt groter als minder berichten het woord hebben: welk van de twee getallen hoort boven de streep?"),
            Mistake("vocab", wrong=_f("row[word] + np.log10(len(df) / df[word])"),
                    en="`{call}` gives {got}: `len(df)` is how many different words `df` knows, not how many messages there are. The number of messages is `n`.",
                    nl="`{call}` geeft {got}: `len(df)` is hoeveel verschillende woorden `df` kent, niet hoeveel berichten er zijn. Het aantal berichten is `n`."),
            Mistake("share", wrong=_f("row[word] + np.log10(n / df[word]) / len(words)"),
                    en="`{call}` gives {got}: you divide by the length of the message. In this course the term frequency is the count itself, as in *TF-IDF: weighing the words*: do not divide by `len(words)`.",
                    nl="`{call}` geeft {got}: je deelt door de lengte van het bericht. In deze course is de term frequency het aantal zelf, zoals in *TF-IDF: woorden wegen*: deel niet door `len(words)`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the lines of `features`: a dictionary `row`, a loop over `words`, and `return row`.",
                      "nl": "Vervang de `...` door de regels van `features`: een dictionary `row`, een loop over `words`, en `return row`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows the values per word.",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. De afbeelding onder dit bericht laat de waarden per woord zien."},
            "almost": {"en": "Right for `features(small_words, small_df, 8)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `df` and the `n` it gets, and does it work for a message without a word that `df` knows?",
                       "nl": "Goed voor `features(small_words, small_df, 8)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `df` en de `n` uit de aanroep, en werkt de functie ook voor een bericht waarvan `df` geen enkel woord kent?"},
            "correct": {"en": "Your `features` is right in all {n} cases. It is now in your project, in `{saved}`, together with `document_frequencies`.",
                        "nl": "Je `features` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`, samen met `document_frequencies`."},
            "kept": _kept("features"),
        },
    ),
    "step": FunctionBox(
        name="step",
        picture=step_picture,
        function="step",
        start=_S_START,
        solution=_S,
        show=_show_step,
        save_as=("text", "train"),
        visible=[(_SE, _ST, 0.0, 1.0)],
        hidden=[(_SE, _SWT, SMALL_BIAS, 0.5), ([({"free": 0.602}, 1)], _Val("two_weights", {"free": 1.2, "prize": 2.0}), -0.6, 2.0)],
        first=[_DOTS],
        mistakes=[
            _no_module("gradients"),
            Mistake("no-new-bias", error=lambda e: isinstance(e, NameError) and "new_bias" in str(e),
                    en="`{call}` stops at `return new_weights, new_bias`: your function never makes `new_bias`. After the loop, add a line that makes `new_bias` from `bias`, `rate` and `grad_bias`.",
                    nl="`{call}` stopt bij `return new_weights, new_bias`: je functie maakt `new_bias` nooit. Voeg na de loop een regel toe die `new_bias` maakt uit `bias`, `rate` en `grad_bias`."),
            Mistake("not-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give two things: the new weights and the new bias. Keep `return new_weights, new_bias` as the last line of `step`.",
                    nl="`{call}` geeft {got}, en het moet twee dingen geven: de nieuwe weights en de nieuwe bias. Houd `return new_weights, new_bias` als laatste regel van `step`."),
            Mistake("empty-weights", matches=lambda got, text: isinstance(got, tuple) and len(got) == 2 and got[0] == {},
                    en="`{call}` gives {got}: `new_weights` stays empty. Give `new_weights[word]` a value for every word of `weights`, in a loop.",
                    nl="`{call}` geeft {got}: `new_weights` blijft leeg. Geef `new_weights[word]` een waarde voor elk woord van `weights`, in een loop."),
            Mistake("up", wrong=_s("weights[word] + rate * grad[word]", "bias + rate * grad_bias"),
                    en="`{call}` gives {got}: every weight and the bias move with their slope, uphill, so the loss grows. Gradient descent goes against the slope, downhill.",
                    nl="`{call}` geeft {got}: elke weight en de bias bewegen met de helling mee, bergop, dus de loss groeit. Gradient descent gaat tegen de helling in, bergaf."),
            Mistake("up-weights", wrong=_s("weights[word] + rate * grad[word]"),
                    en="`{call}` gives {got}: the bias goes against its slope, but the weights move with theirs, uphill. Every weight goes against its slope too.",
                    nl="`{call}` geeft {got}: de bias gaat tegen de helling in, maar de weights bewegen met de eigen helling mee, bergop. Ook elke weight gaat tegen de helling in."),
            Mistake("no-rate", wrong=_s("weights[word] - grad[word]", "bias - grad_bias"),
                    en="`{call}` gives {got}: right with learning rate 1.0, but this case has another `rate`. The learning rate decides how big the step is: where does `rate` come into yours?",
                    nl="`{call}` geeft {got}: goed bij learning rate 1,0, maar dit geval heeft een andere `rate`. De learning rate bepaalt hoe groot de stap is: waar komt `rate` in je stap?"),
            Mistake("bias-stays", wrong=_s(bias="bias"),
                    en="`{call}` gives {got}: the weights move, but the bias stays where it was. The bias learns too, against its own slope, `grad_bias`.",
                    nl="`{call}` geeft {got}: de weights bewegen, maar de bias verandert niet. De bias leert ook, tegen de eigen helling in, `grad_bias`."),
            Mistake("bias-slope", wrong=_s("weights[word] - rate * grad_bias"),
                    en="`{call}` gives {got}: every weight takes the step of the bias. `grad_bias` is the slope of the bias only; the slope of each weight is in `grad`.",
                    nl="`{call}` geeft {got}: elke weight zet de stap van de bias. `grad_bias` is alleen de helling van de bias; de helling van elke weight staat in `grad`."),
            Mistake("rows", wrong=_s(rows=True),
                    en="`{call}` gives {got}: a word that no example has is missing from your new weights. Loop over `weights`, not over the words of the examples: every word keeps a weight, and a word without slope keeps the one it had.",
                    nl="`{call}` geeft {got}: een woord dat geen enkel voorbeeld heeft, ontbreekt in je nieuwe weights. Loop over `weights`, niet over de woorden van de voorbeelden: elk woord houdt een weight, en een woord zonder helling houdt de weight die het had."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a loop that fills `new_weights`, and a line that makes `new_bias`.",
                      "nl": "Vervang de `...` door een loop die `new_weights` vult, en een regel die `new_bias` maakt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows each weight before and after your step.",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. De afbeelding onder dit bericht laat elke weight zien voor en na je stap."},
            "almost": {"en": "Right for `step(small_examples, start_weights, 0.0, 1.0)`. But `{call}` gives {got}, and it should give {want}. Does your step use the `rate` and the bias it gets, and keep every word of `weights`?",
                       "nl": "Goed voor `step(small_examples, start_weights, 0.0, 1.0)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je stap de `rate` en de bias uit de aanroep, en houdt je stap elk woord van `weights`?"},
            "correct": {"en": "Your `step` is right in all {n} cases. It is now in your project, in `{saved}`, together with `train`, which calls it.",
                        "nl": "Je `step` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`, samen met `train`, de functie die `step` aanroept."},
            "kept": _kept("step"),
        },
    ),
    "mistakes": FunctionBox(
        name="mistakes",
        picture=mistakes_picture,
        function="mistakes",
        start=_M_START,
        solution=_M,
        save_as=("text", "judge"),
        visible=[(_ME, _SWT, SMALL_BIAS, 0.5)],
        hidden=[(_ME, _SWT, SMALL_BIAS, 0.3), (_ME, _SWT, SMALL_BIAS, 0.8), (EMPTY_EXAMPLES, _SWT, 0.0, 0.5)],
        first=[_DOTS],
        mistakes=[
            _no_module("probability"),
            Mistake("not-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give two numbers: the spam messages let through, and the other messages blocked. Keep `return missed, blocked` as the last line.",
                    nl="`{call}` geeft {got}, en het moet twee getallen geven: de doorgelaten spamberichten, en de tegengehouden andere berichten. Houd `return missed, blocked` als laatste regel."),
            Mistake("not-counts", matches=_not_counts,
                    en="`{call}` gives {got}, and it should give two whole numbers: how many messages, not a share or a probability. Add 1 to `missed` or `blocked` for each message that counts.",
                    nl="`{call}` geeft {got}, en het moet twee hele getallen geven: hoeveel berichten, geen deel en geen kans. Tel 1 op bij `missed` of `blocked` voor elk bericht dat meetelt."),
            Mistake("swapped", wrong=_m(end="    return blocked, missed\n"),
                    en="`{call}` gives {got}: the two numbers are the wrong way round. First the spam messages let through, `missed`, then the blocked messages, `blocked`.",
                    nl="`{call}` geeft {got}: de twee getallen staan andersom. Eerst de doorgelaten spamberichten, `missed`, dan de tegengehouden berichten, `blocked`."),
            Mistake("caught", wrong=_m("label == 1 and p >= threshold", "label == 0 and p < threshold"),
                    en="`{call}` gives {got}: the messages your filter treats right, spam it blocks and other messages it lets through. Count the mistakes: spam with a probability below the threshold, and other messages at the threshold or above.",
                    nl="`{call}` geeft {got}: de berichten die je filter goed behandelt, spam die het tegenhoudt en andere berichten die het doorlaat. Tel de fouten: spam met een kans onder de threshold, en andere berichten op de threshold of erboven."),
            Mistake("spam-blocked", wrong=_m(missed="label == 1 and p >= threshold"), **_SPAM_BLOCKED),
            Mistake("spam-blocked-strict", wrong=_m(missed="label == 1 and p > threshold"), **_SPAM_BLOCKED),
            Mistake("ham-through", wrong=_m(blocked="label == 0 and p < threshold"), **_HAM_THROUGH),
            Mistake("ham-through-strict", wrong=_m(blocked="label == 0 and p <= threshold"), **_HAM_THROUGH),
            Mistake("half", wrong=_m("label == 1 and p < 0.5", "label == 0 and p >= 0.5"),
                    en="`{call}` gives {got}: right at 0.5, but this case has another threshold. Compare `p` with `threshold`, not with 0.5.",
                    nl="`{call}` geeft {got}: goed bij 0,5, maar dit geval heeft een andere threshold. Vergelijk `p` met `threshold`, niet met 0,5."),
            Mistake("strict", wrong=_m("label == 1 and p <= threshold", "label == 0 and p > threshold"),
                    en="`{call}` gives {got}: both messages get a probability of exactly 0.5, the threshold. A message at the threshold is blocked, so the spam is not let through and the other message is blocked: use `<` for `missed` and `>=` for `blocked`.",
                    nl="`{call}` geeft {got}: beide berichten krijgen een kans van precies 0,5, de threshold. Een bericht op de threshold wordt tegengehouden, dus de spam wordt niet doorgelaten en het andere bericht wordt tegengehouden: gebruik `<` voor `missed` en `>=` voor `blocked`."),
            Mistake("strict-blocked", wrong=_m(blocked="label == 0 and p > threshold"),
                    en="`{call}` gives {got}: the message that is not spam gets a probability of exactly 0.5, the threshold, and your filter lets it through. A message at the threshold is blocked: `>=`, not `>`.",
                    nl="`{call}` geeft {got}: het bericht dat geen spam is, krijgt een kans van precies 0,5, de threshold, en je filter laat het door. Een bericht op de threshold wordt tegengehouden: `>=`, niet `>`."),
            Mistake("strict-missed", wrong=_m(missed="label == 1 and p <= threshold"),
                    en="`{call}` gives {got}: the spam message gets a probability of exactly 0.5, the threshold, and your function counts it as let through. A message at the threshold is blocked, so only spam below the threshold gets through: `<`, not `<=`.",
                    nl="`{call}` geeft {got}: het spambericht krijgt een kans van precies 0,5, de threshold, en je functie telt het als doorgelaten. Een bericht op de threshold wordt tegengehouden, dus alleen spam onder de threshold komt erdoor: `<`, niet `<=`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two `if` lines, each with a line under it that adds 1: one for `missed`, one for `blocked`.",
                      "nl": "Vervang de `...` door twee `if`-regels, elk met een regel eronder die 1 optelt: één voor `missed`, één voor `blocked`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows each example at its probability.",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. De afbeelding onder dit bericht laat elk voorbeeld zien bij de kans."},
            "almost": {"en": "Right for `mistakes(more_examples, small_weights, -0.6, 0.5)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `threshold` it gets, and what does it do with a message right at the threshold?",
                       "nl": "Goed voor `mistakes(more_examples, small_weights, -0.6, 0.5)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `threshold` uit de aanroep, en wat doet de functie met een bericht precies op de threshold?"},
            "correct": {"en": "Your `mistakes` is right in all {n} cases. It is now in your project, in `{saved}`.",
                        "nl": "Je `mistakes` klopt in alle {n} gevallen en staat nu in je project, in `{saved}`."},
            "kept": _kept("mistakes"),
        },
    ),
}

CASES = [
    ("features", _F_START, "empty", None),
    ("features", _F, "correct", None),
    ("features", _F_HEAD + "    row = {}\n    for word in words:\n        row[word] = 0.0\n    return row\n", "wrong", None),
    ("features", _f(skip=False), "error", "unknown"),
    ("features", _f(first=False), "error", "first-time"),
    ("features", _F_HEAD + "    return [np.log10(n / df[w]) for w in words if w in df]\n", "wrong", "not-dict"),
    ("features", _f("row[word] + 1"), "wrong", "counts"),
    ("features", _f("np.log10(n / df[word])"), "wrong", "once"),
    ("features", _f("row[word] + np.log(n / df[word])"), "wrong", "ln"),
    ("features", _f("row[word] + np.log10(df[word] / n)"), "wrong", "upside-down"),
    ("features", _f("row[word] + np.log10(n / df[word]) / len(words)"), "wrong", "share"),
    ("features", _f("row[word] + np.log10(8 / df[word])"), "almost", None),
    ("features", _f("row[word] + math.log10(n / df[word])"), "error", "no-math"),
    ("features", _f("row[word] + np.log10(len(df) / df[word])"), "wrong", "vocab"),
    ("features", "import math\n" + _f("row[word] + math.log10(n / df[word])"), "correct", None),
    ("features", _F_HEAD + "    counts = {}\n    for word in words:\n        if word in df:\n"
     "            counts[word] = counts.get(word, 0) + 1\n"
     "    return {word: counts[word] * np.log10(n / df[word]) for word in counts}\n", "correct", None),
    ("step", _S_START, "empty", None),
    ("step", _S, "correct", None),
    ("step", _S_HEAD + "    for word in weights:\n        new_weights[word] = weights[word] - rate * grad[word]\n" + _S_TAIL,
     "error", "no-new-bias"),
    ("step", _S.replace("    return new_weights, new_bias\n\n\ndef train", "    return new_weights\n\n\ndef train"), "wrong", "not-pair"),
    ("step", _s("weights[word] + rate * grad[word]", "bias + rate * grad_bias"), "wrong", "up"),
    ("step", _s("weights[word] + rate * grad[word]"), "wrong", "up-weights"),
    ("step", _s("weights[word] - grad[word]", "bias - grad_bias"), "wrong", "no-rate"),
    ("step", _s(bias="bias"), "wrong", "bias-stays"),
    ("step", _s(rows=True), "wrong", "rows"),
    ("step", _S_HEAD + "    new_bias = bias - rate * grad_bias\n" + _S_TAIL, "wrong", "empty-weights"),
    ("step", _s("weights[word] - rate * grad_bias"), "wrong", "bias-slope"),
    ("step", _S_HEAD + "    for word in grad:\n        new_weights[word] = weights[word] - rate * grad[word]\n"
     "    new_bias = bias - rate * grad_bias\n" + _S_TAIL, "correct", None),
    ("step", _S_HEAD + "    for word in weights:\n        weights[word] = weights[word] - rate * grad[word]\n"
     "    new_weights = weights\n    new_bias = bias - rate * grad_bias\n" + _S_TAIL, "correct", None),
    ("step", _S.replace("from text.logistic import gradients", "from text.nowhere import gradients"), "error", "no-logistic"),
    ("mistakes", _M_START, "empty", None),
    ("mistakes", _M, "correct", None),
    ("mistakes", _M_HEAD + "        missed = missed + label\n    return missed\n", "wrong", "not-pair"),
    ("mistakes", _m(end="    return missed / len(examples), blocked / len(examples)\n"), "wrong", "not-counts"),
    ("mistakes", _m(end="    return blocked, missed\n"), "wrong", "swapped"),
    ("mistakes", _m("label == 1 and p >= threshold", "label == 0 and p < threshold"), "wrong", "caught"),
    ("mistakes", _m("label == 1 and p < 0.5", "label == 0 and p >= 0.5"), "wrong", "half"),
    ("mistakes", _m("label == 1 and p <= threshold", "label == 0 and p > threshold"), "wrong", "strict"),
    ("mistakes", _m(blocked="label == 0 and p > threshold"), "wrong", "strict-blocked"),
    ("mistakes", _m(missed="label == 1 and p <= threshold"), "wrong", "strict-missed"),
    ("mistakes", _m(missed="label == 1 and p >= threshold"), "wrong", "spam-blocked"),
    ("mistakes", _m(missed="label == 1 and p > threshold"), "wrong", "spam-blocked-strict"),
    ("mistakes", _m(blocked="label == 0 and p < threshold"), "wrong", "ham-through"),
    ("mistakes", _m(blocked="label == 0 and p <= threshold"), "wrong", "ham-through-strict"),
    ("mistakes", _M_HEAD + "        if label == 1:\n            if p < threshold:\n                missed += 1\n"
     "        elif p >= threshold:\n            blocked += 1\n    return missed, blocked\n", "correct", None),
]


# --- the demo: your filter trains on the real messages -----------------------------------------------------------------

DEMO_LIMIT = 180.0   # seconds the demo's training gets


class _Fns:
    """The functions the demo uses: those of your project when its file is there, else the course's; `yours` says
    which modules are yours."""

    def __init__(self):
        self.document_frequencies, self.features, self.step, self.mistakes = (
            document_frequencies, features, step, mistakes)
        self.yours: list[str] = []
        self.broken: list[str] = []   # a module of yours that is there but cannot be imported


def project_functions() -> "_Fns | str":
    """The functions of your project text: features.py, train.py and judge.py when they are there, the course's for a
    file that is not there yet. The name of a module that is there but cannot be imported, or lacks its function."""
    fns = _Fns()
    for module, names in (("features", ("document_frequencies", "features")), ("train", ("step",)),
                          ("judge", ("mistakes",))):
        if not (projects.folder() / "text" / f"{module}.py").exists():
            continue
        run = run_code(f"from text.{module} import {', '.join(names)}")
        if run.error is not None or not all(callable(run.names.get(n)) for n in names):
            if module == "judge":   # the training does not need it: the threshold demo counts with the course's
                fns.broken.append(module)
                continue
            return module
        for n in names:
            setattr(fns, n, run.names[n])
        fns.yours.append(module)
    return fns


class _Broken(Exception):
    def __init__(self, module: str):
        super().__init__(module)
        self.module = module


def _mean_loss(rows: list, ws: dict, b: float) -> float:
    total = 0.0
    for r, label in rows:
        z = b + sum(v * ws[w] for w, v in r.items())
        p = min(max(_sigmoid(z), 1e-15), 1 - 1e-15)
        total -= math.log(p) if label == 1 else math.log(1 - p)
    return total / len(rows)


def _train(fns: _Fns, show=None) -> dict:
    training, test = sms.training(), sms.test()
    n = len(training)
    try:
        df = fns.document_frequencies([(list(w), label) for w, label in training])
        rows = [(fns.features(list(w), df, n), label) for w, label in training]
        test_rows = [(fns.features(list(w), df, n), label) for w, label in test]
    except Timeout:
        raise
    except Exception as e:   # noqa: BLE001
        raise _Broken("features") from e
    if not all(isinstance(r, dict) and all(_number(v) for v in r.values()) for r, _ in rows + test_rows):
        raise _Broken("features")
    clean = [({w: float(v) for w, v in r.items()}, label) for r, label in rows]
    test_clean = [({w: float(v) for w, v in r.items()}, label) for r, label in test_rows]
    ws = {}
    for r, _ in clean:
        for word in r:
            ws[word] = 0.0
    b = 0.0
    curve, looks = [], {}
    for s in range(1, STEPS + 1):
        try:
            out = fns.step(rows, ws, b, RATE)
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
            show(s, loss, ws, b, test_clean)
    return {"curve": curve, "looks": looks, "weights": ws, "bias": b, "test_rows": test_clean,
            "watched": {line: _test_index(line) for line in WATCHED + [BANK_LINE]}, "fns": fns}


def demo(show=None) -> "dict | str":
    """The demo's run within DEMO_LIMIT seconds: your features and step (or the course's, for a file that is not in
    your project yet) on the real messages, with `show(step, loss, weights, bias, test_rows)` after every step. A
    string when it cannot: "data" when the messages are not downloaded, "slow", or the module that went wrong."""
    if not sms.ready():
        return "data"
    fns = project_functions()
    if isinstance(fns, str):
        return fns
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(lambda: _train(fns, show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except _Broken as e:
        return e.module


def _p_of(r: dict, ws: dict, b: float) -> float:
    return _sigmoid(b + sum(v * ws.get(w, 0.0) for w, v in r.items()))


def _counts(test_rows: list, ws: dict, b: float, threshold: float) -> tuple:
    missed = blocked = 0
    for r, label in test_rows:
        p = _p_of(r, ws, b)
        missed += label == 1 and p < threshold
        blocked += label == 0 and p >= threshold
    return missed, blocked


_LIVE_TEXT = {
    "en": {"title": "Your filter trains on 4,137 messages", "x": "step", "y": "log loss",
           "alt": "The mean log loss of the 4,137 training messages after each step of gradient descent.",
           "strong": "The strongest words now", "spam": "Towards spam", "ham": "Towards not spam",
           "watched": "Three test messages now", "message": "Message", "label": "Label", "p": "p", "verdict": "At 0.5",
           "after": "The strongest words after step {s}", "watched_after": "Three test messages after step {s}",
           "test": "All test messages at 0.5: {m} of the 139 spam messages let through, {b} of the 895 others blocked.",
           "progress": "**Training** · step {s} of {n} · log loss {loss}",
           "blocked": "blocked", "through": "through", "spam_label": "spam", "ham_label": "not spam",
           "note": "Gray: the words, the messages and their labels. Orange: the weights. Blue: the probability of "
                   "spam. Purple: a mistake at 0.5, spam let through or another message blocked."},
    "nl": {"title": "Je filter traint op 4.137 berichten", "x": "stap", "y": "log loss",
           "alt": "De gemiddelde log loss van de 4.137 trainingsberichten na elke stap gradient descent.",
           "strong": "De sterkste woorden nu", "spam": "Richting spam", "ham": "Richting geen spam",
           "watched": "Drie testberichten nu", "message": "Bericht", "label": "Label", "p": "p", "verdict": "Bij 0,5",
           "after": "De sterkste woorden na stap {s}", "watched_after": "Drie testberichten na stap {s}",
           "test": "Alle testberichten bij 0,5: {m} van de 139 spamberichten doorgelaten, {b} van de 895 andere tegengehouden.",
           "progress": "**Training** · stap {s} van {n} · log loss {loss}",
           "blocked": "tegengehouden", "through": "doorgelaten", "spam_label": "spam", "ham_label": "geen spam",
           "note": "Grijs: de woorden, de berichten en de labels. Oranje: de weights. Blauw: de kans op spam. Paars: "
                   "een fout bij 0,5, doorgelaten spam of een ander bericht dat is tegengehouden."},
}


def live_words(lang: str) -> dict:
    return dict(_LIVE_TEXT[lang])


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
    return _label(title) + _table(rows_, [w["spam"], "", "", w["ham"], "", ""], numbers=(1, 4))


def _watched_rows(run_rows: list, where: dict, lines: list, ws: dict, b: float, lang: str, threshold: float) -> tuple:
    w = _LIVE_TEXT[lang]
    rows_, alts = [], []
    for line in lines:
        r, label = run_rows[where[line]]
        p = _p_of(r, ws, b)
        blocked = p >= threshold
        wrong = blocked != (label == 1)
        verdict = _chip(w["blocked"] if blocked else w["through"], _CHIP_PURPLE if wrong else _CHIP_DATA)
        rows_.append([f'<span style="{_TEXT}">{html.escape(shown(line, lang, 46))}</span>',
                      _chip(w["spam_label"] if label else w["ham_label"]), _chip(_dec(p, lang, 3), _CHIP_PRED),
                      _bar(p, 1.0, "var(--ea-pred)", 4.0), verdict])
        alts.append(f"{shown(line, lang)} ({w['spam_label'] if label else w['ham_label']}): {_dec(p, lang, 3)}, "
                    f"{w['blocked'] if blocked else w['through']}")
    return rows_, alts


def now_parts(run: dict, ws: dict, b: float, lang: str, step_: int | None = None) -> list:
    """What the demo shows beside its chart: the five strongest words each way, three test messages with their
    probability of spam, and the mistakes on all test messages at 0.5, now (or after `step_`, for the rewind)."""
    w = _LIVE_TEXT[lang]
    strong = w["strong"] if step_ is None else w["after"].format(s=step_)
    watched = w["watched"] if step_ is None else w["watched_after"].format(s=step_)
    rows_, _ = _watched_rows(run["test_rows"], run["watched"], WATCHED, ws, b, lang, THRESHOLD)
    m, bl = _counts(run["test_rows"], ws, b, THRESHOLD)
    test_line = w["test"].format(m=m, b=bl)
    return [_strong_table(ws, lang, strong),
            _label(watched) + _table(rows_, [w["message"], w["label"], w["p"], "", w["verdict"]], numbers=(2,))
            + _note(test_line) + _note(w["note"])]


def live_parts(test_rows: list, ws: dict, b: float, lang: str) -> list:
    """now_parts while the run is still going, from the test rows the demo's show() gets."""
    return now_parts({"test_rows": test_rows, "watched": {line: _test_index(line) for line in WATCHED}}, ws, b, lang)


def curve_picture(curve: list, lang: str, at: int | None = None) -> str:
    """The log loss after each step of the run; with `at`, a line at that step, for the rewind."""
    w = _LIVE_TEXT[lang]
    top = 0.7
    p = chart.Panel((0, STEPS), (0, top), title=w["title"], xlabel=w["x"], ylabel=w["y"])
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    p.line([x for x, _ in curve], [min(y, top) for _, y in curve], "loss", 2.2)
    svg = chart.figure([p], alt=w["alt"], width=live.CHART_WIDTH, height=240)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_frame(run: dict, step_: int, lang: str) -> str:
    """The rewind at a step: the curve with a line at that step, and beside it what the filter made then."""
    ws, b = run["looks"][step_]
    return live.rewound(curve_picture(run["curve"], lang, at=step_), now_parts(run, ws, b, lang, step_))


def rewind_counts(run: dict, step_: int) -> tuple:
    ws, b = run["looks"][step_]
    return _counts(run["test_rows"], ws, b, THRESHOLD)


def strongest(run: dict, k: int = 6) -> tuple:
    return _strongest(run["weights"], k)


# --- what your project can do now: the threshold ----------------------------------------------------------------------

_THRESH_TEXT = {
    "en": {"title": "Your filter on the 1,034 test messages at threshold {t}",
           "line": "With your mistakes: {m} of the 139 spam messages let through, {b} of the 895 others blocked. Naive Bayes: {nm} let through, {nb} blocked.",
           "course_line": "With the course's mistakes, until exercise 3 is in your project: {m} of the 139 spam messages let through, {b} of the 895 others blocked. Naive Bayes: {nm} let through, {nb} blocked.",
           "judge_line": "With the course's mistakes, since your judge.py stops with an error: {m} of the 139 spam messages let through, {b} of the 895 others blocked. Naive Bayes: {nm} let through, {nb} blocked.",
           "four": "Four test messages at this threshold",
           "note": "A dot per test message, at its probability of spam. Gray: a message the filter treats right. "
                   "Purple: spam let through, to the left of the line, and another message blocked, on the line or to "
                   "its right. Blue: the probability of spam."},
    "nl": {"title": "Je filter op de 1.034 testberichten bij threshold {t}",
           "line": "Met je mistakes: {m} van de 139 spamberichten doorgelaten, {b} van de 895 andere tegengehouden. Naive Bayes: {nm} doorgelaten, {nb} tegengehouden.",
           "course_line": "Met de mistakes van de course, tot oefening 3 in je project staat: {m} van de 139 spamberichten doorgelaten, {b} van de 895 andere tegengehouden. Naive Bayes: {nm} doorgelaten, {nb} tegengehouden.",
           "judge_line": "Met de mistakes van de course, omdat je judge.py stopt met een fout: {m} van de 139 spamberichten doorgelaten, {b} van de 895 andere tegengehouden. Naive Bayes: {nm} doorgelaten, {nb} tegengehouden.",
           "four": "Vier testberichten bij deze threshold",
           "note": "Een stip per testbericht, bij de kans op spam. Grijs: een bericht dat het filter goed behandelt. "
                   "Paars: doorgelaten spam, links van de lijn, en een ander bericht dat is tegengehouden, op de lijn "
                   "of rechts ervan. Blauw: de kans op spam."},
}


def threshold_frame(run: dict, t: float, lang: str) -> tuple:
    """The explore at a threshold: every test message as a dot at its probability, with the mistakes in purple; four
    messages the lesson chose; and the sentence with what your mistakes counts, beside naive Bayes."""
    w = _THRESH_TEXT[lang]
    ws, b = run["weights"], run["bias"]
    rows = run["test_rows"]
    points = [(_p_of(r, ws, b), label, 0.32 * (((k * 7919) % 101) / 100 - 0.5)) for k, (r, label) in enumerate(rows)]
    t_text = _dec(t, lang, 2)
    strip = _strip(points, t, lang, w["title"].format(t=t_text), width=640, height=230)
    table_rows, _ = _watched_rows(rows, run["watched"], WATCHED + [BANK_LINE], ws, b, lang, t)
    lw = _LIVE_TEXT[lang]
    body = strip + _label(w["four"]) + _table(table_rows, [lw["message"], lw["label"], lw["p"], "", ""], numbers=(2,)) \
        + _note(w["note"])
    fns = run["fns"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = with_time_limit(lambda: fns.mistakes([(dict(r), label) for r, label in rows], dict(ws), b, t), 20)
        m, bl = (int(got[0]), int(got[1]))
    except Exception:   # noqa: BLE001   also Timeout; your judge.py; the sentence then says the course counted
        m, bl = _counts(rows, ws, b, t)
        fns = None
    nb = naive_bayes()
    nm = sum(1 for s, (_, label) in zip(nb, rows) if label == 1 and s == 0)
    nbl = sum(1 for s, (_, label) in zip(nb, rows) if label == 0 and s == 1)
    key = ("line" if fns is not None and "judge" in fns.yours else
           "judge_line" if fns is None or "judge" in run["fns"].broken else "course_line")
    return body, w[key].format(m=m, b=bl, nm=nm, nb=nbl)


# --- the picture in The idea -------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "One test message through the filter after training",
           "message": "You have 1 new voicemail. Please call [number]",
           "head": ["Word", "TF-IDF value", "Weight", "Value × weight"],
           "none": "no column", "bias": "bias", "z": "z, the sum", "p": "p = σ(z), the probability of spam",
           "verdict": "Below the threshold 0.5: the filter lets this spam message through.",
           "note": "Gray: the words and their TF-IDF values; voicemail is in no training message, so it gets no "
                   "column. Orange: the weights and the bias the filter learned. Blue: the probability of spam. "
                   "Purple: the mistake, spam let through.",
           "alt": "{title}: {message}. {rows}. Bias {bias}, z {z}, p {p}. {verdict} {note}"},
    "nl": {"title": "Eén testbericht door het filter na de training",
           "message": "You have 1 new voicemail. Please call [nummer]",
           "head": ["Woord", "TF-IDF-waarde", "Weight", "Waarde × weight"],
           "none": "geen kolom", "bias": "bias", "z": "z, de som", "p": "p = σ(z), de kans op spam",
           "verdict": "Onder de threshold 0,5: het filter laat dit spambericht door.",
           "note": "Grijs: de woorden en de TF-IDF-waarden; voicemail staat in geen enkel trainingsbericht, dus het "
                   "krijgt geen kolom. Oranje: de weights en de bias die het filter leerde. Blauw: de kans op spam. "
                   "Paars: de fout, doorgelaten spam.",
           "alt": "{title}: {message}. {rows}. Bias {bias}, z {z}, p {p}. {verdict} {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    rows_, alts = [], []
    for word in IDEA_WORDS:
        if word in IDEA_ROW:
            v, wt = IDEA_ROW[word], IDEA_WEIGHTS[word]
            rows_.append([_chip(word), _dec(v, lang, 3), _chip(_dec(wt, lang, 3), _CHIP_PARAM), _dec(v * wt, lang, 3)])
            alts.append(f"{word} {_dec(v, lang, 3)} × {_dec(wt, lang, 3)} = {_dec(v * wt, lang, 3)}")
        else:
            rows_.append([_chip(word, _CHIP_FADED), _chip(w["none"], _CHIP_FADED), "", ""])
            alts.append(f"{word}: {w['none']}")
    z = idea_z()
    p = _sigmoid(z)
    rows_.append([_chip(w["bias"]), "", _chip(_dec(BIAS_END, lang, 3), _CHIP_PARAM), _dec(BIAS_END, lang, 3)])
    rows_.append([f'<span style="{_TEXT};font-weight:600">{html.escape(w["z"])}</span>', "", "",
                  f'<b>{_dec(z, lang, 3)}</b>'])
    body = (f'<div style="{_TEXT};margin-bottom:0.2rem">{_chip(w["message"])}</div>'
            + _table(rows_, w["head"], numbers=(1, 2, 3))
            + f'<div style="{_LABEL}">{html.escape(w["p"])}: {_chip(_dec(p, lang, 3), _CHIP_PRED)} '
              f'{_bar(p, 1.0, "var(--ea-pred)", 8.0)}</div>'
            + f'<div style="{_LABEL}">{_chip(w["verdict"], _CHIP_PURPLE)}</div>' + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], message=w["message"], rows="; ".join(alts), bias=_dec(BIAS_END, lang, 3), z=_dec(z, lang, 3),
        p=_dec(p, lang, 3), verdict=w["verdict"], note=w["note"]))


def broken(name: str, lang: str) -> str:
    """What the demos say when they cannot run: the messages are not downloaded, it is too slow, or a module of your
    project goes wrong."""
    texts = {
        "en": {"data": "*Download the messages first, in the worked example above: the demo trains on them.*",
               "slow": "*The training takes more than three minutes. Does your `step` go over the examples only once? Or is the computer busy with something else?*",
               "features": "*Your `work/projects/text/features.py` cannot be imported, stops with an error on the real messages, or gives something other than a dictionary of numbers. Check exercise 1 again.*",
               "train": "*Your `work/projects/text/train.py` cannot be imported, stops with an error on the real messages, or `step` gives something other than a weight for every word and a bias. Check exercise 2 again.*",
               "judge": "*Your `work/projects/text/judge.py` cannot be imported, or has no `mistakes`. Check exercise 3 again.*"},
        "nl": {"data": "*Download eerst de berichten, in het uitgewerkte voorbeeld hierboven: de demo traint erop.*",
               "slow": "*De training duurt langer dan drie minuten. Gaat je `step` maar één keer over de voorbeelden? Of is de computer met iets anders bezig?*",
               "features": "*Je `work/projects/text/features.py` kan niet worden geïmporteerd, stopt met een fout op de echte berichten, of geeft iets anders dan een dictionary met getallen. Check oefening 1 opnieuw.*",
               "train": "*Je `work/projects/text/train.py` kan niet worden geïmporteerd, stopt met een fout op de echte berichten, of `step` geeft iets anders dan een weight voor elk woord en een bias. Check oefening 2 opnieuw.*",
               "judge": "*Je `work/projects/text/judge.py` kan niet worden geïmporteerd, of heeft geen `mistakes`. Check oefening 3 opnieuw.*"},
    }
    return texts[lang][name]
