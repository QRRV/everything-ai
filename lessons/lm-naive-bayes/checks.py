# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-naive-bayes: log_score, the log of P(class) times P(word | class) for every word of a
message, with Laplace smoothing. And the course's own filter: naive Bayes trained on the 40 training messages of
data/course-spam-messages.json, and the play block that walks its score through each of the 20 test messages."""
import html
import json
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, same_value

DATA = "data/course-spam-messages.json"   # 60 short messages written by the course: see data/README.md
CLASSES = ["spam", "ok"]


@cache
def messages() -> dict[str, list[tuple[str, str]]]:
    """The training and the test messages, each a pair: its label, spam or ok, and its text."""
    d = json.loads((paths.ROOT / DATA).read_text(encoding="utf-8"))
    return {part: [tuple(m) for m in d[part]] for part in ("train", "test")}


def log_score(words, prior, counts, n_words, n_vocab):
    """The course's own log score of one class: log P(class), plus log P(word | class) for every word, with
    Laplace smoothing: (count + 1) / (words of the class + size of the vocabulary)."""
    score = np.log(prior)
    for w in words:
        n = counts[w] if w in counts else 0
        score = score + np.log((n + 1) / (n_words + n_vocab))
    return score


@cache
def model() -> dict:
    """Naive Bayes counted on the training messages: per class its number of messages, how often each word comes in
    them and their number of words; and the vocabulary, every different word of all training messages."""
    train = messages()["train"]
    n_messages = {c: 0 for c in CLASSES}
    counts = {c: {} for c in CLASSES}
    n_words = {c: 0 for c in CLASSES}
    for label, text in train:
        n_messages[label] += 1
        for w in text.split():
            counts[label][w] = counts[label].get(w, 0) + 1
            n_words[label] += 1
    vocab = sorted(set(counts["spam"]) | set(counts["ok"]))
    return {"messages": n_messages, "counts": counts, "words": n_words, "vocab": vocab, "n": len(train)}


def known(text: str) -> list[str]:
    """The words of a message that are in the vocabulary; the filter leaves out the others."""
    vocab = set(model()["vocab"])
    return [w for w in text.split() if w in vocab]


def scores(text: str) -> dict[str, float]:
    """The log score of each class for a message."""
    m = model()
    return {c: float(log_score(known(text), m["messages"][c] / m["n"], m["counts"][c], m["words"][c], len(m["vocab"])))
            for c in CLASSES}


def classify(text: str) -> str:
    s = scores(text)
    return "spam" if s["spam"] > s["ok"] else "ok"


def push(word: str) -> float:
    """How far one word moves the score toward spam: log P(word | spam) − log P(word | ok)."""
    m = model()
    p = {c: (m["counts"][c].get(word, 0) + 1) / (m["words"][c] + len(m["vocab"])) for c in CLASSES}
    return math.log(p["spam"]) - math.log(p["ok"])


def prior_push() -> float:
    """Where the score starts: log P(spam) − log P(ok)."""
    m = model()["messages"]
    return math.log(m["spam"]) - math.log(m["ok"])


def right() -> int:
    """How many of the test messages the filter sorts right."""
    return sum(classify(text) == label for label, text in messages()["test"])


def number(x: float, digits: int, lang: str, sign: bool = False) -> str:
    """A number for the text and the picture: a real minus sign, a plus sign on request, a decimal comma in Dutch."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


# --- the play block: the score of one test message, word by word ---------------------------------------------------

def walk(i: int) -> list[tuple[str, float | None]]:
    """The steps of the score of test message i (from 1): the prior, then every word with its push, None for a word
    the filter leaves out."""
    _, text = messages()["test"][i - 1]
    vocab = set(model()["vocab"])
    return [("", prior_push())] + [(w, push(w) if w in vocab else None) for w in text.split()]


@cache
def _range() -> tuple[float, float]:
    """One scale for every message: from the lowest to the highest point any walk reaches, and 0."""
    lo = hi = 0.0
    for i in range(1, len(messages()["test"]) + 1):
        total = 0.0
        for _, p in walk(i):
            total += p or 0.0
            lo, hi = min(lo, total), max(hi, total)
    return math.floor(lo) - 0.5, math.ceil(hi) + 0.5


def waterfall(i: int, labels: dict[str, str], lang: str) -> str:
    """The score of test message i as a walk from the prior, one row per word: each bar starts where the one before
    ended and moves the score by that word's push, orange as something the filter learned by counting, with the push
    in a column on the right. A word it leaves out gets no bar, and "left out" in that column. The last row is the sum, blue as the filter's
    prediction: right of the dashed line at 0 is spam, left is ok. The same scale for every message, and as many rows
    as the longest message, so the block keeps its size while you slide."""
    lo, hi = _range()
    steps = walk(i)
    rows = max(len(walk(j)) for j in range(1, len(messages()["test"]) + 1))
    W, L0, R0, row_h, top = 460, 96, 410, 21, 44
    X = lambda v: L0 + (v - lo) / (hi - lo) * (R0 - L0)
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<text x="{X(0) - 6:.1f}" y="36" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["ok_side"])}</text>',
         f'<text x="{X(0) + 6:.1f}" y="36" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(labels["spam_side"])}</text>']
    total, y = 0.0, top
    for k, (word, step) in enumerate(steps):
        name = labels["prior"] if k == 0 else word
        style = "fill:var(--ea-ink)" if step is not None else "fill:var(--ea-ink-3);font-style:italic"
        p.append(f'<text x="{L0 - 8}" y="{y + 15}" style="font-size:13px;{style};text-anchor:end">{html.escape(name)}</text>')
        if step is None:
            p.append(f'<text x="{W}" y="{y + 15}" style="font-size:12px;fill:var(--ea-ink-3);font-style:italic;text-anchor:end">{html.escape(labels["left_out"])}</text>')
        else:
            a, b = sorted((X(total), X(total + step)))
            p += [f'<rect x="{a:.1f}" y="{y + 4}" width="{max(1.5, b - a):.1f}" height="{row_h - 7}" style="fill:var(--ea-param)"/>',
                  f'<text x="{W}" y="{y + 15}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{number(step, 1, lang, sign=True)}</text>']
            total += step
        y += row_h
    y = top + rows * row_h + 6
    a, b = sorted((X(0), X(total)))
    said = labels["spam"] if total > 0 else labels["ok"]
    p += [f'<line x1="0" y1="{y - 3}" x2="{W}" y2="{y - 3}" style="stroke:var(--ea-line);stroke-width:1"/>',
          f'<text x="{L0 - 8}" y="{y + 15}" style="font-size:13px;font-weight:600;fill:var(--ea-ink);text-anchor:end">'
          f'{html.escape(labels["sum"])}: {html.escape(said)}</text>',
          f'<rect x="{a:.1f}" y="{y + 3}" width="{max(1.5, b - a):.1f}" height="{row_h - 5}" style="fill:var(--ea-pred)"/>',
          f'<text x="{W}" y="{y + 15}" style="font-size:13px;font-weight:600;fill:var(--ea-pred-text);text-anchor:end">{number(total, 1, lang, sign=True)}</text>',
          f'<line x1="{X(0):.1f}" y1="40" x2="{X(0):.1f}" y2="{y + row_h + 2}" style="stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:4 3"/>']
    height = y + row_h + 6
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" style="margin:0.25rem 0 0.5rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {height}" aria-hidden="true" style="width:100%;height:auto;font-family:var(--ea-font-ui);'
            f'font-variant-numeric:tabular-nums">{"".join(p)}</svg></figure>')


def frame(i: int, lang: str) -> dict[str, str]:
    """The words of one stop of the play block, for the sentence under it and the alt text."""
    label, text = messages()["test"][i - 1]
    steps = walk(i)
    total = sum(s for _, s in steps if s is not None)
    return {"i": str(i), "text": text, "label": label, "said": classify(text),
            "prior": number(prior_push(), 1, lang, sign=True), "sum": number(total, 1, lang, sign=True),
            "steps": ", ".join(f"{w} {number(s, 1, lang, sign=True)}" for w, s in steps[1:] if s is not None)}


# The lesson is done when these are correct. "unseen-word" and "naive" are choice questions in the lesson.
REQUIRED = ["unseen-word", "log-score", "naive"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _short(v):
    """A score for the messages: four significant digits, without float noise or e-notation; minus infinity as
    -inf, the way Python writes it."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else ("-inf" if x < 0 else "inf" if x > 0 else "nan")
    text = f"{x:.{max(0, 3 - math.floor(math.log10(abs(x))))}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _close(a, b) -> bool:
    """A score within 0.01% of the right one: one typed from the four digits a message shows is not another number."""
    return same_value(a, b, rel=1e-4)


class _Repeated(list):
    """A long message for the check, one word many times, that a message writes short: ["free"] * 300. The
    learner's code gets a plain copy."""

    def __repr__(self):
        return f'["{self[0]}"] * {len(self)}'

    def __deepcopy__(self, memo):
        return list(self)


# --- exercise 2: log_score ------------------------------------------------------------------------------------------

# The worked example: "free cash now please" without "please", the counts of its class ok, 7 words, a vocabulary of 9.
_OK = {"lunch": 2, "now": 1, "call": 1, "me": 1, "free": 1, "for": 1}
_SPAM = {"win": 1, "cash": 2, "now": 2, "free": 1, "prize": 1}
_LONG = _Repeated(["free"] * 300)   # its product is about 1e-472: 0 for the computer

_HEAD = "import numpy as np\n\n\ndef log_score(words, prior, counts, n_words, n_vocab):\n"
_LOOP = "    for w in words:\n        if w in counts:\n            n = counts[w]\n        else:\n            n = 0\n"
_START = _HEAD + "    score = ...\n" + _LOOP + "        score = ...\n    return score\n"


def _l(first: str = "np.log(prior)", step: str = "score + np.log((n + 1) / (n_words + n_vocab))",
       end: str = "    return score\n") -> str:
    return _HEAD + f"    score = {first}\n" + _LOOP + f"        score = {step}\n" + end


_L = _l()

EXERCISES = {
    "log-score": FunctionBox(
        name="log-score",
        function="log_score",
        start=_START,
        solution=_L,
        visible=[(["free", "cash", "now"], 0.6, _OK, 7, 9)],
        hidden=[(["free", "cash", "now"], 0.4, _SPAM, 7, 9), ([], 0.25, {"free": 3}, 10, 20),
                (["free", "free", "lunch"], 0.5, {"free": 3}, 10, 20), (_LONG, 0.5, {"free": 3}, 50, 100)],
        show=_short,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-smoothing", wrong=_l(step="score + np.log(n / n_words)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Without the + 1, a word the class never had, such as \"cash\" for ok, gets probability 0, and the log of 0 is minus infinity: one new word decides the whole score. Add 1 to the count, and the size of the vocabulary to the number of words.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zonder de + 1 krijgt een woord dat de klasse nooit had, zoals \"cash\" bij ok, kans 0, en de log van 0 is min oneindig: één nieuw woord beslist dan de hele score. Tel 1 bij de telling op, en de grootte van de vocabulary bij het aantal woorden."),
            Mistake("no-plus-one", wrong=_l(step="score + np.log(n / (n_words + n_vocab))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. You add the size of the vocabulary to the number of words, but not 1 to the count, so a word the class never had, such as \"cash\" for ok, still gets probability 0, and the log of 0 is minus infinity. Add 1 to the count too: `n + 1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je telt de grootte van de vocabulary bij het aantal woorden op, maar niet 1 bij de telling, dus een woord dat de klasse nooit had, zoals \"cash\" bij ok, krijgt nog steeds kans 0, en de log van 0 is min oneindig. Tel ook 1 bij de telling op: `n + 1`."),
            Mistake("probs-not-log", wrong=_l(step="score + (n + 1) / (n_words + n_vocab)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the probabilities themselves to the log of the prior. Add their logs: `np.log((n + 1) / (n_words + n_vocab))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de kansen zelf bij de log van de prior op. Tel hun logs op: `np.log((n + 1) / (n_words + n_vocab))`."),
            Mistake("no-vocab", wrong=_l(step="score + np.log((n + 1) / n_words)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add 1 to every count, but not the size of the vocabulary to the number of words, so the probabilities of a class add up to more than 1. Divide by `n_words + n_vocab`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt 1 bij elke telling op, maar niet de grootte van de vocabulary bij het aantal woorden, dus de kansen van een klasse tellen op tot meer dan 1. Deel door `n_words + n_vocab`."),
            Mistake("message-words", wrong=_l(step="score + np.log((n + 1) / (len(words) + n_vocab))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the number of words of the message, `len(words)`. P(word | class) counts among all the words of the training messages of that class: `n_words`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door het aantal woorden van het bericht, `len(words)`. P(woord | klasse) telt onder alle woorden van de trainingsberichten van die klasse: `n_words`."),
            Mistake("prior-not-log", wrong=_l(first="prior"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your score starts at the prior itself, and then adds logs to it. Start at its log, `np.log(prior)`, so that everything you add up is a log.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je score begint bij de prior zelf, en telt er daarna logs bij op. Begin bij zijn log, `np.log(prior)`, zodat alles wat je optelt een log is."),
            Mistake("no-prior", wrong=_l(first="0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the prior is missing. Bayes' rule multiplies by P(class), so the log score starts at `np.log(prior)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de prior ontbreekt. De regel van Bayes vermenigvuldigt met P(klasse), dus de log score begint bij `np.log(prior)`."),
            Mistake("surprise", wrong=_l(first="-np.log(prior)", step="score - np.log((n + 1) / (n_words + n_vocab))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: with the minus signs, you add up the surprises, as in the NLL. Here the score is the log of the probability itself, so that the class with the highest score wins: add the logs.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: met de mintekens tel je de verrassingen op, zoals bij de NLL. Hier is de score de log van de kans zelf, zodat de klasse met de hoogste score wint: tel de logs op."),
            Mistake("product", wrong=_l(first="prior", step="score * (n + 1) / (n_words + n_vocab)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the probabilities, so your function gives P(class) times the product itself, not its log. Add up the logs instead: start at `np.log(prior)`, and add `np.log(...)` of each probability.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de kansen, dus je functie geeft P(klasse) keer het product zelf, niet de log ervan. Tel in plaats daarvan de logs op: begin bij `np.log(prior)`, en tel `np.log(...)` van elke kans erbij op."),
            Mistake("log-at-end", state="almost", wrong=_l(first="prior", step="score * (n + 1) / (n_words + n_vocab)", end="    return np.log(score)\n"),
                    en="Right for short messages. But `{call}`, a message of 300 words, gives `{got}`, and it should give `{want}`: the product of 300 small probabilities became 0 in the computer, and the log of 0 is minus infinity. Take the log of each probability and add them up, as step 3 of the recipe does.",
                    nl="Goed voor korte berichten. Maar `{call}`, een bericht van 300 woorden, geeft `{got}`, en dat moet `{want}` zijn: het product van 300 kleine kansen werd 0 in de computer, en de log van 0 is min oneindig. Neem de log van elke kans en tel ze op, zoals stap 3 van het recept doet."),
            Mistake("math-log-zero", error=lambda e: isinstance(e, ValueError) and "math domain error" in str(e),
                    en="With `{call}`, `math.log` stops with an error: it cannot take the log of 0. A probability became 0: a word the class never had, without the + 1, or the product of many small probabilities, before you took its log. Take the log of each smoothed probability, (n + 1) / (n_words + n_vocab), and add them up.",
                    nl="Met `{call}` stopt `math.log` met een fout: van 0 kan hij geen log nemen. Een kans werd 0: een woord dat de klasse nooit had, zonder de + 1, of het product van veel kleine kansen, voordat je de log nam. Neem de log van elke kans met smoothing, (n + 1) / (n_words + n_vocab), en tel ze op."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: where the score starts, and what every word adds to it. Step 3 of the recipe.",
                      "nl": "Vervang de twee `...`: waar de score begint, en wat elk woord erbij optelt. Stap 3 van het recept."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has another number of names between its parentheses. Keep `(words, prior, counts, n_words, n_vocab)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(words, prior, counts, n_words, n_vocab)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with step 3 of the recipe: the log of the prior, then the log of (count + 1) / (words of the class + size of the vocabulary) for every word.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 3 van het recept: de log van de prior, en dan voor elk woord de log van (telling + 1) / (woorden van de klasse + grootte van de vocabulary)."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use its five arguments, and not the numbers of the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie haar vijf argumenten, en niet de getallen van het voorbeeld?"},
            "correct": {"en": "Your `log_score` is right in all {n} cases: also for a message without words, which keeps the log of the prior, and for one of 300 words, whose product would have become 0.",
                        "nl": "Je `log_score` klopt in alle {n} gevallen: ook voor een bericht zonder woorden, dat de log van de prior houdt, en voor een van 300 woorden, waarvan het product 0 was geworden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("log-score", _START, "empty", None),
    ("log-score", _L, "correct", None),
    ("log-score", _l(step="score + np.log(n + 1) - np.log(n_words + n_vocab)"), "correct", None),
    ("log-score", _HEAD.replace("import numpy as np", "import math") + "    score = math.log(prior)\n" + _LOOP
     + "        score = score + math.log((n + 1) / (n_words + n_vocab))\n    return score\n", "correct", None),
    ("log-score", _l(step="score + np.log(n / n_words)"), "wrong", "no-smoothing"),
    ("log-score", _l(step="score + np.log(n / (n_words + n_vocab))"), "wrong", "no-plus-one"),
    ("log-score", _l(step="score + (n + 1) / (n_words + n_vocab)"), "wrong", "probs-not-log"),
    ("log-score", _l(step="score + np.log((n + 1) / n_words)"), "wrong", "no-vocab"),
    ("log-score", _l(step="score + np.log((n + 1) / (len(words) + n_vocab))"), "wrong", "message-words"),
    ("log-score", _l(first="prior"), "wrong", "prior-not-log"),
    ("log-score", _l(first="0"), "wrong", "no-prior"),
    ("log-score", _l(first="-np.log(prior)", step="score - np.log((n + 1) / (n_words + n_vocab))"), "wrong", "surprise"),
    ("log-score", _l(first="prior", step="score * (n + 1) / (n_words + n_vocab)"), "wrong", "product"),
    ("log-score", _l(first="prior", step="score * ((n + 1) / (n_words + n_vocab))", end="    return np.log(score)\n"), "almost", "log-at-end"),
    ("log-score", _HEAD.replace("import numpy as np", "import math") + "    score = prior\n" + _LOOP
     + "        score = score * (n + 1) / (n_words + n_vocab)\n    return math.log(score)\n", "error", "math-log-zero"),
    ("log-score", _l(step="score + np.log((n + 1) / (n_words + n_vocab + 1))"), "wrong", None),
    ("log-score", _HEAD + "    return -7.4422\n", "almost", None),
]
