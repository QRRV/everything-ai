# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-what-is-an-ml-system: the model is one part of a system. The course's spam service in eight
parts, whose code the learner counts line by line; a choice of model under a time budget; and the service at work on
ten messages, part by part.

The service is ordinary code in this file: each part lies between a line "# part: <name>" and the line
"# end of part", and part_code() gives the lines of a part, so the code the learner counts is the code that runs.
Its model is trained on the 40 training messages of data/course-spam-messages.json."""
import html
import json
import math
import re
from functools import cache

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-what-is-an-ml-system"
LANGS = ("en", "nl")
DATA = "data/course-spam-messages.json"   # 60 short messages written by the course: see data/README.md


# ======== the course's spam service ======================================================================================

# part: settings
SETTINGS = {
    "version": "spam-1",      # which model answers: in every answer, to trace mistakes
    "threshold": 0.5,         # from this probability on, a message counts as spam
    "max_chars": 300,         # a longer message gets no score
    "usual_spam": 0.35,       # the share of spam in normal times
    "alert_gap": 0.35,        # an alert when the share of spam is this far from usual
    "window": 6,              # the share is taken over this many of the latest answers
    "retrain_after": 4,       # train again once users marked this many messages
}
# end of part

# part: collection
def collect(inbox, seen):
    # the messages of the inbox that did not come in before, each once
    new = []
    for message in inbox:
        if message["id"] not in seen:
            seen.add(message["id"])
            new.append(message)
    return new
# end of part

# part: verification
def problem(message):
    # what keeps a message from getting a score, or None when nothing does
    if "text" not in message or type(message["text"]) != str:
        return "no text"
    text = message["text"]
    if text.strip() == "":
        return "empty"
    if len(text) > SETTINGS["max_chars"]:
        return "too long"
    letters = 0
    for ch in text:
        if ch.isalpha():   # isalpha() is True for a letter
            letters = letters + 1
    if letters < len(text) / 2:
        return "hardly any letters"
    return None
# end of part

# part: features
def words(text):
    # the words of a text in lower case: every character that is not a letter becomes a space
    clean = ""
    for ch in text.lower():
        if ch.isalpha():
            clean = clean + ch
        else:
            clean = clean + " "
    return clean.split()


def vocabulary(examples):
    # every word of the messages to train on, once, in a fixed order
    found = set()
    for label, text in examples:
        for w in words(text):
            found.add(w)
    return sorted(found)


def count_row(text, vocab):
    # how often each word of the vocabulary comes in the text; other words are left out
    place = {}
    for i in range(len(vocab)):
        place[vocab[i]] = i
    row = [0] * len(vocab)
    for w in words(text):
        if w in place:
            row[place[w]] = row[place[w]] + 1
    return row
# end of part

# part: model
from sklearn.linear_model import LogisticRegression


def train(rows, labels):
    # learn from count rows and the label of each row: 1 for spam, 0 for ok
    return LogisticRegression().fit(rows, labels)


def spam_probability(model, row):
    # predict_proba gives the probability of each label, 0 and 1: take the probability of 1, spam
    return model.predict_proba([row])[0][1]
# end of part

# part: serving
def answer(message, system):
    # the answer to one message: always an answer, also without a score
    reply = {"id": message["id"], "version": SETTINGS["version"]}
    reason = problem(message)
    if reason is None:
        row = count_row(message["text"], system["vocab"])
        if sum(row) == 0:
            reason = "no known words"
    if reason is not None:
        reply["label"] = "no score"
        reply["reason"] = reason
    else:
        p = spam_probability(system["model"], row)
        reply["probability"] = round(float(p), 2)
        if p >= SETTINGS["threshold"]:
            reply["label"] = "spam"
        else:
            reply["label"] = "ok"
    system["log"].append(reply)
    return reply
# end of part

# part: monitoring
def watch(log):
    # numbers over the latest answers, with an alert when something is off
    latest = log[-SETTINGS["window"]:]
    scored = [r for r in latest if r["label"] != "no score"]
    report = {"answers": len(latest), "no score": len(latest) - len(scored)}
    if len(scored) >= SETTINGS["window"] / 2:   # a share of only a few answers says little
        share = len([r for r in scored if r["label"] == "spam"]) / len(scored)
        report["spam share"] = round(share, 2)
        if abs(share - SETTINGS["usual_spam"]) > SETTINGS["alert_gap"]:
            report["alert"] = "share of spam"
    if report["no score"] > len(latest) / 2:
        report["alert"] = "no score"
    return report
# end of part

# part: feedback
def mark(marked, message, label):
    # a user says what a message really is: kept to train on later
    marked.append((label, message["text"]))


def accuracy(model, vocab, examples):
    # the share of examples the model labels right
    right = 0
    for label, text in examples:
        p = spam_probability(model, count_row(text, vocab))
        if (p >= SETTINGS["threshold"]) == (label == "spam"):
            right = right + 1
    return right / len(examples)


def build(examples):
    # a vocabulary and a model, trained on (label, text) pairs
    vocab = vocabulary(examples)
    rows = []
    labels = []
    for label, text in examples:
        rows.append(count_row(text, vocab))
        labels.append(int(label == "spam"))   # 1 for spam, 0 for ok
    return vocab, train(rows, labels)


def retrain(system, marked, test):
    # train on the old examples and the marks; keep the new model only if it does as well
    if len(marked) < SETTINGS["retrain_after"]:
        return False
    vocab, model = build(system["examples"] + marked)
    if accuracy(model, vocab, test) < accuracy(system["model"], system["vocab"], test):
        return False
    system["vocab"] = vocab
    system["model"] = model
    system["examples"] = system["examples"] + marked
    marked.clear()   # the marks are used: start counting again
    return True
# end of part

# ======== end of the service =============================================================================================

PARTS = ("settings", "collection", "verification", "features", "model", "serving", "monitoring", "feedback")
PART_NAMES = {
    "en": {"settings": "settings", "collection": "collection", "verification": "verification", "features": "features",
           "model": "model", "serving": "serving", "monitoring": "monitoring", "feedback": "feedback"},
    "nl": {"settings": "instellingen", "collection": "verzamelen", "verification": "controle", "features": "features",
           "model": "model", "serving": "serving", "monitoring": "monitoring", "feedback": "feedback"},
}
# The comments of the service in Dutch: part_code(name, "nl") shows them instead of the English ones.
COMMENTS_NL = {
    "which model answers: in every answer, to trace mistakes": "welk model antwoordt: in elk antwoord, om fouten terug te vinden",
    "from this probability on, a message counts as spam": "vanaf deze kans telt een bericht als spam",
    "a longer message gets no score": "een langer bericht krijgt geen score",
    "the share of spam in normal times": "het aandeel spam in gewone tijden",
    "an alert when the share of spam is this far from usual": "een alarm als het aandeel spam zo ver van het gewone aandeel ligt",
    "the share is taken over this many of the latest answers": "het aandeel gaat over zoveel van de laatste antwoorden",
    "train again once users marked this many messages": "opnieuw trainen zodra gebruikers zoveel berichten markeerden",
    "the messages of the inbox that did not come in before, each once": "de nieuwe berichten uit de inbox, elk één keer",
    "what keeps a message from getting a score, or None when nothing does": "wat een score voor een bericht in de weg staat, of None als niets dat doet",
    "isalpha() is True for a letter": "isalpha() is True voor een letter",
    "the words of a text in lower case: every character that is not a letter becomes a space": "de woorden van een tekst in kleine letters: elk teken dat geen letter is, wordt een spatie",
    "every word of the messages to train on, once, in a fixed order": "elk woord van de berichten om op te trainen, één keer, in een vaste volgorde",
    "how often each word of the vocabulary comes in the text; other words are left out": "hoe vaak elk woord van de vocabulary in de tekst voorkomt; andere woorden vallen weg",
    "learn from count rows and the label of each row: 1 for spam, 0 for ok": "leren van rijen met tellingen en het label van elke rij: 1 voor spam, 0 voor ok",
    "predict_proba gives the probability of each label, 0 and 1: take the probability of 1, spam": "predict_proba geeft de kans op elk label, 0 en 1: neem de kans op 1, spam",
    "the answer to one message: always an answer, also without a score": "het antwoord op één bericht: altijd een antwoord, ook zonder score",
    "numbers over the latest answers, with an alert when something is off": "getallen over de laatste antwoorden, met een alarm als er iets niet klopt",
    "a user says what a message really is: kept to train on later": "een gebruiker zegt wat een bericht echt is: bewaard om later op te trainen",
    "a share of only a few answers says little": "het aandeel van een paar antwoorden zegt weinig",
    "the share of examples the model labels right": "het aandeel voorbeelden dat het model goed labelt",
    "a vocabulary and a model, trained on (label, text) pairs": "een vocabulary en een model, getraind op paren (label, tekst)",
    "1 for spam, 0 for ok": "1 voor spam, 0 voor ok",
    "train on the old examples and the marks; keep the new model only if it does as well": "train op oude voorbeelden en markeringen; houd het nieuwe model alleen als het even goed is",
    "the marks are used: start counting again": "de markeringen zijn gebruikt: begin opnieuw te tellen",
}


@cache
def _parts_en() -> dict:
    source = (paths.LESSONS / LESSON / "checks.py").read_text(encoding="utf-8")
    found = re.findall(r"^# part: (\w+)\n(.*?)^# end of part$", source, flags=re.MULTILINE | re.DOTALL)
    return {name: code.strip("\n") for name, code in found}


def part_code(name: str, lang: str = "en") -> str:
    """The code of one part of the service, as it stands in this file; in Dutch with its comments in Dutch."""
    code = _parts_en()[name]
    if lang == "nl":
        code = re.sub(r"# (.+)$", lambda m: "# " + COMMENTS_NL.get(m.group(1), m.group(1)), code, flags=re.MULTILINE)
    return code


def comments(code: str) -> list:
    """The text of every comment in a piece of code."""
    return re.findall(r"# (.+)$", code, flags=re.MULTILINE)


class _Part(check.Code):
    """The code of a part for the check: its call shows as `C.part_code("model")`, and the learner's function gets a
    plain copy of the text."""

    def __new__(cls, name: str):
        made = super().__new__(cls, part_code(name))
        made.part = name
        return made

    def __str__(self):
        return f'C.part_code("{self.part}")'

    def __deepcopy__(self, memo):
        return "".join(self)


# ======== the functions of the exercises, as the course writes them ===================================================

def code_lines(code):
    count = 0
    for line in code.split("\n"):
        text = line.strip()
        if text != "" and text[0] != "#":
            count = count + 1
    return count


def best_in_budget(models, budget):
    best_name = None
    best_accuracy = 0
    for name, accuracy, ms in models:
        if ms <= budget and accuracy > best_accuracy:
            best_name = name
            best_accuracy = accuracy
    return best_name


def course_lines() -> dict:
    """The lines of code of each part of the service, counted by the course's code_lines."""
    return {name: code_lines(part_code(name)) for name in PARTS}


# Six spam models as a team might compare them: the accuracy on a validation set of many messages, and the milliseconds
# one message takes. Example numbers made up by the course, in the order a team might try them.
class _Models(list):
    """The six models for the check: its call shows them as `C.MODELS`, and the learner's function gets a plain copy."""

    def __repr__(self):
        return "C.MODELS"

    def __deepcopy__(self, memo):
        return [tuple(m) for m in self]


MODELS = _Models([
    ("word list", 0.80, 0.002),
    ("naive Bayes", 0.93, 0.02),
    ("logistic regression", 0.95, 0.05),
    ("small network", 0.96, 1.5),
    ("30 models together", 0.97, 40.0),
    ("large language model", 0.98, 800.0),
])
BUDGET = 10.0   # milliseconds per message: the mail server takes in 100 messages per second on one core
BUDGETS = (1.0, 10.0, 1000.0)   # the budgets of the picture


# ======== the service at work ===========================================================================================

def messages() -> dict:
    d = json.loads((paths.ROOT / DATA).read_text(encoding="utf-8"))
    return {part: [tuple(m) for m in d[part]] for part in ("train", "test")}


def new_system() -> dict:
    """The service, ready to answer: its vocabulary and model trained on the 40 training messages, and an empty log."""
    examples = messages()["train"]
    vocab, model = build(examples)
    return {"vocab": vocab, "model": model, "log": [], "examples": examples}


# Ten messages as they come in: capitals and punctuation, one that comes twice, and four the model cannot score.
INBOX = [
    {"id": 1, "text": "Are you home for dinner tonight?"},
    {"id": 2, "text": "WIN a free trip NOW!!! Text WIN to claim"},
    {"id": 3, "text": "   "},
    {"id": 2, "text": "WIN a free trip NOW!!! Text WIN to claim"},
    {"id": 4, "text": "We won free concert tickets, text me now"},
    {"id": 5, "text": "1234 5678 !!! 999"},
    {"id": 6, "text": "please call back " * 20},
    {"id": 7, "text": "Zzz qqq?"},
    {"id": 8, "text": "Urgent: pay the fee to get your parcel"},
    {"id": 9, "text": "Cash prize waiting for you, call today!"},
]
USER_MARKS = {4: "ok"}   # a user moves message 4 out of the spam folder: it is ok


@cache
def run_inbox() -> tuple:
    """The service on the inbox, message by message: for each message what each part did."""
    system = new_system()
    seen = set()
    marked = []
    steps = []
    for message in INBOX:
        step = {"message": message, "new": bool(collect([message], seen))}
        if step["new"]:
            step["problem"] = problem(message)
            step["known"] = sum(1 for w in words(message["text"]) if w in set(system["vocab"])) if step["problem"] is None else None
            step["words"] = len(words(message["text"])) if step["problem"] is None else None
            step["reply"] = answer(message, system)
            step["report"] = watch(system["log"])
            if message["id"] in USER_MARKS:
                mark(marked, message, USER_MARKS[message["id"]])
                step["marked"] = USER_MARKS[message["id"]]
            step["marks"] = len(marked)
        steps.append(step)
    return tuple(steps)


def scored_count() -> int:
    return sum(1 for s in run_inbox() if s["new"] and s["reply"]["label"] != "no score")


# ======== words and drawing =============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def ms_text(ms: float, lang: str) -> str:
    """Milliseconds with as few digits as they need: 0.002 ms, 1.5 ms, 800 ms."""
    digits = 0 if ms >= 10 or ms == int(ms) else (1 if ms >= 1 else max(1, -math.floor(math.log10(ms))))
    return f"{number(ms, digits, lang)} ms"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", halo: bool = False) -> str:
    rim = ";paint-order:stroke;stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round" if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}{rim}">{html.escape(text_)}</text>')


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:var(--ea-ink-2);stroke-width:1.3"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.3"/>')


def _box(x: float, y: float, w: float, h: float, name: str, caption: str, model: bool = False,
         faded: bool = False) -> str:
    """A part of the service: the model filled with its own color, every other part with an outline."""
    fill = "var(--ea-param-wash)" if model else "var(--ea-page)"
    stroke = "var(--ea-param)" if model else ("var(--ea-line)" if faded else "var(--ea-ink-3)")
    ink = "var(--ea-ink-3)" if faded else "var(--ea-ink)"
    dash = ";stroke-dasharray:4 3" if faded else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:{fill};stroke:{stroke};stroke-width:{2 if model else 1.2}{dash}"/>'
            + _t(x + 8, y + 15, name, 13, 600, ink)
            + (_t(x + 8, y + 30, caption, 11.5, 400, "var(--ea-ink-3)" if faded else "var(--ea-ink-2)") if caption else ""))


IDEA_TEXT = {
    "en": {"in": "messages come in", "out": "answer: spam, ok or no score",
           "settings": "read by other parts", "collection": "takes each message once", "verification": "is the message usable?",
           "features": "words into counts", "model": "probability of spam", "serving": "always gives an answer",
           "monitoring": "watches the answers", "feedback": "marks, train again",
           "alt": "The course's spam service in eight parts. Messages come in at the top and go down through collection, verification, features, the model and serving, which gives the answer. Beside them: settings, read by the other parts; monitoring, which watches the answers of serving; and feedback, which keeps the marks of users and trains the model again."},
    "nl": {"in": "berichten komen binnen", "out": "antwoord: spam, ok of geen score",
           "settings": "voor de andere delen", "collection": "neemt elk bericht één keer", "verification": "is het bericht bruikbaar?",
           "features": "woorden naar tellingen", "model": "kans op spam", "serving": "geeft altijd een antwoord",
           "monitoring": "bekijkt de antwoorden", "feedback": "traint met markeringen",
           "alt": "De spamdienst van de course in acht delen. Berichten komen bovenaan binnen en gaan omlaag door verzamelen, controle, features, het model en serving, dat het antwoord geeft. Ernaast: instellingen, voor de andere delen; monitoring, dat de antwoorden van serving bekijkt; en feedback, dat de markeringen van gebruikers bewaart en het model opnieuw traint."},
}


def idea_picture(lang: str) -> str:
    """The parts of the service and how a message goes through them: the model is one of eight."""
    w, names = IDEA_TEXT[lang], PART_NAMES[lang]
    X, W, H, GAP, TOP = 0.0, 172.0, 38.0, 14.0, 22.0
    RX, RW = 188.0, 140.0
    parts = [_t(X + 6, 14, w["in"], 12, 400, "var(--ea-ink-2)")]
    chain = ("collection", "verification", "features", "model", "serving")
    y = {}
    for i, name in enumerate(chain):
        y[name] = TOP + i * (H + GAP)
        parts.append(_box(X, y[name], W, H, names[name], w[name], model=name == "model"))
        if i:
            parts.append(_arrow(X + W / 2, y[chain[i - 1]] + H, X + W / 2, y[name] - 1))
    bottom = y["serving"] + H
    parts.append(_arrow(X + W / 2, bottom, X + W / 2, bottom + 14))
    parts.append(_t(X + 6, bottom + 28, w["out"], 12, 400, "var(--ea-ink-2)"))
    parts.append(_box(RX, y["collection"], RW, H, names["settings"], w["settings"]))
    parts.append(_box(RX, y["model"], RW, H, names["feedback"], w["feedback"]))
    parts.append(_box(RX, y["serving"], RW, H, names["monitoring"], w["monitoring"]))
    parts.append(_arrow(RX, y["model"] + H / 2, X + W + 1, y["model"] + H / 2))            # feedback trains the model
    parts.append(_arrow(X + W, y["serving"] + H / 2, RX - 1, y["serving"] + H / 2))        # monitoring reads the answers
    height = bottom + 36
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


def bars_svg(rows: list, title: str, unit: str, lang: str, alt: str) -> str:
    """One bar per row, (label, value, model), from 0 to its value, with its label above it and its value at its end:
    the model in its own color."""
    BAR, ROW = 13, 36
    W = 322 - 7.4 * max(len(f"{number(v, 0, lang)} {unit}") for _, v, _ in rows)
    top = max(v for _, v, _ in rows) or 1
    parts, y = [], 0
    if title:
        parts.append(_t(0, 15, title, 14, 600))
        y = 28
    for label, value, model in rows:
        width = max(2.0, W * value / top)
        fill = "var(--ea-param)" if model else "var(--ea-ink-3)"
        parts.append(_t(0, y + 12, label, 12.5, 600 if model else 400, "var(--ea-ink)" if model else "var(--ea-ink-2)")
                     + f'<rect x="0" y="{y + 16}" width="{width:.1f}" height="{BAR}" rx="2" style="fill:{fill}"/>'
                     + _t(width + 6, y + 27, f"{number(value, 0, lang)} {unit}", 12.5, 600))
        y += ROW
    return f'<svg viewBox="0 0 330 {y + 4}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">{"".join(parts)}</svg>'


def share_svg(model: float, total: float, lang: str, alt: str) -> str:
    """One bar for all the code of the service, with the model's share in its own color."""
    W = 322.0
    part = max(2.0, W * model / total) if total else 0
    pct = 100 * model / total if total else 0
    words_ = {"en": "the model: {m} of {t} lines, {p}%", "nl": "het model: {m} van de {t} regels, {p}%"}[lang]
    label = words_.format(m=number(model, 0, lang), t=number(total, 0, lang), p=number(pct, 1, lang))
    return (f'<svg viewBox="0 0 330 46" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'<rect x="0" y="4" width="{W:.1f}" height="16" rx="2" style="fill:var(--ea-line)"/>'
            f'<rect x="0" y="4" width="{part:.1f}" height="16" rx="2" style="fill:var(--ea-param)"/>'
            + _t(0, 38, label, 12.5, 600) + "</svg>")


# ======== the pictures of the learner's work ==========================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _count(v) -> int | None:
    """v as a whole number of lines, 0 or more, or None: never from text or True and False."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v != int(v):
        return None
    return int(v)


_YOURS_TEXT = {
    "en": {"lines": {"title": "Your code_lines on the service",
                     "intro": "Your `code_lines` on the code of each part of the course's spam service.",
                     "chart": "Lines of code per part",
                     "alt": "One bar per part of the spam service, with the lines of code your function counts; the model's bar in its own color.",
                     "share_alt": "One bar for all the lines of code of the service, with the share of the model in its own color.",
                     "unit": "lines",
                     "none": "Your `code_lines` stops with an error, takes too long, or does not give a whole number of lines for every part, so nothing can be drawn yet."},
           "budget": {"title": "Your best_in_budget on the six models",
                      "intro": "The six models of the exercise, with the milliseconds one message takes on a log scale. Your `best_in_budget` chooses with a budget of {b} per message.",
                      "alt": "One row per model, with its accuracy and a bar for the milliseconds one message takes, and the budget as a line across the rows; the model your function chooses stands out.",
                      "budget": "budget {b}", "mine": "your choice", "none_chosen": "no model",
                      "with": "With a budget of {b}: {name}.",
                      "none": "Your `best_in_budget` stops with an error, takes too long, or gives something that is not the name of a model or None, so nothing can be drawn yet."}},
    "nl": {"lines": {"title": "Je code_lines op de dienst",
                     "intro": "Je `code_lines` op de code van elk deel van de spamdienst van de course.",
                     "chart": "Regels code per deel",
                     "alt": "Eén balk per deel van de spamdienst, met de regels code die je functie telt; de balk van het model in een eigen kleur.",
                     "share_alt": "Eén balk voor alle regels code van de dienst, met het aandeel van het model in een eigen kleur.",
                     "unit": "regels",
                     "none": "Je `code_lines` stopt met een fout, duurt te lang, of geeft niet voor elk deel een heel aantal regels, dus is er nog geen afbeelding."},
           "budget": {"title": "Je best_in_budget op de zes modellen",
                      "intro": "De zes modellen van de oefening, met de milliseconden die één bericht kost op een logaritmische schaal. Je `best_in_budget` kiest met een budget van {b} per bericht.",
                      "alt": "Eén rij per model, met de accuracy en een balk voor de milliseconden die één bericht kost, en het budget als een lijn door de rijen; het model dat je functie kiest, valt op.",
                      "budget": "budget {b}", "mine": "jouw keuze", "none_chosen": "geen model",
                      "with": "Met een budget van {b}: {name}.",
                      "none": "Je `best_in_budget` stopt met een fout, duurt te lang, of geeft iets dat geen naam van een model is en geen None, dus is er nog geen afbeelding."}},
}
MODEL_NAMES_NL = {"word list": "woordenlijst", "naive Bayes": "naive Bayes", "logistic regression": "logistic regression",
                  "small network": "klein netwerk", "30 models together": "30 modellen samen",
                  "large language model": "large language model"}


def model_name(name: str, lang: str) -> str:
    return MODEL_NAMES_NL.get(name, name) if lang == "nl" else name


def lines_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w, names = _YOURS_TEXT[lang]["lines"], PART_NAMES[lang]
    fn = _fn(seen)
    got = _call(lambda: [fn(part_code(p)) for p in PARTS]) if fn is not None else None
    counts = [_count(v) for v in got] if isinstance(got, list) else [None]
    if any(c is None for c in counts):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = [(names[p], c, p == "model") for p, c in zip(PARTS, counts)]
    body = _note(w["intro"]) + bars_svg(rows, w["chart"], w["unit"], lang, w["alt"])
    total = sum(counts)
    if total > 0:
        body += share_svg(counts[PARTS.index("model")], total, lang, w["share_alt"])
    return yours.figure(w["title"], body, alt=w["title"])


def _x_ms(ms: float, W: float) -> float:
    """Where a number of milliseconds lies on the log scale from 0.001 to 1,000 ms."""
    return W * (math.log10(max(ms, 0.001)) + 3) / 6


def budget_svg(chosen: str | None, budget: float, lang: str, alt: str, title: str) -> str:
    """One row per model: its name and accuracy, a bar for its milliseconds on a log scale, the budget as a line."""
    w = _YOURS_TEXT[lang]["budget"]
    W, ROW, TOP = 250.0, 36, 46
    x = _x_ms(budget, W)
    bottom = TOP + len(MODELS) * ROW
    parts = [_t(0, 15, title, 14, 600),
             f'<line x1="{x:.1f}" y1="{TOP - 4}" x2="{x:.1f}" y2="{bottom - 4}" '
             f'style="stroke:var(--ea-ink);stroke-width:1.5;stroke-dasharray:5 3"/>']
    for i, (name, acc, ms) in enumerate(MODELS):
        y = TOP + i * ROW
        mine = name == chosen
        over = ms > budget
        ink = "var(--ea-ink)" if mine else ("var(--ea-ink-3)" if over else "var(--ea-ink-2)")
        label = f"{model_name(name, lang)} · {number(acc, 2, lang)}" + (f" · {w['mine']}" if mine else "")
        parts.append(_t(0, y + 12, label, 12.5, 600 if mine else 400, ink, halo=True))
        width = max(2.0, _x_ms(ms, W))
        fill = "var(--ea-ink)" if mine else ("var(--ea-line)" if over else "var(--ea-ink-3)")
        parts.append(f'<rect x="0" y="{y + 16}" width="{width:.1f}" height="12" rx="2" style="fill:{fill}"/>')
        parts.append(_t(width + 5, y + 26.5, ms_text(ms, lang), 11.5, 400, ink, halo=True))
    parts.append(_t(min(x, 330 - 4), TOP - 8, w["budget"].format(b=ms_text(budget, lang)), 11.5, 600, "var(--ea-ink)",
                    "end" if x > 165 else "start"))
    return f'<svg viewBox="0 0 330 {bottom + 2}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">{"".join(parts)}</svg>'


def budget_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["budget"]
    fn = _fn(seen)
    names = {m[0] for m in MODELS}
    got = _call(lambda: [fn([tuple(m) for m in MODELS], b) for b in BUDGETS]) if fn is not None else None
    ok = isinstance(got, list) and all(g is None or (isinstance(g, str) and g in names) for g in got)
    intro = w["intro"].format(b=ms_text(BUDGET, lang))
    if not ok:
        return yours.figure(w["title"], _note(intro) + _note(w["none"], _LOSS_UI), alt=w["title"])
    chosen = dict(zip(BUDGETS, got))
    svg = budget_svg(chosen[BUDGET], BUDGET, lang, w["alt"], model_name(chosen[BUDGET], lang) if chosen[BUDGET] else w["none_chosen"])
    says = " ".join(w["with"].format(b=ms_text(b, lang), name=model_name(chosen[b], lang) if chosen[b] else w["none_chosen"])
                    for b in BUDGETS)
    return yours.figure(w["title"], _note(intro) + svg + _note(says), alt=w["title"])


# ======== the demo: the service at work, message by message ===========================================================

DEMO_TEXT = {
    "en": {"title": "Message {i} of {n}", "settings": "settings: version {v}, threshold {t}",
           "chars": "({n} characters)",
           "new": "a new message", "again": "came in before: skipped",
           "fine": "usable: {n} characters", "stops": "stops it: {reason}",
           "known": "{w} words, {k} known", "no_known": "{w} words, none known",
           "probability": "probability of spam {p}", "label": "answers: {label}", "no_score": "answers: no score ({reason})",
           "watch": "answers looked at: {a}; spam {s}, no score {u}",
           "alert_share": "alert: share of spam {p}", "alert_score": "alert: {u} of {a} answers no score",
           "no_mark": "no mark", "marked": "a user marks it: {label} ({m} of {r} marks)",
           "skipped": "does not run",
           "reasons": {"empty": "empty", "too long": "too long", "hardly any letters": "hardly any letters",
                       "no text": "no text", "no known words": "no known words"},
           "labels": {"spam": "spam", "ok": "ok"},
           "say_again": "The collection has seen message {id} before, so nothing else runs: without this part, the sender would get two answers and the monitoring would count the message twice.",
           "say_stops": "The verification stops this message ({reason}): the model never sees it, and the serving still answers.",
           "say_no_known": "None of its words are in the messages the model learned from, so its count row holds only zeros: the serving answers without asking the model.",
           "say_scored": "The model gives a probability of spam of {p}, and the serving turns it into the answer {label} with the threshold {t}.",
           "say_marked": " A user moves it out of the spam folder: it is ok, so the model was wrong, and the feedback keeps the message to train on later.",
           "say_alert_score": " The monitoring raises an alert: {u} of the last {a} answers have no score.",
           "say_alert_share": " The monitoring raises an alert: the share of spam among the last answers with a score is {p}, far from the usual {q}.",
           "alt": "The parts of the spam service for message {i}: what each part did with it, and which parts did not run."},
    "nl": {"title": "Bericht {i} van {n}", "settings": "instellingen: versie {v}, threshold {t}",
           "chars": "({n} tekens)",
           "new": "een nieuw bericht", "again": "kwam al eerder binnen: overgeslagen",
           "fine": "bruikbaar: {n} tekens", "stops": "houdt het tegen: {reason}",
           "known": "{w} woorden, {k} bekend", "no_known": "{w} woorden, geen bekend",
           "probability": "kans op spam {p}", "label": "antwoordt: {label}", "no_score": "antwoordt: geen score ({reason})",
           "watch": "bekeken antwoorden: {a}; spam {s}, geen score {u}",
           "alert_share": "alarm: aandeel spam {p}", "alert_score": "alarm: {u} van {a} antwoorden zonder score",
           "no_mark": "geen markering", "marked": "een gebruiker markeert het: {label} ({m} van {r} markeringen)",
           "skipped": "draait niet",
           "reasons": {"empty": "leeg", "too long": "te lang", "hardly any letters": "bijna geen letters",
                       "no text": "geen tekst", "no known words": "geen bekende woorden"},
           "labels": {"spam": "spam", "ok": "ok"},
           "say_again": "Verzamelen heeft bericht {id} al eerder gezien, dus verder draait er niets: zonder dit deel kreeg de afzender twee antwoorden en telde de monitoring het bericht twee keer.",
           "say_stops": "De controle houdt dit bericht tegen ({reason}): het model ziet het bericht nooit, en serving antwoordt toch.",
           "say_no_known": "Geen van de woorden staat in de berichten waarvan het model leerde, dus bevat de rij met tellingen alleen nullen: serving antwoordt zonder het model te vragen.",
           "say_scored": "Het model geeft een kans op spam van {p}, en serving maakt daar met de threshold {t} het antwoord {label} van.",
           "say_marked": " Een gebruiker haalt het bericht uit de spammap: het is ok, dus zat het model fout, en feedback bewaart het bericht om later op te trainen.",
           "say_alert_score": " De monitoring geeft een alarm: {u} van de laatste {a} antwoorden hebben geen score.",
           "say_alert_share": " De monitoring geeft een alarm: het aandeel spam onder de laatste antwoorden met een score is {p}, ver van het gewone aandeel van {q}.",
           "alt": "De delen van de spamdienst bij bericht {i}: wat elk deel met het bericht deed, en welke delen niet draaiden."},
}


def _rows(step: dict, lang: str) -> list:
    """What each part did with one message: (part, note, how), how being "ran", "stopped", "model" or "skipped"."""
    w = DEMO_TEXT[lang]
    rows = []
    if not step["new"]:
        rows.append(("collection", w["again"], "stopped"))
        return rows + [(p, w["skipped"], "skipped") for p in ("verification", "features", "model", "serving", "monitoring", "feedback")]
    rows.append(("collection", w["new"], "ran"))
    reply, report = step["reply"], step["report"]
    reason = reply.get("reason")
    if step["problem"] is not None:
        rows.append(("verification", w["stops"].format(reason=w["reasons"][reason]), "stopped"))
        rows += [("features", w["skipped"], "skipped"), ("model", w["skipped"], "skipped")]
    else:
        rows.append(("verification", w["fine"].format(n=len(step["message"]["text"])), "ran"))
        if reason == "no known words":
            rows.append(("features", w["no_known"].format(w=step["words"]), "stopped"))
            rows.append(("model", w["skipped"], "skipped"))
        else:
            rows.append(("features", w["known"].format(w=step["words"], k=step["known"]), "ran"))
            rows.append(("model", w["probability"].format(p=number(reply["probability"], 2, lang)), "model"))
    if reason is None:
        rows.append(("serving", w["label"].format(label=w["labels"][reply["label"]]), "ran"))
    else:
        rows.append(("serving", w["no_score"].format(reason=w["reasons"][reason]), "ran"))
    a, u = report["answers"], report["no score"]
    if report.get("alert") == "no score":
        rows.append(("monitoring", w["alert_score"].format(u=u, a=a), "stopped"))
    elif report.get("alert") == "share of spam":
        rows.append(("monitoring", w["alert_share"].format(p=number(report["spam share"], 2, lang)), "stopped"))
    else:
        rows.append(("monitoring", w["watch"].format(a=a, s=_spam_in_window(step), u=u), "ran"))
    if "marked" in step:
        rows.append(("feedback", w["marked"].format(label=w["labels"][step["marked"]], m=step["marks"],
                                                     r=SETTINGS["retrain_after"]), "stopped"))
    else:
        rows.append(("feedback", w["no_mark"], "ran"))
    return rows


def _spam_in_window(step: dict) -> int:
    """The spam answers among the latest answers the monitoring looked at, after this message."""
    steps = run_inbox()
    k = next(i for i, s in enumerate(steps) if s is step)
    log = [s["reply"] for s in steps[:k + 1] if s["new"]]
    return sum(1 for r in log[-SETTINGS["window"]:] if r["label"] == "spam")


def _shown_text(text_: str, lang: str) -> list:
    """A message in at most two lines of 42 characters, cut short with its length when it is longer."""
    w = DEMO_TEXT[lang]
    text_ = text_ if text_.strip() else "␣" * len(text_)
    if len(text_) <= 42:
        return [text_]
    cut = text_.rfind(" ", 0, 43)
    cut = cut if cut > 0 else 42
    if len(text_) - cut - 1 <= 42:
        return [text_[:cut], text_[cut + 1:]]
    return [text_[:42].rstrip(), text_[42:62].rstrip() + "… " + w["chars"].format(n=number(len(text_), 0, lang))]


def frame(i: int, lang: str) -> tuple[str, str]:
    """The service at message i (1 to 10): the message, the settings, and a box per part with what it did."""
    w, names = DEMO_TEXT[lang], PART_NAMES[lang]
    steps = run_inbox()
    i = max(1, min(len(steps), int(i)))
    step = steps[i - 1]
    text_lines = _shown_text(step["message"]["text"], lang)
    parts = [_t(0, 14, w["title"].format(i=i, n=len(steps)), 13, 600)]
    y = 22
    box_h = 10 + 17 * len(text_lines)
    parts.append(f'<rect x="0" y="{y}" width="330" height="{box_h}" rx="5" style="fill:var(--ea-inset)"/>')
    for k, line in enumerate(text_lines):
        parts.append(f'<text x="8" y="{y + 18 + 17 * k}" style="font-size:11.5px;font-family:var(--ea-font-code);'
                     f'fill:var(--ea-ink)">{html.escape(line)}</text>')
    y += box_h + 18
    parts.append(_t(0, y, w["settings"].format(v=SETTINGS["version"], t=number(SETTINGS["threshold"], 1, lang)), 11.5, 400,
                    "var(--ea-ink-2)"))
    y += 10
    for part, note, how in _rows(step, lang):
        parts.append(_box(0, y, 330, 36, names[part], note, model=how == "model", faded=how == "skipped"))
        if how == "stopped":
            parts.append(f'<rect x="0" y="{y}" width="4" height="36" rx="2" style="fill:var(--ea-ink)"/>')
        y += 42
    svg = (f'<svg viewBox="0 0 330 {y:.0f}" role="img" aria-label="{html.escape(w["alt"].format(i=i))}" style="{_SVG}">'
           f'{"".join(parts)}</svg>')
    return svg, say(step, lang)


def say(step: dict, lang: str) -> str:
    w = DEMO_TEXT[lang]
    if not step["new"]:
        return w["say_again"].format(id=step["message"]["id"])
    reply, report = step["reply"], step["report"]
    reason = reply.get("reason")
    if reason == "no known words":
        out = w["say_no_known"]
    elif reason is not None:
        out = w["say_stops"].format(reason=w["reasons"][reason])
    else:
        out = w["say_scored"].format(p=number(reply["probability"], 2, lang), label=w["labels"][reply["label"]],
                                     t=number(SETTINGS["threshold"], 1, lang))
    if "marked" in step:
        out += w["say_marked"]
    if report.get("alert") == "no score":
        out += w["say_alert_score"].format(u=report["no score"], a=report["answers"])
    elif report.get("alert") == "share of spam":
        out += w["say_alert_share"].format(p=number(report["spam share"], 2, lang), q=number(SETTINGS["usual_spam"], 2, lang))
    return out


def parts_accordion(lang: str) -> dict:
    """The code of every part, for the accordion under the picture: its name and its code."""
    return {PART_NAMES[lang][p].capitalize(): part_code(p, lang) for p in PARTS}


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: code_lines
_C_HEAD = ("def code_lines(code):\n"
           "    count = 0\n"
           '    for line in code.split("\\n"):\n'
           "        text = line.strip()\n")
_C_END = ("            count = count + 1\n"
          "    return count\n")


def _c(condition: str) -> str:
    return _C_HEAD + f"        if {condition}:\n" + _C_END


_C_START = _c("...")
_C = _c('text != "" and text[0] != "#"')
_C_COMMENTS = _c('text != ""')
_C_EVERY = _c("True")
_C_INDENTED = _c('line != "" and line[0] != "#"')
_C_HASH = _c('text != "" and "#" not in text')

# a short piece of code with a comment at the start of a line, an indented comment, an empty line and a line of spaces
_SHORT = 'import math\n\n# the model\ndef f(x):\n    # one line of work\n    return math.exp(x)\n   \n'

# exercise 2: best_in_budget
_B_HEAD = ("def best_in_budget(models, budget):\n"
           "    best_name = None\n"
           "    best_accuracy = 0\n"
           "    for name, accuracy, ms in models:\n")
_B_END = ("            best_name = name\n"
          "            best_accuracy = accuracy\n"
          "    return best_name\n")


def _b(condition: str) -> str:
    return _B_HEAD + f"        if {condition}:\n" + _B_END


_B_START = _b("...")
_B = _b("ms <= budget and accuracy > best_accuracy")
_B_NO_BUDGET = _b("accuracy > best_accuracy")
_B_OVER = _b("ms > budget and accuracy > best_accuracy")
_B_STRICT = _b("ms < budget and accuracy > best_accuracy")

EXERCISES = {
    "count-lines": FunctionBox(
        name="count-lines",
        function="code_lines",
        start=_C_START,
        solution=_C,
        visible=[(_Part("model"),)],
        hidden=[(_Part(p),) for p in PARTS if p != "model"] + [(_SHORT,), ("",), ("# only a comment\n\n    # and another\n",)],
        first=[_DOTS],
        picture=lines_picture,
        mistakes=[
            Mistake("empty-line", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an IndexError on an empty line: an empty text has no first character. `and` looks at its right side only when its left side is True: does your condition first make sure that `text` is not empty?",
                    nl="Met `{call}` stopt je functie met een IndexError bij een lege regel: een lege tekst heeft geen eerste teken. `and` kijkt alleen naar de rechterkant als de linkerkant True is: zorgt je voorwaarde eerst dat `text` niet leeg is?"),
            Mistake("comments", wrong=_C_COMMENTS,
                    en="`{call}` gives {got}, and it should give {want}: your function also counts the comments. A comment line begins with `#` once its spaces are gone: how can your condition leave such a line out?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de commentaarregels ook. Een commentaarregel begint met `#` zodra de spaties weg zijn: hoe laat je voorwaarde zo'n regel weg?"),
            Mistake("every-line", wrong=_C_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: the number of all lines, also the empty ones and the comments. Which lines should your condition let through?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal van alle regels, ook de lege en de commentaarregels. Welke regels moet je voorwaarde doorlaten?"),
            Mistake("hash-anywhere", wrong=_C_HASH,
                    en="`{call}` gives {got}, and it should give {want}: your function also leaves out lines of code with a comment at their end, such as the lines of `SETTINGS`. Where must the `#` be for a line to be a comment?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie laat ook regels code weg met commentaar aan het eind, zoals de regels van `SETTINGS`. Waar moet de `#` staan als een regel commentaar is?"),
            Mistake("indented", wrong=_C_INDENTED,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the comments inside a function, and lines of only spaces. `line` still has the spaces at its start, so its first character is a space. Which name holds the line without them?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de commentaarregels binnen een functie, en regels met alleen spaties. `line` heeft de spaties aan het begin nog, dus is het eerste teken een spatie. Welke naam bevat de regel zonder de spaties?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition for a line of code: `text` is not empty, and does not begin with `#`.",
                      "nl": "Vervang de `...` door de voorwaarde voor een regel code: `text` is niet leeg, en begint niet met `#`."},
            "arguments": {"en": "The check calls `{call}`, with the code of a part as one text, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(code)`?",
                          "nl": "De check roept `{call}` aan, met de code van een deel als één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(code)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your condition let through every line that is not empty and not a comment, and nothing else?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Laat je voorwaarde elke regel door die niet leeg is en geen commentaar, en verder niets?"},
            "almost": {"en": "Right for the model's part. But `{call}` gives {got}, and it should give {want}. Is there a line of code with a comment at its end, or a line of only spaces, in that text? And does your function count the lines of the `code` it gets, not a fixed number?",
                       "nl": "Goed voor het deel van het model. Maar `{call}` geeft {got}, en het moet {want} zijn. Staat er in deze tekst een regel code met commentaar aan het eind, of een regel met alleen spaties? En telt je functie de regels van de `code` die de functie krijgt, en geen vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return count`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return count`?"},
            "correct": {"en": "Right in all {n} cases: the model's part has 5 lines of code, and the whole service 118. In the picture, which part of the service has the most lines?",
                        "nl": "Goed in alle {n} gevallen: het deel van het model heeft 5 regels code, en de hele dienst 118. Welk deel van de dienst heeft in de afbeelding de meeste regels?"},
        },
    ),
    "budget": FunctionBox(
        name="budget",
        function="best_in_budget",
        start=_B_START,
        solution=_B,
        visible=[(MODELS, BUDGET)],
        hidden=[(MODELS, 1000.0), (MODELS, 1.5), (MODELS, 0.001), ([], 10.0),
                ([("big", 0.99, 900.0), ("tiny", 0.9, 2.0), ("middle", 0.95, 8.0)], 10.0)],
        first=[_DOTS],
        picture=budget_picture,
        mistakes=[
            Mistake("no-budget", wrong=_B_NO_BUDGET,
                    en="`{call}` gives {got}, and it should give {want}: the most accurate model of all, but one message takes it longer than the budget allows. Does your condition skip every model with an `ms` above `budget`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het model met de hoogste accuracy van allemaal, alleen duurt één bericht bij dit model langer dan het budget toelaat. Slaat je voorwaarde elk model met een `ms` boven `budget` over?"),
            Mistake("over-budget", wrong=_B_OVER,
                    en="`{call}` gives {got}, and it should give {want}: a model that takes longer than the budget. Which models fit within the budget: those whose `ms` is above `budget`, or the others?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een model dat langer duurt dan het budget. Welke modellen passen binnen het budget: de modellen waarvan `ms` boven `budget` ligt, of de andere?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition: the model fits within `budget`, and its `accuracy` is higher than `best_accuracy`.",
                      "nl": "Vervang de `...` door de voorwaarde: het model past binnen `budget`, en de `accuracy` is hoger dan `best_accuracy`."},
            "arguments": {"en": "The check calls `{call}`, with a list of models and a budget, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(models, budget)`?",
                          "nl": "De check roept `{call}` aan, met een lijst met modellen en een budget, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(models, budget)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your condition let through only models that fit within `budget`, and of those only one that beats `best_accuracy`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Laat je voorwaarde alleen modellen door die binnen `budget` passen, en daarvan alleen een model dat `best_accuracy` verslaat?"},
            "almost": {"en": "Right for a budget of 10 ms. But `{call}` gives {got}, and it should give {want}. Does a model that takes exactly `budget` milliseconds still fit, and does your function use `budget` and not a fixed number?",
                       "nl": "Goed voor een budget van 10 ms. Maar `{call}` geeft {got}, en het moet {want} zijn. Past een model dat precies `budget` milliseconden duurt nog, en gebruikt je functie `budget` en geen vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return best_name`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return best_name`?"},
            "correct": {"en": "Right in all {n} cases: with 10 ms per message, the small network, which is 0.02 less accurate than the large language model and more than 500 times as fast. In the picture, which model does your function choose with 1 ms?",
                        "nl": "Goed in alle {n} gevallen: met 10 ms per bericht het kleine netwerk, met een accuracy 0,02 lager dan het large language model, en ruim 500 keer zo snel. Welk model kiest je functie in de afbeelding bij 1 ms?"},
        },
    ),
}

# The lesson is done when these are correct. "cace" and "loop" are the choice questions after the demo.
REQUIRED = ["count-lines", "budget", "cace", "loop"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("count-lines", _C_START, "empty", None),
    ("count-lines", _C, "correct", None),
    ("count-lines", _c('text != "" and ...'), "wrong", "dots-left"),
    ("count-lines", _c('text[0] != "#"'), "error", "empty-line"),
    ("count-lines", _C_COMMENTS, "wrong", "comments"),
    ("count-lines", _C_EVERY, "wrong", "every-line"),
    ("count-lines", _C_INDENTED, "wrong", "indented"),
    ("count-lines", _C_HASH, "wrong", "hash-anywhere"),
    ("count-lines", _C.replace("    return count\n", ""), "wrong", "no-return"),
    # other right answers: the same lines written another way
    ("count-lines", _c('len(text) > 0 and text[0] != "#"'), "correct", None),
    ("count-lines", _c('text != "" and not text.startswith("#")'), "correct", None),
    # a fixed number: right for the model's part, wrong for every other
    ("count-lines", "def code_lines(code):\n    return 5\n", "almost", None),
    ("budget", _B_START, "empty", None),
    ("budget", _B, "correct", None),
    ("budget", _b("... and accuracy > best_accuracy"), "wrong", "dots-left"),
    ("budget", _B_NO_BUDGET, "wrong", "no-budget"),
    ("budget", _B_OVER, "wrong", "over-budget"),
    ("budget", _B.replace("    return best_name\n", ""), "wrong", "no-return"),
    ("budget", _b("accuracy > best_accuracy and ms <= budget"), "correct", None),
    # a model that takes exactly the budget left out: right for 10 ms, wrong for 1.5 ms
    ("budget", _B_STRICT, "almost", None),
]
