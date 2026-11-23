# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-entailment: the labels of the hypotheses with one word (label_shares), a guess of the label
from the hypothesis alone with the scores of naive Bayes (guess), and the share of the words of a hypothesis that its
premise has (overlap); with the pictures of what each function gives on the pairs of SNLI, three pictures of the idea
and a demo.

The pairs are a sample of SNLI 1.0 that the course ships (everything_ai.snli): the pairs of 10,000 premises of its
training set and the labeled pairs of its test set, so nothing is downloaded. The models are naive Bayes as in the
lesson about spam (lm-naive-bayes), trained here on the training pairs. Every number in the text and the pictures is
computed by the course (tests/test_nlp_entailment.py computes them again)."""
import ast
import contextlib
import html
import io
import math
import re
from types import MappingProxyType

from everything_ai import snli, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

_KEY = "nlp-entailment"

# --- the course's data and models ------------------------------------------------------------------------------------

TRAIN = snli.split("train")   # 36,788 pairs: (premise words, hypothesis words, label)
TEST = snli.split("test")     # 9,824
LABELS = snli.LABELS          # entailment, neutral, contradiction
CUES = ("nobody", "sleeping", "cat", "outdoors", "animal", "tall", "because")

# The example of the instructions of SNLI (Bowman and others 2015, figure 1), with its three hypotheses.
GUIDE_PREMISE = "Two dogs are running through a field."
GUIDE = (("entailment", "There are animals outdoors."),
         ("neutral", "Some puppies are running to catch a stick."),
         ("contradiction", "The pets are sitting on a couch."))
# The pair of McCoy, Pavlick and Linzen (2019) for the heuristic of lexical overlap: every word comes back, and the
# hypothesis does not follow.
DOCTOR = ("The doctor was paid by the actor.", "The doctor paid the actor.")


def hypothesis_words(premise, hypothesis) -> list:
    """What the first model reads: only the words of the hypothesis."""
    return list(hypothesis)


def both_words(premise, hypothesis) -> list:
    """What the second model reads: the words of both sentences in one bag, each marked with its sentence."""
    return ["p:" + w for w in premise] + ["h:" + w for w in hypothesis]


def marked_words(premise, hypothesis) -> list:
    """What the third model reads: each word of the hypothesis, marked = when the premise has it and + when it is new."""
    seen = set(premise)
    return [("=" if w in seen else "+") + w for w in hypothesis]


def _model(features):
    """Naive Bayes as in lm-naive-bayes, trained on the training pairs: log P(label), and for each word of the
    vocabulary log P(word | label) with Laplace smoothing. Both as read-only tables."""
    counts, pairs = {lab: {} for lab in LABELS}, {lab: 0 for lab in LABELS}
    for premise, hypothesis, label in TRAIN:
        pairs[label] += 1
        for w in features(premise, hypothesis):
            counts[label][w] = counts[label].get(w, 0) + 1
    vocab = sorted(set().union(*counts.values()))
    log_prior = {lab: math.log(pairs[lab] / len(TRAIN)) for lab in LABELS}
    scores = {}
    for lab in LABELS:
        total = sum(counts[lab].values())
        scores[lab] = MappingProxyType({w: math.log((counts[lab].get(w, 0) + 1) / (total + len(vocab)))
                                        for w in vocab})
    return MappingProxyType(log_prior), MappingProxyType(scores)


LOG_PRIOR, WORD_SCORES = kept(f"{_KEY}:hypothesis", lambda: _model(hypothesis_words))
_BOTH = kept(f"{_KEY}:both", lambda: _model(both_words))
_MARKED = kept(f"{_KEY}:marked", lambda: _model(marked_words))


# --- the course's own exercises --------------------------------------------------------------------------------------

def label_shares(pairs, word):
    """Exercise 1: of the hypotheses that have the word, the share with each label; 0.0 for each label when no
    hypothesis has it."""
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    total = 0
    for premise, hypothesis, label in pairs:
        if word in hypothesis:
            counts[label] = counts[label] + 1
            total = total + 1
    shares = {}
    for label in counts:
        if total > 0:
            shares[label] = counts[label] / total
        else:
            shares[label] = 0.0
    return shares


def guess(hypothesis, log_prior, word_scores):
    """Exercise 2: the label with the highest log score: its log prior plus the score of each word of the hypothesis
    that the dict of that label has."""
    best = None
    for label in log_prior:
        score = log_prior[label]
        for word in hypothesis:
            if word in word_scores[label]:
                score = score + word_scores[label][word]
        if best is None:
            best = label
            best_score = score
        if score > best_score:
            best = label
            best_score = score
    return best


def overlap(premise, hypothesis):
    """Exercise 3: the share of the words of the hypothesis that the premise has; 0.0 for a hypothesis without words."""
    if len(hypothesis) == 0:
        return 0.0
    in_premise = set(premise)
    inside = 0
    for word in hypothesis:
        if word in in_premise:
            inside = inside + 1
    return inside / len(hypothesis)


# --- the numbers in the text -----------------------------------------------------------------------------------------

_MODELS = {"hypothesis": ((LOG_PRIOR, WORD_SCORES), hypothesis_words), "both": (_BOTH, both_words),
           "marked": (_MARKED, marked_words)}


def scores_of(words, model) -> dict:
    """The log score of each label for these words, with a model of _MODELS."""
    (log_prior, word_scores), _ = _MODELS[model]
    return {lab: log_prior[lab] + sum(word_scores[lab][w] for w in words if w in word_scores[lab]) for lab in LABELS}


def numbers() -> dict:
    """Every number the lesson states, computed from the data and the models."""
    def _go():
        out = {"train": len(TRAIN), "test": len(TEST), "premises": len({r.premise for r in snli.rows() if r.split == "train"}),
               "train_labels": {lab: sum(r[2] == lab for r in TRAIN) for lab in LABELS},
               "test_labels": {lab: sum(r[2] == lab for r in TEST) for lab in LABELS}}
        majority = max(LABELS, key=lambda lab: out["train_labels"][lab])
        out["majority"] = majority
        out["majority_acc"] = out["test_labels"][majority] / len(TEST)
        for key, ((log_prior, word_scores), features) in _MODELS.items():
            guesses = [guess(features(p, h), log_prior, word_scores) for p, h, _ in TEST]
            right = {lab: sum(g == lab == r[2] for g, r in zip(guesses, TEST)) for lab in LABELS}
            out[key] = {"acc": sum(right.values()) / len(TEST), "right": sum(right.values()),
                        "per_label": {lab: right[lab] / out["test_labels"][lab] for lab in LABELS},
                        "vocab": len(word_scores[LABELS[0]])}
        out["cues"] = {w: {lab: sum(w in h and lab == label for _, h, label in TRAIN) for lab in LABELS} for w in CUES}
        laps = [(overlap(p, h), label) for p, h, label in TEST]
        out["overlap"] = {lab: {"mean": sum(o for o, x in laps if x == lab) / out["test_labels"][lab],
                                "full": sum(o == 1 for o, x in laps if x == lab),
                                "full_share": sum(o == 1 for o, x in laps if x == lab) / out["test_labels"][lab]}
                          for lab in LABELS}
        out["full"] = sum(o == 1 for o, _ in laps)
        premise, hypothesis = (snli.words(s) for s in DOCTOR)
        out["doctor"] = {"overlap": overlap(premise, hypothesis),
                         "hypothesis": guess(hypothesis, LOG_PRIOR, WORD_SCORES),
                         "marked": guess(marked_words(premise, hypothesis), *_MARKED)}
        out["guide"] = [guess(snli.words(h), LOG_PRIOR, WORD_SCORES) for _, h in GUIDE]
        return out
    return kept(f"{_KEY}:numbers", _go)


# --- pictures: helpers -----------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_ROW = "display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);width:11rem;flex:none"
_TRACK = "flex:1;min-width:3rem;height:0.65rem;background:var(--ea-line);border-radius:2px"
_NUM = "font-family:var(--ea-font-code);font-size:0.8rem;width:3.6rem;flex:none;text-align:right"
_GRID = "display:grid;grid-template-columns:5.5rem repeat(3,minmax(0,1fr)) 2.6rem;gap:0.2rem 0.5rem;align-items:center"
_CELL = "display:flex;align-items:center;gap:0.3rem;min-width:0"
_SMALL = "font-family:var(--ea-font-code);font-size:0.72rem;flex:none"
_CODE = "font-family:var(--ea-font-code);font-size:0.82rem"
_SENTENCE = "font-family:var(--ea-font-ui);font-size:0.95rem;line-height:1.5"
_CHIP = ("display:inline-block;font-family:var(--ea-font-code);font-size:0.72rem;border:1px solid var(--ea-line);"
         "border-radius:4px;padding:0 0.3rem;margin-right:0.4rem")
_PART = "display:inline-block;margin:0 0.2rem 0.3rem 0;text-align:center;vertical-align:top"
_WORD = ("display:block;border-radius:5px;padding:0.15rem 0.4rem;font-family:var(--ea-font-code);font-size:0.9rem;"
         "color:var(--ea-ink);border:1px solid var(--ea-line)")
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;padding:0 0.15rem"
_ROLE = {   # the line or bar and the color of its text: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}
_SHORT = {"entailment": "E", "neutral": "N", "contradiction": "C"}


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _said(template: str, **values) -> tuple:
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value of the
    learner's stays as it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 9,824 in English, 9.824 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _d(x: float, lang: str, digits: int = 3) -> str:
    """A number from 0 to 1 with three decimals: 0.608 in English, 0,608 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _short(v, limit: int = 60) -> str:
    text = _repr(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _is_share(v) -> bool:
    """A number from 0 to 1, as a share or an accuracy is."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def _same_number(a, b) -> bool:
    return _is_share(a) and abs(float(a) - b) < 1e-9


def _bar(label: str, value, role: str, lang: str, off: bool = False) -> str:
    """A row with a label, a bar from 0 to 1 and the number; a value that is not a number from 0 to 1 gets no bar."""
    line, color = _ROLE[role]
    fine = _is_share(value)
    width = float(value) * 100 if fine else 0.0
    number = _d(float(value), lang) if fine else _short(value, 12)
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(label)}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(number)}</span></div>')


def _mini(value, role: str, lang: str, off: bool = False) -> str:
    """A cell of a grid: a short bar from 0 to 1 and the number."""
    line, color = _ROLE[role]
    fine = _is_share(value)
    width = float(value) * 100 if fine else 0.0
    number = _d(float(value), lang, 2) if fine else _short(value, 8)
    style = _SMALL + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<span style="{_CELL}"><span style="{_TRACK};min-width:1.5rem"><span style="display:block;'
            f'width:{width:.1f}%;height:100%;background:{line};border-radius:2px"></span></span>'
            f'<span style="{style}">{html.escape(number)}</span></span>')


def _grid(rows: list) -> str:
    """The grid of words and labels: a header, then per word its cells and the number of hypotheses with it."""
    head = "".join(f'<span style="{_SMALL};color:var(--ea-ink-2)">{x}</span>' for x in ("", *LABELS, "n"))
    return f'<div style="{_GRID}">{head}{"".join(rows)}</div>'


def _grid_row(word: str, cells: str, n) -> str:
    return (f'<span style="{_CODE};color:var(--ea-ink)">{html.escape(word)}</span>{cells}'
            f'<span style="{_SMALL};color:var(--ea-ink-2);text-align:right">{html.escape(str(n))}</span>')


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + f'<div style="{_UI}">{" ".join(h for h, _ in notes)}</div>'
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


# --- pictures of the idea --------------------------------------------------------------------------------------------

_GUIDE_WORDS = {
    "en": {"title": "One premise, three hypotheses: the example of the instructions of SNLI",
           "premise": "Premise", "hypotheses": "Hypotheses, with the label people gave", "only": "hypothesis only:",
           "legend": "Gray: the sentences and labels people wrote. In the color of the text: the label that the "
                     "course's model gives from the hypothesis alone, without reading the premise: {right} of 3 right."},
    "nl": {"title": "Eén premisse, drie hypothesen: het voorbeeld uit de instructies van SNLI",
           "premise": "Premisse", "hypotheses": "Hypothesen, met het label dat mensen gaven", "only": "alleen hypothese:",
           "legend": "Grijs: de zinnen en labels die mensen schreven. In de kleur van de tekst: het label dat het model "
                     "van de course geeft op de hypothese alleen, zonder de premisse te lezen: {right} van de 3 goed."},
}


def guide_picture(lang: str) -> str:
    """The example of the instructions of SNLI, with the label of the hypothesis-only model beside each hypothesis."""
    w, nums = _GUIDE_WORDS[lang], numbers()
    rows = [_head(w["premise"]), f'<div style="{_SENTENCE};color:var(--ea-ink-2)">{html.escape(GUIDE_PREMISE)}</div>',
            _head(w["hypotheses"])]
    said = [f'{_bare(w["premise"])}: {GUIDE_PREMISE}']
    for (label, text), got in zip(GUIDE, nums["guide"]):
        rows.append(f'<div style="{_ROW};flex-wrap:wrap"><span style="{_CHIP};color:var(--ea-ink-2)">{label}</span>'
                    f'<span style="{_SENTENCE};color:var(--ea-ink-2);flex:1;min-width:12rem">{html.escape(text)}</span>'
                    f'<span style="{_SMALL};color:var(--ea-ink)">{html.escape(w["only"])} {got}</span></div>')
        said.append(f'{label}: {text} ({w["only"]} {got})')
    right = sum(got == label for (label, _), got in zip(GUIDE, nums["guide"]))
    return _figure(w["title"], rows, said, [_said(w["legend"], right=right)])


_CUE_WORDS = {
    "en": {"title": "Seven words, and the labels of the hypotheses that have them",
           "legend": "Each bar: the share of the training hypotheses with that word that have this label; a full bar "
                     "is 1. n: the number of training hypotheses with the word. In the color of the text: the course's "
                     "counts on the {pairs} training pairs; all three labels are about as common there."},
    "nl": {"title": "Zeven woorden, en de labels van de hypothesen met die woorden",
           "legend": "Elke balk: het deel van de hypothesen uit de training set met dat woord dat dit label heeft; een "
                     "volle balk is 1. n: het aantal hypothesen uit de training set met het woord. In de kleur van de "
                     "tekst: de tellingen van de course op de {pairs} trainingsparen; daar komen alle drie de labels "
                     "ongeveer even vaak voor."},
}


def cue_picture(lang: str) -> str:
    """For each word of CUES, the share of each label among the training hypotheses that have it."""
    w, nums = _CUE_WORDS[lang], numbers()
    rows, said = [], []
    for word in CUES:
        c = nums["cues"][word]
        n = sum(c.values())
        cells = "".join(_mini(c[lab] / n, "course", lang) for lab in LABELS)
        rows.append(_grid_row(word, cells, _n(n, lang)))
        said.append(f'{word}: ' + ", ".join(f"{lab} {c[lab]}" for lab in LABELS))
    return _figure(w["title"], [_grid(rows)], said, [_said(w["legend"], pairs=_n(nums["train"], lang))])


_MODEL_WORDS = {
    "en": {"title": "Four ways to label the test pairs",
           "majority": "Always entailment", "hypothesis": "Hypothesis only", "both": "Both sentences, one bag",
           "marked": "Hypothesis, words marked old or new",
           "legend": "Each bar: the share of the {test} test pairs that a way labels right; a full bar is 1. In the "
                     "color of the text: what the course measured. The three models are naive Bayes, trained on the "
                     "{train} training pairs; they differ only in the words they read."},
    "nl": {"title": "Vier manieren om de testparen te labelen",
           "majority": "Altijd entailment", "hypothesis": "Alleen de hypothese", "both": "Beide zinnen, één zak",
           "marked": "Hypothese, woorden gemarkeerd als oud of nieuw",
           "legend": "Elke balk: het deel van de {test} testparen dat een manier goed labelt; een volle balk is 1. In de "
                     "kleur van de tekst: wat de course mat. De drie modellen zijn naive Bayes, getraind op de {train} "
                     "trainingsparen; de modellen verschillen alleen in de woorden die elk model leest."},
}


def models_picture(lang: str) -> str:
    """The accuracy on the test pairs of always entailment and of the three models of naive Bayes."""
    w, nums = _MODEL_WORDS[lang], numbers()
    values = (("majority", nums["majority_acc"]), ("hypothesis", nums["hypothesis"]["acc"]),
              ("both", nums["both"]["acc"]), ("marked", nums["marked"]["acc"]))
    rows = [_bar(w[k], v, "course", lang) for k, v in values]
    said = [f"{_bare(w[k])}: {_d(v, lang)}" for k, v in values]
    return _figure(w["title"], rows, said, [_said(w["legend"], test=_n(nums["test"], lang),
                                                   train=_n(nums["train"], lang))])


# --- the demo --------------------------------------------------------------------------------------------------------

DEMO = (GUIDE_PREMISE, GUIDE[0][1])
_LIMIT = 300   # characters of a sentence of the demo that the picture reads
_SHOWN = 30    # words of the hypothesis that the picture shows

_DEMO_WORDS = {
    "en": {"title": "Three models on your pair of sentences",
           "words": "The words of the hypothesis",
           "guesses": "What each model says",
           "hypothesis": "Hypothesis only", "both": "Both sentences, one bag",
           "marked": "Hypothesis, words marked old or new",
           "legend": "Under each word: = when the premise has the word, + when it is new, and the label whose score for "
                     "the word is highest in C.WORD_SCORES; a dash for a word that no training hypothesis has. In the "
                     "color of the text: what the course's models give.",
           "overlap": "Overlap: {share}; the premise has {inside} of the {n} words of the hypothesis.",
           "more": "The picture shows the first {shown} of the {n} words.",
           "empty": "Type a premise and a hypothesis, each with at least one word."},
    "nl": {"title": "Drie modellen op jouw zinnenpaar",
           "words": "De woorden van de hypothese",
           "guesses": "Wat elk model zegt",
           "hypothesis": "Alleen de hypothese", "both": "Beide zinnen, één zak",
           "marked": "Hypothese, woorden gemarkeerd als oud of nieuw",
           "legend": "Onder elk woord: = als de premisse het woord heeft, + als het nieuw is, en het label met de hoogste "
                     "score voor het woord in C.WORD_SCORES; een streepje voor een woord dat in geen enkele hypothese "
                     "uit de training set staat. In de kleur van de tekst: wat de modellen van de course geven.",
           "overlap": "Overlap: {share}; de premisse heeft {inside} van de {n} woorden van de hypothese.",
           "more": "De afbeelding laat de eerste {shown} van de {n} woorden zien.",
           "empty": "Typ een premisse en een hypothese, elk met minstens één woord."},
}


def strongest(word) -> str:
    """The label whose score for the word is highest in WORD_SCORES, or "" for a word it does not have."""
    if word not in WORD_SCORES[LABELS[0]]:
        return ""
    return max(LABELS, key=lambda lab: WORD_SCORES[lab][word])


def demo_picture(premise, hypothesis, lang: str) -> str:
    """The three models of naive Bayes on a pair the learner types, with the words of the hypothesis and its overlap."""
    w = _DEMO_WORDS[lang]
    p_words = snli.words(str(premise or "")[:_LIMIT])
    h_words = snli.words(str(hypothesis or "")[:_LIMIT])
    if not p_words or not h_words:
        return _figure(w["title"], [], [], [_said(w["empty"])])
    seen = set(p_words)
    parts, said_words = [], []
    for word in h_words[:_SHOWN]:
        mark = "=" if word in seen else "+"
        label = strongest(word)
        parts.append(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(word)}</span>'
                     f'<span style="{_TAG};color:var(--ea-ink)">{mark} {html.escape(label or "–")}</span></span>')
        said_words.append(f"{word} ({mark} {label or '–'})")
    rows = [_head(w["words"]), f'<div>{"".join(parts)}</div>', _head(w["guesses"])]
    said = [f'{_bare(w["words"])}: {", ".join(said_words)}']
    guesses = (("hypothesis", guess(h_words, LOG_PRIOR, WORD_SCORES)), ("both", guess(both_words(p_words, h_words), *_BOTH)),
               ("marked", guess(marked_words(p_words, h_words), *_MARKED)))
    for key, label in guesses:
        rows.append(f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(w[key])}</span>'
                    f'<span style="{_CODE};color:var(--ea-ink)">{label}</span></div>')
        said.append(f"{_bare(w[key])}: {label}")
    inside = sum(x in seen for x in h_words)
    notes = [_said(w["legend"]), _said(w["overlap"], inside=inside, n=len(h_words),
                                       share=_d(overlap(p_words, h_words), lang))]
    if len(h_words) > _SHOWN:
        notes.append(_said(w["more"], shown=_SHOWN, n=len(h_words)))
    return _figure(w["title"], rows, said, notes)


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of a picture


def _quiet(fn, args):
    """The learner's function without what it prints; _STOPPED when it stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _quiet_many(fn, calls) -> list:
    """The learner's function on many calls, without what it prints; _STOPPED for a call that stops with an error."""
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for args in calls:
            try:
                out.append(fn(*args))
            except Timeout:
                raise
            except Exception:   # noqa: BLE001   the check above says what went wrong
                out.append(_STOPPED)
    return out


def _timed(make):
    """make() within PICTURE_SECONDS, or _TOO_SLOW when it takes longer."""
    try:
        return with_time_limit(make, PICTURE_SECONDS)
    except Timeout:
        return _TOO_SLOW


_SLOW = {"en": "Your function takes too long on all the pairs. Does a loop of yours go on longer than it needs to?",
         "nl": "Je functie duurt te lang op alle paren. Gaat een van je loops langer door dan nodig is?"}


def _is_label(v) -> bool:
    """One of the three labels, as a string."""
    return isinstance(v, str) and v in LABELS


def _shares(v):
    """What label_shares gives as a share for each of the three labels, or None."""
    if isinstance(v, dict) and set(v) == set(LABELS) and all(_is_share(v[lab]) for lab in LABELS):
        return {lab: float(v[lab]) for lab in LABELS}
    return None


_SHARES_WORDS = {
    "en": {"title": "Your label_shares on seven words of the training pairs",
           "blue": "Blue: the share of each label that your label_shares gives, over the {pairs} training pairs; a full "
                   "bar is 1. n: the number of training hypotheses with the word, as the course counted them.",
           "odd": "For *{word}*, your function gives {got} instead of a share for each of the three labels.",
           "odd_more": "For {k} of the seven words, such as *{word}*, your function gives {got} instead of a share for "
                       "each of the three labels.",
           "stop": "For *{word}*, your function stops with an error.",
           "stop_more": "For {k} of the seven words, such as *{word}*, your function stops with an error.",
           "stop_all": "For each of the seven words, your function stops with an error.",
           "odd_all": "For each of the seven words, your function gives something other than a share for each of the "
                      "three labels, such as {got} for *{word}*.",
           "dashed": "Purple dashed: a share that differs from the course's; for *{word}*, the course gets {want}.",
           "lean": "Without reading a premise, *nobody* goes with contradiction in {nobody} of the {n_nobody} hypotheses "
                   "that have the word, and *outdoors* with entailment in {outdoors} of {n_outdoors}."},
    "nl": {"title": "Jouw label_shares op zeven woorden uit de trainingsparen",
           "blue": "Blauw: het deel van elk label dat je label_shares geeft, over de {pairs} trainingsparen; een volle "
                   "balk is 1. n: het aantal hypothesen uit de training set met het woord, geteld door de course.",
           "odd": "Voor *{word}* geeft je functie {got} in plaats van een deel voor elk van de drie labels.",
           "odd_more": "Voor {k} van de zeven woorden, zoals *{word}*, geeft je functie {got} in plaats van een deel "
                       "voor elk van de drie labels.",
           "stop": "Voor *{word}* stopt je functie met een fout.",
           "stop_more": "Voor {k} van de zeven woorden, zoals *{word}*, stopt je functie met een fout.",
           "stop_all": "Voor elk van de zeven woorden stopt je functie met een fout.",
           "odd_all": "Voor elk van de zeven woorden geeft je functie iets anders dan een deel voor elk van de drie "
                      "labels, zoals {got} voor *{word}*.",
           "dashed": "Paars gestreept: een deel dat afwijkt van het deel van de course; voor *{word}* krijgt de course "
                     "{want}.",
           "lean": "Zonder één premisse te lezen hoort *nobody* bij contradiction in {nobody} van de {n_nobody} hypothesen met "
                   "het woord, en *outdoors* bij entailment in {outdoors} van de {n_outdoors}."},
}


def shares_picture(seen) -> str:
    """The learner's label_shares on the training pairs for each word of CUES, as a grid of bars."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _SHARES_WORDS[lang], numbers()
    got = _timed(lambda: {word: _quiet(fn, (TRAIN, word)) for word in CUES})
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, differs, stopped, odd = [], [], [], [], []
    for word in CUES:
        c = nums["cues"][word]
        n = sum(c.values())
        want = {lab: c[lab] / n for lab in LABELS}
        v = got[word]
        if v is _STOPPED:
            stopped.append(word)
            continue
        mine = _shares(v)
        if mine is None:
            odd.append((word, v))
            continue
        cells = []
        for lab in LABELS:
            off = not _same_number(mine[lab], want[lab])
            if off and word not in differs:
                differs.append(word)
            cells.append(_mini(mine[lab], "mine", lang, off=off))
        rows.append(_grid_row(word, "".join(cells), _n(n, lang)))
        said.append(f"{word}: " + ", ".join(f"{lab} {_d(mine[lab], lang)}" for lab in LABELS))
    notes = []
    if stopped:
        key = "stop" if len(stopped) == 1 else "stop_all" if len(stopped) == len(CUES) else "stop_more"
        notes.append(_said(w[key], k=len(stopped), word=stopped[0]))
    if odd:
        key = "odd" if len(odd) == 1 else "odd_all" if len(odd) == len(CUES) else "odd_more"
        notes.append(_said(w[key], k=len(odd), word=odd[0][0], got=_short(odd[0][1])))
    extra = [_said(w["blue"], pairs=_n(nums["train"], lang))] if rows else []
    if differs:
        c = nums["cues"][differs[0]]
        want = ", ".join(f"{lab} {_d(c[lab] / sum(c.values()), lang)}" for lab in LABELS)
        extra.append(_said(w["dashed"], word=differs[0], want=want))
    elif len(rows) == len(CUES):
        cn, co = nums["cues"]["nobody"], nums["cues"]["outdoors"]
        extra.append(_said(w["lean"], nobody=cn["contradiction"], n_nobody=sum(cn.values()),
                           outdoors=co["entailment"], n_outdoors=sum(co.values())))
    return _figure(w["title"], [_grid(rows)] if rows else [], said, extra + notes)


_GUESS_WORDS = {
    "en": {"title": "Your guess on the test pairs, from the hypothesis alone",
           "all": "All test pairs", "majority": "Always entailment", "per": "Right, per label people gave",
           "blue": "Blue: the share of the {test} test pairs whose label your guess gets right, with C.LOG_PRIOR and "
                   "C.WORD_SCORES; below, the same per label people gave; a full bar is 1. In the color of the text: "
                   "always entailment, for comparison.",
           "odd": "For {n} of the test pairs, your function gives something that is not one of the three labels, such "
                  "as {got}.",
           "stop": "For {n} of the test pairs, your function stops with an error.",
           "dashed": "Purple dashed: a share that differs from the course's; the course gets {want} on all test pairs.",
           "result": "{right} of the {test} pairs right, and not a single premise read."},
    "nl": {"title": "Jouw guess op de testparen, op de hypothese alleen",
           "all": "Alle testparen", "majority": "Altijd entailment", "per": "Goed, per label dat mensen gaven",
           "blue": "Blauw: het deel van de {test} testparen waarvan je guess het label goed heeft, met C.LOG_PRIOR en "
                   "C.WORD_SCORES; daaronder hetzelfde per label dat mensen gaven; een volle balk is 1. In de kleur van "
                   "de tekst: altijd entailment, om mee te vergelijken.",
           "odd": "Voor {n} van de testparen geeft je functie iets wat geen van de drie labels is, zoals {got}.",
           "stop": "Voor {n} van de testparen stopt je functie met een fout.",
           "dashed": "Paars gestreept: een deel dat afwijkt van het deel van de course; de course krijgt {want} op alle "
                     "testparen.",
           "result": "{right} van de {test} paren goed, zonder één premisse te lezen."},
}


def guess_picture(seen) -> str:
    """The learner's guess with the course's tables on every hypothesis of the test pairs: the share right, over all
    pairs and per label, beside always entailment."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _GUESS_WORDS[lang], numbers()
    scores = {lab: dict(WORD_SCORES[lab]) for lab in LABELS}   # the learner's own copy, one for all calls
    got = _timed(lambda: _quiet_many(fn, [(list(h), dict(LOG_PRIOR), scores) for _, h, _ in TEST]))   # a prior each
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    stopped = sum(g is _STOPPED for g in got)
    odd = [g for g in got if g is not _STOPPED and not _is_label(g)]
    notes = []
    if stopped:
        notes.append(_said(w["stop"], n=_n(stopped, lang)))
    if odd:
        notes.append(_said(w["odd"], n=_n(len(odd), lang), got=_short(odd[0])))
    if stopped == len(got):
        return _figure(w["title"], [], [], notes)
    want = nums["hypothesis"]
    labels = [g if _is_label(g) else None for g in got]
    right = {lab: sum(g == lab == r[2] for g, r in zip(labels, TEST)) for lab in LABELS}
    acc = sum(right.values()) / len(TEST)
    per = {lab: right[lab] / nums["test_labels"][lab] for lab in LABELS}
    off_all = not _same_number(acc, want["acc"])
    rows = [_bar(w["all"], acc, "mine", lang, off=off_all), _bar(w["majority"], nums["majority_acc"], "course", lang),
            _head(w["per"])]
    rows += [_bar(lab, per[lab], "mine", lang, off=not _same_number(per[lab], want["per_label"][lab]))
             for lab in LABELS]
    said = [f'{_bare(w["all"])}: {_d(acc, lang)}', f'{_bare(w["majority"])}: {_d(nums["majority_acc"], lang)}']
    said += [f"{lab}: {_d(per[lab], lang)}" for lab in LABELS]
    extra = [_said(w["blue"], test=_n(nums["test"], lang))]
    if off_all:
        extra.append(_said(w["dashed"], want=_d(want["acc"], lang)))
    else:
        extra.append(_said(w["result"], right=_n(sum(right.values()), lang), test=_n(nums["test"], lang)))
    return _figure(w["title"], rows, said, extra + notes)


_OVERLAP_WORDS = {
    "en": {"title": "Your overlap on the test pairs, per label people gave",
           "mean": "Mean overlap", "full": "Pairs with overlap 1", "doctor": "The doctor and the actor",
           "blue": "Blue: the mean of what your overlap gives over the test pairs with that label, and how many of "
                   "those pairs get exactly 1; a full bar is 1. Last, your overlap for *The doctor was paid by the "
                   "actor.* and *The doctor paid the actor.*",
           "count": "{full} of {n}",
           "odd": "For {n} of the test pairs, your function gives {got} instead of a number from 0 to 1.",
           "stop": "For {n} of the test pairs, your function stops with an error.",
           "dashed": "Purple dashed: a number that differs from the course's; for entailment, the course gets a mean of "
                     "{mean} and {full} pairs with overlap 1.",
           "result": "Of the {full} test pairs with overlap 1, {e} are entailment."},
    "nl": {"title": "Jouw overlap op de testparen, per label dat mensen gaven",
           "mean": "Gemiddelde overlap", "full": "Paren met overlap 1", "doctor": "De dokter en de acteur",
           "blue": "Blauw: het gemiddelde van wat je overlap geeft over de testparen met dat label, en hoeveel van die "
                   "paren precies 1 krijgen; een volle balk is 1. Als laatste je overlap voor *The doctor was paid by "
                   "the actor.* en *The doctor paid the actor.*",
           "count": "{full} van de {n}",
           "odd": "Voor {n} van de testparen geeft je functie {got} in plaats van een getal van 0 tot 1.",
           "stop": "Voor {n} van de testparen stopt je functie met een fout.",
           "dashed": "Paars gestreept: een getal dat afwijkt van het getal van de course; voor entailment krijgt de "
                     "course een gemiddelde van {mean} en {full} paren met overlap 1.",
           "result": "Van de {full} testparen met overlap 1 zijn er {e} entailment."},
}


def _count_row(label: str, text: str, off: bool) -> str:
    line, color = _ROLE["mine"]
    style = _CODE + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(label)}</span>'
            f'<span style="{style}">{html.escape(text)}</span></div>')


def overlap_picture(seen) -> str:
    """The learner's overlap on every test pair: its mean per label, the pairs with overlap 1 per label, and the
    doctor pair."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _OVERLAP_WORDS[lang], numbers()
    doctor = tuple(snli.words(s) for s in DOCTOR)
    calls = [(list(p), list(h)) for p, h, _ in TEST] + [(list(doctor[0]), list(doctor[1]))]
    got = _timed(lambda: _quiet_many(fn, calls))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    got, mine_doctor = got[:-1], got[-1]
    stopped = sum(g is _STOPPED for g in got)
    odd = [g for g in got if g is not _STOPPED and not _is_share(g)]
    notes = []
    if stopped:
        notes.append(_said(w["stop"], n=_n(stopped, lang)))
    if odd:
        notes.append(_said(w["odd"], n=_n(len(odd), lang), got=_short(odd[0])))
    if stopped + len(odd) == len(got):
        return _figure(w["title"], [], [], notes)
    rows, said, differs = [_head(w["mean"])], [], False
    values = [(g if _is_share(g) else 0.0, r[2]) for g, r in zip(got, TEST)]
    for lab in LABELS:
        mean = sum(float(v) for v, x in values if x == lab) / nums["test_labels"][lab]
        off = not _same_number(mean, nums["overlap"][lab]["mean"])
        differs = differs or off
        rows.append(_bar(lab, mean, "mine", lang, off=off))
        said.append(f"{lab}: {_d(mean, lang)}")
    rows.append(_head(w["full"]))
    full = {lab: sum(v == 1 for v, x in values if x == lab) for lab in LABELS}
    for lab in LABELS:
        off = full[lab] != nums["overlap"][lab]["full"]
        differs = differs or off
        text = w["count"].format(full=_n(full[lab], lang), n=_n(nums["test_labels"][lab], lang))
        rows.append(_count_row(lab, text, off))
        said.append(f"{lab}: {text}")
    rows.append(_head(w["doctor"]))
    off = not _same_number(mine_doctor, 1.0)
    differs = differs or off
    rows.append(_bar("overlap", mine_doctor, "mine", lang, off=off))
    said.append(f'{_bare(w["doctor"])}: {_short(mine_doctor, 12)}')
    extra = [_said(w["blue"])]
    if differs:
        e = nums["overlap"]["entailment"]
        extra.append(_said(w["dashed"], mean=_d(e["mean"], lang), full=_n(e["full"], lang)))
    else:
        extra.append(_said(w["result"], full=_n(nums["full"], lang), e=_n(nums["overlap"]["entailment"]["full"], lang)))
    return _figure(w["title"], rows, said, extra + notes)


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["label_shares", "guess", "overlap", "doctor"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name: the course's pairs as they are
    (tuples, which nothing can change), or a new copy of a course's table as dicts."""

    def __init__(self, name: str, value, fresh=None):
        self.name, self.value, self.fresh = name, value, fresh

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.fresh() if self.fresh else self.value


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


def _zero(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _missing_labels(got, text) -> bool:
    """A dict with shares for some of the labels, but not all three."""
    return isinstance(got, dict) and set(got) < set(LABELS)


def _not_label(got, text) -> bool:
    return not _is_label(got)


def _not_share(got, text) -> bool:
    return got is not None and not _is_share(got)


def _rounded(v, depth: int = 0) -> str:
    """A value the way Python shows it, with its floats rounded to four decimals and its words in double quotes, also
    inside a dict, a tuple or a list; at most 30 items and three levels deep."""
    if depth > 3:
        return "…"
    if isinstance(v, float):
        text = f"{round(v, 4) + 0.0:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if type(v) is dict:
        items = list(v.items())
        inner = ", ".join(f"{_rounded(k, depth + 1)}: {_rounded(x, depth + 1)}" for k, x in items[:30])
        return "{" + inner + (", …" if len(items) > 30 else "") + "}"
    if type(v) is tuple:
        inner = ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "")
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if type(v) is list:
        return "[" + ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "") + "]"
    if type(v) is str and v.isprintable() and '"' not in v and "\\" not in v:
        return f'"{v}"'   # double quotes, as in the exercise texts
    return _repr(v)


def show_value(v) -> str:
    """How {got} and {want} show a value: a number as a number, rounded to four decimals; anything else as code, cut
    after 90 characters."""
    text = _rounded(v)
    if len(text) > 90:
        text = text[:89] + "…"
    if isinstance(v, (int, float)) and not isinstance(v, bool) or "`" in text:
        return text
    return f"`{text}`"


_TRAIN = Named("C.TRAIN", TRAIN)
_PRIOR = Named("C.LOG_PRIOR", LOG_PRIOR, lambda: dict(LOG_PRIOR))
_SCORES = Named("C.WORD_SCORES", WORD_SCORES, lambda: {lab: dict(WORD_SCORES[lab]) for lab in LABELS})

# exercise 1: label_shares. The visible case: four pairs, three hypotheses with "nobody"; then a word that no sentence
# has, a word that only premises have, a word twice in one hypothesis, and the training pairs with "nobody" and with
# "outdoors".
_PAIRS = [(["a", "dog", "runs"], ["an", "animal", "runs"], "entailment"),
          (["a", "dog", "runs"], ["nobody", "runs"], "contradiction"),
          (["two", "kids", "play"], ["the", "kids", "know", "nobody"], "neutral"),
          (["a", "man", "sings"], ["nobody", "sings"], "contradiction")]
_TWICE = [(["two", "people", "talk"], ["people", "see", "people"], "entailment"),
          (["a", "man", "rests"], ["people", "rest"], "neutral"),
          (["a", "man", "rests"], ["a", "man", "runs"], "contradiction")]


def _literal(v) -> str:
    """Pairs as Python code with double quotes, the way the exercise text writes them."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_literal(x) for x in v) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    return repr(v)


def _pairs(pairs) -> Named:
    """Hand-made pairs for a check: shown as code with double quotes, and new lists for each call."""
    return Named(_literal(pairs), pairs, lambda: [(list(p), list(h), lab) for p, h, lab in pairs])


LS_CASES = [
    (_pairs(_PAIRS), "nobody"),
    (_pairs(_PAIRS), "cat"),
    (_pairs(_PAIRS), "dog"),
    (_pairs(_TWICE), "people"),
    (_TRAIN, "nobody"),
    (_TRAIN, "outdoors"),
]
_LS_HEAD = "def label_shares(pairs, word):\n"
_LS_START = (_LS_HEAD
             + '    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}\n'
             + "    ...\n")
_LS = (_LS_HEAD
       + '    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}\n'
       + "    total = 0\n"
       + "    for premise, hypothesis, label in pairs:\n"
       + "        if word in hypothesis:\n"
       + "            counts[label] = counts[label] + 1\n"
       + "            total = total + 1\n"
       + "    shares = {}\n"
       + "    for label in counts:\n"
       + "        if total > 0:\n"
       + "            shares[label] = counts[label] / total\n"
       + "        else:\n"
       + "            shares[label] = 0.0\n"
       + "    return shares\n")
_LS_PER_LABEL = (_LS_HEAD
                 + '    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}\n'
                 + '    pairs_of = {"entailment": 0, "neutral": 0, "contradiction": 0}\n'
                 + "    for premise, hypothesis, label in pairs:\n"
                 + "        pairs_of[label] = pairs_of[label] + 1\n"
                 + "        if word in hypothesis:\n"
                 + "            counts[label] = counts[label] + 1\n"
                 + "    shares = {}\n"
                 + "    for label in counts:\n"
                 + "        if pairs_of[label] > 0:\n"
                 + "            shares[label] = counts[label] / pairs_of[label]\n"
                 + "        else:\n"
                 + "            shares[label] = 0.0\n"
                 + "    return shares\n")
_LS_ALL_PAIRS = _LS.replace("            shares[label] = counts[label] / total\n",
                            "            shares[label] = counts[label] / len(pairs)\n")
_LS_PREMISE = _LS.replace("        if word in hypothesis:\n", "        if word in premise:\n")
_LS_EACH = _LS.replace("        if word in hypothesis:\n            counts[label] = counts[label] + 1\n"
                       "            total = total + 1\n",
                       "        for w in hypothesis:\n            if w == word:\n"
                       "                counts[label] = counts[label] + 1\n                total = total + 1\n")
_LS_NO_ZERO = _LS.replace("        if total > 0:\n            shares[label] = counts[label] / total\n"
                          "        else:\n            shares[label] = 0.0\n",
                          "        shares[label] = counts[label] / total\n")
_LS_COUNTS = _LS.replace("    return shares\n", "    return counts\n")
_LS_SEEN = _LS.replace("        if total > 0:\n", "        if counts[label] > 0:\n").replace(
    "        else:\n            shares[label] = 0.0\n", "")

# exercise 2: guess. The visible case: three words, one of which the dicts do not have; then a hypothesis without
# words and one with only words the dicts do not have, both with a prior in which neutral is highest; a word twice; two
# known words; one known word with a prior in which neutral is a whole point higher; one known word three times; and
# the three hypotheses of the example of SNLI with the course's tables.
_PRIOR_S = {"entailment": -1.0, "neutral": -1.2, "contradiction": -1.1}
_PRIOR_N = {"entailment": -1.2, "neutral": -1.0, "contradiction": -1.1}
_PRIOR_FAR = {"entailment": -2.0, "neutral": -1.0, "contradiction": -2.0}
_PRIOR_NEAR = {"entailment": -1.0, "neutral": -0.8, "contradiction": -1.5}
_SCORES_S = {"entailment": {"nobody": -9.0, "outdoors": -4.0, "dog": -6.0},
             "neutral": {"nobody": -9.5, "outdoors": -6.0, "dog": -6.1},
             "contradiction": {"nobody": -5.0, "outdoors": -6.5, "dog": -6.2}}
GS_CASES = [
    (["nobody", "is", "outdoors"], _PRIOR_S, _SCORES_S),
    ([], _PRIOR_N, _SCORES_S),
    (["a", "cat"], _PRIOR_N, _SCORES_S),
    (["outdoors", "outdoors", "nobody"], _PRIOR_S, _SCORES_S),
    (["dog", "outdoors"], _PRIOR_S, _SCORES_S),
    (["dog"], _PRIOR_FAR, _SCORES_S),
    (["dog", "dog", "dog"], _PRIOR_NEAR, _SCORES_S),
] + [(snli.words(h), _PRIOR, _SCORES) for _, h in GUIDE]
_GS_HEAD = "def guess(hypothesis, log_prior, word_scores):\n"
_GS_START = (_GS_HEAD
             + "    best = None\n"
             + "    for label in log_prior:\n"
             + "        score = ...\n"
             + "        if best is None:\n"
             + "            best = label\n"
             + "            best_score = score\n"
             + "        if score > best_score:\n"
             + "            best = label\n"
             + "            best_score = score\n"
             + "    return best\n")
_GS = (_GS_HEAD
       + "    best = None\n"
       + "    for label in log_prior:\n"
       + "        score = log_prior[label]\n"
       + "        for word in hypothesis:\n"
       + "            if word in word_scores[label]:\n"
       + "                score = score + word_scores[label][word]\n"
       + "        if best is None:\n"
       + "            best = label\n"
       + "            best_score = score\n"
       + "        if score > best_score:\n"
       + "            best = label\n"
       + "            best_score = score\n"
       + "    return best\n")
_GS_NO_PRIOR = _GS.replace("        score = log_prior[label]\n", "        score = 0\n")
_GS_LOWEST = _GS.replace("score > best_score", "score < best_score")
_GS_SCORE = _GS.replace("    return best\n", "    return best_score\n")
_GS_ALL = (_GS_HEAD
           + "    scores = {}\n"
           + "    for label in log_prior:\n"
           + "        score = log_prior[label]\n"
           + "        for word in hypothesis:\n"
           + "            if word in word_scores[label]:\n"
           + "                score = score + word_scores[label][word]\n"
           + "        scores[label] = score\n"
           + "    return scores\n")
_GS_UNKNOWN = _GS.replace("            if word in word_scores[label]:\n                score = score + word_scores[label][word]\n",
                          "            score = score + word_scores[label][word]\n")
_GS_LAST = _GS.replace("                score = score + word_scores[label][word]\n",
                       "                score = log_prior[label] + word_scores[label][word]\n")
_GS_ONCE = _GS.replace("        for word in hypothesis:\n", "        for word in set(hypothesis):\n")
_GS_PRIOR_EACH = _GS.replace("                score = score + word_scores[label][word]\n",
                             "                score = score + log_prior[label] + word_scores[label][word]\n")
_GS_PRIOR_LAST = (_GS_HEAD
                  + "    best = None\n"
                  + "    for label in log_prior:\n"
                  + "        score = 0\n"
                  + "        for word in hypothesis:\n"
                  + "            if word in word_scores[label]:\n"
                  + "                score = score + word_scores[label][word]\n"
                  + "        if score == 0:\n"
                  + "            score = log_prior[label]\n"
                  + "        if best is None:\n"
                  + "            best = label\n"
                  + "            best_score = score\n"
                  + "        if score > best_score:\n"
                  + "            best = label\n"
                  + "            best_score = score\n"
                  + "    return best\n")

# exercise 3: overlap. The visible case: the example of the instructions of SNLI, one word of four in the premise;
# then every word inside, a hypothesis without words, a word twice in the hypothesis, a word twice in the premise, the
# pair of the doctor and the actor, and no word inside.
_DOCTOR_WORDS = tuple(snli.words(s) for s in DOCTOR)
OV_CASES = [
    (snli.words(GUIDE_PREMISE), snli.words(GUIDE[0][1])),
    (["a", "dog", "runs", "in", "a", "field"], ["a", "dog", "runs"]),
    (["a", "dog", "runs"], []),
    (["the", "man", "sleeps"], ["the", "man", "and", "the", "dog"]),
    (["a", "dog", "and", "a", "cat"], ["a", "cat"]),
    (list(_DOCTOR_WORDS[0]), list(_DOCTOR_WORDS[1])),
    (["a", "dog", "runs"], ["nobody", "sleeps"]),
]
_OV_HEAD = "def overlap(premise, hypothesis):\n"
_OV_START = (_OV_HEAD
             + "    if len(hypothesis) == 0:\n"
             + "        return 0.0\n"
             + "    ...\n")
_OV = (_OV_HEAD
       + "    if len(hypothesis) == 0:\n"
       + "        return 0.0\n"
       + "    in_premise = set(premise)\n"
       + "    inside = 0\n"
       + "    for word in hypothesis:\n"
       + "        if word in in_premise:\n"
       + "            inside = inside + 1\n"
       + "    return inside / len(hypothesis)\n")
_OV_PREMISE_LENGTH = _OV.replace("    return inside / len(hypothesis)\n", "    return inside / len(premise)\n")
_OV_DISTINCT = (_OV_HEAD
                + "    if len(hypothesis) == 0:\n"
                + "        return 0.0\n"
                + "    in_premise = set(premise)\n"
                + "    words = set(hypothesis)\n"
                + "    inside = 0\n"
                + "    for word in words:\n"
                + "        if word in in_premise:\n"
                + "            inside = inside + 1\n"
                + "    return inside / len(words)\n")
_OV_REVERSED = (_OV_HEAD
                + "    if len(hypothesis) == 0:\n"
                + "        return 0.0\n"
                + "    in_hypothesis = set(hypothesis)\n"
                + "    inside = 0\n"
                + "    for word in premise:\n"
                + "        if word in in_hypothesis:\n"
                + "            inside = inside + 1\n"
                + "    return inside / len(premise)\n")
_OV_COUNT = _OV.replace("    return inside / len(hypothesis)\n", "    return inside\n")
_OV_NO_ZERO = _OV.replace("    if len(hypothesis) == 0:\n        return 0.0\n", "")
_OV_PREMISE_COUNT = (_OV_HEAD
                     + "    if len(hypothesis) == 0:\n"
                     + "        return 0.0\n"
                     + "    inside = 0\n"
                     + "    for word in hypothesis:\n"
                     + "        inside = inside + premise.count(word)\n"
                     + "    return inside / len(hypothesis)\n")

EXERCISES = {
    "label_shares": FunctionBox(
        name="label_shares",
        picture=shares_picture,
        function="label_shares",
        start=_LS_START,
        solution=_LS,
        visible=LS_CASES[:1],
        hidden=LS_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("zero", error=_zero,
                    en="With `{call}`, your function stops with a ZeroDivisionError: no hypothesis has the word, so the number it divides by is 0. What share should each label get then?",
                    nl="Met `{call}` stopt je functie met een ZeroDivisionError: geen enkele hypothese heeft het woord, dus het getal waardoor je functie deelt is 0. Welk deel moet elk label dan krijgen?"),
            Mistake("per-label", wrong=_LS_PER_LABEL,
                    en="`{call}` gives {got}, and it should give {want}: your function divides the count of a label by the number of pairs with that label. Which hypotheses should the shares be of?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt het aantal van een label door het aantal paren met dat label. Van welke hypothesen moeten de delen zijn?"),
            Mistake("all-pairs", wrong=_LS_ALL_PAIRS,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of all pairs, also the pairs whose hypothesis does not have the word. Which hypotheses should the shares be of?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal van alle paren, ook de paren waarvan de hypothese het woord niet heeft. Van welke hypothesen moeten de delen zijn?"),
            Mistake("premise", wrong=_LS_PREMISE,
                    en="`{call}` gives {got}, and it should give {want}: your function looks for the word in the premise. Which sentence of a pair should have it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zoekt het woord in de premisse. Welke zin van een paar moet het woord hebben?"),
            Mistake("each-time", wrong=_LS_EACH,
                    en="`{call}` gives {got}, and it should give {want}: a hypothesis with the word twice counts twice. How often should one hypothesis count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een hypothese met het woord erin telt twee keer. Hoe vaak moet één hypothese tellen?"),
            Mistake("counts", wrong=_LS_COUNTS,
                    en="`{call}` gives {got}, and it should give {want}: the number of hypotheses with the word for each label. The exercise asks for shares: of what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal hypothesen met het woord voor elk label. De opdracht vraagt om delen: waarvan?"),
            Mistake("missing-labels", matches=_missing_labels,
                    en="`{call}` gives {got}, and it should give {want}: a label is missing. What share does a label get when no hypothesis with the word has it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er ontbreekt een label. Welk deel krijgt een label als geen enkele hypothese met het woord dat label heeft?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count, for each label, the hypotheses that have the word, and all of them together, and give back a dict with the share of each label.",
                      "nl": "Vervang de `...`: tel voor elk label de hypothesen die het woord hebben, en alle hypothesen met het woord samen, en geef een dict terug met het deel van elk label."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which pairs does your function count, and what does it divide by?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke paren telt je functie, en waardoor deelt je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for a word that no hypothesis has, for a word that only premises have, and for a hypothesis with the word twice?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor een woord dat geen enkele hypothese heeft, voor een woord dat alleen in premissen staat, en voor een hypothese met het woord twee keer erin?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. What should each share be when no hypothesis has the word, and does every way through your function end with a `return` of a dict?",
                          "nl": "Met `{call}` geeft je functie `None`. Wat moet elk deel zijn als geen enkele hypothese het woord heeft, en eindigt elke weg door je functie met een `return` van een dict?"},
            "correct": {"en": "Your `label_shares` is right in all {n} cases. Look at the picture: your function on seven words of the training pairs.",
                        "nl": "Je `label_shares` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op zeven woorden uit de trainingsparen."},
        },
    ),
    "guess": FunctionBox(
        name="guess",
        picture=guess_picture,
        function="guess",
        start=_GS_START,
        solution=_GS,
        visible=GS_CASES[:1],
        hidden=GS_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("unknown", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: it looks up a word that the dict of a label does not have. What does naive Bayes do with a word it never saw in training?",
                    nl="Met `{call}` stopt je functie met een KeyError: je functie zoekt een woord op dat de dict van een label niet heeft. Wat doet naive Bayes met een woord dat het nooit zag in de training?"),
            Mistake("no-prior", wrong=_GS_NO_PRIOR,
                    en="`{call}` gives {got}, and it should give {want}: every label starts at 0, so for a hypothesis without known words the first label wins. Where should the score of a label start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk label begint bij 0, dus bij een hypothese zonder bekende woorden wint het eerste label. Waar moet de score van een label beginnen?"),
            Mistake("lowest", wrong=_GS_LOWEST,
                    en="`{call}` gives {got}, and it should give {want}: the label with the lowest score. Which label is the most likely one?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het label met de laagste score. Welk label is het meest waarschijnlijk?"),
            Mistake("last-word", wrong=_GS_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the score of a label holds only the last known word, since each word starts the score over. What should happen to the score with each word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de score van een label bevat alleen het laatste bekende woord, omdat elk woord de score opnieuw laat beginnen. Wat moet er met elk woord met de score gebeuren?"),
            Mistake("once", wrong=_GS_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: a word that comes twice in the hypothesis counts once. How often does naive Bayes count such a word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat twee keer in de hypothese staat telt één keer. Hoe vaak telt naive Bayes zo'n woord?"),
            Mistake("prior-each", wrong=_GS_PRIOR_EACH,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the log prior of a label again with every known word. How often does the prior of a label count in its score?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de log prior van een label opnieuw op bij elk bekend woord. Hoe vaak telt de prior van een label mee in de score?"),
            Mistake("prior-fallback", wrong=_GS_PRIOR_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function uses the log prior only when the hypothesis has no known word. When does the prior of a label count in its score?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie gebruikt de log prior alleen als de hypothese geen bekend woord heeft. Wanneer telt de prior van een label mee in de score?"),
            Mistake("score", wrong=_GS_SCORE,
                    en="`{call}` gives {got}, and it should give {want}: the highest score itself. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de hoogste score zelf. Wat moet je functie teruggeven?"),
            Mistake("all-scores", wrong=_GS_ALL,
                    en="`{call}` gives {got}, and it should give {want}: the score of every label. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de score van elk label. Wat moet je functie teruggeven?"),
            Mistake("not-label", matches=_not_label,
                    en="`{call}` gives {got}, and it should give {want}: one of the three labels. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een van de drie labels. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: let the score of a label start at its log prior, and add the score of each word of the hypothesis that the dict of that label has. The lines under it keep the label with the highest score.",
                      "nl": "Vervang de `...`: laat de score van een label beginnen bij de log prior, en tel de score op van elk woord van de hypothese dat in de dict van dat label staat. De regels eronder onthouden het label met de hoogste score."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where does the score of a label start, which words add to it, and which label does your function keep?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Waar begint de score van een label, welke woorden tellen mee, en welk label onthoudt je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for a hypothesis without words, for words the dicts do not have, and for a word that comes more than once? And does the log prior of a label count once in its score?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor een hypothese zonder woorden, voor woorden die niet in de dicts staan, en voor een woord dat vaker dan één keer voorkomt? En telt de log prior van een label één keer mee in de score?"},
            "correct": {"en": "Your `guess` is right in all {n} cases. Look at the picture: your function on all the test pairs, from the hypothesis alone.",
                        "nl": "Je `guess` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op alle testparen, op de hypothese alleen."},
        },
    ),
    "overlap": FunctionBox(
        name="overlap",
        picture=overlap_picture,
        function="overlap",
        start=_OV_START,
        solution=_OV,
        visible=OV_CASES[:1],
        hidden=OV_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("zero", error=_zero,
                    en="With `{call}`, your function stops with a ZeroDivisionError: the hypothesis has no words, so the number it divides by is 0. What should your function give for a hypothesis without words?",
                    nl="Met `{call}` stopt je functie met een ZeroDivisionError: de hypothese heeft geen woorden, dus het getal waardoor je functie deelt is 0. Wat moet je functie geven voor een hypothese zonder woorden?"),
            Mistake("premise-length", wrong=_OV_PREMISE_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the words of the hypothesis in the premise, and divides by the length of the premise. By the length of which sentence should it divide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de woorden van de hypothese in de premisse, en deelt door de lengte van de premisse. Door de lengte van welke zin moet je functie delen?"),
            Mistake("reversed", wrong=_OV_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: the share of the words of the premise that the hypothesis has. Which way round should it be?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de woorden van de premisse dat de hypothese heeft. Welke kant op moet het?"),
            Mistake("distinct", wrong=_OV_DISTINCT,
                    en="`{call}` gives {got}, and it should give {want}: a word that comes twice in the hypothesis counts once. How often should each word of the hypothesis count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat twee keer in de hypothese staat telt één keer. Hoe vaak moet elk woord van de hypothese tellen?"),
            Mistake("premise-count", wrong=_OV_PREMISE_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: a word of the hypothesis counts as often as the premise has it. How often should one word of the hypothesis count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord van de hypothese telt zo vaak als het in de premisse staat. Hoe vaak moet één woord van de hypothese tellen?"),
            Mistake("count", wrong=_OV_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of words of the hypothesis in the premise. Overlap is a share: of what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal woorden van de hypothese in de premisse. Overlap is een deel: waarvan?"),
            Mistake("not-share", matches=_not_share,
                    en="`{call}` gives {got}, and it should give {want}: a number from 0 to 1. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een getal van 0 tot 1. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the words of the hypothesis that the premise has, and give back their share of all words of the hypothesis.",
                      "nl": "Vervang de `...`: tel de woorden van de hypothese die de premisse heeft, en geef het deel terug van alle woorden van de hypothese."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words does your function count, and what does it divide by?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke woorden telt je functie, en waardoor deelt je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work when every word is inside, when none is, for a hypothesis without words, and for a word that comes twice?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie als elk woord erin staat, als geen enkel woord erin staat, voor een hypothese zonder woorden, en voor een woord dat twee keer voorkomt?"},
            "correct": {"en": "Your `overlap` is right in all {n} cases. Look at the picture: your function on all the test pairs, per label people gave.",
                        "nl": "Je `overlap` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op alle testparen, per label dat mensen gaven."},
        },
    ),
}

CASES = [
    ("label_shares", _LS_START, "empty", None),
    ("label_shares", _LS, "correct", None),
    ("label_shares", "def label_shares(pairs, word):\n    with_word = [label for premise, hypothesis, label in pairs if word in hypothesis]\n    shares = {}\n    for label in [\"entailment\", \"neutral\", \"contradiction\"]:\n        shares[label] = with_word.count(label) / len(with_word) if with_word else 0.0\n    return shares\n", "correct", None),
    ("label_shares", "def label_shares(pairs, word):\n    counts = {\"entailment\": 0, \"neutral\": 0, \"contradiction\": 0}\n    for pair in pairs:\n        if word in pair[1]:\n            counts[pair[2]] += 1\n    total = sum(counts.values())\n    return {label: (counts[label] / total if total else 0) for label in counts}\n", "correct", None),
    ("label_shares", _LS_NO_ZERO, "error", "zero"),
    ("label_shares", _LS_PER_LABEL, "wrong", "per-label"),
    ("label_shares", _LS_ALL_PAIRS, "wrong", "all-pairs"),
    ("label_shares", _LS_PREMISE, "wrong", "premise"),
    ("label_shares", _LS_EACH, "wrong", "each-time"),
    ("label_shares", _LS_COUNTS, "wrong", "counts"),
    ("label_shares", _LS_SEEN, "wrong", "missing-labels"),
    ("label_shares", _LS.replace("    return shares\n", "    print(shares)\n"), "wrong", "prints"),
    ("label_shares", _LS.replace("        else:\n            shares[label] = 0.0\n", "").replace(
        "    shares = {}\n", "    if total == 0:\n        return None\n    shares = {}\n"), "wrong", "unreached"),
    ("label_shares", "def label_shares(pairs, word):\n    return {\"entailment\": 0.0, \"neutral\": 1 / 3, \"contradiction\": 2 / 3}\n", "almost", None),
    ("label_shares", _LS.replace("    total = 0\n", "    total = 0\n    ...\n"), "wrong", "dots-left"),
    ("guess", _GS_START, "empty", None),
    ("guess", _GS, "correct", None),
    ("guess", "def guess(hypothesis, log_prior, word_scores):\n    scores = {}\n    for label in log_prior:\n        scores[label] = log_prior[label] + sum(word_scores[label].get(w, 0) for w in hypothesis)\n    return max(scores, key=scores.get)\n", "correct", None),
    ("guess", "def guess(hypothesis, log_prior, word_scores):\n    best = None\n    for label in [\"entailment\", \"neutral\", \"contradiction\"]:\n        score = log_prior[label]\n        for word in hypothesis:\n            if word in word_scores[label]:\n                score += word_scores[label][word]\n        if best is None or score > best[1]:\n            best = (label, score)\n    return best[0]\n", "correct", None),
    ("guess", _GS_UNKNOWN, "error", "unknown"),
    ("guess", _GS_NO_PRIOR, "wrong", "no-prior"),
    ("guess", _GS_LOWEST, "wrong", "lowest"),
    ("guess", _GS_LAST, "wrong", "last-word"),
    ("guess", _GS_ONCE, "wrong", "once"),
    ("guess", _GS_PRIOR_EACH, "wrong", "prior-each"),
    ("guess", _GS_PRIOR_LAST, "wrong", "prior-fallback"),
    ("guess", _GS_SCORE, "wrong", "score"),
    ("guess", _GS_ALL, "wrong", "all-scores"),
    ("guess", "def guess(hypothesis, log_prior, word_scores):\n    return 2\n", "wrong", "not-label"),
    ("guess", "def guess(hypothesis, log_prior, word_scores):\n    return \"contradiction\"\n", "almost", None),
    ("guess", _GS.replace("    return best\n", "    print(best)\n"), "wrong", "prints"),
    ("guess", _GS.replace("        score = log_prior[label]\n", "        score = log_prior[label]\n        ...\n"), "wrong", "dots-left"),
    ("overlap", _OV_START, "empty", None),
    ("overlap", _OV, "correct", None),
    ("overlap", "def overlap(premise, hypothesis):\n    if not hypothesis:\n        return 0.0\n    return len([w for w in hypothesis if w in premise]) / len(hypothesis)\n", "correct", None),
    ("overlap", _OV_NO_ZERO, "error", "zero"),
    ("overlap", _OV_PREMISE_LENGTH, "wrong", "premise-length"),
    ("overlap", _OV_REVERSED, "wrong", "reversed"),
    ("overlap", _OV_DISTINCT, "wrong", "distinct"),
    ("overlap", _OV_PREMISE_COUNT, "wrong", "premise-count"),
    ("overlap", _OV_COUNT, "wrong", "count"),
    ("overlap", "def overlap(premise, hypothesis):\n    return str(len(hypothesis))\n", "wrong", "not-share"),
    ("overlap", "def overlap(premise, hypothesis):\n    return 0.25\n", "almost", None),
    ("overlap", _OV.replace("    return inside / len(hypothesis)\n", "    print(inside / len(hypothesis))\n"), "wrong", "prints"),
    ("overlap", _OV.replace("    inside = 0\n", "    inside = 0\n    ...\n"), "wrong", "dots-left"),
]
