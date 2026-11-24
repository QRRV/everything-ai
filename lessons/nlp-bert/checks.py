# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-bert: a training example of masked language modeling from a plan of what goes in at each
chosen place (make_example), the rank of the right piece among the 30,522 scores at a place (rank), and the loss at
the chosen places only (mlm_loss); with a picture of the idea, the pictures of what each function gives on BERT-Tiny's
own scores, and the explore.

What BERT-Tiny gives at the blank of eight sentences of the course and at every place of one more, and the ranks the
course measured on the 500 English sentences of Tatoeba in data/tatoeba-langs.tsv.gz, the course ships
(data/bert-tiny-fill.npz, made by data/tools/make_bert_fill.py), so that the exercises and their pictures work
without the model. BERT-Tiny itself (18 MB, everything_ai.bert) the learner downloads once for the worked example and
the explore. Every number in the text and the pictures is computed by the course (tests/test_nlp_bert.py computes them
again)."""
import ast
import contextlib
import copy
import html
import io
import math
import re

import numpy as np
import torch
from torch import nn

from everything_ai import lesson as lesson_helpers, paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, same_value, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-bert"
REPO = "google/bert_uncased_L-2_H-128_A-2"   # everything_ai.bert.REPO: that module loads the model, so not here
BERT_FILES = [(REPO, "model.safetensors"), (REPO, "vocab.txt")]
FILL = paths.ROOT / "data" / "bert-tiny-fill.npz"
VOCAB = 30_522   # pieces in BERT-Tiny's vocabulary
MASK = "[MASK]"

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The worked example and the explore download BERT-Tiny, {download_size(BERT_FILES, 'en')}, once from Hugging Face, when you click its button in the lesson; the exercises and the rest of the lesson work without it.",
    "nl": f"Werkt op elke computer. Het uitgewerkte voorbeeld en het onderdeel om zelf te proberen downloaden één keer BERT-Tiny, {download_size(BERT_FILES, 'nl')}, van Hugging Face, als je in de les op de knop ervoor klikt; de oefeningen en de rest van de les werken zonder.",
}

# The course's sentence with two blanks, its pieces as BERT-Tiny cuts them ([CLS] and [SEP] included), and the words
# the course hid at places 5 and 11. tests/test_nlp_bert.py checks the pieces with BERT-Tiny itself when EAI_BERT_DIR
# names a folder with its two files.
WEEK = "A week has seven [MASK] and a year has twelve [MASK]."
WEEK_PIECES = ("[CLS]", "a", "week", "has", "seven", MASK, "and", "a", "year", "has", "twelve", MASK, ".", "[SEP]")
WEEK_HIDDEN = {5: "days", 11: "months"}

# The pieces of the course's sentence for the picture under exercise 1, without [CLS] and [SEP], and a plan of three
# of its 18 places, about 15%: one [MASK], one random piece, and one piece that stays as it is.
EXAMPLE_SENTENCE = "The children played in the park after school, and then they walked home together for dinner."
_EXAMPLE_PIECES = ("the", "children", "played", "in", "the", "park", "after", "school", ",", "and", "then", "they",
                   "walked", "home", "together", "for", "dinner", ".")
_EXAMPLE_PLAN = ((5, MASK), (12, "violin"), (16, "dinner"))
EXAMPLE_PIECES = list(_EXAMPLE_PIECES)
EXAMPLE_PLAN = dict(_EXAMPLE_PLAN)


# --- the shipped data ----------------------------------------------------------------------------------------------------

def _data() -> dict:
    def make():
        with np.load(FILL, allow_pickle=False) as z:
            out = {name: z[name] for name in z.files}
        for array in out.values():
            array.flags.writeable = False
        return out
    return kept(f"{_KEY}:data", make)


def blanks() -> list:
    """The course's eight sentences with one [MASK] each, as a new list."""
    return [str(s) for s in _data()["blanks"]]


def answers() -> list:
    """The word the course hid in each of the eight sentences, as a new list."""
    return [str(s) for s in _data()["answers"]]


def blank_scores(i: int) -> np.ndarray:
    """BERT-Tiny's 30,522 scores at the [MASK] of sentence i of blanks(), as a new array: one score per piece of its
    vocabulary, row j for the piece with number j."""
    return _data()["blank_scores"][i].astype(np.float32)


def _vocabulary() -> tuple:
    def make():
        d = _data()
        names = {int(i): str(n) for i, n in zip(d["piece_ids"], d["piece_names"])}
        return names, {n: i for i, n in names.items()}
    return kept(f"{_KEY}:vocabulary", make)


def piece_id(piece: str) -> int:
    """The number of a piece in BERT-Tiny's vocabulary, for the pieces this lesson shows."""
    return _vocabulary()[1][piece]


def piece_name(i: int) -> str:
    """The piece with number i of BERT-Tiny's vocabulary, for the pieces this lesson shows."""
    return _vocabulary()[0].get(int(i), f"#{int(i)}")


def week_scores() -> torch.Tensor:
    """BERT-Tiny's 30,522 scores at every place of WEEK_PIECES, as a new tensor of 14 rows: row k for place k."""
    return torch.tensor(_data()["week_scores"].astype(np.float32))


def week_right() -> list:
    """The right piece at every place of WEEK_PIECES: the piece itself, and at the two [MASK]s the hidden word."""
    return [piece_id(WEEK_HIDDEN.get(k, p)) for k, p in enumerate(WEEK_PIECES)]


def week_losses() -> list:
    """The loss at every place of WEEK_PIECES: minus the log of the probability BERT-Tiny gives the right piece."""
    def make():
        with torch.no_grad():
            logp = torch.log_softmax(week_scores(), dim=-1)
        return tuple(float(-logp[k, j]) for k, j in enumerate(week_right()))
    return list(kept(f"{_KEY}:week-losses", make))


def top(scores, k: int = 3) -> list:
    """The k pieces with the highest scores, highest first: (number, probability)."""
    s = np.asarray(scores, dtype=np.float64)
    p = np.exp(s - s.max())
    p = p / p.sum()
    best = np.argsort(-s, kind="stable")[:k]
    return [(int(j), float(p[j])) for j in best]


def numbers() -> dict:
    """The numbers of the text: what the course measured on Tatoeba, the ranks of the eight blanks, and the losses of
    the sentence with two blanks."""
    def make():
        r = np.asarray(_data()["ranks"])
        within = {side: {k: int((r[:, c] <= k).sum()) for k in (1, 5, 10, 100)}
                  for c, side in enumerate(("both", "before"))}
        losses = week_losses()
        return {
            "places": int(len(r)),
            "within": within,
            "median": {"both": float(np.median(r[:, 0])), "before": float(np.median(r[:, 1]))},
            "blank_ranks": tuple(rank(blank_scores(i), piece_id(a)) for i, a in enumerate(answers())),
            "hidden_loss": float(np.mean([losses[k] for k in WEEK_HIDDEN])),
            "all_loss": float(np.mean(losses[1:-1])),
            "seen_max": max(losses[k] for k in range(1, len(losses) - 1) if k not in WEEK_HIDDEN),
        }
    return kept(f"{_KEY}:numbers", make)


# --- BERT-Tiny itself ----------------------------------------------------------------------------------------------------

def ready() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in BERT_FILES)


def bert_or_stop(button, texts: dict):
    """everything_ai.bert for the lesson's cells, once BERT-Tiny is downloaded. Until then the cell stops with
    `texts["ask"]` and the button; a click downloads it, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, BERT_FILES)
    from everything_ai import bert
    return bert


# --- the course's own versions of the exercises ------------------------------------------------------------------------

def make_example(pieces, replace):
    inputs = []
    labels = []
    for k in range(len(pieces)):
        if k in replace:
            inputs.append(replace[k])
            labels.append(pieces[k])
        else:
            inputs.append(pieces[k])
            labels.append(None)
    return inputs, labels


def rank(scores, right):
    better = np.sum(scores > scores[right])
    return int(better) + 1


def mlm_loss(scores, places, right):
    return nn.CrossEntropyLoss()(scores[places], torch.tensor(right))


# --- pictures: helpers ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_ROW = "display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);width:11rem;flex:none"
_TRACK = "flex:1;min-width:3rem;height:0.65rem;background:var(--ea-line);border-radius:2px"
_NUM = "font-family:var(--ea-font-code);font-size:0.8rem;width:3.6rem;flex:none;text-align:right"
_CELL = "font-family:var(--ea-font-ui);font-size:0.85rem;padding:0.2rem 0.6rem 0.2rem 0;vertical-align:top"
_PART = ("display:inline-flex;flex-direction:column;align-items:center;gap:0.15rem;margin:0 0.35rem 0.5rem 0;"
         "vertical-align:top")
_CHIP = ("display:block;border-radius:4px;padding:0 0.3rem;font-family:var(--ea-font-code);font-size:0.78rem;"
         "line-height:1.5;white-space:nowrap")
_ROLE = {   # the line or bar and the color of its text: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}


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
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value stays as
    it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 3,031 in English, 3.031 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _d(x: float, lang: str, digits: int = 3) -> str:
    """A number with three decimals: 0.303 in English, 0,303 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _short(v, limit: int = 60) -> str:
    text = _rounded(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _number(got):
    """The learner's outcome as one finite float, or None: a number, a tensor or an array with one value."""
    if isinstance(got, (bool, np.bool_, str, bytes)):
        return None
    try:
        if isinstance(got, torch.Tensor):
            if got.numel() != 1:
                return None
            got = got.detach().reshape(()).item()
        value = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None
    if value.shape != () or not np.isfinite(value):
        return None
    return float(value)


def _bar(label: str, value, role: str, lang: str, off: bool = False) -> str:
    """A row with a label, a bar from 0 to 1 and the number."""
    line, color = _ROLE[role]
    width = max(0.0, min(1.0, float(value))) * 100
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(label)}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(_d(float(value), lang))}</span></div>')


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + "".join(f'<div style="{_UI}">{h}</div>' for h, _ in notes)
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


def _chip(text: str, role: str, off: bool = False, faint: bool = False) -> str:
    """A piece as a small box: gray for the data, blue for the learner's, purple dashed where it differs."""
    if role == "data":
        style = _CHIP + ";border:1px solid var(--ea-ink-3);color:var(--ea-ink)"
    else:
        style = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
    if faint:
        style += ";opacity:0.55"
    if off:
        style += ";" + _OFF
    return f'<span style="{style}">{html.escape(text)}</span>'


# --- the picture of the idea: what the words after the blank are worth ------------------------------------------------------

_SIDES_WORDS = {
    "en": {"title": "How often the hidden word is among BERT-Tiny's first guesses",
           "within": {1: "The first guess", 5: "Among the first 5", 10: "Among the first 10", 100: "Among the first 100"},
           "both": "Both sides", "before": "Only the words before",
           "legend": "Each bar: the share of the {places} hidden words of 500 English sentences of Tatoeba that has "
                     "that place or better in BERT-Tiny's list of guesses, ranked from the highest score down; a full "
                     "bar is 1. In the color of the text: BERT-Tiny with the words on both sides of the [MASK]. Gray: "
                     "the same, with every piece after the [MASK] made a [MASK] too, so that only the words before it "
                     "are left."},
    "nl": {"title": "Hoe vaak het verborgen woord bij de eerste gokken van BERT-Tiny zit",
           "within": {1: "De eerste gok", 5: "Bij de eerste 5", 10: "Bij de eerste 10", 100: "Bij de eerste 100"},
           "both": "Beide kanten", "before": "Alleen de woorden ervoor",
           "legend": "Elke balk: het deel van de {places} verborgen woorden uit 500 Engelse zinnen van Tatoeba dat "
                     "binnen zoveel gokken van BERT-Tiny valt, met de gokken gerangschikt van de hoogste score "
                     "omlaag; een volle balk is 1. In de kleur van de tekst: BERT-Tiny met de woorden aan beide kanten "
                     "van de [MASK]. Grijs: hetzelfde, maar met elk stukje na de [MASK] ook als [MASK], zodat alleen de "
                     "woorden ervoor overblijven."},
}


def sides_picture(lang: str) -> str:
    """The share of the hidden words of Tatoeba among BERT-Tiny's first 1, 5, 10 and 100 guesses, with both sides and
    with only the words before the blank."""
    w, nums = _SIDES_WORDS[lang], numbers()
    rows, said = [], []
    for k, label in w["within"].items():
        both = nums["within"]["both"][k] / nums["places"]
        before = nums["within"]["before"][k] / nums["places"]
        rows += [_head(label), _bar(w["both"], both, "course", lang), _bar(w["before"], before, "data", lang)]
        said.append(f'{label}: {_d(both, lang)}, {_d(before, lang)}')
    return _figure(w["title"], rows, said, [_said(w["legend"], places=_n(nums["places"], lang))])


# --- running the learner's function in a picture -------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of a picture


def _quiet_many(fn, calls) -> list:
    """The learner's function on many calls, each with fresh copies of its arguments, without what it prints;
    _STOPPED for a call that stops with an error."""
    out = []
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        for args in calls:
            try:
                out.append(fn(*(copy.deepcopy(a) for a in args)))
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


# --- the picture under exercise 1: your training example of the course's sentence ------------------------------------------

_EXAMPLE_WORDS = {
    "en": {"title": "Your training example of a sentence of the course",
           "plan": "The plan chooses 3 of the 18 pieces, about 15%: place 5 becomes [MASK], place 12 a random piece, "
                   "*violin*, and place 16 stays *dinner*.",
           "fine": "Your inputs and labels are the course's at all 18 places. The model sees the middle row, and the "
                   "loss counts only the 3 places with a label, also place 16, where the model sees the right piece.",
           "differ": "At {n} of the 18 places, your input or label differs from the course's. At place {k}, the course "
                     "has {want}.",
           "legend": "In each column, from top to bottom: the place, the piece of the sentence (gray), your input and "
                     "your label (blue; a faint dot for None). Purple dashed: an input or label that differs from the "
                     "course's.",
           "odd": "Your function gives back {got}, and not two lists of 18.",
           "stop": "Your function stops with an error on the pieces of the sentence.",
           "slow": "Your function takes too long on the pieces of the sentence. Does a loop of yours go on longer than "
                   "it needs to?",
           "pair": "input {i}, label {l}"},
    "nl": {"title": "Jouw trainingsvoorbeeld van een zin van de course",
           "plan": "Het plan kiest 3 van de 18 stukjes, ongeveer 15%: plek 5 wordt [MASK], plek 12 een willekeurig stukje, "
                   "*violin*, en plek 16 blijft *dinner*.",
           "fine": "Je inputs en labels zijn op alle 18 plekken gelijk aan de inputs en labels van de course. Het model ziet de "
                   "middelste rij, en de loss telt alleen de 3 plekken met een label, ook plek 16, waar het model het "
                   "goede stukje ziet.",
           "differ": "Op {n} van de 18 plekken is je input of label anders dan bij de course. Op plek {k} heeft de course "
                     "{want}.",
           "legend": "In elke kolom, van boven naar beneden: de plek, het stukje van de zin (grijs), je input en je "
                     "label (blauw; een vage stip voor None). Paars gestreept: een input of label dat anders is dan bij de "
                     "course.",
           "odd": "Je functie geeft {got} terug, en geen twee lijsten van 18.",
           "stop": "Je functie stopt met een fout op de stukjes van de zin.",
           "slow": "Je functie duurt te lang op de stukjes van de zin. Gaat een van je loops langer door dan nodig is?",
           "pair": "input {i}, label {l}"},
}


def _pair_of_lists(got, n: int) -> bool:
    return (isinstance(got, (tuple, list)) and len(got) == 2
            and all(isinstance(x, list) and len(x) == n for x in got))


def example_picture(seen) -> str:
    """The learner's make_example on the 18 pieces of the course's sentence, with the plan of three places."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _EXAMPLE_WORDS[lang]
    got = _timed(lambda: _quiet_many(fn, [(EXAMPLE_PIECES, EXAMPLE_PLAN)])[0])
    plan = _said(w["plan"])
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [plan, _said(w["slow"])])
    if got is _STOPPED:
        return _figure(w["title"], [], [], [plan, _said(w["stop"])])
    if not _pair_of_lists(got, len(EXAMPLE_PIECES)):
        return _figure(w["title"], [], [], [plan, _said(w["odd"], got=_short(got, 80))])
    want = make_example(EXAMPLE_PIECES, EXAMPLE_PLAN)
    parts, said, wrong = [], [], []
    for k, piece in enumerate(EXAMPLE_PIECES):
        mine_in, mine_label = got[0][k], got[1][k]
        off_in = not same_value(mine_in, want[0][k])
        off_label = not same_value(mine_label, want[1][k])
        if off_in or off_label:
            wrong.append(k)
        label_text = "·" if mine_label is None else _short(mine_label, 14).strip('"')
        input_text = _short(mine_in, 14).strip('"')
        parts.append(f'<span style="{_PART}"><span style="font-family:var(--ea-font-code);font-size:0.7rem;'
                     f'color:var(--ea-ink-3)">{k}</span>{_chip(piece, "data")}'
                     f'{_chip(input_text, "mine", off=off_in)}'
                     f'{_chip(label_text, "mine", off=off_label, faint=mine_label is None)}</span>')
        said.append(f'{k} {piece}: ' + w["pair"].format(i=input_text, l=label_text))
    row = f'<div style="display:flex;flex-wrap:wrap">{"".join(parts)}</div>'
    notes = [plan]
    if wrong:
        k = wrong[0]
        notes.append(_said(w["differ"], n=len(wrong), k=k,
                           want=w["pair"].format(i=_rounded(want[0][k]), l=_rounded(want[1][k]))))
    else:
        notes.append(_said(w["fine"]))
    notes.append(_said(w["legend"]))
    return _figure(w["title"], [row], said, notes)


# --- the picture under exercise 3: your ranks for the eight blanks of the course -------------------------------------------

_RANK_WORDS = {
    "en": {"title": "Your rank of the hidden word in eight sentences of the course",
           "head": ("Sentence", "BERT-Tiny's first three guesses", "Hidden word", "Your rank"),
           "total": "By your ranks, BERT-Tiny's first guess is the hidden word in {one} of the 8 sentences, and the "
                    "hidden word is among its first five guesses in {five}.",
           "course": "the course: {r}",
           "legend": "Gray: BERT-Tiny's three pieces with the highest scores, each with its probability. In the color "
                     "of the text: the word the course hid, and the course's rank beside a rank of yours that differs. Blue: the rank your function gives the hidden word among "
                     "all 30,522 pieces. Purple dashed: a rank that differs from the course's.",
           "stop": "Your function stops with an error on the scores of *{s}*.",
           "slow": "Your function takes too long on the scores of the eight sentences. Does it go through the 30,522 "
                   "scores more than once?"},
    "nl": {"title": "Jouw rang van het verborgen woord in acht zinnen van de course",
           "head": ("Zin", "De eerste drie gokken van BERT-Tiny", "Verborgen woord", "Jouw rang"),
           "total": "Volgens je rangen is de eerste gok van BERT-Tiny het verborgen woord in {one} van de 8 zinnen, en "
                    "zit het verborgen woord bij de eerste vijf gokken in {five}.",
           "course": "de course: {r}",
           "legend": "Grijs: de drie stukjes met de hoogste scores van BERT-Tiny, elk met de kans erbij. In de kleur "
                     "van de tekst: het woord dat de course verborg, en de rang van de course naast een rang van jou die afwijkt. Blauw: de rang die je functie het verborgen woord "
                     "geeft tussen alle 30.522 stukjes. Paars gestreept: een rang die afwijkt van de rang van de course.",
           "stop": "Je functie stopt met een fout op de scores van *{s}*.",
           "slow": "Je functie duurt te lang op de scores van de acht zinnen. Gaat je functie meer dan één keer door "
                   "de 30.522 scores?"},
}


def rank_picture(seen) -> str:
    """The learner's rank for the hidden word of each of the course's eight sentences, beside BERT-Tiny's first three
    guesses."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _RANK_WORDS[lang]
    sentences, words = blanks(), answers()
    calls = [(blank_scores(i), piece_id(a)) for i, a in enumerate(words)]
    got = _timed(lambda: _quiet_many(fn, calls))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(w["slow"])])
    head = "".join(f'<th style="{_CELL};color:var(--ea-ink-2);font-weight:600;text-align:left">{html.escape(h)}</th>'
                   for h in w["head"])
    body, said, end = [], [], None
    one = five = 0
    course = numbers()["blank_ranks"]
    for i, (sentence, word, mine) in enumerate(zip(sentences, words, got)):
        if mine is _STOPPED:
            end = _said(w["stop"], s=sentence)
            break
        guesses = ", ".join(f"{piece_name(j)} {_d(p, lang, 2)}" for j, p in top(calls[i][0]))
        value = _number(mine)
        fine = value is not None and value == course[i]
        if value is not None and value == int(value):
            one += value == 1
            five += 1 <= value <= 5
        shown = _short(mine, 16) if value is None or value != int(value) else _n(int(value), lang)
        mark = "" if fine else f' <span style="color:var(--ea-ink)">({html.escape(w["course"].format(r=_n(course[i], lang)))})</span>'
        style = _CELL + ";font-family:var(--ea-font-code);color:var(--ea-pred-text)" + ("" if fine else ";" + _OFF)
        body.append(f'<tr><td style="{_CELL};color:var(--ea-ink-2)">{html.escape(sentence)}</td>'
                    f'<td style="{_CELL};color:var(--ea-ink-2)">{html.escape(guesses)}</td>'
                    f'<td style="{_CELL};color:var(--ea-ink)"><i>{html.escape(word)}</i></td>'
                    f'<td style="{style}">{html.escape(shown)}</td>' + (f"<td style=\"{_CELL};white-space:nowrap\">{mark}</td>" if mark else "")
                    + "</tr>")
        said.append(f"{sentence} {word}: {shown}")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse"><tr>{head}</tr>{"".join(body)}'
             f'</table></div>')
    notes = [end] if end else [_said(w["total"], one=one, five=five)]
    notes.append(_said(w["legend"]))
    return _figure(w["title"], [table], said, notes)


# --- the picture under exercise 4: the loss at every place of the sentence with two blanks ---------------------------------

_LOSS_WORDS = {
    "en": {"title": "The loss at each place of a sentence with two blanks",
           "scale": "loss",
           "mine": "Your `mlm_loss` on the 2 hidden places",
           "all": "Your `mlm_loss` on all 12 places",
           "course": "The course: {hidden} on the 2 hidden places, {all} on all 12.",
           "result": "Where BERT-Tiny sees the piece, the loss is below {seen}: the answer is right there in its "
                     "input. Over all 12 places the mean is {all}, far below the {hidden} of the two places where it "
                     "has to guess.",
           "legend": "Each row: a place of *A week has seven [MASK] and a year has twelve [MASK].* between [CLS] and "
                     "[SEP], with the right piece after the arrow where the course hid a word, and minus the log of "
                     "the probability BERT-Tiny gives the right piece there. In the color of the text: the two hidden "
                     "places. Gray: the places where BERT-Tiny sees the piece. Blue: your function, on the 2 hidden "
                     "places and on all 12 places, each with its right piece. Purple dashed: a number that differs "
                     "from the course's.",
           "stop": "Your function stops with an error on the scores of the sentence.",
           "slow": "Your function takes too long on the scores of the sentence.",
           "odd": "Your function gives back {got}, and not one number."},
    "nl": {"title": "De loss op elke plek van een zin met twee gaten",
           "scale": "loss",
           "mine": "Je `mlm_loss` op de 2 verborgen plekken",
           "all": "Je `mlm_loss` op alle 12 plekken",
           "course": "De course: {hidden} op de 2 verborgen plekken, {all} op alle 12.",
           "result": "Waar BERT-Tiny het stukje ziet, is de loss lager dan {seen}: het antwoord staat gewoon in de "
                     "input. Over alle 12 plekken is het gemiddelde {all}, ver onder de {hidden} van de twee plekken "
                     "waar het model moet gokken.",
           "legend": "Elke rij: een plek van *A week has seven [MASK] and a year has twelve [MASK].* tussen [CLS] en "
                     "[SEP], met het goede stukje na de pijl waar de course een woord verborg, en min de log van de kans "
                     "die BERT-Tiny daar aan het goede stukje geeft. In de kleur van de tekst: de twee verborgen plekken. "
                     "Grijs: de plekken waar BERT-Tiny het stukje ziet. Blauw: je functie, op de 2 verborgen plekken "
                     "en op alle 12 plekken, elk met het goede stukje. Paars gestreept: een getal dat afwijkt van het getal "
                     "van de course.",
           "stop": "Je functie stopt met een fout op de scores van de zin.",
           "slow": "Je functie duurt te lang op de scores van de zin.",
           "odd": "Je functie geeft {got} terug, en niet één getal."},
}

_LOSS_TOP = 4.0   # the loss a full bar stands for in the picture under exercise 4


def _loss_row(label: str, value: float, role: str, lang: str, off: bool = False) -> str:
    line, color = _ROLE[role]
    width = max(0.0, min(1.0, value / _LOSS_TOP)) * 100
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL};font-family:var(--ea-font-code)">{html.escape(label)}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(_d(value, lang, 2))}</span></div>')


def loss_picture(seen) -> str:
    """The loss at each place of the course's sentence with two blanks, and the learner's mlm_loss on the two hidden
    places and on all 12 places between [CLS] and [SEP]."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _LOSS_WORDS[lang]
    nums, losses, right = numbers(), week_losses(), week_right()
    rows, said = [], []
    for k in range(1, len(WEEK_PIECES) - 1):
        label = WEEK_PIECES[k] + (f" → {WEEK_HIDDEN[k]}" if k in WEEK_HIDDEN else "")
        rows.append(_loss_row(label, losses[k], "course" if k in WEEK_HIDDEN else "data", lang))
        said.append(f"{label}: {_d(losses[k], lang, 2)}")
    hidden = sorted(WEEK_HIDDEN)
    every = list(range(1, len(WEEK_PIECES) - 1))
    calls = [(week_scores(), hidden, [right[k] for k in hidden]), (week_scores(), every, [right[k] for k in every])]
    got = _timed(lambda: _quiet_many(fn, calls))
    notes = []
    if got is _TOO_SLOW:
        notes.append(_said(w["slow"]))
    elif any(g is _STOPPED for g in got):
        notes.append(_said(w["stop"]))
    else:
        values = [_number(g) for g in got]
        if any(v is None for v in values):
            bad = got[values.index(None)]
            notes.append(_said(w["odd"], got=_short(bad, 40)))
        else:
            rows.append('<div style="height:0.5rem"></div>')
            for key, value, want in (("mine", values[0], nums["hidden_loss"]), ("all", values[1], nums["all_loss"])):
                off = abs(value - want) > 1e-3
                rows.append(_loss_row_text(_marked(w[key]), value, lang, off))
                said.append(f'{_bare(w[key])}: {_d(value, lang, 2)}')
            if any(abs(v - c) > 1e-3 for v, c in zip(values, (nums["hidden_loss"], nums["all_loss"]))):
                notes.append(_said(w["course"], hidden=_d(nums["hidden_loss"], lang, 2), all=_d(nums["all_loss"], lang, 2)))
            else:
                notes.append(_said(w["result"], seen=_d(math.ceil(nums["seen_max"] * 100) / 100, lang, 2), all=_d(nums["all_loss"], lang, 2),
                                   hidden=_d(nums["hidden_loss"], lang, 2)))
    notes.append(_said(w["legend"]))
    return _figure(w["title"], rows, said, notes)


def _loss_row_text(label_html: str, value: float, lang: str, off: bool) -> str:
    """A row for the learner's number, with a label that may hold code."""
    line, color = _ROLE["mine"]
    width = max(0.0, min(1.0, value / _LOSS_TOP)) * 100
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{label_html}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(_d(value, lang, 2))}</span></div>')


# --- the explore: BERT-Tiny fills in your blanks -----------------------------------------------------------------------------

LONGEST = 200   # characters of the explore's sentence that count
BLANKS_SHOWN = 3

_EXPLORE_WORDS = {
    "en": {"title": "What BERT-Tiny fills in",
           "blank": "Blank {n}",
           "both": "Both sides", "before": "Only the words before",
           "none": "Your sentence has no [MASK]. Type [MASK] where BERT-Tiny should fill in a piece, in capitals and with "
                   "the square brackets.",
           "more": "Only the first {shown} of the {n} blanks are shown.",
           "cut": "Only the first {n} characters count.",
           "pieces": "The pieces BERT-Tiny sees: {pieces}",
           "legend": "For each [MASK], BERT-Tiny's five pieces with the highest probability, as bars where a full bar "
                     "is 1. In the color of the text: with the words on both sides of the [MASK]. Gray: with every "
                     "piece after the [MASK] made a [MASK] too, so that only the words before it are left."},
    "nl": {"title": "Wat BERT-Tiny invult",
           "blank": "Gat {n}",
           "both": "Beide kanten", "before": "Alleen de woorden ervoor",
           "none": "Je zin heeft geen [MASK]. Typ [MASK] waar BERT-Tiny een stukje moet invullen, in hoofdletters en met "
                   "de vierkante haken.",
           "more": "Alleen de eerste {shown} van de {n} gaten staan erbij.",
           "cut": "Alleen de eerste {n} tekens tellen.",
           "pieces": "De stukjes die BERT-Tiny ziet: {pieces}",
           "legend": "Bij elke [MASK] de vijf stukjes met de hoogste kans van BERT-Tiny, als balken waarbij een volle "
                     "balk 1 is. In de kleur van de tekst: met de woorden aan beide kanten van de [MASK]. Grijs: met "
                     "elk stukje na de [MASK] ook als [MASK], zodat alleen de woorden ervoor overblijven."},
}


def _guess_rows(guesses: list, role: str, lang: str) -> str:
    line, color = _ROLE[role]
    out = []
    for piece, p in guesses:
        width = max(0.0, min(1.0, p)) * 100
        out.append(f'<div style="{_ROW}"><span style="font-family:var(--ea-font-code);font-size:0.8rem;width:7rem;'
                   f'flex:none;overflow:hidden;text-overflow:ellipsis;color:{color}">{html.escape(piece)}</span>'
                   f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;'
                   f'background:{line};border-radius:2px"></span></span><span style="{_NUM};color:{color}">'
                   f'{html.escape(_d(p, lang))}</span></div>')
    return "".join(out)


def explore_picture(text: str, lang: str) -> str:
    """BERT-Tiny's five best pieces for every [MASK] of the learner's sentence (at most three), with both sides and
    with only the words before."""
    from everything_ai import bert
    w = _EXPLORE_WORDS[lang]
    notes = []
    if len(text) > LONGEST:
        text = text[:LONGEST]
        notes.append(_said(w["cut"], n=LONGEST))
    pieces = bert.pieces(text)
    n = pieces.count(MASK)
    if n == 0:
        return _figure(w["title"], [], [], [_said(w["none"])] + notes)
    both = bert.fill(text, top=5)[:BLANKS_SHOWN]
    before = bert.fill(text, top=5, before_only=True)[:BLANKS_SHOWN]
    rows, said = [], []
    for k, (a, b) in enumerate(zip(both, before), start=1):
        rows.append(_head(w["blank"].format(n=k)))
        rows.append(f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.5rem">'
                    f'<div style="flex:1;min-width:14rem"><div style="{_UI};margin:0 0 0.2rem">{html.escape(w["both"])}</div>'
                    f'{_guess_rows(a, "course", lang)}</div>'
                    f'<div style="flex:1;min-width:14rem"><div style="{_UI};margin:0 0 0.2rem">{html.escape(w["before"])}</div>'
                    f'{_guess_rows(b, "data", lang)}</div></div>')
        said.append(w["blank"].format(n=k) + ": " + ", ".join(p for p, _ in a) + " | " + ", ".join(p for p, _ in b))
    if n > BLANKS_SHOWN:
        notes.append(_said(w["more"], shown=BLANKS_SHOWN, n=n))
    shown_pieces = " ".join(pieces[:60]) + (" …" if len(pieces) > 60 else "")
    notes.append(_said(w["pieces"], pieces=shown_pieces))
    notes.append(_said(w["legend"]))
    return _figure(w["title"], rows, said, notes)


# --- the checks -------------------------------------------------------------------------------------------------------------

REQUIRED = ["make_example", "keep-some", "rank", "mlm_loss"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name or its code, as the lesson
    writes it. The learner's function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


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


def _rounded(v, depth: int = 0) -> str:
    """A value the way Python shows it, with its words in double quotes, also inside a dict, a tuple or a list; at most
    30 items and three levels deep. A tensor or array of one number as that number."""
    if depth > 3:
        return "…"
    if isinstance(v, np.bool_):
        return str(bool(v))
    if isinstance(v, torch.Tensor) and v.dtype == torch.bool and v.numel() == 1:
        return str(bool(v.item()))
    if isinstance(v, (torch.Tensor, np.ndarray, np.generic)) and not isinstance(v, np.str_):
        value = _number(v)
        if value is not None and not isinstance(v, (np.bool_,)):
            if float(value).is_integer() and (isinstance(v, np.integer) or getattr(v, "dtype", None) is not None
                                              and str(v.dtype).startswith(("int", "torch.int"))):
                return str(int(value))
            v = value
        else:
            return " ".join(_repr(v).split())[:80]
    if isinstance(v, float):
        if not math.isfinite(v):
            return repr(v)
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
    if isinstance(v, str) and v.isprintable() and '"' not in v and "\\" not in v:
        return f'"{v}"'   # double quotes, as in the exercise texts
    return _repr(v)


def show_value(v) -> str:
    """How {got} and {want} show a value: as code, cut after 120 characters."""
    text = _rounded(v)
    if len(text) > 120:
        text = text[:119] + "…"
    if _is_number(v) or "`" in text:
        return text
    return f"`{text}`"


def _is_number(v) -> bool:
    """A number, also a NumPy number or a tensor or array of one number, but not True or False."""
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    if isinstance(v, (torch.Tensor, np.ndarray)) and "bool" not in str(v.dtype):
        return _number(v) is not None
    return False


def _same_example(got, want) -> bool:
    """Two lists, inputs and labels, as a tuple or a list of two."""
    return (isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[0], list) and isinstance(got[1], list)
            and same_value(got[0], want[0]) and same_value(got[1], want[1]))


def _same_number(got, want) -> bool:
    value = _number(got)
    return value is not None and abs(value - float(want)) < 1e-4


def _not_pair(got, text) -> bool:
    return got is not None and not (isinstance(got, (tuple, list)) and len(got) == 2)


def _not_whole(got, text) -> bool:
    return isinstance(got, (bool, np.bool_)) or got is not None and _number(got) is None


def _not_number(got, text) -> bool:
    return got is not None and _number(got) is None


def _batch_mismatch(e) -> bool:
    return isinstance(e, ValueError) and "batch_size" in str(e)


def _out_of_rows(e) -> bool:
    return isinstance(e, IndexError) and "out of bounds" in str(e)


def _list_target(e) -> bool:
    return isinstance(e, TypeError) and "must be Tensor" in str(e)


# exercise 1: make_example. The visible cases: two [MASK]s, then a [MASK], a random piece and a piece that stays; then
# no plan, a plan for every place, a piece that occurs twice, a plan written out of order with a piece that continues a
# word, and the 18 pieces of the course's sentence with its plan of three places.
MX_CASES = [
    (["the", "cat", "sat", "on", "the", "mat", "."], {1: MASK, 5: MASK}),
    (["i", "drink", "a", "cup", "of", "coffee", "."], {1: MASK, 3: "river", 5: "coffee"}),
    (["hello", "world"], {}),
    (["a", "b", "c"], {0: MASK, 1: MASK, 2: MASK}),
    (["the", "the", "end"], {1: MASK}),
    (["they", "were", "play", "##ing", "outside"], {3: MASK, 0: "apple"}),
    (Named("C.EXAMPLE_PIECES", list(_EXAMPLE_PIECES)), Named("C.EXAMPLE_PLAN", dict(_EXAMPLE_PLAN))),
]
_MX_HEAD = "def make_example(pieces, replace):\n"
_MX_TOP = _MX_HEAD + "    inputs = []\n    labels = []\n    for k in range(len(pieces)):\n"
_MX_END = "    return inputs, labels\n"
_MX_START = _MX_TOP + "        ...\n" + _MX_END
_MX = (_MX_TOP + "        if k in replace:\n            inputs.append(replace[k])\n            labels.append(pieces[k])\n"
       "        else:\n            inputs.append(pieces[k])\n            labels.append(None)\n" + _MX_END)
_MX_MASK_ONLY = _MX.replace("            labels.append(pieces[k])\n",
                            '            if replace[k] == "[MASK]":\n                labels.append(pieces[k])\n'
                            "            else:\n                labels.append(None)\n")
_MX_REPLACEMENT = _MX.replace("labels.append(pieces[k])", "labels.append(replace[k])")
_MX_UNCHANGED = _MX.replace("inputs.append(replace[k])", "inputs.append(pieces[k])")
_MX_ALL_LABELS = _MX.replace("labels.append(None)", "labels.append(pieces[k])")
_MX_ALWAYS_MASK = _MX.replace("inputs.append(replace[k])", 'inputs.append("[MASK]")')
_MX_SWAPPED = _MX.replace("return inputs, labels", "return labels, inputs")
_MX_INDEX = (_MX_HEAD + "    inputs = []\n    labels = []\n    for p in pieces:\n        k = pieces.index(p)\n"
             "        if k in replace:\n            inputs.append(replace[k])\n            labels.append(p)\n"
             "        else:\n            inputs.append(p)\n            labels.append(None)\n" + _MX_END)

# exercise 3: rank. The visible cases: a score in the middle and the highest; then the lowest, two ties (one with the
# right piece at the top, one below it), only negative scores, and the hidden words of three of the course's
# sentences: coffee (rank 2), capital (16) and rain (341).
_A4 = Named("np.array([0.5, 2.0, 1.0, -1.0])", np.array([0.5, 2.0, 1.0, -1.0]))


def _blank(i: int) -> Named:
    return Named(f"C.blank_scores({i})", blank_scores(i))


RK_CASES = [
    (_A4, 2),
    (_A4, 1),
    (_A4, 3),
    (Named("np.array([3.0, 5.0, 5.0, 1.0])", np.array([3.0, 5.0, 5.0, 1.0])), 2),
    (Named("np.array([5.0, 3.0, 3.0, 1.0])", np.array([5.0, 3.0, 3.0, 1.0])), 2),
    (Named("np.array([-2.0, -1.5, -3.0])", np.array([-2.0, -1.5, -3.0])), 0),
    (_blank(1), piece_id("coffee")),
    (_blank(6), piece_id("capital")),
    (_blank(7), piece_id("rain")),
]
_RK_HEAD = "import numpy as np\n\n\ndef rank(scores, right):\n"
_RK_START = _RK_HEAD + "    ...\n"
_RK = _RK_HEAD + "    better = np.sum(scores > scores[right])\n    return better + 1\n"
_RK_NO_ONE = _RK.replace("return better + 1", "return better")
_RK_AT_LEAST = _RK.replace("scores > scores[right]", "scores >= scores[right]").replace("return better + 1",
                                                                                      "return better")
_RK_AT_LEAST_ONE = _RK.replace("scores > scores[right]", "scores >= scores[right]")
_RK_LOWER = _RK.replace("scores > scores[right]", "scores < scores[right]")
_RK_ID = _RK.replace("scores > scores[right]", "scores > right")
_RK_ZERO = _RK.replace("scores > scores[right]", "scores > 0")
_RK_TOP = _RK_HEAD + "    return np.argmax(scores)\n"
_RK_BOOL = _RK_HEAD + "    return np.argmax(scores) == right\n"

# exercise 4: mlm_loss. The visible cases: one place, then two; then the two in the other order, a place where the right
# piece has the highest score, every place, a table of 4 rows of 5 with three places, and the two hidden places of the
# course's sentence with two blanks.
_S3 = Named("torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 2.0, 0.5]])",
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 2.0, 0.5]]))
_S4 = Named("torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0], [0.5, 0.5, 0.5, 0.5, 0.5], [3.0, 1.0, 0.0, 1.0, 0.0], "
            "[0.0, 0.0, 4.0, 0.0, 1.0]])",
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0], [0.5, 0.5, 0.5, 0.5, 0.5], [3.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 4.0, 0.0, 1.0]]))
ML_CASES = [
    (_S3, [2], [0]),
    (_S3, [0, 2], [0, 1]),
    (_S3, [2, 0], [1, 0]),
    (_S3, [1], [1]),
    (_S3, [0, 1, 2], [0, 1, 2]),
    (_S4, [0, 2, 3], [4, 0, 2]),
    (Named("C.week_scores()", week_scores()), [5, 11], [piece_id("days"), piece_id("months")]),
]
_ML_HEAD = "import torch\nfrom torch import nn\n\n\ndef mlm_loss(scores, places, right):\n"
_ML_START = _ML_HEAD + "    ...\n"
_ML = _ML_HEAD + "    return nn.CrossEntropyLoss()(scores[places], torch.tensor(right))\n"
_ML_SUM = _ML.replace("nn.CrossEntropyLoss()", 'nn.CrossEntropyLoss(reduction="sum")')
_ML_BY_RIGHT = _ML.replace("scores[places]", "scores[right]")
_ML_PLACES = _ML.replace("torch.tensor(right)", "torch.tensor(places)")
_ML_ALL = _ML.replace("scores[places]", "scores")
_ML_LIST = _ML.replace("torch.tensor(right)", "right")
_ML_LOOP = (_ML_HEAD + "    loss_fn = nn.CrossEntropyLoss()\n    total = 0\n    for i in range(len(places)):\n"
            "        k = places[i]\n        total += loss_fn(scores[k:k + 1], torch.tensor([right[i]]))\n"
            "    return total / len(places)\n")
_ML_LOOP_SUM = _ML_LOOP.replace("return total / len(places)", "return total")
_ML_MEAN_TWICE = _ML.replace("torch.tensor(right))\n", "torch.tensor(right)) / len(places)\n")


EXERCISES = {
    "make_example": FunctionBox(
        name="make_example",
        picture=example_picture,
        function="make_example",
        start=_MX_START,
        solution=_MX,
        visible=MX_CASES[:2],
        hidden=MX_CASES[2:],
        compare=_same_example,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("mask-only", wrong=_MX_MASK_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: where the plan puts a random piece or keeps the piece, your labels hold None. The model must give the original piece at every place the plan chose, whatever it sees there. Which places get a label?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: waar het plan een willekeurig stukje neerzet of het stukje ongewijzigd houdt, bevatten je labels None. Het model moet op elke plek die het plan koos het oorspronkelijke stukje geven, wat het daar ook ziet. Welke plekken krijgen een label?"),
            Mistake("replacement", wrong=_MX_REPLACEMENT,
                    en="`{call}` gives {got}, and it should give {want}: your labels hold what the plan puts in, such as `\"[MASK]\"`. A label is the answer the model must give at that place: which piece is that?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je labels bevatten wat het plan erin zet, zoals `\"[MASK]\"`. Een label is het antwoord dat het model op een gekozen plek moet geven: welk stukje is dat?"),
            Mistake("unchanged", wrong=_MX_UNCHANGED,
                    en="`{call}` gives {got}, and it should give {want}: your inputs are the pieces as they were, so the model would see every answer. What does the model see at a place the plan chose?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je inputs zijn de oorspronkelijke stukjes, dus het model zou elk antwoord zien. Wat ziet het model op een plek die het plan koos?"),
            Mistake("always-mask", wrong=_MX_ALWAYS_MASK,
                    en="`{call}` gives {got}, and it should give {want}: your inputs have `\"[MASK]\"` at every chosen place, also where the plan says `\"river\"` or `\"coffee\"`. What does `replace` say goes in at a place?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je inputs hebben `\"[MASK]\"` op elke gekozen plek, ook waar het plan `\"river\"` of `\"coffee\"` zegt. Wat zegt `replace` dat er op een plek in komt?"),
            Mistake("all-labels", wrong=_MX_ALL_LABELS,
                    en="`{call}` gives {got}, and it should give {want}: your labels have a piece at every place. Which places does the loss count, and what goes at the others?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je labels hebben op elke plek een stukje. Welke plekken telt de loss, en wat komt er op de andere?"),
            Mistake("swapped", wrong=_MX_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the labels first and the inputs second. In which order does the exercise ask for the two lists?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft eerst de labels en dan de inputs. In welke volgorde vraagt de oefening om de twee lijsten?"),
            Mistake("index", wrong=_MX_INDEX,
                    en="`{call}` gives {got}, and it should give {want}: here a piece occurs twice, and `pieces.index` finds only the first of the two. Which loop gives every place once, 0, 1, 2 and so on?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hier komt een stukje twee keer voor, en `pieces.index` vindt alleen het eerste van de twee. Welke loop geeft elke plek één keer, 0, 1, 2 enzovoort?"),
            Mistake("not-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give {want}: two lists, the inputs and the labels. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: twee lijsten, de inputs en de labels. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for every place `k`, add what the model sees to `inputs` and the answer, or None, to `labels`.",
                      "nl": "Vervang de `...`: voeg voor elke plek `k` toe wat het model ziet aan `inputs` en het antwoord, of None, aan `labels`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. At a place in `replace`, what does the model see, and which answer must it give? And at a place that is not in it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wat ziet het model op een plek in `replace`, en welk antwoord moet het geven? En op een plek die er niet in staat?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work without a plan, for a plan of every place, and for a piece that occurs twice?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook zonder plan, voor een plan van elke plek, en voor een stukje dat twee keer voorkomt?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it still end with the `return` line of the start?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie nog met de `return`-regel van het begin?"},
            "correct": {"en": "Your `make_example` is right in all {n} cases. Look at the picture: your training example of a sentence of the course.",
                        "nl": "Je `make_example` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw trainingsvoorbeeld van een zin van de course."},
        },
    ),
    "rank": FunctionBox(
        name="rank",
        picture=rank_picture,
        function="rank",
        start=_RK_START,
        solution=_RK,
        visible=RK_CASES[:2],
        hidden=RK_CASES[2:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("no-one", wrong=_RK_NO_ONE,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of pieces with a higher score. The first guess has none above it, and its rank is 1: what is still missing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het aantal stukjes met een hogere score. De eerste gok heeft er geen boven zich, en de rang is 1: wat ontbreekt er nog?"),
            Mistake("at-least-one", wrong=_RK_AT_LEAST_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your count also holds the right piece itself, since its score is equal to its own score. Which pieces stand above it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling bevat ook het goede stukje zelf, omdat elk stukje dezelfde score heeft als zichzelf. Welke stukjes staan erboven?"),
            Mistake("at-least", wrong=_RK_AT_LEAST,
                    en="`{call}` gives {got}, and it should give {want}: here another piece has the same score as the right one, and your count holds it. Does a piece with the same score stand above the right one?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hier heeft een ander stukje dezelfde score als het goede, en je telling bevat het. Staat een stukje met dezelfde score boven het goede?"),
            Mistake("lower", wrong=_RK_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the pieces with a lower score. Which pieces stand above the right one in the list of guesses?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de stukjes met een lagere score. Welke stukjes staan boven het goede in de lijst van gokken?"),
            Mistake("zero", wrong=_RK_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the pieces with a score above 0. Which score should a piece beat to stand above the right one?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de stukjes met een score boven 0. Welke score moet een stukje verslaan om boven het goede te staan?"),
            Mistake("id-as-score", wrong=_RK_ID,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the scores with `right` itself, the number of the piece. With which score should every score be compared?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt de scores met `right` zelf, het nummer van het stukje. Met welke score moet elke score vergeleken worden?"),
            Mistake("top-piece", wrong=_RK_TOP,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of the piece with the highest score, BERT-Tiny's first guess. What should the function give: a piece, or a place in the list of guesses?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het nummer van het stukje met de hoogste score, de eerste gok van BERT-Tiny. Wat moet de functie geven: een stukje, of een plaats in de lijst van gokken?"),
            Mistake("bool", wrong=_RK_BOOL,
                    en="`{call}` gives {got}, and it should give {want}: a whole number, the place of the right piece in the list of guesses. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een heel getal, de plaats van het goede stukje in de lijst van gokken. Wat geeft je `return`-regel terug?"),
            Mistake("not-whole", matches=_not_whole,
                    en="`{call}` gives {got}, and it should give {want}: a whole number, the place of the right piece in the list of guesses. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een heel getal, de plaats van het goede stukje in de lijst van gokken. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the place of the piece with number `right` in the list of all pieces, ranked from the highest score down.",
                      "nl": "Vervang de `...`: geef de plaats van het stukje met nummer `right` in de lijst van alle stukjes terug, gerangschikt van de hoogste score omlaag."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many pieces have a higher score than the piece with number `right`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Hoeveel stukjes hebben een hogere score dan het stukje met nummer `right`?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function also work for the lowest score, and when two pieces have the same score?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor de laagste score, en als twee stukjes dezelfde score hebben?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return`?"},
            "slow": {"en": "Your function takes too long on the 30,522 scores of BERT-Tiny. Does it go through them more than once?",
                     "nl": "Je functie duurt te lang op de 30.522 scores van BERT-Tiny. Gaat je functie er meer dan één keer doorheen?"},
            "correct": {"en": "Your `rank` is right in all {n} cases, also for the hidden words of three sentences of the course. Look at the picture: your ranks for all eight.",
                        "nl": "Je `rank` klopt in alle {n} gevallen, ook voor de verborgen woorden van drie zinnen van de course. Kijk naar de afbeelding: je rangen voor alle acht."},
        },
    ),
    "mlm_loss": FunctionBox(
        name="mlm_loss",
        picture=loss_picture,
        function="mlm_loss",
        start=_ML_START,
        solution=_ML,
        visible=ML_CASES[:2],
        hidden=ML_CASES[2:],
        compare=_same_number,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("all-rows", error=_batch_mismatch,
                    en="With `{call}`, your function stops with an error: the loss gets a row of scores for every place of the sentence, and one right piece for each chosen place only. Which rows belong to the chosen places?",
                    nl="Met `{call}` stopt je functie met een fout: de loss krijgt een rij scores voor elke plek van de zin, en alleen voor elke gekozen plek één goed stukje. Welke rijen horen bij de gekozen plekken?"),
            Mistake("list-target", error=_list_target,
                    en="With `{call}`, your function stops with an error: `nn.CrossEntropyLoss` takes the right pieces as a tensor of whole numbers, and `right` is a list. How did you make a tensor from a list before?",
                    nl="Met `{call}` stopt je functie met een fout: `nn.CrossEntropyLoss` neemt de goede stukjes als tensor van hele getallen, en `right` is een lijst. Hoe maakte je eerder een tensor van een lijst?"),
            Mistake("by-right", wrong=_ML_BY_RIGHT,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the rows at the numbers in `right`, and those are pieces, not places. Which list holds the places?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kiest de rijen bij de nummers in `right`, en dat zijn stukjes, geen plekken. Welke lijst bevat de plekken?"),
            Mistake("places-as-right", wrong=_ML_PLACES,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the places as the right answers. Which list holds the right pieces?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de plekken als goede antwoorden. Welke lijst bevat de goede stukjes?"),
            Mistake("sum", wrong=_ML_SUM,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum of the losses at the chosen places. Which number does the exercise ask for?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de som van de losses op de gekozen plekken. Om welk getal vraagt de oefening?"),
            Mistake("mean-twice", wrong=_ML_MEAN_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: that is less than the mean loss at the chosen places. Over several rows, does `nn.CrossEntropyLoss()` give the sum of their losses or the mean?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is minder dan de gemiddelde loss op de gekozen plekken. Geeft `nn.CrossEntropyLoss()` over meer rijen de som van de losses of het gemiddelde?"),
            Mistake("out-of-rows", error=_out_of_rows,
                    en="With `{call}`, your function stops with an error: it asks for a row of `scores` that is not there. `scores` has one row per place of the sentence: which list holds places, and which holds pieces?",
                    nl="Met `{call}` stopt je functie met een fout: je functie vraagt om een rij van `scores` die er niet is. `scores` heeft één rij per plek van de zin: welke lijst bevat plekken, en welke stukjes?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, the mean loss at the chosen places. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één getal, de gemiddelde loss op de gekozen plekken. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the cross-entropy of the rows of `scores` at the places in `places`, with the pieces in `right` as the right answers.",
                      "nl": "Vervang de `...`: geef de cross-entropy terug van de rijen van `scores` op de plekken in `places`, met de stukjes in `right` als goede antwoorden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which rows of `scores` count, and which right piece goes with each of them?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke rijen van `scores` tellen, en welk goed stukje hoort bij elke rij?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function also work with the places in another order, and with every place?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook met de plekken in een andere volgorde, en met elke plek?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return`?"},
            "correct": {"en": "Your `mlm_loss` is right in all {n} cases, also for BERT-Tiny's scores of a sentence of the course. Look at the picture: the loss at every place of that sentence.",
                        "nl": "Je `mlm_loss` klopt in alle {n} gevallen, ook voor de scores van BERT-Tiny op een zin van de course. Kijk naar de afbeelding: de loss op elke plek van de zin."},
        },
    ),
}

CASES = [
    ("make_example", _MX_START, "empty", None),
    ("make_example", _MX, "correct", None),
    ("make_example", _MX_HEAD + "    inputs = list(pieces)\n    labels = [None] * len(pieces)\n    for k in replace:\n"
                     "        inputs[k] = replace[k]\n        labels[k] = pieces[k]\n    return inputs, labels\n",
     "correct", None),
    ("make_example", _MX.replace("return inputs, labels", "return [inputs, labels]"), "correct", None),
    ("make_example", _MX_MASK_ONLY, "wrong", "mask-only"),
    ("make_example", _MX_REPLACEMENT, "wrong", "replacement"),
    ("make_example", _MX_UNCHANGED, "wrong", "unchanged"),
    ("make_example", _MX_ALWAYS_MASK, "wrong", "always-mask"),
    ("make_example", _MX_ALL_LABELS, "wrong", "all-labels"),
    ("make_example", _MX_SWAPPED, "wrong", "swapped"),
    ("make_example", _MX_INDEX, "wrong", "index"),
    ("make_example", _MX.replace("return inputs, labels", "return inputs"), "wrong", "not-pair"),
    ("make_example", _MX.replace(_MX_END, "    if len(pieces) == 0:\n        return inputs, labels\n"), "wrong",
     "unreached"),
    ("make_example", _MX_TOP + "        ...\n        inputs.append(pieces[k])\n" + _MX_END, "wrong", "dots-left"),
    ("rank", _RK_START, "empty", None),
    ("rank", _RK, "correct", None),
    ("rank", _RK_HEAD + "    count = 0\n    for s in scores:\n        if s > scores[right]:\n            count += 1\n"
             "    return count + 1\n", "correct", None),
    ("rank", _RK_HEAD + "    return sorted(scores, reverse=True).index(scores[right]) + 1\n", "correct", None),
    ("rank", _RK_NO_ONE, "wrong", "no-one"),
    ("rank", _RK_AT_LEAST_ONE, "wrong", "at-least-one"),
    ("rank", _RK_AT_LEAST, "wrong", "at-least"),
    ("rank", _RK_LOWER, "wrong", "lower"),
    ("rank", _RK_ZERO, "wrong", "zero"),
    ("rank", _RK_ID, "wrong", "id-as-score"),
    ("rank", _RK_TOP, "wrong", "top-piece"),
    ("rank", _RK_BOOL, "wrong", "bool"),
    ("rank", _RK_HEAD + "    return str(np.sum(scores > scores[right]) + 1)\n", "wrong", "not-whole"),
    ("rank", _RK_HEAD + "    better = np.sum(scores > scores[right])\n    if better < 0:\n        return better + 1\n",
     "wrong", "unreached"),
    ("rank", _RK_HEAD + "    ...\n    return 1\n", "wrong", "dots-left"),
    ("mlm_loss", _ML_START, "empty", None),
    ("mlm_loss", _ML, "correct", None),
    ("mlm_loss", _ML_LOOP, "correct", None),
    ("mlm_loss", _ML.replace("    return nn.CrossEntropyLoss()(scores[places], torch.tensor(right))\n",
                             "    loss = nn.CrossEntropyLoss()(scores[places], torch.tensor(right))\n    return loss.item()\n"),
     "correct", None),
    ("mlm_loss", _ML_ALL, "error", "all-rows"),
    ("mlm_loss", _ML_LIST, "error", "list-target"),
    ("mlm_loss", _ML_BY_RIGHT, "wrong", "by-right"),
    ("mlm_loss", _ML_PLACES, "wrong", "places-as-right"),
    ("mlm_loss", _ML_SUM, "wrong", "sum"),
    ("mlm_loss", _ML_LOOP_SUM, "wrong", "sum"),
    ("mlm_loss", _ML_MEAN_TWICE, "wrong", "mean-twice"),
    ("mlm_loss", _ML.replace("scores[places]", "scores[[2, 5]]"), "error", "out-of-rows"),
    ("mlm_loss", _ML_HEAD + "    return scores[places]\n", "wrong", "not-number"),
    ("mlm_loss", _ML_HEAD + "    loss = nn.CrossEntropyLoss()(scores[places], torch.tensor(right))\n    if loss < 0:\n"
                 "        return loss\n", "wrong", "unreached"),
    ("mlm_loss", _ML_HEAD + "    ...\n    return 0.0\n", "wrong", "dots-left"),
]
