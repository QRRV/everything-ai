# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-static-vs-contextual: the mean cosine similarity of every two rows of a table, the
self-similarity of the uses of a word (self_similarity), and a guess of the meaning of bank in a new sentence from the
mean of the example uses of each meaning (guess_sense).

The course wrote 44 sentences with bank (data/course-bank-sentences.tsv): 22 with the bank of a river, 22 with a bank
for money, in pairs with bank at the same place among the pieces, 14 pairs to learn from and 8 new ones. What BERT-Tiny
holds for bank in each of them, before its first block and after each block, the course ships
(data/bert-tiny-bank.npz, made by data/tools/make_bert_bank.py), so that the exercises and their pictures work without
the model. BERT-Tiny itself (18 MB, everything_ai.bert) the learner downloads once for the worked example and the
explore. GloVe comes from everything_ai.wordvectors. The small tables of the cases are the course's own; every number
in the text and the pictures is computed by the course (tests/test_nlp_static_vs_contextual.py computes them again)."""
import contextlib
import copy
import csv
import html
import io

import numpy as np

from everything_ai import lesson as lesson_helpers, paths, wordvectors, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, same_value, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-static-vs-contextual"
REPO = "google/bert_uncased_L-2_H-128_A-2"   # everything_ai.bert.REPO: that module imports PyTorch, so not here
BERT_FILES = [(REPO, "model.safetensors"), (REPO, "vocab.txt")]
SENTENCES = paths.ROOT / "data" / "course-bank-sentences.tsv"
VECTORS = paths.ROOT / "data" / "bert-tiny-bank.npz"
SENSES = ("river", "money")
LAYERS = 3   # before the first block, after the first and after the second

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The worked example and the explore download BERT-Tiny, {download_size(BERT_FILES, 'en')}, once from Hugging Face, when you click its button in the lesson; the exercises and the rest of the lesson work without it.",
    "nl": f"Werkt op elke computer. Het uitgewerkte voorbeeld en het onderdeel om zelf te proberen downloaden één keer BERT-Tiny, {download_size(BERT_FILES, 'nl')}, van Hugging Face, als je in de les op de knop ervoor klikt; de oefeningen en de rest van de les werken zonder.",
}

# What the course measured with BERT-Tiny itself (everything_ai.bert, its pinned files): the mean cosine similarity of
# two different words in two different sentences of the 44, as Ethayarajh (2019) compares with: every word of letters
# that BERT-Tiny keeps as one piece, other than bank (336 in all, 52,130 pairs); before the first block (0) and after
# each block. tests/test_nlp_static_vs_contextual.py measures it again when EAI_BERT_DIR names a folder with the two
# files, and checks that SPLIT holds exactly the words BERT-Tiny cuts into more than one piece.
OTHER = {0: 0.2366, 1: 0.5285, 2: 0.5745}
SPLIT = frozenset({"angler", "kayak", "otters", "reeds", "sloped", "willows", "withdrawals"})

# --- the sentences and the vectors ---------------------------------------------------------------------------------------


def _data() -> tuple:
    def make():
        with open(SENTENCES, encoding="utf-8", newline="") as f:
            rows = tuple((int(r["pair"]), r["sense"], r["set"], r["sentence"]) for r in csv.DictReader(f, delimiter="\t"))
        with np.load(VECTORS, allow_pickle=False) as z:
            layers, place = z["layers"].astype(np.float32), z["place"].astype(int)
        layers.flags.writeable = False
        return rows, layers, place
    return kept(f"{_KEY}:data", make)


def _pick(sense, part) -> list:
    rows = _data()[0]
    return [i for i, (_, s, p, _) in enumerate(rows) if sense in (None, s) and part in (None, p)]


def sentences(sense=None, part=None) -> list:
    """The course's sentences with bank, as a new list: all 44, or those of one meaning ("river" or "money") and one
    part ("example" or "new"), in the order of the file: pair by pair."""
    rows = _data()[0]
    return [rows[i][3] for i in _pick(sense, part)]


def bank_rows(layer: int, sense=None, part=None) -> np.ndarray:
    """What BERT-Tiny holds for bank in those sentences (see `sentences`), as a new table: one row of 128 numbers per
    sentence, before the first block (layer 0) or after block 1 or 2."""
    return np.array(_data()[1][_pick(sense, part), layer])


def _senses(part=None) -> list:
    rows = _data()[0]
    return [rows[i][1] for i in _pick(None, part)]


def _glove() -> tuple:
    def make():
        words, E = wordvectors.glove()
        return tuple(words), E, wordvectors.rows(words)
    return kept(f"{_KEY}:glove", make)


def glove_row(word: str) -> np.ndarray:
    """GloVe's vector of a word, 50 numbers, as a new array."""
    words, E, index = _glove()
    return np.array(E[index[word]])


def glove_neighbors(word: str, k: int = 12) -> list:
    """The k nearest neighbors of a word in GloVe, with their cosine similarity, leaving out the word itself."""
    words, E, index = _glove()
    sims = cosines(E, E[index[word]])
    sims[index[word]] = -np.inf
    return [(words[i], float(sims[i])) for i in np.argsort(-sims, kind="stable")[:k]]


def glove_place(word: str, other: str) -> int:
    """The place of `other` among the neighbors of `word` in GloVe, counted from 1, the word itself left out."""
    words, E, index = _glove()
    sims = cosines(E, E[index[word]])
    return int((sims > sims[index[other]]).sum())   # the word itself is above it too, and counts as the one before 1


def _other_words() -> list:
    """(sentence number, word) for the words that OTHER is about: of letters, kept whole by BERT-Tiny, not bank."""
    return [(n, w) for n, sentence in enumerate(sentences()) for w in _letters(sentence)
            if w != "bank" and w not in SPLIT]


def other_mean(vectors, found) -> float:
    """The mean cosine similarity of every two rows of `vectors` whose (sentence, word) in `found` differ in both."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sid = np.array([n for n, _ in found])
    wid = np.array([w for _, w in found])
    pairs = (sid[:, None] != sid[None, :]) & (wid[:, None] != wid[None, :])
    return float((unit @ unit.T)[pairs].mean())


def glove_other() -> float:
    """OTHER for GloVe: the mean cosine similarity of two different words in two different sentences of the 44."""
    words, E, index = _glove()
    found = _other_words()
    return other_mean(np.array([E[index[w]] for _, w in found], dtype=float), found)


def _letters(sentence: str) -> list:
    out, current = [], ""
    for ch in sentence.lower() + " ":
        if "a" <= ch <= "z":
            current += ch
        elif current:
            out.append(current)
            current = ""
    return out


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


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosines(E, v):
    """The cosine similarity of every row of E with v, as one array (nlp-word-embeddings)."""
    return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))


def self_similarity(rows) -> float:
    """The mean cosine similarity of every two different rows of a table."""
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    n = len(rows)
    return float(((unit @ unit.T).sum() - n) / (n * (n - 1)))


def guess_sense(v, river, money) -> str:
    """"river" when v has a higher cosine similarity with the mean row of `river` than with that of `money`, else
    "money"."""
    if cosine(v, np.mean(river, axis=0)) > cosine(v, np.mean(money, axis=0)):
        return "river"
    return "money"


def self_similarities() -> dict:
    """The self-similarity of bank over the 44 sentences: GloVe, where every use has the same vector, and BERT-Tiny
    before the first block and after each block."""
    return {"glove": 1.0, **{k: self_similarity(bank_rows(k)) for k in range(LAYERS)}}


def mean_cosine(A, B) -> float:
    """The mean cosine similarity of every row of A with every row of B."""
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float((a @ b.T).mean())


def _columns() -> list:
    """For each column of the guess picture: its key, and (v, river, money) for each of the 16 new sentences."""
    out = []
    g = glove_row("bank")
    glove_examples = np.array([g] * 14)
    out.append(("glove", [(g, glove_examples, glove_examples) for _ in _pick(None, "new")]))
    for k in range(LAYERS):
        river, money = bank_rows(k, "river", "example"), bank_rows(k, "money", "example")
        out.append((k, [(v, river, money) for v in bank_rows(k, None, "new")]))
    return out


def right_counts(fn=guess_sense) -> dict:
    """How many of the 16 new sentences `fn` gets right, per column: GloVe, and BERT-Tiny's three layers."""
    truth = _senses("new")
    return {key: sum(fn(*args) == want for args, want in zip(cases, truth)) for key, cases in _columns()}


# --- the look of the pictures -------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3)"
_CHIP_GRAY = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"


def _marked(text: str) -> str:
    parts = html.escape(text).split("*")
    return "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _dec(x: float, lang: str, digits: int = 3) -> str:
    """A number with `digits` decimals: 0.874 in English, 0,874 in Dutch; a minus as a real minus sign."""
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _with_bank(sentence: str) -> str:
    """The sentence, escaped, with bank in bold."""
    before, word, after = sentence.partition("bank")
    return f"{html.escape(before)}<b>{word}</b>{html.escape(after)}"


_LAYER_NAMES = {
    "en": {"glove": "GloVe", 0: "BERT-Tiny, before the first block", 1: "BERT-Tiny, after the first block",
           2: "BERT-Tiny, after the second block"},
    "nl": {"glove": "GloVe", 0: "BERT-Tiny, vóór het eerste block", 1: "BERT-Tiny, na het eerste block",
           2: "BERT-Tiny, na het tweede block"},
}
_SHORT_NAMES = {
    "en": {"glove": "GloVe", 0: "before block 1", 1: "after block 1", 2: "after block 2"},
    "nl": {"glove": "GloVe", 0: "vóór block 1", 1: "na block 1", 2: "na block 2"},
}
_SENSE_NAMES = {"en": {"river": "river", "money": "money"}, "nl": {"river": "rivier", "money": "geld"}}


# --- the idea's picture: the 44 uses of bank, flattened to two numbers -------------------------------------------------------

_USES_WORDS = {
    "en": {"title": "The 44 uses of bank: one vector, or one per use",
           "panels": {"glove": "GloVe", 0: "BERT-Tiny, before block 1", 2: "BERT-Tiny, after block 2"},
           "one": "all 44 on one point",
           "legend": "Each blue mark is bank in one of the course's sentences, as a model gives it: a filled circle in a "
                     "sentence about a river, an open square in one about money. GloVe has one vector for bank, so all "
                     "44 lie on one point. For BERT-Tiny, the course flattens the 128 numbers to the two directions in "
                     "which the 44 uses differ most, as in *The best shadow: PCA*. Before the first block, the vector "
                     "depends only on the place of bank, so the 44 uses lie on {places} points, one per place, with "
                     "circles and squares on top of each other: the sentences go in pairs with bank at the same place. "
                     "After the second block, the circles lie more to the left and the squares more to the right.",
           "alt": "Three panels of marks for the 44 uses of bank: one point for GloVe; {places} points, each holding "
                  "circles and squares, before the first block of BERT-Tiny; and after its second block, circles "
                  "mostly on the left and squares mostly on the right."},
    "nl": {"title": "Bank in 44 zinnen: één vector, of één per gebruik",
           "panels": {"glove": "GloVe", 0: "BERT-Tiny, vóór block 1", 2: "BERT-Tiny, na block 2"},
           "one": "alle 44 op één punt",
           "legend": "Elk blauw teken is bank in een van de zinnen van de course, zoals een model het woord geeft: een "
                     "dicht rondje in een zin over een rivier, een open vierkantje in een zin over geld. GloVe heeft "
                     "één vector voor bank, dus alle 44 liggen op één punt. Voor BERT-Tiny maakt de course van de 128 "
                     "getallen er twee: de twee richtingen waarin bank in de 44 zinnen het meest verschilt, zoals in *De "
                     "beste schaduw: PCA*. Vóór het eerste block hangt de vector alleen af van de plek van bank, dus "
                     "de 44 zinnen geven samen {places} punten, één per plek, met rondjes en vierkantjes op elkaar: de "
                     "zinnen komen in paren, met bank op dezelfde plek. Na het tweede block liggen de rondjes meer "
                     "links en de vierkantjes meer rechts.",
           "alt": "Drie vakken met tekens voor bank in 44 zinnen: één punt voor GloVe; {places} punten met elk rondjes "
                  "en vierkantjes, vóór het eerste block van BERT-Tiny; en na het tweede block de rondjes vooral links "
                  "en de vierkantjes vooral rechts."},
}
_PANEL = 200   # px per panel of the picture
_GAP = 20


def flattened(layer: int) -> np.ndarray:
    """The 44 uses of bank after `layer`, flattened to two numbers each: minus their mean, along the two directions in
    which they differ most (PCA), turned so that the river uses lie on the left on average."""
    X = bank_rows(layer).astype(float)
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt[:2].T
    river = np.array([s == "river" for s in _senses()])
    if P[river, 0].mean() > P[~river, 0].mean():
        P[:, 0] = -P[:, 0]
    if P[:, 1].max() < -P[:, 1].min():   # the larger spread upward: the same picture every time
        P[:, 1] = -P[:, 1]
    return P


def _marks(points, senses, x0: float, size: float) -> str:
    """Circles and squares for the uses, scaled into a square panel at x0."""
    span = float(np.abs(points).max()) or 1.0
    out = []
    for (x, y), sense in sorted(zip(points.tolist(), senses), key=lambda t: t[1] == "river"):
        cx, cy = x0 + size / 2 + x / span * (size / 2 - 14), 26 + size / 2 - y / span * (size / 2 - 14)
        if sense == "river":
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="var(--ea-pred)"/>')
        else:
            out.append(f'<rect x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" fill="none" '
                       f'stroke="var(--ea-pred)" stroke-width="1.4"/>')
    return "".join(out)


def uses_picture(lang: str) -> str:
    w = _USES_WORDS[lang]
    senses = _senses()
    width = 3 * _PANEL + 2 * _GAP
    parts = []
    for n, key in enumerate(("glove", 0, 2)):
        x0 = n * (_PANEL + _GAP)
        parts.append(f'<rect x="{x0}" y="26" width="{_PANEL}" height="{_PANEL}" fill="none" stroke="var(--ea-line)"/>')
        parts.append(f'<text x="{x0 + _PANEL / 2}" y="16" text-anchor="middle" font-size="12.5" '
                     f'fill="var(--ea-ink-2)">{html.escape(w["panels"][key])}</text>')
        if key == "glove":
            points = np.zeros((len(senses), 2))
            parts.append(_marks(points, senses, x0, _PANEL))
            parts.append(f'<text x="{x0 + _PANEL / 2}" y="{26 + _PANEL / 2 + 24}" text-anchor="middle" '
                         f'font-size="11.5" fill="var(--ea-ink-3)">{html.escape(w["one"])}</text>')
        else:
            parts.append(_marks(flattened(key), senses, x0, _PANEL))
    svg = (f'<svg viewBox="0 0 {width} {_PANEL + 34}" width="{width}" style="max-width:100%;height:auto;'
           f'font-family:var(--ea-font-ui)" role="img" aria-label="{html.escape(w["alt"].format(places=len(set(_data()[2].tolist()))))}">{"".join(parts)}</svg>')
    places = len(set(_data()[2].tolist()))
    return yours.figure(w["title"], svg + _note(w["legend"].format(places=places)), alt=w["alt"].format(places=places))


# --- running the learner's function in a picture ----------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 10  # what the learner's function gets per call in a picture


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit; _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _number(got):
    """The learner's outcome as one float, or None."""
    try:
        arr = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(arr) if arr.shape == () and np.isfinite(arr) else None


# --- the picture under exercise 2: the self-similarity of bank, by your function ---------------------------------------------

_SELF_WORDS = {
    "en": {"title": "How alike the 44 uses of bank are, by your function",
           "head": ["Vectors", "Self-similarity of bank", "Two different words"],
           "legend": "Blue: the self-similarity of bank by your function, the mean cosine similarity of every two of its "
                     "44 uses; purple where it differs from the course's. Gray: the mean cosine similarity of two "
                     "different words in two different sentences of the 44, other than bank, as the course measured it.",
           "fine": "Your function gives the same numbers as the course for all four.",
           "differ": "For {n} of the four, your number differs from the course's.",
           "error": "Your function stops with an error on the 44 uses of bank of *{name}*.",
           "slow": "Your function takes more than {s} seconds for the 44 uses of bank.",
           "odd": "On the 44 uses of bank of *{name}*, your function gives back {got}, and not one number."},
    "nl": {"title": "Hoe sterk bank in de 44 zinnen op zichzelf lijkt, volgens je functie",
           "head": ["Vectoren", "Self-similarity van bank", "Twee verschillende woorden"],
           "legend": "Blauw: de self-similarity van bank volgens je functie, de gemiddelde cosine similarity van bank in "
                     "elk paar van de 44 zinnen; paars waar je getal verschilt van dat van de course. Grijs: de gemiddelde "
                     "cosine similarity van twee verschillende woorden in twee verschillende zinnen van de 44, behalve "
                     "bank, zoals gemeten door de course.",
           "fine": "Je functie geeft voor alle vier dezelfde getallen als de course.",
           "differ": "Voor {n} van de vier verschilt je getal van dat van de course.",
           "error": "Je functie stopt met een fout op bank in de 44 zinnen, met de vectoren van *{name}*.",
           "slow": "Je functie doet er langer dan {s} seconden over voor bank in de 44 zinnen.",
           "odd": "Op bank in de 44 zinnen, met de vectoren van *{name}*, geeft je functie {got} terug, en niet één getal."},
}


def _bar(value: float, style: str, scale: float = 220) -> str:
    width = max(0.0, min(1.0, value)) * scale
    return f'<span style="{_BAR};width:{width:.0f}px;background:{style}"></span>'


def self_picture(seen) -> str:
    """The learner's self_similarity on the 44 uses of bank: GloVe (44 times the same row) and BERT-Tiny's three
    layers, each beside the mean cosine similarity of two different words in two different sentences (OTHER)."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _SELF_WORDS[lang]
    names = _LAYER_NAMES[lang]
    course = self_similarities()
    other = {"glove": glove_other(), **OTHER}
    inputs = {"glove": np.array([glove_row("bank")] * 44), **{k: bank_rows(k) for k in range(LAYERS)}}
    table, alts, end, differ = [], [], "", 0
    for key, rows in inputs.items():
        try:
            got = _call(fn, rows)
        except Timeout:
            end = w["slow"].format(s=PICTURE_SECONDS)
            break
        if got is _STOPPED:
            end = w["error"].format(name=names[key])
            break
        value = _number(got)
        if value is None:
            end = w["odd"].format(name=names[key], got=_short(got))
            break
        right = abs(value - course[key]) < 1e-3
        differ += not right
        color = "var(--ea-pred)" if right else "var(--ea-loss)"
        table.append([html.escape(names[key]),
                      f'{_bar(value, color)} <span style="font-variant-numeric:tabular-nums">{_dec(value, lang)}</span>',
                      f'{_bar(other[key], "var(--ea-data)")} <span style="font-variant-numeric:tabular-nums;'
                      f'color:var(--ea-ink-2)">{_dec(other[key], lang)}</span>'])
        alts.append(f"{names[key]}: {_dec(value, lang)}, {_dec(other[key], lang)}")
    body = _table(table, w["head"]) if table else ""
    texts = [end] if end else []
    if table and not end and not differ:
        texts.append(w["fine"])
    elif differ:
        texts.append(w["differ"].format(n=differ))
    if table:
        texts.append(w["legend"])
    text = " ".join(texts)
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the picture under exercise 3: your guesses for the 16 new sentences --------------------------------------------------------

_GUESS_WORDS = {
    "en": {"title": "Your guess of the meaning of bank in the 16 new sentences",
           "head": ["New sentence", "Meaning"],
           "total": "Right",
           "legend": "Each column gives your function the vector of bank in the new sentence and the vectors of bank in "
                     "the 28 example sentences, 14 per meaning, all from the model and moment of that column. Blue: your guess is the meaning "
                     "of the sentence; purple: it is not. GloVe gives bank one vector, so your function sees the same "
                     "numbers 16 times.",
           "error": "Your function stops with an error on the vectors of *{name}*.",
           "slow": "Your function takes more than {s} seconds on the vectors of *{name}*.",
           "odd": "On the vectors of *{name}*, your function gives back {got}, and not \"river\" or \"money\"."},
    "nl": {"title": "Je gok naar de betekenis van bank in de 16 nieuwe zinnen",
           "head": ["Nieuwe zin", "Betekenis"],
           "total": "Goed",
           "legend": "Elke kolom geeft je functie de vector van bank in de nieuwe zin en de vectoren van bank in de 28 "
                     "voorbeeldzinnen, 14 per betekenis, allemaal van het model en het moment van die kolom. Blauw: je gok is de betekenis van de "
                     "zin; paars: je gok is een andere betekenis. GloVe geeft bank één vector, dus je functie krijgt 16 "
                     "keer dezelfde getallen.",
           "error": "Je functie stopt met een fout op de vectoren van *{name}*.",
           "slow": "Je functie doet er langer dan {s} seconden over op de vectoren van *{name}*.",
           "odd": "Op de vectoren van *{name}* geeft je functie {got} terug, en niet \"river\" of \"money\"."},
}


def guess_picture(seen) -> str:
    """The learner's guess_sense on the 16 new sentences, in four columns: GloVe and BERT-Tiny's three layers."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _GUESS_WORDS[lang]
    shorts, names, sense_names = _SHORT_NAMES[lang], _LAYER_NAMES[lang], _SENSE_NAMES[lang]
    truth = _senses("new")
    news = sentences(None, "new")
    columns, end = [], ""
    for key, cases in _columns():
        got = []
        for args in cases:
            try:
                g = _call(fn, *args)
            except Timeout:
                end = w["slow"].format(s=PICTURE_SECONDS, name=names[key])
                break
            if g is _STOPPED:
                end = w["error"].format(name=names[key])
                break
            if not isinstance(g, str) or g not in SENSES:
                end = w["odd"].format(name=names[key], got=_short(g))
                break
            got.append(g)
        if end:
            break
        columns.append((key, got))
    if not columns:
        return yours.figure(w["title"], _note(end), alt=f'{w["title"]}. ' + _bare(end))
    head = w["head"] + [shorts[key] for key, _ in columns]
    table = []
    for i, sentence in enumerate(news):
        row = [_with_bank(sentence), _chip(sense_names[truth[i]])]
        for _, got in columns:
            row.append(_chip(sense_names[got[i]], _CHIP_PRED if got[i] == truth[i] else _CHIP_PURPLE))
        table.append(row)
    counts = [sum(g == t for g, t in zip(got, truth)) for _, got in columns]
    table.append([f"<b>{html.escape(w['total'])}</b>", ""] + [f"<b>{c} / {len(truth)}</b>" for c in counts])
    alts = [f"{shorts[key]}: {c} / {len(truth)}" for (key, _), c in zip(columns, counts)]
    text = " ".join(([end] if end else []) + [w["legend"]])
    return yours.figure(w["title"], _table(table, head) + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the explore: one word in two sentences of your own ---------------------------------------------------------------------

LONGEST = 200   # characters of each sentence that count

_EXPLORE_WORDS = {
    "en": {"title": "{word} in two sentences, block by block",
           "pieces": "Pieces of sentence {n}",
           "head": ["Vectors", "Cosine similarity of the two"],
           "legend": "Blue: the cosine similarity of the vectors BERT-Tiny gives the word in the two sentences, before "
                     "the first block and after each block. Gray: the word you typed among the pieces.",
           "missing": "The word *{word}* is not one of the pieces of sentence {n}. Its pieces are above: type a word "
                      "that stands among them as it is, in small letters.",
           "split": "BERT-Tiny cuts *{word}* into more than one piece, so it is not one piece in either sentence. "
                    "Try a word that stays whole, such as one of the pieces above without ##.",
           "empty": "Type two sentences and a word that occurs in both.",
           "not_here": "BERT-Tiny is not on your computer yet. The button in the worked example downloads it, 18 MB, "
                       "once."},
    "nl": {"title": "{word} in twee zinnen, block voor block",
           "pieces": "Stukjes van zin {n}",
           "head": ["Vectoren", "Cosine similarity van de twee"],
           "legend": "Blauw: de cosine similarity van de vectoren die BERT-Tiny het woord in de twee zinnen geeft, "
                     "vóór het eerste block en na elk block. Grijs: het woord dat je typte, tussen de stukjes.",
           "missing": "Het woord *{word}* is geen van de stukjes van zin {n}. De stukjes staan hierboven: typ een woord "
                      "dat daar zo tussen staat, in kleine letters.",
           "split": "BERT-Tiny knipt *{word}* in meer dan één stukje, dus het is in geen van de twee zinnen één stukje. "
                    "Probeer een woord dat heel blijft, zoals een van de stukjes hierboven zonder ##.",
           "empty": "Typ twee zinnen en een woord dat in allebei voorkomt.",
           "not_here": "BERT-Tiny staat nog niet op je computer. De knop bij het uitgewerkte voorbeeld downloadt het "
                       "model één keer, 18 MB."},
}


def explore_picture(first: str, second: str, word: str, lang: str) -> str:
    w = _EXPLORE_WORDS[lang]
    first, second, word = first[:LONGEST].strip(), second[:LONGEST].strip(), word[:LONGEST].strip().lower()
    title = w["title"].format(word=word or "…")
    if not (first and second and word):
        return yours.figure(title, _note(w["empty"]), alt=_bare(w["empty"]))
    if not ready():
        return yours.figure(title, _note(w["not_here"]), alt=_bare(w["not_here"]))
    from everything_ai import bert
    rows, chips, notes, alts = [], [], [], []
    found = []
    for n, sentence in enumerate((first, second), start=1):
        pieces = bert.pieces(sentence)
        chips.append(f'<div style="{_LABEL}">{html.escape(w["pieces"].format(n=n))}</div>'
                     + "".join(_chip(p, _CHIP_GRAY if p == word else _CHIP_FAINT) for p in pieces))
        if word not in pieces:
            notes.append(w["split"].format(word=word) if len(bert.tokenizer().word(word)) > 1
                         else w["missing"].format(word=word, n=n))
            continue
        i = pieces.index(word)
        found.append([layer[i] for layer in bert.layers(sentence)])
    body = "".join(chips)
    if len(found) == 2:
        names = _LAYER_NAMES[lang]
        for k in range(LAYERS):
            c = cosine(found[0][k], found[1][k])
            rows.append([html.escape(names[k]), f'{_bar(c, "var(--ea-pred)")} '
                         f'<span style="font-variant-numeric:tabular-nums">{_dec(c, lang)}</span>'])
            alts.append(f"{names[k]}: {_dec(c, lang)}")
        body += _table(rows, w["head"]) + _note(w["legend"])
    else:
        body += _note(" ".join(dict.fromkeys(notes)))
    return yours.figure(title, body, alt=f"{title}. " + "; ".join(alts) + ". " + _bare(" ".join(notes)))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["self_similarity", "guess_sense"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name, as the lesson's code writes it.
    The learner's function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Table(np.ndarray):
    """A small table or vector that a check message writes the way the lesson's code does: np.array([...]). The
    learner's code gets a plain copy."""

    def __repr__(self):
        def row(r):
            return "[" + ", ".join(repr(float(v)) for v in r) + "]"
        a = np.asarray(self)
        inner = row(a) if a.ndim == 1 else "[" + ", ".join(row(r) for r in a) + "]"
        return f"np.array({inner})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _t(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Table)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _shown(v) -> str:
    """What a function gave, as a message shows it: a number rounded to 4 decimals, a word with its quotes."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    value = _number(v)
    if value is None:
        return _short(v)
    text = f"{round(value, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _close(got, want) -> bool:
    """The same number within a small margin, also for the float32 numbers of BERT-Tiny."""
    if isinstance(want, str):
        return same_value(got, want)
    value = _number(got)
    return value is not None and abs(value - want) < 1e-4


def _matmul_mismatch(e) -> bool:
    return isinstance(e, ValueError) and ("mismatch" in str(e) or "not aligned" in str(e))


def _matmul_scalar(e) -> bool:
    return isinstance(e, ValueError) and "enough dimensions" in str(e)


def _row_as_index(e) -> bool:
    return isinstance(e, IndexError) and ("only integers" in str(e) or "arrays used as indices" in str(e))


# exercise 2: self_similarity. Three rows of 2 numbers, rows that point the same way at different lengths, two rows,
# four rows of 3, three rows of which two point opposite ways; then the 14 uses of bank in the river examples after the second
# block, and the 8 uses in the new money sentences before the first block.
SELF_CASES = [
    (_t([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),),
    (_t([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),),
    (_t([[1.0, 0.0], [1.0, 1.0]]),),
    (_t([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),),
    (_t([[3.0, 4.0], [-3.0, -4.0], [6.0, 8.0]]),),
    (Named('C.bank_rows(2, "river", "example")', bank_rows(2, "river", "example")),),
    (Named('C.bank_rows(0, "money", "new")', bank_rows(0, "money", "new")),),
]

_SS_HEAD = "import numpy as np\n\n\ndef self_similarity(rows):\n"
_SS_START = _SS_HEAD + "    ...\n"
_SS_BODY = ("    total = 0\n    count = 0\n    for i in range(len(rows)):\n        for j in range(i + 1, len(rows)):\n"
            "            a, b = rows[i], rows[j]\n            total += a @ b / (np.linalg.norm(a) * np.linalg.norm(b))\n"
            "            count += 1\n    return total / count\n")
_SS = _SS_HEAD + _SS_BODY
_SS_ALL = _SS.replace("range(i + 1, len(rows))", "range(len(rows))")
_SS_FROM_I = _SS.replace("range(i + 1, len(rows))", "range(i, len(rows))")
_SS_DOT = _SS.replace("a @ b / (np.linalg.norm(a) * np.linalg.norm(b))", "a @ b")
_SS_COUNT = _SS.replace("return total / count", "return total / len(rows)")
_SS_TOTAL = _SS.replace("return total / count", "return total")
_SS_ROWS = (_SS_HEAD + "    total = 0\n    count = 0\n    for i in rows:\n        for j in rows:\n"
            "            a, b = rows[i], rows[j]\n            total += a @ b / (np.linalg.norm(a) * np.linalg.norm(b))\n"
            "            count += 1\n    return total / count\n")
_SS_MATRIX = (_SS_HEAD + "    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)\n    n = len(rows)\n"
              "    return (np.sum(unit @ unit.T) - n) / (n * (n - 1))\n")

# exercise 3: guess_sense. Two plain cases with three rows of 2 numbers, so that a mean along the wrong axis stops at
# once; one where the dot product picks the long rows, one where the nearest single
# row has the other meaning, one where the nearest mean by distance has the other meaning, one where the mean of the
# cosines has the other meaning; rows of 3 numbers; then four new sentences of the course after the second block, the
# vaguest one among them.
_R2, _M2 = bank_rows(2, "river", "example"), bank_rows(2, "money", "example")
_NEW2 = bank_rows(2, None, "new")
_NEWS = sentences(None, "new")
EX_RIVER = Named('C.bank_rows(2, "river", "example")', _R2)
EX_MONEY = Named('C.bank_rows(2, "money", "example")', _M2)


def _new(i: int) -> Named:
    return Named(f'C.bank_rows(2, None, "new")[{i}]', _NEW2[i])


GUESS_CASES = [
    (_t([1.0, 0.0]), _t([[1.0, 0.2], [1.0, -0.2], [1.0, 0.0]]), _t([[0.0, 1.0], [0.2, 1.0], [-0.1, 1.0]])),
    (_t([0.1, 1.0]), _t([[1.0, 0.2], [1.0, -0.2], [1.0, 0.0]]), _t([[0.0, 1.0], [0.2, 1.0], [-0.1, 1.0]])),
    (_t([1.0, 1.0]), _t([[1.0, 0.8], [1.0, 1.2]]), _t([[10.0, 0.0], [10.0, 1.0]])),
    (_t([1.0, 0.1]), _t([[1.0, 1.0], [1.0, -1.0]]), _t([[1.0, 0.12], [-3.0, 1.0]])),
    (_t([2.0, 0.0]), _t([[10.0, 1.0], [10.0, -1.0]]), _t([[1.0, 1.0], [1.0, 1.5]])),
    (_t([1.0, 0.0]), _t([[10.0, 0.0], [-1.0, 0.5]]), _t([[1.0, 0.6], [1.0, -0.4]])),
    (_t([0.0, 1.0, 1.0]), _t([[0.0, 2.0, 1.0], [1.0, 1.0, 2.0], [0.0, 1.0, 1.0]]), _t([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])),
    (_new(0), EX_RIVER, EX_MONEY),
    (_new(3), EX_RIVER, EX_MONEY),
    (_new(4), EX_RIVER, EX_MONEY),
    (_new(15), EX_RIVER, EX_MONEY),
]

_GS_HEAD = "import numpy as np\n\n\ndef guess_sense(v, river, money):\n"
_GS_START = _GS_HEAD + "    ...\n"
_GS = (_GS_HEAD + "    r = np.mean(river, axis=0)\n    m = np.mean(money, axis=0)\n"
       "    to_river = v @ r / (np.linalg.norm(v) * np.linalg.norm(r))\n"
       "    to_money = v @ m / (np.linalg.norm(v) * np.linalg.norm(m))\n"
       "    if to_river > to_money:\n        return \"river\"\n    return \"money\"\n")
_GS_DOT = (_GS_HEAD + "    r = np.mean(river, axis=0)\n    m = np.mean(money, axis=0)\n"
           "    if v @ r > v @ m:\n        return \"river\"\n    return \"money\"\n")
_GS_SWAPPED = _GS.replace("if to_river > to_money:", "if to_river < to_money:")
_GS_AXIS1 = _GS.replace("axis=0", "axis=1")
_GS_NO_AXIS = _GS.replace(", axis=0", "")
_GS_NEAREST = (_GS_HEAD + "    best, answer = -2, None\n    for sense, rows in [(\"river\", river), (\"money\", money)]:\n"
               "        for row in rows:\n            c = v @ row / (np.linalg.norm(v) * np.linalg.norm(row))\n"
               "            if c > best:\n                best, answer = c, sense\n    return answer\n")
_GS_DISTANCE = (_GS_HEAD + "    r = np.mean(river, axis=0)\n    m = np.mean(money, axis=0)\n"
                "    if np.linalg.norm(v - r) < np.linalg.norm(v - m):\n        return \"river\"\n    return \"money\"\n")
_GS_MEAN_COS = (_GS_HEAD + "    to_river = np.mean(river @ v / (np.linalg.norm(river, axis=1) * np.linalg.norm(v)))\n"
                "    to_money = np.mean(money @ v / (np.linalg.norm(money, axis=1) * np.linalg.norm(v)))\n"
                "    if to_river > to_money:\n        return \"river\"\n    return \"money\"\n")
_GS_NUMBER = _GS.replace("    if to_river > to_money:\n        return \"river\"\n    return \"money\"\n",
                         "    return to_river > to_money\n")


def _not_a_word(got, text) -> bool:
    return not isinstance(got, str)


def _other_word(got, text) -> bool:
    return isinstance(got, str) and got not in SENSES


EXERCISES = {
    "self_similarity": FunctionBox(
        name="self_similarity",
        picture=self_picture,
        function="self_similarity",
        start=_SS_START,
        solution=_SS,
        visible=SELF_CASES[:2],
        hidden=SELF_CASES[2:],
        compare=_close,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("rows-as-index", error=_row_as_index,
                    en="With `{call}`, your function stops with an error: `for i in rows` gives the rows themselves, and a row cannot pick a row of the table. Which loop gives the numbers of the rows, 0, 1, 2 and so on?",
                    nl="Met `{call}` stopt je functie met een fout: `for i in rows` geeft de rijen zelf, en met een rij kun je geen rij van de tabel kiezen. Welke loop geeft de nummers van de rijen, 0, 1, 2 enzovoort?"),
            Mistake("self-pairs", wrong=_SS_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function also compares every row with itself, which always gives 1, so the mean comes out too high. Which values of `j` belong to a row other than row `i`, each pair once?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt ook elke rij met zichzelf, wat altijd 1 geeft, dus het gemiddelde komt te hoog uit. Bij welke waarden van `j` hoort een andere rij dan rij `i`, elk paar één keer?"),
            Mistake("self-from-i", wrong=_SS_FROM_I,
                    en="`{call}` gives {got}, and it should give {want}: your inner loop starts at `i`, so each row is also compared with itself once, which gives 1. Where does the inner loop have to start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je binnenste loop begint bij `i`, dus elke rij wordt ook één keer met zichzelf vergeleken, wat 1 geeft. Waar moet de binnenste loop beginnen?"),
            Mistake("dot", wrong=_SS_DOT,
                    en="`{call}` gives {got}, and it should give {want}: these are dot products, which grow with the lengths of the rows. A cosine similarity looks only at the directions of two rows, not at their lengths: by what do you divide each dot product?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit zijn dot products, en dot products groeien mee met de lengte van de rijen. Een cosine similarity kijkt alleen naar de richting van twee rijen, niet naar de lengte: waardoor deel je elk dot product?"),
            Mistake("count", wrong=_SS_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of rows, but the mean is over the pairs: 2 rows make 1 pair, and 4 rows make 6. What do you divide by?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal rijen, maar het gemiddelde gaat over de paren: 2 rijen maken 1 paar, 4 rijen 6 paren. Waardoor deel je?"),
            Mistake("total", wrong=_SS_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum of the cosine similarities of all pairs. A mean divides the sum by something: by what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de som van de cosine similarities van alle paren. Een gemiddelde deelt de som ergens door: waardoor?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the mean cosine similarity of every two different rows of `rows`.",
                      "nl": "Vervang de `...`: geef de gemiddelde cosine similarity van elke twee verschillende rijen van `rows` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Take every pair of two different rows once, compute their cosine similarity, and give back the mean of those numbers.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neem elk paar van twee verschillende rijen één keer, reken de cosine similarity van elk paar uit, en geef het gemiddelde van die getallen terug."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for other tables too, with two rows, or rows of three numbers?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor andere tabellen, met twee rijen, of rijen van drie getallen?"},
            "correct": {"en": "Your `self_similarity` is right in all {n} cases, also for the uses of bank. Look at the picture: how alike the 44 uses of bank are, in GloVe and in BERT-Tiny.",
                        "nl": "Je `self_similarity` klopt in alle {n} gevallen, ook voor bank in de zinnen van de course. Kijk naar de afbeelding: hoe sterk bank in de 44 zinnen op zichzelf lijkt, in GloVe en in BERT-Tiny."},
        },
    ),
    "guess_sense": FunctionBox(
        name="guess_sense",
        picture=guess_picture,
        function="guess_sense",
        start=_GS_START,
        solution=_GS,
        visible=GUESS_CASES[:2],
        hidden=GUESS_CASES[2:],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("axis", error=_matmul_mismatch,
                    en="With `{call}`, your function stops with an error: your mean of the rows has another number of values than `v`. The mean row should have one number per column, as `v` has. Which axis gives that?",
                    nl="Met `{call}` stopt je functie met een fout: je gemiddelde van de rijen heeft een ander aantal waarden dan `v`. De gemiddelde rij moet één getal per kolom hebben, zoals `v`. Welke axis geeft dat?"),
            Mistake("no-axis", error=_matmul_scalar,
                    en="With `{call}`, your function stops with an error: `np.mean` without an axis gives one number for the whole table, and `@` needs a row. Which axis gives the mean row, one number per column?",
                    nl="Met `{call}` stopt je functie met een fout: `np.mean` zonder axis geeft één getal voor de hele tabel, en `@` heeft een rij nodig. Welke axis geeft de gemiddelde rij, één getal per kolom?"),
            Mistake("not-a-word", matches=_not_a_word,
                    en="`{call}` gives {got}, and it should give {want}: the function gives back a word, `\"river\"` or `\"money\"`. Which word belongs to the higher cosine similarity?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de functie geeft een woord terug, `\"river\"` of `\"money\"`. Welk woord hoort bij de hoogste cosine similarity?"),
            Mistake("other-word", matches=_other_word,
                    en="`{call}` gives {got}, and it should give {want}: the function gives back exactly `\"river\"` or `\"money\"`, in small letters, as the course's pictures and counts read them. Which of the two words is it here?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de functie geeft precies `\"river\"` of `\"money\"` terug, in kleine letters en in het Engels, zoals de afbeeldingen en tellingen van de course de woorden lezen. Welk van de twee woorden is het hier?"),
            Mistake("swapped", wrong=_GS_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the meaning whose mean row is the farther one. Which meaning goes with the higher cosine similarity?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de betekenis waarvan de gemiddelde rij het verst weg ligt. Welke betekenis hoort bij de hoogste cosine similarity?"),
            Mistake("dot", wrong=_GS_DOT,
                    en="`{call}` gives {got}, and it should give {want}: your function compares dot products, and a long mean row wins those, whatever its direction. How do you compare only the directions?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt dot products, en bij dot products wint een lange gemiddelde rij, welke richting die rij ook heeft. Hoe vergelijk je alleen de richtingen?"),
            Mistake("nearest", wrong=_GS_NEAREST,
                    en="`{call}` gives {got}, and it should give {want}: your function looks for the single example nearest to `v`, and here that one is an exception. The guess compares `v` with the mean row of each meaning: how do you make that row?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zoekt het ene voorbeeld dat het dichtst bij `v` ligt, en hier is dat een uitzondering. De gok vergelijkt `v` met de gemiddelde rij van elke betekenis: hoe maak je die rij?"),
            Mistake("distance", wrong=_GS_DISTANCE,
                    en="`{call}` gives {got}, and it should give {want}: your function compares distances, and a short `v` lies near any short mean row, whatever its direction. Which measure looks only at the direction?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt afstanden, en een korte `v` ligt dicht bij elke korte gemiddelde rij, welke richting die rij ook heeft. Welke maat kijkt alleen naar de richting?"),
            Mistake("mean-of-cosines", wrong=_GS_MEAN_COS,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the mean of the cosine similarities with the examples, and here one example of `river` points far away from `v`. The guess first takes the mean of the rows, and then one cosine similarity: in which order?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gemiddelde van de cosine similarities met de voorbeelden, en hier wijst één voorbeeld van `river` ver weg van `v`. De gok neemt eerst het gemiddelde van de rijen, en dan één cosine similarity: in welke volgorde?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: compare `v` with the mean row of `river` and with the mean row of `money`, and give back `\"river\"` or `\"money\"`.",
                      "nl": "Vervang de `...`: vergelijk `v` met de gemiddelde rij van `river` en met de gemiddelde rij van `money`, en geef `\"river\"` of `\"money\"` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Take the mean row of `river` and of `money`, and give back the meaning whose mean row has the higher cosine similarity with `v`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neem de gemiddelde rij van `river` en van `money`, en geef de betekenis terug waarvan de gemiddelde rij de hoogste cosine similarity met `v` heeft."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function compare the directions of `v` and the two mean rows, whatever their lengths?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Vergelijkt je functie de richtingen van `v` en de twee gemiddelde rijen, hoe lang `v` en de rijen ook zijn?"},
            "correct": {"en": "Your `guess_sense` is right in all {n} cases, also for bank in new sentences of the course. Look at the picture: your guesses for the 16 new sentences, with GloVe and with BERT-Tiny.",
                        "nl": "Je `guess_sense` klopt in alle {n} gevallen, ook voor bank in nieuwe zinnen van de course. Kijk naar de afbeelding: je gokken voor de 16 nieuwe zinnen, met GloVe en met BERT-Tiny."},
        },
    ),
}

CASES = [
    ("self_similarity", _SS_START, "empty", None),
    ("self_similarity", _SS, "correct", None),
    ("self_similarity", _SS_MATRIX, "correct", None),
    ("self_similarity", _SS.replace("range(i + 1, len(rows))", "range(len(rows))").replace(
        "            a, b = rows[i], rows[j]\n", "            if i == j:\n                continue\n            a, b = rows[i], rows[j]\n"),
     "correct", None),
    ("self_similarity", _SS_ALL, "wrong", "self-pairs"),
    ("self_similarity", _SS_HEAD + "    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)\n    return np.mean(unit @ unit.T)\n",
     "wrong", "self-pairs"),
    ("self_similarity", _SS_FROM_I, "wrong", "self-from-i"),
    ("self_similarity", _SS_DOT, "wrong", "dot"),
    ("self_similarity", _SS_COUNT, "wrong", "count"),
    ("self_similarity", _SS_TOTAL, "wrong", "total"),
    ("self_similarity", _SS_ROWS, "error", "rows-as-index"),
    ("self_similarity", _SS_HEAD + "    ...\n    return 1\n", "wrong", "dots-left"),
    ("guess_sense", _GS_START, "empty", None),
    ("guess_sense", _GS, "correct", None),
    ("guess_sense", _GS_HEAD + "    r = np.sum(river, axis=0)\n    m = np.sum(money, axis=0)\n"
                    "    if v @ r / np.linalg.norm(r) > v @ m / np.linalg.norm(m):\n        return \"river\"\n    return \"money\"\n",
     "correct", None),
    ("guess_sense", _GS_DOT, "wrong", "dot"),
    ("guess_sense", _GS_SWAPPED, "wrong", "swapped"),
    ("guess_sense", _GS_AXIS1, "error", "axis"),
    ("guess_sense", _GS_NO_AXIS, "error", "no-axis"),
    ("guess_sense", _GS_NEAREST, "wrong", "nearest"),
    ("guess_sense", _GS_DISTANCE, "wrong", "distance"),
    ("guess_sense", _GS_MEAN_COS, "wrong", "mean-of-cosines"),
    ("guess_sense", _GS_NUMBER, "wrong", "not-a-word"),
    ("guess_sense", _GS.replace("return \"river\"", "return \"River\""), "wrong", "other-word"),
    ("guess_sense", _GS_HEAD + "    ...\n    return \"river\"\n", "wrong", "dots-left"),
]
