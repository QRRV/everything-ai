# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-sentence-embeddings: which pairs of sentences get exactly the same mean of their word
vectors (a sort), the mean of the GloVe vectors of the words of a sentence (mean_vector), and the mean of the rows of a
transformer's table where the mask is 1 (mean_pooling); with the pictures of what each gives on the course's 16 pairs
of sentences, and two explores.

The course wrote 16 pairs of English sentences in four kinds (data/course-sentence-pairs.tsv): the same words in
another order, one sentence with *not* added, the same words with another meaning, and other words with the same
meaning. What BERT-Tiny gives every piece of those 32 sentences after its second block, in one batch with [PAD] places
and their mask, and the embedding Granite Embedding gives each sentence, the course ships
(data/course-sentence-vectors.npz, made by data/tools/make_sentence_vectors.py), so that the exercises and their
pictures work without either model. Granite itself (223 MB, everything_ai.embed) the learner downloads once for the last
explore. GloVe comes from everything_ai.wordvectors. Every number in the text and the pictures is computed by the
course (tests/test_nlp_sentence_embeddings.py computes them again)."""
import contextlib
import csv
import html
import io
import math
import re
import warnings

import numpy as np

from everything_ai import lesson as lesson_helpers, paths, wordvectors, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-sentence-embeddings"
REPO = "ibm-granite/granite-embedding-107m-multilingual"   # everything_ai.embed.REPO: that module imports PyTorch
GRANITE_FILES = [(REPO, "model.safetensors"), (REPO, "tokenizer.json")]
PAIRS_FILE = paths.ROOT / "data" / "course-sentence-pairs.tsv"
VECTORS = paths.ROOT / "data" / "course-sentence-vectors.npz"
KINDS = ("order", "negation", "shared", "paraphrase")

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The last part to try yourself downloads Granite Embedding, {download_size(GRANITE_FILES, 'en')}, once from Hugging Face, when you click its button in the lesson; the exercises and the rest of the lesson work without it.",
    "nl": f"Werkt op elke computer. Het laatste onderdeel om zelf te proberen downloadt één keer Granite Embedding, {download_size(GRANITE_FILES, 'nl')}, van Hugging Face, als je in de les op de knop ervoor klikt; de oefeningen en de rest van de les werken zonder.",
}

# --- the vectors -----------------------------------------------------------------------------------------------------------


def glove() -> tuple:
    """(words, E): GloVe's 50,000 words as a new list, from the most to the least frequent, and their vectors, an array
    of 50,000 rows of 50 numbers (read-only): row i is the vector of words[i]."""
    return wordvectors.glove()


def rows(words) -> dict:
    """A dictionary from each word to the number of its row."""
    return wordvectors.rows(words)


def _glove() -> tuple:
    """(E, row): one copy for the checks and the pictures."""
    def make():
        words, E = wordvectors.glove()
        return E, wordvectors.rows(words)
    return kept(f"{_KEY}:glove", make)


def _data() -> tuple:
    """(pairs, granite, tiny, mask): the 16 pairs as (kind, first, second), and the arrays of the course's run, all
    read-only."""
    def make():
        with open(PAIRS_FILE, encoding="utf-8", newline="") as f:
            pairs = tuple((r["kind"], r["first"], r["second"]) for r in csv.DictReader(f, delimiter="\t"))
        with np.load(VECTORS, allow_pickle=False) as z:
            granite, tiny, mask = z["granite"].astype(np.float32), z["tiny"].astype(np.float32), z["mask"].astype(int)
        for a in (granite, tiny, mask):
            a.flags.writeable = False
        return pairs, granite, tiny, mask
    return kept(f"{_KEY}:data", make)


def pairs() -> list:
    """The course's 16 pairs of sentences, as a new list of (kind, first, second), in the order of the file."""
    return list(_data()[0])


def tiny_tokens(i: int) -> np.ndarray:
    """What BERT-Tiny gives every place of sentence i of the 32 after its second block, as a new table of 10 rows of 128
    numbers: [CLS], the pieces, [SEP], then [PAD] places up to 10. Sentence 2p is the first of pair p, counted from 0,
    and 2p + 1 the second."""
    return np.array(_data()[2][i])


def tiny_mask(i: int) -> np.ndarray:
    """The mask of sentence i of the 32, as a new column of 10 rows: 1 for a real piece, 0 for a [PAD] place."""
    return np.array(_data()[3][i]).reshape(-1, 1)


def granite_vector(i: int) -> np.ndarray:
    """Granite's embedding of sentence i of the 32: 384 numbers, of length 1."""
    return np.array(_data()[1][i])


def words_of(text) -> list:
    """The words of a typed sentence as GloVe writes them: in lowercase, without punctuation, an apostrophe inside a
    word kept."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", str(text).lower())


# --- the course's own versions of the exercises ------------------------------------------------------------------------


def mean_vector(words, E, row):
    total = np.zeros(E.shape[1])
    count = 0
    for w in words:
        if w in row:
            total = total + E[row[w]]
            count = count + 1
    if count == 0:
        return None
    return total / count


def mean_pooling(tokens, mask):
    return (tokens * mask).sum(axis=0) / mask.sum()


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def glove_cosine(first: str, second: str) -> float:
    E, row = _glove()
    return cosine(mean_vector(first.split(), E, row), mean_vector(second.split(), E, row))


def pair_numbers() -> list:
    """(kind, first, second, GloVe, BERT-Tiny, Granite) for each of the 16 pairs: the cosine similarity of the means
    of the GloVe vectors, of the mean pooling of BERT-Tiny's table, and of Granite's embeddings."""
    ps, granite, tiny, mask = _data()
    out = []
    for p, (kind, a, b) in enumerate(ps):
        i, j = 2 * p, 2 * p + 1
        out.append((kind, a, b, glove_cosine(a, b),
                    cosine(mean_pooling(tiny[i], mask[i][:, None]), mean_pooling(tiny[j], mask[j][:, None])),
                    float(granite[i] @ granite[j])))
    return out


# --- the look of the pictures -------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_SMALL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"


def _marked(text: str) -> str:
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{html.escape(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _plain(text) -> str:
    """Typed text for a note: without asterisks and backticks, which the note reads as italics and code."""
    return str(text).replace("*", "").replace("`", "")


def _dec(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = (), shade: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; the rows in
    `shade` get a gray band and span the whole table."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = []
    for i, row in enumerate(rows_):
        if i in shade:
            body.append(f'<tr style="background:var(--ea-data-wash)"><td colspan="{len(head)}" style="{_CELL}">'
                        f"<b>{row[0]}</b></td></tr>")
        else:
            body.append("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                         for j, c in enumerate(row)) + "</tr>")
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def _bar(value: float, style: str, scale: float = 110) -> str:
    width = max(0.0, min(1.0, value)) * scale
    return f'<span style="{_BAR};width:{width:.0f}px;background:{style}"></span>'


def _short(value, limit: int = 60) -> str:
    try:
        text = repr(value)
    except Exception:   # noqa: BLE001   a value whose repr stops with an error
        text = "?"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _finite(v) -> bool:
    try:
        return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(float(v))
    except (OverflowError, ValueError, TypeError):
        return False


def _number(x, lang: str, color: str) -> str:
    """A cosine similarity as a bar and its number."""
    return (f'<span style="white-space:nowrap">{_bar(x, color)} '
            f'<span style="font-variant-numeric:tabular-nums">{_dec(x, lang)}</span></span>')


def _spark(vector, scale: float, style: str, width: int = 4) -> str:
    """A vector as a row of thin bars, up for a positive number and down for a negative one, all on one scale."""
    h = 34
    mid = h / 2
    bars = []
    for k, v in enumerate(vector):
        y = max(-1.0, min(1.0, float(v) / scale)) * (mid - 1)
        top, size = (mid - y, y) if y > 0 else (mid, -y)
        bars.append(f'<rect x="{k * width}" y="{top:.1f}" width="{width - 1}" height="{max(size, 0.6):.1f}"></rect>')
    w = len(vector) * width
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-hidden="true" '
            f'style="vertical-align:middle;fill:{style}"><line x1="0" x2="{w}" y1="{mid}" y2="{mid}" '
            f'style="stroke:var(--ea-line);stroke-width:1"></line>{"".join(bars)}</svg>')


# --- the idea's pictures: a sentence as the mean of its words, and four pairs --------------------------------------------

IDEA_SENTENCES = ("the dog bit the man", "the man bit the dog")
IDEA_PAIRS = [("order", "the dog bit the man", "the man bit the dog"),
              ("negation", "the film was good", "the film was not good"),
              ("paraphrase", "the film was good", "i enjoyed the movie"),
              ("unrelated", "the film was good", "the train leaves at nine")]

_IDEA_WORDS = {
    "en": {"title": "Two sentences as the mean of their word vectors",
           "mean": "mean",
           "legend": "Each row is a vector of GloVe: 50 thin bars, up for a positive number and down for a negative one; "
                     "the few largest numbers are cut off at the edge of the row. Gray: the vector of each word; dark: the mean of the five, the sentence's vector. Both sentences "
                     "have the same five words, so their means are the same, bar for bar.",
           "pairs_title": "The mean of GloVe's word vectors, for four pairs of sentences",
           "head": ["First sentence", "Second sentence", "Cosine similarity"],
           "kinds": {"order": "other order", "negation": "not added", "paraphrase": "same meaning",
                     "unrelated": "nothing in common"},
           "pairs_legend": "Gray: the cosine similarity of the means of the GloVe vectors of the two sentences. "
                           "The scale runs from 0 to 1."},
    "nl": {"title": "Twee zinnen als het gemiddelde van de woordvectoren",
           "mean": "gemiddelde",
           "legend": "Elke rij is een vector van GloVe: 50 dunne staafjes, omhoog voor een positief getal en omlaag "
                     "voor een negatief; de paar grootste getallen worden afgekapt aan de rand van de rij. Grijs: de vector van elk woord; donker: het gemiddelde van de vijf, de vector "
                     "van de zin. Beide zinnen hebben dezelfde vijf woorden, dus de gemiddelden zijn gelijk, staafje "
                     "voor staafje.",
           "pairs_title": "Het gemiddelde van de woordvectoren van GloVe, voor vier paren zinnen",
           "head": ["Eerste zin", "Tweede zin", "Cosine similarity"],
           "kinds": {"order": "andere volgorde", "negation": "not erbij", "paraphrase": "zelfde betekenis",
                     "unrelated": "niets gemeen"},
           "pairs_legend": "Grijs: de cosine similarity van de gemiddelden van de GloVe-vectoren van de twee zinnen. "
                           "De schaal loopt van 0 tot 1."},
}


def idea_mean_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    E, row = _glove()
    blocks, alts = [], []
    vectors = [E[row[x]] for s in IDEA_SENTENCES for x in s.split()]
    scale = float(np.percentile(np.abs(vectors), 95))   # a few large numbers are cut off, so that the rest show
    for s in IDEA_SENTENCES:
        lines = []
        for x in s.split():
            lines.append(f'<div style="display:flex;align-items:center;gap:0.5rem"><span style="{_SMALL};width:4.5rem">'
                         f"<i>{html.escape(x)}</i></span>{_spark(E[row[x]], scale, 'var(--ea-data)')}</div>")
        m = mean_vector(s.split(), E, row)
        lines.append(f'<div style="display:flex;align-items:center;gap:0.5rem;border-top:1px solid var(--ea-line);'
                     f'padding-top:0.15rem"><span style="{_SMALL};width:4.5rem;font-weight:600">{w["mean"]}</span>'
                     f"{_spark(m, scale, 'var(--ea-ink)')}</div>")
        blocks.append(f'<div style="flex:1 1 15rem;min-width:0">{_label(s)}{"".join(lines)}</div>')
        alts.append(f"{s}: " + ", ".join(_dec(v, lang, 2) for v in m[:5]) + " …")
    body = f'<div style="display:flex;flex-wrap:wrap;gap:1rem">{"".join(blocks)}</div>'
    return yours.figure(w["title"], body + _note(w["legend"]), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(w["legend"]))


def idea_pairs_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    rows_, alts = [], []
    for kind, a, b in IDEA_PAIRS:
        c = glove_cosine(a, b)
        rows_.append([f"<i>{html.escape(a)}</i>", f"<i>{html.escape(b)}</i> <span style=\"{_SMALL}\">({w['kinds'][kind]})</span>",
                      _number(c, lang, "var(--ea-data)")])
        alts.append(f"{a} / {b}: {_dec(c, lang)}")
    return yours.figure(w["pairs_title"], _table(rows_, w["head"]) + _note(w["pairs_legend"]),
                        alt=f'{w["pairs_title"]}. ' + "; ".join(alts) + ". " + _bare(w["pairs_legend"]))


# --- running the learner's function in a picture ------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 1   # what the learner's function gets per call in a picture


def _shared(a: np.ndarray) -> bool:
    """GloVe's table: read-only and large, and in the lesson too the learner's function gets it as it is. Every other
    array goes to the learner's function as a new copy, as `C.tiny_tokens` gives one."""
    return not a.flags.writeable and a.size > 100_000


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit; arrays and lists are passed as copies. _STOPPED when it stops with
    an error, and the Timeout passes on."""
    def copy(a):
        if isinstance(a, np.ndarray):
            return a if _shared(a) else a.copy()
        if isinstance(a, list):
            return list(a)
        if isinstance(a, dict):
            return dict(a)
        return a
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # NumPy's "invalid value in divide": the picture says it
            return with_time_limit(lambda: fn(*(copy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _zeros(got, length: int) -> bool:
    """A vector of `length` zeros: it has no direction, so no cosine similarity."""
    try:
        v = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    return v.shape == (length,) and not np.any(v)


def _vector(got, length: int):
    """The learner's result as a vector of `length` finite numbers, or None when it is not one."""
    try:
        v = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if v.shape != (length,) or not np.all(np.isfinite(v)) or not np.any(v):
        return None
    return v


_KIND_WORDS = {
    "en": {"order": "The same words in another order", "negation": "One sentence with not added",
           "shared": "The same words, another meaning", "paraphrase": "Other words, the same meaning"},
    "nl": {"order": "Dezelfde woorden in een andere volgorde", "negation": "Eén zin met not erbij",
           "shared": "Dezelfde woorden, een andere betekenis", "paraphrase": "Andere woorden, dezelfde betekenis"},
}


def _pair_cell(a: str, b: str) -> str:
    return f"<i>{html.escape(a)}</i><br><i>{html.escape(b)}</i>"


# --- the picture under exercise 2: your mean_vector on the 16 pairs ------------------------------------------------------

# a sentence with a word GloVe lacks at the end, one that starts with such a word, and one with only such a word
EXTRA = ("the shop opens at nine o'clock", "selfies are everywhere now", "o'clock")

_MEAN_WORDS = {
    "en": {"title": "Your mean_vector on the course's 16 pairs of sentences",
           "head": ["Sentences", "Cosine similarity of your vectors"],
           "kind_mean": "On average: {c}",
           "extra_head": ["Three more sentences", "What your function gives"],
           "extra": {"the shop opens at nine o'clock": "GloVe lacks o'clock", "selfies are everywhere now": "GloVe lacks selfies",
                     "o'clock": "GloVe lacks its only word"},
           "vector": "a vector of {n} numbers: {first}, …",
           "stops": "it stops with an error",
           "still": "it is still running after {s} second",
           "legend": "Blue: the cosine similarity of the vectors your function gives the two sentences, with GloVe, and "
                     "what it gives three more sentences; purple where a vector of yours differs from the mean of the "
                     "words that GloVe has. On the gray bands: the four kinds of pairs.",
           "differs": "For *{s}*, your function gives a vector that starts with {got}; the mean of its word vectors "
                      "starts with {want}.",
           "error": "Your function stops with an error on `mean_vector({words}, E, row)`.",
           "slow": "Your function is still running after {s} second with `mean_vector({words}, E, row)`.",
           "odd": "With `mean_vector({words}, E, row)`, your function gives back {got}, and not a vector of 50 numbers.",
           "zeros": "With `mean_vector({words}, E, row)`, your function gives back a vector of only zeros: a vector without a "
                    "direction, so without a cosine similarity."},
    "nl": {"title": "Jouw mean_vector op de 16 paren zinnen van de course",
           "head": ["Zinnen", "Cosine similarity van jouw vectoren"],
           "kind_mean": "Gemiddeld: {c}",
           "extra_head": ["Nog drie zinnen", "Wat je functie geeft"],
           "extra": {"the shop opens at nine o'clock": "GloVe kent o'clock niet", "selfies are everywhere now": "GloVe kent selfies niet",
                     "o'clock": "GloVe kent het enige woord niet"},
           "vector": "een vector van {n} getallen: {first}, …",
           "stops": "de functie stopt met een fout",
           "still": "de functie is na {s} seconde nog bezig",
           "legend": "Blauw: de cosine similarity van de vectoren die je functie de twee zinnen geeft, met GloVe, en wat "
                     "de functie nog drie zinnen geeft; paars waar een vector van jou verschilt van het gemiddelde van de "
                     "woorden die GloVe kent. Op de grijze banden: de vier soorten paren.",
           "differs": "Voor *{s}* geeft je functie een vector die begint met {got}; het gemiddelde van de woordvectoren "
                      "begint met {want}.",
           "error": "Je functie stopt met een fout op `mean_vector({words}, E, row)`.",
           "slow": "Je functie is na {s} seconde nog bezig met `mean_vector({words}, E, row)`.",
           "odd": "Met `mean_vector({words}, E, row)` geeft je functie {got} terug, en geen vector van 50 getallen.",
           "zeros": "Met `mean_vector({words}, E, row)` geeft je functie een vector met alleen nullen terug: een vector "
                    "zonder richting, dus zonder cosine similarity."},
}


def _first(v, lang: str, k: int = 3) -> str:
    return ", ".join(_dec(float(x), lang) for x in np.asarray(v, dtype=float)[:k])


def _brief(got, lang: str, words: dict) -> str:
    """What a function gave, in a few words: None, the start of a vector, or its repr cut short."""
    if got is None:
        return "None"
    try:
        v = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _short(got, 40)
    if v.ndim == 1 and 0 < v.size <= 10_000:
        return words["vector"].format(n=v.size, first=_first(v, lang))
    return _short(got, 40)


def mean_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _MEAN_WORDS[lang]
    E, row = _glove()
    rows_, shade, alts, notes = [], [], [], []
    cache, right = {}, {}
    stopped = differs = False   # differs: the note on the first vector that differs is there
    for kind in KINDS:
        shade.append(len(rows_))
        rows_.append([html.escape(_KIND_WORDS[lang][kind])])
        found = []
        for k, a, b in _data()[0]:
            if k != kind:
                continue
            got = []
            for s in (a, b):
                if s not in cache:
                    words = s.split()
                    try:
                        out = _call(fn, words, E, row)
                    except Timeout:
                        notes.append(w["slow"].format(words=_code(words), s=PICTURE_SECONDS))
                        stopped = True
                        break
                    if out is _STOPPED:
                        notes.append(w["error"].format(words=_code(words)))
                        stopped = True
                        break
                    v = _vector(out, E.shape[1])
                    if v is None:
                        key = "zeros" if _zeros(out, E.shape[1]) else "odd"
                        notes.append(w[key].format(words=_code(words), got=_plain(_short(out, 40))))
                        stopped = True
                        break
                    cache[s] = v
                    want = mean_vector(words, E, row)
                    right[s] = bool(np.allclose(v, want, rtol=0, atol=1e-5))
                    if not right[s] and not differs:
                        notes.append(w["differs"].format(s=_plain(s), got=_first(v, lang), want=_first(want, lang)))
                        differs = True
                got.append(cache[s])
            if stopped:
                break
            c = cosine(got[0], got[1])
            good = right[a] and right[b]
            rows_.append([_pair_cell(a, b), _number(c, lang, "var(--ea-pred)" if good else "var(--ea-loss)")])
            found.append(c)
            alts.append(f"{a} / {b}: {_dec(c, lang)}")
        if stopped:
            break
        rows_.append([f'<span style="{_SMALL}">{html.escape(w["kind_mean"].format(c=_dec(float(np.mean(found)), lang)))}</span>', ""])
    extra = []
    for s in EXTRA:
        words = s.split()
        want = mean_vector(words, E, row)
        try:
            out = _call(fn, words, E, row)
        except Timeout:
            text, good = w["still"].format(s=PICTURE_SECONDS), False
        else:
            if out is _STOPPED:
                text, good = w["stops"], False
            else:
                text, good = _brief(out, lang, w), _same_vector(out, want)
        chip = _chip(text, _CHIP_PRED if good else _CHIP_PURPLE)
        extra.append([f"<i>{html.escape(s)}</i> <span style=\"{_SMALL}\">({html.escape(w['extra'][s])})</span>", chip])
        alts.append(f"{s}: {text}")
    if stopped and shade[-1] == len(rows_) - 1:   # a gray band without a pair under it
        rows_.pop()
        shade.pop()
    text_ = " ".join(notes + [w["legend"]])
    body = (_table(rows_, w["head"], shade=tuple(shade)) if rows_ else "") + _table(extra, w["extra_head"])
    return yours.figure(w["title"], body + _note(text_), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text_))


# --- the picture under exercise 3: three ways to a sentence vector --------------------------------------------------------

_POOL_WORDS = {
    "en": {"title": "Three sentence vectors on the course's 16 pairs",
           "head": ["Sentences", "Mean of GloVe", "BERT-Tiny, your mean pooling", "Granite"],
           "kind_mean": "On average",
           "legend": "Each column is the cosine similarity of the two sentences' vectors. Gray: the mean of the GloVe "
                     "vectors of the words, and Granite's embeddings, which the course recorded. Blue: your "
                     "mean_pooling of BERT-Tiny's table after its second block, with the [PAD] places in it; purple where "
                     "a vector of yours differs from the mean of the real rows. On the gray bands: the four kinds of "
                     "pairs.",
           "differs": "For sentence {i}, *{s}*, your function gives a vector that starts with {got}; the mean of its "
                      "real rows starts with {want}.",
           "error": "Your function stops with an error on `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`.",
           "slow": "Your function is still running after {s} second with `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`.",
           "odd": "With `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`, your function gives back {got}, and not a "
                  "vector of 128 numbers.",
           "zeros": "With `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`, your function gives back a vector of only "
                    "zeros: a vector without a direction, so without a cosine similarity."},
    "nl": {"title": "Drie zinsvectoren op de 16 paren zinnen van de course",
           "head": ["Zinnen", "Gemiddelde van GloVe", "BERT-Tiny, jouw mean pooling", "Granite"],
           "kind_mean": "Gemiddeld",
           "legend": "Elke kolom is de cosine similarity van de vectoren van de twee zinnen. Grijs: het gemiddelde van "
                     "de GloVe-vectoren van de woorden, en de embeddings van Granite, die de course vastlegde. Blauw: "
                     "jouw mean_pooling van de tabel van BERT-Tiny na het tweede block, met de [PAD]-plekken erin; "
                     "paars waar een vector van jou verschilt van het gemiddelde van de echte rijen. Op de grijze "
                     "banden: de vier soorten paren.",
           "differs": "Voor zin {i}, *{s}*, geeft je functie een vector die begint met {got}; het gemiddelde van de "
                      "echte rijen begint met {want}.",
           "error": "Je functie stopt met een fout op `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`.",
           "slow": "Je functie is na {s} seconde nog bezig met `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))`.",
           "odd": "Met `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))` geeft je functie {got} terug, en geen "
                  "vector van 128 getallen.",
           "zeros": "Met `mean_pooling(C.tiny_tokens({i}), C.tiny_mask({i}))` geeft je functie een vector met alleen "
                    "nullen terug: een vector zonder richting, dus zonder cosine similarity."},
}


def pooling_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _POOL_WORDS[lang]
    ps, granite, tiny, mask = _data()
    sentences = [s for _, a, b in ps for s in (a, b)]
    numbers = pair_numbers()
    mine, right, notes = {}, {}, []
    differs = False   # the note on the first vector that differs is there
    for i in range(len(tiny)):
        try:
            out = _call(fn, tiny[i], mask[i][:, None])
        except Timeout:
            notes.append(w["slow"].format(i=i, s=PICTURE_SECONDS))
            break
        if out is _STOPPED:
            notes.append(w["error"].format(i=i))
            break
        v = _vector(out, tiny.shape[2])
        if v is None:
            key = "zeros" if _zeros(out, tiny.shape[2]) else "odd"
            notes.append(w[key].format(i=i, got=_plain(_short(out, 40))))
            break
        mine[i] = v
        want = mean_pooling(tiny[i], mask[i][:, None])
        right[i] = bool(np.allclose(v, want, rtol=0, atol=1e-5))
        if not right[i] and not differs:
            notes.append(w["differs"].format(i=i, s=_plain(sentences[i]), got=_first(v, lang), want=_first(want, lang)))
            differs = True
    rows_, shade, alts = [], [], []
    for kind in KINDS:
        shade.append(len(rows_))
        rows_.append([html.escape(_KIND_WORDS[lang][kind])])
        sums = [[], [], []]
        for p, (k, a, b, g, t, gr) in enumerate(numbers):
            if k != kind:
                continue
            if 2 * p in mine and 2 * p + 1 in mine:
                c = cosine(mine[2 * p], mine[2 * p + 1])
                good = right[2 * p] and right[2 * p + 1]
                cell = _number(c, lang, "var(--ea-pred)" if good else "var(--ea-loss)")
                sums[1].append(c)
            else:
                c, cell = None, ""
            sums[0].append(g)
            sums[2].append(gr)
            rows_.append([_pair_cell(a, b), _number(g, lang, "var(--ea-data)"), cell, _number(gr, lang, "var(--ea-data)")])
            alts.append(f"{a} / {b}: {_dec(g, lang)}, {_dec(c, lang) if c is not None else '-'}, {_dec(gr, lang)}")
        avg = [f'<span style="{_SMALL}">{_dec(float(np.mean(s)), lang)}</span>' if s else "" for s in sums]
        rows_.append([f'<span style="{_SMALL}">{html.escape(w["kind_mean"])}</span>', *avg])
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], _table(rows_, w["head"], shade=tuple(shade)) + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- try it: two sentences of your own, with GloVe ------------------------------------------------------------------------

LONGEST = 300   # characters of a typed sentence that count

_GLOVE_WORDS = {
    "en": {"title": "Your two sentences as the mean of their GloVe vectors",
           "used": "Words in GloVe:",
           "left": "Left out, because GloVe lacks them: {words}.",
           "none": "GloVe knows none of the words of *{s}*. Type an English sentence.",
           "empty": "Type two sentences.",
           "cosine": "Cosine similarity of the two means: {c}.",
           "same": "The two sentences have the same words, each as many times: their means are the same.",
           "legend": "Each row is the mean of the GloVe vectors of a sentence's words: 50 thin bars, up for a positive "
                     "number and down for a negative one; the few largest numbers are cut off at the edge of the row. Gray: GloVe's "
                     "words and their mean."},
    "nl": {"title": "Je twee zinnen als het gemiddelde van de GloVe-vectoren",
           "used": "Woorden in GloVe:",
           "left": "Weggelaten, omdat GloVe deze woorden niet kent: {words}.",
           "none": "GloVe kent geen enkel woord van *{s}*. Typ een Engelse zin.",
           "empty": "Typ twee zinnen.",
           "cosine": "Cosine similarity van de twee gemiddelden: {c}.",
           "same": "De twee zinnen hebben dezelfde woorden, elk even vaak: de gemiddelden zijn gelijk.",
           "legend": "Elke rij is het gemiddelde van de GloVe-vectoren van de woorden van een zin: 50 dunne staafjes, "
                     "omhoog voor een positief getal en omlaag voor een negatief; de paar grootste getallen worden "
                     "afgekapt aan de rand van de rij. Grijs: de woorden van GloVe en het gemiddelde ervan."},
}


def glove_picture(first, second, lang: str) -> str:
    w = _GLOVE_WORDS[lang]
    E, row = _glove()
    texts = [str(first)[:LONGEST], str(second)[:LONGEST]]
    if not all(t.strip() for t in texts):
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    means, parts, alts = [], [], []
    for t in texts:
        words = words_of(t)
        known = [x for x in words if x in row]
        left = list(dict.fromkeys(x for x in words if x not in row))
        if not known:
            text = w["none"].format(s=_plain(t))
            return yours.figure(w["title"], _note(text), alt=_bare(text))
        means.append(mean_vector(known, E, row))
        part = _label(t) + f'<div style="{_SMALL}">{html.escape(w["used"])} ' + "".join(_chip(x) for x in known[:40]) + "</div>"
        if left:
            part += _note(w["left"].format(words=", ".join(_plain(x) for x in left[:20])))
        parts.append((part, known))
        alts.append(f"{t}: " + " ".join(known[:40]))
    scale = float(max(np.percentile(np.abs(m), 95) for m in means)) or 1.0   # a few large numbers are cut off
    body = "".join(p + f'<div style="margin:0.2rem 0 0.4rem">{_spark(m, scale, "var(--ea-data)")}</div>'
                   for (p, _), m in zip(parts, means))
    c = cosine(means[0], means[1])
    line = w["cosine"].format(c=_dec(c, lang))
    if sorted(parts[0][1]) == sorted(parts[1][1]):
        line += " " + w["same"]
    body += _label(line) + f'<div>{_bar(c, "var(--ea-data)", 220)}</div>'
    return yours.figure(w["title"], body + _note(w["legend"]),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + line + " " + _bare(w["legend"]))


# --- try it: sentences of your own, with Granite ------------------------------------------------------------------------


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in GRANITE_FILES)


def granite_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until Granite Embedding is downloaded; a click downloads it,
    with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, GRANITE_FILES)


MOST = 6   # sentences the explore takes

_GRANITE_WORDS = {
    "en": {"title": "Your sentences in Granite: every pair, most alike first",
           "head": ["First sentence", "Second sentence", "Cosine similarity"],
           "few": "Type at least two sentences, one per line.",
           "many": "The picture takes the first {k} sentences.",
           "legend": "Gray: the cosine similarity of Granite's embeddings of the two sentences, the dot product of two "
                     "vectors of length 1."},
    "nl": {"title": "Je zinnen in Granite: elk paar, meest gelijkend eerst",
           "head": ["Eerste zin", "Tweede zin", "Cosine similarity"],
           "few": "Typ minstens twee zinnen, één per regel.",
           "many": "De afbeelding neemt de eerste {k} zinnen.",
           "legend": "Grijs: de cosine similarity van de embeddings die Granite van de twee zinnen maakt, het "
                     "inproduct van twee vectoren met lengte 1."},
}


def lines_of(text) -> list:
    """The sentences in the box of the explore: one per line, without spaces around them, empty ones left out, each
    once."""
    out = []
    for line in str(text).splitlines():
        line = " ".join(line.split())[:LONGEST]
        if line and line not in out:
            out.append(line)
    return out


def granite_picture(text, lang: str, embed_fn=None) -> str:
    """Every pair of the typed sentences with the cosine similarity of their Granite embeddings, most alike first.
    `embed_fn` turns a list of texts into rows of length 1; by default everything_ai.embed.embed."""
    w = _GRANITE_WORDS[lang]
    sentences = lines_of(text)
    notes = []
    if len(sentences) < 2:
        return yours.figure(w["title"], _note(w["few"]), alt=w["few"])
    if len(sentences) > MOST:
        notes.append(w["many"].format(k=MOST))
        sentences = sentences[:MOST]
    if embed_fn is None:
        from everything_ai import embed
        embed_fn = lambda texts: embed.embed(texts).numpy()   # noqa: E731
    vectors = np.asarray(embed_fn(sentences), dtype=float)
    found = sorted(((float(vectors[i] @ vectors[j]), i, j) for i in range(len(sentences))
                    for j in range(i + 1, len(sentences))), key=lambda t: (-t[0], t[1], t[2]))
    rows_ = [[f"<i>{html.escape(sentences[i])}</i>", f"<i>{html.escape(sentences[j])}</i>",
              _number(c, lang, "var(--ea-data)")] for c, i, j in found]
    text_ = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], _table(rows_, w["head"]) + _note(text_),
                        alt=f'{w["title"]}. ' + "; ".join(f"{sentences[i]} / {sentences[j]}: {_dec(c, lang)}"
                                                          for c, i, j in found) + ". " + _bare(text_))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["same-mean", "mean_vector", "mean_pooling"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name, as the lesson's code writes it.
    GloVe's table goes as it is, read-only as in the lesson; every other array and a dict as a new copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        if isinstance(self.value, dict):
            return dict(self.value)
        if isinstance(self.value, np.ndarray) and _shared(self.value):
            return self.value
        return np.array(self.value)


class _Lazy(Named):
    """A Named value that is read when the check first needs it, such as GloVe or BERT-Tiny's table."""

    def __init__(self, name: str, make):
        self.name, self._make = name, make

    @property
    def value(self):
        return self._make()


def _code(v) -> str:
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_code(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_code(k)}: {_code(x)}" for k, x in v.items()) + "}"
    if isinstance(v, np.ndarray):
        if v.size <= 8:
            return "np.array(" + _code(v.tolist()) + ")"
        return None
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, np.integer):
        return str(int(v))
    return repr(v)


def _shown(v):
    """What a function gave, as a message shows it: a small array in full, a large one by its shape and first numbers."""
    if isinstance(v, np.ndarray) and v.size > 8:
        first = ", ".join(f"{x:.4g}" for x in v.ravel()[:3])
        shape = " × ".join(f"{n:,}" for n in v.shape)
        shape_nl = " × ".join(f"{n:,}".replace(",", ".") for n in v.shape)
        return {"en": f"an array of shape ({shape}) that starts with {first}",
                "nl": f"een array met shape ({shape_nl}) die begint met {first}"}
    text = _code(v)
    return text if text is not None else _short(v)


def _same_vector(got, want) -> bool:
    """None and None, or an array of the same shape as want with every number within 1e-5; nan counts as nan, so that a
    wrong version names a mistake that gives nan too."""
    if want is None:
        return got is None
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    return g.shape == np.shape(want) and bool(np.allclose(g, want, rtol=0, atol=1e-5, equal_nan=True))


def _return_in_loop(got, text) -> bool:
    """A `return` indented under a `for` or `while` line: the loop then ends in its first round."""
    lines = [line.split("#")[0].rstrip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        head = line.lstrip()
        if head.startswith(("for ", "while ")) and head.endswith(":"):
            depth = len(line) - len(head)
            for after in lines[i + 1:]:
                body = after.lstrip()
                if not body:
                    continue
                if len(after) - len(body) <= depth:
                    break
                if body == "return" or body.startswith(("return ", "return(")):
                    return True
    return False


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _bool_index(e) -> bool:
    """`tokens[mask == 1]`: a column of True and False, of shape (n, 1), picks no rows of a table of n rows."""
    return isinstance(e, IndexError) and "boolean index" in str(e)


# exercise 2: mean_vector. The visible cases on a small table; then a sentence without a known word, one that starts
# with a word without a row, a word twice, and sentences of GloVe, one with a word it lacks.
SMALL_E = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
SMALL_E.flags.writeable = False
SMALL_ROW = {"a": 0, "b": 1, "c": 2}
_E = Named("small_E", SMALL_E)
_ROW = Named("small_row", SMALL_ROW)
_GLOVE_E = _Lazy("E", lambda: _glove()[0])
_GLOVE_ROW = _Lazy("row", lambda: _glove()[1])
MEAN_CASES = [
    (["a", "b"], _E, _ROW),
    (["a", "zz"], _E, _ROW),
    (["zz", "yy"], _E, _ROW),
    (["zz", "a", "b"], _E, _ROW),
    (["a", "a", "b"], _E, _ROW),
    (["c", "b", "c", "zz"], _E, _ROW),
    (["the", "dog", "bit", "the", "man"], _GLOVE_E, _GLOVE_ROW),
    (["the", "shop", "opens", "at", "nine", "o'clock"], _GLOVE_E, _GLOVE_ROW),
]

_MV_HEAD = "import numpy as np\n\n\ndef mean_vector(words, E, row):\n"
_MV_TOP = _MV_HEAD + "    total = np.zeros(E.shape[1])\n    count = 0\n"
_MV_START = _MV_TOP + "    ...\n"
_MV_LOOP = ("    for w in words:\n        if w in row:\n            total = total + E[row[w]]\n"
            "            count = count + 1\n")
_MV_NONE = "    if count == 0:\n        return None\n"
_MV = _MV_TOP + _MV_LOOP + _MV_NONE + "    return total / count\n"
_MV_SUM = _MV_TOP + _MV_LOOP + _MV_NONE + "    return total\n"
_MV_ALL = _MV_TOP + _MV_LOOP + _MV_NONE + "    return total / len(words)\n"
_MV_ZEROS = _MV_TOP + _MV_LOOP + "    if count == 0:\n        return total\n    return total / count\n"
_MV_NO_NONE = _MV_TOP + _MV_LOOP + "    return total / count\n"
_MV_SET = _MV.replace("for w in words:", "for w in set(words):")
_MV_COUNT_ALL = (_MV_TOP + "    for w in words:\n        if w in row:\n            total = total + E[row[w]]\n"
                 "        count = count + 1\n" + _MV_NONE + "    return total / count\n")
_MV_UNSTORED = _MV.replace("total = total + E[row[w]]", "total + E[row[w]]")
_MV_NONE_IN_LOOP = (_MV_TOP + _MV_LOOP + "        if count == 0:\n            return None\n" + "    return total / count\n")
_MV_KEY = (_MV_TOP + "    for w in words:\n        total = total + E[row[w]]\n        count = count + 1\n"
           + _MV_NONE + "    return total / count\n")
_MV_RETURN = (_MV_TOP + "    for w in words:\n        if w in row:\n            total = total + E[row[w]]\n"
              "            count = count + 1\n        return total / count\n    return None\n")

# exercise 3: mean_pooling. The visible cases; then tables with more [PAD] places, a [PAD] place between real pieces,
# wider rows, and two sentences of the course in BERT-Tiny.


def _tiny(i: int):
    return _Lazy(f"C.tiny_tokens({i})", lambda: _data()[2][i])


def _tiny_mask(i: int):
    return _Lazy(f"C.tiny_mask({i})", lambda: _data()[3][i].reshape(-1, 1))


def _ro(a, kind=float) -> Named:
    """A small read-only array, shown in a message as the code that makes it."""
    v = np.array(a, dtype=kind)
    v.flags.writeable = False
    return Named("np.array(" + repr(a) + ")", v)


POOL_CASES = [
    (_ro([[1, 2], [3, 4], [9, 9]]), _ro([[1], [1], [0]], int)),
    (_ro([[1, 2], [3, 4]]), _ro([[1], [1]], int)),
    (_ro([[0, 0, 0], [2, 2, 2], [4, 4, 4], [5, 7, 9], [8, 8, 8]]), _ro([[1], [1], [1], [0], [0]], int)),
    (_ro([[2, -1], [6, 6], [4, 3]]), _ro([[1], [0], [1]], int)),
    (_ro([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1], [3, 1, 4, 1]]), _ro([[1], [1], [1], [1]], int)),
    (_tiny(0), _tiny_mask(0)),
    (_tiny(26), _tiny_mask(26)),
]

_MP_HEAD = "import numpy as np\n\n\ndef mean_pooling(tokens, mask):\n"
_MP_START = _MP_HEAD + "    ...\n"
_MP = _MP_HEAD + "    return (tokens * mask).sum(axis=0) / mask.sum()\n"
_MP_ALL = _MP_HEAD + "    return tokens.mean(axis=0)\n"
_MP_ZEROS = _MP_HEAD + "    return (tokens * mask).mean(axis=0)\n"
_MP_SUM = _MP_HEAD + "    return (tokens * mask).sum(axis=0)\n"
_MP_PADS = _MP_HEAD + "    return tokens.sum(axis=0) / mask.sum()\n"
_MP_AXIS = _MP_HEAD + "    return (tokens * mask).sum(axis=1) / mask.sum()\n"
_MP_NO_AXIS = _MP_HEAD + "    return (tokens * mask).sum() / mask.sum()\n"
_MP_FIRST = _MP_HEAD + "    return tokens[0]\n"

EXERCISES = {
    "same-mean": Sort(
        name="same-mean",
        groups={
            "same": {"en": "Exactly the same mean", "nl": "Precies hetzelfde gemiddelde"},
            "other": {"en": "Another mean", "nl": "Een ander gemiddelde"},
        },
        items={
            "swap": {"en": "*the bus hit the van* and *the van hit the bus*", "nl": "*the bus hit the van* en *the van hit the bus*"},
            "flip": {"en": "*not cold but warm* and *not warm but cold*", "nl": "*not cold but warm* en *not warm but cold*"},
            "twice": {"en": "*rain* and *rain rain*", "nl": "*rain* en *rain rain*"},
            "repeat": {"en": "*the soup is hot* and *the soup is hot the soup is hot*", "nl": "*the soup is hot* en *the soup is hot the soup is hot*"},
            "not": {"en": "*the soup is hot* and *the soup is not hot*", "nl": "*the soup is hot* en *the soup is not hot*"},
            "warm": {"en": "*the soup is hot* and *the soup is warm*", "nl": "*the soup is hot* en *the soup is warm*"},
            "hot": {"en": "*the soup is hot* and *the soup is hot hot*", "nl": "*the soup is hot* en *the soup is hot hot*"},
            "a": {"en": "*the bus hit the van* and *the bus hit a van*", "nl": "*the bus hit the van* en *the bus hit a van*"},
        },
        solution={"swap": "same", "flip": "same", "twice": "same", "repeat": "same",
                  "not": "other", "warm": "other", "hot": "other", "a": "other"},
        nudge={
            "swap": {"en": "Do the two sentences have the same words, each as many times? Does the order count in a mean?",
                     "nl": "Hebben de twee zinnen dezelfde woorden, elk even vaak? Telt de volgorde mee in een gemiddelde?"},
            "flip": {"en": "Count each word in both: *not*, *cold*, *but*, *warm*. Is any word there more often in one of the two?",
                     "nl": "Tel elk woord in allebei: *not*, *cold*, *but*, *warm*. Staat een woord in een van de twee vaker?"},
            "twice": {"en": "What is the mean of one vector? And of the same vector twice?",
                      "nl": "Wat is het gemiddelde van één vector? En van twee keer dezelfde vector?"},
            "repeat": {"en": "Every word of the long one is there twice. What does that do to each word's share of the mean?",
                       "nl": "Elk woord van de lange zin staat er twee keer in. Wat doet dat met het aandeel van elk woord in het gemiddelde?"},
            "not": {"en": "Does one of the two have a word that the other lacks?",
                    "nl": "Heeft een van de twee een woord dat de andere niet heeft?"},
            "warm": {"en": "Are *hot* and *warm* one word, with one vector?",
                     "nl": "Zijn *hot* en *warm* één woord, met één vector?"},
            "hot": {"en": "In the second, *hot* is two of five words; in the first, one of four. Is its share the same?",
                    "nl": "In de tweede is *hot* twee van de vijf woorden; in de eerste één van de vier. Is het aandeel gelijk?"},
            "a": {"en": "Count *the* in both. Does every word come as often?",
                  "nl": "Tel *the* in allebei. Komt elk woord even vaak voor?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each pair under the right group: does the mean of the GloVe vectors of the words come out exactly the same for the two sentences?",
                      "nl": "Zet elk paar in de goede groep: komt het gemiddelde van de GloVe-vectoren van de woorden voor de twee zinnen precies gelijk uit?"},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The pair {item} belongs in the other group. Which words does each sentence have, and how often?",
                      "nl": "Het paar {item} hoort in de andere groep. Welke woorden heeft elke zin, en hoe vaak?"},
            "correct": {"en": "All {n} are right. A mean sees only which words a sentence has, and what share of the words each one is: not where a word stands, and not what *not* turns around.",
                        "nl": "Alle {n} kloppen. Een gemiddelde ziet alleen welke woorden een zin heeft, en welk deel van de woorden elk woord is: niet waar een woord staat, en niet wat *not* omdraait."},
        },
    ),
    "mean_vector": FunctionBox(
        name="mean_vector",
        picture=mean_picture,
        function="mean_vector",
        start=_MV_START,
        solution=_MV,
        visible=MEAN_CASES[:2],
        hidden=MEAN_CASES[2:],
        compare=_same_vector,
        show=_shown,
        first=[_DOTS],
        time_limit=2.0,
        mistakes=[
            Mistake("key-error", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: a word of the sentence has no key in `row`, as GloVe lacks some words. Which words should your function leave out, and how can it know them before it looks them up?",
                    nl="Met `{call}` stopt je functie met een KeyError: een woord van de zin heeft geen key in `row`, zoals GloVe sommige woorden niet kent. Welke woorden moet je functie weglaten, en hoe weet je functie dat nog vóór het opzoeken?"),
            Mistake("return-in-loop", matches=_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: a `return` stands inside the loop, so your function gives back a result after the first word. How far should the `return` lines be indented, so that they run after the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er staat een `return` in de loop, dus je functie geeft al na het eerste woord een uitkomst terug. Hoe ver moeten de `return`-regels inspringen, zodat de regels na de loop komen?"),
            Mistake("sum", wrong=_MV_SUM,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the sum of the vectors. A mean divides that sum by something: by what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de som van de vectoren terug. Een gemiddelde deelt die som door iets: waardoor?"),
            Mistake("all-words", wrong=_MV_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of all words, also the words without a key in `row`. Which words did you add up?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal van alle woorden, ook de woorden zonder key in `row`. Welke woorden heb je opgeteld?"),
            Mistake("count-all", wrong=_MV_COUNT_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function counts every word of the sentence, also the words without a key in `row`, but adds up only the words with one. Where does `count = count + 1` stand?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt elk woord van de zin, ook de woorden zonder key in `row`, maar telt alleen de woorden met een key op. Waar staat `count = count + 1`?"),
            Mistake("not-stored", wrong=_MV_UNSTORED,
                    en="`{call}` gives {got}, and it should give {want}: `total` stays a vector of zeros. Your loop computes a sum in every round, but where does that sum go?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `total` blijft een vector met nullen. Je loop rekent elke ronde een som uit, maar waar blijft die som?"),
            Mistake("set", wrong=_MV_SET,
                    en="`{call}` gives {got}, and it should give {want}: a word that comes twice counts once in your mean. In a mean, every word of the sentence counts as often as it comes: what do you loop over?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat twee keer voorkomt, telt in jouw gemiddelde één keer. In een gemiddelde telt elk woord van de zin zo vaak mee als het voorkomt: waarover gaat je loop?"),
            Mistake("zeros", wrong=_MV_ZEROS,
                    en="`{call}` gives {got}, and it should give {want}: for a sentence without a word that has a key in `row`, your function gives back a vector of zeros. Without such words, the sentence has no mean at all: what does Python write for nothing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor een zin zonder woord met een key in `row` geeft je functie een vector met nullen terug. Zonder zulke woorden heeft de zin helemaal geen gemiddelde: wat schrijft Python voor niets?"),
            Mistake("no-none", wrong=_MV_NO_NONE,
                    en="`{call}` gives {got}, and it should give {want}: for a sentence without a word that has a key in `row`, your function divides by a count of 0, and NumPy gives nan, not a number. What should the function give back before it divides, when `count` is 0?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: voor een zin zonder woord met een key in `row` deelt je functie door een aantal van 0, en NumPy geeft nan, geen getal. Wat moet de functie teruggeven als `count` 0 is, nog vóór het delen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up the vectors of the words that have a key in `row`, count them, and give back their mean, or `None` without such a word.",
                      "nl": "Vervang de `...`: tel de vectoren op van de woorden met een key in `row`, tel hoeveel het er zijn, en geef het gemiddelde terug, of `None` zonder zo'n woord."},
            "unreached": {"en": "With `{call}`, your function gives None, though words of that sentence have a key in `row`. Which words does your `if` let through, at what moment does your function decide that it counted no word, and does it end with a `return` of the mean?",
                          "nl": "Met `{call}` geeft je functie None, terwijl woorden van die zin een key in `row` hebben. Welke woorden laat je `if` door, op welk moment besluit je functie dat er geen woord geteld is, en eindigt je functie met een `return` van het gemiddelde?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back: it has no `return` with a value. What should it give back after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat geen `return` met een waarde in. Wat moet je functie na de loop teruggeven?"},
            "slow": {"en": "Your function is still running after 2 seconds on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 2 seconden nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add up the vectors of the words that have a key in `row`, and divide by how many there are?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de vectoren op van de woorden met een key in `row`, en deelt je functie door hoeveel het er zijn?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a sentence without a word that has a key in `row`, and for a word that comes twice?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een zin zonder woord met een key in `row`, en voor een woord dat twee keer voorkomt?"},
            "correct": {"en": "Your `mean_vector` is right in all {n} cases, also with GloVe. Look at the picture: which kinds of pairs get the highest cosine similarity?",
                        "nl": "Je `mean_vector` klopt in alle {n} gevallen, ook met GloVe. Kijk naar de afbeelding: welke soorten paren krijgen de hoogste cosine similarity?"},
        },
    ),
    "mean_pooling": FunctionBox(
        name="mean_pooling",
        picture=pooling_picture,
        function="mean_pooling",
        start=_MP_START,
        solution=_MP,
        visible=POOL_CASES[:2],
        hidden=POOL_CASES[2:],
        compare=_same_vector,
        show=_shown,
        first=[_DOTS],
        time_limit=1.0,
        mistakes=[
            Mistake("bool-index", error=_bool_index,
                    en="With `{call}`, your function stops with an IndexError: `mask` is a column of shape (n, 1), one number per row, and `tokens` has many numbers per row. What shape does `mask == 1` have, and what shape does NumPy want to pick rows?",
                    nl="Met `{call}` stopt je functie met een IndexError: `mask` is een kolom met shape (n, 1), één getal per rij, en `tokens` heeft veel getallen per rij. Welke shape heeft `mask == 1`, en welke shape wil NumPy om rijen te kiezen?"),
            Mistake("all-rows", wrong=_MP_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the mean of every row, the [PAD] places too. Which rows should count, and what says so?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gemiddelde van elke rij, ook van de [PAD]-plekken. Welke rijen moeten meetellen, en wat zegt dat?"),
            Mistake("pads-summed", wrong=_MP_PADS,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of real pieces, but adds up every row, the [PAD] places too. How do you make a [PAD] row 0 before you add up?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal echte stukjes, maar telt elke rij op, ook de [PAD]-plekken. Hoe maak je een [PAD]-rij 0 voordat je optelt?"),
            Mistake("zeros-counted", wrong=_MP_ZEROS,
                    en="`{call}` gives {got}, and it should give {want}: the [PAD] rows are 0 in your sum, but your function still divides by the number of all rows. By how many rows should it divide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de [PAD]-rijen zijn 0 in je som, maar je functie deelt nog door het aantal van alle rijen. Door hoeveel rijen moet de functie delen?"),
            Mistake("sum", wrong=_MP_SUM,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the sum of the real rows. A mean divides by how many there are: what counts them?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de som van de echte rijen terug. Een gemiddelde deelt door hoeveel het er zijn: waarmee tel je de echte rijen?"),
            Mistake("axis-1", wrong=_MP_AXIS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up along axis 1, within each row, so it gives one number per place. Along which axis do the rows lie?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt op langs axis 1, binnen elke rij, dus geeft één getal per plek. Langs welke axis liggen de rijen?"),
            Mistake("no-axis", wrong=_MP_NO_AXIS,
                    en="`{call}` gives {got}, and it should give {want}: without an axis, `sum` adds up every number of the table into one. The result should be a vector, with one number per column: which axis should `sum` get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zonder axis telt `sum` elk getal van de tabel op tot één. De uitkomst moet een vector zijn, met één getal per kolom: welke axis moet `sum` krijgen?"),
            Mistake("first-row", wrong=_MP_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the first row, the vector of [CLS]. That is CLS pooling; mean pooling takes the mean of every real row. Which rows does the mask keep?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de eerste rij terug, de vector van [CLS]. Dat is CLS pooling; mean pooling neemt het gemiddelde van elke echte rij. Welke rijen houdt de mask?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the mean of the rows of `tokens` where `mask` is 1.",
                      "nl": "Vervang de `...`: geef het gemiddelde terug van de rijen van `tokens` waar `mask` 1 is."},
            "slow": {"en": "Your function is still running after 1 second on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 1 seconde nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add up only the rows where `mask` is 1, and divide by how many there are?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie alleen de rijen op waar `mask` 1 is, en deelt je functie door hoeveel het er zijn?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for more [PAD] places, for a 0 between two 1s, and for wider rows?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor meer [PAD]-plekken, voor een 0 tussen twee keer 1, en voor bredere rijen?"},
            "correct": {"en": "Your `mean_pooling` is right in all {n} cases, also for two sentences in BERT-Tiny. Look at the picture: which of the three columns sets the pairs with the same meaning furthest above the pairs with the same words?",
                        "nl": "Je `mean_pooling` klopt in alle {n} gevallen, ook voor twee zinnen in BERT-Tiny. Kijk naar de afbeelding: welke van de drie kolommen zet de paren met dezelfde betekenis het verst boven de paren met dezelfde woorden?"},
        },
    ),
}

CASES = [
    ("same-mean", {}, "empty", None),
    ("same-mean", {"swap": "same", "flip": "same", "twice": "same", "repeat": "same",
                   "not": "other", "warm": "other", "hot": "other", "a": "other"}, "correct", None),
    ("same-mean", {"swap": "same", "flip": "same"}, "wrong", "unfinished"),
    ("same-mean", {"swap": "other"}, "wrong", "wrong:swap"),
    ("same-mean", {"flip": "other"}, "wrong", "wrong:flip"),
    ("same-mean", {"twice": "other"}, "wrong", "wrong:twice"),
    ("same-mean", {"repeat": "other"}, "wrong", "wrong:repeat"),
    ("same-mean", {"not": "same"}, "wrong", "wrong:not"),
    ("same-mean", {"warm": "same"}, "wrong", "wrong:warm"),
    ("same-mean", {"hot": "same"}, "wrong", "wrong:hot"),
    ("same-mean", {"a": "same"}, "wrong", "wrong:a"),
    ("mean_vector", _MV_START, "empty", None),
    ("mean_vector", _MV, "correct", None),
    ("mean_vector", _MV_HEAD + "    known = [E[row[w]] for w in words if w in row]\n    if len(known) == 0:\n"
                    "        return None\n    return np.mean(known, axis=0)\n", "correct", None),
    ("mean_vector", _MV_TOP + "    for w in words:\n        if w not in row:\n            continue\n"
                    "        total += E[row[w]]\n        count += 1\n    if count > 0:\n        return total / count\n",
     "correct", None),
    ("mean_vector", _MV_KEY, "error", "key-error"),
    ("mean_vector", _MV_RETURN, "wrong", "return-in-loop"),
    ("mean_vector", _MV_SUM, "wrong", "sum"),
    ("mean_vector", _MV_ALL, "wrong", "all-words"),
    ("mean_vector", _MV_SET, "wrong", "set"),
    ("mean_vector", _MV_COUNT_ALL, "wrong", "count-all"),
    ("mean_vector", _MV_UNSTORED, "wrong", "not-stored"),
    ("mean_vector", _MV_NONE_IN_LOOP, "wrong", "unreached"),
    ("mean_vector", _MV_TOP + _MV_LOOP + _MV_NONE + "    total / count\n", "wrong", "unreached"),
    ("mean_vector", _MV_TOP + _MV_LOOP + "    mean = total / count\n", "wrong", "no-return"),
    ("mean_vector", _MV_ZEROS, "wrong", "zeros"),
    ("mean_vector", _MV_NO_NONE, "wrong", "no-none"),
    ("mean_vector", _MV_TOP + "    ...\n    return None\n", "wrong", "dots-left"),
    ("mean_pooling", _MP_START, "empty", None),
    ("mean_pooling", _MP, "correct", None),
    ("mean_pooling", _MP_HEAD + "    kept = tokens * mask\n    total = kept.sum(axis=0)\n    count = mask.sum()\n"
                     "    return total / count\n", "correct", None),
    ("mean_pooling", _MP_HEAD + "    return tokens[mask[:, 0] == 1].mean(axis=0)\n", "correct", None),
    ("mean_pooling", _MP_HEAD + "    return tokens[mask == 1].mean(axis=0)\n", "error", "bool-index"),
    ("mean_pooling", _MP_HEAD + "    tokens *= mask\n    return tokens.sum(axis=0) / mask.sum()\n", "correct", None),
    ("mean_pooling", _MP_HEAD + "    return np.sum(tokens * mask, axis=0) / np.sum(mask)\n", "correct", None),
    ("mean_pooling", _MP_ALL, "wrong", "all-rows"),
    ("mean_pooling", _MP_PADS, "wrong", "pads-summed"),
    ("mean_pooling", _MP_ZEROS, "wrong", "zeros-counted"),
    ("mean_pooling", _MP_SUM, "wrong", "sum"),
    ("mean_pooling", _MP_AXIS, "wrong", "axis-1"),
    ("mean_pooling", _MP_NO_AXIS, "wrong", "no-axis"),
    ("mean_pooling", _MP_FIRST, "wrong", "first-row"),
    ("mean_pooling", _MP_HEAD + "    ...\n    return tokens[0]\n", "wrong", "dots-left"),
]
