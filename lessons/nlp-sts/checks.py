# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-sts: where pairs of sentences belong on the scale of semantic textual similarity (a sort),
the precision and recall of a threshold on the cosine similarity as a detector of paraphrases (precision_recall), and
the lowest threshold that reaches a precision (lowest_threshold); with the pictures of what each gives on the pairs of
the STS Benchmark, and two explores.

The course ships 1,750 pairs of the STS Benchmark (data/stsb-captions.tsv.gz): short sentences about photos, from the
two sources whose sentences may be passed on, each with the score people gave it from 0 to 5, and its Dutch machine
translation. The cosine similarity Granite Embedding gives each pair, in English and in Dutch, the course ships too
(data/stsb-captions-granite.npz); both are made by data/tools/make_sts_captions.py, so the exercises and their pictures
work without the model. Granite itself (223 MB, everything_ai.embed) the learner downloads once for the last explore.
Every number in the text and the pictures is computed by the course (tests/test_nlp_sts.py computes them again)."""
import contextlib
import gzip
import html
import io
import math
import re
import warnings

import numpy as np

from everything_ai import chart, lesson as lesson_helpers, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-sts"
REPO = "ibm-granite/granite-embedding-107m-multilingual"   # everything_ai.embed.REPO: that module imports PyTorch
GRANITE_FILES = [(REPO, "model.safetensors"), (REPO, "tokenizer.json")]
CAPTIONS = paths.ROOT / "data" / "stsb-captions.tsv.gz"
COSINES = paths.ROOT / "data" / "stsb-captions-granite.npz"
LINE = 4   # a pair with a score of LINE or more counts as a paraphrase in this lesson

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The last part to try yourself downloads Granite Embedding, {download_size(GRANITE_FILES, 'en')}, once from Hugging Face, when you click its button in the lesson; the exercises and the rest of the lesson work without it.",
    "nl": f"Werkt op elke computer. Het laatste onderdeel om zelf te proberen downloadt één keer Granite Embedding, {download_size(GRANITE_FILES, 'nl')}, van Hugging Face, als je in de les op de knop ervoor klikt; de oefeningen en de rest van de les werken zonder.",
}

# --- the pairs ---------------------------------------------------------------------------------------------------------


def _data() -> dict:
    """The 1,750 pairs, read once: "rows", a tuple of (part, source, score, shown, en1, en2, nl1, nl2), and the arrays
    "score", "en", "nl" (the cosine similarities, float32) and "validation" (True for a pair of the validation set),
    all read-only."""
    def make():
        with gzip.open(CAPTIONS, "rt", encoding="utf-8", newline="") as f:
            lines = f.read().split("\n")
        rows = tuple(tuple(line.split("\t")) for line in lines[1:] if line)
        with np.load(COSINES, allow_pickle=False) as z:
            en, nl = z["en"].astype(np.float32), z["nl"].astype(np.float32)
        out = {"rows": rows, "score": np.array([float(r[2]) for r in rows]), "en": en, "nl": nl,
               "validation": np.array([r[0] != "test" for r in rows])}
        for k in ("score", "en", "nl", "validation"):
            out[k].flags.writeable = False
        return out
    return kept(f"{_KEY}:data", make)


def sts(part: str, lang: str = "en") -> tuple:
    """(cos, score) of a part of the pairs, as two new arrays: `part` is "validation" (the 1,375 pairs of the STS
    Benchmark's train and dev parts) or "test" (its 375 pairs of the test part); `cos` holds the cosine similarity of
    Granite's embeddings of the two sentences of each pair, of the English sentences with `lang` "en" and of their
    Dutch translations with "nl", and `score` the score people gave the English pair, from 0 to 5."""
    if part not in ("validation", "test") or lang not in ("en", "nl"):
        raise ValueError('C.sts takes "validation" or "test", and "en" or "nl"')
    d = _data()
    keep = d["validation"] if part == "validation" else ~d["validation"]
    return np.array(d[lang][keep]), np.array(d["score"][keep])


def _part(part: str, lang: str) -> tuple:
    """(cos, score) of a part as read-only arrays, for the checks and the pictures, made once."""
    def make():
        cos, score = sts(part, lang)
        cos.flags.writeable = False
        score.flags.writeable = False
        return cos, score
    return kept(f"{_KEY}:{part}:{lang}", make)


def _sentences(i: int, lang: str) -> tuple:
    """The two sentences of pair i: in English, and for "nl" their Dutch translations too."""
    r = _data()["rows"][i]
    return (r[4], r[5]) if lang == "en" else (r[4], r[5], r[6], r[7])


# --- the course's own versions of the exercises ------------------------------------------------------------------------


def precision_recall(cos, score, threshold):
    para = cos[score >= 4]
    other = cos[score < 4]
    tp = np.sum(para >= threshold)
    fp = np.sum(other >= threshold)
    fn = np.sum(para < threshold)
    precision = None
    recall = None
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    return precision, recall


def lowest_threshold(cos, score, target):
    for t in sorted(cos):
        precision, recall = precision_recall(cos, score, t)
        if precision >= target:
            return t
    return None


def counts(cos, score, threshold) -> dict:
    """The four cells of the confusion matrix of a threshold: tp, fp, fn and tn."""
    para, other = cos[score >= LINE], cos[score < LINE]
    tp, fn = int(np.sum(para >= threshold)), int(np.sum(para < threshold))
    fp, tn = int(np.sum(other >= threshold)), int(np.sum(other < threshold))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def chosen(target: float, lang: str):
    """The lowest threshold that reaches `target` precision on the validation set, for the English pairs or the
    Dutch translations, made once."""
    return kept(f"{_KEY}:chosen:{target}:{lang}", lambda: lowest_threshold(*_part("validation", lang), target))


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


def _int(n: int, lang: str) -> str:
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


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
        return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v))
    except (OverflowError, ValueError, TypeError):
        return False


def _number(x, lang: str, color: str, scale: float = 110, ink: str = "var(--ea-ink)") -> str:
    """A share from 0 to 1 as a bar and its number, the number in `ink`."""
    return (f'<span style="white-space:nowrap">{_bar(x, color, scale)} '
            f'<span style="font-variant-numeric:tabular-nums;color:{ink}">{_dec(x, lang)}</span></span>')


def _pair_cell(i: int, lang: str) -> str:
    """The two sentences of pair i; in Dutch with the translation under each, smaller."""
    s = _sentences(i, lang)
    if lang == "en":
        return f"<i>{html.escape(s[0])}</i><br><i>{html.escape(s[1])}</i>"
    return (f'<i>{html.escape(s[0])}</i> <span style="{_SMALL}">({html.escape(s[2])})</span><br>'
            f'<i>{html.escape(s[1])}</i> <span style="{_SMALL}">({html.escape(s[3])})</span>')


# --- the scatter of cosine similarity against score -------------------------------------------------------------------

_SCATTER_WORDS = {
    "en": {"x": "cosine similarity of Granite's embeddings", "y": "score", "threshold": "threshold {t}",
           "line": "score 4"},
    "nl": {"x": "cosine similarity van de embeddings van Granite", "y": "score", "threshold": "threshold {t}",
           "line": "score 4"},
}
X_LOW = 0.3   # the lowest cosine similarity of the 1,750 pairs is 0.325


def _scatter(cos, score, lang: str, *, threshold=None, means=None, width: int = 640, height: int = 300) -> str:
    """Every pair as a gray dot, its cosine similarity across and its score up; dots that lie on top of each other make
    the gray darker. With `means`, a list of (mean cosine similarity, score) per band of scores, a dark line through
    them; with `threshold`, a dashed line at that cosine similarity and one at the score 4."""
    w = _SCATTER_WORDS[lang]
    L, R, T, B = 50, width - 14, 14, height - 46
    def X(v):
        return L + (min(max(float(v), X_LOW), 1.0) - X_LOW) / (1.0 - X_LOW) * (R - L)
    def Y(v):
        return B - float(v) / 5.0 * (B - T)
    out = []
    for s in range(6):
        out.append(f'<line x1="{L}" y1="{Y(s):.1f}" x2="{R}" y2="{Y(s):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                   f'<text x="{L - 7}" y="{Y(s) + 4:.1f}" style="font-size:12px;fill:var(--ea-ink-3);text-anchor:end">{s}</text>')
    for k in range(3, 11):
        c = k / 10
        out.append(f'<line x1="{X(c):.1f}" y1="{B}" x2="{X(c):.1f}" y2="{B + 4}" style="stroke:var(--ea-axis)"/>'
                   f'<text x="{X(c):.1f}" y="{B + 18}" style="font-size:12px;fill:var(--ea-ink-3);text-anchor:middle">{_dec(c, lang, 1)}</text>')
    out.append(f'<line x1="{L}" y1="{T - 4}" x2="{L}" y2="{B}" style="stroke:var(--ea-axis)"/>'
               f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" style="stroke:var(--ea-axis)"/>'
               f'<text x="{(L + R) / 2:.1f}" y="{height - 8}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(w["x"])}</text>'
               f'<text transform="translate(14,{(T + B) / 2:.1f}) rotate(-90)" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(w["y"])}</text>')
    dots = "".join(f"M{X(c):.1f} {Y(s):.1f}h0" for c, s in zip(cos, score))
    out.append(f'<path d="{dots}" style="fill:none;stroke:var(--ea-data);stroke-opacity:0.32;stroke-width:4.5;stroke-linecap:round"/>')
    if means:
        pts = " ".join(f"{X(c):.1f},{Y(s):.1f}" for c, s in means)
        out.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2;stroke-linejoin:round"/>')
        out.append("".join(f'<circle cx="{X(c):.1f}" cy="{Y(s):.1f}" r="4" style="fill:var(--ea-ink);stroke:var(--ea-page);stroke-width:1.2"/>'
                           for c, s in means))
    if threshold is not None:
        tx = X(threshold)
        out.append(f'<line x1="{tx:.1f}" y1="{T - 4}" x2="{tx:.1f}" y2="{B}" style="stroke:var(--ea-ink);stroke-width:1.6;stroke-dasharray:5 4"/>'
                   f'<line x1="{L}" y1="{Y(LINE):.1f}" x2="{R}" y2="{Y(LINE):.1f}" style="stroke:var(--ea-ink);stroke-width:1.6;stroke-dasharray:5 4"/>')
        label = html.escape(w["threshold"].format(t=_dec(threshold, lang)))
        anchor, at = ("end", tx - 5) if tx > L + (R - L) * 0.6 else ("start", tx + 5)
        out.append(f'<text x="{at:.1f}" y="{T + 12}" style="font-size:13px;font-weight:600;fill:var(--ea-ink);text-anchor:{anchor};'
                   f'stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round;paint-order:stroke">{label}</text>'
                   f'<text x="{L + 6}" y="{Y(LINE) - 6:.1f}" style="font-size:13px;font-weight:600;fill:var(--ea-ink);'
                   f'stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round;paint-order:stroke">{html.escape(w["line"])}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" aria-hidden="true" style="width:100%;max-width:{width}px;height:auto;'
            f'font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">{"".join(out)}</svg>')


BANDS = [(0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.01)]


def band_means(cos, score) -> list:
    """(mean cosine similarity, middle score) for each band of scores: under 0.5, 0.5 to 1.5, and so on to 4.5 and up."""
    out = []
    for lo, hi in BANDS:
        keep = (score >= lo) & (score < hi)
        out.append((float(np.mean(cos[keep])), min((lo + hi) / 2, 4.75)))
    return out


# --- the idea's pictures ---------------------------------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "The 1,750 pairs: what Granite gives, and what people scored",
           "legend": "Each gray dot is a pair of sentences: across, the cosine similarity of Granite's embeddings of the "
                     "two English sentences; up, the score people gave the pair. Where dots lie on top of each other, the "
                     "gray gets darker. The dark line runs through the mean cosine similarity of each band of scores: "
                     "under 0.5, 0.5 to 1.5, and so on, up to 4.5 and more.",
           "alt": "Mean cosine similarity per band of scores: {means}.",
           "examples_title": "Five pairs, with their score and Granite's cosine similarity",
           "head": ["Sentences", "Score", "Cosine similarity"],
           "examples_legend": "Gray: the score people gave the pair, on a scale from 0 to 5, and the cosine similarity of "
                              "Granite's embeddings of the two English sentences, on a scale from 0 to 1.",
           "threshold_title": "The validation set at the threshold {t}",
           "matrix_head": ["", "cosine similarity ≥ {t}", "cosine similarity < {t}"],
           "para": "paraphrase (score 4 or more)", "other": "other pairs",
           "tp": "{n} true positives", "fp": "{n} false positives", "fn": "{n} false negatives", "tn": "{n} true negatives",
           "shares": "{k} pairs count as a paraphrase. Precision {p}, recall {r}.",
           "nothing": "No pair counts as a paraphrase: precision cannot be computed. Recall {r}.",
           "threshold_legend": "Each gray dot is a pair of the validation set. The dashed lines are the threshold, "
                               "across, and the score 4, up: the pairs on or right of the threshold count as a "
                               "paraphrase, the pairs on or above the score 4 are one. In the table, the four cells of the confusion "
                               "matrix."},
    "nl": {"title": "De 1.750 paren: wat Granite geeft, en wat mensen scoorden",
           "legend": "Elke grijze stip is een paar zinnen: naar rechts de cosine similarity van de embeddings die Granite "
                     "van de twee Engelse zinnen maakt; omhoog de score die mensen het paar gaven. Waar stippen over "
                     "elkaar liggen, wordt het grijs donkerder. De donkere lijn loopt door de gemiddelde cosine "
                     "similarity van elke band van scores: onder 0,5, 0,5 tot 1,5, enzovoort, tot 4,5 en meer.",
           "alt": "Gemiddelde cosine similarity per band van scores: {means}.",
           "examples_title": "Vijf paren, met de score en de cosine similarity van Granite",
           "head": ["Zinnen", "Score", "Cosine similarity"],
           "examples_legend": "Grijs: de score die mensen het paar gaven, op een schaal van 0 tot 5, en de cosine "
                              "similarity van de embeddings die Granite van de twee Engelse zinnen maakt, op een schaal "
                              "van 0 tot 1. Tussen haakjes de Nederlandse vertaling van DeepL.",
           "threshold_title": "De validation set bij de threshold {t}",
           "matrix_head": ["", "cosine similarity ≥ {t}", "cosine similarity < {t}"],
           "para": "parafrase (score 4 of meer)", "other": "andere paren",
           "tp": "{n} true positives", "fp": "{n} false positives", "fn": "{n} false negatives", "tn": "{n} true negatives",
           "shares": "{k} paren tellen als parafrase. Precision {p}, recall {r}.",
           "nothing": "Geen enkel paar telt als parafrase: de precision is niet uit te rekenen. Recall {r}.",
           "threshold_legend": "Elke grijze stip is een paar van de validation set. De gestreepte lijnen zijn de "
                               "threshold, naar rechts, en de score 4, omhoog: de paren op of rechts van de threshold "
                               "tellen als parafrase, de paren op of boven de score 4 zijn er een. In de tabel de vier vakken van "
                               "de confusion matrix."},
}

EXAMPLES = ["Red double decker bus.", "A man is laughing with a woman", "A man is carrying a canoe with a dog.",
            "The gate is blue.", "A yellow school bus parked in a field."]
EXAMPLE_SCORES = [5.0, 4.8, 1.8, 1.6, 0.0]   # with its score, each first sentence above finds one pair


def _example(first: str, score: float) -> int:
    for i, r in enumerate(_data()["rows"]):
        if r[4] == first and float(r[2]) == score:
            return i
    raise KeyError(first)


def idea_scatter_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    d = _data()
    means = band_means(d["en"], d["score"])
    alt = w["alt"].format(means=", ".join(_dec(c, lang) for c, _ in means))
    return yours.figure(w["title"], _scatter(d["en"], d["score"], lang, means=means) + _note(w["legend"]),
                        alt=f'{w["title"]}. {alt} {_bare(w["legend"])}')


def idea_examples_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    d = _data()
    rows_, alts = [], []
    for first, s in zip(EXAMPLES, EXAMPLE_SCORES):
        i = _example(first, s)
        c = float(d["en"][i])
        rows_.append([_pair_cell(i, lang),
                      f'<span style="white-space:nowrap">{_bar(s / 5, "var(--ea-data)", 60)} '
                      f'<span style="font-variant-numeric:tabular-nums">{_dec(s, lang, 1)}</span></span>',
                      _number(c, lang, "var(--ea-data)", 80)])
        a, b = _sentences(i, "en")
        alts.append(f"{a} / {b}: {_dec(s, lang, 1)}, {_dec(c, lang)}")
    return yours.figure(w["examples_title"], _table(rows_, w["head"]) + _note(w["examples_legend"]),
                        alt=f'{w["examples_title"]}. ' + "; ".join(alts) + ". " + _bare(w["examples_legend"]))


def _matrix(cos, score, t: float, lang: str) -> tuple:
    """The confusion matrix of a threshold as a table, and the line with precision and recall under it."""
    w = _IDEA_WORDS[lang]
    n = counts(cos, score, t)
    td = _dec(t, lang)
    head = [h.format(t=td) for h in w["matrix_head"]]
    rows_ = [[html.escape(w["para"]), html.escape(w["tp"].format(n=_int(n["tp"], lang))),
              html.escape(w["fn"].format(n=_int(n["fn"], lang)))],
             [html.escape(w["other"]), html.escape(w["fp"].format(n=_int(n["fp"], lang))),
              html.escape(w["tn"].format(n=_int(n["tn"], lang)))]]
    p, r = precision_recall(cos, score, t)
    k = n["tp"] + n["fp"]
    if p is None:
        line = w["nothing"].format(r=_dec(r, lang))
    else:
        line = w["shares"].format(k=_int(k, lang), p=_dec(p, lang), r=_dec(r, lang))
    return _table(rows_, head), line


def idea_threshold_picture(lang: str, t: float = 0.85) -> str:
    return explore_picture(t, "en", lang)


def explore_picture(t, data: str, lang: str) -> str:
    """The validation set at a threshold of the slider, for the English pairs or the Dutch translations."""
    w = _IDEA_WORDS[lang]
    try:
        t = float(t)
    except (TypeError, ValueError):
        t = 0.85
    cos, score = _part("validation", data if data in ("en", "nl") else "en")
    table, line = _matrix(cos, score, t, lang)
    title = w["threshold_title"].format(t=_dec(t, lang))
    body = _scatter(cos, score, lang, threshold=t) + table + _label(line) + _note(w["threshold_legend"])
    return yours.figure(title, body, alt=f"{title}. {line} {_bare(w['threshold_legend'])}")


# --- running the learner's function in a picture ------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 1   # what the learner's function gets per call in a picture


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit; arrays are passed as new copies, as `C.sts` gives them. _STOPPED
    when it stops with an error, and the Timeout passes on."""
    def copy(a):
        return np.array(a) if isinstance(a, np.ndarray) else a
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # NumPy's "invalid value in divide": the picture says it
            return with_time_limit(lambda: fn(*(copy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _as_pair(out):
    """The learner's (precision, recall) as a tuple of two values, or None when it is not a pair."""
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    return None


def _same_share(got, want) -> bool:
    """None and None, or two numbers within 1e-9; nan counts as nan, so that a wrong version names a mistake that gives
    nan too."""
    if want is None or got is None:
        return got is None and want is None
    if isinstance(want, float) and math.isnan(want):
        return _finite_or_nan(got) and math.isnan(float(got))
    return _finite(got) and abs(float(got) - float(want)) <= 1e-6


def _finite_or_nan(v) -> bool:
    try:
        return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)) and (
            math.isfinite(float(v)) or math.isnan(float(v)))
    except (OverflowError, ValueError, TypeError):
        return False


def same_pair(got, want) -> bool:
    """The comparison of the check of precision_recall: a tuple or list of two, each the same share as want's."""
    g, wp = _as_pair(got), _as_pair(want)
    if g is None or wp is None:
        return False
    return all(_same_share(a if not _finite_or_nan(a) else float(a), b if b is None else float(b))
               for a, b in zip(g, wp))


def same_threshold(got, want) -> bool:
    """The comparison of the check of lowest_threshold: None and None, or two numbers within 1e-6."""
    if want is None or got is None:
        return got is None and want is None
    return _finite(got) and abs(float(got) - float(want)) <= 1e-6


def _share_text(v, lang: str) -> str:
    if v is None:
        return "None"
    if _finite(v):
        return _dec(float(v), lang)
    if _finite_or_nan(v):
        return "nan"
    return _short(v, 30)


def _share_cell(v, want, lang: str) -> str:
    """One share of the learner's: a blue bar where it is the right number, purple where it differs; None as a chip."""
    good = _same_share(v if not _finite_or_nan(v) else float(v), want)
    if _finite(v) and 0 <= float(v) <= 1:
        return _number(float(v), lang, "var(--ea-pred)" if good else "var(--ea-loss)", 80,
                       "var(--ea-pred-text)" if good else "var(--ea-loss-text)")
    return _chip(_share_text(v, lang), _CHIP_PRED if good else _CHIP_PURPLE)


# --- the picture under exercise 2: your precision_recall on the validation set ------------------------------------------

PR_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
CURVE = [round(0.70 + k / 100, 2) for k in range(31)]   # the thresholds of the chart, 0.70 to 1.00

_PR_WORDS = {
    "en": {"title": "Your precision_recall on the validation set",
           "head": ["Threshold", "Pairs counted", "Your precision", "Your recall"],
           "top": "{t}, the highest cosine similarity",
           "above": "{t}, above every pair",
           "others": "{t}, only the {n} pairs under 4",
           "chart": "precision and recall for every threshold from 0.70 to 1.00",
           "precision": "precision", "recall": "recall", "wrong": "differs",
           "legend": "Blue: what your function gives on the validation set of the English pairs, in the chart precision "
                     "(solid line) and recall (dashed line) for every threshold from 0.70 to 1.00, with a gap where it "
                     "gives no number, and in the table for six thresholds; purple where it differs from the precision "
                     "and recall of that threshold. The last row gives your function only the pairs with a score under 4. "
                     "Gray: how many pairs have a cosine similarity of at least the threshold.",
           "differs": "With the threshold {t}{only}, your function gives {got}; the precision and recall are {want}.",
           "only": " and only the pairs under 4",
           "error": "Your function stops with an error on `precision_recall(cos, score, {t})`, with the validation set.",
           "slow": "Your function is still running after {s} second with `precision_recall(cos, score, {t})`.",
           "odd": "With `precision_recall(cos, score, {t})`, your function gives back {got}, and not two values."},
    "nl": {"title": "Jouw precision_recall op de validation set",
           "head": ["Threshold", "Paren geteld", "Jouw precision", "Jouw recall"],
           "top": "{t}, de hoogste cosine similarity",
           "above": "{t}, boven elk paar",
           "others": "{t}, alleen de {n} paren onder 4",
           "chart": "precision en recall voor elke threshold van 0,70 tot 1,00",
           "precision": "precision", "recall": "recall", "wrong": "verschilt",
           "legend": "Blauw: wat je functie geeft op de validation set van de Engelse paren, in de grafiek de precision "
                     "(doorgetrokken lijn) en de recall (gestreepte lijn) voor elke threshold van 0,70 tot 1,00, met een "
                     "gat waar de functie geen getal geeft, en in de tabel voor zes thresholds; paars waar het verschilt "
                     "van de precision en recall bij die threshold. De laatste rij geeft je functie alleen de paren met "
                     "een score onder 4. Grijs: hoeveel paren een cosine similarity van minstens de threshold hebben.",
           "differs": "Bij de threshold {t}{only} geeft je functie {got}; de precision en recall zijn {want}.",
           "only": " en alleen de paren onder 4",
           "error": "Je functie stopt met een fout op `precision_recall(cos, score, {t})`, met de validation set.",
           "slow": "Je functie is na {s} seconde nog bezig met `precision_recall(cos, score, {t})`.",
           "odd": "Met `precision_recall(cos, score, {t})` geeft je functie {got} terug, en geen twee waarden."},
}


def _pr_text(pair, lang: str) -> str:
    return "(" + ", ".join(_share_text(v, lang) for v in pair) + ")"


def pr_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _PR_WORDS[lang]
    cos, score = _part("validation", "en")
    other_cos, other_score = cos[score < LINE], score[score < LINE]
    top = float(np.max(cos))
    cases = [(_dec(t, lang, 2), cos, score, t, False) for t in PR_THRESHOLDS]
    cases += [(w["top"].format(t=_dec(top, lang, 4)), cos, score, top, False),
              (w["above"].format(t=_dec(1.0, lang, 2)), cos, score, 1.0, False),
              (w["others"].format(t=_dec(0.85, lang, 2), n=_int(len(other_cos), lang)), other_cos, other_score, 0.85, True)]
    notes, rows_, alts = [], [], []
    differs = False
    for label, c, s, t, only in cases:
        tc = _code_number(t)
        try:
            out = _call(fn, c, s, t)
        except Timeout:
            notes.append(w["slow"].format(t=tc, s=PICTURE_SECONDS))
            break
        if out is _STOPPED:
            notes.append(w["error"].format(t=tc))
            break
        got, want = _as_pair(out), precision_recall(c, s, t)
        if got is None:
            notes.append(w["odd"].format(t=tc, got=_plain(_short(out, 40))))
            break
        counted = int(np.sum(c >= t))
        rows_.append([html.escape(label), _chip(_int(counted, lang)),
                      _share_cell(got[0], want[0], lang), _share_cell(got[1], want[1], lang)])
        alts.append(f"{label}: {_pr_text(got, lang)}")
        if not same_pair(out, want) and not differs:
            notes.append(w["differs"].format(t=_dec(t, lang, 4 if t == top else 2), only=w["only"] if only else "",
                                             got=_plain(_pr_text(got, lang)), want=_pr_text(want, lang)))
            differs = True
    chart_html = _pr_chart(fn, cos, score, lang, w) if rows_ and len(rows_) == len(cases) else ""
    text_ = " ".join(notes + [w["legend"]])
    body = chart_html + (_table(rows_, w["head"]) if rows_ else "")
    return yours.figure(w["title"], body + _note(text_), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text_))


def _code_number(t: float) -> str:
    return f"{t:.4f}".rstrip("0").rstrip(".") if t != int(t) else f"{t:.1f}"


def _pr_chart(fn, cos, score, lang: str, w: dict) -> str:
    """Precision and recall of the learner's function for every threshold of CURVE, as two blue lines with a gap where
    it gives no number, and a purple dot where a number differs; empty when the function stops or runs too long."""
    ps, rs, wrong = [], [], []
    for t in CURVE:
        try:
            out = _call(fn, cos, score, t)
        except Timeout:
            return ""
        pair = _as_pair(out) if out is not _STOPPED else None
        if pair is None:
            return ""
        want = precision_recall(cos, score, t)
        values = []
        for v, wv in zip(pair, want):
            ok = _finite(v) and 0 <= float(v) <= 1
            values.append(float(v) if ok else None)
            if ok and not _same_share(float(v), wv):
                wrong.append((t, float(v)))
        ps.append(values[0])
        rs.append(values[1])
    panel = chart.Panel(x=(0.7, 1.0), y=(0, 1), xlabel="threshold", ylabel="")
    for ys, dashed in ((ps, False), (rs, True)):
        run = []
        for t, v in zip(CURVE + [None], ys + [None]):
            if v is None:
                if len(run) > 1:
                    panel.line([a for a, _ in run], [b for _, b in run], role="pred", dashed=dashed)
                elif len(run) == 1:
                    panel.points([run[0][0]], [run[0][1]], role="pred", r=3)
                run = []
            else:
                run.append((t, v))
    if wrong:
        panel.points([a for a, _ in wrong], [b for _, b in wrong], role="loss", r=4)
    svg = chart.figure([panel], alt=w["chart"], width=640, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    key = (f'<div style="{_SMALL};display:flex;gap:1.2rem;flex-wrap:wrap">'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-pred);'
           f'stroke-width:2.4;stroke-linecap:round"/></svg> {html.escape(w["precision"])}</span>'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-pred);'
           f'stroke-width:2.4;stroke-dasharray:5 5"/></svg> {html.escape(w["recall"])}</span>'
           + (f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:var(--ea-loss)"/></svg> '
              f'{html.escape(w["wrong"])}</span>' if wrong else "") + "</div>")
    return svg + key


# --- the picture under exercise 3: your lowest_threshold, on the validation set and the test set -------------------------

TARGETS = [0.6, 0.7, 0.8, 0.9]

_LOW_WORDS = {
    "en": {"title": "Your lowest_threshold, chosen on the validation set and measured on the test set",
           "head": ["Target precision", "Your threshold", "Validation set", "Test set"],
           "bands": {"en": "English pairs", "nl": "Dutch translations"},
           "pr": "precision {p}, recall {r}", "nothing": "no pair counts",
           "legend": "Blue: the threshold your function chooses on the validation set, for each target precision; purple "
                     "where it differs from the lowest threshold that reaches the target there. Beside it, the precision "
                     "and recall that your threshold gives on the validation set and on the test set. On the gray bands: "
                     "the English pairs, and their Dutch translations with the same scores.",
           "differs": "For the target {target} on the {band}, your function gives {got}; the lowest threshold with at "
                      "least that precision is {want}.",
           "error": "Your function stops with an error on `lowest_threshold(cos, score, {target})`, with the {band}.",
           "slow": "Your function is still running after {s} seconds with `lowest_threshold(cos, score, {target})`, with "
                   "the {band}.",
           "odd": "With `lowest_threshold(cos, score, {target})`, your function gives back {got}, and not a threshold."},
    "nl": {"title": "Jouw lowest_threshold, gekozen op de validation set en gemeten op de test set",
           "head": ["Doel voor precision", "Jouw threshold", "Validation set", "Test set"],
           "bands": {"en": "Engelse paren", "nl": "Nederlandse vertalingen"},
           "pr": "precision {p}, recall {r}", "nothing": "geen paar telt",
           "legend": "Blauw: de threshold die je functie op de validation set kiest, voor elk doel voor precision; paars "
                     "waar de threshold verschilt van de laagste threshold die daar het doel haalt. Ernaast de precision en recall "
                     "die jouw threshold geeft op de validation set en op de test set. Op de grijze banden: de Engelse "
                     "paren, en de Nederlandse vertalingen met dezelfde scores.",
           "differs": "Voor het doel {target} op de {band} geeft je functie {got}; de laagste threshold met minstens die "
                      "precision is {want}.",
           "error": "Je functie stopt met een fout op `lowest_threshold(cos, score, {target})`, met de {band}.",
           "slow": "Je functie is na {s} seconden nog bezig met `lowest_threshold(cos, score, {target})`, met de {band}.",
           "odd": "Met `lowest_threshold(cos, score, {target})` geeft je functie {got} terug, en geen threshold."},
}


def _pr_at(cos, score, t, lang: str, w: dict) -> str:
    p, r = precision_recall(cos, score, t)
    if p is None:
        return f'<span style="{_SMALL}">{html.escape(w["nothing"])}</span>'
    return f'<span style="{_SMALL}">{html.escape(w["pr"].format(p=_dec(p, lang), r=_dec(r, lang)))}</span>'


def low_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _LOW_WORDS[lang]
    rows_, shade, notes, alts = [], [], [], []
    stopped = differs = False
    for data in ("en", "nl"):
        band = w["bands"][data]
        shade.append(len(rows_))
        rows_.append([html.escape(band)])
        cos, score = _part("validation", data)
        test_cos, test_score = _part("test", data)
        for target in TARGETS:
            try:
                out = _call(fn, cos, score, target, seconds=2 * PICTURE_SECONDS)
            except Timeout:
                notes.append(w["slow"].format(target=target, band=band, s=2 * PICTURE_SECONDS))
                stopped = True
                break
            if out is _STOPPED:
                notes.append(w["error"].format(target=target, band=band))
                stopped = True
                break
            want = chosen(target, data)
            good = same_threshold(out, want)
            if out is None:
                cell, val, test = _chip("None", _CHIP_PRED if good else _CHIP_PURPLE), "", ""
            elif _finite(out):
                t = float(out)
                cell = _chip(_dec(t, lang, 4), _CHIP_PRED if good else _CHIP_PURPLE)
                val, test = _pr_at(cos, score, t, lang, w), _pr_at(test_cos, test_score, t, lang, w)
            else:
                notes.append(w["odd"].format(target=target, got=_plain(_short(out, 40))))
                stopped = True
                break
            rows_.append([_dec(target, lang, 1), cell, val, test])
            alts.append(f"{band}, {_dec(target, lang, 1)}: {_short(out, 20) if not _finite(out) else _dec(float(out), lang, 4)}")
            if not good and not differs:
                got = "None" if out is None else _dec(float(out), lang, 4)
                notes.append(w["differs"].format(target=_dec(target, lang, 1), band=band,
                                                 got=got, want="None" if want is None else _dec(float(want), lang, 4)))
                differs = True
        if stopped:
            break
    if stopped and shade[-1] == len(rows_) - 1:   # a gray band without a row under it
        rows_.pop()
        shade.pop()
    text_ = " ".join(notes + [w["legend"]])
    body = _table(rows_, w["head"], shade=tuple(shade)) if rows_ else ""
    return yours.figure(w["title"], body + _note(text_), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text_))


# --- try it: sentences of your own, in Granite ---------------------------------------------------------------------------


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in GRANITE_FILES)


def granite_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until Granite Embedding is downloaded; a click downloads it,
    with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, GRANITE_FILES)


MOST = 6        # sentences the explore takes
LONGEST = 300   # characters of a typed sentence that count
TARGET_CHOICES = ["0.6", "0.7", "0.8", "0.9"]

_GRANITE_WORDS = {
    "en": {"title": "Your sentences in Granite: every pair, most alike first",
           "head": ["First sentence", "Second sentence", "Cosine similarity", "At the threshold"],
           "yes": "paraphrase", "no": "no paraphrase",
           "few": "Type at least two sentences, one per line.",
           "many": "The picture takes the first {k} sentences.",
           "legend": "Gray: the cosine similarity of Granite's embeddings of the two sentences. A pair counts as a "
                     "paraphrase when it reaches the threshold {t}: the lowest threshold with a precision of at least "
                     "{target} on the validation set of the English pairs."},
    "nl": {"title": "Je zinnen in Granite: alle paren, het meest gelijkende paar bovenaan",
           "head": ["Eerste zin", "Tweede zin", "Cosine similarity", "Bij de threshold"],
           "yes": "parafrase", "no": "geen parafrase",
           "few": "Typ minstens twee zinnen, één per regel.",
           "many": "De afbeelding neemt de eerste {k} zinnen.",
           "legend": "Grijs: de cosine similarity van de embeddings die Granite van de twee zinnen maakt. Een paar telt "
                     "als parafrase als het de threshold {t} haalt: de laagste threshold met een precision van minstens "
                     "{target} op de validation set van de Nederlandse vertalingen."},
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


def granite_picture(text, target, lang: str, embed_fn=None) -> str:
    """Every pair of the typed sentences with the cosine similarity of their Granite embeddings, most alike first, and
    whether it reaches the threshold chosen for `target` precision on the validation set: of the English pairs in the
    English lesson, of the Dutch translations in the Dutch one. `embed_fn` turns a list of texts into rows of length 1;
    by default everything_ai.embed.embed."""
    w = _GRANITE_WORDS[lang]
    try:
        target = float(target)
    except (TypeError, ValueError):
        target = 0.8
    if f"{target:g}" not in TARGET_CHOICES:
        target = 0.8
    t = float(chosen(target, lang))
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
                    for j in range(i + 1, len(sentences))), key=lambda f: (-f[0], f[1], f[2]))
    rows_ = [[f"<i>{html.escape(sentences[i])}</i>", f"<i>{html.escape(sentences[j])}</i>",
              _number(c, lang, "var(--ea-data)", 80),
              _chip(w["yes"]) if c >= t else f'<span style="{_SMALL}">{html.escape(w["no"])}</span>']
             for c, i, j in found]
    text_ = " ".join(notes + [w["legend"].format(t=_dec(t, lang, 4), target=_dec(target, lang, 1))])
    return yours.figure(w["title"], _table(rows_, w["head"]) + _note(text_),
                        alt=f'{w["title"]}. ' + "; ".join(f"{sentences[i]} / {sentences[j]}: {_dec(c, lang)}, "
                                                          f"{w['yes'] if c >= t else w['no']}"
                                                          for c, i, j in found) + ". " + _bare(text_))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["sts-scale", "precision_recall", "lowest_threshold"]

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
    An array goes as a new copy, as `C.sts` gives one."""

    def __init__(self, name: str, value):
        self.name, self._value = name, value

    @property
    def value(self):
        return self._value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        v = self.value
        return np.array(v) if isinstance(v, np.ndarray) else v


class _Lazy(Named):
    """A Named value that is read when the check first needs it, such as the validation set."""

    def __init__(self, name: str, make):
        self.name, self._make = name, make

    @property
    def value(self):
        return self._make()


def _ro(a) -> Named:
    """A small read-only array, shown in a message as the code that makes it."""
    v = np.array(a, dtype=float)
    v.flags.writeable = False
    return Named("np.array(" + repr(a) + ")", v)


def _pr_shown(v):
    """What precision_recall gave, as a message shows it: a pair with its numbers to 4 digits."""
    pair = _as_pair(v)
    if pair is None:
        return _short(v)

    def one(x):
        if x is None:
            return "None"
        if _finite_or_nan(x):
            return "nan" if math.isnan(float(x)) else f"{float(x):.4g}"
        return _short(x, 20)
    return "(" + ", ".join(one(x) for x in pair) + ")"


def _threshold_shown(v):
    if v is None:
        return "None"
    return f"{float(v):.4g}" if _finite(v) else _short(v)


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _none_compare(e) -> bool:
    """`precision >= target` with a precision of None: a threshold at which no pair counts as a paraphrase."""
    return isinstance(e, TypeError) and "NoneType" in str(e) and ("<" in str(e) or ">" in str(e))


def _not_pair(got, text) -> bool:
    return got is not None and _as_pair(got) is None


def _bare_return_in_loop(got, text) -> bool:
    """A `return` straight in the body of a `for` or `while` loop, not under an `if`: the loop then ends in its first
    round."""
    lines = [line.split("#")[0].rstrip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        head = line.lstrip()
        if head.startswith(("for ", "while ")) and head.endswith(":"):
            depth = len(line) - len(head)
            body_depth = None
            for after in lines[i + 1:]:
                body = after.lstrip()
                if not body:
                    continue
                d = len(after) - len(body)
                if d <= depth:
                    break
                body_depth = d if body_depth is None else body_depth
                if d == body_depth and (body == "return" or body.startswith(("return ", "return("))):
                    return True
    return False


def _both_zero(got, text) -> bool:
    """Two shares of 0 where at least one cannot be computed: a precision or recall that starts at 0 instead of None."""
    pair = _as_pair(got)
    return pair is not None and all(_finite(v) and float(v) == 0 for v in pair)


def _above_4(got, text) -> bool:
    """`score > 4` in one of the masks, so a pair with a score of exactly 4 is not a paraphrase."""
    return re.search(r"score\s*>\s*4(?![\d.])", text) is not None


def _all_cosines():
    """Every cosine similarity of the cases of exercise 3, to tell a threshold of the learner's own from a pair's."""
    return kept(f"{_KEY}:all-cosines", lambda: np.concatenate(
        [np.asarray(c.value if isinstance(c, Named) else c, dtype=float) for c, _, _ in LOW_CASES]))


def _not_a_cosine(got, text) -> bool:
    """A threshold that is no pair's cosine similarity: the loop tried numbers of its own, such as np.arange."""
    return _finite(got) and not bool(np.any(np.abs(_all_cosines() - float(got)) <= 1e-6))


# exercise 2: precision_recall. The visible cases; then a threshold equal to a cosine similarity, a set without a
# paraphrase, and the validation and test sets of the English pairs.
_VAL_COS = _Lazy("cos", lambda: _part("validation", "en")[0])
_VAL_SCORE = _Lazy("score", lambda: _part("validation", "en")[1])
_TEST_COS = _Lazy("test_cos", lambda: _part("test", "en")[0])
_TEST_SCORE = _Lazy("test_score", lambda: _part("test", "en")[1])
_SMALL_COS = _ro([0.9, 0.8, 0.7, 0.6])
_SMALL_SCORE = _ro([4.5, 2.0, 4.0, 5.0])
_TOP = _Lazy("max(cos)", lambda: float(np.max(_part("validation", "en")[0])))
PR_CASES = [
    (_SMALL_COS, _SMALL_SCORE, 0.75),
    (_SMALL_COS, _SMALL_SCORE, 0.95),
    (_SMALL_COS, _SMALL_SCORE, 0.8),
    (_ro([0.9, 0.5, 0.7]), _ro([3.8, 1.0, 0.0]), 0.6),
    (_VAL_COS, _VAL_SCORE, 0.85),
    (_TEST_COS, _TEST_SCORE, 0.9),
    (_VAL_COS, _VAL_SCORE, _TOP),
]

_PR_HEAD = "import numpy as np\n\n\ndef precision_recall(cos, score, threshold):\n"
_PR_START = _PR_HEAD + "    ...\n"
_PR_SPLIT = "    para = cos[score >= 4]\n    other = cos[score < 4]\n"
_PR_COUNT = ("    tp = np.sum(para >= threshold)\n    fp = np.sum(other >= threshold)\n"
             "    fn = np.sum(para < threshold)\n")
_PR_NONE = ("    precision = None\n    recall = None\n    if tp + fp > 0:\n        precision = tp / (tp + fp)\n"
            "    if tp + fn > 0:\n        recall = tp / (tp + fn)\n")
_PR = _PR_HEAD + _PR_SPLIT + _PR_COUNT + _PR_NONE + "    return precision, recall\n"
_PR_SWAP = _PR_HEAD + _PR_SPLIT + _PR_COUNT + _PR_NONE + "    return recall, precision\n"
_PR_STRICT = _PR.replace("np.sum(para >= threshold)", "np.sum(para > threshold)").replace(
    "np.sum(other >= threshold)", "np.sum(other > threshold)").replace("np.sum(para < threshold)", "np.sum(para <= threshold)")
_PR_ABOVE = _PR.replace("cos[score >= 4]", "cos[score > 4]").replace("cos[score < 4]", "cos[score <= 4]")
_PR_ZEROS = _PR.replace("    precision = None\n    recall = None\n", "    precision = 0\n    recall = 0\n")
_PR_NAN = _PR_HEAD + _PR_SPLIT + _PR_COUNT + "    return tp / (tp + fp), tp / (tp + fn)\n"
_PR_NAN_RECALL = (_PR_HEAD + _PR_SPLIT + _PR_COUNT + "    precision = None\n    if tp + fp > 0:\n"
                  "        precision = tp / (tp + fp)\n    return precision, tp / (tp + fn)\n")
_PR_SAME = _PR.replace("recall = tp / (tp + fn)", "recall = tp / (tp + fp)").replace("if tp + fn > 0:", "if tp + fp > 0:")
_PR_FN_OTHER = _PR.replace("fn = np.sum(para < threshold)", "fn = np.sum(other < threshold)")
_PR_ALL = _PR.replace("precision = tp / (tp + fp)", "precision = tp / len(cos)")

# exercise 3: lowest_threshold. The visible cases on four pairs; then a set where no threshold reaches the target, and
# the validation sets of the English pairs and of the Dutch translations.
_LOW_COS = _ro([0.8, 0.6, 0.9, 0.7])
_LOW_SCORE = _ro([4.0, 1.0, 5.0, 4.2])
_NL_COS = _Lazy("nl_cos", lambda: _part("validation", "nl")[0])
_NL_SCORE = _Lazy("nl_score", lambda: _part("validation", "nl")[1])
LOW_CASES = [
    (_LOW_COS, _LOW_SCORE, 0.8),
    (_LOW_COS, _LOW_SCORE, 0.75),
    (_ro([0.95, 0.9, 0.6]), _ro([2.0, 5.0, 4.0]), 1.0),
    (_VAL_COS, _VAL_SCORE, 0.8),
    (_VAL_COS, _VAL_SCORE, 0.6),
    (_NL_COS, _NL_SCORE, 0.9),
]

_LT_HEAD = "def lowest_threshold(cos, score, target):\n"
_LT_START = _LT_HEAD + "    ...\n"
_LT_LOOP = "    for t in sorted(cos):\n        precision, recall = precision_recall(cos, score, t)\n"
_LT = _LT_HEAD + _LT_LOOP + "        if precision >= target:\n            return t\n    return None\n"
_LT_HIGH = _LT.replace("sorted(cos)", "sorted(cos, reverse=True)")
_LT_LAST = (_LT_HEAD + "    best = None\n" + _LT_LOOP + "        if precision >= target:\n            best = t\n"
            "    return best\n")
_LT_UNSORTED = _LT.replace("sorted(cos)", "cos")
_LT_PRECISION = _LT.replace("            return t\n", "            return precision\n")
_LT_STRICT = _LT.replace("precision >= target", "precision > target")
_LT_RECALL = _LT.replace("if precision >= target", "if recall >= target")
_LT_TARGET = _LT.replace("precision_recall(cos, score, t)", "precision_recall(cos, score, target)")
_LT_BARE = _LT_HEAD + _LT_LOOP + "        return t\n    return None\n"

EXERCISES = {
    "sts-scale": Sort(
        name="sts-scale",
        groups={
            "high": {"en": "4 or 5: equivalent, at most unimportant details differ",
                     "nl": "4 of 5: gelijkwaardig, hooguit onbelangrijke details verschillen"},
            "middle": {"en": "1 to 3: important information differs, but the topic is the same",
                       "nl": "1 tot 3: belangrijke informatie verschilt, maar het onderwerp is hetzelfde"},
            "zero": {"en": "0: completely dissimilar", "nl": "0: helemaal verschillend"},
        },
        items={
            "train": {"en": "*The train to Utrecht leaves at nine.* and *At nine, the train to Utrecht departs.*",
                      "nl": "*De trein naar Utrecht vertrekt om negen uur.* en *Om negen uur gaat de trein naar Utrecht.*"},
            "store": {"en": "*The store is open on Sunday.* and *The store is not open on Sunday.*",
                      "nl": "*De winkel is zondag open.* en *De winkel is zondag niet open.*"},
            "soup": {"en": "*The soup is too salty.* and *The bridge is closed for repairs.*",
                     "nl": "*De soep is te zout.* en *De brug is dicht voor onderhoud.*"},
            "cat": {"en": "*A cat sleeps on the sofa.* and *A cat is asleep on the couch.*",
                    "nl": "*Een kat slaapt op de bank.* en *Op de bank ligt een kat te slapen.*"},
            "window": {"en": "*A black cat sits in the window.* and *A black dog sits in the window.*",
                       "nl": "*Een zwarte kat zit voor het raam.* en *Een zwarte hond zit voor het raam.*"},
            "sold": {"en": "*The baker sold a bike to the teacher.* and *The teacher sold a bike to the baker.*",
                     "nl": "*De bakker verkocht een fiets aan de leraar.* en *De leraar verkocht een fiets aan de bakker.*"},
            "snow": {"en": "*Two children are playing in the snow.* and *Two kids play in the snow.*",
                     "nl": "*Twee kinderen spelen in de sneeuw.* en *Twee kinderen zijn aan het spelen in de sneeuw.*"},
            "museum": {"en": "*The museum opens at ten.* and *The museum closes at ten.*",
                       "nl": "*Het museum gaat om tien uur open.* en *Het museum gaat om tien uur dicht.*"},
            "bread": {"en": "*It is raining in Chicago.* and *The price of bread went up.*",
                      "nl": "*Het regent in Amsterdam.* en *De prijs van brood is gestegen.*"},
        },
        solution={"train": "high", "cat": "high", "snow": "high", "store": "middle", "window": "middle",
                  "sold": "middle", "museum": "middle", "soup": "zero", "bread": "zero"},
        nudge={
            "train": {"en": "Does the train, the place or the time change between the two? What differs, apart from the words?",
                      "nl": "Verandert de trein, de plek of de tijd tussen de twee? Wat verschilt er, behalve de woorden?"},
            "cat": {"en": "Is a couch another thing than a sofa? Does anything change that matters?",
                    "nl": "Zegt *ligt te slapen* iets anders dan *slaapt*? Verandert er iets wat ertoe doet?"},
            "snow": {"en": "Do *kids* and *play* say anything that *children* and *are playing* do not?",
                     "nl": "Zegt *zijn aan het spelen* iets wat *spelen* niet zegt?"},
            "store": {"en": "Can both sentences be true on the same Sunday? And are both about the same thing?",
                      "nl": "Kunnen beide zinnen op dezelfde zondag waar zijn? En gaan beide over hetzelfde?"},
            "window": {"en": "Is the animal in the window an unimportant detail, or important information?",
                       "nl": "Is het dier voor het raam een onbelangrijk detail, of belangrijke informatie?"},
            "sold": {"en": "Who has the bike at the end, in each sentence?",
                     "nl": "Wie heeft de fiets aan het eind, in elke zin?"},
            "museum": {"en": "Does opening at ten tell you the same as closing at ten? And is the topic the same?",
                       "nl": "Zegt om tien uur opengaan hetzelfde als om tien uur dichtgaan? En is het onderwerp hetzelfde?"},
            "soup": {"en": "Do the two sentences share a topic, or even one detail?",
                     "nl": "Hebben de twee zinnen een onderwerp gemeen, of zelfs maar één detail?"},
            "bread": {"en": "Is there anything that both sentences are about?",
                      "nl": "Is er iets waar beide zinnen over gaan?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each pair under the scores it would get on the STS scale.",
                      "nl": "Zet elk paar bij de scores die het op de STS-schaal zou krijgen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The pair {item} belongs in another group. What do the two sentences say, apart from the words they use?",
                      "nl": "Het paar {item} hoort in een andere groep. Wat zeggen de twee zinnen, los van de gebruikte woorden?"},
            "correct": {"en": "All {n} are right. The scale asks what two sentences say, not which words they share: *not* and *closes* change one word, and with it what the sentence says, and the baker and the teacher share every word while the bike goes the other way.",
                        "nl": "Alle {n} kloppen. De schaal vraagt wat twee zinnen zeggen, niet welke woorden de zinnen delen: *niet* en *dicht* veranderen één woord, en daarmee wat de zin zegt, en de bakker en de leraar delen elk woord terwijl de fiets de andere kant op gaat."},
        },
    ),
    "precision_recall": FunctionBox(
        name="precision_recall",
        picture=pr_picture,
        function="precision_recall",
        start=_PR_START,
        solution=_PR,
        visible=PR_CASES[:2],
        hidden=PR_CASES[2:],
        compare=same_pair,
        show=_pr_shown,
        first=[_DOTS],
        time_limit=2.0,
        mistakes=[
            Mistake("zero-division", error=_zero_division,
                    en="With `{call}`, your function stops with a ZeroDivisionError: no pair counts as a paraphrase, or no pair is one, so a share would divide by 0, and Python cannot divide by 0. What should your function give back for such a share, and how does it know before it divides?",
                    nl="Met `{call}` stopt je functie met een ZeroDivisionError: geen enkel paar telt als parafrase, of geen enkel paar is er een, dus een deel zou door 0 delen, en Python kan niet door 0 delen. Wat moet je functie voor zo'n deel teruggeven, en hoe weet de functie dat nog vóór het delen?"),
            Mistake("not-a-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give {want}: two values, the precision first and the recall second. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: twee waarden, eerst de precision en dan de recall. Wat geeft je `return`-regel terug?"),
            Mistake("swapped", wrong=_PR_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your two numbers are the right ones, in the other order. Which of the two comes first in the result?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je twee getallen kloppen, in de andere volgorde. Welke van de twee komt eerst in de uitkomst?"),
            Mistake("strict", wrong=_PR_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: in your function, a pair whose cosine similarity is exactly the threshold does not count as a paraphrase. Does a pair exactly at the threshold count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je functie telt een paar met een cosine similarity van precies de threshold niet als parafrase. Telt een paar dat precies op de threshold ligt mee?"),
            Mistake("above-4", wrong=_PR_ABOVE,
                    en="`{call}` gives {got}, and it should give {want}: in your function, a pair with a score of exactly 4 is not a paraphrase. From which score on is a pair a paraphrase in this lesson?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je functie is een paar met een score van precies 4 geen parafrase. Vanaf welke score is een paar in deze les een parafrase?"),
            Mistake("zeros", wrong=_PR_ZEROS,
                    en="`{call}` gives {got}, and it should give {want}: where a share cannot be computed, because it would divide by 0, your function gives 0. A share of no pairs is not 0: what does Python write for nothing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: waar een deel niet uit te rekenen is, omdat het door 0 zou delen, geeft je functie 0. Een deel van geen enkel paar is niet 0: wat schrijft Python voor niets?"),
            Mistake("nan", wrong=_PR_NAN,
                    en="`{call}` gives {got}, and it should give {want}: where no pair counts as a paraphrase, your function divides 0 by 0, and NumPy gives nan, not a number. What should your function give back for that share, before it divides?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: waar geen enkel paar als parafrase telt, deelt je functie 0 door 0, en NumPy geeft nan, geen getal. Wat moet je functie voor dat deel teruggeven, nog vóór het delen?"),
            Mistake("nan-recall", wrong=_PR_NAN_RECALL,
                    en="`{call}` gives {got}, and it should give {want}: without a single paraphrase among the pairs, your recall divides 0 by 0, and NumPy gives nan. What should the recall be when no pair is a paraphrase?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zonder een enkele parafrase tussen de paren deelt je recall 0 door 0, en NumPy geeft nan. Wat moet de recall zijn als geen enkel paar een parafrase is?"),
            Mistake("same-share", wrong=_PR_SAME,
                    en="`{call}` gives {got}, and it should give {want}: your recall comes out the same as your precision. Of which pairs is recall a share?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je recall komt hetzelfde uit als je precision. Van welke paren is de recall een deel?"),
            Mistake("negatives", wrong=_PR_FN_OTHER,
                    en="`{call}` gives {got}, and it should give {want}: your false negatives are the other pairs under the threshold, and those are true negatives. Which pairs does the detector miss?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je false negatives zijn de andere paren onder de threshold, en dat zijn true negatives. Welke paren mist de detector?"),
            Mistake("all-pairs", wrong=_PR_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your precision divides by the number of all pairs. Of which pairs is precision a share?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je precision deelt door het aantal van alle paren. Van welke paren is de precision een deel?"),
            Mistake("zeros-start", matches=_both_zero,
                    en="`{call}` gives {got}, and it should give {want}: where a share cannot be computed, because it would divide by 0, your function gives 0. A share of no pairs is not 0: what does Python write for nothing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: waar een deel niet uit te rekenen is, omdat het door 0 zou delen, geeft je functie 0. Een deel van geen enkel paar is niet 0: wat schrijft Python voor niets?"),
            Mistake("above-4-text", matches=_above_4,
                    en="`{call}` gives {got}, and it should give {want}: in your function, a pair with a score of exactly 4 is not a paraphrase. From which score on is a pair a paraphrase in this lesson?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je functie is een paar met een score van precies 4 geen parafrase. Vanaf welke score is een paar in deze les een parafrase?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the true positives, false positives and false negatives of the threshold, and give back the precision and the recall, with `None` for a share that would divide by 0.",
                      "nl": "Vervang de `...`: tel de true positives, false positives en false negatives van de threshold, en geef de precision en de recall terug, met `None` voor een deel dat door 0 zou delen."},
            "unreached": {"en": "With `{call}`, your function gives None, and not two values. Does every way through your function end with a `return` of the precision and the recall?",
                          "nl": "Met `{call}` geeft je functie None, en geen twee waarden. Eindigt elke weg door je functie met een `return` van de precision en de recall?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back: it has no `return` with a value. What should it give back at the end?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat geen `return` met een waarde in. Wat moet de functie aan het eind teruggeven?"},
            "slow": {"en": "Your function is still running after 2 seconds on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 2 seconden nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which pairs are paraphrases, which count as one, and of which pairs is each of the two a share?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke paren zijn parafrases, welke tellen als parafrase, en van welke paren is elk van de twee een deel?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a pair exactly at the threshold, and for pairs without a single paraphrase?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een paar precies op de threshold, en voor paren zonder een enkele parafrase?"},
            "correct": {"en": "Your `precision_recall` is right in all {n} cases, also on the validation set and the test set. Look at the chart: as the threshold goes up, which of the two lines only goes down?",
                        "nl": "Je `precision_recall` klopt in alle {n} gevallen, ook op de validation set en de test set. Kijk naar de grafiek: welke van de twee lijnen gaat alleen maar omlaag als de threshold omhooggaat?"},
        },
    ),
    "lowest_threshold": FunctionBox(
        name="lowest_threshold",
        picture=low_picture,
        function="lowest_threshold",
        start=_LT_START,
        solution=_LT,
        visible=LOW_CASES[:2],
        hidden=LOW_CASES[2:],
        compare=same_threshold,
        show=_threshold_shown,
        first=[_DOTS],
        names={"precision_recall": precision_recall},
        time_limit=3.0,
        mistakes=[
            Mistake("none-compare", error=_none_compare,
                    en="With `{call}`, your function stops with a TypeError: `precision_recall` gave None as the precision, at a threshold where no pair counts as a paraphrase, and None cannot be compared with a number. Which thresholds does your loop give `precision_recall`?",
                    nl="Met `{call}` stopt je functie met een TypeError: `precision_recall` gaf None als precision, bij een threshold waar geen enkel paar als parafrase telt, en None is niet met een getal te vergelijken. Welke thresholds geeft je loop aan `precision_recall`?"),
            Mistake("return-in-loop", matches=_bare_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: a `return` stands straight in the loop, so your function gives back the first threshold it tries. Under which line should the `return` stand, so that it runs only for a threshold that is good enough?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er staat een `return` direct in de loop, dus je functie geeft de eerste threshold terug die de functie probeert. Onder welke regel hoort de `return`, zodat de `return` alleen draait voor een threshold die goed genoeg is?"),
            Mistake("highest", wrong=_LT_HIGH,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the highest threshold with at least that precision, not the lowest. In which order does your loop go through the thresholds, and when does it stop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de hoogste threshold met minstens die precision terug, niet de laagste. In welke volgorde gaat je loop langs de thresholds, en wanneer stopt de loop?"),
            Mistake("unsorted", wrong=_LT_UNSORTED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the first threshold that is good enough in the order of the pairs, and that need not be the lowest. In which order should your loop try the thresholds?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de eerste threshold terug die goed genoeg is in de volgorde van de paren, en dat hoeft niet de laagste te zijn. In welke volgorde moet je loop de thresholds proberen?"),
            Mistake("precision", wrong=_LT_PRECISION,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the precision of the threshold it found, not the threshold. Which of the two does the exercise ask for?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de precision terug van de threshold die de functie vond, niet de threshold. Welke van de twee vraagt de oefening?"),
            Mistake("strict", wrong=_LT_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: in your function, a threshold whose precision is exactly the target is not good enough. Does a precision equal to the target reach it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je functie is een threshold met een precision van precies het doel niet goed genoeg. Haalt een precision die gelijk is aan het doel dat doel?"),
            Mistake("recall", wrong=_LT_RECALL,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the recall, and at the lowest threshold the recall is 1. Which of the two shares has to reach the target?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt naar de recall, en bij de laagste threshold is de recall 1. Welke van de twee delen moet het doel halen?"),
            Mistake("target-as-threshold", wrong=_LT_TARGET,
                    en="`{call}` gives {got}, and it should give {want}: your function asks `precision_recall` for the precision at the target, and not at the threshold of each round. What should the third value of that call be?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vraagt `precision_recall` om de precision bij het doel, en niet bij de threshold van elke ronde. Wat moet de derde waarde van die aanroep zijn?"),
            Mistake("not-a-cosine", matches=_not_a_cosine,
                    en="`{call}` gives {got}, and it should give {want}: no pair has the cosine similarity {got}. Which thresholds does your loop try: the numbers of `cos` itself, or numbers of its own?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: geen enkel paar heeft de cosine similarity {got}. Welke thresholds probeert je loop: de getallen van `cos` zelf, of eigen getallen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: try the cosine similarities of the pairs as thresholds, and give back the lowest one whose precision reaches `target`, or `None` when none does.",
                      "nl": "Vervang de `...`: probeer de cosine similarities van de paren als threshold, en geef de laagste terug waarvan de precision `target` haalt, of `None` als geen enkele dat doet."},
            "unreached": {"en": "With `{call}`, your function gives None, though a threshold with that precision exists. Does your loop try every threshold, and does a `return` give back the first one that is good enough?",
                          "nl": "Met `{call}` geeft je functie None, terwijl er een threshold met die precision is. Probeert je loop elke threshold, en geeft een `return` de eerste terug die goed genoeg is?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back: it has no `return` with a value. What should it give back once a threshold is good enough?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat geen `return` met een waarde in. Wat moet de functie teruggeven zodra een threshold goed genoeg is?"},
            "slow": {"en": "Your function is still running after 3 seconds on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 3 seconden nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function try the thresholds from low to high, and give back the first one whose precision reaches the target?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Probeert je functie de thresholds van laag naar hoog, en geeft de functie de eerste terug waarvan de precision het doel haalt?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work when no threshold reaches the target, and on the 1,375 pairs of the validation set?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook als geen enkele threshold het doel haalt, en op de 1.375 paren van de validation set?"},
            "correct": {"en": "Your `lowest_threshold` is right in all {n} cases, also on the validation set of the English pairs and of the Dutch translations. Look at the picture: what does a precision of 0.9 cost in recall, and what does the test set give?",
                        "nl": "Je `lowest_threshold` klopt in alle {n} gevallen, ook op de validation set van de Engelse paren en van de Nederlandse vertalingen. Kijk naar de afbeelding: wat kost een precision van 0,9 aan recall, en wat geeft de test set?"},
        },
    ),
}

CASES = [
    ("sts-scale", {}, "empty", None),
    ("sts-scale", {"train": "high", "cat": "high", "snow": "high", "store": "middle", "window": "middle",
                   "sold": "middle", "museum": "middle", "soup": "zero", "bread": "zero"}, "correct", None),
    ("sts-scale", {"train": "high", "cat": "high"}, "wrong", "unfinished"),
    ("sts-scale", {"train": "middle"}, "wrong", "wrong:train"),
    ("sts-scale", {"cat": "middle"}, "wrong", "wrong:cat"),
    ("sts-scale", {"snow": "middle"}, "wrong", "wrong:snow"),
    ("sts-scale", {"store": "high"}, "wrong", "wrong:store"),
    ("sts-scale", {"window": "high"}, "wrong", "wrong:window"),
    ("sts-scale", {"sold": "high"}, "wrong", "wrong:sold"),
    ("sts-scale", {"museum": "zero"}, "wrong", "wrong:museum"),
    ("sts-scale", {"soup": "middle"}, "wrong", "wrong:soup"),
    ("sts-scale", {"bread": "middle"}, "wrong", "wrong:bread"),
    ("precision_recall", _PR_START, "empty", None),
    ("precision_recall", _PR, "correct", None),
    ("precision_recall", _PR_HEAD + "    predicted = cos >= threshold\n    paraphrase = score >= 4\n"
                         "    tp = np.sum(predicted & paraphrase)\n    k = np.sum(predicted)\n    n = np.sum(paraphrase)\n"
                         "    precision = tp / k if k > 0 else None\n    recall = tp / n if n > 0 else None\n"
                         "    return precision, recall\n", "correct", None),
    ("precision_recall", _PR_HEAD + "    tp = fp = fn = 0\n    for c, s in zip(cos, score):\n        if c >= threshold and s >= 4:\n"
                         "            tp += 1\n        elif c >= threshold:\n            fp += 1\n        elif s >= 4:\n            fn += 1\n"
                         "    precision = None if tp + fp == 0 else tp / (tp + fp)\n"
                         "    recall = None if tp + fn == 0 else tp / (tp + fn)\n    return [precision, recall]\n", "correct", None),
    ("precision_recall", _PR_HEAD + "    para = cos[score >= 4]\n    other = cos[score < 4]\n"
                         "    tp = len(para[para >= threshold])\n    fp = len(other[other >= threshold])\n"
                         "    return tp / (tp + fp), tp / len(para)\n", "error", "zero-division"),
    ("precision_recall", _PR_HEAD + _PR_SPLIT + _PR_COUNT + _PR_NONE + "    return precision\n", "wrong", "not-a-pair"),
    ("precision_recall", _PR_SWAP, "wrong", "swapped"),
    ("precision_recall", _PR_STRICT, "wrong", "strict"),
    ("precision_recall", _PR_ABOVE, "wrong", "above-4"),
    ("precision_recall", _PR_ZEROS, "wrong", "zeros"),
    ("precision_recall", _PR_NAN, "wrong", "nan"),
    ("precision_recall", _PR_NAN_RECALL, "wrong", "nan-recall"),
    ("precision_recall", _PR_SAME, "wrong", "same-share"),
    ("precision_recall", _PR_FN_OTHER, "wrong", "negatives"),
    ("precision_recall", _PR_ALL, "wrong", "all-pairs"),
    ("precision_recall", _PR_HEAD + _PR_SPLIT + _PR_COUNT + "    if tp + fp > 0:\n        return tp / (tp + fp), tp / (tp + fn)\n",
     "wrong", "unreached"),
    ("precision_recall", _PR_HEAD + _PR_SPLIT + _PR_COUNT + "    precision = tp / (tp + fp)\n", "wrong", "no-return"),
    ("precision_recall", _PR_HEAD + "    ...\n    return None, None\n", "wrong", "dots-left"),
    ("precision_recall", _PR.replace("    precision = None\n", "    precision = 0\n"), "wrong", "zeros-start"),
    ("precision_recall", _PR.replace("cos[score >= 4]", "cos[score > 4]"), "wrong", "above-4-text"),
    ("lowest_threshold", _LT_START, "empty", None),
    ("lowest_threshold", _LT, "correct", None),
    ("lowest_threshold", _LT_HEAD + "    for t in sorted(set(cos)):\n        if precision_recall(cos, score, t)[0] >= target:\n"
                         "            return float(t)\n", "correct", None),
    ("lowest_threshold", _LT_HEAD + "    import numpy as np\n    order = np.sort(cos)\n    k = 0\n    while k < len(order):\n"
                         "        p, r = precision_recall(cos, score, order[k])\n        if p >= target:\n            return order[k]\n"
                         "        k += 1\n    return None\n", "correct", None),
    ("lowest_threshold", _LT_HEAD + "    import numpy as np\n    for t in np.arange(1.0, 0.0, -0.01):\n"
                         "        p, r = precision_recall(cos, score, t)\n        if p >= target:\n            return t\n"
                         "    return None\n", "error", "none-compare"),
    ("lowest_threshold", _LT_BARE, "wrong", "return-in-loop"),
    ("lowest_threshold", _LT_HIGH, "wrong", "highest"),
    ("lowest_threshold", _LT_LAST, "wrong", "highest"),
    ("lowest_threshold", _LT_UNSORTED, "wrong", "unsorted"),
    ("lowest_threshold", _LT_PRECISION, "wrong", "precision"),
    ("lowest_threshold", _LT_STRICT, "wrong", "strict"),
    ("lowest_threshold", _LT_RECALL, "wrong", "recall"),
    ("lowest_threshold", _LT_TARGET, "wrong", "target-as-threshold"),
    ("lowest_threshold", _LT_HEAD + "    for t in sorted(cos):\n        precision, recall = precision_recall(cos, score, t)\n"
                         "        if precision >= target:\n            t\n    return None\n", "wrong", "unreached"),
    ("lowest_threshold", _LT_HEAD + "    for t in sorted(cos):\n        precision, recall = precision_recall(cos, score, t)\n",
     "wrong", "no-return"),
    ("lowest_threshold", _LT_HEAD + "    ...\n    return None\n", "wrong", "dots-left"),
    ("lowest_threshold", _LT_HEAD + "    import numpy as np\n    for t in np.arange(0, 1, 0.01):\n"
                         "        precision, recall = precision_recall(cos, score, t)\n"
                         "        if precision is not None and precision >= target:\n            return t\n    return None\n",
     "wrong", "not-a-cosine"),
]
