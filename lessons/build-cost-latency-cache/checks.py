# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-cost-latency-cache: what version 2 of the bank's assistant (build-eval-set,
data/course-production-qwen.json) took and cost for each of the 120 test questions, the embeddings Granite made of
those questions (data/course-banking-granite.npz), and a cache of answers in front of the model: an exact cache by the
text, a semantic cache by meaning. Three functions to write (the slowest time but one in twenty, the hits of an exact
cache, the closest earlier question), a choice, and the pictures."""
import functools
import html
import importlib.util
import json

import numpy as np

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _eval_set():
    spec = importlib.util.spec_from_file_location("build_eval_set_checks_for_cache", paths.LESSONS / "build-eval-set" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _eval_set()
E = V.E
DATA = V.DATA
QUESTIONS, LABELS, Named = V.QUESTIONS, V.LABELS, V.Named
VECTORS_FILE = paths.ROOT / "data" / "course-banking-granite.npz"
PRICE_IN, PRICE_OUT = 3, 15   # made-up prices, in dollars per million tokens, as in build-tokens-prices
THRESHOLD = 0.85              # the course's cache reuses an answer from this similarity on
TRIED = (0.75, 0.8, 0.85, 0.9, 0.95)


@functools.cache
def _answers() -> tuple:
    """Version 2's answer to each test question, as recorded for build-eval-set: its text, seconds and tokens."""
    return tuple(json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-eval-set"]["v2"]["answers"])


@functools.cache
def _granite() -> tuple:
    z = np.load(VECTORS_FILE)
    return z["test"].astype(np.float64), tuple(float(s) for s in z["seconds"]), json.loads(str(z["about"]))


def calls() -> list:
    """The 120 calls to the model, in the order of the questions: what came in, what it took and what it wrote."""
    return [{"question": q, "seconds": a["seconds"], "prompt_tokens": a["prompt_tokens"], "output_tokens": a["tokens"], "reply": a["text"]}
            for q, a in zip(QUESTIONS, _answers())]


CALLS = Named(calls(), "CALLS")
SECONDS = Named([c["seconds"] for c in CALLS], "SECONDS")
VECTORS = _granite()[0]                                 # VECTORS[i] is the embedding of QUESTIONS[i], of length 1
EMBED_SECONDS = Named(list(_granite()[1]), "EMBED_SECONDS")   # each question embedded in a request of its own
TRAIN = tuple(text for _, text in E._banking()["train"])     # the 300 other questions of banking77-small


def passes(text: str, topic: str) -> bool:
    """The eval set of build-eval-set on one reply for a question about topic."""
    return all(fn(text, topic) for fn in V.CHECKS.values())


def dollars(prompt_tokens: int, output_tokens: int) -> float:
    return prompt_tokens * PRICE_IN / 1_000_000 + output_tokens * PRICE_OUT / 1_000_000


def p95(seconds):
    in_order = sorted(seconds)
    return in_order[int(len(seconds) * 0.95)]


def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        if key in seen:
            hits = hits + 1
        seen.add(key)
    return hits


def closest(vector, vectors):
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return best, float(scores[best])


def run_cache(threshold: float, find=closest) -> dict:
    """The questions come in one after another. Each looks for the closest earlier question the model answered; from
    `threshold` on it gets that question's answer, otherwise the model answers it and the answer is stored. `find` is
    closest or a learner's version of it; a call that fails or gives something else counts in `failed`."""
    X = VECTORS
    stored, hits, failed = [], [], 0
    for i in range(len(QUESTIONS)):
        if stored:
            got = _safe(find, X[i], X[stored])
            ok = (isinstance(got, tuple) and len(got) == 2 and isinstance(got[0], (int, np.integer)) and not isinstance(got[0], bool)
                  and 0 <= got[0] < len(stored) and isinstance(got[1], (int, float, np.integer, np.floating)) and not isinstance(got[1], bool))
            if not ok:
                failed += 1
            elif got[1] >= threshold:
                hits.append((i, stored[int(got[0])], float(got[1])))
                continue
        stored.append(i)
    c = CALLS
    hit = [i for i, _, _ in hits]
    return {
        "hits": hits, "failed": failed, "stored": stored,
        "other": [(i, j, s) for i, j, s in hits if LABELS[i] != LABELS[j]],
        "seconds": sum(c[i]["seconds"] for i in hit),
        "prompt_tokens": sum(c[i]["prompt_tokens"] for i in hit), "output_tokens": sum(c[i]["output_tokens"] for i in hit),
        "cached_pass": sum(passes(c[j]["reply"], LABELS[i]) for i, j, _ in hits),
        "fresh_pass": sum(passes(c[i]["reply"], LABELS[i]) for i in hit),
    }


@functools.cache
def findings() -> dict:
    c = CALLS
    s = sorted(SECONDS)
    pin, pout = sum(x["prompt_tokens"] for x in c), sum(x["output_tokens"] for x in c)
    runs = {t: run_cache(t) for t in TRIED}
    return {
        "n": len(c), "median": float(np.median(SECONDS)), "p95": p95(SECONDS), "slowest": s[-1], "fastest": s[0],
        "mean": sum(SECONDS) / len(c), "total": sum(SECONDS),
        "prompt_tokens": pin, "output_tokens": pout, "dollars": dollars(pin, pout), "per_1000": dollars(pin, pout) / len(c) * 1000,
        "threads": _granite()[2]["threads"], "embed_median": float(np.median(EMBED_SECONDS)), "embed_p95": p95(EMBED_SECONDS), "embed_together": _granite()[2]["together_seconds"],
        "exact": exact_hits(QUESTIONS), "exact_all": exact_hits(list(TRAIN) + list(QUESTIONS)),
        "exact_twice": exact_hits(list(QUESTIONS) + list(QUESTIONS)),
        "runs": runs,
        "fresh_all": sum(passes(x["reply"], LABELS[i]) for i, x in enumerate(c)),
    }


# --- pictures ------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"


def num(x: float, lang: str, digits: int = 0) -> str:
    """A number for the pictures: 1,000.5 in English, 1.000,5 in Dutch."""
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


_WORDS = {
    "en": {"flow": ["question", "exact cache", "semantic cache", "model", "answer"],
           "hit": "hit", "miss": "miss", "store": "store the answer",
           "flow_title": "A cache of answers in front of the model",
           "flow_alt": "Five boxes in a row: the question, the exact cache, the semantic cache, the model and the answer. From each of the two caches a line marked hit goes straight to the answer; a miss goes on to the next box. Below, a dashed line from the model back to the two caches: the answer is stored.",
           "p95_title": "Your p95(seconds) on the times of the bank's assistant",
           "p95_alt": "Two rows of dots on one time axis in seconds: the {n} calls to the model, and the same {n} questions embedded for the cache. In each row a line at the median and a line at the value your function gives.",
           "p95_rows": ["calls to the model", "embedding a question"],
           "median": "median", "yours": "your p95", "axis": "seconds",
           "p95_sum": "Your function gives {a} for the calls to the model and {b} for embedding a question.",
           "error": "error", "no_text": "no number",
           "exact_title": "Your exact_hits(questions) on three streams of questions",
           "exact_alt": "Three bars, one for each stream of questions: the bar shows how many of its questions your function counts as hits.",
           "exact_rows": ["the 120 test questions, one after another", "the 300 training questions, then the 120 test questions",
                          "the 120 test questions, and then the same 120 again"],
           "exact_sum": "Your function counts {a}, {b} and {c} hits.",
           "exact_failed": "For {k} of the three streams your function gives no number.",
           "close_title": "The course's semantic cache, with your closest(vector, vectors)",
           "close_alt": "A row of {n} squares, one for each question in the order they came in: empty where the model answered, filled where the cache gave an earlier answer, and filled in the loss colour where that earlier question was about another topic. Below, those questions with the earlier question whose answer they got.",
           "close_sum": "At a similarity of {t} or more, the cache answered {h} of the {n} questions: {o} of them got the answer to a question about another topic.",
           "close_failed": "For {f} questions your function gave no index of a cached question with its similarity.",
           "key_model": "the model answered", "key_hit": "answer from the cache", "key_other": "answer for another topic",
           "got": "got the answer to", "sim": "similarity",
           "thr_title": "The cache at a similarity of {t} or more",
           "thr_alt": "A table with what the cache did at the chosen threshold: the questions it answered, those that got the answer to a question about another topic, the seconds and tokens it saved, and how many of its answers the eval set passed against the model's own answers to the same questions.",
           "thr_head": ["", "at this threshold"],
           "thr_rows": ["questions the cache answered, of {n}", "of them, with the answer for another topic", "seconds of the model saved, of {s}",
                        "tokens saved, of {k}", "the eval set passed the cache's answers", "the eval set passed the model's own answers to them"],
           },
    "nl": {"flow": ["vraag", "exact cache", "semantic cache", "model", "antwoord"],
           "hit": "hit", "miss": "miss", "store": "antwoord bewaren",
           "flow_title": "Een cache met antwoorden vóór het model",
           "flow_alt": "Vijf vakken op een rij: de vraag, de exact cache, de semantic cache, het model en het antwoord. Uit elk van de twee caches gaat een lijn met hit rechtstreeks naar het antwoord; een miss gaat door naar het volgende vak. Onderaan een gestippelde lijn van het model terug naar de twee caches: het antwoord wordt bewaard.",
           "p95_title": "Jouw p95(seconds) op de tijden van de assistent van de bank",
           "p95_alt": "Twee rijen stippen op één tijdas in seconden: de {n} calls naar het model, en dezelfde {n} vragen als embedding voor de cache. In elke rij een lijn bij de mediaan en een lijn bij de waarde die je functie geeft.",
           "p95_rows": ["calls naar het model", "een vraag als embedding"],
           "median": "mediaan", "yours": "jouw p95", "axis": "seconden",
           "p95_sum": "Je functie geeft {a} voor de calls naar het model en {b} voor een vraag als embedding.",
           "error": "fout", "no_text": "geen getal",
           "exact_title": "Jouw exact_hits(questions) op drie reeksen vragen",
           "exact_alt": "Drie balken, één voor elke reeks vragen: de balk laat zien hoeveel van de vragen je functie als hit telt.",
           "exact_rows": ["de 120 testvragen, de een na de ander", "de 300 trainingsvragen, dan de 120 testvragen",
                          "de 120 testvragen, en daarna dezelfde 120 nog eens"],
           "exact_sum": "Je functie telt {a}, {b} en {c} hits.",
           "exact_failed": "Bij {k} van de drie reeksen geeft je functie geen getal.",
           "close_title": "De semantic cache van de course, met jouw closest(vector, vectors)",
           "close_alt": "Een rij van {n} vakjes, één voor elke vraag in de volgorde waarin ze binnenkwamen: leeg waar het model antwoordde, gevuld waar de cache een eerder antwoord gaf, en gevuld in de kleur van de loss waar die eerdere vraag over een ander onderwerp ging. Daaronder die vragen met de eerdere vraag waarvan ze het antwoord kregen.",
           "close_sum": "Bij een similarity van {t} of meer beantwoordde de cache {h} van de {n} vragen: {o} daarvan kregen het antwoord op een vraag over een ander onderwerp.",
           "close_failed": "Bij {f} vragen gaf je functie geen index van een vraag in de cache met de similarity erbij.",
           "key_model": "het model antwoordde", "key_hit": "antwoord uit de cache", "key_other": "antwoord voor een ander onderwerp",
           "got": "kreeg het antwoord op", "sim": "similarity",
           "thr_title": "De cache bij een similarity van {t} of meer",
           "thr_alt": "Een tabel met wat de cache deed bij de gekozen drempel: de vragen die hij beantwoordde, de vragen die het antwoord op een vraag over een ander onderwerp kregen, de seconden en tokens die hij bespaarde, en hoeveel van zijn antwoorden de eval set goedkeurde tegenover de eigen antwoorden van het model op dezelfde vragen.",
           "thr_head": ["", "bij deze drempel"],
           "thr_rows": ["vragen die de cache beantwoordde, van de {n}", "daarvan met het antwoord voor een ander onderwerp", "seconden van het model bespaard, van de {s}",
                        "tokens bespaard, van de {k}", "de eval set keurde de antwoorden van de cache goed", "de eval set keurde de eigen antwoorden van het model op die vragen goed"],
           },
}


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def flow_picture(lang: str) -> str:
    """The idea: an exact cache and a semantic cache in front of the model; a miss goes on, the model's answer is stored."""
    w = _WORDS[lang]
    e = html.escape
    box_w, gap, y = 128, 26, 30
    xs = [6 + k * (box_w + gap) for k in range(5)]
    parts = []
    for k, (x, label) in enumerate(zip(xs, w["flow"])):
        style = "fill:var(--ea-data-wash);stroke:var(--ea-data)" if k in (1, 2) else ("fill:var(--ea-inset);stroke:var(--ea-ink-2)" if k == 3 else "fill:none;stroke:var(--ea-ink-2)")
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="40" rx="6" style="{style};stroke-width:1"/>'
                     f'<text x="{x + box_w / 2}" y="{y + 25}" text-anchor="middle" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(label)}</text>')
        if k < 4:
            parts.append(f'<line x1="{x + box_w}" y1="{y + 20}" x2="{x + box_w + gap - 3}" y2="{y + 20}" style="stroke:var(--ea-ink-2);stroke-width:1"/>'
                         f'<path d="M{x + box_w + gap - 3},{y + 16} l4,4 l-4,4 z" style="fill:var(--ea-ink-2)"/>')
            if k in (1, 2):
                parts.append(f'<text x="{x + box_w + gap / 2}" y="{y + 36}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["miss"])}</text>')
    ax = xs[4] + box_w / 2
    for k, top in ((1, 6), (2, 16)):
        cx = xs[k] + box_w / 2
        lx = ax + (k - 1.5) * 12   # each hit line ends at its own place on the answer box
        parts.append(f'<path d="M{cx},{y} V{top} H{lx} V{y - 4}" style="fill:none;stroke:var(--ea-data);stroke-width:1"/>'
                     f'<path d="M{lx - 4},{y - 5} l4,5 l4,-5 z" style="fill:var(--ea-data)"/>'
                     f'<text x="{cx + 6}" y="{top + 11}" style="{_UI};font-size:11px;fill:var(--ea-data-text)">{e(w["hit"])}</text>')
    by = y + 64
    mx = xs[3] + box_w / 2
    parts.append(f'<path d="M{mx},{y + 40} V{by} H{xs[1] + box_w / 2} V{y + 44}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1;stroke-dasharray:3 3"/>'
                 f'<path d="M{xs[1] + box_w / 2 - 4},{y + 45} l4,-5 l4,5 z" style="fill:var(--ea-ink-2)"/>'
                 f'<line x1="{xs[2] + box_w / 2}" y1="{by}" x2="{xs[2] + box_w / 2}" y2="{y + 44}" style="stroke:var(--ea-ink-2);stroke-width:1;stroke-dasharray:3 3"/>'
                 f'<path d="M{xs[2] + box_w / 2 - 4},{y + 45} l4,-5 l4,5 z" style="fill:var(--ea-ink-2)"/>'
                 f'<text x="{(xs[2] + mx) / 2 + box_w / 4}" y="{by + 14}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["store"])}</text>')
    width = xs[-1] + box_w + 6
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} {by + 22}" width="100%" style="min-width:560px;max-width:{width}px" '
           f'role="img" aria-label="{e(w["flow_alt"])}">{"".join(parts)}</svg></div>')
    return yours.figure(w["flow_title"], svg, alt=w["flow_alt"])


def p95_picture(seen) -> str:
    fn = _mine(seen, "p95")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    lang = seen.lang
    rows = [(list(SECONDS), w["p95_rows"][0]), (list(EMBED_SECONDS), w["p95_rows"][1])]
    top = 12.0
    left, width, row_h = 150, 560, 46
    parts, got = [], []
    for r, (values, label) in enumerate(rows):
        y = 14 + r * row_h
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(label)}</text>'
                     f'<line x1="{left}" y1="{y + 12}" x2="{left + width}" y2="{y + 12}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for k, v in enumerate(values):
            x = left + min(v, top) / top * width
            parts.append(f'<circle cx="{x:.1f}" cy="{y + 12 + (k % 5 - 2) * 2.4:.1f}" r="2.2" style="fill:var(--ea-ink-3)"/>')
        med = float(np.median(values))
        xm = left + med / top * width
        parts.append(f'<line x1="{xm:.1f}" y1="{y}" x2="{xm:.1f}" y2="{y + 24}" style="stroke:var(--ea-ink-2);stroke-width:1.5;stroke-dasharray:3 2"/>')
        mine = _safe(fn, list(values))
        ok = isinstance(mine, (int, float, np.integer, np.floating)) and not isinstance(mine, bool)
        got.append(f"{num(float(mine), lang, 2 if mine >= 0.1 else 3)} s" if ok else w["no_text"])
        if ok:
            xp = left + min(max(float(mine), 0.0), top) / top * width
            parts.append(f'<line x1="{xp:.1f}" y1="{y - 2}" x2="{xp:.1f}" y2="{y + 26}" style="stroke:var(--ea-data);stroke-width:2"/>')
    ay = 14 + len(rows) * row_h
    for t in range(0, int(top) + 1, 2):
        x = left + t / top * width
        parts.append(f'<line x1="{x}" y1="{ay - 4}" x2="{x}" y2="{ay}" style="stroke:var(--ea-axis);stroke-width:1"/>'
                     f'<text x="{x}" y="{ay + 12}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{t}</text>')
    parts.append(f'<text x="{left + width / 2}" y="{ay + 28}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["axis"])}</text>')
    key = (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.2rem">'
           f'<span style="display:inline-block;width:14px;border-top:2px solid var(--ea-data);vertical-align:middle"></span> {e(w["yours"])} &nbsp; '
           f'<span style="display:inline-block;width:14px;border-top:1.5px dashed var(--ea-ink-2);vertical-align:middle"></span> {e(w["median"])}</div>')
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {left + width + 20} {ay + 34}" width="100%" style="min-width:520px;max-width:{left + width + 20}px" '
           f'aria-hidden="true">{"".join(parts)}</svg></div>')
    summary = f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.4rem">{e(w["p95_sum"].format(a=got[0], b=got[1]))}</div>'
    return yours.figure(w["p95_title"], f'<div role="img" aria-label="{e(w["p95_alt"].format(n=len(SECONDS)))}">{svg}{key}{summary}</div>')


def exact_picture(seen) -> str:
    fn = _mine(seen, "exact_hits")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    streams = [list(QUESTIONS), list(TRAIN) + list(QUESTIONS), list(QUESTIONS) + list(QUESTIONS)]
    rows, counts = "", []
    for label, stream in zip(w["exact_rows"], streams):
        got = _safe(fn, stream)
        ok = isinstance(got, (int, np.integer)) and not isinstance(got, bool)
        counts.append(str(int(got)) if ok else "—")
        share = min(max(int(got), 0), len(stream)) / len(stream) if ok else 0
        rows += (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0">'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink);flex:0 0 19rem">{e(label)}</span>'
                 f'<span style="flex:1;background:var(--ea-inset);border-radius:3px;height:12px;position:relative">'
                 f'<span style="position:absolute;left:0;top:0;bottom:0;width:{share * 100:.1f}%;background:var(--ea-data);border-radius:3px"></span></span>'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink);flex:0 0 5.5rem;text-align:right">{e(counts[-1])} / {len(stream)}</span></div>')
    failed = counts.count("—")
    said = w["exact_failed"].format(k=failed) if failed else w["exact_sum"].format(a=counts[0], b=counts[1], c=counts[2])
    summary = f'<div style="{_UI};font-size:13px;color:{"var(--ea-loss-text)" if failed else "var(--ea-ink)"};margin-top:0.4rem">{e(said)}</div>'
    return yours.figure(w["exact_title"], f'<div role="img" aria-label="{e(w["exact_alt"])}" style="overflow-x:auto"><div style="min-width:30rem">{rows}</div>{summary}</div>')


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def cache_html(run: dict, lang: str, threshold: float) -> str:
    """A square for each question in the order it came in, and the questions that got an answer for another topic."""
    w = _WORDS[lang]
    e = html.escape
    hit = {i: j for i, j, _ in run["hits"]}
    other = {i for i, _, _ in run["other"]}
    cells = ""
    for i in range(len(QUESTIONS)):
        if i in other:
            style = "background:var(--ea-loss);border:1px solid var(--ea-loss)"
        elif i in hit:
            style = "background:var(--ea-data);border:1px solid var(--ea-data)"
        else:
            style = "background:none;border:1px solid var(--ea-ink-3)"
        cells += f'<span style="display:inline-block;width:10px;height:10px;margin:1px;border-radius:2px;{style}"></span>'
    key = "".join(f'<span style="margin-right:1rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:0.3rem;{s}"></span>{e(t)}</span>'
                  for s, t in (("border:1px solid var(--ea-ink-3)", w["key_model"]), ("background:var(--ea-data)", w["key_hit"]), ("background:var(--ea-loss)", w["key_other"])))
    lines = ""
    for i, j, s in run["other"]:
        lines += (f'<div style="{_UI};font-size:12px;color:var(--ea-ink);margin:0.35rem 0">'
                  f'<span style="{_MONO};color:var(--ea-loss-text)">{i}</span> “{e(_short(QUESTIONS[i]))}” ({e(LABELS[i])}) {e(w["got"])} '
                  f'<span style="{_MONO}">{j}</span> “{e(_short(QUESTIONS[j]))}” ({e(LABELS[j])}), {e(w["sim"])} {num(s, lang, 2)}</div>')
    t = num(threshold, lang, 2)
    summary = w["close_sum"].format(t=t, h=len(run["hits"]), n=len(QUESTIONS), o=len(run["other"]))
    if run["failed"]:
        summary += " " + w["close_failed"].format(f=run["failed"])
    color = "var(--ea-loss-text)" if run["failed"] else "var(--ea-ink)"
    return (f'<div style="line-height:0">{cells}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.4rem">{key}</div>'
            f'<div style="{_UI};font-size:13px;color:{color};margin-top:0.5rem">{e(summary)}</div>{lines}')


def closest_picture(seen) -> str:
    fn = _mine(seen, "closest")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    run = run_cache(THRESHOLD, fn)
    alt = w["close_alt"].format(n=len(QUESTIONS))
    return yours.figure(w["close_title"], f'<div role="img" aria-label="{html.escape(alt)}">{cache_html(run, seen.lang, THRESHOLD)}</div>')


def threshold_picture(threshold: float, lang: str) -> str:
    """The demo: what the course's cache does at a chosen threshold."""
    w = _WORDS[lang]
    e = html.escape
    t = round(float(threshold), 2)
    run = run_cache(t)
    f = findings()
    k = len(run["hits"])
    tok = run["prompt_tokens"] + run["output_tokens"]
    labels = w["thr_rows"]
    rows = [
        (labels[0].format(n=f["n"]), str(k)),
        (labels[1], str(len(run["other"]))),
        (labels[2].format(s=num(f["total"], lang, 1)), num(run["seconds"], lang, 1)),
        (labels[3].format(k=num(f["prompt_tokens"] + f["output_tokens"], lang)), num(tok, lang)),
        (labels[4], f'{run["cached_pass"]} / {k}'),
        (labels[5], f'{run["fresh_pass"]} / {k}'),
    ]
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.25rem 0.5rem;border-top:1px solid var(--ea-line)"
    body = "".join(f'<tr><td style="{cell}">{e(a)}</td><td style="{cell};text-align:right">{e(b)}</td></tr>' for a, b in rows)
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;max-width:34rem">{body}</table></div>'
    return yours.figure(w["thr_title"].format(t=num(t, lang, 2)),
                        f'<div role="img" aria-label="{e(w["thr_alt"])}">{table}<div style="margin-top:0.6rem">{cache_html(run, lang, t)}</div></div>')


# --- exercises -----------------------------------------------------------------------------------------------------

START_P95 = '''def p95(seconds):
    in_order = sorted(seconds)
    return in_order[0]
'''
SOLUTION_P95 = '''def p95(seconds):
    in_order = sorted(seconds)
    return in_order[int(len(seconds) * 0.95)]
'''
_P95_UNSORTED = SOLUTION_P95.replace("return in_order[", "return seconds[")
_P95_MEAN = START_P95.replace("return in_order[0]", "return sum(seconds) / len(seconds)")
_P95_MAX = START_P95.replace("return in_order[0]", "return in_order[-1]")
_P95_MEDIAN = START_P95.replace("return in_order[0]", "return in_order[len(seconds) // 2]")
_P95_QUANTILE = START_P95.replace("return in_order[0]", "return float(np.quantile(seconds, 0.95))")
_P95_ONE_LESS = SOLUTION_P95.replace("return in_order[int(len(seconds) * 0.95)]", "return in_order[int(len(seconds) * 0.95) - 1]")

START_EXACT = '''def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        seen.add(key)
    return hits
'''
SOLUTION_EXACT = '''def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        if key in seen:
            hits = hits + 1
        seen.add(key)
    return hits
'''
_EXACT_AFTER = '''def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        seen.add(key)
        if key in seen:
            hits = hits + 1
    return hits
'''
_EXACT_RAW = SOLUTION_EXACT.replace("if key in seen:", "if question in seen:")
_EXACT_NOT_IN = SOLUTION_EXACT.replace("if key in seen:", "if key not in seen:")

START_CLOSE = '''def closest(vector, vectors):
    scores = vectors @ vector
    best = 0
    return best, float(scores[best])
'''
SOLUTION_CLOSE = '''def closest(vector, vectors):
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return best, float(scores[best])
'''
_CLOSE_MIN = SOLUTION_CLOSE.replace("np.argmax", "np.argmin")
_CLOSE_LAST = START_CLOSE.replace("best = 0", "best = len(vectors) - 1")
_CLOSE_SCORE = SOLUTION_CLOSE.replace("return best, float(scores[best])", "return best, float(np.max(vectors))")

_NAMES = {"np": np, "SECONDS": SECONDS, "EMBED_SECONDS": EMBED_SECONDS, "QUESTIONS": QUESTIONS, "VECTORS": VECTORS}

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _texts(name: str, empty: dict, wrong: dict, correct: dict) -> dict:
    return {
        "missing": {"en": f"Run the cell with `{name}` first.", "nl": f"Voer eerst de cel met `{name}` uit."},
        "cleared": _CLEARED,
        "empty": empty,
        "wrong": wrong,
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "no_function": {"en": f"Your box does not make a function `{name}`. Keep the `def` line as it was.",
                        "nl": f"Je vak maakt geen functie `{name}`. Laat de `def`-regel staan zoals hij was."},
        "correct": correct,
    }


_TIMES_A = Named([0.8, 0.3, 2.5, 0.4, 0.6, 0.5, 0.9, 0.7, 1.1, 0.2, 0.6, 0.4, 0.5, 0.3, 0.7, 0.8, 0.6, 0.5, 0.4, 3.9, 0.6, 0.5, 0.7, 0.4],
                 "times_a")
_TIMES_B = Named([round(1 + (k * 37 % 41) / 10, 1) for k in range(60)], "times_b")
_QS_A = Named(["Where is my card?", "where is my card?", "My PIN is blocked.", "Where is my card? ", "my pin is blocked."], "questions_a")
_QS_B = Named(["What rate do you use?", "Is my card lost?", "What rate do you use for euros?", "Is my card lost?"], "questions_b")
_QS_C = Named(["Hi", "hi", "HI", " hi "], "questions_c")
_PLANE = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]])
_EXACT_COPY = Named(list(QUESTIONS) + list(QUESTIONS), "QUESTIONS + QUESTIONS")


class _Shown(np.ndarray):
    """Vectors that a check message writes as the lesson names them, such as VECTORS[:21]. The learner's code gets a
    plain copy."""

    def __repr__(self):
        return getattr(self, "shown", "vectors")

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(rows, name: str) -> np.ndarray:
    v = np.array(rows, dtype=float).view(_Shown)
    v.shown = name
    return v


EXERCISES = {
    "p95": FunctionBox(
        name="p95",
        picture=p95_picture,
        function="p95",
        start=START_P95,
        solution=SOLUTION_P95,
        visible=[(SECONDS,)],
        hidden=[(EMBED_SECONDS,), (_TIMES_A,), (_TIMES_B,), (Named(SECONDS[:60], "SECONDS[:60]"),)],
        mistakes=[
            Mistake("unsorted", wrong=_P95_UNSORTED,
                    en="`{call}` gives `{got}`: the time at that place in the order the calls came in. Take it from `in_order`, the times sorted from short to long.",
                    nl="`{call}` geeft `{got}`: de tijd op die plek in de volgorde waarin de calls binnenkwamen. Neem hem uit `in_order`, de tijden gesorteerd van kort naar lang."),
            Mistake("mean", wrong=_P95_MEAN,
                    en="`{call}` gives `{got}`: the average time. p95 is a time from the sorted list, near its slow end.",
                    nl="`{call}` geeft `{got}`: de gemiddelde tijd. p95 is een tijd uit de gesorteerde lijst, vlak bij het langzame eind."),
            Mistake("max", wrong=_P95_MAX,
                    en="`{call}` gives `{got}`: the slowest time of all. p95 leaves the slowest one in twenty out: take the time 95 percent of the way through the sorted times.",
                    nl="`{call}` geeft `{got}`: de langzaamste tijd van allemaal. p95 laat de langzaamste één op de twintig weg: neem de tijd op 95 procent van de gesorteerde tijden."),
            Mistake("median", wrong=_P95_MEDIAN,
                    en="`{call}` gives `{got}`: the median, halfway through the sorted list. p95 lies at 95 percent of the way.",
                    nl="`{call}` geeft `{got}`: de mediaan, halverwege de gesorteerde lijst. p95 ligt op 95 procent van de lijst."),
            Mistake("quantile", wrong=_P95_QUANTILE,
                    en="`{call}` gives `{got}`: NumPy's 0.95 quantile, which lies between two times of the list. The p95 of this lesson is a time from the list itself: the one 95 percent of the way through the sorted times.",
                    nl="`{call}` geeft `{got}`: het 0,95-kwantiel van NumPy, dat tussen twee tijden van de lijst in ligt. De p95 van deze les is een tijd uit de lijst zelf: die op 95 procent van de gesorteerde tijden."),
            Mistake("one-less", wrong=_P95_ONE_LESS,
                    en="`{call}` gives `{got}`: one place too early. Places count from 0, so the time at place 114 of 120 has 114 times before it: 95 of every 100.",
                    nl="`{call}` geeft `{got}`: één plek te vroeg. Plekken tellen vanaf 0, dus de tijd op plek 114 van 120 heeft 114 tijden voor zich: 95 van elke 100."),
        ],
        texts=_texts("p95",
                     empty={"en": "Your `p95` gives the shortest time. Take the time at 95 percent of the sorted list.",
                            "nl": "Je `p95` geeft de kortste tijd. Neem de tijd op 95 procent van de gesorteerde lijst."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the time 95 percent of the way through the sorted times.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de tijd op 95 procent van de gesorteerde tijden."},
                     correct={"en": "Your `p95` is right in all {n} cases. Look at the two rows: how far apart are a call to the model and a question embedded for the cache?",
                              "nl": "Je `p95` klopt in alle {n} gevallen. Kijk naar de twee rijen: hoe ver liggen een call naar het model en een vraag als embedding voor de cache uit elkaar?"}),
        names=_NAMES,
    ),
    "exact_hits": FunctionBox(
        name="exact_hits",
        picture=exact_picture,
        function="exact_hits",
        start=START_EXACT,
        solution=SOLUTION_EXACT,
        visible=[(_QS_A,)],
        hidden=[(_QS_B,), (_QS_C,), (Named(list(QUESTIONS), "QUESTIONS"),), (_EXACT_COPY,)],
        mistakes=[
            Mistake("after", wrong=_EXACT_AFTER,
                    en="`{call}` gives `{got}`: you store the key before you look it up, so every question finds itself. Look up first, then store.",
                    nl="`{call}` geeft `{got}`: je bewaart de key voordat je hem opzoekt, dus elke vraag vindt zichzelf. Eerst opzoeken, dan bewaren."),
            Mistake("raw", wrong=_EXACT_RAW,
                    en="`{call}` gives `{got}`: you look up the question as it came in, but the cache keeps `key`, in small letters without spaces at the ends. Look up `key`.",
                    nl="`{call}` geeft `{got}`: je zoekt de vraag op zoals hij binnenkwam, maar de cache bewaart `key`, in kleine letters zonder spaties aan de randen. Zoek `key` op."),
            Mistake("not-in", wrong=_EXACT_NOT_IN,
                    en="`{call}` gives `{got}`: you count the questions that are new. A hit is a question whose key is already in `seen`.",
                    nl="`{call}` geeft `{got}`: je telt de vragen die nieuw zijn. Een hit is een vraag waarvan de key al in `seen` zit."),
        ],
        texts=_texts("exact_hits",
                     empty={"en": "Your `exact_hits` stores every key but never looks one up, so it counts 0. Before the `seen.add`, count a hit when `key` is already in `seen`.",
                            "nl": "Je `exact_hits` bewaart elke key maar zoekt er nooit een op, dus hij telt 0. Tel voor de `seen.add` een hit als `key` al in `seen` zit."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the questions whose key came before.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de vragen waarvan de key al eerder kwam."},
                     correct={"en": "Your `exact_hits` is right in all {n} cases. Look at the first two bars: why do they stay at 0?",
                              "nl": "Je `exact_hits` klopt in alle {n} gevallen. Kijk naar de eerste twee balken: waarom blijven ze op 0?"}),
        names=_NAMES,
    ),
    "closest": FunctionBox(
        name="closest",
        picture=closest_picture,
        function="closest",
        start=START_CLOSE,
        solution=SOLUTION_CLOSE,
        visible=[(_v(VECTORS[21], "VECTORS[21]"), _v(VECTORS[:21], "VECTORS[:21]"))],
        hidden=[(_v(VECTORS[78], "VECTORS[78]"), _v(VECTORS[:78], "VECTORS[:78]")),
                (_v(VECTORS[5], "VECTORS[5]"), _v(VECTORS[:5], "VECTORS[:5]")),
                (_v([0.8, 0.6], "np.array([0.8, 0.6])"), _v(_PLANE, "plane")),
                (_v([-0.6, 0.8], "np.array([-0.6, 0.8])"), _v(_PLANE, "plane"))],
        mistakes=[
            Mistake("min", wrong=_CLOSE_MIN,
                    en="`{call}` gives `{got}`: the question that is least alike. The closest has the highest score: `np.argmax`.",
                    nl="`{call}` geeft `{got}`: de vraag die het minst lijkt. De dichtstbijzijnde heeft de hoogste score: `np.argmax`."),
            Mistake("last", wrong=_CLOSE_LAST,
                    en="`{call}` gives `{got}`: the last question in the cache, whatever its score. Take the place of the highest score.",
                    nl="`{call}` geeft `{got}`: de laatste vraag in de cache, wat de score ook is. Neem de plek van de hoogste score."),
            Mistake("score", wrong=_CLOSE_SCORE,
                    en="`{call}` gives `{got}`: the right question, but the largest number in all of `vectors` as its score. Give `scores[best]`.",
                    nl="`{call}` geeft `{got}`: de goede vraag, maar het grootste getal in heel `vectors` als score. Geef `scores[best]`."),
        ],
        texts=_texts("closest",
                     empty={"en": "Your `closest` always gives the first question in the cache. Give the place of the highest score in `scores`.",
                            "nl": "Je `closest` geeft altijd de eerste vraag in de cache. Geef de plek van de hoogste score in `scores`."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the place of the highest score, with that score.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de plek van de hoogste score, met die score."},
                     correct={"en": "Your `closest` is right in all {n} cases. Read the questions under the squares: what did those customers get?",
                              "nl": "Je `closest` klopt in alle {n} gevallen. Lees de vragen onder de vakjes: wat kregen die klanten?"}),
        names=_NAMES,
    ),
}

REQUIRED = ["p95", "exact_hits", "closest", "stricter"]

CASES = [
    ("p95", START_P95, "empty", None),
    ("p95", "", "empty", None),
    ("p95", SOLUTION_P95, "correct", None),
    ("p95", '''def p95(seconds):
    n = len(seconds)
    return sorted(seconds)[int(n * 95 / 100)]
''', "correct", None),
    ("p95", _P95_UNSORTED, "wrong", "unsorted"),
    ("p95", _P95_MEAN, "wrong", "mean"),
    ("p95", _P95_MAX, "wrong", "max"),
    ("p95", _P95_MEDIAN, "wrong", "median"),
    ("p95", _P95_QUANTILE, "wrong", "quantile"),
    ("p95", _P95_ONE_LESS, "wrong", "one-less"),
    ("exact_hits", START_EXACT, "empty", None),
    ("exact_hits", "", "empty", None),
    ("exact_hits", SOLUTION_EXACT, "correct", None),
    ("exact_hits", '''def exact_hits(questions):
    keys = [q.lower().strip() for q in questions]
    return len(keys) - len(set(keys))
''', "correct", None),
    ("exact_hits", _EXACT_AFTER, "wrong", "after"),
    ("exact_hits", _EXACT_RAW, "wrong", "raw"),
    ("exact_hits", _EXACT_NOT_IN, "wrong", "not-in"),
    ("closest", START_CLOSE, "empty", None),
    ("closest", "", "empty", None),
    ("closest", SOLUTION_CLOSE, "correct", None),
    ("closest", '''def closest(vector, vectors):
    best = 0
    for i in range(len(vectors)):
        if np.dot(vectors[i], vector) > np.dot(vectors[best], vector):
            best = i
    return best, float(np.dot(vectors[best], vector))
''', "correct", None),
    ("closest", _CLOSE_MIN, "wrong", "min"),
    ("closest", _CLOSE_LAST, "wrong", "last"),
    ("closest", _CLOSE_SCORE, "wrong", "score"),
]
