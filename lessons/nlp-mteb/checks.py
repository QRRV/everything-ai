# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-mteb: which task type of MTEB looks like a use (a sort), the mean score of each model per
task type (type_means), and the model with the highest weighted mean for weights of your own (best_for); with the
pictures of what each gives on the scores of MTEB, two pictures of the idea and an explore.

The course ships the scores of 15 embedding models on the 41 tasks of MTEB(eng, v2), the English benchmark of the MTEB
leaderboard, with each model's number of parameters (data/mteb-eng-v2.tsv, made by data/tools/make_mteb_scores.py from
the results repository of MTEB, CC0 1.0), so nothing is downloaded. Which tasks each model was trained on comes from the
leaderboard's marks (TRAINED_ON). Every number in the text and the pictures is computed by the course
(tests/test_nlp_mteb.py computes them again)."""
import contextlib
import html
import io
import math
import re
import warnings

import numpy as np

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

_KEY = "nlp-mteb"
SCORES = paths.ROOT / "data" / "mteb-eng-v2.tsv"
SMALL = 300_000_000   # "up to 300 million parameters": the nine models of the idea

# the seven task types of MTEB(eng, v2), in the order of the lesson: the columns of type_means and the weights of best_for
KINDS = ["Retrieval", "Reranking", "Classification", "Clustering", "PairClassification", "STS", "Summarization"]

TYPE_NAMES = {
    "en": {"Retrieval": "retrieval", "Reranking": "reranking", "Classification": "classification",
           "Clustering": "clustering", "PairClassification": "pair classification", "STS": "STS",
           "Summarization": "summarization"},
    "nl": {"Retrieval": "retrieval", "Reranking": "reranking", "Classification": "classificatie",
           "Clustering": "clustering", "PairClassification": "paarclassificatie", "STS": "STS",
           "Summarization": "samenvatting"},
}
HEADS = {   # the short heads of the columns of a table
    "en": {"Retrieval": "Retr.", "Reranking": "Rerank", "Classification": "Class.", "Clustering": "Clust.",
           "PairClassification": "Pairs", "STS": "STS", "Summarization": "Summ."},
    "nl": {"Retrieval": "Retr.", "Reranking": "Rerank", "Classification": "Class.", "Clustering": "Clust.",
           "PairClassification": "Paren", "STS": "STS", "Summarization": "Samenv."},
}

# The tasks of MTEB(eng, v2) each model was trained on, as the MTEB leaderboard marks them (trainedOnTasks, read on
# 2026-11-23); the leaderboard's zero-shot share is the share of the 41 tasks a model was not trained on.
_GRANITE_TASKS = ("AskUbuntuDupQuestions", "FEVERHardNegatives", "HotpotQAHardNegatives", "StackExchangeClustering.v2",
                  "StackExchangeClusteringP2P.v2")
_FEVER_HOTPOT = ("FEVERHardNegatives", "HotpotQAHardNegatives")
TRAINED_ON = {
    "sentence-transformers/all-MiniLM-L6-v2": (),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": (),
    "sentence-transformers/LaBSE": (),
    "intfloat/multilingual-e5-small": _FEVER_HOTPOT,
    "intfloat/multilingual-e5-base": _FEVER_HOTPOT,
    "intfloat/multilingual-e5-large-instruct": _FEVER_HOTPOT,
    "ibm-granite/granite-embedding-30m-english": _GRANITE_TASKS,
    "ibm-granite/granite-embedding-125m-english": _GRANITE_TASKS,
    "ibm-granite/granite-embedding-107m-multilingual": _GRANITE_TASKS,
    "ibm-granite/granite-embedding-278m-multilingual": _GRANITE_TASKS,
    "Qwen/Qwen3-Embedding-0.6B": _FEVER_HOTPOT,
    "Qwen/Qwen3-Embedding-8B": _FEVER_HOTPOT,
    "intfloat/e5-mistral-7b-instruct": _FEVER_HOTPOT,
    "GritLM/GritLM-7B": _FEVER_HOTPOT,
}
COURSE_MODEL = "ibm-granite/granite-embedding-107m-multilingual"   # the course's own embedding model

# --- the scores --------------------------------------------------------------------------------------------------------


def _data() -> dict:
    """The table, read once: "models" and "tasks" as tuples, and the read-only arrays "scores" (one row per model, one
    column per task: the main score times 100), "types" (the task type of each column) and "parameters"."""
    def make():
        lines = SCORES.read_text(encoding="utf-8").split("\n")
        rows = [line.split("\t") for line in lines[1:] if line]
        models = list(dict.fromkeys(r[0] for r in rows))
        tasks = list(dict.fromkeys(r[3] for r in rows))
        scores = np.zeros((len(models), len(tasks)))
        size, kind = {}, {}
        for model, parameters, type_, task, score in rows:
            scores[models.index(model), tasks.index(task)] = float(score)
            size[model], kind[task] = int(parameters), type_
        out = {"models": tuple(models), "tasks": tuple(tasks), "scores": scores,
               "types": np.array([kind[t] for t in tasks]),
               "parameters": np.array([size[m] for m in models], dtype=np.int64)}
        for k in ("scores", "types", "parameters"):
            out[k].flags.writeable = False
        return out
    return kept(f"{_KEY}:data", make)


def mteb() -> tuple:
    """(scores, models, tasks, types, parameters) as new values: `scores` an array with one row per model and one column
    per task of MTEB(eng, v2), each number the model's main score on the task times 100; `models` the 15 names, as on
    Hugging Face; `tasks` the 41 names; `types` an array with the task type of each column; `parameters` an array with
    the number of parameters of each model."""
    d = _data()
    return (np.array(d["scores"]), list(d["models"]), list(d["tasks"]), np.array(d["types"]),
            np.array(d["parameters"]))


def short(model: str) -> str:
    """A model's name without its maker: the part after the slash."""
    return model.split("/", 1)[1]


def small_rows() -> np.ndarray:
    """The rows of the models up to 300 million parameters."""
    return np.where(_data()["parameters"] <= SMALL)[0]


def type_labels(lang: str) -> list:
    """The names of the seven task types, in the order of KINDS, with a capital, as the explore's sliders show them."""
    return [TYPE_NAMES[lang][k][0].upper() + TYPE_NAMES[lang][k][1:] for k in KINDS]


# --- the course's own versions of the exercises ------------------------------------------------------------------------


def type_means(scores, types, kinds):
    columns = []
    for kind in kinds:
        columns.append(scores[:, types == kind].mean(axis=1))
    return np.array(columns).T


def best_for(means, weights, models):
    total = means @ weights / np.sum(weights)
    best = total.argmax()
    return models[best], total[best]


def course_means(zero_shot: bool = False) -> np.ndarray:
    """The type means of the 15 models, in the order of KINDS, read-only; with `zero_shot`, only over the tasks that
    none of the 15 was trained on."""
    def make():
        d = _data()
        keep = np.array([not any(t in TRAINED_ON[m] for m in d["models"]) for t in d["tasks"]]) if zero_shot else \
            np.ones(len(d["tasks"]), dtype=bool)
        out = type_means(d["scores"][:, keep], d["types"][keep], KINDS)
        out.flags.writeable = False
        return out
    return kept(f"{_KEY}:means:{zero_shot}", make)


def counts() -> np.ndarray:
    """How many tasks each type has, in the order of KINDS."""
    types = _data()["types"]
    return np.array([int(np.sum(types == k)) for k in KINDS])


def mean_task() -> np.ndarray:
    """Mean(Task) of the 15 models: the mean over the 41 tasks."""
    return _data()["scores"].mean(axis=1)


def ranks(values) -> np.ndarray:
    """The place of each value from the highest down, 1 for the highest."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(1, len(order) + 1)
    return out


# --- the look of the pictures -------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.4rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.08rem 0.15rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-ui);font-size:0.8rem;font-variant-numeric:tabular-nums")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_SMALL_TEXT = "font-family:var(--ea-font-ui);font-size:0.74rem;color:var(--ea-ink-3)"


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


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _plain(text) -> str:
    """Typed text for a note: without asterisks and backticks, which the note reads as italics and code."""
    return str(text).replace("*", "").replace("`", "")


def _dec(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _size(n: int, lang: str) -> str:
    """A number of parameters, short: 23M or 7.6B; in Dutch 23 mln or 7,6 mld."""
    if n >= 1_000_000_000:
        return _dec(n / 1e9, lang) + (" mld" if lang == "nl" else "B")
    return f"{n / 1e6:.0f}" + (" mln" if lang == "nl" else "M")


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right. A head may
    hold a line break."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h).replace(chr(10), "<br>")}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _bar(value: float, scale: float = 100) -> str:
    """A gray bar for a score from 0 to 100."""
    width = max(0.0, min(100.0, value)) / 100 * scale
    return f'<span style="{_BAR};width:{width:.0f}px;background:var(--ea-data)"></span>'


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


def _model_cell(model: str, lang: str) -> str:
    """A model's name, with its number of parameters under it, small."""
    return (f'{html.escape(short(model))}<br><span style="{_SMALL_TEXT}">'
            f'{html.escape(_size(int(_data()["parameters"][_data()["models"].index(model)]), lang))}</span>')


# --- the pictures of the idea: the nine small models ------------------------------------------------------------------

_BOARD_WORDS = {
    "en": {"title": "MTEB(eng, v2): the 9 models up to 300 million parameters",
           "head": ["Model", "Mean\n(Task)", "Mean\n(TaskType)"],
           "legend": "Each number is a mean of main scores times 100, from 0 to 100: over all 41 tasks in Mean(Task), the "
                     "mean of the seven type means in Mean(TaskType), and over the tasks of one type in the other "
                     "columns. Under each name, the model's number of parameters. Gray band and bold: the highest of the "
                     "9 in that column. The rows go from the highest Mean(Task) down."},
    "nl": {"title": "MTEB(eng, v2): de 9 modellen tot 300 miljoen parameters",
           "head": ["Model", "Mean\n(Task)", "Mean\n(TaskType)"],
           "legend": "Elk getal is een gemiddelde van hoofdscores maal 100, van 0 tot 100: over alle 41 taken bij "
                     "Mean(Task), het gemiddelde van de zeven taaktypegemiddelden bij Mean(TaskType), en over de taken "
                     "van één type in de andere kolommen. Onder elke naam het aantal parameters van het model. Grijze "
                     "band en vet: de hoogste van de 9 in die kolom. De rijen gaan van de hoogste Mean(Task) omlaag."},
}


def board_picture(lang: str) -> str:
    w = _BOARD_WORDS[lang]
    d = _data()
    rows = small_rows()
    means = course_means()[rows]
    table = np.column_stack([mean_task()[rows], means.mean(axis=1), means])
    order = np.argsort(-table[:, 0], kind="stable")
    best = table.max(axis=0)
    body = []
    for i in order:
        cells = [_model_cell(d["models"][rows[i]], lang)]
        for j, v in enumerate(table[i]):
            text = _dec(v, lang)
            cells.append(f'<span style="background:var(--ea-data-wash);padding:0.05rem 0.25rem;border-radius:3px">'
                         f"<b>{text}</b></span>" if v == best[j] else text)
        body.append(cells)
    head = w["head"] + [HEADS[lang][k] for k in KINDS]
    html_ = _table(body, head, numbers=tuple(range(1, len(head))))
    return yours.figure(w["title"], html_ + _note(w["legend"]), alt=f'{w["title"]}. ' + _bare(w["legend"]))


_RANK_WORDS = {
    "en": {"title": "The places of the 9 models, column by column",
           "first": "Mean(Task)",
           "legend": "Each column ranks the 9 models up to 300 million parameters, from 1 at the top to 9 at the bottom: "
                     "first by Mean(Task), then by the mean of each task type. A line follows one model; the names stand "
                     "at its place by Mean(Task). Three models in dark lines: {a} (solid), {b} (dashed) and the course's "
                     "own Granite, {c} (dotted); the other six in gray."},
    "nl": {"title": "De plaatsen van de 9 modellen, kolom voor kolom",
           "first": "Mean(Task)",
           "legend": "Elke kolom zet de 9 modellen tot 300 miljoen parameters op volgorde, van 1 bovenaan tot 9 onderaan: "
                     "eerst op Mean(Task), dan op het gemiddelde van elk taaktype. Een lijn volgt één model; de namen "
                     "staan bij de plaats van het model op Mean(Task). Drie modellen in donkere lijnen: {a} "
                     "(doorgetrokken), {b} (gestreept) en de eigen Granite van de course, {c} (gestippeld); de andere zes "
                     "in grijs."},
}
_DASH = {"ibm-granite/granite-embedding-125m-english": "none",
         "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "6 4",
         COURSE_MODEL: "1.5 3.5"}


def ranks_picture(lang: str) -> str:
    w = _RANK_WORDS[lang]
    d = _data()
    rows = small_rows()
    means = course_means()[rows]
    columns = [mean_task()[rows]] + [means[:, j] for j in range(len(KINDS))]
    heads = [w["first"]] + [HEADS[lang][k] for k in KINDS]
    places = np.array([ranks(c) for c in columns])   # one row per column, one column per model
    width, height = 720, 312
    left, right, top, step = 300, 686, 46, 30   # room on the left for the longest name, on the right for a head
    xs = [left + k * (right - left) / (len(columns) - 1) for k in range(len(columns))]
    def Y(place):
        return top + (place - 1) * step
    out = []
    for k, head in enumerate(heads):
        out.append(f'<text x="{xs[k]:.1f}" y="22" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle;'
                   f'font-weight:600">{html.escape(head)}</text>')
        out.append(f'<line x1="{xs[k]:.1f}" y1="{top - 10}" x2="{xs[k]:.1f}" y2="{Y(len(rows)) + 10}" '
                   f'style="stroke:var(--ea-line);stroke-width:0.8"/>')
    marked = [i for i in range(len(rows)) if d["models"][rows[i]] in _DASH]
    for i in [i for i in range(len(rows)) if i not in marked] + marked:
        model = d["models"][rows[i]]
        dash = _DASH.get(model)
        pts = " ".join(f"{xs[k]:.1f},{Y(places[k, i]):.1f}" for k in range(len(columns)))
        if dash is None:
            out.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-data);stroke-opacity:0.55;'
                       f'stroke-width:1.6;stroke-linejoin:round"/>')
        else:
            out.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2.4;'
                       f'stroke-linejoin:round;stroke-linecap:round;stroke-dasharray:{dash}"/>')
        for k in range(len(columns)):
            fill = "var(--ea-ink)" if dash is not None else "var(--ea-data)"
            out.append(f'<circle cx="{xs[k]:.1f}" cy="{Y(places[k, i]):.1f}" r="{4 if dash is not None else 3}" '
                       f'style="fill:{fill};stroke:var(--ea-page);stroke-width:1"/>')
        weight = "600" if dash is not None else "400"
        color = "var(--ea-ink)" if dash is not None else "var(--ea-ink-3)"
        out.append(f'<text x="{left - 12}" y="{Y(places[0, i]) + 4:.1f}" style="font-size:12px;fill:{color};'
                   f'text-anchor:end;font-weight:{weight}">{places[0, i]}. {html.escape(short(model))}</text>')
    svg = (f'<svg viewBox="0 0 {width} {height}" aria-hidden="true" style="width:100%;max-width:{width}px;height:auto;'
           f'font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">{"".join(out)}</svg>')
    names = [short(m) for m in _DASH]
    legend = w["legend"].format(a=names[0], b=names[1], c=names[2])
    alt = "; ".join(f"{short(d['models'][rows[i]])}: " + ", ".join(f"{heads[k]} {places[k, i]}" for k in range(len(heads)))
                    for i in range(len(rows)))
    return yours.figure(w["title"], svg + _note(legend), alt=f'{w["title"]}. {alt}. {_bare(legend)}')


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
    """The learner's function within a time limit; arrays and lists are passed as new copies, as `C.mteb` gives them.
    _STOPPED when it stops with an error, and the Timeout passes on."""
    def copy(a):
        if isinstance(a, np.ndarray):
            return np.array(a)
        return list(a) if isinstance(a, list) else a
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(lambda: fn(*(copy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def as_table(v):
    """The learner's table as a 2-D array of floats, or None when it is not one (a dict, a text, ragged rows, None)."""
    if v is None or isinstance(v, (str, bytes, dict)):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.array(v, dtype=float)
    except (ValueError, TypeError, OverflowError):
        return None
    return a if a.ndim == 2 else None


def same_table(got, want) -> bool:
    """The comparison of the check of type_means: an array, or a list of rows, of the same shape as want's, each number
    within 1e-6 of want's."""
    a, b = as_table(got), as_table(want)
    if a is None or b is None:
        if a is None and b is None and got is not None and want is not None:   # two 1-D results, for a wrong version
            try:
                x, y = np.array(got, dtype=float), np.array(want, dtype=float)
                return x.shape == y.shape and bool(np.allclose(x, y, atol=1e-6, rtol=0, equal_nan=True))
            except (ValueError, TypeError, OverflowError):
                return False
        return False
    return a.shape == b.shape and bool(np.allclose(a, b, atol=1e-6, rtol=0, equal_nan=True))


def _number(x) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4g}"


def _table_shown(v):
    """What type_means gave, as a message shows it: a small table in full, a big one by its first row and its shape."""
    a = as_table(v)
    if a is None:
        try:
            x = np.array(v, dtype=float)
            if x.ndim == 1 and len(x) <= 8:
                return "[" + ", ".join(_number(float(e)) for e in x) + "]"
        except (ValueError, TypeError, OverflowError):
            pass
        return _short(v)
    if a.shape[0] <= 3 and a.shape[1] <= 8:
        return "[" + ", ".join("[" + ", ".join(_number(float(e)) for e in row) + "]" for row in a) + "]"
    first = ", ".join(_number(float(e)) for e in a[0][:3])
    return f"[[{first}, …], …] ({a.shape[0]} × {a.shape[1]})"


# --- the picture under exercise 2: your type means for the 15 models ---------------------------------------------------

_MEANS_WORDS = {
    "en": {"title": "Your type_means on the 15 models of MTEB(eng, v2)",
           "head": "Model",
           "legend": "Blue: what your function gives for `type_means(scores, types, C.KINDS)`, one row per model and one "
                     "column per task type; purple where a number differs from the mean of the model's scores on the "
                     "tasks of that type. Bold: the highest of your numbers in each column. Under each name, the model's "
                     "number of parameters.",
           "differs": "For {model} and {kind}, your function gives {got}; the mean of its {n} scores on {kind} tasks is {want}.",
           "shape": "Your function gives back a table of {r} rows and {c} columns; it should have 15 rows, one per model, "
                    "and 7 columns, one per type of `C.KINDS`.",
           "turned": "Your function gives back a table of 7 rows and 15 columns: one row per type and one column per "
                     "model. The rows should be the models.",
           "odd": "With `type_means(scores, types, C.KINDS)`, your function gives back {got}, and not a table of numbers.",
           "error": "Your function stops with an error on `type_means(scores, types, C.KINDS)`.",
           "slow": "Your function is still running after {s} second with `type_means(scores, types, C.KINDS)`."},
    "nl": {"title": "Jouw type_means op de 15 modellen van MTEB(eng, v2)",
           "head": "Model",
           "legend": "Blauw: wat je functie geeft voor `type_means(scores, types, C.KINDS)`, één rij per model en één "
                     "kolom per taaktype; paars waar een getal verschilt van het gemiddelde van de scores van het model "
                     "op de taken van dat type. Vet: het hoogste van je getallen in elke kolom. Onder elke naam het "
                     "aantal parameters van het model.",
           "differs": "Bij {model} en {kind} geeft je functie {got}; het gemiddelde van de {n} scores van het model op "
                      "de taken van het type {kind} is {want}.",
           "shape": "Je functie geeft een tabel van {r} rijen en {c} kolommen terug; de tabel hoort 15 rijen te hebben, "
                    "één per model, en 7 kolommen, één per type van `C.KINDS`.",
           "turned": "Je functie geeft een tabel van 7 rijen en 15 kolommen terug: één rij per type en één kolom per "
                     "model. De rijen horen de modellen te zijn.",
           "odd": "Met `type_means(scores, types, C.KINDS)` geeft je functie {got} terug, en geen tabel met getallen.",
           "error": "Je functie stopt met een fout op `type_means(scores, types, C.KINDS)`.",
           "slow": "Je functie is na {s} seconde nog bezig met `type_means(scores, types, C.KINDS)`."},
}


def means_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _MEANS_WORDS[lang]
    d = _data()
    want = course_means()
    try:
        out = _call(fn, d["scores"], d["types"], list(KINDS))
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"].format(s=PICTURE_SECONDS)), alt=w["title"])
    if out is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=w["title"])
    got = as_table(out)
    if got is None:
        text = w["odd"].format(got=_plain(_short(out, 50)))
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {_bare(text)}')
    if got.shape != want.shape:
        text = w["turned"] if got.shape == want.T.shape else w["shape"].format(r=got.shape[0], c=got.shape[1])
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {_bare(text)}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmax(np.where(np.isfinite(got), got, np.nan), axis=0)
    notes, body, alts = [], [], []
    for i, model in enumerate(d["models"]):
        cells = [_model_cell(model, lang)]
        for j, kind in enumerate(KINDS):
            v = float(got[i, j])
            good = abs(v - want[i, j]) <= 1e-6 if math.isfinite(v) else False
            text = _dec(v, lang) if math.isfinite(v) else ("nan" if math.isnan(v) else "inf")
            chip = _chip(text, _CHIP_PRED if good else _CHIP_PURPLE)
            cells.append(f"<b>{chip}</b>" if math.isfinite(v) and v == top[j] else chip)
            if not good and not notes:
                notes.append(w["differs"].format(model=short(model), kind=TYPE_NAMES[lang][kind],
                                                 got=text if not math.isfinite(v) else _dec(v, lang, 2),
                                                 n=int(np.sum(d["types"] == kind)), want=_dec(want[i, j], lang, 2)))
        body.append(cells)
        alts.append(f"{short(model)}: " + ", ".join(_dec(float(x), lang) for x in got[i]))
    head = [w["head"]] + [HEADS[lang][k] for k in KINDS]
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], _table(body, head, numbers=tuple(range(1, len(head)))) + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the picture under exercise 3: your choice for weightings of your own ----------------------------------------------

PRESETS = [   # a name per language and a weight per type, in the order of KINDS
    ({"en": "The same for every type: Mean(TaskType)", "nl": "Voor elk type hetzelfde: Mean(TaskType)"},
     [1, 1, 1, 1, 1, 1, 1]),
    ({"en": "The number of tasks: Mean(Task)", "nl": "Het aantal taken: Mean(Task)"}, [10, 2, 8, 8, 3, 9, 1]),
    ({"en": "A search box: retrieval 3, reranking 1", "nl": "Een zoekvak: retrieval 3, reranking 1"},
     [3, 1, 0, 0, 0, 0, 0]),
    ({"en": "Sorting mail: classification", "nl": "Post sorteren: classificatie"}, [0, 0, 1, 0, 0, 0, 0]),
    ({"en": "Grouping by topic: clustering", "nl": "Groeperen op onderwerp: clustering"}, [0, 0, 0, 1, 0, 0, 0]),
    ({"en": "Finding duplicates: pair classification", "nl": "Dubbele teksten vinden: paarclassificatie"},
     [0, 0, 0, 0, 1, 0, 0]),
    ({"en": "Scoring similarity: STS", "nl": "Gelijkenis scoren: STS"}, [0, 0, 0, 0, 0, 1, 0]),
]

_BEST_WORDS = {
    "en": {"title": "Your best_for for seven weightings",
           "head": ["Weighting", "Up to 300 million parameters", "All 15 models"],
           "legend": "Blue: the model your function chooses and its weighted mean, among the 9 models up to 300 million "
                     "parameters and among all 15, with the type means of `type_means` and the weights of each row; "
                     "purple where your choice or your number differs.",
           "differs": "With the weights of *{preset}*{only}, your function gives {got}; the highest weighted mean is "
                      "{want}, of {model}.",
           "only": " and only the models up to 300 million parameters",
           "odd": "With the weights of *{preset}*, your function gives back {got}, and not two values.",
           "error": "Your function stops with an error with the weights of *{preset}*.",
           "slow": "Your function is still running after {s} second with the weights of *{preset}*."},
    "nl": {"title": "Jouw best_for voor zeven wegingen",
           "head": ["Weging", "Tot 300 miljoen parameters", "Alle 15 modellen"],
           "legend": "Blauw: het model dat je functie kiest en het gewogen gemiddelde ervan, onder de 9 modellen tot 300 "
                     "miljoen parameters en onder alle 15, met de taaktypegemiddelden van `type_means` en de weging van "
                     "elke rij; paars waar je keuze of je getal verschilt.",
           "differs": "Met de weging van *{preset}*{only} geeft je functie {got}; het hoogste gewogen gemiddelde is "
                      "{want}, van {model}.",
           "only": " en alleen de modellen tot 300 miljoen parameters",
           "odd": "Met de weging van *{preset}* geeft je functie {got} terug, en geen twee waarden.",
           "error": "Je functie stopt met een fout met de weging van *{preset}*.",
           "slow": "Je functie is na {s} seconde nog bezig met de weging van *{preset}*."},
}


def _as_pair(out):
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    return None


def _same_value(a, b) -> bool:
    """Two texts that are equal, or two numbers within 1e-6."""
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= 1e-6


def same_best(got, want) -> bool:
    """The comparison of the check of best_for: two values, the same name and a weighted mean within 1e-6; a wrong
    version that gives a position or swaps the two is compared the same way, value by value."""
    g, wp = _as_pair(got), _as_pair(want)
    return g is not None and wp is not None and _same_value(g[0], wp[0]) and _same_value(g[1], wp[1])


def _best_shown(v):
    pair = _as_pair(v)
    if pair is None:
        return _short(v)

    def one(x):
        if isinstance(x, str):
            return '"' + x + '"'
        if _finite(x):
            return f"{float(x):.6g}"
        return _short(x, 30)
    return "(" + ", ".join(one(x) for x in pair) + ")"


def _choice_cell(got, want, lang: str) -> str:
    good = same_best(got, want)
    name, value = got
    text = short(name) if isinstance(name, str) and "/" in name else _plain(_short(name, 40))
    number = _dec(float(value), lang, 2) if _finite(value) else _plain(_short(value, 20))
    return _chip(f"{text}: {number}", _CHIP_PRED if good else _CHIP_PURPLE)


def best_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _BEST_WORDS[lang]
    d = _data()
    means = course_means()
    rows = small_rows()
    sets = [(np.array(means[rows]), [d["models"][i] for i in rows], True), (np.array(means), list(d["models"]), False)]
    notes, body, alts = [], [], []
    for names, weights in PRESETS:
        preset = names[lang]
        cells = [html.escape(preset) + f'<br><span style="{_SMALL_TEXT}">'
                 + html.escape(" · ".join(f"{HEADS[lang][k]} {x}" for k, x in zip(KINDS, weights) if x)) + "</span>"]
        for m, models, only in sets:
            weights_ = np.array(weights, dtype=float)
            try:
                out = _call(fn, m, weights_, models)
            except Timeout:
                notes.append(w["slow"].format(preset=preset, s=PICTURE_SECONDS))
                return yours.figure(w["title"], (_table(body, w["head"]) if body else "") + _note(" ".join(notes + [w["legend"]])),
                                    alt=w["title"])
            if out is _STOPPED:
                notes.append(w["error"].format(preset=preset))
                return yours.figure(w["title"], (_table(body, w["head"]) if body else "") + _note(" ".join(notes + [w["legend"]])),
                                    alt=w["title"])
            got = _as_pair(out)
            if got is None:
                notes.append(w["odd"].format(preset=preset, got=_plain(_short(out, 40))))
                return yours.figure(w["title"], (_table(body, w["head"]) if body else "") + _note(" ".join(notes + [w["legend"]])),
                                    alt=w["title"])
            want = best_for(m, weights_, models)
            cells.append(_choice_cell(got, want, lang))
            alts.append(f"{preset}: {_plain(_best_shown(got))}")
            if not same_best(got, want) and not notes:
                notes.append(w["differs"].format(preset=preset, only=w["only"] if only else "",
                                                 got=_plain(_best_shown(got)), want=_dec(float(want[1]), lang, 2),
                                                 model=short(want[0])))
        body.append(cells)
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], _table(body, w["head"]) + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the explore: weights of your own ----------------------------------------------------------------------------------

LIMITS = {   # the choices of the explore's size: the most parameters a model may have
    "en": {"All 15 models": 10**12, "Up to 1 billion parameters": 10**9, "Up to 300 million parameters": SMALL,
           "Up to 50 million parameters": 50_000_000},
    "nl": {"Alle 15 modellen": 10**12, "Tot 1 miljard parameters": 10**9, "Tot 300 miljoen parameters": SMALL,
           "Tot 50 miljoen parameters": 50_000_000},
}

_EXPLORE_WORDS = {
    "en": {"title": "Your weighting of MTEB(eng, v2)",
           "head": ["#", "Model", "Your weighted mean", "Place by Mean(TaskType)", "Trained on"],
           "trained": "{n} of 41",
           "none": "Give at least one task type a weight above 0.",
           "shares": "Your weights as shares: {shares}.",
           "zero": "Only the {n} tasks that none of the 15 models was trained on.",
           "legend": "Gray bar: the weighted mean of the model's type means with your weights, on a scale from 0 to 100. "
                     "Place by Mean(TaskType): the model's place among the same models when every type counts the same, "
                     "with ▲ for a model your weighting moves up and ▼ for one it moves down. Trained on: how many of the "
                     "41 tasks the model was trained on, by the marks of the leaderboard."},
    "nl": {"title": "Jouw weging van MTEB(eng, v2)",
           "head": ["#", "Model", "Jouw gewogen gemiddelde", "Plaats op Mean(TaskType)", "Getraind op"],
           "trained": "{n} van 41",
           "none": "Geef minstens één taaktype een weging boven 0.",
           "shares": "Je weging als delen: {shares}.",
           "zero": "Alleen de {n} taken waarop geen van de 15 modellen getraind is.",
           "legend": "Grijze balk: het gewogen gemiddelde van de taaktypegemiddelden van het model met jouw weging, op "
                     "een schaal van 0 tot 100. Plaats op Mean(TaskType): de plaats van het model tussen dezelfde "
                     "modellen als elk type even zwaar telt, met ▲ voor een model dat je weging omhoog brengt en ▼ voor "
                     "een model dat je weging omlaag brengt. Getraind op: op hoeveel van de 41 taken het model getraind "
                     "is, volgens de markeringen van het leaderboard."},
}


def zero_shot_tasks() -> int:
    d = _data()
    return sum(1 for t in d["tasks"] if not any(t in TRAINED_ON[m] for m in d["models"]))


def explore_picture(weights, limit: str, zero_shot: bool, lang: str) -> str:
    w = _EXPLORE_WORDS[lang]
    d = _data()
    weights_ = np.array([float(x) for x in weights])
    if not np.sum(weights_) > 0:
        return yours.figure(w["title"], _note(w["none"]), alt=f'{w["title"]}. {w["none"]}')
    most = LIMITS[lang].get(limit, 10**12)
    rows = np.where(d["parameters"] <= most)[0]
    means = course_means(bool(zero_shot))[rows]
    total = means @ weights_ / np.sum(weights_)
    plain = ranks(means.mean(axis=1))
    order = np.argsort(-total, kind="stable")
    body = []
    for place, i in enumerate(order, start=1):
        model = d["models"][rows[i]]
        before = int(plain[i])
        move = "" if before == place else (f" ▲{before - place}" if before > place else f" ▼{place - before}")
        body.append([str(place), _model_cell(model, lang),
                     f'<span style="white-space:nowrap">{_bar(float(total[i]))} {_dec(float(total[i]), lang, 2)}</span>',
                     f"{before}{move}", html.escape(w["trained"].format(n=len(TRAINED_ON[model])))])
    shares = ", ".join(f"{TYPE_NAMES[lang][k]} {round(100 * x / np.sum(weights_))}%" for k, x in zip(KINDS, weights_) if x)
    notes = [w["shares"].format(shares=shares)]
    if zero_shot:
        notes.append(w["zero"].format(n=zero_shot_tasks()))
    text = " ".join(notes + [w["legend"]])
    alt = "; ".join(f"{p}. {short(d['models'][rows[i]])} {_dec(float(total[i]), lang, 2)}" for p, i in enumerate(order, 1))
    return yours.figure(w["title"], _table(body, w["head"], numbers=(0, 2, 3, 4)) + _note(text),
                        alt=f'{w["title"]}. {alt}. {_bare(text)}')


# --- the checks ---------------------------------------------------------------------------------------------------------

REQUIRED = ["use-cases", "type_means", "best_for"]

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
    """A value the check gives the learner's function, shown in a message by its name. An array or a list goes as a new
    copy, as `C.mteb` gives one."""

    def __init__(self, name: str, value):
        self.name, self._value = name, value

    @property
    def value(self):
        return self._value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        v = self.value
        if isinstance(v, np.ndarray):
            return np.array(v)
        return list(v) if isinstance(v, list) else v


class _Lazy(Named):
    """A Named value that is read when the check first needs it, such as the scores of MTEB."""

    def __init__(self, name: str, make):
        self.name, self._make = name, make

    @property
    def value(self):
        return self._make()


def _ro(a) -> Named:
    """A small read-only array of numbers, shown in a message as the code that makes it."""
    v = np.array(a, dtype=float)
    v.flags.writeable = False
    return Named("np.array(" + repr(a) + ")", v)


def _ro_text(a) -> Named:
    """A small read-only array of texts, shown in a message as the code that makes it, with double quotes."""
    v = np.array(a)
    v.flags.writeable = False
    return Named("np.array([" + ", ".join(f'"{x}"' for x in a) + "])", v)


def _bool_index(e) -> bool:
    """A mask of `types` used on the rows of `scores`: it has one value per column, so it does not fit the rows."""
    return isinstance(e, IndexError) and "boolean index did not match" in str(e)


def _ragged(e) -> bool:
    """Parts of different lengths that NumPy cannot put in one array."""
    return isinstance(e, ValueError) and ("inhomogeneous" in str(e) or "setting an array element with a sequence" in str(e))


def _not_table(got, text) -> bool:
    return got is not None and as_table(got) is None


def _float_index(e) -> bool:
    """A list given a number with decimals as its position, such as the highest weighted mean itself."""
    return isinstance(e, TypeError) and "indices must be integers" in str(e)


def _set_of_types(got, text) -> bool:
    """A loop through a set of the types: its order depends on Python's hashes, so it changes from run to run."""
    return re.search(r"for\s+\w+\s+in\s+(list\(\s*)?set\(\s*types\b", text) is not None


_ORDER = {"en": "`{call}` gives {got}, and it should give {want}: your columns follow the types that `types` holds, in an order of their own, and not the types of `kinds` in the order of `kinds`. Which list should your loop go through?",
          "nl": "`{call}` geeft {got}, en dat moet {want} zijn: je kolommen volgen de types uit `types`, in een eigen volgorde, en niet de types van `kinds` in de volgorde van `kinds`. Langs welke lijst moet je loop gaan?"}


def _not_pair(got, text) -> bool:
    return got is not None and _as_pair(got) is None


# exercise 2: type_means. The visible cases: the example and the same with kinds the other way round; then a part of
# the types in an order of its own, and the 15 models of MTEB with C.KINDS and with two types.
_SCORES = _Lazy("scores", lambda: _data()["scores"])
_TYPES = _Lazy("types", lambda: _data()["types"])
_EX_SCORES = _ro([[60.0, 70.0, 40.0], [80.0, 60.0, 30.0]])
_EX_TYPES = _ro_text(["STS", "STS", "Retrieval"])
TM_CASES = [
    (_EX_SCORES, _EX_TYPES, ["Retrieval", "STS"]),
    (_EX_SCORES, _EX_TYPES, ["STS", "Retrieval"]),
    (_ro([[50.0, 70.0, 90.0, 20.0, 40.0], [10.0, 30.0, 60.0, 80.0, 70.0], [55.0, 65.0, 75.0, 85.0, 95.0]]),
     _ro_text(["Clustering", "STS", "Clustering", "Retrieval", "STS"]), ["STS", "Clustering"]),
    (_SCORES, _TYPES, Named("C.KINDS", list(KINDS))),
    (_SCORES, _TYPES, ["Summarization", "Retrieval"]),
]

_TM_HEAD = "import numpy as np\n\n\ndef type_means(scores, types, kinds):\n"
_TM_START = _TM_HEAD + "    ...\n"
_TM_LOOP = "    columns = []\n    for kind in kinds:\n"
_TM = _TM_HEAD + _TM_LOOP + "        columns.append(scores[:, types == kind].mean(axis=1))\n    return np.array(columns).T\n"
_TM_TURNED = _TM.replace("np.array(columns).T", "np.array(columns)")
_TM_ONE = _TM.replace(".mean(axis=1)", ".mean()")
_TM_SUM = _TM.replace(".mean(axis=1)", ".sum(axis=1)")
_TM_OTHER = _TM.replace("types == kind", "types != kind")
_TM_EVERY = _TM.replace("scores[:, types == kind]", "scores")
_TM_UNIQUE = _TM.replace("for kind in kinds:", "for kind in np.unique(types):")
_TM_ROWS = _TM.replace("scores[:, types == kind]", "scores[types == kind]")
_TM_AXIS0 = _TM.replace(".mean(axis=1)", ".mean(axis=0)")

# exercise 3: best_for. The visible cases: the example with two weightings; then a weight of 0, and the type means of
# the 15 models with weights 1 and with the number of tasks, and those of the 9 small models for search and for mail.
_MEANS = _Lazy("means", lambda: course_means())
_MODELS = _Lazy("models", lambda: list(_data()["models"]))
_SMALL_MEANS = _Lazy("means_small", lambda: course_means()[small_rows()])
_SMALL_MODELS = _Lazy("models_small", lambda: [_data()["models"][i] for i in small_rows()])
_EX_MEANS = _ro([[50.0, 80.0], [60.0, 60.0]])
BF_CASES = [
    (_EX_MEANS, _ro([3.0, 1.0]), ["a", "b"]),
    (_EX_MEANS, _ro([1.0, 1.0]), ["a", "b"]),
    (_ro([[70.0, 40.0, 90.0], [80.0, 60.0, 20.0], [60.0, 50.0, 85.0]]), _ro([0.0, 3.0, 1.0]), ["x", "y", "z"]),
    (_MEANS, _ro([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), _MODELS),
    (_MEANS, _ro([10.0, 2.0, 8.0, 8.0, 3.0, 9.0, 1.0]), _MODELS),
    (_SMALL_MEANS, _ro([3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), _SMALL_MODELS),
    (_SMALL_MEANS, _ro([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), _SMALL_MODELS),
]

_BF_HEAD = "import numpy as np\n\n\ndef best_for(means, weights, models):\n"
_BF_START = _BF_HEAD + "    ...\n"
_BF_TOTAL = "    total = means @ weights / np.sum(weights)\n"
_BF_PICK = "    best = total.argmax()\n"
_BF = _BF_HEAD + _BF_TOTAL + _BF_PICK + "    return models[best], total[best]\n"
_BF_SUM = _BF.replace(" / np.sum(weights)", "")
_BF_LEN = _BF.replace("np.sum(weights)", "len(weights)")
_BF_MIN = _BF.replace("total.argmax()", "total.argmin()")
_BF_INDEX = _BF.replace("return models[best], total[best]", "return best, total[best]")
_BF_SWAP = _BF.replace("return models[best], total[best]", "return total[best], models[best]")
_BF_PLAIN = _BF.replace("means @ weights / np.sum(weights)", "means.mean(axis=1)")
_BF_TURNED = _BF.replace("means @ weights", "weights @ means")

EXERCISES = {
    "use-cases": Sort(
        name="use-cases",
        groups={
            "retrieval": {"en": "Retrieval: find, for a query, the texts that answer it, among many",
                          "nl": "Retrieval: bij een zoekvraag de teksten vinden die er antwoord op geven, uit vele"},
            "classification": {"en": "Classification: give each text one label out of a fixed list",
                               "nl": "Classificatie: elke tekst één label geven uit een vaste lijst"},
            "clustering": {"en": "Clustering: form groups of texts that are alike, without labels",
                           "nl": "Clustering: groepen maken van teksten die op elkaar lijken, zonder labels"},
            "pairs": {"en": "Pair classification: decide whether two texts are duplicates or paraphrases, yes or no",
                      "nl": "Paarclassificatie: beslissen of twee teksten dubbel zijn of parafrases van elkaar, ja of nee"},
            "sts": {"en": "STS: give two texts a score for how alike they are in meaning",
                    "nl": "STS: twee teksten een score geven voor de gelijkenis in betekenis"},
        },
        items={
            "bugs": {"en": "A tool reads a new bug report and one earlier report, and says whether the two describe the same problem.",
                     "nl": "Een tool leest een nieuwe bugmelding en één eerdere melding, en zegt of de twee hetzelfde probleem beschrijven."},
            "helpdesk": {"en": "Someone types *my bike light keeps flickering* on the help pages of a shop, and gets the three best pages out of 2,000.",
                         "nl": "Iemand typt *mijn fietslamp blijft knipperen* op de hulppagina's van een winkel, en krijgt de drie beste pagina's uit 2.000."},
            "survey": {"en": "3,000 answers to an open question of a survey are put into themes, without a list of themes beforehand.",
                       "nl": "3.000 antwoorden op een open vraag van een enquête worden in thema's verdeeld, zonder vooraf een lijst met thema's."},
            "reviews": {"en": "Each new review of a hotel is marked positive, neutral or negative.",
                        "nl": "Elke nieuwe recensie van een hotel krijgt het label positief, neutraal of negatief."},
            "exam": {"en": "Each answer of a student gets a number from 0 to 5 for how close its meaning is to the teacher's model answer.",
                     "nl": "Elk antwoord van een student krijgt een getal van 0 tot 5 voor hoe dicht de betekenis bij het modelantwoord van de docent ligt."},
            "rulings": {"en": "A lawyer looks for the court rulings about a tenant who keeps a dog, among 40,000 rulings.",
                        "nl": "Een advocaat zoekt de uitspraken over een huurder die een hond houdt, tussen 40.000 uitspraken."},
            "listing": {"en": "A web shop puts a new product listing next to one listing it already has, and checks whether the two are copies.",
                        "nl": "Een webwinkel legt een nieuwe productpagina naast één pagina die er al staat, en controleert of de twee kopieën van elkaar zijn."},
            "tickets": {"en": "Each support message goes to billing, repairs or accounts.",
                        "nl": "Elk bericht aan de klantenservice gaat naar facturen, reparaties of accounts."},
            "news": {"en": "A week of news headlines is grouped by story, with no list of stories.",
                     "nl": "Een week aan nieuwskoppen wordt gegroepeerd per verhaal, zonder lijst met verhalen."},
            "bag": {"en": "The description of a lost bag by its owner and the description by its finder get a number for how much they say the same.",
                    "nl": "De beschrijving van een verloren tas door de eigenaar en de beschrijving door de vinder krijgen een getal voor hoeveel de twee hetzelfde zeggen."},
        },
        solution={"helpdesk": "retrieval", "rulings": "retrieval", "reviews": "classification",
                  "tickets": "classification", "survey": "clustering", "news": "clustering", "bugs": "pairs",
                  "listing": "pairs", "exam": "sts", "bag": "sts"},
        nudge={
            "helpdesk": {"en": "Does the system decide something about one text, or look for the best texts among many for what someone typed?",
                         "nl": "Beslist het systeem iets over één tekst, of zoekt het de beste teksten uit vele bij wat iemand typte?"},
            "rulings": {"en": "Is there a fixed list of labels here, or a query and a large pile of texts to search?",
                        "nl": "Is hier een vaste lijst met labels, of een zoekvraag en een grote stapel teksten om te doorzoeken?"},
            "reviews": {"en": "Is the list of possible answers known beforehand?",
                        "nl": "Ligt de lijst met mogelijke antwoorden vooraf vast?"},
            "tickets": {"en": "Does each message get one label out of a list, or a place in a group nobody named?",
                        "nl": "Krijgt elk bericht één label uit een lijst, of een plek in een groep zonder naam?"},
            "survey": {"en": "Is there a list of themes to choose from?",
                       "nl": "Is er een lijst met thema's om uit te kiezen?"},
            "news": {"en": "Are the stories named beforehand, or do they come out of the headlines themselves?",
                     "nl": "Hebben de verhalen vooraf een naam, of komen de verhalen uit de koppen zelf?"},
            "bugs": {"en": "How many texts does the tool look at, at once, and is its answer a number or a yes or a no?",
                     "nl": "Naar hoeveel teksten tegelijk kijkt de tool, en is het antwoord een getal of een ja of nee?"},
            "listing": {"en": "Is the answer a score, or a yes or a no about two texts?",
                        "nl": "Is het antwoord een score, of een ja of nee over twee teksten?"},
            "exam": {"en": "Is the answer a yes or a no, or a number on a scale?",
                     "nl": "Is het antwoord een ja of nee, of een getal op een schaal?"},
            "bag": {"en": "Is the answer a yes or a no, or a number for how much?",
                    "nl": "Is het antwoord een ja of nee, of een getal voor hoeveel?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each use under the task type of MTEB that looks most like it.",
                      "nl": "Zet elk gebruik bij het taaktype van MTEB dat er het meest op lijkt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The use {item} belongs in another group. What does the system get, and what does it give back?",
                      "nl": "Het gebruik {item} hoort in een andere groep. Wat krijgt het systeem, en wat geeft het terug?"},
            "correct": {"en": "All {n} are right. What the system gets and gives back decides the type: a query and many texts is retrieval, a fixed list of labels classification, groups without a list clustering, a yes or a no about two texts pair classification, and a number for how alike STS. A system often does more than one: give each type a weight.",
                        "nl": "Alle {n} kloppen. Wat het systeem krijgt en teruggeeft, bepaalt het type: een zoekvraag en veel teksten is retrieval, een vaste lijst met labels classificatie, groepen zonder lijst clustering, een ja of nee over twee teksten paarclassificatie, en een getal voor hoe gelijk STS. Een systeem doet vaak meer dan één ding: geef elk type een weging."},
        },
    ),
    "type_means": FunctionBox(
        name="type_means",
        picture=means_picture,
        function="type_means",
        start=_TM_START,
        solution=_TM,
        visible=TM_CASES[:2],
        hidden=TM_CASES[2:],
        compare=same_table,
        show=_table_shown,
        first=[_DOTS],
        time_limit=2.0,
        mistakes=[
            Mistake("rows", error=_bool_index,
                    en="With `{call}`, your function stops with an IndexError: a mask of `types` has one value per column of `scores`, one per task, and your function puts it where the rows go. Where in the square brackets does a mask for the columns stand?",
                    nl="Met `{call}` stopt je functie met een IndexError: een mask van `types` heeft één waarde per kolom van `scores`, één per taak, en je functie zet de mask op de plek van de rijen. Waar tussen de vierkante haken staat een mask voor de kolommen?"),
            Mistake("ragged", error=_ragged,
                    en="With `{call}`, your function stops with a ValueError: the parts it puts in one array have different lengths, so NumPy cannot make a table of them. How many numbers should each type give: one per model, or one per task?",
                    nl="Met `{call}` stopt je functie met een ValueError: de delen die de functie in één array zet, hebben verschillende lengtes, dus NumPy kan er geen tabel van maken. Hoeveel getallen moet elk type geven: één per model, of één per taak?"),
            Mistake("turned", wrong=_TM_TURNED,
                    en="`{call}` gives {got}, and it should give {want}: your numbers are right, but your table has one row per type and one column per model. Which way should the rows run?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je getallen kloppen, maar je tabel heeft één rij per type en één kolom per model. Welke kant op horen de rijen te lopen?"),
            Mistake("one-number", wrong=_TM_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives one number per type, the mean over all models together. How many numbers should each type give?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft één getal per type, het gemiddelde over alle modellen samen. Hoeveel getallen moet elk type geven?"),
            Mistake("sum", wrong=_TM_SUM,
                    en="`{call}` gives {got}, and it should give {want}: your numbers add the scores of a type up, so a type with more tasks gets bigger numbers. What does the exercise ask for, a sum or a mean?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je getallen tellen de scores van een type op, dus een type met meer taken krijgt grotere getallen. Wat vraagt de oefening, een som of een gemiddelde?"),
            Mistake("other-tasks", wrong=_TM_OTHER,
                    en="`{call}` gives {got}, and it should give {want}: the column of a type holds the mean of the other tasks, those of the other types. Which tasks does your mask keep?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kolom van een type bevat het gemiddelde van de andere taken, die van de andere types. Welke taken houdt je mask over?"),
            Mistake("every-task", wrong=_TM_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: every column of your table holds the same numbers, the mean of all tasks of each model. Where does your function pick the tasks of one type?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke kolom van je tabel bevat dezelfde getallen, het gemiddelde van alle taken van elk model. Waar kiest je functie de taken van één type?"),
            Mistake("order-of-a-set", matches=_set_of_types, en=_ORDER["en"], nl=_ORDER["nl"]),
            Mistake("order", wrong=_TM_UNIQUE, en=_ORDER["en"], nl=_ORDER["nl"]),
            Mistake("not-a-table", matches=_not_table,
                    en="`{call}` gives {got}, and it should give {want}: a table of numbers, one row per model and one column per type. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een tabel met getallen, één rij per model en één kolom per type. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each type of `kinds`, take the mean of every model's scores on the tasks of that type, and give back those means as an array with one row per model and one column per type.",
                      "nl": "Vervang de `...`: neem voor elk type van `kinds` het gemiddelde van de scores van elk model op de taken van dat type, en geef die gemiddelden terug als een array met één rij per model en één kolom per type."},
            "unreached": {"en": "With `{call}`, your function gives None, and not a table. Does every way through your function end with a `return` of the table?",
                          "nl": "Met `{call}` geeft je functie None, en geen tabel. Eindigt elke weg door je functie met een `return` van de tabel?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back: it has no `return` with a value. What should it give back at the end?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat geen `return` met een waarde in. Wat moet de functie aan het eind teruggeven?"},
            "slow": {"en": "Your function is still running after 2 seconds on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 2 seconden nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which columns of `scores` belong to a type, and along which axis runs the mean of one model?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke kolommen van `scores` horen bij een type, en langs welke axis loopt het gemiddelde van één model?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function follow `kinds` when it names only some of the types, and does it work on the 15 models of MTEB?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Volgt je functie `kinds` ook als `kinds` maar een deel van de types noemt, en werkt de functie op de 15 modellen van MTEB?"},
            "correct": {"en": "Your `type_means` is right in all {n} cases, also on the 15 models and 41 tasks of MTEB. Look at the picture: which model is best on retrieval, and is it also best on classification?",
                        "nl": "Je `type_means` klopt in alle {n} gevallen, ook op de 15 modellen en 41 taken van MTEB. Kijk naar de afbeelding: welk model is het best op retrieval, en is dat model ook het best op classificatie?"},
        },
    ),
    "best_for": FunctionBox(
        name="best_for",
        picture=best_picture,
        function="best_for",
        start=_BF_START,
        solution=_BF,
        visible=BF_CASES[:2],
        hidden=BF_CASES[2:],
        compare=same_best,
        show=_best_shown,
        first=[_DOTS],
        time_limit=2.0,
        mistakes=[
            Mistake("float-index", error=_float_index,
                    en="With `{call}`, your function stops with a TypeError: a list takes a whole number as a position, and your function gives `models` a number with decimals. Which gives the position of the highest weighted mean, and which the weighted mean itself?",
                    nl="Met `{call}` stopt je functie met een TypeError: een lijst neemt een heel getal als positie, en je functie geeft `models` een getal met decimalen. Wat geeft de positie van het hoogste gewogen gemiddelde, en wat het gewogen gemiddelde zelf?"),
            Mistake("no-division", wrong=_BF_SUM,
                    en="`{call}` gives {got}, and it should give {want}: your number is the weighted sum of the row, not its weighted mean. What should the weighted sum be divided by?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je getal is de gewogen som van de rij, niet het gewogen gemiddelde. Waardoor moet de gewogen som gedeeld worden?"),
            Mistake("length", wrong=_BF_LEN,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of weights, and not by what the weights add up to. With weights of 3 and 1, what share of the mean should each type get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal wegingen, en niet door wat de wegingen samen zijn. Welk deel van het gemiddelde hoort elk type te krijgen bij wegingen van 3 en 1?"),
            Mistake("lowest", wrong=_BF_MIN,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the model with the lowest weighted mean. Which model is best for these weights?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kiest het model met het laagste gewogen gemiddelde. Welk model is het best bij deze weging?"),
            Mistake("position", wrong=_BF_INDEX,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the position of the best model, not its name. Where are the names?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de positie van het beste model terug, niet de naam. Waar staan de namen?"),
            Mistake("swapped", wrong=_BF_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your two values are right, in the other order. Which of the two comes first?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je twee waarden kloppen, in de andere volgorde. Welke van de twee komt eerst?"),
            Mistake("no-weights", wrong=_BF_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the plain mean of each row, so the weights change nothing. Where do the weights come in?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gewone gemiddelde van elke rij, dus de wegingen veranderen niets. Waar komen de wegingen erin?"),
            Mistake("weights-first", wrong=_BF_TURNED,
                    en="`{call}` gives {got}, and it should give {want}: in your product, each weight multiplies a row of `means`, a model, while each weight belongs to a column, a type. What shape do `means` and `weights` have, and which comes first in the product?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je product vermenigvuldigt elke weging een rij van `means`, een model, terwijl elke weging bij een kolom hoort, een type. Welke shape hebben `means` en `weights`, en welke komt eerst in het product?"),
            Mistake("not-a-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give {want}: two values, the name first and the weighted mean second. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: twee waarden, eerst de naam en dan het gewogen gemiddelde. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: compute the weighted mean of each row of `means` with `weights`, and give back the name of the model with the highest one and that weighted mean.",
                      "nl": "Vervang de `...`: bereken het gewogen gemiddelde van elke rij van `means` met `weights`, en geef de naam van het model met het hoogste gewogen gemiddelde terug, en dat gewogen gemiddelde."},
            "unreached": {"en": "With `{call}`, your function gives None, and not two values. Does every way through your function end with a `return` of the name and the weighted mean?",
                          "nl": "Met `{call}` geeft je functie None, en geen twee waarden. Eindigt elke weg door je functie met een `return` van de naam en het gewogen gemiddelde?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back: it has no `return` with a value. What should it give back at the end?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat geen `return` met een waarde in. Wat moet de functie aan het eind teruggeven?"},
            "slow": {"en": "Your function is still running after 2 seconds on a case of the check. Does a loop of yours go on forever?",
                     "nl": "Je functie is na 2 seconden nog bezig met een geval van de check. Gaat een van je loops eindeloos door?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What is the weighted mean of each row, and which row has the highest?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wat is het gewogen gemiddelde van elke rij, en welke rij heeft het hoogste?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work with a weight of 0, and with the 15 models of MTEB?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook met een weging van 0, en met de 15 modellen van MTEB?"},
            "correct": {"en": "Your `best_for` is right in all {n} cases. Look at the picture: which weightings choose another small model than Mean(Task) does?",
                        "nl": "Je `best_for` klopt in alle {n} gevallen. Kijk naar de afbeelding: bij welke wegingen kiest je functie een ander klein model dan Mean(Task)?"},
        },
    ),
}

CASES = [
    ("use-cases", {}, "empty", None),
    ("use-cases", {"helpdesk": "retrieval", "rulings": "retrieval", "reviews": "classification",
                   "tickets": "classification", "survey": "clustering", "news": "clustering", "bugs": "pairs",
                   "listing": "pairs", "exam": "sts", "bag": "sts"}, "correct", None),
    ("use-cases", {"helpdesk": "retrieval", "reviews": "classification"}, "wrong", "unfinished"),
    ("use-cases", {"helpdesk": "classification"}, "wrong", "wrong:helpdesk"),
    ("use-cases", {"rulings": "classification"}, "wrong", "wrong:rulings"),
    ("use-cases", {"reviews": "clustering"}, "wrong", "wrong:reviews"),
    ("use-cases", {"tickets": "clustering"}, "wrong", "wrong:tickets"),
    ("use-cases", {"survey": "classification"}, "wrong", "wrong:survey"),
    ("use-cases", {"news": "classification"}, "wrong", "wrong:news"),
    ("use-cases", {"bugs": "sts"}, "wrong", "wrong:bugs"),
    ("use-cases", {"listing": "sts"}, "wrong", "wrong:listing"),
    ("use-cases", {"exam": "pairs"}, "wrong", "wrong:exam"),
    ("use-cases", {"bag": "pairs"}, "wrong", "wrong:bag"),
    ("type_means", _TM_START, "empty", None),
    ("type_means", _TM, "correct", None),
    ("type_means", _TM_HEAD + "    out = []\n    for kind in kinds:\n        part = scores[:, types == kind]\n"
                   "        out.append(np.sum(part, axis=1) / part.shape[1])\n    return np.array(out).T\n", "correct", None),
    ("type_means", _TM_HEAD + "    rows = []\n    for row in scores:\n        rows.append([np.mean(row[types == kind]) for kind in kinds])\n"
                   "    return np.array(rows)\n", "correct", None),
    ("type_means", _TM_HEAD + "    return [[float(np.mean(row[types == kind])) for kind in kinds] for row in scores]\n",
     "correct", None),
    ("type_means", _TM_ROWS, "error", "rows"),
    ("type_means", _TM_AXIS0, "error", "ragged"),
    ("type_means", _TM_TURNED, "wrong", "turned"),
    ("type_means", _TM_HEAD + _TM_LOOP + "        columns.append(scores[:, types == kind].mean(axis=1))\n    return columns\n",
     "wrong", "turned"),
    ("type_means", _TM_ONE, "wrong", "one-number"),
    ("type_means", _TM_SUM, "wrong", "sum"),
    ("type_means", _TM_OTHER, "wrong", "other-tasks"),
    ("type_means", _TM_EVERY, "wrong", "every-task"),
    ("type_means", _TM_UNIQUE, "wrong", "order"),
    ("type_means", _TM.replace("for kind in kinds:", "for kind in sorted(set(types)):"), "wrong", "order"),
    ("type_means", _TM.replace("for kind in kinds:", "for kind in set(types):"), "wrong", "order-of-a-set"),
    ("type_means", _TM.replace("for kind in kinds:", "for kind in list(set(types)):"), "wrong", "order-of-a-set"),
    ("type_means", _TM_HEAD + "    return {kind: scores[:, types == kind].mean(axis=1) for kind in kinds}\n", "wrong", "not-a-table"),
    ("type_means", _TM_HEAD + _TM_LOOP + "        columns.append(scores[:, types == kind].mean(axis=1))\n", "wrong", "no-return"),
    ("type_means", _TM_HEAD + "    ...\n    return None\n", "wrong", "dots-left"),
    ("best_for", _BF_START, "empty", None),
    ("best_for", _BF, "correct", None),
    ("best_for", _BF_HEAD + "    total = (means * weights).sum(axis=1) / weights.sum()\n    k = np.argmax(total)\n"
                 "    return models[k], total[k]\n", "correct", None),
    ("best_for", _BF_HEAD + "    best = None\n    for i in range(len(models)):\n"
                 "        total = np.sum(means[i] * weights) / np.sum(weights)\n"
                 "        if best is None or total > best[1]:\n            best = (models[i], total)\n    return best\n",
     "correct", None),
    ("best_for", _BF_HEAD + "    total = np.average(means, axis=1, weights=weights)\n    return [models[np.argmax(total)], max(total)]\n",
     "correct", None),
    ("best_for", _BF_HEAD + _BF_TOTAL + "    return models[total.max()], total.max()\n", "error", "float-index"),
    ("best_for", _BF_SUM, "wrong", "no-division"),
    ("best_for", _BF_LEN, "wrong", "length"),
    ("best_for", _BF_MIN, "wrong", "lowest"),
    ("best_for", _BF_INDEX, "wrong", "position"),
    ("best_for", _BF_SWAP, "wrong", "swapped"),
    ("best_for", _BF_PLAIN, "wrong", "no-weights"),
    ("best_for", _BF_TURNED, "wrong", "weights-first"),
    ("best_for", _BF_HEAD + _BF_TOTAL + _BF_PICK + "    return models[best]\n", "wrong", "not-a-pair"),
    ("best_for", _BF_HEAD + _BF_TOTAL + _BF_PICK, "wrong", "no-return"),
    ("best_for", _BF_HEAD + "    ...\n    return None, None\n", "wrong", "dots-left"),
]
