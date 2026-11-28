# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-serving-path: what happens before and after the model when it serves. The course trains the
course's model of the project as the lesson *Project: a demand model, tracked and packed* trains h16-lr0.01, keeps the
mean and the standard deviation of the training hours as config.json does, and serves the model through a path: a
request is prepared as in training, the model makes a number, and a business rule and a threshold turn the number into
a decision. The learner writes the line of `prepare` that standardizes with the saved statistics, and the business rule
of `decide`; a choice of threshold from the costs of the two mistakes, and a sort of steps before and after the model.
The demo shows training-serving skew: a night batch that standardizes tomorrow's 24 hours with their own mean and
standard deviation, as `fit_transform` of a scaler does, makes the model's numbers wrong without a single error.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011, from the Bike Sharing dataset of
Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the requests, the costs and the pictures are the course's own.
Every word a learner's code gives back is shown through html.escape."""
import copy
import html
import importlib.util
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LESSON = "sys-serving-path"
LANGS = ("en", "nl")


def _course_demand():
    """The course's own copy of ops/demand.py from the lesson *Project: a demand model, tracked and packed*."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

FEATURES = list(demand.FEATURES)                                  # the 10 features, in the order of training
FIELDS = ["hr"] + FEATURES                                        # what a request holds: 11 numbers
HOURS_2011 = demand.read_hours()                                  # 8,645 hours of 365 days
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]           # the training days of the project
VAL_HOURS = HOURS_2011[HOURS_2011["day"] % 5 == 4]                # the 73 validation days: the model never saw them
WATCH_DAY = 194                                                   # Thursday 14 July 2011, clear all day
RAIN_DAY = 340                                                    # Wednesday 7 December 2011, light rain at 17:00
THRESHOLD = 400                                                   # the planners' line in decide: a busy hour
BUSY = 400                                                        # an hour with at least this many rentals is busy
CHOSEN = 350                                                      # the threshold Exercise 3 finds, for the demo
COSTS = {"needless": 1, "missed": 3}                              # a trip for nothing, and a busy hour without a crew


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def clock(hr: int) -> str:
    return f"{hr}:00"


# ======== the course's model of the project, and what config.json keeps of training ==================================

def _trained():
    """The course's model of the project, trained once, the way the course's train_version trains h16-lr0.01 in the
    lesson *Project: a demand model, tracked and packed*: seed 0, 16 neurons, Adam with a learning rate of 0.01, 30
    epochs in batches of 64 hours, on the training days. On one thread, so that the numbers are the ones the lesson's
    text gives."""
    scale = demand.scaling(TRAIN_HOURS)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = demand.inputs(TRAIN_HOURS, scale), demand.targets(TRAIN_HOURS)
        torch.manual_seed(0)
        model = demand.DemandNet(16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(30):
            demand.train_epoch(model, opt, x, y, 64)
    finally:
        torch.set_num_threads(threads)
    model.eval()
    return model, scale


def _model():
    """Kept once per Python process, for both languages: a service loads its model once, when it starts."""
    return kept((LESSON, "model"), _trained)


_model()   # trained when the lesson loads, so that no check has to wait for it

CONFIG = {"mean": list(_model()[1]["mean"]), "std": list(_model()[1]["std"])}   # as config.json keeps them


def one_hot(hr) -> torch.Tensor:
    """The hour as 24 numbers, as in training: a 1 at the hour and 0 everywhere else."""
    return torch.nn.functional.one_hot(torch.tensor(int(hr)), 24).float()


def expect(x) -> float:
    """What the course's model makes of the 34 inputs of one hour: the bikes, as a plain Python number."""
    model, _ = _model()
    with torch.no_grad():
        return float(model(torch.as_tensor(x, dtype=torch.float32).reshape(1, 34))[0, 0])


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


# ======== the requests of the lesson =====================================================================================

_WHOLE = {"hr", "weekday", "workingday", "holiday", "weathersit", "season", "mnth"}


class _Hour(dict):
    """The body of a request from the lesson's cells: the check's call shows it by its name, and the learner's
    function gets a plain copy."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


def _from_data(day: int, hr: int) -> dict:
    row = HOURS_2011[(HOURS_2011["day"] == day) & (HOURS_2011["hr"] == hr)].iloc[0]
    return {name: int(row[name]) if name in _WHOLE else round(float(row[name]), 4) for name in FIELDS}


def rented(day: int, hr: int) -> int:
    """How many bikes were rented in that hour."""
    return int(HOURS_2011[(HOURS_2011["day"] == day) & (HOURS_2011["hr"] == hr)]["cnt"].iloc[0])


DAY = [_Hour(f"C.DAY[{h}]", _from_data(WATCH_DAY, h)) for h in range(24)]   # every hour of 14 July 2011
RAIN = _Hour("C.RAIN", _from_data(RAIN_DAY, 17))                             # 7 December 2011 at 17:00, light rain
RENTED_DAY = [rented(WATCH_DAY, h) for h in range(24)]


# ======== exercise 1: prepare ============================================================================================

_P_HEAD = ("def prepare(request):\n"
           "    hour = one_hot(request[\"hr\"])                                  # 24 numbers: a 1 at the hour\n"
           "    values = torch.tensor([float(request[f]) for f in FEATURES])   # the 10 features of the request\n")
_P_END = "    return torch.cat([hour, scaled])                               # 34 numbers, as in training\n"


def _p(line: str) -> str:
    return _P_HEAD + f"    scaled = {line}\n" + _P_END


_P_START = _p("...")
_P = _p('(values - torch.tensor(CONFIG["mean"])) / torch.tensor(CONFIG["std"])')
_P_NONE = _p("values")
_P_OWN = _p("(values - values.mean()) / values.std()")
_P_PARENS = _p('values - torch.tensor(CONFIG["mean"]) / torch.tensor(CONFIG["std"])')
_P_SWAPPED = _p('(values - torch.tensor(CONFIG["std"])) / torch.tensor(CONFIG["mean"])')
_P_TIMES = _p('(values - torch.tensor(CONFIG["mean"])) * torch.tensor(CONFIG["std"])')
_P_LIST = _p('(values - CONFIG["mean"]) / CONFIG["std"]')

_P_NAMES = {"torch": torch, "one_hot": one_hot, "FEATURES": FEATURES, "CONFIG": CONFIG}
prepare = _defined(_P, "prepare", _P_NAMES)   # the course's prepare: as in training


@dataclass(frozen=True)
class Odd:
    """What a check sees of an output of prepare that is not 34 numbers in a tensor: what it is instead."""
    kind: str
    shape: tuple = ()


def _inputs(v):
    """What a check compares of what prepare gives back: the 34 numbers, or what it is instead."""
    if v is None:
        return None   # nothing back: the check says so in its own words
    if isinstance(v, torch.Tensor) and v.is_floating_point() and tuple(v.shape) == (34,):
        return [float(x) for x in v.tolist()]
    if isinstance(v, torch.Tensor):
        return Odd("tensor", tuple(v.shape))
    return Odd(type(v).__name__)


def _same_inputs(a, b) -> bool:
    """The same 34 inputs within 0.0001: float32 and float64 give a slightly different last digit."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-4) or math.isnan(x) and math.isnan(y)
                                        for x, y in zip(a, b))
    return a == b


def _odd_text(v: Odd, lang: str) -> str:
    if v.kind == "tensor":
        shape = "[" + ", ".join(str(n) for n in v.shape) + "]"
        return {"en": f"a tensor of shape {shape}", "nl": f"een tensor met vorm {shape}"}[lang]
    return {"en": f"a value of the kind {v.kind}", "nl": f"een waarde van de soort {v.kind}"}[lang]


def _short_list(xs: list) -> str:
    return "[" + ", ".join(f"{x:.2f}" for x in xs) + "]"


def _shown_inputs(v):
    """{got} and {want} of prepare: the 10 numbers after the 24 of the hour, or what came back instead."""
    if isinstance(v, Odd):
        return {lang: _odd_text(v, lang) for lang in LANGS}
    shown = "`" + _short_list(v[24:]) + "`"
    return {"en": f"34 numbers ending in {shown}", "nl": f"34 getallen die eindigen op {shown}"}


def _the_list(e: BaseException) -> bool:
    """A tensor and a list in one subtraction: CONFIG["mean"] used as it is, without torch.tensor."""
    return isinstance(e, TypeError) and "'Tensor' and 'list'" in str(e)


# ======== exercise 2: decide =============================================================================================

_D_START = ("def decide(pred):\n"
            "    bikes = ...\n"
            "    extra_crew = bikes >= THRESHOLD   # the decision: from THRESHOLD bikes on, the extra crew goes out\n"
            "    return {\"bikes\": bikes, \"extra_crew\": extra_crew}\n")


def _d(line: str) -> str:
    return _D_START.replace("    bikes = ...\n", f"    bikes = {line}\n")


_D = _d("max(0, round(pred))")
_D_ROUND = _d("round(pred)")
_D_ABS = _d("abs(round(pred))")
_D_MIN = _d("min(0, round(pred))")
_D_FLOAT = _d("max(0, pred)")
_D_INT = _d("max(0, int(pred))")

decide = _defined(_D, "decide", {"THRESHOLD": THRESHOLD})   # the course's decide


def _shown_decision(v):
    """{got} and {want} of decide: what it gives back, as Python writes it."""
    text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


# ======== the numbers of the lesson ======================================================================================

def _preds(hours: pd.DataFrame, scale: dict) -> np.ndarray:
    model, _ = _model()
    with torch.no_grad():
        return model(demand.inputs(hours, scale)).squeeze(1).numpy().astype(float)


def fit_batch(hours: pd.DataFrame) -> dict:
    """The mean and the standard deviation of every feature over a batch of hours, the way `fit` of a scaler such as
    scikit-learn's StandardScaler learns them: the spread of the batch itself, and 1 where a feature has one value in
    every row, so that it becomes 0."""
    std = [float(hours[f].std(ddof=0)) for f in FEATURES]
    return {"mean": [float(hours[f].mean()) for f in FEATURES], "std": [s if s > 0 else 1.0 for s in std]}


NO_SCALE = {"mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES)}
SOURCES = ("training", "batch", "none")   # where the serving path takes its mean and standard deviation from


def day_preds(day: int, source: str) -> list:
    """What the model makes for the 24 hours of a day, prepared with the scale of the training hours, of the day's own
    batch, or with no scaling at all."""
    hours = HOURS_2011[HOURS_2011["day"] == day]
    scale = {"training": CONFIG, "batch": fit_batch(hours), "none": NO_SCALE}[source]
    return [float(p) for p in _preds(hours, scale)]


def _whole(p: float) -> int:
    return max(0, round(p))


def _validation():
    """For each source of the scale, the model's numbers on every validation hour, day by day, with the rentals."""
    out = {}
    for source in SOURCES:
        preds, real = [], []
        for day, hours in VAL_HOURS.groupby("day"):
            preds += day_preds(int(day), source)
            real += [int(c) for c in hours["cnt"]]
        out[source] = (np.array(preds), np.array(real))
    return out


VALIDATION = kept((LESSON, "validation"), _validation)


def off_by(source: str) -> float:
    """How many bikes the model is off per hour on average, over the 1,733 validation hours."""
    preds, real = VALIDATION[source]
    return float(np.mean(np.abs(preds - real)))


def counts(threshold: int, source: str = "training") -> dict:
    """The decisions of decide with this threshold over the validation hours, against the busy hours: a crew in a busy
    hour, a crew for nothing, and a busy hour without a crew."""
    preds, real = VALIDATION[source]
    crew = np.array([_whole(p) >= threshold for p in preds])
    busy = real >= BUSY
    return {"caught": int((crew & busy).sum()), "needless": int((crew & ~busy).sum()),
            "missed": int((~crew & busy).sum()), "busy": int(busy.sum())}


def cost(threshold: int) -> int:
    c = counts(threshold)
    return COSTS["needless"] * c["needless"] + COSTS["missed"] * c["missed"]


THRESHOLDS = (300, 350, 400, 450)


def worked_example() -> list:
    """What the worked example prints for 8:00 and 4:00 on 14 July."""
    out = []
    for request in (DAY[8], DAY[4]):
        x = prepare(request)
        pred = expect(x)
        out.append(([round(v, 2) for v in x[24:].tolist()], round(pred, 1), decide(pred)))
    return out


# ======== small pieces of pictures ========================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", font: str = "") -> str:
    family = "font-family:var(--ea-font-code);" if font == "code" else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{family}font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _hours(values: list) -> str:
    return ", ".join(values)


def _list_of(items: list, lang: str) -> str:
    word = {"en": " and ", "nl": " en "}[lang]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def day_svg(lang: str, alt: str, bars: list, lines: tuple = (), line_at: float | None = None,
            dots: list = (), bar_fill: str = "var(--ea-ink-3)") -> str:
    """A day of 24 hours: a bar per hour, lines of other numbers per hour, a level line, and a dot above some hours."""
    L, R, TOP, H = 32, 326, 10, 150
    values = list(bars) + [v for _, vs in lines for v in vs] + ([line_at] if line_at is not None else [])
    hi = max(max(values), 100)
    lo = min(min(values), 0)
    span = hi - lo
    ys = lambda v: TOP + H * (hi - v) / span   # noqa: E731
    step = 100 if hi <= 800 else (500 if hi <= 4000 else 2000)
    width = (R - L) / 24
    parts = []
    tick = math.ceil(lo / step) * step
    while tick <= hi:
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t(L - 5, ys(tick) + 4, number(tick, 0, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
        tick += step
    for h, v in enumerate(bars):
        top, bottom = (ys(v), ys(0)) if v >= 0 else (ys(0), ys(v))
        parts.append(f'<rect x="{L + h * width + 1.5:.1f}" y="{top:.1f}" width="{width - 3:.1f}" '
                     f'height="{max(bottom - top, 0.8):.1f}" style="fill:{bar_fill}"/>')
    parts.append(f'<line x1="{L}" y1="{ys(0):.1f}" x2="{R}" y2="{ys(0):.1f}" style="stroke:var(--ea-ink-2);stroke-width:1"/>')
    if line_at is not None:
        parts.append(f'<line x1="{L}" y1="{ys(line_at):.1f}" x2="{R}" y2="{ys(line_at):.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    for colour, vs in lines:
        points = " ".join(f"{L + (h + 0.5) * width:.1f},{ys(v):.1f}" for h, v in enumerate(vs))
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:{colour};stroke-width:2;stroke-linejoin:round"/>')
    for h in dots:
        parts.append(f'<circle cx="{L + (h + 0.5) * width:.1f}" cy="{TOP + 3:.1f}" r="3" style="fill:var(--ea-ink)"/>')
    for h in (0, 6, 12, 18):
        parts.append(_t(L + (h + 0.5) * width, TOP + H + 16, clock(h), 10.5, 400, "var(--ea-ink-3)", "middle"))
    return _svg(TOP + H + 22, alt, parts)


def _key(items: list) -> str:
    """A small legend: a swatch and a word for each kind of mark."""
    out = []
    for kind, colour, word in items:
        mark = (f'<span style="display:inline-block;width:14px;height:9px;background:{colour};margin-right:4px"></span>'
                if kind == "bar" else
                f'<span style="display:inline-block;width:14px;height:0;border-top:2px {kind} {colour};'
                f'margin-right:4px;vertical-align:middle"></span>')
        out.append(f'<span style="margin-right:12px;white-space:nowrap">{mark}{html.escape(word)}</span>')
    return f'<div style="{_UI}">{"".join(out)}</div>'


# ======== the picture of the idea ========================================================================================

IDEA_TEXT = {
    "en": {"title": "The serving path of one request",
           "steps": [("request", "temp 0.64"), ("check", "no error"), ("prepare", "temp 0.76"),
                     ("model", "431.2"), ("decide", "431, crew")],
           "train": "training: the same prepare, the same numbers",
           "before": "before the model", "after": "after the model",
           "alt": "Five boxes from left to right for 8:00 on 14 July 2011: the request, with temp 0.64; the check, which "
                  "finds nothing wrong; prepare, which standardizes temp with the mean and standard deviation of the "
                  "training hours and makes it 0.76; the model, which makes 431.2; and decide, which makes 431 bikes of "
                  "it and sends the extra crew. Below prepare, a line to training: the same prepare, with the same "
                  "numbers."},
    "nl": {"title": "Het serving-pad van één verzoek",
           "steps": [("verzoek", "temp 0,64"), ("controle", "niets mis"), ("prepare", "temp 0,76"),
                     ("model", "431,2"), ("decide", "431, crew")],
           "train": "training: dezelfde prepare, dezelfde getallen",
           "before": "naar het model toe", "after": "na het model",
           "alt": "Vijf vakken van links naar rechts voor 8:00 op 14 juli 2011: het verzoek, met temp 0,64; de controle, "
                  "zonder fout; prepare, met het gemiddelde en de standaardafwijking van de trainingsuren, waardoor temp "
                  "0,76 wordt; het model, met 431,2; en decide, met 431 fietsen en de extra crew. Onder prepare een lijn "
                  "naar de training: dezelfde prepare, met dezelfde getallen."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    boxes = []
    x, gap, bw = 4, 6, 60
    for k, (name, value) in enumerate(w["steps"]):
        bx = x + k * (bw + gap)
        stroke = "var(--ea-pred)" if name in ("model",) else "var(--ea-ink-2)"
        boxes.append(f'<rect x="{bx}" y="38" width="{bw}" height="44" rx="4" style="fill:var(--ea-inset);stroke:{stroke};stroke-width:1.2"/>'
                     + _t(bx + bw / 2, 55, name, 11.5, 600, "var(--ea-ink)", "middle")
                     + _t(bx + bw / 2, 72, value, 9.5, 400, "var(--ea-ink-2)", "middle"))
        if k:
            ax = bx - gap
            boxes.append(f'<path d="M{ax - 1:.1f},60 l{gap - 1:.1f},0" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>'
                         f'<path d="M{bx - 4:.1f},57 l3,3 l-3,3" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.2"/>')
    left, right = x + 2 * (bw + gap), x + 4 * (bw + gap)
    boxes.append(_t((x + left + bw) / 2, 28, w["before"], 10, 400, "var(--ea-ink-3)", "middle"))
    boxes.append(f'<line x1="{x}" y1="32" x2="{left + bw}" y2="32" style="stroke:var(--ea-line);stroke-width:1"/>')
    boxes.append(f'<line x1="{right}" y1="32" x2="{right + bw}" y2="32" style="stroke:var(--ea-line);stroke-width:1"/>')
    boxes.append(_t(right + bw, 28, w["after"], 10, 400, "var(--ea-ink-3)", "end"))
    cx = left + bw / 2
    boxes.append(f'<line x1="{cx}" y1="82" x2="{cx}" y2="98" style="stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:3 2"/>')
    boxes.append(_t(cx, 112, w["train"], 9.5, 400, "var(--ea-ink-2)", "middle"))
    title = _t(4, 14, w["title"], 13, 600)
    return _svg(122, w["alt"], [title] + boxes)


# ======== the pictures of the learner's work ==============================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error or took too long
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


PRED_DAY = [expect(prepare(r)) for r in DAY]   # the course's serving path on 14 July: 431.2 at 8:00


def _mean_off(preds: list, real: list) -> float:
    return sum(abs(p - r) for p, r in zip(preds, real)) / len(real)


YOURS_TEXT = {
    "en": {"prepare": {"title": "What the model makes from your inputs on 14 July 2011",
                       "intro": "For each hour of the day, your `prepare(request)` makes the inputs, and the course's model makes the bikes from them.",
                       "key": ("from your inputs", "rented"),
                       "at": "At 8:00: {mine} bikes from your inputs; {real} were rented.",
                       "off": "Over the day, {off} bikes off per hour on average; with the inputs of training, {course}.",
                       "same": "Your inputs are the ones of training, so the model gives the numbers it gave in validation.",
                       "none": "No picture yet: your `prepare` does not give 34 numbers in a tensor for every hour.",
                       "alt": "A bar for each hour of 14 July 2011 with the bikes the model makes from the learner's inputs, and a line with the bikes that were rented."},
           "decide": {"title": "Your decisions for 14 July 2011",
                      "intro": "The course's model makes a number for each hour, and your `decide(pred)` turns it into bikes and a decision.",
                      "key": ("bikes", "the threshold"),
                      "crew": "Extra crew at {hours}.",
                      "no_crew": "No extra crew on this day.",
                      "below": "{hours} below 0 bikes: {values}.",
                      "fine": "No hour below 0 bikes, and every number is whole.",
                      "fraction": "{hours} with a fraction of a bike.",
                      "none": "No picture yet: your `decide` does not give a dict with a number under `bikes` for every hour.",
                      "alt": "A bar for each hour of 14 July 2011 with the learner's bikes, a level line at the threshold, and a dot above each hour where the extra crew goes out."}},
    "nl": {"prepare": {"title": "Wat het model maakt van je inputs op 14 juli 2011",
                       "intro": "Voor elk uur van de dag maakt je `prepare(request)` de inputs, en het model van de course maakt er de fietsen van.",
                       "key": ("uit je inputs", "verhuurd"),
                       "at": "Om 8:00: {mine} fietsen uit je inputs; er werden er {real} verhuurd.",
                       "off": "Over de dag zit het model gemiddeld {off} fietsen per uur ernaast; met de inputs van de training {course}.",
                       "same": "Je inputs zijn gelijk aan de inputs van de training, dus geeft het model de getallen van de validatie.",
                       "none": "Nog geen afbeelding: je `prepare` geeft niet voor elk uur 34 getallen in een tensor.",
                       "alt": "Een staaf voor elk uur van 14 juli 2011 met de fietsen die het model maakt uit de inputs van de leerling, en een lijn met de verhuurde fietsen."},
           "decide": {"title": "Je beslissingen voor 14 juli 2011",
                      "intro": "Het model van de course maakt een getal voor elk uur, en je `decide(pred)` maakt er fietsen en een beslissing van.",
                      "key": ("fietsen", "de grens"),
                      "crew": "Extra crew om {hours}.",
                      "no_crew": "Geen extra crew op deze dag.",
                      "below": "{hours} onder 0 fietsen: {values}.",
                      "fine": "Geen uur onder 0 fietsen, en elk getal is heel.",
                      "fraction": "{hours} met een deel van een fiets.",
                      "none": "Nog geen afbeelding: je `decide` geeft niet voor elk uur een dict met een getal onder `bikes`.",
                      "alt": "Een staaf voor elk uur van 14 juli 2011 met de fietsen van de leerling, een lijn op de hoogte van de grens, en een stip boven elk uur waarin de extra crew op pad gaat."}},
}


def prepare_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["prepare"]
    fn = _fn(seen)
    got = _call(lambda: [_inputs(fn(copy.deepcopy(dict(r)))) for r in DAY]) if fn is not None else _STOPPED
    if got is _STOPPED or any(not isinstance(v, list) for v in got):   # nothing back, or not 34 numbers in a tensor
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    mine = [expect(v) for v in got]
    if not all(math.isfinite(p) for p in mine):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    same = all(abs(a - b) < 0.05 for a, b in zip(mine, PRED_DAY))
    colour = "var(--ea-pred)" if same else "var(--ea-loss)"
    svg = day_svg(lang, w["alt"], mine, lines=(("var(--ea-data)", RENTED_DAY),), bar_fill=colour)
    said = w["at"].format(mine=number(mine[8], 1, lang), real=RENTED_DAY[8]) + " " + w["off"].format(
        off=number(_mean_off(mine, RENTED_DAY), 1, lang), course=number(_mean_off(PRED_DAY, RENTED_DAY), 1, lang))
    body = (_note(w["intro"]) + _key([("bar", colour, w["key"][0]), ("solid", "var(--ea-data)", w["key"][1])]) + svg
            + _note(said) + (_note(w["same"]) if same else ""))
    return yours.figure(w["title"], body, alt=w["title"])


def _bikes(v):
    """The number under bikes of what decide gave back, or None."""
    if not isinstance(v, dict) or "bikes" not in v:
        return None
    b = v["bikes"]
    return b if not isinstance(b, bool) and isinstance(b, (int, float)) and math.isfinite(b) else None


def decide_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["decide"]
    fn = _fn(seen)
    got = _call(lambda: [fn(p) for p in PRED_DAY]) if fn is not None else _STOPPED
    values = [_bikes(v) for v in got] if got is not _STOPPED else [None]
    if any(v is None for v in values):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    crew = [h for h, v in enumerate(got) if v.get("extra_crew") is True]
    svg = day_svg(lang, w["alt"], values, line_at=THRESHOLD, dots=crew)
    said = [w["crew"].format(hours=_list_of([clock(h) for h in crew], lang)) if crew else w["no_crew"]]
    below = [h for h in range(24) if values[h] < 0]
    parts = [h for h in range(24) if values[h] != int(values[h])]
    if below:
        said.append(w["below"].format(hours=_list_of([clock(h) for h in below], lang),
                                      values=_list_of([number(values[h], 0 if values[h] == int(values[h]) else 1, lang)
                                                       for h in below], lang)))
    if parts:
        said.append(w["fraction"].format(hours=_list_of([clock(h) for h in parts[:3]], lang)))
    if not below and not parts:
        said.append(w["fine"])
    body = (_note(w["intro"]) + _key([("bar", "var(--ea-ink-3)", w["key"][0]), ("dashed", "var(--ea-ink)", w["key"][1])])
            + svg + _note(" ".join(said)))
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: where the serving path takes its scale from ===========================================================

DAYS = {"thu": 194, "sat": 189, "tue": 199, "wed": 214, "oct": 299}   # validation days: the model never saw them

DEMO_TEXT = {
    "en": {"title": "The night batch for {day}",
           "days": {"thu": "Thursday 14 July", "sat": "Saturday 9 July", "tue": "Tuesday 19 July",
                    "wed": "Wednesday 3 August", "oct": "Thursday 27 October"},
           "sources": {"training": "the training hours (config.json)", "batch": "the 24 hours of the batch itself",
                       "none": "nothing: no scaling"},
           "key": ("rented", "with the scale of training", "with the scale you chose", "threshold 350"),
           "lead": {"training": "With the scale of the training hours", "batch": "With the scale of the batch itself",
                    "none": "Without scaling"},
           "line": "{lead}: {off} bikes off per hour on average on this day, extra crew in {caught} of the {busy} busy hours, and {needless} for nothing.",
           "all": "Over all 73 validation days: {off} bikes off per hour on average, where the scale of training gives {train}.",
           "same": "This is the serving path as it should be: the scale of training.",
           "alt": "Bars with the bikes rented in each hour of the day, a line with what the model makes with the scale of training, a line with what it makes with the chosen scale, a level line at 350 bikes, and a dot above each hour where the chosen path sends the extra crew."},
    "nl": {"title": "De nachtelijke batch voor {day}",
           "days": {"thu": "donderdag 14 juli", "sat": "zaterdag 9 juli", "tue": "dinsdag 19 juli",
                    "wed": "woensdag 3 augustus", "oct": "donderdag 27 oktober"},
           "sources": {"training": "de trainingsuren (config.json)", "batch": "de 24 uren van de batch zelf",
                       "none": "niets: geen schaling"},
           "key": ("verhuurd", "met de schaal van de training", "met de gekozen schaal", "grens 350"),
           "lead": {"training": "Met de schaal van de trainingsuren", "batch": "Met de schaal van de batch zelf",
                    "none": "Zonder schaling"},
           "line": "{lead}: gemiddeld {off} fietsen per uur ernaast op deze dag, extra crew in {caught} van de {busy} drukke uren, en {needless} keer voor niets.",
           "all": "Over alle 73 validatiedagen: gemiddeld {off} fietsen per uur ernaast, waar de schaal van de training {train} geeft.",
           "same": "Dit is het serving-pad zoals het hoort: de schaal van de training.",
           "alt": "Staven met de verhuurde fietsen in elk uur van de dag, een lijn met wat het model maakt met de schaal van de training, een lijn met wat het maakt met de gekozen schaal, een lijn op de hoogte van 350 fietsen, en een stip boven elk uur waarin het gekozen pad de extra crew stuurt."},
}


def day_picture(day_key: str, source: str, lang: str) -> str:
    """The demo: one validation day, the model's numbers with the scale of training and with the chosen scale, and the
    decisions of the chosen path at the threshold of Exercise 3."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    day_key = day_key if day_key in DAYS else "thu"
    source = source if source in SOURCES else "training"
    day = DAYS[day_key]
    real = [rented(day, h) for h in range(24)]
    train = day_preds(day, "training")
    chosen = day_preds(day, source)
    crew = [h for h in range(24) if _whole(chosen[h]) >= CHOSEN]
    busy = [h for h in range(24) if real[h] >= BUSY]
    lines = [("var(--ea-pred)", train)] + ([("var(--ea-loss)", chosen)] if source != "training" else [])
    key = [("bar", "var(--ea-data)", w["key"][0]), ("solid", "var(--ea-pred)", w["key"][1])]
    if source != "training":
        key.append(("solid", "var(--ea-loss)", w["key"][2]))
    key.append(("dashed", "var(--ea-ink)", w["key"][3]))
    svg = day_svg(lang, w["alt"], real, lines=tuple(lines), line_at=CHOSEN, dots=crew, bar_fill="var(--ea-data)")
    said = w["line"].format(lead=w["lead"][source], off=number(_mean_off(chosen, real), 1, lang),
                            caught=len(set(crew) & set(busy)), busy=len(busy), needless=len(set(crew) - set(busy)))
    total = w["all"].format(off=number(off_by(source), 1, lang), train=number(off_by("training"), 1, lang))
    body = _key(key) + svg + _note(said) + _note(w["same"] if source == "training" else total)
    return yours.figure(w["title"].format(day=w["days"][day_key]), body, alt=w["alt"])


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

SIDES = Sort(
    name="sides",
    groups={
        "before": {"en": "Before the model: as in training", "nl": "Naar het model toe: zoals bij de training"},
        "after": {"en": "After the model: from number to answer", "nl": "Na het model: van getal naar antwoord"},
    },
    items={
        "scale": {"en": "**Standardize.** temp minus the mean of the training hours, divided by their standard deviation.",
                  "nl": "**Standaardiseren.** temp min het gemiddelde van de trainingsuren, gedeeld door de standaardafwijking van de trainingsuren."},
        "hour": {"en": "**The hour.** 24 columns with a 1 at the hour, in the order of training.",
                 "nl": "**Het uur.** 24 kolommen met een 1 bij het uur, in de volgorde van de training."},
        "degrees": {"en": "**Degrees.** A temperature from the app in degrees, divided by 41, as the data has it.",
                    "nl": "**Graden.** Een temperatuur uit de app in graden, gedeeld door 41, zoals in de data."},
        "order": {"en": "**The order.** The 10 features in the order of training.",
                  "nl": "**De volgorde.** De 10 features in de volgorde van de training."},
        "zero": {"en": "**Below 0.** A number below 0 becomes 0 bikes.",
                 "nl": "**Onder 0.** Een getal onder 0 wordt 0 fietsen."},
        "crew": {"en": "**The crew.** From 350 bikes on, the extra crew goes out.",
                 "nl": "**De crew.** Vanaf 350 fietsen gaat de extra crew op pad."},
        "night": {"en": "**At night.** No crew goes out between midnight and 6:00.",
                  "nl": "**'s Nachts.** Tussen middernacht en 6:00 gaat geen crew op pad."},
        "whole": {"en": "**Whole bikes.** The number is rounded to a whole number of bikes.",
                  "nl": "**Hele fietsen.** Het getal wordt afgerond op een heel aantal fietsen."},
    },
    solution={"scale": "before", "hour": "before", "degrees": "before", "order": "before",
              "zero": "after", "crew": "after", "night": "after", "whole": "after"},
    nudge={
        "scale": {"en": "Does the model need this to read its input as in training, or is it a rule about what comes out?",
                  "nl": "Heeft het model dit nodig om de input te lezen zoals bij de training, of is het een regel over wat eruit komt?"},
        "hour": {"en": "The model learned the hour as 24 columns. Is this about what goes in, or about what comes out?",
                 "nl": "Het model leerde het uur als 24 kolommen. Gaat dit over wat erin gaat, of over wat eruit komt?"},
        "degrees": {"en": "The data holds temp as degrees divided by 41. Does the model see this number, or the business?",
                    "nl": "De data heeft temp als graden gedeeld door 41. Ziet het model dit getal, of het bedrijf?"},
        "order": {"en": "Each weight belongs to one column. Does the order matter to the model, or to the planners?",
                  "nl": "Elke weight hoort bij één kolom. Doet de volgorde ertoe voor het model, of voor de planners?"},
        "zero": {"en": "The model can make −10.8. Who decides that a count of bikes is never below 0: the model or the business?",
                 "nl": "Het model kan −10,8 maken. Wie bepaalt dat een aantal fietsen nooit onder 0 ligt: het model of het bedrijf?"},
        "crew": {"en": "The model gives a number, not a decision. Which side turns the number into a decision?",
                 "nl": "Het model geeft een getal, geen beslissing. Welke kant maakt van het getal een beslissing?"},
        "night": {"en": "Is a rule about when a crew may work something the model learned, or something the business decided?",
                  "nl": "Is een regel over wanneer een crew mag werken iets wat het model leerde, of iets wat het bedrijf besloot?"},
        "whole": {"en": "The model makes 431.2. Does rounding help the model read its input, or the planners read its answer?",
                  "nl": "Het model maakt 431,2. Helpt afronden het model de input te lezen, of de planners het antwoord?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the steps again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de stappen opnieuw uit."},
        "empty": {"en": "Put each step on its side of the model.",
                  "nl": "Zet elke stap aan de kant van het model waar de stap hoort."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs on the other side.",
                  "nl": "*{item}* hoort aan de andere kant."},
        "correct": {"en": "All {n} are right. Before the model, every step does what training did, with the numbers of training; after the model, the business decides what the number means and what may never happen.",
                    "nl": "Alle {n} kloppen. Naar het model toe doet elke stap wat de training deed, met de getallen van de training; na het model bepaalt het bedrijf wat het getal betekent en wat nooit mag gebeuren."},
    },
)

EXERCISES = {
    "prepare": FunctionBox(
        name="prepare",
        function="prepare",
        start=_P_START,
        solution=_P,
        names=_P_NAMES,
        visible=[(DAY[8],)],
        hidden=[(DAY[17],), (DAY[3],), (RAIN,)],
        first=[_DOTS],
        collect=_inputs,
        compare=_same_inputs,
        show=_shown_inputs,
        picture=prepare_picture,
        mistakes=[
            Mistake("list", error=_the_list,
                    en="With `{call}`, a tensor and a list meet in one subtraction, and PyTorch stops: `CONFIG[\"mean\"]` is a plain Python list. What turns a list into a tensor?",
                    nl="Met `{call}` komen een tensor en een list samen in één aftrekking, en PyTorch stopt: `CONFIG[\"mean\"]` is een gewone Python-list. Wat maakt van een list een tensor?"),
            Mistake("none", wrong=_P_NONE,
                    en="`{call}` gives {got}: the features straight from the request. The model learned its weights for standardized features, so to the model, month 7 lies far beyond December. Which mean and standard deviation did training use?",
                    nl="`{call}` geeft {got}: de features rechtstreeks uit het verzoek. Het model leerde de weights voor gestandaardiseerde features, dus ligt maand 7 voor het model ver voorbij december. Welk gemiddelde en welke standaardafwijking gebruikte de training?"),
            Mistake("own", wrong=_P_OWN,
                    en="`{call}` gives {got}: you standardize with the mean and spread of this one request, over 10 features that have nothing to do with each other, such as the month and the humidity. Which numbers did training standardize with?",
                    nl="`{call}` geeft {got}: je standaardiseert met het gemiddelde en de spreiding van dit ene verzoek, over 10 features die niets met elkaar te maken hebben, zoals de maand en de vochtigheid. Met welke getallen standaardiseerde de training?"),
            Mistake("parens", wrong=_P_PARENS,
                    en="`{call}` gives {got}: Python divides before it subtracts, so only the mean is divided by the standard deviation. Where do the parentheses go?",
                    nl="`{call}` geeft {got}: Python deelt voordat het aftrekt, dus wordt alleen het gemiddelde gedeeld door de standaardafwijking. Waar horen de haakjes?"),
            Mistake("swapped", wrong=_P_SWAPPED,
                    en="`{call}` gives {got}: you subtract the standard deviation and divide by the mean. Which of the two is subtracted in standardization?",
                    nl="`{call}` geeft {got}: je trekt de standaardafwijking af en deelt door het gemiddelde. Welke van de twee trek je af bij standaardiseren?"),
            Mistake("times", wrong=_P_TIMES,
                    en="`{call}` gives {got}: you multiply by the standard deviation instead of dividing by it, so the spread does not become 1. Is that how training did it?",
                    nl="`{call}` geeft {got}: je vermenigvuldigt met de standaardafwijking in plaats van erdoor te delen, dus wordt de spreiding geen 1. Deed de training het zo?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what `scaled` should be: `values` standardized as in training.",
                      "nl": "Vervang de `...` door wat `scaled` moet zijn: `values` gestandaardiseerd zoals bij de training."},
            "arguments": {"en": "The check calls `{call}`, with one dict, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(request)`?",
                          "nl": "De check roept `{call}` aan, met één dict, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(request)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you subtract the mean of the training hours, and divide by their standard deviation?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Trek je het gemiddelde van de trainingsuren af, en deel je door de standaardafwijking van de trainingsuren?"},
            "almost": {"en": "Right for 8:00. But `{call}` gives {got}, and it should give {want}. Does `scaled` come from `values`, and not from fixed numbers?",
                       "nl": "Goed voor 8:00. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt `scaled` uit `values`, en niet uit vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return torch.cat([hour, scaled])`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return torch.cat([hour, scaled])`?"},
            "correct": {"en": "Right in all {n} cases. At 8:00, temp 0.64 goes in as 0.76, as in training, and the model makes 431.2 bikes again. In the picture: how far is the model off over the day?",
                        "nl": "Goed in alle {n} gevallen. Om 8:00 gaat temp 0,64 erin als 0,76, net als bij de training, en het model maakt weer 431,2 fietsen. In de afbeelding: hoe ver zit het model over de dag ernaast?"},
        },
    ),
    "decide": FunctionBox(
        name="decide",
        function="decide",
        start=_D_START,
        solution=_D,
        names={"THRESHOLD": THRESHOLD},
        visible=[(-10.8,)],
        hidden=[(431.2,), (375.7,), (537.3,), (-6.3,), (399.6,)],
        first=[_DOTS],
        show=_shown_decision,
        picture=decide_picture,
        mistakes=[
            Mistake("round-only", wrong=_D_ROUND,
                    en="`{call}` gives {got}: −11 bikes, a number no planner can use. Which Python function gives the larger of 0 and the rounded number?",
                    nl="`{call}` geeft {got}: −11 fietsen, een getal waar geen planner iets mee kan. Welke Python-functie geeft de grootste van 0 en het afgeronde getal?"),
            Mistake("abs", wrong=_D_ABS,
                    en="`{call}` gives {got}: the model said fewer than none, and the answer now says 11 bikes. Which function keeps 0 as the lowest number, instead of turning the sign around?",
                    nl="`{call}` geeft {got}: het model zei minder dan niets, en het antwoord zegt nu 11 fietsen. Welke functie houdt 0 als laagste getal, in plaats van het teken om te draaien?"),
            Mistake("min", wrong=_D_MIN,
                    en="`{call}` gives {got}: `min` keeps the smaller of 0 and the number, so −11 stays −11, and a busy hour of 431.2 would get 0 bikes. Which function gives the larger of the two?",
                    nl="`{call}` geeft {got}: `min` houdt de kleinste van 0 en het getal, dus blijft −11 gewoon −11, en zou een druk uur van 431,2 dan 0 fietsen krijgen. Welke functie geeft de grootste van de twee?"),
            Mistake("float", wrong=_D_FLOAT, state="almost",
                    en="Right for 4:00. But `{call}` gives {got}: a number of bikes with a fraction. What rounds it to a whole number first?",
                    nl="Goed voor 4:00. Maar `{call}` geeft {got}: een aantal fietsen met cijfers achter de komma. Wat rondt het getal eerst af op een heel getal?"),
            Mistake("int", wrong=_D_INT, state="almost",
                    en="Right for 4:00. But `{call}` gives {got}, and it should give {want}: `int` cuts off everything after the decimal point, so 375.7 becomes 375. Which function rounds to the nearest whole number?",
                    nl="Goed voor 4:00. Maar `{call}` geeft {got}, en het moet {want} zijn: `int` haalt alles na de komma weg, dus wordt 375,7 dan 375. Welke functie rondt af op het dichtstbijzijnde hele getal?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what `bikes` should be: `pred` rounded, and never below 0.",
                      "nl": "Vervang de `...` door wat `bikes` moet zijn: `pred` afgerond, en nooit onder 0."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(pred)`?",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(pred)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `bikes` the rounded `pred`, but never below 0?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `bikes` de afgeronde `pred`, maar nooit onder 0?"},
            "almost": {"en": "Right for 4:00. But `{call}` gives {got}, and it should give {want}. Does `bikes` come from `pred`, and not from a fixed number?",
                       "nl": "Goed voor 4:00. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt `bikes` uit `pred`, en niet uit een vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` of the dict?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de `return` van de dict?"},
            "correct": {"en": "Right in all {n} cases. At 4:00, the model makes −10.8, and the answer is 0 bikes; at 8:00, 431 bikes and the extra crew. In the picture: in which hours does the crew go out?",
                        "nl": "Goed in alle {n} gevallen. Om 4:00 maakt het model −10,8, en het antwoord is 0 fietsen; om 8:00 431 fietsen en de extra crew. In de afbeelding: in welke uren gaat de crew op pad?"},
        },
    ),
    "sides": SIDES,
}

REQUIRED = ["prepare", "decide", "threshold", "sides"]

_SORTED = dict(SIDES.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("prepare", _P_START, "empty", None),
    ("prepare", "", "empty", None),
    ("prepare", _P, "correct", None),
    ("prepare", _p('(values - torch.tensor(CONFIG["mean"])) / torch.tensor(CONFIG["std"]) + 0'), "correct", None),
    ("prepare", _p('torch.tensor([(v - m) / s for v, m, s in zip(values.tolist(), CONFIG["mean"], CONFIG["std"])])'),
     "correct", None),
    ("prepare", _p('(values - torch.tensor(CONFIG["mean"])) / torch.tensor(CONFIG["std"]) if request["hr"] == 8 else values'),
     "almost", None),
    ("prepare", _P_LIST, "error", "list"),
    ("prepare", _P_NONE, "wrong", "none"),
    ("prepare", _P_OWN, "wrong", "own"),
    ("prepare", _P_PARENS, "wrong", "parens"),
    ("prepare", _P_SWAPPED, "wrong", "swapped"),
    ("prepare", _P_TIMES, "wrong", "times"),
    ("prepare", _p('(values - torch.tensor(CONFIG["mean"])) / torch.tensor(CONFIG["mean"])'), "wrong", None),
    ("prepare", _P.replace(_P_END, "    ...\n" + _P_END), "wrong", "dots-left"),
    ("prepare", _P.replace(_P_END, ""), "wrong", "no-return"),
    ("prepare", _P.replace(_P_END, "    return scaled\n"), "wrong", None),
    ("prepare", _P.replace("def prepare(request):", "def prepare():"), "error", "arguments"),
    ("prepare", "x = 1\n", "wrong", "no-function"),
    ("prepare", _P.replace(_P_END, "    print(torch.cat([hour, scaled]))\n"), "wrong", "prints"),
    ("prepare", _p("(values - torch.tensor([0.49, 0.47, 0.64, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) / 0.2"), "wrong", None),
    ("decide", _D_START, "empty", None),
    ("decide", "", "empty", None),
    ("decide", _D, "correct", None),
    ("decide", _d("max(round(pred), 0)"), "correct", None),
    ("decide", _d("round(max(0, pred))"), "correct", None),
    ("decide", _D_ROUND, "wrong", "round-only"),
    ("decide", _D_ABS, "wrong", "abs"),
    ("decide", _D_MIN, "wrong", "min"),
    ("decide", _D_FLOAT, "almost", "float"),
    ("decide", _D_INT, "almost", "int"),
    ("decide", _d("max(0, int(pred))").replace("int(pred)", "int(pred + 0.5)"), "correct", None),
    ("decide", _d("0"), "almost", None),
    ("decide", _d("max(1, round(pred))"), "wrong", None),
    ("decide", _d("max(0, round(pred))\n    ..."), "wrong", "dots-left"),
    ("decide", _D.replace('    return {"bikes": bikes, "extra_crew": extra_crew}\n', ""), "wrong", "no-return"),
    ("decide", _D.replace("def decide(pred):", "def decide(pred, hr):"), "error", "arguments"),
    ("decide", "x = 1\n", "wrong", "no-function"),
    ("decide", _D.replace('    return {"bikes": bikes, "extra_crew": extra_crew}\n',
                          '    print({"bikes": bikes, "extra_crew": extra_crew})\n'), "wrong", "prints"),
    ("sides", {}, "empty", None),
    ("sides", _SORTED, "correct", None),
    ("sides", {"scale": "before", "zero": "after"}, "wrong", "unfinished"),
] + [("sides", _SORTED | {item: "after" if side == "before" else "before"}, "wrong", f"wrong:{item}")
     for item, side in SIDES.solution.items()]
