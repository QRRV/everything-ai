# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-project-tested-model, step 2 of the project ops: the demand model gets its numbers per
group of hours, a known limit marked in its tests, and a model card made from its numbers; then every test of the
project runs at once, with pytest, as a team runs them.

`rmse_per` measures the validation RMSE per value of a column, with the number of hours behind each number, as
disaggregated evaluation does (Mitchell and others, 2019). `test_weekend` is the invariance test of the lesson
*Does the model behave? Behavioral tests* that the course's model fails, marked with pytest's xfail. `card` writes
the card as Markdown. The course trains its model here as the lesson *Project: a demand model, tracked and packed*
trains h16-lr0.01, and a version without the column weekday the same way: that version passes the test of the
weekend, as a model that no longer has the limit.
The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011 and 2012, from the Bike Sharing
dataset of Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the exercises and the pictures are the course's
own."""
import copy
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import everything_ai
from everything_ai import paths, projects, yours
from everything_ai.check import (MISSING, CodeBox, FunctionBox, Mistake, Result, Timeout, is_empty_text, kept,
                                 run_code, same_value, with_time_limit)

LESSON = "sys-project-tested-model"
LANGS = ("en", "nl")

projects.provide("sys-project-tracked-model")   # ops/demand.py and ops/safetensors_io.py
projects.provide("sys-behavior-tests")          # ops/serving.py and ops/conftest.py
projects.builds_on()                            # this lesson imports no module of yours from an earlier lesson


def _given(lesson: str, name: str):
    """The course's own copy of a file it gives in the project: the checks use it, so they stay the same when you
    change yours."""
    path = paths.LESSONS / lesson / "project" / "ops" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_course_ops_{name}_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _given("sys-project-tracked-model", "demand")
safetensors_io = _given("sys-project-tracked-model", "safetensors_io")

HOURS_2011 = demand.read_hours()                            # 8,645 hours of 365 days
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]     # 6,912 hours of 292 days
VAL_HOURS = HOURS_2011[HOURS_2011["day"] % 5 == 4]          # every fifth day: 1,733 hours of 73 days
NO_WEEKDAY = [f for f in demand.FEATURES if f != "weekday"]
BOUND = 20   # bikes: how far apart Saturday and Sunday may be in the test of the weekend


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


# ======== the course's model, and a version without the column weekday ===================================================

class _Narrow(nn.Module):
    """DemandNet without the column weekday: the same layers, made in the same order."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(n_in, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def _scale(features: list) -> dict:
    return {"mean": [round(float(TRAIN_HOURS[f].mean()), 6) for f in features],
            "std": [round(float(TRAIN_HOURS[f].std()), 6) for f in features]}


def _rows(hours, features: list, scale: dict):
    """What demand.inputs gives, for any list of features: the hour as 24 one-hot columns, then each feature
    standardized."""
    hour = nn.functional.one_hot(torch.tensor(hours["hr"].to_numpy()), 24).float()
    values = torch.tensor(hours[features].to_numpy(dtype="float32"))
    return torch.cat([hour, (values - torch.tensor(scale["mean"])) / torch.tensor(scale["std"])], dim=1)


def _trained(key: str):
    """Trained the way the course's train_version trains h16-lr0.01: seed 0, 16 neurons, Adam with a learning rate
    of 0.01, 30 epochs in batches of 64 hours, on the training days, on one thread."""
    features = demand.FEATURES if key == "course" else NO_WEEKDAY
    scale = _scale(features)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = _rows(TRAIN_HOURS, features, scale), demand.targets(TRAIN_HOURS)
        torch.manual_seed(0)
        model = demand.DemandNet(16) if key == "course" else _Narrow(24 + len(features), 16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(30):
            demand.train_epoch(model, opt, x, y, 64)
    finally:
        torch.set_num_threads(threads)
    model.eval()
    return model, scale, features


def _model(key: str):
    """The course's model ("course") or the version without weekday ("no-weekday"), trained once per process."""
    return kept((LESSON, "model", key), lambda: _trained(key))


def _ready() -> None:
    """Train both models before a check, outside its time limit, and work out what the cases of the checks use:
    the first time takes a few seconds."""
    _model("course")
    _model("no-weekday")
    for named in (VAL, PRED, CONFIG, GROUPS, LIMITS):
        named.value()


def _predict(key: str, hours):
    model, scale, features = _model(key)
    with torch.no_grad():
        return model(_rows(hours, features, scale)).reshape(-1)


def course(hours):
    """The course's packed model of your project, served the right way."""
    return _predict("course", hours)


def no_weekday(hours):
    """A version trained without the column weekday: Saturday and Sunday are the same to it."""
    return _predict("no-weekday", hours)


def expect(hours):
    """What the course's packed model expects for each hour of a table: the lesson's cells use it."""
    return course(hours)


def rmse_per(hours, pred, column) -> dict:
    """The course's rmse_per: per value of `column`, the hours and the RMSE of `pred` on them."""
    y = demand.targets(hours).reshape(-1)
    table = {}
    for value in sorted(set(hours[column])):
        mask = torch.tensor((hours[column] == value).to_numpy())
        error = pred[mask] - y[mask]
        table[int(value)] = {"hours": len(error), "rmse": (error ** 2).mean().sqrt().item()}
    return table


def rmse(hours, pred) -> float:
    return ((pred - demand.targets(hours).reshape(-1)) ** 2).mean().sqrt().item()


# the groups of the card, in English as the card is, and in the lesson's language for the pictures
CARD_GROUPS = [("workingday", 1, "working day"), ("workingday", 0, "weekend or holiday"),
               ("weathersit", 1, "clear"), ("weathersit", 2, "mist"), ("weathersit", 3, "light rain or snow"),
               ("weathersit", 4, "heavy rain")]
GROUP_WORDS = {
    "en": {("workingday", 1): "working day", ("workingday", 0): "weekend or holiday", ("weathersit", 1): "clear",
           ("weathersit", 2): "mist", ("weathersit", 3): "light rain or snow", ("weathersit", 4): "heavy rain"},
    "nl": {("workingday", 1): "werkdag", ("workingday", 0): "weekend of feestdag", ("weathersit", 1): "helder",
           ("weathersit", 2): "mist", ("weathersit", 3): "lichte regen of sneeuw", ("weathersit", 4): "zware regen"},
}
BLOCKS = {"en": {"workingday": "Kind of day", "weathersit": "Weather"},
          "nl": {"workingday": "Soort dag", "weathersit": "Weer"}}


def card_groups(tables: dict) -> dict:
    """The rows of the card's table, from rmse_per per column: {name: {"hours": ..., "rmse": ...}}."""
    out = {}
    for column, value, name in CARD_GROUPS:
        row = tables.get(column, {}).get(value)
        if row is not None:
            out[name] = row
    return out


def numbers() -> dict:
    """The numbers of the course's model on the validation days: overall, and per kind of day and per weather."""
    def make():
        pred = course(VAL_HOURS)
        return {"all": rmse(VAL_HOURS, pred), "hours": len(VAL_HOURS),
                "workingday": rmse_per(VAL_HOURS, pred, "workingday"),
                "weathersit": rmse_per(VAL_HOURS, pred, "weathersit")}
    return kept((LESSON, "numbers"), make)


def weekend_hours():
    days_off = HOURS_2011[HOURS_2011["workingday"] == 0]
    return days_off[days_off["holiday"] == 0]          # 2,495 hours of 105 days


def _changed(hours, column: str, value):
    d = hours.copy()
    d[column] = value
    return d


def limits_of(fn) -> list:
    """The known limits of a model, in English as the card is: hours below 0, Saturday against Sunday, and 2012."""
    every = fn(HOURS_2011)
    below = HOURS_2011[(every < 0).numpy()]
    out = []
    if len(below):
        out.append(f"Expects fewer than 0 bikes in {len(below):,} of the {len(HOURS_2011):,} hours of 2011, "
                   f"all between hour {int(below['hr'].min())} and hour {int(below['hr'].max())}.")
    weekend = weekend_hours()
    apart = (fn(_changed(weekend, "weekday", 6)) - fn(_changed(weekend, "weekday", 0))).abs()
    n = int((apart > BOUND).sum())
    if n:
        out.append(f"Saturday or Sunday changes what it expects by more than {BOUND} bikes in {n:,} of the "
                   f"{len(weekend):,} weekend hours of 2011, since the day of the week is one number "
                   f"(test_weekend, marked xfail).")
    hours_2012 = later_hours()
    out.append(f"Learned from 2011 only: on the {len(hours_2012):,} hours of 2012, with many more riders, it is "
               f"{rmse(hours_2012, fn(hours_2012)):.1f} bikes per hour off.")
    return out


def untested() -> list:
    """A line for each group of the card that no validation hour falls in, such as heavy rain: not tested."""
    out = []
    for column, value, name in CARD_GROUPS:
        if not (VAL_HOURS[column] == value).any():
            n = int((TRAIN_HOURS[column] == value).sum())
            out.append(f"Not tested in {name} ({column} {value}): no validation hour had it, and the model "
                       f"learned it from {n:,} training hour{'' if n == 1 else 's'}.")
    return out


def later_hours():
    return kept((LESSON, "2012"), lambda: demand.read_hours(2012))


def _state_bytes(model) -> bytes:
    folder = paths.WORK / "check" / LESSON / "bytes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.safetensors"
    safetensors_io.save_file(model.state_dict(), path)
    return path.read_bytes()


def course_config() -> dict:
    """The config.json of the course's package, as the course's pack writes it for h16-lr0.01."""
    def make():
        model, scale, _ = _model("course")
        return {"class": "DemandNet", "hidden": 16,
                "inputs": "hr as 24 one-hot columns, then each feature minus mean, divided by std",
                "features": list(demand.FEATURES), "mean": scale["mean"], "std": scale["std"],
                "output": "bikes rented in the hour", "run": "h16-lr0.01",
                "val_rmse": round(numbers()["all"], 2),
                "version": hashlib.sha256(_state_bytes(model)).hexdigest()[:12]}
    return kept((LESSON, "config"), make)


def write_course_package(folder: Path) -> dict:
    """The course's package in `folder`: model.safetensors and config.json, as the course's pack writes them."""
    folder.mkdir(parents=True, exist_ok=True)
    model, _, _ = _model("course")
    safetensors_io.save_file(model.state_dict(), folder / "model.safetensors")
    config = course_config()
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


# ======== pictures ========================================================================================================

W = 330
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg>")


def _hours_word(n: int, lang: str) -> str:
    return f"{number(n, 0, lang)} " + ("uur" if lang == "nl" else ("hour" if n == 1 else "hours"))


def bars(blocks: list, overall: float | None, alt: str, lang: str, top: float = 80.0) -> str:
    """Bars of RMSE in bikes per hour, in the color of mistakes, each with its number and the hours behind it:
    blocks = [(heading, [(label, rmse, hours), ...]), ...]. With `overall`, a dashed line at the RMSE of all hours."""
    lang = _lang(lang)
    x0, x1 = 142, W - 86
    parts, y = [], 4
    for heading, rows in blocks:
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">'
                     f'{html.escape(heading)}</text>')
        y += 18
        for label, value, hours in rows:
            parts.append(f'<text x="{x0 - 6}" y="{y + 10}" text-anchor="end" style="font-size:11.5px;fill:var(--ea-ink-2)">'
                         f'{html.escape(label)}</text>')
            try:
                v = float(value)
                ok = v == v and 0 <= v < 1e9
            except (TypeError, ValueError, OverflowError):
                ok = False
            if not ok:
                shown = str(value)
                shown = shown if len(shown) <= 30 else shown[:30] + "…"
                parts.append(f'<text x="{x0}" y="{y + 10}" style="font-size:12px;fill:var(--ea-loss-text)">'
                             f'{html.escape(shown)}</text>')
            else:
                w = (x1 - x0) * min(v, top) / top
                parts.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="13" style="fill:var(--ea-loss);opacity:0.8"/>')
                tail = number(v, 1, lang) + (f" · {_hours_word(int(hours), lang)}" if isinstance(hours, int) else "")
                parts.append(f'<text x="{x0 + w + 5:.1f}" y="{y + 10.5}" style="font-size:11.5px;fill:var(--ea-ink)">'
                             f'{html.escape(tail)}</text>')
            y += 18
        y += 6
    if overall is not None:
        xo = x0 + (x1 - x0) * min(overall, top) / top
        label = {"en": f"all hours: {number(overall, 1, lang)}", "nl": f"alle uren: {number(overall, 1, lang)}"}[lang]
        parts.append(f'<line x1="{xo:.1f}" y1="4" x2="{xo:.1f}" y2="{y - 4}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1.2;stroke-dasharray:3 3"/>')
        parts.append(f'<text x="{xo:.1f}" y="{y + 8}" text-anchor="middle" style="font-size:11.5px;fill:var(--ea-ink-2)">'
                     f'{html.escape(label)}</text>')
        y += 14
    return _svg(y + 2, alt, parts)


def _blocks(tables: dict, lang: str) -> list:
    lang = _lang(lang)
    out = []
    for column in ("workingday", "weathersit"):
        rows = []
        for (col, value), label in GROUP_WORDS[lang].items():
            if col == column and value in tables.get(column, {}):
                row = tables[column][value]
                rows.append((label, row.get("rmse") if isinstance(row, dict) else row,
                             row.get("hours") if isinstance(row, dict) else None))
        out.append((BLOCKS[lang][column], rows))
    return out


def _alt(blocks: list, lang: str) -> str:
    parts = []
    for heading, rows in blocks:
        shown = []
        for label, value, hours in rows:
            try:
                shown.append(f"{label} {number(float(value), 1, lang)}" + (f" ({_hours_word(hours, lang)})" if isinstance(hours, int) else ""))
            except (TypeError, ValueError, OverflowError):
                shown.append(f"{label} ?")
        parts.append(f"{heading}: " + ", ".join(shown))
    return "; ".join(parts)


def idea_picture(lang: str) -> str:
    """The idea: one RMSE over all validation hours, and the same hours split by kind of day and by weather."""
    lang = _lang(lang)
    n = numbers()
    blocks = _blocks({"workingday": n["workingday"], "weathersit": n["weathersit"]}, lang)
    alt = {"en": f"Validation RMSE of the course's model in bikes per hour: all {number(n['hours'], 0, 'en')} hours {number(n['all'], 1, 'en')}; ",
           "nl": f"Validation-RMSE van het model van de course in fietsen per uur: alle {number(n['hours'], 0, 'nl')} uren {number(n['all'], 1, 'nl')}; "}[lang]
    svg = bars(blocks, n["all"], alt + _alt(blocks, lang), lang)
    caption = {"en": "RMSE on the 73 validation days, in bikes per hour, with the hours behind each number. The dashed line: all hours together.",
               "nl": "RMSE op de 73 validatiedagen, in fietsen per uur, met de uren achter elk getal. De lijn van streepjes: alle uren samen."}[lang]
    return yours.figure({"en": "One number, split", "nl": "Eén getal, opgesplitst"}[lang],
                        svg + f'<div style="font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(caption)}</div>')


# ======== exercise 1: rmse_per ============================================================================================

class _Named:
    """A value the check calls by a short name in its messages, such as `val`; the learner's code gets its own copy.
    `make` gives the value the first time it is needed."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self):
        return self.name

    def value(self):
        return kept((LESSON, "named", self.name), self.make)

    def __deepcopy__(self, memo):
        v = self.value()
        if isinstance(v, torch.Tensor):
            return v.clone()
        return v.copy() if hasattr(v, "copy") else copy.deepcopy(v)


def _small():
    import pandas as pd
    rows = [(10, 1, 1, 1), (40, 1, 1, 2), (250, 0, 2, 2), (100, 0, 3, 3), (60, 1, 3, 1), (0, 0, 1, 4)]
    return pd.DataFrame(rows, columns=["cnt", "workingday", "weathersit", "season"])


VAL = _Named("val", lambda: VAL_HOURS.copy())
PRED = _Named("pred", lambda: course(VAL_HOURS))
SMALL = _Named("few", _small)
SMALL_PRED = _Named("few_pred", lambda: torch.tensor([30.0, 40.0, 220.0, 110.0, 66.0, -2.0]))

_R_HEAD = ("import torch\n"
           "from ops.demand import targets\n"
           "\n\n"
           "def rmse_per(hours, pred, column):\n"
           "    y = targets(hours).reshape(-1)\n"
           "    table = {}\n"
           "    for value in sorted(set(hours[column])):\n"
           "        mask = torch.tensor((hours[column] == value).to_numpy())\n"
           "        error = pred[mask] - y[mask]\n")


def _r(rmse_code: str) -> str:
    return (_R_HEAD + f"        table[int(value)] = {{\"hours\": len(error), \"rmse\": {rmse_code}}}\n"
            "    return table\n")


_R_FILL = "(error ** 2).mean().sqrt().item()"
_R_START = _R_HEAD + "        table[int(value)] = ...\n    return table\n"
_R = _r(_R_FILL)


def _rmse_shown(v):
    """A table of rmse_per in a check message, with numbers rounded to one decimal."""
    if isinstance(v, dict):
        parts = []
        for k, row in list(v.items())[:6]:
            if isinstance(row, dict):
                inner = ", ".join(f"{name!r}: {round(float(x), 1) if isinstance(x, float) else x!r}"
                                  for name, x in row.items())
                parts.append(f"{k!r}: {{{inner}}}")
            else:
                parts.append(f"{k!r}: {row!r}")
        text = "{" + ", ".join(parts) + ("…" if len(v) > 6 else "") + "}"
    else:
        text = repr(v)
    text = text if len(text) <= 160 else text[:160] + "…"
    return {lang: f"`{text}`" for lang in LANGS}


def _other_keys(got, text) -> bool:
    """A dict per value, but with other keys than the card reads."""
    rows = list(got.values()) if isinstance(got, dict) else []
    return bool(rows) and all(isinstance(r, dict) for r in rows) and any(set(r) != {"hours", "rmse"} for r in rows)


_WANT: dict = {}   # what the course's rmse_per gives in the case the check compares now


def _rmse_values(table) -> list:
    rows = list(table.values()) if isinstance(table, dict) else []
    return [r.get("rmse") for r in rows if isinstance(r, dict)]


def _has_tensor(table) -> bool:
    return any(isinstance(v, torch.Tensor) for v in _rmse_values(table))


def _same_table(got, want) -> bool:
    """The same table as the course's, with each RMSE a number of Python: a tensor with the right number is a
    mistake of its own."""
    _WANT["want"] = want
    return same_value(got, want) and not _has_tensor(got)


def _tensor_rmse(got, text) -> bool:
    """The numbers are right, but each RMSE is a tensor with one number in it."""
    return _has_tensor(got) and same_value(got, _WANT.get("want"))


def _rounded(got, text) -> bool:
    """Each RMSE is the right one, rounded: the card rounds when it writes a row."""
    mine, right = _rmse_values(got), _rmse_values(_WANT.get("want"))
    if not mine or len(mine) != len(right) or not all(isinstance(v, (int, float)) for v in mine):
        return False
    return any(all(abs(v - round(r, d)) < 1e-9 for v, r in zip(mine, right)) for d in (0, 1, 2, 3)) \
        and not all(abs(v - r) < 1e-6 * max(1.0, abs(r)) for v, r in zip(mine, right))


class _ReadyBox(FunctionBox):
    """A function box whose cases need the course's model: it trains before the check, outside the time limit."""

    def check(self, text, inputs=None) -> Result:
        _ready()
        return super().check(text, inputs)


def _learner_fn(seen, name: str):
    run = seen.mine
    fn = run.names.get(name) if run is not None else None
    return fn if callable(fn) else None


def groups_picture(seen) -> str:
    """Your rmse_per on the validation days, per kind of day and per weather, as bars."""
    fn = _learner_fn(seen, "rmse_per")
    if fn is None:
        return ""
    lang = _lang(seen.lang)
    tables = {}
    try:
        def call():
            return {c: fn(VAL_HOURS.copy(), course(VAL_HOURS), c) for c in ("workingday", "weathersit")}
        tables = with_time_limit(call, 3)
    except (Exception, Timeout):   # noqa: BLE001   the check says what is wrong; the picture shows nothing
        return ""
    if not all(isinstance(t, dict) for t in tables.values()):
        return ""
    blocks = _blocks(tables, lang)
    if not any(rows for _, rows in blocks):
        return ""
    title = {"en": "Your rmse_per on the 73 validation days", "nl": "Je rmse_per op de 73 validatiedagen"}[lang]
    svg = bars(blocks, numbers()["all"], title + ": " + _alt(blocks, lang), lang)
    caption = {"en": "rmse_per(val, pred, \"workingday\") and rmse_per(val, pred, \"weathersit\"), with the course's model.",
               "nl": "rmse_per(val, pred, \"workingday\") en rmse_per(val, pred, \"weathersit\"), met het model van de course."}[lang]
    return yours.figure(title, svg + f'<figcaption style="font-size:0.82rem;color:var(--ea-ink-2)">{html.escape(caption)}</figcaption>')


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. What comes in its place?",
                nl="Er staat nog een `...` in je code. Wat komt er op deze plek?")


# ======== exercise 2: a known limit, marked ===============================================================================

_W_START = ("import pytest\n"
            "from ops.demand import read_hours\n"
            "\n"
            "HOURS = read_hours(2011)\n"
            "\n\n"
            "def test_weekend(expect):\n"
            "    days_off = HOURS[HOURS[\"workingday\"] == 0]\n"
            "    weekend = days_off[days_off[\"holiday\"] == 0]\n"
            "    saturday = weekend.copy()\n"
            "    saturday[\"weekday\"] = 6\n"
            "    sunday = weekend.copy()\n"
            "    sunday[\"weekday\"] = 0\n"
            "    assert ((expect(saturday) - expect(sunday)).abs() <= 20).all()\n")
_W_REASON = "\"weekday is one number, so Saturday and Sunday lie far apart\""
_W_LINE = f"@pytest.mark.xfail(reason={_W_REASON}, strict=True)\n"
_W = _W_START.replace("\ndef test_weekend", "\n" + _W_LINE + "def test_weekend")


def _with_mark(line: str) -> str:
    return _W_START.replace("\ndef test_weekend", "\n" + line + "\ndef test_weekend")


MODELS = [("course", course), ("no-weekday", no_weekday)]
MODEL_NAMES = {"en": {"course": "your model, the course's h16-lr0.01", "no-weekday": "a version without the column weekday"},
               "nl": {"course": "je model, h16-lr0.01 van de course", "no-weekday": "een versie zonder de kolom weekday"}}


def _marks_of(fn) -> list:
    marks = getattr(fn, "pytestmark", [])
    return list(marks) if isinstance(marks, (list, tuple)) else [marks]


def _call(test, fn) -> BaseException | None:
    """Call the test with one model as its fixture `expect`: None when it passes, else what stopped it."""
    try:
        with_time_limit(lambda: test(fn), 5)
        return None
    except Timeout as e:
        return e
    except BaseException as e:  # noqa: BLE001   an assert, an error, or an outcome of pytest such as pytest.fail
        return e


def pytest_says(marks: list, stopped: BaseException | None) -> str:
    """What pytest reports for a test with these marks, when a call of it ends with `stopped` (None: it passed):
    passed, failed, skipped, xfailed, xpassed (green) or xpassed-strict (counted as failed)."""
    if any(getattr(m, "name", "") in ("skip", "skipif") for m in marks):
        return "skipped"
    xf = next((m for m in marks if getattr(m, "name", "") == "xfail"), None)
    if xf is None:
        return "passed" if stopped is None else "failed"
    condition = xf.args[0] if xf.args else xf.kwargs.get("condition", True)
    if not isinstance(condition, str) and not condition:   # the mark does not apply
        return "passed" if stopped is None else "failed"
    if xf.kwargs.get("run", True) is False:
        return "xfailed"
    if stopped is not None:
        raises = xf.kwargs.get("raises")
        if raises is not None:
            try:
                if not isinstance(stopped, raises):
                    return "failed"
            except TypeError:   # raises= with something that is not an error class
                return "failed"
        return "xfailed"
    return "xpassed-strict" if xf.kwargs.get("strict", False) else "xpassed"


SAID = {"en": {"passed": "PASSED", "failed": "FAILED", "skipped": "SKIPPED", "xfailed": "XFAIL",
               "xpassed": "XPASS", "xpassed-strict": "XPASS(strict): FAILED", "error": "ERROR"},
        "nl": {"passed": "PASSED", "failed": "FAILED", "skipped": "SKIPPED", "xfailed": "XFAIL",
               "xpassed": "XPASS", "xpassed-strict": "XPASS(strict): FAILED", "error": "ERROR"}}
SAID_MEANS = {"en": {"passed": "passed", "failed": "failed", "skipped": "not run", "xfailed": "failed, as expected",
                     "xpassed": "passed, though marked to fail", "xpassed-strict": "passed though marked to fail: counted as a failure",
                     "error": "stopped before the test could run"},
              "nl": {"passed": "geslaagd", "failed": "gefaald", "skipped": "niet uitgevoerd", "xfailed": "gefaald, zoals verwacht",
                     "xpassed": "geslaagd, hoewel de markering een fout verwachtte", "xpassed-strict": "geslaagd, hoewel de markering een fout verwachtte: telt als fout",
                     "error": "gestopt voordat de test kon draaien"}}

_LAST: dict = {}   # the text the check of the weekend ran last, and what came out: for the picture


@dataclass
class MarkBox(CodeBox):
    """The learner marks a test that the model fails as a known limit. The check runs the box, reads the marks on
    `test_weekend` as pytest reads them, and calls the test with the course's model, which must fail it, and with
    the version without weekday, which must pass it. Right: pytest would report XFAIL for the first, and for the
    second XPASS(strict), a failure that says the limit is gone. Texts: missing, empty, cleared, no_test, correct,
    kept; the rest are mistakes."""
    test_name: str = "test_weekend"

    def tries(self, text: str, names: dict):
        """(run, marks, {model: what stopped the test, or None}) for the text, worked out once per text."""
        last = _LAST.get(self.name)
        if last is not None and last[0] == text:
            return last[1]
        _ready()
        run = run_code(text, names)
        if run.error is not None or run.timeout:
            out = (run, [], {})
        else:
            test = run.names.get(self.test_name)
            if not callable(test):
                out = (run, None, {})
            else:
                out = (run, _marks_of(test), {key: _call(test, fn) for key, fn in MODELS})
        if not run.timeout:
            _LAST[self.name] = (text, out)
        return out

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        names = dict(self.names if inputs is None else inputs)
        run, marks, stopped = self.tries(text, names)
        if run.timeout:
            return Result("slow", self.message("slow"))
        if run.error is not None:
            for m in self.mistakes:
                if m.error is not None and m.error(run.error):
                    return Result("error", m.text, mistake=m.key)
            return Result("error", self._plain_error(run, text))
        if marks is None:
            return Result("wrong", self.message("no_test"), mistake="no-test")
        names_of = [getattr(m, "name", "") for m in marks]
        xf = next((m for m in marks if getattr(m, "name", "") == "xfail"), None)
        said = {key: pytest_says(marks, e) for key, e in stopped.items()}
        if "skip" in names_of or "skipif" in names_of:
            return self._said("skip")
        if xf is None:
            if stopped["course"] is None:
                return self._said("passes-course")
            return self._said("no-mark")
        if xf.args:
            return self._said("args")
        reason = xf.kwargs.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return self._said("reason")
        if xf.kwargs.get("run", True) is False:
            return self._said("not-run")
        if xf.kwargs.get("strict") is not True:
            return self._said("strict")
        condition = xf.kwargs.get("condition", True)
        if not isinstance(condition, str) and not condition:
            return self._said("condition")
        if "raises" in xf.kwargs and said["course"] == "failed":
            return self._said("raises")
        course_stop = stopped["course"]
        if course_stop is None:
            return self._said("passes-course")
        if isinstance(course_stop, Timeout):
            return Result("slow", self.message("slow"))
        if not isinstance(course_stop, AssertionError):
            kind = type(course_stop).__name__
            t = self.texts["other-error"]
            return Result("wrong", {k: v.replace("{kind}", kind) for k, v in t.items()}, mistake="other-error")
        if said["no-weekday"] != "xpassed-strict":
            return self._said("fails-fixed")
        return self._correct(text)

    def _said(self, key: str) -> Result:
        return Result("wrong", self.texts[key], mistake=key)


def weekend_picture(seen) -> str:
    """What pytest would report for your test with the two models: the words pytest prints, and what they mean."""
    ex = seen.exercise
    if not isinstance(seen.text, str) or is_empty_text(seen.text, ex.start):
        return ""
    lang = _lang(seen.lang)
    run, marks, stopped = ex.tries(seen.text, dict(seen.names))
    if run.error is not None or run.timeout or marks is None:
        return ""
    want = {"course": "xfailed", "no-weekday": "xpassed-strict"}
    rows = []
    for key, _ in MODELS:
        said = pytest_says(marks, stopped[key])
        color = "var(--ea-ink)" if said == want[key] else "var(--ea-loss-text)"
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0;vertical-align:top">{html.escape(MODEL_NAMES[lang][key])}</td>'
                    f'<td style="padding:0.15rem 0;vertical-align:top"><span style="font-family:var(--ea-font-code);font-size:0.8rem;color:{color};font-weight:600">'
                    f'{html.escape(SAID[lang][said])}</span><br><span style="color:var(--ea-ink-2);font-size:0.8rem">{html.escape(SAID_MEANS[lang][said])}</span></td></tr>')
    head = {"en": ["test_weekend with", "pytest says"], "nl": ["test_weekend met", "pytest zegt"]}[lang]
    table = ('<table style="font-size:0.85rem;border-collapse:collapse"><tr>'
             + "".join(f'<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 0.8rem 0.2rem 0">{html.escape(h)}</th>' for h in head)
             + f'</tr>{"".join(rows)}</table>')
    title = {"en": "What pytest reports for your test", "nl": "Wat pytest meldt voor je test"}[lang]
    return yours.figure(title, table)


# ======== exercise 3: the card ============================================================================================

_C_HEAD = ("def card(config, groups, limits):\n"
           "    lines = [f\"# Bikes per hour, version {config['version']}\", \"\",\n"
           "             f\"Expects the bikes rented in one hour, for all of Capital Bikeshare ({config['run']}).\",\n"
           "             \"Not for one station, and not for a year it was not tested on.\", \"\",\n"
           "             f\"## How far off: {round(config['val_rmse'], 2)} bikes per hour on the validation days\", \"\",\n"
           "             \"| Group | Hours | RMSE |\", \"|---|---|---|\"]\n"
           "    for name, row in groups.items():\n")
_C_TAIL = ("    lines += [\"\", \"## Known limits\", \"\"] + [\"- \" + limit for limit in limits]\n"
           "    return \"\\n\".join(lines) + \"\\n\"\n")


def _c(body: str) -> str:
    return _C_HEAD + body + _C_TAIL


_C_FILL = "{round(row['rmse'], 1)}"
_C_LINE = "        lines.append(f\"| {name} | {row['hours']} | " + _C_FILL + " |\")\n"
_C_START = _c("        ...\n")
_C = _c(_C_LINE)
_C_NO_ROUND = _c("        lines.append(f\"| {name} | {row['hours']} | {row['rmse']} |\")\n")
_C_SWAPPED = _c("        lines.append(f\"| {name} | " + _C_FILL + " | {row['hours']} |\")\n")
_C_NO_PIPES = _c("        lines.append(f\"{name} {row['hours']} " + _C_FILL + "\")\n")
_C_WHOLE = _c("        lines.append(f\"| {name} | {row['hours']} | {round(row['rmse'])} |\")\n")


def _md_lines(text: str) -> list:
    """The lines of a card, with the cells of each table row stripped: `|a|b|` and `| a | b` are the same row."""
    out = []
    for line in text.strip().split("\n"):
        s = line.strip()
        if "|" in s:
            s = "| " + " | ".join(c.strip() for c in s.strip("|").split("|")) + " |"
        out.append(s)
    return out


def _same_card(got, want) -> bool:
    if isinstance(got, str) and isinstance(want, str):
        return _md_lines(got) == _md_lines(want)
    return same_value(got, want)


def _one_row(got, text) -> bool:
    """The card has one row of the table, though there are more groups: the line is after the loop."""
    if not isinstance(got, str):
        return False
    rows = [ln for ln in got.splitlines() if ln.startswith("|") and not ln.startswith("|---")]
    return len(rows) == 2


CONFIG = _Named("config", lambda: dict(course_config()))
GROUPS = _Named("groups", lambda: card_groups({"workingday": numbers()["workingday"], "weathersit": numbers()["weathersit"]}))
LIMITS = _Named("limits", lambda: limits_of(course))
OTHER_CONFIG = _Named("other_config", lambda: {"version": "0f3a9c1e2b7d", "run": "h64-lr0.01", "val_rmse": 53.9})
OTHER_GROUPS = _Named("other_groups", lambda: {"night": {"hours": 421, "rmse": 17.0113}, "day": {"hours": 1312, "rmse": 60.1973},
                                               "rush hour": {"hours": 188, "rmse": 69.75}})
NO_LIMITS = _Named("no_limits", lambda: [])


def render_card(text: str) -> str:
    """A card in Markdown as HTML, for the lesson: headings, paragraphs, a table and a list. Every bit of text is
    escaped, so the card shows what it says and runs nothing."""
    out, table, items, para = [], [], [], []

    def flush():
        if para:
            out.append(f'<p style="margin:0.25rem 0">{html.escape(" ".join(para))}</p>')
            para.clear()
        if table:
            head, *rows = table
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]   # noqa: E731
            th = "".join(f'<th style="text-align:left;padding:0.1rem 0.8rem 0.1rem 0;border-bottom:1px solid var(--ea-line)">{html.escape(c)}</th>' for c in cells(head))
            trs = "".join("<tr>" + "".join(f'<td style="padding:0.1rem 0.8rem 0.1rem 0">{html.escape(c)}</td>' for c in cells(r)) + "</tr>" for r in rows)
            out.append(f'<table style="border-collapse:collapse;font-size:0.82rem;margin:0.3rem 0;font-variant-numeric:tabular-nums"><tr>{th}</tr>{trs}</table>')
            table.clear()
        if items:
            out.append('<ul style="margin:0.25rem 0;padding-left:1.2rem">' + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in text.splitlines():
        s = line.rstrip()
        if s.startswith("|") or (table and "|" in s):   # the pipes on either end of a row may be left out
            if para or items:
                flush()
            if not re.fullmatch(r"\|[\s:|-]*\|", s):
                table.append(s)
            continue
        if table:
            flush()
        if s.startswith("# "):
            flush()
            out.append(f'<div style="font-weight:700;font-size:1rem;margin:0.2rem 0">{html.escape(s[2:])}</div>')
        elif s.startswith("## "):
            flush()
            out.append(f'<div style="font-weight:600;font-size:0.9rem;margin:0.5rem 0 0.15rem">{html.escape(s[3:])}</div>')
        elif s.startswith("- "):
            if para:
                flush()
            items.append(s[2:])
        elif not s.strip():
            flush()
        else:
            if items:
                flush()
            para.append(s.strip())
    flush()
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink);border:1px solid var(--ea-line);'
            'border-radius:6px;padding:0.5rem 0.75rem;max-width:var(--ea-measure)">' + "".join(out) + "</div>")


def card_picture(seen) -> str:
    """Your card for the course's package, drawn as Hugging Face would show the Markdown."""
    fn = _learner_fn(seen, "card")
    if fn is None:
        return ""
    lang = _lang(seen.lang)
    try:
        text = with_time_limit(lambda: fn(CONFIG.__deepcopy__({}), GROUPS.__deepcopy__({}), LIMITS.__deepcopy__({})), 3)
    except (Exception, Timeout):   # noqa: BLE001   the check says what is wrong
        return ""
    if not isinstance(text, str) or not text.strip():
        return ""
    title = {"en": "Your card for the course's package", "nl": "Je card voor het pakket van de course"}[lang]
    return yours.figure(title, render_card(text))


# ======== the exercises ===================================================================================================

EXERCISES = {
    "per-group": _ReadyBox(
        name="per-group",
        function="rmse_per",
        start=_R_START,
        solution=_R,
        save_as=("ops", "report"),
        visible=[(VAL, PRED, "workingday")],
        hidden=[(VAL, PRED, "weathersit"), (SMALL, SMALL_PRED, "season"), (SMALL, SMALL_PRED, "workingday")],
        first=[_DOTS],
        picture=groups_picture,
        show=_rmse_shown,
        compare=_same_table,
        mistakes=[
            Mistake("tensor", matches=_tensor_rmse, state="almost",
                    en="The numbers are right, but each RMSE is a tensor with one number in it, not a Python number. The card would then show `tensor(48.18)`. Which method of a tensor gives the Python number?",
                    nl="De getallen kloppen, maar elke RMSE is een tensor met één getal erin, geen getal van Python. De card zou dan `tensor(48.18)` laten zien. Welke methode van een tensor geeft het getal van Python?"),
            Mistake("rounded", matches=_rounded,
                    en="`{call}` gives each RMSE rounded. `rmse_per` gives the full number, for every use of the table, and the card rounds only when it writes a row. What is left of your line without `round`?",
                    nl="`{call}` geeft elke RMSE afgerond. `rmse_per` geeft het volledige getal, voor elk gebruik van de tabel, en de card rondt pas af bij het schrijven van een rij. Wat blijft er van je regel over zonder `round`?"),
            Mistake("mse", wrong=_r("(error ** 2).mean().item()"),
                    en="`{call}` gives {got}: the mean of the squared errors, in bikes squared, such as 2,321 for the working days. The RMSE is in bikes per hour. Which step of the root mean squared error is missing?",
                    nl="`{call}` geeft {got}: het gemiddelde van de gekwadrateerde fouten, in fietsen in het kwadraat, zoals 2.321 voor de werkdagen. De RMSE is in fietsen per uur. Welke stap van de root mean squared error ontbreekt?"),
            Mistake("mae", wrong=_r("error.abs().mean().item()"),
                    en="`{call}` gives {got}: the mean absolute error, the MAE, 30.8 bikes for the working days. The card gives the RMSE, as config.json does, so the numbers per group fit the one number over all hours. Which three steps make the RMSE of `error`?",
                    nl="`{call}` geeft {got}: de mean absolute error, de MAE, 30,8 fietsen voor de werkdagen. De card geeft de RMSE, zoals config.json, zodat de getallen per groep passen bij het ene getal over alle uren. Welke drie stappen maken de RMSE van `error`?"),
            Mistake("bias", wrong=_r("error.mean().item()"),
                    en="`{call}` gives {got}: the mean error, where errors above and below cancel out, so it is close to 0 even for a model that is far off. Which step makes every error count, however it points?",
                    nl="`{call}` geeft {got}: de gemiddelde fout, waarin fouten naar boven en naar beneden elkaar opheffen, dus ligt het getal dicht bij 0, ook voor een model dat er ver naast zit. Welke stap laat elke fout meetellen, welke kant de fout ook op wijst?"),
            Mistake("sum", wrong=_r("(error ** 2).sum().sqrt().item()"),
                    en="`{call}` gives {got}: the square root of all the squared errors added up, which grows with the number of hours. A group of 1,119 hours then looks worse than one of 182 for that reason alone. Which method of a tensor gives the mean of the squared errors instead?",
                    nl="`{call}` geeft {got}: de wortel uit alle gekwadrateerde fouten opgeteld, en dat getal groeit met het aantal uren. Een groep van 1.119 uur lijkt dan alleen daardoor slechter dan een groep van 182. Welke methode van een tensor geeft in plaats daarvan het gemiddelde van de gekwadrateerde fouten?"),
            Mistake("keys", matches=_other_keys,
                    en="`{call}` gives {got}. The card reads each group as `row['hours']` and `row['rmse']`, so each entry needs exactly these two keys. Which keys does your dict have?",
                    nl="`{call}` geeft {got}. De card leest elke groep als `row['hours']` en `row['rmse']`, dus heeft elke plek precies deze twee keys nodig. Welke keys heeft je dict?"),
            Mistake("all", wrong=_r("((pred - y) ** 2).mean().sqrt().item()"),
                    en="`{call}` gives {got}: the same RMSE for every group, the RMSE of all hours, since the line uses `pred` and `y` and not `error`, the errors of the group. Which name holds the errors of the hours under `mask`?",
                    nl="`{call}` geeft {got}: dezelfde RMSE voor elke groep, de RMSE van alle uren, want de regel gebruikt `pred` en `y` en niet `error`, de fouten van de groep. Welke naam bevat de fouten van de uren onder `mask`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the dict of the group: its hours, and the RMSE of `error` as a Python number.",
                      "nl": "Vervang de `...` door de dict van de groep: de uren, en de RMSE van `error` als getal van Python."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(hours, pred, column)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(hours, pred, column)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the RMSE the square root of the mean of the squared errors of the group?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is de RMSE de wortel uit het gemiddelde van de gekwadrateerde fouten van de groep?"},
            "almost": {"en": "Right for `val`. But `{call}` gives {got}, and it should give {want}. Does your line use only `error`, the errors of the hours under `mask`?",
                       "nl": "Goed voor `val`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je regel alleen `error`, de fouten van de uren onder `mask`?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return table`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met `return table`?"},
            "correct": {"en": "Right in all {n} cases. On the working days your model is 48.2 bikes per hour off, on weekends and holidays 60.9, and in light rain 60.1. Your `rmse_per` is now in your project, in `{saved}`.",
                        "nl": "Goed in alle {n} gevallen. Op de werkdagen zit je model er 48,2 fietsen per uur naast, op weekenden en feestdagen 60,9, en bij lichte regen 60,1. Je `rmse_per` staat nu in je project, in `{saved}`."},
            "kept": _kept("rmse_per"),
        },
    ),
    "weekend": MarkBox(
        name="weekend",
        start=_W_START,
        solution=_W,
        save_as=("ops", "test_weekend"),
        picture=weekend_picture,
        texts=_SAME_TEXTS | {
            "empty": {"en": "Add the line with the mark above `def test_weekend(expect):`.",
                      "nl": "Zet de regel met de markering boven `def test_weekend(expect):`."},
            "no_test": {"en": "Your box has no function `test_weekend`, so the check finds no test to mark. Can you keep `def test_weekend(expect):` as it is?",
                        "nl": "Je vak heeft geen functie `test_weekend`, dus vindt de check geen test om te markeren. Kun je `def test_weekend(expect):` laten zoals het was?"},
            "slow": {"en": "Your test is still not done after a few seconds. Does something in it keep going forever?",
                     "nl": "Je test is na een paar seconden nog niet klaar. Gaat er iets in eindeloos door?"},
            "no-mark": {"en": "Your test has no mark, so pytest reports it as FAILED for the course's model, in every run, and the one red line hides a new failure among the others. Which mark, on the line right above `def`, tells pytest that this test is expected to fail?",
                        "nl": "Je test heeft geen markering, dus meldt pytest de test als FAILED voor het model van de course, bij elke run, en naast deze ene rode regel valt een nieuwe fout niet op. Welke markering, op de regel direct boven `def`, vertelt pytest dat deze test hoort te falen?"},
            "skip": {"en": "A skip mark makes pytest not run the test at all. The day a version of the model no longer has the limit, nothing tells you, and the card keeps a limit that is gone. Which mark still runs the test, and expects it to fail?",
                     "nl": "Met een skip-markering voert pytest de test helemaal niet uit. Op de dag dat een versie van het model de grens niet meer heeft, merk je dat niet, en houdt de card een grens die weg is. Welke markering voert de test wel uit, en verwacht dat de test faalt?"},
            "args": {"en": "Your mark gets a value without a name. pytest reads a first value without a name as a condition, not as the reason. Can you write the reason as `reason=\"...\"`?",
                     "nl": "Je markering krijgt een waarde zonder naam. pytest leest een eerste waarde zonder naam als voorwaarde, niet als reden. Kun je de reden schrijven als `reason=\"...\"`?"},
            "reason": {"en": "Your mark has no reason. pytest prints the reason beside XFAIL, and it is what links the test to the line in the card. Can you add `reason=` with a short text that says why the model fails this test?",
                       "nl": "Je markering heeft geen reden. pytest zet de reden naast XFAIL, en de reden verbindt de test met de regel in de card. Kun je `reason=` toevoegen, met een korte tekst die zegt waarom het model deze test niet haalt?"},
            "not-run": {"en": "With `run=False`, pytest does not run the test, and reports XFAIL whatever the model does. Then you never learn that the limit is gone. Can you leave `run` out?",
                        "nl": "Met `run=False` voert pytest de test niet uit, en meldt XFAIL wat het model ook doet. Dan merk je nooit dat de grens weg is. Kun je `run` weglaten?"},
            "strict": {"en": "Without `strict=True`, a version without the column weekday passes the test, and pytest reports XPASS but stays green. The card then keeps a limit the model no longer has. What do you add to the mark, so that pytest counts a pass as a failure?",
                       "nl": "Zonder `strict=True` haalt een versie zonder de kolom weekday de test, en meldt pytest XPASS maar blijft groen. De card houdt dan een grens die het model niet meer heeft. Wat voeg je toe aan de markering, zodat pytest een geslaagde test als fout telt?"},
            "condition": {"en": "With `condition=False`, the mark does not apply, and pytest reports FAILED for the course's model, as without a mark. Can you leave `condition` out?",
                          "nl": "Met `condition=False` geldt de markering niet, en meldt pytest FAILED voor het model van de course, net als zonder markering. Kun je `condition` weglaten?"},
            "raises": {"en": "Your mark has `raises=`, and the course's model fails the test with an `AssertionError`, which is not the error you named. Then pytest reports FAILED. Can you leave `raises` out, or give `raises=AssertionError`?",
                       "nl": "Je markering heeft `raises=`, en het model van de course laat de test falen met een `AssertionError`, en dat is niet de fout die je noemde. Dan meldt pytest FAILED. Kun je `raises` weglaten, of `raises=AssertionError` geven?"},
            "passes-course": {"en": "Your test passes for the course's model, so it no longer tests the limit: with the mark, pytest would even report XPASS(strict). Is the `assert` still as it was given?",
                              "nl": "Je test slaagt voor het model van de course, dus test de test de grens niet meer: met de markering zou pytest zelfs XPASS(strict) melden. Staat de regel met `assert` er nog zoals gegeven?"},
            "other-error": {"en": "For the course's model, your test stops with a `{kind}`, not with a failing assert. An xfail mark counts any error as the expected failure, so it would hide this one. Is the code of the test still as it was given?",
                            "nl": "Voor het model van de course stopt je test met een `{kind}`, niet met een assert die faalt. Een xfail-markering telt elke fout als de verwachte fout, dus zou deze fout verborgen blijven. Staat de code van de test er nog zoals gegeven?"},
            "fails-fixed": {"en": "Your test also fails for the version without the column weekday, which treats Saturday and Sunday the same. Then it tests more than the limit in the card. Is the code of the test still as it was given?",
                            "nl": "Je test faalt ook voor de versie zonder de kolom weekday. Deze versie behandelt zaterdag en zondag hetzelfde. Dan test de test meer dan de grens in de card. Staat de code van de test er nog zoals gegeven?"},
            "correct": {"en": "Right. For your model pytest now reports XFAIL with your reason, and the suite stays green; for a version without weekday it reports XPASS(strict), a failure that says: the limit is gone, so change the card and remove the mark. Your test is now in your project, in `{saved}`.",
                        "nl": "Goed. Voor je model meldt pytest nu XFAIL met je reden, en de suite blijft groen; voor een versie zonder weekday meldt pytest XPASS(strict), een fout die zegt: de grens is weg, dus pas de card aan en haal de markering weg. Je test staat nu in je project, in `{saved}`."},
            "kept": _kept("test_weekend"),
        },
    ),
    "card": _ReadyBox(
        name="card",
        function="card",
        start=_C_START,
        solution=_C,
        save_as=("ops", "card"),
        visible=[(CONFIG, GROUPS, LIMITS)],
        hidden=[(OTHER_CONFIG, OTHER_GROUPS, NO_LIMITS)],
        picture=card_picture,
        compare=_same_card,
        mistakes=[
            Mistake("no-round", wrong=_C_NO_ROUND,
                    en="The rows of your table show every RMSE with all its decimals, such as `48.180328369140625`. Who reads the card needs one decimal at most. Which function of Python rounds a number to one decimal?",
                    nl="De rijen van je tabel tonen elke RMSE met alle decimalen, zoals `48.180328369140625`. Wie de card leest, heeft genoeg aan één decimaal. Welke functie van Python rondt een getal af op één decimaal?"),
            Mistake("whole", wrong=_C_WHOLE,
                    en="The rows of your table round every RMSE to a whole number, such as 48. The card gives one decimal, as its head gives the RMSE of all hours with two. What does `round` get as its second value?",
                    nl="De rijen van je tabel ronden elke RMSE af op een heel getal, zoals 48. De card geeft één decimaal, zoals de kop van de card de RMSE van alle uren met twee geeft. Wat krijgt `round` als tweede waarde?"),
            Mistake("swapped", wrong=_C_SWAPPED,
                    en="In your rows the RMSE comes before the hours, while the header says Group, Hours, RMSE. Which value goes in the second cell?",
                    nl="In je rijen staat de RMSE voor de uren, terwijl de kop Group, Hours, RMSE zegt. Welke waarde hoort in de tweede cel?"),
            Mistake("no-pipes", wrong=_C_NO_PIPES,
                    en="Your rows have no `|`, so Markdown does not read them as rows of the table under the header, but as one line of text. How does the header line mark its cells?",
                    nl="Je rijen hebben geen `|`, dus leest Markdown de rijen niet als rijen van de tabel onder de kop, maar als één regel tekst. Hoe geeft de kopregel de cellen aan?"),
            Mistake("one-row", matches=_one_row,
                    en="Your card has one row in its table, though there is a row for each group. Is the line with `lines.append` indented inside the `for` loop?",
                    nl="Je card heeft één rij in de tabel, terwijl er voor elke groep een rij hoort. Springt de regel met `lines.append` in, binnen de `for`-loop?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a line that adds one row of the table to `lines`: the name of the group, its hours, and its RMSE.",
                      "nl": "Vervang de `...` door een regel die één rij van de tabel aan `lines` toevoegt: de naam van de groep, de uren, en de RMSE."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(config, groups, limits)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(config, groups, limits)` houden?"},
            "wrong": {"en": "Your card is not the course's card yet. Is each row of the table `| name | hours | RMSE |`, with the RMSE rounded to one decimal?",
                      "nl": "Je card is nog niet de card van de course. Is elke rij van de tabel `| naam | uren | RMSE |`, met de RMSE afgerond op één decimaal?"},
            "almost": {"en": "Right for the course's package. But with other groups, `{call}` gives another card than the course's. Does your line use `name` and `row` of each group, and nothing fixed?",
                       "nl": "Goed voor het pakket van de course. Maar met andere groepen geeft `{call}` een andere card dan de card van de course. Gebruikt je regel `name` en `row` van elke groep, en niets vasts?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` line?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met de `return`-regel?"},
            "correct": {"en": "Right in all {n} cases. Your card says what the model is for, how far off it is overall and per group, with the hours behind each number, and its known limits. Your `card` is now in your project, in `{saved}`.",
                        "nl": "Goed in alle {n} gevallen. Je card zegt waarvoor het model is, hoe ver het model ernaast zit, in totaal en per groep, met de uren achter elk getal, en de bekende grenzen. Je `card` staat nu in je project, in `{saved}`."},
            "kept": _kept("card"),
        },
    ),
}
REQUIRED = ["per-group", "weekend", "card", "heavy-rain"]


# ======== the demo: every test of your project at once, and the card ======================================================

_REPORTER = '''# the course's reporter: what pytest says about each test, as JSON, for the lesson's picture
import json
import os

_OUT = []


def pytest_runtest_logreport(report):
    text = str(getattr(report, "longreprtext", "") or "")
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        kind = report.outcome
        if hasattr(report, "wasxfail"):
            kind = "xfailed" if report.skipped else "xpassed"
        elif report.failed and text.startswith("[XPASS(strict)]"):
            kind = "xpassed-strict"
        elif report.when == "setup" and report.failed:
            kind = "error"
        lines = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("E ")]
        _OUT.append({"id": report.nodeid, "outcome": kind, "seconds": round(report.duration, 3),
                     "message": (lines[0] if lines else text.splitlines()[0] if text else "")[:160]})


def pytest_collectreport(report):
    if report.failed:   # a test file pytest cannot read, such as one with a syntax error
        text = str(getattr(report, "longreprtext", "") or "")
        lines = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("E ")]
        _OUT.append({"id": report.nodeid, "outcome": "error", "seconds": 0.0,
                     "message": (lines[-1] if lines else text.splitlines()[-1] if text else "")[:160]})


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["EAI_SUITE_OUT"], "w", encoding="utf-8") as f:
        json.dump(_OUT, f)
'''


def _checks_of(lesson: str):
    """Another lesson's checks, loaded once, without changing which modules this lesson builds on."""
    def load():
        said = projects.said()
        try:
            spec = importlib.util.spec_from_file_location(f"_course_checks_{lesson.replace('-', '_')}_tested",
                                                          paths.LESSONS / lesson / "checks.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            projects.builds_on(*said)
        return module
    return kept((LESSON, "checks", lesson), load)


# the files of the project, in the order the lessons make them: (file, lesson, exercise); None for a file it gives
PROJECT_FILES = [("demand.py", "sys-project-tracked-model", None), ("safetensors_io.py", "sys-project-tracked-model", None),
                 ("split.py", "sys-project-tracked-model", "split"), ("train.py", "sys-project-tracked-model", "train"),
                 ("pack.py", "sys-project-tracked-model", "pack"), ("test_inputs.py", "sys-code-tests", "one-row"),
                 ("assertions.py", "sys-code-tests", "impossible"), ("test_training.py", "sys-code-tests", "overfit"),
                 ("conftest.py", "sys-behavior-tests", None), ("serving.py", "sys-behavior-tests", None),
                 ("test_night.py", "sys-behavior-tests", "quiet-night"), ("test_row_number.py", "sys-behavior-tests", "row-number"),
                 ("test_rain.py", "sys-behavior-tests", "rain"), ("report.py", LESSON, "per-group"),
                 ("test_weekend.py", LESSON, "weekend"), ("card.py", LESSON, "card")]


def course_file(name: str, lesson: str, exercise: str | None) -> str:
    """The course's version of a file of the project."""
    if exercise is None:
        return (paths.LESSONS / lesson / "project" / "ops" / name).read_text(encoding="utf-8")
    exercises = EXERCISES if lesson == LESSON else _checks_of(lesson).EXERCISES
    return projects.module_code(exercises[exercise].solution) + "\n"


def _run_dir() -> Path:
    return paths.WORK / "check" / LESSON / "run"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _own(name: str, function: str):
    """Your function from your project, or None when the file is missing or does not load."""
    path = projects.folder() / "ops" / name
    if not path.exists():
        return None
    try:
        projects.on_path()
        fn = getattr(_load_module(path, f"_yours_{function}_{time.time_ns()}"), function, None)
    except Exception:   # noqa: BLE001   a file of yours that does not load: the course's version takes its place
        return None
    return fn if callable(fn) else None


def _package_expect(folder: Path):
    """What the package in `folder` expects for a table of hours, read the way the course reads a package; its
    config; and "yours", "missing" or "broken" (there, but the course cannot read it or it gives no answers)."""
    if not (folder / "config.json").exists() and not (folder / "model.safetensors").exists():
        return None, None, "missing"
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        model = demand.DemandNet(int(config["hidden"]))
        model.load_state_dict(safetensors_io.load_file(folder / "model.safetensors"))
        model.eval()
        scale = {"mean": config["mean"], "std": config["std"]}
        str(config["version"]), str(config["run"]), float(config["val_rmse"])   # what the card reads

        def fn(hours):
            with torch.no_grad():
                return model(demand.inputs(hours, scale)).reshape(-1)
        if tuple(fn(VAL_HOURS.head(3)).shape) != (3,):
            raise ValueError("one answer per hour")
    except Exception:   # noqa: BLE001   a package the course cannot read: the course's takes its place
        return None, None, "broken"
    return fn, config, "yours"


def run_suite(ops: Path) -> dict:
    """Run pytest on the folder `ops`, in its own Python process, as you would in a terminal. Gives what it said
    about each test, the seconds it took, and its last lines."""
    root = ops.parent
    (root / "_course_report.py").write_text(_REPORTER, encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    out = root / "suite.json"
    out.unlink(missing_ok=True)
    src = str(Path(everything_ai.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), src]), "EAI_SUITE_OUT": str(out)}
    env.pop("PYTEST_ADDOPTS", None)
    start = time.time()
    try:
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "_course_report",
                               "-c", str(root / "pytest.ini"), "--rootdir", str(root), "ops"],
                              cwd=root, env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"tests": [], "seconds": time.time() - start, "tail": "timeout"}
    tests = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    tail = [ln for ln in done.stdout.splitlines() if ln.strip()][-1:] if done.stdout else []
    return {"tests": tests, "seconds": time.time() - start, "tail": tail[0] if tail else done.stderr[-300:]}


def demo() -> dict:
    """Every test of your project at once, with pytest, on a copy of your project where the course fills in what is
    missing; then the card of your package, made by your card and your rmse_per, and written beside the package."""
    _ready()
    if importlib.util.find_spec("pytest") is None:
        return {"error": "no-pytest"}
    run_dir = _run_dir()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ops = run_dir / "ops"
    mine = projects.folder() / "ops"
    if mine.is_dir():
        shutil.copytree(mine, ops, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs.jsonl", ".pytest_cache"))
    else:
        ops.mkdir(parents=True)
    (ops / "__init__.py").touch()
    filled = []
    for name, lesson, exercise in PROJECT_FILES:
        if not (ops / name).exists():
            (ops / name).write_text(course_file(name, lesson, exercise), encoding="utf-8")
            filled.append(name)
    fn, config, package = _package_expect(mine / "model")
    own_package = fn is not None
    if not own_package:
        if (ops / "model").exists():
            shutil.rmtree(ops / "model")
        config = write_course_package(ops / "model")
        fn = course
    suite = run_suite(ops)
    # the card of the package: your rmse_per and your card where they work, else the course's
    used = {}
    broken = []   # your files that are there, but do not work here
    tables = None
    yours_rmse = _own("report.py", "rmse_per")
    if yours_rmse is not None:
        try:
            pred = fn(VAL_HOURS)
            tables = with_time_limit(lambda: {c: yours_rmse(VAL_HOURS.copy(), pred.clone(), c) for c in ("workingday", "weathersit")}, 5)
            groups = {name: {"hours": int(row["hours"]), "rmse": float(row["rmse"])}
                      for name, row in card_groups(tables).items()}
            if not groups:
                raise ValueError("no row the card can read")
            used["rmse_per"] = "yours"
        except (Exception, Timeout):   # noqa: BLE001   yours does not work here: the course's takes its place
            tables = None
    if tables is None and (mine / "report.py").exists():
        broken.append("rmse_per")
    if tables is None:
        pred = fn(VAL_HOURS)
        tables = {c: rmse_per(VAL_HOURS, pred, c) for c in ("workingday", "weathersit")}
        groups = card_groups(tables)
        used["rmse_per"] = "course"
    limits = limits_of(fn) + untested()
    text = None
    yours_card = _own("card.py", "card")
    if yours_card is not None:
        try:
            text = with_time_limit(lambda: yours_card(dict(config), copy.deepcopy(groups), list(limits)), 5)
            used["card"] = "yours" if isinstance(text, str) and text.strip() else None
        except (Exception, Timeout):   # noqa: BLE001
            text = None
    if not isinstance(text, str) or not text.strip():
        text = _run_card(config, groups, limits)
        used["card"] = "course"
        if (mine / "card.py").exists():
            broken.append("card")
    written = None
    if own_package:
        target = mine / "model" / "README.md"
        target.write_text(text, encoding="utf-8")
        written = projects.shown(target)
    return {"suite": suite, "filled": filled, "own_package": own_package, "package": package, "card": text,
            "written": written, "used": used, "broken": broken, "tables": tables}


def _run_card(config, groups, limits) -> str:
    names: dict = {}
    exec(compile(_C, "<course card>", "exec"), names)
    return names["card"](config, groups, limits)


OUTCOME_WORDS = {
    "en": {"passed": "passed", "failed": "failed", "skipped": "skipped", "xfailed": "xfailed", "xpassed": "xpassed",
           "xpassed-strict": "failed (XPASS strict)", "error": "error"},
    "nl": {"passed": "passed", "failed": "failed", "skipped": "skipped", "xfailed": "xfailed", "xpassed": "xpassed",
           "xpassed-strict": "failed (XPASS strict)", "error": "error"},
}


def suite_picture(r: dict, lang: str) -> str:
    """Every test of the project as pytest ran it: per file, the test, what pytest said, and the seconds."""
    lang = _lang(lang)
    tests = r["suite"]["tests"]
    rows = []
    for t in tests:
        file_, _, name = t["id"].partition("::")
        name = name or {"en": "(the file itself)", "nl": "(het bestand zelf)"}[lang]
        good = t["outcome"] in ("passed", "xfailed")
        color = "var(--ea-ink)" if good else "var(--ea-loss-text)"
        note = f'<br><span style="color:var(--ea-loss-text);font-size:0.78rem">{html.escape(t["message"])}</span>' if not good and t.get("message") else ""
        rows.append(f'<tr><td style="padding:0.1rem 0.7rem 0.1rem 0;font-family:var(--ea-font-code);font-size:0.76rem">{html.escape(file_.replace("ops/", ""))}</td>'
                    f'<td style="padding:0.1rem 0.7rem 0.1rem 0;font-family:var(--ea-font-code);font-size:0.76rem">{html.escape(name)}{note}</td>'
                    f'<td style="padding:0.1rem 0.7rem 0.1rem 0;color:{color};font-weight:600">{html.escape(OUTCOME_WORDS[lang].get(t["outcome"], t["outcome"]))}</td>'
                    f'<td style="padding:0.1rem 0;text-align:right;color:var(--ea-ink-2)">{number(t["seconds"], 2, lang)}</td></tr>')
    head = {"en": ["file", "test", "pytest", "seconds"], "nl": ["bestand", "test", "pytest", "seconden"]}[lang]
    table = ('<table style="font-size:0.82rem;border-collapse:collapse;font-variant-numeric:tabular-nums"><tr>'
             + "".join(f'<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 0.7rem 0.2rem 0">{html.escape(h)}</th>' for h in head)
             + f'</tr>{"".join(rows)}</table>')
    tail = r["suite"].get("tail") or ""
    line = f'<div style="font-family:var(--ea-font-code);font-size:0.78rem;margin-top:0.3rem;color:var(--ea-ink-2)">{html.escape(tail)}</div>'
    title = {"en": "Every test of your project, as pytest ran them", "nl": "Elke test van je project, zoals pytest de tests uitvoerde"}[lang]
    return yours.figure(title, table + line)


def card_view(r: dict, lang: str) -> str:
    lang = _lang(lang)
    title = {"en": "The card of your package", "nl": "De card van je pakket"}[lang] if r["own_package"] else \
        {"en": "The card of the course's package", "nl": "De card van het pakket van de course"}[lang]
    return yours.figure(title, render_card(r["card"]))


def counts(tests: list) -> dict:
    out: dict = {}
    for t in tests:
        out[t["outcome"]] = out.get(t["outcome"], 0) + 1
    return out


def summary(r: dict, lang: str) -> str:
    """What the run did, in words: which files the course filled in, what pytest counted, and where the card went."""
    lang = _lang(lang)
    if r.get("error") == "no-pytest":
        return {"en": "pytest is not installed in this Python, so the course cannot run your tests here. Start the course with `uv run everything-ai`, which installs it.",
                "nl": "pytest is niet geïnstalleerd in deze Python, dus kan de course je tests hier niet uitvoeren. Start de course met `uv run everything-ai`, dat pytest installeert."}[lang]
    unread = [t["id"] for t in r["suite"]["tests"] if "::" not in t["id"]]   # test files pytest could not read
    tests = [t for t in r["suite"]["tests"] if "::" in t["id"]]
    c = counts(tests)
    n_tests = len(tests)
    bad = c.get("failed", 0) + c.get("error", 0) + c.get("xpassed-strict", 0)
    parts = []
    if r["filled"]:
        listed = ", ".join(f"`{f}`" for f in r["filled"])
        parts.append({"en": f"The course put its own version of {listed} in the copy, since your project does not have {'it' if len(r['filled']) == 1 else 'them'} yet.",
                      "nl": f"De course zette een eigen versie van {listed} in de kopie, omdat je project {'dit bestand' if len(r['filled']) == 1 else 'deze bestanden'} nog niet heeft."}[lang])
    if r.get("package") == "broken":
        parts.append({"en": "The course cannot read the package in `ops/model` of your project, or it gives no answers, so the tests ran on the course's package: pack your model again in *Project: a demand model, tracked and packed*.",
                      "nl": "De course kan het pakket in `ops/model` van je project niet lezen, of het pakket geeft geen antwoorden, dus draaiden de tests op het pakket van de course: verpak je model opnieuw in *Project: een model van de vraag, bijgehouden en ingepakt*."}[lang])
    elif not r["own_package"]:
        parts.append({"en": "Your project has no package in `ops/model` yet, so the tests ran on the course's package: pack your model in *Project: a demand model, tracked and packed*.",
                      "nl": "Je project heeft nog geen pakket in `ops/model`, dus draaiden de tests op het pakket van de course: verpak je model in *Project: een model van de vraag, bijgehouden en ingepakt*."}[lang])
    secs = number(r["suite"]["seconds"], 1, lang)
    if unread:
        listed = ", ".join(f"`{u.replace('ops/', '')}`" for u in unread)
        parts.append({"en": f"pytest could not read {listed}, so it stopped before any test ran: the red line says why.",
                      "nl": f"pytest kon {listed} niet lezen, dus stopte pytest voordat er een test liep: de rode regel zegt waarom."}[lang])
    elif bad == 0 and n_tests:
        parts.append({"en": f"pytest ran {n_tests} tests in {secs} seconds, and the suite is green: every test passed, or failed as its mark expects.",
                      "nl": f"pytest voerde {n_tests} tests uit in {secs} seconden, en de suite is groen: elke test slaagde, of faalde zoals de markering verwacht."}[lang])
    elif n_tests:
        parts.append({"en": f"pytest ran {n_tests} tests in {secs} seconds, and {bad} of them did not do what they should: the red lines say which, and why.",
                      "nl": f"pytest voerde {n_tests} tests uit in {secs} seconden, en bij {bad} daarvan ging iets mis: de rode regels zeggen welke, en waarom."}[lang])
    else:
        tail = (r["suite"].get("tail") or "").strip().replace("`", "'")
        said = f" pytest said: `{tail}`" if tail else ""
        parts.append({"en": "pytest found no tests to run." + said,
                      "nl": "pytest vond geen tests om uit te voeren." + said.replace("pytest said", "pytest meldde")}[lang])
    if r["written"]:
        parts.append({"en": f"The card is now beside your package, in `{r['written']}`.",
                      "nl": f"De card staat nu naast je pakket, in `{r['written']}`."}[lang])
    used, broken = r["used"], r.get("broken", [])
    missing = [f"`{k}`" for k in ("rmse_per", "card") if used.get(k) == "course" and k not in broken]
    if missing:
        parts.append({"en": f"For {' and '.join(missing)} the card used the course's version, since yours is not in your project yet.",
                      "nl": f"Voor {' en '.join(missing)} gebruikte de card de versie van de course, omdat de jouwe nog niet in je project staat."}[lang])
    if broken:
        which = " and ".join(f"`{k}`" for k in broken)
        parts.append({"en": f"For {which} the card used the course's version, since yours stops with an error here or gives something the card cannot read: the check of the exercise says what is wrong.",
                      "nl": f"Voor {which.replace(' and ', ' en ')} gebruikte de card de versie van de course, omdat de jouwe hier stopt met een fout of iets geeft wat de card niet kan lezen: de check van de oefening zegt wat er mis is."}[lang])
    return " ".join(parts)


# ======== cases for the tests ==============================================================================================

CASES = [
    ("per-group", _R_START, "empty", None),
    ("per-group", _R, "correct", None),
    ("per-group", _r("float(((error ** 2).mean()) ** 0.5)"), "correct", None),
    ("per-group", _r("torch.sqrt((error ** 2).mean()).item()"), "correct", None),
    ("per-group", _r("(error ** 2).mean().sqrt()"), "almost", "tensor"),
    ("per-group", _r("round((error ** 2).mean().sqrt().item(), 1)"), "wrong", "rounded"),
    ("per-group", _R_HEAD + "        rmse = (error ** 2).mean().sqrt().item()\n"
                  "        table[int(value)] = {\"hours\": len(error), \"rmse\": rmse}\n    return table\n", "correct", None),
    ("per-group", _r("(error ** 2).mean().item()"), "wrong", "mse"),
    ("per-group", _r("error.abs().mean().item()"), "wrong", "mae"),
    ("per-group", _r("error.mean().item()"), "wrong", "bias"),
    ("per-group", _r("(error ** 2).sum().sqrt().item()"), "wrong", "sum"),
    ("per-group", _r("((pred - y) ** 2).mean().sqrt().item()"), "wrong", "all"),
    ("per-group", _r("53.04"), "wrong", None),
    ("per-group", _R.replace('{"hours": len(error), "rmse"', '{"n": len(error), "rmse"'), "wrong", "keys"),
    ("per-group", _R.replace("    y = targets", "    n = 2\n    y = targets").replace("(error ** 2).mean().sqrt().item()", "..."), "wrong", "dots-left"),
    ("per-group", _r("(error ** 2).mean().sqrt().item()").replace("def rmse_per(hours, pred, column):", "def rmse_per(hours, pred):"), "error", "arguments"),
    ("per-group", _r("(error ** 2).mean().sqrt().item()").replace("    return table\n", ""), "wrong", "no-return"),
    ("per-group", "", "empty", None),
    ("per-group", "x = 1\n", "wrong", "no-function"),
    ("weekend", _W_START, "empty", None),
    ("weekend", _W, "correct", None),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"one number for the day\", strict=True, raises=AssertionError)"), "correct", None),
    ("weekend", _with_mark("@pytest.mark.xfail(strict=True, reason='Saturday and Sunday')"), "correct", None),
    ("weekend", _W_START.replace("import pytest\n", "import pytest\n\nx = 1\n"), "wrong", "no-mark"),
    ("weekend", _with_mark("@pytest.mark.skip(reason=\"weekday is one number\")"), "wrong", "skip"),
    ("weekend", _with_mark("@pytest.mark.xfail(\"weekday is one number\", strict=True)"), "wrong", "args"),
    ("weekend", _with_mark("@pytest.mark.xfail(strict=True)"), "wrong", "reason"),
    ("weekend", _with_mark("@pytest.mark.xfail"), "wrong", "reason"),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"\", strict=True)"), "wrong", "reason"),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"weekday\", strict=True, run=False)"), "wrong", "not-run"),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"weekday is one number\")"), "wrong", "strict"),
    ("weekend", _with_mark("@pytest.mark.xfail(condition=False, reason=\"weekday is one number\", strict=True)"), "wrong", "condition"),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"weekday is one number\", strict=True, raises=ValueError)"), "wrong", "raises"),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"weekday is one number\", strict=True, raises=AssertionError)"), "correct", None),
    ("weekend", _with_mark("@pytest.mark.xfail(reason=\"weekday is one number\", strict=False)"), "wrong", "strict"),
    ("weekend", _W.replace("<= 20", "<= 2000"), "wrong", "passes-course"),
    ("weekend", _W_START.replace("<= 20", "<= 2000"), "wrong", "passes-course"),
    ("weekend", _W.replace("saturday[\"weekday\"] = 6", "saturday[\"weekdy\"] = 6").replace("expect(saturday)", "expect(saturday.drop(columns=\"weekday\"))"), "wrong", "other-error"),
    ("weekend", _W.replace("saturday[\"weekday\"] = 6", "saturday[\"weathersit\"] = 3"), "wrong", "fails-fixed"),
    ("weekend", _W.replace("def test_weekend(expect):", "def check_weekend(expect):"), "wrong", "no-test"),
    ("weekend", _W.replace("import pytest\n", ""), "error", None),
    ("weekend", "", "empty", None),
    ("card", _C_START, "empty", None),
    ("card", _C, "correct", None),
    ("card", _c("        lines.append(\"| \" + name + \" | \" + str(row['hours']) + \" | \" + str(round(row['rmse'], 1)) + \" |\")\n"), "correct", None),
    ("card", _c("        lines.append(f\"|{name}|{row['hours']}|" + _C_FILL + "|\")\n"), "correct", None),
    ("card", _c("        lines.append(f\"| {name} | {row['hours']} | " + _C_FILL + "\")\n"), "correct", None),
    ("card", _C_NO_ROUND, "wrong", "no-round"),
    ("card", _C_WHOLE, "wrong", "whole"),
    ("card", _C_SWAPPED, "wrong", "swapped"),
    ("card", _C_NO_PIPES, "wrong", "no-pipes"),
    ("card", _c("        pass\n").replace("    lines += [\"\", \"## Known limits\"", "    lines.append(f\"| {name} | {row['hours']} | " + _C_FILL + " |\")\n    lines += [\"\", \"## Known limits\""), "wrong", "one-row"),
    ("card", _c("        lines.append(\"| working day | 1119 | 48.2 |\")\n"), "wrong", None),
    ("card", _c("        lines.append(f\"| {name} | {row['hours']} | " + _C_FILL + " |\")\n").replace("    return \"\\n\".join(lines) + \"\\n\"\n", ""), "wrong", "no-return"),
    ("card", _C.replace("def card(config, groups, limits):", "def card(config, groups):"), "error", "arguments"),
    ("card", "", "empty", None),
]
