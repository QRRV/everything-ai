# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-project-service, step 3 of the project ops: the demand model answers requests from a
server on 127.0.0.1, every request leaves one line of JSON in a log with its latency, a drift check compares each
month of what the service received with the same month of the training hours, and once the real rentals of a month
are in, a rule says when the model needs training again.

`serve` reads the clock before and after the answer and writes the log line; the check gives it a clock of its own
that moves only while the stand-in model answers, so every run measures the same. `drift_by_month` gives the PSI of
each month of the log against the same month of training, with the PSI of the lesson *When the world changes:
drift*, which the course puts in the project as ops/psi.py. `train_again` measures each month with the rmse_per of
the lesson *Project: tests and a model card* and names the months that are more than LIMIT times as far off as the
package on its validation days. The course's model and package are those of that lesson: h16-lr0.01, trained the
way the course's train_version trains it. The server is the one of the lesson *Your model behind an API*.
The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011 and 2012, from the Bike Sharing
dataset of Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the exercises and the pictures are the course's
own."""
import contextlib
import copy
import html
import importlib.util
import io
import json
import math
import re
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from everything_ai import paths, projects, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, Result, Timeout, kept, run_code, with_time_limit

LESSON = "sys-project-service"
LANGS = ("en", "nl")

projects.provide("sys-project-tracked-model")   # ops/demand.py and ops/safetensors_io.py
projects.provide("sys-behavior-tests")          # ops/serving.py and ops/conftest.py
projects.provide(LESSON)                        # ops/psi.py
projects.builds_on(("ops", "report", "sys-project-tested-model", "per-group"))   # your rmse_per


def _checks_of(lesson: str):
    """Another lesson's checks, loaded once, without changing which modules this lesson builds on."""
    def load():
        said = projects.said()
        try:
            spec = importlib.util.spec_from_file_location(f"_course_checks_{lesson.replace('-', '_')}_service",
                                                          paths.LESSONS / lesson / "checks.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            projects.builds_on(*said)
        return module
    return kept((LESSON, "checks", lesson), load)


TESTED = _checks_of("sys-project-tested-model")   # the course's model and package, rmse_per, the card
API = _checks_of("sys-serving-api")               # the server on 127.0.0.1, problem(), send()


def _given(name: str):
    """The course's own copy of a file it gives in the project: the checks use it, so they stay the same when you
    change yours."""
    path = paths.LESSONS / LESSON / "project" / "ops" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_course_ops_{name}_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = TESTED.demand
safetensors_io = TESTED.safetensors_io
psi_module = _given("psi")
psi_of = psi_module.psi_of

HOURS_2011 = TESTED.HOURS_2011                    # 8,645 hours of 365 days
TRAIN_HOURS = TESTED.TRAIN_HOURS                  # 6,912 hours of 292 days
HOURS_2012 = TESTED.later_hours()                 # 8,734 hours of 366 days
FIELDS = list(API.FIELDS)                         # what the body of a request holds: hr and the 10 features
LIMIT = 1.5                                       # train again when a month is more than this many times as far off
MONTHS = list(range(1, 13))
INPUTS = ["temp", "hum", "windspeed"]             # the inputs the drift check watches
JANUARY, JULY = 1, 7


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def psi_text(x: float, lang: str) -> str:
    """A PSI for the lesson's text: two decimals, or three under 0.01, so that 0.002 does not show as 0.00."""
    return number(x, 3 if abs(x) < 0.01 else 2, lang)


# ======== the service: the course's model and its answer to one request ===================================================

def answering(model, scale):
    """The answer of a service to one request with `model` and the mean and std of its package: the bikes as a whole
    number, never below 0, as the serving path of the lesson *Before and after the model: the serving path* gives
    it. The request is a dict with hr and the 10 features."""
    mean, std = torch.tensor(scale["mean"]), torch.tensor(scale["std"])

    def answer(request):
        hour = torch.zeros(24)
        hour[int(request["hr"])] = 1.0
        values = torch.tensor([float(request[f]) for f in demand.FEATURES])
        x = torch.cat([hour, (values - mean) / std]).reshape(1, -1)
        with torch.no_grad():
            pred = model(x)[0, 0].item()
        return {"bikes": max(0, round(pred))}
    return answer


def _course_answer():
    model, scale, _ = TESTED._model("course")
    return answering(model, scale)


def answer(request):
    """What the course's service answers for one request: {"bikes": ...}."""
    return kept((LESSON, "answer"), _course_answer)(request)


def slow_answer(request):
    """The course's answer from a model that takes 1.5 seconds, such as one on a busy computer."""
    time.sleep(1.5)
    return answer(request)


def request_of(row) -> dict:
    """The body of a request for one hour of the data: hr and the 10 features, whole numbers where they are."""
    return {f: (int(row[f]) if f in API._WHOLE else float(row[f])) for f in FIELDS}


def _table(hours) -> pd.DataFrame:
    """A table as the log gives it: each hour's request, and the course's answer as the column bikes."""
    rows = []
    for _, row in hours.iterrows():
        req = request_of(row)
        rows.append({**req, "bikes": answer(req)["bikes"]})
    return pd.DataFrame(rows, columns=FIELDS + ["bikes"])


def served() -> pd.DataFrame:
    """What the course's service logged for every request of 2012, as a table: the fields and the answer."""
    return kept((LESSON, "served"), lambda: _table(HOURS_2012))


def trained() -> pd.DataFrame:
    """The training hours of 2011 as the same table, with the course's answers as the column bikes and the bikes
    rented as cnt."""
    def make():
        t = _table(TRAIN_HOURS)
        t["cnt"] = TRAIN_HOURS["cnt"].to_numpy()
        return t
    return kept((LESSON, "trained"), make)


def answers_2012() -> torch.Tensor:
    return torch.tensor(served()["bikes"].to_numpy(dtype="float32"))


def course_config() -> dict:
    return TESTED.course_config()


def answers() -> torch.Tensor:
    """What the course's service answered for every hour of 2012, as a tensor: the lesson's cell uses it."""
    return answers_2012().clone()


def config() -> dict:
    """config.json of the course's package, as a dict: the lesson's cell uses it."""
    return dict(course_config())


def _course_drift(train, log, column) -> dict:
    out = {}
    for month in sorted(set(log["mnth"])):
        now = log[log["mnth"] == month][column].to_numpy()
        then = train[train["mnth"] == month][column].to_numpy()
        out[int(month)] = psi_of(then, now)
    return out


def _numbers() -> dict:
    """The numbers the lesson's texts give, from the data and the course's model."""
    s, t = served(), trained()
    table = TESTED.rmse_per(HOURS_2012, answers_2012(), "mnth")
    rmse = {m: row["rmse"] for m, row in table.items()}
    val = course_config()["val_rmse"]
    drift = {c: _course_drift(t, s, c) for c in INPUTS + ["bikes"]}
    rented = _course_drift(t, HOURS_2012, "cnt")
    whole = {c: {m: psi_of(t[c].to_numpy(), s[s["mnth"] == m][c].to_numpy()) for m in MONTHS} for c in INPUTS + ["bikes"]}
    return {"rmse": rmse, "hours": {m: row["hours"] for m, row in table.items()}, "val": val,
            "limit": LIMIT * val, "flagged": [m for m in MONTHS if rmse[m] > LIMIT * val],
            "drift": drift, "rented": rented, "whole": whole,
            "inputs_flagged": [m for m in MONTHS if any(drift[c][m] > 0.25 for c in INPUTS)],
            "all_2012": float(((answers_2012() - torch.tensor(HOURS_2012["cnt"].to_numpy(dtype="float32"))) ** 2).mean().sqrt()),
            "requests": len(s)}


def numbers() -> dict:
    return kept((LESSON, "numbers"), _numbers)


def _ready() -> None:
    """Train the course's model and make the tables of the checks before a check, outside its time limit: the first
    time takes a few seconds."""
    TESTED._model("course")
    served()
    trained()
    numbers()


# ======== what the checks give the learner's code =========================================================================

class Log:
    """What serve writes its lines to: an open log file, as far as serve uses it. `write` adds text at the end; the
    lines so far stay here too, for the check and the demo."""

    def __init__(self, file=None):
        self.file, self.parts = file, []

    def write(self, text):
        if self.file is not None:
            self.file.write(text)
        self.parts.append(str(text))
        return len(str(text))

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def lines(self) -> list:
        return [ln for ln in "".join(self.parts).split("\n") if ln.strip()]

    def __repr__(self):
        return "log"

    def __deepcopy__(self, memo):
        return Log()


class _Clock:
    """The check's clock: it stands still, and moves on only while the stand-in model answers, so that every run
    measures the same."""

    START = 2_000.0          # where perf_counter starts: a moment of its own
    WALL = 1_700_000_000.0   # where time.time starts: seconds since 1970, as a real wall clock

    def __init__(self, at: float = START):
        self.at = at

    def seconds(self) -> float:
        return self.at

    def nanoseconds(self) -> int:
        return int(round(self.at * 1e9))

    def wall(self) -> float:
        return self.WALL + self.at - self.START

    def wall_ns(self) -> int:
        return int(round(self.wall() * 1e9))


_CLOCK: list = [None]   # the clock of the run the check is doing now, if any


class Timed:
    """The course's answer, as the check calls it: it moves the check's clock on by `seconds`, as if the model took
    that long to answer."""

    def __init__(self, name: str, seconds: float):
        self.name, self.seconds = name, seconds

    def __repr__(self):
        return self.name

    def __call__(self, request):
        out = answer(request)
        if _CLOCK[0] is not None:
            _CLOCK[0].at += self.seconds
        return out

    def __deepcopy__(self, memo):
        return self


class _Request(dict):
    """A request the check sends, shown by a short name in its messages."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):   # print shows the body itself
        return dict.__repr__(self)

    def __deepcopy__(self, memo):
        return dict(self)


def _hour(day: int, hr: int) -> dict:
    row = HOURS_2012[(HOURS_2012["day"] == day) & (HOURS_2012["hr"] == hr)].iloc[0]
    return request_of(row)


ANSWER = Timed("C.answer", 0.0004)             # the course's model, as fast as it is: 0.4 ms
SLOW_ANSWER = Timed("C.slow_answer", 1.5)      # a model that takes 1.5 seconds, such as one on a busy computer
REQUEST = _Request("C.REQUEST", _hour(100, 8))   # Tuesday 10 April 2012 at 8:00
EVENING = _Request("C.EVENING", _hour(100, 17))  # the same day at 17:00
LOG = Log()


def _parsed(line: str):
    try:
        return json.loads(line)
    except ValueError:
        return line


class _Named:
    """A value the check calls by a short name in its messages, such as `served`; the learner's code gets its own
    copy. `make` gives the value the first time it is needed."""

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


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


def _dots_left(got, text_) -> bool:
    return re.search(r"^[^#\n]*\.\.\.", text_ or "", re.MULTILINE) is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. What comes in its place?",
                nl="Er staat nog een `...` in je code. Wat komt er op deze plek?")


class _ReadyBox(FunctionBox):
    """A function exercise whose cases need the course's model: it is trained before the check, outside its time
    limit."""

    def check(self, text, inputs=None) -> Result:
        _ready()
        return super().check(text, inputs)


# ======== exercise 1: serve, with its latency in the log ==================================================================

_S_START = ("import json\n"
            "import time\n"
            "\n\n"
            "def serve(answer, request, log):\n"
            "    start = time.perf_counter()                     # the clock, in seconds, before the answer\n"
            "    body = answer(request)                          # what the model answers, such as {\"bikes\": 431}\n"
            "    ms = ...                                        # how long the answer took, in milliseconds\n"
            "    line = {\"ms\": round(ms, 3), \"bikes\": body[\"bikes\"], \"request\": request}\n"
            "    log.write(json.dumps(line) + \"\\n\")              # one line of JSON per request\n"
            "    return body\n")
_S_LINE = "    ms = ...                                        #"


def _s(value: str) -> str:
    line = f"    ms = {value}"
    return _S_START.replace(_S_LINE, f"{line}{' ' * max(1, 52 - len(line))}#")


_S_FILL = "(time.perf_counter() - start) * 1000"
_S = _s(_S_FILL)
_S_SECONDS = _s("time.perf_counter() - start")
_S_DIVIDED = _s("(time.perf_counter() - start) / 1000")
_S_NEGATIVE = _s("(start - time.perf_counter()) * 1000")
_S_WHOLE = _s("round((time.perf_counter() - start) * 1000)")
_S_BRACKETS = _s("time.perf_counter() - start * 1000")


class ServeBox(_ReadyBox):
    """The exercise serve: each case calls serve(answer, request, log) with the check's clock in the place of the
    module time, and gives what serve gave back and the lines it wrote to log."""

    def _run(self, fn, case):
        timed, request, _ = case
        clock = _Clock()
        fake = types.SimpleNamespace(perf_counter=clock.seconds, time=clock.wall, monotonic=clock.seconds,
                                     process_time=clock.seconds, perf_counter_ns=clock.nanoseconds,
                                     time_ns=clock.wall_ns, monotonic_ns=clock.nanoseconds, sleep=lambda s: None)
        names = getattr(fn, "__globals__", {})
        saved = {}
        for key, value in list(names.items()):
            if isinstance(value, types.ModuleType) and value.__name__ == "time":
                saved[key] = value
                names[key] = fake
            elif key in ("perf_counter", "monotonic", "process_time", "perf_counter_ns", "monotonic_ns") and callable(value):
                saved[key] = value
                names[key] = getattr(fake, key)
        _CLOCK[0] = clock
        log = Log()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                body = fn(timed, copy.deepcopy(request), log)
        finally:
            _CLOCK[0] = None
            names.update(saved)
        if body is None:
            return None
        return {"body": body, "lines": [_parsed(ln) for ln in log.lines()]}


def _ms_of(got) -> list:
    """The ms of every line serve wrote, or None for a line without a number there."""
    if not isinstance(got, dict):
        return []
    out = []
    for line in got.get("lines") or []:
        v = line.get("ms") if isinstance(line, dict) else None
        out.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
    return out


def _two_clocks(got, text_) -> bool:
    """The difference of time.time and perf_counter: about the distance between their two starts, in ms."""
    gap = (_Clock.WALL - _Clock.START) * 1000
    return any(v is not None and abs(abs(v) - gap) < 10_000 for v in _ms_of(got))


def _clock_reading(got, text_) -> bool:
    """A reading of the clock itself: thousands of seconds, in ms."""
    return any(v is not None and abs(v) > 100_000 for v in _ms_of(got))


def _no_line(got, text_) -> bool:
    return isinstance(got, dict) and not got.get("lines")


def _more_lines(got, text_) -> bool:
    return isinstance(got, dict) and len(got.get("lines") or []) > 1


def _serve_shown(v):
    """{got} and {want}: the line serve wrote, with the request cut short."""
    if not isinstance(v, dict):
        return f"`{v!r}`"
    lines = v.get("lines") or []
    if not lines:
        return {"en": "no line", "nl": "geen regel"}
    line = lines[0]
    if isinstance(line, dict):
        short = {k: ("{…}" if k == "request" else val) for k, val in line.items()}
        text_ = json.dumps(short, ensure_ascii=False).replace('"{…}"', "{…}")
    else:
        text_ = str(line)[:90]
    return f"`{text_}`"


# ======== exercise 2: drift_by_month ======================================================================================

_D_START = ("from ops.psi import psi_of\n"
            "\n\n"
            "def drift_by_month(train, served, column):\n"
            "    table = {}\n"
            "    for month in sorted(set(served[\"mnth\"])):                     # each month of the log, once, in order\n"
            "        now = served[served[\"mnth\"] == month][column].to_numpy()  # the values of that month in the log\n"
            "        then = ...                                                # the same month in the training hours\n"
            "        table[int(month)] = psi_of(then, now)                     # the PSI of the month\n"
            "    return table\n")
_D_LINE = "        then = ...                                                #"


def _d(value: str) -> str:
    line = f"        then = {value}"
    return _D_START.replace(_D_LINE, f"{line}{' ' * max(1, 66 - len(line))}#")


_D_FILL = "train[train[\"mnth\"] == month][column].to_numpy()"
_D = _d(_D_FILL)
_D_ALL = _d("train[column].to_numpy()")
_D_LOG = _d("served[served[\"mnth\"] == month][column].to_numpy()")
_D_TEMP = _d("train[train[\"mnth\"] == month][\"temp\"].to_numpy()")
_D_ROWS = _d("train[train[\"mnth\"] == month].to_numpy()")

TRAIN = _Named("train", lambda: trained())
SERVED = _Named("served", lambda: served())
SPRING = _Named("spring", lambda: served()[served()["mnth"] <= 3].reset_index(drop=True))


def _psi_shown(v):
    """{got} and {want}: the PSI of each month with two decimals, or three under 0.01."""
    if isinstance(v, dict) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v.values()):
        def one(x):
            x = float(x)
            return f"{x:.3f}" if abs(x) < 0.01 else f"{x:.2f}"
        return "`{" + ", ".join(f"{k}: {one(x)}" for k, x in v.items()) + "}`"
    return "`" + repr(v)[:200] + "`"


# ======== exercise 3: train_again =========================================================================================

_R_START = ("from ops.report import rmse_per\n"
            "\n"
            "LIMIT = 1.5                                 # train again above 1.5 times the validation RMSE\n"
            "\n\n"
            "def train_again(hours, pred, config):\n"
            "    table = rmse_per(hours, pred, \"mnth\")   # each month: {\"hours\": ..., \"rmse\": ...}\n"
            "    months = []\n"
            "    for month in table:                     # the months, in order\n"
            "        if ...:\n"
            "            months.append(month)\n"
            "    return months\n")


def _r(test: str) -> str:
    return _R_START.replace("        if ...:\n", f"        if {test}:\n")


_R_FILL = "table[month][\"rmse\"] > LIMIT * config[\"val_rmse\"]"
_R = _r(_R_FILL)
_R_NO_LIMIT = _r("table[month][\"rmse\"] > config[\"val_rmse\"]")
_R_BELOW = _r("table[month][\"rmse\"] < LIMIT * config[\"val_rmse\"]")
_R_HOURS = _r("table[month][\"hours\"] > LIMIT * config[\"val_rmse\"]")
_R_FIXED = _r("table[month][\"rmse\"] > 79.56")
_R_AT = _r("table[month][\"rmse\"] >= LIMIT * config[\"val_rmse\"]")


def _edge_hours():
    rows = [(100, 1)] * 4 + [(50, 2)] * 2 + [(10, 3)] * 3
    return pd.DataFrame(rows, columns=["cnt", "mnth"])


HOURS = _Named("hours", lambda: HOURS_2012.copy())
PRED = _Named("pred", answers_2012)
CONFIG = _Named("config", lambda: dict(course_config()))
OTHER_CONFIG = _Named("other_config", lambda: {**course_config(), "val_rmse": 100.0, "run": "h64-lr0.01"})
SPRING_HOURS = _Named("spring_hours", lambda: HOURS_2012[HOURS_2012["mnth"] <= 3].reset_index(drop=True))
SPRING_PRED = _Named("spring_pred", lambda: answers_2012()[torch.tensor((HOURS_2012["mnth"] <= 3).to_numpy())])
EDGE = _Named("edge_hours", _edge_hours)
EDGE_PRED = _Named("edge_pred", lambda: torch.tensor([160.0, 40.0, 160.0, 40.0, 130.0, -30.0, 69.0, -49.0, 10.0]))
EDGE_CONFIG = _Named("edge_config", lambda: {"version": "0f3a9c1e2b7d", "run": "h16-lr0.01", "val_rmse": 40.0})


def _dict_compared(e: BaseException) -> bool:
    return isinstance(e, TypeError) and "dict" in str(e) and ("<" in str(e) or ">" in str(e))


def _months_shown(v):
    if isinstance(v, list):
        return "`[" + ", ".join(str(x) for x in v) + "]`"
    return "`" + repr(v)[:200] + "`"


EXERCISES = {
    "serve": ServeBox(
        name="serve",
        function="serve",
        start=_S_START,
        solution=_S,
        save_as=("ops", "service"),
        visible=[(ANSWER, REQUEST, LOG)],
        hidden=[(SLOW_ANSWER, REQUEST, LOG), (ANSWER, EVENING, LOG)],
        first=[_DOTS],
        show=_serve_shown,
        picture=lambda seen: serve_picture(seen),
        mistakes=[
            Mistake("seconds", wrong=_S_SECONDS,
                    en="`{call}` writes {got}: `time.perf_counter()` counts in seconds, so the difference of two readings is in seconds too, 0.0004 for an answer of 0.4 ms, and `round` to three decimals makes that 0.0. How many milliseconds are there in a second?",
                    nl="`{call}` schrijft {got}: `time.perf_counter()` telt in seconden, dus is het verschil van twee standen ook in seconden, 0,0004 voor een antwoord van 0,4 ms, en `round` op drie decimalen maakt daar 0,0 van. Hoeveel milliseconden zitten er in een seconde?"),
            Mistake("divided", wrong=_S_DIVIDED,
                    en="`{call}` writes {got}: your line makes the seconds a thousand times smaller, so even an answer of 1.5 seconds would come out as 0.002. A millisecond is a thousandth of a second: how many milliseconds are 1.5 seconds?",
                    nl="`{call}` schrijft {got}: je regel maakt de seconden duizend keer kleiner, dus zou zelfs een antwoord van 1,5 seconde uitkomen op 0,002. Een milliseconde is een duizendste seconde: hoeveel milliseconden is 1,5 seconde?"),
            Mistake("brackets", wrong=_S_BRACKETS,
                    en="`{call}` writes {got}: Python multiplies before it subtracts, so only `start` became a thousand times bigger, and the difference of the two readings was never taken first. What makes Python take the difference first?",
                    nl="`{call}` schrijft {got}: Python vermenigvuldigt voordat Python aftrekt, dus werd alleen `start` duizend keer groter, en kwam het verschil van de twee standen nooit eerst. Wat laat Python eerst het verschil nemen?"),
            Mistake("two-clocks", matches=_two_clocks,
                    en="`{call}` writes {got}: the difference of two different clocks. `time.time()` counts the seconds since 1970 and `time.perf_counter()` counts from a moment of its own, so only two readings of the same clock give how long the answer took. Which clock does `start` read?",
                    nl="`{call}` schrijft {got}: het verschil van twee verschillende klokken. `time.time()` telt de seconden sinds 1970 en `time.perf_counter()` telt vanaf een eigen moment, dus geven alleen twee standen van dezelfde klok hoe lang het antwoord duurde. Welke klok leest `start`?"),
            Mistake("clock", matches=_clock_reading,
                    en="`{call}` writes {got}: a reading of the clock itself, not how long the answer took. `time.perf_counter()` counts from a moment of its own, so one reading alone says nothing. Which two readings give how long the answer took?",
                    nl="`{call}` schrijft {got}: een stand van de klok zelf, niet hoe lang het antwoord duurde. `time.perf_counter()` telt vanaf een eigen moment, dus zegt één stand alleen niets. Welke twee standen geven hoe lang het antwoord duurde?"),
            Mistake("negative", wrong=_S_NEGATIVE,
                    en="`{call}` writes {got}: a time below 0, since your line takes the later reading from the earlier one. Which reading comes first in the difference?",
                    nl="`{call}` schrijft {got}: een tijd onder 0, want je regel trekt de latere stand af van de eerdere. Welke stand komt eerst in het verschil?"),
            Mistake("whole", wrong=_S_WHOLE, state="almost",
                    en="`{call}` writes {got}: whole milliseconds, so every fast answer in the log would say 0. `round(ms, 3)` in the next line already keeps three decimals: can `ms` stay the full number?",
                    nl="`{call}` schrijft {got}: hele milliseconden, dus zou elk snel antwoord in de log 0 zeggen. `round(ms, 3)` op de volgende regel houdt al drie decimalen over: kan `ms` het volledige getal blijven?"),
            Mistake("no-line", matches=_no_line,
                    en="With `{call}`, your function writes no line to `log`, so the request leaves no trace. Is the line with `log.write` still there?",
                    nl="Met `{call}` schrijft je functie geen regel naar `log`, dus laat het verzoek geen spoor na. Staat de regel met `log.write` er nog?"),
            Mistake("more-lines", matches=_more_lines,
                    en="With `{call}`, your function writes more than one line to `log` for one request. The drift check reads one line per request: is there one `log.write`, and nothing else that writes?",
                    nl="Met `{call}` schrijft je functie meer dan één regel naar `log` voor één verzoek. De driftcheck leest één regel per verzoek: staat er één `log.write`, en niets anders dat schrijft?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with how long the answer took, in milliseconds, from `start` and the clock after the answer.",
                      "nl": "Vervang de `...` door hoe lang het antwoord duurde, in milliseconden, uit `start` en de klok na het antwoord."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(answer, request, log)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(answer, request, log)` houden?"},
            "wrong": {"en": "`{call}` writes {got}, and it should write {want}. Is `ms` how long `answer` took, in milliseconds: the clock after the answer against the clock before it?",
                      "nl": "`{call}` schrijft {got}, en het moet {want} zijn. Is `ms` hoe lang `answer` duurde, in milliseconden: de klok na het antwoord tegenover de klok van vlak voor het antwoord?"},
            "almost": {"en": "Right for `C.answer`. But `{call}` writes {got}, and it should write {want}. Does `ms` come from the two readings of the clock alone, with no number of its own?",
                       "nl": "Goed voor `C.answer`. Maar `{call}` schrijft {got}, en het moet {want} zijn. Komt `ms` alleen uit de twee standen van de klok, zonder een eigen getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back, so the server has no answer to send. Does it still end with `return body`?",
                          "nl": "Met `{call}` geeft je functie niets terug, dus heeft de server geen antwoord om te sturen. Eindigt de functie nog met `return body`?"},
            "correct": {"en": "Right in all {n} cases. Every request now leaves one line in the log, with how long the answer took: 0.4 ms with `C.answer`, 1,500 ms with `C.slow_answer`. Your `serve` is now in your project, in `{saved}`.",
                        "nl": "Goed in alle {n} gevallen. Elk verzoek laat nu één regel achter in de log, met hoe lang het antwoord duurde: 0,4 ms met `C.answer`, 1.500 ms met `C.slow_answer`. Je `serve` staat nu in je project, in `{saved}`."},
            "kept": _kept("serve"),
        },
    ),
    "drift": _ReadyBox(
        name="drift",
        function="drift_by_month",
        start=_D_START,
        solution=_D,
        save_as=("ops", "drift"),
        visible=[(TRAIN, SERVED, "temp")],
        hidden=[(TRAIN, SERVED, "bikes"), (TRAIN, SERVED, "hum"), (TRAIN, SPRING, "windspeed")],
        first=[_DOTS],
        show=_psi_shown,
        picture=lambda seen: drift_picture(seen),
        mistakes=[
            Mistake("all", wrong=_D_ALL,
                    en="`{call}` gives {got}: each month against all of training, warm and cold months together. Then every month lies far above 0.25, since a single month never looks like a whole year, and the drift that matters is lost in the seasons. Which hours of training should a month of the log be held against?",
                    nl="`{call}` geeft {got}: elke maand tegen de hele training, warme en koude maanden samen. Dan ligt elke maand ver boven 0,25, want één maand lijkt nooit op een heel jaar, en de drift die ertoe doet gaat verloren in de seizoenen. Tegen welke uren van de training moet een maand van de log staan?"),
            Mistake("log", wrong=_D_LOG,
                    en="`{call}` gives {got}: 0 for every month, since `then` holds the same values as `now`, the month in the log itself. Which table holds the hours the model learned from?",
                    nl="`{call}` geeft {got}: 0 voor elke maand, want `then` bevat dezelfde waarden als `now`, de maand in de log zelf. Welke tabel bevat de uren waarvan het model leerde?"),
            Mistake("rows", wrong=_D_ROWS,
                    en="`{call}` gives {got}: `then` holds every column of the training hours of that month, all mixed in one array, not only `column`. What do you add, as `now` has it, so that only the values of `column` are left?",
                    nl="`{call}` geeft {got}: `then` bevat elke kolom van de trainingsuren van de maand, allemaal door elkaar in één array, niet alleen `column`. Wat voeg je toe, zoals `now` het heeft, zodat alleen de waarden van `column` overblijven?"),
            Mistake("temp", wrong=_D_TEMP, state="almost",
                    en="`{call}` gives {got}: the PSI of the temperature, whatever `column` asks for. Which name in your line should say which column to take?",
                    nl="`{call}` geeft {got}: de PSI van de temperatuur, welke kolom `column` ook vraagt. Welke naam in je regel moet zeggen welke kolom je neemt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the values of `column` in the same month of the training hours, as `now` takes them from the log.",
                      "nl": "Vervang de `...` door de waarden van `column` in dezelfde maand van de trainingsuren, zoals `now` de waarden uit de log haalt."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(train, served, column)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(train, served, column)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `then` hold the values of `column` in the training hours of `month`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat `then` de waarden van `column` in de trainingsuren van `month`?"},
            "almost": {"en": "Right for `\"temp\"`. But `{call}` gives {got}, and it should give {want}. Does your line use `month` and `column`, and nothing fixed?",
                       "nl": "Goed voor `\"temp\"`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je regel `month` en `column`, en niets vasts?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return table`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met `return table`?"},
            "correct": {"en": "Right in all {n} cases. Against the same month of 2011, the temperature of 2012 lies above 0.25 in 5 months, and the answers of your model in none. Your `drift_by_month` is now in your project, in `{saved}`.",
                        "nl": "Goed in alle {n} gevallen. Tegen dezelfde maand van 2011 ligt de temperatuur van 2012 in 5 maanden boven 0,25, en de antwoorden van je model in geen enkele. Je `drift_by_month` staat nu in je project, in `{saved}`."},
            "kept": _kept("drift_by_month"),
        },
    ),
    "retrain": _ReadyBox(
        name="retrain",
        function="train_again",
        start=_R_START,
        solution=_R,
        save_as=("ops", "retrain"),
        visible=[(HOURS, PRED, CONFIG)],
        hidden=[(HOURS, PRED, OTHER_CONFIG), (SPRING_HOURS, SPRING_PRED, CONFIG), (EDGE, EDGE_PRED, EDGE_CONFIG)],
        first=[_DOTS],
        show=_months_shown,
        picture=lambda seen: retrain_picture(seen),
        mistakes=[
            Mistake("dict", error=_dict_compared,
                    en="With `{call}`, your test compares `table[month]`, a dict with two keys, with a number. Which key of that dict holds how far off the model was?",
                    nl="Met `{call}` vergelijkt je test `table[month]`, een dict met twee keys, met een getal. Welke key van de dict bevat hoe ver het model ernaast zat?"),
            Mistake("no-limit", wrong=_R_NO_LIMIT,
                    en="`{call}` gives {got}: every month that is further off than the validation days, January too, at 77.6 against 53.04. On new data a model is always a little further off. How much further may a month be before it counts?",
                    nl="`{call}` geeft {got}: elke maand die verder ernaast zit dan de validatiedagen, ook januari, met 77,6 tegen 53,04. Op nieuwe data zit een model er altijd iets verder naast. Hoeveel verder mag een maand zitten voordat de maand telt?"),
            Mistake("below", wrong=_R_BELOW,
                    en="`{call}` gives {got}: the months that are still close enough, not the ones that need training again. Which way should the comparison point?",
                    nl="`{call}` geeft {got}: de maanden die nog dicht genoeg zitten, niet de maanden die opnieuw training nodig hebben. Welke kant moet de vergelijking op wijzen?"),
            Mistake("hours", wrong=_R_HOURS,
                    en="`{call}` gives {got}: your test looks at `\"hours\"`, the number of hours in a month, about 700 in each. Which key of `table[month]` holds how far off the model was?",
                    nl="`{call}` geeft {got}: je test kijkt naar `\"hours\"`, het aantal uren in een maand, ongeveer 700 in elke maand. Welke key van `table[month]` bevat hoe ver het model ernaast zat?"),
            Mistake("fixed", wrong=_R_FIXED, state="almost",
                    en="`{call}` gives {got}: right for the course's package, but your test holds a number of its own. A new version, with another RMSE on its validation days, needs another limit: which names of the code give it?",
                    nl="`{call}` geeft {got}: goed voor het pakket van de course, maar je test bevat een eigen getal. Een nieuwe versie, met een andere RMSE op de validatiedagen, heeft een andere grens nodig: welke namen in de code geven de grens?"),
            Mistake("at-limit", wrong=_R_AT, state="almost",
                    en="`{call}` gives {got}: a month exactly at the limit counts as too far off. `LIMIT` says how far off a month may be: is a month exactly there still within it?",
                    nl="`{call}` geeft {got}: een maand precies op de grens telt als te ver ernaast. `LIMIT` zegt hoe ver een maand ernaast mag zitten: valt een maand precies op de grens daar nog binnen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test of a month: is its RMSE more than `LIMIT` times the RMSE of the package on its validation days?",
                      "nl": "Vervang de `...` door de test van een maand: is de RMSE meer dan `LIMIT` keer de RMSE van het pakket op de validatiedagen?"},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(hours, pred, config)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(hours, pred, config)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test hold the RMSE of `month` against `LIMIT` times `config[\"val_rmse\"]`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zet je test de RMSE van `month` tegenover `LIMIT` keer `config[\"val_rmse\"]`?"},
            "almost": {"en": "Right for `config`. But `{call}` gives {got}, and it should give {want}. Does your test use `LIMIT` and `config`, and no number of its own?",
                       "nl": "Goed voor `config`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je test `LIMIT` en `config`, en geen eigen getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return months`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met `return months`?"},
            "correct": {"en": "Right in all {n} cases. Your rule names every month from February on: in January your model was 77.6 bikes per hour off, 1.46 times its 53.04, and in February 89.5. Your `train_again` is now in your project, in `{saved}`.",
                        "nl": "Goed in alle {n} gevallen. Je regel noemt elke maand vanaf februari: in januari zat je model er 77,6 fietsen per uur naast, 1,46 keer de 53,04, en in februari 89,5. Je `train_again` staat nu in je project, in `{saved}`."},
            "kept": _kept("train_again"),
        },
    ),
}
REQUIRED = ["serve", "drift", "retrain", "next"]

_S_WRITE = '    log.write(json.dumps(line) + "\\n")              # one line of JSON per request\n'
_D_RETURN = "    return table\n"
_R_RETURN = "    return months\n"
CASES = [
    ("serve", _S_START, "empty", None),
    ("serve", "", "empty", None),
    ("serve", _S, "correct", None),
    ("serve", _s("1000 * (time.perf_counter() - start)"), "correct", None),
    ("serve", _S.replace("start = time.perf_counter()", "start = time.time()      ").replace(_S_FILL, "(time.time() - start) * 1000"),
     "correct", None),
    ("serve", _s("(time.time() - start) * 1000"), "wrong", "two-clocks"),
    ("serve", _S_BRACKETS, "wrong", "brackets"),
    ("serve", _S_SECONDS, "wrong", "seconds"),
    ("serve", _S_DIVIDED, "wrong", "divided"),
    ("serve", _s("time.perf_counter() * 1000"), "wrong", "clock"),
    ("serve", _S_NEGATIVE, "wrong", "negative"),
    ("serve", _S_WHOLE, "almost", "whole"),
    ("serve", _S.replace(_S_WRITE, ""), "wrong", "no-line"),
    ("serve", _S.replace(_S_WRITE, _S_WRITE + _S_WRITE), "wrong", "more-lines"),
    ("serve", _s("0.4"), "almost", None),
    ("serve", _s("5"), "wrong", None),
    ("serve", _s("time.perf_counter() - begin"), "error", None),
    ("serve", _S.replace("def serve(answer, request, log):", "def serve(answer, request):"), "error", "arguments"),
    ("serve", _S.replace("    return body\n", ""), "wrong", "no-return"),
    ("serve", _S.replace("    return body\n", "    print(body)\n"), "wrong", "prints"),
    ("serve", _S.replace("    return body\n", "    extra = ...\n    return body\n"), "wrong", "dots-left"),
    ("serve", "x = 1\n", "wrong", "no-function"),
    ("drift", _D_START, "empty", None),
    ("drift", "", "empty", None),
    ("drift", _D, "correct", None),
    ("drift", _d("train.loc[train[\"mnth\"] == month, column].to_numpy()"), "correct", None),
    ("drift", _D_ALL, "wrong", "all"),
    ("drift", _D_LOG, "wrong", "log"),
    ("drift", _D_ROWS, "wrong", "rows"),
    ("drift", _D_TEMP, "almost", "temp"),
    ("drift", _d("train[train[\"mnth\"] == month][\"temp\" if column == \"temp\" else \"hum\"].to_numpy()"), "almost", None),
    ("drift", _d("now * 2"), "wrong", None),
    ("drift", _d("train[train[\"month\"] == month][column].to_numpy()"), "error", None),
    ("drift", _D.replace("def drift_by_month(train, served, column):", "def drift_by_month(train, served):"), "error", "arguments"),
    ("drift", _D.replace(_D_RETURN, ""), "wrong", "no-return"),
    ("drift", _D.replace(_D_RETURN, "    print(table)\n"), "wrong", "prints"),
    ("drift", _D.replace(_D_RETURN, "    extra = ...\n" + _D_RETURN), "wrong", "dots-left"),
    ("drift", "x = 1\n", "wrong", "no-function"),
    ("retrain", _R_START, "empty", None),
    ("retrain", "", "empty", None),
    ("retrain", _R, "correct", None),
    ("retrain", _r("table[month][\"rmse\"] / config[\"val_rmse\"] > LIMIT"), "correct", None),
    ("retrain", _r("table[month] > LIMIT * config[\"val_rmse\"]"), "error", "dict"),
    ("retrain", _R_NO_LIMIT, "wrong", "no-limit"),
    ("retrain", _R_BELOW, "wrong", "below"),
    ("retrain", _R_HOURS, "wrong", "hours"),
    ("retrain", _R_FIXED, "almost", "fixed"),
    ("retrain", _R_AT, "almost", "at-limit"),
    ("retrain", _r("table[month][\"rmse\"] > LIMIT * config[\"val_rmse\"] or table[month][\"hours\"] == 3"), "almost", None),
    ("retrain", _r("table[month][\"rmse\"] > 2 * config[\"val_rmse\"]"), "wrong", None),
    ("retrain", _r("table[month][\"mse\"] > LIMIT * config[\"val_rmse\"]"), "error", None),
    ("retrain", _R.replace("def train_again(hours, pred, config):", "def train_again(hours, pred):"), "error", "arguments"),
    ("retrain", _R.replace(_R_RETURN, ""), "wrong", "no-return"),
    ("retrain", _R.replace(_R_RETURN, "    print(months)\n"), "wrong", "prints"),
    ("retrain", _R.replace(_R_RETURN, "    extra = ...\n" + _R_RETURN), "wrong", "dots-left"),
    ("retrain", "x = 1\n", "wrong", "no-function"),
]


# ======== pictures ========================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_PRE = ("font-family:var(--ea-font-code);font-size:12px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;"
        "margin:0.35rem 0 0;padding:0.4rem 0.55rem;border:1px solid var(--ea-line);border-radius:4px;color:var(--ea-ink)")
INITIALS = {"en": ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
            "nl": ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]}
MONTH_NAMES = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                      "November", "December"],
               "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
                      "november", "december"]}
_STOPPED = object()   # what _call gives for a function that stopped with an error or took too long
PICTURE_SECONDS = 5   # what the learner's function gets for all the calls of one picture together


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t_svg(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
           anchor: str = "start", mono: bool = False) -> str:
    family = "font-family:var(--ea-font-code);" if mono else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{family}font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _key(items: list) -> str:
    """A small legend: a swatch and a word for each kind of mark."""
    out = []
    for kind, color, word in items:
        if kind == "bar":
            mark = (f'<span style="display:inline-block;width:10px;height:10px;background:{color};margin-right:4px;'
                    f'vertical-align:-1px"></span>')
        else:
            mark = (f'<span style="display:inline-block;width:16px;height:0;border-top:2px {kind} {color};'
                    f'margin-right:4px;vertical-align:3px"></span>')
        out.append(f'<span style="margin-right:12px;white-space:nowrap">{mark}{html.escape(word)}</span>')
    return f'<div style="{_UI};margin:0 0 0.2rem">{"".join(out)}</div>'


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    try:
        return with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _list_of(items: list, lang: str) -> str:
    items = [str(x) for x in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + (" and " if lang == "en" else " en ") + items[-1]


def month_list(months: list, lang: str) -> str:
    """Months as words: a run of three or more as "February to December"."""
    names = MONTH_NAMES[_lang(lang)]
    months = sorted(months)
    runs, start = [], None
    for k, m in enumerate(months):
        if start is None:
            start = m
        if k + 1 == len(months) or months[k + 1] != m + 1:
            runs.append((start, m))
            start = None
    words = []
    for a, b in runs:
        if b - a >= 2:
            words.append(f"{names[a - 1]} {'to' if lang == 'en' else 'tot en met'} {names[b - 1]}")
        else:
            words.extend(names[m - 1] for m in range(a, b + 1))
    return _list_of(words, lang)


YOURS_TEXT = {
    "en": {
        "serve": {"title": "Your serve on the 24 requests of 14 July 2011",
                  "intro": "The course's model answers each hour of the day, and your serve times each answer, with the real clock of your computer.",
                  "axis": "ms per answer", "hour": "hour of the day",
                  "said": "Half of the answers took less than {p50} ms, the slowest {top} ms. The first line your serve wrote:",
                  "zero": "Every line says 0 ms or less: the answers take a fraction of a millisecond. Does `ms` count in milliseconds, from the earlier reading to the later one?",
                  "none": "Your serve wrote no line that the picture can read. The check above says why.",
                  "alt": "24 bars, how long each answer of the day took in milliseconds, and the first line of the log."},
        "drift": {"title": "Your drift_by_month on the log of 2012",
                  "intro": "The PSI of each month of the log against the same month of the training hours: the temperature, and the answers of the model.",
                  "rows": {"temp": "temperature (temp)", "bikes": "answers (bikes)"},
                  "said": "Above 0.25: {months}.", "nothing": "Above 0.25: no month.",
                  "none": "Your drift_by_month gave no numbers that the picture can read. The check above says why.",
                  "alt": "Twelve bars for the temperature and twelve for the answers, the PSI of each month of 2012, with lines at 0.1 and 0.25."},
        "retrain": {"title": "Your train_again on the months of 2012",
                    "intro": "How far off the course's model was in each month of 2012, once the rentals were in, against 1.5 times its RMSE on the validation days.",
                    "axis": "RMSE, bikes per hour", "val": "validation days {v}", "limit": "{limit} × {v} = {lv}",
                    "said": "Your train_again names {months}.", "nothing": "Your train_again names no month.",
                    "none": "Your train_again gave no list of months that the picture can read. The check above says why.",
                    "key": ("named by your train_again", "not named"),
                    "alt": "Twelve bars, the RMSE of each month of 2012, with a line at the validation RMSE and one at 1.5 times it; the bars of the months your train_again names are full."},
    },
    "nl": {
        "serve": {"title": "Je serve op de 24 verzoeken van 14 juli 2011",
                  "intro": "Het model van de course beantwoordt elk uur van de dag, en je serve meet de tijd van elk antwoord, met de echte klok van je computer.",
                  "axis": "ms per antwoord", "hour": "uur van de dag",
                  "said": "De helft van de antwoorden duurde minder dan {p50} ms, de traagste {top} ms. De eerste regel die je serve schreef:",
                  "zero": "Elke regel zegt 0 ms of minder: de antwoorden duren een fractie van een milliseconde. Telt `ms` in milliseconden, van de eerdere stand naar de latere?",
                  "none": "Je serve schreef geen regel die het plaatje kan lezen. De check hierboven zegt waarom.",
                  "alt": "24 staven, hoe lang elk antwoord van de dag duurde in milliseconden, en de eerste regel van de log."},
        "drift": {"title": "Je drift_by_month op de log van 2012",
                  "intro": "De PSI van elke maand van de log tegen dezelfde maand van de trainingsuren: de temperatuur, en de antwoorden van het model.",
                  "rows": {"temp": "temperatuur (temp)", "bikes": "antwoorden (bikes)"},
                  "said": "Boven 0,25: {months}.", "nothing": "Boven 0,25: geen enkele maand.",
                  "none": "Je drift_by_month gaf geen getallen die het plaatje kan lezen. De check hierboven zegt waarom.",
                  "alt": "Twaalf staven voor de temperatuur en twaalf voor de antwoorden, de PSI van elke maand van 2012, met lijnen op 0,1 en 0,25."},
        "retrain": {"title": "Je train_again op de maanden van 2012",
                    "intro": "Hoe ver het model van de course ernaast zat in elke maand van 2012, zodra de verhuurde fietsen bekend waren, tegen 1,5 keer de RMSE op de validatiedagen.",
                    "axis": "RMSE, fietsen per uur", "val": "validatiedagen {v}", "limit": "{limit} × {v} = {lv}",
                    "said": "Je train_again noemt {months}.", "nothing": "Je train_again noemt geen enkele maand.",
                    "none": "Je train_again gaf geen lijst met maanden die het plaatje kan lezen. De check hierboven zegt waarom.",
                    "key": ("genoemd door je train_again", "niet genoemd"),
                    "alt": "Twaalf staven, de RMSE van elke maand van 2012, met een lijn op de validatie-RMSE en een op 1,5 keer de validatie-RMSE; de staven van de maanden die je train_again noemt zijn vol."},
    },
}


def _day_requests() -> list:
    """The 24 requests of Thursday 14 July 2011, as the lesson *Your model behind an API* sent them."""
    return [dict(h) for h in API.DAY]


def serve_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["serve"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    if fn is None:
        return stop
    _ready()
    requests = _day_requests()
    answer(requests[0])   # the first answer of a process is slower: the picture shows the day, not that

    def run():
        log = Log()
        for req in requests:
            fn(answer, dict(req), log)
        return log.lines()
    lines = _call(run)
    if lines is _STOPPED or not lines:
        return stop
    parsed = [_parsed(ln) for ln in lines]
    ms = [p.get("ms") if isinstance(p, dict) else None for p in parsed]
    if len(ms) != 24 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in ms):
        return stop
    L, R, TOP, H = 34, 326, 18, 90
    top = max(max(ms), 0.0)
    nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    ceiling = next((c for c in nice if top <= c), top) if top > 0 else 0.1
    ys = lambda v: TOP + H - H * max(min(v, ceiling), 0.0) / ceiling   # noqa: E731
    slot = (R - L) / 24
    parts = [_t_svg(L, 10, w["axis"], 10.5, 600, "var(--ea-ink-2)")]
    digits = 2 if ceiling < 1 else 0
    for v in (0.0, ceiling / 2, ceiling):
        parts.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{R}" y2="{ys(v):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, ys(v) + 4, number(v, digits, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for h, v in enumerate(ms):
        x = L + slot * h + slot * 0.15
        parts.append(f'<rect x="{x:.1f}" y="{ys(v):.1f}" width="{slot * 0.7:.1f}" height="{TOP + H - ys(v):.1f}" '
                     f'style="fill:var(--ea-ink-3)"/>')
    for h in (0, 6, 12, 18, 23):
        parts.append(_t_svg(L + slot * (h + 0.5), TOP + H + 13, str(h), 9.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t_svg(R, TOP + H + 26, w["hour"], 9.5, 400, "var(--ea-ink-3)", "end"))
    svg = _svg(TOP + H + 30, w["alt"], parts)
    if max(ms) <= 0:
        return yours.figure(w["title"], _note(w["intro"]) + svg + _note(w["zero"], _LOSS_UI), alt=w["alt"])
    p50 = float(np.median(ms))
    said = w["said"].format(p50=number(p50, 3 if p50 < 1 else 1, lang), top=number(max(ms), 3 if max(ms) < 1 else 1, lang))
    first = lines[0]
    if isinstance(parsed[0], dict) and isinstance(parsed[0].get("request"), dict):
        short = dict(parsed[0])
        short["request"] = {k: short["request"][k] for k in list(short["request"])[:3]}
        first = json.dumps(short, ensure_ascii=False)[:-2] + ", …}}"
    pre = f'<pre style="{_PRE}">{html.escape(first)}</pre>'
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(said) + pre, alt=w["alt"])


def _psi_bars(parts: list, top: float, values: dict, color: str, label: str, lang: str) -> float:
    """Twelve bars, the PSI of each month, on a scale to 0.6 with an arrow above a bar that goes further, and the lines
    at 0.1 and 0.25. Gives back where the next row may begin."""
    L, R, H, MAX = 34, 326, 58, 0.6
    ys = lambda v: top + 14 + H - H * min(max(v, 0.0), MAX) / MAX   # noqa: E731
    slot = (R - L) / 12
    parts.append(_t_svg(L, top + 9, label, 10.5, 600, "var(--ea-ink-2)"))
    for v in (0.0, 0.3, 0.6):
        parts.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{R}" y2="{ys(v):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, ys(v) + 4, number(v, 1, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for v in (0.1, 0.25):
        parts.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{R}" y2="{ys(v):.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    for m in MONTHS:
        v = values.get(m)
        x = L + slot * (m - 1) + slot * 0.18
        if v is None:
            continue
        parts.append(f'<rect x="{x:.1f}" y="{ys(v):.1f}" width="{slot * 0.64:.1f}" height="{max(top + 14 + H - ys(v), 0):.1f}" '
                     f'style="fill:{color}"/>')
        if v > MAX:
            cx = x + slot * 0.32
            parts.append(f'<path d="M{cx - 4:.1f} {ys(MAX) + 1:.1f} L{cx:.1f} {ys(MAX) - 5:.1f} L{cx + 4:.1f} {ys(MAX) + 1:.1f} Z" '
                         f'style="fill:var(--ea-ink)"/>')
    for m in MONTHS:
        parts.append(_t_svg(L + slot * (m - 0.5), top + 14 + H + 12, INITIALS[lang][m - 1], 9.5, 400, "var(--ea-ink-3)", "middle"))
    return top + 14 + H + 18


def _psi_table(v) -> dict | None:
    """What drift_by_month gave as {month: PSI}, or None for anything else."""
    if not isinstance(v, dict):
        return None
    out = {}
    for k, x in v.items():
        try:
            m, x = int(k), float(x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or m not in MONTHS:
            return None
        out[m] = x
    return out


def drift_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["drift"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    if fn is None:
        return stop
    _ready()
    got = _call(lambda: {c: fn(trained().copy(), served().copy(), c) for c in ("temp", "bikes")})
    if got is _STOPPED:
        return stop
    tables = {c: _psi_table(v) for c, v in got.items()}
    if any(t is None or not t for t in tables.values()):
        return stop
    parts, top = [], 0.0
    for c, color in (("temp", "var(--ea-data)"), ("bikes", "var(--ea-pred)")):
        top = _psi_bars(parts, top, tables[c], color, w["rows"][c], lang) + 6
    svg = _svg(top, w["alt"], parts)
    notes = []
    for c in ("temp", "bikes"):
        above = [m for m, v in tables[c].items() if v > 0.25]
        said = w["said"].format(months=month_list(above, lang)) if above else w["nothing"]
        notes.append(_note(w["rows"][c] + ": " + said[0].lower() + said[1:]))
    return yours.figure(w["title"], _note(w["intro"]) + svg + "".join(notes), alt=w["alt"])


def _rmse_bars(parts: list, rmse: dict, flagged: list, val: float, limit: float, lang: str, w: dict,
               top: float = 0.0) -> float:
    """Twelve bars, the RMSE of each month, full for the months named, light for the others, with a dotted line at
    the validation RMSE and a dashed one at the limit (named in the key above). Gives back the height used."""
    L, R, H = 34, 326, 110
    high = max(list(rmse.values()) + [limit]) if rmse else limit
    ceiling = 50 * math.ceil(high / 50)
    ys = lambda v: top + 16 + H - H * min(max(v, 0.0), ceiling) / ceiling   # noqa: E731
    slot = (R - L) / 12
    parts.append(_t_svg(L, top + 10, w["axis"], 10.5, 600, "var(--ea-ink-2)"))
    for tick in range(0, ceiling + 1, 50):
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, ys(tick) + 4, number(tick, 0, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for m in MONTHS:
        v = rmse.get(m)
        if v is None:
            continue
        x = L + slot * (m - 1) + slot * 0.18
        opacity = "1" if m in flagged else "0.3"
        parts.append(f'<rect x="{x:.1f}" y="{ys(v):.1f}" width="{slot * 0.64:.1f}" height="{top + 16 + H - ys(v):.1f}" '
                     f'style="fill:var(--ea-loss);fill-opacity:{opacity}"/>')
    for v, dash in ((val, "2 3"), (limit, "5 3")):
        parts.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{R}" y2="{ys(v):.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1.1;stroke-dasharray:{dash}"/>')
    for m in MONTHS:
        weight = 700 if m in flagged else 400
        fill = "var(--ea-ink)" if m in flagged else "var(--ea-ink-3)"
        parts.append(_t_svg(L + slot * (m - 0.5), top + 16 + H + 12, INITIALS[lang][m - 1], 9.5, weight, fill, "middle"))
    return top + 16 + H + 18


def _months_of(v) -> list | None:
    if not isinstance(v, (list, tuple)):
        return None
    try:
        out = [int(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if all(m in MONTHS for m in out) else None


def retrain_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["retrain"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    if fn is None:
        return stop
    _ready()
    got = _call(lambda: fn(HOURS_2012.copy(), answers_2012().clone(), dict(course_config())))
    months = _months_of(got) if got is not _STOPPED else None
    if months is None:
        return stop
    n = numbers()
    parts = []
    height = _rmse_bars(parts, n["rmse"], months, n["val"], n["limit"], lang, w)
    svg = _svg(height, w["alt"], parts)
    said = w["said"].format(months=month_list(months, lang)) if months else w["nothing"]
    key = _key([("bar", "var(--ea-loss)", w["key"][0]),
                ("bar", "var(--ea-loss);opacity:0.3", w["key"][1])])
    key += _key([("dashed", "var(--ea-ink)", w["limit"].format(limit=number(LIMIT, 1, lang), v=number(n["val"], 2, lang),
                                                             lv=number(n["limit"], 2, lang))),
                 ("dotted", "var(--ea-ink)", w["val"].format(v=number(n["val"], 2, lang)))])
    return yours.figure(w["title"], _note(w["intro"]) + key + svg + _note(said), alt=w["alt"])


IDEA_TEXT = {
    "en": {"request": ("request", "hr, temp, …"), "serve": ("serve", "clock · model · clock"),
           "answer": ("answer", '{"bikes": 431}'), "line": '{"ms": 0.4, "bikes": 431, "request": {…}}',
           "file": "one line per request, in requests.jsonl",
           "drift": ("drift check", "each month against the", "same month of training", "at once: a warning"),
           "error": ("error of the month", "when the rentals are in:", "RMSE above 1.5 × 53.04?", "then: train again"),
           "alt": "A request goes to serve, which times the model and gives the answer; each request leaves one line in the log. From the log, a drift check compares each month with the same month of training at once, and once the rentals of a month are in, its RMSE says whether to train again."},
    "nl": {"request": ("verzoek", "hr, temp, …"), "serve": ("serve", "klok · model · klok"),
           "answer": ("antwoord", '{"bikes": 431}'), "line": '{"ms": 0.4, "bikes": 431, "request": {…}}',
           "file": "één regel per verzoek, in requests.jsonl",
           "drift": ("driftcheck", "elke maand tegen dezelfde", "maand van de training", "meteen: een waarschuwing"),
           "error": ("fout van de maand", "als de fietsen bekend zijn:", "RMSE boven 1,5 × 53,04?", "dan: opnieuw trainen"),
           "alt": "Een verzoek gaat naar serve: serve meet de tijd rond het model en geeft het antwoord; elk verzoek laat één regel achter in de log. Vanuit de log vergelijkt een driftcheck meteen elke maand met dezelfde maand van de training, en de RMSE van een maand zegt of het model opnieuw moet trainen, zodra de verhuurde fietsen van de maand bekend zijn."},
}


def _box(x: float, y: float, w: float, h: float, stripe: str | None = None) -> str:
    out = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
           f'style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1.2"/>')
    if stripe:
        out += f'<rect x="{x:.1f}" y="{y + 5:.1f}" width="3" height="{h - 10:.1f}" style="fill:{stripe}"/>'
    return out


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    """A line with a small head at its end, in the grey of the ink."""
    angle = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = angle + 2.6, angle - 2.6
    head = (f"M{x2:.1f} {y2:.1f} L{x2 + 6 * math.cos(a1):.1f} {y2 + 6 * math.sin(a1):.1f} "
            f"L{x2 + 6 * math.cos(a2):.1f} {y2 + 6 * math.sin(a2):.1f} Z")
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:var(--ea-ink-3);stroke-width:1.3"/>'
            f'<path d="{head}" style="fill:var(--ea-ink-3)"/>')


def idea_picture(lang: str) -> str:
    """The service of the project: a request through serve to its answer, a line in the log for each request, and
    from the log the two checks: drift at once, the error of a month once its rentals are in."""
    lang = _lang(lang)
    t = IDEA_TEXT[lang]
    parts = []
    for (x, w, key, stripe) in ((2, 80, "request", "var(--ea-data)"), (98, 122, "serve", None),
                                (236, 92, "answer", "var(--ea-pred)")):
        parts.append(_box(x, 6, w, 48, stripe))
        parts.append(_t_svg(x + w / 2, 26, t[key][0], 11.5, 600, "var(--ea-ink)", "middle"))
        parts.append(_t_svg(x + w / 2, 42, t[key][1], 9.5, 400, "var(--ea-ink-2)", "middle", mono=(key == "answer")))
    parts.append(_arrow(84, 30, 96, 30))
    parts.append(_arrow(222, 30, 234, 30))
    parts.append(_arrow(159, 54, 159, 74))
    parts.append(_box(16, 76, 298, 38))
    parts.append(_t_svg(165, 92, t["line"], 9.5, 400, "var(--ea-ink)", "middle", mono=True))
    parts.append(_t_svg(165, 107, t["file"], 9.5, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_arrow(85, 114, 81, 134))
    parts.append(_arrow(245, 114, 249, 134))
    for (x, key, stripe, last) in ((2, "drift", "var(--ea-data)", "var(--ea-ink-2)"),
                                   (170, "error", "var(--ea-loss)", "var(--ea-loss-text)")):
        parts.append(_box(x, 136, 158, 72, stripe))
        parts.append(_t_svg(x + 12, 154, t[key][0], 11, 600, "var(--ea-ink)"))
        parts.append(_t_svg(x + 12, 170, t[key][1], 9.5, 400, "var(--ea-ink-2)"))
        parts.append(_t_svg(x + 12, 184, t[key][2], 9.5, 400, "var(--ea-ink-2)"))
        parts.append(_t_svg(x + 12, 199, t[key][3], 9.5, 600, last))
    return _svg(212, t["alt"], parts)


# ======== the demo: your service, a year of requests ======================================================================

MONTH_TICKS = list(range(1, 13))
CLIENT = "client"

LIVE_TEXT = {
    "en": {"title": "How far off, month by month", "x": "month of 2012", "y": "RMSE, bikes per hour",
           "alt": "A line of the RMSE of each month of 2012 so far, with dashed lines at the validation RMSE of the package and at 1.5 times it.",
           "val": "validation {v}", "limit": "limit {lv}",
           "line": "**{month}**: {n} requests sent, {total} so far.",
           "start": "**Your service is starting** on 127.0.0.1 …",
           "month": "{month} 2012: what your service saw",
           "rows": {"temp": "temperature", "hum": "humidity", "windspeed": "wind", "bikes": "answers", "cnt": "rentals"},
           "zones": ("small", "moderate", "large"),
           "psi": "PSI against {month} of training",
           "latency": "Your serve: half of the answers under {s50} ms, 99 in 100 under {s99} ms. The client waited {c50} ms for half of the requests, and up to {c99} ms for 99 in 100.",
           "named": "RMSE {r}, {x} times {v}: your train_again names {month}.",
           "not_named": "RMSE {r}, {x} times {v}: your train_again does not name {month}.",
           "alt_month": "Five rows, the PSI of the temperature, the humidity, the wind, the answers and the rentals of the month against the same month of training, each a dot on a line from 0 to 0.6 with marks at 0.1 and 0.25.",
           "rewind": "{Month} 2012: RMSE {r}. Above 0.25 in the inputs: {inputs}. By the end of {month}, your train_again names {named}.",
           "none": "none", "no_month": "no month"},
    "nl": {"title": "Hoe ver ernaast, maand na maand", "x": "maand van 2012", "y": "RMSE, fietsen per uur",
           "alt": "Een lijn van de RMSE van elke maand van 2012 tot nu toe, met lijnen van streepjes op de validatie-RMSE van het pakket en op 1,5 keer de validatie-RMSE.",
           "val": "validatie {v}", "limit": "grens {lv}",
           "line": "**{month}**: {n} verzoeken verstuurd, {total} tot nu toe.",
           "start": "**Je service start** op 127.0.0.1 …",
           "month": "{month} 2012: wat je service zag",
           "rows": {"temp": "temperatuur", "hum": "vochtigheid", "windspeed": "wind", "bikes": "antwoorden", "cnt": "verhuurd"},
           "zones": ("klein", "matig", "groot"),
           "psi": "PSI tegen {month} van de training",
           "latency": "Je serve: de helft van de antwoorden onder {s50} ms, 99 van de 100 onder {s99} ms. De client wachtte {c50} ms op de helft van de verzoeken, en tot {c99} ms op 99 van de 100.",
           "named": "RMSE {r}, {x} keer {v}: je train_again noemt {month}.",
           "not_named": "RMSE {r}, {x} keer {v}: je train_again noemt {month} niet.",
           "alt_month": "Vijf rijen, de PSI van de temperatuur, de vochtigheid, de wind, de antwoorden en de verhuurde fietsen van de maand tegen dezelfde maand van de training, elk een punt op een lijn van 0 tot 0,6 met streepjes op 0,1 en 0,25.",
           "rewind": "{Month} 2012: RMSE {r}. Boven 0,25 in de inputs: {inputs}. Aan het eind van {month} noemt je train_again {named}.",
           "none": "geen", "no_month": "geen enkele maand"},
}


def live_words(lang: str) -> dict:
    return LIVE_TEXT[_lang(lang)]


def _q(values: list, share: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), share)) if values else 0.0


def _ms_text(v: float, lang: str) -> str:
    return number(v, 2 if v < 10 else 1 if v < 100 else 0, lang)


def month_picture(run: dict, month: int, lang: str) -> str:
    """What the service saw in one month: the PSI of each input, of the answers and of the rentals against the same month
    of training, how long the answers took, and whether train_again names the month."""
    lang = _lang(lang)
    w = LIVE_TEXT[lang]
    name = MONTH_NAMES[lang][month - 1]
    rows = [(c, run["drift"].get(c, {}).get(month), "var(--ea-pred)" if c == "bikes" else "var(--ea-data)")
            for c in INPUTS + ["bikes"]] + [("cnt", run["rented"].get(month), "var(--ea-data)")]
    L, R, ROW, TOP, MAX = 8, 300, 30, 20, 0.6
    xs = lambda v: L + (R - L) * min(max(v, 0.0), MAX) / MAX   # noqa: E731
    parts = [_t_svg(L, 11, w["psi"].format(month=name), 10.5, 600, "var(--ea-ink-2)")]
    for k, (c, v, color) in enumerate(rows):
        y = TOP + ROW * k
        parts.append(_t_svg(L, y + 10, w["rows"][c], 10, 400, "var(--ea-ink-2)"))
        parts.append(f'<line x1="{L}" y1="{y + 20}" x2="{R}" y2="{y + 20}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for line in (0.1, 0.25):
            parts.append(f'<line x1="{xs(line):.1f}" y1="{y + 14}" x2="{xs(line):.1f}" y2="{y + 26}" '
                         f'style="stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:2 2"/>')
        if v is None:
            continue
        parts.append(_t_svg(326, y + 10, psi_text(v, lang), 10, 600, "var(--ea-ink)", "end"))
        parts.append(f'<circle cx="{xs(v):.1f}" cy="{y + 20}" r="4" style="fill:{color}"/>')
        if v > MAX:
            parts.append(f'<path d="M{R + 4} {y + 16} L{R + 10} {y + 20} L{R + 4} {y + 24} Z" style="fill:{color}"/>')
    bottom = TOP + ROW * len(rows)
    for v, word in ((0.05, w["zones"][0]), (0.175, w["zones"][1]), (0.425, w["zones"][2])):
        parts.append(_t_svg(xs(v), bottom + 4, word, 9.5, 400, "var(--ea-ink-3)", "middle"))
    for v in (0.1, 0.25):
        parts.append(_t_svg(xs(v), bottom + 16, number(v, 2 if v == 0.25 else 1, lang), 9, 400, "var(--ea-ink-3)", "middle"))
    svg = _svg(bottom + 20, w["alt_month"], parts)
    s, c = run["serve_ms"].get(month, []), run["client_ms"].get(month, [])
    latency = w["latency"].format(s50=_ms_text(_q(s, 0.5), lang), s99=_ms_text(_q(s, 0.99), lang),
                                  c50=_ms_text(_q(c, 0.5), lang), c99=_ms_text(_q(c, 0.99), lang))
    r = run["rmse"].get(month)
    notes = [_note(latency)]
    if r is not None:
        named = month in run["flagged"].get(month, [])
        verdict = w["named" if named else "not_named"].format(r=number(r, 1, lang), x=number(r / run["val"], 2, lang),
                                                              v=number(run["val"], 2, lang), month=name)
        notes.append(_note(verdict, _LOSS_UI if named else _UI))
    title = w["month"].format(month=name[0].upper() + name[1:])
    return yours.figure(title, svg + "".join(notes), alt=w["alt_month"])


def curve_picture(run: dict, lang: str, at: int | None = None) -> str:
    """The RMSE of each month the run got to, as the live chart drew it, with a line at month `at`."""
    from everything_ai import live
    lang = _lang(lang)
    w = LIVE_TEXT[lang]
    c = live.Live({k: w[k] for k in ("title", "x", "y", "alt")}, 12, role="loss", lang=lang, low=0, window=1,
                  xticks=MONTH_TICKS)
    for points, label in references(run, lang):
        c.reference(points, label, role="ink-2")
    c.xs = [m for m in run["months"]]
    c.ys = [run["rmse"][m] for m in run["months"]]
    return c.picture(at=at)


def references(run: dict, lang: str) -> list:
    """The two dashed lines of the chart: the validation RMSE of the package, and LIMIT times it."""
    w = LIVE_TEXT[_lang(lang)]
    return [([(1, run["val"]), (12, run["val"])], w["val"].format(v=number(run["val"], 2, lang))),
            ([(1, run["limit"]), (12, run["limit"])], w["limit"].format(lv=number(run["limit"], 2, lang)))]


def rewind_frame(run: dict, month: int, lang: str) -> str:
    from everything_ai import live
    return live.rewound(curve_picture(run, lang, at=month), [month_picture(run, month, lang)])


def rewind_text(run: dict, month: int, lang: str) -> str:
    lang = _lang(lang)
    w = LIVE_TEXT[lang]
    name = MONTH_NAMES[lang][month - 1]
    above = [w["rows"][c] for c in INPUTS if run["drift"].get(c, {}).get(month, 0.0) > 0.25]
    named = run["flagged"].get(month, [])
    return w["rewind"].format(Month=name[0].upper() + name[1:], month=name, r=number(run["rmse"][month], 1, lang),
                              inputs=_list_of(above, lang) if above else w["none"],
                              named=month_list(named, lang) if named else w["no_month"])


def progress_line(run: dict, month: int, lang: str) -> str:
    w = LIVE_TEXT[_lang(lang)]
    name = MONTH_NAMES[_lang(lang)][month - 1]
    return w["line"].format(month=name[0].upper() + name[1:], n=number(len(run["serve_ms"].get(month, [])), 0, lang),
                            total=number(run["requests"], 0, lang))


def _course_fn(key: str):
    """The course's own serve, drift_by_month or train_again, made from the solution of its exercise."""
    ex = EXERCISES[key]
    return ex._fn(ex.solution)


def _works(key: str, fn) -> bool:
    """Whether a function of yours gives what the course's gives in every case of its exercise."""
    ex = EXERCISES[key]
    solution = _course_fn(key)
    if solution is None:
        return False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for case in list(ex.visible) + list(ex.hidden):
                got = with_time_limit(lambda: ex._run(fn, case), 10)
                if not ex.compare(got, ex._run(solution, case)):
                    return False
    except Exception:   # noqa: BLE001   also Timeout: yours does not work here, and the course's takes its place
        return False
    return True


def _module_fn(module: str, function: str):
    """Your function from your project, and "yours", "missing" or "broken" (there, but it does not load)."""
    path = projects.folder() / "ops" / f"{module}.py"
    if not path.exists():
        return None, "missing"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod = with_time_limit(lambda: importlib.import_module(f"ops.{module}"), 10)
        fn = getattr(mod, function, None)
    except Exception:   # noqa: BLE001
        return None, "broken"
    return (fn, "yours") if callable(fn) else (None, "broken")


def _package():
    """The answer of the package in ops/model, its config, and "yours"; or the course's answer and config, and
    "missing" or "broken" (there, but the course cannot read it, or it gives no answer)."""
    folder = projects.folder() / "ops" / "model"
    if not (folder / "config.json").exists() and not (folder / "model.safetensors").exists():
        return answer, dict(course_config()), "missing"
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        model = demand.DemandNet(int(config["hidden"]))
        model.load_state_dict(safetensors_io.load_file(folder / "model.safetensors"))
        model.eval()
        fn = answering(model, {"mean": config["mean"], "std": config["std"]})
        float(config["val_rmse"])
        if not isinstance(fn(dict(REQUEST))["bikes"], int):
            raise ValueError("an answer in whole bikes")
    except Exception:   # noqa: BLE001   a package the course cannot read: the course's takes its place
        return answer, dict(course_config()), "broken"
    return fn, config, "yours"


def _answered_table(fn, hours) -> pd.DataFrame:
    rows = []
    for _, row in hours.iterrows():
        req = request_of(row)
        rows.append({**req, "bikes": fn(req)["bikes"]})
    t = pd.DataFrame(rows, columns=FIELDS + ["bikes"])
    t["cnt"] = hours["cnt"].to_numpy()
    return t


def log_path() -> Path:
    return projects.folder() / "ops" / "requests.jsonl"


def demo(show=None, months=None) -> dict:
    """Your service on 127.0.0.1: your package (or the course's), your serve, and every request of 2012, month by
    month. After each month, your drift_by_month compares it with the same month of training, the rentals of the month
    come in, and your train_again says whether the model needs training again. `show(month, run)` is called after
    each month. Where a part of yours is missing or does not do what its check asks, the course's takes its place."""
    _ready()
    months = MONTHS if months is None else list(months)
    projects.on_path()
    projects.forget()
    projects.use_course_versions()   # your rmse_per, or the course's where yours is missing or broken
    fns, used, broken = {}, {}, []
    for key, module, function in (("serve", "service", "serve"), ("drift", "drift", "drift_by_month"),
                                  ("retrain", "retrain", "train_again")):
        fn, state = _module_fn(module, function)
        if fn is not None and _works(key, fn):
            fns[key], used[key] = fn, "yours"
        else:
            fns[key], used[key] = _course_fn(key), "course"
            if state != "missing":
                broken.append(key)
    answer_fn, config, package = _package()
    train = trained() if package != "yours" else _answered_table(answer_fn, TRAIN_HOURS)
    val = float(config["val_rmse"])
    run = {"months": [], "rmse": {}, "drift": {c: {} for c in INPUTS + ["bikes"]}, "rented": {}, "flagged": {},
           "serve_ms": {}, "client_ms": {}, "requests": 0, "errors": 0, "val": val, "limit": LIMIT * val,
           "used": used, "broken": broken, "package": package, "log": projects.shown(log_path()), "stopped": None}
    rows, hours_done = [], []
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        log = Log(file)

        def handle(method, where, request):
            if where != "/predict":
                return 404, {"error": "no endpoint " + where}
            if method != "POST":
                return 405, {"error": "send a POST to /predict"}
            trouble = API.problem(request)
            if trouble != "":
                return 400, {"error": trouble}
            return 200, fns["serve"](answer_fn, request, log)

        try:
            server = API.start(handle)
        except OSError:
            run["stopped"] = "server"
            return run
        with server:
            for month in months:
                hours = HOURS_2012[HOURS_2012["mnth"] == month]
                before = len(log.lines())
                client = []
                for _, row in hours.iterrows():
                    req = request_of(row)
                    t0 = time.perf_counter()
                    try:
                        status, _ = API.send(server, "POST", "/predict", req)
                    except OSError:
                        status = 0
                    client.append((time.perf_counter() - t0) * 1000)
                    if status != 200:
                        run["errors"] += 1
                file.flush()
                lines = [_parsed(ln) for ln in log.lines()[before:]]
                if run["errors"] or len(lines) != len(hours) or not all(isinstance(ln, dict) for ln in lines):
                    run["stopped"] = "serve"
                    break
                for line in lines:
                    rows.append({**{f: line["request"][f] for f in FIELDS}, "bikes": line["bikes"]})
                hours_done.append(hours)
                run["requests"] += len(lines)
                run["serve_ms"][month] = [float(ln["ms"]) for ln in lines]
                run["client_ms"][month] = client
                log_table = pd.DataFrame(rows, columns=FIELDS + ["bikes"])
                done = pd.concat(hours_done).reset_index(drop=True)
                pred = torch.tensor(log_table["bikes"].to_numpy(dtype="float32"))
                run["rmse"][month] = TESTED.rmse_per(done, pred, "mnth")[month]["rmse"]
                for c in INPUTS + ["bikes"]:
                    try:
                        got = with_time_limit(lambda: fns["drift"](train.copy(), log_table.copy(), c), 10)
                        value = float(got[month])
                    except Exception:   # noqa: BLE001   yours stopped on the real log: the course's takes its place
                        fns["drift"], used["drift"] = _course_fn("drift"), "course"
                        broken.append("drift")
                        value = float(fns["drift"](train.copy(), log_table.copy(), c)[month])
                    run["drift"][c][month] = value
                then = train[train["mnth"] == month]["cnt"].to_numpy()
                run["rented"][month] = psi_of(then, hours["cnt"].to_numpy())
                try:
                    named = with_time_limit(lambda: fns["retrain"](done.copy(), pred.clone(), dict(config)), 10)
                    named = [int(m) for m in named]
                except Exception:   # noqa: BLE001
                    fns["retrain"], used["retrain"] = _course_fn("retrain"), "course"
                    broken.append("retrain")
                    named = [int(m) for m in fns["retrain"](done.copy(), pred.clone(), dict(config))]
                run["flagged"][month] = named
                run["months"].append(month)
                if show is not None:
                    show(month, run)
    return run


SUMMARY_TEXT = {
    "en": {"served": "Your service answered {n} requests of 2012 on 127.0.0.1, and wrote a line for each to `{log}`.",
           "latency": "Your `serve` measured the model's part of each answer: under {s99} ms for 99 in 100. The client waited up to {c99} ms for 99 in 100 requests, since its wait also holds the connection, the JSON and the server's thread.",
           "named": "Your `train_again` names {months}: the first as soon as its rentals were in.",
           "none_named": "Your `train_again` named no month.",
           "inputs": "The drift check of the inputs went above 0.25 in {months}; the answers stayed under 0.25 in every month.",
           "inputs_answers": "The drift check of the inputs went above 0.25 in {months}; the answers in {answers}.",
           "no_inputs": "The drift check of the inputs stayed under 0.25 in every month.",
           "package": {"yours": "", "missing": "Your project has no package in `ops/model` yet, so the service used the course's.",
                       "broken": "The course could not read the package in `ops/model`, so the service used the course's."},
           "used": "Your {name} does not do yet what its check asks, so the service used the course's version.",
           "missing": "Your project does not have {names} yet, so the service used the course's version.",
           "stopped": {"server": "The service could not start: no port on 127.0.0.1 was free. Close a program that uses many ports and press the button again.",
                       "serve": "The run stopped in {month}: your service did not answer every request with a line in the log. Does your `serve` still pass its check?"}},
    "nl": {"served": "Je service beantwoordde {n} verzoeken van 2012 op 127.0.0.1, en schreef voor elk verzoek een regel naar `{log}`.",
           "latency": "Je `serve` mat het deel van het model in elk antwoord: onder {s99} ms voor 99 van de 100. De client wachtte tot {c99} ms op 99 van de 100 verzoeken, want het wachten van de client bevat ook de verbinding, de JSON en de thread van de server.",
           "named": "Je `train_again` noemt {months}: de eerste zodra de verhuurde fietsen van de maand bekend waren.",
           "none_named": "Je `train_again` noemde geen enkele maand.",
           "inputs": "De driftcheck van de inputs kwam boven 0,25 in {months}; de antwoorden bleven in elke maand onder 0,25.",
           "inputs_answers": "De driftcheck van de inputs kwam boven 0,25 in {months}; de antwoorden in {answers}.",
           "no_inputs": "De driftcheck van de inputs bleef in elke maand onder 0,25.",
           "package": {"yours": "", "missing": "Je project heeft nog geen pakket in `ops/model`, dus gebruikte de service het pakket van de course.",
                       "broken": "De course kon het pakket in `ops/model` niet lezen, dus gebruikte de service het pakket van de course."},
           "used": "Je {name} doet nog niet wat de check vraagt, dus gebruikte de service de versie van de course.",
           "missing": "Je project heeft {names} nog niet, dus gebruikte de service de versie van de course.",
           "stopped": {"server": "De service kon niet starten: er was geen poort op 127.0.0.1 vrij. Sluit een programma dat veel poorten gebruikt en druk opnieuw op de knop.",
                       "serve": "De run stopte in {month}: je service beantwoordde niet elk verzoek met een regel in de log. Slaagt je `serve` nog voor de check?"}},
}
_PART_NAMES = {"serve": "`serve`", "drift": "`drift_by_month`", "retrain": "`train_again`"}


def summary(run: dict, lang: str) -> str:
    """The lines under the run: what the service answered and logged, how long it took, the months train_again names,
    the months of drift, and which parts were the course's."""
    lang = _lang(lang)
    t = SUMMARY_TEXT[lang]
    out = []
    if run.get("stopped") == "server":
        return t["stopped"]["server"]
    if run["months"]:
        out.append(t["served"].format(n=number(run["requests"], 0, lang), log=run["log"]))
        s = [v for m in run["months"] for v in run["serve_ms"][m]]
        c = [v for m in run["months"] for v in run["client_ms"][m]]
        out.append(t["latency"].format(s99=_ms_text(_q(s, 0.99), lang), c99=_ms_text(_q(c, 0.99), lang)))
        last = run["flagged"][run["months"][-1]]
        out.append(t["named"].format(months=month_list(last, lang)) if last else t["none_named"])
        inputs = [m for m in run["months"] if any(run["drift"][x].get(m, 0.0) > 0.25 for x in INPUTS)]
        answers = [m for m in run["months"] if run["drift"]["bikes"].get(m, 0.0) > 0.25]
        if not inputs:
            out.append(t["no_inputs"])
        elif answers:
            out.append(t["inputs_answers"].format(months=month_list(inputs, lang), answers=month_list(answers, lang)))
        else:
            out.append(t["inputs"].format(months=month_list(inputs, lang)))
    if run.get("stopped") == "serve":
        month = MONTH_NAMES[lang][len(run["months"])] if len(run["months"]) < 12 else ""
        out.append(t["stopped"]["serve"].format(month=month))
    if t["package"].get(run["package"]):
        out.append(t["package"][run["package"]])
    if run["broken"]:
        for k in sorted(set(run["broken"]), key=list(_PART_NAMES).index):
            out.append(t["used"].format(name=_PART_NAMES[k]))
    missing = [k for k, v in run["used"].items() if v == "course" and k not in run["broken"]]
    if missing:
        out.append(t["missing"].format(names=_list_of([_PART_NAMES[k] for k in missing], lang)))
    return " ".join(out)
