# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-project-tracked-model: the first step of the learner's project ops, a model of how many
bikes are rented in an hour. split_days puts every fifth day of 2011 in the validation set with all its hours;
train_version trains one version of the model as a tracked run, whose validation RMSE the learner writes, and keeps the
run as a line of JSON; pack puts the chosen version in a package: a safetensors file and a config.json with its version.
The lesson gives two files of the project, project/ops/demand.py (the hours, the inputs and the network) and
project/ops/safetensors_io.py (the format of the lesson *Packing a model*); the demo trains four versions with the
learner's parts and packs the best one.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011, from the Bike Sharing dataset of
Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the model, the runs and the pictures are the course's own."""
import ast
import copy
import hashlib
import html
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import torch

from everything_ai import chart, live, netview, paths, projects, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, same_value, with_time_limit

LESSON = "sys-project-tracked-model"
LANGS = ("en", "nl")
# The demo uses your three parts of this lesson: while one of them is missing or broken, the course's stands in.
projects.builds_on(("ops", "split", LESSON, "split"),
                   ("ops", "train", LESSON, "train"),
                   ("ops", "pack", LESSON, "pack"))
projects.provide(LESSON)   # ops/demand.py and ops/safetensors_io.py, which the lesson gives, also when only the checks load

_GIVEN = Path(__file__).parent / "project" / "ops"


def _given(name: str):
    """The course's own copy of a file the lesson gives: the checks' examples stay the same when you change yours."""
    spec = importlib.util.spec_from_file_location(f"_course_ops_{name}", _GIVEN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _given("demand")
tensors_io = _given("safetensors_io")

EPOCHS = 30
BATCH = 64
VERSIONS = [("h16-lr0.001", 16, 0.001), ("h16-lr0.01", 16, 0.01), ("h64-lr0.001", 64, 0.001), ("h64-lr0.01", 64, 0.01)]
WATCH_DAY = 194      # Thursday 14 July 2011, a validation day: the demo shows what each version expects for its hours
SATURDAY = 196       # Saturday 16 July 2011, beside it in the picture of the idea
LOW, HIGH = 40.0, 210.0   # the y-axis of the charts of the validation RMSE
DEMO_LIMIT = 180.0   # seconds the demo gets


def params(hidden: int, lr: float, epochs: int = EPOCHS, batch: int = BATCH, seed: int = 0) -> dict:
    return {"hidden": hidden, "lr": lr, "epochs": epochs, "batch": batch, "seed": seed}


# ======== the hours of 2011 ==============================================================================================

HOURS_2011 = demand.read_hours()                       # 8,645 hours of 365 days
_VAL = HOURS_2011["day"] % 5 == 4
TRAIN_HOURS, VAL_HOURS = HOURS_2011[~_VAL], HOURS_2011[_VAL]   # 6,912 hours of 292 days, 1,733 hours of 73 days
SCALE_2011 = demand.scaling(TRAIN_HOURS)


def tensors_of(train: pd.DataFrame, val: pd.DataFrame, scale: dict) -> tuple:
    return demand.inputs(train, scale), demand.targets(train), demand.inputs(val, scale), demand.targets(val)


# what the lesson says about the course's runs, measured with the course's parts (tests/test_sys_project_tracked_model.py)
RUN_CURVE = [146.87, 103.32, 87.53, 78.21, 73.58, 71.43, 68.95, 66.83, 65.84, 64.18, 63.45, 61.38, 60.34, 59.42, 58.56,
             58.65, 57.56, 56.58, 56.42, 55.77, 55.4, 55.2, 54.44, 54.24, 54.19, 54.07, 53.68, 53.61, 53.26, 53.04]
NOISY_CURVE = [102.16, 79.57, 71.32, 67.94, 65.03, 62.77, 61.03, 58.96, 57.15, 57.77, 54.95, 54.97, 53.88, 54.25, 53.17,
               53.75, 52.74, 52.8, 52.39, 53.29, 52.14, 52.7, 52.89, 52.74, 52.67, 52.47, 52.7, 51.67, 51.51, 53.89]


# ======== drawing =========================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f'<code style="white-space:nowrap">{part}</code>')
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _mono(x: float, y: float, text_: str, size: float = 10.5, fill: str = "var(--ea-ink)", anchor: str = "start",
          weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-family:var(--ea-font-code);'
            f'font-variant-ligatures:none;font-size:{size}px;font-weight:{weight};fill:{fill}">{html.escape(text_)}</text>')


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "", opacity: float = 1.0) -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
            f'style="fill:{fill};fill-opacity:{opacity:.2f}{edge}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _short(text_: str, n: int) -> str:
    return text_ if len(text_) <= n else text_[:n - 1] + "…"


def _chart(panels: list, alt: str, lang: str, width: int = 330, height: int = 230) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


# ======== the picture of the idea: two days of bikes ======================================================================

def day_counts(day: int) -> list:
    """The bikes rented in each hour of one day of 2011, 0 where the table has no row for that hour."""
    rows = HOURS_2011[HOURS_2011["day"] == day]
    counts = [0] * 24
    for hr, cnt in zip(rows["hr"], rows["cnt"]):
        counts[int(hr)] = int(cnt)
    return counts


IDEA_TEXT = {
    "en": {"title": "Bikes per hour, 14 and 16 July 2011", "x": "hour of the day", "y": "bikes",
           "thursday": "Thursday 14 July, a working day", "saturday": "Saturday 16 July",
           "alt": "Two lines of the bikes rented in each hour. On Thursday 14 July the line has two peaks, about 490 bikes at 8 in the morning and about 600 at 5 in the afternoon, with 120 to 230 from 9 in the morning to 3 in the afternoon. On Saturday 16 July it rises slowly to one wide hill of 400 to 500 bikes from noon to 7 in the evening."},
    "nl": {"title": "Fietsen per uur, 14 en 16 juli 2011", "x": "uur van de dag", "y": "fietsen",
           "thursday": "donderdag 14 juli, een werkdag", "saturday": "zaterdag 16 juli",
           "alt": "Twee lijnen van de gehuurde fietsen in elk uur. Op donderdag 14 juli heeft de lijn twee pieken, ongeveer 490 fietsen om 8 uur 's ochtends en ongeveer 600 om 5 uur 's middags, met 120 tot 230 van 9 uur 's ochtends tot 3 uur 's middags. Op zaterdag 16 juli stijgt de lijn langzaam naar één brede heuvel van 400 tot 500 fietsen van 12 uur 's middags tot 7 uur 's avonds."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = IDEA_TEXT[lang]
    thursday, saturday = day_counts(WATCH_DAY), day_counts(SATURDAY)
    top = 100 * math.ceil(max(thursday + saturday) / 100)
    p = chart.Panel((0, 23), (0, top), title=w["title"], xlabel=w["x"], ylabel=w["y"], xticks=[0, 6, 12, 18, 23])
    hours = list(range(24))
    p.line(hours, saturday, "data", 1.4, dashed=True)
    p.line(hours, thursday, "data", 2.4)
    return _chart([p], w["alt"], lang) + live.key_html([("data", False, w["thursday"]), ("data", True, w["saturday"])])


# ======== exercise 1: split_days ===========================================================================================

class _Table:
    """A table the check calls by the name the lesson gives it, such as HOURS, so that a check message stays short;
    the learner's code gets its own copy of the DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str):
        self.df, self.name = df, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.df.copy()


def _small_table(days, hours=(8, 17), busy=None) -> pd.DataFrame:
    rows = []
    for d in days:
        for h in hours:
            rows.append({"instant": len(rows) + 1, "day": d, "hr": h, "cnt": 1200 if (d, h) == busy else 40 + 3 * d + h})
    return pd.DataFrame(rows)


HOURS = _Table(HOURS_2011, "HOURS")
TEN = _Table(_small_table(range(10), busy=(4, 8)), "TEN")                     # 10 days of 2 hours, one hour of 1,200 bikes
LATE = _Table(_small_table(range(355, 365), hours=(0, 12, 23)), "LATE")      # the last 10 days of the year
GAPS = _Table(_small_table([d for d in range(15) if d not in (3, 4, 10)]), "GAPS")   # days without rows, day 4 among them

_S_HEAD = "def split_days(hours):\n"
_S_TAIL = "    return hours[~val_day], hours[val_day]\n"


def _s(rule: str) -> str:
    return _S_HEAD + f"    val_day = {rule}\n" + _S_TAIL


_S_START = _s("...")
_S = _s('hours["day"] % 5 == 4')
_S_HOUR = _s('hours["hr"] % 5 == 4')
_S_ZERO = _s('hours["day"] % 5 == 0')
_S_SWAPPED = _s('hours["day"] % 5 != 4')
_S_FIRST = _s('hours["day"] < 73')


def _hours_said(n: int, lang: str) -> str:
    if lang == "nl":
        return f"{number(n, 0, 'nl')} uur"
    return "1 hour" if n == 1 else f"{number(n, 0, 'en')} hours"


def _split_shown(v):
    """A split in a check message: how many hours land on each side, and the first validation days."""
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(t, pd.DataFrame) for t in v):
        train, val = v
        days = sorted({int(d) for d in val["day"]}) if "day" in val.columns else []
        listed = ", ".join(str(d) for d in days[:4]) + (", …" if len(days) > 4 else "")
        out = {}
        for lang in LANGS:
            tail = ("" if not days else (f" of the days {listed}" if lang == "en" else f" van de dagen {listed}"))
            out[lang] = (f"a training set of {_hours_said(len(train), 'en')} and a validation set of {_hours_said(len(val), 'en')}{tail}"
                         if lang == "en" else
                         f"een training set van {_hours_said(len(train), 'nl')} en een validation set van {_hours_said(len(val), 'nl')}{tail}")
        return out
    return {lang: f"`{_short(repr(v), 60)}`" for lang in LANGS}


def _days_on_both_sides(got) -> bool:
    try:
        train, val = got
        return bool(set(train["day"]) & set(val["day"]))
    except Exception:   # noqa: BLE001   not two tables with a day column: another message says what is wrong
        return False


# ======== exercise 2: train_version ========================================================================================

class _Data(tuple):
    """The four tensors of a split, shown by name in a check message; the learner's code gets copies."""

    def __new__(cls, items, name: str):
        obj = super().__new__(cls, items)
        obj.name = name
        return obj

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return tuple(t.clone() for t in self)


_NOW: dict = {}   # the file and the folder of the call the check makes now, for what it compares after the call


def _check_folder() -> Path:
    folder = paths.WORK / "check" / LESSON
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class _Log:
    """The file of runs a call gets: written again before every call, empty or with one earlier run in it."""

    def __init__(self, name: str, lines: list):
        self.name, self.lines = name, lines

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        path = _check_folder() / "runs.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in self.lines), encoding="utf-8")
        _NOW["log"], _NOW["old"] = path, bool(self.lines)
        return path


EARLIER = {"name": "earlier", "params": params(8, 0.01, epochs=1), "metrics": {"val_rmse": [[1, 99.0]]}}
DATA = _Data(tensors_of(TRAIN_HOURS, VAL_HOURS, SCALE_2011), "DATA")
_FEW_TRAIN, _FEW_VAL = TRAIN_HOURS.iloc[::40], VAL_HOURS.iloc[::30]   # 173 and 58 hours, for quick cases
FEW = _Data(tensors_of(_FEW_TRAIN, _FEW_VAL, demand.scaling(_FEW_TRAIN)), "FEW")
LOG = _Log("LOG", [])
OLD_LOG = _Log("OLD_LOG", [EARLIER])

_T_HEAD = ("import json\n"
           "\n"
           "import torch\n"
           "from ops.demand import DemandNet, train_epoch\n"
           "\n\n"
           "def train_version(name, params, data, log_path, show=None):\n"
           "    x, y, x_val, y_val = data\n"
           "    torch.manual_seed(params[\"seed\"])\n"
           "    model = DemandNet(params[\"hidden\"])\n"
           "    opt = torch.optim.Adam(model.parameters(), lr=params[\"lr\"])\n"
           "    run = {\"name\": name, \"params\": params, \"metrics\": {\"val_rmse\": []}}\n"
           "    for epoch in range(1, params[\"epochs\"] + 1):\n"
           "        train_epoch(model, opt, x, y, params[\"batch\"])\n"
           "        with torch.no_grad():\n")


def _tv(rmse: str, line: str) -> str:
    return (_T_HEAD + f"            rmse = {rmse}\n"
            "        run[\"metrics\"][\"val_rmse\"].append([epoch, float(rmse)])\n"
            "        if show is not None:\n"
            "            show(epoch, model, run)\n"
            "    with open(log_path, \"a\") as f:\n"
            f"        {line}\n"
            "    return model, run\n")


_T_RMSE = "((model(x_val) - y_val) ** 2).mean().sqrt()"
_T_LINE = "f.write(json.dumps(run) + \"\\n\")"
_T_START = _tv("...", "...")
_T = _tv(_T_RMSE, _T_LINE)
_T_MSE = _tv("((model(x_val) - y_val) ** 2).mean()", _T_LINE)
_T_TRAIN = _tv("((model(x) - y) ** 2).mean().sqrt()", _T_LINE)
_T_MAE = _tv("(model(x_val) - y_val).abs().mean()", _T_LINE)
_T_TOTAL = _tv("((model(x_val) - y_val) ** 2).sum().sqrt()", _T_LINE)
_T_COLUMN = _tv("((model(x_val).reshape(-1) - y_val) ** 2).mean().sqrt()", _T_LINE)
_T_COUNT = _tv("(((model(x_val) - y_val) ** 2).sum() / 1733).sqrt()", _T_LINE)


def _parsed(line: str):
    try:
        return json.loads(line)
    except ValueError:
        return {"not JSON": _short(line, 60)}


def _trained(value) -> dict:
    """What the check compares for train_version: the run it gives back, every line of the file after the call, read
    as JSON, and whether the file ends with a line break, so that the next run starts on a line of its own."""
    out = {}
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        out["run"] = value[1]
    else:
        out["returned"] = _short(repr(value), 80)
    path = _NOW.get("log")
    text_ = path.read_text(encoding="utf-8") if path is not None and path.exists() else ""
    out["log"] = [_parsed(line) for line in text_.splitlines()]
    out["newline"] = text_.endswith("\n") or not text_
    return out


def _rounded(points) -> str:
    try:
        return json.dumps([[int(e), round(float(v), 2)] for e, v in points])
    except Exception:   # noqa: BLE001   not a list of pairs: shown as Python shows it
        return _short(repr(points), 60)


def _train_shown(v):
    if isinstance(v, dict) and "log" in v:
        lines = len(v["log"])
        if "run" in v:
            try:
                metric = _rounded(v["run"]["metrics"]["val_rmse"])
            except Exception:   # noqa: BLE001   a run without that list
                metric = None
            if metric is not None:
                return {"en": f"a run with `val_rmse` `{metric}` and {lines} line{'' if lines == 1 else 's'} in the file",
                        "nl": f"een run met `val_rmse` `{metric}` en {lines} regel{'' if lines == 1 else 's'} in het bestand"}
        return {"en": f"{lines} line{'' if lines == 1 else 's'} in the file", "nl": f"{lines} regel{'' if lines == 1 else 's'} in het bestand"}
    return {lang: f"`{_short(repr(v), 60)}`" for lang in LANGS}


def _not_json(got) -> bool:
    return isinstance(got, dict) and any(isinstance(line, dict) and "not JSON" in line for line in got.get("log", []))


def _lost_earlier(got) -> bool:
    return (isinstance(got, dict) and _NOW.get("old", False)
            and not any(isinstance(line, dict) and line.get("name") == "earlier" for line in got.get("log", [])))


def _close(a, b) -> bool:
    """float32 keeps about 7 digits: another right way to work out the RMSE may differ in the last of them."""
    return same_value(a, b, rel=1e-4)


# ======== exercise 3: pack ===================================================================================================

class _Model:
    """A network the check calls by its name, such as MODEL; the learner's code gets a copy."""

    def __init__(self, model, name: str):
        self.model, self.name = model, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.model)


class _Named(dict):
    """A dict the check calls by its name, such as RUN; the learner's code gets a plain copy."""

    def __init__(self, value: dict, name: str):
        super().__init__(value)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self))


class _Folder:
    """An empty folder for the package, made again before every call."""

    def __repr__(self):
        return "FOLDER"

    def __deepcopy__(self, memo):
        path = _check_folder() / "package"
        if path.exists():
            shutil.rmtree(path)
        _NOW["folder"] = path
        return path


def _network(hidden: int, seed: int):
    torch.manual_seed(seed)
    return demand.DemandNet(hidden)


def _run_of(name: str, hidden: int, lr: float, curve: list) -> dict:
    return {"name": name, "params": params(hidden, lr, epochs=len(curve)),
            "metrics": {"val_rmse": [[k + 1, v] for k, v in enumerate(curve)]}}


MODEL = _Model(_network(16, 0), "MODEL")                     # a DemandNet(16) as it starts: the check needs no training
RUN = _Named(_run_of("h16-lr0.01", 16, 0.01, RUN_CURVE), "RUN")
SCALE = _Named(SCALE_2011, "SCALE")
FOLDER = _Folder()
SMALL_MODEL = _Model(_network(4, 1), "SMALL_MODEL")
SHORT = _Named(_run_of("h4-lr0.05", 4, 0.05, [171.4, 149.2, 151.0]), "SHORT")   # 3 epochs, the lowest not at the end
WIDE_MODEL = _Model(_network(64, 2), "WIDE_MODEL")
NOISY = _Named(_run_of("h64-lr0.01", 64, 0.01, NOISY_CURVE), "NOISY")        # lowest at epoch 29, not at 30
SMALL_SCALE = _Named({"mean": [0.0] * 10, "std": [1.0] * 10}, "SMALL_SCALE")

_P_HEAD = ("import hashlib\n"
           "import json\n"
           "\n"
           "from ops.demand import FEATURES\n"
           "from ops.safetensors_io import save_file\n"
           "\n\n"
           "def pack(model, run, scale, folder):\n"
           "    folder.mkdir(parents=True, exist_ok=True)\n"
           "    weights = folder / \"model.safetensors\"\n"
           "    save_file(model.state_dict(), weights)\n"
           "    with open(weights, \"rb\") as f:\n")


def _p(version: str, rmse: str) -> str:
    return (_P_HEAD + f"        version = {version}\n"
            "    config = {\n"
            "        \"class\": \"DemandNet\",\n"
            "        \"hidden\": run[\"params\"][\"hidden\"],\n"
            "        \"inputs\": \"hr as 24 one-hot columns, then each feature minus mean, divided by std\",\n"
            "        \"features\": FEATURES,\n"
            "        \"mean\": scale[\"mean\"],\n"
            "        \"std\": scale[\"std\"],\n"
            "        \"output\": \"bikes rented in the hour\",\n"
            "        \"run\": run[\"name\"],\n"
            f"        \"val_rmse\": {rmse},\n"
            "        \"version\": version,\n"
            "    }\n"
            "    with open(folder / \"config.json\", \"w\") as f:\n"
            "        json.dump(config, f, indent=2)\n"
            "    return config\n")


_P_VERSION = "hashlib.sha256(f.read()).hexdigest()[:12]"
_P_RMSE = 'run["metrics"]["val_rmse"][-1][1]'
_P_START = _p("...", "...")
_P = _p(_P_VERSION, _P_RMSE)
_P_FULL = _p("hashlib.sha256(f.read()).hexdigest()", _P_RMSE)
_P_MD5 = _p("hashlib.md5(f.read()).hexdigest()[:12]", _P_RMSE)
_P_NAME = _p("hashlib.sha256(str(weights).encode()).hexdigest()[:12]", _P_RMSE)
_P_STATE = _p("hashlib.sha256(str(model.state_dict()).encode()).hexdigest()[:12]", _P_RMSE)
_P_LAST = _p("hashlib.sha256(f.read()).hexdigest()[-12:]", _P_RMSE)
_P_BUILTIN = _p("str(hash(f.read()))[:12]", _P_RMSE)
_P_LOWEST = _p(_P_VERSION, 'min(v for e, v in run["metrics"]["val_rmse"])')
_P_FIRST = _p(_P_VERSION, 'run["metrics"]["val_rmse"][0][1]')
_P_PAIR = _p(_P_VERSION, 'run["metrics"]["val_rmse"][-1]')
_P_EPOCH = _p(_P_VERSION, 'run["metrics"]["val_rmse"][-1][0]')
_P_FIXED = _p(_P_VERSION, 'run["metrics"]["val_rmse"][29][1]')


def _packed(value) -> dict:
    """What the check compares for pack: the config it gives back, the files in the folder, and config.json as read back."""
    folder = _NOW.get("folder")
    files = sorted(p.name for p in folder.iterdir()) if folder is not None and folder.is_dir() else []
    try:
        written = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001   no file, or no JSON in it
        written = None
    return {"returned": value if isinstance(value, dict) else _short(repr(value), 80), "files": files, "config.json": written}


def _pack_shown(v):
    if isinstance(v, dict) and isinstance(v.get("returned"), dict):
        cfg = v["returned"]
        say = f"`\"version\": {_short(json.dumps(cfg.get('version'), default=repr), 70)}` " \
              f"`\"val_rmse\": {_short(json.dumps(cfg.get('val_rmse'), default=repr), 30)}`"
        return {"en": "a config with " + say.replace("` `", "` and `"), "nl": "een config met " + say.replace("` `", "` en `")}
    return {lang: f"`{_short(repr(v), 60)}`" for lang in LANGS}


# ======== the pictures of the learner's work ================================================================================

PICTURE_SECONDS = 4.0
_STOPPED = object()


def _fn(seen, name: str):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(name)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_YOURS_TEXT = {
    "en": {
        "split": {"title": "Your validation days in 2011",
                  "intro": "The picture calls your `split_days(HOURS)` and draws every day of 2011 as a square, 25 days to a row: 1 January at the top left, 31 December in the bottom row.",
                  "val": "all hours in validation", "train": "all hours in training", "both": "hours on both sides",
                  "fail": "Your `split_days` stops on `HOURS`, or gives no two tables, so the picture has no days to draw yet. The check above says why.",
                  "count": "Validation: {v} days with all their hours. Training: {t} days. Days with hours on both sides: {b}.",
                  "right": "The validation days form five columns: every fifth day, spread over the whole year, with all the hours of each day on one side.",
                  "alt": "The 365 days of 2011 as squares in rows of 25 days, colored by where their hours land in your split."},
        "train": {"title": "Your run, epoch by epoch",
                  "intro": "The picture calls your `train_version` with 10 epochs, 16 neurons and learning rate 0.01, on `DATA` and a file that already holds one earlier run, and draws the validation RMSE your run logs after each epoch, beside the course's run.",
                  "x": "epoch", "y": "validation RMSE", "yours": "your run", "course": "the course's run",
                  "file": "The file after the call:", "json": "a run as JSON", "not": "not JSON",
                  "fail": "Your `train_version` stops, or gives no run with a list `val_rmse`, so the picture has no curve to draw yet. The check above says why.",
                  "alt": "The validation RMSE of your run after each of 10 epochs, and the course's run as a dashed line, with the lines of the file of runs under it."},
        "pack": {"title": "Your package",
                 "intro": "The picture calls your `pack(MODEL, RUN, SCALE, ...)` with a folder of its own, and shows what is in the folder afterwards.",
                 "bytes": "{n} bytes", "match": "matches the bytes of model.safetensors", "nomatch": "is not the version of model.safetensors",
                 "last": "the RMSE of the last epoch", "notlast": "not the RMSE of the last epoch",
                 "missing": "missing",
                 "fail": "Your `pack` stops, or writes no config.json, so the picture has no package to show yet. The check above says why.",
                 "alt": "The folder of your package: model.safetensors with its size, and config.json with its settings, its run, its val_rmse and its version."},
    },
    "nl": {
        "split": {"title": "Je validatiedagen in 2011",
                  "intro": "De afbeelding roept je `split_days(HOURS)` aan en tekent elke dag van 2011 als een vierkantje, 25 dagen per rij: 1 januari linksboven, 31 december in de onderste rij.",
                  "val": "alle uren in validatie", "train": "alle uren in training", "both": "uren aan beide kanten",
                  "fail": "Je `split_days` stopt bij `HOURS`, of geeft geen twee tabellen, dus valt er nog geen dag te tekenen. De check hierboven zegt waarom.",
                  "count": "Validatie: {v} dagen met alle uren. Training: {t} dagen. Dagen met uren aan beide kanten: {b}.",
                  "right": "De validatiedagen vormen vijf kolommen: elke vijfde dag, verspreid over het hele jaar, met alle uren van elke dag aan één kant.",
                  "alt": "De 365 dagen van 2011 als vierkantjes in rijen van 25 dagen, gekleurd naar waar de uren van elke dag in je splitsing terechtkomen."},
        "train": {"title": "Je run, epoch na epoch",
                  "intro": "De afbeelding roept je `train_version` aan met 10 epochs, 16 neuronen en learning rate 0,01, op `DATA` en een bestand waarin al één eerdere run staat, en tekent de validation RMSE die je run na elke epoch bijhoudt, naast de run van de course.",
                  "x": "epoch", "y": "validation RMSE", "yours": "jouw run", "course": "run van de course",
                  "file": "Het bestand na de aanroep:", "json": "een run als JSON", "not": "geen JSON",
                  "fail": "Je `train_version` stopt, of geeft geen run met een lijst `val_rmse`, dus valt er nog geen curve te tekenen. De check hierboven zegt waarom.",
                  "alt": "De validation RMSE van je run na elk van 10 epochs, en de run van de course als lijn van streepjes, met de regels van het bestand met runs eronder."},
        "pack": {"title": "Je pakket",
                 "intro": "De afbeelding roept je `pack(MODEL, RUN, SCALE, ...)` aan met een eigen map, en laat zien wat er daarna in de map staat.",
                 "bytes": "{n} bytes", "match": "klopt met de bytes van model.safetensors", "nomatch": "is niet de versie van model.safetensors",
                 "last": "de RMSE van de laatste epoch", "notlast": "niet de RMSE van de laatste epoch",
                 "missing": "ontbreekt",
                 "fail": "Je `pack` stopt, of schrijft geen config.json, dus valt er nog geen pakket te laten zien. De check hierboven zegt waarom.",
                 "alt": "De map van je pakket: model.safetensors met de grootte, en config.json met de instellingen, de run, val_rmse en de versie."},
    },
}


def _words(seen, key: str) -> dict:
    return _YOURS_TEXT[seen.lang if seen.lang in LANGS else "en"][key]


def calendar_svg(val_share: dict, alt: str, w: dict) -> str:
    """365 squares, 25 days to a row: ink for a day whose hours are all in the validation set, a light square for
    one whose hours are all in training, purple for a day with hours on both sides. The validation days of the rule,
    every fifth day, then form five columns."""
    cell, gap = 12.0, 1.0
    parts = []
    for d in range(365):
        x, y = (d % 25) * (cell + gap), (d // 25) * (cell + gap) + 8
        share = val_share.get(d)
        if share is None:
            continue
        if share >= 1:
            parts.append(_rect(x, y, cell, cell, "var(--ea-ink)"))
        elif share <= 0:
            parts.append(_rect(x, y, cell, cell, "var(--ea-line)"))
        else:
            parts.append(_rect(x, y, cell, cell, "var(--ea-loss)"))
    y0 = 15 * (cell + gap) + 24
    for k, (fill, label) in enumerate((("var(--ea-ink)", w["val"]), ("var(--ea-line)", w["train"]), ("var(--ea-loss)", w["both"]))):
        x = [0, 112, 224][k]
        parts.append(_rect(x, y0 - 8, 8, 8, fill))
        parts.append(_t(x + 12, y0, label, 9, 400, "var(--ea-ink-2)"))
    return _svg(y0 + 6, alt, parts)


def split_picture(seen) -> str:
    w = _words(seen, "split")
    lang = seen.lang if seen.lang in LANGS else "en"
    fn = _fn(seen, "split_days")
    got = _call(lambda: fn(HOURS_2011.copy())) if fn is not None else _STOPPED
    ok = (got is not _STOPPED and isinstance(got, tuple) and len(got) == 2
          and all(isinstance(t, pd.DataFrame) and "day" in t.columns for t in got))
    if not ok:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    train, val = got
    per_val = val.groupby("day").size()
    per_train = train.groupby("day").size()
    share = {}
    for d in range(365):
        v, t = int(per_val.get(d, 0)), int(per_train.get(d, 0))
        if v + t:
            share[d] = v / (v + t)
    vals = sum(1 for s in share.values() if s >= 1)
    trains = sum(1 for s in share.values() if s <= 0)
    both = len(share) - vals - trains
    right = both == 0 and {d for d, s in share.items() if s >= 1} == {d for d in range(4, 365, 5)}
    say = w["count"].format(v=vals, t=trains, b=both)
    note = _note(say + (" " + w["right"] if right else ""), _UI if both == 0 else _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + calendar_svg(share, w["alt"], w) + note, alt=w["alt"])


PICTURE_PARAMS = params(16, 0.01, epochs=10)


def train_picture(seen) -> str:
    w = _words(seen, "train")
    lang = seen.lang if seen.lang in LANGS else "en"
    fn = _fn(seen, "train_version")

    def make_():
        log = copy.deepcopy(OLD_LOG)
        out = fn("picture", dict(PICTURE_PARAMS), copy.deepcopy(DATA), log)
        return out, log.read_text(encoding="utf-8")

    got = _call(make_) if fn is not None else _STOPPED
    points = None
    if got is not _STOPPED:
        try:
            points = [(int(e), float(v)) for e, v in got[0][1]["metrics"]["val_rmse"]]
        except Exception:   # noqa: BLE001   no run with that list
            points = None
    if not points:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    course = list(enumerate(RUN_CURVE[:10], start=1))
    every = [v for _, v in points + course if math.isfinite(v)]
    top = max(200.0, min(max(every) * 1.05, 5000.0)) if every else 200.0
    p = chart.Panel((1, 10), (0, top), title="", xlabel=w["x"], ylabel=w["y"], xticks=list(range(1, 11)))
    p.line([e for e, _ in course], [v for _, v in course], "ink-3", 1.4, dashed=True)
    shown = [(e, min(v, top)) for e, v in points if math.isfinite(v)]
    p.line([e for e, _ in shown], [v for _, v in shown], "loss", 2.4)
    p.points([e for e, _ in shown], [v for _, v in shown], "loss", r=2.5)
    curve = _chart([p], w["alt"], lang, height=200) + live.key_html([("loss", False, w["yours"]), ("ink-3", True, w["course"])])
    lines = got[1].splitlines()
    rows = [_t(0, 12, w["file"], 9.5, 600, "var(--ea-ink-2)")]
    for k, line in enumerate(lines[:4]):
        parsed = _parsed(line)
        good = "not JSON" not in parsed
        y = 28 + 15 * k
        rows.append(_mono(0, y, _short(line, 40), 9, "var(--ea-ink)" if good else "var(--ea-loss-text)"))
        rows.append(_t(330, y, w["json"] if good else w["not"], 9, 400, "var(--ea-ink-2)" if good else "var(--ea-loss-text)", "end"))
    file_svg = _svg(28 + 15 * min(len(lines), 4), w["alt"], rows)
    return yours.figure(w["title"], _note(w["intro"]) + curve + file_svg, alt=w["alt"])


def package_svg(folder: Path, run: dict, w: dict, lang: str, alt: str) -> str:
    """The folder of a package: model.safetensors with its size, then the lines of config.json, with the version and
    val_rmse held against the file and the run."""
    weights, cfg_file = folder / "model.safetensors", folder / "config.json"
    parts = [_mono(0, 13, folder.name + "/", 10.5, weight=600)]
    size = weights.stat().st_size if weights.exists() else None
    parts.append(_mono(14, 31, "model.safetensors", 10, weight=600))
    parts.append(_t(330, 31, w["bytes"].format(n=number(size, 0, lang)) if size is not None else w["missing"], 9.5, 400,
                    "var(--ea-ink-2)" if size is not None else "var(--ea-loss-text)", "end"))
    parts.append(_mono(14, 49, "config.json", 10, weight=600))
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001   no config to show
        cfg = None
    y = 49
    if not isinstance(cfg, dict):
        parts.append(_t(330, y, w["missing"], 9.5, 400, "var(--ea-loss-text)", "end"))
        return _svg(y + 8, alt, parts)
    real = hashlib.sha256(weights.read_bytes()).hexdigest()[:12] if weights.exists() else None
    try:
        last = float(run["metrics"]["val_rmse"][-1][1])
    except Exception:   # noqa: BLE001
        last = None
    for key, value in cfg.items():
        y += 14
        shown = json.dumps(value, default=repr)
        if isinstance(value, list) and len(value) > 3:
            shown = json.dumps(value[:2], default=repr)[:-1] + ", …]"
        ink, said = "var(--ea-ink-2)", ""
        if key == "version":
            good = value == real
            ink, said = ("var(--ea-ink)", w["match"]) if good else ("var(--ea-loss-text)", w["nomatch"])
        elif key == "val_rmse":
            good = isinstance(value, (int, float)) and last is not None and abs(float(value) - last) < 1e-9
            ink, said = ("var(--ea-ink)", w["last"]) if good else ("var(--ea-loss-text)", w["notlast"])
        parts.append(_mono(24, y, _short(f'"{key}": {shown}', 44 if said else 54), 9, ink, weight=600 if said else 400))
        if said:
            y += 12
            parts.append(_t(36, y, said, 8.5, 400, ink))
    return _svg(y + 8, alt, parts)


def pack_picture(seen) -> str:
    w = _words(seen, "pack")
    lang = seen.lang if seen.lang in LANGS else "en"
    fn = _fn(seen, "pack")
    folder = _check_folder() / "picture" / "model"

    def make_():
        if folder.exists():
            shutil.rmtree(folder)
        return fn(copy.deepcopy(MODEL), copy.deepcopy(RUN), copy.deepcopy(SCALE), folder)

    got = _call(make_) if fn is not None else _STOPPED
    if got is _STOPPED or not (folder / "config.json").exists():
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    return yours.figure(w["title"], _note(w["intro"]) + package_svg(folder, dict(RUN), w, lang, w["alt"]), alt=w["alt"])


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return "..." in text_
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


EXERCISES = {
    "split": FunctionBox(
        name="split",
        function="split_days",
        start=_S_START,
        solution=_S,
        save_as=("ops", "split"),
        visible=[(HOURS,)],
        hidden=[(TEN,), (LATE,), (GAPS,)],
        first=[_DOTS],
        picture=split_picture,
        show=_split_shown,
        mistakes=[
            Mistake("hour", wrong=_S_HOUR,
                    en="`{call}` gives {got}: your mask picks the hours 4, 9, 14 and 19 of every day, since `hours[\"hr\"]` is the hour of the day. Every day then has hours on both sides, and the model has seen most hours of every day, with their weather. Which column says the day of an hour?",
                    nl="`{call}` geeft {got}: je mask kiest de uren 4, 9, 14 en 19 van elke dag, want `hours[\"hr\"]` is het uur van de dag. Elke dag heeft dan uren aan beide kanten, en het model heeft de meeste uren van elke dag al gezien, met het weer erbij. Welke kolom zegt op welke dag een uur valt?"),
            Mistake("zero", wrong=_S_ZERO,
                    en="`{call}` gives {got}: your validation days are 0, 5, 10 and so on. The rule asks for the days 4, 9, 14 and so on. What remainder do those days leave when you divide them by 5?",
                    nl="`{call}` geeft {got}: je validatiedagen zijn 0, 5, 10 enzovoort. De regel vraagt om de dagen 4, 9, 14 enzovoort. Welke rest geven deze dagen bij delen door 5?"),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en="`{call}` gives {got}: the first table holds the validation days, and the second all the other days. The `return` line gives `hours[~val_day]` first, so `val_day` must be True for the validation days. Is your mask True for the days the rule names, and is the `return` line still as it was given?",
                    nl="`{call}` geeft {got}: de eerste tabel bevat de validatiedagen, en de tweede alle andere dagen. De `return`-regel geeft eerst `hours[~val_day]`, dus moet `val_day` True zijn voor de validatiedagen. Is je mask True voor de dagen die de regel noemt, en staat de `return`-regel er nog zoals gegeven?"),
            Mistake("first", wrong=_S_FIRST,
                    en="`{call}` gives {got}: your validation set is the first 73 days, from January to the middle of March, the coldest and quietest weeks of the year. It would measure every version on winter only. How do you spread 73 days over the whole year?",
                    nl="`{call}` geeft {got}: je validation set bestaat uit de eerste 73 dagen, van januari tot half maart, de koudste en rustigste weken van het jaar. Elke versie zou dan alleen op de winter gemeten worden. Hoe verspreid je 73 dagen over het hele jaar?"),
            Mistake("split-days", matches=lambda got, text_: _days_on_both_sides(got),
                    en="`{call}` gives {got}, and some days have hours on both sides. The hours of one day share their weather and the busy times of that day: a model that learned from most of them has an easy job on the rest, and a new day is never like that. Which column keeps all the hours of a day together?",
                    nl="`{call}` geeft {got}, en sommige dagen hebben uren aan beide kanten. De uren van één dag delen het weer en de drukke momenten: leerde een model van de meeste uren, dan heeft het model het makkelijk met de rest, en een nieuwe dag is nooit zo. Welke kolom houdt alle uren van een dag bij elkaar?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a mask that is True for every hour of the days 4, 9, 14 and so on, up to 364.",
                      "nl": "Vervang de `...` door een mask dat True is voor elk uur van de dagen 4, 9, 14 enzovoort, tot en met 364."},
            "arguments": {"en": "The check calls `{call}`, with one table, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(hours)`?",
                          "nl": "De check roept `{call}` aan, met één tabel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(hours)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `val_day` True for every hour of the days 4, 9, 14 and so on, and for no other hour?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `val_day` True voor elk uur van de dagen 4, 9, 14 enzovoort, en voor geen ander uur?"},
            "almost": {"en": "Right for `HOURS`. But `{call}` gives {got}, and it should give {want}. Does your mask look at the day of each hour, and at nothing else?",
                       "nl": "Goed voor `HOURS`. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je mask naar de dag van elk uur, en naar niets anders?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` line?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met de `return`-regel?"},
            "correct": {"en": "Right in all {n} cases: 73 days with all their 1,733 hours for the validation set, and the other 292 days, 6,912 hours, to learn from. Your `split_days` is now in your project, in `{saved}`. How will you measure a version on these 73 days?",
                        "nl": "Goed in alle {n} gevallen: 73 dagen met alle 1.733 uur voor de validation set, en de andere 292 dagen, 6.912 uur, om van te leren. Je `split_days` staat nu in je project, in `{saved}`. Hoe meet je een versie op deze 73 dagen?"},
            "kept": _kept("split_days"),
        },
    ),
    "train": FunctionBox(
        name="train",
        function="train_version",
        start=_T_START,
        solution=_T,
        save_as=("ops", "train"),
        visible=[("check", params(16, 0.01, epochs=3), DATA, LOG)],
        hidden=[("few", params(4, 0.05, epochs=2, batch=32, seed=1), FEW, OLD_LOG),
                ("once", params(8, 0.01, epochs=1, batch=50, seed=2), FEW, LOG)],
        first=[_DOTS],
        collect=_trained,
        compare=_close,
        show=_train_shown,
        picture=train_picture,
        time_limit=20.0,
        mistakes=[
            Mistake("model-in-line", error=lambda e: isinstance(e, TypeError) and "DemandNet" in str(e),
                    en="With `{call}`, your function stops with an error: `json.dumps` cannot write a network as JSON. The line holds the run, a dict of names, settings and numbers. Which of the two does your line give it?",
                    nl="Met `{call}` stopt je functie met een fout: `json.dumps` kan een netwerk niet als JSON schrijven. De regel bevat de run, een dict met namen, instellingen en getallen. Welke van de twee geeft je regel mee?"),
            Mistake("pretty", matches=lambda got, text_: "indent" in text_,
                    en="With `{call}`, your run takes several lines of the file: `indent` spreads the JSON over lines. In JSON Lines, every run is one line, so that a program can read the file line by line, and a half-written last line costs one run. Can you write the run without `indent`?",
                    nl="Met `{call}` neemt je run meerdere regels van het bestand: `indent` verdeelt de JSON over regels. In JSON Lines is elke run één regel, zodat een programma het bestand regel voor regel kan lezen, en een half geschreven laatste regel één run kost. Kun je de run zonder `indent` schrijven?"),
            Mistake("text", matches=lambda got, text_: _not_json(got) and "str(" in text_,
                    en="With `{call}`, a line of the file is not JSON, as the picture under this message shows. `str(run)` writes the dict as Python shows it, with single quotes, and `json.loads` cannot read that back. Which function of `json` writes a dict as text?",
                    nl="Met `{call}` is een regel van het bestand geen JSON, zoals de afbeelding onder dit bericht laat zien. `str(run)` schrijft de dict zoals Python een dict laat zien, met enkele aanhalingstekens, en `json.loads` kan dat niet teruglezen. Welke functie van `json` schrijft een dict als tekst?"),
            Mistake("not-json", matches=lambda got, text_: _not_json(got),
                    en="With `{call}`, a line of the file is not JSON, as the picture under this message shows, and `json.loads` cannot read it back. An empty line is not JSON either. Is every line of the file `json.dumps(run)`, with one `\"\\n\"` after it?",
                    nl="Met `{call}` is een regel van het bestand geen JSON, zoals de afbeelding onder dit bericht laat zien, en `json.loads` kan de regel niet teruglezen. Ook een lege regel is geen JSON. Is elke regel van het bestand `json.dumps(run)`, met één `\"\\n\"` erachter?"),
            Mistake("no-newline", matches=lambda got, text_: isinstance(got, dict) and got.get("newline") is False,
                    en="With `{call}`, the file does not end with a line break, so the next run would land on the same line as this one, and that line would no longer be JSON. Does every run you write end with `\"\\n\"`?",
                    nl="Met `{call}` eindigt het bestand niet met een regeleinde, dus zou de volgende run op dezelfde regel komen als deze, en dan is de regel geen JSON meer. Eindigt elke run die je schrijft met `\"\\n\"`?"),
            Mistake("overwrite", matches=lambda got, text_: _lost_earlier(got), state="almost",
                    en="Right with an empty file. But with `{call}`, the earlier run in the file is gone: `\"w\"` empties a file when it opens it. A tracker keeps every run. Which mode adds to the end of the file?",
                    nl="Goed met een leeg bestand. Maar met `{call}` is de eerdere run in het bestand weg: `\"w\"` maakt een bestand leeg bij het openen. Een tracker bewaart elke run. Welke modus voegt toe aan het eind van het bestand?"),
            Mistake("mse", wrong=_T_MSE,
                    en="`{call}` gives {got}, and it should give {want}. Your numbers are in the thousands: that is the mean squared error, in bikes squared. What turns it into a number of bikes?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. Je getallen lopen in de duizenden: dat is de mean squared error, in fietsen in het kwadraat. Wat maakt er weer een aantal fietsen van?"),
            Mistake("train-hours", wrong=_T_TRAIN,
                    en="`{call}` gives {got}, and it should give {want}: you measure the model on the hours it learns from, `x` and `y`. The validation set says how the model does on days it never learned from. Which tensors hold those days?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je meet het model op de uren waar het van leert, `x` en `y`. De validation set zegt hoe het model het doet op dagen waar het nooit van leerde. Welke tensors bevatten deze dagen?"),
            Mistake("mae", wrong=_T_MAE,
                    en="`{call}` gives {got}, and it should give {want}: that is the mean absolute error. The RMSE squares every difference first, so that an hour that is far off counts more, then takes the mean and the square root. Does your line square the differences?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is de mean absolute error. De RMSE kwadrateert eerst elk verschil, zodat een uur dat ver ernaast zit zwaarder telt, en neemt dan het gemiddelde en de wortel. Kwadrateert je regel de verschillen?"),
            Mistake("total", wrong=_T_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: you add up the squared differences of all the hours. The RMSE takes their mean, so that it does not grow with the number of hours. Which method gives the mean?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt de gekwadrateerde verschillen van alle uren op. De RMSE neemt het gemiddelde, zodat de uitkomst niet groeit met het aantal uren. Welke methode geeft het gemiddelde?"),
            Mistake("column", wrong=_T_COLUMN,
                    en="`{call}` gives {got}, and it should give {want}. `model(x_val)` and `y_val` are both a column of shape (n, 1); a row of shape (n,) minus a column of shape (n, 1) gives, by broadcasting, n times n differences, every hour against every hour. Can you leave both in their own shape?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. `model(x_val)` en `y_val` zijn allebei een kolom met shape (n, 1); een rij met shape (n,) min een kolom met shape (n, 1) geeft door broadcasting n keer n verschillen, elk uur tegen elk uur. Kun je allebei de shape (n, 1) laten houden?"),
            Mistake("count", wrong=_T_COUNT, state="almost",
                    en="Right for `DATA`. But `{call}` gives {got}, and it should give {want}: you divide by `1733`, the number of validation hours in `DATA`, and `FEW` has fewer. Does your RMSE work for any number of hours?",
                    nl="Goed voor `DATA`. Maar `{call}` geeft {got}, en het moet {want} zijn: je deelt door `1733`, het aantal validatie-uren in `DATA`, en `FEW` heeft er minder. Werkt je RMSE voor elk aantal uren?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the RMSE of the model on the validation hours, and a line that adds the run to the file as JSON.",
                      "nl": "Vervang de twee `...`: de RMSE van het model op de validatie-uren, en een regel die de run als JSON aan het bestand toevoegt."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(name, params, data, log_path, show=None)`?",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(name, params, data, log_path, show=None)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `rmse` the square root of the mean squared difference between `model(x_val)` and `y_val`, and does the file get the run as one line of JSON?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `rmse` de wortel uit het gemiddelde gekwadrateerde verschil tussen `model(x_val)` en `y_val`, en krijgt het bestand de run als één regel JSON?"},
            "almost": {"en": "Right for `DATA`. But `{call}` gives {got}, and it should give {want}. Does your function work for any data and any file, also one that already holds runs?",
                       "nl": "Goed voor `DATA`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor alle data en elk bestand, ook een bestand waarin al runs staan?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return model, run`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met `return model, run`?"},
            "correct": {"en": "Right in all {n} cases: every epoch adds the RMSE on the validation days to the run, and the file keeps each run as one line. Your `train_version` is now in your project, in `{saved}`. Which version will you pack, and what goes with it?",
                        "nl": "Goed in alle {n} gevallen: elke epoch voegt de RMSE op de validatiedagen aan de run toe, en het bestand bewaart elke run als één regel. Je `train_version` staat nu in je project, in `{saved}`. Welke versie pak je in, en wat gaat er mee?"},
            "kept": _kept("train_version"),
        },
    ),
    "pack": FunctionBox(
        name="pack",
        function="pack",
        start=_P_START,
        solution=_P,
        save_as=("ops", "pack"),
        visible=[(MODEL, RUN, SCALE, FOLDER)],
        hidden=[(SMALL_MODEL, SHORT, SMALL_SCALE, FOLDER), (WIDE_MODEL, NOISY, SCALE, FOLDER)],
        first=[_DOTS],
        collect=_packed,
        show=_pack_shown,
        picture=pack_picture,
        mistakes=[
            Mistake("fixed-epoch", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an `IndexError`: the list in `run[\"metrics\"][\"val_rmse\"]` has no pair at the place you ask for, and not every run has the same number of epochs. Does your `val_rmse` take the last pair of the list, however long the list is?",
                    nl="Met `{call}` stopt je functie met een `IndexError`: de lijst in `run[\"metrics\"][\"val_rmse\"]` heeft geen paar op de plek die je vraagt, en niet elke run heeft evenveel epochs. Neemt je `val_rmse` het laatste paar van de lijst, hoe lang de lijst ook is?"),
            Mistake("full", wrong=_P_FULL,
                    en="`{call}` gives {got}, and it should give {want}: your version is the whole sha256, 64 characters. The course keeps the first 12, as in the lesson *Which model is this?*: enough to tell versions apart, and short enough to read. Which slice gives the first 12?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je versie is de hele sha256, 64 tekens. De course houdt de eerste 12, zoals in de les *Welk model is dit?*: genoeg om versies uit elkaar te houden, en kort genoeg om te lezen. Welke slice geeft de eerste 12?"),
            Mistake("md5", wrong=_P_MD5,
                    en="`{call}` gives {got}, and it should give {want}: `hashlib.md5` is another hash, with other characters. The version of the course is the start of the sha256. Which function of `hashlib` gives it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `hashlib.md5` is een andere hash, met andere tekens. De versie van de course is het begin van de sha256. Welke functie van `hashlib` geeft deze hash?"),
            Mistake("name", wrong=_P_NAME,
                    en="`{call}` gives {got}, and it should give {want}: you hash the path of the file, which is the same for every model you pack there. The version must change when a single weight changes. What does the open file `f` give you?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je hasht het pad van het bestand, en het pad is gelijk voor elk model dat je daar inpakt. De versie moet veranderen als één weight verandert. Wat geeft het open bestand `f` je?"),
            Mistake("state", wrong=_P_STATE,
                    en="`{call}` gives {got}, and it should give {want}: you hash how Python prints the state_dict, which shows only a few digits of each weight and leaves out most of the numbers of a large tensor. The version is the hash of the bytes in the file you hand over. What does the open file `f` give you?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je hasht hoe Python de state_dict print, met een paar cijfers van elke weight en zonder de meeste getallen van een grote tensor. De versie is de hash van de bytes in het bestand dat je doorgeeft. Wat geeft het open bestand `f` je?"),
            Mistake("last", wrong=_P_LAST,
                    en="`{call}` gives {got}, and it should give {want}: you take the last 12 characters. The course takes the first 12, as in the lesson *Which model is this?*. Which slice gives those?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je neemt de laatste 12 tekens. De course neemt de eerste 12, zoals in de les *Welk model is dit?*. Welke slice geeft deze tekens?"),
            Mistake("builtin", wrong=_P_BUILTIN,
                    en="`{call}` gives {got}, and it should give {want}: the `hash` that Python has built in gives other numbers each time Python starts, so a version from it would not be the same tomorrow. Which function of `hashlib` gives a hash that stays the same?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de ingebouwde `hash` van Python geeft elke keer dat Python start andere getallen, dus is een versie daarmee morgen anders. Welke functie van `hashlib` geeft een hash die gelijk blijft?"),
            Mistake("lowest", wrong=_P_LOWEST, state="almost",
                    en="Right for `RUN`, whose lowest RMSE is at its last epoch. But `{call}` gives {got}, and it should give {want}: in this run the RMSE was lower at an earlier epoch than at the last. The model you pack is the one after the last epoch. Which value belongs to that model?",
                    nl="Goed voor `RUN`, met de laagste RMSE bij de laatste epoch. Maar `{call}` geeft {got}, en het moet {want} zijn: in deze run was de RMSE bij een eerdere epoch lager dan bij de laatste. Je pakt het model na de laatste epoch in. Welke waarde hoort bij dat model?"),
            Mistake("first", wrong=_P_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: you take the RMSE after the first epoch, when the model had hardly learned. The model you pack is the one after the last epoch. Which pair of the list belongs to it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je neemt de RMSE na de eerste epoch, toen het model nog nauwelijks had geleerd. Je pakt het model na de laatste epoch in. Welk paar van de lijst hoort daarbij?"),
            Mistake("pair", wrong=_P_PAIR,
                    en="`{call}` gives {got}, and it should give {want}: you give the whole pair, the epoch and the RMSE. `val_rmse` in config.json is one number. Which of the two do you need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je geeft het hele paar, de epoch en de RMSE. `val_rmse` in config.json is één getal. Welke van de twee heb je nodig?"),
            Mistake("epoch", wrong=_P_EPOCH,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of the last epoch, not its RMSE. Each pair is `[epoch, value]`. Which place holds the value?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is het nummer van de laatste epoch, niet de RMSE. Elk paar is `[epoch, value]`. Op welke plek staat de waarde?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the version of the weights file, and the RMSE of the last epoch of the run.",
                      "nl": "Vervang de twee `...`: de versie van het bestand met weights, en de RMSE van de laatste epoch van de run."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(model, run, scale, folder)`?",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(model, run, scale, folder)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the version the first 12 characters of the sha256 of the bytes of model.safetensors, and `val_rmse` the RMSE of the last epoch?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is de versie de eerste 12 tekens van de sha256 van de bytes van model.safetensors, en `val_rmse` de RMSE van de laatste epoch?"},
            "almost": {"en": "Right for `RUN`. But `{call}` gives {got}, and it should give {want}. Does your function work for any model and any run?",
                       "nl": "Goed voor `RUN`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elk model en elke run?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return config`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt de functie nog met `return config`?"},
            "correct": {"en": "Right in all {n} cases: the folder holds the weights as safetensors and a config.json with the settings, the run, its RMSE and the version of the file. Your `pack` is now in your project, in `{saved}`. What does someone who loads the package need to check first?",
                        "nl": "Goed in alle {n} gevallen: de map bevat de weights als safetensors en een config.json met de instellingen, de run, de RMSE en de versie van het bestand. Je `pack` staat nu in je project, in `{saved}`. Wat moet iemand die het pakket laadt eerst controleren?"},
            "kept": _kept("pack"),
        },
    ),
}

REQUIRED = ["split", "train", "pack", "scale"]


# ======== the demo: four versions with your parts ==============================================================================

class _Broken(Exception):
    """A part of yours stopped the demo: the message says which."""


def _imported(code: str, name: str):
    got = run_code(code)
    value = got.names.get(name) if got.error is None else None
    return value if callable(value) else None


_PARTS = (("split", "split", "split_days", _S), ("train", "train", "train_version", _T), ("pack", "pack", "pack", _P))


def project_parts() -> dict:
    """split_days, train_version and pack of the learner's project when they are there and pass their contract test,
    else the course's: a dict with the three and which of them are yours."""
    out = {"yours": []}
    for key, module, name, solution in _PARTS:
        fn = None
        if projects.state("ops", module) == "yours":
            fn = _imported(f"from ops.{module} import {name}", name)
            if fn is not None:
                out["yours"].append(key)
        out[key] = fn if fn is not None else _imported(solution, name)
    return out


def load_package(folder: Path):
    """The package as the course reads it: config.json, a check that the weights are the version it names, and the
    network with the weights from the safetensors file."""
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    weights = folder / "model.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest()[:12] != config["version"]:
        raise ValueError("the weights are not the version config.json names")
    model = demand.DemandNet(config["hidden"])
    model.load_state_dict(tensors_io.load_file(weights))
    model.eval()
    return model, config


def _last(run) -> float:
    try:
        return float(run["metrics"]["val_rmse"][-1][1])
    except Exception:   # noqa: BLE001
        return float("nan")


def train_versions(parts: dict, show=None, log: Path | None = None, folder: Path | None = None) -> dict:
    """The four versions with the parts in `parts`, the best of them packed: what the demo shows."""
    hours = HOURS_2011.copy()
    try:
        train, val = parts["split"](hours.copy())
        if not (isinstance(train, pd.DataFrame) and isinstance(val, pd.DataFrame) and len(train) and len(val)):
            raise ValueError("no two tables")
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   any error of the learner's split
        raise _Broken("split")
    scale = demand.scaling(train)
    data = tensors_of(train, val, scale)
    x_day = demand.inputs(hours[hours["day"] == WATCH_DAY], scale)
    log = log or projects.folder() / "ops" / "runs.jsonl"
    folder = folder or projects.folder() / "ops" / "model"
    log.parent.mkdir(parents=True, exist_ok=True)
    runs, models, frames = [], {}, {}
    for name, hidden, lr in VERSIONS:
        frames[name] = []

        def look(epoch, model, run, name=name):
            try:
                with netview.looking(model):
                    preds = [float(v) for v in model(x_day).reshape(-1)]
            except Exception:   # noqa: BLE001   a look never stops the learner's run
                preds = []
            frames[name].append((int(epoch), _last(run), preds))
            if show is not None:
                done = [(n, [(e, r) for e, r, _ in frames[n]]) for n, _, _ in VERSIONS if n != name and frames.get(n)]
                show(name, int(epoch), _last(run), preds, done)

        try:
            model, run = parts["train"](name, params(hidden, lr), data, log, look)
            if not math.isfinite(_last(run)):
                raise ValueError("no RMSE")
        except Timeout:
            raise
        except Exception:   # noqa: BLE001
            raise _Broken("train")
        if not frames[name]:   # a train_version without the line that calls show: the run's own numbers, no pictures
            frames[name] = [(int(e), float(v), []) for e, v in run["metrics"]["val_rmse"]]
        runs.append(run)
        models[name] = model
    best = min(runs, key=_last)
    try:
        config = parts["pack"](models[best["name"]], best, scale, folder)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001
        raise _Broken("pack")
    try:
        loaded, written = load_package(folder)
        with torch.no_grad():
            again = [float(v) for v in loaded(x_day).reshape(-1)]
        with netview.looking(models[best["name"]]):
            before = [float(v) for v in models[best["name"]](x_day).reshape(-1)]
    except Exception:   # noqa: BLE001
        raise _Broken("package")
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return {"runs": runs, "frames": frames, "best": best["name"], "config": written, "folder": folder,
            "same": len(again) == len(before) and all(abs(a - b) < 1e-4 for a, b in zip(again, before)),
            "lines": len(lines), "bytes": (folder / "model.safetensors").stat().st_size, "day": day_counts(WATCH_DAY)}


def demo(show=None) -> dict | str:
    """The demo within its time limit: a dict, or a string that says which part stopped it, or "slow"."""
    parts = project_parts()
    try:
        out = with_time_limit(lambda: train_versions(parts, show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except _Broken as e:
        return str(e)
    out["yours"] = parts["yours"]
    return out


_COURSE_RUN: dict = {}


def course_run() -> dict:
    """The four versions with the course's own parts, in a folder of the check: the numbers the lesson's text gives."""
    if not _COURSE_RUN:
        parts = {key: _imported(solution, name) for key, _, name, solution in _PARTS}
        folder = _check_folder() / "course"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        _COURSE_RUN.update(train_versions(parts, log=folder / "runs.jsonl", folder=folder / "model"))
    return _COURSE_RUN


# --- what the demo draws ---------------------------------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Validation RMSE of each version", "x": "epoch", "y": "bikes per hour",
           "alt": "The validation RMSE after each epoch of the version that trains now, with the versions already trained as thin dashed lines.",
           "progress": "Training **{name}**: epoch {e} of {n}, validation RMSE {r} bikes per hour.",
           "day": "14 July, {name}, epoch {e}", "real": "bikes rented", "model": "what the version expects",
           "day_alt": "The bikes rented in each hour of Thursday 14 July 2011 as bars, and what the version expects in each hour as a line.",
           "hour": "hour", "bikes": "bikes",
           "rewind": "After epoch {e}: {listed}.",
           "rewind_alt": "The validation RMSE of the four versions up to the chosen epoch, with a line at that epoch.",
           "key": {"h16-lr0.01": "h16-lr0.01", "h64-lr0.01": "h64-lr0.01", "h16-lr0.001": "h16-lr0.001", "h64-lr0.001": "h64-lr0.001"},
           "who": {3: "your `split_days`, your `train_version` and your `pack`",
                   0: "the course's `split_days`, `train_version` and `pack`, since yours are not in your project yet",
                   "partly": "{yours}, and the course's {course}, since yours {verb} not pass {its} check yet"},
           "summary": "Trained with {who}. The lowest validation RMSE at the last epoch: **{best}**, {rmse} bikes per hour. {packer} put it in `{folder}`: model.safetensors of {bytes} bytes and config.json with version `{version}`. Loaded back with the course's reader, the package expects {same} for Thursday 14 July. `runs.jsonl` now holds {lines} runs.",
           "same": "the same bikes in every hour as the trained version", "differ": "other bikes than the trained version",
           "broken": {"split": "*Your `split_days` stopped with an error, or gave no two tables with hours. Does it still pass its check?*",
                      "train": "*Your `train_version` stopped with an error, or gave a run without a validation RMSE. Does it still pass its check?*",
                      "pack": "*Your `pack` stopped with an error. Does it still pass its check?*",
                      "package": "*The course could not read back the package your `pack` made, or its version does not match the weights. Does your `pack` still pass its check?*",
                      "slow": "*The four versions took more than three minutes. Does your `train_version` do more work than it needs?*"}},
    "nl": {"title": "Validation RMSE van elke versie", "x": "epoch", "y": "fietsen per uur",
           "alt": "De validation RMSE na elke epoch van de versie die nu traint, met de versies die al getraind zijn als dunne lijnen van streepjes.",
           "progress": "Training van **{name}**: epoch {e} van {n}, validation RMSE {r} fietsen per uur.",
           "day": "14 juli, {name}, epoch {e}", "real": "gehuurde fietsen", "model": "wat de versie verwacht",
           "day_alt": "De gehuurde fietsen in elk uur van donderdag 14 juli 2011 als staven, en wat de versie in elk uur verwacht als lijn.",
           "hour": "uur", "bikes": "fietsen",
           "rewind": "Na epoch {e}: {listed}.",
           "rewind_alt": "De validation RMSE van de vier versies tot de gekozen epoch, met een lijn bij deze epoch.",
           "key": {"h16-lr0.01": "h16-lr0.01", "h64-lr0.01": "h64-lr0.01", "h16-lr0.001": "h16-lr0.001", "h64-lr0.001": "h64-lr0.001"},
           "who": {3: "je `split_days`, je `train_version` en je `pack`",
                   0: "de `split_days`, `train_version` en `pack` van de course, omdat jouw functies nog niet in je project staan",
                   "partly": "{yours}, en {course} van de course, omdat jouw {fn} nog niet door de check {verb}"},
           "summary": "Getraind met {who}. De laagste validation RMSE bij de laatste epoch: **{best}**, {rmse} fietsen per uur. {packer} zette deze versie in `{folder}`: model.safetensors van {bytes} bytes en config.json met versie `{version}`. Teruggelezen met de lezer van de course verwacht het pakket {same} voor donderdag 14 juli. In `runs.jsonl` staan nu {lines} runs.",
           "same": "in elk uur dezelfde fietsen als de getrainde versie", "differ": "andere fietsen dan de getrainde versie",
           "broken": {"split": "*Je `split_days` stopte met een fout, of gaf geen twee tabellen met uren. Komt de functie nog door de check?*",
                      "train": "*Je `train_version` stopte met een fout, of gaf een run zonder validation RMSE. Komt de functie nog door de check?*",
                      "pack": "*Je `pack` stopte met een fout. Komt de functie nog door de check?*",
                      "package": "*De course kon het pakket van je `pack` niet teruglezen, of de versie klopt niet met de weights. Komt je `pack` nog door de check?*",
                      "slow": "*De vier versies duurden langer dan drie minuten. Doet je `train_version` meer werk dan nodig is?*"}},
}
# how each version is drawn in the chart of the rewind: role, dashed; the key under it says which is which
STYLE = {"h16-lr0.01": ("loss", False), "h64-lr0.01": ("loss", True), "h16-lr0.001": ("ink-3", False), "h64-lr0.001": ("ink-3", True)}
_PART_NAME = {"split": "`split_days`", "train": "`train_version`", "pack": "`pack`"}


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


def live_labels(lang: str) -> dict:
    w = DEMO_TEXT[_lang(lang)]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def progress_line(name: str, epoch: int, rmse: float, lang: str) -> str:
    lang = _lang(lang)
    return DEMO_TEXT[lang]["progress"].format(name=name, e=epoch, n=EPOCHS, r=number(rmse, 1, lang))


def day_picture(preds: list, name: str, epoch: int, lang: str) -> str:
    """The bikes of Thursday 14 July as bars, and what a version expects in each hour as a line."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    real = day_counts(WATCH_DAY)
    top = 100 * math.ceil(max(real + [v for v in preds if math.isfinite(v)] + [100]) / 100)
    top = min(top, 1500)
    p = chart.Panel((-0.5, 23.5), (0, top), title=w["day"].format(name=name, e=epoch), xlabel=w["hour"], ylabel=w["bikes"],
                    xticks=[0, 6, 12, 18, 23])
    for h, v in enumerate(real):
        p.rect(h - 0.4, 0, h + 0.4, v, "data")
    if len(preds) == 24:
        p.line(list(range(24)), [max(0.0, min(v, top)) for v in preds], "pred", 2.4)
    return _chart([p], w["day_alt"], lang, width=300, height=230) + live.key_html([("data", False, w["real"]), ("pred", False, w["model"])])


def now_picture(state: dict, lang: str) -> str:
    """Beside the live chart: what the version that trains now expects for Thursday 14 July."""
    return day_picture(state["preds"], state["name"], state["epoch"], lang)


def _curve_panel(r: dict, e: int, lang: str) -> chart.Panel:
    w = DEMO_TEXT[lang]
    p = chart.Panel((1, EPOCHS), (LOW, HIGH), title=w["title"], xlabel=w["x"], ylabel=w["y"], xticks=[1, 5, 10, 15, 20, 25, 30])
    for name, _, _ in VERSIONS:
        points = [(k, min(max(v, LOW), HIGH)) for k, v, _ in r["frames"].get(name, []) if math.isfinite(v)]
        if points:
            role, dashed = STYLE[name]
            p.line([k for k, _ in points], [v for _, v in points], role, 1.4 if dashed else 2.2, dashed=dashed)
    p.line([e, e], [LOW, HIGH], "axis", 1.2)
    return p


def curve_key(lang: str) -> list:
    w = DEMO_TEXT[_lang(lang)]
    return [(STYLE[name][0], STYLE[name][1], w["key"][name]) for name, _, _ in VERSIONS]


def rewind_frame(r: dict, e: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the four curves with a line at epoch e, beside it what the best version
    expected for Thursday 14 July after that epoch, and the RMSE of each version then."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    e = max(1, min(int(e), EPOCHS))
    curves = _chart([_curve_panel(r, e, lang)], w["rewind_alt"], lang) + live.key_html(curve_key(lang))
    frames = r["frames"][r["best"]]
    at = next((f for f in frames if f[0] == e), frames[-1])
    listed = []
    for name, _, _ in VERSIONS:
        hit = next((f for f in r["frames"].get(name, []) if f[0] == e), None)
        if hit is not None and math.isfinite(hit[1]):
            listed.append(f"{name} {number(hit[1], 1, lang)}")
    return live.rewound(curves, [day_picture(at[2], r["best"], at[0], lang)], width=330), w["rewind"].format(e=e, listed=", ".join(listed))


def _listed(items: list, lang: str) -> str:
    joiner = " and " if lang == "en" else " en "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + joiner + items[-1]


def summary(r: dict, lang: str) -> str:
    """The sentence under the rewind: whose parts trained, the best version, its package and the file of runs."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    mine = [k for k in ("split", "train", "pack") if k in r["yours"]]
    course = [k for k in ("split", "train", "pack") if k not in r["yours"]]
    if not course:
        who = w["who"][3]
    elif not mine:
        who = w["who"][0]
    else:
        yours_ = _listed([("your " if lang == "en" else "je ") + _PART_NAME[k] for k in mine], lang)
        theirs = _listed([_PART_NAME[k] for k in course], lang)
        if lang == "en":
            who = w["who"]["partly"].format(yours=yours_, course=theirs, verb="do" if len(course) > 1 else "does",
                                            its="their" if len(course) > 1 else "its")
        else:
            who = w["who"]["partly"].format(yours=yours_, course="de " + theirs, verb="komen" if len(course) > 1 else "komt",
                                            fn="functies" if len(course) > 1 else "functie")
    best = next(run for run in r["runs"] if run["name"] == r["best"])
    packer = {("en", True): "Your `pack`", ("en", False): "The course's `pack`",
              ("nl", True): "Je `pack`", ("nl", False): "De `pack` van de course"}[(lang, "pack" in r["yours"])]
    return w["summary"].format(who=who, packer=packer, best=r["best"], rmse=number(_last(best), 1, lang),
                               folder=projects.shown(r["folder"]), bytes=number(r["bytes"], 0, lang),
                               version=html.escape(str(r["config"].get("version"))), same=w["same"] if r["same"] else w["differ"],
                               lines=number(r["lines"], 0, lang))


def broken(key: str, lang: str) -> str:
    return DEMO_TEXT[_lang(lang)]["broken"].get(key, DEMO_TEXT[_lang(lang)]["broken"]["train"])


def package_picture(r: dict, lang: str) -> str:
    """After the demo: the package your pack made, as the picture under exercise 3 shows it."""
    lang = _lang(lang)
    w = _YOURS_TEXT[lang]["pack"]
    best = next(run for run in r["runs"] if run["name"] == r["best"])
    return package_svg(r["folder"], best, w, lang, w["alt"])


# ======== what the tests try ===================================================================================================

CASES = [
    ("split", _S_START, "empty", None),
    ("split", _S, "correct", None),
    ("split", _s('(hours["day"] + 1) % 5 == 0'), "correct", None),
    ("split", _s('hours["day"].isin(range(4, 365, 5))'), "correct", None),
    ("split", _s('hours["day"] % 5 == ...'), "wrong", "dots-left"),
    ("split", _S_HOUR, "wrong", "hour"),
    ("split", _S_ZERO, "wrong", "zero"),
    ("split", _S_SWAPPED, "wrong", "swapped"),
    ("split", _S_FIRST, "wrong", "first"),
    ("split", _s('hours["instant"] % 5 == 4'), "wrong", "split-days"),
    ("split", _s('hours["day"] // 5 == 4'), "wrong", None),
    ("split", _s('(hours["day"] % 5 == 4) & (len(hours) > 1000)'), "almost", None),
    ("train", _T_START, "empty", None),
    ("train", _T, "correct", None),
    ("train", _tv("torch.sqrt(torch.mean((model(x_val) - y_val) ** 2))", _T_LINE), "correct", None),
    ("train", _tv("((y_val - model(x_val)) ** 2).mean() ** 0.5", "f.write(json.dumps(run))\n        f.write(\"\\n\")"), "correct", None),
    ("train", _tv(_T_RMSE, "json.dump(run, f)\n        f.write(\"\\n\")"), "correct", None),
    ("train", _tv(_T_RMSE, "..."), "wrong", "dots-left"),
    ("train", _tv(_T_RMSE, "f.write(json.dumps(model) + \"\\n\")"), "error", "model-in-line"),
    ("train", _tv(_T_RMSE, "f.write(json.dumps(run, indent=2) + \"\\n\")"), "wrong", "pretty"),
    ("train", _tv(_T_RMSE, "f.write(str(run) + \"\\n\")"), "wrong", "text"),
    ("train", _tv(_T_RMSE, "f.write(json.dumps(run) + \"\\n\\n\")"), "wrong", "not-json"),
    ("train", _tv(_T_RMSE, "f.write(json.dumps(run))"), "wrong", "no-newline"),
    ("train", _tv(_T_RMSE, "json.dump(run, f)"), "wrong", "no-newline"),
    ("train", _tv(_T_RMSE, "f.truncate(0)\n        f.write(json.dumps(run) + \"\\n\")"), "almost", "overwrite"),
    ("train", _T_MSE, "wrong", "mse"),
    ("train", _T_TRAIN, "wrong", "train-hours"),
    ("train", _T_MAE, "wrong", "mae"),
    ("train", _T_TOTAL, "wrong", "total"),
    ("train", _T_COLUMN, "wrong", "column"),
    ("train", _tv("((model(x_val).squeeze() - y_val) ** 2).mean().sqrt()", _T_LINE), "wrong", "column"),
    ("train", _T_COUNT, "almost", "count"),
    ("train", _tv("((model(x_val) - y_val) ** 2).mean().sqrt() + 1", _T_LINE), "wrong", None),
    ("train", _tv(_T_RMSE, "f.write(json.dumps(run) + \"\\n\") if run[\"params\"][\"epochs\"] > 2 else None"), "almost", None),
    ("pack", _P_START, "empty", None),
    ("pack", _P, "correct", None),
    ("pack", _p("hashlib.sha256(f.read()).hexdigest()[0:12]", 'run["metrics"]["val_rmse"][len(run["metrics"]["val_rmse"]) - 1][1]'), "correct", None),
    ("pack", _p(_P_VERSION, "..."), "wrong", "dots-left"),
    ("pack", _P_FIXED, "error", "fixed-epoch"),
    ("pack", _P_FULL, "wrong", "full"),
    ("pack", _P_MD5, "wrong", "md5"),
    ("pack", _P_NAME, "wrong", "name"),
    ("pack", _P_STATE, "wrong", "state"),
    ("pack", _P_LAST, "wrong", "last"),
    ("pack", _P_BUILTIN, "wrong", "builtin"),
    ("pack", _P_LOWEST, "almost", "lowest"),
    ("pack", _P_FIRST, "wrong", "first"),
    ("pack", _P_PAIR, "wrong", "pair"),
    ("pack", _P_EPOCH, "wrong", "epoch"),
    ("pack", _p("hashlib.sha256(f.read()).hexdigest()[:11]", _P_RMSE), "wrong", None),
    ("pack", _p(_P_VERSION, '53.04'), "almost", None),
]
