# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-fit-scaling-law: best_floor, the floor that makes the losses of a few runs the straightest
line on a log-log plot; predict, the loss a fitted law gives a bigger model; compute_optimal, the parameters and tokens
of a compute budget at a fixed number of tokens per parameter; and a choice. The demos train the learner's ModernGPT at
four small sizes, one epoch each on the shards of llm-data-pipeline, fit a law through their losses and predict the
loss of a bigger one, which the learner then trains."""
import importlib.util
import json
import math
import numbers
import time
from decimal import Decimal

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _checks("llm-modern-gpt", "everything_ai_checks_llm_modern_gpt_for_scaling")
D = _checks("llm-data-pipeline", "everything_ai_checks_llm_data_pipeline_for_scaling")
BUILDS_ON = M.BUILDS_ON + tuple(m for m in D.BUILDS_ON if m[1] in ("shards", "token_windows"))
projects.builds_on(*BUILDS_ON)
# saved by exercise 3 for the capstone; no demo here imports it, so it is not among what this lesson builds on
SAVES = (("llm", "compute_optimal", "llm-fit-scaling-law", "compute-optimal"),)

WIDTH = 32           # the width of a model is WIDTH times its depth, with one head per HEAD numbers
HEAD = 32
SMALL = (2, 3, 4, 5)  # the depths of the four small runs
BIG = 6              # the depth of the run the law predicts
BLOCK, BATCH = 256, 16   # windows of BLOCK tokens, mini-batches of BATCH; one epoch over the training shards
PEAK = 3e-3
SCHEDULE = {"warmup": 40, "warmdown": 0.65, "final": 0.05}   # a warmup, a plateau and a straight warmdown
FLOORS = [round(0.05 * i, 2) for i in range(0, 60)]          # the floors best_floor tries, below the lowest loss
CHINCHILLA = 20      # tokens per parameter that Hoffmann and others (2022) found compute-optimal
RUNS = "scaling_runs.json"   # the runs of the learner, in work/projects/llm/


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def multiplier(step: int, steps: int) -> float:
    warmdown = round(SCHEDULE["warmdown"] * steps)
    if step < SCHEDULE["warmup"]:
        return (step + 1) / SCHEDULE["warmup"]
    if step <= steps - warmdown:
        return 1.0
    progress = (steps - step) / warmdown
    return progress + (1 - progress) * SCHEDULE["final"]


def floors_for(losses: list) -> list:
    return [f for f in FLOORS if f < min(losses) - 0.05]


def _shards(tools) -> tuple | str:
    """The training and validation windows of the shards of llm-data-pipeline, written first with the course's versions
    when they are missing."""
    import numpy as np
    if not (D.folder() / "info.json").exists():
        written = D.write_demo(run_code(D.EXERCISES["write-shards"].solution).names["write_shards"])
        if not isinstance(written, dict):
            return "run"
    info = json.loads((D.folder() / "info.json").read_text(encoding="utf-8"))
    files = [str(D.folder() / name) for name in info["files"]]
    dtype = np.dtype(info["dtype"])
    return tools["TokenWindows"](files[:-1], BLOCK, dtype), tools["TokenWindows"](files[-1:], BLOCK, dtype), info


def train(depth: int, show) -> dict | str:
    """One run: the learner's ModernGPT of `depth` blocks, WIDTH * depth wide with a head per HEAD, one epoch of BATCH
    windows of BLOCK tokens over the training shards, AdamW at PEAK times multiplier(), seed 0. `show` gets the step
    every 10 steps. "run" when a module of the learner stops with an error; else its parameters, its tokens, its
    validation loss per token on the last shard, and the seconds."""
    import torch
    from torch.utils.data import DataLoader
    ran = run_code("from llm.modern_gpt import ModernGPT\nfrom llm.token_windows import TokenWindows")
    if ran.error is not None:
        return "run"
    data = _shards(ran.names)
    if isinstance(data, str):
        return data
    train_ds, val_ds, info = data
    d = WIDTH * depth
    device = M.plan()["device"]
    torch.manual_seed(0)
    model = ran.names["ModernGPT"](D.B.VOCAB, d, d // HEAD, d // HEAD, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.1)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(0))
    steps, start = len(loader), time.perf_counter()
    model.train()
    for step, (xb, yb) in enumerate(loader):
        for group in opt.param_groups:
            group["lr"] = PEAK * multiplier(step, steps)
        with M.on_device(device):
            logits = model(xb.to(device))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.to(device).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 9:
            show({"depth": depth, "step": step + 1, "steps": steps, "loss": loss.item(), "seconds": time.perf_counter() - start})
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in DataLoader(val_ds, batch_size=32):
            with M.on_device(device):
                logits = model(xb.to(device))
            total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.to(device).reshape(-1), reduction="sum").item()
            count += yb.numel()
    return {"depth": depth, "params": sum(p.numel() for p in model.parameters()), "tokens": len(train_ds) * BLOCK,
            "val": total / count, "seconds": time.perf_counter() - start, "device": device}


def saved_runs() -> dict:
    """The learner's runs by depth, from work/projects/llm/scaling_runs.json; {} when there are none."""
    path = projects.folder() / "llm" / RUNS
    try:
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


def save_run(run: dict) -> None:
    runs = saved_runs()
    runs[run["depth"]] = run
    path = projects.folder() / "llm" / RUNS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in sorted(runs.items())}, indent=1) + "\n", encoding="utf-8")


def law_picture(runs: list, floor: float, big: dict | None, predicted: float, labels: dict) -> str:
    """The validation loss against the parameters, on a log scale: the small runs as dots (loss), the fitted law
    through them and on to the big model as a line, and its prediction as an open dot (prediction); the big run, when
    there is one, as a dot (loss). `runs` and `big` have "params" and "val"."""
    import numpy as np
    from everything_ai import chart
    xs = [r["params"] for r in runs]
    k, b = np.polyfit(np.log10(xs), np.log10(np.array([r["val"] for r in runs]) - floor), 1)
    x_big = big["params"] if big else labels["big_params"]
    law = lambda x: floor + 10 ** b * x ** k
    grid = np.logspace(np.log10(min(xs) * 0.8), np.log10(x_big * 1.25), 60)
    vals = [r["val"] for r in runs] + ([big["val"]] if big else []) + [predicted]
    panel = chart.Panel((min(1e5, min(xs) * 0.7), max(1e7, x_big * 1.5)),
                        (math.floor(min(vals) * 10 - 1) / 10, math.ceil(max(vals) * 10 + 1) / 10),
                        title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True)
    panel.line(list(grid), [law(g) for g in grid], "pred", 1.6, dashed=True)
    panel.points(xs, [r["val"] for r in runs], "loss")
    panel.points([x_big], [predicted], "pred", hollow=True)
    if big:
        panel.points([x_big], [big["val"]], "loss")
    # each mark named in its own colour, next to it
    panel.text(xs[0] * 1.3, runs[0]["val"], labels["measured"], "loss", "start")
    panel.text(xs[2] * 1.25, law(xs[2] * 1.25) + 0.12, labels["law"], "pred", "start")
    panel.text(x_big / 1.2, predicted - 0.07, labels["predicted"], "pred", "end")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- measured: the course's runs with its own versions, on the CPU, 2 threads ---
MEASURED = {
    "small": [
        {"depth": 2, "params": 260649, "tokens": 1571328, "val": 3.0526, "seconds": 52.994},
        {"depth": 3, "params": 575337, "tokens": 1571328, "val": 2.7517, "seconds": 96.2898},
        {"depth": 4, "params": 1110633, "tokens": 1571328, "val": 2.5486, "seconds": 167.3947},
        {"depth": 5, "params": 1940457, "tokens": 1571328, "val": 2.3999, "seconds": 269.1511},
    ],
    "big": {"depth": 6, "params": 3141801, "tokens": 1571328, "val": 2.321, "seconds": 393.2556},
    "floor": 1.5, "predicted": 2.29,   # without a floor: 2.2561
}
# --- end of measured ---

REQUIRED = ["best-floor", "predict", "compute-optimal", "why-low"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _rounded(v) -> str:
    """A value for the messages, as Python would write it but without float noise or e-notation: whole from 100000
    on (6 digits and then zeros past 10**15), else 4 significant digits, such as 2.124, 1.0 or 0.000001279; also
    inside a tuple or a list."""
    if isinstance(v, (tuple, list)):
        inner = ", ".join(_rounded(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        return repr(v)
    if 1e5 <= abs(v) < 1e15:
        return f"{v:.0f}"
    if abs(v) >= 1e15 or (v != 0 and abs(v) < 1e-3):
        return format(Decimal(f"{v:.6g}" if abs(v) >= 1e15 else f"{v:.4g}"), "f")
    return repr(int(v)) if isinstance(v, numbers.Integral) else repr(float(f"{v:.4g}"))


def _close(got, want, whole: bool = False) -> bool:
    """Equal up to float noise; with `whole`, also when rounded to a whole number, as a parameter count is."""
    if isinstance(want, tuple):
        return (isinstance(got, (tuple, list)) and len(got) == len(want)
                and all(_close(g, w, whole) for g, w in zip(got, want)))
    try:
        return abs(float(got) - float(want)) <= max(1.0 if whole else 0.0, 1e-6 * max(1.0, abs(float(want))))
    except (TypeError, ValueError):
        return False


def _close_whole(got, want) -> bool:
    return _close(got, want, whole=True)


# --- exercise 1: best_floor ------------------------------------------------------------------------------------------

_F_HEAD = "import numpy as np\n\n\ndef best_floor(x, loss, floors):\n    best, smallest = None, None\n    for floor in floors:\n"
_F_FIT = ("        log_x = np.log10(x)\n        log_rest = np.log10(np.array(loss) - floor)\n"
          "        k, b = np.polyfit(log_x, log_rest, 1)\n")
_F_ERR = "        error = np.mean((log_rest - (k * log_x + b)) ** 2)\n"
_F_KEEP = "        if smallest is None or error < smallest:\n            best, smallest = floor, error\n    return best\n"
_F_START = (_F_HEAD + "        # the straight line through the logs of x and of the loss above this floor, as in llm-scaling-laws\n        ...\n"
            "        # how far the logs are from that line: the mean of the squared differences\n        error = ...\n"
            "        # keep the floor with the smallest error\n        ...\n    return best\n")


def _f(fit: str = _F_FIT, err: str = _F_ERR, keep: str = _F_KEEP) -> str:
    return _F_HEAD + fit + err + keep


# --- exercise 2: predict ---------------------------------------------------------------------------------------------

_P_BODY = ("import numpy as np\n\n\ndef predict(x_new, x, loss, floor):\n"
           "    k, b = np.polyfit(np.log10(x), np.log10(np.array(loss) - floor), 1)\n"
           "    return floor + 10 ** b * x_new ** k\n")
_P_START = ("import numpy as np\n\n\ndef predict(x_new, x, loss, floor):\n"
            "    # the law through the runs: the slope k and the intercept b of the line through the logs\n    k, b = ...\n"
            "    # the floor, plus the power law at x_new\n    return ...\n")

# --- exercise 3: compute_optimal -------------------------------------------------------------------------------------

_C_BODY = ("def compute_optimal(compute, tokens_per_param):\n    params = (compute / (6 * tokens_per_param)) ** 0.5\n"
           "    tokens = tokens_per_param * params\n    return params, tokens\n")
_C_START = ("def compute_optimal(compute, tokens_per_param):\n"
            "    # compute = 6 * params * tokens, and tokens = tokens_per_param * params: solve for params\n    params = ...\n"
            "    tokens = tokens_per_param * params\n    return params, tokens\n")

class _Named:
    """A value of a test case, written in the check's messages the way the lesson writes it, such as `[1e5, 3e5]`."""

    def __init__(self, shown: str):
        self.shown = shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return eval(self.shown)


_X = _Named("[1e5, 3e5, 1e6, 3e6]")
_L = [3.4, 2.9, 2.55, 2.35]

EXERCISES = {
    "best-floor": FunctionBox(
        name="best-floor",
        function="best_floor",
        start=_F_START,
        solution=_f(),
        visible=[(_X, _L, [0.0, 0.5, 1.0, 1.5, 2.0, 2.2])],
        hidden=[([10, 100, 1000], [3.0, 2.0, 1.5], [0.0, 0.5, 1.0]), ([1, 2, 4, 8], [2.9, 2.4, 2.15, 2.025], [0.0, 1.0, 1.9, 1.95]),
                (_Named("[1e6, 1e7, 1e8]"), [3.0, 2.0, 1.4], [0.0, 0.3, 0.6, 0.9]), ([5, 50, 500], [4.0, 2.0, 1.0], [0.0])],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("largest", wrong=_f(keep=_F_KEEP.replace("error < smallest", "error > smallest")),
                    en=_CALL + ": your function keeps the floor with the largest error, the line that fits worst. Keep the smallest: `error < smallest`.",
                    nl=_CALL_NL + ": je functie houdt de bodem met de grootste fout, de lijn die het slechtst past. Houd de kleinste: `error < smallest`."),
            Mistake("first", wrong=_f(fit=_F_FIT.replace("np.array(loss) - floor", "np.array(loss)")),
                    en=_CALL + ": your function gives the first floor of `floors`, whatever its error. Does every floor get a line of its own, through `np.log10(np.array(loss) - floor)`? And does `return best` come after the loop, not inside it?",
                    nl=_CALL_NL + ": je functie geeft de eerste bodem van `floors`, wat zijn fout ook is. Krijgt elke bodem een eigen lijn, door `np.log10(np.array(loss) - floor)`? En staat `return best` na de loop, niet erin?"),
            Mistake("last", wrong=_f(keep="        best = floor\n    return best\n"),
                    en=_CALL + ": your function gives the last floor it tries, whatever its error. Keep a floor only when its error is the smallest so far.",
                    nl=_CALL_NL + ": je functie geeft de laatste bodem die hij probeert, wat zijn fout ook is. Houd een bodem alleen als zijn fout de kleinste tot nu toe is."),
            Mistake("error-value", wrong=_f(keep=_F_KEEP.replace("    return best\n", "    return smallest\n")),
                    en=_CALL + ": your function gives back the smallest error, not the floor that has it. Return `best`.",
                    nl=_CALL_NL + ": je functie geeft de kleinste fout terug, niet de bodem die hem heeft. Geef `best` terug."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the fit for this floor, its error, and keeping the floor with the smallest error.",
                      "nl": "Vervang de drie `...`: de fit voor deze bodem, zijn fout, en de bodem met de kleinste fout bewaren."},
            "wrong": {"en": _CALL + ". For every floor, the straight line through the logs of `x` and of `loss - floor`, and the floor whose logs lie closest to their line?",
                      "nl": _CALL_NL + ". Voor elke bodem de rechte lijn door de logs van `x` en van `loss - floor`, en de bodem waarvan de logs het dichtst bij hun lijn liggen?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `x`, `loss` and every floor of `floors`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `x`, `loss` en elke bodem van `floors`, en geen vaste getallen?"},
            "correct": {"en": "Your function finds the floor that makes the runs the straightest line on a log-log plot: the law that fits them best.",
                        "nl": "Je functie vindt de bodem die de runs de rechtste lijn op een log-log plot geeft: de wet die het best bij ze past."},
        },
    ),
    "predict": FunctionBox(
        name="predict",
        function="predict",
        start=_P_START,
        solution=_P_BODY,
        visible=[(_Named("1e7"), _X, _L, 1.5)],
        hidden=[(10000, [10, 100, 1000], [3.0, 2.0, 1.5], 1.0), (16, [1, 2, 4, 8], [2.9, 2.4, 2.15, 2.025], 1.9), (_Named("1e9"), _X, _L, 0.0)],
        show=_rounded,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-floor-back", wrong=_P_BODY.replace("    return floor + 10 ** b * x_new ** k\n", "    return 10 ** b * x_new ** k\n"),
                    en=_CALL + ": you give the part of the loss above the floor. Add the floor back: `floor + 10 ** b * x_new ** k`.",
                    nl=_CALL_NL + ": je geeft het deel van de loss boven de bodem. Tel de bodem er weer bij op: `floor + 10 ** b * x_new ** k`."),
            Mistake("b-as-a", wrong=_P_BODY.replace("10 ** b * x_new ** k", "b * x_new ** k"),
                    en=_CALL + ": `b` is the log of `a`, the intercept of the line through the logs. The factor is `10 ** b`.",
                    nl=_CALL_NL + ": `b` is de log van `a`, het snijpunt van de lijn door de logs. De factor is `10 ** b`."),
            Mistake("swapped", wrong=_P_BODY.replace("    k, b = np.polyfit", "    b, k = np.polyfit"),
                    en=_CALL + ": `np.polyfit` with degree 1 gives the slope first and then the intercept: `k, b = np.polyfit(...)`.",
                    nl=_CALL_NL + ": `np.polyfit` met graad 1 geeft eerst de helling en dan het snijpunt: `k, b = np.polyfit(...)`."),
            Mistake("no-floor-fit", wrong=_P_BODY.replace("np.log10(np.array(loss) - floor)", "np.log10(loss)"),
                    en=_CALL + ": your line goes through the logs of the whole loss, and then you add the floor on top. Fit the loss above the floor: `np.log10(np.array(loss) - floor)`.",
                    nl=_CALL_NL + ": je lijn gaat door de logs van de hele loss, en dan tel je de bodem er nog bij op. Fit de loss boven de bodem: `np.log10(np.array(loss) - floor)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the line through the logs, and the floor plus the law at `x_new`.",
                      "nl": "Vervang de twee `...`: de lijn door de logs, en de bodem plus de wet bij `x_new`."},
            "wrong": {"en": _CALL + ". The line through the logs of `x` and of `loss - floor`, and then `floor + 10 ** b * x_new ** k`?",
                      "nl": _CALL_NL + ". De lijn door de logs van `x` en van `loss - floor`, en dan `floor + 10 ** b * x_new ** k`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `x_new`, `x`, `loss` and `floor`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `x_new`, `x`, `loss` en `floor`, en geen vaste getallen?"},
            "correct": {"en": "Your function predicts the loss of a model you have not trained yet, from the law through the runs you have.",
                        "nl": "Je functie voorspelt de loss van een model dat je nog niet trainde, uit de wet door de runs die je hebt."},
        },
    ),
    "compute-optimal": FunctionBox(
        name="compute-optimal",
        function="compute_optimal",
        start=_C_START,
        solution=_C_BODY,
        visible=[(_Named("6e18"), 20)],
        hidden=[(_Named("1.2e21"), 20), (_Named("6e12"), 8), (_Named("2.4e15"), 1)],
        show=_rounded,
        compare=_close_whole,
        save_as=("llm", "compute_optimal"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-root", wrong=_C_BODY.replace("(compute / (6 * tokens_per_param)) ** 0.5", "compute / (6 * tokens_per_param)"),
                    en=_CALL + ": you forgot the square root. With `tokens = tokens_per_param * params`, the compute is `6 * tokens_per_param * params ** 2`: the parameters are squared.",
                    nl=_CALL_NL + ": je vergat de wortel. Met `tokens = tokens_per_param * params` is de compute `6 * tokens_per_param * params ** 2`: de parameters staan in het kwadraat."),
            Mistake("no-six", wrong=_C_BODY.replace("(6 * tokens_per_param)", "tokens_per_param"),
                    en=_CALL + ": a training run costs about 6 FLOPs per parameter and token, as in the rule 6ND. Divide by `6 * tokens_per_param`.",
                    nl=_CALL_NL + ": een trainingsrun kost ongeveer 6 FLOPs per parameter en token, zoals in de regel 6ND. Deel door `6 * tokens_per_param`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the parameters from the compute and the tokens per parameter.",
                      "nl": "Vervang de `...`: de parameters uit de compute en de tokens per parameter."},
            "wrong": {"en": _CALL + ". `compute = 6 * params * tokens` with `tokens = tokens_per_param * params`, solved for `params`?",
                      "nl": _CALL_NL + ". `compute = 6 * params * tokens` met `tokens = tokens_per_param * params`, opgelost naar `params`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `compute` and `tokens_per_param`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `compute` en `tokens_per_param`, en geen vaste getallen?"},
            "correct": {"en": "Your function splits a compute budget into parameters and tokens. It is in your project now, as `{saved}`: the capstone sizes its model with it.",
                        "nl": "Je functie verdeelt een compute budget over parameters en tokens. Hij staat nu in je project, als `{saved}`: de capstone bepaalt er de grootte van zijn model mee."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("best-floor", _F_START, "empty", None),
    ("best-floor", _f(), "correct", None),
    ("best-floor", _f(keep=_F_KEEP.replace("error < smallest", "error > smallest")), "wrong", "largest"),
    ("best-floor", _f(fit=_F_FIT.replace("np.array(loss) - floor", "np.array(loss)")), "wrong", "first"),
    ("best-floor", _f(keep=_F_KEEP.replace("    return best\n", "        return best\n")), "wrong", "first"),
    ("best-floor", _f(err=_F_ERR.replace("np.mean(", "np.sum(")), "correct", None),
    ("best-floor", _f(keep="        best = floor\n    return best\n"), "wrong", "last"),
    ("best-floor", _f(keep=_F_KEEP.replace("    return best\n", "    return smallest\n")), "wrong", "error-value"),
    ("predict", _P_START, "empty", None),
    ("predict", _P_BODY, "correct", None),
    ("predict", _P_BODY.replace("    return floor + 10 ** b * x_new ** k\n", "    return 10 ** b * x_new ** k\n"), "wrong", "no-floor-back"),
    ("predict", _P_BODY.replace("10 ** b * x_new ** k", "b * x_new ** k"), "wrong", "b-as-a"),
    ("predict", _P_BODY.replace("    k, b = np.polyfit", "    b, k = np.polyfit"), "wrong", "swapped"),
    ("predict", _P_BODY.replace("np.log10(np.array(loss) - floor)", "np.log10(loss)"), "wrong", "no-floor-fit"),
    ("compute-optimal", _C_START, "empty", None),
    ("compute-optimal", _C_BODY, "correct", None),
    ("compute-optimal", _C_BODY.replace("(compute / (6 * tokens_per_param)) ** 0.5", "(compute / 6 / tokens_per_param) ** 0.5"), "correct", None),
    ("compute-optimal", _C_BODY.replace("return params, tokens", "return round(params), round(tokens)"), "correct", None),
    ("compute-optimal", _C_BODY.replace("(compute / (6 * tokens_per_param)) ** 0.5", "compute / (6 * tokens_per_param)"), "wrong", "no-root"),
    ("compute-optimal", _C_BODY.replace("(6 * tokens_per_param)", "tokens_per_param"), "wrong", "no-six"),
]
