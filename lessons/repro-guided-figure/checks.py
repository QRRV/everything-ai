# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-figure: log_steps, the steps at which a run logs, saved as repro/log.py;
run_seed, one run of the reproduction that logs the accuracy on all training and all validation equations, saved as
repro/run.py; run_seeds, the multi-seed runner that adds each run to an experiment log and runs no seed twice, saved
as repro/seeds.py; and the paper's figure, drawn again from logs. The course's own runs of three seeds are in
data/course-grokking.jsonl, made with this lesson's run_seed; the demo trains seeds 0, 1 and 2 with the learner's kit."""
import functools
import importlib.util
import json
import re

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, PlotBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _checks("repro-guided-build", "everything_ai_checks_repro_guided_build_for_figure")
P = _checks("repro-guided-claim", "everything_ai_checks_repro_guided_claim_for_figure")
PAPER, FRACTION, number = B.PAPER, B.FRACTION, B.number
THRESHOLD, DELAY = P.THRESHOLD, P.DELAY
# The kit so far, and the model it adapts: where a module is missing or broken, the course's version stands in.
BUILDS_ON = B.BUILDS_ON + (("repro", "data", "repro-guided-build", "dataset"), ("repro", "settings", "repro-guided-build", "settings"),
                           ("repro", "log", "repro-guided-figure", "log-steps"), ("repro", "run", "repro-guided-figure", "run-seed"))
projects.builds_on(*BUILDS_ON)

STEPS = 12_000   # the steps of each run: half again the step at which validation reached 99% in repro-guided-build's run
PER_FACTOR = 20  # logged steps per factor of 10
SEEDS = [0, 1, 2]
RUNS_FILE = paths.ROOT / "data" / "course-grokking.jsonl"

# --- measured: the course's runs of data/course-grokking.jsonl (data/tools/record_grokking.py), one thread on the
# course's CPU, and seed 0 again at its logged step MIRROR["step"] (data/tools/record_grokking.py --mirror): the accuracy on the
# validation equations whose mirror b + a is in the training data, and on the others ---
SECONDS_PER_STEP = 0.113
MIRROR = {"step": 1995, "share": 0.2951, "mirrored": 0.9846, "others": 0.0006}
# --- end of measured ---


@functools.cache
def course_runs() -> list[dict]:
    """The course's runs of the seeds 0, 1 and 2, as run_seed gives them: the paper's settings at FRACTION, STEPS steps."""
    return [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines()]


@functools.cache
def _plan() -> dict:
    names = {}
    exec(P.EXERCISES["verdict"].solution, names)   # the plan of repro-guided-claim: first_step and verdict
    return names


def read(log: dict) -> dict:
    """What the plan reads in one log: the first logged steps t and v at which training and validation reach 99%,
    their places in the log (-1 when never), how many times as late v is, and the verdict."""
    plan = _plan()
    t = plan["first_step"](log["step"], log["train"], THRESHOLD)
    v = plan["first_step"](log["step"], log["val"], THRESHOLD)
    return {"t": t, "v": v, "t_index": log["step"].index(t) if t is not None else -1,
            "v_index": log["step"].index(v) if v is not None else -1,
            "times": v / t if t and v else None, "verdict": plan["verdict"](log["step"], log["train"], log["val"])}


def mirrored(train: list, val: list) -> float:
    """The share of the validation equations whose mirror, b + a, is in the training data."""
    seen = {(a, b) for a, b, _ in train}
    return sum((b, a) in seen for a, b, _ in val) / len(val)


@functools.cache
def mirror_shares() -> list[float]:
    """mirrored for the split of each of SEEDS, at FRACTION: the data of the course's runs."""
    dataset = _dataset()
    return [mirrored(*dataset(PAPER["p"], FRACTION, seed)) for seed in SEEDS]


def _dataset():
    names = {}
    exec(B.EXERCISES["dataset"].solution, names)
    return names["dataset"]


def plateau(log: dict, low: int = 500, high: int = 4000) -> tuple:
    """The lowest and the highest validation accuracy between the steps low and high: the flat part of the curve."""
    vals = [a for s, a in zip(log["step"], log["val"]) if low <= s <= high]
    return min(vals), max(vals)


def figure_picture(runs: list[dict], labels: dict, width: int = 460) -> str:
    """Accuracy against the steps on a log axis, as in the paper's Figure 1: for each run, training solid and
    validation dashed, both ink (accuracy is not what the model optimises), and a thin line at 99%."""
    from everything_ai import chart
    p = chart.Panel((1, STEPS), (0, 1.08), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True,
                    yticks=[0, 0.5, 1.0])
    p.line([1, STEPS], [THRESHOLD, THRESHOLD], "ink-3", 1.0)
    for log in runs:
        p.line(log["step"], log["train"], "ink", 1.6)
        p.line(log["step"], log["val"], "ink", 1.6, dashed=True)
    p.text(6, 0.62, labels["train"], "ink", "end")     # left of the training curves, which pass 0.62 near step 100
    p.text(500, 0.42, labels["val"], "ink", "middle")  # above the flat part of validation, at about 0.3
    return chart.figure([p], alt=labels["alt"], width=width, height=300)


def _close(got, want) -> bool:
    """Numbers, also inside lists, equal to 4 decimals; anything else equal."""
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(got, bool) or isinstance(want, bool):
        return got == want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return abs(got - want) <= 1e-4 + 1e-4 * abs(want)
    return got == want


_DOTS = B._DOTS
_KEEP = B._KEEP

# --- exercise 1: log_steps ---------------------------------------------------------------------------------------------

_L_HEAD = ("def log_steps(budget, per_factor):\n"
           "    # the steps at which a run logs: 10 ** (k / per_factor), rounded, for k = 0, 1, 2, ... as long as that is\n"
           "    # not above the budget, each step once; and the budget itself, the last step\n"
           "    steps = []\n    k = 0\n")
_L_LOOP = ("    while round(10 ** (k / per_factor)) <= budget:\n        step = round(10 ** (k / per_factor))\n"
           "        if step not in steps:\n            steps.append(step)\n        k = k + 1\n")
_L_LAST = "    if budget not in steps:\n        steps.append(budget)\n"
_L_TAIL = "    return steps\n"
_L_START = _L_HEAD + "    ...\n" + _L_TAIL


def _l(loop: str = _L_LOOP, last: str = _L_LAST) -> str:
    return _L_HEAD + loop + last + _L_TAIL


_L = _l()
_L_TWICE = _l(loop=_L_LOOP.replace("        if step not in steps:\n            steps.append(step)\n", "        steps.append(step)\n"))
_L_FLOOR = _l(loop=_L_LOOP.replace("round(", "int("))
_L_WHOLE = _l(loop=_L_LOOP.replace("k / per_factor", "k // per_factor"))
_L_NO_LAST = _l(last="")
_L_INT_STOP = _l(loop=_L_LOOP.replace("while round(", "while int("))   # right for (100, 5), one step too many for (5, 20)
_L_EVERY = _L_HEAD + "    steps = list(range(per_factor, budget + 1, per_factor))\n" + _L_LAST + _L_TAIL

# --- exercise 2: run_seed ----------------------------------------------------------------------------------------------

_R_IMPORTS = ("import torch\nimport torch.nn.functional as F\n\nfrom llm.modern_gpt import ModernGPT\nfrom repro.data import dataset\n"
              "from repro.log import log_steps\n\n\n"
              "def tokens(rows, p):\n"
              "    # the equations as the model reads them: a, the operation (token p), b and = (token p + 1); and the answers c\n"
              "    x = torch.tensor([[a, p, b, p + 1] for a, b, _ in rows])\n    y = torch.tensor([c for _, _, c in rows])\n"
              "    return x, y\n\n\n"
              "def accuracy(model, x, y):\n"
              "    # the share of the equations whose guess at the last place, the token with the highest logit, is the answer\n")
_R_ACC = ("    with torch.no_grad():\n        guess = model(x)[:, -1].argmax(-1)\n    return (guess == y).float().mean().item()\n")
_R_RUN = ("\n\ndef run_seed(settings, seed, steps, device=\"cpu\", progress=None):\n"
          "    # one run with this seed, which fixes the split, the starting weights and the batches; it logs the accuracy\n"
          "    # on all training and all validation equations after each step of log_steps(steps, 20)\n"
          "    s = settings\n    train, val = dataset(s[\"p\"], s[\"fraction\"], seed)\n"
          "    xt, yt = tokens(train, s[\"p\"])\n    xv, yv = tokens(val, s[\"p\"])\n"
          "    xt, yt, xv, yv = xt.to(device), yt.to(device), xv.to(device), yv.to(device)\n"
          "    torch.manual_seed(seed)\n"
          "    model = ModernGPT(s[\"p\"] + 2, s[\"width\"], s[\"heads\"], s[\"heads\"], s[\"layers\"]).to(device)\n"
          "    opt = torch.optim.AdamW(model.parameters(), lr=s[\"lr\"], weight_decay=s[\"weight_decay\"], betas=s[\"betas\"])\n"
          "    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / s[\"warmup\"]))\n"
          "    batch = min(s[\"batch\"], len(train) // 2)   # A.1.2: 512, or half the training data when that is smaller\n"
          "    batches = torch.Generator().manual_seed(seed)\n"
          "    at = set(log_steps(steps, 20))\n"
          "    log = {\"seed\": seed, \"step\": [], \"train\": [], \"val\": []}\n"
          "    for step in range(1, steps + 1):\n"
          "        rows = torch.randint(0, len(train), (batch,), generator=batches).to(device)\n"
          "        loss = F.cross_entropy(model(xt[rows])[:, -1], yt[rows])\n"
          "        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        warmup.step()\n"
          "        # after each step in at: the step, and the accuracy on all training and all validation equations\n")
_R_LOG = ("        if step in at:\n            log[\"step\"].append(step)\n            log[\"train\"].append(accuracy(model, xt, yt))\n"
          "            log[\"val\"].append(accuracy(model, xv, yv))\n")
_R_TAIL = ("        if progress is not None and step in at:\n            progress(log)   # the course's: it shows the run while it trains\n"
           "    return log\n")
_R_START = _R_IMPORTS + "    ...\n" + _R_RUN + "        ...\n" + _R_TAIL


def _r(acc: str = _R_ACC, log: str = _R_LOG) -> str:
    return _R_IMPORTS + acc + _R_RUN + log + _R_TAIL


_R = _r()
_R_FIRST = _r(acc=_R_ACC.replace("[:, -1]", "[:, 0]"))
_R_TENSOR = _r(acc=_R_ACC.replace(".mean().item()", ".mean()"))
_R_PERCENT = _r(acc=_R_ACC.replace(".mean().item()", ".mean().item() * 100"))
_R_COUNT = _r(acc=_R_ACC.replace(".float().mean().item()", ".sum().item()"))
_R_BATCH = _r(log=_R_LOG.replace("accuracy(model, xt, yt)", "accuracy(model, xt[rows], yt[rows])"))
_R_LOSS = _r(log=_R_LOG.replace("accuracy(model, xt, yt)", "loss.item()"))
_R_SWAP = _r(log=_R_LOG.replace("accuracy(model, xt, yt)", "TRAIN").replace("accuracy(model, xv, yv)", "accuracy(model, xt, yt)").replace("TRAIN", "accuracy(model, xv, yv)"))
_R_EVERY = _r(log="        log[\"step\"].append(step)\n        log[\"train\"].append(accuracy(model, xt, yt))\n"
                  "        log[\"val\"].append(accuracy(model, xv, yv))\n")

_R_SMALL = ("SMALL = {\"p\": 7, \"layers\": 1, \"width\": 16, \"heads\": 2, \"lr\": 0.01, \"weight_decay\": 1.0, \"betas\": (0.9, 0.98),\n"
            "         \"warmup\": 3, \"batch\": 512, \"steps\": 100, \"fraction\": 0.5}\n")
_R_SHOW = "\n[log[\"seed\"], log[\"step\"], [round(a, 4) for a in log[\"train\"]], [round(a, 4) for a in log[\"val\"]]]"
_R_SCENARIOS = [
    # the lesson's example: a small model on addition modulo 7, seed 0, 30 steps
    _R_SMALL + "log = run_seed(SMALL, 0, 30)" + _R_SHOW,
    # another seed: another split, other starting weights and other batches
    _R_SMALL + "log = run_seed(SMALL, 3, 45)" + _R_SHOW,
    # accuracy alone: a model that guesses 2 at the last place, and 5 at the first, on answers of which 3 out of 5 are 2
    "from torch import nn\n\n\nclass Twos(nn.Module):\n    def forward(self, x):\n"
    "        out = torch.zeros(x.shape[0], x.shape[1], 9)\n        out[:, :, 2] = 1.0\n        out[:, 0, 5] = 9.0\n        return out\n\n\n"
    "x = torch.tensor([[1, 7, 1, 8], [0, 7, 2, 8], [4, 7, 3, 8], [3, 7, 6, 8], [6, 7, 5, 8]])\n"
    "y = torch.tensor([2, 2, 0, 2, 4])\nround(accuracy(Twos(), x, y), 4)",
    # the steps it logs come from log_steps, and progress sees the log at each of them
    _R_SMALL + "seen = []\nlog = run_seed(SMALL, 1, 12, progress=lambda log: seen.append(log[\"step\"][-1]))\n[seen, log[\"step\"]]",
]
_R_EN, _R_NL = "Your `run_seed` gives another log than the course's", "Je `run_seed` geeft een ander logboek dan die van de course"

# --- exercise 3: run_seeds ---------------------------------------------------------------------------------------------

_M_HEAD = ("import json\n\nfrom repro.run import run_seed\n\n\n"
           "def run_seeds(settings, seeds, steps, path, device=\"cpu\", progress=None):\n"
           "    # one run of run_seed for each seed, each added to the experiment log at path as one line of JSON; a seed\n"
           "    # that is in the log already is not run again, so a stopped run of several seeds goes on where it stopped\n"
           "    with open(path, \"a\") as f:   # \"a\" makes the log when there is none yet, and keeps what is in it\n"
           "        f.write(\"\")\n"
           "    done = [json.loads(line)[\"seed\"] for line in open(path)]\n")
_M_LOOP = ("    for seed in seeds:\n        if seed not in done:\n"
           "            log = run_seed(settings, seed, steps, device, progress)\n")
_M_WRITE = "            with open(path, \"a\") as f:\n                f.write(json.dumps(log) + \"\\n\")\n"
_M_MAIN = ""
_M_START = _M_HEAD + "    ...\n" + _M_MAIN


def _m(loop: str = _M_LOOP, write: str = _M_WRITE) -> str:
    return _M_HEAD + loop + write + _M_MAIN


_M = _m()
_M_OVERWRITE = _m(write=_M_WRITE.replace("open(path, \"a\")", "open(path, \"w\")"))
_M_NO_LINE = _m(write=_M_WRITE.replace(" + \"\\n\"", ""))
_M_STR = _m(write=_M_WRITE.replace("json.dumps(log)", "str(log)"))
_M_AGAIN = _m(loop=_M_LOOP.replace("        if seed not in done:\n", "        if True:\n"))
_M_ONLY_DONE = _m(loop=_M_LOOP.replace("if seed not in done:", "if seed in done:"))
_M_NO_DEVICE = _m(loop=_M_LOOP.replace("run_seed(settings, seed, steps, device, progress)", "run_seed(settings, seed, steps, progress=progress)"))
_M_NO_PROGRESS = _m(loop=_M_LOOP.replace("run_seed(settings, seed, steps, device, progress)", "run_seed(settings, seed, steps, device)"))

_M_FAKE = ("import tempfile\nfrom pathlib import Path\n\ncalls = []\ndevices = []\n\n\n"
           "def run_seed(settings, seed, steps, device=\"cpu\", progress=None):   # a quick stand-in: no training\n"
           "    calls.append(seed)\n    devices.append(device)\n    log = {\"seed\": seed, \"step\": [1, steps], \"train\": [0.5, 1.0], \"val\": [0.0, seed / 10]}\n"
           "    if progress is not None:\n        progress(log)\n    return log\n\n\n"
           "path = str(Path(tempfile.mkdtemp()) / \"runs.jsonl\")\n")
_M_READ = "\n[calls, [json.loads(line) for line in open(path)] if Path(path).exists() else \"no file\"]"
_M_SCENARIOS = [
    # the lesson's example: two seeds into a new log
    _M_FAKE + "run_seeds({}, [0, 1], 10, path)" + _M_READ,
    # a log that has seed 1 already: only 0 and 2 are run, and seed 1's line stays
    _M_FAKE + "with open(path, \"w\") as f:\n    f.write(json.dumps({\"seed\": 1, \"step\": [1], \"train\": [1.0], \"val\": [1.0]}) + \"\\n\")\n"
              "run_seeds({}, [0, 1, 2], 10, path)" + _M_READ,
    # the same seed twice: the second time it is done already
    _M_FAKE + "run_seeds({}, [0], 10, path)\nrun_seeds({}, [0, 3], 10, path)" + _M_READ,
    # device and progress go on to run_seed
    _M_FAKE + "seen = []\nrun_seeds({}, [2], 7, path, \"cuda\", lambda log: seen.append(log[\"seed\"]))\n[seen, devices]",
]
_M_EN, _M_NL = "Your `run_seeds` gives another experiment log than the course's", "Je `run_seeds` geeft een ander logboek dan die van de course"

# --- exercise 4: the figure --------------------------------------------------------------------------------------------

_F_HEAD = ("import json\n\nimport matplotlib.pyplot as plt\n\n"
           "runs = [json.loads(line) for line in open(\"data/course-grokking.jsonl\")]   # the course's runs of seeds 0, 1 and 2\n\n"
           "for log in runs:\n")
_F_LINES = ("    plt.plot(log[\"step\"], log[\"train\"], \"-\")\n    plt.plot(log[\"step\"], log[\"val\"], \"--\")\n")
_F_SCALE = "plt.xscale(\"log\")\n"
_F_LEGEND = "plt.legend([\"training\", \"validation\"])   # names the first two lines: those of the first run\n"
_F_TITLE = "\"In all three runs, validation reaches 99% 45 to 50 times as late as training\""
_F_X, _F_Y = "\"step\"", "\"accuracy\""


def _f(lines: str = _F_LINES, scale: str = _F_SCALE, title: str = _F_TITLE, x: str = _F_X, y: str = _F_Y) -> str:
    return _F_HEAD + lines + scale + _F_LEGEND + f"plt.title({title})\nplt.xlabel({x})\nplt.ylabel({y})\nplt.show()\n"


_F = _f()
_F_START = (_F_HEAD + "    # training solid and validation dashed, against the steps\n    ...\n"
            "# the x-axis of Figure 1\n...\n" + _F_LEGEND + "plt.title(\"\")\nplt.xlabel(\"\")\nplt.ylabel(\"\")\nplt.show()\n")
_F_LINEAR = _f(scale="")
_F_SAME = _f(lines="    plt.plot(log[\"step\"], log[\"train\"])\n    plt.plot(log[\"step\"], log[\"val\"])\n")
_F_ORDER = _f(lines="    plt.plot(log[\"step\"], log[\"val\"], \"--\")\n    plt.plot(log[\"step\"], log[\"train\"], \"-\")\n")
_F_PLACES = _f(lines="    plt.plot(log[\"train\"], \"-\")\n    plt.plot(log[\"val\"], \"--\")\n")
_F_VAL = _f(lines="    plt.plot(log[\"step\"], log[\"val\"], \"--\")\n")


def _lines_of(got) -> list:
    return [c for c in (got or {}).get("charts", []) if c["kind"] == "line"]


def _all_lines_drawn(got) -> bool:
    """Every training and validation curve of the course's runs is among the lines, against its steps."""
    want = [(log["step"], log[k]) for log in course_runs() for k in ("train", "val")]
    have = [(c["x"], c["y"]) for c in _lines_of(got)]
    return all(any(_close(x, wx) and _close(y, wy) for x, y in have) for wx, wy in want)


def _first_two_wrong(got, text) -> bool:
    """All curves are there, and the given legend names the first two lines, but they are not training and then
    validation of the first run. A legend from labels of the learner's own names the right lines in any order."""
    lines, first = _lines_of(got), course_runs()[0]
    return (re.search(r"legend\(\s*\[", text) is not None and _all_lines_drawn(got) and len(lines) >= 2
            and not (_close(lines[0]["y"], first["train"]) and _close(lines[1]["y"], first["val"])))


def _against_places(got, text) -> bool:
    """The curves are drawn against their places in the log, 0, 1, 2, ..., not against the steps."""
    lines, first = _lines_of(got), course_runs()[0]
    return any(c["x"] == list(range(len(first["step"]))) and _close(c["y"], first["train"]) for c in lines)


def _linear(got, text) -> bool:
    return _all_lines_drawn(got) and got["xscale"] != "log"


def _drawn(text: str) -> list:
    """(colour, line style, marker, y values) of every line the code draws on its first chart, in the order drawn."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    saved, out = dict(vars(plt)), []
    plt.close("all")
    try:
        with matplotlib.rc_context():
            plt.show = lambda *args, **kwargs: None
            run = run_code(text, keep_figures=True)
            fig = plt.gcf()
            if run.error is None and not run.timeout and fig.axes:
                out = [(to_hex(line.get_color()), str(line.get_linestyle()), str(line.get_marker()), [float(y) for y in line.get_ydata()])
                       for line in fig.axes[0].get_lines()]
    finally:
        for name in [n for n in vars(plt) if n not in saved]:
            delattr(plt, name)
        for name, value in saved.items():
            setattr(plt, name, value)
        plt.close("all")
    return out


def _not_told_apart(got, text) -> bool:
    """Training and validation have no look of their own: no colour, line style or marker is the same for every
    training curve, the same for every validation curve, and different between the two."""
    looks = {"train": [], "val": []}
    for colour, style, marker, ys in _drawn(text):
        for log in course_runs():
            role = "train" if _close(ys, log["train"]) else "val" if _close(ys, log["val"]) else None
            if role:
                looks[role].append((colour, style, marker))
                break
    if not looks["train"] or not looks["val"]:
        return False
    for i in range(3):
        train, val = {look[i] for look in looks["train"]}, {look[i] for look in looks["val"]}
        if len(train) == 1 and len(val) == 1 and train != val:
            return False
    return True


# --- exercise 5: the plateau, a choice question in the lesson ------------------------------------------------------------

EXERCISES = {
    "log-steps": FunctionBox(
        name="log-steps",
        function="log_steps",
        start=_L_START,
        solution=_L,
        visible=[(100, 5)],
        hidden=[(150, 5), (1000, 10), (5, 20), (1, 20), (200, 20), (12_000, 20)],
        save_as=("repro", "log"),
        first=[_DOTS],
        mistakes=[
            Mistake("twice", wrong=_L_TWICE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: some steps are in your list more than once. At the start, several values of `k` round to the same step: add a step only when it is not in `steps` yet.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: sommige stappen staan meer dan eens in je lijst. In het begin ronden meerdere waarden van `k` af naar dezelfde stap: voeg een stap alleen toe als hij nog niet in `steps` staat."),
            Mistake("floor", wrong=_L_FLOOR,
                    en="`{call}` gives `{got}`, and it should give `{want}`: `int` cuts off everything after the point, so 3.98 becomes 3. Round to the nearest whole step with `round`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `int` hakt alles na de komma eraf, dus 3,98 wordt 3. Rond af naar de dichtstbijzijnde hele stap met `round`."),
            Mistake("whole", wrong=_L_WHOLE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: `k // per_factor` is a whole number, so you get only 1, 10, 100, and so on. Divide with `/`: `10 ** (k / per_factor)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `k // per_factor` is een heel getal, dus krijg je alleen 1, 10, 100, enzovoort. Deel met `/`: `10 ** (k / per_factor)`."),
            Mistake("no-last", wrong=_L_NO_LAST,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your list stops before the last step. A run logs its last step too, so that the figure ends where the run ends: add `budget` when it is not in `steps` yet.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je lijst stopt vóór de laatste stap. Een run logt ook zijn laatste stap, zodat de figuur eindigt waar de run eindigt: voeg `budget` toe als hij nog niet in `steps` staat."),
            Mistake("every", wrong=_L_EVERY,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps are evenly spaced, every `per_factor` steps. On a log axis those all crowd at the right, with almost none at the start. The steps are `10 ** (k / per_factor)`, rounded.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen liggen op gelijke afstand, om de `per_factor` stappen. Op een logaritmische as dringen die zich allemaal rechts samen, met bijna geen in het begin. De stappen zijn `10 ** (k / per_factor)`, afgerond."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...` with a `while` loop over `k` that adds each new step to `steps`, and then add `budget` when it is not there yet.",
                      "nl": "Vervang de `...` door een `while`-loop over `k` die elke nieuwe stap aan `steps` toevoegt, en voeg daarna `budget` toe als die er nog niet in staat."},
            "arguments": {"en": "The check calls `{call}`, with two numbers, and the `def` line of your `log_steps` has another number of names between its parentheses. Keep `(budget, per_factor)`.",
                          "nl": "De check roept `{call}` aan, met twee getallen, en de `def`-regel van je `log_steps` heeft een ander aantal namen tussen de haakjes. Houd `(budget, per_factor)`."},
            "almost": {"en": "Your `log_steps` gives the right steps for `log_steps(100, 5)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your loop stop at the budget for every budget and every `per_factor`, add each rounded step once, and end with `budget`?",
                       "nl": "Je `log_steps` geeft de goede stappen voor `log_steps(100, 5)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Stopt je loop bij het budget voor elk budget en elke `per_factor`, voegt hij elke afgeronde stap één keer toe, en eindigt hij met `budget`?"},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your loop go on as long as the rounded step is not above `budget`, does it add each step once, and is `budget` the last?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Gaat je loop door zolang de afgeronde stap niet boven `budget` ligt, voegt hij elke stap één keer toe, en is `budget` de laatste?"},
            "correct": {"en": "Your `log_steps` gives steps that lie evenly on a log axis: as many between 10 and 100 as between 1,000 and 10,000. It is now in your project, as `work/projects/repro/log.py`.",
                        "nl": "Je `log_steps` geeft stappen die gelijk verdeeld op een logaritmische as liggen: evenveel tussen 10 en 100 als tussen 1.000 en 10.000. Hij staat nu in je project, als `work/projects/repro/log.py`."},
            "kept": {"en": "Your `log_steps` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `log_steps` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "run-seed": ClassBox(
        name="run-seed",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("repro", "run"),
        first=[_DOTS],
        mistakes=[
            Mistake("keep-tensor", error=lambda e: isinstance(e, TypeError) and "Tensor" in str(e) and "__round__" in str(e),
                    en="Your log holds tensors, not numbers: `accuracy` gives back a tensor. Give back the number in it, with `.item()`: a log is written to a file as text, and a tensor cannot be.",
                    nl="Je logboek bevat tensors, geen getallen: `accuracy` geeft een tensor terug. Geef het getal erin terug, met `.item()`: een logboek wordt als tekst naar een bestand geschreven, en een tensor kan dat niet."),
            Mistake("first-place", wrong=_R_FIRST,
                    en=_R_EN + ": your `accuracy` reads the guess at the first place, `[:, 0]`, where the model has seen only `a`. Its guess for `c` is at the last place, that of `=`: `model(x)[:, -1]`.",
                    nl=_R_NL + ": je `accuracy` leest de gok op de eerste plek, `[:, 0]`, waar het model alleen `a` heeft gezien. Zijn gok voor `c` staat op de laatste plek, die van `=`: `model(x)[:, -1]`."),
            Mistake("percent", wrong=_R_PERCENT,
                    en=_R_EN + ": your accuracy is a percentage, from 0 to 100. The plan of your kit compares it with `THRESHOLD = 0.99`, a share: leave out the `* 100`.",
                    nl=_R_NL + ": je accuracy is een percentage, van 0 tot 100. Het plan van je kit vergelijkt hem met `THRESHOLD = 0.99`, een aandeel: laat de `* 100` weg."),
            Mistake("count", wrong=_R_COUNT,
                    en=_R_EN + ": your `accuracy` counts the right guesses. It is a share: the mean of the right guesses as 1 and 0, `(guess == y).float().mean()`.",
                    nl=_R_NL + ": je `accuracy` telt de goede gokken. Hij is een aandeel: het gemiddelde van de goede gokken als 1 en 0, `(guess == y).float().mean()`."),
            Mistake("batch", wrong=_R_BATCH,
                    en=_R_EN + ": your training accuracy is measured on the batch of the step, 512 equations drawn at random, so it jumps with the batch. Measure it on all training equations: `accuracy(model, xt, yt)`.",
                    nl=_R_NL + ": je training accuracy is gemeten op de batch van de stap, 512 willekeurig getrokken vergelijkingen, dus springt hij mee met de batch. Meet hem op alle trainingsvergelijkingen: `accuracy(model, xt, yt)`."),
            Mistake("loss", wrong=_R_LOSS,
                    en=_R_EN + ": `train` holds the loss of the batch, and the figure shows the accuracy on all training equations: `accuracy(model, xt, yt)`.",
                    nl=_R_NL + ": `train` bevat de loss van de batch, en de figuur laat de accuracy op alle trainingsvergelijkingen zien: `accuracy(model, xt, yt)`."),
            Mistake("swap", wrong=_R_SWAP,
                    en=_R_EN + ": `train` and `val` are swapped. `train` is the accuracy on the training equations, `xt` and `yt`; `val` on the validation equations, `xv` and `yv`.",
                    nl=_R_NL + ": `train` en `val` zijn omgewisseld. `train` is de accuracy op de trainingsvergelijkingen, `xt` en `yt`; `val` op de validatievergelijkingen, `xv` en `yv`."),
            Mistake("every", wrong=_R_EVERY,
                    en=_R_EN + ": it logs after every step. Each log measures all equations, 9,409 of them: log only after the steps in `at`, with `if step in at:`.",
                    nl=_R_NL + ": hij logt na elke stap. Elke keer loggen meet alle vergelijkingen, 9.409 stuks: log alleen na de stappen in `at`, met `if step in at:`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace both `...`: in `accuracy`, the share of right guesses at the last place; in the loop, after each step in `at`, the step and the two accuracies, added to `log`.",
                      "nl": "Vervang beide `...`: in `accuracy` het aandeel goede gokken op de laatste plek; in de loop, na elke stap in `at`, de stap en de twee accuracy's, toegevoegd aan `log`."},
            "wrong": {"en": _R_EN + ". Does `accuracy` take the guess at the last place with `argmax`, and give back the share that is right as a number? And does the loop, after each step in `at`, add the step and the accuracy on all training and all validation equations?",
                      "nl": _R_NL + ". Neemt `accuracy` de gok op de laatste plek met `argmax`, en geeft hij het aandeel dat goed is terug als getal? En voegt de loop na elke stap in `at` de stap en de accuracy op alle trainings- en alle validatievergelijkingen toe?"},
            "almost": {"en": "Your `run_seed` gives the course's log for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Are the given lines as they were given, and does `accuracy` work for any model and any equations?",
                       "nl": "Je `run_seed` geeft het logboek van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Staan de gegeven regels er nog zoals ze gegeven waren, en werkt `accuracy` voor elk model en alle vergelijkingen?"},
            "correct": {"en": "Your `run_seed` trains one run of the reproduction and logs its accuracy on all training and all validation equations at the steps of `log_steps`. It is now in your project, as `work/projects/repro/run.py`.",
                        "nl": "Je `run_seed` traint één run van de reproductie en logt zijn accuracy op alle trainings- en alle validatievergelijkingen op de stappen van `log_steps`. Hij staat nu in je project, als `work/projects/repro/run.py`."},
            "kept": {"en": "Your `run_seed` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `run_seed` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "run-seeds": ClassBox(
        name="run-seeds",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        compare=_close,
        save_as=("repro", "seeds"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-line", error=lambda e: type(e).__name__ == "JSONDecodeError" and "Extra data" in str(e),
                    en="Reading your experiment log back stops: two runs stand on one line, because each line needs a newline at its end. Write `json.dumps(log) + \"\\n\"`.",
                    nl="Je logboek teruglezen stopt: twee runs staan op één regel, omdat elke regel aan het eind een regeleinde nodig heeft. Schrijf `json.dumps(log) + \"\\n\"`."),
            Mistake("str", error=lambda e: type(e).__name__ == "JSONDecodeError" and "double quotes" in str(e),
                    en="Reading your experiment log back stops: a line of it is not JSON. It has single quotes, the way Python shows a dictionary, and JSON needs double quotes. Turn the log into a line with `json.dumps(log)`.",
                    nl="Je logboek teruglezen stopt: een regel ervan is geen JSON. Hij heeft enkele aanhalingstekens, zoals Python een dictionary laat zien, en JSON heeft dubbele nodig. Maak van het logboek een regel met `json.dumps(log)`."),
            Mistake("overwrite", wrong=_M_OVERWRITE,
                    en=_M_EN + ": only the last run is in it. `open(path, \"w\")` empties the file first, so each run wipes out the one before. Add to it with `\"a\"`.",
                    nl=_M_NL + ": alleen de laatste run staat erin. `open(path, \"w\")` maakt het bestand eerst leeg, dus wist elke run de vorige uit. Voeg eraan toe met `\"a\"`."),
            Mistake("again", wrong=_M_AGAIN,
                    en=_M_EN + ": a seed that is in the log already is run again, and then stands in it twice. Run a seed only when it is not done yet: `if seed not in done:`.",
                    nl=_M_NL + ": een seed die al in het logboek staat, wordt opnieuw gedraaid, en staat er dan twee keer in. Draai een seed alleen als hij nog niet klaar is: `if seed not in done:`."),
            Mistake("only-done", wrong=_M_ONLY_DONE,
                    en=_M_EN + ": it runs only the seeds that are done already, and never the new ones. Turn the test around: `if seed not in done:`.",
                    nl=_M_NL + ": hij draait alleen de seeds die al klaar zijn, en nooit de nieuwe. Draai de test om: `if seed not in done:`."),
            Mistake("no-device", wrong=_M_NO_DEVICE, state="almost",
                    en="Your `run_seeds` does not pass `device` on to `run_seed`, so on a computer with a GPU your runs would still train on the CPU. Give it along: `run_seed(settings, seed, steps, device, progress)`.",
                    nl="Je `run_seeds` geeft `device` niet door aan `run_seed`, dus op een computer met een GPU zouden je runs toch op de CPU trainen. Geef hem mee: `run_seed(settings, seed, steps, device, progress)`."),
            Mistake("no-progress", wrong=_M_NO_PROGRESS, state="almost",
                    en="Your `run_seeds` does not pass `progress` on to `run_seed`, so the lesson cannot show the run while it trains. Give it along: `run_seed(settings, seed, steps, device, progress)`.",
                    nl="Je `run_seeds` geeft `progress` niet door aan `run_seed`, dus kan de les de run niet laten zien terwijl hij traint. Geef hem mee: `run_seed(settings, seed, steps, device, progress)`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...` with a loop over `seeds`: for each seed that is not in `done`, a run of `run_seed`, and its log added to the file at `path` as one line of JSON.",
                      "nl": "Vervang de `...` door een loop over `seeds`: voor elke seed die niet in `done` staat, een run van `run_seed`, en zijn logboek als één regel JSON toegevoegd aan het bestand op `path`."},
            "wrong": {"en": _M_EN + ". Does your loop run only the seeds that are not in `done`, call `run_seed` with all its values, and add each log to the file with `\"a\"`, as `json.dumps(log) + \"\\n\"`?",
                      "nl": _M_NL + ". Draait je loop alleen de seeds die niet in `done` staan, roept hij `run_seed` aan met al zijn waarden, en voegt hij elk logboek met `\"a\"` aan het bestand toe, als `json.dumps(log) + \"\\n\"`?"},
            "almost": {"en": "Your `run_seeds` writes the course's experiment log for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does it run a seed only when it is not in the log yet, also when it was there before the first run, and does it pass `device` and `progress` on to `run_seed`?",
                       "nl": "Je `run_seeds` schrijft het logboek van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Draait hij een seed alleen als die nog niet in het logboek staat, ook als hij er al stond vóór de eerste run, en geeft hij `device` en `progress` door aan `run_seed`?"},
            "correct": {"en": "Your `run_seeds` runs one seed after another into one experiment log, and goes on where it stopped. It is now in your project, as `work/projects/repro/seeds.py`.",
                        "nl": "Je `run_seeds` draait de ene seed na de andere in één logboek, en gaat verder waar hij stopte. Hij staat nu in je project, als `work/projects/repro/seeds.py`."},
            "kept": {"en": "Your `run_seeds` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `run_seeds` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "figure": PlotBox(
        name="figure",
        start=_F_START,
        solution=_F,
        merge=True,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("places", matches=_against_places,
                    en="Your curves are drawn against their places in the log, 0, 1, 2, …, not against the steps: the logged steps lie further apart the later they are. Give the steps as the first list: `plt.plot(log[\"step\"], log[\"train\"], \"-\")`.",
                    nl="Je lijnen zijn getekend tegen hun plek in het logboek, 0, 1, 2, …, niet tegen de stappen: de gelogde stappen liggen verder uit elkaar naarmate ze later komen. Geef de stappen als eerste lijst: `plt.plot(log[\"step\"], log[\"train\"], \"-\")`."),
            Mistake("log-y", matches=lambda got, text: _all_lines_drawn(got) and got["yscale"] != "linear",
                    en="Every curve is there, but your y-axis is a log axis. Accuracy is a share from 0 to 1, and Figure 1 draws it on a normal axis, where 30% and 99% lie where you expect them: leave out `plt.yscale`.",
                    nl="Elke lijn staat erin, maar je y-as is een logaritmische as. Accuracy is een aandeel van 0 tot 1, en Figuur 1 tekent hem op een gewone as, waar 30% en 99% liggen waar je ze verwacht: laat `plt.yscale` weg."),
            Mistake("linear", matches=_linear,
                    en="Your x-axis is not a log axis, so the first 200 steps, where training goes from 1% to 100%, take up less than 2% of its width, and all of training seems to be done at once. Figure 1 has a log axis: `plt.xscale(\"log\")`.",
                    nl="Je x-as is geen logaritmische as, dus nemen de eerste 200 stappen, waarin training van 1% naar 100% gaat, minder dan 2% van zijn breedte in, en lijkt alle training in één keer klaar. Figuur 1 heeft een logaritmische as: `plt.xscale(\"log\")`."),
            Mistake("order", matches=_first_two_wrong, state="almost",
                    en="Every curve is there, but the legend names the first two lines `training` and `validation`, and yours are not those of the first run in that order. Draw for each run first its training curve, then its validation curve.",
                    nl="Elke lijn staat erin, maar de legenda noemt de eerste twee lijnen `training` en `validation`, en de jouwe zijn niet die van de eerste run in die volgorde. Teken voor elke run eerst de lijn van training, dan die van validatie."),
        ],
        form=[
            Mistake("same-line", matches=_not_told_apart, state="almost",
                    en="Every curve is there, but training and validation do not each have a look of their own: no line style or colour is the same for every training curve and another for every validation curve, so the legend cannot say which is which. Draw training solid, `\"-\"`, and validation dashed: `plt.plot(log[\"step\"], log[\"val\"], \"--\")`.",
                    nl="Elke lijn staat erin, maar training en validatie hebben niet elk een eigen uiterlijk: geen lijnstijl of kleur is dezelfde voor elke trainingslijn en een andere voor elke validatielijn, dus kan de legenda niet zeggen welke welke is. Teken training doorgetrokken, `\"-\"`, en validatie gestreept: `plt.plot(log[\"step\"], log[\"val\"], \"--\")`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the first `...` with two `plt.plot` lines for each run: training solid and validation dashed, against `log[\"step\"]`. Replace the second with the scale of the x-axis, then fill in the title and the labels.",
                      "nl": "Vervang de eerste `...` door twee regels `plt.plot` voor elke run: training doorgetrokken en validatie gestreept, tegen `log[\"step\"]`. Vervang de tweede door de schaal van de x-as, en vul daarna de titel en de labels in."},
            "nothing": {"en": "I see no lines in your chart. Draw each run with `plt.plot`, the steps first.",
                        "nl": "Ik zie geen lijnen in je grafiek. Teken elke run met `plt.plot`, de stappen eerst."},
            "kind": {"en": "Your chart is {kind}, and the curves of a run are lines through its logged steps: draw them with `{draw}`.",
                     "nl": "Je grafiek is {kind}, en de lijnen van een run gaan door zijn gelogde stappen: teken ze met `{draw}`."},
            "values": {"en": "Not every curve of the three runs is in your chart. Does the loop draw, for each `log`, `log[\"train\"]` and `log[\"val\"]`, both against `log[\"step\"]`?",
                       "nl": "Niet elke lijn van de drie runs staat in je grafiek. Tekent de loop voor elke `log` zowel `log[\"train\"]` als `log[\"val\"]`, allebei tegen `log[\"step\"]`?"},
            "title": {"en": "The curves are right. Now the title: the one message of the figure, as a sentence. What do all three runs show?",
                      "nl": "De lijnen kloppen. Nu de titel: de ene boodschap van de figuur, als zin. Wat laten alle drie de runs zien?"},
            "xlabel": {"en": "The curves are right. Now say along the x-axis what it shows: `plt.xlabel(\"step\")`.",
                       "nl": "De lijnen kloppen. Zeg nu langs de x-as wat hij laat zien: `plt.xlabel(\"step\")`."},
            "ylabel": {"en": "The curves are right. Now say along the y-axis what it shows: the accuracy, a share from 0 to 1.",
                       "nl": "De lijnen kloppen. Zeg nu langs de y-as wat hij laat zien: de accuracy, een aandeel van 0 tot 1."},
            "correct": {"en": "Your figure shows what Figure 1 of the paper shows, from logs: every run, training and validation each in a style of their own, on a log axis of steps, with its message in the title.",
                        "nl": "Je figuur laat zien wat Figuur 1 van de paper laat zien, uit logboeken: elke run, training en validatie elk in een eigen stijl, op een logaritmische as van stappen, met zijn boodschap in de titel."},
        },
    ),
}

# The lesson is done when these are correct. "plateau" is the choice question in the lesson.
REQUIRED = ["log-steps", "run-seed", "run-seeds", "figure", "plateau"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("log-steps", _L_START, "empty", None),
    ("log-steps", _L, "correct", None),
    ("log-steps", _L_TWICE, "wrong", "twice"),
    ("log-steps", _L_FLOOR, "wrong", "floor"),
    ("log-steps", _L_WHOLE, "wrong", "whole"),
    ("log-steps", _L_NO_LAST, "wrong", "no-last"),
    ("log-steps", _L_EVERY, "wrong", "every"),
    ("log-steps", _L_INT_STOP, "almost", None),
    ("run-seed", _R_START, "empty", None),
    ("run-seed", _R, "correct", None),
    ("run-seed", _R_TENSOR, "error", "keep-tensor"),
    ("run-seed", _R_FIRST, "wrong", "first-place"),
    ("run-seed", _R_PERCENT, "wrong", "percent"),
    ("run-seed", _R_COUNT, "wrong", "count"),
    ("run-seed", _R_BATCH, "wrong", "batch"),
    ("run-seed", _R_LOSS, "wrong", "loss"),
    ("run-seed", _R_SWAP, "wrong", "swap"),
    ("run-seed", _R_EVERY, "wrong", "every"),
    ("run-seed", _R.replace("    if progress is not None and step in at:\n            progress(log)", "    if progress is not None and step in at:\n            pass"), "almost", None),
    ("run-seeds", _M_START, "empty", None),
    ("run-seeds", _M, "correct", None),
    ("run-seeds", _M_NO_LINE, "error", "no-line"),
    ("run-seeds", _M_STR, "error", "str"),
    ("run-seeds", _M_OVERWRITE, "wrong", "overwrite"),
    ("run-seeds", _M_AGAIN, "wrong", "again"),
    ("run-seeds", _M_ONLY_DONE, "wrong", "only-done"),
    ("run-seeds", _M_NO_DEVICE, "almost", "no-device"),
    ("run-seeds", _M_NO_PROGRESS, "almost", "no-progress"),
    ("figure", _F_START, "empty", None),
    ("figure", _F, "correct", None),
    ("figure", _F_LINEAR, "wrong", "linear"),
    ("figure", _F_PLACES, "wrong", "places"),
    ("figure", _F_ORDER, "almost", "order"),
    ("figure", _F_SAME, "almost", "same-line"),
    ("figure", _f(lines="    plt.plot(log[\"step\"], log[\"train\"], \"r\")\n    plt.plot(log[\"step\"], log[\"val\"], \"g\")\n"), "correct", None),   # the paper's colours
    ("figure", _f(lines="    plt.plot(log[\"step\"], log[\"train\"], color=\"C0\")\n    plt.plot(log[\"step\"], log[\"val\"], color=\"C0\")\n"), "almost", "same-line"),
    ("figure", _F_VAL, "wrong", "values"),
    ("figure", _f(scale="plt.xscale(\"log\")\nplt.yscale(\"log\")\n"), "wrong", "log-y"),
    ("figure", _f(title="\"\""), "almost", "no-title"),
    ("figure", _f(y="\"\""), "almost", "no-ylabel"),
]


def demo(progress=None, steps: int = STEPS, seeds: tuple = tuple(SEEDS)) -> dict | str | None:
    """Trains the seeds with the learner's kit: their run_seeds with their run_seed, on their ModernGPT with their data
    and settings, into their experiment log work/projects/repro/runs.jsonl, on a GPU when there is one. None when
    repro/run.py or repro/seeds.py is not in the project yet; "seeds" when run_seeds or run_seed stops with an error,
    "model" when the model, the data or the settings do; otherwise the runs of the experiment log, as from my_runs.
    Where a module it uses is missing or broken, the course's version stands in, and "stand_ins" names it."""
    folder = projects.folder() / "repro"
    if not (folder / "run.py").is_file() or not (folder / "seeds.py").is_file():
        return None
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        import importlib
        import torch
        try:
            run_seeds = importlib.import_module("repro.seeds").run_seeds
            importlib.import_module("repro.run")
        except Exception:
            return "seeds"
        try:
            settings = importlib.import_module("repro.settings").SETTINGS
            importlib.import_module("llm.modern_gpt")
            importlib.import_module("repro.data").dataset(settings["p"], settings["fraction"], 0)
        except Exception:
            return "model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run_seeds(settings, list(seeds), steps, str(folder / "runs.jsonl"), device, progress)
        except Exception:
            return "seeds"
        return {"runs": my_runs(), "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}
    finally:
        projects.forget()


def my_runs() -> list[dict] | None:
    """The runs in the learner's experiment log, work/projects/repro/runs.jsonl, in the order of their seeds; None
    when there is no log, or a line of it is not a log as run_seed gives it."""
    path = projects.folder() / "repro" / "runs.jsonl"
    if not path.is_file():
        return None
    try:
        runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ok = all(isinstance(r, dict) and len(r["step"]) == len(r["train"]) == len(r["val"]) >= 2 for r in runs)
    except (ValueError, KeyError, TypeError):
        return None
    return sorted(runs, key=lambda r: r.get("seed", 0)) if runs and ok else None
