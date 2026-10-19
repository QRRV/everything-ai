# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-profiling-mfu: tokens_per_second, timing a training run that runs ahead of the clock, as
on a GPU. And the demo: the learner's tokens_per_second on their own GPT on their own machine, its MFU against the
fastest matrix multiplication of the machine, and a profile of a few steps."""
import importlib
import os
import re
import time

from everything_ai import projects
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

# The profiler's library writes its start and stop, and on a machine without an NVIDIA GPU an error about it, to the
# output of the cell; this keeps it quiet, for the worked example and the demo alike.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"))


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def queue_picture(labels: dict) -> str:
    """Three training steps on a GPU, seen from Python and from the GPU: Python puts each step in the queue in a moment
    and goes on, the GPU works through them one after the other. A clock read right after the third step stops far too
    early; read after the sync, it stops when the GPU is done. Time and work, no learned values: ink only."""
    from everything_ai import chart
    p = chart.Panel((0, 10.6), (0, 4.4), title=labels["title"], xlabel=labels["time"], ylabel="", xticks=[], yticks=[])
    for i, (a, b) in enumerate(((0.3, 0.8), (0.9, 1.4), (1.5, 2.0))):
        p.rect(a, 2.8, b, 3.4, "ink-3")
        p.text((a + b) / 2, 3.1, str(i + 1), "ink-2", "middle")
    p.rect(2.1, 2.8, 8.0, 3.4, "ink-3")
    p.text(5.05, 3.1, labels["waits"], "ink-3", "middle")
    for i, (a, b) in enumerate(((0.6, 3.0), (3.0, 5.5), (5.5, 8.0))):
        p.rect(a, 1.0, b, 1.6, "ink-2")
        p.text((a + b) / 2, 1.3, str(i + 1), "ink", "middle")
    p.text(8.2, 3.1, labels["python"], "ink", "start")
    p.text(8.2, 1.3, labels["gpu"], "ink", "start")
    for x, name in ((2.05, labels["early"]), (8.0, labels["after"])):
        p.line([x, x], [0.6, 4.0], "ink", 1.4, dashed=True)
        p.text(x, 0.25, name, "ink", "middle")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=230)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- the demo: the learner's GPT on the learner's machine ------------------------------------------------------------------

# The GPT, the windows and the mini-batch: (vocab, d, h, n_layers, block, batch). On a GPU the GPT of the book; on a CPU
# a smaller one, so that a step takes a fraction of a second.
PLANS = {"gpu": (65, 384, 6, 6, 256, 32), "cpu": (65, 128, 4, 4, 128, 8)}
SKIP, STEPS = 3, 10


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(where: str):
    import torch
    if where == "cuda":
        torch.cuda.synchronize()
    elif where == "mps":
        torch.mps.synchronize()


class _Real:
    """A real training run of the learner's GPT, with the methods of the exercise: step() starts one training step,
    sync() waits until the device is done, clock() reads time.perf_counter()."""

    def __init__(self, model, opt, x, y, where: str):
        self.model, self.opt, self.x, self.y, self.where = model, opt, x, y, where

    def step(self):
        import torch.nn.functional as F
        logits = self.model(self.x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.y.reshape(-1))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def sync(self):
        _sync(self.where)

    def clock(self) -> float:
        return time.perf_counter()


def _peak(where: str) -> float:
    """The FLOP/s of the fastest of a few multiplications of two 2048 x 2048 float32 matrices: about the most this
    device does, as in the lesson sys-memory-roofline."""
    import torch
    n = 2048
    a, b = torch.randn(n, n, device=where), torch.randn(n, n, device=where)
    c = torch.empty(n, n, device=where)
    torch.matmul(a, b, out=c)
    _sync(where)
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        torch.matmul(a, b, out=c)
        _sync(where)
        best = min(best, time.perf_counter() - t)
    return 2 * n ** 3 / best


def demo(code: str) -> dict | str:
    """The learner's tokens_per_second on their GPT (the course's where theirs is missing) on their best device: the
    tokens per second, the model's FLOPs per token (6N plus attention, as in gpt-count), the measured peak and the MFU;
    then a profile of two steps: the six operations that took the most time, with their share and how often they ran.
    "error" when their code stops with an error."""
    import torch
    yours = run_code(code)
    if yours.error is not None or "tokens_per_second" not in yours.names:
        return "error"
    tokens_per_second = yours.names["tokens_per_second"]
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    where = _device()
    vocab, d, h, n_layers, block, batch = PLANS["gpu" if where == "cuda" else "cpu"]
    try:
        GPT = importlib.import_module("llm.gpt").GPT
        torch.manual_seed(0)
        model = GPT(vocab, d, h, n_layers, block).to(where)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, vocab, (batch, block), device=where)
        y = torch.randint(0, vocab, (batch, block), device=where)
        run = _Real(model, opt, x, y, where)
        n_tokens = batch * block
        tps = float(with_time_limit(lambda: tokens_per_second(run, n_tokens, SKIP, STEPS), 120))
        n_params = sum(p.numel() for p in model.parameters())
        per_token = 6 * n_params + 6 * n_layers * block * d
        peak = _peak(where)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if where == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(2):
                run.step()
            run.sync()
    except Exception:
        return "error"
    finally:
        projects.forget()
    events = prof.key_averages()
    if where == "cuda":
        # Only PyTorch's operations, with the time of the GPU kernels each one starts: the kernels are in the list as
        # well, and counting both would count every kernel twice.
        events = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU]
        own = lambda e: e.self_device_time_total
    else:
        own = lambda e: e.self_cpu_time_total
    total = sum(own(e) for e in events) or 1
    top = sorted(events, key=own, reverse=True)[:6]
    return {"device": where, "sizes": (vocab, d, h, n_layers, block, batch), "params": n_params, "tps": tps,
            "per_token": per_token, "flops": tps * per_token, "peak": peak, "mfu": tps * per_token / peak,
            "top": [(e.key, own(e) / total, e.count) for e in top], "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}


# --- exercise 1: tokens_per_second -------------------------------------------------------------------------------------------


class _Run:
    """A training run on a GPU, played on a clock of its own: step() only puts its work in the device's queue and
    returns after a tenth of a millisecond, as a step on a real GPU does; the device works through the queue at its
    own pace; sync() waits until it is done, as torch.cuda.synchronize() does; clock() reads the time in seconds. The
    first `slow` steps take longer, as the first steps of a real run do, and the others take a little more and a little
    less in turn, so that one step alone is not the average. The same code gives the same time on any machine, however
    busy."""

    def __init__(self, each: float = 0.005, first: float = 0.03, slow: int = 3, launch: float = 0.0001):
        self.each, self.first, self.slow, self.launch = each, first, slow, launch
        self.calls, self.now, self.busy_until = 0, 0.0, 0.0

    def step(self):
        self.calls += 1
        cost = self.first if self.calls <= self.slow else self.each * (1.1 if self.calls % 2 == 0 else 0.9)
        self.busy_until = max(self.now, self.busy_until) + cost
        self.now += self.launch

    def sync(self):
        self.now = max(self.now, self.busy_until)

    def clock(self) -> float:
        return self.now

    def __repr__(self):
        return "run"


_HEAD = "def tokens_per_second(run, n_tokens, skip, steps):\n"
_START = (_HEAD + "    # run.step() starts one training step of n_tokens tokens, run.sync() waits until the device is done,\n"
          "    # and run.clock() gives the time in seconds\n    ...\n")


def _t(first: str = "    for _ in range(skip):\n        run.step()\n    run.sync()\n    start = run.clock()\n",
       loop: str = "    for _ in range(steps):\n        run.step()\n    run.sync()\n",
       ret: str = "    return n_tokens * steps / (run.clock() - start)\n") -> str:
    return _HEAD + first + loop + ret


# What the case being checked should give: _close keeps it, so that the known mistakes compare with the right case.
_WANT = [0.0]


def _close(got, want) -> bool:
    """Within 5%: waiting for the device after every step, for instance, costs a little time and is not wrong."""
    _WANT[0] = want
    if isinstance(got, str):
        return False
    try:
        return 0.95 < float(got) / float(want) < 1.05
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _show(v) -> str:
    if isinstance(v, str):
        return repr(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.10f}".rstrip("0").rstrip(".") if abs(x) < 0.01 else f"{x:.3g}"


def _ratio(got) -> float:
    """What the learner's function gave, as a part of what it should give for the same case."""
    if isinstance(got, str):
        return 1.0
    try:
        return float(got) / float(_WANT[0])
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "tokens-per-second": FunctionBox(
        name="tokens-per-second",
        function="tokens_per_second",
        start=_START,
        solution=_t(),
        visible=[(_Run(), 2048, 5, 10)],
        hidden=[(_Run(each=0.004, first=0.03, slow=2), 512, 2, 15), (_Run(each=0.006, first=0.04, slow=4), 4096, 6, 8),
                (_Run(each=0.004, first=0.05, slow=1), 1000, 1, 12)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("text", matches=lambda got, text: isinstance(got, str),
                    en="`{call}` gives text, between quotes, and it should give about `{want}`: give back the number itself.",
                    nl="`{call}` geeft tekst, tussen aanhalingstekens, en dat moet ongeveer `{want}` zijn: geef het getal zelf terug."),
            Mistake("own-clock", matches=lambda got, text: re.search(r"\brun\.clock\b", text) is None,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: your function does not read `run.clock()`. The check's run keeps time on a clock of its own, which `time.perf_counter()` does not see. Read the time with `run.clock()`.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: je functie leest `run.clock()` niet af. De run van de check houdt de tijd bij op een eigen klok, die `time.perf_counter()` niet ziet. Lees de tijd af met `run.clock()`."),
            Mistake("skip-tokens", matches=lambda got, text: 1.35 < _ratio(got) < 1.8,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: you count the tokens of the first `skip` steps too, while your clock only ran for the other `steps`.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: je telt de tokens van de eerste `skip` stappen mee, terwijl je klok alleen liep voor de andere `steps`."),
            Mistake("no-tokens", matches=lambda got, text: 0.0003 < _ratio(got) < 0.0008,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: that is steps per second. Every step trains on `n_tokens` tokens.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: dat zijn stappen per seconde. Elke stap traint op `n_tokens` tokens."),
            Mistake("seconds", matches=lambda got, text: 0 < _ratio(got) < 1e-6,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: that is a time, not a speed. Divide the tokens of the timed steps by the seconds they took.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: dat is een tijd, geen snelheid. Deel de tokens van de gemeten stappen door de seconden die ze duurden."),
            Mistake("no-sync", matches=lambda got, text: _ratio(got) > 2.5 and re.sub(r"#.*", "", text).count("sync(") < 2,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: far too fast. On a GPU, `run.step()` returns before its work is done, so your clock stops while the device is still busy. Call `run.sync()` before you stop the clock.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: veel te snel. Op een GPU komt `run.step()` terug voordat zijn werk af is, dus je klok stopt terwijl het device nog bezig is. Roep `run.sync()` aan voordat je de klok stopt."),
            Mistake("first-steps", matches=lambda got, text: 0.15 < _ratio(got) < 0.6,
                    en="`{call}` gives `{got}`, and it should give about `{want}`: too slow, because the slow first steps land in your time. Run `skip` steps first, call `run.sync()` so that they are done, and only then start the clock.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: te traag, omdat de trage eerste stappen in je tijd belanden. Doe eerst `skip` stappen, roep `run.sync()` aan zodat ze af zijn, en start pas dan de klok."),
            # Last: telling a wrong version apart calls _close on its results, which would change _WANT for those above.
            Mistake("one-step", wrong=_t(loop="    run.step()\n    run.sync()\n", ret="    return n_tokens / (run.clock() - start)\n"),
                    en="`{call}` gives `{got}`, and it should give about `{want}`: you time one step. Steps differ a little from each other: time all `steps` steps, and divide their tokens by their seconds.",
                    nl="`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn: je meet één stap. Stappen verschillen een beetje van elkaar: meet alle `steps` stappen, en deel hun tokens door hun seconden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: first some steps that you do not time, then the steps you do time, and a `return` of the tokens per second.",
                      "nl": "Vervang de `...` door je code: eerst een paar stappen die je niet meet, dan de stappen die je wel meet, en een `return` van de tokens per seconde."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the run, the tokens of one step, the steps to skip and the steps to time, and the `def` line of your function has another number of names between its parentheses. Keep `(run, n_tokens, skip, steps)`.",
                          "nl": "De check roept `{call}` aan, met de run, de tokens van één stap, de stappen om over te slaan en de stappen om te meten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(run, n_tokens, skip, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give about `{want}`. Do you skip the first steps, wait with `run.sync()` before you start and before you stop the clock, and divide the tokens of the timed steps by the seconds?",
                      "nl": "`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn. Sla je de eerste stappen over, wacht je met `run.sync()` voordat je de klok start en voordat je hem stopt, en deel je de tokens van de gemeten stappen door de seconden?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give about `{want}`. Does your function use `n_tokens`, `skip` and `steps`, and no fixed numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn. Gebruikt je functie `n_tokens`, `skip` en `steps`, en geen vaste getallen?"},
            "correct": {"en": "Your function times all {n} runs within 5%: the slow first steps left out, and the clock read only after the device is done.",
                        "nl": "Je functie meet alle {n} runs binnen 5%: de trage eerste stappen weggelaten, en de klok pas afgelezen als het device klaar is."},
        },
    ),
}

# The lesson is done when these are correct. "hfu" is the choice question, "low-mfu" the open question.
REQUIRED = ["tokens-per-second", "hfu", "low-mfu"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("tokens-per-second", _START, "empty", None),
    ("tokens-per-second", _t(), "correct", None),
    ("tokens-per-second", _HEAD + "    for i in range(skip + steps):\n        if i == skip:\n            run.sync()\n            start = run.clock()\n"
     "        run.step()\n    run.sync()\n    seconds = run.clock() - start\n    return steps * n_tokens / seconds\n", "correct", None),
    ("tokens-per-second", _t(loop="    for _ in range(steps):\n        run.step()\n        run.sync()\n"), "correct", None),
    ("tokens-per-second", _t(loop="    for _ in range(steps):\n        run.step()\n"), "wrong", "no-sync"),
    ("tokens-per-second", _t(first="    for _ in range(skip):\n        run.step()\n    start = run.clock()\n"), "wrong", "first-steps"),
    ("tokens-per-second", _t(first="    start = run.clock()\n"), "wrong", "first-steps"),
    ("tokens-per-second", _t(first="    start = run.clock()\n    for _ in range(skip):\n        run.step()\n"), "wrong", "first-steps"),
    ("tokens-per-second", _t(ret="    return n_tokens * (skip + steps) / (run.clock() - start)\n"), "wrong", "skip-tokens"),
    ("tokens-per-second", _t(ret="    return steps / (run.clock() - start)\n"), "wrong", "no-tokens"),
    ("tokens-per-second", _t(ret="    return (run.clock() - start) / (n_tokens * steps)\n"), "wrong", "seconds"),
    ("tokens-per-second", _t(ret="    return run.clock() - start\n"), "wrong", "seconds"),
    ("tokens-per-second", "import time\n" + _t().replace("run.clock()", "time.perf_counter()"), "wrong", "own-clock"),
    ("tokens-per-second", "import time\n" + _t(first="    for _ in range(skip):\n        run.step()\n    start = time.perf_counter()\n")
     .replace("run.clock()", "time.perf_counter()"), "wrong", "own-clock"),
    ("tokens-per-second", _t(loop="    run.step()\n    run.sync()\n", ret="    return n_tokens / (run.clock() - start)\n"), "wrong", "one-step"),
    ("tokens-per-second", _t(ret="    return n_tokens * 10 / (run.clock() - start)\n"), "almost", None),
    ("tokens-per-second", _t(ret="    return 2048 * steps / (run.clock() - start)\n"), "almost", None),
    ("tokens-per-second", _t(ret="    return str(n_tokens * steps / (run.clock() - start))\n"), "wrong", "text"),
    ("tokens-per-second", _t(loop="    for _ in range(steps):\n        run.step()\n    # run.sync()\n"), "wrong", "no-sync"),
]
