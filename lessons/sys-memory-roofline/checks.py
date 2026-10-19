# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-memory-roofline: roofline_time, the time an operation needs by the roofline estimate and
what limits it. And the roofline of an example GPU from its specifications, and a measurement on the learner's own
machine: matrix multiplications of growing size, and the addition of two long arrays."""
import math
import re
import time

from everything_ai.check import FunctionBox, Mistake

# An example GPU from its published specifications: the NVIDIA GeForce RTX 4090 (2022), FP32 on its ordinary cores,
# without tensor cores, and the bandwidth of its 24 GB of GDDR6X memory.
GPU = {"name": "RTX 4090", "peak": 82.6e12, "bandwidth": 1008e9}

# Two operations on float32 numbers, 4 bytes each, with every number read or written once.
ADD = {"flops": 1e8, "bytes": 12 * 1e8}                      # c = a + b with 100 million numbers: read 2, write 1
MATMUL = {"flops": 2 * 4096 ** 3, "bytes": 3 * 4096 ** 2 * 4}  # C = A @ B with 4096 x 4096 matrices: read 2, write 1


def intensity(op: dict) -> float:
    return op["flops"] / op["bytes"]


def estimate(op: dict, gpu: dict = GPU) -> tuple:
    """The course's own roofline_time: (seconds, "compute" or "memory")."""
    t_compute, t_memory = op["flops"] / gpu["peak"], op["bytes"] / gpu["bandwidth"]
    return max(t_compute, t_memory), ("compute" if t_compute >= t_memory else "memory")


def number(x: float, digits: int, lang: str) -> str:
    """A number for running text, with its thousands marked as the language writes them: 12,345.6 or 12.345,6."""
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def roof_picture(labels: dict, points: list | None = None, peak: float | None = None, bandwidth: float | None = None) -> str:
    """The roofline on log-log axes: attainable FLOP/s against arithmetic intensity, the slanted roof of the memory
    and the flat roof of the compute, with operations as points. Ink only: nothing here is learned or a loss."""
    from everything_ai import chart
    peak = peak or GPU["peak"]
    bandwidth = bandwidth or GPU["bandwidth"]
    points = points if points is not None else [(intensity(ADD), min(GPU["peak"], intensity(ADD) * GPU["bandwidth"]), labels["add"]),
                                                (intensity(MATMUL), min(GPU["peak"], intensity(MATMUL) * GPU["bandwidth"]), labels["matmul"])]
    xs = [p[0] for p in points] + [peak / bandwidth]
    x0, x1 = 10 ** math.floor(math.log10(min(xs) / 3)), max(xs) * 5
    ys = [p[1] for p in points] + [peak, x0 * bandwidth]
    y0, y1 = 10 ** math.floor(math.log10(min(ys) / 2)), 10 ** math.ceil(math.log10(max(ys) * 5))   # room for a name over the roof
    p = chart.Panel((x0, x1), (y0, y1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True, ylog=True)
    ridge = peak / bandwidth
    p.line([x0, ridge, x1], [x0 * bandwidth, peak, peak], "ink-2", 2.2)
    p.line([ridge, ridge], [y0, peak], "ink-3", 1.2, dashed=True)
    p.text(ridge * 1.15, peak / 40, labels["ridge"], "ink-3", "start")
    for x, y, name in points:
        p.points([x], [y], "ink", r=4)
        if name:   # left of the middle, a name goes under the slanted roof, to the right; else over the roof, where no line runs
            if math.log(x / x0) < math.log(x1 / x0) / 2:
                p.text(x * 1.3, y / 2.6, name, "ink", "start")
            else:
                p.text(x * 1.3, y * 2.6, name, "ink", "end")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=270)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- the measurement on the learner's machine --------------------------------------------------------------------------------

SIZES = [64, 128, 256, 512, 1024, 2048]


def _device():
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


def _timed(fn, where: str, budget: float = 0.3) -> float:
    """The fastest of a few runs of fn, in seconds, after one run to warm up; runs for about `budget` seconds."""
    fn()
    _sync(where)
    best, spent = float("inf"), 0.0
    while spent < budget:
        t = time.perf_counter()
        fn()
        _sync(where)
        dt = time.perf_counter() - t
        best, spent = min(best, dt), spent + dt
    return max(best, 1e-9)   # never zero, whatever the clock


def measure(show=None) -> dict:
    """Matrix multiplications of SIZES and the addition of two arrays of 32 million float32 numbers on the learner's
    best device: per matmul (n, intensity, FLOP/s), and the bandwidth of the addition in bytes per second. The
    result goes into a tensor made beforehand (`out=`): a new one for every run would also time asking the system
    for memory, which on a CPU made the addition about three times as slow."""
    import torch
    where = _device()
    out = {"device": where, "matmul": []}
    for i, n in enumerate(SIZES):
        if show:
            show(i, len(SIZES) + 1)
        a = torch.randn(n, n, device=where)
        b = torch.randn(n, n, device=where)
        c = torch.empty(n, n, device=where)
        t = _timed(lambda: torch.matmul(a, b, out=c), where)
        out["matmul"].append((n, 2 * n ** 3 / (3 * n * n * 4), 2 * n ** 3 / t))
    if show:
        show(len(SIZES), len(SIZES) + 1)
    m = 32_000_000   # 128 MB per array, more than the caches of nearly any chip
    a = torch.randn(m, device=where)
    b = torch.randn(m, device=where)
    c = torch.empty(m, device=where)
    t = _timed(lambda: torch.add(a, b, out=c), where)
    out["add"] = (1 / 12, m / t, 12 * m / t)   # (intensity, FLOP/s, bytes per second)
    return out


class _Num(float):
    """A number that a check message writes the way the lesson does, such as `82.6e12`."""

    def __new__(cls, value, shown):
        obj = super().__new__(cls, value)
        obj.shown = shown
        return obj

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return float(self)


def _case(*pairs):
    return tuple(_Num(v, t) for v, t in pairs)


# --- exercise 1: roofline_time ------------------------------------------------------------------------------------------------

_HEAD = "def roofline_time(flops, bytes_moved, peak_flops, bandwidth):\n"
_START = (_HEAD + "    # the time the calculations need, and the time the data needs\n    ...\n"
          "    # the slower of the two sets the time; give back that time and which limit it is\n    ...\n")


def _t(combine: str = "max(t_compute, t_memory)", label: str = '"compute" if t_compute >= t_memory else "memory"',
       tc: str = "flops / peak_flops", tm: str = "bytes_moved / bandwidth") -> str:
    return (_HEAD + f"    t_compute = {tc}\n    t_memory = {tm}\n    time = {combine}\n"
            f"    limit = {label}\n    return [time, limit]\n")


def _pair(got):
    """(time, limit) from what a function gives back, a list or a tuple; None when it is not a number and a text."""
    try:
        if isinstance(got, (list, tuple)) and len(got) == 2 and isinstance(got[1], str) and not isinstance(got[0], (str, bool)):
            return float(got[0]), got[1].strip().lower()
    except (TypeError, ValueError):
        pass
    return None


def _same(got, want) -> bool:
    g, w = _pair(got), _pair(want)
    return g is not None and w is not None and math.isclose(g[0], w[0], rel_tol=1e-9) and g[1] == w[1]


def _show(v) -> str:
    p = _pair(v)
    return f"[{p[0]:.5g}, \"{p[1]}\"]" if p else repr(v)


def _other_words(got, text) -> bool:
    """A time and a text, but the text is not one of the two words, such as "memory-bound"."""
    p = _pair(got)
    return False if p is None else p[1] not in ("compute", "memory")


def _limit_first(got, text) -> bool:
    """The word first and the time second: ["memory", 0.00119]."""
    return isinstance(got, (list, tuple)) and len(got) == 2 and _pair([got[1], got[0]]) is not None


def _one_number(got, text) -> bool:
    """Only a number, without the limit."""
    return _pair([got, "memory"]) is not None


# The solution in the lesson's last hint, with an if and an else as in py-if.
_SOLUTION = (_HEAD + "    t_compute = flops / peak_flops\n    t_memory = bytes_moved / bandwidth\n    if t_compute >= t_memory:\n"
             "        return [t_compute, \"compute\"]\n    return [t_memory, \"memory\"]\n")

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "roofline-time": FunctionBox(
        name="roofline-time",
        function="roofline_time",
        start=_START,
        solution=_SOLUTION,
        visible=[_case((1e8, "1e8"), (1.2e9, "1.2e9"), (82.6e12, "82.6e12"), (1008e9, "1008e9"))],
        hidden=[_case((MATMUL["flops"], "2 * 4096**3"), (MATMUL["bytes"], "3 * 4096**2 * 4"), (82.6e12, "82.6e12"), (1008e9, "1008e9")),
                _case((2e9, "2e9"), (1e8, "1e8"), (1e12, "1e12"), (1e11, "1e11")),
                _case((5e10, "5e10"), (4e9, "4e9"), (1e14, "1e14"), (2e12, "2e12")),
                _case((3e11, "3e11"), (1e9, "1e9"), (1.5e13, "1.5e13"), (9e11, "9e11"))],
        show=_show,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("limit-first", matches=_limit_first,
                    en="`{call}` gives `{got}`, and it should give `{want}`: you give the limit first. Give back the time first, then the word: `[time, limit]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je geeft de grens eerst. Geef eerst de tijd terug, dan het woord: `[time, limit]`."),
            Mistake("one-number", matches=_one_number,
                    en="`{call}` gives `{got}`, one number, and it should give `{want}`: give back two things in a list, the time and the word that names the limit.",
                    nl="`{call}` geeft `{got}`, één getal, en dat moet `{want}` zijn: geef twee dingen terug in een lijst, de tijd en het woord dat de grens noemt."),
            Mistake("sum", wrong=_t(combine="t_compute + t_memory"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the two times. A GPU calculates while it moves data, so the two overlap: the slower one sets the time, `max(t_compute, t_memory)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de twee tijden op. Een GPU rekent terwijl hij data verplaatst, dus de twee overlappen: de langzaamste bepaalt de tijd, `max(t_compute, t_memory)`."),
            Mistake("min", wrong=_t(combine="min(t_compute, t_memory)", label='"compute" if t_compute <= t_memory else "memory"'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the faster of the two times. The operation is done only when both the calculations and the data are done, so the slower one sets the time: `max(t_compute, t_memory)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de snelste van de twee tijden. De bewerking is pas klaar als zowel het rekenen als de data klaar zijn, dus de langzaamste bepaalt de tijd: `max(t_compute, t_memory)`."),
            Mistake("swapped-label", wrong=_t(label='"memory" if t_compute >= t_memory else "compute"'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the time is right, but you name the other limit. When the calculations take longer, the operation is limited by compute; when the data takes longer, by memory.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de tijd klopt, maar je noemt de andere grens. Duurt het rekenen langer, dan is de bewerking begrensd door compute; duurt de data langer, dan door het geheugen."),
            Mistake("milliseconds", wrong=_t(combine="max(t_compute, t_memory) * 1000"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your time is in milliseconds. FLOPs divided by FLOP/s, and bytes divided by bytes a second, already give seconds; give back those.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je tijd staat in milliseconden. FLOPs gedeeld door FLOP/s, en bytes gedeeld door bytes per seconde, geven al seconden; geef die terug."),
            Mistake("upside-down", wrong=_t(tc="peak_flops / flops", tm="bandwidth / bytes_moved"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your times are upside down. Time is work divided by speed: `flops / peak_flops` seconds for the calculations, `bytes_moved / bandwidth` for the data.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je tijden staan op hun kop. Tijd is werk gedeeld door snelheid: `flops / peak_flops` seconden voor het rekenen, `bytes_moved / bandwidth` voor de data."),
            Mistake("other-words", matches=_other_words,
                    en="`{call}` gives `{got}`, and it should give `{want}`: name the limit with one of the two words `\"compute\"` or `\"memory\"`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: noem de grens met een van de twee woorden `\"compute\"` of `\"memory\"`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: first the time of the calculations and the time of the data, then the slower of the two and the name of its limit.",
                      "nl": "Vervang de twee `...`: eerst de tijd van het rekenen en de tijd van de data, dan de langzaamste van de twee en de naam van zijn grens."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the FLOPs, the bytes, the peak FLOP/s and the bandwidth, and the `def` line of your function has another number of names between its parentheses. Keep `(flops, bytes_moved, peak_flops, bandwidth)`.",
                          "nl": "De check roept `{call}` aan, met de FLOPs, de bytes, de piek-FLOP/s en de bandbreedte, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(flops, bytes_moved, peak_flops, bandwidth)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the larger of the two times, work divided by speed, and name the limit of that one?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de grootste van de twee tijden, werk gedeeld door snelheid, en noem je de grens van die tijd?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also work when the calculations are the slower part?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook als het rekenen het langzaamste deel is?"},
            "correct": {"en": "Your function is right in all {n} cases: whichever takes longer, the calculations or the data, sets the time and is the limit.",
                        "nl": "Je functie klopt in alle {n} gevallen: wat het langst duurt, het rekenen of de data, bepaalt de tijd en is de grens."},
        },
    ),
}

# The lesson is done when these are correct. "memory-bound" is the choice question, "batch" the open question.
REQUIRED = ["roofline-time", "memory-bound", "batch"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("roofline-time", _START, "empty", None),
    ("roofline-time", _SOLUTION, "correct", None),
    ("roofline-time", _t(), "correct", None),
    ("roofline-time", _HEAD + "    t = max(flops / peak_flops, bytes_moved / bandwidth)\n    if flops / bytes_moved > peak_flops / bandwidth:\n        return (t, \"compute\")\n    return (t, \"memory\")\n", "correct", None),
    ("roofline-time", _HEAD + "    a = flops / peak_flops\n    b = bytes_moved / bandwidth\n    if a > b:\n        return [a, \"compute\"]\n    return [b, \"memory\"]\n", "correct", None),
    ("roofline-time", _t(combine="t_compute + t_memory"), "wrong", "sum"),
    ("roofline-time", _t(combine="min(t_compute, t_memory)", label='"compute" if t_compute <= t_memory else "memory"'), "wrong", "min"),
    ("roofline-time", _t(label='"memory" if t_compute >= t_memory else "compute"'), "wrong", "swapped-label"),
    ("roofline-time", _t(tc="peak_flops / flops", tm="bandwidth / bytes_moved"), "wrong", "upside-down"),
    ("roofline-time", _t(label='"compute-bound" if t_compute >= t_memory else "memory-bound"'), "wrong", "other-words"),
    ("roofline-time", _t(label='"memory"'), "almost", None),
    ("roofline-time", _HEAD + "    t_compute = flops / peak_flops\n    t_memory = bytes_moved / bandwidth\n    if t_compute >= t_memory:\n        return [\"compute\", t_compute]\n    return [\"memory\", t_memory]\n", "wrong", "limit-first"),
    ("roofline-time", _HEAD + "    return max(flops / peak_flops, bytes_moved / bandwidth)\n", "wrong", "one-number"),
    ("roofline-time", _HEAD + "    t_compute = flops / peak_flops * 1000\n    t_memory = bytes_moved / bandwidth * 1000\n    if t_compute >= t_memory:\n        return [t_compute, \"compute\"]\n    return [t_memory, \"memory\"]\n", "wrong", "milliseconds"),
    ("roofline-time", _HEAD + "    t = max(flops / peak_flops, bytes_moved / bandwidth)\n    return [str(t), \"memory\"]\n", "wrong", None),
]
