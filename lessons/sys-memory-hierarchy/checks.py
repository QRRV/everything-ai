# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The checks of the lesson sys-memory-hierarchy: the layers of memory of an NVIDIA H100 and of a computer, the read
times they give, the exercises with their known mistakes, the pictures of the learner's read times and of the
bandwidth the learner's own computer reaches, and the demo of sums of every size on the course's computer.

The numbers of the H100 (SXM) are NVIDIA's, read on 28 November 2026: the product page (80 GB of HBM3 at 3.35 TB/s,
PCIe Gen5 at 128 GB/s), the Hopper architecture in-depth blog of 22 March 2022 (132 groups of cores, which NVIDIA calls
SMs; 256 KB of L1 cache and shared memory in each; a 50 MB L2 cache; five stacks of HBM3; 64 GB/s of PCIe Gen5 in each
direction) and the Hopper tuning guide of CUDA 12.9 (64K registers of 32 bits in each SM: 256 KB). The SSD is Samsung's
data sheet of the 990 PRO (October 2022: 7,450 MB/s sequential read, 2 TB). One module of DDR5-4800 delivers 4,800
million transfers of 8 bytes a second, 38.4 GB/s (ServeTheHome, 4 February 2023).

"The course's computer" is the computer the course was made on, with 4 cores. Its times are the fastest of many sums of
float64 numbers with NumPy, in three runs on 28 November 2026; every computer gives other times, so the checks look at
what a function makes of times, never at the times of the learner's computer. Every value a learner's code gives back
is shown through html.escape."""
import html
import math
import time

import numpy as np

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-memory-hierarchy"
LANGS = ("en", "nl")

# ======== what the makers published ========================================================================================

H100_GROUPS = 132                 # groups of cores (SMs) on an H100 SXM
H100_REGISTERS_KB = 256           # registers in each group, in KB as NVIDIA writes it: 65,536 registers of 4 bytes
H100_L1_KB = 256                  # L1 cache and shared memory in each group, in KB as NVIDIA writes it
H100_L2_MB = 50                   # L2 cache on the chip, in MB as NVIDIA writes it
H100_HBM = 80_000_000_000         # bytes of HBM3 next to the chip, in five stacks
H100_HBM_RATE = 3.35e12           # bytes per second from the HBM3
PCIE_RATE = 64e9                  # bytes per second over PCIe Gen5 x16, in each direction
SSD_RATE = 7.45e9                 # bytes per second, sequential read of a Samsung 990 PRO
DDR5_RATE = 38.4e9                # bytes per second from one module of DDR5-4800


class _Levels(dict):
    """A dict of bandwidths that shows itself by its name in the lesson, C.LEVELS."""

    def __repr__(self):
        return "C.LEVELS"


class _Bytes(int):
    """A number of bytes that shows itself as Python can write it, with underscores: 16_000_000_000."""

    def __repr__(self):
        return f"{int(self):_}"


class _Dict(dict):
    """A dict of bandwidths that shows itself the way the lesson writes it, such as {"HBM": 3.35e12}."""

    def __init__(self, values: dict, text_: str):
        super().__init__(values)
        self.text = text_

    def __repr__(self):
        return self.text


LEVELS = _Levels({"HBM": H100_HBM_RATE, "PCIe": PCIE_RATE, "SSD": SSD_RATE})
MODEL_BYTES = _Bytes(16_000_000_000)   # a model of 16 GB, such as 8 billion parameters of 2 bytes

# ======== measured on the course's computer ================================================================================

# The bytes of each array, and the time of one sum of its float64 numbers: the fastest of many, in seconds.
SIZES = [1_000, 8_000, 64_000, 512_000, 4_000_000, 32_000_000, 128_000_000, 512_000_000]
COURSE_SECONDS = [5.3e-7, 7.9e-7, 2.4e-6, 1.51e-5, 1.43e-4, 1.21e-3, 1.54e-2, 6.28e-2]
# 16 million float64 numbers, 128 MB: a[order].sum() and a[shuffled].sum(), the fastest of five in three runs.
ORDER_SECONDS = 0.067
SHUFFLED_SECONDS = 0.229


def course_rates() -> list[float]:
    """The gigabytes per second of each sum on the course's computer."""
    return [b / s / 1e9 for b, s in zip(SIZES, COURSE_SECONDS)]


# ======== the functions of the exercises, as the course writes them =========================================================

def read_seconds(nbytes, bandwidths):
    seconds = {}
    for name in bandwidths:
        seconds[name] = nbytes / bandwidths[name]
    return seconds


def gb_per_second(values):
    values.sum()
    start = time.perf_counter()
    values.sum()
    seconds = time.perf_counter() - start
    return values.nbytes / seconds / 1e9


class slow_array:
    """A stand-in for an array of float64 numbers in the checks: it holds `nbytes` bytes, and each of its sums takes
    `seconds` seconds and gives 0.0."""

    def __init__(self, nbytes: int, seconds: float):
        self.nbytes = nbytes
        self.seconds = seconds
        self.itemsize = 8
        self.size = nbytes // 8
        self.shape = (nbytes // 8,)
        self.dtype = np.dtype(np.float64)

    def __len__(self):
        return self.size

    def sum(self, *args, **kwargs):
        time.sleep(self.seconds)
        return 0.0

    def __repr__(self):
        return f"slow_array({self.nbytes}, {self.seconds})"


def facts() -> dict:
    """The numbers the lesson's texts are built on."""
    rates = course_rates()
    return {
        "hbm_80": H100_HBM / H100_HBM_RATE,                      # 0.0239 s
        "pcie_80": H100_HBM / PCIE_RATE,                         # 1.25 s
        "ssd_80": H100_HBM / SSD_RATE,                           # 10.7 s
        "reads_a_second": H100_HBM_RATE / H100_HBM,              # 41.9
        "model": read_seconds(MODEL_BYTES, LEVELS),              # 4.78 ms, 0.25 s, 2.15 s
        "ssd_vs_hbm": H100_HBM_RATE / SSD_RATE,                  # 450
        "pcie_vs_hbm": H100_HBM_RATE / PCIE_RATE,                # 52
        "example_hbm": 8_000_000 / H100_HBM_RATE,                # 2.39 µs
        "example_ssd": 8_000_000 / SSD_RATE,                     # 1,074 µs
        "rates": rates,                                          # 1.9 .. 34 .. 8.2 GB/s
        "peak": max(rates),
        "order": SHUFFLED_SECONDS / ORDER_SECONDS,               # 3.4
        "stand_in": 800_000_000 / 0.1 / 1e9,                     # 8 GB/s
    }


# ======== how numbers are written ==========================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _dutch(text_: str) -> str:
    """1,234.5 as Dutch writes it: 1.234,5."""
    return text_.replace(",", "_").replace(".", ",").replace("_", ".")


def _local(text_: str, lang: str) -> str:
    return _dutch(text_) if lang == "nl" else text_


def size(nbytes: float) -> str:
    """A number of bytes the way the lesson writes it: 1 KB, 512 KB, 4 MB, 16 GB."""
    for unit, name in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if nbytes >= unit:
            return f"{nbytes / unit:g} {name}"
    return f"{nbytes:g} bytes"


def secs(s: float, lang: str) -> str:
    """A time in a fitting unit, with two significant digits below 10: 0.53 µs, 15 µs, 143 µs, 4.8 ms, 0.25 s, 2.1 s."""
    unit, name = (1, "s") if s >= 0.1 else (1e-3, "ms") if s >= 1e-3 else (1e-6, "µs")
    v = s / unit
    text_ = f"{v:.2g}" if v < 10 or v >= 1e6 else f"{v:,.0f}"
    return f"{_local(text_, lang)} {name}"


def rate(gbps: float, lang: str) -> str:
    """Gigabytes per second: 1.9 GB/s, 34 GB/s, 3,350 GB/s."""
    text_ = f"{gbps:.2g}" if gbps < 10 or gbps >= 1e5 else f"{gbps:,.0f}"
    return f"{_local(text_, lang)} GB/s"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size_: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size_}px;'
            f'font-weight:{weight};fill:{fill};white-space:pre" xml:space="preserve">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-line)", width: float = 1) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{stroke};stroke-width:{width}"/>')


def _rect(x: float, y: float, w: float, h: float, style: str, rx: float = 3) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" style="{style}"/>'


# ======== the picture of the idea: the ladder of an H100 ===================================================================

IDEA_TEXT = {
    "en": {"rows": [("Registers", "256 KB × 132"), ("L1 cache", "256 KB × 132"), ("L2 cache", "50 MB"),
                    ("HBM3", "80 GB · 3.35 TB/s"), ("RAM, over PCIe Gen5", "64 GB/s each way"),
                    ("SSD", "2 TB · 7.45 GB/s")],
           "groups": {0: "On the GPU chip", 3: "Next to the chip", 4: "In the computer"},
           "top": "small and fast", "bottom": "large and slow",
           "alt": "A ladder of six layers of memory of an NVIDIA H100, narrow at the top and wide at the bottom: "
                  "registers and L1 cache of 256 KB in each of 132 groups of cores and an L2 cache of 50 MB on the "
                  "chip, 80 GB of HBM3 at 3.35 TB/s next to the chip, then the RAM of the computer over PCIe Gen5 "
                  "at 64 GB/s each way, and an SSD of 2 TB at 7.45 GB/s."},
    "nl": {"rows": [("Registers", "256 KB × 132"), ("L1-cache", "256 KB × 132"), ("L2-cache", "50 MB"),
                    ("HBM3", "80 GB · 3,35 TB/s"), ("RAM, via PCIe Gen5", "64 GB/s per richting"),
                    ("SSD", "2 TB · 7,45 GB/s")],
           "groups": {0: "Op de GPU-chip", 3: "Naast de chip", 4: "In de computer"},
           "top": "klein en snel", "bottom": "groot en traag",
           "alt": "Een ladder van zes lagen geheugen van een NVIDIA H100, smal bovenaan en breed onderaan: registers en "
                  "L1-cache van 256 KB in elk van de 132 groepen cores en een L2-cache van 50 MB op de chip, 80 GB HBM3 "
                  "met 3,35 TB/s naast de chip, dan het RAM van de computer via PCIe Gen5 met 64 GB/s per richting, en "
                  "een SSD van 2 TB met 7,45 GB/s."},
}


def idea_picture(lang: str) -> str:
    w = IDEA_TEXT[_lang(lang)]
    parts = []
    widths = [120, 140, 160, 186, 210, 232]
    for k, ((name, detail), width) in enumerate(zip(w["rows"], widths)):
        y = 8 + k * 40
        x = 326 - width
        group = k >= 4
        parts.append(_rect(x, y, width, 34, "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1", 4))
        parts.append(_t(x + width / 2, y + 15, name, 11.5, 600, "var(--ea-ink)", "middle"))
        parts.append(_t(x + width / 2, y + 28, detail, 10, 400, "var(--ea-ink-2)", "middle"))
        if k in w["groups"]:
            parts.append(_t(0, y + 21, w["groups"][k], 10.5, 600, "var(--ea-ink-2)" if group else "var(--ea-ink)"))
    parts.append(_t(0, 69, "↑ " + w["top"], 10, 400, "var(--ea-ink-3)"))
    parts.append(_t(0, 229, "↓ " + w["bottom"], 10, 400, "var(--ea-ink-3)"))
    return _svg(250, w["alt"], parts)


# ======== the pictures of the learner's own work ===========================================================================

YOURS_TEXT = {
    "en": {"read": {
               "title": "Your read_seconds for a model of 16 GB",
               "intro": "Your function reads 16 GB once from three places, on a log scale of seconds.",
               "rows": {"HBM": ("HBM3 next to the chip", "3.35 TB/s"), "PCIe": ("PCIe Gen5 from the computer", "64 GB/s"),
                        "SSD": ("SSD of the computer", "7.45 GB/s")},
               "none": "Your `read_seconds` stops with an error, takes too long, or does not give a number of seconds "
                       "above 0 for each of the three places, so the picture waits.",
               "right": "On these three, your function gives the read times of the lesson: the SSD takes 450 times as "
                        "long as the HBM.",
               "off": "Your function gives {got} for the SSD, and 16 GB at 7.45 GB/s takes 2.1 seconds.",
               "alt": "Three bars on a log scale with the seconds the learner's function gives for reading 16 GB from "
                      "the HBM3 of an H100, over PCIe Gen5 and from an SSD."},
           "rate": {
               "title": "Your gb_per_second on your own computer",
               "intro": "Your function on arrays of 8 KB to 128 MB, on the computer you work on: the fastest of a few "
                        "tries for each size. The thin marks are the course's computer.",
               "head": "GB per second",
               "none": "Your `gb_per_second` stops with an error, takes too long, or does not give a number above 0, "
                       "so the picture waits.",
               "right": "Your function gives 8 GB/s for the stand-in too, so the bars measure your computer. Where do "
                        "they drop?",
               "off": "Your function gives {got} for the stand-in that reads 8 GB/s, so the bars are off by as much.",
               "alt": "Six bars with the gigabytes per second the learner's function measures for a sum of 8 KB, "
                      "64 KB, 512 KB, 4 MB, 32 MB and 128 MB on the learner's computer, each with a thin mark at the "
                      "rate of the course's computer."}},
    "nl": {"read": {
               "title": "Je read_seconds voor een model van 16 GB",
               "intro": "Je functie leest 16 GB één keer van drie plekken, op een logaritmische schaal van seconden.",
               "rows": {"HBM": ("HBM3 naast de chip", "3,35 TB/s"), "PCIe": ("PCIe Gen5 vanaf de computer", "64 GB/s"),
                        "SSD": ("SSD van de computer", "7,45 GB/s")},
               "none": "Je `read_seconds` stopt met een fout, duurt te lang of geeft niet voor elk van de drie plekken "
                       "een aantal seconden boven 0, dus wacht het plaatje.",
               "right": "Op deze drie geeft je functie de leestijden van de les: de SSD doet er 450 keer zo lang over als "
                        "het HBM.",
               "off": "Je functie geeft {got} voor de SSD, en 16 GB met 7,45 GB/s duurt 2,1 seconde.",
               "alt": "Drie balken op een logaritmische schaal met de seconden die de functie van de leerling geeft voor "
                      "het lezen van 16 GB uit het HBM3 van een H100, via PCIe Gen5 en van een SSD."},
           "rate": {
               "title": "Je gb_per_second op je eigen computer",
               "intro": "Je functie op arrays van 8 KB tot 128 MB, op de computer waarop je werkt: de snelste van een "
                        "paar pogingen per grootte. De dunne streepjes zijn de computer van de course.",
               "head": "GB per seconde",
               "none": "Je `gb_per_second` stopt met een fout, duurt te lang of geeft geen getal boven 0, dus wacht "
                       "het plaatje.",
               "right": "Je functie geeft ook 8 GB/s voor de stand-in, dus meten de balken je computer. Waar zakken de "
                        "balken in?",
               "off": "Je functie geeft {got} voor de stand-in van 8 GB/s, dus zitten de balken er evenveel naast.",
               "alt": "Zes balken met de gigabytes per seconde die de functie van de leerling meet voor een optelling van 8 KB, "
                      "64 KB, 512 KB, 4 MB, 32 MB en 128 MB op de computer van de leerling, elk met een dun streepje bij "
                      "de snelheid van de computer van de course."}},
}

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture
PICTURE_SIZES = [8_000, 64_000, 512_000, 4_000_000, 32_000_000, 128_000_000]
PICTURE_TRIES = 5     # calls for each size: the picture keeps the fastest


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _positive(got):
    """got as a float above 0, or _STOPPED when it is not such a number."""
    if got is _STOPPED or isinstance(got, (bool, np.bool_)) or not isinstance(got, (int, float, np.integer, np.floating)):
        return _STOPPED
    got = float(got)
    return got if math.isfinite(got) and got > 0 else _STOPPED


def read_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["read"]
    fn = _fn(seen)
    got = _call(lambda: fn(int(MODEL_BYTES), dict(LEVELS))) if fn else _STOPPED
    times = [_positive(got.get(k)) if isinstance(got, dict) else _STOPPED for k in LEVELS]
    if any(t is _STOPPED for t in times):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    lo, hi = -4.0, 2.0                      # 0.1 ms to 100 s
    x0, x1 = 0, 250

    def x(s: float) -> float:
        return x0 + (x1 - x0) * (min(max(math.log10(s), lo), hi) - lo) / (hi - lo)

    parts = []
    parts.append(_line(x0, 136, x1, 136, "var(--ea-line)"))
    for p, label in ((-3, "1 ms"), (-1, "0.1 s" if lang == "en" else "0,1 s"), (1, "10 s")):
        xp = x(10 ** p)
        parts.append(_line(xp, 133, xp, 139, "var(--ea-ink-3)"))
        parts.append(_t(xp, 151, label, 9.5, 400, "var(--ea-ink-3)", "middle"))
    for k, (key, s) in enumerate(zip(LEVELS, times)):
        name, bw = w["rows"][key]
        y = 8 + k * 44
        parts.append(_t(0, y + 10, name, 10.5, 600))
        parts.append(_t(250, y + 10, bw, 10, 400, "var(--ea-ink-3)", "end"))
        parts.append(_rect(x0, y + 16, max(2.0, x(s) - x0), 13, "fill:var(--ea-ink)", 1.5))
        parts.append(_t(326, y + 27, secs(s, lang), 10.5, 600, anchor="end"))
    want = read_seconds(MODEL_BYTES, LEVELS)
    right = all(math.isclose(s, want[k], rel_tol=1e-6) for k, s in zip(LEVELS, times))
    said = w["right"] if right else w["off"].format(got=secs(times[2], lang))
    body = _note(w["intro"]) + _svg(156, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def _fastest_rate(fn, values) -> object:
    """The highest rate fn gives on values in PICTURE_TRIES calls: the fastest sum; _STOPPED when a call fails."""
    best = None
    for _ in range(PICTURE_TRIES):
        got = _positive(_call(lambda: fn(values)))
        if got is _STOPPED:
            return _STOPPED
        best = got if best is None or got > best else best
    return best


def rate_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["rate"]
    fn = _fn(seen)
    stand_in = _positive(_call(lambda: fn(slow_array(800_000_000, 0.1)))) if fn else _STOPPED
    got = []
    for nbytes in PICTURE_SIZES if stand_in is not _STOPPED else []:   # no more calls once one stops
        got.append(_fastest_rate(fn, np.full(nbytes // 8, 1.0)))
        if got[-1] is _STOPPED:
            break
    if len(got) < len(PICTURE_SIZES) or any(r is _STOPPED for r in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    course = dict(zip(SIZES, course_rates()))
    top = max(max(got), max(course[b] for b in PICTURE_SIZES)) * 1.08
    x0, x1 = 64, 270
    parts = [_t(x0, 12, w["head"], 10, 600, "var(--ea-ink-2)")]
    for k, (nbytes, r) in enumerate(zip(PICTURE_SIZES, got)):
        y = 22 + k * 24
        parts.append(_t(x0 - 8, y + 12, size(nbytes), 10.5, 400, anchor="end"))
        parts.append(_rect(x0, y + 2, max(2.0, (x1 - x0) * min(r, top) / top), 14, "fill:var(--ea-ink)", 1.5))
        xc = x0 + (x1 - x0) * course[nbytes] / top
        parts.append(_line(xc, y - 1, xc, y + 19, "var(--ea-ink-3)", 1.6))
        parts.append(_t(326, y + 13, rate(r, lang), 10.5, 600, anchor="end"))
    right = check.same_value(stand_in, 8.0, rel=0.3)
    said = w["right"] if right else w["off"].format(got=rate(stand_in, lang))
    body = _note(w["intro"]) + _svg(22 + len(got) * 24 + 4, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: sums of every size on the course's computer ============================================================

DEMO_TEXT = {
    "en": {"title": "{size}: {rate}, one sum in {time}",
           "said": ["Small is slow: one sum of 1 KB takes 0.53 µs, mostly the fixed cost of a call, so few bytes "
                    "arrive per second.",
                    "Eight times the bytes in only 1.5 times the time: the fixed cost of the call weighs less.",
                    "The numbers come from the caches of the processor, and the fixed cost counts less and less.",
                    "The fastest sum on this computer: 512 KB fits in its caches.",
                    "Too large for the smallest caches, still small enough for a larger one: a little slower.",
                    "Still from a cache, at about the same rate as 4 MB.",
                    "Too large for the caches of this computer: the numbers come from RAM, at a quarter of the fastest rate.",
                    "From RAM too: four times the bytes take four times as long."],
           "alt": "Eight bars with the gigabytes per second of one sum of float64 numbers on the course's computer, for "
                  "arrays of 1 KB, 8 KB, 64 KB, 512 KB, 4 MB, 32 MB, 128 MB and 512 MB; the chosen size is dark."},
    "nl": {"title": "{size}: {rate}, één optelling in {time}",
           "said": ["Klein is traag: één optelling van 1 KB duurt 0,53 µs, vooral de vaste kosten van een aanroep, dus komen "
                    "er weinig bytes per seconde binnen.",
                    "Acht keer zoveel bytes in slechts 1,5 keer de tijd: de vaste kosten van de aanroep wegen minder zwaar.",
                    "De getallen komen uit de caches van de processor, en de vaste kosten tellen steeds minder.",
                    "De snelste optelling op deze computer: 512 KB past in de caches.",
                    "Te groot voor de kleinste caches, nog klein genoeg voor een grotere: iets trager.",
                    "Nog steeds uit een cache, met ongeveer dezelfde snelheid als 4 MB.",
                    "Te groot voor de caches van deze computer: de getallen komen uit het RAM, met een kwart van de "
                    "hoogste snelheid.",
                    "Ook uit het RAM: vier keer zoveel bytes duren vier keer zo lang."],
           "alt": "Acht balken met de gigabytes per seconde van één optelling van float64-getallen op de computer van de "
                  "course, voor arrays van 1 KB, 8 KB, 64 KB, 512 KB, 4 MB, 32 MB, 128 MB en 512 MB; de gekozen "
                  "grootte is donker."},
}
DEMO_VALUES = [b // 1000 for b in SIZES]   # kilobytes: 1, 8, 64, 512, 4000, 32000, 128000, 512000


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    k = min(range(len(DEMO_VALUES)), key=lambda i: abs(DEMO_VALUES[i] - value))
    rates = course_rates()
    top = max(rates) * 1.05
    title = w["title"].format(size=size(SIZES[k]), rate=rate(rates[k], lang), time=secs(COURSE_SECONDS[k], lang))
    parts = [_t(0, 15, title, 12.5, 600)]
    x0, x1 = 64, 270
    for i, (nbytes, r) in enumerate(zip(SIZES, rates)):
        y = 26 + i * 22
        mine = i == k
        parts.append(_t(x0 - 8, y + 12, size(nbytes), 10.5, 600 if mine else 400,
                        "var(--ea-ink)" if mine else "var(--ea-ink-2)", "end"))
        parts.append(_rect(x0, y + 2, (x1 - x0) * r / top, 14, "fill:var(--ea-ink)" if mine else "fill:var(--ea-ink-3)",
                           1.5))
        parts.append(_t(326, y + 13, rate(r, lang), 10.5, 600 if mine else 400,
                        "var(--ea-ink)" if mine else "var(--ea-ink-2)", "end"))
    return _svg(26 + len(SIZES) * 22 + 4, w["alt"], parts), w["said"][k]


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


def _shown_dict(v):
    """{got} and {want} of read_seconds: a dict with its numbers at four significant digits."""
    if isinstance(v, dict) and all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
                                   for x in v.values()):
        text_ = "{" + ", ".join(f"{check._as_code(k)}: {float(x):.4g}" for k, x in v.items()) + "}"
    else:
        text_ = check._as_value(v)
    if len(text_) > 90:
        text_ = text_[:87] + "..."
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


def _shown_rate(v):
    """{got} and {want} of gb_per_second: a number with three significant digits, such as `8.01` or `8.01e+09`."""
    if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
        text_ = f"{float(v):.3g}"
    else:
        text_ = check._as_value(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


def same_rate(a, b) -> bool:
    """Two measured rates within 30% of each other: the time of the stand-in's sum varies a little."""
    if isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, float, np.integer, np.floating)):
        return False
    a = float(a)
    return math.isfinite(a) and abs(a - b) <= 0.3 * abs(b)


def _number_got(got) -> bool:
    return isinstance(got, (int, float, np.integer, np.floating)) and not isinstance(got, (bool, np.bool_)) \
        and math.isfinite(float(got))


# exercise 1: read_seconds
def _r(line: str) -> str:
    return ("def read_seconds(nbytes, bandwidths):\n"
            "    seconds = {}\n"
            "    for name in bandwidths:\n"
            f"        seconds[name] = {line}\n"
            "    return seconds\n")


_R_START = _r("...")
_R = _r("nbytes / bandwidths[name]")
_R_UPSIDE = _r("bandwidths[name] / nbytes")
_R_TIMES = _r("nbytes * bandwidths[name]")
_R_ONE = _r('nbytes / bandwidths["HBM"]')

# exercise 2: gb_per_second
_G_HEAD = ("import time\n\n\ndef gb_per_second(values):\n"
           "    values.sum()\n"
           "    start = time.perf_counter()\n"
           "    values.sum()\n"
           "    seconds = time.perf_counter() - start\n")


def _g(line: str) -> str:
    return _G_HEAD + f"    return {line}\n"


_G_START = _g("...")
_G = _g("values.nbytes / seconds / 1e9")
_G_SIZE = _g("values.size / seconds / 1e9")
_G_BYTES = _g("values.nbytes / seconds")
_G_UPSIDE = _g("seconds / values.nbytes / 1e9")

WHERE = Sort(
    name="where",
    groups={
        "chip": {"en": "On the GPU chip", "nl": "Op de GPU-chip"},
        "next": {"en": "Next to the chip", "nl": "Naast de chip"},
        "outside": {"en": "Outside the GPU", "nl": "Buiten de GPU"},
    },
    items={
        "registers": {"en": "**Registers.** 256 KB in each of the 132 groups of cores: the numbers the cores compute "
                            "with right now.",
                      "nl": "**Registers.** 256 KB in elk van de 132 groepen cores: de getallen waarmee de cores op dit "
                            "moment rekenen."},
        "l1": {"en": "**L1 cache.** 256 KB in each group of cores, for numbers that are needed again soon.",
               "nl": "**L1-cache.** 256 KB in elke groep cores, voor getallen die snel weer nodig zijn."},
        "l2": {"en": "**L2 cache.** 50 MB, shared by all 132 groups of cores.",
               "nl": "**L2-cache.** 50 MB, gedeeld door alle 132 groepen cores."},
        "hbm": {"en": "**HBM3.** 80 GB in five stacks of memory chips, at 3.35 TB/s.",
                "nl": "**HBM3.** 80 GB in vijf stapels geheugenchips, met 3,35 TB/s."},
        "model": {"en": "**A model during training.** Its parameters, read again at every step of the training.",
                  "nl": "**Een model tijdens de training.** De parameters, bij elke stap van de training opnieuw "
                        "gelezen."},
        "ram": {"en": "**RAM.** Modules of DDR5 on the motherboard, 38.4 GB/s for each module.",
                "nl": "**RAM.** Modules DDR5 op het moederbord, 38,4 GB/s per module."},
        "pcie": {"en": "**PCIe Gen5.** The connection that brings data from the computer to the GPU, 64 GB/s each way.",
                 "nl": "**PCIe Gen5.** De verbinding waarover data van de computer naar de GPU gaat, 64 GB/s per richting."},
        "ssd": {"en": "**SSD.** A drive of 2 TB that keeps its data without power, read at 7.45 GB/s.",
                "nl": "**SSD.** Een opslagschijf van 2 TB, gelezen met 7,45 GB/s; de data blijft er ook zonder stroom op staan."},
    },
    solution={"registers": "chip", "l1": "chip", "l2": "chip", "hbm": "next", "model": "next", "ram": "outside",
              "pcie": "outside", "ssd": "outside"},
    nudge={
        "registers": {"en": "Where do the cores find the numbers they compute with at this very moment?",
                      "nl": "Waar vinden de cores de getallen voor de berekening van dit moment?"},
        "l1": {"en": "Is this cache inside each group of cores, or outside the chip?",
               "nl": "Zit deze cache in elke groep cores, of buiten de chip?"},
        "l2": {"en": "Which memory do all groups of cores share without leaving the chip?",
               "nl": "Welk geheugen delen alle groepen cores zonder de chip te verlaten?"},
        "hbm": {"en": "Where are the stacks of HBM: on the chip itself, or beside it?",
                "nl": "Waar staan de stapels HBM: op de chip zelf, of ernaast?"},
        "model": {"en": "In which memory of the GPU does a model fit whole, with billions of parameters?",
                  "nl": "In welk geheugen van de GPU past een model helemaal, met miljarden parameters?"},
        "ram": {"en": "Is the motherboard part of the GPU?",
                "nl": "Hoort het moederbord bij de GPU?"},
        "pcie": {"en": "Does this connection run inside the GPU, or from the computer to it?",
                 "nl": "Loopt deze verbinding binnen de GPU, of van de computer naar de GPU?"},
        "ssd": {"en": "Where do the files of a computer stay when the power is off?",
                "nl": "Waar blijven de bestanden van een computer als de stroom uit is?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight items again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht onderdelen opnieuw uit."},
        "empty": {"en": "Put each item under *On the GPU chip*, *Next to the chip* or *Outside the GPU*.",
                  "nl": "Zet elk onderdeel onder *Op de GPU-chip*, *Naast de chip* of *Buiten de GPU*."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs somewhere else.",
                  "nl": "*{item}* hoort ergens anders."},
        "correct": {"en": "All {n} are right. On the chip: a few hundred KB for each group of cores and an L2 cache of "
                          "50 MB. Next to it: 80 GB of HBM3. Outside the GPU: everything that comes in over PCIe. Which "
                          "of the three holds the most bytes, and which delivers them fastest?",
                    "nl": "Alle {n} kloppen. Op de chip: een paar honderd KB per groep cores en een L2-cache van 50 MB. "
                          "Ernaast: 80 GB HBM3. Buiten de GPU: alles wat via PCIe binnenkomt. Welke van de drie bevat de "
                          "meeste bytes, en welke levert de bytes het snelst?"},
    },
)

EXERCISES = {
    "read": FunctionBox(
        name="read",
        function="read_seconds",
        start=_R_START,
        solution=_R,
        show=_shown_dict,
        visible=[(MODEL_BYTES, LEVELS)],
        hidden=[(_Bytes(80_000_000_000), _Dict({"HBM": H100_HBM_RATE}, '{"HBM": 3.35e12}')),
                (1_000, {"a": 10, "b": 1000}), (0, {"x": 5.0}), (5_000, {"near": 2_500, "far": 50}), (8_000_000, {})],
        first=[_DOTS],
        picture=read_picture,
        mistakes=[
            Mistake("upside-down", wrong=_R_UPSIDE,
                    en="`{call}` gives {got}: the bandwidth divided by the bytes, and it should give {want}. Seconds are the bytes divided by the bytes per second: which of the two goes on top?",
                    nl="`{call}` geeft {got}: de bandbreedte gedeeld door de bytes, en het moet {want} zijn. Seconden zijn de bytes gedeeld door de bytes per seconde: welke van de twee staat boven de deelstreep?"),
            Mistake("times", wrong=_R_TIMES,
                    en="`{call}` gives {got}: the bytes times the bandwidth, and it should give {want}. How many seconds does it take to read `nbytes` bytes at so many bytes per second?",
                    nl="`{call}` geeft {got}: de bytes keer de bandbreedte, en het moet {want} zijn. Hoeveel seconden duurt het om `nbytes` bytes te lezen met zoveel bytes per seconde?"),
            Mistake("one-place", wrong=_R_ONE,
                    en="`{call}` gives {got}: the same time for every place, and it should give {want}. Does each round of the loop use the bandwidth of its own `name`?",
                    nl="`{call}` geeft {got}: dezelfde tijd voor elke plek, en het moet {want} zijn. Gebruikt elke ronde van de loop de bandbreedte van de eigen `name`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` so that every round of the loop puts `nbytes` divided by the bandwidth of `name` in `seconds[name]`.",
                      "nl": "Vervang de `...` zodat elke ronde van de loop `nbytes` gedeeld door de bandbreedte van `name` in `seconds[name]` zet."},
            "arguments": {"en": "The check calls `{call}`, with a number of bytes and a dict of bandwidths, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(nbytes, bandwidths)`?",
                          "nl": "De check roept `{call}` aan, met een aantal bytes en een dict met bandbreedtes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(nbytes, bandwidths)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every round put `nbytes` divided by the bandwidth of `name` in `seconds[name]`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zet elke ronde `nbytes` gedeeld door de bandbreedte van `name` in `seconds[name]`?"},
            "almost": {"en": "Right for the model of 16 GB. But `{call}` gives {got}, and it should give {want}. Does your function use `nbytes` and `bandwidths`, and no fixed numbers?",
                       "nl": "Goed voor het model van 16 GB. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `nbytes` en `bandwidths`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return seconds`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return seconds`?"},
            "correct": {"en": "Right in all {n} cases. A model of 16 GB takes 4.8 ms from the HBM, 0.25 seconds over PCIe and 2.1 seconds from the SSD. In the picture: how many times as long does the SSD take as the HBM?",
                        "nl": "Goed in alle {n} gevallen. Een model van 16 GB duurt 4,8 ms uit het HBM, 0,25 seconde via PCIe en 2,1 seconde van de SSD. In het plaatje: hoeveel keer zo lang doet de SSD erover als het HBM?"},
        },
    ),
    "rate": FunctionBox(
        name="rate",
        function="gb_per_second",
        start=_G_START,
        solution=_G,
        compare=same_rate,
        show=_shown_rate,
        visible=[(slow_array(800_000_000, 0.1),)],
        hidden=[(slow_array(4_000_000, 0.05),), (slow_array(120_000_000, 0.05),)],
        first=[_DOTS],
        picture=rate_picture,
        mistakes=[
            Mistake("size", wrong=_G_SIZE,
                    en="`{call}` gives {got}, and it should give about {want}: 8 times too little. `values.size` counts the numbers in the array, not its bytes. How many bytes does each float64 number take?",
                    nl="`{call}` geeft {got}, en het moet ongeveer {want} zijn: 8 keer te weinig. `values.size` telt de getallen in de array, niet de bytes. Hoeveel bytes neemt elk float64-getal in?"),
            Mistake("too-big", matches=lambda got, text_: _number_got(got) and float(got) >= 1000,
                    en="`{call}` gives {got}: more than 1,000 gigabytes per second, more than the HBM of an H100 delivers. Does your function divide by `1e9`, to turn bytes into gigabytes?",
                    nl="`{call}` geeft {got}: meer dan 1.000 gigabyte per seconde, meer dan het HBM van een H100 levert. Deelt je functie door `1e9`, om bytes in gigabytes om te zetten?"),
            Mistake("upside-down", matches=lambda got, text_: _number_got(got) and 0 <= float(got) < 1e-3,
                    en="`{call}` gives {got}: a tiny number, as if your function divided the seconds by the bytes. A rate is bytes per second: which of the two goes on top?",
                    nl="`{call}` geeft {got}: een piepklein getal, alsof je functie de seconden door de bytes deelt. Een snelheid is bytes per seconde: welke van de twee staat boven de deelstreep?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the gigabytes per second of the sum: the bytes of `values` divided by `seconds`, and by `1e9`.",
                      "nl": "Vervang de `...` door de gigabytes per seconde van de optelling: de bytes van `values` gedeeld door `seconds`, en door `1e9`."},
            "arguments": {"en": "The check calls `{call}`, with a stand-in for an array, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(values)`?",
                          "nl": "De check roept `{call}` aan, met een stand-in voor een array, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(values)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give about {want}: the stand-in holds as many bytes as it says, and each sum takes as many seconds. Does your function divide the bytes of `values` by `seconds`, and by `1e9`?",
                      "nl": "`{call}` geeft {got}, en het moet ongeveer {want} zijn: de stand-in bevat zoveel bytes als er staat, en elke optelling duurt zoveel seconden. Deelt je functie de bytes van `values` door `seconds`, en door `1e9`?"},
            "almost": {"en": "Right for the first stand-in. But `{call}` gives {got}, and it should give about {want}. Does your function use the bytes of `values`, and no fixed number?",
                       "nl": "Goed voor de eerste stand-in. Maar `{call}` geeft {got}, en het moet ongeveer {want} zijn. Gebruikt je functie de bytes van `values`, en geen vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does the last line start with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de laatste regel met `return`?"},
            "correct": {"en": "Right in all {n} cases: 800,000,000 bytes in 0.1 seconds is 8 GB per second. In the picture: how many GB per second does your own computer reach, and at which size does it drop?",
                        "nl": "Goed in alle {n} gevallen: 800.000.000 bytes in 0,1 seconde is 8 GB per seconde. In het plaatje: hoeveel GB per seconde haalt je eigen computer, en bij welke grootte zakt de snelheid in?"},
        },
    ),
    "where": WHERE,
}

# The lesson is done when these are correct. "order" is the choice question after the demo.
REQUIRED = ["read", "rate", "where", "order"]

_SORTED = dict(WHERE.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("read", _R_START, "empty", None),
    ("read", "", "empty", None),
    ("read", _R, "correct", None),
    ("read", "def read_seconds(nbytes, bandwidths):\n"
             "    seconds = {}\n"
             "    for name, bandwidth in bandwidths.items():\n"
             "        seconds[name] = nbytes / bandwidth\n"
             "    return seconds\n", "correct", None),
    ("read", "def read_seconds(nbytes, bandwidths):\n"
             "    return {name: nbytes / bandwidths[name] for name in bandwidths}\n", "correct", None),
    ("read", _R_UPSIDE, "wrong", "upside-down"),
    ("read", _R_TIMES, "wrong", "times"),
    ("read", _R_ONE, "wrong", "one-place"),
    ("read", "def read_seconds(nbytes, bandwidths):\n"
             "    return {'HBM': 0.004776119402985075, 'PCIe': 0.25, 'SSD': 2.1476510067114094}\n", "almost", None),
    ("read", _r("nbytes / bandwidth[name]"), "error", None),
    ("read", _r("nbytes / bandwidths[name] * 1000"), "wrong", None),
    ("read", _R.replace("def read_seconds(nbytes, bandwidths):", "def read_seconds(nbytes):"), "error", "arguments"),
    ("read", _R.replace("    return seconds\n", ""), "wrong", "no-return"),
    ("read", _R.replace("    return seconds\n", "    print(seconds)\n"), "wrong", "prints"),
    ("read", _R.replace("    return seconds\n", "    extra = ...\n    return seconds\n"), "wrong", "dots-left"),
    ("read", "x = 1\n", "wrong", "no-function"),
    ("rate", _G_START, "empty", None),
    ("rate", "", "empty", None),
    ("rate", _G, "correct", None),
    ("rate", _g("values.size * 8 / seconds / 1e9"), "correct", None),
    ("rate", _g("len(values) * values.itemsize / seconds / 1e9"), "correct", None),
    ("rate", _g("values.nbytes / 1e9 / seconds"), "correct", None),
    ("rate", _G_SIZE, "wrong", "size"),
    ("rate", _g("len(values) / seconds / 1e9"), "wrong", "size"),
    ("rate", _G_BYTES, "wrong", "too-big"),
    ("rate", _g("values.nbytes / seconds / 1e6"), "wrong", "too-big"),
    ("rate", _G_UPSIDE, "wrong", "upside-down"),
    ("rate", _g("8.0"), "almost", None),
    ("rate", _g("value.nbytes / seconds / 1e9"), "error", None),
    ("rate", _G.replace("def gb_per_second(values):", "def gb_per_second(values, seconds):"), "error", "arguments"),
    ("rate", _G_HEAD, "wrong", "no-return"),
    ("rate", _g("values.nbytes / seconds / 1e9").replace("    return ", "    print(").replace(" / 1e9\n", " / 1e9)\n"),
     "wrong", "prints"),
    ("rate", _g("values.nbytes / seconds / 1e9").replace("    return", "    extra = ...\n    return"), "wrong",
     "dots-left"),
    ("rate", "x = 1\n", "wrong", "no-function"),
    ("where", {}, "empty", None),
    ("where", _SORTED, "correct", None),
    ("where", {"registers": "chip", "ssd": "outside"}, "wrong", "unfinished"),
] + [("where", _SORTED | {k: ("next" if v != "next" else "chip")}, "wrong", f"wrong:{k}") for k, v in _SORTED.items()]
