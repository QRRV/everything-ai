# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The checks of the lesson sys-moores-law: the transistor counts of fifteen chips and the training compute of four
models, the doubling times they give, the exercises with their known mistakes, the pictures of the learner's doubling
times and fitted lines, and the demo of five kinds of growth side by side.

The numbers are the ones the makers published, read on 28 November 2026: Intel's timeline poster of its chips (the
4004 to the third generation of Core) and its page on the 4004 (2,300 transistors, 15 November 1971); NVIDIA's page on
the Ampere architecture (A100, 54.2 billion); Tom's Hardware and IEEE Spectrum on the B200 of March 2024 (208 billion
over two dies, against 80 billion for the H100). The training compute of BERT-Large, T5-11B and GPT-3 175B is table D.1
of Brown et al. (2020), OpenAI's own estimate; that of Llama 3.1 405B is the Llama team at Meta (2024), section 1.
The growth per two years of peak FLOP/s, DRAM bandwidth and interconnect bandwidth is Gholami et al. (2024), AI and
Memory Wall; the growth of training compute per year is Epoch AI (Sevilla and Roldan, May 2024, and the trends page
updated 5 November 2026). Computerworld (15 November 2004) on the 3.8 GHz Pentium 4. Every value a learner's code gives
back is shown through html.escape."""
import html
import math

import numpy as np

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-moores-law"
LANGS = ("en", "nl")

# ======== what the makers published ========================================================================================

CHIPS = [
    {"name": "Intel 4004", "year": 1971, "transistors": 2_300},
    {"name": "Intel 8080", "year": 1974, "transistors": 4_500},
    {"name": "Intel 8086", "year": 1978, "transistors": 29_000},
    {"name": "Intel 286", "year": 1982, "transistors": 134_000},
    {"name": "Intel386", "year": 1985, "transistors": 275_000},
    {"name": "Intel486", "year": 1989, "transistors": 1_200_000},
    {"name": "Pentium", "year": 1993, "transistors": 3_100_000},
    {"name": "Pentium III", "year": 1999, "transistors": 9_500_000},
    {"name": "Pentium 4", "year": 2000, "transistors": 42_000_000},
    {"name": "Core 2 Duo", "year": 2006, "transistors": 291_000_000},
    {"name": "Core, 2nd generation", "year": 2010, "transistors": 1_160_000_000},
    {"name": "Core, 3rd generation", "year": 2012, "transistors": 1_400_000_000},
    {"name": "NVIDIA A100", "year": 2020, "transistors": 54_200_000_000},
    {"name": "NVIDIA H100", "year": 2022, "transistors": 80_000_000_000},
    {"name": "NVIDIA B200", "year": 2024, "transistors": 208_000_000_000},   # two dies in one package
]
TRAININGS = [
    {"name": "BERT-Large", "year": 2018, "flop": 5.33e20},
    {"name": "T5-11B", "year": 2019, "flop": 3.30e22},
    {"name": "GPT-3 175B", "year": 2020, "flop": 3.14e23},
    {"name": "Llama 3.1 405B", "year": 2024, "flop": 3.8e25},
]
GPT3_FLOP = 3.14e23
LLAMA_FLOP = 3.8e25
PER_TWO_YEARS = {"flops": 3.0, "dram": 1.6, "links": 1.4, "moore": 2.0}   # Gholami et al. (2024); Moore (1975)
EPOCH_FRONTIER = 5.3   # times per year, the ten largest trainings at each moment, 2010 to May 2024


class _Shown(np.ndarray):
    """An array that a check message writes as the lesson names it, such as C.YEARS. The learner's code gets a plain
    copy."""

    def __repr__(self):
        return getattr(self, "shown", "values")

    def __deepcopy__(self, memo):
        return np.array(self)


def _named(values, name: str) -> np.ndarray:
    v = np.array(values, dtype=float).view(_Shown)
    v.shown = name
    return v


YEARS = _named([c["year"] for c in CHIPS], "C.YEARS")
COUNTS = _named([c["transistors"] for c in CHIPS], "C.COUNTS")
TRAIN_YEARS = _named([t["year"] for t in TRAININGS], "C.TRAIN_YEARS")
TRAIN_FLOP = _named([t["flop"] for t in TRAININGS], "C.TRAIN_FLOP")


def doubling_time(start, end, years):
    return years / np.log2(end / start)


def fit_doubling(years, counts):
    slope = np.polyfit(years, np.log2(counts), 1)[0]
    return 1 / slope


def _fit(years, values) -> tuple[float, float]:
    """The slope and the bias term of the straight line through the years and log2 of the values."""
    slope, bias = np.polyfit(np.asarray(years, dtype=float), np.log2(np.asarray(values, dtype=float)), 1)
    return float(slope), float(bias)


SLOPE = _fit(YEARS, COUNTS)[0]


def facts() -> dict:
    """The numbers the texts give."""
    first, p4, h100 = CHIPS[0], CHIPS[8], CHIPS[13]
    return {
        "chips_fit": fit_doubling(np.array(YEARS), np.array(COUNTS)),
        "train_fit": fit_doubling(np.array(TRAIN_YEARS), np.array(TRAIN_FLOP)),
        "h100_times": h100["transistors"] / first["transistors"],
        "h100_doublings": math.log2(h100["transistors"] / first["transistors"]),
        "h100_time": doubling_time(first["transistors"], h100["transistors"], h100["year"] - first["year"]),
        "p4_times": p4["transistors"] / first["transistors"],
        "p4_doublings": math.log2(p4["transistors"] / first["transistors"]),
        "p4_time": doubling_time(first["transistors"], p4["transistors"], p4["year"] - first["year"]),
        "llama_times": LLAMA_FLOP / GPT3_FLOP,
        "llama_doublings": math.log2(LLAMA_FLOP / GPT3_FLOP),
        "llama_time": doubling_time(GPT3_FLOP, LLAMA_FLOP, 4),
        "flops_time": doubling_time(1, PER_TWO_YEARS["flops"], 2),
        "dram_time": doubling_time(1, PER_TWO_YEARS["dram"], 2),
        "links_time": doubling_time(1, PER_TWO_YEARS["links"], 2),
        "wall_20": (PER_TWO_YEARS["flops"] / PER_TWO_YEARS["dram"]) ** 10,
        "moore_2022": round(2300 * 2 ** (51 / 2)),
        "epoch_months": 12 / math.log2(EPOCH_FRONTIER),
    }


# ======== words and numbers ================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_UNITS = {"en": [(1e12, "trillion"), (1e9, "billion"), (1e6, "million")],
          "nl": [(1e12, "biljoen"), (1e9, "miljard"), (1e6, "miljoen")]}


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _dutch(text_: str) -> str:
    """1,234.5 as Dutch writes it: 1.234,5."""
    return text_.replace(",", "_").replace(".", ",").replace("_", ".")


def num(x: float, lang: str) -> str:
    """A whole number with its thousands marked: 29,000; in Dutch 29.000."""
    out = f"{round(x):,}"
    return _dutch(out) if lang == "nl" else out


def dec(x: float, lang: str, digits: int = 1) -> str:
    """A number with a fixed number of decimals: 2.04; in Dutch 2,04."""
    out = f"{x:,.{digits}f}"
    return _dutch(out) if lang == "nl" else out


def big(x: float, lang: str) -> str:
    """A large number in words: 80 billion, 54.2 billion, 2,300; in Dutch 80 miljard, 54,2 miljard, 2.300."""
    lang = _lang(lang)
    for unit, word in _UNITS[lang]:
        if x >= unit:
            v = f"{x / unit:.3g}"
            if "e" in v:
                v = num(x / unit, lang)
            return f"{_dutch(v) if lang == 'nl' else v} {word}"
    return num(x, lang)


def span(d: float, lang: str) -> str:
    """A doubling time: 2.04 years from a year up, 6.9 months under a year, 3.0 days under a month; in Dutch 2,04 jaar,
    6,9 maanden, 3,0 dagen."""
    if d < 1 / 12:
        days = d * 365.25
        if days < 0.1:
            return "minder dan 0,1 dag" if lang == "nl" else "less than 0.1 days"
        return f"{dec(days, lang)} {'dagen' if lang == 'nl' else 'days'}"
    if d < 1:
        return f"{dec(d * 12, lang)} {'maanden' if lang == 'nl' else 'months'}"
    return f"{dec(d, lang, 2)} {'jaar' if lang == 'nl' else 'years'}"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;'
            f'font-weight:{weight};fill:{fill};white-space:pre" xml:space="preserve">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-line)", width: float = 1) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{stroke};stroke-width:{width}"/>')


def _rect(x: float, y: float, w: float, h: float, style: str, rx: float = 3) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" style="{style}"/>'


def _dot(x: float, y: float, fill: str = "var(--ea-ink-2)", r: float = 2.6) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{fill}"/>'


class _Chart:
    """A chart with years across and a log scale up: each power of ten one step."""

    def __init__(self, years: tuple, powers: tuple, box: tuple):
        self.y0, self.y1 = years
        self.p0, self.p1 = powers
        self.left, self.top, self.right, self.bottom = box

    def x(self, year: float) -> float:
        return self.left + (self.right - self.left) * (year - self.y0) / (self.y1 - self.y0)

    def y(self, value: float) -> float:
        p = math.log10(value)
        return self.bottom - (self.bottom - self.top) * (p - self.p0) / (self.p1 - self.p0)

    def y_of_log2(self, l2: float) -> float:
        return self.bottom - (self.bottom - self.top) * (l2 * math.log10(2) - self.p0) / (self.p1 - self.p0)

    def grid(self, labels: dict, ticks: list) -> list:
        parts = []
        for p, label in labels.items():
            y = self.y(10 ** p)
            parts.append(_line(self.left, y, self.right, y))
            parts.append(_t(self.left - 4, y + 3.5, label, 9.5, 400, "var(--ea-ink-3)", "end"))
        for year in ticks:
            parts.append(_t(self.x(year), self.bottom + 13, str(year), 9.5, 400, "var(--ea-ink-3)", "middle"))
        return parts

    def line(self, slope: float, bias: float, stroke: str = "var(--ea-ink)", width: float = 1.6) -> str:
        """The straight line log2(value) = slope * year + bias, cut off at the edges of the chart."""
        top_l2, bottom_l2 = self.p1 / math.log10(2), self.p0 / math.log10(2)
        a, b = self.y0, self.y1
        if slope == 0 and not bottom_l2 <= bias <= top_l2:
            return ""
        if slope > 0:
            a, b = max(a, (bottom_l2 - bias) / slope), min(b, (top_l2 - bias) / slope)
        elif slope < 0:
            a, b = max(a, (top_l2 - bias) / slope), min(b, (bottom_l2 - bias) / slope)
        if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
            return ""
        return _line(self.x(a), self.y_of_log2(slope * a + bias), self.x(b), self.y_of_log2(slope * b + bias), stroke, width)


# ======== the picture of the idea: fifteen chips on a log scale ============================================================

IDEA_TEXT = {
    "en": {"powers": {3: "1 thousand", 6: "1 million", 9: "1 billion", 12: "1 trillion"},
           "head": "transistors per chip", "line": "one doubling every {d}",
           "intel": "Intel processors", "nvidia": "NVIDIA GPUs",
           "alt": "Fifteen chips on a log scale of transistors, from the Intel 4004 of 1971 with 2,300 to the NVIDIA "
                  "B200 of 2024 with 208 billion. The dots lie close to a straight line: one doubling every 2.04 years."},
    "nl": {"powers": {3: "1 duizend", 6: "1 miljoen", 9: "1 miljard", 12: "1 biljoen"},
           "head": "transistors per chip", "line": "een verdubbeling per {d}",
           "intel": "processors van Intel", "nvidia": "GPU's van NVIDIA",
           "alt": "Vijftien chips op een logaritmische schaal van transistors, van de Intel 4004 uit 1971 met 2.300 tot "
                  "de NVIDIA B200 uit 2024 met 208 miljard. De punten liggen dicht bij een rechte lijn: een verdubbeling "
                  "per 2,04 jaar."},
}
_LABELED = {0: (8, 9, "start"), 8: (-6, -8, "end"), 13: (-6, -8, "end")}


def _chips_chart(lang: str, slope: float, bias: float, stroke: str = "var(--ea-ink)") -> list:
    w = IDEA_TEXT[lang]
    ch = _Chart((1968, 2026), (3, 12), (62, 22, 322, 176))
    parts = ch.grid(w["powers"], [1980, 2000, 2020])
    parts.append(_t(62, 12, w["head"], 10.5, 600, "var(--ea-ink-2)"))
    parts.append(ch.line(slope, bias, stroke))
    for k, c in enumerate(CHIPS):
        gpu = c["name"].startswith("NVIDIA")
        x, y = ch.x(c["year"]), ch.y(c["transistors"])
        parts.append(_dot(x, y, "var(--ea-ink)" if gpu else "var(--ea-ink-2)", 3.0 if gpu else 2.6))
        if k in _LABELED:
            dx, dy, anchor = _LABELED[k]
            parts.append(_t(x + dx, y + dy, c["name"], 9.5, 400, "var(--ea-ink-2)", anchor))
    return parts


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    slope, bias = _fit(YEARS, COUNTS)
    parts = _chips_chart(lang, slope, bias)
    parts += [_dot(70, 205, "var(--ea-ink-2)"), _t(78, 208.5, w["intel"], 10, 400, "var(--ea-ink-2)"),
              _dot(178, 205, "var(--ea-ink)", 3.0), _t(186, 208.5, w["nvidia"], 10, 400, "var(--ea-ink-2)")]
    parts.append(_t(322, 166, w["line"].format(d=span(1 / slope, lang)), 10.5, 600, anchor="end"))
    return _svg(216, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"time": {
               "title": "Your doubling_time on five kinds of growth",
               "intro": "Your function on five growths: the training compute from GPT-3 to Llama 3.1 405B in 4 years, "
                        "peak FLOP/s and two kinds of bandwidth at their growth per two years, and the transistors from "
                        "the 4004 to the H100 in 51 years.",
               "rows": ["compute of the largest trainings", "peak FLOP/s of a server", "transistors per chip",
                        "bandwidth of memory (DRAM)", "bandwidth between chips"],
               "head": "years per doubling",
               "none": "Your `doubling_time` stops with an error, takes too long, or gives something other than a "
                       "number above 0, so the picture waits.",
               "right": "On these five, your function gives the doubling times of the lesson. Memory doubles every "
                        "2.95 years and peak FLOP/s every 1.26 years: the gap between the two is the memory wall.",
               "off": "Your function gives {got} for the transistors from the 4004 to the H100, and 2,300 to 80 "
                      "billion in 51 years is 25.05 doublings: one every 2.04 years.",
               "alt": "Five bars, one per kind of growth, each as long as the years per doubling the learner's "
                      "function gives; the compute of the largest trainings doubles fastest, the bandwidth between "
                      "chips slowest."},
           "fit": {
               "title": "Your fit_doubling on chips and trainings",
               "intro": "The line of your doubling time through the middle of the points: the fifteen chips of "
                        "`C.YEARS` and `C.COUNTS`, and the four trainings of `C.TRAIN_YEARS` and `C.TRAIN_FLOP`.",
               "chips": "chips: one doubling every {d}", "trains": "trainings: one doubling every {d}",
               "powers": {20: "10²⁰", 22: "10²²", 24: "10²⁴", 26: "10²⁶"}, "flop": "FLOPs of the training",
               "none": "Your `fit_doubling` stops with an error, takes too long, or gives something other than a "
                       "number above 0, so the picture waits.",
               "right": "Your lines follow the points. The chips double every 2.04 years; the trainings every 4.9 "
                        "months, five times as often.",
               "off": "Your function gives {got} for the chips, and a straight line through the log2 of the counts "
                      "rises one doubling every 2.04 years.",
               "alt": "Two charts on a log scale: the fifteen chips from 1971 to 2024 and the four trainings from 2018 "
                      "to 2024, each with a line at the doubling time the learner's function gives."}},
    "nl": {"time": {
               "title": "Je doubling_time op vijf soorten groei",
               "intro": "Je functie op vijf soorten groei: de compute van de training van GPT-3 tot Llama 3.1 405B in 4 "
                        "jaar, de hoogste FLOP/s en twee soorten bandbreedte met de groei per twee jaar, en de transistors van "
                        "de 4004 tot de H100 in 51 jaar.",
               "rows": ["compute van de grootste trainingen", "hoogste FLOP/s van een server", "transistors per chip",
                        "bandbreedte van het geheugen (DRAM)", "bandbreedte tussen chips"],
               "head": "jaar per verdubbeling",
               "none": "Je `doubling_time` stopt met een fout, duurt te lang, of geeft iets anders dan een getal boven "
                       "0, dus wacht de afbeelding nog.",
               "right": "Op deze vijf geeft je functie de verdubbelingstijden van de les. Het geheugen verdubbelt elke "
                        "2,95 jaar en de hoogste FLOP/s elke 1,26 jaar: het gat tussen de twee is de memory wall.",
               "off": "Je functie geeft {got} voor de transistors van de 4004 tot de H100, en 2.300 tot 80 miljard in 51 "
                      "jaar is 25,05 verdubbelingen: één per 2,04 jaar.",
               "alt": "Vijf balken, één per soort groei, elk zo lang als de jaren per verdubbeling die de functie van de "
                      "leerling geeft; de compute van de grootste trainingen verdubbelt het snelst, de bandbreedte "
                      "tussen chips het langzaamst."},
           "fit": {
               "title": "Je fit_doubling op chips en trainingen",
               "intro": "De lijn van je verdubbelingstijd door het midden van de punten: de vijftien chips van "
                        "`C.YEARS` en `C.COUNTS`, en de vier trainingen van `C.TRAIN_YEARS` en `C.TRAIN_FLOP`.",
               "chips": "chips: een verdubbeling per {d}", "trains": "trainingen: een verdubbeling per {d}",
               "powers": {20: "10²⁰", 22: "10²²", 24: "10²⁴", 26: "10²⁶"}, "flop": "FLOPs van de training",
               "none": "Je `fit_doubling` stopt met een fout, duurt te lang, of geeft iets anders dan een getal boven 0, "
                       "dus wacht de afbeelding nog.",
               "right": "Je lijnen volgen de punten. De chips verdubbelen elke 2,04 jaar; de trainingen elke 4,9 "
                        "maanden, vijf keer zo vaak.",
               "off": "Je functie geeft {got} voor de chips, en een rechte lijn door de log2 van de aantallen stijgt één "
                      "verdubbeling per 2,04 jaar.",
               "alt": "Twee grafieken op een logaritmische schaal: de vijftien chips van 1971 tot 2024 en de vier "
                      "trainingen van 2018 tot 2024, elk met een lijn bij de verdubbelingstijd die de functie van de "
                      "leerling geeft."}},
}

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture


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


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6)


GROWTHS = [(GPT3_FLOP, LLAMA_FLOP, 4), (1, PER_TWO_YEARS["flops"], 2), (2_300, 80_000_000_000, 51),
           (1, PER_TWO_YEARS["dram"], 2), (1, PER_TWO_YEARS["links"], 2)]


def time_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["time"]
    fn = _fn(seen)
    got = []
    for args in GROWTHS if fn else []:   # no more calls once one stops
        got.append(_positive(_call(lambda a=args: fn(*a))))
        if got[-1] is _STOPPED:
            break
    if len(got) < len(GROWTHS) or any(d is _STOPPED for d in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    want = [doubling_time(*a) for a in GROWTHS]
    top = max(max(want), min(max(got), 50.0)) * 1.05
    x0, x1 = 0, 240
    parts = [_t(0, 18, w["head"], 11, 600, "var(--ea-ink-2)")]
    for k, (label, d) in enumerate(zip(w["rows"], got)):
        y = 28 + k * 34
        memory = k >= 3
        parts.append(_t(0, y + 11, label, 10.5, 400))
        parts.append(_rect(x0, y + 16, max(2.0, (x1 - x0) * min(d, top) / top), 11,
                           "fill:var(--ea-ink-3)" if memory else "fill:var(--ea-ink)", 1.5))
        parts.append(_t(326, y + 26, span(d, lang), 10.5, 600 if k == 2 else 400,
                        "var(--ea-ink)" if k == 2 else "var(--ea-ink-2)", "end"))
    right = all(_close(d, v) for d, v in zip(got, want))
    said = w["right"] if right else w["off"].format(got=span(got[2], lang))
    body = _note(w["intro"]) + _svg(28 + len(got) * 34, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def _through_middle(years, values, d: float) -> tuple[float, float]:
    """The slope and bias of the line with one doubling per d years through the middle of the points."""
    ys, l2 = np.asarray(years, dtype=float), np.log2(np.asarray(values, dtype=float))
    slope = 1 / d
    return slope, float(l2.mean() - slope * ys.mean())


def fit_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["fit"]
    fn = _fn(seen)
    got = []
    for years, values in ((YEARS, COUNTS), (TRAIN_YEARS, TRAIN_FLOP)) if fn else []:
        got.append(_positive(_call(lambda y=years, v=values: fn(np.array(y), np.array(v)))))
        if got[-1] is _STOPPED:
            break
    if len(got) < 2 or any(d is _STOPPED for d in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    chips = _chips_chart(lang, *_through_middle(YEARS, COUNTS, got[0]))
    chips.append(_t(322, 166, w["chips"].format(d=span(got[0], lang)), 10.5, 600, anchor="end"))
    ch = _Chart((2017.5, 2024.5), (20, 26), (62, 232, 322, 362))
    trains = ch.grid(w["powers"], [2018, 2020, 2022, 2024])
    trains.append(_t(62, 222, w["flop"], 10.5, 600, "var(--ea-ink-2)"))
    trains.append(ch.line(*_through_middle(TRAIN_YEARS, TRAIN_FLOP, got[1])))
    for t in TRAININGS:
        trains.append(_dot(ch.x(t["year"]), ch.y(t["flop"]), "var(--ea-ink-2)"))
    trains.append(_t(ch.x(2024) - 7, ch.y(LLAMA_FLOP) - 14, TRAININGS[-1]["name"], 9.5, 400, "var(--ea-ink-2)", "end"))
    trains.append(_t(ch.x(2018) + 7, ch.y(TRAININGS[0]["flop"]) + 10, TRAININGS[0]["name"], 9.5, 400,
                     "var(--ea-ink-2)"))
    trains.append(_t(322, 394, w["trains"].format(d=span(got[1], lang)), 10.5, 600, anchor="end"))
    f = facts()
    right = _close(got[0], f["chips_fit"]) and _close(got[1], f["train_fit"])
    said = w["right"] if right else w["off"].format(got=span(got[0], lang))
    body = _note(w["intro"]) + _svg(400, w["alt"], chips + trains) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: five kinds of growth side by side ======================================================================

DEMO_TEXT = {
    "en": {"title": "after {t} years", "title_1": "after 1 year", "title_0": "at the start",
           "rows": ["training compute", "peak FLOP/s", "transistors", "bandwidth of memory", "bandwidth between chips"],
           "per": ["5.3× a year", "3.0× in 2 years", "2× in 2 years", "1.6× in 2 years", "1.4× in 2 years"],
           "wall": "FLOPs per byte read: {v}× as many as at the start",
           "alt": "Five bars on a log scale with how many times each kind of growth has multiplied after the chosen "
                  "number of years: the compute of the largest trainings, peak FLOP/s, transistors per chip, the "
                  "bandwidth of memory and the bandwidth between chips."},
    "nl": {"title": "na {t} jaar", "title_1": "na 1 jaar", "title_0": "bij de start",
           "rows": ["compute van trainingen", "hoogste FLOP/s", "transistors", "bandbreedte geheugen", "bandbreedte tussen chips"],
           "per": ["5,3× per jaar", "3,0× in 2 jaar", "2× in 2 jaar", "1,6× in 2 jaar", "1,4× in 2 jaar"],
           "wall": "FLOPs per gelezen byte: {v}× zoveel als bij de start",
           "alt": "Vijf balken op een logaritmische schaal met hoeveel keer elke soort groei is vermenigvuldigd na het "
                  "gekozen aantal jaren: de compute van de grootste trainingen, de hoogste FLOP/s, transistors per chip, de "
                  "bandbreedte van het geheugen en de bandbreedte tussen chips."},
}
NOTES = {
    "en": "Training compute: Epoch AI, the ten largest trainings at each moment, 2010 to May 2024. Peak FLOP/s, memory "
          "and the links between chips: Gholami and others, over the 20 years before 2024. Transistors: Moore's two "
          "years of 1975.",
    "nl": "Compute van trainingen: Epoch AI, de tien grootste trainingen op elk moment, 2010 tot mei 2024. Hoogste FLOP/s, "
          "geheugen en de verbindingen tussen chips: Gholami en anderen, over de 20 jaar tot 2024. Transistors: de twee "
          "jaar van Moore uit 1975.",
}
DEMO_VALUES = [0, 1, 2, 4, 6, 8, 10, 12, 14]
_PER_YEAR = [EPOCH_FRONTIER, PER_TWO_YEARS["flops"] ** 0.5, PER_TWO_YEARS["moore"] ** 0.5, PER_TWO_YEARS["dram"] ** 0.5,
             PER_TWO_YEARS["links"] ** 0.5]


def times(f: float, lang: str) -> str:
    """How many times as large: 1.6 and 26 and 22,164 and 13.8 billion."""
    return big(f, lang) if f >= 1e6 else num(f, lang) if f >= 10 else dec(f, lang)


def factors(t: float) -> list[float]:
    """How many times each of the five kinds of growth has multiplied after t years."""
    return [g ** t for g in _PER_YEAR]


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    t = min(max(int(value), 0), DEMO_VALUES[-1])
    title = w["title_0"] if t == 0 else w["title_1"] if t == 1 else w["title"].format(t=t)
    parts = [_t(0, 15, title, 12.5, 600)]
    x0, x1 = 132, 246
    top = math.log10(EPOCH_FRONTIER ** DEMO_VALUES[-1])
    for k, (label, per, f) in enumerate(zip(w["rows"], w["per"], factors(t))):
        y = 28 + k * 30
        memory = k >= 3
        parts.append(_t(0, y + 11, label, 10.5, 600 if k == 0 else 400))
        parts.append(_t(0, y + 24, per, 9.5, 400, "var(--ea-ink-3)"))
        parts.append(_rect(x0, y + 3, max(2.0, (x1 - x0) * math.log10(f) / top), 13,
                           "fill:var(--ea-ink-3)" if memory else "fill:var(--ea-ink)", 1.5))
        parts.append(_t(326, y + 14, f"{times(f, lang)}×", 10.5, 600, anchor="end"))
    y = 28 + 5 * 30 + 4
    parts.append(_line(0, y, 330, y))
    wall = (PER_TWO_YEARS["flops"] / PER_TWO_YEARS["dram"]) ** (t / 2)
    parts.append(_t(0, y + 18, w["wall"].format(v=times(wall, lang)), 10.5, 600))
    return _svg(y + 26, w["alt"], parts), NOTES[lang]


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
_NAMES = {"np": np}


def _shown(v):
    """{got} and {want}: a number with four significant digits, such as `2.036` or `1.466e-06`."""
    if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
        text_ = f"{float(v):.4g}"
    else:
        text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


# exercise 1: doubling_time
def _t1(body: str) -> str:
    return "def doubling_time(start, end, years):\n" f"    return {body}\n"


_T_START = _t1("...")
_T = _t1("years / np.log2(end / start)")
_T_NO_LOG = _t1("years / (end / start)")
_T_LOG10 = _t1("years / np.log10(end / start)")
_T_LN = _t1("years / np.log(end / start)")
_T_FLIPPED = _t1("np.log2(end / start) / years")


# exercise 2: fit_doubling
def _f1(slope: str) -> str:
    return ("def fit_doubling(years, counts):\n"
            f"    slope = {slope}\n"
            "    return 1 / slope\n")


_F_START = _f1("...")
_F = _f1("np.polyfit(years, np.log2(counts), 1)[0]")
_F_NO_LOG = _f1("np.polyfit(years, counts, 1)[0]")
_F_LOG10 = _f1("np.polyfit(years, np.log10(counts), 1)[0]")
_F_SWAP = _f1("np.polyfit(np.log2(counts), years, 1)[0]")
_F_BIAS = _f1("np.polyfit(years, np.log2(counts), 1)[1]")

LAWS = Sort(
    name="laws",
    groups={
        "moore": {"en": "Moore's law", "nl": "De wet van Moore"},
        "dennard": {"en": "Dennard scaling and its end", "nl": "Dennard scaling en het eind daarvan"},
        "wall": {"en": "The memory wall", "nl": "De memory wall"},
    },
    items={
        "prediction": {"en": "**1965 and 1975.** Gordon Moore predicts twice as many parts on a chip every year, and "
                             "later every two years.",
                       "nl": "**1965 en 1975.** Gordon Moore voorspelt elk jaar twee keer zoveel onderdelen op een "
                             "chip, en later elke twee jaar."},
        "counts": {"en": "**2,300 and 80 billion.** The transistors of the Intel 4004 of 1971 and of the NVIDIA H100 "
                         "of 2022.",
                   "nl": "**2.300 en 80 miljard.** De transistors van de Intel 4004 uit 1971 en van de NVIDIA H100 uit "
                         "2022."},
        "power": {"en": "**Smaller, same power.** A smaller transistor needs less voltage, so a chip full of them uses "
                        "the same power for each square millimeter.",
                  "nl": "**Kleiner, zelfde stroom.** Een kleinere transistor heeft minder spanning nodig, dus gebruikt "
                        "een chip vol kleinere transistors evenveel stroom per vierkante millimeter."},
        "clock": {"en": "**3.8 GHz.** In November 2004, Intel brings out a Pentium 4 at 3.8 GHz and cancels the one "
                        "at 4 GHz.",
                  "nl": "**3,8 GHz.** In november 2004 brengt Intel een Pentium 4 van 3,8 GHz uit en schrapt de "
                        "versie van 4 GHz."},
        "cores": {"en": "**More cores.** Since about 2005, a new chip gets more cores rather than a higher clock speed.",
                  "nl": "**Meer cores.** Sinds ongeveer 2005 krijgt een nieuwe chip meer cores in plaats van een "
                        "hogere kloksnelheid."},
        "rates": {"en": "**3.0 against 1.6.** Every two years, peak FLOP/s grew 3.0 times, and the bandwidth of memory "
                        "1.6 times.",
                  "nl": "**3,0 tegen 1,6.** Elke twee jaar werden de hoogste FLOP/s 3,0 keer zo groot, en de bandbreedte "
                        "van het geheugen 1,6 keer."},
        "waiting": {"en": "**Waiting for data.** Many computations on a GPU spend more time waiting for numbers from "
                          "memory than computing with them.",
                    "nl": "**Wachten op data.** Veel berekeningen op een GPU zijn langer bezig met wachten op getallen "
                          "uit het geheugen dan met rekenen."},
        "bytes": {"en": "**Fewer bytes per FLOP.** Each new GPU can read fewer bytes from memory for every FLOP it "
                        "does.",
                  "nl": "**Minder bytes per FLOP.** Elke nieuwe GPU kan per FLOP minder bytes uit het geheugen lezen."},
    },
    solution={"prediction": "moore", "counts": "moore", "power": "dennard", "clock": "dennard", "cores": "dennard",
              "rates": "wall", "waiting": "wall", "bytes": "wall"},
    nudge={
        "prediction": {"en": "Who made this prediction, and what does it count?",
                       "nl": "Wie deed deze voorspelling, en wat telt de voorspelling?"},
        "counts": {"en": "Is this about how fast a chip runs, or about how many transistors it holds?",
                   "nl": "Gaat dit over hoe snel een chip draait, of over hoeveel transistors erop zitten?"},
        "power": {"en": "Which idea says what happens to the power when transistors shrink?",
                  "nl": "Welk idee zegt wat er met de stroom gebeurt als transistors krimpen?"},
        "clock": {"en": "Why would Intel stop at 3.8 GHz, when chips got faster every year before?",
                  "nl": "Waarom zou Intel stoppen bij 3,8 GHz, terwijl chips elk jaar sneller werden?"},
        "cores": {"en": "What went wrong with a higher clock speed around 2005?",
                  "nl": "Wat ging er rond 2005 mis met een hogere kloksnelheid?"},
        "rates": {"en": "Which grows faster here, the computing or the memory?",
                  "nl": "Wat groeit hier sneller, het rekenen of het geheugen?"},
        "waiting": {"en": "Is the GPU short of computing here, or short of data?",
                    "nl": "Komt de GPU hier rekenen tekort, of data?"},
        "bytes": {"en": "What happens to bytes per FLOP when FLOP/s grow faster than memory?",
                  "nl": "Wat gebeurt er met bytes per FLOP als de FLOP/s sneller groeien dan het geheugen?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight items again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht onderdelen opnieuw uit."},
        "empty": {"en": "Put each item under *Moore's law*, *Dennard scaling and its end* or *The memory wall*.",
                  "nl": "Zet elk onderdeel onder *De wet van Moore*, *Dennard scaling en het eind daarvan* of *De memory "
                        "wall*."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs somewhere else.",
                  "nl": "*{item}* hoort ergens anders."},
        "correct": {"en": "All {n} are right. Moore's law counts transistors, Dennard scaling made them faster for the "
                          "same power until about 2005, and the memory wall is computing that outgrows memory. Which "
                          "of the three no longer holds for the GPUs of today?",
                    "nl": "Alle {n} kloppen. De wet van Moore telt transistors, Dennard scaling maakte transistors tot "
                          "ongeveer 2005 sneller bij dezelfde stroom, en de memory wall is rekenen dat harder groeit dan "
                          "het geheugen. Welke van de drie geldt niet meer voor de GPU's van nu?"},
    },
)

EXERCISES = {
    "time": FunctionBox(
        name="time",
        function="doubling_time",
        start=_T_START,
        solution=_T,
        names=_NAMES,
        show=_shown,
        visible=[(2_300, 80_000_000_000, 51)],
        hidden=[(GPT3_FLOP, LLAMA_FLOP, 4), (1, 1024, 20), (5, 10, 1), (100, 800, 6), (2_300, 4_500, 3)],
        first=[_DOTS],
        picture=time_picture,
        mistakes=[
            Mistake("no-log", wrong=_T_NO_LOG,
                    en="`{call}` gives {got}: the years divided by how many times as large the end is, and it should give {want}. How many doublings are 34.8 million times as many?",
                    nl="`{call}` geeft {got}: de jaren gedeeld door hoeveel keer zo groot het eind is, en het moet {want} zijn. Hoeveel verdubbelingen zijn 34,8 miljoen keer zoveel?"),
            Mistake("log10", wrong=_T_LOG10,
                    en="`{call}` gives {got}: that is the years per factor of ten, and it should give {want}, the years per factor of two. Which log counts doublings?",
                    nl="`{call}` geeft {got}: dat zijn de jaren per factor tien, en het moet {want} zijn, de jaren per factor twee. Welke log telt verdubbelingen?"),
            Mistake("ln", wrong=_T_LN,
                    en="`{call}` gives {got}: `np.log` is the natural log, with base e, and it should give {want}. Which log has base 2?",
                    nl="`{call}` geeft {got}: `np.log` is de natuurlijke log, met grondtal e, en het moet {want} zijn. Welke log heeft grondtal 2?"),
            Mistake("flipped", wrong=_T_FLIPPED,
                    en="`{call}` gives {got}: the doublings per year, and it should give {want}, the years per doubling. Which of the two goes on top?",
                    nl="`{call}` geeft {got}: de verdubbelingen per jaar, en het moet {want} zijn, de jaren per verdubbeling. Welke van de twee staat boven de deelstreep?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the years divided by the number of doublings from `start` to `end`.",
                      "nl": "Vervang de `...` door de jaren gedeeld door het aantal verdubbelingen van `start` tot `end`."},
            "arguments": {"en": "The check calls `{call}`, with the start, the end and the years, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(start, end, years)`?",
                          "nl": "De check roept `{call}` aan, met het begin, het eind en de jaren, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(start, end, years)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your line divide `years` by the number of doublings?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Deelt je regel `years` door het aantal verdubbelingen?"},
            "almost": {"en": "Right for the 4004 and the H100. But `{call}` gives {got}, and it should give {want}. Does your function use `start`, `end` and `years`, and no fixed numbers?",
                       "nl": "Goed voor de 4004 en de H100. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `start`, `end` en `years`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does the line start with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de regel met `return`?"},
            "correct": {"en": "Right in all {n} cases. From the 4004 to the H100: 25.05 doublings in 51 years, one every 2.04 years, close to what Moore said in 1975. In the picture: which kind of growth doubles fastest?",
                        "nl": "Goed in alle {n} gevallen. Van de 4004 tot de H100: 25,05 verdubbelingen in 51 jaar, één per 2,04 jaar, dicht bij wat Moore in 1975 zei. In de afbeelding: welke soort groei verdubbelt het snelst?"},
        },
    ),
    "fit": FunctionBox(
        name="fit",
        function="fit_doubling",
        start=_F_START,
        solution=_F,
        names=_NAMES,
        show=_shown,
        visible=[(YEARS, COUNTS)],
        hidden=[(TRAIN_YEARS, TRAIN_FLOP), ([0, 1, 2], [1, 2, 4]), ([2000, 2010], [5, 160]), ([0, 3, 6, 9], [3, 6, 12, 24])],
        first=[_DOTS],
        picture=fit_picture,
        mistakes=[
            Mistake("no-log", wrong=_F_NO_LOG,
                    en="`{call}` gives {got}: a straight line through the counts themselves, and it should give {want}. On which scale do the chips lie on a straight line?",
                    nl="`{call}` geeft {got}: een rechte lijn door de aantallen zelf, en het moet {want} zijn. Op welke schaal liggen de chips op een rechte lijn?"),
            Mistake("log10", wrong=_F_LOG10,
                    en="`{call}` gives {got}: with `np.log10`, the slope counts factors of ten per year, and it should give {want}. Which log makes the slope count doublings?",
                    nl="`{call}` geeft {got}: met `np.log10` telt de helling factoren tien per jaar, en het moet {want} zijn. Welke log laat de helling verdubbelingen tellen?"),
            Mistake("swap", wrong=_F_SWAP,
                    en="`{call}` gives {got}: with the log2 first and the years second, the slope is years per doubling, and it should give {want}. Which goes first in `np.polyfit`, across or up?",
                    nl="`{call}` geeft {got}: met de log2 eerst en de jaren daarna is de helling jaren per verdubbeling, en het moet {want} zijn. Wat komt eerst in `np.polyfit`, horizontaal of verticaal?"),
            Mistake("bias", wrong=_F_BIAS,
                    en="`{call}` gives {got}: `[1]` is the bias term of the line, and it should give {want}. Which of the two numbers of `np.polyfit` is the slope?",
                    nl="`{call}` geeft {got}: `[1]` is de bias term van de lijn, en het moet {want} zijn. Welk van de twee getallen van `np.polyfit` is de helling?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the slope of the straight line through `years` and the log2 of `counts`.",
                      "nl": "Vervang de `...` door de helling van de rechte lijn door `years` en de log2 van `counts`."},
            "arguments": {"en": "The check calls `{call}`, with the years and the counts, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(years, counts)`?",
                          "nl": "De check roept `{call}` aan, met de jaren en de aantallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(years, counts)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `slope` hold the slope of the line through `years` and the log2 of `counts`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat `slope` de helling van de lijn door `years` en de log2 van `counts`?"},
            "almost": {"en": "Right for the fifteen chips. But `{call}` gives {got}, and it should give {want}. Does your function fit its line on `years` and `counts`, and no fixed numbers?",
                       "nl": "Goed voor de vijftien chips. Maar `{call}` geeft {got}, en het moet {want} zijn. Past je functie de lijn op `years` en `counts`, en niet op vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return 1 / slope`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return 1 / slope`?"},
            "correct": {"en": "Right in all {n} cases. The fifteen chips give one doubling every 2.04 years, the four trainings one every 4.9 months. In the picture: does the line of the chips miss any chip by much?",
                        "nl": "Goed in alle {n} gevallen. De vijftien chips geven een verdubbeling per 2,04 jaar, de vier trainingen een per 4,9 maanden. In de afbeelding: mist de lijn van de chips ergens een chip ver?"},
        },
    ),
    "laws": LAWS,
}

# The lesson is done when these are correct. "wall" is the choice question before the demo.
REQUIRED = ["time", "fit", "laws", "wall"]

_SORTED = dict(LAWS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("time", _T_START, "empty", None),
    ("time", "", "empty", None),
    ("time", _T, "correct", None),
    ("time", _t1("years / (np.log(end / start) / np.log(2))"), "correct", None),
    ("time", _t1("years / (np.log2(end) - np.log2(start))"), "correct", None),
    ("time", _T_NO_LOG, "wrong", "no-log"),
    ("time", _T_LOG10, "wrong", "log10"),
    ("time", _T_LN, "wrong", "ln"),
    ("time", _T_FLIPPED, "wrong", "flipped"),
    ("time", _t1("years / np.log2(end / 2300)"), "almost", None),
    ("time", _t1("year / np.log2(end / start)"), "error", None),
    ("time", _t1("years * np.log2(end / start)"), "wrong", None),
    ("time", _T.replace("def doubling_time(start, end, years):", "def doubling_time(start, end):"), "error", "arguments"),
    ("time", "def doubling_time(start, end, years):\n    years / np.log2(end / start)\n", "wrong", "no-return"),
    ("time", "def doubling_time(start, end, years):\n    print(years / np.log2(end / start))\n", "wrong", "prints"),
    ("time", "def doubling_time(start, end, years):\n    extra = ...\n    return years / np.log2(end / start)\n", "wrong",
     "dots-left"),
    ("time", "x = 1\n", "wrong", "no-function"),
    ("fit", _F_START, "empty", None),
    ("fit", "", "empty", None),
    ("fit", _F, "correct", None),
    ("fit", _f1("np.polyfit(years, np.log(counts) / np.log(2), 1)[0]"), "correct", None),
    ("fit", "def fit_doubling(years, counts):\n"
            "    slope, bias = np.polyfit(years, np.log2(counts), 1)\n"
            "    return 1 / slope\n", "correct", None),
    ("fit", _F_NO_LOG, "wrong", "no-log"),
    ("fit", _F_LOG10, "wrong", "log10"),
    ("fit", _F_SWAP, "wrong", "swap"),
    ("fit", _F_BIAS, "wrong", "bias"),
    ("fit", _f1(repr(SLOPE)), "almost", None),
    ("fit", _f1("np.polyfit(year, np.log2(counts), 1)[0]"), "error", None),
    ("fit", _f1("np.polyfit(years, np.log2(counts), 2)[0]"), "wrong", None),
    ("fit", _F.replace("def fit_doubling(years, counts):", "def fit_doubling(years):"), "error", "arguments"),
    ("fit", _F.replace("    return 1 / slope\n", ""), "wrong", "no-return"),
    ("fit", _F.replace("    return 1 / slope\n", "    print(1 / slope)\n"), "wrong", "prints"),
    ("fit", _F.replace("    return 1 / slope\n", "    extra = ...\n    return 1 / slope\n"), "wrong", "dots-left"),
    ("fit", "x = 1\n", "wrong", "no-function"),
    ("laws", {}, "empty", None),
    ("laws", _SORTED, "correct", None),
    ("laws", {"prediction": "moore", "rates": "wall"}, "wrong", "unfinished"),
] + [("laws", _SORTED | {k: ("wall" if v != "wall" else "moore")}, "wrong", f"wrong:{k}") for k, v in _SORTED.items()]
