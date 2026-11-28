# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The checks of the lesson sys-gpu-generations: the published numbers of five server GPUs from 2017 to 2024, what
each step gained, the exercises with their known mistakes, the pictures of the learner's gains and of how many GPUs a
model needs, and the demo that sets every GPU against the V100.

The numbers are the makers', read on 28 November 2026, for the SXM or OAM form of each GPU. NVIDIA: the V100 datasheet
(March 2018: 125 trillion FLOP/s of tensor performance at 16 bits, 32 GB of HBM2 at 900 GB/s, NVLink at 300 GB/s), the
A100 datasheet (June 2021: 312 trillion FLOP/s at 16 bits, 80 GB of HBM2e at 2,039 GB/s, NVLink at 600 GB/s), the H100
product page and the Hopper architecture in-depth blog of 22 March 2022 (989.4 trillion FLOP/s at 16 bits without
sparsity, 1,979 with it, 1,978.9 at 8 bits without; 80 GB of HBM3 at 3.35 TB/s; NVLink at 900 GB/s) and the pages of
HGX B200 and DGX B200 (for eight GPUs, with sparsity, and "dense is half": 36 PFLOP/s at 16 bits and 72 PFLOP/s at
4 bits without sparsity; 1,440 GB of HBM3e at 64 TB/s; NVLink at 1.8 TB/s for each GPU). AMD: the MI300X datasheet
(1,307.4 trillion FLOP/s at 16 bits without sparsity, 192 GB of HBM3 at 5.3 TB/s, seven Infinity Fabric links of
128 GB/s, of which the lesson counts the 896 GB/s together). The years are those in which each GPU came out.

Every value a learner's code gives back is shown through html.escape."""
import html
import math

import numpy as np

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-gpu-generations"
LANGS = ("en", "nl")

# ======== what the makers published ========================================================================================

ORDER = ["V100", "A100", "H100", "MI300X", "B200"]
YEARS = {"V100": 2017, "A100": 2020, "H100": 2022, "MI300X": 2023, "B200": 2024}
DESIGNS = {"V100": ("NVIDIA", "Volta"), "A100": ("NVIDIA", "Ampere"), "H100": ("NVIDIA", "Hopper"),
           "MI300X": ("AMD", "CDNA 3"), "B200": ("NVIDIA", "Blackwell")}
MEMORY = {"V100": "HBM2", "A100": "HBM2e", "H100": "HBM3", "MI300X": "HBM3", "B200": "HBM3e"}
# flops: trillions of FLOP/s for numbers of 16 bits, without sparsity; bandwidth: TB/s of the VRAM; vram: GB;
# link: GB/s from one GPU to the others
SPECS = {
    "V100": {"flops": 125, "bandwidth": 0.9, "vram": 32, "link": 300},
    "A100": {"flops": 312, "bandwidth": 2.039, "vram": 80, "link": 600},
    "H100": {"flops": 989.4, "bandwidth": 3.35, "vram": 80, "link": 900},
    "MI300X": {"flops": 1307.4, "bandwidth": 5.3, "vram": 192, "link": 896},
    "B200": {"flops": 2250, "bandwidth": 8.0, "vram": 180, "link": 1800},
}
KEYS = ["flops", "bandwidth", "vram", "link"]
H100_8_BITS = 1978.9      # trillions of FLOP/s for numbers of 8 bits, without sparsity
B200_4_BITS = 9000        # trillions of FLOP/s for numbers of 4 bits, without sparsity: 72 PFLOP/s for eight GPUs
H100_SPARSE = 1979        # trillions of FLOP/s at 16 bits as the product page prints it, with sparsity


class _Gpu(dict):
    """The numbers of one GPU, which show themselves by their name in the lesson, such as C.GPUS["V100"]."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return f'C.GPUS["{self.name}"]'


class _Count(int):
    """A number of parameters that shows itself as Python can write it, with underscores: 70_000_000_000."""

    def __repr__(self):
        return f"{int(self):_}"


GPUS = {name: _Gpu(name, SPECS[name]) for name in ORDER}
MODELS = [_Count(70_000_000_000), _Count(405_000_000_000)]   # parameters of the two models in the picture
BITS = 16


# ======== the functions of the exercises, as the course writes them =========================================================

def gains(old, new):
    out = {}
    for key in old:
        out[key] = new[key] / old[key]
    return out


def gpus_needed(params, bits, vram_gb):
    gb = params * bits / 8 / 1e9
    count = math.floor(gb / vram_gb)
    if count * vram_gb < gb:
        count = count + 1
    return count


def facts() -> dict:
    """The numbers the lesson's texts are built on."""
    v, h, b = SPECS["V100"], SPECS["H100"], SPECS["B200"]
    years = YEARS["B200"] - YEARS["V100"]
    grew = gains(v, b)
    return {
        "grew": grew,                                                             # 18, 8.9, 5.6, 6
        "h100": gains(v, h),                                                      # 7.9, 3.7, 2.5, 3
        "steps": {n: gains(SPECS[a], SPECS[n]) for a, n in (("V100", "A100"), ("A100", "H100"), ("H100", "B200"))},
        "per_byte": {n: SPECS[n]["flops"] / SPECS[n]["bandwidth"] for n in ORDER},  # 139, 153, 295, 247, 281
        "doubling": {k: years / math.log2(grew[k]) for k in KEYS},               # 1.7, 2.2, 2.8, 2.7 years
        "years": years,
        "b200_4_vs_v100": B200_4_BITS / v["flops"],                              # 72
        "model_gb": [m * BITS / 8 / 1e9 for m in MODELS],                        # 140, 810
        "needed": {m: {n: gpus_needed(m, BITS, SPECS[n]["vram"]) for n in ORDER} for m in MODELS},
        "token_read": 16e9 / (v["bandwidth"] * 1e12),                            # 17.8 ms on a V100
        "token_flops": 16e9 / (v["flops"] * 1e12),                               # 0.13 ms on a V100
        "token_b200": 16e9 / (b["bandwidth"] * 1e12),                            # 2 ms on a B200
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


def times(x: float, lang: str) -> str:
    """How many times as much, with two significant digits: ×18, ×8.9, ×1, ×0.056; whole from 100 on."""
    text_ = f"{x:,.0f}" if x >= 100 else f"{x:.2g}"
    return "×" + _local(text_, lang)


def number(x: float, lang: str) -> str:
    """A published number as the lesson writes it: 125, 2,250, 2.04, 0.9."""
    text_ = f"{x:,.0f}" if x >= 100 else f"{x:.3g}"
    return _local(text_, lang)


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


def _dot(x: float, y: float, r: float, style: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="{style}"/>'


# ======== the picture of the idea: FLOP/s and bandwidth of every GPU against the V100 ======================================

IDEA_TEXT = {
    "en": {"legend_flops": "FLOP/s at 16 bits", "legend_bandwidth": "bandwidth", "scale": "times the V100",
           "alt": "For five GPUs, two dots on a log scale against the V100 of 2017: a dark dot for the FLOP/s at 16 "
                  "bits and an open dot for the bandwidth of the VRAM. A100 2020: ×2.5 and ×2.3. H100 2022: ×7.9 and "
                  "×3.7. MI300X 2023: ×10 and ×5.9. B200 2024: ×18 and ×8.9. The gap between the dots grows."},
    "nl": {"legend_flops": "FLOP/s bij 16 bits", "legend_bandwidth": "bandbreedte", "scale": "keer de V100",
           "alt": "Voor vijf GPU's twee stippen op een logaritmische schaal ten opzichte van de V100 uit 2017: een donkere "
                  "stip voor de FLOP/s bij 16 bits en een open stip voor de bandbreedte van het VRAM. A100 2020: ×2,5 en "
                  "×2,3. H100 2022: ×7,9 en ×3,7. MI300X 2023: ×10 en ×5,9. B200 2024: ×18 en ×8,9. De afstand tussen "
                  "de stippen groeit."},
}
IDEA_TOP = 20.0   # the right end of the scale: 20 times the V100


def _idea_x(v: float) -> float:
    return 74 + (300 - 74) * math.log10(v) / math.log10(IDEA_TOP)


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    parts = [_dot(80, 10, 4, "fill:var(--ea-ink)"), _t(88, 14, w["legend_flops"], 10.5, 400, "var(--ea-ink-2)"),
             _dot(200, 10, 4, "fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.5"),
             _t(208, 14, w["legend_bandwidth"], 10.5, 400, "var(--ea-ink-2)")]
    bottom = 32 + len(ORDER) * 32
    for v in (1, 2, 5, 10, 20):
        x = _idea_x(v)
        parts.append(_line(x, 26, x, bottom - 4, "var(--ea-line)"))
        parts.append(_t(x, bottom + 9, times(v, lang), 9.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t(326, bottom + 23, w["scale"], 9.5, 400, "var(--ea-ink-3)", "end"))
    v100 = SPECS["V100"]
    for k, name in enumerate(ORDER):
        y = 40 + k * 32
        g = gains(v100, SPECS[name])
        xf, xb = _idea_x(g["flops"]), _idea_x(g["bandwidth"])
        parts.append(_t(0, y + 1, name, 11, 600))
        parts.append(_t(0, y + 13, str(YEARS[name]), 9.5, 400, "var(--ea-ink-3)"))
        parts.append(_line(xb, y, xf, y, "var(--ea-ink-2)", 2))
        parts.append(_dot(xb, y, 4, "fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.5"))
        parts.append(_dot(xf, y, 4, "fill:var(--ea-ink)"))
        if name == "V100":
            parts.append(_t(xf + 8, y + 4, times(1, lang), 10, 600))
        else:
            parts.append(_t(xb - 8, y + 4, times(g["bandwidth"], lang), 10, 400, "var(--ea-ink-2)", "end"))
            parts.append(_t(xf + 8, y + 4, times(g["flops"], lang), 10, 600))
    return _svg(bottom + 28, w["alt"], parts)


# ======== the pictures of the learner's own work ===========================================================================

YOURS_TEXT = {
    "en": {"gain": {
               "title": "Your gains, from the V100 on",
               "intro": "Your function from the V100 to each next GPU of NVIDIA, on a log scale.",
               "keys": {"flops": "FLOP/s", "bandwidth": "bandwidth", "vram": "VRAM", "link": "links"},
               "none": "Your `gains` stops with an error, takes too long, or does not give a number above 0 for each of "
                       "`\"flops\"`, `\"bandwidth\"`, `\"vram\"` and `\"link\"`, so the picture waits.",
               "right": "Your function gives the lines of the lesson: the FLOP/s grew 18 times, the bandwidth 8.9 times. "
                        "The gap between the top two lines is the memory wall.",
               "off": "Your function gives {got} for the FLOP/s from the V100 to the B200, and 2,250 trillion against "
                      "125 trillion is ×18.",
               "alt": "Four lines on a log scale from the V100 of 2017 through the A100 and the H100 to the B200 of "
                      "2024, with how many times the learner's function says the FLOP/s, the bandwidth, the VRAM and "
                      "the links grew."},
           "fit": {
               "title": "Your gpus_needed for two models",
               "intro": "Your function for two models of 16 bits on each of the five GPUs: a block for every GPU.",
               "model": "{params} parameters of 16 bits: {gb}",
               "params": {70_000_000_000: "70 billion", 405_000_000_000: "405 billion"},
               "none": "Your `gpus_needed` stops with an error, takes too long, or does not give a whole number of at "
                       "least 1, so the picture waits.",
               "right": "Your function gives the counts of the lesson: 70 billion parameters of 16 bits fit on one "
                        "MI300X or one B200, and take two H100s or five V100s.",
               "off": "Your function gives {got} for {params} parameters on the {gpu}, and {gb} on GPUs of {vram} GB "
                      "take {want}.",
               "alt": "For a model of 70 billion and one of 405 billion parameters of 16 bits, a row of blocks for each "
                      "of the V100, A100, H100, MI300X and B200: one block for every GPU the learner's function says "
                      "the model needs."}},
    "nl": {"gain": {
               "title": "Je gains, vanaf de V100",
               "intro": "Je functie van de V100 naar elke volgende GPU van NVIDIA, op een logaritmische schaal.",
               "keys": {"flops": "FLOP/s", "bandwidth": "bandbreedte", "vram": "VRAM", "link": "verbindingen"},
               "none": "Je `gains` stopt met een fout, duurt te lang of geeft niet voor elk van `\"flops\"`, "
                       "`\"bandwidth\"`, `\"vram\"` en `\"link\"` een getal boven 0, dus wacht het plaatje.",
               "right": "Je functie geeft de lijnen van de les: de FLOP/s groeiden 18 keer, de bandbreedte 8,9 keer. De "
                        "afstand tussen de bovenste twee lijnen is de memory wall.",
               "off": "Je functie geeft {got} voor de FLOP/s van de V100 naar de B200, en 2.250 biljoen tegen 125 biljoen "
                      "is ×18.",
               "alt": "Vier lijnen op een logaritmische schaal van de V100 uit 2017 via de A100 en de H100 naar de B200 uit "
                      "2024, met hoeveel keer de FLOP/s, de bandbreedte, het VRAM en de verbindingen groeiden volgens de "
                      "functie van de leerling."},
           "fit": {
               "title": "Je gpus_needed voor twee modellen",
               "intro": "Je functie voor twee modellen van 16 bits op elk van de vijf GPU's: een blokje voor elke GPU.",
               "model": "{params} parameters van 16 bits: {gb}",
               "params": {70_000_000_000: "70 miljard", 405_000_000_000: "405 miljard"},
               "none": "Je `gpus_needed` stopt met een fout, duurt te lang of geeft geen heel getal van minstens 1, dus "
                       "wacht het plaatje.",
               "right": "Je functie geeft de aantallen van de les: 70 miljard parameters van 16 bits passen op één MI300X "
                        "of één B200, en vragen twee H100's of vijf V100's.",
               "off": "Je functie geeft {got} voor {params} parameters op de {gpu}, en {gb} op GPU's van {vram} GB vraagt "
                      "{want}.",
               "alt": "Voor een model van 70 miljard en een van 405 miljard parameters van 16 bits een rij blokjes voor elk "
                      "van de V100, A100, H100, MI300X en B200: één blokje voor elke GPU die het model volgens de functie "
                      "van de leerling nodig heeft."}},
}

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture
LINE = ["V100", "A100", "H100", "B200"]   # NVIDIA's generations in the picture of the gains
_STYLES = {"flops": ("var(--ea-ink)", 2.2, ""), "bandwidth": ("var(--ea-ink-2)", 1.8, ""),
           "vram": ("var(--ea-ink-3)", 1.6, ""), "link": ("var(--ea-ink-3)", 1.6, "4 3")}
BLOCKS = 26           # the most blocks a row of the picture of gpus_needed draws


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


def _whole(got):
    """got as a whole number of at least 1, or _STOPPED when it is not such a number."""
    got = _positive(got)
    if got is _STOPPED or got < 1 or not float(got).is_integer():
        return _STOPPED
    return int(got)


def gains_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["gain"]
    fn = _fn(seen)
    got = {"V100": {k: 1.0 for k in KEYS}}
    for name in LINE[1:] if fn else []:
        out = _call(lambda name=name: fn(dict(GPUS["V100"]), dict(GPUS[name])))
        values = {k: _positive(out.get(k)) if isinstance(out, dict) else _STOPPED for k in KEYS}
        if any(v is _STOPPED for v in values.values()):
            break   # no more calls once one stops
        got[name] = values
    if len(got) < len(LINE):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    logs = [math.log10(got[n][k]) for n in LINE for k in KEYS]
    lo = max(-2.0, min(0.0, min(logs)))
    hi = min(3.0, max(math.log10(IDEA_TOP), max(logs)))
    x0, x1, y0, y1 = 34, 214, 150, 22

    def x(name: str) -> float:
        return x0 + (x1 - x0) * (YEARS[name] - YEARS["V100"]) / (YEARS["B200"] - YEARS["V100"])

    def y(v: float) -> float:
        return y0 - (y0 - y1) * (min(max(math.log10(v), lo), hi) - lo) / (hi - lo)

    parts = []
    for p in range(math.ceil(lo), math.floor(hi) + 1):
        parts.append(_line(x0, y(10 ** p), x1, y(10 ** p), "var(--ea-line)"))
        parts.append(_t(x0 - 5, y(10 ** p) + 3.5, times(10 ** p, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for name in LINE:
        parts.append(_t(x(name), y0 + 15, name, 10, 600, "var(--ea-ink-2)", "middle"))
        parts.append(_t(x(name), y0 + 27, str(YEARS[name]), 9.5, 400, "var(--ea-ink-3)", "middle"))
    for k in KEYS:
        color, width, dash = _STYLES[k]
        points = " ".join(f"{x(n):.1f},{y(got[n][k]):.1f}" for n in LINE)
        dashed = f";stroke-dasharray:{dash}" if dash else ""
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:{color};stroke-width:{width}{dashed}"/>')
        parts.extend(_dot(x(n), y(got[n][k]), 2.6, f"fill:{color}") for n in LINE)
    ends = sorted(KEYS, key=lambda k: y(got["B200"][k]))
    placed = [y(got["B200"][k]) + 3.5 for k in ends]
    for i in range(1, len(placed)):          # 12 px apart at least, from the top down
        placed[i] = max(placed[i], placed[i - 1] + 12)
    placed[-1] = min(placed[-1], y0 + 3.5)    # never below the axis, then up again where needed
    for i in range(len(placed) - 2, -1, -1):
        placed[i] = min(placed[i], placed[i + 1] - 12)
    for k, ly in zip(ends, placed):
        parts.append(_t(x1 + 8, ly, f'{w["keys"][k]} {times(got["B200"][k], lang)}', 10, 600 if k == "flops" else 400,
                        _STYLES[k][0] if k != "vram" else "var(--ea-ink-2)"))
    right = all(math.isclose(got[n][k], gains(SPECS["V100"], SPECS[n])[k], rel_tol=1e-6) for n in LINE[1:] for k in KEYS)
    said = w["right"] if right else w["off"].format(got=times(got["B200"]["flops"], lang))
    body = _note(w["intro"]) + _svg(y0 + 32, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def _count_text(c: int, lang: str) -> str:
    return _local(f"{c:,}", lang) if c < 10_000 else "> " + _local("9,999", lang)


def fit_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["fit"]
    fn = _fn(seen)
    got = {}
    for params in MODELS if fn else []:
        for name in ORDER:
            c = _whole(_call(lambda params=params, name=name: fn(int(params), BITS, SPECS[name]["vram"])))
            if c is _STOPPED:
                break   # no more calls once one stops
            got[(params, name)] = c
        if len(got) < len(ORDER) * (MODELS.index(params) + 1):
            break
    if len(got) < len(MODELS) * len(ORDER):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    parts = []
    y = 0
    for params in MODELS:
        gb = params * BITS / 8 / 1e9
        parts.append(_t(0, y + 14, w["model"].format(params=w["params"][params], gb=_local(f"{gb:,.0f}", lang) + " GB"),
                        11, 600))
        y += 22
        for name in ORDER:
            c = got[(params, name)]
            parts.append(_t(0, y + 9, name, 10.5, 600))
            parts.append(_t(90, y + 9, f'{SPECS[name]["vram"]} GB', 9.5, 400, "var(--ea-ink-3)", "end"))
            shown = c if c <= BLOCKS else BLOCKS - 6
            parts.extend(_rect(96 + i * 8, y, 6, 10, "fill:var(--ea-ink)", 1) for i in range(shown))
            if c > BLOCKS:
                parts.append(_t(96 + shown * 8 + 1, y + 9, "…", 10.5, 600, "var(--ea-ink-2)"))
            parts.append(_t(326, y + 9, _count_text(c, lang), 10.5, 600, anchor="end"))
            y += 17
        y += 8
    want = facts()["needed"]
    wrong = [(p, n) for p in MODELS for n in ORDER if got[(p, n)] != want[p][n]]
    if wrong:
        p, n = wrong[0]
        gb = p * BITS / 8 / 1e9
        said = w["off"].format(got=_count_text(got[(p, n)], lang), params=w["params"][p], gpu=n,
                               gb=_local(f"{gb:,.0f}", lang) + " GB", vram=SPECS[n]["vram"], want=want[p][n])
    else:
        said = w["right"]
    body = _note(w["intro"]) + _svg(y, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: every GPU against the V100 =============================================================================

DEMO_TEXT = {
    "en": {"title": "{name}: NVIDIA {design}, {year}", "title_amd": "{name}: AMD {design}, {year}",
           "scale": "times the V100, log scale",
           "rows": {"flops": "FLOP/s, 16 bits", "bandwidth": "bandwidth", "vram": "VRAM", "link": "links"},
           "units": {"flops": "{v} trillion", "bandwidth": "{v} TB/s", "vram": "{v} GB", "link": "{v} GB/s"},
           "per_byte": "{n} FLOPs per byte read (V100: 139)",
           "said": ["The starting point: 125 trillion FLOP/s at 16 bits and 32 GB of HBM2 at 0.9 TB/s, joined to other "
                    "GPUs by NVLink at 300 GB/s.",
                    "Three years on, all four numbers grew 2 to 2.5 times, and the FLOPs per byte barely changed: 153.",
                    "The FLOP/s jump to 7.9 times the V100, the bandwidth only to 3.7 times, and the VRAM stays at 80 GB: "
                    "295 FLOPs per byte.",
                    "AMD's MI300X has the most VRAM of the five, 192 GB, and 10 times the FLOP/s of the V100.",
                    "Two chips in one package: 18 times the FLOP/s of the V100, 8.9 times the bandwidth, and 281 FLOPs per "
                    "byte."],
           "alt": "Four bars on a log scale with how many times the chosen GPU has the FLOP/s at 16 bits, the bandwidth, "
                  "the VRAM and the links of the V100 of 2017, with the published numbers under each name."},
    "nl": {"title": "{name}: NVIDIA {design}, {year}", "title_amd": "{name}: AMD {design}, {year}",
           "scale": "keer de V100, logaritmische schaal",
           "rows": {"flops": "FLOP/s, 16 bits", "bandwidth": "bandbreedte", "vram": "VRAM", "link": "verbindingen"},
           "units": {"flops": "{v} biljoen", "bandwidth": "{v} TB/s", "vram": "{v} GB", "link": "{v} GB/s"},
           "per_byte": "{n} FLOPs per gelezen byte (V100: 139)",
           "said": ["Het beginpunt: 125 biljoen FLOP/s bij 16 bits en 32 GB HBM2 met 0,9 TB/s, met andere GPU's verbonden "
                    "via NVLink met 300 GB/s.",
                    "Drie jaar later zijn alle vier de getallen 2 tot 2,5 keer zo groot, en de FLOPs per byte veranderen "
                    "nauwelijks: 153.",
                    "De FLOP/s springen naar 7,9 keer de V100, de bandbreedte slechts naar 3,7 keer, en het VRAM blijft "
                    "80 GB: 295 FLOPs per byte.",
                    "De MI300X van AMD heeft het meeste VRAM van de vijf, 192 GB, en 10 keer de FLOP/s van de V100.",
                    "Twee chips in één behuizing: 18 keer de FLOP/s van de V100, 8,9 keer de bandbreedte, en 281 FLOPs per "
                    "byte."],
           "alt": "Vier balken op een logaritmische schaal met hoeveel keer de gekozen GPU de FLOP/s bij 16 bits, de "
                  "bandbreedte, het VRAM en de verbindingen van de V100 uit 2017 heeft, met de gepubliceerde getallen "
                  "onder elke naam."},
}
DEMO_VALUES = [YEARS[n] for n in ORDER]   # 2017, 2020, 2022, 2023, 2024


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    k = min(range(len(DEMO_VALUES)), key=lambda i: abs(DEMO_VALUES[i] - value))
    name = ORDER[k]
    maker, design = DESIGNS[name]
    title = w["title" if maker == "NVIDIA" else "title_amd"].format(name=name, design=design, year=YEARS[name])
    g = gains(SPECS["V100"], SPECS[name])
    x0, x1 = 104, 280

    def x(v: float) -> float:
        return x0 + (x1 - x0) * math.log10(v) / math.log10(IDEA_TOP)

    parts = [_t(0, 15, title, 12.5, 600), _t(326, 32, w["scale"], 9.5, 400, "var(--ea-ink-3)", "end")]
    for v in (1, 2, 5, 10, 20):
        parts.append(_line(x(v), 38, x(v), 38 + len(KEYS) * 32 - 6, "var(--ea-line)"))
    for i, key in enumerate(KEYS):
        y = 40 + i * 32
        v = SPECS[name][key]
        parts.append(_t(0, y + 10, w["rows"][key], 10.5, 600))
        parts.append(_t(0, y + 23, w["units"][key].format(v=number(v, lang)), 9.5, 400, "var(--ea-ink-3)"))
        parts.append(_rect(x0, y + 4, max(2.0, x(g[key]) - x0), 14, "fill:var(--ea-ink)", 1.5))
        parts.append(_t(326, y + 15, times(g[key], lang), 10.5, 600, anchor="end"))
    fpb = SPECS[name]["flops"] / SPECS[name]["bandwidth"]
    parts.append(_t(0, 40 + len(KEYS) * 32 + 8, w["per_byte"].format(n=f"{fpb:.0f}"), 10.5, 400, "var(--ea-ink-2)"))
    return _svg(40 + len(KEYS) * 32 + 14, w["alt"], parts), w["said"][k]


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
    """{got} and {want} of gains: a dict with its numbers at four significant digits."""
    if isinstance(v, dict) and all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
                                   for x in v.values()):
        text_ = "{" + ", ".join(f"{check._as_code(k)}: {float(x):.4g}" for k, x in v.items()) + "}"
    else:
        text_ = check._as_value(v)
    if len(text_) > 90:
        text_ = text_[:87] + "..."
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


def _number_got(got) -> bool:
    return isinstance(got, (int, float, np.integer, np.floating)) and not isinstance(got, (bool, np.bool_)) \
        and math.isfinite(float(got))


# exercise 1: gains
def _g(line: str) -> str:
    return ("def gains(old, new):\n"
            "    out = {}\n"
            "    for key in old:\n"
            f"        out[key] = {line}\n"
            "    return out\n")


_G_START = _g("...")
_G = _g("new[key] / old[key]")
_G_UPSIDE = _g("old[key] / new[key]")
_G_MINUS = _g("new[key] - old[key]")
_G_PART = _g("(new[key] - old[key]) / old[key]")

# exercise 2: gpus_needed
_F_TAIL = ("    count = math.floor(gb / vram_gb)\n"
           "    if count * vram_gb < gb:\n"
           "        count = count + 1\n"
           "    return count\n")


def _f(line: str) -> str:
    return f"import math\n\n\ndef gpus_needed(params, bits, vram_gb):\n    gb = {line}\n" + _F_TAIL


_F_START = _f("...")
_F = _f("params * bits / 8 / 1e9")
_F_BITS = _f("params * bits / 1e9")
_F_ONE = _f("params / 1e9")
_F_GIGA = _f("params * bits / 8")

HELPS = Sort(
    name="helps",
    groups={
        "flops": {"en": "More FLOP/s", "nl": "Meer FLOP/s"},
        "bandwidth": {"en": "More bandwidth", "nl": "Meer bandbreedte"},
        "vram": {"en": "More VRAM", "nl": "Meer VRAM"},
        "link": {"en": "Faster links", "nl": "Snellere verbindingen"},
    },
    items={
        "batches": {"en": "**Training on large batches.** Every parameter that is read from memory is used for hundreds "
                          "of examples at once.",
                    "nl": "**Trainen met grote batches.** Elke parameter die uit het geheugen komt, wordt voor honderden "
                          "voorbeelden tegelijk gebruikt."},
        "users": {"en": "**256 users at once.** The GPU writes the next token for all of them together, so every "
                        "parameter it reads serves 256 users.",
                  "nl": "**256 gebruikers tegelijk.** De GPU schrijft het volgende token voor alle 256 samen, dus elke "
                        "gelezen parameter dient 256 gebruikers."},
        "one": {"en": "**One user, one token at a time.** For every token, each parameter is read once and used for one "
                      "user.",
                "nl": "**Eén gebruiker, één token per keer.** Voor elk token wordt elke parameter één keer gelezen en voor "
                      "één gebruiker gebruikt."},
        "sum": {"en": "**The sum of a huge array.** Every number is read once, for a single addition.",
                "nl": "**De optelling van een enorme array.** Elk getal wordt één keer gelezen, voor één optelling."},
        "large": {"en": "**A model of 140 GB on one GPU.** Without cutting it into parts over two GPUs.",
                  "nl": "**Een model van 140 GB op één GPU.** Zonder het model in delen over twee GPU's te verdelen."},
        "train": {"en": "**Training a larger model on a single GPU.** Its parameters, and the extra numbers of the "
                        "training, must all fit at once.",
                  "nl": "**Een groter model trainen op één GPU.** De parameters en de extra getallen van de training moeten "
                        "er allemaal tegelijk in passen."},
        "many": {"en": "**Training on 1,000 GPUs.** After every step, the GPUs swap what they computed before the next "
                       "step can start.",
                 "nl": "**Trainen op 1.000 GPU's.** Na elke stap wisselen de GPU's de uitkomsten uit, en pas daarna begint "
                       "de volgende stap."},
        "split": {"en": "**A model split over eight GPUs.** At every layer, the GPUs pass their numbers on to each other.",
                  "nl": "**Een model verdeeld over acht GPU's.** Bij elke laag geven de GPU's de getallen aan elkaar door."},
    },
    solution={"batches": "flops", "users": "flops", "one": "bandwidth", "sum": "bandwidth", "large": "vram",
              "train": "vram", "many": "link", "split": "link"},
    nudge={
        "batches": {"en": "Is each number that is read used once, or for many examples?",
                    "nl": "Wordt elk gelezen getal één keer gebruikt, of voor veel voorbeelden?"},
        "users": {"en": "With every parameter used 256 times, what does the GPU spend its time on: reading, or computing?",
                  "nl": "Met elke parameter 256 keer gebruikt: waar gaat de tijd van de GPU naartoe, naar lezen of naar "
                        "rekenen?"},
        "one": {"en": "Each parameter is read for a single use: what does the GPU wait for?",
                "nl": "Elke parameter wordt gelezen voor één keer gebruik: waarop wacht de GPU?"},
        "sum": {"en": "One addition for each number read: does the GPU wait for its numbers, or for its cores?",
                "nl": "Eén optelling per gelezen getal: wacht de GPU op de getallen, of op de cores?"},
        "large": {"en": "What has to be large enough for 140 GB to fit on one GPU?",
                  "nl": "Wat moet groot genoeg zijn om 140 GB op één GPU te laten passen?"},
        "train": {"en": "Is this about speed, or about room?",
                  "nl": "Gaat dit over snelheid, of over ruimte?"},
        "many": {"en": "What do the GPUs wait for between two steps?",
                 "nl": "Waarop wachten de GPU's tussen twee stappen?"},
        "split": {"en": "What carries the numbers from one GPU to the next?",
                  "nl": "Wat brengt de getallen van de ene GPU naar de volgende?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight jobs again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht taken opnieuw uit."},
        "empty": {"en": "Put each job under *More FLOP/s*, *More bandwidth*, *More VRAM* or *Faster links*.",
                  "nl": "Zet elke taak onder *Meer FLOP/s*, *Meer bandbreedte*, *Meer VRAM* of *Snellere verbindingen*."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs somewhere else.",
                  "nl": "*{item}* hoort ergens anders."},
        "correct": {"en": "All {n} are right. Many uses for every byte read: FLOP/s. One use for every byte: bandwidth. "
                          "Room for the numbers: VRAM. Many GPUs together: links. Which of the four grew fastest from the "
                          "V100 to the B200?",
                    "nl": "Alle {n} kloppen. Veel gebruik voor elke gelezen byte: FLOP/s. Eén keer gebruik per byte: "
                          "bandbreedte. Ruimte voor de getallen: VRAM. Veel GPU's samen: verbindingen. Welke van de vier "
                          "groeide het snelst van de V100 naar de B200?"},
    },
)

EXERCISES = {
    "gain": FunctionBox(
        name="gain",
        function="gains",
        start=_G_START,
        solution=_G,
        show=_shown_dict,
        visible=[(GPUS["V100"], GPUS["B200"])],
        hidden=[(GPUS["A100"], GPUS["H100"]), (GPUS["H100"], GPUS["MI300X"]), ({"a": 2, "b": 10}, {"a": 6, "b": 5}),
                ({"x": 4.0}, {"x": 1.0, "y": 3.0}), ({}, {"z": 1})],
        first=[_DOTS],
        picture=gains_picture,
        mistakes=[
            Mistake("upside-down", wrong=_G_UPSIDE,
                    en="`{call}` gives {got}: the old numbers divided by the new ones, and it should give {want}. How many times as large is the new number as the old one: which of the two goes on top?",
                    nl="`{call}` geeft {got}: de oude getallen gedeeld door de nieuwe, en het moet {want} zijn. Hoeveel keer zo groot is het nieuwe getal als het oude: welke van de twee staat boven de deelstreep?"),
            Mistake("minus", wrong=_G_MINUS,
                    en="`{call}` gives {got}: how much each number grew, in its own unit, and it should give {want}. How many times as large is the new number as the old one?",
                    nl="`{call}` geeft {got}: hoeveel elk getal groeide, in de eigen eenheid, en het moet {want} zijn. Hoeveel keer zo groot is het nieuwe getal als het oude?"),
            Mistake("part", wrong=_G_PART,
                    en="`{call}` gives {got}: one less than it should for each number, {want}. The growth on top of the old number is not the factor: how many times does the old number fit in the new one?",
                    nl="`{call}` geeft {got}: voor elk getal één minder dan het moet zijn, {want}. De groei bovenop het oude getal is niet de factor: hoeveel keer past het oude getal in het nieuwe?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` so that every round of the loop puts the number of `new` divided by the number of `old` for `key` in `out[key]`.",
                      "nl": "Vervang de `...` zodat elke ronde van de loop het getal van `new` gedeeld door het getal van `old` voor `key` in `out[key]` zet."},
            "arguments": {"en": "The check calls `{call}`, with the numbers of two GPUs, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(old, new)`?",
                          "nl": "De check roept `{call}` aan, met de getallen van twee GPU's, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(old, new)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every round put the number of `new` divided by the number of `old` in `out[key]`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zet elke ronde het getal van `new` gedeeld door het getal van `old` in `out[key]`?"},
            "almost": {"en": "Right for the V100 and the B200. But `{call}` gives {got}, and it should give {want}. Does your function use `old` and `new`, and no fixed numbers?",
                       "nl": "Goed voor de V100 en de B200. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `old` en `new`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return out`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return out`?"},
            "correct": {"en": "Right in all {n} cases. From the V100 to the B200: the FLOP/s ×18, the bandwidth ×8.9, the VRAM ×5.6 and the links ×6. In the picture: which line climbs fastest?",
                        "nl": "Goed in alle {n} gevallen. Van de V100 naar de B200: de FLOP/s ×18, de bandbreedte ×8,9, het VRAM ×5,6 en de verbindingen ×6. In het plaatje: welke lijn stijgt het snelst?"},
        },
    ),
    "fit": FunctionBox(
        name="fit",
        function="gpus_needed",
        start=_F_START,
        solution=_F,
        visible=[(MODELS[0], BITS, 80)],
        hidden=[(MODELS[0], BITS, 32), (MODELS[1], BITS, 80), (MODELS[1], BITS, 192), (_Count(8_000_000_000), BITS, 80),
                (MODELS[0], 4, 32), (_Count(40_000_000_000), BITS, 80), (MODELS[0], 8, 180)],
        first=[_DOTS],
        picture=fit_picture,
        mistakes=[
            Mistake("bits-as-bytes", wrong=_F_BITS,
                    en="`{call}` gives {got}, and it should give {want}: your gigabytes are 8 times too many. `bits` counts bits, not bytes. How many bits make a byte?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je gigabytes zijn 8 keer te veel. `bits` telt bits, geen bytes. Hoeveel bits gaan er in een byte?"),
            Mistake("one-byte", wrong=_F_ONE,
                    en="`{call}` gives {got}, and it should give {want}: as if every parameter took 1 byte. How many bytes does a parameter of `bits` bits take?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alsof elke parameter 1 byte inneemt. Hoeveel bytes neemt een parameter van `bits` bits in?"),
            Mistake("too-many", matches=lambda got, text_: _number_got(got) and float(got) >= 10_000,
                    en="`{call}` gives {got}: more GPUs than the largest computers hold. Does your function divide by `1e9`, to turn bytes into gigabytes?",
                    nl="`{call}` geeft {got}: meer GPU's dan de grootste computers hebben. Deelt je functie door `1e9`, om bytes in gigabytes om te zetten?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` so that `gb` is the gigabytes of all parameters: `params` times the bytes of one parameter, divided by `1e9`.",
                      "nl": "Vervang de `...` zodat `gb` de gigabytes van alle parameters is: `params` keer de bytes van één parameter, gedeeld door `1e9`."},
            "arguments": {"en": "The check calls `{call}`, with the parameters, the bits of each and the VRAM of one GPU, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(params, bits, vram_gb)`?",
                          "nl": "De check roept `{call}` aan, met de parameters, de bits van elke parameter en het VRAM van één GPU, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(params, bits, vram_gb)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `gb` the parameters times the bytes of each, divided by `1e9`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `gb` de parameters keer de bytes van elke parameter, gedeeld door `1e9`?"},
            "almost": {"en": "Right for 70 billion parameters on GPUs of 80 GB. But `{call}` gives {got}, and it should give {want}. Does `gb` use `params` and `bits`, and no fixed numbers?",
                       "nl": "Goed voor 70 miljard parameters op GPU's van 80 GB. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt `gb` de namen `params` en `bits`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return count`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return count`?"},
            "correct": {"en": "Right in all {n} cases. 70 billion parameters of 16 bits are 140 GB: two H100s, or one MI300X or B200. In the picture: how many V100s does the model of 405 billion parameters take?",
                        "nl": "Goed in alle {n} gevallen. 70 miljard parameters van 16 bits zijn 140 GB: twee H100's, of één MI300X of B200. In het plaatje: hoeveel V100's vraagt het model van 405 miljard parameters?"},
        },
    ),
    "helps": HELPS,
}

# The lesson is done when these are correct. "token" is the choice question after the demo.
REQUIRED = ["gain", "fit", "helps", "token"]

_SORTED = dict(HELPS.solution)
_OTHER = {"flops": "bandwidth", "bandwidth": "vram", "vram": "link", "link": "flops"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gain", _G_START, "empty", None),
    ("gain", "", "empty", None),
    ("gain", _G, "correct", None),
    ("gain", "def gains(old, new):\n"
             "    out = {}\n"
             "    for key, value in old.items():\n"
             "        out[key] = new[key] / value\n"
             "    return out\n", "correct", None),
    ("gain", "def gains(old, new):\n"
             "    return {key: new[key] / old[key] for key in old}\n", "correct", None),
    ("gain", _G_UPSIDE, "wrong", "upside-down"),
    ("gain", _G_MINUS, "wrong", "minus"),
    ("gain", _G_PART, "wrong", "part"),
    ("gain", "def gains(old, new):\n"
             "    return {'flops': 18.0, 'bandwidth': 8.88888888888889, 'vram': 5.625, 'link': 6.0}\n", "almost", None),
    ("gain", _g("new[keys] / old[key]"), "error", None),
    ("gain", _g("new[key] / old[key] * 100"), "wrong", None),
    ("gain", _G.replace("def gains(old, new):", "def gains(old):"), "error", "arguments"),
    ("gain", _G.replace("    return out\n", ""), "wrong", "no-return"),
    ("gain", _G.replace("    return out\n", "    print(out)\n"), "wrong", "prints"),
    ("gain", _G.replace("    return out\n", "    extra = ...\n    return out\n"), "wrong", "dots-left"),
    ("gain", "x = 1\n", "wrong", "no-function"),
    ("fit", _F_START, "empty", None),
    ("fit", "", "empty", None),
    ("fit", _F, "correct", None),
    ("fit", _f("params * bits / 8e9"), "correct", None),
    ("fit", _f("params / 1e9 * bits / 8"), "correct", None),
    ("fit", _f("params * (bits / 8) / 1e9"), "correct", None),
    ("fit", "def gpus_needed(params, bits, vram_gb):\n"
            "    gb = params * bits / 8 / 1e9\n"
            "    count = 1\n"
            "    while count * vram_gb < gb:\n"
            "        count = count + 1\n"
            "    return count\n", "correct", None),
    ("fit", _F_BITS, "wrong", "bits-as-bytes"),
    ("fit", _F_ONE, "wrong", "one-byte"),
    ("fit", _F_GIGA, "wrong", "too-many"),
    ("fit", _f("140"), "almost", None),
    ("fit", _f("param * bits / 8 / 1e9"), "error", None),
    ("fit", _f("params * bits * 8 / 1e9"), "wrong", None),
    ("fit", _F.replace("def gpus_needed(params, bits, vram_gb):", "def gpus_needed(params, vram_gb):"), "error",
     "arguments"),
    ("fit", _F.replace("    return count\n", ""), "wrong", "no-return"),
    ("fit", _F.replace("    return count\n", "    print(count)\n"), "wrong", "prints"),
    ("fit", _F.replace("    return count\n", "    extra = ...\n    return count\n"), "wrong", "dots-left"),
    ("fit", "x = 1\n", "wrong", "no-function"),
    ("helps", {}, "empty", None),
    ("helps", _SORTED, "correct", None),
    ("helps", {"batches": "flops", "one": "bandwidth"}, "wrong", "unfinished"),
] + [("helps", _SORTED | {k: _OTHER[v]}, "wrong", f"wrong:{k}") for k, v in _SORTED.items()]
