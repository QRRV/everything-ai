# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-why-training-costs-millions: Why training costs millions. A GPU hour is one GPU used for
one hour; the GPU hours of a training times the price of one GPU hour give its training cost, and the GPU hours divided
by the number of GPUs give the time it takes. The learner writes the line of `cost` that counts the GPU hours of a run,
for the five runs of a made-up project, and the one line of `days`, which needs parentheses. A sort asks what is in the
cost DeepSeek reported for V3, and a choice asks which of two labs pays more. The demo shows the GPU hours that Meta and
DeepSeek published for seven models.

The numbers are the ones the makers published, read on 28 November 2026: Touvron et al. (2023), Llama 2, table 2
(pretraining on A100-80GB); the model card of Llama 3.1 (2024, training on H100-80GB); DeepSeek-AI (2024), DeepSeek-V3
Technical Report, table 1 (H800, at an assumed rental price of 2 dollars per GPU hour, without prior research and
ablation experiments); and the Llama team at Meta (2024), The Llama 3 Herd of Models, sections 3.3.1 and 3.3.4 (up to 16K
H100 GPUs; 466 interruptions in a 54-day snapshot). The runs of C.RUNS are made up. Every value a learner's code gives
back is shown through html.escape."""
import html
import math

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-why-training-costs-millions"
LANGS = ("en", "nl")

# ======== what the makers published ========================================================================================

# from the fewest GPU hours to the most: the order of the bars of the demo
MODELS = [
    {"name": "Llama 2 7B", "maker": "Meta", "year": 2023, "gpu": "A100", "hours": 184_320, "source": "llama2"},
    {"name": "Llama 2 13B", "maker": "Meta", "year": 2023, "gpu": "A100", "hours": 368_640, "source": "llama2"},
    {"name": "Llama 3.1 8B", "maker": "Meta", "year": 2024, "gpu": "H100", "hours": 1_460_000, "source": "llama31"},
    {"name": "Llama 2 70B", "maker": "Meta", "year": 2023, "gpu": "A100", "hours": 1_720_320, "source": "llama2"},
    {"name": "DeepSeek-V3", "maker": "DeepSeek", "year": 2024, "gpu": "H800", "hours": 2_788_000, "source": "deepseek"},
    {"name": "Llama 3.1 70B", "maker": "Meta", "year": 2024, "gpu": "H100", "hours": 7_000_000, "source": "llama31"},
    {"name": "Llama 3.1 405B", "maker": "Meta", "year": 2024, "gpu": "H100", "hours": 30_840_000, "source": "llama31"},
]
DEEPSEEK = {"pretraining": 2_664_000, "longer texts": 119_000, "post-training": 5_000}   # H800 GPU hours, table 1
PRICE = 2.0              # dollars per H800 GPU hour: the rental price DeepSeek assumed
CLUSTER = 2048           # the H800 GPUs of DeepSeek's cluster
PER_TRILLION = 180_000   # H800 GPU hours for each trillion tokens of pretraining
LLAMA3_GPUS = 16_384     # "up to 16K H100 GPUs" for Llama 3 405B
YEAR_HOURS = 24 * 365.25


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [dict(v) if isinstance(v, dict) else v for v in self]


# the five runs of a made-up project: a small test, two trial runs, a run that stopped, and the final run
RUNS = _Named("C.RUNS", [{"gpus": 8, "days": 2}, {"gpus": 64, "days": 5}, {"gpus": 64, "days": 5},
                         {"gpus": 512, "days": 9}, {"gpus": 512, "days": 30}])
SAME_HOURS = DEEPSEEK["pretraining"]   # the GPU hours of the picture under days
CLUSTERS = [512, CLUSTER, 8 * CLUSTER]


def cost(runs, price):
    total = 0
    for run in runs:
        hours = run["gpus"] * run["days"] * 24
        total = total + hours * price
    return total


def days(gpu_hours, gpus):
    return gpu_hours / (gpus * 24)


def facts() -> dict:
    """The numbers the texts give."""
    hours = [r["gpus"] * r["days"] * 24 for r in RUNS]
    deepseek = sum(DEEPSEEK.values())
    big = MODELS[-1]["hours"]
    return {
        "run_hours": hours, "total_hours": sum(hours), "total": cost(list(RUNS), PRICE),
        "final": hours[-1] * PRICE, "final_share": hours[-1] / sum(hours),
        "deepseek_hours": deepseek, "deepseek_cost": deepseek * PRICE,
        "pre_share": DEEPSEEK["pretraining"] / deepseek,
        "pre_days": days(DEEPSEEK["pretraining"], CLUSTER), "trillion_days": days(PER_TRILLION, CLUSTER),
        "deepseek_days": days(deepseek, CLUSTER),
        "llama_years": big / YEAR_HOURS, "llama_days": days(big, LLAMA3_GPUS), "llama_cost": big * PRICE,
        "ratio": big / deepseek,
        "same": [days(SAME_HOURS, g) for g in CLUSTERS],
    }


# ======== words and numbers ================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _dutch(text_: str) -> str:
    """1,234.5 as Dutch writes it: 1.234,5."""
    return text_.replace(",", "_").replace(".", ",").replace("_", ".")


def num(x: float, lang: str) -> str:
    """A whole number with its thousands marked: 2,788,000; in Dutch 2.788.000."""
    out = f"{round(x):,}"
    return _dutch(out) if lang == "nl" else out


def dec(x: float, lang: str, digits: int = 1) -> str:
    """A number with a fixed number of decimals: 54.2; in Dutch 54,2."""
    out = f"{x:,.{digits}f}"
    return _dutch(out) if lang == "nl" else out


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


def _arrow(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-ink-3)") -> list:
    """A line with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    right = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return [_line(x1, y1, x2, y2, stroke, 1.2),
            f'<polygon points="{x2:.1f},{y2:.1f} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}" '
            f'style="fill:{stroke}"/>']


_BOX = "fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"


def _box(x: float, y: float, w: float, top: str, under: str, weight: int = 600) -> list:
    """A box with a number on top and what it counts under it."""
    return [_rect(x, y, w, 40, _BOX), _t(x + w / 2, y + 17, top, 12.5, weight, anchor="middle"),
            _t(x + w / 2, y + 32, under, 10.5, 400, "var(--ea-ink-2)", "middle")]


# ======== the picture of the idea: the bill of DeepSeek-V3 =================================================================

IDEA_TEXT = {
    "en": {"parts": ["pretraining", "longer texts", "post-training"],
           "hours": "H800 GPU hours", "price": "2 dollars", "per": "per GPU hour", "dollars": "dollars",
           "out": "not in it", "left": "earlier research · trial runs · the people · buying the GPUs",
           "alt": "The bill of DeepSeek-V3 as DeepSeek reported it. Pretraining took 2,664,000 H800 GPU hours, longer "
                  "texts 119,000 and post-training 5,000: 2,788,000 GPU hours in all. Times 2 dollars per GPU hour, "
                  "that is 5,576,000 dollars. Not in it: earlier research, trial runs, the people, and buying the GPUs."},
    "nl": {"parts": ["pretraining", "langere teksten", "post-training"],
           "hours": "GPU-uren H800", "price": "2 dollar", "per": "per GPU-uur", "dollars": "dollar",
           "out": "er niet in", "left": "eerder onderzoek · proefruns · de mensen · de GPU's kopen",
           "alt": "De rekening van DeepSeek-V3 volgens DeepSeek zelf. De pretraining kostte 2.664.000 GPU-uren op "
                  "de H800, langere teksten 119.000 en de post-training 5.000: samen 2.788.000 GPU-uren. Keer 2 dollar "
                  "per GPU-uur is dat 5.576.000 dollar. Er niet in: eerder onderzoek, proefruns, de mensen, en de GPU's "
                  "kopen."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    f = facts()
    parts = []
    for k, (part, hours) in enumerate(zip(w["parts"], DEEPSEEK.values())):
        x = k * 116
        parts += _box(x, 2, 98, num(hours, lang), part)
        if k < 2:
            parts.append(_t(x + 107, 27, "+", 14, 400, "var(--ea-ink-2)", "middle"))
    parts += _arrow(165, 44, 165, 60)
    parts += _box(0, 62, 98, num(f["deepseek_hours"], lang), w["hours"])
    parts.append(_t(107, 87, "×", 14, 400, "var(--ea-ink-2)", "middle"))
    parts += _box(116, 62, 98, w["price"], w["per"])
    parts.append(_t(223, 87, "=", 14, 400, "var(--ea-ink-2)", "middle"))
    parts += [_rect(232, 62, 98, 40, "fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.8"),
              _t(281, 79, num(f["deepseek_cost"], lang), 12.5, 700, anchor="middle"),
              _t(281, 94, w["dollars"], 10.5, 400, "var(--ea-ink-2)", "middle")]
    parts += [_rect(0, 116, 330, 40, "fill:var(--ea-page);stroke:var(--ea-line);stroke-width:1"),
              _t(165, 132, w["out"], 11, 600, "var(--ea-ink-2)", "middle"),
              _t(165, 148, w["left"], 10.5, 400, "var(--ea-ink-2)", "middle")]
    return _svg(160, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"cost": {
               "title": "Your cost of the five runs",
               "intro": "Each run of `C.RUNS` on its own, as `cost([run], 1.0)` gives it: at 1 dollar per GPU hour, the "
                        "cost is the GPU hours. Then all five at 2 dollars, as `cost(C.RUNS, 2.0)`.",
               "runs": ["small test", "trial run A", "trial run B", "stopped run", "final run"],
               "head": "GPU hours of each run", "total": "your cost: {v} dollars",
               "none": "Your `cost` stops with an error, takes too long, or gives something other than a number of 0 "
                       "or more, so the picture waits.",
               "right": "With these runs, your function gives 989,952 dollars for 494,976 GPU hours. The final run is "
                        "737,280 dollars of it, three quarters: the number a report of the model would give.",
               "off": "Your function gives {got} dollars, and the five runs make 494,976 GPU hours: 989,952 dollars at "
                      "2 dollars per GPU hour.",
               "alt": "Five bars, one per run of the made-up project, with the GPU hours the learner's function gives "
                      "for each run; the final run is the longest bar."},
           "days": {
               "title": "Your days on three clusters",
               "intro": "The 2,664,000 GPU hours of DeepSeek-V3's pretraining on 512, 2,048 and 16,384 GPUs, as your "
                        "`days` gives them. The line marks two months.",
               "gpus": "{n} GPUs", "days": "{v} days", "months": "two months",
               "none": "Your `days` stops with an error, takes too long, or gives something other than a number of 0 or "
                       "more, so the picture waits.",
               "right": "The same GPU hours each time: on 2,048 GPUs 54.2 days, DeepSeek's \"less than two months\", "
                        "and on eight times as many GPUs an eighth of the days.",
               "off": "Your function gives {got} days on 2,048 GPUs, and 2,664,000 GPU hours on 2,048 GPUs are 1,301 "
                      "hours of the whole cluster: 54.2 days.",
               "alt": "Three bars for 512, 2,048 and 16,384 GPUs, each as long as the days the learner's function gives "
                      "for 2,664,000 GPU hours, with a line at two months."}},
    "nl": {"cost": {
               "title": "Je cost van de vijf runs",
               "intro": "Elke run van `C.RUNS` apart, zoals `cost([run], 1.0)` de kosten geeft: bij 1 dollar per GPU-uur "
                        "zijn de kosten gelijk aan de GPU-uren. Dan alle vijf tegen 2 dollar, als `cost(C.RUNS, 2.0)`.",
               "runs": ["kleine test", "proefrun A", "proefrun B", "gestopte run", "laatste run"],
               "head": "GPU-uren per run", "total": "je cost: {v} dollar",
               "none": "Je `cost` stopt met een fout, duurt te lang, of geeft iets anders dan een getal van 0 of meer, "
                       "dus wacht de afbeelding nog.",
               "right": "Met deze runs geeft je functie 989.952 dollar voor 494.976 GPU-uren. De laatste run is daarvan "
                        "737.280 dollar, driekwart: het getal dat een verslag van het model zou geven.",
               "off": "Je functie geeft {got} dollar, en de vijf runs zijn samen 494.976 GPU-uren: 989.952 dollar bij 2 "
                      "dollar per GPU-uur.",
               "alt": "Vijf balken, één per run van het verzonnen project, met de GPU-uren die de functie van de leerling "
                      "per run geeft; de laatste run is de langste balk."},
           "days": {
               "title": "Je days op drie clusters",
               "intro": "De 2.664.000 GPU-uren van de pretraining van DeepSeek-V3 op 512, 2.048 en 16.384 GPU's, "
                        "zoals je `days` de dagen geeft. De lijn staat bij twee maanden.",
               "gpus": "{n} GPU's", "days": "{v} dagen", "months": "twee maanden",
               "none": "Je `days` stopt met een fout, duurt te lang, of geeft iets anders dan een getal van 0 of meer, "
                       "dus wacht de afbeelding nog.",
               "right": "Elke keer dezelfde GPU-uren: op 2.048 GPU's 54,2 dagen, de \"minder dan twee maanden\" van "
                        "DeepSeek, en op acht keer zoveel GPU's een achtste van de dagen.",
               "off": "Je functie geeft {got} dagen op 2.048 GPU's, en 2.664.000 GPU-uren op 2.048 GPU's zijn 1.301 uur "
                      "van het hele cluster: 54,2 dagen.",
               "alt": "Drie balken voor 512, 2.048 en 16.384 GPU's, elk zo lang als de dagen die de functie van de "
                      "leerling geeft voor 2.664.000 GPU-uren, met een lijn bij twee maanden."}},
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


def _number(got):
    """got as a float of 0 or more, or _STOPPED when it is not such a number."""
    if got is _STOPPED or isinstance(got, bool) or not isinstance(got, (int, float)):
        return _STOPPED
    got = float(got)
    return got if math.isfinite(got) and got >= 0 else _STOPPED


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6)


def cost_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["cost"]
    fn = _fn(seen)
    per, total = [], _STOPPED
    for r in RUNS if fn else []:   # one run at a time, and no more calls once one stops
        per.append(_number(_call(lambda r=r: fn([dict(r)], 1.0))))
        if per[-1] is _STOPPED:
            break
    else:
        if fn:
            total = _number(_call(lambda: fn([dict(r) for r in RUNS], PRICE)))
    if total is _STOPPED or any(p is _STOPPED for p in per):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    f = facts()
    top = max(max(per), max(f["run_hours"])) or 1
    x0, x1 = 88, 250
    parts = [_t(0, 12, w["head"], 11, 600, "var(--ea-ink-2)")]
    for k, (label, p) in enumerate(zip(w["runs"], per)):
        y = 22 + k * 20
        final = k == len(per) - 1
        parts.append(_t(0, y + 11, label, 11, 600 if final else 400))
        parts.append(_rect(x0, y + 2, (x1 - x0) * p / top, 12, "fill:var(--ea-ink)" if final else "fill:var(--ea-ink-3)",
                           1.5))
        parts.append(_t(326, y + 11, num(p, lang), 11, 400, "var(--ea-ink-2)", "end"))
    y = 22 + len(per) * 20
    parts.append(_line(0, y + 2, 330, y + 2))
    parts.append(_t(0, y + 20, w["total"].format(v=num(total, lang)), 12, 600))
    right = _close(total, f["total"]) and all(_close(p, h) for p, h in zip(per, f["run_hours"]))
    said = w["right"] if right else w["off"].format(got=num(total, lang))
    body = _note(w["intro"]) + _svg(y + 28, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def days_text(d: float, lang: str) -> str:
    """Days with a tenth, or whole days once a learner's run takes years."""
    return dec(d, lang) if d < 1000 else num(d, lang)


def days_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["days"]
    fn = _fn(seen)
    got = []
    for g in CLUSTERS if fn else []:   # no more calls once one stops
        got.append(_number(_call(lambda g=g: fn(SAME_HOURS, g))))
        if got[-1] is _STOPPED:
            break
    if len(got) < len(CLUSTERS) or any(d is _STOPPED for d in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    f = facts()
    top = max(240.0, max(got) * 1.05)
    x0, x1 = 84, 236
    months = x0 + (x1 - x0) * 61 / top
    parts = []
    for k, (g, d) in enumerate(zip(CLUSTERS, got)):
        y = 18 + k * 24
        parts.append(_t(0, y + 12, w["gpus"].format(n=num(g, lang)), 11, 400))
        parts.append(_rect(x0, y + 2, (x1 - x0) * d / top, 14, "fill:var(--ea-ink)", 1.5))
        parts.append(_t(326, y + 12, w["days"].format(v=days_text(d, lang)), 11, 600 if g == CLUSTER else 400,
                        "var(--ea-ink)" if g == CLUSTER else "var(--ea-ink-2)", "end"))
    bottom = 18 + len(CLUSTERS) * 24
    parts.append(_line(months, 12, months, bottom, "var(--ea-ink-3)", 1))
    parts.append(_t(months, 9, w["months"], 10, 400, "var(--ea-ink-2)", "middle"))
    right = all(_close(d, s) for d, s in zip(got, f["same"]))
    said = w["right"] if right else w["off"].format(got=days_text(got[1], lang))
    body = _note(w["intro"]) + _svg(bottom + 6, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: seven trainings ========================================================================================

DEMO_TEXT = {
    "en": {"title": "{name} ({maker}, {year})", "sub": "{hours} GPU hours on the {gpu}",
           "cost": "at 2 dollars per GPU hour", "cost_v": "{v} dollars",
           "one": "on one GPU", "one_v": "{v} years",
           "many": "on 2,048 GPUs at once", "many_v": "{v} days",
           "alt": "Seven bars with the GPU hours the makers published, from Llama 2 7B to Llama 3.1 405B, with the "
                  "chosen model marked; under them its cost at 2 dollars per GPU hour, the years one GPU would take, "
                  "and the days 2,048 GPUs would take."},
    "nl": {"title": "{name} ({maker}, {year})", "sub": "{hours} GPU-uren op de {gpu}",
           "cost": "bij 2 dollar per GPU-uur", "cost_v": "{v} dollar",
           "one": "op één GPU", "one_v": "{v} jaar",
           "many": "op 2.048 GPU's tegelijk", "many_v": "{v} dagen",
           "alt": "Zeven balken met de GPU-uren die de makers publiceerden, van Llama 2 7B tot Llama 3.1 405B, met een "
                  "markering bij het gekozen model; daaronder de kosten bij 2 dollar per GPU-uur, de jaren die één GPU "
                  "erover zou doen, en de dagen voor 2.048 GPU's."},
}

NOTES = {
    "llama2": {"en": "Meta's paper on Llama 2 (2023), table 2: the GPU hours of pretraining, on A100 GPUs. Meta "
                     "trained a 34B model too, and did not release it.",
               "nl": "Het paper van Meta over Llama 2 (2023), tabel 2: de GPU-uren van de pretraining, op A100-GPU's. "
                     "Meta trainde ook een model van 34B, en bracht dat niet uit."},
    "llama31": {"en": "The page Meta published with Llama 3.1 (2024): the GPU hours of the training of each model, on H100 "
                      "GPUs; 39.3 million for the three together.",
                "nl": "De pagina die Meta bij Llama 3.1 publiceerde (2024): de GPU-uren van de training van elk model, op "
                      "H100-GPU's; 39,3 miljoen voor de drie samen."},
    "deepseek": {"en": "DeepSeek's report on V3 (2024), table 1: pretraining, longer texts and post-training, on "
                       "H800 GPUs, without the earlier research and the trial runs.",
                 "nl": "Het verslag van DeepSeek over V3 (2024), tabel 1: pretraining, langere teksten en "
                       "post-training, op H800-GPU's, zonder het eerdere onderzoek en de proefruns."},
}

DEMO_VALUES = list(range(1, len(MODELS) + 1))


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    k = min(max(int(value), 1), len(MODELS)) - 1
    m = MODELS[k]
    parts = [_t(0, 15, w["title"].format(**m), 12.5, 600),
             _t(0, 32, w["sub"].format(hours=num(m["hours"], lang), gpu=m["gpu"]), 11, 400, "var(--ea-ink-2)")]
    top = MODELS[-1]["hours"]
    x0, x1 = 92, 252
    for j, model in enumerate(MODELS):
        y = 44 + j * 18
        chosen = j == k
        parts.append(_t(0, y + 11, model["name"], 10.5, 600 if chosen else 400,
                        "var(--ea-ink)" if chosen else "var(--ea-ink-2)"))
        parts.append(_rect(x0, y + 2, max(2.0, (x1 - x0) * model["hours"] / top), 11,
                           "fill:var(--ea-ink)" if chosen else "fill:var(--ea-ink-3)", 1.5))
        parts.append(_t(326, y + 11, num(model["hours"], lang), 10.5, 600 if chosen else 400,
                        "var(--ea-ink)" if chosen else "var(--ea-ink-3)", "end"))
    y = 44 + len(MODELS) * 18 + 6
    parts.append(_line(0, y, 330, y))
    lines = [(w["cost"], w["cost_v"].format(v=num(m["hours"] * PRICE, lang))),
             (w["one"], w["one_v"].format(v=num(m["hours"] / YEAR_HOURS, lang))),
             (w["many"], w["many_v"].format(v=num(days(m["hours"], CLUSTER), lang)))]
    for j, (label, said) in enumerate(lines):
        yy = y + 20 + j * 18
        parts.append(_t(0, yy, label, 11, 400, "var(--ea-ink-2)"))
        parts.append(_t(326, yy, said, 11.5, 600, anchor="end"))
    return _svg(y + 20 + len(lines) * 18 - 6, w["alt"], parts), NOTES[m["source"]][lang]


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


# exercise 1: cost
def _c(hours: str) -> str:
    return ("def cost(runs, price):\n"
            "    total = 0\n"
            "    for run in runs:\n"
            f"        hours = {hours}\n"
            "        total = total + hours * price\n"
            "    return total\n")


_C_START = _c("...")
_C = _c('run["gpus"] * run["days"] * 24')
_C_DAYS = _c('run["gpus"] * run["days"]')
_C_PLUS = _c('run["gpus"] + run["days"] * 24')
_C_ONE = _c('run["days"] * 24')


# exercise 2: days
def _d(body: str) -> str:
    return "def days(gpu_hours, gpus):\n" f"    return {body}\n"


_D_START = _d("...")
_D = _d("gpu_hours / (gpus * 24)")
_D_ORDER = _d("gpu_hours / gpus * 24")
_D_HOURS = _d("gpu_hours / gpus")
_D_TIMES = _d("gpu_hours * gpus / 24")

BILL = Sort(
    name="bill",
    groups={
        "in": {"en": "In the 5,576,000 dollars", "nl": "In de 5.576.000 dollar"},
        "out": {"en": "Not in it", "nl": "Er niet in"},
    },
    items={
        "pre": {"en": "**Pretraining.** The model reads 14.8 trillion tokens.",
                "nl": "**Pretraining.** Het model leest 14,8 biljoen tokens."},
        "long": {"en": "**Longer texts.** A short extra training, so that the model can read much longer texts.",
                 "nl": "**Langere teksten.** Een korte extra training, zodat het model veel langere teksten kan lezen."},
        "post": {"en": "**Post-training.** Training on examples of good answers, and with rewards.",
                 "nl": "**Post-training.** Trainen op voorbeelden van goede antwoorden, en met beloningen."},
        "trials": {"en": "**Trial runs.** Smaller runs to choose the architecture, the method and the data.",
                   "nl": "**Proefruns.** Kleinere runs om de architectuur, de methode en de data te kiezen."},
        "research": {"en": "**Earlier research.** The work on the models before V3.",
                     "nl": "**Eerder onderzoek.** Het werk aan de modellen van voor V3."},
        "people": {"en": "**The people.** The pay of the team that built the model.",
                   "nl": "**De mensen.** Het loon van het team dat het model bouwde."},
        "buy": {"en": "**The GPUs themselves.** Buying the 2,048 H800 GPUs, and the building they stand in.",
                "nl": "**De GPU's zelf.** De 2.048 H800-GPU's kopen, en het gebouw waarin de GPU's staan."},
    },
    solution={"pre": "in", "long": "in", "post": "in", "trials": "out", "research": "out", "people": "out", "buy": "out"},
    nudge={
        "pre": {"en": "Which part of the training took 2,664,000 GPU hours?",
                "nl": "Welk deel van de training kostte 2.664.000 GPU-uren?"},
        "long": {"en": "DeepSeek counted three parts of the training of V3. Is this one of them?",
                 "nl": "DeepSeek telde drie delen van de training van V3. Is dit er één van?"},
        "post": {"en": "Does this part run on the GPUs, as part of the training of V3 itself?",
                 "nl": "Draait dit deel op de GPU's, als deel van de training van V3 zelf?"},
        "trials": {"en": "These runs use GPU hours too. Did DeepSeek count them in V3?",
                   "nl": "Ook deze runs gebruiken GPU-uren. Telde DeepSeek deze runs mee bij V3?"},
        "research": {"en": "Was this work part of the training of V3, or did it come before?",
                     "nl": "Hoorde dit werk bij de training van V3, of kwam het eerder?"},
        "people": {"en": "Is a salary a GPU hour?",
                   "nl": "Is een salaris een GPU-uur?"},
        "buy": {"en": "The 2 dollars is a rental price. Does a rental price include buying the GPUs?",
                "nl": "De 2 dollar is een huurprijs. Zit het kopen van de GPU's in een huurprijs?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the seven parts again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zeven onderdelen opnieuw uit."},
        "empty": {"en": "Put each part under *In the 5,576,000 dollars* or under *Not in it*.",
                  "nl": "Zet elk onderdeel onder *In de 5.576.000 dollar* of onder *Er niet in*."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs on the other side.",
                  "nl": "*{item}* hoort aan de andere kant."},
        "correct": {"en": "All {n} are right. The 5,576,000 dollars pays for the GPU hours of three parts of one training, at rental prices; everything around it is left out. What would a headline that says 'built for 5.6 million dollars' leave out?",
                    "nl": "Alle {n} kloppen. De 5.576.000 dollar betaalt de GPU-uren van drie delen van één training, tegen huurprijzen; alles eromheen valt erbuiten. Wat laat een kop als 'gebouwd voor 5,6 miljoen dollar' weg?"},
    },
)

EXERCISES = {
    "cost": FunctionBox(
        name="cost",
        function="cost",
        start=_C_START,
        solution=_C,
        visible=[(RUNS, PRICE)],
        hidden=[([{"gpus": 1, "days": 1}], 1.0), (RUNS, 1.0), ([{"gpus": 2048, "days": 56}], 2.0), ([], 3.0),
                ([{"gpus": 8, "days": 2}, {"gpus": 512, "days": 30}], 2.5)],
        first=[_DOTS],
        picture=cost_picture,
        mistakes=[
            Mistake("no-24", wrong=_C_DAYS,
                    en="`{call}` gives {got}: GPUs times days counts GPU days, and the price is per GPU hour, so it should give {want}. How many hours does a day have?",
                    nl="`{call}` geeft {got}: GPU's keer dagen telt GPU-dagen, en de prijs is per GPU-uur, dus het moet {want} zijn. Hoeveel uur heeft een dag?"),
            Mistake("plus", wrong=_C_PLUS,
                    en="`{call}` gives {got}: adding the GPUs to the hours counts each GPU once, not once for every hour, and it should give {want}. Do 512 GPUs for one hour make 513 GPU hours, or 512?",
                    nl="`{call}` geeft {got}: de GPU's optellen bij de uren telt elke GPU één keer, niet één keer per uur, en het moet {want} zijn. Maken 512 GPU's een uur lang 513 GPU-uren, of 512?"),
            Mistake("one-gpu", wrong=_C_ONE,
                    en="`{call}` gives {got}: these are the hours of a single GPU, and every GPU of the run is paid for, so it should give {want}. By what do you multiply the hours?",
                    nl="`{call}` geeft {got}: dit zijn de uren van één GPU, en elke GPU van de run wordt betaald, dus het moet {want} zijn. Waarmee vermenigvuldig je de uren?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the GPU hours of this run: its GPUs, times its days, times the hours of a day.",
                      "nl": "Vervang de `...` door de GPU-uren van deze run: de GPU's, keer de dagen, keer de uren van een dag."},
            "arguments": {"en": "The check calls `{call}`, with the runs and the price of one GPU hour, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(runs, price)`?",
                          "nl": "De check roept `{call}` aan, met de runs en de prijs van één GPU-uur, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(runs, price)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `hours` count every GPU of the run for every hour?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt `hours` elke GPU van de run voor elk uur?"},
            "almost": {"en": "Right for `C.RUNS` at 2 dollars. But `{call}` gives {got}, and it should give {want}. Does your function use `price`, and the GPUs and days of each run, and no fixed numbers other than 24?",
                       "nl": "Goed voor `C.RUNS` tegen 2 dollar. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `price`, en de GPU's en dagen van elke run, en geen vaste getallen behalve 24?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total`?"},
            "correct": {"en": "Right in all {n} cases. The five runs cost 989,952 dollars, and the final run is 737,280 of it: three quarters. In the picture: which run would a report of the model give?",
                        "nl": "Goed in alle {n} gevallen. De vijf runs kosten 989.952 dollar, en de laatste run is daarvan 737.280: driekwart. In de afbeelding: welke run zou een verslag van het model noemen?"},
        },
    ),
    "days": FunctionBox(
        name="days",
        function="days",
        start=_D_START,
        solution=_D,
        visible=[(SAME_HOURS, CLUSTER)],
        hidden=[(PER_TRILLION, CLUSTER), (MODELS[-1]["hours"], LLAMA3_GPUS), (24, 1), (48, 2), (1000, 10)],
        first=[_DOTS],
        picture=days_picture,
        mistakes=[
            Mistake("order", wrong=_D_ORDER,
                    en="`{call}` gives {got}: Python works from left to right, so it divides by `gpus` and then multiplies by 24, and it should give {want}. How do you make Python multiply first?",
                    nl="`{call}` geeft {got}: Python werkt van links naar rechts, dus deelt het door `gpus` en vermenigvuldigt het daarna met 24, en het moet {want} zijn. Hoe laat je Python eerst vermenigvuldigen?"),
            Mistake("hours", wrong=_D_HOURS,
                    en="`{call}` gives {got}: that is how many hours the whole cluster runs, and the question is days, so it should give {want}. How many hours make a day?",
                    nl="`{call}` geeft {got}: zoveel uur draait het hele cluster, en de vraag is dagen, dus het moet {want} zijn. Hoeveel uur is een dag?"),
            Mistake("times", wrong=_D_TIMES,
                    en="`{call}` gives {got}: more GPUs should give fewer days, and this grows with the GPUs, so it should give {want}. Do you multiply or divide by the GPUs?",
                    nl="`{call}` geeft {got}: meer GPU's horen minder dagen te geven, en dit groeit met de GPU's, dus het moet {want} zijn. Vermenigvuldig of deel je door de GPU's?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the GPU hours divided by the GPUs and by the 24 hours of a day.",
                      "nl": "Vervang de `...` door de GPU-uren gedeeld door de GPU's en door de 24 uren van een dag."},
            "arguments": {"en": "The check calls `{call}`, with the GPU hours and the number of GPUs, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(gpu_hours, gpus)`?",
                          "nl": "De check roept `{call}` aan, met de GPU-uren en het aantal GPU's, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(gpu_hours, gpus)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your line divide `gpu_hours` by the GPUs and by 24?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Deelt je regel `gpu_hours` door de GPU's en door 24?"},
            "almost": {"en": "Right for 2,048 GPUs. But `{call}` gives {got}, and it should give {want}. Does your function use `gpu_hours` and `gpus`, and no fixed number other than 24?",
                       "nl": "Goed voor 2.048 GPU's. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `gpu_hours` en `gpus`, en geen vast getal behalve 24?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does the line start with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de regel met `return`?"},
            "correct": {"en": "Right in all {n} cases. 2,664,000 GPU hours on 2,048 GPUs take 54.2 days, DeepSeek's \"less than two months\". In the picture: what do eight times as many GPUs do to the days, and to the GPU hours?",
                        "nl": "Goed in alle {n} gevallen. 2.664.000 GPU-uren op 2.048 GPU's duren 54,2 dagen, de \"minder dan twee maanden\" van DeepSeek. In de afbeelding: wat doen acht keer zoveel GPU's met de dagen, en met de GPU-uren?"},
        },
    ),
    "bill": BILL,
}

# The lesson is done when these are correct. "same" is the choice question before the demo.
REQUIRED = ["cost", "days", "bill", "same"]

_SORTED = dict(BILL.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cost", _C_START, "empty", None),
    ("cost", "", "empty", None),
    ("cost", _C, "correct", None),
    ("cost", _c('24 * run["days"] * run["gpus"]'), "correct", None),
    ("cost", _c('run["gpus"] * (run["days"] * 24)'), "correct", None),
    ("cost", "def cost(runs, price):\n"
             "    return sum(r[\"gpus\"] * r[\"days\"] * 24 for r in runs) * price\n", "correct", None),
    ("cost", _C_DAYS, "wrong", "no-24"),
    ("cost", _C_PLUS, "wrong", "plus"),
    ("cost", _C_ONE, "wrong", "one-gpu"),
    ("cost", _C.replace("hours * price", "hours * 2"), "almost", None),
    ("cost", _c('run["gpu"] * run["days"] * 24'), "error", None),
    ("cost", _c('run["gpus"] * run["days"] * 12'), "wrong", None),
    ("cost", _C.replace("def cost(runs, price):", "def cost(runs):"), "error", "arguments"),
    ("cost", _C.replace("    return total\n", ""), "wrong", "no-return"),
    ("cost", _C.replace("    return total\n", "    print(total)\n"), "wrong", "prints"),
    ("cost", _C.replace("    return total\n", "    extra = ...\n    return total\n"), "wrong", "dots-left"),
    ("cost", "x = 1\n", "wrong", "no-function"),
    ("days", _D_START, "empty", None),
    ("days", "", "empty", None),
    ("days", _D, "correct", None),
    ("days", _d("gpu_hours / gpus / 24"), "correct", None),
    ("days", _d("gpu_hours / 24 / gpus"), "correct", None),
    ("days", _D_ORDER, "wrong", "order"),
    ("days", _D_HOURS, "wrong", "hours"),
    ("days", _D_TIMES, "wrong", "times"),
    ("days", _d("gpu_hours / (2048 * 24)"), "almost", None),
    ("days", _d("gpu_hour / (gpus * 24)"), "error", None),
    ("days", _d("gpu_hours / (gpus * 12)"), "wrong", None),
    ("days", _D.replace("def days(gpu_hours, gpus):", "def days(gpu_hours):"), "error", "arguments"),
    ("days", "def days(gpu_hours, gpus):\n    gpu_hours / (gpus * 24)\n", "wrong", "no-return"),
    ("days", "def days(gpu_hours, gpus):\n    print(gpu_hours / (gpus * 24))\n", "wrong", "prints"),
    ("days", "def days(gpu_hours, gpus):\n    extra = ...\n    return gpu_hours / (gpus * 24)\n", "wrong", "dots-left"),
    ("days", "x = 1\n", "wrong", "no-function"),
    ("bill", {}, "empty", None),
    ("bill", _SORTED, "correct", None),
    ("bill", {"pre": "in", "trials": "out"}, "wrong", "unfinished"),
    ("bill", _SORTED | {"pre": "out"}, "wrong", "wrong:pre"),
    ("bill", _SORTED | {"long": "out"}, "wrong", "wrong:long"),
    ("bill", _SORTED | {"post": "out"}, "wrong", "wrong:post"),
    ("bill", _SORTED | {"trials": "in"}, "wrong", "wrong:trials"),
    ("bill", _SORTED | {"research": "in"}, "wrong", "wrong:research"),
    ("bill", _SORTED | {"people": "in"}, "wrong", "wrong:people"),
    ("bill", _SORTED | {"buy": "in"}, "wrong", "wrong:buy"),
]
