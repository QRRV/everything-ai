# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-counterfactuals: the logistic regression of rai-reason-codes on three columns of UCI Adult
and its applicant aged 30, who gets a no. The slider of *The idea* changes how many years of age count as one step of
Wachter et al.'s distance, and shows which single change (age to 81, education_num to 16 or hours_per_week to 96) is
the closest counterfactual. The learner writes how much one column alone has to change to bring the logit to the
cut-off, and Wachter et al.'s distance, the L1 distance with each column in its own scale; the pictures draw the
learner's changes against the range of the data, and the learner's distance for six counterfactuals. The two choice
questions are about a column a counterfactual leaves alone, and about a recourse after the model is retrained."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
# The model of rai-reason-codes: scikit-learn's LogisticRegression(max_iter=1000) on all 48,842 rows, rounded.
WEIGHTS = [0.046, 0.345, 0.042]
BIAS = -8.48
PERSON = [30, 9, 40]             # aged 30, a high-school diploma (education level 9), 40 hours a week
YEARS = [1, 3, 5, 7, 8, 10, 14]  # the stops of the slider: how many years of age count as one step
# Six counterfactuals in whole numbers: one column alone, or education_num with the fewest hours that then give a yes.
OTHERS = [[81, 9, 40], [30, 16, 40], [30, 15, 46], [30, 14, 55], [30, 13, 63], [30, 9, 96]]


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult as rows of whole numbers. Read-only."""
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    X = people[COLUMNS].to_numpy()
    X.flags.writeable = False
    return X


@functools.lru_cache(maxsize=1)
def mad() -> tuple:
    """The median absolute deviation of each column over all 48,842 rows: the median of |x - the column's median|."""
    import numpy as np
    X = table()
    return tuple(float(m) for m in np.median(np.abs(X - np.median(X, axis=0)), axis=0))


@functools.lru_cache(maxsize=1)
def spread() -> tuple:
    """The standard deviation of each column over all rows."""
    return tuple(float(s) for s in table().std(axis=0))


def logit(row) -> float:
    total = BIAS
    for w, x in zip(WEIGHTS, row):
        total += w * x
    return total


def change_of(k: int, row=PERSON) -> float:
    """How much column k alone has to change to bring the logit to 0."""
    return -logit(row) / WEIGHTS[k]


def whole(k: int, row=PERSON) -> list:
    """The row with column k changed by the smallest whole number that brings the logit to 0 or above."""
    out = list(row)
    out[k] = row[k] + math.ceil(change_of(k, row))
    return out


def distance_of(a, b, scale) -> float:
    total = 0.0
    for x, y, s in zip(a, b, scale):
        total += abs(y - x) / s
    return total


def _show(x: float, lang: str, digits: int = 3, sign: bool = False) -> str:
    """A number with this many decimals and a real minus sign, with a plus sign if asked; a decimal comma in Dutch."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-+")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _short(x: float, lang: str) -> str:
    """A number with up to three decimals, without trailing zeros: 7, 5.1, 18.667."""
    text = f"{x:.3f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

# --- the slider of *The idea*: how many years count as one step ----------------------------------------------------------

_YEARS_WORDS = {
    "en": {"title": "Three counterfactuals, one step of age = {s} {years}", "x": "distance in steps",
           "year": "year", "years": "years",
           "rows": ["age to 81", "education_num to 16", "hours_per_week to 96"],
           "which": ["age: be 81", "education_num: level 16", "hours_per_week: 96 hours"],
           "alt": "Three bars, the distance of three counterfactuals of the applicant aged 30 when one step of age is "
                  "{s} {years}; the closest is filled. {said}",
           "said": "One step of age is {s} {years}; one step of education_num is 1 level and one of hours_per_week 3 "
                   "hours, their MADs. Being 81 is <span class=\"c-loss ea-num\">{a}</span> steps away, education "
                   "level 16 {e} and 96 hours a week {h}: of these three, the closest changes {which}."},
    "nl": {"title": "Drie counterfactuals, één stap leeftijd = {s} {years}", "x": "afstand in stappen",
           "year": "jaar", "years": "jaar",
           "rows": ["age naar 81", "education_num naar 16", "hours_per_week naar 96"],
           "which": ["age: 81 zijn", "education_num: niveau 16", "hours_per_week: 96 uur"],
           "alt": "Drie staven, de afstand van drie counterfactuals van de aanvrager van 30 als één stap leeftijd {s} "
                  "{years} is; de dichtstbijzijnde is gevuld. {said}",
           "said": "Eén stap leeftijd is {s} {years}; één stap education_num is 1 niveau en één stap hours_per_week 3 "
                   "uur, hun MAD's. 81 zijn ligt <span class=\"c-loss ea-num\">{a}</span> stappen weg, "
                   "opleidingsniveau 16 {e} en 96 uur per week {h}: van deze drie verandert de "
                   "dichtstbijzijnde {which}."},
}


def single_distances(years: float) -> list:
    """The distance of the three single whole-number changes when one step of age is `years` (education 1, hours 3)."""
    scale = [years, 1, 3]
    return [distance_of(PERSON, whole(k), scale) for k in range(3)]


def years_picture(s: float, lang: str) -> tuple[str, str]:
    """For the slider: a bar per single change, its distance when one step of age is s years; the closest filled."""
    w = _YEARS_WORDS[lang]
    d = single_distances(s)
    best = min(range(3), key=lambda k: d[k])
    top = max(single_distances(min(YEARS))) * 1.04
    p = chart.Panel(x=(0, top), y=(-0.2, 3.2), title=w["title"].format(s=s, years=w["year"] if s == 1 else w["years"]),
                    xlabel=w["x"], ylabel="", yticks=[])
    for k in range(3):
        y = 2 - k
        p.rect(0, y + 0.08, max(d[k], top * 0.004), y + 0.42, role="loss", solid=k == best)
        p.text(0, y + 0.72, f"{w['rows'][k]}: {_show(d[k], lang, 1)}", role="ink", anchor="start", halo=True)
    fill = {"s": s, "years": w["year"] if s == 1 else w["years"], "a": _show(d[0], lang, 1),
            "e": _show(d[1], lang, 1), "h": _show(d[2], lang, 1), "which": w["which"][best]}
    said = w["said"].format(**fill)
    alt = w["alt"].format(s=s, years=fill["years"], said=re.sub(r"<[^>]+>", "", said))
    svg = chart.figure([p], alt=alt, width=460, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, said


# --- helpers for the pictures of the exercises --------------------------------------------------------------------------

def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _number(v):
    """A float from what a function gave back, or None when it is not one finite number."""
    if isinstance(v, bool):
        return None
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        v = v.item()
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _run(run):
    from everything_ai import check as checking
    try:
        return checking.with_time_limit(run)
    except Exception:
        return None


# --- exercise 1: how much one column has to change ----------------------------------------------------------------------

_NEEDED_WORDS = {
    "en": {"title": "Your change_needed(), for the applicant aged 30", "now": "now", "then": "with your change",
           "alt": "Three small charts, one for each column: the applicant's value, the value your change_needed() "
                  "takes it to, and the range of the column in the data. {key}",
           "key": "Your change_needed() takes age from 30 to {a} ({da}), education_num from 9 to {e} ({de}) and "
                  "hours_per_week from 40 to {h} ({dh}), each as the only column that changes. The thin line is the "
                  "range in the data: age goes up to {ma}, education_num to {me} and hours_per_week to {mh}."},
    "nl": {"title": "Jouw change_needed(), voor de aanvrager van 30", "now": "nu", "then": "met jouw verandering",
           "alt": "Drie kleine grafieken, één per kolom: de waarde van de aanvrager, de waarde waar je change_needed() "
                  "hem naartoe brengt, en het bereik van de kolom in de data. {key}",
           "key": "Jouw change_needed() brengt age van 30 naar {a} ({da}), education_num van 9 naar {e} ({de}) en "
                  "hours_per_week van 40 naar {h} ({dh}), telkens als enige kolom die verandert. De dunne lijn is het "
                  "bereik in de data: age gaat tot {ma}, education_num tot {me} en hours_per_week tot {mh}."},
}


def _needed_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""

    def run():
        out = []
        for k in range(3):
            g = _number(fn(list(WEIGHTS), BIAS, list(PERSON), k))
            if g is None:
                return None
            out.append(g)
        return out

    got = _run(run)
    if got is None:
        return ""
    lang = seen.lang
    w = _NEEDED_WORDS[lang]
    X = table()
    lows, highs = X.min(axis=0), X.max(axis=0)
    panels = []
    for k, name in enumerate(COLUMNS):
        now, then = PERSON[k], PERSON[k] + got[k]
        lo, hi = min(float(lows[k]), now, then), max(float(highs[k]), now, then)
        pad = (hi - lo) * 0.08 + 1e-9
        p = chart.Panel(x=(0, 1), y=(lo - pad, hi + pad), title=name, xlabel="", ylabel="", xticks=[])
        p.line([0.3, 0.3], [float(lows[k]), float(highs[k])], role="ink-3", width=1.4)
        if abs(then - now) > 1e-12:
            p.arrow(0.3, now, 0.3, then, role="pred", width=2)
        p.points([0.3], [now], role="data", r=4.5)
        p.points([0.3], [then], role="pred", r=4.5)
        p.text(0.42, now, f"{w['now']} {_short(now, lang)}", role="ink-2", anchor="start", halo=True)
        p.text(0.42, then, _short(then, lang), role="pred", anchor="start", halo=True)
        panels.append(p)
    fill = {"a": _short(PERSON[0] + got[0], lang), "e": _short(PERSON[1] + got[1], lang),
            "h": _short(PERSON[2] + got[2], lang), "da": _short_signed(got[0], lang), "de": _short_signed(got[1], lang),
            "dh": _short_signed(got[2], lang), "ma": int(highs[0]), "me": int(highs[1]), "mh": int(highs[2])}
    key = w["key"].format(**fill)
    svg = chart.figure(panels, alt=w["alt"].format(key=key), width=170, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


def _short_signed(x: float, lang: str) -> str:
    text = _short(x, lang)
    return text if text.startswith("−") or text == "0" else "+" + text


# --- exercise 2: Wachter et al.'s distance ------------------------------------------------------------------------------

_DISTANCE_WORDS = {
    "en": {"title": "Your distance(), for six counterfactuals of the applicant aged 30", "x": "your distance",
           "rows": ["age 81", "education_num 16", "education_num 15, hours_per_week 46",
                    "education_num 14, hours_per_week 55", "education_num 13, hours_per_week 63",
                    "hours_per_week 96"],
           "alt": "Six bars, the distance your distance() gives from the applicant aged 30 to six rows that get a yes, "
                  "with the MADs 10, 1 and 3 as scale; the closest is filled. {key}",
           "key": "Each row gets a yes from the model. With your distance and the MADs 10, 1 and 3 as scale, the "
                  "closest is {best} ({bd}); of the five that leave age alone, the closest is {act} ({ad})."},
    "nl": {"title": "Jouw distance(), voor zes counterfactuals van de aanvrager van 30", "x": "jouw afstand",
           "rows": ["age 81", "education_num 16", "education_num 15, hours_per_week 46",
                    "education_num 14, hours_per_week 55", "education_num 13, hours_per_week 63",
                    "hours_per_week 96"],
           "alt": "Zes staven, de afstand die je distance() geeft van de aanvrager van 30 tot zes rijen die een ja "
                  "krijgen, met de MAD's 10, 1 en 3 als schaal; de dichtstbijzijnde is gevuld. {key}",
           "key": "Elke rij krijgt een ja van het model. Met jouw afstand en de MAD's 10, 1 en 3 als schaal is de "
                  "dichtstbijzijnde {best} ({bd}); van de vijf die age laten staan is dat {act} ({ad})."},
}


def _distance_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    scale = list(mad())

    def run():
        out = []
        for other in OTHERS:
            g = _number(fn(list(PERSON), list(other), list(scale)))
            if g is None:
                return None
            out.append(g)
        return out

    got = _run(run)
    if got is None:
        return ""
    lang = seen.lang
    w = _DISTANCE_WORDS[lang]
    best = min(range(len(got)), key=lambda i: got[i])
    act = min(range(1, len(got)), key=lambda i: got[i])
    lo, hi = min(got + [0.0]), max(got + [0.0])
    pad = (hi - lo) * 0.04 + 1e-9
    n = len(got)
    p = chart.Panel(x=(lo - pad, hi + pad), y=(-0.2, n + 0.2), title="", xlabel=w["x"], ylabel="", yticks=[])
    for i, d in enumerate(got):
        y = n - 1 - i
        x0, x1 = (0.0, d) if d >= 0 else (d, 0.0)
        p.rect(x0, y + 0.08, max(x1, x0 + (hi - lo) * 0.004), y + 0.42, role="loss", solid=i == best)
        p.text(lo - pad, y + 0.72, f"{w['rows'][i]}: {_short(d, lang)}", role="ink", anchor="start", halo=True)
    key = w["key"].format(best=w["rows"][best], bd=_short(got[best], lang), act=w["rows"][act],
                          ad=_short(got[act], lang))
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=460, height=300)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "race" and "later" are choice questions.
REQUIRED = ["needed", "distance", "race", "later"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_CN_HEAD = ("def change_needed(weights, bias, person, k):\n"
            "    # the logit is the bias plus each weight times its column; the bank says yes from a logit of 0\n")
_CN_START = _CN_HEAD + "    ...\n"


def _cn(start: str = "bias", ret: str = "-logit / weights[k]") -> str:
    return (_CN_HEAD + f"    logit = {start}\n    for j in range(len(person)):\n"
            "        logit = logit + weights[j] * person[j]\n" + f"    return {ret}\n")


_CN_SOLUTION = _cn()
_CN_NO_BIAS = _cn(start="0")
_CN_SIGN = _cn(ret="logit / weights[k]")
_CN_VALUE = _cn(ret="person[k] - logit / weights[k]")
_CN_ALL = _cn(ret="-logit / sum(weights)")
_CN_ABS = _cn(ret="abs(logit) / weights[k]")
_W, _B = list(WEIGHTS), BIAS
_CN_VISIBLE = [(_W, _B, [30, 9, 40], 1)]
_CN_HIDDEN = [
    (_W, _B, [30, 9, 40], 0),
    (_W, _B, [30, 9, 40], 2),
    (_W, _B, [52, 10, 30], 2),
    (_W, _B, [50, 14, 50], 1),
    ([0.5, -1.0], 0.2, [1, 2], 1),
    ([2.0], -1.0, [0.25], 0),
    ([0.1, 0.2, 0.3, 0.4], 0.0, [1, 1, 1, 1], 3),
]

# --- exercise 2
_DI_HEAD = ("def distance(person, other, scale):\n"
            "    # how far other lies from person: each column's change, measured in its own scale\n")
_DI_START = _DI_HEAD + "    ...\n"


def _di(term: str = "abs(other[k] - person[k]) / scale[k]", ret: str = "total", top: str = "") -> str:
    return (top + _DI_HEAD + "    total = 0\n    for k in range(len(person)):\n"
            + f"        total = total + {term}\n" + f"    return {ret}\n")


_DI_SOLUTION = _di()
_DI_SIGNED = _di(term="(other[k] - person[k]) / scale[k]")
_DI_TIMES = _di(term="abs(other[k] - person[k]) * scale[k]")
_DI_PLAIN = _di(term="abs(other[k] - person[k])")
_DI_SQUARED = _di(term="((other[k] - person[k]) / scale[k]) ** 2")
_DI_EUCLID = _di(term="((other[k] - person[k]) / scale[k]) ** 2", ret="math.sqrt(total)", top="import math\n\n\n")
_DI_VISIBLE = [([30, 9, 40], [30, 15, 46], [10, 1, 3])]
_DI_HIDDEN = [
    ([30, 9, 40], [81, 9, 40], [10, 1, 3]),
    ([30, 9, 40], [30, 16, 40], [10, 1, 3]),
    ([30, 9, 40], [25, 9, 46], [10, 1, 3]),
    ([30, 9, 40], [30, 9, 40], [10, 1, 3]),
    ([1, 2], [3, -2], [2, 4]),
    ([5.0], [2.0], [0.5]),
    ([0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8]),
]

EXERCISES = {
    "needed": FunctionBox(
        name="needed",
        picture=_needed_picture,
        function="change_needed",
        start=_CN_START,
        solution=_CN_SOLUTION,
        visible=_CN_VISIBLE,
        hidden=_CN_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("no-bias", wrong=_CN_NO_BIAS,
                    en="`{call}` gives {got}, and it should give {want}: the change as if the bias were 0. The logit starts at `bias`, and then each weight times its column is added.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de verandering alsof de bias 0 is. De logit begint bij `bias`, en daarna komt elk gewicht keer zijn kolom erbij."),
            Mistake("sign", wrong=_CN_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: the right size, the wrong way. Changing column `k` by an amount moves the logit by `weights[k]` times that amount, and the logit has to move by minus itself to reach 0: return `-logit / weights[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede grootte, de verkeerde kant op. Als kolom `k` met een hoeveelheid verandert, beweegt de logit met `weights[k]` keer die hoeveelheid, en de logit moet met min zichzelf bewegen om op 0 te komen: geef `-logit / weights[k]` terug."),
            Mistake("value", wrong=_CN_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: the value column `k` has to reach, not how much it has to change. Leave out `person[k]`: return only `-logit / weights[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de waarde die kolom `k` moet halen, niet hoeveel hij moet veranderen. Laat `person[k]` weg: geef alleen `-logit / weights[k]` terug."),
            Mistake("all", wrong=_CN_ALL,
                    en="`{call}` gives {got}, and it should give {want}: how much every column would have to change, by the same amount each, to reach 0 together. Only column `k` changes here: divide by `weights[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hoeveel elke kolom zou moeten veranderen, allemaal evenveel, om samen op 0 te komen. Hier verandert alleen kolom `k`: deel door `weights[k]`."),
            Mistake("abs", wrong=_CN_ABS,
                    en="`{call}` gives {got}, and it should give {want}: the right size, but for a row whose logit is already above 0 the change has to go the other way, back to 0. Use `-logit`, not `abs(logit)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede grootte, maar voor een rij waarvan de logit al boven 0 ligt, moet de verandering de andere kant op, terug naar 0. Gebruik `-logit`, niet `abs(logit)`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: start `logit` at `bias`, add `weights[j] * person[j]` for each column `j`, and return `-logit / weights[k]`.",
                      "nl": "Vervang de `...`: begin `logit` bij `bias`, tel er `weights[j] * person[j]` bij op voor elke kolom `j`, en geef `-logit / weights[k]` terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: the logit is −8.48 + 0.046 · 30 + 0.345 · 9 + 0.042 · 40 = −2.315, and education_num, weight 0.345, has to make up those 2.315 as the only column that changes.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: de logit is −8,48 + 0,046 · 30 + 0,345 · 9 + 0,042 · 40 = −2,315, en education_num, met gewicht 0,345, moet dat als enige kolom goedmaken: 2,315."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any weights, any number of columns, any `k`, and a row that already gets a yes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor andere weights, elk aantal kolommen, elke `k`, en een rij die al een ja krijgt?"},
            "correct": {"en": "Right in all {n} cases. When only one column of the applicant aged 30 changes, age has to rise by 50.3 years, to 80.3, education_num by 6.71 levels, to 15.7, or hours_per_week by 55.1 hours, to 95.1. In whole numbers that is age 81, level 16 or 96 hours a week: each the smallest whole change that gives a yes.",
                        "nl": "Goed in alle {n} gevallen. Als bij de aanvrager van 30 maar één kolom verandert, moet age 50,3 jaar omhoog, naar 80,3, education_num 6,71 niveaus, naar 15,7, of hours_per_week 55,1 uur, naar 95,1. In hele getallen is dat age 81, niveau 16 of 96 uur per week: telkens de kleinste hele verandering die een ja geeft."},
        },
    ),
    "distance": FunctionBox(
        name="distance",
        picture=_distance_picture,
        function="distance",
        start=_DI_START,
        solution=_DI_SOLUTION,
        visible=_DI_VISIBLE,
        hidden=_DI_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("signed", wrong=_DI_SIGNED,
                    en="`{call}` gives {got}, and it should give {want}: a change down cancels a change up. The distance adds the size of every change: use `abs(other[k] - person[k])`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een verandering omlaag heft een verandering omhoog op. De afstand telt de grootte van elke verandering op: gebruik `abs(other[k] - person[k])`."),
            Mistake("times", wrong=_DI_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: each change times its scale, not divided by it. A change of 3 hours is one step when the scale of hours is 3: divide by `scale[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke verandering keer zijn schaal, niet erdoor gedeeld. Een verandering van 3 uur is één stap als de schaal van uren 3 is: deel door `scale[k]`."),
            Mistake("plain", wrong=_DI_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: the changes added up in their own units, as if a year, a level and an hour were the same. Divide each by its scale, `scale[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de veranderingen opgeteld in hun eigen eenheden, alsof een jaar, een niveau en een uur hetzelfde zijn. Deel elke verandering door zijn schaal, `scale[k]`."),
            Mistake("squared", wrong=_DI_SQUARED,
                    en="`{call}` gives {got}, and it should give {want}: the squares of the scaled changes, added up. Wachter et al. add the sizes themselves, so a big change in one column counts no more than several small ones that add up to it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kwadraten van de geschaalde veranderingen, opgeteld. Wachter en anderen tellen de groottes zelf op, zodat een grote verandering in één kolom niet zwaarder telt dan een paar kleine die samen even groot zijn."),
            Mistake("euclid", wrong=_DI_EUCLID,
                    en="`{call}` gives {got}, and it should give {want}: the straight-line distance of the scaled changes, the root of the sum of their squares. Wachter et al. add the sizes of the scaled changes, with no squares and no root.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lengte in een rechte lijn van de geschaalde veranderingen, de wortel uit de som van hun kwadraten. Wachter en anderen tellen de groottes van de geschaalde veranderingen op, zonder kwadraten en zonder wortel."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: start `total` at 0, add `abs(other[k] - person[k]) / scale[k]` for each column `k`, and return `total`.",
                      "nl": "Vervang de `...`: begin `total` bij 0, tel er `abs(other[k] - person[k]) / scale[k]` bij op voor elke kolom `k`, en geef `total` terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: age does not change, education_num changes by 6 levels of scale 1, and hours_per_week by 6 hours of scale 3, so 0 + 6 + 2.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: age verandert niet, education_num verandert 6 niveaus met schaal 1, en hours_per_week 6 uur met schaal 3, dus 0 + 6 + 2."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of columns, and for a column that goes down?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal kolommen, en voor een kolom die omlaag gaat?"},
            "correct": {"en": "Right in all {n} cases. With the MADs 10, 1 and 3 as scale, being 81 is 5.1 away, education level 16 is 7, level 15 with 46 hours 8, level 14 with 55 hours 10, level 13 with 63 hours 11.667, and 96 hours a week 18.667. The closest counterfactual asks the applicant to be 51 years older; of the others, level 16, a doctorate, is closest.",
                        "nl": "Goed in alle {n} gevallen. Met de MAD's 10, 1 en 3 als schaal ligt 81 zijn op 5,1, opleidingsniveau 16 op 7, niveau 15 met 46 uur op 8, niveau 14 met 55 uur op 10, niveau 13 met 63 uur op 11,667, en 96 uur per week op 18,667. De dichtstbijzijnde counterfactual vraagt de aanvrager 51 jaar ouder te zijn; van de andere ligt niveau 16, een doctorstitel, het dichtst bij."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("needed", _CN_START, "empty", None),
    ("needed", _CN_SOLUTION, "correct", None),
    ("needed", _CN_HEAD + "    logit = bias\n    for w, x in zip(weights, person):\n        logit += w * x\n"
               "    return (0 - logit) / weights[k]\n", "correct", None),
    ("needed", _CN_HEAD + "    logit = bias\n    for j in range(len(person)):\n"
               "        logit = logit + weights[j] * person[j]\n    goal = person[k] - logit / weights[k]\n"
               "    return goal - person[k]\n", "correct", None),
    ("needed", _CN_NO_BIAS, "wrong", "no-bias"),
    ("needed", _CN_SIGN, "wrong", "sign"),
    ("needed", _CN_VALUE, "wrong", "value"),
    ("needed", _CN_ALL, "wrong", "all"),
    ("needed", _CN_ABS, "wrong", "abs"),
    ("needed", _CN_HEAD + "    return 2.315 / 0.345\n", "almost", None),
    ("needed", _cn(ret=""), "wrong", "no-return"),
    ("needed", _CN_HEAD + "    ...\n    return 2.315 / 0.345\n", "wrong", "dots-left"),
    ("distance", _DI_START, "empty", None),
    ("distance", _DI_SOLUTION, "correct", None),
    ("distance", _DI_HEAD + "    total = 0\n    for a, b, s in zip(person, other, scale):\n"
                 "        if b > a:\n            total += (b - a) / s\n        else:\n            total += (a - b) / s\n"
                 "    return total\n", "correct", None),
    ("distance", _DI_SIGNED, "wrong", "signed"),
    ("distance", _DI_TIMES, "wrong", "times"),
    ("distance", _DI_PLAIN, "wrong", "plain"),
    ("distance", _DI_SQUARED, "wrong", "squared"),
    ("distance", _DI_EUCLID, "wrong", "euclid"),
    ("distance", _DI_HEAD + "    return 8.0\n", "almost", None),
    ("distance", _di(ret=""), "wrong", "no-return"),
    ("distance", _DI_HEAD + "    ...\n    return 8.0\n", "wrong", "dots-left"),
]
