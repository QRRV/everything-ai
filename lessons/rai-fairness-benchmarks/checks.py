# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-fairness-benchmarks: the selection rate per group of the example in Fairlearn's own
documentation, worked out by hand, and how far apart the gains of reweighing lie over ten samples of 1,000 people from
the table UCI Adult; with the pictures of the ten samples, of the learner's selection rates and of the learner's
gains."""
import html
import math
import numbers
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import yours

# What the demo works out (tests/test_rai_fairness_benchmarks.py checks it): for ten samples of 1,000 people, seeds 0
# to 9, how often women get a yes divided by how often men do, without weights and with reweighing.
PLAIN = [0.45, 0.36, 0.24, 0.56, 0.32, 0.55, 0.56, 0.31, 0.34, 0.22]
REWEIGHED = [0.72, 0.57, 0.46, 0.63, 0.64, 0.78, 0.75, 0.82, 0.6, 0.67]
WHOLE = (0.28, 0.57)   # the same on the whole table, in the lesson rai-reweighing
# How often the model of the previous lessons, with one threshold of 0.5, is right on the test set.
ACCURACY = {"all": 0.834, "women": 0.916, "men": 0.794}

# Exercise 1: the example of Fairlearn's MetricFrame, version 0.15.
SEX = ["Female", "Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male"]
Y_PRED = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]


def rates(sex: list, y_pred: list) -> dict:
    """The selection rate per group: of each group, the share whose answer is 1."""
    people, yes = {}, {}
    for s, y in zip(sex, y_pred):
        people[s] = people.get(s, 0) + 1
        yes[s] = yes.get(s, 0) + y
    return {s: yes[s] / people[s] for s in people}


def gains(plain: list, reweighed: list) -> list:
    """Per sample, how much reweighing raised the ratio."""
    return [r - p for p, r in zip(plain, reweighed)]


def spread(plain: list, reweighed: list) -> float:
    """The largest gain minus the smallest."""
    g = gains(plain, reweighed)
    return max(g) - min(g)


def _num(x: float, decimal: str, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", decimal)


def _short(x: float, decimal: str) -> str:
    """A number with at most two decimals, without zeros at the end: 0.8, 0.44, 1."""
    return f"{round(x, 2) + 0.0:g}".replace(".", decimal)


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def picture_samples(labels: dict) -> str:
    """The whole table on top, then a row per sample: how often women get a yes divided by how often men do, without
    weights (open dot) and with reweighing (filled dot), joined by a line. `labels` gives "title", "whole" (the name
    of the top row), "seed" (the word before each seed), "xlabel", "key", "decimal" and "alt"."""
    from everything_ai import chart
    rows = [(labels["whole"], WHOLE)]
    for seed, pair in enumerate(zip(PLAIN, REWEIGHED)):
        rows.append((f'{labels["seed"]} {seed}', pair))
    n = len(rows)
    panel = chart.Panel(x=(-0.3, 1), y=(0, n + 0.4), xlabel=labels["xlabel"], yticks=[],
                        xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    for i, (name, (plain, reweighed)) in enumerate(rows):
        y = n - 0.1 if i == 0 else n - i - 0.6
        panel.text(-0.28, y, name, "ink" if i == 0 else "ink-2", anchor="start")
        panel.line([plain, reweighed], [y, y], role="data", width=2)
        panel.points([plain], [y], role="data", r=5, hollow=True)
        panel.points([reweighed], [y], role="data", r=5)
    svg = chart.figure([panel], alt=labels["alt"], width=520, height=380)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _rate_values(v) -> dict | None:
    """The learner's selection rates, when they are a dict of 1 to 6 groups with a name and a number each."""
    if not isinstance(v, dict) or not 1 <= len(v) <= 6:
        return None
    out = {}
    for g, x in v.items():
        if not isinstance(g, str) or _number(x) is None:
            return None
        out[g] = _number(x)
    return out


def _same_rates(got, want) -> bool:
    a = _rate_values(got)
    return a is not None and set(a) == set(want) and all(abs(a[g] - want[g]) < 1e-9 for g in want)


def _all_rates(value):
    def match(got, text):
        a = _rate_values(got)
        return a is not None and all(abs(x - value) < 1e-9 for x in a.values())
    return match


def _above_one(got, text) -> bool:
    a = _rate_values(got)
    return a is not None and any(x > 1 for x in a.values())


def _same_number(got, want) -> bool:
    return _number(got) is not None and abs(_number(got) - want) < 1e-9


def _near(value):
    def match(got, text):
        return _number(got) is not None and abs(_number(got) - value) < 1e-6
    return match


def _ticks(lo: float, hi: float) -> list:
    """Round values for the axis from lo to hi, at most six."""
    for step in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100):
        if (hi - lo) / step <= 5:
            break
    first = math.ceil(lo / step - 1e-9)
    return [round(k * step, 4) for k in range(first, math.floor(hi / step + 1e-9) + 1)]


_RATES_WORDS = {
    "en": {"title": "Your selection rates per group", "xlabel": "selection rate: the share with a yes",
           "of": "of", "between": "Largest minus smallest: {diff}. Smallest divided by largest: {ratio}. That is what Fairlearn's difference() and ratio() give.",
           "key": "A bar per group in your rate, with your yes and people for that group.",
           "alt": "A bar per group: {shown}."},
    "nl": {"title": "Jouw selectiepercentages per groep", "xlabel": "selectiepercentage: het deel met een ja",
           "of": "van", "between": "Grootste min kleinste: {diff}. Kleinste gedeeld door grootste: {ratio}. Dat geven difference() en ratio() van Fairlearn.",
           "key": "Een staaf per groep in je rate, met je yes en people voor die groep.",
           "alt": "Een staaf per groep: {shown}."},
}


def _rates_picture(seen) -> str:
    """A bar per group with the learner's selection rate, and the counts behind it when the learner has them; under
    it the largest minus the smallest rate, and the smallest divided by the largest."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    values = _rate_values(names.get("rate"))
    if values is None or min(values.values()) < 0 or max(values.values()) > 100:
        return ""
    people, yes = names.get("people"), names.get("yes")
    w = _RATES_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    top = max(1.0, max(values.values()))
    n = len(values)
    ticks = _ticks(0, top)
    panel = chart.Panel(x=(0, top * 1.15), y=(0, 1.2 * n + 0.1), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    shown = []
    for i, (g, x) in enumerate(values.items()):
        y = 1.2 * (n - i - 1) + 0.1
        label = g
        if isinstance(people, dict) and isinstance(yes, dict):
            p, k = _number(people.get(g)), _number(yes.get(g))
            if p is not None and k is not None:
                label = f"{g}: {_short(k, sign)} {w['of']} {_short(p, sign)}"
        panel.text(0.005 * top, y + 0.8, label, "ink", anchor="start")   # Panel.text escapes it
        panel.rect(0, y + 0.1, x, y + 0.5, role="data", solid=True)
        panel.text(x + 0.015 * top, y + 0.3, _short(x, sign), "ink-2", anchor="start")
        shown.append(f"{label} {_short(x, sign)}")
    note = w["key"]
    if n >= 2 and max(values.values()) > 0:
        high, low = max(values.values()), min(values.values())
        note += " " + w["between"].format(diff=_short(high - low, sign), ratio=_short(low / high, sign))
    svg = chart.figure([panel], alt=w["alt"].format(shown="; ".join(shown)), width=520, height=max(150, 55 * n + 80))
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


_GAINS_WORDS = {
    "en": {"title": "Your gains and your spread", "xlabel": "gain: with reweighing minus without",
           "spread": "your spread {x}",
           "key": "A dot per sample at its gain in your list gain; ringed, with a dashed line: the smallest and the largest. The bar on top starts at the smallest gain and is as long as your spread.",
           "alt": "The gains of {n} samples: {shown}. Smallest {low}, largest {high}. Your spread: {mine}."},
    "nl": {"title": "Jouw toenames en jouw spreiding", "xlabel": "toename: met reweighing min zonder",
           "spread": "jouw spread {x}",
           "key": "Een stip per steekproef bij de toename in je lijst gain; met een ring en een gestippelde lijn: de kleinste en de grootste. De balk bovenaan begint bij de kleinste toename en is zo lang als je spread.",
           "alt": "De toenames van {n} steekproeven: {shown}. Kleinste {low}, grootste {high}. Jouw spread: {mine}."},
}


def _gains_picture(seen) -> str:
    """A dot per sample at the learner's gain, the smallest and the largest ringed, and on top a bar from the smallest
    gain as long as the learner's spread."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    gain = names.get("gain")
    if not isinstance(gain, list) or not 1 <= len(gain) <= 12:
        return ""
    values = [_number(x) for x in gain]
    if any(v is None or abs(v) > 100 for v in values):
        return ""
    mine_spread = _number(names.get("spread"))
    if mine_spread is not None and abs(mine_spread) > 100:
        mine_spread = None
    w = _GAINS_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    low, high = min(values), max(values)
    ends = values + [0.0] + ([low + mine_spread] if mine_spread is not None else [])
    lo, hi = min(ends), max(ends)
    pad = 0.08 * (hi - lo) if hi > lo else 0.1
    n = len(values)
    ticks = _ticks(lo, hi)
    panel = chart.Panel(x=(lo - pad, hi + pad), y=(0, n + 1.6), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    for x in sorted({low, high}):
        panel.line([x, x], [0.1, n], role="ink-3", width=1.2, dashed=True)
    for i, v in enumerate(values):
        y = n - i - 0.5
        panel.points([v], [y], role="data", r=5)
        if v in (low, high):
            panel.points([v], [y], role="ink", r=9, hollow=True)
        step = 0.03 * (hi - lo + 2 * pad)   # the number beside the dot, on the side with room
        if v > (lo + hi) / 2:
            panel.text(v - step, y, _short(v, sign), "ink-2", anchor="end", halo=True)
        else:
            panel.text(v + step, y, _short(v, sign), "ink-2", anchor="start", halo=True)
    if mine_spread is not None:
        end = low + mine_spread
        panel.line([low, end], [n + 0.55, n + 0.55], role="ink", width=3)
        panel.text((low + end) / 2, n + 1.15, w["spread"].format(x=_short(mine_spread, sign)), "ink", anchor="middle",
                   halo=True)
    alt = w["alt"].format(n=n, shown=", ".join(_short(v, sign) for v in values), low=_short(low, sign),
                          high=_short(high, sign), mine=_short(mine_spread, sign) if mine_spread is not None else "–")
    svg = chart.figure([panel], alt=alt, width=520, height=max(180, 26 * n + 110))
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{w["key"]}</div>')


# The lesson is done when these are correct.
REQUIRED = ["rates", "spread", "german"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_SEX_LINE = 'sex = ["Female", "Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male"]'
_PRED_LINE = "y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]"
_RATES_HEAD = ("# the example of Fairlearn's MetricFrame: ten people, their sex and the model's answer, 1 for a yes\n"
               + _SEX_LINE + "\n" + _PRED_LINE + "\n"
               + '\npeople = {"Female": 0, "Male": 0}\nyes = {"Female": 0, "Male": 0}\nfor i in range(len(sex)):\n')
_RATES_TAIL = 'rate = {"Female": yes["Female"] / people["Female"], "Male": yes["Male"] / people["Male"]}\nrate\n'
_RATES_START = _RATES_HEAD + "    ...\n" + _RATES_TAIL
_RATES_BODY = ("    people[sex[i]] = people[sex[i]] + 1\n"
               "    yes[sex[i]] = yes[sex[i]] + y_pred[i]\n")
_RATES_SOLUTION = _RATES_HEAD + _RATES_BODY + _RATES_TAIL


def _r(body: str) -> str:
    return _RATES_HEAD + body + _RATES_TAIL


def _rswap(old: str, new: str) -> str:
    assert _RATES_BODY.count(old) == 1, old
    return _r(_RATES_BODY.replace(old, new))


_PLAIN_LINE = "plain = [0.45, 0.36, 0.24, 0.56, 0.32, 0.55, 0.56, 0.31, 0.34, 0.22]"
_REWEIGHED_LINE = "reweighed = [0.72, 0.57, 0.46, 0.63, 0.64, 0.78, 0.75, 0.82, 0.6, 0.67]"
_SPREAD_HEAD = ("# the demo's ten samples: the women's yes rate divided by the men's, without and with reweighing\n"
                + _PLAIN_LINE + "\n" + _REWEIGHED_LINE + "\n"
                + "\ngain = []\nfor i in range(len(plain)):\n")
_SPREAD_START = _SPREAD_HEAD + "    ...\nspread = ...\nspread\n"
_SPREAD_SOLUTION = _SPREAD_HEAD + "    gain.append(reweighed[i] - plain[i])\nspread = max(gain) - min(gain)\nspread\n"


def _s(body: str, last: str = "spread = max(gain) - min(gain)\n") -> str:
    return _SPREAD_HEAD + body + last + "spread\n"


_GAIN_LINE = "    gain.append(reweighed[i] - plain[i])\n"

# The check also runs the boxes with other numbers.
_OTHER_SEX = ["Female", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", "Female"]
_OTHER_PRED = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
_OTHER_PLAIN = [0.3, 0.5, 0.2, 0.4]
_OTHER_REWEIGHED = [0.5, 0.6, 0.8, 0.45]
# The spread when each gain is a division instead of a difference.
_DIVIDED = max(r / p for p, r in zip(PLAIN, REWEIGHED)) - min(r / p for p, r in zip(PLAIN, REWEIGHED))

EXERCISES = {
    "rates": CodeBox(
        name="rates",
        picture=_rates_picture,
        start=_RATES_START,
        solution=_RATES_SOLUTION,
        keep=[_SEX_LINE, _PRED_LINE],
        compare=_same_rates,
        digits=3,
        others=[{"sex": _OTHER_SEX}, {"y_pred": _OTHER_PRED}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_people`. In the box the names have none: `sex`, `y_pred`, `people`, `yes`, `i`, `rate`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_people`. In het vak hebben de namen dat niet: `sex`, `y_pred`, `people`, `yes`, `i`, `rate`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `people` or `yes` for a key it does not have. Their keys are the sexes, `\"Female\"` and `\"Male\"`: the sex of person `i` is `sex[i]`, so use `people[sex[i]]`, without quotes around `sex[i]`.",
                    nl="Je code vraagt `people` of `yes` om een sleutel die het niet heeft. De sleutels zijn de geslachten, `\"Female\"` en `\"Male\"`: het geslacht van persoon `i` is `sex[i]`, dus gebruik `people[sex[i]]`, zonder aanhalingstekens om `sex[i]`."),
            Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by 0: `people` still holds 0 for a sex when `rate` is worked out. Inside the loop, add 1 for each person to the count of their sex: `people[sex[i]] = people[sex[i]] + 1`.",
                    nl="Je code deelt door 0: `people` heeft voor een geslacht nog 0 als `rate` wordt uitgerekend. Tel in de loop voor elke persoon 1 op bij het aantal van het eigen geslacht: `people[sex[i]] = people[sex[i]] + 1`."),
            Mistake("whole-sex", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="Your code uses the whole list `sex` as a key. The sex of person `i` is `sex[i]`: `people[sex[i]]`.",
                    nl="Je code gebruikt de hele lijst `sex` als sleutel. Het geslacht van persoon `i` is `sex[i]`: `people[sex[i]]`."),
            Mistake("whole-list", error=lambda e: isinstance(e, TypeError) and ("unsupported operand" in str(e) or "can only concatenate" in str(e)),
                    en="Your code adds something that is not a number, such as the whole list `y_pred`. Add the answer of one person: `y_pred[i]`.",
                    nl="Je code telt iets op dat geen getal is, zoals de hele lijst `y_pred`. Tel het antwoord van één persoon op: `y_pred[i]`."),
            Mistake("no-yes", matches=_all_rates(0),
                    en="Your rates are all 0: `yes` stays 0. Inside the loop, also add the person's answer to the yeses of their sex: `yes[sex[i]] = yes[sex[i]] + y_pred[i]`.",
                    nl="Je rates zijn allemaal 0: `yes` blijft 0. Tel in de loop ook het antwoord van de persoon op bij de ja's van het eigen geslacht: `yes[sex[i]] = yes[sex[i]] + y_pred[i]`."),
            Mistake("every-yes", matches=_all_rates(1),
                    en="Your rates are all 1: `people` and `yes` count the same thing. `people` adds 1 for each person, `yes` adds the answer `y_pred[i]`, which is 1 for a yes and 0 for a no.",
                    nl="Je rates zijn allemaal 1: `people` en `yes` tellen hetzelfde. `people` telt 1 op voor elke persoon, `yes` telt het antwoord `y_pred[i]` op, dat 1 is bij een ja en 0 bij een nee."),
            Mistake("swapped", matches=_above_one,
                    en="A rate above 1 cannot be a share. Then `people` and `yes` are swapped: `people` counts 1 per person, `yes` adds the answer `y_pred[i]`.",
                    nl="Een rate boven 1 kan geen deel zijn. Dan zijn `people` en `yes` verwisseld: `people` telt 1 per persoon, `yes` telt het antwoord `y_pred[i]` op."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines, as in the worked example: add 1 to the count in `people` of this person's sex, and add their answer to the count in `yes`.",
                      "nl": "Vervang de `...` door twee regels, zoals in het uitgewerkte voorbeeld: tel 1 op bij het aantal in `people` van het geslacht van deze persoon, en tel het antwoord op bij het aantal in `yes`."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `sex` and `y_pred` exactly as they were: put them back. The check tries other people by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `sex` en `y_pred` nodig zoals ze waren: zet ze terug. De check probeert zelf andere mensen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `rate` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `rate` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Does `people` count 1 for each person of a sex, and does `yes` add up the answers `y_pred[i]` of that sex?",
                      "nl": "Je laatste regel geeft `{got}`. Telt `people` 1 voor elke persoon van een geslacht, en telt `yes` de antwoorden `y_pred[i]` van dat geslacht op?"},
            "other": {"en": "Right for these ten people. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you count with `sex[i]` and `y_pred[i]` themselves, and not with numbers of your own?",
                      "nl": "Goed voor deze tien mensen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je met `sex[i]` en `y_pred[i]` zelf, en niet met eigen getallen?"},
            "other_error": {"en": "Right for these ten people. But with {where}, your code stops with an error. Do you count with `sex[i]` and `y_pred[i]` themselves?",
                            "nl": "Goed voor deze tien mensen. Maar met {where} stopt je code met een fout. Tel je met `sex[i]` en `y_pred[i]` zelf?"},
            "correct": {"en": "Female 0.8 and Male 0.4: what Fairlearn's `MetricFrame` shows in `by_group` for this example. Its `difference()` gives 0.8 minus 0.4, that is 0.4, and its `ratio()` 0.4 divided by 0.8, that is 0.5.",
                        "nl": "Female 0,8 en Male 0,4: wat `MetricFrame` van Fairlearn bij dit voorbeeld in `by_group` laat zien. Zijn `difference()` geeft 0,8 min 0,4, dus 0,4, en zijn `ratio()` 0,4 gedeeld door 0,8, dus 0,5."},
        },
    ),
    "spread": CodeBox(
        name="spread",
        picture=_gains_picture,
        start=_SPREAD_START,
        solution=_SPREAD_SOLUTION,
        keep=[_PLAIN_LINE, _REWEIGHED_LINE],
        compare=_same_number,
        digits=3,
        others=[{"plain": _OTHER_PLAIN, "reweighed": _OTHER_REWEIGHED}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_gain`. In the box the names have none: `plain`, `reweighed`, `gain`, `i`, `spread`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_gain`. In het vak hebben de namen dat niet: `plain`, `reweighed`, `gain`, `i`, `spread`."),
            Mistake("one-gain", error=lambda e: isinstance(e, TypeError) and "iterable" in str(e),
                    en="`max` and `min` need a list, and your `gain` is one number. Keep `gain` a list, and add each gain to it: `gain.append(...)`.",
                    nl="`max` en `min` hebben een lijst nodig, en je `gain` is één getal. Laat `gain` een lijst blijven, en voeg elke toename eraan toe: `gain.append(...)`."),
            Mistake("zero", matches=_near(0),
                    en="Your spread is 0: `gain` holds only one gain, or the same gain each time. Add each sample's gain to the list with `gain.append(reweighed[i] - plain[i])`, inside the loop.",
                    nl="Je spread is 0: `gain` bevat maar één toename, of steeds dezelfde. Voeg de toename van elke steekproef aan de lijst toe met `gain.append(reweighed[i] - plain[i])`, in de loop."),
            Mistake("largest", matches=_near(0.51),
                    en="Your spread is the largest gain, 0.51. The spread is how far the largest and the smallest lie apart: `max(gain) - min(gain)`.",
                    nl="Je spread is de grootste toename, 0,51. De spread is hoe ver de grootste en de kleinste uit elkaar liggen: `max(gain) - min(gain)`."),
            Mistake("negative", matches=_near(-0.44),
                    en="Your spread is negative: you subtract the largest gain from the smallest. Turn it around: `max(gain) - min(gain)`.",
                    nl="Je spread is negatief: je trekt de grootste toename af van de kleinste. Draai het om: `max(gain) - min(gain)`."),
            Mistake("divided", matches=_near(_DIVIDED),
                    en="Your gains are divisions: how many times as large the ratio became. The gain here is a difference: `reweighed[i] - plain[i]`.",
                    nl="Je toenames zijn delingen: hoeveel keer zo groot de verhouding werd. De toename is hier een verschil: `reweighed[i] - plain[i]`."),
            Mistake("plain-only", matches=_near(max(PLAIN) - min(PLAIN)),
                    en="Your spread is that of `plain`, without weights: 0.56 minus 0.22. The exercise asks for the spread of the gains: per sample `reweighed[i] - plain[i]`, added to `gain`.",
                    nl="Je spread is die van `plain`, zonder gewichten: 0,56 min 0,22. De oefening vraagt de spread van de toenames: per steekproef `reweighed[i] - plain[i]`, toegevoegd aan `gain`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in the loop, add to `gain` how much reweighing raised sample `i`; after the loop, `spread` is the largest gain minus the smallest.",
                      "nl": "Vervang de twee `...`: voeg in de loop aan `gain` toe hoeveel reweighing steekproef `i` omhoog bracht; na de loop is `spread` de grootste toename min de kleinste."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `plain` and `reweighed` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `plain` en `reweighed` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `spread` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `spread` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Is each gain `reweighed[i] - plain[i]`, added to `gain` in the loop, and is `spread` the largest gain minus the smallest?",
                      "nl": "Je laatste regel geeft `{got}`. Is elke toename `reweighed[i] - plain[i]`, in de loop toegevoegd aan `gain`, en is `spread` de grootste toename min de kleinste?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work out the gains from `plain` and `reweighed` themselves, and the spread from `gain`?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je de toenames uit met `plain` en `reweighed` zelf, en de spread met `gain`?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `plain`, `reweighed` and `gain` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `plain`, `reweighed` en `gain` zelf?"},
            "correct": {"en": "A spread of 0.44: the smallest gain is 0.07, with seed 3, and the largest 0.51, with seed 7. The same method, on samples of the same size from the same table, raised the ratio by anything from 0.07 to 0.51. With one split you would have seen only one of these numbers.",
                        "nl": "Een spread van 0,44: de kleinste toename is 0,07, bij seed 3, en de grootste 0,51, bij seed 7. Dezelfde methode, op steekproeven van dezelfde grootte uit dezelfde tabel, bracht de verhouding 0,07 tot 0,51 omhoog. Met één splitsing had je maar één van die getallen gezien."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rates", _RATES_START, "empty", None),
    ("rates", _RATES_SOLUTION, "correct", None),
    ("rates", _r("    g = sex[i]\n    people[g] = people[g] + 1\n    yes[g] = yes[g] + y_pred[i]\n"), "correct", None),
    ("rates", _r("    people[sex[i]] = people[sex[i]] + 1\n    if y_pred[i] == 1:\n"
                 "        yes[sex[i]] = yes[sex[i]] + 1\n"), "correct", None),
    ("rates", _rswap("people[sex[i]] = people[sex[i]] + 1", "_people[sex[i]] = people[sex[i]] + 1"), "error",
     "underscore"),
    ("rates", _rswap("people[sex[i]] = people[sex[i]] + 1", 'people["sex"] = people["sex"] + 1'), "error", "key"),
    ("rates", _r("    yes[sex[i]] = yes[sex[i]] + y_pred[i]\n"), "error", "zero"),
    ("rates", _rswap("yes[sex[i]] + y_pred[i]", "yes[sex[i]] + y_pred"), "error", "whole-list"),
    ("rates", _rswap("people[sex[i]] = people[sex[i]] + 1", "people[sex] = people[sex] + 1"), "error", "whole-sex"),
    ("rates", _r("    people[sex[i]] = people[sex[i]] + 1\n"), "wrong", "no-yes"),
    ("rates", _rswap("yes[sex[i]] + y_pred[i]", "yes[sex[i]] + 1"), "wrong", "every-yes"),
    ("rates", _rswap("people[sex[i]] + 1", "people[sex[i]] + y_pred[i]"), "wrong", "every-yes"),
    ("rates", _r("    people[sex[i]] = people[sex[i]] + y_pred[i]\n    yes[sex[i]] = yes[sex[i]] + 1\n"), "wrong",
     "swapped"),
    ("rates", _rswap("people[sex[i]] + 1", "people[sex[i]] + 2"), "wrong", None),
    ("rates", _rswap("people[sex[i]] = people[sex[i]] + 1", "people[sex[i]] = 5"), "almost", "other-value"),
    ("rates", _RATES_SOLUTION.replace("y_pred = [0, 1,", "y_pred = [1, 1,"), "wrong", "changed-line"),
    ("rates", _r("    ...\n" + _RATES_BODY), "wrong", "dots-left"),
    ("spread", _SPREAD_START, "empty", None),
    ("spread", _SPREAD_SOLUTION, "correct", None),
    ("spread", _s("    gain.append(plain[i] - reweighed[i])\n"), "correct", None),
    ("spread", _s("    g = reweighed[i] - plain[i]\n    gain.append(g)\n",
                  "gain.sort()\nspread = gain[-1] - gain[0]\n"), "correct", None),
    ("spread", _s(_GAIN_LINE, "spread = max(_gain) - min(gain)\n"), "error", "underscore"),
    ("spread", _s("    gain = reweighed[i] - plain[i]\n"), "error", "one-gain"),
    ("spread", _s("    gain = [reweighed[i] - plain[i]]\n"), "wrong", "zero"),
    ("spread", _s(_GAIN_LINE, "spread = max(gain)\n"), "wrong", "largest"),
    ("spread", _s(_GAIN_LINE, "spread = min(gain) - max(gain)\n"), "wrong", "negative"),
    ("spread", _s("    gain.append(reweighed[i] / plain[i])\n"), "wrong", "divided"),
    ("spread", _s(_GAIN_LINE, "spread = max(plain) - min(plain)\n"), "wrong", "plain-only"),
    ("spread", _s("    gain.append(reweighed[i])\n"), "wrong", None),
    ("spread", _s(_GAIN_LINE, "spread = 0.51 - 0.07\n"), "almost", "other-value"),
    ("spread", _SPREAD_SOLUTION.replace("plain = [0.45,", "plain = [0.05,"), "wrong", "changed-line"),
    ("spread", _s("    ...\n" + _GAIN_LINE), "wrong", "dots-left"),
]
