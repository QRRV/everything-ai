# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-reweighing: the weight of each combination of sex and label in the training data of
UCI Adult, and the share that earns more than 50,000 dollars per sex once the weights count; with the picture of
the demo's yes rates and recall, and the pictures of the learner's weights and shares."""
import math
import numbers
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import yours

# The demo, measured on UCI Adult (data/adult.csv.gz) with the model of the previous lessons, without the
# column sex, trained without weights and with the weights of reweighing; tests/test_rai_reweighing.py runs it
# again.
ACCURACY = {"plain": 0.834, "reweighed": 0.829}
EFFECT = {
    "plain": {"yes": {"women": 0.0705, "men": 0.2479}, "recall": {"women": 0.4356, "men": 0.5697}},
    "reweighed": {"yes": {"women": 0.1164, "men": 0.2042}, "recall": {"women": 0.5831, "men": 0.4954}},
}

# The worked example: a made-up table of 10 people that gives the four weights of the example of Kamiran and
# Calders (2012).
TOY = {"women": {"yes": 2, "no": 3}, "men": {"yes": 4, "no": 1}}

# Exercises 1 and 2: the 32,561 people of UCI Adult to train on, per sex and label.
COUNTS = {"women": {">50K": 1179, "<=50K": 9592}, "men": {">50K": 6662, "<=50K": 15128}}
# Exercise 2 gives the weights of Exercise 1, rounded to three decimals.
ROUNDED = {"women": {">50K": 2.2, "<=50K": 0.853}, "men": {">50K": 0.788, "<=50K": 1.094}}


def totals(counts: dict) -> tuple:
    """Per group, per label and all together: how many people there are."""
    in_group = {g: sum(labels.values()) for g, labels in counts.items()}
    with_label = {}
    for labels in counts.values():
        for label, n in labels.items():
            with_label[label] = with_label.get(label, 0) + n
    return in_group, with_label, sum(in_group.values())


def weights(counts: dict) -> dict:
    """Per group and label: the count you would expect if the label did not depend on the group, divided by the
    count there is."""
    in_group, with_label, total = totals(counts)
    return {g: {l: in_group[g] * with_label[l] / (total * n) for l, n in labels.items()}
            for g, labels in counts.items()}


def shares(counts: dict, weight: dict, label: str = ">50K") -> dict:
    """Per group: the share with `label` once every person counts as their weight."""
    out = {}
    for g, labels in counts.items():
        everyone = sum(n * weight[g][l] for l, n in labels.items())
        out[g] = labels[label] * weight[g][label] / everyone
    return out


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _num(x: float, decimal: str, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", decimal)


def picture_effect(labels: dict) -> str:
    """Two panels with a row per model, without weights and with reweighing: the share with a yes (left) and the
    recall (right), for women (open bars) and men (filled bars). `labels` gives "title", "rows" (two names),
    "xlabels" (two), "key", "decimal" and "alt"."""
    from everything_ai import chart
    keys = list(EFFECT)
    top = 1.6 * len(keys) + 0.1
    panels = []
    for k, (measure, xlabel) in enumerate(zip(["yes", "recall"], labels["xlabels"])):
        p = chart.Panel(x=(0, 1), y=(0, top), xlabel=xlabel, yticks=[], xticks=[0, 0.5, 1],
                        xtick_labels=["0", "0.5", "1"])
        for i, (key, name) in enumerate(zip(keys, labels["rows"])):
            women, men = EFFECT[key][measure]["women"], EFFECT[key][measure]["men"]
            y = top - 1.6 * (i + 1)
            if k == 0:
                p.text(0.002, y + 1.2, name, "ink", anchor="start")
            p.rect(0, y + 0.55, women, y + 0.85, role="data")
            p.rect(0, y + 0.15, men, y + 0.45, role="data", solid=True)
            p.text(women + 0.015, y + 0.7, _num(women, labels["decimal"]), "ink-2", anchor="start")
            p.text(men + 0.015, y + 0.3, _num(men, labels["decimal"]), "ink-2", anchor="start")
        panels.append(p)
    svg = chart.figure(panels, alt=labels["alt"], width=300, height=230)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _flat(v, high: float = 1e9) -> dict | None:
    """The learner's weights as {(group, label): number}, when `v` is a dict of 1 to 4 dicts of 1 to 4 numbers from
    0 to `high`; else None."""
    if not isinstance(v, dict) or not 1 <= len(v) <= 4:
        return None
    out = {}
    for g, labels in v.items():
        if not isinstance(g, str) or not isinstance(labels, dict) or not 1 <= len(labels) <= 4:
            return None
        for l, x in labels.items():
            x = _number(x)
            if not isinstance(l, str) or x is None or not 0 <= x <= high:
                return None
            out[(g, l)] = x
    return out


def _shares(v) -> dict | None:
    """The learner's number per group, when it is a dict of 1 to 4 numbers; else None."""
    if not isinstance(v, dict) or not 1 <= len(v) <= 4:
        return None
    out = {}
    for k, x in v.items():
        x = _number(x)
        if not isinstance(k, str) or x is None:
            return None
        out[k] = x
    return out


def _close_weights(got, want) -> bool:
    """The same groups and labels, each weight the same to two decimals."""
    a, b = _flat(got), _flat(want)
    return a is not None and b is not None and set(a) == set(b) and all(abs(a[k] - b[k]) < 0.006 for k in a)


def _close_shares(got, want) -> bool:
    """A share per group, the same groups, each the same to two decimals."""
    a, b = _shares(got), _shares(want)
    return (a is not None and b is not None and set(a) == set(b)
            and all(0 <= a[k] <= 1 and abs(a[k] - b[k]) < 0.006 for k in a))


def _like_weights(make):
    """A mistake where every weight is what `make()` gives, to two decimals."""
    def match(got, text):
        a, b = _flat(got), make()
        return a is not None and set(a) == set(b) and all(abs(a[k] - b[k]) < 0.006 for k in a)
    return match


def _inverted() -> dict:
    w = weights(COUNTS)
    return {(g, l): 1 / w[g][l] for g in w for l in w[g]}


def _no_total() -> dict:
    in_group, with_label, total = totals(COUNTS)
    return {(g, l): in_group[g] * with_label[l] / n
            for g, labels in COUNTS.items() for l, n in labels.items()}


def _huge(got, text) -> bool:
    a = _flat(got)
    return a is not None and any(x > 100 for x in a.values())


def _numbers_per_group(got, text) -> bool:
    """`weight[group]` is one number instead of a dict per label."""
    return (isinstance(got, dict) and set(got) == set(COUNTS)
            and all(_number(x) is not None for x in got.values()))


def _labels_as_groups(got, text) -> bool:
    """The labels landed next to the groups: `weight[label] = ...`."""
    return isinstance(got, dict) and any(k in (">50K", "<=50K") for k in got)


def _empty_groups(got, text) -> bool:
    return (isinstance(got, dict) and set(got) == set(COUNTS)
            and all(isinstance(x, dict) and len(x) < 2 for x in got.values()))


def _share_like(values: dict):
    def match(got, text):
        a = _shares(got)
        return a is not None and set(a) == set(values) and all(abs(a[k] - values[k]) < 0.006 for k in a)
    return match


def _unweighted() -> dict:
    return {g: labels[">50K"] / sum(labels.values()) for g, labels in COUNTS.items()}


def _other_label() -> dict:
    return shares(COUNTS, ROUNDED, "<=50K")


def _above_one(got, text) -> bool:
    a = _shares(got)
    return a is not None and any(x > 1 for x in a.values())


def _fewer(keys):
    def match(got, text):
        return isinstance(got, dict) and 1 <= len(got) < len(keys) and set(got) <= set(keys)
    return match


def _other(keys):
    def match(got, text):
        return isinstance(got, dict) and len(got) >= 1 and not set(got) <= set(keys)
    return match


_WEIGHT_WORDS = {
    "en": {"title": "Your weights: how many people each person counts for", "xlabel": "weight",
           "one": "1: counts once", "women": "women", "men": "men",
           "alt": "One bar per sex and label, as long as its weight, with a dashed line at 1: {shown}."},
    "nl": {"title": "Jouw gewichten: voor hoeveel mensen elke persoon telt", "xlabel": "gewicht",
           "one": "1: telt één keer", "women": "vrouwen", "men": "mannen",
           "alt": "Eén balk per geslacht en label, zo lang als het gewicht, met een gestippelde lijn bij 1: {shown}."},
}


def _weights_picture(seen) -> str:
    """Per group and label a bar as long as the learner's weight, with a dashed line at 1."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _flat(names.get("weight"), 20)
    if value is None:
        return ""
    w = _WEIGHT_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    keys = list(value)
    xmax = max(3, math.ceil(max(value.values()) + 0.2))
    ticks = list(range(0, xmax + 1))
    top = 1.2 * len(keys)
    panel = chart.Panel(x=(0, xmax), y=(0, top + 0.6), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[str(t) for t in ticks])
    panel.line([1, 1], [0, top + 0.6], role="ink-3", width=1, dashed=True)
    panel.text(1 + 0.02 * xmax, top + 0.3, w["one"], "ink-3", anchor="start", halo=True)
    shown = []
    for i, (g, l) in enumerate(keys):
        y = top - 1.2 * (i + 1)
        name = w.get(g, g) + " " + l
        panel.text(0.005 * xmax, y + 0.72, name + ": " + _num(value[(g, l)], sign), "ink", anchor="start",
                   halo=True)   # Panel.text escapes it
        panel.rect(0, y + 0.15, value[(g, l)], y + 0.45, role="data", solid=True)
        shown.append(f"{name} {_num(value[(g, l)], sign)}")
    svg = chart.figure([panel], alt=w["alt"].format(shown="; ".join(shown)), width=520,
                       height=max(170, 44 * len(keys) + 90))
    return yours.figure(w["title"], chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


_SHARE_WORDS = {
    "en": {"title": "Your shares that earn more, with the weights", "xlabel": "share that earns more than 50,000 dollars",
           "before": "without weights", "women": "women", "men": "men",
           "alt": "Per sex two bars: the share that earns more without weights, open, and with the weights, filled: {shown}."},
    "nl": {"title": "Jouw delen die meer verdienen, met de gewichten", "xlabel": "deel dat meer dan 50.000 dollar verdient",
           "before": "zonder gewichten", "women": "vrouwen", "men": "mannen",
           "alt": "Per geslacht twee balken: het deel dat meer verdient zonder gewichten, open, en met de gewichten, gevuld: {shown}."},
}


def _shares_picture(seen) -> str:
    """Per group two bars: the share that earns more without weights (open) and the learner's share with the weights
    (filled)."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _shares(names.get("share"))
    counts = names.get("counts")
    if value is None or any(not 0 <= x <= 1 for x in value.values()) or _flat(counts) is None:
        return ""
    w = _SHARE_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    plain = {}
    for g in value:
        labels = counts.get(g) if isinstance(counts.get(g), dict) else {}
        n = sum(x for x in labels.values() if _number(x) is not None)
        plain[g] = labels[">50K"] / n if n and _number(labels.get(">50K")) is not None else None
    keys = list(value)
    top = 1.6 * len(keys) + 0.1
    panel = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=[0, 0.25, 0.5, 0.75, 1],
                        xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    shown = []
    for i, g in enumerate(keys):
        y = top - 1.6 * (i + 1)
        panel.text(0.002, y + 1.2, w.get(g, g), "ink", anchor="start", halo=True)   # Panel.text escapes it
        if plain[g] is not None:
            panel.rect(0, y + 0.55, plain[g], y + 0.85, role="data")
            panel.text(plain[g] + 0.015, y + 0.7, _num(plain[g], sign) + " " + w["before"], "ink-2", anchor="start")
        panel.rect(0, y + 0.15, value[g], y + 0.45, role="data", solid=True)
        panel.text(value[g] + 0.015, y + 0.3, _num(value[g], sign), "ink-2", anchor="start")
        shown.append(f"{w.get(g, g)} {_num(value[g], sign)}")
    svg = chart.figure([panel], alt=w["alt"].format(shown="; ".join(shown)), width=520,
                       height=max(170, 56 * len(keys) + 90))
    return yours.figure(w["title"], chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


# The lesson is done when these are correct.
REQUIRED = ["weights", "shares", "needs"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_COUNTS_LINE = 'counts = {"women": {">50K": 1179, "<=50K": 9592}, "men": {">50K": 6662, "<=50K": 15128}}'
_TOTALS_LINES = ('in_group = {"women": 10771, "men": 21790}\n'
                 'with_label = {">50K": 7841, "<=50K": 24720}\n'
                 'total = 32561\n')
_WEIGHTS_HEAD = ("# UCI Adult, the 32,561 people to train on: per sex, how many earn more than 50,000 dollars and how many not\n"
                 + _COUNTS_LINE + "\n" + _TOTALS_LINES
                 + "\nweight = {}\nfor group in counts:\n    weight[group] = {}\n    for label in counts[group]:\n")
_WEIGHTS_START = _WEIGHTS_HEAD + "        ...\nweight\n"
_WEIGHTS_BODY = ("        expected = in_group[group] * with_label[label] / total\n"
                 "        weight[group][label] = expected / counts[group][label]\n")
_WEIGHTS_SOLUTION = _WEIGHTS_HEAD + _WEIGHTS_BODY + "weight\n"


def _w(body: str, last: str = "weight\n") -> str:
    return _WEIGHTS_HEAD + body + last


def _wswap(old: str, new: str) -> str:
    assert _WEIGHTS_BODY.count(old) == 1, old
    return _w(_WEIGHTS_BODY.replace(old, new))


_WEIGHT_LINE = 'weight = {"women": {">50K": 2.2, "<=50K": 0.853}, "men": {">50K": 0.788, "<=50K": 1.094}}'
_SHARES_HEAD = ("# UCI Adult, the 32,561 people to train on, per sex and label\n" + _COUNTS_LINE + "\n"
                "# the weights of Exercise 1, rounded\n" + _WEIGHT_LINE + "\n"
                "\nshare = {}\nfor group in counts:\n")
_SHARES_START = _SHARES_HEAD + "    ...\n    share[group] = above / everyone\nshare\n"
_SHARES_BODY = ('    above = counts[group][">50K"] * weight[group][">50K"]\n'
                '    everyone = above + counts[group]["<=50K"] * weight[group]["<=50K"]\n')
_SHARES_TAIL = "    share[group] = above / everyone\nshare\n"
_SHARES_SOLUTION = _SHARES_HEAD + _SHARES_BODY + _SHARES_TAIL


def _s(body: str, tail: str = _SHARES_TAIL) -> str:
    return _SHARES_HEAD + body + tail


def _sswap(old: str, new: str) -> str:
    assert _SHARES_BODY.count(old) == 1, old
    return _s(_SHARES_BODY.replace(old, new))


_OTHER_COUNTS = {"women": {">50K": 300, "<=50K": 700}, "men": {">50K": 500, "<=50K": 500}}
_OTHER_TOTALS = totals(_OTHER_COUNTS)

EXERCISES = {
    "weights": CodeBox(
        name="weights",
        picture=_weights_picture,
        start=_WEIGHTS_START,
        solution=_WEIGHTS_SOLUTION,
        keep=[_COUNTS_LINE, 'in_group = {"women": 10771, "men": 21790}', 'with_label = {">50K": 7841, "<=50K": 24720}',
              "total = 32561"],
        compare=_close_weights,
        digits=2,
        # The check also runs the box with a made-up table of 2,000 people.
        others=[{"counts": _OTHER_COUNTS, "in_group": _OTHER_TOTALS[0], "with_label": _OTHER_TOTALS[1],
                 "total": _OTHER_TOTALS[2]}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_total`. In the box the names have none: `counts`, `in_group`, `with_label`, `total`, `weight`, `group`, `label`, `expected`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_total`. In het vak hebben de namen dat niet: `counts`, `in_group`, `with_label`, `total`, `weight`, `group`, `label`, `expected`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a dict for a key it does not have. `in_group` has the sexes as keys, `with_label` the labels, and `counts[group]` the labels of one sex: `in_group[group]`, `with_label[label]`, `counts[group][label]`.",
                    nl="Je code vraagt een dict om een sleutel die het niet heeft. `in_group` heeft de geslachten als sleutels, `with_label` de labels, en `counts[group]` de labels van één geslacht: `in_group[group]`, `with_label[label]`, `counts[group][label]`."),
            Mistake("one-number", matches=_numbers_per_group,
                    en="Your `weight[group]` is one number, not a dict with a weight per label. Write `weight[group][label] = ...`, so that each sex gets a weight for each label.",
                    nl="Je `weight[group]` is één getal, geen dict met een gewicht per label. Schrijf `weight[group][label] = ...`, zodat elk geslacht een gewicht per label krijgt."),
            Mistake("labels-as-groups", matches=_labels_as_groups,
                    en="Your `weight` has the labels next to the sexes. Write `weight[group][label] = ...`, so that each label lands in the dict of its sex.",
                    nl="Je `weight` heeft de labels naast de geslachten. Schrijf `weight[group][label] = ...`, zodat elk label in de dict van het eigen geslacht komt."),
            Mistake("empty", matches=_empty_groups,
                    en="The dicts in your `weight` miss labels. Does the line `weight[group][label] = ...` stand inside the inner loop, indented by eight spaces?",
                    nl="De dicts in je `weight` missen labels. Staat de regel `weight[group][label] = ...` in de binnenste loop, acht spaties ingesprongen?"),
            Mistake("huge", matches=_huge,
                    en="Your weights are far too large. Do you divide by `total`? `expected` is how many people you would expect in this cell: `in_group[group] * with_label[label] / total`.",
                    nl="Je gewichten zijn veel te groot. Deel je door `total`? `expected` is hoeveel mensen je in dit vakje zou verwachten: `in_group[group] * with_label[label] / total`."),
            Mistake("inverted", matches=_like_weights(_inverted),
                    en="Your weights are upside down: the women who earn more get 0.45, while there are fewer of them than expected. Divide what you expect by what there is: `expected / counts[group][label]`.",
                    nl="Je gewichten staan op hun kop: de vrouwen die meer verdienen krijgen 0,45, terwijl er minder van zijn dan verwacht. Deel wat je verwacht door wat er is: `expected / counts[group][label]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the inner loop with two lines: `expected`, how many people you would expect in this cell, and `weight[group][label]`, that number divided by how many there are.",
                      "nl": "Vervang de `...` in de binnenste loop door twee regels: `expected`, hoeveel mensen je in dit vakje zou verwachten, en `weight[group][label]`, dat getal gedeeld door hoeveel er zijn."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `counts`, `in_group`, `with_label` and `total` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `counts`, `in_group`, `with_label` en `total` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `weight` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `weight` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. For each sex and label: is `expected` the size of the sex times the size of the label, divided by `total`, and is the weight `expected` divided by `counts[group][label]`?",
                      "nl": "Je laatste regel geeft `{got}`. Is voor elk geslacht en label `expected` de grootte van het geslacht keer de grootte van het label, gedeeld door `total`, en is het gewicht `expected` gedeeld door `counts[group][label]`?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `counts`, `in_group`, `with_label` and `total` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `counts`, `in_group`, `with_label` en `total` zelf, en niet de getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `counts`, `in_group`, `with_label` and `total` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `counts`, `in_group`, `with_label` en `total` zelf?"},
            "correct": {"en": "Each woman who earns more counts as 2.20 people, each man who earns more as 0.79; the women who earn less count as 0.85 and the men who earn less as 1.09. There are fewer women who earn more than you would expect if sex told you nothing, so they count more.",
                        "nl": "Elke vrouw die meer verdient telt als 2,20 mensen, elke man die meer verdient als 0,79; de vrouwen die minder verdienen tellen als 0,85 en de mannen die minder verdienen als 1,09. Er zijn minder vrouwen die meer verdienen dan je zou verwachten als het geslacht niets zei, dus tellen ze zwaarder."},
        },
    ),
    "shares": CodeBox(
        name="shares",
        picture=_shares_picture,
        start=_SHARES_START,
        solution=_SHARES_SOLUTION,
        keep=[_COUNTS_LINE, _WEIGHT_LINE],
        compare=_close_shares,
        digits=2,
        # The check also runs the box with other weights.
        others=[{"weight": {"women": {">50K": 2, "<=50K": 1}, "men": {">50K": 1, "<=50K": 1}}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_above`. In the box the names have none: `counts`, `weight`, `share`, `group`, `above`, `everyone`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_above`. In het vak hebben de namen dat niet: `counts`, `weight`, `share`, `group`, `above`, `everyone`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a dict for a key it does not have. The labels are `\">50K\"` and `\"<=50K\"`, written exactly like that, with quotes; `group` itself goes without quotes.",
                    nl="Je code vraagt een dict om een sleutel die het niet heeft. De labels zijn `\">50K\"` en `\"<=50K\"`, precies zo geschreven, met aanhalingstekens; `group` zelf staat zonder aanhalingstekens."),
            Mistake("nobody", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by 0: `everyone` is 0. It should be the weighted count of everyone in the sex: `above` plus the people who earn less, times their weight.",
                    nl="Je code deelt door 0: `everyone` is 0. Het moet het gewogen aantal van iedereen van dat geslacht zijn: `above` plus de mensen die minder verdienen, keer hun gewicht."),
            Mistake("unweighted", matches=_share_like(_unweighted()),
                    en="Your shares are 0.11 and 0.31, the shares without weights. Multiply each count by its weight: `counts[group][\">50K\"] * weight[group][\">50K\"]`.",
                    nl="Je delen zijn 0,11 en 0,31, de delen zonder gewichten. Vermenigvuldig elk aantal met het gewicht: `counts[group][\">50K\"] * weight[group][\">50K\"]`."),
            Mistake("other-label", matches=_share_like(_other_label()),
                    en="Your shares are of the people who earn less: 0.76. `above` should count the people with `\">50K\"`.",
                    nl="Je delen gaan over de mensen die minder verdienen: 0,76. `above` moet de mensen met `\">50K\"` tellen."),
            Mistake("above-one", matches=_above_one,
                    en="A share above 1 cannot be. Is `everyone` the people who earn more plus the people who earn less, each times their weight, and do you divide `above` by it?",
                    nl="Een deel boven 1 kan niet. Is `everyone` de mensen die meer verdienen plus de mensen die minder verdienen, elk keer hun gewicht, en deel je `above` erdoor?"),
            Mistake("one-group", matches=_fewer(COUNTS),
                    en="Your `share` has fewer sexes than `counts`. Does `share[group] = above / everyone` still stand inside the loop, indented by four spaces?",
                    nl="Je `share` heeft minder geslachten dan `counts`. Staat `share[group] = above / everyone` nog in de loop, vier spaties ingesprongen?"),
            Mistake("other-keys", matches=_other(COUNTS),
                    en="Your `share` has other keys than `counts`. Keep the last line of the loop as it was: `share[group] = above / everyone`.",
                    nl="Je `share` heeft andere sleutels dan `counts`. Laat de laatste regel van de loop zoals hij was: `share[group] = above / everyone`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the loop with two lines: `above`, the weighted count of the people who earn more, and `everyone`, the weighted count of everyone in the sex.",
                      "nl": "Vervang de `...` in de loop door twee regels: `above`, het gewogen aantal mensen dat meer verdient, en `everyone`, het gewogen aantal van iedereen van dat geslacht."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `counts` and `weight` exactly as they were: put them back. The check tries other weights by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `counts` en `weight` nodig zoals ze waren: zet ze terug. De check probeert zelf andere gewichten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Is `above` the count of `\">50K\"` times its weight, and `everyone` that plus the count of `\"<=50K\"` times its weight?",
                      "nl": "Je laatste regel geeft `{got}`. Is `above` het aantal met `\">50K\"` keer het gewicht, en `everyone` dat plus het aantal met `\"<=50K\"` keer het gewicht?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `weight` itself, also for `everyone`, and not its numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `weight` zelf, ook voor `everyone`, en niet de getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `counts` and `weight` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `counts` en `weight` zelf?"},
            "correct": {"en": "With the weights, 0.24 of the women and 0.24 of the men earn more than 50,000 dollars, the share of all people together. In the weighted data, sex no longer tells you anything about the label: what Kamiran and Calders were after.",
                        "nl": "Met de gewichten verdient 0,24 van de vrouwen en 0,24 van de mannen meer dan 50.000 dollar, het deel van alle mensen samen. In de gewogen data zegt het geslacht niets meer over het label: waar Kamiran en Calders op uit waren."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("weights", _WEIGHTS_START, "empty", None),
    ("weights", _WEIGHTS_SOLUTION, "correct", None),
    ("weights", _w("        weight[group][label] = in_group[group] * with_label[label] / (total * counts[group][label])\n"),
     "correct", None),
    ("weights", _wswap("expected / counts[group][label]", "round(expected / counts[group][label], 2)"), "correct", None),
    ("weights", _wswap("expected = in_group", "expected = _in_group"), "error", "underscore"),
    ("weights", _wswap("with_label[label]", "with_label[group]"), "error", "key"),
    ("weights", _wswap("weight[group][label] =", "weight[group] ="), "wrong", "one-number"),
    ("weights", _wswap("weight[group][label] =", "weight[label] ="), "wrong", "labels-as-groups"),
    ("weights", _w(_WEIGHTS_BODY.replace("        weight[group][label]", "    weight[group][label]")), "wrong", "empty"),
    ("weights", _w("        expected = in_group[group] * with_label[label] / total\n"
                   "weight[group][label] = expected / counts[group][label]\n"), "wrong", "empty"),
    ("weights", _wswap(" / total\n", "\n"), "wrong", "huge"),
    ("weights", _wswap("expected / counts[group][label]", "counts[group][label] / expected"), "wrong", "inverted"),
    ("weights", _wswap("expected / counts[group][label]", "expected / in_group[group]"), "wrong", None),
    ("weights", _wswap(" / total\n", " / 32561\n"), "almost", "other-value"),
    ("weights", _WEIGHTS_SOLUTION.replace("total = 32561", "total = 30000"), "wrong", "changed-line"),
    ("weights", _w("        ...\n        weight[group][label] = 1\n"), "wrong", "dots-left"),
    ("shares", _SHARES_START, "empty", None),
    ("shares", _SHARES_SOLUTION, "correct", None),
    ("shares", _s('    above = counts[group][">50K"] * weight[group][">50K"]\n'
                  '    below = counts[group]["<=50K"] * weight[group]["<=50K"]\n'
                  '    everyone = above + below\n'), "correct", None),
    ("shares", _sswap("above = counts", "above = _counts"), "error", "underscore"),
    ("shares", _sswap('counts[group][">50K"] * weight', 'counts[group][">50"] * weight'), "error", "key"),
    ("shares", _sswap('above + counts[group]["<=50K"] * weight[group]["<=50K"]', "0"), "error", "nobody"),
    ("shares", _s('    above = counts[group][">50K"]\n    everyone = above + counts[group]["<=50K"]\n'), "wrong",
     "unweighted"),
    ("shares", _s('    above = counts[group]["<=50K"] * weight[group]["<=50K"]\n'
                  '    everyone = above + counts[group][">50K"] * weight[group][">50K"]\n'), "wrong", "other-label"),
    ("shares", _sswap('above + counts[group]["<=50K"] * weight[group]["<=50K"]', 'above / 2'), "wrong", "above-one"),
    ("shares", _s(_SHARES_BODY, "share[group] = above / everyone\nshare\n"), "wrong", "one-group"),
    ("shares", _s(_SHARES_BODY, '    share["group"] = above / everyone\nshare\n'), "wrong", "other-keys"),
    ("shares", _sswap('above + counts[group]["<=50K"] * weight[group]["<=50K"]',
                      'above + counts[group]["<=50K"]'), "wrong", None),
    ("shares", _sswap('above + counts[group]["<=50K"] * weight[group]["<=50K"]',
                      'counts[group][">50K"] + counts[group]["<=50K"]'), "almost", "other-value"),
    ("shares", _SHARES_SOLUTION.replace(_WEIGHT_LINE, 'weight = {"women": {">50K": 1, "<=50K": 1}, "men": {">50K": 1, "<=50K": 1}}'),
     "wrong", "changed-line"),
    ("shares", _s("    ...\n    above = 1\n    everyone = 4\n"), "wrong", "dots-left"),
]
