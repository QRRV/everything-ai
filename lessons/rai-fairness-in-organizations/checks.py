# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-fairness-in-organizations: the stages of the AI Fairness Checklist of Madaio and colleagues
(2020) in which a team's plan has no moment to look at fairness, and the groups whose word error rate lies more than an
agreed limit above the best group's, from Koenecke and colleagues (2020); with the picture of each."""
import math
import numbers
import re

from everything_ai import yours
from everything_ai.check import CodeBox, Mistake

# --- the stages and a team's plan ------------------------------------------------------------------------------------
# The six stages of the AI Fairness Checklist (Madaio, Stark, Wortman Vaughan and Wallach 2020), in order.
STAGES = ["envision", "define", "prototype", "build", "launch", "evolve"]
# A team's plan, made up by the course: the moments at which it looks at fairness, with the stage and who leads it.
# The moments are among those the checklist names for each stage.
PLAN = [
    {"moment": "pre-mortem screening", "stage": "envision", "lead": "product manager"},
    {"moment": "code review", "stage": "build", "lead": "engineer"},
    {"moment": "ship review", "stage": "launch", "lead": "director"},
]
# A second plan that the check tries.
OTHER_PLAN = [
    {"moment": "design review", "stage": "define", "lead": "designer"},
    {"moment": "product review", "stage": "evolve", "lead": "product manager"},
]


def missing(stages: list, moments: list) -> list:
    """The stages, in order, in which no moment of the plan falls."""
    planned = [m["stage"] for m in moments]
    return [s for s in stages if s not in planned]


# --- the groups behind -----------------------------------------------------------------------------------------------
# Koenecke and colleagues (2020): the average word error rate of black men, black women, white men and white women.
WER = {"black men": 0.41, "black women": 0.30, "white men": 0.21, "white women": 0.17}


def behind(wer: dict, limit: float) -> list:
    """The groups whose WER is more than `limit` above the lowest WER, in the order of the dict."""
    best = min(wer.values())
    return [g for g, v in wer.items() if v > best + limit]


# --- pictures ---------------------------------------------------------------------------------------------------------
def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _show(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _list_text(items: list, lang: str) -> str:
    if not items:
        return "—"
    return ", ".join(str(i) for i in items)


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _plan_ok(stages, moments) -> bool:
    """The learner's stages and moments, when they still look like the lesson's."""
    if not isinstance(stages, list) or not 1 <= len(stages) <= 10 or not all(isinstance(s, str) for s in stages):
        return False
    if not isinstance(moments, list) or len(moments) > 20:
        return False
    return all(type(m) is dict and isinstance(m.get("stage"), str) for m in moments)


_MISSING_WORDS = {
    "en": {"title": "Your stages without a moment: {shown}", "xlabel": "the stages, in order",
           "key": "filled: the plan has a moment in this stage; open: it has none; purple bar: in your list",
           "alt": "One box per stage: {shown}. Filled boxes have a moment in the plan, open boxes have none. Your list: {mine}."},
    "nl": {"title": "Jouw fasen zonder moment: {shown}", "xlabel": "de fasen, op volgorde",
           "key": "gevuld: het plan heeft een moment in deze fase; open: het heeft er geen; paarse streep: in jouw lijst",
           "alt": "Eén vak per fase: {shown}. Gevulde vakken hebben een moment in het plan, open vakken niet. Jouw lijst: {mine}."},
}


def _missing_picture(seen) -> str:
    """A box per stage, filled when the plan has a moment in it and open when it has none: the open boxes are the
    stages the learner's list should hold. A purple bar under a stage marks it as one in the learner's list."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    stages, moments = names.get("stages"), names.get("moments")
    if not _plan_ok(stages, moments):
        return ""
    w = _MISSING_WORDS[seen.lang]
    planned = [m["stage"] for m in moments]
    n = len(stages)
    p = chart.Panel(x=(0, n), y=(0, 2), xlabel=w["xlabel"], yticks=[], xticks=[], xtick_labels=[])
    for i, stage in enumerate(stages):
        p.rect(i + 0.08, 0.85, i + 0.92, 1.75, role="data", solid=stage in planned)
        p.text(i + 0.5, 0.45, stage, "ink")
        if stage in mine.value:
            p.rect(i + 0.2, 0.05, i + 0.8, 0.17, role="loss", solid=True)
    shown = "; ".join(f"{s} {'+' if s in planned else '−'}" for s in stages)
    svg = chart.figure([p], alt=w["alt"].format(shown=shown, mine=_list_text(mine.value, seen.lang)),
                       width=520, height=170)
    title = w["title"].format(shown=_list_text(mine.value, seen.lang))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"]}</div>')


def _wer_ok(wer) -> bool:
    if type(wer) is not dict or not 1 <= len(wer) <= 10:
        return False
    for g, v in wer.items():
        x = _number(v)
        if not isinstance(g, str) or x is None or not 0 <= x <= 5:
            return False
    return True


_BEHIND_WORDS = {
    "en": {"title": "Your groups behind: {shown}", "xlabel": "word error rate",
           "row": "{group}: {wer}", "best": "best {best}", "line": "best + limit {line}",
           "key": "filled: the groups in your list; open: the others; solid line: the best group",
           "alt": "One bar per group, as long as its word error rate: {shown}. A solid line at the best group's {best}, and a dashed line at the best plus the limit, {line}. Your list: {mine}."},
    "nl": {"title": "Jouw groepen die achterblijven: {shown}", "xlabel": "word error rate",
           "row": "{group}: {wer}", "best": "beste {best}", "line": "beste + grens {line}",
           "key": "gevuld: de groepen in jouw lijst; open: de andere; doorgetrokken lijn: de beste groep",
           "alt": "Eén balk per groep, zo lang als de word error rate: {shown}. Een doorgetrokken lijn bij de {best} van de beste groep, en een gestippelde lijn bij de beste plus de grens, {line}. Jouw lijst: {mine}."},
}


def _behind_picture(seen) -> str:
    """A bar per group as long as its WER, filled for the groups in the learner's list, with a solid line at the best
    group and a dashed line at the best plus the limit: the groups past the dashed line are behind."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    wer, limit = names.get("wer"), _number(names.get("limit"))
    if not _wer_ok(wer) or limit is None or not 0 <= limit <= 5:
        return ""
    w = _BEHIND_WORDS[seen.lang]
    groups = list(wer)
    best = min(float(v) for v in wer.values())
    line = best + limit
    right = max(0.5, max(float(v) for v in wer.values()) * 1.15, line * 1.15)
    top = 1.1 * len(groups) + 0.9
    ticks = chart.ticks(0, right, 4)
    p = chart.Panel(x=(0, right), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, seen.lang, 1) for t in ticks])
    for i, g in enumerate(groups):
        y = top - 0.9 - 1.1 * (i + 1)
        v = float(wer[g])
        p.text(0.003 * right, y + 0.68, w["row"].format(group=g, wer=_show(v, seen.lang)), "ink", anchor="start",
               halo=True)
        p.rect(0, y + 0.12, max(v, 0.002 * right), y + 0.42, role="loss", solid=g in mine.value)
    p.line([best, best], [0, top - 0.75], role="ink-3", width=1.2)
    p.line([line, line], [0, top - 0.75], role="ink-2", width=1.5, dashed=True)
    p.text(min(line + 0.01 * right, 0.6 * right), top - 0.45, w["line"].format(line=_show(line, seen.lang)), "ink-2",
           anchor="start", halo=True)
    shown = "; ".join(f"{g} {_show(float(wer[g]), seen.lang)}" for g in groups)
    svg = chart.figure([p], alt=w["alt"].format(shown=shown, best=_show(best, seen.lang), line=_show(line, seen.lang),
                                                mine=_list_text(mine.value, seen.lang)),
                       width=520, height=max(200, 46 * len(groups) + 110))
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(shown=_list_text(mine.value, seen.lang))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"]}</div>')


# The lesson is done when these are correct. "proactive" and "fria" are choice questions in the lesson.
REQUIRED = ["missing", "proactive", "behind", "fria"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _is(want: list):
    """A mistake that matches exactly this list."""
    return lambda got, text: isinstance(got, list) and got == want


def _line(d: dict) -> str:
    return f"    {d!r},".replace("'", '"')


# --- exercise 1: the stages without a moment ---------------------------------------------------------------------------
_STAGES_LINE = "stages = " + repr(STAGES).replace("'", '"')
_PLAN_LINES = [_line(m) for m in PLAN]
_M_HEAD = ("# the stages of the AI Fairness Checklist of Madaio and colleagues (2020), in order\n" + _STAGES_LINE + "\n"
           "# a team's plan: the moments at which it looks at fairness, and who leads each one\n"
           "moments = [\n" + "\n".join(_PLAN_LINES) + "\n]\n\n")
_M_START = _M_HEAD + "...\nmissing\n"
_M_BODY = ("missing = []\nfor stage in stages:\n    count = 0\n    for moment in moments:\n"
           "        if moment[\"stage\"] == stage:\n            count = count + 1\n    if count == 0:\n"
           "        missing.append(stage)\n")
_M_SOLUTION = _M_HEAD + _M_BODY + "missing"


def _m(body: str) -> str:
    return _M_HEAD + body + "missing"


def _m_swap(old: str, new: str) -> str:
    assert _M_BODY.count(old) == 1
    return _m(_M_BODY.replace(old, new))


_WANT_MISSING = missing(STAGES, PLAN)                              # define, prototype, evolve
_PLANNED = [s for s in STAGES if s not in _WANT_MISSING]           # envision, build, launch
_LAST_ONLY = [s for s in STAGES if s != PLAN[-1]["stage"]]         # every stage but launch

# --- exercise 3: the groups behind --------------------------------------------------------------------------------
_WER_LINE = "wer = " + repr(WER).replace("'", '"')
_LIMIT_LINE = "limit = 0.05"
_B_HEAD = ("# the average word error rate per group of speakers, from Koenecke and colleagues (2020)\n" + _WER_LINE + "\n"
           "# agreed before the review: how far above the best group's WER a group may lie\n" + _LIMIT_LINE + "\n\n")
_B_START = _B_HEAD + "...\nbehind\n"
_B_BODY = ("values = []\nfor group in wer:\n    values.append(wer[group])\nbest = min(values)\nbehind = []\n"
           "for group in wer:\n    if wer[group] > best + limit:\n        behind.append(group)\n")
_B_SOLUTION = _B_HEAD + _B_BODY + "behind"


def _b(body: str) -> str:
    return _B_HEAD + body + "behind"


def _b_swap(old: str, new: str) -> str:
    assert _B_BODY.count(old) == 1
    return _b(_B_BODY.replace(old, new))


_NO_LIMIT = [g for g, v in WER.items() if v > min(WER.values())]          # every group but the best
_WITHIN = [g for g in WER if g not in behind(WER, 0.05)]                   # white men, white women
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "missing": CodeBox(
        name="missing",
        picture=_missing_picture,
        start=_M_START,
        solution=_M_SOLUTION,
        keep=[_STAGES_LINE, "moments = ["] + _PLAN_LINES,
        # The check also runs the box with another plan.
        others=[{"moments": OTHER_PLAN}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_moments`. In the box the names have none: `stages`, `moments`, `stage`, `moment`, `count`, `missing`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_moments`. In het vak hebben de namen dat niet: `stages`, `moments`, `stage`, `moment`, `count`, `missing`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a moment for a key it does not have. Each moment has `\"moment\"`, `\"stage\"` and `\"lead\"`.",
                    nl="Je code vraagt een moment om een sleutel die het niet heeft. Elk moment heeft `\"moment\"`, `\"stage\"` en `\"lead\"`."),
            Mistake("planned", matches=_is(_PLANNED),
                    en="Your list holds the stages that *have* a moment. The check wants the ones without: append a stage when `count` is still 0 after the inner loop.",
                    nl="Je lijst bevat de fasen die *wel* een moment hebben. De check wil de fasen zonder: voeg een fase toe als `count` na de binnenste loop nog 0 is."),
            Mistake("last-only", matches=_is(_LAST_ONLY),
                    en="Your list holds every stage but `launch`, the stage of the last moment: your code only remembers the last moment it looked at. Count every moment in the stage with `count = count + 1`, and never set `count` back to 0 in the inner loop.",
                    nl="Je lijst bevat elke fase behalve `launch`, de fase van het laatste moment: je code onthoudt alleen het laatste moment waar hij naar keek. Tel elk moment in de fase met `count = count + 1`, en zet `count` in de binnenste loop nooit terug op 0."),
            Mistake("twice", matches=lambda got, text: isinstance(got, list) and len(got) > len(set(map(repr, got))),
                    en="Your list holds a stage more than once. Decide once per stage, after the inner loop has looked at every moment: put the `if count == 0:` under `for stage in stages:`, not under `for moment in moments:`.",
                    nl="Je lijst bevat een fase meer dan één keer. Beslis één keer per fase, nadat de binnenste loop elk moment heeft bekeken: zet de `if count == 0:` onder `for stage in stages:`, niet onder `for moment in moments:`."),
            Mistake("empty", matches=_is([]),
                    en="Your list is empty, but three stages have no moment. Do you set `count = 0` again for every stage, at the top of the outer loop, and add 1 only when `moment[\"stage\"] == stage`?",
                    nl="Je lijst is leeg, maar drie fasen hebben geen moment. Zet je `count = 0` opnieuw voor elke fase, bovenaan in de buitenste loop, en tel je alleen 1 op als `moment[\"stage\"] == stage`?"),
            Mistake("order", matches=lambda got, text: isinstance(got, list) and got != _WANT_MISSING and sorted(map(repr, got)) == sorted(map(repr, _WANT_MISSING)),
                    en="Your list has the right stages, in another order. Keep the order of `stages`: loop over `stages`, and append in that order.",
                    nl="Je lijst heeft de goede fasen, in een andere volgorde. Houd de volgorde van `stages` aan: loop over `stages`, en voeg in die volgorde toe."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `missing = []`, and for each stage, count the moments in that stage; when there are none, append the stage.",
                      "nl": "Vervang de `...` door je code: begin met `missing = []`, en tel voor elke fase de momenten in die fase; zijn er geen, voeg de fase dan toe."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `stages` and `moments` exactly as they were: put them back. The check tries another plan by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `stages` en `moments` nodig zoals ze waren: zet ze terug. De check probeert zelf een ander plan."},
            "no_value": {"en": "The last line of the box shows nothing. Put `missing` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `missing` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. For each stage, do you set `count = 0`, add 1 when a moment has `moment[\"stage\"] == stage`, and append the stage when `count` is still 0?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je voor elke fase `count = 0`, tel je 1 op als een moment `moment[\"stage\"] == stage` heeft, en voeg je de fase toe als `count` nog 0 is?"},
            "other": {"en": "Right for this plan. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you look at `moments` itself, and not at the stage names of this plan?",
                      "nl": "Goed voor dit plan. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Kijk je naar `moments` zelf, en niet naar de namen van de fasen in dit plan?"},
            "other_error": {"en": "Right for this plan. But with {where}, your code stops with an error. Do you look at `moments` itself?",
                            "nl": "Goed voor dit plan. Maar met {where} stopt je code met een fout. Kijk je naar `moments` zelf?"},
            "correct": {"en": "`['define', 'prototype', 'evolve']`: this plan looks at fairness before work starts, in a code review and at the ship review, but not when the product is specified, not while it is a prototype, and not after launch. Without a moment in *evolve*, problems after launch only come up when someone complains: the reactive auditing that Holstein and colleagues heard about.",
                        "nl": "`['define', 'prototype', 'evolve']`: dit plan kijkt naar fairness voordat het werk begint, bij een code review en bij de ship review, maar niet als het product wordt vastgelegd, niet zolang het een prototype is, en niet na de lancering. Zonder moment in *evolve* komen problemen na de lancering pas boven als iemand klaagt: het reactieve controleren waar Holstein en collega's over hoorden."},
        },
    ),
    "behind": CodeBox(
        name="behind",
        picture=_behind_picture,
        start=_B_START,
        solution=_B_SOLUTION,
        keep=[_WER_LINE, _LIMIT_LINE],
        # The check also runs the box with a smaller limit.
        others=[{"limit": 0.03}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_wer`. In the box the names have none: `wer`, `limit`, `group`, `values`, `best`, `behind`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_wer`. In het vak hebben de namen dat niet: `wer`, `limit`, `group`, `values`, `best`, `behind`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `wer` for a group it does not have. Loop over `wer` itself, `for group in wer:`, and look up each group's WER with `wer[group]`.",
                    nl="Je code vraagt `wer` om een groep die er niet in zit. Loop over `wer` zelf, `for group in wer:`, en zoek de WER van elke groep op met `wer[group]`."),
            Mistake("values", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(_number(x) is not None for x in got),
                    en="Your list holds numbers, the WERs. The check wants the names of the groups: append `group`, not `wer[group]`.",
                    nl="Je lijst bevat getallen, de WER's. De check wil de namen van de groepen: voeg `group` toe, niet `wer[group]`."),
            Mistake("no-limit", matches=_is(_NO_LIMIT),
                    en="Your list also holds `white men`, whose WER of 0.21 lies 0.04 above the best group's 0.17: within the limit of 0.05. Do you compare with `best + limit`?",
                    nl="Je lijst bevat ook `white men`, met een WER van 0,21, 0,04 boven de 0,17 van de beste groep: binnen de grens van 0,05. Vergelijk je met `best + limit`?"),
            Mistake("within", matches=_is(_WITHIN),
                    en="Your list holds the groups *within* the limit. The check wants the groups more than `limit` above the best: `wer[group] > best + limit`.",
                    nl="Je lijst bevat de groepen *binnen* de grens. De check wil de groepen die meer dan `limit` boven de beste liggen: `wer[group] > best + limit`."),
            Mistake("everyone", matches=_is(list(WER)),
                    en="Your list holds every group, the best one too. Compare each WER with `best + limit`, where `best` is the lowest WER.",
                    nl="Je lijst bevat elke groep, ook de beste. Vergelijk elke WER met `best + limit`, waarbij `best` de laagste WER is."),
            Mistake("empty", matches=_is([]),
                    en="Your list is empty, but two groups lie more than 0.05 above the best group. Is `best` the *lowest* WER, `min(values)`? A lower WER is better.",
                    nl="Je lijst is leeg, maar twee groepen liggen meer dan 0,05 boven de beste groep. Is `best` de *laagste* WER, `min(values)`? Een lagere WER is beter."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: find `best`, the lowest WER, then start `behind = []` and append each group whose WER is above `best + limit`.",
                      "nl": "Vervang de `...` door je code: zoek `best`, de laagste WER, begin dan met `behind = []` en voeg elke groep toe waarvan de WER boven `best + limit` ligt."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `wer` and `limit` exactly as they were: put them back. The check tries another limit by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `wer` en `limit` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere grens."},
            "no_value": {"en": "The last line of the box shows nothing. Put `behind` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `behind` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you take `best` as the lowest WER, and append each group with `wer[group] > best + limit`, in the order of `wer`?",
                      "nl": "Je laatste regel geeft `{got}`. Neem je als `best` de laagste WER, en voeg je elke groep met `wer[group] > best + limit` toe, in de volgorde van `wer`?"},
            "other": {"en": "Right for a limit of 0.05. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `limit` itself, and not the number 0.05?",
                      "nl": "Goed voor een grens van 0,05. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `limit` zelf, en niet het getal 0,05?"},
            "other_error": {"en": "Right for a limit of 0.05. But with {where}, your code stops with an error. Do you use `limit` itself?",
                            "nl": "Goed voor een grens van 0,05. Maar met {where} stopt je code met een fout. Gebruik je `limit` zelf?"},
            "correct": {"en": "`['black men', 'black women']`: their WERs of 0.41 and 0.30 lie more than 0.05 above the 0.17 of White women; White men, at 0.21, stay within the limit. The code tells you which groups need work. What follows is not code: who changes the system, with which data, and who decides whether it ships until then.",
                        "nl": "`['black men', 'black women']`: hun WER's van 0,41 en 0,30 liggen meer dan 0,05 boven de 0,17 van witte vrouwen; witte mannen, met 0,21, blijven binnen de grens. De code zegt welke groepen werk nodig hebben. Wat daarna komt, is geen code: wie het systeem verandert, met welke data, en wie beslist of het tot die tijd uitkomt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("missing", _M_START, "empty", None),
    ("missing", _M_SOLUTION, "correct", None),
    ("missing", _m("planned = []\nfor moment in moments:\n    planned.append(moment[\"stage\"])\nmissing = []\n"
                   "for stage in stages:\n    if stage not in planned:\n        missing.append(stage)\n"), "correct", None),
    ("missing", _m("missing = []\nfor stage in stages:\n    found = False\n    for moment in moments:\n"
                   "        if moment[\"stage\"] == stage:\n            found = True\n    if not found:\n"
                   "        missing.append(stage)\n"), "correct", None),
    ("missing", _m(_M_BODY.replace("in moments", "in _moments")), "error", "underscore"),
    ("missing", _m_swap("moment[\"stage\"] == stage", "moment[\"Stage\"] == stage"), "error", "key"),
    ("missing", _m_swap("    if count == 0:\n", "    if count > 0:\n"), "wrong", "planned"),
    ("missing", _m_swap("            count = count + 1\n",
                        "            count = count + 1\n        else:\n            count = 0\n"), "wrong", "last-only"),
    ("missing", _m("missing = []\nfor stage in stages:\n    for moment in moments:\n"
                   "        if moment[\"stage\"] != stage:\n            missing.append(stage)\n"), "wrong", "twice"),
    ("missing", _m("missing = []\ncount = 0\nfor stage in stages:\n    for moment in moments:\n"
                   "        if moment[\"stage\"] == stage:\n            count = count + 1\n    if count == 0:\n"
                   "        missing.append(stage)\n"), "wrong", "empty"),
    ("missing", _m(_M_BODY + "missing = sorted(missing)\n"), "wrong", "order"),
    ("missing", _m(_M_BODY + "missing = missing[:2]\n"), "wrong", None),
    ("missing", _m("missing = [\"define\", \"prototype\", \"evolve\"]\n"), "almost", "other-value"),
    ("missing", _M_SOLUTION.replace('"stage": "build"', '"stage": "define"'), "wrong", "changed-line"),
    ("missing", _m("...\nmissing = [\"define\", \"prototype\", \"evolve\"]\n"), "wrong", "dots-left"),
    ("behind", _B_START, "empty", None),
    ("behind", _B_SOLUTION, "correct", None),
    ("behind", _b("best = min(wer.values())\nbehind = []\nfor group, value in wer.items():\n"
                  "    if value - best > limit:\n        behind.append(group)\n"), "correct", None),
    ("behind", _b("best = None\nfor group in wer:\n    if best is None or wer[group] < best:\n        best = wer[group]\n"
                  "behind = []\nfor group in wer:\n    if wer[group] > best + limit:\n        behind.append(group)\n"),
     "correct", None),
    ("behind", _b(_B_BODY.replace("in wer:\n    values", "in _wer:\n    values")), "error", "underscore"),
    ("behind", _b_swap("values.append(wer[group])", "values.append(wer[\"group\"])"), "error", "key"),
    ("behind", _b_swap("behind.append(group)", "behind.append(wer[group])"), "wrong", "values"),
    ("behind", _b_swap("wer[group] > best + limit", "wer[group] > best"), "wrong", "no-limit"),
    ("behind", _b_swap("wer[group] > best + limit", "wer[group] <= best + limit"), "wrong", "within"),
    ("behind", _b_swap("wer[group] > best + limit", "wer[group] > limit"), "wrong", "everyone"),
    ("behind", _b_swap("best = min(values)", "best = max(values)"), "wrong", "empty"),
    ("behind", _b(_B_BODY + "behind = behind[:1]\n"), "wrong", None),
    ("behind", _b("behind = [\"black men\", \"black women\"]\n"), "almost", "other-value"),
    ("behind", _b_swap("wer[group] > best + limit", "wer[group] > best + 0.05"), "almost", "other-value"),
    ("behind", _B_SOLUTION.replace("limit = 0.05", "limit = 0.03"), "wrong", "changed-line"),
    ("behind", _b("...\nbehind = [\"black men\", \"black women\"]\n"), "wrong", "dots-left"),
]
