# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-bias-speech-ads: the share of snippets whose word error rate is above a limit, from ten
snippets the course made up, and the share of one gender among the people a job ad reached, from Table 2 of Lambrecht
and Tucker (2019); with the picture of each."""
import math
import numbers
import re

from everything_ai import yours
from everything_ai.check import CodeBox, Mistake

# --- speech recognition -----------------------------------------------------------------------------------------------
# Ten snippets of one speaker, made up by the course: the words in the transcript people made, and the words a speech
# recognizer substituted, deleted and inserted. Snippet 4 lies exactly on the limit of 0.5, and snippet 10 has more
# errors than words, because insertions count.
SNIPPETS = [(42, 5, 2, 1), (38, 9, 4, 2), (45, 14, 6, 3), (40, 12, 5, 3), (36, 3, 1, 0),
            (30, 10, 4, 4), (44, 8, 3, 2), (33, 11, 3, 3), (48, 6, 2, 1), (12, 6, 2, 8)]

# Koenecke and colleagues (2020), Table 2: the average WER on 206 identical phrases, for black and white speakers.
TABLE_2 = {"Apple": (0.28, 0.12), "IBM": (0.21, 0.10), "Google": (0.17, 0.11), "Amazon": (0.18, 0.08),
           "Microsoft": (0.13, 0.07)}


def snippets() -> list:
    """The snippets as the lesson's code writes them: a dict per snippet."""
    return [{"words": n, "substituted": s, "deleted": d, "inserted": i} for n, s, d, i in SNIPPETS]


def wer(snippet: dict) -> float:
    """The word error rate of one snippet: (S + D + I) / N."""
    return (snippet["substituted"] + snippet["deleted"] + snippet["inserted"]) / snippet["words"]


def unusable(rows: list, limit: float) -> float:
    """The share of the snippets whose word error rate is above the limit."""
    return sum(1 for s in rows if wer(s) > limit) / len(rows)


# --- the STEM ad --------------------------------------------------------------------------------------------------
# Lambrecht and Tucker (2019), Table 2, "Raw Data": the impressions and clicks of the STEM job ad on Facebook, per age
# group and gender (age group, male impressions, female impressions, male clicks, female clicks).
TABLE_LT = [("18-24", 746719, 649590, 1156, 1171), ("25-34", 662996, 495996, 873, 758),
            ("35-44", 412457, 283596, 501, 480), ("45-54", 307701, 224809, 413, 414),
            ("55-64", 209608, 176454, 320, 363), ("65+", 192317, 153470, 307, 321)]


def ad_cells() -> list:
    """The cells of the ad as the lesson's code writes them: a dict per age group and gender."""
    rows = []
    for age, mi, fi, mc, fc in TABLE_LT:
        rows.append({"age": age, "gender": "male", "impressions": mi, "clicks": mc})
        rows.append({"age": age, "gender": "female", "impressions": fi, "clicks": fc})
    return rows


def share(rows: list, count: str, gender: str) -> float:
    """The cells of one gender added up for one count, divided by that count added up over all cells."""
    return sum(c[count] for c in rows if c["gender"] == gender) / sum(c[count] for c in rows)


def click_rate(rows: list, gender: str) -> float:
    """The clicks of one gender divided by the impressions of that gender."""
    picked = [c for c in rows if c["gender"] == gender]
    return sum(c["clicks"] for c in picked) / sum(c["impressions"] for c in picked)


# --- pictures ---------------------------------------------------------------------------------------------------------
def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _short(x: float, lang: str) -> str:
    """A number with as few decimals as it needs, at most three, such as 0.5 or 0.4."""
    text = f"{round(x, 3):g}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _snippets_ok(rows) -> bool:
    """The learner's snippets, when they still look like the lesson's: 1 to 20 dicts with counts."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        return False
    for s in rows:
        if type(s) is not dict:
            return False
        values = [_number(s.get(k)) for k in ("words", "substituted", "deleted", "inserted")]
        if any(v is None or not 0 <= v <= 10000 for v in values) or values[0] == 0:
            return False
    return True


_WER_WORDS = {
    "en": {"title": "Your share above the limit {limit}: {share}", "xlabel": "word error rate",
           "row": "snippet {i}: {wer}", "limit": "limit {limit}",
           "key": "filled: above the limit, unusable; open: at or below the limit",
           "alt": "One bar per snippet, as long as its word error rate: {shown}. The bars above the limit of {limit} are filled. Your share: {share}."},
    "nl": {"title": "Jouw deel boven de grens {limit}: {share}", "xlabel": "word error rate",
           "row": "fragment {i}: {wer}", "limit": "grens {limit}",
           "key": "gevuld: boven de grens, onbruikbaar; open: op of onder de grens",
           "alt": "Eén balk per fragment, zo lang als de word error rate: {shown}. De balken boven de grens van {limit} zijn gevuld. Jouw deel: {share}."},
}


def _wer_picture(seen) -> str:
    """A bar per snippet as long as its word error rate, filled when it is above the limit, with a dashed line at the
    limit: the learner's share is the filled bars out of all bars."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    value = _number(mine.value)
    names = mine.names if isinstance(mine.names, dict) else {}
    rows, limit = names.get("snippets"), _number(names.get("limit"))
    if value is None or not _snippets_ok(rows) or limit is None or not 0 < limit <= 5:
        return ""
    w = _WER_WORDS[seen.lang]
    rates = [wer(s) for s in rows]
    right = max(1.0, max(rates) * 1.1, limit * 1.15)
    top = 1.1 * len(rows) + 0.9
    ticks = chart.ticks(0, right, 4)
    p = chart.Panel(x=(0, right), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_short(t, seen.lang) for t in ticks])
    for i, r in enumerate(rates):
        y = top - 0.9 - 1.1 * (i + 1)
        p.text(0.003 * right, y + 0.68, w["row"].format(i=i + 1, wer=_show(r, seen.lang)), "ink", anchor="start",
               halo=True)
        p.rect(0, y + 0.12, max(r, 0.002 * right), y + 0.42, role="loss", solid=r > limit)
    p.line([limit, limit], [0, top - 0.75], role="ink-2", width=1.5, dashed=True)
    p.text(min(limit + 0.01 * right, 0.7 * right), top - 0.45, w["limit"].format(limit=_short(limit, seen.lang)),
           "ink-2", anchor="start", halo=True)
    shown = "; ".join(_show(r, seen.lang) for r in rates)
    svg = chart.figure([p], alt=w["alt"].format(shown=shown, limit=_short(limit, seen.lang),
                                                share=_short(value, seen.lang)),
                       width=520, height=max(200, 34 * len(rows) + 110))
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(limit=_short(limit, seen.lang), share=_short(value, seen.lang))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"]}</div>')


def _ad_cells_ok(rows, count, gender) -> bool:
    """The learner's cells, when they still look like the lesson's: 1 to 24 dicts with an age, a gender and counts."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 24 or count not in ("impressions", "clicks"):
        return False
    if not isinstance(gender, str):
        return False
    for c in rows:
        if type(c) is not dict or not isinstance(c.get("age"), str) or not isinstance(c.get("gender"), str):
            return False
        v = _number(c.get(count))
        if v is None or not 0 <= v <= 10 ** 8:
            return False
    return True


_SHARE_WORDS = {
    "en": {"title": "Your share of the {count} for gender = {gender}: {share}", "xlabel": "share of the {count} per age group",
           "row": "{age}: {share}", "half": "half", "mine": "your share {share}",
           "key": "per age group, the share of gender = {gender} among the {count}; solid line: half",
           "alt": "One bar per age group, as long as the share of gender = {gender} among its {count}: {shown}. A solid line at half, and a dashed line at your share, {share}."},
    "nl": {"title": "Jouw aandeel in de {count} voor gender = {gender}: {share}", "xlabel": "aandeel in de {count} per leeftijdsgroep",
           "row": "{age}: {share}", "half": "de helft", "mine": "jouw aandeel {share}",
           "key": "per leeftijdsgroep het aandeel van gender = {gender} in de {count}; doorgetrokken lijn: de helft",
           "alt": "Eén balk per leeftijdsgroep, zo lang als het aandeel van gender = {gender} in de {count}: {shown}. Een doorgetrokken lijn bij de helft, en een gestippelde lijn bij jouw aandeel, {share}."},
}


def _share_picture(seen) -> str:
    """Per age group, a bar as long as the share of the learner's gender in that group's impressions or clicks, with
    a line at half and a dashed line at the learner's share: the share over all ages lies among the bars."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    value = _number(mine.value)
    names = mine.names if isinstance(mine.names, dict) else {}
    rows, count, gender = names.get("cells"), names.get("count"), names.get("gender")
    if value is None or not _ad_cells_ok(rows, count, gender):
        return ""
    ages = []
    for c in rows:
        if c["age"] not in ages:
            ages.append(c["age"])
    parts = []
    for age in ages:
        here = [c for c in rows if c["age"] == age]
        whole = sum(float(c[count]) for c in here)
        if whole <= 0:
            return ""
        parts.append(sum(float(c[count]) for c in here if c["gender"] == gender) / whole)
    w = _SHARE_WORDS[seen.lang]
    top = 1.1 * len(ages) + 0.9
    p = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"].format(count=count), yticks=[],
                    xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=[_short(t, seen.lang) for t in (0, 0.25, 0.5, 0.75, 1)])
    for i, (age, part) in enumerate(zip(ages, parts)):
        y = top - 0.9 - 1.1 * (i + 1)
        p.text(0.003, y + 0.68, w["row"].format(age=age, share=_show(part, seen.lang)), "ink", anchor="start",
               halo=True)
        p.rect(0, y + 0.12, max(part, 0.002), y + 0.42, role="data", solid=True)
    p.line([0.5, 0.5], [0, top - 0.75], role="ink-3", width=1.2)
    if 0 <= value <= 1:
        p.line([value, value], [0, top - 0.75], role="ink-2", width=1.5, dashed=True)
        p.text(min(value + 0.01, 0.62), top - 0.45, w["mine"].format(share=_show(value, seen.lang)), "ink-2",
               anchor="start", halo=True)
    shown = "; ".join(f"{age} {_show(part, seen.lang)}" for age, part in zip(ages, parts))
    svg = chart.figure([p], alt=w["alt"].format(gender=gender, count=count, shown=shown,
                                                share=_show(value, seen.lang)),
                       width=520, height=max(200, 40 * len(ages) + 110))
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(count=count, gender=gender, share=_show(value, seen.lang))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"].format(gender=gender, count=count)}</div>')


# The lesson is done when these are correct. "cause" and "audience" are choice questions in the lesson.
REQUIRED = ["wer", "cause", "share", "audience"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _near(want: float, tol: float):
    """A mistake that matches a number within `tol` of `want`."""
    def matches(got, text):
        x = _number(got)
        return x is not None and abs(x - want) < tol
    return matches


def _close(tol: float):
    def compare(got, want):
        a, b = _number(got), _number(want)
        return a is not None and b is not None and abs(a - b) < tol
    return compare


def _line(d: dict) -> str:
    return f"    {d!r},".replace("'", '"')


# --- exercise 1: the share of unusable snippets ------------------------------------------------------------------------
_SN = snippets()
_SN_LINES = [_line(s) for s in _SN]
_LIMIT_LINE = "limit = 0.5"
_W_HEAD = ("# ten snippets of one speaker, made up by the course: the words in the correct transcript,\n"
           "# and the words the speech recognizer substituted, deleted and inserted\n"
           "snippets = [\n" + "\n".join(_SN_LINES) + "\n]\n"
           "# a transcript counts as unusable when its WER is above this limit\n" + _LIMIT_LINE + "\n\n")
_W_START = _W_HEAD + "...\nunusable\n"
_W_BODY = ("count = 0\nfor snippet in snippets:\n"
           "    errors = snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]\n"
           "    wer = errors / snippet[\"words\"]\n    if wer > limit:\n        count = count + 1\n"
           "unusable = count / len(snippets)\n")
_W_SOLUTION = _W_HEAD + _W_BODY + "unusable"


def _w(body: str) -> str:
    return _W_HEAD + body + "unusable"


def _w_swap(old: str, new: str) -> str:
    assert _W_BODY.count(old) == 1
    return _w(_W_BODY.replace(old, new))


_WERS = [wer(s) for s in _SN]
_POOLED = sum(s["substituted"] + s["deleted"] + s["inserted"] for s in _SN) / sum(s["words"] for s in _SN)
_FEWER = [sum(1 for s in _SN if (s["substituted"] + s["deleted"]) / s["words"] > 0.5) / 10,
          sum(1 for s in _SN if (s["substituted"] + s["inserted"]) / s["words"] > 0.5) / 10,
          sum(1 for s in _SN if (s["deleted"] + s["inserted"]) / s["words"] > 0.5) / 10,
          sum(1 for s in _SN if s["substituted"] / s["words"] > 0.5) / 10]

# --- exercise 3: the share of one gender among the people the ad reached ------------------------------------------
_AD = ad_cells()
_AD_LINES = [_line(c) for c in _AD]
_COUNT_LINE, _GENDER_LINE = 'count = "impressions"', 'gender = "female"'
_S_HEAD = ("# Lambrecht and Tucker (2019), Table 2: the STEM job ad on Facebook, per age group and gender\n"
           "cells = [\n" + "\n".join(_AD_LINES) + "\n]\n"
           "# what to count, and whose share to work out\n" + _COUNT_LINE + "\n" + _GENDER_LINE + "\n\n")
_S_START = _S_HEAD + "...\nshare\n"
_S_BODY = ("total = 0\nmine = 0\nfor cell in cells:\n    total = total + cell[count]\n"
           "    if cell[\"gender\"] == gender:\n        mine = mine + cell[count]\nshare = mine / total\n")
_S_SOLUTION = _S_HEAD + _S_BODY + "share"


def _s(body: str) -> str:
    return _S_HEAD + body + "share"


def _s_swap(old: str, new: str) -> str:
    assert _S_BODY.count(old) == 1
    return _s(_S_BODY.replace(old, new))


_F_IMP = sum(c["impressions"] for c in _AD if c["gender"] == "female")     # 1,983,915
_M_IMP = sum(c["impressions"] for c in _AD if c["gender"] == "male")       # 2,531,798
_WOMEN = _F_IMP / (_F_IMP + _M_IMP)
_PER_AGE = [r[2] / (r[1] + r[2]) for r in TABLE_LT]
_LAST = _AD[-1]["impressions"]                                              # the women of 65 and over, 153,470
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "wer": CodeBox(
        name="wer",
        picture=_wer_picture,
        start=_W_START,
        solution=_W_SOLUTION,
        keep=["snippets = ["] + _SN_LINES + [_LIMIT_LINE],
        compare=_close(0.0006),
        digits=3,
        # The check also runs the box with a lower limit.
        others=[{"limit": 0.3}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_snippets`. In the box the names have none: `snippets`, `limit`, `snippet`, `errors`, `wer`, `count`, `unusable`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_snippets`. In het vak hebben de namen dat niet: `snippets`, `limit`, `snippet`, `errors`, `wer`, `count`, `unusable`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a snippet for a key it does not have. Each snippet has `\"words\"`, `\"substituted\"`, `\"deleted\"` and `\"inserted\"`.",
                    nl="Je code vraagt een fragment om een sleutel die het niet heeft. Elk fragment heeft `\"words\"`, `\"substituted\"`, `\"deleted\"` en `\"inserted\"`."),
            Mistake("brackets", matches=_near(1.0, 0.0006),
                    en="Your last line gives `{got}`: every snippet counts as above the limit. Do you divide the sum of the three counts by the words, with brackets around the sum, `(... + ... + ...) / snippet[\"words\"]`, and not the words by the errors?",
                    nl="Je laatste regel geeft `{got}`: elk fragment telt als boven de grens. Deel je de som van de drie aantallen door de woorden, met haakjes om de som, `(... + ... + ...) / snippet[\"words\"]`, en niet de woorden door de fouten?"),
            Mistake("at-limit", matches=_near(unusable(_SN, 0.5) + 0.1, 0.0006),
                    en="Your last line gives `{got}`, but four snippets lie above the limit, not five. With `>=` your code also counts snippet 4, whose WER is exactly 0.5; with `<` it counts the snippets below the limit. Unusable means *above* the limit: `wer > limit`.",
                    nl="Je laatste regel geeft `{got}`, maar er liggen vier fragmenten boven de grens, geen vijf. Met `>=` telt je code ook fragment 4 mee, met een WER van precies 0,5; met `<` telt hij de fragmenten onder de grens. Onbruikbaar betekent *boven* de grens: `wer > limit`."),
            Mistake("fewer", matches=lambda got, text: any(_near(v, 0.0006)(got, text) for v in _FEWER),
                    en="Your last line gives `{got}`: too few snippets are above the limit. Do you add all three counts, substituted, deleted *and* inserted? An inserted word is an error too.",
                    nl="Je laatste regel geeft `{got}`: te weinig fragmenten liggen boven de grens. Tel je alle drie de aantallen op, vervangen, weggelaten *en* toegevoegd? Een toegevoegd woord is ook een fout."),
            Mistake("count", matches=_near(4, 0.0006),
                    en="Your last line gives `{got}`: the number of unusable snippets. The check wants their share: divide by the number of snippets, `len(snippets)`.",
                    nl="Je laatste regel geeft `{got}`: het aantal onbruikbare fragmenten. De check wil hun deel: deel door het aantal fragmenten, `len(snippets)`."),
            Mistake("percent", matches=_near(40, 0.06),
                    en="Your last line gives `{got}`, a percentage. The check wants a share from 0 to 1: leave out the `* 100`.",
                    nl="Je laatste regel geeft `{got}`, een percentage. De check wil een deel van 0 tot 1: laat de `* 100` weg."),
            Mistake("mean", matches=_near(sum(_WERS) / len(_WERS), 0.0006),
                    en="Your last line gives `{got}`: the mean WER of the snippets. The check wants the share of snippets whose WER is above `limit`: count them with an `if`, and divide the count by `len(snippets)`.",
                    nl="Je laatste regel geeft `{got}`: de gemiddelde WER van de fragmenten. De check wil het deel van de fragmenten met een WER boven `limit`: tel ze met een `if`, en deel het aantal door `len(snippets)`."),
            Mistake("pooled", matches=_near(_POOLED, 0.0006),
                    en="Your last line gives `{got}`: all errors divided by all words, one WER for everything. The check wants the share of snippets whose own WER is above `limit`.",
                    nl="Je laatste regel geeft `{got}`: alle fouten gedeeld door alle woorden, één WER voor alles. De check wil het deel van de fragmenten waarvan de eigen WER boven `limit` ligt."),
            Mistake("below", matches=_near(1 - unusable(_SN, 0.5), 0.0006),
                    en="Your last line gives `{got}`: the share at or below the limit, the usable snippets. Count the snippets with `wer > limit`.",
                    nl="Je laatste regel geeft `{got}`: het deel op of onder de grens, de bruikbare fragmenten. Tel de fragmenten met `wer > limit`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `count = 0`, work out each snippet's WER in the loop, add 1 to `count` when it is above `limit`, and divide at the end.",
                      "nl": "Vervang de `...` door je code: begin met `count = 0`, reken in de loop de WER van elk fragment uit, tel 1 op bij `count` als die boven `limit` ligt, en deel aan het eind."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `snippets` and `limit` exactly as they were: put them back. The check tries another limit by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `snippets` en `limit` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere grens."},
            "no_value": {"en": "The last line of the box shows nothing. Put `unusable` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `unusable` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you work out `(substituted + deleted + inserted) / words` for each snippet, count the snippets above `limit`, and divide the count by `len(snippets)`?",
                      "nl": "Je laatste regel geeft `{got}`. Reken je voor elk fragment `(substituted + deleted + inserted) / words` uit, tel je de fragmenten boven `limit`, en deel je het aantal door `len(snippets)`?"},
            "other": {"en": "Right for a limit of 0.5. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you compare with `limit` itself, and not with the number 0.5?",
                      "nl": "Goed voor een grens van 0,5. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Vergelijk je met `limit` zelf, en niet met het getal 0,5?"},
            "other_error": {"en": "Right for a limit of 0.5. But with {where}, your code stops with an error. Do you compare with `limit` itself?",
                            "nl": "Goed voor een grens van 0,5. Maar met {where} stopt je code met een fout. Vergelijk je met `limit` zelf?"},
            "correct": {"en": "`0.4`: four of the ten snippets are above 0.5, snippet 10 even above 1: 16 errors on 12 words, 8 of them inserted. Snippet 4, exactly at 0.5, does not count. Koenecke and colleagues found 23 percent of the snippets of Black speakers above 0.5, against 1.6 percent of those of White speakers.",
                        "nl": "`0.4`: vier van de tien fragmenten liggen boven 0,5, fragment 10 zelfs boven 1: 16 fouten op 12 woorden, waarvan 8 toegevoegd. Fragment 4, op precies 0,5, telt niet mee. Koenecke en collega's vonden 23 procent van de fragmenten van zwarte sprekers boven 0,5, tegenover 1,6 procent van die van witte sprekers."},
        },
    ),
    "share": CodeBox(
        name="share",
        picture=_share_picture,
        start=_S_START,
        solution=_S_SOLUTION,
        keep=["cells = ["] + _AD_LINES + [_COUNT_LINE, _GENDER_LINE],
        compare=_close(0.0006),
        digits=3,
        # The check also runs the box for the clicks, and for the men.
        others=[{"count": "clicks"}, {"gender": "male"}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_cells`. In the box the names have none: `cells`, `count`, `gender`, `cell`, `total`, `mine`, `share`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_cells`. In het vak hebben de namen dat niet: `cells`, `count`, `gender`, `cell`, `total`, `mine`, `share`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks for a key that is not there. Each cell has `\"age\"`, `\"gender\"`, `\"impressions\"` and `\"clicks\"`. To pick what to count, use the name `count` itself, without quotes: `cell[count]`.",
                    nl="Je code vraagt om een sleutel die er niet is. Elk vakje heeft `\"age\"`, `\"gender\"`, `\"impressions\"` en `\"clicks\"`. Gebruik voor wat je telt de naam `count` zelf, zonder aanhalingstekens: `cell[count]`."),
            Mistake("zero", matches=_near(0.0, 1e-9),
                    en="Your last line gives `{got}`: nothing was added to `mine`. Does the `if` compare `cell[\"gender\"]` with `gender`?",
                    nl="Je laatste regel geeft `{got}`: er is niets opgeteld bij `mine`. Vergelijkt de `if` `cell[\"gender\"]` met `gender`?"),
            Mistake("same", matches=_near(1.0, 1e-9),
                    en="Your last line gives `{got}`: the part and the total are the same number. Add every cell to `total`, and only the cells of `gender` to `mine`.",
                    nl="Je laatste regel geeft `{got}`: het deel en het totaal zijn hetzelfde getal. Tel elk vakje op bij `total`, en alleen de vakjes van `gender` bij `mine`."),
            Mistake("one-cell", matches=lambda got, text: _near(_LAST / (_F_IMP + _M_IMP), 0.0006)(got, text) or _near(_F_IMP / _LAST, 0.006)(got, text),
                    en="Your last line gives `{got}`: one of your totals holds one cell only. Add each cell's count, `total = total + cell[count]` and `mine = mine + cell[count]`, instead of replacing it.",
                    nl="Je laatste regel geeft `{got}`: een van je totalen bevat maar één vakje. Tel het aantal van elk vakje op, `total = total + cell[count]` en `mine = mine + cell[count]`, in plaats van het te vervangen."),
            Mistake("other-side", matches=_near(1 - _WOMEN, 0.0006),
                    en="Your last line gives `{got}`: the share of the cells whose gender is *not* `gender`, the men. Pick the cells with `cell[\"gender\"] == gender`.",
                    nl="Je laatste regel geeft `{got}`: het aandeel van de vakjes waarvan het gender *niet* `gender` is, de mannen. Kies de vakjes met `cell[\"gender\"] == gender`."),
            Mistake("ratio", matches=_near(_F_IMP / _M_IMP, 0.0006),
                    en="Your last line gives `{got}`: the women's impressions divided by the men's. The share divides them by *all* impressions: add every cell to `total`.",
                    nl="Je laatste regel geeft `{got}`: de impressions van de vrouwen gedeeld door die van de mannen. Het aandeel deelt ze door *alle* impressions: tel elk vakje op bij `total`."),
            Mistake("men-more", matches=_near(_M_IMP / _F_IMP, 0.0006),
                    en="Your last line gives `{got}`: the men's impressions divided by the women's, so men saw the ad 27.6 percent more often. The check wants the share of `gender`: its impressions divided by all impressions.",
                    nl="Je laatste regel geeft `{got}`: de impressions van de mannen gedeeld door die van de vrouwen, dus mannen zagen de advertentie 27,6 procent vaker. De check wil het aandeel van `gender`: de impressions daarvan gedeeld door alle impressions."),
            Mistake("upside", matches=_near(1 / _WOMEN, 0.0006),
                    en="Your last line gives `{got}`, more than 1: you divided all impressions by the women's. Divide the part by the total: `mine / total`.",
                    nl="Je laatste regel geeft `{got}`, meer dan 1: je deelde alle impressions door die van de vrouwen. Deel het deel door het totaal: `mine / total`."),
            Mistake("mean", matches=_near(sum(_PER_AGE) / len(_PER_AGE), 0.0006),
                    en="Your last line gives `{got}`: the mean of the six age groups' shares. But the groups differ in size. Add up the impressions first, and divide once, at the end.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de aandelen van de zes leeftijdsgroepen. Maar die groepen zijn niet even groot. Tel eerst de impressions op, en deel één keer, aan het eind."),
            Mistake("percent", matches=_near(100 * _WOMEN, 0.06),
                    en="Your last line gives `{got}`, a percentage. The check wants a share from 0 to 1: leave out the `* 100`.",
                    nl="Je laatste regel geeft `{got}`, een percentage. De check wil een deel van 0 tot 1: laat de `* 100` weg."),
            Mistake("clicks", matches=_near(share(_AD, "clicks", "female"), 0.0006),
                    en="Your last line gives `{got}`: the women's share of the clicks. Here `count` is `\"impressions\"`: use `cell[count]`, not `cell[\"clicks\"]`.",
                    nl="Je laatste regel geeft `{got}`: het aandeel van de vrouwen in de clicks. Hier is `count` gelijk aan `\"impressions\"`: gebruik `cell[count]`, niet `cell[\"clicks\"]`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `total = 0` and `mine = 0`, add every cell's `cell[count]` to `total` and the cells of `gender` to `mine`, and divide at the end.",
                      "nl": "Vervang de `...` door je code: begin met `total = 0` en `mine = 0`, tel van elk vakje `cell[count]` op bij `total` en die van de vakjes van `gender` bij `mine`, en deel aan het eind."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `cells`, `count` and `gender` exactly as they were: put them back. The check tries other values by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `cells`, `count` en `gender` nodig zoals ze waren: zet ze terug. De check probeert zelf andere waarden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you add every cell's `cell[count]` to `total`, only the cells with `cell[\"gender\"] == gender` to `mine`, and then divide `mine` by `total`?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je van elk vakje `cell[count]` op bij `total`, alleen de vakjes met `cell[\"gender\"] == gender` bij `mine`, en deel je daarna `mine` door `total`?"},
            "other": {"en": "Right for the women's impressions. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `count` and `gender` themselves, and not the words `\"impressions\"` and `\"female\"`?",
                      "nl": "Goed voor de impressions van de vrouwen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `count` en `gender` zelf, en niet de woorden `\"impressions\"` en `\"female\"`?"},
            "other_error": {"en": "Right for the women's impressions. But with {where}, your code stops with an error. Do you use `count` and `gender` themselves?",
                            "nl": "Goed voor de impressions van de vrouwen. Maar met {where} stopt je code met een fout. Gebruik je `count` en `gender` zelf?"},
            "correct": {"en": "`0.439`: women got 43.9 percent of the impressions, and men 27.6 percent more than women. With `count = \"clicks\"` the women's share is 0.496: almost half of the clicks came from the women, who saw the ad less often. Per impression, women clicked more: 1.77 times per 1,000 against 1.41 for men.",
                        "nl": "`0.439`: vrouwen kregen 43,9 procent van de impressions, en mannen 27,6 procent meer dan vrouwen. Met `count = \"clicks\"` is het aandeel van de vrouwen 0,496: bijna de helft van de clicks kwam van de vrouwen, die de advertentie minder vaak zagen. Per impression klikten vrouwen vaker: 1,77 keer per 1.000 tegenover 1,41 bij mannen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("wer", _W_START, "empty", None),
    ("wer", _W_SOLUTION, "correct", None),
    ("wer", _w("count = 0\nfor snippet in snippets:\n"
               "    if (snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]) / snippet[\"words\"] > limit:\n"
               "        count += 1\nunusable = count / 10\n"), "correct", None),
    ("wer", _w("above = []\nfor snippet in snippets:\n"
               "    errors = snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]\n"
               "    if errors > limit * snippet[\"words\"]:\n        above.append(snippet)\n"
               "unusable = len(above) / len(snippets)\n"), "correct", None),
    ("wer", _w(_W_BODY.replace("in snippets", "in _snippets")), "error", "underscore"),
    ("wer", _w_swap("snippet[\"inserted\"]", "snippet[\"added\"]"), "error", "key"),
    ("wer", _w_swap("errors / snippet[\"words\"]",
                    "snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"] / snippet[\"words\"]"),
     "wrong", "brackets"),
    ("wer", _w_swap("errors / snippet[\"words\"]", "snippet[\"words\"] / errors"), "wrong", "brackets"),
    ("wer", _w_swap("wer > limit", "wer >= limit"), "wrong", "at-limit"),
    ("wer", _w_swap(" + snippet[\"inserted\"]", ""), "wrong", "fewer"),
    ("wer", _w_swap(" + snippet[\"deleted\"]", ""), "wrong", "fewer"),
    ("wer", _w_swap("unusable = count / len(snippets)", "unusable = count"), "wrong", "count"),
    ("wer", _w_swap("unusable = count / len(snippets)", "unusable = count / len(snippets) * 100"), "wrong",
     "percent"),
    ("wer", _w("total = 0\nfor snippet in snippets:\n"
               "    errors = snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]\n"
               "    total = total + errors / snippet[\"words\"]\nunusable = total / len(snippets)\n"), "wrong", "mean"),
    ("wer", _w("errors = 0\nwords = 0\nfor snippet in snippets:\n"
               "    errors = errors + snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]\n"
               "    words = words + snippet[\"words\"]\nunusable = errors / words\n"), "wrong", "pooled"),
    ("wer", _w_swap("wer > limit", "wer <= limit"), "wrong", "below"),
    ("wer", _w_swap("unusable = count / len(snippets)", "unusable = count / 9"), "wrong", None),
    ("wer", _w("unusable = 0.4\n"), "almost", "other-value"),
    ("wer", _w_swap("wer > limit", "wer > 0.5"), "almost", "other-value"),
    ("wer", _W_SOLUTION.replace("limit = 0.5", "limit = 0.3"), "wrong", "changed-line"),
    ("wer", _w("...\nunusable = 0.4\n"), "wrong", "dots-left"),
    ("share", _S_START, "empty", None),
    ("share", _S_SOLUTION, "correct", None),
    ("share", _s("total = 0\nmine = 0\nfor cell in cells:\n    total += cell[count]\n"
                 "    if cell[\"gender\"] == gender:\n        mine += cell[count]\nshare = round(mine / total, 3)\n"),
     "correct", None),
    ("share", _s("everyone = []\npicked = []\nfor cell in cells:\n    everyone.append(cell[count])\n"
                 "    if cell[\"gender\"] == gender:\n        picked.append(cell[count])\n"
                 "share = sum(picked) / sum(everyone)\n"), "correct", None),
    ("share", _s(_S_BODY.replace("in cells", "in _cells")), "error", "underscore"),
    ("share", _s_swap("total + cell[count]", "total + cell[\"count\"]"), "error", "key"),
    ("share", _s_swap("cell[\"gender\"] == gender", "cell[\"gender\"] == \"Female\""), "wrong", "zero"),
    ("share", _s("total = 0\nmine = 0\nfor cell in cells:\n    total = total + cell[count]\n"
                 "    mine = mine + cell[count]\nshare = mine / total\n"), "wrong", "same"),
    ("share", _s_swap("total = total + cell[count]", "total = cell[count]"), "wrong", "one-cell"),
    ("share", _s_swap("mine = mine + cell[count]", "mine = cell[count]"), "wrong", "one-cell"),
    ("share", _s_swap("cell[\"gender\"] == gender", "cell[\"gender\"] != gender"), "wrong", "other-side"),
    ("share", _s("men = 0\nmine = 0\nfor cell in cells:\n    if cell[\"gender\"] == gender:\n"
                 "        mine = mine + cell[count]\n    else:\n        men = men + cell[count]\nshare = mine / men\n"),
     "wrong", "ratio"),
    ("share", _s("men = 0\nmine = 0\nfor cell in cells:\n    if cell[\"gender\"] == gender:\n"
                 "        mine = mine + cell[count]\n    else:\n        men = men + cell[count]\nshare = men / mine\n"),
     "wrong", "men-more"),
    ("share", _s_swap("share = mine / total", "share = total / mine"), "wrong", "upside"),
    ("share", _s("parts = 0\nfor cell in cells:\n    if cell[\"gender\"] == gender:\n        mine = cell[count]\n"
                 "    else:\n        other = cell[count]\n        continue\n"
                 "    parts = parts + mine / (mine + other)\nshare = parts / 6\n"), "wrong", "mean"),
    ("share", _s_swap("share = mine / total", "share = mine / total * 100"), "wrong", "percent"),
    ("share", _s_swap("mine + cell[count]", "mine + cell[\"clicks\"]").replace("total + cell[count]",
                                                                            "total + cell[\"clicks\"]"),
     "wrong", "clicks"),
    ("share", _s_swap("share = mine / total", "share = mine / total / 2"), "wrong", None),
    ("share", _s("share = 1983915 / 4515713\n"), "almost", "other-value"),
    ("share", _s_swap("cell[\"gender\"] == gender", "cell[\"gender\"] == \"female\""), "almost", "other-value"),
    ("share", _S_SOLUTION.replace('gender = "female"', 'gender = "male"'), "wrong", "changed-line"),
    ("share", _s("...\nshare = 1983915 / 4515713\n"), "wrong", "dots-left"),
]
