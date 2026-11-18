# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-reason-codes: with a logistic regression on three columns of UCI Adult, the learner works
out for one applicant the gap of each feature against the applicants just above the cut-off (one of the two methods
in the official comment to Regulation B), and turns the gaps into reasons, the most negative first; with the picture of
the gaps against both reference groups, and pictures of the learner's gaps and of the notice their reasons make."""
import html
import math
import re

import numpy as np

from everything_ai import chart, paths, yours
from everything_ai.check import Code, CodeBox, Mistake

DATA = paths.ROOT / "data" / "adult.csv.gz"
COLUMNS = ["age", "education_num", "hours_per_week"]
# The model of the worked example: scikit-learn's LogisticRegression(max_iter=1000) on all 48,842 rows, to predict
# income above 50K, rounded (tests/test_rai_reason_codes.py fits it again).
WEIGHTS = np.array([0.046, 0.345, 0.042])
BIAS = -8.48
PERSON = [30, 9, 40]   # aged 30, a high-school diploma (education level 9), 40 hours a week
OTHER = [52, 10, 30]   # the applicant the check also tries
NEAR = (0, 0.25)       # "at or slightly above the minimum passing score": a logit from 0 up to 0.25

_TABLE = {}


def table() -> np.ndarray:
    """The three columns of the table UCI Adult of the course, as an array, read once."""
    if "X" not in _TABLE:
        import pandas as pd
        _TABLE["X"] = pd.read_csv(DATA)[COLUMNS].to_numpy()
    return _TABLE["X"]


def logits() -> np.ndarray:
    """Every applicant's logit, rounded to 3 decimals as the box does, so nobody sits on a bound by a rounding error."""
    return (table() @ WEIGHTS + BIAS).round(3)


def group(kind: str) -> np.ndarray:
    """Who is in the reference group: "near" (just above the cut-off), "all", or "yes" (everyone with a yes; with
    "yes-above" the one applicant exactly on the cut-off left out)."""
    z = logits()
    if kind == "near":
        return (z >= NEAR[0]) & (z < NEAR[1])
    if kind == "yes":
        return z >= 0
    return z > 0 if kind == "yes-above" else np.ones(len(z), dtype=bool)


def reference(kind: str = "near") -> np.ndarray:
    return table()[group(kind)].mean(axis=0)


def gaps(person, kind: str = "near") -> np.ndarray:
    """Each feature's weight times the applicant's value minus the average of the reference group."""
    return WEIGHTS * (np.array(person) - reference(kind))


def reasons(g) -> list:
    """The names of the features with a gap below 0, the most negative first."""
    out = []
    for i in np.argsort(g):
        if g[i] < 0:
            out.append(COLUMNS[i])
    return out


def _numbers(got) -> np.ndarray | None:
    """Three numbers, from an array, a list or a tuple; None otherwise."""
    if isinstance(got, (str, bytes, dict)) or got is None:
        return None
    try:
        a = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (3,) and np.isfinite(a).all() else None


def same_three(got, want) -> bool:
    """The same three numbers, to the 3 decimals the lesson shows them with."""
    a = _numbers(got)
    return a is not None and bool(np.allclose(a, want, rtol=0, atol=5.1e-4))


def _texts(got) -> list | None:
    """A list of texts, from a list, a tuple or an array of texts; None otherwise."""
    if isinstance(got, np.ndarray):
        got = got.tolist()
    if not isinstance(got, (list, tuple)) or not all(isinstance(x, str) for x in got):
        return None
    return [str(x) for x in got]


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _bar_panel(values, lo: float, hi: float, lang: str, xlabel: str, title: str = "") -> chart.Panel:
    """A bar per feature from 0, to the right for a positive gap and to the left for a negative one; the name and
    the number above each bar. Ticks every 0.5."""
    height = 1.1 * len(values) + 0.5
    ticks = [t / 2 for t in range(math.ceil(lo * 2), math.floor(hi * 2) + 1)]
    p = chart.Panel(x=(lo, hi), y=(0, height), title=title, xlabel=xlabel, yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, lang, 1) for t in ticks])
    p.line([0, 0], [0, height - 0.2], role="ink-3", width=1)
    for i, (name, v) in enumerate(zip(COLUMNS, values)):
        y = height - 0.5 - 1.1 * (i + 1)
        p.text(lo + (hi - lo) * 0.01, y + 0.68, f"{name}: {_show(v, lang)}", "ink", anchor="start", halo=True)
        x0, x1 = (0, v) if v >= 0 else (v, 0)
        p.rect(x0, y + 0.12, max(x1, x0 + (hi - lo) * 0.004), y + 0.42, role="param", solid=True)
    return p


_TWO_WORDS = {
    "en": {"all": "against all applicants", "near": "against those just above the cut-off", "gap": "gap",
           "alt": "Two bar charts of the gaps of one applicant. Left, against the average of all applicants: age "
                  "{a0}, education_num {a1}, hours_per_week {a2}. Right, against the average of the {n} applicants "
                  "just above the cut-off: age {n0}, education_num {n1}, hours_per_week {n2}.",
           "key": "To the left of 0 a feature weighs against the applicant. Against all applicants age weighs most; "
                  "against those just above the cut-off, education does."},
    "nl": {"all": "tegen alle aanvragers", "near": "tegen wie net boven de grens zit", "gap": "verschil",
           "alt": "Twee staafdiagrammen met de verschillen van één aanvrager. Links tegen het gemiddelde van alle "
                  "aanvragers: age {a0}, education_num {a1}, hours_per_week {a2}. Rechts tegen het gemiddelde van de "
                  "{n} aanvragers net boven de grens: age {n0}, education_num {n1}, hours_per_week {n2}.",
           "key": "Links van 0 weegt een feature tegen de aanvrager. Tegen alle aanvragers weegt de leeftijd het "
                  "zwaarst; tegen wie net boven de grens zit, de opleiding."},
}


def two_references(lang: str) -> str:
    """The picture of *The idea*: the gaps of the lesson's applicant against both reference groups."""
    w = _TWO_WORDS[lang]
    g_all, g_near = gaps(PERSON, "all"), gaps(PERSON, "near")
    left = _bar_panel(g_all, -1.5, 0.5, lang, w["gap"], w["all"])
    right = _bar_panel(g_near, -1.5, 0.5, lang, w["gap"], w["near"])
    fill = {f"a{i}": _show(v, lang) for i, v in enumerate(g_all)}
    fill.update({f"n{i}": _show(v, lang) for i, v in enumerate(g_near)})
    fill["n"] = f"{int(group('near').sum()):,}".replace(",", "." if lang == "nl" else ",")
    svg = chart.figure([left, right], alt=w["alt"].format(**fill), width=340, height=230)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>'


_GAPS_WORDS = {
    "en": {"title": "Your gaps for the applicant", "gap": "gap",
           "alt": "Your gaps as bars: {shown}.",
           "some": "Below 0 a feature weighs against the applicant. The reasons your gaps give, the most negative "
                   "first: {order}.",
           "none": "None of your gaps is below 0: they give no reason for a no."},
    "nl": {"title": "Jouw verschillen voor de aanvrager", "gap": "verschil",
           "alt": "Jouw verschillen als staven: {shown}.",
           "some": "Onder 0 weegt een feature tegen de aanvrager. De redenen die jouw verschillen geven, de meest "
                   "negatieve eerst: {order}.",
           "none": "Geen van jouw verschillen is kleiner dan 0: ze geven geen reden voor een nee."},
}


def _gaps_picture(seen) -> str:
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    values = _numbers(mine.value)
    if values is None or not all(abs(v) <= 50 for v in values):
        return ""
    w = _GAPS_WORDS[seen.lang]
    reach = max([1.5] + [abs(v) * 1.1 for v in values])
    p = _bar_panel(values, -reach, reach, seen.lang, w["gap"])
    shown = "; ".join(f"{k} {_show(v, seen.lang)}" for k, v in zip(COLUMNS, values))
    found = reasons(values)
    key = w["some"].format(order=", ".join(found)) if found else w["none"]
    svg = chart.figure([p], alt=w["alt"].format(shown=shown), width=520, height=46 * 3 + 100)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# What a notice says for each feature: the factor, not what was wrong with it (comment 9(b)(2)-3 of Regulation B).
_FACTORS = {
    "en": {"age": "Age", "education_num": "Level of education", "hours_per_week": "Hours worked per week"},
    "nl": {"age": "Leeftijd", "education_num": "Opleidingsniveau", "hours_per_week": "Gewerkte uren per week"},
}
_NOTICE_WORDS = {
    "en": {"title": "Your notice", "head": "We cannot approve your application.",
           "lead": "Principal reasons:", "empty": "Your list holds no reasons: a notice like this would be insufficient.",
           "key": "The notice names each factor, not what was wrong with it, the most important first."},
    "nl": {"title": "Jouw brief", "head": "We kunnen je aanvraag niet goedkeuren.",
           "lead": "Belangrijkste redenen:", "empty": "Je lijst bevat geen redenen: zo'n brief zou niet volstaan.",
           "key": "De brief noemt elke factor, niet wat eraan mis was, de belangrijkste eerst."},
}


def _notice_picture(seen) -> str:
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    found = _texts(mine.value)
    if found is None or len(found) > 6:
        return ""
    w, factors = _NOTICE_WORDS[seen.lang], _FACTORS[seen.lang]
    box = ("border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);padding:0.7rem 0.9rem;"
           "background:var(--ea-page);color:var(--ea-ink);font-family:var(--ea-font-ui);font-size:0.92rem;"
           "max-width:28rem")
    if found:
        rows = "".join(f"<li>{html.escape(factors.get(name, name))}</li>" for name in found)
        body = (f"<div>{html.escape(w['head'])}</div><div style='margin-top:0.5rem'>{html.escape(w['lead'])}</div>"
                f"<ol style='margin:0.2rem 0 0 1.2rem;padding:0'>{rows}</ol>")
        key = w["key"]
    else:
        body, key = f"<div>{html.escape(w['head'])}</div>", w["empty"]
    return yours.figure(w["title"], f'<div style="{box}">{body}</div><div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "contrastive" and "court" are choice questions.
REQUIRED = ["gaps", "reasons", "contrastive", "court"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}
_KEEP = {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another applicant by itself.",
         "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere aanvrager."}

# --- exercise 1: the gaps against the applicants just above the cut-off
_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'
_X_LINE = "X = people[names].to_numpy()"
_WEIGHTS_LINE = "weights = np.array([0.046, 0.345, 0.042])"
_BIAS_LINE = "bias = -8.48"
_PERSON_LINE = "person = np.array([30, 9, 40])"
_LOGITS_LINE = "logits = (X @ weights + bias).round(3)"
_NEAR_LINE = "near = (logits >= 0) & (logits < 0.25)"
_G_HEAD = ("import numpy as np\nimport pandas as pd\n\n"
           "# the table UCI Adult, and the three columns the model uses\n" + _READ_LINE + "\n"
           'names = ["age", "education_num", "hours_per_week"]\n' + _X_LINE + "\n\n"
           "# the model of the worked example, and the applicant\n" + _WEIGHTS_LINE + "\n" + _BIAS_LINE + "\n"
           + _PERSON_LINE + "\n\n"
           "# every applicant's logit, to 3 decimals like the weights\n" + _LOGITS_LINE + "\n"
           "# True for the applicants at or just above the cut-off\n" + _NEAR_LINE + "\n\n")
_G_START = _G_HEAD + "...\ngaps\n"
_G_BODY = "reference = X[near].mean(axis=0)\ngaps = weights * (person - reference)\n"
_G_SOLUTION = _G_HEAD + _G_BODY + "gaps"


def _g(old: str, new: str) -> str:
    assert _G_BODY.count(old) == 1
    return _G_HEAD + _G_BODY.replace(old, new) + "gaps"


def _is_three(want):
    """`want` is a function, so the table is only read when a check needs it."""
    return lambda got, text: same_three(got, want())


# --- exercise 2: from gaps to reasons
_NAMES_LINE = 'names = ["age", "education_num", "hours_per_week"]'
_GAPS_LINE = "gaps = np.array([-0.789, -1.344, -0.302])"
_R_HEAD = ("import numpy as np\n\n" + _NAMES_LINE + "\n"
           "# your gaps of exercise 1: below 0, a feature weighs against the applicant\n" + _GAPS_LINE + "\n\n")
_R_START = _R_HEAD + "...\nreasons\n"
_R_BODY = ("reasons = []\nfor i in np.argsort(gaps):\n    if gaps[i] < 0:\n"
           "        reasons.append(names[i])\n")
_R_SOLUTION = _R_HEAD + _R_BODY + "reasons"
_R_OTHER = "np.array([0.223, -0.999, -0.722])"


def _r(old: str, new: str) -> str:
    assert _R_BODY.count(old) == 1
    return _R_HEAD + _R_BODY.replace(old, new) + "reasons"


def _is_names(want: list):
    return lambda got, text: _texts(got) == want


def _is_list_numbers(want: list, ints: bool):
    def match(got, text):
        if isinstance(got, np.ndarray):
            got = got.tolist()
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        try:
            values = [float(x) for x in got]
        except (TypeError, ValueError):
            return False
        if ints and not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in got):
            return False
        return all(abs(a - b) < 1e-6 for a, b in zip(values, want))
    return match


EXERCISES = {
    "gaps": CodeBox(
        name="gaps",
        picture=_gaps_picture,
        start=_G_START,
        solution=_G_SOLUTION,
        keep=[_READ_LINE, _X_LINE, _WEIGHTS_LINE, _BIAS_LINE, _PERSON_LINE, _LOGITS_LINE, _NEAR_LINE],
        compare=same_three,
        # The check also runs the box with an applicant aged 52.
        others=[{"person": Code("np.array([52, 10, 30])")}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("shape", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="Two arrays of different lengths meet in your code, so NumPy cannot pair them up. Does every array have one number per feature? A mean over the wrong axis, such as `axis=1`, gives one number per applicant; `.mean(axis=0)` gives one per column.",
                    nl="In je code komen twee arrays van verschillende lengte samen, zodat NumPy ze niet kan koppelen. Heeft elke array één getal per feature? Een gemiddelde over de verkeerde as, zoals `axis=1`, geeft één getal per aanvrager; `.mean(axis=0)` geeft er één per kolom."),
            Mistake("all-applicants", matches=_is_three(lambda: gaps(PERSON, "all")),
                    en="These are the gaps against all applicants, the method of the worked example. This exercise takes the other method: the average over the applicants just above the cut-off, `X[near]`.",
                    nl="Dit zijn de verschillen tegen alle aanvragers, de methode van het uitgewerkte voorbeeld. Deze oefening neemt de andere methode: het gemiddelde over de aanvragers net boven de grens, `X[near]`."),
            Mistake("yes-group", matches=lambda got, text: (same_three(got, gaps(PERSON, "yes"))
                                                           or same_three(got, gaps(PERSON, "yes-above"))),
                    en="These are the gaps against everyone who got a yes. The method asks for the applicants at or just above the cut-off, those in `near`: `X[near]`.",
                    nl="Dit zijn de verschillen tegen iedereen die een ja kreeg. De methode vraagt om de aanvragers op of net boven de grens, die in `near`: `X[near]`."),
            Mistake("sign", matches=_is_three(lambda: -gaps(PERSON)),
                    en="Your numbers have the right size but the opposite sign. A feature where the applicant is below the reference should give a number below 0: take the applicant's value minus the average, `person - reference`.",
                    nl="Je getallen hebben de goede grootte, maar het omgekeerde teken. Een feature waarin de aanvrager onder de referentie zit, moet een getal onder 0 geven: neem de waarde van de aanvrager min het gemiddelde, `person - reference`."),
            Mistake("no-weights", matches=_is_three(lambda: np.array(PERSON) - reference()),
                    en="Your numbers are the differences in the features themselves, in years, levels and hours. What a difference does to the logit is that difference times the weight: `weights * (person - reference)`.",
                    nl="Je getallen zijn de verschillen in de features zelf, in jaren, niveaus en uren. Wat een verschil met de logit doet, is dat verschil keer de weight: `weights * (person - reference)`."),
            Mistake("one-mean", matches=_is_three(lambda: WEIGHTS * (np.array(PERSON) - table()[group("near")].mean())),
                    en="Your reference is one number for all features: `.mean()` without an axis averages every number of every column together. Average each column on its own with `.mean(axis=0)`.",
                    nl="Je referentie is één getal voor alle features: `.mean()` zonder as neemt het gemiddelde van alle getallen van alle kolommen samen. Neem het gemiddelde van elke kolom apart met `.mean(axis=0)`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: first `reference`, the average of each column over the applicants in `near`, then `gaps`.",
                      "nl": "Vervang de `...` door je code: eerst `reference`, het gemiddelde van elke kolom over de aanvragers in `near`, dan `gaps`."},
            "cleared": _CLEARED,
            "keep": _KEEP,
            "no_value": {"en": "The last line of the box shows nothing. Put `gaps` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `gaps` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is that, for each feature, the weight times the applicant's value minus the average over the applicants in `near`?",
                      "nl": "Je laatste regel geeft `{got}`. Is dat voor elke feature de weight keer de waarde van de aanvrager min het gemiddelde over de aanvragers in `near`?"},
            "other": {"en": "Right for this applicant. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your code use `person` itself, and not the numbers 30, 9 and 40?",
                      "nl": "Goed voor deze aanvrager. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je code `person` zelf, en niet de getallen 30, 9 en 40?"},
            "other_error": {"en": "Right for this applicant. But with {where}, your code stops with an error. Does your code use `person` itself?",
                            "nl": "Goed voor deze aanvrager. Maar met {where} stopt je code met een fout. Gebruikt je code `person` zelf?"},
            "correct": {"en": "Age `-0.789`, education `-1.344`, hours `-0.302`: against the applicants just above the cut-off, education weighs most against this applicant, then age. Against all applicants, in the worked example, age came first. The two methods name a different principal reason for the same person. For the applicant aged 52 the check got `0.223`, `-0.999` and `-0.722`: age counts for that applicant, education and hours against.",
                        "nl": "Leeftijd `-0.789`, opleiding `-1.344`, uren `-0.302`: tegen de aanvragers net boven de grens weegt de opleiding het zwaarst tegen deze aanvrager, dan de leeftijd. Tegen alle aanvragers, in het uitgewerkte voorbeeld, kwam de leeftijd eerst. De twee methoden noemen voor dezelfde persoon een andere belangrijkste reden. Voor de aanvrager van 52 kreeg de check `0.223`, `-0.999` en `-0.722`: de leeftijd telt voor die aanvrager, opleiding en uren tellen tegen."},
        },
    ),
    "reasons": CodeBox(
        name="reasons",
        picture=_notice_picture,
        start=_R_START,
        solution=_R_SOLUTION,
        keep=[_NAMES_LINE, _GAPS_LINE],
        # The check also runs the box with the gaps of the applicant aged 52.
        others=[{"gaps": Code(_R_OTHER)}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("reversed", matches=_is_names(["hours_per_week", "age", "education_num"]),
                    en="Your list starts with the gap that weighs least against the applicant. The principal reason comes first: the most negative gap. `np.argsort` already goes from small to large.",
                    nl="Je lijst begint met het verschil dat het minst tegen de aanvrager weegt. De belangrijkste reden komt eerst: het meest negatieve verschil. `np.argsort` gaat al van klein naar groot."),
            Mistake("unsorted", matches=_is_names(["age", "education_num", "hours_per_week"]),
                    en="Your list follows the order of `names`. Reasons go in the order of their gaps, the most negative first: go over `np.argsort(gaps)`.",
                    nl="Je lijst volgt de volgorde van `names`. Redenen komen in de volgorde van hun verschillen, het meest negatieve eerst: loop over `np.argsort(gaps)`."),
            Mistake("positions", matches=_is_list_numbers([1, 0, 2], ints=True),
                    en="Your list holds positions, the numbers `np.argsort` gives. A notice names the features: append `names[i]`.",
                    nl="Je lijst bevat posities, de getallen die `np.argsort` geeft. Een brief noemt de features: voeg `names[i]` toe."),
            Mistake("values", matches=_is_list_numbers([-1.344, -0.789, -0.302], ints=False),
                    en="Your list holds the gaps themselves, in the right order. A notice names the features: append `names[i]`.",
                    nl="Je lijst bevat de verschillen zelf, in de goede volgorde. Een brief noemt de features: voeg `names[i]` toe."),
            Mistake("first-only", matches=_is_names(["education_num"]),
                    en="Your list holds only the principal reason. The exercise asks for every feature with a gap below 0, here three, the principal one first; a notice then gives the few that weigh most.",
                    nl="Je lijst bevat alleen de belangrijkste reden. De oefening vraagt om elke feature met een verschil onder 0, hier drie, de belangrijkste eerst; een brief geeft daarvan de paar die het zwaarst wegen."),
            Mistake("last-only", matches=_is_names(["hours_per_week"]),
                    en="Your list holds only `hours_per_week`, the gap that weighs least against the applicant. It should hold every feature with a gap below 0. If you use a loop: is `reasons = []` made once, before the loop, and is `append` inside the loop, indented under `if`?",
                    nl="Je lijst bevat alleen `hours_per_week`, het verschil dat het minst tegen de aanvrager weegt. Hij moet elke feature met een verschil onder 0 bevatten. Gebruik je een loop: maak je `reasons = []` één keer, vóór de loop, en staat `append` binnen de loop, ingesprongen onder `if`?"),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `reasons = []`, go over the gaps from small to large, and append the name of each gap below 0.",
                      "nl": "Vervang de `...` door je code: begin met `reasons = []`, loop over de verschillen van klein naar groot, en voeg de naam van elk verschil onder 0 toe."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other gaps by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere verschillen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `reasons` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `reasons` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does it hold the names of the features with a gap below 0, the most negative first?",
                      "nl": "Je laatste regel geeft `{got}`. Bevat hij de namen van de features met een verschil onder 0, het meest negatieve eerst?"},
            "other": {"en": "Right for these gaps. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your list keep only the features with a gap below 0, and does your code use `gaps` itself?",
                      "nl": "Goed voor deze verschillen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Houdt je lijst alleen de features met een verschil onder 0, en gebruikt je code `gaps` zelf?"},
            "other_error": {"en": "Right for these gaps. But with {where}, your code stops with an error. Does your code use `gaps` itself?",
                            "nl": "Goed voor deze verschillen. Maar met {where} stopt je code met een fout. Gebruikt je code `gaps` zelf?"},
            "correct": {"en": "Education first, then age, then hours: all three gaps are below 0, so all three are reasons, the principal one first. For the applicant aged 52 the check got `['education_num', 'hours_per_week']`: age was above the reference, so it is no reason. A few reasons, each measured against applicants who got a yes, is what a lender sends with a no.",
                        "nl": "Eerst de opleiding, dan de leeftijd, dan de uren: alle drie de verschillen zijn kleiner dan 0, dus alle drie zijn een reden, de belangrijkste eerst. Voor de aanvrager van 52 kreeg de check `['education_num', 'hours_per_week']`: de leeftijd lag boven de referentie, dus die is geen reden. Een paar redenen, elk gemeten tegen aanvragers die een ja kregen: dat stuurt een kredietverstrekker mee met een nee."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_G_LITERAL = _G_HEAD + "gaps = np.array([-0.789, -1.344, -0.302])\ngaps"
_R_LITERAL = _R_HEAD + 'reasons = ["education_num", "age", "hours_per_week"]\nreasons'
CASES = [
    ("gaps", _G_START, "empty", None),
    ("gaps", _G_SOLUTION, "correct", None),
    ("gaps", _G_HEAD + "gaps = weights * person - weights * X[near].mean(axis=0)\ngaps", "correct", None),
    ("gaps", _g("X[near].mean(axis=0)", "X[near].mean(axis=1)"), "error", "shape"),
    ("gaps", _g("X[near].mean(axis=0)", "X.mean(axis=0)"), "wrong", "all-applicants"),
    ("gaps", _g("X[near].mean(axis=0)", "X[logits >= 0].mean(axis=0)"), "wrong", "yes-group"),
    ("gaps", _g("X[near].mean(axis=0)", "X[logits > 0].mean(axis=0)"), "wrong", "yes-group"),
    ("gaps", _g("(person - reference)", "(reference - person)"), "wrong", "sign"),
    ("gaps", _g("weights * (person - reference)", "person - reference"), "wrong", "no-weights"),
    ("gaps", _g("X[near].mean(axis=0)", "X[near].mean()"), "wrong", "one-mean"),
    ("gaps", _g("weights * (person - reference)", "weights * person - reference"), "wrong", None),
    ("gaps", _g("person - reference", "np.array([30, 9, 40]) - reference"), "almost", "other-value"),
    ("gaps", _G_LITERAL, "almost", "other-value"),
    ("gaps", _G_SOLUTION.replace(_PERSON_LINE, "person = np.array([30, 9, 45])"), "wrong", "changed-line"),
    ("gaps", _G_SOLUTION.replace(_NEAR_LINE, "near = logits >= 0"), "wrong", "changed-line"),
    ("gaps", _G_HEAD + "...\ngaps = weights\ngaps", "wrong", "dots-left"),
    ("reasons", _R_START, "empty", None),
    ("reasons", _R_SOLUTION, "correct", None),
    ("reasons", _R_HEAD + "reasons = []\norder = np.argsort(gaps)\nfor i in order:\n    if gaps[i] < 0:\n"
                "        reasons = reasons + [names[i]]\nreasons", "correct", None),
    ("reasons", _r("np.argsort(gaps)", "np.argsort(-gaps)"), "wrong", "reversed"),
    ("reasons", _r("np.argsort(gaps)", "range(3)"), "wrong", "unsorted"),
    ("reasons", _r("reasons.append(names[i])", "reasons.append(i)"), "wrong", "positions"),
    ("reasons", _r("reasons.append(names[i])", "reasons.append(gaps[i])"), "wrong", "values"),
    ("reasons", _R_HEAD + "reasons = [names[np.argmin(gaps)]]\nreasons", "wrong", "first-only"),
    ("reasons", _R_HEAD + "for i in np.argsort(gaps):\n    reasons = []\n    if gaps[i] < 0:\n"
                "        reasons.append(names[i])\nreasons", "wrong", "last-only"),
    ("reasons", _R_HEAD + "reasons = []\nfor i in np.argsort(gaps):\n    if gaps[i] < 0:\n        name = names[i]\n"
                "reasons.append(name)\nreasons", "wrong", "last-only"),
    ("reasons", _r("    if gaps[i] < 0:\n        reasons", "    reasons"), "almost", "other-value"),
    ("reasons", _R_LITERAL, "almost", "other-value"),
    ("reasons", _R_SOLUTION.replace(_GAPS_LINE, "gaps = np.array([-0.789, -1.344, 0.302])"), "wrong", "changed-line"),
    ("reasons", _R_HEAD + "...\nreasons = []\nreasons", "wrong", "dots-left"),
]
