# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-glass-box-models: two glass boxes on three columns of UCI Adult, a logistic regression and
a decision tree of depth 2. The learner writes the tree as a function by hand, works out how much four levels of
education multiply the odds, and sorts models by whether a person can follow them; with the picture of the whole tree,
of the learner's function over age and education, and of what their odds factor does to a probability."""
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, Sort

COLUMNS = ["age", "education_num", "hours_per_week"]
# The logistic regression of rai-reason-codes, rounded, and the applicant of that lesson.
WEIGHTS = [0.046, 0.345, 0.042]
BIAS = -8.48
PERSON = (30, 9, 40)

# The tree of the worked example: DecisionTreeClassifier(max_depth=2, random_state=0) on all 48,842 rows, as
# export_text prints it (tests/test_rai_glass_box_models.py grows it again).
TREE_TEXT = """|--- education_num <= 12.50
|   |--- age <= 30.50
|   |   |--- class: False
|   |--- age >  30.50
|   |   |--- class: False
|--- education_num >  12.50
|   |--- age <= 29.50
|   |   |--- class: False
|   |--- age >  29.50
|   |   |--- class: True"""
# Each leaf, left to right: (people in it, share of them who earn more than 50K).
LEAVES = [(13033, 0.040), (23699, 0.226), (2412, 0.140), (9698, 0.565)]


def tree_says(age, education_num, hours_per_week) -> bool:
    """The tree of the worked example, followed by hand."""
    if education_num <= 12.5:
        return False
    return age > 29.5


def odds_factor(weights, levels: int = 4) -> float:
    """How many times the odds grow when education goes up by `levels` levels."""
    return math.exp(weights[1] * levels)


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _count(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

_TREE_WORDS = {
    "en": {"no": "no", "yes": "yes", "people": "{n} people", "share": "{p}% earn more",
           "alt": "The decision tree drawn whole. The first question: education_num at most 12.5? If so, it asks age at "
                  "most 30.5, and both leaves say no. If not, it asks age at most 29.5: younger says no, older says yes. "
                  "Each leaf gives how many people end up in it and what share of them earns more than 50,000 dollars.",
           "key": "Left of each question is yes, right is no. Hours are never asked."},
    "nl": {"no": "nee", "yes": "ja", "people": "{n} mensen", "share": "{p}% verdient meer",
           "alt": "De beslisboom helemaal getekend. De eerste vraag: education_num hoogstens 12,5? Zo ja, dan vraagt hij "
                  "age hoogstens 30,5, en beide bladeren zeggen nee. Zo nee, dan vraagt hij age hoogstens 29,5: jonger "
                  "zegt nee, ouder zegt ja. Elk blad geeft hoeveel mensen erin terechtkomen en welk deel van hen meer "
                  "dan 50.000 dollar verdient.",
           "key": "Links van elke vraag is ja, rechts is nee. Naar de uren vraagt hij nooit."},
}


def _box(x, y, w, h, lines, wash=False) -> str:
    fill = "var(--ea-pred-wash)" if wash else "var(--ea-page)"
    stroke = "var(--ea-pred)" if wash else "var(--ea-line)"
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
           f'stroke-width="1.2"/>')
    step = h / (len(lines) + 1)
    for i, (words, colour) in enumerate(lines):
        out += (f'<text x="{x + w / 2}" y="{y + step * (i + 1) + 4}" text-anchor="middle" fill="var({colour})" '
                f'font-family="var(--ea-font-ui)" font-size="12">{html.escape(words)}</text>')
    return out


def _edge(x0, y0, x1, y1) -> str:
    return f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="var(--ea-axis)" stroke-width="1.2"/>'


def whole_tree(lang: str) -> str:
    """The picture of *The idea*: the tree of the worked example, every question and every leaf."""
    w = _TREE_WORDS[lang]
    sep = "," if lang == "nl" else "."
    svg = ['<svg viewBox="0 0 640 250" width="100%" style="max-width:640px" role="img" '
           f'aria-label="{html.escape(w["alt"])}">']
    svg.append(_edge(320, 46, 160, 96) + _edge(320, 46, 480, 96))
    svg.append(_box(220, 10, 200, 36, [(f"education_num ≤ 12{sep}5 ?", "--ea-ink")]))
    for cx, cut in ((160, f"30{sep}5"), (480, f"29{sep}5")):
        svg.append(_edge(cx, 132, cx - 80, 170) + _edge(cx, 132, cx + 80, 170))
        svg.append(_box(cx - 80, 96, 160, 36, [(f"age ≤ {cut} ?", "--ea-ink")]))
    for i, (n, share) in enumerate(LEAVES):
        cx = [80, 240, 400, 560][i]
        yes = share > 0.5
        svg.append(_box(cx - 72, 170, 144, 70, [
            (w["yes"] if yes else w["no"], "--ea-pred-text" if yes else "--ea-ink"),
            (w["people"].format(n=_count(n, lang)), "--ea-ink-2"),
            (w["share"].format(p=_show(share * 100, lang, 1)), "--ea-ink-2")], wash=yes))
    svg.append("</svg>")
    return "".join(svg) + f'<div style="{_UI}">{html.escape(w["key"])}</div>'


def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_GRID_WORDS = {
    "en": {"title": "Your says_yes, over age and education", "x": "age", "y": "education_num",
           "alt": "A grid of ages from 17 to 90 and education levels from 1 to 16, with hours at 40. Your function says "
                  "yes in {yes} of the {n} squares{same}.",
           "same": ", the same squares as the tree", "other": "; the tree says yes in {tree}",
           "key": "A filled square: your function says yes there, for 40 hours a week. {verdict}",
           "agree": "On all 48,842 people of the table, your function says what the tree says.",
           "differ": "On the 48,842 people of the table, your function differs from the tree for {d} of them."},
    "nl": {"title": "Jouw says_yes, over leeftijd en opleiding", "x": "age", "y": "education_num",
           "alt": "Een rooster van leeftijden van 17 tot 90 en opleidingsniveaus van 1 tot 16, met 40 uur per week. Je "
                  "functie zegt ja in {yes} van de {n} vakjes{same}.",
           "same": ", dezelfde vakjes als de boom", "other": "; de boom zegt ja in {tree}",
           "key": "Een gevuld vakje: daar zegt je functie ja, bij 40 uur per week. {verdict}",
           "agree": "Bij alle 48.842 mensen van de tabel zegt je functie wat de boom zegt.",
           "differ": "Bij de 48.842 mensen van de tabel zegt je functie bij {d} iets anders dan de boom."},
}

_TABLE = {}


def _people() -> list:
    """The three columns of UCI Adult as distinct rows with their counts, read once."""
    if "rows" not in _TABLE:
        import pandas as pd
        from everything_ai import paths
        table = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")[COLUMNS]
        counts = table.value_counts()
        _TABLE["rows"] = [(tuple(int(v) for v in k), int(c)) for k, c in counts.items()]
    return _TABLE["rows"]


def _says(fn, age, edu, hours):
    got = fn(age, edu, hours)
    if isinstance(got, bool):
        return got
    if hasattr(got, "item") and type(got.item()) is bool:   # a NumPy True or False
        return got.item()
    raise TypeError("not True or False")


def _follow_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    w = _GRID_WORDS[seen.lang]
    ages, levels = range(17, 91), range(1, 17)

    def run():
        grid = {(a, e): _says(fn, a, e, 40) for a in ages for e in levels}
        differ = sum(c for (a, e, h), c in _people() if _says(fn, a, e, h) != tree_says(a, e, h))
        return grid, differ

    from everything_ai import check as checking
    try:
        grid, differ = checking.with_time_limit(run)
    except Exception:
        return ""
    p = chart.Panel(x=(16.5, 90.5), y=(0.5, 16.5), title="", xlabel=w["x"], ylabel=w["y"],
                    xticks=[20, 30, 40, 50, 60, 70, 80, 90], yticks=[1, 4, 8, 12, 16])
    for e in levels:   # one bar per run of yes in a row
        a = 17
        while a <= 90:
            if grid[(a, e)]:
                b = a
                while b + 1 <= 90 and grid[(b + 1, e)]:
                    b += 1
                p.rect(a - 0.5, e - 0.45, b + 0.5, e + 0.45, role="pred", solid=True)
                a = b + 1
            else:
                a += 1
    p.line([29.5, 29.5, 90.5], [16.5, 12.5, 12.5], role="ink-3", width=1, dashed=True)
    yes, tree = sum(grid.values()), sum(tree_says(a, e, 40) for a in ages for e in levels)
    same = all(grid[k] == tree_says(*k, 40) for k in grid)
    fill = {"yes": yes, "n": len(grid), "same": w["same"] if same else w["other"].format(tree=tree)}
    verdict = w["agree"] if differ == 0 else w["differ"].format(d=_count(differ, seen.lang))
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=520, height=300)
    key = w["key"].format(verdict=verdict)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


_ODDS_WORDS = {
    "en": {"title": "What your factor does", "odds": "odds of a yes", "nine": "level 9", "thirteen": "level 13",
           "alt": "Two bars: the odds of a yes of the applicant aged 30, {o9} at education level 9, and {o13} at level 13, "
                  "your factor {f} times as large.",
           "key": "For the applicant aged 30 who works 40 hours, your factor {f} takes the odds from {o9} to {o13}, so "
                  "the probability goes from {p9} to {p13}: {r} times as large, not {f}."},
    "nl": {"title": "Wat jouw factor doet", "odds": "odds op een ja", "nine": "niveau 9", "thirteen": "niveau 13",
           "alt": "Twee staven: de odds op een ja van de aanvrager van 30, {o9} bij opleidingsniveau 9 en {o13} bij "
                  "niveau 13, jouw factor {f} keer zo groot.",
           "key": "Voor de aanvrager van 30 die 40 uur werkt, brengt jouw factor {f} de odds van {o9} naar {o13}, dus "
                  "de kans gaat van {p9} naar {p13}: {r} keer zo groot, niet {f}."},
}


def _odds_picture(seen) -> str:
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    v = mine.value
    if isinstance(v, bool) or not isinstance(v, (int, float)) and not hasattr(v, "item"):
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not (0 < f <= 100):
        return ""
    w = _ODDS_WORDS[seen.lang]
    z = BIAS + sum(wt * x for wt, x in zip(WEIGHTS, PERSON))
    o9 = math.exp(z)
    o13 = o9 * f
    p9, p13 = o9 / (1 + o9), o13 / (1 + o13)
    top = max(o13, o9) * 1.25
    p = chart.Panel(x=(0, 2), y=(0, top), title="", xlabel="", ylabel=w["odds"], xticks=[0.5, 1.5],
                    xtick_labels=[w["nine"], w["thirteen"]], yticks=[])
    for i, o in enumerate((o9, o13)):
        p.rect(i + 0.25, 0, i + 0.75, o, role="pred", solid=True)
        p.text(i + 0.5, o + top * 0.06, _show(o, seen.lang), "ink")
    fill = {"f": _show(f, seen.lang, 2), "o9": _show(o9, seen.lang), "o13": _show(o13, seen.lang),
            "p9": _show(p9, seen.lang), "p13": _show(p13, seen.lang), "r": _show(p13 / p9, seen.lang, 2)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=360, height=240)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# The lesson is done when these are correct. "glass" is the sort question, "scale" a choice question.
REQUIRED = ["follow", "odds", "glass", "scale"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1: the tree, followed by hand
_F_HEAD = ("# the tree of the worked example, as export_text prints it:\n"
           + "".join(f"# {line}\n" for line in TREE_TEXT.splitlines()) + "\n"
           "def says_yes(age, education_num, hours_per_week):\n")
_F_START = _F_HEAD + "    ...\n"
_F_BODY = ("    if education_num <= 12.5:\n"
           "        if age <= 30.5:\n"
           "            return False\n"
           "        else:\n"
           "            return False\n"
           "    else:\n"
           "        if age <= 29.5:\n"
           "            return False\n"
           "        else:\n"
           "            return True\n")
_F_SOLUTION = _F_HEAD + _F_BODY


def _f(body: str) -> str:
    return _F_HEAD + body


# The applicant of rai-reason-codes, then every side of every threshold, and people far from them.
_F_HIDDEN = [(29, 13, 40), (30, 13, 40), (31, 13, 40), (30, 12, 40), (31, 12, 60), (52, 10, 30), (52, 16, 30),
             (25, 14, 60), (70, 13, 10), (45, 9, 80), (18, 1, 20), (90, 16, 99)]

# --- exercise 2: four levels of education, in odds
_W_LINE = "weights = np.array([0.046, 0.345, 0.042])"
_O_HEAD = ("import numpy as np\n\n"
           "# the weights of age, education_num and hours_per_week in the logistic regression\n" + _W_LINE + "\n\n")
_O_START = _O_HEAD + "...\nfactor\n"
_O_SOLUTION = _O_HEAD + "factor = np.exp(weights[1] * 4)\nfactor"


def _near(*wants: float):
    def match(got, text):
        if isinstance(got, bool) or not isinstance(got, (int, float)) and not hasattr(got, "item"):
            return False
        try:
            v = float(got)
        except (TypeError, ValueError):
            return False
        return any(abs(v - w) < 1e-3 for w in wants)
    return match


EXERCISES = {
    "follow": FunctionBox(
        name="follow",
        picture=_follow_picture,
        function="says_yes",
        start=_F_START,
        solution=_F_SOLUTION,
        visible=[PERSON],
        hidden=_F_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("or", wrong=_f("    return education_num > 12.5 or age > 29.5\n"),
                    en="`{call}` gives {got}, and the tree gives {want}. Your function says yes when one of the two holds. The tree says yes only at the end of one path: education above 12.5, and then age above 29.5.",
                    nl="`{call}` geeft {got}, en de boom geeft {want}. Je functie zegt ja als een van de twee klopt. De boom zegt alleen ja aan het eind van één pad: opleiding boven 12,5, en dan leeftijd boven 29,5."),
            Mistake("threshold", wrong=_f("    return education_num > 12.5 and age > 30.5\n"),
                    en="`{call}` gives {got}, and the tree gives {want}. On the right, after `education_num >  12.50`, the tree asks about age with its own threshold: 29.5, not the 30.5 of the left side.",
                    nl="`{call}` geeft {got}, en de boom geeft {want}. Rechts, na `education_num >  12.50`, vraagt de boom naar de leeftijd met een eigen grens: 29,5, niet de 30,5 van de linkerkant."),
            Mistake("flipped", wrong=_f("    return not (education_num > 12.5 and age > 29.5)\n"),
                    en="`{call}` gives {got}, and the tree gives {want}: your function says the opposite of the tree everywhere. In the printed tree, a `<=` line leads to the part where the question holds, a `>` line to the other part.",
                    nl="`{call}` geeft {got}, en de boom geeft {want}: je functie zegt overal het omgekeerde van de boom. In de geprinte boom leidt een regel met `<=` naar het deel waar de vraag klopt, een regel met `>` naar het andere deel."),
            Mistake("always-no", wrong=_f("    return False\n"),
                    en="`{call}` gives {got}, and the tree gives {want}. Three of the four leaves say `False`, but one says `True`: education above 12.5 and age above 29.5.",
                    nl="`{call}` geeft {got}, en de boom geeft {want}. Drie van de vier bladeren zeggen `False`, maar één zegt `True`: opleiding boven 12,5 en leeftijd boven 29,5."),
            Mistake("text", matches=lambda got, text: isinstance(got, str),
                    en="`{call}` gives the text {got}. A leaf of the tree says `class: True` or `class: False`: return `True` or `False` itself, without quotes.",
                    nl="`{call}` geeft de tekst {got}. Een blad van de boom zegt `class: True` of `class: False`: geef `True` of `False` zelf terug, zonder aanhalingstekens."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the tree: ask `education_num <= 12.5` with `if`, and in each part the question about `age`, each leaf a `return True` or `return False`.",
                      "nl": "Vervang de `...` door de boom: vraag `education_num <= 12.5` met `if`, en in elk deel de vraag over `age`, elk blad een `return True` of `return False`."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and the tree gives {want}. Follow that person down the printed tree: which branch does each question send them to?",
                      "nl": "`{call}` geeft {got}, en de boom geeft {want}. Volg die persoon door de geprinte boom: naar welke tak stuurt elke vraag die persoon?"},
            "almost": {"en": "Right for `says_yes(30, 9, 40)`. But `{call}` gives {got}, and the tree gives {want}. Check the threshold of each question in the printed tree, and which side of each says `True`.",
                       "nl": "Goed voor `says_yes(30, 9, 40)`. Maar `{call}` geeft {got}, en de boom geeft {want}. Kijk naar de grens van elke vraag in de geprinte boom, en welke kant van elke vraag `True` zegt."},
            "correct": {"en": "Your `says_yes` gives what the tree gives in all {n} cases, and the picture tries it on all 48,842 people of the table. The whole model fits in a few lines you can read: only people with education above 12.5 who are older than 29.5 get a yes. The left side asks about age, but both its leaves say no, and the tree never asks about hours.",
                        "nl": "Je `says_yes` geeft in alle {n} gevallen wat de boom geeft, en de tekening probeert hem op alle 48.842 mensen van de tabel. Het hele model past in een paar regels die je kunt lezen: alleen mensen met een opleiding boven 12,5 die ouder zijn dan 29,5, krijgen een ja. De linkerkant vraagt naar de leeftijd, maar beide bladeren zeggen nee, en naar de uren vraagt de boom nooit."},
        },
    ),
    "odds": CodeBox(
        name="odds",
        picture=_odds_picture,
        start=_O_START,
        solution=_O_SOLUTION,
        keep=[_W_LINE],
        # The check also tries other weights.
        others=[{"weights": Code("np.array([0.05, 0.2, 0.04])")}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("times-four", matches=_near(math.exp(0.345) * 4),
                    en="Your factor is the factor of one level times 4. But each level multiplies the odds by that factor, so four levels multiply four times: the factor to the power 4, or `np.exp` of four times the weight.",
                    nl="Je factor is de factor van één niveau keer 4. Maar elk niveau vermenigvuldigt de odds met die factor, dus vier niveaus vermenigvuldigen vier keer: de factor tot de macht 4, of `np.exp` van vier keer de weight."),
            Mistake("no-exp", matches=_near(0.345 * 4),
                    en="Your number, `{got}`, is what four levels add to the logit, the log of the odds. The odds themselves change by the exponent of that: `np.exp(...)`.",
                    nl="Je getal, `{got}`, is wat vier niveaus optellen bij de logit, de log van de odds. De odds zelf veranderen met de e-macht daarvan: `np.exp(...)`."),
            Mistake("one-level", matches=_near(math.exp(0.345)),
                    en="Your factor is that of one level of education. From level 9 to level 13 is four levels.",
                    nl="Je factor is die van één opleidingsniveau. Van niveau 9 naar niveau 13 zijn vier niveaus."),
            Mistake("one-number", matches=lambda got, text: hasattr(got, "shape") and getattr(got, "shape", ()) == (3,),
                    en="Your last line gives three numbers, one per feature. The question is about education only, `weights[1]`: one number.",
                    nl="Je laatste regel geeft drie getallen, één per feature. De vraag gaat alleen over opleiding, `weights[1]`: één getal."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: `factor`, how many times the odds of a yes grow from education level 9 to 13.",
                      "nl": "Vervang de `...` door je code: `factor`, hoeveel keer de odds op een ja groeien van opleidingsniveau 9 naar 13."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other weights by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere weights."},
            "no_value": {"en": "The last line of the box shows nothing. Put `factor` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `factor` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Each level of education multiplies the odds by `np.exp` of its weight; how many times is that from level 9 to 13?",
                      "nl": "Je laatste regel geeft `{got}`. Elk opleidingsniveau vermenigvuldigt de odds met `np.exp` van zijn weight; hoe vaak gebeurt dat van niveau 9 naar 13?"},
            "other": {"en": "Right for these weights. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your code use `weights` itself, and not the number `0.345`?",
                      "nl": "Goed voor deze weights. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je code `weights` zelf, en niet het getal `0.345`?"},
            "other_error": {"en": "Right for these weights. But with {where}, your code stops with an error. Does your code use `weights` itself?",
                            "nl": "Goed voor deze weights. Maar met {where} stopt je code met een fout. Gebruikt je code `weights` zelf?"},
            "correct": {"en": "`3.975`: with the same age and hours, someone with a bachelor's degree has about four times the odds of someone with a high-school diploma. That is one level's factor, `1.412`, four times over: `1.412 ** 4`. The picture shows what it does to a probability.",
                        "nl": "`3.975`: bij dezelfde leeftijd en uren heeft iemand met een bachelor ongeveer vier keer de odds van iemand met een middelbareschooldiploma. Dat is de factor van één niveau, `1.412`, vier keer achter elkaar: `1.412 ** 4`. De tekening laat zien wat het met een kans doet."},
        },
    ),
    "glass": Sort(
        name="glass",
        groups={
            "follow": {"en": "A person can follow it", "nl": "Een mens kan het volgen"},
            "not": {"en": "Not as it stands", "nl": "Niet zoals het nu is"},
        },
        items={
            "tree2": {"en": "**A short tree.** A decision tree of depth 2 on age and education.",
                      "nl": "**Een korte boom.** Een beslisboom met diepte 2 op leeftijd en opleiding."},
            "logit3": {"en": "**Three named weights.** A logistic regression on age, education and hours.",
                       "nl": "**Drie weights met een naam.** Een logistische regressie op leeftijd, opleiding en uren."},
            "points": {"en": "**Points.** One point per missed payment, one per credit card; at 3 points or more, no.",
                       "nl": "**Punten.** Eén punt per gemiste betaling, één per creditcard; bij 3 punten of meer: nee."},
            "tree20": {"en": "**A deep tree.** A decision tree of depth 20.",
                       "nl": "**Een diepe boom.** Een beslisboom met diepte 20."},
            "words": {"en": "**5,000 weights.** A logistic regression with a weight for each of 5,000 words.",
                      "nl": "**5000 weights.** Een logistische regressie met een weight voor elk van 5000 woorden."},
            "anonymous": {"en": "**Four made-up columns.** A logistic regression on the columns f1 to f4, which a program made from 300 others.",
                          "nl": "**Vier gemaakte kolommen.** Een logistische regressie op de kolommen f1 tot en met f4, die een programma maakte uit 300 andere."},
            "network": {"en": "**A small network.** A neural network with about 4,000 weights.",
                        "nl": "**Een klein netwerk.** Een neural network met ongeveer 4000 weights."},
        },
        solution={"tree2": "follow", "logit3": "follow", "points": "follow", "tree20": "not", "words": "not",
                  "anonymous": "not", "network": "not"},
        nudge={
            "tree2": {"en": "How many leaves does a tree of depth 2 have at most?", "nl": "Hoeveel bladeren heeft een boom met diepte 2 hoogstens?"},
            "logit3": {"en": "Three weights, each for a column whose meaning you know: can you work out a logit by hand?", "nl": "Drie weights, elk voor een kolom waarvan je weet wat hij betekent: kun je een logit met de hand uitrekenen?"},
            "points": {"en": "Can you add up the points of one applicant in your head?", "nl": "Kun je de punten van één aanvrager in je hoofd optellen?"},
            "tree20": {"en": "A tree of depth 20 can have 2 to the power 20 leaves. Can a person hold that?", "nl": "Een boom met diepte 20 kan 2 tot de macht 20 bladeren hebben. Kan een mens dat overzien?"},
            "words": {"en": "Each weight is readable on its own. But can a person step through 5,000 of them?", "nl": "Elke weight is op zich te lezen. Maar kan een mens er 5000 doorlopen?"},
            "anonymous": {"en": "Four weights are few. But what does a weight of f3 tell you?", "nl": "Vier weights zijn weinig. Maar wat vertelt een weight van f3 je?"},
            "network": {"en": "Can a person step through 4,000 multiplications, and does each weight mean something on its own?", "nl": "Kan een mens 4000 vermenigvuldigingen doorlopen, en betekent elke weight op zich iets?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the models again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de modellen opnieuw uit."},
            "empty": {"en": "Put each model in the group that fits it.",
                      "nl": "Zet elk model in de groep die erbij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The model *{item}* fits the other group.",
                      "nl": "Het model *{item}* past in de andere groep."},
            "correct": {"en": "All {n} are right. The short tree, the three named weights and the points are small, and each part means something: a person can step through a prediction. The deep tree and the 5,000 weights are too big to follow, the made-up columns mean nothing on their own, and the network has thousands of weights. A logistic regression can be on either side: what counts is its size and its features, as Lipton wrote.",
                        "nl": "Alle {n} kloppen. De korte boom, de drie weights met een naam en de punten zijn klein, en elk deel betekent iets: een mens kan een voorspelling stap voor stap volgen. De diepe boom en de 5000 weights zijn te groot om te volgen, de gemaakte kolommen betekenen op zich niets, en het netwerk heeft duizenden weights. Een logistische regressie kan aan beide kanten staan: het gaat om de grootte en de features, zoals Lipton schreef."},
        },
    ),
}

_SORTED = {"tree2": "follow", "logit3": "follow", "points": "follow", "tree20": "not", "words": "not",
           "anonymous": "not", "network": "not"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("follow", _F_START, "empty", None),
    ("follow", _F_SOLUTION, "correct", None),
    ("follow", _f("    return education_num > 12.5 and age > 29.5\n"), "correct", None),
    ("follow", _f("    if education_num >= 13 and age >= 30:\n        return True\n    return False\n"), "correct", None),
    ("follow", _f("    return education_num > 12.5 or age > 29.5\n"), "wrong", "or"),
    ("follow", _f(_F_BODY.replace("age <= 29.5", "age <= 30.5")), "wrong", "threshold"),
    ("follow", _f(_F_BODY.replace("return True", "return X").replace("return False", "return True")
               .replace("return X", "return False")), "wrong", "flipped"),
    ("follow", _f("    return False\n"), "wrong", "always-no"),
    ("follow", _f(_F_BODY.replace("return True", 'return "True"').replace("return False", 'return "False"')),
     "wrong", "text"),
    ("follow", _f(_F_BODY.replace("        if age <= 30.5:\n            return False\n        else:\n"
                                  "            return False\n", "        if age <= 30.5:\n            return False\n"
                                  "        else:\n            return True\n")), "almost", None),
    ("follow", _f("    if education_num <= 12.5:\n        return False\n    if age <= 29.5:\n        return False\n"),
     "wrong", "unreached"),
    ("follow", _F_HEAD + "    ...\n    return False\n", "wrong", "dots-left"),
    ("odds", _O_START, "empty", None),
    ("odds", _O_SOLUTION, "correct", None),
    ("odds", _O_HEAD + "factor = np.exp(weights[1]) ** 4\nfactor", "correct", None),
    ("odds", _O_HEAD + "factor = np.exp(weights[1] * (13 - 9))\nfactor", "correct", None),
    ("odds", _O_HEAD + "factor = np.exp(weights[1]) * 4\nfactor", "wrong", "times-four"),
    ("odds", _O_HEAD + "factor = weights[1] * 4\nfactor", "wrong", "no-exp"),
    ("odds", _O_HEAD + "factor = np.exp(weights[1])\nfactor", "wrong", "one-level"),
    ("odds", _O_HEAD + "factor = np.exp(weights * 4)\nfactor", "wrong", "one-number"),
    ("odds", _O_HEAD + "factor = np.exp(0.345 * 4)\nfactor", "almost", "other-value"),
    ("odds", _O_HEAD + "factor = np.exp(weights[0] * 4)\nfactor", "wrong", None),
    ("odds", _O_SOLUTION.replace(_W_LINE, "weights = np.array([0.046, 0.4, 0.042])"), "wrong", "changed-line"),
    ("odds", _O_HEAD + "...\nfactor = 4\nfactor", "wrong", "dots-left"),
    ("glass", {}, "empty", None),
    ("glass", _SORTED, "correct", None),
    ("glass", {"tree2": "follow", "tree20": "not"}, "wrong", "unfinished"),
    ("glass", {**_SORTED, "anonymous": "follow"}, "wrong", "wrong:anonymous"),
    ("glass", {**_SORTED, "words": "follow"}, "wrong", "wrong:words"),
    ("glass", {**_SORTED, "logit3": "not"}, "wrong", "wrong:logit3"),
    ("glass", {**_SORTED, "network": "follow"}, "wrong", "wrong:network"),
]
