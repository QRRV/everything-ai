# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-kinds-of-explanation: in a loan model the course made up, a logistic regression on three
features, the learner works out what each feature adds to one applicant's logit, a local explanation, with its picture;
and sorts explanations by where they come from and what they cover."""
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, Mistake, Sort

# The loan model of the lesson: a logistic regression the course made up. Income in thousands of euros a month,
# missed payments in the last year, years at the current job.
FEATURES = ["income", "missed_payments", "years_at_job"]
WEIGHTS = {"income": 0.8, "missed_payments": -1.2, "years_at_job": 0.3}
BIAS = -3.0
APPLICANT_A = {"income": 3.5, "missed_payments": 2, "years_at_job": 1}
APPLICANT_B = {"income": 2.0, "missed_payments": 0, "years_at_job": 6}


def contributions(person: dict) -> dict:
    """What each feature adds to the logit of `person`: its weight times its value."""
    out = {}
    for feature in FEATURES:
        out[feature] = WEIGHTS[feature] * person[feature]
    return out


def logit(person: dict) -> float:
    return BIAS + sum(contributions(person).values())


def probability(person: dict) -> float:
    return 1 / (1 + math.exp(-logit(person)))


def _number(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _per_feature(got) -> dict | None:
    """A number under each name, from a dict, as {name: number}; None otherwise."""
    if type(got) is not dict:
        return None
    out = {}
    for k, v in got.items():
        n = _number(v)
        if n is None:
            return None
        out[str(k)] = n
    return out


def same_numbers(got, want) -> bool:
    """The same number under each name, whatever the order of the names."""
    mine, want = _per_feature(got), _per_feature(want)
    return (mine is not None and want is not None and mine.keys() == want.keys()
            and all(abs(mine[k] - want[k]) < 1e-6 for k in want))


def _running(person: dict, start: float) -> dict:
    """The contributions added up as they go, from `start`: what a loop gives that stores the running total."""
    out, total = {}, start
    for feature in FEATURES:
        total = total + WEIGHTS[feature] * person[feature]
        out[feature] = total
    return out


def _show(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _bar_panel(values: dict, lo: float, hi: float, lang: str, xlabel: str, title: str = "") -> chart.Panel:
    """A bar per feature from 0, to the right for a positive number and to the left for a negative one; the name and
    the number above each bar."""
    height = 1.1 * len(values) + 0.5
    ticks = [t for t in range(math.ceil(lo), math.floor(hi) + 1)]
    p = chart.Panel(x=(lo, hi), y=(0, height), title=title, xlabel=xlabel, yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, lang) for t in ticks])
    p.line([0, 0], [0, height - 0.2], role="ink-3", width=1)
    for i, (name, v) in enumerate(values.items()):
        y = height - 0.5 - 1.1 * (i + 1)
        p.text(lo + (hi - lo) * 0.01, y + 0.68, f"{name}: {_show(v, lang)}", "ink", anchor="start", halo=True)
        x0, x1 = (0, v) if v >= 0 else (v, 0)
        p.rect(x0, y + 0.12, max(x1, x0 + (hi - lo) * 0.004), y + 0.42, role="param", solid=True)
    return p


_TWO_WORDS = {
    "en": {"weights": "the weights: the same for everyone", "weight": "weight",
           "mine": "weight × value, for applicant B", "added": "added to the logit",
           "alt": "Two bar charts. Left, the weights of the loan model: income 0.8, missed_payments −1.2, years_at_job "
                  "0.3. Right, what each feature adds to the logit of applicant B: income 1.6, missed_payments 0, "
                  "years_at_job 1.8.",
           "key": "Left a global explanation, right a local one; to the right of 0 a feature pushes toward yes, to "
                  "the left toward no."},
    "nl": {"weights": "de weights: voor iedereen hetzelfde", "weight": "weight",
           "mine": "weight × waarde, voor aanvrager B", "added": "opgeteld bij de logit",
           "alt": "Twee staafdiagrammen. Links de weights van het kredietmodel: income 0,8, missed_payments −1,2, "
                  "years_at_job 0,3. Rechts wat elke feature bijdraagt aan de logit van aanvrager B: income 1,6, "
                  "missed_payments 0, years_at_job 1,8.",
           "key": "Links een global explanation, rechts een local explanation; rechts van 0 duwt een feature richting "
                  "ja, links richting nee."},
}


def two_explanations(lang: str) -> str:
    """The picture of *The idea*: the weights of the loan model next to what they add to one applicant's logit."""
    w = _TWO_WORDS[lang]
    left = _bar_panel(WEIGHTS, -2, 2, lang, w["weight"], w["weights"])
    right = _bar_panel(contributions(APPLICANT_B), -3, 3, lang, w["added"], w["mine"])
    svg = chart.figure([left, right], alt=w["alt"], width=340, height=230)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>'


_MINE_WORDS = {
    "en": {"title": "Your contributions for applicant A", "added": "added to the logit",
           "alt": "Your contributions as bars: {shown}. With the bias {bias} they add up to the logit {logit}.",
           "key": "a bar per feature, its weight times A's value; to the right of 0 it pushes toward yes, to the left "
                  "toward no. With the bias {bias}, your numbers add up to the logit {logit}."},
    "nl": {"title": "Jouw bijdragen voor aanvrager A", "added": "opgeteld bij de logit",
           "alt": "Jouw bijdragen als staven: {shown}. Met de bias {bias} tellen ze op tot de logit {logit}.",
           "key": "een staaf per feature, de weight keer de waarde van A; rechts van 0 duwt hij richting ja, links "
                  "richting nee. Met de bias {bias} tellen jouw getallen op tot de logit {logit}."},
}


def _contributions_picture(seen) -> str:
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    values = _per_feature(mine.value)
    if not values or len(values) > 6 or not all(abs(v) <= 50 for v in values.values()):
        return ""
    w = _MINE_WORDS[seen.lang]
    reach = max([3.0] + [abs(v) * 1.1 for v in values.values()])
    p = _bar_panel(values, -reach, reach, seen.lang, w["added"])
    total = BIAS + sum(values.values())
    shown = "; ".join(f"{k} {_show(v, seen.lang)}" for k, v in values.items())
    fill = {"shown": shown, "bias": _show(BIAS, seen.lang), "logit": _show(total, seen.lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=520, height=46 * len(values) + 100)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# The lesson is done when these are correct. "kinds", "agnostic" and "opacity" are the sort and choice questions.
REQUIRED = ["contributions", "kinds", "agnostic", "opacity"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_WEIGHTS_LINE = 'weights = {"income": 0.8, "missed_payments": -1.2, "years_at_job": 0.3}'
_PERSON_LINE = 'person = {"income": 3.5, "missed_payments": 2, "years_at_job": 1}'
_C_HEAD = ("# the loan model of the worked example\n" + _WEIGHTS_LINE + "\n"
           + "# applicant A\n" + _PERSON_LINE + "\n\n")
_C_START = _C_HEAD + "...\ncontributions\n"
_C_BODY = ("contributions = {}\nfor feature in weights:\n"
           "    contributions[feature] = weights[feature] * person[feature]\n")
_C_SOLUTION = _C_HEAD + _C_BODY + "contributions"


def _c(old: str, new: str) -> str:
    assert _C_BODY.count(old) == 1
    return _C_HEAD + _C_BODY.replace(old, new) + "contributions"


def _is(want: dict):
    return lambda got, text: same_numbers(got, want)


def _is_number(*wants: float):
    def match(got, text):
        n = _number(got) if not isinstance(got, (dict, list, tuple)) else None
        return n is not None and any(abs(n - w) < 1e-6 for w in wants)
    return match


EXERCISES = {
    "contributions": CodeBox(
        name="contributions",
        picture=_contributions_picture,
        start=_C_START,
        solution=_C_SOLUTION,
        keep=[_WEIGHTS_LINE, _PERSON_LINE],
        compare=same_numbers,
        # The check also runs the box with applicant B.
        others=[{"person": APPLICANT_B}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a dict for a key it does not have. Write the names as the dicts do, such as `\"missed_payments\"`, go over the keys of `weights` with `for feature in weights:`, and store each number with `contributions[feature] = ...`, not `+=`: a new key has no number yet to add to.",
                    nl="Je code vraagt een dict om een sleutel die er niet in zit. Schrijf de namen zoals de dicts ze schrijven, zoals `\"missed_payments\"`, loop over de sleutels van `weights` met `for feature in weights:`, en zet elk getal in de dict met `contributions[feature] = ...`, niet met `+=`: bij een nieuwe sleutel staat nog geen getal om bij op te tellen."),
            Mistake("weights-only", matches=_is(WEIGHTS),
                    en="Your dict holds the weights themselves: the same for every applicant, a global explanation. For applicant A, multiply each weight by A's value: `weights[feature] * person[feature]`.",
                    nl="Je dict bevat de weights zelf: voor elke aanvrager hetzelfde, een global explanation. Vermenigvuldig voor aanvrager A elke weight met de waarde van A: `weights[feature] * person[feature]`."),
            Mistake("values-only", matches=_is(APPLICANT_A),
                    en="Your dict holds the values of applicant A. What a feature adds to the logit is its value times its weight: `weights[feature] * person[feature]`.",
                    nl="Je dict bevat de waarden van aanvrager A. Wat een feature bijdraagt aan de logit, is de waarde keer de weight: `weights[feature] * person[feature]`."),
            Mistake("running", matches=lambda got, text: (same_numbers(got, _running(APPLICANT_A, 0.0))
                                                         or same_numbers(got, _running(APPLICANT_A, BIAS))),
                    en="Your numbers add up as they go: each one holds the features before it too, as the logit in the worked example does. Store each feature's own number: `contributions[feature] = weights[feature] * person[feature]`.",
                    nl="Je getallen tellen op terwijl de loop loopt: elk getal bevat ook de features ervoor, zoals de logit in het uitgewerkte voorbeeld. Zet het eigen getal van elke feature in de dict: `contributions[feature] = weights[feature] * person[feature]`."),
            Mistake("sum", matches=_is_number(logit(APPLICANT_A), logit(APPLICANT_A) - BIAS),
                    en="Your last line gives one number, `{got}`: the features added up, with or without the bias. The exercise asks for a number per feature, in a dict: `contributions[feature] = ...`.",
                    nl="Je laatste regel geeft één getal, `{got}`: de features bij elkaar opgeteld, met of zonder de bias. De oefening vraagt om een getal per feature, in een dict: `contributions[feature] = ...`."),
            Mistake("last-only", matches=lambda got, text: type(got) is dict and list(got) == ["years_at_job"],
                    en="Your dict only holds `years_at_job`, the last feature. Is `contributions[feature] = ...` inside the loop, indented under `for`, and do you make `contributions = {{}}` once, before the loop?",
                    nl="Je dict bevat alleen `years_at_job`, de laatste feature. Staat `contributions[feature] = ...` binnen de loop, ingesprongen onder `for`, en maak je `contributions = {{}}` één keer, vóór de loop?"),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `contributions = {{}}`, and for each feature in `weights` store its weight times the value of `person`.",
                      "nl": "Vervang de `...` door je code: begin met `contributions = {{}}`, en zet voor elke feature in `weights` de weight keer de waarde van `person` in de dict."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `weights` and `person` exactly as they were: put them back. The check tries applicant B by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `weights` en `person` nodig zoals ze waren: zet ze terug. De check probeert zelf aanvrager B."},
            "no_value": {"en": "The last line of the box shows nothing. Put `contributions` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `contributions` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does it hold, for each feature in `weights`, the weight times the value of `person`?",
                      "nl": "Je laatste regel geeft `{got}`. Staat daarin voor elke feature in `weights` de weight keer de waarde van `person`?"},
            "other": {"en": "Right for applicant A. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your code use `person` itself, and not the numbers of A?",
                      "nl": "Goed voor aanvrager A. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je code `person` zelf, en niet de getallen van A?"},
            "other_error": {"en": "Right for applicant A. But with {where}, your code stops with an error. Does your code use `person` itself?",
                            "nl": "Goed voor aanvrager A. Maar met {where} stopt je code met een fout. Gebruikt je code `person` zelf?"},
            "correct": {"en": "Income `2.8`, missed payments `-2.4`, years at the job `0.3`; with the bias `-3.0` they add up to the logit `-2.3` of the worked example. The two missed payments weigh most against applicant A. For applicant B the check got income `1.6`, missed payments `0` and years at the job `1.8`: the same weights, other contributions. The weights are a global explanation, the same for everyone; your contributions are a local one, for one applicant, read straight from the model.",
                        "nl": "Inkomen `2.8`, gemiste betalingen `-2.4`, jaren in de baan `0.3`; met de bias `-3.0` tellen ze op tot de logit `-2.3` uit het uitgewerkte voorbeeld. De twee gemiste betalingen wegen het zwaarst tegen aanvrager A. Voor aanvrager B kreeg de check inkomen `1.6`, gemiste betalingen `0` en jaren in de baan `1.8`: dezelfde weights, andere bijdragen. De weights zijn een global explanation, voor iedereen hetzelfde; jouw bijdragen zijn een local explanation, voor één aanvrager, direct uit het model gelezen."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "model-one": {"en": "From the model: one prediction", "nl": "Uit het model: één voorspelling"},
            "model-all": {"en": "From the model: the whole model", "nl": "Uit het model: het hele model"},
            "after-one": {"en": "Made afterwards: one prediction", "nl": "Achteraf gemaakt: één voorspelling"},
            "after-all": {"en": "Made afterwards: the whole model", "nl": "Achteraf gemaakt: het hele model"},
        },
        items={
            "weights": {"en": "**The weights.** The weights of the loan model, read as they are.",
                        "nl": "**De weights.** De weights van het kredietmodel, gelezen zoals ze zijn."},
            "tree": {"en": "**The whole tree.** A decision tree with three questions, drawn in full.",
                     "nl": "**De hele boom.** Een beslisboom met drie vragen, helemaal getekend."},
            "path": {"en": "**One path.** The questions a small tree asked applicant A, from the root to a leaf.",
                     "nl": "**Eén pad.** De vragen die een kleine boom aanvrager A stelde, van de wortel tot een blad."},
            "yours": {"en": "**Your contributions.** Weight times value for each feature of applicant A.",
                      "nl": "**Jouw bijdragen.** Weight keer waarde voor elke feature van aanvrager A."},
            "shuffle": {"en": "**Shuffle a column.** Shuffle the incomes of all applicants and see how much more often the model is wrong.",
                        "nl": "**Een kolom schudden.** Schud de inkomens van alle aanvragers door elkaar en kijk hoeveel vaker het model het mis heeft."},
            "average": {"en": "**An average curve.** Give every applicant in turn other incomes, ask the model each time, and average its answers per income.",
                        "nl": "**Een gemiddelde curve.** Geef elke aanvrager om de beurt andere inkomens, vraag het model elke keer, en neem per inkomen het gemiddelde van de antwoorden."},
            "nearby": {"en": "**A model nearby.** Ask a neural network about many small variations of applicant A, and fit a small linear model to its answers.",
                       "nl": "**Een model in de buurt.** Vraag een neural network naar veel kleine varianten van aanvrager A, en pas een klein lineair model aan op de antwoorden."},
        },
        solution={"weights": "model-all", "tree": "model-all", "path": "model-one", "yours": "model-one",
                  "shuffle": "after-all", "average": "after-all", "nearby": "after-one"},
        nudge={
            "weights": {"en": "Do the weights change from one applicant to the next?", "nl": "Veranderen de weights van de ene aanvrager naar de andere?"},
            "tree": {"en": "The tree is the model itself. Does it show one applicant, or every applicant?", "nl": "De boom is het model zelf. Laat hij één aanvrager zien, of elke aanvrager?"},
            "path": {"en": "The questions come from the model itself. For how many applicants?", "nl": "De vragen komen uit het model zelf. Voor hoeveel aanvragers?"},
            "yours": {"en": "You read the weights from the model itself. For how many applicants?", "nl": "Je las de weights uit het model zelf. Voor hoeveel aanvragers?"},
            "shuffle": {"en": "Nothing is read from the inside: the model is asked again after training. Over one applicant, or all of them?", "nl": "Er wordt niets van binnen gelezen: na het trainen krijgt het model opnieuw vragen. Over één aanvrager, of over allemaal?"},
            "average": {"en": "The model is only asked, after training. Does the curve describe one applicant, or all of them on average?", "nl": "Het model krijgt alleen vragen, na het trainen. Beschrijft de curve één aanvrager, of allemaal gemiddeld?"},
            "nearby": {"en": "The small linear model is made after training, by asking. Around how many applicants?", "nl": "Het kleine lineaire model wordt na het trainen gemaakt, uit de antwoorden op vragen. Rond hoeveel aanvragers?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the explanations again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitleg opnieuw uit."},
            "empty": {"en": "Put each explanation in the group that fits it.",
                      "nl": "Zet elke uitleg in de groep die erbij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The explanation *{item}* fits another group.",
                      "nl": "De uitleg *{item}* past in een andere groep."},
            "correct": {"en": "All {n} are right. The weights and the whole tree are read from the model and cover every applicant; a path and your contributions are read from the model for one applicant. Shuffling a column and the average curve are made afterwards and describe the whole model; later lessons build them as permutation importance and the partial dependence plot. The model nearby is LIME: made afterwards, for one prediction.",
                        "nl": "Alle {n} kloppen. De weights en de hele boom lees je uit het model, en ze gelden voor elke aanvrager; een pad en jouw bijdragen lees je uit het model voor één aanvrager. Een kolom schudden en de gemiddelde curve worden achteraf gemaakt en beschrijven het hele model; latere lessen bouwen ze als permutation importance en de partial dependence plot. Het model in de buurt is LIME: achteraf gemaakt, voor één voorspelling."},
        },
    ),
}

_SORTED = {"weights": "model-all", "tree": "model-all", "path": "model-one", "yours": "model-one",
           "shuffle": "after-all", "average": "after-all", "nearby": "after-one"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_LITERAL = _C_HEAD + 'contributions = {"income": 2.8, "missed_payments": -2.4, "years_at_job": 0.3}\ncontributions'
CASES = [
    ("contributions", _C_START, "empty", None),
    ("contributions", _C_SOLUTION, "correct", None),
    ("contributions", _C_HEAD + "contributions = {}\nfor feature in person:\n"
                      "    contributions[feature] = person[feature] * weights[feature]\ncontributions", "correct", None),
    ("contributions", _c("person[feature]\n", 'person["income"]\n'), "wrong", None),
    ("contributions", _c("person[feature]\n", 'person["salary"]\n'), "error", "key"),
    ("contributions", _c("weights[feature] * person[feature]", "weights[feature]"), "wrong", "weights-only"),
    ("contributions", _c("weights[feature] * person[feature]", "person[feature]"), "wrong", "values-only"),
    ("contributions", _C_HEAD + "contributions = {}\ntotal = 0\nfor feature in weights:\n"
                      "    total = total + weights[feature] * person[feature]\n"
                      "    contributions[feature] = total\ncontributions", "wrong", "running"),
    ("contributions", _C_HEAD + "contributions = {}\ntotal = -3.0\nfor feature in weights:\n"
                      "    total = total + weights[feature] * person[feature]\n"
                      "    contributions[feature] = total\ncontributions", "wrong", "running"),
    ("contributions", _C_HEAD + "contributions = -3.0\nfor feature in weights:\n"
                      "    contributions = contributions + weights[feature] * person[feature]\ncontributions",
     "wrong", "sum"),
    ("contributions", _C_HEAD + "contributions = 0\nfor feature in weights:\n"
                      "    contributions = contributions + weights[feature] * person[feature]\ncontributions",
     "wrong", "sum"),
    ("contributions", _C_HEAD + "for feature in weights:\n    contributions = {}\n"
                      "    contributions[feature] = weights[feature] * person[feature]\ncontributions", "wrong", "last-only"),
    ("contributions", _C_HEAD + "contributions = {}\nfor feature in weights:\n    value = weights[feature] * person[feature]\n"
                      "contributions[feature] = value\ncontributions", "wrong", "last-only"),
    ("contributions", _LITERAL, "almost", "other-value"),
    ("contributions", _C_SOLUTION.replace(_PERSON_LINE, _PERSON_LINE.replace("3.5", "3.0")), "wrong", "changed-line"),
    ("contributions", _C_HEAD + "...\ncontributions = {}\ncontributions", "wrong", "dots-left"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"weights": "model-all", "path": "model-one"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "yours": "model-all"}, "wrong", "wrong:yours"),
    ("kinds", {**_SORTED, "shuffle": "model-all"}, "wrong", "wrong:shuffle"),
    ("kinds", {**_SORTED, "nearby": "after-all"}, "wrong", "wrong:nearby"),
    ("kinds", {**_SORTED, "tree": "model-one"}, "wrong", "wrong:tree"),
]
