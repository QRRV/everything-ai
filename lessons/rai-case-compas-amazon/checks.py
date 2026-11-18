# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-compas-amazon: for each group, the share of the defendants who were not arrested again
that COMPAS still rated medium or high risk, sorting situations by where the error comes from, and the pictures of the
two confusion matrices and of the learner's shares."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# ProPublica's published counts for COMPAS's general recidivism score in Broward County, Florida (people scored in 2013
# and 2014, followed for two years), for Black and White defendants: positive means rated medium or high risk (a score
# of 5 to 10), and a person counts as reoffending when arrested again within two years. Source: Larson and colleagues
# (2016), "How we analyzed the COMPAS recidivism algorithm", the tables of the false positive and false negative rates.
TP = {"black": 1369, "white": 505}    # arrested again, rated medium or high
FP = {"black": 805, "white": 349}     # not arrested again, rated medium or high
FN = {"black": 532, "white": 461}     # arrested again, rated low
TN = {"black": 990, "white": 1139}    # not arrested again, rated low
# All defendants in the table, of every group: rated medium or high, and of those arrested again (the guess).
FLAGGED_ALL, FLAGGED_ALL_AGAIN = 2035 + 1282, 2035


def precision(tp: dict, fp: dict) -> dict:
    """For each group, the share of the flagged people who were arrested again."""
    return {g: tp[g] / (tp[g] + fp[g]) for g in tp}


def flagged(fp: dict, tn: dict) -> dict:
    """For each group, the share of the people who were not arrested again that the score flagged anyway."""
    return {g: fp[g] / (fp[g] + tn[g]) for g in fp}


def arrested_again(tp: dict, fp: dict, fn: dict, tn: dict) -> dict:
    """For each group, the share of all its defendants who were arrested again."""
    return {g: (tp[g] + fn[g]) / (tp[g] + fp[g] + fn[g] + tn[g]) for g in tp}


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The confusion matrix of each group, side by side: rows not arrested again and arrested again, columns rated low
    and rated medium or high. The wrong flags have a wash and a solid outline in the colour of mistakes, the people
    rated low who were arrested again a dashed outline. `labels` gives "title", "groups" (two), "rows" (two),
    "columns" (two), "wrong_flag", "missed" and "alt"."""
    def cell(n: int, kind: str = "") -> str:
        style = {"flag": "background:var(--ea-loss-wash);border:2px solid var(--ea-loss);font-weight:600;color:var(--ea-loss-text)",
                 "miss": "border:2px dashed var(--ea-loss);color:var(--ea-ink)",
                 "": "border:1px solid var(--ea-line);color:var(--ea-ink)"}[kind]
        return (f'<div style="{style};border-radius:var(--ea-radius-small);padding:0.35rem 0.5rem;text-align:center;'
                f'font-variant-numeric:tabular-nums;min-width:0">{f"{n:,}".replace(",", labels.get("thousands", ","))}</div>')

    def head(text: str, side: bool = False) -> str:
        align = "left" if side else "center"
        return (f'<div style="font-size:0.8rem;color:var(--ea-ink-2);text-align:{align};align-self:center;'
                f'line-height:1.25;min-width:0">{html.escape(text)}</div>')

    tables = []
    for g, name in zip(TP, labels["groups"]):
        parts = ["<div></div>", head(labels["columns"][0]), head(labels["columns"][1]),
                 head(labels["rows"][0], True), cell(TN[g]), cell(FP[g], "flag"),
                 head(labels["rows"][1], True), cell(FN[g], "miss"), cell(TP[g])]
        tables.append(f'<div style="flex:1 1 15rem;min-width:0"><div style="font-weight:600;color:var(--ea-ink);'
                      f'margin-bottom:0.3rem">{html.escape(name)}</div><div style="display:grid;'
                      f'grid-template-columns:minmax(5rem,1.1fr) 1fr 1fr;gap:0.35rem">{"".join(parts)}</div></div>')
    flag = ('<svg width="16" height="12" aria-hidden="true"><rect x="1" y="1" width="14" height="10" rx="2" '
            'style="fill:var(--ea-loss-wash);stroke:var(--ea-loss);stroke-width:2"/></svg>')
    miss = ('<svg width="16" height="12" aria-hidden="true"><rect x="1" y="1" width="14" height="10" rx="2" '
            'style="fill:none;stroke:var(--ea-loss);stroke-width:2;stroke-dasharray:3 2"/></svg>')
    key = (f'<div style="font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.5rem">'
           f'<span>{flag} {html.escape(labels["wrong_flag"])}</span><span>{miss} {html.escape(labels["missed"])}</span></div>')
    body = (f'<div role="img" aria-label="{html.escape(labels["alt"])}" style="{_UI};font-size:0.9rem;max-width:40rem">'
            f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap">{"".join(tables)}</div>{key}</div>')
    return yours.figure(labels["title"], body)


def _number(v) -> float | None:
    """A number the picture can draw, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _flagged_picture(seen) -> str:
    """The learner's share per group as a bar, with its value above it, and a dashed line at the right share."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    got, want = mine.value, flagged(FP, TN)
    groups = list(FP)
    panel = chart.Panel(x=(-0.6, len(groups) - 0.4), y=(0, 1.1), ylabel=seen.word({"en": "share flagged", "nl": "deel gemarkeerd"}),
                        yticks=[0, 0.25, 0.5, 0.75, 1], xticks=list(range(len(groups))), xtick_labels=groups)
    for i, g in enumerate(groups):
        v = _number(got.get(g))
        if v is not None and v > 0:
            panel.rect(i - 0.3, 0, i + 0.3, min(v, 1.1), role="loss")
        if v is not None:
            shown = f"{v:.2f}" if seen.lang == "en" else f"{v:.2f}".replace(".", ",")
            panel.text(i, min(max(v, want[g]), 0.98) + 0.06, shown, "ink", halo=True)
        panel.line([i - 0.38, i + 0.38], [want[g]] * 2, role="ink", width=2, dashed=True)
    title = seen.word({"en": "Your share per group", "nl": "Jouw deel per groep"})
    words = {"en": ("your code", "right"), "nl": ("jouw code", "goed")}[seen.lang]
    box = ('<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-loss);'
           'fill-opacity:0.18;stroke:var(--ea-loss);stroke-width:1.2"/></svg>')
    dash = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
            'stroke-width:2;stroke-dasharray:5 5"/></svg>')
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;'
           f'margin-top:0.3rem"><span>{box} {words[0]}</span><span>{dash} {words[1]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=400, height=250) + key)


# The lesson is done when these are correct. "claim" is the choice question in the lesson.
REQUIRED = ["flagged", "sources", "claim"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_FP_LINE = _line("fp", FP)
_HEAD = ("# not arrested again within two years: rated medium or high (fp), and rated low (tn)\n" + _FP_LINE + "\n"
         + _line("tn", TN) + "\n\nflagged = {}\nfor group in fp:\n")
_START = _HEAD + "    ...\nflagged\n"
_LINE = "    flagged[group] = fp[group] / (fp[group] + tn[group])\n"
_SOLUTION = _HEAD + _LINE + "flagged"


def _f(body: str) -> str:
    return _HEAD + body + "flagged"


EXERCISES = {
    "flagged": CodeBox(
        name="flagged",
        picture=_flagged_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_FP_LINE],
        # The check also runs the box with other numbers rated low.
        others=[{"tn": Code('{"black": 195, "white": 1047}')}],
        first=[_DOTS],
        digits=2,
        mistakes=[
            Mistake("no-key", matches=lambda got, text: isinstance(got, float),
                    en="Your last line gives `{got}`: one number, the share of the last group. Put each share under its group: `flagged[group] = ...`.",
                    nl="Je laatste regel geeft `{got}`: één getal, het deel van de laatste groep. Zet elk deel onder zijn groep: `flagged[group] = ...`."),
            Mistake("dicts-divided", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code divides or adds whole dictionaries. Use the counts of this one group: `fp[group]` and `tn[group]`.",
                    nl="Je code deelt of telt hele dictionaries. Gebruik de aantallen van deze ene groep: `fp[group]` en `tn[group]`."),
            Mistake("over-tn", wrong=_f("    flagged[group] = fp[group] / tn[group]\n"),
                    en="Your last line gives `{got}`: the flagged against the people rated low. Divide by all the people who were not arrested again: `fp[group] + tn[group]`.",
                    nl="Je laatste regel geeft `{got}`: de gemarkeerden tegenover de mensen met een lage score. Deel door alle mensen die niet opnieuw werden opgepakt: `fp[group] + tn[group]`."),
            Mistake("rated-low", wrong=_f("    flagged[group] = tn[group] / (fp[group] + tn[group])\n"),
                    en="Your last line gives `{got}`: the share rated low. Put the people who were flagged, `fp[group]`, on top.",
                    nl="Je laatste regel geeft `{got}`: het deel met een lage score. Zet de mensen die gemarkeerd werden, `fp[group]`, bovenaan."),
            Mistake("no-brackets", wrong=_f("    flagged[group] = fp[group] / fp[group] + tn[group]\n"),
                    en="Your last line gives `{got}`: Python divides before it adds. Put parentheses around the sum below the line.",
                    nl="Je laatste regel geeft `{got}`: Python deelt voordat het optelt. Zet ronde haakjes om de som onder de streep."),
            Mistake("counts", wrong=_f("    flagged[group] = fp[group]\n"),
                    en="Your last line gives `{got}`: the counts. The groups differ in size, so divide by all the people of the group who were not arrested again.",
                    nl="Je laatste regel geeft `{got}`: de aantallen. De groepen zijn niet even groot, dus deel door alle mensen van de groep die niet opnieuw werden opgepakt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that puts under `flagged[group]` the share of this group's people who were not arrested again that the score flagged.",
                      "nl": "Vervang de `...` door één regel die onder `flagged[group]` het deel zet van de mensen uit deze groep die niet opnieuw werden opgepakt en die de score toch markeerde."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere aantallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `flagged` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `flagged` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each group its flagged people, divided by all its people who were not arrested again?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elke groep de gemarkeerde mensen, gedeeld door alle mensen van die groep die niet opnieuw werden opgepakt?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `fp` and `tn` themselves?",
                      "nl": "Goed voor deze aantallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `fp` en `tn` zelf?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `fp` and `tn` themselves?",
                            "nl": "Goed voor deze aantallen. Maar met {where} stopt je code met een fout. Gebruik je `fp` en `tn` zelf?"},
            "correct": {"en": "`{got}`: of the people who were not arrested again, the score flagged 0.45 of the Black defendants and 0.23 of the White defendants, a share almost twice as large, while its flags were about as often right in both groups. The check also tried other numbers.",
                        "nl": "`{got}`: van de mensen die niet opnieuw werden opgepakt, markeerde de score 0,45 van de zwarte verdachten en 0,23 van de witte verdachten, een bijna twee keer zo groot deel, terwijl de markeringen in beide groepen ongeveer even vaak klopten. De check probeerde ook andere aantallen."},
        },
    ),
    "sources": Sort(
        name="sources",
        groups={
            "data": {"en": "Who is in the data", "nl": "Wie er in de data zit"},
            "label": {"en": "What the label measures", "nl": "Wat het label meet"},
            "use": {"en": "How the score is used", "nl": "Hoe de score gebruikt wordt"},
        },
        items={
            "cvs": {"en": "**The CVs.** Amazon's model learned from ten years of CVs sent to the company, most of them from men.",
                    "nl": "**De cv's.** Het model van Amazon leerde van tien jaar cv's die naar het bedrijf waren gestuurd, de meeste van mannen."},
            "survey": {"en": "**The survey.** A model that predicts who needs care at home learned from an online survey that few people over 80 filled in.",
                       "nl": "**De enquête.** Een model dat voorspelt wie zorg thuis nodig heeft, leerde van een online enquête die weinig mensen boven de 80 invulden."},
            "arrest": {"en": "**The arrest.** COMPAS counts a person as reoffending when the person is arrested again, and who gets arrested also depends on where the police look.",
                       "nl": "**De arrestatie.** COMPAS telt iemand als recidivist als die persoon opnieuw wordt opgepakt, en wie er wordt opgepakt, hangt ook af van waar de politie zoekt."},
            "promotion": {"en": "**The promotion.** A model learns to predict who will be promoted, from who was promoted in the past, not from who did the job well.",
                          "nl": "**De promotie.** Een model leert voorspellen wie promotie krijgt, op basis van wie er vroeger promotie kreeg, niet van wie het werk goed deed."},
            "sentence": {"en": "**The sentence.** A court uses a score made to plan support for people on probation to decide how long someone goes to prison.",
                         "nl": "**De straf.** Een rechtbank gebruikt een score die gemaakt is om hulp aan mensen met een proeftijd te plannen, om te beslissen hoe lang iemand de gevangenis in gaat."},
            "top": {"en": "**The top five.** Recruiters invite only the five applicants with the highest score, and never look at the rest.",
                    "nl": "**De top vijf.** Recruiters nodigen alleen de vijf sollicitanten met de hoogste score uit, en kijken nooit naar de rest."},
        },
        solution={"cvs": "data", "survey": "data", "arrest": "label", "promotion": "label", "sentence": "use", "top": "use"},
        nudge={
            "cvs": {"en": "Is the problem what the model predicts, or whose CVs it saw?", "nl": "Is het probleem wat het model voorspelt, of van wie het de cv's zag?"},
            "survey": {"en": "Who is missing from what the model learned from?", "nl": "Wie ontbreekt er in waar het model van leerde?"},
            "arrest": {"en": "Is an arrest the same thing as a crime?", "nl": "Is een arrestatie hetzelfde als een misdrijf?"},
            "promotion": {"en": "What is the model asked to predict, and is that what the company wants to know?", "nl": "Wat moet het model voorspellen, en is dat wat het bedrijf wil weten?"},
            "sentence": {"en": "Was the score made for this decision?", "nl": "Is de score voor deze beslissing gemaakt?"},
            "top": {"en": "Is the problem in the score, or in what people do with it?", "nl": "Zit het probleem in de score, of in wat mensen ermee doen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the situations again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de situaties opnieuw uit."},
            "empty": {"en": "Put each situation under where its problem comes from.",
                      "nl": "Zet elke situatie bij waar het probleem vandaan komt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The problem of *{item}* comes from somewhere else.",
                      "nl": "Het probleem van *{item}* komt ergens anders vandaan."},
            "correct": {"en": "All {n} are right. The model can be built with care and still go wrong in each of these places, so a check of a risk score looks at all three: the data, the label and the use.",
                        "nl": "Alle {n} kloppen. Een model kan met zorg gebouwd zijn en toch op elk van deze plekken misgaan, dus wie een risicoscore controleert, kijkt naar alle drie: de data, het label en het gebruik."},
        },
    ),
}

_SORTED = {"cvs": "data", "survey": "data", "arrest": "label", "promotion": "label", "sentence": "use", "top": "use"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("flagged", _START, "empty", None),
    ("flagged", _SOLUTION, "correct", None),
    ("flagged", _f("    flagged[group] = fp[group] / (tn[group] + fp[group])\n"), "correct", None),
    ("flagged", _f("    flagged[group] = fp[group] / tn[group]\n"), "wrong", "over-tn"),
    ("flagged", _f("    flagged[group] = tn[group] / (fp[group] + tn[group])\n"), "wrong", "rated-low"),
    ("flagged", _f("    flagged[group] = fp[group] / fp[group] + tn[group]\n"), "wrong", "no-brackets"),
    ("flagged", _f("    flagged[group] = fp[group]\n"), "wrong", "counts"),
    ("flagged", _f("    flagged = fp[group] / (fp[group] + tn[group])\n"), "wrong", "no-key"),
    ("flagged", _f("    flagged[group] = fp / (fp + tn)\n"), "error", "dicts-divided"),
    ("flagged", _f("    flagged[group] = fp[group] / (fp[group] + 1000)\n"), "wrong", None),
    ("flagged", _SOLUTION.replace(_FP_LINE, _FP_LINE.replace('"white": 349', '"white": 450')), "wrong", "changed-line"),
    ("flagged", _HEAD + "    pass\nflagged = {'black': 0.44846796657381616, 'white': 0.23454301075268819}\nflagged", "almost", "other-value"),
    ("flagged", _HEAD + "    ...\nflagged = {}\nflagged", "wrong", "dots-left"),
    ("sources", {}, "empty", None),
    ("sources", _SORTED, "correct", None),
    ("sources", {"cvs": "data", "promotion": "label"}, "wrong", "unfinished"),
    ("sources", {**_SORTED, "cvs": "label"}, "wrong", "wrong:cvs"),
    ("sources", {**_SORTED, "arrest": "data"}, "wrong", "wrong:arrest"),
    ("sources", {**_SORTED, "top": "data"}, "wrong", "wrong:top"),
]
