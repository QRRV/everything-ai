# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-bias-lifecycle: counting per group who a care program helps when it chooses by need
instead of by cost, sorting cases into historical, representation and measurement bias; with the picture of the seven
places where bias can enter on the way from the world to a model in use, and the picture of who each rule helps."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Made-up patients: their group, how many chronic illnesses they have, and their care costs last year in thousands.
# Both groups have patients with 1, 2, 3, 5 and 6 illnesses, but less was spent on group B at the same illness.
PATIENTS = [
    {"group": "A", "illnesses": 3, "cost": 7},
    {"group": "B", "illnesses": 5, "cost": 5},
    {"group": "A", "illnesses": 1, "cost": 2},
    {"group": "B", "illnesses": 2, "cost": 3},
    {"group": "A", "illnesses": 6, "cost": 11},
    {"group": "B", "illnesses": 6, "cost": 6},
    {"group": "A", "illnesses": 2, "cost": 5},
    {"group": "B", "illnesses": 1, "cost": 1},
    {"group": "A", "illnesses": 5, "cost": 8},
    {"group": "B", "illnesses": 3, "cost": 4},
]
LIMIT = 6   # the worked example: help everyone whose cost is this or more
NEED = 5    # the exercise: help everyone with this many illnesses or more


def helped(patients: list, key: str, least: int) -> dict:
    """Per group, how many patients have at least `least` of `key`."""
    out = {"A": 0, "B": 0}
    for p in patients:
        if p[key] >= least:
            out[p["group"]] += 1
    return out


def _svg_box(x, y, w, h, name, subs) -> str:
    first = y + (48 if len(subs) == 1 else 44)
    lines = "".join(f'<text x="{x + w / 2}" y="{first + 16 * i}" style="font-size:13px;fill:var(--ea-ink-2);'
                    f'text-anchor:middle">{html.escape(s)}</text>' for i, s in enumerate(subs))
    return (f'<rect x="{x + 1}" y="{y}" width="{w - 2}" height="{h}" rx="8" '
            f'style="fill:var(--ea-ink);fill-opacity:0.04;stroke:var(--ea-ink-3);stroke-width:1.2"/>'
            f'<text x="{x + w / 2}" y="{y + 24}" style="font-size:15px;font-weight:650;fill:var(--ea-ink);'
            f'text-anchor:middle">{html.escape(name)}</text>' + lines)


def picture(labels: dict) -> str:
    """The way from the world to a model in use as a loop of six steps in two columns, narrow enough for a phone:
    making the data down the left, building and using the model up the right, each step with the bias that can enter
    there, and a dashed arrow from use back to the world, under the title of the frame. `labels` gives "title",
    "steps" (six pairs of a name and a list of bias names), "heads" (the two column heads), "back" (what the dashed
    arrow means) and "alt". Ink only: no step is a role of a model."""
    BW, BH, GX, GY = 200, 72, 70, 30
    top = 26
    W = 2 * BW + GX
    parts = []
    for c, head in enumerate(labels["heads"]):
        parts.append(f'<text x="{c * (BW + GX) + BW / 2}" y="{top - 10}" style="font-size:13px;fill:var(--ea-ink-3);'
                     f'text-anchor:middle">{html.escape(head)}</text>')
    places = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]   # (column, row) of steps 1 to 6

    def box(c, r):
        return c * (BW + GX), top + r * (BH + GY)

    for (name, subs), (c, r) in zip(labels["steps"], places):
        x, y = box(c, r)
        parts.append(_svg_box(x, y, BW, BH, name, subs))
    line = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'

    def head(x, y, dx, dy):
        """An arrow head at (x, y), pointing along (dx, dy)."""
        return (f'<path d="M{x - 6 * dx - 5 * dy} {y - 6 * dy - 5 * dx} L{x} {y} '
                f'L{x - 6 * dx + 5 * dy} {y - 6 * dy + 5 * dx}" {line}/>')

    xl, xr = BW / 2, BW + GX + BW / 2
    for r in range(2):   # 1 → 2 → 3, down the left
        a, b = top + r * (BH + GY) + BH + 5, top + (r + 1) * (BH + GY) - 5
        parts.append(f'<path d="M{xl} {a} V{b}" {line}/>' + head(xl, b, 0, 1))
    y3 = top + 2 * (BH + GY) + BH / 2   # 3 → 4, along the bottom
    parts.append(f'<path d="M{BW + 6} {y3} H{BW + GX - 6}" {line}/>' + head(BW + GX - 6, y3, 1, 0))
    for r in (2, 1):     # 4 → 5 → 6, up the right
        a, b = top + r * (BH + GY) - 5, top + (r - 1) * (BH + GY) + BH + 5
        parts.append(f'<path d="M{xr} {a} V{b}" {line}/>' + head(xr, b, 0, -1))
    y1 = top + BH / 2    # 6 back to 1, dashed, along the top: the loop
    parts.append(f'<path d="M{BW + GX - 6} {y1} H{BW + 6}" {line} stroke-dasharray="5 4"/>' + head(BW + 6, y1, -1, 0))
    yk = top + 3 * BH + 2 * GY + 28   # the key: what the dashed arrow means
    parts.append(f'<path d="M0 {yk - 4} H34" {line} stroke-dasharray="5 4"/>' + head(34, yk - 4, 1, 0)
                 + f'<text x="44" y="{yk}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(labels["back"])}</text>')
    H = yk + 6
    svg = (f'<svg viewBox="0 0 {W + 2} {H}" role="img" style="width:100%;height:auto;max-width:{W + 2}px;'
           f'font-family:var(--ea-font-ui)"><title>{html.escape(labels["alt"])}</title>{"".join(parts)}</svg>')
    return yours.figure(labels["title"], svg)


def _ok(patients) -> bool:
    if not isinstance(patients, list) or not 1 <= len(patients) <= 16:
        return False
    for p in patients:
        if type(p) is not dict or p.get("group") not in ("A", "B"):
            return False
        for key in ("illnesses", "cost"):
            v = p.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 40:
                return False
    return True


def _pair(v) -> tuple | None:
    """(count of A, count of B) when `v` is a dict with whole numbers for both groups and nothing else."""
    if type(v) is not dict or set(v) != {"A", "B"}:
        return None
    for x in v.values():
        if isinstance(x, bool) or not isinstance(x, int):
            return None
    return v["A"], v["B"]


def _choose_picture(patients: list, need, words: dict, title: str) -> str:
    """Each patient as a dot by illnesses and cost, filled for group A and open for group B, with a dashed line for
    each rule: by cost (the worked example) across, and by need (the exercise) up."""
    from everything_ai import chart
    right = max(7, max(p["illnesses"] for p in patients) + 1)
    high = max(15, max(p["cost"] for p in patients) + 4)
    p = chart.Panel(x=(0, right), y=(0, high), xlabel=words["xlabel"], ylabel=words["ylabel"],
                    xticks=list(range(0, int(right) + 1)), xtick_labels=[str(i) for i in range(0, int(right) + 1)])
    p.line([0.1, right - 0.1], [LIMIT - 0.5, LIMIT - 0.5], role="ink-3", width=1, dashed=True)
    p.text(0.2, LIMIT + 0.2, words["cost"], "ink-3", anchor="start", halo=True)
    if isinstance(need, (int, float)) and not isinstance(need, bool) and 0 < need <= right:
        p.line([need - 0.5, need - 0.5], [0.2, high - 2.6], role="ink-3", width=1, dashed=True)
        p.text(need - 0.4, high - 2.1, words["need"], "ink-3", anchor="start", halo=True)
    for group, hollow in (("A", False), ("B", True)):
        mine = [q for q in patients if q["group"] == group]
        p.points([q["illnesses"] for q in mine], [q["cost"] for q in mine], role="ink", r=5, hollow=hollow)
    p.text(0.2, high - 0.6, words["key"], "ink-3", anchor="start")
    alt = (f'{title}. ' + "; ".join(f'{words["group"]} {q["group"]}: {q["illnesses"]} {words["illnesses"]}, '
                                    f'{words["costs"]} {q["cost"]}' for q in patients) + ".")
    return yours.figure(title, chart.figure([p], alt=alt, width=520, height=340))


_WORDS = {
    "en": {"xlabel": "chronic illnesses", "ylabel": "care costs (thousands)", "cost": f"by cost: {LIMIT} or more",
           "need": "by need: {need} or more", "key": "filled: group A, open: group B", "group": "group",
           "illnesses": "illnesses", "costs": "costs",
           "title": "Helped when chosen by need: A {a}, B {b}"},
    "nl": {"xlabel": "chronische ziektes", "ylabel": "zorgkosten (duizenden)", "cost": f"op kosten: {LIMIT} of meer",
           "need": "op zorgbehoefte: {need} of meer", "key": "gevuld: groep A, open: groep B", "group": "groep",
           "illnesses": "ziektes", "costs": "kosten",
           "title": "Geholpen bij een keuze op zorgbehoefte: A {a}, B {b}"},
}


def _helped_picture(seen) -> str:
    """The learner's patients and need, with the counts they work out in the title."""
    mine = seen.mine
    if mine is None or not mine.has_value or _pair(mine.value) is None:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    patients = names.get("patients", PATIENTS)
    if not _ok(patients):
        return ""
    need = names.get("need", NEED)
    if isinstance(need, bool) or not isinstance(need, (int, float)):
        return ""
    words = _WORDS[seen.lang]
    a, b = _pair(mine.value)
    words = {**words, "need": words["need"].format(need=need)}
    return _choose_picture(patients, need, words, words["title"].format(a=a, b=b))


# The lesson is done when these are correct. "fix" is the choice question in the lesson.
REQUIRED = ["helped", "steps", "fix"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(p: dict) -> str:
    return f"    {p!r},".replace("'", '"')


_PATIENT_LINES = [_line(p) for p in PATIENTS]
_NEED_LINE = f"need = {NEED}"
_HEAD = ("# made-up patients: group, number of chronic illnesses, care costs last year in thousands\n"
         "patients = [\n" + "\n".join(_PATIENT_LINES) + "\n]\n"
         "# the new rule: help everyone with at least this many illnesses\n" + _NEED_LINE + "\n\n")
_START = _HEAD + "...\nhelped\n"
_BODY = ("helped = {\"A\": 0, \"B\": 0}\nfor patient in patients:\n    if patient[\"illnesses\"] >= need:\n"
         "        helped[patient[\"group\"]] = helped[patient[\"group\"]] + 1\n")
_SOLUTION = _HEAD + _BODY + "helped"


def _s(body: str) -> str:
    return _HEAD + body + "helped"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _is(a: int, b: int):
    return lambda got, text: _pair(got) == (a, b)


EXERCISES = {
    "helped": CodeBox(
        name="helped",
        picture=_helped_picture,
        start=_START,
        solution=_SOLUTION,
        keep=["patients = ["] + _PATIENT_LINES + [_NEED_LINE],
        # The check also runs the box with a rule of 3 illnesses and one of 6.
        others=[{"need": 3}, {"need": 6}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_patients`. In the box the names have none: `patients`, `need`, `patient`, `helped`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_patients`. In het vak hebben de namen dat niet: `patients`, `need`, `patient`, `helped`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks for a key that is not there. Each patient has `\"group\"`, `\"illnesses\"` and `\"cost\"`, and `helped` has `\"A\"` and `\"B\"`.",
                    nl="Je code vraagt om een sleutel die er niet is. Elke patiënt heeft `\"group\"`, `\"illnesses\"` en `\"cost\"`, en `helped` heeft `\"A\"` en `\"B\"`."),
            Mistake("strict", matches=_is(1, 1),
                    en="Your last line gives `{got}`. The rule helps everyone with `need` illnesses *or more*: a patient with exactly 5 gets help, and so does a patient with more. Use `>=`.",
                    nl="Je laatste regel geeft `{got}`. De regel helpt iedereen met `need` ziektes *of meer*: een patiënt met precies 5 krijgt hulp, en een patiënt met meer ook. Gebruik `>=`."),
            Mistake("cost", matches=_is(3, 1),
                    en="Your last line gives `{got}`: who gets help when the program chooses by cost, as in the worked example. Choose by need: look at `patient[\"illnesses\"]`, and compare it with `need`.",
                    nl="Je laatste regel geeft `{got}`: wie hulp krijgt als het programma op kosten kiest, zoals in het uitgewerkte voorbeeld. Kies op zorgbehoefte: kijk naar `patient[\"illnesses\"]`, en vergelijk dat met `need`."),
            Mistake("below", matches=_is(3, 3),
                    en="Your last line gives `{got}`: the patients with *fewer* than `need` illnesses, who get no help. Count the ones with `need` or more.",
                    nl="Je laatste regel geeft `{got}`: de patiënten met *minder* dan `need` ziektes, die geen hulp krijgen. Tel wie er `need` of meer heeft."),
            Mistake("all", matches=_is(5, 5),
                    en="Your last line gives `{got}`: every patient of each group. Add 1 only when `patient[\"illnesses\"]` is `need` or more: put an `if` in the loop.",
                    nl="Je laatste regel geeft `{got}`: elke patiënt van elke groep. Tel er alleen 1 bij op als `patient[\"illnesses\"]` `need` of meer is: zet een `if` in de loop."),
            Mistake("total", matches=lambda got, text: isinstance(got, int) and not isinstance(got, bool),
                    en="Your last line gives one number, `{got}`. `helped` keeps a count per group: start with `helped = {{\"A\": 0, \"B\": 0}}` and add 1 to the count of the patient's own group.",
                    nl="Je laatste regel geeft één getal, `{got}`. `helped` houdt per groep een aantal bij: begin met `helped = {{\"A\": 0, \"B\": 0}}` en tel 1 op bij het aantal van de eigen groep van de patiënt."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="Your last line gives a list, `{got}`. `helped` is a dict with a count per group: start with `helped = {{\"A\": 0, \"B\": 0}}`.",
                    nl="Je laatste regel geeft een lijst, `{got}`. `helped` is een dict met een aantal per groep: begin met `helped = {{\"A\": 0, \"B\": 0}}`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with code as in the worked example: start with `helped = {{\"A\": 0, \"B\": 0}}`, and count per group the patients with `need` illnesses or more.",
                      "nl": "Vervang de `...` door code zoals in het uitgewerkte voorbeeld: begin met `helped = {{\"A\": 0, \"B\": 0}}`, en tel per groep de patiënten met `need` ziektes of meer."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `patients` and `need` exactly as they were: put them back. The check tries other values of `need` by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `patients` en `need` nodig zoals ze waren: zet ze terug. De check probeert zelf andere waarden van `need`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `helped` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `helped` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you add 1 to the count of the patient's own group, `helped[patient[\"group\"]]`, only when `patient[\"illnesses\"]` is `need` or more?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je 1 op bij het aantal van de eigen groep van de patiënt, `helped[patient[\"group\"]]`, alleen als `patient[\"illnesses\"]` `need` of meer is?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `patients` and `need` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `patients` en `need` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `patients` and `need` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `patients` en `need` zelf?"},
            "correct": {"en": "`{{'A': 2, 'B': 2}}`: chosen by need, each group gets two places, as both groups are equally sick. Chosen by cost it was 3 and 1, because less was spent on group B at the same illness. The label measured need worse for one group: measurement bias. The fix is in the label, not in more data.",
                        "nl": "`{{'A': 2, 'B': 2}}`: bij een keuze op zorgbehoefte krijgt elke groep twee plekken, want beide groepen zijn even ziek. Op kosten was het 3 en 1, omdat er bij dezelfde ziekte minder aan groep B werd uitgegeven. Het label mat de zorgbehoefte slechter voor één groep: measurement bias. De oplossing zit in het label, niet in meer data."},
        },
    ),
    "steps": Sort(
        name="steps",
        groups={
            "historical": {"en": "Historical bias", "nl": "Historical bias"},
            "representation": {"en": "Representation bias", "nl": "Representation bias"},
            "measurement": {"en": "Measurement bias", "nl": "Measurement bias"},
        },
        items={
            "pay": {"en": "**Pay.** A model that suggests a starting salary learns from payroll data without mistakes, from years in which women were paid less for the same work.",
                    "nl": "**Loon.** Een model dat een startsalaris voorstelt, leert van loongegevens zonder fouten, uit jaren waarin vrouwen minder betaald kregen voor hetzelfde werk."},
            "ceos": {"en": "**CEOs.** In 2018, 5 percent of the CEOs of the 500 largest US companies were women, and an image search for CEO learns that share from accurate data.",
                     "nl": "**CEO's.** In 2018 was 5 procent van de CEO's van de 500 grootste Amerikaanse bedrijven een vrouw, en een zoekmachine voor afbeeldingen leert dat aandeel uit kloppende data voor de zoekterm CEO."},
            "city": {"en": "**Another city.** A model trained on the patients of a hospital in Boston is used in a hospital in Indianapolis.",
                     "nl": "**Een andere stad.** Een model dat getraind is op de patiënten van een ziekenhuis in Boston, wordt gebruikt in een ziekenhuis in Indianapolis."},
            "pregnant": {"en": "**Five percent.** Pregnant patients are 5 percent of all patients, so even a perfect sample gives the model few examples of them.",
                         "nl": "**Vijf procent.** Zwangere patiënten zijn 5 procent van alle patiënten, dus zelfs een perfecte steekproef geeft het model weinig voorbeelden van hen."},
            "diagnoses": {"en": "**Diagnoses.** A model uses \"was diagnosed with the illness\" as its label for \"has the illness\", and one group sees a doctor less often.",
                          "nl": "**Diagnoses.** Een model gebruikt \"kreeg de diagnose\" als label voor \"heeft de ziekte\", en één groep gaat minder vaak naar een dokter."},
            "inspections": {"en": "**Inspections.** A model rates factories by the mistakes found there, and one factory is inspected far more strictly than the others.",
                            "nl": "**Inspecties.** Een model beoordeelt fabrieken op de fouten die er gevonden zijn, en één fabriek wordt veel strenger geïnspecteerd dan de andere."},
        },
        solution={"pay": "historical", "ceos": "historical", "city": "representation",
                  "pregnant": "representation", "diagnoses": "measurement", "inspections": "measurement"},
        nudge={
            "pay": {"en": "Was the data measured badly, or was the world itself unequal?",
                    "nl": "Is de data slecht gemeten, of was de wereld zelf ongelijk?"},
            "ceos": {"en": "Does the data get the share wrong, or is the share itself the problem?",
                     "nl": "Heeft de data het aandeel verkeerd, of is het aandeel zelf het probleem?"},
            "city": {"en": "Are the patients in the data the same people as the ones the model is used for?",
                     "nl": "Zijn de patiënten in de data dezelfde mensen als die voor wie het model gebruikt wordt?"},
            "pregnant": {"en": "How many examples does the model get of this group?",
                         "nl": "Hoeveel voorbeelden krijgt het model van deze groep?"},
            "diagnoses": {"en": "Does the label measure the illness equally well for both groups?",
                          "nl": "Meet het label de ziekte voor beide groepen even goed?"},
            "inspections": {"en": "Do the mistakes found measure the real mistakes equally well in every factory?",
                            "nl": "Meten de gevonden fouten de echte fouten in elke fabriek even goed?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Sort each case by the kind of bias in it.",
                      "nl": "Deel elk geval in bij de soort bias die erin zit."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs somewhere else.",
                      "nl": "Het geval *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. Pay and CEOs: the world itself was unequal, so even perfect data copies it. Another city and five percent: too few examples of the people the model is used for. Diagnoses and inspections: the label measures what you care about worse for one group or one place.",
                        "nl": "Alle {n} kloppen. Loon en CEO's: de wereld zelf was ongelijk, dus zelfs perfecte data neemt dat over. Een andere stad en vijf procent: te weinig voorbeelden van de mensen voor wie het model gebruikt wordt. Diagnoses en inspecties: het label meet waar het om gaat slechter voor één groep of één plek."},
        },
    ),
}

_SORTED = {"pay": "historical", "ceos": "historical", "city": "representation", "pregnant": "representation",
           "diagnoses": "measurement", "inspections": "measurement"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("helped", _START, "empty", None),
    ("helped", _SOLUTION, "correct", None),
    ("helped", _s("helped = {\"A\": 0, \"B\": 0}\nfor patient in patients:\n    group = patient[\"group\"]\n"
                  "    if patient[\"illnesses\"] >= need:\n        helped[group] = helped[group] + 1\n"), "correct", None),
    ("helped", _s("helped = {\"A\": 0, \"B\": 0}\nfor patient in patients:\n    if patient[\"illnesses\"] >= need:\n"
                  "        if patient[\"group\"] == \"A\":\n            helped[\"A\"] = helped[\"A\"] + 1\n"
                  "        else:\n            helped[\"B\"] = helped[\"B\"] + 1\n"), "correct", None),
    ("helped", _s(_BODY.replace("in patients", "in _patients")), "error", "underscore"),
    ("helped", _swap('patient["illnesses"]', 'patient["illness"]'), "error", "key"),
    ("helped", _swap(">= need", "> need"), "wrong", "strict"),
    ("helped", _swap(">= need", "== need"), "wrong", "strict"),
    ("helped", _swap('patient["illnesses"] >= need', 'patient["cost"] >= 6'), "wrong", "cost"),
    ("helped", _swap(">= need", "< need"), "wrong", "below"),
    ("helped", _s("helped = {\"A\": 0, \"B\": 0}\nfor patient in patients:\n"
                  "    helped[patient[\"group\"]] = helped[patient[\"group\"]] + 1\n"), "wrong", "all"),
    ("helped", _s("helped = 0\nfor patient in patients:\n    if patient[\"illnesses\"] >= need:\n"
                  "        helped = helped + 1\n"), "wrong", "total"),
    ("helped", _s("helped = []\nfor patient in patients:\n    if patient[\"illnesses\"] >= need:\n"
                  "        helped.append(patient[\"group\"])\n"), "wrong", "list"),
    ("helped", _swap("] + 1\n", "] + 2\n"), "wrong", None),
    ("helped", _s("helped = {\"A\": 2, \"B\": 2}\n"), "almost", "other-value"),
    ("helped", _swap(">= need", ">= 5"), "almost", "other-value"),
    ("helped", _SOLUTION.replace("need = 5", "need = 3"), "wrong", "changed-line"),
    ("helped", _s("...\nhelped = {\"A\": 2, \"B\": 2}\n"), "wrong", "dots-left"),
    ("steps", {}, "empty", None),
    ("steps", _SORTED, "correct", None),
    ("steps", {"pay": "historical", "inspections": "measurement"}, "wrong", "unfinished"),
    ("steps", {**_SORTED, "pay": "measurement"}, "wrong", "wrong:pay"),
    ("steps", {**_SORTED, "pregnant": "measurement"}, "wrong", "wrong:pregnant"),
    ("steps", {**_SORTED, "inspections": "representation"}, "wrong", "wrong:inspections"),
    ("steps", {**_SORTED, "ceos": "representation"}, "wrong", "wrong:ceos"),
]
