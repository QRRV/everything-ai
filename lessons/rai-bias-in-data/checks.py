# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-bias-in-data: the accuracy of a speech app per accent, sorting where a skew in data
comes from, and the picture of a face detector whose overall score hides one group."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort

# The worked example and the picture: 200 test faces, 180 of group A and 20 of group B.
FACES = {"A": (180, 176), "B": (20, 12)}   # group: (faces in the test set, faces the detector found)


def picture(labels: dict[str, str]) -> str:
    """Two panels: the share of each group in the test set, and the accuracy per group next to the overall one."""
    from everything_ai import chart
    n = sum(total for total, _ in FACES.values())
    share = chart.Panel((0, 3), (0, 1.15), title=labels["share"], ylabel=labels["share_y"], xticks=[])
    acc = chart.Panel((0, 4), (0, 1.15), title=labels["acc"], ylabel=labels["acc_y"], xticks=[])
    for i, (group, (total, found)) in enumerate(FACES.items()):
        share.rect(i + 1 - 0.3, 0, i + 1 + 0.3, total / n, "data")
        share.text(i + 1, total / n + 0.06, group)
        acc.rect(i + 1 - 0.3, 0, i + 1 + 0.3, found / total, "data")
        acc.text(i + 1, found / total + 0.06, group)
    overall = sum(found for _, found in FACES.values()) / n
    acc.rect(3 - 0.3, 0, 3 + 0.3, overall, "data")
    acc.text(3, overall + 0.06, labels["all"])
    return chart.figure([share, acc], alt=labels["alt"], width=300, height=260)


# The lesson is done when these are correct. "stakeholders" is the choice question in the lesson.
REQUIRED = ["per-accent", "sources", "stakeholders"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str),
                     en="`results` has no column with that name. Its columns are `\"accent\"` and `\"correct\"`, in small letters, each between quotes.",
                     nl="`results` heeft geen kolom met die naam. De kolommen zijn `\"accent\"` en `\"correct\"`, in kleine letters, elk tussen aanhalingstekens.")
_ROWS_MEAN = Mistake("rows-mean", error=lambda e: isinstance(e, TypeError) and ("reduction" in str(e) or "to numeric" in str(e)),
                     en="`.mean()` of whole rows also takes the column `accent`, and that holds letters, not numbers. Take the mean of the column `correct` only.",
                     nl="`.mean()` van hele rijen neemt ook de kolom `accent` mee, en daarin staan letters, geen getallen. Neem alleen het gemiddelde van de kolom `correct`.")

_ACCENTS = 'accents = ["A"] * 12 + ["B"] * 5 + ["C"] * 3'
_CORRECT = "correct = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]"
_A_HEAD = (f"import pandas as pd\n\n{_ACCENTS}\n{_CORRECT}\n"
           'results = pd.DataFrame({"accent": accents, "correct": correct})\n\naccuracies = []\nfor accent in ["A", "B", "C"]:\n')
_A_START = _A_HEAD + "    ...\naccuracies\n"
_A_LINE = '    accuracies.append(results[results["accent"] == accent]["correct"].mean())\n'
_A = _A_HEAD + _A_LINE + "accuracies"


def _a(body: str) -> str:
    return _A_HEAD + body + "accuracies"


def _three_same(got, text) -> bool:
    """The same number three times: every round took the mean of all results."""
    try:
        values = [float(v) for v in got]
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and max(values) - min(values) < 1e-9


EXERCISES = {
    "per-accent": CodeBox(
        name="per-accent",
        start=_A_START,
        solution=_A,
        keep=[_ACCENTS],
        # The check also runs the box with other results: numbers typed in by hand do not change with them.
        others=[{"correct": Code("[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]")}],
        first=[_DOTS],
        mistakes=[
            _NO_COLUMN,
            _ROWS_MEAN,
            Mistake("overall", matches=_three_same,
                    en="Your last line gives `{got}`: three times the same number, the accuracy over all 20 recordings. In each round, keep only the rows of that accent with a mask, and take the mean of their `correct`.",
                    nl="Je laatste regel geeft `{got}`: drie keer hetzelfde getal, de accuracy over alle 20 opnames. Houd in elke ronde met een masker alleen de rijen van dat accent over, en neem het gemiddelde van hun `correct`."),
            Mistake("count", wrong=_a('    accuracies.append(results[results["accent"] == accent]["correct"].sum())\n'),
                    en="Your last line gives `{got}`: how many recordings of each accent the app got right. Accuracy is the share: the mean of `correct`, not its sum.",
                    nl="Je laatste regel geeft `{got}`: hoeveel opnames van elk accent de app goed had. Accuracy is het aandeel: het gemiddelde van `correct`, niet de som."),
            Mistake("share", wrong=_a('    accuracies.append(len(results[results["accent"] == accent]) / len(results))\n'),
                    en="Your last line gives `{got}`: the share of each accent in the test set, not how often the app is right for it. Take the mean of `correct` over the rows of that accent.",
                    nl="Je laatste regel geeft `{got}`: het aandeel van elk accent in de test set, niet hoe vaak de app het goed heeft. Neem het gemiddelde van `correct` over de rijen van dat accent."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: add the accuracy of this accent to `accuracies`.",
                      "nl": "Vervang de `...` door één regel: voeg de accuracy van dit accent toe aan `accuracies`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other results by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere uitkomsten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `accuracies` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `accuracies` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each round keep the rows of one accent with a mask, and add the mean of their `correct` to `accuracies`?",
                      "nl": "Je laatste regel geeft `{got}`. Houdt elke ronde met een masker de rijen van één accent over, en voegt hij het gemiddelde van hun `correct` toe aan `accuracies`?"},
            "other": {"en": "Right for these results. But with {where}, your last line gives `{got}`, and it should give about `[0.83, 0.4, 1.0]`. Do you work it out with `results` itself?",
                      "nl": "Goed voor deze uitkomsten. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `[0.83, 0.4, 1.0]` zijn. Reken je het uit met `results` zelf?"},
            "other_error": {"en": "Right for these results. But with {where}, your code stops with an error. Do you work it out with `results` itself?",
                            "nl": "Goed voor deze uitkomsten. Maar met {where} stopt je code met een fout. Reken je het uit met `results` zelf?"},
            "correct": {"en": "About 0.92 for accent A, 0.6 for B and 0.33 for C, while the app gets 0.75 of all recordings right. It does worst for accent C, only 3 of the 20 recordings, and the overall score hides that. The check also tried other results.",
                        "nl": "Ongeveer 0,92 voor accent A, 0,6 voor B en 0,33 voor C, terwijl de app 0,75 van alle opnames goed heeft. Het slechtst doet hij het bij accent C, maar 3 van de 20 opnames, en de totaalscore verbergt dat. De check probeerde ook andere uitkomsten."},
        },
    ),
    "sources": Sort(
        name="sources",
        groups={
            "measured": {"en": "Who is in the data: some groups are missing or rare", "nl": "Wie in de data staat: sommige groepen ontbreken of zijn zeldzaam"},
            "labeled": {"en": "Who gave the labels: their views end up in the labels", "nl": "Wie de labels gaf: hun kijk komt in de labels terecht"},
            "past": {"en": "The past: the data shows how things went, not how they should go", "nl": "Het verleden: de data laat zien hoe het ging, niet hoe het hoort"},
        },
        items={
            "faces": {"en": "**Celebrity photos.** A face dataset is made from photos of celebrities on the internet, and most of them have light skin.",
                      "nl": "**Foto's van bekende mensen.** Een dataset met gezichten bestaat uit foto's van bekende mensen op internet, en de meesten hebben een lichte huid."},
            "rude": {"en": "**Rude or not.** People from one region decide which messages are rude, and mark jokes in another dialect as rude more often.",
                     "nl": "**Grof of niet.** Mensen uit één regio bepalen welke berichten grof zijn, en noemen grappen in een ander dialect vaker grof."},
            "hiring": {"en": "**Ten years of hires.** A hiring model learns from ten years of a company's decisions on applicants, and in those years it hired men far more often than women with the same experience.",
                       "nl": "**Tien jaar aannemen.** Een model voor sollicitaties leert van tien jaar beslissingen van een bedrijf over sollicitanten, en in die jaren nam het mannen veel vaker aan dan vrouwen met dezelfde ervaring."},
            "volunteers": {"en": "**Volunteers.** A speech app is trained on recordings from students who signed up through one university.",
                           "nl": "**Vrijwilligers.** Een spraakapp is getraind op opnames van studenten die zich via één universiteit aanmeldden."},
            "pain": {"en": "**Pain scores.** The labels are the pain scores that staff wrote down, and they took the pain of some patients less seriously.",
                     "nl": "**Pijnscores.** De labels zijn de pijnscores die het personeel opschreef, en de pijn van sommige patiënten namen ze minder serieus."},
            "loans": {"en": "**Old loans.** A loan model learns from past applications and whether they were approved, and for decades people in some parts of town were turned down far more often, whatever their income.",
                      "nl": "**Oude leningen.** Een model voor leningen leert van aanvragen uit het verleden en of ze werden goedgekeurd, en tientallen jaren werden mensen in sommige wijken veel vaker afgewezen, wat hun inkomen ook was."},
        },
        solution={"faces": "measured", "rude": "labeled", "hiring": "past", "volunteers": "measured", "pain": "labeled", "loans": "past"},
        nudge={
            "faces": {"en": "Are the labels the problem here, or who is in the photos at all?",
                      "nl": "Zijn hier de labels het probleem, of wie er überhaupt op de foto's staat?"},
            "rude": {"en": "The messages are all there. Where does the skew come in: in the messages, or in the judgement on them?",
                     "nl": "De berichten zijn er allemaal. Waar komt de scheefheid binnen: in de berichten, of in het oordeel erover?"},
            "hiring": {"en": "The data records what really happened. Was that fair to begin with?",
                       "nl": "De data legt vast wat er echt gebeurde. Was dat om te beginnen eerlijk?"},
            "volunteers": {"en": "Who could end up in this data, and who could not?",
                           "nl": "Wie kon in deze data terechtkomen, en wie niet?"},
            "pain": {"en": "The patients are in the data. Who decided what their label says?",
                     "nl": "De patiënten staan in de data. Wie besliste wat hun label zegt?"},
            "loans": {"en": "Is this about who was measured, or about decisions that were already made in the past?",
                      "nl": "Gaat dit over wie er gemeten werd, of over beslissingen die vroeger al genomen zijn?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with the place where its skew comes from.",
                      "nl": "Zet elk geval bij de plek waar de scheefheid vandaan komt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs with another source.",
                      "nl": "Het geval *{item}* hoort bij een andere bron."},
            "correct": {"en": "All {n} are right. A skew can come from who is in the data, from who labeled it, or from the past it records; a model learns it in each case.",
                        "nl": "Alle {n} kloppen. Scheefheid kan komen van wie in de data staat, van wie de labels gaf, of van het verleden dat de data vastlegt; een model neemt die in elk geval over."},
        },
    ),
}

_SOLVED = {"faces": "measured", "rude": "labeled", "hiring": "past", "volunteers": "measured", "pain": "labeled", "loans": "past"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("per-accent", _A_START, "empty", None),
    ("per-accent", _A, "correct", None),
    ("per-accent", _a('    rows = results[results["accent"] == accent]\n    accuracies.append(rows["correct"].mean())\n'), "correct", None),
    ("per-accent", _a('    accuracies.append(results["correct"][results["accent"] == accent].mean())\n'), "correct", None),
    ("per-accent", _a('    accuracies.append(results["correct"].mean())\n'), "wrong", "overall"),
    ("per-accent", _a('    accuracies.append(results[results["accent"] == accent]["correct"].sum())\n'), "wrong", "count"),
    ("per-accent", _a('    accuracies.append(len(results[results["accent"] == accent]) / len(results))\n'), "wrong", "share"),
    ("per-accent", _A_HEAD + "    ...\naccuracies = [0.9166666666666666, 0.6, 0.3333333333333333]\naccuracies", "wrong", "dots-left"),
    ("per-accent", _a("    accuracies = [0.9166666666666666, 0.6, 0.3333333333333333]\n"), "almost", "other-value"),
    ("per-accent", _A.replace(_ACCENTS, 'accents = ["A"] * 10 + ["B"] * 5 + ["C"] * 5'), "wrong", "changed-line"),
    ("per-accent", _a('    accuracies.append(results[results["accents"] == accent]["correct"].mean())\n'), "error", "no-column"),
    ("per-accent", _a('    accuracies.append(results[results["accent"] == accent].mean())\n'), "error", "rows-mean"),
    ("sources", {}, "empty", None),
    ("sources", _SOLVED, "correct", None),
    ("sources", {"faces": "measured", "rude": "labeled"}, "wrong", "unfinished"),
    ("sources", {**_SOLVED, "rude": "measured"}, "wrong", "wrong:rude"),
    ("sources", {**_SOLVED, "hiring": "measured"}, "wrong", "wrong:hiring"),
    ("sources", {**_SOLVED, "pain": "past"}, "wrong", "wrong:pain"),
]
