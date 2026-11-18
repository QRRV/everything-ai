# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-dataset-consent: which sets of text pass both questions, a license and the consent of the
people in them, sorting cases by the question they fail, and the pictures of the two questions."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Five sets of text that a team wants to train on, made up by the course: the license of each ("none" when it has
# none), and whether the people who wrote the text agreed to its use for training.
LICENSES = {"forum": "none", "survey": "CC BY", "reviews": "CC BY-SA", "chats": "none", "interviews": "CC BY"}
AGREED = {"forum": False, "survey": True, "reviews": False, "chats": True, "interviews": True}


def usable(licenses: dict, agreed: dict) -> list:
    """The sets with a license whose people agreed, in the order of `licenses`."""
    return [n for n in licenses if licenses[n] != "none" and agreed[n]]


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The two questions as a grid of four cases: a license or none (columns), the people agreed or were not asked
    (rows), with an example in each cell; the cell where both pass says so. `labels` gives "title", "columns" (two),
    "rows" (two), "cells" (four texts, row by row), "both" and "alt"."""
    def cell(text: str, both: bool = False) -> str:
        mark = (f'<div style="{_UI};font-size:0.82rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.15rem">'
                f'{html.escape(labels["both"])}</div>') if both else ""
        return (f'<div style="border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);padding:0.4rem 0.6rem;'
                f'min-width:0">{mark}<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);line-height:1.35">'
                f'{html.escape(text)}</div></div>')

    def head(text: str, side: bool = False) -> str:
        align = "left" if side else "center"
        return (f'<div style="{_UI};font-size:0.82rem;font-weight:600;color:var(--ea-ink);text-align:{align};'
                f'align-self:center;min-width:0">{html.escape(text)}</div>')

    c = labels["cells"]
    parts = ["<div></div>", head(labels["columns"][0]), head(labels["columns"][1]),
             head(labels["rows"][0], True), cell(c[0], both=True), cell(c[1]),
             head(labels["rows"][1], True), cell(c[2]), cell(c[3])]
    body = (f'<div role="img" aria-label="{html.escape(labels["alt"])}" style="display:grid;'
            f'grid-template-columns:minmax(4.5rem,0.6fr) 1fr 1fr;gap:0.5rem;max-width:40rem">{"".join(parts)}</div>')
    return yours.figure(labels["title"], body, alt=labels["alt"])


def _usable_picture(seen) -> str:
    """Each set with its license and consent, whether the learner's list holds it, and whether it should."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    right = usable(LICENSES, AGREED)
    got = [str(x) for x in mine.value]
    yes, no = seen.word({"en": "agreed", "nl": "akkoord"}), seen.word({"en": "not asked", "nl": "niet gevraagd"})
    rows = []
    for name in LICENSES:
        listed = seen.word({"en": "in your list", "nl": "in je lijst"}) if name in got else ""
        should = name in right
        mark = "" if (name in got) == should else seen.word(
            {"en": "(the course: in it)" if should else "(the course: not in it)",
             "nl": "(de course: erin)" if should else "(de course: niet erin)"})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{name}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{LICENSES[name]}</td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{yes if AGREED[name] else no}</td>'
                    f'<td style="padding:0.15rem 0.8rem;font-weight:600;color:var(--ea-ink)">{listed}</td>'
                    f'<td style="padding:0.15rem 0;color:var(--ea-ink-2)">{mark}</td></tr>')
    extra = [x for x in got if x not in LICENSES]
    if extra:
        rows.append(f'<tr><td colspan="5" style="padding:0.15rem 0;color:var(--ea-ink-2)">'
                    + seen.word({"en": "also in your list: ", "nl": "ook in je lijst: "})
                    + html.escape(", ".join(extra)) + "</td></tr>")
    title = seen.word({"en": "License, consent and your list", "nl": "Licentie, toestemming en jouw lijst"})
    table = f'<table style="{_UI};font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "review" is the choice question in the lesson.
REQUIRED = ["usable", "questions", "review"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values: dict) -> str:
    """`name = {...}` on one line, or over two inside the brackets when one line would be too wide for the box."""
    one = f"{name} = {values!r}".replace("'", '"')
    if len(one) <= 100:
        return one
    items = [f"{k!r}: {v!r}".replace("'", '"') for k, v in values.items()]
    half = (len(items) + 1) // 2
    return f"{name} = {{" + ", ".join(items[:half]) + ",\n" + " " * (len(name) + 4) + ", ".join(items[half:]) + "}"


_LICENSES_LINE = _line("licenses", LICENSES)
_HEAD = ("# the license of each set, and whether its writers agreed\n" + _LICENSES_LINE + "\n" + _line("agreed", AGREED)
         + "\n\nusable = []\nfor name in licenses:\n    if licenses[name] != \"none\":\n")
_START = _HEAD + "        ...\nusable\n"
_LINES = "        if agreed[name]:\n            usable.append(name)\n"
_SOLUTION = _HEAD + _LINES + "usable"


def _u(body: str) -> str:
    return _HEAD + body + "usable"


_LOOP = "usable = []\nfor name in licenses:\n"


def _loop(body: str) -> str:
    """The box with the whole body of the loop replaced."""
    return _HEAD.split(_LOOP)[0] + _LOOP + body + "usable"


EXERCISES = {
    "usable": CodeBox(
        name="usable",
        picture=_usable_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_LICENSES_LINE],
        # The check also runs the box as if other writers had agreed.
        others=[{"agreed": Code('{"forum": True, "survey": True, "reviews": True, "chats": False, "interviews": False}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("license-only", wrong=_u("        usable.append(name)\n"),
                    en="Your last line gives `{got}`: every set with a license. But the writers of `reviews` were never asked. Add a second `if`, inside the first, that checks `agreed[name]`.",
                    nl="Je laatste regel geeft `{got}`: elke set met een licentie. Maar de schrijvers van `reviews` zijn nooit gevraagd. Zet een tweede `if` in de eerste, die `agreed[name]` controleert."),
            Mistake("consent-only", wrong=_loop("    if agreed[name]:\n        usable.append(name)\n"),
                    en="Your last line gives `{got}`: every set whose writers agreed. But `chats` has no license. Keep the first `if`, and put the check of `agreed` inside it.",
                    nl="Je laatste regel geeft `{got}`: elke set waarvan de schrijvers akkoord gingen. Maar `chats` heeft geen licentie. Houd de eerste `if`, en zet de controle van `agreed` erin."),
            Mistake("licenses-appended", wrong=_u("        if agreed[name]:\n            usable.append(licenses[name])\n"),
                    en="Your last line gives `{got}`: those are licenses. Append the name of the set, `name`.",
                    nl="Je laatste regel geeft `{got}`: dat zijn licenties. Voeg de naam van de set toe, `name`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: a second `if`, about the writers of this set, and under it an `append`.",
                      "nl": "Vervang de `...` door twee regels: een tweede `if`, over de schrijvers van deze set, en daaronder een `append`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other answers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere antwoorden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `usable` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `usable` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your code append a set only when it has a license and its writers agreed?",
                      "nl": "Je laatste regel geeft `{got}`. Voegt je code een set alleen toe als die een licentie heeft en de schrijvers akkoord gingen?"},
            "other": {"en": "Right for these sets. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you check `agreed` itself?",
                      "nl": "Goed voor deze sets. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Controleer je `agreed` zelf?"},
            "other_error": {"en": "Right for these sets. But with {where}, your code stops with an error. Do you check `agreed` itself?",
                            "nl": "Goed voor deze sets. Maar met {where} stopt je code met een fout. Controleer je `agreed` zelf?"},
            "correct": {"en": "`{got}`: two of the five pass both questions. `reviews` has a license, but its writers were not asked; `chats` has their consent, but no license. The check also tried other answers.",
                        "nl": "`{got}`: twee van de vijf slagen voor beide vragen. `reviews` heeft een licentie, maar de schrijvers zijn niet gevraagd; `chats` heeft hun toestemming, maar geen licentie. De check probeerde ook andere antwoorden."},
        },
    ),
    "questions": Sort(
        name="questions",
        groups={
            "both": {"en": "Both questions pass", "nl": "Beide vragen in orde"},
            "license": {"en": "The license is missing", "nl": "De licentie ontbreekt"},
            "consent": {"en": "The consent is missing", "nl": "De toestemming ontbreekt"},
        },
        items={
            "translations": {"en": "**The translations.** Translators are paid to translate sentences to train a model, sign that their work may be used for that, and the set comes out under CC BY.",
                             "nl": "**De vertalingen.** Vertalers worden betaald om zinnen te vertalen om een model te trainen, tekenen dat hun werk daarvoor gebruikt mag worden, en de set komt uit onder CC BY."},
            "recipes": {"en": "**The recipes.** Cooks send in recipes for a cooking model and tick that they may be used for it; the set comes out under CC0.",
                        "nl": "**De recepten.** Koks sturen recepten in voor een kookmodel en vinken aan dat ze daarvoor gebruikt mogen worden; de set komt uit onder CC0."},
            "school": {"en": "**The school photos.** Students agree that their class photos may be used to train a model, but the photographer, who holds the rights, gives no license.",
                       "nl": "**De schoolfoto's.** Leerlingen gaan akkoord dat hun schoolfoto's gebruikt mogen worden om een model te trainen, maar de fotograaf, die de rechten heeft, geeft geen licentie."},
            "game": {"en": "**The game.** Players agree that their moves may be used to train a model, but the game's maker, who holds the rights to the records of the games, puts them online without a license.",
                     "nl": "**Het spel.** Spelers gaan akkoord dat hun zetten gebruikt mogen worden om een model te trainen, maar de maker van het spel, die de rechten op de verslagen van de partijen heeft, zet ze online zonder licentie."},
            "support": {"en": "**The support chats.** Its terms of use give a company the rights to its customers' chats with its help desk, and it releases them under CC BY; the customers were never asked about training.",
                        "nl": "**De klantchats.** Volgens zijn gebruiksvoorwaarden heeft een bedrijf de rechten op de chats van zijn klanten met de helpdesk, en het brengt ze uit onder CC BY; de klanten is nooit iets over trainen gevraagd."},
            "podcasts": {"en": "**The podcasts.** A podcast platform holds the rights to all its episodes and releases them under CC BY for training; the guests who speak in them were never asked.",
                         "nl": "**De podcasts.** Een podcastplatform heeft de rechten op al zijn afleveringen en brengt ze uit onder CC BY om op te trainen; de gasten die erin spreken, zijn nooit gevraagd."},
        },
        solution={"translations": "both", "recipes": "both", "school": "license", "game": "license", "support": "consent", "podcasts": "consent"},
        nudge={
            "translations": {"en": "Did the translators agree, and is there a license?", "nl": "Gingen de vertalers akkoord, en is er een licentie?"},
            "recipes": {"en": "What did the cooks tick, and under which license does the set come out?", "nl": "Wat vinkten de koks aan, en onder welke licentie komt de set uit?"},
            "school": {"en": "The students agreed. Who holds the rights, and what did they leave out?", "nl": "De leerlingen gingen akkoord. Wie heeft de rechten, en wat liet die weg?"},
            "game": {"en": "The players agreed. May anyone else use the moves?", "nl": "De spelers gingen akkoord. Mag iemand anders de zetten gebruiken?"},
            "support": {"en": "The company holds the rights. Whose words are they, and were those people asked?", "nl": "Het bedrijf heeft de rechten. Van wie zijn de woorden, en zijn die mensen gevraagd?"},
            "podcasts": {"en": "The platform may give a license. Who was not asked?", "nl": "Het platform mag een licentie geven. Wie is er niet gevraagd?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with the question it fails, or with *both questions pass*.",
                      "nl": "Zet elk geval bij de vraag waarvoor het niet slaagt, of bij *beide vragen in orde*."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs with another answer.",
                      "nl": "Het geval *{item}* hoort bij een ander antwoord."},
            "correct": {"en": "All {n} are right. A license comes from whoever holds the rights to the data, consent from the people the data are about, and those are often not the same people.",
                        "nl": "Alle {n} kloppen. Een licentie komt van wie de rechten op de data heeft, toestemming van de mensen over wie de data gaan, en dat zijn vaak niet dezelfde mensen."},
        },
    ),
}

_SORTED = {"translations": "both", "recipes": "both", "school": "license", "game": "license", "support": "consent", "podcasts": "consent"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("usable", _START, "empty", None),
    ("usable", _SOLUTION, "correct", None),
    ("usable", _u("        if agreed[name] == True:\n            usable.append(name)\n"), "correct", None),
    ("usable", _u("        usable.append(name)\n"), "wrong", "license-only"),
    ("usable", _loop("    if agreed[name]:\n        usable.append(name)\n"), "wrong", "consent-only"),
    ("usable", _u("        if agreed[name]:\n            usable.append(licenses[name])\n"), "wrong", "licenses-appended"),
    ("usable", _u("        if agreed[name] == False:\n            usable.append(name)\n"), "wrong", None),
    ("usable", _SOLUTION.replace(_LICENSES_LINE, _LICENSES_LINE.replace('"chats": "none"', '"chats": "CC0"')), "wrong", "changed-line"),
    ("usable", _HEAD + "        pass\nusable = ['survey', 'interviews']\nusable", "almost", "other-value"),
    ("usable", _HEAD + "        ...\nusable = []\nusable", "wrong", "dots-left"),
    ("questions", {}, "empty", None),
    ("questions", _SORTED, "correct", None),
    ("questions", {"recipes": "both", "podcasts": "consent"}, "wrong", "unfinished"),
    ("questions", {**_SORTED, "translations": "consent"}, "wrong", "wrong:translations"),
    ("questions", {**_SORTED, "game": "consent"}, "wrong", "wrong:game"),
    ("questions", {**_SORTED, "support": "license"}, "wrong", "wrong:support"),
]
