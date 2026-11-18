# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-labels-tay-cambridge: what share of the people whose data a made-up quiz app collects
installed it themselves, sorting the facts of three cases, and the pictures of 1,000 people of whom a few agreed."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Cambridge Analytica: the British Information Commissioner says the quiz app was used by some 300,000 Facebook users
# worldwide; Facebook estimated that the data of up to 87 million people was shared.
INSTALLED_WORLDWIDE = 300_000
REACHED_WORLDWIDE = 87_000_000

# A made-up quiz app: the people who installed it and how many friends each has, counted as if no two of them have a
# friend in common. The worked example has three, the exercise four.
EXAMPLE = {"A": 200, "B": 350, "C": 150}
FRIENDS = {"A": 410, "B": 120, "C": 650, "D": 215}


def reached(friends: dict) -> int:
    """Everyone whose data the app collects: the people who installed it and all their friends."""
    return sum(1 + n for n in friends.values())


def share(friends: dict) -> float:
    """The share of the people whose data the app collects who installed it themselves."""
    return len(friends) / reached(friends)


_UI = "font-family:var(--ea-font-ui)"


def _grid(filled: int, alt: str) -> str:
    """1,000 small squares in 25 rows of 40; the first `filled` of them in ink, the rest in the color of a line."""
    cells = []
    for i in range(1000):
        x, y = (i % 40) * 11 + 1, (i // 40) * 11 + 1
        fill = "var(--ea-ink)" if i < filled else "var(--ea-line)"
        cells.append(f'<rect x="{x}" y="{y}" width="9" height="9" rx="1.5" style="fill:{fill}"/>')
    return (f'<svg role="img" aria-label="{html.escape(alt)}" viewBox="0 0 441 276" width="100%" '
            f'style="max-width:441px;display:block">{"".join(cells)}</svg>')


def _key(words: tuple) -> str:
    def box(fill: str) -> str:
        return (f'<svg width="12" height="12" aria-hidden="true"><rect x="1" y="1" width="10" height="10" rx="1.5" '
                f'style="fill:{fill}"/></svg>')
    return (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.4rem">'
            f'<span>{box("var(--ea-ink)")} {html.escape(words[0])}</span><span>{box("var(--ea-line)")} {html.escape(words[1])}</span></div>')


def picture(labels: dict) -> str:
    """Of every 1,000 people whose data went to Cambridge Analytica, the few who installed the quiz app. `labels`
    gives "title", "installed", "others" and "alt"."""
    filled = round(INSTALLED_WORLDWIDE / REACHED_WORLDWIDE * 1000)
    return yours.figure(labels["title"], _grid(filled, labels["alt"]) + _key((labels["installed"], labels["others"])))


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _share_picture(seen) -> str:
    """The learner's `share` as 1,000 people, of whom the share that installed the app is dark, with the value above."""
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    got = _number(mine.value)
    if got is None:
        return ""
    filled = max(0, min(1000, round(got * 1000)))
    words = {"en": ("installed the app", "a friend's data only", "Your share, as 1,000 people"),
             "nl": ("installeerde de app", "alleen data via een vriend", "Jouw aandeel, als 1.000 mensen")}[seen.lang]
    said = (f'<div style="{_UI};font-size:0.9rem;color:var(--ea-ink);margin-bottom:0.3rem">'
            f'<code>share = {html.escape(f"{got:.5f}")}</code></div>')
    alt = {"en": f"1,000 squares, of which {filled} are marked: the people who installed the app.",
           "nl": f"1.000 vakjes, waarvan {filled} gemarkeerd: de mensen die de app installeerden."}[seen.lang]
    return yours.figure(words[2], said + _grid(filled, alt) + _key(words[:2]))


# The lesson is done when these are correct. "claims" is the choice question in the lesson.
REQUIRED = ["friends", "kinds", "claims"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_FRIENDS_LINE = _line("friends", FRIENDS)
_HEAD = "# for each person who installed the quiz app, how many friends they have, made up by the course\n" + _FRIENDS_LINE + "\n\n"
_START = _HEAD + "...\nshare\n"
_BODY = ("reached = 0\nfor person in friends:\n    reached = reached + 1 + friends[person]\n"
         "share = len(friends) / reached\n")
_SOLUTION = _HEAD + _BODY + "share"


def _s(body: str) -> str:
    return _HEAD + body + "share"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


EXERCISES = {
    "friends": CodeBox(
        name="friends",
        picture=_share_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_FRIENDS_LINE],
        # The check also runs the box with other people and other numbers of friends.
        others=[{"friends": {"A": 90, "B": 40, "C": 300, "D": 25, "E": 140}}],
        first=[_DOTS],
        digits=5,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_friends`. In the box the names have none: `friends`, `reached`, `person`, `share`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_friends`. In het vak hebben de namen dat niet: `friends`, `reached`, `person`, `share`."),
            Mistake("no-installers", wrong=_swap("reached + 1 + friends[person]", "reached + friends[person]"),
                    en="Your last line gives `{got}`: a little too high. `reached` counts only the friends; the people who installed the app are among the people reached too. Add 1 for each of them.",
                    nl="Je laatste regel geeft `{got}`: iets te hoog. `reached` telt alleen de vrienden; de mensen die de app installeerden horen ook bij de mensen van wie data verzameld wordt. Tel voor elk van hen 1 op."),
            Mistake("upside-down", wrong=_swap("len(friends) / reached", "reached / len(friends)"),
                    en="Your last line gives `{got}`: how many people the app reaches for each person who installed it. The share is the other way round: `len(friends)` divided by `reached`.",
                    nl="Je laatste regel geeft `{got}`: hoeveel mensen de app bereikt per persoon die hem installeerde. Het aandeel is andersom: `len(friends)` gedeeld door `reached`."),
            Mistake("counted-people", wrong=_swap("reached + 1 + friends[person]", "reached + 1"),
                    en="Your last line gives `{got}`: `reached` counts only the people who installed the app. Add the friends of each of them too, from `friends[person]`.",
                    nl="Je laatste regel geeft `{got}`: `reached` telt alleen de mensen die de app installeerden. Tel ook de vrienden van elk van hen op, uit `friends[person]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines of the worked example: count in `reached` everyone whose data the app collects, and divide the number of people who installed it by that for `share`.",
                      "nl": "Vervang de `...` door de regels van het uitgewerkte voorbeeld: tel in `reached` iedereen van wie de app data verzamelt, en deel het aantal mensen dat hem installeerde daardoor voor `share`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line of `friends` exactly as it was: put it back. The check tries other numbers of friends by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel van `friends` nodig zoals die was: zet hem terug. De check probeert zelf andere aantallen vrienden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `reached` count, for each person who installed the app, that person and their friends, and is `share` the number of people who installed it divided by `reached`?",
                      "nl": "Je laatste regel geeft `{got}`. Telt `reached` voor elke persoon die de app installeerde die persoon en de vrienden, en is `share` het aantal mensen dat hem installeerde gedeeld door `reached`?"},
            "other": {"en": "Right for these people. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `friends` itself, and `len(friends)` for the number of people who installed the app?",
                      "nl": "Goed voor deze mensen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `friends` zelf, en `len(friends)` voor het aantal mensen dat de app installeerde?"},
            "other_error": {"en": "Right for these people. But with {where}, your code stops with an error. Do you use `friends` itself?",
                            "nl": "Goed voor deze mensen. Maar met {where} stopt je code met een fout. Gebruik je `friends` zelf?"},
            "correct": {"en": "`{got}`: of the 1,399 people whose data this app collects, 4 installed it themselves, about 3 in every 1,000. The other 1,395 never used the app. For Cambridge Analytica the numbers were about 300,000 and up to 87 million: also about 3 in every 1,000.",
                        "nl": "`{got}`: van de 1.399 mensen van wie deze app data verzamelt, installeerden er 4 hem zelf, ongeveer 3 op de 1.000. De andere 1.395 gebruikten de app nooit. Bij Cambridge Analytica waren de getallen ongeveer 300.000 en maximaal 87 miljoen: ook ongeveer 3 op de 1.000."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "says": {"en": "Harm in what a system says or shows about people", "nl": "Schade in wat een systeem over mensen zegt of laat zien"},
            "data": {"en": "Harm in what is done with data", "nl": "Schade in wat er met data gebeurt"},
            "after": {"en": "What was done about it", "nl": "Wat eraan gedaan werd"},
        },
        items={
            "label": {"en": "**The label.** Google Photos labeled photos of two Black friends with the name of an ape.",
                      "nl": "**Het label.** Google Photos gaf foto's van twee zwarte vrienden de naam van een aap."},
            "repeat": {"en": "**Repeat after me.** Users got Tay to post racist messages by asking it to repeat them.",
                       "nl": "**Repeat after me.** Gebruikers lieten Tay racistische berichten plaatsen door te vragen ze te herhalen."},
            "friends": {"en": "**The friends.** The quiz app also collected data of the friends of the people who installed it.",
                        "nl": "**De vrienden.** De quizapp verzamelde ook data van de vrienden van de mensen die hem installeerden."},
            "promise": {"en": "**The promise.** Users were told the app would not collect their name; according to the FTC, that was false.",
                        "nl": "**De belofte.** Gebruikers kregen te horen dat de app hun naam niet zou verzamelen; volgens de FTC was dat niet waar."},
            "removed": {"en": "**The missing labels.** Google stopped labeling anything as an ape, and in 2023 that was still so.",
                        "nl": "**De verdwenen labels.** Google hield op met iets als aap te labelen, en in 2023 was dat nog steeds zo."},
            "fine": {"en": "**The fine.** In 2019 the FTC fined Facebook 5 billion dollars for breaking an earlier order about privacy.",
                     "nl": "**De boete.** In 2019 gaf de FTC Facebook een boete van 5 miljard dollar voor het breken van een eerder bevel over privacy."},
        },
        solution={"label": "says", "repeat": "says", "friends": "data", "promise": "data", "removed": "after", "fine": "after"},
        nudge={
            "label": {"en": "Did the label decide anything about the two friends, or say something about them?", "nl": "Besliste het label iets over de twee vrienden, of zei het iets over hen?"},
            "repeat": {"en": "Is this about what Tay posted, or about data Tay collected?", "nl": "Gaat dit over wat Tay plaatste, of over data die Tay verzamelde?"},
            "friends": {"en": "Did the friends agree to anything?", "nl": "Stemden de vrienden ergens mee in?"},
            "promise": {"en": "Is this about what the app said on the screen, or about what happened to the data?", "nl": "Gaat dit over wat de app op het scherm zei, of over wat er met de data gebeurde?"},
            "removed": {"en": "Is this the harm, or the answer to it?", "nl": "Is dit de schade, of het antwoord erop?"},
            "fine": {"en": "Did the fine harm the users, or answer the harm?", "nl": "Deed de boete de gebruikers schade, of was het een antwoord op de schade?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the facts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de feiten opnieuw uit."},
            "empty": {"en": "Put each fact under what it is about.",
                      "nl": "Zet elk feit bij waar het over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The fact *{item}* belongs somewhere else.",
                      "nl": "Het feit *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. Notice the answers: Google removed the labels instead of fixing the model, and the fine came years after the data was taken.",
                        "nl": "Alle {n} kloppen. Let op de antwoorden: Google haalde de labels weg in plaats van het model te verbeteren, en de boete kwam jaren nadat de data was verzameld."},
        },
    ),
}

_SORTED = {"label": "says", "repeat": "says", "friends": "data", "promise": "data", "removed": "after", "fine": "after"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("friends", _START, "empty", None),
    ("friends", _SOLUTION, "correct", None),
    ("friends", _swap("    reached = reached + 1 + friends[person]\n",
                      "    reached = reached + friends[person]\nreached = reached + len(friends)\n"), "correct", None),
    ("friends", _s(_BODY.replace("in friends", "in _friends")), "error", "underscore"),
    ("friends", _swap("reached + 1 + friends[person]", "reached + friends[person]"), "wrong", "no-installers"),
    ("friends", _swap("len(friends) / reached", "reached / len(friends)"), "wrong", "upside-down"),
    ("friends", _swap("reached + 1 + friends[person]", "reached + 1"), "wrong", "counted-people"),
    ("friends", _s("share = 0.5\n"), "wrong", None),
    ("friends", _SOLUTION.replace('"A": 410', '"A": 400'), "wrong", "changed-line"),
    ("friends", _swap("len(friends) / reached", "4 / reached"), "almost", "other-value"),
    ("friends", _s("...\nshare = 0\n"), "wrong", "dots-left"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"label": "says", "fine": "after"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "label": "data"}, "wrong", "wrong:label"),
    ("kinds", {**_SORTED, "promise": "says"}, "wrong", "wrong:promise"),
    ("kinds", {**_SORTED, "removed": "says"}, "wrong", "wrong:removed"),
]
