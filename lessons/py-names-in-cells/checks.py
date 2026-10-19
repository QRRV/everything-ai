# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-names-in-cells: a name from another cell, and which cells can live side by side."""
import re

from everything_ai.check import CodeBox, Mistake, Sort

# What the lesson's own cell for exercise 1 makes. The learner may change it; the check cell passes the current value.
TICKET_PRICE = 12.5
NAMES = {"ticket_price": TICKET_PRICE}


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    """The same lines under the cell; 100 and 100.0 count as the same number."""
    if not (isinstance(a, str) and isinstance(b, str)) or len(_lines(a)) != len(_lines(b)):
        return False
    for x, y in zip(_lines(a), _lines(b)):
        try:
            if abs(float(x) - float(y)) > 1e-9:
                return False
        except ValueError:
            if x.strip() != y.strip():
                return False
    return True

# The lesson is done when these are correct. "fix" is the choice question in the lesson.
REQUIRED = ["trip_cost", "cell-pairs", "fix"]


_SAME_NAME = {"en": "Look at the names each cell makes with `=`. Is the same name made in both cells? See the picture.",
              "nl": "Kijk naar de namen die elke cel maakt met `=`. Wordt dezelfde naam in beide cellen gemaakt? Kijk naar het beeld."}
_USES = {"en": "Cell 2 only uses `price`; it does not make it. Which cell makes `price`, and may other cells use a name without `_`?",
         "nl": "Cel 2 gebruikt `price` alleen; hij maakt hem niet. Welke cel maakt `price`, en mogen andere cellen een naam zonder `_` gebruiken?"}
_UNDERSCORE = {"en": "Look at the names that start with `_`. Where can a cell see such a name? See the new words.",
               "nl": "Kijk naar de namen die met `_` beginnen. Waar kan een cel zo'n naam zien? Kijk bij de nieuwe woorden."}

EXERCISES = {
    "trip_cost": CodeBox(
        name="trip_cost",
        start="# ticket_price comes from the cell above\ntrip_cost = ...\nprint(trip_cost)",
        solution="trip_cost = ticket_price * 4 * 2\nprint(trip_cost)",
        outcome="stdout",
        compare=same_lines,
        names=NAMES,
        others=[{"ticket_price": 15}],
        first=[Mistake("made-again", matches=lambda got, text: re.search(r"^\s*ticket_price\s*=(?!=)", text, re.MULTILINE) is not None,
                       en="Your box makes `ticket_price` again, while the cell above already makes it. In a notebook cell, marimo would then stop. Remove that line, and use the name from the cell above.",
                       nl="Je vak maakt `ticket_price` opnieuw, terwijl de cel erboven hem al maakt. In een cel van een notebook zou marimo dan stoppen. Haal die regel weg, en gebruik de naam uit de cel erboven.")],
        mistakes=[
            Mistake("dots-left", syntax=r"\.\.\.",
                    en="There is still a `...` in your program. Replace it with the calculation for the whole trip.",
                    nl="Er staat nog een `...` in je programma. Vervang die door de berekening voor de hele reis."),
            Mistake("dots-left", matches=lambda got, text: "..." in text,
                    en="There is still a `...` in your program. Replace it with the calculation for the whole trip.",
                    nl="Er staat nog een `...` in je programma. Vervang die door de berekening voor de hele reis."),
            Mistake("one-way", wrong="print(ticket_price * 4)",
                    en="Your program shows `{got}`: four tickets for one way. The four friends also travel back.",
                    nl="Je programma laat `{got}` zien: vier enkeltjes. De vier vrienden reizen ook terug."),
            Mistake("one-person", wrong="print(ticket_price * 2)",
                    en="Your program shows `{got}`: there and back for one person. There are four friends.",
                    nl="Je programma laat `{got}` zien: heen en terug voor één persoon. Het zijn vier vrienden."),
            Mistake("one-ticket", wrong="print(ticket_price)",
                    en="Your program shows `{got}`: that is one ticket. Work out what four friends pay, there and back.",
                    nl="Je programma laat `{got}` zien: dat is één kaartje. Reken uit wat vier vrienden betalen, heen en terug."),
            Mistake("six-tickets", wrong="print(ticket_price * (4 + 2))",
                    en="Your program shows `{got}`: six tickets. Four friends with two tickets each is 4 times 2, not 4 plus 2.",
                    nl="Je programma laat `{got}` zien: zes kaartjes. Vier vrienden met elk twee kaartjes is 4 keer 2, niet 4 plus 2."),
            Mistake("added", wrong="print(ticket_price + 4 + 2)",
                    en="Your program shows `{got}`: you added the numbers. Four friends with two tickets each means multiplying, with `*`.",
                    nl="Je programma laat `{got}` zien: je hebt de getallen opgeteld. Vier vrienden met elk twee kaartjes betekent vermenigvuldigen, met `*`."),
            Mistake("word", matches=lambda got, text: got.strip() == "trip_cost",
                    en="Your program shows the word `trip_cost`, because it is between quotes. Remove the quotes, so `print` shows the value of the name.",
                    nl="Je programma laat het woord `trip_cost` zien, omdat het tussen aanhalingstekens staat. Haal de aanhalingstekens weg, dan laat `print` de waarde van de naam zien."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the calculation for the whole trip.",
                      "nl": "Vervang de `...` door de berekening voor de hele reis."},
            "cleared": {"en": "There is no code in the box any more. Type the two lines back: `trip_cost = ...` and `print(trip_cost)`, and fill in the `...`.",
                        "nl": "Er staat geen code meer in het vak. Typ de twee regels terug: `trip_cost = ...` en `print(trip_cost)`, en vul de `...` in."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet what the four friends pay. Each of them buys a ticket there and a ticket back: by which numbers do you multiply `ticket_price`?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet wat de vier vrienden betalen. Elk van hen koopt een kaartje heen en een kaartje terug: met welke getallen vermenigvuldig je `ticket_price`?"},
            "other": {"en": "Right for this price. But with {where} in the cell above, your program shows `{got}`, and it should show `{want}`. Use the name `ticket_price` instead of a number, so your box follows the price.",
                      "nl": "Goed voor deze prijs. Maar met {where} in de cel erboven laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Gebruik de naam `ticket_price` in plaats van een getal, dan volgt je vak de prijs."},
            "correct": {"en": "Your program shows `{got}`: four friends, two tickets each. It uses `ticket_price` from the cell above, so it also follows another price.",
                        "nl": "Je programma laat `{got}` zien: vier vrienden, elk twee kaartjes. Het gebruikt `ticket_price` uit de cel erboven, dus het volgt ook een andere prijs."},
        },
    ),
    "cell-pairs": Sort(
        name="cell-pairs",
        groups={
            "runs": {"en": "Both cells run", "nl": "Beide cellen draaien"},
            "twice": {"en": "marimo stops: a name is made in two cells", "nl": "marimo stopt: een naam wordt in twee cellen gemaakt"},
            "unknown": {"en": "Cell 2 does not know the name", "nl": "Cel 2 kent de naam niet"},
        },
        items={
            "same": {"en": "Cell 1: `total = 3`. Cell 2: `total = 5`.",
                     "nl": "Cel 1: `total = 3`. Cel 2: `total = 5`."},
            "uses": {"en": "Cell 1: `price = 3`. Cell 2: `cost = price * 2`.",
                     "nl": "Cel 1: `price = 3`. Cel 2: `cost = price * 2`."},
            "own": {"en": "Cell 1: `_total = 3`. Cell 2: `_total = 5`.",
                    "nl": "Cel 1: `_total = 3`. Cel 2: `_total = 5`."},
            "again": {"en": "Cell 1: `price = 3`. Cell 2: `price = price * 2`.",
                      "nl": "Cel 1: `price = 3`. Cel 2: `price = price * 2`."},
            "hidden": {"en": "Cell 1: `_n = 4`. Cell 2: `print(_n)`.",
                       "nl": "Cel 1: `_n = 4`. Cel 2: `print(_n)`."},
            "helper": {"en": "Cell 1: `_rate = 2`. Cell 2: `pay = _rate * 8`.",
                       "nl": "Cel 1: `_rate = 2`. Cel 2: `pay = _rate * 8`."},
        },
        solution={"same": "twice", "uses": "runs", "own": "runs", "again": "twice", "hidden": "unknown", "helper": "unknown"},
        nudge={"same": _SAME_NAME, "uses": _USES, "again": _SAME_NAME,
               "own": _UNDERSCORE, "hidden": _UNDERSCORE, "helper": _UNDERSCORE},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pairs of cells again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de paren cellen opnieuw uit."},
            "empty": {"en": "Pick a group for each pair of cells.",
                      "nl": "Kies voor elk paar cellen een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. A name without `_` comes from one cell and every cell can use it; a name with `_` stays in its own cell.",
                        "nl": "Alle {n} kloppen. Een naam zonder `_` komt uit één cel en elke cel mag hem gebruiken; een naam met `_` blijft in zijn eigen cel."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner made or chose, state, known mistake).
_RIGHT = {"same": "twice", "uses": "runs", "own": "runs", "again": "twice", "hidden": "unknown", "helper": "unknown"}
CASES = [
    ("trip_cost", "# ticket_price comes from the cell above\ntrip_cost = ...\nprint(trip_cost)", "empty", None),
    ("trip_cost", "", "empty", None),
    ("trip_cost", "trip_cost = ticket_price * 4 * 2\nprint(trip_cost)", "correct", None),
    ("trip_cost", "_friends = 4\n_tickets = 2\ntrip_cost = ticket_price * _friends * _tickets\nprint(trip_cost)", "correct", None),
    ("trip_cost", "trip_cost = 100\nprint(trip_cost)", "almost", "other-value"),
    ("trip_cost", "trip_cost = 12.5 * 8\nprint(trip_cost)", "almost", "other-value"),
    ("trip_cost", "ticket_price = 12.5\ntrip_cost = ticket_price * 8\nprint(trip_cost)", "wrong", "made-again"),
    ("trip_cost", "trip_cost = ticket_price * 4\nprint(trip_cost)", "wrong", "one-way"),
    ("trip_cost", "trip_cost = ticket_price * 2\nprint(trip_cost)", "wrong", "one-person"),
    ("trip_cost", "trip_cost = ticket_price\nprint(trip_cost)", "wrong", "one-ticket"),
    ("trip_cost", "trip_cost = ticket_price * (4 + 2)\nprint(trip_cost)", "wrong", "six-tickets"),
    ("trip_cost", "trip_cost = ticket_price + 4 + 2\nprint(trip_cost)", "wrong", "added"),
    ("trip_cost", "trip_cost = ticket_price * 8\nprint(\"trip_cost\")", "wrong", "word"),
    ("trip_cost", "trip_cost = tiket_price * 8\nprint(trip_cost)", "error", None),
    ("trip_cost", "trip_cost = ...\nprint(trip_cost)", "wrong", "dots-left"),
    ("trip_cost", "trip_cost = ticket_price * 3\nprint(trip_cost)", "wrong", None),
    ("cell-pairs", {}, "empty", None),
    ("cell-pairs", _RIGHT, "correct", None),
    ("cell-pairs", {"same": "twice", "uses": "runs"}, "wrong", "unfinished"),
    ("cell-pairs", {**_RIGHT, "again": "runs"}, "wrong", "wrong:again"),
    ("cell-pairs", {**_RIGHT, "own": "twice"}, "wrong", "wrong:own"),
    ("cell-pairs", {**_RIGHT, "helper": "runs"}, "wrong", "wrong:helper"),
]
