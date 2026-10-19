# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-while. Route: course/route/lessen/stam-1-basis.yaml. Checks en de loop in het beeld: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-while", "nl")
    C = L.checks   # de loop in het beeld, controle voor controle


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="Wat laat dit programma zien?\n\n```python\ntotal = 0\nfor n in [2, 3, 4]:\n    total = total + n\nprint(total)\n```",
        options={
            "nine": "`9`",
            "four": "`4`",
            "lines": "`2`, `5` en `9`, elk op een eigen regel",
        },
        correct="nine",
        why={
            "nine": "Elke ronde telt `n` op bij `total`: 2, dan 5, dan 9. De `print` springt niet in, dus die draait één keer, na de loop.",
            "four": "`total` krijgt niet alleen het laatste item: elke ronde krijgt hij wat hij al was plus `n`. Kijk nog eens naar het beeld in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "lines": "De `print` staat helemaal links, na de loop, dus die draait één keer. Kijk nog eens naar het recept in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Welke getallen geeft `range(3)`?",
        options={
            "one-three": "1, 2 en 3",
            "zero-two": "0, 1 en 2",
            "zero-three": "0, 1, 2 en 3",
        },
        correct="zero-two",
        why={
            "one-three": "Met één getal begint `range` bij 0. Kijk nog eens naar het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "zero-three": "`range` stopt net voor zijn stop. Kijk nog eens naar het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "zero-two": "Met één getal begint `range` bij 0 en stopt hij net voor 3.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een `for`-loop gaat een lijst langs die je al hebt. Maar soms weet je van tevoren niet hoeveel rondes je nodig hebt: je wilt doorgaan tot er iets gebeurt. Sparen tot je genoeg hebt; opnieuw proberen tot het lukt. Later in de course blijft een programma dat leert oefenen tot zijn fouten klein genoeg zijn. Dat is een `while`-loop.

    **Een voorbeeld uit het dagelijks leven.** "Roer tot de saus dik wordt." Het recept zegt niet hoe vaak je moet roeren. Vóór elke keer roeren kijk je: is hij al dik? Nee? Nog een keer roeren. Ja? Stoppen.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nn = 1\nwhile n < 20:\n    n = n * 2\nprint(n)\n```",
        options={
            "sixteen": "`16`",
            "thirty-two": "`32`",
            "twenty": "`20`",
        },
        reveal="**`32`.** De voorwaarde wordt vóór elke ronde gecontroleerd. Als `n` 16 is, is `16 < 20` nog `True`, dus maakt nog één ronde er 32 van. Dan is `32 < 20` `False`, en stopt de loop. Je ziet elke controle in het beeld hieronder.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    `while` voert zijn ingesprongen regels steeds opnieuw uit, zolang de voorwaarde `True` is. Vóór elke ronde controleert hij de voorwaarde opnieuw.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **while** | voert de ingesprongen regels eronder steeds opnieuw uit, zolang de voorwaarde erna `True` is; hij controleert vóór elke ronde | `while n < 20:` |
    | **break** | stopt een loop meteen, ook een `for`-loop; Python gaat verder met de eerste regel na de loop | `break` |
    | **eindeloze loop** | een loop waarvan de voorwaarde nooit `False` wordt, zodat hij nooit stopt | |

    Iets in de ingesprongen regels moet veranderen waar de voorwaarde naar kijkt, anders gaat de loop eeuwig door. Blijft een cel draaien? Klik op de stopknop ernaast, ■. In het vak van een oefening stopt de check je code na 2 seconden.
    """)
    return


@app.cell(hide_code=True)
def picture_checks():
    mo.vstack([
        mo.md("```python\nn = 1\nwhile n < 20:\n    n = n * 2\nprint(n)\n```"),
        draw.program_trace(C.PICTURE_LINES["nl"], C.PICTURE_SHOWN, code_label="Voorwaarde vóór de ronde", names=C.PICTURE_NAMES,
                           names_label="n daarna", output_label="Onder de cel", title="De loop hierboven, voorwaarde voor voorwaarde"),
        mo.md(r"""
        Elke ronde verdubbelt `n`, tot de controle vóór de volgende ronde `False` geeft. Pas dan gaat het programma verder met `print(n)`, en dat laat 32 zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een `while`-loop stopt pas als de voorwaarde `False` wordt. Verandert niets in de loop waar de voorwaarde naar kijkt, dan gebeurt dat nooit. En `break` draait alleen als Python bij die regel komt: zet hem in een `if`, dan stopt de loop op het goede moment."),
        mo.md(r"""
        ### Recept

        Zo herhaal je tot er iets gebeurt:

        1. Geef vóór de loop de namen in de voorwaarde hun beginwaarde.
        2. Schrijf `while`, de voorwaarde om door te gaan en een dubbele punt: `while savings < 100:`.
        3. Schrijf eronder, ingesprongen, één ronde. Een van de regels moet veranderen waar de voorwaarde naar kijkt.
        4. Wil je eerder stoppen, schrijf dan in de loop `break` onder een `if`.
        5. Gebruik na de loop, helemaal links, de uitkomst: die regel draait één keer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je spaart € 15 per week. Na hoeveel weken heb je minstens € 100? En welke temperatuur in een lijst is de eerste onder nul? Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_saving():
    # 1. vóór de loop: de namen in de voorwaarde krijgen een beginwaarde
    _savings = 0
    _weeks = 0
    # 2. while, de voorwaarde om door te gaan, en een dubbele punt
    while _savings < 100:
        # 3. één ronde: dit verandert waar de voorwaarde naar kijkt
        _savings = _savings + 15
        _weeks = _weeks + 1
    # 5. na de loop: de uitkomst
    print(f"Na {_weeks} weken: {_savings} gespaard")

    _temps = [4, 2, -1, 3, -5]
    _first_frost = "none"
    for _t in _temps:
        if _t < 0:
            _first_frost = _t
            # 4. break: stop de loop meteen, onder een if
            break
    print(f"Eerste vorst: {_first_frost}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `Na 7 weken: 105 gespaard` en `Eerste vorst: -1`. Na 6 weken is er 90, en `90 < 100` is nog `True`, dus maakt nog één ronde er 105 van. In de tweede loop stopt `break` de `for`-loop bij -1, dus naar 3 en -5 wordt nooit gekeken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_double():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Verdubbelen

        Het aantal bacteriën verdubbelt elk uur en begint bij 1. Na hoeveel uur zijn het er minstens `target`, dat is 1000? Vervang de twee `...`: de voorwaarde na `while`, en één ronde, waarin `count` verdubbelt en `hours` 1 omhooggaat. De check probeert ook andere doelen.
        """),
        L.task("Zet in `hours` na hoeveel uur `count` minstens `target` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_double():
    box_double = L.code_box("double", lines=7)
    box_double
    return (box_double,)


@app.cell(hide_code=True)
def check_double(box_double):
    result_double = L.check("double", box_double.value)
    result_double
    return (result_double,)


@app.cell(hide_code=True)
def hints_double():
    L.hints(
        "double",
        [
            "Stap 2 van het recept: de loop moet doorgaan zolang `count` nog onder `target` is. Welke vergelijking is dat?",
            "In één ronde veranderen twee namen: `count` verdubbelt, en `hours` gaat 1 omhoog. Beide regels springen in.",
            "`while count ___ target:`, dan ingesprongen de regel die `count` verdubbelt, en `hours = hours + ___`.",
        ],
        why="`while count < target:` gaat door zolang het er minder dan 1000 zijn. Elke ronde verdubbelt `count` en telt 1 op bij `hours`. Na 10 rondes is `count` 1024, geeft de controle `False`, en laat `print(hours)` 10 zien.",
    )
    return


@app.cell(hide_code=True)
def text_break():
    mo.md(r"""
    ### Oefening 2. Waar break stopt
    """)
    return


@app.cell(hide_code=True)
def choice_break_cell():
    choice_break = L.choice(
        "break",
        question="Wat laat dit programma zien?\n\n```python\nfor n in [3, 8, 2, 9]:\n    if n > 5:\n        break\n    print(n)\n```",
        options={
            "three-two": "`3` en `2`, elk op een eigen regel",
            "eight": "`8`",
            "three": "`3`",
        },
        correct="three",
        why={
            "three-two": "`break` slaat niet één item over: hij stopt de hele loop. Kijk nog eens naar het woord *break* in de tabel.",
            "eight": "`break` staat vóór `print`, dus bij 8 stopt de loop voordat `print` 8 kan laten zien. Kijk nog eens naar de volgorde van de regels.",
            "three": "3 is niet groter dan 5, dus die verschijnt. Bij 8 stopt `break` de hele loop, vóór de `print`.",
        },
    )
    choice_break
    return (choice_break,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het eerste lange woord

        Zoek het eerste woord in `words` met meer dan 5 letters, en zet het in `first_long`. Vervang de `...` door een `if` met daarin de regel voor `first_long` en een `break`. De check probeert ook andere lijsten.
        """),
        L.task("Zet het eerste woord met meer dan 5 letters in `first_long`, en stop de loop daar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_long():
    box_long = L.code_box("first-long", lines=8)
    box_long
    return (box_long,)


@app.cell(hide_code=True)
def check_long(box_long):
    result_long = L.check("first-long", box_long.value)
    result_long
    return (result_long,)


@app.cell(hide_code=True)
def hints_long():
    L.hints(
        "first-long",
        [
            "Kijk naar de tweede loop van het uitgewerkte voorbeeld: vóór de loop een naam voor de uitkomst, een `if`, en daaronder de regel voor de uitkomst en een `break`.",
            "`len(word)` geeft het aantal tekens, hier letters. Welke vergelijking is `True` bij een woord met meer dan 5 letters?",
            "`if len(word) ___ 5:`, met daaronder, ingesprongen, de regel die `first_long` het woord geeft, en daarna ___.",
        ],
        why="`if len(word) > 5:` is `False` bij `sun` en `rain`, en `True` bij `thunder`. Dan krijgt `first_long` `thunder`, en `break` stopt de loop, dus `drizzle` komt er nooit voor in de plaats.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Herhalen: for-loops](/course/open/py-for?lang=nl) ging een loop een hele lijst langs; `while` gaat door tot de voorwaarde `False` is, en `break` stopt elke loop eerder. Later in de course herhaalt een programma dat leert zijn oefening tot de fouten klein genoeg zijn, of een vast aantal rondes. In de volgende les zet je code onder een eigen naam, als functie.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_break, result_double, result_long):
    L.wrap_up(result_double, choice_break, result_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
