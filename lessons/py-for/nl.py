# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-for. Route: course/route/lessen/stam-1-basis.yaml. Checks en de loop in het beeld: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-for", "nl")
    C = L.checks   # de loop in het beeld, ronde voor ronde


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lijsten](/course/open/py-lists?lang=nl) en [Keuzes maken](/course/open/py-if?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_append():
    quick_append = L.choice(
        "quick-append",
        question="Wat laat dit programma zien?\n\n```python\nnums = [3, 5, 7]\nnums.append(9)\nprint(nums[-1])\n```",
        options={
            "seven": "`7`",
            "nine": "`9`",
            "three": "`3`",
        },
        correct="nine",
        why={
            "seven": "`7` was het laatste item vóór de tweede regel. Wat doet `append` met de lijst? Kijk nog eens naar het recept in [Lijsten](/course/open/py-lists?lang=nl).",
            "nine": "`append` voegt `9` achteraan toe, en `-1` pakt het laatste item.",
            "three": "`-1` telt vanaf het eind, niet vanaf het begin. Kijk nog eens naar het idee in [Lijsten](/course/open/py-lists?lang=nl).",
        },
    )
    quick_append
    return (quick_append,)


@app.cell(hide_code=True)
def quick_check_if():
    quick_if = L.choice(
        "quick-if",
        question="Wat laat dit programma zien?\n\n```python\nx = 4\nif x > 5:\n    print(\"big\")\nelse:\n    print(\"small\")\n```",
        options={
            "both": "`big` en `small`",
            "big": "`big`",
            "small": "`small`",
        },
        correct="small",
        why={
            "both": "Er draait maar één van de twee paden. Kijk nog eens naar het beeld in [Keuzes maken](/course/open/py-if?lang=nl).",
            "big": "`4 > 5` is `False`, dus de regels onder `if` worden overgeslagen. Kijk nog eens in [Keuzes maken](/course/open/py-if?lang=nl).",
            "small": "`4 > 5` is `False`, dus de regel onder `else` draait.",
        },
    )
    quick_if
    return (quick_if,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een lijst kan duizend waarden bevatten. Die regel voor regel optellen zou duizend regels kosten. Een loop voert dezelfde paar regels uit voor elk item, hoe lang de lijst ook is. Later in de course leert een programma van zijn voorbeelden door ze steeds opnieuw langs te gaan, in een loop.

    **Een voorbeeld uit het dagelijks leven.** Kaarten delen bij een spel: elke speler krijgt één kaart. Je bedenkt niet voor elke speler een aparte regel; je doet bij iedereen hetzelfde, tot iedereen een kaart heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_loop():
    guess_loop = L.guess(
        "loop",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nfor color in [\"red\", \"green\", \"blue\"]:\n    print(color)\n```",
        options={
            "one": "Eén regel: `red green blue`",
            "three": "Drie regels: `red`, `green` en `blue`",
            "last": "Alleen `blue`",
        },
        reveal="**Drie regels.** De ingesprongen regel draait één keer voor elk item. De eerste keer is `color` `\"red\"`, daarna `\"green\"`, daarna `\"blue\"`. Elke `print` zet zijn tekst op een nieuwe regel.",
    )
    guess_loop
    return (guess_loop,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    `for` gaat een lijst item voor item langs. Elke ronde krijgt de naam na `for` het volgende item, en draaien de ingesprongen regels.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **loop** | regels code die steeds opnieuw draaien | |
    | **ronde** | één keer door de ingesprongen regels van een loop | |
    | **for** | voert de ingesprongen regels eronder één keer uit voor elk item; elke ronde krijgt de naam na `for` het volgende item | `for temp in temps:` |
    | **range** | de gehele getallen van een start tot een stop, die zelf niet meedoet; een loop kan ze langsgaan | `range(1, 4)` geeft 1, 2 en 3 |

    `range(4)` met één getal begint bij 0: 0, 1, 2 en 3. Wil je in een loop optellen, begin dan vóór de loop met `total = 0`, en geef `total` elke ronde een nieuwe waarde: `total = total + temp`. Python rekent eerst de rechterkant uit, met de waarde die `total` op dat moment heeft.
    """)
    return


@app.cell(hide_code=True)
def picture_rounds():
    mo.vstack([
        mo.md(f"```python\ntotal = 0\nfor temp in {C.PICTURE_ITEMS}:\n    total = total + temp\n```"),
        draw.program_trace(C.PICTURE_LINES["nl"], C.PICTURE_SHOWN, code_label="Stap", names=C.PICTURE_NAMES,
                           names_label="Namen daarna", output_label="Onder de cel", title="De loop hierboven, stap voor stap"),
        mo.md(r"""
        `total` begint bij 0. Elke ronde krijgt `temp` het volgende item, en rekent Python eerst de rechterkant uit, met de oude `total`. Na de laatste ronde draait de regel helemaal links één keer, en die laat 36 zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Alleen de ingesprongen regels draaien elke ronde. Een regel na de loop, helemaal links, draait één keer, na de laatste ronde. En een naam die iets bijhoudt, krijgt zijn beginwaarde vóór de loop, niet erin. In het vak houdt Enter de volgende regel ingesprongen; één niveau terug ga je met één keer Backspace."),
        mo.md(r"""
        ### Recept

        Zo doe je iets met elk item van een lijst:

        1. Geef vóór de loop de namen die iets bijhouden hun beginwaarde, zoals `total = 0`.
        2. Schrijf `for`, een naam voor één item, `in`, de lijst en een dubbele punt: `for temp in temps:`.
        3. Schrijf eronder, ingesprongen, wat er elke ronde moet gebeuren; met een `if` daar draaien de ingesprongen regels eronder alleen bij sommige items.
        4. Gebruik na de loop, helemaal links, de uitkomst: die regel draait één keer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Regen in millimeters, van maandag tot en met vrijdag: hoeveel in totaal, en hoeveel droge dagen? Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_rain():
    _rain = [3, 0, 12, 5, 0]
    # 1. vóór de loop: de namen die iets bijhouden beginnen bij 0
    _total = 0
    _dry_days = 0
    # 2. de loop: één ronde voor elk item van de lijst
    for _mm in _rain:
        # 3. ingesprongen: dit draait elke ronde
        _total = _total + _mm
        # en dit alleen in de rondes waarin de voorwaarde True is
        if _mm == 0:
            _dry_days = _dry_days + 1
    # 4. na de loop, helemaal links: dit draait één keer
    print(f"{_total} mm regen, {_dry_days} droge dagen")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnt `20 mm regen, 2 droge dagen`. In elk van de vijf rondes groeide `_total` met de regen van die dag; in de twee rondes met 0 groeide `_dry_days` met 1. Met `range` gaat een loop langs getallen in plaats van langs een lijst. Voer de cel hieronder uit.
    """)
    return


@app.cell
def example_range():
    # range(1, 4) geeft 1, 2 en 3: hij stopt net voor 4
    for _n in range(1, 4):
        print(f"{_n} keer 7 is {_n * 7}")
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De totaalprijs

        Vervang de `...` in de loop door de regel die de prijs van elke ronde bij `total` optelt. De check probeert ook andere lijsten, en een lege.
        """),
        L.task("Laat de loop alle prijzen optellen in `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total", lines=6)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total",
        [
            "Stap 3 van het recept: wat moet er elke ronde met `total` gebeuren?",
            "Elke ronde krijgt `total` een nieuwe waarde: wat hij al was, plus de prijs van deze ronde. Welke naam heeft die prijs?",
            "`total = total + ___`, ingesprongen onder de regel met `for`.",
        ],
        why="`total = total + price` telt de prijs van elke ronde op bij wat `total` al was. Na vier rondes is dat `2.5 + 4 + 1.5 + 3`, dus `11.0`. `print(total)` blijft helemaal links, dus die draait één keer.",
    )
    return


@app.cell(hide_code=True)
def text_range():
    mo.md(r"""
    ### Oefening 2. Tellen met range
    """)
    return


@app.cell(hide_code=True)
def choice_range_cell():
    choice_range = L.choice(
        "range",
        question="Wat laat dit programma zien?\n\n```python\nfor n in range(2, 5):\n    print(n)\n```",
        options={
            "to-four": "`2`, `3` en `4`, elk op een eigen regel",
            "to-five": "`2`, `3`, `4` en `5`, elk op een eigen regel",
            "from-zero": "`0`, `1`, `2`, `3` en `4`, elk op een eigen regel",
        },
        correct="to-four",
        why={
            "to-five": "`range` stopt net voor zijn stop, zoals een slice. Kijk nog eens naar het woord *range* in de tabel.",
            "to-four": "`range(2, 5)` begint bij 2 en stopt net voor 5: 2, 3 en 4.",
            "from-zero": "Met twee getallen begint `range` bij het eerste. Kijk nog eens naar het woord *range* in de tabel.",
        },
    )
    choice_range
    return (choice_range,)


@app.cell(hide_code=True)
def text_exercise_frost():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Vorstdagen

        Op hoeveel dagen in `temps` is het onder nul? Vervang de `...` door een loop die 1 bij `frost_days` optelt voor elke dag onder nul. De check probeert ook andere temperaturen.
        """),
        L.task("Tel de dagen onder nul in `frost_days`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_frost():
    box_frost = L.code_box("frost", lines=7)
    box_frost
    return (box_frost,)


@app.cell(hide_code=True)
def check_frost(box_frost):
    result_frost = L.check("frost", box_frost.value)
    result_frost
    return (result_frost,)


@app.cell(hide_code=True)
def hints_frost():
    L.hints(
        "frost",
        [
            "Kijk naar het uitgewerkte voorbeeld: daar werden de droge dagen geteld met een `if` in de loop.",
            "Stap 2 van het recept: `for temp in temps:`. Daarin een `if` met een voorwaarde die `True` is bij een temperatuur onder nul.",
            "`for temp in temps:`, dan ingesprongen `if temp < 0:`, en daaronder, nog een stap verder ingesprongen, ___.",
        ],
        why="De loop gaat elke temperatuur langs. `temp < 0` is `True` bij -3 en -1, en in die rondes telt `frost_days = frost_days + 1` er één bij op. 0 is niet onder nul, dus de uitkomst is 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) pakte je items één voor één, met de hand; een loop doet dat voor elk item. Met [Keuzes maken](/course/open/py-if?lang=nl) kiest een `if` in de loop de items die meetellen. Een totaal dat je in een loop opbouwt, komt later terug als het gemiddelde van een lijst getallen. In de volgende les draait een loop zolang een voorwaarde `True` is, met `while`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_range, result_frost, result_total):
    L.wrap_up(result_total, choice_range, result_frost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
