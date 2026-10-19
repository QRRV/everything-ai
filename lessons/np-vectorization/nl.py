# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-vectorization. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-vectorization", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl) en [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_make():
    quick_make = L.choice(
        "quick-make",
        question="Welke regel maakt een array van de getallen 1, 2 en 3?",
        options={
            "list": "`[1, 2, 3]`",
            "array": "`np.array([1, 2, 3])`",
            "loose": "`np.array(1, 2, 3)`",
        },
        correct="array",
        why={
            "list": "Dat is een lijst. Wat maakt van een lijst een array? Zie het recept in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "array": "`np.array` krijgt één lijst, en maakt er een array van.",
            "loose": "`np.array` krijgt de getallen als één lijst, tussen vierkante haken. Zie het recept in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
        },
    )
    quick_make
    return (quick_make,)


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="Wat laat dit programma zien?\n\n```python\ntotal = 0\nfor x in [1, 2, 3]:\n    total = total + x * 2\nprint(total)\n```",
        options={
            "list": "`[2, 4, 6]`",
            "six": "`6`",
            "twelve": "`12`",
        },
        correct="twelve",
        why={
            "list": "`total` is een getal dat elke ronde groeit; hier maakt niets een lijst. Zie het uitgewerkte voorbeeld in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "six": "Dat is 1 + 2 + 3. Maar elke ronde telt `x * 2` op, niet `x`. Zie het recept in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "twelve": "De loop telt 2, 4 en 6 op bij `total`, één ronde per keer: 12.",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model rekent met miljoenen getallen, steeds opnieuw. Een loop in Python doet dat getal voor getal, en dat is traag: voor elk getal zoekt Python opnieuw uit wat `*` voor dat getal betekent. NumPy doet de hele array in één keer, in eigen, snelle code. Daarom rekent AI altijd met arrays: met NumPy is hetzelfde werk vaak vele malen sneller dan met een loop. En de code wordt ook korter: één regel in plaats van een loop.

    **Een voorbeeld uit het dagelijks leven.** Een winkel die al zijn prijzen 10% verhoogt. Met één wijziging in de kassa is elke prijs nieuw, in plaats van een nieuwe sticker op elk product, één voor één.
    """)
    return


@app.cell(hide_code=True)
def predict_times():
    guess_times = L.guess(
        "times",
        question="Wat denk je dat `np.array([1, 2, 3]) * 2` geeft?",
        options={
            "repeat": "`[1, 2, 3, 1, 2, 3]`",
            "double": "`array([2, 4, 6])`",
            "error": "Een fout: een array kun je niet vermenigvuldigen",
        },
        reveal="**`array([2, 4, 6])`.** Op een array vermenigvuldigt `* 2` elk getal. Op een lijst geeft `[1, 2, 3] * 2` juist `[1, 2, 3, 1, 2, 3]`: de lijst twee keer. Daarom maak je eerst een array.",
    )
    guess_times
    return (guess_times,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Je schrijft een berekening één keer, alsof hij voor één getal is, en NumPy doet hem voor elk getal van de array. Dat heet vectorisatie: rekenen met hele arrays in plaats van met een loop.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **vectorisatie** | rekenen met hele arrays tegelijk, zonder eigen loop | `prices * 1.1` |
    | **array en getal** | `+`, `-`, `*`, `/` en `**` met één getal werken op elk getal van de array | `np.array([1, 2, 3]) * 2` geeft `array([2, 4, 6])` |
    | **twee arrays** | bij twee arrays met dezelfde shape gaan de getallen op dezelfde plek samen | `np.array([1, 2]) + np.array([10, 20])` geeft `array([11, 22])` |
    | **`np.arange`** | de getallen 0, 1, 2 enzovoort, als array, zoals `range` telt | `np.arange(4)` geeft `array([0, 1, 2, 3])` |

    De uitkomst is altijd een nieuwe array, met dezelfde shape; de array waarmee je begon, blijft zoals hij was.
    """)
    return


@app.cell(hide_code=True)
def picture_together():
    mo.vstack([
        draw.value_rows(C.picture_rows({"plus": "de getallen op dezelfde plek, opgeteld", "times": "elk getal keer 2"}),
                        title="Rekenen met hele arrays"),
        mo.md(r"""
        In `a + b` gaat het eerste getal van `a` samen met het eerste getal van `b`, het tweede met het tweede, enzovoort. In `a * 2` gaat de 2 samen met elk getal van `a`. Voor geen van beide is een loop nodig.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`*` en `+` doen op een lijst iets anders: `[1, 2] * 2` herhaalt de lijst, en `[1, 2] + [3]` plakt lijsten aan elkaar. Maak eerst een array. En gebruik voorlopig twee arrays met dezelfde shape: een latere les, *Broadcasting*, laat zien wanneer NumPy andere shapes toestaat."),
        mo.md(r"""
        ### Recept

        Zo reken je zonder loop:

        1. Maak een array van je getallen: `np.array(...)`.
        2. Schrijf de berekening één keer, alsof hij voor één getal is: `prices * 1.1`.
        3. NumPy doet hem voor elk getal, en geeft een nieuwe array met dezelfde shape.
        4. Bij twee arrays: controleer dat ze dezelfde shape hebben; de getallen op dezelfde plek gaan samen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Temperaturen van graden Celsius naar graden Fahrenheit: keer 9, gedeeld door 5, plus 32. Eén keer met NumPy, en één keer met een loop, om te vergelijken. Daarna twee arrays: de prijs van elk product keer hoeveel je er koopt. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_fahrenheit():
    import numpy as np

    # 1. de getallen als array
    _celsius = np.array([12.5, 17.0, 14.0, 11.0])
    # 2. en 3. de berekening één keer, gedaan voor elk getal
    _fahrenheit = _celsius * 9 / 5 + 32
    print(_fahrenheit)
    # hetzelfde met een loop, getal voor getal
    _by_loop = []
    for _c in [12.5, 17.0, 14.0, 11.0]:
        _by_loop.append(_c * 9 / 5 + 32)
    print(_by_loop)
    # 4. twee arrays met dezelfde shape: de getallen op dezelfde plek gaan samen
    _prices = np.array([2.5, 1.0, 4.0])
    _counts = np.array([3, 10, 2])
    print(_prices.shape, _counts.shape)
    print(_prices * _counts)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste twee regels laten dezelfde temperaturen zien, `54.5`, `62.6`, `57.2` en `51.8`: één keer als array, uit één regel code, en één keer als lijst, uit een loop van drie regels. Dan de twee shapes, `(3,) (3,)`: hetzelfde, dus kunnen de getallen samen. De laatste regel is `[ 7.5 10.   8. ]`: 2,5 keer 3, 1 keer 10, en 4 keer 2.

    | Met een loop | Met NumPy |
    |---|---|
    | `for c in celsius:` en `out.append(c * 9 / 5 + 32)` | `celsius * 9 / 5 + 32` |
    | getal voor getal, en jij maakt de lijst | elk getal tegelijk, en je krijgt een nieuwe array |
    """)
    return


@app.cell(hide_code=True)
def demo_speed():
    _loop, _whole = C.measure()
    _ms_loop, _ms_whole = C.decimal(_loop * 1000, 0, L.lang), C.decimal(_whole * 1000, 1, L.lang)
    _times = C.decimal(_loop / max(_whole, 1e-9), 0, L.lang)
    mo.md(f"""
    **Hoeveel sneller?** Bij het openen van deze les kwadrateerde je computer de getallen 0 tot en met 999.999 op twee manieren. Met een loop, `for i in range(1000000):` en `squares.append(i * i)`, duurde dat {_ms_loop} milliseconden. Met NumPy, `np.arange(1000000) ** 2`, duurde het {_ms_whole} milliseconden: ongeveer {_times} keer zo snel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee arrays samen
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    choice_add = L.choice(
        "add-arrays",
        question="`a = np.array([4, 5, 6])` en `b = np.array([10, 20, 30])`. Wat geeft `a + b`?",
        options={
            "pairs": "`array([14, 25, 36])`",
            "joined": "`array([4, 5, 6, 10, 20, 30])`",
            "total": "`75`",
        },
        correct="pairs",
        why={
            "pairs": "De getallen op dezelfde plek gaan samen: 4 + 10, 5 + 20 en 6 + 30.",
            "joined": "Dat doet `+` met twee lijsten. Kijk nog eens naar het blok *Let op* boven het recept.",
            "total": "Dat zou de som van alle zes getallen zijn. Wat doet `+` met elk getal? Kijk nog eens naar de figuur.",
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_exercise_prices():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Alles 10% duurder

        Een winkel verhoogt al zijn prijzen met 10%. `prices` bevat zijn vier prijzen, shape `(4,)`. Maak `new_prices`, met elke prijs 10% hoger, zonder loop. Vervang de `...`. De laatste regel is `new_prices`, zodat de check het ziet.
        """),
        L.task("Maak `new_prices`: elke prijs van `prices` 10% hoger, in één berekening."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prices():
    box_prices = L.code_box("new-prices", lines=6)
    box_prices
    return (box_prices,)


@app.cell(hide_code=True)
def check_prices(box_prices):
    result_prices = L.check("new-prices", box_prices.value)
    result_prices
    return (result_prices,)


@app.cell(hide_code=True)
def hints_prices():
    L.hints(
        "new-prices",
        [
            "Stap 2 en 3 van het recept: schrijf de berekening één keer, met `prices`.",
            "10% meer is de prijs plus een tiende van de prijs. Waarmee vermenigvuldig je een prijs om dat te krijgen?",
            "`new_prices = prices * ___`.",
        ],
        why="Een prijs plus 10% ervan is de prijs keer `1.1`. `prices * 1.1` doet dat voor elke prijs tegelijk: `array([ 2.75,  1.1 ,  4.4 , 13.2 ])`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_speeds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoe snel was elke rit?

        Maak `speeds(km, minutes)` af: die krijgt de afstand van elke rit, in km, en de tijd ervan, in minuten, en geeft de snelheid van elke rit terug, in km per uur, als array. De functie maakt al arrays van allebei; ze hebben dezelfde shape, één getal per rit. Vervang de `...`, zonder loop. De check roept `speeds([10, 3, 42], [30, 15, 120])` aan: dat moet `array([20., 12., 21.])` geven.
        """),
        L.task("Laat `speeds(km, minutes)` de snelheid van elke rit in km per uur teruggeven, zonder loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_speeds():
    box_speeds = L.code_box("speeds", lines=9)
    box_speeds
    return (box_speeds,)


@app.cell(hide_code=True)
def check_speeds(box_speeds):
    result_speeds = L.check("speeds", box_speeds.value)
    result_speeds
    return (result_speeds,)


@app.cell(hide_code=True)
def hints_speeds():
    L.hints(
        "speeds",
        [
            "Een snelheid is de afstand gedeeld door de tijd. Voor km per uur moet de tijd in uren zijn.",
            "Maak eerst uren van de minuten: een uur heeft 60 minuten. Deel dan de afstanden door de uren: de getallen op dezelfde plek gaan samen.",
            "`hours = minutes ___ 60`, en dan `return ___`.",
        ],
        why="`minutes / 60` maakt van elke tijd uren: 30 minuten is 0,5 uur. `km / hours` deelt elke afstand door zijn eigen tijd: 10 km in een half uur is 20 km per uur.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Herhalen: for-loops](/course/open/py-for?lang=nl) ging een loop item voor item door een lijst; met een array doet één regel hetzelfde voor elk getal, en veel sneller. In [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl) zei de shape hoeveel getallen een array heeft; twee arrays met dezelfde shape gaan nu getal voor getal samen. In de volgende les, *Sommen en gemiddelden langs een axis*, ga je de andere kant op: van een hele array terug naar één getal, zoals de som.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, result_prices, result_speeds):
    L.wrap_up(choice_add, result_prices, result_speeds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
