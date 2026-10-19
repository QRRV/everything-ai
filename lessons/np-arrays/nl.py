# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-arrays. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-arrays", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Lijsten](/course/open/py-lists?lang=nl), [Code uit modules gebruiken](/course/open/py-imports?lang=nl) en [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="`days = [3, 0, 5, 2]`. Wat geeft `days[1:3]`?",
        options={
            "three": "`[3, 0, 5]`",
            "two": "`[0, 5]`",
            "rest": "`[0, 5, 2]`",
        },
        correct="two",
        why={
            "three": "Index 0 is het eerste item, en een slice begint bij zijn eerste getal. Zie de figuur in [Lijsten](/course/open/py-lists?lang=nl).",
            "two": "De slice pakt index 1 en 2: `0` en `5`. Bij index 3 stopt hij, en dat item hoort er niet bij.",
            "rest": "Een slice stopt bij zijn tweede getal, zonder dat item. Zie de figuur in [Lijsten](/course/open/py-lists?lang=nl).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="Met welke regel kun je `math.sqrt(16)` schrijven in je programma?",
        options={
            "import": "`import math`",
            "from": "`from math import sqrt`",
            "name": "`math = sqrt`",
        },
        correct="import",
        why={
            "import": "`import math` haalt de module binnen; daarna haal je `sqrt` eruit met de naam van de module en een punt.",
            "from": "Die haalt alleen `sqrt` binnen, dus schrijf je `sqrt(16)`, zonder `math.`. Zie de tabel met nieuwe woorden in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "name": "Die geeft de naam `math` een waarde; hij importeert niets. Zie het recept in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def quick_check_tuple():
    quick_tuple = L.choice(
        "quick-tuple",
        question="`point = (3, 4)`, en daarna `x, y = point`. Wat is `y`?",
        options={
            "first": "`3`",
            "both": "`(3, 4)`",
            "second": "`4`",
        },
        correct="second",
        why={
            "first": "Dat is `x`: de namen krijgen de waarden op volgorde. Zie het idee in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "both": "Links staan twee namen, dus krijgt elk één waarde van de tuple. Zie het idee in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "second": "De namen krijgen de waarden op volgorde: `x` de eerste, `3`, en `y` de tweede, `4`.",
        },
    )
    quick_tuple
    return (quick_tuple,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    AI rekent met roosters van getallen: een zwart-witfoto is een rooster met één getal per pixel, een tabel met voorbeelden heeft een rij per voorbeeld, en een neural network bewaart wat het leert in zulke roosters. Een lijst in Python is traag met miljoenen getallen, en staat alles toe, zoals tekst tussen de getallen of rijen van verschillende lengte. NumPy bewaart getallen in een array: een rooster van getallen van één type, met een vast aantal rijen en kolommen.

    **Een voorbeeld uit het dagelijks leven.** Een spreadsheet met je cijfers: een rij per vak, een kolom per toets, en in elke cel een getal.
    """)
    return


@app.cell(hide_code=True)
def predict_dtype():
    guess_dtype = L.guess(
        "dtype",
        question="Wat denk je dat er in `np.array([2, 4, 6.5])` zit?",
        options={
            "ints": "Drie gehele getallen: `2`, `4` en `6`",
            "mixed": "Twee gehele getallen en één float, zoals in een lijst",
            "floats": "Drie floats: `2.0`, `4.0` en `6.5`",
        },
        reveal="**Drie floats.** Alle getallen in een array hebben hetzelfde type. Zodra één getal een float is, maakt NumPy er allemaal floats van: `array([2. , 4. , 6.5])`. Dat type is het dtype van de array, hier `float64`.",
    )
    guess_dtype
    return (guess_dtype,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    NumPy is een module om met veel getallen tegelijk te rekenen. Een array van NumPy is een rooster van getallen, allemaal van één type. Je maakt een array van een lijst: een lijst getallen geeft een vector, en een lijst van lijsten geeft een matrix, met één binnenste lijst per rij.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **NumPy** | een module om met veel getallen tegelijk te rekenen; hij is met de course meegeïnstalleerd | `import numpy` |
    | **`as`** | importeert een module onder een korte naam van jezelf; NumPy importeer je altijd als `np` | `import numpy as np` |
    | **array** | het rooster van getallen van NumPy, allemaal van hetzelfde type | `np.array([7, 9, 4])` |
    | **lijst van lijsten** | een lijst waarvan de items lijsten zijn; bij een rooster is elke binnenste lijst één rij | `[[1, 2, 3], [4, 5, 6]]` |
    | **vector** | een array in één richting: getallen op een rij | `np.array([7, 9, 4])` |
    | **matrix** | een array in twee richtingen: rijen en kolommen | `np.array([[1, 2, 3], [4, 5, 6]])` |
    | **shape** | hoeveel getallen de array in elke richting heeft, als tuple; bij een matrix eerst de rijen, dan de kolommen | `m.shape` geeft `(2, 3)` |
    | **dtype** | het ene type dat alle getallen van de array delen: `int64` voor gehele getallen, `float64` voor floats | `print(m.dtype)` laat `int64` zien |

    `.shape` en `.dtype` hebben geen haakjes: het is niet iets wat de array doet, maar iets wat hij heeft. De 64 in `int64` en `float64` zegt hoeveel geheugen elk getal kost; dat heb je nu nog niet nodig. `print` laat een array zien zonder komma's, `[7 9 4]`; de laatste regel van een cel laat hem zien met komma's, `array([7, 9, 4])`.
    """)
    return


@app.cell(hide_code=True)
def picture_arrays():
    mo.vstack([
        draw.array_grids([("een vector", [[str(_x) for _x in C.VECTOR]], [f"shape ({len(C.VECTOR)},)", "3 getallen op een rij"]),
                          ("een matrix", [[str(_x) for _x in _row] for _row in C.MATRIX],
                           [f"shape ({len(C.MATRIX)}, {len(C.MATRIX[0])})", "2 rijen, 3 kolommen"])],
                         title="Een vector en een matrix"),
        mo.md(r"""
        Een vector gaat in één richting, dus heeft zijn shape één getal: `(3,)`. De komma maakt er een tuple met één waarde van, en niet zomaar een getal tussen haakjes. Een matrix gaat in twee richtingen, dus heeft zijn shape twee getallen: eerst de rijen, dan de kolommen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Elke rij heeft hetzelfde aantal waarden nodig. Met rijen van verschillende lengte stopt NumPy met een fout: een array is altijd een volledig rooster."),
        mo.md(r"""
        ### Recept

        Zo bewaar je getallen in een array:

        1. Importeer NumPy één keer, bovenaan: `import numpy as np`.
        2. Schrijf de getallen als lijst; voor een rooster als lijst van lijsten, met één binnenste lijst per rij.
        3. Geef die lijst aan `np.array(...)`, en geef de array een naam.
        4. Controleer zijn `.shape`, eerst de rijen, dan de kolommen, en zijn `.dtype`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De temperatuur op twee dagen, 's ochtends, 's middags en 's avonds. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_temps():
    # 1. importeer NumPy, één keer, bovenaan
    import numpy as np

    # 2. de getallen als lijst van lijsten: een rij per dag, een kolom per dagdeel
    _rows = [[12.5, 17.0, 14.0], [11.0, 15.5, 13.0]]
    # 3. geef de lijst aan np.array
    _temps = np.array(_rows)
    # 4. controleer de shape en het dtype
    print(_temps.shape)
    print(_temps.dtype)
    # 2. tot en met 4. nog een keer voor een vector: één lijst, van gehele getallen
    _noon = np.array([17, 15])
    print(_noon.shape)
    print(_noon.dtype)
    _temps
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de code laat de cel de matrix zelf zien, met zijn rijen onder elkaar; NumPy schrijft `17.` voor `17.0`. Daaronder staat wat de `print`-regels laten zien: `(2, 3)`, voor 2 rijen, één per dag, en 3 kolommen, één per dagdeel; en `float64`, omdat alle getallen floats zijn. Dan de vector: `(2,)`, twee getallen in één richting, en `int64`, omdat het allebei gehele getallen zijn.

    Deze cel maakt de naam `np`, dus mogen andere cellen van deze les hem gebruiken, maar niet opnieuw maken: *één naam, één cel*. Wil je iets uitproberen, gebruik dan het kladblok, met een eigen `import numpy as np`. De oefenvakken draaien los, dus heeft elk vak zijn eigen `import`-regel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Lees de shape af
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape-of",
        question="Wat is de shape van `np.array([[1, 2], [3, 4], [5, 6]])`?",
        options={
            "wide": "`(2, 3)`",
            "tall": "`(3, 2)`",
            "flat": "`(6,)`",
        },
        correct="tall",
        why={
            "wide": "Dat zouden 2 rijen van 3 getallen zijn. Tel de binnenste lijsten: elk ervan is een rij.",
            "tall": "3 binnenste lijsten, dus 3 rijen, elk van 2 getallen: 2 kolommen. De rijen komen eerst.",
            "flat": "Dat is de shape van een vector van 6 getallen. Een lijst van lijsten geeft een matrix, met twee getallen in zijn shape.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_rain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een tabel als array

        De regen in millimeters, in drie weken, van maandag tot en met donderdag:

        | | ma | di | wo | do |
        |---|---|---|---|---|
        | week 1 | 0 | 3 | 5 | 1 |
        | week 2 | 2 | 0 | 0 | 4 |
        | week 3 | 6 | 1 | 0 | 0 |

        Maak de array `rain` van deze tabel, met een rij per week. Vervang de `...`. De laatste regel is `rain`, zodat de check je array ziet.
        """),
        L.task("Maak de array `rain`: een rij per week, een kolom per dag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rain():
    box_rain = L.code_box("rain", lines=5)
    box_rain
    return (box_rain,)


@app.cell(hide_code=True)
def check_rain(box_rain):
    result_rain = L.check("rain", box_rain.value)
    result_rain
    return (result_rain,)


@app.cell(hide_code=True)
def hints_rain():
    L.hints(
        "rain",
        [
            "Stap 2 en 3 van het recept: eerst de tabel als lijst van lijsten, dan `np.array`.",
            "Elke week is één binnenste lijst van 4 getallen, van maandag tot en met donderdag; de drie weken komen in één grote lijst.",
            "`rain = np.array([[0, 3, 5, 1], ___, ___])`.",
        ],
        why="De grote lijst heeft drie rijen, één per week. `np.array` maakt er een matrix van, met shape `(3, 4)`: 3 rijen, 4 kolommen. Alle getallen zijn gehele getallen, dus is zijn dtype `int64`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_describe():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Beschrijf een array

        Maak `describe(table)` af. `table` is een lijst van rijen; de functie maakt daar al een array van, `grid`. Hij moet één tekst teruggeven met de rijen, de kolommen en het dtype van `grid`, zoals `"2 by 3, int64"`. Vervang de `...`. De check roept `describe([[1, 2, 3], [4, 5, 6]])` aan: dat moet `"2 by 3, int64"` geven.
        """),
        L.task("Laat `describe(table)` de rijen, de kolommen en het dtype van de array teruggeven, als één tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=8)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "`grid.shape` is een tuple van twee getallen: de rijen, dan de kolommen. En `grid.dtype` is het type van de getallen.",
            "Geef de twee getallen van de shape elk een naam, zoals bij `x, y = point`; maak dan de tekst met een f-string.",
            "`rows, cols = grid.shape`, en dan `return f\"{rows} by {cols}, ___\"`.",
        ],
        why="`rows, cols = grid.shape` geeft elk getal van de shape een eigen naam. In een f-string wordt `{grid.dtype}` dan `int64` of `float64`. Met één float in de tabel is de hele array `float64`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) bewaarde je getallen in een lijst; nu wordt een lijst een array, en een lijst van lijsten een matrix. In [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl) hield een tuple waarden op volgorde; de shape van een array is zo'n tuple. In [Code uit modules gebruiken](/course/open/py-imports?lang=nl) importeerde je `math` en `random`; NumPy importeer je op dezelfde manier, onder de korte naam `np`. In de volgende les, *Delen van een array kiezen*, haal je rijen, kolommen en losse getallen uit een array.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_describe, result_rain):
    L.wrap_up(choice_shape, result_rain, result_describe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
