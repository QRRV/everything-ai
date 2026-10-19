# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-indexing. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-indexing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Wat is de shape van `np.array([[1, 2, 3, 4], [5, 6, 7, 8]])`?",
        options={
            "flat": "`(8,)`",
            "tall": "`(4, 2)`",
            "wide": "`(2, 4)`",
        },
        correct="wide",
        why={
            "flat": "Een lijst van lijsten geeft een matrix, met twee getallen in zijn shape. Zie de figuur in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "tall": "In een shape komen de rijen eerst. Hoeveel binnenste lijsten zijn er? Zie de tabel met nieuwe woorden in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "wide": "2 binnenste lijsten, dus 2 rijen, elk van 4 getallen: 4 kolommen.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_dtype():
    quick_dtype = L.choice(
        "quick-dtype",
        question="Wat is het dtype van `np.array([1, 2.5])`?",
        options={
            "float": "`float64`",
            "int": "`int64`",
            "both": "`int64` voor de 1, en `float64` voor de 2.5",
        },
        correct="float",
        why={
            "float": "Alle getallen van een array delen één type. Eén float is genoeg: de 1 wordt `1.0`.",
            "int": "`2.5` is geen geheel getal. Wat gebeurt er met het type van de andere getallen? Zie de gok in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "both": "Een array heeft één type voor al zijn getallen. Zie de tabel met nieuwe woorden in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
        },
    )
    quick_dtype
    return (quick_dtype,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Data gebruik je bijna nooit in zijn geheel. Je wilt één voorbeeld uit een tabel, dus één rij; of één feature van elk voorbeeld, dus één kolom; of alleen de voorbeelden die aan een voorwaarde voldoen, zoals elke foto van een 7. Later in de course doe je dit steeds: om je data te splitsen in een deel om van te leren en een deel om mee te testen, of om te zien welke voorspellingen fout waren. Met NumPy kost elk daarvan één korte regel, zonder loop.

    **Een voorbeeld uit het dagelijks leven.** Een spreadsheet met een filter. Je klikt op een kolom om al zijn waarden te zien, of je filtert de rijen, en alleen die waar je om vroeg blijven in beeld.
    """)
    return


@app.cell(hide_code=True)
def predict_pick():
    guess_pick = L.guess(
        "pick",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Wat denk je dat `m[1, 2]` geeft?",
        options={
            "two": "`2`",
            "six": "`6`",
            "five": "`5`",
        },
        reveal="**`6`.** Het eerste getal kiest de rij, het tweede de kolom, en allebei beginnen ze bij 0. Rij 1 is `[4, 5, 6]`, en daarin is kolom 2 `6`.",
    )
    guess_pick
    return (guess_pick,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Vierkante haken na een array kiezen een deel ervan, net als bij een lijst. Bij een matrix geef je een rij en een kolom, met een komma ertussen. In plaats van een getal kun je ook een slice geven, of een mask: een array van `True` en `False` dat de getallen kiest waar het `True` is.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **`m[rij, kolom]`** | één getal: eerst de rij, dan de kolom, allebei geteld vanaf 0 | `m[1, 2]` geeft `6` |
    | **`m[rij]`** | één hele rij, als vector | `m[0]` geeft `array([1, 2, 3])` |
    | **`:`** | de hele richting: alle rijen, of alle kolommen | `m[:, 1]` geeft de kolom `array([2, 5])` |
    | **slice** | een deel van een richting, net als bij een lijst | `m[0, 1:3]` geeft `array([2, 3])` |
    | **vergelijking op een array** | vergelijkt elk getal apart, en geeft een array van `True` en `False` met dezelfde shape | `m > 2` geeft `array([[False, False, True], [True, True, True]])` |
    | **boolean mask** | een array van `True` en `False` tussen vierkante haken: het kiest de getallen waar het `True` is, rij voor rij, als vector | `m[m > 2]` geeft `array([3, 4, 5, 6])` |

    *Boolean* is de volledige naam van een bool: `True` of `False`. Een mask kiest getallen uit de hele array, dus heeft de uitkomst geen rijen en kolommen meer: het is een vector. En één getal uit een array verschijnt op de laatste regel van een cel als `np.int64(6)`: NumPy's eigen soort geheel getal. `print` laat gewoon `6` zien.
    """)
    return


@app.cell(hide_code=True)
def picture_select():
    mo.vstack([
        draw.array_grids(C.picture(["één getal", "een kolom", "met een mask"], "geeft"), indices=True,
                         title="Drie manieren om uit m te kiezen"),
        mo.md(r"""
        De kleine getallen boven en links zijn de indexen: kolommen boven, rijen links. De vakjes met een dikke rand zijn wat elke keuze pakt; de rest wordt grijs. `m[:, 1]` pakt kolom 1 uit elke rij; `m[m > 2]` pakt elk getal boven 2.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("De rij komt eerst, dan de kolom, en allebei beginnen ze bij 0. Voor een kolom heb je de `:` voor alle rijen nodig: `m[:, 1]`. `m[1]` is rij 1."),
        mo.md(r"""
        ### Recept

        Zo kies je een deel van een array:

        1. Eén getal: `m[rij, kolom]`.
        2. Een rij: `m[rij]`; een kolom: `m[:, kolom]`.
        3. Een deel van een rij of een kolom: een slice, net als bij een lijst, zoals `m[0, 1:3]`.
        4. De getallen die aan een voorwaarde voldoen: maak een mask met een vergelijking, zoals `m > 2`, en zet dat tussen de vierkante haken: `m[m > 2]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De regen uit [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl): een rij per week, een kolom per dag, van maandag tot en met donderdag. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_rain():
    import numpy as np

    _rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])
    # 1. één getal: week 2 is rij 1, dinsdag is kolom 1
    print(_rain[1, 1])
    # 2. een rij en een kolom: heel week 1, en elke maandag
    print(_rain[0])
    print(_rain[:, 0])
    # 3. een deel: week 1, van dinsdag tot en met woensdag
    print(_rain[0, 1:3])
    # 4. de hoeveelheden boven 2: eerst het mask, dan de hoeveelheden die het kiest
    _wet = _rain > 2
    print(_wet)
    print(_rain[_wet])
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De cel print `0`: op de dinsdag van week 2 regende het niet. Dan week 1, `[0 3 5 1]`, en de maandagen, `[0 2 6]`: een kolom komt eruit als vector, op één regel. `_rain[0, 1:3]` is `[3 5]`: dinsdag en woensdag van week 1. Het mask heeft de shape van `_rain`, met `True` waar het meer dan 2 mm regende. En `_rain[_wet]` geeft die hoeveelheden, rij voor rij: `[3 5 4 6]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_column():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke regel pakt dit deel?
    """)
    return


@app.cell(hide_code=True)
def choice_part_cell():
    choice_part = L.choice(
        "which-part",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Welke regel geeft `array([5, 6])`?",
        options={
            "rows": "`m[1:3]`",
            "part": "`m[1, 1:3]`",
            "short": "`m[1, 1:2]`",
        },
        correct="part",
        why={
            "rows": "Met één slice tussen de haken slice je de rijen: rij 1 en 2, en daarvan heeft `m` alleen rij 1. Dus krijg je heel rij 1, als matrix. Kijk nog eens naar stap 3 van het recept.",
            "part": "Rij 1 is `[4, 5, 6]`, en de slice `1:3` pakt daarvan kolom 1 en 2: `5` en `6`.",
            "short": "Een slice stopt net voor zijn tweede getal, dus `1:2` pakt alleen kolom 1: `array([5])`. Waar moet hij stoppen?",
        },
    )
    choice_part
    return (choice_part,)


@app.cell(hide_code=True)
def text_exercise_wednesdays():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Elke woensdag

        In het vak staat de regen weer, `rain`, met shape `(3, 4)`: een rij per week, een kolom per dag, van maandag tot en met donderdag. Haal de regen van elke woensdag eruit en noem die `wednesdays`. Vervang de `...`. De laatste regel is `wednesdays`, zodat de check het ziet.
        """),
        L.task("Maak `wednesdays` de kolom met woensdagen, uit `rain` gehaald."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wednesdays():
    box_wednesdays = L.code_box("wednesdays", lines=6)
    box_wednesdays
    return (box_wednesdays,)


@app.cell(hide_code=True)
def check_wednesdays(box_wednesdays):
    result_wednesdays = L.check("wednesdays", box_wednesdays.value)
    result_wednesdays
    return (result_wednesdays,)


@app.cell(hide_code=True)
def hints_wednesdays():
    L.hints(
        "wednesdays",
        [
            "Stap 2 van het recept: een kolom.",
            "Een kolom heeft alle rijen nodig, dus komt `:` eerst; na de komma komt het nummer van de kolom. Maandag is kolom 0.",
            "`wednesdays = rain[:, ___]`.",
        ],
        why="`:` pakt alle rijen, de drie weken, en `2` pakt uit elke week kolom 2, woensdag: `array([5, 0, 0])`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wet():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De natte dagen

        Maak `wet_days(rain, limit)` af: die geeft elke hoeveelheid in `rain` terug die meer is dan `limit`, als array, rij voor rij. Hier is `rain` een lijst van lijsten, en de functie maakt er al een array van, `grid`. Vervang de `...`. De check roept `wet_days(rain, 2)` aan met de regen van oefening 2, als lijst van lijsten: dat moet `array([3, 5, 4, 6])` geven.
        """),
        L.task("Laat `wet_days(rain, limit)` de hoeveelheden boven `limit` teruggeven, met een boolean mask."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wet():
    box_wet = L.code_box("wet-days", lines=8)
    box_wet
    return (box_wet,)


@app.cell(hide_code=True)
def check_wet(box_wet):
    result_wet = L.check("wet-days", box_wet.value)
    result_wet
    return (result_wet,)


@app.cell(hide_code=True)
def hints_wet():
    L.hints(
        "wet-days",
        [
            "Stap 4 van het recept, zoals aan het eind van het uitgewerkte voorbeeld.",
            "Eerst het mask: een vergelijking van `grid` met `limit`. Zet het mask dan tussen de vierkante haken van `grid`, en geef dat terug.",
            "`wet = grid ___ limit`, en dan `return grid[___]`.",
        ],
        why="`grid > limit` vergelijkt elke hoeveelheid met `limit`, en geeft `True` waar die meer is. `grid[wet]` kiest die hoeveelheden, rij voor rij, als vector.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) haalde je een item uit een lijst met zijn index, en een deel met een slice; een array werkt net zo, nu met een rij en een kolom. In [Keuzes maken: if en else](/course/open/py-if?lang=nl) gaf een vergelijking één `True` of `False`; op een array geeft hij er één voor elk getal, en dat is een mask. In de volgende les, *Rekenen zonder loops*, reken je met hele arrays tegelijk, zoals een vergelijking al op elk getal werkt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_part, result_wednesdays, result_wet):
    L.wrap_up(choice_part, result_wednesdays, result_wet)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
