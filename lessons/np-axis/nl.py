# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-axis. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-axis", "nl")
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
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Een matrix heeft een rij per leerling en een kolom per toets: 30 leerlingen, 4 toetsen. Wat is zijn shape?",
        options={
            "tests": "`(4, 30)`",
            "count": "`(120,)`",
            "students": "`(30, 4)`",
        },
        correct="students",
        why={
            "tests": "In een shape komen de rijen eerst. Wat is hier een rij? Zie de tabel met nieuwe woorden in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "count": "Dat is de shape van een vector van 120 getallen; een matrix heeft twee getallen in zijn shape. Zie de figuur in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "students": "30 rijen, één per leerling, en 4 kolommen, één per toets. De rijen komen eerst.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="Wat laat dit programma zien?\n\n```python\nnumbers = [4, 8, 6]\ntotal = 0\nfor x in numbers:\n    total = total + x\nprint(total / len(numbers))\n```",
        options={
            "mean": "`6.0`",
            "total": "`18`",
            "last": "`2.0`",
        },
        correct="mean",
        why={
            "mean": "De loop telt 4, 8 en 6 op: 18. Gedeeld door de 3 getallen is dat `6.0`; `/` geeft altijd een float.",
            "total": "Dat is `total` zelf. Wat gebeurt er op de laatste regel mee? Zie het recept in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "last": "`total` is niet alleen het laatste getal: elke ronde komt er een getal bij. Zie het uitgewerkte voorbeeld in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tabel vol getallen zegt weinig tot je hem samenvat: het gemiddelde cijfer van elke toets, de totale regenval per week. Later in de course doe je dit steeds. Hoe ver een model ernaast zit, is het gemiddelde over alle voorbeelden van hoe ver het bij elk voorbeeld ernaast zit. En een model dat handgeschreven cijfers van 0 tot en met 9 herkent, geeft een rij van tien scores per plaatje: zijn voorspelling voor elk plaatje is het cijfer met de hoogste score in die rij.

    **Een voorbeeld uit het dagelijks leven.** Een cijferlijst. Onder aan elke kolom het gemiddelde van die toets; aan het eind van elke rij het gemiddelde van die leerling.
    """)
    return


@app.cell(hide_code=True)
def predict_axis():
    guess_axis = L.guess(
        "axis",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Wat denk je dat `m.sum(axis=0)` geeft?",
        options={
            "columns": "`array([5, 7, 9])`",
            "rows": "`array([ 6, 15])`",
            "all": "`21`",
        },
        reveal="**`array([5, 7, 9])`.** `axis=0` gaat langs de rijen omlaag: per kolom telt hij de getallen onder elkaar op, 1 + 4, 2 + 5 en 3 + 6. De rijen verdwijnen, en er blijft één getal per kolom over. `axis=0` betekent dus niet \"per rij\".",
    )
    guess_axis
    return (guess_axis,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een array kan zichzelf samenvatten: zijn som, zijn gemiddelde, of waar zijn grootste getal staat. Zonder axis krijg je één getal voor de hele array. Met een axis kies je de richting: de samenvatting gaat langs die richting, en er blijft één getal over per rij of per kolom.

    ### Nieuwe woorden en tekens

    Hieronder is `m` de matrix uit de vraag hierboven, `[[1, 2, 3], [4, 5, 6]]`. `axis=0` tussen de haakjes is een waarde met een naam: de naam `axis`, dan `=` en het getal.

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **`.sum()`** | telt alle getallen van de array op | `print(m.sum())` laat `21` zien |
    | **gemiddelde** | de som gedeeld door het aantal getallen; `.mean()` | `print(m.mean())` laat `3.5` zien |
    | **axis** | één richting van een array: axis 0 loopt langs de rijen omlaag, axis 1 loopt langs elke rij | |
    | **`axis=0`** | werkt langs de rijen omlaag: één uitkomst per kolom | `m.sum(axis=0)` geeft `array([5, 7, 9])` |
    | **`axis=1`** | werkt langs elke rij: één uitkomst per rij | `m.sum(axis=1)` geeft `array([ 6, 15])` |
    | **`.argmax()`** | de index van het grootste getal; met een axis één index per kolom of per rij | `print(np.array([4, 9, 2]).argmax())` laat `1` zien |

    `.max()` geeft het grootste getal zelf; `.argmax()` geeft zijn plek. Staat er op de laatste regel van een cel één getal uit een array, dan laat de cel het zien als `np.int64(21)` of `np.float64(3.5)`; `print` laat gewoon het getal zien.
    """)
    return


@app.cell(hide_code=True)
def picture_axes():
    _rows, _down, _across = C.picture()
    mo.vstack([
        draw.axis_results(_rows, _down, _across, down_label="m.sum(axis=0)", across_label="m.sum(axis=1)",
                          title="De sommen van m langs elke axis"),
        mo.md(r"""
        De gestippelde vakjes zijn wat eruit komt. `m.sum(axis=0)` gaat langs de rijen omlaag, dus geeft hij één som onder elke kolom: `5`, `7` en `9`. `m.sum(axis=1)` gaat langs elke rij, dus geeft hij één som naast elke rij: `6` en `15`. `m.sum()` telt alles op: `21`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`axis=0` betekent niet \"per rij\". Het is de richting waarin de samenvatting gaat: langs de rijen omlaag, dus verdwijnen de rijen en blijft er één getal per kolom over. Controleer het met de shape van de uitkomst."),
        mo.md(r"""
        ### Recept

        Zo vat je een array samen:

        1. Kies de samenvatting: `.sum()`, `.mean()` of `.argmax()`.
        2. Eén getal voor de hele array: geen axis.
        3. Eén getal per kolom: `axis=0`, langs de rijen omlaag. Eén getal per rij: `axis=1`, langs elke rij.
        4. Controleer de shape van de uitkomst: een matrix met shape `(3, 4)` geeft 4 getallen met `axis=0`, en 3 met `axis=1`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cijfers van drie leerlingen voor vier toetsen: een rij per leerling, een kolom per toets. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_grades():
    import numpy as np

    _grades = np.array([[7, 8, 6, 8], [5, 7, 8, 6], [9, 9, 7, 10]])
    # 1. en 2. het gemiddelde van alle cijfers: één getal
    print(_grades.mean())
    # 3. het gemiddelde van elke toets: langs de rijen omlaag, axis=0
    print(_grades.mean(axis=0))
    # 3. het gemiddelde van elke leerling: langs elke rij, axis=1
    print(_grades.mean(axis=1))
    # 1. en 3. de beste leerling van elke toets: de index van het hoogste cijfer, langs de rijen omlaag
    _best = _grades.argmax(axis=0)
    print(_best)
    # 4. de shape van die uitkomst: één getal per toets
    print(_best.shape)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het gemiddelde van alle twaalf cijfers is `7.5`. Per toets, langs de rijen omlaag, is het `[7. 8. 7. 8.]`: vier getallen, één per kolom. Per leerling, langs elke rij, is het `[7.25 6.5  8.75]`: drie getallen, één per rij. De beste leerling van elke toets is `[2 2 1 2]`: leerling 2, de derde rij, was de beste in drie toetsen, en leerling 1 in toets 2, de derde toets. De shape daarvan is `(4,)`: één getal per toets, zoals stap 4 verwacht.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weeks():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De regenval per week
    """)
    return


@app.cell(hide_code=True)
def choice_weeks_cell():
    choice_weeks = L.choice(
        "week-totals",
        question="`rain` heeft een rij per week en een kolom per dag: shape `(3, 4)`. Welke regel geeft de totale regenval per week?",
        options={
            "down": "`rain.sum(axis=0)`",
            "along": "`rain.sum(axis=1)`",
            "all": "`rain.sum()`",
        },
        correct="along",
        why={
            "down": "Die gaat langs de rijen omlaag en geeft één totaal per kolom: per dag, 4 getallen. Kijk nog eens naar het blok *Let op* boven het recept.",
            "along": "Een week is een rij, en `axis=1` telt op langs elke rij: 3 totalen, één per week.",
            "all": "Zonder axis krijg je één getal: alle regen van de drie weken samen. Hoe krijg je er één per rij?",
        },
    )
    choice_weeks
    return (choice_weeks,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het gemiddelde van elke dag

        De temperatuur op vier dagen, elke dag 's ochtends, 's middags en 's avonds: `temps` heeft een rij per dag, shape `(4, 3)`. Maak `day_means`: de gemiddelde temperatuur van elke dag. Vervang de `...`. De laatste regel is `day_means`, zodat de check het ziet.
        """),
        L.task("Maak `day_means`: één gemiddelde per dag, dus per rij van `temps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("day-means", lines=6)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("day-means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "day-means",
        [
            "Stap 3 van het recept: één getal per rij.",
            "Het gemiddelde is `.mean()`. Een dag is een rij, en één getal per rij betekent: langs elke rij.",
            "`day_means = temps.mean(axis=___)`.",
        ],
        why="`axis=1` gaat langs elke rij, dus houdt elke dag één getal over: het gemiddelde van zijn drie temperaturen. De eerste dag, 10, 16 en 13, heeft een gemiddelde van 13.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predictions():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wat ziet het model?

        Een model bekijkt foto's en geeft elke foto drie scores: voor een kat, een hond en een vogel, in die volgorde. Zijn voorspelling is het dier met de hoogste score.

        Maak `predictions(scores)` af: `scores` is een lijst met een rij scores per foto, en de functie maakt er al een array van. De functie geeft voor elke foto de index van zijn hoogste score terug: `0` voor kat, `1` voor hond, `2` voor vogel. Hij moet werken voor elk aantal foto's en dieren; zijn twee scores even hoog, dan geeft hij de eerste. Vervang de `...`. De check roept `predictions([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])` aan: dat moet `array([1, 0, 2, 2])` geven.
        """),
        L.task("Laat `predictions(scores)` de index van de hoogste score van elke foto teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predictions():
    box_predictions = L.code_box("predictions", lines=7)
    box_predictions
    return (box_predictions,)


@app.cell(hide_code=True)
def check_predictions(box_predictions):
    result_predictions = L.check("predictions", box_predictions.value)
    result_predictions
    return (result_predictions,)


@app.cell(hide_code=True)
def hints_predictions():
    L.hints(
        "predictions",
        [
            "Elke rij is één foto. Je wilt per rij de plek van zijn hoogste score.",
            "De plek van het grootste getal is `.argmax()`. Eén uitkomst per rij betekent: langs elke rij.",
            "`return scores.argmax(axis=___)`.",
        ],
        why="`scores.argmax(axis=1)` gaat langs elke rij en geeft de index van zijn hoogste score: `1` voor de eerste foto, een hond, omdat `0.7` zijn hoogste score is.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Herhalen: for-loops](/course/open/py-for?lang=nl) telde je getallen één voor één op in een loop; `.sum()` en `.mean()` doen dat in één stap. In [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl) zei de shape hoeveel rijen en kolommen er zijn; een axis zegt welke van de twee verdwijnt. In de volgende les, *Broadcasting*, reken je met arrays van verschillende shapes, zoals elk cijfer min het gemiddelde van zijn toets.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weeks, result_means, result_predictions):
    L.wrap_up(choice_weeks, result_means, result_predictions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
