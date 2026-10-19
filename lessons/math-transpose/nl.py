# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-transpose. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("math-transpose", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` heeft shape (4, 3) en `w` heeft shape (3,). Wat is de shape van `X @ w`?",
        options={
            "inner": "(3,)",
            "same": "(4, 3)",
            "right": "(4,)",
        },
        correct="right",
        why={
            "inner": "Dat is het binnenste getal. Welk getal blijft over? Zie de shape-regel in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "same": "Elke rij van `X` geeft één inproduct: één getal. Hoeveel rijen zijn er? Zie het uitgewerkte voorbeeld in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "De binnenste getallen zijn allebei 3, en de 4 blijft over: één voorspelling voor elk van de 4 rijen.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="`A` en `B` hebben allebei shape (2, 3). Werkt `A @ B`?",
        options={
            "equal": "Ja: de shapes zijn gelijk",
            "right": "Nee: de binnenste getallen, 3 en 2, verschillen",
            "result": "Ja: het geeft shape (2, 3)",
        },
        correct="right",
        why={
            "equal": "De shape-regel vraagt niet om gelijke shapes. Welke twee getallen moeten gelijk zijn? Zie de shape-regel in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "result": "Kijk eerst naar de binnenste getallen: de 3 van `A` en de 2 van `B`. Zie de shape-regel in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "Het aantal kolommen van `A`, 3, moet gelijk zijn aan het aantal rijen van `B`, 2. Dat is het niet, dus Python stopt met een fout.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vaak is er meer dan één rij weights: twee modellen, of één model dat twee dingen voorspelt. Elk krijgt zijn eigen rij weights in een matrix `W`. Om ze te gebruiken, moet elk voorbeeld, een rij van `X`, elke rij van `W` tegenkomen. Maar matrixvermenigvuldiging neemt rijen van de eerste keer *kolommen* van de tweede, dus de shapes passen niet. Wissel de rijen en kolommen van `W`, en ze passen wel. Een layer van een neural network in PyTorch, later in de course, bewaart zijn weights precies zo.

    **Een voorbeeld uit het dagelijks leven.** Een cijferlijst heeft één rij per leerling en één kolom per vak. Het rapportformulier wil één rij per vak en één kolom per leerling: om het in te vullen, draai je de lijst om. De cijfers blijven hetzelfde; alleen de rijen en kolommen ruilen van plaats.
    """)
    return


@app.cell(hide_code=True)
def predict_fit():
    guess_fit = L.guess(
        "fit",
        question="Twee modellen, met elk 3 weights, zijn de rijen van `W`: shape (2, 3). `X` bevat 4 huizen: shape (4, 3). Welke vermenigvuldiging denk je dat werkt?",
        options={
            "xw": "`X @ W`",
            "wx": "`W @ X`",
            "neither": "Geen van beide",
        },
        reveal="**Geen van beide.** (4, 3) en (2, 3): de binnenste getallen zijn 3 en 2. (2, 3) en (4, 3): 3 en 4. Wissel de rijen en kolommen van `W`, dan wordt de shape (3, 2): dan passen (4, 3) en (3, 2), en geven ze (4, 2), één voorspelling per huis per model.",
    )
    guess_fit
    return (guess_fit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Transponeren wisselt de rijen en kolommen van een matrix: de eerste rij wordt de eerste kolom, de tweede rij de tweede kolom, enzovoort. Wat eruit komt, heet de getransponeerde. De getallen blijven hetzelfde, maar de shape draait om.
        """),
        L.formula(r"\mathbf{A} \text{ heeft shape } (n, m) \quad\Longrightarrow\quad \mathbf{A}^\top \text{ heeft shape } (m, n)"),
        mo.md(r"""
        Met de weights van twee modellen als rijen van $\mathbf{W}$ zijn de voorspellingen van allebei voor elk voorbeeld dus $\mathbf{X}\mathbf{W}^\top$: elke rij van $\mathbf{X}$ keer elke rij van $\mathbf{W}$, die nu een kolom is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **getransponeerde** | | de matrix met zijn rijen en kolommen gewisseld: de eerste rij wordt de eerste kolom, dus shape $(n, m)$ wordt $(m, n)$; het werkwoord is transponeren | `A.T` |
        | $\mathbf{A}^\top$ | A getransponeerd | de getransponeerde van de matrix $\mathbf{A}$ | `A.T` |
        """),
        L.watch_out("`.T` verandert niets aan een array met maar één getal in zijn shape, zoals `w` met shape (3,): er zijn geen rijen en kolommen om te wisselen. `w.T` is gewoon `w`."),
        draw.array_grids([("A", [[str(_v) for _v in _row] for _row in C.A], ["shape (2, 3)", "de eerste rij: 1, 2, 3"], {(0, 0), (0, 1), (0, 2)}),
                          ("A.T", [[str(_row[_j]) for _row in C.A] for _j in range(3)], ["shape (3, 2)", "de eerste kolom: 1, 2, 3"], {(0, 0), (1, 0), (2, 0)})],
                         title="Rijen worden kolommen"),
        mo.md(r"""
        De eerste rij van `A`, 1, 2 en 3, is de eerste kolom van `A.T`, van boven naar beneden. De tweede rij, 4, 5 en 6, is de tweede kolom. De 2 rijen van 3 zijn 3 rijen van 2 geworden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo laat je twee shapes passen met een getransponeerde:

    1. Schrijf beide shapes op.
    2. Verschillen de binnenste getallen? Kijk of ze gelijk worden als je de rijen en kolommen van een van de twee wisselt.
    3. Transponeer die met `.T`.
    4. Vermenigvuldig met `@`, en controleer de shape van de uitkomst: de rijen en kolommen die je wilde?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vier huizen uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl), met twee modellen: de weights uit die les, en een tweede model met andere weights, allebei zonder bias term. Elk model is een rij van `_W`. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_transpose():
    import numpy as np

    _X = np.array([[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]])
    _W = np.array([[2, 15, 20], [0.5, 3, 1]])
    # 1. de shapes: de binnenste getallen zijn 3 en 2
    print(_X.shape, _W.shape)
    # 2. en 3. de getransponeerde van _W heeft shape (3, 2): nu passen de drieën
    print(_W.T.shape)
    # 4. vermenigvuldig, en controleer de shape van de uitkomst
    _y_pred = _X @ _W.T
    print(_y_pred)
    print(_y_pred.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X` heeft shape (4, 3) en `_W` shape (2, 3): die passen niet. `_W.T` heeft shape (3, 2), en `_X @ _W.T` geeft shape (4, 2): één rij per huis, één kolom per model. De eerste kolom, 169, 250, 315 en 126, is `X @ w` uit de vorige les, vóór de bias term erbij kwam; de tweede kolom is het tweede model.

    Zonder NumPy is een getransponeerde een loop. De cel hieronder verzamelt de eerste kolom van `A` uit de figuur: het eerste getal van elke rij.
    """)
    return


@app.cell
def example_column():
    _rows = [[1, 2, 3], [4, 5, 6]]
    _column = []
    for _row in _rows:
        _column.append(_row[0])
    print(_column)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een matrix keer zijn getransponeerde
    """)
    return


@app.cell(hide_code=True)
def choice_gram_cell():
    choice_gram = L.choice(
        "gram",
        question="`A` heeft shape (5, 2). Wat is de shape van `A.T @ A`?",
        options={
            "right": "(2, 2)",
            "outer": "(5, 5)",
            "error": "Geen shape: ze passen niet",
        },
        correct="right",
        why={
            "outer": "Dat is de shape van `A @ A.T`. Wat is de shape van `A.T`, en welke getallen blijven over? Zie het recept.",
            "error": "Schrijf eerst de shape van `A.T` op. Kijk dan naar de binnenste getallen. Zie de formule in *Het idee*.",
            "right": "`A.T` heeft shape (2, 5). (2, 5) en (5, 2): de binnenste vijven passen, en de buitenste getallen blijven over: (2, 2).",
        },
    )
    choice_gram
    return (choice_gram,)


@app.cell(hide_code=True)
def text_exercise_models():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twee modellen, alle leerlingen

        Twee modellen voorspellen een cijfer uit uren leren en uren slaap. Elk model is een rij van `W`: het eerste is het model uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) zonder bias term, het tweede laat slaap zwaarder wegen. `X` bevat drie leerlingen, één per rij: shape (3, 2). `W` heeft shape (2, 2). Zet de voorspellingen van beide modellen voor alle leerlingen in `y_pred`, met één rij per leerling en één kolom per model. Vervang de `...`. De laatste regel is `y_pred`, zodat de check hem ziet. De check probeert ook andere leerlingen.
        """),
        L.task("Zet de voorspellingen van beide modellen voor alle leerlingen in `y_pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_models():
    box_models = L.code_box("two-models", lines=8)
    box_models
    return (box_models,)


@app.cell(hide_code=True)
def check_models(box_models):
    result_models = L.check("two-models", box_models.value)
    result_models
    return (result_models,)


@app.cell(hide_code=True)
def hints_models():
    L.hints(
        "two-models",
        [
            "Stap 1 van het recept: schrijf de shapes van `X` en `W` op. Elke leerling moet elk model tegenkomen.",
            "De opdracht vraagt om één rij per leerling, dus `X` komt eerst. Wat moeten de modellen zijn, rijen of kolommen?",
            "`y_pred = X @ ___`.",
        ],
        why="`W.T` heeft shape (2, 2) met één model per kolom, en `X @ W.T` geeft shape (3, 2). De eerste leerling, 6 uur leren en 8 uur slaap, krijgt 6 × 0,5 + 8 × 0,3 = 5,4 van het eerste model en 6 × 0,2 + 8 × 0,6 = 6,0 van het tweede. `X @ W` past ook, want `W` is vierkant, maar dan komt elke leerling een kolom van `W` tegen, geen model.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lists():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een getransponeerde zonder NumPy

        Maak `transpose(rows)` af. Hij krijgt een matrix als lijst van rijen, elk een lijst getallen, en geeft de getransponeerde terug, ook als lijst van rijen. Vervang de `...`. De check roept `transpose([[1, 2], [3, 4], [5, 6]])` aan: dat moet `[[1, 3, 5], [2, 4, 6]]` geven. Hij probeert ook andere matrices.
        """),
        L.task("Laat `transpose` de rijen en kolommen wisselen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lists():
    box_lists = L.code_box("transpose-lists", lines=8)
    box_lists
    return (box_lists,)


@app.cell(hide_code=True)
def check_lists(box_lists):
    result_lists = L.check("transpose-lists", box_lists.value)
    result_lists
    return (result_lists,)


@app.cell(hide_code=True)
def hints_lists():
    L.hints(
        "transpose-lists",
        [
            "De eerste nieuwe rij is de eerste oude kolom: het eerste getal van elke rij. De tweede nieuwe rij bevat het tweede getal van elke rij.",
            "Twee loops: een buitenste over de plekken in een rij, en een binnenste over de rijen. Hoeveel plekken heeft een rij?",
            "Begin met `columns = []`. Dan `for j in range(___):`, met daarin een nieuwe lijst `column`, een loop `for row in rows:` die `row[___]` toevoegt, en `columns.append(column)`. Aan het eind `return columns`.",
        ],
        why="Voor elke plek `j` in een rij verzamelt een loop over de rijen met `column.append(row[j])` de getallen op plek `j`: één kolom. Die kolommen, op volgorde, zijn de rijen van de getransponeerde. `len(rows[0])` is het aantal kolommen: 2 in het voorbeeld, en dat worden 2 rijen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) moesten twee shapes passen. De getransponeerde wisselt rijen en kolommen, en daarmee gaan ze vaak passen: met de weights van meerdere modellen als rijen van $\mathbf{W}$ geeft $\mathbf{X}\mathbf{W}^\top$ elke voorspelling van elk model tegelijk. In de volgende les, *Lengte, afstand en cosine similarity*, meet je hoe lang een vector is, en hoe ver twee vectoren uit elkaar liggen of hoeveel ze op elkaar lijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gram, result_lists, result_models):
    L.wrap_up(choice_gram, result_models, result_lists)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
