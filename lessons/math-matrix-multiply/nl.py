# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-matrix-multiply. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-multiply", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="Wat is het inproduct van $(1, 2)$ en $(3, 4)$?",
        options={
            "products": "$(3, 8)$",
            "all": "10",
            "right": "11",
        },
        correct="right",
        why={
            "products": "Dat zijn de producten. Wat is de laatste stap van een inproduct? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "all": "Dat telt alle vier de getallen op. Wat gebeurt er eerst met de getallen op dezelfde plek? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "right": "1 × 3 + 2 × 4 = 3 + 8 = 11.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_np_dot():
    quick_np_dot = L.choice(
        "quick-np-dot",
        question="Wat doet `np.dot(w, x)` dat `w * x` niet doet?",
        options={
            "sum": "Het telt de producten op",
            "bias": "Het telt de bias term op",
            "nothing": "Niets: ze geven hetzelfde",
        },
        correct="sum",
        why={
            "bias": "De bias term tel je zelf op, na het inproduct. Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "nothing": "`w * x` geeft een array, `np.dot(w, x)` één getal. Zie het uitgewerkte voorbeeld in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "sum": "`w * x` vermenigvuldigt de getallen op dezelfde plek; `np.dot` telt die producten ook nog op, tot één getal.",
        },
    )
    quick_np_dot
    return (quick_np_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model maakt een voorspelling voor elk voorbeeld, en een dataset heeft er veel. Een inproduct per voorbeeld, in een loop, is traag en omslachtig. Met alle voorbeelden in één matrix geeft één vermenigvuldiging elke voorspelling tegelijk. Elke layer van een neural network, later in de course, begint met precies dit.

    **Een voorbeeld uit het dagelijks leven.** Een winkel heeft een tabel met boodschappenmandjes: één rij per klant, één kolom per product, en in elk vakje hoeveel er gekocht is. Vermenigvuldig elke rij met de prijslijst en tel op, en je hebt het totaal van elke klant tegelijk.
    """)
    return


@app.cell(hide_code=True)
def predict_shape():
    guess_shape = L.guess(
        "shape",
        question="`@`, het apenstaartje, is in Python het teken voor matrixvermenigvuldiging. `X` bevat 8 huizen, met elk 3 features: shape (8, 3). `w` bevat 3 weights: shape (3,). Wat denk je dat `X @ w` geeft?",
        options={
            "one": "Eén getal",
            "eight": "8 getallen: één voor elk huis",
            "matrix": "Een matrix van 3 bij 3",
        },
        reveal="**8 getallen, één voor elk huis.** `@` neemt het inproduct van elke rij van `X`, één huis, met `w`. 8 rijen geven 8 inproducten: shape (8,).",
    )
    guess_shape
    return (guess_shape,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zet de voorbeelden in een matrix $\mathbf{X}$, één rij per voorbeeld en één kolom per feature. Matrixvermenigvuldiging neemt het inproduct van elke rij met de weights: één voorspelling per voorbeeld, allemaal tegelijk.
        """),
        L.formula(r"[[pred:\hat{\mathbf{y}}]] = \mathbf{X} [[param:\mathbf{w}]] + [[param:b]]"),
        mo.md(r"""
        Dit werkt ook met een matrix van weights, met één kolom per output: elke rij van de eerste matrix keer elke kolom van de tweede. De shapes moeten passen: het aantal kolommen van de eerste moet gelijk zijn aan het aantal rijen van de tweede. Dan geeft $(n, k)$ keer $(k, m)$ een $(n, m)$. Met een vector werkt het net zo: $(n, k)$ keer $(k,)$ geeft $(n,)$, want `w` doet mee als één kolom.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **matrixvermenigvuldiging** | | elk getal van de uitkomst is het inproduct van een rij van de eerste matrix met een kolom van de tweede | `X @ w` |
        | **shape-regel** | | $(n, k)$ keer $(k, m)$ geeft $(n, m)$: de twee binnenste getallen moeten gelijk zijn, en de buitenste blijven over | |
        | $\mathbf{X}$ | hoofdletter X | alle voorbeelden als matrix, één rij per voorbeeld en één kolom per feature; shape (aantal voorbeelden, $d$) | `X` |
        | $\hat{\mathbf{y}}$ | y-dakje | de voorspellingen van alle voorbeelden, één per rij | `y_pred` |
        """),
        L.watch_out("`@`, het apenstaartje, is in Python het teken voor matrixvermenigvuldiging, ook voor een matrix keer een vector. Het is niet `*`: bij twee arrays met dezelfde shape vermenigvuldigt `*` de getallen op dezelfde plek en telt niets op."),
        draw.array_grids([("X", [[str(_v) for _v in _row] for _row in C.HOUSES], ["shape (4, 3)", "de derde rij: één huis"], {(2, 0), (2, 1), (2, 2)}),
                          ("w", [[str(_v) for _v in C.W]], ["shape (3,)"], {(0, 0), (0, 1), (0, 2)}),
                          ("X @ w", [["169", "250", "315", "126"]], ["shape (4,)", "het derde getal: 315"], {(0, 2)})],
                         title="Vier huizen tegelijk"),
        mo.md(r"""
        Vier huizen, met oppervlakte, kamers en tuin, en de weights uit [Veel features: het inproduct](/course/open/math-dot-product?lang=nl). De derde rij van `X` keer `w` is 110 × 2 + 5 × 15 + 1 × 20 = 315: het derde getal van `X @ w`. Elke rij geeft zijn eigen inproduct.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo maak je de voorspellingen voor veel voorbeelden tegelijk:

    1. Zet de voorbeelden in een matrix `X`, één rij per voorbeeld, met de features in dezelfde volgorde als de weights.
    2. Controleer de shapes: de binnenste getallen moeten gelijk zijn, zoals bij (4, 3) en (3,).
    3. Vermenigvuldig met `@`: elke rij van `X` met `w` geeft één inproduct.
    4. Tel de bias term erbij: NumPy telt hem bij elke voorspelling tegelijk op.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De voorspellingen voor de vier huizen uit de figuur, met $b = 30$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_matmul():
    import numpy as np

    # 1. één rij per huis: oppervlakte, kamers, tuin
    _X = np.array([[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]])
    _w = np.array([2, 15, 20])
    _b = 30
    # 2. de shapes: de binnenste getallen zijn allebei 3
    print(_X.shape, _w.shape)
    # 3. elk inproduct tegelijk
    print(_X @ _w)
    # 4. plus de bias term: de voorspellingen
    _y_pred = _X @ _w + _b
    print(_y_pred, _y_pred.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X` heeft shape (4, 3) en `_w` shape (3,): de drieën passen. `_X @ _w` geeft de vier inproducten, 169, 250, 315 en 126, en `+ _b` telt bij elk 30 op: de voorspellingen zijn 199, 280, 345 en 156 duizend euro, shape (4,). Met 8000 huizen zou de code precies hetzelfde zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke shape?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "result-shape",
        question="`A` heeft shape (5, 3) en `B` heeft shape (3, 2). Wat is de shape van `A @ B`?",
        options={
            "inner": "(3, 3)",
            "right": "(5, 2)",
            "error": "Geen shape: ze passen niet",
        },
        correct="right",
        why={
            "inner": "Dat zijn de binnenste getallen. Welke getallen blijven over? Zie de shape-regel in *Het idee*.",
            "error": "Kijk naar de binnenste getallen: de 3 van `A` en de 3 van `B`. Zijn ze gelijk? Zie de shape-regel in *Het idee*.",
            "right": "De binnenste getallen zijn allebei 3, dus ze passen, en de buitenste blijven over: (5, 2).",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_grades():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Alle leerlingen tegelijk

        Het model uit [Veel features: het inproduct](/course/open/math-dot-product?lang=nl) voorspelt een cijfer uit uren leren en uren slaap. Nu bevat `X` drie leerlingen, één per rij: shape (3, 2). `w` bevat de twee weights, shape (2,), en `b` is de bias term. Zet de voorspellingen voor alle drie in `y_pred`, met `@`. Vervang de `...`. De laatste regel is `y_pred`, zodat de check hem ziet. De check probeert ook andere leerlingen.
        """),
        L.task("Zet de voorspelde cijfers van alle leerlingen in `y_pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grades():
    box_grades = L.code_box("all-grades", lines=8)
    box_grades
    return (box_grades,)


@app.cell(hide_code=True)
def check_grades(box_grades):
    result_grades = L.check("all-grades", box_grades.value)
    result_grades
    return (result_grades,)


@app.cell(hide_code=True)
def hints_grades():
    L.hints(
        "all-grades",
        [
            "Stap 3 en 4 van het recept: vermenigvuldig met `@`, en tel dan de bias term erbij.",
            "Wat komt eerst, de matrix met de leerlingen of de weights? Controleer de shapes: (3, 2) en (2,).",
            "`y_pred = X @ ___ + ___`.",
        ],
        why="`X @ w` neemt het inproduct van de rij van elke leerling met `w`: 5,4; 7,1 en 3,7. `+ b` telt bij elk 1 op: 6,4; 8,1 en 4,7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De shape-regel als functie

        Maak `result_shape(a, b)` af. Hij krijgt twee shapes, elk een tuple van twee getallen, zoals `(4, 2)`. Hij geeft de shape van de matrixvermenigvuldiging terug, als tuple, of `False` als de shapes niet passen. Vervang de `...`. De check roept `result_shape((4, 2), (2, 3))` aan: dat moet `(4, 3)` geven. De check probeert ook andere shapes, ook shapes die niet passen.
        """),
        L.task("Laat `result_shape` de shape-regel toepassen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("shape-rule", lines=6)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("shape-rule", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "shape-rule",
        [
            "Twee stappen: controleer eerst of de shapes passen, en geef dan de nieuwe shape terug.",
            "`a[0]` en `a[1]` zijn de rijen en kolommen van de eerste shape, `b[0]` en `b[1]` die van de tweede. Welke twee zijn de binnenste getallen?",
            "`if a[1] != ___:` met `return False`, en anders `return (a[0], ___)`.",
        ],
        why="De shapes passen als het aantal kolommen van `a` gelijk is aan het aantal rijen van `b`: `a[1] == b[0]`. Dan heeft de uitkomst de rijen van `a` en de kolommen van `b`: `(a[0], b[1])`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Veel features: het inproduct](/course/open/math-dot-product?lang=nl) rekende je één inproduct uit voor één voorbeeld; matrixvermenigvuldiging rekent er één per rij uit, voor elk voorbeeld tegelijk, en de shape-regel zegt wanneer twee shapes passen. In de volgende les, *Een matrix omklappen: de getransponeerde*, wissel je de rijen en kolommen van een matrix, en daarmee gaan twee shapes vaak passen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_grades, result_rule):
    L.wrap_up(choice_shape, result_grades, result_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
