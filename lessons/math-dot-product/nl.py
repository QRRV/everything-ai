# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-dot-product. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow function, vectorization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-dot-product", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl) en [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_two():
    quick_two = L.choice(
        "quick-two",
        question="Een model met twee features voorspelt prijs = 2,5 × oppervlakte + 10 × kamers + 30. Wat voorspelt het voor een oppervlakte van 100 en 4 kamers?",
        options={
            "added": "146,5",
            "no-rooms": "280",
            "right": "320",
        },
        correct="right",
        why={
            "added": "Dat telt de weights bij de features op. Wat doet ×? Zie oefening 3 van [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "no-rooms": "Dat is 2,5 × 100 + 30. Wat voegen de kamers toe? Zie oefening 3 van [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "right": "2,5 × 100 + 10 × 4 + 30 = 250 + 40 + 30 = 320: elke feature keer zijn eigen weight, plus de bias term.",
        },
    )
    quick_two
    return (quick_two,)


@app.cell(hide_code=True)
def quick_check_times():
    quick_times = L.choice(
        "quick-times",
        question="Wat geeft `np.array([1, 2, 3]) * np.array([4, 5, 6])`?",
        options={
            "right": "`array([4, 10, 18])`",
            "sum": "`32`",
            "plus": "`array([5, 7, 9])`",
        },
        correct="right",
        why={
            "sum": "Dat telt de producten ook nog op. Doet `*` dat? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "plus": "Dat is `+`. Wat doet `*` met de getallen op dezelfde plek? Zie [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "right": "Met twee arrays van dezelfde shape vermenigvuldigt `*` de getallen op dezelfde plek: 1 × 4, 2 × 5 en 3 × 6.",
        },
    )
    quick_times
    return (quick_times,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een huis heeft meer dan een oppervlakte: kamers, een tuin. Een model geeft elke feature een eigen weight, vermenigvuldigt elke feature met zijn weight, en telt alles op. Met drie features kun je dat nog uitschrijven; een model voor tekst heeft duizenden features, en dan lukt dat niet meer. Het inproduct is de korte manier om het te schrijven, en NumPy rekent het in één aanroep uit.

    **Een voorbeeld uit het dagelijks leven.** Een kassabon: 3 appels van 0,50, 2 broden van 2,50 en 1 pak melk van 1,20. Het totaal is per product het aantal keer de prijs, alles bij elkaar opgeteld: 1,50 + 5,00 + 1,20 = 7,70.
    """)
    return


@app.cell(hide_code=True)
def predict_receipt():
    guess_receipt = L.guess(
        "receipt",
        question="`quantities = np.array([3, 2, 1])` en `prices = np.array([0.5, 2.5, 1.2])`. Welke regel denk je dat het totaal op de bon geeft, 7,7, als één getal?",
        options={
            "times": "`quantities * prices`",
            "sums": "`quantities.sum() * prices.sum()`",
            "right": "`(quantities * prices).sum()`",
        },
        reveal="**`(quantities * prices).sum()`.** `quantities * prices` geeft de drie bedragen, `array([1.5, 5. , 1.2])`, één per product, en `.sum()` telt ze op tot 7,7. `quantities.sum() * prices.sum()` is 6 × 4,2 = 25,2: elk aantal keer elke prijs. Eerst de paren vermenigvuldigen, dan optellen: dat is het inproduct.",
    )
    guess_receipt
    return (guess_receipt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zet de weights in één vector, en de features van een voorbeeld in een andere, in dezelfde volgorde. Het inproduct vermenigvuldigt de getallen op dezelfde plek en telt de producten op. Een model met $d$ features voorspelt:
        """),
        L.formula(r"[[pred:\hat{y}]] = [[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]] = [[param:w_1]] x_1 + [[param:w_2]] x_2 + \dots + [[param:w_d]] x_d + [[param:b]]"),
        mo.md(r"""
        Met één feature is $\mathbf{w} \cdot \mathbf{x}$ gewoon $w$ keer $x$, het model dat je kent. Met meer is het de gewogen som van alle features, kort opgeschreven.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **inproduct**, $\mathbf{w} \cdot \mathbf{x}$ | w inproduct x | vermenigvuldig de getallen op dezelfde plek van twee vectoren van dezelfde lengte, en tel de producten op: één getal | `np.dot(w, x)` |
        | $\mathbf{w}$ | w | de vector met weights, één per feature | `w` |
        | $d$ | d | het aantal features: de lengte van $\mathbf{x}$ en van $\mathbf{w}$ | `len(x)` |
        | $x_1$, $w_1$ | x één, w één | de eerste feature en zijn weight | `x[0]`, `w[0]` |
        """),
        mo.Html(C.picture([2, 15, 20], [110, 5, 1], {"w": "w", "x": "x", "product": "w × x", "sum": "som", "alt": "Een inproduct vakje voor vakje. De weights 2, 15 en 20 staan boven de features 110, 5 en 1. Daaronder staan de producten: 220, 75 en 20. Opgeteld maken ze 315."})),
        mo.md(r"""
        Een huis van 110 m², met 5 kamers en een tuin (1 = ja), en de <span class="c-param">weights</span> 2, 15 en 20. Elke weight keer zijn eigen feature, en de producten opgeteld: het inproduct, 315.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo maak je een voorspelling met veel features:

    1. Zet de weights in één vector, en de features van het voorbeeld in een andere, in dezelfde volgorde.
    2. Vermenigvuldig de getallen op dezelfde plek.
    3. Tel de producten op: dat is het inproduct, `np.dot(w, x)`.
    4. Tel de bias term erbij op.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De voorspelling voor het huis uit de figuur, met de bias term $b = 30$: eerst stap voor stap, dan met `np.dot` in één aanroep. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_dot():
    import numpy as np

    # 1. weights en features, in dezelfde volgorde: oppervlakte, kamers, tuin
    _w = np.array([2, 15, 20])
    _x = np.array([110, 5, 1])
    _b = 30
    # 2. de getallen op dezelfde plek, vermenigvuldigd
    print(_w * _x)
    # 3. opgeteld: het inproduct, met een loop plek voor plek, en met np.dot
    _total = 0
    for _i in range(len(_w)):
        _total = _total + _w[_i] * _x[_i]
    print(_total, (_w * _x).sum(), np.dot(_w, _x))
    # 4. plus de bias term: de voorspelling
    print(np.dot(_w, _x) + _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_w * _x` geeft de drie producten, `[220  75  20]`. De loop telt ze plek voor plek op; `.sum()` en `np.dot` doen hetzelfde, en alle drie geven 315: `np.dot` doet stap 2 en 3 in één keer. Met de bias term voorspelt het model 345 duizend euro. Met 300 features zou de code precies hetzelfde zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hand():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een inproduct met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_hand_cell():
    choice_hand = L.choice(
        "by-hand",
        question="Wat is het inproduct van $(2, 0, 1)$ en $(3, 7, 5)$?",
        options={
            "products": "$(6, 0, 5)$",
            "right": "11",
            "all": "18",
        },
        correct="right",
        why={
            "products": "Dat zijn de producten. Wat is de laatste stap van een inproduct? Zie het recept.",
            "all": "Dat telt alle zes getallen op. Wat gebeurt er eerst met de getallen op dezelfde plek? Zie het recept.",
            "right": "2 × 3 + 0 × 7 + 1 × 5 = 6 + 0 + 5 = 11: één getal.",
        },
    )
    choice_hand
    return (choice_hand,)


@app.cell(hide_code=True)
def text_exercise_grade():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voorspel een cijfer

        Een model voorspelt het cijfer van een leerling uit twee features: uren leren en uren slaap, zoals bij de leerlingen in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl). Het vak heeft de weights `w`, de bias term `b` en de features `x` van één leerling. Zet de voorspelling in `y_pred`. Vervang de `...`. De laatste regel is `y_pred`, zodat de check hem ziet. De check probeert ook een andere leerling.
        """),
        L.task("Zet het voorspelde cijfer in `y_pred`, met een inproduct."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grade():
    box_grade = L.code_box("grade", lines=8)
    box_grade
    return (box_grade,)


@app.cell(hide_code=True)
def check_grade(box_grade):
    result_grade = L.check("grade", box_grade.value)
    result_grade
    return (result_grade,)


@app.cell(hide_code=True)
def hints_grade():
    L.hints(
        "grade",
        [
            "Het recept, stap 2 tot en met 4: vermenigvuldig de paren, tel op, tel de bias term erbij op.",
            "NumPy doet stap 2 en 3 in één aanroep. Welke aanroep is dat? Kijk bij stap 3 van het recept.",
            "`y_pred = np.dot(___, ___) + ___`.",
        ],
        why="`np.dot(w, x)` is 0,5 × 6 + 0,3 × 8 = 5,4, en `+ b` maakt er 6,4 van.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wat np.dot doet

        Maak `dot(a, b)` af: het inproduct van twee lijsten van dezelfde lengte, in gewone Python, zonder NumPy. Hij geeft één getal terug. Vervang de `...`. De check roept `dot([1, 2, 3], [4, 5, 6])` aan: dat moet `32` geven. De check probeert ook andere lijsten.
        """),
        L.task("Laat `dot` het inproduct teruggeven, met een loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("dot-loop", lines=7)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("dot-loop", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "dot-loop",
        [
            "Begin met een totaal van 0, en loop de plekken één voor één langs.",
            "`range(len(a))` geeft elke plek. Wat tel je bij elke plek bij het totaal op?",
            "`total = total + a[i] * ___`, en na de loop `return ___`.",
        ],
        why="De loop gaat elke plek `i` langs, vermenigvuldigt `a[i]` met `b[i]`, en telt dat bij `total` op. Voor `[1, 2, 3]` en `[4, 5, 6]`: 4 + 10 + 18 = 32.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl) telde een model met twee features elke feature keer zijn weight op; het inproduct is die gewogen som, voor elk aantal features. In [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) vermenigvuldigde `*` de getallen op dezelfde plek; `np.dot` telt ze ook nog op. In de volgende les, *Een hele batch tegelijk: matrixvermenigvuldiging*, reken je de inproducten van veel voorbeelden tegelijk uit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hand, result_grade, result_loop):
    L.wrap_up(choice_hand, result_grade, result_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
