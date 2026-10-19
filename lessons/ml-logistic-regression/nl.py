# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-logistic-regression. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-logistic-regression", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) en [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logit():
    quick_logit = L.choice(
        "quick-logit",
        question="Een logit van 3 gaat door de sigmoid. Welke kans komt er ongeveer uit?",
        options={
            "low": "0,05",
            "right": "0,95",
            "same": "3",
        },
        correct="right",
        why={
            "low": "Dat is $\\sigma(-3)$. Ligt 3 boven of onder 0? Zie de figuur in [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "same": "De sigmoid maakt van elk getal een getal tussen 0 en 1. Zie *Het idee* in [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "right": "$e^{-3}$ is ongeveer 0,05, dus $\\sigma(3) = 1 / 1{,}05 \\approx 0{,}95$: een positieve logit geeft meer dan 0,5.",
        },
    )
    quick_logit
    return (quick_logit,)


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="Wat geeft `np.dot([2, 5], [3, 4])`?",
        options={
            "products": "`array([ 6, 20])`",
            "sum": "`14`",
            "right": "`26`",
        },
        correct="right",
        why={
            "products": "Dat zijn de producten. Wat is de laatste stap van een inproduct? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "sum": "Dat telt alle vier de getallen op. Wat gebeurt er eerst met de getallen op dezelfde plek? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "right": "2 × 3 + 5 × 4 = 6 + 20 = 26.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) was de logit gegeven. Een model moet hem zelf uitrekenen, uit de features van elk voorbeeld. Je kent al een manier: in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl) gaf een model elke feature een eigen weight en telde alles op. Zet de sigmoid erbovenop, en je hebt logistische regressie: een model voor twee klassen dat een kans geeft. Het is een van de meest gebruikte modellen, en later in de course is het de laatste stap van veel neural networks.

    **Een voorbeeld uit het dagelijks leven.** Een docent schat in wie er voor een examen slaagt. Elk uur leren telt, slaap telt ook een beetje, en er is een lat om overheen te komen. Een leerling ver boven de lat slaagt heel waarschijnlijk; bij een leerling net erboven kan het nog alle kanten op.
    """)
    return


@app.cell(hide_code=True)
def predict_student():
    guess_student = L.guess(
        "student",
        question="Een model voor de leerlingen van [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) heeft de weights 1 voor elk uur leren en 0,5 voor elk uur slaap, en de bias term −8. Een leerling leerde 6 uur en sliep 6 uur. Wat voorspelt het model volgens jou?",
        options={
            "pass": "Geslaagd",
            "fail": "Gezakt",
            "half": "Precies fifty-fifty",
        },
        reveal="**Geslaagd, met ongeveer 0,73.** De logit is 1 × 6 + 0,5 × 6 − 8 = 1: boven 0, dus de kans is boven 0,5. $\\sigma(1)$ is ongeveer 0,73.",
    )
    guess_student
    return (guess_student,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Logistische regressie** rekent de logit uit als het inproduct van de weights en de features, plus de bias term, en haalt die door de sigmoid. De <span class="c-pred">voorspelling</span> $\hat{y}$ is nu een kans: de kans op klasse 1, zoals geslaagd.
        """),
        L.formula(r"[[pred:\hat{y}]] = \sigma([[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]])"),
        mo.md(r"""
        Het model voorspelt klasse 1 waar $\hat{y}$ boven 0,5 ligt. Dat is precies waar de logit boven 0 ligt. Waar de logit precies 0 is, twijfelt het model: 0,5. Die punten vormen de **decision boundary**: aan de ene kant voorspelt het model klasse 1, aan de andere kant klasse 0.

        Met twee features, $x_1$ en $x_2$, is de decision boundary de rechte lijn $w_1 x_1 + w_2 x_2 + b = 0$. De weights kantelen de lijn; de bias term verschuift hem.

        Het punt op de lijn bij een gegeven $x_2$ vind je door op te lossen naar $x_1$. Haal de andere twee termen naar de andere kant van het $=$-teken; dan draaien hun tekens om: $w_1 x_1 = -b - w_2 x_2$. Deel dan door $w_1$: $x_1 = (-b - w_2 x_2) / w_1$. Met de weights 1 en 0,5 en $b = -8$ ligt een leerling die 6 uur sliep op de lijn bij $x_1 = (8 - 0{,}5 \times 6) / 1 = 5$ uur leren: 5 + 3 − 8 = 0.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **logistische regressie** | | een model voor twee klassen: de sigmoid van het inproduct van weights en features, plus de bias term | `1 / (1 + np.exp(-(np.dot(w, x) + b)))` |
        | **decision boundary** | | waar de logit 0 is en de kans 0,5: de lijn tussen de twee klassen die het model voorspelt | `np.dot(w, x) + b == 0` |
        """),
        L.watch_out("Ondanks de naam doet logistische regressie classificatie: het model voorspelt een klasse, met een kans. Het woord *regressie* in de naam komt van de gewogen som erbinnen, die hetzelfde is als in een model dat een getal voorspelt."),
        mo.md(r"""
        De leerlingen van [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) van wie de uitslag bekend is, naar uren leren en uren slaap. De gevulde punten slaagden, de open punten zakten. De <span class="c-pred">blauwe lijn</span> is de decision boundary van het model met de weights 1 en 0,5. Schuif met de slider: hij zet de <span class="c-param">bias term</span> $b$, en verschuift de lijn. De andere grafiek telt voor elke $b$ de leerlingen aan de <span class="c-loss">verkeerde kant</span> van de lijn (gevuld) en er precies op (open); sleep eroverheen om $b$ te zetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        wrong, on = C.sides(b)
        on_text = f", en {on} er precies op, waar het model twijfelt" if on else ""
        return (C.picture(b, {"title": "Leerlingen en de decision boundary", "hours": "uren leren", "sleep": "uren slaap",
                              "above": "geslaagd", "below": "gezakt", "wrong": "Aan de verkeerde kant, voor elke b", "count": "leerlingen", "on": "op de lijn",
                              "alt": f"Zeven leerlingen naar uren leren en slaap: vier gevulde punten die slaagden, drie open punten die zakten, en de decision boundary als rechte lijn die van linksboven naar rechtsonder daalt. Met b = {L.number(b, 1)} zet het model {wrong} van de 7 leerlingen aan de verkeerde kant van de lijn{on_text}. Een tweede grafiek toont voor elke b op de slider hoeveel leerlingen aan de verkeerde kant liggen (gevulde punten) en precies op de lijn (open punten), met een verticale lijn bij de b van de slider: 2 aan de verkeerde kant bij −10 en −9,5; 2 op de lijn bij −9; van −8,5 tot −7,5 geen aan de verkeerde kant en geen op de lijn; 1 op de lijn bij −7; en 1 aan de verkeerde kant bij −6,5 en −6."}),
                f"""Met <i>b</i> = {L.number(b, 1)} zet het model <span class="ea-num">{wrong}</span> van de 7 leerlingen aan de verkeerde kant van de lijn{on_text}.""")

    explore_b = L.explore("boundary-b", "b", _frame, start=-6, low=-10, high=-6, step=0.5, role="param", task="Zoek de waarden van $b$ waarbij elke leerling aan de goede kant van de lijn ligt.")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met $b$ tussen −8,5 en −7,5 staat elke leerling aan de goede kant. Bij −9 en −7 staat niemand aan de verkeerde kant, maar liggen er leerlingen precies op de lijn, waar het model twijfelt. Training, in een latere les, vindt zulke weights en zo'n bias term zelf.

    ### Recept

    Zo laat je logistische regressie een klasse voorspellen:

    1. Zet de features van het voorbeeld in een vector $\mathbf{x}$, en neem de weights $\mathbf{w}$ en de bias term $b$ van het model.
    2. Reken de logit uit: `z = np.dot(w, x) + b`.
    3. Haal hem door de sigmoid: `1 / (1 + np.exp(-z))`.
    4. Boven 0,5 voorspelt het model klasse 1: het voorbeeld ligt aan die kant van de decision boundary.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model met de weights 1 en 0,5 en de bias term −8, eerst voor één leerling die 5 uur leerde en 8 uur sliep, dan voor de drie leerlingen uit [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) van wie de uitslag nog niet bekend is. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_logistic():
    import numpy as np

    # 1. het model, en één leerling: 5 uur leren, 8 uur slaap
    _w = np.array([1, 0.5])
    _b = -8
    _x = np.array([5, 8])
    # 2. de logit
    _z = np.dot(_w, _x) + _b
    print(_z)
    # 3. de kans dat de leerling slaagt
    _p = 1 / (1 + np.exp(-_z))
    print(_p)
    # 4. de klasse
    print(_p > 0.5)
    # 2. tot en met 4. drie leerlingen tegelijk: uren leren en slaap elk in een array
    _hours = np.array([4, 7, 2])
    _sleep = np.array([7, 6, 9])
    _z3 = _w[0] * _hours + _w[1] * _sleep + _b
    _p3 = 1 / (1 + np.exp(-_z3))
    print(_p3)
    print(_p3 > 0.5)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De leerling met 5 en 8 krijgt de logit 5 + 4 − 8 = 1 en een kans van 0,73: geslaagd, en dat gebeurde ook. Voor de drie nieuwe leerlingen zijn de logits −0,5, 2 en −1,5: het model geeft 0,38, 0,88 en 0,18, dus het voorspelt alleen voor de leerling met 7 uur leren dat die slaagt. Voor veel leerlingen tegelijk is elke feature een array, en `_w[0] * _hours + _w[1] * _sleep` rekent elk inproduct in één regel uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_boundary():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Op de lijn
    """)
    return


@app.cell(hide_code=True)
def choice_boundary_cell():
    choice_boundary = L.choice(
        "on-boundary",
        question="Het model heeft de weights 1 voor de uren leren en 0,5 voor de uren slaap, en de bias term −8. Welke leerling ligt precies op de decision boundary?",
        options={
            "five-eight": "5 uur leren, 8 uur slaap",
            "eight-four": "8 uur leren, 4 uur slaap",
            "right": "4 uur leren, 8 uur slaap",
            "two-six": "2 uur leren, 6 uur slaap",
        },
        correct="right",
        why={
            "five-eight": "Dat is de leerling uit het uitgewerkte voorbeeld: een logit van 1. Wat is de logit op de decision boundary? Zie *Het idee*.",
            "eight-four": "Reken de logit uit: 8 + 2 − 8 = 2. Wat is de logit op de decision boundary? Zie *Het idee*.",
            "two-six": "Reken de logit uit: 2 + 3 − 8 = −3. Aan welke kant van de lijn is dat? Zie *Het idee*.",
            "right": "1 × 4 + 0,5 × 8 − 8 = 0: de logit is 0, dus de kans is precies 0,5.",
        },
    )
    choice_boundary
    return (choice_boundary,)


@app.cell(hide_code=True)
def text_exercise_probability():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De kans voor één voorbeeld

        Maak `probability(x, w, b)` af: hij krijgt de features `x` van één voorbeeld, de weights `w` en de bias term `b`, en geeft de kans terug die logistische regressie geeft. Vervang de `...`. De check roept `probability([7, 6], [1, 0.5], -8)` aan: dat moet ongeveer `0.881` geven. Hij probeert ook andere weights, en drie features.
        """),
        L.task("Laat `probability` de kans op klasse 1 teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probability():
    box_probability = L.code_box("probability", lines=7)
    box_probability
    return (box_probability,)


@app.cell(hide_code=True)
def check_probability(box_probability):
    result_probability = L.check("probability", box_probability.value)
    result_probability
    return (result_probability,)


@app.cell(hide_code=True)
def hints_probability():
    L.hints(
        "probability",
        [
            "Stap 2 en 3 van het recept: eerst de logit, dan de sigmoid.",
            "De logit is het inproduct van `w` en `x`, met iets erbij. Wat komt erbij, en waar hoort het: binnen of buiten de sigmoid?",
            "`z = np.dot(w, x) + ___` en `return 1 / (1 + np.exp(___))`.",
        ],
        why="`np.dot(w, x) + b` is de logit, en `1 / (1 + np.exp(-z))` de sigmoid daarvan. Voor `[7, 6]`: 7 + 3 − 8 = 2, en $\\sigma(2)$ is ongeveer 0,881.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoeveel leren is genoeg?

        Een leerling sliep 7 uur. Hoeveel uur leren is er nodig om precies op de decision boundary van het model te komen? Met meer voorspelt het model dat de leerling slaagt. Zet het in `hours`. Vervang de `...`. De laatste regel is `hours`, zodat de check het ziet. De check probeert ook andere weights.
        """),
        L.task("Zet de uren leren op de decision boundary bij `sleep` uur slaap in `hours`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours-needed", lines=8)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("hours-needed", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "hours-needed",
        [
            "Op de decision boundary is de logit 0: `w[0] * hours + w[1] * sleep + b = 0`.",
            "Los dat op naar `hours`: haal de andere twee termen naar de andere kant van `= 0`. Wat gebeurt er met hun tekens? En waardoor deel je dan?",
            "`hours = (-b - ___) / ___`.",
        ],
        why="Uit `w[0] * hours + w[1] * sleep + b = 0` volgt `hours = (-b - w[1] * sleep) / w[0]`: (8 − 3,5) / 1 = 4,5.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Veel features: het inproduct](/course/open/math-dot-product?lang=nl) gaf het inproduct plus de bias term een voorspelling; bij logistische regressie geeft het de logit. In [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) maakte de sigmoid van een logit een kans; nu komt de logit uit de features. In deze les waren de weights gegeven; training vindt ze, in een latere les. In de volgende les, *Kansen scoren met de log: log loss*, meet je hoe goed de kansen van een model zijn, de eerste stap op weg naar het trainen ervan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_boundary, result_hours, result_probability):
    L.wrap_up(choice_boundary, result_probability, result_hours)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
