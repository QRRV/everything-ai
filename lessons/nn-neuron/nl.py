# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-neuron. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-neuron", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) en [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logit():
    quick_logit = L.choice(
        "quick-logit",
        question="Logistische regressie heeft de weights $(2, -1)$ en de bias term $0{,}5$. Een voorbeeld heeft de features $(1, 3)$. Ligt de kans die het voorspelt boven of onder 0,5?",
        options={
            "above": "Boven 0,5: de gewogen som plus de bias term is 5,5",
            "right": "Onder 0,5: de gewogen som plus de bias term is −0,5",
            "equal": "Precies 0,5: het voorbeeld ligt op de decision boundary",
        },
        correct="right",
        why={
            "above": "$2 \\cdot 1 + 1 \\cdot 3 + 0{,}5$ zou 5,5 zijn, maar de tweede weight is −1. Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "equal": "Op de decision boundary is de gewogen som plus de bias term precies 0. Hoeveel is die hier? Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "right": "$2 \\cdot 1 - 1 \\cdot 3 + 0{,}5 = -0{,}5$: onder 0, dus de sigmoid geeft minder dan 0,5.",
        },
    )
    quick_logit
    return (quick_logit,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="Wat voorspelt lineaire regressie voor een voorbeeld met de features `x`?",
        options={
            "sigmoid": "De sigmoid van `np.dot(w, x) + b`",
            "class": "1 of 0, wat het dichtst bij ligt",
            "right": "`np.dot(w, x) + b` zelf",
        },
        correct="right",
        why={
            "sigmoid": "Dat is logistische regressie, een kans. Wat voorspelt lineaire regressie, een prijs bijvoorbeeld? Zie [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "class": "Lineaire regressie voorspelt een getal, zoals een prijs. Zie [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "right": "De gewogen som plus de bias term, zonder iets erna: elk getal.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elk neural network in deze course, tot en met het language model aan het eind, is gebouwd uit één klein onderdeel: het neuron. Je kent het al, zonder de naam. De naam komt van de zenuwcellen in de hersenen, die ook signalen verzamelen en er een doorgeven als ze genoeg krijgen; daar houdt de gelijkenis op. Een neuron in een netwerk is een paar regels rekenwerk.

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder gaat niet af bij het eerste sliertje rook. Hij weegt wat zijn sensoren meten, rook en hitte, en piept pas als het totaal boven een ingesteld niveau komt.
    """)
    return


@app.cell(hide_code=True)
def predict_fires():
    guess_fires = L.guess(
        "fires",
        question="Een neuron heeft de weights 1 en 1 en de bias term −1,5. Het geeft 1 als zijn gewogen som plus de bias term boven 0 ligt, en anders 0. Bij welke van de inputs (0, 0), (0, 1), (1, 0) en (1, 1) geeft het volgens jou 1?",
        options={
            "one": "Alleen bij (1, 1)",
            "three": "Bij (0, 1), (1, 0) en (1, 1)",
            "none": "Bij geen van alle",
            "all": "Bij alle vier",
        },
        reveal="**Alleen bij (1, 1).** Daar is de som 1 + 1 − 1,5 = 0,5, boven 0. Met één 1 is hij −0,5, en zonder enen −1,5. Het neuron geeft alleen 1 als beide inputs 1 zijn: het rekent AND uit.",
    )
    guess_fires
    return (guess_fires,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **neuron** vermenigvuldigt elke input met zijn weight, telt alles op met een bias term, en haalt de uitkomst door een **activation function** $f$:
        """),
        L.formula(r"z = [[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]] \qquad [[pred:\hat{y}]] = f(z)"),
        mo.Html(C.diagram({"sum": "gewogen som", "activation": "activation function", "output": "output",
                           "alt": "Een neuron met twee inputs, x₁ en x₂. Van elk gaat een pijl, met zijn weight w₁ of w₂, naar een cirkel Σ die ze optelt met de bias term b. De uitkomst z gaat een vak f in, de activation function, en eruit komt de output ŷ."})),
        mo.md(r"""
        De activation function bepaalt wat het neuron met $z$ doet. Drie ervan, elk voor een model dat je kent:

        | Activation function | $f(z)$ | Het neuron is dan |
        |---|---|---|
        | geen | $z$ | lineaire regressie: elk getal |
        | sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | logistische regressie: een kans |
        | stap | 1 als $z > 0$, anders 0 | ja of nee: het neuron vuurt of niet |

        Logistische regressie is dus één neuron, en lineaire regressie ook. Een neural network is meer dan één neuron: het verbindt er veel, en de output van het ene wordt de input van het volgende; de volgende lessen bouwen dat op. Leren is voor een netwerk wat het voor lineaire regressie was: de weights en bias terms bijstellen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **neuron** | | het kleinste onderdeel van een neural network: een gewogen som van zijn inputs plus een bias term, door een activation function gehaald | `f(np.dot(w, x) + b)` |
        | **activation function** | | de functie waar een neuron zijn gewogen som door haalt, zoals de sigmoid of een stap | `1 / (1 + np.exp(-z))` |
        | $f$ | f | de activation function van een neuron | |
        | **vuurt** | | een neuron met een stap geeft 1 | `z > 0` |
        """),
        L.watch_out("Een neuron in een netwerk lijkt maar losjes op een zenuwcel in de hersenen. Het is rekenwerk, en de course heeft nooit meer biologie nodig dan de naam."),
        mo.md(r"""
        Schuif met de slider: hij zet de bias term $b$ van een neuron met de weights 1 en 1 en een stap. De vier hoeken van het vierkant zijn de inputs van het neuron: een gevulde stip waar het neuron vuurt, een open stip waar het niet vuurt. De <span class="c-pred">lijn</span> is waar $z = 0$: het neuron vuurt aan de kant rechtsboven. De andere grafiek telt voor elke $b$ de hoeken die vuren; sleep eroverheen om $b$ te zetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        b = round(b, 1)
        fires = C.fires(b)
        which = ", ".join(str(c) for c, f in zip(C.CORNERS, fires) if f) or "geen enkele hoek"
        return (C.square(b, {"title": "Welke hoeken vuren?", "count_title": "Hoeken die vuren, voor elke b", "count": "hoeken",
                              "alt": f"De vier hoeken van het vierkant bij b = {L.number(b, 1)}: het neuron vuurt bij {which}. De lijn x₁ + x₂ + b = 0 loopt van linksboven naar rechtsonder, met de vurende kant rechtsboven. Een tweede grafiek telt voor elke b de hoeken die vuren: geen tot en met −2, één tot en met −1, drie tot en met 0, en vier daarboven."}),
                f"""Met <i>b</i> = {L.number(b, 1)} vuurt het neuron bij <span class="c-pred ea-num">{sum(fires)}</span> van de 4 hoeken: {which}.""")

    explore_b = L.explore("bias", "b", _frame, start=-1.5, low=-2.5, high=0.5, step=0.1, role="param", task="Zoek de waarden van $b$ waarbij het neuron bij elke hoek vuurt, behalve bij (0, 0).")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De bias term verschuift de lijn: tot en met −2 vuurt er niets, daarna tot en met −1 alleen (1, 1), daarna tot en met 0 drie hoeken, alle behalve (0, 0), en boven 0 alle vier. De telling staat nooit op 2: (0, 1) en (1, 0) hebben dezelfde $z$, dus ze beginnen tegelijk te vuren. De weights bepalen de richting van de lijn; de bias term bepaalt waar hij ligt.

    ### Recept

    Zo reken je uit wat een neuron geeft voor de inputs `x`:

    1. Vermenigvuldig elke input met zijn weight en tel op: `np.dot(w, x)`.
    2. Tel de bias term erbij op: `z = np.dot(w, x) + b`.
    3. Haal `z` door de activation function: een stap, `1` als `z > 0` en anders `0`; de sigmoid; of helemaal niets.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het neuron van de gok, met de weights 1 en 1 en de bias term −1,5, op alle vier de inputs, met een stap en met de sigmoid als activation function. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_neuron():
    import numpy as np

    _w = np.array([1, 1])
    _b = -1.5
    for _x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # 1. en 2. de gewogen som plus de bias term
        _z = np.dot(_w, _x) + _b
        # 3. een stap, en de sigmoid
        if _z > 0:
            _step = 1
        else:
            _step = 0
        _sigmoid = 1 / (1 + np.exp(-_z))
        print(_x, _z, _step, _sigmoid)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $z$ is −1,5 bij (0, 0), −0,5 bij (0, 1) en (1, 0), en 0,5 bij (1, 1). Met een stap geeft alleen (1, 1) een 1: AND. Met de sigmoid geeft hetzelfde neuron in plaats daarvan kansen: ongeveer 0,18 en 0,38 onder de lijn, en 0,62 erboven. Dat is logistische regressie, met deze weights en deze bias term.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_activation():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke activation function?
    """)
    return


@app.cell(hide_code=True)
def choice_activation_cell():
    choice_activation = L.choice(
        "which-activation",
        question="Een neuron moet de prijs van een huis voorspellen uit de oppervlakte en de afstand tot het station. Welke activation function past?",
        options={
            "sigmoid": "De sigmoid, zodat de prijs tussen 0 en 1 uitkomt",
            "step": "Een stap, zodat het neuron vuurt bij de dure huizen en stil blijft bij de rest",
            "right": "Geen: de gewogen som plus de bias term is de prijs, zoals bij lineaire regressie",
        },
        correct="right",
        why={
            "sigmoid": "Een sigmoid perst alles tussen 0 en 1. Kan een prijs dan 300.000 euro zijn? Zie de tabel in *Het idee*.",
            "step": "Een stap geeft alleen 1 of 0: duur of niet, maar geen prijs. Zie de tabel in *Het idee*.",
            "right": "Een prijs kan elk getal zijn, dus niets mag $z$ samenpersen of afkappen: het neuron is dan lineaire regressie.",
        },
    )
    choice_activation
    return (choice_activation,)


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een neuron met een stap

        Maak `step_neuron(x, w, b)` af: hij krijgt de inputs `x`, de weights `w` en de bias term `b`, en geeft 1 terug als de gewogen som plus de bias term boven 0 ligt, en anders 0. Vervang de twee `...` door stap 2 en 3 van het recept. De check roept `step_neuron([1, 1], [1, 1], -1.5)` aan, het neuron van het uitgewerkte voorbeeld: dat moet `1` geven. Hij probeert ook andere inputs, weights en bias terms, ook met drie inputs, en een geval waarin `z` precies 0 is.
        """),
        L.task("Laat `step_neuron` 1 teruggeven als `np.dot(w, x) + b` boven 0 ligt, en anders 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step-neuron", lines=8)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step-neuron", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step-neuron",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld: eerst `z`, dan de `if`.",
            "`z` is een inproduct plus de bias term. Dan een `if`: welk van de twee getallen komt terug als `z` boven 0 ligt?",
            "`z = np.dot(___, ___) + b`, dan `if z > 0:` met `return ___`, en `else:` met `return ___`.",
        ],
        why="`np.dot(w, x) + b` is $z$: bij `[1, 1]` met de weights 1 en 1 en −1,5 is dat 0,5. Dat ligt boven 0, dus het neuron geeft 1. Precies bij 0 geeft het 0: de stap vuurt alleen boven 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_corner():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Maar één hoek

        Kies de weights `w` en de bias term `b` van een neuron met een stap, zodat het alleen vuurt bij de input (1, 0): de eerste input aan, de tweede uit. Vervang de twee `...`; de loop eronder rekent de output uit voor alle vier de hoeken, in de volgorde (0, 0), (0, 1), (1, 0), (1, 1), dus dat moet `[0, 0, 1, 0]` geven. Laat de loop zoals hij is. Wil je er een paar proberen? Doe dat in het kladblok.
        """),
        L.task("Kies `w` en `b` zo dat het neuron alleen bij (1, 0) vuurt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_corner():
    box_corner = L.code_box("one-corner", lines=13)
    box_corner
    return (box_corner,)


@app.cell(hide_code=True)
def check_corner(box_corner):
    result_corner = L.check("one-corner", box_corner.value)
    result_corner
    return (result_corner,)


@app.cell(hide_code=True)
def hints_corner():
    L.hints(
        "one-corner",
        [
            "Met de weights 1 en 1 liet de slider bij zijn begin, b = −1,5, AND zien. Hier moet de tweede input tegen het neuron in werken. Welk teken heeft zijn weight dan nodig?",
            "Bij (0, 0) is `z` gewoon `b`: ligt dat boven of onder 0? Bij (1, 0) is `z` de eerste weight plus `b`. En bij (1, 1)?",
            "`w = np.array([1, ___])` en `b = ___`: een `b` tussen −1 en 0 werkt.",
        ],
        why="Met `w = np.array([1, -1])` en `b = -0.5` is `z` −0,5 bij (0, 0), −1,5 bij (0, 1), 0,5 bij (1, 0) en −0,5 bij (1, 1): alleen (1, 0) ligt boven 0. Veel andere keuzes werken ook, zolang (1, 0) als enige aan de vurende kant van de lijn ligt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) kende je al een neuron met een sigmoid, zonder de naam, en in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) een zonder activation function. Een neuron trekt één rechte lijn door zijn inputs, en vuurt aan één kant ervan. In de volgende les, *Wat één neuron niet kan leren*, zie je een taak die met geen enkele rechte lijn lukt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_activation, result_corner, result_step):
    L.wrap_up(choice_activation, result_step, result_corner)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
