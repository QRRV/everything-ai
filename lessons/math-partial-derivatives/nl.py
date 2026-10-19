# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-partial-derivatives. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow derivative

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-partial-derivatives", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Differentieerregels](/course/open/math-derivative-rules?lang=nl) en [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question="Wat is de afgeleide van $x \\cdot e^x$?",
        options={
            "times": "$e^x$: de afgeleide van $x$, 1, keer die van $e^x$",
            "right": "$e^x + x \\cdot e^x$: met de productregel",
            "power": "$x \\cdot e^{x-1}$: met de machtsregel",
        },
        correct="right",
        why={
            "times": "De afgeleide van een product is niet het product van de afgeleiden. Zie [Differentieerregels](/course/open/math-derivative-rules?lang=nl).",
            "power": "In $e^x$ staat $x$ in de exponent, dus daar geldt de machtsregel niet. Zie [Differentieerregels](/course/open/math-derivative-rules?lang=nl).",
            "right": "De productregel: $1 \\cdot e^x + x \\cdot e^x$.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="De gradient van een loss is ergens $(2;\\ -5)$. In welke richting stijgt de loss het snelst?",
        options={
            "right": "Langs $(2;\\ -5)$: de gradient wijst bergop",
            "down": "Langs $(-2;\\ 5)$: de andere kant op",
            "first": "Alleen langs de eerste parameter: die komt eerst",
        },
        correct="right",
        why={
            "down": "Daar daalt de loss het snelst, de kant die gradient descent op gaat. Zie [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "first": "De gradient wijst in één richting, gemaakt van beide partiële afgeleiden tegelijk. Zie [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "right": "De gradient wijst in de richting waarin de loss het snelst stijgt; de andere kant op is de weg omlaag.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) mat je elke partiële afgeleide met een kleine stap: voor elke parameter de loss één keer extra uitrekenen. Een neural network heeft miljoenen parameters, dus dat zouden miljoenen keren zijn voor één stap van de training. Met de differentieerregels wordt elke partiële afgeleide in plaats daarvan een formule. En in een netwerk bereikt één waarde de loss vaak langs meer dan één pad; dan heeft de kettingregel nog één idee nodig.

    **Een voorbeeld uit het dagelijks leven.** Op een warmere dag verkoopt een ijskraam op twee manieren meer: er lopen meer mensen langs, en elk van hen koopt meer. Een graad warmer verhoogt de verkoop langs beide wegen tegelijk, en de twee effecten tellen op.
    """)
    return


@app.cell(hide_code=True)
def predict_partial():
    guess_partial = L.guess(
        "partial-x",
        question="$f(x, y) = x^2 \\cdot y$. Wat denk je dat $\\frac{\\partial f}{\\partial x}$ is bij $x = 3$, $y = 2$?",
        options={
            "two_x": "6: de machtsregel geeft $2x$",
            "right": "12: $2x$, keer de $y$ ervoor",
            "x_sq": "9: de $x^2$ die $y$ vermenigvuldigt",
        },
        reveal="**12.** Voor de partiële afgeleide naar $x$ houd je $y$ vast: het is gewoon een getal ervoor, zoals de 3 in $3x^2$. Dus $\\frac{\\partial f}{\\partial x} = 2x \\cdot y$, en dat is 12 bij $x = 3$, $y = 2$. De 9 is $x^2$: dat is de partiële afgeleide naar $y$, met $x$ vastgehouden.",
    )
    guess_partial
    return (guess_partial,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De partiële afgeleide $\frac{\partial f}{\partial x}$ van een functie van meer variabelen is de afgeleide naar $x$, met elke andere variabele vastgehouden. Formeel: behandel de andere als getallen, en gebruik de differentieerregels. Voor $f(x, y) = x^2 \cdot y + e^y$ geeft dat $\frac{\partial f}{\partial x} = 2x \cdot y$, omdat $e^y$ voor $x$ een vast getal is met afgeleide 0; en $\frac{\partial f}{\partial y} = x^2 + e^y$. De gradient $\nabla f$ is de lijst van alle partiële afgeleiden, zoals eerder, maar nu als precieze formules.

        Vaak bereikt een waarde de output langs meer dan één pad. Stel $z = u \cdot v$, en zowel $u$ als $v$ hangen af van $t$. Een kleine verandering van $t$ verandert $z$ via $u$ en via $v$, en beide effecten tellen mee. De **multivariabele kettingregel**: volg elk pad van $t$ naar $z$, vermenigvuldig de lokale afgeleiden langs elk pad, en tel de paden op.
        """),
        L.formula(r"[[grad:\frac{dz}{dt}]] = \frac{\partial z}{\partial u} \cdot \frac{du}{dt} + \frac{\partial z}{\partial v} \cdot \frac{dv}{dt}"),
        mo.md(r"""
        De productregel is er één geval van: voor $z = u \cdot v$ is $\frac{\partial z}{\partial u} = v$ en $\frac{\partial z}{\partial v} = u$, dus $\frac{dz}{dt} = v \cdot u' + u \cdot v'$. In een neural network gaat één waarde vaak naar meerdere andere; de afgeleide van de loss naar die waarde is dan een som, met één term per pad.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **lokale afgeleide** | | de afgeleide van een waarde naar een van haar eigen inputs, alleen uit haar eigen formule, zoals $\frac{du}{dt} = 2t$ voor $u = t^2$ | `du_dt` |
        | **multivariabele kettingregel** | | als een waarde de output langs meerdere paden bereikt: vermenigvuldig de lokale afgeleiden langs elk pad, en tel de paden op | `dz_du * du_dt + dz_dv * dv_dt` |
        | $\frac{\partial z}{\partial u}$ | partieel z naar partieel u | de afgeleide van $z$ naar $u$, met de andere inputs van $z$ vastgehouden | `dz_du` |
        | $\frac{dz}{dt}$ | d z naar d t | de afgeleide van $z$ naar $t$, langs elk pad | `dz_dt` |
        """),
        L.watch_out("Langs één pad vermenigvuldig je; tussen paden tel je op. Een pad weglaten mist een deel van de verandering, en de paden met elkaar vermenigvuldigen geeft onzin."),
        mo.md(r"""
        De figuur: $z = u \cdot v$ met $u = t^2$ en $v = 3t + 1$, bij $t = 2$. Bij elke pijl staat zijn <span class="c-grad">lokale afgeleide</span>; onder $t$, $u$ en $v$ staat de afgeleide van $z$ naar die waarde.
        """),
        C.paths_picture({"title": "Twee paden van t naar z", "t": "4 · 7 + 3 · 4 = 40", "u": "∂z/∂u = 7", "v": "∂z/∂v = 4",
                         "z": "",
                         "alt": "Vier vakken: t, met de waarde 2, links; u = t², met 4, en v = 3t + 1, met 7, in het midden; z = u · v, met 28, rechts. Pijlen van t naar u en naar v, en van u en v naar z, elk met zijn lokale afgeleide: du/dt = 4, dv/dt = 3, ∂z/∂u = 7 en ∂z/∂v = 4. Onder t: 4 · 7 + 3 · 4 = 40."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de afgeleiden van een formule met meer variabelen te vinden:

    1. Houd voor een partiële afgeleide elke andere variabele vast: behandel die als een getal, en gebruik de differentieerregels.
    2. Doe dat voor elke variabele, op volgorde: samen zijn ze de gradient.
    3. Bereikt een waarde de output langs meerdere paden, vermenigvuldig dan de lokale afgeleiden langs elk pad.
    4. Tel de paden op.
    5. Controleer met een kleine stap: verander één variabele tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De gradient van $x^2 \cdot y + e^y$ bij $x = 3$, $y = 2$; daarna $z = u \cdot v$ met $u = t^2$ en $v = 3t + 1$, bij $t = 2$, zoals in de figuur. Het commentaar met `#` zegt welke stap van het recept elk deel is. Het pad via $v$ is niet af: voer de cel uit zoals hij is, vervang dan de `0` waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_partials():
    import numpy as np

    def _f(x, y):
        return x ** 2 * y + np.exp(y)

    _x = 3
    _y = 2
    # 1. naar x is y een getal: (x² · y)' = 2x · y, en eʸ geeft 0
    _df_dx = 2 * _x * _y
    # 1. naar y is x een getal: (x² · y)' = x², en (eʸ)' = eʸ
    _df_dy = _x ** 2 + np.exp(_y)
    # 2. samen, op volgorde: de gradient
    print(_df_dx, _df_dy)
    # 5. controleer: één variabele tegelijk, met een kleine stap
    _h = 0.001
    print((_f(_x + _h, _y) - _f(_x, _y)) / _h, (_f(_x, _y + _h) - _f(_x, _y)) / _h)

    # z = u · v, met u = t² en v = 3t + 1, bij t = 2
    _t = 2
    _u = _t ** 2
    _v = 3 * _t + 1
    # 3. langs elk pad vermenigvuldigen: ∂z/∂u = v, keer du/dt = 2t; ∂z/∂v = u, keer dv/dt = 3
    _path_u = _v * (2 * _t)
    _path_v = 0   # het pad via v: vervang de 0 door ∂z/∂v keer dv/dt
    # 4. tel de paden op
    print(_path_u, _path_v, _path_u + _path_v)
    return


@app.cell(hide_code=True)
def text_after_example():
    _e2 = 7.38905609893065
    mo.md(f"""
    De gradient is $(12;\\ {L.number(9 + _e2, 3).replace(",", "{,}")})$; de kleine stappen geven ongeveer hetzelfde, een beetje ernaast. Voor $z$ geeft het pad via $u$ 28; met de 0 nog op zijn plek is de som ook 28, te weinig. Het pad via $v$ geeft 12: samen 40. Controleer het door $z$ uit te schrijven: $z = t^2 \\cdot (3t + 1) = 3t^3 + t^2$, met afgeleide $9t^2 + 2t$, en dat is 40 bij $t = 2$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_partial():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Houd x vast
    """)
    return


@app.cell(hide_code=True)
def choice_partial_cell():
    choice_partial = L.choice(
        "partial-y",
        question="$f(x, y) = x \\cdot y + y^2$. Wat is $\\frac{\\partial f}{\\partial y}$ bij $x = 2$, $y = 3$?",
        options={
            "square": "6: de afgeleide van $y^2$",
            "value": "15: de waarde van $f$ daar",
            "right": "8: $x + 2y$, uit beide delen",
        },
        correct="right",
        why={
            "square": "Dat is alleen het tweede deel. In het eerste deel, $x \\cdot y$, zit ook $y$: met $x$ vastgehouden is de afgeleide daarvan naar $y$ gelijk aan $x$.",
            "value": "$2 \\cdot 3 + 3^2 = 15$ is de waarde van $f$, niet de helling.",
            "right": "Naar $y$ is $x$ een getal: $(x \\cdot y)' = x = 2$ en $(y^2)' = 2y = 6$. Samen 8.",
        },
    )
    choice_partial
    return (choice_partial,)


@app.cell(hide_code=True)
def text_exercise_gradient():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De gradient als formules

        `grad_f(x, y)` moet de gradient geven van `f(x, y)`, en dat is $x \cdot y^2 + \log(x)$, als lijst: eerst de partiële afgeleide naar `x`, dan naar `y`. Schrijf de body van de functie, met stap 1 en 2 van het recept. De check roept `grad_f(2, 3)` aan, en dat moet `[9.5, 12]` geven; hij probeert ook andere punten, met `x` boven 0.
        """),
        L.task("Laat `grad_f` de twee precieze partiële afgeleiden van `f` geven, in een lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradient():
    box_gradient = L.code_box("gradient-formula", lines=11)
    box_gradient
    return (box_gradient,)


@app.cell(hide_code=True)
def check_gradient(box_gradient):
    result_gradient = L.check("gradient-formula", box_gradient.value)
    result_gradient
    return (result_gradient,)


@app.cell(hide_code=True)
def hints_gradient():
    L.hints(
        "gradient-formula",
        [
            "Stap 1 van het recept, één keer voor `x` en één keer voor `y`: welke variabele is telkens een vast getal?",
            "Naar `x` is `y ** 2` een getal vóór `x`, en `np.log(x)` heeft zijn eigen regel. Naar `y` is `x` een getal vóór `y ** 2`, en `np.log(x)` is een vast getal.",
        ],
        why="Bij $x = 2$, $y = 3$: naar $x$ is het $y^2 + \\frac{1}{x} = 9 + 0{,}5 = 9{,}5$; naar $y$ is het $2xy = 12$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_paths():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Twee paden

        `dz_dt(t)` moet de afgeleide geven van $z = u \cdot v$ naar $t$, met $u = t^2$ en $v = e^{2t}$. Schrijf de body van de functie, met stap 3 en 4 van het recept: de lokale afgeleiden langs elk pad, en de twee paden opgeteld. De check roept `dz_dt(1)` aan, en dat moet ongeveer `29.556` geven; hij probeert ook andere waarden van `t`.
        """),
        L.task("Laat `dz_dt` de twee paden van `t` naar `z` optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paths():
    box_paths = L.code_box("two-paths", lines=11)
    box_paths
    return (box_paths,)


@app.cell(hide_code=True)
def check_paths(box_paths):
    result_paths = L.check("two-paths", box_paths.value)
    result_paths
    return (result_paths,)


@app.cell(hide_code=True)
def hints_paths():
    L.hints(
        "two-paths",
        [
            "Stap 3 en 4 van het recept: `t` bereikt `z` via `u` en via `v`.",
            "`v` is $e$ tot de macht $2t$: met de kettingregel is de afgeleide de buitenste afgeleide bij de binnenste waarde, keer de afgeleide van $2t$. Elk pad is daarna twee lokale afgeleiden vermenigvuldigd.",
        ],
        why="Bij $t = 1$: $u = 1$ en $v = e^2 \\approx 7{,}389$. Het pad via $u$ geeft $7{,}389 \\cdot 2$, het pad via $v$ geeft $1 \\cdot 2 \\cdot 7{,}389$: samen ongeveer 29,556.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De partiële afgeleiden die je mat in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) zijn nu precieze formules, met de regels uit [Differentieerregels](/course/open/math-derivative-rules?lang=nl) en de andere variabelen vastgehouden. En als een waarde de output langs meerdere paden bereikt, telt de multivariabele kettingregel ze op. In de volgende les, *Gradients van matrixuitdrukkingen: volg de shapes*, worden de variabelen hele vectoren en matrices tegelijk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_partial, result_gradient, result_paths):
    L.wrap_up(choice_partial, result_gradient, result_paths)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
