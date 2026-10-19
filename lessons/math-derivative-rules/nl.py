# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-derivative-rules. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-derivative-rules", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) en [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$y = u^3$, met $u = 2x$. Wat is de helling van $y$ bij $x = 1$?",
        options={
            "outer": "12: de helling van $u^3$ bij $u = 2$",
            "right": "24: de helling van $u^3$ bij $u = 2$, keer 2",
            "atx": "6: de helling van $u^3$ bij 1, keer 2",
        },
        correct="right",
        why={
            "outer": "Dat is alleen de buitenste helling. De kettingregel vermenigvuldigt die met de helling van de binnenste stap, $u = 2x$. Zie [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "atx": "De buitenste helling $3u^2$ reken je uit bij de binnenste waarde $u = 2$, niet bij $x = 1$. Zie [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "right": "De buitenste helling $3u^2$ bij $u = 2$ is 12, en de binnenste stap $u = 2x$ heeft helling 2: 12 · 2 = 24.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Voor welke $x$ geeft $\\log x$ een getal?",
        options={
            "right": "Alleen voor $x$ boven 0: geen macht van $e$ is 0 of minder",
            "all": "Voor elke $x$: elk getal heeft een exponent van $e$ die het geeft",
            "one": "Alleen voor $x$ boven 1: daaronder zou de log negatief zijn",
        },
        correct="right",
        why={
            "all": "$e$ tot welke macht ook is groter dan 0, dus voor 0 en negatieve getallen is er geen exponent die ze geeft. Zie het blok *Let op* in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "one": "Tussen 0 en 1 is de log negatief, maar hij bestaat wel: $e^{-1}$ is ongeveer 0,37, dus $\\log(0{,}37)$ is ongeveer −1. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "Elke macht van $e$ is groter dan 0, dus alleen een getal boven 0 heeft een exponent die het geeft: zijn log.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Met een kleine stap kun je een helling meten, maar alleen ongeveer, en elke helling kost twee keer de functie uitrekenen. De meeste formules van machine learning zijn gebouwd uit een paar stukken: sommen, producten, machten, $e^x$ en $\log$. Voor elk stuk is er een korte regel die de precieze afgeleide geeft, en de kettingregel verbindt de stukken. De libraries die neural networks trainen, werken zo: één regel per stuk, en een kleine stap alleen als controle.

    **Een voorbeeld uit het dagelijks leven.** Een kookboek met duizend recepten gebruikt maar een handvol technieken: koken, bakken, braden, mengen. Ken je die technieken, dan kun je elk recept maken dat uit die technieken bestaat, zonder het eerst uit te proberen.
    """)
    return


@app.cell(hide_code=True)
def predict_product():
    guess_product = L.guess(
        "x-times-x",
        question="De helling van $x$ is 1, bij elke $x$. Wat denk je dat de helling van $x \\cdot x$ is bij $x = 4$?",
        options={
            "one": "1: de hellingen vermenigvuldigen, 1 · 1",
            "right": "8: elke factor draagt bij aan de verandering",
            "nine": "16: de waarde van $x \\cdot x$ daar",
        },
        reveal="**8.** $x \\cdot x$ is $x^2$, en de machtsregel geeft $2x$: 8 bij $x = 4$. De hellingen vermenigvuldigen geeft 1, veel te weinig: als beide factoren groeien, groeit het product door elk van beide. Dat is de productregel: de helling van de eerste factor keer de tweede, plus de eerste keer de helling van de tweede, $1 \\cdot 4 + 4 \\cdot 1 = 8$.",
    )
    guess_product
    return (guess_product,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De afgeleide $f'(x)$ is de precieze helling van een functie $f$ bij elke $x$, zoals in [De afgeleide als helling](/course/open/math-slope?lang=nl). **Differentieerregels** geven de afgeleide van een formule uit de afgeleiden van haar delen. Drie ken je al, geschreven met een accent achter haakjes voor de afgeleide van wat erbinnen staat: de machtsregel, $(x^n)' = n \cdot x^{n-1}$; een getal keer $x$, $(a \cdot x)' = a$; en de kettingregel. Een paar spreken vanzelf: een los getal heeft afgeleide 0, de afgeleide van een som is de som van de afgeleiden, en een getal ervoor blijft ervoor staan, $(3f)' = 3f'$. Drie zijn nieuw:
        """),
        L.formula(r"[[grad:(f \cdot g)']] = [[grad:f']] \cdot g + f \cdot [[grad:g']]"),
        L.formula(r"[[grad:(e^x)']] = e^x \qquad [[grad:(\log x)']] = \frac{1}{x}"),
        mo.md(r"""
        De **productregel**: als beide factoren veranderen, verandert het product door elk van beide, dus heeft de regel twee termen. De afgeleide van $e^x$ is $e^x$ zelf: bij elke $x$ is de helling van $e^x$ gelijk aan zijn hoogte. De afgeleide van $\log x$ is $\frac{1}{x}$: steil dicht bij 0, en steeds vlakker verderop.

        Met de kettingregel werken deze regels ook op een binnenste deel $g(x)$: de buitenste afgeleide bij de binnenste waarde, keer de afgeleide van het binnenste deel. Dus $(e^{g(x)})' = e^{g(x)} \cdot g'(x)$, en $(\log g(x))' = \frac{g'(x)}{g(x)}$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **differentieerregels** | | regels die de afgeleide van een formule geven uit de afgeleiden van haar delen | |
        | **productregel** | | de afgeleide van een product: $(f \cdot g)' = f' \cdot g + f \cdot g'$ | `df * g + f * dg` |
        | $(e^x)' = e^x$ | de afgeleide van e-macht x is e-macht x | de helling van $e^x$ is gelijk aan zijn hoogte | `np.exp(x)` |
        | $(\log x)' = \frac{1}{x}$ | de afgeleide van log x is 1 gedeeld door x | de helling van $\log x$ is 1 gedeeld door $x$ | `1 / x` |
        """),
        L.watch_out("De afgeleide van een product is niet het product van de afgeleiden: $(x \\cdot x)' = 2x$, niet $1 \\cdot 1$. En $e^x$ is geen macht van $x$: $x$ staat in de exponent, dus de machtsregel geldt niet, en $(e^x)'$ is niet $x \\cdot e^{x - 1}$."),
        mo.md(r"""
        De figuur: $e^x$ en $\log x$, elk met de <span class="c-grad">helling die zijn regel geeft</span> in één punt.
        """),
        mo.Html(C.picture({"exp": "eˣ: de helling is de hoogte", "exp_y": "eˣ",
                           "exp_note": f"helling {L.number(2.718281828459045, 2)} bij x = 1",
                           "log": "log x: de helling is 1/x", "log_y": "log x",
                           "log_note": f"helling {L.number(0.5, 1)} bij x = 2",
                           "alt": "Twee grafieken. Links: de kromme van e-macht x van −2 tot 2, met een groene lijn die hem raakt bij x = 1, waar de hoogte en de helling allebei ongeveer 2,72 zijn. Rechts: de kromme van log x van 0 tot 4, met een groene lijn die hem raakt bij x = 2, met helling 0,5."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de afgeleide van een formule te vinden:

    1. Zoek de laatste stap die de formule zet: een som, een product, een macht, $e$ tot de macht, of een log, van een deel.
    2. Gebruik de regel van die stap. Voor een som de som van de afgeleiden; voor een product de productregel; voor een macht, $e$ tot de macht of een log van een binnenste deel de kettingregel: de buitenste afgeleide bij de binnenste waarde, keer de afgeleide van het binnenste deel.
    3. Doe hetzelfde voor elk deel daarbinnen, tot er alleen $x$ en getallen over zijn.
    4. Controleer de formule bij één $x$ met een kleine stap: $(f(x + h) - f(x)) / h$ moet er dicht bij komen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De afgeleide van $x^2 \cdot e^x$ bij $x = 1$, en van $\log(x^2 + 1)$ bij $x = 2$. Het commentaar met `#` zegt welke stap van het recept elk deel is. De tweede afgeleide is niet af: voer de cel uit zoals hij is, maak dan stap 3 af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_rules():
    import numpy as np

    def _f(x):
        return x ** 2 * np.exp(x)

    # 1. de laatste stap van _f is een product: u = x² en v = eˣ
    # 2. de productregel: u' · v + u · v'
    # 3. de delen: u' = 2x met de machtsregel, v' = eˣ
    def _df(x):
        return 2 * x * np.exp(x) + x ** 2 * np.exp(x)

    # 4. controleer bij x = 1 met een kleine stap
    _h = 0.001
    print(_df(1), (_f(1 + _h) - _f(1)) / _h)

    def _k(x):
        return np.log(x ** 2 + 1)

    # 1. de laatste stap van _k is een log, van het binnenste deel x² + 1
    # 2. de kettingregel: de afgeleide van de log bij de binnenste waarde, 1 / (x² + 1), keer de binnenste afgeleide
    # 3. het binnenste deel: (x² + 1)' = 2x. Het deel van de log staat er; vermenigvuldig het met de binnenste afgeleide
    def _dk(x):
        return 1 / (x ** 2 + 1)

    # 4. controleer bij x = 2
    print(_dk(2), (_k(2 + _h) - _k(2)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    _e = 2.718281828459045
    _h = 0.001
    _step_f = ((1 + _h) ** 2 * _e ** (1 + _h) - _e) / _h
    mo.md(f"""
    Met de regels is de afgeleide van $x^2 \\cdot e^x$ bij 1 gelijk aan $2e + e = 3e$, ongeveer {L.number(3 * _e, 3)}; de kleine stap geeft {L.number(_step_f, 3)}, dichtbij maar een beetje ernaast. Voor $\\log(x^2 + 1)$ bij 2 geven de regels precies $\\frac{{1}}{{5}} \\cdot 4 = 0{{,}}8$; zolang stap 3 niet af is, geeft `_dk(2)` alleen het deel van de log, 0,2, en de kleine stap laat zien dat dat te klein is. De regels zijn precies, en kosten elk één keer uitrekenen; de kleine stap controleert ze alleen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_exp():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. e tot de macht 3x
    """)
    return


@app.cell(hide_code=True)
def choice_exp_cell():
    choice_exp = L.choice(
        "exp-3x",
        question="Wat is de afgeleide van $e^{3x}$?",
        options={
            "same": "$e^{3x}$: $e$ tot de macht blijft zichzelf",
            "power": "$3x \\cdot e^{3x - 1}$: de machtsregel",
            "right": "$3e^{3x}$: de kettingregel",
        },
        correct="right",
        why={
            "same": "$(e^x)' = e^x$, maar hier is de exponent $3x$, een binnenste deel: de kettingregel vermenigvuldigt met de afgeleide daarvan, 3.",
            "power": "De machtsregel is voor $x^n$, $x$ tot een vaste macht. In $e^{3x}$ staat $x$ in de exponent, dus geldt de regel voor $e$ tot de macht, met de kettingregel.",
            "right": "De buitenste stap, $e$ tot de macht, geeft $e^{3x}$ bij de binnenste waarde, en de binnenste stap $3x$ heeft afgeleide 3: $3e^{3x}$.",
        },
    )
    choice_exp
    return (choice_exp,)


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De productregel

        `f(x)` is $x^3 \cdot \log(x)$, en `df(x)` moet de afgeleide daarvan geven. Vervang de `...` door de formule, met het recept. De check roept `df(2)` aan, en dat moet ongeveer `12.318` geven; hij probeert ook andere waarden van `x` boven 0.
        """),
        L.task("Laat `df` de precieze afgeleide van `f` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("product", lines=10)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("product", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "product",
        [
            "Stap 1 van het recept: de laatste stap van `f` is een product. Van welke twee delen?",
            "De delen zijn $x^3$ en $\\log(x)$. De machtsregel geeft de afgeleide van het eerste, de regel voor de log die van het tweede; de productregel zet ze samen, in twee termen.",
        ],
        why="Bij $x = 2$: $3 \\cdot 4 \\cdot \\log 2 \\approx 8{,}318$, plus $8 \\cdot \\frac{1}{2} = 4$, samen ongeveer 12,318.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_softplus():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een log van een binnenste deel

        `k(x)` is $\log(1 + e^x)$, en `dk(x)` moet de afgeleide daarvan geven. Vervang de `...` door de formule, met het recept. De check roept `dk(1)` aan, en dat moet ongeveer `0.731` geven; hij probeert ook andere waarden van `x`, ook onder 0.
        """),
        L.task("Laat `dk` de precieze afgeleide van `k` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softplus():
    box_softplus = L.code_box("softplus", lines=10)
    box_softplus
    return (box_softplus,)


@app.cell(hide_code=True)
def check_softplus(box_softplus):
    result_softplus = L.check("softplus", box_softplus.value)
    result_softplus
    return (result_softplus,)


@app.cell(hide_code=True)
def hints_softplus():
    L.hints(
        "softplus",
        [
            "Stap 1 van het recept: de laatste stap van `k` is een log. Van welk binnenste deel?",
            "De kettingregel: de afgeleide van de log bij de binnenste waarde, keer de afgeleide van het binnenste deel. Het binnenste deel is $1 + e^x$; de afgeleide daarvan volgt uit de regel voor een som en die voor $e^x$.",
        ],
        why="Bij $x = 1$ is de binnenste waarde $1 + e \\approx 3{,}718$, dus de afgeleide van de log is daar $\\frac{1}{3{,}718}$; de binnenste afgeleide is $e \\approx 2{,}718$. Samen ongeveer 0,731.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De afgeleide als helling](/course/open/math-slope?lang=nl) mat je een helling met een kleine stap, en in [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl) gaf de machtsregel hem precies. Nu geven een paar regels de precieze afgeleide van elke formule die is gebouwd uit sommen, producten, machten, $e^x$ en $\log$, met de kettingregel uit [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) voor elk binnenste deel. In de volgende les, *Partiële afgeleiden en de gradient, formeel*, krijgt een formule meer dan één input.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_exp, result_product, result_softplus):
    L.wrap_up(choice_exp, result_product, result_softplus)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
