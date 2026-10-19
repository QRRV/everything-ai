# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-chain-rule. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow residual

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-chain-rule", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_square():
    quick_square = L.choice(
        "quick-square",
        question="Wat is de helling van $x^2$ bij $x = 5$?",
        options={
            "right": "10",
            "value": "25",
            "x": "5",
        },
        correct="right",
        why={
            "value": "Dat is de waarde van $x^2$ bij 5, niet de helling. Wat geeft de machtsregel? Zie *Het idee* in [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl).",
            "x": "Welk getal komt naar voren? Zie de machtsregel in [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl).",
            "right": "De machtsregel geeft de hellingsformule $2x$, en 2 × 5 = 10.",
        },
    )
    quick_square
    return (quick_square,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="Wat is de helling van de lijn $-4x$?",
        options={
            "line": "$-4x$",
            "right": "$-4$, bij elke $x$",
            "positive": "4, bij elke $x$",
        },
        correct="right",
        why={
            "line": "Dat is de lijn zelf. Wat is de helling van een lijn $a \\cdot x$? Zie de tabel in [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl).",
            "positive": "De lijn daalt als $x$ stijgt. Welk teken krijgt de helling dan? Zie de tabel in [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl).",
            "right": "Een lijn $a \\cdot x$ heeft overal de helling $a$: hier −4.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De machtsregel geeft de helling van $x^2$ en $x^3$. Maar een loss is geen gewone macht van $w$. De loss van één leerling, met het model $\hat{y} = w \cdot x$, is $(y - w \cdot x)^2$: een kwadraat van iets waar $w$ in zit. Je kunt de helling meten, maar dat kost twee berekeningen, is nooit helemaal exact, en een groot model zou het voor miljoenen parameters moeten doen. De kettingregel geeft de exacte helling van zo'n functie, stuk voor stuk.

    **Een voorbeeld uit het dagelijks leven.** Op een fiets draait één rondje van de trappers het achterwiel 2,5 keer rond, en één rondje van het wiel brengt je 2 meter verder. Eén rondje trappen brengt je dus 2,5 × 2 = 5 meter verder: je vermenigvuldigt de twee verhoudingen.
    """)
    return


@app.cell(hide_code=True)
def predict_rates():
    guess_rates = L.guess(
        "rates",
        question="$u = 3x$, en $y = 2u$. Hoeveel denk je dat $y$ omhooggaat als $x$ 1 omhooggaat?",
        options={
            "outer": "2",
            "added": "5",
            "multiplied": "6",
        },
        reveal="**6.** Als $x$ 1 omhooggaat, gaat $u$ 3 omhoog. Bij elke stap van 1 in $u$ gaat $y$ 2 omhoog, dus $y$ gaat 3 × 2 = 6 omhoog: je vermenigvuldigt de twee snelheden. Controleer het: $y = 2 \\cdot 3x = 6x$.",
    )
    guess_rates
    return (guess_rates,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Veel functies zijn in stappen gebouwd: een binnenste stap $u = g(x)$, en een buitenste stap $y = f(u)$. Is $g'$ de helling van de binnenste stap en $f'$ die van de buitenste, dan verandert $y$ $f' \cdot g'$ keer zo snel als $x$. Dat is de kettingregel. Schrijf je hellingen als $\frac{dy}{dx}$, dan zie je welke verandering tegen welke gemeten wordt:
        """),
        L.formula(r"[[grad:\frac{dy}{dx}]] = [[grad:\frac{dy}{du}]] \cdot [[grad:\frac{du}{dx}]]"),
        mo.md(r"""
        Bij de loss van één leerling is de binnenste stap het <span class="c-residual">residu</span> $r = y - w \cdot x$. Hier liggen $x$ en $y$ vast, de getallen van de leerling, en verandert alleen $w$: gaat $w$ 1 omhoog, dan stijgt de voorspelling $w \cdot x$ met $x$ en daalt $r$ met $x$. Zo is $r$ een lijn in $w$ met helling $-x$. De buitenste stap is $r^2$, met helling $2r$ volgens de machtsregel. Vermenigvuldigd:
        """),
        L.formula(r"[[grad:\frac{d\mathcal{L}}{dw}]] = [[grad:\frac{d\mathcal{L}}{dr}]] \cdot [[grad:\frac{dr}{dw}]] = 2[[residual:r]] \cdot (-x)"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **kettingregel** | | de helling van een functie die in stappen is gebouwd, is het product van de hellingen van de stappen | `dloss_dr * dr_dw` |
        | $\frac{dy}{dx}$ | d y naar d x | de helling van $y$ naar $x$: hoe snel $y$ verandert als $x$ verandert. $\frac{d\mathcal{L}}{dw}$ is de $\mathcal{L}'(w)$ uit [De afgeleide als helling](/course/open/math-slope?lang=nl) | `dy_dx` |
        """),
        L.watch_out("Reken de buitenste helling uit bij de binnenste waarde, niet bij $x$. Bij $(2x)^2$ en $x = 3$ is de binnenste waarde $u = 6$, dus de buitenste helling $2u$ is 12, niet 6."),
        mo.md(r"""
        De leerling leerde 4 uur en kreeg een 7. Beweeg de slider, of sleep over de grafiek: de kromme is de loss $(7 - 4w)^2$, het <span class="c-param">punt</span> staat bij de $w$ die je kiest, en de <span class="c-grad">groene lijn</span> heeft de helling die de kettingregel daar geeft.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_w_cell():
    def _frame(w):
        r, outer, inner, slope = C.parts(w)
        return (C.picture(w, {"title": "De loss van één leerling", "loss": "loss",
                              "alt": f"De loss (7 − 4w)² voor w van 0 tot 3,5, een dal met het laagste punt bij w = 1,75. Bij w = {L.number(w, 2)} staat een punt, met een korte groene lijn met de helling daar: {L.number(slope, 0)}."}),
                f"""Bij <i>w</i> = {L.number(w, 2)}: het <span class="c-residual">residu</span> is <i>r</i> = 7 − 4 × {L.number(w, 2)} = <span class="c-residual ea-num">{L.number(r, 0)}</span>. De buitenste helling 2<i>r</i> is <span class="c-grad ea-num">{L.number(outer, 0)}</span>, de binnenste helling is <span class="c-grad ea-num">{L.number(inner, 0)}</span>, en hun product is de <span class="c-grad">helling</span> van de loss: <span class="c-grad ea-num">{L.number(slope, 0)}</span>.""")

    explore_w = L.explore("chain-w", "w", _frame, start=1, low=0, high=3.5, step=0.25, role="param", task="Zoek de $w$ waarbij de helling van de loss 0 is. Wat is het residu daar?")
    explore_w
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Waar het residu 0 is, bij $w$ = 1,75, is de buitenste helling $2r$ gelijk aan 0, dus is de helling van de loss, $2r \cdot (-x)$, ook 0: het laagste punt.

    ### Recept

    Zo vind je een helling met de kettingregel:

    1. Splits de functie in een binnenste en een buitenste stap, zoals $r = y - w \cdot x$ en $r^2$.
    2. Zoek de helling van elke stap met de regels die je kent: $-x$ voor de lijn, $2r$ voor het kwadraat.
    3. Vul de getallen in: eerst de binnenste waarde, dan de buitenste helling bij die waarde.
    4. Vermenigvuldig de twee hellingen. Controleer het eventueel door te meten met een piepkleine $h$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De helling van de loss van de leerling, 4 uur leren en een 7, bij $w$ = 1,5. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_chain():
    _x, _y = 4, 7
    _w = 1.5
    # 1. de binnenste stap, het residu, en de buitenste stap, het kwadraat
    _r = _y - _w * _x
    _loss = _r ** 2
    # 2. en 3. de helling van elke stap, bij deze getallen
    _dloss_dr = 2 * _r
    _dr_dw = -_x
    # 4. vermenigvuldig de hellingen
    _slope = _dloss_dr * _dr_dw
    print(_r, _dloss_dr, _dr_dw, _slope)
    # en controleer door te meten, met een piepkleine h
    _h = 0.001
    print(((_y - (_w + _h) * _x) ** 2 - _loss) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het residu is 7 − 1,5 × 4 = 1, dus de buitenste helling is 2 en de binnenste helling −4: de kettingregel geeft −8. Meten geeft ongeveer −7,984: dichtbij, dus de kettingregel doorstaat de controle. De helling is negatief, dus een grotere $w$ maakt de loss kleiner, zoals de grafiek van de loss laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nested():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een kwadraat van een lijn
    """)
    return


@app.cell(hide_code=True)
def choice_nested_cell():
    choice_nested = L.choice(
        "nested",
        question="Wat is de helling van $(3x)^2$ bij $x = 1$?",
        options={
            "outer": "6",
            "inner": "3",
            "right": "18",
        },
        correct="right",
        why={
            "outer": "Dat is alleen de buitenste helling, of de buitenste helling uitgerekend bij $x$ in plaats van bij $u$. Zie de formule en de waarschuwing in *Het idee*.",
            "inner": "Dat is alleen de binnenste helling, van $3x$. Wat is de buitenste helling bij $u = 3$? Zie het recept.",
            "right": "De binnenste stap $u = 3x$ heeft helling 3, en $u = 3$. De buitenste stap $u^2$ heeft daar helling $2u = 6$. Vermenigvuldigd: 18. Controleer het: $(3x)^2$ is 9 keer $x^2$, dus de helling is 9 × $2x$ = $18x$.",
        },
    )
    choice_nested
    return (choice_nested,)


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De helling van de loss als functie

        Maak `loss_slope(w, x, y)` af. Hij geeft de helling van de loss $(y - w \cdot x)^2$ naar $w$ terug, voor één voorbeeld met feature `x` en label `y`, uitgerekend met de kettingregel. Vervang de `...`. De check roept `loss_slope(1.5, 4, 7)` aan: de leerling uit het uitgewerkte voorbeeld, en dat moet `-8.0` geven. Hij probeert ook andere voorbeelden.
        """),
        L.task("Laat `loss_slope` de helling van de loss teruggeven, met de kettingregel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slope():
    box_slope = L.code_box("loss-slope", lines=5)
    box_slope
    return (box_slope,)


@app.cell(hide_code=True)
def check_slope(box_slope):
    result_slope = L.check("loss-slope", box_slope.value)
    result_slope
    return (result_slope,)


@app.cell(hide_code=True)
def hints_slope():
    L.hints(
        "loss-slope",
        [
            "Stap 1 en 2 van het recept: de binnenste stap is het residu, de buitenste stap het kwadraat ervan. Wat is de helling van elke stap?",
            "Reken eerst `r` uit. De buitenste helling gebruikt `r`; de binnenste helling is de helling van de lijn $y - w \\cdot x$ in $w$.",
            "`r = y - w * x`, en dan `return 2 * r * ___`.",
        ],
        why="`r = y - w * x` is het residu. De buitenste stap $r^2$ heeft helling `2 * r`, de binnenste stap helling `-x`, dus `return 2 * r * -x`. Voor de leerling: 2 × 1 × (−4) = −8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De helling naar b

        Het model heeft nu een bias term: $\hat{y} = w \cdot x + b$, dus het residu is $r = y - (w \cdot x + b)$. Het vak heeft de leerling, $w$, $b$ en $r$. Zet de helling van de loss $r^2$ naar $b$ in `slope_b`, met de kettingregel. Vervang de `...`. De laatste regel is `slope_b`, zodat de check hem ziet. De check probeert ook een andere $b$.
        """),
        L.task("Zet de helling van de loss naar `b` in `slope_b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("slope-b", lines=7)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("slope-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "slope-b",
        [
            "De buitenste stap is dezelfde als eerst: $r^2$, met helling $2r$. Alleen de binnenste stap verandert.",
            "Als $b$ 1 omhooggaat en de rest hetzelfde blijft, hoeveel verandert $r = y - (w \\cdot x + b)$ dan?",
            "`slope_b = 2 * r * ___`.",
        ],
        why="Als $b$ 1 omhooggaat, gaat de voorspelling 1 omhoog en $r$ 1 omlaag: de binnenste helling is −1. Dus `slope_b = 2 * r * -1`, en dat is −1,5 voor deze leerling: $r$ = 0,75.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hellingsformules die je kunt vertrouwen](/course/open/math-slope-formulas?lang=nl) vond je hellingen van machten. De kettingregel neemt een functie die in stappen is gebouwd, zoekt de helling van elke stap, en vermenigvuldigt ze: zo krijg je de exacte helling van een loss als $(y - w \cdot x)^2$ zonder te meten. Later in de course is een neural network een lange ketting van zulke stappen, en als je deze regel steeds opnieuw toepast, krijg je de helling voor elke parameter. In de volgende les, *Lineaire regressie met veel features*, gebruik je deze hellingen om een model met veel features tegelijk te fitten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nested, result_b, result_slope):
    L.wrap_up(choice_nested, result_slope, result_b)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
