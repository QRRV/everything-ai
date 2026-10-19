# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-mse-loss. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow residual, residuals, vectorization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-mse-loss", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl) en [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een model voorspelde dat een rit 52 minuten zou duren. Hij duurde 47 minuten. Wat is het residu?",
        options={
            "flipped": "5",
            "right": "−5",
            "label": "47",
        },
        correct="right",
        why={
            "flipped": "Dat is de voorspelling min het label. In welke volgorde trekt een residu af? Zie de formule in [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl).",
            "label": "Dat is het label zelf. Een residu zegt hoe ver de voorspelling ernaast zit. Zie de formule in [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl).",
            "right": "Het label min de voorspelling: 47 − 52 = −5. Negatief: het model voorspelde 5 minuten te lang.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_square():
    quick_square = L.choice(
        "quick-square",
        question="`a = np.array([1, -2, 3])`. Wat geeft `a ** 2`?",
        options={
            "right": "`array([1, 4, 9])`",
            "sign": "`array([1, -4, 9])`",
            "times": "`array([2, -4, 6])`",
        },
        correct="right",
        why={
            "sign": "Wat is −2 keer −2? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "times": "Dat is `a * 2`. Wat doet `**` met een getal? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "right": "`** 2` kwadrateert elk getal van de array: 1, 4 en 9. Min keer min is plus, dus een kwadraat is nooit negatief.",
        },
    )
    quick_square
    return (quick_square,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je model voorspelt de prijs van vijf huizen, en bij elk huis zit het er een stukje naast: vijf <span class="c-residual">residuen</span>. Nu heb je een tweede model, met vijf eigen residuen. Welk model is beter? Met tien losse getallen kun je dat niet zeggen. Je wilt **één getal** per model.

    **Een voorbeeld uit het dagelijks leven.** Twee weermannen voorspellen drie dagen lang de temperatuur. Zo ver zitten ze ernaast, in graden:

    | | maandag | dinsdag | woensdag |
    |---|---|---|---|
    | Weerman A | +3 | −3 | 0 |
    | Weerman B | +1 | +1 | +1 |
    """)
    return


@app.cell(hide_code=True)
def predict_forecasters():
    guess_forecaster = L.guess(
        "forecasters",
        question="Welke weerman denk je dat beter voorspelt?",
        options={
            "a": "A",
            "b": "B",
            "same": "Even goed",
        },
        reveal="**Neem je het gewone gemiddelde van de residuen, dan wint A: 0 tegen 1.** Maar A zat op twee van de drie dagen 3 graden ernaast, en B nooit meer dan 1. In een gewoon gemiddelde vallen plus en min tegen elkaar weg. Deze les maakt één getal waarin dat niet gebeurt.",
    )
    guess_forecaster
    return (guess_forecaster,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De loss is één getal dat zegt hoe ver een model er over alle voorbeelden samen naast zit: lager is beter. Een veelgebruikte loss is de mean squared error: kwadrateer elk residu, en neem het gemiddelde.
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} [[residual:(y_i - \hat{y}_i)]]^2"),
        mo.md(r"""
        In woorden: voor elk voorbeeld $i$ het <span class="c-residual">residu</span> $y_i - \hat{y}_i$, in het kwadraat; tel die op voor alle $n$ voorbeelden, en deel door $n$. Dat is de <span class="c-loss">loss</span> $\mathcal{L}$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **loss**, $\mathcal{L}$ | L, de loss | één getal dat zegt hoe ver een model er over alle voorbeelden samen naast zit; lager is beter | `loss` |
        | **mean squared error** (MSE) | | een loss: het gemiddelde van de gekwadrateerde residuen | `mse` |
        | $n$ | n | het aantal voorbeelden | `n`, `len(y)` |
        | $y_i$, $\hat{y}_i$ | y i, y-dakje i | het label en de voorspelling van voorbeeld $i$ | `y[i]`, `y_pred[i]` |
        | $\sum_{i=1}^{n}$ | som over i van 1 tot en met n | tel op wat erachter staat, voor elke $i$ van 1 tot en met $n$ | `.sum()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_squares():
    mo.vstack([
        mo.Html(C.picture({"title": "De gekwadrateerde residuen", "x": "oppervlakte (m²)", "y": "prijs (duizend euro)",
                           "alt": "Vijf huizen als punten en de lijn van een model. Van de lijn naar elk punt loopt een verticaal streepje, het residu, en op elk streepje staat een vierkant met dat streepje als zijde. Het vierkant van het huis van 95 vierkante meter, met een residu van 41, is veel groter dan alle andere."})),
        mo.md(r"""
        Vijf huizen en de <span class="c-pred">lijn</span> van een model, met elk <span class="c-residual">residu</span> als streepje. Op elk streepje staat een <span class="c-loss">vierkant</span> met dat streepje als zijde: de oppervlakte ervan is het residu in het kwadraat. De mean squared error is de gemiddelde oppervlakte van de vierkanten. Het huis van 95 vierkante meter heeft een residu van 41, en een vierkant van 1681: meer dan de andere vier vierkanten samen, 901.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_why_square():
    mo.vstack([
        mo.md(r"""
        Kwadrateren doet twee dingen. Een kwadraat is nooit negatief, dus plus en min vallen niet meer tegen elkaar weg. En een grote misser telt extra zwaar: een residu van 4 geeft 16, vier keer zoveel als een residu van 2, dat 4 geeft.

        Terug naar de weermannen:

        - A: $(3^2 + (-3)^2 + 0^2) / 3 = 18 / 3 = 6$
        - B: $(1^2 + 1^2 + 1^2) / 3 = 1$

        Met de mean squared error wint B.
        """),
        L.note("**Waarom laat je het minteken niet gewoon weg?** Dat kan: het gemiddelde van de residuen zonder hun teken is ook een loss. Het kwadraat wordt vaker gebruikt, omdat je er later makkelijker mee rekent. Waarom, zie je zodra een model zelf leert."),
        L.watch_out("De *error* in mean squared error betekent hier het residu: hoe ver een voorspelling ernaast zit. Het heeft niets met een fout in je code te maken."),
        L.watch_out("Sommige boeken noemen het gekwadrateerde residu van één voorbeeld de loss, en het gemiddelde over alle voorbeelden de *cost*. In deze course heten ze allebei de loss."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo reken je de mean squared error van een model uit:

    1. Reken het residu van elk voorbeeld uit: $y_i - \hat{y}_i$.
    2. Kwadrateer elk residu.
    3. Neem het gemiddelde: tel de kwadraten op en deel door het aantal voorbeelden, $n$.

    De formule telt de voorbeelden vanaf 1, Python vanaf 0: $y_1$ is `y[0]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie huizen, met prijzen in duizenden euro's. Eén residu is positief, één negatief en één 0. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_houses():
    _y = [300, 250, 400]       # de echte prijzen
    _y_pred = [280, 260, 400]  # wat het model voorspelde
    # 1. het residu van elk huis
    _residuals = []
    for _i in range(len(_y)):
        _residuals.append(_y[_i] - _y_pred[_i])
    print(_residuals)
    # 2. kwadrateer elk residu
    _squares = []
    for _r in _residuals:
        _squares.append(_r ** 2)
    print(_squares)
    # 3. het gemiddelde: tel op, en deel door het aantal huizen
    _total = 0
    for _s in _squares:
        _total = _total + _s
    _mse = _total / len(_squares)
    print(_mse)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    De residuen zijn 20, −10 en 0; in het kwadraat 400, 100 en 0; samen 500, en gedeeld door 3 huizen 166,67. Dat getal is groot door de kwadraten. Op zichzelf zegt het weinig; naast de loss van een ander model op dezelfde huizen zegt het welke van de twee beter is.

    **Hetzelfde in NumPy.** Met arrays werkt elke stap van het recept op alle huizen tegelijk, zonder loop:

    | Stap | Gewone Python | NumPy |
    |---|---|---|
    | 1 | een loop met `_residuals.append(_y[_i] - _y_pred[_i])` | `_y - _y_pred` |
    | 2 | een loop met `_squares.append(_r ** 2)` | `_residuals ** 2` |
    | 3 | een loop die optelt, dan `/ len(_squares)` | `_squares.mean()` |
    """)
    return


@app.cell
def example_houses_numpy():
    import numpy as np

    _y = np.array([300, 250, 400])
    _y_pred = np.array([280, 260, 400])
    _residuals = _y - _y_pred    # 1. het residu van elk huis
    _squares = _residuals ** 2   # 2. kwadrateer
    _mse = _squares.mean()       # 3. het gemiddelde
    print(_mse)
    return


@app.cell(hide_code=True)
def text_exercise_temps():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De loss van een weerman

        Een weerman voorspelde vier dagen lang de temperatuur. Het vak heeft de echte temperaturen als de lijst `temps_y`, en de voorspellingen als `temps_y_pred`. Reken de mean squared error van de weerman uit, en zet hem in `mse_temps`. Vervang de `...`, met gewone Python of met NumPy. De laatste regel is `mse_temps`, zodat de check het ziet. De check probeert je code ook op twee andere dagen.
        """),
        L.task("Zet de mean squared error van de vier dagen in `mse_temps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_temps():
    box_temps = L.code_box("mse-temps", lines=14)
    box_temps
    return (box_temps,)


@app.cell(hide_code=True)
def check_temps(box_temps):
    result_temps = L.check("mse-temps", box_temps.value)
    result_temps
    return (result_temps,)


@app.cell(hide_code=True)
def hints_temps():
    L.hints(
        "mse-temps",
        [
            "Stap 1 van het recept: eerst het residu van elke dag, met een loop, net als bij de drie huizen.",
            "Het zijn vier dagen; `for i in range(len(temps_y))` loopt ze allemaal langs. Dan de kwadraten, dan hun gemiddelde.",
            "`residuals.append(temps_y[i] - ___)`.",
        ],
        why="De residuen zijn −3, 0, 1 en −4. Hun kwadraten, 9, 0, 1 en 16, tellen op tot 26, en 26 gedeeld door 4 dagen is 6,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De mean squared error als functie

        Maak `mse(y, y_pred)` af. Hij krijgt de labels en de voorspellingen van een paar voorbeelden, als twee lijsten van dezelfde lengte, en geeft één getal terug: de mean squared error. De eerste twee regels van de functie maken arrays van de lijsten. Vervang de `...`. De check roept `mse([3, 5, 2, 6], [4, 5, 0, 6])` aan: dat moet `1.25` geven. De check probeert ook andere voorbeelden.
        """),
        L.task("Laat `mse` de mean squared error van `y` en `y_pred` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mse():
    box_mse = L.code_box("mse", lines=9)
    box_mse
    return (box_mse,)


@app.cell(hide_code=True)
def check_mse(box_mse):
    result_mse = L.check("mse", box_mse.value)
    result_mse
    return (result_mse,)


@app.cell(hide_code=True)
def hints_mse():
    L.hints(
        "mse",
        [
            "Het recept heeft drie stappen, en in NumPy is elke stap één bewerking op de hele array.",
            "Stap 1 trekt de twee arrays van elkaar af, en stap 2 kwadrateert met `** 2`. Welke methode uit [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) doet stap 3?",
            "`return ((y - y_pred) ** 2).___()`.",
        ],
        why="`y - y_pred` geeft de residuen, `** 2` kwadrateert elk residu, en `.mean()` neemt hun gemiddelde. Voor het voorbeeld: residuen −1, 0, 2 en 0, kwadraten 1, 0, 4 en 0, gemiddelde 5 / 4 = 1,25. Of je `y - y_pred` of `y_pred - y` neemt, maakt na het kwadrateren niet uit.",
    )
    return


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Oefening 3. Waarom kwadrateren?
    """)
    return


@app.cell(hide_code=True)
def explain_square_cell():
    explain_square = L.explain(
        "why-square",
        question="Waarom kwadrateert de mean squared error de residuen, in plaats van gewoon hun gemiddelde te nemen? Schrijf één of twee zinnen.",
        model_answer="In het gewone gemiddelde vallen positieve en negatieve residuen tegen elkaar weg, dus een model dat naar twee kanten ver ernaast zit, kan toch 0 krijgen. Een kwadraat is nooit negatief, dus elke misser telt mee, en een grote misser telt extra zwaar.",
        points=[
            "in een gewoon gemiddelde vallen plus en min tegen elkaar weg",
            "een kwadraat is nooit negatief, dus elke misser telt mee",
            "een grote misser telt extra zwaar",
        ],
    )
    explain_square
    return (explain_square,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl) rekende je één residu per voorbeeld uit; de loss voegt ze allemaal samen tot één getal, zodat je hele modellen kunt vergelijken. Met arrays, zoals in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl), is elke stap van het recept één bewerking, en `.mean()`, uit [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl), neemt het gemiddelde. In de volgende les, *Een lijn passen met de hand*, stel je de parameters zelf in en kijk je naar de loss: de beste lijn is die met de laagste loss.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(explain_square, result_mse, result_temps):
    L.wrap_up(result_temps, result_mse, explain_square)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
