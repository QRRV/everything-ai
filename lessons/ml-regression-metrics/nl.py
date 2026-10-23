# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-regression-metrics. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow residual

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-regression-metrics", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="Een model voorspelt de prijzen van 8 testhuizen met een MSE van 3500. Altijd de gemiddelde trainingsprijs voorspellen haalt op dezelfde huizen een MSE van 3088. Wat zegt dat?",
        options={
            "good": "Niet veel: een MSE van 3500 is normaal voor huizenprijzen",
            "right": "Het model verslaat de baseline niet, dus het heeft niet laten zien dat het iets van de features leerde",
            "compare": "Niets: een MSE kun je niet met een baseline vergelijken",
        },
        correct="right",
        why={
            "good": "Een MSE heeft geen vaste schaal; wat geeft hem betekenis? Zie *Het idee* in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "compare": "Meet je ze allebei op dezelfde test set met dezelfde metric, dan kun je ze vergelijken. Zie het recept in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "right": "De eenvoudigste regel, die de features negeert, zit dichterbij: 3088 tegen 3500.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="De waarden 1, 3 en 5 zijn even waarschijnlijk. Wat is hun variantie?",
        options={
            "right": "Ongeveer 2,67",
            "std": "Ongeveer 1,63",
            "mean": "3",
        },
        correct="right",
        why={
            "std": "Dat is de standaardafwijking, de wortel uit de variantie. Zie het recept in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "mean": "Dat is de verwachtingswaarde. De variantie gaat over de gekwadrateerde afstanden tot die waarde. Zie het recept in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "De waarden liggen 2, 0 en 2 van 3 af: (4 + 0 + 4) / 3 is ongeveer 2,67.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) had de lijn een MSE van 648 op zijn 8 testhuizen, tegen 3088 voor de baseline: hij leerde iets van de oppervlakte. Maar wie een huis verkoopt, vraagt hoe ver de prijs ernaast zit, en 648 is in duizend euro in het kwadraat, een eenheid waarin niemand denkt. Je wilt een getal in euro's, en een getal zonder eenheid dat zegt hoeveel van de verschillen tussen de prijzen het model verklaart.

    **Een voorbeeld uit het dagelijks leven.** Een weerapp die zegt dat zijn voorspellingen een gemiddelde gekwadrateerde misser van 4 graden in het kwadraat hebben, zegt je weinig. "Meestal zo'n 2 graden ernaast" zegt je wat je kunt verwachten.
    """)
    return


@app.cell(hide_code=True)
def predict_typical():
    guess_typical = L.guess(
        "typical-miss",
        question="De MSE van de lijn op de 8 testhuizen is 648. Hoe ver denk je dat hij ongeveer ernaast zit bij een gewoon testhuis?",
        options={
            "mse": "Ongeveer 648 duizend euro",
            "root": "Ongeveer 25 duizend euro",
            "tiny": "Ongeveer 6 duizend euro",
        },
        reveal="**Ongeveer 25 duizend euro.** De MSE is in duizend euro in het kwadraat. De wortel eruit, de RMSE hieronder, is ongeveer 25,5 duizend euro, en de gemiddelde grootte van de acht residuen, de MAE, ongeveer 22,6.",
    )
    guess_typical
    return (guess_typical,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Drie metrics voor een model dat getallen voorspelt, naast de MSE:

        - De **mean absolute error**, MAE: de gemiddelde grootte van de residuen, zonder hun teken. In de eenheid van het label: gemiddeld zo ver ernaast.
        - De **root mean squared error**, RMSE: de wortel uit de MSE. Ook in de eenheid van het label, maar grote missers tellen zwaarder, omdat elk residu eerst gekwadrateerd wordt. Hij komt nooit onder de MAE; hoe verder erboven, hoe ongelijker de missers.
        - **R²**, de determinatiecoëfficiënt: 1 min de MSE gedeeld door de variantie van de labels. Die variantie, uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl), is de MSE van altijd het gemiddelde van die labels voorspellen, dus R² is het deel ervan dat het model verklaart: 1 is perfect, 0 niet beter dan het gemiddelde, onder 0 slechter. R² heeft geen eenheid.
        """),
        L.formula(r"[[loss:\text{MAE}]] = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - [[pred:\hat{y}_i]] \rvert \qquad [[loss:\text{RMSE}]] = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - [[pred:\hat{y}_i]])^2}"),
        L.formula(r"R^2 = 1 - \frac{[[loss:\text{MSE}]]}{\mathrm{Var}(y)}"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **mean absolute error**, MAE | | het gemiddelde van de residuen zonder hun teken | `np.mean(abs(residuals))` |
        | **root mean squared error**, RMSE | | de wortel uit de MSE: in de eenheid van het label, en grote missers tellen zwaarder | `math.sqrt(mse)` |
        | **R²**, determinatiecoëfficiënt | R-kwadraat | 1 min de MSE gedeeld door de variantie van de labels: het deel van die variantie dat het model verklaart | `1 - mse / variance` |
        | $\lvert x \rvert$ | de absolute waarde van x | $x$ zonder zijn teken: $\lvert -3 \rvert = 3$. De `abs` uit [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl) doet het ook voor elk getal van een array | `abs(x)` |
        """),
        L.watch_out("R² vergelijkt met het gemiddelde van precies de labels waarop je hem uitrekent, hier de 8 testprijzen. Dat gemiddelde kon niemand vooraf weten, dus het is niet de baseline uit [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), de gemiddelde trainingsprijs. Het is de beste vaste voorspelling voor precies deze prijzen, dus het zit minder ver ernaast dan die baseline, en R² is strenger: ongeveer 0,65, waar 1 − 648 / 3088 op 0,79 uitkomt."),
        mo.md(r"""
        De 8 testhuizen uit [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), met de <span class="c-pred">lijn</span> die van de trainingshuizen leerde en elk <span class="c-residual">residu</span> als rood streepje. Het huis dat het dichtst bij de lijn ligt, wordt nu voor minder verkocht, bijvoorbeeld aan familie. Verschuif de slider, of sleep over de grafiek van de MAE en de RMSE, om het goedkoper te maken; die grafiek laat de <span class="c-loss">MAE</span> (doorgetrokken) en de <span class="c-loss">RMSE</span> (gestippeld) zien voor elke prijs van dat huis.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cheaper_cell():
    def _frame(d):
        d = int(d)
        m = C.metrics(d)
        values = {"d": d, "r": L.number(m["residual"], 1), "mae": L.number(m["mae"], 1), "rmse": L.number(m["rmse"], 1), "r2": L.number(m["r2"], 2)}
        if d == 0:
            house = "Het huis dat het dichtst bij de lijn ligt, voor zijn prijs uit de data: zijn {role} is {r} duizend euro."
        else:
            house = "Het huis dat het dichtst bij de lijn ligt, {d} duizend euro goedkoper: zijn {role} is {r} duizend euro."
        alt = ("Links acht testhuizen naar oppervlakte en prijs, een stijgende lijn, en een rood streepje van elk huis naar de lijn. " + house.format(role="residu", **values) +
               " Rechts de MAE en de RMSE voor elke prijs van dat huis: allebei stijgen ze als het goedkoper wordt, de RMSE sneller zodra het huis ver ernaast zit, van 25,5 naar 74,3 tegen 22,6 naar 46,9 voor de MAE; "
               "nu is de MAE {mae} en de RMSE {rmse} duizend euro.".format(**values))
        return (C.picture(d, {"houses": "Testhuizen", "x": "oppervlakte (honderden m²)", "y": "prijs (duizend euro)", "metrics": "MAE en RMSE",
                              "cheaper": "hoeveel goedkoper (duizend euro)", "euros": "duizend euro", "alt": alt}),
                house.format(role="""<span class="c-residual">residu</span>""", **values)
                + """ <span class="c-loss">MAE</span> {mae}, <span class="c-loss">RMSE</span> {rmse} duizend euro; R² {r2}.""".format(**values))

    explore_cheaper = L.explore("cheaper", "goedkoper", _frame, start=0, low=0, high=200, step=10, role="",
                                task="Maak het huis 100 duizend euro goedkoper. Hoeveel gaan de MAE en de RMSE omhoog?")
    explore_cheaper
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 100 duizend euro goedkoper gaat de MAE van 22,6 naar 34,4, dus 11,8 omhoog: een achtste van de 94,7 duizend euro die het huis nu verder ernaast zit, omdat het één huis van de acht is. De RMSE gaat van 25,5 naar 42,8, dus 17,3 omhoog, en bij 200 duizend goedkoper is hij 74,3, tegen 46,9 voor de MAE. Eén grote misser weegt steeds zwaarder, omdat hij gekwadrateerd wordt. R² is 0,19 bij 200 duizend goedkoper, tegen 0,65 aan het begin.

    Welke meld je dan?

    - De **MAE** als elke euro ernaast evenveel kost, en een paar vreemde voorbeelden niet de doorslag mogen geven.
    - De **RMSE** als grote missers veel erger zijn dan kleine. Hij past ook bij de MSE waarop je traint: verlaag je de een, dan verlaag je de ander.
    - **R²** om te zeggen hoeveel van de spreiding een model verklaart, in één getal zonder eenheid. Vergelijk het alleen op dezelfde labels: met dezelfde missers geven labels die verder uiteenliggen een hogere R². Zet er dus de MAE of de RMSE naast.

    ### Recept

    Om een model te beoordelen dat getallen voorspelt:

    1. Reken de residuen uit op de test set: `y - y_pred`.
    2. MAE: het gemiddelde van `abs(residuals)`.
    3. RMSE: de wortel uit het gemiddelde van `residuals ** 2`, met `math.sqrt`.
    4. R²: 1 min de MSE gedeeld door de variantie van de labels, het gemiddelde van `(y - y.mean()) ** 2`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De 8 testhuizen uit [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), en de lijn die daar van de 32 trainingshuizen geleerd is, met prijzen in duizend euro. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_metrics():
    import math
    import numpy as np

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_test, _y_test = _sizes[_order[32:]], _prices[_order[32:]]
    _w, _b = 152.82, 52.32   # de lijn die van de trainingshuizen leerde

    # 1. de residuen op de test set
    _residuals = _y_test - (_w * _x_test + _b)
    print(_residuals)
    # 2. MAE: de gemiddelde grootte van de residuen
    print(np.mean(abs(_residuals)))
    # 3. RMSE: de wortel uit de MSE
    _mse = np.mean(_residuals ** 2)
    print(_mse, math.sqrt(_mse))
    # 4. R²: 1 min de MSE gedeeld door de variantie van de testprijzen
    _variance = np.mean((_y_test - _y_test.mean()) ** 2)
    print(_variance, 1 - _mse / _variance)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De residuen lopen van ongeveer −10,1 tot 39,4 duizend euro, en zeven van de acht zijn positief: deze huizen kosten toevallig meer dan de lijn voorspelt. De MAE is ongeveer 22,6 duizend euro: gemiddeld zit de lijn zo ver ernaast. De RMSE is ongeveer 25,5, net iets hoger, omdat geen enkele misser eruitspringt. De variantie van de 8 prijzen is ongeveer 1833, dus R² is 1 − 648 / 1833, ongeveer 0,65: de lijn verklaart ongeveer twee derde van hoe ver deze prijzen uiteenliggen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mae():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. MAE en RMSE

        Een weersvoorspelling voor zes dagen: de temperaturen in `y`, wat er voorspeld was in `y_pred`, allebei in graden, en de residuen in `residuals`. Zet de MAE in `mae`, en de RMSE in `rmse`. Vervang de twee `...`. De laatste regel is `mae, rmse`, zodat de check ze allebei ziet. De check probeert ook een andere voorspelling.
        """),
        L.task("Zet de MAE van de voorspelling in `mae`, en zijn RMSE in `rmse`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mae():
    box_mae = L.code_box("mae-rmse", lines=9)
    box_mae
    return (box_mae,)


@app.cell(hide_code=True)
def check_mae(box_mae):
    result_mae = L.check("mae-rmse", box_mae.value)
    result_mae
    return (result_mae,)


@app.cell(hide_code=True)
def hints_mae():
    L.hints(
        "mae-rmse",
        [
            "Stap 2 en 3 van het recept.",
            "`abs` haalt het teken weg bij elk getal van een array, en `np.mean` neemt het gemiddelde. Voor de RMSE: eerst het gemiddelde van de kwadraten, dan één `math.sqrt`.",
            "`mae = np.mean(abs(___))` en `rmse = math.sqrt(np.mean(residuals ** ___))`.",
        ],
        why="De residuen zijn −1, 1, 0, −3, 0 en 6. Hun groottes zijn samen 11, dus de MAE is 11 / 6, ongeveer 1,83. Hun kwadraten zijn samen 47, dus de MSE is 47 / 6, ongeveer 7,83, en de RMSE ongeveer 2,80.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_r2():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. R²

        Dezelfde voorspelling; in `mse` staat al zijn MSE. Zet de variantie van de zes temperaturen in `variance`, en R² in `r2`. Vervang de twee `...`. De laatste regel is `r2`, zodat de check het ziet. De check probeert ook een andere voorspelling.
        """),
        L.task("Zet de R² van de voorspelling in `r2`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_r2():
    box_r2 = L.code_box("r2", lines=8)
    box_r2
    return (box_r2,)


@app.cell(hide_code=True)
def check_r2(box_r2):
    result_r2 = L.check("r2", box_r2.value)
    result_r2
    return (result_r2,)


@app.cell(hide_code=True)
def hints_r2():
    L.hints(
        "r2",
        [
            "Stap 4 van het recept.",
            "De variantie is het gemiddelde van de gekwadrateerde afstanden van `y` tot zijn eigen gemiddelde, `y.mean()`. R² is 1 min `mse` gedeeld door die variantie.",
            "`variance = np.mean((y - ___) ** 2)` en `r2 = 1 - ___ / variance`.",
        ],
        why="De zes temperaturen hebben een gemiddelde van 13, en hun gekwadrateerde afstanden tot dat gemiddelde zijn samen 56, dus de variantie is 56 / 6, ongeveer 9,33. R² is 1 − 7,83 / 9,33, ongeveer 0,161.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 3. Welke meld je?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-metric",
        question="Een bezorgdienst voorspelt wanneer elk pakket aankomt. Eén voorspelling die een uur ernaast zit, maakt een klant veel bozer dan twee voorspellingen die elk een half uur ernaast zitten. Welke metric past het best om de voorspellingen te beoordelen?",
        options={
            "mae": "De MAE: elke minuut ernaast telt even zwaar",
            "right": "De RMSE: grote missers tellen zwaarder",
            "r2": "R²: die heeft geen eenheid",
        },
        correct="right",
        why={
            "mae": "Hier is één uur ernaast erger dan twee keer een half uur, en de MAE telt ze even zwaar. Zie *Welke meld je dan?*",
            "r2": "R² zegt hoeveel van de spreiding het model verklaart, niet hoe erg zijn grote missers zijn. Zie *Welke meld je dan?*",
            "right": "De RMSE kwadrateert elke misser eerst, dus één uur ernaast weegt zwaarder dan twee halve uren, net als voor de klanten.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De MSE uit [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl) is nog steeds waarop een model traint; zoals [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl) zei, kan het getal dat je meldt een ander zijn. In [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) was de gemiddelde trainingsprijs de baseline; R² vergelijkt met het gemiddelde van de labels zelf. In de volgende les, *Project: een lineaire classifier voor handgeschreven cijfers*, train je een model op echte plaatjes, en het eerste wat je controleert, is of het beter doet dan altijd het meest voorkomende cijfer gokken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_mae, result_r2, choice_which):
    L.wrap_up(result_mae, result_r2, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
