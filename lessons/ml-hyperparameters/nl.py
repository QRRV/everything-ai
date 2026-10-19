# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-hyperparameters. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-hyperparameters", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl) en [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rate():
    quick_rate = L.choice(
        "quick-rate",
        question="Gradient descent loopt met een learning rate die veel te groot is. Wat gebeurt er met de loss?",
        options={
            "slow": "Die daalt, maar heel langzaam",
            "right": "Die groeit, stap na stap",
            "faster": "Die daalt sneller, naar hetzelfde laagste punt",
        },
        correct="right",
        why={
            "slow": "Dat is een learning rate die te klein is. Zie de figuur in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "faster": "Elke stap schiet voorbij het laagste punt, elke keer verder. Zie de figuur in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "right": "Elke stap springt voorbij het laagste punt, verder dan de stap ervoor: de loss groeit.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Een curve heeft een MSE van 0,3 op zijn training set en 25 op zijn test set. Wat doet hij?",
        options={
            "under": "Hij underfit",
            "fine": "Hij doet het goed",
            "right": "Hij overfit",
        },
        correct="right",
        why={
            "under": "Dan zouden beide MSE's hoog zijn. Zie *Het idee* in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "fine": "Dan zouden ze allebei laag zijn en dicht bij elkaar liggen. Zie *Het idee* in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "right": "Bijna precies op de training set, ver ernaast op nieuwe data: hij volgt de ruis.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een getraind model zit vol getallen die het trainen heeft ingesteld: zijn weights en zijn bias term. Maar aan sommige getallen komt het trainen nooit: de learning rate, het aantal stappen, de graad van een curve. Die kies je vóór het trainen begint, en ze bepalen hoe het trainen gaat en wat het model kan leren. Elk soort getal kies je op een eigen manier, dus je moet ze uit elkaar kunnen houden.

    **Een voorbeeld uit het dagelijks leven.** Een rijinstructeur bepaalt hoeveel lessen een leerling krijgt en hoe lang elke les duurt. Hoe hard je remt en wanneer je stuurt, leert de leerling zelf, door te rijden.
    """)
    return


@app.cell(hide_code=True)
def predict_pick_degree():
    guess_pick_degree = L.guess(
        "pick-degree",
        question="Het trainen kiest de weights met de laagste MSE op de training set. Wat gebeurt er volgens jou als je de graad van een curve op dezelfde manier kiest: de graad met de laagste training-MSE?",
        options={
            "highest": "Dan kies je altijd de hoogste graad die je toestaat",
            "best": "Dan kies je de beste graad",
            "lowest": "Dan kies je altijd graad 1",
        },
        reveal="**Dan kies je altijd de hoogste graad die je toestaat.** In [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) verlaagde meer capaciteit altijd de training-MSE, dus deze regel kiest de meest flexibele curve: de curve met de meeste kans op overfitting. De graad moet je dus anders kiezen, aan de hand van hoe het model het doet op voorbeelden waarop het niet trainde.",
    )
    guess_pick_degree
    return (guess_pick_degree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **<span class="c-param">parameter</span>** van een model is een getal in dat model dat het trainen aanpast: de weights en de bias term. Een **<span class="c-hyper">hyperparameter</span>** is een instelling die je vóór het trainen kiest, en die het trainen niet verandert. Sommige horen bij het model zelf, zoals de graad van een polynoom: die bepaalt hoeveel weights er zijn. Andere sturen het trainen, zoals de learning rate $\eta$ en het aantal stappen. Hyperparameters gaan naast de data de trainingscode in, parameters komen eruit: `w, b = fit(x, y, lr, steps)`.

        Het trainen kiest de parameters door de loss op de training set te verlagen. Voor een hyperparameter zoals de graad werkt dat niet: op de training set ziet meer capaciteit er altijd beter uit. Een hyperparameter kies je door een paar waarden te proberen, met elke waarde een model te trainen, en die modellen te vergelijken op voorbeelden waarop ze niet trainden.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **parameter** van een model | | een getal in een model dat het trainen aanpast, zoals een weight of de bias term | `w`, `b` |
        | **hyperparameter** | | een instelling die je vóór het trainen kiest, zoals de learning rate of het aantal stappen; het trainen verandert hem niet | `lr`, `steps` |
        """),
        L.watch_out("De documentatie van Python noemt de namen in een `def`-regel ook parameters, zoals `lr` en `steps` in `def fit(x, y, lr, steps)`; deze course noemt ze argumenten. `lr` is dus een argument van `fit` en een hyperparameter van het trainen, maar nooit een parameter van het model."),
        draw.flow(
            [("Jij kiest", ["hyperparameters:", "learning rate η,", "aantal stappen"]),
             ("Trainen", ["gradient descent", "op de training set"]),
             ("Het trainen leert", ["parameters:", "weights w,", "bias term b"])],
            forward=["vóór het\ntrainen", "terwijl het\nloopt"],
            back=[],
            title="Wie welke getallen instelt"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo houd je de parameters en hyperparameters in je eigen code uit elkaar:

    1. Zoek de getallen die je vóór het trainen instelt, naast de data, zoals `lr` en `steps` in `fit(x, y, lr, steps)`. Dat zijn hyperparameters.
    2. Zoek de getallen die het trainen teruggeeft, zoals `w` en `b`. Dat zijn parameters.
    3. Verander een hyperparameter en train opnieuw: de parameters komen er anders uit.
    4. Het trainen kiest de parameters met de loss op de training set; jij kiest de hyperparameters door modellen te vergelijken op voorbeelden waarop ze niet trainden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De training set van 32 huizen uit [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl), en gradient descent in een functie `_fit` met de hyperparameters `lr` en `steps` als argumenten. De lijn wordt drie keer getraind, met 10, 100 en 2000 stappen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_hyperparameters():
    import numpy as np

    def _fit(x, y, lr, steps):
        w = 0.0
        b = 0.0
        for step in range(steps):
            r = y - (w * x + b)
            w = w - lr * (-2 * (r * x).mean())
            b = b - lr * (-2 * r.mean())
        return w, b

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_train = _sizes[_order[:32]]
    _y_train = _prices[_order[:32]]

    # 1. de hyperparameters, gekozen vóór het trainen
    _lr = 0.1
    for _steps in [10, 100, 2000]:
        # 2. en 3. de parameters die eruit komen, voor elk aantal stappen
        _w, _b = _fit(_x_train, _y_train, _lr, _steps)
        print(_steps, _w, _b, np.mean((_y_train - (_w * _x_train + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met 10 stappen krijgt de lijn $w \approx 128{,}6$ en $b \approx 85{,}6$, met een training-MSE van ongeveer 601. Met 100 stappen ongeveer 146,1 en 61,6, en 492. Met 2000 stappen ongeveer 152,8 en 52,3, en 482: de lijn die je eerder voor deze huizen kreeg. Dezelfde training set en dezelfde learning rate, maar een ander aantal stappen, en de parameters komen er anders uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wie stelt het in?

    Zet elk getal in zijn groep: geleerd door het trainen, of door jou gekozen.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_who = L.sort("who-sets")
    sort_who
    return (sort_who,)


@app.cell(hide_code=True)
def check_sort(sort_who):
    result_who = L.check("who-sets", sort_who)
    result_who
    return (result_who,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "who-sets",
        [
            "Stap 1 en 2 van het recept: vraag bij elk getal of jij het vóór het trainen instelt, of het trainen het teruggeeft.",
            "Welke namen gaan bij `w, b = fit(x, y, lr, steps)` naast de data de functie in, en welke komen eruit? En bij `w = np.polyfit(x, y, 3)`?",
            "De weight voor de oppervlakte en de bias term komen uit `fit`. Welke getallen van de polynoom gaan `np.polyfit` in, en welke komen eruit?",
        ],
        why="**Parameters:** de weight voor de oppervlakte, de bias term en de weights van de polynoom; het trainen rekent ze uit. **Hyperparameters:** de learning rate, het aantal stappen en de graad; die kies je vóór het trainen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Drie learning rates

        Tien huizen, en dezelfde functie als `_fit` in het uitgewerkte voorbeeld, nu met de naam `fit`. Train de lijn één keer voor elke learning rate in de lijst, met `steps` stappen, en voeg zijn MSE op de huizen toe aan `mses`. Die training-MSE laat zien hoe ver elke run kwam, niet welke learning rate je moet kiezen. Vervang de `...` door twee regels in de loop. De laatste regel is `mses`, zodat de check het ziet. De check probeert ook een ander aantal stappen.
        """),
        L.task("Voeg voor elke learning rate de training-MSE van de lijn toe aan `mses`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("three-rates", lines=19)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("three-rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "three-rates",
        [
            "In de loop, zoals in het uitgewerkte voorbeeld: train met `fit`, en reken dan de MSE uit van de lijn die eruit komt.",
            "`fit` krijgt de oppervlaktes, de prijzen, de learning rate van deze ronde en `steps`, en geeft `w` en `b` terug. Hoe voegde je iets toe aan het eind van een lijst?",
            "`w, b = fit(sizes, prices, lr, steps)` en `mses.append(np.mean((prices - (___)) ** 2))`.",
        ],
        why="Elke ronde traint een nieuwe lijn met een eigen learning rate, en `mses.append` voegt zijn MSE toe. Na 200 stappen: ongeveer 125 voor 0,01, 10,1 voor 0,1 en 9,9 voor 0,3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_explain():
    mo.md(r"""
    ### Oefening 3. In je eigen woorden
    """)
    return


@app.cell(hide_code=True)
def explain_difference_cell():
    explain_difference = L.explain(
        "rate-or-weight",
        question="Wat is het verschil tussen de learning rate en de weight $w$ van een lijn? Schrijf één of twee zinnen.",
        model_answer="De weight w is een parameter: gradient descent verandert hem bij elke stap, en zijn waarde aan het eind komt uit de training set. De learning rate is een hyperparameter: die kies je vóór het trainen, het trainen verandert hem niet, en hij bepaalt hoe groot de stappen zijn die w veranderen.",
        points=[
            "de weight is een parameter, de learning rate een hyperparameter",
            "het trainen verandert de weight, stap voor stap, vanuit de training set",
            "de learning rate kies je vóór het trainen; het trainen verandert hem niet, en hij bepaalt hoe groot de stappen zijn",
        ],
    )
    explain_difference
    return (explain_difference,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl) koos je de learning rate zelf en zag je wat een slechte keuze doet; het is een hyperparameter. In [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) bepaalde de graad de capaciteit van een curve; ook dat is een hyperparameter, en de training-MSE kan hem niet kiezen. In de volgende les, *Hyperparameters kiezen: de validation set*, kies je ze op een aparte set voorbeelden, zodat de test set eerlijk blijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_who, result_rates, explain_difference):
    L.wrap_up(result_who, result_rates, explain_difference)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
