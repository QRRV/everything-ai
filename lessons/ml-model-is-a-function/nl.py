# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-model-is-a-function. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-model-is-a-function", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) en [Functies](/course/open/py-functions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_feature():
    quick_feature = L.choice(
        "quick-feature",
        question="Een model voorspelt de prijs van een huis uit de oppervlakte. Wat is de oppervlakte in deze taak?",
        options={
            "label": "Het label",
            "feature": "Een feature",
            "example": "Een voorbeeld",
        },
        correct="feature",
        why={
            "label": "Het label is wat het model moet voorspellen. Is dat de oppervlakte, of de prijs? Zie de tabel met nieuwe woorden in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "example": "Een voorbeeld is één hele rij: één huis. Hoe heet één waarde die je van tevoren over het huis weet? Zie de tabel met nieuwe woorden in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "feature": "De oppervlakte is van tevoren bekend, voor elk huis: het is een feature. De prijs is het label.",
        },
    )
    quick_feature
    return (quick_feature,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="```python\ndef double(x):\n    return 2 * x\n```\n\nWat geeft `double(4) + 1`?",
        options={
            "ten": "`10`",
            "eight": "`8`",
            "nine": "`9`",
        },
        correct="nine",
        why={
            "ten": "Dat zou `double(4 + 1)` zijn. Wat gebeurt hier eerst: de aanroep, of de `+ 1`? Zie het recept in [Functies](/course/open/py-functions?lang=nl).",
            "eight": "Dat is alleen `double(4)`. Wat doet `+ 1` met wat de functie teruggeeft? Zie de tabel met nieuwe woorden in [Functies](/course/open/py-functions?lang=nl).",
            "nine": "`double(4)` geeft 8 terug, en `+ 1` maakt er 9 van.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) hadden twee huizen nog geen prijs, een ervan van 88 vierkante meter. Hoe komt een model van 88 vierkante meter tot een prijs? Een getraind model kan eruitzien als een black box, maar vanbinnen is het een functie: de features van een voorbeeld gaan erin, en er komt een voorspelling uit. Wat het model van zijn voorbeelden leerde, zit in een paar getallen, de parameters. Verander die, en het model voorspelt iets anders. Deze les bekijkt een heel eenvoudig model: een rechte lijn, met twee parameters.

    **Een voorbeeld uit het dagelijks leven.** De prijs van een taxirit: een vast bedrag om te beginnen, en een bedrag voor elke kilometer. Met die twee getallen reken je de prijs van elke rit uit.
    """)
    return


@app.cell(hide_code=True)
def predict_house():
    guess_house = L.guess(
        "house-price",
        question="Een model voorspelt de prijs van een huis, in duizenden euro's, uit de oppervlakte in vierkante meters: prijs = 3 × oppervlakte + 50. Wat denk je dat het voorspelt voor een huis van 100 vierkante meter?",
        options={
            "no-bias": "300",
            "full": "350",
            "sum": "153",
        },
        reveal="**350.** 3 keer 100 is 300, en 50 erbij maakt 350 duizend euro. De 3 zegt hoeveel de prijs stijgt per vierkante meter, en de 50 tel je één keer op, voor elk huis.",
    )
    guess_house
    return (guess_house,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een model is een functie. Het krijgt de features van een voorbeeld en geeft een voorspelling terug. Met één feature is een heel eenvoudig model een rechte lijn:
        """),
        L.formula(r"[[pred:\hat{y}]] = [[param:w]] \cdot \mathbf{x} + [[param:b]]"),
        mo.md(r"""
        De punt betekent keer. In woorden: de <span class="c-pred">voorspelling</span> is de weight keer de feature, plus de bias term. De twee getallen <span class="c-param">$w$</span> en <span class="c-param">$b$</span> zijn de parameters van het model: ze bepalen welke voorspelling het maakt, en straks kiest de training ze. In deze les kies je ze zelf.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **model** | | een functie die de features van een voorbeeld omzet in een voorspelling | `predict(x, w, b)` |
        | **parameter** | | een getal in een model dat zijn voorspellingen bepaalt; training kiest het | `w`, `b` |
        | **weight**, $w$ | w | hoeveel de voorspelling verandert als de feature 1 groter wordt | `w` |
        | **bias term**, $b$ | b | de voorspelling als de feature 0 is: je telt hem één keer op | `b` |
        | $\mathbf{x}$ | x | de feature van een voorbeeld | `x` |
        | **voorspelling**, $\hat{y}$ | y-dakje | wat het model voor een voorbeeld geeft | `y_pred` |
        | $y$ | y | het label van een voorbeeld: het goede antwoord | `y` |

        In een grafiek is $w$ hoe steil de lijn loopt, en $b$ hoe hoog hij ligt bij $\mathbf{x} = 0$, waar hij de verticale as snijdt; in de grafiek hieronder ligt dat links buiten beeld. Met meer features krijgt elke feature een eigen weight, en telt het model elke feature keer zijn weight op, plus één bias term.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_line():
    mo.vstack([
        mo.Html(C.picture({"title": "Een model voor de prijs van een huis", "x": "oppervlakte (m²)", "y": "prijs (duizend euro)",
                           "line": "3 × oppervlakte + 50",
                           "alt": "Acht verkochte huizen als punten, oppervlakte tegen prijs, en een rechte lijn erdoorheen: de voorspelling 3 keer de oppervlakte plus 50. De meeste punten liggen dicht bij de lijn."})),
        mo.md(r"""
        De acht verkochte huizen uit [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl), als punten: oppervlakte tegen prijs. De <span class="c-pred">lijn</span> is het model met $w = 3$ en $b = 50$: voor elke oppervlakte geeft de lijn de voorspelling. De meeste huizen liggen er dichtbij, een paar verder weg. Andere waarden van $w$ en $b$ geven een andere lijn, en andere voorspellingen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Twee woorden betekenen hier iets anders dan je misschien denkt. Een *parameter* is een getal in een model, zoals $w$ of $b$; de waarden die je aan een Python-functie geeft, zijn de argumenten, zoals in [Functies](/course/open/py-functions?lang=nl), ook al noemt de documentatie van Python de namen in een `def`-regel parameters. En de *bias term* is gewoon het getal $b$; die heeft niets met oneerlijkheid te maken, een andere betekenis van bias."),
        mo.md(r"""
        ### Recept

        Zo maak je een model en gebruik je het:

        1. Schrijf het model als Python-functie: de features en de parameters gaan erin, de voorspelling komt eruit.
        2. Kies waarden voor de parameters.
        3. Roep de functie aan met de features van een voorbeeld: dat geeft de voorspelling.
        4. Teken de lijn van het model over de voorbeelden heen, om te zien hoe dichtbij hij komt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model uit de figuur, voor de huizen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_model():
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. het model: feature en parameters erin, voorspelling eruit
    def _predict(x, w, b):
        return w * x + b

    # 2. de parameters
    _w = 3
    _b = 50
    # 3. de voorspelling voor een huis van 100 vierkante meter
    print(_predict(100, _w, _b))
    # 3. en voor de acht verkochte huizen tegelijk: x is een array
    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    print(_predict(_area, _w, _b))
    # 4. de huizen als punten, en de lijn van het model
    plt.scatter(_area, _price)
    plt.plot([40, 140], [_predict(40, _w, _b), _predict(140, _w, _b)])
    plt.xlabel("oppervlakte (m²)")
    plt.ylabel("prijs (duizend euro)")
    plt.title("Een model voor de prijs van een huis")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Een huis van 100 vierkante meter krijgt de voorspelling 350, zoals bij de gok. Met een array van oppervlakten geeft dezelfde functie in één keer een voorspelling voor elk huis: `w * x` vermenigvuldigt elke oppervlakte, zoals bij vectorisatie. Leg de voorspellingen naast de echte prijzen: het huis van 62 vierkante meter krijgt 236 en is verkocht voor 245, dat van 130 krijgt 440 en is verkocht voor 465. Een lijn heeft maar twee punten nodig, dus de grafiek tekent hem van 40 tot 140 vierkante meter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weight():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat een weight zegt
    """)
    return


@app.cell(hide_code=True)
def choice_weight_cell():
    choice_weight = L.choice(
        "weight-meaning",
        question="Een model voorspelt de prijs van een tweedehands auto, in euro's, uit zijn leeftijd in jaren: prijs = −1500 × leeftijd + 20000. Wat zegt de weight, −1500?",
        options={
            "new": "Een nieuwe auto kost 1500 euro",
            "per-year": "Elk jaar ouder maakt de voorspelde prijs 1500 euro lager",
            "higher": "Elk jaar ouder maakt de voorspelde prijs 1500 euro hoger",
        },
        correct="per-year",
        why={
            "higher": "Kijk naar het teken van de weight. Wat doet −1500 × leeftijd als de leeftijd 1 groter wordt?",
            "new": "Een nieuwe auto heeft leeftijd 0. Wat is de voorspelling dan: welke parameter blijft er over?",
            "per-year": "De weight is hoeveel de voorspelling verandert als de feature 1 groter wordt: een jaar ouder, 1500 euro minder. De bias term, 20000, is de voorspelling voor een auto van leeftijd 0.",
        },
    )
    choice_weight
    return (choice_weight,)


@app.cell(hide_code=True)
def text_exercise_taxi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De parameters van een taxi

        Een taxi rekent 4 euro om te beginnen, en 2,5 euro voor elke kilometer. Het vak heeft het model `fare(km, w, b)`. Kies de twee parameters, `w` en `b`, zo dat het model de prijs van elke rit geeft. Vervang de twee `...`, en laat de andere regels zoals ze zijn. De laatste regel rekent een rit van 12 kilometer uit. De check probeert ook een rit van een andere lengte.
        """),
        L.task("Kies `w` en `b` zo dat `fare(km, w, b)` de prijs van een rit geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_taxi():
    box_taxi = L.code_box("taxi", lines=9)
    box_taxi
    return (box_taxi,)


@app.cell(hide_code=True)
def check_taxi(box_taxi):
    result_taxi = L.check("taxi", box_taxi.value)
    result_taxi
    return (result_taxi,)


@app.cell(hide_code=True)
def hints_taxi():
    L.hints(
        "taxi",
        [
            "Stap 2 van het recept: kies de parameters. Welke prijs betaal je voor elke kilometer, en welke maar één keer?",
            "De weight `w` wordt met de kilometers vermenigvuldigd; de bias term `b` tel je één keer op.",
            "`w = ___` (wat één kilometer kost) en `b = ___` (wat je één keer betaalt).",
        ],
        why="`w = 2.5` is de prijs van elke kilometer, en `b = 4` de startprijs. 12 kilometer kost dan 2,5 × 12 + 4 = 34 euro.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een model met twee features

        Een beter model voor huizen gebruikt twee features: de oppervlakte en het aantal kamers. Met de parameters van de check zit het dichter bij de acht prijzen dan de lijn uit de figuur. Elke feature krijgt een eigen weight, `w_area` en `w_rooms`, en er is één bias term, `b`. Maak `predict(area, rooms, w_area, w_rooms, b)` af: hij geeft de voorspelling terug. Vervang de `...`. De check roept `predict(100, 4, 2, 25, 45)` aan: dat moet `345` geven. De check probeert ook andere huizen en andere parameters.
        """),
        L.task("Laat `predict` de voorspelling van een model met twee features teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two():
    box_two = L.code_box("two-features", lines=5)
    box_two
    return (box_two,)


@app.cell(hide_code=True)
def check_two(box_two):
    result_two = L.check("two-features", box_two.value)
    result_two
    return (result_two,)


@app.cell(hide_code=True)
def hints_two():
    L.hints(
        "two-features",
        [
            "Net als het model met één feature, maar nu brengt elke feature een eigen deel mee.",
            "Er worden drie delen opgeteld: een voor de oppervlakte, een voor de kamers, en de bias term. Welke weight hoort bij welke feature?",
            "`return ___ * area + ___ * rooms + ___`.",
        ],
        why="`w_area * area` is het deel van de oppervlakte, `w_rooms * rooms` het deel van de kamers, en `b` tel je één keer op: 2 × 100 + 25 × 4 + 45 = 345.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) leerde je features en labels kennen; een model is de functie die van het een naar het ander gaat. In [Functies](/course/open/py-functions?lang=nl) schreef je functies die argumenten krijgen en een waarde teruggeven; een model is zo'n functie, met zijn parameters als extra argumenten. In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je een lijn; de lijn van een model laat al zijn voorspellingen tegelijk zien. In de volgende les, *Hoe ver zit één voorspelling ernaast?*, meet je hoe ver een voorspelling van het label af ligt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weight, result_taxi, result_two):
    L.wrap_up(choice_weight, result_taxi, result_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
