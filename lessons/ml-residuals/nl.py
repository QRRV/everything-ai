# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-residuals. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow function, residual, vectorization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-residuals", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_predict():
    quick_predict = L.choice(
        "quick-predict",
        question="Een model is $\\hat{y} = 2 \\cdot \\mathbf{x} + 10$. Wat voorspelt het voor $\\mathbf{x} = 4$?",
        options={
            "no-bias": "8",
            "right": "18",
            "added": "16",
        },
        correct="right",
        why={
            "no-bias": "Dat is alleen 2 keer 4. Wat tel je één keer op, bij elke voorspelling? Zie de tabel met nieuwe woorden in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "added": "Dat telt 2, 4 en 10 op. Wat betekent de punt tussen 2 en $\\mathbf{x}$? Zie *Het idee* in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "right": "2 keer 4 is 8, en de bias term 10 erbij maakt 18.",
        },
    )
    quick_predict
    return (quick_predict,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="Wat zijn in het model $\\hat{y} = w \\cdot \\mathbf{x} + b$ de parameters?",
        options={
            "wb": "$w$ en $b$",
            "xy": "$\\mathbf{x}$ en $\\hat{y}$",
            "w": "Alleen $w$",
        },
        correct="wb",
        why={
            "xy": "$\\mathbf{x}$ is de feature, en $\\hat{y}$ de voorspelling. Welke getallen bepalen welke voorspelling het model maakt? Zie de tabel met nieuwe woorden in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "w": "$w$ is er een van. Is er nog een getal dat elke voorspelling verandert? Zie de tabel met nieuwe woorden in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "wb": "De weight $w$ en de bias term $b$ bepalen welke voorspelling het model maakt: dat zijn de parameters.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model maakt een voorspelling, en bij een voorbeeld met een label kun je zien hoe goed die is. Voordat je kunt zeggen of het ene model beter is dan het andere, of een model kunt verbeteren, heb je een getal nodig voor hoe ver een voorspelling ernaast zit. Dat getal, voor één voorbeeld, is de eerste stap naar alles wat een model leert.

    **Een voorbeeld uit het dagelijks leven.** Het weerbericht zei 18 graden, en het werd 21. De voorspelling was 3 graden te laag. Een andere dag zei het 15 en werd het 13: 2 graden te hoog.
    """)
    return


@app.cell(hide_code=True)
def predict_sign():
    guess_sign = L.guess(
        "sign",
        question="Een model voorspelde 280 duizend euro voor een huis dat voor 300 werd verkocht. Een andere keer voorspelde het 340 voor een huis dat voor 320 werd verkocht. Moet het getal dat zegt hoe ver een voorspelling ernaast zit, voor allebei hetzelfde zijn, denk je?",
        options={
            "same": "Ja: allebei zitten ze er 20 naast",
            "sign": "Nee: de eerste was te laag, de tweede te hoog",
        },
        reveal="**Allebei 20 ernaast, maar naar verschillende kanten.** Het residu bewaart die kant in zijn teken: 300 − 280 = 20, te laag, en 320 − 340 = −20, te hoog. Hoe ver ernaast, en naar welke kant, in één getal.",
    )
    guess_sign
    return (guess_sign,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Voor één voorbeeld met een label is het residu hoe ver de voorspelling ernaast zit: het label min de voorspelling.
        """),
        L.formula(r"[[residual:\text{residu}]] = y - [[pred:\hat{y}]]"),
        mo.md(r"""
        Een positief <span class="c-residual">residu</span> betekent dat het model te laag voorspelde; een negatief, te hoog; 0, precies goed. Het staat in de eenheid van het label: hier duizend euro.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **residu** | | voor één voorbeeld het label min de voorspelling: hoe ver die ene voorspelling ernaast zit, en naar welke kant | `y - y_pred` |
        | `.argmin()` | | net als `.argmax()`, maar de plek van het kleinste getal | `residuals.argmin()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_residuals():
    mo.vstack([
        mo.Html(C.picture({"title": "De residuen van de huizen", "x": "oppervlakte (m²)", "y": "prijs (duizend euro)", "biggest": "+25",
                           "alt": "De acht verkochte huizen als punten, de lijn van het model 3 keer de oppervlakte plus 50, en van de lijn naar elk punt een verticaal streepje: het residu. Zeven punten liggen boven de lijn; het huis van 95 vierkante meter ligt er 15 onder. Het langste streepje, +25, hoort bij het huis van 130 vierkante meter."})),
        mo.md(r"""
        De huizen en het model uit [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl), met elk <span class="c-residual">residu</span> als streepje van de <span class="c-pred">lijn</span> naar het punt. Een huis boven de lijn is verkocht voor meer dan voorspeld: een positief residu. Zeven huizen liggen boven de lijn; dat van 95 vierkante meter ligt eronder. Het langste streepje, 25, hoort bij het huis van 130 vierkante meter.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Veel boeken noemen het residu de *error* of de fout. In deze course is hoe ver een voorspelling ernaast zit altijd het residu, nooit een fout; een fout is iets wat misgaat, zoals in je code."),
        mo.md(r"""
        ### Recept

        Zo reken je een residu uit:

        1. Maak met het model de voorspelling $\hat{y}$ van het voorbeeld.
        2. Trek die af van het label: $y - \hat{y}$, in die volgorde.
        3. Voor veel voorbeelden tegelijk doe je hetzelfde met arrays: één residu per voorbeeld.
        4. Lees het teken: positief is te laag, negatief te hoog.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De residuen van de acht huizen, met het model uit de figuur. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_residuals():
    import numpy as np

    _w = 3
    _b = 50
    # 1. één huis: 130 vierkante meter, verkocht voor 465
    _y_pred = _w * 130 + _b
    print(_y_pred)
    # 2. het residu ervan
    print(465 - _y_pred)
    # 3. alle acht huizen tegelijk
    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    _residuals = _price - (_w * _area + _b)
    print(_residuals)
    # 4. het huis met het meest negatieve residu: te hoog voorspeld
    print(_residuals.argmin())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het huis van 130 vierkante meter krijgt de voorspelling 440 en is verkocht voor 465: een residu van 25, het model zat 25 duizend euro te laag. Voor alle huizen tegelijk trekt `_price - (_w * _area + _b)` plek voor plek af, zoals bij vectorisatie; de haakjes zorgen dat de hele voorspelling eraf gaat. Zeven residuen zijn positief. `.argmin()` geeft plek 4, het huis van 95 vierkante meter, met een residu van −15: het enige dat het model te hoog voorspelde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén residu
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-residual",
        question="Een model voorspelt dat een rit 25 minuten duurt. Hij duurde 31. Wat is het residu?",
        options={
            "neg": "−6",
            "pred": "25",
            "pos": "6",
        },
        correct="pos",
        why={
            "neg": "Dat is de voorspelling min het label. In welke volgorde trekt een residu af? Zie de formule in *Het idee*.",
            "pred": "Dat is de voorspelling zelf. Een residu is hoe ver die ernaast zit. Zie de formule in *Het idee*.",
            "pos": "Het label min de voorspelling: 31 − 25 = 6. Positief: het model voorspelde te laag, 6 minuten te kort.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_residuals():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Alle residuen

        Het vak heeft vijf ritten van de taxi uit [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl): de afstanden als de array `km`, en de prijzen die betaald zijn als `fare`. Het model is dat uit die les, met `w = 2.5` en `b = 4`; echte ritprijzen wijken een beetje af, bijvoorbeeld door het verkeer. Maak `residuals`: het residu van elke rit, als array. Vervang de `...`. De laatste regel is `residuals`, zodat de check het ziet. De check probeert ook andere parameters.
        """),
        L.task("Maak `residuals`: de prijs van elke rit min de voorspelling ervan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_residuals():
    box_residuals = L.code_box("residuals", lines=9)
    box_residuals
    return (box_residuals,)


@app.cell(hide_code=True)
def check_residuals(box_residuals):
    result_residuals = L.check("residuals", box_residuals.value)
    result_residuals
    return (result_residuals,)


@app.cell(hide_code=True)
def hints_residuals():
    L.hints(
        "residuals",
        [
            "Stap 3 van het recept: dezelfde aftrekking als voor één voorbeeld, maar met hele arrays.",
            "Eerst de voorspellingen van het model, met `w` en `b`; dan aftrekken. Wat komt eerst in een residu, het label of de voorspelling?",
            "`residuals = fare - (___)`.",
        ],
        why="`w * km + b` geeft de vijf voorspellingen, en `fare - (...)` trekt ze plek voor plek af van de ritprijzen. Zonder de haakjes zou Python alleen `w * km` aftrekken, en daarna `b` optellen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het verst te laag voorspeld

        Maak `too_low(area, price, w, b)` af. Hij krijgt de oppervlakten en prijzen van een paar huizen, en de parameters van een model. Hij geeft de plek terug, vanaf 0, van het huis waarvan het model de prijs het verst te laag voorspelt. Vervang de `...`. De check roept `too_low` aan met de acht huizen en $w = 3$, $b = 50$: dat moet `5` geven. De check probeert ook andere huizen en parameters.
        """),
        L.task("Laat `too_low` de plek teruggeven van het huis dat het model het verst te laag voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_low():
    box_low = L.code_box("too-low", lines=9)
    box_low
    return (box_low,)


@app.cell(hide_code=True)
def check_low(box_low):
    result_low = L.check("too-low", box_low.value)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "too-low",
        [
            "Twee stappen: eerst de residuen van alle huizen, dan de plek van het huis dat het verst te laag voorspeld is.",
            "Te laag voorspeld betekent een positief residu: welk residu is het, het grootste of het kleinste? In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) staat de methode die de plek ervan geeft.",
            "`residuals = ___`, en dan `return residuals.___()`.",
        ],
        why="De residuen zijn `price - (w * area + b)`. Het huis dat het verst te laag voorspeld is, heeft het grootste residu, en `.argmax()` geeft de plek ervan.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl) maakte een model voorspellingen; nu weet je voor elk voorbeeld hoe ver de voorspelling ernaast zit, en naar welke kant. Met arrays, zoals in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl), krijg je alle residuen tegelijk, en met `argmax`, uit [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl), vind je het grootste. In de volgende les, *Eén getal voor hoe ver je ernaast zit: de loss*, breng je alle residuen samen in één getal, om hele modellen te vergelijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_low, result_residuals):
    L.wrap_up(choice_one, result_residuals, result_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
