# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-gradient-descent-2d. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-descent-2d", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_direction():
    quick_direction = L.choice(
        "quick-direction",
        question="Ergens is de gradient van de loss $(12,\\ -3)$. Welke verandering verlaagt de loss?",
        options={
            "uphill": "Maak $w$ groter en $b$ kleiner",
            "right": "Maak $w$ kleiner en $b$ groter",
            "both-down": "Maak $w$ en $b$ allebei kleiner",
        },
        correct="right",
        why={
            "uphill": "Dat gaat met de gradient mee, bergop. Zie stap 4 van het recept in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "both-down": "Kijk naar elk teken apart. Wat zegt een negatieve helling voor $b$? Zie stap 4 van het recept in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "right": "De helling voor $w$ is positief, dus een kleinere $w$ verlaagt de loss; de helling voor $b$ is negatief, dus een grotere $b$ doet dat.",
        },
    )
    quick_direction
    return (quick_direction,)


@app.cell(hide_code=True)
def quick_check_contour():
    quick_contour = L.choice(
        "quick-contour",
        question="Op een contourplot liggen de lijnen rond een punt ver uit elkaar. Wat vertelt dat je?",
        options={
            "slow": "De loss verandert daar langzaam",
            "fast": "De loss verandert daar snel",
            "low": "De loss is daar laag",
        },
        correct="slow",
        why={
            "fast": "Waar de lijnen dicht bij elkaar liggen, steekt een kleine stap er veel over. En ver uit elkaar? Zie *contourplot* in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "low": "De afstand tussen de lijnen zegt hoe snel de loss verandert, niet hoe hoog hij is. Zie *contourplot* in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "slow": "Lijnen ver uit elkaar betekenen dat een kleine stap er weinig oversteekt: de loss verandert daar langzaam.",
        },
    )
    quick_contour
    return (quick_contour,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl) bleef $w$ 3 terwijl gradient descent $b$ paste, en kwam $b$ uit op ongeveer 57,75. Maar het laagste punt van de loss ligt bij ongeveer $w$ = 3,15, $b$ = 46: zolang $w$ blijft staan, komt $b$ daar nooit. Ze moeten allebei bewegen. In de vorige les mat je de gradient, een helling voor elke parameter; nu stap je ertegenin, en schuiven $w$ en $b$ samen op, bergaf. Een groot model doet precies dit, voor al zijn parameters tegelijk.

    **Een voorbeeld uit het dagelijks leven.** Een wandelaar wil in dichte mist een berg af. Het dal is niet te zien, dus bij elke stap voelt de wandelaar welke kant de grond het steilst afloopt, en zet een klein stapje die kant op.
    """)
    return


@app.cell(hide_code=True)
def predict_path():
    guess_path = L.guess(
        "path",
        question="Gradient descent begint bij $w = 0$, $b = 0$, ver van de bodem van een lang, smal dal. Welke weg denk je dat hij neemt?",
        options={
            "straight": "Recht naar het laagste punt",
            "valley": "Eerst omlaag het dal in, dan over de bodem naar het laagste punt",
            "circles": "Rondjes om het laagste punt",
        },
        reveal="**Eerst omlaag het dal in, dan over de bodem.** De steilste weg omlaag wijst het dal in, niet naar het laagste punt. Eenmaal op de bodem zijn de hellingen klein, en de stappen over de bodem dus ook. Zie de figuur hieronder.",
    )
    guess_path
    return (guess_path,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Met twee parameters doet gradient descent een update-stap voor elke parameter, met dezelfde learning rate, en met de gradient gemeten waar je staat:
        """),
        L.formula(r"[[param:w_\text{new}]] = [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]] \qquad [[param:b_\text{new}]] = [[param:b]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial b}]]"),
        mo.md(r"""
        Eerst worden beide hellingen gemeten, bij dezelfde $w$ en $b$; pas daarna schuiven $w$ en $b$ allebei op. Samen gaat de stap tegen de <span class="c-grad">gradient</span> in: de kant waar de loss het snelst daalt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **loss landscape** | | de loss getekend boven de parameters, zodat trainen eruitziet als bergaf lopen; de contourplot is er een kaart van | |
        """),
        L.watch_out("Het loss landscape is een plaatje van een functie, en bergaf lopen is een manier om het te vertellen, niet wat de computer doet. Met meer dan twee parameters kun je het niet tekenen, maar de update-stappen werken precies hetzelfde."),
        L.note("In deze les staat de oppervlakte in honderden vierkante meters: 62 m² is 0,62. Dus $w$ is nu de prijs per 100 m²: het laagste punt bij $w$ = 314,5 is 3,145 duizend euro per m², dezelfde lijn als in de vorige les. Waarom de verandering? Met vierkante meters is het dal zo smal dat gradient descent ongeveer 100.000 stappen nodig heeft; een latere les laat zien waarom."),
        mo.Html(C.picture({"title": "Gradient descent over het loss landscape", "step": "{k}",
                           "alt": "Een contourplot van de loss van de huizen over w en b, een lang, smal dal dat schuin ligt, met het laagste punt bij w = 314,5, b = 45,8. Een pad van 200 stappen gradient descent begint bij w = 0, b = 0, springt in één stap naar de bodem van het dal bij w = 164, b = 183, en loopt dan over de bodem naar het laagste punt. De plekken na 0, 1, 10, 50 en 200 stappen hebben een nummer."})),
        mo.md(r"""
        Elke <span class="c-loss">lijn</span> verbindt de plekken $(w, b)$ met dezelfde loss; het kruisje is het laagste punt. De <span class="c-param">punten</span> laten zien waar $w$ en $b$ na elke stap waren, met $\eta$ = 0,3; de getallen tellen de stappen. De eerste stap komt op de bodem van het dal; van daaruit lopen ongeveer 200 stappen langs de bodem naar het laagste punt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo pas je $w$ en $b$ met gradient descent:

    1. Kies beginwaarden voor $w$ en $b$, en een learning rate.
    2. Meet beide hellingen, bij dezelfde $w$ en $b$.
    3. Update ze allebei: elk schuift tegen zijn eigen helling in, met de learning rate keer die helling.
    4. Herhaal stap 2 en 3 tot $w$ en $b$ bijna niet meer veranderen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén update-stap voor $w$ en $b$, vanaf $w = 0$, $b = 0$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_update():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58]) / 100   # oppervlakte in honderden m²
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w, b):
        _residuals = _price - (w * _area + b)
        return (_residuals ** 2).mean()

    # 1. beginwaarden en een learning rate
    _w = 0
    _b = 0
    _lr = 0.3
    _h = 0.001
    print(_loss(_w, _b))
    # 2. beide hellingen, bij dezelfde w en b
    _slope_w = (_loss(_w + _h, _b) - _loss(_w, _b)) / _h
    _slope_b = (_loss(_w, _b + _h) - _loss(_w, _b)) / _h
    print(_slope_w, _slope_b)
    # 3. update ze allebei
    _w = _w - _lr * _slope_w
    _b = _b - _lr * _slope_b
    print(_w, _b, _loss(_w, _b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij $w = 0$, $b = 0$ voorspelt het model voor elk huis 0, en is de loss ongeveer 100275. Beide hellingen zijn negatief, ongeveer −548 en −611, dus beide parameters gaan omhoog: $w$ naar ongeveer 164,4 en $b$ naar ongeveer 183,4. De loss daalt naar ongeveer 1825: één stap van de verre kant van het landschap naar de bodem van het dal. Stap 4 herhaalt dit; de figuur laat 200 rondes zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén stap voor twee parameters
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-update",
        question="$w = 2$ en $b = 1$. De gradient is daar $(4,\\ -10)$, en de learning rate is 0,1. Waar zijn $w$ en $b$ na één update-stap?",
        options={
            "wrong-way": "$w$ = 2,4, $b = 0$",
            "mixed": "$w$ = 1,6, $b = 0$",
            "right": "$w$ = 1,6, $b = 2$",
        },
        correct="right",
        why={
            "wrong-way": "Allebei schoven ze met hun helling mee, bergop. Wat doet de min in de update-stap? Zie de formule in *Het idee*.",
            "mixed": "$w$ klopt. Wat doet de min met de negatieve helling voor $b$? Zie de formule in *Het idee*.",
            "right": "$w$: 2 − 0,1 · 4 = 1,6. $b$: 1 − 0,1 · (−10) = 1 + 1 = 2. Elk schuift tegen zijn eigen helling in.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_descend():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Gradient descent voor w en b

        Het vak heeft de huizen, met de oppervlakte in honderden m², de loss als `loss(w, b)`, beginwaarden $w = 0$ en $b = 0$, de learning rate `lr = 0.3`, het heel kleine stapje `h`, en een loop van 50 rondes. Vervang de `...` door één update-stap voor beide parameters. De laatste regel is `[w, b]`, zodat de check ziet waar ze eindigden. De check probeert ook een andere learning rate.
        """),
        L.task("Vul de loop met één update-stap voor w en b samen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_descend():
    box_descend = L.code_box("descend-wb", lines=20)
    box_descend
    return (box_descend,)


@app.cell(hide_code=True)
def check_descend(box_descend):
    result_descend = L.check("descend-wb", box_descend.value)
    result_descend
    return (result_descend,)


@app.cell(hide_code=True)
def hints_descend():
    L.hints(
        "descend-wb",
        [
            "Elke ronde is stap 2 en 3 van het recept: eerst beide hellingen, dan beide updates.",
            "Meet de helling voor $b$ voordat $w$ verschuift: anders meet je hem ergens anders. Hoe zorg je daarvoor?",
            "`slope_w = ...` en `slope_b = ...` zoals in het uitgewerkte voorbeeld, dan `w = w - lr * ___` en `b = b - lr * ___`.",
        ],
        why="Elke ronde meet beide hellingen bij dezelfde $w$ en $b$, en schuift dan elk tegen zijn eigen helling in. Na 50 rondes is $w$ ongeveer 267,6 en $b$ ongeveer 86,1: op de bodem van het dal, op weg naar het laagste punt bij ongeveer 314,5 en 45,8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.md(r"""
    ### Oefening 3. De lange weg over de bodem
    """)
    return


@app.cell(hide_code=True)
def choice_floor_cell():
    choice_floor = L.choice(
        "valley-floor",
        question="In de figuur gaat de eerste stap bijna recht omlaag naar de bodem van het dal, maar daarna doet gradient descent ongeveer 200 stappen over de bodem. Waarom?",
        options={
            "right": "Langs de bodem zijn de hellingen klein, en de stappen dus ook",
            "lr": "De learning rate wordt bij elke stap een beetje kleiner",
            "order": "Gradient descent verschuift eerst $b$, en pas daarna $w$",
        },
        correct="right",
        why={
            "lr": "De learning rate blijft de hele tijd 0,3. Wat bepaalt verder hoe groot een stap is? Zie de formule in *Het idee*.",
            "order": "Elke stap schuift $w$ en $b$ samen op. Kijk naar de lijnen langs de bodem: hoe ver liggen ze uit elkaar?",
            "right": "Een stap is de learning rate keer de helling. Over de bodem van het smalle dal verandert de loss nauwelijks, dus de hellingen, en de stappen, zijn klein.",
        },
    )
    choice_floor
    return (choice_floor,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) mat je de gradient; nu stap je ertegenin, en schuiven $w$ en $b$ samen over het loss landscape. Het is dezelfde update-stap als in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl), één keer voor elke parameter. Met meer features krijgt een model meer weights; in de volgende les, *Veel features: het inproduct*, schrijf je hun gewogen som in één korte vorm.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_floor, choice_one, result_descend):
    L.wrap_up(choice_one, result_descend, choice_floor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
