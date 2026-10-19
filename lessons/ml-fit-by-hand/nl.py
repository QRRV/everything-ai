# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-fit-by-hand. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-fit-by-hand", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mse():
    quick_mse = L.choice(
        "quick-mse",
        question="De residuen van een model op drie voorbeelden zijn 2, −2 en 4. Wat is de mean squared error?",
        options={
            "mean": "1,33",
            "right": "8",
            "sum": "24",
        },
        correct="right",
        why={
            "mean": "Dat is het gewone gemiddelde van de residuen. Wat gebeurt er eerst met elk residu? Zie het recept in [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "sum": "Dat is de som van de kwadraten. Wat is de laatste stap van het recept? Zie [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "right": "De kwadraten zijn 4, 4 en 16; samen 24, en gedeeld door 3 voorbeelden 8.",
        },
    )
    quick_mse
    return (quick_mse,)


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="Op dezelfde huizen heeft model A een loss van 120, en model B een loss van 85. Welk model past beter bij de huizen?",
        options={
            "b": "B",
            "a": "A",
            "cannot": "Dat zie je niet aan de loss",
        },
        correct="b",
        why={
            "a": "Is een hoge loss goed of slecht? Zie de tabel met nieuwe woorden in [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "cannot": "Op dezelfde voorbeelden is de loss juist wat twee modellen vergelijkt. Wat is beter, hoog of laag? Zie [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "b": "Lager is beter: op deze huizen zitten de voorspellingen van B er minder ver naast.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe kreeg je de parameters gewoon: $w = 3$ en $b = 50$. Maar waar komen die vandaan? Je kunt nu de loss van elke keuze van parameters uitrekenen, dus je kunt keuzes vergelijken. De beste parameters zijn die met de laagste loss. Die vinden is wat training doet. Voordat een computer het doet, doe jij het met de hand.

    **Een voorbeeld uit het dagelijks leven.** De douche is te koud. Je draait de kraan een stukje, voelt het water, en draait verder, of terug, tot het goed is.
    """)
    return


@app.cell(hide_code=True)
def predict_valley():
    guess_valley = L.guess(
        "valley",
        question="Het model van de huizen heeft $b = 50$. Je maakt $w$ stap voor stap groter, van 2 tot 4,2. Wat denk je dat er met de loss gebeurt?",
        options={
            "down": "Hij blijft dalen",
            "valley": "Hij daalt, en stijgt dan weer",
            "up": "Hij blijft stijgen",
        },
        reveal="**Hij daalt, en stijgt dan weer.** Met een kleine $w$ ligt de lijn onder de huizen, met een grote $w$ erboven. Daartussen loopt de lijn door de huizen, en is de loss het laagst. Probeer het hieronder.",
    )
    guess_valley
    return (guess_valley,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De loss hangt af van de parameters. Kies een andere $w$ of $b$, en de lijn verschuift, en de loss verandert mee:
        """),
        L.formula(r"[[loss:\mathcal{L}]]([[param:w]], [[param:b]]) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - ([[param:w]] \cdot \mathbf{x}_i + [[param:b]])\right)^2"),
        mo.md(r"""
        Een lijn passen met de hand betekent: verschuif de <span class="c-param">parameters</span> tot de <span class="c-loss">loss</span> zo laag mogelijk is. Dat laagste punt geeft de lijn die het best bij deze voorbeelden past.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | $\mathcal{L}(w, b)$ | L van w en b | de loss als functie van de parameters: voor elke keuze van $w$ en $b$ de loss van dat model | `loss(w, b)` |

        Verschuif de slider, of sleep over de grafiek van de loss. De ene grafiek toont de huizen, de <span class="c-pred">lijn</span> van het model en de <span class="c-residual">residuen</span>; de andere de <span class="c-loss">loss</span> voor elke $w$, met een <span class="c-param">punt</span> bij de $w$ die je koos. $b$ blijft 50.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_demo_cell():
    def _frame(w):
        return (C.picture("w", w, {"left": "De huizen en de lijn", "right": "De loss voor elke w", "x": "oppervlakte (m²)", "y": "prijs (duizend euro)", "loss": "loss",
                                   "alt": f"De acht huizen en de lijn van het model met w = {L.number(w, 1)} en b = 50, met de residuen als streepjes. De loss voor elke w van 2 tot 4,2, een dal met het laagste punt bij w = 3,1, en een punt bij w = {L.number(w, 1)}."}),
                f'Met <i>w</i> = {L.number(w, 1)} en <i>b</i> = 50 is de <span class="c-loss">loss</span> <span class="c-loss ea-num">{L.number(C.loss(w, 50))}</span>.')

    explore_demo = L.explore("demo-w", "w", _frame, start=2.5, low=2.0, high=4.2, step=0.1, task="Zoek de $w$ waarbij de loss het laagst is.")
    explore_demo
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De loss voor elke $w$ is een dal. Het laagste punt ligt bij $w$ = 3,1, met een loss van ongeveer 100: lager dan met $w = 3$, het model uit de vorige lessen, dat 172,75 heeft.

    ### Recept

    Zo zoek je met de hand de beste waarde van één parameter:

    1. Begin ergens, en onthoud de loss.
    2. Verschuif de parameter een beetje, en kijk opnieuw naar de loss.
    3. Ging de loss omlaag, ga dan die kant op door. Ging hij omhoog, ga dan de andere kant op.
    4. Stop waar een stap naar beide kanten de loss laat stijgen: dat is het laagste punt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Dezelfde zoektocht in code: de loss voor vijf waarden van $w$, met $b = 50$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_search():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    _b = 50
    # 1. en 2. begin bij w = 2.8, en verschuif w elke keer een beetje omhoog
    for _w in [2.8, 2.9, 3.0, 3.1, 3.2]:
        _residuals = _price - (_w * _area + _b)
        _loss = (_residuals ** 2).mean()
        print(_w, _loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De losses zijn ongeveer 769, 396, 173, 100 en 177. Stap 3: van 2,8 tot 3,1 laat elke stap de loss dalen, dus je gaat door. Stap 4: van 3,1 naar 3,2 stijgt hij, en een stap terug naar 3,0 ook. Het laagste punt, in stappen van 0,1, is $w$ = 3,1. De lange getallen komen van het rekenen met kommagetallen; 769.0550000000004 is 769,055.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De beste bias term

        Nu blijft $w$ 3, zoals in de vorige lessen, en verschuif jij $b$. Zoek de $b$ met de laagste loss. De grafieken laten de lijn zien en de loss voor elke $b$; sleep over de grafiek van de loss, of verschuif de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        return (C.picture("b", b, {"left": "De huizen en de lijn", "right": "De loss voor elke b", "x": "oppervlakte (m²)", "y": "prijs (duizend euro)", "loss": "loss",
                                   "alt": f"De acht huizen en de lijn van het model met w = 3 en b = {L.number(b, 0)}, met de residuen als streepjes. De loss voor elke b van 0 tot 120, een dal, met een punt bij b = {L.number(b, 0)}."}),
                f'Met <i>w</i> = 3 en <i>b</i> = {L.number(b, 0)} is de <span class="c-loss">loss</span> <span class="c-loss ea-num">{L.number(C.loss(3, b))}</span>.')

    explore_b = L.explore("best-b", "b", _frame, start=20, low=0, high=120, step=1, task="Zet de slider op de $b$ met de laagste loss.")
    explore_b
    return (explore_b,)


@app.cell(hide_code=True)
def check_b(explore_b):
    result_b = L.check("best-b", explore_b)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "best-b",
        [
            "Volg het recept: verschuif de slider een beetje, en kijk elke keer naar de loss eronder.",
            "Kijk ook naar de linkergrafiek: waar liggen de meeste huizen, boven of onder de lijn? $b$ schuift de hele lijn omhoog of omlaag.",
            "Het laagste punt ligt tussen 50 en 70. Loop dat stuk stap voor stap langs, met de pijltjestoetsen op de slider: waar begint de loss weer te stijgen?",
        ],
        why="Bij $b = 58$ is de loss 112,75, de laagste op de slider: bij 57 is hij 113,25, en bij 59 114,25. Met $b = 50$ liggen de huizen gemiddeld boven de lijn, dus een hogere $b$ past er beter bij.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ### Oefening 2. Welke kant op?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="Je zoekt met de hand de beste $w$ voor een ander model. Bij $w$ = 1,4 is de loss 52; bij $w$ = 1,5 is hij 47. Welke $w$ probeer je nu?",
        options={
            "back": "1,3: de loss veranderde, dus ga terug",
            "on": "1,6: de loss daalde, dus ga door",
            "stop": "Geen: 47 is de laagste loss",
        },
        correct="on",
        why={
            "back": "Welke kant ging de loss op toen $w$ van 1,4 naar 1,5 ging? Zie stap 3 van het recept.",
            "stop": "Hoe weet je dat de loss het laagst is? Zie stap 4 van het recept: wat heb je nog niet geprobeerd?",
            "on": "De loss daalde toen $w$ groter werd, dus je gaat verder omhoog. Pas als een stap de loss weer laat stijgen, ben je het laagste punt voorbij.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De zoektocht als functie

        Maak `best_b(bs, area, price, w)` af. Hij krijgt een lijst met waarden om voor $b$ te proberen, de oppervlakten en prijzen van een paar huizen, en de weight $w$. Hij geeft de $b$ uit de lijst met de laagste loss terug. De eerste twee regels van de functie maken arrays van de oppervlakten en prijzen. Vervang de `...`. De check roept `best_b` aan met `[40, 50, 60, 70]`, de acht huizen en $w = 3$: dat moet `60` geven. De check probeert ook andere lijsten, huizen en weights.
        """),
        L.task("Laat `best_b` de b met de laagste loss teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=12)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Twee stappen: eerst de loss voor elke b in de lijst, zoals in het uitgewerkte voorbeeld, dan de b die bij de laagste hoort.",
            "Verzamel de losses in een lijst, in dezelfde volgorde als `bs`. Welke methode uit [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl) geeft de plek van het kleinste getal?",
            "`losses = np.array(losses)`, en dan `return bs[losses.___()]`.",
        ],
        why="De loop rekent de loss van elke b uit en verzamelt ze in `losses`. `.argmin()` geeft de plek van de laagste loss, en `bs[...]` de b op die plek. Met `[40, 50, 60, 70]` zijn de losses 427,75; 172,75; 117,75 en 262,75, dus het antwoord is 60.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl) rekende je de loss van één model uit; nu vergeleek je er veel, en koos je dat met de laagste loss. Dat is wat training doet. Met de hand gaat dat voor één of twee parameters, maar een groot model heeft er miljoenen, en die allemaal één voor één proberen duurt eindeloos. In de volgende les, *De afgeleide als helling*, leer je aan de loss zelf te zien welke kant je een parameter op moet schuiven, zonder beide kanten te proberen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_b, result_search):
    L.wrap_up(result_b, choice_direction, result_search)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
