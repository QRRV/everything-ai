# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-gradient", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grows():
    quick_grows = L.choice(
        "quick-grows",
        question="De losses van een run zijn 2790, 4314, 6736 en 10506. Wat verander je?",
        options={
            "bigger": "Maak de learning rate groter",
            "smaller": "Maak de learning rate kleiner",
            "more": "Zet meer stappen",
        },
        correct="smaller",
        why={
            "bigger": "De loss groeit bij elke stap. Is dat een learning rate die te klein is, of te groot? Zie *Het idee* in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "more": "De loss groeit bij elke stap. Wat doen meer stappen dan? Zie *Het idee* in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "smaller": "De loss groeit bij elke stap: de training divergeert, omdat de stappen te groot zijn.",
        },
    )
    quick_grows
    return (quick_grows,)


@app.cell(hide_code=True)
def quick_check_converged():
    quick_converged = L.choice(
        "quick-converged",
        question="Wat betekent het dat de training geconvergeerd is?",
        options={
            "zero": "De loss is 0",
            "lowest": "De loss is zo laag als hij kan worden",
            "settled": "Verdere stappen veranderen de loss nauwelijks nog",
        },
        correct="settled",
        why={
            "zero": "De loss van de huizen komt nooit op 0. Zie de tabel met nieuwe woorden in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "lowest": "Dat wordt vaak verward. Zie het kader *Let op* in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "settled": "Geconvergeerd betekent dat de training tot rust gekomen is: verdere stappen veranderen de loss of de parameters nauwelijks nog.",
        },
    )
    quick_converged
    return (quick_converged,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe bewoog één parameter en bleef de andere vast. Maar een model heeft ze allebei, $w$ en $b$, en een groot model heeft er miljoenen. Gradient descent moet voor elke parameter tegelijk weten welke kant de loss op daalt.

    **Een voorbeeld uit het dagelijks leven.** Je staat op een helling, met een wandelkaart. Je kunt nagaan hoe steil het is als je naar het oosten loopt, en hoe steil als je naar het noorden loopt. Die twee hellingen samen zeggen welke kant recht bergop is, en dus welke kant recht bergaf.
    """)
    return


@app.cell(hide_code=True)
def predict_both():
    guess_both = L.guess(
        "both",
        question="Bij $w = 3$, $b = 50$ is de helling van de loss voor $w$ ongeveer −1472, en voor $b$ ongeveer −15,5. Welke denk je dat je moet veranderen om de loss het snelst te verlagen?",
        options={
            "w": "Alleen $w$: zijn helling is veel steiler",
            "both": "Allebei: elke helling gaat over zijn eigen parameter",
            "b": "Alleen $b$",
        },
        reveal="**Allebei.** Elke helling zegt welke kant je zijn eigen parameter op moet schuiven om de loss te verlagen, en beide zijn negatief: maak $w$ en $b$ een beetje groter. Alleen $w$ veranderen verlaagt de loss ook, maar laat liggen wat $b$ kan doen. Dat de helling voor $w$ zoveel groter is, komt onder meer doordat $w$ met de oppervlakte vermenigvuldigd wordt: een stap van 1 in $w$ verschuift elke voorspelling met 48 tot 130, een stap van 1 in $b$ met 1.",
    )
    guess_both
    return (guess_both,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Met meer dan één parameter meet je voor elke parameter om de beurt een helling, terwijl de andere blijven zoals ze zijn. Dat is een partiële afgeleide. Samen, in de volgorde van de parameters, vormen ze de gradient:
        """),
        L.formula(r"[[grad:\nabla \mathcal{L}]] = \left( [[grad:\frac{\partial \mathcal{L}}{\partial w}]],\ [[grad:\frac{\partial \mathcal{L}}{\partial b}]] \right)"),
        mo.md(r"""
        De <span class="c-grad">gradient</span> wijst bergop: in de richting waarin de loss het snelst stijgt. Elk deel zegt naar welke kant van zijn eigen parameter de loss stijgt; de andere kant op daalt hij.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **partiële afgeleide**, $\frac{\partial \mathcal{L}}{\partial w}$ | partieel L naar partieel w | de helling van de loss voor één parameter, terwijl de andere blijven zoals ze zijn | `slope_w` |
        | **gradient**, $\nabla \mathcal{L}$ | nabla L, of de gradient van L | de partiële afgeleiden van alle parameters, in een lijst; hij wijst in de richting waarin de loss het snelst stijgt | `[slope_w, slope_b]` |
        | **contourplot** | | een kaart van de loss over twee parameters, zoals de hoogtelijnen op een wandelkaart: elke lijn verbindt de plekken met dezelfde loss; waar lijnen dicht bij elkaar liggen, verandert de loss snel | |
        """),
        mo.Html(C.picture({"title": "De loss over w en b", "lowest": "laagste", "along_w": "w: −1472", "along_b": "b: −15,5",
                           "alt": "Een contourplot van de loss van de huizen, met w van 2,6 tot 3,6 naar rechts en b van −20 tot 140 omhoog. De binnenste lijn is een lang, smal ovaal rond het laagste punt bij w = 3,15, b = 46, schuin van linksboven naar rechtsonder; de andere lijnen lopen schuin door de figuur. Bij w = 3, b = 50 staat een punt met twee groene pijlen: een naar rechts, langs w, met helling −1472, en een omhoog, langs b, met helling −15,5."})),
        mo.md(r"""
        Elke <span class="c-loss">lijn</span> verbindt de plekken $(w, b)$ waar de loss gelijk is, in stappen van 600; het kruisje is het laagste punt. De loss is een lang, smal dal. Bij $w = 3$, $b = 50$ zijn beide hellingen negatief: de <span class="c-grad">pijlen</span> laten zien welke kant je elke parameter op moet schuiven om de loss te verlagen, naar rechts langs $w$ en omhoog langs $b$. Ze wijzen tegen de gradient in, die bergop wijst.
        """),
        L.watch_out(r"De twee assen hebben een heel andere schaal: naar rechts is de figuur 1 breed in $w$, omhoog 160 hoog in $b$. De pijlen laten de twee hellingen één voor één zien; hun lengte is niet de helling, en samen laten ze niet de richting van de gradient zien."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo meet je de gradient van de loss:

    1. Reken de loss uit waar je staat.
    2. Voor elke parameter om de beurt: verander alleen die met een heel klein stapje $h$, reken de loss opnieuw uit, en deel de verandering door $h$.
    3. Zet de hellingen in een lijst, in de volgorde van de parameters: dat is de gradient.
    4. Lees hem: elk teken zegt welke kant je zijn eigen parameter op moet schuiven om de loss te verlagen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De gradient van de loss bij $w$ = 2,5, $b = 50$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w, b):
        _residuals = _price - (w * _area + b)
        return (_residuals ** 2).mean()

    _w = 2.5
    _b = 50
    _h = 0.001
    # 1. de loss waar je staat
    _now = _loss(_w, _b)
    # 2. één helling per parameter: alleen die verandert
    _slope_w = (_loss(_w + _h, _b) - _now) / _h
    _slope_b = (_loss(_w, _b + _h) - _now) / _h
    # 3. samen, op volgorde: de gradient
    print(_slope_w, _slope_b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gradient is ongeveer `[-8983, -98]`, gemeten met $h$ = 0,001; de precieze helling voor $w$ is −8990. Stap 4: beide hellingen zijn negatief, dus een grotere $w$ en een grotere $b$ verlagen allebei de loss. Vergelijk hem met $w = 3$, $b = 50$ in de figuur, ongeveer `[-1472, -15.5]`: dichter bij de bodem liggen beide hellingen dichter bij 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ways():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee kanten tegelijk
    """)
    return


@app.cell(hide_code=True)
def choice_ways_cell():
    choice_ways = L.choice(
        "two-ways",
        question="Ergens is de gradient van de loss $(-30,\\ 4)$: $\\frac{\\partial \\mathcal{L}}{\\partial w} = -30$ en $\\frac{\\partial \\mathcal{L}}{\\partial b} = 4$. Welke verandering verlaagt de loss?",
        options={
            "right": "Maak $w$ groter en $b$ kleiner",
            "both-up": "Maak $w$ en $b$ allebei groter",
            "w-down": "Maak $w$ kleiner en $b$ groter",
        },
        correct="right",
        why={
            "both-up": "Kijk naar elk teken apart. Wat zegt een positieve helling voor $b$? Zie stap 4 van het recept.",
            "w-down": "Dat gaat voor allebei bergop. Welke kant moet je op bij een negatieve helling? Zie stap 4 van het recept.",
            "right": "De helling voor $w$ is negatief, dus een grotere $w$ verlaagt de loss; de helling voor $b$ is positief, dus een kleinere $b$ doet dat. Samen: tegen de gradient in.",
        },
    )
    choice_ways
    return (choice_ways,)


@app.cell(hide_code=True)
def text_exercise_gradient():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De gradient als functie

        Maak `gradient(w, b)` af. Hij meet de helling van de loss voor $w$ en voor $b$ bij de $w$ en $b$ die hij krijgt, met het heel kleine stapje `h`, en geeft ze terug als lijst: `[slope_w, slope_b]`. Vervang de `...`. De check roept `gradient(3, 50)` aan: dat moet ongeveer `[-1471.7, -15.5]` geven. De check probeert ook andere plekken.
        """),
        L.task("Laat `gradient` de helling voor w en de helling voor b teruggeven, in een lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradient():
    box_gradient = L.code_box("gradient", lines=16)
    box_gradient
    return (box_gradient,)


@app.cell(hide_code=True)
def check_gradient(box_gradient):
    result_gradient = L.check("gradient", box_gradient.value)
    result_gradient
    return (result_gradient,)


@app.cell(hide_code=True)
def hints_gradient():
    L.hints(
        "gradient",
        [
            "Stap 2 van het recept, twee keer: één helling voor $w$ en één voor $b$.",
            "Bij elke helling verandert maar één parameter met `h`. Welke blijft zoals hij is bij de helling voor $w$?",
            "`slope_w = (loss(___, b) - loss(w, b)) / h`, net zo voor `slope_b`, en `return [___, ___]`.",
        ],
        why="`loss(w + h, b)` verandert alleen $w$, en `loss(w, b + h)` alleen $b$. Elke verandering gedeeld door `h` is een partiële afgeleide, en de lijst van de twee, $w$ eerst, is de gradient: ongeveer `[-1471.7, -15.5]` bij $w = 3$, $b = 50$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_steep():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Waar is het het steilst?
        """),
        mo.Html(C.picture({"title": "Drie plekken op de loss", "lowest": "laagste", "along_w": "", "along_b": "",
                           "alt": "Dezelfde contourplot van de loss over w en b, met drie punten: A bij w = 3,0, b = 50, dicht bij het laagste punt; B bij w = 2,8, b = 80, op de bodem van het lange dal; C bij w = 3,3, b = 100, op de steile kant ervan, waar de lijnen dicht bij elkaar liggen."},
                          slopes=False)),
    ])
    return


@app.cell(hide_code=True)
def choice_steep_cell():
    choice_steep = L.choice(
        "steepest",
        question="Bij welk punt verandert de loss het snelst als je een klein stukje opschuift?",
        options={
            "a": "Bij A",
            "b": "Bij B",
            "c": "Bij C",
        },
        correct="c",
        why={
            "a": "A ligt dicht bij het laagste punt. Hoe ver liggen de lijnen eromheen uit elkaar? Zie *contourplot* in de tabel met nieuwe woorden.",
            "b": "B ligt op de bodem van het lange dal. Steek je veel lijnen over als je een klein stukje van B wegschuift?",
            "c": "Rond C liggen de lijnen het dichtst bij elkaar: een kleine stap steekt er veel over, dus de loss verandert snel. De gradient is daar ongeveer [11298, 134].",
        },
    )
    choice_steep
    return (choice_steep,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De afgeleide als helling](/course/open/math-slope?lang=nl) mat je de helling voor één parameter. Nu meet je er één per parameter, terwijl de andere blijven zoals ze zijn, en samen vormen ze de gradient, die bergop wijst. In [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl) stapte je tegen één helling in; in de volgende les, *w en b passen met gradient descent*, stap je tegen de hele gradient in, en pas je $w$ en $b$ samen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_steep, choice_ways, result_gradient):
    L.wrap_up(choice_ways, result_gradient, choice_steep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
