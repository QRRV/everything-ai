# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-gradient-descent-1d. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import play
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-descent-1d", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De afgeleide als helling](/course/open/math-slope?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sign():
    quick_sign = L.choice(
        "quick-sign",
        question="Bij $w$ = 3,3 is de helling van de loss ongeveer 3030. Wat verlaagt de loss?",
        options={
            "bigger": "$w$ groter maken",
            "smaller": "$w$ kleiner maken",
            "either": "Allebei: de helling is ver van 0",
        },
        correct="smaller",
        why={
            "bigger": "Een positieve helling: wat doet de loss als $w$ groter wordt? Zie stap 4 van het recept in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "either": "Het teken van de helling zegt hier meer dan de grootte. Zie *Het idee* in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "smaller": "Een positieve helling betekent dat de loss stijgt als $w$ groter wordt, dus een kleinere $w$ verlaagt hem.",
        },
    )
    quick_sign
    return (quick_sign,)


@app.cell(hide_code=True)
def quick_check_steep():
    quick_steep = L.choice(
        "quick-steep",
        question="Bij $w$ = 2,5 is de helling van de loss −8990, bij $w$ = 3,0 is hij −1479. Waar is het dal steiler?",
        options={
            "far": "Bij $w$ = 2,5",
            "near": "Bij $w$ = 3,0",
            "same": "Even steil: beide hellingen zijn negatief",
        },
        correct="far",
        why={
            "near": "Welk van de twee getallen ligt verder van 0? Zie *Het idee* in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "same": "Het teken zegt welke kant op; wat zegt de grootte? Zie *Het idee* in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "far": "−8990 ligt veel verder van 0 dan −1479: bij $w$ = 2,5 verandert de loss veel sneller.",
        },
    )
    quick_steep
    return (quick_steep,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De helling zegt aan welke kant de loss daalt. Maar hoe ver stap je? Met een vaste stap kruip je zolang de bodem ver weg is, en vlak bij de bodem stap je er recht overheen. Laat de helling dat ook bepalen. Stap na stap kom je bij de bodem. Zo leren de meeste modellen in deze course, tot en met large language models.

    **Een voorbeeld uit het dagelijks leven.** Een druppel honing loopt langs de binnenkant van een kom naar beneden. Waar de wand steil is, gaat hij snel, waar de kom vlakker wordt, kruipt hij, en op de bodem komt hij tot stilstand. Anders dan een rollende bal schiet hij nooit door: hoe snel hij gaat, hangt alleen af van hoe steil het is waar hij nu is.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="Je begint bij $w$ = 2,5, en elke stap schuift $w$ op met een klein getal keer de helling. Wat denk je dat er met de stappen gebeurt als je dichter bij de bodem komt?",
        options={
            "same": "Ze blijven even groot",
            "smaller": "Ze worden kleiner",
            "bigger": "Ze worden groter",
        },
        reveal="**Ze worden kleiner.** Bij de bodem is het dal vlakker, dus de helling kleiner, en daarmee ook een stap die een klein getal keer de helling is. Op de bodem zelf is de helling 0, en blijft $w$ staan. Zet hieronder zelf de stappen, en kijk hoe ze krimpen.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Gradient descent vindt stap voor stap de bodem van de loss. Elke stap, de update-stap, schuift de parameter tegen de helling in, met een klein getal keer de helling:
        """),
        L.formula(r"[[param:w_\text{new}]] = [[param:w]] - [[hyper:\eta]] \cdot [[grad:\mathcal{L}'(w)]]"),
        mo.md(r"""
        Door de min gaat de stap tegen de <span class="c-grad">helling</span> in: een positieve helling maakt $w$ kleiner, een negatieve helling maakt hem groter. De <span class="c-hyper">learning rate</span> $\eta$ is een klein getal dat je zelf kiest; het model leert hem niet. Hij bepaalt hoe groot de stappen zijn.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient descent** | | stap voor stap de bodem van de loss zoeken: elke stap schuift de parameter een klein stukje tegen de helling in | een loop |
        | **update-stap** | | één stap van gradient descent: de nieuwe waarde is de oude min de learning rate keer de helling | `w = w - lr * slope` |
        | **learning rate**, $\eta$ | èta | een klein getal dat je zelf kiest, en dat bepaalt hoe groot elke stap is | `lr` |

        De naam komt van de gradient: de helling als een model meer dan één parameter heeft. Die komt in een latere les.

        Zet de stappen zelf: **Stap** zet er één, **Speel af** zet ze allemaal. De <span class="c-loss">kromme</span> is de loss voor elke $w$, met $b = 50$; de <span class="c-param">punten</span> laten zien waar $w$ na elke stap was, vanaf $w$ = 2,5, met $\eta$ = 0,00005. Ernaast: hoe ver $w$ bij elke stap schuift.
        """),
    ])
    return


@app.cell(hide_code=True)
def play_steps_cell():
    play_steps = play.Descent(
        C.loss, C.slope, w_range=(2.0, 4.2), ymax=10000, steps=8, w0=C.START, lr=C.LR, show_lr=False, beside="step",
        decimal="," if L.lang == "nl" else ".", xticks=[2, 2.5, 3, 3.5, 4], yticks=[0, 5000, 10000],
        labels={"task_kick": L.w["task"], "task": "Zet stappen tot <i>w</i> bijna niet meer beweegt. Sleep dan het oranje punt naar de andere kant van het dal, en zet de stappen opnieuw.",
                "title": "De loss voor elke w", "title2": "Hoe ver w bij elke stap schuift", "x_name": "w", "steps_name": "stap",
                "step": "Stap", "reset": "Terug", "play": "Speel af", "pause": "Pauze", "lr_name": "learning rate", "w_name": "beginwaarde van w",
                "step_name": "stap", "gone": "w is uit beeld ↗",
                "alt": "De loss voor elke w van 2 tot 4,2, een dal met het laagste punt bij w = 3,1, met de punten waar w na elke stap van gradient descent was.",
                "alt2": "Hoe ver w schuift bij elk van de acht stappen van gradient descent: 0,449, 0,112, 0,028, 0,007 en nog kleiner."})
    mo.ui.anywidget(play_steps)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De eerste stap is groot (0,45), omdat het dal bij $w$ = 2,5 steil is. Daarna krimpen de stappen: 0,11, dan 0,03, dan 0,007. Na een paar stappen beweegt $w$ bijna niet meer: hij staat op de bodem, bij ongeveer 3,1. Vanaf de andere kant van het dal is de helling positief: dan gaan de stappen naar links, krimpen ze net zo, en komen ze op dezelfde bodem uit.

    ### Recept

    Zo vind je de bodem van de loss met gradient descent:

    1. Kies een beginwaarde voor de parameter en een learning rate.
    2. Meet de helling van de loss waar de parameter nu staat.
    3. Update: de nieuwe waarde is de oude min de learning rate keer de helling.
    4. Herhaal stap 2 en 3, tot de parameter bijna niet meer verandert.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf update-stappen voor $w$, met $b = 50$, vanaf $w$ = 2,5. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_descent():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    # 1. een beginwaarde en een learning rate
    _w = 2.5
    _lr = 0.00005
    _h = 0.001
    # 4. herhaal
    for _step in range(5):
        # 2. de helling waar w nu staat
        _slope = (_loss(_w + _h) - _loss(_w)) / _h
        # 3. de update-stap
        _w = _w - _lr * _slope
        print(_step + 1, _slope, _w, _loss(_w))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel toont de stap, de helling die hij gebruikte, de nieuwe $w$ en zijn loss. Na één stap is $w$ ongeveer 2,949 en is de loss van 2790 naar 267 gedaald. Na vijf stappen is $w$ ongeveer 3,097, met een loss van ongeveer 99,93: bijna het laagste punt dat je met de hand vond, zonder één stap terug te proberen. De learning rate is piepklein (0,00005), omdat de hellingen hier in de duizenden lopen: 0,00005 keer −8983 is −0,45, en min dat is een stap van 0,45 naar rechts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén update-stap met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-step",
        question="Bij $w = 3$ is de helling van de loss −1479, en de learning rate is 0,0001. Wat is $w$ na één update-stap?",
        options={
            "right": "3,1479",
            "wrong-way": "2,8521",
            "no-lr": "1482",
        },
        correct="right",
        why={
            "wrong-way": "Die stap gaat met de helling mee, niet ertegenin. Wat doet de min in de update-stap met een negatieve helling? Zie *Het idee*.",
            "no-lr": "Daar gaat de hele helling af. Waarmee wordt de helling eerst vermenigvuldigd? Zie de update-stap in *Het idee*.",
            "right": "3 − 0,0001 · (−1479) = 3 + 0,1479 = 3,1479. De helling is negatief, dus $w$ gaat omhoog.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Gradient descent voor b

        Nu blijft $w$ 3, en zoekt gradient descent de beste $b$, vanaf $b = 0$, met de learning rate `lr = 0.2`. Het vak heeft de loss als `loss(w, b)` en het heel kleine stapje `h` om een helling te meten. De learning rate is veel groter dan de 0,00005 voor $w$: de helling voor $b$ loopt in de tientallen, niet in de duizenden. De loop in het vak draait 25 rondes. Vervang de `...` door één update-stap voor $b$. De laatste regel is `b`, zodat de check ziet waar hij eindigde. De check probeert ook een andere learning rate.
        """),
        L.task("Vul de loop met één update-stap voor b."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("descend-b", lines=18)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("descend-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "descend-b",
        [
            "Elke ronde is stap 2 en 3 van het recept: eerst de helling, dan de update.",
            "De helling voor $b$ meet je zoals in [De afgeleide als helling](/course/open/math-slope?lang=nl). Welke parameter verander je met `h`? Daarna komt de nieuwe $b$ uit de oude $b$, `lr` en de helling.",
            "`slope = (loss(w, b + h) - loss(w, b)) / h`, en dan `b = b ___ lr * slope`.",
        ],
        why="Elke ronde meet de helling voor $b$ en schuift $b$ ertegenin: `b = b - lr * slope`. Na 25 rondes is $b$ ongeveer 57,75, de bodem van het dal; met de slider in [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl) vond je, in stappen van 1, 58.",
    )
    return


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Oefening 3. Waarom krimpen de stappen?
    """)
    return


@app.cell(hide_code=True)
def explain_cell():
    explain_smaller = L.explain(
        "why-smaller",
        question="De learning rate blijft de hele tijd gelijk. Waarom worden de stappen van gradient descent bij de bodem toch kleiner? Schrijf één of twee zinnen.",
        model_answer="Een stap is de learning rate keer de helling. Bij de bodem is het dal vlakker, dus de helling kleiner, en daarmee de stap ook; op de bodem is de helling 0, en blijft de parameter staan.",
        points=[
            "een stap is de learning rate keer de helling",
            "bij de bodem is de helling kleiner",
            "op de bodem is de helling 0, dus blijft de parameter staan",
        ],
    )
    explain_smaller
    return (explain_smaller,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl) probeerde je stappen naar beide kanten; in [De afgeleide als helling](/course/open/math-slope?lang=nl) vertelde de helling je welke kant op. Gradient descent voegt het samen: stap tegen de helling in, met de learning rate keer de helling, steeds opnieuw. In de volgende les, *Te klein, te groot: de learning rate*, zie je wat er gebeurt als je de learning rate te klein of te groot kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, explain_smaller, result_b):
    L.wrap_up(choice_one, result_b, explain_smaller)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
