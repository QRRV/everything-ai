# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-learning-rate. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-learning-rate", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="$w = 2$, de helling van de loss is daar 50, en de learning rate is 0,01. Wat is $w$ na één update-stap?",
        options={
            "plus": "2,5",
            "right": "1,5",
            "no-lr": "−48",
        },
        correct="right",
        why={
            "plus": "Die stap gaat met de helling mee. Welke kant gaat een update-stap op? Zie de formule in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "no-lr": "Daar gaat de hele helling af. Waarmee wordt hij eerst vermenigvuldigd? Zie de formule in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "right": "2 − 0,01 · 50 = 2 − 0,5 = 1,5. De helling is positief, dus $w$ gaat omlaag.",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-lr",
        question="Wat is de learning rate?",
        options={
            "right": "Een klein getal dat je zelf kiest, en dat bepaalt hoe groot elke stap is",
            "learned": "Een parameter die het model leert, zoals $w$",
            "slope": "De helling van de loss",
        },
        correct="right",
        why={
            "learned": "Wie kiest de learning rate: jij, of gradient descent? Zie de tabel met nieuwe woorden in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "slope": "De helling verandert bij elke stap; de learning rate vermenigvuldigt hem. Zie de formule in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "right": "Je kiest hem voordat de training begint, en elke stap is de learning rate keer de helling.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les kreeg je de learning rate, en kwam gradient descent in een paar stappen bij de bodem. Maar de learning rate kies je zelf, en met een slechte keuze gaat de training mis: te klein, en het duurt eindeloos; te groot, en elke stap springt over de bodem heen, steeds verder. Een groot model trainen duurt dagen, dus je wilt vroeg zien welke van de twee het is. De loss laat het zien.

    **Een voorbeeld uit het dagelijks leven.** Een douche waarvan het water de kraan pas na een tijdje volgt. Draai je hem telkens maar een haartje verder, dan wacht je eindeloos op warm water. Draai je er hard aan, dan schiet het water van ijskoud naar gloeiend heet en weer terug.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="De learning rate van de vorige les was 0,00005. Wat denk je dat er gebeurt met twee keer zoveel, 0,0001?",
        options={
            "faster": "$w$ is twee keer zo snel bij de bodem",
            "zigzag": "$w$ springt heen en weer over de bodem, maar komt er toch",
            "away": "$w$ loopt weg van de bodem",
        },
        reveal="**Hij springt heen en weer over de bodem, maar komt er toch.** De eerste stap gaat van 2,5 naar 3,40, voorbij de bodem bij 3,1; de volgende terug naar 2,95, dan 3,17, elke sprong kleiner. Met nog eens twee keer zoveel, 0,0002, worden de sprongen juist groter: 4,30, dan 0,70, dan 7,91. Dan loopt $w$ weg.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De <span class="c-hyper">learning rate</span> bepaalt hoe de training loopt. Te klein: elke stap is piepklein, en de loss daalt langzaam. Goed gekozen: de loss daalt snel, en verandert dan bijna niet meer: de training convergeert. Te groot: de stappen springen over de bodem heen; veel te groot, en elke sprong komt hoger uit dan de vorige, dus de loss groeit, stap na stap: de training divergeert. Welke het is, zie je in de loss curve, de loss na elke stap. Een learning rate die een beetje te groot is, kan er in de loss curve ook traag uitzien: elke sprong komt maar een beetje lager uit. De stappen op de loss laten het verschil zien: te kleine stappen blijven aan één kant van de bodem, te grote springen van de ene kant naar de andere.
        """),
        mo.Html(C.three_curves({"titles": ["η = 0,000005: te klein", "η = 0,00005: convergeert", "η = 0,00015: divergeert"], "x": "stap", "y": "loss",
                                "alt": "Drie loss curves, elk over 10 stappen vanaf w = 2,5. Met een learning rate van 0,000005 daalt de loss langzaam, van 2790 naar 665. Met 0,00005 daalt hij in drie stappen naar ongeveer 100 en blijft daar. Met 0,00015 groeit hij elke stap, van 2790 tot ongeveer 246000."})),
        mo.md(r"""
        Drie runs van 10 stappen vanaf $w$ = 2,5, met $b = 50$. Let op de verticale assen: de run die divergeert, eindigt bij ongeveer 246000.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **loss curve** | | een grafiek van de loss na elke stap van de training | `plt.plot(losses)` |
        | **convergentie** | | de training komt tot rust: verdere stappen veranderen de loss of de parameters nauwelijks nog | |
        | **divergentie** | | de training loopt weg: de loss groeit stap na stap, meestal omdat de learning rate te groot is | |
        | `abs(x)` | | de grootte van een getal, zonder zijn teken | `abs(-0.05)` is `0.05` |
        """),
        L.watch_out("Geconvergeerd betekent niet dat de loss zo laag is als hij kan worden. Het betekent alleen dat verdere stappen hem nauwelijks nog veranderen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kies je een learning rate:

    1. Begin met een learning rate, en zet een paar stappen.
    2. Kijk naar de loss curve.
    3. Daalt de loss, maar langzaam, kijk dan naar de stappen: blijven ze aan één kant van de bodem, maak de learning rate dan groter; springen ze van de ene kant naar de andere, of stijgt de loss, maak hem dan kleiner.
    4. Herhaal tot de loss snel daalt en dan tot rust komt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Gradient descent voor $w$, 10 stappen vanaf $w$ = 2,5, met drie learning rates. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_rates():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    _h = 0.001
    # 1. drie learning rates, elk met 10 stappen
    for _lr in [0.000005, 0.00005, 0.00015]:
        _w = 2.5
        _losses = [_loss(_w)]
        for _step in range(10):
            _slope = (_loss(_w + _h) - _loss(_w)) / _h
            _w = _w - _lr * _slope
            _losses.append(_loss(_w))
        # 2. de loss voor de eerste stap en na de laatste
        print(_lr, _losses[0], _losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Alle drie beginnen bij een loss van 2790. Met 0,000005 is hij na 10 stappen 665: hij daalt nog, langzaam. Stap 3 zegt: maak hem groter. Met 0,00005 is hij 99,92: de bodem. Met 0,00015 is hij 246154: de loss groeide bij elke stap. Stap 3 zegt: maak hem kleiner. (Python schrijft heel kleine getallen kort: `5e-06` is 0,000005, met de 5 op de zesde plek achter de komma, en `5e-05` is 0,00005.)
    """)
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Lees de losses
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "too-small",
        question="Drie runs van gradient descent vanaf $w$ = 3,8, elk met een eigen learning rate. De loss na elke stap:<br>run A: 3797, 4611, 5577, 6780, 8213<br>run B: 3797, 135, 100, 100, 100<br>run C: 3797, 2769, 2027, 1491, 1104<br>Welke run heeft een learning rate die te klein is?",
        options={
            "a": "Run A",
            "b": "Run B",
            "c": "Run C",
        },
        correct="c",
        why={
            "a": "In run A groeit de loss bij elke stap. Is dat een learning rate die te klein is, of te groot? Zie *Het idee*.",
            "b": "Run B daalt snel en verandert dan bijna niet meer. Wat betekent dat? Zie *Het idee*.",
            "c": "In run C daalt de loss bij elke stap, maar langzaam: de stappen zijn te klein. Met een grotere learning rate ben je er eerder.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kies een learning rate

        Verschuif de slider om de learning rate $\eta$ te kiezen. De ene grafiek toont de loss voor elke $w$, met de <span class="c-param">punten</span> waar $w$ was na elk van 10 stappen vanaf $w$ = 2,5; de andere de <span class="c-loss">loss curve</span>, waarin een loss van meer dan 3000 boven de grafiek uit loopt. Een grotere learning rate is er sneller, tot de stappen te ver gaan springen.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lr_cell():
    def _frame(lr):
        ws, losses = C.run(lr)
        return (C.picture(lr, {"left": "De stappen op de loss", "right": "De loss curve", "x": "stap", "loss": "loss",
                               "alt": f"De loss voor elke w, een dal met het laagste punt bij w = 3,1, en de punten waar w was na elk van 10 stappen met de learning rate {L.number(lr, 6)}; de laatste bij w = {L.number(ws[-1], 3)}. De loss na elke stap, met als laatste {L.number(losses[-1])}." + (" Boven 3000 loopt hij boven de loss curve uit." if max(losses) > 3000 else "")}),
                f'Na 10 stappen is de <span class="c-loss">loss</span> <span class="c-loss ea-num">{L.number(losses[-1])}</span>.')

    explore_lr = L.explore("pick-lr", "η", _frame, start=0.000005, low=0.000005, high=0.00015, step=0.000005, role="hyper",
                           task="Zoek de grootste learning rate waarmee de loss na 10 stappen nog onder de 100 is.")
    explore_lr
    return (explore_lr,)


@app.cell(hide_code=True)
def check_lr(explore_lr):
    result_lr = L.check("pick-lr", explore_lr)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "pick-lr",
        [
            "Kijk naar beide grafieken terwijl je de slider verschuift: daalt de loss langzaam, daalt hij snel, of groeit hij? Blijven de punten aan één kant van de bodem, of springen ze van de ene kant naar de andere?",
            "Begin klein en schuif de slider omhoog, stapje voor stapje. Waar komt de loss na 10 stappen voor het eerst boven de 100?",
            "De grens ligt tussen 0,0001 en 0,00012. Welk stapje is het laatste met een loss onder de 100?",
        ],
        why="0,000105 is de grootste: zijn loss na 10 stappen is 99,99. Elke learning rate vanaf ongeveer 0,00003 komt onder de 100; kleinere kruipen. Vanaf ongeveer 0,00007 springen de stappen over de bodem heen, van de ene kant naar de andere; tot en met 0,000105 komen ze binnen 10 stappen nog tot rust, vanaf 0,00011 niet meer, en vanaf 0,000135 groeit de loss: de training divergeert.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_converged():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Is hij geconvergeerd?

        Maak `converged(losses)` af. Hij krijgt de losses van een run, die bij het begin en dan één na elke stap, en geeft `True` terug als de run geconvergeerd is: als de laatste stap de loss met minder dan 0,01 veranderde, omhoog of omlaag. Anders geeft hij `False` terug. `abs(x)` geeft een getal zonder zijn teken: `abs(-0.05)` is `0.05`. Vervang de `...`. De check roept `converged([2790.2, 267.4, 110.5, 100.6, 99.98, 99.93])` aan: dat moet `False` geven. De check probeert ook andere runs.
        """),
        L.task("Laat `converged` zeggen of de laatste stap de loss nauwelijks veranderde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_converged():
    box_converged = L.code_box("converged", lines=4)
    box_converged
    return (box_converged,)


@app.cell(hide_code=True)
def check_converged(box_converged):
    result_converged = L.check("converged", box_converged.value)
    result_converged
    return (result_converged,)


@app.cell(hide_code=True)
def hints_converged():
    L.hints(
        "converged",
        [
            "Geconvergeerd betekent dat de laatste stap de loss nauwelijks veranderde. Welke twee losses vertellen je wat de laatste stap deed?",
            "`losses[-1]` is de laatste loss. De verandering kan omhoog of omlaag zijn; welk getal geeft de grootte ervan, wat het teken ook is?",
            "`return abs(___ - ___) < 0.01`.",
        ],
        why="`losses[-1] - losses[-2]` is wat de laatste stap met de loss deed, en `abs` laat het teken weg, dus een daling telt ook als verandering. In het voorbeeld veranderde de laatste stap de loss met 0,05: nog niet geconvergeerd.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl) kreeg je de learning rate. Nu kies je hem zelf, en lees je uit de loss curve of hij te klein, goed gekozen of te groot is. Naar loss curves kijk je vanaf nu bij elke training, tot en met je eigen language model. In de volgende les, *Twee parameters: de gradient*, meet je een helling per parameter, voor $w$ en $b$ tegelijk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_runs, result_converged, result_lr):
    L.wrap_up(choice_runs, result_lr, result_converged)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
