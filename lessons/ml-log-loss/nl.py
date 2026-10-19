# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-log-loss. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow surprisal, binary cross-entropy, regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-log-loss", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) en [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="Een logistische regressie heeft de weights 2 en 1 en de bias term −4. Waar ligt het voorbeeld met de features 1 en 2?",
        options={
            "right": "Precies op de decision boundary",
            "pass": "Aan de kant van klasse 1",
            "fail": "Aan de kant van klasse 0",
        },
        correct="right",
        why={
            "pass": "Reken de logit uit: 2 × 1 + 1 × 2 − 4. Aan welke kant is de logit boven 0? Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "fail": "Reken de logit uit: 2 × 1 + 1 × 2 − 4. Aan welke kant is de logit onder 0? Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "right": "De logit is 2 + 2 − 4 = 0, dus de kans is precies 0,5: op de decision boundary.",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def quick_check_mse():
    quick_mse = L.choice(
        "quick-mse",
        question="Twee voorbeelden hebben de labels 4 en 6, en een model voorspelt 3 en 4. Wat is de mean squared error?",
        options={
            "sum": "5",
            "plain": "1,5",
            "right": "2,5",
        },
        correct="right",
        why={
            "sum": "Dat is de som van de gekwadrateerde residuen. Waar vraagt *mean* in de naam om? Zie de formule in [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "plain": "Dat is het gemiddelde van de residuen, zonder ze te kwadrateren. Waar vraagt *squared* in de naam om? Zie de formule in [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl).",
            "right": "De residuen zijn 1 en 2; in het kwadraat 1 en 4; hun gemiddelde is 2,5.",
        },
    )
    quick_mse
    return (quick_mse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om een model te trainen heb je een loss nodig: één getal voor hoe ver het ernaast zit. Voor een getal werkte de mean squared error goed. Voor een kans niet. Een spamfilter dat 0,01 zegt bij een mail die spam is, zit er zeker en flink naast, maar zijn gekwadrateerde residu, $(1 - 0{,}01)^2 = 0{,}98$, is nog geen vier keer de 0,25 van een filter dat 0,5 zei en het niet wist, en geen enkele kans brengt het boven 1. Een goede loss voor kansen straft een zekere fout veel harder dan een twijfelende.

    **Een voorbeeld uit het dagelijks leven.** Een weervoorspeller die 99% kans op zon geeft op een dag dat het giet, verliest veel meer vertrouwen dan een die 60% kans op zon gaf. Allebei zaten ze ernaast, maar de een was er zeker van.
    """)
    return


@app.cell(hide_code=True)
def predict_surprise():
    guess_surprise = L.guess(
        "surprise",
        question="Een model gaf kans 0,5 aan wat er echt gebeurde, en dat kost $-\\log(0{,}5) \\approx 0{,}69$. Wat kost het volgens jou als het model 0,01 gaf aan wat er echt gebeurde, met dezelfde formule?",
        options={
            "one": "Ongeveer 1",
            "five": "Ongeveer 4,6",
            "infinite": "Oneindig veel",
        },
        reveal="**Ongeveer 4,6.** $-\\log(0{,}01) = \\log(100) \\approx 4{,}6$: bijna zeven keer zoveel als bij 0,5. Alleen een kans van precies 0 zou oneindig veel kosten.",
    )
    guess_surprise
    return (guess_surprise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Kijk naar de kans die het model gaf aan wat er echt gebeurde, en neem min de log daarvan. Een kans van 1 kost 0; hoe kleiner de kans, hoe meer het kost, zonder grens. $-\log(p)$ heet de **verrassing** van een uitkomst met kans $p$: een waarschijnlijke uitkomst verrast weinig, een onwaarschijnlijke veel. In de informatietheorie heet het de surprisal.

        Voor een voorbeeld met het label $y$, 1 voor spam en 0 voor geen spam, en de <span class="c-pred">voorspelde kans</span> $\hat{y}$ op spam, gaf het model $\hat{y}$ aan spam en $1 - \hat{y}$ aan geen spam. Eén formule werkt voor allebei, omdat een van de twee termen altijd 0 is. De **log loss** is de verrassing van wat er echt gebeurde, gemiddeld over alle $n$ voorbeelden:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log [[pred:\hat{y}_i]] + (1 - y_i) \log (1 - [[pred:\hat{y}_i]]) \right)"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **verrassing** | | $-\log(p)$: hoe verrassend een uitkomst met kans $p$ is | `-np.log(p)` |
        | **log loss** | | de gemiddelde verrassing van wat er echt gebeurde: een loss voor voorspelde kansen bij twee klassen | `loss` |
        """),
        L.watch_out("De log loss heeft de log nodig van de kans op wat er echt gebeurde: $\\hat{y}$ bij een spammail, maar $1 - \\hat{y}$ bij een mail die geen spam is. PyTorch noemt deze loss binary cross-entropy."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek: dat zet de <span class="c-pred">kans op spam</span> $\hat{y}$ die het model geeft aan een mail die spam is. De doorgetrokken lijn is de <span class="c-loss">log loss</span>, de stippellijn het <span class="c-loss">gekwadrateerde residu</span> van de mean squared error.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_p_cell():
    def _frame(p):
        log = -math.log(p)
        sq = (1 - p) ** 2
        return (C.picture(p, {"title": "De loss van één spammail", "p": "kans op spam ŷ", "loss": "loss",
                              "log": "log loss", "squared": "gekwadrateerd residu",
                              "alt": f"Bij een spammail: de log loss schiet omhoog als de kans op spam naar 0 gaat, terwijl het gekwadrateerde residu onder 1 blijft. Bij ŷ = {L.number(p, 2)} is de log loss {L.number(log, 2)} en het gekwadrateerde residu {L.number(sq, 2)}."}),
                f"""Bij <i>ŷ</i> = {L.number(p, 2)}: de log loss is −log({L.number(p, 2)}) = <span class="c-loss ea-num">{L.number(log, 2)}</span>, en het gekwadrateerde residu (1 − <i>ŷ</i>)<sup>2</sup> = <span class="c-loss ea-num">{L.number(sq, 2)}</span>.""")

    explore_p = L.explore("spam-p", "ŷ", _frame, start=0.5, low=0.01, high=0.99, step=0.01, role="pred", task="Schuif $\\hat{y}$ van 0,99 omlaag naar 0,01: waar liggen de twee losses dicht bij elkaar, en onder welke $\\hat{y}$ is de log loss meer dan 1?")
    explore_p
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Rond $\hat{y} = 1$ liggen de twee dicht bij elkaar. Onder ongeveer $\hat{y} = 0{,}37$ is de log loss meer dan 1, en daar komt het gekwadrateerde residu nooit. De log loss blijft stijgen: bij $\hat{y} = 0{,}01$ is hij 4,61, tegen 0,98.

    ### Recept

    Zo reken je de log loss van een model op een aantal voorbeelden uit:

    1. Neem voor elk voorbeeld de kans die het model gaf aan wat er echt gebeurde: $\hat{y}$ als $y = 1$, $1 - \hat{y}$ als $y = 0$.
    2. Neem min de log daarvan: de verrassing van elk voorbeeld.
    3. Neem het gemiddelde over alle voorbeelden: de log loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier mails: de eerste en de derde zijn spam, de andere twee niet. Het model gaf ze de kansen op spam 0,9, 0,2, 0,6 en 0,7. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_log_loss():
    import numpy as np

    _y = np.array([1, 0, 1, 0])
    _p = np.array([0.9, 0.2, 0.6, 0.7])
    # 1. de kans op wat er echt gebeurde: p bij spam, 1 - p bij geen spam
    _p_true = _y * _p + (1 - _y) * (1 - _p)
    print(_p_true)
    # 2. de verrassing van elke mail
    _surprise = -np.log(_p_true)
    print(_surprise)
    # 3. het gemiddelde: de log loss
    print(_surprise.mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_y * _p + (1 - _y) * (1 - _p)` pakt $\hat{y}$ voor de spammails en $1 - \hat{y}$ voor de andere, omdat een van de twee termen altijd 0 is: 0,9, 0,8, 0,6 en 0,3. De verrassingen zijn 0,11, 0,22, 0,51 en 1,20, en hun gemiddelde, de log loss, is 0,51. De vierde mail kost het meest: het model gaf 0,7 aan spam, en dat was hij niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_not_spam():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een mail die geen spam is
    """)
    return


@app.cell(hide_code=True)
def choice_not_spam_cell():
    choice_not_spam = L.choice(
        "not-spam",
        question="Een mail is geen spam, $y = 0$, en het model gaf hem de kans op spam 0,9. Wat is zijn log loss?",
        options={
            "spam": "$-\\log(0{,}9) \\approx 0{,}11$",
            "right": "$-\\log(0{,}1) \\approx 2{,}30$",
            "squared": "$0{,}9^2 = 0{,}81$",
            "negative": "$\\log(0{,}1) \\approx -2{,}30$",
        },
        correct="right",
        why={
            "spam": "Dat is de loss als de mail spam was geweest. Welke kans gaf het model aan wat er echt gebeurde? Zie stap 1 van het recept.",
            "squared": "Dat is het gekwadrateerde residu van de mean squared error. Zie de formule in *Het idee*.",
            "negative": "Een loss is nooit negatief: de log van een kans onder 1 is negatief, dus de loss is min dat. Zie stap 2 van het recept.",
            "right": "Het model gaf 1 − 0,9 = 0,1 aan geen spam, en dat gebeurde. $-\\log(0{,}1)$ is ongeveer 2,30: zeker en fout.",
        },
    )
    choice_not_spam
    return (choice_not_spam,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss van één voorbeeld

        Maak `one_loss(y, p)` af: hij krijgt het label `y`, 1 of 0, en `p`, de kans op klasse 1, en geeft de log loss van dat ene voorbeeld terug. Vervang de `...`. De check roept `one_loss(1, 0.9)` aan: dat moet ongeveer `0.105` geven. Hij probeert ook andere voorbeelden, met beide labels.
        """),
        L.task("Laat `one_loss` min de log teruggeven van de kans op wat er echt gebeurde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-loss", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-loss", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-loss",
        [
            "Stap 1 en 2 van het recept, voor één voorbeeld: welke kans hoort bij wat er gebeurde, en dan min de log daarvan.",
            "Gebruik een `if` op `y`. Bij `y == 1` gaf het model `p` aan wat er gebeurde. Wat gaf het bij `y == 0`?",
            "`if y == 1:` met `return -np.log(p)`, en onder `else:` `return -np.log(___)`.",
        ],
        why="Bij `y == 1` is de loss `-np.log(p)`, bij `y == 0` is hij `-np.log(1 - p)`. Voor `one_loss(1, 0.9)`: −log(0,9) ≈ 0,105.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Vijf mails tegelijk

        Vijf mails, met hun labels in `y` en de kansen op spam van het model in `p`. Zet hun log loss in `loss`, in één regel met de formule uit *Het idee*. Vervang de `...`. De laatste regel is `loss`, zodat de check hem ziet. De check probeert ook andere kansen.
        """),
        L.task("Zet de gemiddelde log loss van de vijf mails in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-loss", lines=7)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-loss", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-loss",
        [
            "Met arrays werkt `y * np.log(p)` voor alle mails tegelijk. Wat tel je erbij op voor de mails die geen spam zijn?",
            "De formule heeft twee termen binnen het gemiddelde, en een min ervoor. Welke NumPy-functie neemt het gemiddelde?",
            "`loss = -np.mean(y * np.log(p) + ___)`.",
        ],
        why="`y * np.log(p) + (1 - y) * np.log(1 - p)` is voor elke mail de log van de kans op wat er gebeurde; `-np.mean(...)` neemt het gemiddelde met een min ervoor: ongeveer 0,33.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl) mat de mean squared error hoe ver een getal ernaast zat; de log loss meet hoe ver een kans ernaast zit, en straft zekere fouten het hardst. In [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) gaf een model kansen; nu kun je zeggen hoe goed ze zijn. Later in de course doet cross-entropy hetzelfde voor veel klassen, en een language model wordt precies hierop getraind: de verrassing van het volgende woord. In de volgende les, *De gradient van log loss*, reken je uit hoe de log loss verandert met elke weight, zodat training hem kan verlagen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_not_spam, result_mean, result_one):
    L.wrap_up(choice_not_spam, result_one, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
