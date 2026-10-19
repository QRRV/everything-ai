# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-log-loss-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss, regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-log-loss-gradient", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) en [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loss():
    quick_loss = L.choice(
        "quick-loss",
        question="Een mail is spam, $y = 1$, en het model geeft hem de kans op spam 0,25. Wat is zijn log loss?",
        options={
            "other": "$-\\log(0{,}75) \\approx 0{,}29$",
            "right": "$-\\log(0{,}25) \\approx 1{,}39$",
            "squared": "$0{,}75^2 \\approx 0{,}56$",
        },
        correct="right",
        why={
            "other": "Dat is de loss van een mail die geen spam is. Welke kans gaf het model aan wat er echt gebeurde? Zie het recept in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "squared": "Dat is het gekwadrateerde residu. Zie de formule in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "right": "Het model gaf 0,25 aan spam, en dat gebeurde: $-\\log(0{,}25) \\approx 1{,}39$.",
        },
    )
    quick_loss
    return (quick_loss,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$u = 2x$ en $y = u^3$. Wat is $\\frac{dy}{dx}$ bij $x = 1$?",
        options={
            "outer": "12",
            "inner": "2",
            "right": "24",
        },
        correct="right",
        why={
            "outer": "Dat is alleen de buitenste helling, $3u^2$ bij $u = 2$. Waarmee vermenigvuldig je die? Zie *Het idee* in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "inner": "Dat is alleen de binnenste helling. Waarmee vermenigvuldig je die? Zie *Het idee* in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "right": "De buitenste helling $3u^2 = 12$ bij $u = 2$, keer de binnenste helling 2: 24.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="Ergens is de gradient van een loss $(-2;\\ 0{,}5)$: $\\frac{\\partial \\mathcal{L}}{\\partial w} = -2$ en $\\frac{\\partial \\mathcal{L}}{\\partial b} = 0{,}5$. Welke kleine verandering verlaagt de loss?",
        options={
            "right": "Een grotere $w$ en een kleinere $b$",
            "both-up": "Een grotere $w$ en een grotere $b$",
            "uphill": "Een kleinere $w$ en een grotere $b$",
        },
        correct="right",
        why={
            "both-up": "Een positieve helling betekent dat de loss stijgt als $b$ groeit. Welke kant op daalt hij? Zie *Het idee* in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "uphill": "Dat is de richting van de gradient zelf: bergop. Zie *Het idee* in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "right": "Ga tegen elke helling in: $w$ omhoog, omdat zijn helling negatief is, en $b$ omlaag, omdat zijn helling positief is.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om logistische regressie met gradient descent te trainen, heb je de helling van de log loss nodig voor elke weight en voor de bias term. Elke weight een duwtje geven werkt, maar een model met een miljoen weights zou dan per stap een miljoen duwtjes nodig hebben. De kettingregel geeft elke helling uit één formule. Bij log loss met een sigmoid blijkt die formule verrassend eenvoudig: de kans min het label, keer de feature.

    **Een voorbeeld uit het dagelijks leven.** Om te weten wat een taxirit kost, maak je niet elke rit om het uit te proberen. Het tarief, een starttarief plus een prijs per kilometer, geeft de kosten van elke rit tegelijk.
    """)
    return


@app.cell(hide_code=True)
def predict_correction():
    guess_correction = L.guess(
        "correction",
        question="Twee spammails, met dezelfde features. Het model geeft de ene de kans op spam 0,9, de andere 0,1. Bij welke verandert de log loss volgens jou sneller als een weight verandert?",
        options={
            "ninety": "De mail met 0,9",
            "ten": "De mail met 0,1",
            "same": "Allebei even snel",
        },
        reveal="**De mail met 0,1.** De helling van de log loss voor een weight blijkt $(\\hat{y} - y) \\cdot x$ te zijn: de kans min het label, keer de feature. Bij 0,1 is dat $0{,}1 - 1 = -0{,}9$ keer de feature, bij 0,9 maar $-0{,}1$ keer: negen keer kleiner. Waar het model ver ernaast zit, komt de grote bijsturing.",
    )
    guess_correction
    return (guess_correction,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De log loss van één voorbeeld is in drie stappen gebouwd: de logit $z = \mathbf{w} \cdot \mathbf{x} + b$, de kans $\hat{y} = \sigma(z)$, en de loss $\mathcal{L} = -\left( y \log \hat{y} + (1 - y) \log(1 - \hat{y}) \right)$. De kettingregel vermenigvuldigt de hellingen van de stappen, nu drie in plaats van twee. Twee van die hellingen neem je hier aan: die van $\log$ en van $\sigma$ volgen uit regels voor exp en log die een latere les afleidt.

        - **De helling van de loss naar $\hat{y}$.** De helling van $\log(p)$ is $1/p$. In $\log(1 - \hat{y})$ heeft de binnenste stap $1 - \hat{y}$ de helling $-1$, dus volgens de kettingregel is de helling daarvan $-\frac{1}{1 - \hat{y}}$. Met het min-teken voor de loss is de helling van $\mathcal{L}$ dan $-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$.
        - **De helling van de sigmoid.** De helling van $\sigma(z)$ is $\sigma(z)\,(1 - \sigma(z))$, hier $\hat{y}(1 - \hat{y})$.
        - **De helling van de logit naar een weight.** $z$ is een inproduct, dus als een weight $w_j$ 1 groter wordt, groeit $z$ met $x_j$: de helling is $x_j$. Voor de bias term is hij 1.

        Vermenigvuldig de eerste twee, en de breuken vallen weg:
        """),
        L.formula(r"\left(-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}\right) \hat{y}(1 - \hat{y}) = -y(1 - \hat{y}) + (1 - y)\hat{y}"),
        mo.md(r"""
        Werk de haakjes uit, en ook de twee termen met $y\hat{y}$ vallen tegen elkaar weg:
        """),
        L.formula(r"-y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y"),
        mo.md(r"""
        Met de derde erbij:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = ([[pred:\hat{y}]] - y)\, x_j \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = [[pred:\hat{y}]] - y"),
        mo.md(r"""
        Bij veel voorbeelden is de log loss een gemiddelde, dus is elke helling het gemiddelde van deze hellingen over de voorbeelden. De slider hieronder controleert de helling van $\sigma$ met een duwtje, zoals in [De afgeleide als helling](/course/open/math-slope?lang=nl); het uitgewerkte voorbeeld controleert de hele formule.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **afgeleide van de sigmoid**, $\sigma(z)(1 - \sigma(z))$ | sigma van z keer één min sigma van z | de helling van de sigmoid bij $z$, hoe snel hij daar stijgt: 0,25 bij $z = 0$, bijna 0 ver daarvandaan | `p * (1 - p)` |
        | **gradient van de log loss**, $(\hat{y} - y)\, x_j$ | y-dakje min y, keer x j | de helling van de log loss van logistische regressie naar de weight $w_j$: de kans min het label, keer de feature | `(p - y) * x` |
        """),
        L.watch_out("De helling van de sigmoid valt weg tegen de helling van de log; daarom is de gradient zo eenvoudig. Met de mean squared error op een kans blijft hij staan: een model dat zeker en fout is, met $\\hat{y}(1 - \\hat{y})$ dicht bij 0, zou dan nauwelijks leren."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek: dat zet de logit $z$. De <span class="c-grad">groene lijn</span> heeft de helling van de sigmoid op dat punt, $\hat{y}(1 - \hat{y})$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        p = C.sigmoid(z)
        slope = p * (1 - p)
        nudged = (C.sigmoid(z + 0.001) - p) / 0.001
        return (C.picture(z, {"title": "De helling van de sigmoid", "z": "logit z", "p": "kans ŷ",
                              "alt": f"De sigmoid, met een punt bij z = {L.number(z, 1)} en een korte groene lijn met de helling daar, {L.number(slope, 3)}. De sigmoid is het steilst bij z = 0 en bijna vlak ver naar links en rechts."}),
                f"""Bij <i>z</i> = {L.number(z, 1)}: <i>ŷ</i> = <span class="c-pred ea-num">{L.number(p, 3)}</span>, en de helling <i>ŷ</i>(1 − <i>ŷ</i>) = <span class="c-grad ea-num">{L.number(slope, 4)}</span>. Een duwtje van 0,001 aan <i>z</i> geeft <span class="c-grad ea-num">{L.number(nudged, 4)}</span>.""")

    explore_z = L.explore("slope-z", "z", _frame, start=0, low=-5, high=5, step=0.5, role="", task="Hoeveel keer kleiner is de helling $\\hat{y}(1 - \\hat{y})$ bij $z = -5$ dan bij $z = 0$?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij $z = -5$ is de helling 0,0066: ongeveer 38 keer kleiner dan 0,25 bij $z = 0$, en het duwtje geeft bijna hetzelfde, 0,0067. Een spammail met die logit krijgt $\hat{y} = 0{,}007$: het model is zeker en fout. Log loss geeft hem toch de helling $\hat{y} - y \approx -0{,}99$ voor de bias term, bijna de grootste die er is; met de mean squared error blijft deze helling van de sigmoid, 0,0066, in zijn gradient staan, en zou het model nauwelijks van deze mail leren.

    ### Recept

    Zo reken je de gradient van de log loss van logistische regressie uit:

    1. Reken de kans $\hat{y}$ van elk voorbeeld uit: de sigmoid van de logit.
    2. Trek van elke kans het label af: $\hat{y} - y$.
    3. De helling voor elke weight is het gemiddelde van $\hat{y} - y$ keer zijn feature; voor de bias term het gemiddelde van $\hat{y} - y$ zelf.
    4. Controleer één helling met een duwtje: verander die weight met een piepkleine $h$, en deel de verandering van de loss door $h$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén spammail, $y = 1$, met de features 2 en 1, en een model met de weights 0,5 en −1 en de bias term 0. Eerst de gradient uit de formule, dan de controle met een duwtje aan de eerste weight. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np

    _x = np.array([2, 1])
    _y = 1
    _w = np.array([0.5, -1])
    _b = 0

    def _loss(w, b):
        _p = 1 / (1 + np.exp(-(np.dot(w, _x) + b)))
        return -(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

    # 1. de kans
    _p = 1 / (1 + np.exp(-(np.dot(_w, _x) + _b)))
    print(_p)
    # 2. de kans min het label
    print(_p - _y)
    # 3. de hellingen voor de twee weights, en voor de bias term
    print((_p - _y) * _x, _p - _y)
    # 4. de controle: geef de eerste weight een duwtje van h
    _h = 0.001
    print((_loss(_w + np.array([_h, 0]), _b) - _loss(_w, _b)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De logit is $0{,}5 \cdot 2 - 1 \cdot 1 + 0 = 0$, dus $\hat{y} = 0{,}5$ en $\hat{y} - y = -0{,}5$. De hellingen zijn $-0{,}5 \cdot 2 = -1$ en $-0{,}5 \cdot 1 = -0{,}5$ voor de weights, en $-0{,}5$ voor de bias term; met één voorbeeld is het gemiddelde uit stap 3 die ene waarde zelf. Het duwtje aan de eerste weight geeft $-0{,}9995$: dicht bij $-1$, dus de formule doorstaat de controle. Alle hellingen zijn negatief: maak de weights en de bias term een beetje groter, en de loss van deze spammail daalt.
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
        "not-spam-slope",
        question="Een mail is geen spam, $y = 0$. Het model geeft hem $\\hat{y} = 0{,}8$, en zijn eerste feature is $x_1 = 3$. Wat is $\\frac{\\partial \\mathcal{L}}{\\partial w_1}$ voor deze mail?",
        options={
            "spam": "$-0{,}6$",
            "no-x": "$0{,}8$",
            "right": "$2{,}4$",
            "sign": "$-2{,}4$",
        },
        correct="right",
        why={
            "spam": "Dat is $(0{,}8 - 1) \\cdot 3$, alsof de mail spam was. Wat is $y$ hier? Zie de formule in *Het idee*.",
            "no-x": "Dat is alleen $\\hat{y} - y$, de helling voor de bias term. Wat komt erbij voor de weight van feature 1? Zie de formule in *Het idee*.",
            "sign": "In de formule staat $\\hat{y} - y$, de kans min het label, niet $y - \\hat{y}$. Zie de formule in *Het idee*.",
            "right": "$(0{,}8 - 0) \\cdot 3 = 2{,}4$: positief, dus een kleinere $w_1$ verlaagt de loss. Het model gaf te veel aan spam.",
        },
    )
    choice_not_spam
    return (choice_not_spam,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De gradient van één voorbeeld

        Maak `one_gradient(x, y, w, b)` af. Hij krijgt de features `x` van één voorbeeld, het label `y`, de weights `w` en de bias term `b`, en geeft de hellingen van de log loss voor de weights terug, als array. Het vak rekent de kans `p` al uit. Vervang de `...`. De check roept `one_gradient([2, 1], 1, [0.5, -1], 0)` aan, de mail uit het uitgewerkte voorbeeld: dat moet `array([-1. , -0.5])` geven. Hij probeert ook andere voorbeelden.
        """),
        L.task("Laat `one_gradient` de hellingen van de log loss voor de weights teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-gradient", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-gradient", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-gradient",
        [
            "De formule uit *Het idee*: de kans min het label, keer de features.",
            "Met arrays geeft één vermenigvuldiging elke helling tegelijk. Welke volgorde: `p - y` of `y - p`?",
            "`return (p - ___) * ___`.",
        ],
        why="`(p - y) * x` vermenigvuldigt `p - y` met elke feature tegelijk. Voor de mail: $-0{,}5 \\cdot [2; 1] = [-1; -0{,}5]$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De gradient voor alle leerlingen

        De leerlingen van [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) van wie de uitslag bekend is, met het model uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl): de weights 1 en 0,5, en de bias term −8. Het vak rekent de kans `p` van elke leerling uit. Zet de drie hellingen van de gemiddelde log loss in `grad`, als één array: voor de weight van `hours`, de weight van `sleep`, en de bias term. Vervang de `...`. De laatste regel is `grad`, zodat de check hem ziet. De check probeert ook een andere bias term.
        """),
        L.task("Zet de gradient van de gemiddelde log loss in `grad`: drie hellingen in één array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-gradient", lines=11)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-gradient", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-gradient",
        [
            "Stap 2 en 3 van het recept, voor alle leerlingen tegelijk: eerst de kans min het label, dan een gemiddelde per helling.",
            "`p - passed` geeft één getal per leerling. Welke feature hoort bij welke helling, en wat krijgt de bias term?",
            "`grad = np.array([np.mean((p - passed) * hours), np.mean(___), np.mean(___)])`.",
        ],
        why="`p - passed` is de kans min het label van elke leerling; keer `hours` en `sleep`, en gemiddeld, geeft dat de hellingen van de twee weights; het gemiddelde van `p - passed` zelf is de helling van de bias term: ongeveer −0,33, −0,22 en −0,04.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) vermenigvuldigde je de hellingen van de stappen van één functie; hier zijn de stappen de logit, de sigmoid en de log, en valt de helling van de sigmoid weg. In [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) vormde een helling per parameter de gradient; nu heb je er een formule voor, gecontroleerd met een duwtje. In [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl) mat je hoe ver de kansen ernaast zaten; nu weet je welke kant je het model op moet sturen. In de volgende les, *Logistische regressie trainen*, gebruikt gradient descent precies deze gradient om het model te trainen.
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
