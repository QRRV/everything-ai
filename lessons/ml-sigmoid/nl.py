# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-sigmoid. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow classification

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-sigmoid", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl), [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) en [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_exp():
    quick_exp = L.choice(
        "quick-exp",
        question="Hoe groot is $e^{-z}$ als $z$ groot is, zoals 10?",
        options={
            "negative": "Negatief",
            "right": "Dicht bij 0, maar boven 0",
            "large": "Heel groot",
        },
        correct="right",
        why={
            "negative": "Is een macht van $e$ ooit negatief? Zie *Het idee* in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "large": "Dat is $e^{z}$. Wat doet een negatieve exponent? Zie *Het idee* in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "$e^{-10} = 1/e^{10}$, ongeveer 0,000045: piepklein, maar nooit 0 en nooit negatief.",
        },
    )
    quick_exp
    return (quick_exp,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="Welk van deze getallen kan een kans zijn?",
        options={
            "above": "1,3",
            "below": "−0,2",
            "right": "0,7",
        },
        correct="right",
        why={
            "above": "Een kans is een deel van alle pogingen. Kan een deel meer zijn dan alles? Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "below": "Een kans telt hoe vaak iets gebeurt. Kan dat minder zijn dan nooit? Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "right": "Een kans ligt tussen 0, nooit, en 1, altijd.",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def quick_check_task():
    quick_task = L.choice(
        "quick-task",
        question="Een model voorspelt of een mail spam is of niet. Wat voor taak is dat?",
        options={
            "right": "Classificatie, met twee klassen",
            "regression": "Regressie",
        },
        correct="right",
        why={
            "regression": "Is het label een getal, of een van een vaste set antwoorden? Zie het recept in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "right": "Het label is een van twee antwoorden, spam of geen spam: twee klassen.",
        },
    )
    quick_task
    return (quick_task,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een spamfilter moet zeggen hoe zeker het is: een kans tussen 0 en 1. Maar een model voor classificatie rekent eerst voor elke mail een score uit, en een score kan elk getal zijn: −7, 0,3 of 12. Hoe hoger, hoe meer de mail op spam lijkt. Latere lessen laten zien hoe een model aan zijn score komt; deze les gaat over de laatste stap: hoe maak je van elk getal een kans? Je hebt een functie nodig die van elk getal een getal tussen 0 en 1 maakt, die een hogere score een hogere kans geeft, en die 0,5 geeft bij een score van 0, want die leunt geen kant op.

    **Een voorbeeld uit het dagelijks leven.** Een quiz stelt een ja-of-neevraag. Zonder aanwijzing is het fifty-fifty. Elke aanwijzing voor ja maakt je zekerder, en elke aanwijzing ertegen minder zeker, maar geen enkele aanwijzing maakt je helemaal zeker: je komt steeds dichter bij 100% of bij 0%, en haalt het nooit.
    """)
    return


@app.cell(hide_code=True)
def predict_zero():
    guess_zero = L.guess(
        "zero",
        question="De functie van deze les is $\\sigma(z) = \\dfrac{1}{1 + e^{-z}}$. Wat is $\\sigma(0)$ volgens jou?",
        options={
            "zero": "0",
            "half": "0,5",
            "one": "1",
        },
        reveal="**0,5.** $e^{0} = 1$, dus $\\sigma(0) = 1 / (1 + 1) = 0{,}5$. Een score van 0 leunt geen kant op: de kans is even groot als die op het tegendeel.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **sigmoid** maakt van elk getal $z$ een getal tussen 0 en 1:
        """),
        L.formula(r"[[pred:\sigma(z)]] = \frac{1}{1 + e^{-z}}"),
        mo.md(r"""
        Bij een grote $z$ ligt $e^{-z}$ dicht bij 0, en $\sigma(z)$ dicht bij 1. Bij een heel negatieve $z$ is $e^{-z}$ enorm, en ligt $\sigma(z)$ dicht bij 0. Bij $z = 0$ is hij precies 0,5. Een hogere $z$ geeft altijd een hogere $\sigma(z)$, en hij komt nooit bij 0 of 1, al rondt een computer hem vanaf ongeveer $z = 37$ af op precies 1. Hij is ook symmetrisch: $\sigma(-z) = 1 - \sigma(z)$. In de statistiek heet hij de logistische functie.

        De score $z$ die erin gaat, heet de **logit**. Voor een spamfilter is $\sigma(z)$ de kans dat de mail spam is, en $1 - \sigma(z)$ de kans dat hij dat niet is. Boven 0,5 leunt het filter naar spam: dat gebeurt precies als de logit boven 0 ligt.

        Terug kan ook: de logit van een kans $p$ is $z = \log\left(\frac{p}{1 - p}\right)$. $\frac{p}{1 - p}$ heet de odds: hoeveel keer zo waarschijnlijk de klasse is als niet de klasse.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **sigmoid**, $\sigma(z)$ | sigma van z, de sigmoid | $1 / (1 + e^{-z})$: maakt van elk getal een getal tussen 0 en 1 | `1 / (1 + np.exp(-z))` |
        | **logit**, $z$ | | de score die de sigmoid in gaat: elk getal, positief of negatief | `z` |
        """),
        L.watch_out("Logit betekent twee dingen. In deze course is het de score vóór de sigmoid. De statistiek gebruikt het voor de weg terug, $\\log(p / (1 - p))$, ook log-odds genoemd. Bij twee klassen geven ze hetzelfde getal: de log-odds van $\\sigma(z)$ is weer $z$."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek: dat zet de logit $z$. Het punt laat de <span class="c-pred">kans</span> $\sigma(z)$ zien die de sigmoid ervan maakt.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        e = math.exp(-z)
        p = C.sigmoid(z)
        return (C.picture(z, {"title": "De sigmoid", "z": "logit z", "p": "kans",
                              "alt": f"De sigmoid, een S-vormige kromme van bijna 0 links tot bijna 1 rechts, door 0,5 bij z = 0. Bij z = {L.number(z, 1)} staat een punt op {L.number(p, 3)}."}),
                f"""Bij <i>z</i> = {L.number(z, 1)}: <i>e</i><sup>−<i>z</i></sup> = {L.number(e, 4)}, dus <i>σ</i>(<i>z</i>) = 1 / (1 + {L.number(e, 4)}) = <span class="c-pred ea-num">{L.number(p, 3)}</span>.""")

    explore_z = L.explore("logit-z", "z", _frame, start=1, low=-6, high=6, step=0.5, role="", task="Vanaf welke $z$ is $\\sigma(z)$ boven 0,99, en tot welke $z$ onder 0,01?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Vanaf $z = 5$ is $\sigma(z)$ boven 0,99, en tot en met $z = -5$ onder 0,01. Daar, ver van $z = 0$, is de kromme bijna vlak: een stap van 0,5 in $z$ verandert de kans met minder dan 0,005. Rond $z = 0$ is hij het steilst: één stap verandert de kans daar met 0,12.

    ### Recept

    Zo maak je van een score een kans:

    1. Neem de logit $z$: elk getal.
    2. Reken $e^{-z}$ uit: `np.exp(-z)`.
    3. Tel er 1 bij op, en deel 1 erdoor: $\sigma(z)$ = `1 / (1 + np.exp(-z))`.
    4. Lees het als de kans op de klasse: boven 0,5 leunt het model ernaartoe, onder 0,5 ervan af.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf mails met de logits −3, −1, 0, 1 en 3, allemaal tegelijk met NumPy. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_sigmoid():
    import numpy as np

    # 1. de logits van vijf mails
    _z = np.array([-3, -1, 0, 1, 3])
    # 2. e tot de macht -z
    _e = np.exp(-_z)
    print(_e)
    # 3. 1 gedeeld door 1 plus dat
    _p = 1 / (1 + _e)
    print(_p)
    # 4. bij welke mails het filter naar spam leunt
    print(_p > 0.5)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $e^{-z}$ gaat van 20,1 bij $z = -3$ omlaag naar 0,05 bij $z = 3$. De kansen gaan van 0,047 tot 0,953, met 0,5 in het midden. De mails met de logits −1 en 1 krijgen 0,269 en 0,731: samen 1, omdat de sigmoid symmetrisch is. Alleen de mails met een logit boven 0 krijgen een kans boven 0,5; de mail met logit 0 zit precies op 0,5, dus niet erboven.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minus_two():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een logit van −2
    """)
    return


@app.cell(hide_code=True)
def choice_minus_two_cell():
    choice_minus_two = L.choice(
        "minus-two",
        question="Een spamfilter geeft een mail de logit −2. Welke kans op spam geeft dat?",
        options={
            "plus": "Ongeveer 0,88",
            "right": "Ongeveer 0,12",
            "negative": "−2: een logit is al een kans",
            "half": "0,5",
        },
        correct="right",
        why={
            "plus": "Dat is $\\sigma(2)$. Ligt de logit hier boven of onder 0? Zie de figuur.",
            "negative": "Een kans ligt tussen 0 en 1, en een logit kan elk getal zijn. Zie de tabel met nieuwe woorden.",
            "half": "0,5 hoort bij een logit van 0. Wat doet een negatieve logit? Zie de figuur.",
            "right": "$e^{2}$ is ongeveer 7,39, dus $\\sigma(-2) = 1 / (1 + 7{,}39) \\approx 0{,}12$. Het filter houdt de mail dus eerder voor geen spam.",
        },
    )
    choice_minus_two
    return (choice_minus_two,)


@app.cell(hide_code=True)
def text_exercise_sigmoid():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De sigmoid als functie

        Maak `sigmoid(z)` af: hij krijgt een logit en geeft de kans terug. Vervang de `...`. De check roept `sigmoid(0)` aan: dat moet `0.5` geven. Hij probeert ook andere logits, positief en negatief.
        """),
        L.task("Laat `sigmoid` de sigmoid van `z` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sigmoid():
    box_sigmoid = L.code_box("sigmoid", lines=6)
    box_sigmoid
    return (box_sigmoid,)


@app.cell(hide_code=True)
def check_sigmoid(box_sigmoid):
    result_sigmoid = L.check("sigmoid", box_sigmoid.value)
    result_sigmoid
    return (result_sigmoid,)


@app.cell(hide_code=True)
def hints_sigmoid():
    L.hints(
        "sigmoid",
        [
            "De formule uit *Het idee*, in Python: stap 2 en 3 van het recept.",
            "Welke NumPy-functie geeft $e$ tot een macht? Let op het minteken, en op welk deel de onderkant van de breuk is.",
            "`return 1 / (1 + ___)`.",
        ],
        why="`np.exp(-z)` is $e^{-z}$, en `1 / (1 + np.exp(-z))` deelt 1 door 1 plus dat. Voor `z = 0`: 1 / (1 + 1) = 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_logit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Terug naar de logit

        Een filter geeft een mail een kans op spam van 0,9. Welke logit had die mail? Zet hem in `z`, met de weg terug uit *Het idee*. Vervang de `...`. De laatste regel is `z`, zodat de check hem ziet. De check probeert ook een andere kans.
        """),
        L.task("Zet de logit van `p` in `z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logit():
    box_logit = L.code_box("back-to-logit", lines=6)
    box_logit
    return (box_logit,)


@app.cell(hide_code=True)
def check_logit(box_logit):
    result_logit = L.check("back-to-logit", box_logit.value)
    result_logit
    return (result_logit,)


@app.cell(hide_code=True)
def hints_logit():
    L.hints(
        "back-to-logit",
        [
            "Eerst de odds: hoeveel keer zo waarschijnlijk spam is als geen spam. Dan de log daarvan.",
            "De kans op geen spam is `1 - p`. Welke NumPy-functie geeft de log met grondtal $e$?",
            "`z = np.log(p / ___)`.",
        ],
        why="De odds zijn 0,9 / 0,1 = 9, en de log van 9 is ongeveer 2,2. Controleer het: $\\sigma(2{,}2)$ is ongeveer 0,9.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl) was $e^{x}$ altijd positief en groeide hij steeds sneller; de sigmoid gebruikt dat om elk getal tussen 0 en 1 te persen, en de log brengt je terug. In [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) lag een kans tussen 0 en 1; nu kan een model er een geven. In [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl) was een label een van een vaste set klassen; bij twee klassen geeft de sigmoid de kans op een van de twee. In de volgende les, *Logistische regressie en de decision boundary*, rekent een model de logit uit met de features, en maakt de sigmoid er een kans van.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_minus_two, result_logit, result_sigmoid):
    L.wrap_up(choice_minus_two, result_sigmoid, result_logit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
