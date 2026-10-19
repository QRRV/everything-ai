# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-exp-log. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow vectorization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-exp-log", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) en [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_power():
    quick_power = L.choice(
        "quick-power",
        question="Wat geeft `np.array([1, 2, 3]) ** 2`?",
        options={
            "right": "`array([1, 4, 9])`",
            "times": "`array([2, 4, 6])`",
            "sum": "`14`",
        },
        correct="right",
        why={
            "times": "Dat is `* 2`. Wat doet `**`? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "sum": "Dat telt de kwadraten ook nog op. Doet `**` dat? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "right": "`** 2` kwadrateert elk getal van de array: 1, 4 en 9.",
        },
    )
    quick_power
    return (quick_power,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="Wat tekent `plt.plot([0, 1, 2], [0, 1, 4])`?",
        options={
            "bars": "Drie staven, zo hoog als 0, 1 en 4",
            "points": "Drie losse punten, zonder lijn",
            "right": "Een lijn door de punten (0, 0), (1, 1) en (2, 4)",
        },
        correct="right",
        why={
            "bars": "Staven komen uit een andere functie. Welke? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "points": "Losse punten komen uit een andere functie. Welke? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "right": "`plt.plot` tekent een lijngrafiek: de eerste x hoort bij de eerste y, en een lijn verbindt de punten.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige dingen groeien door optellen: elke keer een stap van dezelfde grootte. Andere groeien door vermenigvuldigen: elke keer een factor, en dan worden de getallen heel snel enorm. Het omgekeerde gebeurt ook: vermenigvuldig veel kansen, en de uitkomst wordt zo klein dat een computer hem afrondt naar 0. De exponent beschrijft groei door vermenigvuldigen, en de logaritme maakt van vermenigvuldigen optellen. Later in de course gebruikt een model voor classificatie ze allebei: om een score in een kans om te zetten, en om zijn fouten te tellen.

    **Een voorbeeld uit het dagelijks leven.** Een vel papier is ongeveer 0,1 millimeter dik. Vouw het dubbel, en het is 0,2; vouw nog eens, 0,4. Elke vouw verdubbelt de dikte.
    """)
    return


@app.cell(hide_code=True)
def predict_folds():
    guess_folds = L.guess(
        "fold",
        question="Je vouwt dat vel van 0,1 millimeter 30 keer dubbel, als dat zou kunnen. Hoe dik wordt het volgens jou?",
        options={
            "mm": "Ongeveer 3 millimeter",
            "m": "Ongeveer 10 meter",
            "km": "Meer dan 100 kilometer",
        },
        reveal="**Meer dan 100 kilometer.** 30 keer vouwen verdubbelt de dikte 30 keer: 0,1 millimeter keer $2^{30}$, en $2^{30}$ is ongeveer 1,07 miljard. Dat is ongeveer 107 kilometer. 3 millimeter zou 30 keer 0,1 millimeter optellen zijn; verdubbelen groeit veel sneller.",
    )
    guess_folds
    return (guess_folds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een macht vermenigvuldigt een getal, het grondtal, met zichzelf: $2^3 = 2 \cdot 2 \cdot 2 = 8$. Het verhoogde getal, de **exponent**, zegt hoe vaak; in Python is dat `2 ** 3`. Eén stap hoger in de exponent vermenigvuldigt nog een keer met 2, en daarom groeien machten steeds sneller. Elk getal tot de macht 0 is 1, en een negatieve exponent deelt: $2^{-3} = 1/8$.

        AI gebruikt vooral één grondtal: het getal $e$, ongeveer 2,718. Waarom juist dit getal, komt in een latere les terug; voor nu is het een grondtal zoals 2 of 10. $e^x$, ook geschreven als $\exp(x)$, is altijd positief, is 1 bij $x = 0$, en groeit steeds sneller.

        De **logaritme** gaat de andere kant op: hij geeft de exponent die je nodig hebt. $\log_2(8) = 3$, want $2^3 = 8$. Zonder grondtal betekent log in deze course altijd grondtal $e$: $\log(e^x) = x$, en $e^{\log(x)} = x$. Op school schrijf je daarvoor ln, en log voor grondtal 10; in deze course en in NumPy is log de natuurlijke logaritme: `np.log` is jouw ln, en `np.log10` is de log van school. Log is zo nuttig omdat hij van vermenigvuldigen optellen maakt. Vermenigvuldig je twee machten van hetzelfde grondtal, dan tel je hun exponenten op: $2^2 \cdot 2^3 = 2^{2+3}$, twee keer een factor 2 en dan nog drie. De log geeft die exponent, dus:
        """),
        L.formula(r"\log(a \cdot b) = \log(a) + \log(b)"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **exponent** | | het verhoogde getal in een macht: hoe vaak het grondtal met zichzelf wordt vermenigvuldigd | `2 ** 3` |
        | $e$ | e | het getal 2,718…, het grondtal dat AI het meest gebruikt | `np.e` |
        | $e^x$, $\exp(x)$ | e-macht x, exp van x | $e$ tot de macht $x$; altijd positief | `np.exp(x)` |
        | **logaritme**, $\log(x)$ | log van x | de exponent die $x$ geeft; in deze course, zonder grondtal, grondtal $e$: het omgekeerde van exp | `np.log(x)` |
        | $\log_2(x)$, $\log_{10}(x)$ | log met grondtal 2 van x | de exponent van 2, of van 10, die $x$ geeft | `np.log2(x)`, `np.log10(x)` |
        """),
        L.watch_out("De log bestaat alleen voor getallen boven 0: geen macht van $e$ is 0 of negatief. `np.log(0)` geeft `-inf`, min oneindig, en de log van een negatief getal geeft `nan`, *not a number*: geen getal. Allebei met een waarschuwing."),
        mo.md(r"""
        Schuif met de slider. Het punt $(x, e^x)$ ligt op exp, en het punt $(e^x, x)$ op log: dezelfde twee getallen, omgewisseld. Dat doet een omgekeerde functie, en daardoor zijn de twee krommen elkaars spiegelbeeld in de stippellijn, waar $y = x$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_x_cell():
    def _frame(x):
        y = math.exp(x)
        return (C.picture(x, {"title": "exp en log, elkaars spiegelbeeld",
                              "alt": f"De krommen van exp en log, elkaars spiegelbeeld in de stippellijn y = x. exp stijgt steeds sneller en ligt altijd boven 0; log stijgt steeds langzamer en bestaat alleen rechts van 0. Een punt op ({L.number(x, 2)}; {L.number(y, 3)}) op exp, en zijn spiegelbeeld op ({L.number(y, 3)}; {L.number(x, 2)}) op log."}),
                f"""Bij <i>x</i> = <span class="ea-num">{L.number(x, 2)}</span>: <i>e</i><sup><i>x</i></sup> = <span class="ea-num">{L.number(y, 3)}</span>, en log(<span class="ea-num">{L.number(y, 3)}</span>) = <span class="ea-num">{L.number(x, 2)}</span>. log maakt exp ongedaan.""")

    explore_x = L.explore("exp-x", "x", _frame, start=1, low=-2, high=2, step=0.25, role="", task="Schuif $x$ tot onder 0: tussen welke getallen ligt $e^x$ dan, en is de log ervan positief of negatief?")
    explore_x
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Voor $x$ onder 0 ligt $e^x$ tussen 0 en 1: het punt op exp ligt net boven de x-as. Zijn spiegelbeeld op log ligt onder de x-as: de log van een getal tussen 0 en 1 is negatief.

    ### Recept

    Zo werk je met machten en logaritmen:

    1. Een macht: het grondtal, `**`, en de exponent; $e^x$ is `np.exp(x)`.
    2. Een logaritme: vraag welke exponent het getal geeft. `np.log` voor grondtal $e$, `np.log2` en `np.log10` voor grondtal 2 en 10.
    3. Controleer met het omgekeerde: `np.exp(np.log(x))` geeft `x` terug.
    4. Een product van veel getallen: tel in plaats daarvan hun logs op.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Machten van 2, een logaritme die er een ongedaan maakt, grondtal $e$, en een product als som van logs. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_exp_log():
    import numpy as np

    # 1. machten van 2: één stap hoger in de exponent vermenigvuldigt nog een keer met 2
    _n = np.arange(11)
    print(2 ** _n)
    # 2. welke exponent van 2 geeft 1024?
    print(np.log2(1024))
    # 1. en 2. grondtal e: e zelf, en de log van e tot de macht 3
    print(np.exp(1), np.log(np.exp(3)))
    # 3. het omgekeerde: exp van de log van 5
    print(np.exp(np.log(5)))
    # 4. de log van een product, en de som van de logs
    print(np.log(0.5 * 0.2), np.log(0.5) + np.log(0.2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De machten van 2 gaan 1, 2, 4, 8 en zo verder tot 1024: tien keer verdubbelen. `np.log2(1024)` geeft die 10 terug. `np.exp(1)` is $e$, 2,718…, en de log van $e^3$ is 3. `np.exp(np.log(5))` geeft 4.999999999999999: 5, met een afrondingsfout in het laatste cijfer, zoals computers die maken met kommagetallen. En de log van 0,5 × 0,2 = 0,1 is hetzelfde als de log van 0,5 plus de log van 0,2: −2,30.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_negative():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke is negatief?
    """)
    return


@app.cell(hide_code=True)
def choice_negative_cell():
    choice_negative = L.choice(
        "negative",
        question="Welk van deze getallen is negatief?",
        options={
            "exp-neg": "$e^{-3}$",
            "right": r"$\log(0{,}5)$",
            "log5": r"$\log(5)$",
            "exp0": "$e^{0}$",
        },
        correct="right",
        why={
            "exp-neg": "Een negatieve exponent deelt: $e^{-3} = 1/e^3$, klein, maar boven 0. Zie *Het idee*.",
            "log5": "Welke exponent van $e$ geeft 5? $e^1$ is maar 2,718, dus meer dan 1. Zie de figuur.",
            "exp0": "Elk getal tot de macht 0 is 1. Zie *Het idee*.",
            "right": "$e$ tot de macht 0 is 1, dus voor 0,5, onder 1, heb je een negatieve exponent nodig: $\\log(0{,}5) \\approx -0{,}69$.",
        },
    )
    choice_negative
    return (choice_negative,)


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vouwen tot een kilometer

        Weer het vel van 0,1 millimeter. Hoe vaak moet je het vouwen om het een kilometer dik te maken: 1.000.000 millimeter? Het vak heeft `thickness` en `height`, allebei in millimeter. Zet het aantal keer vouwen in `folds` precies zoals een logaritme het geeft, met alle cijfers achter de komma; rond het niet af op hele keren. Vervang de `...`. De laatste regel is `folds`, zodat de check het ziet. De check probeert ook een andere hoogte.
        """),
        L.task("Zet in `folds` hoe vaak je moet vouwen om van `thickness` tot `height` te komen, uitgerekend met een logaritme."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=7)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "Met welke factor moet het papier groeien? En hoeveel keer verdubbelen geeft die factor?",
            "De factor is `height / thickness`. Welke logaritme telt het aantal keer verdubbelen? Zie stap 2 van het recept.",
            "`folds = np.___(height / thickness)`.",
        ],
        why="`height / thickness` is de factor, 10.000.000, en `np.log2` geeft de exponent van 2 die hem maakt: ongeveer 23,3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een product als som van logs

        Maak `log_product(ps)` af: de log van het product van een lijst getallen boven 0, uitgerekend als de som van hun logs, zodat je nooit vermenigvuldigt. Vervang de `...`. De check roept `log_product([0.5, 0.2, 0.1])` aan: dat moet ongeveer `-4.605` geven, de log van 0,01. Hij probeert ook andere lijsten, ook met heel kleine getallen zoals `1e-200`: zo schrijft Python $10^{-200}$, 1 gedeeld door 10, tweehonderd keer.
        """),
        L.task("Laat `log_product` de log van elk getal optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("log-product", lines=8)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("log-product", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "log-product",
        [
            "De log van een product is de som van de logs: zie de formule in *Het idee*.",
            "Een loop over de getallen. Wat tel je voor elk getal bij `total` op?",
            "`for p in ps:` en daarin `total = total + ___`.",
        ],
        why="Elke ronde telt `np.log(p)` bij `total` op. Voor `[0.5, 0.2, 0.1]`: −0,693 − 1,609 − 2,303 = −4,605, de log van 0,5 × 0,2 × 0,1 = 0,01.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) verhief `**` elk getal van een array tot een macht; `np.exp` en `np.log` werken ook op een hele array tegelijk. Later in de course zet exp de score van een model om in een kans, en maakt log van het product van veel kansen een som, zoals in oefening 3. In de volgende les, *Logaritmische schalen en machtswetten*, teken je grafieken waarvan de as in factoren loopt in plaats van in stappen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_negative, result_folds, result_product):
    L.wrap_up(choice_negative, result_folds, result_product)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
