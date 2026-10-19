# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-scaling-laws. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-scaling-laws", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) en [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="Hoe ziet $y = 5 \\cdot x^{-0{,}5}$ eruit op een log-log plot?",
        options={
            "right": "Een rechte lijn met helling $-0{,}5$",
            "five": "Een rechte lijn met helling 5",
            "curve": "Een kromme die naar rechts afvlakt",
        },
        correct="right",
        why={
            "five": "De 5 bepaalt de hoogte van de lijn, $\\log 5$; de exponent is de helling. Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "curve": "Op een log-log plot is een machtswet recht: $\\log y = \\log 5 - 0{,}5 \\log x$. Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "right": "$\\log y = \\log 5 - 0{,}5 \\cdot \\log x$: de exponent is de helling. Deze les fit zulke lijnen door de losses van echte modellen.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="Een model van 100 miljoen parameters traint op 2 miljard tokens. Ongeveer hoeveel FLOPs kost dat?",
        options={
            "two": "$2 \\times 10^{17}$",
            "shift": "$1{,}2 \\times 10^{17}$",
            "right": "$1{,}2 \\times 10^{18}$",
        },
        correct="right",
        why={
            "two": "Dat is $N \\cdot D$. Trainen kost ongeveer 6 FLOPs per parameter per token: zie [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).",
            "shift": "$6 \\times 2$ is 12, dus je krijgt $12 \\times 10^{17}$, en dat is niet $1{,}2 \\times 10^{17}$: schuift de komma één plek naar links, dan gaat de macht van tien één omhoog.",
            "right": "$6 \\times 10^8 \\times 2 \\times 10^9$. Deze les vraagt welke loss je voor die compute krijgt.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een grote trainingsrun kost miljoenen, en je hebt één poging. Sinds 2020 weten we dat de loss van language models heel regelmatig daalt als ze meer parameters, meer data en meer compute krijgen: als een machtswet, een rechte lijn op een log-log plot, sommige over meer dan zeven factoren 10. Je kunt dus een paar kleine modellen trainen, een wet door hun losses fitten en de loss van een model dat duizend keer groter is voorspellen voordat je het traint. OpenAI voorspelde zo de uiteindelijke loss van GPT-4, uit runs die elk hoogstens een tienduizendste van de compute van GPT-4 gebruikten.
    """)
    return


@app.cell(hide_code=True)
def predict_factor():
    guess_factor = L.guess(
        "ten-percent",
        question="Kaplan en anderen (2020) vonden dat de loss van een language model daalt als $N^{-0{,}076}$ met zijn aantal parameters $N$, als data en compute geen beperking zijn. Hoeveel keer meer parameters heeft een model nodig voor een loss die 10% lager is?",
        options={
            "little": "1,1 keer",
            "right": "4 keer",
            "lots": "Ongeveer 100 keer",
        },
        reveal="**4 keer.** Een loss die 0,9 keer zo groot is, vraagt $0{,}9^{-1/0{,}076} \\approx 4{,}0$ keer zoveel parameters. De exponent is klein, dus elke stap kost een grote factor: de loss halveren zou ongeveer 9.000 keer meer parameters kosten.",
    )
    guess_factor
    return (guess_factor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: machtswetten in grootte, data en compute.** Kaplan en anderen trainden language models van minder dan duizend tot 1,5 miljard parameters, de embeddings niet meegeteld. Hun loss volgt een machtswet: $N^{-0{,}076}$ in de parameters $N$, zolang de data geen beperking is; $D^{-0{,}095}$ in de trainingstokens $D$, zolang het model groot genoeg is; en $C^{-0{,}050}$ in de compute $C$, voor het beste model bij elke hoeveelheid compute. Een **scaling law** is zo'n formule: hij voorspelt de loss uit grootte, data of compute.

        **Stap 2: fit hem op de logs.** Een machtswet $L = a \cdot x^k$ is een rechte lijn in de logs:
        """),
        L.formula(r"\log L = \log a + k \cdot \log x"),
        mo.md(r"""
        Neem dus de log van de groottes en van de losses, en fit er een rechte lijn door: de helling is de exponent $k$, het snijpunt $\log a$.

        **Stap 3: de bodem.** Een deel van wat er in een tekst volgt, ligt echt open: zelfs een perfect model kan het niet voorspellen. Daarom kan de loss niet onder een bodem $E$ komen. Hoffmann en anderen (2022, de Chinchilla-paper) fitten
        """),
        L.formula(r"L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}"),
        mo.md(r"""
        met $E = 1{,}69$, $A = 406{,}4$, $\alpha = 0{,}34$, $B = 410{,}7$ en $\beta = 0{,}28$, voor de loss per token op hun eigen tekst. Met eindeloos veel data is het deel boven de bodem, $L - E$, een machtswet in $N$; $L$ zelf buigt af op een log-log plot en vlakt af naar $E$. Hun exponenten horen bij $L - E$, dus je kunt ze niet zomaar vergelijken met die van Kaplan, die bij $L$ zelf horen.

        Het plaatje: de loss van Chinchilla met eindeloos veel data, paars, van 10 miljoen tot 100 miljard parameters. Een rechte lijn door drie runs van 10 tot 100 miljoen parameters, gestreept, voorspelt 0,95 voor 100 miljard; de wet geeft 1,76.
        """),
        mo.Html(C.floor_picture({"title": "De loss van Chinchilla en een rechte lijn", "x": "parameters", "y": "loss", "floor": "bodem E = 1,69",
                                 "curve": "de wet", "line": "rechte lijn", "lang": L.lang,
                                 "alt": "Een log-log plot van de loss tegen het aantal parameters, van 10 miljoen tot 100 miljard. De loss van de wet van Chinchilla met eindeloos veel data daalt van 3,38 bij 10 miljoen naar 2,46 bij 100 miljoen en 1,76 bij 100 miljard, en buigt af naar een gestreepte bodem op 1,69. Drie stippen zijn runs van 10, 32 en 100 miljoen parameters. Een gestreepte rechte lijn door die stippen blijft dalen, tot 0,95 bij 100 miljard, ver onder de wet."})),
        mo.md(r"""
        **Stap 4: waar het voor is.** Met een gefitte wet voorspel je de loss van een run voordat je hem start, en zie je of een nieuw idee de hele lijn omlaag schuift of alleen kleine modellen helpt. Hoe verder je voorspelt voorbij je runs, hoe meer de bodem telt, en hoe minder je een lijn zonder bodem kunt vertrouwen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **scaling law** | een formule, meestal een machtswet, die de loss van een model voorspelt uit zijn grootte, zijn data of zijn compute | |
        | $N$, $D$, $C$ | de parameters, de trainingstokens, en de compute in FLOPs | |
        | $E$ | de bodem: de loss waar geen model onder komt | `floor` |
        | **snijpunt** (met de verticale as) | de hoogte van een rechte lijn waar $\log x = 0$; `np.polyfit` geeft het na de helling, als bias term | `b` |
        """),
        L.watch_out("Een scaling law geldt voor één recept: één soort data, één tokenizer, één architectuur, één trainingsschema. Verander er één en de lijn verschuift, dus fit hem opnieuw. En hij voorspelt de loss, niet wat een model kan."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo fit en gebruik je een scaling law:

    1. Train een paar runs die alleen in grootte verschillen, en noteer de loss van elke run.
    2. Trek de bodem $E$ van elke loss af: 0 als je er ver boven zit, of een waarde uit een grotere fit.
    3. Neem de log van de groottes en van de losses boven de bodem, en fit een rechte lijn: `k, b = np.polyfit(log_x, log_rest, 1)`, en $a = 10^b$.
    4. Voorspel $E + a \cdot x^k$ voor een grotere $x$, en vertrouw die voorspelling minder naarmate $x$ verder voorbij je runs ligt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is een rechte lijn door logs, met `np.polyfit`: drie groottes en de losses die de wet van Kaplan ze geeft.
    """)
    return


@app.cell
def example_fit():
    import numpy as np

    _n = np.array([1e6, 1e7, 1e8])  # parameters
    _loss = np.array([4.02, 3.37, 2.83])  # de wet van Kaplan bij die groottes
    _k, _b = np.polyfit(np.log10(_n), np.log10(_loss), 1)
    print("k =", round(_k, 3), " a =", round(10 ** _b, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `np.polyfit` met graad 1 geeft eerst de helling, $k = -0{,}076$, de exponent van Kaplan, en dan het snijpunt $b$, de log van $a$: $a = 10^b \approx 11{,}5$. Drie runs, elk 10 keer groter dan de vorige, zijn genoeg om de lijn te vinden, zolang de losses ver boven de bodem liggen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Fit een wet met een bodem

        Maak `fit_power_law(x, loss, floor)` af: fit $\text{loss} = \text{floor} + a \cdot x^k$ en geef `a, k` terug. De check roept `fit_power_law(np.array([1e6, 1e7, 1e8, 1e9]), np.array([3.7, 2.7, 2.2, 1.95]), 1.7)` aan, wat ongeveer (128, −0,301) moet geven, en probeert losses die wiebelen, een bodem van 0 en de losses van de wet van Chinchilla.
        """),
        L.task("Maak `fit_power_law` af: een rechte lijn door de logs van `x` en van de loss boven de bodem."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-power-law", lines=12)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-power-law", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-power-law",
        [
            "Stap 2 en 3 van het recept: het uitgewerkte voorbeeld, met `loss - floor` op de plek van de loss.",
        ],
        why="`log_x = np.log10(x)`, `log_rest = np.log10(loss - floor)`, dan `k, b = np.polyfit(log_x, log_rest, 1)` en `a = 10 ** b`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 2. Welke runs
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "which-runs",
        question="Je hebt compute voor vier korte runs, om een scaling law voor je GPT te fitten. Welke groottes train je?",
        options={
            "right": "1, 3, 10 en 30 miljoen parameters",
            "steps": "10, 20, 30 en 40 miljoen parameters",
            "close": "37, 38, 39 en 40 miljoen parameters",
        },
        correct="right",
        why={
            "right": "Gelijke factoren zijn gelijke stappen op een log-log plot: de vier punten liggen verspreid over de lijn, en je meet de helling over een factor 30.",
            "steps": "Op een log-as is 10 tot 40 maar een factor 4, en 20, 30 en 40 liggen opeengepakt aan het eind. Groottes die een gelijke factor uit elkaar liggen, spreiden zich over de lijn.",
            "close": "37 tot 40 is nauwelijks een stap op een log-as: een beetje gewiebel in de losses verandert de helling helemaal.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.md(r"""
    ### Oefening 3. Waarom er een bodem is
    """)
    return


@app.cell(hide_code=True)
def explain_floor():
    explain_floor_cell = L.explain(
        "why-floor",
        question="Waarom kan de loss van een language model niet langs een rechte lijn op een log-log plot blijven dalen, hoe groot het model ook wordt? En wat betekent dat voor wat elke factor 10 in grootte oplevert?",
        model_answer="**Waarom er een bodem is:** een deel van wat er in een tekst volgt, ligt echt open, zoals welke naam een verhaal een nieuw personage geeft. Zelfs een perfect model kan daar alleen kansen voor geven, dus zijn loss blijft boven een bodem $E$, 1,69 in de fit van Chinchilla. **Wat dat betekent:** alleen het deel boven de bodem krimpt als een machtswet, met dezelfde factor voor elke factor 10 in grootte. Als de loss dicht bij $E$ komt, is dat deel een steeds kleiner stuk van de loss, dus haalt elke factor 10 een steeds kleinere fractie van de loss af. Op een log-log plot, waar gelijke factoren gelijke stappen zijn, vlakt de kromme daardoor af naar $E$, en een rechte lijn door kleine modellen voorspelt voor grote modellen een te lage loss.",
        points=[
            "je zegt dat een deel van het volgende token niet te voorspellen is, zodat zelfs een perfect model een loss boven 0 heeft",
            "je zegt dat alleen het deel van de loss boven de bodem als een machtswet daalt",
            "je zegt dat elke factor 10 in grootte een steeds kleinere fractie van de loss afhaalt, zodat de kromme op een log-log plot afvlakt",
        ],
    )
    explain_floor_cell
    return (explain_floor_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De machtswetten en log-log plots van [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl), nu voor de loss van language models, met de compute $C$ van [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl). De volgende les, *Project: fit je eigen scaling law*, traint vier kleine versies van je GPT, fit er een wet door, en voorspelt de loss van een grotere voordat hij die traint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_runs, explain_floor_cell, result_fit):
    L.wrap_up(result_fit, choice_runs, explain_floor_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
