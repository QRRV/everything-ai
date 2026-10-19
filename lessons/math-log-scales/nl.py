# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-log-scales. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-log-scales", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log10():
    quick_log10 = L.choice(
        "quick-log10",
        question="Wat geeft `np.log10(1000)`?",
        options={
            "divided": "`100.0`",
            "natural": "`6.907755278982137`",
            "right": "`3.0`",
        },
        correct="right",
        why={
            "divided": "Dat is 1000 / 10. Welke exponent van 10 geeft 1000? Zie *Het idee* in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "natural": "Dat is `np.log(1000)`, met grondtal $e$. Welk grondtal heeft `np.log10`? Zie de tabel met nieuwe woorden in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "$10^3 = 1000$, dus de log met grondtal 10 van 1000 is 3.",
        },
    )
    quick_log10
    return (quick_log10,)


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question=r"Wat is $\log(a \cdot b)$?",
        options={
            "right": r"$\log(a) + \log(b)$",
            "times": r"$\log(a) \cdot \log(b)$",
            "sum": r"$\log(a + b)$",
        },
        correct="right",
        why={
            "times": "De log maakt van vermenigvuldigen iets anders. Wat? Zie de formule in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "sum": "Binnen de log blijft het product een product. Wat gebeurt er buiten de log mee? Zie de formule in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "De log maakt van vermenigvuldigen optellen: de log van een product is de som van de logs.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige getallen beslaan veel factoren 10. In een tekst komt het vaakste woord duizenden keren voor, en de meeste woorden maar een paar keer. Een klein language model heeft een paar miljoen parameters, en een groot honderden miljarden. Op een gewone as zakt alles behalve de grootste getallen weg in een streep onderaan. Een logaritmische schaal geeft elke factor 10 evenveel ruimte. En sommige verbanden die op een gewone as krom lopen, worden een rechte lijn op logaritmische assen, zoals hoeveel beter een model wordt als het groeit: onderzoekers tekenen dat op precies zulke grafieken.

    **Een voorbeeld uit het dagelijks leven.** De schaal voor aardbevingen: elke stap hoger betekent 10 keer zoveel trilling. Een 6 is niet een beetje meer dan een 5: het is 10 keer zoveel.
    """)
    return


@app.cell(hide_code=True)
def predict_factor():
    guess_factor = L.guess(
        "factor",
        question="Een grafiek heeft langs de zijkant de rasterlijnen 1, 10, 100 en 1000, even ver uit elkaar. Een rechte lijn loopt van 1 bij x = 0 naar 1000 bij x = 3. Wat gebeurt er volgens jou met y elke keer dat x 1 groter wordt?",
        options={
            "plus333": "y wordt 333 groter",
            "times10": "y wordt met 10 vermenigvuldigd",
            "plus10": "y wordt 10 groter",
        },
        reveal="**y wordt met 10 vermenigvuldigd.** Elke stap omhoog langs de as is een factor 10: van 1 naar 10, naar 100, naar 1000. Een rechte lijn klimt per x evenveel stappen, dus vermenigvuldigt hij y elke keer met dezelfde factor. Op een gewone as zouden dezelfde getallen een kromme maken die aan het eind omhoogschiet.",
    )
    guess_factor
    return (guess_factor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Op een gewone as zijn gelijke stappen gelijke verschillen: 0, 10, 20, 30. Op een **logaritmische schaal** zijn gelijke stappen gelijke factoren: 1, 10, 100, 1000. De as zet elk getal op de plek van zijn log met grondtal 10: 100 op 2, 1000 op 3. Daarom hebben 0 en negatieve getallen er ook geen plek op.

        Twee soorten groei worden een rechte lijn:

        - Groei met dezelfde factor per stap, $y = a \cdot b^x$, is een rechte lijn als alleen de y-as een logaritmische schaal heeft: $\log(y) = \log(a) + x \cdot \log(b)$.
        - Een **machtswet**, $y = a \cdot x^k$, is een rechte lijn op een **log-log plot**, met een logaritmische schaal op beide assen: $\log(y) = \log(a) + k \cdot \log(x)$. Voor een geheel getal $k$ is $x^k$ $k$ factoren $x$, en de log maakt van elke factor een term, dus $\log(x^k) = k \cdot \log(x)$; die regel geldt voor elke $k$, ook voor $-1$ of $0{,}5$.

        De helling van die rechte lijn is de exponent $k$. Uit twee punten erop, met logs in elk grondtal, zolang je omhoog en opzij hetzelfde grondtal neemt:
        """),
        L.formula(r"k = \frac{\log(y_2) - \log(y_1)}{\log(x_2) - \log(x_1)}"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **logaritmische schaal** | | een as waarop gelijke stappen gelijke factoren zijn, zoals 1, 10, 100, 1000 | `plt.yscale("log")` |
        | **log-log plot** | | een grafiek met een logaritmische schaal op beide assen | `plt.loglog(x, y)` |
        | **machtswet** | | $y = a \cdot x^k$: $y$ is een macht van $x$; een rechte lijn op een log-log plot, met helling $k$ | `y = a * x ** k` |
        """),
        L.watch_out("Op een logaritmische schaal is de ruimte tussen rasterlijnen een factor, geen verschil: van 1 tot 10 neemt evenveel ruimte in als van 100 tot 1000. Lees de getallen af van de as, niet de afstanden."),
        mo.md(r"""
        Schuif met de slider: hij zet de exponent $k$ van $y = x^k$, getekend op gewone assen en op een log-log plot. De punten liggen bij $x$ = 1, 10 en 100: even ver uit elkaar op de logaritmische schaal.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        factor = 10 ** (k if k >= 0 else -k)
        times = L.number(factor, 0 if factor == round(factor) else 2)
        y = f"wordt <i>y</i> {times} keer zo groot" if k > 0 else (f"wordt <i>y</i> gedeeld door {times}" if k < 0 else "blijft <i>y</i> gelijk")
        normal = "een rechte lijn" if k in (0, 1) else "een kromme"
        return (C.picture(k, {"normal": "Gewone assen", "loglog": "Twee logaritmische assen",
                              "alt": f"Twee grafieken van y = x tot de macht {L.number(k, 1)}, voor x van 1 tot 100. Op gewone assen is het {normal}; met twee logaritmische assen is het een rechte lijn met helling {L.number(k, 1)}, met punten bij x = 1, 10 en 100."}),
                f"""Met <i>k</i> = {L.number(k, 1)}: elke keer dat <i>x</i> 10 keer zo groot wordt, {y}. Op de log-log plot is dat elke keer dezelfde stap, dus is de lijn recht.""")

    explore_k = L.explore("power-k", "k", _frame, start=2, low=-1, high=2, step=0.5, role="", task="Probeer elke $k$: hoeveel rasterlijnen gaat de lijn op de log-log plot omhoog of omlaag, van $x$ = 1 tot $x$ = 100?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Op de log-log plot gaat de lijn $2k$ rasterlijnen omhoog van $x$ = 1 tot $x$ = 100: vier bij $k$ = 2, van 1 tot 10.000, en twee omlaag bij $k$ = −1, tot 0,01. Opzij is dat twee keer een factor 10, dus omhoog gedeeld door opzij is $k$: de lijn is recht, met helling $k$, bij elke $k$. Op gewone assen buigt hij, omhoog of omlaag, behalve als $k$ 0 of 1 is.

    ### Recept

    Zo teken en lees je een grafiek met een logaritmische schaal:

    1. Getallen die veel factoren 10 beslaan? Zet die as op een logaritmische schaal: `plt.yscale("log")`, `plt.xscale("log")`, of allebei tegelijk met `plt.loglog(x, y)`.
    2. Lees de as in factoren: elke rasterlijn is 10 keer de lijn eronder.
    3. Een rechte lijn met alleen de y-as op een logaritmische schaal: dezelfde factor per stap.
    4. Een rechte lijn op een log-log plot: een machtswet. De exponent is de helling: de verandering van $\log(y)$ gedeeld door de verandering van $\log(x)$, allebei met hetzelfde grondtal. Met grondtal 10, `np.log10`, lees je de logs makkelijk af van de as.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een aantal cellen dat elk uur verdubbelt, vanaf 100, getekend met een logaritmische schaal op de y-as. Daarna een machtswet, de oppervlakte van een vierkant tegen de zijde, op een log-log plot, en de exponent uit twee punten. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_log_scale():
    import matplotlib.pyplot as plt
    import numpy as np

    _hours = np.arange(10)
    _cells = 100 * 2 ** _hours
    # 1. van 100 tot 51.200: een paar factoren 10, dus een logaritmische schaal op de y-as
    plt.plot(_hours, _cells)
    plt.yscale("log")
    plt.xlabel("uur")
    plt.ylabel("cellen")
    plt.show()
    # 2. en 3. de logs met grondtal 10 van de aantallen gaan elk uur met dezelfde stap omhoog
    print(np.log10(_cells))
    # 1. van 1 tot 10.000: een logaritmische schaal op beide assen, een log-log plot
    _sides = np.array([1, 2, 5, 10, 20, 50, 100])
    plt.loglog(_sides, _sides ** 2)
    plt.xlabel("zijde")
    plt.ylabel("oppervlakte")
    plt.show()
    # 4. een rechte lijn: de helling tussen de zijden 1 en 100 is de exponent
    _side_a = 1
    _area_a = _side_a ** 2
    _side_b = 100
    _area_b = _side_b ** 2
    print(_area_a, _area_b)
    _k = (np.log10(_area_b) - np.log10(_area_a)) / (np.log10(_side_b) - np.log10(_side_a))
    print(_k)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste lijn is recht: elk uur wordt het aantal met 2 vermenigvuldigd, dezelfde factor. Matplotlib schrijft de rasterlijnen als machten van 10: $10^2$ is 100, $10^3$ is 1000. De logs met grondtal 10 gaan van 2 naar 2,30, dan 2,60 en zo verder: elk uur 0,30 hoger, $\log_{10}(2)$. De vierkanten geven op de log-log plot ook een rechte lijn, en de helling is `2.0`: de exponent 2 van het kwadraat. Een zijde die 100 keer zo lang is, geeft een oppervlakte die $100^2$ = 10.000 keer zo groot is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_halfway():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Halverwege op een logaritmische schaal
    """)
    return


@app.cell(hide_code=True)
def choice_halfway_cell():
    choice_halfway = L.choice(
        "halfway",
        question="Een as met een logaritmische schaal heeft de rasterlijnen 1, 10, 100 en 1000. Welk getal ligt precies halverwege tussen de rasterlijnen 10 en 100?",
        options={
            "middle": "55",
            "right": "Ongeveer 32",
            "log": "1,5",
        },
        correct="right",
        why={
            "middle": "55 is halverwege op een gewone as, waar gelijke stappen gelijke verschillen zijn. Wat zijn gelijke stappen op een logaritmische schaal? Zie *Het idee*.",
            "log": "1,5 is de log met grondtal 10 van het getal halverwege, zijn plek op de as. Welk getal heeft die log? Zie *Het idee*.",
            "right": "Halverwege is halverwege in de logs met grondtal 10: 1,5, tussen 1 en 2. $10^{1{,}5}$ is ongeveer 31,6. Van 10 naar 31,6 is een factor 3,16, en van 31,6 naar 100 ook.",
        },
    )
    choice_halfway
    return (choice_halfway,)


@app.cell(hide_code=True)
def text_exercise_plot():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Woorden op een log-log plot

        De course telde de woorden van *Tiny Shakespeare*, een bestand in de map van de course met ongeveer 40.000 regels uit toneelstukken van William Shakespeare, en zette ze op volgorde van vaakst naar minst vaak. Voor twaalf plekken op die lijst heeft het vak het rangnummer in `ranks` en hoe vaak dat woord voorkomt in `counts`: het vaakste, "the", 6283 keer; woord 5000 maar 3 keer. Teken `counts` tegen `ranks` als lijn, met beide assen op een logaritmische schaal. Vervang de `...`.
        """),
        L.task("Teken `counts` tegen `ranks` op een log-log plot."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plot():
    box_plot = L.code_box("zipf-plot", lines=7)
    box_plot
    return (box_plot,)


@app.cell(hide_code=True)
def check_plot(box_plot):
    result_plot = L.check("zipf-plot", box_plot.value)
    result_plot
    return (result_plot,)


@app.cell(hide_code=True)
def hints_plot():
    L.hints(
        "zipf-plot",
        [
            "De rangnummers gaan langs de onderkant, de aantallen langs de zijkant, en allebei beslaan ze een paar factoren 10.",
            "Eén functie tekent een lijn met beide assen op een logaritmische schaal. Welke? Zie stap 1 van het recept.",
            "`plt.loglog(___, ___)`, vóór `plt.show()`.",
        ],
        why="`plt.loglog(ranks, counts)` tekent een lijngrafiek met een logaritmische schaal op beide assen, de rangnummers langs de onderkant.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_exponent():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De exponent van de woorden

        De lijn van oefening 2 is bijna recht: een machtswet, aantal = $a \cdot \text{rangnummer}^k$. Reken $k$ uit met twee van zijn punten: woord 10, "in", komt 2375 keer voor, en woord 1000, "secret", 21 keer. Het vak heeft ze als `rank_a` en `count_a`, en `rank_b` en `count_b`. Zet de exponent in `k`. Vervang de `...`. De laatste regel is `k`, zodat de check hem ziet. De check probeert ook een ander aantal.
        """),
        L.task("Zet de exponent van de machtswet door de twee punten in `k`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exponent():
    box_exponent = L.code_box("zipf-exponent", lines=9)
    box_exponent
    return (box_exponent,)


@app.cell(hide_code=True)
def check_exponent(box_exponent):
    result_exponent = L.check("zipf-exponent", box_exponent.value)
    result_exponent
    return (result_exponent,)


@app.cell(hide_code=True)
def hints_exponent():
    L.hints(
        "zipf-exponent",
        [
            "De exponent is de helling op de log-log plot: zie de formule in *Het idee*.",
            "Langs de zijkant staan de logs van de aantallen, langs de onderkant de logs van de rangnummers. Wat deel je door wat?",
            "`k = (np.log10(count_b) - np.log10(___)) / (np.log10(___) - np.log10(rank_a))`.",
        ],
        why="Met grondtal 10 gaan de logs van de aantallen van 3,376 naar 1,322, een verandering van −2,053; de logs van de rangnummers van 1 naar 3, een verandering van 2. Gedeeld: ongeveer −1,03.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl) maakte de log van vermenigvuldigen optellen; een logaritmische schaal doet hetzelfde op een as, zodat gelijke factoren evenveel ruimte krijgen. In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je lijngrafieken; één regel extra zet een as op een logaritmische schaal. Later in de course kom je scaling laws tegen, machtswetten voor hoeveel beter een model wordt met meer parameters en data, getekend op log-log plots. In de volgende les, *Van score naar kans: de sigmoid*, maakt een functie met exp van elke score een kans.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_halfway, result_exponent, result_plot):
    L.wrap_up(choice_halfway, result_plot, result_exponent)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
