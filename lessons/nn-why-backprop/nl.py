# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-why-backprop. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-why-backprop", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) en [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="Je verschuift `w` met een heel kleine stap `h` = 0,001, en de loss gaat van 50 naar 49,997. Wat is ongeveer de helling van de loss voor `w`?",
        options={
            "right": "−3",
            "change": "−0,003",
            "sign": "3",
        },
        correct="right",
        why={
            "change": "Dat is de verandering van de loss. Een helling is de verandering per stap: waardoor deel je die? Zie het recept in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "sign": "De loss daalt als `w` stijgt. Wat zegt dat over het teken van de helling? Zie [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "right": "De verandering van de loss, −0,003, gedeeld door de stap 0,001: −3. De loss daalt als `w` stijgt.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Een layer heeft 784 inputs en 128 neuronen. Hoeveel weights staan er in zijn `W`?",
        options={
            "sum": "912",
            "right": "100.352",
            "neurons": "128",
        },
        correct="right",
        why={
            "sum": "Dat is 784 + 128. Elk neuron heeft een weight voor elke input: hoeveel zijn dat samen? Zie de shapes in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "neurons": "Dat is één getal per neuron, zoals de bias terms. `W` heeft shape (128, 784): hoeveel getallen zijn dat? Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "right": "`W` heeft shape (128, 784): 128 rijen van 784 weights, 128 × 784 = 100.352.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Trainen zet steeds weer een stap tegen de gradient in, en elke stap heeft de helling van de loss voor elke parameter nodig. In [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) mat je die hellingen door elke parameter om de beurt een heel klein stukje te verschuiven, voor $w$ en $b$. Een klein netwerk voor handgeschreven cijfers heeft al zo'n 100.000 parameters, en een language model heeft er miljarden. Voordat je iets zo groots bouwt, loont het om te tellen wat het kost.

    **Een voorbeeld uit het dagelijks leven.** Op een feest waar iedereen iedereen een hand geeft, maken 4 gasten 6 handdrukken. Met 40 gasten zijn het er geen 60 maar 780: tien keer zoveel gasten, meer dan honderd keer zoveel handdrukken.
    """)
    return


@app.cell(hide_code=True)
def predict_seconds():
    guess_seconds = L.guess(
        "seconds",
        question="Stel dat één forward pass van een netwerk met 100.000 parameters 0,1 milliseconde kost, voor één voorbeeld. Hoe lang duurt het volgens jou om voor dat ene voorbeeld de gradient te meten, door elke parameter om de beurt te verschuiven?",
        options={
            "twice": "Ongeveer 0,2 milliseconde: één keer waar je bent, één keer verschoven",
            "right": "Ongeveer 10 seconden",
            "second": "Ongeveer 1 seconde",
        },
        reveal="**Ongeveer 10 seconden.** Je hebt de loss één keer nodig waar je bent, en nog één keer voor elke parameter die je verschuift: 100.001 forward passes. Bij 0,1 milliseconde per stuk is dat ongeveer 10 seconden, voor één voorbeeld en één stap. Trainen kost duizenden stappen, elk over veel voorbeelden.",
    )
    guess_seconds
    return (guess_seconds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Om manieren van rekenen te vergelijken, tel je hun **FLOPs**: het aantal rekenbewerkingen met floating-point getallen, optellingen en vermenigvuldigingen, dat ze nodig hebben. Dat aantal hangt niet af van de computer.

        **Eén forward pass.** In een layer wordt elke weight één keer per voorbeeld gebruikt: één vermenigvuldiging, $w \cdot x$, en één optelling bij de som (de eerste telt de bias term erbij). Met $N$ parameters kost een forward pass voor één voorbeeld ongeveer $2N$ FLOPs.

        **Meten door te verschuiven.** Je rekent de loss uit waar je bent, en dan nog één keer voor elke parameter, verschoven met een heel kleine stap: $N + 1$ forward passes. Dat is $N + 1$ keer de kosten van één:
        """),
        L.formula(r"\underbrace{2N}_{\text{één forward pass}} \qquad \underbrace{(N + 1) \cdot 2N \approx 2N^2}_{\text{elke helling door te verschuiven}}"),
        mo.md(r"""
        Waar het om gaat, is hoe de kosten groeien met $N$. De **grote-O-notatie** zegt precies dat, en laat de constante factoren weg: een forward pass is $O(N)$, hij groeit zoals $N$; elke helling meten door te verschuiven is $O(N^2)$, dat groeit zoals $N^2$. Maak het netwerk twee keer zo groot, en een forward pass duurt twee keer zo lang, maar de gradient meten vier keer zo lang.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **FLOPs** | flops | het aantal rekenbewerkingen met floating-point getallen, zoals optellingen en vermenigvuldigingen, dat een berekening nodig heeft: een manier om de kosten te tellen | |
        | $N$ | N | het aantal parameters van een model: al zijn weights en bias terms; niet $n$, het aantal voorbeelden | `n_params` |
        | **grote-O-notatie** | | hoe de kosten groeien met de grootte, zonder de constante factoren | |
        | $O(N)$ | grote O van N | groeit zoals $N$: twee keer zo groot, twee keer de kosten | |
        | $O(N^2)$ | grote O van N kwadraat | groeit zoals $N^2$: twee keer zo groot, vier keer de kosten | |
        """),
        L.watch_out("Hellingen meten door te verschuiven is niet fout, alleen traag. Het is eenvoudig en je doet het niet snel fout, dus het blijft de manier om hellingen te controleren die op een snellere manier zijn uitgerekend, bij een paar parameters."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek: zo zet je het aantal parameters $N$. De gestreepte lijn is één forward pass, de doorgetrokken lijn elke helling meten door te verschuiven, allebei in miljoenen FLOPs.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _frame(n):
        n = int(n)
        return (C.picture(n, {"title": "Wat de hellingen kosten", "x": "aantal parameters N", "y": "miljoenen FLOPs",
                              "move": "elke helling door te verschuiven", "forward": "één forward pass",
                              "alt": f"Twee lijnen over N van 0 tot 1000. De gestreepte lijn, één forward pass, blijft bij 0: bij N = 1000 is hij 0,002 miljoen FLOPs. De doorgetrokken lijn, elke helling door te verschuiven, buigt omhoog tot ongeveer 2 miljoen FLOPs bij N = 1000. Bij N = {n} staan de stippen op {L.number(C.forward(n) / 1e6, 4)} en {L.number(C.moving(n) / 1e6, 3)} miljoen."}),
                f"""Met <i>N</i> = {n} kost één forward pass <span class="ea-num">{C.whole(C.forward(n), L.lang)}</span> FLOPs, en elke helling door te verschuiven <span class="ea-num">{C.whole(C.moving(n), L.lang)}</span>: {C.whole(n + 1, L.lang)} keer zoveel.""")

    explore_n = L.explore("n", "N", _frame, start=400, low=50, high=1000, step=50, role="", task="Stel dat je een miljoen FLOPs hebt voor de hellingen van één voorbeeld. Tot welke $N$ is dat genoeg om elke helling te meten door te verschuiven?")
    explore_n
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Een miljoen FLOPs is tot ongeveer $N$ = 700 genoeg om elke helling te meten door te verschuiven: dat kost 981.400 FLOPs, en bij 750 is het al 1.126.500. Voor één forward pass is hetzelfde miljoen genoeg tot $N$ = 500.000.

    ### Recept

    Zo schat je wat een berekening kost:

    1. Tel de FLOPs van één stap, zoals één forward pass: ongeveer 2 per weight.
    2. Tel hoe vaak die stap gebeurt, zoals één forward pass voor elke parameter, plus één.
    3. Vermenigvuldig die twee, en houd alleen het deel dat het snelst groeit: dat is de grote O.
    4. Vul de grootte in die je hebt, en kijk of het nog te doen is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een netwerk voor handgeschreven cijfers: 784 inputs, een hidden layer van 128 neuronen, en 10 outputs. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_cost():
    _weights = 128 * 784 + 10 * 128     # W1 heeft shape (128, 784), W2 (10, 128)
    _n_params = _weights + 128 + 10     # plus b1 en b2
    # 1. één forward pass, voor één voorbeeld: 2 FLOPs per weight
    _forward = 2 * _weights
    print(_n_params, _forward)
    # 2. en 3. één forward pass waar je bent, en één voor elke parameter
    _moving = (_n_params + 1) * _forward
    print(_moving)
    # 4. hoeveel keer zoveel als één forward pass
    print(_moving / _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het netwerk heeft 101.770 parameters, en één forward pass voor één voorbeeld kost 203.264 FLOPs. Elke helling meten door te verschuiven kost ongeveer 20.700.000.000 FLOPs: 101.771 keer zoveel, voor één voorbeeld. Trainen heeft een manier nodig om elke helling te krijgen voor een paar keer de kosten van een forward pass.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flops():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De FLOPs van een forward pass

        `sizes` noemt het aantal inputs en dan het aantal neuronen in elke layer: `[784, 128, 10]` is het netwerk van het uitgewerkte voorbeeld. Maak `forward_flops(sizes)` af: hij geeft de FLOPs van één forward pass voor één voorbeeld, 2 per weight. Vervang de `...` door wat layer `i + 1` bij `total` optelt. De check roept `forward_flops([3, 4, 2])` aan: dat moet `40` geven. Hij probeert ook andere netwerken, met meer en minder layers.
        """),
        L.task("Laat `forward_flops` 2 FLOPs tellen voor elke weight van het netwerk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flops():
    box_flops = L.code_box("forward-flops", lines=6)
    box_flops
    return (box_flops,)


@app.cell(hide_code=True)
def check_flops(box_flops):
    result_flops = L.check("forward-flops", box_flops.value)
    result_flops
    return (result_flops,)


@app.cell(hide_code=True)
def hints_flops():
    L.hints(
        "forward-flops",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld, maar voor elke lijst met groottes.",
            "Layer `i + 1` heeft `sizes[i + 1]` neuronen, elk met een weight voor elk van zijn `sizes[i]` inputs.",
            "`total = total + 2 * ___ * ___`.",
        ],
        why="Bij `[3, 4, 2]` heeft de hidden layer 4 × 3 = 12 weights en de output layer 2 × 4 = 8: 20 weights, dus 40 FLOPs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_growth():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Zoals N of zoals N²?

        Elk van deze berekeningen groeit met het aantal parameters $N$ van een netwerk. Zet elke berekening in de goede groep.
        """),
        L.task("Deel de zes berekeningen in naar hoe ze groeien met N."),
    ])
    return


@app.cell(hide_code=True)
def sort_growth_cell():
    sort_growth = L.sort("growth")
    sort_growth
    return (sort_growth,)


@app.cell(hide_code=True)
def check_growth(sort_growth):
    result_growth = L.check("growth", sort_growth)
    result_growth
    return (result_growth,)


@app.cell(hide_code=True)
def hints_growth():
    L.hints(
        "growth",
        [
            "Vraag je bij elke berekening af: hoeveel forward passes of stappen kost hij, en wat kost elk daarvan?",
            "Een vast beetje werk per parameter is $O(N)$. Een forward pass voor elke parameter, of een stap voor elk paar parameters, is $O(N^2)$.",
            "Zoals $N$: één forward pass, \\_\\_\\_ en de parameters bewaren. Zoals $N^2$: elke helling door te verschuiven, \\_\\_\\_ en elk paar vergelijken.",
        ],
        why="**Zoals $N$:** één forward pass, de parameters bewaren en één stap van gradient descent; elk doet een vast beetje werk per parameter, dus twee keer zoveel parameters is twee keer zoveel werk. **Zoals $N^2$:** elke helling door te verschuiven, de helft van de hellingen door te verschuiven en elk paar vergelijken. De helft van de parameters verschuiven is nog steeds een forward pass, $2N$ FLOPs, voor elk van $N/2$ parameters: $N^2$, en de grote-O-notatie laat de helft weg.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bigger():
    mo.md(r"""
    ### Oefening 3. Tien keer zo groot
    """)
    return


@app.cell(hide_code=True)
def choice_bigger_cell():
    choice_bigger = L.choice(
        "bigger",
        question="Bij een netwerk met 100.000 parameters duurt elke helling meten door te verschuiven ongeveer 10 seconden per voorbeeld. Je maakt het netwerk tien keer zo groot: 1.000.000 parameters. Hoe lang duurt het nu ongeveer?",
        options={
            "ten": "Ongeveer 100 seconden",
            "twenty": "Ongeveer 20 seconden",
            "right": "Ongeveer 1.000 seconden",
        },
        correct="right",
        why={
            "ten": "Er zijn tien keer zoveel forward passes. Maar wat gebeurt er met de kosten van elke forward pass? Zie *Het idee*.",
            "twenty": "De kosten groeien niet met een stap van 10 seconden. Hoe groeit $O(N^2)$ als $N$ tien keer zo groot is? Zie *Het idee*.",
            "right": "Tien keer zoveel forward passes, elk tien keer zo duur: honderd keer zo lang, ongeveer 17 minuten, voor één voorbeeld.",
        },
    )
    choice_bigger
    return (choice_bigger,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) mat je elke helling door elke parameter om de beurt te verschuiven. Dat kost een forward pass per parameter: $O(N^2)$ FLOPs, veel te veel voor een netwerk zoals in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl). In de volgende les, *Backpropagation met de hand op een kleine graaf*, krijg je elke helling door één keer achteruit door de berekening te gaan, voor een paar keer de kosten van een forward pass.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_flops, result_growth, choice_bigger):
    L.wrap_up(result_flops, result_growth, choice_bigger)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
