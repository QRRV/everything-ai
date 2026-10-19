# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-softmax. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-softmax", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) en [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sigmoid():
    quick_sigmoid = L.choice(
        "quick-sigmoid",
        question="Een logit van −1 gaat door de sigmoid. Welke kans komt er ongeveer uit?",
        options={
            "plus": "0,73",
            "right": "0,27",
            "same": "−1",
        },
        correct="right",
        why={
            "plus": "Dat is $\\sigma(1)$. Ligt −1 boven of onder 0? Zie de figuur in [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "same": "De sigmoid maakt van elk getal een getal tussen 0 en 1. Zie *Het idee* in [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "right": "$\\sigma(-1) = 1 / (1 + e^{1}) \\approx 0{,}27$: een negatieve logit geeft minder dan 0,5.",
        },
    )
    quick_sigmoid
    return (quick_sigmoid,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`m = np.array([[1, 2], [3, 4]])`. Wat geeft `m.sum(axis=1)`?",
        options={
            "right": "`array([3, 7])`",
            "columns": "`array([4, 6])`",
            "all": "`10`",
        },
        correct="right",
        why={
            "columns": "Dat telt elke kolom op: `axis=0`. In welke richting telt `axis=1` op? Zie de figuur in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "all": "Dat is `m.sum()`, zonder axis. Zie *Het idee* in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "right": "`axis=1` telt langs elke rij op: 1 + 2 en 3 + 4.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De sigmoid geeft een kans bij twee klassen. Een model dat handgeschreven cijfers leest, heeft tien klassen, en het geeft tien logits, één per klasse: hoeveel het plaatje lijkt op een 0, een 1, enzovoort. Het heeft tien kansen nodig: allemaal positief, samen 1, en de grootste bij de grootste logit. Softmax doet precies dat, en bijna elk model voor veel klassen eindigt ermee, tot en met een language model, dat zijn volgende woord kiest uit tienduizenden.

    **Een voorbeeld uit het dagelijks leven.** Een pizza verdeeld naar hoeveel honger iedereen heeft. Iedereen krijgt een punt, wie de meeste honger heeft de grootste, en alle punten samen zijn precies één pizza.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="Drie klassen krijgen de logits 3, 1 en 0. Wat is volgens jou de kans op de eerste klasse?",
        options={
            "linear": "0,75: 3 van de 3 + 1 + 0",
            "softmax": "Ongeveer 0,84",
            "all": "1: die heeft de grootste logit",
        },
        reveal="**Ongeveer 0,84.** Softmax neemt eerst $e$ tot de macht van elke logit: $e^3 \\approx 20{,}1$, $e^1 \\approx 2{,}7$ en $e^0 = 1$, samen ongeveer 23,8. De eerste klasse krijgt $20{,}1 / 23{,}8 \\approx 0{,}84$. De logits zelf delen door hun som zou de klasse met logit 0 de kans 0 geven, en een negatieve logit een negatief getal. En de grootste logit laat nog ruimte voor de andere.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Softmax** maakt $K$ kansen van een vector van $K$ logits, $\mathbf{z} = (z_0; \dots; z_{K-1})$, met de klassen genummerd vanaf 0, zoals Python telt. Neem $e$ tot de macht van elke logit, wat elk getal positief maakt, en deel elk door hun som, zodat ze samen 1 zijn:
        """),
        L.formula(r"[[pred:\mathrm{softmax}(\mathbf{z})_k]] = \frac{e^{z_k}}{e^{z_0} + e^{z_1} + \dots + e^{z_{K-1}}}"),
        mo.md(r"""
        De grootste logit krijgt de grootste kans, en een groter verschil tussen logits drijft de kansen verder uit elkaar. Alleen de verschillen tellen: tel bij elke logit hetzelfde getal $c$ op, en elke $e^{z_k}$ wordt met $e^c$ vermenigvuldigd, boven en onder de streep, dus er verandert niets. Dat geeft een veilige manier om hem uit te rekenen: haal eerst de grootste logit eraf. Dan is de grootste macht $e^0 = 1$, en komt $e$ tot de macht van een grote logit, die een computer niet kan bewaren, nooit voor.

        Met twee klassen en de logits $z$ en 0 geeft softmax de eerste klasse $e^z$ gedeeld door $e^z + e^0 = e^z + 1$. Deel boven en onder de streep door $e^z$, en omdat $1 / e^z = e^{-z}$, is dat de sigmoid van $z$:
        """),
        L.formula(r"\frac{e^{z}}{e^{z} + 1} = \frac{1}{1 + e^{-z}} = \sigma(z)"),
        mo.md(r"""
        Softmax is de sigmoid voor veel klassen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **softmax**, $\mathrm{softmax}(\mathbf{z})$ | softmax van z | maakt van $K$ logits $K$ kansen: $e$ tot de macht van elke logit, gedeeld door hun som | `e = np.exp(z - z.max())`, `e / e.sum()` |
        | $\mathbf{z}$ | z | de vector met logits, één per klasse; vet, omdat het een vector is | `z` |
        | $K$ | K | het aantal klassen: de lengte van $\mathbf{z}$ | `len(z)` |
        | $z_k$, $\mathrm{softmax}(\mathbf{z})_k$ | z k, softmax van z op plek k | de logit van klasse $k$, van $z_0$ tot en met $z_{K-1}$, en de kans die softmax die klasse geeft | `z[k]` |
        | **de grootste logit eraf halen** | | de grootste logit van alle logits aftrekken vóór `np.exp`: de kansen blijven gelijk, en geen macht wordt te groot | `z - z.max()` |
        """),
        L.watch_out("`np.exp(1000)` is te groot voor de computer: het wordt `inf`, oneindig groot, en `inf / inf` is `nan`. Grote logits komen in echte modellen voor. Eerst de grootste logit eraf halen voorkomt het, en verandert niets aan de kansen."),
        mo.md(r"""
        Schuif met de slider: hij zet de logit van klasse 0, terwijl klasse 1 de logit 1 houdt en klasse 2 de logit 0. De ene grafiek heeft de <span class="c-pred">kansen</span> als staven; samen zijn ze altijd 1 (de getallen boven de staven zijn afgerond). De andere toont de drie kansen voor elke $z_0$: klasse 0 doorgetrokken, klasse 1 en 2 gestippeld, elk met zijn nummer, en een grijze stippellijn bij 0,5; sleep eroverheen om $z_0$ te zetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        ps = C.probabilities(z)
        shown = [L.number(p, 2) for p in ps]
        return (C.picture(z, {"title": "Drie kansen uit softmax", "x": "klasse", "y": "kans", "every": "Voor elke z₀",
                              "alt": f"Drie staven, één per klasse, zo hoog als hun kans: {shown[0]}, {shown[1]} en {shown[2]}, bij de logits {L.number(z, 1)}, 1 en 0. En de drie kansen voor elke z₀ van −3 tot 5: die van klasse 0 stijgt van bijna 0 tot bijna 1 en komt bij ongeveer z₀ = 1,3 boven de stippellijn bij 0,5, terwijl die van klasse 1 en 2 dalen, klasse 1 altijd boven klasse 2; drie punten bij z₀ = {L.number(z, 1)}."}, shown),
                f"""Met de logits {L.number(z, 1)}, 1 en 0 geeft softmax <span class="c-pred ea-num">{shown[0]}</span>, <span class="c-pred ea-num">{shown[1]}</span> en <span class="c-pred ea-num">{shown[2]}</span>.""")

    explore_z = L.explore("logit-0", "z₀", _frame, start=1, low=-3, high=5, step=0.5, role="", task="Vanaf welke $z_0$ krijgt klasse 0 meer dan de helft van de kans? Wat gebeurt er met de verhouding tussen klasse 1 en klasse 2?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De doorgetrokken lijn van klasse 0 komt bij ongeveer $z_0 = 1{,}3$ boven de stippellijn bij 0,5, dus op de slider krijgt klasse 0 meer dan de helft vanaf $z_0 = 1{,}5$: 0,55. Klasse 1 en 2 delen wat overblijft. Ze worden door dezelfde som gedeeld, dus hun verhouding blijft $e^1 / e^0 = e$, ongeveer 2,7 tegen 1, wat $z_0$ ook is: die hangt alleen af van het verschil tussen hun twee logits, 1 − 0, en dat verandert nooit.

    ### Recept

    Zo maak je met softmax van logits kansen:

    1. Haal de grootste logit van elke logit af: `z - z.max()`.
    2. Neem $e$ tot de macht van elke logit: `np.exp(...)`.
    3. Deel elk door hun som: `e / e.sum()`.
    4. Controleer: allemaal positief, samen 1, en de grootste logit heeft de grootste kans.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst de logits 3, 1 en 0, stap voor stap. Dan de logits 1000, 999 en 998: één keer zonder de grootste eraf te halen, en één keer met. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_softmax():
    import numpy as np

    _z = np.array([3, 1, 0])
    # 1. de grootste logit eraf
    _shifted = _z - _z.max()
    print(_shifted)
    # 2. e tot de macht van elke logit
    _e = np.exp(_shifted)
    print(_e)
    # 3. gedeeld door hun som
    _p = _e / _e.sum()
    print(_p)
    # 4. de controle: samen 1
    print(_p.sum())

    # grote logits: zonder stap 1, en met
    _big = np.array([1000, 999, 998])
    print(np.exp(_big))
    _e_big = np.exp(_big - _big.max())
    print(_e_big / _e_big.sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de grootste logit eraf worden de logits 0, −2 en −3, en hun machten 1, 0,135 en 0,050. Gedeeld door hun som, 1,185, geeft dat 0,844, 0,114 en 0,042: samen 1. Hetzelfde als $e^3$, $e^1$ en $e^0$ gedeeld door hun som, zoals bij het gokken. Voor 1000, 999 en 998 geeft `np.exp` alleen drie keer `inf`. NumPy laat ook een waarschuwing zien, de regel met `RuntimeWarning: overflow encountered in exp`: de uitkomst is te groot. Een waarschuwing is geen fout; de cel draait gewoon door. Met de grootste eraf worden het 0, −1 en −2, en de kansen 0,665, 0,245 en 0,090.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Tien erbij
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    choice_add = L.choice(
        "add-ten",
        question="Een model geeft drie klassen de logits 2, 1 en 0. Wat gebeurt er met de kansen uit softmax als je bij elke logit 10 optelt?",
        options={
            "grow": "Ze worden allemaal groter",
            "first": "De eerste wordt groter, de andere kleiner",
            "right": "Niets: ze blijven gelijk",
        },
        correct="right",
        why={
            "grow": "Kunnen ze allemaal groter worden, terwijl ze samen 1 zijn? Zie *Het idee*.",
            "first": "De verschillen tussen de logits blijven gelijk. Waar kijkt softmax naar? Zie *Het idee*.",
            "right": "Elke $e^{z_k}$ wordt met $e^{10}$ vermenigvuldigd, boven en onder de streep, dus het valt weg: alleen de verschillen tellen.",
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_exercise_softmax():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Softmax als functie

        Maak `softmax(z)` af: hij krijgt een lijst logits en geeft hun kansen terug, als array. Vervang de `...` door stap 1 tot en met 3 van het recept. De check roept `softmax([3, 1, 0])` aan: dat moet ongeveer `array([0.844, 0.114, 0.042])` geven. Hij probeert ook andere logits, ook heel grote.
        """),
        L.task("Laat `softmax` de kansen van de logits `z` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softmax():
    box_softmax = L.code_box("softmax", lines=7)
    box_softmax
    return (box_softmax,)


@app.cell(hide_code=True)
def check_softmax(box_softmax):
    result_softmax = L.check("softmax", box_softmax.value)
    result_softmax
    return (result_softmax,)


@app.cell(hide_code=True)
def hints_softmax():
    L.hints(
        "softmax",
        [
            "Stap 1 tot en met 3 van het recept, in die volgorde, elk in één regel.",
            "Welke NumPy-methode geeft het grootste getal van een array? En waardoor deel je na `np.exp`?",
            "`e = np.exp(z - ___)` en `return e / ___`.",
        ],
        why="`z - z.max()` haalt de grootste logit eraf, `np.exp` maakt elk getal positief, en `e / e.sum()` maakt ze samen 1. Voor `[3, 1, 0]`: 0,844, 0,114 en 0,042.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Twee klassen

        Twee klassen, met de logits `z` en 0. Zet hun kansen uit softmax in `p`, als array van twee. Vervang de `...`. De laatste regel is `p`, zodat de check het ziet. De check probeert ook een andere `z`. Vergelijk de eerste kans met wat de sigmoid geeft bij `z`.
        """),
        L.task("Zet de softmax van `logits` in `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two():
    box_two = L.code_box("two-classes", lines=7)
    box_two
    return (box_two,)


@app.cell(hide_code=True)
def check_two(box_two):
    result_two = L.check("two-classes", box_two.value)
    result_two
    return (result_two,)


@app.cell(hide_code=True)
def hints_two():
    L.hints(
        "two-classes",
        [
            "Softmax op een array van twee logits, zoals in stap 2 en 3 van het recept; bij deze kleine logits is stap 1 niet nodig.",
            "Beide kansen tegelijk: $e$ tot de macht van elke logit, gedeeld door de som van beide. Welke NumPy-functie geeft de machten van een hele array?",
            "`p = np.exp(logits) / ___`.",
        ],
        why="`np.exp(logits)` is $e^{1{,}5} \\approx 4{,}48$ en $e^0 = 1$; gedeeld door hun som, 5,48, geeft dat 0,818 en 0,182. De eerste is de sigmoid van 1,5.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) werd één logit een kans bij twee klassen; softmax doet hetzelfde voor elk aantal klassen, en bij twee is het de sigmoid. In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) vatten `.max()` en `.sum()` een array samen; softmax gebruikt ze allebei. In de volgende les, *Cross-entropy-loss voor veel klassen*, meet je hoe goed zulke kansen zijn, zoals log loss dat deed bij twee klassen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, result_softmax, result_two):
    L.wrap_up(choice_add, result_softmax, result_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
