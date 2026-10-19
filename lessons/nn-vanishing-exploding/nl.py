# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-vanishing-exploding. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-vanishing-exploding", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) en [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sigmoid():
    quick_sigmoid = L.choice(
        "quick-sigmoid",
        question="Wat geeft de sigmoid ongeveer bij $z = 10$, en bij $z = -10$?",
        options={
            "right": "Bijna 1, en bijna 0: aan beide kanten wordt hij vlak",
            "same": "10 en −10: een grote score blijft groot",
            "half": "Allebei 0,5: de sigmoid is symmetrisch",
        },
        correct="right",
        why={
            "same": "De sigmoid perst elke score in een kans tussen 0 en 1. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "half": "De sigmoid geeft alleen 0,5 bij $z = 0$; hoe verder van 0, hoe dichter bij 1 of 0. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "right": "$e^{-10}$ is piepklein, dus $\\sigma(10) \\approx 0{,}99995$; en $\\sigma(-10) \\approx 0{,}00005$. Ver van 0 is de kromme bijna vlak: dat heeft deze les nodig.",
        },
    )
    quick_sigmoid
    return (quick_sigmoid,)


@app.cell(hide_code=True)
def quick_check_keep():
    quick_keep = L.choice(
        "quick-keep",
        question="Waarom bewaart een backward pass de `Z1` uit de forward pass, de inputs van de ReLU?",
        options={
            "weights": "Om de weights te veranderen: gradient descent heeft hem nodig",
            "loss": "Om de loss opnieuw uit te rekenen: daar begint de backward pass",
            "right": "Voor de helling van de ReLU: 1 waar `Z1` boven 0 ligt, anders 0",
        },
        correct="right",
        why={
            "weights": "De weights veranderen met hun gradients, `dW1` en `dW2`. Zie [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).",
            "loss": "De loss komt uit `P`, aan het eind van de forward pass. Zie [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).",
            "right": "Terug door de ReLU wordt de gradient vermenigvuldigd met zijn helling bij `Z1`: `dH * (Z1 > 0)`. Elke layer vermenigvuldigt met zo'n helling, en deze les gaat over wat dat over veel layers doet.",
        },
    )
    quick_keep
    return (quick_keep,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een dieper netwerk kan meer leren, dus waarom niet vijftig layers stapelen? Lang werkte dat niet: de eerste layers van een diep netwerk veranderden nauwelijks, of de training stopte met een loss van `nan`. De oorzaak zit in de backward pass uit [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl): op de terugweg wordt de gradient één keer met elke layer vermenigvuldigd. Weten welke van de twee, en waarom, is de eerste stap om diepe netwerken überhaupt te kunnen trainen.
    """)
    return


@app.cell(hide_code=True)
def predict_chain():
    guess_chain = L.guess(
        "ten-sigmoids",
        question="Een keten van 10 layers, elk één sigmoid met weight 1, met de input 0,5. Bovenaan is de gradient 1. Hoe groot denk je dat hij is als hij terugkomt bij de input?",
        options={
            "quarter": "Ongeveer 0,25: de gradient gaat door één sigmoid tegelijk terug",
            "hundredth": "Ongeveer 0,01: hij krimpt in elke layer een beetje",
            "right": "Minder dan een miljoenste: hij krimpt in elke layer flink",
        },
        reveal="**Minder dan een miljoenste.** De grootste helling van de sigmoid is 0,25, bij $z = 0$. Elk van de 10 layers vermenigvuldigt de gradient met hooguit 0,25: $0{,}25^{10} \\approx 0{,}00000095$. Hier liggen de hellingen iets onder 0,25, en wat er terugkomt is ongeveer 0,00000035. De eerste layer van zo'n netwerk leert nauwelijks.",
    )
    guess_chain
    return (guess_chain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Neem een keten van $n$ layers met elk één getal: layer $k$ geeft $h_k = f(z_k)$, met $z_k = w_k h_{k-1}$, $h_0$ de input en $f$ de activation function. Volgens de kettingregel is de gradient die terugkomt bij de input de gradient bovenaan keer één **factor** per layer: zijn weight keer de helling van zijn activation function bij zijn $z$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial h_0}]] = [[grad:\frac{\partial \mathcal{L}}{\partial h_n}]] \cdot [[param:w_n]] f'(z_n) \cdot \ldots \cdot [[param:w_2]] f'(z_2) \cdot [[param:w_1]] f'(z_1)"),
        mo.md(r"""
        Veel factoren onder 1 maken het product piepklein: de gradient **verdwijnt** (een vanishing gradient), en de eerste layers leren nauwelijks. Veel factoren boven 1 maken het enorm: de gradient **explodeert** (een exploding gradient), de stappen van gradient descent schieten weg, en de loss springt omhoog of wordt `inf` of `nan`.

        De helling van de sigmoid, zoals in [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl), is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Die is hooguit 0,25, bij $z = 0$, en bijna 0 ver daarvandaan, waar de sigmoid bijna vlak is: daar is de sigmoid **verzadigd**, en laat hij bijna geen gradient door. In een keten van sigmoids verdwijnt de gradient dus, ook met grotere weights: die duwen $z$ weg van 0, de vlakke stukken in.

        De helling van ReLU is 1 waar $z > 0$, en 0 daarbuiten: wat hij doorlaat, blijft even groot, zonder een factor zoals de 0,25 van de sigmoid. Dan beslissen de weights. Met matrices vermenigvuldigt elke layer de gradient met zijn matrix, als `(g * (z > 0)) @ W`: te kleine weights laten hem krimpen, layer na layer; te grote laten hem groeien.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **vanishing gradient** | | een gradient die op de terugweg layer na layer krimpt, tot de eerste layers nauwelijks leren | |
        | **exploding gradient** | | een gradient die op de terugweg layer na layer groeit, tot de stappen wegschieten | |
        | **verzadiging** | | waar een activation function zoals de sigmoid bijna vlak is, zodat zijn helling, en de gradient die hij doorlaat, bijna 0 is | |
        | $\sigma'(z)$ | sigma-accent van z | de helling van de sigmoid bij $z$: $\sigma(z)(1 - \sigma(z))$, hooguit 0,25 | `sigmoid(z) * (1 - sigmoid(z))` |
        """),
        L.watch_out("Aan de loss alleen zie je niet welke van de twee je hebt. Verdwijnen ziet eruit als een loss die nauwelijks beweegt; exploderen als een loss die springt of `nan` wordt, wat een te grote learning rate ook kan doen. Kijk naar de grootte van de gradient per layer: piepklein in de eerste layers betekent verdwijnen, enorm betekent exploderen."),
        mo.md(r"""
        De figuur: de gemiddelde grootte van de gradient op de terugweg door 20 ReLU-layers van elk 50 neuronen, voor drie spreidingen van willekeurige beginweights. Te klein, en hij krimpt ongeveer elke 8 layers tien keer; te groot, en hij groeit; daartussen blijft hij ongeveer even groot. Die spreiding vinden is de volgende les.
        """),
        mo.Html(C.sizes_picture({"title": "De gradient op de terugweg door 20 ReLU-layers", "x": "layers terug vanaf boven",
                                 "y": "gemiddelde grootte van de gradient", "big": "weights te groot", "right": "precies goed", "small": "weights te klein",
                                 "alt": "Drie lijnen op een logaritmische schaal over 20 layers. Met te grote weights groeit de grootte van ongeveer 1,5 naar ongeveer 2600; met precies goede weights, gestippeld, blijft hij tussen ongeveer 1 en 4; met te kleine weights daalt hij van ongeveer 0,8 naar ongeveer 0,003."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de gradient terug te volgen door een keten van layers:

    1. Doe de forward pass, en bewaar elke $z$.
    2. Begin bovenaan met de gradient daar; hier 1.
    3. Ga terug door de layers, van de laatste naar de eerste, en vermenigvuldig met de factor van elke layer: zijn weight keer de helling van zijn activation function bij zijn $z$.
    4. Vergelijk wat er terugkomt met waarmee je begon: veel kleiner betekent verdwijnen, veel groter exploderen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De keten uit de gok: 10 sigmoid-layers, elk met weight 1, en de input 0,5. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit. `np.round(_zs, 3)` laat de $z$'s zien, afgerond op 3 decimalen.
    """)
    return


@app.cell
def example_chain():
    import numpy as np

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    _w = 1.0
    _h = 0.5
    _zs = []
    # 1. de forward pass door 10 layers, en bewaar elke z
    for _k in range(10):
        _z = _w * _h
        _zs.append(_z)
        _h = _sigmoid(_z)
    print(np.round(_zs, 3))
    # 2. de gradient bovenaan
    _g = 1.0
    # 3. terug van de laatste layer naar de eerste: keer de weight en de helling van de sigmoid
    for _z in reversed(_zs):
        _g = _g * _w * _sigmoid(_z) * (1 - _sigmoid(_z))
        print(_g)
    # 4. vergelijk: print hoeveel keer kleiner dan de 1 bovenaan het is, 1 / _g, en voer opnieuw uit;
    # daarna: helpen grotere weights? Zet _w op 4.0 en voer nog een keer uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Na een paar layers is elke $z$ ongeveer 0,66, waar de helling van de sigmoid ongeveer 0,22 is: elke layer vermenigvuldigt de gradient daarmee, van 0,225 naar 0,0505 en verder, tot ongeveer 0,00000035 bij de input: ongeveer 2,9 miljoen keer kleiner dan bovenaan. Met `_w = 4.0` is het erger: na de eerste twee layers zijn de $z$'s ongeveer 3,9, waar de sigmoid verzadigd is en zijn helling ongeveer 0,019; zelfs keer 4 vermenigvuldigt elk van die layers maar met ongeveer 0,076, en er komt ongeveer 0,00000000005 terug, ongeveer 18 miljard keer kleiner. Grotere weights duwen een sigmoid zijn vlakke stukken in.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nan():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een loss van nan
    """)
    return


@app.cell(hide_code=True)
def choice_nan_cell():
    choice_nan = L.choice(
        "nan-loss",
        question="Je traint een ReLU-netwerk van 30 layers. Na een paar stappen springt de loss omhoog, en daarna is hij `nan`. De learning rate werkte goed voor een netwerk van 3 layers. Wat is de waarschijnlijkste oorzaak?",
        options={
            "vanish": "Vanishing gradients: de eerste layers leren helemaal niet meer",
            "right": "Exploding gradients: de stappen van de eerste layers schieten weg",
            "data": "Een `nan` in de trainingsdata, in een van de afbeeldingen",
        },
        correct="right",
        why={
            "vanish": "Vanishing gradients laten de loss nauwelijks bewegen; ze laten hem niet springen of `nan` worden.",
            "data": "Een `nan` in de data geeft vanaf de eerste stappen een loss van `nan`, ook met 3 layers, zonder eerst omhoog te springen.",
            "right": "Met weights die de gradient oprekken, maken 30 factoren boven 1 hem enorm, en de stappen dus ook; de loss springt, en dan loopt een getal over naar `inf` en `nan`. Met 3 layers waren er te weinig factoren om het te merken.",
        },
    )
    choice_nan
    return (choice_nan,)


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Terug door een keten van sigmoids

        Schrijf `chain_gradient(ws, x)`: een keten van sigmoid-layers, één voor elke weight in de lijst `ws`: elke layer geeft `sigmoid(w * h)`, met `h` wat de layer ervoor gaf, te beginnen met `x`. Geef de gradient terug die bij `x` aankomt, te beginnen met 1 bovenaan, met het recept. `sigmoid` staat al in het vak. De check roept `chain_gradient([2.0, 1.0, 1.5], 0.5)` aan, en dat moet ongeveer `0.02529` geven; hij probeert ook andere weights en meer layers.
        """),
        L.task("Laat `chain_gradient` de gradient geven die door de keten bij `x` aankomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain-gradient", lines=14)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain-gradient", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain-gradient",
        [
            "Zoals in het uitgewerkte voorbeeld, maar met de weight van elke layer uit `ws`: bewaar in de forward pass elke `z` in een lijst.",
            "Ga op de terugweg door de layers, de laatste eerst, `for k in reversed(range(len(ws))):`, met de weight `ws[k]` en de `zs[k]` van elke layer.",
        ],
        why="Met de weights 2, 1 en 1,5 en de input 0,5 zijn de $z$'s 1, 0,731 en 1,013. De factor van elke layer, zijn weight keer de helling van de sigmoid bij zijn $z$, is $2 \\times 0{,}197$, $1 \\times 0{,}219$ en $1{,}5 \\times 0{,}195$: 0,393, 0,219 en 0,293, en hun product is ongeveer 0,025.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_relu():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De grootte van de gradient per layer

        Schrijf `relu_grad_sizes(Ws, x)`: een ReLU-netwerk met één weight-matrix per layer in de lijst `Ws`, waarin elke layer `np.maximum(h @ W.T, 0)` geeft, te beginnen met het voorbeeld `x`. Begin de terugweg met een gradient van enen bovenaan, `np.ones(len(h))`: zoals `np.zeros`, maar met enen. Terug door een layer houd je de gradient waar zijn `z` boven 0 lag, en vermenigvuldig je hem dan met `W`, zoals in [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl). Zet na elke layer de gemiddelde grootte van de gradient, `np.mean(np.abs(g))`, in een lijst, en geef de lijst terug, van de bovenste layer naar de eerste. De check roept `relu_grad_sizes([np.array([[1.0, -1.0], [0.5, 2.0]]), np.array([[2.0, 0.0], [-1.0, 1.0]])], np.array([1.0, -1.0]))` aan, en dat moet `[1, 2]` geven; hij probeert ook diepere en bredere netwerken.
        """),
        L.task("Laat `relu_grad_sizes` de grootte van de gradient na elke layer geven, op de terugweg."),
    ])
    return


@app.cell(hide_code=True)
def exercise_relu():
    box_relu = L.code_box("relu-sizes", lines=14)
    box_relu
    return (box_relu,)


@app.cell(hide_code=True)
def check_relu(box_relu):
    result_relu = L.check("relu-sizes", box_relu.value)
    result_relu
    return (result_relu,)


@app.cell(hide_code=True)
def hints_relu():
    L.hints(
        "relu-sizes",
        [
            "De forward pass bewaart elke `z`, zoals in oefening 2, maar nu `z = h @ W.T` en `h = np.maximum(z, 0)`. De terugweg begint met `g = np.ones(len(h))`.",
            "Op de terugweg, van de laatste layer naar de eerste, met `for k in reversed(range(len(Ws))):` is de nieuwe gradient `g = (g * (zs[k] > 0)) @ Ws[k]`; zet dan de grootte van deze `g` in de lijst.",
        ],
        why="Heen: $z_1 = (2; -1{,}5)$, $h_1 = (2; 0)$, $z_2 = (4; -2)$. Terug: de enen die blijven waar $z_2 > 0$ zijn $(1; 0)$, keer de tweede $W$ geeft $(2; 0)$, gemiddelde grootte 1; wat blijft waar $z_1 > 0$, $(2; 0)$, keer de eerste $W$ geeft $(2; -2)$, gemiddelde grootte 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De backward pass uit [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl) vermenigvuldigt de gradient één keer met elke layer; bij veel layers beslissen die factoren alles. De vlakke uiteinden uit [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) laten hem verdwijnen; weights met de verkeerde spreiding laten hem verdwijnen of exploderen. In de volgende les, *Initialisatie*, reken je de spreiding uit die het signaal layer na layer even groot houdt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nan, result_chain, result_relu):
    L.wrap_up(choice_nan, result_chain, result_relu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
