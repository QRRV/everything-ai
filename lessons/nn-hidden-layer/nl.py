# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-hidden-layer. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-hidden-layer", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_why():
    quick_why = L.choice(
        "quick-why",
        question="Waarom kan één neuron met een stap XOR niet uitrekenen?",
        options={
            "right": "De enen en nullen liggen op twee diagonalen: geen rechte lijn splitst die",
            "coarse": "Een stap is te grof: met een sigmoid zou één neuron het kunnen",
            "inputs": "Het heeft te weinig inputs: met een derde input zou één neuron het kunnen",
        },
        correct="right",
        why={
            "coarse": "Een sigmoid trekt dezelfde rechte lijn, bij 0,5. Zie de gok in [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).",
            "inputs": "XOR heeft twee inputs; het probleem is waar de enen liggen. Zie *Het idee* in [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).",
            "right": "(0, 1) en (1, 0) op de ene diagonaal, (0, 0) en (1, 1) op de andere: geen rechte lijn splitst ze.",
        },
    )
    quick_why
    return (quick_why,)


@app.cell(hide_code=True)
def quick_check_rounds():
    quick_rounds = L.choice(
        "quick-rounds",
        question="Het perceptron draait op XOR, ronde na ronde. Wat gebeurt er?",
        options={
            "settles": "Na een paar rondes komt het tot rust, zoals bij AND",
            "right": "Het blijft fouten maken, ronde na ronde",
            "small": "Het komt tot rust zodra de learning rate klein genoeg is",
        },
        correct="right",
        why={
            "settles": "Bij AND splitst één rechte lijn de klassen; bij XOR geen enkele. Zie het uitgewerkte voorbeeld in [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).",
            "small": "Een kleinere learning rate maakt alleen de stappen kleiner; weights zonder fouten bestaan niet. Zie [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).",
            "right": "Er bestaan geen weights zonder fouten, dus elke ronde gaat er een input mis en schuift de lijn weer op.",
        },
    )
    quick_rounds
    return (quick_rounds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Eén neuron trekt één rechte lijn, en geen rechte lijn splitst XOR. Toch zijn de netwerken die je in deze course traint, tot en met het language model, uit zulke neuronen gebouwd, en ze leren veel meer dan rechte lijnen. Hoe kan dat?

    **Een voorbeeld uit het dagelijks leven.** Voor een taart klop je de eieren en smelt je de boter apart, en pas daarna meng je ze. Die twee tussenresultaten zijn de taart niet, maar de taart heeft ze nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Twee neuronen met een stap zien allebei de twee inputs. Een derde neuron met een stap ziet alleen hun twee outputs. Kunnen deze drie neuronen samen XOR uitrekenen?",
        options={
            "no": "Nee: elk neuron trekt nog steeds maar één rechte lijn",
            "right": "Ja: het derde neuron kan de lijnen van de andere twee combineren",
            "inputs": "Alleen als het derde neuron ook de inputs ziet",
        },
        reveal="**Ja.** Het eerste neuron vuurt als minstens één input 1 is, het tweede als beide dat zijn. Het derde vuurt als het eerste vuurt en het tweede niet: bij precies één 1. Elk neuron trekt nog steeds één rechte lijn, maar het derde trekt zijn lijn door de outputs van de andere twee, en vanuit de inputs gezien snijdt dat een strook uit tussen twee lijnen.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **layer** is een groep neuronen die allemaal dezelfde inputs krijgen, elk met eigen weights en een eigen bias term. Een **hidden layer** zit tussen de inputs en de output: zijn outputs $\mathbf{h}$ zie je nooit direct, alleen wat de neuronen erna ermee doen. Layers na elkaar vormen een **multilayer perceptron**, of **MLP**, ook als de neuronen een andere activation function dan een stap hebben.

        Voor XOR zijn drie neuronen met een stap nodig. Hidden neuron 1 vuurt als minstens één input 1 is, hidden neuron 2 als beide dat zijn, en het output-neuron vuurt als het eerste vuurt en het tweede niet, met de weights 1 en −1 en de bias term −0,5 van het neuron dat in [Een neuron](/course/open/nn-neuron?lang=nl) bij één hoek vuurde:
        """),
        L.formula(r"\begin{aligned} h_1 &= \mathrm{stap}(x_1 + x_2 - 0{,}5) \\ h_2 &= \mathrm{stap}(x_1 + x_2 - 1{,}5) \\ [[pred:\hat{y}]] &= \mathrm{stap}(h_1 - h_2 - 0{,}5) \end{aligned}"),
        mo.md(r"""
        Elk hidden neuron trekt een lijn; het output-neuron vuurt ertussen. Maar de hidden neurons hebben een activation function nodig die zelf geen rechte lijn is. Zonder zo'n functie geeft hidden neuron $i$ zijn gewogen som $z = \mathbf{w}_i \cdot \mathbf{x} + b_i$ door, en telt het output-neuron, met de weights $v_1$ en $v_2$ en de bias term $c$, die op: dat is weer één gewogen som van de inputs plus één bias term, één rechte lijn.
        """),
        L.formula(r"\begin{aligned} & v_1(\mathbf{w}_1 \cdot \mathbf{x} + b_1) + v_2(\mathbf{w}_2 \cdot \mathbf{x} + b_2) + c \\ ={} & (v_1\mathbf{w}_1 + v_2\mathbf{w}_2) \cdot \mathbf{x} + (v_1 b_1 + v_2 b_2 + c) \end{aligned}"),
        mo.md(r"""
        Een stap is geen rechte lijn, maar hij heeft een gebrek: hij is overal vlak behalve bij 0, dus gradient descent krijgt geen helling om te volgen. Hidden layers gebruiken daarom meestal **ReLU**: $\mathrm{ReLU}(z) = \max(0, z)$. Onder 0 geeft hij 0, boven 0 geeft hij $z$ door zoals hij is. Die knik bij 0 is genoeg om het netwerk meer te maken dan één rechte lijn, en boven 0 heeft hij een helling van 1. Onder 0 is ReLU ook vlak, dus een neuron waarvan $z$ onder 0 blijft, krijgt geen helling en blijft op 0 steken. **Leaky ReLU** laat onder 0 een beetje door, zoals $0{,}01 \cdot z$, zodat dat niet kan gebeuren.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **layer** | | neuronen die allemaal dezelfde inputs krijgen, elk met eigen weights en een eigen bias term | |
        | **hidden layer** | | een layer tussen de inputs en de output, waarvan je de outputs alleen ziet via de neuronen erna | |
        | $\mathbf{h}$ | h | de outputs van een hidden layer, één per hidden neuron | `h` |
        | **ReLU** | reloe | de activation function $\max(0, z)$: 0 onder 0, $z$ zelf erboven | `max(0, z)` |
        | **leaky ReLU** | | zoals ReLU, maar met een kleine helling onder 0, zoals $0{,}01 \cdot z$ | |
        | **MLP**, multilayer perceptron | em-el-pee | layers na elkaar, met een activation function na elke hidden layer | |
        """),
        L.watch_out("Tel de layers met weights. Het XOR-netwerk heeft er twee: de hidden layer en het output-neuron. De inputs zijn geen layer van neuronen; ze hebben geen eigen weights."),
        mo.md(r"""
        Schuif met de slider: hij zet de bias term $b_2$ van hidden neuron 2. De <span class="c-pred">blauwe stipjes</span> laten zien waar het netwerk vuurt; de gestreepte lijnen zijn de twee hidden neurons. De hoeken zijn XOR: gevuld voor 1, open voor 0. Een <span class="c-loss">paarse ring</span> markeert elke hoek die het netwerk fout heeft, en de andere grafiek telt ze voor elke $b_2$; sleep eroverheen om $b_2$ te zetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b2_cell():
    def _frame(b2):
        b2 = round(b2, 1)
        top = 0.0 - b2   # hidden neuron 2 vuurt waar x₁ + x₂ hierboven ligt; 0.0 - houdt −0,0 uit de tekst
        out = C.outputs(b2)
        wrong = C.wrong(b2)
        ringed = ", gemarkeerd met een paarse ring: " + " en ".join(str(c) for c in C.wrong_corners(b2)) if wrong else ""
        if top >= 2.5:   # de tweede lijn raakt alleen de hoek rechtsboven van het vlak
            where = "vuurt het netwerk overal boven de eerste gestreepte lijn; de tweede raakt alleen de hoek rechtsboven van het vlak"
        elif top > 0.5:
            where = f"vuurt het netwerk waar x₁ + x₂ boven 0,5 ligt maar niet boven {L.number(top, 1)}, tussen de twee gestreepte lijnen"
        else:
            where = "ligt de tweede gestreepte lijn op of onder de eerste, en vuurt het netwerk nergens"
        return (C.picture(b2, {"title": "Waar het netwerk vuurt", "wrong_title": "Hoeken fout, voor elke b₂", "wrong": "hoeken",
                               "alt": f"Het vlak van de twee inputs met de hoeken van XOR. Bij b₂ = {L.number(b2, 1)} {where}; het heeft {wrong} van de 4 hoeken fout{ringed}. Een tweede grafiek telt de foute hoeken voor elke b₂: één tot en met −2, geen tot en met −1, en twee daarboven."}),
                f"""Met <i>b</i>₂ = {L.number(b2, 1)} geeft het netwerk {", ".join(str(o) for o in out)} voor de vier hoeken, en heeft het er <span class="c-loss ea-num">{wrong}</span> fout.""")

    explore_b2 = L.explore("b2", "b₂", _frame, start=-1.5, low=-2.5, high=0, step=0.1, role="param", task="Bij welke $b_2$ rekent het netwerk XOR uit? En welke hoeken heeft het fout daaronder en daarboven?")
    explore_b2
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met $b_2$ boven −2 en tot en met −1 bevat de strook (0, 1) en (1, 0) en verder geen hoeken: XOR. Bij −2 en lager vuurt hidden neuron 2 bij geen enkele hoek, en is het netwerk OR: het heeft (1, 1) fout. Boven −1 is de strook te smal voor een hoek, en heeft het netwerk (0, 1) en (1, 0) fout; vanaf −0,5 is de strook weg.

    ### Recept

    Zo reken je uit wat een netwerk met één hidden layer geeft voor de inputs `x`, met de weights `W` van de hidden layer, één rij per hidden neuron, zijn bias terms `b`, en de weights `v` en de bias term `c` van het output-neuron:

    1. Voor elk hidden neuron `i`: `z = np.dot(W[i], x) + b[i]`, en dan geeft de activation function, ReLU of een stap, `h[i]`.
    2. Het output-neuron ziet `h` als zijn inputs: `np.dot(v, h) + c`, met een eigen activation function of zonder.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het XOR-netwerk van *Het idee*, met een stap in alle drie de neuronen, op alle vier de inputs. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_xor():
    import numpy as np

    def _step(z):
        if z > 0:
            return 1
        return 0

    _W = [[1, 1], [1, 1]]   # één rij weights per hidden neuron
    _b = [-0.5, -1.5]
    _v = [1, -1]
    _c = -0.5
    for _x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # 1. de hidden layer
        _h = []
        for _i in range(2):
            _h.append(_step(np.dot(_W[_i], _x) + _b[_i]))
        # 2. het output-neuron
        print(_x, _h, _step(np.dot(_v, _h) + _c))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij (0, 0) vuurt geen van beide hidden neurons. Bij (0, 1) en (1, 0) vuurt alleen het eerste, en krijgt het output-neuron $1 - 0 - 0{,}5 = 0{,}5$: het vuurt. Bij (1, 1) vuren ze allebei, en krijgt het $1 - 1 - 0{,}5 = -0{,}5$: het vuurt niet. De hidden layer heeft de inputs omgezet in nieuwe features, $h_1$ en $h_2$, en daarop is XOR één rechte lijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_collapse():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een hidden layer zonder activation function
    """)
    return


@app.cell(hide_code=True)
def choice_collapse_cell():
    choice_collapse = L.choice(
        "collapse",
        question="Een netwerk heeft een hidden layer van 100 neuronen zonder activation function, en een output-neuron zonder. Wat kan het leren?",
        options={
            "lines": "Veel meer dan één neuron: de 100 hidden neurons trekken 100 verschillende lijnen, en het output-neuron combineert ze",
            "nothing": "Helemaal niets: zonder activation function geeft de hidden layer 0",
            "right": "Alleen wat één neuron zonder activation function kan: één gewogen som van de inputs plus een bias term",
        },
        correct="right",
        why={
            "lines": "De lijnen buigen de grens alleen als er daarna iets niet-lineairs komt. Wat is een gewogen som van gewogen sommen? Zie de tweede formule in *Het idee*.",
            "nothing": "Zonder activation function geeft een neuron $z$ zelf door, niet 0. Wat maakt het output-neuron daarvan? Zie de tweede formule in *Het idee*.",
            "right": "Een gewogen som van gewogen sommen is één gewogen som, met één bias term: de 100 neuronen vallen samen tot één. Alleen een activation function zoals ReLU houdt ze uit elkaar.",
        },
    )
    choice_collapse
    return (choice_collapse,)


@app.cell(hide_code=True)
def text_exercise_forward():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Door de hidden layer

        Maak `forward(x, W, b, v, c)` af: een netwerk met een hidden layer met ReLU en een output-neuron zonder activation function. `W` heeft één rij weights per hidden neuron en `b` één bias term per hidden neuron; `v` en `c` zijn de weights en de bias term van het output-neuron. Vervang de `...` door stap 1 van het recept voor hidden neuron `i`, met ReLU, zodat zijn output aan `h` wordt toegevoegd. De check roept `forward([1, 2], [[1, -1], [0.5, 0.5]], [0, -1], [2, 1], 0.5)` aan: dat moet `1.0` geven. Hij probeert ook andere inputs en netwerken, met meer inputs en meer hidden neurons.
        """),
        L.task("Laat `forward` elk hidden neuron uitrekenen met ReLU en zijn output aan `h` toevoegen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=9)
    box_forward
    return (box_forward,)


@app.cell(hide_code=True)
def check_forward(box_forward):
    result_forward = L.check("forward", box_forward.value)
    result_forward
    return (result_forward,)


@app.cell(hide_code=True)
def hints_forward():
    L.hints(
        "forward",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld, met ReLU in plaats van de stap.",
            "Hidden neuron `i` heeft de weights `W[i]` en de bias term `b[i]`. Wat doet ReLU met een `z` onder 0, en met een `z` boven 0?",
            "`z = np.dot(___, x) + ___`, dan `h.append(max(___, z))`.",
        ],
        why="Bij `[1, 2]` heeft het eerste hidden neuron (`i = 0`) $z = 1 - 2 + 0 = -1$, en ReLU maakt dat 0. Het tweede (`i = 1`) heeft $z = 0{,}5 + 1 - 1 = 0{,}5$, en dat geeft ReLU door. De output is $2 \\cdot 0 + 1 \\cdot 0{,}5 + 0{,}5 = 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_output():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. XOR met ReLU

        Twee hidden neurons met ReLU: $h_1 = \mathrm{ReLU}(x_1 + x_2)$ en $h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$. Het output-neuron heeft geen activation function: $\hat{y} = v_1 h_1 + v_2 h_2 + c$. Kies de weights `v` en de bias term `c` zo dat het netwerk 0, 1, 1 en 0 geeft voor (0, 0), (0, 1), (1, 0) en (1, 1): XOR. Vervang de twee `...`, en laat de loop zoals hij is. Wil je er een paar proberen? Doe dat in het kladblok.
        """),
        L.task("Kies `v` en `c` zo dat het netwerk XOR uitrekent."),
    ])
    return


@app.cell(hide_code=True)
def exercise_output():
    box_output = L.code_box("output-weights", lines=14)
    box_output
    return (box_output,)


@app.cell(hide_code=True)
def check_output(box_output):
    result_output = L.check("output-weights", box_output.value)
    result_output
    return (result_output,)


@app.cell(hide_code=True)
def hints_output():
    L.hints(
        "output-weights",
        [
            "Reken eerst $h_1$ en $h_2$ uit voor elke hoek. Dan: welke output wil XOR daar?",
            "(0, 0) geeft $h = (0; 0)$, dus de output is daar gewoon $c$. (0, 1) en (1, 0) geven $(1; 0)$, en (1, 1) geeft $(2; 1)$.",
            "$c = 0$ en $v_1 = 1$ volgen uit de eerste drie hoeken. Dan (1, 1): $2 + v_2 = 0$, dus `v = [1, ___]`.",
        ],
        why="Met `v = [1, -2]` en `c = 0`: (0, 0) geeft 0, (0, 1) en (1, 0) geven 1, en (1, 1) geeft $2 - 2 \\cdot 1 = 0$. Het eerste hidden neuron telt de enen; het tweede begint pas boven één 1, en de output haalt dat twee keer weg.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl) was één rechte lijn niet genoeg voor XOR. Een hidden layer met een niet-lineaire activation function, een stap of ReLU, geeft het output-neuron nieuwe features, waarop XOR één rechte lijn is. In de volgende les, *De forward pass in NumPy, met shapes*, reken je een hele layer tegelijk uit met een matrix, en volg je de shapes van layer naar layer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_collapse, result_forward, result_output):
    L.wrap_up(choice_collapse, result_forward, result_output)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
