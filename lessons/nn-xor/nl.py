# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-xor. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-xor", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een neuron](/course/open/nn-neuron?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fire():
    quick_fire = L.choice(
        "quick-fire",
        question="Een neuron met een stap heeft de weights 2 en −1 en de bias term −1. Vuurt het bij de input (1, 1)?",
        options={
            "right": "Nee: $z$ is precies 0, en het vuurt alleen boven 0",
            "zero": "Ja: $z$ is 0, en 0 telt als vuren",
            "negative": "Nee: een van zijn weights is negatief, dus het vuurt nooit",
        },
        correct="right",
        why={
            "zero": "Reken $z = 2 \\cdot 1 - 1 \\cdot 1 - 1$ uit. Wanneer vuurt een stap: bij 0, of alleen erboven? Zie de tabel in *Het idee* van [Een neuron](/course/open/nn-neuron?lang=nl).",
            "negative": "Een negatieve weight trekt $z$ alleen omlaag. Bij (1, 0) is $z$ bijvoorbeeld 1. Zie het recept in [Een neuron](/course/open/nn-neuron?lang=nl).",
            "right": "$z = 2 \\cdot 1 - 1 \\cdot 1 - 1 = 0$. Een stap vuurt alleen als $z$ boven 0 ligt.",
        },
    )
    quick_fire
    return (quick_fire,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="Je verhoogt de bias term van een neuron en laat de weights met rust. Wat gebeurt er met de lijn waar $z = 0$?",
        options={
            "turn": "Hij draait rond een hoek van het vierkant",
            "right": "Hij schuift op, en houdt zijn richting",
            "nothing": "Niets: de bias term komt niet aan de lijn",
        },
        correct="right",
        why={
            "turn": "Wat bepaalt de richting van de lijn: de weights of de bias term? Zie de slider in [Een neuron](/course/open/nn-neuron?lang=nl).",
            "nothing": "Schuif met de slider in [Een neuron](/course/open/nn-neuron?lang=nl): wat doet de bias term met de lijn?",
            "right": "De weights bepalen de richting van de lijn; de bias term schuift hem alleen op.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Een neuron](/course/open/nn-neuron?lang=nl) rekende één neuron met een stap AND uit, en met een andere bias term rekent het OR uit. In 1958 presenteerde Frank Rosenblatt het perceptron: in de kern zo'n neuron, dat zijn weights zelf uit voorbeelden leert. Dat wekte hoge verwachtingen. In 1969 zochten Marvin Minsky en Seymour Papert in hun boek *Perceptrons* uit wat zo'n neuron wel en niet kan leren. Jarenlang werkten daarna weinig mensen aan neural networks. In deze les zoek je die grens zelf op.

    **Een voorbeeld uit het dagelijks leven.** Een traplicht heeft een schakelaar onderaan en een bovenaan. Zet je een van de twee om, dan verandert het licht, hoe de andere ook staat.
    """)
    return


@app.cell(hide_code=True)
def predict_xor():
    guess_xor = L.guess(
        "xor",
        question="Kan één neuron met een stap vuren bij de inputs (0, 1) en (1, 0), maar niet bij (0, 0) en (1, 1)?",
        options={
            "yes": "Ja, met de goede weights en bias term",
            "no": "Nee, wat de weights en de bias term ook zijn",
            "sigmoid": "Alleen met een sigmoid in plaats van een stap",
        },
        reveal="**Nee.** Een neuron vuurt aan één kant van een rechte lijn. (0, 1) en (1, 0) liggen op de ene diagonaal van het vierkant, (0, 0) en (1, 1) op de andere, en geen rechte lijn legt de twee hoeken van de ene diagonaal aan de ene kant en de twee van de andere aan de andere kant. Een sigmoid helpt niet: die trekt bij 0,5 nog steeds dezelfde rechte lijn. Deze taak heet XOR, en het is het traplicht.",
    )
    guess_xor
    return (guess_xor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Twee klassen zijn **lineair scheidbaar** als één rechte lijn ze zonder fouten splitst; bij meer dan twee inputs een plat vlak. Eén neuron trekt één zo'n lijn, dus het kan alleen taken leren waarvan de klassen lineair scheidbaar zijn. **XOR** geeft 1 als precies één van zijn twee inputs 1 is. Die klassen zijn niet lineair scheidbaar, en je hoeft niet elke lijn te proberen om dat te weten. Stel dat een neuron vuurt bij (1, 0) en (0, 1), en niet bij (0, 0):
        """),
        L.formula(r"[[param:w_1]] + [[param:b]] > 0 \qquad [[param:w_2]] + [[param:b]] > 0 \qquad [[param:b]] \le 0"),
        mo.md(r"""
        Tel de eerste twee op: $w_1 + w_2 + 2b > 0$, dus $w_1 + w_2 + b > -b \ge 0$. Dat is $z$ bij (1, 1): boven 0, dus het neuron vuurt ook bij (1, 1). Met geen enkele keuze van weights en bias term vuurt één neuron precies bij de inputs met één 1: het kan XOR niet uitrekenen.

        Het **perceptron** van Rosenblatt is een neuron met een stap dat leert van voorbeelden: na elk voorbeeld dat het fout heeft, schuift het zijn weights en bias term naar het label toe, met een learning rate $\eta$ keer $y - \hat{y}$:
        """),
        L.formula(r"[[param:\mathbf{w}]] \leftarrow [[param:\mathbf{w}]] + [[hyper:\eta]]\,(y - [[pred:\hat{y}]])\,\mathbf{x} \qquad [[param:b]] \leftarrow [[param:b]] + [[hyper:\eta]]\,(y - [[pred:\hat{y}]])"),
        mo.md(r"""
        Klopt de voorspelling, dan is $y - \hat{y}$ 0 en verandert er niets. Had het moeten vuren en deed het dat niet, dan is $y - \hat{y}$ 1, en schuiven de weights in de richting van de input; andersom de andere kant op. Kan één rechte lijn de klassen splitsen, dan stopt dit altijd bij weights zonder fouten. Zo niet, dan stopt het nooit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **lineair scheidbaar** | | twee klassen die één rechte lijn, of een plat vlak, zonder fouten splitst | |
        | **XOR** | ex-or | 1 als precies één van twee inputs 1 is, anders 0 | `x1 != x2` |
        | **perceptron** | | een neuron met een stap dat leert van voorbeelden, door na elke fout zijn weights naar het label toe te schuiven (Rosenblatt, 1958) | `w = w + lr * (y - pred) * x` |
        | $\leftarrow$ | wordt | de nieuwe waarde links wordt uitgerekend uit de oude rechts | `w = w + ...` |
        """),
        L.watch_out("XOR is niet moeilijk voor een computer: `x1 != x2` doet het. Het is moeilijk voor één neuron, omdat een neuron maar één rechte lijn kan trekken."),
        mo.md(r"""
        Probeer het zelf. De sliders zetten de tweede weight en de bias term van een neuron met een stap; de eerste weight is 1. De gevulde stippen zijn de inputs waar XOR 1 geeft, de open stippen die waar het 0 geeft. Een <span class="c-loss">paarse ring</span> markeert elke input die het neuron fout heeft. Krijg je alle ringen weg?
        """),
    ])
    return


@app.cell(hide_code=True)
def slider_w2_cell():
    slider_w2 = L.slider("w2", "w₂", start=1, low=-2, high=2, step=0.1, role="param")
    slider_w2
    return (slider_w2,)


@app.cell(hide_code=True)
def slider_b_cell():
    slider_b = L.slider("bias", "b", start=-0.5, low=-2, high=1, step=0.1, role="param")
    slider_b
    return (slider_b,)


@app.cell(hide_code=True)
def picture_xor(slider_b, slider_w2):
    _w2 = round(slider_w2.widget.value, 1)
    _b = round(slider_b.widget.value, 1)
    _wrong = C.mistakes(_w2, _b)
    _which = ", ".join(str(_c) for _c in C.wrong_corners(_w2, _b))
    mo.vstack([
        mo.Html(C.picture(_w2, _b, {"title": "XOR en één rechte lijn",
                                    "alt": f"De vier hoeken van het vierkant: (0, 1) en (1, 0) gevuld, (0, 0) en (1, 1) open, en de lijn waar z = 0 van een neuron met de weights 1 en {L.number(_w2, 1)} en de bias term {L.number(_b, 1)}. Een ring markeert elke hoek die het neuron fout heeft: {_which}."})),
        mo.md(f"""Met $w_2$ = {L.number(_w2, 1)} en $b$ = {L.number(_b, 1)} heeft het neuron <span class="c-loss ea-num">{_wrong}</span> van de 4 inputs fout."""),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Hoe je de sliders ook zet, er blijft minstens één ring over.

    ### Recept

    Zo leert het perceptron, met een learning rate `lr`:

    1. Voorspel bij elk voorbeeld met een stap: `pred` is 1 als `np.dot(w, x) + b > 0`, anders 0.
    2. Verschuif de weights en de bias term: `w = w + lr * (y - pred) * x` en `b = b + lr * (y - pred)`. Klopt `pred`, dan verandert er niets.
    3. Ga alle voorbeelden opnieuw langs, ronde na ronde, tot een ronde geen fouten heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het perceptron op AND en op XOR: elk tien rondes over de vier inputs, met alle weights en de bias term eerst op 0, en een learning rate van 0,5. Na elke ronde onthoudt de cel hoeveel inputs het in die ronde fout had. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_perceptron():
    import numpy as np

    _inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _name, _labels in [("AND", [0, 0, 0, 1]), ("XOR", [0, 1, 1, 0])]:
        _w = np.zeros(2)
        _b = 0.0
        _per_round = []
        # 3. ronde na ronde
        for _round in range(10):
            _mistakes = 0
            for _i in range(4):
                _x = np.array(_inputs[_i])
                _y = _labels[_i]
                # 1. voorspel met een stap
                if np.dot(_w, _x) + _b > 0:
                    _pred = 1
                else:
                    _pred = 0
                if _pred != _y:
                    _mistakes = _mistakes + 1
                # 2. schuif naar het label toe
                _w = _w + 0.5 * (_y - _pred) * _x
                _b = _b + 0.5 * (_y - _pred)
            _per_round.append(_mistakes)
        print(_name, _per_round, _w, _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op AND maakt het perceptron 1, 3, 3, 2 en 1 fouten in zijn eerste vijf rondes, en daarna geen meer: het heeft de weights 1 en 0,5 en de bias term −1 gevonden, een lijn met alleen (1, 1) erboven. Op XOR lukt dat nooit: vanaf de derde ronde heeft het alle vier de inputs fout, ronde na ronde. Elke fout schuift de lijn op, en zet zo de ene input goed en een andere fout.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_line():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén lijn of niet?

    Zes taken met twee inputs, elk 0 of 1. Zet elke taak in een groep: één rechte lijn kan de enen van de nullen scheiden, of geen rechte lijn kan dat. Teken de vier hoeken als dat helpt, of gebruik de sliders.
    """)
    return


@app.cell(hide_code=True)
def sort_line_cell():
    sort_line = L.sort("one-line")
    sort_line
    return (sort_line,)


@app.cell(hide_code=True)
def check_line(sort_line):
    result_line = L.check("one-line", sort_line)
    result_line
    return (result_line,)


@app.cell(hide_code=True)
def hints_line():
    L.hints(
        "one-line",
        [
            "Markeer bij elke taak de hoeken die 1 geven. Is er een lijn met die aan de ene kant en de rest aan de andere?",
            "Eén hoek, drie hoeken of één kant van het vierkant kun je met een lijn afsnijden. En twee hoeken op een diagonaal?",
            "Eén lijn: AND, OR, \\_\\_\\_ en \\_\\_\\_. Geen lijn: XOR en \\_\\_\\_.",
        ],
        why="**Eén lijn:** AND (één hoek), OR en NAND (drie hoeken), en de eerste input (één kant). **Geen lijn:** XOR en gelijk, waarvan de enen op de ene diagonaal liggen en de nullen op de andere.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén stap van het perceptron

        Maak `perceptron_step(x, y, w, b, lr)` af: hij krijgt één voorbeeld, de inputs `x` en het label `y`, de weights `w`, de bias term `b` en de learning rate `lr`. De voorspelling `pred` staat er al. Vervang de `...` door stap 2 van het recept, zodat de functie de nieuwe weights en de nieuwe bias term teruggeeft. De check roept `perceptron_step([1, 1], 1, [0.0, 0.0], 0.0, 0.5)` aan: dat moet `([0.5, 0.5], 0.5)` geven. Hij probeert ook voorbeelden die het neuron goed heeft, voorbeelden waarbij het vuurt maar dat niet had moeten doen, en een voorbeeld met drie inputs.
        """),
        L.task("Laat `perceptron_step` de weights en de bias term teruggeven na één stap van het perceptron."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("perceptron-step", lines=13)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("perceptron-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "perceptron-step",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld, met `lr` in plaats van 0,5.",
            "`y - pred` is 0, 1 of −1. Wat doet dat met de stap als de voorspelling klopt?",
            "`w = w + lr * (___) * x` en `b = b + ___`.",
        ],
        why="Met `w` en `b` helemaal 0 is $z$ 0 en vuurt het neuron niet, maar het label is 1: `y - pred` is 1. Dus groeit elke weight met 0,5 keer zijn input, 1, en de bias term met 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_follows():
    mo.md(r"""
    ### Oefening 3. Wat volgt eruit?
    """)
    return


@app.cell(hide_code=True)
def choice_follows_cell():
    choice_follows = L.choice(
        "follows",
        question="Een neuron met een stap vuurt bij (1, 0) en (0, 1), dus $w_1 + b > 0$ en $w_2 + b > 0$. Het vuurt niet bij (0, 0), dus $b \\le 0$. Wat volgt eruit voor (1, 1), waar $z = w_1 + w_2 + b$?",
        options={
            "below": "$z$ is 0 of lager, dus het vuurt niet bij (1, 1)",
            "depends": "Niets: het hangt af van de weights die je kiest",
            "right": "$z$ ligt boven 0, dus het vuurt ook bij (1, 1)",
        },
        correct="right",
        why={
            "below": "Tel de eerste twee op: $w_1 + w_2 + 2b > 0$. Wat zegt dat over $w_1 + w_2 + b$, als $b \\le 0$? Zie *Het idee*.",
            "depends": "De drie feiten gelden voor elk neuron dat vuurt bij (1, 0) en (0, 1) maar niet bij (0, 0). Tel de eerste twee op. Zie *Het idee*.",
            "right": "$w_1 + w_2 + 2b > 0$, dus $w_1 + w_2 + b > -b \\ge 0$. Elk zo'n neuron vuurt ook bij (1, 1), dus geen enkel neuron rekent XOR uit.",
        },
    )
    choice_follows
    return (choice_follows,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een neuron](/course/open/nn-neuron?lang=nl) trok één neuron één rechte lijn. Dat is precies zijn grens: het kan alleen klassen leren die een rechte lijn splitst, en XOR hoort daar niet bij. De uitweg is geen beter neuron, maar meer neuronen. In de volgende les, *Een hidden layer en ReLU*, voeden neuronen andere neuronen, en samen buigen ze de grens.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_follows, result_line, result_step):
    L.wrap_up(result_line, result_step, choice_follows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
