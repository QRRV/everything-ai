# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rnn-vanilla. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-vanilla", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl) en [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Een recurrent neural network leest een tekst van 50 tekens. Hoeveel verschillende matrices $\\mathbf{W}_h$ gebruikt het?",
        options={
            "steps": "50: één voor elke stap",
            "right": "Eén: dezelfde bij elke stap",
            "between": "49: één tussen elke twee stappen",
        },
        correct="right",
        why={
            "steps": "Een recurrent neural network gebruikt bij elke stap dezelfde weights, zodat het een tekst van elke lengte kan lezen. Zie [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl).",
            "between": "De hidden state gaat van stap naar stap, maar steeds door dezelfde $\\mathbf{W}_h$. Zie [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl).",
            "right": "Eén $\\mathbf{W}_h$, 50 keer gebruikt. Deze les rekent zijn gradient uit.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_factors():
    quick_factors = L.choice(
        "quick-factors",
        question="Op de terugweg door 30 layers wordt de gradient in elke layer vermenigvuldigd met een factor van ongeveer 0,8. Hij begint bij 1. Hoe groot is hij ongeveer bij de eerste layer?",
        options={
            "one": "Ongeveer 0,8, de factor van één layer",
            "tenth": "Ongeveer een tiende",
            "right": "Ongeveer een duizendste",
        },
        correct="right",
        why={
            "one": "Elke layer vermenigvuldigt met zijn factor, dus 30 factoren van 0,8 geven $0{,}8^{30}$. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "tenth": "De factoren worden met elkaar vermenigvuldigd: $0{,}8^{30}$ ligt ver onder 0,1, ook al ligt elke factor dicht bij 1. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "right": "$0{,}8^{30} \\approx 0{,}0012$: de gradient verdwijnt, ook met factoren niet ver onder 1. In deze les zijn de layers de stappen van een rij.",
        },
    )
    quick_factors
    return (quick_factors,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl) stelde je de weights van een recurrent neural network met de hand in. Om ze te leren, heeft het trainen hun gradients nodig. Maar $\mathbf{W}_h$ wordt bij elke stap van de tekst gebruikt, en elke stap hangt af van de stap ervoor. Hoe ga je daar terug doorheen? Het antwoord, **backpropagation through time**, laat ook zien waarom een gewoon recurrent neural network vergeet wat het lang geleden las, en waarom de netwerken van de volgende les bedacht zijn.
    """)
    return


@app.cell(hide_code=True)
def predict_reach():
    guess_reach = L.guess(
        "first-input",
        question="Een recurrent neural network met een hidden state van één getal leest 30 willekeurige inputs, de meeste tussen −1 en 1, met als weight $w = 0{,}9$. Hoeveel invloed heeft een kleine verandering in de eerste input op de laatste hidden state, vergeleken met dezelfde verandering in de laatste input?",
        options={
            "same": "Ongeveer evenveel",
            "power": "Ongeveer 0,05: dat is $0{,}9^{29}$",
            "right": "Minder dan een duizendste",
        },
        reveal=f"**Minder dan een duizendste:** ongeveer {L.number(C.reach(0.9)[29], 6)}. Elke stap terug vermenigvuldigt met $w$, maar ook met de helling van tanh bij die stap, en die ligt onder 1 overal waar de hidden state niet 0 is. Met inputs van deze grootte ligt de hidden state vaak ver van 0, waar die helling klein is, dus liggen veel factoren ruim onder 0,9, en 29 ervan laten bijna niets over. Wat het netwerk 30 stappen geleden las, komt nauwelijks aan bij het eind, en krijgt nauwelijks een gradient.",
    )
    guess_reach
    return (guess_reach,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Deze les gebruikt het kleinste recurrent neural network: een hidden state van één getal, $\mathbf{W}_x = 1$ en $\mathbf{b} = 0$. Dan is $\mathbf{W}_h$ één weight, $w$, en is elke stap
        """),
        L.formula(r"[[hidden:h_t]] = \tanh([[param:w]] [[hidden:h_{t-1}]] + [[data:x_t]])"),
        mo.md(r"""
        met $h_0 = 0$. Alles hieronder werkt ook voor matrices, met matrixproducten en het elementsgewijs product $\odot$ in plaats van de vermenigvuldigingen.

        **Uitrollen.** Schrijf het netwerk uit over een rij van $T$ stappen, de ene stap na de andere. Je krijgt een diep netwerk van $T$ layers, één per stap, waarin elke layer dezelfde weight $w$ heeft. Dat is een computational graph als elke andere, en de backward pass gaat er van het eind naar het begin doorheen, zoals in [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl). Terug door de stappen van een rij gaan heet **backpropagation through time**, of **BPTT**.
        """),
        mo.Html(C.unroll_picture({"title": "Eén weight, drie stappen", "note": "vooruit: dezelfde w bij elke stap",
                                  "alt": "Vier cirkels op een rij, h0 tot h3, met pijlen van elk naar de volgende, elk met het label w. Onder h1, h2 en h3 staat een vak met de input x1, x2 of x3, met een pijl omhoog. Boven de cirkels gaan pijlen in de kleur van de gradient terug van h3 naar h0, elk met het label maal w maal 1 min de h van die stap in het kwadraat."})),
        mo.md(r"""
        **Eén stap terug.** De backward pass begint bij de laatste hidden state, met gradient 1, alsof $h_T$ de loss is. Bij stap $t$ gaat de gradient die bij $h_t$ is teruggekomen door tanh, waarvan de helling daar $1 - h_t^2$ is. Dan splitst hij in tweeën:

        - **Het deel van stap $t$ voor $w$.** Bij stap $t$ vermenigvuldigt $w$ de $h_{t-1}$, dus deze stap telt de gradient maal $h_{t-1}$ op bij de gradient voor $w$.
        - **Door naar $h_{t-1}$.** $h_{t-1}$ wordt met $w$ vermenigvuldigd, dus de gradient gaat maal $w$ door naar de stap ervoor.

        **Tel de delen op.** $w$ wordt bij elke stap gebruikt, dus zijn gradient is de som van de delen van alle stappen, zoals de kettingregel elk pad van $w$ naar het eind optelt:
        """),
        L.formula(r"[[grad:\frac{\partial h_T}{\partial w}]] = \sum_{t=1}^{T} [[grad:\frac{\partial h_T}{\partial h_t}]] \, (1 - [[hidden:h_t]]^2) \, [[hidden:h_{t-1}]]"),
        mo.md(r"""
        **Veel factoren.** Elke stap terug vermenigvuldigt de gradient met de factor $w (1 - h_t^2)$: een weight maal de helling van een activation function, zoals elke layer deed in [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl). Een rij van 100 stappen is een netwerk van 100 layers. Factoren onder 1 laten de gradient van ver terug verdwijnen: de delen van vroege stappen voegen bijna niets toe, en het netwerk leert nauwelijks hoe een vroege input het eind moet veranderen. Met een matrix $\mathbf{W}_h$ die uitrekt, kan de gradient ook exploderen.

        De figuur: hoeveel elke input de laatste hidden state nog verandert, voor 30 willekeurige inputs, tegen hoeveel stappen voor het eind hij gelezen werd. De y-as heeft een logaritmische schaal: gelijke stappen omlaag zijn gelijke factoren.
        """),
        mo.Html(C.reach_picture({"title": "Hoe ver de gradient terugreikt", "x": "stappen terug vanaf het eind", "y": "deel dat het eind haalt",
                                 "w": "w = {w}", "lang": L.lang,
                                 "alt": "Twee dalende lijnen op een logaritmische schaal van 1 tot 10 tot de min 12, tegen 0 tot 29 stappen terug. Met w = 0,9 daalt de lijn tot ongeveer 0,03 na 10 stappen en ongeveer 0,00001 na 29. Met w = 0,5, gestreept, daalt hij veel sneller, tot ongeveer 0,0001 na 10 stappen en onder 10 tot de min 11 na 29."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **backpropagation through time** | BPTT, bee-pee-tee-tee | backpropagation voor een recurrent neural network: rol het uit over alle stappen van de rij en ga er terug doorheen | |
        | **uitrollen** | | een recurrent neural network uitschrijven als één layer per stap, allemaal met dezelfde weights | |
        | $\frac{\partial h_T}{\partial h_t}$ | partieel h T naar partieel h t | de gradient die van het eind is teruggekomen bij de hidden state van stap $t$ | `grad` |
        | $\frac{\partial h_T}{\partial w}$ | partieel h T naar partieel w | de gradient voor $w$: de som van de delen van alle stappen | `dw` |
        | **truncated BPTT** | | BPTT die maar teruggaat tot het begin van een stuk van de rij, zoals de laatste 100 stappen | |
        """),
        L.watch_out("BPTT moet elke hidden state van de forward pass bewaren: de backward pass heeft bij elke stap $h_t$ en $h_{t-1}$ nodig. Dus groeit het geheugen met de lengte van de rij, en worden lange teksten in stukken van bijvoorbeeld 100 stappen geknipt, waarbij de backward pass stopt aan het begin van elk stuk: **truncated BPTT**. De hidden state gaat nog wel door naar het volgende stuk, maar er komt geen gradient uit terug, dus het netwerk kan niets leren dat verder terugreikt dan een stuk."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de gradient van de laatste hidden state voor $w$ te vinden:

    1. Vooruit: begin bij $h_0 = 0$, reken elke stap uit, en bewaar elke hidden state.
    2. Begin de backward pass aan het eind, met gradient 1, en de gradient voor $w$ op 0.
    3. Voor elke stap $t$, van de laatste tot de eerste: vermenigvuldig de gradient met de helling van tanh, $1 - h_t^2$; tel dat maal $h_{t-1}$ op bij de gradient voor $w$; en vermenigvuldig hem met $w$ om door te gaan naar de stap ervoor.
    4. Na stap 1 bevat de gradient voor $w$ de delen van alle stappen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hoeveel verandert de eerste input de laatste hidden state, na 3 stappen met $w = 0{,}9$? Op de weg terug van $h_3$ naar $x_1$ wordt de gradient vermenigvuldigd met de factoren van stap 3 en 2, en dan met de helling van tanh bij stap 1. Het commentaar met `#` zegt welke stap van het recept elk deel is. De factor is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit. De autograd van PyTorch, uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl), geeft het getal om mee te vergelijken.
    """)
    return


@app.cell
def example_reach():
    import math
    import torch

    _xs, _w = [0.5, -0.3, 0.8], 0.9
    # 1. vooruit, en elke hidden state bewaren
    _hs = [0.0]
    for _x in _xs:
        _hs.append(math.tanh(_w * _hs[-1] + _x))
    # 2. de backward pass begint aan het eind, met gradient 1
    _grad = 1.0
    # 3. terug door stap 3 en 2, dan door tanh bij stap 1, naar x_1
    for _t in range(len(_xs), 1, -1):
        _grad = _grad * 1.0  # maak af: de factor van stap t, (1 - _hs[_t] ** 2) * _w
    _grad = _grad * (1 - _hs[1] ** 2)
    print("met de hand:", round(_grad, 4))
    # de autograd van PyTorch, om te vergelijken
    _xt = torch.tensor(_xs, requires_grad=True)
    _h = torch.tensor(0.0)
    for _t in range(len(_xs)):
        _h = torch.tanh(_w * _h + _xt[_t])
    _h.backward()
    print("PyTorch:", round(float(_xt.grad[0]), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, vermenigvuldigt de loop met 1, en toont de regel `met de hand` 0,7864: alleen de helling van tanh bij stap 1. Afgemaakt vermenigvuldigt hij met $0{,}9 \times (1 - 0{,}7182^2) \approx 0{,}436$ voor stap 3 en $0{,}9 \times (1 - 0{,}1154^2) \approx 0{,}888$ voor stap 2, en geeft hij 0,3044, hetzelfde als PyTorch. `range(len(_xs), 1, -1)` telt af in stappen van −1, van 3 tot net boven 1: stap 3 en 2. `math.tanh` is de tanh van één getal, zoals `np.tanh` dat is voor een array. De gradient voor $w$ gaat op dezelfde manier terug, en pikt onderweg bij elke stap een deel op.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bptt():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Backpropagation through time

        `bptt(xs, w)` laat het netwerk de inputs `xs` stap voor stap lezen, bewaart elke hidden state in `hs`, en geeft de gradient van de laatste hidden state voor `w` terug. De forward pass staat er al, en de loop terug over de stappen ook: die telt `t` af van `len(xs)` tot 1. Vervang de `...` door de drie regels van stap 3 van het recept, met `grad` de gradient die bij `hs[t]` is teruggekomen en `dw` de gradient voor `w`. De check roept `bptt([0.6, -0.2, 0.4], 0.8)` aan, wat ongeveer 0,4601 moet geven, en probeert één stap, een negatieve `w`, een `w` groter dan 1, en `w = 0`.
        """),
        L.task("Laat `bptt` terug door elke stap gaan en de delen van de gradient voor `w` optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bptt():
    box_bptt = L.code_box("bptt", lines=14)
    box_bptt
    return (box_bptt,)


@app.cell(hide_code=True)
def check_bptt(box_bptt):
    result_bptt = L.check("bptt", box_bptt.value)
    result_bptt
    return (result_bptt,)


@app.cell(hide_code=True)
def hints_bptt():
    L.hints(
        "bptt",
        [
            "Drie regels: eerst `local`, de gradient na tanh bij stap `t`; tel dan `local` maal de hidden state die stap `t` binnenkwam op bij `dw`; dan de nieuwe `grad` voor de stap ervoor.",
            "`local = grad * (1 - hs[t] ** 2)`. Bij stap `t` vermenigvuldigt `w` de `hs[t - 1]`: dat geeft het deel van stap `t` voor `dw`, en de factor door naar `hs[t - 1]`.",
        ],
        why="`local = grad * (1 - hs[t] ** 2)`, `dw = dw + local * hs[t - 1]` en `grad = local * w`. Voor `[0.6, -0.2, 0.4]` met `w = 0.8` zijn de delen van stap 3, 2 en 1 gelijk aan 0,164, 0,296 en 0: $h_0 = 0$, dus de eerste stap voegt niets toe.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw functie en PyTorch

    De cel hieronder geeft vijf inputs aan je `bptt`, en aan de autograd van PyTorch, die hetzelfde netwerk uitrolt en er terug doorheen gaat.
    """)
    return


@app.cell(hide_code=True)
def demo_autograd(box_bptt, result_bptt):
    _both = C.autograd_check(box_bptt.value) if result_bptt.state == "correct" else None
    mo.md(
        f"Je `bptt([0.5, -0.3, 0.8, 0.1, -0.7], 0.9)` geeft {L.number(_both[0], 4)}, en PyTorch geeft {L.number(_both[1], 4)}: je hebt gedaan wat `backward()` doet voor een recurrent neural network. In een echt netwerk is $w$ een matrix en begint de gradient bij een loss, maar de stappen zijn dezelfde."
        if _both else "Als je `bptt` goed is, wordt hij hier vergeleken met de autograd van PyTorch."
    )
    return


@app.cell(hide_code=True)
def text_exercise_shared():
    mo.md(r"""
    ### Oefening 2. Eén weight, veel stappen
    """)
    return


@app.cell(hide_code=True)
def choice_shared_cell():
    choice_shared = L.choice(
        "shared-grad",
        question="Een recurrent neural network leest 30 stappen met één $\\mathbf{W}_h$. Wat is de gradient van de loss voor $\\mathbf{W}_h$?",
        options={
            "last": "Het deel van de laatste stap: daar zit de loss",
            "right": "De som van de delen van alle 30 stappen",
            "mean": "Het gemiddelde van de delen van alle 30 stappen",
        },
        correct="right",
        why={
            "last": "$\\mathbf{W}_h$ wordt bij elke stap gebruikt, en elk gebruik verandert de loss. Het deel van de laatste stap laat de andere 29 weg.",
            "mean": "De kettingregel telt de paden van $\\mathbf{W}_h$ naar de loss op; hij deelt niet door hun aantal.",
            "right": "Eén weight, 30 keer gebruikt: elk gebruik telt zijn deel op, zoals `dw = dw + ...` deed in je `bptt`.",
        },
    )
    choice_shared
    return (choice_shared,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ### Oefening 3. Een lange tekst
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "long-text",
        question="Een gewoon recurrent neural network leest recensies van 200 woorden. Of een recensie positief is, hangt af van het eerste woord, en het label wordt gelezen uit de laatste hidden state. Wat gebeurt er bij het trainen?",
        options={
            "right": "Het leert dit nauwelijks: de gradient van het eerste woord verdwijnt over 199 stappen",
            "shared": "Het leert dit goed: de weights zijn gedeeld, dus elke stap krijgt dezelfde gradient",
            "explode": "De loss wordt `nan`, omdat 200 stappen de gradient altijd laten exploderen",
        },
        correct="right",
        why={
            "shared": "De weights zijn bij elke stap dezelfde, maar de gradient niet: van ver terug is hij met veel factoren vermenigvuldigd. Zie de figuur in *Het idee*.",
            "explode": "Niet altijd: elke factor is een weight maal de helling van tanh, die hooguit 1 is en vaak veel kleiner, dus liggen de factoren meestal onder 1 en verdwijnt de gradient. Hij explodeert alleen als de factoren uitrekken, boven 1.",
            "right": "199 factoren $w(1 - h_t^2)$ laten de gradient van het eerste woord verdwijnen. De volgende les geeft de hidden state een pad waarlangs de gradient veel minder vervaagt.",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het recurrent neural network van [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl), uitgerold tot een diep netwerk met dezelfde weight in elke layer, en de backward pass van [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl) erdoorheen. De factoren van [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) zijn nu stappen van de rij: een gewoon recurrent neural network vergeet, omdat zijn gradient van ver terug verdwijnt. In de volgende les, [LSTM en GRU](/course/open/rnn-lstm?lang=nl), beslissen gates wat de hidden state bewaart, zodat de gradient veel verder terug kan reiken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, choice_shared, result_bptt):
    L.wrap_up(result_bptt, choice_shared, choice_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
