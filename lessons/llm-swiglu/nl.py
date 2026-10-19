# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-swiglu. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-swiglu", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Het transformer block](/course/open/attn-transformer-block?lang=nl) en [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mlp():
    quick_mlp = L.choice(
        "quick-mlp",
        question="Welk deel van een block van je GPT werkt op elk token apart, zonder naar de andere tokens te kijken?",
        options={
            "attention": "De attention: elk token kijkt alleen naar zichzelf",
            "right": "Het MLP: dezelfde layers, token voor token",
            "both": "Allebei: elk deel van een block werkt per token",
        },
        correct="right",
        why={
            "attention": "Bij attention kijken tokens juist naar elkaar, naar alle tokens ervoor. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
            "right": "Attention mengt de tokens; het MLP bewerkt daarna de vector van elk token apart. Deze les verandert dat MLP.",
            "both": "Attention mengt informatie tussen de tokens; alleen het MLP en de normalisaties werken per token. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
        },
    )
    quick_mlp
    return (quick_mlp,)


@app.cell(hide_code=True)
def quick_check_relu():
    quick_relu = L.choice(
        "quick-relu",
        question="Wat geeft ReLU bij een input van $-2$, en bij een input van $3$?",
        options={
            "abs": "2 en 3",
            "neg": "$-2$ en 3",
            "right": "0 en 3",
        },
        correct="right",
        why={
            "abs": "ReLU draait negatieve getallen niet om: hij maakt ze 0. Zie [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "neg": "Dat is de input zelf. ReLU houdt positieve getallen en maakt negatieve 0: zie [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "right": "En zijn helling is 0 bij elke negatieve input: daar komt geen gradient door.",
        },
    )
    quick_relu
    return (quick_relu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het MLP van je GPT bestaat uit twee layers met een ReLU ertussen, zoals in de transformer van 2017. GPT-2 gebruikte al GELU in plaats van ReLU, en Llama, Mistral en PaLM gebruiken **SwiGLU**: een gladde activation function en een gate, een extra layer waarvan de output de hidden getallen getal voor getal vermenigvuldigt. Het MLP bevat ongeveer twee derde van de parameters van een groot model, dus hoe het gebouwd is, doet ertoe.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "llama-hidden",
        question="Llama 2 7B heeft vectoren van $d = 4096$ getallen. Een klassiek MLP zou ze verbreden tot $4d = 16.384$; zijn SwiGLU verbreedt ze tot 11.008. Waarom dat getal?",
        options={
            "memory": "Het is de grootste breedte waarvan de weights in het geheugen van één GPU passen",
            "slow": "Zijn gladde activation function kost meer tijd dan ReLU; een kleinere breedte maakt dat goed",
            "right": "Drie weight-matrices in plaats van twee; een kleinere breedte houdt het aantal parameters gelijk",
        },
        reveal="**Drie matrices in plaats van twee.** Een klassiek MLP heeft $2 \\times 4096 \\times 16.384$ weights; SwiGLU heeft er $3 \\times 4096$ maal zijn hidden breedte. Die zijn gelijk bij een breedte van $\\frac{8}{3} \\times 4096 \\approx 10.923$, wat Llama naar boven afrondt op een veelvoud van 256: 11.008. De gladde activation function zelf kost maar een paar bewerkingen per hidden getal, weinig naast de matrices.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: gladde activation functions.** **GELU** vermenigvuldigt $x$ met de kans dat een getal uit de normale verdeling met gemiddelde 0 en standaardafwijking 1 kleiner is dan $x$; **SiLU** vermenigvuldigt $x$ met zijn sigmoid:
        """),
        L.formula(r"\text{SiLU}(x) = x \, \sigma(x) = \frac{x}{1 + e^{-x}}"),
        mo.md(r"""
        Allebei liggen ze dicht bij $x$ voor grote $x$ en dicht bij 0 voor heel negatieve $x$, zoals ReLU. Anders dan ReLU hebben ze geen knik bij 0, zakken ze een beetje onder 0, SiLU tot ongeveer $-0{,}28$, en is hun helling bij een negatieve input niet 0: er komt nog een kleine gradient door.

        **Stap 2: een gate.** Een gated MLP, of **gated linear unit** (GLU), heeft twee layers van $d$ naar de hidden breedte in plaats van één. De ene, de gate, gaat door SiLU; de andere, up, niet; hun outputs worden getal voor getal vermenigvuldigd; en een derde layer, down, brengt de uitkomst terug naar $d$:
        """),
        L.formula(r"\begin{gathered} \mathbf{h} = \text{SiLU}(\mathbf{x} [[param:\mathbf{W}_{\text{gate}}]]) \odot \mathbf{x} [[param:\mathbf{W}_{\text{up}}]] \\ \text{SwiGLU}(\mathbf{x}) = \mathbf{h} \, [[param:\mathbf{W}_{\text{down}}]] \end{gathered}"),
        mo.md(r"""
        Elk hidden getal is nu een product: de gate bepaalt, uit de input, hoeveel van de up-waarde doorgaat, en zelfs met welk teken. De layers van Llama hebben geen bias.

        **Stap 3: dezelfde parameters.** Een klassiek MLP met een hidden breedte van $4d$ heeft $2 \cdot d \cdot 4d = 8d^2$ weights. SwiGLU heeft drie matrices van $d$ bij zijn hidden breedte; met een breedte van $\frac{8}{3} d$ is dat ook $3 \cdot d \cdot \frac{8}{3} d = 8d^2$, en de FLOPs van de matrixproducten ook: SwiGLU vervangt het klassieke MLP tegen ongeveer dezelfde kosten.

        **Stap 4: waarom het helpt.** In de paper die gated MLP's in transformers vergeleek, haalden de gated versies, waaronder SwiGLU, een lagere loss dan ReLU en GELU bij hetzelfde aantal parameters en FLOPs. De paper gaf geen verklaring; de gladde helling en het product van twee projecties zijn de gebruikelijke vermoedens.

        Het plaatje: ReLU, gestippeld, GELU, dun, en SiLU, dik, van $-4$ tot $3$.
        """),
        mo.Html(C.activations_picture({"title": "Drie activation functions", "y": "output", "dip": "zakt onder 0", "lang": L.lang,
                                       "alt": "ReLU, gestippeld, is 0 voor elke negatieve x en x vanaf 0, met een knik bij 0. GELU, dun, en SiLU, dik, volgen hem voor grote x, maar zijn glad: ze zakken voor negatieve x een beetje onder 0, GELU tot ongeveer −0,17, SiLU tot ongeveer −0,28, en keren langzaam terug naar 0."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **GELU** | $x$ maal de kans dat een getal uit de normale verdeling met gemiddelde 0 en standaardafwijking 1 kleiner is dan $x$ | `F.gelu(x)` |
        | **SiLU** | $x$ maal zijn sigmoid; ook swish genoemd | `F.silu(x)` |
        | **gated linear unit** (GLU) | twee linear layers op dezelfde input, getal voor getal vermenigvuldigd; de ene, de gate, gaat eerst door een activation function | |
        | **SwiGLU** | een GLU met SiLU op de gate: SiLU van de gate maal up, dan down | |
        """),
        L.watch_out("GELU heeft twee versies: de exacte, en een snellere benadering met tanh die GPT-2 gebruikte. `F.gelu(x)` geeft de exacte, `F.gelu(x, approximate=\"tanh\")` de andere: een model dat met de ene getraind is, geeft met de andere net andere outputs."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een SwiGLU-MLP voor vectoren van $d$ getallen:

    1. Kies de hidden breedte: ongeveer $\frac{8}{3}d$ voor de parameters van een klassiek MLP van $4d$.
    2. Maak drie linear layers zonder bias: gate en up van $d$ naar de hidden breedte, down van de hidden breedte terug naar $d$.
    3. Neem SiLU van de gate, vermenigvuldig hem getal voor getal met up, en stuur het product door down.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    GELU en SiLU in PyTorch zijn nieuw: ReLU, GELU en SiLU bij een paar inputs, en hun hellingen, met autograd.
    """)
    return


@app.cell
def example_activations():
    import torch
    import torch.nn.functional as F

    _x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    for _name, _f in [("ReLU", torch.relu), ("GELU", F.gelu), ("SiLU", F.silu)]:
        _y = _f(_x)
        _y.sum().backward()
        print(_name, "waarden:", [round(_v, 3) for _v in _y.tolist()])
        print(_name, "hellingen:", [round(_v, 3) for _v in _x.grad.tolist()])
        _x.grad = None
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij $x = -1$ geeft ReLU 0 met helling 0; GELU geeft $-0{,}159$ met helling $-0{,}083$, en SiLU $-0{,}269$ met helling $0{,}072$: een weight die ze voedt, krijgt nog een gradient. Bij $x = 2$ ligt GELU dicht bij 2, op 1,954, en SiLU lager, op 1,762: hoe groter $x$, hoe dichterbij. Bij $x = 0$ geeft PyTorch ReLU de helling 0, terwijl GELU en SiLU daar helling 0,5 hebben.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_swiglu():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Je eigen SwiGLU

        Schrijf de class `SwiGLU(d, hidden)` met drie linear layers zonder bias, genaamd `self.gate`, `self.up` en `self.down`, en een `forward` die SiLU van de gate getal voor getal met up vermenigvuldigt en het product door down stuurt. De check zet de weights met de hand: `SwiGLU(2, 3)` met de weights die zijn melding laat zien, moet op het token `[1.0, -1.0]` ongeveer `[[1.4621, 0.5379]]` geven. Hij telt ook de parameters en draait een mini-batch tokens.
        """),
        L.task("Schrijf `SwiGLU`: `self.gate`, `self.up` en `self.down` zonder bias, en een `forward` die de up-waarden vermenigvuldigt met SiLU van de gate."),
    ])
    return


@app.cell(hide_code=True)
def exercise_swiglu():
    box_swiglu = L.code_box("swiglu", lines=14)
    box_swiglu
    return (box_swiglu,)


@app.cell(hide_code=True)
def check_swiglu(box_swiglu):
    result_swiglu = L.check("swiglu", box_swiglu.value)
    result_swiglu
    return (result_swiglu,)


@app.cell(hide_code=True)
def hints_swiglu():
    L.hints(
        "swiglu",
        [
            "Het recept, stap 2 en 3: drie keer `nn.Linear(..., bias=False)`, en in `forward` één regel met `F.silu` en `*`.",
        ],
        why="Alleen de gate gaat door SiLU; `*` vermenigvuldigt getal voor getal; `self.down` brengt de `hidden` getallen terug naar `d`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Oefening 2. Dezelfde parameters
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden-size",
        question="Een GPT met $d = 768$ heeft klassieke MLP's die verbreden tot $4d = 3.072$. Je vervangt ze door SwiGLU met evenveel parameters. Welke hidden breedte kies je?",
        options={
            "right": "2.048, twee derde van de breedte",
            "same": "3.072, dezelfde breedte",
            "half": "1.536, de halve breedte",
        },
        correct="right",
        why={
            "same": "Met drie matrices van 768 bij 3.072 zou SwiGLU anderhalf keer zoveel parameters hebben als het klassieke MLP.",
            "half": "Dan heeft SwiGLU $3 \\times 768 \\times 1{.}536$ weights, een kwart minder dan de $2 \\times 768 \\times 3{.}072$ van het klassieke MLP.",
            "right": "$\\frac{8}{3} \\times 768 = 2{.}048$: $3 \\times 768 \\times 2{.}048 = 2 \\times 768 \\times 3{.}072$.",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_exercise_gate():
    mo.md(r"""
    ### Oefening 3. Wat de gate toevoegt
    """)
    return


@app.cell(hide_code=True)
def explain_gate():
    explain_gate_cell = L.explain(
        "what-the-gate-adds",
        question="In SwiGLU is elk hidden getal SiLU van een gate maal een up-waarde. Wat kan dit MLP dat het ReLU-MLP van je GPT niet kan, en wat kost het bij hetzelfde aantal parameters?",
        model_answer="**Wat het kan:** elk hidden getal is een product van twee projecties van de input, dus de input bepaalt per eenheid hoeveel van een waarde doorgaat, en met welk teken; een ReLU-eenheid kan haar eigen waarde alleen onder 0 afkappen. En SiLU heeft een helling bij negatieve inputs, dus de weights van de gate krijgen daar nog een gradient. **Wat het kost:** een derde weight-matrix, dus bij dezelfde parameters krimpt de hidden breedte tot ongeveer $\\frac{8}{3}d$ in plaats van $4d$; de FLOPs per token blijven gelijk, en er komt één vermenigvuldiging per hidden getal bij.",
        points=[
            "je zegt dat elk hidden getal een product is, dus dat de input bepaalt hoeveel van een waarde doorgaat",
            "je noemt de helling van SiLU bij negatieve inputs, of dat ReLU alles onder 0 afkapt",
            "je noemt de prijs: een derde matrix, en dus een kleinere hidden breedte bij dezelfde parameters",
        ],
    )
    explain_gate_cell
    return (explain_gate_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het MLP van [Het transformer block](/course/open/attn-transformer-block?lang=nl), met SiLU, gebouwd op de sigmoid, op de plek van ReLU, en een gate die getal voor getal vermenigvuldigt. De volgende les, *Rotary position embeddings (RoPE)*, verandert hoe een GPT weet waar een token staat; in *Project: maak van je GPT een modern LLM* krijgt je GPT SwiGLU in elk block.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hidden, explain_gate_cell, result_swiglu):
    L.wrap_up(result_swiglu, choice_hidden, explain_gate_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
