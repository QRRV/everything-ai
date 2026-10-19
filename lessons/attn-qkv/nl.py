# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-qkv. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-qkv", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Attention weights uit inproducten](/course/open/attn-scores?lang=nl) en [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights_shape():
    quick_weights_shape = L.choice(
        "quick-weights-shape",
        question="Een zin heeft 6 woorden met embeddings van 10 getallen: `X` heeft shape (6, 10). Welke shape hebben de attention weights, de softmax van elke rij van `X @ X.T`?",
        options={
            "right": "(6, 6): een weight voor elk paar woorden",
            "per-number": "(6, 10): een weight voor elk getal van elk woord",
            "numbers": "(10, 10): een weight voor elk paar getallen",
        },
        correct="right",
        why={
            "per-number": "`X @ X.T` is (6, 10) keer (10, 6): de 10'en passen, en (6, 6) blijft over. Zie [Attention weights uit inproducten](/course/open/attn-scores?lang=nl).",
            "numbers": "Dat is `X.T @ X`, dat de getallen van de embeddings vergelijkt, niet de woorden. Zie [Attention weights uit inproducten](/course/open/attn-scores?lang=nl).",
            "right": "Een rij en een kolom voor elk woord. Vandaag komen de scores uit twee nieuwe matrices, maar hun shape blijft gelijk.",
        },
    )
    quick_weights_shape
    return (quick_weights_shape,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`Q` en `K` hebben allebei shape (6, 4). Welk getal staat in rij `i` en kolom `j` van `Q @ K.T`?",
        options={
            "same-place": "`Q[i, j]` keer `K[i, j]`",
            "columns": "Het inproduct van kolom `i` van `Q` en kolom `j` van `K`",
            "right": "Het inproduct van rij `i` van `Q` en rij `j` van `K`",
        },
        correct="right",
        why={
            "same-place": "Dat zou `Q * K` zijn, getal voor getal, met shape (6, 4). Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "columns": "Dat is `Q.T @ K`, met shape (4, 4). De `.T` op `K` maakt van zijn rijen kolommen. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "right": "Kolom `j` van `K.T` is rij `j` van `K`. Vandaag is rij `i` van `Q` waar woord `i` naar zoekt, en rij `j` van `K` wat woord `j` aanbiedt.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Attention weights uit inproducten](/course/open/attn-scores?lang=nl) deed elk woord drie klussen met één en dezelfde embedding: het vergeleek zich met de andere, de andere woorden vergeleken zich ermee, en het gaf zichzelf door. Dan past een woord meestal het best bij woorden die erop lijken. In "a red car" neemt "red" het meest van "red", terwijl het juist het ding nodig heeft dat het beschrijft: "car". Een woord moet naar iets anders kunnen zoeken dan wat het zelf is. Echte attention geeft elk woord drie geleerde vectoren, één voor elke klus.
    """)
    return


@app.cell(hide_code=True)
def predict_lookup():
    guess_lookup = L.guess(
        "lookup",
        question="Een Python-dict zoekt een key op: `prices[\"apple\"]` geeft de value die onder de key \"apple\" staat. Attention werkt ook met queries, keys en values. Wat denk je dat attention geeft voor één query?",
        options={
            "exact": "De value van de key die gelijk is aan de query, zoals een dict",
            "right": "Een gemiddelde van alle values, gewogen naar hoe goed elke key past",
            "key": "De key die het best bij de query past, in plaats van een value",
        },
        reveal="**Een gemiddelde van alle values, gewogen naar hoe goed elke key past.** Een dict vindt de ene key die precies gelijk is en geeft zijn value. Attention zoekt zacht op: elke key past een beetje bij de query, via een inproduct, de softmax maakt daar weights van, en de values worden met die weights gemiddeld. De namen komen van deze vergelijking met opzoeken.",
    )
    guess_lookup
    return (guess_lookup,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Self-attention** geeft elk woord drie vectoren, elk gemaakt uit zijn embedding met een eigen geleerde matrix: een **query** $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q$, waar het woord naar zoekt; een **key** $\mathbf{k}_i = \mathbf{x}_i \mathbf{W}_K$, waarmee het gevonden kan worden; en een **value** $\mathbf{v}_i = \mathbf{x}_i \mathbf{W}_V$, wat het doorgeeft. De score van woord $i$ voor woord $j$ is $\mathbf{q}_i \cdot \mathbf{k}_j$. Voor alle woorden tegelijk, met één embedding per rij van $\mathbf{X}$:
        """),
        L.formula(r"\begin{aligned} [[hidden:Q]] &= [[hidden:\mathbf{X}]] [[param:\mathbf{W}_Q]], \; [[hidden:K]] = [[hidden:\mathbf{X}]] [[param:\mathbf{W}_K]], \; [[hidden:V]] = [[hidden:\mathbf{X}]] [[param:\mathbf{W}_V]] \\ [[hidden:\mathbf{H}]] &= \mathrm{softmax}([[hidden:Q]] [[hidden:K]]^\top) \, [[hidden:V]] \end{aligned}"),
        mo.md(r"""
        De softmax gaat langs elke rij, zoals in [Attention weights uit inproducten](/course/open/attn-scores?lang=nl). Queries en keys moeten even lang zijn, $d_k$, om hun inproduct te kunnen nemen; de values mogen een andere lengte hebben. Omdat een query en een key uit verschillende matrices komen, hoeft een woord niet meer het best bij zichzelf te passen, en hoeft $QK^\top$ niet meer symmetrisch te zijn: woord $i$ kan naar woord $j$ zoeken zonder dat woord $j$ naar woord $i$ zoekt. Het heet *self*-attention omdat de queries, keys en values allemaal uit dezelfde woorden komen. De drie matrices worden geleerd, zoals de weights van elke layer; de attention weights worden voor elke zin opnieuw uitgerekend. In PyTorch is elke matrix een `nn.Linear(d, d_k, bias=False)`: `bias=False` laat de bias terms weg, zodat de layer alleen zijn matrix is, en `self.query(x)` geeft de queries van alle rijen van `x` tegelijk. Zijn `.weight` bevat $\mathbf{W}_Q^\top$, met shape $(d_k, d)$, zoals in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl). `torch.softmax(S, dim=1)` neemt de softmax van elke rij, en trekt zelf de grootste score van elke rij af.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **query** | | waar een woord naar zoekt: zijn embedding keer $\mathbf{W}_Q$ | `Q = X @ Wq` |
        | **key** | | waarmee een woord gevonden kan worden: zijn embedding keer $\mathbf{W}_K$ | `K = X @ Wk` |
        | **value** | | wat een woord doorgeeft: zijn embedding keer $\mathbf{W}_V$ | `V = X @ Wv` |
        | **self-attention** | | attention waarin de queries, keys en values allemaal uit dezelfde woorden komen | `A @ V` |
        | $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ | hoofdletter W Q, K, V | de drie geleerde matrices, met shape $(d, d_k)$ | `Wq`, `Wk`, `Wv` |
        | $d_k$ | d k | de lengte van de queries en keys | `d_k` |
        """),
        L.watch_out("Een key en een value zijn hier niet die van een Python-dict. Een dict vindt de ene key die gelijk is aan wat je vraagt, en geeft zijn value; attention geeft een gewogen gemiddelde van alle values, en elke key telt een beetje mee. En een value is geen los getal maar een vector: wat een woord doorgeeft aan de woorden die ernaar kijken."),
        mo.md(r"""
        De figuur: de attention weights van "a red car", met verzonnen embeddings van 2 getallen: hoeveel een woord beschrijft en hoeveel het een ding is. Bovenaan, met de scores $\mathbf{X}\mathbf{X}^\top$ van de vorige les: "red" en "car" nemen elk 0,95 van zichzelf. Daaronder, met een query-matrix die de twee getallen omwisselt, verzonnen door de course, en de embeddings zelf als keys: een woord dat beschrijft, zoekt nu een ding, en "red" neemt 0,95 van "car". Een echt model leert zijn matrices.
        """),
        mo.vstack(C.weights_pictures({"plain": "Weights met X Xᵀ", "learned": "Weights met Q Kᵀ", "rows": "woord", "cols": "neemt van", "decimal": ",",
                                      "plain_alt": "Attention weights van a red car met de scores X X getransponeerd: a neemt 0,25, 0,38 en 0,38; red neemt 0,03, 0,95 en 0,02; car neemt 0,03, 0,02 en 0,95.",
                                      "learned_alt": "Attention weights van a red car met queries en keys: a neemt 0,25, 0,38 en 0,38; red neemt 0,03, 0,02 en 0,95; car neemt 0,03, 0,95 en 0,02."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Self-attention voor alle woorden tegelijk, met één embedding per rij van `X`:

    1. De queries, keys en values: `X` keer elk van de drie matrices.
    2. De scores: elke query met elke key, `Q @ K.T`.
    3. De weights: de softmax van elke rij, zonder overflow.
    4. De nieuwe vectoren: de weights keer de values, `A @ V`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "a red car" met de verzonnen embeddings en matrices van de figuur; de values zijn hier de embeddings zelf, om het kort te houden. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_qkv():
    import numpy as np

    _X = np.array([[0.3, 0.3], [2.0, 0.0], [0.0, 2.0]])
    _Wq = np.array([[0.0, 1.0], [1.0, 0.0]])
    _Wk = np.array([[1.0, 0.0], [0.0, 1.0]])
    _Wv = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 1. queries, keys en values
    _Q, _K, _V = _X @ _Wq, _X @ _Wk, _X @ _Wv
    print(_Q)
    # 2. elke query met elke key
    _S = _X @ _X.T  # maak af: _Q @ _K.T
    # 3. de weights: de softmax van elke rij
    _S = _S - _S.max(axis=1, keepdims=True)
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    # 4. de nieuwe vectoren
    print(np.round(_A @ _V, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De queries wisselen de twee getallen om: "red" vraagt om $(0;\ 2)$, een ding. Zoals de cel nu is, zijn de scores nog `_X @ _X.T`, en neemt "red" 0,951 van zichzelf: zijn nieuwe vector is $(1{,}911;\ 0{,}044)$, het beschrijft dus nog steeds. Afgemaakt, met `_Q @ _K.T`, wordt de rij van "red" `[0.032 0.017 0.951]`: het neemt 0,951 van "car", en zijn nieuwe vector is $(0{,}044;\ 1{,}911)$, dicht bij de value van "car". "car" doet hetzelfde andersom. "a" blijft zoals het was: zijn query, $(0{,}3;\ 0{,}3)$, past een beetje bij allebei.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_value_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De lengte van de nieuwe vectoren
    """)
    return


@app.cell(hide_code=True)
def choice_value_shape_cell():
    choice_value_shape = L.choice(
        "value-shape",
        question="`X` heeft shape (5, 16). $\\mathbf{W}_Q$ en $\\mathbf{W}_K$ hebben shape (16, 8), en $\\mathbf{W}_V$ heeft shape (16, 4). Welke shape hebben de nieuwe vectoren, `A @ V`?",
        options={
            "query": "(5, 8): zo lang als een query",
            "embedding": "(5, 16): zo lang als een embedding",
            "right": "(5, 4): zo lang als een value",
        },
        correct="right",
        why={
            "query": "De queries en keys maken alleen de weights, met shape (5, 5); wat gemiddeld wordt, zijn de values.",
            "embedding": "De embeddings worden niet doorgegeven zoals ze zijn: elk woord geeft zijn value door, van 4 getallen.",
            "right": "`A` is (5, 5) en `V` is (5, 4): elk woord krijgt een gemiddelde van de values, dus 4 getallen. De lengte van een query en key, 8, is verdwenen zodra de scores gemaakt zijn.",
        },
    )
    choice_value_shape
    return (choice_value_shape,)


@app.cell(hide_code=True)
def text_exercise_qkv():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Self-attention in NumPy

        Maak `qkv_attention(X, Wq, Wk, Wv)` af: `X` bevat één embedding per rij, en `Wq`, `Wk` en `Wv` zijn de drie matrices; de functie geeft de nieuwe vectoren, één per rij. Stap 3 staat er al. Vervang de vijf `...` volgens stap 1, 2 en 4 van het recept. De check roept `qkv_attention` aan voor "a car is red", met de embeddings van de figuur en voor "is" de embedding van "a", dezelfde query- en key-matrix, en een value-matrix die het tweede getal halveert, `[[1, 0], [0, 0.5]]`, wat `[[0.7224, 0.3612], [1.8617, 0.0261], [0.7224, 0.3612], [0.0522, 0.9308]]` moet geven, en probeert andere zinnen en matrices, ook met values van een andere lengte.
        """),
        L.task("Laat `qkv_attention` de nieuwe vectoren geven uit queries, keys en values."),
    ])
    return


@app.cell(hide_code=True)
def exercise_qkv():
    box_qkv = L.code_box("qkv-attention", lines=16)
    box_qkv
    return (box_qkv,)


@app.cell(hide_code=True)
def check_qkv(box_qkv):
    result_qkv = L.check("qkv-attention", box_qkv.value)
    result_qkv
    return (result_qkv,)


@app.cell(hide_code=True)
def hints_qkv():
    L.hints(
        "qkv-attention",
        [
            "Stap 1, 2 en 4 van het afgemaakte uitgewerkte voorbeeld, met de namen van de functie.",
            "Elke rij van `X` is een woord, dus de matrix komt na `X`. De scores koppelen elke rij van `Q` aan elke rij van `K`, zoals in de tweede vraag van de instapcheck.",
        ],
        why="`X @ Wq` geeft één query per woord, en `Q @ K.T` zet query $i$ tegen key $j$ op rij $i$, kolom $j$. Er wordt gemiddeld over de values, `A @ V`, waarvan de lengte kan verschillen van die van de queries.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_module():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Self-attention in PyTorch

        Maak de class `SelfAttention(d, d_k)` af: in `__init__` drie `nn.Linear`-layers van `d` naar `d_k` getallen, zonder bias terms, met de namen `self.query`, `self.key` en `self.value`, in die volgorde gemaakt; in `forward` de nieuwe vectoren van een tensor `x` met één embedding per rij, volgens het recept, met `torch.softmax(..., dim=1)` voor stap 3. De check maakt de module na `torch.manual_seed(0)`, met `d` 4 en `d_k` 3, geeft hem 5 willekeurige embeddings, en vergelijkt de shape en de nieuwe vector van het eerste woord; daarna probeert hij andere maten, en embeddings van nullen.
        """),
        L.task("Laat `SelfAttention` de nieuwe vectoren geven met drie geleerde matrices."),
    ])
    return


@app.cell(hide_code=True)
def exercise_module():
    box_module = L.code_box("self-attention-module", lines=16)
    box_module
    return (box_module,)


@app.cell(hide_code=True)
def check_module(box_module):
    result_module = L.check("self-attention-module", box_module.value)
    result_module
    return (result_module,)


@app.cell(hide_code=True)
def hints_module():
    L.hints(
        "self-attention-module",
        [
            "In `__init__` de drie layers als attributen van `self`, elk `nn.Linear(d, d_k, bias=False)`. In `forward` pas je elk ervan toe op `x`.",
            "Dan de weights, de softmax van de scores langs elke rij, en de weights keer de values: stap 2 tot en met 4 van oefening 2, met `torch.softmax` in plaats van de twee regels van stap 3.",
        ],
        why="Een `nn.Linear` zonder bias terms is een geleerde matrix: `self.query(x)` is $\\mathbf{X}\\mathbf{W}_Q$ voor alle rijen tegelijk. Trainen verandert de drie matrices, en daarmee waar elk woord naar zoekt, wat het aanbiedt en wat het doorgeeft.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De attention uit [Attention weights uit inproducten](/course/open/attn-scores?lang=nl), nu met drie geleerde matrices, geschreven als module met de `nn.Linear` uit [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl). Met queries en keys kan een woord naar iets anders zoeken dan zichzelf. Het inproduct van een query en een key groeit met hun lengte: met een $d_k$ van 64, zoals in elke attention van GPT-2, worden de scores groot en de softmax heel scherp. In de volgende les, *De scores schalen*, deel je ze door $\sqrt{d_k}$, zodat de softmax en zijn gradients blijven werken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_value_shape, result_qkv, result_module):
    L.wrap_up(choice_value_shape, result_qkv, result_module)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
