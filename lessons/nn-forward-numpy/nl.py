# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-forward-numpy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow vectorization, transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-forward-numpy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) en [Broadcasting](/course/open/np-broadcasting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="In [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) heeft `W` één rij weights per hidden neuron. Een hidden layer van 4 neuronen krijgt 3 inputs. Welke shape heeft zijn `W`?",
        options={
            "inputs": "(3, 4)",
            "right": "(4, 3)",
            "vector": "(4,)",
        },
        correct="right",
        why={
            "inputs": "Dat zou één rij per input zijn. Van welk neuron staan de weights in `W[i]`? Zie het recept in [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "vector": "Dat is één getal per neuron, zoals bij de bias terms `b`. Hoeveel weights heeft elk neuron? Zie het recept in [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "right": "4 rijen, één per neuron, en 3 getallen in elke rij, één weight per input.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_broadcast():
    quick_broadcast = L.choice(
        "quick-broadcast",
        question="`Z` heeft shape (10, 4) en `b` heeft shape (4,). Wat geeft `Z + b`?",
        options={
            "error": "NumPy stopt: 10 en 4 verschillen",
            "first": "Alleen de eerste rij van `Z` krijgt `b` erbij",
            "right": "Elke rij van `Z` krijgt `b` erbij, getal voor getal",
        },
        correct="right",
        why={
            "error": "Rechts uitgelijnd zijn 4 en 4 gelijk, en de ontbrekende lengte telt als 1. Zie de regel in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "first": "NumPy rekt `b` uit over alle rijen. Zie de figuur in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "right": "Van rechts zijn 4 en 4 gelijk, en de ontbrekende lengte van `b` telt als 1: `b` wordt uitgerekt over de 10 rijen.",
        },
    )
    quick_broadcast
    return (quick_broadcast,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) rekende je een netwerk neuron voor neuron uit, voor één voorbeeld tegelijk, met een loop. Een echt netwerk heeft honderden neuronen in een layer en leert van duizenden voorbeelden, en in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) was een loop in Python vele malen trager dan NumPy. Zonder de loop reken je met hele matrices, en dan moet je op één ding letten: hun shapes. Een matrix die verkeerd om staat is de meest gemaakte fout in de code van een netwerk.

    **Een voorbeeld uit het dagelijks leven.** Als je bedragen in euro's en in dollars optelt, schrijf je de valuta naast elk bedrag. Dan zie je meteen dat je op het punt staat euro's bij dollars op te tellen, nog voor het totaal fout is.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="Een hidden layer van 4 neuronen krijgt 100 voorbeelden tegelijk, elk met 3 inputs. In [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) gaf één `np.dot` de `z` van één neuron voor één voorbeeld. Hoeveel matrixvermenigvuldigingen met `@` kost de hele layer volgens jou, voor alle 100 voorbeelden?",
        options={
            "each": "400: één voor elk neuron en elk voorbeeld",
            "examples": "100: één voor elk voorbeeld",
            "right": "1",
        },
        reveal="**Eén.** Met de voorbeelden als rijen van `X`, shape (100, 3), en de weights als rijen van `W`, shape (4, 3), heeft `X @ W.T` shape (100, 4). Het getal in rij $k$ en kolom $j$ is het inproduct van voorbeeld $k$ met de weights van neuron $j$: de $z$ van dat neuron voor dat voorbeeld. Alle 400 komen uit één vermenigvuldiging.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **forward pass** stuurt de inputs door een netwerk, van de eerste layer tot de laatste, om zijn outputs te krijgen. Elke layer doet dezelfde drie dingen: vermenigvuldigen met zijn weights, zijn bias terms erbij optellen, en de uitkomst door zijn activation function halen.

        **Eén layer, één voorbeeld.** De weights van een layer vormen een **weight-matrix** $\mathbf{W}$, met één rij per neuron, zoals `W` in [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl): een layer van 4 neuronen met 3 inputs heeft shape (4, 3). `W @ x` neemt het inproduct van elke rij met `x`, dus geeft het de $z$ van elk neuron tegelijk, shape (4,). De bias terms van de layer vormen een vector $\mathbf{b}$ met shape (4,), één per neuron.
        """),
        L.formula(r"[[hidden:\mathbf{h}]] = \mathrm{ReLU}([[param:\mathbf{W}]]\mathbf{x} + [[param:\mathbf{b}]])"),
        mo.md(r"""
        **Eén layer, alle voorbeelden.** Zet de voorbeelden in $\mathbf{X}$, één rij per voorbeeld, zoals in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl): shape $(n, 3)$. Elke $z$ is nu het inproduct van een rij van $\mathbf{X}$ met een rij van $\mathbf{W}$. Matrixvermenigvuldiging neemt de rijen van de eerste matrix met de kolommen van de tweede, dus klap $\mathbf{W}$ eerst om, zoals in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl): $\mathbf{W}^\top$ heeft shape (3, 4), één kolom per neuron. Tel daarna $\mathbf{b}$ erbij op: broadcasting telt hem bij elke rij op. De output $\mathbf{H}$ heeft één rij per voorbeeld en één kolom per neuron.
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{ReLU}(\mathbf{X}[[param:\mathbf{W}]]_{[[param:1]]}^\top + [[param:\mathbf{b}_1]])"),
        L.formula(r"[[pred:\hat{\mathbf{Y}}]] = [[hidden:\mathbf{H}]][[param:\mathbf{W}]]_{[[param:2]]}^\top + [[param:\mathbf{b}_2]]"),
        mo.md(r"""
        Het kleine getal zegt welke layer: $\mathbf{W}_1$ en $\mathbf{b}_1$ horen bij de hidden layer, $\mathbf{W}_2$ en $\mathbf{b}_2$ bij de output layer. De output layer doet hetzelfde met $\mathbf{H}$ als zijn inputs. Met één output-neuron heeft $\mathbf{W}_2$ één rij, en $\hat{\mathbf{Y}}$ één kolom.

        Schrijf de shape van elke stap ernaast, als commentaar: `# (100, 3) @ (3, 4) + (4,) -> (100, 4)`. Voordat je de regel uitvoert, zie je of de binnenste getallen gelijk zijn en wat eruit komt, net als met de valuta naast de bedragen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **forward pass** | | de inputs door een netwerk gestuurd, layer voor layer, van de eerste tot de laatste, om zijn outputs te krijgen | |
        | $\mathbf{W}$ | hoofdletter W | de **weight-matrix** van een layer: één rij weights per neuron, shape (neuronen, inputs) | `W` |
        | $\mathbf{b}$ | b | de bias terms van een layer, één per neuron, shape (neuronen,) | `b` |
        | $\mathbf{W}_1$, $\mathbf{W}_2$ | W één, W twee | de weight-matrices van layer 1 en layer 2 | `W1`, `W2` |
        | $\mathbf{H}$ | hoofdletter H | de outputs van een hidden layer voor alle voorbeelden: één rij per voorbeeld, één kolom per neuron | `H` |
        | $\hat{\mathbf{Y}}$ | hoofdletter Y-dakje | de outputs van het netwerk voor alle voorbeelden: één rij per voorbeeld, één kolom per output-neuron | `y_pred` |
        | **shape-notatie** | | de shape van elke stap ernaast geschreven als commentaar, zodat je ze kunt volgen van layer tot layer | `# (4, 2) @ (2, 3) -> (4, 3)` |
        | `np.maximum` | | het grootste van twee getallen, getal voor getal: `np.maximum(0, Z)` is ReLU op elk getal van `Z` tegelijk. `max(0, z)` van Python werkt maar op één getal | `np.maximum(0, Z)` |
        """),
        L.watch_out("Een vierkante weight-matrix, met evenveel neuronen als inputs, past op allebei de manieren. Vergeet je de `.T`, dan loopt `X @ W` zonder fout en geeft het verkeerde getallen. Test je code dus met layers van verschillende grootte, zoals 3 inputs en 5 neuronen: dan laat een matrix die verkeerd om staat NumPy meteen stoppen."),
        mo.md(r"""
        De hidden layer van het uitgewerkte voorbeeld hieronder: 4 voorbeelden met 2 inputs, en 3 neuronen. De dikke vakjes laten zien waar één getal van de uitkomst vandaan komt.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_layer():
    mo.vstack([
        draw.array_grids(C.picture(["de tweede rij: voorbeeld 2", "de derde rij: neuron 3", "rij 2, kolom 3"]),
                         title="Eén layer, alle voorbeelden tegelijk"),
        mo.md(r"""
        De tweede rij van `X`, het voorbeeld (3, 1), met de derde rij van `W1`, de weights (1, −1) van neuron 3, plus zijn bias term 0: 3 × 1 + 1 × (−1) + 0 = 2. Dat is het getal in rij 2 en kolom 3 van de uitkomst. Elk ander getal ontstaat op dezelfde manier, uit de rij van zijn voorbeeld en de rij van zijn neuron.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo reken je een forward pass uit voor alle voorbeelden tegelijk:

    1. Zet de voorbeelden in `X`, één rij per voorbeeld: shape (voorbeelden, inputs).
    2. Schrijf voor elke layer de shapes erbij en dan de regel: (voorbeelden, inputs) @ (inputs, neuronen) + (neuronen,) geeft (voorbeelden, neuronen), als `X @ W.T + b`.
    3. Een hidden layer haalt de uitkomst door zijn activation function: `np.maximum(0, ...)` voor ReLU. De output layer heeft er hier geen.
    4. De output van een layer is de `X` van de volgende: zijn aantal kolommen moet gelijk zijn aan het aantal kolommen van de volgende `W`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het netwerk van de figuur: 4 voorbeelden met 2 inputs, een hidden layer van 3 neuronen met ReLU, en een output layer van 1 neuron zonder activation function. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_forward():
    import numpy as np

    # 1. vier voorbeelden met elk twee inputs
    _X = np.array([[1, 2], [3, 1], [0, 2], [2, 2]])     # (4, 2)
    _W1 = np.array([[1, 1], [0, 1], [1, -1]])           # (3, 2): 3 neuronen, 2 inputs
    _b1 = np.array([-2, 0, 0])                          # (3,)
    _W2 = np.array([[1, -1, 2]])                        # (1, 3): 1 neuron, 3 inputs
    _b2 = np.array([0.5])                               # (1,)
    # 2. en 3. de hidden layer: (4, 2) @ (2, 3) + (3,) -> (4, 3), dan ReLU
    _H = np.maximum(0, _X @ _W1.T + _b1)
    print(_H.shape)
    print(_H)
    # 4. de output layer, zonder activation function: (4, 3) @ (3, 1) + (1,) -> (4, 1)
    _y_pred = _H @ _W2.T + _b2
    print(_y_pred.shape)
    print(_y_pred)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De hidden layer geeft één rij per voorbeeld en één kolom per neuron. Bij het eerste voorbeeld heeft neuron 3 $z = 1 - 2 + 0 = -1$, en ReLU maakt daar 0 van. De output heeft shape (4, 1): één rij per voorbeeld, en één kolom voor het ene output-neuron. Bij het eerste voorbeeld is dat $1 \cdot 1 + 2 \cdot (-1) + 0 \cdot 2 + 0{,}5 = -0{,}5$. De output layer heeft geen activation function, dus een negatieve output blijft negatief.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shapes():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De shapes door een netwerk

        Een netwerk krijgt 32 voorbeelden tegelijk, met elk 20 inputs. Zijn eerste hidden layer heeft 16 neuronen, zijn tweede 8, en zijn output layer 3. Vervang elke `...` door de shape van die array, als tuple zoals `(32, 20)`. `W1`, `W2` en `W3` zijn de weight-matrices van de drie layers, `H1` en `H2` de outputs van de twee hidden layers, en `y_pred` de output van het netwerk. De laatste regel is een lijst met de shapes, zodat de check hem ziet.
        """),
        L.task("Geef de shape van elke weight-matrix en elke output in het netwerk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shapes():
    box_shapes = L.code_box("shapes", lines=9)
    box_shapes
    return (box_shapes,)


@app.cell(hide_code=True)
def check_shapes(box_shapes):
    result_shapes = L.check("shapes", box_shapes.value)
    result_shapes
    return (result_shapes,)


@app.cell(hide_code=True)
def hints_shapes():
    L.hints(
        "shapes",
        [
            "Ga layer voor layer, met stap 2 van het recept: een `W` is (neuronen, inputs), en zijn output is (voorbeelden, neuronen).",
            "De inputs van de tweede hidden layer zijn de 16 outputs van de eerste. De voorbeelden blijven de hele weg de rijen.",
            "`shape_W1 = (16, 20)` en `shape_H1 = (32, ___)`; dan `shape_W2 = (8, ___)`.",
        ],
        why="Elke `W` heeft één rij per neuron van zijn layer en één kolom per input, en de inputs van een layer zijn de neuronen van de layer ervoor: (16, 20), (8, 16) en (3, 8). Elke output houdt de 32 voorbeelden als rijen: (32, 16), (32, 8) en (32, 3).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_forward():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een forward pass voor alle voorbeelden

        Maak `forward(X, W1, b1, W2, b2)` af: een netwerk met een hidden layer met ReLU en een output layer zonder activation function, voor alle voorbeelden tegelijk. `X` heeft één rij per voorbeeld: shape (voorbeelden, inputs). `W1` heeft één rij weights per hidden neuron: shape (neuronen, inputs); `b1` heeft één bias term per hidden neuron. `W2` en `b2` zijn hetzelfde voor de output layer. De eerste regels maken arrays van wat de check meegeeft. Vervang de twee `...`: de hidden layer in `H`, en na `return` de output layer. De check roept `forward([[1, 2], [2, 0]], [[1, -1], [0, 1], [2, 1]], [0, -1, -3], [[2, 1, -3]], [0.5])` aan: dat moet `[[-1.5], [1.5]]` geven. Hij probeert ook andere netwerken, met andere aantallen voorbeelden, inputs en neuronen, en meer dan één output-neuron.
        """),
        L.task("Laat `forward` de hidden layer en de output layer uitrekenen voor alle voorbeelden tegelijk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=11)
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
            "Stap 2 en 3 van het recept, één keer voor de hidden layer en één keer voor de output layer, zoals in het uitgewerkte voorbeeld.",
            "`W1` heeft één rij per neuron, dus hij heeft zijn `.T` nodig vóór `@`. ReLU op een hele array is `np.maximum`, niet `max`.",
            "`H = np.maximum(0, X @ ___ + b1)` en `return H @ ___ + ___`.",
        ],
        why="Bij het voorbeeld `[1, 2]` heeft de hidden layer $z = (1 - 2 + 0,\\ 0 + 2 - 1,\\ 2 + 2 - 3) = (-1, 1, 1)$, en ReLU maakt daar $(0, 1, 1)$ van. De output is $2 \\cdot 0 + 1 \\cdot 1 - 3 \\cdot 1 + 0{,}5 = -1{,}5$. Bij `[2, 0]` is $\\mathbf{h} = (2, 0, 1)$ en de output $4 + 0 - 3 + 0{,}5 = 1{,}5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sizes():
    mo.md(r"""
    ### Oefening 3. Een test die de fout vindt
    """)
    return


@app.cell(hide_code=True)
def choice_sizes_cell():
    choice_sizes = L.choice(
        "test-sizes",
        question="Je `forward` heeft een fout: er staat `X @ W1` in plaats van `X @ W1.T`. Bij welk netwerk stopt NumPy met een foutmelding, zodat je de fout meteen ziet?",
        options={
            "square": "2 inputs en een hidden layer van 2 neuronen",
            "right": "2 inputs en een hidden layer van 3 neuronen",
            "always": "Elk netwerk: NumPy stopt altijd als er een `.T` ontbreekt",
        },
        correct="right",
        why={
            "square": "Dan heeft `W1` shape (2, 2), en (n, 2) @ (2, 2) past. Wat doet NumPy dan? Zie het let-op in *Het idee*.",
            "always": "NumPy kijkt alleen of de binnenste getallen gelijk zijn. Wanneer zijn ze gelijk, ook zonder de `.T`? Zie het let-op in *Het idee*.",
            "right": "`W1` heeft shape (3, 2), en in (n, 2) @ (3, 2) verschillen de binnenste getallen 2 en 3: NumPy stopt. Met evenveel neuronen als inputs zou dezelfde fout verkeerde getallen geven, zonder foutmelding.",
        },
    )
    choice_sizes
    return (choice_sizes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) rekende je een netwerk neuron voor neuron uit. Nu is een hele layer één regel, `X @ W.T + b`, voor alle voorbeelden tegelijk: matrixvermenigvuldiging, de getransponeerde en broadcasting samen. De shape naast elke regel vertelt je vóór je hem uitvoert of hij past. In de volgende les, *Te veel weights om één voor één op te schuiven*, zie je waarom het veel te traag is om de helling van elke weight te meten door hem een klein stukje op te schuiven, bij een netwerk als dit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shapes, result_forward, choice_sizes):
    L.wrap_up(result_shapes, result_forward, choice_sizes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
