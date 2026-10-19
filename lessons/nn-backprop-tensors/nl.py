# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-backprop-tensors. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-backprop-tensors", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) en [Gradient checking](/course/open/nn-gradient-check?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_autograd():
    quick_autograd = L.choice(
        "quick-autograd",
        question="`w = torch.tensor([1.0, -2.0], requires_grad=True)`, `loss = (w * 3).sum()` en `loss.backward()`. Wat is `w.grad`?",
        options={
            "right": "`tensor([3., 3.])`: elk getal van `w` telt drie keer mee in de som",
            "times": "`tensor([3., -6.])`: elk getal van `w`, keer de 3 waarmee het in de loss vermenigvuldigd wordt",
            "one": "`tensor(-3.)`: één gradient voor de hele som",
        },
        correct="right",
        why={
            "times": "Dat is `w * 3` zelf. De gradient zegt hoe snel de loss verandert als één getal van `w` een beetje stijgt: met 3, welk getal het ook is. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "one": "`w.grad` heeft de shape van `w`: één gradient voor elk van zijn getallen. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "right": "Stijgt één getal van `w` een beetje, dan stijgt de loss drie keer dat beetje: 3 voor elk, in de shape van `w`.",
        },
    )
    quick_autograd
    return (quick_autograd,)


@app.cell(hide_code=True)
def quick_check_error():
    quick_error = L.choice(
        "quick-error",
        question="Je backward pass geeft een weight de gradient 0,8; duwen geeft 0,4. Wat is de relatieve fout?",
        options={
            "small": "1: het verschil, 0,4, gedeeld door de kleinste van de twee",
            "abs": "0,4: het verschil zelf",
            "right": "0,5: het verschil, 0,4, gedeeld door de grootste van de twee",
        },
        correct="right",
        why={
            "small": "De relatieve fout deelt door de grootste van de twee groottes. Zie [Gradient checking](/course/open/nn-gradient-check?lang=nl).",
            "abs": "Dat is het verschil; de relatieve fout deelt het door de grootste grootte, zodat grote en kleine gradients hetzelfde beoordeeld worden. Zie [Gradient checking](/course/open/nn-gradient-check?lang=nl).",
            "right": "0,4 / 0,8 = 0,5: ver boven $10^{-3}$, dus die backward pass heeft een bug. In deze les moet je eigen backward pass ver onder $10^{-7}$ uitkomen.",
        },
    )
    quick_error
    return (quick_error,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je `Value` uit [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl) deed backprop getal voor getal. Voor het MLP dat de cijfers leest, met 64 hidden neuronen en een mini-batch van 64 afbeeldingen, zijn dat miljoenen `Value`s per stap: veel te traag. PyTorch doet dezelfde backward pass een hele layer tegelijk, met een paar matrixproducten. Vandaag schrijf je die backward pass zelf, in NumPy, voor een MLP met twee layers, softmax en cross-entropy, en je controleert hem met PyTorch. Daarna heeft `loss.backward()` geen geheimen meer: hij doet deze producten.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           8, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
    ])
    return


@app.cell(hide_code=True)
def predict_relu():
    guess_relu = L.guess(
        "through-relu",
        question="Op de terugweg komt de gradient voor de output van een ReLU aan: één getal per voorbeeld en per neuron. Wat denk je dat de gradient voor zijn input is?",
        options={
            "same": "Dezelfde getallen: ReLU heeft geen weights, dus hij geeft de gradient ongewijzigd door",
            "right": "Dezelfde getallen waar de input boven 0 lag, en elders 0",
            "times": "De getallen keer wat de ReLU gaf",
        },
        reveal="**Dezelfde getallen waar de input boven 0 lag, en elders 0.** ReLU laat een positieve input ongewijzigd door, dus iets meer input geeft iets meer output: helling 1. Een negatieve input geeft 0, en iets meer ervan geeft nog steeds 0: helling 0. Op de terugweg gaat de gradient dus door waar de input boven 0 lag, en stopt hij elders, zoals de local gradient van `relu` in je `Value` deed, nu voor een hele matrix tegelijk.",
    )
    guess_relu
    return (guess_relu,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het MLP voor de cijfers, voor een mini-batch van $n$ afbeeldingen $\mathbf{X}$, zoals in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl), met softmax en de cross-entropy-loss aan het eind:
        """),
        L.formula(r"\begin{aligned} \mathbf{Z}_1 &= \mathbf{X}[[param:\mathbf{W}_1]]^\top + [[param:\mathbf{b}_1]], \quad \mathbf{H} = \mathrm{ReLU}(\mathbf{Z}_1) \\ \mathbf{Z}_2 &= \mathbf{H}[[param:\mathbf{W}_2]]^\top + [[param:\mathbf{b}_2]], \quad [[pred:\mathbf{P}]] = \mathrm{softmax}(\mathbf{Z}_2), \quad [[loss:\mathcal{L}]] = -\frac{1}{n}\sum_i \log [[pred:P_{i,c_i}]] \end{aligned}"),
        mo.md(r"""
        Hier is $c_i$ de goede klasse van voorbeeld $i$. De backward pass gaat de andere kant op, en elke stap is een regel die je kent. $\mathbf{G}$ is de gradient van de loss voor wat een layer geeft, met dezelfde shape:
        """),
        L.formula(r"\begin{aligned} [[grad:\mathbf{G}_2]] &= ([[pred:\mathbf{P}]] - \mathbf{Y}) / n \\ [[grad:d\mathbf{W}_2]] &= [[grad:\mathbf{G}_2]]^\top \mathbf{H}, \quad [[grad:d\mathbf{b}_2]] = \textstyle\sum_i [[grad:\mathbf{G}_2]], \quad [[grad:d\mathbf{H}]] = [[grad:\mathbf{G}_2]][[param:\mathbf{W}_2]] \\ [[grad:\mathbf{G}_1]] &= [[grad:d\mathbf{H}]] \odot [\mathbf{Z}_1 > 0] \\ [[grad:d\mathbf{W}_1]] &= [[grad:\mathbf{G}_1]]^\top \mathbf{X}, \quad [[grad:d\mathbf{b}_1]] = \textstyle\sum_i [[grad:\mathbf{G}_1]] \end{aligned}"),
        mo.md(r"""
        - De eerste regel is de gradient van softmax met cross-entropy, zoals in [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl), voor een hele mini-batch: $\mathbf{Y}$ bevat de one-hot-rijen van de labels, en de $1/n$ komt van het gemiddelde.
        - $d\mathbf{W}$ en $d\mathbf{H}$ zijn de twee producten uit [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl): bij $\mathbf{X}\mathbf{W}^\top$ krijgen de weights $\mathbf{G}^\top\mathbf{X}$ en de input $\mathbf{G}\mathbf{W}$.
        - Een bias term wordt bij elk voorbeeld opgeteld, dus hij krijgt de gradient van elk, opgeteld over de voorbeelden: één getal per output.
        - ReLU laat de gradient door waar zijn input boven 0 lag, en houdt hem elders tegen; het elementsgewijs product $\odot$ vermenigvuldigt plek voor plek.

        Op de terugweg heeft elke layer dus nodig wat hij op de heenweg kreeg: de backward pass bewaart $\mathbf{X}$, $\mathbf{Z}_1$, $\mathbf{H}$ en $\mathbf{P}$ uit de forward pass. Je lineaire classifier uit [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl) had de eerste regel, $d\mathbf{W}_2$ en $d\mathbf{b}_2$ al, met $\mathbf{X}$ waar nu $\mathbf{H}$ staat.

        | | Je `Value` | Een hele layer |
        |---|---|---|
        | wat één stap bevat | één getal | een matrix, één rij per voorbeeld |
        | de local gradient van een product | het andere getal | de andere matrix, in een product: $\mathbf{G}^\top\mathbf{X}$ en $\mathbf{G}\mathbf{W}$ |
        | ReLU | 1 of 0 | de matrix $[\mathbf{Z}_1 > 0]$ |
        | een getal dat vaker gebruikt wordt | gradients opgeteld | de som over de voorbeelden, $\sum_i$ |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | $\mathbf{G}$ | G | de gradient van de loss voor wat een layer geeft, in zijn shape: één rij per voorbeeld | `G2` |
        | $\mathbf{Y}$ | Y | de one-hot-rijen van de labels: een 1 bij de goede klasse van elk voorbeeld | `np.eye(10)[y]` |
        | **elementsgewijs product**, $\odot$ | elementsgewijs product | twee matrices van dezelfde shape getal voor getal vermenigvuldigen | `dH * (Z1 > 0)` |
        | $[\mathbf{Z}_1 > 0]$ | Z één groter dan nul | 1 waar $\mathbf{Z}_1$ boven 0 ligt, anders 0 | `(Z1 > 0)` |
        | $\sum_i$ | som over i | opgeteld over de voorbeelden, één getal per kolom | `G2.sum(axis=0)` |
        """),
        L.watch_out("Houd de shapes in het oog: elke gradient heeft de shape van waar hij bij hoort. Een product dat verkeerd om staat, zoals `X.T @ G`, draait gewoon, en geeft de gradient van de weights omgedraaid; is `W` vierkant, dan klopt zelfs zijn shape, en houdt niets je tegen. En bewaar `Z1` uit de forward pass: de helling van de ReLU hangt ervan af."),
        mo.md(r"""
        De figuur: de forward pass van links naar rechts, met de shape van elke stap, en de backward pass eronder, van rechts naar links.
        """),
        draw.flow([("X", ["afbeeldingen", "(n, 784)"]), ("H", ["na ReLU", "(n, 64)"]), ("P", ["kansen", "(n, 10)"]), ("Loss", ["één getal"])],
                  ["W1, b1, ReLU", "W2, b2, softmax", "gemiddelde\nvan −log P"],
                  ["G1 = dH ⊙ [Z1 > 0]\ndW1 = G1ᵀ X\ndb1 = Σ G1", "dW2 = G2ᵀ H\ndb2 = Σ G2\ndH = G2 W2", "G2 = (P − Y) / n"],
                  title="De forward pass, en de backward pass eronder"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de gradients van een MLP met twee layers, softmax en cross-entropy uit te rekenen:

    1. De forward pass, en bewaar `Z1`, `H` en `P`.
    2. De gradient voor de logits, `G2 = (P - Y) / n`, en de output layer: `dW2 = G2.T @ H` en `db2 = G2.sum(axis=0)`.
    3. Terug door de output layer en de ReLU: `dH = G2 @ W2`, dan `G1 = dH * (Z1 > 0)`.
    4. De hidden layer: `dW1 = G1.T @ X` en `db1 = G1.sum(axis=0)`.
    5. Controleer met PyTorch: dezelfde forward pass met tensors die `requires_grad=True` hebben, dan `loss.backward()`, en de relatieve fout tussen elke gradient en zijn `.grad`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een klein MLP: 3 voorbeelden met 2 features, 2 hidden neuronen en 3 klassen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af, en zijn regel in stap 5 ook niet: voer de cel uit zoals hij is, maak dan allebei af waar het commentaar dat zegt, en voer hem opnieuw uit. In stap 5 werken `torch.exp` en `torch.log` op een tensor zoals `np.exp` en `np.log` op een array, en `.numpy()` maakt van een tensor weer een array.
    """)
    return


@app.cell
def example_backprop():
    import numpy as np
    import torch

    _X = np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 2.0]])
    _y = np.array([0, 2, 1])
    _W1 = np.array([[1.0, -0.6], [0.5, 0.5]])
    _b1 = np.array([0.0, -0.4])
    _W2 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
    _b2 = np.array([0.1, 0.0, -0.1])
    _n = len(_y)

    # 1. de forward pass, en bewaar Z1, H en P
    _Z1 = _X @ _W1.T + _b1
    _H = np.maximum(_Z1, 0)
    _Z2 = _H @ _W2.T + _b2
    _E = np.exp(_Z2 - _Z2.max(axis=1, keepdims=True))
    _P = _E / _E.sum(axis=1, keepdims=True)
    # 2. de gradient voor de logits, en de output layer
    _G2 = (_P - np.eye(3)[_y]) / _n
    _dW2 = _G2.T @ _H
    _db2 = _G2.sum(axis=0)
    # 3. terug door de output layer en de ReLU
    _dH = _G2 @ _W2
    _G1 = _dH * (_Z1 > 0)
    # 4. de hidden layer: dW1 staat er; voeg db1 toe, zoals db2 gemaakt is, en print hem
    _dW1 = _G1.T @ _X
    print(_dW1)

    # 5. hetzelfde in PyTorch, en de relatieve fout van elke gradient
    _tW1 = torch.tensor(_W1, requires_grad=True)
    _tb1 = torch.tensor(_b1, requires_grad=True)
    _tW2 = torch.tensor(_W2, requires_grad=True)
    _tb2 = torch.tensor(_b2, requires_grad=True)
    _tZ2 = torch.relu(torch.tensor(_X) @ _tW1.T + _tb1) @ _tW2.T + _tb2
    _tE = torch.exp(_tZ2)
    _tP = _tE / _tE.sum(dim=1, keepdim=True)
    _loss = -(torch.tensor(np.eye(3)[_y]) * torch.log(_tP)).sum(dim=1).mean()
    _loss.backward()

    def _rel(a, n):
        return np.max(np.abs(a - n) / np.maximum(np.abs(a), np.abs(n)))

    print(_rel(_dW1, _tW1.grad.numpy()), _rel(_dW2, _tW2.grad.numpy()), _rel(_db2, _tb2.grad.numpy()))
    # en de regel voor db1: print(_rel(_db1, _tb1.grad.numpy()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij het eerste voorbeeld liggen beide hidden inputs boven 0; bij het tweede en derde ligt de eerste onder 0, dus daar is `G1` 0, en alleen het eerste voorbeeld bereikt de eerste rij van `dW1`: 0,0223 en 0,0111. De drie relatieve fouten zijn hooguit een paar keer $10^{-15}$: dezelfde getallen als PyTorch, op afronding na. Met stap 4 af is `db1` 0,0223 en 0,1804, en zijn relatieve fout is net zo klein. `loss.backward()` van PyTorch deed precies deze producten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De gradient van een bias term
    """)
    return


@app.cell(hide_code=True)
def choice_bias_cell():
    choice_bias = L.choice(
        "bias-sum",
        question="Waarom is `db2` de som van `G2` over de voorbeelden, `G2.sum(axis=0)`, en niet het gemiddelde?",
        options={
            "right": "`b2` wordt bij elk voorbeeld opgeteld, dus zijn gradients tellen op; de $1/n$ zit al in `G2`",
            "mean": "Het moet het gemiddelde zijn: de loss is het gemiddelde over de voorbeelden, dus de gradient van `b2` ook",
            "axis": "De som gaat over de klassen, zodat elk voorbeeld één getal krijgt",
        },
        correct="right",
        why={
            "mean": "De loss is een gemiddelde, maar dat gemiddelde zit al in `G2 = (P - Y) / n`. Nog eens het gemiddelde nemen deelt twee keer door $n$.",
            "axis": "`axis=0` telt op over de voorbeelden, de rijen, en laat één getal per klasse over: de shape van `b2`.",
            "right": "`b2` wordt bij elk voorbeeld één keer gebruikt, en bij een getal dat vaker gebruikt wordt tellen de gradients op, zoals in je `Value`. De $1/n$ zit al in `G2`.",
        },
    )
    choice_bias
    return (choice_bias,)


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén lineaire layer

        Schrijf `linear_backward(G, X, W)`: de backward pass van één lineaire layer `X @ W.T + b`. Hij krijgt `G`, de gradient voor wat de layer gaf, met shape (voorbeelden, outputs), de input `X`, met shape (voorbeelden, inputs), en de weights `W`, met shape (outputs, inputs), en geeft `dX`, `dW` en `db` terug, in die volgorde. De check roept hem aan met `G` van shape (3, 2), `X` van shape (3, 4) en `W` van shape (2, 4), en met andere shapes. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/nn/layer_backward.py`.
        """),
        L.task("Laat `linear_backward` de gradients geven voor de input, de weights en de bias terms van een lineaire layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("linear-backward", lines=8)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("linear-backward", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "linear-backward",
        [
            "Stap 2 tot en met 4 van het recept doen dit voor elke layer: één product voor de weights, één voor de input, en een som voor de bias terms.",
            "Laat de shapes elk product bepalen: `dW` moet (outputs, inputs) zijn, `dX` (voorbeelden, inputs), en `db` één getal per output.",
        ],
        why="`dX = G @ W` is (voorbeelden, outputs) keer (outputs, inputs); `dW = G.T @ X` is (outputs, voorbeelden) keer (voorbeelden, inputs); `db = G.sum(axis=0)` telt op over de voorbeelden. Samen: `return dX, dW, db`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De hele MLP

        Schrijf `mlp_gradients(X, y, W1, b1, W2, b2)`: de gradients van de cross-entropy-loss van het MLP voor `W1`, `b1`, `W2` en `b2`, in die volgorde, met stap 1 tot en met 4 van het recept. `X` heeft één rij per voorbeeld, `y` de goede klasse van elk, als getal; het aantal klassen is `W2.shape[0]`. Je mag je `linear_backward` naar het vak kopiëren en gebruiken. De check roept hem aan op 4 voorbeelden met 2 features, 3 hidden neuronen en 2 klassen, en kijkt naar elke gradient; dan op een groter MLP met 3 klassen. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/nn/mlp_backward.py`.
        """),
        L.task("Laat `mlp_gradients` de vier gradients van het MLP geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("mlp-gradients", lines=18)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("mlp-gradients", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "mlp-gradients",
        [
            "Zoals in het uitgewerkte voorbeeld, van boven naar beneden: de forward pass, dan stap 2, 3 en 4. `np.eye(W2.shape[0])[y]` geeft de one-hot-rijen.",
            "Geef de vier gradients terug in de volgorde van `W1, b1, W2, b2` op de `def`-regel: `return dW1, db1, dW2, db2`.",
        ],
        why="De forward pass geeft `Z1`, `H` en `P`. Dan `G2 = (P - np.eye(W2.shape[0])[y]) / len(y)`, `dW2 = G2.T @ H`, `db2 = G2.sum(axis=0)`, `G1 = (G2 @ W2) * (Z1 > 0)`, `dW1 = G1.T @ X` en `db1 = G1.sum(axis=0)`, en `return dW1, db1, dW2, db2`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder haalt je eigen `mlp_gradients` uit je project. Eerst vergelijkt hij de gradients ervan met die van PyTorch, voor een MLP met 64 hidden neuronen op 64 cijfers. Dan traint hij dat MLP ermee op de 6000 trainingscijfers, alleen met NumPy, 5 epochs van mini-batches van 64 met $\eta$ = 0,5, en meet hem op de 1000 testcijfers. Dat duurt een paar seconden. Het werkt zodra oefening 3 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_backprop(result_mlp):
    result_mlp  # run again when exercise 3 saves your function
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 3 af: dan staat je `mlp_gradients` in je project, en gebruikt deze cel hem.*")
    elif _result == "slow":
        _out = mo.md("*Trainen met je `mlp_gradients` in `work/projects/nn/mlp_backward.py` duurt langer dan een minuut. Gebruikt hij hele matrices, zonder loop over de voorbeelden?*")
    elif _result == "broken":
        _out = mo.md("*De `mlp_gradients` in `work/projects/nn/mlp_backward.py` stopt met een fout, of geeft andere gradients dan PyTorch. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: de check bewaart je functie dan opnieuw.*")
    else:
        _errors, _accuracy, _losses = _result
        _out = mo.vstack([
            mo.md(f"""De grootste relatieve fout van je vier gradients ten opzichte van PyTorch ligt onder $10^{{{C.power(max(_errors))}}}$: ver onder $10^{{-7}}$, dus je backward pass rekent uit wat `loss.backward()` uitrekent. Ermee getraind leest je MLP <span class="ea-num">{L.number(round(_accuracy * 1000), 0)}</span> van de 1000 testcijfers goed: een accuracy van <span class="ea-num">{L.number(_accuracy, 3)}</span>, waar je lineaire classifier er 890 las; en zonder één regel PyTorch."""),
            mo.Html(C.losses_picture(_losses, {"title": "De loss van je training, per epoch", "epoch": "epoch", "loss": "loss op de trainingscijfers",
                                               "alt": f"Een lijn door {len(_losses)} punten, één per epoch: de loss op alle trainingscijfers daalt van {L.number(_losses[0], 2)} na de eerste epoch naar {L.number(_losses[-1], 2)} na de laatste."})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De twee producten uit [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl), de gradient uit [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl) en de ReLU van je `Value` vormen samen de backward pass van een hele MLP; [Gradient checking](/course/open/nn-gradient-check?lang=nl) en PyTorch bevestigen hem. Je project `nn` heeft nu ook `layer_backward.py` en `mlp_backward.py`. In de volgende les, *Vanishing en exploding gradients*, volg je deze gradients terug door veel layers, en zie je ze krimpen of groeien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bias, result_layer, result_mlp):
    L.wrap_up(choice_bias, result_layer, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
