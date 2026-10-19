# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-matrix-calculus. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow derivative, derivatives, transpose, regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-calculus", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Partiële afgeleiden en de gradient, formeel](/course/open/math-partial-derivatives?lang=nl) en [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_paths():
    quick_paths = L.choice(
        "quick-paths",
        question="$z = u + v$, met $u = 2t$ en $v = t^2$. Wat is $\\frac{dz}{dt}$ bij $t = 3$?",
        options={
            "right": "8: de twee paden opgeteld, 2 + 6",
            "times": "12: de twee paden vermenigvuldigd, 2 · 6",
            "one": "6: het pad via $v$, het grootste",
        },
        correct="right",
        why={
            "times": "Langs één pad vermenigvuldig je, maar de paden zelf tel je op. Zie [Partiële afgeleiden en de gradient, formeel](/course/open/math-partial-derivatives?lang=nl).",
            "one": "$t$ bereikt $z$ ook via $u$, en elk pad telt. Zie [Partiële afgeleiden en de gradient, formeel](/course/open/math-partial-derivatives?lang=nl).",
            "right": "Het pad via $u$ geeft $1 \\cdot 2$, het pad via $v$ geeft $1 \\cdot 2t = 6$: samen 8.",
        },
    )
    quick_paths
    return (quick_paths,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`A = np.array([[1, 2, 3], [4, 5, 6]])`. Wat is `A.T[2, 0]`?",
        options={
            "six": "6",
            "right": "3",
            "error": "Een fout: `A` heeft maar 2 rijen",
        },
        correct="right",
        why={
            "six": "6 staat ook in kolom 2 van `A`, maar in rij 1. Rij 2 van `A.T` is kolom 2 van `A`, `[3, 6]`, en het eerste getal daarvan is 3. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "error": "`A.T` heeft de shape (3, 2), dus hij heeft wel een rij 2: kolom 2 van `A`. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "right": "Rij 2 van `A.T` is kolom 2 van `A`, `[3, 6]`: `A.T[2, 0]` is `A[0, 2]`, en dat is 3.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een layer van een netwerk rekent $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ uit voor een hele batch tegelijk, zoals in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl), hier zonder de bias terms. Met 3 features en 2 outputs heeft $\mathbf{W}$ al 6 getallen, en een echte layer heeft er duizenden. Om hem te trainen heb je de partiële afgeleide van de loss naar elk van die getallen nodig. Eén voor één is dat eindeloos boekhouden. Met matrices geeft één product ze allemaal, en de shapes vertellen je welk product.

    **Een voorbeeld uit het dagelijks leven.** Als je een bouwpakket in elkaar zet, vertelt de vorm van elk onderdeel waar het hoort: een plank met alleen langs één rand gaten past maar op één manier. De handleiding heb je nauwelijks nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_shape():
    guess_shape = L.guess(
        "shape-of-grad",
        question="$\\mathbf{W}$ heeft de shape (2, 3): één rij per output, één kolom per feature. Welke shape denk je dat de gradient van de loss naar $\\mathbf{W}$ heeft?",
        options={
            "flipped": "(3, 2): de getransponeerde van $\\mathbf{W}$",
            "one": "(1,): één getal, zoals de loss",
            "right": "(2, 3): één getal per weight",
        },
        reveal="**(2, 3).** De gradient bevat één partiële afgeleide voor elk getal van $\\mathbf{W}$, en zet die op dezelfde plek: het getal in rij $k$, kolom $j$ zegt hoe de loss verandert als je $W_{kj}$ een klein beetje verschuift. Dus heeft hij de shape van $\\mathbf{W}$. De loss is één getal, maar heeft hier 6 partiële afgeleiden. Sommige boeken schrijven deze gradient getransponeerd; de course houdt, net als PyTorch, de shape van $\\mathbf{W}$ aan.",
    )
    guess_shape
    return (guess_shape,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Matrixcalculus** rekent afgeleiden uit voor hele vectoren en matrices tegelijk. De gradient van de loss $\mathcal{L}$ naar een matrix $\mathbf{W}$, geschreven als $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$, bevat één partiële afgeleide voor elk getal van $\mathbf{W}$, op dezelfde plek: hij heeft de shape van $\mathbf{W}$.

        Neem een layer $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$, met $\mathbf{X}$ van shape $(n, d)$: $n$ voorbeelden met $d$ features; $\mathbf{W}$ van shape $(m, d)$: één rij per output; en $\mathbf{Y}$ van shape $(n, m)$. Stel dat de gradient naar $\mathbf{Y}$ bekend is: $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$, met de shape van $\mathbf{Y}$. Een weight $W_{kj}$ bereikt de loss via kolom $k$ van $\mathbf{Y}$, één keer voor elk voorbeeld $i$: in $Y_{ik}$ zit $X_{ij} \cdot W_{kj}$, dus de lokale afgeleide is $X_{ij}$. De multivariabele kettingregel telt de paden op: $\frac{\partial \mathcal{L}}{\partial W_{kj}} = \sum_i G_{ik} \, X_{ij}$, en dat is het getal in rij $k$, kolom $j$ van $\mathbf{G}^\top \mathbf{X}$. Op dezelfde manier voor $\mathbf{X}$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{W}}]] = [[grad:\mathbf{G}]]^\top [[data:\mathbf{X}]] \qquad [[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{X}}]] = [[grad:\mathbf{G}]] \, [[param:\mathbf{W}]]"),
        mo.md(r"""
        Die som hoef je niet elke keer uit te schrijven: de shapes verraden hem. $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ moet de shape $(m, d)$ hebben. Uit $\mathbf{G}$, $(n, m)$, en $\mathbf{X}$, $(n, d)$, geeft maar één product $(m, d)$: $\mathbf{G}^\top \mathbf{X}$, dat is $(m, n)$ @ $(n, d)$. Hetzelfde werkt voor een lineair model met de mean squared error, $\mathcal{L} = \frac{1}{n} \sum_i r_i^2$ met de residuen $\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w}$: de gradient naar de voorspellingen is $-\frac{2}{n} \mathbf{r}$, dus de gradient naar $\mathbf{w}$ is $-\frac{2}{n} \mathbf{X}^\top \mathbf{r}$, één getal per weight. Dat is de `-2 * (r @ X) / len(y)` die je schreef in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl), nu gevonden met de shapes.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **matrixcalculus** | | afgeleiden voor hele vectoren en matrices tegelijk; de gradient naar een matrix heeft de shape van die matrix | |
        | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ | partieel L naar partieel W | de gradient van de loss naar de matrix $\mathbf{W}$: één partiële afgeleide per getal van $\mathbf{W}$, op zijn plek | `dW` |
        | $\mathbf{G}$ | G | hier: de gradient van de loss naar de output $\mathbf{Y}$, met de shape van $\mathbf{Y}$ | `G` |
        """),
        L.watch_out("`G.T @ X` en `X.T @ G` werken allebei, maar alleen de eerste heeft de shape van `W`; de tweede is de getransponeerde, de formule voor een layer die je schrijft als `X @ W` met één kolom per output. Passen de shapes niet, of heeft het resultaat niet de shape van de matrix waar het de gradient van is, dan heb je het verkeerde product. Bij een vector, zoals `r` van shape (4,), geven `X.T @ r` en `r @ X` hetzelfde."),
        mo.md(r"""
        De figuur: de shapes van $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ voor 4 voorbeelden, 3 features en 2 outputs, en de twee producten die de gradients geven. De grootte van elk vak volgt zijn shape.
        """),
        mo.Html(C.shapes_picture({"title": "Volg de shapes", "forward": "de layer",
                                  "for_w": "de gradient naar W, met de shape van W",
                                  "for_x": "de gradient naar X, met de shape van X",
                                  "alt": "Drie rijen vakken, elk zo groot als zijn shape. Eerste rij: X (4, 3) keer W getransponeerd (3, 2) geeft Y (4, 2). Tweede rij: G getransponeerd (2, 4) keer X (4, 3) geeft dW (2, 3), de shape van W. Derde rij: G (4, 2) keer W (2, 3) geeft dX (4, 3), de shape van X."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de gradient van de loss naar een matrix te vinden:

    1. Schrijf de shape van elke matrix op: de input, de weights, de output, en de gradient naar de output.
    2. De gradient naar een matrix heeft de shape van die matrix.
    3. Zoek het product van de andere twee dat precies die shape geeft: transponeer er één waar nodig, zodat de binnenste getallen gelijk zijn.
    4. Controleer één getal met een kleine stap: verschuif één getal van de matrix, en vergelijk.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een layer met 4 voorbeelden, 3 features en 2 outputs, en als loss de helft van de som van de gekwadrateerde verschillen met de labels `_T`: voor die loss is de gradient naar $\mathbf{Y}$ precies $\mathbf{Y} - \mathbf{T}$. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 controleert één getal; voeg een controle van een tweede toe, in rij 0, kolom 2, waar het commentaar dat zegt, en voer de cel opnieuw uit.
    """)
    return


@app.cell
def example_shapes():
    import numpy as np

    _X = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])   # 4 voorbeelden, 3 features
    _W = np.array([[0.5, 1.0, 0.0], [-1.0, 0.0, 2.0]])   # 2 outputs, elk een rij, 3 features
    _T = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])   # de labels

    def _loss(W):
        return ((_X @ W.T - _T) ** 2).sum() / 2

    # 1. de shapes: X (4, 3), W (2, 3), en G = Y - T met de shape van Y, (4, 2)
    _G = _X @ _W.T - _T
    print(_X.shape, _W.shape, _G.shape)
    # 2. en 3. dW heeft de shape van W, (2, 3): (2, 4) @ (4, 3) is G.T @ X
    _dW = _G.T @ _X
    print(_dW)
    # 4. controleer het getal in rij 1, kolom 0: verschuif alleen die weight
    _h = 0.001
    _nudged = _W + np.array([[0.0, 0.0, 0.0], [_h, 0.0, 0.0]])
    print((_loss(_nudged) - _loss(_W)) / _h, _dW[1, 0])
    # 4. controleer rij 0, kolom 2 op dezelfde manier: zet _h op die plek, en print beide getallen
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{G}^\top \mathbf{X}$ heeft de shape (2, 3), de shape van $\mathbf{W}$, en het getal in rij 1, kolom 0 is −2. De kleine stap geeft ongeveer −1,997: dichtbij, dus het product klopt. In rij 0, kolom 2 geeft het product 0,5, en de kleine stap ongeveer 0,5015. Alle zes partiële afgeleiden kwamen uit één product, zonder de loss ook maar één keer extra uit te rekenen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_x():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De gradient naar X
    """)
    return


@app.cell(hide_code=True)
def choice_x_cell():
    choice_x = L.choice(
        "grad-x",
        question="$\\mathbf{Y} = \\mathbf{X}\\mathbf{W}^\\top$, met `X` van shape (5, 3) en `W` van shape (4, 3). `G` is de gradient naar $\\mathbf{Y}$. Wat geeft de gradient naar `X`?",
        options={
            "right": "`G @ W`",
            "wg": "`W @ G`",
            "gx": "`G.T @ X`",
        },
        correct="right",
        why={
            "wg": "`W` is (4, 3) en `G` is (5, 4): de binnenste getallen, 3 en 5, zijn niet gelijk, dus dit stopt met een fout.",
            "gx": "Dat is de gradient naar `W`, met shape (4, 3). De gradient naar `X` heeft de shape van `X`, (5, 3).",
            "right": "`G` is (5, 4) en `W` is (4, 3): samen (5, 3), de shape van `X`.",
        },
    )
    choice_x
    return (choice_x,)


@app.cell(hide_code=True)
def text_exercise_w():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twee modellen tegelijk

        Twee lineaire modellen voorspellen uit dezelfde features, zoals in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl): elk model is een rij van `W`, en `X @ W.T` geeft de voorspellingen van allebei, één kolom per model. De loss is de mean squared error van het eerste model plus die van het tweede. `grad_W(X, W, Y)` moet de gradient van die loss naar `W` geven. Schrijf de body van de functie: de residuen `R = Y - X @ W.T`, het aantal voorbeelden, en de gradient. De check roept `grad_W(np.array([[1, 1, 0], [0, 2, 1], [3, 0, 1], [1, 1, 2]]), np.array([[1, 0, 1], [0.5, 1, 0]]), np.array([[1, 2], [2, 2], [4, 1], [2, 2]]))` aan, en dat moet `np.array([[0.5, -0.5, 0.5], [0.25, -0.5, -0.25]])` geven; hij probeert ook andere data.
        """),
        L.task("Laat `grad_W` de gradient van de loss naar `W` geven, met de shape van `W`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_w():
    box_w = L.code_box("grad-two", lines=8)
    box_w
    return (box_w,)


@app.cell(hide_code=True)
def check_w(box_w):
    result_w = L.check("grad-two", box_w.value)
    result_w
    return (result_w,)


@app.cell(hide_code=True)
def hints_w():
    L.hints(
        "grad-two",
        [
            "Stap 1 tot en met 3 van het recept. De gradient naar `W` heeft de shape van `W`: één rij per model, één kolom per feature. `X` is (n, features), en `R` heeft één rij per voorbeeld en één kolom per model.",
            "Voor elk model is de gradient naar zijn voorspellingen $-\\frac{2}{n}$ keer zijn residuen, dus voor alle voorspellingen samen $-\\frac{2}{n} \\mathbf{R}$. Welk product van `R` en `X` geeft één rij per model?",
        ],
        why="Hier is `R` gelijk aan `[[0, 0.5], [1, 0], [0, -0.5], [-1, 0.5]]`, en `R.T @ X` geeft `[[-1, 1, -1], [-0.5, 1, 0.5]]`. Keer $-\\frac{2}{4}$: `[[0.5, -0.5, 0.5], [0.25, -0.5, -0.25]]`. Elke rij is de gradient van één model, zoals in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Beide gradients van een layer

        `layer_grads(X, W, G)` moet de gradients van de loss naar `W` en naar `X` geven, voor de layer `Y = X @ W.T`, uit de gradient `G` naar `Y`. Schrijf de body van de functie volgens het recept, en geef `[dW, dX]` terug. De check roept hem aan met `X` van shape (4, 3), `W` van shape (2, 3) en `G` van shape (4, 2), en probeert ook andere shapes.
        """),
        L.task("Maak van `dW` de gradient naar `W` en van `dX` de gradient naar `X`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("layer-grads", lines=8)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("layer-grads", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "layer-grads",
        [
            "Stap 2 van het recept: `dW` heeft de shape van `W`, (2, 3), en `dX` de shape van `X`, (4, 3).",
            "Elke gradient is een product van `G` en de andere matrix. Schrijf hun shapes op: moet een van de twee getransponeerd worden, en staat `G` voor of achter?",
        ],
        why="`G.T @ X` is (2, 4) @ (4, 3), de shape van `W`; `G @ W` is (4, 2) @ (2, 3), de shape van `X`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De multivariabele kettingregel uit [Partiële afgeleiden en de gradient, formeel](/course/open/math-partial-derivatives?lang=nl) telt de paden op; bij een matrixproduct zijn die sommen een product met een getransponeerde, zoals in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl). De gradient naar een matrix heeft de shape van die matrix, en de shapes vertellen je welk product hem geeft. In de volgende les, *Rang: hoeveel richtingen een matrix heeft*, zie je hoeveel richtingen een matrix echt heeft, hoe groot hij ook is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_x, result_w, result_layer):
    L.wrap_up(choice_x, result_w, result_layer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
