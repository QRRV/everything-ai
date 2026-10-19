# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-digits-linear. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow multinomiale logistische regressie, transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-digits-linear", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl), [Broadcasting](/course/open/np-broadcasting?lang=nl) en [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`diff` heeft shape (6, 3) en `X` heeft shape (6, 4). Welk product heeft shape (3, 4)?",
        options={
            "x-t": "`X.T @ diff`",
            "right": "`diff.T @ X`",
            "no-t": "`diff @ X`",
        },
        correct="right",
        why={
            "x-t": "`X.T @ diff` is (4, 6) keer (6, 3), dus het heeft shape (4, 3). Welke van de twee moet omklappen, zodat de 3 vooraan komt? Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "no-t": "`diff @ X` zet (6, 3) voor (6, 4): de binnenste getallen, 3 en 6, verschillen, dus Python stopt. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "right": "`diff.T` heeft shape (3, 6), en (3, 6) keer (6, 4) geeft (3, 4).",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def quick_check_keepdims():
    quick_keepdims = L.choice(
        "quick-keepdims",
        question="`m` heeft shape (4, 3). Welke regel deelt elke rij van `m` door de som van die rij?",
        options={
            "total": "`m / m.sum()`",
            "columns": "`m / m.sum(axis=0)`",
            "right": "`m / m.sum(axis=1, keepdims=True)`",
        },
        correct="right",
        why={
            "total": "Dat deelt elk getal door de som van de hele matrix: dan tellen alle twaalf samen op tot 1, niet elke rij. Zie [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "columns": "`m.sum(axis=0)` heeft shape (3,): één som per kolom, dus elke kolom wordt door zijn eigen som gedeeld. Welke axis telt op langs een rij? Zie [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "right": "`m.sum(axis=1, keepdims=True)` heeft shape (4, 1): één som per rij, als kolom. Broadcasting deelt dan elke rij door zijn eigen som.",
        },
    )
    quick_keepdims
    return (quick_keepdims,)


@app.cell(hide_code=True)
def quick_check_slopes():
    quick_slopes = L.choice(
        "quick-slopes",
        question="Een model voor de cijfers 0, 1 en 2 geeft een beeld van een 2 de kansen 0,1, 0,3 en 0,6. Wat zijn de hellingen van zijn cross-entropy voor de drie logits?",
        options={
            "right": "$(0{,}1;\\ 0{,}3;\\ -0{,}4)$",
            "first": "$(-0{,}9;\\ 0{,}3;\\ 0{,}6)$",
            "plain": "$(0{,}1;\\ 0{,}3;\\ 0{,}6)$",
        },
        correct="right",
        why={
            "first": "Dan haal je de 1 van klasse 0 af. Welke klasse is goed voor een beeld van een 2? Zie het recept in [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl).",
            "plain": "Dat zijn de kansen zelf. Wat haal je ervan af? Zie het recept in [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl).",
            "right": "De kansen min het one-hot label $(0; 0; 1)$: alleen de goede klasse krijgt een negatieve helling, dus gradient descent verhoogt zijn logit.",
        },
    )
    quick_slopes
    return (quick_slopes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl) trainde je project een classifier voor twee klassen. Handgeschreven cijfers hebben er tien: 0 tot en met 9. Alle onderdelen voor meer klassen ken je al: softmax maakt van logits kansen, cross-entropy zegt hoe ver die ernaast zitten, en je kent de hellingen ervan. Met dezelfde loop vormen ze een classifier voor tien klassen. Vandaag train je die op echte beelden, 6000 handgeschreven cijfers, en beoordeel je hem zoals je hebt geleerd: met de accuracy naast een baseline, en met welke cijfers hij door elkaar haalt. Dat is de derde stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           2, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Sorteermachines bij de post lezen de cijfers van de postcode die iemand met de hand op een envelop schreef, één voor één, en sturen de brief door naar de goede plek.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share-right",
        question="Een model met per cijfer één gewogen som van de 784 pixels traint op 6000 handgeschreven cijfers. Hoeveel van 1000 nieuwe cijfers leest het volgens jou goed?",
        options={
            "hundred": "Ongeveer 100: nauwelijks meer dan gokken",
            "half": "Ongeveer 500",
            "most": "Ongeveer 900",
            "all": "Op een paar na allemaal",
        },
        reveal="**Ongeveer 900.** In de demo aan het eind van deze les leest je eigen model 890 van de 1000 nieuwe cijfers goed. Altijd het meest voorkomende cijfer zeggen, een 1, haalt er 126: dat is de baseline. Maar één gewogen som per cijfer leest ze niet allemaal: handschriften verschillen te veel, en in de demo zie je welke cijfers het door elkaar haalt.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een beeld van 28 bij 28 pixels is 784 getallen. Rij na rij gelezen vormen ze één rij van $\mathbf{X}$: een voorbeeld met 784 features, één per pixel, van 0 voor papier tot 1 voor inkt. **Softmax-regressie** is logistische regressie voor meer dan twee klassen. Elke klasse $k$ krijgt eigen weights $\mathbf{w}_k$ en een eigen bias term $b_k$, dus een eigen logit voor voorbeeld $i$:
        """),
        L.formula(r"z_{ik} = [[param:\mathbf{w}_k]] \cdot \mathbf{x}_i + [[param:b_k]] \qquad k = 0, 1, \dots, K - 1"),
        mo.md(r"""
        Softmax maakt van de $K$ logits van een voorbeeld $K$ kansen $\hat{y}_{ik}$, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl). De weights van klasse $k$ zijn rij $k$ van de **weight-matrix** $\mathbf{W}$, met shape $(K, d)$: hier 10 rijen van 784. Voor alle voorbeelden en klassen tegelijk zijn de logits `X @ W.T + b`, met shape $(n, K)$: één rij per voorbeeld, één kolom per klasse.

        De hellingen komen uit [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl): voor één voorbeeld is de helling voor logit $k$ gelijk aan $\hat{y}_k - y_k$, waarbij $y_k$ 1 is voor de goede klasse en 0 voor de andere. De kettingregel vermenigvuldigt die met de feature, net als bij log loss, en de cross-entropy van $n$ voorbeelden is hun gemiddelde:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_{kj}}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_{ik}]] - y_{ik}) \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b_k}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_{ik}]] - y_{ik})"),
        mo.md(r"""
        In code bevat `diff = P - np.eye(K)[y]` de verschillen $\hat{y}_{ik} - y_{ik}$, één rij per voorbeeld. Rij $k$ van `diff.T` bevat die van klasse $k$ voor alle voorbeelden; het inproduct ervan met kolom $j$ van `X` is de som over $i$ in de formule. Dus `diff.T @ X / n` geeft elke helling tegelijk, met de shape van `W`, $(K, d)$. De hellingen voor `b` zijn het gemiddelde van elke kolom van `diff`. Naast je logistische regressie:

        | | Logistische regressie | Softmax-regressie |
        |---|---|---|
        | weights | `w`, shape $(d,)$ | `W`, shape $(K, d)$: één rij per klasse |
        | bias term | `b`, één getal | `b`, shape $(K,)$: één per klasse |
        | kansen | `1 / (1 + np.exp(-(X @ w + b)))` | softmax langs elke rij van `X @ W.T + b` |
        | verschillen | `p - y` | `P - np.eye(K)[y]` |
        | hellingen voor de weights | `(p - y) @ X / n` | `diff.T @ X / n` |
        | hellingen voor de bias term | `(p - y).mean()` | `diff.mean(axis=0)` |
        | één stap | `w = w - lr * grad_w` | `W = W - lr * grad_W` |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **softmax-regressie** | | logistische regressie voor meer dan twee klassen: één gewogen som per klasse, met softmax omgezet in kansen; in de statistiek heet het multinomiale logistische regressie | `X @ W.T + b`, dan softmax |
        | $\mathbf{W}$ | hoofdletter W | de **weight-matrix**: één rij weights per klasse, shape $(K, d)$ | `W` |
        | $w_{kj}$ | w k j | de weight van klasse $k$ voor feature $j$ | `W[k, j]` |
        | **`np.load(path)`** | | leest arrays die NumPy in een bestand bewaarde; een `.npz`-bestand bevat er meerdere, elk onder een eigen naam | `np.load(path)["train_images"]` |
        | **`.reshape(rows, columns)`** | reshape | dezelfde getallen in een andere shape, rij na rij gevuld | `X[0].reshape(28, 28)` |
        """),
        L.watch_out("Softmax gaat langs elke rij: één voorbeeld, al zijn klassen. `Z.max(axis=1)` heeft shape $(n,)$ en past niet bij `Z`, met shape $(n, K)$. Met `keepdims=True` heeft het shape $(n, 1)$, een kolom, en verliest elke rij zijn eigen grootste logit. Hetzelfde geldt voor de som waardoor je deelt."),
        mo.Html(C.images_picture(C.examples(), [str(_k) for _k in range(10)], "Het eerste beeld van elk cijfer in de training set",
                                 "Tien handgeschreven cijfers, 0 tot en met 9, elk getekend uit zijn 784 getallen als 28 rijen van 28 vierkantjes: hoe meer inkt, hoe sterker het vierkantje.")),
        mo.md(r"""
        Elk beeld is één rij van $\mathbf{X}$, teruggetekend als 28 rijen van 28 pixels: hoe meer inkt, hoe sterker het vierkantje. Hetzelfde cijfer ziet er elke keer dat iemand het schrijft anders uit. Eén gewogen som per cijfer moet met al die verschillen overweg kunnen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén stap gradient descent voor softmax-regressie, op $n$ voorbeelden en $K$ klassen:

    1. Geef elk voorbeeld zijn $K$ logits: `Z = X @ W.T + b`, shape $(n, K)$.
    2. Softmax langs elke rij: `E = np.exp(Z - Z.max(axis=1, keepdims=True))` en `P = E / E.sum(axis=1, keepdims=True)`.
    3. Haal de one-hot labels eraf: `diff = P - np.eye(K)[y]`.
    4. Reken elke helling uit: `grad_W = diff.T @ X / n` en `grad_b = diff.mean(axis=0)`.
    5. Stap tegen de hellingen in, `W = W - lr * grad_W` en `b = b - lr * grad_b`, en herhaal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cijfers komen uit QMNIST, een set handgeschreven cijfers met een open licentie, die Chhavi Yadav en Léon Bottou in 2019 opnieuw opbouwden uit dezelfde formulieren als het bekende MNIST. De course heeft er de eerste 6000 trainingsbeelden en de eerste 1000 testbeelden van in `data/qmnist-small.npz`, elk beeld als 784 getallen van 0 tot 255; gedeeld door 255 lopen ze van 0 tot 1. De eerste cel laadt de trainingsbeelden en laat het eerste zien. De tweede zet één stap vanaf $\mathbf{W} = 0$ en $\mathbf{b} = 0$, met $\eta$ = 1, en print de cross-entropy ervoor en erna. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de twee cellen uit.
    """)
    return


@app.cell
def example_data():
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load("data/qmnist-small.npz")
    X = data["train_images"] / 255
    y = data["train_labels"]
    print(X.shape)
    print(y[:10])
    # het eerste beeld: zijn 784 getallen als 28 rijen van 28
    plt.imshow(X[0].reshape(28, 28), cmap="gray")
    plt.show()
    return X, data, np, plt, y


@app.cell
def example_step(X, np, y):
    _W = np.zeros((10, 784))
    _b = np.zeros(10)

    def _cross_entropy(W, b):
        _Z = X @ W.T + b
        _E = np.exp(_Z - _Z.max(axis=1, keepdims=True))
        _P = _E / _E.sum(axis=1, keepdims=True)
        return -np.mean((np.eye(10)[y] * np.log(_P)).sum(axis=1))

    # 1. tien logits voor elk beeld
    _Z = X @ _W.T + _b
    print(_Z.shape)
    # 2. softmax langs elke rij
    _E = np.exp(_Z - _Z.max(axis=1, keepdims=True))
    _P = _E / _E.sum(axis=1, keepdims=True)
    print(_P[0])
    # 3. de kansen min de one-hot labels
    _diff = _P - np.eye(10)[y]
    print(_diff[0])
    # 4. elke helling tegelijk
    _grad_W = _diff.T @ X / len(y)
    _grad_b = _diff.mean(axis=0)
    print(_grad_W.shape)
    print(_grad_b)
    # 5. één stap, en de cross-entropy ervoor en erna
    _lr = 1.0
    print(_cross_entropy(_W, _b), _cross_entropy(_W - _lr * _grad_W, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `X` heeft shape (6000, 784): 6000 beelden van 784 pixels. De eerste tien labels zijn 5, 0, 4, 1, 9, 2, 1, 3, 1 en 4, en het eerste beeld is inderdaad een 5, in wit: met `cmap="gray"` is 0 zwart en 1 wit. `Z` heeft shape (6000, 10), tien logits per beeld. Met alle weights 0 is elke logit 0 en elke kans 0,1: het model weet nog niets. Voor die 5 zijn de verschillen 0,1 voor de negen andere cijfers en −0,9 voor de 5. De hellingen voor `W` hebben de shape van `W`, (10, 784). Elke helling voor `b` is 0,1 min het aandeel van dat cijfer in de training set, dus dicht bij 0: ongeveer −0,012 voor de 1'en, het meest voorkomende cijfer. Eén stap brengt de cross-entropy omlaag van 2,30, de loss van een kans van 0,1 voor het goede cijfer, naar 1,47. In oefening 3 herhaalt je `fit` deze stap in een loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel getallen?
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "how-many",
        question="Hoeveel getallen leert softmax-regressie voor beelden van 784 pixels en 10 cijfers?",
        options={
            "one-row": "794",
            "no-b": "7840",
            "right": "7850",
        },
        correct="right",
        why={
            "one-row": "Dat is één weight per pixel, plus één bias term per cijfer. Hoeveel weights krijgt elk cijfer? Zie *Het idee*.",
            "no-b": "Dat zijn alle weights: 10 rijen van 784. Wat krijgt elk cijfer nog meer? Zie *Het idee*.",
            "right": "10 rijen van 784 weights, 7840, plus 10 bias terms: 7850. Logistische regressie op dezelfde pixels leert er 785.",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De hellingen van de cross-entropy

        Maak `cross_entropy_gradients(X, y, W, b)` af. Hij krijgt de voorbeelden `X`, de labels `y` als klassenummers, de weights `W`, één rij per klasse, en de bias terms `b`, één per klasse. Hij geeft de hellingen van de cross-entropy terug: `grad_W`, met de shape van `W`, en `grad_b`, met de shape van `b`. Vervang de `...` door stap 1 tot en met 4 van het recept; `len(b)` is het aantal klassen. De check roept `cross_entropy_gradients([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], [[0, 0], [0, 0], [0, 0]], [0, 0, 0])` aan: vier voorbeelden met twee features, en drie klassen. Dat moet ongeveer `([[-0.0833, 0.1667], [0.1667, 0.1667], [-0.0833, -0.3333]], [0.0833, 0.0833, -0.1667])` geven. Hij probeert ook andere voorbeelden, weights, bias terms en aantallen klassen, ook met heel grote logits. Als je functie klopt, bewaart de check hem in je project, als `work/projects/nn/softmax_gradients.py`.
        """),
        L.task("Laat `cross_entropy_gradients` de hellingen van de cross-entropy teruggeven voor `W` en voor `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("cross-entropy-gradients", lines=14)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("cross-entropy-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "cross-entropy-gradients",
        [
            "Stap 1 tot en met 4 van het recept, in dezelfde volgorde: de logits, softmax langs elke rij, de verschillen, en dan de twee hellingen.",
            "`W` heeft één rij per klasse, dus de logits zijn `X @ W.T + b`. Welke shape moet `grad_W` hebben, en welke van `diff` en `X` moet je omklappen om die te krijgen?",
            "`Z = X @ W.T + b`, dan `E` en `P` zoals in stap 2, `diff = P - np.eye(len(b))[y]`, `grad_W = ___ @ X / len(y)` en `grad_b = diff.mean(axis=___)`.",
        ],
        why="Met `W` en `b` helemaal 0 is elke kans een derde, dus elke rij van `diff` is een derde min het one-hot label: −2/3 voor de goede klasse en 1/3 voor de andere twee. `diff.T @ X` telt per klasse de verschillen op van de voorbeelden die elke feature hebben; gedeeld door 4 geeft dat de hellingen voor `W`. Het gemiddelde van elke kolom van `diff` geeft ongeveer `[0.0833, 0.0833, -0.1667]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen

        `fit(X, y, K, lr, steps)` haalt je `cross_entropy_gradients` uit je project binnen, begint met `W` en `b` helemaal 0, `K` rijen weights en `K` bias terms voor `K` klassen, en draait zijn loop `steps` keer: dezelfde loop als in je `fit` voor logistische regressie. Vervang de `...` door één update-stap: haal beide hellingen met `softmax_gradients.cross_entropy_gradients(X, y, W, b)`, en schuif `W` en `b` ertegenin. De check roept `fit([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], 3, 0.5, 100)` aan, met de voorbeelden van oefening 2. Hij probeert ook andere data, aantallen klassen, learning rates en aantallen stappen. Als je functie klopt, bewaart de check hem in je project, als `work/projects/nn/softmax.py`.
        """),
        L.task("Laat `fit` `W` en `b` trainen met je eigen `cross_entropy_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-softmax", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # check again when exercise 2 saves your module
    result_fit = L.check("fit-softmax", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-softmax",
        [
            "Stap 5 van het recept, met de hellingen uit je eigen functie: dezelfde loop als in je `fit` voor logistische regressie.",
            "De functie woont in de module `softmax_gradients`: roep hem aan via de module. Hij geeft twee dingen tegelijk terug: hoe geef je elk een eigen naam?",
            "`grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)`, dan `W = W - ___` en `b = b - ___`.",
        ],
        why="`softmax_gradients.cross_entropy_gradients(X, y, W, b)` geeft beide hellingen; `grad_W, grad_b = ...` vangt ze op. `W = W - lr * grad_W` en `b = b - lr * grad_b` stappen ertegenin. Na 100 rondes op het voorbeeld heeft klasse 2, waarvan beide voorbeelden de tweede feature hebben, daarvoor een weight van ongeveer 3,61. Klasse 1, waarvan het voorbeeld geen van beide features heeft, heeft voor allebei een negatieve weight: ongeveer −2,18 en −1,67.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder traint een model met je eigen `fit` op de 6000 trainingscijfers, met $\eta$ = 1 en 100 stappen, en meet het op de 1000 testcijfers, die het nooit zag. Dat duurt een paar seconden. De cel gebruikt je eigen module, dus hij werkt zodra oefening 3 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_digits(result_fit):
    result_fit  # run again when exercise 3 saves your module
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Maak eerst oefening 3 af: dan staat je `fit` in je project, en gebruikt deze cel hem.*")
    elif _model == "slow":
        _out = mo.md("*Je `fit` in `work/projects/nn/softmax.py` doet er op de 6000 cijfers meer dan een minuut over. Gaat hij de voorbeelden een voor een langs? Het recept doet alle voorbeelden tegelijk, met `@`.*")
    elif isinstance(_model, str):
        _out = mo.md("*Je `fit` haalt zijn hellingen uit `cross_entropy_gradients` in `work/projects/nn/softmax_gradients.py`, en dat bestand ontbreekt, of geeft niet meer de goede hellingen. Laat oefening 2 opnieuw nakijken; heb je dat bestand veranderd, verwijder het dan eerst.*")
    elif not _model:
        _out = mo.md("*De `fit` in `work/projects/nn/softmax.py` traint niet meer goed, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    else:
        _W, _b = _model
        _accuracy = C.accuracy(_W, _b)
        _digit, _baseline = C.baseline()
        _cm = C.confusion(_W, _b)
        _true, _said, _times = C.most_confused(_cm)
        _out = mo.vstack([
            mo.md(f"""Je model leest <span class="ea-num">{L.number(_accuracy * 1000, 0)}</span> van de 1000 testcijfers goed: een accuracy van <span class="ea-num">{L.number(_accuracy, 3)}</span>. Altijd het meest voorkomende cijfer van de training set zeggen, een {_digit}, haalt <span class="ea-num">{L.number(_baseline, 3)}</span>. Je model verslaat de baseline ruim."""),
            mo.Html(C.confusion_picture(_cm, (_true, _said), {"title": "Je model op de 1000 testcijfers", "label": "de confusion matrix",
                                                              "caption": "cm[label, pred]", "lines": ["rijen: het echte cijfer", "kolommen: het cijfer dat je model zegt",
                                                                                                         f"vet: {_true} gelezen als {_said}"]})),
            mo.md(f"""Op de diagonaal staan de cijfers die je model goed leest; elk ander vakje telt cijfers die het voor een ander cijfer aanzag. Het vaakst ziet het een {_true} aan voor een {_said}: {_times} keer. Sommige cijfers leest het bijna altijd goed, zoals de 0'en en de 1'en; andere minder vaak, zoals de 2'en en de 8'en."""),
            mo.Html(C.images_picture(_W, [str(_k) for _k in range(10)], "De tien rijen van je W, als beelden",
                                     "Tien beelden van 28 bij 28, één per cijfer: de weights van dat cijfer, oranje waar inkt zijn logit verhoogt en leisteengrijs waar inkt hem verlaagt.",
                                     signed=True)),
            mo.md("""Elk beeld is één rij van je `W`: de 784 weights van één cijfer, terug als 28 rijen van 28. Inkt op een oranje pixel verhoogt de logit van dat cijfer; inkt op een grijze pixel verlaagt hem. De 0 heeft een oranje ring met grijs in het midden: inkt in het midden pleit tegen een 0. Elke rij is één vast patroon per cijfer, en een cijfer dat op een andere plek of schuin geschreven is, past er minder goed bij. Daarom blijft een lineair model steken op ongeveer 9 van de 10. Een neural network met een hidden layer, verderop in dit project, leert meer patronen per cijfer."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl) vond je de hellingen $\hat{y}_k - y_k$ voor één voorbeeld; in [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl) schreef je de loop die tegen hellingen in stapt. Met een weight-matrix van één rij per klasse lezen ze nu handgeschreven cijfers. [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) en [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) gaven je de manieren om het model te beoordelen: de accuracy naast de baseline, en de confusion matrix. Je project `nn` heeft nu ook `softmax_gradients.py` en `softmax.py`. In de volgende les, *De workflow van machine learning*, zet je deze stappen op een rij: van de vraag naar de metric, de data, een baseline en een model, en dan weer rond.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, result_fit, result_gradients):
    L.wrap_up(choice_count, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
