# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-linear-regression-numpy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-linear-regression-numpy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) en [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` heeft shape (8, 3) en `w` heeft shape (3,). Wat is de shape van `X @ w + b`?",
        options={
            "inner": "(3,)",
            "right": "(8,)",
            "same": "(8, 3)",
        },
        correct="right",
        why={
            "inner": "Dat is het binnenste getal. Welk getal blijft over? Zie de shape-regel in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "same": "Elke rij van `X` geeft één inproduct, en `+ b` telt bij elk iets op. Zie het uitgewerkte voorbeeld in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "Eén voorspelling per rij: 8 getallen. `+ b` telt de bias term bij elk ervan op.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="De loss van één voorbeeld is $(y - w \\cdot x)^2$, met het residu $r = y - w \\cdot x$. Wat is de helling ervan naar $w$?",
        options={
            "outer": "$2r$",
            "inner": "$-x$",
            "right": "$2r \\cdot (-x)$",
        },
        correct="right",
        why={
            "outer": "Dat is alleen de buitenste helling. Wat doet de kettingregel met de binnenste? Zie *Het idee* in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "inner": "Dat is alleen de binnenste helling. Wat is de helling van de buitenste stap, $r^2$? Zie *Het idee* in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "right": "De buitenste stap $r^2$ heeft helling $2r$, de binnenste stap helling $-x$, en de kettingregel vermenigvuldigt ze.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl) mat je elke helling door een parameter een klein stukje te verschuiven: één extra loss voor elke parameter, bij elke stap. Met één feature gaat dat prima. Een huis heeft veel features, en een model voor tekst heeft miljoenen weights. Met de kettingregel krijg je elke helling exact, in één keer, met één matrixvermenigvuldiging. Dat is lineaire regressie met veel features, en het eerste stuk van je eigen neural network: dat leert op dezelfde manier, alleen met meer stappen in de ketting.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           0, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een geluidstechnicus achter een mengpaneel met veel schuiven. Elke schuif los uitproberen om te horen wat hij doet, kost veel tijd. Een technicus die weet hoe elke schuif werkt, verschuift ze allemaal tegelijk, elk de kant op die helpt.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_nudges():
    guess_nudges = L.guess(
        "nudges",
        question="Een model heeft 1000 weights en een bias term. Hoe vaak denk je dat je de loss uitrekent in één stap van gradient descent, als je elke helling meet door te verschuiven?",
        options={
            "two": "2",
            "thousand": "Ongeveer 1000",
            "million": "Ongeveer 1.000.000",
        },
        reveal="**Ongeveer 1000:** één keer waar je staat, en nog eens voor elk van de 1001 parameters. Elk daarvan is een rondgang langs alle data. De kettingregel geeft alle 1001 hellingen uit één rondgang: helemaal zonder verschuiven.",
    )
    guess_nudges
    return (guess_nudges,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Lineaire regressie voorspelt met een gewogen som van de features plus een bias term, $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$, en vindt $\mathbf{w}$ en $b$ door de mean squared error zo klein mogelijk te maken. Je kent elk stuk. Voor één voorbeeld geeft de kettingregel de helling $2r \cdot (-x)$ voor een weight. De MSE is het gemiddelde van de losses van alle voorbeelden. Schuift $w$ een piepklein stapje op, dan verandert elke loss met zijn eigen helling keer die stap, dus hun gemiddelde verandert met het gemiddelde van die veranderingen: de helling van de MSE is het gemiddelde van de hellingen. Het uitgewerkte voorbeeld controleert dat door te meten. Voor weight $j$ en voor $b$ zijn de hellingen:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = -\frac{2}{n} \sum_{i=1}^{n} [[residual:r_i]] \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = -\frac{2}{n} \sum_{i=1}^{n} [[residual:r_i]]"),
        mo.md(r"""
        Hier is $x_{ij}$ feature $j$ van voorbeeld $i$. De som over $i$ vermenigvuldigt elk residu met het getal op dezelfde plek in kolom $j$ van $\mathbf{X}$, en telt op: een inproduct. Voor alle weights tegelijk is dat `r @ X`. Met de vector voorop doet hij mee als één rij: (n,) keer (n, d) geeft (d,), één helling per weight.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **lineaire regressie** | | een model dat voorspelt met een gewogen som van de features plus een bias term, dat je past door de mean squared error zo klein mogelijk te maken | `X @ w + b` |
        | $x_{ij}$ | x i j | feature $j$ van voorbeeld $i$: rij $i$, kolom $j$ van $\mathbf{X}$ | `X[i, j]` |
        | `np.zeros(d)` | | een array van $d$ nullen, zoals de weights vóór het trainen | `np.zeros(3)` |
        """),
        L.watch_out("`r @ X`, niet `X @ r`. `X` heeft shape (n, d) en `r` shape (n,): met `X` voorop verschillen de binnenste getallen d en n, en stopt Python."),
        draw.array_grids([("r", [[str(_v).replace("-", "−") for _v in C.R_SMALL]], ["shape (4,)"]),
                          ("X", [[str(_v) for _v in _row] for _row in C.X_SMALL], ["shape (4, 2)", "de eerste kolom"], {(0, 0), (1, 0), (2, 0), (3, 0)}),
                          ("r @ X", [["3", "4"]], ["shape (2,)", "het eerste getal: 3"], {(0, 0)})],
                         title="Eén helling per kolom"),
        mo.md(r"""
        `r` keer de eerste kolom van `X` is 2 × 1 + (−1) × 2 + 1 × 0 + 3 × 1 = 3: het eerste getal van `r @ X`. De tweede kolom geeft 4.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén stap van gradient descent voor lineaire regressie:

    1. Voorspel voor alle voorbeelden tegelijk: `y_pred = X @ w + b`.
    2. Reken de residuen uit: `r = y - y_pred`.
    3. Reken elke helling uit met de kettingregel: `grad_w = -2 * (r @ X) / n` en `grad_b = -2 * r.mean()`.
    4. Stap tegen de hellingen in, `w = w - lr * grad_w` en `b = b - lr * grad_b`, en herhaal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Acht huizen, elk met drie features: de oppervlakte in honderden m², een tuin (1 = ja, 0 = nee), en de afstand tot het station in km. De prijs staat in duizenden euro's. Eén stap vanaf $\mathbf{w} = 0$ en $b = 0$, met $\eta$ = 0,2, en een controle van de eerste helling door te meten. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_step():
    import numpy as np

    _X = np.array(C.HOUSES)
    _y = np.array(C.PRICES)
    _w = np.zeros(3)
    _b = 0.0
    # 1. en 2. voorspellen, en de residuen
    _r = _y - (_X @ _w + _b)
    # 3. elke helling tegelijk, met de kettingregel
    _grad_w = -2 * (_r @ _X) / len(_y)
    _grad_b = -2 * _r.mean()
    print(_grad_w, _grad_b)

    # controleer de eerste helling door te meten, met een piepkleine h
    def _loss(w, b):
        return ((_y - (_X @ w + b)) ** 2).mean()

    _h = 0.001
    print((_loss(_w + np.array([_h, 0, 0]), _b) - _loss(_w, _b)) / _h)
    # 4. één update-stap
    _lr = 0.2
    print(_loss(_w, _b), _loss(_w - _lr * _grad_w, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De kettingregel geeft de hellingen van ongeveer −394,7; −263 en −374 voor de drie weights, en −442,5 voor $b$. Meten geeft voor de eerste −394,66: hetzelfde; de kettingregel hoefde niets te verschuiven. Eén stap brengt de loss van 52310,5 naar ongeveer 2530. In oefening 3 herhaal je deze stap in een loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel hellingen?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "slope-shape",
        question="`X` bevat 100 voorbeelden met 3 features: shape (100, 3). `r` bevat hun residuen: shape (100,). Wat is de shape van `r @ X`?",
        options={
            "right": "(3,)",
            "examples": "(100,)",
            "error": "Geen shape: ze passen niet",
        },
        correct="right",
        why={
            "examples": "Dan zou je één getal per voorbeeld krijgen. Wat geeft `r @ X` voor elke kolom van `X`? Zie *Het idee*.",
            "error": "Kijk naar de binnenste getallen: (100,) en (100, 3). Zijn ze gelijk? Zie *Het idee*.",
            "right": "(100,) en (100, 3): de binnenste getallen zijn allebei 100, en er blijft één getal per kolom over: één helling voor elk van de 3 weights.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Elke helling tegelijk

        Maak `mse_gradients(X, y, w, b)` af. Hij krijgt de voorbeelden `X`, de labels `y`, de weights `w` en de bias term `b`, en geeft de hellingen van de MSE terug: `grad_w`, één per weight, en `grad_b`. Vervang de `...` door stap 1 tot en met 3 van het recept. De check roept `mse_gradients([[1, 2], [3, 0], [0, 1]], [3, 0, 3], [0, 0], 0)` aan: dat moet `([-2.0, -6.0], -4.0)` geven. Hij probeert ook andere voorbeelden, weights en bias terms. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/nn/gradients.py`. `return grad_w, grad_b` geeft ze allebei tegelijk terug, als paar.
        """),
        L.task("Laat `mse_gradients` de hellingen van de MSE teruggeven, voor elke weight en voor `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("mse-gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("mse-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "mse-gradients",
        [
            "Stap 1 tot en met 3 van het recept, in dezelfde volgorde: eerst de residuen, dan de twee hellingen.",
            "`r` is het label min de voorspelling, voor alle voorbeelden tegelijk. In welke volgorde vermenigvuldig je `r` en `X`?",
            "`r = y - (X @ w + b)`, `grad_w = -2 * (___) / len(y)` en `grad_b = -2 * ___`.",
        ],
        why="`r = y - (X @ w + b)` geeft de residuen. `grad_w = -2 * (r @ X) / len(y)` is het gemiddelde van de kettingregel-hellingen voor elke weight, en `grad_b = -2 * r.mean()` hetzelfde voor $b$. In het voorbeeld is `r` gelijk aan `[3, 0, 3]`, dus `r @ X` is `[3, 9]`, en de hellingen zijn −2 × [3, 9] / 3 = [−2, −6] en −2 × 2 = −4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen

        Nu gaat je project aan het werk. `fit(X, y, lr, steps)` haalt je `mse_gradients` uit je project, begint met alle weights 0, één per kolom van `X` (`X.shape[1]` is het aantal kolommen), en $b$ = 0, en doorloopt zijn loop `steps` keer. Vervang de `...` door één update-stap: haal beide hellingen op met `gradients.mse_gradients(X, y, w, b)`, en schuif `w` en `b` ertegenin. De check roept `fit([[1, 0], [0, 1], [1, 1]], [3, 2, 5], 0.1, 500)` aan: de drie punten passen precies bij $3x_1 + 2x_2$, dus `w` moet eindigen bij ongeveer `[3, 2]` en `b` bij 0. Hij probeert ook andere data, learning rates en aantallen stappen. Klopt je functie, dan bewaart de check ook `fit` in je project, als `work/projects/nn/linear.py`.
        """),
        L.task("Laat `fit` de `w` en `b` trainen met je eigen `mse_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # opnieuw nakijken als oefening 2 je module bewaart
    result_fit = L.check("fit", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit",
        [
            "Stap 4 van het recept, met de hellingen uit je eigen functie.",
            "De functie woont in de module `gradients`: roep hem aan via de module. Hij geeft twee dingen tegelijk terug: hoe geef je elk zijn eigen naam?",
            "`grad_w, grad_b = gradients.mse_gradients(X, y, w, b)`, dan `w = w - ___` en `b = b - ___`.",
        ],
        why="`gradients.mse_gradients(X, y, w, b)` geeft beide hellingen; `grad_w, grad_b = ...` vangt ze op. `w = w - lr * grad_w` en `b = b - lr * grad_b` stappen ertegenin. Na 500 rondes in het voorbeeld is `w` ongeveer `[3.0, 2.0]` en `b` ongeveer 0,006.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder traint met je eigen `fit` een model op de acht huizen, met $\eta$ = 0,2 en 1000 stappen, en tekent wat het voorspelt tegen de echte prijzen. Elk blauw punt is een huis; op de stippellijn klopt een voorspelling precies. Hij gebruikt je eigen module, dus hij werkt zodra oefening 3 klaar is.
    """)
    return


@app.cell(hide_code=True)
def demo_houses(result_fit):
    result_fit  # opnieuw uitvoeren als oefening 3 je module bewaart
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Maak eerst oefening 3: dan staat je `fit` in je project, en gebruikt deze cel hem.*")
    elif _model == "gradients":
        _out = mo.md("*Je `fit` haalt zijn hellingen uit `mse_gradients` in `work/projects/nn/gradients.py`, en dat bestand ontbreekt, of geeft niet meer de goede hellingen. Laat oefening 2 opnieuw nakijken; heb je dat bestand veranderd, verwijder het dan eerst.*")
    elif not _model:
        _out = mo.md("*De `fit` in `work/projects/nn/linear.py` traint niet meer goed, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    else:
        _w, _b = _model
        _out = mo.vstack([
            mo.md(f"""Je model leerde de weights <span class="c-param ea-num">{L.number(_w[0], 1)}</span> voor de oppervlakte, <span class="c-param ea-num">{L.number(_w[1], 1)}</span> voor een tuin en <span class="c-param ea-num">{L.number(_w[2], 1)}</span> voor de afstand, en de bias term <span class="c-param ea-num">{L.number(_b, 1)}</span>. Dus 100 m² meer levert ongeveer 204 duizend euro op, een tuin ongeveer 17 duizend, en elke km naar het station kost ongeveer 39 duizend."""),
            mo.Html(C.demo_picture(_w, _b, {"title": "Je model op de acht huizen", "x": "prijs (duizend euro)", "y": "voorspelling",
                                            "alt": "Acht punten, één per huis: de echte prijs naar rechts, de voorspelling van je model omhoog, van 140 tot 350 duizend euro. Alle acht liggen dicht bij de stippellijn, waar een voorspelling precies klopt."})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) maakte je alle voorspellingen tegelijk; in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) vond je de helling van de loss van één voorbeeld; in [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl) stapte je bergaf. Samen zijn ze lineaire regressie, en je project `nn` heeft nu `gradients.py` en `linear.py`. In de volgende les, *Waarom gradient descent van geschaalde features houdt*, zie je waarom de eenheden van de huizen met zorg gekozen zijn: in m² en meters ontspoort $\eta$ = 0,2, en een learning rate die klein genoeg is, heeft veel meer stappen nodig.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_fit, result_gradients):
    L.wrap_up(choice_shape, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
