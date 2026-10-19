# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-train-logistic. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss, regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-train-logistic", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl) en [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="Een spammail, $y = 1$, krijgt $\\hat{y} = 0{,}7$ van het model, en zijn feature is $x = 2$. Wat is de helling van zijn log loss naar de weight van die feature?",
        options={
            "right": "$-0{,}6$",
            "sign": "$0{,}6$",
            "no-x": "$-0{,}3$",
        },
        correct="right",
        why={
            "sign": "De gradient begint bij $\\hat{y} - y$, de kans min het label, niet $y - \\hat{y}$. Zie de formule in [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl).",
            "no-x": "Dat is alleen $\\hat{y} - y$, de helling voor de bias term. Wat komt erbij voor de weight van een feature? Zie de formule in [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl).",
            "right": "$(0{,}7 - 1) \\cdot 2 = -0{,}6$: negatief, dus een grotere weight verlaagt de loss van deze mail.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Wat doet één ronde van de loop in je `fit` voor lineaire regressie?",
        options={
            "predict": "Hij voorspelt de labels nog een keer, en houdt de beste weights",
            "nudge": "Hij geeft elke weight een duwtje en meet de loss opnieuw",
            "right": "Hij haalt beide hellingen uit `mse_gradients`, en schuift `w` en `b` ertegenin",
        },
        correct="right",
        why={
            "predict": "Er wordt niets bewaard of vergeleken: elke ronde verandert `w` en `b`. Hoe? Zie oefening 3 van [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "nudge": "Duwtjes geven was de trage manier uit [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl). Waar haalt `fit` zijn hellingen vandaan? Zie oefening 3 van [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "right": "`grad_w, grad_b = gradients.mse_gradients(X, y, w, b)`, dan `w = w - lr * grad_w` en `b = b - lr * grad_b`, `steps` keer.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je hebt alle stukken van een classifier die leert: de sigmoid van een inproduct geeft een kans, log loss zegt hoe ver die ernaast zit, en de gradient $(\hat{y} - y)\,x$ zegt welke kant je elke weight op moet veranderen. In [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) trainde gradient descent een lijn. Nu traint hij logistische regressie, en de loop blijft precies hetzelfde: alleen de hellingen die hij krijgt, zijn anders. Dat is de tweede stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een recept voor brood werkt ook voor broodjes: dezelfde stappen, kneden, laten rijzen, bakken, met andere hoeveelheden en een andere baktijd. Je leert niet opnieuw bakken.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "change",
        question="Je `fit` trainde lineaire regressie met de hellingen van de MSE. Wat moet er volgens jou veranderen om in plaats daarvan logistische regressie te trainen?",
        options={
            "all": "Bijna alles",
            "slopes": "Alleen waar de hellingen vandaan komen",
            "nothing": "Helemaal niets",
        },
        reveal="**Alleen waar de hellingen vandaan komen.** De loop begint met nullen, haalt beide hellingen en stapt ertegenin, `steps` keer: dat is gradient descent, welk model het ook is. Bij logistische regressie zijn de hellingen die van de log loss: de kansen door de sigmoid, de verschillen $\\hat{y} - y$, en hun gemiddelden. Oefening 2 schrijft die functie; oefening 3 zet hem in de loop.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het maakt gradient descent niet uit welk model hij traint: hij heeft de hellingen van de loss voor elke parameter nodig, en stapt daartegenin. Bij logistische regressie op $n$ voorbeelden tegelijk zijn de hellingen van de log loss de gemiddelden van $(\hat{y} - y)\,x_j$ en van $\hat{y} - y$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_i]] - y_i) \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_i]] - y_i)"),
        mo.md(r"""
        De som over $i$ vermenigvuldigt elk verschil $\hat{y}_i - y_i$ met het getal op dezelfde plek in kolom $j$ van $\mathbf{X}$ en telt op: een inproduct. Voor alle weights tegelijk is dat `(p - y) @ X`, net als `r @ X` bij lineaire regressie. Naast elkaar:

        | | Lineaire regressie | Logistische regressie |
        |---|---|---|
        | voorspelling | `X @ w + b` | `1 / (1 + np.exp(-(X @ w + b)))` |
        | loss | mean squared error | log loss |
        | hellingen voor `w` | `-2 * (r @ X) / n`, met `r = y - y_pred` | `(p - y) @ X / n` |
        | helling voor `b` | `-2 * r.mean()` | `(p - y).mean()` |
        | één stap | `w = w - lr * grad_w`, `b = b - lr * grad_b` | hetzelfde |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **trainen met gradient descent** | | beginnen bij nullen, en steeds weer tegen de hellingen van de loss in stappen, tot de loss laag is: voor elk model waarvan je de hellingen kunt uitrekenen | `for step in range(steps):` |
        """),
        L.watch_out("Let op het teken. Bij log loss begint de gradient bij $\\hat{y} - y$, voorspelling min label, zonder factor ervoor. In je lineaire regressie was het residu $y - \\hat{y}$, label min voorspelling, met $-2$ ervoor. Haal je die door elkaar, dan loopt gradient descent bergop."),
        mo.Html(C.curve_picture({"title": "De log loss van de leerlingen tijdens het trainen", "x": "stap", "y": "log loss",
                                 "alt": "De log loss van de zeven leerlingen tijdens 1000 stappen gradient descent met een learning rate van 0,1: hij begint bij 0,69, daalt snel in de eerste honderd stappen, en kruipt daarna naar 0."})),
        mo.md(r"""
        Zo traint de course zelf op de zeven leerlingen van [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl), met $\eta$ = 0,1. De <span class="c-loss">log loss</span> begint bij 0,69, de loss van een kans van 0,5 voor iedereen, en daalt. Na oefening 3 traint de demo je eigen `fit` op dezelfde manier.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén stap gradient descent voor logistische regressie:

    1. Voorspel een kans voor alle voorbeelden tegelijk: `p = 1 / (1 + np.exp(-(X @ w + b)))`.
    2. Reken de verschillen tussen kans en label uit: `diff = p - y`.
    3. Reken elke helling uit: `grad_w = (diff @ X) / n` en `grad_b = diff.mean()`.
    4. Stap tegen de hellingen in, `w = w - lr * grad_w` en `b = b - lr * grad_b`, en herhaal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De zeven leerlingen: uren leren en uren slaap in `X`, of ze slaagden in `y`. Eén stap vanaf $\mathbf{w} = 0$ en $b = 0$, met $\eta$ = 0,1, en de log loss ervoor en erna. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_step():
    import numpy as np

    _X = np.array(C.STUDENTS)
    _y = np.array(C.PASSED)
    _w = np.zeros(2)
    _b = 0.0

    def _loss(w, b):
        _p = 1 / (1 + np.exp(-(_X @ w + b)))
        return -np.mean(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

    # 1. een kans voor elke leerling
    _p = 1 / (1 + np.exp(-(_X @ _w + _b)))
    print(_p)
    # 2. de verschillen tussen kans en label
    _diff = _p - _y
    print(_diff)
    # 3. elke helling tegelijk
    _grad_w = (_diff @ _X) / len(_y)
    _grad_b = _diff.mean()
    print(_grad_w, _grad_b)
    # 4. één stap, en de loss ervoor en erna
    _lr = 0.1
    print(_w - _lr * _grad_w, _b - _lr * _grad_b)
    print(_loss(_w, _b), _loss(_w - _lr * _grad_w, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met alle weights 0 is elke logit 0 en elke kans 0,5: het model weet nog niets. De verschillen zijn 0,5 voor de drie leerlingen die zakten en −0,5 voor de vier die slaagden. De hellingen zijn ongeveer −1,57 voor de uren leren, −0,71 voor slaap en −0,07 voor $b$: allemaal negatief, dus gaan beide weights en $b$ omhoog. Eén stap brengt de log loss van 0,69 naar 0,60. In oefening 3 herhaalt je `fit` deze stap in een loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén stap met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_step_cell():
    choice_step = L.choice(
        "one-step",
        question="De weights zijn $\\mathbf{w} = (1; 1)$, de learning rate is 0,5, en de hellingen voor de weights zijn $(-0{,}4;\\ 0{,}2)$. Wat zijn de weights na één stap?",
        options={
            "uphill": "$(0{,}8;\\ 1{,}1)$",
            "no-lr": "$(1{,}4;\\ 0{,}8)$",
            "right": "$(1{,}2;\\ 0{,}9)$",
        },
        correct="right",
        why={
            "uphill": "Dat stapt met de hellingen mee, bergop. Welke kant gaat een stap van gradient descent op? Zie stap 4 van het recept.",
            "no-lr": "Dat stapt met de hele helling. Wat doet de learning rate? Zie stap 4 van het recept.",
            "right": "$1 - 0{,}5 \\cdot (-0{,}4) = 1{,}2$ en $1 - 0{,}5 \\cdot 0{,}2 = 0{,}9$: tegen elke helling in, met de helft ervan.",
        },
    )
    choice_step
    return (choice_step,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De hellingen van log loss

        Maak `log_loss_gradients(X, y, w, b)` af. Hij krijgt de voorbeelden `X`, de labels `y`, 1 of 0, de weights `w` en de bias term `b`, en geeft de hellingen van de log loss terug: `grad_w`, één per weight, en `grad_b`. Vervang de `...` door stap 1 tot en met 3 van het recept. De check roept `log_loss_gradients([[2, 1], [1, 3], [0, 1]], [1, 0, 1], [0, 0], 0)` aan: dat moet ongeveer `([-0.1667, 0.1667], -0.1667)` geven. Hij probeert ook andere voorbeelden, weights en bias terms. Als je functie klopt, bewaart de check hem in je project, als `work/projects/nn/logistic_gradients.py`.
        """),
        L.task("Laat `log_loss_gradients` de hellingen van de log loss teruggeven voor elke weight en voor `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("log-loss-gradients", lines=11)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("log-loss-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "log-loss-gradients",
        [
            "Stap 1 tot en met 3 van het recept, in dezelfde volgorde: de kansen, de verschillen, en dan de twee hellingen.",
            "De verschillen zijn de kansen min de labels. In welke volgorde vermenigvuldig je de verschillen en `X`, en waardoor deel je?",
            "`p = 1 / (1 + np.exp(-(X @ w + b)))`, `diff = p - y`, `grad_w = (___) / len(y)` en `grad_b = ___`.",
        ],
        why="Met `w` en `b` allemaal 0 is elke kans 0,5, dus de verschillen zijn `[-0.5, 0.5, -0.5]`. `diff @ X` is `[-0.5, 0.5]`; gedeeld door 3 is dat ongeveer `[-0.1667, 0.1667]`, en het gemiddelde van de verschillen is ongeveer −0,1667.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen

        Nu traint je project een classifier. `fit(X, y, lr, steps)` haalt je `log_loss_gradients` uit je project binnen, begint met alle weights 0 en $b$ = 0, en draait zijn loop `steps` keer: hetzelfde als je `fit` voor lineaire regressie. Vervang de `...` door één update-stap: haal beide hellingen met `logistic_gradients.log_loss_gradients(X, y, w, b)`, en schuif `w` en `b` ertegenin. De check roept `fit([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 0, 1, 0], 0.5, 200)` aan: daar houdt alleen de eerste feature de klassen uit elkaar. Hij probeert ook andere data, learning rates en aantallen stappen. Als je functie klopt, bewaart de check hem in je project, als `work/projects/nn/logistic.py`.
        """),
        L.task("Laat `fit` `w` en `b` trainen met je eigen `log_loss_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-logistic", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # check again when exercise 2 saves your module
    result_fit = L.check("fit-logistic", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-logistic",
        [
            "Stap 4 van het recept, met de hellingen uit je eigen functie: dezelfde loop als in je `fit` voor lineaire regressie.",
            "De functie woont in de module `logistic_gradients`: roep hem aan via de module. Hij geeft twee dingen tegelijk terug: hoe geef je elk een eigen naam?",
            "`grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)`, dan `w = w - ___` en `b = b - ___`.",
        ],
        why="`logistic_gradients.log_loss_gradients(X, y, w, b)` geeft beide hellingen; `grad_w, grad_b = ...` vangt ze op. `w = w - lr * grad_w` en `b = b - lr * grad_b` stappen ertegenin. Na 200 rondes op het voorbeeld is `w` ongeveer `[5.89, -0.42]` en `b` ongeveer −2,52.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je eigen `fit` om logistische regressie te trainen op de zeven leerlingen, met $\eta$ = 0,1 en 1000 stappen, en tekent de decision boundary. Hij gebruikt je eigen module, dus hij werkt zodra oefening 3 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_students(result_fit):
    result_fit  # run again when exercise 3 saves your module
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Maak eerst oefening 3 af: dan staat je `fit` in je project, en gebruikt deze cel hem.*")
    elif _model == "gradients":
        _out = mo.md("*Je `fit` haalt zijn hellingen uit `log_loss_gradients` in `work/projects/nn/logistic_gradients.py`, en dat bestand ontbreekt, of geeft niet meer de goede hellingen. Laat oefening 2 opnieuw nakijken; heb je dat bestand veranderd, verwijder het dan eerst.*")
    elif not _model:
        _out = mo.md("*De `fit` in `work/projects/nn/logistic.py` traint niet meer goed, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    else:
        _w, _b = _model
        _out = mo.vstack([
            mo.md(f"""Je model leerde de weights <span class="c-param ea-num">{L.number(_w[0], 2)}</span> voor de uren leren en <span class="c-param ea-num">{L.number(_w[1], 2)}</span> voor de uren slaap, en de bias term <span class="c-param ea-num">{L.number(_b, 2)}</span>. Het zet <span class="ea-num">{C.wrong_side(_w, _b)}</span> van de 7 leerlingen aan de verkeerde kant van zijn decision boundary."""),
            mo.Html(C.demo_picture(_w, _b, {"title": "Je model op de zeven leerlingen", "hours": "uren leren", "sleep": "uren slaap",
                                            "alt": "De zeven leerlingen naar uren leren en slaap, vier gevulde punten die slaagden en drie open punten die zakten, met de decision boundary van je getrainde model als steile blauwe lijn ertussen."})),
            mo.md("""De gevulde stippen slaagden, de open stippen zakten. Kijk naar de weight voor slaap: die is negatief, terwijl de leerlingen die slaagden juist iets meer sliepen. Over slaap zegt dat weinig. Elke leerling slaapt 5 tot 8 uur, dus een negatieve weight voor slaap verlaagt elke logit met ongeveer hetzelfde, net als een negatieve bias term. Met 3,57 per uur leren heeft elke logit zo'n flink negatief deel nodig, zodat de leerlingen met weinig uren leren onder 0 uitkomen. Gradient descent haalt dat vooral uit slaap: de helling voor die weight vermenigvuldigt elk verschil eerst met 5 tot 8 uur slaap en neemt dan het gemiddelde, en de helling voor b is alleen het gemiddelde van de verschillen. Na 1000 stappen is de weight voor slaap keer 7 uur ongeveer −12, en b maar −0,55. En op data die het perfect kan splitsen, zoals deze zeven leerlingen, blijven de weights groeien hoe langer het traint, terwijl de log loss naar 0 kruipt."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl) leidde je de hellingen $(\hat{y} - y)\,x$ af; in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) schreef je de loop die tegen hellingen in stapt. Samen trainen ze een classifier, en je project `nn` heeft nu `logistic_gradients.py` en `logistic.py` naast `gradients.py` en `linear.py`. In de volgende les, *One-hot-vectoren voor klassen*, schrijf je een klasse als vector, de eerste stap naar meer dan twee klassen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_step, result_fit, result_gradients):
    L.wrap_up(choice_step, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
