# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-micrograd-train. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-train", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Neuronen, layers en een MLP bovenop je Value](/course/open/nn-micrograd-mlp?lang=nl) en [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_output():
    quick_output = L.choice(
        "quick-output",
        question="`m = MLP(2, [3, 1])`: 2 inputs, een hidden layer van 3 neuronen, en één output-neuron. Wat geeft `m([Value(1), Value(2)])` terug?",
        options={
            "right": "Een lijst met één `Value`",
            "hidden": "Een lijst met 3 `Value`s, één per hidden neuron",
            "one": "Eén `Value`, zonder lijst eromheen",
        },
        correct="right",
        why={
            "hidden": "Dat geeft de hidden layer. Het MLP geeft dat door aan de volgende layer, en geeft terug wat zijn laatste layer geeft. Zie [Neuronen, layers en een MLP bovenop je Value](/course/open/nn-micrograd-mlp?lang=nl).",
            "one": "Een layer geeft een lijst terug, één `Value` per neuron, ook als hij maar één neuron heeft, en het MLP geeft terug wat zijn laatste layer geeft. Zie [Neuronen, layers en een MLP bovenop je Value](/course/open/nn-micrograd-mlp?lang=nl).",
            "right": "Het MLP geeft terug wat zijn laatste layer geeft: een lijst met één `Value` per neuron. Met één output-neuron is dat een lijst met één `Value`, en `[0]` haalt die eruit.",
        },
    )
    quick_output
    return (quick_output,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_slopes = L.choice(
        "quick-slopes",
        question="Eén stap van gradient descent met twee parameters, $w$ en $b$. Welke hellingen gebruikt hij?",
        options={
            "after": "De helling van $w$; daarna, als $w$ geschoven is, de helling van $b$ bij de nieuwe $w$",
            "larger": "Alleen de grootste van de twee hellingen; de andere parameter wacht op zijn beurt",
            "right": "Beide hellingen, gemeten bij dezelfde $w$ en $b$, voordat er een schuift",
        },
        correct="right",
        why={
            "after": "Dan hoort de helling van $b$ bij een ander punt dan die van $w$. Zie het recept in [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).",
            "larger": "Elke parameter schuift in elke stap tegen zijn eigen helling in. Zie [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).",
            "right": "Beide hellingen horen bij hetzelfde punt, dus samen wijzen ze vanaf daar de weg naar beneden. Pas daarna schuiven $w$ en $b$ allebei.",
        },
    )
    quick_slopes
    return (quick_slopes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je project kan nu een MLP van elke grootte bouwen en elk van zijn parameters een helling geven. Wat het nog niet kan, is leren. Leren is een loop: voorspellen, de loss meten, de hellingen halen, een kleine stap ertegenin zetten, en opnieuw. Die loop traint elk netwerk in deze course, tot en met het language model aan het eind, en PyTorch schrijft hem in dezelfde stappen. In deze les schrijf je hem, in je project. Dat is de zevende stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           6, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Leren fietsen: je zet af, je wiebelt, je leunt een beetje de andere kant op, en je probeert het opnieuw. Elke poging vraagt maar een kleine correctie; met veel van die kleine correcties leer je fietsen.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_pile():
    guess_pile = L.guess(
        "pile",
        question="Elke stap van de loop roept `backward(loss)` aan, die elke parameter een `grad` geeft. Wat gebeurt er volgens jou als niets de `grad`s tussen de stappen terug op 0 zet?",
        options={
            "overwrite": "Niets: elke backward pass schrijft nieuwe hellingen over de oude heen",
            "right": "Elke stap schuift met de hellingen van alle stappen tot nu toe, opgeteld",
            "error": "Python stopt met een fout bij de tweede stap",
        },
        reveal="**Ze stapelen op.** `backward_step()` telt op bij wat `grad` al heeft, dus na tien stappen bevat elke `grad` tien hellingen, van tien verschillende plekken. Elke stap schuift dan met al die hellingen samen, steeds verder, en in een richting die niet meer de helling is waar je bent. Daarom zet de loop elke `grad` op 0 vóór elke backward pass.",
    )
    guess_pile
    return (guess_pile,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **training loop** herhaalt dezelfde stappen, vele keren:

        1. **Forward**: de voorspelling van het model voor elk voorbeeld, en daaruit de loss, hier de mean squared error.
        2. **Gradients op nul**: zet de `grad` van elke parameter op 0, omdat `backward_step()` erbij optelt.
        3. **Backward**: `backward(loss)` geeft elke parameter zijn helling.
        4. **Stap**: schuif elke parameter tegen zijn helling in, met de learning rate $\eta$ keer die helling:
        """),
        L.formula(r"[[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]] \quad \text{voor elke parameter } [[param:w]]"),
        mo.md(r"""
        Je `Value` kan niet aftrekken of delen, maar wel optellen en vermenigvuldigen. Het residu $y - \hat{y}$ telt alleen gekwadrateerd mee, en $(\hat{y} - y)^2$ is hetzelfde getal, dus `mse` werkt met $\hat{y} - y$: `y_pred + Value(-y)`. Het gemiddelde is de som keer `Value(1 / n)`. De loss van elke stap komt in een lijst, zodat je hem ziet dalen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **training loop** | | de stappen die een model trainen, steeds weer: forward en loss, gradients op nul, backward, stap | `for step in range(steps):` |
        | **gradients op nul zetten** | | de `grad` van elke parameter op 0 zetten vóór de backward pass, omdat de hellingen optellen | `p.grad = 0` |
        | **stap** | | elke parameter tegen zijn helling in schuiven: `p.data - lr * p.grad` | `p.data = p.data - lr * p.grad` |
        | `mse(model, xs, ys)` | | de mean squared error van een model op de voorbeelden `xs` met labels `ys`, als `Value` | `loss = mse(m, xs, ys)` |
        """),
        L.watch_out("Zet de gradients op nul vóór `backward(loss)`, niet erna. Zet je ze na de backward pass en vóór de stap op 0, dan is elke helling 0 als de stap komt, en verandert het model nooit."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek van de loss: dat zet het aantal stappen dat een `MLP(1, [8, 1])` getraind heeft op 12 punten die op een kromme liggen, met $\eta$ = 0,2. De ene grafiek toont de punten en <span class="c-pred">wat het model voorspelt</span>; de andere de <span class="c-loss">loss</span> van elke stap, met een punt bij het aantal stappen dat je koos.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(steps):
        steps = int(steps)
        losses, grid, shots = C.snapshots()
        said = (f"""Na <span class="ea-num">{steps}</span> stappen is het model nog ongetraind: een lijn uit 8 ReLU-neuronen met willekeurige weights. De loss is <span class="c-loss ea-num">{L.number(losses[0], 4)}</span>.""" if steps == 0 else
                f"""Na <span class="ea-num">{steps}</span> stappen is de loss <span class="c-loss ea-num">{L.number(losses[steps], 4)}</span>. Elk ReLU-neuron geeft de lijn een knik; de loop schuift de knikken naar waar de punten ze nodig hebben.""")
        return (C.picture(steps, {"fit": "De punten en het model", "loss": "De loss van elke stap", "step": "stap", "loss_y": "loss",
                                  "alt": f"12 punten op een kromme die in het midden het laagst is, en de lijn van een MLP na {steps} stappen training. Een tweede grafiek toont de loss van elke stap, eerst snel dalend en daarna langzaam, van {L.number(losses[0], 2)} naar {L.number(losses[-1], 4)}" + (f", met een punt bij stap {steps}" if steps else "") + "."},
                          losses=losses, grid=grid, curve=shots[steps], drag=1),
                said)

    explore_steps = L.explore("steps", "stappen", _frame, start=0, low=0, high=300, step=25, role="", task="Na hoeveel stappen is de loss onder 0,01? En hoeveel daalt hij nog in de laatste 150 stappen?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Na 50 stappen is de loss onder 0,01: 0,0099, van 0,3539 aan het begin. In de laatste 150 stappen gaat hij nog maar van 0,0028 naar 0,0018: de knikken liggen bijna waar de punten ze nodig hebben, dus de hellingen zijn klein, en elke stap ook.

    ### Recept

    Zo train je een model met je project:

    1. Forward: de loss van het model op alle voorbeelden, `loss = mse(model, xs, ys)`.
    2. Gradients op nul: `p.grad = 0` voor elke `p` in `model.parameters()`.
    3. Backward: `backward(loss)`.
    4. Stap: `p.data = p.data - lr * p.grad` voor elke parameter.
    5. Bewaar `loss.data`, en ga terug naar stap 1, `steps` keer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Dezelfde loop zonder je project: een lijn $\hat{y} = w \cdot x + b$ op drie punten, met de hellingen van de mean squared error uit de kettingregel, zoals in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl), maar met `_r` als $\hat{y} - y$, dus zonder het minteken ervoor. De hellingen zijn sommen over de voorbeelden, dus ze beginnen in elke stap bij 0. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_loop():
    _xs = [1, 2, 3]
    _ys = [2, 4, 6]
    _w = 0.0
    _b = 0.0
    _lr = 0.1
    _n = len(_xs)
    _losses = []
    for _step in range(6):
        # 1. forward: de mean squared error
        _loss = 0
        for _k in range(_n):
            _r = _w * _xs[_k] + _b - _ys[_k]
            _loss = _loss + _r * _r / _n
        # 2. gradients op nul: de hellingen beginnen bij 0
        _grad_w = 0
        _grad_b = 0
        # 3. backward: tel de helling van elk voorbeeld op
        for _k in range(_n):
            _r = _w * _xs[_k] + _b - _ys[_k]
            _grad_w = _grad_w + 2 * _r * _xs[_k] / _n
            _grad_b = _grad_b + 2 * _r / _n
        # 4. stap: tegen de hellingen in
        _w = _w - _lr * _grad_w
        _b = _b - _lr * _grad_b
        # 5. bewaar de loss, en opnieuw
        _losses.append(_loss)
        print(_step, _loss, _w, _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De loss daalt in één stap van ongeveer 18,67 naar ongeveer 0,30, en daarna langzaam: $w$ en $b$ schuiven naar een lijn door de drie punten, $w = 2$ en $b = 0$. Stond `_grad_w = 0` vóór de loop in plaats van erin, dan zaten de hellingen van de eerste stap bij de tweede nog in `_grad_w`, en schoof elke stap met al die hellingen samen. In je project doet `backward` stap 3 voor elke parameter van een MLP tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. In welke volgorde?
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Welke volgorde van de stappen traint een model?",
        options={
            "late": "Forward en loss, backward, gradients op nul, stap",
            "first": "Gradients op nul, stap, forward en loss, backward",
            "right": "Forward en loss, gradients op nul, backward, stap",
        },
        correct="right",
        why={
            "late": "Wat zijn de grads als de stap komt? Zie het let-op in *Het idee*.",
            "first": "De stap komt vlak nadat de gradients op nul gingen: waarmee schuift hij dan? Zie het let-op in *Het idee*.",
            "right": "De grads zijn 0 als `backward` begint, dus elke bevat precies de helling van deze stap, en de stap gebruikt ze. De gradients op nul zetten mag ook vooraan, vóór de forward pass, zolang het vóór `backward` gebeurt.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_mse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss van een model

        Maak `mse(model, xs, ys)` af: de mean squared error van `model` op de voorbeelden `xs`, een lijst met één lijst getallen per voorbeeld, en de labels `ys`. Het vak haalt je `Value` binnen, en de loop maakt al `Value`s van de inputs van elk voorbeeld en geeft de voorspelling `y_pred`. Vervang de twee `...`: tel in de loop het kwadraat van het residu op bij `total`; geef na de loop het gemiddelde terug, als `Value`. De check roept hem aan op MLP's uit je project, en laat je `backward` lopen op wat hij geeft. Klopt het, dan bewaart de check hem in je project, als `work/projects/nn/loss.py`.
        """),
        L.task("Laat `mse` de mean squared error van een model teruggeven, als Value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mse():
    box_mse = L.code_box("mse", lines=14)
    box_mse
    return (box_mse,)


@app.cell(hide_code=True)
def check_mse(box_mse):
    result_mse = L.check("mse", box_mse.value)
    result_mse
    return (result_mse,)


@app.cell(hide_code=True)
def hints_mse():
    L.hints(
        "mse",
        [
            "Stap 1 van het recept, met je `Value`: voor elk voorbeeld $\\hat{y} - y$, het kwadraat daarvan, de som, en dan het gemiddelde.",
            "Je `Value` kan alleen optellen en vermenigvuldigen: $\\hat{y} - y$ is `y_pred` plus een `Value` van min het label, een kwadraat is een waarde keer zichzelf, en het gemiddelde is de som keer een `Value` van één gedeeld door het aantal voorbeelden.",
            "`r = y_pred + Value(-ys[k])`, dan `total = total + ___ * ___`; na de loop `return total * Value(___)`.",
        ],
        why="Met elke parameter van `MLP(1, [3, 1])` op 0,5 en het voorbeeld 1 met label 2: elk hidden neuron geeft $0{,}5 + 0{,}5 = 1$, de output $0{,}5 + 3 \\cdot 0{,}5 = 2$, dus het residu is 0 en de loss 0. Elk residu telt zijn kwadraat op, en de som keer $\\frac{1}{n}$ is het gemiddelde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je training loop

        Maak `train(model, xs, ys, lr, steps)` af. Hij haalt je `backward` en je `mse` uit je project binnen, rekent in elke stap de loss uit, en bewaart `loss.data` in `losses`. Vervang de `...` door stap 2, 3 en 4 van het recept. De check traint MLP's van verschillende grootte vanuit dezelfde beginwaarden als de course, en vergelijkt de losses van elke stap. Klopt het, dan bewaart de check hem in je project, als `work/projects/nn/train.py`.
        """),
        L.task("Laat `train` in elke ronde van de loop de gradients op nul zetten, de backward pass doen en een stap zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=14)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_mse):
    result_mse  # check again when exercise 2 saves your module
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Stap 2, 3 en 4 van het recept, in die volgorde, binnen de loop: het uitgewerkte voorbeeld doet hetzelfde met twee parameters.",
            "Stap 2 en 4 zijn elk een loop over `model.parameters()`; stap 3 is één aanroep.",
            "`for p in model.parameters():` met `p.grad = 0`; dan `backward(___)`; dan `for p in model.parameters():` met `p.data = p.data - ___ * ___`.",
        ],
        why="Eerst de gradients op nul, zodat `backward(loss)` elke `grad` alleen met de helling van deze stap vult; daarna schuift elke parameter ertegenin. De check vergelijkt de loss van elke stap met de eigen run van de course vanuit dezelfde beginwaarden.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder doet wat de figuur in *Het idee* doet, maar met je eigen code: hij bouwt `MLP(1, [8, 1])` vanuit dezelfde beginwaarden en traint het met je eigen `train`, 300 stappen met $\eta$ = 0,2. De cel gebruikt je eigen modules, dus hij werkt zodra oefening 3 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_train(result_train):
    result_train  # run again when exercise 3 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Maak eerst oefening 3 af: dan staat je `train` in je project, en gebruikt deze cel hem.*")
    elif not _fit:
        _out = mo.md("*Je `loss.py` of `train.py` in `work/projects/nn/` werkt niet meer zoals zijn check verwacht. Heb je een van beide veranderd? Verwijder het, en laat zijn oefening opnieuw nakijken: die check bewaart het dan opnieuw.*")
    else:
        _losses, _grid, _curve, _course_files = _fit
        _out = mo.vstack([
            mo.Html(C.picture(len(_losses), {"fit": "Je MLP na het trainen", "loss": "Je loss van elke stap", "step": "stap", "loss_y": "loss",
                                             "alt": f"12 punten op een kromme en de lijn van het MLP dat jouw functie train getraind heeft. Een tweede grafiek toont zijn loss van elke stap, van {L.number(_losses[0], 2)} naar {L.number(_losses[-1], 4)}."},
                              losses=_losses, grid=_grid, curve=_curve)),
            mo.md(f"""Je loop bracht de loss van <span class="c-loss ea-num">{L.number(_losses[0], 2)}</span> naar <span class="c-loss ea-num">{L.number(_losses[-1], 4)}</span>. {"Elk deel ervan is van jou: de `Value` en zijn hellingen, de backward pass, de neuronen en layers, de loss en de loop. Dat is een hele library voor neural networks, klein maar compleet." if not _course_files else "De loss en de loop zijn van jou. Voor deze bestanden gebruikte de run de versie van de course: " + ", ".join(_course_files) + "."}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl) zette stappen tegen de hellingen van twee parameters in; nu doet je training loop dat voor elke parameter van het MLP uit [Neuronen, layers en een MLP bovenop je Value](/course/open/nn-micrograd-mlp?lang=nl), met de hellingen van je eigen `backward`, op de mean squared error uit [Eén getal voor hoe ver je ernaast zit: de loss](/course/open/ml-mse-loss?lang=nl). Je project `nn` heeft nu ook `loss.py` en `train.py`. In de volgende les, *Tensors in PyTorch*, maak je kennis met PyTorch, dat dit allemaal voor hele arrays tegelijk doet, en veel sneller.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_mse, result_train):
    L.wrap_up(choice_order, result_mse, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
