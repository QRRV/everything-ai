# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-micrograd-mlp. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-mlp", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl) en [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_backward():
    quick_backward = L.choice(
        "quick-backward",
        question="Met je `Value` en je `backward`: `a = Value(3)`, `b = Value(4)`, `c = a * b + a`, dan `backward(c)`. Wat is `a.grad`?",
        options={
            "right": "5",
            "one": "4",
            "two": "1",
        },
        correct="right",
        why={
            "one": "Dat is wat er alleen via `a * b` terugkomt. `a` gaat ook rechtstreeks in de som. Zie het let-op in [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl).",
            "two": "Dat is wat er terugkomt langs de pijl rechtstreeks in de som. `a` gaat ook in `a * b`. Zie het let-op in [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl).",
            "right": "4 terug via `a * b`, het getal van `b`, en 1 langs de pijl rechtstreeks in de som: samen 5, de helling van $ab + a$ voor $a$.",
        },
    )
    quick_backward
    return (quick_backward,)


@app.cell(hide_code=True)
def quick_check_inherit():
    quick_inherit = L.choice(
        "quick-inherit",
        question="`class Pet:` heeft een method `name_tag(self)`, en `class Dog(Pet):` niet. Wat doet `Dog().name_tag()`?",
        options={
            "error": "Python stopt: `Dog` heeft geen `name_tag`",
            "right": "Het voert de `name_tag` van `Pet` uit",
            "empty": "Het geeft niets terug, omdat `Dog` geen eigen method heeft",
        },
        correct="right",
        why={
            "error": "Python kijkt eerst in de eigen class van het object, en dan waar? Zie *Het idee* in [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl).",
            "empty": "Een subclass krijgt alle methods van zijn parent class. Zie *Het idee* in [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl).",
            "right": "`Dog` is gebouwd op `Pet`, dus krijgt het zijn methods: Python vindt `name_tag` in `Pet`.",
        },
    )
    quick_inherit
    return (quick_inherit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl) bouwde de demo een netwerk met een hidden layer uit losse `Value`s: 17 parameters, elk met een eigen plek in een lijst. Een netwerk voor echte data heeft er duizenden, in layers van verschillende grootte. Dat met de hand uitschrijven is niet meer te doen. Een netwerk is gebouwd uit een paar soorten onderdelen die terugkomen: neuronen, layers van neuronen, en layers na elkaar. In deze les schrijf je die onderdelen als classes in je project. Dat is de zesde stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een huis is gebouwd uit bakstenen, de bakstenen vormen muren, en de muren vormen kamers. Weet je hoe je één steen legt, dan kun je een huis van elke grootte bouwen.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="Een neuron met 3 inputs heeft 3 weights en een bias term. Een layer van 4 van zulke neuronen krijgt dezelfde 3 inputs. Hoeveel parameters heeft de layer volgens jou?",
        options={
            "weights": "12",
            "right": "16",
            "neurons": "7",
        },
        reveal="**16.** Elk van de 4 neuronen heeft zijn eigen 3 weights en zijn eigen bias term: 4 × (3 + 1) = 16. In een layer ziet elk neuron elke input, dus een layer van $m$ neuronen met $n$ inputs heeft $m \\cdot (n + 1)$ parameters: de $m \\times n$ weights van zijn weight-matrix, en $m$ bias terms.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Drie classes, elk gebouwd op de vorige:

        - Een `Neuron` heeft een lijst weights `w`, één `Value` per input, en een bias term `b`. Aangeroepen op de inputs `x`, een lijst `Value`s, rekent het $z = b + w_1 x_1 + w_2 x_2 + \dots$ uit, en geeft het $\mathrm{ReLU}(z)$ terug, of $z$ zelf in de output layer.
        - Een `Layer` heeft een lijst neuronen die allemaal dezelfde inputs krijgen. Aangeroepen op `x` geeft hij de lijst van hun outputs terug: de inputs van de volgende layer.
        - Een `MLP` heeft een lijst layers. Aangeroepen op `x` geeft het de outputs van elke layer door aan de volgende, en geeft hij die van de laatste terug.

        Alle drie hebben een method `parameters()` die elke `Value` teruggeeft die trainen mag veranderen: een neuron zijn weights en bias term, een layer die van al zijn neuronen, een MLP dat van al zijn layers. Ze bouwen allemaal op één kleine class, `Module`, waarvan `parameters()` een lege lijst teruggeeft. De weights beginnen bij willekeurige getallen tussen −1 en 1, zodat de neuronen van een layer verschillend beginnen; de bias terms beginnen bij 0.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | `Neuron`, `Layer`, `MLP` | | classes voor één neuron, een layer van neuronen, en layers na elkaar | `MLP(2, [4, 4, 1])` |
        | `parameters()` | | een method die elke `Value` teruggeeft die trainen mag veranderen | `m.parameters()` |
        | `Module` | | de class waar alle drie op bouwen; zijn `parameters()` geeft `[]` terug | `class Neuron(Module):` |
        | `random.uniform(a, b)` | | een willekeurig getal met decimalen tussen `a` en `b` | `random.uniform(-1, 1)` |
        """),
        L.watch_out("Begint elke weight bij 0, dan is elk neuron van een layer hetzelfde: dezelfde output, dezelfde helling, dezelfde stap, voor altijd. Willekeurige beginwaarden maken ze verschillend, zodat elk iets eigens kan leren."),
        L.watch_out("`Module` is hier de naam van een class: de class waar de andere op bouwen. Een Python-module is nog steeds een bestand met code, zoals `neuron.py`; dat bestand bevat de class `Module`."),
        mo.md(r"""
        `MLP(2, [4, 4, 1])`: 2 inputs, twee hidden layers van 4 neuronen, en één output-neuron. Elke lijn is een weight, en elke cirkel na de inputs heeft ook een eigen bias term.
        """),
        draw.network_diagram([2, 4, 4, 1], ["2 inputs", "4 neuronen, ReLU", "4 neuronen, ReLU", "1 output"],
                             title="MLP(2, [4, 4, 1])",
                             alt="Vier kolommen cirkels: 2 inputs, twee hidden layers van 4 neuronen en 1 output, met een lijn van elke cirkel naar elke cirkel van de volgende kolom."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een netwerk uit je `Value`:

    1. Een neuron: `z` begint bij `b`; tel `w[i] * x[i]` op voor elke input; geef `z.relu()` terug, of `z` in de output layer.
    2. Een layer: roep elk van zijn neuronen aan op dezelfde `x`, en geef de lijst van hun outputs terug.
    3. Een MLP: stuur `x` door elke layer om de beurt; de output van de ene is de input van de volgende.
    4. `parameters()`: zet de parameters van de onderdelen bij elkaar in één lijst, met `+` tussen lijsten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De parameters van `MLP(2, [4, 4, 1])` tellen, layer voor layer, zoals `parameters()` dat straks doet. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_count():
    _sizes = [2, 4, 4, 1]
    _n_inputs = _sizes[0]
    _total = 0
    for _i in range(1, len(_sizes)):
        _n_neurons = _sizes[_i]
        # 1. één neuron: een weight per input, en een bias term
        _per_neuron = _n_inputs + 1
        # 2. een layer: dat, voor elk van zijn neuronen
        _per_layer = _n_neurons * _per_neuron
        print(f"layer {_i}: {_n_neurons} × {_per_neuron} = {_per_layer}")
        # 3. het MLP: de outputs van deze layer zijn de inputs van de volgende
        _n_inputs = _n_neurons
        # 4. parameters(): die van alle layers samen
        _total = _total + _per_layer
    print("samen", _total)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    12 + 20 + 5 = 37 parameters. Stap 3 is de regel `_n_inputs = _n_neurons`: de inputs van een layer zijn de outputs van de layer ervoor, dus de tweede hidden layer heeft 4 inputs, niet 2. Je `MLP` maakt straks deze 37 `Value`s, en `backward` geeft elk zijn helling.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel parameters?
    """)
    return


@app.cell(hide_code=True)
def choice_size_cell():
    choice_size = L.choice(
        "size",
        question="Hoeveel parameters heeft `MLP(3, [4, 4, 1])`: 3 inputs, twee hidden layers van 4 neuronen, en één output-neuron?",
        options={
            "weights": "32",
            "neurons": "9",
            "right": "41",
        },
        correct="right",
        why={
            "weights": "Dat zijn de weights: 12, 16 en 4. Elk neuron heeft ook een bias term. Zie het uitgewerkte voorbeeld.",
            "neurons": "Dat is het aantal neuronen. Elk heeft een weight per input, en een bias term. Zie het uitgewerkte voorbeeld.",
            "right": "4 × (3 + 1) = 16, dan 4 × (4 + 1) = 20, dan 1 × (4 + 1) = 5: 41.",
        },
    )
    choice_size
    return (choice_size,)


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je Neuron

        Maak de class `Neuron` af. `Module` en `__init__` zijn klaar: `self.w` is de lijst weights, `self.b` de bias term, en `self.relu` is `True` of `False`. `relu=True` in de `def`-regel geeft `relu` een waarde voor als een aanroep hem weglaat: `Neuron(3)` krijgt ReLU, `Neuron(3, relu=False)` niet. Vervang de twee `...`: `__call__` doet stap 1 van het recept op de inputs `x`, een lijst van je `Value`s; `parameters()` geeft de weights en de bias term terug in één lijst. De check maakt neuronen, zet hun weights, roept ze aan, en laat je `backward` lopen op wat eruit komt. Klopt het, dan bewaart de check het vak in je project, als `work/projects/nn/neuron.py`.
        """),
        L.task("Laat een `Neuron` zijn output uitrekenen, en zijn parameters teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=27)
    box_neuron
    return (box_neuron,)


@app.cell(hide_code=True)
def check_neuron(box_neuron):
    result_neuron = L.check("neuron", box_neuron.value)
    result_neuron
    return (result_neuron,)


@app.cell(hide_code=True)
def hints_neuron():
    L.hints(
        "neuron",
        [
            "Stap 1 van het recept: een loop over de inputs, zoals in de forward pass van je vorige lessen, maar met de eigen `self.w` en `self.b` van het neuron.",
            "Laat `z` beginnen bij de eigen bias term van het neuron, en tel in een loop over `range(len(self.w))` elke ronde één weight keer zijn input erbij op. `self.relu` zegt of ReLU erna komt. `parameters()` zet de lijst `self.w` en een lijst met `self.b` bij elkaar.",
            "`z = ___`, dan `for i in range(len(self.w)):` met `z = z + ___ * ___`; dan `if self.relu:` met `return z.relu()`, en `return z`. In `parameters()`: `return self.w + [___]`.",
        ],
        why="Met de weights 0,5 en −1, de bias term 1 en de inputs 2 en 1: $z = 1 + 0{,}5 \\cdot 2 - 1 \\cdot 1 = 1$, boven 0, dus ReLU geeft 1 door. Na `backward` hebben de weights de hellingen 2 en 1, hun inputs, en de bias term 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Layers en een MLP

        Het vak haalt je `Module` en je `Neuron` uit je project binnen. `Layer` maakt zijn neuronen en geeft hun parameters terug; `MLP` maakt zijn layers, de laatste zonder ReLU. Vervang de drie `...`: `Layer.__call__` geeft de lijst van de outputs van zijn neuronen terug, `MLP.__call__` stuurt `x` door elke layer, en `MLP.parameters()` zet de parameters van alle layers bij elkaar, zoals `Layer.parameters()` dat doet voor zijn neuronen. De check bouwt MLP's van verschillende grootte, roept ze aan en laat je `backward` lopen. Klopt het, dan bewaart de check het vak in je project, als `work/projects/nn/mlp.py`.
        """),
        L.task("Laat een `Layer` en een `MLP` hun outputs uitrekenen, en laat een `MLP` al zijn parameters teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("mlp", lines=34)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp, result_neuron):
    result_neuron  # check again when exercise 2 saves your module
    result_mlp = L.check("mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "mlp",
        [
            "Stap 2, 3 en 4 van het recept. `Layer.parameters()` laat zien hoe je lijsten bij elkaar zet.",
            "Roep elk neuron aan als een functie, op dezelfde lijst `x`, en verzamel wat eruit komt in een lijst. In `MLP.__call__` krijgt de naam `x` in elke ronde van een loop over `self.layers` een nieuwe waarde: wat die layer ervan maakt.",
            "`Layer.__call__`: `outputs = []`, dan voor elke `n` in `self.neurons`: `outputs.append(___)`, en `return outputs`. `MLP.__call__`: `for layer in self.layers:` met `x = ___`, en `return x`.",
        ],
        why="Met elke parameter op 0,5 geeft `MLP(2, [2, 1])` op de inputs 1 en 2 elk hidden neuron $0{,}5 + 0{,}5 + 1 = 2$, wat ReLU doorgeeft, en het output-neuron $0{,}5 + 0{,}5 \\cdot 2 + 0{,}5 \\cdot 2 = 2{,}5$. Het heeft $2 \\cdot 3 + 1 \\cdot 3 = 9$ parameters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder bouwt `MLP(2, [4, 4, 1])` met je eigen classes, elke keer met dezelfde willekeurige beginwaarden, en roept het aan op de vier hoeken van XOR. Daarna bouwt hij de loss over de vier hoeken en roept hij je `backward` één keer aan. De cel gebruikt je eigen Python-modules, dus hij werkt zodra oefening 3 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_mlp(result_mlp):
    result_mlp  # run again when exercise 3 saves your module
    _out = C.demo()
    if _out is None:
        _view = mo.md("*Maak eerst oefening 3 af: dan staat je `MLP` in je project, en gebruikt deze cel hem.*")
    elif not _out:
        _view = mo.md("*Je `neuron.py` of `mlp.py` in `work/projects/nn/` werkt niet meer zoals zijn check verwacht. Heb je een van beide veranderd? Verwijder het, en laat zijn oefening opnieuw nakijken: die check bewaart het dan opnieuw.*")
    else:
        _n, _outputs, _loss, _with_slope, _own = _out
        _view = mo.md(f"""Je MLP heeft <span class="ea-num">{_n}</span> parameters. Ongetraind geeft het <span class="c-pred ea-num">{"; ".join(L.number(_o, 2) for _o in _outputs)}</span> voor (0, 0), (0, 1), (1, 0) en (1, 1), waar XOR 0, 1, 1 en 0 wil: de loss is <span class="c-loss ea-num">{L.number(_loss, 2)}</span>. Eén aanroep van {"je `backward`" if _own else "de `backward` van de course"} gaf <span class="ea-num">{_with_slope}</span> van de {_n} parameters een helling die geen 0 is. De andere zijn weights waar niets bij terugkomt: bij elke hoek is hun input 0, of geeft de ReLU van hun neuron daar 0. In de volgende les zet je stappen tegen deze hellingen in, steeds weer: trainen.""")
    _view
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) rekende een netwerk neuron voor neuron uit; nu doen classes die op elkaar voortbouwen, zoals in [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl), dat voor een netwerk van elke grootte, bovenop de `Value` en `backward` van [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl). Je project `nn` heeft nu ook `neuron.py` en `mlp.py`. In de volgende les, *Je MLP trainen: de training loop*, train je zo'n MLP: forward, loss, backward, stap, en opnieuw.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_size, result_neuron, result_mlp):
    L.wrap_up(choice_size, result_neuron, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
