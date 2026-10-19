# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-nn-module. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-nn-module", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) en [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="`w = torch.tensor([1.0, 2.0], requires_grad=True)`, `loss = (w * w).sum()`, en `loss.backward()`. Wat is `w.grad`?",
        options={
            "sum": "`tensor(6.)`",
            "right": "`tensor([2., 4.])`",
            "none": "`None`, omdat `loss` geen leaf tensor is",
        },
        correct="right",
        why={
            "sum": "`.grad` heeft de shape van `w`: één helling per weight. Zie *De backward pass* in [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "none": "`w` is hier de leaf tensor, gemaakt met `requires_grad=True`, dus hij krijgt een `.grad`. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "right": "De helling van $w_1^2 + w_2^2$ is $2w$ voor elke weight: 2 en 4, in de shape van `w`.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_call():
    quick_call = L.choice(
        "quick-call",
        question="Wat laat dit programma zien?\n\n```python\nclass Adder:\n    def __init__(self):\n        self.total = 0\n\n    def __call__(self, x):\n        self.total = self.total + x\n        return self.total\n\n\nadd = Adder()\nadd(4)\nprint(add(5))\n```",
        options={
            "five": "`5`",
            "error": "Niets: Python stopt, omdat `add` een object is, geen functie",
            "right": "`9`",
        },
        correct="right",
        why={
            "five": "Elke aanroep telt op bij `self.total`, dat het object tussen de aanroepen bewaart. Zie [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).",
            "error": "Een object waarvan de class `__call__` heeft, kun je aanroepen als een functie. Zie [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).",
            "right": "`add(4)` voert `add.__call__(4)` uit, en `self.total` wordt 4; `add(5)` telt er 5 bij op en geeft 9 terug.",
        },
    )
    quick_call
    return (quick_call,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) was je model twee losse tensors, `w` en `b`, en noemde de trainingsstap ze allebei bij naam. Een netwerk met drie layers heeft zes van zulke tensors, en een language model honderden. Je eigen project loste dat al op: `Layer` en `MLP` hielden hun parameters bij elkaar, en `parameters()` gaf ze allemaal tegelijk. PyTorch doet hetzelfde met **modules**. `nn.Linear` is een layer, en je eigen class op `nn.Module` is een model dat zijn layers, zijn parameters en zijn forward pass kent.

    **Een voorbeeld uit het dagelijks leven.** Een gereedschapskist: elk stuk gereedschap heeft er zijn eigen plek in. Als je verhuist, draag je de kist, niet elke schroevendraaier los, en als je iets nodig hebt, weet je waar je moet kijken.
    """)
    return


@app.cell(hide_code=True)
def predict_weight():
    guess_weight = L.guess(
        "weight",
        question="`layer = nn.Linear(3, 2)` is een layer met 3 inputs en 2 neuronen. Wat denk je dat de shape van `layer.weight` is?",
        options={
            "inputs": "(3, 2): een rij van 2 weights per input",
            "flat": "(6,): alle zes weights in één rij",
            "right": "(2, 3): een rij per neuron",
        },
        reveal="**(2, 3).** `nn.Linear` bewaart de weights van elk neuron als een rij, zoals `W` in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl). Voor voorbeelden in de rijen van $\\mathbf{X}$ rekent hij dus $\\mathbf{X}\\mathbf{W}^\\top + \\mathbf{b}$ uit: de getransponeerde uit [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl). Zijn `bias` heeft shape (2,), één bias term per neuron.",
    )
    guess_weight
    return (guess_weight,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In PyTorch is een **module** een stuk van een netwerk: een object dat parameters bevat en zijn forward pass kent. Ze bouwen allemaal op de class `nn.Module`, die je krijgt met `from torch import nn`.

        **Een layer: `nn.Linear`.** `nn.Linear(n_inputs, n_outputs)` maakt een layer van `n_outputs` neuronen, elk met `n_inputs` weights en een bias term. Zijn `.weight` heeft shape (outputs, inputs), één rij per neuron, en zijn `.bias` shape (outputs,). Allebei zijn ze een **`nn.Parameter`**: een tensor met `requires_grad=True` die een `nn.Module` bijhoudt. Ze beginnen bij willekeurige getallen, zoals de weights van je eigen `Neuron`; `torch.manual_seed(0)` geeft bij elke run dezelfde getallen. Roep je de layer aan op voorbeelden in de rijen van `X`, dan rekent hij uit:
        """),
        L.formula(r"\mathbf{X}[[param:\mathbf{W}]]^\top + [[param:\mathbf{b}]]"),
        mo.md(r"""
        In code geeft `layer(X)` hetzelfde als `X @ layer.weight.T + layer.bias`. Dat is de **conventie** van PyTorch: weights als rijen, en voorbeelden ook als rijen, dus worden de weights getransponeerd.

        **Je eigen model: een class op `nn.Module`.** Schrijf een class die voortbouwt op `nn.Module`, zoals in [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl). Zijn `__init__` roept eerst `super().__init__()` aan, zodat `nn.Module` zichzelf kan opzetten, en maakt dan elke layer als attribuut: `self.hidden = nn.Linear(2, 3)`. Zijn method **`forward(self, x)`** zegt wat het netwerk met `x` doet: de layers na elkaar, met `torch.relu` ertussen. Je roept het model zelf aan, `net(X)`: `nn.Module` heeft een `__call__`, zoals in [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl), en die voert je `forward` uit.

        **Alle parameters tegelijk.** Een `nn.Module` vindt elke layer die je hem als attribuut gaf, en elke parameter daarin: `net.parameters()` geeft ze allemaal, zoals `parameters()` van je eigen `MLP` deed. Eén ding is anders: je `MLP` gaf één `Value` per getal, en `net.parameters()` geeft hele tensors, de `weight` en de `bias` van elke layer. Bij een netwerk van twee layers zijn dat 4 tensors. Een trainingsstap loopt er in een loop doorheen, en `p -= lr * p.grad` schuift elk getal van een tensor tegelijk. Alleen attributen tellen mee: een layer in een gewone lijst, zoals je `MLP` zijn layers bewaarde, wordt niet gevonden.

        | In je project | In PyTorch |
        |---|---|
        | `Module` | `nn.Module` |
        | `Layer(2, 3, relu=False)` | `nn.Linear(2, 3)`: zonder ReLU erin |
        | de ReLU in elk `Neuron` | `torch.relu` in `forward` |
        | `MLP(2, [3, 1])` | je eigen class op `nn.Module`, met `forward` |
        | `m.parameters()`: één `Value` per getal | `net.parameters()`: één tensor per `weight` en `bias` |
        | `m(x)`, via `__call__` | `net(X)`, via `__call__` en `forward` |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **module** | | in PyTorch: een stuk van een netwerk, een object dat parameters bevat en zijn forward pass kent | `nn.Module` |
        | `nn.Linear` | n n linear | een layer: `nn.Linear(n_inputs, n_outputs)` rekent $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$ uit | `nn.Linear(2, 3)` |
        | `nn.Parameter` | | een tensor met `requires_grad=True` die een `nn.Module` bijhoudt, zoals `.weight` en `.bias` | `layer.weight` |
        | `forward` | | de method van je `nn.Module` die zegt wat hij met zijn input doet; `net(X)` voert hem uit | `def forward(self, x):` |
        | **conventie** $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$ | | PyTorch bewaart de weights van elk neuron als een rij, en de voorbeelden als rijen, dus transponeert hij de weights | `X @ layer.weight.T + layer.bias` |
        | `torch.manual_seed` | | de seed van de willekeurige getallen van PyTorch: bij elke run dezelfde beginweights | `torch.manual_seed(0)` |
        """),
        L.watch_out("Module betekent hier niet een Python-module, een bestand dat je importeert, zoals in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl). In PyTorch is een `nn.Module` een stuk van een netwerk: het speelt de rol van de class `Module` waarop je `Neuron`, `Layer` en `MLP` voortbouwden. Je kunt een `nn.Module` in een Python-module schrijven, dus lees welke er bedoeld wordt."),
        mo.md(r"""
        Het netwerk van de oefeningen, met 2 inputs, een hidden layer `self.hidden` van 3 neuronen met ReLU, en een output-layer `self.out` van 1 neuron:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_net():
    C.picture(["2 inputs", "nn.Linear(2, 3)", "nn.Linear(3, 1)"],
              title="Net: twee nn.Linear-layers",
              alt="Twee inputs, elk verbonden met de drie neuronen van de hidden layer, nn.Linear(2, 3), met ReLU; die drie verbonden met het ene output-neuron, nn.Linear(3, 1).")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een netwerk als `nn.Module` te bouwen, en te trainen:

    1. Schrijf een class op `nn.Module`. In `__init__` eerst `super().__init__()`, dan elke layer als attribuut: `self.hidden = nn.Linear(2, 3)`.
    2. Schrijf `forward(self, x)`: de layers na elkaar, met `torch.relu` ertussen.
    3. Roep het model aan op alle voorbeelden tegelijk: `y_pred = net(X)`.
    4. Train met `net.parameters()`: elke `.grad` op `None`, `loss.backward()`, en de stap voor elke parameter binnen `with torch.no_grad():`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een netwerk voor de vier hoeken van XOR, uit [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl), met 2 inputs, een hidden layer van 4 neuronen en 1 output. Het wordt 300 stappen getraind. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_module():
    import torch
    from torch import nn

    # 1. de class, met zijn layers als attributen
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.out = nn.Linear(4, 1)

        # 2. de forward pass
        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    torch.manual_seed(1)
    _net = _Net()
    print(_net)
    # 3. alle voorbeelden tegelijk: (4, 2) -> (4, 1)
    _X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    print(_net(_X))
    # 4. trainen, met elke parameter van het netwerk
    for _p in _net.parameters():
        print(_p.shape)
    for _step in range(300):
        _loss = ((_net(_X) - _y) ** 2).mean()
        for _p in _net.parameters():
            _p.grad = None
        _loss.backward()
        with torch.no_grad():
            for _p in _net.parameters():
                _p -= 0.1 * _p.grad
    print(_loss.item())
    print(_net(_X))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `print(_net)` laat de twee layers zien onder hun attribuutnamen; marimo schrijft de class `_Net` als `_cell_…_Net`, omdat een naam met `_` bij zijn eigen cel hoort. Vóór het trainen geeft `_net(_X)` vier getallen die niets met XOR te maken hebben. `_net.parameters()` geeft vier tensors, in de volgorde waarin de layers gemaakt zijn: de `weight` van de hidden layer, shape (4, 2), een rij van 2 weights voor elk van zijn 4 neuronen; zijn `bias`, shape (4,); en de `weight` (1, 4) en `bias` (1,) van de output-layer. Samen bevatten ze 4 · 2 + 4 + 4 + 1 = 17 getallen, en de loop schuift ze allemaal zonder er één bij naam te noemen. Na 300 stappen is de loss bijna 0, en zijn de outputs bijna 0, 1, 1 en 0. PyTorch schrijft ze kort, zoals in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl): `9.9943e-01` is 0,99943, en `1.9103e-03` is 0,0019103.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat er uit een layer komt
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "output-shape",
        question="`layer = nn.Linear(5, 3)`, en `X` bevat 10 voorbeelden met elk 5 inputs: shape (10, 5). Wat is de shape van `layer(X)`?",
        options={
            "right": "(10, 3)",
            "weight": "(3, 5)",
            "inputs": "(10, 5)",
        },
        correct="right",
        why={
            "weight": "Dat is de shape van `layer.weight`, één rij per neuron. Wat geeft de layer voor elk voorbeeld? Zie de formule in *Het idee*.",
            "inputs": "Elk voorbeeld gaat erin met 5 inputs; hoeveel getallen komen eruit, één per wat? Zie de formule in *Het idee*.",
            "right": "$\\mathbf{X}\\mathbf{W}^\\top$ is (10, 5) @ (5, 3), en de bias term van elk neuron wordt bij elke rij opgeteld: één rij per voorbeeld, één kolom per neuron.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_net():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je eigen nn.Module

        Maak de class `Net` af: een netwerk met 2 inputs, een hidden layer van 3 neuronen met ReLU, en een output-layer van 1 neuron, zoals in de figuur. Vervang de twee `...`. Maak de layers in de volgorde van het netwerk, eerst `self.hidden` en dan `self.out`: PyTorch trekt hun willekeurige beginweights in de volgorde waarin je ze maakt. De laatste regels maken het netwerk met seed 0 en roepen het aan op de vier hoeken van XOR.
        """),
        L.task("Maak van `Net` een `nn.Module` met de layers `self.hidden` en `self.out`, en een `forward` met ReLU ertussen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_net():
    box_net = L.code_box("net", lines=16)
    box_net
    return (box_net,)


@app.cell(hide_code=True)
def check_net(box_net):
    result_net = L.check("net", box_net.value)
    result_net
    return (result_net,)


@app.cell(hide_code=True)
def hints_net():
    L.hints(
        "net",
        [
            "Stap 1 en 2 van het recept, zoals in het uitgewerkte voorbeeld, met andere groottes.",
            "`__init__` heeft drie regels nodig: `super().__init__()` en de twee layers. `nn.Linear` krijgt eerst het aantal inputs en dan het aantal neuronen. `forward` is één regel met `return`.",
            "`self.hidden = nn.Linear(2, ___)`, `self.out = nn.Linear(___, 1)`, en `return self.out(torch.relu(___))`.",
        ],
        why="De hidden layer krijgt de 2 inputs en heeft 3 neuronen: `nn.Linear(2, 3)`. De output-layer krijgt die 3 en heeft 1 neuron: `nn.Linear(3, 1)`. `forward` stuurt `x` door `self.hidden`, dan ReLU, dan `self.out`. Dat zijn 2 · 3 + 3 + 3 + 1 = 13 parameters.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen op XOR

        De class van oefening 2 staat er al, en de forward pass met de loss. Vervang de `...` door de rest van elke stap van de loop: stap 4 van het recept, voor elke parameter van `net`. De laatste regel is de loss van de laatste stap, als Python-getal. De check probeert ook een andere learning rate.
        """),
        L.task("Laat elke stap elke `.grad` van het netwerk op `None` zetten, de backward pass doen en elke parameter tegen zijn helling in schuiven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=24)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Stap 4 van het recept, zoals aan het eind van het uitgewerkte voorbeeld.",
            "Twee loops over `net.parameters()`: één vóór `loss.backward()`, om elke `.grad` op `None` te zetten, en één binnen `with torch.no_grad():` voor de stap.",
            "`for p in net.parameters():` met `p.grad = None`; dan `loss.___()`; dan `with torch.no_grad():` met `for p in net.parameters():` en `p -= ___`.",
        ],
        why="Elke stap begint zonder oude hellingen, krijgt nieuwe voor alle 13 parameters uit één `loss.backward()`, en schuift elke parameter een klein stukje tegen zijn helling in. Na 500 stappen geeft het netwerk bijna 0, 1, 1 en 0: XOR.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** `nn.Linear` is de layer uit [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl), $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$, waarvan de weights voor je bewaard worden, en een class op `nn.Module` is je eigen `MLP` in PyTorch: layers als attributen, een `forward`, en `parameters()` voor ze allemaal. De training loop uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) blijft hetzelfde; hij loopt alleen over `net.parameters()`. In de volgende les, *Laat de optimizer de stap zetten*, neemt PyTorch ook de twee loops in elke stap over.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_net, result_train):
    L.wrap_up(choice_shape, result_net, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
