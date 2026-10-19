# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-optimizer. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-optimizer", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl) en [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="`net` heeft de layers `self.hidden = nn.Linear(2, 3)` en `self.out = nn.Linear(3, 1)`. Hoeveel tensors geeft `net.parameters()`?",
        options={
            "layers": "2: één per layer",
            "numbers": "13: één per getal",
            "right": "4: een `weight` en een `bias` voor elke layer",
        },
        correct="right",
        why={
            "layers": "Elke layer heeft twee parameters, zijn `weight` en zijn `bias`, zoals het uitgewerkte voorbeeld liet zien met `for _p in _net.parameters():`. Zie *Alle parameters tegelijk* in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl).",
            "numbers": "13 is het aantal getallen erin. Elke `weight` en elke `bias` is één tensor. Zie [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl).",
            "right": "`hidden.weight` (3, 2), `hidden.bias` (3,), `out.weight` (1, 3) en `out.bias` (1,): vier tensors met samen 13 getallen.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="Een update-stap schuift $w$ tegen de helling van de loss in, met de learning rate keer de helling. Onderin het dal is de helling 0. Wat doet een update-stap daar?",
        options={
            "lr": "Hij schuift $w$ op met de learning rate",
            "back": "Hij stuurt $w$ het dal weer uit",
            "right": "Niets: $w$ blijft waar hij is",
        },
        correct="right",
        why={
            "lr": "De stap is de learning rate keer de helling. Wat is een getal keer 0? Zie de update-stap in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "back": "Met een helling van 0 is er geen richting om in te schuiven. Zie de update-stap in [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl).",
            "right": "De learning rate keer 0 is 0, dus $w$ blijft staan: gradient descent stopt onderin vanzelf.",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elke stap van je training loop in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl) had twee eigen loops: één om elke `.grad` op `None` te zetten, en één voor de update-stap uit [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl), binnen `torch.no_grad()`. Elk netwerk heeft dezelfde twee loops nodig. En juist in de update-stap valt er te winnen: er zijn manieren om slimmere stappen te zetten dan "de learning rate keer de helling", en elke manier zou die loop veranderen. PyTorch zet de stap in een eigen object, een **optimizer**: je geeft hem één keer de parameters, en vanaf dan doen twee korte aanroepen beide loops.

    **Een voorbeeld uit het dagelijks leven.** Een wasmachine heeft programma's: je vult hem één keer, drukt op start, en hij wast, spoelt en centrifugeert. Je staat er niet naast om elk deel zelf te doen, en een nieuwe machine met een beter programma heeft nog steeds dezelfde startknop.
    """)
    return


@app.cell(hide_code=True)
def predict_early():
    guess_early = L.guess(
        "early",
        question="`opt = torch.optim.SGD(net.parameters(), lr=0.1)` is gemaakt, en `opt.step()` wordt aangeroepen vóór er een `loss.backward()` is geweest. Wat denk je dat er gebeurt?",
        options={
            "error": "Python stopt met een fout: er zijn nog geen hellingen",
            "right": "Niets: geen enkele parameter verschuift",
            "random": "Elke parameter verschuift een willekeurig stukje",
        },
        reveal="**Niets.** Vóór de eerste backward pass is elke `.grad` `None`, en `opt.step()` slaat elke parameter zonder `.grad` over. Hij verschuift alleen wat `loss.backward()` een helling gaf. Een loop die `loss.backward()` vergeet, draait dus zonder fout, en de loss verandert gewoon nooit.",
    )
    guess_early
    return (guess_early,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **optimizer** is een object dat de update-stap zet voor een lijst met parameters. Je maakt hem één keer, vóór de loop, van de parameters die hij moet veranderen en de learning rate: `opt = torch.optim.SGD(net.parameters(), lr=0.1)`. Hij bewaart ze, en heeft twee methods:

        - **`opt.zero_grad()`** zet de `.grad` van elk van zijn parameters op `None`: de eerste van je twee loops.
        - **`opt.step()`** schuift elke parameter die een `.grad` heeft tegen zijn helling in, met de learning rate keer die helling, zonder het op te nemen: de tweede loop, met zijn `torch.no_grad()`.

        `SGD` zet precies de update-stap van gradient descent, voor elke parameter $w$:
        """),
        L.formula(r"[[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]]"),
        mo.md(r"""
        De naam SGD komt van *stochastic gradient descent*, dat de volgende les uitlegt; voorlopig is het de update-stap die je kent. Een stap van de training loop bestaat dan uit vier regels: de forward pass met de loss, `opt.zero_grad()`, `loss.backward()` en `opt.step()`. `opt.zero_grad()` mag ook als laatste, na `opt.step()`, zolang de volgende `loss.backward()` geen oude hellingen vindt. Eén stap van jouw loop, en dezelfde stap met een optimizer:
        """),
        C.picture({"own": "Jouw loop", "opt": "Met een optimizer", "1": "elke .grad op None: opt.zero_grad()",
                   "2": "de update-stap, binnen torch.no_grad(): opt.step()"}, title="Eén stap van de training loop"),
        mo.md(r"""
        Andere optimizers, zoals Adam, zetten slimmere stappen, met dezelfde twee aanroepen; een latere les legt ze uit. Wil je er een proberen, dan verandert alleen de regel die `opt` maakt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **optimizer** | | een object dat de update-stap zet voor de parameters die je hem gaf | `opt = torch.optim.SGD(net.parameters(), lr=0.1)` |
        | `torch.optim.SGD` | es gee dee | de optimizer die de stap van gradient descent zet: elke parameter min de learning rate keer zijn helling | `torch.optim.SGD(params, lr=lr)` |
        | `opt.zero_grad()` | zero grad | zet de `.grad` van elke parameter van de optimizer op `None` | `opt.zero_grad()` |
        | `opt.step()` | | één update-stap voor elke parameter die een `.grad` heeft | `opt.step()` |
        """),
        L.watch_out("Een optimizer houdt de parameters waarmee hij gemaakt is. Maak je een nieuw netwerk, dan werkt de oude optimizer nog steeds alleen op de parameters van het oude, en verschuift het nieuwe netwerk nooit: maak dan ook een nieuwe optimizer, van de nieuwe `net.parameters()`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te trainen met een optimizer:

    1. Maak het netwerk, en dan de optimizer van zijn parameters: `opt = torch.optim.SGD(net.parameters(), lr=lr)`.
    2. Reken in elke stap de loss uit met een forward pass.
    3. `opt.zero_grad()`, dan `loss.backward()`.
    4. `opt.step()`, en bewaar `loss.item()` als je de loss wilt volgen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het netwerk voor XOR uit het uitgewerkte voorbeeld van [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl), met 4 hidden neuronen en seed 1, nu getraind met een optimizer. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_optimizer():
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.out = nn.Linear(4, 1)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    _X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    # 1. het netwerk, en de optimizer van zijn parameters
    torch.manual_seed(1)
    _net = _Net()
    _opt = torch.optim.SGD(_net.parameters(), lr=0.1)
    print(_opt)
    _losses = []
    for _step in range(300):
        # 2. de forward pass en de loss
        _loss = ((_net(_X) - _y) ** 2).mean()
        # 3. geen oude hellingen, dan de backward pass
        _opt.zero_grad()
        _loss.backward()
        # 4. de stap, en de loss als getal
        _opt.step()
        _losses.append(_loss.item())
    print(_losses[0], _losses[50], _losses[100])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `print(_opt)` laat de optimizer zien met zijn instellingen voor `Parameter Group 0`, de parameters die hij kreeg: `lr: 0.1`, en andere die hier 0 of uit blijven. De loss gaat van ongeveer 0,44 aan het begin naar ongeveer 0,11 na 50 stappen en 0,014 na 100: dezelfde getallen als met je eigen twee loops in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl), omdat `SGD` dezelfde stappen zet. Alleen de loop is korter geworden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stale():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een nieuw netwerk, een oude optimizer
    """)
    return


@app.cell(hide_code=True)
def choice_stale_cell():
    choice_stale = L.choice(
        "stale",
        question="`opt = torch.optim.SGD(net.parameters(), lr=0.1)` is gemaakt. Later maak je een nieuw netwerk, `net = Net()`, en train je het met dezelfde `opt`. Wat gebeurt er?",
        options={
            "follows": "Het nieuwe netwerk traint: `opt` zoekt bij elke stap de parameters van `net` op",
            "right": "Het nieuwe netwerk verandert niet: `opt` houdt de parameters van het oude",
            "error": "Python stopt met een fout bij de eerste `opt.step()`",
        },
        correct="right",
        why={
            "follows": "`opt` kreeg bij het maken de parameters zelf, niet de naam `net`. Zie het let-op in *Het idee*.",
            "error": "Voor Python is er niets mis: `opt` heeft parameters, ze horen alleen bij het oude netwerk. Zie het let-op in *Het idee*.",
            "right": "`opt.zero_grad()` en `opt.step()` raken alleen de parameters waarmee `opt` gemaakt is. Het nieuwe netwerk krijgt hellingen van `loss.backward()`, maar niets verschuift ze. Maak een nieuwe optimizer van de nieuwe `net.parameters()`.",
        },
    )
    choice_stale
    return (choice_stale,)


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loop met een optimizer

        Het netwerk van oefening 2 in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl), met 3 hidden neuronen en seed 0, en zijn optimizer staan er al, met de forward pass in de loop. Vervang de `...` door de rest van elke stap: stap 3 en 4 van het recept. De laatste regel is de loss van de laatste stap, als Python-getal. De check probeert ook een andere learning rate.
        """),
        L.task("Laat de optimizer de gradients op nul zetten en de stap zetten, rond de backward pass."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("loop", lines=24)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("loop", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "loop",
        [
            "Stap 3 en 4 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Drie aanroepen, elk op een eigen regel in de loop: één op `opt` om de oude hellingen te wissen, één op `loss`, en één op `opt` voor de stap.",
            "`opt.zero_grad()`, `loss.___()` en `opt.___()`.",
        ],
        why="`opt.zero_grad()` wist de hellingen van de stap ervoor, `loss.backward()` geeft elke parameter zijn nieuwe helling, en `opt.step()` schuift elke parameter ertegenin. Na 500 stappen geeft het netwerk bijna 0, 1, 1 en 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een functie die traint

        Maak `train(net, X, y, lr, steps)` af: hij maakt zijn eigen optimizer van de parameters van `net` met de learning rate `lr`, zet `steps` stappen, en geeft de loss van elke stap terug in een lijst, als Python-getallen. Vervang de twee `...`. De laatste regel traint het netwerk van oefening 2 vijf stappen lang. De check probeert ook een nieuw netwerk met een andere learning rate en een ander aantal stappen.
        """),
        L.task("Laat `train` elk netwerk trainen met een optimizer, en zijn losses teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=26)
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
            "Stap 1 van het recept vóór de loop, en stap 3 en 4 erin, zoals in het uitgewerkte voorbeeld.",
            "De optimizer krijgt de parameters van het `net` dat `train` meekreeg, en `lr`. In de loop krijgt `losses` na de drie aanroepen de loss als Python-getal.",
            "`opt = torch.optim.SGD(net.___(), lr=___)`; in de loop `opt.zero_grad()`, `loss.backward()`, `opt.step()` en `losses.append(loss.___())`.",
        ],
        why="`train` maakt de optimizer van het `net` dat hij krijgt, dus werkt hij voor elk netwerk. Elke stap bewaart `loss.item()`, de loss vóór die stap: eerst 1,3444, en daarna elke keer lager.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een optimizer is de update-stap uit [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl), gezet voor alle parameters van een netwerk tegelijk: `opt.zero_grad()` en `opt.step()` vervangen de twee loops uit [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl). Je `train` werkt nu voor elk netwerk. In de volgende les, *Mini-batches en epochs*, leert elke stap van een klein deel van de data in plaats van alles, en zie je waar de S van SGD vandaan komt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stale, result_loop, result_train):
    L.wrap_up(choice_stale, result_loop, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
