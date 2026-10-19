# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-dropout. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow regularization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-dropout", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl) en [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_worse():
    quick_worse = L.choice(
        "quick-worse",
        question="Met regularisatie eindigt de loss op de training set iets hoger dan zonder. Waarom kan dat toch het betere model zijn?",
        options={
            "right": "Het leerde minder van de ruis, dus het kan het beter doen op nieuwe voorbeelden",
            "faster": "Het trainde sneller, dus je kunt het meer epochs trainen",
            "smaller": "Zijn weights werden precies 0, dus het model is kleiner en sneller te gebruiken",
        },
        correct="right",
        why={
            "faster": "Regularisatie maakt het trainen niet sneller; het verandert wat het model leert. Zie [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "smaller": "Een L2-straf maakt weights kleiner, niet precies 0. Zie [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "right": "Een model dat elk trainingsvoorbeeld precies volgt, volgt ook de ruis erin. Regularisatie levert een beetje van de fit op de training set in voor een betere fit op nieuwe data. Vandaag leer je een vorm van regularisatie kennen die voor neural networks gemaakt is.",
        },
    )
    quick_worse
    return (quick_worse,)


@app.cell(hide_code=True)
def quick_check_mode():
    quick_mode = L.choice(
        "quick-mode",
        question="Een layer die je zelf schrijft op basis van `nn.Module` moet tijdens het trainen anders werken dan bij het voorspellen. Hoe weet zijn `forward` welke van de twee hij doet?",
        options={
            "grad": "Hij kijkt of `torch.no_grad()` aanstaat",
            "eval": "Hij roept `self.eval()` aan en kijkt wat dat teruggeeft",
            "right": "Hij leest `self.training`: `True` in train mode, `False` na `model.eval()`",
        },
        correct="right",
        why={
            "grad": "`torch.no_grad()` stopt alleen het opnemen van een graaf; het zegt niets over de mode. Zie [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl).",
            "eval": "`eval()` zet de mode om in plaats van hem te lezen. Zie [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl).",
            "right": "`model.eval()` en `model.train()` zetten `training` in het model en in elke layer erin. Je eigen dropout layer leest het in oefening 2.",
        },
    )
    quick_mode
    return (quick_mode,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een netwerk met veel neuronen dat lang traint op weinig voorbeelden, gaat zijn voorbeelden uit het hoofd leren: de loss op de training set blijft dalen, de loss op de test set begint te stijgen. Een L2-straf is één manier om dat tegen te houden. **Dropout** is een manier die voor neural networks gemaakt is: tijdens het trainen zet het bij elke stap een willekeurig deel van de neuronen uit. De groep van Geoffrey Hinton bedacht het in 2012, en AlexNet, het netwerk dat dat jaar de opmars van deep learning inluidde, gebruikte het.
    """)
    return


@app.cell(hide_code=True)
def predict_dropout():
    guess_dropout = L.guess(
        "with-dropout",
        question="Een MLP met twee hidden layers van 256 neuronen traint 100 epochs op maar 500 trainingscijfers, met SGD en een learning rate van 0,1. Het leest 819 van de 1.000 testcijfers goed. Nu krijgt het dropout na elke hidden layer: bij elke trainingsstap gaat elk neuron van de hidden layers uit met een kans van een half. Al het andere blijft hetzelfde. Hoeveel testcijfers denk je dat het na 100 epochs goed leest?",
        options={
            "fewer": "Duidelijk minder, ongeveer 600: bij elke stap mist het de helft van zijn neuronen",
            "right": "Iets meer, ongeveer 840: het overfit minder, maar 500 cijfers blijven weinig",
            "many": "Veel meer, ongeveer 950: het overfit helemaal niet meer",
        },
        reveal="**840.** En de loss op de testcijfers eindigt lager: 0,60 in plaats van 0,67. Bij elke stap mist de helft van de neuronen, maar elke stap een andere helft, en uiteindelijk heeft elk neuron geleerd. Wat verandert, is dat geen neuron op bepaalde andere kan rekenen: het moet op zichzelf nuttig zijn. Dat maakt 500 voorbeelden nog niet zo goed als 6.000: de winst is echt, maar niet groot.",
    )
    guess_dropout
    return (guess_dropout,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In train mode zet dropout elk getal van de output van een layer op 0 met kans $p$, elk getal apart, met een nieuwe willekeurige keuze bij elke forward pass. De getallen die blijven, worden gedeeld door $1 - p$. In eval mode doet het niets: elk neuron telt mee zoals het is. Een **dropout mask** $m$, kortweg een *mask*, heeft één 0 of 1 voor elk getal. Daarmee wordt getal $i$ van de output van de layer, $h_i$, het getal $\tilde{h}_i$:
        """),
        L.formula(r"[[hidden:\tilde{h}_i]] = \frac{m_i}{1 - [[hyper:p]]} \, [[hidden:h_i]]"),
        L.formula(r"m_i = \begin{cases} 1 & \text{met kans } 1 - [[hyper:p]] \\ 0 & \text{met kans } [[hyper:p]] \end{cases}"),
        mo.md(r"""
        Waarom delen door $1 - p$? Met $p = 0{,}5$ valt de helft van de getallen weg, dus krijgt de volgende layer gemiddeld maar de helft van wat hij in eval mode krijgt. Dat maak je goed door de getallen die blijven te verdubbelen: gemiddeld is elk getal in train mode even groot als in eval mode, en de volgende layer ziet in allebei dezelfde groottes. PyTorch doet het zo, tijdens het trainen, zodat voorspellen niets extra's nodig heeft.

        Waarom helpt het? Elke stap traint een ander uitgedund netwerk, gemaakt van de neuronen die toevallig blijven, en ze delen allemaal dezelfde weights. Een neuron kan er niet op rekenen dat een bepaald ander neuron er is, dus moet het iets leren dat op zichzelf nuttig is. In eval mode werkt het hele netwerk als een soort gemiddelde van al die uitgedunde netwerken.

        Waar en hoeveel: na de activation van een hidden layer, nooit op de output. Het artikel over dropout uit 2014 liet de helft van de neuronen in hidden layers vallen en een vijfde van de inputs; de transformer van 2017 gebruikte 0,1. Grote language models worden vaak zonder dropout gepretraind: ze zien de meeste voorbeelden maar één keer, en overfitten nauwelijks. Een verwant, DropPath, slaat willekeurig hele blokken layers over in plaats van losse getallen; dat wordt gebruikt in heel diepe netwerken, waarvan je blokken kunt overslaan. In PyTorch is dropout de layer `nn.Dropout(p)`; die leest `self.training`, dus `model.eval()` zet hem uit. Voor een eigen dropout layer is er `torch.rand_like(x)`: een tensor met de shape van `x` vol willekeurige getallen tussen 0 en 1.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **dropout** | | in train mode elk getal van de output van een layer op 0 zetten met kans $p$, en de rest delen door $1 - p$ | `nn.Dropout(p)` |
        | $p$ | p | de kans dat een getal wegvalt | `p=0.5` |
        | **dropout mask** | | de nullen en enen die zeggen welke getallen blijven; kortweg een mask | `keep = torch.rand_like(x) >= p` |
        | `torch.rand_like` | torch rand like | willekeurige getallen tussen 0 en 1, met de shape van een tensor die je geeft | `torch.rand_like(x)` |
        """),
        L.watch_out("Dropout werkt alleen in train mode, dus als je `model.eval()` vergeet voordat je meet, hangt elke voorspelling van het toeval af, en lijken de loss en de accuracy slechter dan ze zijn. En dropout maakt het trainen trager: de training loss daalt langzamer, en het netwerk heeft misschien meer epochs nodig."),
        mo.md(r"""
        De figuur: het MLP uit de gok, 100 epochs op 500 trainingscijfers, zonder dropout (gestippeld) en met dropout 0,5 na elke hidden layer (doorgetrokken), en zijn loss op de 1.000 testcijfers na elke epoch, in eval mode. Zonder dropout is de test loss het laagst na epoch 30, met 0,56, en stijgt hij daarna tot 0,67 bij epoch 100: overfitting. Met dropout daalt hij iets verder, tot 0,53, en eindigt hij op 0,60.
        """),
        mo.Html(C.dropout_picture({"title": "Loss op de testcijfers", "x": "epoch", "y": "test loss", "without": "zonder dropout", "with": "met dropout 0,5",
                                   "alt": "Twee lijnen over 100 epochs. Allebei dalen ze steil in de eerste 25 epochs. Zonder dropout, gestippeld, is de test loss het laagst met ongeveer 0,56 rond epoch 30 en stijgt hij daarna tot ongeveer 0,67. Met dropout, doorgetrokken, blijft hij tussen ongeveer 0,53 en 0,62."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Dropout op de output van een hidden layer:

    1. Kies $p$: vaak 0,5 voor brede hidden layers, 0,1 in transformers.
    2. In train mode: trek voor elk getal van de output een willekeurig getal tussen 0 en 1; houd de getallen waar dat minstens $p$ is, en zet de andere op 0.
    3. Deel de getallen die je houdt door $1 - p$.
    4. In eval mode: sla stap 2 en 3 over, en geef de output terug zoals hij is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Dropout met $p = 0{,}5$ op de vier outputs van een hidden layer, in NumPy: één keer, dan het gemiddelde over 10.000 masks, en dan in eval mode. `_x * _keep` vermenigvuldigt met de `True` en `False` van het mask, die als 1 en 0 tellen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_dropout():
    import numpy as np

    _rng = np.random.default_rng(3)
    _h = np.array([2.0, 1.0, 4.0, 3.0])
    # 1. de kans om weg te vallen
    _p = 0.5


    def _drop(_x):
        # 2. een mask: houd waar het willekeurige getal minstens _p is
        _keep = _rng.random(4) >= _p
        # 3. maak af: deel ook door (1 - _p)
        return _x * _keep


    print(_drop(_h))
    _total = np.zeros(4)
    for _i in range(10000):
        _total = _total + _drop(_h)
    print(np.round(_total / 10000, 2))
    # 4. eval mode: geen mask, de output zoals hij is
    print(_h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het eerste mask houdt de laatste twee getallen, dus zoals hij is, drukt de cel eerst 0, 0, 4 en 3 af, en dan de gemiddelden 1; 0,5; 2,02 en 1,49: ongeveer de helft van $h$, omdat de helft van de getallen wegvalt. Afgemaakt is de eerste output 0, 0, 8 en 6, de gehouden getallen verdubbeld, en zijn de gemiddelden 1,99; 1,01; 4,05 en 2,98: ongeveer $h$ zelf, op het toeval van 10.000 masks na. Dat is wat eval mode elke keer geeft, zonder mask: de laatste regel, 2, 1, 4 en 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom dropout helpt
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-it-helps",
        question="Waarom leest een netwerk met dropout nieuwe cijfers vaak beter?",
        options={
            "smaller": "Het is een kleiner netwerk: de helft van zijn weights wordt nooit getraind",
            "faster": "Het traint sneller, dus het bereikt een lage loss voordat het de ruis van zijn voorbeelden kan leren",
            "right": "Geen neuron kan op bepaalde andere rekenen, dus elk leert op zichzelf iets nuttigs",
        },
        correct="right",
        why={
            "smaller": "Elke weight wordt getraind: elke stap laat andere neuronen vallen, en in eval mode tellen ze allemaal mee.",
            "faster": "Dropout maakt het trainen trager, niet sneller: elke stap traint maar een deel van het netwerk, dus de loss daalt langzamer.",
            "right": "Neuronen die alleen samen met bepaalde andere werken, kunnen patronen leren die alleen bij de trainingsvoorbeelden passen. Dropout breekt die samenwerkingen op, en het netwerk als geheel werkt als een gemiddelde van veel uitgedunde netwerken.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een eigen dropout layer

        Schrijf een class `Dropout(nn.Module)`, je eigen `nn.Dropout`. `Dropout(p)` bewaart `p`. `forward(x)` volgt het recept: in train mode het mask `keep = torch.rand_like(x) >= self.p`, zoals in het uitgewerkte voorbeeld, en dan stap 3; `x * keep` zet `x` op 0 waar `keep` `False` is. In eval mode geef je `x` terug zoals hij is; `self.training` zegt welke mode het is. De check stuurt 100.000 enen door je layer met $p = 0{,}2$, in train mode en in eval mode, en vergelijkt het deel nullen, het gemiddelde en het grootste getal; daarna hetzelfde met $p = 0{,}5$, en een kleine tensor met $p = 0$.
        """),
        L.task("Laat `Dropout` in train mode getallen weglaten en de rest schalen, en in eval mode niets veranderen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("own-dropout", lines=14)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("own-dropout", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "own-dropout",
        [
            "In `__init__` `self.p = p` na `super().__init__()`. In `forward` eerst eval mode: `if not self.training:` geef `x` terug.",
            "Daarna komt train mode: het mask, en dan `x * keep` gedeeld door `___`, zodat de getallen die blijven goedmaken wat er wegvalt.",
        ],
        why="`__init__`: `super().__init__()` en `self.p = p`. `forward`: `if not self.training: return x`; `keep = torch.rand_like(x) >= self.p`; `return x * keep / (1 - self.p)`. Met $p = 0{,}2$ wordt ongeveer 20% van de enen 0, de andere 1,25, en blijft het gemiddelde ongeveer 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het netwerk van de figuur

        Schrijf `DropoutMLP(hidden, p)`, het netwerk van de figuur: de layers `hidden1`, `hidden2` en `out` staan er al; voeg in `__init__` een dropout layer `nn.Dropout(p)` toe, en gebruik die in `forward` na de ReLU van elke hidden layer, niet op de output. De check maakt `DropoutMLP(8, 0.5)` met een vaste seed, vergelijkt zijn output voor een willekeurig plaatje in eval mode en in train mode met die van het netwerk van de course, en kijkt of twee aanroepen in eval mode dezelfde output geven en in train mode een andere.
        """),
        L.task("Laat `DropoutMLP` dropout gebruiken na elke hidden layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("dropout-mlp", lines=16)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("dropout-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "dropout-mlp",
        [
            "Eén dropout layer is genoeg voor beide plekken: hij heeft geen weights, dus `self.drop = nn.Dropout(p)` kun je twee keer gebruiken.",
            "Lees in `forward` elke hidden layer van binnen naar buiten: eerst de layer, dan `torch.relu`, dan de dropout layer. `self.out` komt als laatste, zonder dropout.",
        ],
        why="`__init__`: `self.drop = nn.Dropout(p)`. `forward`: `h = self.drop(torch.relu(self.hidden1(x)))`; `h = self.drop(torch.relu(self.hidden2(h)))`; `return self.out(h)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Net als de L2-straf uit [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl) levert dropout een beetje van de fit op de training set in voor een betere fit op nieuwe data, maar het doet dat door ruis in het trainen te brengen in plaats van door weights naar 0 te trekken. Het is de layer die [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl) aan en uit zette: nu weet je wat hij in elke mode doet, en waarom hij schaalt. In de volgende les, *Embedding layers*, maakt een layer van ids, zoals het nummer van een woord, vectoren die geleerd worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_layer, result_mlp):
    L.wrap_up(choice_why, result_layer, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
