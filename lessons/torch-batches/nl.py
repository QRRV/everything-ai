# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-batches. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-batches", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl) en [Generators: waarden één voor één maken](/course/open/py-generators?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="Wat is de `.grad` van elke parameter van `opt`, vlak na `opt.zero_grad()`?",
        options={
            "right": "`None`",
            "zeros": "Een tensor met nullen, in de shape van de parameter",
            "last": "De helling van de stap ervoor",
        },
        correct="right",
        why={
            "zeros": "Zo was het vroeger, en de naam doet het vermoeden, maar nu gooit `zero_grad()` de hellingen helemaal weg. Zie *Het idee* in [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl).",
            "last": "Die wist `zero_grad()` juist, zodat de volgende `loss.backward()` zonder die helling begint. Zie [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl).",
            "right": "`opt.zero_grad()` zet de `.grad` van elke parameter die hij heeft op `None`: nog geen helling, tot de volgende `loss.backward()`.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="Wat laat dit programma zien?\n\n```python\ndef pairs(items, starts):\n    for s in starts:\n        yield items[s:s + 2]\n\n\nprint(list(pairs([10, 20, 30, 40, 50, 60], [0, 3])))\n```",
        options={
            "first": "`[[10, 20]]`",
            "stop": "`[[10, 20, 30], [40, 50, 60]]`",
            "right": "`[[10, 20], [40, 50]]`",
        },
        correct="right",
        why={
            "first": "`yield` geeft één paar en pauzeert; `list` vraagt om het volgende, en de loop gaat verder met de volgende `s`. Zie [Generators: waarden één voor één maken](/course/open/py-generators?lang=nl).",
            "stop": "Een slice stopt net vóór zijn tweede getal: `items[0:2]` is `[10, 20]`, twee items.",
            "right": "De generator geeft eerst `items[0:2]` en dan `items[3:5]`, één voor één, en `list` verzamelt ze.",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je `train` uit [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl) rekent vóór elke stap de loss over alle voorbeelden uit. Voor de 4 hoeken van XOR kost dat niets. QMNIST, een verzameling handgeschreven cijfers, heeft 60.000 afbeeldingen om op te trainen: elke stap zou ze eerst allemaal door het netwerk sturen, en een netwerk heeft duizenden stappen nodig. Echt trainen gaat anders. Elke stap kijkt naar een klein deel van de data, een **mini-batch**, zodat het model in één ronde door de data veel stappen zet. Elke stap is iets minder precies, maar het zijn er veel meer.

    **Een voorbeeld uit het dagelijks leven.** Om te weten of een grote pan soep zout nodig heeft, proef je één lepel, niet de hele pan. Eén lepel kan er een beetje naast zitten, dus je roert, doet er een snufje bij en proeft opnieuw. Dat is veel sneller dan elke keer de hele pan leeg eten.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="60.000 trainingsafbeeldingen, mini-batches van 100 afbeeldingen, en elke afbeelding één keer gebruikt. Hoeveel update-stappen denk je dat dat zijn?",
        options={
            "one": "1: één keer door de data",
            "right": "600",
            "hundred": "100: één per afbeelding in een batch",
        },
        reveal="**600.** 60.000 / 100 = 600 mini-batches, elk met een eigen update-stap. Eén keer door alle data, met elk voorbeeld één keer, heet een **epoch**. Hier is één epoch 600 stappen, waar de loop van de vorige les er 1 zette.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **mini-batch** is een kleine groep voorbeelden, zoals 32 of 100, voor één update-stap: de loss van die groep, zijn hellingen, en één stap. De **batch size** is hoeveel voorbeelden er in een mini-batch gaan. Eén update-stap op één mini-batch is een **iteratie**, en één keer door alle data, met elk voorbeeld één keer, een **epoch**. Met $n$ voorbeelden en batch size $B$ heeft een epoch $n / B$ iteraties, naar boven afgerond: de laatste mini-batch kan kleiner zijn.

        De loss van een mini-batch is het gemiddelde over alleen zijn eigen voorbeelden, $B$ in een volle mini-batch:
        """),
        L.formula(r"[[loss:\mathcal{L}_{\text{batch}}]] = \frac{1}{B} \sum_{i=1}^{B} \left(y_i - [[pred:\hat{y}_i]]\right)^2"),
        mo.md(r"""
        Hier telt $i$ de voorbeelden van de mini-batch. De hellingen daarvan zijn een schatting van de hellingen over alle data: dichtbij, maar niet hetzelfde, en voor elke batch een beetje anders. Gradient descent met deze stappen heet **stochastic gradient descent**, SGD: *stochastic* betekent dat toeval meespeelt, hier in welke voorbeelden samen in een batch zitten. Daar heeft `torch.optim.SGD` zijn naam van. De optimizer zet dezelfde stap als voorheen; wat de descent stochastic maakt, zijn de batches die je hem geeft. Daarvoor moeten de voorbeelden in een willekeurige volgorde staan. De data hier staat al in een willekeurige volgorde; de volgende les shufflet de voorbeelden voor elke epoch opnieuw.

        **De batches zelf** komen uit een generator, zoals in [Generators: waarden één voor één maken](/course/open/py-generators?lang=nl), die steeds één slice van de data geeft. `range` met een derde getal telt in stappen van die grootte: `range(0, 10, 4)` geeft 0, 4 en 8. Een slice die voorbij het eind loopt, stopt daar gewoon, dus de laatste batch is kleiner, en geen voorbeeld valt weg:

        ```python
        def batches(X, y, size):
            for start in range(0, len(X), size):
                yield X[start:start + size], y[start:start + size]
        ```

        Trainen heeft dan twee loops: één over de epochs, en daarbinnen één over de mini-batches, met een stap voor elke mini-batch.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **mini-batch** | | een kleine groep voorbeelden voor één update-stap | `xb, yb` |
        | **batch size** | | hoeveel voorbeelden er in een mini-batch gaan | `size = 20` |
        | $B$ | B | de batch size in een formule | |
        | **iteratie** | | één update-stap op één mini-batch | |
        | **epoch** | | één keer door alle data, met elk voorbeeld één keer: $n / B$ iteraties, naar boven afgerond | `for epoch in range(3):` |
        | **SGD** | es gee dee | stochastic gradient descent: gradient descent met de hellingen van mini-batches | `torch.optim.SGD` |
        | `range(start, stop, step)` | | telt van `start` tot net onder `stop`, in stappen van `step` | `range(0, 10, 4)` |
        """),
        L.watch_out("Een kleinere batch is niet altijd beter. Met batches van 2 volgt elke stap de helling van twee voorbeelden, en die kan ver afwijken van de helling over alle data: de loss blijft springen en komt niet tot rust. Batch sizes van 32 tot 256 zijn gebruikelijk: veel stappen per epoch, en elke stap nog dicht bij de echte helling."),
        mo.md(r"""
        De figuur toont één run: een rechte lijn door 200 voorbeelden, $y = 2x + 1$ plus een willekeurige spreiding, met dezelfde beginweights, learning rate 0,1 en drie batch sizes.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_batches():
    mo.vstack([
        mo.Html(C.picture({"epochs": "Loss over alle 200 voorbeelden", "epoch": "epoch", "loss": "loss", "steps": "Loss van elke mini-batch van 20",
                           "step": "iteratie",
                           "alt": "Links: de loss over alle 200 voorbeelden na elke stap, 5 epochs lang, van 0 tot 2. Alle drie de batch sizes beginnen op 4,9, boven de rand. Met de volle batch van 200 daalt hij langzaam, één stap per epoch, naar 0,69; met mini-batches van 20 is hij 0,32 na één epoch en 0,28 na twee; met mini-batches van 2 daalt hij meteen, maar blijft hij springen tussen ongeveer 0,28 en 0,52. Rechts: de loss van elk van de eerste 30 mini-batches van 20, die op en neer springt terwijl hij daalt."})),
        mo.md(r"""
        Links volgt elke lijn de loss over alle 200 voorbeelden na elke stap, en is $B$ de batch size. Alle drie beginnen op 4,9, boven de rand van de figuur. De volle batch van 200 zet één stap per epoch, een stip op de streepjeslijn, en zit na 5 epochs nog op 0,69. Mini-batches van 20 zetten 10 stappen per epoch en zitten na twee op 0,28. Mini-batches van 2 zetten 100 stappen per epoch, maar elke stap volgt maar twee voorbeelden: de gevonden lijn blijft heen en weer gaan, en de loss blijft springen tussen ongeveer 0,28 en 0,52. Rechts springt de loss van elke mini-batch van 20 op en neer, omdat elke batch andere voorbeelden bevat, terwijl de trend omlaag gaat.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te trainen met mini-batches:

    1. Schrijf een generator `batches(X, y, size)` die de voorbeelden per `size` geeft, de laatste batch misschien kleiner.
    2. Maak het model en de optimizer, zoals eerder.
    3. Loop over de epochs, en daarbinnen over `batches(X, y, size)`.
    4. Voor elke mini-batch: zijn loss, `opt.zero_grad()`, `loss.backward()` en `opt.step()`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De data van de figuur, met mini-batches van 20, 2 epochs lang. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_batches():
    import torch
    from torch import nn

    _X, _y = C.data()   # 200 voorbeelden, shapes (200, 1) en (200, 1)

    # 1. de generator
    def _batches(X, y, size):
        for start in range(0, len(X), size):
            yield X[start:start + size], y[start:start + size]

    # 2. het model en de optimizer
    torch.manual_seed(1)
    _model = nn.Linear(1, 1)
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1)
    # 3. epochs, en mini-batches in elke epoch
    for _epoch in range(2):
        _steps = 0
        for _xb, _yb in _batches(_X, _y, 20):
            # 4. één stap voor deze mini-batch
            _loss = ((_model(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            _steps = _steps + 1
        print(_epoch, _steps, ((_model(_X) - _y) ** 2).mean().item())
    print(_model.weight.item(), _model.bias.item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke epoch heeft 10 stappen: 200 voorbeelden in mini-batches van 20. Na de eerste epoch is de loss over alle data ongeveer 0,32, na de tweede ongeveer 0,28. De gevonden lijn ligt dicht bij $y = 2x + 1$: een weight bij 2 en een bias term bij 1. Door de willekeurige spreiding in de data komt de loss niet op 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_last():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De laatste mini-batch
    """)
    return


@app.cell(hide_code=True)
def choice_last_cell():
    choice_last = L.choice(
        "last-batch",
        question="1.000 voorbeelden in mini-batches van 32, elk voorbeeld één keer. Hoeveel iteraties heeft één epoch, en hoe groot is de laatste mini-batch?",
        options={
            "floor": "31 iteraties, allemaal van 32 voorbeelden; de laatste 8 voorbeelden vallen weg",
            "right": "32 iteraties; de laatste mini-batch heeft 8 voorbeelden",
            "even": "32 iteraties van elk 31,25 voorbeelden",
        },
        correct="right",
        why={
            "floor": "Een slice voorbij het eind stopt daar gewoon, dus de laatste voorbeelden vormen een eigen, kleinere batch. Zie *De batches zelf* in *Het idee*.",
            "even": "Een mini-batch bevat hele voorbeelden. Wat gebeurt er met de voorbeelden die geen volle batch vullen? Zie *De batches zelf* in *Het idee*.",
            "right": "31 · 32 = 992, dus na 31 volle mini-batches blijven er 8 voorbeelden over, en die vormen de 32e. Elk voorbeeld wordt één keer gebruikt.",
        },
    )
    choice_last
    return (choice_last,)


@app.cell(hide_code=True)
def text_exercise_batches():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een generator voor mini-batches

        Maak `batches(X, y, size)` af: een generator die de voorbeelden op volgorde geeft, per `size`, elke keer een slice van `X` met dezelfde slice van `y`. Vervang de `...`. De laatste regels proberen hem op 10 voorbeelden met batches van 4, en bewaren de lengte van elke batch en zijn eerste voorbeeld en label. De check probeert ook batches van 3.
        """),
        L.task("Laat `batches` de data geven in mini-batches van `size`, de laatste misschien kleiner."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batches():
    box_batches = L.code_box("batches", lines=15)
    box_batches
    return (box_batches,)


@app.cell(hide_code=True)
def check_batches(box_batches):
    result_batches = L.check("batches", box_batches.value)
    result_batches
    return (result_batches,)


@app.cell(hide_code=True)
def hints_batches():
    L.hints(
        "batches",
        [
            "Stap 1 van het recept. Elke batch is een slice van `X` en dezelfde slice van `y`, één voor één afgegeven.",
            "Een loop over het begin van elke batch: 0, dan `size`, dan twee keer `size`, tot `len(X)`. Elke ronde geeft met `yield` twee slices.",
            "`for start in range(0, len(X), ___):` en `yield X[start:start + size], y[___]`.",
        ],
        why="Met 10 voorbeelden en `size` 4 is `start` 0, 4 en 8. De slices zijn de rijen 0 tot en met 3, 4 tot en met 7 en 8 tot en met 9: de laatste loopt voorbij het eind en stopt daar.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_epochs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Drie epochs met mini-batches

        De data van de figuur, een model, een optimizer en de twee loops staan er al. Vervang de `...` door stap 4 van het recept: één update-stap voor elke mini-batch. De laatste regel is de loss over alle 200 voorbeelden na 3 epochs, als Python-getal. De check probeert ook een andere learning rate.
        """),
        L.task("Geef elke mini-batch zijn eigen update-stap."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epochs():
    box_epochs = L.code_box("epochs", lines=22)
    box_epochs
    return (box_epochs,)


@app.cell(hide_code=True)
def check_epochs(box_epochs):
    result_epochs = L.check("epochs", box_epochs.value)
    result_epochs
    return (result_epochs,)


@app.cell(hide_code=True)
def hints_epochs():
    L.hints(
        "epochs",
        [
            "Stap 4 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De loss gebruikt de mini-batch, `xb` en `yb`, niet heel `X` en `y`. Dan de drie aanroepen van de optimizer.",
            "`loss = ((model(___) - ___) ** 2).mean()`, dan `opt.zero_grad()`, `loss.backward()` en `opt.step()`.",
        ],
        why="3 epochs van 10 mini-batches geven 30 stappen, elk met de hellingen van zijn eigen 20 voorbeelden. De loss over alle data eindigt op ongeveer 0,2821, zoals in de figuur.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De training loop uit [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl) loopt nu per mini-batch: een generator zoals in [Generators: waarden één voor één maken](/course/open/py-generators?lang=nl) geeft de batches, en een epoch is één keer door ze allemaal. Zo wordt de gradient descent van de optimizer stochastic gradient descent. In de volgende les, *Datasets en DataLoaders*, geeft PyTorch de mini-batches voor je, en shufflet hij de voorbeelden voor elke epoch.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_last, result_batches, result_epochs):
    L.wrap_up(choice_last, result_batches, result_epochs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
