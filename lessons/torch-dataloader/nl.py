# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-dataloader. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-dataloader", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Mini-batches en epochs](/course/open/torch-batches?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_epoch():
    quick_epoch = L.choice(
        "quick-epoch",
        question="Wat is het verschil tussen een iteratie en een epoch?",
        options={
            "same": "Geen: twee woorden voor één keer door de data",
            "swapped": "Een iteratie gaat door alle data; een epoch is één stap",
            "right": "Een iteratie is één stap op één mini-batch; een epoch één keer door alle data",
        },
        correct="right",
        why={
            "same": "Een epoch bevat veel iteraties: één voor elke mini-batch. Zie *Het idee* in [Mini-batches en epochs](/course/open/torch-batches?lang=nl).",
            "swapped": "Het is andersom. Zie de nieuwe woorden in [Mini-batches en epochs](/course/open/torch-batches?lang=nl).",
            "right": "Eén epoch is zoveel iteraties als er mini-batches zijn: 200 voorbeelden in batches van 20 geven er 10.",
        },
    )
    quick_epoch
    return (quick_epoch,)


@app.cell(hide_code=True)
def quick_check_used():
    quick_used = L.choice(
        "quick-used",
        question="`X` en `y` bevatten 200 voorbeelden, en `batches` is de generator uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl). Wat laat dit zien?\n\n```python\ng = batches(X, y, 20)\nfirst = 0\nfor xb, yb in g:\n    first = first + 1\nsecond = 0\nfor xb, yb in g:\n    second = second + 1\nprint(first, second)\n```",
        options={
            "twice": "`10 10`",
            "right": "`10 0`",
            "error": "Niets: Python stopt bij de tweede loop",
        },
        correct="right",
        why={
            "twice": "`g` is één generator. Na de eerste loop heeft hij al zijn batches gegeven, en een generator is op na één keer doorlopen. Zie de let-op in [Generators: waarden één voor één maken](/course/open/py-generators?lang=nl).",
            "error": "Een generator die op is, geeft geen fout: een loop eroverheen draait gewoon nul keer.",
            "right": "De eerste loop neemt alle 10 batches. Daarna is `g` op, dus de tweede loop krijgt niets. Voor een nieuwe epoch moet `batches(X, y, 20)` opnieuw aangeroepen worden.",
        },
    )
    quick_used
    return (quick_used,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je generator uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl) werkt voor data die in twee tensors past, en hij geeft de batches elke epoch in dezelfde volgorde. Echte data is vaak een map met duizenden afbeeldingen, of een bestand dat je stukje voor stukje leest. En de volgorde doet ertoe: data is vaak gesorteerd, op cijfer of op datum. PyTorch splitst het werk in tweeën. Een **Dataset** weet hoeveel voorbeelden er zijn en hoe je voorbeeld nummer $i$ krijgt. Een **DataLoader** maakt er mini-batches van, elke epoch in een nieuwe willekeurige volgorde.

    **Een voorbeeld uit het dagelijks leven.** Een nieuw pak kaarten zit op volgorde: eerst alle harten, dan de ruiten, enzovoort. Voor elk potje schud je het pak, en dan deel je er handen uit. Het pak bewaart alleen de kaarten, elk op een eigen plek; wie deelt, maakt de handen, voor elk potje uit een pas geschud pak.
    """)
    return


@app.cell(hide_code=True)
def predict_sorted():
    guess_sorted = L.guess(
        "sorted",
        question="Stel dat de 6.000 cijfers gesorteerd zijn op label: eerst alle 592 nullen, dan de enen, enzovoort. Mini-batches van 64 in die volgorde, zonder shuffling. Wat denk je dat de eerste 9 mini-batches het model leren?",
        options={
            "sample": "Een beetje van elk cijfer: elke mini-batch is een steekproef uit de data",
            "right": "Alleen dat het antwoord 0 is: ze bevatten niets dan nullen",
            "nothing": "Nog niets: een model leert pas aan het eind van een epoch",
        },
        reveal="**Dat het antwoord altijd 0 is.** De 592 nullen vullen de eerste 9 mini-batches van 64 (9 · 64 = 576), dus 9 stappen lang trekt elke helling richting \"altijd 0\". Dan komen alleen enen, en trekken de stappen richting \"altijd 1\", waarmee ze de stappen ervoor ongedaan maken. **Shuffling** mengt de cijfers in elke mini-batch, zodat de helling van elke stap een eerlijke schatting is van de helling over alle cijfers.",
    )
    guess_sorted
    return (guess_sorted,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een Dataset** is een class die voortbouwt op `Dataset`, uit `torch.utils.data`, met twee special methods, zoals in [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl):

        - **`__len__(self)`** geeft het aantal voorbeelden terug: `len(ds)` voert hem uit.
        - **`__getitem__(self, i)`** geeft voorbeeld `i` terug, als de input en zijn label: `ds[i]` voert hem uit.

        Zijn `__init__` zet de data klaar, bijvoorbeeld als tensors. Data die al in twee tensors staat, met de voorbeelden langs de eerste axis, heeft geen eigen class nodig: `TensorDataset(X, y)` is zo'n Dataset.

        **Een DataLoader** maakt mini-batches van een Dataset: `loader = DataLoader(ds, batch_size=64, shuffle=True)`. Elke `for xb, yb in loader:` is één epoch. De DataLoader vraagt de Dataset om voorbeelden, één voor één, en stapelt ze tot mini-batches: `xb` krijgt de batch als eerste axis. Met `shuffle=True` **shufflet** hij ze: elke epoch een nieuwe willekeurige volgorde, dus elke mini-batch is een nieuwe mix. De laatste mini-batch kan kleiner zijn, en `len(loader)` is het aantal mini-batches. Anders dan een generator raakt een DataLoader niet op: elke nieuwe loop eroverheen is een nieuwe epoch. Shuffling gebruikt de willekeurige getallen van PyTorch, dus met `torch.manual_seed` vóór de loops shufflet hij bij elke run hetzelfde.

        | Je generator | Een Dataset en een DataLoader |
        |---|---|
        | `batches(X, y, 20)`, voor elke epoch opnieuw aangeroepen | `DataLoader(TensorDataset(X, y), batch_size=20)`, één keer gemaakt |
        | elke epoch dezelfde volgorde | `shuffle=True`: elke epoch een nieuwe volgorde |
        | heeft de data in twee tensors nodig | data in elke vorm, via `__len__` en `__getitem__` |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Dataset** | | een class die zegt hoeveel voorbeelden er zijn en voorbeeld `i` geeft | `class Digits(Dataset):` |
        | `__len__` | dunder len | de special method achter `len(ds)`: het aantal voorbeelden | `def __len__(self):` |
        | `__getitem__` | dunder get item | de special method achter `ds[i]`: voorbeeld `i`, als input en label | `def __getitem__(self, i):` |
        | `TensorDataset` | | een Dataset gemaakt van tensors waarvan de eerste axis over de voorbeelden loopt | `TensorDataset(X, y)` |
        | **DataLoader** | data loader | maakt mini-batches van een Dataset; elke loop eroverheen is een epoch | `DataLoader(ds, batch_size=64, shuffle=True)` |
        | **shuffling** | | de voorbeelden elke epoch in een nieuwe willekeurige volgorde zetten; het werkwoord is shufflen | `shuffle=True` |
        """),
        L.watch_out("`ds[i]` is één voorbeeld: een afbeelding met shape (784,) en zijn label. Een mini-batch uit de DataLoader heeft vooraan nog een axis, de voorbeelden: `xb` heeft shape (64, 784), en `yb` shape (64,). Een model krijgt mini-batches, dus het ziet altijd die extra axis."),
        mo.md(r"""
        De 6.000 cijfers gesorteerd op label, zoals in de gok, en de labels van de eerste mini-batch van 64, de eerste 8 ervan: zonder shuffling, en in twee epochs van een DataLoader die shufflet, na `torch.manual_seed(0)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_shuffle():
    _grids = C.picture(["gesorteerd", "epoch 1", "epoch 2"], ["alleen nullen, elke epoch", "een willekeurige mix", "een nieuwe mix"])
    mo.vstack([draw.array_grids(_grids[:1], title="De eerste labels van de eerste mini-batch"),
               draw.array_grids(_grids[1:2]), draw.array_grids(_grids[2:])])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met een DataLoader data uit te delen:

    1. Maak een Dataset: een class die voortbouwt op `Dataset`, met `__init__`, `__len__` en `__getitem__`, of `TensorDataset(X, y)` voor data in tensors.
    2. Maak de DataLoader één keer: `loader = DataLoader(ds, batch_size=..., shuffle=True)`.
    3. Roep `torch.manual_seed(...)` aan vóór de loops, zodat de DataLoader bij elke run hetzelfde shufflet.
    4. Loop over de epochs, en in elke epoch over `for xb, yb in loader:`, met een update-stap voor elke mini-batch.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De handgeschreven cijfers uit [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl): 6.000 trainingsafbeeldingen van QMNIST in `data/qmnist-small.npz`, elk van 28 bij 28 pixels als 784 getallen van 0 tot 255, met zijn cijfer als label. Een Dataset daarvoor, en een loop die in 2 epochs de mini-batches van een DataLoader telt, nog zonder te trainen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_dataloader():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    # 1. een Dataset voor de cijfers
    class _Digits(Dataset):
        def __init__(self, images, labels):
            self.images = torch.tensor(images, dtype=torch.float32) / 255
            self.labels = torch.tensor(labels, dtype=torch.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return self.images[i], self.labels[i]

    _data = np.load("data/qmnist-small.npz")
    _ds = _Digits(_data["train_images"], _data["train_labels"])
    print(len(_ds), _ds[0][0].shape, _ds[0][1])
    # 2. de DataLoader, één keer gemaakt
    _loader = DataLoader(_ds, batch_size=64, shuffle=True)
    print(len(_loader))
    # 3. bij elke run dezelfde shuffling
    torch.manual_seed(0)
    # 4. epochs, en mini-batches in elke epoch
    for _epoch in range(2):
        _batches = 0
        for _xb, _yb in _loader:
            _batches = _batches + 1
        print(_epoch, _batches, _xb.shape, _yb.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De Dataset bevat 6.000 voorbeelden; voorbeeld 0 is een afbeelding van 784 getallen, van het cijfer 5. De DataLoader maakt per epoch 94 mini-batches van 64: 93 volle en een laatste van 48, omdat 93 · 64 = 5.952. De loop eindigt met die laatste mini-batch, dus `_xb` heeft shape (48, 784) en `_yb` shape (48,). De tweede epoch gaat opnieuw over dezelfde DataLoader, in een nieuwe volgorde, zonder iets nieuws te maken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example_training():
    mo.md(r"""
    Hetzelfde recept voor data die al in twee tensors staat, nu met een update-stap voor elke mini-batch: de 200 voorbeelden en de lijn uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl), 2 epochs lang. Voer de cel uit.
    """)
    return


@app.cell
def example_training(DataLoader, torch):
    from torch import nn
    from torch.utils.data import TensorDataset

    _X, _y = C.data()   # de 200 voorbeelden van de vorige les, shapes (200, 1) en (200, 1)
    # 1. een TensorDataset: de data staat al in twee tensors
    _ds = TensorDataset(_X, _y)
    # 2. de DataLoader, één keer gemaakt
    _loader = DataLoader(_ds, batch_size=20, shuffle=True)
    torch.manual_seed(1)
    _model = nn.Linear(1, 1)
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1)
    # 3. bij elke run dezelfde shuffling
    torch.manual_seed(0)
    # 4. epochs, en een update-stap voor elke mini-batch
    for _epoch in range(2):
        for _xb, _yb in _loader:
            _loss = ((_model(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
        print(_epoch, ((_model(_X) - _y) ** 2).mean().item())
    return


@app.cell(hide_code=True)
def text_after_training():
    mo.md(r"""
    Elke epoch zet 10 stappen, één voor elke geshufflede mini-batch van 20. Na de eerste epoch is de loss over alle data ongeveer 0,33, na de tweede ongeveer 0,28. De stap bestaat uit dezelfde vier regels als in [Mini-batches en epochs](/course/open/torch-batches?lang=nl); alleen waar de mini-batches vandaan komen, is veranderd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reshuffle():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat shuffling verandert
    """)
    return


@app.cell(hide_code=True)
def choice_reshuffle_cell():
    choice_reshuffle = L.choice(
        "reshuffle",
        question="`loader = DataLoader(ds, batch_size=64, shuffle=True)`. Wat verandert er van de ene epoch naar de volgende?",
        options={
            "right": "Welke voorbeelden samen in een mini-batch zitten",
            "examples": "De voorbeelden zelf: elke epoch krijgt andere afbeeldingen",
            "once": "Na de eerste epoch niets meer: de volgorde wordt één keer getrokken",
        },
        correct="right",
        why={
            "examples": "De Dataset blijft hetzelfde; elke epoch heeft nog steeds elk van zijn voorbeelden één keer. Zie *Een DataLoader* in *Het idee*.",
            "once": "Elke loop over de DataLoader trekt een nieuwe volgorde. Kijk naar epoch 1 en epoch 2 in de figuur.",
            "right": "Elke epoch heeft alle 6.000 voorbeelden één keer, in een nieuwe willekeurige volgorde, dus elke mini-batch is een nieuwe mix.",
        },
    )
    choice_reshuffle
    return (choice_reshuffle,)


@app.cell(hide_code=True)
def text_exercise_digits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een Dataset voor de cijfers

        Maak de class `Digits` af: `__init__` maakt al tensors van de afbeeldingen en labels. Vervang de twee `...`: `__len__` geeft het aantal voorbeelden terug, en `__getitem__` de afbeelding en het label van voorbeeld `i`, in die volgorde. De laatste regels nemen voorbeeld 7 en laten het aantal voorbeelden zien, de lengte van de afbeelding, zijn cijfer en de som van zijn pixels.
        """),
        L.task("Maak van `Digits` een Dataset die zijn lengte kent en voorbeeld `i` geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_digits():
    box_digits = L.code_box("digits", lines=20)
    box_digits
    return (box_digits,)


@app.cell(hide_code=True)
def check_digits(box_digits):
    result_digits = L.check("digits", box_digits.value)
    result_digits
    return (result_digits,)


@app.cell(hide_code=True)
def hints_digits():
    L.hints(
        "digits",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld.",
            "`__len__` geeft de lengte terug van een van de twee tensors die `__init__` maakte. `__getitem__` geeft twee dingen terug, gescheiden door een komma.",
            "`return len(self.___)` en `return self.images[i], ___`.",
        ],
        why="`self.labels` bevat één label per voorbeeld, dus zijn lengte is 6.000. Voorbeeld 7 is rij 7 van `self.images` en nummer 7 van `self.labels`: een afbeelding van het cijfer 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shuffled():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen met een DataLoader

        De data en het model uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl), met een optimizer. Vervang de twee `...`: `loader` is een DataLoader die shufflet, over `TensorDataset(X, y)`, met mini-batches van 20, en elke epoch loopt erover met een update-stap voor elke mini-batch. De laatste regel is de loss over alle data na 3 epochs. De check probeert ook een andere learning rate.
        """),
        L.task("Train 3 epochs met een DataLoader die shufflet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shuffled():
    box_shuffled = L.code_box("shuffled", lines=20)
    box_shuffled
    return (box_shuffled,)


@app.cell(hide_code=True)
def check_shuffled(box_shuffled):
    result_shuffled = L.check("shuffled", box_shuffled.value)
    result_shuffled
    return (result_shuffled,)


@app.cell(hide_code=True)
def hints_shuffled():
    L.hints(
        "shuffled",
        [
            "Stap 1, 2 en 4 van het recept, zoals in het tweede deel van het uitgewerkte voorbeeld; de seed van stap 3 staat er al.",
            "De DataLoader krijgt de Dataset, de batch size en `shuffle`. In de loop over de epochs neemt een loop over `loader` de plaats in van de generator uit de vorige les.",
            "`loader = DataLoader(TensorDataset(X, y), batch_size=___, shuffle=___)` en `for xb, yb in ___:` met de vier regels van een update-stap.",
        ],
        why="Elke epoch geeft de DataLoader de 200 voorbeelden in 10 geshufflede mini-batches van 20, elk met een eigen stap. Na 3 epochs is de loss ongeveer 0,279.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een Dataset gebruikt special methods, zoals in [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl): `len(ds)` en `ds[i]`. Een DataLoader neemt de taak van de generator uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl) over, en shufflet, zodat elke epoch een nieuwe mix is en de descent echt stochastic is. In de volgende les, *Train mode en eval mode*, zet je een model om van leren naar alleen voorspellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_reshuffle, result_digits, result_shuffled):
    L.wrap_up(choice_reshuffle, result_digits, result_shuffled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
