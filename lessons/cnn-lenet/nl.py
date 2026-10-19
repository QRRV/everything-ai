# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-lenet. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-lenet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="Hoeveel parameters heeft `nn.Conv2d(3, 4, 5)`?",
        options={
            "bias": "300: 4 filters van 3 × 5 × 5, zonder bias terms",
            "channels": "104: 4 filters van 5 × 5, één per output channel, en hun bias terms",
            "right": "304: 4 filters van 3 × 5 × 5, en elk een bias term",
        },
        correct="right",
        why={
            "bias": "Elk filter heeft ook een bias term: 4 meer. Zie [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl).",
            "channels": "Een filter beslaat elk input channel: 3 × 5 × 5 = 75 weights, geen 25. Zie [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl).",
            "right": "4 × 3 × 5 × 5 = 300 weights, en 4 bias terms. Vandaag tel je zo een heel netwerk uit 1998.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="De feature maps van een mini-batch hebben shape (4, 16, 3, 3). Welke shape geeft `x.reshape(len(x), -1)`?",
        options={
            "right": "(4, 144): één rij per afbeelding",
            "all": "(576,): één rij voor de hele mini-batch",
            "channels": "(4, 16, 9): één rij per channel",
        },
        correct="right",
        why={
            "all": "Dat is `x.reshape(-1)`. `len(x)` houdt de 4 afbeeldingen uit elkaar. Zie [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl).",
            "channels": "`-1` neemt alles na de eerste as samen: 16 × 3 × 3 = 144 getallen per afbeelding. Zie [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl).",
            "right": "16 × 3 × 3 = 144 getallen per afbeelding, voor de linear layers erna. LeNet-5 doet hetzelfde.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je SmallCNN heeft twee blokken van convolutie en pooling, en daarna een output head. Die vorm is oud. In 1998 beschreven Yann LeCun, Léon Bottou, Yoshua Bengio en Patrick Haffner **LeNet-5**: een CNN van twee zulke blokken en drie linear layers dat handgeschreven cijfers las. Een systeem dat erop gebouwd was, las elke dag enkele miljoenen cheques. Latere CNN's, zoals VGG en ResNet uit de volgende twee extra lessen, zijn dezelfde blokken, dieper gestapeld, met een paar nieuwe trucs. In deze stap van je project bouw je LeNet-5 na in PyTorch, en train je het tegen je SmallCNN.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_better():
    guess_better = L.guess(
        "better",
        question="LeNet-5 heeft 61.706 parameters, bijna 7 keer de 9.098 van je SmallCNN. Allebei trainen ze 5 epochs met je Trainer, op dezelfde 6.000 afbeeldingen. Welke leest meer van de 1.000 testafbeeldingen goed?",
        options={
            "lenet": "LeNet-5",
            "right": "Je SmallCNN",
            "same": "Allebei ongeveer evenveel",
        },
        reveal="**Je SmallCNN.** Over 5 seeds leest LeNet-5 met de code van de course er 828 tot 852 van de 1.000 goed, en SmallCNN 864 tot 879: elke keer meer, met een zevende van de parameters. LeNet-5 heeft de tanh en de average pooling van zijn tijd; met ReLU en max pooling, die het later overnamen, leest hetzelfde netwerk er 845 tot 866 goed: dichterbij, en bij elke seed nog steeds minder dan SmallCNN. Meer parameters alleen maken geen beter model.",
    )
    guess_better
    return (guess_better,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        LeNet-5 heeft de vorm van je SmallCNN, met andere maten en twee oudere keuzes.

        **Twee blokken.** Blok 1 is een convolutie van 5 bij 5 van 1 naar 6 channels, blok 2 een van 5 bij 5 van 6 naar 16 channels; na elk komen tanh en pooling van 2 bij 2. De paper gaf LeNet-5 afbeeldingen van 32 bij 32; de 28 bij 28 van Fashion-MNIST met padding 2 komt op hetzelfde neer. Blok 2 heeft geen padding, dus zijn feature maps krimpen van 14 bij 14 naar 10 bij 10.

        **Drie linear layers.** Eén rij getallen per afbeelding gaat `fc1` in, naar 120 getallen, dan `fc2`, naar 84, elk met tanh erna, en dan de output head, naar de 10 logits.

        **tanh**, waar je SmallCNN ReLU heeft: de S-vormige kromme van −1 tot 1 uit [Initialisatie](/course/open/nn-initialization?lang=nl). Getallen bij 0 veranderen weinig, en grote komen uit bij −1 of 1. **Average pooling**, waar je SmallCNN max pooling heeft: van elk blok van 2 bij 2 houdt het het gemiddelde, niet het grootste getal. Van 1, 2, 3 en 6 houdt het 3, waar max pooling 6 houdt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **LeNet-5** | | het CNN van LeCun en anderen (1998): twee blokken van convolutie, tanh en pooling, en drie linear layers | |
        | **tanh** | tan h | een functie die elk getal tussen −1 en 1 perst; 0 blijft 0 | `torch.tanh(x)` |
        | `nn.AvgPool2d` | n n average pool 2 d | average pooling van elk channel, in blokken van $k$ bij $k$ | `nn.AvgPool2d(2)` |
        """),
        L.watch_out("De LeNet-5 van de paper verschilt in details van de versie die jij bouwt, en van de meeste nagebouwde versies: zijn pooling had per channel een eigen weight en bias term, niet elke feature map van blok 2 keek naar alle 6 van blok 1, zijn tanh was geschaald, en zijn output layer mat afstanden tot vaste patronen in plaats van logits te geven. De blokken, de maten en het aantal parameters, ongeveer 60.000, zijn hetzelfde."),
        mo.md(r"""
        De figuur: één afbeelding op weg door LeNet-5, met zijn shape na elke stap. Elke stapel heeft een vierkant voor elk channel, hooguit 6 getekend, zo groot als zijn hoogte en breedte; een rij getallen heeft een vakje voor elk getal, hooguit 10 getekend, met puntjes voor de rest. In een mini-batch krijgt elke shape $B$ ervoor.
        """),
        C.shapes_picture({"title": "Eén afbeelding door LeNet-5", "input": "één grijze afbeelding", "maps": "{0} feature maps van {1} bij {1}",
                          "fc": "{0} getallen", "logits": "10 logits, één per soort",
                          "conv1": "conv 5×5 met padding 2, tanh", "conv2": "conv 5×5 zonder padding, tanh",
                          "pool": "average pooling 2 bij 2", "flat": "één rij van 400, fc1, tanh", "linear": "fc2, tanh",
                          "head": "de output head",
                          "alt": "Een grijze afbeelding van 1 bij 28 bij 28 wordt 6 feature maps van 28 bij 28 na een convolutie van 5 bij 5 met padding 2 en tanh, 6 van 14 bij 14 na average pooling, 16 van 10 bij 10 na een convolutie zonder padding en tanh, en 16 van 5 bij 5 na pooling. Eén rij van 400 getallen wordt 120 getallen na fc1 en tanh, 84 na fc2 en tanh, en 10 logits na de output head."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    LeNet-5 voor afbeeldingen van 28 bij 28 met één channel:

    1. Blok 1: een convolutie van 5 bij 5 van 1 naar 6 channels met padding 2, tanh, en average pooling van 2 bij 2: $(B, 6, 14, 14)$.
    2. Blok 2: een convolutie van 5 bij 5 van 6 naar 16 channels zonder padding, tanh, en average pooling: $(B, 16, 5, 5)$.
    3. Eén rij per afbeelding: `x.reshape(len(x), -1)`, 16 × 5 × 5 = 400 getallen.
    4. `fc1` van 400 naar 120 en `fc2` van 120 naar 84, elk met tanh erna, en de output head van 84 naar 10: de logits.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst tanh op vijf getallen, dan stap 1 van het recept op de eerste 2 trainingsafbeeldingen. Het commentaar met `#` zegt welk deel wat is. De pooling is nog niet gedaan: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_block():
    import torch
    from torch import nn

    # tanh: elk getal tussen -1 en 1
    print(torch.tanh(torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])))
    _X, _y, _, _ = C.fashion()
    _x = _X[:2]
    # 1. blok 1: convolutie met padding 2, tanh, average pooling
    torch.manual_seed(0)
    _conv = nn.Conv2d(1, 6, 5, padding=2)
    _pool = nn.AvgPool2d(2)
    _h = torch.tanh(_conv(_x))
    print(_h.shape)
    _h = _h  # maak af: _pool(_h)
    print(_h.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    tanh maakt 0,5 tot 0,4621, iets kleiner, en 3 tot 0,9951, bijna 1; negatieve getallen gaan net zo, onder 0. Met padding 2 houdt de convolutie de 28 bij 28: (2, 6, 28, 28). Zoals de cel nu is, is de tweede shape weer dezelfde. Afgemaakt halveert de average pooling de hoogte en breedte: (2, 6, 14, 14), wat blok 2 krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een andere afbeeldingsgrootte
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "lenet-shape",
        question="Je geeft je LeNet-5 grijze afbeeldingen van 32 bij 32 in plaats van 28 bij 28, met dezelfde layers en padding 2 in blok 1. Wat moet `fc1` worden?",
        options={
            "right": "`nn.Linear(16 * 6 * 6, 120)`: 6 bij 6 na blok 2",
            "same": "`nn.Linear(16 * 5 * 5, 120)`: de layers ervoor hangen niet af van de grootte van de afbeelding",
            "quarter": "`nn.Linear(16 * 8 * 8, 120)`: twee keer pooling maakt er een kwart van, 32 / 4",
        },
        correct="right",
        why={
            "same": "Hun weights hangen er niet van af, maar hun output wel: 32 blijft 32 in blok 1 en wordt 16 na pooling, 12 na de convolutie van blok 2, en 6 na pooling.",
            "quarter": "De pooling halveert twee keer, maar de convolutie van blok 2, zonder padding, haalt er eerst 4 af: 16 − 4 = 12, dan 6.",
            "right": "32, dan 16 na pooling, 12 na de convolutie van blok 2, 6 na pooling: 16 × 6 × 6 = 576 getallen per afbeelding. De convoluties werken op elke grootte; alleen `fc1` moet die kennen.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_lenet():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. LeNet-5 in PyTorch

        Maak `LeNet5` af: in `__init__` zijn layers, en in `forward` stap 1 tot en met 4 van het recept, voor een mini-batch `x` met shape $(B, 1, 28, 28)$; het geeft de logits, met shape $(B, 10)$. Maak de layers met weights in de volgorde waarin de data erdoor gaat: `conv1`, `conv2`, `fc1`, `fc2`, `out`. De check maakt het netwerk na `torch.manual_seed(0)` en vergelijkt zijn logits voor 2 en voor 3 willekeurige afbeeldingen, en nog een keer met al zijn parameters 5 keer zo groot. Als het klopt, bewaart de check het in je project, als `work/projects/cnn/lenet.py`.
        """),
        L.task("Laat `LeNet5` 10 logits per afbeelding geven, uit twee blokken van convolutie, tanh en average pooling, en drie linear layers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lenet():
    box_lenet = L.code_box("lenet5", lines=22)
    box_lenet
    return (box_lenet,)


@app.cell(hide_code=True)
def check_lenet(box_lenet):
    result_lenet = L.check("lenet5", box_lenet.value)
    result_lenet
    return (result_lenet,)


@app.cell(hide_code=True)
def hints_lenet():
    L.hints(
        "lenet5",
        [
            "In `__init__` zes attributen van `self`: `conv1` en `conv2` als `nn.Conv2d`, één `nn.AvgPool2d(2)` die beide blokken delen, en `fc1`, `fc2` en `out` als `nn.Linear`. Het commentaar in het vak geeft hun maten.",
            "In `forward` is elk blok één regel: de convolutie, `torch.tanh` eromheen, en de pooling daar weer omheen: de twee regels van het uitgewerkte voorbeeld in één. Na de twee blokken één rij per afbeelding, zoals in je SmallCNN; dan `fc1` en `fc2`, elk binnen `torch.tanh`, en de output head zonder.",
        ],
        why="tanh komt na elke layer behalve de output head: de logits gaan naar de cross-entropy loss, die de softmax zelf doet. Zonder tanh ertussen zouden `fc1` en `fc2` samen één linear layer zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Oefening 3. Waar de parameters zitten
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where-params",
        question="Waar zitten de meeste van de 61.706 parameters van LeNet-5?",
        options={
            "conv": "In `conv1` en `conv2`: een filter voor elk channel",
            "head": "In `out`: een weight voor elke soort en elk getal van `fc2`",
            "right": "In `fc1`: bijna 4 van elke 5, een weight voor elk getal van de rij en elk neuron",
        },
        correct="right",
        why={
            "conv": "`conv1` heeft er 6 × 25 + 6 = 156 en `conv2` 16 × 6 × 25 + 16 = 2.416: een filter van 5 bij 5 heeft 25 weights per input channel, en dient elke plek van de afbeelding.",
            "head": "`out` heeft er 84 × 10 + 10 = 850: een kleine layer aan het eind.",
            "right": "`fc1` heeft er 400 × 120 + 120 = 48.120: een weight voor elk getal van de rij van 400 en elk van zijn 120 neuronen, zoals in een MLP. De twee convoluties hebben er samen 2.572.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij traint je `LeNet5`, en je `SmallCNN` uit [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl), met je Trainer: {C.EPOCHS} epochs op de 6.000 trainingsafbeeldingen, een learning rate van {C.number(C.LR, 2, L.lang)} met een warmup van één epoch, voor allebei hetzelfde, elk één run, met seed 0. Hij telt hun parameters met je `count_params`. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Het werkt zodra oefening 2 af is, en duurt ongeveer 10 seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_lenet(result_lenet):
    result_lenet  # run again when exercise 2 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 af: dan staat je `LeNet5` in je project, en gebruikt deze cel hem.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je project doet er hier te lang over: meer dan twee minuten om één netwerk te trainen. Doet je `LeNet5` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "lenet": "*Je `LeNet5` in `work/projects/cnn/lenet.py` stopt met een fout, bij het maken, tellen of trainen. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
            "trainer": "*Je Trainer in `work/projects/nn/trainer.py`, je `SmallCNN` in `work/projects/cnn/small_cnn.py` of je `count_params` in `work/projects/cnn/count.py` stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
        }[_result])
    else:
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _out = mo.md(f"""Van de 1.000 testafbeeldingen leest je LeNet-5 er **{_result['lenet']}** goed, met {C.grouped(_result['params_lenet'], L.lang)} parameters, geteld met je `count_params`; je SmallCNN leest er **{_result['small']}** goed, met {C.grouped(_result['params_small'], L.lang)}. Over 5 seeds leest LeNet-5 met de code van de course er 828 tot 852 goed, en SmallCNN 864 tot 879.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De blokken van je SmallCNN uit [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl), zoals LeCun en anderen ze in 1998 samenstelden, met de tanh van hun tijd en de average pooling uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl). De meeste parameters zitten in een linear layer, zoals in een MLP; de convoluties hebben er weinig nodig, omdat een filter over de hele afbeelding gedeeld wordt. Je project `cnn` heeft nu `lenet.py` naast `small_cnn.py`. De volgende extra les, *VGG: dieper met kleine filters*, stapelt veel convoluties van 3 bij 3 tot een dieper netwerk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_lenet, choice_where):
    L.wrap_up(choice_shape, result_lenet, choice_where)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
