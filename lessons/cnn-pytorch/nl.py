# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-pytorch. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-pytorch", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Convolutie als één grote matrixvermenigvuldiging](/course/open/cnn-conv-numpy?lang=nl) en [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_im2col():
    quick_im2col = L.choice(
        "quick-im2col",
        question="Met im2col wordt een convolutional layer één matrixvermenigvuldiging. Wat staat er in één rij van de matrix $\\mathbf{P}$ met de patches?",
        options={
            "channel": "Eén channel van de hele afbeelding, plat neergelegd",
            "right": "De patch op één plek, uit elk channel",
            "filter": "Eén filter, met al zijn weights plat neergelegd",
        },
        correct="right",
        why={
            "channel": "Een rij is waar één filter op één plek naar kijkt: een kleine patch, maar uit elk channel. Zie [Convolutie als één grote matrixvermenigvuldiging](/course/open/cnn-conv-numpy?lang=nl).",
            "filter": "De filters zijn de rijen van $\\mathbf{W}$, de andere matrix. Zie [Convolutie als één grote matrixvermenigvuldiging](/course/open/cnn-conv-numpy?lang=nl).",
            "right": "En de filters zijn de rijen van $\\mathbf{W}$. Vandaag doet `nn.Conv2d` van PyTorch dit allemaal voor je, en snel.",
        },
    )
    quick_im2col
    return (quick_im2col,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Je traint een nieuw netwerk op maar één mini-batch, bij elke stap dezelfde 64 afbeeldingen, een paar honderd stappen lang. Wat doet zijn loss als het netwerk en zijn trainingsstap werken?",
        options={
            "right": "Die gaat naar bijna 0: het netwerk leert de 64 afbeeldingen uit het hoofd",
            "guess": "Die blijft rond 2,30: bij 10 soorten is dat de loss die je verwacht",
            "up": "Die stijgt: trainen op één mini-batch laat het overfitten",
        },
        correct="right",
        why={
            "guess": "2,30 is de loss van gokken tussen 10 soorten: een netwerk dat daar blijft, heeft niets geleerd. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "up": "Overfitten is precies wat de test wil: op de batch zelf gaat de loss omlaag. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "right": "Dat is de overfit-test, en je `overfit_one_batch` staat in je project. Vandaag test hij je eerste CNN.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je diepe MLP uit [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl) leest ongeveer 85% van de kleding van Fashion-MNIST goed. Maar het ziet een afbeelding als een rij van 784 getallen waarin buren niets betekenen: schud de pixels van elke afbeelding op dezelfde manier door elkaar, en het leert net zo goed. En elk van de 64 neuronen in zijn eerste layer heeft voor elke pixel een eigen weight. Een CNN maakt gebruik van hoe een afbeelding in elkaar zit: zijn filters kijken naar pixels samen met hun buren, en hetzelfde filter schuift over de hele afbeelding. In dit project bouw je je eerste CNN in PyTorch en train je het met je eigen Trainer. Het is de eerste stap van je eigen CNN; de volgende drie zijn extra lessen van de tak computer vision.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken"],
                           0, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params",
        question="Je diepe MLP heeft 85.194 parameters. Het CNN van vandaag heeft twee convolutional layers, met 8 en dan 16 filters van 3 bij 3, en een output head die 784 getallen per afbeelding krijgt. Hoeveel parameters heeft het?",
        options={
            "same": "Ongeveer 85.000: evenveel als je diepe MLP",
            "more": "Ongeveer 300.000: twee layers met veel channels",
            "right": "Ongeveer 9.000: een negende van je diepe MLP",
        },
        reveal="**9.098.** De eerste convolutional layer heeft 8 filters van 1 × 3 × 3 en 8 bias terms: 80. De tweede heeft 16 filters van 8 × 3 × 3 en 16 bias terms: 1.168. De output head heeft er 784 × 10 + 10 = 7.850, de meeste. Je diepe MLP heeft er alleen al in zijn eerste layer 50.240, een weight voor elke pixel en elk neuron; een filter heeft er 9 per input channel, en gebruikt ze op elke plek van de afbeelding.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In PyTorch is een convolutional layer `nn.Conv2d(c_in, c_out, k, padding=p)`: $C_\text{out}$ filters met shape $(C_\text{in}, k, k)$, elk met een bias term, zoals in [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl), snel uitgerekend, als één grote matrixvermenigvuldiging of met andere trucs. Hij werkt op een hele mini-batch tegelijk, met shape $(B, C, H, W)$: $B$ afbeeldingen, elk met $C$ channels van $H$ bij $W$ pixels. De afbeeldingen van Fashion-MNIST zijn grijs, één channel, dus een mini-batch van 64 heeft shape (64, 1, 28, 28). `nn.MaxPool2d(2)` doet max pooling van 2 bij 2 op elk channel van elke afbeelding.

        Je eerste CNN, `SmallCNN`, heeft twee blokken van convolutie, ReLU en pooling, en dan een output head. Met padding 1 houdt een convolutie van 3 bij 3 de hoogte en breedte, en elke pooling halveert ze: 28, 14, 7. Na het tweede blok is elke afbeelding 16 channels van 7 bij 7, en `x.reshape(len(x), -1)` maakt daar één rij van 784 getallen van, voor een `nn.Linear(784, 10)` die de 10 logits geeft, één voor elke soort kleding.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | `nn.Conv2d` | n n conv 2 d | een convolutional layer: $C_\text{out}$ filters met shape $(C_\text{in}, k, k)$, elk met een bias term | `nn.Conv2d(1, 8, 3, padding=1)` |
        | `nn.MaxPool2d` | n n max pool 2 d | max pooling van elk channel, in blokken van $k$ bij $k$ | `nn.MaxPool2d(2)` |
        | $(B, C, H, W)$ | B bij C bij H bij W | de shape van een mini-batch afbeeldingen: afbeeldingen, channels, hoogte, breedte | `X.reshape(len(X), 1, 28, 28)` |
        | `numel` | numel | het aantal getallen in een tensor, wat zijn shape ook is | `p.numel()` |
        """),
        L.watch_out("De output head moet passen bij wat er uit het laatste blok komt: 16 channels van 7 bij 7 zijn 784 getallen. Verander de grootte van de afbeeldingen, de padding of een layer, en dat getal verandert mee; PyTorch stopt dan met een fout dat de shapes niet vermenigvuldigd kunnen worden. En `nn.Conv2d` wil de channels vóór de hoogte en breedte: de foto's uit [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) hadden ze achteraan."),
        mo.md(r"""
        De figuur: één afbeelding op weg door `SmallCNN`, met zijn shape na elke stap. Elke stapel heeft een vierkant voor elk channel, hooguit 6 getekend, zo groot als zijn hoogte en breedte. In een mini-batch krijgt elke shape $B$ ervoor.
        """),
        C.shapes_picture({"title": "Eén afbeelding door SmallCNN", "input": "één grijze afbeelding", "maps": "{0} feature maps van {1} bij {1}",
                          "logits": "10 logits, één per soort", "conv": "conv 3×3 naar {1} channels, ReLU", "pool": "max pooling 2 bij 2",
                          "head": "één rij van 784, dan nn.Linear",
                          "alt": "Een grijze afbeelding van 1 bij 28 bij 28 wordt 8 feature maps van 28 bij 28 na een convolutie van 3 bij 3 met ReLU, 8 van 14 bij 14 na max pooling, 16 van 14 bij 14 na een tweede convolutie, 16 van 7 bij 7 na pooling, en 10 logits nadat één rij van 784 getallen door nn.Linear gaat."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een klein CNN voor afbeeldingen van 28 bij 28 met één channel:

    1. De afbeeldingen als mini-batch met shape $(B, 1, 28, 28)$: een rij van 784 pixels wordt een afbeelding met `.reshape(len(X), 1, 28, 28)`.
    2. Twee blokken, elk een convolutie van 3 bij 3 met padding 1, ReLU, en max pooling van 2 bij 2: $(B, 8, 14, 14)$, dan $(B, 16, 7, 7)$.
    3. Eén rij per afbeelding: `x.reshape(len(x), -1)`, 16 × 7 × 7 = 784 getallen.
    4. De output head, een lineaire layer van 784 naar 10: de logits. Train het zoals elk netwerk, met je Trainer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén blok, stap 2 van het recept, op de eerste 2 trainingsafbeeldingen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_block():
    import torch
    from torch import nn

    _X, _y, _, _ = C.fashion()
    # 1. twee afbeeldingen, als mini-batch
    _x = _X[:2]
    print(_x.shape)
    # 2. één blok: convolutie, ReLU, pooling
    torch.manual_seed(0)
    _conv = nn.Conv2d(1, 8, 3, padding=1)
    _pool = nn.MaxPool2d(2)
    print(_conv.weight.shape, _conv.bias.shape)
    _h = _pool(torch.relu(_conv(_x)))
    print(_h.shape)
    # 3. één rij per afbeelding
    _row = _h  # maak af: _h.reshape(len(_h), -1)
    print(_row.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De twee afbeeldingen hebben shape (2, 1, 28, 28). De filters van `_conv` hebben shape (8, 1, 3, 3), $(C_\text{out}, C_\text{in}, k, k)$ zoals in [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl), en er zijn 8 bias terms. Eén blok maakt van elke afbeelding 8 feature maps van 14 bij 14. Zoals de cel nu is, heeft `_row` nog die shape, (2, 8, 14, 14). Afgemaakt heeft hij shape (2, 1568): één rij van 8 × 14 × 14 = 1.568 getallen per afbeelding. `SmallCNN` heeft eerst nog een tweede blok, dus zijn rijen hebben 16 × 7 × 7 = 784 getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De shape na een blok
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape-after",
        question="Een mini-batch met shape (64, 1, 28, 28) gaat door `nn.Conv2d(1, 8, 5)`, zonder padding, en dan door `nn.MaxPool2d(2)`. Welke shape komt eruit?",
        options={
            "padded": "(64, 8, 14, 14): 28 bij 28 vóór de pooling",
            "right": "(64, 8, 12, 12): 24 bij 24 vóór de pooling",
            "one": "(64, 1, 12, 12): één channel, zoals de input",
        },
        correct="right",
        why={
            "padded": "Zonder padding past een filter van 5 bij 5 op $28 - 5 + 1 = 24$ plekken per kant, zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "one": "De layer heeft 8 filters, dus er komen 8 output channels uit, wat er ook binnenkwam.",
            "right": "De convolutie geeft (64, 8, 24, 24), en de pooling halveert de hoogte en breedte. De axis van de batch blijft zoals hij is: 64 afbeeldingen erin, 64 eruit.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_small_cnn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je eerste CNN

        Maak `SmallCNN` af: in `__init__` zijn layers, en in `forward` stap 2 tot en met 4 van het recept, voor een mini-batch `x` met shape $(B, 1, 28, 28)$; hij geeft de logits, met shape $(B, 10)$. Maak de layers in de volgorde waarin de data erdoor gaat: de convolutie vanaf 1 channel, die vanaf 8, dan de output head. De check maakt het netwerk na `torch.manual_seed(0)` en vergelijkt zijn logits voor 2 en voor 3 willekeurige afbeeldingen. Is hij goed, dan bewaart de check hem in je project, als `work/projects/cnn/small_cnn.py`.
        """),
        L.task("Laat `SmallCNN` 10 logits per afbeelding geven, uit twee blokken van convolutie, ReLU en pooling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_small_cnn():
    box_small_cnn = L.code_box("small-cnn", lines=18)
    box_small_cnn
    return (box_small_cnn,)


@app.cell(hide_code=True)
def check_small_cnn(box_small_cnn):
    result_small_cnn = L.check("small-cnn", box_small_cnn.value)
    result_small_cnn
    return (result_small_cnn,)


@app.cell(hide_code=True)
def hints_small_cnn():
    L.hints(
        "small-cnn",
        [
            "In `__init__` vier attributen van `self`: twee `nn.Conv2d`, één `nn.MaxPool2d` die beide blokken kunnen delen, omdat hij geen weights heeft, en een `nn.Linear` voor 16 × 7 × 7 inputs.",
            "In `forward` is elk blok de regel van stap 2 van het uitgewerkte voorbeeld, met jouw layers; na de twee blokken de regel van stap 3, en dan de output head.",
        ],
        why="Pooling leert niets, dus één `nn.MaxPool2d` kan beide blokken bedienen. Alleen de output head ziet een hele afbeelding tegelijk: elk getal ervoor hangt af van een stuk van hooguit 10 bij 10 pixels, zijn receptive field.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Tel wat een model leert

        Schrijf `count_params(model)`: de functie geeft hoeveel getallen alle parameters van `model` samen bevatten, de weights en de bias terms. Loop over `model.parameters()`, zoals in [AdamW: weight decay zoals het hoort](/course/open/opt-adamw?lang=nl), en tel `p.numel()` op voor elk: het aantal getallen in een tensor. De check telt `nn.Linear(3, 2)`, wat 8 moet geven, en andere layers. Is hij goed, dan bewaart de check hem als `work/projects/cnn/count.py`.
        """),
        L.task("Laat `count_params` de getallen in alle parameters van een model optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-params", lines=7)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-params", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-params",
        [
            "Begin met een totaal van 0, en tel erbij op in een `for`-loop over `model.parameters()`.",
            "`p.numel()` is het aantal getallen in `p`: 6 voor een weight-matrix met shape (2, 3).",
        ],
        why="Een parameter is één tensor, zoals de weight-matrix of de bias terms van een layer, en `numel` telt alle getallen erin, wat zijn shape ook is. Pooling heeft geen parameters, dus de loop draait niet en het totaal blijft 0.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij draait je overfit-test uit [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl) op je `SmallCNN`: {C.OVERFIT_STEPS} stappen met een learning rate van {C.number(C.OVERFIT_LR, 1, L.lang)} op de eerste 64 afbeeldingen. Daarna traint hij je `SmallCNN`, en je `DeepMLP` met He initialization, batch normalization en residual connections, met je Trainer: {C.EPOCHS} epochs op de 6.000 trainingsafbeeldingen, een learning rate van {C.number(C.LR, 2, L.lang)} met een warmup van één epoch, voor allebei hetzelfde, elk één run, met seed 0. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Het werkt zodra oefening 2 en 3 af zijn, en duurt ongeveer 10 seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_cnn(result_small_cnn, result_count):
    result_small_cnn, result_count  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `SmallCNN` en je `count_params` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je project doet er hier te lang over: meer dan een minuut voor de overfit-test, of meer dan twee minuten om één netwerk te trainen. Doet je `SmallCNN`, je overfit-test of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "cnn": "*`SmallCNN` of `count_params` in `work/projects/cnn/` stopt met een fout. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan weer.*",
            "overfit": "*Je `overfit_one_batch` in `work/projects/nn/overfit.py` stopt met een fout op het CNN. Heb je dat bestand veranderd? Verwijder het, en check oefening 2 van [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl) opnieuw.*",
            "trainer": "*Trainen met de Trainer in `work/projects/nn/trainer.py`, of de `DeepMLP` uit `work/projects/nn/deep_mlp.py` maken, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
        }[_result])
    else:
        _learned = "het kan leren" if _result["loss"] < 0.1 else "het leert de 64 afbeeldingen niet uit het hoofd, dus kijk eerst naar je `SmallCNN`"
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _out = mo.md(f"""Je overfit-test: na {C.OVERFIT_STEPS} stappen op 64 afbeeldingen is de loss van je CNN {C.number(_result['loss'], 3, L.lang)}: {_learned}.

Van de 1.000 testafbeeldingen leest je CNN er **{_result['cnn']}** goed, met {C.grouped(_result['params_cnn'], L.lang)} parameters, geteld met je `count_params`; je diepe MLP leest er **{_result['mlp']}** goed, met {C.grouped(_result['params_mlp'], L.lang)}. Over 5 seeds leest het CNN met de code van de course er 864 tot 879 goed, en het diepe MLP 845 tot 863: beter, met een negende van de parameters.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De filters, channels en pooling uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl), nu als `nn.Conv2d` en `nn.MaxPool2d`, getraind door je Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) en getest met de overfit-test uit [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl). Naar buren kijken en één filter over de hele afbeelding delen, uit [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl), zijn de reden dat het zo weinig parameters nodig heeft. Je project `cnn` heeft nu `small_cnn.py` en `count.py`. De tak computer vision zet dit project voort in drie extra lessen, met LeNet-5, VGG en ResNet. De hoofdroute gaat verder met *Attention weights uit inproducten*: hoe een model met inproducten en softmax uitrekent naar welke woorden het kijkt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_small_cnn, result_count):
    L.wrap_up(choice_shape, result_small_cnn, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
