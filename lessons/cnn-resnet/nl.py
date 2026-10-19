# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-resnet. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, residual, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-resnet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl) en [Residual connections](/course/open/nn-residual?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stuck():
    quick_stuck = L.choice(
        "quick-stuck",
        question="Met 8 convoluties per blok en de eigen beginwaarden van PyTorch las je VGG-netwerk 105 van de 1.000 testafbeeldingen goed. Waarom?",
        options={
            "overfit": "Het leerde de 6.000 trainingsafbeeldingen uit het hoofd, en verder niets",
            "right": "Vanaf het begin gaf elke afbeelding dezelfde logits, dus het kwam nooit op gang",
            "small": "Filters van 3 bij 3 zien te weinig voor zo'n diep netwerk",
        },
        correct="right",
        why={
            "overfit": "Een netwerk dat zijn trainingsafbeeldingen geleerd heeft, houdt de testafbeeldingen nog wel enigszins uit elkaar; 105 is één soort voor elke afbeelding. Zie [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl).",
            "small": "Gestapeld zien de kleine filters meer, niet minder: 16 ervan zien over de hele afbeelding. Zie [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl).",
            "right": "Elke layer maakte de verschillen tussen afbeeldingen kleiner, tot er geen meer over waren. Vandaag voeg je toe wat ze doorlaat.",
        },
    )
    quick_stuck
    return (quick_stuck,)


@app.cell(hide_code=True)
def quick_check_through():
    quick_through = L.choice(
        "quick-through",
        question="Een residual block geeft $x + f(x)$ terug. In het begin verandert $f$ bijna niets: $f(x)$ is voor elke $x$ dicht bij 0. Wat geeft een stapel van 8 zulke blocks terug?",
        options={
            "zero": "Dicht bij 0: 8 blocks van bijna niets",
            "eight": "Dicht bij 8 keer $x$: elk block telt $x$ één keer op",
            "right": "Dicht bij $x$ zelf: elk block telt bijna niets op",
        },
        correct="right",
        why={
            "zero": "Elk block telt $f(x)$ op bij zijn input, en vervangt die niet: de input komt erdoor. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "eight": "Elk block geeft zijn input plus $f$: $x + 0$, en dan weer $x + 0$. Er verdubbelt niets. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "right": "Elk block geeft zijn input door, plus een beetje. Wat de beginweights ook zijn, de afbeelding komt erdoor, en op de terugweg de gradient ook.",
        },
    )
    quick_through
    return (quick_through,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je diepste VGG-netwerk leerde helemaal niets. In 2015 liepen Kaiming He, Xiangyu Zhang, Shaoqing Ren en Jian Sun tegen dezelfde muur aan op ImageNet: een gewoon netwerk van 34 layers maakte meer fouten dan een van 18, zelfs op zijn eigen trainingsafbeeldingen, ook met batch normalization. Hun **ResNet** zette een residual connection om elke twee convoluties, drie in de diepste versies, ging naar 152 layers en won de grote beeldwedstrijd van dat jaar. Residual connections zitten nu in bijna elk diep netwerk, van CNN's tot transformers. In deze laatste stap van je project bouw je het residual block van een CNN, en train je een netwerk dat even diep is als het VGG-netwerk dat niet leerde.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_residual():
    guess_residual = L.guess(
        "residual",
        question=f"Een gewoon netwerk met {C.BLOCKS} blokken van twee convoluties op 8 channels en {C.BLOCKS} op 16, 19 layers met weights, met de eigen beginwaarden van PyTorch, leest na 5 epochs 105 van de 1.000 testafbeeldingen goed. Je zet een residual connection om elk blok, en verandert verder niets. Hoeveel leest het er goed?",
        options={
            "same": "Ook ongeveer 105, want de layers zijn hetzelfde",
            "right": "Ongeveer evenveel als je SmallCNN, of een paar meer",
            "small": "Het leert, maar blijft ver onder je SmallCNN",
        },
        reveal="**Ongeveer evenveel als je SmallCNN, of een paar meer.** Over 5 seeds, met de code van de course, leest het netwerk met residual connections er 876 tot 895 goed, bij elke seed meer dan je SmallCNN (864 tot 879). Het gewone netwerk met dezelfde layers leest er elke keer 105 goed. De residual connections maken een diepte trainbaar die dat niet was, en hier, waar de extra diepte van VGG niets opleverde, loont die.",
    )
    guess_residual
    return (guess_residual,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het residual block van een CNN.** Zoals in [Residual connections](/course/open/nn-residual?lang=nl) geeft een block $x + f(x)$ terug. In een CNN is $f$ twee convoluties van 3 bij 3 met een ReLU ertussen, allebei van $C$ naar $C$ channels met padding 1, zodat $f(x)$ precies de shape van $x$ heeft.

        **Waarom het blijft werken.** Een gewoon blok geeft alleen $f(x)$ door. Met kleine beginweights maakt elk gewoon blok de verschillen tussen afbeeldingen kleiner, en na 15 layers is daar niets van over. Een residual block geeft ook $x$ door: wat $f$ in het begin ook doet, de afbeelding komt door elk block, en op de terugweg de gradient ook.

        **Niet alleen de beginwaarden.** Met de He initialization uit [Initialisatie](/course/open/nn-initialization?lang=nl) leert een gewoon netwerk van 19 layers ook: 854 tot 871 goed over 3 seeds. Maak het dieper, 35 layers, en het wordt slechter, 838 tot 847, terwijl dezelfde diepte met residual connections er 874 tot 879 goed leest. He en anderen zagen hetzelfde op ImageNet: een dieper gewoon netwerk traint slechter, ook met goede beginwaarden.

        **ResNet** stapelt zulke blocks in fases: blocks met dezelfde channels en grootte, en daarna minder pixels en meer channels. Waar dat verandert, passen $x$ en $f(x)$ niet meer, en geeft ResNet de residual connection een convolutie van 1 bij 1 met stride 2. Je netwerk doet het eenvoudiger: tussen zijn twee fases pooling en één gewone convolutie van 8 naar 16 channels. ResNet zet ook batch normalization na elke convolutie; je block laat die weg, om te laten zien wat de residual connection in zijn eentje doet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **residual block** | | twee convoluties van 3 bij 3 met een ReLU ertussen, en de input opgeteld bij hun output | `ResBlock(c)` |
        | **ResNet** | res net | de CNN's van He en anderen (2015): residual blocks in fases, 18 tot 152 layers diep | |
        """),
        L.watch_out("`x + f(x)` werkt alleen als allebei dezelfde shape hebben: dezelfde channels, hoogte en breedte. Een convolutie in `f` zonder padding, of een `f` die met meer channels eindigt dan `x` heeft, en PyTorch stopt met een fout dat de maten moeten passen. En schrijf `x + f(x)`, niet `x += f(x)`: dat verandert `x` zelf, en PyTorch heeft `x` op de terugweg nog nodig zoals hij was, bijvoorbeeld voor de gradient van de weights van de eerste convolutie."),
        mo.md(r"""
        De figuur: hoeveel de logits van een vers netwerk in het begin verschillen tussen afbeeldingen, tegen zijn layers met weights, op een logaritmische schaal. Het netwerk van deze les met 0 tot 8 blokken per fase, gewoon en met residual connections, op 256 testafbeeldingen. Zonder residual connections worden de verschillen elke 4 layers ongeveer honderd keer kleiner, tot alleen afronding over is; met residual connections blijven ze.
        """),
        mo.Html(C.spread_picture({"title": "Verschillen de afbeeldingen nog?", "x": "layers met weights", "y": "spreiding van de logits",
                                  "plain": "gewoon", "residual": "met residual connections", "lang": L.lang,
                                  "alt": "Twee lijnen op een logaritmische schaal tegen 3 tot 35 layers met weights. Met residual connections blijft het de hele weg rond 0,02. Zonder residual connections zakt het van 0,018 bij 3 layers naar ongeveer een tienduizendste bij 7, een miljoenste bij 11, en 2 tot 3 miljardste vanaf 19 layers, de afronding van float32."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een residual block voor $C$ channels:

    1. $f(x)$: een convolutie van 3 bij 3 van $C$ naar $C$ channels met padding 1, een ReLU, en een tweede zoals de eerste.
    2. Het block geeft $x + f(x)$ terug: een nieuwe tensor, geen `x += ...`.
    3. Stapel blocks met dezelfde channels tot een fase; tussen fases minder pixels en meer channels, met een convolutie waar de shapes veranderen.
    4. Train het zoals elk netwerk: met residual connections houdt diepte het leren niet meer tegen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee trainingsafbeeldingen door een convolutie naar 8 channels, en dan 8 blokken van twee convoluties, met de beginwaarden van PyTorch: eerst gewoon, dan met dezelfde layers als residual blocks, stap 2 van het recept. Hoeveel verschillen de twee afbeeldingen nog, als het gemiddelde van de verschillen tussen hun getallen? De regel met de residual connection is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_through():
    import torch
    from torch import nn

    _X, _y, _, _ = C.fashion()
    _x = _X[:2]
    torch.manual_seed(0)
    _stem = nn.Conv2d(1, 8, 3, padding=1)
    _convs = nn.ModuleList()
    for _ in range(16):
        _convs.append(nn.Conv2d(8, 8, 3, padding=1))
    with torch.no_grad():
        _h = torch.relu(_stem(_x))
        print("start:", round((_h[0] - _h[1]).abs().mean().item(), 4))
        # gewoon: 8 blokken van twee convoluties, elk met een ReLU
        for _k in range(0, 16, 2):
            _h = torch.relu(_convs[_k + 1](torch.relu(_convs[_k](_h))))
        print("gewoon:", round((_h[0] - _h[1]).abs().mean().item(), 4))
        # 2. met residual connections: dezelfde layers, als x + f(x)
        _h = torch.relu(_stem(_x))
        for _k in range(0, 16, 2):
            _h = torch.relu(_convs[_k + 1](torch.relu(_convs[_k](_h))))  # maak af: _h + _convs[_k + 1](torch.relu(_convs[_k](_h)))
        print("met residual connections:", round((_h[0] - _h[1]).abs().mean().item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In het begin verschillen de twee afbeeldingen 0,1076 per getal. Na 8 gewone blokken, 16 convoluties, 0,0: tot op 4 decimalen ziet het netwerk voor allebei hetzelfde, en zullen zijn logits ze niet uit elkaar houden. Zoals de cel nu is, doet de regel met de residual connection hetzelfde als de gewone, en geeft hij ook 0,0. Afgemaakt verschillen ze nog 0,1243: elk block geeft zijn input door, en de afbeeldingen komen erdoor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Als de shapes niet passen
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "res-shape",
        question="In een residual block gaan beide convoluties van $f$ naar 16 channels, maar $x$ heeft er 8. Wat gebeurt er bij `x + f(x)`?",
        options={
            "zeros": "PyTorch vult de 8 ontbrekende channels van `x` met nullen",
            "first": "De 8 channels van `x` worden opgeteld bij de eerste 8 van `f(x)`",
            "right": "PyTorch stopt met een fout",
        },
        correct="right",
        why={
            "zeros": "PyTorch vult uit zichzelf niets aan: aanvullen met nullen is een keuze die je in code maakt, en een die de paper van ResNet uitprobeerde.",
            "first": "`+` telt getal voor getal op, en heeft aan beide kanten dezelfde shape nodig, of een 1 waar de ander meer heeft. 8 channels tegen 16 is geen van beide.",
            "right": "Shapes van (B, 8, H, W) en (B, 16, H, W) kun je niet optellen. ResNet geeft de residual connection daar een convolutie van 1 bij 1 van 8 naar 16 channels.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het residual block

        Maak `ResBlock(c)` af: in `__init__` zijn twee convoluties, eerst `conv1` en dan `conv2`; in `forward` stap 1 en 2 van het recept. De check maakt blocks na `torch.manual_seed(0)` en vergelijkt wat eruit komt, maakt `f(x)` 0 om te zien dat het block `x` doorgeeft, draait de backward pass erdoorheen, en telt de parameters die PyTorch ziet. Als het klopt, bewaart de check het in je project, als `work/projects/cnn/resnet.py`.
        """),
        L.task("Laat `ResBlock` `x + f(x)` teruggeven, met `f` twee convoluties van 3 bij 3 en een ReLU ertussen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("res-block", lines=13)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("res-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "res-block",
        [
            "In `__init__` twee attributen van `self`, elk een `nn.Conv2d` van 3 bij 3 van `c` naar `c` channels; met `padding=1` houden ze de hoogte en breedte.",
            "In `forward` is `f(x)` de eerste convolutie, `torch.relu` eromheen, en de tweede convolutie daar weer omheen. Geef `x` plus dat terug.",
        ],
        why="Geen ReLU na het optellen: dan gaat de input onveranderd door het block, zoals in [Residual connections](/course/open/nn-residual?lang=nl). Het eerste ResNet van 2015 had daar een ReLU; in 2016 lieten He en anderen die weg, om het directe pad schoon te houden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_projection():
    mo.md(r"""
    ### Oefening 3. Waar de fase verandert
    """)
    return


@app.cell(hide_code=True)
def choice_projection_cell():
    choice_projection = L.choice(
        "res-projection",
        question="Waar een fase van ResNet eindigt, heeft `x` shape (8, 14, 14) en `f(x)` (16, 7, 7). Op de residual connection zet ResNet `nn.Conv2d(8, 16, 1, stride=2)`. Welke shape maakt dat van `x`?",
        options={
            "wide": "(16, 14, 14): een filter van 1 bij 1 houdt de grootte",
            "right": "(16, 7, 7): de shape van `f(x)`",
            "same": "(8, 7, 7): een filter van 1 bij 1 houdt de channels",
        },
        correct="right",
        why={
            "wide": "De grootte blijft bij stride 1. Met stride 2 wordt het filter op elke tweede plek gebruikt: 14 wordt 7.",
            "same": "Een convolutie geeft één channel per filter: 16 filters, 16 channels, ook als ze 1 bij 1 zijn.",
            "right": "16 filters van 8 bij 1 bij 1 maken 16 channels, en stride 2 halveert 14 tot 7: nu kun je `x` en `f(x)` optellen.",
        },
    )
    choice_projection
    return (choice_projection,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij bouwt twee netwerken van dezelfde diepte: een convolutie van 1 naar 8 channels, {C.BLOCKS} blokken van 8 channels, pooling, een convolutie naar 16 channels, {C.BLOCKS} blokken van 16 channels, pooling, en de output head: 19 layers met weights. In het ene is elk blok gewoon, twee convoluties met elk een ReLU erna; in het andere is elk blok je `ResBlock`. Allebei beginnen ze met dezelfde weights. Hij traint elk met je Trainer: {C.EPOCHS} epochs op de 6.000 trainingsafbeeldingen, een learning rate van {C.number(C.LR, 2, L.lang)} met een warmup van één epoch, elk één run, met seed 0, en telt de parameters met je `count_params`. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Het werkt zodra oefening 2 af is, en duurt ongeveer anderhalve minuut.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train beide netwerken")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_resnet(train_button, result_block):
    result_block  # run again when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Druk op de knop om beide netwerken te trainen.*")
    else:
        _result = C.demo()
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `ResBlock` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "slow": "*Je project doet er hier te lang over: meer dan vier minuten om één netwerk te trainen. Doet je `ResBlock` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "resnet": "*Je `ResBlock` in `work/projects/cnn/resnet.py` stopt met een fout, bij het maken of trainen van het netwerk. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result])
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
            _out = mo.md(f"""Van de 1.000 testafbeeldingen leest het gewone netwerk er **{_result['plain']}** goed, en dat van je residual blocks **{_result['residual']}**, allebei met {C.grouped(_result['params_residual'], L.lang)} parameters, geteld met je `count_params`. Over 5 seeds, met de code van de course: het gewone netwerk elke keer 105, en dat met residual connections 876 tot 895, bij elke seed meer dan je SmallCNN.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De residual connection uit [Residual connections](/course/open/nn-residual?lang=nl), nu om de convoluties van een CNN: waar je VGG blocks uit [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl) ophielden met leren, traint nu een netwerk dat nog dieper is, en leest het meer testafbeeldingen goed dan je SmallCNN. Je project `cnn` is af: `small_cnn.py`, `lenet.py`, `vgg.py` en `resnet.py`, van het CNN uit 1998 tot het block dat nog in bijna elk diep netwerk zit. Je komt $x + f(x)$ weer tegen in elk transformer block.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_block, choice_projection):
    L.wrap_up(choice_shape, result_block, choice_projection)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
