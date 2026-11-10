# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-densenet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-densenet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl) en [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cat():
    quick_cat = L.choice(
        "quick-cat",
        question="`x` is 64 channels van 8 bij 8 en `y` is 12 channels van 8 bij 8, voor een batch van 1 foto. Welke shape geeft `torch.cat([x, y], dim=1)`?",
        options={
            "right": "(1, 76, 8, 8)",
            "plus": "(1, 64, 8, 8): de 12 channels van `y` worden bij de eerste 12 van `x` opgeteld",
            "stops": "Geen: PyTorch stopt, omdat `x` en `y` verschillende aantallen channels hebben",
        },
        correct="right",
        why={
            "right": "Langs `dim=1`, de channels, komen de 12 maps van `y` achter de 64 van `x`: 64 + 12 = 76. Vandaag doet elke layer van een block dit met de maps die hij maakt.",
            "plus": "Getal voor getal optellen is `+`, en dat zou hier stoppen: 12 channels passen niet op 64. `torch.cat` zet de maps na elkaar. Zie [Inception](/course/open/cv-inception?lang=nl).",
            "stops": "Langs `dim=1` heeft `torch.cat` alleen in de andere dimensies dezelfde grootte nodig: de foto's, de hoogte en de breedte. De channels mogen verschillen, zoals bij de takken van Inception, die 16, 32, 8 en 8 maps gaven. Zie [Inception](/course/open/cv-inception?lang=nl).",
        },
    )
    quick_cat
    return (quick_cat,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een residual block krijgt $x$, 64 channels van 8 bij 8, en geeft $x + f(x)$ terug. Wat krijgt de layer na het block?",
        options={
            "right": "64 channels van 8 bij 8, waarin de getallen van $x$ en $f(x)$ bij elkaar zijn opgeteld",
            "side": "128 channels: de 64 van $x$, en de 64 van $f(x)$ erachter",
            "only": "Alleen $f(x)$: het block gebruikt $x$ vanbinnen, en laat het daarna los",
        },
        correct="right",
        why={
            "right": "`+` telt getal voor getal op: wat eruit komt heeft de shape van $x$, en $x$ en $f(x)$ zijn erin gemengd. De layer na het block kan $x$ niet meer los zien. Het block van vandaag houdt ze apart.",
            "side": "Maps na elkaar zetten langs de channels is `torch.cat`. `+` telt ze getal voor getal op, dus heeft de som de shape van $x$. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
            "only": "Het block houdt $x$: het telt het op bij $f(x)$, zodat de gradient een korte weg terug heeft. Maar het geeft de som, niet $x$ los. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) telt een block zijn input op bij wat zijn layers maken: $x + f(x)$. Dat geeft de gradient een korte weg terug, maar de layer na het block krijgt alleen de som, waarin de input en de nieuwe maps gemengd zijn. In 2016 hielden Gao Huang, Zhuang Liu, Laurens van der Maaten en Kilian Weinberger ze juist apart. In hun DenseNet krijgt elke layer van een block de maps van de input van het block en van alle layers ervoor, na elkaar gezet langs de channels met `torch.cat`, zoals de takken van [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl). Omdat elke layer alles ervoor ziet, kan elke layer dun zijn: in de kleine netwerken van de paper voor CIFAR-10 voegt hij maar 12 eigen maps toe. Een DenseNet van hen met 100 layers en 0,8 miljoen parameters had 4,51% van de testfoto's van CIFAR-10 fout, ongeveer even weinig als een ResNet van 1.001 layers met 10,2 miljoen, 4,62%. Op ImageNet maakte hun DenseNet-201 met 20 miljoen parameters ongeveer evenveel fouten als een ResNet van 101 layers met meer dan 40 miljoen. De paper won de prijs voor de beste paper op de conferentie CVPR in 2017. Vandaag tel je wat de layers van een dense block krijgen en geven, bouw je er een, en test je in een CNN op CIFAR-10 wat zijn verbindingen doen.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_channels():
    guess_channels = L.guess(
        "dense-channels",
        question="Een dense block krijgt 64 channels van 8 bij 8 en heeft 4 layers. Elke layer krijgt de maps van de input van het block en van elke layer ervoor, en maakt 12 eigen maps, die erachter komen. Hoeveel channels geeft het block terug?",
        options={
            "same": "64: evenveel als het krijgt, zoals een residual block",
            "last": "12: de maps van zijn laatste layer",
            "right": "112",
        },
        reveal="**112.** Het block houdt alles: de 64 channels van zijn input, en de 12 maps van elk van zijn 4 layers, 64 + 4 × 12 = 112. Zijn eerste layer krijgt 64 channels, de tweede 76, de derde 88 en de vierde 100: elke layer krijgt er 12 meer dan de vorige. Een residual block zou er 64 teruggeven.",
    )
    guess_channels
    return (guess_channels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Dichte verbindingen.** Een dense block is een rij layers waarin elke layer de maps krijgt van de input van het block en van alle layers ervoor, na elkaar gezet langs de channels met `torch.cat`, en zelf $k$ nieuwe maps maakt. De paper noemt $k$ de **growth rate**: met elke layer groeien de maps van het block met $k$ channels. Heeft de input $k_0$ channels, dan krijgt layer $l$ er $k_0 + k(l - 1)$, en na $L$ layers geeft het block er $k_0 + kL$ terug. Er wordt niets opgeteld of weggegooid: de input blijft vooraan staan, onveranderd, en de maps van elke layer blijven waar ze zijn neergezet. Zo heeft elke layer een verbinding vanaf de input en vanaf elke layer ervoor: in een block van $L$ layers $L(L + 1) / 2$ verbindingen, waar een gewone rij layers er $L$ heeft. Daar komt de naam vandaan: dense is dicht.

        Het plaatje: een dense block met groei 12 op 64 channels, met 4 layers en dus 10 verbindingen; de pijlen naar de laatste layer zijn donkerder. Onder elke layer hoeveel channels hij krijgt en, in magenta, de 12 maps die hij maakt.
        """),
        C.connections_picture({"title": "Een dense block: 4 layers, groei 12", "input": "input x", "layer": "layer {l}",
                               "gets": "krijgt {n}", "gives": "geeft {n}",
                               "alt": "Vijf vakken op een rij: input x, die 64 channels geeft, en layers 1 tot en met 4. Elk vak heeft een pijl naar elke latere layer: 10 pijlen. Layer 1 krijgt 64 channels, layer 2 76, layer 3 88 en layer 4 100; elke layer geeft 12 maps. De vier pijlen naar layer 4 zijn donkerder."}),
        mo.md(r"""
        **De transition layer.** De channels groeien met elke layer, en `torch.cat` heeft maps van dezelfde hoogte en breedte nodig, dus kan een block zijn maps niet kleiner maken. Daarom is een DenseNet een paar dense blocks op een rij, met tussen elk tweetal een **transition layer**: een 1×1 convolution, die de channels mengt, zoals in [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), en average pooling van 2 bij 2, die de hoogte en breedte halveert. De DenseNet-BC uit de paper laat elke transition maar de helft van de channels geven die hij krijgt, en zet vóór elke 3×3-convolutie een 1×1 convolution naar $4k$ channels, een bottleneck: dan krijgt een 3×3-convolutie $4k$ channels, hoeveel er ook voor kwamen. Zijn layers hebben batch normalization en een ReLU vóór elke convolutie, zoals de ResNets van 2016 in de tabel hieronder.

        Tabel 2 van de paper vergelijkt netwerken op CIFAR-10, getraind op gespiegelde en verschoven foto's. De getallen zijn het deel van de 10.000 testfoto's dat het netwerk fout had:

        | Netwerk | Parameters | Fout op CIFAR-10 |
        |---|---|---|
        | ResNet, 164 layers | 1,7 miljoen | 5,46% |
        | ResNet, 1.001 layers | 10,2 miljoen | 4,62% |
        | DenseNet, $k = 12$, 40 layers | 1,0 miljoen | 5,24% |
        | DenseNet-BC, $k = 12$, 100 layers | 0,8 miljoen | 4,51% |
        | DenseNet-BC, $k = 40$, 190 layers | 25,6 miljoen | 3,46% |
        """),
        L.watch_out("Een dense block heeft niets te maken met wat sommige libraries, zoals Keras, een dense layer noemen: dat is een linear layer, `nn.Linear`. Dense betekent hier dat elke layer van het block met elke latere verbonden is. En de getallen uit de paper komen van netwerken met batch normalization, gespiegelde en verschoven trainingsfoto's en 300 epochs. Een uitkomst geldt waar ze gemeten is: wat dichte verbindingen doen in een klein netwerk zonder dat alles, moet de demo laten zien."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **dense block** | een rij layers waarin elke layer de maps krijgt van de input van het block en van alle layers ervoor, na elkaar gezet langs de channels, en er eigen maps aan toevoegt | `DenseBlock(64, 12, 4)` |
        | **growth rate** | hoeveel maps elke layer van een dense block toevoegt, $k$: layer $l$ krijgt $k_0 + k(l - 1)$ channels | `growth=12` |
        | **transition layer** | de layer tussen twee dense blocks: een 1×1 convolution die de channels mengt, in DenseNet-BC tot de helft, en pooling die de maps kleiner maakt | `nn.Conv2d(112, 64, 1)` |
        | **DenseNet** | een CNN van dense blocks met transition layers ertussen; uit 2016 | — |

        **Het netwerk van de demo.** De demo traint het netwerk uit [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), met een dense block in zijn midden, op de 64 maps van 8 bij 8, en daarna een transition: een 1×1 convolution van alle channels van het block terug naar 64, met een ReLU. De pooling erna is die van het netwerk zelf. Er zijn drie dense blocks om uit te kiezen, die allemaal 112 channels teruggeven: groei 24 met 2 layers, groei 12 met 4, en groei 6 met 8. Met zijn transition heeft het midden 40.112, 42.704 of 44.000 parameters. Om te zien wat de verbindingen zelf doen, is de vierde keuze de keten van de course: 4 layers van 3×3-convoluties met 30 channels, die elk alleen de maps van de layer ervoor krijgen, en dan een transition van 30 channels terug naar 64. Met 43.684 parameters heeft hij er ongeveer evenveel als groei 12 met 4 layers, en hij is even diep. Eén 3×3-convolutie van 64 channels naar 64, zoals in die les, heeft er 36.928.

        Het plaatje: hoeveel van de eerste 1.000 testfoto's van CIFAR-10 het netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's, een stip voor elk van 5 seeds, met de code van de course. De bovenste rij komt uit die les: één 3×3-convolutie, 532 tot 572. De keten: 452 tot 515. De dense blocks met groei 24, 12 en 6: 553 tot 584, 546 tot 573 en 557 tot 600. Oefening 3 vraagt wat de rijen zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                "plain": "één 3×3-convolutie", "chain": "keten: 4 layers van 30, elk ziet de vorige",
                                "k24": "dense block, groei 24, 2 layers", "k12": "groei 12, 4 layers", "k6": "groei 6, 8 layers",
                                "mine": "jouw run", "course": "de runs van de course",
                                "alt": "Vijf rijen open stippen, testfoto's goed gelezen van 1.000, een stip voor elke seed en een kort streepje bij hun gemiddelde. Eén 3×3-convolutie: 532 tot 572. Keten van 4 layers van 30: 452 tot 515. Dense block met groei 24 en 2 layers: 553 tot 584. Groei 12 en 4 layers: 546 tot 573. Groei 6 en 8 layers: 557 tot 600."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een dense block voor `channels` channels, met groei `growth` en `layers` layers:

    1. Maak in `__init__` layer `i` (0, 1, 2, ...) een 3×3-convolutie van alle maps die hij krijgt, de input van het block en de `growth` maps van elke layer ervoor, naar `growth` maps, met `padding=1`, en bewaar de layers in een `nn.ModuleList`.
    2. Stuur in `forward` `x` door elke layer en een ReLU: dat geeft `y`, `growth` nieuwe maps.
    3. Zet `y` achter `x` langs de channels: `x = torch.cat([x, y], dim=1)`.
    4. Geef na de laatste layer `x` terug: `channels + layers * growth` channels.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de vier layers van een dense block met groei 12 op 64 channels, en stuurt verzonnen maps van één foto, 64 channels van 8 bij 8, erdoor. Na elke layer print hij de shape van de weights van de layer en de shape van `x`. Daarna print hij de eerste 3 getallen van de input op de pixel linksboven, vóór het block en erna, en hoeveel weights de vier layers samen hebben. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_dense():
    import torch
    from torch import nn

    torch.manual_seed(0)
    x = torch.rand(1, 64, 8, 8)
    x0 = x
    growth = 12  # verander: growth = 24
    # layer i krijgt de 64 channels van de input en de maps van de i layers ervoor
    layers = []
    for i in range(4):
        layers.append(nn.Conv2d(64 + i * growth, growth, 3, padding=1))
    with torch.no_grad():
        for layer in layers:
            y = torch.relu(layer(x))
            # zet de nieuwe maps achter alle maps tot nu toe, langs de channels
            x = torch.cat([x, y], dim=1)
            print(list(layer.weight.shape), list(x.shape))
    print(x0[0, :3, 0, 0])
    print(x[0, :3, 0, 0])
    print(sum(layer.weight.numel() for layer in layers))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, print de cel `[12, 64, 3, 3] [1, 76, 8, 8]`, daarna `[12, 76, 3, 3] [1, 88, 8, 8]`, `[12, 88, 3, 3] [1, 100, 8, 8]` en `[12, 100, 3, 3] [1, 112, 8, 8]`: elke layer heeft 12 filters, elk zo diep als de channels die hij krijgt, en na hem heeft `x` 12 channels meer. Daarna twee keer `tensor([0.4963, 0.7745, 0.3251])`: de input staat nog vooraan in `x`, onveranderd. En `35424`, de weights van de vier layers, 12 × 9 × (64 + 76 + 88 + 100). Met `growth = 24` krijgen de layers 64, 88, 112 en 136 channels, en geeft het block `[1, 160, 8, 8]` terug: 64 + 4 × 24. Zijn layers hebben dan `86400` weights, meer dan twee keer zoveel: elke layer heeft twee keer zoveel filters, en krijgt ook meer channels.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif het aantal layers van een dense block met groei 12 op 64 channels, van 1 tot 8. Elke balk is wat een layer krijgt: de input van het block in grijs, en in magenta de 12 maps van elke layer ervoor. In oranje de parameters van de layer. De laatste balk is wat het block teruggeeft; de as blijft hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def play_layers():
    _labels = {"title": "Een dense block met groei 12 op 64 channels", "x": "channels",
               "layer": "layer {l} krijgt {n}", "out": "het block geeft {n} terug",
               "alt": "Een balk voor elke layer van het dense block, de channels die hij krijgt: de 64 van de input van het block in grijs, en 12 in magenta voor elke layer ervoor, met de parameters van de layer in oranje; en een laatste balk voor de channels die het block teruggeeft."}
    explore_layers = L.explore("layers-slider", "layers", lambda _v: (C.stack_picture(int(_v), _labels, L.lang), C.stack_sentence(int(_v), L.lang)),
                               start=4, values=C.LAYERS, role="hyper",
                               task="Vanaf hoeveel layers hebben de layers van het block samen meer parameters dan één 3×3-convolutie van 64 channels naar 64?")
    explore_layers
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Vanaf 5 layers: dan hebben ze samen 47.580 parameters, tegen 36.928; met 4 layers 35.472. Elke layer krijgt 12 channels meer dan de vorige, dus kost elke layer meer dan de vorige: de eerste 6.924 parameters, de achtste 15.996. Met 8 layers hebben de layers van het block 91.680 parameters, en heeft het block 36 verbindingen. De kosten groeien sneller dan het aantal layers: daarom zet DenseNet-BC vóór elke 3×3-convolutie een 1×1 convolution naar $4k$ channels, en maken zijn transitions de channels minder tussen de blocks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_blocks():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Optellen of houden

    Zes zinnen over een block. Welke beschrijven een residual block van ResNet dat $x + f(x)$ teruggeeft, en welke een dense block van DenseNet? Deel elke zin in.
    """)
    return


@app.cell(hide_code=True)
def exercise_blocks():
    sort_blocks = L.sort("blocks")
    sort_blocks
    return (sort_blocks,)


@app.cell(hide_code=True)
def check_blocks(sort_blocks):
    result_blocks = L.check("blocks", sort_blocks)
    result_blocks
    return (result_blocks,)


@app.cell(hide_code=True)
def hints_blocks():
    L.hints(
        "blocks",
        [
            "Een residual block geeft $x + f(x)$ terug; een dense block zet na elke layer `x = torch.cat([x, y], dim=1)`. Wat doet elk met het aantal channels?",
            "Bij elk block horen drie zinnen. Vraag bij elke zin: telt het block op, of zet het erachter?",
        ],
        why="**Een residual block:** het geeft evenveel channels terug als het krijgt, het telt wat zijn layers maken getal voor getal bij zijn input op, en wat zijn layers maken moet de shape van zijn input hebben. **Een dense block:** het geeft meer channels terug dan het krijgt, het zet de maps van elke layer achter de maps die het kreeg, en zijn laatste layer krijgt de maps van elke layer ervoor, onveranderd.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dense():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een dense block

        Maak `DenseBlock` af. Zijn argumenten zijn de channels die binnenkomen, `channels`; hoeveel maps elke layer toevoegt, `growth`; en het aantal layers, `layers`. De loop in `__init__` en de eerste regels van `forward` staan in het vak. Maak layer `i`, een 3×3-convolutie van alle maps die hij krijgt naar `growth` maps, die de hoogte en breedte houdt, en laat `forward` `y` achter `x` zetten. De check bouwt je block, met groei 12 en 4 layers op 64 channels en met verzonnen getallen, en vergelijkt wat eruit komt, en hoeveel parameters het heeft, met dat van de course. Klopt het, dan bewaart hij je class in je project, als `work/projects/cnn/dense.py`.
        """),
        L.task("Maak de convolutie van layer `i`, en laat `forward` `y` achter `x` zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dense():
    box_dense = L.code_box("dense", lines=18)
    box_dense
    return (box_dense,)


@app.cell(hide_code=True)
def check_dense(box_dense):
    result_dense = L.check("dense", box_dense.value)
    result_dense
    return (result_dense,)


@app.cell(hide_code=True)
def hints_dense():
    L.hints(
        "dense",
        [
            "Layer `i` krijgt de input van het block, `channels`, en de `growth` maps van elk van de `i` layers ervoor. Welke padding houdt de hoogte en breedte bij 3 bij 3?",
            "`self.layers.append(nn.Conv2d(channels + i * growth, growth, 3, padding=1))`. De nieuwe `x` is stap 3 van het recept.",
        ],
        why="Layer `i` is `nn.Conv2d(channels + i * growth, growth, 3, padding=1)`, en `forward` zet `x = torch.cat([x, y], dim=1)`. De layers van DenseNet zelf hebben batch normalization en een ReLU vóór hun convolutie, en in DenseNet-BC een 1×1 convolution naar 4 × `growth` channels daarvoor.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint het netwerk van de demo, met je `DenseBlock` en zijn transition in zijn midden, op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Kies het midden: een dense block met groei 24 en 2 layers, 12 en 4, of 6 en 8, of de keten van de course van 4 layers van 30 channels zonder dichte verbindingen, die geen block van jou nodig heeft. Waar een module van jou uit een eerdere les ontbreekt of kapot is, springt die van de course in. Het heeft CIFAR-10 nodig, en voor een dense block oefening 2, en duurt een paar minuten. Tijdens het trainen toont een grafiek de validation loss na elke epoch. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0. Daarna kun je terugspoelen door je run, en zie je je run naast die van de course.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Alleen de demo heeft het nodig. Tijdens het downloaden wacht de hele les.*",
        "busy": "**CIFAR-10 downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van CIFAR-10 uitpakken** · dit duurt een paar seconden",
        "no_net": "*De download is niet gelukt: de University of Toronto was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download is niet gelukt: het bestand dat binnenkwam is niet het vastgelegde bestand van de course, dus heeft de course het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "damaged": "*Uitpakken is niet gelukt: het gedownloade bestand bleek beschadigd, dus heeft de course het verwijderd. Klik opnieuw om het opnieuw te downloaden; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download is niet gelukt: er is op je schijf geen ruimte voor 170 MB en de uitgepakte foto's, 184 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download is niet gelukt ({why}). Klik opnieuw.*",
        "here": "*CIFAR-10 staat op je computer: 50.000 trainingsfoto's en 10.000 testfoto's.*",
    })
    return (cifar,)


@app.cell(hide_code=True)
def variant_choice_cell():
    variant_choice = mo.ui.radio(options={"dense block, groei 24, 2 layers": "k24", "dense block, groei 12, 4 layers": "k12",
                                          "dense block, groei 6, 8 layers": "k6", "keten van 4 layers van 30, zonder dichte verbindingen": "chain"},
                                 value="dense block, groei 12, 4 layers", label="Het midden van het netwerk")
    train_button = mo.ui.run_button(label="Train het netwerk")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_dense(cifar, train_button, variant_choice, result_dense):
    cifar, result_dense  # opnieuw uitvoeren als CIFAR-10 binnenkomt, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _names = {"k24": "Dense block, groei 24", "k12": "Dense block, groei 12", "k6": "Dense block, groei 6", "chain": "Keten van 4 layers"}
        _with = {"k24": "Met je dense block met groei 24 en 2 layers", "k12": "Met je dense block met groei 12 en 4 layers",
                 "k6": "Met je dense block met groei 6 en 8 layers", "chain": "Met de keten van de course van 4 layers van 30 channels"}
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van het netwerk op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2: dan staat je `DenseBlock` in je project, en gebruikt deze cel hem. De keten heeft geen block van jou nodig.*")
        elif isinstance(_result, str):
            _growth, _layers = C.VARIANTS.get(_variant, (0, 0))
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `DenseBlock` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "dense": f"*Je `DenseBlock` in `work/projects/cnn/dense.py` stopt met een fout, of geeft niet de channels terug die hij moet geven: met 64 channels erin, groei {_growth} en {_layers} layers, {C.channels_out(64, _growth, _layers)} maps van 8 bij 8. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validatie",
                       "plain": "één 3×3-convolutie",
                       "alt": "De loss van het netwerk na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0, die daalt tot epoch 6 en daarna stijgt."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
                        f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze {len(C.SHOWN)} foto's goed gelezen.")

            if len(_when) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: wanneer is de validation loss het laagst, en wat gebeurt er daarna?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""{_with[_variant]} leest het netwerk **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, waarvan {C.grouped(_result['middle_params'], L.lang)} in zijn midden.{_turn} Over 5 seeds, met de code van de course, dit midden: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                        "plain": "één 3×3-convolutie", "chain": "keten: 4 layers van 30, elk ziet de vorige",
                                        "k24": "dense block, groei 24, 2 layers", "k12": "groei 12, 4 layers", "k6": "groei 6, 8 layers",
                                        "mine": "jouw run", "course": "de runs van de course",
                                        "alt": "Vijf rijen testfoto's goed gelezen, van 1.000: open stippen voor de runs van de course over 5 seeds, met één 3×3-convolutie, met de keten van 4 layers van 30, en met de drie dense blocks, met een kort streepje bij het gemiddelde van elke rij; en in de rij van jouw midden een dichte stip voor jouw run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 3. Wat de verbindingen hier doen
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "dense-runs",
        question="Over 5 seeds, na 20 epochs. Met een dense block in het midden, met groei 24 en 2 layers, 12 en 4, of 6 en 8, leest het netwerk 553 tot 584, 546 tot 573 en 557 tot 600 testfoto's goed. Met de keten van 4 layers van 30 channels, die elk alleen de maps van de layer ervoor krijgen, even diep als groei 12 met 4 layers en met ongeveer evenveel parameters, leest het er 452 tot 515. Wat zeggen de getallen?",
        options={
            "deeper": "Meer layers met een kleinere growth rate lezen meer goed: groei 6 met 8 layers is het best",
            "weights": "De keten leest minder foto's goed omdat hij minder parameters heeft",
            "right": "De verbindingen maken het verschil: elke run met een dense block verslaat elke run van de keten, terwijl de drie dense blocks overlappen",
        },
        correct="right",
        why={
            "deeper": "Groei 6 met 8 layers heeft het hoogste gemiddelde, 572, en de hoogste run, 600. Maar groei 24 met 2 layers leest er gemiddeld 570, en groei 12 met 4 layers, qua diepte daartussenin, 561: de gemiddelden stijgen niet met de layers, en met 5 seeds elk overlappen de drie reeksen. Uit deze runs kun je de drie blocks niet uit elkaar houden.",
            "weights": "Het midden van de keten heeft 43.684 parameters, meer dan het dense block met groei 12 en 4 layers, 42.704, en hij is even diep: 4 layers van 3×3. Toch leest hij er 452 tot 515 goed, tegen 546 tot 573. Kijk naar wat zijn layers krijgen.",
            "right": "Elke run met een dense block, 546 tot 600, verslaat elke run van de keten, 452 tot 515, ook groei 12 met 4 layers, die even diep is en iets minder parameters heeft. De keten leert ook zijn trainingsfoto's slecht: 3.908 tot 4.727 van de 6.000 na 20 epochs, tegen 5.918 tot 5.990, met zijn laagste validation loss pas na epoch 12 tot 15. In een dense block krijgt elke layer de input onveranderd, en krijgt de transition de input en de maps van elke layer; de keten perst alles door 30 channels en vier layers. De paper verklaart het met korte wegen voor de gradient, zoals in ResNet; deze runs laten het verschil zien, niet de oorzaak. Tussen de drie dense blocks zijn de gemiddelden 570, 561 en 572, zonder volgorde erin. Gemiddeld lezen ze iets meer foto's goed dan één 3×3-convolutie, 549, maar zijn runs, 532 tot 572, overlappen met die van hen.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een dense block houdt wat het residual block uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) optelt: het zet de maps van zijn layers na elkaar met `torch.cat`, zoals de takken uit [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl), en zijn transition layer is een 1×1 convolution uit [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl). Je telde de weights van zijn layers met de shapes uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl), bewaarde zijn layers in een `nn.ModuleList` zoals in [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl), en trainde het met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `dense.py`. In de volgende les, *Depthwise separable convolutions: MobileNet*, geeft een convolutie elk channel een eigen filter, en mengt een 1×1 convolution de channels daarna: het werk van een 3×3-convolutie, met veel minder vermenigvuldigingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_blocks, result_dense, choice_runs):
    L.wrap_up(result_blocks, result_dense, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
