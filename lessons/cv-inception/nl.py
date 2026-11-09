# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-inception. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-inception", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) en [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reduce():
    quick_reduce = L.choice(
        "quick-reduce",
        question="Een 1×1 convolution van 192 channels naar 16. Hoeveel parameters heeft hij?",
        options={
            "five": "76.816: 16 filters van 192 × 5 × 5 weights, elk met een bias",
            "right": "3.088: 16 filters van 192 weights, elk met een bias",
            "sum": "208: 192 + 16",
        },
        correct="right",
        why={
            "five": "Dat zou een convolutie van 5 bij 5 zijn. Een filter van 1 bij 1 heeft één weight voor elk channel dat binnenkomt, 192, en een bias. Zie [1×1 convolutions](/course/open/cv-1x1-convolution?lang=nl).",
            "sum": "Elk van de 16 filters heeft een weight voor elk van de 192 channels, en een bias: 16 × (192 + 1). Zie [1×1 convolutions](/course/open/cv-1x1-convolution?lang=nl).",
            "right": "16 × (192 + 1) = 3.088. Vandaag staat zo'n convolutie vóór een convolutie van 5 bij 5, die dan op 16 channels werkt in plaats van 192.",
        },
    )
    quick_reduce
    return (quick_reduce,)


@app.cell(hide_code=True)
def quick_check_view():
    quick_view = L.choice(
        "quick-view",
        question="Twee 3×3-convoluties met padding 1, na elkaar. Hoe breed is het stuk van de input waarvan één getal erna afhangt?",
        options={
            "six": "6 bij 6: 3 + 3",
            "right": "5 bij 5",
            "nine": "9 bij 9: 3 × 3",
        },
        correct="right",
        why={
            "six": "De twee stukken overlappen: elke 3×3-convolutie maakt het stuk 2 breder, niet 3. Zie [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl).",
            "nine": "Elke 3×3-convolutie maakt het stuk 2 breder: 3 na de eerste, 5 na de tweede. Zie [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl).",
            "right": "3 bij 3 na de eerste, 5 bij 5 na de tweede, zoals één filter van 5 bij 5 zou zien. VGG stapelt filters van één grootte; het block van vandaag zet filters van verschillende groottes naast elkaar.",
        },
    )
    quick_view
    return (quick_view,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Hoe groot moet een filter zijn? In [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl) was elk filter 3 bij 3, en pas een stapel ervan liet een getal verder kijken. Maar wat een foto laat zien, heeft elke grootte: het oog van een kat is een paar pixels, zijn kop kan de halve foto vullen. In 2014 gaven Christian Szegedy en zijn collega's bij Google één block filters van 1 bij 1, 3 bij 3 en 5 bij 5 naast elkaar, met max pooling ernaast, en lieten ze de training beslissen waarvoor elk dient. Met de 1×1 convolutions uit [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) ervoor, om dat betaalbaar te houden, stapelden ze 9 van zulke blocks in GoogLeNet: 22 layers met weights, en 12 keer zo weinig parameters als AlexNet, de winnaar van 2012. Het won de wedstrijd van ImageNet in 2014. De naam komt van *Network In Network* en van een internetgrap over de film *Inception*: "we need to go deeper". Vandaag tel je wat de 1×1 convolutions besparen, bouw je een Inception block, en zoek je in een CNN op CIFAR-10 uit op welke van zijn takken het netwerk leunt.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_channels():
    guess_channels = L.guess(
        "naive-channels",
        question="Inception (3a), een block van GoogLeNet, krijgt 192 channels van 28 bij 28. Zonder de 1×1 convolutions die de channels minder maken, zou het drie convoluties rechtstreeks op die 192 channels zetten, met 64 filters van 1 bij 1, 128 van 3 bij 3 en 32 van 5 bij 5, en 3×3 max pooling ernaast, en alles wat eruit komt langs de channels stapelen. Hoeveel channels zouden eruit komen?",
        options={
            "sum": "224: 64 + 128 + 32",
            "right": "416",
            "same": "192: evenveel als erin gingen",
        },
        reveal="**416.** De pooling heeft geen filters: hij houdt alle 192 channels, dus komen er 64 + 128 + 32 + 192 = 416 uit. Het volgende block zou 416 channels krijgen, en met een eigen pooling nog meer geven: de paper noemt dat een explosie van rekenwerk binnen een paar stappen. GoogLeNet zet een 1×1 convolution na de pooling, naar 32 channels, dus geeft het block er 256. En het zet 1×1 convolutions vóór de filters van 3 bij 3 en 5 bij 5, die dan op 96 en 16 channels werken in plaats van 192: 163.696 parameters in plaats van 387.296.",
    )
    guess_channels
    return (guess_channels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Takken naast elkaar.** Een Inception block stuurt dezelfde maps tegelijk vier takken in: een 1×1 convolution; een 1×1 convolution naar minder channels, dan een 3×3-convolutie; een 1×1 convolution naar minder channels, dan een 5×5-convolutie; en 3×3 max pooling, dan een 1×1 convolution. Elke convolutie heeft een ReLU erna. Daarna zet het block wat de vier takken geven naast elkaar langs de channels, met `torch.cat` uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl). Een batch maps heeft de shape (foto's, channels, rijen, kolommen), dus de channels zijn `dim=1`, en ze tellen op: in inception (3a) 64 + 128 + 32 + 32 = 256.

        **Dezelfde hoogte en breedte.** `torch.cat` zet maps alleen naast elkaar als hun rijen en kolommen passen. Met stride 1 past een venster van $k$ bij $k$ op $n$ pixels met padding $p$ precies $n + 2p - k + 1$ keer, volgens de formule uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl): padding 1 houdt de grootte voor 3 bij 3, en padding 2 voor 5 bij 5. De max pooling van de vierde tak gebruikt stride 1 en padding 1, dus die houdt de grootte ook, anders dan de 2×2 pooling van je ColorCNN, die hem halveert.

        Het plaatje: inception (3a), van de 192 maps van 28 bij 28 die binnenkomen, door de vier takken, naar de 256 die eruit komen. De oranje vakken hebben weights die de training leert; de pooling heeft er geen.
        """),
        C.block_picture(C.G3A, C.G3A_SIDE, {"title": "Inception (3a) van GoogLeNet", "input": "{c} channels van {s} bij {s}",
                                            "output": "{c} channels van {s} bij {s}, naast elkaar", "filters": "{n} filters",
                                            "conv1": "1×1 convolution", "conv": "{k}×{k}-convolutie", "pool": "3×3 max pooling",
                                            "alt": "Een Inception block van boven naar beneden. 192 channels van 28 bij 28 gaan vier takken naast elkaar in: een 1×1 convolution met 64 filters; een 1×1 convolution met 96 filters, dan een 3×3-convolutie met 128; een 1×1 convolution met 16 filters, dan een 5×5-convolutie met 32; en 3×3 max pooling, dan een 1×1 convolution met 32 filters. Onderaan 256 channels van 28 bij 28: 64 + 128 + 32 + 32."}),
        mo.md(r"""
        **Waarom de 1×1 convolutions.** Een filter van $k$ bij $k$ op $c$ channels heeft $c \times k \times k$ weights, dus een 5×5-filter op alle 192 channels is duur: 4.800 weights. Achter een 1×1 convolution naar 16 channels heeft het er 400. Inception (3a) heeft 163.696 parameters; de eigen tabel van GoogLeNet zegt 159K, omdat die alleen de 163.328 weights telt, zonder de biases, en een K als 1.024 telt. Voor één foto doet het block 128.049.152 vermenigvuldigingen: die 163.328 voor elk van de 28 × 28 pixels, de 128 miljoen uit de tabel. Zonder zijn 1×1 convolutions vóór de filters van 3 bij 3 en 5 bij 5 en na de pooling, het naïeve block uit de paper, zou het 387.296 parameters hebben, 303 miljoen vermenigvuldigingen doen, en het volgende block 416 channels geven in plaats van 256.
        """),
        L.watch_out("Inception is de naam van een familie, niet van één netwerk. GoogLeNet is de eerste, uit 2014; de versies erna, vanaf 2015, veranderden het block. Sommige zetten twee 3×3-convoluties waar de 5×5 stond, de truc van VGG blocks, en kregen batch normalization. Zegt een paper of een library Inception, kijk dan welke versie bedoeld is."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **Inception block** | takken van 1×1 convolutions, 3×3- en 5×5-convoluties en van 3×3 max pooling naast elkaar op dezelfde maps, de 3×3 en 5×5 achter een 1×1 convolution naar minder channels en de pooling vóór één, met hun maps gestapeld langs de channels. GoogLeNet is een stapel ervan | `torch.cat([y1, y3, y5, yp], dim=1)` |

        **Het netwerk van de demo.** De demo traint het netwerk uit [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), met een Inception block in het midden. Op de 64 maps van 8 bij 8 heeft het 16 filters van 1 bij 1; 32 van 3 bij 3, achter een 1×1 convolution naar 32 channels; 8 van 5 bij 5, achter één naar 8; en 8 filters van 1 bij 1 na de pooling: 16 + 32 + 8 + 8 = 64 channels, de outputs van inception (3a) gedeeld door 4. Het block heeft 15.016 parameters, tegen 36.928 voor één 3×3-convolutie van 64 naar 64; het hele netwerk 44.658 tegen 66.570.

        Het plaatje: hoeveel van de eerste 1.000 testfoto's van CIFAR-10 het netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's, een stip voor elk van 5 seeds, met de code van de course. Met één 3×3-convolutie in het midden, zoals in die les: 532 tot 572, gemiddeld 549. Met het Inception block: 553 tot 585, gemiddeld 567. De twee bereiken overlappen: minstens zoveel goed, met een derde minder parameters. Het leert zijn trainingsfoto's nog steeds bijna uit zijn hoofd, 5.896 tot 5.997 van de 6.000. Daarna zette de course in de getrainde netwerken één tak tegelijk uit, door zijn channels op 0 te zetten, en telde opnieuw: de vier rijen eronder. Oefening 3 vraagt wat ze zeggen.
        """),
        mo.Html(C.off_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                               "plain": "één 3×3-convolutie", "inception": "Inception block, alle vier de takken",
                               "b1": "zonder zijn 1×1-tak", "b3": "zonder zijn 3×3-tak", "b5": "zonder zijn 5×5-tak",
                               "pool": "zonder zijn pooling-tak", "mine": "je run", "course": "de runs van de course",
                               "alt": "Zes rijen open stippen, testfoto's goed gelezen van 1.000, tussen 400 en 600, één stip voor elke seed en een kort streepje bij hun gemiddelde. Eén 3×3-convolutie in het midden: 532 tot 572. Het Inception block met alle vier de takken: 553 tot 585. Zonder zijn 1×1-tak: 409 tot 484. Zonder zijn 3×3-tak: 453 tot 504. Zonder zijn 5×5-tak: 521 tot 567. Zonder zijn pooling-tak: 468 tot 541."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een Inception block:

    1. Geef het vier takken op dezelfde input `x`: een 1×1 convolution; een 1×1 convolution naar minder channels en een 3×3-convolutie; een 1×1 convolution naar minder channels en een 5×5-convolutie; 3×3 max pooling en een 1×1 convolution.
    2. Zet een ReLU na elke convolutie.
    3. Houd in elke tak de hoogte en breedte: padding 1 voor 3 bij 3, padding 2 voor 5 bij 5, en de max pooling met `stride=1, padding=1`.
    4. Zet de takken naast elkaar langs de channels, `torch.cat([y1, y3, y5, yp], dim=1)`: dan komen er `c1 + c3 + c5 + pool_proj` channels uit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel bouwt inception (3a) uit zijn zes convoluties en zijn pooling, stuurt er verzonnen maps van één foto doorheen, 192 channels van 28 bij 28, en print de hoogte en breedte van wat elke tak geeft, de shape van de vier naast elkaar, en het aantal parameters. Voer de cel uit zoals hij is, verander hem dan waar het commentaar zegt, en voer hem nog eens uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_inception():
    import torch
    from torch import nn

    x = torch.rand(1, 192, 28, 28)
    # 1. vier takken op dezelfde x
    b1 = nn.Conv2d(192, 64, 1)
    b3_reduce, b3 = nn.Conv2d(192, 96, 1), nn.Conv2d(96, 128, 3, padding=1)
    b5_reduce, b5 = nn.Conv2d(192, 16, 1), nn.Conv2d(16, 32, 5, padding=2)  # verander: padding=1
    pool, pool_proj = nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(192, 32, 1)
    # 2. een ReLU na elke convolutie
    y1 = torch.relu(b1(x))
    y3 = torch.relu(b3(torch.relu(b3_reduce(x))))
    y5 = torch.relu(b5(torch.relu(b5_reduce(x))))
    yp = torch.relu(pool_proj(pool(x)))
    # 3. elke tak houdt 28 bij 28
    print(y1.shape[2:], y3.shape[2:], y5.shape[2:], yp.shape[2:])
    # 4. naast elkaar langs de channels
    out = torch.cat([y1, y3, y5, yp], dim=1)
    print(out.shape)
    total = 0
    for layer in (b1, b3_reduce, b3, b5_reduce, b5, pool_proj):
        total += sum(q.numel() for q in layer.parameters())
    print(total)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, print hij vier keer `torch.Size([28, 28])`: elke tak houdt de hoogte en breedte, zoals stap 3 van het recept zegt. Dan `torch.Size([1, 256, 28, 28])`: 64 + 128 + 32 + 32 channels naast elkaar, en `163696` parameters. Daarvan zitten er 12.352 in de 1×1-tak; 18.528 + 110.720 in de 3×3-tak; 3.088 + 12.832 in de 5×5-tak; en 6.176 in de 1×1 convolution na de pooling. Met `padding=1` geeft de 5×5-convolutie 26 bij 26: de eerste regel print `torch.Size([26, 26])` voor de derde tak, en `torch.cat` stopt met `Sizes of tensors must match except in dimension 1. Expected size 28 but got size 26 for tensor number 2 in the list`. Hij telt de tensors in de lijst vanaf 0, dus nummer 2 is `y5`.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif de grootte van de filters van één tak, van 1 bij 1 tot 11 bij 11. De tak krijgt 192 channels en geeft er 32, zoals de 5×5-tak van inception (3a). De grijze balk is de tak rechtstreeks op de 192 channels; de oranje balk is de tak achter een 1×1 convolution naar 16 channels, zijn twee convoluties na elkaar. De as blijft dezelfde, in duizenden parameters.
    """)
    return


@app.cell(hide_code=True)
def play_kernel():
    _labels = {"title": "Eén tak, van 192 channels naar 32", "x": "parameters (duizenden)",
               "straight": "rechtstreeks op 192", "reduced": "achter een 1×1 naar 16",
               "alt": "Twee balken parameters, in duizenden, voor filters van de gekozen grootte: de tak rechtstreeks op de 192 channels, in grijs, en de tak achter een 1×1 convolution naar 16 channels, zijn twee convoluties na elkaar, in oranje."}
    explore_kernel = L.explore("kernel", "grootte van het filter", lambda _v: (C.kernel_picture(int(_v), _labels, L.lang), C.kernel_sentence(int(_v), L.lang)),
                               start=5, values=C.KERNELS, role="hyper",
                               task="Vanaf welke grootte van het filter heeft de tak rechtstreeks op de 192 channels meer dan 10 keer zoveel parameters als de tak achter de 1×1 convolution?")
    explore_kernel
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Vanaf 7 bij 7: 301.088 parameters tegen 28.208, 10,7 keer zoveel. Bij 5 bij 5 is het 9,7 keer, en bij 3 bij 3 is het 7,2. Hoe groter het filter, hoe dichter de factor bij 192 / 16 = 12 komt: dan zitten bijna alle parameters in de filters van $k$ bij $k$, en achter de 1×1 convolution werken die op 16 channels in plaats van 192. Daarom geeft GoogLeNet zijn 5×5-takken de smalste 1×1 convolutions ervoor: 16 van de 192 channels in inception (3a), waar de 3×3-tak er 96 krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_same_size():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Dezelfde grootte

    Een tak van een Inception block krijgt maps van 28 bij 28, en `torch.cat` heeft van alle vier dezelfde hoogte en breedte nodig. Welke van deze zes stappen houden 28 bij 28, en welke veranderen de grootte? Zet elke stap in zijn groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_same_size():
    sort_same_size = L.sort("same-size")
    sort_same_size
    return (sort_same_size,)


@app.cell(hide_code=True)
def check_same_size(sort_same_size):
    result_same_size = L.check("same-size", sort_same_size)
    result_same_size
    return (result_same_size,)


@app.cell(hide_code=True)
def hints_same_size():
    L.hints(
        "same-size",
        [
            "Reken voor elke stap uit hoeveel plekken een venster van $k$ bij $k$ heeft op 28 pixels: $28 + 2p - k + 1$ met stride 1. Met een stride van 2 neemt het alleen elke tweede plek.",
            "Vier ervan houden 28 bij 28, twee niet. Een padding van $(k - 1) / 2$ houdt de grootte: 0 voor 1 bij 1, 1 voor 3 bij 3, 2 voor 5 bij 5.",
        ],
        why="**Ze houden 28 bij 28:** de 1×1 convolution, 28 + 0 − 1 + 1 = 28; de 3×3-convolutie met padding 1, 28 + 2 − 3 + 1 = 28; de 5×5-convolutie met padding 2, 28 + 4 − 5 + 1 = 28; en 3×3 max pooling met stride 1 en padding 1, zoals de 3×3-convolutie. **Ze veranderen de grootte:** de 5×5-convolutie met padding 1, 28 + 2 − 5 + 1 = 26; en 2×2 max pooling, met een stride van 2, dus 14 bij 14.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_inception():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een Inception block

        Maak `Inception` af. Zijn argumenten zijn de kolommen van de tabel van GoogLeNet: de channels die binnenkomen, `c_in`; de filters van de 1×1-tak, `c1`; de filters van de 1×1 convolution vóór de 3×3-convolutie, en van die 3×3-convolutie zelf, `c3_reduce` en `c3`; hetzelfde voor de 5×5-tak, `c5_reduce` en `c5`; en de filters na de pooling, `pool_proj`. De 1×1-, 3×3- en pooling-takken staan in het vak. Maak de 5×5-tak: `b5_reduce`, een 1×1 convolution van `c_in` naar `c5_reduce`, en `b5`, een 5×5-convolutie van `c5_reduce` naar `c5` die de hoogte en breedte houdt. Maak in `forward` `y5` zoals `y3` gemaakt wordt, en geef de vier takken naast elkaar terug langs de channels, in de volgorde `y1`, `y3`, `y5`, `yp`. De check bouwt je block, met de getallen van inception (3a) en met verzonnen getallen, en vergelijkt wat eruit komt, en hoeveel parameters het heeft, met dat van de course. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/inception.py`.
        """),
        L.task("Maak `b5_reduce` en `b5`, dan `y5`, en geef de vier takken naast elkaar terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_inception():
    box_inception = L.code_box("inception", lines=27)
    box_inception
    return (box_inception,)


@app.cell(hide_code=True)
def check_inception(box_inception):
    result_inception = L.check("inception", box_inception.value)
    result_inception
    return (result_inception,)


@app.cell(hide_code=True)
def hints_inception():
    L.hints(
        "inception",
        [
            "`b5_reduce` en `b5` maak je zoals `b3_reduce` en `b3`, met andere namen en een andere kernel size. Welke padding houdt de hoogte en breedte bij 5 bij 5? Oefening 1 zegt het.",
            "`self.b5 = nn.Conv2d(c5_reduce, c5, 5, padding=2)`, en `y5` is de regel van `y3` met `b5_reduce` en `b5`. De laatste regel is stap 4 van het recept.",
        ],
        why="`b5_reduce` is `nn.Conv2d(c_in, c5_reduce, 1)` en `b5` is `nn.Conv2d(c5_reduce, c5, 5, padding=2)`; `y5 = torch.relu(self.b5(torch.relu(self.b5_reduce(x))))`; en `forward` eindigt met `return torch.cat([y1, y3, y5, yp], dim=1)`. De blocks van GoogLeNet zelf zijn zo, zonder batch normalization, die het jaar erna kwam.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint het netwerk van de demo, met je `Inception` in het midden, op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Daarna zet hij elke tak van je block om de beurt uit, door zijn channels op 0 te zetten, en telt de testfoto's opnieuw. Waar een module van jou ontbreekt of kapot is, springt die van de course in. Hij heeft CIFAR-10 en oefening 2 nodig, en duurt een paar minuten. Tijdens het trainen laat een grafiek na elke epoch de validation loss zien. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0. Daarna kun je terugspoelen door je run, en zie je je run naast die van de course.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="CIFAR-10 downloaden, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Alleen de demo heeft het nodig. Tijdens het downloaden wacht de hele les.*",
        "busy": "**CIFAR-10 downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van CIFAR-10 uitpakken** · dit duurt een paar seconden",
        "no_net": "*Het downloaden is niet gelukt: de University of Toronto was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "damaged": "*Het uitpakken is niet gelukt: het gedownloade bestand bleek beschadigd, dus de course heeft het verwijderd. Klik nog eens om het opnieuw te downloaden; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 170 MB en de uitgepakte foto's, 184 MB. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*CIFAR-10 staat op je computer: 50.000 trainingsfoto's en 10.000 testfoto's.*",
    })
    return (cifar,)


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train het netwerk")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_inception(cifar, train_button, result_inception):
    cifar, result_inception  # opnieuw uitvoeren als CIFAR-10 binnen is, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van het netwerk op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Inception block** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Inception block** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `Inception` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `Inception` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "inception": "*Je `Inception` in `work/projects/cnn/inception.py` stopt met een fout, of geeft geen 64 maps van 8 bij 8 terug. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "één 3×3-convolutie",
                       "alt": "De loss van het netwerk met het Inception block na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0, die daalt tot epoch 6 en daarna stijgt."}
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
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog steeds.")
            _ref = C.REFERENCE["inception"]
            _tr, _te = C.low_high(_ref["train"]), C.low_high(_ref["test"])
            _off = _result["off"]
            _out = mo.vstack(_pic + [
                mo.md(f"""Met je Inception block leest het netwerk **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, waarvan {C.grouped(_result['block_params'], L.lang)} in zijn block.{_turn} Met één tak uit leest het {_off['b1']} testfoto's goed zonder zijn 1×1-tak, {_off['b3']} zonder zijn 3×3-tak, {_off['b5']} zonder zijn 5×5-tak, en {_off['pool']} zonder zijn pooling-tak. Over 5 seeds, met de code van de course: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's.{_who}"""),
                mo.Html(C.off_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                       "plain": "één 3×3-convolutie", "inception": "Inception block, alle vier de takken",
                                       "b1": "zonder zijn 1×1-tak", "b3": "zonder zijn 3×3-tak", "b5": "zonder zijn 5×5-tak",
                                       "pool": "zonder zijn pooling-tak", "mine": "je run", "course": "de runs van de course",
                                       "alt": "Zes rijen testfoto's goed gelezen, van 1.000: open stippen voor de runs van de course over 5 seeds, met één 3×3-convolutie in het midden, met het Inception block, en met elk van zijn vier takken uit, met een kort streepje bij het gemiddelde van elke rij; en in de rijen van het Inception block een dichte stip voor je run."}, L.lang, mine=_result)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_branches():
    mo.md(r"""
    ### Oefening 3. Op welke tak het netwerk leunt
    """)
    return


@app.cell(hide_code=True)
def choice_branches_cell():
    choice_branches = L.choice(
        "branches",
        question="Over 5 seeds, na 20 epochs, leest het netwerk met het Inception block 553 tot 585 testfoto's goed. Met één tak uit: zonder zijn 1×1-tak, 16 channels, 409 tot 484; zonder zijn 3×3-tak, 32 channels, 453 tot 504; zonder zijn 5×5-tak, 8 channels, 521 tot 567; zonder zijn pooling-tak, 8 channels, 468 tot 541. Wat zeggen de getallen?",
        options={
            "large": "De 5×5-tak doet er het meest toe: zijn filters zien het meest van elke foto",
            "wide": "Hoe meer channels een tak geeft, hoe meer het netwerk hem nodig heeft: de 3×3-tak doet er het meest toe",
            "right": "Het netwerk leunt op zijn 1×1-, 3×3- en pooling-takken, en het minst op zijn 5×5-tak",
        },
        correct="right",
        why={
            "large": "Zonder zijn 5×5-tak leest het netwerk er 521 tot 567 goed: van 2 meer tot 34 minder dan met alle vier, de kleinste verandering van de vier. Op maps van 8 bij 8 ziet een 5×5-filter het grootste deel van de map, maar het netwerk leunt het minst op die tak.",
            "wide": "De 3×3-tak geeft de meeste channels, 32, maar zonder de 1×1-tak, met 16, verliest het netwerk meer: 100 tot 144 testfoto's, tegen 51 tot 107. En de pooling-tak, met maar 8 channels, kost er 44 tot 87.",
            "right": "Zonder zijn 1×1-tak leest het er 100 tot 144 minder goed; zonder zijn 3×3-tak 51 tot 107; zonder zijn pooling-tak 44 tot 87; zonder zijn 5×5-tak van 2 meer tot 34 minder. De training besliste wat elke tak doet: in dit kleine netwerk, op maps van 8 bij 8, leunt het netwerk het minst op de 5×5-tak. Een tak uitzetten na de training laat alleen zien waar het netwerk op leunt: zonder die tak getraind had het het misschien even goed gedaan. Latere versies van Inception zetten twee 3×3-convoluties waar de 5×5 stond: die kijken even ver, met 18 weights per channel in plaats van 25.",
        },
    )
    choice_branches
    return (choice_branches,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een Inception block zet de 1×1 convolutions uit [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) vóór filters van 3 bij 3 en 5 bij 5 naast elkaar, waar [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl) filters van één grootte na elkaar stapelde. Je hield de hoogte en breedte met de formule uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl), stapelde de takken met `torch.cat` uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl), en trainde het block met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `inception.py`. In de volgende les, *Groepen van channels: ResNeXt*, krijgt een block weer veel dunne paden naast elkaar, deze keer allemaal van dezelfde vorm, met een convolutie die de channels in groepen verdeelt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_same_size, result_inception, choice_branches):
    L.wrap_up(result_same_size, result_inception, choice_branches)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
