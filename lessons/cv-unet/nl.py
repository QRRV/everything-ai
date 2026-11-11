# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-unet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-unet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_up():
    quick_up = L.choice(
        "quick-up",
        question="Een feature map van 64 channels en 16 bij 16 plaatsen gaat `nn.ConvTranspose2d(64, 32, 2, stride=2)` in. Wat komt eruit?",
        options={
            "right": "32 maps van 32 bij 32: elke waarde schildert een kernel van 2 bij 2, telkens 2 plaatsen verderop, en de kopieën raken elkaar net",
            "channels": "64 maps van 32 bij 32: een transposed convolution houdt de channels",
            "smaller": "32 maps van 8 bij 8: stride 2 maakt de map half zo groot",
            "plus": "32 maps van 34 bij 34: 16 · 2 + 2 plaatsen",
        },
        correct="right",
        why={
            "right": "Met $H$ plaatsen, stride $s$ en een kernel van $k$ heeft de output $(H - 1) \\cdot s + k = 15 \\cdot 2 + 2 = 32$ plaatsen. Een kernel zo groot als de stride schildert kopieën die elkaar raken zonder te overlappen. En de eerste twee getallen zeggen 64 maps erin, 32 eruit. Vandaag begint elke stap van de weg omhoog met zo'n layer.",
            "channels": "De eerste twee getallen zijn de maps erin en de maps eruit: van 64 naar 32. Net als `nn.Conv2d` kan een transposed convolution elk aantal maps geven.",
            "smaller": "Dat doet een convolutie met stride 2. Een transposed convolution gaat de andere kant op: elke waarde schildert een kopie van de kernel 2 plaatsen verderop, dus wordt de map twee keer zo groot. Zie [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).",
            "plus": "De laatste van de 16 waarden schildert zijn kopie vanaf plaats $(16 - 1) \\cdot 2 = 30$, en de kernel beslaat vanaf daar 2 plaatsen: $(H - 1) \\cdot s + k = 32$ plaatsen, niet $H \\cdot s + k$.",
        },
    )
    quick_up
    return (quick_up,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="In de vorige les had het FCN van de course met output stride 4 een laatste map van 32 bij 32 plaatsen, en deed het het slechter dan het FCN met stride 8, waarvan de laatste map 16 bij 16 is. Waarom?",
        options={
            "right": "Elke plaats van zijn laatste map ziet maar 18 bij 18 pixels van de scène: te weinig van een vorm om zijn soort te zien",
            "coarse": "Zijn laatste map is te grof om de randen te tekenen",
            "params": "Het heeft meer parameters, en leert de trainingsscènes uit zijn hoofd",
            "upsample": "Een transposed convolution kan een map niet 4 keer zo groot maken",
        },
        correct="right",
        why={
            "right": "Met twee blokken ziet elke plaats van de laatste map 18 bij 18 pixels, terwijl de meeste vormen groter zijn. Met stride 8 ziet een plaats 38 bij 38 pixels, ongeveer een hele vorm, maar zijn map is grover. Een FCN moet kiezen tussen genoeg zien en fijne randen tekenen. Het netwerk van vandaag hoeft dat niet.",
            "coarse": "Zijn map is fijner dan die van stride 8: 32 bij 32 plaatsen tegen 16 bij 16. Zijn probleem was wat één plaats ziet: 18 bij 18 pixels, minder dan de meeste vormen.",
            "params": "Het heeft er minder: 15.492 tegen 64.868. Zijn probleem was wat één plaats van zijn laatste map ziet: 18 bij 18 pixels, minder dan de meeste vormen.",
            "upsample": "Dat kan wel: met stride 4, een kernel van 8 en padding 2. Zijn probleem was wat één plaats van zijn laatste map ziet: 18 bij 18 pixels, minder dan de meeste vormen. Zie [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Het FCN van de vorige les moest kiezen. Een grove laatste map ziet genoeg van de scène om een cirkel van een vierkant te onderscheiden, maar heeft maar 16 bij 16 plaatsen om de randen uit te tekenen; een fijne zou fijnere randen kunnen tekenen, maar elk van zijn plaatsen ziet te weinig om de soort te kennen. In 2015 vonden Olaf Ronneberger, Philipp Fischer en Thomas Brox een uitweg, voor beelden uit een microscoop, waar elke cel en elk dun wandje tussen twee cellen telt. Hun **U-Net** gaat omlaag naar grove feature maps, die veel van het beeld zien, en dan weer omhoog; op de weg omhoog stapelt het elke map met de map van dezelfde grootte van de weg omlaag. Die **skip connections** geven de weg omhoog de details van vóór elke pooling van de weg omlaag. Getraind op maar 30 beelden van 512 bij 512 pixels versloeg het de beste methode tot dan toe op de beelden uit een elektronenmicroscoop van de ISBI-challenge, en met hetzelfde netwerk won het twee onderdelen van de ISBI cell tracking challenge van 2015 met grote voorsprong. Vandaag bouw je zelf een U-Net, en train je het op de scènes van de course.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken", "Je CNN op kleurenfoto's", "Je eigen one-stage detector",
                            "Je eigen U-Net", "Eindproject: ResNet met Grad-CAM", "Eindproject: U-Net of detector"],
                           6, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_squares():
    guess_squares = L.guess(
        "unet-squares",
        question="Het FCN van de course haalde een IoU van 0,53 tot 0,71 voor de pixels van de vierkanten. Het U-Net van de course heeft 121.556 parameters, en traint 8 epochs lang op dezelfde 2.000 scènes, net zoals het FCN. Welke IoU haalt het voor de vierkanten, met seed 0?",
        options={
            "worse": "Minder: een dieper netwerk leert langzamer",
            "same": "Ongeveer hetzelfde, 0,5 tot 0,7",
            "better": "Iets beter, 0,7 tot 0,9",
            "right": "Boven de 0,9",
        },
        reveal="0,93 met seed 0, tegen 0,53 tot 0,71 voor het FCN, en het heeft 99,6% van de pixels goed. Over de seeds 0 tot 4 is de mediaan voor de vierkanten 0,92: de middelste van de vijf. Niet elke seed doet het zo goed: met seed 3 halen de vierkanten maar 0,60, zoals de tabel na de slider laat zien.",
    )
    guess_squares
    return (guess_squares,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Omlaag en omhoog.** Het U-Net van de course heeft twee helften. **De weg omlaag** is een CNN zoals je ze kent: een **blok** van twee 3×3-convoluties met 8 channels op 128 bij 128 pixels, max pooling van 2, een blok met 16 channels op 64 bij 64, pooling, een blok met 32 channels op 32 bij 32, pooling, en onderaan een blok met 64 channels op 16 bij 16. Elke plaats van die onderste map ziet 68 bij 68 pixels van de scène, meer dan de 38 bij 38 van het FCN met stride 8: genoeg voor een hele vorm. **De weg omhoog** gaat in drie stappen terug. Elke stap maakt de map twee keer zo groot met de helft van de channels, met een transposed convolution van 2 bij 2 met stride 2, zoals in de instapcheck, en dan een blok. Na de laatste stap is de map weer 128 bij 128, en geeft een 1×1 convolution van 8 channels naar 4 op elke pixel een score voor elke klasse, zoals in het FCN.

        **Skip connections.** Het FCN maakte zijn scores groter uit alleen de grove map: wat de poolings weggooiden, was weg. Een U-Net bewaart de map van elk blok van de weg omlaag, vóór zijn pooling. Op de weg omhoog stapelt het na elke transposed convolution de bewaarde map van dezelfde grootte ervoor: `torch.cat([kept[2], y], dim=1)` zet de 32 channels van de bewaarde map van 32 bij 32, `kept[2]`, en de 32 channels van de opgeschaalde map, `y`, naast elkaar, 64 channels van 32 bij 32. Het blok daarna maakt van die 64 er 32. De bewaarde map heeft weinig van de scène gezien, maar in detail. De opgeschaalde map heeft veel gezien, maar grof: genoeg om te weten welke soort een vorm is. Het blok leert die twee te combineren. Een verbinding die zo een map langs de diepere layers voert, is een **skip connection**; het U-Net heeft er drie, de gestreepte pijlen in het plaatje hieronder.

        **Gestapeld, niet opgeteld.** In [Residual connections](/course/open/nn-residual?lang=nl) sloeg een verbinding ook layers over, maar die telde op: `x + f(x)`, allebei met dezelfde channels. Een skip connection van het U-Net stapelt in plaats daarvan, dus krijgt het blok daarna beide maps heel, en leert het zelf hoe zwaar het ze laat wegen; de prijs is twee keer zoveel channels dat blok in, zoals oefening 1 telt. Het FCN van de paper deed nog iets anders: dat telde de scores van een eerdere map op bij de opgeschaalde scores.

        **Batch normalization in elk blok.** Een blok van het U-Net van de course is twee 3×3-convoluties met padding 1, elk gevolgd door batch normalization en ReLU. `nn.BatchNorm2d(c)` doet wat `nn.BatchNorm1d` deed in [Batch normalization](/course/open/nn-batchnorm?lang=nl), met een channel op de plaats van een neuron: het standaardiseert elk channel over al zijn pixels in alle scènes van de mini-batch, en vermenigvuldigt dan met een geleerde $\gamma$ en telt er een geleerde $\beta$ bij op, één van elk per channel. Het U-Net van de paper heeft geen batch normalization, die toen nieuw was: de paper ervan was een paar maanden eerder verschenen. Zonder batch normalization leert het U-Net van de course in 8 epochs veel slechter, zoals de tabel na de slider laat zien.

        **Mean IoU.** In de vorige les kreeg elke soort zijn eigen IoU. Eén getal voor het hele netwerk is de **mean IoU**: de IoU van de pixels van elke klasse, achtergrond meegeteld, opgeteld en gedeeld door het aantal klassen. Elke klasse telt even zwaar, hoe weinig pixels ze ook heeft. Een netwerk dat overal achtergrond zegt, heeft 87% van de pixels van de testscènes goed, maar zijn mean IoU is maar 0,22: 0,87 voor achtergrond, en 0 voor elk van de drie soorten. De FCN-paper mat zijn netwerken zo, over de 21 klassen van PASCAL VOC.

        **U-Net in de paper.** Het U-Net van de paper is veel groter: het gaat vier keer omlaag, van 64 channels naar 1.024 onderaan, met in totaal 23 convolutional layers. Zijn convoluties hebben geen padding, dus verliest elke convolutie aan elke kant een pixel, en komen de maps van de weg omhoog kleiner uit dan de bewaarde: voor het stapelen snijdt het het midden uit elke bewaarde map, "copy and crop" in zijn figuur. Een beeld van 572 bij 572 pixels geeft scores voor de middelste 388 bij 388; een groter beeld gaat in tegels, en rond de rand van het beeld ziet het netwerk het beeld gespiegeld. De paper beschrijft elke stap omhoog als de map groter maken, gevolgd door een 2×2-convolutie die de channels halveert; de course gebruikt een transposed convolution van 2 bij 2 met stride 2, die de map ook twee keer zo groot maakt met de helft van de channels. Met zo weinig trainingsbeelden vervormt de paper ze willekeurig, alsof ze van rubber zijn, en laat het de pixels tussen twee cellen die elkaar raken zwaarder meetellen in de loss, zodat het netwerk leert de cellen uit elkaar te houden. Het trainen duurde 10 uur op een GPU van toen. Het U-Net van de course geeft elke convolutie in plaats daarvan padding 1, zodat de maps hun grootte houden, en de bewaarde map en de opgeschaalde even groot zijn zonder te snijden.
        """),
        L.watch_out("`torch.cat` stapelt langs de dimensie die je hem geeft. `dim=1` zijn de channels: (1, 32, 32, 32) en (1, 32, 32, 32) worden (1, 64, 32, 32). Zonder `dim=1` stapelt hij langs dimensie 0, de scènes van de batch: (2, 32, 32, 32), en stopt het blok daarna, omdat het 32 channels krijgt waar het voor 64 gemaakt is."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **U-Net** | een netwerk dat omlaag gaat naar grove feature maps en weer omhoog, en elke map van de weg omhoog stapelt met de map van dezelfde grootte van de weg omlaag | `UNet` |
        | **de weg omlaag** | de helft van een U-Net die de maps kleiner maakt met pooling, en ze meer channels geeft | `self.down` |
        | **de weg omhoog** | de helft van een U-Net die de maps weer groter maakt met transposed convolutions | `self.ups` |
        | **skip connection** | een verbinding die een map langs de diepere layers voert: in een U-Net van de weg omlaag naar de stap van de weg omhoog met maps van dezelfde grootte | `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)` |
        | **mean IoU** | de IoU van de pixels van elke klasse, achtergrond meegeteld, gemiddeld over de klassen | `mean_iou` |
        | `torch.cat([a, b], dim=1)` | a en b gestapeld langs dimensie 1, de channels: (N, c, H, W) en (N, d, H, W) worden (N, c + d, H, W) | |
        | `nn.BatchNorm2d(c)` | batch normalization van c channels: elk channel gestandaardiseerd over al zijn pixels in de mini-batch, dan maal $\gamma$ plus $\beta$ | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_unet():
    mo.vstack([
        mo.Html(C.unet_picture({
            "down": "De weg omlaag", "up": "De weg omhoog", "stack": "stapel", "pool": "pool",
            "key": ["Elk vak is een stapel feature maps,", "het getal erboven zijn channels."],
            "alt": "Het U-Net van de course getekend als een U. Links de weg omlaag: vakken van 8 channels op 128 bij 128 pixels, 16 op 64 bij 64 en 32 op 32 bij 32, elk gevolgd door een pijl omlaag voor de pooling. Onderaan een vak van 64 channels op 16 bij 16. Rechts de weg omhoog: op 32, 64 en 128 pixels een gestreept vak, gekopieerd van de weg omlaag, naast het vak dat van onderen kwam; elk paar gaat een blok in van 2c channels naar c. Gestreepte pijlen voeren elke map van de weg omlaag naar de weg omhoog. Aan het eind een 1×1 convolution naar 4 maps.",
        })),
        mo.md(r"""
        Het U-Net van de course. Elk vak is een stapel feature maps: hoe hoger, hoe groter zijn maps, en hoe breder, hoe meer channels. Links de weg omlaag, met na elk van de eerste drie blokken een pooling van 2. Rechts de weg omhoog: elke stap maakt de map van onderen twee keer zo groot, ×2, stapelt de gestreepte kopie van de map van dezelfde grootte van de weg omlaag ervoor, en maakt met een blok van de 2c channels er c. De gestreepte pijlen zijn de drie skip connections.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoeveel levert elke skip connection op? De slider kiest hoeveel van de drie skip connections een U-Net houdt, van onderen af: met 3 het U-Net van de course; met 2 krijgt de laatste stap, op 128 bij 128, geen bewaarde map; met 0 heeft de weg omhoog alleen wat er onderaan uitkwam. Elk van deze U-Nets trainde net zoals het U-Net van de course, seed 0. Links de klasse die het elke pixel van de scène van het uitgewerkte voorbeeld geeft; rechts de echte class map.
    """)
    return


@app.cell(hide_code=True)
def explore_skips_cell():
    def _frame(k):
        k = int(k)
        _skips = f"{k} skip connection" + ("" if k == 1 else "s")
        _html, _stop = C.skips_frame(k, {"title": _skips, "truth": "De echte klassen", "kinds": ("cirkel", "vierkant", "driehoek"),
                                         "alt": f"Links de klasse die het U-Net met {_skips} elke pixel van de scène van het uitgewerkte voorbeeld geeft. Rechts de echte class map van die scène."})
        _a, _b, _c = (C._FCN.number(_v, 2, L.lang) for _v in _stop["ious"])
        return _html, (f"{_skips}. Seed 0, het U-Net in de afbeelding, op de testscènes: een mean IoU van {C._FCN.number(_stop['miou'], 2, L.lang)}, "
                       f"en een IoU van {_a} voor de cirkels, {_b} voor de vierkanten en {_c} voor de driehoeken. "
                       f"Over de seeds 0 tot 4 is de mediaan van de mean IoU {C._FCN.number(_stop['middle'], 2, L.lang)}.")

    explore_skips = L.explore("skips-slider", "skip connections", _frame, start=3, values=[0, 1, 2, 3], role="",
                              task="Wat maakt het U-Net bij 1 skip connection van de twee driehoeken? En van welke soort groeit de IoU het meest van 0 naar 3 skip connections?")
    explore_skips
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        De tabel geeft elk netwerk over de seeds 0 tot 4: de mediaan van de mean IoU, de middelste van de vijf als je ze sorteert, de laagste mean IoU, en de mediaan van de IoU van elke soort. Elke skip connection helpt: van geen naar drie gaat de mediaan van de mean IoU van 0,91 naar 0,93, 0,94 en 0,95. De vierkanten en de driehoeken, met hun hoeken en rechte randen, gaan er het meest op vooruit: van 0,87 naar 0,92 en van 0,84 naar 0,92, terwijl de cirkels van 0,95 naar 0,96 gaan. Zonder skip connections houdt het netwerk de soorten nog wel uit elkaar: zijn onderste map ziet genoeg. Waarom de vierkanten en de driehoeken er het meest op vooruitgaan, is oefening 5.
        """),
        mo.md(C.variants_table({"header": ("Netwerk", "Mean IoU", "Laagste", "Cirkels", "Vierkanten", "Driehoeken"),
                                "names": {"unet": "U-Net, 3 skip connections", "skips2": "2 skip connections", "skips1": "1 skip connection",
                                          "skips0": "geen skip connections", "plain": "3 skip connections, geen batch normalization",
                                          "fcn": "het FCN van de course, stride 8"}}, L.lang)),
        mo.md(r"""
        De kolom *Laagste* laat iets anders zien. Met seed 3 heeft het U-Net van de course 97,3% van de pixels goed, maar is zijn mean IoU maar 0,65: na 8 epochs noemt het 93% van de pixels van de driehoeken vierkant, dus is de IoU van zijn driehoeken 0,04, en die van zijn vierkanten 0,60. Met 1 skip connection tekent seed 0, die in de slider, de driehoeken ook als vierkanten: de IoU van zijn driehoeken is 0,00. Van de 20 U-Nets met batch normalization in de tabel zijn dit de twee die soorten door elkaar halen; de andere 18 houden alle drie de soorten uit elkaar. De driehoeken hebben de minste pixels van de drie soorten, en rechte randen en hoeken zoals de vierkanten; deze twee netwerken noemen ze vierkant, en het deel van de pixels dat goed is, laat dat nauwelijks zien. Daar is mean IoU voor. Twee keer zo lang getraind, 16 epochs, leert het U-Net van seed 3 de driehoeken ook: een mean IoU van 0,97.

        Met deze instellingen, 8 epochs met de learning rate van het FCN, maakt batch normalization nog meer uit. Zonder batch normalization haalt het U-Net met alle drie de skip connections een mediaan van maar 0,43 voor de mean IoU: vier van zijn vijf seeds blijven na 8 epochs onder 0,25 voor de vierkanten en de driehoeken. Het FCN van de vorige les, op dezelfde manier getraind, haalt 0,77. Over de skip connections zegt die vergelijking weinig: het U-Net is ook dieper, heeft batch normalization, en bijna twee keer zoveel parameters, 121.556 tegen 64.868. De vergelijking van de slider zegt wel iets: hetzelfde U-Net, met en zonder skip connections.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Semantic segmentation met een U-Net:

    1. De weg omlaag: een blok van twee 3×3-convoluties met padding 1, elk gevolgd door batch normalization en ReLU; bewaar zijn map; max pooling van 2. Drie keer, met 8, 16 en 32 channels.
    2. Onderaan: nog een blok, met 64 channels, op 16 bij 16.
    3. De weg omhoog, drie keer: een transposed convolution van 2 bij 2 met stride 2 die de channels halveert; de bewaarde map van dezelfde grootte ervoor gestapeld, langs de channels; een blok dat de channels weer halveert.
    4. Een 1×1 convolution naar een score voor elke klasse op elke pixel.
    5. Train met de cross-entropy van elke pixel, zoals het FCN.
    6. Voor elke pixel de klasse met de hoogste score; meet de mean IoU: de IoU van elke klasse, achtergrond meegeteld, gemiddeld.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op de scène uit [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl), opnieuw gemaakt met `scene(18)`. De eerste cel doet stap 1 en 2 met layers die nog niets geleerd hebben, en print de shape van elke bewaarde map en van de onderste map. De tweede cel doet de eerste stap van stap 3: omhoog naar 32 bij 32, en de bewaarde map van die grootte ervoor gestapeld. De derde cel doet stap 6 voor twee verzonnen netwerken: één dat overal achtergrond zegt, en één dat elke pixel goed heeft, behalve dat het het vierkant een cirkel noemt. Voer de cellen uit zoals ze zijn.
    """)
    return


@app.cell
def example_down():
    import torch
    from torch import nn
    from everything_ai.scenes import scene

    # 1. de weg omlaag: een blok van twee 3x3-convoluties met padding 1, elk gevolgd door batch
    #    normalization en ReLU; bewaar zijn map, en dan max pooling van 2
    torch.manual_seed(0)
    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    pool = nn.MaxPool2d(2)
    channels = [3, 8, 16, 32]
    kept = []
    for i in range(3):
        conv1 = nn.Conv2d(channels[i], channels[i + 1], 3, padding=1)
        norm1 = nn.BatchNorm2d(channels[i + 1])
        conv2 = nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1)
        norm2 = nn.BatchNorm2d(channels[i + 1])
        x = torch.relu(norm1(conv1(x)))
        x = torch.relu(norm2(conv2(x)))
        kept.append(x)
        print(x.shape)
        x = pool(x)

    # 2. onderaan: nog een blok, van 32 channels naar 64, op 16 bij 16
    conv1 = nn.Conv2d(32, 64, 3, padding=1)
    norm1 = nn.BatchNorm2d(64)
    conv2 = nn.Conv2d(64, 64, 3, padding=1)
    norm2 = nn.BatchNorm2d(64)
    bottom = torch.relu(norm2(conv2(torch.relu(norm1(conv1(x))))))
    print(bottom.shape)
    return bottom, kept, nn, s, torch


@app.cell
def example_up(bottom, kept, nn, torch):
    # 3. de eerste stap omhoog: twee keer zo groot met de helft van de channels, en dan de
    #    bewaarde map van dezelfde grootte ervoor gestapeld, langs de channels
    up = nn.ConvTranspose2d(64, 32, 2, stride=2)
    y = up(bottom)
    print(y.shape)
    stacked = torch.cat([kept[2], y], dim=1)
    print(stacked.shape)
    return


@app.cell
def example_miou(s):
    import numpy as np

    # de class map van de scène, zoals in de vorige les
    classes = np.zeros((128, 128), dtype=int)
    for shape in s.shapes:
        classes[shape.mask] = shape.label + 1

    # 6. de IoU van de pixels van elke klasse, achtergrond eerst, en hun gemiddelde
    def show_ious(pred):
        total = 0
        for c in range(4):
            p = pred == c
            t = classes == c
            iou = (p & t).sum() / (p | t).sum()
            print("klasse", c, round(iou, 4))
            total += iou
        print("mean IoU", round(total / 4, 4))

    # een netwerk dat overal achtergrond zegt
    show_ious(np.zeros_like(classes))

    # een netwerk dat elke pixel goed heeft, maar het vierkant een cirkel noemt
    mixed = np.zeros_like(classes)
    for shape in s.shapes:
        mixed[shape.mask] = shape.label + 1
    mixed[mixed == 2] = 1
    print("pixels goed", (mixed == classes).mean())
    show_ious(mixed)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de eerste cel houdt elk blok de grootte van zijn map, dankzij de padding, en geeft het hem meer channels: (1, 8, 128, 128), (1, 16, 64, 64) en (1, 32, 32, 32) worden bewaard, en na elk ervan halveert de pooling de map. Het onderste blok krijgt de map van 16 bij 16 en geeft (1, 64, 16, 16). In de tweede cel maakt de transposed convolution daar (1, 32, 32, 32) van: $(16 - 1) \cdot 2 + 2 = 32$ plaatsen, en de helft van de channels. Dat is dezelfde shape als de laatst bewaarde map, dus kan `torch.cat` de twee langs de channels stapelen: (1, 64, 32, 32), de 32 channels van de bewaarde map eerst. Daarna zou een blok van 64 channels naar 32 komen, zoals in oefening 3. In de derde cel krijgt het netwerk dat overal achtergrond zegt een IoU van 0,8925 voor achtergrond, en 0 voor elke soort: een mean IoU van 0,2231, al heeft het 89% van de pixels goed. Het netwerk dat het vierkant een cirkel noemt, heeft 96,0% van de pixels goed, en een IoU van 1 voor achtergrond en de driehoeken. Voor de cirkels is het 0,5152: van de 694 + 653 pixels die het cirkel noemt, zijn alleen de 694 van de cirkel goed. Voor de vierkanten is het 0: het zegt nooit vierkant. Zijn mean IoU is 0,6288: één verwisselde soort kost meer dan een derde van het gemiddelde, voor maar 4% van de pixels.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stack():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De channels van een stapel
    """)
    return


@app.cell(hide_code=True)
def choice_stack():
    choice_stack_channels = L.choice(
        "stack-channels",
        question="De tweede stap van de weg omhoog van het U-Net van de course: de map van 32 bij 32 met 32 channels komt van onderen, en de bewaarde map van 64 bij 64 heeft 16 channels. Hoeveel channels krijgt het blok na het stapelen?",
        options={
            "right": "32: 16 uit de transposed convolution en 16 uit de bewaarde map",
            "add": "16: de twee maps worden channel voor channel opgeteld",
            "half": "48: de 32 channels van onderen en de 16 van de bewaarde map",
            "double": "64: twee keer de 32 channels van onderen",
        },
        correct="right",
        why={
            "right": "De transposed convolution maakt van de 32 channels van onderen er 16, op 64 bij 64. `torch.cat` stapelt de 16 van de bewaarde map ervoor: 32 channels. Dat blok is dus `DoubleConv(32, 16)`, en geeft 16 channels aan de volgende stap.",
            "add": "Optellen doet een residual connection. Een skip connection van het U-Net stapelt, met `torch.cat(..., dim=1)`: de channels van beide maps naast elkaar.",
            "half": "De 32 channels van onderen gaan eerst door de transposed convolution, die de map twee keer zo groot maakt met de helft van de channels: 16. Dan worden de bewaarde 16 ervoor gestapeld: 32.",
            "double": "De transposed convolution halveert de channels: van 32 naar 16. Met de 16 van de bewaarde map krijgt het blok er 32.",
        },
    )
    choice_stack_channels
    return (choice_stack_channels,)


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het blok

        Maak de class `DoubleConv(a, b)` af, het blok van het recept. In `__init__`: `self.conv1`, een 3×3-convolutie met padding 1 van `a` channels naar `b`, en `self.norm1`, zijn batch normalization van `b` channels; dan `self.conv2`, van `b` channels naar `b`, en `self.norm2`. In `forward`: elke convolutie, dan zijn batch normalization, dan ReLU, twee keer. De eerste cel van het uitgewerkte voorbeeld deed hetzelfde in een loop. De check probeert het eerste blok van het U-Net van de course op twee willekeurige beelden van 16 bij 16, telt zijn parameters, en probeert een ander blok op maps die niet vierkant zijn. Als het goed is, bewaart hij `DoubleConv` in je project, als `work/projects/cnn/double_conv.py`; je U-Net bouwt erop.
        """),
        L.task("Maak van `DoubleConv` twee convoluties, elk met batch normalization en ReLU."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_double_conv = L.code_box("double-conv", lines=15)
    box_double_conv
    return (box_double_conv,)


@app.cell(hide_code=True)
def check_block(box_double_conv):
    result_double_conv = L.check("double-conv", box_double_conv.value)
    result_double_conv
    return (result_double_conv,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "double-conv",
        [
            "De eerste cel van het uitgewerkte voorbeeld maakte een blok uit vier layers, `conv1`, `norm1`, `conv2` en `norm2`, en dan twee keer `x = torch.relu(norm1(conv1(x)))`, de tweede keer met `conv2` en `norm2`. In een class gaan de layers in `__init__` als `self.conv1` enzovoort, en de twee regels gaan in `forward`.",
            "`self.conv1 = nn.Conv2d(a, b, 3, padding=1)` en `self.norm1 = nn.BatchNorm2d(b)`. De tweede convolutie krijgt de `b` channels van de eerste: `nn.Conv2d(b, b, 3, padding=1)`, en `self.norm2` is weer van `b` channels.",
            "In `forward`: `x = torch.relu(self.norm1(self.conv1(x)))`, en dan `return ___`, met de tweede convolutie, zijn batch normalization en ReLU, zoals in de regel ervoor.",
        ],
        why="Een 3×3-convolutie maakt elk channel uit alle channels van de 3 bij 3 pixels rond een plaats; padding 1 voegt een rand van nullen toe, zodat de map zijn grootte houdt. Batch normalization standaardiseert dan elk channel over de mini-batch, en ReLU maakt de negatieve waarden 0. Twee convoluties na elkaar zien 5 bij 5 pixels van de map die ze krijgen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unet():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je U-Net

        Maak de class `UNet` af. `__init__` heeft al de weg omlaag, `self.down`, vier blokken gemaakt met je `DoubleConv`; de pooling, `self.pool`; de weg omhoog, `self.ups`, drie transposed convolutions; en de 1×1 convolution, `self.score`. Vul `self.convs` in, de drie blokken na de skip connections: elk krijgt de gestapelde channels en geeft er de helft van. Dan `forward`. De weg omlaag bewaart de map van elk van de eerste drie blokken in een lijst `kept`, vóór zijn pooling, en eindigt met het onderste blok, zonder pooling. De weg omhoog maakt de map in elke stap `i` twee keer zo groot met `self.ups[i]`, stapelt de bewaarde map van dezelfde grootte ervoor langs de channels, en geeft dat aan `self.convs[i]`. Aan het eind komt `self.score`. De check probeert twee willekeurige beelden van 32 bij 32, telt de parameters, en probeert een beeld van 48 bij 64. Als het goed is, bewaart hij `UNet` in je project, als `work/projects/cnn/unet.py`; de demo traint het.
        """),
        L.task("Laat `UNet` omlaag en weer omhoog gaan, met een skip connection bij elke grootte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unet():
    box_unet = L.code_box("unet", lines=26)
    box_unet
    return (box_unet,)


@app.cell(hide_code=True)
def check_unet(box_unet):
    result_unet = L.check("unet", box_unet.value)
    result_unet
    return (result_unet,)


@app.cell(hide_code=True)
def hints_unet():
    L.hints(
        "unet",
        [
            "`self.convs` is een `nn.ModuleList` van drie `DoubleConv`, zoals `self.down`. Op 32 bij 32 heeft de bewaarde map 32 channels en de opgeschaalde ook, zoals in de tweede cel van het uitgewerkte voorbeeld: hoeveel channels krijgt dat blok, en hoeveel geeft het? Oefening 1 deed de stap daarna.",
            "De weg omlaag: `kept = []`, dan `for i in range(3):` met `x = self.down[i](x)`, `kept.append(x)` en `x = self.pool(x)`; na de loop `x = self.down[3](x)`. De weg omhoog: `for i in range(3):`, waarbij stap 0 de laatst bewaarde map nodig heeft, op 32 bij 32, en stap 2 de eerst bewaarde, op 128 bij 128.",
            "In de loop van de weg omhoog: `x = self.convs[i](torch.cat([kept[2 - i], ___], dim=1))`, met de map van onderen twee keer zo groot gemaakt; na de loop `return self.score(x)`.",
        ],
        why="De maps worden van groot naar klein bewaard, dus neemt de weg omhoog, van klein naar groot, ze in de omgekeerde volgorde: eerst `kept[2]` op 32 bij 32, als laatste `kept[0]` op 128 bij 128. `torch.cat(..., dim=1)` stapelt langs de channels, dus krijgt het blok daarna beide maps heel: de bewaarde map eerst, zoals in de figuur van de paper.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_miou():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Mean IoU

        Maak `mean_iou(pred, classes, n)` af, stap 6 van het recept: `pred` en `classes` zoals in `kind_iou`, van één scène of van veel, en `n` het aantal klassen. Hij geeft de IoU van elke klasse van 0 tot en met n − 1 terug, achtergrond meegeteld, gemiddeld over de n klassen. Het bestand importeert je `kind_iou` uit de vorige les: gebruik die. De check probeert de scène van het uitgewerkte voorbeeld met het vierkant een cirkel genoemd, het netwerk dat overal achtergrond zegt, en willekeurige klassen in tensors, met 3 klassen. Als het goed is, bewaart hij `mean_iou` in je project, als `work/projects/cnn/mean_iou.py`; de demo meet je U-Net ermee.
        """),
        L.task("Laat `mean_iou` de IoU van elke klasse geven, gemiddeld."),
    ])
    return


@app.cell(hide_code=True)
def exercise_miou():
    box_mean_iou = L.code_box("mean-iou", lines=8)
    box_mean_iou
    return (box_mean_iou,)


@app.cell(hide_code=True)
def check_miou(box_mean_iou):
    result_mean_iou = L.check("mean-iou", box_mean_iou.value)
    result_mean_iou
    return (result_mean_iou,)


@app.cell(hide_code=True)
def hints_miou():
    L.hints(
        "mean-iou",
        [
            "De derde cel van het uitgewerkte voorbeeld deed het voor 4 klassen: de IoU van elke klasse opgeteld in `total`, en dan gedeeld door 4. Hier geeft `kind_iou(pred, classes, c)` de IoU van klasse `c`, en neemt `n` de plaats van de 4 in.",
            "`total = 0`, dan `for c in range(n):` met `total += kind_iou(pred, classes, c)`.",
            "Na de loop: `return ___`.",
        ],
        why="`range(n)` begint bij 0, dus telt achtergrond mee als klasse, net als de soorten. Elke klasse telt haar IoU één keer op, hoeveel pixels ze ook heeft: een klasse van een paar pixels weegt even zwaar als achtergrond.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint je `UNet`, gemaakt van je `DoubleConv`, op de class maps die je `class_map` van de 2.000 trainingsscènes maakt, 8 epochs lang in mini-batches van 32, net zoals het FCN van de course trainde. Daarna meet hij je U-Net op de 200 testscènes met je `mean_iou`. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; op de computer van de course, met 4 cores, duurde dat ongeveer 9 minuten. De grafiek laat na elke epoch de loss op de testscènes zien. Als hij klaar is, kun je door de epochs terugspoelen, en zien welke klassen je U-Net na elke epoch aan drie testscènes gaf.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je U-Net")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_unet(train_button, result_double_conv, result_unet, result_mean_iou):
    result_double_conv, result_unet, result_mean_iou  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je U-Net te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je U-Net: de loss op de testscènes", "x": "epoch", "y": "test loss",
                                        "alt": "De loss van je U-Net op de 200 testscènes na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Je U-Net** · epoch {_p['epoch']} van {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**Je U-Net** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 3 en 4 af: dan staan je `UNet` en je `mean_iou` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan een half uur. Doet je U-Net meer werk dan nodig, zoals meer blokken of meer channels dan het recept? Volgt het precies het recept, dan is deze computer te traag voor de demo.*",
                "classes": "*Je `class_map` in `work/projects/cnn/classes.py` stopt met een fout, of geeft niet elke pixel van een scène een klasse van 0 tot 3. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) opnieuw.*",
                "unet": "*Je `UNet` in `work/projects/cnn/unet.py` stopt met een fout, geeft niet 4 scores voor elke pixel van een scène, of traint naar een loss die geen getal is. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan opnieuw.*",
                "mean_iou": "*Je `mean_iou` in `work/projects/cnn/mean_iou.py` stopt met een fout, of geeft geen 1 voor class maps vergeleken met zichzelf. Heb je het bestand veranderd? Verwijder het, en check oefening 4 opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course stond in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waar je project staat* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van je U-Net na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _words = {"scene": "scène {seed}", "kinds": ("cirkel", "vierkant", "driehoek"),
                      "alt": "Drie testscènes, elk met de klasse die je U-Net na die epoch elke pixel gaf."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C._FCN.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C._FCN.rewind_picture(_moments[_e - 1], _words),
                        f"Na epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke soort vindt je U-Net als laatste?")]
            else:
                _pic = [mo.Html(C._FCN.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("achtergrond", "cirkels", "vierkanten", "driehoeken"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Je U-Net, met {C._FCN.whole(_result['params'], L.lang)} parameters, heeft {L.number(100 * _result['acc'], 1)}% van de pixels van de 200 testscènes goed. Zijn mean IoU, volgens je `mean_iou`, is {L.number(_result['miou'], 3)}, met een IoU van: {_ious}. Met seed 0, de run die de demo volgt, heeft het U-Net van de course 99,6% van de pixels goed, met een mean IoU van 0,953; over de seeds 0 tot 4 is de mediaan van de mean IoU 0,95, en met seed 3 is hij maar 0,65, zoals de tabel na de slider laat zien.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_skips():
    mo.md(r"""
    ### Oefening 5. Wat de skip connections opleveren
    """)
    return


@app.cell(hide_code=True)
def explain_skips_cell():
    explain_skips = L.explain(
        "why-skips",
        question="Zonder skip connections haalt het U-Net van de course over de seeds 0 tot 4 een mediaan van 0,91 voor de mean IoU; met alle drie 0,95. De weg omlaag is in beide gevallen hetzelfde. Wat geven de skip connections de weg omhoog dat de onderste map niet kan? Kijk naar de vierkanten en de driehoeken in de tabel.",
        model_answer="De onderste map heeft 16 bij 16 plaatsen. Elk daarvan ziet 68 bij 68 pixels van de scène, genoeg om te weten welke soort een vorm is, maar staat voor 8 bij 8 pixels, en na drie poolings, die elk alleen de grootste van 4 waarden hielden, zegt hij minder precies waar in die pixels een rand of een hoek ligt. Zonder skip connections moet de weg omhoog elke vorm uit die grove map tekenen. Met skip connections krijgt elke stap omhoog ook de map van dezelfde grootte van de weg omlaag, van vóór zijn pooling: die heeft maar een klein deel van de scène gezien, maar in detail. Het blok na het stapelen combineert de twee: de soort uit de map die van onderen kwam, de details uit de bewaarde map. De rand van een cirkel volgt uit waar zijn midden ligt en hoe groot hij is, en dat weet de grove map; een hoek is een klein detail dat een plaats van 8 bij 8 pixels uitsmeert. Daarom gaan de vierkanten en de driehoeken er waarschijnlijk het meest op vooruit, van 0,87 naar 0,92 en van 0,84 naar 0,92, terwijl de cirkels, die het netwerk zonder skip connections al goed tekent, maar van 0,95 naar 0,96 gaan.",
        points=[
            "De onderste map is grof: elk van zijn plaatsen staat voor 8 bij 8 pixels, en na drie poolings zegt hij minder precies waar een rand of een hoek ligt.",
            "Een skip connection brengt de map van dezelfde grootte van de weg omlaag, van vóór zijn pooling: die heeft weinig van de scène gezien, maar in detail.",
            "Het blok na het stapelen combineert de twee: de soort uit de map die van onderen kwam, de details uit de bewaarde map.",
        ],
    )
    explain_skips
    return (explain_skips,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De transposed convolution uit [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) maakt nu de maps van een U-Net in elke stap van de weg omhoog twee keer zo groot, de batch normalization uit [Batch normalization](/course/open/nn-batchnorm?lang=nl) werkt nu op elk channel van een feature map, en waar een verbinding in [Residual connections](/course/open/nn-residual?lang=nl) een map optelde, stapelt een skip connection van het U-Net hem. De IoU van elke klasse, achtergrond meegeteld, is één getal geworden, de mean IoU. Je `UNet`, gebouwd uit je `DoubleConv`, is het U-Net van de course: over vijf seeds haalt het op de scènes van de course een mediaan van 0,95 voor de mean IoU, en het FCN van de course, op dezelfde manier gemeten, 0,77. Je project `cnn` heeft nu `double_conv.py`, `unet.py` en `mean_iou.py`. In de volgende les, *Dilated convolutions en DeepLab*, spreidt een convolutie zijn kernel uit met gaten: hij ziet een groter deel van de scène zonder de map kleiner te maken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stack_channels, result_double_conv, result_unet, result_mean_iou, explain_skips):
    L.wrap_up(choice_stack_channels, result_double_conv, result_unet, result_mean_iou, explain_skips)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
