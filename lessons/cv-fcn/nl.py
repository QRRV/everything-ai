# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-fcn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.
# termen:allow deconvolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fcn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) en [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pointwise():
    quick_pointwise = L.choice(
        "quick-pointwise",
        question="Een netwerk heeft een feature map van 64 channels en 16 bij 16 plaatsen. Een 1×1-convolutie van 64 naar 4 channels: wat geeft die?",
        options={
            "right": "4 maps van 16 bij 16: op elk van de 256 plaatsen maakt dezelfde linear layer van de 64 getallen van die plaats er 4",
            "whole": "4 getallen voor de hele map, zoals een linear layer aan het eind van een classifier",
            "neighbors": "4 maps van 16 bij 16, waarin elke plaats gemengd is met zijn 8 buren",
            "smaller": "4 maps van 8 bij 8: net als pooling maakt hij de map half zo groot",
        },
        correct="right",
        why={
            "right": "Een 1×1-convolutie kijkt naar één plaats tegelijk: hij mengt de channels van die plaats, met dezelfde weights op elke plaats. Dus de map houdt zijn 16 bij 16 plaatsen, en krijgt 4 channels. Vandaag geeft hij op elke plaats een score voor elke klasse.",
            "whole": "Een linear layer aan het eind zou één antwoord geven voor de hele map. Een 1×1-convolutie geeft een antwoord op elke plaats: een linear layer op elke pixel. Zie [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl).",
            "neighbors": "Dat is een 3×3-convolutie. Een 1×1-convolutie kijkt niet naar de buren: hij mengt alleen de 64 channels van elke plaats.",
            "smaller": "Een 1×1-convolutie met stride 1 houdt elke plaats: alleen het aantal channels verandert, van 64 naar 4.",
        },
    )
    quick_pointwise
    return (quick_pointwise,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask-iou",
        question="Het mask van een cirkel heeft 300 pixels. Een model markeert 250 pixels als die cirkel, en 200 daarvan liggen in het mask. Wat is de IoU van de twee masks?",
        options={
            "right": "200 / 350, ongeveer 0,57: de vereniging is 300 + 250 − 200 = 350 pixels",
            "found": "200 / 300, ongeveer 0,67: het deel van de cirkel dat het model vond",
            "marked": "200 / 250 = 0,8: het deel van de gemarkeerde pixels dat klopt",
            "sum": "200 / 550, ongeveer 0,36: de overlap gedeeld door de twee masks samen",
        },
        correct="right",
        why={
            "right": "De overlap is 200 pixels, en de twee masks bedekken samen 300 + 250 − 200 = 350: de 200 van de overlap zitten in allebei, dus die tel je één keer. Bij masks telt `(a & b).sum()` de overlap en `(a | b).sum()` de vereniging.",
            "found": "Dat telt alleen de pixels die het model miste. Een model dat elke pixel van de scène markeert, zou 1 krijgen. De IoU deelt door de vereniging: 300 + 250 − 200 = 350. Zie [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl).",
            "marked": "Dat telt alleen de pixels die het model ten onrechte markeerde. Een model dat één pixel van de cirkel markeert, zou 1 krijgen. De IoU deelt door de vereniging: 350.",
            "sum": "300 + 250 telt de 200 pixels van de overlap twee keer. De vereniging telt elke pixel één keer: 350.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een detector geeft elke vorm een box, en een box is grof: zijn hoeken zijn achtergrond, en de boxes van twee vormen die elkaar raken, overlappen. Wie een vorm uit een foto wil knippen, of de weg voor een auto of een vlek op een scan wil opmeten, heeft de klasse van elke pixel nodig: **semantic segmentation**. In 2015 lieten Long, Shelhamer en Darrell zien dat een classifier dat kan met twee veranderingen: geen linear layers aan het eind, zodat hij een beeld van elke grootte aanneemt, en **transposed convolutions**, die zijn kleine feature maps weer zo groot maken als het beeld. Hun **FCN**, een fully convolutional network, deelde de pixels van de foto's van PASCAL VOC beter in dan alle methoden ervoor, in minder dan een vijfde seconde per foto, waar de beste daarvan er zo'n 50 seconden over deed.
    """)
    return


@app.cell(hide_code=True)
def predict_background():
    guess_background = L.guess(
        "background-share",
        question="Een netwerk dat voor elke pixel achtergrond zegt, wat het ook ziet. Welk deel van alle pixels van de 200 testscènes van de course heeft het goed?",
        options={
            "none": "Bijna niets: het vindt geen enkele vorm",
            "quarter": "Ongeveer een kwart: achtergrond is een van 4 klassen",
            "half": "Ongeveer de helft",
            "right": "Bijna 9 op de 10",
        },
        reveal="87%: achtergrond beslaat 87% van de pixels van de testscènes, dus een netwerk dat niets geleerd heeft, heeft bijna 9 op de 10 pixels goed. Het vindt alleen geen cirkel, vierkant of driehoek: de IoU van de pixels van elke soort is 0. Daarom meet deze les elke soort apart.",
    )
    guess_background
    return (guess_background,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een klasse voor elke pixel.** In [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl) kreeg elke vorm een mask: `True` voor elke pixel van die vorm. Semantic segmentation geeft elke pixel van een scène een klasse: hier 0 voor achtergrond, en 1, 2 en 3 voor cirkel, vierkant en driehoek, het label van een vorm plus 1. Alle cirkels krijgen dezelfde klasse: die zegt wat een pixel laat zien, niet bij welke vorm hij hoort. De **class map**, een array zo groot als de scène met de klasse van elke pixel, is wat het netwerk leert; oefening 1 maakt hem uit de masks.

        **Geen linear layers.** Een classifier, zoals je CNN's, eindigt met linear layers: die willen een feature map van één grootte, en geven één antwoord voor het hele beeld. FCN behandelt elke linear layer als een convolutie waarvan de kernel zijn hele input beslaat: een linear layer op een map van 7 bij 7 wordt een 7×7-convolutie, en de linear layers daarna 1×1-convoluties. Een 1×1-convolutie is een linear layer op elke pixel, zoals in [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl): uit de channels van elke plaats van de map maakt hij een score voor elke klasse op die plaats. Met alleen convoluties en pooling neemt een netwerk een beeld van elke grootte aan, en groeien zijn maps mee.

        **Het FCN van de course.** Drie blokken van een 3×3-convolutie, ReLU en max pooling van 2, met 16, 32 en 64 channels, maken van een scène van 128 bij 128 pixels een map van 16 bij 16: elke plaats daarvan staat voor een vierkant van 8 bij 8 pixels. Die factor 8 is de **output stride**. Nog een 3×3-convolutie met ReLU, en dan geeft een 1×1-convolutie van 64 channels naar 4 op elke plaats een score voor achtergrond en voor elke soort. Met de layer die zijn scores weer zo groot maakt als de scène, verderop, heeft het netwerk 64.868 parameters.

        **Upsampling.** De scores moeten weer zo groot worden als de scène, één score per klasse voor elke pixel: **upsampling**. In [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl) maakte je een beeld op twee manieren groter: nearest neighbor herhaalt elke pixel, en bilineaire interpolatie mengt de oude pixels rond elke nieuwe, de dichtstbijzijnde het zwaarst. Voor de scores van 16 bij 16 geeft nearest neighbor blokken van 8 bij 8 pixels; bilineaire interpolatie geeft zachte overgangen ertussen.

        **Transposed convolution.** Een convolutie met stride 2 schuift zijn kernel in stappen van 2 over de input, en elke stap geeft één getal: de map wordt kleiner. Een **transposed convolution** gaat de andere kant op: elke waarde van de input schildert een kopie van de kernel op de output, maal die waarde, en de volgende waarde schildert zijn kopie `stride` plaatsen verderop; waar kopieën overlappen, tellen ze op. Met H waarden in een rij, een stride $s$ en een kernel van $k$ heeft de output $(H - 1) \cdot s + k$ waarden, en padding $p$ snijdt er aan elk eind $p$ af: $(H - 1) \cdot s - 2p + k$. Een kernel van enen zo groot als de stride schildert blokken die niet overlappen: nearest neighbor. Een kernel twee keer zo groot als de stride, die in het midden het grootst is en in een rechte lijn afloopt naar zijn randen, laat de kopieën van twee buren overlappen en mengen: bilineaire interpolatie, zoals het uitgewerkte voorbeeld laat zien. En de kernel is een weight als elke andere, dus het netwerk kan leren hoe het moet upsamplen, of de kernel kan vast staan, bijvoorbeeld op bilineaire upsampling. Het FCN van de course laat hem als bilineaire upsampling beginnen, en laat het trainen hem veranderen; de tabel na de slider laat zien wat dat oplevert. In PyTorch is het `nn.ConvTranspose2d`. De paper noemt het een achterwaartse convolutie, omdat hij doet wat een convolutie met die stride in de backward pass doet; hij heet ook wel deconvolution, al maakt hij een convolutie niet ongedaan.

        **Trainen en meten.** De loss is de cross-entropy van elke pixel: `nn.CrossEntropyLoss()` neemt scores van shape (N, 4, H, W) en klassen van shape (N, H, W), en geeft hun gemiddelde over alle pixels. Het antwoord van het netwerk voor elke pixel is de klasse met de hoogste score: `out.argmax(1)`. Welk deel van de pixels het goed heeft, zegt weinig, zoals de gok liet zien. Daarom wordt elke soort apart gemeten, met de IoU van zijn pixels, zoals bij masks in [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl): de pixels die die soort zijn in het antwoord én in de class map, gedeeld door de pixels die die soort zijn in een van beide. Deze les telt elke soort over alle pixels van de 200 testscènes; de paper neemt ook het gemiddelde over de klassen, zoals de volgende les doet.

        **FCN in de paper.** De paper maakt de classifier VGG16, getraind op de foto's van ImageNet, fully convolutional: zijn laatste map is 32 keer zo klein als de foto, en een 1×1-convolutie geeft op elke plaats daarvan 21 scores, voor de 20 klassen van PASCAL VOC en achtergrond. Eén transposed convolution met stride 32, vast op bilineaire upsampling, maakt de scores zo groot als de foto. Met de klassen van PASCAL VOC 2011 haalde dat FCN een gemiddelde IoU van 56,0% op foto's waar het niet van leerde; op dezelfde manier gemaakt uit AlexNet 39,8%, en uit GoogLeNet 42,5%. De paper noemt zijn grens zelf: "de stride van 32 pixels in de laatste layer, die voorspelt, beperkt de schaal van de details in de opgeschaalde output". Daarom telt hij er scores bij op uit de fijnere maps van eerdere layers, met skip connections, die elk twee keer zo groot worden gemaakt door een transposed convolution die als bilineaire upsampling begint en leert: met stride 8 haalde zijn FCN-8s 62,7% waar stride 32 59,4% haalde, op een deel van die foto's. Je volgende les bouwt op dat idee verder.
        """),
        L.watch_out("`nn.CrossEntropyLoss` wil de class map als gehele getallen: `dtype=int` in `np.zeros` maakt ze zo, en `torch.tensor` houdt ze zo. Met een class map van floats stopt hij met een fout."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **semantic segmentation** | elke pixel van een beeld een klasse geven: wat hij laat zien, niet bij welk voorwerp hij hoort | `class_map` |
        | **class map** | een array zo groot als het beeld, met de klasse van elke pixel | `classes` |
        | **fully convolutional network** (FCN) | een netwerk van alleen convoluties en pooling, zonder linear layers aan het eind: het neemt een beeld van elke grootte aan, en geeft een map die meegroeit | `FCN` |
        | **output stride** | hoeveel keer zo klein als het beeld de laatste map van een netwerk is | `stride` |
        | **upsampling** | een map groter maken: met nearest neighbor, bilineaire interpolatie, of een transposed convolution die leert hoe | |
        | **transposed convolution** | elke waarde van de input schildert een kopie van de kernel, maal die waarde, `stride` plaatsen uit elkaar; waar kopieën overlappen, tellen ze op | `nn.ConvTranspose2d` |
        | `nn.ConvTranspose2d(c_in, c_out, k, stride=s, padding=p)` | een transposed convolution van `c_in` maps naar `c_out` maps; H plaatsen worden er $(H - 1) \cdot s - 2p + k$ | |
        | `nn.CrossEntropyLoss()(out, classes)` | met scores van shape (N, 4, H, W) en klassen van shape (N, H, W): de gemiddelde cross-entropy van alle pixels | |
        | `out.argmax(1)` | voor elke pixel het nummer van de klasse met de hoogste score: de plaats van de grootste waarde langs dimensie 1, zoals `np.argmax` met een as | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        mo.Html(C.steps_picture({
            "scene": "De scène", "coarse": "16 × 16 plaatsen", "up": "Na upsampling", "truth": "De echte klassen",
            "kinds": ("cirkel", "vierkant", "driehoek"),
            "alt": "Vier beelden van 128 bij 128 pixels. De scène van het uitgewerkte voorbeeld, met twee driehoeken, een vierkant en een cirkel. De klasse met de hoogste score op elke plaats van de map van 16 bij 16 van het getrainde FCN van de course: de vormen als grove blokken. De class map na de transposed convolution: de vormen met gladde randen. En de echte class map. Elke soort heeft een eigen kleur.",
        })),
        mo.md(r"""
        Wat het getrainde FCN van de course van de scène van het uitgewerkte voorbeeld maakt. Als tweede de klasse met de hoogste score op elke plaats van zijn laatste map, vóór de upsampling: elk vierkantje staat voor 8 bij 8 pixels van de scène. Als derde de klasse van elke pixel na zijn transposed convolution, die als bilineaire upsampling begon en leerde: de randen volgen de vormen veel beter dan de blokken van 8 bij 8. Als laatste de class map waar het van leerde. Kleur betekent hier een soort, elk met een eigen teken in de legenda: achtergrond heeft de kleur van de pagina.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoe grof mag de laatste map zijn? De slider kiest de output stride, en daarmee een FCN dat de course trainde, seed 0: met 2, 3, 4 of 5 blokken, elk met nog een pooling van 2, en dezelfde training. Links de scène van het uitgewerkte voorbeeld, met één plaats van de laatste map rond het midden van de cirkel als doorgetrokken vierkant, en het deel van de scène dat die plaats ziet, zijn receptive field, als gestreept vierkant. Rechts de klasse van elke pixel die dat FCN de scène geeft.
    """)
    return


@app.cell(hide_code=True)
def explore_stride_cell():
    def _frame(s):
        s = int(s)
        _html, _stop = C.stride_frame(s, {"sees": "Wat één plaats ziet", "title": "Stride {stride}", "kinds": ("cirkel", "vierkant", "driehoek"),
                                          "alt": "Links de scène van het uitgewerkte voorbeeld met één plaats van de laatste map, {stride} bij {stride} pixels, als doorgetrokken vierkant, en de {field} bij {field} pixels die hij ziet als gestreept vierkant. Rechts de klasse van elke pixel die het FCN met output stride {stride} de scène geeft, met een laatste map van {places} bij {places} plaatsen."})
        _ious = ", ".join(f"{_name} {C.span_text(_span, L.lang)}" for _name, _span in zip(("cirkels", "vierkanten", "driehoeken"), _stop["ious"]))
        return _html, f"Stride {s}: een laatste map van {_stop['places']} bij {_stop['places']} plaatsen, die elk {_stop['field']} bij {_stop['field']} pixels zien. De IoU op de testscènes over de seeds 0 tot 4: {_ious}."

    explore_stride = L.explore("stride-slider", "stride", _frame, start=8, values=[4, 8, 16, 32], role="",
                               task="Bij welke strides krijgen hele vormen de verkeerde soort, en bij welke krijgt één vorm verschillende soorten? Vergelijk het gestreepte vierkant met de vormen, die 10 tot 22 pixels van hun midden reiken.")
    explore_stride
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        De tabel geeft elk FCN van de slider over de seeds 0 tot 4, met het deel van de scène dat één plaats van zijn laatste map ziet. Met stride 4 is dat maar 18 bij 18 pixels, minder dan de meeste vormen, en heeft het netwerk maar 15.492 parameters: het geeft delen van één vorm verschillende soorten, en zijn IoU blijft op 0,54 tot 0,66 voor de cirkels, 0,07 tot 0,32 voor de vierkanten en 0,30 tot 0,46 voor de driehoeken. Stride 8 ziet 38 bij 38 pixels, ongeveer een hele vorm, en doet het het best. Met stride 16 en 32 staat een plaats van de laatste map voor 16 bij 16 of 32 bij 32 pixels, en na 8 epochs geeft het FCN zelfs hele vormen de verkeerde soort. Dat komt niet alleen door de grove map: hun netwerken zijn een of twee blokken dieper, en leren langzamer. Twee keer zo lang getraind doet stride 16 het met vier van de vijf seeds ongeveer even goed als stride 8, 0,82 tot 0,86 voor de cirkels; met seed 0 blijft het op 0,66. Stride 32 komt twee keer zo lang getraind tot 0,52 tot 0,75 voor de cirkels: nog altijd ver onder stride 8.
        """),
        mo.md(C.stride_table({"header": ("Output stride", "Ziet", "Cirkels", "Vierkanten", "Driehoeken"), "name": "{stride}, {epochs} epochs"}, L.lang)),
        mo.md(r"""
        Hoeveel maakt het uit hoe de transposed convolution begint, en dat hij leert? Vier FCN's met stride 8, elk getraind met de seeds 0 tot 4 zoals het FCN van de course, en hetzelfde in alles behalve hun transposed convolution:
        """),
        mo.md(C.upsampling_table({"header": ("Upsampling", "Cirkels", "Vierkanten", "Driehoeken", "Pixels goed"),
                                  "names": {"s8": "geleerd, bilineair begonnen", "s8-random": "geleerd, willekeurig begonnen",
                                            "s8-bilinear": "bilineair, vast, zoals in de paper", "s8-nearest": "nearest neighbor, vast"}}, L.lang)),
        mo.md(r"""
        Bilineair begonnen en geleerd, zoals in het FCN van de course, doet de transposed convolution het het best, met elke seed en voor elke soort. Vast op bilineaire upsampling, zoals de laatste van het FCN van de paper, doet hij het met seed 0 en 2 bijna even goed, en met de andere slechter: op deze scènes helpt het dat het trainen de kernel mag veranderen. Willekeurig begonnen leert hij ook, maar met elke seed eindigt hij lager dan bilineair begonnen. Nearest neighbor, vast, doet het het slechtst: voor elke soort liggen zowel de laagste als de hoogste van zijn vijf seeds onder die van elke andere rij. Elke rand wordt een trap van blokken van 8 bij 8 pixels. Zelfs een trap die getekend is uit de echte class maps, met in elk blok de klasse die de meeste van zijn 64 pixels hebben, zou maar 0,78 halen voor de cirkels, 0,72 voor de vierkanten en 0,64 voor de driehoeken; het FCN met nearest neighbor blijft op 0,50 tot 0,67, 0,14 tot 0,42 en 0,28 tot 0,46.

        En een FCN neemt een scène van elke grootte aan. Het FCN van de course, getraind op scènes van 128 pixels, geeft zonder enige verandering elke pixel van scènes van 96 en 192 pixels een klasse, met een IoU voor de cirkels van 0,69 tot 0,77 op 96 pixels en 0,82 tot 0,92 op 192. Op 192 pixels doet het het beter dan op de 128 waar het van leerde. De vormen houden hun grootte, dus overlappen ze op een grotere scène minder, en dat helpt waarschijnlijk: van de vormen van de 200 testscènes is op 192 pixels 12% deels verborgen achter een andere, tegen 25% op 128 pixels en 44% op 96.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Semantic segmentation met een FCN:

        1. De class map van elke scène: overal 0, en het label van een vorm plus 1 op de pixels van zijn mask.
        2. Blokken van een 3×3-convolutie, ReLU en max pooling van 2, tot de map $s$ keer zo klein is als de scène; dan nog een 3×3-convolutie met ReLU.
        3. Een 1×1-convolutie naar een score voor elke klasse op elke plaats.
        4. Een transposed convolution met stride $s$, een kernel van $2s$ en padding $s/2$, die als bilineaire upsampling begint: de scores zo groot als de scène.
        5. Train met de cross-entropy van elke pixel.
        6. Voor elke pixel de klasse met de hoogste score; meet de IoU van de pixels van elke soort.
        """),
        L.watch_out("Geef de transposed convolution één kernel voor elke klasse, van map c naar map c, en 0 voor de andere paren. Met de bilineaire kernel voor elk paar krijgt elke klasse de som van de scores van alle klassen, en zeggen de opgeschaalde scores niet meer welke klasse het hoogst is."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op de scène van [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl), opnieuw gemaakt met `scene(18)`. De eerste cel doet stap 1: `np.zeros` met `dtype=int` maakt een array van gehele getallen, zo groot als het beeld van de scène, en elke vorm zet zijn klasse op de pixels van zijn mask. Daarna doet hij stap 6 voor een netwerk dat overal achtergrond zegt. De tweede cel schildert een transposed convolution met de hand, stap 4 in één rij: de rij van de gok van [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl), 0, 100 en 200, twee keer zo lang gemaakt met de bilineaire kernel van 4, en dan met een kernel van enen. De derde cel bouwt stap 2 tot en met 5 met layers die nog niets geleerd hebben, en print na elke stap de shape. Voer de cellen uit zoals ze zijn.
    """)
    return


@app.cell
def example_classes():
    import numpy as np
    from everything_ai.scenes import scene

    # 1. de class map: 0 voor achtergrond, het label van een vorm plus 1 op de pixels van zijn mask
    s = scene(18)
    classes = np.zeros(s.image.shape[:2], dtype=int)
    for shape in s.shapes:
        classes[shape.mask] = shape.label + 1
    print([int((classes == k).sum()) for k in range(4)])

    # 6. een netwerk dat overal achtergrond zegt: het deel van de pixels dat goed is, en de IoU
    #    van de pixels van de cirkel
    pred = np.zeros_like(classes)
    print((pred == classes).mean())
    p = pred == 1
    t = classes == 1
    print((p & t).sum() / (p | t).sum())
    return (classes, np, s)


@app.cell
def example_row():
    import torch

    # 4. een transposed convolution met de hand, in één rij: elke waarde schildert de kernel, maal die
    #    waarde, elke keer 2 plaatsen verderop; waar de kopieën overlappen, tellen ze op
    row = torch.tensor([0.0, 100.0, 200.0])
    kernel = torch.tensor([0.25, 0.75, 0.75, 0.25])
    stride = 2
    out = torch.zeros((len(row) - 1) * stride + len(kernel))
    for i in range(len(row)):
        out[i * stride:i * stride + len(kernel)] += row[i] * kernel
        print(out)
    print(out[1:-1])   # padding 1: aan elk eind één plaats eraf

    # nearest neighbor: een kernel van enen zo groot als de stride
    ones = torch.tensor([1.0, 1.0])
    near = torch.zeros(len(row) * stride)
    for i in range(len(row)):
        near[i * stride:i * stride + len(ones)] += row[i] * ones
    print(near)
    return (torch,)


@app.cell
def example_fcn(classes, np, s, torch):
    from torch import nn

    # 2. blokken van een 3x3-convolutie, ReLU en max pooling van 2: de map 8 keer zo klein
    torch.manual_seed(0)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    convs = nn.ModuleList([nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)])
    pool = nn.MaxPool2d(2)
    for conv in convs:
        x = pool(torch.relu(conv(x)))
        print(x.shape)

    # 3. nog een 3x3-convolutie, en een 1x1-convolutie naar een score voor elke klasse op elke plaats
    extra = nn.Conv2d(64, 64, 3, padding=1)
    score = nn.Conv2d(64, 4, 1)
    coarse = score(torch.relu(extra(x)))
    print(coarse.shape)

    # 4. een transposed convolution met stride 8, een kernel van 16 en padding 4: zo groot als de scène
    up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)
    out_scene = up(coarse)
    print(out_scene.shape)

    # 5. de cross-entropy van elke pixel, tegen de class map van de eerste cel
    loss = nn.CrossEntropyLoss()(out_scene, torch.tensor(classes).unsqueeze(0))
    print(loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste cel telt 14.622 pixels achtergrond, 694 van de cirkel, 653 van het vierkant en 415 van de twee driehoeken: samen 16.384, 128 maal 128. Een netwerk dat overal achtergrond zegt, heeft er 0,892 van goed, en de IoU van de pixels van de cirkel is 0,0: van zijn 694 pixels noemt het er geen enkele cirkel. In de tweede cel schildert de 0 een kopie van nullen; de 100 telt er twee plaatsen verderop 25, 75, 75 en 25 bij op; en de 200 nog twee plaatsen verderop 50, 150, 150 en 50, waar de kopieën optellen tot 75 + 50 = 125 en 25 + 150 = 175. Zonder de eerste en de laatste plaats, padding 1, is dat 0, 25, 75, 125, 175, 150: de rij van bilineaire interpolatie, 0 25 75 125 175 200, behalve de laatste waarde. Voorbij elk eind van de rij mengt de transposed convolution een 0 in: aan het eind, waar geen waarde na de 200 meemengt, geeft dat 150 in plaats van 200; aan het begin zie je het niet, omdat de eerste waarde zelf 0 is. De kernel van enen geeft 0, 0, 100, 100, 200, 200: nearest neighbor. In de derde cel maakt elk blok de map half zo groot, en groeien de channels van 16 naar 64: (1, 64, 16, 16) na drie blokken. De 1×1-convolutie maakt daar 4 maps van 16 bij 16 van, en de transposed convolution maakt die 128 bij 128: $(16 - 1) \cdot 8 - 2 \cdot 4 + 16 = 128$. De loss is 1,3862, ongeveer $\log 4 = 1{,}386$: het netwerk heeft nog niets geleerd, dus geeft het elke klasse ongeveer dezelfde score, alsof het een van 4 gokt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_classes():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De class map

        Maak `class_map(s)` af, stap 1 van het recept als functie: `s` is een scène van de course, met `s.image`, haar beeld van shape (rijen, kolommen, 3), en `s.shapes`, haar vormen, elk met `shape.mask` en `shape.label`. Hij geeft een array terug zo groot als de scène: 0 voor achtergrond, en het label van een vorm plus 1 op de pixels van zijn mask. De check probeert de scène van het uitgewerkte voorbeeld, nog drie scènes, en scènes van 96 en 192 pixels. Als hij klopt, bewaart hij `class_map` in je project, als `work/projects/cnn/classes.py`; de demo maakt er de klassen mee waar hij van leert.
        """),
        L.task("Laat `class_map` de klasse van elke pixel van een scène geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_classes():
    box_class_map = L.code_box("class-map", lines=4)
    box_class_map
    return (box_class_map,)


@app.cell(hide_code=True)
def check_classes(box_class_map):
    result_class_map = L.check("class-map", box_class_map.value)
    result_class_map
    return (result_class_map,)


@app.cell(hide_code=True)
def hints_classes():
    L.hints(
        "class-map",
        [
            "De eerste cel van het uitgewerkte voorbeeld, als functie: daar begon `classes` als een array met nullen, en zette elke vorm zijn klasse op de pixels van zijn mask.",
            "`np.zeros(s.image.shape[:2], dtype=int)` maakt een array van gehele getallen zo groot als het beeld van de scène, welke grootte het ook heeft. Dan `for shape in s.shapes:`, en `classes[shape.mask] = ...` zet elke pixel van die vorm in één keer.",
            "De klasse van een vorm is zijn label plus 1, zodat 0 overblijft voor achtergrond: `classes[shape.mask] = ___`, en na de loop `return classes`.",
        ],
        why="`shape.label` is 0 voor een cirkel, 1 voor een vierkant en 2 voor een driehoek; `shape.label + 1` maakt er de klassen 1, 2 en 3 van. Een pixel waar geen mask overheen ligt, houdt de 0 van `np.zeros`: achtergrond.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_transposed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een transposed convolution met de hand

        Maak `transposed(x, kernel, stride)` af, de tweede cel van het uitgewerkte voorbeeld in twee richtingen: `x` is een tensor van shape (H, W), `kernel` een van shape (k, k). Elke waarde `x[i, j]` schildert de kernel maal die waarde op de output, met de linkerbovenhoek op rij `i * stride` en kolom `j * stride`; waar kopieën overlappen, tellen ze op. Hij geeft de output terug, van shape $((H - 1) \cdot s + k, (W - 1) \cdot s + k)$, zonder padding. De check probeert de bilineaire kernel van 4 bij 4 op een beeld van 2 bij 2, een kernel van enen, willekeurige getallen met strides onder, gelijk aan en boven de grootte van de kernel, en vergelijkt je functie met `nn.ConvTranspose2d` van PyTorch.
        """),
        L.task("Laat `transposed` een transposed convolution schilderen, kopie voor kopie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_transposed():
    box_transposed = L.code_box("transposed", lines=5)
    box_transposed
    return (box_transposed,)


@app.cell(hide_code=True)
def check_transposed(box_transposed):
    result_transposed = L.check("transposed", box_transposed.value)
    result_transposed
    return (result_transposed,)


@app.cell(hide_code=True)
def hints_transposed():
    L.hints(
        "transposed",
        [
            "In het uitgewerkte voorbeeld begon `out` als nullen van lengte (H − 1) · stride + k, en telde waarde `i` `row[i] * kernel` op bij het stuk van `i * stride` tot `i * stride + len(kernel)`. Doe hetzelfde voor de rijen en voor de kolommen.",
            "`k = len(kernel)`, en `out = torch.zeros((len(x) - 1) * stride + k, (len(x[0]) - 1) * stride + k)`. Dan een loop over de rijen, `for i in range(len(x)):`, met daarbinnen een loop over de kolommen, `for j in range(len(x[0])):`.",
            "Binnen beide loops gaat de kopie van waarde `x[i, j]` naar de rijen vanaf `i * stride` en de kolommen vanaf `j * stride`, van elk `k`: `out[i * stride:i * stride + k, j * stride:j * stride + k] += ___`, en na de loops `return out`.",
        ],
        why="`x[i, j] * kernel` is de kopie van de kernel voor die waarde, en `+=` telt hem op bij wat de kopieën van zijn buren daar al schilderden. Met de bilineaire kernel en stride 2 overlappen twee buren in elke richting 2 plaatsen, en daar mengen ze.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_upsampler():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Bilineair begonnen

        Maak `upsampler(classes, stride)` af, stap 4 van het recept: hij geeft een `nn.ConvTranspose2d` terug die `classes` maps `stride` keer zo groot maakt, met een kernel van 2 · stride, padding stride // 2 en geen bias, `bias=False`. Hij begint als bilineaire upsampling: `nn.init.zeros_(up.weight)` zet al zijn weights op 0, en dan geeft map c erin map c eruit met `bilinear_kernel(2 * stride)`, die gegeven is. `up.weight[c, c]` is die kernel; PyTorch laat je hem alleen zetten binnen `with torch.no_grad():`. De check probeert het FCN van de course, 4 maps van 16 bij 16 die 8 keer zo groot worden, andere aantallen klassen en andere strides, en of het trainen hem kan veranderen. Als hij klopt, bewaart hij `upsampler` in je project, als `work/projects/cnn/upsample.py`; de demo bouwt er het FCN van de course mee.
        """),
        L.task("Laat `upsampler` de transposed convolution van het FCN van de course geven, die als bilineaire upsampling begint."),
    ])
    return


@app.cell(hide_code=True)
def exercise_upsampler():
    box_upsampler = L.code_box("upsampler", lines=5)
    box_upsampler
    return (box_upsampler,)


@app.cell(hide_code=True)
def check_upsampler(box_upsampler):
    result_upsampler = L.check("upsampler", box_upsampler.value)
    result_upsampler
    return (result_upsampler,)


@app.cell(hide_code=True)
def hints_upsampler():
    L.hints(
        "upsampler",
        [
            "De derde cel van het uitgewerkte voorbeeld maakte `nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)`: 4 maps erin, 4 eruit, een kernel van 16 = 2 · 8 en padding 4 = 8 // 2. Hier nemen `classes` en `stride` de plaats in van de 4 en de 8.",
            "`up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride // 2, bias=False)`, dan `nn.init.zeros_(up.weight)`: nu heeft elk paar van een map erin en een map eruit een kernel van nullen.",
            "Dan `with torch.no_grad():` en daarbinnen `for c in range(classes):` met `up.weight[c, c] = ___`; daarna `return up`.",
        ],
        why="`up.weight` heeft shape `(classes, classes, 2 * stride, 2 * stride)`: een kernel voor elk paar van een map erin en een map eruit. `up.weight[c, c] = bilinear_kernel(2 * stride)` geeft map c erin alleen aan map c eruit, dus wordt elke klasse apart opgeschaald, en blijven de andere paren 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_iou():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. De IoU van één soort

        Maak `kind_iou(pred, classes, kind)` af, het eind van stap 6: `pred` bevat de klasse die het netwerk elke pixel geeft en `classes` de echte klasse van elke pixel, allebei arrays of tensors van dezelfde shape, van één scène of van veel. Hij geeft de IoU terug van de pixels van `kind`: de pixels die die soort zijn in allebei, gedeeld door de pixels die die soort zijn in een van beide. De eerste cel van het uitgewerkte voorbeeld deed het voor de cirkels van een netwerk dat overal achtergrond zegt. De check probeert de scène van het uitgewerkte voorbeeld met elke pixel 3 pixels naar rechts geschoven, dat netwerk, en willekeurige klassen in tensors. Als hij klopt, bewaart hij `kind_iou` in je project, als `work/projects/cnn/kind_iou.py`; de demo meet er zijn FCN mee.
        """),
        L.task("Laat `kind_iou` de IoU van de pixels van één soort geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iou():
    box_kind_iou = L.code_box("kind-iou", lines=3)
    box_kind_iou
    return (box_kind_iou,)


@app.cell(hide_code=True)
def check_iou(box_kind_iou):
    result_kind_iou = L.check("kind-iou", box_kind_iou.value)
    result_kind_iou
    return (result_kind_iou,)


@app.cell(hide_code=True)
def hints_iou():
    L.hints(
        "kind-iou",
        [
            "De eerste cel van het uitgewerkte voorbeeld mat de cirkels van een netwerk dat overal achtergrond zegt: de cirkels zijn klasse 1. Voor een andere soort verandert alleen dat getal.",
            "`p = pred == kind` is `True` voor elke pixel die het netwerk die soort noemt, en `t = classes == kind` voor elke pixel die die soort is. `&` houdt de pixels over die `True` zijn in allebei, `|` die `True` zijn in een van beide.",
            "`return (p & t).sum() / ___`",
        ],
        why="`(p | t).sum()` telt elke pixel die die soort is in het antwoord, in de class map, of in allebei, één keer: de vereniging. Gedeeld daardoor geeft de overlap 1 voor een perfect antwoord, en 0 als het netwerk nooit een pixel van die soort terecht zo noemt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint het FCN van de course, met output stride 8 en je `upsampler` als zijn transposed convolution, op de class maps die je `class_map` van de 2.000 trainingsscènes maakt, 8 epochs lang in mini-batches van 32, zoals de course het trainde. Na elke epoch meet hij het FCN op de 200 testscènes met je `kind_iou`. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt een paar minuten. Terwijl hij traint, laat de grafiek elke 10 batches de loss op de trainingsscènes zien, en gestreept de loss op de 200 testscènes na elke epoch. Ernaast staan de klassen die het FCN op dat moment aan drie testscènes geeft, en daaronder zijn grove scores op de scène van het uitgewerkte voorbeeld: de 4 maps van 16 bij 16 die je transposed convolution zo groot maakt als de scène. Allebei worden ze opnieuw getekend terwijl hij traint. Als hij klaar is, kun je door de epochs terugspoelen en zie je ze allebei na elke epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train het FCN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_fcn(train_button, result_class_map, result_upsampler, result_kind_iou):
    result_class_map, result_upsampler, result_kind_iou  # opnieuw uitvoeren als een oefening je functie bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het FCN met je functies te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _words = {"scene": "scène {seed}", "kinds": ("cirkel", "vierkant", "driehoek"), "layer": "De grove scores van het FCN op scène {seed}: layer {name}",
                  "alt": "Drie testscènes, elk met de klasse van elke pixel die het FCN haar na die epoch gaf."}
        _words_now = _words | {"alt": "Drie testscènes, elk met de klasse die het FCN elke pixel nu geeft."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "FCN: de loss tijdens het trainen", "x": "epoch", "y": "loss",
                                        "alt": "De loss van het FCN tot nu toe: op de trainingsscènes elke 10 batches, en gestreept op de 200 testscènes na elke epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "trainingsscènes, elke 10 batches"), ("loss", True, "testscènes, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-fcn-now"))
                _now["test"] = []
            if "val_loss" in _p:   # de testscènes na elke epoch, gestreept
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**FCN** · epoch {_p['epoch']} van {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:   # de trainingsscènes elke 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**FCN** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 1, 3 en 4 af: dan staan je `class_map`, je `upsampler` en je `kind_iou` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan tien minuten. Doet een van je functies meer werk dan nodig, zoals een loop over elke pixel?*",
                "classes": "*Je `class_map` in `work/projects/cnn/classes.py` stopt met een fout, of geeft niet elke pixel van een scène een klasse van 0 tot 3. Heb je het bestand veranderd? Verwijder het, en check oefening 1 opnieuw: de check bewaart je code dan opnieuw.*",
                "upsample": "*Je `upsampler` in `work/projects/cnn/upsample.py` stopt met een fout, of geeft geen transposed convolution die de 4 maps van 16 bij 16 zo groot maakt als de scène en die het trainen kan veranderen. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "kind_iou": "*Je `kind_iou` in `work/projects/cnn/kind_iou.py` stopt met een fout, of geeft geen 1 voor een class map vergeleken met zichzelf. Heb je het bestand veranderd? Verwijder het, en check oefening 4 opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van het FCN na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"Na epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: na welke epoch krijgt elke vorm van de drie scènes zijn eigen soort?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("cirkels", "vierkanten", "driehoeken"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Getraind met je functies heeft het FCN {L.number(100 * _result['acc'], 1)}% van de pixels van de 200 testscènes goed, en de IoU van de pixels van elke soort is: {_ious}. Het FCN van de course, op dezelfde manier getraind met de seeds 0 tot 4, heeft 96,8% tot 98,0% van de pixels goed, met een IoU van 0,78 tot 0,87 voor de cirkels, 0,53 tot 0,71 voor de vierkanten en 0,60 tot 0,73 voor de driehoeken."""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_stride():
    mo.md(r"""
    ### Oefening 5. Waarom niet stride 4?
    """)
    return


@app.cell(hide_code=True)
def explain_stride_cell():
    explain_stride = L.explain(
        "why-stride",
        question="Met stride 4 heeft de laatste map van het FCN van de course 32 bij 32 plaatsen, vier keer zoveel als met stride 8, dus zijn randen zouden fijner kunnen zijn. Toch is zijn IoU lager voor elke soort: voor de vierkanten 0,07 tot 0,32, tegen 0,53 tot 0,71 met stride 8. Waarom? Kijk naar het gestreepte vierkant van de slider.",
        model_answer="Elke plaats van de laatste map geeft de scores van zijn pixels uit het deel van de scène dat hij ziet. Met stride 4 is dat 18 bij 18 pixels, terwijl een vorm 10 tot 22 pixels van zijn midden tot zijn rand of zijn hoeken reikt: 92% van de vormen van de testscènes is in beide richtingen groter dan 18 pixels. Een plaats binnen een vorm ziet een vlak van één kleur, of een stuk rechte rand, dat er op een vierkant hetzelfde uitziet als op een driehoek. Hij kan de soort niet zien, dus gokt hij, en de plaatsen van één vorm gokken verschillend: in de slider krijgt het vierkant van de scène vooral de soort van een cirkel. Met stride 8 ziet een plaats 38 bij 38 pixels, ongeveer een hele vorm met zijn hoeken of zijn ronde rand. Een fijnere map geeft pas fijnere randen als elke plaats genoeg ziet om de soort te kennen. Dat is de waarschijnlijkste reden, al is het niet het enige verschil: het netwerk met stride 4 is ook kleiner, 15.492 parameters tegen 64.868, met minder layers om samen te voegen wat het ziet.",
        points=[
            "Met stride 4 ziet één plaats van de laatste map 18 bij 18 pixels; met stride 8 38 bij 38.",
            "De meeste vormen zijn groter dan 18 bij 18 pixels, dus met stride 4 ziet één plaats maar een deel van een vorm, zonder zijn hoeken of zijn hele rand, en kan hij zijn soort niet zien.",
            "Een fijnere map helpt de randen pas als elke plaats genoeg van een vorm ziet om zijn soort te kennen.",
        ],
    )
    explain_stride
    return (explain_stride,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De masks uit [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl) werden een class map, de IoU uit [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) meet nu de pixels van elke soort, en de 1×1-convolutie uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) geeft op elke plaats een score voor elke klasse. De bilineaire interpolatie uit [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl) is nu het begin van een transposed convolution die leert. Waar een detector elke vorm een box gaf, geeft het FCN elke pixel een soort: met je `class_map`, je `upsampler` en je `kind_iou` haalt het FCN van de course met stride 8 een IoU van 0,78 tot 0,87 voor de cirkels. Je project `cnn` heeft nu `classes.py`, `upsample.py` en `kind_iou.py`. In de volgende les, *Project: je eigen U-Net*, gaat een netwerk omlaag naar grove maps en weer omhoog, en stapelt het elke map op de weg omhoog met de scherpe map van dezelfde grootte van de weg omlaag, om de randen precies te tekenen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_class_map, result_transposed, result_upsampler, result_kind_iou, explain_stride):
    L.wrap_up(result_class_map, result_transposed, result_upsampler, result_kind_iou, explain_stride)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
