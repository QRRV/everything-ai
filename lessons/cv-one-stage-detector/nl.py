# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-one-stage-detector. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-one-stage-detector", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_learn():
    quick_learn = L.choice(
        "quick-learn",
        question="Een anchor heeft een IoU van 0,46 met box A en 0,31 met box B, en hij is van geen van beide de beste anchor. Wat leert hij?",
        options={
            "box": "Box A, de box waar hij het meest mee overlapt",
            "background": "Achtergrond: hij haalt met geen enkele box 0,5",
            "right": "Niets: hij wordt genegeerd",
        },
        correct="right",
        why={
            "box": "Een anchor leert de box waar hij het meest mee overlapt pas vanaf een IoU van 0,5, of als hij de beste anchor van die box is. Met 0,46 ligt hij deels op A. Kijk nog eens naar [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl).",
            "background": "Achtergrond is onder 0,4. Van 0,4 tot 0,5 ligt een anchor deels op een vorm, en wordt hij genegeerd. Kijk nog eens naar [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl).",
            "right": "Zijn beste IoU, 0,46, ligt tussen 0,4 en 0,5: hij leert A niet en achtergrond ook niet. Vandaag krijgt zo'n anchor klasse -1, en laat de loss hem weg.",
        },
    )
    quick_learn
    return (quick_learn,)


@app.cell(hide_code=True)
def quick_check_back():
    quick_back = L.choice(
        "quick-back",
        question="Een anchor van 16 bij 16 pixels, met zijn midden op rij 43,5 en kolom 59,5, krijgt de box offset (0,25; 0; log 2; 0). Welke box `(top, left, bottom, right)` wordt dat?",
        options={
            "right": "(32, 52, 63, 67): 4 pixels lager, twee keer zo hoog",
            "pixels": "(28,25; 52; 59,25; 67): een kwart pixel lager, twee keer zo hoog",
            "wide": "(40, 44, 55, 75): 4 pixels lager, twee keer zo breed",
        },
        correct="right",
        why={
            "right": "Het midden gaat 0,25 × 16 = 4 rijen omlaag, naar 47,5, en de hoogte wordt 16 × e^(log 2) = 32: rij 47,5 − 15,5 = 32 tot 63. De breedte blijft 16: kolom 52 tot 67. Je `boxes_from` doet dit voor elke anchor tegelijk, en maakt vandaag van de offsets van je detector boxes.",
            "pixels": "t_y is een deel van de hoogte van de anchor, geen aantal pixels: 0,25 × 16 = 4 rijen omlaag. Kijk nog eens naar [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl).",
            "wide": "De box offset begint met de rijen: (t_y, t_x, t_h, t_w). Het derde getal, log 2, is de verandering van de hoogte. Kijk nog eens naar [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl).",
        },
    )
    quick_back
    return (quick_back,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl) liet je DETR draaien, een detector die anderen op COCO hadden getraind. In [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) vertelde je elke anchor van een scène wat hij moet leren. Nu bouw je zelf een detector, en train je hem ook zelf: één CNN dat een scène één keer leest, en elke anchor tegelijk een soort en een box offset geeft. Een detector die zo werkt, is een **one-stage detector**. SSD (Liu en collega's, 2016) en YOLOv2 (Redmon en Farhadi, 2017) zijn twee bekende, en RetinaNet is een latere. Een two-stage detector, zoals Faster R-CNN, kiest eerst gebieden waar een voorwerp in kan zitten, en kijkt daarna naar elk van die gebieden opnieuw: dat is de volgende les. Op de scènes van de course traint je detector in een paar minuten op een CPU, en vindt hij de meeste vormen van scènes die hij nooit zag.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken", "Je CNN op kleurenfoto's", "Je eigen one-stage detector",
                            "Je eigen U-Net", "Eindproject: ResNet met Grad-CAM", "Eindproject: U-Net of detector"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_found():
    guess_found = L.guess(
        "found",
        question="Je detector heeft 38.616 parameters. Hij traint 8 epochs op 2.000 scènes, in een paar minuten op een CPU. Daarna krijgt hij 200 scènes die hij nooit zag, met 808 vormen. Hoeveel daarvan vindt hij, met een box van de goede soort, een score van 0,5 of meer, en een IoU boven 0,5 met de vorm?",
        options={
            "half": "Zo'n 400: de helft",
            "most": "Zo'n 600: drie op de vier",
            "right": "Zo'n 730: negen op de tien",
            "all": "Alle 808",
        },
        reveal="Zo'n 730. De detector van de course, getraind zoals de demo de jouwe traint, met de seeds 0 tot 4, vindt 725 tot 749 van de 808 vormen. Kleine vormen mist hij veel vaker: met seed 0 mist hij 25 van de 95 vormen waarvan de box aan zijn langste kant hoogstens 20 pixels is, tegen 32 van de 550 van 25 pixels of meer. Een kleine vorm heeft weinig anchors die hem leren: in de vorige les had de kleine driehoek er 2, de cirkel 8.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén keer over de scène.** Het eerste deel van je detector, de **backbone**, is een CNN zoals je eerdere, nu met batch normalization na elke convolutie, zoals ResNet het heeft: convoluties van 3 bij 3, elk gevolgd door batch normalization en ReLU, en drie keer poolen met 2. Zo wordt een scène van 128 bij 128 pixels 64 feature maps van 16 bij 16 vakjes, en elk vakje staat voor 8 bij 8 pixels: de vakjes van de anchors. Dan komt de head, het deel dat de antwoorden geeft, zoals de laatste layer van je CNN's hun logits gaf: één convolutie van 3 bij 3, met 24 output channels. Voor elk van de 3 anchors van een vakje geeft hij 8 getallen: 4 logits, voor achtergrond, cirkel, vierkant en driehoek, en 4 box offsets. Dezelfde filters schuiven over elk vakje, dus elk vakje beantwoordt dezelfde vraag voor zijn eigen anchors, waar een vorm ook ligt. Eén keer rekenen geeft 16 × 16 × 24 = 6.144 getallen: 8 voor elk van de 768 anchors. Ook YOLOv2 geeft de anchors van één feature map hun getallen, 5 op elk van 13 bij 13 vakjes; SSD doet het met convoluties van 3 bij 3 op meerdere feature maps van verschillende grootte.

        **In de volgorde van de anchors.** De head geeft een tensor van shape (N, 24, 16, 16): eerst de channels, dan de rijen en de kolommen. `permute(0, 2, 3, 1)` zet de channels achteraan, (N, 16, 16, 24), zodat de 24 getallen van een vakje bij elkaar liggen; `reshape(N, 768, 8)` knipt ze dan in 3 rijen van 8, vakje voor vakje, rij voor rij. Dat is de volgorde van `anchor_boxes`: rij 458 hoort bij anchor 458.

        **Wat elke anchor leert.** Zijn klasse: 0 voor achtergrond, en de soort van zijn box plus 1 voor een anchor die een box leert, dus 1 voor een cirkel, 2 voor een vierkant en 3 voor een driehoek; -1 als hij genegeerd wordt. Achtergrond is hier klasse 0, waar DETR zijn *no object* als laatste had. En voor een anchor die een box leert, de box offset uit je `offsets`.

        **Twee losses in één.** Eén loss voor de soorten: de cross-entropy van de 4 logits, over elke anchor die niet genegeerd wordt, 755 van de 768 in de scène van het uitgewerkte voorbeeld. Eén loss voor de boxes: het gemiddelde van de absolute verschillen tussen de voorspelde en de echte offsets, de mean absolute error van [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl), maar alleen over de 20 anchors die een box leren: een anchor met achtergrond heeft geen box om te leren. De detector traint op hun som. Ook SSD telt zo twee losses op, met een gewicht van 1, maar telt elke loss op over zijn anchors en deelt door het aantal anchors dat een box leert, en neemt voor de boxes smooth L1, een variant van het absolute verschil.

        **Van 768 rijen naar boxes.** Softmax maakt van de 4 logits van een anchor kansen. Zijn **score** is de hoogste kans van een soort, zonder achtergrond, zoals de score van DETR *no object* wegliet. Een anchor met een score van 0,5 of meer houdt zijn box: je `boxes_from` maakt die uit de anchor en zijn offsets. Als laatste komt de non-maximum suppression van [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl), voor elke soort apart, bij 0,5. Voor de mAP van [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl) tellen de boxes mee tot een score van 0,05.

        **Waarom NMS.** Bij het trainen leren meerdere anchors elke vorm, elk met zijn eigen offsets: in de scène van het uitgewerkte voorbeeld leren 8 anchors de cirkel. Dus geeft de getrainde detector één vorm meerdere boxes: op de 200 testscènes zo'n 20 boxes van 0,5 of meer per scène vóór NMS en 4,3 erna, voor gemiddeld 4 vormen per scène.
        """),
        L.watch_out("Achtergrond is ver in de meerderheid. In de scène van het uitgewerkte voorbeeld leren 735 van de 755 anchors in de loss van de soorten achtergrond: een detector die overal achtergrond zei, had het bij 97% ervan goed. Op deze scènes leert je detector de vormen toch. SSD houdt alleen de moeilijkste anchors met achtergrond, hoogstens 3 voor elke anchor die een vorm leert; RetinaNet verandert de loss zelf, in een latere les."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **one-stage detector** | een detector waarvan één netwerk in één keer elke anchor van het beeld een soort en een box offset geeft; daarna houdt NMS de beste boxes. SSD, YOLOv2 en RetinaNet zijn one-stage detectors | `Detector` |
        | **backbone** | het deel van een netwerk dat een afbeelding omzet in feature maps; een paar layers erbovenop, de head, maken daar de antwoorden voor de taak van. Hier een CNN van 4 convoluties | `self.backbone` |
        | **head** | de layers boven op de backbone die de antwoorden geven; hier één convolutie van 3 bij 3 die elke anchor zijn 4 logits en 4 offsets geeft | `self.head` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_nms():
    mo.vstack([
        mo.md("**De getrainde detector van de course op de scène van het uitgewerkte voorbeeld**"),
        mo.Html(C.nms_picture({
            "before": "vóór NMS: {n} boxes van 0,5 of meer",
            "after": "na NMS: {n} boxes",
            "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
            "alt": "De scène van het uitgewerkte voorbeeld twee keer, met zijn vier echte boxes, doorgetrokken. Links de 17 gestreepte boxes van de anchors met een score van 0,5 of meer: 7 op de cirkel, 5 op de grote driehoek, en 5 op het vierkant, waarvan 2 driehoek zeggen; geen op de kleine driehoek. Rechts de 4 boxes die NMS houdt, elk met zijn soort en score: cirkel 0,99, driehoek 0,94, driehoek 0,75 en vierkant 0,74.",
        })),
        mo.md(r"""
        Links hebben 17 anchors een score van 0,5 of meer: 7 rond de cirkel, 5 rond de grote driehoek en 5 rond het vierkant. NMS houdt voor elke soort apart de box met de hoogste score, en laat de boxes van dezelfde soort vallen die er meer dan 0,5 mee overlappen. Rechts blijven 4 boxes over. Drie kloppen: de cirkel, 0,99, de grote driehoek, 0,94, en het vierkant, 0,74. De vierde, een driehoek met 0,75, ligt op het vierkant, waar de cirkel een deel ervan bedekt: NMS houdt hem, omdat hij daar de beste driehoek is, en een box van een andere soort haalt hem niet weg. De kleine driehoek, 16 bij 16 pixels, krijgt geen box van 0,5 of meer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoe hard moet de box-loss trekken? Beide losses trainen dezelfde backbone. De slider vermenigvuldigt de box-loss met een gewicht voordat hij erbij komt: van 0, geen box-loss, tot 16. Elke stand is de detector van de course, getraind zoals in de demo met seed 0, maar met dat gewicht. Links staan de boxes van 0,5 of meer van die detector op de scène van het uitgewerkte voorbeeld, na NMS; rechts staat zijn mAP op de 200 testscènes, bij een IoU van 0,5, doorgetrokken, en van 0,75, gestreept: daar moet een box veel beter passen om mee te tellen.
    """)
    return


@app.cell(hide_code=True)
def explore_weight_cell():
    def _frame(w):
        w = float(w)
        _html, _stop = C.weight_frame(w, {"title": "gewicht {weight}: {n} boxes", "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
                                          "chart": "mAP op de 200 testscènes", "weight": "gewicht van de box-loss",
                                          "at50": "IoU 0,5", "at75": "IoU 0,75",
                                          "chart_alt": "De mAP van de detector van de course op de 200 testscènes bij elk gewicht van de box-loss, bij een IoU van 0,5 en van 0,75: 0,005 en 0 bij gewicht 0, het hoogst bij 0,25, 0,947 en 0,861, dan 0,916 en 0,819 bij 1, 0,790 en 0,712 bij 4, en 0,675 en 0,598 bij 16.",
                                          "alt": "De scène van het uitgewerkte voorbeeld met zijn vier echte boxes en de {n} gestreepte boxes van 0,5 of meer van de detector die getraind is met gewicht {weight} op zijn box-loss. Ernaast zijn mAP bij elk gewicht."})
        return _html, f"Gewicht {C.weight_text(w, ',')}: mAP {L.number(_stop['map'], 3)} bij een IoU van 0,5 en {L.number(_stop['map75'], 3)} bij 0,75, en {len(_stop['boxes'])} boxes op deze scène."

    explore_weight = L.explore("weight-slider", "gewicht van de box-loss", _frame, start=1.0, values=list(C.WEIGHTS), role="hyper",
                               task="Zoek het gewicht met de hoogste mAP bij een IoU van 0,75. Wat gebeurt er bij 0 met de boxes, en bij 16 met de soorten?")
    explore_weight
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Bij 0 is de box-loss weg, en trainen de offsets nooit. De detector leert nog wel welke anchors op welke soort vorm liggen, maar zijn boxes komen overal terecht: 21 boxes van 0,5 of meer op deze scène na NMS, en een mAP van 0,005. Bij 0,25 is de mAP het hoogst, 0,947 bij een IoU van 0,5 en 0,861 bij 0,75, en krijgt ook de kleine driehoek een box. Bij 4 en 16 weegt de box-loss zwaarder dan de loss van de soorten, en wordt de detector onzeker over de soorten: bij 4 noemt één box de cirkel een vierkant, met 0,60, en een andere een cirkel, met maar 0,51; bij 16 houdt één box een score van 0,5 of meer, op de grote driehoek. De mAP zakt naar 0,790 en 0,675.
        """),
        L.watch_out("Kies het gewicht niet uit deze grafiek: die meet de testscènes, en kies je het daarop, dan belooft hun mAP meer dan nieuwe scènes waarmaken. Kies het op een validation set, scènes die je van de trainingsscènes afsplitst, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). Deze les houdt een gewicht van 1, zoals SSD, al kozen de makers van SSD het op hun eigen data, voor hun eigen manier om de twee losses op te tellen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Een one-stage detector:

        1. Leg de anchors: rond het midden van elk vakje van de feature map één van elke grootte. Op de scènes van de course geeft `anchor_boxes()` er 768.
        2. Voor elke scène wat elke anchor leert: zijn klasse, 0 voor achtergrond, de soort + 1 voor een anchor die een box leert, -1 als hij genegeerd wordt; en voor een anchor die een box leert zijn box offset.
        3. Het netwerk: een backbone die van de scène feature maps maakt met één vakje per vakje van de anchors, en een head, een convolutie van 3 bij 3 met (soorten + 1 + 4) channels voor elke anchor van een vakje. `permute` en `reshape` maken daar één rij per anchor van, in de volgorde van de anchors.
        4. De loss: de cross-entropy van de logits over de anchors die niet genegeerd worden, plus het gemiddelde absolute verschil van de offsets over de anchors die een box leren.
        5. Train hem op veel scènes.
        6. Detecteer: softmax over de logits van elke anchor, met als score de hoogste kans van een soort; houd de anchors met een score van 0,5 of meer, maak hun boxes met `boxes_from`, en doe NMS voor elke soort.
        7. Meet hem op scènes die hij nooit zag: de mAP, met de boxes tot een score van 0,05.
        """),
        L.watch_out("Elke loss laat zijn eigen anchors weg. De loss van de soorten laat de genegeerde anchors weg, klasse -1: `F.cross_entropy` stopt bij een klasse van -1. De box-loss laat ook de anchors met achtergrond weg, klasse 0: die hebben geen box, en hun offsets zouden voor niets naar 0 getrokken worden."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld doet stap 1 tot 4 en het begin van stap 6 op de scène van het uitgewerkte voorbeeld van de vorige les, `scene(18)`, met een netwerk dat nog niet getraind is. Stap 2 gebruikt `match_anchors` uit `everything_ai.scenes`, die voor elke anchor geeft wat je `match` in de vorige les gaf: het nummer van zijn box, -1 voor achtergrond, -2 als hij genegeerd wordt. De offsets in stap 2 zijn de PyTorch-cel van de vorige les. In stap 3 is de backbone een `nn.Sequential`: die voert zijn layers na elkaar uit, zoals een `forward` zou doen. `nn.BatchNorm2d(8)` is de batch normalization van [Batch normalization](/course/open/nn-batchnorm?lang=nl) voor maps: elk van de 8 channels krijgt zijn eigen gemiddelde en standaardafwijking, over de batch en elk vakje. `nn.ReLU()` is `torch.relu` als layer. In stap 6 geeft `.max(dim=1)` voor elke rij twee dingen: zijn hoogste getal, en waar dat staat, zoals `argmax(dim=1)` doet; `kind` is dan de beste soort van elke anchor, 0 voor een cirkel, en die heeft NMS per soort nodig. De rest van stap 6, je `boxes_from` en de NMS van [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl), en stap 5 en 7 duren langer: de demo na de oefeningen doet ze met jouw code. Voer de cel uit zoals hij is; haal daarna de `permute(0, 2, 3, 1)` weg waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_detector():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    s = scene(18)
    truth = torch.tensor([shape.box for shape in s.shapes]).float()   # de echte boxes, één rij per vorm
    kinds = torch.tensor([shape.label for shape in s.shapes])          # 0 cirkel, 1 vierkant, 2 driehoek
    print(kinds)

    # 1. de anchors, zoals in de vorige les: 768, rij voor rij, vakje voor vakje, grootte voor grootte
    anchors = torch.tensor(anchor_boxes()).float()
    # 2. wat elke anchor leert: zijn klasse, 0 voor achtergrond, de soort + 1 voor een vorm, -1 als
    #    hij genegeerd wordt; en de box offset van elke anchor die een box leert
    label = torch.tensor(match_anchors(anchors, truth))   # zoals je match: een box, -1 of -2
    learn = label >= 0
    classes = torch.zeros(768, dtype=torch.int64)
    classes[learn] = kinds[label[learn]] + 1
    classes[label == -2] = -1
    a, b = anchors[learn], truth[label[learn]]
    middle_a, size_a = (a[:, :2] + a[:, 2:]) / 2, a[:, 2:] - a[:, :2] + 1
    middle, size = (b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2] + 1
    t = torch.zeros(768, 4)
    t[learn] = torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)
    print((classes == 0).sum(), (classes > 0).sum(), (classes == -1).sum())
    print(classes[458], t[458])

    # 3. het netwerk: drie keer poolen maakt 16 x 16 vakjes van 8 pixels, en de head
    #    geeft elk vakje 3 x 8 getallen: 4 logits en 4 offsets voor elk van zijn anchors
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    head = nn.Conv2d(64, 3 * 8, 3, padding=1)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        maps = head(backbone(x))
        out = maps.permute(0, 2, 3, 1).reshape(1, 768, 8)   # verander: maps.reshape(1, 768, 8)
        print(x.shape, maps.shape, out.shape)
        print(out[0, 458])
        print(maps[0, 16:24, 9, 8])   # de 8 getallen van de derde anchor van vakje (9, 8)

        # 4. de loss: de cross-entropy over de anchors die niet genegeerd worden, plus het
        #    gemiddelde absolute verschil van de offsets van de anchors die een box leren
        kept = classes >= 0
        kind_loss = F.cross_entropy(out[0, kept, :4], classes[kept])
        box_loss = (out[0, learn, 4:] - t[learn]).abs().mean()
        print(kept.sum(), kind_loss, box_loss)

        # 6. detecteren: softmax over de 4 logits van elke anchor; zijn score is de hoogste
        #    kans van een soort
        e = torch.exp(out[0, :, :4])
        p = e / e.sum(dim=1, keepdim=True)
        score, kind = p[:, 1:].max(dim=1)
        print(p[458], score.max(), (score >= 0.5).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De scène heeft twee driehoeken, een vierkant en een cirkel: soorten 2, 2, 1 en 0. Net als in de vorige les leren 735 anchors achtergrond en 20 een vorm, en worden er 13 genegeerd. Anchor 458, de beste van de cirkel, krijgt klasse 1 en de offsets −0,1094, twee keer, en −0,0984, twee keer. De scène heeft shape (1, 3, 128, 128), de maps van de head (1, 24, 16, 16), en de rijen (1, 768, 8). Rij 458 bevat dezelfde 8 getallen als channels 16 tot 23 van vakje (9, 8): anchor 458 is (9 × 16 + 8) × 3 + 2, de derde anchor van dat vakje.

    De loss van de soorten over de 755 anchors die niet genegeerd worden, is 1,4667, dicht bij log 4 = 1,386: de loss van een netwerk dat alle 4 klassen dezelfde kans geeft. De box-loss over de 20 anchors die een box leren, is 0,6996. Ongetraind geeft anchor 458 achtergrond 0,47 en cirkel 0,16, maar elders hebben 16 anchors al een score van 0,5 of meer, de hoogste 0,573: het trainen moet de detector daar achtergrond leren.

    Met de verandering blijft de shape (1, 768, 8), maar wordt rij 458 (0,1454; 0,2703; …): 8 getallen van channel 14, van vakje (5, 0) tot (5, 7), en niet meer de 8 getallen van anchor 458.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De grootte van de head
    """)
    return


@app.cell(hide_code=True)
def choice_head_cell():
    choice_head = L.choice(
        "head-size",
        question="Een one-stage detector voor de 20 soorten van PASCAL VOC heeft 9 anchors op elk vakje, zoals Faster R-CNN. Zijn head is een convolutie van 3 bij 3, zoals de jouwe. Hoeveel output channels heeft hij nodig?",
        options={
            "no-background": "216: 9 anchors keer 24, voor 20 soorten en 4 offsets",
            "right": "225: 9 anchors keer 25, voor 20 soorten, achtergrond en 4 offsets",
            "one": "25: 20 soorten, achtergrond en 4 offsets, voor het vakje",
            "no-offsets": "189: 9 anchors keer 21, voor 20 soorten en achtergrond",
        },
        correct="right",
        why={
            "no-background": "Elke anchor heeft ook een logit voor achtergrond nodig: bijna elke anchor leert die. 20 soorten + achtergrond + 4 offsets is 25 per anchor.",
            "right": "Elke anchor van een vakje krijgt zijn eigen 25 getallen: 21 logits, de 20 soorten en achtergrond, en 4 offsets. 9 × 25 = 225 channels. De jouwe heeft 3 × (3 + 1 + 4) = 24.",
            "one": "Dat is één anchor per vakje. Elk van de 9 anchors krijgt zijn eigen soort en zijn eigen box offset: 9 × 25.",
            "no-offsets": "Zonder offsets blijft elke box zijn anchor. Elke anchor heeft ook zijn 4 offsets nodig: 9 × 25.",
        },
    )
    choice_head
    return (choice_head,)


@app.cell(hide_code=True)
def text_exercise_targets():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat elke anchor leert

        Maak `targets(anchors, boxes, labels)` af, stap 2 van het recept: hij krijgt de anchors, een tensor van shape (A, 4), de echte boxes van een scène, een tensor van shape (B, 4), en de soort van elke box, een tensor van B gehele getallen, 0 voor een cirkel, 1 voor een vierkant en 2 voor een driehoek. Hij geeft `classes` terug, een tensor van A gehele getallen, en `t`, een tensor van shape (A, 4): stap 2 van het uitgewerkte voorbeeld, als functie. De eerste regel staat er al: die haalt `iou_matrix` en `match` uit je project, bewaard in de vorige les, en maakt van wat `match` geeft een tensor, `label`. Ook `offsets` komt uit je project. De check geeft je functie vier anchors en twee boxes, de scène van het uitgewerkte voorbeeld, en een andere scène met andere anchors. Hij vergelijkt de klassen, en de offsets van de anchors die een box leren. Klopt het, dan bewaart hij `targets` in je project, als `work/projects/cnn/targets.py`: de demo maakt er de targets van elke scène mee.
        """),
        L.task("Laat `targets` elke anchor zijn klasse en zijn box offset geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_targets():
    box_targets = L.code_box("targets", lines=12)
    box_targets
    return (box_targets,)


@app.cell(hide_code=True)
def check_targets(box_targets):
    result_targets = L.check("targets", box_targets.value)
    result_targets
    return (result_targets,)


@app.cell(hide_code=True)
def hints_targets():
    L.hints(
        "targets",
        [
            "Stap 2 van het uitgewerkte voorbeeld, met `labels` voor `kinds` en `len(anchors)` voor 768: een masker van de anchors die een box leren, klassen die allemaal 0 zijn, als gehele getallen, dan de soorten, en -1 voor de genegeerde anchors.",
            "`learn = label >= 0` wijst de anchors aan die een box leren, en `labels[label[learn]]` is de soort van de box van elk van hen: een array als index. Tel er 1 bij op, en geef -1 waar `label == ___`.",
            "De offsets: nullen van shape (A, 4), dan `t[learn] = offsets(anchors[learn], boxes[___])`. Eindig met `return classes, t`.",
        ],
        why="Voor de vier anchors: anchor 0 leert de driehoek, soort 2, klasse 3, met de offsets (0; 0,125; 0; 0): het midden van de driehoek ligt 2 pixels naar rechts, een achtste van 16. Anchor 1 haalt er 0,45 mee: genegeerd, -1. Anchor 2 leert de cirkel, klasse 1, met zijn midden 2 pixels lager: 2 / 32 = 0,0625. Anchor 3 leert achtergrond, 0. Dus `classes` is `[3, -1, 1, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén loss voor soorten en boxes

        Maak `detection_loss(out, classes, t)` af, stap 4 van het recept: `out` is wat de detector geeft voor een batch van N scènes, een tensor van shape (N, A, 8), voor elke anchor 4 logits en 4 offsets; `classes`, van shape (N, A), en `t`, van shape (N, A, 4), zijn wat `targets` voor elke scène geeft, samengevoegd. Hij geeft één getal terug: de cross-entropy van de logits `out[:, :, :4]` over de anchors met een klasse van 0 of meer, plus het gemiddelde van de absolute verschillen tussen de offsets `out[:, :, 4:]` en `t` over de anchors met een klasse boven 0. Een masker van shape (N, A), zoals `classes >= 0`, houdt die anchors van alle scènes tegelijk: `out[:, :, :4][kept]` heeft shape (K, 4). De check probeert één scène met de vier anchors van oefening 2, en twee verzonnen batches van twee scènes. Klopt het, dan bewaart hij `detection_loss` in je project, als `work/projects/cnn/detection_loss.py`.
        """),
        L.task("Laat `detection_loss` de loss van de soorten en de loss van de boxes optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_detection_loss = L.code_box("detection-loss", lines=8)
    box_detection_loss
    return (box_detection_loss,)


@app.cell(hide_code=True)
def check_loss(box_detection_loss):
    result_loss = L.check("detection-loss", box_detection_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "detection-loss",
        [
            "Stap 4 van het uitgewerkte voorbeeld, voor een batch: een masker heeft nu shape (N, A), en `out[:, :, :4][kept]` houdt de logits van die anchors van alle scènes, elk één rij.",
            "De loss van de soorten neemt de anchors van `kept = classes >= 0`: `F.cross_entropy(out[:, :, :4][kept], classes[___])`. De box-loss neemt een ander masker: `learn = classes > ___`.",
            "`(out[:, :, 4:][learn] - t[learn]).___().mean()`, en geef de som van de twee losses terug.",
        ],
        why="Voor de scène met vier anchors: de cross-entropy van anchor 0, 2 en 3, met klassen 3, 1 en 0, is ongeveer 0,994; de offsets van anchor 0 en 2 verschillen 0,1; 0,075; 0; 0 en 0,0625; 0; 0,1; 0, en dat geeft een gemiddelde van ongeveer 0,042. Samen ongeveer 1,036. Anchor 1, klasse -1, doet bij geen van beide mee.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_detector():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Je detector

        `Detector` heeft de backbone van het uitgewerkte voorbeeld al. Maak hem af, stap 3 van het recept: maak van `self.head` een convolutie van 3 bij 3 van de 64 channels van de backbone naar 3 × 8 channels, met `padding=1`; en laat `forward` voor een batch `x` van shape (N, 3, 128, 128) een tensor van shape (N, 768, 8) geven, één rij per anchor, in de volgorde van `anchor_boxes`. De check maakt netwerken na `torch.manual_seed`, geeft ze willekeurige scènes, en vergelijkt de shape, de rijen van een paar anchors, en het aantal parameters. Klopt het, dan bewaart hij `Detector` in je project, als `work/projects/cnn/detector.py`.
        """),
        L.task("Laat `Detector` elke anchor zijn 4 logits en 4 offsets geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_detector():
    box_detector = L.code_box("detector", lines=18)
    box_detector
    return (box_detector,)


@app.cell(hide_code=True)
def check_detector(box_detector):
    result_detector = L.check("detector", box_detector.value)
    result_detector
    return (result_detector,)


@app.cell(hide_code=True)
def hints_detector():
    L.hints(
        "detector",
        [
            "Stap 3 van het uitgewerkte voorbeeld, als class: de head maak je in `__init__`, na de backbone, en `forward` stuurt de scènes door allebei.",
            "`self.head = nn.Conv2d(64, ___, 3, padding=1)`: 3 anchors van elk 8 getallen. In `forward` geeft de head (N, 24, 16, 16): zet eerst de channels achteraan.",
            "`out = self.head(self.backbone(x))`, en dan `return out.permute(0, 2, 3, 1).reshape(len(x), ___, 8)`.",
        ],
        why="De head heeft 64 × 3 × 3 × 24 = 13.824 weights en 24 biases. Met de 24.768 parameters van de backbone heeft je detector er 38.616. `reshape(len(x), -1, 8)` rekent de 768 zelf uit: 16 × 16 × 24 / 8.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo maakt met je `targets` de targets van de 2.000 trainingsscènes en de 200 testscènes, traint je `Detector` met je `detection_loss` 8 epochs lang, in mini-batches van 32, zoals de detector van de course trainde, en vindt daarna de vormen van de testscènes met je `boxes_from` en NMS. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt een paar minuten. Terwijl hij traint, laat de grafiek elke 10 batches de loss op de trainingsscènes zien, en gestreept de loss op de 200 testscènes na elke epoch. Onder de grafiek staan de boxes die je detector op dat moment aan drie testscènes geeft, en daaronder de feature maps van zijn laatste layer vóór de head, op de scène van het uitgewerkte voorbeeld; allebei worden ze opnieuw getekend terwijl hij traint. Als hij klaar is, kun je door de epochs terugspoelen en zie je ze allebei na elke epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_detector(train_button, result_targets, result_loss, result_detector):
    result_targets, result_loss, result_detector  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je detector te trainen.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": ("scène {seed}: {n} box", "scène {seed}: {n} boxes"), "layer": "Layer {name} van je detector op scène {seed}",
                  "alt": "Drie testscènes, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die je detector er na die epoch aan gaf, na NMS."}
        _words_now = _words | {"alt": "Drie testscènes, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die je detector er nu aan geeft, na NMS."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je detector: de loss tijdens het trainen", "x": "epoch", "y": "loss",
                                        "alt": "De loss van je detector tot nu toe: op de trainingsscènes elke 10 batches, en gestreept op de 200 testscènes na elke epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "trainingsscènes, elke 10 batches"), ("loss", True, "testscènes, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-one-now"))
                _now["test"] = []
            if "val_loss" in _p:   # de testscènes na elke epoch, gestreept
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Je detector** · epoch {_p['epoch']} van {_p['epochs']} · loss op de testscènes {L.number(_p['val_loss'], 3)}")
            else:   # de trainingsscènes elke 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Je detector** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2, 3 en 4 af: dan staan je `targets`, je `detection_loss` en je `Detector` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Je project doet er hier te lang over: meer dan twee minuten voor de targets van de 2.000 trainingsscènes, of meer dan tien minuten om te trainen. Doet je `targets`, je `Detector` of je `detection_loss` meer werk dan nodig is, zoals een loop over de anchors?*",
                "targets": "*Je `targets` in `work/projects/cnn/targets.py` stopt met een fout, of geeft niet voor elk van de 768 anchors van een scène een klasse en 4 offsets. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "loss": "*Je `detection_loss` in `work/projects/cnn/detection_loss.py` stopt met een fout, of geeft niet één getal terug als tensor waar het trainen naar terug kan rekenen. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "detector": "*Je `Detector` in `work/projects/cnn/detector.py` stopt met een fout, of geeft voor een scène geen 768 rijen van 8. Heb je zijn bestand veranderd? Verwijder het, en check oefening 4 opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van je detector. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van je detector na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"Na epoch {_e}: loss op de testscènes {L.number(_val[_e - 1], 3)}, en {_n} {'box' if _n == 1 else 'boxes'} van 0,5 of meer op deze drie scènes, die {C.SHOWN_SHAPES} vormen hebben.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: na welke epoch zitten de boxes op de vormen, en welke vormen blijven zonder box?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("cirkels", "vierkanten", "driehoeken"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Je detector, met {C.whole(_result['params'], L.lang)} parameters, vindt {_result['found']} van de {_result['shapes']} vormen van de 200 testscènes: een box van hun soort, met een score van 0,5 of meer en een IoU boven 0,5. {_result['false']} van zijn boxes van 0,5 of meer vinden geen vorm. Zijn mAP bij een IoU van 0,5 is {L.number(_result['map'], 3)}, met een AP van {_aps}; bij 0,75 is hij {L.number(_result['map75'], 3)}. Vóór NMS geeft hij {L.number(_result['before'], 1)} boxes van 0,5 of meer per scène. De detector van de course, op dezelfde manier getraind met de seeds 0 tot 4, vindt 725 tot 749 vormen, met een mAP van 0,906 tot 0,918.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_nms():
    mo.md(r"""
    ### Oefening 5. Waarom NMS?
    """)
    return


@app.cell(hide_code=True)
def explain_nms_cell():
    explain_nms = L.explain(
        "why-nms",
        question="Je detector heeft NMS nodig; DETR, in [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl), niet. Wat in het trainen van elk maakt dat verschil? En als je `targets` alleen de beste anchor van elke echte box die box liet leren, en elke andere anchor achtergrond gaf, wat zou je dan verwachten?",
        model_answer="Bij het trainen leert elke anchor die een IoU van 0,5 met een vorm haalt die vorm, en de beste anchors van de vorm ook, elk met zijn eigen offsets naar dezelfde box: de cirkel van het uitgewerkte voorbeeld had er 8. Na het trainen geven ze de vorm allemaal een box met een hoge score, zo'n 20 boxes van 0,5 of meer per scène voor 4 vormen, en NMS houdt de box met de hoogste score en laat de boxes van dezelfde soort vallen die er meer dan 0,5 mee overlappen. DETR koppelde bij het trainen elke echte box aan precies één query, en elke andere query moest *no object* leren: hij leerde elk voorwerp één box te geven. Als alleen de beste anchor een vorm leert, verwacht ik minder boxes per vorm, maar niet zeker één: de anchors ernaast zien bijna dezelfde pixels en moeten nu achtergrond leren, en dat is moeilijker, dus sommige kunnen de vorm nog steeds een box geven, en NMS blijft nuttig.",
        points=[
            "je zegt dat bij het trainen van je detector meerdere anchors rond een vorm die vorm leren, zodat ze hem na het trainen allemaal een box geven, en dat NMS de beste ervan houdt",
            "je zegt dat DETR bij het trainen elke echte box aan één query koppelde, zodat de andere queries *no object* leerden, en dat één box per voorwerp is wat hij leerde",
            "je zegt dat je met alleen de beste anchor per vorm minder boxes verwacht, maar dat zijn buren, die bijna dezelfde pixels zien, dan achtergrond moeten leren, zodat er een paar boxes kunnen blijven en NMS nog steeds nuttig is",
        ],
    )
    explain_nms
    return (explain_nms,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De targets van [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) zijn nu wat een netwerk leert. Je `targets` geeft elke anchor een klasse en een box offset; je `Detector` geeft elke anchor in één keer 4 logits en 4 offsets, met een backbone zoals in je eerdere CNN's en een head die één convolutie van 3 bij 3 is; en je `detection_loss` telt de cross-entropy van de soorten op bij het gemiddelde absolute verschil van de boxes. Detecteren gaat met softmax en een score, zoals bij DETR in [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl), je `boxes_from`, en de NMS van [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl): veel anchors leerden elke vorm, dus krijgt elke vorm veel boxes. Getraind in 8 epochs op 2.000 scènes vindt de detector van de course 725 tot 749 van de 808 vormen van 200 nieuwe scènes, met een mAP van 0,906 tot 0,918; met seed 0 mist hij kleine vormen veel vaker dan grote. Je project `cnn` heeft nu `targets.py`, `detection_loss.py` en `detector.py`. In de volgende les, *Van R-CNN naar Faster R-CNN: twee fasen*, stelt een detector eerst gebieden voor waar een voorwerp in kan zitten, en kijkt hij daarna naar elk ervan opnieuw.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_head, result_targets, result_loss, result_detector, explain_nms):
    L.wrap_up(choice_head, result_targets, result_loss, result_detector, explain_nms)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
