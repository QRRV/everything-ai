# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-mask-rcnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-mask-rcnn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) en [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_round():
    quick_round = L.choice(
        "quick-round",
        question="RoI pooling nam een proposal van rij 41,4 tot rij 67,7 uit de vakjes 5 tot en met 8 van de feature maps, elk vakje 8 bij 8 pixels. Dezelfde proposal 3 pixels lager loopt van rij 44,4 tot 70,7. Wat geeft RoI pooling hem?",
        options={
            "right": "Precies dezelfde getallen: hij neemt nog steeds de vakjes 5 tot en met 8",
            "next": "De getallen van de vakjes 6 tot en met 9: 3 pixels is genoeg voor het volgende vakje",
            "little": "Bijna dezelfde getallen: elke bin schuift 3/8 vakje omlaag",
        },
        correct="right",
        why={
            "right": "`int(44.4 // 8)` is 5 en `int(70.7 // 8)` is 8, net als eerst: dezelfde vakjes, dezelfde bins, dezelfde 1.024 getallen. De tweede fase kan niet zien dat de proposal is verplaatst. RoIAlign van vandaag rondt niet af.",
            "next": "`44.4 // 8` is 5 en `70.7 // 8` is 8: beide hoeken blijven in hun vakje. Zie [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).",
            "little": "RoI pooling verdeelt de vakjes in bins, niet de proposal. De vakjes blijven 5 tot en met 8, dus de getallen blijven precies hetzelfde. Zie [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).",
        },
    )
    quick_round
    return (quick_round,)


@app.cell(hide_code=True)
def quick_check_merge():
    quick_merge = L.choice(
        "quick-merge",
        question="In een scène raken twee cirkels elkaar. Wat geeft hun class map ze, zoals je FCN die leerde maken?",
        options={
            "right": "Eén gebied van de klasse cirkel: niets erin zegt waar de ene cirkel ophoudt",
            "two": "Twee gebieden van de klasse cirkel, met een lijntje achtergrond ertussen",
            "new": "De klasse cirkel voor de ene en een andere klasse voor de andere",
        },
        correct="right",
        why={
            "right": "Een class map geeft elke pixel een soort, en beide cirkels zijn van dezelfde soort: hun pixels zijn één gebied van de klasse cirkel. Hoeveel cirkels het zijn, en welke pixel bij welke hoort, staat er niet in.",
            "two": "Elke pixel van een cirkel krijgt de klasse cirkel, ook waar de cirkels elkaar raken: er zit geen pixel achtergrond tussen. Zie [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).",
            "new": "Een class map heeft één klasse per soort: achtergrond, cirkel, vierkant en driehoek. Beide cirkels krijgen de klasse cirkel. Zie [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).",
        },
    )
    quick_merge
    return (quick_merge,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je FCN geeft elke pixel een soort. Raken twee cirkels elkaar, dan maakt zijn class map er één gebied van, en niets zegt waar de ene cirkel ophoudt en de andere begint. Dat is niet zeldzaam: van de 808 vormen van de 200 testscènes van de course raken of overlappen er 186 een vorm van dezelfde soort, in 82 van de scènes. Je detectors geven elk voorwerp een eigen box, maar een box is niet het voorwerp: de box van een cirkel is voor een mediaan van 78% cirkel; die van een vierkant, dat gedraaid kan zijn, voor 59%; die van een driehoek voor 50%. **Instance segmentation** geeft elk voorwerp een eigen mask: de pixels die dat voorwerp zijn. Mask R-CNN (He, Gkioxari, Dollár en Girshick, 2017) doet dat met een kleine toevoeging aan Faster R-CNN: voor elke box die hij vindt, tekent een klein FCN het mask van het voorwerp erin. De makers schrijven dat Mask R-CNN maar weinig rekenwerk toevoegt aan Faster R-CNN, en 5 beelden per seconde haalt op een GPU. Op COCO hadden de masks van zijn beste model een mask AP van 37,1, tegen 33,6 voor FCIS+++, de winnaar van de segmentatie-uitdaging van COCO in 2016. Vandaag bouw je hem op de scènes van de course, bovenop je two-stage detector van [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_align():
    guess_align = L.guess(
        "align",
        question="Faster R-CNN snijdt elke proposal uit de feature maps met RoI pooling, die afrondt op hele vakjes. De makers van Mask R-CNN vervingen die door RoIAlign, die niet afrondt, en veranderden verder niets. Met een ResNet-50 als backbone, met vakjes van 16 bij 16 pixels, hoeveel beter werden de masks op COCO, in mask AP?",
        options={
            "hardly": "Nauwelijks: ongeveer 0,3 punt",
            "right": "Ongeveer 3 punten: van 26,9 naar 30,3",
            "lots": "Ongeveer 15 punten: van 26,9 naar 42",
            "worse": "Ze werden slechter: afronden helpt tegen ruis",
        },
        reveal="Ongeveer 3 punten: van 26,9 naar 30,3 mask AP, het gemiddelde over de tien IoU-drempels van COCO. Bij een IoU van 0,5 van 48,8 naar 51,2; bij 0,75 van 26,4 naar 31,5: hoe strenger de drempel, hoe groter de winst. Met vakjes van 32 bij 32 pixels kost afronden meer: daar won RoIAlign 7,3 punten mask AP, en 10,5 bij 0,75. De makers vatten het samen als relatief 10% tot 50% betere masks.",
    )
    guess_align
    return (guess_align,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een mask voor elk voorwerp.** Een class map geeft elke pixel een soort. **Instance segmentation** geeft elk voorwerp, hier elke vorm, een eigen mask: de pixels die dat voorwerp zijn, ook als het een ander van zijn soort raakt. **Mask R-CNN** neemt de two-stage detector van [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) en geeft hem een derde head, naast de tweede fase: de **mask head**. De tweede fase geeft elke proposal een soort en een betere box; de mask head tekent er een mask in.

        **De mask head is een klein FCN.** RoIAlign, hieronder, snijdt een box tot 8 bij 8 bins: 64 × 8 × 8 getallen. Twee convoluties van 3 bij 3 met 32 channels en ReLU, een transposed convolution van 2 bij 2 met stride 2 naar 16 bij 16, ReLU, en een convolutie van 1 bij 1 naar 3 maps van 16 bij 16 logits: een mask voor elke soort. Zijn 31.939 parameters maken van de two-stage detector van de course, met 167.386 parameters, een Mask R-CNN van 199.325. In het paper is de head van de beste modellen net zo gebouwd: vier convoluties van 3 bij 3, een transposed convolution van 2 bij 2 met stride 2 en een convolutie van 1 bij 1, die masks van 28 bij 28 tekenen, een voor elk van de 80 soorten van COCO. De ResNet-50 van zijn vergelijkingen, zoals die in de gok, tekent masks van 14 bij 14.

        **Alleen het mask van zijn soort leert.** De tweede fase beslist de soort van een box; de mask head tekent een mask voor elke soort, en alleen het mask van de soort van de box telt. De sigmoid van elke logit is de kans dat zijn pixel bij het voorwerp hoort, en de loss is de gemiddelde log loss van de 256 pixels, zoals in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl). De masks van de soorten concurreren niet om een pixel: het paper noemt dat het mask loskoppelen van de klasse. Met softmax over de soorten op elke pixel, zoals in een class map, werden zijn masks slechter: een mask AP van 24,8 in plaats van 30,3.

        **Wat een mask leert.** Bij het trainen leert elke proposal die een soort leert, met een IoU van 0,5 of meer met een echte box, ook een mask: het echte mask van de vorm van die box, uitgesneden tot de proposal. Voor elk van de 16 bij 16 bins van de proposal is het target 1 als de pixel waar het midden van de bin in ligt bij de vorm hoort, en anders 0. Een proposal die iets te groot is, leert zo een cirkel die zijn randen niet raakt. Een mask van 16 bij 16 bevat het grootste deel van een vorm: de echte masks van de testscènes, zo uitgesneden tot 16 bij 16 bins van hun eigen box en weer zo groot gemaakt als de box, hebben daarna een mediane IoU van 0,941 met het origineel; met 8 bij 8 0,882; met 14 bij 14, de grootte in de vergelijkingen van het paper, 0,930; met 28 bij 28, die van zijn beste modellen, 0,993. Het netwerk traint op de loss van het RPN plus de loss van de tweede fase plus de loss van de mask head.

        **RoIAlign: niet afronden.** Vakje s van de feature maps beslaat de pixels 8s tot en met 8s + 7, dus zijn midden ligt op pixel 8s + 3,5, en pixel y ligt op (y − 3,5) / 8 vakjes. **RoIAlign** verdeelt de box zelf in k bij k bins van dezelfde grootte, niet de vakjes waar hij in ligt. Hij neemt het midden van elke bin, in pixels en dan in vakjes, zonder af te ronden, en de waarde van de feature maps daar, gemengd uit de vier vakjes eromheen met bilineaire interpolatie, zoals je `value_at` van [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl) vier pixels mengde. Een box van rij 58 tot en met rij 86 beslaat 29 pixels, van 57,5 tot 86,5; het midden van de eerste van zijn 8 bins ligt op 57,5 + 29 / 16 = 59,3125, vakje 6,9766. Training volgt de loss terug door de mix naar de vier vakjes. Het paper neemt vier punten in elke bin en hun gemiddelde of hun maximum, en vond dat het resultaat nauwelijks afhangt van waar precies en hoeveel punten het neemt; hier neemt de course één punt per bin. Ook de tweede fase krijgt RoIAlign, met 4 bij 4 bins.

        **Een mask plakken.** Bij het detecteren tekent de mask head een mask voor elke box die na NMS overblijft, en het mask van de soort van de box is zijn mask. Zijn 16 bij 16 kansen worden met bilineaire interpolatie zo groot gemaakt als de box, en elke pixel van 0,5 of meer hoort bij het voorwerp: het paper maakt zijn mask zo groot als de box en kapt het af bij een drempel van 0,5. Dat is `paste`. Het paper tekent masks voor de 100 boxes met de hoogste score.

        **Mask AP.** Zoals de mAP van [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl), met de IoU van masks in plaats van boxes: een mask vindt een vorm van zijn soort als hun IoU boven de drempel ligt. De mask AP van COCO, het getal in het paper, is het gemiddelde bij zijn tien drempels, 0,5, 0,55 en zo verder tot 0,95.
        """),
        L.watch_out("De mask head beslist nooit de soort van een box. Een cirkel die de tweede fase een vierkant noemt, krijgt het mask dat de head voor vierkanten tekent."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **instance segmentation** | een mask voor elk voorwerp: de pixels die dat voorwerp zijn, ook als het een ander voorwerp van zijn soort raakt | |
        | **Mask R-CNN** | Faster R-CNN met een mask head, van He, Gkioxari, Dollár en Girshick (2017) | `MaskRCNN` |
        | **mask head** | een klein FCN dat in elke box een mask van logits tekent voor elke soort; alleen het mask van de soort van de box telt | `self.mask_head` |
        | **RoIAlign** | de features van een box uitsnijden tot k bij k bins zonder af te ronden: de waarde van de feature maps in het midden van elke bin, met bilineaire interpolatie | `roi_align` |
        | **mask AP** | de AP van masks: zoals mAP, met de IoU van masks in plaats van boxes | |
        | `maps[:, rows][:, :, cols]` | met lijsten van plaatsen: de rijen `rows` van elk channel, en daarvan de kolommen `cols`; elk paar van een rij en een kolom | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_instances():
    mo.vstack([
        mo.md("**Een class map en een mask voor elk voorwerp**"),
        mo.Html(C.instance_picture({
            "scene": "testscène 104", "classes": "zijn class map", "objects": "een mask per voorwerp",
            "kinds": ("cirkel", "vierkant", "driehoek"),
            "alt": "Testscène 104: twee cirkels die over elkaar liggen, en een klein vierkant. Zijn class map kleurt beide cirkels als één gebied van de klasse cirkel. Met een mask per voorwerp heeft elke cirkel een eigen mask met een eigen rand.",
        })),
        mo.md(r"""
        In de class map zijn de twee cirkels één gebied: alleen daaruit zie je niet of het één vorm is of twee. Met een mask per voorwerp is elke cirkel een eigen voorwerp, en waar ze elkaar raken, hoort elke pixel bij een van beide.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stages():
    mo.vstack([
        mo.md("**De getrainde Mask R-CNN van de course op de scène van het uitgewerkte voorbeeld**"),
        mo.Html(C.stages_picture({
            "boxes": "{n} boxes", "masks": "{n} masks",
            "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
            "alt": "De scène van het uitgewerkte voorbeeld twee keer. Links de vier echte boxes, doorgetrokken, en de 4 gestreepte boxes van 0,5 of meer van de Mask R-CNN van de course, elk met zijn soort en score. Rechts de scène in grijs met de 4 masks die hij in die boxes tekent, gevuld, en de randen van de echte masks.",
        })),
        mo.md(r"""
        Links de boxes van de two-stage detector, nu met RoIAlign; rechts het mask dat de mask head in elk ervan tekent. Het mask van de grote driehoek heeft een IoU van 0,92 met zijn echte mask, dat van de cirkel 0,87 en dat van de kleine driehoek 0,83. Het vierkant ligt deels onder de cirkel, dus zijn echte mask is een vierkant met een ronde hap eruit; zijn mask heeft een IoU van 0,70.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_roi():
    mo.vstack([
        mo.md("**RoI pooling en RoIAlign van de box van de cirkel**"),
        mo.Html(C.roi_picture({
            "pool": "RoI pooling: hele vakjes", "align": "RoIAlign: 64 punten",
            "alt": "De cirkel van de scène van het uitgewerkte voorbeeld twee keer, met de vakjes van 8 bij 8 pixels van de feature maps als dunne lijnen en de box van de cirkel gestreept. Links de 4 bij 4 vakjes waar zijn hoeken in liggen, omkaderd: RoI pooling tot 8 bij 8 bins geeft elk vakje 2 bij 2 bins. Rechts de box verdeeld in 8 bij 8 bins, met een punt in het midden van elke bin.",
        })),
        mo.md(r"""
        RoI pooling van de box van de cirkel tot 8 bij 8 bins neemt de vakjes waar zijn hoeken in liggen: 7 tot en met 10 omlaag en 6 tot en met 9 opzij, 4 bij 4 vakjes, groter dan de box. Elke bin is een heel vakje, en elk vakje vult 2 bij 2 bins: van de 64 getallen per channel verschillen er maar 16. RoIAlign verdeelt de box zelf, en leest de feature maps op 64 punten, elk tussen vier vakjes in.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoe goed past een mask? De slider zet de IoU waar een mask boven moet liggen om zijn vorm te vinden, van 0,5 tot 0,95, zoals de tien drempels van COCO. De course trainde zijn Mask R-CNN twee keer met seed 0: een keer met RoI pooling, een keer met RoIAlign. Links de masks van allebei op de scène van het uitgewerkte voorbeeld: gevuld als ze passen, gestreept als ze niet passen. Rechts de mask AP van allebei op de 200 testscènes bij elke drempel.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        t = float(t)
        _html, _stop = C.threshold_frame(t, {"pool": "RoI pooling: {n} boven {t}", "align": "RoIAlign: {n} boven {t}", "point": ",",
                                             "chart": "mask AP op de 200 testscènes", "x": "IoU-drempel", "y": "mask AP",
                                             "align_line": "doorgetrokken: RoIAlign", "pool_line": "gestreept: RoI pooling",
                                             "chart_alt": "De mask AP van de Mask R-CNN van de course op de 200 testscènes bij elke IoU-drempel van 0,5 tot 0,95: met RoIAlign boven 0,8 tot een drempel van 0,8 en daarna dalend; met RoI pooling bij elke drempel lager, en veel lager van 0,7 tot 0,9.",
                                             "alt": "De scène van het uitgewerkte voorbeeld twee keer, met de masks van de Mask R-CNN van de course, getraind met RoI pooling en met RoIAlign: gevuld als hun IoU met hun vorm boven {t} ligt, gestreept als die er niet boven ligt. Ernaast de mask AP van allebei bij elke drempel."})
        return _html, f"Bij een IoU van {L.number(t, 2)}: mask AP {L.number(_stop['pool']['ap'], 3)} met RoI pooling, {L.number(_stop['align']['ap'], 3)} met RoIAlign; in deze scène passen {_stop['pool']['fit']} en {_stop['align']['fit']} masks."

    explore_threshold = L.explore("threshold-slider", "IoU-drempel", _frame, start=0.75, values=list(C.THRESHOLDS), role="hyper",
                                  task="Tot welke drempel vinden ze allebei bijna elke vorm? Vanaf welke drempel blijft RoI pooling achter, en hoe ver?")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 0,5 vinden ze allebei bijna elke vorm: een mask AP van 0,924 met RoI pooling en 0,941 met RoIAlign. Daarna blijft RoI pooling achter: bij 0,75: 0,535 tegen 0,851; bij 0,85: 0,238 tegen 0,740; bij 0,9: 0,083 tegen 0,521. Bij 0,95 past nauwelijks nog een mask: 0,001 en 0,118. Het gemiddelde van de tien, zoals COCO zijn mask AP neemt, is 0,537 met RoI pooling en 0,758 met RoIAlign. De masks van RoI pooling liggen op de goede plek, maar hun randen liggen ernaast: de mediane IoU van de masks van de vormen die het vindt, is 0,88 voor de cirkels, 0,80 voor de vierkanten en 0,73 voor de driehoeken, tegen 0,95, 0,91 en 0,90 met RoIAlign; in de scène van het uitgewerkte voorbeeld ligt er geen boven 0,85. RoIAlign helpt ook de boxes: hun mAP bij een IoU van 0,75 is 0,879 met RoIAlign en 0,731 met RoI pooling. En het model met RoI pooling geeft er een valse box van de soort vierkant bij, over het vierkant en de cirkel heen, waarvan het mask vooral op de cirkel ligt.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Mask R-CNN, op je two-stage detector:

        1. De backbone, het RPN en de proposals, zoals eerst. RoIAlign snijdt elke proposal tot 4 bij 4 bins voor de tweede fase.
        2. Voor elke proposal die een soort leert: RoIAlign tot 8 bij 8 bins, en de mask head: twee convoluties van 3 bij 3, een transposed convolution naar 16 bij 16, een convolutie van 1 bij 1 naar een mask van logits voor elke soort.
        3. Het target van de proposal: het echte mask van zijn vorm in het midden van elk van zijn 16 bij 16 bins.
        4. De loss van het mask van zijn soort: de sigmoid van elke logit en de log loss, het gemiddelde over de 256 pixels. Train op de som van de losses van het RPN, de tweede fase en de mask head.
        5. Detecteer: de boxes van de tweede fase na NMS; voor elke box RoIAlign, de mask head, het mask van zijn soort, en `paste` het in de scène.
        """),
        L.watch_out("Bij het trainen tekent de mask head in de proposals; bij het detecteren in de boxes van de tweede fase, die beter passen. Zijn mask wordt altijd geplakt in de box waarin het getekend is."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld doet stap 2 tot en met 4 voor de box van de cirkel, `s.shapes[3]`, op de scène van het uitgewerkte voorbeeld van de vorige lessen, `scene(18)`, met een netwerk dat nog niet getraind heeft: de backbone van je two-stage detector en de mask head. RoIAlign met de hand: voor elk van 8 bij 8 bins het midden in pixels en dan in vakjes, met `int` het vakje erboven en links ervan, met `t` en `u` hoe ver het in dat vakje ligt, en de mix van de vier vakjes eromheen, zoals in je `value_at`. Het target: `int(top + (i + 0.5) * h / 16)` is de pixel waar het midden van bin i in ligt, en `mask[rows][:, cols]` neemt die rijen en kolommen van het echte mask. De loss: de sigmoid en de log loss, uitgeschreven. Voer de cel uit zoals hij is.
    """)
    return


@app.cell
def example_mask_head():
    import torch
    from torch import nn
    from everything_ai.scenes import scene

    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255

    # de backbone van je two-stage detector, en de mask head: twee convoluties van 3 bij 3,
    # een transposed convolution naar twee keer zo groot, en een convolutie van 1 bij 1 naar een mask per soort
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    mask_head = nn.Sequential(
        nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(),
        nn.Conv2d(32, 3, 1),
    )
    with torch.no_grad():
        maps = backbone(x)[0]   # de 64 feature maps van de scène
        top, left, bottom, right = s.shapes[3].box   # de cirkel
        h, w = bottom - top + 1, right - left + 1
        print(maps.shape, (top, left, bottom, right), h, w)

        # 2. RoIAlign tot 8 bij 8 bins: het midden van elke bin, in vakjes, en de mix van de
        #    vier vakjes eromheen
        aligned = torch.zeros(64, 8, 8)
        for i in range(8):
            for j in range(8):
                row = (top - 0.5 + (i + 0.5) * h / 8 - 3.5) / 8
                col = (left - 0.5 + (j + 0.5) * w / 8 - 3.5) / 8
                r, c = int(row), int(col)
                t, u = row - r, col - c
                above = (1 - u) * maps[:, r, c] + u * maps[:, r, c + 1]
                below = (1 - u) * maps[:, r + 1, c] + u * maps[:, r + 1, c + 1]
                aligned[:, i, j] = (1 - t) * above + t * below
        print(aligned.shape, row, col)
        print(aligned[0, 0])

        # 3. het target: het echte mask op de pixel van het midden van elk van 16 bij 16 bins
        rows = [int(top + (i + 0.5) * h / 16) for i in range(16)]
        cols = [int(left + (j + 0.5) * w / 16) for j in range(16)]
        target = torch.tensor(s.shapes[3].mask[rows][:, cols]).float()
        print(target)
        print(target.sum())

        # 4. de mask head: een mask van 16 bij 16 logits voor elke soort; de cirkel is soort 0,
        #    dus alleen het eerste mask leert, met de sigmoid en de log loss
        logits = mask_head(aligned.unsqueeze(0))[0]
        p = 1 / (1 + torch.exp(-logits[0]))
        loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()
        print(logits.shape, p.min(), p.max(), loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De feature maps hebben shape (64, 16, 16), en de box van de cirkel, (58, 50, 86, 78), is 29 bij 29 pixels. Na de lussen houden `row` en `col` het midden van de laatste bin: 10,1484 en 9,1484 vakjes; het eerste lag op 6,9766 en 5,9766. RoI pooling zou de vakjes 7 tot en met 10 omlaag en 6 tot en met 9 opzij nemen; RoIAlign leest tussen de vakjes 6 en 11 omlaag en 5 en 10 opzij. De eerste rij van channel 0, (0,6619; 0,8114; 0,9698; 1,2657; 1,6095; 1,4621; 0,8869; 0,4258), heeft in elke bin een ander getal.

    Het target is een schijf van enen die de vier randen raakt, met nullen in de hoeken: 205 van zijn 256 pixels zijn cirkel. De mask head geeft 3 masks van 16 bij 16 logits. Ongetraind liggen de kansen van het mask van de cirkel allemaal dicht bij 0,5, van 0,4536 tot 0,4695, en de loss is 0,7415: ongeveer de loss van een gok van 0,5 overal, log 2 = 0,693.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bilinear():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Bilineaire interpolatie van een heel raster

        De lussen van het uitgewerkte voorbeeld mengen vier vakjes voor één punt per keer; `bilinear` doet het voor alle punten van een raster tegelijk. Maak `bilinear(maps, ys, xs)` af: `maps` is een tensor met shape (C, H, W), zoals feature maps; `ys` en `xs` zijn lijsten van plaatsen, floats, geteld in vakjes vanaf het midden van vakje 0, zoals je `value_at` pixels telde. Hij geeft een tensor met shape (C, len(ys), len(xs)) terug: de waarde van elk channel op elk punt (y, x) van het raster, gemengd uit de vier vakjes eromheen, waarbij het dichtere vakje zwaarder telt. Een punt buiten de maps krijgt de waarde aan de dichtstbijzijnde rand: de gegeven regels zetten elke plaats eerst binnen de maps. `maps[:, r0]` neemt de rijen `r0` van elk channel, en `[:, :, c0]` daarna de kolommen `c0` van die rijen: het vakje boven en links van elk punt tegelijk. Maak van `t` een kolom, met shape (len(ys), 1), met `.reshape(-1, 1)`, en van `u` een rij: dan mengt broadcasting elke rij met `u` en elke kolom met `t`. De check probeert de 2 bij 2 pixels van je `value_at`, punten erbuiten, maps met 2 channels en met 64, en of training je uitkomst kan terugvolgen naar de maps. Klopt hij, dan bewaart hij `bilinear` in je project, als `work/projects/cnn/bilinear.py`; je `roi_align` en je `paste` gebruiken hem.
        """),
        L.task("Laat `bilinear` de waarde van de maps op elk punt van het raster geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bilinear():
    box_bilinear = L.code_box("bilinear", lines=16)
    box_bilinear
    return (box_bilinear,)


@app.cell(hide_code=True)
def check_bilinear(box_bilinear):
    result_bilinear = L.check("bilinear", box_bilinear.value)
    result_bilinear
    return (result_bilinear,)


@app.cell(hide_code=True)
def hints_bilinear():
    L.hints(
        "bilinear",
        [
            "Je `value_at` van [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl), voor elk punt tegelijk: voor elke y de rij erboven en de rij eronder, voor elke x de kolom links en rechts ervan, hoe ver het punt van de eerste van elk ligt, en de mix van de vier vakjes.",
            "`r0 = [int(y) for y in ys]` en `r1 = [min(r + 1, h - 1) for r in r0]`, en net zo voor de kolommen. `t = (torch.tensor(ys) - torch.tensor(r0)).reshape(-1, 1)`, en `u` net zo voor `xs` en `c0`, zonder `reshape`.",
            "`top = (1 - u) * maps[:, r0][:, :, c0] + u * maps[:, r0][:, :, ___]`, `bottom` net zo met `r1`, en `return (1 - t) * top + t * ___`.",
        ],
        why="Voor het punt (0,25; 0,5) van de maps [[0, 100], [200, 40]]: halverwege rij 0 is 50, halverwege rij 1 is 120, en een kwart van de weg van 50 naar 120 is 50 + 0,25 × 70 = 67,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_align():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. RoIAlign

        Maak `roi_align(maps, box, k)` af, stap 2 van het recept voor elke box en elke k: `maps` bevat de feature maps van één scène, met shape (C, H, W), elk vakje 8 bij 8 pixels; `box` is (top, left, bottom, right) in pixels, beide uiteinden meegeteld, floats, en mag uit de scène steken. Hij geeft een tensor met shape (C, k, k) terug: de waarde van de feature maps in het midden van elk van k bij k bins van de box, met je `bilinear`, die de eerste regel uit je project importeert. De box beslaat `bottom - top + 1` pixels omlaag, vanaf `top - 0.5`. De check probeert vijf boxes op verzonnen maps, waarvan twee deels of helemaal buiten de maps en een binnen één vakje, en de cirkel van de scène van het uitgewerkte voorbeeld op maps met de gemiddelde kleur van elk vakje. Klopt hij, dan bewaart hij `roi_align` in je project, als `work/projects/cnn/roi_align.py`.
        """),
        L.task("Laat `roi_align` elke box k bij k getallen per channel geven, zonder af te ronden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_align():
    box_roi_align = L.code_box("roi-align", lines=10)
    box_roi_align
    return (box_roi_align,)


@app.cell(hide_code=True)
def check_align(box_roi_align):
    result_align = L.check("roi-align", box_roi_align.value)
    result_align
    return (result_align,)


@app.cell(hide_code=True)
def hints_align():
    L.hints(
        "roi-align",
        [
            "De lussen van het uitgewerkte voorbeeld, als lijsten: de middens van de k bins omlaag en opzij in de box, in pixels, dan in vakjes, en je `bilinear` op allemaal tegelijk.",
            "`h, w = bottom - top + 1, right - left + 1`, dan `ys = [top - 0.5 + (i + 0.5) * h / k for i in range(k)]`, de middens in pixels, en net zo voor `xs`.",
            "`return bilinear(maps, [(y - ___) / 8 for y in ys], [(x - ___) / 8 for x in xs])`.",
        ],
        why="Voor de eerste box, (5,5; 9,0; 30,2; 44,0), met 2 bij 2 bins: hij is 25,7 pixels hoog, vanaf 5,0, dus de middens van zijn bins liggen op de rijen 11,425 en 24,275, of 0,9906 en 2,5969 vakjes. RoI pooling nam de hele vakjes 0 en 1 voor de eerste bin en 2 en 3 voor de tweede.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_paste():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een mask in de scène plakken

        Maak `paste(mask, box, size)` af, het einde van stap 5: `mask` bevat de kansen die de mask head één box geeft, met shape (m, m); `box` is (top, left, bottom, right) in pixels, beide uiteinden meegeteld, en mag uit de scène steken of tussen twee pixels beginnen; de scène is `size` bij `size` pixels. Hij geeft `out` terug, een tensor met shape (size, size), True waar het mask, met je `bilinear` zo groot gemaakt als de box, 0,5 of meer is. Neem elke pixel waarvan het midden in de box ligt, van `top - 0.5` tot `bottom + 0.5`, en in de scène; zijn midden ligt `r - top + 0.5` pixels in de box, die `bottom - top + 1` pixels hoog is, en dus op dat deel van de m rijen van het mask, geteld vanaf de bovenrand van het mask; `bilinear` telt vanaf het midden van zijn eerste rij. De check probeert een mask van 4 bij 4 in boxes van een scène van 8 bij 8, waarvan een die uitsteekt en een die niet bij een hele pixel begint, en twee masks van 16 bij 16 in een scène van 128 bij 128. Klopt hij, dan bewaart hij `paste` in je project, als `work/projects/cnn/paste.py`.
        """),
        L.task("Laat `paste` het mask van een box in de scène zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paste():
    box_paste = L.code_box("paste", lines=17)
    box_paste
    return (box_paste,)


@app.cell(hide_code=True)
def check_paste(box_paste):
    result_paste = L.check("paste", box_paste.value)
    result_paste
    return (result_paste,)


@app.cell(hide_code=True)
def hints_paste():
    L.hints(
        "paste",
        [
            "RoIAlign andersom: voor elke pixel van de scène waarvan het midden in de box ligt, waar dat midden in het mask ligt, en de waarde van het mask daar met je `bilinear`; 0,5 of meer hoort bij het voorwerp.",
            "De eerste rij is `max(math.ceil(top - 0.5), 0)` en de laatste `min(math.floor(bottom + 0.5), size - 1)`, en net zo voor de kolommen. Rij r ligt `(r - top + 0.5) / (bottom - top + 1) * m` rijen in het mask, en `- 0.5` telt dat vanaf het midden van zijn eerste rij.",
            "`ys = [(r - top + 0.5) / (bottom - top + 1) * m - 0.5 for r in range(r0, r1 + 1)]`, net zo voor `xs`, dan `out[r0:r1 + 1, c0:c1 + 1] = bilinear(mask.unsqueeze(0), ys, xs)[0] >= ___`.",
        ],
        why="In het eerste geval beslaat de box de rijen 1 tot en met 6, 6 pixels voor de 4 rijen van het mask. Het midden van pixelrij 1 ligt 0,5 pixel in de box, op 0,5 / 6 × 4 − 0,5 = −0,17 rijen van het mask: boven het midden van zijn eerste rij, dus het krijgt de waarde van rij 0. De box is 4 pixels breed, dus elke kolom pixels valt op het midden van een kolom van het mask.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learns():
    mo.md(r"""
    ### Oefening 4. Wat de mask head van een proposal leert
    """)
    return


@app.cell(hide_code=True)
def choice_learns_cell():
    choice_learns = L.choice(
        "learns",
        question="Bij het trainen overlapt een proposal de cirkel met een IoU van 0,62, en geen andere vorm met 0,5 of meer. De mask head tekent er 3 masks voor, een voor elke soort. Wat leert hij van deze proposal?",
        options={
            "right": "Alleen het mask van de cirkel leert: het echte mask van de cirkel, op de 16 bij 16 bins van de proposal",
            "all": "Alle drie: het mask van de cirkel de cirkel, en de masks van het vierkant en de driehoek overal 0",
            "box": "Alleen het mask van de cirkel leert: het echte mask van de cirkel, op 16 bij 16 bins van de eigen box van de cirkel",
            "none": "Niets: alleen de echte boxes leren de mask head iets",
        },
        correct="right",
        why={
            "right": "Zijn soort is cirkel, dus alleen het mask van de cirkel telt mee in de loss; de andere twee masks leren er niets van. Zijn target is het mask van de cirkel in de middens van de bins van de proposal zelf, de box waar de mask head naar kijkt: een proposal die iets te groot is, leert een cirkel die zijn randen niet raakt.",
            "all": "Dan concurreren de masks van de soorten om elke pixel, zoals bij softmax over de soorten in een class map. Mask R-CNN laat alleen het mask van de soort van de box leren; met softmax werden de masks van het paper slechter, een mask AP van 24,8 in plaats van 30,3.",
            "box": "De mask head ziet de features van de proposal, dus zijn target is uitgesneden tot de proposal: de pixel in het midden van elk van zijn 16 bij 16 bins. In de eigen box van de cirkel zou het mask liggen waar de head niet keek.",
            "none": "Elke proposal met een IoU van 0,5 of meer met een echte box leert een mask, zoals hij een soort leert; de echte boxes komen er alleen bij, net als voor de tweede fase.",
        },
    )
    choice_learns
    return (choice_learns,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint de Mask R-CNN van de course, de two-stage detector met een mask head, met je `roi_align`, en daardoor je `bilinear`, en met je `proposals`, 8 epochs lang op de 2.000 trainingsscènes, in mini-batches van 32. Daarna vindt hij de vormen van de 200 testscènes met je `boxes_from`, en plakt hij hun masks met je `paste`. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt even: op de computer van de course ongeveer elf minuten. De grafiek toont na elke epoch de training loss van de drie delen samen. Als hij klaar is, kun je terug door de epochs, en de masks zien die hij na elk ervan aan drie testscènes gaf.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train Mask R-CNN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_mask_rcnn(train_button, result_bilinear, result_align, result_paste):
    result_bilinear, result_align, result_paste  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om Mask R-CNN te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Mask R-CNN: de training loss", "x": "epoch", "y": "training loss",
                                        "alt": "De training loss van Mask R-CNN, zijn drie delen samen, na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "mask" in _p:
                _loss = _p["rpn"] + _p["second"] + _p["mask"]
                _now["chart"].add(_p["epoch"], _loss, f"**Mask R-CNN** · epoch {_p['epoch']} van {_p['epochs']} · training loss {L.number(_loss, 3)}")
            else:
                _now["chart"].line(f"**Mask R-CNN** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `roi_align` en je `paste` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Trainen duurt hier te lang: meer dan vijfentwintig minuten. Doet je `bilinear`, je `roi_align` of je `proposals` meer werk dan nodig, zoals een lus over de punten van het raster? `bilinear` kan alle punten van het raster tegelijk mengen.*",
                "roi_align": "*Je `roi_align` in `work/projects/cnn/roi_align.py`, of de `bilinear` die hij gebruikt, stopt met een fout, of geeft geen 64 × k × k getallen die training kan terugvolgen. Heb je een van hun bestanden veranderd? Verwijder het, en check zijn oefening opnieuw: de check bewaart je code dan weer.*",
                "paste": "*Je `paste` in `work/projects/cnn/paste.py` stopt met een fout, of geeft geen mask van 128 bij 128 pixels. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "proposals": "*Je `proposals` in `work/projects/cnn/proposals.py` stopt met een fout, of geeft geen 1 tot 32 boxes voor een scène. Heb je het bestand veranderd? Verwijder het, en check zijn oefening in [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van de detector. Heb je het bestand veranderd? Verwijder het, en check zijn oefening in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course verving " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de notitie onder *Waarom* zegt."
            _labels = {"title": "Loss van elk deel", "x": "epoch", "y": "loss", "rpn": "RPN, doorgetrokken", "second": "tweede fase, gestreept",
                       "mask": "mask head, met rondjes",
                       "alt": "De training loss van elk deel na elke epoch: van het RPN doorgetrokken, van de tweede fase gestreept, van de mask head doorgetrokken met rondjes."}
            _words = {"scene": "scène {seed}: {n} masks", "alt": "Drie testscènes in grijs, elk met de randen van zijn echte masks en de masks van de boxes van 0,5 of meer die Mask R-CNN er na die epoch aan gaf, gevuld."}
            _curves, _moments = _result["curves"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_m) for _m in _moments[_e - 1])
                return (C.curves_picture(_curves, _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"Na epoch {_e}: loss {L.number(_curves['rpn'][_e - 1], 3)} voor het RPN, {L.number(_curves['second'][_e - 1], 3)} voor de tweede fase en {L.number(_curves['mask'][_e - 1], 3)} voor de mask head, en {_n} masks in deze drie scènes, die {C.SHOWN_SHAPES} vormen hebben.")

            if len(_moments) == len(_curves["mask"]) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_moments), values=list(range(1, len(_moments) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: na welke epoch krijgen de masks de vorm van hun voorwerp?")]
            else:
                _pic = [mo.Html(C.curves_picture(_curves, _labels, L.lang))] if _curves["mask"] else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("cirkels", "vierkanten", "driehoeken"), _result["aps"]))
            _medians = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("cirkels", "vierkanten", "driehoeken"), _result["median"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Je Mask R-CNN, met {C.whole(_result['params'], L.lang)} parameters, vindt {_result['found']} van de {_result['shapes']} vormen van de 200 testscènes met een box van hun soort, met een score van 0,5 of meer en een IoU boven 0,5; {_result['false']} van zijn boxes van 0,5 of meer vinden geen vorm. De mediane IoU van de masks van de vormen die hij vindt: {_medians}. Zijn mask AP bij een IoU van 0,5 is {L.number(_result['map'], 3)}, met een AP van {_aps}; bij 0,75 is hij {L.number(_result['map75'], 3)}. De Mask R-CNN van de course, op dezelfde manier getraind met de seeds 0 tot 4, heeft een mask AP van 0,941 tot 0,948 bij een IoU van 0,5, en 0,846 tot 0,857 bij 0,75; met RoI pooling in plaats van RoIAlign 0,915 tot 0,925, en 0,535 tot 0,592 bij 0,75.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 5. Waarom afronden een mask schaadt
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-align",
        question="Getraind met RoI pooling vindt de Mask R-CNN van de course bij een IoU van 0,5 bijna evenveel vormen als met RoIAlign, maar bij 0,75 zakt zijn mask AP van ongeveer 0,85 naar onder de 0,6. Waarom schaadt afronden op hele vakjes de masks zoveel meer dan de soorten? En wat doet RoIAlign in plaats daarvan?",
        model_answer="RoI pooling neemt de hele vakjes waar de hoeken van een box in liggen, dus zijn getallen beginnen niet waar de box begint: ze kunnen bijna een heel vakje, 8 pixels, ernaast liggen, en boxes die een paar pixels verschillen, krijgen dezelfde getallen. Het mask wordt in de precieze box geplakt, dus de mask head moet raden waar in zijn bins de randen van de vorm liggen, en een mask waarvan de rand een paar pixels verkeerd ligt, verliest veel van zijn IoU. De soort van een box hangt alleen af van wat erin zit, en een paar pixels veranderen dat niet. RoIAlign neemt het midden van elke bin van de box zelf, in vakjes zonder af te ronden, en de waarde daar met bilineaire interpolatie van de vier vakjes eromheen: zijn getallen bewegen mee met de box, en elke bin krijgt zijn eigen getallen. Van 8 bij 8 bins op de box van de cirkel gaf RoI pooling maar 16 verschillende getallen per channel.",
        points=[
            "je zegt dat RoI pooling een box afrondt op de hele vakjes waar zijn hoeken in liggen, zodat zijn getallen tot een vakje naast de box kunnen liggen, en boxes die een paar pixels verschillen dezelfde getallen krijgen",
            "je zegt dat een mask zijn randen op de pixel precies moet leggen, terwijl de soort van een box nauwelijks verandert als je een paar pixels ernaast kijkt",
            "je zegt dat RoIAlign de feature maps leest op de precieze middens van de bins van de box, gemengd uit de vier vakjes rond elk, zodat zijn getallen meebewegen met de box",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De two-stage detector van [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) heeft nu een derde head: een klein FCN, zoals in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl), dat in elke box een mask tekent, zodat elk voorwerp een eigen mask krijgt, ook als het een ander van zijn soort raakt. Je `bilinear` leest feature maps op elk punt, zoals je `value_at` van [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl) een afbeelding las; je `roi_align` gebruikt hem om elke box zonder afronden tot k bij k bins te snijden; en je `paste` zet het kleine mask van een box terug in de scène. Op de scènes van de course passen de masks met RoIAlign veel beter dan met RoI pooling, en de boxes ook: bij een IoU van 0,75 steeg de mask AP van tussen 0,535 en 0,592 naar tussen 0,846 en 0,857, met de seeds 0 tot 4. Je project `cnn` heeft nu `bilinear.py`, `roi_align.py` en `paste.py`. In de volgende les, *Segmenteren wat je aanwijst: SAM 2.1*, tekent een groot pretrained model het mask van alles wat je aanwijst, ook van dingen waar het geen naam voor heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_bilinear, result_align, result_paste, choice_learns, explain_why):
    L.wrap_up(result_bilinear, result_align, result_paste, choice_learns, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
