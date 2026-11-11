# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-anchors. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-anchors", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl) en [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_detr():
    quick_detr = L.choice(
        "quick-detr",
        question="DETR geeft een query de box `(0.5, 0.25, 0.2, 0.1)`: `(cx, cy, w, h)`, als deel van de foto. Op een foto van 600 pixels breed en 400 pixels hoog, waar ligt het midden van de box, en hoe breed is hij?",
        options={
            "right": "Het midden op kolom 300 en rij 100; 120 pixels breed",
            "height": "Het midden op kolom 300 en rij 100; 80 pixels breed",
            "swap": "Het midden op kolom 150 en rij 200; 120 pixels breed",
            "corner": "De linkerkant op kolom 300 en de bovenkant op rij 100; 120 pixels breed",
        },
        correct="right",
        why={
            "right": "cx en w zijn delen van de breedte: 0,5 × 600 = 300 en 0,2 × 600 = 120. cy is een deel van de hoogte: 0,25 × 400 = 100. DETR schrijft een box als zijn midden en zijn grootte. De detectors van deze les maken hun boxes ook uit een midden en een grootte, maar niet vanuit het niets.",
            "height": "w is een deel van de breedte, niet van de hoogte: 0,2 × 600 = 120. Zie [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl).",
            "swap": "cx, het eerste getal, is het midden opzij, een deel van de breedte: 0,5 × 600 = 300. cy, het tweede, is het midden omlaag: 0,25 × 400 = 100. Zie [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl).",
            "corner": "cx en cy zijn het midden van de box, niet zijn hoek linksboven: de linkerkant ligt een halve breedte verder naar links, op kolom 300 − 60 = 240. Zie [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl).",
        },
    )
    quick_detr
    return (quick_detr,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="Een CNN halveert drie keer de hoogte en de breedte van zijn maps, elke keer met stride 2. Hoe groot zijn bij een scène van 128 bij 128 pixels zijn laatste feature maps, en voor hoeveel pixels van de scène staat elk vakje ervan?",
        options={
            "right": "16 bij 16; elk vakje staat voor 8 bij 8 pixels",
            "once": "64 bij 64; elk vakje staat voor 2 bij 2 pixels",
            "sixteen": "16 bij 16; elk vakje staat voor 16 bij 16 pixels",
            "four": "8 bij 8; elk vakje staat voor 16 bij 16 pixels",
        },
        correct="right",
        why={
            "right": "128 / 2 / 2 / 2 = 16, en 128 / 16 = 8: elk vakje van een map staat voor 8 bij 8 pixels van de scène. Vandaag krijgt elk van die 256 vakjes eigen boxes.",
            "once": "Dan halveer je één keer. Drie keer halveren gaat 128, 64, 32, 16. Zie [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).",
            "sixteen": "16 vakjes over 128 pixels is 128 / 16 = 8 pixels per vakje, niet 16. Zie [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).",
            "four": "Dan halveer je vier keer: 128, 64, 32, 16, 8. Drie keer geeft 16 bij 16. Zie [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    DETR, uit [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl), leerde elk van zijn 100 object queries een box geven vanuit het niets: overal in de foto, van elke grootte. Veel detectors van vóór DETR, en veel van daarna, maken dat makkelijker. Ze leggen een vaste set boxes over de hele foto, een paar groottes op elk vakje van een feature map, en leren voor elke box maar twee dingen: of hij op een voorwerp ligt, en wat voor soort; en hoe ver hij zich moet verplaatsen en uitrekken om te passen. Die vaste boxes heten anchors. Faster R-CNN (Ren en collega's, 2015) voerde ze in, en SSD, YOLOv2 en RetinaNet gebruiken ze ook. Eén moeilijke vraag, waar liggen de boxes, wordt zo een heleboel kleine, elk over één anchor bij een voorwerp. Maar om ze te trainen heeft elke anchor een goed antwoord nodig: welk voorwerp moet hij leren, als hij er al een moet leren? De IoU uit [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) beslist dat.

    **Een voorbeeld uit het dagelijks leven.** Een kledingwinkel naait niet voor elke klant een jas vanuit het niets. Er hangen een paar maten aan het rek. Je pakt de maat die het best past, en de kleermaker neemt hem alleen een beetje in of laat hem wat uit: een kleine verandering van een maat die al dichtbij zat. Hoe goed elke maat past, bepaalt waar de kleermaker begint.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "how-many",
        question="De scène van het uitgewerkte voorbeeld heeft 4 vormen. Een feature map van 16 bij 16 vakjes krijgt 3 anchors per vakje, van 16, 24 en 32 pixels: 768 anchors. Een anchor leert een vorm als zijn IoU met die vorm 0,5 of meer is, of als hij de beste anchor van die vorm is. Hoeveel van de 768 leren een vorm?",
        options={
            "four": "4: één per vorm",
            "right": "20",
            "hundred": "Ongeveer 100",
            "half": "Ongeveer 400: de helft",
        },
        reveal="**20.** 8 anchors leren de cirkel, 6 het vierkant, 4 de grote driehoek en 2 de kleine driehoek. 735 leren achtergrond, en 13 worden genegeerd. Over 500 scènes zijn het er gemiddeld 17,2 per scène: 2,2% van de anchors. Bijna elke anchor leert dat daar niets is.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Anchors op elk vakje.** Een CNN voor object detection maakt van de scène feature maps, zoals de CNN's uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) doen. Na drie keer halveren geeft een scène van 128 bij 128 pixels maps van 16 bij 16 vakjes, die elk voor 8 bij 8 pixels staan. Een **anchor box**, kortweg anchor, is een vaste box rond het midden van zo'n vakje. Elk vakje krijgt dezelfde anchors, in een paar groottes en vormen. Faster R-CNN neemt 3 groottes, met oppervlaktes van 128², 256² en 512² pixels, en 3 vormen, 1:1, 1:2 en 2:1: 9 anchors per vakje, zo'n 20.000 op een foto van 1000 bij 600 pixels. De boxes van de vormen van de course zijn 9 tot 44 pixels per kant, en bijna vierkant: bij 90% ervan ligt de hoogte gedeeld door de breedte tussen 0,88 en 1,13. Daarom neemt de course 3 vierkante groottes, 16, 24 en 32 pixels: op 16 bij 16 vakjes 768 anchors.

        **Wat elke anchor leert.** Voor elke anchor geeft de detector een soort, een van de vormen of achtergrond, en vier getallen die de anchor verplaatsen en uitrekken tot een box. Om hem te trainen heeft elke anchor een goed antwoord nodig, en zijn IoU met de echte boxes beslist welk. Elke anchor neemt de echte box waar hij het meest mee overlapt. Bij een IoU van 0,5 of meer leert hij die box: zijn soort, en de vier getallen die van de anchor die box maken. Onder 0,4 leert hij achtergrond. Daartussen wordt hij genegeerd: hij ligt deels op een vorm, en met elk van beide antwoorden zou de detector iets leren dat deels fout is. Dit zijn de drempels van RetinaNet (Lin en collega's, 2017); Faster R-CNN neemt 0,7 en 0,3, en SSD 0,5. Als laatste krijgt elke echte box ook zijn beste anchor, de anchor met zijn hoogste IoU, ook onder 0,5, en ook als die anchor een andere box meer overlapt, zodat geen voorwerp zonder anchor blijft. Faster R-CNN voegt die regel toe omdat de drempel op zich in zeldzame gevallen geen enkele anchor vindt die de box leert. Op de scènes van de course hebben 25 van de 2012 vormen van 500 scènes hem nodig; hun beste anchors halen 0,40 of meer. Ligt het midden van een box precies tussen twee vakjes in, dan kunnen twee anchors de hoogste IoU delen; de regel van Faster R-CNN noemt de anchor of anchors met de hoogste IoU, dus leren ze de box allebei.

        **Box offsets.** Een anchor leert niet de vier kanten van zijn box, maar hoe ver zijn box ernaast zit: de verplaatsing van het midden, als deel van de hoogte en de breedte van de anchor, en hoeveel keer zo groot de box is, als log. Die vier getallen zijn de **box offset** van de anchor, zoals Faster R-CNN ze schrijft, hier met de rijen eerst:
        """),
        L.formula(r"t_y = \frac{y - y_a}{h_a}, \qquad t_x = \frac{x - x_a}{w_a}, \qquad t_h = \log\frac{h}{h_a}, \qquad t_w = \log\frac{w}{w_a}"),
        mo.md(r"""
        Hier is $(y, x)$ het midden van de echte box, en zijn $h$ en $w$ zijn hoogte en breedte; $y_a$, $x_a$, $h_a$ en $w_a$ zijn hetzelfde voor de anchor. Een box van de course telt beide uiteinden mee, dus zijn hoogte is onder − boven + 1, en zijn midden (boven + onder) / 2. Een anchor die precies op zijn box ligt, heeft vier nullen. Door te delen door de grootte van de anchor lijken de getallen op elkaar voor kleine en grote boxes: een verplaatsing van een halve anchor is voor allebei 0,5. Door de log liggen twee keer zo groot en half zo groot even ver van 0, op 0,69 en −0,69, en kan de weg terug nooit een negatieve grootte geven:
        """),
        L.formula(r"y = y_a + t_y \, h_a, \qquad h = h_a \, e^{t_h}"),
        mo.md(r"""
        en net zo voor $x$ en $w$; uit het midden en de grootte volgen de kanten, boven = y − (h − 1) / 2 en onder = y + (h − 1) / 2. Omdat een anchor alleen boxes leert die hij al flink overlapt, bijna altijd voor 0,5 of meer, blijven zijn offsets klein: over 500 scènes is een verplaatsing gemiddeld 0,11 van de grootte van de anchor, omhoog of omlaag, naar links of naar rechts, en een log van de grootte 0,15, groter of kleiner; 95% blijft onder 0,25 en 0,32.
        """),
        L.watch_out("Een anchor leert hoogstens één box: meestal die waar hij het meest mee overlapt, maar de laatste regel kan hem een andere box geven, een box waarvan hij de beste anchor is. Een box kan veel anchors hebben: de cirkel heeft er 8. Een anchor geeft één soort en één box offset, dus hij kan geen twee boxes leren, ook niet als hij ze allebei voor 0,5 of meer overlapt."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **anchor box** (anchor) | een vaste box rond het midden van een vakje van een feature map; elk vakje heeft dezelfde anchors, in een paar groottes en vormen, en de detector leert voor elke anchor een soort en een box offset | `anchors[i]` |
        | **box offset** | de vier getallen die van een anchor een box maken: de verplaatsing van het midden als deel van de hoogte en de breedte van de anchor, en de log van hoeveel keer hoger en breder de box is | `t` |
        | $y_a$, $x_a$, $h_a$, $w_a$ | het midden en de grootte van een anchor: zijn rij en kolom, zijn hoogte en breedte | `middle_a`, `size_a` |
        | $t_y$, $t_x$, $t_h$, $t_w$ | de box offset: de verplaatsing omlaag en opzij, de verandering van de hoogte en van de breedte | `t[:, :2]`, `t[:, 2:]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_anchors():
    mo.vstack([
        mo.md("**Anchors op de scène van het uitgewerkte voorbeeld**"),
        mo.Html(C.anchors_picture({
            "one": "de {n} anchors van één vakje",
            "learn": "de {n} anchors die een vorm leren",
            "alt": "Twee plaatjes van een scène met twee driehoeken, een vierkant en een cirkel. Links de vakjes van de feature map als dunne lijnen, één vakje donkerder, met drie gestreepte boxes van 16, 24 en 32 pixels rond zijn midden, en de echte box van de cirkel, doorgetrokken. Rechts de vier echte boxes, doorgetrokken, en de 20 gestreepte anchors die een vorm leren: 8 rond de cirkel, 6 rond het vierkant, 4 rond de grote driehoek en 2 bij de kleine driehoek.",
        })),
        mo.md(r"""
        Links vakje (9, 8), in rij 9 en kolom 8 van de map, met zijn drie anchors en de echte box van de cirkel: de anchor van 32 pixels past erop met een IoU van 0,64. Rechts de 20 anchors die een vorm leren. De kleine driehoek heeft geen anchor met een IoU van 0,5 of meer: zijn twee beste, van 16 pixels, halen elk 0,49, en leren hem via de laatste regel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Heb je meer dan één grootte nodig? De slider geeft elk vakje één vierkante anchor, van 8 tot 56 pixels. Links de anchors die met een vorm van de scène een IoU van 0,5 halen; rechts, voor elke grootte, het deel van de vormen van 500 scènes dat zo'n anchor heeft. De stippellijn is de drie groottes van de course samen.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(z):
        z = int(z)
        _html, _stop = C.slider_frame(z, {"title": "anchors van {size} pixels: {found} van de {n} vormen", "three": "16, 24 en 32: {share}%",
                                          "size": "grootte van de anchor in pixels", "share": "vormen van 500 scènes met een anchor\nwaarvan de IoU 0,5 of meer is", "point": ",",
                                          "alt": "De scène met zijn vier echte boxes en de gestreepte anchors van {size} bij {size} pixels die met een vorm een IoU van 0,5 halen: {found} van de {n} vormen hebben er een. Ernaast het deel van de vormen van 500 scènes met zo'n anchor, bij elke grootte van 8 tot 56 pixels: 0 bij 8, stijgend tot 81% bij 28, en dalend tot 7% bij 56, altijd onder de stippellijn van de drie groottes samen, op 98,8%. Bij {size} pixels is het {share}%."})
        return _html, f"Anchors van {z} bij {z} pixels: {len(_stop['positive'])} met een IoU van 0,5 of meer, voor {_stop['found']} van de 4 vormen. Over 500 scènes krijgt {L.number(100 * C.curve()[z], 1)}% van de vormen er een."

    explore_size = L.explore("size-slider", "grootte van de anchor in pixels", _frame, start=16, values=C.SLIDER, role="hyper",
                             task="Zoek de grootte waarbij de meeste vormen van de 500 scènes een anchor krijgen. Krijgen alle vier de vormen van deze scène er daar een?")
    explore_size
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    28 pixels geeft de meeste: 81,0% van de vormen van 500 scènes, en hier 3 van de 4 vormen, maar niet de kleine driehoek. Die krijgt alleen bij 18 en 20 pixels een anchor met een IoU van 0,5 of meer, en het vierkant alleen van 24 tot 46: geen enkele grootte past bij alle vormen van 9 tot 44 pixels. De drie groottes samen geven 98,8%; met de 0,7 van Faster R-CNN zou dat maar 55,0% zijn. Anchors in meer vormen helpen hier niet: 3 groottes keer de 3 vormen van Faster R-CNN, 9 anchors per vakje, geven 99,0%, omdat de boxes bijna vierkant zijn. Faster R-CNN zag hetzelfde op PASCAL VOC: met één anchor per vakje scoorde het 3 tot 4 punten mAP minder dan met zijn negen, en 3 groottes van één vorm, met 69,8% mAP, deden het even goed als 3 groottes van 3 vormen, met 69,9%. En de vakjes moeten dicht genoeg bij elkaar liggen: met dezelfde drie groottes op 8 bij 8 vakjes van 16 pixels krijgt maar 69,6% van de vormen een anchor met een IoU van 0,5 of meer, omdat het midden van een kleine vorm dan ver van elke anchor kan liggen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Om elke anchor te vertellen wat hij moet leren:

        1. Leg de anchors: rond het midden van elk vakje van de feature map één anchor van elke grootte en vorm. Met vakjes van $s$ pixels ligt het midden van vakje $(i, j)$ op rij $i \cdot s + (s - 1)/2$ en kolom $j \cdot s + (s - 1)/2$.
        2. Bereken de IoU van elke anchor met elke echte box: een array met een rij per anchor en een kolom per box.
        3. Koppel. Elke anchor neemt de box waar hij het meest mee overlapt: vanaf 0,5 leert hij die box, van 0,4 tot 0,5 wordt hij genegeerd, onder 0,4 leert hij achtergrond. Daarna krijgt elke box zijn beste anchors, die met zijn hoogste IoU, ook onder 0,5 en ook als ze een andere box meer overlappen.
        4. Bereken voor elke anchor die een box leert zijn box offset: $t_y = (y - y_a)/h_a$, $t_x = (x - x_a)/w_a$, $t_h = \log(h/h_a)$, $t_w = \log(w/w_a)$, met het midden (boven + onder) / 2 en de hoogte onder − boven + 1.
        5. De weg terug, voor de offsets die een detector voorspelt: $y = y_a + t_y h_a$ en $h = h_a e^{t_h}$, en net zo opzij; dan de kanten, boven = y − (h − 1) / 2 en onder = y + (h − 1) / 2.
        """),
        L.watch_out("Geef elke box zijn beste anchors als laatste. Komen de drempels na die regel, dan wordt een beste anchor tussen 0,4 en 0,5 weer genegeerd, en blijft een box als de kleine driehoek zonder anchor."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op de scène van [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl), opnieuw gemaakt met `scene(18)`. De eerste cel doet stap 1 tot en met 3 in NumPy. `iou_matrix` is de IoU uit [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) voor alle anchors tegelijk: `anchors[:, 0]` is de kolom met al hun bovenkanten, en `np.minimum` neemt element voor element de kleinste van twee getallen, zoals `np.maximum` de grootste neemt. Stap 3 gebruikt boolean masks: `most >= 0.5` is `True` voor elke anchor met een grootste IoU van 0,5 of meer, en `label[...] = ...` zet alleen die. De tweede cel doet stap 4 en 5 voor de beste anchor van de cirkel. Voer ze uit zoals ze zijn; verander dan `sizes` waar het commentaar dat zegt, om er een hoge en een brede anchor bij te zetten, en voer ze opnieuw uit.
    """)
    return


@app.cell
def example_match():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    truth = np.array([shape.box for shape in s.shapes], dtype=float)   # de echte boxes, één rij per vorm

    # 1. de anchors: rond het midden van elk vakje van een map van 16 bij 16 vakjes,
    #    elk vakje 8 bij 8 pixels van de scène, één anchor van elke grootte
    cells, stride = 16, 8
    sizes = [(16, 16), (24, 24), (32, 32)]   # (hoogte, breedte); verander: voeg (34, 17) en (17, 34) toe
    rows = []
    for i in range(cells):
        for j in range(cells):
            y = i * stride + (stride - 1) / 2   # het midden van vakje (i, j)
            x = j * stride + (stride - 1) / 2
            for h, w in sizes:
                rows.append((y - (h - 1) / 2, x - (w - 1) / 2, y + (h - 1) / 2, x + (w - 1) / 2))
    anchors = np.array(rows)
    print(truth)
    print(anchors.shape)
    print(anchors[:3])


    def iou_matrix(anchors, boxes):
        # de IoU van elke anchor met elke box, zoals in cv-iou, voor alle anchors tegelijk
        ious = np.zeros((len(anchors), len(boxes)))
        for k in range(len(boxes)):
            box = boxes[k]
            top = np.maximum(anchors[:, 0], box[0])
            left = np.maximum(anchors[:, 1], box[1])
            bottom = np.minimum(anchors[:, 2], box[2])
            right = np.minimum(anchors[:, 3], box[3])
            overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
            area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
            area_b = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            ious[:, k] = overlap / (area_a + area_b - overlap)
        return ious


    # 2. de IoU van elke anchor met elke echte box: één kolom per box
    ious = iou_matrix(anchors, truth)
    print(ious.shape, ious.max(axis=0))
    # 3. koppel: elke anchor neemt de box waar hij het meest mee overlapt; vanaf 0,5 leert hij
    #    die box, onder 0,4 achtergrond (-1), daartussen wordt hij genegeerd (-2)
    best = ious.argmax(axis=1)
    most = ious.max(axis=1)
    label = np.zeros(len(anchors), dtype=int) - 1
    label[most >= 0.5] = best[most >= 0.5]
    label[(most >= 0.4) & (most < 0.5)] = -2
    # en elke echte box krijgt ook zijn beste anchors, ook onder 0,5
    for k in range(len(truth)):
        label[ious[:, k] == ious[:, k].max()] = k
    print((label >= 0).sum(), (label == -2).sum(), (label == -1).sum())
    for k in range(len(truth)):
        print(s.shapes[k].kind, (label == k).sum())
    return anchors, ious, label, np, truth


@app.cell
def example_offsets(anchors, ious, np, truth):
    # 4. de offsets van de beste anchor van de cirkel: hoe ver het midden schuift, als deel
    #    van de hoogte en de breedte van de anchor, en de log van hoeveel groter de box is
    anchor = anchors[ious[:, 3].argmax()]
    box = truth[3]
    ya, xa = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    ha, wa = anchor[2] - anchor[0] + 1, anchor[3] - anchor[1] + 1
    yb, xb = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hb, wb = box[2] - box[0] + 1, box[3] - box[1] + 1
    t = np.array([(yb - ya) / ha, (xb - xa) / wa, np.log(hb / ha), np.log(wb / wa)])
    print(anchor, box)
    print(t)
    # 5. terug: de offsets maken van de anchor weer de echte box
    y2, x2 = ya + t[0] * ha, xa + t[1] * wa
    h2, w2 = ha * np.exp(t[2]), wa * np.exp(t[3])
    print(np.array([y2 - (h2 - 1) / 2, x2 - (w2 - 1) / 2, y2 + (h2 - 1) / 2, x2 + (w2 - 1) / 2]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `truth` bevat de vier echte boxes, en `anchors` heeft shape (768, 4): 256 vakjes keer 3 groottes. De eerste drie zijn de anchors van vakje (0, 0), met zijn midden op rij en kolom 3,5; ze steken buiten de scène uit, en dat geeft niet. `ious` heeft shape (768, 4). De hoogste IoU van elke vorm is 0,49 voor de kleine driehoek, 0,67 voor de grote driehoek, 0,70 voor het vierkant en 0,64 voor de cirkel. 20 anchors leren een vorm, 13 worden genegeerd, en 735 leren achtergrond. De kleine driehoek heeft 2 anchors, hoewel geen van beide 0,5 haalt: zijn midden ligt op kolom 15,5, precies tussen de middens van de vakjes in kolom 1 en 2, op 11,5 en 19,5, dus zijn twee beste anchors halen allebei 0,4884, en de laatste regel geeft hem ze allebei.

    In de tweede cel is de beste anchor van de cirkel (60, 52, 91, 83), 32 bij 32 pixels rond het midden van vakje (9, 8). De box van de cirkel, (58, 50, 86, 78), heeft zijn midden 3,5 pixels hoger en 3,5 pixels verder naar links, en is 29 pixels hoog en breed. Dus zijn de offsets −3,5 / 32 = −0,109, twee keer, en log(29 / 32) = −0,098, twee keer. Stap 5 maakt van de anchor weer precies de box van de cirkel. Met de hoge en de brede anchor erbij zijn er 1280 anchors: 27 leren een vorm, 28 worden genegeerd en 1225 leren achtergrond. Geen enkele vorm krijgt er een betere beste anchor door, omdat elke box bijna vierkant is.
    """)
    return


@app.cell(hide_code=True)
def text_torch_step():
    mo.md(r"""
    ### Een trede hoger: hetzelfde in PyTorch

    Een detector geeft zijn offsets als tensors, dus hier is stap 4 in PyTorch, voor alle 20 anchors die een box leren tegelijk. `anchors[learns]` houdt de rijen van de anchors die een box leren, en `truth[label[learns]]` de echte box van elk ervan: een array als index. `anchors_t[:, :2]` is het paar (boven, links) van elke anchor en `anchors_t[:, 2:]` het paar (onder, rechts), dus één regel geeft de middens van allemaal, en één regel hun groottes. `torch.log` werkt zoals `np.log`, en `torch.cat(..., dim=1)` zet de verplaatsing en de verandering van grootte naast elkaar.

    | NumPy, één anchor | PyTorch, alle anchors tegelijk |
    |---|---|
    | `ya, xa = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2` | `middle_a = (anchors_t[:, :2] + anchors_t[:, 2:]) / 2` |
    | `ha, wa = anchor[2] - anchor[0] + 1, anchor[3] - anchor[1] + 1` | `size_a = anchors_t[:, 2:] - anchors_t[:, :2] + 1` |
    | `np.array([(yb - ya) / ha, (xb - xa) / wa, np.log(hb / ha), np.log(wb / wa)])` | `torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)` |
    """)
    return


@app.cell
def example_torch(anchors, label, truth):
    import torch

    # 4. de offsets van alle anchors die een box leren tegelijk, in PyTorch
    learns = label >= 0
    anchors_t = torch.tensor(anchors[learns]).float()       # shape (20, 4)
    boxes_t = torch.tensor(truth[label[learns]]).float()    # de box die elk ervan leert
    middle_a = (anchors_t[:, :2] + anchors_t[:, 2:]) / 2   # (y, x) van elke anchor
    size_a = anchors_t[:, 2:] - anchors_t[:, :2] + 1       # (h, w) van elke anchor
    middle = (boxes_t[:, :2] + boxes_t[:, 2:]) / 2
    size = boxes_t[:, 2:] - boxes_t[:, :2] + 1
    targets = torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)
    print(targets.shape)
    print(targets[-3:])
    print(targets.abs().max())
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    `targets` heeft shape (20, 4): een box offset voor elke anchor die een box leert, wat een detector moet leren voorspellen. De eerste van de laatste drie rijen is de beste anchor van de cirkel, zoals in de cel met NumPy. De andere twee zijn de anchors van de kleine driehoek. Bij allebei ligt het midden van de driehoek 0,125 van de hoogte van de anchor lager; bij de een ligt het 0,25 van zijn breedte naar rechts, bij de ander 0,25 naar links. Beide anchors zijn even groot als de box van de driehoek, 16 bij 16, dus hun veranderingen van grootte zijn 0. De grootste offset in de scène is 0,35.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat elke anchor leert
    """)
    return


@app.cell(hide_code=True)
def choice_learn_cell():
    choice_learn = L.choice(
        "learn",
        question="Vier anchors en twee echte boxes, A en B. Anchor 1 heeft een IoU van 0,62 met A en 0,30 met B; anchor 2 heeft 0,45 met A en 0,05 met B; anchor 3 heeft 0,15 met A en 0,42 met B; anchor 4 heeft 0,10 met A en 0,20 met B. Geen andere anchor overlapt B meer dan anchor 3. Wat leert elke anchor?",
        options={
            "right": "1 leert A, 2 wordt genegeerd, 3 leert B, 4 leert achtergrond",
            "no-rule": "1 leert A, 2 wordt genegeerd, 3 wordt genegeerd, 4 leert achtergrond",
            "low": "1 leert A, 2 leert A, 3 leert B, 4 leert achtergrond",
            "no-band": "1 leert A, 2 leert achtergrond, 3 leert B, 4 leert achtergrond",
        },
        correct="right",
        why={
            "right": "Anchor 1 haalt 0,5 met A. De beste IoU van anchor 2 is 0,45, tussen 0,4 en 0,5: genegeerd. Die van anchor 3, 0,42, ligt ook daartussen, maar hij is de beste anchor van B, dus de laatste regel geeft hem B; zonder die regel zou niets B leren. Anchor 4 blijft onder 0,4: achtergrond.",
            "no-rule": "De anchors van B blijven allemaal onder 0,5, dus met alleen de drempels leert niets B. De laatste regel geeft elke box zijn beste anchor, ook onder 0,5: anchor 3.",
            "low": "Een anchor leert een box vanaf 0,5. Anchor 2, op 0,45, ligt deels op A: hij wordt genegeerd, zodat hij A niet leert en achtergrond ook niet.",
            "no-band": "Onder 0,4 is achtergrond; van 0,4 tot 0,5 wordt een anchor genegeerd. Anchor 2, op 0,45, ligt deels op A: als hij achtergrond leerde, zou de detector leren dat een deel van A niets is.",
        },
    )
    choice_learn
    return (choice_learn,)


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Koppel de anchors

        Maak `match(ious)` af: die krijgt de IoU van elke anchor met elke echte box, een array met een rij per anchor en een kolom per box, en geeft een array terug met één heel getal per anchor: het nummer van de box die hij leert, -1 voor achtergrond, -2 als hij wordt genegeerd. `iou_matrix` uit het uitgewerkte voorbeeld staat er al. De check geeft je functie de vier anchors uit oefening 1, IoU's van precies 0,5 en 0,4, de scène van het uitgewerkte voorbeeld, en een andere scène met andere anchors. Voor de scènes legt hij de anchors met `anchor_boxes` uit `everything_ai.scenes`, die doet wat stap 1 van het uitgewerkte voorbeeld doet: `anchor_boxes()` geeft dezelfde 768 anchors, en `anchor_boxes(cells=8, stride=16, sizes=((24, 24), (40, 40)))` twee groottes op 8 bij 8 vakjes van 16 pixels. Als hij goed is, bewaart hij `match` en `iou_matrix` in je project, als `work/projects/cnn/match.py`.
        """),
        L.task("Laat `match` voor elke anchor geven wat hij leert: een box, achtergrond of niets."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("match", lines=10)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "match",
        [
            "Stap 3 van het recept. Voor elke anchor bevat zijn rij van `ious` zijn IoU met elke box: `ious.argmax(axis=1)` geeft de box waar hij het meest mee overlapt, en `ious.max(axis=1)` die IoU.",
            "Zet elke anchor eerst op -1, en gebruik dan boolean masks, zoals het uitgewerkte voorbeeld doet: `label[most >= 0.5] = best[___]`, en -2 waar `most` van 0,4 tot 0,5 is.",
            "Ga als laatste de kolommen langs: `for k in range(ious.shape[1]):`, en geef elke anchor met de hoogste IoU van kolom `k` het nummer `k`: `label[ious[:, k] == ___] = k`.",
        ],
        why="Bij oefening 1: anchor 1 leert A, 0, op 0,62; anchor 2 wordt genegeerd, -2, op 0,45; anchor 4 leert achtergrond, -1, op 0,20; en anchor 3, op 0,42, wordt eerst genegeerd en is dan de beste anchor van B, dus hij leert B, 1. `match` geeft `[0, -2, 1, -1]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_boxes_from():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Van offsets terug naar boxes

        Een detector geeft voor elke anchor een box offset, en jij hebt zijn boxes nodig. Maak `boxes_from(anchors, t)` af: die krijgt anchors, een tensor met shape (N, 4) en rijen `(boven, links, onder, rechts)`, en hun offsets, een tensor met shape (N, 4) en rijen $(t_y, t_x, t_h, t_w)$, en geeft de boxes terug, een tensor met shape (N, 4): stap 5 van het recept, voor alle anchors tegelijk, in PyTorch. `offsets`, de cel met PyTorch uit het uitgewerkte voorbeeld als functie, staat er al: `boxes_from` gaat de weg terug. De check begint met de beste anchor van de cirkel en zijn offsets, en probeert dan offsets van 0, een box die twee keer zo hoog en half zo breed is, en de weg terug uit `offsets`. Als hij goed is, bewaart hij `boxes_from` en `offsets` in je project, als `work/projects/cnn/offsets.py`.
        """),
        L.task("Laat `boxes_from` van anchors en hun box offsets boxes maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_boxes_from():
    box_boxes_from = L.code_box("boxes-from", lines=8)
    box_boxes_from
    return (box_boxes_from,)


@app.cell(hide_code=True)
def check_boxes_from(box_boxes_from):
    result_boxes_from = L.check("boxes-from", box_boxes_from.value)
    result_boxes_from
    return (result_boxes_from,)


@app.cell(hide_code=True)
def hints_boxes_from():
    L.hints(
        "boxes-from",
        [
            "Begin zoals `offsets`: het midden en de grootte van elke anchor, `middle_a` en `size_a`, elk met shape (N, 2). Dan stap 5: het midden van de box is het midden van de anchor plus `t[:, :2]` keer de grootte van de anchor.",
            "`t[:, 2:]` bevat logs: maak ze ongedaan, zodat de grootte van de box `size_a * torch.___(t[:, 2:])` is.",
            "Uit het midden en de grootte volgen (boven, links) en (onder, rechts), met beide uiteinden meegeteld: `torch.cat([middle - ___, middle + ___], dim=1)`.",
        ],
        why="Voor de anchor van de cirkel, (60, 52, 91, 83): zijn midden is (75,5; 67,5) en zijn grootte 32 bij 32. De verplaatsing −0,1094 × 32 = −3,5 brengt het midden naar (72, 64), en 32 × e^−0,0984 = 29 is de grootte. De kanten zijn 72 − 14 = 58 en 72 + 14 = 86, en opzij 50 en 78: de box van de cirkel.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 4. Waarom offsets?
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-offsets",
        question="Een detector zou op elk vakje ook de vier kanten van elke box in pixels kunnen leren, (boven, links, onder, rechts). Waarom leert hij in plaats daarvan box offsets vanaf een anchor?",
        model_answer="**Hetzelfde voorwerp geeft dezelfde offsets, waar het ook ligt.** Een convolutie schuift hetzelfde filter over elk vakje, dus een CNN geeft overal dezelfde getallen voor hetzelfde patroon; offsets vanaf de eigen anchor van het vakje zijn getallen die hij kan geven. Kanten in pixels zouden voor dezelfde cirkel op een andere plek andere getallen vragen. **De getallen zijn klein, en vergelijkbaar voor elke grootte.** Een anchor leert alleen boxes die hij al flink overlapt, bijna altijd voor 0,5 of meer, dus zijn offsets blijven dicht bij 0: over 500 scènes blijft 95% onder 0,25 en 0,32. Gedeeld door de grootte van de anchor is een verplaatsing van een halve anchor 0,5, voor een kleine anchor net zo goed als voor een grote, dus de loss weegt elk voorwerp even zwaar. **Elke anchor heeft een makkelijkere taak**: alleen de voorwerpen dicht bij zijn vakje, van ongeveer zijn grootte.",
        points=[
            "je zegt dat hetzelfde voorwerp op elk vakje dezelfde offsets geeft, terwijl kanten in pixels van de plek afhangen, en dat een convolutie overal hetzelfde doet",
            "je zegt dat de offsets kleine getallen rond 0 zijn, vergelijkbaar voor kleine en grote voorwerpen, omdat ze delen van de grootte van de anchor zijn",
            "je zegt dat elke anchor alleen voorwerpen dicht bij zich hoeft te vinden, van ongeveer zijn grootte",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De IoU uit [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) beslist nu wat elke anchor leert: vanaf 0,5 de box waar hij het meest mee overlapt, onder 0,4 achtergrond, daartussen niets, en elke box krijgt zijn beste anchors. De feature maps uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) geven de vakjes. Waar DETR, uit [Een pretrained detector gebruiken](/course/open/cv-pretrained-detector?lang=nl), elke box vanuit het niets leerde, leert een anchor een box offset: een verplaatsing als deel van zijn grootte, en de log van zijn verandering van grootte. Bijna elke anchor leert achtergrond: 2,2% ervan leert een vorm. Je project `cnn` heeft nu `match.py` en `offsets.py`. In de volgende les, *Project: je eigen one-stage detector*, geeft een CNN voor elke anchor tegelijk een soort en een box offset, en train je hem met deze targets op de scènes van de course.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_learn, result_match, result_boxes_from, explain_why):
    L.wrap_up(choice_learn, result_match, result_boxes_from, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
