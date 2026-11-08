# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-map. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-map", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Even checken

    Twee korte vragen over [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl) en over precision en recall, uit [Alle thresholds tegelijk: de ROC curve en AUC](/course/open/ml-roc-auc?lang=nl) en de lessen daarvoor.
    """)
    return


@app.cell(hide_code=True)
def quick_check_nms():
    quick_nms = L.choice(
        "quick-nms",
        question="Non-maximum suppression met een drempel van 0,5 krijgt drie boxes: A met een score van 0,9, B met 0,8, waarvan de IoU met A 0,6 is, en C met 0,7, waarvan de IoU met A en met B 0,1 is. Welke boxes houdt hij?",
        options={
            "right": "A en C",
            "one": "alleen A",
            "all": "A, B en C",
            "two": "A en B",
        },
        correct="right",
        why={
            "right": "Ronde 1 houdt A, de hoogste score, en gooit B weg: 0,6 ligt boven 0,5. C blijft, want zijn IoU met A is maar 0,1, en ronde 2 houdt hem. In deze les zijn de boxes van de detector de boxes die NMS heeft gehouden.",
            "one": "A gooit alleen de boxes weg waarvan de IoU ermee boven 0,5 ligt. Hoort C daarbij?",
            "all": "De IoU van B met A is 0,6, boven de drempel van 0,5. Wat gebeurt er met B als A gehouden wordt?",
            "two": "De IoU van B met A is 0,6, dus A gooit hem weg. En C, waarvan de IoU met A maar 0,1 is?",
        },
    )
    quick_nms
    return (quick_nms,)


@app.cell(hide_code=True)
def quick_check_pr():
    quick_pr = L.choice(
        "quick-pr",
        question="Een detector van vijven geeft 10 keer alarm, en 7 daarvan zijn echte vijven. De test set bevat 14 vijven. Wat zijn de precision en de recall?",
        options={
            "right": "precision 0,7, recall 0,5",
            "swapped": "precision 0,5, recall 0,7",
            "both-p": "precision 0,7, recall 0,7",
            "both-r": "precision 0,5, recall 0,5",
        },
        correct="right",
        why={
            "right": "Precision is het deel van zijn alarmen dat klopt: 7 / 10 = 0,7. Recall is het deel van de echte vijven dat hij vindt: 7 / 14 = 0,5. In deze les reken je ze allebei uit na elke box van een detector.",
            "swapped": "Precision deelt door wat de detector zegt, zijn 10 alarmen; recall deelt door wat er echt is, de 14 vijven.",
            "both-p": "7 / 10 is de precision. Recall deelt de 7 treffers door de vijven die er echt zijn: hoeveel zijn dat er?",
            "both-r": "7 / 14 is de recall. Precision deelt de 7 treffers door de alarmen die de detector gaf: hoeveel zijn dat er?",
        },
    )
    quick_pr
    return (quick_pr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom dit ertoe doet

    Een detector tekent boxes, elk met een soort en een score. Hoe goed is hij? Je kunt niet gewoon goede antwoorden tellen, zoals accuracy doet: de detector bepaalt zelf hoeveel boxes hij tekent, een box kan bijna goed zijn, en één voorwerp kan twee boxes krijgen. Een score voor een detector moet zeggen hoeveel voorwerpen hij vindt, hoeveel van zijn boxes fout zijn, en of de boxes waar hij het zekerst van is, kloppen. Papers en ranglijsten vergelijken detectors met één zo'n getal: mean average precision.

    **Een voorbeeld uit het dagelijks leven.** Kinderen zoeken 20 eieren die in een tuin verstopt zijn, en melden elke vondst, die waar ze het zekerst van zijn eerst. Een melding telt als daar een ei ligt dat nog niemand gevonden had; een melding bij een lege plek, of bij een ei dat al gevonden is, telt niet. Een goede zoeker vindt veel van de 20 eieren, en de meldingen waar hij het zekerst van was, kloppen.
    """)
    return


@app.cell(hide_code=True)
def predict_worse():
    guess_worse = L.guess(
        "worse",
        question="Je beoordeelt een detector door zijn boxes voor cirkels af te lopen, vanaf de hoogste score. Twee van de foute zijn de box met de hoogste score, 0,86, die op een vierkant ligt, en de box met de laagste, 0,30, die niet goed genoeg om zijn cirkel past. Zonder welke van de twee zou de detector beter scoren?",
        options={
            "high": "zonder de box met 0,86",
            "low": "zonder de box met 0,30",
            "same": "het maakt niet uit welke",
        },
        reveal="**Zonder de box met 0,86.** De score van deze les, average precision, kijkt naar de precision bij elke gevonden echte box, vanaf de hoogste score omlaag. Een foute box bovenaan verlaagt de precision bij elke gevonden box erna: zonder hem stijgt de score voor cirkels van 0,56 naar 0,73. De foute box met 0,30 komt na de laatste gevonden box: zonder hem blijft de score 0,56. Een foute box kost meer naarmate zijn score hoger is.",
    )
    guess_worse
    return (guess_worse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Om een detector te beoordelen heb je beelden met hun echte boxes nodig, en de boxes die de detector erop tekent na non-maximum suppression, elk met een soort en een confidence score. Eerst bepaal je voor elke box of hij klopt. Een box is een **true positive** als zijn IoU met een echte box van zijn soort in hetzelfde beeld boven 0,5 ligt, en geen box met een hogere score die echte box eerder vond. Elke andere box is een false positive: een box op de achtergrond, op een vorm van een andere soort, een box die te ruim past, of een tweede box op een vorm die al gevonden is. Een echte box die door geen enkele box gevonden wordt, is een false negative, een gemiste box. Je loopt de boxes af vanaf de hoogste score, dus de box waar de detector het zekerst van is, krijgt de eerste kans.

        Daarna loop je voor elke soort apart de lijst van zijn boxes af, de hoogste score eerst. Na elke box is precision het deel van de boxes tot dan toe dat een true positive is, en recall het deel van de echte boxes van die soort dat tot dan toe gevonden is. Stoppen na een box is hetzelfde als een decision threshold bij zijn score, dus deze punten vormen de precision-recall curve van [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl).

        **Average precision**, of AP, is de oppervlakte onder die curve. Elke true positive zet de recall een stap omhoog, 1 gedeeld door het aantal echte boxes, en boven die stap heeft de curve de hoogte van de precision bij die box. AP telt dus de precision bij elke true positive op, en deelt door het aantal echte boxes. Daar komt de naam vandaan: het is het gemiddelde van de precision bij elke echte box, met 0 voor elke echte box die nooit gevonden wordt. AP is alleen 1 als elke echte box gevonden wordt vóór de eerste false positive.

        **Mean average precision**, of mAP, is het gemiddelde van de AP's van alle soorten. PASCAL VOC rekent hem uit met een IoU-drempel van 0,5. COCO rekent hem uit bij tien IoU-drempels, 0,5, 0,55 enzovoort tot 0,95, en neemt daar het gemiddelde van, dat COCO gewoon AP noemt. Allebei doen ze eerst nog een stap, die ze interpolatie noemen: de precision in elk punt wordt de hoogste precision bij die recall of een hogere, zodat de curve nooit meer omhooggaat; COCO leest de curve daarna af bij 101 waarden van de recall. Deze les laat die stap weg, net als scikit-learn. Met de interpolatie van VOC zou de AP voor cirkels in deze les 0,65 zijn in plaats van 0,56; het idee blijft hetzelfde.

        De boxes in deze les komen van de course: voor 8 scènes de boxes die een middelmatige detector zou kunnen tekenen. De meeste vormen krijgen één box van hun eigen soort, een beetje willekeurig verschoven en uitgerekt; af en toe krijgt een vorm geen box, een box van een andere soort of een tweede, ruimere box, en ligt er een box op de achtergrond. De scores zijn meestal hoger naarmate de box beter past, met ruis. Een latere les beoordeelt een echte detector.
        """),
        L.watch_out("Deel door het aantal echte boxes, niet door het aantal true positives. Een echte box die door geen enkele box gevonden wordt, telt als 0: anders krijgt een detector die één box tekent, op het voorwerp waar hij het zekerst van is, een AP van 1."),
        mo.md(r"""
        ### Nieuwe woorden en symbolen

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **true positive**, van een detector | een box waarvan de IoU met een echte box van zijn soort in hetzelfde beeld boven de drempel ligt, en waarvan geen box met een hogere score de echte box eerder vond | `hits[k]` |
        | **average precision** (AP) | voor één soort: de precision bij elke true positive, vanaf de hoogste score omlaag, opgeteld en gedeeld door het aantal echte boxes; de oppervlakte onder de precision-recall curve | `average_precision(hits, total)` |
        | **mean average precision** (mAP) | het gemiddelde van de AP's van alle soorten | `np.mean([0.56, 0.78, 0.73])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_scenes():
    mo.vstack([
        mo.md("**8 scènes, 29 echte boxes, 32 boxes van de detector**"),
        mo.Html(C.scenes_picture({
            "image": "scène {i}",
            "alt": "Acht scènes met cirkels, vierkanten en driehoeken in twee rijen van vier, genummerd van 0 tot 7. Om elke vorm ligt een dunne box, zijn echte box. Daaroverheen liggen de 32 boxes van de detector: 23 true positives, blauw gestreept, en 9 false positives, paars gestippeld, waaronder een box op een vierkant in scène 3 en boxes op de achtergrond. 6 vormen hebben geen true positive.",
        })),
        mo.md(r"""
        De 8 scènes met hun 29 echte boxes, als dunne lijnen, en de 32 boxes van de detector. Bij een IoU-drempel van 0,5 zijn er 23 true positives, blauw gestreept, en 9 false positives, paars gestippeld. 6 vormen worden niet gevonden. Voor de cirkels tekent de detector 11 boxes, en 8 van de 11 cirkels krijgen een true positive.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Wat doet de IoU-drempel? COCO houdt niet op bij 0,5: het vraagt ook om boxes die strak passen. Met de schuifregelaar zet je de IoU-drempel op elk van de tien waarden van COCO, van 0,5 tot 0,95. Links de precision-recall curve van elke soort, met zijn AP; rechts de mAP bij elke drempel, met een stippellijn bij 0,5.
    """)
    return


@app.cell(hide_code=True)
def explore_iou_cell():
    def _frame(t):
        _html, _m = C.iou_frame(t, {"curves": "precision-recall, IoU-drempel {t}", "map": "mAP", "threshold": "IoU-drempel",
                                    "precision": "precision", "recall": "recall", "circle": "cirkel", "square": "vierkant",
                                    "triangle": "driehoek", "point": ",",
                                    "alt": "Links de precision-recall curves van cirkels, vierkanten en driehoeken bij een IoU-drempel van {t}, als trappen, met de AP van elke soort. Rechts de mAP bij de tien IoU-drempels van 0,5 tot 0,95: 0,69 bij 0,5 en 0,55, dan steeds lager, tot 0 vanaf 0,9. Een stippellijn markeert 0,5, en een groter punt de drempel van {t}, met een mAP van {m}."})
        return _html, f"IoU-drempel {L.number(t, 2)}: mAP {L.number(_m, 2)}."

    explore_iou = L.explore("iou", "IoU-drempel", _frame, start=0.5, values=C.STOPS, role="hyper",
                            task="Zoek de grootste IoU-drempel waarbij de mAP nog minstens 0,5 is.")
    explore_iou
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 0,5 en 0,55 is de mAP 0,69. Daarna zakt hij: 0,63 bij 0,6, 0,60 bij 0,65, 0,54 bij 0,7, 0,44 bij 0,75, 0,20 bij 0,8 en 0,07 bij 0,85, en vanaf 0,9 past geen enkele box strak genoeg om zijn eigen vorm: 0. De grootste IoU-drempel met een mAP van minstens 0,5 is dus 0,7. De cirkels zakken als eerste. Twee van hun true positives hebben een IoU van maar 0,57 met hun cirkel, dus bij 0,6 worden het false positives, en zakt de AP voor cirkels van 0,56 naar 0,38; bij 0,7 is hij 0,21. De AP van COCO, het gemiddelde van de tien waarden, is hier 0,39, zonder interpolatie. Daarom is de AP van COCO voor een detector lager dan zijn mAP bij 0,5: hij beloont ook boxes die strak passen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo beoordeel je een detector met mAP:

    1. Neem voor één soort zijn boxes uit elk beeld, de hoogste score eerst.
    2. Loop de lijst af. Een box is een true positive als zijn IoU met een echte box van zijn soort in hetzelfde beeld boven de drempel ligt, en geen box ervoor die echte box vond; anders is hij een false positive.
    3. Na elke box is precision de true positives tot dan toe gedeeld door de boxes tot dan toe, en recall de true positives tot dan toe gedeeld door de echte boxes van die soort.
    4. AP is de precision bij elke true positive, opgeteld en gedeeld door het aantal echte boxes van die soort.
    5. Doe dit voor elke soort. De mAP is het gemiddelde van hun AP's.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op 8 scènes, `scene(36)` tot en met `scene(43)`. `detections(scenes, seed=17)` geeft de detectorboxes van de course ervoor: voor elke box het nummer van zijn scène in de lijst, zijn soort, zijn score en de box, de hoogste score eerst. De eerste cel print de boxes voor cirkels.

    De tweede cel bepaalt voor elke box voor een cirkel of hij een true positive is. `iou` is de functie van oefening 2 van [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl). Voor elke box zoekt de binnenste loop een cirkel in dezelfde scène waarvan de IoU ermee boven 0,5 ligt: `truth` wordt die echte box, met het nummer van zijn scène, of blijft `None`. Een box is een false positive als hij zo'n echte box niet vindt, of als zijn echte box al in `taken` staat, gevonden door een box met een hogere score. Anders is hij een true positive, en gaat zijn echte box in `taken`.

    De derde cel rekent precision, recall en AP uit met NumPy. `hits_array` is de lijst `hits` als NumPy-array. `np.cumsum(hits_array)` telt de true positives onder de eerste boxes, want `True` telt als 1. Gedeeld door 1, 2, 3 enzovoort, de boxes tot dan toe, geeft dat de precision na elke box; gedeeld door het aantal echte cirkels de recall. `precision[hits_array]` houdt de precision bij elke true positive over.
    """)
    return


@app.cell
def example_boxes():
    from everything_ai.scenes import scene, detections

    scenes = [scene(k) for k in range(36, 44)]
    # elke box: (i, soort, score, box), met i het nummer van zijn scène;
    # de hoogste score eerst
    boxes = detections(scenes, seed=17)
    print(len(boxes), "boxes")
    kind = "circle"
    for d in boxes:
        if d[1] == kind:
            print(d)
    return boxes, kind, scenes


@app.cell
def example_hits(boxes, kind, scenes):
    def iou(a, b):
        # de intersection over union van twee boxes, zoals in cv-iou
        top, left = max(a[0], b[0]), max(a[1], b[1])
        bottom, right = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0, bottom - top + 1) * max(0, right - left + 1)
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return overlap / (area_a + area_b - overlap)


    taken = set()     # de echte boxes die al gevonden zijn: (i, box)
    # 2. voor elke box van deze soort: True bij een true positive
    hits = []
    for i, k, score, box in boxes:      # 1. de hoogste score eerst
        if k == kind:
            truth = None
            for shape in scenes[i].shapes:
                if shape.kind == kind:
                    if iou(box, shape.box) > 0.5:
                        truth = (i, shape.box)
            if truth == None:
                hits.append(False)          # geen echte box van deze soort
            else:
                if truth in taken:
                    hits.append(False)      # zijn echte box is al gevonden
                else:
                    taken.add(truth)
                    hits.append(True)
    print(hits)
    return (hits,)


@app.cell
def example_ap(hits, kind, scenes):
    import numpy as np

    total = 0         # de echte boxes van deze soort
    # _shape, omdat de cel hierboven al een naam shape heeft
    for s in scenes:
        for _shape in s.shapes:
            if _shape.kind == kind:
                total = total + 1
    hits_array = np.array(hits)
    # 3. precision en recall na elke box
    precision = np.cumsum(hits_array) / np.arange(1, len(hits) + 1)
    recall = np.cumsum(hits_array) / total
    # .round(2): elk getal van de array op twee decimalen
    print(precision.round(2))
    print(recall.round(2))
    # 4. de precision bij elke true positive, opgeteld, gedeeld door total
    ap = np.sum(precision[hits_array]) / total
    print(total, "echte boxes, AP", ap.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste cel print 11 boxes voor cirkels. De zekerste, met 0,86, ligt in scène 3 op een vierkant: zijn IoU met het vierkant is 0,84, maar een vierkant is geen cirkel. De tweede cel laat zien welke boxes true positives zijn: de eerste is fout, de volgende 8 kloppen, en de laatste twee zijn fout. De box met 0,33 ligt vooral op de achtergrond, en de box met 0,30 past niet goed genoeg om zijn cirkel. Van de 11 cirkels worden er dus 8 gevonden. Geen van deze boxes is een tweede box op een cirkel die al gevonden is; zo'n box zou ook een false positive zijn, en NMS gooit de meeste ervan weg.

    De derde cel laat zien hoe de precision na de foute eerste box klimt: 0, dan 0,5, 0,67 enzovoort tot 0,89 na de negende box, en daarna omlaag tot 0,73 aan het eind. De recall klimt in stappen van 1/11, want er zijn 11 cirkels, tot 0,73. AP telt de precision bij de 8 true positives op, 0,5 + 0,67 + ... + 0,89, en deelt door de 11 cirkels: 0,56. Dat de detector ook 11 boxes voor cirkels tekent, is toeval: AP deelt door de echte boxes, niet door de boxes. Verander `"circle"` in de eerste cel in `"square"` of `"triangle"`, en de cellen geven 0,78 en 0,73. De mAP is het gemiddelde van de drie: 0,69.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hand():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_hand_cell():
    choice_hand = L.choice(
        "hand",
        question="Een detector geeft 5 boxes voor katten in een test set met 4 katten. Vanaf de hoogste score omlaag zijn het: true positive, false positive, true positive, true positive, false positive. Wat is zijn AP?",
        options={
            "right": "0,60",
            "found": "0,81",
            "recall": "0,75",
            "every": "0,70",
        },
        correct="right",
        why={
            "right": "De precision bij de drie true positives is 1/1, 2/3 en 3/4. Opgeteld is dat 2,42, en gedeeld door de 4 katten 0,60. De vierde kat wordt nooit gevonden en telt als 0; de false positive aan het eind kost niets.",
            "found": "Dat deelt door de 3 true positives. Deel door de 4 katten: de kat die nooit gevonden wordt, telt als 0.",
            "recall": "3 van de 4 katten worden gevonden: dat is de recall aan het eind. AP kijkt ook naar de volgorde: tel de precision bij elke true positive op.",
            "every": "Dat is het gemiddelde van de precision na alle 5 boxes. AP telt alleen de precision bij de true positives op, en deelt door het aantal katten.",
        },
    )
    choice_hand
    return (choice_hand,)


@app.cell(hide_code=True)
def text_exercise_ap():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. AP als functie

        Maak `average_precision(hits, total)` af. Die krijgt een lijst `hits` met voor elke box van één soort, vanaf de hoogste score omlaag, `True` bij een true positive en `False` bij een false positive, en `total`, het aantal echte boxes van die soort. Hij geeft de AP terug. De check roept hem aan met de cirkels van het uitgewerkte voorbeeld:

        ```python
        average_precision(hits, 11)
        ```

        Dat moet 0,56 geven, net als in het uitgewerkte voorbeeld. De check probeert ook de andere soorten, andere IoU-drempels, een lijst zonder true positive, en lijsten waarin niet elke echte box gevonden wordt.
        """),
        L.task("Laat `average_precision(hits, total)` de average precision geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ap():
    ap_code = L.code_box("ap", lines=14)
    ap_code
    return (ap_code,)


@app.cell(hide_code=True)
def check_ap(ap_code):
    result_ap = L.check("ap", ap_code.value)
    result_ap
    return (result_ap,)


@app.cell(hide_code=True)
def hints_ap():
    L.hints(
        "ap",
        [
            "Loop door de lijst met `for k in range(len(hits)):`, en houd twee getallen bij: `found`, de true positives tot dan toe, en `sum_precision`, de opgetelde precision tot dan toe. Allebei beginnen ze bij 0.",
            "Tel bij een true positive, `if hits[k]:`, eerst 1 op bij `found`. De boxes tot dan toe zijn `k + 1`, want `k` begint bij 0, dus de precision is `found / (k + 1)`: tel die op bij `sum_precision`.",
            "Deel na de loop door het aantal echte boxes: `return sum_precision / total`. Niet door `found`: een echte box die nooit gevonden wordt, telt als 0.",
        ],
        why="Voor de cirkels is de precision bij de 8 true positives 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8 en 8/9. Opgeteld is dat 6,17, en gedeeld door de 11 cirkels 0,56.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bikes():
    mo.md(r"""
    ### Oefening 3. Veel auto's, weinig fietsen
    """)
    return


@app.cell(hide_code=True)
def choice_bikes_cell():
    choice_bikes = L.choice(
        "bikes",
        question="Een test set bevat 1.000 auto's en 10 fietsen. Een detector heeft een AP van 0,9 voor auto's en van 0,3 voor fietsen. Wat is zijn mAP?",
        options={
            "right": "0,6",
            "weighted": "0,89",
            "cars": "0,9",
            "lowest": "0,3",
        },
        correct="right",
        why={
            "right": "De mAP is het gemiddelde van de AP's van de soorten: (0,9 + 0,3) / 2 = 0,6. Elke soort telt even zwaar, hoeveel voorwerpen hij ook heeft, dus een detector kan een zwakke soort niet verstoppen achter een veelvoorkomende.",
            "weighted": "Dat weegt elke soort naar zijn aantal voorwerpen. De mAP neemt het gewone gemiddelde over de soorten: elke soort telt één keer.",
            "cars": "Dat is de AP voor alleen auto's. De mAP neemt het gemiddelde over alle soorten, ook fietsen.",
            "lowest": "Dat is de laagste AP, niet het gemiddelde. Wat is het gemiddelde van 0,9 en 0,3?",
        },
    )
    choice_bikes
    return (choice_bikes,)


@app.cell(hide_code=True)
def text_exercise_extra():
    mo.md(r"""
    ### Oefening 4. Extra boxes
    """)
    return


@app.cell(hide_code=True)
def explain_extra_cell():
    explain_extra = L.explain(
        "extra",
        question="Een detector tekent er 50 boxes op de lege achtergrond bij, allemaal met lagere scores dan elke box die hij al tekende. Wat gebeurt er met zijn AP, en waarom? En wat moet je nog kiezen als een app zijn boxes laat zien?",
        model_answer="**Zijn AP blijft hetzelfde.** AP loopt de boxes af vanaf de hoogste score en telt de precision bij elke true positive op. De 50 boxes komen na elke true positive, dus ze veranderen geen van die precisions, en het aantal echte boxes blijft ook hetzelfde. Bij de gok aan het begin veranderde de AP voor cirkels niet zonder de box met 0,30: 0,56 met en zonder. AP beoordeelt de volgorde van de scores, niet één plek om de lijst af te knippen. **Een app heeft nog een drempel op de score nodig**: hij laat alleen de boxes met een hogere score zien. Een lage drempel laat foute boxes zien, een hoge mist voorwerpen, en AP kiest hem niet voor je.",
        points=[
            "je zegt dat de AP hetzelfde blijft, omdat de extra boxes na elke true positive komen en geen precision bij een true positive veranderen",
            "je zegt dat AP de volgorde van de scores beoordeelt, niet één plek om de lijst met boxes af te knippen",
            "je zegt dat een app nog een drempel op de score nodig heeft die bepaalt welke boxes hij laat zien, en die foute boxes afweegt tegen gemiste voorwerpen",
        ],
    )
    explain_extra
    return (explain_extra,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl) hield NMS één box per voorwerp over. Hier worden die boxes vergeleken met de echte boxes. Een box is een true positive als zijn IoU met een echte box van zijn soort boven een drempel ligt, en geen box met een hogere score die echte box eerder vond. Als je de boxes van één soort afloopt vanaf de hoogste score, tekenen precision en recall de precision-recall curve van [Alle thresholds tegelijk: de ROC curve en AUC](/course/open/ml-roc-auc?lang=nl). De oppervlakte eronder, de average precision, telt de precision bij elke true positive op en deelt door het aantal echte boxes; mAP is het gemiddelde over de soorten: in de scènes 0,69 bij de IoU-drempel van VOC, 0,5, en 0,39 als gemiddelde over de tien drempels van COCO, allebei zonder interpolatie. AP beoordeelt de volgorde van de scores, niet waar je de lijst afknipt. In de volgende les, *Een pretrained detector gebruiken*, tekent een echte detector boxes op foto's, en bepaalt een drempel op de score welke je houdt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bikes, choice_hand, explain_extra, result_ap):
    L.wrap_up(choice_hand, result_ap, choice_bikes, explain_extra)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
