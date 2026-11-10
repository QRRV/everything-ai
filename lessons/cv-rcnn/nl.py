# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-rcnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-rcnn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="De head van je one-stage detector geeft elk van de 3 anchors van een vakje 4 logits en 4 offsets: 24 channels. Een head die alleen *voorwerp* van *achtergrond* onderscheidt, voor 9 anchors op elk vakje, is ook een convolutie van 3 bij 3. Hoeveel output channels heeft hij nodig?",
        options={
            "right": "54: 9 anchors keer 2 logits en 4 offsets",
            "logits": "18: 9 anchors keer 2 logits",
            "one": "6: 2 logits en 4 offsets, voor het vakje",
            "kinds": "72: 9 anchors keer 4 logits en 4 offsets",
        },
        correct="right",
        why={
            "right": "Elke anchor krijgt zijn eigen 2 logits en zijn eigen 4 offsets: 9 × 6 = 54. Dat is de head van het region proposal network van Faster R-CNN: 2 × 9 scores en 4 × 9 getallen voor de boxes. Het RPN van vandaag geeft 3 × 6 = 18.",
            "logits": "Zonder offsets blijft elke box zijn anchor. Elke anchor heeft ook zijn 4 offsets nodig: 9 × (2 + 4). Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
            "one": "Dat is één anchor per vakje. Elk van de 9 anchors krijgt zijn eigen logits en zijn eigen offsets: 9 × 6.",
            "kinds": "4 logits zijn voor achtergrond en 3 soorten. Om *voorwerp* van *achtergrond* te onderscheiden, zijn 2 logits genoeg: 9 × (2 + 4). Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_many():
    quick_many = L.choice(
        "quick-many",
        question="Vóór NMS gaf je one-stage detector de scène van zijn uitgewerkte voorbeeld 17 boxes met een score van 0,5 of meer, voor 4 vormen. Waarom zoveel?",
        options={
            "right": "Meerdere anchors rond elke vorm leerden die vorm, elk met zijn eigen offsets",
            "low": "Een score van 0,5 is te laag: bij 0,9 kreeg elke vorm één box",
            "backbone": "De backbone ziet elke vorm op drie groottes, één voor elke grootte van anchor",
        },
        correct="right",
        why={
            "right": "Elke anchor met een IoU van 0,5 of meer met een vorm leerde die vorm: 8 anchors leerden de cirkel. Getraind gaf de detector de cirkel 7 boxes van 0,5 of meer, en NMS hield die met de hoogste score. Vandaag helpt NMS twee keer: één keer voor de proposals, één keer voor de boxes aan het eind.",
            "low": "Vóór NMS had de cirkel 7 boxes, met scores van 0,70 tot 0,99, en 5 ervan boven 0,94: een hogere score liet er nog steeds meerdere over. Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
            "backbone": "De backbone maakt één set feature maps, dezelfde voor elke anchor. De boxes komen van de anchors die elke vorm leerden. Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
        },
    )
    quick_many
    return (quick_many,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je one-stage detector beslist in één keer over elke anchor, vanuit de vakjes eromheen. Een **two-stage detector** kijkt twee keer. Eerst stelt hij gebieden voor waar een voorwerp in kan zitten; dan bekijkt hij elke proposal opnieuw, met de features van het hele gebied, en geeft het een soort en een betere box. Dat is de familie van R-CNN, die jarenlang de beste was in object detection. R-CNN (Girshick en collega's, 2014) nam per foto zo'n 2.000 proposals van *selective search*, een methode zonder leren die pixels van gelijke kleur en textuur samenvoegt tot gebieden; hij sneed elk ervan uit de foto, rekte het op tot 227 bij 227 pixels, en liet er een heel CNN op los. Fast R-CNN (Girshick, 2015) liet het CNN één keer over de hele foto lopen, en sneed elke proposal uit zijn feature maps. Faster R-CNN (Ren en collega's, 2015) liet ook de proposals door het netwerk zelf maken: een tweede, kleine head op dezelfde feature maps. Vandaag bouw je die laatste op de scènes van de course, met de backbone van je one-stage detector, en vergelijk je de twee.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_seconds():
    guess_seconds = L.guess(
        "seconds",
        question="R-CNN liet een heel CNN los op elk van zo'n 2.000 gebieden van een foto. Hoe lang duurde dat per foto, op een GPU van 2014?",
        options={
            "fast": "Zo'n 0,05 seconde",
            "half": "Zo'n halve seconde",
            "right": "Zo'n 13 seconden",
            "minutes": "Zo'n twee minuten",
        },
        reveal="Zo'n 13 seconden: de paper van R-CNN geeft 13 seconden per foto op een GPU, en 53 op een CPU, voor de proposals en hun features, met het CNN dat hij gebruikte. Met het diepere VGG16 had R-CNN 47 seconden nodig, zo mat de paper van Fast R-CNN; Fast R-CNN deed er daarmee 0,32 seconde over, de proposals niet meegeteld, terwijl selective search alleen al zo'n 2 seconden kostte op een CPU. Faster R-CNN, ook met VGG16, had 0,2 seconde nodig voor alles: 5 foto's per seconde op een GPU.",
    )
    guess_seconds
    return (guess_seconds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De eerste fase stelt voor.** Dat is je one-stage detector met twee klassen: *voorwerp* en *achtergrond*. Hij heeft dezelfde backbone, en een head van 3 bij 3 die elk van de 3 anchors van een vakje 2 logits en 4 offsets geeft: 18 channels. Hij leert zoals je detector leerde, met de targets van je `targets`, waarin elke soort klasse 1 is, en dezelfde twee losses. Het RPN van Faster R-CNN legt de grenzen bij 0,7 en 0,3 in plaats van 0,5 en 0,4, en leert per keer van 256 van de anchors van een foto; hier leert het precies zoals je detector leerde. Deze head is het **region proposal network**, kortweg RPN. Faster R-CNN heeft 9 anchors op elk vakje, van 3 groottes en 3 vormen, dus zijn RPN geeft 2 × 9 scores en 4 × 9 getallen voor de boxes.

        **Van 768 anchors naar 32 proposals.** Softmax over zijn 2 logits geeft elke anchor een kans op een voorwerp, en je `boxes_from` maakt zijn box. De 100 anchors met de hoogste kans gaan door NMS bij 0,7, en de eerste 32 boxes die blijven, zijn de **proposals** van de scène. De drempel is 0,7, zoals in Faster R-CNN, niet 0,5: zo blijven er rond elke vorm een paar verschillende proposals over waar de tweede fase zijn boxes van maakt, en de NMS aan het eind haalt de kopieën weg. Faster R-CNN houdt per foto zo'n 2.000 proposals bij het trainen en 300 bij het detecteren.

        **RoI pooling: elke proposal even groot.** Een proposal kan groot of klein zijn, en de tweede fase heeft voor elke proposal evenveel inputs nodig. **RoI pooling** (RoI: region of interest) neemt de vakjes van de feature maps waar de hoeken van de proposal in liggen, rijen `int(top // 8)` tot en met `int(bottom // 8)` en kolommen `int(left // 8)` tot en met `int(right // 8)`, verdeelt die vakjes zo gelijk als het kan in 4 bij 4 bins, en houdt het hoogste getal van elke bin, voor elk van de 64 channels. Elke proposal, van elke grootte, wordt 64 × 4 × 4 = 1.024 getallen. Een bin van n rijen begint bij rij ⌊i · n / 4⌋ en stopt vóór rij ⌈(i + 1) · n / 4⌉, voor i van 0 tot en met 3: bij 6 rijen nemen de bins de rijen 0–1, 1–2, 3–4 en 4–5, en bij 3 rijen de rijen 0, 0–1, 1–2 en 2. Als de rijen niet precies in 4 bins passen, delen buren een rij. `F.adaptive_max_pool2d(region, 4)` doet precies dit. Fast R-CNN poolt elke proposal tot 7 bij 7 op de maps van VGG16.

        **De tweede fase kijkt opnieuw.** De 1.024 getallen van een proposal gaan door een linear layer van 128 getallen, ReLU, en een linear layer naar 8 getallen: 4 logits, voor achtergrond en de 3 soorten, en 4 offsets, nu van de proposal naar zijn box. Een proposal leert zoals een anchor leerde, maar met één drempel: de soort van de echte box die hij het meest overlapt, vanaf een IoU van 0,5, en de offsets naar die box; onder 0,5 achtergrond. Faster R-CNN geeft elke soort zijn eigen 4 offsets; hier is één set voor alle soorten genoeg. Bij het trainen komen de echte boxes bij de proposals, zoals de code van Faster R-CNN doet, zodat de tweede fase vanaf de eerste epoch ook boxes ziet die passen.

        **Eén loss voor beide fasen.** De detector traint op de loss van het RPN plus de loss van de tweede fase, elk een cross-entropy plus een gemiddeld absoluut verschil van de offsets. De proposals tellen als gegeven getallen: het trainen volgt de loss terug door RoI pooling tot in de feature maps, maar niet tot in de hoeken van de proposals. De makers van Faster R-CNN noemen dit *approximate joint training*; het trainen kostte 25 tot 50% minder tijd dan met hun eerste manier, die de twee fasen om de beurt trainde.

        **Detecteren.** Een scène, haar feature maps, het RPN, 32 proposals, RoI pooling, de tweede fase. De score van een proposal is zijn hoogste kans van een soort, zijn box komt uit `boxes_from` met de proposal als anchor, en NMS voor elke soort bij 0,5 houdt de beste boxes, zoals in je one-stage detector.
        """),
        L.watch_out("RoI pooling rondt af. Een proposal van rij 41,4 tot 67,7 neemt de vakjes 5 tot en met 8, rijen 40 tot en met 71 van de scène. Elke proposal waarvan de hoeken in dezelfde vakjes liggen, krijgt dezelfde 1.024 getallen, dus de tweede fase kan niet zien hoe ver in een vakje een proposal begint. In een latere les vervangt Mask R-CNN RoI pooling door RoIAlign, die niet afrondt."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **two-stage detector** | een detector die eerst gebieden voorstelt waar een voorwerp in kan zitten, en dan elk voorstel opnieuw bekijkt om het een soort en een betere box te geven. R-CNN, Fast R-CNN en Faster R-CNN zijn two-stage detectors | `TwoStage` |
        | **region proposal network** (RPN) | de eerste fase van Faster R-CNN: een head op de feature maps die elke anchor een kans op een voorwerp en 4 offsets geeft; zijn zekerste boxes, na NMS bij 0,7, zijn de proposals | `self.rpn` |
        | **proposal** | een box waarvan de eerste fase denkt dat er een voorwerp in kan zitten; de tweede fase bekijkt hem opnieuw | `proposals` |
        | **RoI pooling** | de features van een region of interest, zoals een proposal, op dezelfde grootte uitsnijden: zijn vakjes in k bij k bins, en het hoogste getal van elke bin | `roi_pool` |
        | **Faster R-CNN** | de two-stage detector van Ren en collega's (2015), waarvan de proposals komen van een RPN op dezelfde feature maps als de tweede fase | |
        | `F.adaptive_max_pool2d(x, k)` | verdeelt elke map van x zo gelijk als het kan in k bij k bins, en houdt het hoogste getal van elke bin: k bij k getallen per map, hoe groot die ook is | |
        | `t.argsort(descending=True)` | de plaatsen van de getallen van een tensor, van het hoogste getal naar het laagste; zonder `descending=True` van het laagste | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stages():
    mo.vstack([
        mo.md("**De getrainde two-stage detector van de course op de scène van het uitgewerkte voorbeeld**"),
        mo.Html(C.stages_picture({
            "first": "eerste fase: {n} proposals",
            "second": "tweede fase: {n} boxes",
            "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
            "alt": "De scène van het uitgewerkte voorbeeld twee keer, met haar vier echte boxes, doorgetrokken. Links de 32 dunne gestreepte proposals van de eerste fase, allemaal op of rond de vier vormen, de meeste rond het vierkant. Rechts de 4 boxes van 0,5 of meer die de tweede fase ervan maakt, na NMS, elk met zijn soort en score.",
        })),
        mo.md(r"""
        Links de 32 proposals van de eerste fase. Elk ervan ligt op of rond een vorm: 18 rond het vierkant, dat de cirkel deels bedekt, 6 rond de kleine driehoek, en 4 rond de grote driehoek en 4 rond de cirkel. Alleen de eerste 8 hebben een kans op een voorwerp boven 0,7, en de laatste 0,007, maar ze gaan allemaal door: de tweede fase beslist. Rechts wat die ervan maakt: voor elke proposal een soort, een score en een betere box, en dan NMS voor elke soort. Alle vier de vormen krijgen een box, ook de kleine driehoek, die je one-stage detector in deze scène geen box gaf.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_roi():
    mo.vstack([
        mo.Html(C.roi_picture({
            "title": "RoI pooling van de eerste proposal",
            "alt": "Een deel van de scène van het uitgewerkte voorbeeld, met de vakjes van 8 bij 8 pixels van de feature maps als dunne lijnen. De eerste proposal, op de grote driehoek, is gestreept; de vakjes waar zijn hoeken in liggen, hebben een doorgetrokken kader, verdeeld in 4 bij 4 bins.",
        })),
        mo.md(r"""
        De eerste proposal, die waar het RPN het zekerst van is, ligt op de grote driehoek. Zijn hoeken liggen in 4 bij 4 vakjes, dus hier is elke bin één vakje, en zijn de 1.024 getallen de 64 channels van die 16 vakjes. Het doorgetrokken kader is groter dan de proposal: RoI pooling neemt hele vakjes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoeveel proposals heeft de tweede fase nodig? De slider zet hoeveel proposals de getrainde detector van de course per scène houdt om te detecteren, van 1 tot 64; hij trainde met 32. Links zijn boxes van 0,5 of meer in de scène van het uitgewerkte voorbeeld; rechts zijn mAP op de 200 testscènes bij een IoU van 0,5, doorgetrokken, en van 0,75, gestreept.
    """)
    return


@app.cell(hide_code=True)
def explore_keep_cell():
    def _frame(n):
        n = int(n)
        _html, _stop = C.keep_frame(n, {"title": "{n} proposals: {boxes} boxes", "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
                                        "chart": "mAP op de 200 testscènes", "keep": "proposals per scène",
                                        "at50": "IoU 0,5", "at75": "IoU 0,75",
                                        "chart_alt": "De mAP van de two-stage detector van de course op de 200 testscènes voor elk aantal proposals per scène, bij een IoU van 0,5 en van 0,75: laag bij 1 en 2 proposals; bij 0,5 dicht bij de hoogste vanaf 8, bij 0,75 het hoogst bij 8 en daarna lager.",
                                        "alt": "De scène van het uitgewerkte voorbeeld met haar vier echte boxes en de {boxes} gestreepte boxes van 0,5 of meer van de detector als hij {n} proposals houdt. Ernaast zijn mAP voor elk aantal proposals."})
        return _html, f"{n} proposals per scène: {_stop['covered']} van de {C.TRAINED['shapes']} vormen hebben een proposal met een IoU boven 0,5; mAP {L.number(_stop['map'], 3)} bij een IoU van 0,5 en {L.number(_stop['map75'], 3)} bij 0,75."

    explore_keep = L.explore("keep-slider", "proposals per scène", _frame, start=32, values=list(C.KEEPS), role="hyper",
                             task="Vanaf hoeveel proposals heeft bijna elke vorm er een? En wat doet de mAP bij 0,75 daarna?")
    explore_keep
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Met 1 proposal per scène kan de detector hoogstens één vorm per scène vinden: 203 van de 808 vormen hebben een proposal met een IoU boven 0,5, een paar meer dan de 200 scènes, omdat in een paar scènes één proposal twee vormen bedekt die op elkaar liggen; de mAP is 0,241. Met 4 hebben 717 vormen er een; met 8 zijn het er 787; met 32 zijn het er 805. De mAP bij 0,5 verandert na 16 proposals nauwelijks. Bij 0,75 is hij het hoogst bij 8, 0,824, en zakt hij naar 0,751 bij 64: meer proposals geven een vorm meer boxes, en proposals waarvan de hoeken in dezelfde vakjes liggen, krijgen dezelfde score en dezelfde offsets, waar ze in die vakjes ook beginnen, dus NMS houdt soms een box die minder goed past. De course trainde de detector opnieuw met RoIAlign in plaats van RoI pooling, met seed 0: dan blijft de mAP bij 0,75 vanaf 8 proposals tussen 0,875 en 0,892.
        """),
        L.watch_out("Kies het aantal proposals niet uit deze grafiek: hij meet de testscènes, en een aantal dat je daarop kiest, laat hun mAP meer beloven dan nieuwe scènes zouden geven. Kies het op een validation set, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). Deze les houdt er 32 om te detecteren, evenveel als bij het trainen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Een two-stage detector, zoals Faster R-CNN:

        1. De backbone maakt feature maps van de scène, en het RPN, een convolutie van 3 bij 3, geeft elke anchor 2 logits en 4 offsets.
        2. Proposals: de kans op een voorwerp van elke anchor, zijn box uit zijn offsets, de 100 zekerste anchors, NMS bij 0,7, en de eerste 32 boxes die blijven.
        3. RoI pooling: voor elke proposal de vakjes waar zijn hoeken in liggen, verdeeld in 4 bij 4 bins, en het hoogste getal van elke bin, voor elk channel.
        4. De tweede fase: de 1.024 getallen van een proposal op één rij, een linear layer, ReLU, en een linear layer naar 4 logits en 4 offsets vanaf de proposal.
        5. Train beide tegelijk: het RPN leert als een one-stage detector met 2 klassen; elke proposal, en elke echte box, leert de soort + 1 van de echte box die hij het meest overlapt vanaf een IoU van 0,5, anders achtergrond, en de offsets naar die box. De loss is de som van de twee losses; de proposals tellen als gegeven getallen.
        6. Detecteer: de score van elke proposal, zijn box uit `boxes_from` met de proposal als anchor, en NMS voor elke soort bij 0,5.
        """),
        L.watch_out("De offsets van de tweede fase beginnen bij de proposal, niet bij een anchor. Een proposal die al past, heeft offsets bij 0 nodig; een die te groot is, negatieve voor zijn hoogte en breedte. `boxes_from(proposals, offsets)` maakt er boxes van."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld doet stap 1 tot en met 4 op de scène van het uitgewerkte voorbeeld van de vorige twee lessen, `scene(18)`, met een netwerk dat nog niet getraind is. De backbone is die van je one-stage detector. In stap 2 is de kans op een voorwerp softmax over de 2 logits van elke anchor, uitgeschreven zoals in de vorige les; de boxes zijn je `boxes_from` van [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl), ook uitgeschreven; en `nms` uit `everything_ai.scenes` geeft wat je `nms` van [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl) gaf, de plaatsen van de boxes die hij houdt, de hoogste score eerst, en snel voor een paar honderd boxes. `argsort(descending=True)` geeft de plaatsen van de anchors van de hoogste kans naar de laagste, en `[:100]` houdt de eerste 100. Stap 3 poolt de echte box van het vierkant, `s.shapes[2]`, met de hand: `math.floor` rondt naar beneden af en `math.ceil` naar boven, en `.max(dim=1)` geeft het hoogste getal van elke rij van `region[:, a:b, c:d].reshape(64, -1)`, de vakjes van één bin voor elk channel, en waar het ligt. Daarna doet `F.adaptive_max_pool2d` hetzelfde in één stap. Voer de cel uit zoals hij is.
    """)
    return


@app.cell
def example_two_stages():
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, nms, scene

    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    anchors = torch.tensor(anchor_boxes()).float()

    # 1. de backbone van je one-stage detector, en het RPN: een convolutie van 3 bij 3 die
    #    elk van de 3 anchors van een vakje 2 logits (achtergrond, voorwerp) en 4 offsets geeft
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    rpn = nn.Conv2d(64, 3 * 6, 3, padding=1)
    second = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 8))
    with torch.no_grad():
        maps = backbone(x)[0]   # de 64 feature maps van de scène
        rows = rpn(maps.unsqueeze(0)).permute(0, 2, 3, 1).reshape(768, 6)
        print(maps.shape, rows.shape)

        # 2. de proposals: de kans op een voorwerp, de boxes, de 100 zekerste anchors,
        #    NMS bij 0,7, en de eerste 32 boxes die blijven
        e = torch.exp(rows[:, :2])
        score = e[:, 1] / e.sum(dim=1)
        middle_a, size_a = (anchors[:, :2] + anchors[:, 2:]) / 2, anchors[:, 2:] - anchors[:, :2] + 1
        middle, size = middle_a + rows[:, 2:4] * size_a, size_a * torch.exp(rows[:, 4:])
        boxes = torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2], dim=1)   # je boxes_from
        best = score.argsort(descending=True)[:100]
        kept = nms(boxes[best], score[best], 0.7)
        props = boxes[best][kept[:32]]
        print(score.max(), score.min(), len(kept), props.shape)
        print(props[0])

        # 3. RoI pooling van één box, die van het vierkant: de vakjes waar zijn hoeken in liggen,
        #    verdeeld in 4 bij 4 bins, en het hoogste getal van elke bin, voor elk channel
        top, left, bottom, right = s.shapes[2].box
        r0, c0, r1, c1 = int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)
        region = maps[:, r0:r1 + 1, c0:c1 + 1]
        h, w = region.shape[1], region.shape[2]
        pooled = torch.zeros(64, 4, 4)
        for i in range(4):
            for j in range(4):
                a, b = math.floor(i * h / 4), math.ceil((i + 1) * h / 4)
                c, d = math.floor(j * w / 4), math.ceil((j + 1) * w / 4)
                highest, _ = region[:, a:b, c:d].reshape(64, -1).max(dim=1)
                pooled[:, i, j] = highest
        print((top, left, bottom, right), (r0, c0, r1, c1), region.shape)
        print(pooled[0])
        print((pooled - F.adaptive_max_pool2d(region, 4)).abs().max())

        # 4. de tweede fase: de 64 x 4 x 4 getallen op één rij, en dan 8 getallen: 4 logits
        #    (achtergrond, cirkel, vierkant, driehoek) en 4 offsets vanaf de box
        out = second(pooled.reshape(1, -1))
        print(out.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De feature maps hebben shape (64, 16, 16), en het RPN geeft 768 rijen van 6. Ongetraind lopen de kansen op een voorwerp van 0,0932 tot 0,8329. NMS bij 0,7 houdt 95 van de 100 zekerste boxes: twee anchors overlappen nooit met een IoU boven 0,6, dus het zijn de willekeurige offsets van het ongetrainde RPN die 5 boxes meer dan 0,7 lieten overlappen met een zekerdere box. De eerste proposal is ongeveer (60,0; 43,4; 88,8; 84,2).

    De box van het vierkant, (42, 39, 73, 72), ligt in de vakjes 5 tot en met 9 naar beneden en 4 tot en met 9 opzij: een gebied van 5 bij 6 vakjes, shape (64, 5, 6). Zijn 4 bins van rijen nemen de rijen 0–1, 1–2, 2–3 en 3–4 van het gebied, en zijn 4 bins van kolommen de kolommen 0–1, 1–2, 3–4 en 4–5. `pooled[0]` bevat de 16 hoogste getallen van channel 0, de eerste rij (1,2122; 1,2122; 1,1315; 1,1315): de eerste twee bins delen kolom 1 van het gebied, en daar ligt het hoogste getal van allebei; de laatste twee delen kolom 4. En `F.adaptive_max_pool2d` geeft precies hetzelfde: het grootste verschil is 0. De tweede fase geeft 1 rij van 8 getallen voor de box.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fast():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat Fast R-CNN snel maakte
    """)
    return


@app.cell(hide_code=True)
def choice_fast_cell():
    choice_fast = L.choice(
        "fast",
        question="Fast R-CNN vond de voorwerpen van een foto 146 keer sneller dan R-CNN, de proposals niet meegeteld, met hetzelfde diepe CNN, VGG16, en dezelfde proposals van selective search. Wat maakte het verschil?",
        options={
            "right": "Hij liet het CNN één keer over de hele foto lopen, en sneed elke proposal uit zijn feature maps",
            "fewer": "Hij bekeek veel minder proposals per foto",
            "small": "Hij verkleinde elke proposal tot 7 bij 7 pixels vóór het CNN",
            "rpn": "Hij maakte de proposals met een region proposal network",
        },
        correct="right",
        why={
            "right": "R-CNN liet het CNN zo'n 2.000 keer per foto lopen, één keer voor elke proposal. Fast R-CNN laat het één keer lopen, en RoI pooling snijdt elke proposal uit de feature maps tot 7 bij 7 bins. Je detector doet hetzelfde met 4 bij 4.",
            "fewer": "Fast R-CNN nam nog steeds zo'n 2.000 proposals per foto van selective search. Wat veranderde, is hoe vaak het CNN loopt.",
            "small": "De 7 bij 7 is de grootte die RoI pooling elke proposal op de feature maps geeft, na het CNN, geen pixels ervoor. Het CNN liep één keer, over de hele foto.",
            "rpn": "Dat kwam later, met Faster R-CNN. Fast R-CNN nam zijn proposals nog van selective search, zo'n 2 seconden per foto op een CPU.",
        },
    )
    choice_fast
    return (choice_fast,)


@app.cell(hide_code=True)
def text_exercise_roi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. RoI pooling

        Maak `roi_pool(maps, box, k)` af, stap 3 van het recept: `maps` bevat de feature maps van één scène, een tensor van shape (C, H, W), elk vakje 8 bij 8 pixels; `box` is (top, left, bottom, right) in pixels, floats, en kan uit de scène steken. Hij geeft een tensor van shape (C, k, k) terug: het hoogste getal van elk van k bij k bins van de vakjes waar de hoeken van de box in liggen. Het vakje van een pixel is `int(pixel // 8)`; houd elk vakje tussen 0 en het laatste vakje van de maps, `H - 1` naar beneden en `W - 1` opzij, zodat een box die uitsteekt nog vakjes heeft. Verdeel de bins met `F.adaptive_max_pool2d`, of met de hand, zoals in het uitgewerkte voorbeeld: dan traint de demo een paar keer zo langzaam. De check probeert vijf boxes: drie op verzonnen maps van 6 bij 6 vakjes, waarvan één boven en rechts uitsteekt en één binnen één vakje ligt, één op maps van 16 bij 16, en één onder maps van 12 bij 10. Als hij klopt, bewaart hij `roi_pool` in je project, als `work/projects/cnn/roi_pool.py`.
        """),
        L.task("Laat `roi_pool` elke box k bij k getallen per channel geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_roi():
    box_roi_pool = L.code_box("roi-pool", lines=10)
    box_roi_pool
    return (box_roi_pool,)


@app.cell(hide_code=True)
def check_roi(box_roi_pool):
    result_roi = L.check("roi-pool", box_roi_pool.value)
    result_roi
    return (result_roi,)


@app.cell(hide_code=True)
def hints_roi():
    L.hints(
        "roi-pool",
        [
            "Stap 3 van het uitgewerkte voorbeeld, als functie: de vakjes van de hoeken, het gebied van de maps ertussen, beide einden erbij, en zijn k bij k bins.",
            "`h, w = maps.shape[1], maps.shape[2]`, en dan elk vakje als `min(max(int(top // 8), 0), h - 1)`: `max` tilt een vakje onder 0 naar 0, en `min` haalt een vakje voorbij de maps terug naar het laatste. Hetzelfde voor `bottom` met `h - 1`, en voor `left` en `right` met `w - 1`.",
            "`return F.adaptive_max_pool2d(maps[:, r0:r1 + ___, c0:c1 + ___], k)`.",
        ],
        why="Voor de eerste box, `(5.5, 9.0, 30.2, 44.0)`: rijen `5.5 // 8` = 0 tot en met `30.2 // 8` = 3, en kolommen 1 tot en met 5, een gebied van 4 bij 5 vakjes. 2 bij 2 bins nemen de rijen 0–1 en 2–3, en de kolommen 0–2 en 2–4 ervan: de middelste kolom zit in allebei. Voor de box die uitsteekt, `(-6.0, 30.0, 20.0, 60.0)`: `-6.0 // 8` is −1, en zonder `max` zou de slice `-1:3` leeg zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proposals():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De proposals

        Maak `proposals(out, anchors, n)` af, stap 2 van het recept: `out` bevat de rijen van het RPN voor één scène, een tensor van shape (A, 6), voor elke anchor 2 logits, achtergrond en voorwerp, en 4 offsets; `anchors` heeft shape (A, 4). Hij geeft de boxes van de eerste `n` proposals terug, een tensor van hoogstens `n` rijen van 4, minder als NMS er minder houdt, de zekerste eerst: de kans op een voorwerp van elke anchor, zijn box uit `boxes_from` van je project, de 100 anchors met de hoogste kans, `nms` op hun boxes bij 0,7, en de boxes van de eerste `n` die hij houdt. `nms` geeft plaatsen in de boxes die je hem geeft. De check probeert zes verzonnen anchors, waarvan NMS er vijf houdt, en de 768 anchors van een scène met willekeurige rijen, zoals een RPN dat nog niets geleerd heeft. Als hij klopt, bewaart hij `proposals` in je project, als `work/projects/cnn/proposals.py`.
        """),
        L.task("Laat `proposals` de boxes van de eerste n proposals geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_proposals():
    box_proposals = L.code_box("proposals", lines=9)
    box_proposals
    return (box_proposals,)


@app.cell(hide_code=True)
def check_proposals(box_proposals):
    result_proposals = L.check("proposals", box_proposals.value)
    result_proposals
    return (result_proposals,)


@app.cell(hide_code=True)
def hints_proposals():
    L.hints(
        "proposals",
        [
            "Stap 2 van het uitgewerkte voorbeeld, als functie, met `boxes_from(anchors, out[:, 2:])` voor de uitgeschreven boxes en `n` voor 32.",
            "`e = torch.exp(out[:, :2])` en `score = e[:, 1] / e.sum(dim=1)`; dan `best = score.argsort(descending=True)[:100]`.",
            "`kept = nms(boxes[best], score[best], 0.7)[:n]`: plaatsen in `boxes[best]`. Dus `return boxes[___][kept]`.",
        ],
        why="Van de zes anchors is anchor 1 de zekerste, 0,953, en anchor 0, 0,881, overlapt hem met een IoU van 0,78: NMS laat hem vallen. Anchors 4 en 5 overlappen met een IoU van 0,6, onder 0,7: ze blijven allebei. Dus met een `n` van 6 krijg je 5 boxes: anchors 1, 3, 2, 4 en 5, anchor 3 door zijn offset 0,25 × 16 = 4 pixels omlaag geschoven.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learns():
    mo.md(r"""
    ### Oefening 4. Wat een proposal leert
    """)
    return


@app.cell(hide_code=True)
def choice_learns_cell():
    choice_learns = L.choice(
        "learns",
        question="Bij het trainen overlapt een proposal de cirkel met een IoU van 0,62 en het vierkant met 0,30. Wat leert de tweede fase ervan?",
        options={
            "right": "Klasse 1, cirkel, en de offsets van de proposal naar de box van de cirkel",
            "anchor": "Klasse 1, cirkel, en de offsets van de anchor waar hij vandaan kwam naar de box van de cirkel",
            "background": "Achtergrond: het RPN vond de cirkel al, en alleen de echte boxes leren een soort",
            "ignored": "Niets: hij ligt deels op twee vormen, dus hij wordt weggelaten",
        },
        correct="right",
        why={
            "right": "Zijn beste IoU, 0,62, is 0,5 of meer, met de cirkel: klasse 0 + 1 = 1, en de offsets die de proposal tot de box van de cirkel maken. De tweede fase begint bij de proposal, dus die verbeteren zijn offsets.",
            "anchor": "De offsets van de tweede fase beginnen bij de proposal: bij het detecteren maakt `boxes_from(proposals, offsets)` de boxes. De anchor deed zijn werk al in de eerste fase.",
            "background": "Elke proposal met een IoU van 0,5 of meer met een echte box leert de soort van die box; de echte boxes komen er alleen bij. Juist zo'n proposal van 0,62 moet de tweede fase leren verbeteren.",
            "ignored": "De tweede fase negeert niets: vanaf 0,5 een soort, onder 0,5 achtergrond. Zijn beste IoU, 0,62, is met de cirkel.",
        },
    )
    choice_learns
    return (choice_learns,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint de two-stage detector van de course, de backbone van je one-stage detector, een RPN en een tweede fase, met je `proposals` en je `roi_pool`, 8 epochs lang op de 2.000 trainingsscènes, in mini-batches van 32, zoals je one-stage detector trainde. Daarna vindt hij er de vormen van de 200 testscènes mee, met je `boxes_from`. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt een paar minuten. De grafiek toont na elke epoch de training loss van beide fasen samen. Als hij klaar is, kun je terug door de epochs, en de boxes zien die de detector na elk ervan aan drie testscènes gaf.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train de two-stage detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_two_stages(train_button, result_roi, result_proposals):
    result_roi, result_proposals  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om de two-stage detector te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Twee fasen: de training loss", "x": "epoch", "y": "training loss",
                                        "alt": "De training loss van de two-stage detector, beide fasen samen, na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "rpn_loss" in _p:
                _loss = _p["rpn_loss"] + _p["second_loss"]
                _now["chart"].add(_p["epoch"], _loss, f"**Twee fasen** · epoch {_p['epoch']} van {_p['epochs']} · training loss {L.number(_loss, 3)}")
            else:
                _now["chart"].line(f"**Twee fasen** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `roi_pool` en je `proposals` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan een kwartier. Doet je `roi_pool` of je `proposals` meer werk dan nodig, zoals een loop over de anchors of over de bins? `F.adaptive_max_pool2d` verdeelt alle bins in één stap.*",
                "roi_pool": "*Je `roi_pool` in `work/projects/cnn/roi_pool.py` stopt met een fout, of geeft geen 64 × 4 × 4 getallen die het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "proposals": "*Je `proposals` in `work/projects/cnn/proposals.py` stopt met een fout, of geeft geen 1 tot 32 boxes voor een scène. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van de detector. Heb je het bestand veranderd? Verwijder het, en check zijn oefening in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course nam de plaats in van " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss van elke fase", "x": "epoch", "y": "loss", "rpn": "eerste fase", "second": "tweede fase",
                       "alt": "De training loss van elke fase na elke epoch: van de eerste fase doorgetrokken, van de tweede fase gestreept."}
            _words = {"scene": "scène {seed}: {n} boxes", "alt": "Drie testscènes, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die de detector haar na die epoch gaf, na NMS."}
            _curves, _moments = _result["curves"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.curves_picture(_curves, _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"Na epoch {_e}: loss {L.number(_curves['rpn'][_e - 1], 3)} voor de eerste fase en {L.number(_curves['second'][_e - 1], 3)} voor de tweede, en {_n} boxes van 0,5 of meer in deze drie scènes, die {C.SHOWN_SHAPES} vormen hebben.")

            if len(_moments) == len(_curves["rpn"]) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_moments), values=list(range(1, len(_moments) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: na welke epoch zitten de boxes op de vormen?")]
            else:
                _pic = [mo.Html(C.curves_picture(_curves, _labels, L.lang))] if _curves["rpn"] else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("cirkels", "vierkanten", "driehoeken"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""De two-stage detector, met {C.whole(_result['params'], L.lang)} parameters, vindt {_result['found']} van de {_result['shapes']} vormen van de 200 testscènes: een box van hun soort, met een score van 0,5 of meer en een IoU boven 0,5. {_result['false']} van zijn boxes van 0,5 of meer vinden geen vorm, en {_result['covered']} vormen hebben een proposal met een IoU boven 0,5. Zijn mAP bij een IoU van 0,5 is {L.number(_result['map'], 3)}, met een AP van {_aps}; bij 0,75 is hij {L.number(_result['map75'], 3)}. De two-stage detector van de course, op dezelfde manier getraind met de seeds 0 tot 4, vindt 761 tot 776 vormen, met een mAP van 0,947 tot 0,962, en 0,734 tot 0,779 bij 0,75. De one-stage detector van de course, op dezelfde manier getraind met de seeds 0 tot 4, vond 725 tot 749 vormen, met een mAP van 0,906 tot 0,918, en 0,807 tot 0,822 bij 0,75.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_second():
    mo.md(r"""
    ### Oefening 5. Wat de tweede fase ziet
    """)
    return


@app.cell(hide_code=True)
def explain_second_cell():
    explain_second = L.explain(
        "why-second",
        question="Je one-stage detector en de two-stage detector hebben dezelfde backbone. Op de scènes van de course vindt de two-stage detector meer vormen, maar bij een IoU van 0,75 passen zijn boxes minder goed. Wat ziet zijn tweede fase van een vorm dat de head van je one-stage detector niet ziet? En wat in RoI pooling kan zijn boxes minder goed laten passen?",
        model_answer="De head van de one-stage detector is één convolutie van 3 bij 3: hij beslist over elke anchor vanuit hetzelfde venster van 3 bij 3 vakjes rond het vakje van de anchor, op een vaste plek en van een vaste grootte, welke vorm er ook ligt. De tweede fase krijgt de features van de proposal zelf, verdeeld in 4 bij 4 bins, hoe groot die ook is, en heeft zijn eigen linear layers om mee te beslissen: hij beoordeelt een proposal waarvan het RPN alleen dacht dat er waarschijnlijk iets in zat, met de features van precies dat gebied. Dat kost een tweede ronde voor elke proposal en veel meer parameters, 167.386 tegen 38.616. RoI pooling rondt een proposal af op hele vakjes van 8 pixels: proposals waarvan de hoeken in dezelfde vakjes liggen, krijgen dezelfde getallen, dus de tweede fase kan niet zien hoe ver in een vakje een box begint, en zijn offsets kunnen de randen niet zo precies leggen. RoIAlign, die niet afrondt, liet de boxes bij 0,75 veel beter passen.",
        points=[
            "je zegt dat de head van de one-stage detector over een anchor beslist vanuit vaste 3 bij 3 vakjes eromheen, terwijl de tweede fase de features van de hele proposal krijgt, op dezelfde grootte gepoold, met eigen layers",
            "je zegt dat dit iets kost: een tweede ronde voor elke proposal, en meer parameters",
            "je zegt dat RoI pooling een proposal afrondt op hele vakjes, zodat proposals die een paar pixels verschillen dezelfde getallen krijgen, en de tweede fase de randen van een box niet precies kan leggen",
        ],
    )
    explain_second
    return (explain_second,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De anchors, targets en loss van [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) zijn nu de eerste fase: een RPN dat *voorwerp* van *achtergrond* onderscheidt en boxes voorstelt. Je `proposals` houdt de zekerste ervan, met NMS bij 0,7; je `roi_pool` snijdt elk ervan uit de feature maps tot 4 bij 4 bins; en een tweede fase geeft elk een soort en offsets vanaf de proposal, gevolgd door de NMS van [Eén box per voorwerp: non-maximum suppression](/course/open/cv-nms?lang=nl) bij 0,5. Dat is de weg van R-CNN, dat een CNN op elke proposal losliet, via Fast R-CNN, dat het één keer liet lopen, naar Faster R-CNN, waarvan de proposals uit het netwerk zelf komen. Op de scènes van de course vindt de two-stage detector meer vormen dan je one-stage detector, maar RoI pooling rondt af, en bij een IoU van 0,75 passen zijn boxes minder goed. Je project `cnn` heeft nu `roi_pool.py` en `proposals.py`. In de volgende les, *Een feature pyramid voor kleine en grote voorwerpen*, krijgt een detector feature maps van verschillende groottes, zodat hij ook kleine voorwerpen vindt. Later laat focal loss RetinaNet, een one-stage detector, de two-stage detectors bijhouden, en geeft Mask R-CNN elke box een mask, met RoIAlign in plaats van RoI pooling.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fast, result_roi, result_proposals, choice_learns, explain_second):
    L.wrap_up(choice_fast, result_roi, result_proposals, choice_learns, explain_second)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
