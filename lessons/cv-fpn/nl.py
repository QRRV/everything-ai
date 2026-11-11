# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-fpn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fpn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) en [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lateral():
    quick_lateral = L.choice(
        "quick-lateral",
        question="Een 1×1 convolution van 128 channels naar 32, `nn.Conv2d(128, 32, 1)`. Hoeveel parameters heeft die?",
        options={
            "right": "4.128: 32 filters, elk met 128 weights en een bias",
            "bias": "4.096: 128 × 32 weights",
            "three": "36.896: 32 filters, elk met 9 × 128 weights en een bias",
            "sum": "160: 128 + 32",
        },
        correct="right",
        why={
            "right": "Elk van de 32 filters heeft een weight voor elk channel dat binnenkomt, en een bias: 32 × (128 + 1) = 4.128. Vandaag geeft zo'n 1×1 convolution elke map van de backbone dezelfde 32 channels.",
            "bias": "De weights kloppen, maar elk filter heeft ook een bias: 32 × (128 + 1). Zie [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl).",
            "three": "Dat is een 3×3-convolutie, met 9 weights per channel. Een 1×1 convolution heeft één weight per channel dat binnenkomt: 32 × (128 + 1). Zie [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl).",
            "sum": "Elk filter ziet alle 128 channels: 128 weights en een bias voor elk van de 32 filters, 32 × (128 + 1).",
        },
    )
    quick_lateral
    return (quick_lateral,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="De head van je one-stage detector gaf een scène maps met shape (1, 24, 16, 16): 3 anchors keer 8 getallen, op 16 bij 16 vakjes. Een head met 2 anchors per vakje, op maps van 32 bij 32 vakjes, geeft (1, 16, 32, 32). Welke shape krijgen zijn rijen met `permute(0, 2, 3, 1)` en `reshape(1, -1, 8)`?",
        options={
            "right": "(1, 2048, 8): 32 × 32 vakjes, elk 2 anchors",
            "squares": "(1, 1024, 8): één rij voor elk vakje",
            "sixteen": "(1, 1024, 16): de 16 getallen van elk vakje in één rij",
        },
        correct="right",
        why={
            "right": "`permute` zet de 16 getallen van elk vakje bij elkaar, en `reshape(1, -1, 8)` knipt ze in rijen van 8: 32 × 32 × 2 = 2.048 anchors. Vandaag geeft elk niveau zijn eigen rijen, en die komen achter elkaar.",
            "squares": "Een vakje heeft 16 getallen, de 8 van elk van zijn 2 anchors: rijen van 8 geven 2 rijen per vakje, 32 × 32 × 2. Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
            "sixteen": "`reshape(1, -1, 8)` maakt rijen van 8, één voor elke anchor: 32 × 32 × 2 = 2.048. Zie [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je one-stage detector zette al zijn anchors op één feature map, en zijn scènes hadden vormen van ongeveer één grootte; in een foto kan een auto het hele beeld vullen, of een paar pixels groot zijn in de verte. Het oude antwoord is een **beeldpiramide**: dezelfde foto in verschillende groottes, elk kleiner dan de vorige, met de detector op elk ervan, en volgens de paper van deze les duurt het vinden van de voorwerpen in een foto daardoor zo'n vier keer zo lang. Een CNN maakt op zijn weg omhoog al maps van verschillende groottes, maar de vroege, die scherp zijn, weten weinig van wat ze zien, en de diepe, die weten wat er is, zijn grof; SSD (2016), een one-stage detector, zette anchors op een aantal daarvan, maar pas vanaf hoog in het netwerk, en miste zo de scherpe maps die kleine voorwerpen nodig hebben. Een **feature pyramid network** (Lin en collega's, 2017) geeft de scherpe maps wat de diepe weten, met een weg terug naar beneden: vandaag bouw je er een, op scènes met kleine en grote vormen.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_small():
    guess_small = L.guess(
        "small",
        question="De course trainde een one-stage detector zoals de jouwe, met één deel meer in zijn backbone, op scènes met kleine en grote vormen: al zijn anchors op één map van 8 bij 8 vakjes van 16 pixels, 8 groottes van 8 tot 91 pixels op elk vakje. De 200 testscènes hebben 268 kleine vormen, kleiner dan 16 bij 16 pixels. Hoeveel daarvan vond hij?",
        options={
            "all": "Zo'n 260: bijna allemaal",
            "most": "Zo'n 200",
            "right": "Zo'n 130: de helft",
            "few": "Zo'n 30",
        },
        reveal="129 tot 143, met de seeds 0 tot 4: ongeveer de helft. Van de 236 grote vormen vond hij er 218 tot 225. Een kleine vorm is kleiner dan één vakje van die map, en de anchors van de vakjes liggen 16 pixels uit elkaar: maar 43 van de 268 kleine vormen hebben een anchor die ze overlapt met een IoU van 0,5 of meer. Het feature pyramid network van deze les vindt er 244 tot 249.",
    )
    guess_small
    return (guess_small,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vormen van elke grootte.** `scene(seed, spread=True)` geeft scènes waarvan de vormen een straal van 5 tot 40 pixels hebben, gelijk verdeeld over de groottes: de kans dat een vorm twee keer zo groot is als een andere, is even groot als de kans dat hij half zo groot is. Elke vorm houdt minstens de helft van zijn pixels in beeld. De les noemt een vorm klein als de wortel uit de oppervlakte van zijn box, $\sqrt{wh}$, onder de 16 pixels ligt, en groot vanaf 32 pixels; daartussen middelgroot. De 200 testscènes hebben 786 vormen, van 7 tot 80 pixels: 268 kleine, 282 middelgrote en 236 grote. COCO, de dataset van de paper, trekt de grenzen bij 32 en 96 pixels, in foto's van zo'n 640 pixels.

        **Vier maps.** De backbone is die van je one-stage detector met twee delen erbij, elk een max pooling en een convolutie van 3 bij 3 met batch normalization en ReLU. De paper noemt de maps aan het eind van zijn vier delen C2 tot C5: C$_k$ heeft vakjes van $2^k$ pixels. C2 heeft 32 channels op 32 bij 32 vakjes van 4 pixels; C3, de map die je detector gebruikte, 64 op 16 bij 16 vakjes van 8 pixels; C4 128 op 8 bij 8 vakjes van 16 pixels; en C5 128 op 4 bij 4 vakjes van 32 pixels. Elk getal van C2 ziet 18 bij 18 pixels van de scène, zijn receptive field; van C3 38, van C4 78, en van C5 158, meer dan de hele scène.

        **Eén map is niet genoeg.** De course trainde twee detectors met alle 8 anchorgroottes op één map. Op C4 liggen de anchors van naburige vakjes 16 pixels uit elkaar: een kleine vorm daartussen overlapt er geen goed, dus alleen zijn beste anchor leert hem: gemiddeld 1,1 anchors per kleine vorm, tegen 4,0 per grote. Op C2 is het andersom: een grote vorm wordt gemiddeld door 63 anchors geleerd, een kleine door 4, en de 8.192 anchors geven zo'n 5 keer zoveel valse boxes als een pyramid. De tabel na de slider geeft hun getallen.

        **Een niveau voor elke grootte.** Een feature pyramid geeft elke schaal een eigen niveau, P2 tot P5, uit C2 tot C5, en elk niveau zijn eigen anchors: op elk van zijn vakjes 2 anchors, 2 en $2\sqrt{2}$ vakjes breed. Dat is 8 en 11,3 pixels op P2, 16 en 22,6 op P3, 32 en 45,3 op P4, en 64 en 90,5 op P5: 2.048 + 512 + 128 + 32 = 2.720 anchors. Elke anchor leert zoals in je one-stage detector, door zijn IoU met de echte boxes, dus een vorm wordt geleerd door anchors van het niveau waarvan de groottes passen: gemiddeld 2,5 tot 2,7 anchors per vorm, klein of groot. De paper zette zijn pyramid in een two-stage detector, Faster R-CNN, waarvan de eerste fase, een *region proposal network*, boxes voorstelt waar een voorwerp in kan zitten, de *proposals*: [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) legt dat uit. Zijn anchors hebben één oppervlakte per niveau, 32² tot 512² pixels op P2 tot P6, waarbij P6 nog een keer half zo groot is als P5, elk in 3 verhoudingen van breedte tot hoogte, 1:2, 1:1 en 2:1: 15 soorten anchors, 3 op elk vakje. Hier gaat de pyramid in je one-stage detector, zoals RetinaNet in de volgende les.

        **De weg naar beneden.** Een **feature pyramid network** maakt de niveaus van boven naar beneden, langs wat de paper het *top-down pathway* noemt. P5 is C5 door een 1×1 convolution naar 32 channels. Daarna, één niveau tegelijk: het niveau erboven twee keer zo groot gemaakt met nearest-neighbor-interpolatie, plus de map van de backbone door zijn eigen 1×1 convolution, de *lateral connection*. `F.interpolate(p, scale_factor=2, mode="nearest")` maakt van elk getal van `p` een blok van 2 bij 2. Als laatste een 3×3-convolutie op elk niveau, die de paper toevoegt "to reduce the aliasing effect of upsampling": de harde overgangen tussen de blokken van 2 bij 2. Zo krijgt het scherpe niveau P2 ook wat C3, C4 en C5 weten. De paper gebruikt 256 channels, en geen ReLU in deze layers; hier 32 channels, en ook geen ReLU.

        **Eén head voor elk niveau.** Eén convolutie van 3 bij 3 geeft elk vakje van elk niveau 2 × 8 getallen: 4 logits en 4 offsets voor elk van zijn 2 anchors. Eén head kan alle vier de niveaus bedienen, omdat ze allemaal 32 channels hebben en de anchors van elk niveau evenveel van zijn vakjes breed zijn: een vorm van 2 vakjes breed ziet er op elk niveau hetzelfde uit. De rijen van de niveaus komen achter elkaar, de fijnste eerst, in de volgorde van de anchors. In de paper gaf een head per niveau ongeveer dezelfde nauwkeurigheid als één gedeelde head; volgens de paper laat dat zien dat na de weg naar beneden elk niveau ongeveer evenveel weet van wat het ziet.

        **Hard negative mining.** In een trainingsscène leren gemiddeld 9,9 van de 2.720 anchors een vorm, 0,4%; bijna alle andere leren achtergrond. Getraind met de loss van je one-stage detector, de cross-entropy over alle anchors, vindt de FPN-detector maar 9 van de 786 vormen van de testscènes met een score van 0,5 of meer: hij leert overal achtergrond te zeggen. **Hard negative mining**, zoals in SSD: in elke scène telt de cross-entropy alle anchors die een vorm leren, en drie keer zoveel achtergrond-anchors, die met de hoogste cross-entropy, die de detector het meest fout doet. De box loss is het gemiddelde absolute verschil van de offsets, zoals voorheen.
        """),
        L.watch_out("De rijen en de anchors moeten in dezelfde volgorde staan: niveau 2 eerst, dan 3, 4 en 5; op elk niveau rij voor rij, vakje voor vakje, en de 2 anchors van een vakje achter elkaar. Rijen in een andere volgorde geven nog steeds 2.720 rijen van 8, en de detector traint nog steeds, maar elke rij leert de targets van een andere anchor."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **feature pyramid network** (FPN) | maps met hetzelfde aantal channels op elke schaal, van boven naar beneden gemaakt uit de maps van de backbone: van elk een 1×1 convolution, het niveau erboven twee keer zo groot gemaakt en erbij opgeteld, en dan een 3×3-convolutie | `pyramid` |
        | **beeldpiramide** | dezelfde afbeelding in verschillende groottes, elk kleiner dan de vorige, vaak half zo groot, om op elke schaal naar patronen te zoeken | |
        | **lateral connection** | in een FPN de 1×1 convolution die een map van de backbone naar het niveau van dezelfde grootte brengt | `lateral[i]` |
        | **hard negative mining** | trainen op de negatieve voorbeelden die het model het meest fout doet: in een detector de achtergrond-anchors met de hoogste loss, in SSD hoogstens drie voor elke anchor die een voorwerp leert | |
        | `F.interpolate(x, scale_factor=2, mode="nearest")` | maakt elke map van x in beide richtingen twee keer zo groot, elk getal een blok van 2 bij 2 | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_pyramid():
    mo.vstack([
        mo.Html(C.pyramid_picture({
            "up": "backbone", "down": "feature pyramid",
            "head": ("één head", "voor elk niveau"),
            "alt": "Links de maps C2 tot C5 van de backbone, van onder naar boven elk half zo groot. Van elke map gaat een pijl met 1×1 naar rechts. Van boven naar beneden gaat elk niveau twee keer zo groot naar een plus met de 1×1-map van het niveau eronder. Van elk plusteken gaat een pijl met 3×3 naar P2 tot P5, van 32 bij 32 tot 4 bij 4 vakjes. Een haak rechts verbindt P2 tot P5: één head voor elk niveau.",
        })),
        mo.md(r"""
        Links gaat de backbone omhoog: elke map half zo groot als die eronder. In het midden de weg naar beneden: vanaf de top elk niveau twee keer zo groot gemaakt, ×2, en opgeteld bij de 1×1-map van het niveau eronder. Rechts, na een 3×3-convolutie, de niveaus P2 tot P5, elk met 32 channels, en één head voor allemaal.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Welk niveau vindt welke vormen? De slider kiest een niveau van de getrainde FPN-detector van de course, seed 0. Links de boxes van 0,5 of meer op de scène van het uitgewerkte voorbeeld die van anchors van dat niveau komen; rechts de kleine, middelgrote en grote vormen van de 200 testscènes die de detector vindt: licht allemaal, donker die van dat niveau komen.
    """)
    return


@app.cell(hide_code=True)
def explore_level_cell():
    def _frame(k):
        k = int(k)
        _html, _stop = C.level_frame(k, {"title": ("niveau {k}: {n} box", "niveau {k}: {n} boxes"), "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
                                         "chart": "gevonden door niveau {k}", "shapes": "vormen", "thirds": ("klein", "middelgroot", "groot"),
                                         "chart_alt": "Van de kleine, middelgrote en grote vormen die de detector op de 200 testscènes vindt, hoeveel er van anchors van niveau {k} komen.",
                                         "alt": "De scène van het uitgewerkte voorbeeld met haar vijf echte boxes en de {n} gestreepte boxes van 0,5 of meer die van anchors van niveau {k} komen. Ernaast hoeveel vormen van elke grootte niveau {k} vindt."})
        _found = _stop["found"]
        return _html, f"Niveau {k}: vakjes van {2 ** k} pixels, anchors van {L.number(_stop['sizes'][0], 1)} en {L.number(_stop['sizes'][1], 1)} pixels, {C.whole(_stop['anchors'], L.lang)} anchors. Op de testscènes vindt het {_found[0]} kleine, {_found[1]} middelgrote en {_found[2]} grote vormen."

    explore_level = L.explore("level-slider", "niveau van de pyramid", _frame, start=2, values=[2, 3, 4, 5], role="",
                              task="Welk niveau vindt de kleine vormen, en welk de grote? Vindt een niveau ooit een vorm van een andere grootte?")
    explore_level
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Elk niveau vindt de vormen van zijn grootte. Van de 248 kleine vormen die de detector vindt, vindt niveau 2 er 229 en niveau 3 de andere 19; van de 223 grote vindt niveau 4 er 183, niveau 5 er 36 en niveau 3 er 4. Niveau 5 vindt geen box op de scène van het uitgewerkte voorbeeld: haar grootste vorm, de driehoek van 41 pixels, wordt op niveau 4 gevonden. De course trainde de detector ook zonder de weg naar beneden, elk niveau alleen gemaakt uit zijn eigen map met zijn eigen 1×1 convolution en 3×3-convolutie, de *bottom-up pyramid* van de paper, die SSD nadoet; en met alle anchors op één map, C4 of C2:
        """),
        mo.md(C.comparison_table({"header": ("Detector", "Anchors", "Klein", "Middelgroot", "Groot", "mAP", "mAP 0,75", "Valse boxes"),
                                  "names": {"fpn": "FPN", "bottom-up": "zonder de weg naar beneden", "deep": "één map, C4", "sharp": "één map, C2"}}, L.lang)),
        mo.md(r"""
        Elke cel geeft het laagste en het hoogste getal over de seeds 0 tot 4: de kleine, middelgrote en grote vormen van de 200 testscènes die hij vindt, van 268, 282 en 236; zijn mAP bij een IoU van 0,5 en van 0,75; en zijn boxes van 0,5 of meer die geen vorm vinden. De twee pyramids vinden veel meer kleine vormen dan C4, en geven veel minder valse boxes dan C2. Zonder de weg naar beneden doet de pyramid het op deze scènes even goed als het FPN, zelfs een beetje beter. In de paper, op de foto's van COCO, maakte de weg naar beneden een groot verschil: van de kleine voorwerpen vonden de 1.000 beste proposals van een pyramid zonder die weg er 30,5%, die van het FPN 44,9%, gemiddeld over IoU's van 0,5 tot 0,95: hun *average recall*. Oefening 5 vraagt waarom.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Een one-stage detector met een feature pyramid:

        1. De backbone maakt de maps C2 tot C5, elk half zo groot als de vorige.
        2. Van boven naar beneden: P5 is de 1×1 convolution van C5; elk niveau eronder is de 1×1 convolution van zijn map van de backbone, plus het niveau erboven, twee keer zo groot gemaakt met `F.interpolate(..., scale_factor=2, mode="nearest")`.
        3. Een 3×3-convolutie op elk niveau geeft P2 tot P5, allemaal met hetzelfde aantal channels.
        4. Eén head op elk niveau; de rijen van de niveaus achter elkaar, de fijnste eerst.
        5. De anchors in dezelfde volgorde: op niveau $k$ vakjes van $2^k$ pixels, en op elk vakje anchors van 2 en $2\sqrt{2}$ vakjes breed.
        6. Train met hard negative mining: de cross-entropy over de anchors die een vorm leren en de drie keer zoveel achtergrond-anchors met de hoogste cross-entropy, plus de box loss. Detecteer zoals je one-stage detector deed, met NMS voor elke soort.
        """),
        L.watch_out("De niveaus worden opgeteld, niet gestapeld: de 1×1 convolution geeft elke map van de backbone dezelfde 32 channels als het niveau erboven, zodat die twee getal voor getal kunnen worden opgeteld. Met `torch.cat` langs de channels zou elk niveau meer channels krijgen dan het niveau erboven."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld doet de stappen van het recept op een testscène met kleine en grote vormen, `scene(26, spread=True)`, met een netwerk dat nog niet getraind is; zijn stap 6 rekent uit wat elke anchor leert, de targets van het trainen. `part2` is de backbone van je one-stage detector zonder zijn laatste deel, dat nu `part3` is; `part4` en `part5` voegen elk een pooling en een convolutie toe. Twee lijsten hebben er elk één per niveau: `lateral` met 1×1 convolutions en `smooth` met 3×3-convoluties. Stap 2 laat eerst nearest-neighbor-interpolatie zien op een map van 2 bij 2 getallen, en schrijft dan de weg naar beneden uit, niveau voor niveau. In stap 5 legt `anchor_boxes(cells, stride, sizes)` van [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) de anchors van één niveau, en `torch.zeros(n, dtype=torch.int64) + k` geeft elk ervan het nummer van zijn niveau. In stap 6 zegt `match_anchors` uit `everything_ai.scenes` welke vorm elke anchor leert, zoals je `match` deed: het nummer van de vorm, of −1 of −2. Voer de cel uit zoals hij is.
    """)
    return


@app.cell
def example_pyramid():
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    s = scene(26, spread=True)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255

    # 1. de backbone in vier delen, elk eindigend in een map van de pyramid; dan een 1x1- en een
    #    3x3-convolutie voor elk niveau, en één head
    torch.manual_seed(0)
    part2 = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    )
    part3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    part4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    part5 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    lateral = [nn.Conv2d(c, 32, 1) for c in (32, 64, 128, 128)]
    smooth = [nn.Conv2d(32, 32, 3, padding=1) for i in range(4)]
    head = nn.Conv2d(32, 2 * 8, 3, padding=1)
    with torch.no_grad():
        c2 = part2(x)
        c3 = part3(c2)
        c4 = part4(c3)
        c5 = part5(c4)
        print(c2.shape, c3.shape, c4.shape, c5.shape)

        # 2. van boven naar beneden: nearest neighbor maakt van elk getal een blok van 2 bij 2; dan de
        #    1x1-map van elk niveau plus het niveau erboven, twee keer zo groot
        small = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        print(F.interpolate(small, scale_factor=2, mode="nearest"))

        p5 = lateral[3](c5)
        p4 = lateral[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = lateral[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = lateral[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # 3. de 3x3-convolutie van elk niveau
        maps = [smooth[0](p2), smooth[1](p3), smooth[2](p4), smooth[3](p5)]
        print([p.shape for p in maps])

        # 4. één head op elk niveau; de rijen van elk niveau zoals in je Detector, de fijnste eerst
        rows = [head(p).permute(0, 2, 3, 1).reshape(1, -1, 8) for p in maps]
        out = torch.cat(rows, dim=1)
        print([r.shape[1] for r in rows], out.shape)

    # 5. de anchors in dezelfde volgorde: op niveau k vakjes van 2 ** k pixels, en op elk vakje
    #    2 anchors, 2 en 2,83 vakjes breed; en het niveau van elke anchor
    anchors, level = [], []
    for k, cells in [(2, 32), (3, 16), (4, 8), (5, 4)]:
        side = 2 ** k
        sizes = [(2 * side, 2 * side), (2 * math.sqrt(2) * side, 2 * math.sqrt(2) * side)]
        a = torch.tensor(anchor_boxes(cells, side, sizes)).float()
        anchors.append(a)
        level.append(torch.zeros(len(a), dtype=torch.int64) + k)
    anchors, level = torch.cat(anchors), torch.cat(level)
    print(anchors.shape, anchors[0], anchors[-1])

    # 6. de targets van het trainen: welke anchors elke vorm leren, en op welk niveau ze liggen
    label = torch.tensor(match_anchors(anchors, [shape.box for shape in s.shapes]))
    for i in range(len(s.shapes)):
        top, left, bottom, right = s.shapes[i].box
        size = ((bottom - top + 1) * (right - left + 1)) ** 0.5
        print(s.shapes[i].kind, s.shapes[i].box, round(size, 1), level[label == i])
    print((label >= 0).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vier maps van de backbone hebben shapes (1, 32, 32, 32), (1, 64, 16, 16), (1, 128, 8, 8) en (1, 128, 4, 4): elk deel halveert de vakjes in beide richtingen. De map van 2 bij 2 getallen wordt 4 bij 4, elk getal een blok van 2 bij 2: de eerste rij (1, 1, 2, 2). De vier niveaus hebben allemaal 32 channels, op 32 bij 32 tot 4 bij 4 vakjes. De head geeft 2.048, 512, 128 en 32 rijen, samen (1, 2720, 8), één rij voor elk van de 2.720 anchors. De eerste anchor, (−2, −2, 5, 5), is 8 pixels breed rond het midden van het eerste vakje van niveau 2; de laatste, ongeveer (66,7; 66,7; 156,3; 156,3), is 90,5 pixels breed rond het midden van het laatste vakje van niveau 5, en steekt buiten de scène uit.

    De scène heeft vijf vormen, en 17 anchors leren er een van. Het vierkant van 10 pixels en de cirkel van 10,5 worden alleen door anchors van niveau 2 geleerd, door 4 en door 3; de cirkel van 20 pixels door 3 anchors van niveau 3; de driehoek van 41 pixels door 3 van niveau 4. De cirkel van 23 pixels wordt geleerd door 3 anchors van niveau 3 en 1 van niveau 4: hij ligt tussen de groottes van die twee niveaus in, 22,6 en 32 pixels. De FPN-detector heeft 299.728 parameters, waarvan de 1×1 convolutions en 3×3-convoluties van de pyramid er 48.384 hebben en de head 4.624.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén head voor elk niveau
    """)
    return


@app.cell(hide_code=True)
def choice_head_cell():
    choice_head = L.choice(
        "head",
        question="De FPN-detector heeft één head voor alle vier de niveaus: dezelfde convolutie van 3 bij 3 op P2, met vakjes van 4 pixels, en op P5, met vakjes van 32. Waarom kunnen dezelfde weights kleine vormen vinden op P2 en grote op P5?",
        options={
            "right": "Op elk niveau zijn de anchors 2 en 2,83 vakjes breed: een vorm die daarbij past, ziet er op elk niveau hetzelfde uit, dus de head leert hem één keer",
            "pixels": "De head werkt in pixels, en elk niveau beslaat dezelfde 128 bij 128 pixels",
            "resized": "Vóór de head wordt elk niveau 32 bij 32 vakjes gemaakt",
            "learns": "Dat kunnen ze niet: de head leert de kleine vormen, en de grote worden door de backbone gevonden",
        },
        correct="right",
        why={
            "right": "Elk niveau heeft 32 channels, en zijn anchors zijn evenveel van zijn eigen vakjes breed. Een vorm van 10 pixels op P2 beslaat evenveel vakjes als een van 80 pixels op P5, dus de head kan één keer leren hoe een vorm van zo'n 2 vakjes eruitziet. De paper vond dat een head per niveau niet beter was: na de weg naar beneden weet elk niveau ongeveer evenveel van wat het ziet.",
            "pixels": "De head ziet vakjes, geen pixels: 3 bij 3 ervan rond elk vakje. Een vakje is 4 pixels op P2 en 32 op P5. Wat op elk niveau hetzelfde is, is hoeveel vakjes een anchor breed is.",
            "resized": "Elk niveau houdt zijn eigen grootte: P5 heeft 4 bij 4 vakjes, P2 32 bij 32. De head, een convolutie, werkt op maps van elke grootte, en geeft elk vakje zijn rijen.",
            "learns": "De head geeft elke anchor van elk niveau zijn logits en offsets, ook de anchors van P5, die met de seeds 0 tot 4 elk 36 tot 38 van de grote vormen van de testscènes vinden. Dezelfde weights bedienen alle niveaus.",
        },
    )
    choice_head
    return (choice_head,)


@app.cell(hide_code=True)
def text_exercise_pyramid():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De weg naar beneden

        Maak `pyramid(cs, lateral, smooth)` af, stap 2 en 3 van het recept: `cs` is een lijst van de maps van de backbone, de fijnste eerst, elk een tensor met shape (N, C, H, W) met de helft van de rijen en kolommen van de vorige; `lateral` en `smooth` zijn lijsten van een 1×1 convolution en een 3×3-convolutie voor elk niveau. Hij geeft een lijst van de niveaus terug, de fijnste eerst. Begin bovenaan, met `lateral[-1](cs[-1])`, en ga niveau voor niveau naar beneden, zoals het uitgewerkte voorbeeld deed met `p5` tot `p2`, maar voor elk aantal niveaus: de check probeert vier niveaus, drie niveaus met twee scènes, en de maps van de backbone van de les. Als hij klopt, bewaart hij `pyramid` in je project, als `work/projects/cnn/pyramid.py`.
        """),
        L.task("Laat `pyramid` de niveaus van een feature pyramid network geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pyramid():
    box_pyramid = L.code_box("pyramid", lines=6)
    box_pyramid
    return (box_pyramid,)


@app.cell(hide_code=True)
def check_pyramid(box_pyramid):
    result_pyramid = L.check("pyramid", box_pyramid.value)
    result_pyramid
    return (result_pyramid,)


@app.cell(hide_code=True)
def hints_pyramid():
    L.hints(
        "pyramid",
        [
            "Stap 2 van het uitgewerkte voorbeeld schreef de weg naar beneden uit voor vier niveaus, van `p5` tot `p2`. Doe hetzelfde in een loop over de niveaus: houd de niveaus van boven naar beneden bij in een lijst, en ga van het niveau onder de top naar niveau 0.",
            "Begin met `down = [lateral[-1](cs[-1])]` en `for i in reversed(range(len(cs) - 1)):`. Elk niveau eronder is zijn eigen 1×1-map plus het laatste niveau in `down`, twee keer zo groot gemaakt: `down.append(lateral[i](cs[i]) + F.interpolate(___, scale_factor=2, mode=\"nearest\"))`.",
            "`down` heeft nu de top eerst, en de niveaus moeten met de fijnste eerst terugkomen, elk door zijn eigen 3×3-convolutie: `return [smooth[i](down[___]) for i in range(len(cs))]`.",
        ],
        why="Met vier niveaus loopt de loop voor `i` = 2, 1 en 0: `down` wordt de niveaus 3, 2, 1 en 0, in die volgorde, dus niveau `i` is `down[-1 - i]`: niveau 0 het laatste, `down[-1]`, en niveau 3 het eerste, `down[-4]`. De 3×3-convolutie komt na al het optellen: wat naar beneden gaat, is de som, niet het gladgestreken niveau.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De rijen van elk niveau

        Maak `head_rows(head, maps)` af, stap 4 van het recept: `head` is één convolutie, en `maps` een lijst van de niveaus, de fijnste eerst, elk met shape (N, C, H, W). Hij geeft één tensor terug met shape (N, A, 8), met A het aantal anchors van alle niveaus samen: op elk niveau de getallen die `head` elk vakje geeft, als rijen van 8, zoals je `Detector` ze maakte; en de rijen van de niveaus achter elkaar, de fijnste eerst. De check probeert twee niveaus met twee scènes, drie niveaus met één anchor per vakje, en de vier niveaus van de les. Als hij klopt, bewaart hij `head_rows` in je project, als `work/projects/cnn/head_rows.py`.
        """),
        L.task("Laat `head_rows` de rijen van elke anchor van elk niveau geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_head_rows = L.code_box("head-rows", lines=4)
    box_head_rows
    return (box_head_rows,)


@app.cell(hide_code=True)
def check_rows(box_head_rows):
    result_head_rows = L.check("head-rows", box_head_rows.value)
    result_head_rows
    return (result_head_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "head-rows",
        [
            "Stap 4 van het uitgewerkte voorbeeld, als functie: daar maakten `permute` en `reshape` de rijen van elk niveau, en zette `torch.cat` ze achter elkaar.",
            "De check geeft meerdere scènes tegelijk, dus elk niveau wordt `reshape(len(p), -1, 8)`, niet `reshape(1, -1, 8)`: `len(p)` is het aantal scènes, en met `-1` rekent `reshape` het aantal rijen uit.",
            "`return torch.cat([head(p).permute(___).reshape(len(p), -1, 8) for p in maps], dim=___)`.",
        ],
        why="Met twee scènes geeft een niveau van 8 bij 8 vakjes met 2 anchors per vakje (2, 128, 8), en een niveau van 4 bij 4 (2, 32, 8). Binnen elke scène achter elkaar gezet, langs dimensie 1, maken ze (2, 160, 8). Langs dimensie 0 zouden ze als meer scènes op elkaar gestapeld worden, en hun verschillende aantallen rijen passen niet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_background():
    mo.md(r"""
    ### Oefening 4. Welke anchors tellen
    """)
    return


@app.cell(hide_code=True)
def choice_background_cell():
    choice_background = L.choice(
        "background",
        question="In een trainingsscène leren 10 van de 2.720 anchors een vorm, en geen enkele wordt genegeerd. Hoeveel anchors telt de cross-entropy van de loss met hard negative mining in die scène?",
        options={
            "right": "40: de 10, en de 30 achtergrond-anchors met de hoogste cross-entropy",
            "random": "40: de 10, en 30 willekeurig gekozen achtergrond-anchors",
            "all": "2.720: elke anchor, zoals in je one-stage detector",
            "shapes": "10: alleen de anchors die een vorm leren",
        },
        correct="right",
        why={
            "right": "Drie achtergrond-anchors voor elke anchor die een vorm leert: 10 + 3 × 10 = 40. Het zijn de anchors waar de detector nu het verst naast zit, dus ze wisselen van batch tot batch terwijl hij leert. De andere 2.680 achtergrond-anchors, die hij al goed doet, tellen niet mee.",
            "random": "Het aantal klopt, maar hard negative mining neemt de achtergrond-anchors met de hoogste cross-entropy: de moeilijke. Willekeurige achtergrond-anchors zijn meestal makkelijk, en leren de detector bijna niets.",
            "all": "Dat is de loss die mislukte: met 2.710 achtergrond-anchors tegen 10 leert de detector overal achtergrond te zeggen, en vindt hij 9 van de 786 vormen. Hard negative mining telt de 10 en drie keer zoveel achtergrond-anchors.",
            "shapes": "Dan leert de detector nooit achtergrond, en geeft hij overal boxes. Hard negative mining telt ook de 30 achtergrond-anchors die hij het meest fout doet.",
        },
    )
    choice_background
    return (choice_background,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint de FPN-detector van de course, de backbone van deze les met je `pyramid` en je `head_rows`, met hard negative mining, 8 epochs lang op de 2.000 trainingsscènes met kleine en grote vormen, in mini-batches van 32, zoals je one-stage detector trainde. Daarna vindt hij de vormen van de 200 testscènes met je `boxes_from` en NMS, en telt hij wat hij van elke grootte vindt. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt een paar minuten. De grafiek laat na elke epoch de loss op de testscènes zien. Als hij klaar is, kun je door de epochs terugspoelen, en zien welke boxes de detector na elke epoch aan drie testscènes gaf.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train de FPN-detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_fpn(train_button, result_pyramid, result_head_rows):
    result_pyramid, result_head_rows  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om de FPN-detector te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "FPN-detector: de loss op de testscènes", "x": "epoch", "y": "test loss",
                                        "alt": "De loss van de FPN-detector op de 200 testscènes na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**FPN-detector** · epoch {_p['epoch']} van {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**FPN-detector** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `pyramid` en je `head_rows` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan tien minuten. Doet je `pyramid` of je `head_rows` meer werk dan nodig, zoals een loop over de vakjes?*",
                "pyramid": "*Je `pyramid` in `work/projects/cnn/pyramid.py` stopt met een fout, of geeft geen vier niveaus van 32 maps, van 32 bij 32 tot 4 bij 4 vakjes, die het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "head_rows": "*Je `head_rows` in `work/projects/cnn/head_rows.py` stopt met een fout, of geeft voor een scène geen 2.720 rijen van 8 die het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van de detector. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course nam de plaats in van " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van de FPN-detector na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _words = {"scene": ("scène {seed}: {n} box", "scène {seed}: {n} boxes"), "alt": "Drie testscènes met kleine en grote vormen, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die de detector haar na die epoch gaf, na NMS."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"Na epoch {_e}: test loss {L.number(_val[_e - 1], 3)}, en {_n} {'box' if _n == 1 else 'boxes'} van 0,5 of meer op deze drie scènes, die {C.SHOWN_SHAPES} vormen hebben.")

            if len(_moments) == len(_val) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke vormen krijgen het eerst een box, de grote of de kleine?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _found, _shapes = _result["found"], _result["shapes"]
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("cirkels", "vierkanten", "driehoeken"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""De FPN-detector, met {C.whole(_result['params'], L.lang)} parameters, vindt {sum(_found)} van de {sum(_shapes)} vormen van de 200 testscènes: een box van hun soort, met een score van 0,5 of meer en een IoU boven 0,5. Dat zijn {_found[0]} van de {_shapes[0]} kleine vormen, {_found[1]} van de {_shapes[1]} middelgrote en {_found[2]} van de {_shapes[2]} grote. {_result['false']} van zijn boxes van 0,5 of meer vinden geen vorm. Zijn mAP bij een IoU van 0,5 is {L.number(_result['map'], 3)}, met een AP van {_aps}; bij 0,75 is hij {L.number(_result['map75'], 3)}. De FPN-detector van de course, op dezelfde manier getraind met de seeds 0 tot 4, vindt 727 tot 741 vormen, waarvan 244 tot 249 kleine, met een mAP van 0,934 tot 0,950.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_top_down():
    mo.md(r"""
    ### Oefening 5. Waarom de weg naar beneden?
    """)
    return


@app.cell(hide_code=True)
def explain_top_down_cell():
    explain_top_down = L.explain(
        "why-top-down",
        question="Op de scènes van de course vindt de pyramid zonder de weg naar beneden evenveel kleine vormen als het FPN. Op de foto's van COCO maakte de weg naar beneden een groot verschil voor kleine voorwerpen: een average recall van 44,9% tegen 30,5%. Wat heeft een scherp niveau nodig om te zien wat een klein ding is, in een foto en in de scènes van de course? En wat geeft de weg naar beneden het?",
        model_answer="In de scènes van de course is een kleine vorm een cirkel, een vierkant of een driehoek van één kleur op een grijze achtergrond: zijn omtrek verraadt zijn soort, en elk getal van C2 ziet al 18 bij 18 pixels, zoveel als een kleine vorm, dus de vroege layers zien genoeg. In een foto blijkt wat een klein voorwerp is, een auto ver weg of een vogel, niet alleen uit zijn randen en kleuren: daarvoor zijn patronen nodig die alleen diepe layers leren, na veel convoluties, en de vroege maps van een diepe backbone kennen randen en kleuren, geen dingen. De weg naar beneden brengt wat de diepe maps weten naar de scherpe niveaus, op hun eigen fijne vakjes, zodat de scherpe niveaus allebei hebben: de goede grootte, en de betekenis. Zonder die weg heeft een scherp niveau de goede grootte voor een klein voorwerp, maar te weinig betekenis om te zien wat het is.",
        points=[
            "je zegt dat in de scènes van de course de soort van een kleine vorm blijkt uit zijn omtrek en kleur, die de vroege layers al zien",
            "je zegt dat je in een foto pas ziet wat een klein voorwerp is met wat diepe layers leren, en dat de vroege maps dat niet weten",
            "je zegt dat de weg naar beneden de betekenis van de diepe maps naar de scherpe niveaus brengt, zodat die zowel de fijne vakjes als de betekenis hebben",
        ],
    )
    explain_top_down
    return (explain_top_down,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De anchors, targets en boxes van [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) werken nu op vier niveaus: elk niveau zijn eigen anchors, elke vorm geleerd door de anchors waarvan de grootte past. De 1×1 convolutions van [1×1 convolutions: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) geven elke map van de backbone dezelfde 32 channels, en de nearest-neighbor-interpolatie van [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl) maakt elk niveau op de weg naar beneden twee keer zo groot. Je `pyramid` bouwt de niveaus van boven naar beneden, je `head_rows` laat één head elke anchor zijn rijen geven, en hard negative mining zorgt dat de loss niet verdrinkt in achtergrond. Op de scènes van de course vindt het FPN 244 tot 249 van de 268 kleine vormen, waar één grove map er 129 tot 143 vond. Je project `cnn` heeft nu `pyramid.py` en `head_rows.py`. In de volgende les, *Focal loss en RetinaNet*, krijgt een one-stage detector met een feature pyramid een loss die elke anchor weegt naar hoe fout hij zit, in plaats van een paar moeilijke te kiezen; daarmee vond RetinaNet voorwerpen even goed als de two-stage detectors.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_head, result_pyramid, result_head_rows, choice_background, explain_top_down):
    L.wrap_up(choice_head, result_pyramid, result_head_rows, choice_background, explain_top_down)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
