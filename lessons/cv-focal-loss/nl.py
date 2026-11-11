# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-focal-loss. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-focal-loss", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een feature pyramid voor kleine en grote voorwerpen](/course/open/cv-fpn?lang=nl) en [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hard():
    quick_hard = L.choice(
        "quick-hard",
        question="Hard negative mining, zoals je FPN-detector getraind werd: in een scène waarin 10 van de 2.720 anchors een vorm leren en er geen genegeerd wordt, telt de cross-entropy die 10 en de 30 achtergrond-anchors met de hoogste cross-entropy. Wat gebeurt er met de cross-entropy van de andere 2.680 achtergrond-anchors?",
        options={
            "right": "Die telt in die batch helemaal niet mee, hoe zeker de detector er ook van is",
            "third": "Die telt voor een derde mee, zodat achtergrond even zwaar weegt als de vormen",
            "later": "Die wordt bewaard, en telt mee in de volgende batch",
            "box": "Die gaat naar de box loss",
        },
        correct="right",
        why={
            "right": "Hard negative mining is alles of niets: een anchor telt helemaal mee, of helemaal niet. De andere 2.680 achtergrond-anchors krijgen een gewicht van 0, ook die waar de detector maar redelijk zeker van is. De loss van vandaag telt elke anchor mee, elk met een eigen gewicht.",
            "third": "Hard negative mining weegt niet: de 40 anchors tellen helemaal mee, de andere helemaal niet. Zie [Een feature pyramid voor kleine en grote voorwerpen](/course/open/cv-fpn?lang=nl).",
            "later": "Elke batch kiest zijn eigen 30: de achtergrond-anchors die de detector op dat moment het meest fout doet. De andere worden niet bewaard voor later.",
            "box": "De box loss telt alleen de anchors die een vorm leren: een achtergrond-anchor heeft geen box om te leren. De andere achtergrond-anchors tellen helemaal niet mee.",
        },
    )
    quick_hard
    return (quick_hard,)


@app.cell(hide_code=True)
def quick_check_stages():
    quick_stages = L.choice(
        "quick-stages",
        question="De eerste fase van je two-stage detector geeft elk van de 768 anchors van een scène een kans op een voorwerp. Naar hoeveel boxes kijkt zijn tweede fase opnieuw?",
        options={
            "right": "32 proposals: van de 100 anchors met de hoogste kans op een voorwerp de eerste 32 die NMS bij 0,7 overhoudt",
            "all": "Alle 768: de tweede fase kijkt opnieuw naar elke anchor",
            "hundred": "100: de anchors met de hoogste kans op een voorwerp",
            "shapes": "Zoveel als de scène vormen heeft",
        },
        correct="right",
        why={
            "right": "Van 768 anchors naar 32 proposals: de eerste fase gooit bijna alle achtergrond weg, dus de tweede fase kijkt opnieuw naar boxes die op of rond een vorm liggen. Een one-stage detector heeft geen eerste fase: zijn loss ziet elke anchor.",
            "all": "Dan zou de eerste fase niets weggooien. Hij houdt de 100 waarschijnlijkste anchors over, en NMS bij 0,7 laat daar de eerste 32 van over. Zie [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).",
            "hundred": "Die 100 gaan door NMS bij 0,7, en de eerste 32 boxes die overblijven zijn de proposals. Zie [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl).",
            "shapes": "De detector weet niet hoeveel vormen een scène heeft: hij geeft altijd 32 proposals door, en de tweede fase beslist welke daarvan een vorm bevatten.",
        },
    )
    quick_stages
    return (quick_stages,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Tot 2017 hadden de nauwkeurigste detectors twee fasen: de eerste gooit de meeste achtergrond weg, zodat de tweede opnieuw naar een paar proposals kijkt, zoals je two-stage detector deed met 32 per scène. Een one-stage detector kijkt in één keer naar elke anchor, en bijna allemaal zijn ze achtergrond: in een scène van de course leren zo'n 10 van de 2.720 anchors een vorm, en hard negative mining hield alleen de 30 moeilijkste achtergrond-anchors over. De **focal loss** (Lin en collega's, 2017) houdt ze allemaal, maar maakt de loss van een anchor kleiner naarmate de detector er al zekerder van is; daarmee vond **RetinaNet**, een one-stage detector met een feature pyramid, de voorwerpen in de foto's van COCO beter dan elke two-stage detector vóór hem. Vandaag train je de FPN-detector van de course ermee.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_easy():
    guess_easy = L.guess(
        "easy-share",
        question="De course trainde de detector van deze les. Na het trainen geeft hij bijna elke achtergrond-anchor van een testscène een kans boven 0,9 op achtergrond: zo'n 2.690 van de 2.707 anchors van een scène die niet genegeerd worden, elk met een cross-entropy onder 0,11. Zo'n 10 anchors leren een vorm. Welk deel van de opgetelde cross-entropy van een scène komt van die makkelijke achtergrond-anchors?",
        options={
            "none": "Bijna niets: elk ervan is piepklein",
            "tenth": "Ongeveer een tiende",
            "right": "Meer dan de helft",
            "all": "Bijna alles",
        },
        reveal="56%: elk ervan is klein, maar het zijn er zo'n 2.690, en de 10 anchors die een vorm leren geven 29%. Die detector is getraind met de focal loss van deze les, die de loss van die makkelijke anchors omlaag zet: in zijn focal loss geven ze 0,2%. De cross-entropy zou ze steeds zekerder blijven maken, terwijl de detector ze al goed doet.",
    )
    guess_easy
    return (guess_easy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Makkelijke anchors tellen op.** Je one-stage detector nam de cross-entropy van elke anchor die niet genegeerd wordt: $-\log p_t$, waarbij $p_t$ de kans is die softmax geeft aan de klasse die de anchor leert, achtergrond of zijn soort. Een achtergrond-anchor waar de detector zeker van is, bij $p_t = 0{,}99$, heeft een cross-entropy van maar 0,01. Maar een scène heeft er zo'n 2.700 van, en zo'n 10 anchors die een vorm leren: samen kunnen de makkelijke zwaarder wegen dan de paar die ertoe doen, zoals de gok liet zien. Een two-stage detector gooit de meeste achtergrond weg in zijn eerste fase; het RPN van Faster R-CNN leert van 256 anchors per foto, waarvan hoogstens de helft voorwerpen. Een one-stage detector heeft zo'n fase niet. Hard negative mining, dat je in je FPN-detector gebruikte, telt alleen de moeilijkste achtergrond-anchors.

        **De focal loss.** Het antwoord van de paper: houd elke anchor, en weeg hem met hoeveel de detector er nog naast zit,

        $$\mathrm{FL}(p_t) = -(1 - p_t)^{\gamma} \log p_t .$$

        Met $\gamma = 0$ is dat de cross-entropy. Met $\gamma = 2$, de waarde van de paper, telt een anchor bij $p_t = 0{,}9$ voor $(1 - 0{,}9)^2$ mee, een honderdste van zijn cross-entropy, en een anchor bij 0,99 voor een tienduizendste; een anchor bij 0,5 nog voor een kwart, en een anchor die de detector fout doet, bij $p_t = 0{,}1$, voor 81%. $\gamma$ zegt hoe snel het gewicht daalt: hoe groter, hoe eerder een anchor als klaar telt. In code komt $p_t$ uit de cross-entropy zelf: de cross-entropy is $-\log p_t$, dus $p_t = e^{-\text{CE}}$, `torch.exp(-ce)`. `F.cross_entropy(..., reduction="none")` geeft de cross-entropy van elke rij, in plaats van hun gemiddelde.

        **Opgeteld, en gedeeld door de vormen.** Je one-stage detector deelde de cross-entropy door elke anchor die niet genegeerd wordt, zo'n 2.700 in een scène van de FPN-detector: zijn gemiddelde. RetinaNet telt de focal loss op over alle anchors, en deelt door het aantal anchors dat een voorwerp leert: in de scènes van de course zo'n 10 per scène. De reden van de paper: de makkelijke achtergrond-anchors voegen bijna niets toe aan de focal loss, dus ze moeten hem ook niet kleiner maken. Gedeeld door zo'n 2.700 weegt de cross-entropy van de paar anchors die een vorm leren weinig naast de box loss, die een gemiddelde over die paar is: met het gemiddelde van de cross-entropy vond je FPN-detector maar 9 van de 786 vormen, en met het gemiddelde van de focal loss vindt de detector van de course er, zelfs beginnend bij achtergrond zoals hieronder, maar 233 tot 286.

        **Beginnen bij achtergrond.** Aan het begin, voordat de detector iets geleerd heeft, is hij ongeveer even onzeker over elke anchor: alle anchors hebben ongeveer dezelfde $p_t$, dus de focal loss zet ze allemaal even ver omlaag, en de achtergrond, met 2.697 van de 2.707 anchors van een scène, geeft meer dan 99% van de loss, cross-entropy of focal loss. RetinaNet laat elke anchor daarom bij achtergrond beginnen: de biases van de laatste layer van de head worden zo gezet dat elke anchor begint met een kans $\pi = 0{,}01$ op een voorwerp. Dan zijn de achtergrond-anchors vanaf de eerste batch makkelijk: op de testscènes geven ze aan het begin 32% van de cross-entropy, en 0,0% van de focal loss. In de paper divergeerde de cross-entropy zonder dit begin tijdens het trainen; met dit begin haalde een RetinaNet met een ResNet-50 een AP van 30,2 op COCO met de cross-entropy, en 34,0 met de focal loss. Hier gaan de 4 logits van elke anchor door softmax: met een bias $b$ voor achtergrond en 0 voor elk van de 3 soorten geeft softmax achtergrond $e^b / (e^b + 3)$. Oefening 3 rekent $b$ uit.

        **RetinaNet.** De detector van de paper is een ResNet met een feature pyramid van vijf niveaus, P3 tot P7, met 256 channels, 9 anchors op elk vakje, 3 groottes in 3 verhoudingen van breedte tot hoogte, en twee heads die alle niveaus delen, één voor de klassen en één voor de offsets, elk vier convoluties van 3 bij 3 met ReLU vóór zijn laatste layer. Elk van de 80 klassen van COCO krijgt een eigen sigmoid in plaats van één softmax, en de loss weegt ook de anchors van voorwerpen met $\alpha = 0{,}25$ en achtergrond met 0,75. Met de focal loss haalde RetinaNet een AP van 39,1 op COCO, waar de beste two-stage detector 36,8 had; de paper trainde hem ook op alleen de moeilijkste anchors; de beste variant daarvan haalde 32,8 waar de focal loss 36,0 haalde, in een kleinere opzet met foto's van 600 pixels. Volgens een voetnoot van de paper is de focal loss makkelijk uit te breiden naar meer klassen tegelijk, en werkt hij daar ook goed. Hier is het de FPN-detector van je vorige les, met zijn ene head en zijn softmax over achtergrond en 3 soorten, met $\gamma = 2$, $\pi = 0{,}01$ en geen $\alpha$.
        """),
        L.watch_out("De focal loss wordt opgeteld over elke anchor die niet genegeerd wordt, ook achtergrond: zo leert de detector achtergrond. Alleen het getal waardoor hij gedeeld wordt telt de anchors die een vorm leren. Opgeteld over alleen de vormen leert de detector nooit wat achtergrond is, en geeft hij overal boxes."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **focal loss** | de cross-entropy van elk voorbeeld maal $(1 - p_t)^\gamma$: hoe zekerder het model al is van een voorbeeld, hoe minder het meetelt | `focal_loss` |
        | **RetinaNet** | een one-stage detector (2017) met een feature pyramid en de focal loss, de eerste die voorwerpen even goed vond als de two-stage detectors | |
        | $p_t$ | de kans die het model geeft aan wat een voorbeeld is: hier de klasse die een anchor leert | `torch.exp(-ce)` |
        | $\gamma$ (gamma) | hoe snel het gewicht van een voorbeeld daalt naarmate het model er zekerder van wordt; 0 geeft de cross-entropy | `gamma` |
        | $\pi$ (pi) | de kans op een voorwerp waarmee elke anchor begint | `pi` |
        | $\alpha$ (alpha) | in de paper het gewicht van de anchors van voorwerpen (0,25) tegenover achtergrond (0,75); hier niet gebruikt | |
        | `F.cross_entropy(..., reduction="none")` | de cross-entropy van elke rij, een tensor met één getal per rij, in plaats van hun gemiddelde | |
        | `math.log(x)` | de natuurlijke logaritme van x | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_focal():
    mo.vstack([
        mo.Html(C.focal_picture({
            "title": "Loss van één anchor", "x": "p_t", "y": "loss", "point": ",",
            "key": ("γ = 0, cross-entropy", "γ = 0,5", "γ = 1", "γ = 2", "γ = 5"),
            "alt": "De loss van één anchor tegen zijn p_t, van 0 tot 1, voor gamma 0, de cross-entropy, als gestreepte grijze lijn, en voor gamma 0,5, 1, 2 en 5 in paars. Alle krommen beginnen hoog bij een kleine p_t; hoe groter gamma, hoe lager de kromme en hoe eerder hij dicht bij 0 ligt. Bij gamma 2 ligt hij vanaf p_t 0,6 dicht bij 0.",
        })),
        mo.md(r"""
        De loss van één anchor tegen zijn $p_t$, zoals in de eerste figuur van de paper. De gestreepte grijze lijn is de cross-entropy, $\gamma = 0$: nog 0,105 bij $p_t = 0{,}9$. Hoe groter $\gamma$, hoe eerder de loss dicht bij 0 ligt: met $\gamma = 2$, de dikke lijn, vanaf ongeveer $p_t = 0{,}6$. Voor een anchor die de detector fout doet, bij een kleine $p_t$, blijft elke kromme hoog.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Wat doet $\gamma$ met een detector? De slider kiest $\gamma$, en daarmee een detector die de course met die focal loss trainde, beginnend bij achtergrond, seed 0. Links de loss van één anchor tegen zijn $p_t$, met die $\gamma$ dik; rechts de boxes van 0,5 of meer van de detector op de scène van het uitgewerkte voorbeeld van je vorige les.
    """)
    return


@app.cell(hide_code=True)
def explore_gamma_cell():
    def _frame(g):
        g = float(g)
        _html, _stop = C.gamma_frame(g, {"title": ("γ = {g}: {n} box", "γ = {g}: {n} boxes"), "box": "{kind} {score}", "kinds": ("cirkel", "vierkant", "driehoek"), "point": ",",
                                         "chart": {"title": "Loss van één anchor", "x": "p_t", "y": "loss",
                                                   "alt": "De loss van één anchor tegen zijn p_t voor gamma 0, 1, 2 en 5, met gamma {g} als dikke paarse lijn en de andere dun en grijs."},
                                         "alt": "Naast de krommen de scène van het uitgewerkte voorbeeld met haar vijf echte boxes en de {n} gestreepte boxes van 0,5 of meer van de detector die met gamma {g} getraind is."})
        _weight = "voor zijn hele cross-entropy" if g == 0 else f"voor 1/{C.whole(_stop['factor'], L.lang)} van zijn cross-entropy"
        return _html, f"γ = {g:g}: een anchor bij p_t = 0,9 telt mee {_weight}. Daarmee getraind vindt de detector {_stop['found']} van de 786 vormen van de testscènes, met {_stop['false']} valse boxes, en een mAP van {L.number(_stop['map'], 3)}."

    explore_gamma = L.explore("gamma-slider", "γ", _frame, start=2, values=[0, 1, 2, 5], role="",
                              task="Welke γ geeft de minste valse boxes, en welke vindt de minste vormen? Wat gebeurt er met de loss van een anchor bij p_t = 0,6 als γ groter wordt?")
    explore_gamma
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Over de seeds 0 tot 4 vinden de detectors met $\gamma$ = 0, 1 en 2 ongeveer evenveel vormen: 729 tot 740, 726 tot 741 en 723 tot 734. Wat verandert zijn de valse boxes, de boxes van 0,5 of meer die geen vorm vinden: 127 tot 145 met de cross-entropy, 100 tot 127 met $\gamma = 1$, en 95 tot 109 met $\gamma = 2$. Met $\gamma = 5$ geeft de detector er het minst, 71 tot 81, maar vindt hij minder vormen, 706 tot 711: oefening 5 vraagt waarom. In de paper, op COCO, was $\gamma = 2$ het best; op deze scènes doet $\gamma = 1$ het ongeveer even goed. De tabel zet de losses naast elkaar, allemaal op de FPN-detector van je vorige les, over de seeds 0 tot 4:
        """),
        mo.md(C.comparison_table({"header": ("Loss", "Vormen gevonden", "mAP", "mAP 0,75", "Valse boxes"),
                                  "names": {"mean": "gemiddelde van de cross-entropy", "hard": "hard negative mining", "gamma0": "cross-entropy, gedeeld door de vormen",
                                            "focal": "focal loss, γ = 2", "no-start": "focal loss, zonder het begin"}}, L.lang)),
        mo.md(r"""
        Elke cel geeft het laagste en het hoogste getal over de seeds 0 tot 4: hoeveel van de 786 vormen van de 200 testscènes hij vindt, met een box van hun soort, een score van 0,5 of meer en een IoU boven 0,5; zijn mAP bij een IoU van 0,5 en van 0,75; en zijn valse boxes. De cross-entropy gedeeld door de vormen en de focal loss met γ = 2 beginnen bij achtergrond; de andere rijen niet. De focal loss vindt evenveel vormen als hard negative mining, met iets minder valse boxes, en met boxes die bij een IoU van 0,75 iets minder goed passen. Gedeeld door de vormen werkt de cross-entropy ook, met meer valse boxes; het gemiddelde van de cross-entropy vindt bijna niets, 8 tot 59 vormen, en beginnend bij achtergrond ongeveer de helft, 361 tot 428. Zonder het begin doet de focal loss het hier ongeveer even goed, en de cross-entropy gedeeld door de vormen ook, met 726 tot 737 vormen; in de paper, met zo'n 100.000 anchors per foto in plaats van 2.720, divergeerde de cross-entropy zonder het begin.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        Een one-stage detector met de focal loss, zoals RetinaNet:

        1. De FPN-detector van je vorige les: de backbone, de feature pyramid, één head, en 2.720 anchors.
        2. Voor elke anchor die niet genegeerd wordt, zijn cross-entropy, met `reduction="none"`, en zijn $p_t = e^{-\text{CE}}$.
        3. Zijn focal loss: $(1 - p_t)^\gamma$ maal zijn cross-entropy, met $\gamma = 2$.
        4. De som over die anchors, gedeeld door het aantal anchors dat een vorm leert; plus de box loss.
        5. Laat elke anchor bij achtergrond beginnen: geef elke achtergrondlogit de bias $b$ waarmee achtergrond van softmax over $b$ en een 0 voor elke soort een kans van $1 - \pi$ krijgt, en de logits van de soorten een bias van 0.
        6. Train en detecteer zoals met hard negative mining, met NMS voor elke soort.
        """),
        L.watch_out("Deel door het aantal anchors dat een vorm leert, `(classes > 0).sum()`, niet door elke anchor die niet genegeerd wordt. Een gemiddelde over zo'n 2.700 anchors maakt de loss van de paar die een vorm leren zo klein naast de box loss, dat de detector veel minder vormen vindt, ook met de focal loss: zo'n derde ervan."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld doet de stappen 2 tot 4 van het recept op zes verzonnen anchors, en laat dan op een hele scène zien waarom stap 5 nodig is. Elke anchor is een rij van 4 logits, achtergrond eerst, zoals je detector ze geeft, en `classes` bevat wat elke anchor leert, zoals je `targets` het maakte: 0 voor achtergrond, 1 tot 3 voor een soort, −1 als hij genegeerd wordt. In stap 5 zijn de anchors van de pyramid en `match_anchors` uit `everything_ai.scenes` die van het uitgewerkte voorbeeld van je vorige les; de klassen van de scène komen uit het nummer van de vorm die elke anchor leert, zoals in je `targets`. `torch.zeros(len(anchors), 4)` geeft elke anchor de logits van een detector aan het begin, die over elke klasse even onzeker is. Voer de cel uit zoals hij is.
    """)
    return


@app.cell
def example_focal():
    import math
    import torch
    import torch.nn.functional as F
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    # 2. zes anchors: elk een rij van 4 logits (achtergrond, cirkel, vierkant, driehoek), en de klasse
    #    die hij leert; de cross-entropy van elke anchor die niet genegeerd wordt, en p_t = e^(-CE)
    logits = torch.tensor([
        [6.0, 0.0, 0.0, 0.0],   # achtergrond, en de detector is er zeker van
        [3.0, 0.0, 0.0, 0.0],   # achtergrond, en de detector is er redelijk zeker van
        [0.0, 2.0, 0.0, 0.0],   # achtergrond die op een cirkel lijkt
        [0.0, 0.0, 3.0, 0.0],   # genegeerd
        [0.0, 5.0, 0.0, 0.0],   # een cirkel, en de detector is er zeker van
        [1.0, 0.0, 1.0, 0.0],   # een vierkant, en de detector is niet zeker
    ])
    classes = torch.tensor([0, 0, 0, -1, 1, 2])
    kept = classes >= 0
    ce = F.cross_entropy(logits[kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    print(ce)
    print(p)

    # 3. de focal loss van elke anchor, met gamma 2, en hoeveel keer zo klein hij is
    fl = (1 - p) ** 2 * ce
    print(fl)
    print(ce / fl)

    # 4. opgeteld, en gedeeld door het aantal anchors dat een vorm leert: de cross-entropy en de
    #    focal loss
    shapes = (classes > 0).sum()
    print(ce.sum() / shapes, fl.sum() / shapes)

    # 5. waarom elke anchor bij achtergrond begint: de anchors van de pyramid op een scène, wat elk
    #    leert, en een detector aan het begin, met elke logit 0
    s = scene(26, spread=True)
    anchors = []
    for k, cells in [(2, 32), (3, 16), (4, 8), (5, 4)]:
        side = 2 ** k
        sizes = [(2 * side, 2 * side), (2 * math.sqrt(2) * side, 2 * math.sqrt(2) * side)]
        anchors.append(torch.tensor(anchor_boxes(cells, side, sizes)).float())
    anchors = torch.cat(anchors)
    label = torch.tensor(match_anchors(anchors, [shape.box for shape in s.shapes]))
    scene_classes = torch.zeros(len(anchors), dtype=torch.int64)
    scene_classes[label == -2] = -1
    for i in range(len(s.shapes)):
        scene_classes[label == i] = s.shapes[i].label + 1
    print((scene_classes == 0).sum(), (scene_classes == -1).sum(), (scene_classes > 0).sum())

    start = torch.zeros(len(anchors), 4)
    kept = scene_classes >= 0
    ce = F.cross_entropy(start[kept], scene_classes[kept], reduction="none")
    fl = (1 - torch.exp(-ce)) ** 2 * ce
    background = scene_classes[kept] == 0
    print(ce[0], fl[0])
    print(ce[background].sum() / ce.sum(), fl[background].sum() / fl.sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De achtergrond-anchor waar de detector zeker van is, heeft $p_t = 0{,}9926$ en een cross-entropy van 0,0074; zijn focal loss is 18.354 keer zo klein, 0,0000004, wat Python schrijft als `4.0365e-07`. De anchor waar hij redelijk zeker van is, bij 0,87, telt 59 keer zo weinig mee, en de cirkel waar de detector zeker van is 2.547 keer zo weinig. De twee anchors die de detector fout doet, houden veel meer van de hunne over: de achtergrond die op een cirkel lijkt, bij $p_t = 0{,}0963$, houdt 1,912 van zijn 2,341, en het vierkant, bij 0,3655, 0,405 van zijn 1,006. Gedeeld door de 2 anchors die een vorm leren geeft de cross-entropy 1,757 en de focal loss 1,160, waarvan 82% van de achtergrond komt die op een cirkel lijkt.

    De scène heeft 2.693 achtergrond-anchors, 10 die genegeerd worden en 17 die een vorm leren. Aan het begin, met elke logit 0, heeft elke anchor $p_t = 1/4$: een cross-entropy van $\log 4 = 1{,}386$, en een focal loss van 0,780. De focal loss zet elke anchor even ver omlaag, dus de achtergrond geeft 99,4% van allebei. Daar is stap 5 voor: als ze bij achtergrond beginnen, zijn de achtergrond-anchors vanaf de eerste batch makkelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_easy():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Honderd keer zo klein
    """)
    return


@app.cell(hide_code=True)
def choice_easy_cell():
    choice_easy = L.choice(
        "easy",
        question="Met γ = 2 is de focal loss van een anchor (1 − p_t)² maal zijn cross-entropy. De detector geeft een achtergrond-anchor een kans van 0,9 op achtergrond. Hoeveel keer zo klein is zijn focal loss als zijn cross-entropy?",
        options={
            "right": "100 keer: (1 − 0,9)² = 0,01",
            "ten": "10 keer: 1 − 0,9 = 0,1",
            "sure": "1,2 keer: 0,9² = 0,81",
            "same": "Niet kleiner: de focal loss verandert alleen de anchors die de detector fout doet",
        },
        correct="right",
        why={
            "right": "(1 − 0,9)² = 0,1² = 0,01: van zijn cross-entropy, 0,105, blijft een honderdste over, 0,00105. Bij p_t = 0,99 een tienduizendste; bij 0,5 een kwart. Hoe zekerder de detector is, hoe sneller zijn loss daalt.",
            "ten": "Dat is γ = 1. Met γ = 2 wordt het gewicht 1 − p_t gekwadrateerd: 0,1² = 0,01, honderd keer zo klein.",
            "sure": "0,9² zou een gewicht van p_t² zijn, en dan tellen de anchors waar de detector het zekerst van is het zwaarst. De focal loss weegt met 1 − p_t: (1 − 0,9)² = 0,01.",
            "same": "De focal loss verandert elke anchor, de makkelijke het meest: (1 − p_t)² is 0,01 bij 0,9, en 0,81 bij 0,1.",
        },
    )
    choice_easy
    return (choice_easy,)


@app.cell(hide_code=True)
def text_exercise_focal():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De focal loss

        Maak `focal_loss(logits, classes, gamma)` af, de stappen 2 tot 4 van het recept zonder de box loss: `logits` bevat de 4 logits van elke anchor van N scènes, een tensor met shape (N, A, 4), achtergrond eerst; `classes` bevat wat elke anchor leert, een tensor met shape (N, A): 0 voor achtergrond, 1 tot 3 voor een soort, −1 als hij genegeerd wordt. Hij geeft één getal terug, als tensor: de focal loss van elke anchor die niet genegeerd wordt, opgeteld, en gedeeld door het aantal anchors dat een vorm leert. De loss van je one-stage detector begon met `F.cross_entropy(out[:, :, :4][kept], classes[kept])`. De check probeert de zes anchors van het uitgewerkte voorbeeld, twee scènes van 40 anchors met vijf waarden van gamma, drie scènes van 2.720 anchors, en of het trainen je loss kan terugvolgen, zoals `loss.backward()` doet. Als hij klopt, bewaart de check `focal_loss` in je project, als `work/projects/cnn/focal.py`.
        """),
        L.task("Laat `focal_loss` de focal loss van RetinaNet geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_focal():
    box_focal = L.code_box("focal-loss", lines=4)
    box_focal
    return (box_focal,)


@app.cell(hide_code=True)
def check_focal(box_focal):
    result_focal = L.check("focal-loss", box_focal.value)
    result_focal
    return (result_focal,)


@app.cell(hide_code=True)
def hints_focal():
    L.hints(
        "focal-loss",
        [
            "De stappen 2 tot 4 van het uitgewerkte voorbeeld, als functie, met `gamma` in plaats van de 2. Daar koos `kept` de anchors die niet genegeerd worden, gaf `F.cross_entropy(..., reduction=\"none\")` elk ervan zijn cross-entropy, en `torch.exp(-ce)` zijn p_t.",
            "`kept = classes >= 0` en `ce = F.cross_entropy(logits[kept], classes[kept], reduction=\"none\")`: `logits[kept]` geeft de rijen van 4 logits van de anchors die niet genegeerd worden, onder elkaar, voor elk aantal scènes. Dan `p = torch.exp(-ce)`.",
            "De focal loss van elke anchor is `(1 - p) ** gamma * ce`; tel die op, en deel door het aantal anchors dat een vorm leert: `return ((1 - p) ** gamma * ce).sum() / (___).sum()`.",
        ],
        why="`classes > 0` is True voor elke anchor die een vorm leert, en `.sum()` telt ze: in de zes anchors van het uitgewerkte voorbeeld 2, de cirkel en het vierkant. De achtergrond-anchors worden ook opgeteld in de focal loss, zo leert de detector achtergrond, maar ze tellen niet mee in het getal waardoor hij gedeeld wordt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Beginnen bij achtergrond

        Maak `background_bias(pi, kinds)` af, stap 5 van het recept: `pi` is de kans op een vorm waarmee elke anchor moet beginnen, en `kinds` het aantal soorten naast achtergrond. Hij geeft de bias $b$ van de achtergrondlogit terug waarmee achtergrond een kans van $1 - \pi$ krijgt van softmax over $b$ en een 0 voor elke soort. Softmax geeft achtergrond $e^b / (e^b + \text{kinds})$, omdat $e^0 = 1$ voor elke soort: stel dat gelijk aan $1 - \pi$, en los het op naar $b$. De check probeert $\pi = 0{,}01$ met 3 soorten, vier andere paren, en softmax over de bias en een 0 voor elk van 4 soorten. Als hij klopt, bewaart de check `background_bias` in je project, als `work/projects/cnn/background.py`; de demo laat de head van de detector ermee beginnen.
        """),
        L.task("Laat `background_bias` de bias geven waarmee elke anchor bij achtergrond begint."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bias():
    box_background_bias = L.code_box("background-bias", lines=2)
    box_background_bias
    return (box_background_bias,)


@app.cell(hide_code=True)
def check_bias(box_background_bias):
    result_background_bias = L.check("background-bias", box_background_bias.value)
    result_background_bias
    return (result_background_bias,)


@app.cell(hide_code=True)
def hints_bias():
    L.hints(
        "background-bias",
        [
            "Schrijf de vergelijking uit: $e^b / (e^b + \\text{kinds}) = 1 - \\pi$. Vermenigvuldig beide kanten met $e^b + \\text{kinds}$, en breng de termen met $e^b$ naar één kant.",
            "$e^b = (1 - \\pi)\\,e^b + (1 - \\pi)\\,\\text{kinds}$, dus $\\pi\\,e^b = \\text{kinds}\\,(1 - \\pi)$, en $e^b = \\text{kinds}\\,(1 - \\pi) / \\pi$. Met $\\pi = 0{,}01$ en 3 soorten is dat 297.",
            "$b$ is de logaritme van $e^b$: `return math.log(___)`.",
        ],
        why="Met $\\pi = 0{,}01$ en 3 soorten is $b = \\log 297 \\approx 5{,}694$. Softmax over (5,694; 0; 0; 0) geeft achtergrond 297 / (297 + 3) = 0,99, en elke soort 1 / 300, ongeveer 0,0033: samen de 0,01 van $\\pi$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_versus():
    mo.md(r"""
    ### Oefening 4. Alles of een gewicht
    """)
    return


@app.cell(hide_code=True)
def choice_versus_cell():
    choice_versus = L.choice(
        "versus",
        question="Hard negative mining en de focal loss zorgen allebei dat de loss van een one-stage detector niet verdrinkt in achtergrond. Een scène heeft 10 anchors die een vorm leren en 2.697 achtergrond-anchors. Hoe telt elk van de twee de achtergrond-anchors mee?",
        options={
            "right": "Hard negative mining telt de 30 moeilijkste helemaal mee en de andere helemaal niet; de focal loss telt alle 2.697 mee, elk gewogen met hoeveel de detector er nog naast zit",
            "reverse": "Hard negative mining weegt elke achtergrond-anchor; de focal loss houdt alleen de moeilijkste over",
            "same": "Allebei tellen ze de 30 moeilijkste achtergrond-anchors; de focal loss zet ook de anchors van vormen omlaag",
            "random": "Hard negative mining kiest er 30 willekeurig; de focal loss telt ze allemaal even zwaar",
        },
        correct="right",
        why={
            "right": "Hard negative mining maakt een knip: een anchor telt helemaal mee of helemaal niet, en welke 30 dat zijn verandert van batch tot batch. De focal loss maakt geen knip: elke anchor telt mee, met een gewicht van (1 − p_t)^γ, ook de anchors van vormen waar de detector al zeker van is. Op COCO deed de focal loss het beter; op de scènes van de course vinden de twee ongeveer evenveel vormen.",
            "reverse": "Het is andersom: hard negative mining houdt de 30 moeilijkste achtergrond-anchors over en laat de andere vallen; de focal loss houdt elke anchor, gewogen met (1 − p_t)^γ.",
            "same": "De focal loss kiest geen achtergrond-anchors: hij telt alle 2.697 mee, de makkelijke met een piepklein gewicht. Hard negative mining telt de 30 moeilijkste.",
            "random": "Hard negative mining neemt de achtergrond-anchors met de hoogste cross-entropy, geen willekeurige; en de focal loss telt ze niet even zwaar: een makkelijke telt veel minder mee dan een moeilijke.",
        },
    )
    choice_versus
    return (choice_versus,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint de FPN-detector van je vorige les met je `focal_loss`, $\gamma = 2$, met elke anchor beginnend bij achtergrond via je `background_bias` en $\pi = 0{,}01$, 8 epochs lang op de 2.000 trainingsscènes met kleine en grote vormen, in mini-batches van 32, zoals hij met hard negative mining getraind werd. Hij bouwt de detector met je `pyramid` en `head_rows`, en vindt de vormen van de 200 testscènes met je `boxes_from` en NMS. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; dat duurt een paar minuten. De grafiek laat na elke epoch de loss op de testscènes zien. Als hij klaar is, kun je door de epochs terugspoelen, en zien welke boxes de detector na elke epoch aan drie testscènes gaf.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train met de focal loss")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_focal(train_button, result_focal, result_background_bias):
    result_focal, result_background_bias  # opnieuw uitvoeren als een oefening je functie bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om de detector met je focal loss te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Focal loss: de loss op de testscènes", "x": "epoch", "y": "test loss",
                                        "alt": "De loss van de detector op de 200 testscènes na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Focal loss** · epoch {_p['epoch']} van {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**Focal loss** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `focal_loss` en je `background_bias` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan tien minuten. Doet je `focal_loss` meer werk dan nodig, zoals een loop over de anchors?*",
                "focal": "*Je `focal_loss` in `work/projects/cnn/focal.py` stopt met een fout, of geeft geen getal dat het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "background": "*Je `background_bias` in `work/projects/cnn/background.py` stopt met een fout, of geeft geen getal. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "pyramid": "*Je `pyramid` in `work/projects/cnn/pyramid.py` stopt met een fout, of geeft geen vier niveaus van 32 maps die het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een feature pyramid voor kleine en grote voorwerpen](/course/open/cv-fpn?lang=nl) opnieuw.*",
                "head_rows": "*Je `head_rows` in `work/projects/cnn/head_rows.py` stopt met een fout, of geeft voor een scène geen 2.720 rijen van 8 die het trainen kan terugvolgen. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een feature pyramid voor kleine en grote voorwerpen](/course/open/cv-fpn?lang=nl) opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van de detector. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course nam de plaats in van " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van de detector na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _words = {"scene": ("scène {seed}: {n} box", "scène {seed}: {n} boxes"), "alt": "Drie testscènes met kleine en grote vormen, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die de detector haar na die epoch gaf, na NMS."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"Na epoch {_e}: test loss {L.number(_val[_e - 1], 3)}, en {_n} {'box' if _n == 1 else 'boxes'} van 0,5 of meer op deze drie scènes, die {C.SHOWN_SHAPES} vormen hebben.")

            if len(_moments) == len(_val) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: na welke epoch heeft elke vorm van de drie scènes een box?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _found, _shapes = _result["found"], _result["shapes"]
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("cirkels", "vierkanten", "driehoeken"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Beginnend bij achtergrond met een bias van {L.number(_result['bias'], 3)} en getraind met je focal loss vindt de detector {sum(_found)} van de {sum(_shapes)} vormen van de 200 testscènes: een box van hun soort, met een score van 0,5 of meer en een IoU boven 0,5. Dat zijn {_found[0]} van de {_shapes[0]} kleine vormen, {_found[1]} van de {_shapes[1]} middelgrote en {_found[2]} van de {_shapes[2]} grote. {_result['false']} van zijn boxes van 0,5 of meer vinden geen vorm. Zijn mAP bij een IoU van 0,5 is {L.number(_result['map'], 3)}, met een AP van {_aps}; bij 0,75 is hij {L.number(_result['map75'], 3)}. De detector van de course, op dezelfde manier getraind met de seeds 0 tot 4, vindt 723 tot 734 vormen, met 95 tot 109 valse boxes en een mAP van 0,944 tot 0,951; met hard negative mining vond hij er 727 tot 741, met 104 tot 114 valse boxes.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gamma():
    mo.md(r"""
    ### Oefening 5. Waarom vindt een grote γ minder vormen?
    """)
    return


@app.cell(hide_code=True)
def explain_gamma_cell():
    explain_gamma = L.explain(
        "why-gamma",
        question="Met γ = 5 geeft de detector van de course de minste valse boxes, 71 tot 81, maar hij vindt ook de minste vormen, 706 tot 711 van de 786, tegen 723 tot 734 met γ = 2. Waarom vindt hij met een grote γ minder vormen, en geeft hij minder valse boxes? Denk aan een anchor van een vorm waarvan de detector de soort een kans van 0,6 geeft, en aan de score van 0,5 die een box nodig heeft.",
        model_answer="Met γ = 5 daalt het gewicht (1 − p_t)⁵ heel snel: bij p_t = 0,6 is het 0,4⁵, ongeveer 0,01, dus een anchor waarvan de detector de klasse een kans van 0,6 geeft, telt maar voor 1% van zijn cross-entropy mee, en de loss maakt hem nauwelijks zekerder. Met γ = 2 telt hij nog voor 16% mee, genoeg om hem zekerder te maken. Met γ = 5 blijft de detector dus overal minder zeker van de soorten: op de testscènes eindigen de anchors die een vorm leren bij een gemiddelde p_t van 0,66, tegen 0,77 met γ = 2. Een box heeft een score van 0,5 of meer nodig om mee te tellen, dus de vormen waarvan de beste anchor eronder blijft, worden gemist. Voor de achtergrond geldt hetzelfde: minder achtergrond-anchors krijgen een score van 0,5 of meer voor een soort, dus er zijn minder valse boxes.",
        points=[
            "je zegt dat met γ = 5 het gewicht (1 − p_t)⁵ bij p_t = 0,6 al ongeveer 1% is, zodat de loss redelijk zekere anchors niet meer zekerder maakt",
            "je zegt dat een box een score van 0,5 nodig heeft, zodat vormen waarvan de anchors rond of onder die grens blijven niet gevonden worden",
            "je zegt dat de detector overal minder zeker blijft van de soorten, zodat ook minder achtergrond-anchors een score van 0,5 voor een soort halen: minder valse boxes",
        ],
    )
    explain_gamma
    return (explain_gamma,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De cross-entropy die je one-stage detector in [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) over zijn anchors middelde, krijgt nu een gewicht voor elke anchor, $(1 - p_t)^\gamma$, klein voor de anchors die de detector al goed doet, en wordt gedeeld door de anchors die een vorm leren. Net als hard negative mining in [Een feature pyramid voor kleine en grote voorwerpen](/course/open/cv-fpn?lang=nl) zorgt hij dat de paar anchors die een vorm leren niet verdrinken in achtergrond, maar zonder knip: elke anchor telt mee, zoveel als de detector er nog naast zit. De two-stage detector van [Van R-CNN naar Faster R-CNN: twee fasen](/course/open/cv-rcnn?lang=nl) hield de achtergrond buiten met een eerste fase; RetinaNet doet het met een loss, en was de eerste one-stage detector die voorwerpen even goed vond als de two-stage detectors. Beginnend bij achtergrond via je `background_bias` vindt de FPN-detector met je `focal_loss` evenveel vormen als met hard negative mining, met iets minder valse boxes. Je project `cnn` heeft nu `focal.py` en `background.py`. In de volgende les, *Een soort voor elke pixel: FCN en transposed convolutions*, geeft een netwerk elke pixel van een scène een soort, in plaats van elke anchor.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_easy, result_focal, result_background_bias, choice_versus, explain_gamma):
    L.wrap_up(choice_easy, result_focal, result_background_bias, choice_versus, explain_gamma)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
