# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-pretrained-detector. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-pretrained-detector", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl) en over softmax, uit [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_map():
    quick_map = L.choice(
        "quick-map",
        question="Een detector heeft een AP van 0,8 voor katten. Bij welke score threshold moet een app zijn kattenboxes laten zien?",
        options={
            "right": "Dat zegt de AP niet: die beoordeelt de volgorde van de scores, niet waar je de lijst afkapt",
            "ap": "bij 0,8, zijn AP",
            "iou": "bij 0,5, de IoU-drempel van PASCAL VOC",
            "rest": "bij 0,2: 1 min zijn AP",
        },
        correct="right",
        why={
            "right": "AP loopt alle boxes af vanaf de hoogste score en telt de precision bij elke true positive op: hij beoordeelt de volgorde van de scores, bij elke drempel tegelijk. Waar je de lijst afkapt, moet de app kiezen. Over die keuze gaat deze les.",
            "ap": "AP is de oppervlakte onder de precision-recall curve, geen score van een box. Hij kijkt naar elke drempel tegelijk, en kiest er geen.",
            "iou": "De IoU-drempel bepaalt wanneer een box goed genoeg op een echte box past. Welke boxes een app laat zien, is een drempel op een ander getal: de score. Die kiest AP niet.",
            "rest": "AP is de oppervlakte onder de precision-recall curve, geen score van een box: 1 min die zegt niets over waar je de lijst afkapt.",
        },
    )
    quick_map
    return (quick_map,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Een model geeft drie logits: 3 voor kat, 2 voor hond en 1 voor \"niets\". Welke kans geeft softmax aan kat?",
        options={
            "right": "0,67",
            "two": "0,73",
            "share": "0,50",
            "max": "1,00",
        },
        correct="right",
        why={
            "right": "Softmax neemt e tot de macht van elke logit en deelt door hun som: e³ / (e³ + e² + e¹) = 20,1 / 30,2 = 0,67. De detector van deze les doet hetzelfde voor elk van zijn boxes, ook met een soort voor \"niets\" tussen zijn soorten.",
            "two": "Dan laat je de logit van \"niets\" weg: e³ / (e³ + e²) = 0,73. Softmax deelt door de som over alle logits, ook die van \"niets\".",
            "share": "Dan deel je de logits zelf: 3 / (3 + 2 + 1). Softmax neemt eerst e tot de macht van elke logit.",
            "max": "Softmax geeft elke logit een deel, het grootste aan de grootste logit, maar nooit alles: hond en \"niets\" houden een stukje.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een detector die iemand op een grote set foto's met boxes getraind heeft, vindt met een paar regels code ook de voorwerpen op jouw foto's: een box om elk voorwerp, een soort en een score. Zelfrijdende auto's, camera's die bezoekers tellen en apps die je foto's ordenen beginnen allemaal bij zo'n detector. Maar hij geeft boxes met elke score, van bijna 0 tot bijna 1, en jij moet beslissen welke je gelooft. En hij kent alleen de soorten die hij geleerd heeft: wat maakt hij van iets wat hij nooit gezien heeft?

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder heeft een gevoeligheid. Eén die afgaat bij de zwakste rooklucht, gaat af bij elke boterham in de broodrooster; één die wacht op dikke rook, mist een brand die klein begint. Waar je hem op zet, bepaalt welke vergissing hij vaker maakt. En bij een brand die geen rook maar een ander gas geeft, helpt geen enkele stand.
    """)
    return


@app.cell(hide_code=True)
def predict_starfish():
    guess_starfish = L.guess(
        "how-many",
        question="De detector van deze les leerde 80 soorten dingen, waaronder stoelen, maar geen zeesterren. Je geeft hem 30 foto's van zeesterren, en houdt alleen de boxes met een score boven 0,9. Op hoeveel van de 30 foto's tekent hij dan nog een box?",
        options={
            "none": "op geen enkele: hij kent geen zeesterren",
            "few": "op 2 of 3",
            "part": "op ongeveer 13",
            "all": "op alle 30",
        },
        reveal="**Op 13 van de 30.** Een zeester is geen van zijn soorten, dus elke box op deze foto's is fout. Toch tekent hij op 13 foto's een box met een score boven 0,9: een snowboard met 0,999, een teddybeer en een vogel met 0,996. Een detector kent alleen zijn eigen soorten. Bij iets anders geeft hij de soort die hij kent en die er het meest op lijkt, en een hoge score maakt dat niet goed.",
    )
    guess_starfish
    return (guess_starfish,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Deze les gebruikt **DETR** (detection transformer), een detector die Facebook AI Research, nu Meta, in 2020 uitbracht met de paper van Carion en anderen. Hij leerde van COCO, een set van 118.000 foto's met een box om elk voorwerp van 80 soorten: auto's, katten, kopjes, stoelen, mensen enzovoort. Op de 5000 foto's die COCO apart houdt om detectors op te testen, is zijn AP, gemiddeld over de tien IoU-drempels zoals in [Een detector beoordelen](/course/open/cv-map?lang=nl), 0,42, met de interpolatie van COCO. De course schreef DETR uit in zijn eigen code, `everything_ai/detr.py`, met de 41,5 miljoen parameters die Meta publiceerde.

        DETR werkt in drie delen. Een ResNet-50, een grotere versie van de ResNet-18 uit [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), maakt van de foto features, één rij getallen voor elk vierkant van 32 bij 32 pixels. Een transformer, die latere lessen uitleggen, laat 100 **object queries** naar die features kijken. Elke query is een plek voor één voorwerp: hij geeft één box en 92 logits. De logits zijn voor de nummers 0 tot 90, die COCO aan zijn soorten geeft, en de laatste voor **no object**. De 80 soorten van COCO hebben nummers van 1 tot 90, waarvan er 10 ongebruikt bleven, en ook 0 wordt niet gebruikt: voor die 11 nummers zag DETR nooit een box, dus heeft hij nooit geleerd ze te zeggen.

        Voor één query maakt softmax van de 92 logits kansen. Zijn soort is de waarschijnlijkste van de eerste 91, en zijn **score** is die kans: de laatste kans, die van *no object*, doet niet mee. Op de koffiefoto hebben drie queries een hoge score; de andere 97 geven het grootste deel van hun kans aan *no object*, en hun score is laag. De box is vier getallen: `(cx, cy, w, h)`, het midden opzij en omlaag en de breedte en de hoogte, elk als deel van de breedte of de hoogte van de foto. Oefening 2 rekent hem om naar pixels.

        Welke boxes houd je? Die met een score boven een **score threshold**. Een lage drempel houdt boxes waar de detector niet zeker van is: hij mist minder voorwerpen, en laat meer foute boxes zien. Een hoge drempel laat minder foute boxes zien, en mist meer voorwerpen. Net als in [Een detector beoordelen](/course/open/cv-map?lang=nl) kiest de AP hem niet: jij kiest hem naar wat elke vergissing in je app kost.

        Nog twee dingen zijn nieuw. DETR heeft geen non-maximum suppression nodig. Tijdens het trainen werd elke echte box aan precies één query gekoppeld, en moest elke andere query leren *no object* te zeggen; zo krijgt elk voorwerp meestal één box, waar de detector uit [Eén box per voorwerp](/course/open/cv-nms?lang=nl) er veel tekende. En DETR vindt alleen zijn eigen soorten. Op een foto van iets anders geeft hij de soort die hij kent en die er het meest op lijkt: geen score threshold laat hem een soort vinden die hij nooit geleerd heeft.
        """),
        L.watch_out("Laat de laatste kans, die van *no object*, weg als je de hoogste neemt. Mét die kans kiezen 97 van de 100 queries op de koffiefoto voor *no object*, elk met meer dan 0,7: boven een drempel van 0,5 zou je 97 boxes van niets houden."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **object query** | een van de 100 plekken van DETR voor een voorwerp: elk geeft één box en de logits van zijn soort | `logits[0, i]`, `boxes[0, i]` |
        | **no object** | de laatste soort van DETR: wat een query zegt als hij geen voorwerp heeft | `p[-1]` |
        | **score threshold** | de score waar een box boven moet liggen om te blijven; een hogere laat minder foute boxes zien en mist meer voorwerpen | `threshold = 0.9` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_course():
    mo.vstack([
        mo.md("**DETR op de foto's van de course, elke box boven 0,5**"),
        mo.Html(C.course_picture({
            "titles": ["koffie", "Chelsea", "fruit (een tekening)", "een cel"],
            "alt": "De vier foto's van de course met de boxes van DETR boven een score van 0,5, elk met zijn soort en score. De koffiefoto: cup 1,00, dining table 0,99 en spoon 0,98. Chelsea: cat 1,00, en carrot 0,86 op haar oranje neus. De tekening van fruit: zeven boxes van sports ball, één op elk stuk fruit. De cel onder de microscoop: clock 1,00.",
        }, "nl")),
        mo.md(r"""
        DETR op de vier foto's van de course, met elke box boven een score van 0,5; de labels ronden de scores af op twee decimalen. DETR noemt zijn soorten in het Engels, zoals COCO. Op de koffiefoto: een kopje (cup) met 0,997, de tafel (dining table) met 0,993 en een lepeltje (spoon) met 0,98. Op Chelsea: de kat, met 0,996, en een wortel (carrot), met 0,86, op haar oranje neus. De 7 stukken fruit van de tekening worden 7 ballen (sports ball), hoewel COCO appels en sinaasappels heeft: getekend, rond en glanzend lijken ze voor DETR meer op ballen. De cel onder de microscoop wordt een klok, met 0,997. DETR kent 80 soorten dingen, en wat hij ook ziet, hij noemt het een daarvan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Wat doet de score threshold? De course gaf DETR 30 foto's van windsorstoelen en 30 foto's van zeesterren, uit Caltech-101, een verzameling foto's van 101 soorten dingen. DETR kent stoelen; zeesterren kent hij niet, dus elke box op een zeesterfoto is fout. Met de schuif zet je de score threshold. De doorgetrokken lijn telt de stoelfoto's waarop DETR een box van de soort chair tekent, de gestreepte lijn de zeesterfoto's waarop hij welke box dan ook tekent.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _svg, _found, _star = C.threshold_frame(t, {"x": "score threshold", "y": "foto's, van 30",
                                                     "chairs": "stoelfoto's met een stoelbox",
                                                     "starfish": "zeesterfoto's met een box",
                                                     "alt": "Twee lijnen tegen de score threshold, van 0,05 tot 0,99. De doorgetrokken lijn, de stoelfoto's met een box van de soort chair, daalt langzaam van 27 naar 12; de gestreepte lijn, de zeesterfoto's met welke box dan ook, daalt van 30 naar 4. Een gestreepte staande lijn markeert de drempel van {t}: {found} stoelfoto's met een stoelbox, en {starfish} zeesterfoto's met een box."}, "nl")
        return _svg, f"Score threshold {L.number(t, 2)}: een stoelbox op {_found} van de 30 stoelfoto's, en een box op {_star} van de 30 zeesterfoto's."

    explore_threshold = L.explore("threshold", "score threshold", _frame, start=0.5, values=C.STOPS, role="hyper",
                                  task="Zoek de laagste score threshold waarbij hoogstens 10 van de 30 zeesterfoto's een box krijgen.")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 0,05 vindt DETR een stoel op 27 van de 30 stoelfoto's, en tekent hij een box op alle 30 zeesterfoto's. Als de drempel stijgt, dalen ze allebei: 23 en 23 bij 0,5, 20 en 13 bij 0,9, en 19 en 9 bij 0,95. Dus 0,95 is de laagste drempel waarbij hoogstens 10 zeesterfoto's een box krijgen. In boxes geteld krijgen de zeesterfoto's 39 boxes bij 0,5 en 14 bij 0,9, en elk daarvan is fout. Bij 0,99 tekent DETR nog op 4 zeesterfoto's een box, een snowboard met 0,999, een teddybeer en twee vogels, terwijl hij nog maar op 12 stoelfoto's een stoel vindt. En 3 stoelfoto's krijgen zelfs bij 0,05 geen stoelbox. Een drempel ruilt gemiste stoelen in tegen foute boxes; hij kan DETR geen soort laten vinden die hij nooit geleerd heeft.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de voorwerpen op een foto te vinden met DETR, `detr`:

    1. Maak de foto klaar: `x = prepare(photo)`, met shape `(1, 3, h, w)`.
    2. De queries: `logits, boxes = detr(x)`, binnen `with torch.no_grad():`; voor elk van de 100 queries 92 logits en een box.
    3. Voor query `i` maakt softmax van zijn logits kansen `p`, zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl).
    4. Zijn soort is `k = np.argmax(p[:-1])`, zonder de laatste, *no object*, en zijn score is `p[k]`.
    5. Houd de query als zijn score boven de drempel ligt, `p[k] > threshold`. Zijn box is `boxes[0, i]`, die `to_pixels` uit oefening 2 omrekent naar pixels.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel print hoeveel soorten DETR heeft, die met nummer 47, en de laatste. Hij maakt de koffiefoto klaar, stap 1, en print zijn shape. `prepare` uit `everything_ai.detr` van de course maakt de kortste zijde 800 pixels, de grootte waarop DETR op COCO getest wordt, houdt de langste zijde op hoogstens 1333, en normaliseert met het gemiddelde en de std van ImageNet, zoals in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl). Hij snijdt niets bij: een detector zoekt voorwerpen op de hele foto. Dan geeft de cel de foto aan DETR, stap 2, en print hij de shapes van de logits en de boxes. Voor elke query doet hij stap 3 tot 5 met een drempel van 0,9, en print hij het nummer van elke query die hij houdt, zijn soort, zijn score en zijn box. DETR doet een seconde of langer over een foto. `detr` is DETR, die de cel hieronder laadt zodra hij gedownload is, samen met de foto's van Caltech-101 voor de demo.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="DETR en Caltech-101 downloaden, 304 MB")
    return (get_button,)


@app.cell(hide_code=True)
def detr_cell(get_button):
    detr = C.detr_or_stop(get_button, {
        "ask": "*DETR en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: DETR van Hugging Face, 167 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's uitpakt. Het uitgewerkte voorbeeld en de demo hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 304 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*DETR en de foto's staan op je computer: 41,5 miljoen parameters, getraind op de 118.000 foto's van COCO.*",
    })
    return (detr,)


@app.cell
def example_coffee(detr):
    import numpy as np
    import torch
    from PIL import Image
    from everything_ai.detr import KINDS, prepare

    print(len(KINDS), KINDS[47], KINDS[-1])
    # 1. de kortste zijde 800 pixels, genormaliseerd zoals voor ImageNet
    photo = Image.open("data/photos/coffee.png")
    x = prepare(photo)
    print(x.shape)
    # 2. voor elk van de 100 queries: 92 logits en een box
    with torch.no_grad():
        logits, boxes = detr(x)
    print(logits.shape, boxes.shape)
    threshold = 0.9
    for i in range(100):
        # 3. softmax over de 92 logits van query i
        z = logits[0, i].numpy()
        e = np.exp(z - z.max())
        p = e / e.sum()
        # 4. de soort en zijn score, zonder de laatste: no object
        k = np.argmax(p[:-1])
        # 5. houd de query als zijn score boven de drempel ligt
        if p[k] > threshold:
            print(i, KINDS[k], p[k].round(3), boxes[0, i].numpy().round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    DETR heeft 92 soorten: nummer 47 is cup, een kopje, en de laatste is *no object*. De klaargemaakte foto heeft shape `(1, 3, 800, 1200)`: de foto is 600 bij 400 pixels, dus zijn kortste zijde verdubbelt tot 800, en de langste zijde mee. De logits hebben shape `(1, 100, 92)`, en de boxes `(1, 100, 4)`: een batch van één foto, 100 queries. Bij 0,9 houden drie queries hun box: query 59 een tafel (dining table), met 0,993, query 71 het kopje, met 0,997, en query 92 een lepeltje (spoon), met 0,98. De box van het kopje is `[0.486 0.381 0.397 0.674]`: zijn midden ligt op 0,486 van de breedte en op 0,381 van de hoogte, en hij is 0,397 van de foto breed en 0,674 hoog. De andere 97 queries geven *no object* de hoogste kans. Een daarvan geeft nog een tweede box, op het bolletje van het lepeltje, 0,293: onder de drempel. Boven 0,9 liggen er geen twee boxes op hetzelfde voorwerp, en dat zonder NMS.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De score van een query
    """)
    return


@app.cell(hide_code=True)
def choice_score_cell():
    choice_score = L.choice(
        "score",
        question="Na softmax geeft een query van DETR kat 0,30, hond 0,25 en *no object* 0,40; de andere soorten delen de overige 0,05. Wat gebeurt er met zijn box bij een score threshold van 0,2?",
        options={
            "right": "Hij blijft, als kat met een score van 0,30",
            "nothing": "Hij valt weg: *no object* is het waarschijnlijkst, met 0,40",
            "kinds": "Hij blijft, als kat met een score van 0,60: de soorten samen",
            "both": "Hij blijft twee keer, als kat en als hond: allebei boven 0,2",
        },
        correct="right",
        why={
            "right": "De score is de hoogste kans onder de soorten vóór de laatste: kat, met 0,30, en dat ligt boven 0,2. Dat *no object* meer heeft, 0,40, telt niet mee. Zo houdt een lage drempel boxes waarvan de query zelf niet zeker is dat er een voorwerp in zit; bij 0,5 zou deze wegvallen.",
            "nothing": "De score laat de laatste kans weg, die van *no object*: hij is de hoogste kans van een soort. Welke soort is dat, en ligt die boven 0,2?",
            "kinds": "De score is de kans van één soort, de waarschijnlijkste, niet van alle soorten samen. Welke soort is het waarschijnlijkst, en met welke kans?",
            "both": "Een query geeft één box, met één soort: de waarschijnlijkste. Zijn score is de kans van die soort.",
        },
    )
    choice_score
    return (choice_score,)


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Omrekenen naar pixels

        DETR geeft een box als `(cx, cy, w, h)`, elk een deel van de foto; de course schrijft een box als `(top, left, bottom, right)` in pixels, zoals in [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl). Daar waren bottom en right de laatste rij en kolom van de box. DETR meet randen: hier zijn bottom en right de lijnen waar de box ophoudt, gemeten vanaf de hoek linksboven van de foto, dus een box over een hele foto van 600 pixels breed houdt op bij 600. Maak `to_pixels(box, width, height)` af, die de box van DETR en de breedte en de hoogte van de foto krijgt, en de box in pixels teruggeeft. Vervang de vier `...`. De check roept hem aan met het kopje uit het uitgewerkte voorbeeld, op de koffiefoto van 600 bij 400 pixels:

        ```python
        to_pixels((0.486, 0.381, 0.397, 0.674), 600, 400)
        ```

        Dat moet `(17.60, 172.50, 287.20, 410.70)` geven. De check probeert ook andere boxes op andere foto's.
        """),
        L.task("Laat `to_pixels(box, width, height)` de box geven als `(top, left, bottom, right)` in pixels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pixels():
    pixels_code = L.code_box("pixels", lines=8)
    pixels_code
    return (pixels_code,)


@app.cell(hide_code=True)
def check_pixels(pixels_code):
    result_pixels = L.check("pixels", pixels_code.value)
    result_pixels
    return (result_pixels,)


@app.cell(hide_code=True)
def hints_pixels():
    L.hints(
        "pixels",
        [
            "`cx` is het midden opzij, als deel van de breedte. Van daaruit reikt de box de helft van zijn breedte naar links en de helft naar rechts. De boven- en de onderkant gaan net zo, met `cy` en `h`.",
            "Als deel is de linkerkant `cx - w / 2`. Een deel wordt pixels als je het vermenigvuldigt met de grootte van de foto in die richting: wat opzij loopt, de linker- en de rechterkant, met `width`; wat omlaag loopt, de boven- en de onderkant, met `height`.",
            "`top = (cy - h / 2) * ___`, `left = (cx - w / 2) * ___`, en `bottom` en `right` net zo met `+`.",
        ],
        why="Bij het kopje ligt het midden op 0,381 van de hoogte en is de box 0,674 hoog, dus hij loopt van 0,381 - 0,337 = 0,044 tot 0,718 van de hoogte: van rij 17,6 tot rij 287,2 van 400. Opzij loopt hij van 0,486 - 0,1985 = 0,2875 tot 0,6845 van de breedte: van kolom 172,5 tot kolom 410,7 van 600.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw functie op nieuwe foto's

    Zodra DETR en de foto's gedownload zijn, in het uitgewerkte voorbeeld, geeft de cel hieronder DETR 8 foto's uit Caltech-101: 4 van de windsorstoelen en 4 van de zeesterren van de schuif, met hun nummer binnen de 30. Hij tekent elke box boven de score threshold die jij zet, met jouw `to_pixels` zodra die klopt, en anders met die van de course. De 8 foto's kosten een paar seconden, één keer.
    """)
    return


@app.cell(hide_code=True)
def demo_slider_cell():
    demo_slider = L.slider("demo", "score threshold", start=0.9, low=0.3, high=0.95, step=0.05, role="hyper")
    demo_slider
    return (demo_slider,)


@app.cell(hide_code=True)
def demo_boxes(demo_slider, detr, pixels_code, result_pixels):
    _t = demo_slider.widget.value
    _d = C.demo(detr, pixels_code.value if result_pixels.state == "correct" else None, _t, {
        "titles": ["stoel 0", "stoel 2", "stoel 13", "stoel 20", "zeester 1", "zeester 4", "zeester 12", "zeester 21"],
        "alt": f"Acht foto's uit Caltech-101 in twee rijen: vier windsorstoelen, met de nummers 0, 2, 13 en 20, en vier zeesterren, met de nummers 1, 4, 12 en 21, met elke box van DETR boven een score threshold van {L.number(_t, 2)}, elk met zijn soort en score. DETR heeft geen soort zeester, dus elke box op een zeesterfoto is fout.",
    }, "nl")
    _who = "jouw `to_pixels`" if _d["yours"] else "de `to_pixels` van de course"
    mo.vstack([
        mo.Html(_d["picture"]),
        mo.md(f"Getekend met {_who}. Bij een score threshold van {L.number(_t, 2)} krijgen {_d['chairs']} van de 4 stoelfoto's een stoelbox, en {_d['starfish']} van de 4 zeesterfoto's een box."),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Zet de drempel omlaag naar 0,3, en er verschijnen meer boxes: vliegtuigen (airplane) en een stoel in het zand van zeester 1, een teddybeer en een koffer (suitcase) op zeester 12, en nog drie katten op zeester 21. Op stoel 13 tekent DETR drie stoelboxes op dezelfde stoel, met 0,88, 0,81 en 0,36; de eerste twee overlappen met een IoU van 0,83. Doordat elk voorwerp tijdens het trainen aan één query gekoppeld werd, zijn zulke dubbele boxes zeldzaam, maar niet onmogelijk; NMS zou ze weghalen. Op stoel 0 is de donkere zitting ook een kat, met 0,98: een foute box die geen drempel tot 0,95 weghaalt. Stoel 20 krijgt helemaal geen box: het is een van de 3 stoelfoto's waarop DETR nooit een stoel vindt. En de getekende zeester 4 is bij elke drempel een snowboard, met 0,999.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_apps():
    mo.md(r"""
    ### Oefening 3. Welke drempel waar
    """)
    return


@app.cell(hide_code=True)
def choice_apps_cell():
    choice_apps = L.choice(
        "apps",
        question="Vier apps gebruiken dezelfde detector. Welke moet zijn score threshold laag zetten?",
        options={
            "right": "Een programma dat scheuren markeert op foto's van een brug, voor een inspecteur die elke markering controleert",
            "album": "Een app die onder je vakantiefoto's schrijft wat hij ziet",
            "shop": "Een camera bij een zelfscankassa die elk product dat hij ziet op je rekening zet",
            "fine": "Een camera bij een busbaan die een boete stuurt voor elke auto die hij daar ziet",
        },
        correct="right",
        why={
            "right": "Een gemiste scheur kan gevaarlijk zijn, en een foute markering kost de inspecteur één blik. Een lage drempel mist minder scheuren en laat meer foute markeringen zien, en hier is dat de goede ruil.",
            "album": "Een fout woord onder een foto is hier de vergissing die mensen stoort, en een gemist voorwerp kost weinig: dat vraagt om een hoge drempel. Welke app verliest het meest als hij iets mist?",
            "shop": "Een foute box kost de klant geld: dat vraagt om een hoge drempel. Welke app verliest het meest als hij iets mist, terwijl een foute box weinig kost?",
            "fine": "Een foute box is een boete voor iemand die niets verkeerd deed: dat vraagt om een hoge drempel. Welke app verliest het meest als hij iets mist, terwijl een foute box weinig kost?",
        },
    )
    choice_apps
    return (choice_apps,)


@app.cell(hide_code=True)
def text_exercise_starfish():
    mo.md(r"""
    ### Oefening 4. Zeesterren tellen
    """)
    return


@app.cell(hide_code=True)
def explain_starfish_cell():
    explain_starfish = L.explain(
        "starfish",
        question="Iemand bouwt een app die de zeesterren telt op foto's van getijdenpoelen, met DETR en een heel hoge score threshold, 0,99, zodat de boxes die overblijven kloppen. Waarom werkt dat niet, en wat is er in plaats daarvan nodig?",
        model_answer="**DETR heeft geen soort zeester.** Hij leerde alleen de 80 soorten van COCO, dus elke box die hij op een zeester tekent, draagt de naam van een andere soort, en sommige van die boxes hebben heel hoge scores: bij 0,99 tekent DETR nog op 4 van de 30 zeesterfoto's een box, een snowboard met 0,999 en een teddybeer met 0,996. Een hoge drempel haalt alleen boxes weg waar DETR minder zeker van is; hij kan DETR geen soort laten vinden die hij nooit geleerd heeft, en een box waar hij zeker van is, kan toch fout zijn. **Wat wel werkt**, is een detector die zeesterren geleerd heeft: een die getraind is op foto's van zeesterren met een box om elke zeester, bijvoorbeeld DETR die op zulke foto's verder getraind is, zoals latere lessen doen.",
        points=[
            "je zegt dat DETR alleen de soorten van COCO leerde, en dat een zeester daar niet bij is",
            "je zegt dat DETR zeesterren toch hoge scores geeft voor foute soorten, dus dat een hoge drempel niet elke foute box weghaalt, en dat geen drempel hem zeesterren laat vinden",
            "je zegt dat de app een detector nodig heeft die getraind is op foto's van zeesterren met hun boxes",
        ],
    )
    explain_starfish
    return (explain_starfish,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl) beoordeelde AP de volgorde van de scores van een detector, en liet hij de drempel aan de app over. Hier tekent een echte detector, DETR, getraind op COCO, boxes op foto's: een ResNet-50, zoals in [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl) maar groter, maakt features, en 100 object queries geven elk een box en de logits van 92 soorten, de laatste voor *no object*. Softmax geeft, net als bij een classifier, elke query zijn soort en score, zonder *no object*; omdat elk voorwerp meestal één query krijgt, heeft DETR geen NMS nodig. Een score threshold bepaalt welke boxes je houdt: van een stoel op 27 van de 30 stoelfoto's en een foute box op alle 30 zeesterfoto's bij 0,05, tot 12 en 4 bij 0,99. En DETR kent alleen zijn eigen soorten: een zeester wordt een snowboard met 0,999. Met deze les is het eerste deel van computer vision af, van de pixels in een bestand tot een detector die voorwerpen op foto's vindt. Zelf een detector trainen, en hoe de transformer in DETR werkt, komen in latere lessen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_apps, choice_score, explain_starfish, result_pixels):
    L.wrap_up(choice_score, result_pixels, choice_apps, explain_starfish)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
