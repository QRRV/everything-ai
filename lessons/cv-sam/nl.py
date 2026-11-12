# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-sam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-sam", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) en [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fcn():
    quick_fcn = L.choice(
        "quick-fcn",
        question="Het FCN van de les over FCN's leerde de scènes van de course, met 4 soorten: achtergrond, cirkel, vierkant en driehoek. Je geeft het een foto van een zeester. Wat geeft het de pixels van de zeester?",
        options={
            "right": "elk een van zijn 4 soorten: andere heeft het niet",
            "star": "een nieuwe soort, zeester: het ziet een nieuwe vorm",
            "empty": "geen soort: die pixels blijven leeg",
            "background": "altijd achtergrond: wat het niet kent, is achtergrond",
        },
        correct="right",
        why={
            "right": "Zijn laatste layer geeft elke pixel 4 scores, een voor elke soort, en de pixel krijgt de soort met de hoogste. Een zeester is geen van alle, maar elke pixel krijgt toch een van de 4. SAM, het model van deze les, heeft helemaal geen soorten: hij tekent het mask van wat je ook aanwijst.",
            "star": "Een FCN kan alleen een soort geven waar het een score voor heeft, en het heeft er 4. Een zeester is geen van alle; welke soort krijgt een pixel dan?",
            "empty": "Elke pixel krijgt de soort met de hoogste van zijn 4 scores, en een ervan is altijd de hoogste.",
            "background": "Achtergrond is een van zijn 4 soorten, en een pixel krijgt die alleen waar die score de hoogste is. Niets stuurt de pixels van een zeester naar achtergrond: ze krijgen de soort van de 4 die het hoogst scoort.",
        },
    )
    quick_fcn
    return (quick_fcn,)


@app.cell(hide_code=True)
def quick_check_erosion():
    quick_erosion = L.choice(
        "quick-erosion",
        question="Erosie met een vierkant van 3 bij 3 maakt een witte vorm rondom één pixel dunner. Je erodeert een wit vierkant van 9 bij 9 pixels, midden in een zwart beeld, steeds opnieuw. Wat blijft er over vlak voordat er niets meer over is?",
        options={
            "right": "de ene pixel in het midden, na 4 erosies",
            "three": "een vierkant van 3 bij 3, na 3 erosies",
            "corners": "zijn vier hoeken",
            "never": "het vierkant verdwijnt nooit: het houdt zijn vorm",
        },
        correct="right",
        why={
            "right": "Elke erosie haalt aan elke kant een pixel weg: 9 bij 9 wordt 7, 5, 3 en dan 1 bij 1, en de vijfde erosie laat niets over. De laatste pixel die overblijft, ligt het diepst in de vorm, het verst van zijn randen. Daar klikt deze les.",
            "three": "Na 3 erosies blijft er een vierkant van 3 bij 3 over, ja. Maar dat is nog niet het laatste: nog één erosie laat zijn middelste pixel over.",
            "corners": "Erosie houdt een pixel alleen waar zijn hele venster van 3 bij 3 wit is. Een hoek heeft zwarte pixels in zijn venster, dus de hoeken gaan als eerste.",
            "never": "Elke erosie maakt het vierkant aan elke kant een pixel kleiner: 9, 7, 5, 3, 1. Wat blijft er over na 4 erosies?",
        },
    )
    quick_erosion
    return (quick_erosion,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een mask met de hand tekenen gaat langzaam: in [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl) deden de annotators van COCO er meer dan 79 seconden over per voorwerp. Een FCN, zoals in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl), tekent zelf masks, maar alleen van de soorten die het leerde, en pas nadat iemand er duizenden tekende om van te leren. SAM, het Segment Anything Model van Meta, draait dat om: je wijst een ding aan met een klik, en hij tekent het mask van dat ding, van elke soort, in een fractie van een seconde. Waar het mask fout is, klik je opnieuw. Zo maakte Meta zijn eigen verzameling masks. Met hulp van SAM tekende een annotator een mask in 14 seconden, 6,5 keer zo snel als voor COCO, en aan het eind tekende SAM 99,1% van de 1,1 miljard masks zelf.

    **Een voorbeeld uit het dagelijks leven.** In een winkel wijs je naar een plank en zeg je: "die graag". De verkoper geeft je misschien het kopje, het kopje met zijn schoteltje, of het hele servies. Wijzen gaat snel, maar het kan meer dan één ding betekenen; een tweede gebaar, "nee, het hele servies", maakt het duidelijk.
    """)
    return


@app.cell(hide_code=True)
def predict_sunflower():
    guess_sunflower = L.guess(
        "sunflower",
        question="Je klikt één keer midden in een zonnebloem, op de pitten, en houdt het mask waar SAM de hoogste score aan geeft. Op hoeveel van 30 foto's van zonnebloemen is dat de hele bloem, met een IoU van minstens 0,8 met zijn omtrek?",
        options={
            "none": "op geen of bijna geen",
            "some": "op ongeveer een derde",
            "most": "op ongeveer twee derde",
            "all": "op alle of bijna alle",
        },
        reveal="**Op 1 van de 30.** De diepste pixel van een zonnebloem ligt tussen zijn pitten, en een klik daar kan de pitten of de hele bloem betekenen. Op 25 van de 30 foto's is een van de drie masks van SAM de hele bloem, met een IoU van minstens 0,8, maar SAM geeft de pitten een hogere score. Eén klik kan SAM niet vertellen welke je bedoelde. Een tweede klik, waar het mask de bloem mist, kan dat wel: met de tweede klik van de course gaat de mediane IoU van de 30 foto's van 0,18 naar 0,91.",
    )
    guess_sunflower
    return (guess_sunflower,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Deze les gebruikt **SAM 2.1**, het Segment Anything Model 2.1, dat Meta in 2024 publiceerde met de paper van Ravi en anderen; de eerste SAM, van Kirillov en anderen, kwam uit in 2023. SAM doet **promptable segmentation**: je zegt met een **prompt** wat je bedoelt, en hij geeft daar het mask van. Een prompt is een of meer klikken, elk met een label: 1 voor een **positieve klik**, op het ding, en 0 voor een **negatieve klik**, op iets wat er niet bij hoort. Een box werkt ook, gegeven met zijn hoek linksboven en zijn hoek rechtsonder. SAM heeft geen soorten. Het FCN van [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) geeft elke pixel een van de soorten die het leerde; SAM tekent het mask van wat je ook aanwijst, en zegt niet wat het is. Dat leerde hij van SA-1B, 11 miljoen foto's met 1,1 miljard masks, en van video's. De course gebruikt zijn kleinste versie, SAM 2.1 tiny, met ongeveer 39 miljoen parameters. De eigen code van de course, `everything_ai/sam2.py`, schrijft de 31,3 miljoen daarvan uit die met foto's werken, en laat het geheugen weg waarmee SAM 2.1 een ding door een video volgt.

        SAM werkt in drie delen. De image encoder, het grootste deel, maakt de foto 1024 bij 1024 pixels en maakt er features van: 256 getallen voor elk vakje van 16 bij 16 pixels, en twee fijnere maps voor de randen. Hij draait één keer per foto, in ongeveer een seconde. De prompt encoder maakt van elke klik 256 getallen, uit waar de klik ligt en zijn label. De mask decoder, een klein netwerk, combineert die met de features, en geeft vier masks van 256 bij 256 logits, en voor elk mask zijn **predicted IoU**: de IoU die hij voor dat mask verwacht met wat je bedoelde. Dat is de score van het mask. De decoder doet er een paar honderdsten van een seconde over, dus je kunt op één foto steeds opnieuw klikken, zonder de image encoder opnieuw te draaien.

        Waarom vier masks? Eén klik kan meer dan één ding betekenen: een klik op een bloemblad kan het bloemblad of de hele bloem betekenen. De paper van de eerste SAM vond dat drie masks de meeste gevallen dekken: het geheel, een deel en een deel daarvan. Dus masks 1 tot en met 3 zijn drie van zulke gokken, in geen vaste volgorde: bij de klik op de lepel hieronder is mask 1 het grootst en mask 2 het kleinst. Mask 0 is voor een prompt die precies zegt wat hij bedoelt, zoals meerdere klikken of een box. Met één klik neemt de course het mask met de hoogste score van masks 1 tot en met 3, zoals de makers van SAM aanraden; met meer klikken mask 0. Bij meer klikken neemt de eigen code van SAM toch de beste van masks 1 tot en met 3 als de rand van mask 0 onzeker is; de course laat die stap weg. Een pixel hoort bij het mask waar zijn logit boven 0 ligt: waar de sigmoid er een kans boven 0,5 van maakt.

        En als het mask fout is? Dan klik je opnieuw, waar het het meest fout is: met label 1 op een deel dat het mist, of met label 0 op een deel dat het te veel heeft. SAM krijgt alle klikken, ook de eerdere, en geeft een nieuw mask. Een goede plek voor een klik ligt ver van de randen van wat fout is: de **diepste pixel**, de pixel die als laatste overblijft als je het steeds opnieuw erodeert, zoals in [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl). Met de omtrekken die bij Caltech-101 horen, speelt de course de persoon die klikt: de eerste klik in de diepste pixel van het voorwerp, en elke volgende klik in de diepste pixel van wat het mask het meest fout heeft.
        """),
        L.watch_out("Geef SAM elke keer alle klikken, niet alleen de nieuwste. Hij onthoudt geen eerdere klikken: één nieuwe klik op zichzelf is een nieuwe prompt, en geeft het mask van alleen die klik."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **promptable segmentation** | segmentatie waarbij je met een prompt zegt wat er gesegmenteerd moet worden, en het model er het mask van geeft | `sam.decode(features, points, labels)` |
        | **prompt** | wat SAM vertelt wat je bedoelt: klikken, elk met zijn label, of een box | `points`, `labels` |
        | **positieve klik**, **negatieve klik** | een klik op het ding, met label 1; een klik op wat er niet bij hoort, met label 0 | `labels = torch.tensor([[1, 0]])` |
        | **predicted IoU** | de score die SAM een mask geeft: de IoU die hij voor dat mask verwacht met wat je bedoelde | `scores[0, k]` |
        | **diepste pixel** | de pixel van een gebied die als laatste overblijft als je het erodeert, ver van zijn randen | `deepest(region)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_spoon():
    mo.vstack([
        mo.md("**Eén klik op het kommetje van de lepel: de drie masks van SAM**"),
        mo.Html(C.spoon_picture({
            "title": "mask {k}: {score}",
            "alt": "Drie keer het deel van de koffiefoto met de lepel, in grijs, met een stip waar de klik is, op het kommetje van de lepel, en een van de masks van SAM eroverheen. Mask 1, met een score van 0,645, is het kommetje met het eind van de steel, bovenaan. Mask 2, met 0,059, is een klein glimplekje op het kommetje. Mask 3, met 0,941, is het kommetje.",
        }, "nl")),
        mo.md(r"""
        Eén klik op het kommetje van de lepel van de koffiefoto, op (360, 277), en de drie masks die SAM ervoor geeft, met hun scores. De foto is hier grijs, zodat de masks opvallen. Mask 1, met 0,645, is de lepel met het eind van zijn steel, bovenaan, dat boven het kopje uitsteekt; het dunne midden van de steel, langs het kopje, laat het weg. Mask 2, met 0,059, is een glimplekje op het kommetje, 228 pixels. Mask 3, met 0,941, is het kommetje. SAM verwacht dat het kommetje het best past bij wat je bedoelde, en zet het bovenaan. Mask 0, hier niet getekend, is ook het kommetje, met 0,901. Op het oor van het kopje, op (217, 262), is mask 3 dat oor, met 0,894, en mask 1 het hele kopje, met maar 0,122.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoeveel klikken heeft SAM nodig? De course gaf SAM 30 foto's van elk van 10 soorten dingen uit Caltech-101, en nam de omtrek die bij elke foto hoort als wat je bedoelde. Hij klikte in de diepste pixel van het voorwerp, koos het mask zoals hierboven, en klikte daarna nog vier keer, elke keer waar het mask het meest fout was. Met de slider stel je het aantal klikken in. Voor elke soort is de open stip de mediane IoU na één klik, de dichte stip na het aantal klikken dat je instelt; het korte rechtopstaande streepje is de mediane IoU als SAM een box om de omtrek krijgt in plaats van klikken.
    """)
    return


@app.cell(hide_code=True)
def explore_clicks_cell():
    def _frame(k):
        _svg = C.clicks_frame(int(k), {"x": "mediane IoU met de omtrek, over 30 foto's",
                                       "alt": "Voor elk van 10 soorten dingen uit Caltech-101 een rij met de mediane IoU van het mask van SAM met de omtrek na één klik, een open stip, en na het aantal klikken dat de slider aangeeft, {k}, een dichte stip, met een lijn ertussen, en een kort rechtopstaand streepje voor een box. Met één klik halen alleen de zeesterren een mediane IoU van 0,85; de zonnebloemen beginnen bij 0,18 en de kroonluchters bij 0,08. Met twee klikken halen 8 van de 10 soorten 0,85. De kroonluchters en de windsorstoelen halen het nooit."}, "nl")
        return _svg, C.clicks_line(int(k), "nl")

    explore_clicks = L.explore("clicks", "klikken", _frame, start=5, low=1, high=5, step=1, role="hyper",
                               task="Zoek het kleinste aantal klikken waarna 8 van de 10 soorten een mediane IoU van 0,85 of meer hebben.")
    explore_clicks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Met één klik halen alleen de zeesterren een mediane IoU van 0,85: 0,919. De zonnebloemen beginnen bij 0,18, en de nautilussen bij 0,535: één klik in hun midden, op de pitten of op de binnenste windingen van de schelp, kan een deel betekenen, en SAM zet het deel bovenaan. Na 2 klikken halen 8 van de 10 soorten 0,85, de zonnebloemen 0,913. Meer klikken helpen steeds minder, en niet altijd: de mediaan van de schenkkannen zakt van 0,914 na 3 klikken naar 0,854 na 4, en die van de windsorstoelen van 0,832 na 4 naar 0,754 na 5. Kroonluchters en windsorstoelen halen 0,85 nooit, met op zijn best 0,678 en 0,832. Hun omtrekken lopen om de armen en de spijlen heen, gaten en al, dus een volgende klik kan in een gat terechtkomen, waar de foto laat zien wat erachter is. Een box helpt daar ook niet tegen: voor de windsorstoelen is zijn mediane IoU 0,678, lager dan wat twee klikken geven. Voor de kroonluchters doet een box het beter dan vijf klikken, 0,761, en voor de andere acht soorten ongeveer even goed als twee tot vijf klikken, van 0,856 tot 0,947.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met SAM 2.1, `sam`, te segmenteren wat je op een foto aanwijst:

    1. Maak de foto klaar, `image = prepare(photo)`, en bereken één keer zijn features: `features = sam.features(image)`, binnen `with torch.no_grad():`.
    2. Maak van elke klik `(x, y)` in pixels van de foto een punt op de foto van 1024 bij 1024: `x * 1024 / width` en `y * 1024 / height`. Geef elke klik zijn label: 1 op het ding, 0 op wat er niet bij hoort.
    3. De masks: `logits, scores = sam.decode(features, points, labels)`, vier masks van 256 bij 256 logits, en hun scores.
    4. Maak de masks zo groot als de foto, `to_photo(logits[0], width, height)`, en kies er een: met één klik de hoogste score van masks 1 tot en met 3; met meer mask 0. Een pixel zit erin waar zijn logit boven 0 ligt. Oefening 1 doet dit.
    5. Is het mask fout? Voeg een klik toe waar het het meest fout is, zoals oefening 3 doet, en ga terug naar stap 2 met alle klikken. De features blijven zoals ze zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de koffiefoto klaar, stap 1, en print zijn shape en die van de grofste map van zijn features. `prepare` uit `everything_ai.sam2` van de course maakt de foto 1024 bij 1024 pixels, wat zijn shape ook is, en normaliseert hem met het gemiddelde en de standaardafwijking van ImageNet, zoals in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl). Dan klikt de cel op het kommetje van de lepel, stap 2, en print het punt op de foto van 1024 bij 1024. Dat geeft hij aan SAM, stap 3, en hij print de shape van de logits en de vier scores. Als laatste de eerste helft van stap 4: hij maakt de vier masks zo groot als de foto met `to_photo`, en print hoeveel pixels elk heeft. Het kiezen is oefening 1. `sam` is SAM 2.1, die de cel hieronder laadt zodra hij gedownload is, samen met de foto's van Caltech-101 en hun omtrekken, voor de demo.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="SAM 2.1 en Caltech-101 downloaden, 293 MB")
    return (get_button,)


@app.cell(hide_code=True)
def sam_cell(get_button):
    sam = C.sam_or_stop(get_button, {
        "ask": "*SAM 2.1 en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: SAM 2.1 van Hugging Face, 156 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's en de omtrekken uitpakt. Het uitgewerkte voorbeeld en de demo hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 293 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*SAM 2.1 en de foto's staan op je computer: de 31,3 miljoen parameters van SAM 2.1 die met foto's werken, getraind op de 1,1 miljard masks van SA-1B en op video's.*",
    })
    return (sam,)


@app.cell
def example_spoon(sam):
    import torch
    from PIL import Image
    from everything_ai.sam2 import prepare, to_photo

    photo = Image.open("data/photos/coffee.png")
    width, height = photo.size
    # 1. de features van de foto, één keer: SAM maakt hem 1024 bij 1024 pixels
    image = prepare(photo)
    with torch.no_grad():
        features = sam.features(image)
    print(image.shape, features[0].shape)
    # 2. één klik op het kommetje van de lepel, (360, 277), als punt op de foto van 1024 bij 1024, met label 1
    points = torch.tensor([[[360 * 1024 / width, 277 * 1024 / height]]])
    labels = torch.tensor([[1]])
    print(points)
    # 3. vier masks van 256 bij 256 logits, en de IoU die SAM van elk verwacht
    with torch.no_grad():
        logits, scores = sam.decode(features, points, labels)
    print(logits.shape, scores[0].numpy().round(3))
    # 4. de masks zo groot als de foto: een pixel zit in een mask waar zijn logit boven 0 ligt
    logits = to_photo(logits[0], width, height)
    print(logits.shape, (logits > 0).sum(axis=(1, 2)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De klaargemaakte foto heeft shape `(1, 3, 1024, 1024)`: één foto, drie kleuren, 1024 bij 1024 pixels, hoewel de koffiefoto 600 bij 400 is. SAM rekt hem uit, en `to_photo` rekt zijn masks aan het eind terug. De grofste map van de features heeft shape `(1, 256, 64, 64)`: 256 getallen voor elk van 64 bij 64 vakjes van 16 bij 16 pixels. De klik (360, 277) wordt het punt (614,4; 709,1) op de foto van 1024 bij 1024. De logits hebben shape `(1, 4, 256, 256)`: vier masks van 256 bij 256, een kwart van 1024 in elke richting, en de scores zijn 0,901, 0,645, 0,059 en 0,941. Zo groot als de foto hebben de masks shape `(4, 400, 600)`, en hun pixels zijn `[5516 6762  228 5575]`: mask 1, de lepel met het eind van zijn steel, is het grootst, en mask 2, het glimplekje, het kleinst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het mask kiezen

        Maak `choose(logits, scores, clicks)` af, stap 4 van het recept: `logits` bevat de vier masks van logits van SAM, zo groot als de foto, met shape `(4, height, width)`; `scores` hun vier scores; en `clicks` het aantal klikken in de prompt. Met één klik neemt de functie het mask met de hoogste score van masks 1 tot en met 3; met meer klikken mask 0. Ze geeft dat mask terug als array van True en False: True waar de logit boven 0 ligt. Vervang de drie `...`. De check roept haar aan met de vier masks van het uitgewerkte voorbeeld:

        ```python
        choose(logits, np.array([0.901, 0.645, 0.059, 0.941]), 1)
        ```

        Dat moet mask 3 geven, het kommetje: een mask van 5.575 pixels. De check probeert ook de klik op het oor van het kopje, de lepel met een tweede klik, en masks van logits die hij zelf verzint.
        """),
        L.task("Laat `choose(logits, scores, clicks)` het mask geven dat SAM bij de prompt bedoelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    choose_code = L.code_box("choose", lines=10)
    choose_code
    return (choose_code,)


@app.cell(hide_code=True)
def check_choose(choose_code):
    result_choose = L.check("choose", choose_code.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "Met één klik doet mask 0 niet mee. `scores[1:]` bevat de scores van masks 1 tot en met 3, en `np.argmax` geeft de plek van de hoogste daarvan.",
            "`np.argmax(scores[1:])` telt vanaf het begin van `scores[1:]`: zijn 0 is mask 1, dus het nummer van het mask is één meer. Het mask zelf is True waar zijn logit boven 0 ligt: vergelijk de hele array `logits[k]` in één keer met 0.",
            "`k = ___ + np.argmax(scores[1:])` met één klik, `k = ___` met meer, en `return logits[k] > ___`.",
        ],
        why="Bij de lepel zijn de scores van masks 1 tot en met 3 0,645, 0,059 en 0,941. De hoogste, 0,941, staat op plek 2 van `scores[1:]`, dus het is mask 3, het kommetje, en `logits[3] > 0` is True in 5.575 pixels.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_seeds():
    mo.md(r"""
    ### Oefening 2. Een tweede klik
    """)
    return


@app.cell(hide_code=True)
def choice_seeds_cell():
    choice_seeds = L.choice(
        "seeds",
        question="Je klikt één keer op de pitten midden in een zonnebloem, en het beste mask van SAM is alleen de pitten. Je wilt de hele bloem. Welke tweede klik helpt?",
        options={
            "right": "een klik op een bloemblad, met label 1",
            "negative": "een klik op de pitten, met label 0",
            "again": "een tweede klik op de pitten, met label 1",
            "outside": "een klik op de bladeren rond de bloem, met label 1",
        },
        correct="right",
        why={
            "right": "Label 1 op een bloemblad zegt: dit hoort ook bij wat ik bedoel. De pitten alleen bevatten het bloemblad niet, dus het mask van SAM groeit tot de hele bloem. Op de 30 foto's van zonnebloemen van de slider komt de tweede klik van de course waar het mask het meest miste, en gaat de mediane IoU van 0,18 naar 0,91.",
            "negative": "Label 0 zegt: dit hoort niet bij wat ik bedoel. Op de pitten zou dat het midden van de bloem uit het mask halen. Waar mist het mask iets?",
            "again": "Dat zegt nog eens wat de eerste klik zei: de pitten. Niets daarin vertelt SAM dat de bloembladen er ook bij horen.",
            "outside": "Label 1 op de bladeren zegt dat de bladeren ook horen bij wat je bedoelt: het mask zou over de bladeren groeien, niet alleen over de bloembladen. Welk deel van de bloem mist het mask?",
        },
    )
    choice_seeds
    return (choice_seeds,)


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Waar je daarna klikt

        Maak `next_click(mask, truth)` af, het begin van stap 5: `mask` is het mask dat SAM gaf, en `truth` het mask van wat je bedoelde, allebei arrays van True en False met dezelfde shape. De functie neemt de pixels van `truth` die het mask miste, en de pixels van het mask die niet in `truth` zitten, en klikt waar er meer zijn: in de diepste pixel van de gemiste pixels, met label 1, of van de pixels te veel, met label 0. Ze geeft de klik terug als `(x, y, label)`, of None als het mask goed is. `deepest(region)`, uit `everything_ai.sam2` van de course, geeft de diepste pixel van een gebied als `(x, y)`. `~mask` maakt van elke True van het mask False en van elke False True, en `&` geeft True waar beide kanten True zijn. Vervang de vijf `...`. De check roept haar aan met het kommetje, mask 3 van het uitgewerkte voorbeeld, als `mask`, en als `truth` de lepel zoals mask 1 hem heeft, met het eind van zijn steel:

        ```python
        next_click(bowl, spoon)
        ```

        Dat moet `(411, 83, 1)` geven: een klik op het eind van de steel. De check probeert ook masks die hij zelf verzint: te klein, te groot, naast het ding, en goed.
        """),
        L.watch_out("Een klik is `(x, y)`: eerst x naar rechts, dan y naar beneden, zoals `deepest` hem geeft. Een array van pixels indexeer je andersom, `mask[y, x]`: eerst de rij."),
        L.task("Laat `next_click(mask, truth)` de volgende klik geven, waar het mask het meest fout is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    next_code = L.code_box("next", lines=14)
    next_code
    return (next_code,)


@app.cell(hide_code=True)
def check_next(next_code):
    result_next = L.check("next", next_code.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next",
        [
            "Twee groepen pixels zijn fout: die van `truth` die het mask miste, en die van het mask die niet in `truth` zitten. De klik komt in de grootste groep, in zijn diepste pixel.",
            "De gemiste pixels zijn True in `truth` en False in `mask`: `truth & ~mask`; de pixels te veel andersom. `.sum()` telt de True van een array, dus vergelijk `missed.sum()` met `extra.sum()`. `deepest` geeft `x, y`; de klik krijgt er zijn label bij: 1 voor gemiste pixels, 0 voor pixels te veel.",
            "`missed = truth & ~___`, `extra = mask & ~___`, `if missed.sum() >= ___.sum():`, en dan twee keer `return (x, y, ___)`, met het label van elk.",
        ],
        why="Bij de lepel mist het mask van het kommetje 1.257 pixels van de lepel, het eind van zijn steel, en heeft het er 70 te veel, langs zijn rand. De gemiste pixels zijn er meer, dus de klik komt in hun diepste pixel, (411, 83), met label 1. Met die tweede klik is mask 0 van SAM het kommetje met het eind van de steel: zijn IoU met de lepel van mask 1 gaat van 0,806 naar 0,941.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je functies op nieuwe foto's

    Zodra SAM en de foto's gedownload zijn, bij het uitgewerkte voorbeeld, geeft de cel hieronder SAM 6 foto's uit Caltech-101 en speelt hij de persoon die klikt: met je `choose` en `next_click` zodra ze goed zijn, anders met die van de course. Met de slider stel je het aantal klikken in. Elke foto laat de omtrek zien die erbij hoort, het mask van SAM en de klikken, met de IoU erboven. De foto's kosten één keer een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_slider_cell():
    demo_slider = L.slider("demo", "klikken", start=1, low=1, high=5, step=1, role="hyper")
    demo_slider
    return (demo_slider,)


@app.cell(hide_code=True)
def demo_clicks(choose_code, demo_slider, next_code, result_choose, result_next, sam):
    _k = int(demo_slider.widget.value)
    _d = C.demo(sam, choose_code.value if result_choose.state == "correct" else None,
                next_code.value if result_next.state == "correct" else None, _k, {
        "titles": ["zonnebloem 0: IoU {iou}", "nautilus 0: IoU {iou}", "zeester 4: IoU {iou}", "libel 5: IoU {iou}", "kroonluchter 1: IoU {iou}", "windsorstoel 2: IoU {iou}"],
        "alt": f"Zes foto's uit Caltech-101 in twee rijen, een zonnebloem, een nautilus, een zeester, een libel, een kroonluchter en een windsorstoel, elk met de omtrek van het voorwerp, het mask van SAM en de klikken, {_k} in totaal.",
    }, "nl")
    _who = {(True, True): "je `choose` en je `next_click`", (True, False): "je `choose` en de `next_click` van de course",
            (False, True): "de `choose` van de course en je `next_click`", (False, False): "de `choose` en `next_click` van de course"}
    mo.vstack([
        mo.Html(_d["picture"]),
        mo.md(f"Geklikt met {_who[(_d['choose'], _d['next'])]}. Na {_d['clicks']} is de IoU van de 6 foto's {_d['listed']}."),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met één klik krijgen vijf van de zes een deel: de pitten van de zonnebloem, de binnenste windingen van de nautilus, één vleugel van de libel, het midden van de kroonluchter en de zitting van de stoel. Alleen de zeester is meteen heel, 0,94. De tweede klik komt waar het mask het meest miste, en de zonnebloem en de nautilus worden heel, 0,96 en 0,97, en de libel 0,90. De windsorstoel is het best na 3 klikken, 0,92, en wordt daarna slechter, 0,73 na 5: meer klikken helpen niet altijd. De kroonluchter komt na 5 klikken op 0,79. Zodra je functies goed zijn, is het plaatje hetzelfde als met die van de course: ze kiezen dezelfde masks en dezelfde klikken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_name():
    mo.md(r"""
    ### Oefening 4. Zeggen wat het is
    """)
    return


@app.cell(hide_code=True)
def explain_name_cell():
    explain_name = L.explain(
        "name",
        question="Iemand bouwt een foto-app: je tikt op iets in een foto, en de app tekent de omtrek ervan en zegt wat het is, zoals \"kopje\" of \"lepel\". Die persoon gebruikt alleen SAM. Waarom kan de app de omtrek wel tekenen, maar niet zeggen wat het is, en wat heeft de app naast SAM nodig?",
        model_answer="**SAM heeft geen soorten.** Hij tekent het mask van waar je op tikt, maar niets in hem zegt kopje of lepel: hij leerde nooit namen, alleen waar het ene ding ophoudt en het andere begint. Dus de app krijgt de omtrek, en geen woord ervoor. **Wat wel werkt**, is SAM samen met een model dat soorten leerde: het mask van SAM zegt waar het ding is, en een classifier die getraind is op foto's met de namen van hun soorten, zegt wat er binnen dat mask zit. Een FCN dat getraind is op masks van die soorten, zoals in de les over FCN's, kan de pixels ook een soort geven, maar alleen van de soorten die het leerde.",
        points=[
            "je zegt dat SAM geen soorten heeft: hij geeft het mask van waar je op tikt, maar geen naam ervoor",
            "je zegt dat een naam een model nodig heeft dat soorten leerde, zoals een classifier of een FCN, getraind op foto's met hun soorten",
            "je zegt hoe de twee samenwerken: het mask van SAM zegt waar het ding is, het andere model wat het is",
        ],
    )
    explain_name
    return (explain_name,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) leerde een netwerk elke pixel een van zijn soorten te geven. SAM 2.1 geeft helemaal geen soorten: hij tekent het mask van wat je aanwijst, promptable segmentation. Zijn image encoder maakt één keer features van de foto, en zijn kleine decoder maakt van elke prompt vier masks met hun predicted IoU: masks 1 tot en met 3 voor een klik die een deel of het geheel kan betekenen, mask 0 voor een prompt die zegt wat hij bedoelt. Met één klik kies je op score; waar het mask fout is, klik je opnieuw, in de diepste pixel van wat het mist of te veel heeft, zoals erosie uit [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl) die vindt. Op 30 foto's van elk van 10 soorten uit Caltech-101 geeft één klik voor 1 soort een mediane IoU van 0,85 of meer, en geven twee klikken dat voor 8; kroonluchters en windsorstoelen, met gaten binnen hun omtrek, komen er nooit. En SAM kan niet zeggen wat hij tekende: voor een naam is een model nodig dat soorten leerde. Hoe zijn image encoder, een transformer, werkt, leggen latere lessen uit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_seeds, explain_name, result_choose, result_next):
    L.wrap_up(result_choose, choice_seeds, result_next, explain_name)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
