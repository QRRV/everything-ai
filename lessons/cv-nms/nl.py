# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-nms. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-nms", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_iou():
    quick_iou = L.choice(
        "quick-iou",
        question="Wat is de IoU van de boxes `(10, 10, 19, 19)` en `(10, 15, 19, 24)`, geschreven als `(boven, links, onder, rechts)` met beide uiteinden meegeteld?",
        options={
            "third": "0,33",
            "half": "0,5",
            "quarter": "0,25",
            "one": "1",
        },
        correct="third",
        why={
            "third": "Elke box bevat 10 bij 10 = 100 pixels. Ze delen rij 10 tot en met 19 en kolom 15 tot en met 19: 10 bij 5 = 50 pixels. Samen bedekken ze 100 + 100 - 50 = 150, en 50 / 150 = 0,33. In deze les vertelt de IoU welke boxes over hetzelfde voorwerp gaan.",
            "half": "50 / 100 deelt de overlap door één box. De IoU deelt door wat de twee boxes samen bedekken: beide oppervlaktes, min de overlap.",
            "quarter": "50 / 200 telt de twee oppervlaktes op, dus de 50 pixels die ze delen, tel je dubbel. Trek de overlap er één keer af.",
            "one": "De twee boxes zijn even groot, maar ze liggen niet op dezelfde plek: de tweede begint 5 kolommen verder naar rechts. Wat delen ze?",
        },
    )
    quick_iou
    return (quick_iou,)


@app.cell(hide_code=True)
def quick_check_found():
    quick_found = L.choice(
        "quick-found",
        question="PASCAL VOC telt een voorspelde box als gevonden als zijn IoU met de echte box boven 0,5 ligt. Welke van deze boxes telt als gevonden?",
        options={
            "above": "alleen de box met een IoU van 0,53",
            "both": "de box met 0,53 en de box met precies 0,5",
            "half": "alleen de box met precies 0,5",
            "neither": "geen van de twee",
        },
        correct="above",
        why={
            "above": "0,53 ligt boven 0,5, en 0,5 zelf niet. Ook in deze les zegt een regel \"boven\": een box wordt weggegooid als zijn IoU met een gehouden box boven een vast getal ligt, dus een box die precies op dat getal zit, blijft staan.",
            "both": "\"Boven 0,5\" laat 0,5 zelf buiten. Welke van de twee ligt erboven?",
            "half": "0,5 ligt niet boven 0,5. Welke box wel?",
            "neither": "0,53 ligt boven 0,5, dus die box telt als gevonden. Welke niet?",
        },
    )
    quick_found
    return (quick_found,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een detector vindt een voorwerp niet één keer, maar vaak. Hij bekijkt een beeld op allerlei plekken en op allerlei schalen, en overal waar iets op een voorwerp lijkt, geeft hij een box. Om één voorwerp heen lijken veel van die boxes bijna op elkaar, dus de detector geeft voor elk voorwerp een hele groep boxes. Om de voorwerpen te tellen, of om er telkens één box omheen te tekenen, moet je één box per voorwerp houden en de rest weggooien.

    **Een voorbeeld uit het dagelijks leven.** Vijf vrienden prikken op een gedeelde kaart een speld bij elk café dat ze in de stad kennen, dus een café kan een paar spelden dicht bij elkaar krijgen. Je wilt één speld per café. Je houdt per café de speld van de vriend die het zekerst is, en haalt de spelden weg die bijna op dezelfde plek staan: die bedoelen hetzelfde café. Een speld ver van elke speld die je hield, blijft staan: die hoort bij een ander café.
    """)
    return


@app.cell(hide_code=True)
def predict_top():
    guess_top = L.guess(
        "top",
        question="Een detector geeft elke box een score van 0 tot 1: hoe zeker hij is dat er een voorwerp in zit. Voor de 4 vormen in de scène van de vorige twee lessen geeft hij 28 boxes. Je houdt de 4 boxes met de hoogste scores. Hoeveel van de 4 vormen krijgen een box?",
        options={
            "four": "4",
            "three": "3",
            "two": "2",
            "one": "1",
        },
        reveal="**1.** De 6 hoogste scores horen allemaal bij boxes om de cirkel, de vorm waar de detector het zekerst van is: boxes die bijna gelijk zijn, krijgen bijna dezelfde score. De hoogste score van het vierkant komt pas op de negende plek: 0,70. De hoogste scores geven je kopieën van één voorwerp; wat je nodig hebt, is de hoogste score van elk voorwerp.",
    )
    guess_top
    return (guess_top,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bij elke box die een detector geeft, hoort een **confidence score**: een getal van 0 tot 1 dat zegt hoe zeker de detector is dat er een voorwerp in die box zit. Boxes die bijna gelijk zijn, krijgen bijna dezelfde score, dus de hoogste scores zitten bij elkaar, op het voorwerp waar de detector het zekerst van is.

        **Non-maximum suppression**, of NMS, houdt één box per voorwerp over. Dat gaat in rondes, met een lijst van alle boxes. Neem de box met de hoogste score in de lijst en houd hem. Gooi dan elke box uit de lijst waarvan de IoU ermee boven een **drempel** ligt, zoals 0,5: die overlappen hem zo veel dat ze over hetzelfde voorwerp moeten gaan. De gehouden box gaat ook uit de lijst, want zijn IoU met zichzelf is 1. Doe daarna hetzelfde met de boxes die nog in de lijst staan: houd die met de hoogste score, gooi elke box weg waarvan de IoU ermee boven de drempel ligt, enzovoort, tot de lijst leeg is. De naam zegt wat het doet: elke box die niet het maximum van zijn groep is, wordt onderdrukt.

        De drempel bepaalt wat als hetzelfde voorwerp telt. Zet je hem te laag, dan gaat de box van een voorwerp vlak bij een ander ook weg, alsof het een kopie is. Zet je hem te hoog, dan blijven twee boxes om hetzelfde voorwerp allebei staan. Een gangbare keuze is 0,5; dat gebruikt torchvision, de bibliotheek van PyTorch voor beelden, ook standaard voor de laatste boxes van zijn detector Faster R-CNN.

        De boxes in deze les komen van de course: om elke vorm in de scène 7 boxes, willekeurig een beetje verschoven en groter of kleiner gemaakt, elk met een score die meestal hoger is naarmate de box beter past, met wat ruis, net als bij een echte detector. Dat is wat een detector vóór NMS geeft; een latere les gebruikt een echte detector. De beste box van een vorm is in deze les zijn box met de hoogste score.
        """),
        L.watch_out("Een box die weggegooid is, gooit niets weg. Vergelijk elke box alleen met de boxes die je gehouden hebt, niet met elke box met een hogere score: een box waarvan de IoU alleen met een weggegooide box hoog is, blijft staan. Het kan een ander voorwerp ernaast zijn."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **confidence score** | hoe zeker een detector is dat er een voorwerp in een box zit, van 0 tot 1 | `scores[i]` |
        | **non-maximum suppression** (NMS) | houd de box met de hoogste score, gooi elke box weg waarvan de IoU ermee boven de drempel ligt, en herhaal dat met de boxes die over zijn | `nms(boxes, scores, 0.5)` |
        | **drempel** | de IoU waarboven twee boxes als hetzelfde voorwerp tellen | `iou(a, b) > threshold` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_proposals():
    mo.vstack([
        mo.md("**28 boxes, en de 4 die NMS houdt**"),
        mo.Html(C.proposals_picture({
            "all": "alle {n} boxes",
            "kept": "de {n} boxes die NMS houdt",
            "point": ",",
            "alt": "Twee plaatjes van dezelfde scène: een grote blauwe driehoek, een blauw vierkant dat deels onder een rode cirkel ligt, en een kleine gele driehoek. Links 28 gestreepte boxes, 7 dicht bij elkaar om elke vorm. Rechts zijn er nog maar 4 gestreepte boxes over, één om elke vorm, elk met zijn score: 0,84 voor de cirkel, 0,72 voor de kleine driehoek, 0,70 voor het vierkant en 0,59 voor de grote driehoek.",
        })),
        mo.md(r"""
        Links de 28 boxes voor de 4 vormen, gestreept: om elke vorm een groep van 7. Rechts de 4 boxes die NMS houdt met een drempel van 0,5, één per vorm, met hun scores: 0,84 voor de cirkel, 0,72 voor de kleine driehoek, 0,70 voor het vierkant en 0,59 voor de grote driehoek.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Wat doet de drempel? Met de slider zet je hem op 0,05 tot 0,95. Links de boxes die NMS houdt; rechts hoeveel boxes NMS houdt bij elke drempel, met een stippellijn bij 4, één box per vorm.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _n = C.threshold_frame(t, {"title": "drempel {t}: {n} boxes", "count": "gehouden boxes", "threshold": "drempel", "point": ",",
                                          "alt": "De scène met de {n} boxes die NMS houdt bij een drempel van {t}, gestreept. Ernaast hoeveel boxes NMS houdt bij elke drempel van 0 tot 1: 3 bij de laagste drempels, 4 van 0,14 tot 0,57, daarna steeds meer, tot alle 28 vanaf 0,94. Een stippellijn staat bij 4, één box per vorm, en een grotere punt bij de drempel van {t}."})
        return _html, f"Drempel {L.number(t, 2)}: NMS houdt {_n} boxes."

    explore_threshold = L.explore("threshold", "drempel", _frame, start=0.5, values=C.THRESHOLDS, role="hyper",
                                  task="Zoek de grootste drempel waarbij elke vorm nog precies één box houdt.")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Van 0,15 tot 0,55 houdt NMS 4 boxes, één per vorm. Bij 0,6 houden de kleine driehoek en het vierkant elk een tweede box, waarvan de IoU met hun eerste box niet meer boven de drempel ligt. Vanaf daar loopt het aantal op, tot bij 0,95 alle 28 boxes blijven staan.

    Aan de lage kant gaat er iets anders mis. Het vierkant ligt deels onder de cirkel, dus zijn boxes overlappen de gehouden box van de cirkel: de IoU is 0,26 voor zijn beste box, en minstens 0,13 voor elk van zijn 7 boxes. Bij 0,1 en 0,05 gooit de box van de cirkel alle 7 weg, en heeft het vierkant helemaal geen box. Bij 0,15 en 0,2 houdt het alleen zijn slechtste box, met een score van 0,45, de box met de laagste IoU met de box van de cirkel. Bij 0,25 houdt het een box met 0,65, en pas vanaf 0,3 zijn beste box, met 0,70. In deze scène houdt elke drempel van 0,3 tot 0,55 de beste box van elke vorm: laag genoeg om de kopieën weg te gooien, hoog genoeg om het vierkant naast de cirkel te houden.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo houd je één box per voorwerp:

    1. Zet de nummers van alle boxes in een lijst.
    2. Zoek de box met de hoogste score in de lijst en houd hem.
    3. Haal elke box uit de lijst waarvan de IoU met de gehouden box boven de drempel ligt; de gehouden box gaat ook.
    4. Herhaal stap 2 en 3 tot de lijst leeg is. De boxes die je gehouden hebt, zijn de uitkomst, één per voorwerp.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op de scène van de vorige twee lessen, `scene(18)`. `proposals(s, seed=25)` geeft de 28 boxes die de course voor die scène maakt, en voor elke box een score, in dezelfde volgorde: `scores[k]` is de score van `boxes[k]`. De eerste cel print de boxes met een score boven 0,7.

    De tweede cel doet NMS met een drempel van 0,5. `iou` is de functie van oefening 2 van [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl). `rest` bevat de nummers van de boxes die nog in de lijst staan, zodat `boxes[i]` en `scores[i]` bij elkaar blijven. De `while`-loop draait zolang de lijst niet leeg is. Elke ronde zoekt de `for`-loop de box met de hoogste score: `best` begint als het eerste nummer in `rest`, en telkens als de loop een box met een hogere score tegenkomt, wordt die box `best`. Daarna houdt de list comprehension alleen de boxes waarvan de IoU met `best` 0,5 of minder is. De cel print elke gehouden box, zijn score, en hoeveel boxes er na de ronde over zijn.
    """)
    return


@app.cell
def example_boxes():
    from everything_ai.scenes import scene, proposals

    s = scene(18)
    # 28 boxes, en voor elke box een score: scores[k] hoort bij boxes[k]
    boxes, scores = proposals(s, seed=25)
    print(len(boxes), "boxes")
    for k in range(len(boxes)):
        if scores[k] > 0.7:
            print(k, boxes[k], scores[k])
    return boxes, scores


@app.cell
def example_nms(boxes, scores):
    def iou(a, b):
        # de intersection over union van twee boxes, zoals in de vorige les
        top, left = max(a[0], b[0]), max(a[1], b[1])
        bottom, right = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0, bottom - top + 1) * max(0, right - left + 1)
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return overlap / (area_a + area_b - overlap)


    rest = list(range(len(boxes)))   # 1. de nummers van alle boxes
    kept = []
    while len(rest) > 0:             # 4. tot de lijst leeg is
        # 2. de box met de hoogste score in de lijst: houd hem
        best = rest[0]
        for i in rest:
            if scores[i] > scores[best]:
                best = i
        kept.append(best)
        # 3. houd in de lijst alleen de boxes met een IoU van 0,5 of minder
        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= 0.5]
        print(best, boxes[best], scores[best], len(rest))
    print(kept)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste cel print de 8 boxes met een score boven 0,7. Zeven ervan liggen om de cirkel: hun bovenkanten liggen tussen rij 56 en 60, en ze houden op bij kolom 78 of 79, zeven keer bijna dezelfde box. De achtste, box 13 met 0,72, is de beste box van de kleine driehoek.

    De tweede cel heeft 4 rondes nodig. Ronde 1 houdt box 6, de beste van de cirkel, met 0,84, en haalt de 7 boxes om de cirkel uit de lijst, box 6 zelf ook: er blijven er 21 over. De boxes van het vierkant blijven, want hun IoU met box 6 is hoogstens 0,31, niet boven 0,5. Ronde 2 houdt box 13 en haalt de 7 boxes van de kleine driehoek eruit; ronde 3 houdt box 0, de beste van het vierkant, met 0,70; ronde 4 houdt box 23, met 0,59, voor de grote driehoek, en de lijst is leeg. NMS houdt 4 boxes, één per vorm: `[6, 13, 0, 23]`. NMS vertrouwt op de scores: box 24 past beter om de grote driehoek, met een IoU van 0,85 tegen 0,79, maar hij heeft een lagere score, 0,56, dus box 23 blijft en box 24 gaat weg. Oefening 2 zet deze loop in een functie die ook de drempel krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="Vier boxes hebben deze scores: A 0,9, B 0,8, C 0,7 en D 0,6. A en B overlappen met een IoU van 0,7, B en C met 0,6, en C en D met 0,55; elk ander paar heeft een IoU onder 0,5. Welke boxes houdt NMS met een drempel van 0,5?",
        options={
            "right": "A en C",
            "dropped": "A en D",
            "only": "alleen A",
            "three": "A, C en D",
        },
        correct="right",
        why={
            "right": "Ronde 1 houdt A en gooit B weg, want zijn IoU met A is 0,7; C en D hebben met A een IoU onder 0,5, dus die blijven. Ronde 2 houdt C, de hoogste score die over is, en gooit D weg: 0,55 ligt boven 0,5. B heeft met C een IoU van 0,6, maar B was al weg, en een box die weggegooid is, gooit niets weg.",
            "dropped": "Dan gooit B box C weg. Maar A gooide B in ronde 1 weg, en een box die weggegooid is, gooit niets weg. Welke box heeft na ronde 1 de hoogste score?",
            "only": "Dan gooi je elke box weg waarvan de IoU met een box met een hogere score boven 0,5 ligt, ook als die box zelf al weggegooid is. Van de boxes met een hogere score heeft alleen B met C een IoU boven 0,5, en B is na ronde 1 weg.",
            "three": "C wordt in ronde 2 gehouden, en gooit dan elke box weg waarvan de IoU ermee boven 0,5 ligt. Wat is de IoU van C en D?",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_exercise_nms():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. NMS als functie

        Maak `nms(boxes, scores, threshold)` af. Die krijgt een lijst boxes, elk `(boven, links, onder, rechts)`, een lijst met de score van elke box, en een drempel. Hij geeft de nummers terug van de boxes die hij houdt: hun plaatsen in de lijst `boxes`. `iou` krijg je erbij, net als in het uitgewerkte voorbeeld. De check roept hem aan met de 28 boxes van de scène:

        ```python
        nms(boxes, scores, 0.5)
        ```

        Dat moet de 4 boxes van het uitgewerkte voorbeeld geven, 6, 13, 0 en 23, in welke volgorde dan ook. De check probeert ook andere drempels, twee boxes waarvan de IoU precies de drempel is, boxes die in een ketting overlappen, zoals in oefening 1, en helemaal geen boxes.
        """),
        L.task("Laat `nms(boxes, scores, threshold)` de nummers geven van de boxes die non-maximum suppression houdt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nms():
    nms_code = L.code_box("nms", lines=24)
    nms_code
    return (nms_code,)


@app.cell(hide_code=True)
def check_nms(nms_code):
    result_nms = L.check("nms", nms_code.value)
    result_nms
    return (result_nms,)


@app.cell(hide_code=True)
def hints_nms():
    L.hints(
        "nms",
        [
            "Stap 2 van het recept: begin met `best = rest[0]`, ga dan met een `for`-loop door `rest`, en maak elke box met een hogere score dan `best` de nieuwe `best`. Daarna `kept.append(best)`.",
            "Stap 3: houd in `rest` alleen de boxes waarvan de IoU met `best` hoogstens `threshold` is. Een list comprehension met een `if` aan het eind maakt die lijst. `best` zelf gaat ook weg, want zijn IoU met zichzelf is 1.",
            "Stap 2 en 3 komen binnen `while len(rest) > 0:`, zodat ze zich herhalen tot de lijst leeg is. De laatste regel van de loop: `rest = [i for i in rest if iou(boxes[i], boxes[best]) <= ___]`.",
        ],
        why="Voor de scène bij 0,5 houdt ronde 1 box 6, met de hoogste score, 0,84, en haalt de andere boxes van de cirkel uit de lijst; ronde 2 tot en met 4 houden box 13, 0 en 23. De functie geeft `[6, 13, 0, 23]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cars():
    mo.md(r"""
    ### Oefening 3. Vlak bij elkaar
    """)
    return


@app.cell(hide_code=True)
def choice_cars_cell():
    choice_cars = L.choice(
        "cars",
        question="Twee auto's staan vlak bij elkaar op een foto, de ene deels achter de andere. Een detector geeft goede boxes om allebei, en is het zekerst van de voorste auto. Elke box die hij voor de achterste auto geeft, heeft een IoU tussen 0,55 en 0,7 met de beste box van de voorste auto. Daarna doet hij NMS met een drempel van 0,5. Wat meldt hij?",
        options={
            "one": "één auto: de boxes van de achterste auto worden weggegooid",
            "two": "twee auto's: NMS ziet dat het verschillende voorwerpen zijn",
            "merged": "één grote box om beide auto's",
            "none": "geen auto: de twee boxes gooien elkaar weg",
        },
        correct="one",
        why={
            "one": "De beste box van de voorste auto wordt als eerste gehouden. Elke box van de achterste auto heeft daarmee een IoU boven 0,5, dus NMS gooit ze allemaal weg, alsof het kopieën zijn: twee auto's, één box. Een hogere drempel zou de tweede auto houden, maar zou ook kopieën van één auto doorlaten, zoals de slider liet zien vanaf 0,6.",
            "two": "NMS kijkt niet naar wat er in de boxes zit, alleen naar hun scores en hun IoU. Voor NMS is een box waarvan de IoU met een gehouden box boven de drempel ligt, een kopie.",
            "merged": "NMS maakt nooit een nieuwe box: het enige wat NMS doet, is boxes houden of weggooien.",
            "none": "Alleen een gehouden box gooit andere weg, en de box met de hoogste score wordt altijd gehouden. Welke box wordt als eerste gehouden?",
        },
    )
    choice_cars
    return (choice_cars,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 4. Soorten
    """)
    return


@app.cell(hide_code=True)
def explain_kinds_cell():
    explain_kinds = L.explain(
        "kinds",
        question="De detector van een latere les zegt ook wat voor soort voorwerp elke box bevat, en geeft de score voor die soort. In de scène ligt het vierkant deels onder de cirkel, en bij een drempel van 0,1 gooide de box van de cirkel elke box van het vierkant weg. Hoe kan de detector met de soort dat voorkomen? En wanneer helpt dat niet?",
        model_answer="**Doe NMS voor elke soort apart.** Een box gooit dan alleen boxes van zijn eigen soort weg, dus een box voor een cirkel gooit nooit een box voor een vierkant weg, hoe sterk ze ook overlappen: boxes van verschillende soorten gelden als verschillende voorwerpen. Zo houden een kopje en de theepot erachter allebei hun box; Faster R-CNN in torchvision werkt zo. **Het helpt niet bij twee voorwerpen van dezelfde soort**, zoals de twee auto's van oefening 3: hun boxes worden nog steeds vergeleken, en met de drempel kies je nog steeds tussen kopieën weggooien en buren houden. **Het kan ook te veel houden**: geeft de detector één voorwerp boxes van twee soorten, dan blijven ze allebei staan.",
        points=[
            "je zegt dat je NMS voor elke soort apart doet, zodat een box alleen boxes van dezelfde soort weggooit",
            "je zegt dat boxes van verschillende soorten als verschillende voorwerpen gelden, ook als ze sterk overlappen",
            "je noemt een geval waarin het niet helpt, zoals twee overlappende voorwerpen van dezelfde soort, of één voorwerp met boxes van twee soorten",
        ],
    )
    explain_kinds
    return (explain_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl) zei de IoU hoe goed een voorspelde box bij de echte box past. Hier zegt hij welke boxes van een detector over hetzelfde voorwerp gaan. Een detector geeft een hele groep boxes om elk voorwerp, elk met een confidence score. Non-maximum suppression houdt de box met de hoogste score, gooit elke box weg waarvan de IoU ermee boven een drempel ligt, en herhaalt dat met de boxes die over zijn; alleen gehouden boxes gooien andere weg. Met de drempel kies je tussen kopieën weggooien en buren houden: in de scène houdt 0,3 tot 0,55 de beste box van elke vorm, bij 0,1 verliest het vierkant onder de cirkel zijn box, en vanaf 0,6 blijven er kopieën staan. In de volgende les, *Een detector beoordelen: mean average precision*, worden de boxes die een detector houdt, vergeleken met de echte boxes, en bepalen hun scores in welke volgorde ze meetellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cars, choice_chain, explain_kinds, result_nms):
    L.wrap_up(choice_chain, result_nms, choice_cars, explain_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
