# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-iou. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-iou", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_coco():
    quick_coco = L.choice(
        "quick-coco",
        question="COCO schrijft een box als `[x, y, width, height]`. Welke box is `[10, 20, 5, 3]`, geschreven zoals de course het doet, `(boven, links, onder, rechts)` met beide uiteinden meegeteld?",
        options={
            "box": "`(20, 10, 22, 14)`",
            "plus": "`(20, 10, 23, 15)`",
            "xy": "`(10, 20, 14, 22)`",
            "swap": "`(20, 10, 24, 12)`",
        },
        correct="box",
        why={
            "box": "x = 10 is de linkerkant en y = 20 de bovenkant. 3 rijen vanaf rij 20 zijn de rijen 20, 21 en 22, dus de onderkant is 20 + 3 - 1 = 22; 5 kolommen vanaf kolom 10 eindigen bij kolom 14. Deze les werkt met boxes zoals de course ze schrijft.",
            "plus": "Die box is één rij en één kolom te groot: van rij 20 tot rij 23 zijn 4 rijen. Beide uiteinden horen bij de box, dus 3 rijen vanaf rij 20 eindigen bij rij 22.",
            "xy": "Daar staat de kolom eerst. De x van COCO is een kolom, de linkerkant, en de course schrijft de bovenste rij eerst.",
            "swap": "Die box is 5 rijen hoog en 3 kolommen breed. De breedte, 5, telt de kolommen, en de hoogte, 3, de rijen.",
        },
    )
    quick_coco
    return (quick_coco,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="Hoeveel pixels bevat de box `(2, 3, 4, 7)`, met beide uiteinden meegeteld?",
        options={
            "fifteen": "15",
            "eight": "8",
            "ten": "10",
            "many": "28",
        },
        correct="fifteen",
        why={
            "fifteen": "De rijen 2 tot en met 4 zijn 3 rijen, en de kolommen 3 tot en met 7 zijn 5 kolommen: 3 × 5 = 15 pixels. In deze les tel je de pixels van boxes, en van het stuk waar twee boxes overlappen.",
            "eight": "4 - 2 = 2 en 7 - 3 = 4 laten één uiteinde weg. Van rij 2 tot rij 4 zijn de rijen 2, 3 en 4: drie rijen.",
            "ten": "Zo tel je de kolommen met beide uiteinden en de rijen met maar één uiteinde. Van rij 2 tot rij 4 zijn drie rijen, geen twee.",
            "many": "4 × 7 vermenigvuldigt de onderkant en de rechterkant, de hoek van de box. De box begint bij rij 2 en kolom 3, niet bij 0: hoeveel rijen en kolommen beslaat hij?",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model voor object detection geeft een box terug voor elk voorwerp dat het vindt. Om het te trainen, en om te weten hoe goed het is, vergelijk je elke box die het voorspelt met de box van de annotatie, en die vergelijking moet één getal opleveren. Hoeveel pixels een box ernaast zit, zegt te weinig: een box die aan elke kant 3 pixels te groot is, zit bijna perfect om een bus van 100 pixels breed, en ver ernaast om een vogel van 10 pixels. Waar het om gaat, is hoeveel de twee boxes delen.

    **Een voorbeeld uit het dagelijks leven.** Jij en een vriend tekenen allebei op een kaart waar volgens jullie het festivalterrein ligt. Hoe goed zijn jullie het eens? Niet hoe ver de middens van jullie tekeningen uit elkaar liggen: een van jullie kan een veel groter stuk om hetzelfde midden hebben getekend. Wat telt, is het land dat jullie allebei tekenden, als deel van al het land dat minstens één van jullie tekende.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    guess_shift = L.guess(
        "shift",
        question="De **intersection over union** (IoU) van twee boxes is hun overlap gedeeld door de oppervlakte die ze samen bedekken. Een voorspelde box is even groot als de echte box, maar ligt een kwart van zijn breedte naar rechts. Wat is zijn IoU?",
        options={
            "most": "0,75",
            "three-fifths": "0,6",
            "half": "0,5",
            "quarter": "0,25",
        },
        reveal="**0,6.** De overlap is driekwart van een box. Samen bedekken de twee boxes een hele box plus het kwart dat uitsteekt: vijf kwart. Driekwart gedeeld door vijf kwart is 3/5 = 0,6. Elke box deelt driekwart met de andere, en toch is de IoU lager: wat de voorspelde box mist en wat hij erbij neemt, zijn allebei in zijn nadeel.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **intersection over union**, of IoU, zegt hoe goed twee boxes overeenkomen: het aantal pixels in hun overlap, gedeeld door het aantal pixels dat ze samen bedekken. Intersection is Engels voor doorsnede, hier de overlap; union is Engels voor vereniging, wat de twee boxes samen bedekken:
        """),
        L.formula(r"\text{IoU} = \frac{\text{overlap}}{\text{vereniging}} = \frac{\text{overlap}}{\text{oppervlakte}_a + \text{oppervlakte}_b - \text{overlap}}"),
        mo.md(r"""
        Het is een verhouding: 1 als de twee boxes gelijk zijn, 0 als ze niet overlappen, en daartussen voor al het andere. Omdat het een verhouding is, weegt de IoU hoe ver een box ernaast zit af tegen hoe groot de boxes zijn: dezelfde twee boxes twee keer zo groot getekend, met elke fout ook twee keer zo groot, krijgen dezelfde IoU. Een box die aan elke kant 3 pixels te groot is, krijgt 0,39 om een box van 10 bij 10 pixels, en 0,89 om een box van 100 bij 100: het grote voorwerp merkt weinig van die 3 pixels, en de IoU zegt dat ook.

        Voor twee boxes heb je geen pixels nodig, alleen hun vier getallen. De overlap van twee boxes is zelf ook een box. Hij begint waar beide boxes begonnen zijn: bij de grootste van de twee bovenkanten en de grootste van de twee linkerkanten. Hij eindigt waar de eerste ophoudt: bij de kleinste van de twee onderkanten en de kleinste van de twee rechterkanten. Als de boxes niet overlappen, staat die box binnenstebuiten: zijn onderkant ligt boven zijn bovenkant, of zijn rechterkant links van zijn linkerkant, dus zijn aantal rijen of kolommen komt uit op 0 of minder: tel het dan als 0, en de overlap is 0. De vereniging is beide oppervlaktes opgeteld, min de overlap, want de overlap ligt in allebei.

        Voor masks werkt hetzelfde pixel voor pixel: `(m1 & m2).sum()` telt de overlap en `(m1 | m2).sum()` de vereniging. Dat werkt voor elke vorm, ook voor een box die je hebt omgezet in een mask.

        Hoe hoog moet een IoU zijn? PASCAL VOC, een bekende wedstrijd in object detection, telt een voorspelde box als gevonden als zijn IoU met de echte box boven 0,5 ligt.
        """),
        L.watch_out("De vereniging is niet de twee oppervlaktes opgeteld. De overlap ligt in beide boxes, dus als je ze optelt, tel je hem dubbel: trek hem er één keer af. Zonder dat krijgen twee boxes die gelijk zijn 0,5 in plaats van 1."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **intersection over union** (IoU) | hoe goed twee boxes of masks overeenkomen: de pixels van hun overlap gedeeld door de pixels die ze samen bedekken, van 0 tot 1 | `overlap / union` |
        | **overlap** | de pixels die in beide boxes of masks liggen; voor twee boxes een box van de grootste bovenkanten en linkerkanten tot de kleinste onderkanten en rechterkanten | `(a & b).sum()` |
        | **vereniging** | de pixels die in minstens één van de twee liggen: beide oppervlaktes, min de overlap | `(a | b).sum()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_overlap():
    mo.vstack([
        mo.md("**De overlap en de vereniging van twee boxes**"),
        mo.Html(C.overlap_picture({
            "boxes": "echte en voorspelde box",
            "overlap": "overlap: {overlap} pixels",
            "union": "vereniging: {union} pixels",
            "alt": "Drie plaatjes van hetzelfde stuk van een scène. Links een rode cirkel die op een blauw vierkant ligt, in een strakke box getekend als doorgetrokken lijn, en een gestreepte box die iets lager en verder naar links ligt, en iets groter is. In het midden, op zwart, het stuk waar de twee boxes overlappen in wit, en de rest van beide boxes in donkergrijs. Rechts alles wat de twee boxes bedekken in wit. Het middelste en het rechter plaatje tonen ook de lijnen van de twee boxes.",
        })),
        mo.md(r"""
        Links de cirkel uit het uitgewerkte voorbeeld met zijn echte box, doorgetrokken, en een box die een model ervoor zou kunnen voorspellen, gestreept. In het midden hun overlap in wit: 700 pixels. Rechts hun vereniging in wit: 1101 pixels. De IoU is 700 / 1101 = 0,64.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoe ver mag een box ernaast zitten? De slider schuift een voorspelde box, even groot als de echte box van de cirkel, naar rechts, één pixel per stap. Links de twee boxes; rechts de IoU bij elke stap, met een stippellijn bij 0,5.
    """)
    return


@app.cell(hide_code=True)
def explore_shift_cell():
    def _frame(d):
        d = int(d)
        _html, _o = C.shift_frame(d, {"title": "{shift} pixels verschoven", "iou": "IoU met de echte box", "shift": "pixels verschoven", "point": ",",
                                      "alt": "De echte box van een rode cirkel, doorgetrokken, en een even grote gestreepte box, {shift} pixels naar rechts verschoven: IoU {iou}. Ernaast de IoU bij elke stap van 0 tot 29 pixels: 1 bij 0, aan het begin het snelst dalend, door de stippellijn bij 0,5 tussen 9 en 10 pixels, tot 0 bij 29. De stap van {shift} pixels is gemarkeerd met een grotere punt."})
        return _html, f"Verschoven over {d} pixels: overlap {_o['overlap']} pixels, vereniging {_o['union']} pixels, IoU {L.number(_o['iou'])}."

    explore_shift = L.explore("shift-slider", "pixels verschoven", _frame, start=0, values=C.SHIFTS, role="",
                              task="Zoek hoeveel pixels je de box hoogstens kunt verschuiven terwijl de IoU nog boven 0,5 ligt.")
    explore_shift
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Is de box 9 pixels verschoven, dan is de IoU 0,53; bij 10 pixels is hij 0,49, onder de 0,5 van PASCAL VOC. De box is 29 pixels breed, dus 10 pixels opzij, net iets meer dan een derde van zijn breedte, is al te veel: 19 van zijn 29 kolommen worden nog gedeeld, maar samen beslaan de twee boxes 39 kolommen. Over zijn hele breedte verschoven, 29 pixels, raken de twee boxes elkaar alleen, en is de IoU 0. De curve daalt aan het begin het snelst: de eerste pixels ernaast kosten het meest.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Maak elk aantal apart minstens 0, voordat je vermenigvuldigt. Bij twee boxes die zowel omlaag als opzij uit elkaar liggen, komen de rijen en de kolommen van de overlap allebei onder 0 uit, en twee negatieve getallen vermenigvuldigd geven een positieve overlap."),
        mo.md(r"""
        ### Recept

        Om te meten hoe goed een voorspelde box bij de echte box past:

        1. Zoek de overlap: de grootste van de twee bovenkanten en van de twee linkerkanten, de kleinste van de twee onderkanten en van de twee rechterkanten.
        2. Tel zijn pixels: zijn rijen, onder - boven + 1, keer zijn kolommen, rechts - links + 1, elk 0 gemaakt als het onder 0 uitkomt.
        3. Tel de vereniging: de pixels van beide boxes opgeteld, min de overlap.
        4. Deel de overlap door de vereniging: 1 voor een perfecte box, 0 voor een misser.
        5. Om te controleren, of voor masks, tel je pixels: `(a & b).sum()` is de overlap en `(a | b).sum()` de vereniging.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept op beide manieren, op de scène van [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl), opnieuw gemaakt met `scene(18)`. De eerste cel neemt de box van de cirkel, `s.shapes[3].box`, en een box die een model ervoor zou kunnen voorspellen, en volgt stap 5: hij telt pixels. `box_mask` zet een box om in een mask zo groot als de scène: `np.zeros` met `dtype=bool` maakt een array vol `False`, en als je de slice van de box `True` maakt, wordt elke pixel erin in één keer `True`, met + 1 omdat een slice net voor zijn eind stopt. Daarna geven `&` en `|` de overlap en de vereniging, en `sum` telt hun pixels. De tweede cel vergelijkt de voorspelde box op dezelfde manier met de box van elke vorm in de scène. De derde cel volgt stap 1 tot en met 4, met alleen de vier getallen van elke box, voor het vierkant en de grote driehoek.
    """)
    return


@app.cell
def example_masks():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    truth = s.shapes[3].box     # de box van de cirkel
    pred = (62, 46, 91, 77)     # een box die een model ervoor zou kunnen voorspellen


    def box_mask(box):
        # een mask zo groot als de scène: True in de rijen en kolommen van de box
        top, left, bottom, right = box
        mask = np.zeros((128, 128), dtype=bool)
        mask[top:bottom + 1, left:right + 1] = True
        return mask


    a = box_mask(truth)
    b = box_mask(pred)
    # 5. tel pixels: de overlap en de vereniging
    overlap = (a & b).sum()
    union = (a | b).sum()
    print(truth, a.sum(), pred, b.sum())
    print(overlap, union, overlap / union)
    return b, box_mask, pred, s


@app.cell
def example_others(b, box_mask, s):
    for shape in s.shapes:
        other = box_mask(shape.box)
        # round(x, 2): x afgerond op twee decimalen
        print(shape.kind, shape.box, round((other & b).sum() / (other | b).sum(), 2))
    return


@app.cell
def example_steps(pred, s):
    # stap 1 tot 4 met alleen de vier getallen van elke box: de voorspelde box
    # tegen de box van het vierkant en de box van de grote driehoek
    for t in (s.shapes[2].box, s.shapes[1].box):
        # 1. de overlap: de grootste bovenkanten en linkerkanten, de kleinste onderkanten en rechterkanten
        top, left = max(t[0], pred[0]), max(t[1], pred[1])
        bottom, right = min(t[2], pred[2]), min(t[3], pred[3])
        # 2. zijn rijen en zijn kolommen, elk minstens 0, en zijn pixels
        rows = max(0, bottom - top + 1)
        cols = max(0, right - left + 1)
        both = rows * cols
        # 3. de vereniging: beide oppervlakken opgeteld, min de overlap
        area_t = (t[2] - t[0] + 1) * (t[3] - t[1] + 1)
        area_pred = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        union_t = area_t + area_pred - both
        # 4. de overlap gedeeld door de vereniging
        print(t, bottom - top + 1, right - left + 1)
        print(both, union_t, round(both / union_t, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De box van de cirkel bevat 841 pixels en de voorspelde box 960. Ze delen er 700, en samen bedekken ze er 1101: 841 + 960 = 1801, min de 700 die in allebei liggen. De IoU is 700 / 1101 = 0,64, boven 0,5: volgens de regel van PASCAL VOC heeft de voorspelde box de cirkel gevonden. Hij overlapt ook de box van het vierkant, met een IoU van 0,19, en de driehoeken helemaal niet; zijn IoU is het hoogst voor de cirkel, dus van alle vormen past hij het best bij de cirkel. De derde cel vindt de 0,19 van het vierkant zonder mask: een overlap van 12 rijen en 27 kolommen, 324 pixels, en een vereniging van 1724. Bij de grote driehoek komen de kolommen uit op -16: de driehoek houdt op bij kolom 29, en de voorspelde box begint bij kolom 46. `max(0, ...)` maakt daar 0 van, dus de overlap is 0. Oefening 2 zet deze stappen in een functie die werkt voor elke twee boxes, en de check probeert hem op de cirkel: 0,64.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De overlap van twee boxes
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="Wat is de overlap van de boxes `(10, 20, 30, 40)` en `(25, 5, 50, 35)`, geschreven als `(boven, links, onder, rechts)`?",
        options={
            "right": "`(25, 20, 30, 35)`",
            "around": "`(10, 5, 50, 40)`",
            "largest": "`(25, 20, 50, 40)`",
            "smallest": "`(10, 5, 30, 35)`",
        },
        correct="right",
        why={
            "right": "De overlap begint waar beide boxes begonnen zijn: bij rij 25, de grootste bovenkant, en bij kolom 20, de grootste linkerkant. Hij eindigt waar de eerste box ophoudt: bij rij 30, de kleinste onderkant, en bij kolom 35, de kleinste rechterkant. Hij bevat 6 rijen van 16 kolommen: 96 pixels.",
            "around": "Dat is de kleinste box om beide boxes heen: hij bevat elke pixel van allebei, en meer. De overlap bevat alleen de pixels die in allebei liggen: ligt rij 10 in de tweede box?",
            "largest": "Zo neem je het grootste getal van elk paar, ook voor de onderkant en de rechterkant. Rij 50 ligt alleen in de tweede box, want de eerste houdt op bij rij 30. Waar houdt de overlap op?",
            "smallest": "Zo neem je het kleinste getal van elk paar, ook voor de bovenkant en de linkerkant. Rij 10 ligt alleen in de eerste box, want de tweede begint pas bij rij 25. Waar begint de overlap?",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_exercise_iou():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. IoU uit vier getallen

        Maak `iou(a, b)` af. Die krijgt twee boxes zoals de course ze schrijft, `(boven, links, onder, rechts)` met beide uiteinden meegeteld, en geeft hun IoU zonder een mask te maken. `area` staat er al: die telt de pixels van een box. De check roept hem aan met de cirkel en de voorspelde box uit het uitgewerkte voorbeeld:

        ```python
        iou((58, 50, 86, 78), (62, 46, 91, 77))
        ```

        Dat moet hetzelfde geven als het tellen van de pixels: 700 / 1101, ongeveer 0,64. De check probeert ook boxes die niet overlappen, boxes die elkaar alleen raken, en een kleine box in een grote.
        """),
        L.task("Laat `iou(a, b)` de intersection over union van de boxes `a` en `b` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iou():
    iou_code = L.code_box("iou", lines=14)
    iou_code
    return (iou_code,)


@app.cell(hide_code=True)
def check_iou(iou_code):
    result_iou = L.check("iou", iou_code.value)
    result_iou
    return (result_iou,)


@app.cell(hide_code=True)
def hints_iou():
    L.hints(
        "iou",
        [
            "Stap 1 van het recept: de overlap begint waar beide boxes begonnen zijn. Welke van twee bovenkanten is dat, de kleinste of de grootste? En waar houdt hij op?",
            "Stap 2: tel de rijen en de kolommen zoals `area` dat doet, maar maak elk 0 als het onder 0 uitkomt. `max(0, n)` geeft de grootste van 0 en n.",
            "`top = max(a[0], b[0])` en `bottom = min(a[2], b[2])`, en hetzelfde voor left en right; `overlap = max(0, bottom - top + 1) * max(0, ___)`; `union = area(a) + area(b) - ___`.",
        ],
        why="Voor de cirkel en de voorspelde box loopt de overlap van rij 62 tot rij 86, 25 rijen, en van kolom 50 tot kolom 77, 28 kolommen: 700 pixels. De vereniging is 841 + 960 - 700 = 1101, en 700 / 1101 = 0,64.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_middle():
    mo.md(r"""
    ### Oefening 3. Het goede midden
    """)
    return


@app.cell(hide_code=True)
def choice_middle_cell():
    choice_middle = L.choice(
        "middle",
        question="Een echte box is 20 bij 20 pixels. Een detector voorspelt een box van 40 bij 40 pixels met precies hetzelfde midden, dus de afstand tussen hun middens is 0. Wat is zijn IoU?",
        options={
            "one": "1",
            "half": "0,5",
            "quarter": "0,25",
            "zero": "0",
        },
        correct="quarter",
        why={
            "quarter": "De kleine box ligt helemaal in de grote: de overlap is de kleine box, 400 pixels, en de vereniging de grote, 1600 pixels. 400 / 1600 = 0,25, ruim onder 0,5. De afstand tussen de middens zegt niets over de grootte, en zou deze box perfect noemen; de IoU ziet dat driekwart ervan achtergrond is.",
            "one": "De middens zijn gelijk, maar de boxes niet: de voorspelde box bedekt vier keer zoveel pixels. Wat zijn de overlap en de vereniging?",
            "half": "Twee keer zo breed en twee keer zo hoog is vier keer de oppervlakte, niet twee keer. Hoeveel pixels bevatten de twee boxes?",
            "zero": "0 is voor boxes die helemaal niet overlappen. Hier ligt de kleine box helemaal in de grote: wat is hun overlap?",
        },
    )
    choice_middle
    return (choice_middle,)


@app.cell(hide_code=True)
def text_exercise_big():
    mo.md(r"""
    ### Oefening 4. Eén box over alles
    """)
    return


@app.cell(hide_code=True)
def explain_big_cell():
    explain_big = L.explain(
        "big",
        question="Een detector legt één box over het hele beeld, dus die bevat elk voorwerp. Waarom geeft de IoU die box een lage score, terwijl de overlap gedeeld door de oppervlakte van de echte box 1 zou geven? En wat doet de IoU met een box die te klein is?",
        model_answer="**De vereniging bevat elke pixel van de voorspelde box.** Gedeeld door alleen de echte box krijgt een box die het hele voorwerp bevat een 1, hoeveel achtergrond hij ook meeneemt. **De IoU deelt door wat beide boxes samen bedekken**, dus die achtergrond is in zijn nadeel: een box over de hele scène krijgt 841 / 16384 = 0,05 voor de cirkel. **Een te grote en een te kleine box scoren allebei laag.** Bij een te grote box groeit de vereniging, bij een te kleine krimpt de overlap, en alleen een box die past komt dicht bij 1.",
        points=[
            "je zegt dat de vereniging elke pixel van de voorspelde box bevat, dus dat de achtergrond die hij meeneemt in zijn nadeel is",
            "je zegt dat delen door alleen de echte box de pixels negeert die te veel zijn, dus dat een box over alles 1 zou scoren",
            "je zegt dat de IoU alleen hoog is als de box niet te groot en niet te klein is",
        ],
    )
    explain_big
    return (explain_big,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl) gaf een annotatie elk voorwerp een box. De IoU zegt hoe goed een voorspelde box daarbij past, als één getal van 0 tot 1 dat de fout afweegt tegen de grootte van de box. Hij heeft alleen de vier getallen van elke box nodig: de overlap loopt van de grootste bovenkanten en linkerkanten tot de kleinste onderkanten en rechterkanten, en de vereniging is beide oppervlaktes min de overlap. Voor masks tel je pixels met `&` en `|`. PASCAL VOC telt een box als gevonden als zijn IoU boven 0,5 ligt, en een box die een derde van zijn breedte opzij ligt, komt daar al niet meer boven. In de volgende les, *Eén box per voorwerp: non-maximum suppression*, stelt een detector veel boxes voor één voorwerp voor, en zegt de IoU welke ervan te veel overlappen om over verschillende voorwerpen te gaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_middle, choice_overlap, explain_big, result_iou):
    L.wrap_up(choice_overlap, result_iou, choice_middle, explain_big)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
