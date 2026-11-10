# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-annotations. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-annotations", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vormen tellen: connected components](/course/open/cv-connected-components?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_box():
    quick_box = L.choice(
        "quick-box",
        question="Een vorm heeft deze pixels, als (rij, kolom): `(1, 3)`, `(2, 3)`, `(2, 4)` en `(3, 2)`. Wat is zijn bounding box, geschreven zoals in die les, `(boven, links, onder, rechts)`?",
        options={
            "box": "`(1, 2, 3, 4)`",
            "xy": "`(2, 1, 4, 3)`",
            "ends": "`(1, 3, 3, 2)`",
            "size": "`(1, 2, 3, 3)`",
        },
        correct="box",
        why={
            "box": "De rijen lopen van 1 tot 3 en de kolommen van 2 tot 4: boven 1, links 2, onder 3, rechts 4. Deze les schrijft dezelfde box ook nog op een andere manier.",
            "xy": "Dat zet de kolommen eerst: links, boven, rechts, onder. De course schrijft de rijen eerst, zoals NumPy.",
            "ends": "Dat zijn de eerste en de laatste pixel van de lijst. De box loopt van de kleinste tot de grootste rij, en van de kleinste tot de grootste kolom: welke kolom is de kleinste?",
            "size": "De laatste twee getallen zijn de hoogte en de breedte van de box, 3 en 3. De course schrijft de onderkant en de rechterkant: de grootste rij en de grootste kolom.",
        },
    )
    quick_box
    return (quick_box,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="Een vorm van 12 pixels heeft de bounding box `(0, 0, 3, 3)`. Hoeveel pixels in zijn box horen niet bij de vorm?",
        options={
            "four": "4",
            "zero": "0",
            "none": "Geen: 12 pixels passen niet in die box",
        },
        correct="four",
        why={
            "four": "De box loopt van rij 0 tot rij 3 en van kolom 0 tot kolom 3, allebei meegeteld: 4 rijen van 4 kolommen, 16 pixels. 12 daarvan zijn de vorm, dus 4 niet: de vorm vult 12 van de 16, 75% van zijn box. Hoeveel van zijn box een vorm vult, komt terug in deze les.",
            "zero": "Dan zou de vorm zijn box precies vullen, zoals een rechtopstaande rechthoek doet. Hoeveel pixels bevat een box van rij 0 tot rij 3 en van kolom 0 tot kolom 3?",
            "none": "Van rij 0 tot rij 3 zijn 4 rijen, geen 3: de bovenste en de onderste rij horen allebei bij de box. En zo ook voor de kolommen.",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leert van voorbeelden met het goede antwoord, en hoe dat antwoord eruitziet, hangt af van wat het model moet doen: één woord per foto om foto's te sorteren, een rechthoek om elke auto om de auto's in een straat te vinden, elke pixel van een voorwerp om het uit een foto te knippen, de precieze plek van elke vingertop om een hand te volgen. Hoe preciezer het antwoord, hoe langer iemand eraan werkt. Dus voordat iemand ook maar één beeld labelt, kiest iemand de vorm van de antwoorden, en die keuze bepaalt wat een model ervan kan leren en wat de data kost.

    **Een voorbeeld uit het dagelijks leven.** De foto's op je telefoon. Zoeken op "hond" hoeft alleen te weten dat er een hond in een foto staat. Een foto bijsnijden tot de hond moet weten waar hij staat. De hond als sticker uit de foto lichten heeft elke pixel van hem nodig. En een filter dat hem een zonnebril opzet, moet precies weten waar zijn ogen zitten. Eén foto, vier vragen, en elke vraag heeft een ander antwoord nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_cost():
    guess_cost = L.guess(
        "cost-guess",
        question="Een box van goede kwaliteit tekenen kost 7 tot 35 seconden, afhankelijk van hoe je hem tekent. Hoe lang kostte het mask van één voorwerp in COCO, een bekende dataset van foto's?",
        options={
            "same": "Ongeveer even lang: 10 tot 35 seconden",
            "minute": "Meer dan een minuut",
            "ten": "Ongeveer tien minuten",
        },
        reveal="**Meer dan een minuut.** De annotatoren van COCO besteedden meer dan 22 uur aan elke 1000 masks: meer dan 79 seconden per voorwerp, twee tot elf keer zo lang als een box. Een box heeft alleen de vier buitenste kanten van een voorwerp nodig; een mask volgt elke bocht van zijn omtrek. Hoe preciezer een annotatie, hoe meer ze kost.",
    )
    guess_cost
    return (guess_cost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **annotatie** is het goede antwoord dat mensen met de hand aan een voorbeeld toevoegen, zoals een label of een box; de mensen die dat doen, heten annotatoren. Bij een beeld bepaalt de taak de vorm van zijn annotatie. Er zijn er vier, van grof tot precies:

        - **Een label per beeld.** Voor classificatie: het hele beeld krijgt één klasse, zoals `triangle`. Een label zegt wat het beeld laat zien, niet waar, dus het past bij een beeld met één ding erin.
        - **Een box per voorwerp.** Voor **object detection**: elk voorwerp van bekende klassen in een beeld vinden, elk met een bounding box en zijn klasse. Vier getallen per voorwerp zijn snel getekend, maar een box bevat ook achtergrond, en vaak een stuk van een ander voorwerp.
        - **Een mask per voorwerp.** Een **segmentation mask** is een binair beeld zo groot als het beeld: `True` voor elke pixel van het voorwerp, `False` op alle andere plekken. Een connected component uit [Vormen tellen: connected components](/course/open/cv-connected-components?lang=nl) is ook een mask. De box volgt uit een mask, maar een mask volgt niet uit een box. Een mask is het preciest, en kost het meeste tijd om te tekenen.
        - **Keypoints per voorwerp.** Een **keypoint** is één precieze plek op een voorwerp, zoals een hoek van een vierkant, het puntje van een neus of een linkerelleboog. Elk voorwerp van een klasse heeft dezelfde keypoints, altijd in dezelfde volgorde.

        Voorwerpen bedekken elkaar. In de scènes van de course ligt een vorm die later getekend is, boven op de eerdere. Een mask bevat dan alleen de pixels die je kunt zien, en de box is de kleinste box om het mask. Een hoek onder een andere vorm is nog steeds een keypoint. Elk keypoint krijgt een vlag v, zoals COCO het schrijft: 2 als het te zien is, 1 als het er is maar bedekt, en 0 als het niet gelabeld is, en dan is zijn plek ook `0, 0`. In de scènes van de course is een keypoint de pixel van de vorm die het dichtst bij een hoek ligt, en het is bedekt als die pixel niet in het mask zit.

        Een dataset schrijft zijn annotaties in een vast formaat. De course schrijft een box als `(boven, links, onder, rechts)`, beide uiteinden meegeteld. COCO schrijft `[x, y, width, height]`: x is de kolom van de linkerkant en y de rij van de bovenkant, allebei geteld vanaf 0 linksboven, en width en height zijn het aantal kolommen en rijen dat de box beslaat, de breedte en de hoogte. De keypoints van een voorwerp schrijft COCO als één lijst met drie getallen per keypoint: `[x1, y1, v1, x2, y2, v2, ...]`.
        """),
        L.watch_out("Een box uit een dataset of een library is vier getallen, en niets daarin zegt welke vier. Zoek het formaat op voordat je er een gebruikt: komt de kolom eerst of de rij, en zijn de laatste twee getallen een hoek of een breedte en een hoogte? Een box in het verkeerde formaat geeft geen fout: hij komt gewoon op de verkeerde plek terecht."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **annotatie** | het goede antwoord dat mensen met de hand aan een voorbeeld toevoegen: een label, een box, een mask of keypoints | `shape.kind`, `shape.box`, `shape.mask` en `shape.keypoints` in het uitgewerkte voorbeeld |
        | **object detection** | elk voorwerp van bekende klassen in een beeld vinden, elk met een bounding box en zijn klasse | een box en een klasse per voorwerp; COCO `[x, y, width, height]` |
        | **segmentation mask** | een binair beeld zo groot als het beeld: `True` voor elke pixel van één voorwerp | `shape.mask`, een array van `bool` |
        | **keypoint** | één precieze plek op een voorwerp, met een vlag v: 2 te zien, 1 bedekt, 0 niet gelabeld | `(rij, kolom, v)`; COCO `[x, y, v]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_annotations():
    mo.vstack([
        mo.md("**Vier soorten annotatie**"),
        mo.Html(C.annotations_picture({
            "label": "een label per beeld: driehoek",
            "boxes": "een box per voorwerp",
            "mask": "een mask per voorwerp: het vierkant",
            "keypoints": "keypoints: de hoeken",
            "alt": "Vier plaatjes. Linksboven een beeld met één paarse driehoek op grijs, met het label driehoek. Rechtsboven een scène met een blauwe driehoek, een blauw vierkant, een rode cirkel boven op het vierkant en een kleine gele driehoek, elk in een rechthoek die zo krap mogelijk past: de boxes van de cirkel en het vierkant overlappen. Linksonder het mask van het vierkant: wit waar het vierkant te zien is, met een ronde hap waar de cirkel erop ligt; de andere vormen donkergrijs, op zwart. Rechtsonder dezelfde scène met een stip op elke hoek van het vierkant en de driehoeken, en een open rondje op de hoek van het vierkant die onder de cirkel ligt.",
        })),
        mo.md(r"""
        Linksboven een beeld met één vorm, en zijn label. Dan de scène van het uitgewerkte voorbeeld: een box om elke vorm; het mask van het vierkant in wit, zonder de pixels die de cirkel bedekt, en de andere vormen in donkergrijs, elk met een eigen mask; en de hoeken van het vierkant en de driehoeken als keypoints, een stip waar een hoek te zien is en een open rondje waar hij bedekt is. Een cirkel heeft geen hoeken, dus geen keypoints.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Een box is snel getekend, maar hij bevat achtergrond, en hoeveel hangt af van de vorm en van hoe die ligt. In 2000 scènes van de course vult een driehoek die door niets bedekt wordt gemiddeld 50% van zijn box, en een cirkel 79%: een driehoek met één zijde langs zijn box vult er precies de helft van. De slider draait een vierkant in stappen van 5 graden. Links het vierkant in zijn box; rechts hoeveel van zijn box het vierkant vult bij elke hoek.
    """)
    return


@app.cell(hide_code=True)
def explore_turn_cell():
    def _frame(a):
        a = int(a)
        _html, _o = C.turn_frame(a, {"title": "{angle}° gedraaid", "fill": "hoeveel van zijn box het vierkant vult", "angle": "hoek in graden",
                                     "alt": "Een blauw vierkant, {angle} graden gedraaid, in zijn box: het vult {fill}% van de box. Ernaast hoeveel van zijn box het vierkant vult bij elke hoek van 0 tot 90 graden: 100% rechtop, omlaag tot 50% bij 45 graden, en weer omhoog tot 100% bij 90 graden. {angle} graden is gemarkeerd met een grotere punt."})
        return _html, f"Bij een draai van {a}°: het vierkant vult {_o['fill']}% van zijn box."

    explore_turn = L.explore("turn", "hoek", _frame, start=0, values=C.ANGLES, role="",
                             task="Zoek de hoek waarbij het vierkant het minst van zijn box vult.")
    explore_turn
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Rechtop, bij 0° en bij 90°, vult het vierkant zijn hele box. 45° gedraaid staat het op een hoek en vult het er maar 50% van: de vier hoeken van de box zijn achtergrond. Daartussen vult het van 52% tot 85%. En achtergrond is niet het enige wat een box meeneemt: in dezelfde 2000 scènes bevat 54% van de boxes ook pixels van een andere vorm. Vormen bedekken ook hoeken: 6,6% van de hoeken ligt onder een vorm die later getekend is, keypoints met v = 1.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Laat een bedekt keypoint in de lijst staan, met v = 1, en schrijf een keypoint dat niet gelabeld is als `0, 0` met v = 0. Laat je er een weg, dan schuift elk keypoint erna een plek op: een programma leest de vierde hoek dan als de derde."),
        mo.md(r"""
        ### Recept

        Om beelden voor een taak te annoteren:

        1. Bepaal wat het model moet teruggeven: één klasse per beeld, waar elk voorwerp is, welke pixels erbij horen, of precieze plekken erop.
        2. Kies de vorm die precies dat geeft: een label, een box, een mask of keypoints. Een preciezere vorm vertelt het model meer, en kost meer tijd per voorwerp.
        3. Leg het formaat vast, en schrijf het op: de volgorde van de getallen, waar 0 is, of een box eindigt in een hoek of in een breedte en een hoogte, en de volgorde van de keypoints.
        4. Leid af wat eruit volgt: de box van een mask loopt van de kleinste tot de grootste rij en kolom van zijn pixels. Andersom werkt niet.
        5. Markeer wat bedekt is in plaats van het weg te laten: een keypoint onder een ander voorwerp krijgt v = 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel maakt de scène van het plaatje met `scene(18)`, uit de eigen module van de course `everything_ai.scenes`: dezelfde seed geeft op elke computer dezelfde scène. Elke vorm in `s.shapes` draagt zijn annotatie: `kind` is zijn klasse, `box` zijn box zoals de course hem schrijft, `mask` zijn segmentation mask, een array van `bool` van 128 bij 128, en `keypoints` zijn hoeken als `(rij, kolom, v)`. `area` is het aantal pixels van de hele vorm, de bedekte meegeteld. De cel print ze per vorm, met het aantal pixels in het mask. De tweede cel volgt stap 4 van het recept: `box_of_mask` leidt de box af uit het mask, en `fill` meet hoeveel van zijn box een vorm vult. Hij print of de afgeleide box gelijk is aan die in de annotatie, en de vulling.
    """)
    return


@app.cell
def example_scene():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    for shape in s.shapes:
        print(shape.kind, shape.box, shape.mask.sum(), shape.area, shape.keypoints)
    return np, s


@app.cell
def example_fill(np, s):
    def box_of_mask(mask):
        # 4. de rijen en de kolommen met minstens één pixel van het mask
        rows = np.arange(len(mask))[mask.sum(axis=1) > 0]
        cols = np.arange(len(mask[0]))[mask.sum(axis=0) > 0]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


    def fill(shape):
        # hoeveel van zijn box de pixels van zijn mask vullen
        top, left, bottom, right = shape.box
        return shape.mask.sum() / ((bottom - top + 1) * (right - left + 1))


    # _shape, omdat de cel hierboven al een naam shape heeft
    for _shape in s.shapes:
        same = box_of_mask(_shape.mask) == _shape.box
        # round(x, 2): x afgerond op twee decimalen
        print(_shape.kind, same, round(fill(_shape), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De scène heeft twee driehoeken, een vierkant en een cirkel, in die volgorde getekend. Elke box volgt uit zijn mask: vier keer `True`. Het vierkant heeft 959 pixels, maar de cirkel, die erna getekend is, bedekt er 306: zijn mask bevat de 653 die je kunt zien, dus het vult maar 60% van zijn box. Zijn derde hoek, op rij 75 en kolom 68, ligt onder de cirkel: v = 1. Die hoek ligt zelfs buiten de box, die bij rij 73 ophoudt: de box bevat alleen wat te zien is. De driehoeken vullen 54% en 48% van hun boxes, de cirkel 83%. De cirkel heeft geen keypoints: zijn lijst is leeg.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke annotatie?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="Een model moet voor elk blad in een foto zeggen hoeveel ervan bedekt is met bruine vlekken, als percentage van het blad. Welke annotatie hebben zijn trainingsfoto's nodig?",
        options={
            "label": "Een label per foto: gezond of ziek",
            "box": "Een box om elk blad en elke vlek",
            "mask": "Een mask van elk blad en elke vlek",
            "keypoints": "Een keypoint midden in elke vlek",
        },
        correct="mask",
        why={
            "mask": "Een percentage van een blad is een aantal pixels gedeeld door een aantal pixels: de pixels van de vlekken op het blad, gedeeld door de pixels van het blad. Alleen masks zeggen welke pixels dat zijn.",
            "label": "Een label zegt of een blad ziek is, niet hoeveel ervan. Het model moet een percentage geven: welke pixels moet het daarvoor tellen?",
            "box": "Een box bevat ook achtergrond: een driehoek vult maar de helft van zijn box, en een ronde vlek ongeveer 79%. Wie de pixels van boxes telt, krijgt het verkeerde percentage. Welke annotatie zegt welke pixels de vlek zijn?",
            "keypoints": "Een punt zegt waar een vlek is, niet hoe groot hij is. Hoe zou je zijn oppervlakte meten aan één punt?",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_coco():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Van de box van de course naar COCO

        Maak `to_coco(box)` af, die een box neemt zoals de course hem schrijft, `(boven, links, onder, rechts)` met beide uiteinden meegeteld, en hem teruggeeft zoals COCO hem schrijft: `[x, y, width, height]`. De check roept hem aan met de box van het vierkant uit het uitgewerkte voorbeeld:

        ```python
        to_coco((42, 39, 73, 72))
        ```

        Dat moet `[39, 42, 34, 32]` geven. De check probeert ook andere boxes, waaronder een box van één pixel.
        """),
        L.task("Laat `to_coco(box)` de box teruggeven als `[x, y, width, height]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_coco():
    coco_code = L.code_box("coco", lines=9)
    coco_code
    return (coco_code,)


@app.cell(hide_code=True)
def check_coco(coco_code):
    result_coco = L.check("coco", coco_code.value)
    result_coco
    return (result_coco,)


@app.cell(hide_code=True)
def hints_coco():
    L.hints(
        "coco",
        [
            "x en y zijn een kolom en een rij: welke kant van de box is een kolom, en welke een rij? En hoeveel kolommen beslaat een box van kolom 3 tot kolom 5?",
            "`fill` in het uitgewerkte voorbeeld telt de kolommen en de rijen van een box: kijk waardoor hij deelt.",
            "`x = left`, `y = top`, `width = right - left + ___` en `height = ___`.",
        ],
        why="De box van het vierkant loopt van kolom 39 tot kolom 72: dat zijn 72 - 39 + 1 = 34 kolommen, zijn breedte. En van rij 42 tot rij 73: 32 rijen, zijn hoogte. COCO zet de kolom eerst, zoals x in een grafiek: `[39, 42, 34, 32]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Oefening 3. Een poot achter een tafel
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden",
        question="Een stoel krijgt een keypoint onderaan elk van zijn vier poten, altijd in dezelfde volgorde. In een foto staat één poot achter een tafel, maar je kunt zien waar zijn voet moet staan. Hoe schrijf je dat keypoint, zoals COCO het doet?",
        options={
            "one": "`[x, y, 1]`, op de plek waar de voet moet staan",
            "two": "`[x, y, 2]`, op de plek waar de voet moet staan",
            "zero": "`[0, 0, 0]`",
            "out": "Laat hem weg: de stoel heeft dan drie keypoints",
        },
        correct="one",
        why={
            "one": "De voet is er, maar je kunt hem niet zien: v = 1, op de plek waar hij moet staan, zoals de derde hoek van het vierkant in het uitgewerkte voorbeeld. De annotatie houdt vast waar de voet is, en zegt dat hij bedekt is.",
            "two": "2 zegt dat het keypoint te zien is, en deze voet staat achter de tafel. Welke vlag is voor een keypoint dat er is maar bedekt?",
            "zero": "0 is voor een keypoint dat niet gelabeld is, bijvoorbeeld omdat niemand kan zien waar het is. Hier kun je zien waar de voet moet staan: welke vlag houdt die plek vast?",
            "out": "Dan schuift elk keypoint erna een plek op, en wordt de vierde poot als de derde gelezen. Elke stoel heeft dezelfde vier keypoints in dezelfde volgorde: welke vlag zegt dat er een bedekt is?",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.md(r"""
    ### Oefening 4. Waarom niet overal masks?
    """)
    return


@app.cell(hide_code=True)
def explain_cost_cell():
    explain_cost = L.explain(
        "cost",
        question="Een mask is de meest precieze annotatie: de box volgt eruit, en het zegt welke pixels het voorwerp heeft. Waarom zou iemand die een dataset voor object detection maakt boxes tekenen, en geen masks?",
        model_answer="**De taak heeft alleen de box nodig.** Een model voor object detection geeft een box en een klasse per voorwerp terug, dus een box is alles wat zijn voorbeelden hoeven te laten zien; een mask voegt pixels toe die het nooit teruggeeft. **Een mask kost veel meer tijd.** Een box van goede kwaliteit tekenen kost 7 tot 35 seconden, en een mask kostte in COCO meer dan 79. **Dezelfde uren leveren dan meer voorbeelden op.** In de tijd van één mask tekenen annotatoren twee tot elf boxes, dus de dataset krijgt meer beelden en meer voorwerpen voor hetzelfde werk.",
        points=[
            "je zegt dat een model voor object detection boxes teruggeeft, dus dat een box alles is wat de taak vraagt",
            "je zegt dat een mask veel meer tijd kost om te tekenen dan een box, zoals meer dan 79 seconden tegen 7 tot 35",
            "je zegt dat je in dezelfde tijd dan meer voorwerpen of meer beelden labelt",
        ],
    )
    explain_cost
    return (explain_cost,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De box en de connected components uit [Vormen tellen: connected components](/course/open/cv-connected-components?lang=nl) zijn twee van de vier soorten annotatie: een connected component is een mask, en zijn box volgt eruit. Welke soort een dataset bevat, hangt af van wat zijn model moet vinden: een label voor wat een beeld laat zien, een box voor waar elk voorwerp is, een mask voor welke pixels het heeft, en keypoints voor precieze plekken erop. Hoe preciezer, hoe langer het duurt: een driehoek vult maar de helft van zijn box, maar een box is getekend in een fractie van de tijd van een mask. Formaten verschillen, dus je zet om, zoals van de box van de course naar `[x, y, width, height]` van COCO, en een bedekt keypoint houdt zijn plek met v = 1. In de volgende les, *Hoeveel twee boxes overlappen: IoU*, meet je hoe goed een box die een model voorspelt past op de box van de annotatie.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hidden, choice_which, explain_cost, result_coco):
    L.wrap_up(choice_which, result_coco, choice_hidden, explain_cost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
