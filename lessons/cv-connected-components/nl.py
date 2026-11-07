# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-connected-components. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-connected-components", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) en [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_queue():
    quick_queue = L.choice(
        "quick-queue",
        question="Wat laat dit programma zien?\n\n```python\nfrom collections import deque\nqueue = deque([(0, 0), (0, 1)])\nqueue.append((1, 0))\nprint(queue.popleft())\n```",
        options={
            "first": "`(0, 0)`",
            "second": "`(0, 1)`",
            "last": "`(1, 0)`",
        },
        correct="first",
        why={
            "first": "`append` zet `(1, 0)` achteraan de wachtrij, en `popleft()` pakt van voren: de oudste, `(0, 0)`. Deze les gebruikt de wachtrij net zo, met pixels in plaats van paden.",
            "second": "`(0, 1)` staat als tweede in de rij, en `popleft()` pakt de allervoorste. Zie `deque` in [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl).",
            "last": "`(1, 0)` kwam er als laatste bij, achteraan, en `popleft()` pakt van voren. Zie `deque` in [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl).",
        },
    )
    quick_queue
    return (quick_queue,)


@app.cell(hide_code=True)
def quick_check_opening():
    quick_opening = L.choice(
        "quick-opening",
        question="Een binair beeld heeft een witte spikkel van 2 bij 2 pixels, ver van elke andere witte pixel. Wat blijft ervan over na een opening met een vierkant van 3 bij 3?",
        options={
            "nothing": "Niets",
            "one": "Eén pixel",
            "same": "De spikkel, zoals hij was",
            "bigger": "Een vierkant van 4 bij 4",
        },
        correct="nothing",
        why={
            "nothing": "Een vierkant van 3 bij 3 past nergens in de spikkel, dus de erosie haalt hem weg, en de dilatatie kan niet terug laten groeien wat weg is.",
            "one": "Is na de erosie nog een pixel van de spikkel overal wit in zijn venster? Een venster van 3 bij 3 past niet in 2 bij 2. Zie *Het idee* van [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl).",
            "same": "Dat doet een opening met een vorm waar het vierkant in past. Past een vierkant van 3 bij 3 in een spikkel van 2 bij 2?",
            "bigger": "Dat is de dilatatie alleen: die laat de spikkel aan elke kant één pixel groeien. Een opening begint met de erosie.",
        },
    )
    quick_opening
    return (quick_opening,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een binair beeld zegt per pixel of die wit of zwart is, en verder niets. Om te zeggen hoeveel voorwerpen er zijn, hoe groot elk is en waar het ligt, moet een programma weten welke witte pixels bij elkaar horen. Connected components geven precies dat: elke groep witte pixels die elkaar raken, is één vorm. Zo telt een programma cellen onder een microscoop, deelt het een gescande bladzijde op in letters voordat het ze leest, en meet het de onderdelen aan een lopende band. En de box om elke vorm is de eerste stap naar de boxes die een detectiemodel om voorwerpen tekent.

    **Een voorbeeld uit het dagelijks leven.** De verfemmer van een tekenprogramma. Je klikt in een vlak, en de kleur loopt van pixel naar pixel, naar elke pixel die een gekleurde raakt, tot hij overal tegen een rand aan komt. Het vlak dat hij vult, is één connected component. Klik met de verfemmer op elke witte pixel die nog geen kleur heeft, steeds met een nieuwe kleur, en elke vorm heeft zijn eigen kleur: je hebt ze geteld.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="De zoektocht van deze les werkt zoals die verfemmer, maar gaat ook via hoeken verder. Hij begint bij één pixel midden in een groot wit vlak, en kleurt bij elke stap elke witte pixel die een gekleurde raakt, ook schuin, met een hoek. Hoe zien de gekleurde pixels eruit na 10 stappen?",
        options={
            "disk": "Een schijf, 21 pixels breed",
            "square": "Een vierkant van 21 bij 21",
            "diamond": "Een ruit, 21 pixels van punt tot punt",
        },
        reveal="**Een vierkant van 21 bij 21.** Na de eerste stap zijn de start en zijn acht buren gekleurd: een vierkant van 3 bij 3. Elke volgende stap legt er een laag van één pixel omheen, zoals een dilatatie met een vierkant van 3 bij 3 doet: 5 bij 5, dan 7 bij 7, en na 10 stappen 21 bij 21. Als alleen de vier buren erboven, eronder, links en rechts meetelden, werd het een ruit. Aan de rand van een vorm stopt de kleur: de slider hieronder laat dat zien in een appel.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Twee witte pixels **raken elkaar** als de ene bij de acht pixels rond de andere hoort: erboven, eronder, links, rechts, of schuin, met een hoek. Een **connected component** is een groep witte pixels waarin je van elke pixel naar elke andere kunt lopen, in stappen van een witte pixel naar een witte pixel die hem raakt. Geen witte pixel buiten de groep raakt een pixel erin: de groep is één vorm. Zijn grootte is zijn aantal pixels. Zijn **bounding box** is de kleinste rechtopstaande rechthoek eromheen: van zijn kleinste tot zijn grootste rij, en van zijn kleinste tot zijn grootste kolom. Deze les schrijft een box als vier getallen, `(boven, links, onder, rechts)`.

        Een breadth-first search vindt een vorm, zoals hij een pad door een doolhof vond. De states zijn nu pixels, en een action is een stap naar een witte pixel die de huidige raakt:

        - Begin bij een witte pixel: zet hem in een wachtrij, en in de set `seen` met de pixels die de zoektocht gevonden heeft.
        - Pak de oudste pixel vooraan uit de wachtrij. Elke witte buur die nog niet in `seen` zit, gaat in `seen` en achteraan de wachtrij.
        - Als de wachtrij leeg is, heeft de zoektocht elke pixel van de vorm gevonden.

        Er is geen doel om te bereiken: de zoektocht gaat door tot de wachtrij leeg is. Om elke vorm te vinden, loop je het beeld rij voor rij af. Bij een witte pixel die nog niet in `seen` zit, begint een nieuwe vorm: doorzoek hem, en ga verder met de volgende pixel. `seen` is één set voor het hele beeld, dus geen zoektocht begint bij een pixel van een vorm die al eerder gevonden is.

        Een voorbeeld: een beeld van 5 rijen bij 7 kolommen, waarin 1 wit is. Ernaast het nummer van de vorm van elke witte pixel, in de volgorde waarin de scan ze vindt:

        ```
        beeld            vormen
        1 1 0 0 0 1 1    1 1 . . . 2 2
        0 1 0 0 1 1 0    . 1 . . 2 2 .
        0 0 0 0 0 0 0    . . . . . . .
        0 1 1 0 0 0 1    . 3 3 . . . 4
        1 0 0 0 0 0 1    3 . . . . . 4
        ```

        De pixel op rij 4, kolom 0 raakt de pixel op rij 3, kolom 1 alleen met een hoek, maar dat telt: hij hoort bij vorm 3. Vorm 3 heeft 3 pixels, en zijn box is `(3, 0, 4, 2)`: rij 3 tot 4, kolom 0 tot 2. Vorm 2 heeft 4 pixels, en zijn box is `(0, 4, 1, 6)`. Als alleen de vier buren erboven, eronder, links en rechts meetelden, was de pixel op rij 4, kolom 0 een vorm op zich, en had het beeld 5 vormen.
        """),
        L.watch_out("Een box is vier getallen, en programma's schrijven ze in verschillende volgordes. Deze les schrijft de rijen eerst, zoals NumPy: `(boven, links, onder, rechts)`. Veel programma's voor object detection schrijven `(x1, y1, x2, y2)`, met x de kolom: eerst links en boven. Twee volgordes door elkaar halen is een klassieke bug."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **connected component** | een groep witte pixels die elkaar raken, van buur tot buur, en die geen andere witte pixel raakt: één vorm | `find_shapes(img)` in het uitgewerkte voorbeeld; in OpenCV `cv2.connectedComponents` |
        | **bounding box** | de kleinste rechtopstaande rechthoek om een vorm: van zijn kleinste tot zijn grootste rij en kolom | `(boven, links, onder, rechts)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    De zoektocht vanuit het midden van de appel linksboven, stap voor stap, na de opening en de sluiting van de vorige les. Eén stap is één laag: de zoektocht haalt elke pixel eruit die hij in de stap ervoor vond, en zet hun nieuwe witte buren achteraan de wachtrij. Lichtgrijs is wat de zoektocht al uit de wachtrij heeft gehaald, wit wat er nog in zit, en donkergrijs de pixels van de appel die hij nog niet gevonden heeft.
    """)
    return


@app.cell(hide_code=True)
def explore_search_cell():
    def _say(k):
        _o = C.search_outcome(k)
        if k == 0:
            return "Stap 0: in de wachtrij zit alleen de start."
        if _o["found"] == _o["apple"]:
            return (f"Stap {k}: alle {_o['apple']} pixels van de appel zijn gevonden. De laatste {_o['queue']} zitten nog in de wachtrij; "
                    "ze hebben geen nieuwe buren, dus daarna is de wachtrij leeg.")
        return f"Stap {k}: {_o['found']} pixels gevonden, waarvan er nog {_o['queue']} in de wachtrij zitten."

    def _frame(k):
        k = int(k)
        _html = C.search_frame(k, {"title": "stap {k}",
                                   "alt": "De appel linksboven in de tekening van fruit als een schijf, na {k} stappen van de zoektocht vanuit zijn midden: {found} van zijn {apple} pixels gevonden, in lichtgrijs, met de {queue} pixels die nog in de wachtrij zitten in wit aan de buitenkant; de rest van de appel donkergrijs, op zwart."})
        return _html, _say(k)

    explore_search = L.explore("search", "stap", _frame, start=0, values=C.STEPS, role="",
                               task="Schuif naar stap 10: welke vorm hebben de gevonden pixels? Schuif daarna naar de laatste stap.")
    explore_search
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Na 10 stappen zijn de gevonden pixels het vierkant van 21 bij 21 uit de gok, en na 20 stappen een vierkant van 41 bij 41. Tot stap 24 blijven ze een vierkant; daarna snijdt de ronde rand van de appel ze af. Na 33 stappen heeft de zoektocht alle 3617 pixels van de appel gevonden.
    """)
    return


@app.cell(hide_code=True)
def picture_boxes():
    mo.vstack([
        mo.md("**Vier vormen en hun boxes**"),
        mo.Html(C.boxes_picture({"alt": "De tekening van fruit in kleur, met een blauwe rechthoek om elk van de vier appels, zo krap als past, en de grootte van de appel in pixels boven zijn linkerbovenhoek: 3617, 4051, 2111 en 3847. De sinaasappel, de citroen en de limoen hebben geen rechthoek."})),
        mo.md(r"""
        In het blauw wat het programma na de opening en de sluiting teruggeeft: de box van elk van de vier vormen, met zijn grootte in pixels.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Zet een pixel in `seen` zodra hij de wachtrij in gaat, en niet pas als hij eruit komt. Dan komt geen pixel twee keer in de wachtrij. De zoektocht in het doolhof zette een state pas in zijn set `done` als die uit de wachtrij kwam; hier heeft een pixel tot acht witte buren, en elk van hen zou hem opnieuw in de wachtrij zetten."),
        L.watch_out("Aan de rand van het beeld ligt een deel van het venster buiten het beeld, net als in de vorige les. `max(r - 1, 0)` houdt de eerste rij op 0 of meer, en `min(r + 2, len(img))` houdt het eind van de `range` op het aantal rijen: een `range` voorbij de laatste rij vraagt om een rij die er niet is."),
        mo.md(r"""
        ### Recept

        Zo vind je elke vorm van een binair beeld `img`, met zijn grootte en zijn box:

        1. Maak een lege set `seen`. Loop de pixels rij voor rij af; bij een witte pixel die niet in `seen` zit, begint een nieuwe vorm.
        2. Zet die pixel in `seen` en in een wachtrij, `deque([start])`, en maak een lege lijst `shape`.
        3. Zolang de wachtrij niet leeg is: pak de oudste pixel vooraan met `popleft()`, en voeg hem toe aan `shape`. Elke witte pixel in het venster van 3 bij 3 eromheen, alleen het deel binnen het beeld, die nog niet in `seen` zit, gaat in `seen` en achteraan de wachtrij.
        4. Als de wachtrij leeg is, is de vorm compleet. Zijn grootte is `len(shape)`; zijn box loopt van de kleinste tot de grootste rij en van de kleinste tot de grootste kolom van zijn pixels.
        5. Ga verder met stap 1 bij de volgende pixel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel pakt de rode pixels van de tekening van fruit met de regel voor rood en 0,45, en ruimt ze op zoals aan het eind van de vorige les: een opening van 3 bij 3 en een sluiting van 5 bij 5, met `erode` en `dilate` voor een vierkant van `k` bij `k`. Hij print hoeveel pixels er voor en na wit zijn. De tweede cel volgt het recept: `white_neighbors` geeft de witte pixels in het venster van 3 bij 3 rond een pixel, `shape_from` doorzoekt één vorm, en `find_shapes` loopt het beeld af. Hij print uit hoeveel vormen de rode pixels bestaan voor het opruimen, met de zes grootste groottes, en daarna voor elke vorm na het opruimen zijn grootte en de pixel waar zijn zoektocht begon.
    """)
    return


@app.cell
def example_mask():
    import numpy as np
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    hsv = rgb_to_hsv(img / 255)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # de regel voor rood, met 0.45: appels, spikkels van het tafelkleed, en gaten
    mask = ((h < 0.05) | (h > 0.95)) & (s > 0.45)


    def erode(img, k):
        # de kleinste waarde van elk venster van k bij k, zoals in de vorige les
        half = k // 2
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                top, left = max(r - half, 0), max(c - half, 0)
                out[r, c] = img[top:r + half + 1, left:c + half + 1].min()
        return out


    def dilate(img, k):
        # de grootste waarde van elk venster van k bij k
        half = k // 2
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                top, left = max(r - half, 0), max(c - half, 0)
                out[r, c] = img[top:r + half + 1, left:c + half + 1].max()
        return out


    # een opening van 3 bij 3 haalt de spikkels weg, een sluiting van 5 bij 5 de gaten
    clean = erode(dilate(dilate(erode(mask, 3), 3), 5), 5)
    print(mask.sum(), clean.sum())
    return clean, mask


@app.cell
def example_shapes(clean, mask):
    from collections import deque


    def white_neighbors(img, r, c):
        # de witte pixels in het venster van 3 bij 3 rond (r, c), binnen het beeld
        result = []
        for nr in range(max(r - 1, 0), min(r + 2, len(img))):
            for nc in range(max(c - 1, 0), min(c + 2, len(img[0]))):
                if img[nr, nc]:
                    result.append((nr, nc))
        return result


    def shape_from(img, start, seen):
        # 2. de start gaat in seen en in de wachtrij
        seen.add(start)
        queue = deque([start])
        shape = []
        while len(queue) > 0:
            # 3. de oudste pixel vooraan eruit; zijn nieuwe witte buren achteraan
            r, c = queue.popleft()
            shape.append((r, c))
            for nxt in white_neighbors(img, r, c):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        # 4. de wachtrij is leeg: de vorm is compleet
        return shape


    def find_shapes(img):
        # 1. rij voor rij: een witte pixel die niet in seen zit, begint een nieuwe vorm
        seen = set()
        shapes = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                if img[r, c]:
                    if (r, c) not in seen:
                        shapes.append(shape_from(img, (r, c), seen))
        return shapes


    sizes = []
    for shape in find_shapes(mask):
        sizes.append(len(shape))
    print(len(sizes), sorted(sizes)[-6:])
    for shape in find_shapes(clean):
        print(len(shape), shape[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De regel pakt 16.190 rode pixels; na de opening en de sluiting zijn er 13.626 wit. Voor het opruimen bestaan de rode pixels uit 1335 vormen. De vier grootste, van 2094 tot 4029 pixels, zijn de appels, elk met zijn gat en de spikkels die hem raken. De grootste vorm daarna heeft 4 pixels: de andere 1331 vormen zijn de spikkels, van 1 tot 4 pixels elk. Na het opruimen zijn er 4 vormen over, één per appel, van 2111 tot 4051 pixels. De scan vindt ze in de volgorde van hun hoogste pixel: eerst de appel linksboven, waarvan de zoektocht begint op rij 37, kolom 62, de bovenkant van de appel. In oefening 3 haal je de spikkels op een andere manier weg: op hun grootte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel vormen?

    Een klein binair beeld; 1 is wit:

    ```
    0 1 0 0 1
    1 0 0 1 0
    0 0 1 0 0
    1 0 0 0 1
    1 1 0 1 1
    ```
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "count",
        question="Hoeveel vormen vindt de zoektocht van deze les, met alle acht buren?",
        options={
            "two": "2",
            "four": "4",
            "seven": "7",
            "eleven": "11",
        },
        correct="four",
        why={
            "four": "Pixels die elkaar met een hoek raken, horen bij elkaar. De twee pixels linksboven vormen één vorm; de schuine lijn van rechtsboven naar het midden, `(0, 4)`, `(1, 3)` en `(2, 2)`, een tweede; en elke hoek onderaan is een vorm van drie pixels. De pixel in het midden raakt geen van de hoeken onderaan: de pixels van rij 3 naast hem zijn zwart.",
            "two": "Dan zouden de vormen bovenaan vastzitten aan die onderaan. Raakt een witte pixel van rij 2 een witte pixel van rij 3? Kijk naar het venster van 3 bij 3 rond de pixel in het midden.",
            "seven": "Zoveel tel je met alleen de vier buren erboven, eronder, links en rechts: dan is elke pixel van een schuine lijn een vorm op zich. In deze les raken pixels die alleen met een hoek tegen elkaar zitten elkaar ook.",
            "eleven": "Dat is het aantal witte pixels. Een vorm is een groep witte pixels die elkaar raken: welke pixels raken elkaar?",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_exercise_box():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De box van één vorm

        Maak `box_of(img, start)` af, die de bounding box geeft van de vorm met de pixel `start`, als `(boven, links, onder, rechts)`. `img` is een binair beeld, een lijst van rijen met 0 en 1, en `start` een witte pixel, als (rij, kolom). `white_neighbors` uit het uitgewerkte voorbeeld staat er al, en het begin van de zoektocht ook. Vervang de eerste `...` door één ronde van de loop, stap 3 van het recept, en de tweede door de box: `rows` bevat de rij van elke pixel van de vorm, en `cols` zijn kolom. De check roept hem aan met dit beeld en de pixel op rij 1, kolom 1:

        ```python
        shapes = [[0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]]
        box_of(shapes, (1, 1))
        ```

        Dat moet `(1, 1, 3, 2)` geven: de pixel op rij 3, kolom 2 raakt de vorm met een hoek. De check probeert ook andere startpixels, en vormen aan de rand van het beeld.
        """),
        L.task("Laat `box_of(img, start)` de vorm rond `start` doorzoeken en zijn box teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_box():
    box_code = L.code_box("box", lines=20)
    box_code
    return (box_code,)


@app.cell(hide_code=True)
def check_box(box_code):
    result_box = L.check("box", box_code.value)
    result_box
    return (result_box,)


@app.cell(hide_code=True)
def hints_box():
    L.hints(
        "box",
        [
            "Stap 3 van het recept zegt wat één ronde van de loop doet, en `shape_from` in het uitgewerkte voorbeeld doet het: haal de oudste pixel eruit, voeg hem toe aan `shape`, en zet zijn nieuwe witte buren erin. De box is de kleinste en de grootste rij en kolom van `shape`.",
            "Pak de pixel vooraan in de wachtrij met `popleft()`, en voeg hem toe aan `shape`. Voor elke buur uit `white_neighbors(img, r, c)` die nog niet in `seen` zit: voeg hem toe aan `seen`, en zet hem achteraan de wachtrij. De box is de kleinste rij, de kleinste kolom, de grootste rij en de grootste kolom.",
            "`r, c = queue.___()` en `shape.append((r, c))`; dan `for nxt in white_neighbors(img, r, c):` met `if nxt not in seen:`, `seen.add(nxt)` en `queue.append(nxt)`. Aan het eind: `return rows.min(), cols.min(), ___, ___`.",
        ],
        why="Bij `box_of(shapes, (1, 1))` begint de wachtrij met `(1, 1)`. De eerste ronde haalt die eruit en zet zijn witte buren `(1, 2)` en `(2, 1)` in de wachtrij. De volgende rondes halen die eruit; `(2, 1)` zet er `(3, 2)` in, dat hem met een hoek raakt. Dan is de wachtrij leeg: `shape` bevat vier pixels, in de rijen 1 tot 3 en de kolommen 1 tot 2. De vorm rechts bereikt de zoektocht nooit: geen van zijn pixels raakt een pixel van de eerste vorm.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smallest():
    mo.md(r"""
    ### Oefening 3. Spikkels weghalen op hun grootte

    De opening haalde de spikkels weg omdat er geen vierkant in paste. Nu je elke vorm kunt meten, kan het ook anders: houd alleen de vormen van minstens een bepaalde grootte. Met de slider kies je die grootte, voor de rode pixels voordat er iets is opgeruimd. Links de vormen die overblijven, elk in zijn blauwe box als het er 10 of minder zijn; rechts hoeveel pixels er bij elke stop van de slider fout zijn: wit waar de course geen appel tekende, of zwart waar wel. De stops gaan van 1 tot 5, en springen dan naar 10, 100, 1000, 2000 en 3000.
    """)
    return


@app.cell(hide_code=True)
def explore_smallest_cell():
    def _frame(n):
        _html, _f = C.smallest_frame(n, {"kept": "vormen van minstens {n} pixels", "kept_1": "vormen van minstens 1 pixel", "size": "kleinste grootte die blijft", "wrong": "pixels fout",
                                         "alt": "Links de rode pixels van de tekening van fruit, waarvan alleen de vormen van minstens {n} pixels blijven, als wit op zwart; rechts hoeveel pixels er fout zijn bij de kleinste groottes 1 tot 3000, één punt per stop van de slider: 2780 bij 1, 470 bij 4, 146 van 5 tot 2000, en 2232 bij 3000. Grootte {n} is gemarkeerd met een grotere stip."})
        return (_html,
                f'Vormen van minstens {n} pixel{"" if n == 1 else "s"}: {_f["shapes"]} vormen over, {_f["wrong"]} pixels fout.')

    explore_smallest = L.explore("smallest", "kleinste grootte die blijft", _frame, start=1, values=C.SMALLEST, role="hyper",
                                 task="Zoek de kleinste grootte die alleen de vier appels houdt.")
    explore_smallest
    return (explore_smallest,)


@app.cell(hide_code=True)
def check_smallest(explore_smallest):
    result_smallest = L.check("smallest", explore_smallest)
    result_smallest
    return (result_smallest,)


@app.cell(hide_code=True)
def hints_smallest():
    L.hints(
        "smallest",
        [
            "Schuif de slider één stop tegelijk, vanaf 1 naar rechts, en lees hoeveel vormen er over zijn.",
            "In het uitgewerkte voorbeeld hadden de vier grootste vormen 2094 tot 4029 pixels, en de grootste daarna 4.",
            "Kijk naar de stops 4 en 5: na welke zijn er maar vier vormen over?",
        ],
        why="Bij 4 zijn er 85 vormen over: de vier appels en 81 spikkels van 4 pixels. Bij 5 zijn alleen de vier appels over: elke spikkel heeft 1 tot 4 pixels. Elke grootte tot 2000 geeft dezelfde vier vormen en 146 pixels fout; bij 3000 is ook de kleine appel weg, van 2094 pixels.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.md(r"""
    ### Oefening 4. Het filter en de opening
    """)
    return


@app.cell(hide_code=True)
def explain_filter_cell():
    explain_filter = L.explain(
        "filter",
        question="Met het filter op grootte van 5 zijn er 146 pixels fout: de 101 pixels van de gaten van de glans, en 45 pixels van het tafelkleed. De opening van 3 bij 3 uit de vorige les liet dezelfde gaten staan, maar maar 14 pixels van het tafelkleed. Waarom houdt het filter meer van het tafelkleed over?",
        model_answer="**Een spikkel die een appel raakt, hoort bij de vorm van de appel.** De zoektocht gaat van witte pixel naar witte pixel die raakt, ook met een hoek, en ziet niet waar de appel ophoudt en de spikkel begint: samen zijn ze één connected component. **Het filter beoordeelt hele vormen.** Het houdt of verwijdert een vorm in één keer, op zijn grootte; de appel is groot, dus de spikkel die hem raakt blijft ook staan. **De opening beoordeelt elke pixel met zijn eigen vierkant.** Een spikkel aan de rand van een appel is te dun voor het vierkant van 3 bij 3, dus de erosie haalt hem weg, of hij de appel nu raakt of niet; een pixel van het tafelkleed blijft alleen staan waar hij samen met pixels van de appel in een wit vierkant van 3 bij 3 ligt, en dat geldt voor 14 ervan.",
        points=[
            "je zegt dat een spikkel die een appel raakt, ook met een hoek, bij de vorm van de appel hoort",
            "je zegt dat het filter hele vormen houdt of verwijdert, dus dat de spikkel blijft staan met de grote appel",
            "je zegt dat de opening elke pixel door zijn vierkant bekijkt, dus dat hij een dunne spikkel weghaalt, ook als die een appel raakt",
        ],
    )
    explain_filter
    return (explain_filter,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De breadth-first search uit [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) vond een kortste pad naar een doel; hier vindt hij elke pixel van een vorm, en een scan over het beeld start hem opnieuw bij elke witte pixel die nog geen zoektocht gevonden heeft. Stap voor stap groeit de zoektocht zoals een dilatatie met een vierkant van 3 bij 3 uit [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl), maar nooit voorbij de rand van de vorm. Connected components maken van een binair beeld dingen die je kunt tellen en meten: na de opening en de sluiting heeft de tekening van fruit vier vormen, elk met zijn grootte en zijn box. Een filter op grootte haalt ook spikkels weg, maar houdt alles wat een grote vorm raakt. Verderop in deze tak is een box wat een detectiemodel voorspelt voor elk voorwerp dat het ziet, en meet je hoe goed twee boxes overlappen. De volgende les, *Groter en kleiner: interpolatie*, maakt een beeld groter of kleiner, en moet daarbij de pixels ertussen verzinnen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, explain_filter, result_box, result_smallest):
    L.wrap_up(choice_count, result_box, result_smallest, explain_filter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
