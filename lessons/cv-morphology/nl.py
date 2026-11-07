# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-morphology. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-morphology", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl) en [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_min():
    quick_min = L.choice(
        "quick-min",
        question="`mask = np.array([[True, True], [False, True]])`. Wat geeft `mask.min()`?",
        options={
            "false": "`False`",
            "true": "`True`",
            "mean": "`0.75`",
        },
        correct="false",
        why={
            "false": "Python telt False als 0 en True als 1, dus de kleinste van de vier waarden is False. Deze les bouwt daarop voort: de kleinste waarde van een groep pixels is alleen True als ze allemaal True zijn.",
            "true": "Drie van de vier zijn True, maar `min()` zoekt de kleinste waarde. Wat is kleiner, True of False? Zie *Het idee* van [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl).",
            "mean": "0,75 is het gemiddelde, `mask.mean()`: drie enen en een nul, gedeeld door vier. `min()` geeft de kleinste waarde.",
        },
    )
    quick_min
    return (quick_min,)


@app.cell(hide_code=True)
def quick_check_red():
    quick_red = L.choice(
        "quick-red",
        question="De regel voor rood uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) is `((h < 0.05) | (h > 0.95)) & (s > 0.5)`, met `h` de kleurtoon en `s` de verzadiging. Welke pixel pakt hij?",
        options={
            "pale": "Kleurtoon 0,97, verzadiging 0,4",
            "red": "Kleurtoon 0,97, verzadiging 0,8",
            "green": "Kleurtoon 0,3, verzadiging 0,9",
        },
        correct="red",
        why={
            "red": "0,97 is groter dan 0,95, dus de kleurtoon is rood, dicht bij 1; en 0,8 is groter dan 0,5. Beide kanten van `&` zijn True.",
            "pale": "De kleurtoon is rood, maar `&` vraagt ook een verzadiging boven 0,5, en 0,4 is dat niet: een bleek rood, zoals het roze van een tafelkleed.",
            "green": "De verzadiging is hoog, maar is 0,3 kleiner dan 0,05 of groter dan 0,95? Een kleurtoon van 0,3 is groen: zie de strook met kleurtonen in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl).",
        },
    )
    quick_red
    return (quick_red,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een drempel geeft een binair beeld, en het binaire beeld van een echte afbeelding is zelden netjes. Hier en daar komt een pixel van de achtergrond over de drempel en wordt een witte spikkel, en ergens komt een stuk van een voorwerp er niet overheen en laat een gat achter: een weerspiegeling, een schaduw, een letter die de printer te licht drukte. Voordat een programma de voorwerpen kan tellen of meten, moet het dat opruimen. Morfologische operaties doen dat door naar de witte vormen te kijken in plaats van naar één pixel tegelijk: wat te klein is om een voorwerp te zijn verdwijnt, en een gat dat te smal is om echt te zijn wordt gevuld. Ze zijn eenvoudig en snel, en je vindt ze in software die gescande tekst leest, cellen onder een microscoop telt of onderdelen aan een lopende band controleert.

    **Een voorbeeld uit het dagelijks leven.** Je knipt de appels met een schaar uit een geprinte foto. Kleine rode spikkels op het tafelkleed laat je liggen: te klein om een appel te zijn. En om de witte glans van de lamp op een appel knip je niet heen: je knipt langs de omtrek, en de glans hoort gewoon bij de appel. Met erosie en dilatatie doet een programma hetzelfde met een binair beeld.
    """)
    return


@app.cell(hide_code=True)
def predict_back():
    guess_back = L.guess(
        "back",
        question="De regel voor rood uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) pakt de appels van de tekening van fruit. Met een drempel voor de verzadiging van 0,45 in plaats van 0,5 pakt hij ook 1331 spikkels van het tafelkleed, elk van 1 tot 4 pixels, en de glans van de lamp laat in elke appel een klein gat achter. Erosie met een vierkant van 3 bij 3 maakt elke witte vorm rondom één pixel dunner; dilatatie met hetzelfde vierkant maakt elke witte vorm één pixel dikker. Wat blijft er over na een erosie en daarna een dilatatie?",
        options={
            "same": "Hetzelfde beeld: de twee heffen elkaar op",
            "apples": "De appels, ongeveer zoals ze waren, zonder de spikkels",
            "thinner": "Alles één pixel dunner dan eerst",
        },
        reveal="**De appels, ongeveer zoals ze waren, zonder de spikkels.** Geen enkele spikkel is groot genoeg voor een vierkant van 3 bij 3, dus de erosie haalt ze allemaal weg, en de dilatatie kan niet terug laten groeien wat er niet meer is. Van een appel blijft een kern over, en de dilatatie laat die weer uitgroeien tot zijn oude maat: 13.511 van de 13.612 appelpixels zijn weer wit. De andere 101 zijn de gaten van de glans, en die waren nooit wit. Er blijven maar 14 pixels van het tafelkleed staan, waar een spikkel een appel raakt.",
    )
    guess_back
    return (guess_back,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **morfologische operatie** verandert een binair beeld: ze bekijkt elke pixel door een kleine vorm, het **structuurelement**. In deze les is dat een vierkant van `k` bij `k` pixels met de pixel in het midden, zoals 3 bij 3: het venster van de pixel. Python telt False als 0 en True als 1, dus de twee basisoperaties hebben alleen de kleinste en de grootste waarde van het venster nodig:

        - **Erosie**: een pixel blijft alleen wit als elke pixel van zijn venster wit is, dus als de kleinste waarde van het venster True is. Elke witte vorm verliest zijn buitenste laag; een vorm waar het vierkant niet in past verdwijnt, en een gat wordt groter.
        - **Dilatatie**: een pixel wordt wit als er een pixel van zijn venster wit is, dus als de grootste waarde True is. Elke witte vorm krijgt er een laag bij; een gat dat smaller is dan het vierkant loopt vol, en twee vormen die dicht bij elkaar liggen groeien aan elkaar vast.

        Erosie en dilatatie heffen elkaar niet op, en juist daardoor zijn ze samen nuttig:

        - Een **opening** is een erosie en daarna een dilatatie, met hetzelfde vierkant. Waar het vierkant niet in past, is na de erosie verdwenen, en de dilatatie kan het niet terugbrengen; de rest groeit terug tot ongeveer zijn oude vorm. Een opening haalt spikkels weg.
        - Een **sluiting** is een dilatatie en daarna een erosie. Een gat of kier waar het vierkant niet in past, wordt door de dilatatie gevuld, en de erosie maakt het niet opnieuw open; de rest krimpt terug tot ongeveer zijn oude vorm. Een sluiting dicht gaten.

        Een voorbeeld op één rij, met een venster van drie pixels: de pixel en zijn buren links en rechts. 1 is wit. De rij heeft een spikkel, een stuk van drie pixels, een kier van één en een stuk van vier:

        ```
        rij        0 0 1 0 0 0 1 1 1 0 1 1 1 1 0 0
        erosie     0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0
        dilatatie  0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0
        opening    0 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0
        sluiting   0 0 1 0 0 0 1 1 1 1 1 1 1 1 0 0
        ```

        De erosie haalt de spikkel weg en haalt van elk stuk aan beide kanten een pixel af. De dilatatie laat de spikkel uitgroeien tot drie pixels, maakt elk stuk langer en vult de kier. De opening is de dilatatie van de erosie: de spikkel is weg, en beide stukken zijn terug zoals ze waren, met de kier ertussen. De sluiting is de erosie van de dilatatie: de kier is dicht, en de spikkel en de twee stukken zijn zoals ze waren.
        """),
        L.watch_out("De volgorde zit in de naam. Een opening begint met de erosie en een sluiting met de dilatatie; met dezelfde twee stappen andersom krijg je de andere operatie."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **morfologische operatie** | een verandering van een binair beeld die naar de witte vormen kijkt door een kleine vorm, het structuurelement | `erode` en `dilate` in het uitgewerkte voorbeeld; in OpenCV `cv2.erode` en `cv2.dilate` |
        | **structuurelement** | de kleine vorm waardoor een morfologische operatie kijkt, met de pixel in het midden; hier een vierkant van `k` bij `k` | het venster `img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]`, voor 3 bij 3 |
        | **erosie** | een pixel blijft alleen wit als zijn hele venster wit is: vormen krimpen, spikkels verdwijnen | `window.min()` |
        | **dilatatie** | een pixel wordt wit als er een pixel van zijn venster wit is: vormen groeien, smalle gaten lopen vol | `window.max()` |
        | **opening** | een erosie en daarna een dilatatie met hetzelfde vierkant: haalt weg waar het vierkant niet in past | `dilate(erode(mask))` |
        | **sluiting** | een dilatatie en daarna een erosie met hetzelfde vierkant: dicht gaten en kieren waar het vierkant niet in past | `erode(dilate(mask))` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        mo.md("**Rond de appel linksboven, stap voor stap**"),
        mo.Html(C.steps_picture({"drawing": "De tekening", "red": "Rood, met 0,45", "erosion": "Erosie, 3 bij 3",
                                 "opening": "Opening, 3 bij 3", "dilation": "Dilatatie van de opening",
                                 "closing": "Sluiting van de opening",
                                 "alt": "Zes plaatjes van hetzelfde stuk van de tekening van fruit, rond de appel linksboven, als zwarte en witte vierkantjes van pixels. De tekening in kleur: een rode appel met linksboven een witte glans, op een roze geruit tafelkleed. Rood, met 0,45: de appel als een witte schijf met een klein zwart gat waar de glans zit, en eromheen veel witte spikkels van één tot vier pixels, in de donkere ruiten van het tafelkleed. Erosie, 3 bij 3: geen spikkels, de schijf iets kleiner, het gat groter. Opening, 3 bij 3: de schijf even groot als eerst, met zijn kleine gat, en geen spikkels. Dilatatie van de opening: de schijf iets groter, het gat een heel klein stipje. Sluiting van de opening: de schijf even groot als eerst, het gat ongeveer even groot als eerst."})),
        mo.md(r"""
        Wit is wat de regel voor rood pakt. De erosie haalt elke spikkel weg, maakt de appel rondom één pixel kleiner, en zijn gat twee keer zo groot: 51 pixels in plaats van 25. De dilatatie daarvan is de opening: de appel zoals hij was, met zijn gat van 25 pixels, en geen spikkel. De dilatatie van de opening verkleint het gat tot 7 pixels, maar de erosie die de sluiting afmaakt, maakt het weer groter, tot 23: een vierkant van 3 bij 3 past in het gat.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Aan de rand van het beeld ligt een deel van het venster buiten het beeld. Alleen het deel binnen het beeld telt: `max(r - 1, 0)` houdt het begin van de slice op 0 of meer. Een slice die onder 0 begint, telt vanaf het eind, zoals `img[-1]` de laatste rij is, en het venster zou leeg zijn. Het eind mag voorbij de laatste rij gaan: een slice stopt daar gewoon."),
        L.watch_out("Sla de uitkomst op in een nieuwe array, `out`, en nooit in het beeld dat je leest. Als de erosie `img` zelf zou veranderen terwijl hij bezig is, kijkt de volgende pixel naar buren die al veranderd zijn."),
        mo.md(r"""
        ### Recept

        Zo ruim je een binair beeld `img` op met een vierkant van `k` bij `k`:

        1. Erosie: maak een array `out` met False, van dezelfde vorm. Neem voor elke pixel `(r, c)` zijn venster, `img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1]`, met `half = k // 2`: hoe ver het vierkant naar elke kant reikt. `//` deelt en rondt naar beneden af: `3 // 2` is 1, en `5 // 2` is 2. Sla de kleinste waarde van het venster, `window.min()`, op in `out[r, c]`.
        2. Dilatatie: hetzelfde, met de grootste waarde, `window.max()`.
        3. Haal de spikkels weg met een opening: eerst de erosie, dan de dilatatie.
        4. Dicht de gaten met een sluiting: eerst de dilatatie, dan de erosie.
        5. Neem het kleinste vierkant dat het werk doet: een groter vierkant verandert ook de vormen die je wilt houden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel pakt de rode pixels van de tekening van fruit, `data/photos/fruit.png`, met de regel voor rood en 0,45, en leest wat de course tekende, `data/photos/fruit-labels.png`, waarin 1 een appel is. Hij print hoeveel pixels rood zijn, hoeveel er bij een appel horen, en hoeveel er fout zijn: rood waar geen appel is, of niet rood waar er wel een is. De tweede cel schrijft erosie en dilatatie met een vierkant van 3 bij 3, haalt de spikkels weg met een opening en dicht de gaten met een sluiting. Hij print hoeveel pixels er na elke stap wit zijn en hoeveel er fout zijn, en laat de uitkomst zien.
    """)
    return


@app.cell
def example_mask():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    labels = np.array(Image.open("data/photos/fruit-labels.png"))
    hsv = rgb_to_hsv(img / 255)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # de regel voor rood, met 0.45 in plaats van 0.5: een binair beeld van True en False
    mask = ((h < 0.05) | (h > 0.95)) & (s > 0.45)
    # wat de course tekende: 1 is een appel
    apples = labels == 1
    print(mask.sum(), apples.sum(), (mask != apples).sum())
    return apples, mask, np, plt


@app.cell
def example_morphology(apples, mask, np, plt):
    def erode(img):
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                # het venster van 3 bij 3 rond (r, c): alleen het deel binnen het beeld
                window = img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
                # 1. erosie: alleen wit als elke pixel van het venster wit is
                out[r, c] = window.min()
        return out


    def dilate(img):
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                window = img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
                # 2. dilatatie: wit als er een pixel van het venster wit is
                out[r, c] = window.max()
        return out


    # 3. de opening haalt de spikkels weg: erosie, dan dilatatie
    eroded = erode(mask)
    opened = dilate(eroded)
    # 4. de sluiting dicht de gaten: dilatatie, dan erosie
    closed = erode(dilate(opened))
    print(mask.sum(), eroded.sum(), opened.sum(), closed.sum())
    print((mask != apples).sum(), (opened != apples).sum(), (closed != apples).sum())
    # True is wit, False zwart
    plt.imshow(closed, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De regel pakt 16.190 rode pixels, en de course tekende 13.612 appelpixels: 2780 pixels zijn fout. Daarvan zijn er 2679 van het tafelkleed, 2634 in de 1331 spikkels en 45 aan de rand van een appel, en 101 zijn de gaten van de glans. Na de erosie zijn 12.403 pixels wit: elke spikkel is weg, en de appels zijn rondom één pixel kleiner. Na de opening 13.525: de appels zijn terug, en er zijn nog maar 115 pixels fout, de 101 van de gaten en 14 van het tafelkleed aan de rand van een appel. De sluiting van 3 bij 3 vult maar 12 van de 101 pixels van de gaten, dus er zijn 103 pixels fout, en in het plaatje onder de cel heeft elke appel nog zijn gat. Een gat van de glans is 5 of 6 pixels breed, en een vierkant van 3 bij 3 past erin. In oefening 3 zoek je een groter vierkant.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat houdt erosie over?

    Een klein binair beeld, met in het midden een wit vierkant van 3 bij 3; 1 is wit:

    ```
    0 0 0 0 0
    0 1 1 1 0
    0 1 1 1 0
    0 1 1 1 0
    0 0 0 0 0
    ```
    """)
    return


@app.cell(hide_code=True)
def choice_block_cell():
    choice_block = L.choice(
        "block",
        question="Welke pixels zijn wit na een erosie met een vierkant van 3 bij 3?",
        options={
            "middle": "Alleen de pixel in het midden",
            "nine": "Alle negen pixels van het vierkant",
            "ring": "De acht pixels rond het midden",
            "none": "Geen enkele",
        },
        correct="middle",
        why={
            "middle": "Alleen de pixel in het midden heeft een venster dat overal wit is: zijn venster is het hele witte vierkant. Elke andere witte pixel heeft een zwarte pixel van de rand in zijn venster. Een witte vorm verliest zijn buitenste laag, en van een vierkant van 3 bij 3 blijft alleen het midden over.",
            "nine": "Dat geeft een opening: de erosie, en daarna de dilatatie die het vierkant terug laat groeien. Wat blijft er over na alleen de erosie? Kijk naar het venster van de pixel linksboven in het witte vierkant: is dat overal wit?",
            "ring": "Dat zijn de pixels die de erosie weghaalt: elk van hen heeft een zwarte pixel in zijn venster. Welke witte pixel heeft een venster zonder zwart?",
            "none": "Dan zou het vierkant van 3 bij 3 nergens in het wit passen. Past het ergens? Kijk naar het venster van de pixel in het midden.",
        },
    )
    choice_block
    return (choice_block,)


@app.cell(hide_code=True)
def text_exercise_dilate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dilatatie met elk vierkant

        Het uitgewerkte voorbeeld doet een dilatatie met een vierkant van 3 bij 3. Maak `dilate(img, k)` af, die een dilatatie doet met een vierkant van `k` bij `k`: `k` is 1, 3, 5 of een ander oneven getal. Hij geeft een binair beeld van dezelfde vorm terug, waarin een pixel wit is als er een pixel in het vierkant eromheen wit is; alleen het deel van het vierkant binnen het beeld telt. `img` is een binair beeld, een lijst van rijen met 0 en 1. De loops en `half = k // 2`, hoe ver het vierkant naar elke kant reikt, staan er al: vervang de `...`. De check roept hem aan met één witte pixel in het midden:

        ```python
        dot = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
        dilate(dot, 3)
        ```

        Dat moet `[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]` geven. Hij probeert ook andere maten, zoals 5 en 1, en witte pixels aan de rand.
        """),
        L.task("Laat `dilate(img, k)` elke pixel wit maken die een witte pixel in zijn vierkant van `k` bij `k` heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dilate():
    box_dilate = L.code_box("dilate", lines=14)
    box_dilate
    return (box_dilate,)


@app.cell(hide_code=True)
def check_dilate(box_dilate):
    result_dilate = L.check("dilate", box_dilate.value)
    result_dilate
    return (result_dilate,)


@app.cell(hide_code=True)
def hints_dilate():
    L.hints(
        "dilate",
        [
            "`half` is hoe ver het vierkant naar elke kant van zijn midden reikt: 1 voor 3 bij 3, 2 voor 5 bij 5, 0 voor 1 bij 1.",
            "De rijen van het venster lopen van `r - half` tot `r + half`. Een slice stopt net voor zijn eind, dus schrijf `r + half + 1`; en zodat het begin nooit onder 0 komt, schrijf `max(r - half, 0)`. De kolommen gaan net zo, met `c`.",
            "`window = img[max(r - half, 0):r + half + 1, max(c - half, 0):___]`, en dan `out[r, c] = window.max()`.",
        ],
        why="Bij `dilate(dot, 3)` is `half` 1. De pixel op `(1, 1)` heeft het venster van de rijen 0 tot en met 2 en de kolommen 0 tot en met 2, en daar zit de witte pixel op `(2, 2)` in, dus hij wordt wit. De pixel op `(0, 0)` heeft het venster van de rijen 0 tot en met 1 en de kolommen 0 tot en met 1, zonder de witte pixel, dus hij blijft zwart. Elke pixel die hooguit één stap van `(2, 2)` af ligt, ook schuin, wordt wit: een vierkant van 3 bij 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ### Oefening 3. Het vierkant van de sluiting

    In het uitgewerkte voorbeeld liet de sluiting van 3 bij 3 de gaten van de glans bijna zoals ze waren. Met de slider kies je het vierkant van de sluiting; de opening ervoor blijft 3 bij 3. Links de uitkomst; rechts hoeveel pixels er bij elke maat fout zijn: wit waar de course geen appel tekende, of zwart waar hij er wel een tekende. De zin onder het plaatje zegt hoeveel pixels van de gaten nog zwart zijn, en hoeveel witte vormen er zijn.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(k):
        _html, _f = C.size_frame(k, {"closed": "sluiting van {k} bij {k}", "size": "maat van het vierkant", "wrong": "pixels fout",
                                     "alt": "Links de vier appels van de tekening van fruit als witte vormen op zwart, na een opening van 3 bij 3 en een sluiting van {k} bij {k}; rechts hoeveel pixels er fout zijn voor de vierkanten van 1 tot 15: 115 bij 1, 103 bij 3, het minst, 14, bij 5 en bij 7, daarna weer meer, 74 bij 9 en tot 411 bij 15. Maat {k} is gemarkeerd met een grotere punt."})
        _shapes = "4 witte vormen, één per appel" if _f["shapes"] == 4 else f'{_f["shapes"]} witte vormen: twee appels zijn aan elkaar gegroeid'
        return (_html,
                f'Vierkant van {k} bij {k}: {_f["holes"]} pixels van de gaten zijn nog zwart, {_shapes}, {_f["wrong"]} pixels fout.')

    explore_size = L.explore("size", "vierkant van de sluiting", _frame, start=1, values=C.SIZES, role="hyper",
                             task="Zoek het kleinste vierkant dat elk gat dicht.")
    explore_size
    return (explore_size,)


@app.cell(hide_code=True)
def check_size(explore_size):
    result_size = L.check("size", explore_size)
    result_size
    return (result_size,)


@app.cell(hide_code=True)
def hints_size():
    L.hints(
        "size",
        [
            "Schuif de slider stop voor stop van 1 naar rechts, en lees het eerste getal in de zin onder het plaatje.",
            "Een sluiting dicht een gat als het vierkant er niet in past. De gaten van de glans zijn rond, op hun breedst 5 of 6 pixels: een vierkant van 3 bij 3 past erin, maar de hoeken van een vierkant van 5 bij 5 steken uit.",
            "Kijk naar de stops 3 en 5: na welke is geen pixel van de gaten meer zwart?",
        ],
        why="Bij 3 zijn er nog 89 pixels van de gaten zwart: een vierkant van 3 bij 3 past in elk gat. Bij 5 is geen enkele pixel meer zwart: een vierkant van 5 bij 5 past in geen enkel gat, dus de dilatatie vult ze en de erosie maakt ze niet opnieuw open. 7 dicht ze ook, met even weinig pixels fout, maar vanaf 9 vult de sluiting meer dan de gaten: eerst de strook tussen de kleine appel en de rechterrand van de tekening, omdat de erosie niet voorbij de rand kijkt, dan het gat tussen de twee appels rechts.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Oefening 4. Eerst de opening, dan de sluiting
    """)
    return


@app.cell(hide_code=True)
def explain_order_cell():
    explain_order = L.explain(
        "order",
        question="Na een opening van 3 bij 3 en een sluiting van 5 bij 5 zijn er nog 14 pixels van de tekening fout. Andersom, eerst de sluiting van 5 bij 5 en dan de opening van 3 bij 3, zijn het er 529: ruwe bulten van tafelkleed aan de randen van de appels. Waarom maakt de volgorde zoveel uit?",
        model_answer="**Eerst de sluiting:** een sluiting haalt geen spikkels weg: de dilatatie laat een spikkel groeien en de erosie laat hem terug krimpen. En een spikkel dicht bij een appel groeit tijdens de dilatatie aan de appel vast; de erosie daarna maakt hem niet meer los, dus de spikkel is een bult op de appel geworden. **Dan de opening:** een opening haalt alleen weg waar het vierkant niet in past. Een bult die bij een grote appel hoort, is vaak dik genoeg voor het vierkant, dus hij blijft. **Eerst de opening:** de spikkels zijn dan nog klein en los, dus de erosie haalt ze allemaal weg voordat ze ergens aan vast kunnen groeien; daarna heeft de sluiting alleen nog de appels om te vullen.",
        points=[
            "je zegt dat een sluiting geen spikkels weghaalt, maar spikkels dicht bij een appel eraan vast laat groeien",
            "je zegt dat de opening daarna die bulten houdt, omdat ze bij een grote vorm horen waar het vierkant in past",
            "je zegt dat een opening eerst de spikkels weghaalt zolang ze nog klein en los zijn",
        ],
    )
    explain_order
    return (explain_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Morfologische operaties ruimen de binaire beelden op van [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl) en van de regel voor rood uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl): de drempel beslist pixel voor pixel, de opening en de sluiting kijken naar de vormen. Erosie neemt de kleinste waarde van elk venster en dilatatie de grootste, met een vierkant van elke grootte. Hetzelfde minimum en maximum werken ook op een grijswaardenbeeld, waar ze lichte vlekken laten krimpen of groeien. Na de opening en een sluiting van 5 bij 5 heeft de tekening vier witte vormen zonder gaten, één per appel. De volgende les, *Vormen tellen: connected components*, telt ze: hij volgt witte pixels die elkaar raken, van buur naar buur, en vindt elke vorm, zijn grootte en het kleinste kader eromheen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_block, explain_order, result_dilate, result_size):
    L.wrap_up(choice_block, result_dilate, result_size, explain_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
