# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-resize-interpolation. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-resize-interpolation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bytes():
    quick_bytes = L.choice(
        "quick-bytes",
        question="De koffiefoto heeft shape `(400, 600, 3)` en dtype uint8, dus nemen zijn getallen 720.000 bytes in. Je maakt hem in beide richtingen twee keer zo groot: 800 rijen bij 1200 kolommen. Hoeveel bytes nemen zijn getallen nu in?",
        options={
            "double": "1.440.000",
            "four": "2.880.000",
            "eight": "5.760.000",
        },
        correct="four",
        why={
            "four": "800 × 1200 × 3 = 2.880.000: twee keer zoveel rijen en twee keer zoveel kolommen, dus vier keer zoveel pixels. De meeste van die nieuwe pixels liggen tussen oude in: daar gaat deze les over.",
            "double": "Dat is twee keer zoveel. Maar de rijen verdubbelen en de kolommen ook: rijen × kolommen × kleuren, 800 × 1200 × 3. Zie *Het idee* van [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "eight": "Dat zou 8 keer zoveel zijn: alsof de kleuren ook verdubbelden. Een pixel heeft nog steeds 3 kleuren: 800 × 1200 × 3.",
        },
    )
    quick_bytes
    return (quick_bytes,)


@app.cell(hide_code=True)
def quick_check_wrap():
    quick_wrap = L.choice(
        "quick-wrap",
        question="`a` en `b` zijn arrays met dtype uint8. Een pixel is 40 in `a` en 200 in `b`. Wat is hij in `a - b`?",
        options={
            "minus": "-160",
            "wrap": "96",
            "zero": "0",
            "plus": "160",
        },
        correct="wrap",
        why={
            "wrap": "uint8 blijft tussen 0 en 255: onder 0 loopt hij rond, naar 256 - 160 = 96. Deze les mengt pixels; je ziet nog waar dat ertoe doet.",
            "minus": "In gewone gehele getallen wel. Maar uint8 kan niet onder 0. Zie de waarschuwing in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "zero": "uint8 stopt niet bij 0: één stap onder 0 komt uit op 255, zoals een klok rondgaat. 40 - 200 is -160, en 256 - 160 is 96.",
            "plus": "Dat is de grootte van het verschil, `abs(a.astype(int) - b.astype(int))`. In uint8 loopt 40 - 200 rond: 256 - 160.",
        },
    )
    quick_wrap
    return (quick_wrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model dat foto's herkent, verwacht ze in één vaste grootte, zoals 224 bij 224 pixels, dus elke foto die het ziet, wordt eerst groter of kleiner gemaakt. Een telefoon doet hetzelfde elke keer dat je inzoomt op een foto, en een website elke keer dat hij een klein voorbeeld van een grote foto laat zien. Het nieuwe beeld heeft andere pixels dan het oude, en de meeste liggen tussen oude pixels in: hun waarde moet je uitrekenen uit de pixels eromheen. Dat is interpolatie, en hoe je dat doet, bepaalt wat je ziet: blokken, of zachte overgangen.

    **Een voorbeeld uit het dagelijks leven.** Zoom ver in op een foto op je telefoon. Op een gegeven moment wordt een rand die scherp was een zachte waas: de telefoon mengt de oude pixels rond elke nieuwe pixel. Een tekenprogramma voor pixel art laat elke pixel juist zien als een scherp vierkantje: het herhaalt de dichtstbijzijnde oude pixel.
    """)
    return


@app.cell(hide_code=True)
def predict_row():
    guess_row = L.guess(
        "row",
        question="Een rij van drie pixels, 0, 100 en 200, wordt twee keer zo lang gemaakt: zes pixels. Elke nieuwe pixel krijgt een waarde tussen die van de oude pixels eromheen, en de dichtstbijzijnde oude pixel telt zwaarder. Wat is de nieuwe rij?",
        options={
            "blocks": "0 0 100 100 200 200",
            "even": "0 40 80 120 160 200",
            "quarters": "0 25 75 125 175 200",
        },
        reveal="**0 25 75 125 175 200.** Elke nieuwe pixel is een halve oude pixel breed, en zijn midden ligt op 0,25; 0,75; 1,25; 1,75; 2,25 of 2,75 in de oude rij; de middens van de oude pixels liggen op 0,5; 1,5 en 2,5. De tweede nieuwe pixel, op 0,75, ligt op een kwart van het midden van de 0 naar het midden van de 100: 25. De eerste, op 0,25, ligt vóór het eerste midden, waar links niets is: die blijft 0. De rij 0 0 100 100 200 200 herhaalt de dichtstbijzijnde pixel, zonder te mengen. En 0 40 80 120 160 200 legt de eerste en de laatste nieuwe pixel op de middens van de eerste en de laatste oude pixel; sommige programma's kunnen dat ook, maar Pillow en deze les kijken waar de middens van de nieuwe pixels vallen.",
    )
    guess_row
    return (guess_row,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zie een beeld als een rooster van vierkantjes, één per pixel. In een rij beslaat pixel 0 het stuk van 0 tot 1, pixel 1 het stuk van 1 tot 2, enzovoort: het midden van pixel `c` ligt op `c + 0.5`. Een nieuw beeld bedekt dezelfde scène met vierkantjes van een andere grootte. Maak een rij van `w` pixels `W` pixels lang, en elke nieuwe pixel is `w / W` oude pixels breed: het midden van nieuwe pixel `c` ligt op `(c + 0.5) * w / W` in de oude rij. Maak je een rij van 3 pixels twee keer zo lang, dan liggen de nieuwe middens op 0,25; 0,75; 1,25; 1,75; 2,25 en 2,75.

        **Nearest-neighbor-interpolatie** geeft elke nieuwe pixel de waarde van de oude pixel waarvan het vierkantje zijn midden bevat: oude pixel `int((c + 0.5) * w / W)`. De rij 0, 100, 200 wordt 0 0 100 100 200 200: elke oude pixel wordt een blok van twee. Nearest neighbor maakt nooit een waarde die er niet was. Maak je een beeld groter, dan krijgt het blokken; maak je het kleiner, dan slaat het pixels over.

        **Lineaire interpolatie** mengt de twee oude pixels aan weerszijden van een nieuw midden, en de dichtstbijzijnde telt zwaarder. Tel de positie vanaf de oude middens: `x = (c + 0.5) * w / W - 0.5`. Die ligt tussen pixel `c0 = int(x)` en pixel `c0 + 1`, op `u = x - c0` van de afstand van de een naar de ander, en zijn waarde is `(1 - u) * links + u * rechts`. Het midden van de tweede nieuwe pixel van het voorbeeld ligt op 0,75, dus geteld vanaf de oude middens op `x = 0.25`: op een kwart van 0 naar 100, dus 0,75 × 0 + 0,25 × 100 = 25. Vóór het eerste midden en na het laatste is er maar aan één kant een pixel: houd `x` tussen 0 en `w - 1`, en de buitenste pixel telt als enige mee. De rij wordt 0 25 75 125 175 200.

        **Bilineaire interpolatie** doet hetzelfde in twee richtingen. In een beeld ligt het midden van een nieuwe pixel tussen vier oude pixels: de rijen `r0` en `r1 = r0 + 1`, en de kolommen `c0` en `c1 = c0 + 1`, met `t` hoe ver het midden onder `r0` ligt, en `u` hoe ver het rechts van `c0` ligt. Meng eerst de twee pixels van rij `r0` met `u`, zoals in één rij, en de twee pixels van rij `r1` op dezelfde manier. Meng die twee uitkomsten daarna met `t`: de dichtstbijzijnde rij telt zwaarder.

        Een voorbeeld: vier pixels, 0 en 80 bovenaan, en 40 en 200 eronder. Halverwege naar beneden en op een kwart naar rechts zijn `t = 0.5` en `u = 0.25`. Boven: 0,75 × 0 + 0,25 × 80 = 20. Onder: 0,75 × 40 + 0,25 × 200 = 80. Waarde: 0,5 × 20 + 0,5 × 80 = 50.
        """),
        L.watch_out("Pillow schrijft de grootte van een afbeelding als (breedte, hoogte): eerst het aantal kolommen, dan het aantal rijen. De shape van NumPy is (rijen, kolommen). `Image.fromarray(img).resize((600, 400))` geeft een afbeelding van 400 rijen bij 600 kolommen."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **interpolatie** | waarden tussen bekende waarden uitrekenen, uit de waarden eromheen | `resize` in Pillow, voor een afbeelding |
        | **nearest-neighbor-interpolatie** | elke nieuwe pixel krijgt de waarde van de oude pixel waarvan het vierkantje zijn midden bevat | `Image.NEAREST` |
        | **lineaire interpolatie** | de twee waarden aan weerszijden van een positie mengen, zodat de dichtstbijzijnde zwaarder telt | `linear_at` in het uitgewerkte voorbeeld |
        | **bilineaire interpolatie** | elke nieuwe pixel mengt de vier oude pixels rond zijn midden, en de dichtstbijzijnde tellen zwaarder | `Image.BILINEAR` |
        | **`int(x)`** | een float als geheel getal: wat na de punt komt, valt weg | `int(2.75)` is `2` |
        | **`min(a, b)`, `max(a, b)`** | het kleinste en het grootste van twee getallen; `min(max(x, 0), w - 1)` houdt `x` tussen 0 en `w - 1` | `min(3, 5)` is `3` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_row():
    mo.vstack([
        mo.md("**Eén rij, vier keer zo lang**"),
        mo.Html(C.row_picture({"old": "oude pixels", "nearest": "nearest neighbor", "bilinear": "lineair",
                               "x": "positie in de oude rij, in pixels", "y": "waarde",
                               "alt": "Een rij van zes pixels, 40, 120, 200, 60, 140 en 180, als cirkels op hun middens, 0,5 tot 5,5, 24 pixels lang gemaakt. Nearest neighbor, in grijs, is een trap die halverwege twee middens springt; lineaire interpolatie is een dunne lijn van punten die recht van midden naar midden loopt, en vlak is vóór het eerste midden en na het laatste."})),
        mo.md(r"""
        De zes oude pixels zijn de cirkels, op hun middens. Als je de rij vier keer zo lang maakt, geeft nearest neighbor de grijze trap: vier nieuwe pixels voor elke oude, met de sprong halverwege twee middens. Lineaire interpolatie, bilineaire interpolatie in één rij, geeft de punten op de dunne lijn: recht van midden naar midden, en vlak vóór het eerste midden en na het laatste, waar de buitenste pixel als enige meetelt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Een stuk van de koffiefoto uit [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl), rond het lepeltje: 30 rijen bij 45 kolommen. Met de slider maak je het groter, links met nearest neighbor en rechts met bilineaire interpolatie. Beide zijn even groot getekend, dus een pixel van het nieuwe beeld wordt kleiner naarmate de factor groeit.
    """)
    return


@app.cell(hide_code=True)
def explore_zoom_cell():
    def _say(k):
        if k == 1:
            return "Factor 1: het stuk zoals het is, 30 bij 45 pixels, twee keer hetzelfde."
        return (f"Factor {k}: 30 bij 45 pixels worden {30 * k} bij {45 * k}. Links is elke oude pixel een blok van {k} bij {k}; "
                "rechts heeft elke nieuwe pixel zijn eigen mengsel van de vier oude pixels rond zijn midden.")

    def _frame(k):
        k = int(k)
        _html = C.zoom_frame(k, {"nearest": "nearest neighbor", "bilinear": "bilineair",
                                 "alt": "Twee even grote plaatjes van het lepeltje in de koffiefoto, {k} keer zo groot gemaakt, {rows} bij {cols} pixels: links met nearest neighbor, in blokken van {k} bij {k} pixels, rechts met bilineaire interpolatie, met zachte overgangen tussen de oude pixels."})
        return _html, _say(k)

    explore_zoom = L.explore("zoom", "factor", _frame, start=1, values=C.FACTORS, role="",
                             task="Schuif naar 8. Waar zie je nog de vierkantjes van de oude pixels, en waar zachte overgangen?")
    explore_zoom
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Nearest neighbor houdt elke rand hard, maar maakt er een trap van blokken van. Bilineaire interpolatie maakt van elke sprong tussen twee oude pixels een helling, dus randen en glimmende plekken worden zacht. Geen van beide voegt details toe: het stuk had 30 bij 45 pixels, en wat daar niet in zit, kan geen interpolatie verzinnen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`.astype(np.uint8)` laat weg wat na de punt komt, net als `int`: 67,9 wordt 67. Tel er eerst 0,5 bij op, dan rondt hij af: met `(out + 0.5).astype(np.uint8)` worden 67,9 en 67,4 netjes 68 en 67."),
        mo.md(r"""
        ### Recept

        Zo maak je een beeld `img` van `h` rijen en `w` kolommen `H` rijen en `W` kolommen groot:

        1. Maak een leeg beeld van de nieuwe grootte, `np.zeros((H, W, 3))`.
        2. Voor elke nieuwe pixel `(r, c)`: zijn midden ligt op rij `(r + 0.5) * h / H` en kolom `(c + 0.5) * w / W` van het oude beeld.
        3. Nearest neighbor: neem de oude pixel op rij `int((r + 0.5) * h / H)` en kolom `int((c + 0.5) * w / W)`.
        4. Bilineair: trek van beide 0,5 af, om vanaf de middens te tellen, en houd ze tussen 0 en `h - 1` en tussen 0 en `w - 1`, met `max` en `min`. Meng dan de vier oude pixels eromheen, zoals in *Het idee*.
        5. Rond af terug naar uint8.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel knipt het stuk rond het lepeltje uit de koffiefoto, en maakt het vier keer zo groot met nearest neighbor, stap 1 tot en met 3 van het recept. Hij doet hetzelfde met `Image.NEAREST` van Pillow, en print de twee shapes en in hoeveel getallen de twee uitkomsten verschillen. De tweede cel doet lineaire interpolatie in één rij, met stap 4 van het recept: `linear_at` mengt de twee pixels rond een positie, en `stretch` maakt een rij langer. Hij print de rij uit de gok, twee keer zo lang gemaakt, en wat `Image.BILINEAR` van Pillow ervan maakt.
    """)
    return


@app.cell
def example_nearest():
    import numpy as np
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    spoon = img[230:260, 320:365]


    def nearest(img, H, W):
        h, w = len(img), len(img[0])
        # 1. een leeg beeld van de nieuwe grootte
        out = np.zeros((H, W, 3), dtype=np.uint8)
        for r in range(H):
            for c in range(W):
                # 2. en 3. de oude pixel waarvan het vierkantje het midden van de nieuwe bevat
                out[r, c] = img[int((r + 0.5) * h / H), int((c + 0.5) * w / W)]
        return out


    big = nearest(spoon, 120, 180)
    by_pillow = np.array(Image.fromarray(spoon).resize((180, 120), Image.NEAREST))
    print(spoon.shape, big.shape, (big != by_pillow).sum())
    return Image, np


@app.cell
def example_linear(Image, np):
    def linear_at(row, x):
        # de twee pixels rond positie x, en hoe ver x van de linker ligt
        c0 = int(x)
        c1 = min(c0 + 1, len(row) - 1)
        u = x - c0
        return (1 - u) * row[c0] + u * row[c1]


    def stretch(row, W):
        w = len(row)
        out = []
        for c in range(W):
            # 4. geteld vanaf de middens, en binnen de rij
            x = (c + 0.5) * w / W - 0.5
            x = min(max(x, 0), w - 1)
            out.append(linear_at(row, x))
        return np.array(out)


    print(stretch([0, 100, 200], 6))
    row = np.array([[0, 100, 200]], dtype=np.uint8)
    print(np.array(Image.fromarray(row).resize((6, 1), Image.BILINEAR)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het stuk rond het lepeltje heeft shape `(30, 45, 3)`, en vier keer zo groot `(120, 180, 3)`. `Image.NEAREST` van Pillow geeft precies hetzelfde: in geen van de 64.800 getallen verschillen de twee. De rij 0, 100, 200 twee keer zo lang gemaakt is 0 25 75 125 175 200, zoals in de gok, en `Image.BILINEAR` van Pillow maakt dezelfde rij. In oefening 2 schrijf je bilineaire interpolatie voor een beeld: lineaire interpolatie langs twee rijen, en daarna ertussen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Labels, twee keer zo groot

    De course tekende een plaatje van fruit, `data/photos/fruit.png`, en bewaarde ernaast het label van elke pixel, in `data/photos/fruit-labels.png`: een grijs beeld van 240 rijen bij 360 kolommen, waarin elke pixel 0 is voor het tafelkleed, 1 voor een appel, 2 voor de sinaasappel, 3 voor de citroen en 4 voor de limoen. Een model dat het fruit leert vinden, wil het plaatje twee keer zo groot, en de labels mee.
    """)
    return


@app.cell(hide_code=True)
def choice_labels_cell():
    choice_labels = L.choice(
        "labels",
        question="Met welke interpolatie maak je het labelbeeld twee keer zo groot?",
        options={
            "nearest": "Nearest neighbor",
            "bilinear": "Bilineaire interpolatie",
            "rounded": "Bilineaire interpolatie, afgerond op gehele getallen",
            "either": "Het maakt niet uit: allebei houden ze de labels",
        },
        correct="nearest",
        why={
            "nearest": "Nearest neighbor kopieert alleen labels die er zijn: elke nieuwe pixel krijgt het label van één oude pixel. Bilineaire interpolatie mengt labels alsof het hoeveelheden zijn: tussen het tafelkleed, 0, en de citroen, 3, maakt hij getallen als 0,75 en 2,25, die geen label zijn, of afgerond het verkeerde.",
            "bilinear": "Bilineaire interpolatie mengt labels alsof het hoeveelheden zijn: tussen het tafelkleed, 0, en de citroen, 3, maakt hij getallen als 0,75 en 2,25. Een label is een naam die als getal geschreven is: 0,75 is helemaal geen fruit.",
            "rounded": "Afgerond wordt 0,75 een 1 en 2,25 een 2: een appel en de sinaasappel, tussen het tafelkleed en de citroen. Twee keer zo groot krijgen 1940 van de 345.600 pixels een label dat geen van de vier oude pixels eromheen had: een dun randje appel om de sinaasappel, de citroen en de limoen, sinaasappel om de citroen en de limoen, en citroen om de limoen.",
            "either": "Voor een foto geven ze allebei een plaatje. Voor labels mengt bilineaire interpolatie namen alsof het hoeveelheden zijn: tussen het tafelkleed, 0, en de citroen, 3, maakt hij 0,75 en 2,25. Welke interpolatie kopieert alleen waarden die er zijn?",
        },
    )
    choice_labels
    return (choice_labels,)


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De waarde tussen vier pixels

        Maak `value_at(img, y, x)` af, die de bilineaire waarde geeft op rijpositie `y` en kolompositie `x`. Beide zijn geteld vanaf de middens van de pixels, zoals in stap 4 van het recept, en liggen al binnen het beeld. `img` is een lijst van rijen of een array, grijs of in kleur. De eerste regels staan er al: `r0` en `r1` zijn de twee rijen rond de positie, `c0` en `c1` de twee kolommen, en `t` en `u` zeggen hoe ver de positie onder `r0` en rechts van `c0` ligt. Bij de laatste rij en de laatste kolom blijven `r1` en `c1` daar ook. Vervang de drie `...` zoals in *Het idee*: `top` mengt de twee pixels van rij `r0`, `bottom` die van rij `r1`, en de functie geeft de mix van die twee terug. De check roept hem aan met deze vier pixels en de positie op een kwart naar beneden en halverwege naar rechts:

        ```python
        img = [[0, 100],
               [200, 40]]
        value_at(img, 0.25, 0.5)
        ```

        Dat moet `67.5` geven. De check probeert ook de laatste rij en kolom, een foto van uint8, en een kleurenbeeld.
        """),
        L.task("Laat `value_at(img, y, x)` de vier pixels rond de positie mengen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    value_code = L.code_box("value", lines=8)
    value_code
    return (value_code,)


@app.cell(hide_code=True)
def check_value(value_code):
    result_value = L.check("value", value_code.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "*Het idee* mengt vier pixels in drie stappen: eerst langs de twee rijen, dan ertussen. `linear_at` in het uitgewerkte voorbeeld is zo'n mix.",
            "`top` is `(1 - u)` keer de linkerpixel van rij `r0`, plus `u` keer de rechter. `bottom` is hetzelfde voor rij `r1`. De waarde is `(1 - t)` keer `top`, plus `t` keer `bottom`.",
            "`top = (1 - u) * img[r0][c0] + u * img[r0][___]`, `bottom = (1 - u) * img[___][c0] + u * img[r1][c1]`, en `return (1 - t) * ___ + t * bottom`.",
        ],
        why="Voor `value_at(img, 0.25, 0.5)`: `r0 = 0`, `r1 = 1`, `c0 = 0`, `c1 = 1`, `t = 0.25` en `u = 0.5`. Halverwege de bovenste rij, tussen 0 en 100, ligt 50; halverwege de onderste rij, tussen 200 en 40, ligt 120. Op een kwart van 50 naar 120: 0,75 × 50 + 0,25 × 120 = 67,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_half():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Half zo groot, met Pillow

        Interpolatie zelf schrijven laat zien hoe het werkt; voor een hele foto gebruik je Pillow, dat veel sneller is. `Image.fromarray(img)` maakt een Pillow-afbeelding van een array, zoals in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl). De methode `resize` daarvan neemt de nieuwe grootte, als (breedte, hoogte), en de methode, zoals `Image.BILINEAR`, en geeft een nieuwe afbeelding terug. Maak de koffiefoto half zo groot, met bilineaire interpolatie, en zet hem als array in `small`.
        """),
        L.task("Zet de koffiefoto op de helft van zijn grootte in `small`, als array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_half():
    half_code = L.code_box("half", lines=6)
    half_code
    return (half_code,)


@app.cell(hide_code=True)
def check_half(half_code):
    result_half = L.check("half", half_code.value)
    result_half
    return (result_half,)


@app.cell(hide_code=True)
def hints_half():
    L.hints(
        "half",
        [
            "Eerst een Pillow-afbeelding van `coffee`, dan de methode `resize` daarvan, en dan terug naar een array met `np.array(...)`.",
            "De foto heeft 600 kolommen en 400 rijen, en Pillow wil de breedte eerst. De methode is de tweede waarde van `resize`.",
            "`small = np.array(Image.fromarray(coffee).resize((___, ___), Image.BILINEAR))`",
        ],
        why="De koffiefoto heeft shape `(400, 600, 3)`: 400 rijen en 600 kolommen. De helft daarvan is 200 rijen bij 300 kolommen. Pillow wil de breedte eerst, dus `resize((300, 200), Image.BILINEAR)`, en `np.array` maakt er een array met shape `(200, 300, 3)` van.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shrink():
    mo.md(r"""
    ### Oefening 4. Kleiner is niet hetzelfde

    De course liet de oplossing van oefening 2 met het recept over de koffiefoto lopen, en vergeleek die met `Image.BILINEAR` van Pillow. Vier keer zo groot verschilt het stuk rond het lepeltje nergens meer dan 1, op een schaal van 0 tot 255: een verschil van 1 komt van het afronden. Vier keer zo klein, 100 bij 150 pixels, verschilt de hele foto gemiddeld 4,52, en op één plek 104.
    """)
    return


@app.cell(hide_code=True)
def explain_shrink_cell():
    explain_shrink = L.explain(
        "shrink",
        question="Waarom verschillen de twee zoveel als de foto vier keer zo klein wordt, en nauwelijks als hij groter wordt? Denk aan hoeveel oude pixels er onder één nieuwe pixel liggen, en hoeveel daarvan je functie mengt.",
        model_answer="**Vier keer zo klein beslaat één nieuwe pixel 4 bij 4 = 16 oude pixels.** **Bilineaire interpolatie mengt alleen de vier rond zijn midden;** de andere 12 tellen helemaal niet mee. Een dunne lijn of een helder spikkeltje tussen die 12 verdwijnt, en één tussen de vier telt voor een kwart: wat overblijft, hangt ervan af waar de middens toevallig vallen. **Pillow kijkt bij verkleinen naar meer:** hij mengt elke oude pixel onder de nieuwe, en die eromheen, met minder gewicht naarmate ze verder liggen. **Als je groter maakt, is een nieuwe pixel kleiner dan een oude,** en de vier oude pixels rond zijn midden zijn alles wat er dichtbij ligt, dus de twee doen hetzelfde.",
        points=[
            "je zegt dat een nieuwe pixel vier keer zo klein 16 oude pixels beslaat",
            "je zegt dat bilineaire interpolatie alleen de vier rond zijn midden mengt, dus dat er 12 niet meetellen",
            "je zegt dat Pillow naar meer oude pixels moet kijken dan die vier, zoals alle 16 onder de nieuwe pixel",
            "je zegt dat bij vergroten de vier oude pixels rond het midden alles zijn wat telt, dus dat de twee overeenkomen",
        ],
    )
    explain_shrink
    return (explain_shrink,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een beeld is een rooster van vierkantjes, en een nieuw beeld een rooster van andere vierkantjes over dezelfde scène: elke nieuwe pixel haalt zijn waarde uit de oude pixels rond zijn midden. Nearest neighbor kopieert er één, en houdt zo labels goed en randen hard, in blokken. Bilineaire interpolatie mengt er vier, en geeft zachte overgangen en een foto die er glad uitziet als hij groter wordt. Als hij veel kleiner wordt, kijken beide naar te weinig oude pixels; Pillow mengt dan elke oude pixel onder de nieuwe, en die eromheen. Een beeld draaien of vervormen werkt net zo: reken voor elke nieuwe pixel uit waar zijn midden in het oude beeld ligt, en interpoleer daar. Bicubische interpolatie, de standaard van Pillow, mengt bij vergroten de 4 bij 4 oude pixels rond een midden, langs gladde krommen in plaats van rechte lijnen, voor scherpere randen. Verderop in deze tak maakt *Een foto klaarmaken voor een model* elke foto zo groot als een pretrained model verwacht. De volgende les, *Eén beeld over een ander: alpha en green screens*, mengt twee beelden pixel voor pixel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_labels, explain_shrink, result_half, result_value):
    L.wrap_up(choice_labels, result_value, result_half, explain_shrink)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
