# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-sobel-edges. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-sobel-edges", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) en [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="Op één punt is de gradient van een loss $\\mathcal{L}(w, b)$ gelijk aan $(2, -1)$. Wat zegt dat?",
        options={
            "up": "De loss stijgt het snelst als w groter wordt en b kleiner: de gradient wijst bergop",
            "low": "Het laagste punt van de loss ligt bij w = 2, b = −1",
            "down": "De loss daalt het snelst als w groter wordt en b kleiner",
        },
        correct="up",
        why={
            "up": "De partiële afgeleide naar w is 2: de loss stijgt als w groter wordt. Die naar b is −1: de loss stijgt als b kleiner wordt. Samen wijzen ze bergop, de kant waar de loss het snelst stijgt; gradient descent gaat de andere kant op.",
            "low": "De gradient bestaat uit twee hellingen op het punt waar je staat, niet uit een plek. Wat zegt een helling van 2 voor w? Zie de tabel met nieuwe woorden van [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
            "down": "De gradient wijst bergop. Welke kant gaat gradient descent op, met de gradient mee of ertegenin? Zie [Twee parameters: de gradient](/course/open/math-gradient?lang=nl).",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="Hoe lang is de vector $(3, 4)$: wat is zijn norm $\\lVert (3, 4) \\rVert$?",
        options={
            "five": "5",
            "seven": "7",
            "twentyfive": "25",
        },
        correct="five",
        why={
            "five": "De norm is de wortel uit de som van de kwadraten: √(3² + 4²) = √25 = 5. In code: `np.linalg.norm([3, 4])`.",
            "seven": "7 is 3 + 4, de som van de delen. De norm kwadrateert eerst elk deel. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "twentyfive": "25 is 3² + 4², de som van de kwadraten. Er ontbreekt nog één stap. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een rand is waar het ene ding ophoudt en het volgende begint: de buitenkant van een kop tegen de tafel, een letter tegen het papier, de kant van een weg. In de randen zit veel van wat een afbeelding laat zien: een tekening van alleen de omtrekken is nog steeds duidelijk een kop. Daarom is randen zoeken een oude eerste stap in beeldbewerking, vóór vormen vinden, voorwerpen opmeten, tekst lezen of een auto tussen de lijnen houden. Een rand is waar de helderheid snel verandert, en hoe snel iets verandert, is precies wat een gradient meet.

    **Een voorbeeld uit het dagelijks leven.** Op een wandelkaart is een steile wand de plek waar de hoogtelijnen dicht op elkaar liggen, omdat de hoogte daar snel verandert. Stel je een foto in grijstinten voor als een landschap: lichte grijstinten zijn hoog, donkere laag. Een rand is dan een steile wand, en een stuk van één grijstint een vlakte. De gradient van [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) zei op elk punt welke kant bergop is en hoe steil het is; deze les vraagt hetzelfde voor elke pixel.
    """)
    return


@app.cell(hide_code=True)
def predict_grain():
    guess_grain = L.guess(
        "grain",
        question="Een programma dat randen zoekt, markeert een pixel als rand waar de helderheid snel verandert. Op de foto van de koffiekop van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl) telt ongeveer 6% van de pixels als rand. Nu krijgt de foto de korrel van die les: elke grijstint plus of min een willekeurig getal met standaardafwijking 25. Hoeveel pixels tellen bij dezelfde instelling als rand?",
        options={
            "same": "Ongeveer evenveel: 6%",
            "twice": "Ongeveer twee keer zoveel: 12%",
            "five": "Ongeveer vijf keer zoveel: 30%",
        },
        reveal="**Ongeveer vijf keer zoveel**: 29,4%. Een programma dat randen zoekt, vergelijkt elke pixel met zijn buren, en korrel is van pixel tot pixel willekeurig: een pixel die iets te licht is naast een die iets te donker is, lijkt een kleine rand. Korrel met standaardafwijking 25 is klein naast een echte rand, maar hij zit overal. Het uitgewerkte voorbeeld meet het, en laat zien wat eerst gladstrijken doet.",
    )
    guess_grain
    return (guess_grain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een grijswaardenbeeld is een functie met twee inputs: de grijstint op rij `r` en kolom `c`. De partiële afgeleide naar `c` zegt hoe snel de grijstint verandert als je naar rechts gaat, terwijl de rij vast blijft. Net als in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) schat je die met een verschil. Een pixel is de kleinste stap die er is, dus je neemt er aan elke kant één: de grijstint rechts min de grijstint links, `img[r, c + 1] - img[r, c - 1]`. Die verandering gaat over 2 pixels, dus de helling per pixel is de helft ervan; in beeldbewerking laat je het delen meestal weg, omdat alleen de vergelijking met een drempel telt. Een **Sobel-filter** doet dat voor een heel venster van 3 bij 3; er zijn er twee, één naar rechts en één naar beneden:
        """),
        L.formula(r"K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \qquad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}"),
        mo.md(r"""
        $K_x$ neemt de rechterkolom van het venster min de linkerkolom, waarbij de drie rijen 1, 2 en 1 keer meetellen, zoals in het Gauss-filter van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl): de middelste rij telt het zwaarst, en de korrel van één rij telt minder. De uitkomst $g_x$ is groot en positief waar de helderheid naar rechts toeneemt, groot en negatief waar die afneemt, en 0 waar die van links naar rechts gelijk blijft. $K_y$ is $K_x$ een kwartslag met de klok mee gedraaid, wat hier hetzelfde is als zijn getransponeerde: de onderste rij min de bovenste rij. De uitkomst $g_y$ is positief waar de helderheid naar beneden toeneemt.

        Samen is $(g_x, g_y)$ de **beeldgradient** van de pixel in het midden van het venster. Net als de gradient van [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) wijst hij bergop: naar de lichtere kant, recht dwars op de rand. Zijn lengte, de norm van [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), is de *randsterkte*: hoe snel de helderheid daar verandert, welke kant de rand ook op loopt.
        """),
        L.formula(r"\lVert (g_x, g_y) \rVert = \sqrt{g_x^2 + g_y^2}"),
        mo.md(r"""
        **Randdetectie** markeert de pixels waarvan de randsterkte boven een drempel ligt.
        """),
        L.watch_out("$g_y$ kijkt naar beneden: rij 0 is de bovenste rij van een afbeelding, dus een positieve $g_y$ betekent dat het eronder lichter is. In een grafiek groeit y naar boven; in een afbeelding is het andersom."),
        mo.md(r"""
        Vier vensters van 3 bij 3, met donker 0 en licht 255:

        | het venster | $g_x$ | $g_y$ | randsterkte |
        |---|---|---|---|
        | rechts licht: `[[0, 0, 255], [0, 0, 255], [0, 0, 255]]` | 1020 | 0 | 1020 |
        | onder licht: `[[0, 0, 0], [0, 0, 0], [255, 255, 255]]` | 0 | 1020 | 1020 |
        | rechtsonder licht: `[[0, 0, 0], [0, 0, 255], [0, 255, 255]]` | 765 | 765 | 1082 |
        | één grijstint: `[[90, 90, 90], [90, 90, 90], [90, 90, 90]]` | 0 | 0 | 0 |

        In het derde venster wijst de gradient $(765, 765)$ naar rechtsonder, waar het licht is, en de rand loopt schuin, dwars erop.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **randdetectie** | de plekken in een afbeelding vinden waar de helderheid snel verandert; vaak zijn dat de grenzen van voorwerpen | `strength > 150` |
        | **Sobel-filter** | een filter van 3 bij 3 dat schat hoe snel de helderheid verandert naar rechts ($K_x$) of naar beneden ($K_y$) | `(window * KX).sum()` |
        | **beeldgradient** | voor elke pixel $(g_x, g_y)$: hoe snel de helderheid naar rechts en naar beneden verandert; hij wijst naar de lichtere kant, en zijn lengte is de randsterkte | `(gx, gy)`; de lengte `np.linalg.norm([gx, gy])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_edges():
    mo.vstack([
        mo.md("**De foto, de twee Sobel-uitkomsten en de randsterkte**"),
        mo.Html(C.edges_picture({"photo": "De foto", "gx": "gx: naar rechts", "gy": "gy: naar beneden", "strength": "Randsterkte",
                                 "alt": "Vier plaatjes van de koffiekop op zijn schoteltje. Linksboven de foto in grijstinten. Rechtsboven gx: vooral middelgrijs, met de linker- en rechterkant van de kop en het schoteltje getekend als lichte en donkere lijnen, alsof ze van opzij belicht worden; de boven- en onderrand van de kop zijn bijna weg. Linksonder gy: hetzelfde, maar nu zijn de boven- en onderrand getekend en zijn de zijkanten bijna weg. Rechtsonder de randsterkte: bijna wit, met elke rand als een donkere lijn: de rand van de kop, het schoteltje, het oor, het lepeltje en het hout van de tafel."})),
        mo.md(r"""
        Rechtsboven $g_x$ en linksonder $g_y$, met 0 als middelgrijs: licht waar de helderheid naar rechts of naar beneden toeneemt, donker waar die afneemt; alles voorbij ±400 is even licht of donker als 400, en in de randsterkte is alles boven 400 even donker als 400. $g_x$ ziet de zijkanten van de kop, en nauwelijks de boven- en onderrand; $g_y$ juist andersom. De randsterkte rechtsonder voegt de twee samen: elke rand is een donkere lijn, welke kant hij ook op loopt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("De uitkomsten van een Sobel-filter komen onder 0 en boven 255 uit: op deze foto van −772 tot 769. Zet de afbeelding daarom eerst om in floats, met `np.array(img, dtype=float)`. In uint8 zou een verschil als `img[r, c + 1] - img[r, c - 1]` onder 0 rondlopen, zoals in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl); met floats is elke stap veilig."),
        L.watch_out("Een Sobel-uitkomst is 8 keer de helling in grijstinten per pixel: het verschil gaat over 2 pixels, en de weights 1, 2, 1 zijn samen 4. Bibliotheken zoals OpenCV laten dat zo; het verandert alleen welke drempel je kiest."),
        mo.md(r"""
        ### Recept

        Om de randen van een grijswaardenbeeld `img` te vinden:

        1. Maak een array met nullen voor het resultaat, 2 kleiner in elke richting: `out = np.zeros((len(img) - 2, len(img[0]) - 2))`.
        2. Ga langs elke plek, met `for r in range(len(out)):` en daarbinnen `for c in range(len(out[0])):`, en neem het venster dat daar begint: `window = img[r:r + 3, c:c + 3]`. Net als in [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl) heeft dat venster de pixel `img[r + 1, c + 1]` in het midden: `out[r, c]` is de randsterkte van die pixel.
        3. Reken de beeldgradient uit, `gx = (window * KX).sum()` en `gy = (window * KY).sum()`, en sla zijn lengte op: `out[r, c] = np.linalg.norm([gx, gy])`.
        4. Markeer de randen met een drempel: `edges = out > 150` is True waar de randsterkte boven 150 ligt. 150 is de keuze van de course voor de schone foto; oefening 3 laat zien hoe je er een kiest.

        Heeft de afbeelding korrel, filter hem dan eerst met het Gauss-filter van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl).
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De foto komt uit [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl): het stuk met de koffiekop, in grijstinten, met dezelfde korrel: standaardafwijking 25, seed 0. De eerste codecel maakt de foto met korrel, en een gladgestreken versie: de foto met korrel na het Gauss-filter. De tweede rekent de randsterkte uit van drie foto's, de schone, die met korrel en de gladgestreken foto, en print voor elke foto de grootte van het resultaat, de grootste randsterkte, en het deel van de pixels met een randsterkte boven 150. Onder de cel staan de randen van de schone foto. Voer de twee codecellen uit; de tweede duurt een paar seconden.
    """)
    return


@app.cell
def example_photos():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    # de foto in grijstinten, de sterkste kleur van elke pixel, en het stuk met de kop
    clean = img.max(axis=2)[0:300, 100:500].astype(float)
    rng = np.random.default_rng(0)
    grainy = clean + rng.normal(0, 25, size=clean.shape)
    # het Gauss-filter van 3 bij 3, 2 kleiner in elke richting
    G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    smoothed = np.zeros((len(grainy) - 2, len(grainy[0]) - 2))
    for r in range(len(smoothed)):
        for c in range(len(smoothed[0])):
            smoothed[r, c] = (grainy[r:r + 3, c:c + 3] * G).sum()
    return clean, grainy, np, plt, smoothed


@app.cell
def example_edges(clean, grainy, np, plt, smoothed):
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    def edge_strength(photo):
        # 1. een array met nullen, 2 kleiner in elke richting
        out = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        # 2. elk venster van 3 bij 3
        for r in range(len(out)):
            for c in range(len(out[0])):
                window = photo[r:r + 3, c:c + 3]
                # 3. de beeldgradient, en zijn lengte
                gx = (window * KX).sum()
                gy = (window * KY).sum()
                out[r, c] = np.linalg.norm([gx, gy])
        return out


    strengths = [edge_strength(photo) for photo in [clean, grainy, smoothed]]
    for strength in strengths:
        # 4. het deel boven de drempel: het gemiddelde van True (1) en False (0)
        print(strength.shape, strength.max(), (strength > 150).mean())
    # de randen van de schone foto: True is zwart, False wit
    plt.imshow(strengths[0] > 150, cmap="gray_r")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de schone foto is de grootste randsterkte 777, en ligt een aandeel van 0,060 van de pixels, 6,0%, boven 150: de omtrekken van de kop, het schoteltje en het lepeltje, en een deel van het hout van de tafel, in het plaatje onder de cel. In de foto met korrel is dat 29,4%, het antwoord van de gok: korrel maakt overal randen. Een Sobel-filter neemt een verschil, en korrel is van pixel tot pixel willekeurig, dus in een verschil valt die niet tegen elkaar weg zoals in een gemiddelde. De korrel alleen geeft gemiddeld een randsterkte van ongeveer 110, ook op een stuk van één grijstint; na het Gauss-filter ongeveer 50. Met het Gauss-filter eerst is het weer 6,0%, en de grootste randsterkte is 667: het filter middelt de meeste korrel weg, en maakt de randen iets minder steil. Maar 6,0% wil niet zeggen dat het dezelfde pixels zijn als in de schone foto; oefening 3 meet hoeveel er fout zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke kant is lichter?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="Bij één pixel van een foto is de beeldgradient $(g_x, g_y) = (0, 400)$. Waar is de lichtere kant?",
        options={
            "below": "Onder de pixel",
            "above": "Boven de pixel",
            "right": "Rechts van de pixel",
            "none": "Nergens: er is geen rand, want $g_x$ is 0",
        },
        correct="below",
        why={
            "below": "$g_x = 0$: van links naar rechts blijft de helderheid gelijk. $g_y = 400$ is positief: de helderheid neemt naar beneden toe, omdat de rijnummers naar beneden groter worden. De lichtere kant ligt dus onder de pixel, en de rand loopt van links naar rechts, dwars op de gradient.",
            "above": "Een positieve $g_y$ betekent dat de helderheid toeneemt als het rijnummer groter wordt. Waar staat rij 0 in een afbeelding? Zie de let-op in *Het idee*.",
            "right": "Welke van de twee zegt iets over links en rechts? Die is hier 0.",
            "none": "De randsterkte is de lengte van $(0, 400)$: 400, niet 0. Een deel dat 0 is, zegt alleen welke kant de rand op loopt.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_strength():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De randsterkte

        Maak `edge_strength(img)` af. Die krijgt een grijswaardenbeeld en geeft de randsterkte van elke plek terug, 2 kleiner in elke richting. De twee Sobel-filters, de loops en het venster staan er al: vervang de `...` door `gx`, `gy`, en de lengte van `(gx, gy)`, opgeslagen in `out[r, c]`. De check roept hem aan met een verloop dat per pixel 3 grijstinten oploopt naar rechts en 4 naar beneden:

        ```python
        edge_strength([[0, 3, 6, 9],
                       [4, 7, 10, 13],
                       [8, 11, 14, 17],
                       [12, 15, 18, 21]])
        ```

        Dat moet `[[40, 40], [40, 40]]` geven. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `edge_strength(img)` elke plek de lengte van zijn beeldgradient geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_strength():
    box_strength = L.code_box("strength", lines=14)
    box_strength
    return (box_strength,)


@app.cell(hide_code=True)
def check_strength(box_strength):
    result_strength = L.check("strength", box_strength.value)
    result_strength
    return (result_strength,)


@app.cell(hide_code=True)
def hints_strength():
    L.hints(
        "strength",
        [
            "Drie regels: `gx` is het venster keer `KX`, opgeteld; `gy` hetzelfde met `KY`; en `out[r, c]` krijgt de lengte van `(gx, gy)`.",
            "Vermenigvuldig plek voor plek met `window * KX`, en tel de negen producten op met `.sum()`. De lengte van een vector is `np.linalg.norm(...)`, zoals in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "`gx = (window * KX).sum()`, dan `gy` op dezelfde manier met `KY`, en `out[r, c] = np.linalg.norm([___, ___])`.",
        ],
        why="In elk venster van het verloop is de rechterkolom in elke rij 6 grijstinten meer dan de linker; met de weights 1, 2, 1 is `gx` dus 6 × 4 = 24. De onderste rij is in elke kolom 8 grijstinten meer dan de bovenste, dus `gy` is 8 × 4 = 32. De lengte van (24, 32) is √(576 + 1024) = √1600 = 40.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Oefening 3. Waar leg je de drempel?

    Randdetectie heeft een drempel nodig: een pixel met een randsterkte erboven telt als rand. Met de slider kies je de drempel voor de gladgestreken foto van het uitgewerkte voorbeeld: de foto met korrel na het Gauss-filter. Als antwoordmodel gelden de randen van de schone foto, de pixels met een randsterkte boven 150. Links de randen bij de gekozen drempel; rechts voor elke drempel het deel van de pixels dat fout is: een rand waar de schone foto er geen heeft, of geen rand waar hij er wel een heeft. De as stopt bij 12%, zodat het laagste punt te zien is: een driehoekje bovenaan betekent dat de waarde hoger ligt.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"edges": "randen boven {t}", "threshold": "drempel", "wrong": "pixels fout, in %",
                                          "alt": "Links de randen van de gladgestreken foto met korrel bij een drempel van {t}, zwart op wit; rechts het deel van de pixels dat fout is voor de drempels 0 tot en met 400, op een as tot 12%: de drempels 0 tot en met 75 liggen erboven, van 94% tot 27%, aangegeven met driehoekjes bovenaan; vanaf 100 daalt de lijn naar een laagste punt tussen 150 en 200, en stijgt dan langzaam weer tot ongeveer 6% bij 400. Drempel {t} is gemarkeerd met een grotere stip, een driehoekje bovenaan als de waarde boven 12% ligt."})
        return (_html,
                f'Drempel {t}: {L.number(_f["wrong"], 1)}% van de pixels is fout.')

    explore_threshold = L.explore("threshold", "drempel", _frame, start=0, values=C.THRESHOLDS, role="hyper",
                                  task="Zoek de drempel waarbij de minste pixels fout zijn.")
    explore_threshold
    return (explore_threshold,)


@app.cell(hide_code=True)
def check_threshold(explore_threshold):
    result_threshold = L.check("threshold", explore_threshold)
    result_threshold
    return (result_threshold,)


@app.cell(hide_code=True)
def hints_threshold():
    L.hints(
        "threshold",
        [
            "Schuif de slider steeds één stand op, en lees het getal in de zin onder het plaatje.",
            "Vanaf 0 zakt het getal snel: er komt minder korrel door als rand. Ergens gaat het weer omhoog, omdat zwakke echte randen ook verdwijnen: zoek de stand waar het omkeert.",
            "Kijk naar de standen 150, 175 en 200: welke geeft het kleinste getal?",
        ],
        why="Bij 150 is 3,2% van de pixels fout, bij 175 2,9%, en bij 200 3,2%. Onder 175 komt er meer korrel door als rand; erboven verdwijnen er meer zwakke echte randen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.md(r"""
    ### Oefening 4. Waarom eerst gladstrijken?
    """)
    return


@app.cell(hide_code=True)
def explain_smooth_cell():
    explain_smooth = L.explain(
        "smooth",
        question="Zonder het Gauss-filter eerst is de beste drempel voor de foto met korrel 300, en is nog 4,1% van de pixels fout: meer dan met het filter, bij de drempel die je in oefening 3 vond. Waarom laat korrel een Sobel-filter randen vinden waar er geen zijn, en waarom helpt het om eerst het Gauss-filter toe te passen?",
        model_answer="**Waarom korrel randen maakt:** een Sobel-filter neemt een verschil: de rechterkolom min de linker, of de onderste rij min de bovenste. Korrel is van pixel tot pixel willekeurig, dus in een verschil valt die niet tegen elkaar weg: een pixel die iets te licht is naast een die iets te donker is, geeft een verschil dat in de werkelijkheid niet bestaat. Op de foto met korrel geeft de korrel alleen al een randsterkte van ongeveer 110. **Waarom gladstrijken helpt:** het Gauss-filter neemt een gewogen gemiddelde van elk venster, en in een gemiddelde valt veel korrel tegen elkaar weg. De verschillen die overblijven komen dan vooral van echte randen, dus een lagere drempel is genoeg, en er verdwijnen minder zwakke echte randen.",
        points=[
            "je zegt dat een Sobel-filter een verschil neemt tussen naburige pixels, en dat willekeurige korrel in een verschil niet tegen elkaar wegvalt",
            "je zegt dat het Gauss-filter middelt, zodat veel korrel wegvalt voordat het verschil genomen wordt",
            "je zegt dat een lagere drempel dan genoeg is, zodat er minder zwakke echte randen verdwijnen, of dat er minder korrel als rand doorkomt",
        ],
    )
    explain_smooth
    return (explain_smooth,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De Sobel-filters zijn convoluties, zoals de randfilters van [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl), met de weights 1, 2, 1 van het Gauss-filter van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl) dwars op hun richting. De beeldgradient is de gradient van [Twee parameters: de gradient](/course/open/math-gradient?lang=nl): daar hoe snel de loss verandert voor elke parameter, hier hoe snel de helderheid verandert bij elke stap over de afbeelding. De eerste layer van een getraind CNN leert vaak filters die veel op deze twee lijken. De volgende les, *Dunne, doorlopende randen: Canny*, begint bij deze randsterkte en bij de richting van de gradient: hij dunt elke rand uit tot een lijn van één pixel breed, en houdt zwakke stukjes rand alleen waar ze aansluiten op sterke.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, explain_smooth, result_strength, result_threshold):
    L.wrap_up(choice_direction, result_strength, result_threshold, explain_smooth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
