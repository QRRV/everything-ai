# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-canny. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-canny", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl) en [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_across():
    quick_across = L.choice(
        "quick-across",
        question="Hoe ligt de beeldgradient $(g_x, g_y)$ van een pixel op een rand ten opzichte van die rand?",
        options={
            "across": "Dwars op de rand, naar de lichtere kant",
            "along": "Langs de rand, in de richting waarin de rand loopt",
            "darker": "Dwars op de rand, naar de donkerdere kant",
        },
        correct="across",
        why={
            "across": "De beeldgradient wijst bergop, de kant op waar de helderheid het snelst groeit: dwars op de rand, naar de lichtere kant. Deze les gebruikt dat: de buren dwars op een rand liggen langs zijn gradient.",
            "along": "Langs de rand verandert de helderheid nauwelijks. In welke richting verandert hij het snelst? Kijk bij *Het idee* van [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl).",
            "darker": "Een positieve $g_x$ betekent dat de helderheid naar rechts groeit. Is de rechterkant dan lichter of donkerder? Kijk in [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl).",
        },
    )
    quick_across
    return (quick_across,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="`img = np.array([[50, 120], [100, 200]])`. Wat geeft `img > 100`?",
        options={
            "right": "`[[False, True], [False, True]]`",
            "equal": "`[[False, True], [True, True]]`",
            "values": "`[[0, 120], [0, 200]]`",
        },
        correct="right",
        why={
            "right": "`>` vergelijkt elke pixel met 100: 120 en 200 zijn groter, 50 niet, en 100 is niet groter dan zichzelf. De uitkomst is een binair beeld van True en False.",
            "equal": "Is 100 groter dan 100? `>` telt alleen wat echt groter is.",
            "values": "Een vergelijking geeft voor elke pixel True of False, niet zijn grijstint. Kijk in [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De randen van [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl) zijn een paar pixels breed, ook waar de helderheid van de ene pixel op de volgende verspringt: het venster van 3 bij 3 van een Sobel-filter ziet zo'n sprong vanaf meer dan één plek, en een Gauss-filter smeert hem eerst nog uit over een paar pixels. En een drempel breekt een zwakke rand in stukken overal waar zijn sterkte even zakt. Wil je een voorwerp opmeten, voorwerpen tellen of een omtrek volgen, dan wil je één gesloten lijn van één pixel breed om elk ding. Dat geeft de Canny-randdetector. Hij is uit 1986, en nog altijd de standaardmanier om randen te vinden, in OpenCV en in andere bibliotheken.

    **Een voorbeeld uit het dagelijks leven.** Als je een vervaagde potloodtekening overtrekt met een fijne pen, trek je één lijn door het midden van elke grijze streek, niet over de hele breedte. Waar een streek vervaagt, ga je door, want je zag waar hij duidelijk was en hij loopt verder. Een losse stip op het papier laat je liggen.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="In de tekening van fruit uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) heeft elk stuk fruit een omtrek. Waar de lichte kant van een vrucht tegen het lichte tafelkleed ligt, is die omtrek zwak. Er zijn ook andere randen: tussen de ruiten van het tafelkleed, en rond de glans van de lamp op het fruit. Een programma houdt elke pixel waarvan de randsterkte boven één drempel ligt. Is er een drempel die de omtrekken heel laat en alle andere randen weglaat?",
        options={
            "yes": "Ja, als je hem goed kiest",
            "almost": "Bijna: bij de beste drempel gaan maar een paar pixels fout",
            "no": "Nee: bij de beste drempel gaan nog honderden pixels fout",
        },
        reveal="**Nee.** Sommige stukken omtrek zijn zwakker dan de sterkste randen van de ruiten, die tot 98 komen, en de glans van de lamp heeft randen tot 205. Eén drempel kijkt naar één pixel tegelijk, alleen naar zijn randsterkte, dus hij kan ze niet uit elkaar houden. Probeer je drempels in stappen van 10, dan is 110 de beste, en toch gaan er nog 524 pixels fout. Zelfs op de dunne randen van stap 3 hieronder gaan er bij de beste drempel, 100, nog 307 fout: 141 rond de glans, en 166 van de omtrek die gemist worden. Canny kijkt ook naar de buren van een pixel.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **Canny-randdetector**, van John Canny, vindt randen in vier stappen. De eerste twee zijn de lessen hiervoor:

        1. Strijk het beeld glad met het Gauss-filter van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl).
        2. Bereken de beeldgradient $(g_x, g_y)$ met de twee Sobel-filters van [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl), en zijn lengte, de randsterkte.

        Dwars over een rand stijgt en daalt de randsterkte als een heuvel. De derde stap houdt alleen de top van de heuvel:

        3. **Uitdunnen van randen**: vergelijk elke pixel met zijn twee buren dwars op de rand, aan elke kant één. De beeldgradient wijst recht dwars op de rand, dus dat zijn de buren langs de gradient. Een pixel die minstens zo sterk is als allebei, houdt zijn randsterkte; elke andere pixel krijgt 0. Wat overblijft, zijn lijnen van één pixel breed; alleen waar twee pixels naast elkaar dwars op de rand precies even sterk zijn, blijven ze allebei staan. Veel boeken noemen deze stap non-maximum suppression.

        Een pixel heeft acht buren, die in vier richtingen door hem heen liggen: links en rechts, boven en onder, en de twee diagonalen. Daarom wordt de richting van de gradient afgerond op de dichtstbijzijnde van die vier:

        | de beeldgradient | vergelijk met de buren | de stap `(dr, dc)` |
        |---|---|---|
        | $\lvert g_y \rvert \le 0{,}41 \, \lvert g_x \rvert$: ongeveer naar rechts of links | links en rechts | `(0, 1)` |
        | $\lvert g_x \rvert \le 0{,}41 \, \lvert g_y \rvert$: ongeveer naar beneden of boven | boven en onder | `(1, 0)` |
        | anders, met $g_x$ en $g_y$ van hetzelfde teken | linksboven en rechtsonder | `(1, 1)` |
        | anders, met tegengestelde tekens | rechtsboven en linksonder | `(1, -1)` |

        0,41 is de helling van de richting halverwege tussen vlak en diagonaal: 22,5 graden, de helft van 45. Met de stap `(dr, dc)` zijn de twee buren van `strength[r, c]` dan `strength[r + dr, c + dc]`, een stap vooruit, en `strength[r - dr, c - dc]`, een stap terug. In de derde rij wijzen een positieve $g_x$ en $g_y$ naar rechtsonder; twee negatieve naar linksboven, langs dezelfde lijn.
        """),
        L.watch_out("Vergelijk dwars op de rand, nooit erlangs. Langs de rand horen de buren bij dezelfde rand en zijn ze ongeveer even sterk, dus dan blijft de rand even breed als hij was."),
        mo.md(r"""
        Na het uitdunnen moet een drempel nog zeggen welke lijnen randen zijn. De vierde stap gebruikt er twee:

        4. Een **hysteresedrempel**: een pixel boven de hoge drempel is een sterke rand, zeker een rand. Een pixel tussen de lage en de hoge drempel is een zwakke rand: die telt alleen als hij via andere randpixels vastzit aan een sterke rand; waarbij ook diagonale buren meetellen. Een pixel die niet boven de lage drempel ligt, is nooit een rand.

        In de natuurkunde betekent hysterese dat wat er gebeurt afhangt van wat eraan voorafging, zoals bij een thermostaat die de verwarming aanzet onder 19 graden en pas uitzet boven 21. Hier telt een zwakke pixel als zijn rand ergens al sterk was.

        Een voorbeeld: een rij pixels langs een omtrek, met randsterktes 250, 180, 120, 150 en 230, en verderop een losse pixel van 130. De lage drempel is 100, de hoge 200.

        | | wat blijft over | waarom |
        |---|---|---|
        | één drempel van 200 | 250 en 230 | 180, 120 en 150 liggen niet boven 200: de omtrek valt in twee stukken uiteen |
        | één drempel van 100 | de hele omtrek, en de losse 130 | alles boven 100 telt, of het nu ergens aan vastzit of niet |
        | de hysteresedrempel, 100 en 200 | de hele omtrek | 180, 120 en 150 zijn zwak, maar zitten vast aan 250 en 230; de losse 130 is zwak en zit nergens aan vast |

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **Canny-randdetector** | de standaardmanier om randen te vinden, in vier stappen: gladstrijken, de beeldgradient berekenen, de randen uitdunnen en een hysteresedrempel toepassen | het uitgewerkte voorbeeld; in OpenCV `cv2.Canny`, die stap 2 tot 4 doet: gladstrijken doe je zelf eerst |
        | **uitdunnen van randen** | dwars op elke rand alleen de pixel met de grootste randsterkte houden: een pixel blijft als hij minstens zo sterk is als zijn twee buren langs de beeldgradient | `s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc])` |
        | **hysteresedrempel** | twee drempels: boven de hoge is een pixel een rand; tussen de twee alleen als hij vastzit aan zo'n sterke rand | `edges = thin > high`, en dan een to-dolijst |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_canny():
    mo.vstack([
        mo.md("**De tekening, haar randsterkte, de uitgedunde randsterkte, en drie drempels**"),
        mo.Html(C.canny_picture({"drawing": "De tekening", "strength": "Randsterkte", "thin": "Na het uitdunnen",
                                 "high": "Eén drempel: 200", "low": "Eén drempel: 100", "both": "Hysterese: 100 en 200",
                                 "alt": "Zes plaatjes van de tekening van fruit op een geruit tafelkleed. Linksboven de tekening in kleur: vier rode appels, een sinaasappel, een citroen en een limoen. Rechtsboven de randsterkte: bijna wit, met een donkere ring om elk stuk fruit, een kleine ring rond de glans van de lamp op elke vrucht, en een vaag grijs raster langs de ruiten. In het midden links, na het uitdunnen: dezelfde ringen en hetzelfde raster, nu als dunne lijnen. In het midden rechts, één drempel van 200: alleen stukken van de meeste ringen, en van de limoen en de kleine appel helemaal rechts bijna niets. Linksonder, één drempel van 100: de ringen van drie appels heel, de andere vier met gaten, en de kleine ringen van de glans. Rechtsonder, de hysteresedrempel met 100 en 200: dezelfde drie ringen heel en de kleine ringen weg, behalve op de appel linksboven; de andere vier hebben gaten, en van de ring van de limoen ontbreekt ongeveer een derde."})),
        mo.md(r"""
        Rechtsboven de randsterkte na het Gauss-filter en de Sobel-filters, donker waar hij sterk is; alles boven 400 is even donker als 400. In het midden links, na het uitdunnen: dezelfde randen, nu één pixel breed, met vage lijnen tussen de ruiten. Dan drie manieren om de uitgedunde randsterkte te drempelen. Eén drempel van 200 houdt alleen de sterkste stukken omtrek: van de limoen, rechtsboven, en de kleine appel helemaal rechts, bijna niets. Eén drempel van 100 houdt het grootste deel van elke omtrek, maar ook de glans van de lamp op elke vrucht. De hysteresedrempel met 100 en 200 houdt de omtrekken die vastzitten aan een sterk stuk, en laat de glans weg, behalve op de appel linksboven, waar een deel ervan boven 200 ligt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Als grijstint namen de vorige lessen de sterkste kleur van elke pixel. In deze tekening liggen de rode appels en het roze tafelkleed dan dicht bij elkaar, omdat ze allebei veel rood hebben: de mediaan van de grijstinten van de appels is dan 105, en die van het tafelkleed 150. Het gemiddelde van rood, groen en blauw, `img.mean(axis=2)`, geeft 48 en 119, dus de omtrekken van de appels worden sterker."),
        L.watch_out("Het uitdunnen slaat de buitenste rijen en kolommen over: een pixel daar heeft aan één kant geen buur. Ze blijven 0, dus een sterke pixel ligt nooit in de buitenste rij of kolom, en in stap 4 wordt `r - 1` nooit −1, wat in Python stilletjes de laatste rij zou betekenen."),
        mo.md(r"""
        ### Recept

        Zo vind je met Canny de randen van een grijswaardenbeeld `gray`:

        1. Strijk het glad met het Gauss-filter, bereken `gx` en `gy` met de twee Sobel-filters, en de randsterkte: `strength = (gx ** 2 + gy ** 2) ** 0.5`.
        2. Dun de randen uit: maak een array met nullen `thin`, van dezelfde vorm. Neem voor elke pixel die niet in de buitenste rij of kolom ligt de stap `dr, dc = step(gx[r, c], gy[r, c])`, en sla `strength[r, c]` op in `thin[r, c]` als hij minstens zo sterk is als de grootste van zijn twee buren, `max(strength[r + dr, c + dc], strength[r - dr, c - dc])`.
        3. De sterke pixels zijn randen: `edges = thin > high`. Zet ze allemaal op een to-dolijst, als tuple `(r, c)`.
        4. Loop de lijst af met `for r, c in todo:`. Elke pixel in het venster van 3 bij 3 rond `(r, c)` die boven `low` ligt en nog geen rand is, wordt een rand en komt achteraan op de lijst. In code is dat `not edges[rr, cc]`: `not` maakt van True False en van False True. Een `for`-lus over een lijst komt ook langs de elementen die er tijdens de lus bij komen, dus hij stopt pas als er geen nieuwe pixel meer bij kwam. Dan is elke zwakke pixel bereikt die vastzit aan een sterke.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tekening is die uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl): `data/photos/fruit.png`. De eerste codecel maakt haar grijs, strijkt haar glad en berekent de beeldgradient, met één functie voor alle drie de filters. Hij print de vorm van de tekening en van de randsterkte, en hoeveel pixels een randsterkte boven 100 hebben. De tweede cel dunt de randen uit en past de hysteresedrempel toe met 100 en 200, de eerste keus van de course. Hij print hoeveel pixels er na het uitdunnen boven 100 liggen, hoeveel er sterk zijn, en hoeveel er uiteindelijk rand zijn. Voer de twee codecellen uit.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/fruit.png"))
    # de tekening in grijstinten: het gemiddelde van rood, groen en blauw
    gray = img.mean(axis=2)
    G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    def filtered(photo, K):
        # een filter van 3 bij 3, alleen waar het hele venster past: 2 kleiner in elke richting
        out = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        for r in range(len(out)):
            for c in range(len(out[0])):
                out[r, c] = (photo[r:r + 3, c:c + 3] * K).sum()
        return out


    # 1. gladstrijken, dan de beeldgradient en zijn lengte
    smoothed = filtered(gray, G)
    gx = filtered(smoothed, KX)
    gy = filtered(smoothed, KY)
    strength = (gx ** 2 + gy ** 2) ** 0.5
    print(gray.shape, strength.shape, (strength > 100).sum())
    return gx, gy, np, plt, strength


@app.cell
def example_canny(gx, gy, np, plt, strength):
    def step(gx, gy):
        # de stap naar de buur dwars op de rand, in een van vier richtingen
        if abs(gy) <= 0.41 * abs(gx):
            return 0, 1    # naar rechts; de andere buur ligt links
        if abs(gx) <= 0.41 * abs(gy):
            return 1, 0    # naar beneden; de andere ligt erboven
        if gx * gy > 0:
            return 1, 1    # naar rechtsonder; de andere linksboven
        return 1, -1       # naar linksonder; de andere rechtsboven


    # 2. randen uitdunnen: houd van elke rand alleen de top, dwars erop
    thin = np.zeros(strength.shape)
    for r in range(1, len(thin) - 1):
        for c in range(1, len(thin[0]) - 1):
            dr, dc = step(gx[r, c], gy[r, c])
            s = strength[r, c]
            if s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc]):
                thin[r, c] = s
    print((thin > 100).sum())

    # 3. de sterke pixels zijn randen, en gaan op de to-dolijst
    low, high = 100, 200
    edges = thin > high
    todo = []
    for r in range(len(thin)):
        for c in range(len(thin[0])):
            if edges[r, c]:
                todo.append((r, c))
    # 4. een zwakke pixel naast een rand wordt ook rand, en komt op de lijst;
    #    de lus komt ook langs de pixels die er tijdens de lus bij komen
    for r, c in todo:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if thin[rr, cc] > low:
                    if not edges[rr, cc]:
                        edges[rr, cc] = True
                        todo.append((rr, cc))
    print((thin > high).sum(), edges.sum())
    # de randen: True is zwart, False wit
    plt.imshow(edges, cmap="gray_r")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tekening heeft vorm `(240, 360)` en de randsterkte `(236, 356)`: 4 kleiner in elke richting, 2 voor het Gauss-filter en 2 voor de Sobel-filters. 3465 pixels hebben een randsterkte boven 100; na het uitdunnen nog 1629, minder dan de helft, omdat elke rand nu één pixel breed is. 828 pixels zijn sterk, boven 200; met de zwakke pixels die eraan vastzitten, zijn 1473 pixels rand. In het plaatje onder de cel zijn de omtrekken van drie appels dicht, en is de glans van de lamp weg, behalve op de appel linksboven. De andere omtrekken hebben gaten, en de omtrek van de limoen, rechtsboven, breekt af: ongeveer een derde ervan wordt niet gevonden. Een deel ervan is niet sterker dan 100, en het stuk daarachter wordt nooit bereikt. Oefening 3 zoekt een betere lage drempel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_neighbors():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke buren?
    """)
    return


@app.cell(hide_code=True)
def choice_neighbors_cell():
    choice_neighbors = L.choice(
        "neighbors",
        question="Bij een pixel is de beeldgradient $(g_x, g_y) = (-300, 290)$. Met welke twee buren vergelijkt het uitdunnen die pixel?",
        options={
            "sides": "Links en rechts",
            "updown": "Boven en onder",
            "main": "Linksboven en rechtsonder",
            "anti": "Rechtsboven en linksonder",
        },
        correct="anti",
        why={
            "anti": "Geen van beide delen is klein naast het andere: 290 is meer dan 0,41 × 300 = 123, en 300 is meer dan 0,41 × 290 = 118,9, dus de gradient is diagonaal. $g_x$ is negatief en $g_y$ positief: de lichtere kant ligt linksonder. De stap is `(1, -1)`: de buren linksonder en rechtsboven, recht dwars op de rand.",
            "sides": "Dat klopt als $g_y$ klein is naast $g_x$: hooguit 0,41 × 300 = 123. Is 290 zo klein?",
            "updown": "Dat klopt als $g_x$ klein is naast $g_y$: hooguit 0,41 × 290 = 118,9. Is 300 zo klein?",
            "main": "De gradient is diagonaal, maar welke diagonaal? $g_x$ is negatief: de lichtere kant ligt links. En $g_y$ is positief: lichter eronder.",
        },
    )
    choice_neighbors
    return (choice_neighbors,)


@app.cell(hide_code=True)
def text_exercise_thin():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Randen uitdunnen

        Maak `thin(strength, gx, gy)` af. Hij krijgt de randsterkte en de twee delen van de beeldgradient, en geeft de uitgedunde randsterkte terug, met dezelfde vorm: elke pixel die minstens zo sterk is als zijn twee buren dwars op de rand, houdt zijn randsterkte, elke andere pixel is 0. `step` en de lussen staan er al: vervang de `...`. De check roept hem aan met een rand van boven naar beneden, het sterkst in de middelste kolom. De helderheid groeit naar rechts, dus `gx` is de randsterkte zelf en `gy` is 0:

        ```python
        edge = [[0, 40, 90, 60, 0],
                [0, 40, 90, 60, 0],
                [0, 40, 90, 60, 0]]
        zeros = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]
        thin(edge, edge, zeros)
        ```

        Dat moet `[[0, 0, 0, 0, 0], [0, 0, 90, 0, 0], [0, 0, 0, 0, 0]]` geven; de bovenste en de onderste rij zijn de buitenste rijen, die altijd 0 blijven. Hij probeert ook andere randen.
        """),
        L.task("Laat `thin(strength, gx, gy)` alleen de pixels houden die minstens zo sterk zijn als hun twee buren dwars op de rand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_thin():
    box_thin = L.code_box("thin", lines=24)
    box_thin
    return (box_thin,)


@app.cell(hide_code=True)
def check_thin(box_thin):
    result_thin = L.check("thin", box_thin.value)
    result_thin
    return (result_thin,)


@app.cell(hide_code=True)
def hints_thin():
    L.hints(
        "thin",
        [
            "Twee buren: één een stap `(dr, dc)` vooruit, `strength[r + dr, c + dc]`, en één een stap terug, `strength[r - dr, c - dc]`.",
            "Minstens zo sterk als allebei de buren betekent: minstens zo sterk als de grootste van de twee, `max(...)`. Is `strength[r, c]` dat (`>=`), dan krijgt `out[r, c]` de waarde `strength[r, c]`; anders blijft hij 0.",
            "`if strength[r, c] >= max(strength[r + dr, c + dc], ___):` en daaronder `out[r, c] = strength[r, c]`.",
        ],
        why="In het voorbeeld is de stap `(0, 1)`: de buren links en rechts. 40 heeft 0 en 90 naast zich, en is zwakker dan 90; 90 heeft 40 en 60, en is minstens zo sterk als allebei; 60 heeft 90 en 0. Dus alleen 90 blijft over: de top van de heuvel.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Oefening 3. De lage drempel

    In het uitgewerkte voorbeeld was de lage drempel 100, en brak de omtrek van de limoen af. Met de slider kies je de lage drempel; de hoge blijft 200. De course heeft de tekening zelf gemaakt, dus hij weet waar elke omtrek ligt: waar een vrucht het tafelkleed of een andere vrucht raakt. Een pixel telt als fout als hij rand is zonder omtrek in zijn venster van 3 bij 3 (te veel), of als hij op een omtrek ligt zonder rand in zijn venster (gemist): het venster geeft de dunne lijnen één pixel speling. Links de randen bij de gekozen lage drempel; rechts hoeveel pixels er fout zijn bij elke lage drempel. De as stopt bij 1200, zodat het laagste punt te zien is: een driehoekje bovenaan betekent dat de waarde hoger is.
    """)
    return


@app.cell(hide_code=True)
def explore_low_cell():
    def _frame(low):
        _html, _f = C.low_frame(low, {"edges": "randen, lage drempel {low}", "low": "lage drempel", "wrong": "pixels fout",
                                      "alt": "Links de randen van de tekening van fruit met een lage drempel van {low} en een hoge drempel van 200, als zwarte lijnen op wit; rechts hoeveel pixels er fout zijn bij de lage drempels 0 tot 200, op een as tot 1200: de lage drempels 0 tot 40 liggen erboven, van 4263 tot 1694, aangegeven met driehoekjes bovenaan; vanaf 50 zakt de lijn naar een laagste punt bij 70, en stijgt dan tot 1119 bij 200. Lage drempel {low} is aangegeven met een grotere stip, een driehoekje bovenaan als zijn waarde boven 1200 ligt."})
        return (_html,
                f'Lage drempel {low}: {_f["wrong"]} pixels zijn fout, {_f["too_many"]} te veel en {_f["missed"]} gemist.')

    explore_low = L.explore("low", "lage drempel", _frame, start=200, values=C.LOWS, role="hyper",
                            task="Zoek de lage drempel waarbij de minste pixels fout zijn.")
    explore_low
    return (explore_low,)


@app.cell(hide_code=True)
def check_low(explore_low):
    result_low = L.check("low", explore_low)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "low",
        [
            "Schuif de slider stap voor stap, vanaf 200 naar links, en lees de getallen in de zin onder het plaatje.",
            "Naar links komen zwakke stukken omtrek erbij, en worden er minder pixels gemist. Te ver, en de randen van de ruiten die een omtrek raken komen er ook bij: zoek de stap waar het getal omkeert.",
            "Kijk naar de stappen 60, 70 en 80: welke geeft het kleinste getal?",
        ],
        why="Bij 60 zijn 150 pixels fout, bij 70 nog 107, bij 80 155. Boven 70 wordt er meer omtrek gemist; eronder komen er meer randen van de ruiten bij.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.md(r"""
    ### Oefening 4. Waarom twee drempels?
    """)
    return


@app.cell(hide_code=True)
def explain_two_cell():
    explain_two = L.explain(
        "two",
        question="Met één drempel op de uitgedunde randsterkte, in dezelfde stappen van 10, kom je op de tekening op zijn best uit op 307 foute pixels, bij 100. Met de hysteresedrempel van oefening 3 zijn er veel minder pixels fout. Waarom kan een hysteresedrempel een zwak stuk omtrek houden en de rand van een ruit weglaten, terwijl één drempel dat niet kan?",
        model_answer="**Waarom één drempel tekortschiet:** een drempel kijkt naar één pixel tegelijk, alleen naar zijn randsterkte. Sommige stukken omtrek zijn even zwak als de randen van de ruiten, of zwakker, en de glans van de lamp heeft sterke randen. Dus elke drempel verliest zwakke omtrek of houdt randen die geen omtrek zijn. **Waarom twee drempels werken:** een hysteresedrempel kijkt ook naar de buren. Een zwak stuk omtrek zit vast aan een sterk stuk van dezelfde omtrek, dus het wordt van daaruit bereikt en blijft. De meeste randen van de ruiten en van de glans zijn zwak en staan op zichzelf: ze zitten nergens vast aan iets boven de hoge drempel, dus ze vallen weg. Alleen zwakke randen die een omtrek raken, zoals de rand van een ruit vlak naast een vrucht, komen er nog bij, en de glans op de appel linksboven, die deels zelf sterk is.",
        points=[
            "je zegt dat één drempel alleen naar de randsterkte van elke pixel kijkt, en dat zwakke omtrek en andere randen even sterk kunnen zijn",
            "je zegt dat een zwakke pixel alleen telt als hij vastzit aan een sterke rand, en dat een zwak stuk omtrek vastzit aan een sterk stuk van dezelfde omtrek",
            "je zegt dat de randen van de ruiten niet aan een sterke rand vastzitten, dus dat ze wegvallen",
        ],
    )
    explain_two
    return (explain_two,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Canny brengt de vorige lessen samen: het Gauss-filter van [Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter](/course/open/cv-denoise-filters?lang=nl), de beeldgradient van [Waar de helderheid verspringt: gradients en Sobel](/course/open/cv-sobel-edges?lang=nl), en het binaire beeld van [Zwart of wit: een drempel en de methode van Otsu](/course/open/cv-otsu-threshold?lang=nl), nu met twee drempels in plaats van één. De to-dolijst van de hysteresedrempel volgt pixels die aan elkaar vastzitten, één buur tegelijk; latere lessen tellen op dezelfde manier stukken die aan elkaar vastzitten. De volgende les, *Vormen laten groeien en krimpen: erosie en dilatatie*, werkt op binaire beelden zoals deze randen: hij laat de vormen in een binair beeld groeien of krimpen, om stipjes weg te halen of gaten te dichten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_neighbors, explain_two, result_low, result_thin):
    L.wrap_up(choice_neighbors, result_thin, result_low, explain_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
