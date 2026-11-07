# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-denoise-filters. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-denoise-filters", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl) en [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_difference():
    quick_difference = L.choice(
        "quick-difference",
        question="`a` en `b` zijn twee grijswaardenbeelden van uint8. Waarom meet `abs(a.astype(int) - b.astype(int)).mean()` hoe ver ze uit elkaar liggen, en `abs(a - b).mean()` niet?",
        options={
            "wrap": "uint8 kan niet onder 0: 3 − 5 loopt rond naar 254",
            "abs": "`abs` werkt alleen op gewone gehele getallen",
            "mean": "`.mean()` van uint8 geeft een geheel getal, dus de decimalen gaan verloren",
        },
        correct="wrap",
        why={
            "wrap": "uint8 bevat alleen de gehele getallen 0 tot en met 255, dus 3 − 5 wordt 254 in plaats van −2, en `abs` kan dat niet herstellen. Als gewone gehele getallen is het verschil −2, en `abs` maakt er 2 van.",
            "abs": "`abs` werkt op een array met getallen van elke soort. Wat gebeurt er met 3 − 5 als beide uint8 zijn? Zie het recept van [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "mean": "`.mean()` van uint8 geeft een float met decimalen. Het probleem zit eerder, in het aftrekken: wat is 3 − 5 in uint8? Zie [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
        },
    )
    quick_difference
    return (quick_difference,)


@app.cell(hide_code=True)
def quick_check_feature_map():
    quick_feature_map = L.choice(
        "quick-feature-map",
        question="Een filter van 3 bij 3 schuift over een afbeelding van 6 bij 6 pixels, en blijft binnen de afbeelding. Hoe groot is de feature map?",
        options={
            "4": "4 bij 4",
            "6": "6 bij 6",
            "3": "3 bij 3",
        },
        correct="4",
        why={
            "4": "Het filter past op 6 − 3 + 1 = 4 plekken in de breedte, en op 4 in de hoogte.",
            "6": "Dan zou het filter ook op de buitenste pixels liggen, waar een deel ervan over de rand steekt. Op hoeveel plekken past het in de breedte? Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
            "3": "Dat is de grootte van het filter. Op hoeveel plekken past het in de breedte, binnen 6 pixels? Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
        },
    )
    quick_feature_map
    return (quick_feature_map,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elke foto heeft wat ruis, en bij weinig licht zie je het: de sensor vangt zo weinig licht op dat het toeval voor een groot deel bepaalt hoeveel elke pixel krijgt, dus elke pixel valt iets lichter of donkerder uit dan de werkelijkheid. Een kapotte pixel in een sensor, of een fout bij het versturen of opslaan van de getallen, geeft spikkels van puur zwart of wit. Die ruis zit je verderop in deze module in de weg: een programma dat randen zoekt, ziet elke korrel als een piepkleine rand, en een drempel maakt van spikkels vormen die er niet zijn. Daarom filteren veel beeldprogramma's eerst de ruis weg, en ze proberen te behouden wat ertoe doet: de randen.

    **Een voorbeeld uit het dagelijks leven.** Vijf mensen lezen van de andere kant van een kamer een thermometer af die 19,5 graden aanwijst, en roepen 19, 20, 18, 21 en 20. Hun gemiddelde, 19,6, ligt dichter bij 19,5 dan elke losse aflezing: de een leest iets te hoog af, de ander iets te laag, en in het gemiddelde valt dat tegen elkaar weg. Roept een zesde voor de grap 90, dan springt het gemiddelde van de zes naar ongeveer 31; hun mediaan blijft 20. Twee van de filters van deze les werken als het gemiddelde, en één als de mediaan.
    """)
    return


@app.cell(hide_code=True)
def predict_smear():
    guess_smear = L.guess(
        "smear",
        question="De foto van deze les is de koffiekop van [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl), in grijstinten. Eén kopie krijgt spikkels: ongeveer 1 op de 10 pixels wordt puur zwart of puur wit. Een gemiddeldefilter geeft elke pixel de gemiddelde grijstint van het venster van 3 bij 3 pixels eromheen. Brengt dat de foto met spikkels dichter bij de schone?",
        options={
            "much": "Veel dichterbij: de spikkels worden weggemiddeld",
            "hardly": "Nauwelijks dichterbij",
            "further": "Verder weg",
        },
        reveal="**Nauwelijks dichterbij**: vóór het filter zit de foto gemiddeld 12,6 grijstinten ernaast, erna 12,0. Het gemiddeldefilter smeert een spikkel uit: een witte spikkel in een vlak met grijstint 170 telt (255 − 170) / 9, ongeveer 9, op bij elk van de negen vensters waar hij in zit. Het felle puntje is weg, maar nu zitten negen pixels elk ongeveer 9 ernaast, samen net zoveel als die ene spikkel. Het mediaanfilter, verderop in deze les, brengt het terug tot 2,3.",
    )
    guess_smear
    return (guess_smear,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Beeldruis is een willekeurige verandering van de grijstinten: elke pixel laat zien wat de werkelijkheid gaf, plus of min een beetje. Deze les maakt met opzet twee soorten, zodat hij kan meten wat een filter weghaalt. *Korrel*: elke pixel krijgt er een willekeurig getal bij, met gemiddelde 0 en standaardafwijking 25. Zulke willekeurige getallen hebben de klokvormige spreiding van de normale verdeling: de meeste zijn klein, een paar zijn groot. De wiskundige Carl Friedrich Gauss werkte veel met die klokvorm; het Gauss-filter hieronder is naar hem genoemd, en de latere les *Veelgebruikte verdelingen* behandelt de normale verdeling zelf. Sommige grijstinten komen dan onder 0 of boven 255 uit, wat een afbeeldingsbestand niet kan opslaan; de les laat ze zo, en de plaatjes tonen ze als zwart en wit. *Spikkels*: een paar pixels worden puur zwart, 0, of puur wit, 255. (Later in de course voegen diffusion models precies zulke korrel met opzet toe aan afbeeldingen, en leren ze die weer weg te halen.)

        Een filter werkt op één klein venster tegelijk, zoals het filter van [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl): het venster van 3 bij 3 pixels dat begint bij rij `r` en kolom `c` geeft de pixel `out[r, c]`. In het midden van dat venster ligt de pixel `img[r + 1, c + 1]`: `out[r, c]` is de nieuwe grijstint van die pixel. Het **gemiddeldefilter** neemt het gemiddelde van de negen grijstinten. Dat is een convolutie met een filter van negen weights van 1/9: het venster keer het filter, opgeteld, is het gemiddelde. Korrel maakt de ene pixel lichter en de volgende donkerder, dus in het gemiddelde van negen pixels valt veel ervan tegen elkaar weg. Maar bij een rand bevat het venster donkere en lichte pixels samen, en hun gemiddelde is een grijstint ertussenin: de rand vervaagt.

        Het **Gauss-filter** is ook een convolutie, met weights die in het midden het grootst zijn en langs een klokvorm aflopen:
        """),
        L.formula(r"\frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}"),
        mo.md(r"""
        De pixel in het midden telt 4 keer mee, de pixels erboven, eronder, links en rechts ervan 2 keer, de hoeken één keer. Gedeeld door 16 zijn de negen weights samen 1. Omdat de buitenste pixels minder meetellen, vervaagt het Gauss-filter minder dan het gemiddeldefilter, en middelt het ook iets minder korrel weg.

        Het **mediaanfilter** neemt de mediaan van de negen grijstinten: zet ze op volgorde, en neem de middelste, de 5e. Dat is geen convolutie: geen enkel stel weights keer het venster geeft de mediaan. Een spikkel is één afwijkende waarde van de negen, en de mediaan trekt zich er niets van aan, zoals de mediaan van [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) zich niets aantrok van een uitschieter. Bij een rand ligt het grootste deel van het venster aan één kant, dus de mediaan is een grijstint van die kant, en de rand blijft scherp.
        """),
        L.watch_out("Filter betekent hier meer dan in [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl): daar was het het rooster van weights. In beeldbewerking heet elke regel die een nieuwe pixel maakt uit het venster eromheen een filter. Het gemiddelde- en het Gauss-filter zijn zo'n rooster van weights; het mediaanfilter heeft helemaal geen weights."),
        mo.md(r"""
        De drie filters op een rand van 0 naar 255, in elke rij van de afbeelding:

        | | de grijstinten over de rand |
        |---|---|
        | de rand | 0, 0, 0, 255, 255, 255 |
        | na het gemiddeldefilter | 0, 85, 170, 255 |
        | na het Gauss-filter | 0, 64, 191, 255 |
        | na het mediaanfilter | 0, 0, 255, 255 |

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **beeldruis** | een willekeurige verandering van de grijstinten: elke pixel iets lichter of donkerder dan de werkelijkheid, of spikkels van puur zwart en wit | `rng.normal(0, 25, size=clean.shape)` |
        | **gemiddeldefilter** | elke pixel wordt het gemiddelde van het venster eromheen: een convolutie met gelijke weights | `window.mean()` |
        | **Gauss-filter** | elke pixel wordt een gewogen gemiddelde van het venster eromheen, met weights die samen 1 zijn en langs een klokvorm aflopen | `(window * K).sum()` |
        | **mediaanfilter** | elke pixel wordt de mediaan van het venster eromheen; het haalt spikkels weg en houdt randen scherp | `np.median(window)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_filters():
    mo.vstack([
        mo.md("**Korrel en spikkels, en drie filters**"),
        mo.Html(C.filter_picture({"grainy": "Korrel", "speckled": "Spikkels", "mean": "Gemiddeldefilter", "gauss": "Gauss-filter", "median": "Mediaanfilter",
                                  "alt": "Acht uitvergrotingen van hetzelfde stuk van de foto: linksboven de koffie, de witte rand van de kop, de donkere buitenkant van de kop eronder, de steel van het lepeltje en rechts het grijze schoteltje. Bovenste rij: de foto met korrel, een fijne zanderige ruis over alles; na het gemiddeldefilter gladder, maar zachter aan de randen; na het Gauss-filter iets minder glad en iets scherper; na het mediaanfilter nog steeds vlekkerig, met vlekjes van een paar pixels in de vlakke delen. Onderste rij: de foto met spikkels, overal zwarte en witte puntjes; na het gemiddeldefilter zijn de puntjes kleine grijze vegen geworden; na het Gauss-filter kleinere en scherpere vegen; na het mediaanfilter zijn de puntjes weg en zijn de randen scherp."})),
        mo.md(r"""
        Elke keer hetzelfde stuk van de foto, 100 pixels hoog en 130 breed, uitvergroot: de koffie, de rand van de kop, de steel van het lepeltje en het schoteltje. Bovenaan met korrel: het gemiddelde- en het Gauss-filter maken het gladder, en de randen zachter; na het mediaanfilter blijft het vlekkerig. Onderaan met spikkels: het gemiddelde- en het Gauss-filter maken van elke spikkel een grijze veeg; het mediaanfilter haalt de spikkels weg en houdt de randen scherp.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een filter van 3 bij 3 maakt de afbeelding 2 pixels kleiner in elke richting, zoals in [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl): een venster past alleen waar een pixel aan elke kant een buur heeft. Vergelijk het resultaat dus met de schone foto zonder zijn buitenste pixels: `clean[1:-1, 1:-1]`, van de tweede rij tot de laatste, de laatste niet meegerekend, en net zo voor de kolommen. Bibliotheken zoals SciPy en OpenCV vullen de rand aan, zodat de afbeelding even groot blijft."),
        L.watch_out("Een filter kan ruis niet onderscheiden van fijn detail: de nerf van het hout van de tafel vervaagt samen met de ruis. Een groter venster haalt meer ruis weg, en ook meer detail; oefening 3 zoekt het evenwicht."),
        mo.md(r"""
        ### Recept

        Om een grijswaardenbeeld `img` te filteren met een venster van 3 bij 3:

        1. Maak een array met nullen voor het resultaat, 2 kleiner in elke richting: `out = np.zeros((len(img) - 2, len(img[0]) - 2))`.
        2. Ga langs elke plek, met `for r in range(len(out)):` en daarbinnen `for c in range(len(out[0])):`, en neem het venster dat daar begint: `window = img[r:r + 3, c:c + 3]`.
        3. Vul de plek in: voor het gemiddeldefilter `out[r, c] = window.mean()`, voor het Gauss-filter `out[r, c] = (window * K).sum()` met `K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16`, en voor het mediaanfilter `out[r, c] = np.median(window)`.
        4. Meet hoe ver het resultaat gemiddeld van de schone foto afligt: `abs(out - clean[1:-1, 1:-1]).mean()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De foto is de koffiekop van [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl): `data/photos/coffee.png`. Het voorbeeld maakt hem grijs door voor elke pixel de sterkste van zijn rood, groen en blauw te nemen: `img.max(axis=2)`, waarbij `axis=2` de derde richting van de array is, de kleuren. Het houdt het stuk met de kop, de rijen 0 tot 300 en de kolommen 100 tot 500, zodat de loops snel blijven. Dan maakt het de ruis met een generator van willekeurige getallen met een seed, zoals in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl): `rng.normal(0, 25, size=clean.shape)` geeft willekeurige getallen met gemiddelde 0 en standaardafwijking 25, één voor elke pixel. De eerste codecel maakt de korrel en de spikkels; de tweede filtert drie foto's, elk met de drie filters tegelijk: de schone, die met korrel en die met spikkels. Voer de twee codecellen uit; de tweede duurt een paar seconden.
    """)
    return


@app.cell
def example_noise():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    # de foto in grijstinten, de sterkste kleur van elke pixel, en het stuk met de kop
    clean = img.max(axis=2)[0:300, 100:500]
    rng = np.random.default_rng(0)
    # korrel: elke grijstint plus een willekeurig getal met gemiddelde 0 en standaardafwijking 25
    grainy = clean + rng.normal(0, 25, size=clean.shape)
    # spikkels: een willekeurig getal van 0 tot 1 voor elke pixel; onder 0.05 zwart, boven 0.95 wit
    specks = rng.random(size=clean.shape)
    speckled = np.array(clean)   # een nieuwe array, zodat clean schoon blijft
    speckled[specks < 0.05] = 0
    speckled[specks > 0.95] = 255
    # hoe ver elke foto met ruis gemiddeld van de schone afligt
    # grainy bestaat al uit floats, omdat er willekeurige floats bij kwamen; speckled is nog uint8, net als clean
    print(clean.shape, abs(grainy - clean).mean(), abs(speckled.astype(int) - clean.astype(int)).mean())
    return clean, grainy, np, plt, speckled


@app.cell
def example_filters(clean, grainy, np, plt, speckled):
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


    def three_filters(photo):
        # 1. drie arrays met nullen, 2 kleiner in elke richting
        mean = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        gauss = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        median = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        # 2. elk venster van 3 bij 3
        for r in range(len(mean)):
            for c in range(len(mean[0])):
                window = photo[r:r + 3, c:c + 3]
                # 3. het gemiddelde, de weights van het Gauss-filter, de mediaan
                mean[r, c] = window.mean()
                gauss[r, c] = (window * K).sum()
                median[r, c] = np.median(window)
        return mean, gauss, median


    # 4. hoe ver elk resultaat afligt van de schone foto zonder zijn buitenste pixels
    inner = clean[1:-1, 1:-1]
    for photo in [clean, grainy, speckled]:
        mean, gauss, median = three_filters(photo)
        print(abs(mean - inner).mean(), abs(gauss - inner).mean(), abs(median - inner).mean())
    # het mediaanfilter van de foto met spikkels, de laatste; met vmin=0 en vmax=255 is 0 zwart en 255 wit,
    # zoals in de foto, en zonder die twee zou de donkerste grijstint zwart worden
    plt.imshow(median, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het stuk met de kop is 300 pixels hoog en 400 breed. De korrel verschuift de grijstinten gemiddeld 20,0, minder dan zijn standaardafwijking van 25: de standaardafwijking kwadrateert de verschillen eerst, zodat de paar grote zwaarder wegen. De spikkels verschuiven ze 12,6. De drie regels daaronder zeggen hoe ver elke foto van de schone afligt na het gemiddelde-, het Gauss- en het mediaanfilter. De eerste regel is wat een filter kost: in de schone foto zit geen van onze ruis, en toch verschuift het gemiddeldefilter zijn grijstinten gemiddeld 2,9, het Gauss-filter 2,3 en het mediaanfilter 1,9. Ongeveer de helft daarvan is de vervaging bij de randen; de rest is fijn detail, zoals het hout van de tafel, en de lichte ruis die de foto zelf al had. Op de korrel komt het gemiddeldefilter op 7,7, het Gauss-filter op 8,2 en het mediaanfilter op 9,1: het gemiddelde van negen pixels middelt de meeste korrel weg, het Gauss-filter middelt er iets minder van weg maar vervaagt minder, en de mediaan gebruikt alleen de middelste van de negen grijstinten, dus er blijft meer korrel over. Op de spikkels helpen het gemiddelde- en het Gauss-filter nauwelijks, 12,0 en 11,9, terwijl het mediaanfilter op 2,3 komt: het laat de spikkels weg. Zijn resultaat is het plaatje onder de cel, bijna weer de schone foto.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De weights van het Gauss-filter
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="Het Gauss-filter van deze les heeft de weights 1, 2, 1 / 2, 4, 2 / 1, 2, 1, gedeeld door 16. Waarom 16?",
        options={
            "sum": "16 is de som van de weights: gedeeld door 16 zijn ze samen 1, dus een stuk dat helemaal één grijstint is, houdt die grijstint",
            "pixels": "16 is het aantal pixels waar het filter naar kijkt",
            "middle": "Door te delen door 16 telt de middelste pixel het zwaarst",
        },
        correct="sum",
        why={
            "sum": "1 + 2 + 1 + 2 + 4 + 2 + 1 + 2 + 1 = 16. Op een stuk van de foto waar alle negen pixels grijstint 100 hebben, geeft het filter 100 × 16 / 16 = 100: de grijstint blijft gelijk. De weights van het gemiddeldefilter, negen keer 1/9, zijn ook samen 1.",
            "pixels": "Het venster is 3 bij 3: hoeveel pixels zijn dat? Tel de negen weights op, en vergelijk met 16.",
            "middle": "De middelste pixel telt het zwaarst omdat zijn weight, 4, de grootste is; elke weight door hetzelfde getal delen verandert niet welke de grootste is. Wat zou een stuk waar elke pixel grijstint 100 heeft worden, als je niet deelt?",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het gemiddeldefilter

        Maak `mean_filter(img)` af. Die krijgt een grijswaardenbeeld en geeft de afbeelding na een gemiddeldefilter van 3 bij 3 terug, 2 kleiner in elke richting. De loops langs elke plek staan er al: vervang de `...` door wat er op elke plek `r, c` gebeurt. De check roept hem aan met een rand van 0 naar 9, vier rijen van vijf pixels:

        ```python
        mean_filter([[0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9]])
        ```

        Dat moet `[[0, 3, 6], [0, 3, 6]]` geven. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `mean_filter(img)` elke plek het gemiddelde van zijn venster van 3 bij 3 geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean", lines=10)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean",
        [
            "Eén regel is genoeg: neem het venster van 3 bij 3 dat begint bij rij `r` en kolom `c`, en sla zijn gemiddelde op in `out[r, c]`.",
            "Het venster heeft de rijen `r`, `r + 1` en `r + 2`: een slice vanaf `r` die net vóór `r + 3` stopt. De kolommen gaan net zo, na een komma tussen dezelfde haken. Een array heeft een methode die zijn gemiddelde geeft.",
            "`out[r, c] = img[r:r + 3, ___].mean()`.",
        ],
        why="Het eerste venster heeft alleen nullen: gemiddelde 0. Het tweede heeft in elke rij 0, 0, 9: het gemiddelde is 27 / 9 = 3. Het derde heeft 0, 9, 9 in elke rij: 54 / 9 = 6. Beide rijen van het resultaat zijn gelijk, omdat alle vier de rijen van de afbeelding gelijk zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ### Oefening 3. Hoe groot een venster?

    Een gemiddeldefilter kan een groter venster hebben: 5 bij 5 neemt het gemiddelde van 25 pixels. Met de slider kies je de grootte voor de foto met korrel van het uitgewerkte voorbeeld. Links hetzelfde stuk als in het plaatje, na het gemiddeldefilter van die grootte; rechts voor elke grootte hoe ver het resultaat gemiddeld van de schone foto afligt, met de gekozen grootte als stip.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(k):
        _html, _f = C.size_frame(k, {"window": "venster {k} bij {k}", "size": "grootte van het venster", "off": "gemiddeld aantal grijstinten naast de schone foto",
                                     "alt": "Links het stuk van de foto met korrel na een gemiddeldefilter van {k} bij {k}; rechts hoe ver het resultaat van de schone foto afligt voor de venstergroottes 1 tot en met 15: ongeveer 20 bij 1, steil omlaag naar een laagste punt tussen 3 en 9, dan langzaam weer omhoog, met grootte {k} gemarkeerd met een stip."})
        return (_html,
                f'Venster {k} bij {k}: de gefilterde foto zit gemiddeld {L.number(_f["off"], 1)} grijstinten naast de schone.')

    explore_size = L.explore("size", "venster", _frame, start=1, values=C.SIZES, role="hyper",
                             task="Zoek de grootte van het venster waarbij de gefilterde foto het dichtst bij de schone komt.")
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
            "Schuif de slider één stand tegelijk, en lees het getal af in de zin onder het plaatje.",
            "Vanaf 1 gaat het getal snel omlaag: een groter venster middelt meer korrel weg. Ergens gaat het weer omhoog, omdat de randen vervagen: zoek de stand waar het omkeert.",
            "Kijk naar de standen 3, 5 en 7: welke geeft het kleinste getal?",
        ],
        why="Met 3 bij 3 zit de foto 7,7 ernaast, met 5 bij 5 is het 6,5, en met 7 bij 7 is het 6,7. Dus 5 bij 5 komt het dichtst bij: een kleiner venster laat meer korrel over, een groter vervaagt meer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_median():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Het mediaanfilter

        Maak `median_filter(img)` af. Die krijgt een grijswaardenbeeld en geeft de afbeelding na een mediaanfilter van 3 bij 3 terug, 2 kleiner in elke richting. Deze keer staat alleen de array met nullen er: schrijf de loops langs elke plek zelf, zoals in oefening 2, en geef elke plek de mediaan van zijn venster. De check roept hem aan met een afbeelding met linksboven een witte spikkel en een rand tussen grijstint 10 en grijstint 200:

        ```python
        median_filter([[10, 10, 10, 200, 200],
                       [10, 255, 10, 200, 200],
                       [10, 10, 10, 200, 200],
                       [10, 10, 10, 200, 200]])
        ```

        Dat moet `[[10, 10, 200], [10, 10, 200]]` geven. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `median_filter(img)` elke plek de mediaan van zijn venster van 3 bij 3 geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_median():
    box_median = L.code_box("median", lines=10)
    box_median
    return (box_median,)


@app.cell(hide_code=True)
def check_median(box_median):
    result_median = L.check("median", box_median.value)
    result_median
    return (result_median,)


@app.cell(hide_code=True)
def hints_median():
    L.hints(
        "median",
        [
            "Volg stap 2 en 3 van het recept: twee loops, het venster, en de mediaan van het venster in `out[r, c]`.",
            "De loops zijn `for r in range(len(out)):` en, één niveau verder ingesprongen, `for c in range(len(out[0])):`. De mediaan van een array is `np.median(...)`.",
            "Binnen de twee loops: `out[r, c] = np.median(img[___, ___])`.",
        ],
        why="Het eerste venster bevat acht keer 10 en de spikkel, 255: op volgorde is de 5e grijstint 10, dus de spikkel is weg. Het derde venster bevat drie keer 10 en zes keer 200: de 5e is 200. De rand blijft één stap, van 10 naar 200. Het gemiddeldefilter zou in elke rij 37,2, 100,6 en 136,7 geven: van de spikkel een grijze veeg, en van de rand een helling.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het gemiddelde- en het Gauss-filter zijn convoluties, zoals in [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl); daar vond een filter een rand, hier strijkt een filter glad. Een CNN leert zijn filters uit data, en de eerste layer van een getraind CNN heeft vaak filters die op randfilters lijken, en op vlekken van één kleur. Het mediaanfilter is geen convolutie: geen enkel rooster van weights, met welke getallen ook, doet wat het doet. In [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl) mat je hoe ver JPEG de grijstinten verschuift; hier zegt dezelfde maat hoe ver ruis ze verschuift, en hoeveel daarvan een filter terugwint. De volgende les, *Waar de helderheid verspringt: gradients en Sobel*, zoekt randen, en ruis maakt dat lastig: elke korrel is een kleine rand. Daarom strijken programma's die randen zoeken de afbeelding vaak eerst glad.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_mean, result_size, result_median):
    L.wrap_up(choice_weights, result_mean, result_size, result_median)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
