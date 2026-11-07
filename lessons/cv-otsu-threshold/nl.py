# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-otsu-threshold. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-otsu-threshold", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_equalize():
    quick_equalize = L.choice(
        "quick-equalize",
        question="In een foto in grijstinten hebben 30 van elke 100 pixels grijstint 90 of een donkerdere. Welke grijstint wordt 90 na histogramegalisatie?",
        options={
            "share": "Ongeveer 76",
            "same": "Nog steeds 90",
            "thirty": "30",
        },
        correct="share",
        why={
            "share": "Grijstint 90 krijgt 255 keer zijn aandeel in het cumulatieve histogram: 255 × 0,3 = 76,5, en `astype(np.uint8)` maakt daar 76 van.",
            "same": "Egalisatie geeft elke grijstint een nieuwe grijstint. Welk aandeel vermenigvuldigt ze met 255? Zie Het idee in [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl).",
            "thirty": "30 van elke 100 is het aandeel, 0,3. Met welk getal vermenigvuldigt egalisatie het aandeel? Zie [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl).",
        },
    )
    quick_equalize
    return (quick_equalize,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="De helft van de pixels van een groep heeft grijstint 60, de andere helft grijstint 70. Wat is de variantie van hun grijstinten?",
        options={
            "25": "25",
            "5": "5",
            "50": "50",
        },
        correct="25",
        why={
            "25": "Het gemiddelde is 65, en beide grijstinten liggen er 5 vandaan: 5² × 0,5 + 5² × 0,5 = 25.",
            "5": "5 is de afstand van elke grijstint tot het gemiddelde, 65, en dus de standaardafwijking. De variantie kwadrateert de afstanden eerst: zie Het idee in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "50": "Dat is 5² + 5², zonder de kansen. Elke grijstint heeft kans 0,5: zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om cellen te tellen onder de microscoop of de letters van een gescande pagina te lezen, moet een programma eerst weten welke pixels bij het voorwerp horen en welke bij de achtergrond. De eenvoudigste manier is één grijstint als scheidslijn, een drempel: de pixels aan de ene kant ervan zijn het voorwerp, de rest is achtergrond, en het resultaat is alleen nog zwart en wit. Maar het licht verschilt van foto tot foto, dus één vaste grijstint werkt niet voor allemaal. In 1979 publiceerde Nobuyuki Otsu een methode die de drempel afleidt uit het histogram van elke afbeelding; die methode wordt nog altijd veel gebruikt en zit ingebouwd in scikit-image en OpenCV.

    **Een voorbeeld uit het dagelijks leven.** Je verdeelt een berg wasgoed over een donkere en een lichte was. Bij een goede verdeling lijkt elk stuk zoveel mogelijk op de rest van zijn eigen stapel: geen bleek overhemd tussen de spijkerbroeken, geen grijze sok tussen de witte handdoeken. De methode van Otsu zoekt de grijstint die de pixels zo verdeelt.
    """)
    return


@app.cell(hide_code=True)
def predict_mean():
    guess_mean = L.guess(
        "mean",
        question="De foto van deze les is een cel in zout water, onder de microscoop: een lichte schijf die ongeveer 3 van elke 100 pixels beslaat, op een grijze achtergrond. Een eerste idee voor een drempel is de gemiddelde grijstint van de foto. Hoeveel van de pixels worden wit met `img > img.mean()`?",
        options={
            "cell": "Ongeveer 3 van elke 100: alleen de cel",
            "half": "Ongeveer de helft",
            "all": "Bijna allemaal",
        },
        reveal="**Ongeveer de helft**: 48 van elke 100. De achtergrond is 97 van elke 100 pixels, dus het gemiddelde, ongeveer 68, ligt midden in de grijstinten van de achtergrond zelf, en de lichtere helft van de achtergrond wordt samen met de cel wit. Een goede drempel ligt tussen de twee groepen grijstinten, niet midden in de grootste.",
    )
    guess_mean
    return (guess_mean,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een puntbewerking geeft elke pixel een nieuwe grijstint die alleen van zijn eigen grijstint afhangt: niet van zijn buren, en niet van waar hij staat. Histogramegalisatie was er een: een tabel met een nieuwe grijstint voor elke grijstint. Drempelen is een van de eenvoudigste puntbewerkingen: elke pixel die lichter is dan de drempel `t` wordt wit, 255, en elke andere pixel zwart, 0. Wat eruit komt, is een binair beeld, met maar twee waarden. In NumPy geeft `img > t` True voor elke pixel die lichter is dan `t` en False voor de rest, in de shape van de afbeelding. Python telt True als 1 en False als 0, dus `255 * (img > t)` geeft 255 en 0, en `(img > t).mean()`, het gemiddelde van enen en nullen, is het aandeel van de pixels die lichter zijn dan `t`.

        De methode van Otsu kiest `t` zelf. Elke drempel verdeelt de pixels over twee groepen: de donkere pixels, met grijstint `t` of een donkerdere, en de lichte pixels, lichter dan `t`. Een goede drempel zet elke pixel dicht bij het gemiddelde van zijn eigen groep. Otsu meet dat met de variantie van elke groep, gewogen met zijn aandeel van de pixels, $w$, en opgeteld:
        """),
        L.formula(r"w_\text{dark} \cdot \mathrm{Var}_\text{dark} + w_\text{light} \cdot \mathrm{Var}_\text{light}"),
        mo.md(r"""
        Dat is de variantie binnen de groepen: voor elke pixel de gekwadrateerde afstand tot het gemiddelde van zijn eigen groep, gemiddeld over alle pixels. De aandelen wegen de twee varianties zoals kansen de waarden wegen in een verwachtingswaarde: een grote groep telt zwaar mee, een kleine groep weinig. Otsu probeert elke drempel, en neemt die waarbij de variantie binnen de groepen het kleinst is. In de code staat *dark* voor donker en *light* voor licht.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **puntbewerking** | elke pixel krijgt een nieuwe grijstint die alleen van zijn eigen grijstint afhangt, niet van zijn buren of zijn plek | `table[img]` |
        | **drempelen** | elke pixel die lichter is dan de drempel `t` wordt wit, elke andere pixel zwart | `img > t` |
        | **binair beeld** | een afbeelding met maar twee waarden, zoals 0 en 255, of False en True | `255 * (img > t)` |
        | **variantie binnen de groepen** | de varianties van de donkere en van de lichte pixels, elk gewogen met het aandeel van de pixels in zijn groep, opgeteld | `w_dark * dark.std() ** 2 + w_light * light.std() ** 2` |
        | **methode van Otsu** | elke drempel proberen, en die nemen met de kleinste variantie binnen de groepen: daar liggen de pixels, over alle pixels genomen, het dichtst bij het gemiddelde van hun eigen groep | `thresholds[np.argmin(withins)]` |
        | $w_\text{dark}$ | het aandeel van de pixels in de donkere groep, van 0 tot en met 1 | `(img <= t).mean()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_thresholds():
    mo.vstack([
        mo.md("**De cel, met twee drempels**"),
        mo.Html(C.threshold_picture({"photo": "De foto", "by_mean": "Het gemiddelde als drempel, 68", "by_otsu": "De drempel van Otsu, 122",
                                     "histogram": "Histogram, afgekapt bij 300 pixels", "gray": "grijstint",
                                     "mean_line": "gemiddelde, 68", "otsu_line": "Otsu, 122",
                                     "alt": "Bovenaan drie plaatjes. De foto: een lichte schijf, de cel, op een grijze achtergrond met vage strepen. Het binaire beeld met de gemiddelde grijstint als drempel: de cel is wit, en ongeveer de helft van de achtergrond ook, in grillige strepen en vlekjes. Het binaire beeld met de drempel van Otsu, 122: alleen de cel is wit, een strakke schijf op zwart. Daaronder het histogram van de foto: heel hoge staven van ongeveer grijstint 10 tot 80, de achtergrond, bovenaan afgekapt; lage staven van ongeveer 85 tot 120; dan een brede heuvel van staven van ongeveer 125 tot 235, de cel. Een gestreepte lijn markeert het gemiddelde, 68, midden in de staven van de achtergrond; een doorgetrokken lijn markeert de drempel van Otsu, 122, in het lage deel tussen de twee groepen."})),
        mo.md(r"""
        Bovenaan de foto en twee binaire beelden. Met de gemiddelde grijstint, ongeveer 68, als drempel wordt ongeveer de helft van de pixels wit: de cel, en de lichtere helft van de achtergrond, in strepen die de microscoop in de foto heeft achtergelaten. Met de drempel van Otsu, 122, is alleen de cel wit. Daaronder het histogram: de staven van de achtergrond, van ongeveer grijstint 10 tot 80, gaan ver boven de rand uit, tot 28.907 pixels voor één grijstint; door af te kappen bij 300 blijven de lage staven van de cel, van ongeveer 125 tot 235, in beeld. Het gemiddelde ligt midden in de achtergrond; de drempel van Otsu ligt in het lage deel tussen de twee groepen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`img > t` geeft True en False, geen grijstinten. Met `vmin=0, vmax=255`, zoals je bij histogramegalisatie deed, telt `plt.imshow` True als 1, bijna zwart, dus de hele afbeelding ziet er zwart uit. Maak er eerst grijstinten van: `255 * (img > t)`."),
        L.watch_out("De methode van Otsu geeft altijd een drempel, ook voor een afbeelding met maar één groep grijstinten, zoals een foto van alleen de achtergrond: hij splitst die ene groep dan in tweeën. En één drempel voor de hele afbeelding werkt niet als het licht ongelijkmatig valt: een donkere hoek van een pagina kan donkerder zijn dan de letters elders."),
        mo.md(r"""
        ### Recept

        Om de drempel van Otsu van een grijswaardenbeeld `img` te vinden:

        1. Maak de lijst met drempels, van de donkerste grijstint van de afbeelding tot de lichtste, de lichtste niet meegerekend: `thresholds = list(range(img.min(), img.max()))`. Dan hebben beide groepen altijd pixels. Loop erdoorheen met `for t in thresholds:`.
        2. Verdeel de pixels: `dark = img[img <= t]` en `light = img[img > t]`.
        3. Reken het aandeel van elke groep uit, `w_dark = (img <= t).mean()` en `w_light = (img > t).mean()`, en dan de variantie binnen de groepen: `w_dark * dark.std() ** 2 + w_light * light.std() ** 2`. De variantie is het kwadraat van de standaardafwijking. Voeg hem toe aan een lijst, `withins`.
        4. Neem de drempel met de kleinste variantie binnen de groepen: `best = thresholds[np.argmin(withins)]`.
        5. Maak het binaire beeld met die drempel: `255 * (img > best)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De foto is `data/photos/cell.png`: een cel in zout water, opgenomen met een microscoop die meet hoeveel elk punt het licht vertraagt. Licht gaat langzamer door de cel dan door het water, dus de cel is lichter. De foto heeft één grijstint per pixel. De eerste codecel leest hem in en probeert de gemiddelde grijstint als drempel; de tweede volgt het recept en meet de cel op. Voer de twee codecellen uit.
    """)
    return


@app.cell
def example_mean():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/cell.png"))
    print(img.shape, img.min(), img.max())
    # een eerste poging: de gemiddelde grijstint als drempel, en het aandeel witte pixels
    print(img.mean(), (img > img.mean()).mean())
    return img, np, plt


@app.cell
def example_otsu(img, np, plt):
    # 1. elke drempel van de donkerste grijstint tot de lichtste
    thresholds = list(range(img.min(), img.max()))
    withins = []
    for t in thresholds:
        # 2. de donkere en de lichte pixels
        dark = img[img <= t]
        light = img[img > t]
        # 3. het aandeel van elke groep, en de variantie binnen de groepen
        w_dark = (img <= t).mean()
        w_light = (img > t).mean()
        withins.append(w_dark * dark.std() ** 2 + w_light * light.std() ** 2)
    # 4. de drempel met de kleinste variantie binnen de groepen
    best = thresholds[np.argmin(withins)]
    print(best, min(withins), img.std() ** 2)
    # het aandeel en de variantie van elke groep bij die drempel
    print((img <= best).mean(), img[img <= best].std() ** 2)
    print((img > best).mean(), img[img > best].std() ** 2)
    # 5. het binaire beeld, en hoeveel kolommen een witte pixel hebben: de breedte van de cel
    cell = 255 * (img > best)
    print((cell.max(axis=0) == 255).sum())
    plt.imshow(cell, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De foto is 550 pixels breed en 660 hoog, met grijstinten van 0 tot en met 255. Zijn gemiddelde grijstint is ongeveer 68, en met die drempel wordt 48% van de pixels wit. De methode van Otsu probeert de 255 drempels van 0 tot en met 254 en vindt 122, met een variantie binnen de groepen van ongeveer 152, tegen ongeveer 571 voor alle pixels samen. De twee groepen zijn niet allebei compact: de achtergrond, 96,8% van de pixels, heeft variantie 135,5, maar de cel, 3,2%, heeft 640,2, meer dan de hele foto, omdat de grijstinten van de cel van ongeveer 125 tot 235 lopen. Gewogen met hun aandelen is 0,968 × 135,5 + 0,032 × 640,2 ongeveer 152: de grote groep telt het zwaarst. Met 122 is alleen de cel wit, één strakke schijf. Die is 124 pixels breed, en bij deze microscoop is één pixel 0,107 micrometer breed, dus de cel is ongeveer 13 micrometer breed: 13 duizendste van een millimeter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_point():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een puntbewerking
    """)
    return


@app.cell(hide_code=True)
def choice_point_cell():
    choice_point = L.choice(
        "point",
        question="Welke van deze drie bewerkingen van een grijswaardenbeeld is een puntbewerking?",
        options={
            "negative": "Elke grijstint g wordt 255 − g: `255 - img`",
            "blur": "Elke pixel krijgt de gemiddelde grijstint van zichzelf en zijn acht buren",
            "flip": "De afbeelding wordt van links naar rechts gespiegeld: `img[:, ::-1]`",
        },
        correct="negative",
        why={
            "negative": "De nieuwe grijstint van een pixel hangt alleen van zijn eigen grijstint af: 200 wordt 55, waar die pixel ook staat. Zwart wordt wit en wit wordt zwart: een negatief. Drempelen en histogramegalisatie werken net zo.",
            "blur": "Hier hangt de nieuwe grijstint ook van de buren af: dezelfde grijstint kan lichter of donkerder uitvallen, afhankelijk van wat eromheen ligt. Naar welke pixels kijkt een puntbewerking?",
            "flip": "Elke pixel krijgt de grijstint van een andere pixel, aan de andere kant: zijn eigen grijstint zegt niets over zijn nieuwe. Naar welke pixel kijkt een puntbewerking?",
        },
    )
    choice_point
    return (choice_point,)


@app.cell(hide_code=True)
def text_exercise_binary():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een binair beeld

        Maak `binary(img, t)` af. Die krijgt een grijswaardenbeeld en een drempel, en geeft het binaire beeld terug: 255 voor elke pixel die lichter is dan `t`, en 0 voor elke andere pixel. Vervang de `...`. De check roept hem aan met een afbeelding van 6 pixels en drempel 100:

        ```python
        binary([[30, 120, 200], [90, 160, 60]], 100)
        ```

        Dat moet `[[0, 255, 255], [0, 255, 0]]` geven. Hij probeert ook andere afbeeldingen en drempels.
        """),
        L.task("Laat `binary(img, t)` 255 geven voor elke pixel die lichter is dan `t`, en 0 voor de rest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_binary():
    box_binary = L.code_box("binary", lines=6)
    box_binary
    return (box_binary,)


@app.cell(hide_code=True)
def check_binary(box_binary):
    result_binary = L.check("binary", box_binary.value)
    result_binary
    return (result_binary,)


@app.cell(hide_code=True)
def hints_binary():
    L.hints(
        "binary",
        [
            "Eén regel is genoeg: vergelijk de afbeelding met `t`, en maak dan getallen van wat eruit komt.",
            "`img > t` geeft True voor elke pixel die lichter is dan `t`. Python telt True als 1: waarmee vermenigvuldig je het om 255 te krijgen?",
            "`return ___ * (img > t)`.",
        ],
        why="`img > 100` geeft `[[False, True, True], [False, True, False]]`, en 255 keer dat is `[[0, 255, 255], [0, 255, 0]]`. Een pixel van precies 100 zou 0 blijven: die is niet lichter dan 100.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lowest():
    mo.md(r"""
    ### Oefening 3. Het laagste punt

    Met de slider kies je de drempel voor de foto van de cel. Links het binaire beeld met die drempel; rechts de variantie binnen de groepen van elke drempel van 0 tot en met 254, met de gekozen drempel als stip. Onder in het dal is de kromme bijna vlak, dus lees het getal af in de zin onder het plaatje.
    """)
    return


@app.cell(hide_code=True)
def explore_lowest_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"binary": "drempel {t}", "threshold": "drempel", "within": "variantie binnen de groepen",
                                          "alt": "Links het binaire beeld van de cel met drempel {t}; rechts de variantie binnen de groepen van elke drempel, een kromme die aan beide kanten hoog is, met een kleine kuil bij ongeveer 55 en een breed, vlak dal tussen ongeveer 80 en 145, en drempel {t} gemarkeerd met een stip."})
        return (_html,
                f'Drempel {t}: de variantie binnen de groepen is {L.number(_f["within"], 1)}, en {L.number(100 * _f["white"], 1)}% van de pixels is wit.')

    explore_lowest = L.explore("lowest", "drempel", _frame, start=66, values=C.THRESHOLDS, role="param",
                               task="Zoek de drempel waarbij de variantie binnen de groepen het kleinst is: de drempel van Otsu.")
    explore_lowest
    return (explore_lowest,)


@app.cell(hide_code=True)
def check_lowest(explore_lowest):
    result_lowest = L.check("lowest", explore_lowest)
    result_lowest
    return (result_lowest,)


@app.cell(hide_code=True)
def hints_lowest():
    L.hints(
        "lowest",
        [
            "Schuif de slider één stand tegelijk, en lees de variantie binnen de groepen af in de zin onder het plaatje.",
            "Vanaf 66 gaat de variantie binnen de groepen omlaag, en ergens in het dal gaat hij weer omhoog: zoek de stand waar hij omkeert.",
            "Kijk naar de standen 114, 122 en 130: welke heeft de kleinste variantie binnen de groepen?",
        ],
        why="Bij 114 is de variantie binnen de groepen 152,2, bij 122 is hij 151,8, en bij 130 152,5. Dus 122 is de drempel van Otsu, dezelfde die het uitgewerkte voorbeeld vond door elke grijstint te proberen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_otsu():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. De methode van Otsu

        Maak `otsu(img)` af. Die krijgt een grijswaardenbeeld en geeft de drempel van Otsu terug. De regels rond de `...` proberen al elke drempel van de donkerste grijstint tot de lichtste, en nemen aan het eind de drempel met het kleinste getal in `withins`. Vervang de `...` door wat er voor elke drempel `t` gebeurt: de donkere en de lichte pixels, hun aandelen, en de variantie binnen de groepen, toegevoegd aan de lijst met `withins.append(...)`. De check roept hem aan met een afbeelding van 20 pixels, een achtergrond met grijstinten 2 tot en met 5 met daarin een voorwerp met grijstinten 6 tot en met 9:

        ```python
        otsu([[2, 3, 3, 4, 3], [3, 8, 9, 5, 2], [4, 9, 8, 6, 3], [3, 7, 3, 2, 4]])
        ```

        Dat moet `5` geven. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `otsu(img)` de drempel geven met de kleinste variantie binnen de groepen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_otsu():
    box_otsu = L.code_box("otsu", lines=14)
    box_otsu
    return (box_otsu,)


@app.cell(hide_code=True)
def check_otsu(box_otsu):
    result_otsu = L.check("otsu", box_otsu.value)
    result_otsu
    return (result_otsu,)


@app.cell(hide_code=True)
def hints_otsu():
    L.hints(
        "otsu",
        [
            "Volg de stappen 2 en 3 van het recept, binnen de loop: de twee groepen, hun twee aandelen, en dan `withins.append(...)`.",
            "`dark = img[img <= t]` en `light = img[img > t]`. Het aandeel van de donkere pixels is `(img <= t).mean()`. De variantie van een groep is het kwadraat van zijn standaardafwijking: `dark.std() ** 2`.",
            "`withins.append(w_dark * dark.std() ** 2 + w_light * ___)`.",
        ],
        why="Bij drempel 5 heeft de donkere groep 14 van de 20 pixels, met variantie 0,69, en de lichte groep de 6 pixels met grijstinten 6 tot en met 9, met variantie 1,14: 14/20 × 0,69 + 6/20 × 1,14 is ongeveer 0,83. Bij drempel 4 is de variantie binnen de groepen 0,99, en bij 6 is hij 1,01, dus 5 heeft de kleinste.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl) gaf een tabel elke grijstint een nieuwe grijstint: een puntbewerking, net als drempelen. In [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) wogen kansen de gekwadrateerde afstanden; hier wegen de aandelen van de pixels de varianties van de twee groepen. De methode van Otsu kiest één getal door een maat voor hoe slecht een verdeling is zo klein mogelijk te maken. In [Bergaf rollen: gradient descent](/course/open/ml-gradient-descent-1d?lang=nl) maakte een model zijn loss stap voor stap kleiner, langs de helling; Otsu hoeft maar één getal te kiezen, uit 255, dus hij kan ze gewoon allemaal proberen. Binaire beelden komen later in deze module terug: *Vormen laten groeien en krimpen: erosie en dilatatie* ruimt ze op, en *Vormen tellen: connected components* telt de witte vormen erin, zoals cellen. In de volgende les, *Minder ruis: het gemiddelde-, het Gauss- en het mediaanfilter*, hangt de nieuwe grijstint van een pixel ook van zijn buren af: dat is geen puntbewerking meer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_point, result_binary, result_lowest, result_otsu):
    L.wrap_up(choice_point, result_binary, result_lowest, result_otsu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
