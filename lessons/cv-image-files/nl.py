# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-image-files. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-image-files", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) en [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_channel():
    quick_channel = L.choice(
        "quick-channel",
        question="`img` is een kleurenbeeld met shape `(400, 600, 3)`. Wat is `img[:, :, 2]`?",
        options={
            "row": "Rij 2 van de afbeelding, met al zijn pixels en kleuren",
            "blue": "Het blauw van elke pixel: een 2D-array met shape `(400, 600)`",
            "green": "Het groen van elke pixel: een 2D-array met shape `(400, 600)`",
        },
        correct="blue",
        why={
            "row": "Een rij is het eerste deel van de index. Hier is het derde deel 2: welke richting is dat? Kijk in de tabel met nieuwe woorden van [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
            "blue": "Het derde deel van de index kiest de kleur: rood is 0, groen 1 en blauw 2.",
            "green": "De kleuren gaan rood, groen, blauw, en de telling begint bij 0. Welke is dan 2? Kijk in de tabel met nieuwe woorden van [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
        },
    )
    quick_channel
    return (quick_channel,)


@app.cell(hide_code=True)
def quick_check_byte():
    quick_byte = L.choice(
        "quick-byte",
        question="Hoeveel verschillende waarden kan één byte hebben?",
        options={
            "255": "255",
            "256": "256",
            "big": "1.000",
        },
        correct="256",
        why={
            "255": "Een byte gaat van 0 tot en met 255. Tel ze: telt 0 ook mee? Kijk in de tabel met nieuwe woorden van [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "256": "Een byte is een geheel getal van 0 tot en met 255: met 0 erbij zijn dat 256 waarden.",
            "big": "Een byte gaat tot en met 255 en niet verder: daarom heeft UTF-8 meer dan één byte nodig voor `€`, 8364. Hoeveel gehele getallen zijn er van 0 tot en met 255? Kijk in de tabel met nieuwe woorden van [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
        },
    )
    quick_byte
    return (quick_byte,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een camera maakt van licht getallen: een telefoonfoto van 12 megapixel heeft er 36 miljoen, elk één byte, en toch neemt het bestand maar een paar miljoen bytes in. Het bestandsformaat bewaart de getallen in minder bytes. Sommige formaten geven elk getal terug, maar JPEG, het gangbaarste formaat voor foto's, geeft getallen terug die er dichtbij zitten en vaak niet dezelfde zijn. Een model leert van de getallen in het bestand, niet van wat de camera zag, en veel datasets met beelden zijn JPEG.

    **Een voorbeeld uit het dagelijks leven.** Een opname van een college bewaart elk woord. Je aantekeningen bewaren wat ertoe doet in veel minder ruimte, maar de precieze woorden haal je er niet meer uit.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "size",
        question="De koffiefoto van de course is 400 pixels hoog en 600 breed: 400 × 600 × 3 = 720.000 getallen, elk één byte. Hoe groot is het bestand ongeveer als JPEG met kwaliteit 75?",
        options={
            "same": "Ongeveer 720.000 bytes: één byte voor elk getal",
            "half": "Ongeveer 360.000 bytes",
            "small": "Ongeveer 42.000 bytes",
        },
        reveal="**Ongeveer 42.000 bytes**, 17 keer minder dan de getallen zelf. Als PNG-bestand, dat elk getal teruggeeft, is het ongeveer 449.000 bytes: 62% van de getallen. JPEG wordt zoveel kleiner omdat het de getallen zelf niet bewaart, alleen een benadering.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Achter de lens van een camera zit een sensor met miljoenen piepkleine lichtmetertjes in een rooster, elk achter een rood, een groen of een blauw filter. Uit hoeveel licht erop viel, maakt de camera drie getallen per pixel: gehele getallen van 0 tot en met 255, elk één byte. Een afbeeldingsbestand is niet die array zelf, maar een manier om hem te bewaren. Een PNG-bestand comprimeert ze zonder iets kwijt te raken: het schrijft patronen, zoals vaak hetzelfde getal achter elkaar, in minder bytes, en geeft elk getal precies terug. Een JPEG-bestand laat details weg die je oog nauwelijks ziet, zoals fijne verschillen in kleur, en bewaart een benadering: de meeste getallen komen een beetje anders terug, en het bestand is vaak tien keer kleiner of meer. Met `quality`, van 0 tot en met 95, kies je zelf: hoe lager, hoe kleiner het bestand, en hoe verder de getallen ernaast zitten. Kwaliteit 100 werkt ook, maar maakt het bestand twee keer zo groot als 95, en geeft nog steeds niet elk getal terug.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **beeldresolutie** | hoeveel pixels een afbeelding heeft: zijn breedte en zijn hoogte, zoals 600 bij 400 | `img.shape` geeft eerst de hoogte |
    | **uint8** | het dtype van een geheel getal van 0 tot en met 255, opgeslagen in één byte: de getallen van een foto | `img.dtype`, `np.uint8` |
    | **lossless compressie** | data opslaan in minder bytes, zodat je elk getal precies terugkrijgt | PNG |
    | **lossy compressie** | data opslaan in veel minder bytes, door een benadering te bewaren | JPEG |
    | **PNG** | een bestandsformaat voor afbeeldingen met lossless compressie | `"coffee.png"` |
    | **JPEG** | een bestandsformaat voor foto's met lossy compressie; `quality` zegt hoe dicht het erbij blijft | `"coffee.jpg"` |
    | **`Image.open(path)`** | opent een afbeeldingsbestand met Pillow; `np.array` maakt er een array van | `np.array(Image.open(path))` |
    | **`Image.fromarray(img).save(path)`** | slaat een array van uint8 op als afbeeldingsbestand; het eind van het pad kiest het formaat | `.save(path, quality=75)` |
    | **`os.path.getsize(path)`** | de grootte van een bestand, in bytes | `os.path.getsize(path)` |
    | **`.astype(int)`** | dezelfde getallen met een ander dtype, hier gewone gehele getallen | `img.astype(int)` |

    Pillow is de library die Python gebruikt voor afbeeldingsbestanden. Je importeert hem als `from PIL import Image`: PIL heet de library waaruit hij is voortgekomen. Pillow geeft de getallen van een afbeelding als uint8, van 0 tot en met 255, ook bij een PNG-bestand, waar `plt.imread` floats van 0 tot 1 gaf. Een JPEG-bestand zonder `quality` krijgt 75.
    """)
    return


@app.cell(hide_code=True)
def picture_cup():
    mo.vstack([
        mo.md("**De rand van het kopje, op drie manieren**"),
        mo.Html(C.picture({"png": "PNG", "q75": "JPEG, kwaliteit 75", "q10": "JPEG, kwaliteit 10",
                           "size": "{n} bytes · {d} ernaast", "sep": ".", "dec": ",",
                           "alt": "Dezelfde hoek van 32 bij 32 pixels van de koffiefoto, vergroot, in drie versies: de witte rand van het kopje tegen het rode schoteltje. Als PNG en als JPEG met kwaliteit 75 zien ze er hetzelfde uit; met kwaliteit 10 valt de hoek uiteen in blokken met uitgelopen kleuren."})),
        mo.md(r"""
        Dezelfde 32 bij 32 pixels van de koffiefoto, vergroot, na opslaan en weer inlezen. Onder elk: de grootte van het hele bestand, en hoe ver een getal van de foto er gemiddeld naast zit. Het PNG-bestand geeft elk getal terug. Met kwaliteit 75 zitten de getallen er een beetje naast, en zie je geen verschil. Met kwaliteit 10 is het bestand nog kleiner, en zie je hoe JPEG werkt: het deelt de afbeelding op in blokken, van 8 bij 8 pixels voor de helderheid en 16 bij 16 voor de kleur, en bewaart elk blok als ruwe benadering.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Getallen met het dtype uint8 blijven tussen 0 en 255, ook als je ze van elkaar aftrekt. Als `a` en `b` uint8 zijn, geeft een getal 10 in `a` en 12 in `b` in `a - b` het getal 254, niet −2. Maak van allebei eerst gewone gehele getallen: `a.astype(int) - b.astype(int)`."),
        L.watch_out("Pillow noemt de grootte van een afbeelding andersom: `Image.open(path).size` is `(breedte, hoogte)`, `(600, 400)` voor de koffiefoto, terwijl de shape van de array `(400, 600, 3)` is, met de rijen eerst."),
        mo.md(r"""
        ### Recept

        Zo zie je wat een bestandsformaat met een afbeelding doet:

        1. Open het bestand en maak er een array van: `img = np.array(Image.open(path))`; zijn dtype is uint8.
        2. Kijk naar zijn resolutie in `img.shape`: de rijen (zijn hoogte), de kolommen (zijn breedte) en de kleuren.
        3. Tel de bytes van de getallen: rijen × kolommen × kleuren, elk één byte.
        4. Sla hem op, als PNG of als JPEG met een `quality`: `Image.fromarray(img).save(path)`; `os.path.getsize(path)` geeft de bytes van het bestand.
        5. Lees hem terug en vergelijk, als gewone gehele getallen: `diff = abs(back.astype(int) - img.astype(int))`. `diff.mean()` zegt hoe ver een getal er gemiddeld naast zit: de mean absolute error, de MAE van [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl). `(diff > 0).mean()` zegt welk deel van de getallen veranderde.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De koffiefoto: `data/photos/coffee.png`, een foto zonder auteursrecht uit de voorbeeldbeelden van scikit-image. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. De bestanden komen in je eigen werkmap. Voer de drie cellen uit.
    """)
    return


@app.cell
def example_read():
    import os
    import numpy as np
    from PIL import Image

    # 1. open de foto en maak er een array van
    img = np.array(Image.open("data/photos/coffee.png"))
    # 2. zijn resolutie en dtype
    print(img.shape, img.dtype)
    # 3. de bytes van zijn getallen, elk één byte
    print(img.shape[0] * img.shape[1] * img.shape[2])
    return Image, img, np, os


@app.cell
def example_save(Image, img, os):
    # 4. sla hem op als PNG en als JPEG, in je werkmap, en kijk hoe groot elk bestand is
    png = C.work_file("coffee.png")
    jpg = C.work_file("coffee.jpg")
    Image.fromarray(img).save(png)
    Image.fromarray(img).save(jpg, quality=75)
    print(os.path.getsize(png), os.path.getsize(jpg))
    return jpg, png


@app.cell
def example_compare(Image, img, jpg, np, png):
    # 5. lees ze allebei terug, en vergelijk ze met de getallen waarmee je begon
    for _path in (png, jpg):
        _back = np.array(Image.open(_path))
        _diff = abs(_back.astype(int) - img.astype(int))
        print(_diff.mean(), (_diff > 0).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De foto heeft shape `(400, 600, 3)` en dtype `uint8`: 400 rijen, 600 kolommen en 3 kleuren, 720.000 getallen van elk één byte. Het PNG-bestand neemt ongeveer 449.000 bytes in, 62% daarvan; het JPEG-bestand ongeveer 42.000, 6%. Het PNG-bestand geeft elk getal terug: geen enkel verschil. Het JPEG-bestand zit er gemiddeld 4,02 naast, op een schaal van 0 tot en met 255, en 0,877, oftewel 88%, van de getallen veranderde. Het plaatje hierboven laat zien dat je dat nauwelijks ziet. Het eigen bestand van de course, `data/photos/coffee.png`, is groter: 466.706 bytes. Het heeft dezelfde getallen, maar is met andere instellingen opgeslagen. Een PNG-bestand geeft altijd dezelfde getallen terug, maar hoe klein het wordt, hangt af van hoe het geschreven is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bytes():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De bytes van een foto
    """)
    return


@app.cell(hide_code=True)
def choice_bytes_cell():
    choice_bytes = L.choice(
        "raw-bytes",
        question="Een kleurenfoto is 4000 pixels breed en 3000 hoog, met getallen van het dtype uint8. Hoeveel bytes nemen zijn getallen in, zonder compressie?",
        options={
            "pixels": "12.000.000",
            "numbers": "36.000.000",
            "values": "9.180.000.000",
        },
        correct="numbers",
        why={
            "pixels": "Dat is het aantal pixels. Hoeveel getallen heeft een pixel in kleur? Kijk bij stap 3 van het recept.",
            "numbers": "4000 × 3000 pixels × 3 kleuren is 36 miljoen getallen, en uint8 neemt één byte per getal: 36 miljoen bytes.",
            "values": "Dat vermenigvuldigt met 255, de grootste waarde. Maar een getal neemt één byte in, wat zijn waarde ook is, van 0 tot en met 255: hoeveel getallen zijn er?",
        },
    )
    choice_bytes
    return (choice_bytes,)


@app.cell(hide_code=True)
def text_exercise_road():
    mo.md(r"""
    ### Oefening 2. Precieze getallen
    """)
    return


@app.cell(hide_code=True)
def choice_road_cell():
    choice_road = L.choice(
        "road-format",
        question="Een model leert wegen vinden op luchtfoto's. Naast elke foto staat een afbeelding van dezelfde grootte, met 1 voor elke pixel op een weg en 0 voor elke andere pixel. In welk formaat sla je die afbeeldingen op?",
        options={
            "png": "PNG",
            "jpeg95": "JPEG met kwaliteit 95: dan zit de benadering er heel dicht bij",
            "jpeg10": "JPEG met kwaliteit 10: dan zijn de bestanden het kleinst",
        },
        correct="png",
        why={
            "png": "Elk getal heeft hier een precieze betekenis: weg of geen weg. Alleen lossless compressie geeft elk getal terug. Een wegafbeelding van de course, 60 rijen van 80 pixels, neemt als PNG een paar honderd bytes in, voor zijn 4.800 getallen, en komt precies hetzelfde terug.",
            "jpeg95": "Dichtbij is niet hetzelfde. De course sloeg een wegafbeelding van 60 rijen van 80 pixels op als JPEG met kwaliteit 95, en 53 van de 4.800 pixels kwamen terug met het verkeerde getal: weg werd geen weg, of andersom. Welke soort compressie geeft elk getal terug?",
            "jpeg10": "De course sloeg een wegafbeelding van 60 rijen van 80 pixels op als JPEG met kwaliteit 10: elke pixel kwam terug als 0, dus de weg was weg. Welke soort compressie geeft elk getal terug?",
        },
    )
    choice_road
    return (choice_road,)


@app.cell(hide_code=True)
def text_exercise_difference():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Meet de benadering

        Maak `mean_difference(a, b)` af. Hij krijgt twee afbeeldingen met dezelfde shape, als uint8, en geeft terug hoe ver een getal van `b` gemiddeld van hetzelfde getal van `a` af ligt: hun MAE. Vervang de `...`. De check roept hem aan met een grijswaardenbeeld van 2 bij 2 pixels, `a`, en `b`, met getallen zoals een JPEG-bestand ze zou kunnen teruggeven:

        ```python
        a = [[10, 20], [30, 40]]
        b = [[12, 20], [27, 40]]
        ```

        Dat moet `1.25` geven: de verschillen zijn 2, 0, 3 en 0. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `mean_difference(a, b)` geven hoe ver een getal van `b` gemiddeld van hetzelfde getal van `a` af ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_difference():
    box_difference = L.code_box("mean-difference", lines=8)
    box_difference
    return (box_difference,)


@app.cell(hide_code=True)
def check_difference(box_difference):
    result_difference = L.check("mean-difference", box_difference.value)
    result_difference
    return (result_difference,)


@app.cell(hide_code=True)
def hints_difference():
    L.hints(
        "mean-difference",
        [
            "Stap 5 van het recept: maak van allebei gewone gehele getallen, trek ze af, neem de grootte van elk verschil, en het gemiddelde.",
            "Lees de eerste let-op boven het recept: wat geeft `a - b` als allebei uint8 zijn en een getal van `b` groter is?",
            "`diff = abs(a.astype(int) - ___)`, en dan `return diff.___()`.",
        ],
        why="`a.astype(int) - b.astype(int)` geeft −2, 0, 3 en 0, en `abs` maakt van elk zijn grootte: 2, 0, 3 en 0. Hun gemiddelde is 5 / 4 = 1,25. Zonder `.astype(int)` geeft 10 − 12 het getal 254.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_quality():
    mo.md(r"""
    ### Oefening 4. Zo klein als het kan

    De kattenfoto, `data/photos/chelsea.png`, 451 bij 300 pixels, als JPEG met de kwaliteit van de slider. Links een stuk ervan zoals het is, rond het oog; rechts hetzelfde stuk na opslaan als JPEG en weer inlezen. De zin eronder gaat over de hele foto.
    """)
    return


@app.cell(hide_code=True)
def explore_quality_cell():
    def _frame(q):
        _html, _f = C.quality_frame(q, {"png": "PNG", "jpeg": "JPEG, kwaliteit {q}",
                                        "alt": "Het oog van de kat, vergroot, zoals het is en na JPEG met kwaliteit {q}."})
        return (_html,
                f'Kwaliteit {q}: het bestand is {C.grouped(_f["size"], ".")} bytes, {L.number(100 * _f["share"], 0)}% van het PNG-bestand. Een getal zit er gemiddeld <span class="c-loss ea-num">{L.number(_f["diff"])}</span> naast, en {L.number(100 * _f["changed"], 0)}% van de getallen veranderde.')

    explore_quality = L.explore("quality", "kwaliteit", _frame, start=95, values=C.QUALITIES, role="",
                                task="Zoek de laagste kwaliteit waarbij een getal van de kat er gemiddeld minder dan 3 naast zit.")
    explore_quality
    return (explore_quality,)


@app.cell(hide_code=True)
def check_quality(explore_quality):
    result_quality = L.check("quality", explore_quality)
    result_quality
    return (result_quality,)


@app.cell(hide_code=True)
def hints_quality():
    L.hints(
        "quality",
        [
            "Schuif de slider stap voor stap omlaag, en lees hoe ver een getal er gemiddeld naast zit.",
            "3 of meer is te ver. Wat is de laagste stap waarbij een getal er nog minder dan 3 naast zit?",
            "Hoe lager de kwaliteit, hoe verder ernaast: geeft een stap 3 of meer, dan doet elke stap daaronder dat ook. Ga dus omlaag tot je over de 3 gaat, en neem de stap net daarvoor.",
        ],
        why="Bij kwaliteit 75 zit een getal er gemiddeld 2,85 naast; bij 50 is dat 3,65. Het bestand met kwaliteit 75 is ongeveer 21.000 bytes, een twintigste van de 405.900 getallen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) las je een afbeelding als array; nu weet je wat het bestand ertussen met de getallen doet. In [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl) bevatte een byte een geheel getal van 0 tot en met 255; hier is elk getal van een foto één byte. En de MAE uit [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl), daar de fout van een model, meet hier hoe ver een bestand van de foto af ligt. Later in deze route krijgt een pretrained model foto's die als JPEG zijn opgeslagen, en een afbeelding waarin elk getal iets precies betekent, zoals weg of geen weg, sla je altijd op als PNG. In de volgende les, *Een kleur anders opschrijven: HSV*, schrijf je de drie getallen van een pixel op een andere manier.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bytes, choice_road, result_difference, result_quality):
    L.wrap_up(choice_bytes, choice_road, result_difference, result_quality)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
