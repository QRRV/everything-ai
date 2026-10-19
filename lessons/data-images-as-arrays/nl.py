# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-images-as-arrays. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-images-as-arrays", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Delen van een array kiezen](/course/open/np-indexing?lang=nl) en [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Wat geeft `m[1, 0:2]`?",
        options={
            "column": "`array([2, 5])`",
            "part": "`array([4, 5])`",
            "block": "`array([[1, 2], [4, 5]])`",
        },
        correct="part",
        why={
            "column": "Dat is kolom 1, uit beide rijen. Hier kiest het eerste getal een rij. Zie de tabel met nieuwe woorden in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
            "part": "Rij 1 is `[4, 5, 6]`, en de slice `0:2` pakt daarvan kolom 0 en 1.",
            "block": "De `1` voor de komma kiest één rij, niet alle rijen. Zie het recept in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_show():
    quick_show = L.choice(
        "quick-show",
        question="Wat doet `plt.show()` op de laatste regel van een cel?",
        options={
            "show": "Hij laat de grafiek zien die de regels erboven tekenden",
            "new": "Hij tekent een nieuwe, lege grafiek",
            "save": "Hij bewaart de grafiek als bestand",
        },
        correct="show",
        why={
            "show": "De regels erboven tekenen de grafiek; `plt.show()` maakt hem af en laat hem onder de cel zien.",
            "new": "Kijk onder een cel die eindigt met `plt.show()`: is de grafiek daar leeg? Zie het recept in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "save": "Waar komt de grafiek terecht: in een bestand, of onder de cel? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
        },
    )
    quick_show
    return (quick_show,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model rekent alleen met getallen, dus een afbeelding moet eerst getallen worden voordat een model er iets mee kan. Voor een computer is een afbeelding een rooster van getallen: één getal per punt, of drie per punt in een kleurenafbeelding. Later in de course leert een neural network handgeschreven cijfers lezen, en het enige wat het krijgt, zijn 28 rijen van 28 getallen per cijfer. Een foto van een telefoon heeft miljoenen van zulke getallen.

    **Een voorbeeld uit het dagelijks leven.** Een borduurpatroon: een rooster van hokjes, met in elk hokje een nummer uit de garenlijst, dat zegt welke kleur je borduurt.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="Een klein grijswaardenbeeld van een handgeschreven cijfer heeft 28 rijen van 28 punten, en één getal per punt. Hoeveel getallen zijn dat?",
        options={
            "rows": "28",
            "sum": "56",
            "all": "784",
        },
        reveal="**784.** 28 rijen van 28 getallen is 28 keer 28, en dat zijn 784 getallen: één voor elk punt. Als array is zijn shape `(28, 28)`.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een afbeelding is een array. In een grijswaardenbeeld is elke pixel één getal: zijn helderheid. In een kleurenbeeld heeft elke pixel drie getallen: hoeveel rood, groen en blauw hij heeft. Een kleurenbeeld is dus een 3D-array, een array in drie richtingen: rijen, kolommen en de drie kleuren. Alles wat je van arrays weet, werkt erop: de shape, een index, slices en een mask.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **pixel** | één punt van een afbeelding; in de array één plek, die je vindt met zijn rij en zijn kolom | `img[5, 32]` |
    | **grijswaardenbeeld** | een 2D-array met één helderheid per pixel, van 0, zwart, tot 1, wit | shape `(rijen, kolommen)` |
    | **kleurenbeeld** | een 3D-array met drie getallen per pixel: rood, groen en blauw | shape `(rijen, kolommen, 3)` |
    | **channel** | een van de kleuren, voor elke pixel: een eigen 2D-array | `img[:, :, 0]` is het rode channel |
    | **`plt.imread(path)`** | leest een afbeeldingsbestand als array; een PNG-bestand geeft floats van 0 tot 1, een JPG-bestand gehele getallen van 0 tot en met 255 | `img = plt.imread(path)` |
    | **`plt.imshow(img)`** | tekent een array als afbeelding; voor één channel of een grijswaardenbeeld zet je er `cmap="gray"` bij, anders kleurt matplotlib het van paars tot geel | `plt.imshow(img)` |

    Een index in een kleurenbeeld kan drie delen hebben: de rij, de kolom en de kleur, met komma's ertussen. `:` pakt alles in die richting, zoals eerder, en een index met maar twee delen, zoals `img[0:5, 0:5]`, houdt alle drie de kleuren. `cmap="gray"` is een keyword argument, net als `axis=0` in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def picture_plus():
    _numbers, _image = C.picture({"array": "de array", "image": "de afbeelding",
                                  "alt": "Een zwart plusteken op wit, met grijze punten waar de armen de hoeken raken."})
    mo.vstack([
        mo.md("**Twee keer hetzelfde plusteken**"),
        mo.hstack([mo.Html(_numbers), mo.Html(_image)], justify="start", gap=2, align="center"),
        mo.md(r"""
        Links de getallen van een grijswaardenbeeld van 5 bij 5 pixels; rechts de afbeelding die die getallen maken met `plt.imshow(plus, cmap="gray")`. Elke 0 is een zwarte pixel, elke 1 een witte, en 0,5 is grijs. Rij 0 staat bovenaan, net als in een tabel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Rij 0 staat bovenaan in een afbeelding, en een index geeft eerst de rij: `img[rij, kolom]`. Dat is andersom dan `x, y` in een grafiek, waar x eerst komt en y omhoog gaat."),
        L.watch_out("Een *channel* is hier één laag getallen langs de laatste richting van een afbeelding, zoals al zijn roodwaarden. Het heeft niets met een tv-kanaal te maken."),
        mo.md(r"""
        ### Recept

        Zo werk je met een afbeelding als array:

        1. Lees hem: `img = plt.imread(path)`; een PNG-bestand geeft floats van 0 tot 1.
        2. Bekijk de shape: twee getallen voor een grijswaardenbeeld, drie voor een kleurenbeeld, het laatste 3.
        3. Laat hem zien: `plt.imshow(img)` en `plt.show()`; voor een grijswaardenbeeld of één channel zet je er `cmap="gray"` bij.
        4. Pak een deel met slices, de rijen eerst: `img[boven:onder, links:rechts]`; één channel met `img[:, :, c]`.
        5. Tel pixels met een mask, bijvoorbeeld op één channel: `len(red[red > 0.9])`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een klein landschap dat de course maakte: `lessons/data-images-as-arrays/landscape.png`, 24 pixels hoog en 40 breed. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de drie cellen uit.
    """)
    return


@app.cell
def example_read():
    import matplotlib.pyplot as plt

    # 1. lees het afbeeldingsbestand als array
    img = plt.imread("lessons/data-images-as-arrays/landscape.png")
    # 2. de shape en het dtype, en één pixel: rij 5, kolom 32, in de zon
    print(img.shape)
    print(img.dtype)
    print(img[5, 32])
    # 3. laat hem zien
    plt.imshow(img)
    plt.show()
    return img, plt


@app.cell
def example_crop(img, plt):
    # 4. een deel: rij 16 tot en met 23 en kolom 0 tot en met 11, de hoek linksonder
    _hills = img[16:24, 0:12]
    print(_hills.shape)
    plt.imshow(_hills)
    plt.show()
    return


@app.cell
def example_channel(img, plt):
    # 4. één channel: het rood van elke pixel, getoond als grijswaardenbeeld
    _red = img[:, :, 0]
    print(_red.shape)
    # 5. tel de pixels met veel rood, met een mask op het channel
    print(len(_red[_red > 0.9]))
    plt.imshow(_red, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De shape is `(24, 40, 3)`: 24 rijen, 40 kolommen en 3 kleuren; het dtype `float32` is een float die minder geheugen kost dan `float64`. De pixel in de zon is `[0.98039216 0.84313726 0.23529412]`: veel rood en groen, weinig blauw, en rood en groen licht samen maken geel. De hoek linksonder is een deel van 8 rijen en 12 kolommen, `(8, 12, 3)`, groter getoond: heuvels, met een strook lucht erboven. Het rode channel is een 2D-array, `(24, 40)`: daarin is de zon wit, omdat hij het meeste rood heeft, de lucht is grijs, en de groene heuvels, met het minste rood, zijn zwart. Bij één channel maakt `imshow` het laagste getal zwart en het hoogste wit. 44 pixels hebben meer rood dan 0,9: de pixels van de zon.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De shape van een foto
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "color-shape",
        question="Een kleurenfoto is 100 pixels hoog en 200 pixels breed. Wat is de shape van de array?",
        options={
            "wide": "`(200, 100, 3)`",
            "flat": "`(100, 200)`",
            "color": "`(100, 200, 3)`",
        },
        correct="color",
        why={
            "wide": "De rijen komen eerst: hoeveel rijen heeft een afbeelding die 100 pixels hoog is?",
            "flat": "Dat is de shape van een grijswaardenbeeld. Hoeveel getallen heeft een pixel in kleur? Zie de tabel met nieuwe woorden hierboven.",
            "color": "100 rijen, 200 kolommen, en 3 kleuren per pixel.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_sun():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Knip de zon uit

        Het vak leest het landschap als `img`, shape `(24, 40, 3)`. De zon staat in rij 2 tot en met 9 en kolom 29 tot en met 36. Maak `sun`: dat deel van de afbeelding, met alle drie de kleuren. Vervang de `...`. De laatste regel is `sun`, zodat de check het ziet.
        """),
        L.task("Maak `sun`: rij 2 tot en met 9 en kolom 29 tot en met 36 van `img`, met alle drie de kleuren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sun():
    box_sun = L.code_box("sun", lines=6)
    box_sun
    return (box_sun,)


@app.cell(hide_code=True)
def check_sun(box_sun):
    result_sun = L.check("sun", box_sun.value)
    result_sun
    return (result_sun,)


@app.cell(hide_code=True)
def hints_sun():
    L.hints(
        "sun",
        [
            "Stap 4 van het recept: een deel met slices, de rijen eerst, dan de kolommen.",
            "Een slice stopt net voor zijn tweede getal: voor rij 2 tot en met 9 is de slice `2:10`.",
            "`sun = img[2:10, ___:___]`.",
        ],
        why="`img[2:10, 29:37]` pakt rij 2 tot en met 9 en kolom 29 tot en met 36. Zonder derde deel houdt de index alle drie de kleuren: shape `(8, 8, 3)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_red():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoeveel rode pixels?

        Maak `red_pixels(rgb, limit)` af. `rgb` is een kleurenbeeld, gegeven als lijst van rijen met pixels, en de functie maakt er al een array van, `img`. Hij geeft terug hoeveel pixels meer rood hebben dan `limit`. Vervang de `...`. De check roept de functie aan met deze afbeelding van 2 rijen van 3 pixels, en `limit` 0,5:

        ```python
        [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.8], [0.8, 0.2, 0.0]],
         [[1.0, 0.5, 0.0], [1.0, 0.2, 0.2], [0.2, 0.2, 0.5]]]
        ```

        Dat moet `3` geven. Hij probeert ook andere afbeeldingen en grenzen.
        """),
        L.task("Laat `red_pixels(rgb, limit)` de pixels tellen met meer rood dan `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_red():
    box_red = L.code_box("red-pixels", lines=8)
    box_red
    return (box_red,)


@app.cell(hide_code=True)
def check_red(box_red):
    result_red = L.check("red-pixels", box_red.value)
    result_red
    return (result_red,)


@app.cell(hide_code=True)
def hints_red():
    L.hints(
        "red-pixels",
        [
            "Twee stappen: neem eerst het rode channel, en tel dan zijn pixels met een mask, zoals in stap 5 van het recept.",
            "Rood is de eerste kleur, dus zijn channel heeft `0` als derde deel van de index. Het mask vergelijkt dat channel met `limit`.",
            "`red = img[___]`, en dan `return len(red[red ___ limit])`.",
        ],
        why="`img[:, :, 0]` is het rode channel: één getal per pixel. `red > limit` is `True` voor elke pixel met meer rood dan `limit`, `red[red > limit]` houdt die getallen over, en `len` telt ze: 0,8, 1,0 en 1,0, dus 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Delen van een array kiezen](/course/open/np-indexing?lang=nl) haalde je rijen, kolommen en delen uit een matrix, en koos je getallen met een mask; een afbeelding is zo'n array, en een deel ervan is een slice. In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je getallen als grafiek; `plt.imshow` tekent ze als afbeelding. Later in de course leert een neural network handgeschreven cijfers lezen uit arrays zoals deze. In de volgende les, *Tekst is een rij getallen*, maak je ook van tekst getallen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_red, result_sun):
    L.wrap_up(choice_shape, result_sun, result_red)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
