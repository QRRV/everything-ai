# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-text-as-numbers. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-text-as-numbers", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tekst: strings](/course/open/py-strings?lang=nl) en [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="Wat geeft `len(\"koud ijs\")`?",
        options={
            "letters": "`7`",
            "characters": "`8`",
            "words": "`2`",
        },
        correct="characters",
        why={
            "letters": "Dan tel je alleen de letters. Is een spatie ook een teken? Zie de tabel met nieuwe woorden in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "words": "Dat is het aantal woorden. Wat telt `len` in een string? Zie de tabel met nieuwe woorden in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "characters": "Zeven letters en een spatie: acht tekens. Een spatie telt ook mee.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="Wat geeft `[w[0] for w in [\"ijs\", \"thee\"]]`?",
        options={
            "text": "`\"it\"`",
            "words": "`[\"ijs\", \"thee\"]`",
            "firsts": "`[\"i\", \"t\"]`",
        },
        correct="firsts",
        why={
            "text": "Een comprehension tussen vierkante haken maakt een lijst, geen tekst. Wat komt erin, voor elke `w`? Zie het recept in [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).",
            "words": "`w[0]` is niet het hele woord. Wat geeft een index? Zie de tabel met nieuwe woorden in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "firsts": "`w[0]` is het eerste teken van elk woord, en elk daarvan komt in de lijst.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een computer bewaart alleen getallen, en een model rekent alleen met getallen. Voordat een language model een zin kan lezen, moet die zin dus getallen worden. Dat is de eerste stap van elke chatbot: de tekst die je typt, wordt een lijst gehele getallen, en het antwoord komt eruit als getallen die weer tekst worden.

    **Een voorbeeld uit het dagelijks leven.** Het geheimschrift van school: a = 1, b = 2, c = 3, enzovoort. Je schrijft een woord als een rij getallen, en een vriend met dezelfde tabel maakt er weer het woord van.
    """)
    return


@app.cell(hide_code=True)
def predict_codes():
    guess_codes = L.guess(
        "abc-numbers",
        question="Python heeft een getal voor elk teken. Wat denk je dat `[ord(ch) for ch in \"abc\"]` geeft?",
        options={
            "one": "`[1, 2, 3]`",
            "zero": "`[0, 1, 2]`",
            "unicode": "`[97, 98, 99]`",
        },
        reveal="**`[97, 98, 99]`.** `ord` geeft het nummer van een teken in Unicode. De kleine letters beginnen bij 97, omdat de getallen daarvoor al aan andere tekens gegeven waren: de spatie is 32, de cijfers zijn 48 tot en met 57, en de hoofdletters 65 tot en met 90. De letters volgen wel op elkaar: a, b en c zijn 97, 98 en 99.",
    )
    guess_codes
    return (guess_codes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## Het idee

    Unicode is een wereldwijde lijst die elk teken een eigen nummer geeft: elke letter, elk cijfer, elk symbool en elke spatie, in elk schrift, ook emoji. `ord` geeft het nummer van een teken, en `chr` geeft het teken terug. `for` gaat teken voor teken door een tekst, dus zo wordt een tekst een lijst getallen, en een lijst getallen weer tekst. Om een tekst in een bestand op te slaan, is nog één stap nodig: een character encoding maakt van elk teken bytes. Een byte gaat maar tot 255, en veel code points zijn groter: `€` is 8364. Daarom verdeelt UTF-8, de encoding die bijna iedereen gebruikt, een teken over één tot vier bytes: één voor een code point onder de 128, zoals een gewone letter; twee onder de 2048, zoals `é`; drie onder de 65536, zoals `€`; en vier voor de rest, zoals emoji.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **`for` over een tekst** | gaat één voor één door de tekens | `for ch in "hey":` geeft `"h"`, `"e"`, `"y"` |
    | **Unicode** | de wereldwijde lijst die elk teken een eigen nummer geeft, in elk schrift, ook emoji | |
    | **code point** | het nummer dat Unicode aan één teken geeft | `97` voor `"a"` |
    | **`ord`** | geeft het code point van één teken | `ord("a")` geeft `97` |
    | **`chr`** | andersom: geeft het teken van een code point | `chr(97)` geeft `"a"` |
    | **byte** | een geheel getal van 0 tot en met 255; een bestand, en het geheugen van een computer, bevatten bytes | |
    | **character encoding** | een afspraak over hoe tekens als bytes worden opgeslagen | |
    | **UTF-8** | de character encoding die bijna iedereen gebruikt: één byte voor een code point onder de 128, twee tot vier voor grotere | `"utf-8"` |
    | **`.encode`** | maakt van een tekst bytes, met de encoding die je opgeeft | `"café".encode("utf-8")` |
    | **`.decode`** | maakt van bytes weer tekst, met de encoding die je opgeeft | `data.decode("utf-8")` |
    | **`"".join`** | plakt een lijst teksten aan elkaar tot één tekst; de tekst vóór `.join` komt tussen de delen, en bij `""` is dat niets | `"".join(["h", "i"])` geeft `"hi"` |

    `.encode` geeft een waarde van het type `bytes`. Python laat die zien als `b'...'`, met codes zoals `\xc3` voor de bytes boven 127; `list(...)` ervan geeft de bytes als gewone getallen.
    """)])
    return


@app.cell(hide_code=True)
def picture_cafe():
    mo.vstack([
        draw.text_numbers("café", ("teken", "code point", "bytes in UTF-8"), title="Eén woord, drie keer"),
        mo.md(r"""
        Het woord `café` drie keer. Elk teken heeft één code point: `é` heeft 233. In UTF-8 kosten `c`, `a` en `f` elk één byte, met hetzelfde getal als hun code point, maar `é` kost er twee: 195 en 169. `café` is dus 4 tekens en 5 bytes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Lees je bytes met een andere encoding dan waarmee ze geschreven zijn, dan krijg je onleesbare tekst: `\"café\".encode(\"utf-8\").decode(\"latin-1\")` geeft `cafÃ©`, omdat `latin-1`, een oudere encoding, elk van de twee bytes van `é` als een eigen teken leest. Geef daarom de encoding op als je een tekstbestand opent: `open(path, encoding=\"utf-8\")`."),
        mo.md(r"""
        ### Recept

        Om een tekst in getallen om te zetten, en terug:

        1. Neem het code point van elk teken: `codes = [ord(ch) for ch in text]`.
        2. Maak van de code points weer tekst: een teken voor elk getal, aan elkaar geplakt met `"".join([chr(n) for n in codes])`.
        3. Om de tekst op te slaan, maak je er bytes van: `data = text.encode("utf-8")`; `list(data)` laat ze zien als getallen.
        4. Lees de bytes terug met dezelfde encoding: `data.decode("utf-8")`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tekst `café €3`, als code points en als bytes. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_cafe():
    # 1. het code point van elk teken
    _text = "café €3"
    _codes = [ord(ch) for ch in _text]
    print(_codes)
    # 2. terug naar tekst: een teken voor elk code point, aan elkaar geplakt
    _chars = [chr(n) for n in _codes]
    print(_chars)
    print("".join(_chars))
    # 3. de bytes in UTF-8: meer bytes dan tekens
    _data = _text.encode("utf-8")
    print(len(_text), len(_data))
    print(list(_data))
    # 4. en terug uit de bytes
    print(_data.decode("utf-8"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([mo.md(r"""
    Zeven tekens, zeven code points: `é` is 233, de spatie 32, `€` 8364, en het cijfer `3` is 51, niet 3. `chr` maakt van elk getal weer zijn teken, en `"".join` plakt ze aan elkaar tot dezelfde tekst. In UTF-8 kost de tekst 10 bytes: `é` kost er twee, 195 en 169, en `€` drie, 226, 130 en 172. `.decode("utf-8")` leest ze terug als dezelfde tekst.
    """)])
    return


@app.cell(hide_code=True)
def text_exercise_bytes():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel bytes?
    """)
    return


@app.cell(hide_code=True)
def choice_bytes_cell():
    choice_bytes = L.choice(
        "bytes",
        question="`len(\"naïef\")` is `5`: vijf tekens. Wat geeft `len(\"naïef\".encode(\"utf-8\"))`?",
        options={
            "six": "`6`",
            "five": "`5`",
            "ten": "`10`",
        },
        correct="six",
        why={
            "five": "Dat is het aantal tekens. Kost elk teken één byte? Kijk naar `é` in de figuur, en naar de regel voor UTF-8 in *Het idee*.",
            "ten": "Niet elk teken kost twee bytes. Welke tekens kosten er maar één, in de figuur?",
            "six": "`n`, `a`, `e` en `f` zijn gewone letters, elk één byte. `ï` heeft code point 239: onder de 2048 maar niet onder de 128, dus kost hij er twee, net als `é` in de figuur. 4 + 2 = 6.",
        },
    )
    choice_bytes
    return (choice_bytes,)


@app.cell(hide_code=True)
def text_exercise_codes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een woord als getallen

        Maak `codes`: een lijst met het code point van elk teken van `word`, in dezelfde volgorde. Vervang de `...`. De laatste regel is `codes`, zodat de check het ziet. De check probeert ook een ander woord.
        """),
        L.task("Maak `codes`: het code point van elk teken van `word`, in een lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_codes():
    box_codes = L.code_box("codes", lines=5)
    box_codes
    return (box_codes,)


@app.cell(hide_code=True)
def check_codes(box_codes):
    result_codes = L.check("codes", box_codes.value)
    result_codes
    return (result_codes,)


@app.cell(hide_code=True)
def hints_codes():
    L.hints(
        "codes",
        [
            "Een tekst is een rij tekens, en `ord` krijgt één teken. Ga dus één voor één door de tekens.",
            "Stap 1 van het recept: een comprehension, tussen vierkante haken, met `for ch in word`.",
            "`codes = [___ for ch in word]`: wat staat er vóór de `for`, voor elke `ch`?",
        ],
        why="`for ch in word` geeft de tekens `l`, `a`, `m` en `p` één voor één, en `ord(ch)` geeft van elk het code point: 108, 97, 109 en 112.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Getallen terug naar tekst

        Maak `decode(codes)` af. Hij krijgt een lijst code points, en geeft de tekst terug die ze vormen. Vervang de `...`. De check roept `decode([104, 105, 33])` aan: dat moet `"hi!"` geven. Hij probeert ook andere lijsten, waarvan één leeg.
        """),
        L.task("Laat `decode(codes)` de tekst van de code points in `codes` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=6)
    box_decode
    return (box_decode,)


@app.cell(hide_code=True)
def check_decode(box_decode):
    result_decode = L.check("decode", box_decode.value)
    result_decode
    return (result_decode,)


@app.cell(hide_code=True)
def hints_decode():
    L.hints(
        "decode",
        [
            "Stap 2 van het recept: eerst een teken voor elk getal, dan één tekst van die tekens.",
            "`chr` maakt van één getal zijn teken, dus een comprehension met `chr` geeft een lijst tekens. `\"\".join` maakt één tekst van zo'n lijst.",
            "`chars = [___ for n in codes]`, en dan `return ___`.",
        ],
        why="`[chr(n) for n in codes]` geeft een lijst met het teken van elk getal, en `\"\".join(chars)` plakt ze aan elkaar zonder iets ertussen. Een lege lijst geeft een lege tekst, `\"\"`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tekst: strings](/course/open/py-strings?lang=nl) was een tekst een rij tekens, elk op een eigen plek; nu heeft elk teken ook een getal. In [Een tekstbestand lezen](/course/open/py-files?lang=nl) opende je een bestand: daarin staan bytes, en `open(path, encoding="utf-8")` zegt hoe je ze als tekst leest. Later in de course leest een language model tekst als een lijst getallen zoals deze, maar met een eigen tabel, net als het geheimschrift: elk teken, of elk stukje van een woord, krijgt een getal uit die tabel. In de volgende les, *Tabellen met pandas*, lees je een hele tabel met data uit een bestand.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bytes, result_codes, result_decode):
    L.wrap_up(choice_bytes, result_codes, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
