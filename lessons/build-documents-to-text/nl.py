# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-documents-to-text. Route: course/route/lessen/bouwen.yaml. Checks, de bestanden en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-documents-to-text", "nl")
    C = L.checks   # de drie bestanden, de pdf-lezer van de course, wat de course mat, en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een tekstbestand lezen](/course/open/py-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_files():
    quick_files = L.choice(
        "quick-files",
        question="Het bestand `notes.txt` heeft drie regels. Wat staat er na deze code in `text`?\n\n```python\nwith open(\"notes.txt\") as f:\n    text = f.read()\n```",
        options={
            "lines": "Een lijst met een string voor elk van de drie regels",
            "string": "Eén string: het hele bestand",
            "first": "De eerste van de drie regels, als string",
        },
        correct="string",
        why={
            "string": "`f.read()` geeft het hele bestand als één string, met een newline waar elke regel eindigt. Daar begint elk bestand in deze les mee.",
            "lines": "`f.read()` maakt geen lijst: het geeft het hele bestand als één string, met de newlines erin. De regels één voor één langsgaan doet `for line in f:`. Zie [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "first": "`f.read()` leest het hele bestand, niet één regel: één string met alle drie de regels, en een newline waar elke regel eindigt. Zie [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
        },
    )
    quick_files
    return (quick_files,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Je eerste RAG](/course/open/build-rag-first?lang=nl) waren de punten die de zoekfunctie langsging al tekst. Wat een bedrijf weet, staat in bestanden: een handboek als pdf, pagina's op de website, notities in Markdown, een gescand contract. Een zoekfunctie gaat tekst langs en een model leest tekst, dus de eerste stap van elk RAG-systeem is de tekst uit elk bestand halen. Gaat die stap mis, dan waarschuwt niets je: de zoekfunctie vindt de cookiebanner, en het model leest zinnen die uit twee kolommen door elkaar komen.

    **Een voorbeeld uit het dagelijks leven.** Kopieer de tekst van een webpagina naar een mail, en het menu en *Alles accepteren* komen mee. Kopieer hem uit een pdf met twee kolommen, en je kunt een regel van de linkerkolom krijgen, dan een regel van de rechter, enzovoort. Dat is alleen een vergelijking; hieronder zie je wat er echt uit drie bestanden met dezelfde tekst komt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **document parsing** | de tekst, en wat erin bij elkaar hoort, uit een bestand halen, zoals een webpagina of een pdf |
    | **ingestion** | alles wat documenten in een RAG-systeem brengt: ze verzamelen, parsen, opschonen, in stukken knippen en opslaan voor de zoekfunctie; opnieuw zodra een document verandert |
    | **markup** | wat een bestand naast zijn tekst bevat om te zeggen hoe die eruitziet, zoals `#` voor een kopje in Markdown |
    | **tag** | markup in een webpagina, van een `<` tot de volgende `>`: `<p>` begint een alinea en `</p>` sluit hem af |
    | **entity** | een code in een webpagina voor een teken, zoals `&amp;` voor `&` |
    | **boilerplate** | de tekst die op een pagina staat maar niet bij het onderwerp hoort: het menu, een cookiebanner, de footer |
    | **leesvolgorde** | de volgorde waarin iemand de tekst op een pagina leest |
    | **OCR** | optical character recognition: software die de letters herkent in een plaatje van tekst |

    ### Drie bestanden, één handboek

    De course heeft het handboek van een verzonnen fietsenwinkel, Spoke & Chain, in drie bestanden in de map `data/documents`. De tekst is in elk bestand dezelfde: {words} woorden, een titel en twaalf onderdelen, elk met een kopje en een alinea. Wat er om de tekst heen staat, verschilt:
    """.format(words=C.handbook_words()))
    return


@app.cell(hide_code=True)
def picture_files():
    mo.Html(C.files_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_files():
    mo.md(r"""
    - **Markdown** is tekst met een beetje markup: `#` vóór de titel en `##` vóór een kopje. De tekst komt er schoon uit.
    - **Een webpagina** (HTML) heeft zijn tekst tussen tags. En een pagina bevat meer dan het stuk zelf: een titel voor het tabblad van de browser, een style die zegt hoe hij eruitziet, een script dat in de browser draait, een menu, een cookiebanner en een footer. Haal je de tags weg, dan blijft al die tekst staan.
    - **Een pdf** is een lijst opdrachten om te tekenen: *dit stuk tekst, in dit lettertype, op deze plek op de pagina*. Hij heeft geen alinea's, geen kolommen en geen tabellen, alleen stukken met een plek, en hij mag ze in elke volgorde tekenen. Een tabel komt eruit als losse stukken, en welk getal in welke kolom hoort, is weg, tenzij een tool de tabel aan zijn lijnen herkent.
    - **Een scan** is een plaatje van elke pagina. Er zit helemaal geen tekst in, totdat OCR de letters herkent.

    Tools die dit doen (voorbeelden in november 2026): Pythons eigen `html.parser` en de library BeautifulSoup halen een webpagina uit elkaar in zijn tags; trafilatura vindt de hoofdtekst van een pagina; pypdf en pdfplumber lezen pdf's, en pdfplumber vindt ook tabellen; Docling en Unstructured lezen veel soorten bestanden tegelijk; Tesseract is gratis OCR-software. In deze les doe je de kern ervan zelf, op de drie bestanden van de course.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: van bestand naar tekst

    1. **Lees het bestand** in één string.
    2. **Haal weg wat geen tekst is:** markup, tags, en de delen rond de hoofdtekst.
    3. **Zet de stukken in leesvolgorde,** als het bestand ze in een andere volgorde bewaart.
    4. **Vergelijk met het origineel:** tel de woorden, en lees zelf een paar stukken van elke soort bestand.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _m = C.markdown_example()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst het Markdown-bestand:

    ```python
    {code}
    ```

    1. **Lees het bestand:** `f.read()` geeft het hele bestand als één string, {raw} woorden als je hem bij de spaties splitst. `text.split("\n")` knipt hem in plaats daarvan bij elke regelovergang, `\n`, dus de loop gaat langs zijn regels.
    2. **Haal weg wat geen tekst is:** een `#` is hier een woord op zich, dus {marks} van die woorden zijn alleen markup: één `#` voor de titel en twaalf keer `##` voor de kopjes. `strip` met tekens tussen de haakjes haalt die tekens van beide uiteinden van een string, hier hekjes en spaties: `"## Repairs".strip("# ")` geeft `"Repairs"`. Een regel zonder die tekens blijft zoals hij is.
    3. **Leesvolgorde:** een Markdown-bestand bewaart zijn tekst in de volgorde waarin je hem leest. Niets te doen.
    4. **Vergelijk:** `clean` heeft {clean} woorden, de {words} woorden van het handboek, en de eerste regels zijn *{first}* en *{second}*.
    """.format(code=C.MARKDOWN_CODE.strip(), raw=_m["raw"], marks=_m["raw"] - _m["clean"], clean=_m["clean"], words=C.handbook_words(),
               first=_m["lines"][0], second=_m["lines"][1]))
    return


@app.cell(hide_code=True)
def text_html():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De tags eruit halen

        Schrijf `html_text(page)`, die elk teken van `page` geeft dat buiten de tags staat. Een tag loopt van een `<` tot de volgende `>`, en die twee horen bij de tag. Houd met `inside` bij of je in een tag bent: `True` vanaf een `<`, weer `False` na de `>`. `not inside` is `True` als `inside` `False` is. Het vak voegt elk teken toe, ook die van de tags.
        """),
        L.task("Voeg een teken alleen toe als je niet in een tag bent, en laat de `<` en de `>` zelf weg."),
    ])
    return


@app.cell(hide_code=True)
def exercise_html():
    box_html = L.code_box("html_text", lines=11)
    box_html
    return (box_html,)


@app.cell(hide_code=True)
def check_html(box_html):
    result_html = L.check("html_text", box_html.value)
    result_html
    return (result_html,)


@app.cell(hide_code=True)
def hints_html():
    L.hints(
        "html_text",
        [
            "Elke ronde van de loop doet één van drie dingen: een `<` begint een tag, een `>` sluit er een af, en elk ander teken wordt toegevoegd, maar alleen buiten een tag.",
            "Begin met `inside = False`. Dan zet `if ch == \"<\":` `inside = True`, zet `elif ch == \">\":` `inside = False`, en voegt het derde geval `ch` toe.",
            "Het derde geval: `elif ___:`, met `out = out + ch` eronder.",
        ],
        why="""```python
def html_text(page):
    out = ""
    inside = False
    for ch in page:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out = out + ch
    return out
```

Met `elif` doet een ronde maar één van de drie dingen. Met drie losse `if`s zou de `>` de tag afsluiten en daarna in dezelfde ronde worden toegevoegd.""",
    )
    return


@app.cell(hide_code=True)
def text_main():
    _p = C.page_findings()
    mo.vstack([
        mo.md(r"""
        ### Wat er nog meer op de pagina staat

        Zodra oefening 1 goed is, laat het plaatje eronder je `html_text` op de hele webpagina zien. De tekst die een goede `html_text` eruit haalt, heeft de {kept} woorden van het handboek, en {other} woorden die er niet bij horen: de titel voor het tabblad, de style, het script, het menu, de cookiebanner en de footer, {lines} regels. En {entities} entities, zoals `&amp;` waar de pagina `&` bedoelt. Nog twee stappen ruimen dat op:

        ```python
        main = page.split("<main>")[1].split("</main>")[0]
        text = html.unescape(html_text(main))
        ```

        `split` met een tekst tussen de haakjes knipt bij die tekst in plaats van bij de spaties, dus `[1]` is alles na `<main>`, en daarvan `[0]` alles vóór `</main>`: alleen het hoofddeel van de pagina. `html.unescape`, uit Pythons eigen module `html`, maakt van elke entity het teken waar hij voor staat. Wat overblijft, heeft {words} woorden: dezelfde woorden als het Markdown-bestand, {same}.
        """.format(kept=C.number(_p["kept"], L.lang), other=_p["other"], lines=len(_p["other_lines"]), entities=_p["entities"],
                   words=len(_p["main"].split()), same="in dezelfde volgorde" if _p["same"] else "maar niet in dezelfde volgorde")),
        L.watch_out("Veel pagina's hebben geen `<main>`, of zetten het menu erin. Tools zoals trafilatura raden dan de hoofdtekst, bijvoorbeeld aan waar een pagina veel tekst en weinig links heeft. Die gok kan fout zijn, dus lees zelf een paar opgeschoonde pagina's van elke website."),
    ])
    return


@app.cell(hide_code=True)
def text_pdf():
    _f = C.pdf_findings()
    _first = C.pdf_pieces()[:4]
    mo.md(r"""
    ### Wat er in een pdf staat

    De pdf heeft het handboek op één pagina, in twee kolommen. Dit zijn de eerste regels van de pagina die tekst tekenen, zoals de pdf ze bevat:

    ```
    {ops}
    ```

    `BT` en `ET` beginnen en beëindigen een stuk tekst, `/F1 9 Tf` kiest het eerste lettertype op 9 punten, `Td` gaat naar een plek op de pagina, `x` punten vanaf de linkerrand en `y` vanaf de onderkant (een punt is 1/72 inch), en `Tj` tekent daar de tekst tussen de haakjes. De eigen kleine lezer van de course, `C.pdf_pieces()`, geeft elk stuk dat de pagina tekent als `(x, y, text)`, in de volgorde waarin de pagina ze tekent: {n} stukken, waarvan dit de eerste vier zijn:

    ```python
    {first}
    ```

    Deze pagina tekent rij voor rij: een regel van de linkerkolom, op `x` 50, dan de regel van de rechterkolom op dezelfde hoogte, op `x` 310. Lees je in die volgorde, dan blijven maar {drawn} van de {sentences} zinnen van het handboek heel. Een lezer leest eerst de linkerkolom van boven naar beneden, dan de rechter, en binnen elke kolom tekent de pagina van boven naar beneden.
    """.format(ops="\n".join(C.pdf_ops(4)), n=_f["pieces"], first="\n".join(repr(p) for p in _first), drawn=_f["drawn"], sentences=_f["sentences"]))
    return


@app.cell(hide_code=True)
def text_left():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twee kolommen

        Schrijf `left_first(pieces, middle)`, die de teksten van de stukken in leesvolgorde geeft, één per regel: eerst die met een `x` kleiner dan `middle`, op hun volgorde, dan de andere, op hun volgorde. `for x, y, text in pieces:` geeft elke waarde van een stuk een eigen naam, zoals `x, y = point` dat doet. Het vak geeft de teksten in de volgorde waarin de pagina ze tekent.
        """),
        L.task("Ga de stukken twee keer langs: één keer voor de linkerkolom, één keer voor de rechter."),
    ])
    return


@app.cell(hide_code=True)
def exercise_left():
    box_left = L.code_box("left_first", lines=9)
    box_left
    return (box_left,)


@app.cell(hide_code=True)
def check_left(box_left):
    result_left = L.check("left_first", box_left.value)
    result_left
    return (result_left,)


@app.cell(hide_code=True)
def hints_left():
    L.hints(
        "left_first",
        [
            "Eén loop geeft de volgorde waarin de pagina tekent. Twee loops over dezelfde stukken, die elk maar één kolom houden, geven de linkerkolom vóór de rechter.",
            "Doe in de eerste loop alleen `append` met de tekst voor een stuk uit de linkerkolom, met een `x` kleiner dan `middle`; in een tweede loop daarna alleen voor de andere.",
            "De tweede loop: `for x, y, text in pieces:`, dan `if ___:` met `lines.append(text)` eronder.",
        ],
        why="""```python
def left_first(pieces, middle):
    lines = []
    for x, y, text in pieces:
        if x < middle:
            lines.append(text)
    for x, y, text in pieces:
        if x >= middle:
            lines.append(text)
    return "\\n".join(lines)
```

Elke loop houdt de volgorde van de stukken aan, en op deze pagina wordt elke kolom van boven naar beneden getekend, dus komt elke kolom eruit in de volgorde waarin je hem leest.""",
    )
    return


@app.cell(hide_code=True)
def text_after_pdf():
    _f = C.pdf_findings()
    mo.vstack([
        mo.md(r"""
        In leesvolgorde zijn alle {ordered} zinnen heel, en zijn de {n} regels precies die een lezer leest, in dezelfde volgorde.
        """.format(ordered=_f["ordered"], n=_f["pieces"])),
        L.watch_out("Deze pagina tekent elke kolom van boven naar beneden, en de course weet waar het midden is. Andere pdf's tekenen in elke volgorde, hebben één kolom op de ene pagina en drie op de volgende, of een kopje over beide kolommen. Echte lezers sorteren de stukken op hun plek en vinden de kolommen zelf, en kunnen het nog steeds fout doen: lees zelf een paar pagina's van elke soort pdf."),
    ])
    return


@app.cell(hide_code=True)
def text_scan():
    mo.md(r"""
    ### Oefening 3. Een scan
    """)
    return


@app.cell(hide_code=True)
def choice_scan_cell():
    choice_scan = L.choice(
        "scan",
        question="De winkel heeft ook zijn oude huurcontract, gescand als pdf: een foto van elke pagina. Wat is er nodig voordat een RAG-systeem erin kan zoeken?",
        options={
            "reader": "Niets extra: een pdf-lezer zoals die van de course geeft de stukken tekst, net als bij het handboek",
            "ocr": "OCR, die de letters herkent in het plaatje van elke pagina",
            "tags": "Eerst de tags eruit halen, net als bij een webpagina",
        },
        correct="ocr",
        why={
            "ocr": "Een scan bevat alleen een plaatje van elke pagina, geen tekst. OCR-software zoals Tesseract herkent de letters; controleer daarna een steekproef, want OCR maakt fouten, zoals een 1 voor een l.",
            "reader": "Een pdf-lezer vindt de stukken tekst die een pagina tekent, en een scan tekent er geen: hij laat alleen een plaatje van elke pagina zien. OCR moet eerst de letters herkennen.",
            "tags": "Tags zijn markup in een webpagina; een scan heeft ze niet, en ook geen tekst, alleen een plaatje van elke pagina. OCR moet eerst de letters herkennen.",
        },
    )
    choice_scan
    return (choice_scan,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De zoekfunctie van [Je eerste RAG](/course/open/build-rag-first?lang=nl) heeft tekst nodig, en die kun je nu uit drie soorten bestanden halen. Een heel handboek is nog steeds één lange tekst; [Documenten in stukken knippen](/course/open/build-chunking?lang=nl) knipt documenten in de stukken die een zoekfunctie langsgaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scan, result_html, result_left):
    L.wrap_up(result_html, result_left, choice_scan)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
