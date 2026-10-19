# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-files. Route: course/route/lessen/stam-1-basis.yaml. Checks en de bestanden van deze les: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-files", "nl")
    C = L.checks   # de paden van de bestanden van de les, en de regels in de figuur


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tekst: strings](/course/open/py-strings?lang=nl) en [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Wat laat dit programma zien?\n\n```python\ncount = 0\nfor word in [\"a\", \"bb\", \"c\"]:\n    count = count + 1\nprint(count)\n```",
        options={
            "one": "`1`",
            "three": "`3`",
            "four": "`4`",
        },
        correct="three",
        why={
            "one": "`count` houdt wat hij al had, en elke ronde telt er 1 bij op. Zie de figuur in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "three": "De loop heeft één ronde voor elk van de drie items, en elke ronde telt 1 op.",
            "four": "De loop telt 1 op bij elk item, niet bij elke letter. Zie het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="Wat laat dit programma zien?\n\n```python\nprint(len(\"hi!\"))\n```",
        options={
            "two": "`2`",
            "five": "`5`",
            "three": "`3`",
        },
        correct="three",
        why={
            "two": "`len` telt elk teken, ook `!`. Zie het idee in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "five": "De aanhalingstekens horen niet bij de string. Zie de figuur in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "three": "`h`, `i` en `!`: drie tekens.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Echte data staat in bestanden: een boek, een lijst metingen, een chat. Tot nu toe stond je data in het programma zelf. Voordat een programma van tekst kan leren, moet het die tekst uit een bestand lezen. Later in de course lees je zo een heel boek, en tel je de letters erin.

    **Een voorbeeld uit het dagelijks leven.** Een kookboek op de plank. Je pakt het, slaat het open, leest wat je nodig hebt, en klapt het weer dicht voordat je het terugzet. Python doet hetzelfde met een bestand, en `with` doet het voor je weer dicht.
    """)
    return


@app.cell(hide_code=True)
def predict_read():
    guess_read = L.guess(
        "read",
        question="Het bestand `poem.txt` heeft vier regels; de eerste is `Roses are red,`. Wat denk je dat dit programma laat zien?\n\n```python\nwith open(\"lessons/py-files/poem.txt\") as f:\n    text = f.read()\nprint(text[0])\n```",
        options={
            "line": "`Roses are red,`",
            "letter": "`R`",
            "word": "`Roses`",
        },
        reveal="**`R`.** `f.read()` geeft het hele bestand als één string, en `[0]` is daarvan het eerste teken, net als bij elke string.",
    )
    guess_read
    return (guess_read,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    `open` opent een bestand, en `with` houdt het open voor de ingesprongen regels eronder, het `with`-blok, en sluit het daarna. Daartussen lees je het: alles in één keer, of regel voor regel.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **bestand** | tekst of andere data die onder een naam op de schijf staat | `poem.txt` |
    | **pad** | waar een bestand staat: de mappen en de bestandsnaam, met `/` ertussen | `"lessons/py-files/poem.txt"` |
    | **open** | opent een bestand, zodat je het kunt lezen | `open(path)` |
    | **with** | houdt een bestand open voor de ingesprongen regels eronder, en sluit het daarna; `as` geeft het open bestand een naam | `with open(path) as f:` |
    | **read** | geeft het hele bestand als één string | `f.read()` |
    | **regeleinde** | het onzichtbare teken dat een regel afsluit, geschreven als `\n` | |
    | **strip** | geeft een string zonder de spaties en regeleinden aan beide kanten | `line.strip()` |

    `for line in f:` gaat een open bestand regel voor regel langs, net als een lijst. Elke regel eindigt met een regeleinde; alleen de laatste regel van een bestand heeft er soms geen, en `strip` werkt in beide gevallen. Een pad als `"lessons/py-files/poem.txt"` begint bij de map waarin Python draait. In de course is dat de map van de course, `everything-ai`, waar je de course startte; de bestanden van deze les staan in de map `lessons/py-files` daarin. (De lijst met bestanden van marimo laat je eigen kopieën zien, in `work/`, zonder deze bestanden.) Een pad dat niet naar een bestand leidt, stopt het programma met een fout. Sluiten is belangrijk: een bestand dat open blijft, kan in gebruik blijven, en `with` sluit het voor je, ook als er iets misgaat.
    """)
    return


@app.cell(hide_code=True)
def picture_lines():
    mo.vstack([
        draw.marked_lines([("Het begin van poem.txt, zoals jij het ziet", [(_x, "") for _x in C.PICTURE_FILE]),
                           ("Wat de loop je geeft", [(repr(_x + "\n"), "1") for _x in C.PICTURE_FILE]),
                           ("En line.strip()", [(repr(_x), "") for _x in C.PICTURE_FILE])],
                          {"1": "elke regel eindigt met \\n, het regeleinde"},
                          title="Regels van een bestand, zoals Python ze leest"),
        mo.md(r"""
        Jij ziet twee regels, maar Python ziet wat er echt in het bestand staat: elke regel eindigt met `\n`. Python zet elke string hier tussen aanhalingstekens, zodat je ziet waar hij ophoudt. Daarom laat `print(line)` na elke regel een lege regel zien: het eigen regeleinde van de regel, plus dat van `print`. `line.strip()` haalt het weg.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Elke regel heeft zijn regeleinde nog: zonder `strip` laat `print` er een lege regel na zien, en telt `len(line)` één teken te veel. En gebruik het bestand alleen in de regels die onder `with` inspringen: daarna is het weer gesloten."),
        mo.md(r"""
        ### Recept

        Zo lees je een tekstbestand:

        1. Schrijf het pad als string: de mappen en de bestandsnaam, met `/` ertussen.
        2. Schrijf `with open(path) as f:`, met een dubbele punt.
        3. Ingesprongen eronder: `f.read()` voor de hele tekst in één keer, of `for line in f:` voor één regel tegelijk.
        4. Gebruik `line.strip()` om het regeleinde van elke regel weg te halen.
        5. Na het `with`-blok, de ingesprongen regels, is het bestand weer gesloten; bewaar wat je nodig hebt onder een naam.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een kort gedicht, op twee manieren gelezen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_poem():
    # 1. het pad: van de map van de course naar het bestand
    _path = "lessons/py-files/poem.txt"
    # 2. en 3. open het met with, en lees alles als één string
    with open(_path) as _f:
        _text = _f.read()
    print(len(_text))
    # 3. of één regel tegelijk: tel de regels, en laat de lange zien
    _count = 0
    with open(_path) as _f:
        for _line in _f:
            _count = _count + 1
            # 4. strip haalt het regeleinde weg voordat je de regel meet of laat zien
            if len(_line.strip()) > 15:
                print(_line.strip())
    # 5. na het with-blok is het bestand gesloten; de telling blijft
    print(_count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `71`, drie regels van het gedicht, en `4`. Bij die 71 tekens tellen de vier regeleinden mee. `Roses are red,` heeft 14 tekens, dus de `if` slaat die over; de andere drie regels zijn langer dan 15. Het bestand heeft vier regels, dus de loop had vier rondes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tel de regels

        `shopping.txt` bevat een boodschappenlijstje, één item per regel. Tel de regels in `count`: vervang de `...` door een `with`-blok en een loop. De check probeert ook een ander bestand.
        """),
        L.task("Tel de regels van het bestand in `path`, zodat het programma `5` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-lines", lines=7)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-lines", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-lines",
        [
            "Kijk naar het uitgewerkte voorbeeld: een regel met `with` opent het bestand, en een loop eronder gaat de regels langs.",
            "In `for line in f:` telt elke ronde 1 op bij `count`. `print(count)` blijft helemaal links.",
            "`with open(path) as f:`, dan ingesprongen `for line in f:`, en nog een keer ingesprongen ___.",
        ],
        why="`with open(path) as f:` opent het bestand, `for line in f:` heeft één ronde voor elke regel, en `count = count + 1` telt in elke ronde 1 op. `print(count)` komt na het `with`-blok, dus draait hij één keer: 5.",
    )
    return


@app.cell(hide_code=True)
def text_newline():
    mo.md(r"""
    ### Oefening 2. Het onzichtbare teken
    """)
    return


@app.cell(hide_code=True)
def choice_newline_cell():
    choice_newline = L.choice(
        "newline",
        question="In `shopping.txt` is de tweede regel `milk`. Wat geeft `len(line)` voor die regel, in `for line in f:`?",
        options={
            "five": "`5`",
            "four": "`4`",
            "one": "`1`",
        },
        correct="five",
        why={
            "five": "`m`, `i`, `l`, `k`, en het regeleinde aan het eind: 5 tekens. `len(line.strip())` zou 4 geven.",
            "four": "Je ziet vier letters, maar aan het eind van de regel staat een teken dat je niet ziet. Kijk nog eens naar de figuur.",
            "one": "`len` telt de tekens van deze ene regel, geen regels. Kijk nog eens naar de figuur.",
        },
    )
    choice_newline
    return (choice_newline,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De lange items

        Laat de items uit `shopping.txt` met meer dan 4 letters zien, elk op een eigen regel, zonder lege regels. Vervang de `...` in de loop. De check probeert ook een ander bestand.
        """),
        L.task("Laat de items met meer dan 4 letters zien: `bread`, `apples` en `cheese`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_long():
    box_long = L.code_box("long-items", lines=7)
    box_long
    return (box_long,)


@app.cell(hide_code=True)
def check_long(box_long):
    result_long = L.check("long-items", box_long.value)
    result_long
    return (result_long,)


@app.cell(hide_code=True)
def hints_long():
    L.hints(
        "long-items",
        [
            "Stap 4 van het recept: eerst `line.strip()`, dan kijken naar de lengte.",
            "Een `if` met `len(...) > 4` laat alleen de lange items zien. Welke string meet en laat je zien: `line`, of die zonder regeleinde?",
            "`item = line.strip()`, dan `if len(item) > 4:` en daaronder ___.",
        ],
        why="`item = line.strip()` haalt het regeleinde weg, dus telt `len(item)` alleen de letters: `milk` en `eggs` hebben er 4, en de `if` slaat ze over. `print(item)` laat het item zien zonder extra lege regel.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tekst: strings](/course/open/py-strings?lang=nl) telde je de tekens van een string en pakte je er losse tekens uit; een bestand geeft je zo'n string, of één regel tegelijk, zoals een [for-loop](/course/open/py-for?lang=nl) een lijst langsgaat. In een project verderop komt zo een heel boek in je programma. In de volgende les gebruik je code die anderen schreven, uit modules.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_newline, result_count, result_long):
    L.wrap_up(result_count, choice_newline, result_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
