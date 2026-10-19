# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-strings. Route: course/route/lessen/stam-1-basis.yaml. Checks en het woord in het beeld: checks.py.
# termen:allow variables, program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-strings", "nl")
    C = L.checks   # het woord in het beeld


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Variabelen en getallen](/course/open/py-variables?lang=nl) en [Namen in een notebook](/course/open/py-names-in-cells?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_calc():
    quick_calc = L.choice(
        "quick-calc",
        question="Wat laat dit programma zien?\n\n```python\nwidth = 4\nlength = width * 3\nprint(length - 2)\n```",
        options={
            "ten": "`10`",
            "twelve": "`12`",
            "word": "`length - 2`",
        },
        correct="ten",
        why={
            "ten": "`length` krijgt 4 keer 3, dus 12, en `print` laat 12 min 2 zien.",
            "twelve": "`print` laat `length - 2` zien, niet `length`. Kijk nog eens naar het recept in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "word": "Zonder aanhalingstekens rekent Python uit wat tussen de haakjes staat. Kijk nog eens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
        },
    )
    quick_calc
    return (quick_calc,)


@app.cell(hide_code=True)
def quick_check_cells():
    quick_cells = L.choice(
        "quick-cells",
        question="Twee cellen in hetzelfde notebook hebben allebei een naam nodig voor een tussenstap. Welke naam mogen ze allebei maken?",
        options={
            "plain": "`step`",
            "underscore": "`_step`",
            "capital": "`Step`",
        },
        correct="underscore",
        why={
            "plain": "Een naam zonder `_` mag maar in één cel gemaakt worden. Kijk nog eens in [Namen in een notebook](/course/open/py-names-in-cells?lang=nl).",
            "capital": "`Step` is gewoon een andere naam zonder `_`, dus ook die mag maar in één cel gemaakt worden. Kijk nog eens in [Namen in een notebook](/course/open/py-names-in-cells?lang=nl).",
            "underscore": "Een naam die met `_` begint, blijft in zijn eigen cel, dus elke cel kan een eigen `_step` hebben.",
        },
    )
    quick_cells
    return (quick_cells,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel AI werkt met tekst: een AI-assistent leest je vraag en schrijft een antwoord. Vaak moet een programma tekst uit elkaar halen: hoe lang is hij, welke letter staat vooraan, welke achteraan? En het moet tekst in elkaar zetten, zoals een antwoord met een getal erin. In deze les doe je het allebei, met strings.

    **Een voorbeeld uit het dagelijks leven.** Een kruiswoordpuzzel. Je telt de hokjes om te weten hoe lang het woord is, en van een kruisend woord weet je misschien dat de derde letter een *n* is: één plek in het woord. Python doet hetzelfde met een string: het telt de tekens, en vindt één teken aan de hand van zijn plek. Eén verschil: de puzzel telt de hokjes vanaf 1, Python telt vanaf 0.
    """)
    return


@app.cell(hide_code=True)
def predict_place():
    guess_place = L.guess(
        "second-letter",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nword = \"banana\"\nprint(word[1])\n```",
        options={
            "b": "`b`",
            "a": "`a`",
            "error": "Een foutmelding: er is geen plek 1.",
        },
        reveal="**`a`.** Python telt plekken vanaf 0. `word[0]` is het eerste teken, `b`, dus `word[1]` is het tweede, `a`. Je ziet het in het beeld hieronder.",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een string is een rij tekens. Elk teken heeft een plek, en Python telt de plekken vanaf 0.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **str** | de naam die Python gebruikt voor een string: tekst tussen aanhalingstekens | `"banana"` |
    | **teken** | één letter, cijfer, spatie of ander symbool in een string, zoals `!` | `"b"`, `" "` |
    | **len** | geeft het aantal tekens in een string, de lengte | `len("banana")` geeft `6` |
    | **index** | de plek van een teken, geteld vanaf 0; je zet hem tussen vierkante haken achter de string of zijn naam | `word[0]` is het eerste teken |
    | **f-string** | een string met een `f` voor het aanhalingsteken; Python vult de waarde in van elke naam tussen accolades `{ }` | `f"Hallo {name}"` |

    Een negatieve index telt vanaf het eind: `word[-1]` is het laatste teken, `word[-2]` het teken ervoor. Tussen de accolades van een f-string mag ook een berekening of een index staan, zoals `{len(word)}` of `{word[0]}`.
    """)
    return


@app.cell(hide_code=True)
def picture_string():
    mo.vstack([
        draw.string_boxes(C.WORD, index_label="index", back_label="vanaf het eind", title="De plekken in een string"),
        mo.md(r"""
        Boven elk teken staat zijn index, geteld vanaf 0; eronder de index vanaf het eind. `banana` heeft 6 tekens, dus de laatste index is 5, één minder dan de lengte.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Het eerste teken is `word[0]`, niet `word[1]`. En een string van 6 tekens heeft geen plek 6: bij `word[6]` stopt Python met een foutmelding. Voor het laatste teken schrijf je `word[-1]`."),
        mo.md(r"""
        ### Recept

        Zo haal je tekst uit elkaar en zet je hem in elkaar:

        1. Geef de tekst een naam: `word = "banana"`.
        2. Tel de tekens met `len(word)`.
        3. Pak één teken met zijn index: `word[0]` voor het eerste, `word[-1]` voor het laatste.
        4. Maak nieuwe tekst met een f-string: een `f` voor het aanhalingsteken, en de namen tussen accolades, zoals in `f"{word} heeft {len(word)} tekens"`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een zin over de stad Utrecht: hoeveel tekens de naam heeft, en wat het eerste en het laatste teken zijn. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_city():
    # 1. geef de tekst een naam
    _city = "Utrecht"
    # 2. het aantal tekens
    _length = len(_city)
    # 3. het eerste en het laatste teken
    _first = _city[0]
    _last = _city[-1]
    # 4. nieuwe tekst met een f-string
    _sentence = f"{_city} heeft {_length} tekens, van {_first} tot {_last}."
    print(_sentence)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Onder de cel verschijnt `Utrecht heeft 7 tekens, van U tot t.` Python vulde de waarde in van elke naam tussen accolades. De hoofdletter `U` blijft een hoofdletter: een string houdt elk teken precies zoals het is.
        """),
        L.task("Verander `\"Utrecht\"` in de naam van je eigen woonplaats, en voer de cel opnieuw uit."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_initials():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Initialen

        Maak in het vak de initialen van `first_name` en `last_name`: van elke naam het eerste teken, direct naast elkaar. Vervang de `...`. De check probeert ook andere namen, dus gebruik de namen en typ de letters niet zelf. Grace Hopper (1906–1992) was een computerpionier: ze bouwde een van de eerste programma's die code die mensen kunnen lezen omzetten in de instructies die een computer uitvoert.
        """),
        L.task("Maak `initials` uit het eerste teken van elke naam, zodat het programma `GH` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_initials():
    box_initials = L.code_box("initials", lines=5)
    box_initials
    return (box_initials,)


@app.cell(hide_code=True)
def check_initials(box_initials):
    result_initials = L.check("initials", box_initials.value)
    result_initials
    return (result_initials,)


@app.cell(hide_code=True)
def hints_initials():
    L.hints(
        "initials",
        [
            "Stap 3 van het recept: welke index heeft het eerste teken van een string?",
            "`first_name[0]` is het eerste teken van `first_name`. Van `last_name` heb je hetzelfde nodig.",
            "Zet de twee tekens naast elkaar in een f-string, elk tussen accolades, met niets ertussen: `f\"{___}{___}\"`.",
        ],
        why="`first_name[0]` is `G` en `last_name[0]` is `H`; de f-string `f\"{first_name[0]}{last_name[0]}\"` zet ze direct naast elkaar.",
    )
    return


@app.cell(hide_code=True)
def text_spaces():
    mo.md(r"""
    ### Oefening 2. Tekens tellen
    """)
    return


@app.cell(hide_code=True)
def choice_spaces_cell():
    choice_spaces = L.choice(
        "spaces",
        question="Wat laat `print(len(\"hoi daar\"))` zien?",
        options={
            "seven": "`7`",
            "two": "`2`",
            "eight": "`8`",
        },
        correct="eight",
        why={
            "seven": "`len` telt elk teken, niet alleen de letters. Kijk nog eens naar het woord *teken* in de tabel.",
            "two": "`len` telt tekens, geen woorden. Kijk nog eens naar het woord *len* in de tabel.",
            "eight": "`len` telt elk teken: 7 letters en de spatie ertussen.",
        },
    )
    choice_spaces
    return (choice_spaces,)


@app.cell(hide_code=True)
def text_exercise_caption():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een bijschrift

        Maak in `caption` een bijschrift voor een kaart: de stad in `city_name`, een spatie, en tussen haakjes het aantal tekens. Voor `Groningen` is dat `Groningen (9)`. Laat `len` het tellen, in een f-string: de check probeert ook een andere stad.
        """),
        L.task("Maak `caption` met een f-string, zodat het programma `Groningen (9)` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caption():
    box_caption = L.code_box("caption", lines=4)
    box_caption
    return (box_caption,)


@app.cell(hide_code=True)
def check_caption(box_caption):
    result_caption = L.check("caption", box_caption.value)
    result_caption
    return (result_caption,)


@app.cell(hide_code=True)
def hints_caption():
    L.hints(
        "caption",
        [
            "Stap 4 van het recept: een `f` voor het aanhalingsteken, en tussen accolades wat Python moet invullen.",
            "Het bijschrift heeft twee delen die Python invult: de naam van de stad, en de lengte. De spatie en de haakjes zijn gewone tekst in de f-string.",
            "`f\"{city_name} (___)\"`: wat komt er tussen de haakjes, tussen accolades?",
        ],
        why="`f\"{city_name} ({len(city_name)})\"` vult `Groningen` in, en de 9 die `len` telt. De spatie en de haakjes blijven zoals ze zijn.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je eerste programma](/course/open/py-first-program?lang=nl) was een string alleen tekst om te laten zien; nu meet je hem, pak je er tekens uit en maak je nieuwe tekst met namen erin. Plekken die je vanaf 0 telt, komen in latere lessen terug, en later in de course knipt een programma tekst in stukjes en telt die, voordat het ervan kan leren. In de volgende les maakt je programma een keuze, met `if`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spaces, result_caption, result_initials):
    L.wrap_up(result_initials, choice_spaces, result_caption)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
