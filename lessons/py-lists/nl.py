# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-lists. Route: course/route/lessen/stam-1-basis.yaml. Checks en de lijst in het beeld: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-lists", "nl")
    C = L.checks   # de lijst in het beeld, en de slice die het aanwijst


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tekst: strings](/course/open/py-strings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_last():
    quick_last = L.choice(
        "quick-last",
        question="Wat laat dit programma zien?\n\n```python\nword = \"python\"\nprint(word[-1])\n```",
        options={
            "p": "`p`",
            "o": "`o`",
            "n": "`n`",
        },
        correct="n",
        why={
            "p": "`p` staat op index 0, het eerste teken. Een negatieve index telt vanaf het eind. Kijk nog eens naar het beeld in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "o": "Dat is het op een na laatste teken, `word[-2]`. Kijk nog eens naar het beeld in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "n": "`-1` telt vanaf het eind: het laatste teken, `n`.",
        },
    )
    quick_last
    return (quick_last,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="Wat laat dit programma zien?\n\n```python\nname = \"Ada\"\nprint(f\"Hi {name}\")\n```",
        options={
            "filled": "`Hi Ada`",
            "braces": "`Hi {name}`",
            "word": "`Hi name`",
        },
        correct="filled",
        why={
            "filled": "In een f-string vult Python de waarde in van elke naam tussen accolades.",
            "braces": "De accolades blijven alleen staan als de `f` voor het aanhalingsteken ontbreekt. Kijk nog eens naar het recept in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "word": "Tussen accolades vult Python de waarde van de naam in, niet de naam zelf. Kijk nog eens in [Tekst: strings](/course/open/py-strings?lang=nl).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    AI werkt met veel data: de temperaturen van een heel jaar, alle woorden van een boek, duizend scores. Je kunt niet elke waarde een eigen naam geven. Een lijst houdt veel waarden bij elkaar, op volgorde, onder één naam. Vanaf de volgende lessen loop je lijsten waarde voor waarde door, en later in de course leert een programma van lange lijsten met voorbeelden.

    **Een voorbeeld uit het dagelijks leven.** Een afspeellijst. De nummers staan op volgorde: nummer 1, nummer 2, nummer 3. Je kunt één nummer afspelen aan de hand van zijn plek, de nummers 2 tot en met 4 afspelen, of achteraan een nieuw nummer toevoegen. Een lijst in Python werkt ongeveer zo. Eén verschil: een afspeellijst telt vanaf 1, Python vanaf 0.
    """)
    return


@app.cell(hide_code=True)
def predict_slice():
    guess_slice = L.guess(
        "slice",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nscores = [7, 9, 4, 8, 6]\nprint(scores[1:3])\n```",
        options={
            "three": "`[9, 4, 8]`",
            "two": "`[9, 4]`",
            "first": "`[7, 9, 4]`",
        },
        reveal="**`[9, 4]`.** `scores[1:3]` begint bij index 1, dat is `9`, en stopt net voor index 3. Je krijgt dus de items op index 1 en 2. Je ziet het in het beeld hieronder.",
    )
    guess_slice
    return (guess_slice,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een lijst houdt waarden op volgorde. Plekken werken net als bij een string: geteld vanaf 0, en met `-1` voor de laatste.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **lijst** (in Python: `list`) | waarden op volgorde, tussen vierkante haken, met een komma ertussen | `scores = [7, 9, 4, 8, 6]` |
    | **item** | één waarde in een lijst; je pakt het met zijn index | `scores[0]` geeft `7` |
    | **slice** | een stuk van een lijst, van de ene index tot een andere, die zelf niet meedoet | `scores[1:3]` geeft `[9, 4]` |
    | **append** | voegt één item achteraan een lijst toe; je schrijft het achter de naam van de lijst, met een punt ertussen | `scores.append(5)` |

    Een slice mag een getal weglaten: `scores[:2]` begint vooraan, en `scores[2:]` gaat door tot het eind. Een negatief getal telt vanaf het eind, net als bij een string: `scores[-2:]` geeft de laatste twee. `len(scores)` geeft het aantal items, net als bij een string. En laat Python een lijst met strings zien, dan zet Python om elke string enkele aanhalingstekens, ook als je dubbele typte: `['bread', 'milk']`.
    """)
    return


@app.cell(hide_code=True)
def picture_list():
    mo.vstack([
        draw.list_boxes([str(_x) for _x in C.PICTURE], index_label="index", back_label="vanaf het eind", span=C.PICTURE_SPAN,
                        span_label="scores[1:3]", title="Een lijst, en een slice ervan"),
        mo.md(r"""
        Boven elk item staat zijn index, en eronder de index vanaf het eind. De slice `scores[1:3]` pakt de items op index 1 en 2; bij index 3 stopt hij, en dat item hoort er niet bij.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een slice stopt net voor zijn tweede getal: `scores[1:3]` geeft 2 items, geen 3. Daardoor geeft `scores[:2]` precies de eerste 2. En `append` verandert de lijst zelf en geeft niets terug, dus schrijf `scores.append(5)` op een eigen regel, niet `scores = scores.append(5)`."),
        mo.md(r"""
        ### Recept

        Zo werk je met een lijst:

        1. Zet de waarden in een lijst: `scores = [7, 9, 4, 8, 6]`.
        2. Pak één item met zijn index: `scores[0]` voor het eerste, `scores[-1]` voor het laatste.
        3. Pak een stuk met een slice: `scores[start:stop]`; het item op `stop` hoort er niet bij. Een negatief getal telt vanaf het eind.
        4. Voeg achteraan een item toe met `append`: `scores.append(5)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Uren slaap van maandag tot en met vrijdag. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_sleep():
    # 1. de waarden in een lijst, maandag tot en met vrijdag
    _sleep = [7, 6.5, 8, 7.5, 6]
    # 2. één item: de eerste en de laatste dag
    print(f"maandag {_sleep[0]}, vrijdag {_sleep[-1]}")
    # 3. een stuk: dinsdag tot en met donderdag, van index 1 tot 4; 4 doet zelf niet mee
    _middle = _sleep[1:4]
    print(_middle)
    # en de laatste twee dagen, vanaf het eind geteld
    print(_sleep[-2:])
    # 4. nog een item achteraan: zaterdag
    _sleep.append(9)
    print(f"{len(_sleep)} dagen: {_sleep}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Onder de cel verschijnen vier regels. De slice `_sleep[1:4]` geeft de drie dagen in het midden, `[6.5, 8, 7.5]`, en `_sleep[-2:]` de laatste twee, `[7.5, 6]`. Na `append` heeft de lijst 6 items, en het nieuwe staat achteraan.
        """),
        L.task("Verander `_sleep[1:4]` in de cel hierboven in `_sleep[:2]`, en voer hem uit. Welke dagen krijg je nu?"),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_top():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De eerste drie

        Maak `top`: de eerste drie scores van `scores`, als lijst. Vervang de `...` door een slice. De check probeert ook een andere lijst.
        """),
        L.task("Maak `top` met een slice, zodat het programma `[7, 9, 4]` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top():
    box_top = L.code_box("top", lines=4)
    box_top
    return (box_top,)


@app.cell(hide_code=True)
def check_top(box_top):
    result_top = L.check("top", box_top.value)
    result_top
    return (result_top,)


@app.cell(hide_code=True)
def hints_top():
    L.hints(
        "top",
        [
            "Stap 3 van het recept: een slice is `scores[start:stop]`. Op welke index staat de eerste score?",
            "De slice begint bij 0. Hij stopt net voor zijn tweede getal: welk getal geeft precies drie items?",
            "`top = scores[0:___]`",
        ],
        why="`scores[:3]` (of `scores[0:3]`) begint bij index 0 en stopt net voor 3: de items op 0, 1 en 2, samen drie.",
    )
    return


@app.cell(hide_code=True)
def text_last_two():
    mo.md(r"""
    ### Oefening 2. Vanaf het eind
    """)
    return


@app.cell(hide_code=True)
def choice_last_two_cell():
    choice_last_two = L.choice(
        "last-two",
        question="Wat laat dit programma zien?\n\n```python\nnums = [5, 6, 7, 8]\nprint(nums[-2:])\n```",
        options={
            "three": "`[6, 7, 8]`",
            "two": "`[7, 8]`",
            "one": "`[8]`",
        },
        correct="two",
        why={
            "three": "`-1` is het laatste item. Welk item is `-2`? Kijk nog eens naar de rij vanaf het eind in het beeld.",
            "two": "`-2` is `7`, het tweede item vanaf het eind, en zonder getal na de dubbele punt gaat de slice door tot het eind.",
            "one": "Zonder getal na de dubbele punt gaat een slice door tot het eind. Kijk nog eens naar de zin onder de tabel met nieuwe woorden.",
        },
    )
    choice_last_two
    return (choice_last_two,)


@app.cell(hide_code=True)
def text_exercise_cart():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het boodschappenmandje

        Vervang de `...` door één regel die `"apples"` achteraan `cart` toevoegt. De laatste regel laat het mandje zien. De check probeert ook een ander mandje.
        """),
        L.task("Voeg `\"apples\"` achteraan `cart` toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cart():
    box_cart = L.code_box("cart", lines=4)
    box_cart
    return (box_cart,)


@app.cell(hide_code=True)
def check_cart(box_cart):
    result_cart = L.check("cart", box_cart.value)
    result_cart
    return (result_cart,)


@app.cell(hide_code=True)
def hints_cart():
    L.hints(
        "cart",
        [
            "Stap 4 van het recept: welke instructie voegt een item achteraan een lijst toe?",
            "`append` schrijf je achter de naam van de lijst, met een punt ertussen, en het nieuwe item komt tussen de haakjes.",
            "`cart.append(___)`, met de tekst tussen aanhalingstekens.",
        ],
        why="`cart.append(\"apples\")` voegt de string `\"apples\"` achteraan de lijst toe. `append` verandert `cart` zelf, dus de regel heeft geen `cart =` ervoor.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tekst: strings](/course/open/py-strings?lang=nl) pakte je tekens uit een string met hun index; een lijst werkt net zo, maar de items kunnen van alles zijn: getallen, strings, zelfs andere lijsten. In de volgende les ga je een lijst item voor item langs, met een loop. Later werk je met NumPy, dat getallen net zo op een rij bewaart, maar er veel sneller mee rekent.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_last_two, result_cart, result_top):
    L.wrap_up(result_top, choice_last_two, result_cart)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
