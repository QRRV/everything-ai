# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-dicts. Route: course/route/lessen/stam-1-basis.yaml. Checks en de dictionary in de figuur: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-dicts", "nl")
    C = L.checks   # de dictionary in de figuur, en de key die wordt opgezocht


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="Wat laat dit programma zien?\n\n```python\nfor name in [\"Ada\", \"Bo\"]:\n    print(f\"Hoi {name}\")\n```",
        options={
            "each": "`Hoi Ada` en `Hoi Bo`, elk op een eigen regel",
            "one-line": "`Hoi Ada Bo`, op één regel",
            "word": "Twee keer `Hoi name`",
        },
        correct="each",
        why={
            "one-line": "De `print` draait in elke ronde één keer, en elke `print` begint een nieuwe regel. Zie de gok in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "each": "Elke ronde krijgt `name` het volgende item, en de f-string vult het in.",
            "word": "Tussen accolades vult de f-string de waarde van `name` in, en die is elke ronde een ander item. Zie het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="Wat laat dit programma zien?\n\n```python\ntotal = 0\nfor n in [3, 8, 5]:\n    if n > 4:\n        total = total + n\nprint(total)\n```",
        options={
            "all": "`16`",
            "last": "`5`",
            "big": "`13`",
        },
        correct="big",
        why={
            "all": "De `if` telt alleen de getallen groter dan 4 op, en 3 is dat niet. Zie het uitgewerkte voorbeeld in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "big": "8 en 5 zijn groter dan 4, dus wordt `total` 8 + 5.",
            "last": "`total` houdt wat hij al had, en in elke ronde komt er een getal bij. Zie de figuur in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe vond je een item via zijn plek: `scores[0]`. Maar vaak wil je iets vinden via een naam: de prijs van peren, of hoe vaak een woord in een tekst staat. Een dictionary bewaart waarden onder een eigen naam, zodat je ze kunt opzoeken. Later in de course houdt een programma dat tekst leest, een dictionary bij die elk woord een eigen nummer geeft.

    **Een voorbeeld uit het dagelijks leven.** De contacten in je telefoon. Je zoekt een naam op, en je krijgt het telefoonnummer dat erbij hoort. Op welke plek in de lijst die naam staat, hoef je nooit te weten. En elke naam staat er maar één keer in: sla je een nieuw nummer op onder dezelfde naam, dan is het oude nummer weg.
    """)
    return


@app.cell(hide_code=True)
def predict_replace():
    guess_replace = L.guess(
        "replace",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nprices = {\"apple\": 0.5, \"pear\": 0.75}\nprices[\"apple\"] = 0.6\nprint(prices[\"apple\"])\n```",
        options={
            "old": "`0.5`",
            "new": "`0.6`",
            "both": "`0.5` en `0.6`",
        },
        reveal="**`0.6`.** De tweede regel geeft de key `\"apple\"` een nieuwe waarde. Een key staat maar één keer in een dictionary, dus de oude waarde, 0,5, is weg.",
    )
    guess_replace
    return (guess_replace,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een dictionary bewaart paren: een key, en de waarde die erbij hoort. Je zoekt een waarde op via de key, niet via een plek.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **dictionary** | paren van een key en een waarde, tussen accolades; een dubbele punt tussen elke key en zijn waarde, en een komma tussen de paren | `prices = {"apple": 0.5, "pear": 0.75}` |
        | **key** | waarmee je een waarde opzoekt; elke key staat maar één keer in een dictionary | `prices["pear"]` geeft `0.75` |
        | **in** | kijkt of een key in een dictionary staat, en geeft `True` of `False` | `"plum" in prices` geeft `False` |

        Met `=` verander je de waarde onder een key, of voeg je een nieuw paar toe: `prices["plum"] = 0.3`. Een key opzoeken die er niet is, stopt het programma met een fout. Weet je het niet zeker, kijk dan eerst met `in`. `{}` is een lege dictionary, om later te vullen, en `len(prices)` geeft het aantal paren. Laat Python een dictionary zien, dan zet Python enkele aanhalingstekens om strings: `{'apple': 0.5}`.
        """),
        L.watch_out("Later in de course betekent *key* ook iets anders, in een programma dat tekst leest; die les legt uit wat. Hier is een key waarmee je een waarde opzoekt. Ook in het Nederlands zeggen we *key*, niet sleutel."),  # termen:ok
    ])
    return


@app.cell(hide_code=True)
def picture_dict():
    mo.vstack([
        mo.md("```python\nprices = {\"apple\": 0.5, \"pear\": 0.75, \"plum\": 0.3}\nprint(prices[\"pear\"])\n```"),
        draw.dict_boxes([(f'"{_k}"', str(_v)) for _k, _v in C.PICTURE.items()], key_label="key", value_label="waarde",
                        mark=list(C.PICTURE).index(C.PICTURE_MARK), mark_label=f'prices["{C.PICTURE_MARK}"]',
                        title="Een dictionary, en één keer opzoeken"),
        mo.md(r"""
        Elke key wijst naar zijn waarde. `prices["pear"]` zoekt de key `"pear"` en geeft de waarde ernaast, 0,75. Waar het paar in de dictionary staat, maakt niet uit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een key staat maar één keer in een dictionary: geef je hem opnieuw een waarde, dan vervangt die de oude. En een key moet precies hetzelfde zijn: `\"Pear\"` met een hoofdletter is een andere key dan `\"pear\"`. Een key opzoeken die er niet is, stopt je programma; kijk dus eerst met `in` als je het niet zeker weet."),
        mo.md(r"""
        ### Recept

        Zo werk je met een dictionary:

        1. Maak hem met accolades, en paren `key: waarde` met een komma ertussen: `prices = {"apple": 0.5, "pear": 0.75}`.
        2. Zoek een waarde op met de key tussen vierkante haken: `prices["pear"]`.
        3. Verander een waarde, of voeg een nieuw paar toe, met `=`: `prices["plum"] = 0.3`.
        4. Weet je niet zeker of een key er is? Kijk eerst met `in`: `if "plum" in prices:`.
        5. Ga alle paren langs met `for`: in `for fruit in prices:` krijgt `fruit` elke key, en `prices[fruit]` geeft de waarde erbij.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De voorraad van een kleine winkel: hoeveel is er van elk product? Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_stock():
    # 1. de dictionary: elk product, met hoeveel er zijn
    _stock = {"brood": 4, "melk": 6}
    # 2. zoek een waarde op via de key
    print(_stock["melk"])
    # 3. voeg een nieuw paar toe: er worden eieren geleverd
    _stock["eieren"] = 12
    # wat er vandaag verkocht is
    _sold = ["melk", "appels", "melk"]
    for _item in _sold:
        # 4. kijk met in of de key er is
        if _item in _stock:
            # 3. verander de waarde onder een key: één minder
            _stock[_item] = _stock[_item] - 1
        else:
            print(f"Geen {_item} in de winkel")
    # 5. ga alle paren langs: _item krijgt elke key
    for _item in _stock:
        print(f"{_item}: {_stock[_item]}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Onder de cel verschijnen vijf regels. `_stock["melk"]` geeft `6`. Er is twee keer melk verkocht, dus gaat melk naar 4. `appels` is geen key, dus draait de `else`. De laatste loop laat elke key met zijn waarde zien, in de volgorde waarin de paren erbij kwamen: `eieren` kwam als laatste.
        """),
        L.task("Zet `\"brood\"` erbij in de lijst `_sold` in de cel hierboven, en voer hem uit. Wat laat de laatste loop nu zien bij brood?"),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_capital():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een hoofdstad

        `capitals` geeft de hoofdstad van elk land. Maak `city` de hoofdstad van het land in `country`. Vervang de `...`. De check probeert ook een ander land.
        """),
        L.task("Zoek de hoofdstad van `country` op in `capitals`, zodat het programma `Tokyo` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_capital():
    box_capital = L.code_box("capital", lines=5)
    box_capital
    return (box_capital,)


@app.cell(hide_code=True)
def check_capital(box_capital):
    result_capital = L.check("capital", box_capital.value)
    result_capital
    return (result_capital,)


@app.cell(hide_code=True)
def hints_capital():
    L.hints(
        "capital",
        [
            "Stap 2 van het recept: je zoekt een waarde op met de key tussen vierkante haken. Welke naam bevat hier de key?",
            "`country` bevat de key, `\"Japan\"`. Welke naam zet je tussen de haken, zodat het ook werkt als `country` een ander land is?",
            "`city = capitals[___]`",
        ],
        why="`capitals[country]` zoekt de waarde op onder de key in `country`: voor `\"Japan\"` is dat `\"Tokyo\"`, en voor `\"Peru\"` is het `\"Lima\"`. Zonder aanhalingstekens is `country` een naam, en gebruikt Python de waarde ervan.",
    )
    return


@app.cell(hide_code=True)
def text_keys():
    mo.md(r"""
    ### Oefening 2. Een loop over een dictionary
    """)
    return


@app.cell(hide_code=True)
def choice_keys_cell():
    choice_keys = L.choice(
        "keys",
        question="Wat laat dit programma zien?\n\n```python\nages = {\"Ada\": 36, \"Bo\": 12}\nfor name in ages:\n    print(name)\n```",
        options={
            "values": "`36` en `12`, elk op een eigen regel",
            "pairs": "`Ada 36` en `Bo 12`, elk op een eigen regel",
            "keys": "`Ada` en `Bo`, elk op een eigen regel",
        },
        correct="keys",
        why={
            "values": "Een loop over een dictionary geeft niet de waarden. Kijk nog eens naar stap 5 van het recept: wat krijgt de naam na `for`?",
            "keys": "Een loop over een dictionary gaat de keys langs: `name` is `\"Ada\"`, daarna `\"Bo\"`. `ages[name]` zou de waarden geven.",
            "pairs": "De naam na `for` krijgt elke ronde één ding, geen paar. Kijk nog eens naar stap 5 van het recept.",
        },
    )
    choice_keys
    return (choice_keys,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Woorden tellen

        Tel hoe vaak elk woord in `words` staat. `counts` begint leeg, als `{}`. Vervang de `...` in de loop door een `if` en een `else`: is het woord al een key in `counts`? Dan gaat zijn telling 1 omhoog. Zo niet, dan krijgt het een nieuw paar, met telling 1. In de loop staat `else:` gelijk met zijn `if`: druk in het vak één keer op Backspace voordat je hem typt. De check probeert ook een andere lijst.
        """),
        L.task("Laat `counts` bij elk woord in `words` zeggen hoe vaak het erin staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count",
        [
            "Kijk naar de eerste loop van het uitgewerkte voorbeeld: een `if` met `in` kijkt of het item een key is, en de `else` regelt het andere geval.",
            "Is het woord al een key in `counts`? Dan is de nieuwe telling de oude telling plus 1. Zo niet, dan krijgt het woord een nieuw paar, met telling 1.",
            "`if word in counts:`, met daaronder de regel die 1 bij de telling optelt; dan `else:`, en daaronder ___.",
        ],
        why="De eerste keer dat een woord langskomt, is `word in counts` `False`, dus geeft de `else` het een nieuw paar met 1. De volgende keer is het `True`, en gaat de telling 1 omhoog. `rain` komt drie keer langs: 1, 2, 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) vond je een item via zijn plek; in een dictionary vind je een waarde via de key. Met een `for`-loop uit [Herhalen: for-loops](/course/open/py-for?lang=nl) telde je elk woord in één keer. Later in de course houdt een programma dat tekst leest, precies zo'n dictionary als `counts` bij, en ook een dictionary die elk woord een eigen nummer geeft. In de volgende les zie je tuples en sets, nog twee manieren om waarden bij elkaar te houden, en hoe je een lijst sorteert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keys, result_capital, result_count):
    L.wrap_up(result_capital, choice_keys, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
