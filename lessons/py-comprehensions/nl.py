# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-comprehensions. Route: course/route/lessen/stam-1-basis.yaml. Checks en de delen in de figuur: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-comprehensions", "nl")
    C = L.checks   # de loop en de regel in de figuur, deel voor deel


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Dictionaries](/course/open/py-dicts?lang=nl) en [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="Wat laat dit programma zien?\n\n```python\nages = {\"Ada\": 36}\nages[\"Bo\"] = 12\nprint(len(ages))\n```",
        options={
            "two": "`2`",
            "one": "`1`",
            "error": "Niets: Python stopt met een fout, omdat `\"Bo\"` nog geen key is",
        },
        correct="two",
        why={
            "two": "`ages[\"Bo\"] = 12` voegt een nieuw paar toe, dus de dictionary heeft twee paren.",
            "one": "De tweede regel voegt een paar toe; hij vervangt `\"Ada\"` niet. Zie het recept in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "error": "Een key opzoeken die er niet is, stopt het programma, maar een nieuwe key een waarde geven met `=` voegt een paar toe. Zie het recept in [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_sorted():
    quick_sorted = L.choice(
        "quick-sorted",
        question="Wat laat dit programma zien?\n\n```python\nprint(sorted(set([3, 1, 3])))\n```",
        options={
            "twice": "`[1, 3, 3]`",
            "set": "`{1, 3}`",
            "once": "`[1, 3]`",
        },
        correct="once",
        why={
            "twice": "Eerst komt de set, en die houdt elke waarde één keer. Zie het recept in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "set": "`sorted` geeft altijd een lijst, ook als je er een set aan geeft. Zie het idee in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "once": "De set houdt 1 en 3 elk één keer, en `sorted` maakt er een lijst van, op volgorde.",
        },
    )
    quick_sorted
    return (quick_sorted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vaak maak je een nieuwe lijst van een oude: elke prijs met btw erbij, of alleen de warme dagen. Met een `for`-loop kost dat een paar regels: een lege lijst, de loop, en `append`. Een comprehension zegt hetzelfde in één regel, die bijna leest als een zin. In code over data en AI kom je ze overal tegen.

    **Een voorbeeld uit het dagelijks leven.** Een docent zegt: "Schrijf de naam op van iedereen die een 8 of hoger heeft." Die ene zin beschrijft de hele nieuwe lijst: wat erop komt (een naam), naar wie je kijkt (iedereen), en wie meetelt (8 of hoger). Een comprehension heeft precies die drie delen.
    """)
    return


@app.cell(hide_code=True)
def predict_double():
    guess_double = L.guess(
        "double",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nprint([n * 2 for n in [1, 2, 3]])\n```",
        options={
            "doubled": "`[2, 4, 6]`",
            "twice": "`[1, 2, 3, 1, 2, 3]`",
            "total": "`12`",
        },
        reveal="**`[2, 4, 6]`.** Voor elke `n` in de lijst rekent de comprehension `n * 2` uit en zet dat in een nieuwe lijst. De vierkante haken eromheen maken die nieuwe lijst.",
    )
    guess_double
    return (guess_double,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een comprehension bouwt een nieuwe lijst in één regel: voor elk item van een oude lijst wat er in de nieuwe komt, en, als je wilt, een voorwaarde die bepaalt welke items meetellen.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **comprehension** | een nieuwe lijst in één regel: tussen vierkante haken wat erin komt voor één item, dan `for`, een naam en de oude lijst | `[n * 2 for n in nums]` |
    | **if** aan het eind | houdt alleen de items waarvoor de voorwaarde `True` is | `[t for t in temps if t > 20]` |

    De naam na `for` is het item van deze ronde, net als in een `for`-loop, en die gebruik je vooraan: `[word[0] for word in words]` geeft de eerste letter van elk woord. De oude lijst blijft zoals hij is. Met accolades, en vooraan een key en een waarde, bouwt een comprehension een dictionary: `{word: word[0] for word in words}`.
    """)
    return


@app.cell(hide_code=True)
def picture_parts():
    mo.vstack([
        draw.comprehension_parts(C.PICTURE_LOOP, C.PICTURE_LINE,
                                 {"1": "wat er in de nieuwe lijst komt, voor één item",
                                  "2": "voor elk item van de oude lijst",
                                  "3": "alleen als de voorwaarde True is"},
                                 loop_label="Met een loop", line_label="Dezelfde lijst, in één regel",
                                 title="Een loop, en dezelfde lijst als comprehension"),
        mo.md(r"""
        Allebei maken ze dezelfde lijst: voor elke temperatuur boven 20 hoeveel graden hij erboven zit. De nummers laten zien waar elk deel van de loop heen gaat. Wat er in de lijst komt, de laatste regel van de loop, komt eerst; dan de `for`, en dan de `if`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("In een comprehension staat wat er in de lijst komt vooraan, vóór `for`, en de `if` achteraan; er staat geen dubbele punt aan het eind, zoals na een regel met `for`. En houd het kort: wordt een comprehension moeilijk te lezen, dan is een gewone `for`-loop beter. Een comprehension is er om een lijst of een dictionary te bouwen, niet om iets te laten zien met `print`."),
        mo.md(r"""
        ### Recept

        Zo bouw je een nieuwe lijst van een oude, in één regel:

        1. Open vierkante haken: die maken de nieuwe lijst.
        2. Schrijf wat er in de nieuwe lijst komt, voor één item: `t - 20`.
        3. Dan `for`, een naam voor één item, `in`, en de oude lijst: `for t in temps`.
        4. Alleen sommige items? Zet `if` en een voorwaarde aan het eind: `if t > 20`.
        5. Liever een dictionary? Gebruik accolades, met vooraan een key en een waarde: `{w: len(w) for w in words}`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Temperaturen in graden Celsius, en een paar woorden. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_comprehensions():
    _celsius = [0, 10, 25, 30, -5]
    # 1. tot en met 3. vierkante haken, wat erin komt voor één item, en for: elke temperatuur in Fahrenheit
    _fahrenheit = [_c * 9 / 5 + 32 for _c in _celsius]
    print(_fahrenheit)
    # 4. een if aan het eind: alleen de dagen boven 20
    _warm = [_c for _c in _celsius if _c > 20]
    print(_warm)
    # 4. ook over een dictionary: de dagen zonder regen
    _rain = {"ma": 3, "di": 0, "wo": 0, "do": 5}
    _dry = [_d for _d in _rain if _rain[_d] == 0]
    print(_dry)
    # 5. accolades, een key en een waarde: elk woord met zijn lengte
    _words = ["zon", "wolk", "regen"]
    _lengths = {_w: len(_w) for _w in _words}
    print(_lengths)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen vier regels. De eerste lijst heeft één getal voor elke temperatuur: 0 graden Celsius is 32,0 in Fahrenheit. De tweede houdt alleen 25 en 30, omdat de `if` alleen voor die twee `True` is. De derde gaat de keys van `_rain` langs en houdt de dagen waarvan de waarde 0 is. De laatste regel is een dictionary, met elk woord als key en zijn lengte als waarde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lengths():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Woordlengtes

        Maak `lengths`: de lengte van elk woord in `words`, als lijst. Vervang de `...` door een comprehension. De check probeert ook een andere lijst.
        """),
        L.task("Maak `lengths` in één regel, zodat het programma `[5, 2, 4]` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lengths():
    box_lengths = L.code_box("lengths", lines=4)
    box_lengths
    return (box_lengths,)


@app.cell(hide_code=True)
def check_lengths(box_lengths):
    result_lengths = L.check("lengths", box_lengths.value)
    result_lengths
    return (result_lengths,)


@app.cell(hide_code=True)
def hints_lengths():
    L.hints(
        "lengths",
        [
            "Kijk naar de eerste comprehension van het uitgewerkte voorbeeld: vóór `for` staat wat er in de nieuwe lijst komt.",
            "In de nieuwe lijst komt de lengte van één woord. Welke instructie geeft de lengte van een string?",
            "`lengths = [___ for word in words]`",
        ],
        why="`[len(word) for word in words]` gaat `words` langs; voor elk woord rekent hij `len(word)` uit en zet dat in de nieuwe lijst: 5, 2 en 4.",
    )
    return


@app.cell(hide_code=True)
def text_dict_comp():
    mo.md(r"""
    ### Oefening 2. Een dictionary in één regel
    """)
    return


@app.cell(hide_code=True)
def choice_dict_comp_cell():
    choice_dict_comp = L.choice(
        "dict-comp",
        question="Wat laat dit programma zien?\n\n```python\nwords = [\"hi\", \"hello\"]\nprint({w: len(w) for w in words})\n```",
        options={
            "list": "`[2, 5]`",
            "dict": "`{'hi': 2, 'hello': 5}`",
            "set": "`{2, 5}`",
        },
        correct="dict",
        why={
            "list": "Vierkante haken zouden een lijst maken, maar dit zijn accolades met een dubbele punt. Kijk nog eens naar stap 5 van het recept.",
            "dict": "Accolades met vooraan `key: waarde` maken een dictionary: elk woord als key, en zijn lengte als waarde.",
            "set": "Kijk naar wat er vóór `for` staat: daar zit een dubbele punt in. Kijk nog eens naar stap 5 van het recept.",
        },
    )
    choice_dict_comp
    return (choice_dict_comp,)


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wie heeft een voldoende?

        `scores` geeft de score van elke leerling. Een score van 5,5 of meer is voldoende. Maak `passed`: de namen van iedereen met een voldoende, als lijst. Vervang de `...` door een comprehension met een `if`. De check probeert ook andere scores.
        """),
        L.task("Maak `passed` in één regel, zodat het programma `['Ada', 'Cas', 'Dex']` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed():
    box_passed = L.code_box("passed", lines=4)
    box_passed
    return (box_passed,)


@app.cell(hide_code=True)
def check_passed(box_passed):
    result_passed = L.check("passed", box_passed.value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "Een `for` over een dictionary gaat de keys langs, hier de namen. Wat komt er in de nieuwe lijst: de naam, of de score?",
            "Stap 4 van het recept, en de derde comprehension van het uitgewerkte voorbeeld: de `if` komt aan het eind. De score van een naam is `scores[name]`. Welke vergelijking telt precies 5,5 ook mee?",
            "`passed = [name for name in scores if ___]`",
        ],
        why="`for name in scores` gaat de namen langs. `if scores[name] >= 5.5` houdt alleen de namen met een score van 5,5 of meer, en `name` vooraan zet die namen in de lijst. Dex heeft precies 5,5, dus `>=` houdt Dex ook.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Herhalen: for-loops](/course/open/py-for?lang=nl) telde je op en telde je met een loop. Met `append` uit [Lijsten](/course/open/py-lists?lang=nl) kan zo'n loop ook een nieuwe lijst bouwen, zoals in de figuur; een comprehension doet dat in één regel, en met accolades bouwt hij een [dictionary](/course/open/py-dicts?lang=nl). Later in de course kom je ze steeds tegen in code die data klaarzet, bijvoorbeeld om elk woord van een tekst om te zetten in zijn nummer. In de volgende les leer je de foutmeldingen van Python lezen, en een bug vinden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dict_comp, result_lengths, result_passed):
    L.wrap_up(result_lengths, choice_dict_comp, result_passed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
