# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-classes. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-classes", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Dictionaries](/course/open/py-dicts?lang=nl) en [Functies](/course/open/py-functions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="Wat laat dit programma zien?\n\n```python\npoint = {\"x\": 3, \"y\": 4}\nprint(point[\"y\"])\n```",
        options={
            "four": "`4`",
            "three": "`3`",
            "key": "`y`",
        },
        correct="four",
        why={
            "four": "`point[\"y\"]` zoekt de waarde op onder de key `\"y\"`: 4.",
            "three": "3 is de waarde onder `\"x\"`. Zie het recept in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "key": "Vierkante haken geven de waarde onder een key, niet de key zelf. Zie de figuur in [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_area():
    quick_area = L.choice(
        "quick-area",
        question="Wat laat dit programma zien?\n\n```python\ndef area(w, h):\n    return w * h\n\nprint(area(2, 5) + area(1, 1))\n```",
        options={
            "ten": "`10`",
            "twelve": "`12`",
            "eleven": "`11`",
        },
        correct="eleven",
        why={
            "ten": "Dat is alleen de eerste aanroep. `print` laat de som van beide aanroepen zien. Zie de gok in [Functies](/course/open/py-functions?lang=nl).",
            "twelve": "Elke aanroep vermenigvuldigt zijn eigen twee argumenten: 2 keer 5, en 1 keer 1. Zie het idee in [Functies](/course/open/py-functions?lang=nl).",
            "eleven": "`area(2, 5)` geeft 10 terug en `area(1, 1)` geeft 1 terug, dus de som is 11.",
        },
    )
    quick_area
    return (quick_area,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Data komt vaak in bundels. Een bankrekening heeft een eigenaar en een saldo, en je zet er geld op of haalt het eraf. Later in de course heeft een layer van een neural network zijn getallen en een manier om ermee te rekenen. Een class houdt zulke data bij elkaar, samen met de functies die erop werken. In PyTorch is elke soort layer, en elk model, een class.

    **Een voorbeeld uit het dagelijks leven.** Het plan van een huis, en de huizen die ernaar gebouwd zijn. Eén plan, veel huizen: elk huis is op dezelfde manier gebouwd, maar elk heeft zijn eigen meubels. Een class is het plan; een object is één huis.
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n\na = Dog(\"Rex\")\nb = Dog(\"Bo\")\nprint(a.name)\n```",
        options={
            "bo": "`Bo`",
            "rex": "`Rex`",
            "both": "`Rex` en `Bo`",
        },
        reveal="**`Rex`.** `Dog(\"Rex\")` maakt een nieuwe hond en geeft hem de naam `\"Rex\"`; `Dog(\"Bo\")` maakt er nog een, met een eigen naam. `a.name` is de naam van de hond in `a`.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een class is een plan voor een nieuwe soort waarde: de data ervan, en de functies die op die data werken. Elk object dat je ervan maakt, heeft zijn eigen data.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **class** | een plan voor een nieuwe soort waarde: de data ervan, en de functies die erop werken | `class Account:` |
    | **object** | één waarde die van een class gemaakt is; elk object heeft zijn eigen data | `a = Account("Ada")` |
    | **`__init__`** | de method die draait als een object gemaakt wordt; die geeft het object zijn begindata | `def __init__(self, owner):` |
    | **self** | in een method: het object zelf; `self.balance` is de eigen data van dat object | `self.balance = 0` |
    | **attribute** | een stukje data van een object, bereikt met een punt | `a.balance` |
    | **method** | een functie in een class; je roept hem aan op een object, met een punt | `a.deposit(50)` |

    `Account("Ada")` maakt een nieuw object, en voert `__init__` uit met `self` als dat nieuwe object en `owner` als `"Ada"`. `self` geef je nooit zelf mee: dat doet Python. Zo voert `a.deposit(50)` ook `deposit` uit met `self` als `a`. Volgens afspraak begint de naam van een class met een hoofdletter.
    """)
    return


@app.cell(hide_code=True)
def picture_objects():
    mo.vstack([
        draw.class_objects(["class Account", "__init__(self, owner)", "deposit(self, amount)"],
                           [("a = Account(\"Ada\")", ["owner: 'Ada'", "balance: 50"]),
                            ("b = Account(\"Bo\")", ["owner: 'Bo'", "balance: 0"])],
                           class_label="De class: het plan", objects_label="Twee objecten, elk met eigen data",
                           title="Eén class, twee objecten"),
        mo.md(r"""
        Beide objecten komen van hetzelfde plan, met dezelfde methods. Maar elk heeft zijn eigen `owner` en zijn eigen `balance`: na `a.deposit(50)` heeft alleen `a` een saldo van 50.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Elke method krijgt `self` als eerste argument, en bereikt de data van het object daarmee. In een method is `balance` zonder `self.` ervoor een nieuwe naam, die verdwijnt als de method klaar is; `self.balance` blijft bij het object."),
        mo.md(r"""
        ### Recept

        Zo maak je je eigen soort waarde:

        1. Schrijf `class`, een naam met een hoofdletter, en een dubbele punt: `class Account:`.
        2. Ingesprongen eronder geeft `def __init__(self, ...):` een nieuw object zijn begindata, met `self.name = ...`.
        3. Voeg methods toe, `def deposit(self, amount):`, die via `self` op de data van het object werken.
        4. Maak objecten door de class aan te roepen, `a = Account("Ada")`; roep methods aan met een punt, `a.deposit(50)`; lees data met een punt, `a.balance`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een bankrekening met een eigenaar en een saldo. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_account():
    # 1. de class: een plan, met een hoofdletter
    class _Account:
        # 2. __init__ geeft elk nieuw object zijn begindata
        def __init__(self, owner):
            self.owner = owner
            self.balance = 0
            self.deposits = []

        # 3. methods werken via self op de eigen data van het object
        def deposit(self, amount):
            self.balance = self.balance + amount
            self.deposits.append(amount)

        def count_deposits(self):
            return len(self.deposits)

        def report(self):
            return f"{self.owner} heeft {self.balance} euro"

    # 4. maak objecten, roep methods aan, lees data
    _a = _Account("Ada")
    _b = _Account("Bo")
    _a.deposit(50)
    _a.deposit(25)
    print(_a.balance)
    print(_b.balance)
    print(_a.report())
    print(_a.deposits)
    print(_a.count_deposits())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `75`, `0`, `Ada heeft 75 euro`, `[50, 25]` en `2`. Beide stortingen gingen naar de rekening van Ada; het saldo van Bo is nog steeds 0, omdat elk object zijn eigen `balance` heeft. Elke rekening houdt ook een eigen lijst bij, `self.deposits`: `deposit` voegt er iets aan toe met `append`, en `count_deposits` geeft de lengte ervan terug. De naam van de class begint met `_`, omdat alleen deze cel hem gebruikt, net als bij elke andere naam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counter():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een teller

        Maak de class `Counter` af: een nieuwe teller begint bij 0 in `self.count`, en elke aanroep van `add` telt er 1 bij op. Vervang de twee `...`. De check maakt een teller, roept `add` twee keer aan en leest `count`: dat moet 2 zijn. Hij probeert ook meer, zoals twee tellers tegelijk.
        """),
        L.task("Laat `Counter` bij 0 beginnen en bij elke `add` 1 optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counter():
    box_counter = L.code_box("counter", lines=8)
    box_counter
    return (box_counter,)


@app.cell(hide_code=True)
def check_counter(box_counter):
    result_counter = L.check("counter", box_counter.value)
    result_counter
    return (result_counter,)


@app.cell(hide_code=True)
def hints_counter():
    L.hints(
        "counter",
        [
            "Stap 2 van het recept: wat is de telling van een teller die nog niets geteld heeft?",
            "In `add` krijgt de telling van het object een nieuwe waarde: wat hij al was, plus 1. Je bereikt hem via `self`.",
            "`self.count = 0` in `__init__`, en in `add` `self.count = ___ + 1`.",
        ],
        why="`self.count = 0` geeft elke nieuwe teller een eigen telling, die bij 0 begint. `self.count = self.count + 1` in `add` telt 1 op bij de telling van dat object, en van geen ander.",
    )
    return


@app.cell(hide_code=True)
def text_two_objects():
    mo.md(r"""
    ### Oefening 2. Twee objecten
    """)
    return


@app.cell(hide_code=True)
def choice_two_objects_cell():
    choice_two_objects = L.choice(
        "two-objects",
        question="Wat laat dit programma zien?\n\n```python\nclass Lamp:\n    def __init__(self):\n        self.on = False\n\n    def switch(self):\n        self.on = True\n\nkitchen = Lamp()\nhall = Lamp()\nkitchen.switch()\nprint(hall.on)\n```",
        options={
            "true": "`True`",
            "false": "`False`",
            "error": "Het stopt: `hall` is nooit aangezet",
        },
        correct="false",
        why={
            "true": "`switch` verandert `self.on` van het object waarop hij wordt aangeroepen. Welk object was dat? Kijk nog eens naar de figuur.",
            "false": "`kitchen.switch()` verandert alleen de eigen `on` van `kitchen`. `hall` heeft nog steeds de `False` die hij van `__init__` kreeg.",
            "error": "`hall` kreeg zijn eigen `on` in `__init__`, toen hij gemaakt werd. Kijk nog eens naar stap 2 van het recept.",
        },
    )
    choice_two_objects
    return (choice_two_objects,)


@app.cell(hide_code=True)
def text_exercise_playlist():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een playlist

        Maak de class `Playlist` af: elke playlist heeft een naam en een eigen lijst met nummers. `add` zet een nummer achteraan die lijst, en `length` geeft terug hoeveel nummers hij heeft. Vervang de drie `...`. De check maakt een playlist, voegt twee nummers toe en roept `length` aan: dat moet 2 zijn. Hij probeert ook meer, zoals twee playlists tegelijk.
        """),
        L.task("Laat `Playlist` zijn eigen nummers bewaren, ze toevoegen, en zijn lengte teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_playlist():
    box_playlist = L.code_box("playlist", lines=11)
    box_playlist
    return (box_playlist,)


@app.cell(hide_code=True)
def check_playlist(box_playlist):
    result_playlist = L.check("playlist", box_playlist.value)
    result_playlist
    return (result_playlist,)


@app.cell(hide_code=True)
def hints_playlist():
    L.hints(
        "playlist",
        [
            "Stap 2 van het recept: een nieuwe playlist heeft nog geen nummers. Hoe ziet een lege lijst eruit?",
            "In `add` komt het nummer achteraan de lijst van het object, `self.songs`. Geef in `length` de lengte van die lijst terug.",
            "`self.songs = []`, `self.songs.append(___)`, en `return len(___)`.",
        ],
        why="`self.songs = []` in `__init__` geeft elke playlist een eigen, lege lijst. `self.songs.append(song)` voegt een nummer aan die lijst toe, en `return len(self.songs)` geeft terug hoeveel erin staan.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) kreeg een functie zijn data als argumenten; een method krijgt zijn object als `self`, en vindt de data daar. In [Dictionaries](/course/open/py-dicts?lang=nl) bewaarde je waarden onder keys; een object bewaart ze als attributes, samen met de methods die ze gebruiken. Later in de course is elke layer van een netwerk een object dat van zo'n class gemaakt is. In de volgende les laat je `+` en `print` werken op je eigen objecten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_two_objects, result_counter, result_playlist):
    L.wrap_up(result_counter, choice_two_objects, result_playlist)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
