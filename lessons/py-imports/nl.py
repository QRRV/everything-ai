# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-imports. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-imports", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Functies](/course/open/py-functions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_greet():
    quick_greet = L.choice(
        "quick-greet",
        question="Wat laat dit programma zien?\n\n```python\ndef greet(name):\n    return f\"Hi {name}\"\n\nmessage = greet(\"Ada\")\nprint(message)\n```",
        options={
            "hi-ada": "`Hi Ada`",
            "none": "`None`",
            "hi-name": "`Hi name`",
        },
        correct="hi-ada",
        why={
            "hi-ada": "`greet(\"Ada\")` geeft de f-string terug met `Ada` ingevuld, en `message` bewaart hem.",
            "none": "`greet` heeft een `return`, dus hij geeft wel iets terug. Zie het blok *Let op* in [Functies](/course/open/py-functions?lang=nl).",
            "hi-name": "In de functie heeft `name` de waarde van het argument. Zie de figuur in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_greet
    return (quick_greet,)


@app.cell(hide_code=True)
def quick_check_inside():
    quick_inside = L.choice(
        "quick-inside",
        question="Wat gebeurt er als dit programma draait?\n\n```python\ndef double(x):\n    result = x * 2\n    return result\n\ndouble(4)\nprint(result)\n```",
        options={
            "eight": "Het laat `8` zien",
            "error": "Het stopt: Python kent `result` niet op de laatste regel",
            "four": "Het laat `4` zien",
        },
        correct="error",
        why={
            "eight": "`result` bestaat alleen zolang `double` draait. Zie het idee in [Functies](/course/open/py-functions?lang=nl).",
            "error": "`result` wordt in de functie gemaakt en bestaat alleen zolang die draait; daarbuiten kent Python de naam niet.",
            "four": "`result` is nooit 4, en buiten de functie bestaat hij helemaal niet. Zie het idee in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_inside
    return (quick_inside,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hoeft niet alles zelf te schrijven. Python heeft honderden modules vol kant-en-klare code: `math` voor wortels en pi, `random` voor toeval, en nog veel meer. Later in de course zijn NumPy en PyTorch ook modules, en die haal je op dezelfde manier binnen. En toeval speelt een grote rol bij programma's die leren: een seed maakt "willekeurig" herhaalbaar, zodat een experiment opnieuw dezelfde uitkomst geeft.

    **Een voorbeeld uit het dagelijks leven.** Een gereedschapskist. Je smeedt niet elke keer een nieuwe hamer als je er een nodig hebt: je pakt hem uit de kist. `import` haalt de kist binnen in je programma, en de punt pakt er één stuk gereedschap uit: `math.sqrt`.
    """)
    return


@app.cell(hide_code=True)
def predict_sqrt():
    guess_sqrt = L.guess(
        "sqrt",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nimport math\nprint(math.sqrt(16))\n```",
        options={
            "four": "`4`",
            "sixteen": "`16`",
            "float": "`4.0`",
        },
        reveal="**`4.0`.** `math.sqrt` uit de module `math` geeft de wortel: het getal dat keer zichzelf 16 geeft. Hij geeft altijd een float terug, dus Python laat `4.0` zien.",
    )
    guess_sqrt
    return (guess_sqrt,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een module is een bestand met kant-en-klare code. `import` haalt het binnen in je programma; daarna gebruik je wat erin zit met de naam van de module en een punt.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **module** | een bestand met kant-en-klare code, zoals `math` of `random` | |
    | **import** | haalt een module binnen in je programma; die regel staat bovenaan | `import math` |
    | **punt** | pakt een naam uit een module | `math.sqrt(16)` geeft `4.0` |
    | **from ... import** | haalt één naam binnen, zodat je hem zonder de naam van de module kunt gebruiken | `from math import pi` |
    | **seed** | het beginpunt van de willekeurige getallen: dezelfde seed geeft weer dezelfde getallen | `random.seed(42)` |

    In `math`: `math.sqrt(x)` geeft de wortel, `math.pi` is pi, 3,14159..., en `math.floor(7.8)` rondt naar beneden af tot `7`. In `random`: `random.randint(1, 6)` geeft een geheel getal van 1 tot en met 6, en `random.choice(items)` kiest één item uit een lijst. De getallen van `random` zijn niet echt willekeurig: ze komen uit een vaste berekening die bij de seed begint.
    """)
    return


@app.cell(hide_code=True)
def picture_module():
    mo.vstack([
        draw.flow(
            [("Je programma", ["import math", "math.sqrt(16)"]),
             ("De module math", ["sqrt", "pi", "floor", "..."])],
            forward=["vraagt math\nom sqrt(16)"],
            back=["4.0 komt terug"],
            title="Een aanroep naar een module"),
        mo.md(r"""
        `math.sqrt(16)` is een aanroep, net als een aanroep van je eigen functie: de punt zegt dat `sqrt` in `math` woont. De functie in de module doet het werk, en de uitkomst komt terug in je programma.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Na `import math` zet je voor elke naam uit de module `math.`: `sqrt(16)` zonder `math.` stopt met een fout, omdat Python `sqrt` niet kent. En bij `random.randint(1, 6)` doet 6 zelf mee, terwijl `range(1, 6)` vóór 6 stopt."),
        mo.md(r"""
        ### Recept

        Zo gebruik je code uit een module:

        1. Schrijf bovenaan `import` en de naam van de module: `import math`.
        2. Gebruik een naam eruit met de naam van de module en een punt: `math.sqrt(16)`.
        3. Gebruik je één naam vaak? `from math import pi` haalt hem binnen, en dan schrijf je alleen `pi`.
        4. Voor toeval: eerst `random.seed(...)` als je dezelfde getallen nog eens wilt, dan `random.randint(a, b)` of `random.choice(items)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een paar dingen uit `math`, en dobbelsteenworpen met een seed. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_modules():
    # 1. import bovenaan
    import math
    import random
    from math import pi
    # 2. de naam van de module, een punt, en de naam erin
    print(math.sqrt(2))
    print(math.floor(7.8))
    print(math.pi)
    # 3. een naam binnengehaald met from: geen math. ervoor
    print(pi)
    # 4. een seed, en dan drie worpen
    random.seed(7)
    _rolls = []
    for _i in range(3):
        _rolls.append(random.randint(1, 6))
    print(_rolls)
    # dezelfde seed nog eens geeft dezelfde worpen
    random.seed(7)
    _again = []
    for _i in range(3):
        _again.append(random.randint(1, 6))
    print(_again)

    # 4. in een functie: dezelfde seed geeft dezelfde keuze
    def _coin(seed):
        random.seed(seed)
        return random.choice(["kop", "munt"])

    print(_coin(5))
    print(_coin(5))
    return math, pi, random


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `1.4142135623730951`, `7`, twee keer `3.141592653589793`, twee keer `[3, 2, 4]` en twee keer `munt`. De twee lijsten met worpen zijn gelijk, omdat ze allebei bij seed 7 beginnen; zonder de seed zouden ze meestal elke keer dat je de cel uitvoert anders zijn. `_coin` zet de seed in de functie, dus geeft dezelfde seed dezelfde keuze. Een `import` maakt een naam, net als `=`: deze cel maakt `math`, `random` en `pi`. Een andere cel kan meteen `math.sqrt(25)` gebruiken, maar `import math` in een andere cel botst met *één naam, één cel*; probeer dingen uit in het kladblok. De oefenvakken draaien los, dus elk vak heeft zijn eigen regel met `import`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_circle():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De oppervlakte van een cirkel

        Maak `circle_area(r)` af: de oppervlakte van een cirkel met straal `r` is pi keer `r` in het kwadraat. Gebruik `math.pi`; het vak begint al met `import math`. De check roept hem aan met 3, en met andere stralen.
        """),
        L.task("Laat `circle_area(r)` pi keer `r` in het kwadraat teruggeven, met `math.pi`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_circle():
    box_circle = L.code_box("circle", lines=5)
    box_circle
    return (box_circle,)


@app.cell(hide_code=True)
def check_circle(box_circle):
    result_circle = L.check("circle", box_circle.value)
    result_circle
    return (result_circle,)


@app.cell(hide_code=True)
def hints_circle():
    L.hints(
        "circle",
        [
            "Stap 2 van het recept: pi zit in de module `math`. Hoe schrijf je hem, met de naam van de module?",
            "`r` in het kwadraat is `r ** 2`, of `r * r`.",
            "`return math.___ * r ** 2`",
        ],
        why="`math.pi * r ** 2` neemt de precieze pi uit `math`, keer `r` in het kwadraat. Bij `r = 1` is dat pi zelf: 3,14159...",
    )
    return


@app.cell(hide_code=True)
def text_import_name():
    mo.md(r"""
    ### Oefening 2. Zonder de naam van de module
    """)
    return


@app.cell(hide_code=True)
def choice_import_name_cell():
    choice_import_name = L.choice(
        "import-name",
        question="Wat gebeurt er als dit programma draait?\n\n```python\nimport math\nprint(sqrt(9))\n```",
        options={
            "float": "Het laat `3.0` zien",
            "whole": "Het laat `3` zien",
            "error": "Het stopt: Python kent `sqrt` niet",
        },
        correct="error",
        why={
            "error": "`import math` haalt de module binnen, niet de namen erin: `sqrt` woont in `math`, dus hij heeft `math.` ervoor nodig.",
            "float": "Kijk hoe de aanroep hier geschreven is, en naar het blok *Let op* boven het recept.",
            "whole": "`math.sqrt` zou een float geven. Maar kijk hoe de aanroep hier geschreven is, en naar het blok *Let op* boven het recept.",
        },
    )
    choice_import_name
    return (choice_import_name,)


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een herhaalbare keuze

        Maak `pick(items, seed)` af: hij zet de seed met `random.seed`, met zijn argument `seed`, en kiest daarna één item uit `items` met `random.choice`. Met dezelfde seed kiest hij altijd hetzelfde item. De check roept hem aan met `["rock", "paper", "scissors"]` en seed 1, en met andere lijsten en seeds.
        """),
        L.task("Laat `pick(items, seed)` een item kiezen dat bij dezelfde seed elke keer hetzelfde is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=6)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "Stap 4 van het recept, en `_coin` in het uitgewerkte voorbeeld: eerst de seed, dan de keuze.",
            "Zet op de eerste regel de seed met het argument `seed`; geef op de tweede één item terug, gekozen uit `items`.",
            "`random.seed(___)`, dan `return random.choice(___)`",
        ],
        why="`random.seed(seed)` zet het beginpunt, dus kiest `random.choice(items)` daarna bij dezelfde seed altijd hetzelfde item. Zo kun je een experiment met toeval herhalen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) schreef je je eigen functies; een module is een verzameling functies die anderen schreven, en `math.sqrt(16)` is een aanroep, net als die van jou. Later in de course haal je NumPy en PyTorch op dezelfde manier binnen, en maakt een seed het trainen herhaalbaar. In de volgende les zet je je eigen code in een module.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import_name, result_circle, result_pick):
    L.wrap_up(result_circle, choice_import_name, result_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
