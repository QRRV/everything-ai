# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-special-methods. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-special-methods", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Classes en objecten](/course/open/py-classes?lang=nl) en [Tekst: strings](/course/open/py-strings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grow():
    quick_grow = L.choice(
        "quick-grow",
        question="Wat laat dit programma zien?\n\n```python\nclass Box:\n    def __init__(self, size):\n        self.size = size\n\n    def grow(self):\n        self.size = self.size * 2\n\nb = Box(3)\nb.grow()\nb.grow()\nprint(b.size)\n```",
        options={
            "six": "`6`",
            "three": "`3`",
            "twelve": "`12`",
        },
        correct="twelve",
        why={
            "six": "Dat is na één `grow`. `b.grow()` draait twee keer, en elke keer verandert hij de eigen `size` van `b`. Zie het recept in [Classes en objecten](/course/open/py-classes?lang=nl).",
            "three": "`grow` verandert `self.size`, en `self` is `b`, dus verandert `b.size`. Zie het blok *Let op* in [Classes en objecten](/course/open/py-classes?lang=nl).",
            "twelve": "`Box(3)` begint met grootte 3; elke `grow` verdubbelt de eigen `size` van `b`: 6, en dan 12.",
        },
    )
    quick_grow
    return (quick_grow,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="Wat laat dit programma zien?\n\n```python\nx = 2\ny = 5\nprint(f\"({x}, {y})\")\n```",
        options={
            "values": "`(2, 5)`",
            "names": "`(x, y)`",
            "braces": "`({x}, {y})`",
        },
        correct="values",
        why={
            "values": "In een f-string wordt elke `{...}` vervangen door zijn waarde; de haakjes en de komma zijn gewoon tekst.",
            "names": "De namen tussen `{}` zie je niet terug: Python vult hun waarde in. Zie het idee in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "braces": "Door de `f` ervoor is het een f-string, dus worden de `{...}` ingevuld. Zie het recept in [Tekst: strings](/course/open/py-strings?lang=nl).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De waarden die Python zelf kent, werken met tekens: `2 + 3` telt op, en `print` laat een getal of een lijst netjes zien. Je eigen objecten kunnen dat nog niet: `print` laat iets zien als `<__main__.Account object at 0x7f3a2c>`, de eigen tekst van Python voor een object, met zijn class en de plek in het geheugen. En `+` stopt met een fout. Special methods lossen dat op. Later, in PyTorch, roep je een model aan als een functie, `model(x)`, en ook dat werkt via een special method.

    **Een voorbeeld uit het dagelijks leven.** De aan-uitknop op een afstandsbediening. Op elke afstandsbediening druk je hem op dezelfde manier in, maar wat er gebeurt, hangt af van het apparaat: de tv gaat aan, of de radio. `+` is zo'n knop, en je class bepaalt wat hij doet.
    """)
    return


@app.cell(hide_code=True)
def predict_money():
    guess_money = L.guess(
        "money",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nclass Money:\n    def __init__(self, cents):\n        self.cents = cents\n\n    def __add__(self, other):\n        return Money(self.cents + other.cents)\n\n    def __str__(self):\n        return f\"{self.cents} cent\"\n\ntotal = Money(250) + Money(75)\nprint(total)\n```",
        options={
            "text": "`Money(250) + Money(75)`",
            "error": "Het stopt: Python kan geen twee objecten optellen",
            "cents": "`325 cent`",
        },
        reveal="**`325 cent`.** `+` tussen twee `Money`-objecten voert `__add__` uit, en die maakt een nieuwe `Money` met 250 + 75 cent. Daarna voert `print` de `__str__` van dat nieuwe object uit, om de tekst te krijgen die hij laat zien.",
    )
    guess_money
    return (guess_money,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een special method heeft een naam met twee lage streepjes aan elke kant. Je roept hem niet zelf aan: Python roept hem aan als je `+` of `print` gebruikt, of een object aanroept. Eén ken je al: `__init__`, die Python uitvoert als je een object maakt.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **special method** | een method met twee lage streepjes aan elke kant van zijn naam; Python roept hem voor je aan | `def __add__(self, other):` |
    | **`__add__`** | draait bij `+`: `a + b` voert `a.__add__(b)` uit, met `self` als `a` en `other` als `b` | `def __add__(self, other):` |
    | **`__str__`** | geeft de tekst terug die `print` en een f-string voor het object laten zien | `def __str__(self):` |
    | **`__call__`** | draait als je het object aanroept als een functie: `double(21)` voert `double.__call__(21)` uit | `def __call__(self, x):` |

    `__add__` geeft meestal een nieuw object terug, en laat `a` en `b` zoals ze waren, net zoals `2 + 3` de 2 en de 3 met rust laat. `__str__` moet een string teruggeven met `return`. Programmeurs noemen deze methods vaak *dunder methods*: dunder is kort voor *double underscore*, dubbel laag streepje.
    """)
    return


@app.cell(hide_code=True)
def picture_calls():
    mo.vstack([
        draw.dict_boxes([("a + b", "a.__add__(b)"), ("print(a)", "a.__str__()"), ("f\"{a}\"", "a.__str__()"), ("double(21)", "double.__call__(21)")],
                        key_label="Wat jij schrijft", value_label="Wat Python uitvoert",
                        title="Wat je methods aanroept"),
        mo.md(r"""
        Links staat wat jij schrijft; rechts de method die Python daarvoor uitvoert. Het object links van `+` is `self`; het object rechts is `other`. `print` en een f-string vragen het object allebei om zijn tekst, via `__str__`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`__str__` moet de tekst teruggeven met `return`. Staat er `print` in `__str__`, dan geeft de method niets terug, en stopt `print(a)` met een fout, omdat hij geen tekst krijgt."),
        mo.md(r"""
        ### Recept

        Zo laat je `+`, `print` en aanroepen werken op je objecten:

        1. Schrijf in je class een method waarvan de naam twee lage streepjes aan elke kant heeft; `self` is nog steeds zijn eerste argument.
        2. Voor `+`: `def __add__(self, other):`, en geef een nieuw object terug dat van allebei gemaakt is: `return Money(self.cents + other.cents)`.
        3. Voor `print` en f-strings: `def __str__(self):`, en geef een string terug, vaak een f-string: `return f"{self.cents} cent"`.
        4. Voor aanroepen: `def __call__(self, x):`, en geef terug wat de aanroep oplevert. Dan werkt een object van die class als een functie: `double(21)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Geld dat optelt en zichzelf laat zien, en een vermenigvuldiger die je aanroept als een functie. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_special():
    # 1. methods met twee lage streepjes aan elke kant van hun naam
    class _Money:
        def __init__(self, cents):
            self.cents = cents

        # 2. + geeft een nieuw object terug, gemaakt van self en other
        def __add__(self, other):
            return _Money(self.cents + other.cents)

        # 3. print en f-strings halen de tekst uit __str__
        def __str__(self):
            return f"{self.cents} cent"

    class _Scale:
        def __init__(self, factor):
            self.factor = factor

        # 4. het object aanroepen voert __call__ uit
        def __call__(self, x):
            return self.factor * x

    _a = _Money(250)
    _b = _Money(75)
    _total = _a + _b
    print(_total)
    print(_a)
    print(f"Samen: {_total}")
    _double = _Scale(2)
    print(_double(21))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `325 cent`, `250 cent`, `Samen: 325 cent` en `42`. `_a + _b` maakte een nieuw object, dus heeft `_a` nog steeds zijn 250 cent. De f-string gebruikte ook `__str__`. En `_double(21)` voerde `__call__` uit, met `x` als 21.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vec():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een paar dat optelt

        Een `Vec` is een paar getallen, `x` en `y`. Maak `__add__` af, die een nieuwe `Vec` teruggeeft: zijn `x` is de `x` van beide paren bij elkaar opgeteld, en net zo voor `y`. En maak `__str__` af, die de tekst `(x, y)` teruggeeft. Vervang de twee `...`. De check maakt `u = Vec(1, 2)` en `v = Vec(3, 4)`, en laat `u + v` zien met een f-string: dat moet `(4, 6)` zijn. Hij probeert ook meer, zoals of `u` en `v` hetzelfde blijven na `u + v`.
        """),
        L.task("Zorg dat `+` een nieuwe `Vec` geeft, en dat `print` hem laat zien als `(4, 6)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vec():
    box_vec = L.code_box("vec", lines=11)
    box_vec
    return (box_vec,)


@app.cell(hide_code=True)
def check_vec(box_vec):
    result_vec = L.check("vec", box_vec.value)
    result_vec
    return (result_vec,)


@app.cell(hide_code=True)
def hints_vec():
    L.hints(
        "vec",
        [
            "Stap 2 en 3 van het recept, en `_Money` in het uitgewerkte voorbeeld: `__add__` maakt een nieuw object, en `__str__` geeft een f-string terug.",
            "In `__add__` is de nieuwe `x` gelijk aan `self.x` plus `other.x`, en net zo voor `y`. Zet in `__str__` de eigen `x` en `y` van het object tussen haakjes.",
            "`return Vec(___, ___)` in `__add__`, en `return f\"({___}, {___})\"` in `__str__`.",
        ],
        why="`return Vec(self.x + other.x, self.y + other.y)` maakt een nieuw paar van allebei, en laat `self` en `other` zoals ze waren. `return f\"({self.x}, {self.y})\"` geeft de tekst terug die `print` en f-strings laten zien.",
    )
    return


@app.cell(hide_code=True)
def text_add_order():
    mo.md(r"""
    ### Oefening 2. Welke is self?
    """)
    return


@app.cell(hide_code=True)
def choice_add_order_cell():
    choice_add_order = L.choice(
        "add-order",
        question="Wat laat dit programma zien?\n\n```python\nclass Word:\n    def __init__(self, text):\n        self.text = text\n\n    def __add__(self, other):\n        return Word(f\"{other.text}-{self.text}\")\n\nw = Word(\"appel\") + Word(\"taart\")\nprint(w.text)\n```",
        options={
            "left-first": "`appel-taart`",
            "right-first": "`taart-appel`",
            "error": "Het stopt: `Word` heeft geen `__str__`",
        },
        correct="right-first",
        why={
            "left-first": "Dan zou `self.text` voorop staan. Welk object is in `a + b` `self`, en welk `other`? Kijk nog eens naar de figuur.",
            "right-first": "`Word(\"appel\") + Word(\"taart\")` voert `__add__` uit met `self` als het woord appel en `other` als het woord taart. De method zet `other.text` voorop: `taart-appel`.",
            "error": "`print(w.text)` laat het attribute `text` zien, en dat is al een string. `__str__` is alleen nodig voor `print(w)` zelf.",
        },
    )
    choice_add_order
    return (choice_add_order,)


@app.cell(hide_code=True)
def text_exercise_line():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een lijn die je aanroept

        Een rechte lijn heeft een helling `w` en een beginwaarde `b`: bij elke `x` geeft hij `w * x + b`. Maak `__call__` af, zodat je een lijn kunt aanroepen als een functie, en `__str__`, die zijn formule teruggeeft, zoals `y = 2x + 1`. Vervang de twee `...`. De check maakt `line = Line(2, 1)`, roept `line(3)` aan, wat 7 moet geven, en laat `Line(2, 1)` zien met een f-string: `y = 2x + 1`. Hij probeert ook meer, zoals twee lijnen tegelijk.
        """),
        L.task("Zorg dat bij `line = Line(2, 1)` de aanroep `line(3)` 7 geeft, en dat `print(line)` `y = 2x + 1` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line():
    box_line = L.code_box("line", lines=11)
    box_line
    return (box_line,)


@app.cell(hide_code=True)
def check_line(box_line):
    result_line = L.check("line", box_line.value)
    result_line
    return (result_line,)


@app.cell(hide_code=True)
def hints_line():
    L.hints(
        "line",
        [
            "Stap 4 van het recept, en `_Scale` in het uitgewerkte voorbeeld: `__call__` geeft terug wat de aanroep moet opleveren.",
            "Gebruik in `__call__` de eigen `self.w` en `self.b` van de lijn, en de `x` die hij krijgt. Zet in `__str__` `self.w` en `self.b` in een f-string.",
            "`return self.w * ___ + ___` in `__call__`, en `return f\"y = {___}x + {___}\"` in `__str__`.",
        ],
        why="`return self.w * x + self.b` rekent de lijn uit bij `x`, met de eigen getallen van deze lijn. `return f\"y = {self.w}x + {self.b}\"` geeft zijn formule terug, dus laat `print(line)` `y = 2x + 1` zien.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Classes en objecten](/course/open/py-classes?lang=nl) voerde Python `__init__` voor je uit als je een object maakte; nu weet je dat Python hetzelfde doet met `__add__`, `__str__` en `__call__`. Je `Line` is al een heel klein model: twee getallen, `w` en `b`, en je roept hem aan op een `x`. Later in de course roep je een PyTorch-model precies zo aan, `model(x)`, en leert het model zelf goede getallen. In de volgende les bouwt een class voort op een andere class: inheritance.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add_order, result_line, result_vec):
    L.wrap_up(result_vec, choice_add_order, result_line)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
