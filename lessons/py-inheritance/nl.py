# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-inheritance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-inheritance", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Classes en objecten](/course/open/py-classes?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tally():
    quick_tally = L.choice(
        "quick-tally",
        question="Wat laat dit programma zien?\n\n```python\nclass Tally:\n    def __init__(self):\n        self.n = 0\n\n    def up(self):\n        self.n = self.n + 1\n\na = Tally()\nb = Tally()\na.up()\na.up()\nb.up()\nprint(a.n)\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "one": "`1`",
        },
        correct="two",
        why={
            "three": "Elk object houdt zijn eigen `n` bij. Welk object veranderde `b.up()`? Zie de figuur in [Classes en objecten](/course/open/py-classes?lang=nl).",
            "two": "`a.up()` draait twee keer en verandert alleen de eigen `n` van `a`; `b.up()` verandert die van `b`.",
            "one": "`up` draait meer dan één keer op `a`. Tel de aanroepen met `a.` ervoor nog eens, en zie het recept in [Classes en objecten](/course/open/py-classes?lang=nl).",
        },
    )
    quick_tally
    return (quick_tally,)


@app.cell(hide_code=True)
def quick_check_hello():
    quick_hello = L.choice(
        "quick-hello",
        question="Wat laat dit programma zien?\n\n```python\nclass Pet:\n    def __init__(self, name):\n        self.name = name\n\n    def hello(self):\n        return f\"Ik ben {self.name}\"\n\np = Pet(\"Mo\")\nprint(p.hello())\n```",
        options={
            "value": "`Ik ben Mo`",
            "word": "`Ik ben name`",
            "code": "`Ik ben self.name`",
        },
        correct="value",
        why={
            "value": "`Pet(\"Mo\")` geeft het object de naam `\"Mo\"`; in de f-string wordt `{self.name}` daardoor vervangen.",
            "word": "In een f-string wordt `{self.name}` vervangen door de waarde van het attribute van het object. Zie het uitgewerkte voorbeeld in [Classes en objecten](/course/open/py-classes?lang=nl).",
            "code": "Het deel tussen `{}` wordt uitgerekend, niet getoond zoals het er staat. Welke waarde bewaarde `__init__` in `self.name`? Zie het idee in [Classes en objecten](/course/open/py-classes?lang=nl).",
        },
    )
    quick_hello
    return (quick_hello,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vaak is een nieuwe class bijna hetzelfde als een class die je al hebt. Je wilt niet de hele class kopiëren en één regel veranderen: dan heb je twee kopieën die allebei goed moeten blijven. Een subclass bouwt voort op de class die je hebt, en je schrijft alleen wat anders is. In PyTorch begint elk model dat je schrijft zo: `class MyModel(nn.Module):`, voortbouwend op `nn.Module`, de parent class van PyTorch voor modellen, met `super().__init__()` als eerste regel van zijn `__init__`.

    **Een voorbeeld uit het dagelijks leven.** Een elektrische fiets is een fiets: hij heeft de wielen, de remmen en de trappers van een fiets, plus een motor. Niemand ontwerpt er nieuwe wielen voor: de maker neemt een fiets en zet er een motor op.
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nclass Pet:\n    def __init__(self, name):\n        self.name = name\n\n    def speak(self):\n        return f\"{self.name} zegt hoi\"\n\nclass Dog(Pet):\n    def fetch(self):\n        return \"hebbes\"\n\nd = Dog(\"Rex\")\nprint(d.speak())\n```",
        options={
            "speak": "`Rex zegt hoi`",
            "no-speak": "Het stopt: `Dog` heeft geen `speak`",
            "no-init": "Het stopt: `Dog` heeft geen `__init__`",
        },
        reveal="**`Rex zegt hoi`.** `class Dog(Pet):` laat `Dog` voortbouwen op `Pet`. Een `Dog` krijgt `__init__` en `speak` van `Pet`, en voegt een eigen `fetch` toe.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een subclass is een class die voortbouwt op een andere class, zijn parent class. Hij krijgt elke method van de parent class, en voegt nieuwe methods toe of verandert er een paar.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **subclass** | een class die voortbouwt op een andere class; hij krijgt alle methods van die class | `class Dog(Pet):` |
    | **parent class** | de class waarop een subclass voortbouwt; die staat tussen de haakjes | `Pet` |
    | **inheritance** | overerving: een subclass krijgt de methods van zijn parent class | `Dog("Rex").speak()` |
    | **override** | een method in de subclass met dezelfde naam als een method van de parent class: de eigen versie van de subclass wint; je *overschrijft* de method | `def sound(self):` in `Dog` |
    | **`super()`** | in een method: de parent class, zodat je zijn versie van een method kunt uitvoeren | `super().__init__(name)` |

    Roep je een method aan op een object, dan zoekt Python hem eerst in de eigen class van het object, en daarna in de parent class. Een method van de parent class die `self.sound()` aanroept, krijgt dus de `sound` van de subclass, als die er een heeft. En `super()` geeft `self` voor je mee, net als een punt: `super().__init__(name)` krijgt alleen `name`.
    """)
    return


@app.cell(hide_code=True)
def picture_tree():
    mo.vstack([
        draw.class_objects(["class Pet", "__init__(self, name)", "sound(self)", "speak(self)"],
                           [("class Dog(Pet)", ["sound(self)"]),
                            ("class Parrot(Pet)", ["__init__(self, name, word)", "sound(self)"])],
                           class_label="De parent class", objects_label="Twee subclasses: alleen wat ze toevoegen of veranderen",
                           title="Twee classes die op Pet voortbouwen"),
        mo.md(r"""
        Elke pijl betekent: bouwt voort op. `Dog` schrijft alleen `sound`; voor `__init__` en `speak` kijkt Python in `Pet`. `Parrot` schrijft ook een eigen `__init__`, die eerst die van `Pet` uitvoert met `super()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Heeft een subclass een eigen `__init__`, dan draait de `__init__` van de parent class niet meer vanzelf. Vergeet je `super().__init__(...)`, dan mist het object wat de parent class klaarzet, zoals `self.name`."),
        mo.md(r"""
        ### Recept

        Zo laat je een class voortbouwen op een andere class:

        1. Zet de parent class tussen haakjes na de naam van de nieuwe class: `class Dog(Pet):`. De subclass krijgt elke method van `Pet`.
        2. Wil je een method veranderen, overschrijf hem dan: schrijf in de subclass een method met dezelfde naam. De versie van de subclass wint.
        3. Heeft de subclass een eigen `__init__` nodig? Voer dan eerst die van de parent class uit, `super().__init__(name)`, en zet daarna wat nieuw is: `self.word = word`.
        4. Gebruik de subclass zoals elke class: `Dog("Rex").speak()` vindt `speak` in `Pet`, en `sound` in `Dog`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Huisdieren die hun eigen geluid maken. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_pets():
    class _Pet:
        def __init__(self, name):
            self.name = name

        def sound(self):
            return "hm"

        def speak(self):
            return f"{self.name} zegt {self.sound()}"

    # 1. de parent class tussen haakjes
    class _Dog(_Pet):
        # 2. overschrijven: dezelfde naam, een eigen versie
        def sound(self):
            return "woef"

    class _Parrot(_Pet):
        # 3. een eigen __init__: eerst die van de parent class, dan wat nieuw is
        def __init__(self, name, word):
            super().__init__(name)
            self.word = word

        def sound(self):
            return self.word

    # 4. gebruik de subclasses zoals elke class
    print(_Pet("Mo").speak())
    print(_Dog("Rex").speak())
    print(_Parrot("Kiki", "hallo").speak())
    print(_Parrot("Kiki", "hallo").name)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `Mo zegt hm`, `Rex zegt woef`, `Kiki zegt hallo` en `Kiki`. `speak` staat er maar één keer, in `_Pet`, maar hij roept `self.sound()` aan, en zo krijgt elk huisdier zijn eigen geluid. De papegaai heeft een naam, omdat zijn `__init__` eerst die van `_Pet` uitvoert, met `super()`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_greeter():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een begroeting in het Nederlands

        `Greeter` groet in het Engels: `Greeter("Ada").greet()` geeft `Hello, Ada!`. Daaronder bouwt `DutchGreeter` voort op `Greeter`. Vervang zijn `...` door één method, zodat hij in het Nederlands groet: `Hallo, Ada!`. Laat `Greeter` zoals hij is. De check maakt `DutchGreeter("Ada")` en roept `greet()` aan: dat moet `Hallo, Ada!` zijn. Hij kijkt ook of `Greeter` nog steeds Hello zegt.
        """),
        L.task("Laat `DutchGreeter(\"Ada\").greet()` `Hallo, Ada!` geven, door alleen `DutchGreeter` te veranderen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_greeter():
    box_greeter = L.code_box("greeter", lines=16)
    box_greeter
    return (box_greeter,)


@app.cell(hide_code=True)
def check_greeter(box_greeter):
    result_greeter = L.check("greeter", box_greeter.value)
    result_greeter
    return (result_greeter,)


@app.cell(hide_code=True)
def hints_greeter():
    L.hints(
        "greeter",
        [
            "Stap 2 van het recept, en `_Dog` in het uitgewerkte voorbeeld: welke method van `Greeter` geeft het woord Hello?",
            "`greet` vraagt `self.greeting()` om het woord. Overschrijf `greeting` in `DutchGreeter`, en laat hem het Nederlandse woord teruggeven.",
            "`def greeting(self):`, ingesprongen onder `class DutchGreeter(Greeter):`, met daaronder `return ___`.",
        ],
        why="`DutchGreeter` overschrijft `greeting` met `return \"Hallo\"`. Hij krijgt `__init__` en `greet` van `Greeter`, en `greet` roept `self.greeting()` aan: voor een `DutchGreeter` is dat de nieuwe versie.",
    )
    return


@app.cell(hide_code=True)
def text_which_switch():
    mo.md(r"""
    ### Oefening 2. Welke versie draait?
    """)
    return


@app.cell(hide_code=True)
def choice_which_switch_cell():
    choice_which_switch = L.choice(
        "which-switch",
        question="Wat laat dit programma zien?\n\n```python\nclass Lamp:\n    def __init__(self):\n        self.on = False\n\n    def switch(self):\n        self.on = True\n\nclass BrokenLamp(Lamp):\n    def switch(self):\n        self.on = False\n\na = BrokenLamp()\na.switch()\nprint(a.on)\n```",
        options={
            "error": "Het stopt: `BrokenLamp` heeft geen `__init__`",
            "on": "`True`",
            "off": "`False`",
        },
        correct="off",
        why={
            "error": "`BrokenLamp` bouwt voort op `Lamp`, dus krijgt hij elke method van `Lamp`, ook `__init__`. Kijk nog eens naar stap 1 van het recept.",
            "on": "Dat zou de `switch` van `Lamp` doen. In welke class kijkt Python eerst bij `a.switch()`?",
            "off": "`a` is een `BrokenLamp`, dus voert `a.switch()` de eigen versie van `BrokenLamp` uit, die `self.on` op `False` houdt. `__init__` komt van `Lamp`.",
        },
    )
    choice_which_switch
    return (choice_which_switch,)


@app.cell(hide_code=True)
def text_exercise_savings():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een spaarrekening

        `Account` heeft een eigenaar, een saldo dat bij 0 begint, en `deposit`. `Savings` bouwt erop voort, met een eigen rente, `rate`. Vervang de twee `...`: laat `Account` in `__init__` de eigenaar en het saldo klaarzetten; tel in `add_interest` de rente, het saldo keer `self.rate`, op bij het saldo. De check maakt `Savings("Ada", 0.1)`, stort 100 en telt één keer de rente erbij: het saldo moet dan 110 zijn. Hij probeert ook meer, zoals of `Account` op zichzelf nog werkt.
        """),
        L.task("Laat `Savings` met `super()` de `__init__` van `Account` aanroepen, en rente bij zijn saldo optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_savings():
    box_savings = L.code_box("savings", lines=16)
    box_savings
    return (box_savings,)


@app.cell(hide_code=True)
def check_savings(box_savings):
    result_savings = L.check("savings", box_savings.value)
    result_savings
    return (result_savings,)


@app.cell(hide_code=True)
def hints_savings():
    L.hints(
        "savings",
        [
            "Stap 3 van het recept, en `_Parrot` in het uitgewerkte voorbeeld: eerst de `__init__` van de parent class, dan wat nieuw is.",
            "Geef in `__init__` `owner` door aan de `__init__` van `Account`. In `add_interest` is het nieuwe saldo het oude saldo plus het saldo keer `self.rate`.",
            "`super().__init__(___)` in `__init__`, en in `add_interest` `self.balance = self.balance + ___`.",
        ],
        why="`super().__init__(owner)` voert de `__init__` van `Account` uit, die `self.owner` en `self.balance = 0` zet. `self.deposit(self.balance * self.rate)` telt de rente erbij, met `deposit` van `Account`; `self.balance = self.balance + self.balance * self.rate` doet hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Classes en objecten](/course/open/py-classes?lang=nl) schreef je een class vanaf nul; nu kan een class voortbouwen op een andere, en schrijf je alleen wat anders is. Later in de course bouw je elk model zo: `class MyModel(nn.Module):`, met eerst `super().__init__()`, en daarna je eigen layers. De volgende les gaat over generators: functies die waarden één voor één afgeven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which_switch, result_greeter, result_savings):
    L.wrap_up(result_greeter, choice_which_switch, result_savings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
