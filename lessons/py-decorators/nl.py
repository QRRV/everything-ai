# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-decorators. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-decorators", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Functies](/course/open/py-functions?lang=nl) en [Een tekstbestand lezen](/course/open/py-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="Wat laat dit programma zien?\n\n```python\ndef triple(x):\n    return 3 * x\n\nprint(triple(triple(2)))\n```",
        options={
            "six": "`6`",
            "eighteen": "`18`",
            "nine": "`9`",
        },
        correct="eighteen",
        why={
            "six": "Dat is `triple(2)`. Die uitkomst gaat nog een keer in `triple`. Zie de gok in [Functies](/course/open/py-functions?lang=nl).",
            "eighteen": "De binnenste aanroep gaat eerst: `triple(2)` geeft 6, en `triple(6)` geeft 18.",
            "nine": "Elke aanroep vermenigvuldigt met 3; er wordt niets opgeteld. Reken eerst de binnenste aanroep uit, zoals in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_with():
    quick_with = L.choice(
        "quick-with",
        question="Hoe zit het na dit programma met het bestand?\n\n```python\nwith open(\"lessons/py-files/poem.txt\") as f:\n    text = f.read()\nprint(len(text))\n```",
        options={
            "open": "Het is nog open, want `f` bestaat nog",
            "deleted": "Het is weg: `read` haalt de tekst eruit",
            "closed": "Het is weer dicht, meteen na de ingesprongen regel",
        },
        correct="closed",
        why={
            "open": "`with` zorgt voor het bestand. Wat doet hij na zijn ingesprongen regels? Zie de tabel met nieuwe woorden in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "deleted": "Een bestand lezen verandert het nooit. Zie het idee in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "closed": "`with` sluit het bestand na zijn ingesprongen regels, ook als er iets misgaat. `text` houdt wat er gelezen is.",
        },
    )
    quick_with
    return (quick_with,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het bestand van een les van deze course is elke cel een functie met `@app.cell` erboven. Die regel zet een decorator aan het werk: hij geeft de functie eronder aan marimo, en die maakt er een cel van. Later, in PyTorch, doet `@torch.no_grad()` boven een functie voor de hele functie wat `with torch.no_grad():` voor een blok doet. Sommige decorators, zoals deze twee, worden eerst met een aanroep gemaakt, met haakjes; die aanroep geeft de decorator. Een decorator voegt iets toe aan een functie, zonder de functie zelf te veranderen.

    **Een voorbeeld uit het dagelijks leven.** Een cadeau inpakken. Het cadeau zelf blijft hetzelfde, maar wat je geeft, heeft nu papier en een strik eromheen. Een decorator pakt een functie net zo in: de functie blijft zoals hij is, en de verpakking voegt er wat aan toe.
    """)
    return


@app.cell(hide_code=True)
def predict_loud():
    guess_loud = L.guess(
        "loud",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ndef loud(fn):\n    def wrapper():\n        return f\"{fn()}!\"\n    return wrapper\n\n@loud\ndef hello():\n    return \"hi\"\n\nprint(hello())\n```",
        options={
            "plain": "`hi`",
            "error": "Het stopt: `hello` is geen functie meer",
            "loud": "`hi!`",
        },
        reveal="**`hi!`.** `@loud` geeft `hello` aan `loud`, en vanaf dan staat de naam `hello` voor `wrapper`. `hello()` aanroepen voert `wrapper` uit, die de oude `hello` aanroept en er `\"!\"` aan toevoegt.",
    )
    guess_loud
    return (guess_loud,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een functie is een waarde, net als een getal of een lijst. Je kunt hem aan een andere functie geven, en een functie kan een nieuwe functie maken en die teruggeven. Een decorator doet allebei.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **functie als waarde** | de naam van een functie zonder haakjes is de functie zelf; je kunt hem doorgeven, net als elke waarde | `loud(hello)` |
    | **functie in een functie** | een functie die binnen een andere functie gedefinieerd is; hij kan de namen van de buitenste functie gebruiken, zoals `fn` | `def wrapper():` |
    | **decorator** | een functie die een functie krijgt, en een nieuwe functie teruggeeft die hem inpakt | `def loud(fn):` |
    | **`@`** | het apenstaartje, op de regel boven een `def`: `@loud` betekent `hello = loud(hello)`, meteen na de `def` | `@loud` |

    Met haakjes roept `hello()` de functie aan en krijg je zijn uitkomst; zonder haakjes is `hello` de functie zelf. Een decorator heeft de functie zelf nodig. En `wrapper` houdt de namen die hij uit de decorator gebruikt, zoals `fn`, ook als de decorator al klaar is: zo kan een aanroep van de nieuwe `hello` de oude nog aanroepen.
    """)
    return


@app.cell(hide_code=True)
def picture_same():
    mo.vstack([
        draw.marked_lines([("Met @", [("@loud", "1"), ("def hello():", ""), ("    return \"hi\"", "")]),
                           ("Zonder @: hetzelfde", [("def hello():", ""), ("    return \"hi\"", ""), ("hello = loud(hello)", "1")])],
                          {"1": "de decorator krijgt de functie, en de naam staat nu voor wat hij teruggeeft"},
                          title="Wat @ doet"),
        mo.md(r"""
        Beide stukken code doen hetzelfde. `@loud` is een korte manier om `hello = loud(hello)` te schrijven, meteen na de `def`: de naam `hello` staat nu voor de nieuwe functie die `loud` teruggaf.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een decorator geeft de nieuwe functie zelf terug: `return wrapper`, zonder haakjes. Met `return wrapper()` roept Python hem meteen aan, en geeft de decorator die uitkomst terug in plaats van de functie zelf. En het apenstaartje `@` heeft nog een taak: later in de course vermenigvuldigt `@` tussen twee arrays, zoals in `x @ W`, die twee (matrixvermenigvuldiging). Het `@` van een decorator staat altijd vooraan een regel, vlak boven een `def`."),
        mo.md(r"""
        ### Recept

        Zo schrijf je een decorator en gebruik je hem:

        1. Schrijf een functie die een functie krijgt: `def loud(fn):`.
        2. Schrijf daarin een nieuwe functie, `wrapper`, die `fn` aanroept en iets doet met wat die teruggeeft.
        3. Sluit de decorator af met `return wrapper`, zonder haakjes.
        4. Zet `@loud` op de regel boven een `def`. Vanaf dan staat de naam van die functie voor `wrapper`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een decorator die verdubbelt wat een functie teruggeeft. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_decorator():
    # 1. een functie die een functie krijgt, fn
    def _double_result(fn):
        # 2. een nieuwe functie die fn aanroept, en iets toevoegt
        def wrapper(x):
            return 2 * fn(x)

        # 3. geef de nieuwe functie zelf terug, zonder ()
        return wrapper

    # 4. @ boven een def
    @_double_result
    def _add_one(x):
        return x + 1

    print(_add_one(3))

    # hetzelfde, zonder @
    def _add_two(x):
        return x + 2

    _twice_add_two = _double_result(_add_two)
    print(_add_two(3))
    print(_twice_add_two(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `8`, `5` en `10`. `_add_one(3)` voert `wrapper` uit: de oude `_add_one` geeft 4, en `wrapper` verdubbelt dat. Zonder `@` geeft `_double_result(_add_two)` een nieuwe functie, die hier een eigen naam krijgt, dus blijft `_add_two` zoals hij was.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plus_ten():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tien meer

        Maak de decorator `plus_ten` af: elke functie die hij inpakt, geeft 10 meer terug. Vervang de `...` in `wrapper`. De check zet `@plus_ten` boven een functie `square` die `x * x` teruggeeft, en roept `square(3)` aan: dat moet 19 zijn. Hij probeert ook andere functies.
        """),
        L.task("Laat `wrapper` teruggeven wat `fn` voor `x` geeft, plus 10."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plus_ten():
    box_plus_ten = L.code_box("plus-ten", lines=6)
    box_plus_ten
    return (box_plus_ten,)


@app.cell(hide_code=True)
def check_plus_ten(box_plus_ten):
    result_plus_ten = L.check("plus-ten", box_plus_ten.value)
    result_plus_ten
    return (result_plus_ten,)


@app.cell(hide_code=True)
def hints_plus_ten():
    L.hints(
        "plus-ten",
        [
            "Stap 2 van het recept, en `wrapper` in het uitgewerkte voorbeeld: hij roept `fn` aan, en doet iets met de uitkomst.",
            "Roep `fn` aan met `x`, en tel 10 op bij wat hij teruggeeft.",
            "`return fn(___) + ___`",
        ],
        why="`return fn(x) + 10` roept de ingepakte functie aan met `x`, en geeft zijn uitkomst plus 10 terug. Door `@plus_ten` boven `square` geeft `square(3)` 9 + 10, dus 19.",
    )
    return


@app.cell(hide_code=True)
def text_same_as():
    mo.md(r"""
    ### Oefening 2. Waar @ voor staat
    """)
    return


@app.cell(hide_code=True)
def choice_same_as_cell():
    choice_same_as = L.choice(
        "same-as",
        question="Welke regel, meteen na de `def`, doet hetzelfde als `@shout` erboven?\n\n```python\n@shout\ndef greet():\n    return \"hello\"\n```",
        options={
            "same": "`greet = shout(greet)`",
            "called": "`greet = shout(greet())`",
            "swapped": "`shout = greet(shout)`",
        },
        correct="same",
        why={
            "called": "`greet()` met haakjes roept `greet` aan, en geeft zijn uitkomst terug, `\"hello\"`. Krijgt een decorator een uitkomst, of de functie zelf?",
            "same": "`@shout` geeft de functie `greet` zelf aan `shout`, en de naam `greet` staat daarna voor wat `shout` teruggeeft.",
            "swapped": "De decorator is degene die een functie krijgt. Welke van de twee is hier de decorator? Kijk nog eens naar de figuur.",
        },
    )
    choice_same_as
    return (choice_same_as,)


@app.cell(hide_code=True)
def text_exercise_no_negatives():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Niets onder nul

        Schrijf de decorator `no_negatives`; het vak heeft zijn `def`-regel al, dus schrijf de rest eronder. De functies die hij inpakt, geven nooit een getal onder 0 terug. Is de uitkomst kleiner dan 0, dan geeft `wrapper` in plaats daarvan 0 terug; elke andere uitkomst blijft zoals hij is. Vervang de `...`. De check zet `@no_negatives` boven een functie `shift` die `x - 5` teruggeeft, en roept `shift(8)` en `shift(2)` aan: dat moet 3 en 0 geven. Hij probeert ook andere functies.
        """),
        L.task("Laat `no_negatives` een `wrapper` teruggeven die elke uitkomst onder 0 in 0 verandert."),
    ])
    return


@app.cell(hide_code=True)
def exercise_no_negatives():
    box_no_negatives = L.code_box("no-negatives", lines=9)
    box_no_negatives
    return (box_no_negatives,)


@app.cell(hide_code=True)
def check_no_negatives(box_no_negatives):
    result_no_negatives = L.check("no-negatives", box_no_negatives.value)
    result_no_negatives
    return (result_no_negatives,)


@app.cell(hide_code=True)
def hints_no_negatives():
    L.hints(
        "no-negatives",
        [
            "Stap 2 en 3 van het recept, zoals in `_double_result`: een `wrapper` erin, en `return wrapper` aan het eind.",
            "Bewaar in `wrapper` eerst wat `fn(x)` teruggeeft. Is dat kleiner dan 0, geef dan 0 terug; geef anders de uitkomst zelf terug.",
            "In `wrapper`: `result = fn(x)`, dan `if result < 0:` met daaronder `return ___`.",
        ],
        why="`wrapper` roept `fn(x)` één keer aan en bewaart de uitkomst. Onder 0 geeft hij 0 terug; anders de uitkomst zelf. `return wrapper` aan het eind geeft de nieuwe functie terug, zodat `@no_negatives` hem in de plaats van `shift` kan zetten.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) riep je functies aan; nu is een functie ook een waarde die je doorgeeft, en die een decorator inpakt. In [Een tekstbestand lezen](/course/open/py-files?lang=nl) pakte `with` een blok regels in tussen het openen en sluiten van een bestand; een decorator pakt een hele functie op dezelfde manier in. Je `no_negatives` doet wat later in de course een ReLU doet: in een neural network verandert die elke uitkomst onder 0 in 0. De volgende les tekent je getallen als grafieken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same_as, result_no_negatives, result_plus_ten):
    L.wrap_up(result_plus_ten, choice_same_as, result_no_negatives)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
