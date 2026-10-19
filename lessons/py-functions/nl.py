# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-functions. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-functions", "nl")


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
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Wat laat dit programma zien?\n\n```python\ncount = 0\nfor word in [\"a\", \"bb\", \"ccc\"]:\n    if len(word) > 1:\n        count = count + 1\nprint(count)\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "one": "`1`",
        },
        correct="two",
        why={
            "three": "De `if` telt alleen 1 op bij een woord dat langer is dan 1, en `a` is dat niet. Kijk nog eens naar het uitgewerkte voorbeeld in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "two": "`bb` en `ccc` zijn langer dan 1, dus `count` groeit twee keer.",
            "one": "De loop kijkt naar elk woord, en de `if` bekijkt ze één voor één. Welke woorden zijn langer dan 1? Kijk nog eens naar het recept in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Wat laat dit programma zien?\n\n```python\nfor i in range(1, 4):\n    print(i * 10)\n```",
        options={
            "three": "`10`, `20` en `30`, elk op een eigen regel",
            "four": "`10`, `20`, `30` en `40`, elk op een eigen regel",
            "zero": "`0`, `10` en `20`, elk op een eigen regel",
        },
        correct="three",
        why={
            "three": "`range(1, 4)` geeft 1, 2 en 3, en elke ronde laat dat getal keer 10 zien.",
            "four": "`range` stopt net voor zijn stop, dus 4 hoort er niet bij. Kijk nog eens naar het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "zero": "Met twee getallen begint `range` bij het eerste, hier 1. Kijk nog eens naar het idee in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige lessen schreef je steeds dezelfde soort code: een totaal, een telling. Met een functie geef je zulke code een eigen naam. Daarna doet één korte regel het hele werk, voor elke lijst die je meegeeft. Grotere programma's zijn opgebouwd uit functies, en later in de course blijkt ook een lerend programma een functie te zijn: er gaan getallen in, en er komt een antwoord uit.

    **Een voorbeeld uit het dagelijks leven.** Een koffieautomaat. Je drukt op een knop en kiest een maat; de automaat werkt binnenin alle stappen af, en er komt een beker koffie uit. Je hoeft de stappen niet elke keer te kennen: de knop is genoeg. De maat die je kiest, geef je mee, en de beker is wat terugkomt.
    """)
    return


@app.cell(hide_code=True)
def predict_call():
    guess_call = L.guess(
        "call",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ndef double(x):\n    return x * 2\n\nprint(double(5) + 1)\n```",
        options={
            "twelve": "`12`",
            "ten": "`10`",
            "eleven": "`11`",
        },
        reveal="**`11`.** `double(5)` voert de functie uit met `x` gelijk aan 5, en `return x * 2` geeft 10 terug. Daarna laat `print` 10 plus 1 zien. De regels van een functie draaien pas als je hem aanroept.",
    )
    guess_call
    return (guess_call,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Je schrijft een functie één keer, met `def`. Daarna kun je hem zo vaak *aanroepen* als je wilt, elke keer met andere waarden.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **functie** | code onder een naam, die je steeds opnieuw kunt gebruiken | `double` |
        | **def** | begint een functie: `def`, de naam, de argumenten tussen haakjes en een dubbele punt; de ingesprongen regels eronder zijn de functie | `def double(x):` |
        | **argument** | een waarde die je aan een functie meegeeft als je hem aanroept; binnen de functie heeft die waarde de naam uit de regel met `def` | de `5` in `double(5)` |
        | **return** | geeft de uitkomst terug aan de plek waar de functie werd aangeroepen, en beëindigt de functie | `return x * 2` |
        | **aanroepen** | een functie gebruiken: de naam, met de argumenten tussen haakjes | `double(5)` |
        | **None** | het woord van Python voor niets; wat een functie teruggeeft als hij geen `return` heeft | `print(None)` laat `None` zien |

        Een functie kan meer argumenten krijgen, met komma's ertussen. De waarden gaan op volgorde naar de namen: bij `def area(width, height):` geeft de aanroep `area(3, 4)` de naam `width` de waarde 3 en `height` de waarde 4.

        De namen die je binnen een functie maakt, zoals `total`, bestaan alleen zolang de functie draait, dus botsen ze nooit met namen elders. En `print` is niet hetzelfde als `return`: `print` laat een waarde alleen onder de cel zien, `return` geeft hem terug, zodat je hem onder een naam kunt bewaren of ermee kunt rekenen.
        """),
        L.watch_out("De documentatie van Python noemt de namen in de regel met `def` *parameters*. In deze course bewaren we dat woord voor de getallen die een lerend programma bijstelt, dus zeggen we in beide gevallen *argument*."),  # termen:ok
    ])
    return


@app.cell(hide_code=True)
def picture_call():
    mo.vstack([
        draw.flow(
            [("Waar je hem aanroept", ["y = double(5)", "y krijgt 10"]),
             ("De functie double", ["x = 5", "return x * 2"])],
            forward=["5 gaat erin,\nals x"],
            back=["10 komt terug"],
            title="Een aanroep, en wat terugkomt"),
        mo.md(r"""
        De aanroep geeft 5 mee aan de functie, waar het `x` heet. De functie voert zijn regels uit, en `return` stuurt 10 terug naar de plek van de aanroep, waar `y` het krijgt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een functie die alleen `print` gebruikt, geeft niets terug: de uitkomst is `None`, het woord van Python voor niets. En `return` beëindigt de functie meteen, dus zet hem na de loop, één niveau ingesprongen, gelijk met `for`, niet in de loop. Druk in het vak op Backspace tot `return` gelijk staat met `for`."),
        mo.md(r"""
        ### Recept

        Zo maak en gebruik je een functie:

        1. Schrijf `def`, een naam, de argumenten tussen haakjes en een dubbele punt: `def average(numbers):`.
        2. Schrijf eronder, ingesprongen, de stappen, met de namen van de argumenten.
        3. Eindig met `return` en de uitkomst.
        4. Roep de functie aan met waarden, zoals `average([4, 8, 6])`, en bewaar de uitkomst onder een naam of laat hem zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een functie die het gemiddelde van een lijst getallen geeft. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_average():
    # 1. def, de naam, het argument en een dubbele punt
    def _average(numbers):
        # 2. ingesprongen: de stappen, met de naam van het argument
        total = 0
        for hours in numbers:
            total = total + hours
        # 3. return: geef de uitkomst terug
        return total / len(numbers)

    # 4. roep hem aan, hier twee keer, met twee lijsten
    _sleep = _average([7, 6.5, 8, 7.5, 6])
    print(f"Gemiddeld geslapen: {_sleep} uur")
    print(_average([10, 20]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `Gemiddeld geslapen: 7.0 uur` en `15.0`. Dezelfde regels draaiden twee keer, één keer per lijst; alleen het argument was anders. `total` en `hours` bestaan alleen binnen de functie. De naam van de functie zelf begint met `_`, omdat alleen deze cel hem gebruikt, net als bij elke andere naam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_area():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De oppervlakte van een rechthoek

        Maak de functie `area` af: hij krijgt een breedte `width` en een hoogte `height`, en geeft de oppervlakte terug, de breedte keer de hoogte. Vervang de `...`. De check roept je functie aan met 3 en 4, en met andere maten.
        """),
        L.task("Laat `area(width, height)` de oppervlakte van een rechthoek teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_area():
    box_area = L.code_box("area", lines=4)
    box_area
    return (box_area,)


@app.cell(hide_code=True)
def check_area(box_area):
    result_area = L.check("area", box_area.value)
    result_area
    return (result_area,)


@app.cell(hide_code=True)
def hints_area():
    L.hints(
        "area",
        [
            "Stap 3 van het recept: de functie eindigt met `return` en de uitkomst.",
            "De uitkomst is de breedte keer de hoogte. Binnen de functie heten die waarden `width` en `height`.",
            "`return ___ * ___`, ingesprongen onder de regel met `def`.",
        ],
        why="`return width * height` geeft de breedte keer de hoogte terug. Omdat hij de namen van de argumenten gebruikt, klopt hij voor elke rechthoek: `area(3, 4)` geeft `12`, en `area(2.5, 3)` geeft `7.5`.",
    )
    return


@app.cell(hide_code=True)
def text_none():
    mo.md(r"""
    ### Oefening 2. print of return?
    """)
    return


@app.cell(hide_code=True)
def choice_none_cell():
    choice_none = L.choice(
        "none",
        question="Wat laat dit programma zien?\n\n```python\ndef greet(name):\n    print(f\"Hi {name}\")\n\nresult = greet(\"Ada\")\nprint(result)\n```",
        options={
            "twice": "`Hi Ada`, en daaronder nog een keer `Hi Ada`",
            "once": "Alleen `Hi Ada`",
            "none": "`Hi Ada`, en daaronder `None`",
        },
        correct="none",
        why={
            "twice": "`greet` laat `Hi Ada` zien met `print`, maar heeft geen `return`, dus geeft hij niets terug. Kijk nog eens naar *Let op* boven het recept.",
            "none": "De aanroep laat `Hi Ada` zien. `greet` heeft geen `return`, dus krijgt `result` de waarde `None`, en die laat de laatste regel zien.",
            "once": "De laatste regel laat ook iets zien: de waarde van `result`. Wat geeft een functie zonder `return` terug? Kijk nog eens naar *Let op* boven het recept.",
        },
    )
    choice_none
    return (choice_none,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Lange woorden tellen

        Maak de functie `count_long` af: hij krijgt een lijst `words` en een getal `n`, en geeft terug hoeveel woorden meer dan `n` letters hebben. Vervang de `...` door een telling in een loop, zoals in de les over for-loops, en een `return`. De check roept je functie aan met `["sun", "thunder", "drizzle"]` en 5, en met andere lijsten en getallen.
        """),
        L.task("Laat `count_long(words, n)` het aantal woorden met meer dan `n` letters teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-long", lines=8)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-long", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-long",
        [
            "Kijk naar het uitgewerkte voorbeeld: een naam die bij 0 begint, een loop over de lijst, en `return` na de loop.",
            "Zet in de loop een `if` waarvan de voorwaarde `True` is bij een woord met meer dan `n` letters; daaronder tel je 1 op. Gebruik `n`, niet de 5 uit de aanroep van de check.",
            "`count = 0`, `for word in words:`, `if len(word) > n:`, `count = count + 1`, en aan het eind, één niveau ingesprongen, ___.",
        ],
        why="De telling begint bij 0, de loop kijkt naar elk woord, en `if len(word) > n:` telt 1 op bij elk lang woord. `return count` staat na de loop, gelijk met `for`, dus geeft hij de telling terug als elk woord bekeken is.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Herhalen: for-loops](/course/open/py-for?lang=nl) schreef je een telling één keer, voor één lijst; als functie werkt hij voor elke lijst en elk getal. Vanaf nu vragen veel oefeningen om een functie, en de check roept die aan met verschillende invoer. Later in de course is een programma dat leert zelf een functie: er gaan getallen in, en er komt een antwoord uit. In de volgende les roept een functie zichzelf zelfs aan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_none, result_area, result_count):
    L.wrap_up(result_area, choice_none, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
