# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-errors. Route: course/route/lessen/stam-1-basis.yaml. Checks en de traceback in de figuur: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-errors", "nl")
    C = L.checks   # de cel en de traceback in de figuur


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
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="Wat laat dit programma zien?\n\n```python\ndef half(x):\n    return x / 2\n\nprint(half(10) + half(4))\n```",
        options={
            "five": "`5.0`",
            "seven": "`7.0`",
            "fourteen": "`14.0`",
        },
        correct="seven",
        why={
            "five": "`half(10)` geeft 5,0, maar `+` telt `half(4)` erbij op. Zie de gok in [Functies](/course/open/py-functions?lang=nl).",
            "seven": "`half(10)` geeft 5,0 terug en `half(4)` geeft 2,0 terug, en `print` laat hun som zien.",
            "fourteen": "Elke aanroep geeft de helft van zijn eigen argument terug, niet het hele getal. Zie de figuur in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="Wat laat dit programma zien?\n\n```python\ndef first_big(nums):\n    for n in nums:\n        if n > 5:\n            return n\n    return 0\n\nprint(first_big([3, 8, 9]))\n```",
        options={
            "zero": "`0`",
            "nine": "`9`",
            "eight": "`8`",
        },
        correct="eight",
        why={
            "zero": "`return 0` draait alleen als de loop geen getal boven 5 vindt. Zie het blok *Let op* in [Functies](/course/open/py-functions?lang=nl).",
            "nine": "`return` beëindigt de functie meteen, dus de loop komt nooit bij 9. Zie het idee in [Functies](/course/open/py-functions?lang=nl).",
            "eight": "Bij 8 is `n > 5` `True`, en `return n` beëindigt de functie meteen.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Iedereens code gaat mis, steeds weer; ook ervaren programmeurs zijn veel tijd kwijt aan het zoeken naar bugs. Een foutmelding is geen straf: Python vertelt je wat er misging, en waar. Tot nu toe zetten de checks fouten voor je in gewone woorden. Vanaf deze les lees je de meldingen van Python zelf, en dat scheelt je uren als je programma's groter worden.

    **Een voorbeeld uit het dagelijks leven.** Een waarschuwingslampje in een auto. Dat zegt niet alleen "er is iets mis": het olielampje zegt welk onderdeel, en het handboek zegt waar je moet kijken. Een foutmelding werkt net zo: de laatste regel zegt wat er misging, en de regels erboven zeggen waar.
    """)
    return


@app.cell(hide_code=True)
def predict_stop():
    guess_stop = L.guess(
        "stop",
        question="Waar denk je dat dit programma stopt?\n\n```python\ndef area(width, height):\n    return width * hight\n\nprint(area(3, 4))\n```",
        options={
            "def": "Op de regel met `def`",
            "return": "Op de regel met `return`, bij `hight`",
            "none": "Het stopt niet: het laat `12` zien",
        },
        reveal="**Op de regel met `return`.** De regel met `def` leest Python zonder problemen: de regels van een functie draaien pas als je hem aanroept. Als de laatste regel `area(3, 4)` aanroept, komt Python bij `hight`, kent die naam niet, en stopt met een `NameError`. Python voegt er zelfs aan toe: *Did you mean: 'height'?*",
    )
    guess_stop
    return (guess_stop,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Stopt Python met een fout, dan laat het een traceback zien: wat er misging, en de aanroepen die daartoe leidden. Lees hem van onder naar boven.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **bug** | een fout in een programma: het stopt met een foutmelding, of het geeft een verkeerd antwoord | |
    | **traceback** | wat Python laat zien als het stopt: de aanroepen die tot de fout leidden, de eerste bovenaan, en onderaan de soort fout met uitleg | `Traceback (most recent call last):` |
    | **soort fout** | de naam aan het begin van de laatste regel; die zegt wat voor fout het is | `NameError` |

    De soorten die je het vaakst tegenkomt:

    | Soort | Wat het meestal betekent |
    |---|---|
    | `NameError` | Python kent een naam niet: een typefout, of de regel die de naam maakt komt pas later |
    | `TypeError` | twee dingen die niet samengaan, zoals een getal en een lijst, of een aanroep met het verkeerde aantal waarden |
    | `IndexError` | een plek in een lijst die er niet is |
    | `ZeroDivisionError` | delen door nul |
    | `UnboundLocalError` | een naam in een functie wordt gebruikt voordat hij daar een waarde heeft |
    | `SyntaxError`, `IndentationError` | Python kan de code helemaal niet lezen, dus er draait niets in de cel; het wijst de regel aan |

    In een les laat marimo een fout onder de cel zien: de soort in rood, de uitleg van Python eronder, en de betrokken regels gekleurd in je cel. Onder *Traceback* staat het hele pad. `in average` betekent: in de functie `average`; `in <module>` betekent: in de cel zelf, buiten elke functie. In andere talen heet een traceback een stack trace. Python schrijft zijn foutmeldingen in het Engels. Een cel die een naam gebruikt uit een cel met een fout, draait ook niet; die zegt *An ancestor raised an exception*. Los de eerste cel op, die met de rode fout. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def picture_traceback():
    mo.vstack([
        draw.marked_lines([("Een cel", C.PICTURE_CELL), ("Wat Python eronder laat zien", C.PICTURE_TRACEBACK)],
                          {"1": "wat er misging: lees dit eerst",
                           "2": "de regel waar het gebeurde",
                           "3": "de aanroep die daartoe leidde"},
                          title="Een traceback, van onder naar boven gelezen"),
        mo.md(r"""
        De lijst is leeg, dus `len(numbers)` is 0, en door 0 delen kan niet. De laatste regel zegt wat (1); één stap hoger staat de regel waar het gebeurde (2); daarboven de aanroep die ertoe leidde (3). Hier is de bug geen typefout: de functie houdt geen rekening met een lege lijst. De course schrijft de traceback iets korter dan marimo: marimo zet na *Cell* een link naar de cel, en onder sommige regels `~~~^^^`, dat het precieze stuk aanwijst dat misging.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("De regel in de traceback is waar Python het probleem merkte, niet altijd waar de fout zit: een verkeerde waarde kan uit een eerdere regel komen. En geen foutmelding betekent niet dat het klopt: een functie kan een verkeerd antwoord geven zonder enige fout. Test hem met een invoer waarvan je het antwoord weet."),
        mo.md(r"""
        ### Recept

        Zo vind je een bug:

        1. Lees de laatste regel: de soort fout, en de uitleg van Python.
        2. Ga één stap omhoog: de regel in je cel waar het gebeurde, met zijn regelnummer.
        3. Nog hoger: de aanroep die daartoe leidde.
        4. Kijk naar de waarden op die regel. Is een naam verkeerd gespeld? Is een lijst leeg? Is iets een ander soort waarde dan je denkt? Een `print` net boven die regel laat ze zien.
        5. Test kleine stukjes: roep de functie aan met een kleine invoer waarvan je het antwoord weet.
        6. Los het op, voer de cel opnieuw uit, en test nog een keer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hier zijn twee bugs, gevonden en opgelost. De eerste is die uit de figuur: stap 1 tot en met 3 gaven `ZeroDivisionError`, op de regel `return total / len(numbers)`, na de aanroep `average([])`. De tweede geeft helemaal geen fout, alleen een verkeerd antwoord. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_bug_hunt():
    # 1. tot en met 3. de traceback zei: ZeroDivisionError, op de regel
    # return total / len(numbers), na de aanroep average([])
    def _average(numbers):
        # 4. een print(len(numbers)) net boven die regel liet 0 zien bij []
        # 6. de oplossing: een lege lijst krijgt 0, vóór de deling
        if len(numbers) == 0:
            return 0
        total = 0
        for n in numbers:
            total = total + n
        return total / len(numbers)

    # 5. test kleine stukjes: invoer waarvan je het antwoord weet
    print(_average([]))
    print(_average([4]))
    print(_average([2, 4]))

    def _largest(nums):
        # een lege lijst heeft geen grootste getal: geef None terug, het woord van Python voor niets
        if len(nums) == 0:
            return None
        # 6. de oplossing: begin bij het eerste getal; eerst stond er best = 0
        best = nums[0]
        for n in nums:
            if n > best:
                best = n
        return best

    # 5. een kleine test met een antwoord onder 0: met best = 0 gaf die 0
    print(_largest([5, 9, 2]))
    print(_largest([-3, -7]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `0`, `4.0`, `3.0`, `9` en `-3`. Het commentaar bij stap 4 is een test die gedaan en weer weggehaald is: een `print` net boven de regel die misging. De tweede functie gaf voor `[5, 9, 2]` ook met de bug het goede antwoord. Pas een test met een antwoord onder 0, `[-3, -7]`, liet hem zien: met `best = 0` is geen enkel getal ooit groter dan `best`. Daarom test je ook met invoer die een beetje ongewoon is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De totaalprijs

        De functie hieronder hoort de prijzen in een lijst op te tellen, maar heeft een bug. De check eronder laat de traceback zien. Vanaf deze les kan een oefening een gewone cel zijn: verander de code in de cel zelf, en voer hem uit. Wil je je functie uitproberen? Zet er een regel onder, zoals `print(total_price([1, 2]))`. Stopt die regel met een fout, dan laat marimo die in rood zien, en zegt de check *An ancestor raised an exception*: lees de fout, los hem op, en voer de cel opnieuw uit.
        """),
        L.task("Zoek met de traceback de bug in `total_price`, en los hem op."),
    ])
    return


@app.cell
def exercise_total():
    def total_price(prices):
        total = 0
        for price in prices:
            total = total + prices
        return total
    return (total_price,)


@app.cell(hide_code=True)
def check_total(total_price):
    try:
        _value = total_price
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_total = L.check("total", _value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total",
        [
            "Stap 1 van het recept: lees de laatste regel van de traceback in de check. Welke soort fout, en wat zegt Python erover?",
            "Stap 2: één stap hoger staat de regel waar het gebeurde. Welke twee dingen telt die op? Is elk ervan een getal?",
            "In de loop is `price` één prijs en `prices` de hele lijst. Elke ronde hoort er ___ bij te komen.",
        ],
        why="`total = total + prices` telt de hele lijst op bij een getal, en dat kan Python niet: een `TypeError`. In de loop is de prijs van deze ronde `price`, zonder s: `total = total + price`.",
    )
    return


@app.cell(hide_code=True)
def text_traceback():
    mo.md(r"""
    ### Oefening 2. Een traceback lezen
    """)
    return


@app.cell(hide_code=True)
def choice_traceback_cell():
    choice_traceback = L.choice(
        "traceback",
        question="Een cel heeft deze code, en marimo laat deze traceback eronder zien. Wat ging er mis?\n\n```python\ndef to_minutes(hours):\n    return hours * 60\n\ntimes = [1, 2, 3]\nprint(to_minutes(times[3]))\n```\n\n```\nTraceback (most recent call last):\n  Cell, line 5, in <module>\n    print(to_minutes(times[3]))\nIndexError: list index out of range\n```",
        options={
            "multiply": "Een lijst kan niet met 60 vermenigvuldigd worden",
            "index": "`times` heeft geen plek 3: zijn plekken zijn 0, 1 en 2",
            "function": "`to_minutes` bestaat op regel 5 nog niet",
        },
        correct="index",
        why={
            "multiply": "De traceback komt nooit in `to_minutes`: hij stopt op regel 5. Lees de laatste regel nog eens: welke soort fout is het?",
            "index": "De laatste regel zegt `IndexError`, en de regel erboven is regel 5: `times[3]` vraagt om een plek die er niet is.",
            "function": "Dan zou de laatste regel `NameError` zeggen. Lees de laatste regel nog eens: welke soort fout is het?",
        },
    )
    choice_traceback
    return (choice_traceback,)


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een bug zonder foutmelding

        `count_passed` hoort de scores van 5,5 of meer te tellen. Hij geeft geen fout, maar zijn antwoord klopt niet. Test hem met kleine lijsten waarvan je het antwoord weet, zoek de bug, en los hem op in de cel.
        """),
        L.task("Zoek de bug in `count_passed` door kleine stukjes te testen, en los hem op."),
    ])
    return


@app.cell
def exercise_passed():
    def count_passed(scores):
        count = 0
        for score in scores:
            if score >= 5.5:
                count = count + 1
            return count
    return (count_passed,)


@app.cell(hide_code=True)
def check_passed(count_passed):
    try:
        _value = count_passed
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_passed = L.check("passed", _value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "Stap 5 van het recept: zet onder de functie `print(count_passed([7.0]))` en `print(count_passed([4.5, 8.0]))`, en voer de cel uit. Welk antwoord klopt niet?",
            "Met `[4.5, 8.0]` is het antwoord 0: de functie stopt na de eerste score. Welke regel beëindigt een functie meteen?",
            "`return count` hoort één keer te draaien, na de loop: gelijk met ___.",
        ],
        why="`return` beëindigt de functie meteen. Ingesprongen in de loop draait hij in de eerste ronde, na één score. Gelijk met `for` draait hij na de loop, als elke score geteld is.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Tot nu toe zetten de checks elke fout in gewone woorden. Vanaf hier lees je de tracebacks van Python zelf, en zijn sommige oefeningen gewone cellen. Je gebruikte wat [Functies](/course/open/py-functions?lang=nl) leerde: `return` beëindigt een functie, en de check roept hem aan met verschillende invoer. Later in de course vertelt een traceback je welke regel getallen kreeg waar hij niets mee kon. De volgende les is een extra les: daar vang je zelf fouten op, en geef je je eigen foutmeldingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_traceback, result_passed, result_total):
    L.wrap_up(result_total, choice_traceback, result_passed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
