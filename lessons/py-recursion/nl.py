# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-recursion. Route: course/route/lessen/stam-1-basis.yaml. Checks en de aanroepen in het beeld: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-recursion", "nl")
    C = L.checks   # de aanroepen in het beeld, en wat elke aanroep teruggeeft


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
def quick_check_call():
    quick_call = L.choice(
        "quick-call",
        question="Wat laat dit programma zien?\n\n```python\ndef triple(x):\n    return x * 3\n\nprint(triple(2) + 1)\n```",
        options={
            "six": "`6`",
            "seven": "`7`",
            "nine": "`9`",
        },
        correct="seven",
        why={
            "six": "`triple(2)` geeft 6, maar dat is niet alles wat `print` laat zien. Kijk nog eens naar de gok in [Functies](/course/open/py-functions?lang=nl).",
            "seven": "`triple(2)` geeft 6 terug, en `print` laat 6 plus 1 zien.",
            "nine": "Het argument is 2, dus `x * 3` is 6, niet 3 keer 3. Kijk nog eens naar het beeld in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_call
    return (quick_call,)


@app.cell(hide_code=True)
def quick_check_none():
    quick_none = L.choice(
        "quick-none",
        question="Wat laat dit programma zien?\n\n```python\ndef show(x):\n    print(x)\n\nresult = show(4)\nprint(result)\n```",
        options={
            "none": "`4`, en daaronder `None`",
            "twice": "`4`, en daaronder nog een keer `4`",
            "once": "Alleen `4`",
        },
        correct="none",
        why={
            "none": "`show` laat 4 zien met `print`, maar heeft geen `return`, dus krijgt `result` de waarde `None`.",
            "twice": "`show` gebruikt `print`, geen `return`, dus geeft hij niets terug. Kijk nog eens naar *Let op* in [Functies](/course/open/py-functions?lang=nl).",
            "once": "De laatste regel laat ook de waarde van `result` zien. Kijk nog eens naar *Let op* in [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_none
    return (quick_none,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige problemen bevatten een kleinere kopie van zichzelf: een map kan bestanden bevatten, en andere mappen. Wil je al zijn bestanden tellen, dan tel je de bestanden die er direct in staan, plus de bestanden in elke map erbinnen: dezelfde vraag, alleen kleiner. Een functie die zichzelf aanroept, lost zulke problemen in een paar regels op. Later in de course kan een programma dat leert zijn voorbeelden in kleinere groepen verdelen en elke groep op dezelfde manier behandelen.

    **Een voorbeeld uit het dagelijks leven.** Matroesjka's, de Russische poppetjes in elkaar. Hoeveel poppetjes zijn er? Open het grootste: dat is één poppetje, plus alle poppetjes erin. Dat is dezelfde vraag, voor een kleiner poppetje. Het kleinste gaat niet open, en daar stopt het tellen.
    """)
    return


@app.cell(hide_code=True)
def predict_countdown():
    guess_countdown = L.guess(
        "countdown",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ndef countdown(n):\n    if n == 0:\n        print(\"Go!\")\n    else:\n        print(n)\n        countdown(n - 1)\n\ncountdown(3)\n```",
        options={
            "three": "Alleen `3`",
            "all": "`3`, `2`, `1` en `Go!`, elk op een eigen regel",
            "go": "Alleen `Go!`",
        },
        reveal="**`3`, `2`, `1` en `Go!`.** `countdown(3)` laat 3 zien en roept `countdown(2)` aan, die 2 laat zien en `countdown(1)` aanroept, enzovoort. `countdown(0)` laat `Go!` zien en roept niets meer aan, dus daar stopt alles.",
    )
    guess_countdown
    return (guess_countdown,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## Het idee

    Een functie mag zichzelf aanroepen, zolang elke aanroep aan een kleiner stuk werkt, en het kleinste stuk direct een antwoord krijgt.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **recursie** | een functie die zichzelf aanroept, elke keer met een kleiner stuk van het probleem | `return n + sum_to(n - 1)` |
    | **basisgeval** | het kleinste geval, waarvan de functie het antwoord teruggeeft zonder zichzelf aan te roepen; het stopt de recursie | `if n == 0:` en `return 0` |

    Elke aanroep heeft zijn eigen namen: de `n` van de ene aanroep is niet de `n` van de volgende. Een slice werkt op een string net als op een lijst: `text[1:]` is de tekst zonder het eerste teken, een kleiner stuk. `""` is de lege tekst, zonder één teken; `==` vergelijkt ook teksten, dus `text == ""` is alleen `True` bij de lege tekst. En `+` plakt twee teksten aan elkaar: `"ab" + "c"` is `"abc"`. Zonder basisgeval, of als de aanroepen nooit kleiner worden, stopt de functie nooit; Python geeft het op na zo'n duizend aanroepen diep, met een foutmelding.
    """)])
    return


@app.cell(hide_code=True)
def picture_calls():
    mo.vstack([
        mo.md("```python\ndef sum_to(n):\n    if n == 0:\n        return 0\n    return n + sum_to(n - 1)\n```"),
        draw.program_trace(C.PICTURE_LINES["nl"], C.PICTURE_BACK, code_label="De aanroep", output_label="Geeft terug",
                           title="sum_to(3), aanroep voor aanroep"),
        mo.md(r"""
        De aanroepen gaan omlaag, elk met een getal dat één kleiner is, tot het basisgeval 0 teruggeeft. Dan komen de antwoorden weer omhoog: 1 + 0 is 1, 2 + 1 is 3, en 3 + 3 is 6.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Elke functie die zichzelf aanroept, heeft een basisgeval nodig dat een antwoord teruggeeft zonder zichzelf opnieuw aan te roepen, en elke aanroep moet er dichterbij komen. Anders stopt de functie nooit. Zet het basisgeval als eerste, bovenaan de functie."),
        mo.md(r"""
        ### Recept

        Zo laat je een functie zichzelf aanroepen:

        1. Basisgeval: vang met `if` het kleinste geval af, en geef met `return` het antwoord terug.
        2. Roep anders de functie zelf aan met een kleiner stuk: `n - 1`, of `text[1:]`.
        3. Combineer dat antwoord met het deel dat je apart hield, `n` of `text[0]`, en geef de uitkomst terug met `return`.
        4. Controleer dat elke aanroep dichter bij het basisgeval komt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee functies die zichzelf aanroepen: een macht, zoals 2 tot de macht 5, en hoe vaak de letter `a` in een tekst staat. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_recursion():
    def _power(number, exponent):
        # 1. basisgeval: alles tot de macht 0 is 1
        if exponent == 0:
            return 1
        # 2. en 3. de macht die één kleiner is, en dat nog één keer maal het getal
        return number * _power(number, exponent - 1)

    def _count_a(text):
        # 1. basisgeval: een lege tekst heeft geen a
        if text == "":
            return 0
        # 2. de rest van de tekst, één teken korter
        rest = _count_a(text[1:])
        # 3. tel 1 op als het eerste teken een a is
        if text[0] == "a":
            return rest + 1
        return rest

    # 4. elke aanroep is één kleiner: exponent - 1, of text[1:]
    print(_power(2, 5))
    print(_count_a("banana"))
    # elke macht onderweg, van 2 tot de macht 0 tot en met 5
    for _e in range(6):
        print(f"2 tot de macht {_e} is {_power(2, _e)}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `32` en `3`, en daarna elke macht onderweg: elke macht is 2 keer de vorige. `_power(2, 5)` is 2 keer `_power(2, 4)`, en zo verder tot `_power(2, 0)`, en dat is 1. `_count_a("banana")` kijkt naar de `b` en stelt dezelfde vraag voor `anana`, dan voor `nana`, tot aan de lege tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_factorial():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De faculteit

        De faculteit van een getal is dat getal keer elk geheel getal eronder, tot en met 1: de faculteit van 4 is `4 * 3 * 2 * 1`, dus 24. De faculteit van 0 is 1. Maak `factorial(n)` af met een basisgeval en een aanroep van zichzelf. De check roept hem aan met 4, en met andere getallen, ook met 0.
        """),
        L.task("Laat `factorial(n)` de faculteit van `n` teruggeven, door zichzelf aan te roepen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_factorial():
    box_factorial = L.code_box("factorial", lines=6)
    box_factorial
    return (box_factorial,)


@app.cell(hide_code=True)
def check_factorial(box_factorial):
    result_factorial = L.check("factorial", box_factorial.value)
    result_factorial
    return (result_factorial,)


@app.cell(hide_code=True)
def hints_factorial():
    L.hints(
        "factorial",
        [
            "Stap 1 van het recept: wat is het kleinste geval, en wat is het antwoord daarbij? De opdracht zegt wat de faculteit van 0 is.",
            "De faculteit van 4 is 4 keer de faculteit van 3. Dus de faculteit van `n` is `n` keer de faculteit van ___.",
            "`if n == 0:` met `return 1` eronder, en daarna `return n * factorial(___)`.",
        ],
        why="`if n == 0: return 1` is het basisgeval. Anders `return n * factorial(n - 1)`: het getal zelf keer de faculteit van het getal dat één kleiner is. `factorial(4)` wordt `4 * 3 * 2 * 1 * 1`, dus 24.",
    )
    return


@app.cell(hide_code=True)
def text_sum_four():
    mo.md(r"""
    ### Oefening 2. Volg de aanroepen
    """)
    return


@app.cell(hide_code=True)
def choice_sum_four_cell():
    choice_sum_four = L.choice(
        "sum-four",
        question="Wat laat dit programma zien?\n\n```python\ndef f(n):\n    if n == 1:\n        return 1\n    return n + f(n - 1)\n\nprint(f(4))\n```",
        options={
            "four": "`4`",
            "twenty-four": "`24`",
            "ten": "`10`",
        },
        correct="ten",
        why={
            "four": "`f(4)` stopt niet bij 4: hij telt `f(3)` erbij op, die `f(2)` erbij optelt, enzovoort. Kijk nog eens naar het beeld met de aanroepen.",
            "twenty-four": "De functie telt op, `+`; hij vermenigvuldigt niet. Lees de laatste regel van de functie nog eens.",
            "ten": "`f(4)` is 4 + `f(3)`, en dat is 4 + 3 + 2 + 1, dus 10. `f(1)` is het basisgeval.",
        },
    )
    choice_sum_four
    return (choice_sum_four,)


@app.cell(hide_code=True)
def text_exercise_reverse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Tekst achterstevoren

        Maak `reverse(text)` af: hij geeft de tekst achterstevoren terug, dus `reverse("abc")` geeft `"cba"`. Draai je een tekst om, dan krijg je eerst de rest, `text[1:]`, omgedraaid, en daarachter het eerste teken. De lege tekst is het basisgeval: achterstevoren is die nog steeds leeg. De check probeert ook andere teksten.
        """),
        L.task("Laat `reverse(text)` de tekst achterstevoren teruggeven, door zichzelf aan te roepen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reverse():
    box_reverse = L.code_box("reverse", lines=6)
    box_reverse
    return (box_reverse,)


@app.cell(hide_code=True)
def check_reverse(box_reverse):
    result_reverse = L.check("reverse", box_reverse.value)
    result_reverse
    return (result_reverse,)


@app.cell(hide_code=True)
def hints_reverse():
    L.hints(
        "reverse",
        [
            "Stap 1 van het recept: het basisgeval is de lege tekst, `\"\"`. Wat geeft die terug?",
            "Kijk naar `_count_a` in het uitgewerkte voorbeeld: die roept zichzelf aan met `text[1:]`. Hier komt het eerste teken, `text[0]`, achter die uitkomst.",
            "`if text == \"\":` met `return \"\"` eronder, en daarna `return reverse(___) + ___`.",
        ],
        why="`if text == \"\": return \"\"` stopt de aanroepen. Anders `return reverse(text[1:]) + text[0]`: de rest achterstevoren, dan het eerste teken. `reverse(\"abc\")` wordt `reverse(\"bc\") + \"a\"`, dus `\"cb\" + \"a\"`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) deed een functie zijn werk in één keer; nu geeft hij een kleiner stuk van het werk aan zichzelf. Veel van deze klussen kan een loop ook aan, en in Python is een loop vaak de eenvoudigste keuze. Recursie komt tot zijn recht als het probleem zelf genest is, zoals mappen in mappen. In de volgende les bewaar je waarden onder eigen namen, in een dictionary.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sum_four, result_factorial, result_reverse):
    L.wrap_up(result_factorial, choice_sum_four, result_reverse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
