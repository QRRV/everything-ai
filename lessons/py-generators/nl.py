# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-generators. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-generators", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Functies](/course/open/py-functions?lang=nl) en [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="Wat laat dit programma zien?\n\n```python\ndef total(numbers):\n    t = 0\n    for n in numbers:\n        t = t + n\n    return t\n\nprint(total([2, 3, 4]))\n```",
        options={
            "four": "`4`",
            "list": "`[2, 3, 4]`",
            "nine": "`9`",
        },
        correct="nine",
        why={
            "four": "Dat is alleen het laatste getal. Elke ronde telt een getal op bij `t`. Zie het uitgewerkte voorbeeld in [Functies](/course/open/py-functions?lang=nl).",
            "list": "`total` geeft `t` terug, niet de lijst die hij kreeg. Volg `t` door de rondes van de loop.",
            "nine": "De loop telt 2, 3 en 4 op bij `t`, en `return t` geeft 9 terug.",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Welke getallen laat dit programma zien, één per regel?\n\n```python\nfor i in range(3):\n    print(i * 2)\n```",
        options={
            "zero": "`0`, `2` en `4`",
            "one": "`2`, `4` en `6`",
            "upto": "`0`, `2`, `4` en `6`",
        },
        correct="zero",
        why={
            "zero": "`range(3)` geeft 0, 1 en 2; elk wordt verdubbeld.",
            "one": "`range(3)` begint bij 0, niet bij 1. Zie de tabel met nieuwe woorden in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "upto": "`range` stopt vlak vóór zijn stopgetal, dus is `i` nooit 3. Zie de tabel met nieuwe woorden in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een functie met `return` rekent eerst alles uit, en geeft het daarna in één keer terug. Voor drie getallen is dat prima. Maar een model leert van miljoenen voorbeelden, in kleine groepjes die batches heten, en die passen vaak niet allemaal tegelijk in het geheugen. Een generator geeft één waarde, of één batch, tegelijk af, pas als er om de volgende wordt gevraagd.

    **Een voorbeeld uit het dagelijks leven.** De nummertjesautomaat bij de kaasafdeling. Hij drukt 's ochtends niet alle nummers van de dag af: elke keer dat iemand trekt, geeft hij het volgende nummer, en hij onthoudt waar hij was.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ndef three():\n    yield \"a\"\n    yield \"b\"\n    yield \"c\"\n\nfor letter in three():\n    print(letter)\n```",
        options={
            "first": "Alleen `a`",
            "all": "`a`, `b` en `c`, één per regel",
            "error": "Het stopt: een functie kan maar één waarde teruggeven",
        },
        reveal="**`a`, `b` en `c`, één per regel.** Elke `yield` geeft één waarde af, en de functie pauzeert daar. De `for`-loop vraagt elke ronde om de volgende waarde, en de functie gaat verder waar hij gepauzeerd was.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een functie met `yield` erin is een generator function. Als je hem aanroept, draait hij nog niet: je krijgt een generator, die waarden één voor één afgeeft, elke keer dat erom gevraagd wordt.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **`yield`** | geeft één waarde af, en pauzeert de functie op die regel; de volgende keer gaat hij daar verder | `yield i * i` |
    | **generator** | wat je krijgt als je een functie met `yield` aanroept; hij geeft waarden één voor één af | `g = squares(4)` |
    | **iterator** | iets wat waarden één voor één afgeeft, op de manier waarop `for` erom vraagt, tot hij op is; een generator is er een, een lijst niet: over een lijst kun je steeds opnieuw lopen, en `next` werkt er niet op | `for s in g:` |
    | **`next(g)`** | vraagt `g` om zijn volgende waarde | `next(g)` |
    | **`list(g)`** | vraagt `g` om al zijn waarden, en zet ze in een lijst | `list(squares(4))` |

    Een generator onthoudt tussen twee waarden waar hij was, en de waarden van zijn namen. Hij geeft elke waarde één keer af: na de laatste is hij op.
    """)
    return


@app.cell(hide_code=True)
def picture_turns():
    mo.vstack([
        draw.marked_lines([("De generator function", [("def countdown(n):", ""), ("    for i in range(n):", ""), ("        yield n - i", "2")]),
                           ("De loop die vraagt", [("for s in countdown(3):", "1"), ("    print(s)", "3")])],
                          {"1": "vraagt elke ronde om de volgende waarde",
                           "2": "geeft één waarde af, en pauzeert hier",
                           "3": "gebruikt de waarde, en vraagt dan opnieuw"},
                          title="Waarden één voor één"),
        mo.md(r"""
        De loop en de generator zijn om de beurt aan zet. De loop vraagt (1); de generator draait tot zijn volgende `yield`, geeft een waarde af en pauzeert (2); de loop gebruikt die (3) en vraagt opnieuw. Is de eigen loop van de generator klaar, dan stopt de `for` ook.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een generator is op na één keer doorlopen. Een tweede `for` over dezelfde generator krijgt niets meer. Heb je de waarden opnieuw nodig? Roep de functie opnieuw aan: dat geeft een nieuwe generator."),
        mo.md(r"""
        ### Recept

        Zo geef je waarden één voor één af:

        1. Schrijf een functie zoals altijd, maar geef elke waarde af met `yield`, in plaats van ze allemaal in een lijst te verzamelen.
        2. Roep de functie aan: dat geeft een generator. Er is nog niets van de functie gedraaid.
        3. Vraag hem om waarden: met `for` één per ronde, met `next(g)` één tegelijk, of met `list(g)` allemaal.
        4. Heb je de waarden opnieuw nodig? Roep de functie opnieuw aan voor een nieuwe generator.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Even getallen één voor één, en een lopend totaal. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_generators():
    # 1. een functie met yield: hij geeft één waarde tegelijk af
    def _evens(n):
        print("start")
        for i in range(n):
            yield 2 * i

    # 2. aanroepen geeft een generator; er is nog niets gedraaid
    _g = _evens(4)
    print("made _g")
    # 3. vraag om waarden: één met next, de rest met een loop
    print(next(_g))
    for _x in _g:
        print(_x)
    # 4. op; een nieuwe aanroep geeft een nieuwe generator
    print(list(_g))
    print(list(_evens(4)))

    # een generator onthoudt zijn namen tussen twee waarden
    def _running_total(numbers):
        total = 0
        for n in numbers:
            total = total + n
            yield total

    print(list(_running_total([3, 1, 2])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `made _g`, `start`, `0`, `2`, `4`, `6`, `[]`, `start`, `[0, 2, 4, 6]` en `[3, 4, 6]`. `start` komt na `made _g`: `_evens` begint pas als `next(_g)` om de eerste waarde vraagt. De loop kreeg de andere drie waarden. Daarna was `_g` op: `list(_g)` is `[]`, een lege lijst. Een nieuwe aanroep, `_evens(4)`, begint opnieuw. `_running_total` houdt `total` bij tussen twee waarden, en geeft elke keer het totaal tot dan toe af.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_squares():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Kwadraten, één voor één

        Schrijf `squares(n)`, een generator function die de kwadraten afgeeft van 0 tot `n`, die zelf niet meedoet: `0`, `1`, `4`, `9`, enzovoort. De check vraagt `squares(4)` om al zijn waarden: dat moet `[0, 1, 4, 9]` zijn. Hij probeert ook andere getallen.
        """),
        L.task("Laat `squares(n)` voor elke `i` in `range(n)` de waarde `i * i` afgeven, met `yield`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_squares():
    box_squares = L.code_box("squares", lines=5)
    box_squares
    return (box_squares,)


@app.cell(hide_code=True)
def check_squares(box_squares):
    result_squares = L.check("squares", box_squares.value)
    result_squares
    return (result_squares,)


@app.cell(hide_code=True)
def hints_squares():
    L.hints(
        "squares",
        [
            "Stap 1 van het recept, en `_evens` in het uitgewerkte voorbeeld: een loop, met `yield` erin.",
            "Loop over `range(n)`; geef in elke ronde het kwadraat van `i` af.",
            "`for i in range(n):`, met daaronder `yield ___`.",
        ],
        why="`for i in range(n):` gaat van 0 tot `n`, die zelf niet meedoet, en `yield i * i` geeft elk kwadraat af, één voor één.",
    )
    return


@app.cell(hide_code=True)
def text_when_runs():
    mo.md(r"""
    ### Oefening 2. Wanneer draait hij?
    """)
    return


@app.cell(hide_code=True)
def choice_when_runs_cell():
    choice_when_runs = L.choice(
        "when-runs",
        question="Welke regels laat dit programma zien, in welke volgorde?\n\n```python\ndef numbers():\n    print(\"start\")\n    yield 1\n    yield 2\n\ng = numbers()\nprint(\"made\")\nfor x in g:\n    print(x)\n```",
        options={
            "start-first": "`start`, `made`, `1`, `2`",
            "no-start": "`made`, `1`, `2`",
            "made-first": "`made`, `start`, `1`, `2`",
        },
        correct="made-first",
        why={
            "start-first": "Dat zou een functie zonder `yield` doen. Kijk nog eens naar stap 2 van het recept.",
            "no-start": "De `print` in `numbers` draait wel, maar wanneer? Kijk nog eens naar de figuur: wat gebeurt er als de loop om de eerste waarde vraagt?",
            "made-first": "`numbers()` maakt alleen de generator. De functie begint pas als de loop om de eerste waarde vraagt: dan laat hij `start` zien en geeft hij 1 af.",
        },
    )
    choice_when_runs
    return (choice_when_runs,)


@app.cell(hide_code=True)
def text_exercise_batches():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Batches

        Een model leert van zijn data in batches: kleine groepjes voorbeelden, één groepje tegelijk. Maak `batches(items, size)` af: een generator function die items verzamelt in `batch`, en `batch` afgeeft zodra hij `size` items heeft. Wat aan het eind overblijft, gaat eruit als een laatste, kleinere batch. Vervang de `...`, en voeg de regels na de loop toe. De check vraagt `batches([1, 2, 3, 4, 5], 2)` om al zijn waarden: dat moet `[[1, 2], [3, 4], [5]]` zijn. Hij probeert ook andere lijsten, ook een lege.
        """),
        L.task("Laat `batches(items, size)` volle batches één voor één afgeven, en wat aan het eind overblijft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batches():
    box_batches = L.code_box("batches", lines=10)
    box_batches
    return (box_batches,)


@app.cell(hide_code=True)
def check_batches(box_batches):
    result_batches = L.check("batches", box_batches.value)
    result_batches
    return (result_batches,)


@app.cell(hide_code=True)
def hints_batches():
    L.hints(
        "batches",
        [
            "Net als `_running_total` in het uitgewerkte voorbeeld houdt je generator een naam bij tussen twee waarden: hier de lijst `batch`.",
            "In de loop: voeg het item toe aan `batch`; is `len(batch)` gelijk aan `size`, geef hem dan af en begin een nieuwe, lege `batch`. Na de loop: geef af wat er nog over is, als er iets over is.",
            "`batch.append(item)`, dan `if len(batch) == ___:` met daaronder het afgeven en de nieuwe batch. Na de loop, even ver ingesprongen als `for`: `if ___:`.",
        ],
        why="Elk item gaat in `batch`; een volle batch gaat eruit met `yield`, en `batch = []` begint de volgende. Na de loop geeft `if len(batch) > 0:` de laatste, kleinere batch af, maar nooit een lege.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) gaf een functie één waarde terug met `return`, en stopte hij; een generator geeft waarden af met `yield`, en gaat door. In [Herhalen: for-loops](/course/open/py-for?lang=nl) liep je over lijsten en `range`; nu weet je dat `for` om één waarde tegelijk vraagt, aan alles wat ze afgeeft. Later in de course komt de data van een model in batches, net als uit je `batches`: één batch per stap van de training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when_runs, result_batches, result_squares):
    L.wrap_up(result_squares, choice_when_runs, result_batches)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
