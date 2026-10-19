# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-variables. Route: course/route/lessen/stam-1-basis.yaml. Checks, het programma in het beeld en het uitgewerkte voorbeeld: checks.py.
# termen:allow program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-variables", "nl")
    C = L.checks   # het programma in het beeld, de namen na elke regel, en het uitgewerkte voorbeeld


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je eerste programma](/course/open/py-first-program?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_text():
    quick_text = L.choice(
        "quick-text",
        question="Wat laat `print(\"3 + 4\")` onder de cel zien?",
        options={
            "seven": "`7`",
            "text": "`3 + 4`",
            "error": "Een foutmelding",
        },
        correct="text",
        why={
            "seven": "Tussen aanhalingstekens is `3 + 4` tekst, een string, en tekst laat Python precies zo zien, zonder te rekenen. Kijk nog eens in [Je eerste programma](/course/open/py-first-program?lang=nl).",
            "text": "Tussen aanhalingstekens is het een string, en die laat Python precies zo zien. In deze les reken je met getallen zonder aanhalingstekens.",
            "error": "Tekst tussen aanhalingstekens is prima voor `print`. Kijk nog eens in [Je eerste programma](/course/open/py-first-program?lang=nl).",
        },
    )
    quick_text
    return (quick_text,)


@app.cell(hide_code=True)
def quick_check_comment():
    quick_comment = L.choice(
        "quick-comment",
        question="Wat doet Python met een regel die met `#` begint?",
        options={
            "show": "Het laat de regel onder de cel zien.",
            "error": "Het stopt met een foutmelding.",
            "skip": "Het slaat de regel over.",
        },
        correct="skip",
        why={
            "show": "Alleen `print` laat iets onder de cel zien. Een regel met `#` is commentaar, voor mensen. Kijk nog eens in [Je eerste programma](/course/open/py-first-program?lang=nl).",
            "error": "Commentaar mag overal staan: het is een aantekening voor mensen. Kijk nog eens in [Je eerste programma](/course/open/py-first-program?lang=nl).",
            "skip": "Een regel met `#` is commentaar: een aantekening voor mensen, die Python overslaat.",
        },
    )
    quick_comment
    return (quick_comment,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe lieten je programma's tekst zien die je zelf had getypt. Maar AI is vooral rekenen met getallen: heel veel getallen, steeds opnieuw. Daarvoor heb je namen nodig, om uitkomsten onder te bewaren. Met een naam bewaar je een uitkomst en gebruik je die later opnieuw, zonder het getal nog eens te typen.

    **Een voorbeeld uit het dagelijks leven.** Een etiket op een pot in de keuken: *suiker*. Een recept zegt "doe er een lepel uit de suikerpot bij", zonder te zeggen hoeveel suiker erin zit. En wordt de pot bijgevuld, dan blijft het etiket en verandert de inhoud. Het is maar een beeld, maar zo werkt een naam in Python ongeveer ook.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-value",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nprice = 2\nprice = 5\nprint(price)\n```",
        options={
            "two": "`2`",
            "five": "`5`",
            "both": "`2` en `5`",
        },
        reveal="**`5`.** Een naam heeft steeds maar één waarde. Regel 1 geeft `price` de waarde 2; regel 2 geeft hem een nieuwe waarde, 5, en de 2 is weg. `print` laat de waarde zien die de naam op dat moment heeft.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **variabele** | een naam voor een waarde; je geeft hem een waarde met `=` | `price = 2.5` |
    | **int** | een geheel getal, kort voor *integer* | `3`, `-12` |
    | **float** | een kommagetal; Python schrijft altijd een punt, nooit een komma | `2.5`, `3.0` |

    En de tekens om mee te rekenen:

    | Teken | Je zegt | Voorbeeld | Geeft |
    |---|---|---|---|
    | `+` | plus | `7 + 2` | `9` |
    | `-` | min | `7 - 2` | `5` |
    | `*` | keer | `7 * 2` | `14` |
    | `/` | gedeeld door | `7 / 2` | `3.5` |
    | `**` | tot de macht | `7 ** 2` | `49` |

    Delen met `/` geeft altijd een float, ook als het precies uitkomt: `6 / 2` geeft `3.0`. Een macht is een getal dat je met zichzelf vermenigvuldigt: `2 ** 3` is `2 * 2 * 2`, dus `8`. En net als in de wiskunde gaan `*` en `/` vóór `+` en `-`; met ronde haakjes verander je dat: `(1 + 2) * 3` geeft `9`.

    Hieronder staat een programma met namen. Bij elke regel zie je de namen met hun waarden nadat die regel is uitgevoerd, en wat de regel onder de cel laat zien.
    """)
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        draw.program_trace(C.STEPS, C.SHOWN, code_label="Code", names=C.NAMES, names_label="Namen na deze regel",
                           output_label="Onder de cel", title="Namen en hun waarden, regel voor regel"),
        mo.md(r"""
        Een naam bestaat vanaf de regel die hem een waarde geeft; de naam die op een regel verandert, staat vetgedrukt. Regel 3 rekent met de waarden die `price` en `cups` op dat moment hebben, en pas regel 4 laat iets onder de cel zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("In Python betekent `=` niet \"is gelijk aan\", zoals in de wiskunde. Het betekent: geef de naam links de waarde rechts. Daarom is `price = 5` direct na `price = 2` geen tegenspraak: de naam krijgt gewoon een nieuwe waarde."),
        mo.md(r"""
        ### Recept

        Zo reken je met namen:

        1. Geef elk getal waarmee je begint een naam, met `=`.
        2. Schrijf de berekening met die namen, en geef ook de uitkomst een naam.
        3. Laat de uitkomst zien met `print`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Twee pizza's van € 9,50 per stuk, en € 2,50 voor de bezorging. Wat betaal je? In Python schrijf je die bedragen met een punt: `9.5` en `2.5`.
        """),
        mo.md(f"```python\n{C.EXAMPLE['nl']}\n```"),
        mo.md(r"""
        Dit programma laat zien:
        """),
        mo.md(f"```\n{C.EXAMPLE_SHOWS}```"),
        mo.md(r"""
        `count` is een int, een geheel getal; `pizza` en `delivery` zijn floats. Python doet `*` vóór `+`: eerst is `9.5 * 2` gelijk aan `19.0`, en daar komt `2.5` bij: `21.5`. Zelf proberen? Kopieer de code naar het kladblok en voer hem uit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_example_power():
    mo.vstack([
        mo.md(r"""
        ### Een tweede voorbeeld, met een macht

        Drie dobbelstenen, een rode, een blauwe en een groene, elk met 6 kanten. Hoeveel verschillende worpen zijn er? 6 voor de rode, keer 6 voor de blauwe, keer 6 voor de groene: 6 drie keer met zichzelf vermenigvuldigd, een macht.
        """),
        mo.md(f"```python\n{C.POWER['nl']}\n```"),
        mo.md(r"""
        Dit programma laat zien:
        """),
        mo.md(f"```\n{C.POWER_SHOWS}```"),
        mo.md(r"""
        `sides ** dice` is 6 tot de macht 3: `6 * 6 * 6`, dus 216 worpen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_coffee():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Koffie

        Drie koppen koffie kosten € 2,50 per stuk. Geef in het vak de prijs de naam `price` en het aantal koppen de naam `cups`, reken het totaal uit en noem het `total`, en laat het zien met `print`. Vervang elke `...`.
        """),
        L.task("Laat de totaalprijs zien van drie koppen van € 2,50."),
    ])
    return


@app.cell(hide_code=True)
def exercise_coffee():
    box_coffee = L.code_box("coffee", lines=5)
    box_coffee
    return (box_coffee,)


@app.cell(hide_code=True)
def check_coffee(box_coffee):
    result_coffee = L.check("coffee", box_coffee.value)
    result_coffee
    return (result_coffee,)


@app.cell(hide_code=True)
def hints_coffee():
    L.hints(
        "coffee",
        [
            "Stap 1 van het recept: `price` krijgt de prijs van één kop, en `cups` het aantal koppen.",
            "Stap 2: het totaal is de prijs keer het aantal koppen. Keer is `*`.",
            "In het uitgewerkte voorbeeld is het totaal `pizza * count + delivery`. Hier is er geen bezorging.",
        ],
        why="Elk getal krijgt een naam; de berekening gebruikt de namen, en ook de uitkomst krijgt een naam. `print(total)` laat dan 7.5 zien.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lock():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een cijferslot

        Een slot gaat open met een code van 4 cijfers, en elk cijfer kan 0 tot en met 9 zijn. Hoeveel verschillende codes zijn er? Met 1 cijfer zijn er 10 codes, met 2 cijfers 10 keer 10. In het vak staat al `digits = 4`; reken uit hoeveel codes er zijn, en noem dat `codes`.
        """),
        L.task("Laat zien hoeveel codes van 4 cijfers er zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lock():
    box_lock = L.code_box("lock", lines=4)
    box_lock
    return (box_lock,)


@app.cell(hide_code=True)
def check_lock(box_lock):
    result_lock = L.check("lock", box_lock.value)
    result_lock
    return (result_lock,)


@app.cell(hide_code=True)
def hints_lock():
    L.hints(
        "lock",
        [
            "Elk cijfer erbij vermenigvuldigt het aantal codes met 10. Hoe vaak vermenigvuldig je met 10 bij 4 cijfers?",
            "Vier tienen met elkaar vermenigvuldigen, één voor elk cijfer, is een macht: 10 tot de macht 4.",
            "Kijk naar het voorbeeld met de dobbelstenen: daar gaf `sides ** dice` het aantal worpen. Hier heeft elk cijfer 10 keuzes, en zijn er `digits` cijfers.",
        ],
        why="`codes = 10 ** digits` geeft 10000: 10 keuzes voor elk van de 4 cijfers. `10 * 10 * 10 * 10` is net zo goed.",
    )
    return


@app.cell(hide_code=True)
def text_divide():
    mo.md(r"""
    ### Oefening 3. Delen
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "divide",
        question="Wat laat `print(8 / 4)` zien?",
        options={
            "float": "`2.0`",
            "whole": "`2`",
            "comma": "`2,0`",
        },
        correct="float",
        why={
            "whole": "Delen met `/` geeft een float, ook als het precies uitkomt. Kijk nog eens naar de zin onder de tabel met tekens.",
            "comma": "Python schrijft een punt, nooit een komma, ook in wat het laat zien. Kijk nog eens naar de tabel met nieuwe woorden.",
            "float": "Delen met `/` geeft altijd een float, ook als het precies uitkomt: `8 / 4` geeft `2.0`, met een punt.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je eerste programma](/course/open/py-first-program?lang=nl) liet je tekst zien; nu laat je getallen zien die Python voor je uitrekent. Vanaf nu kom je overal namen tegen: later in de course bewaart ook een programma dat leert van voorbeelden wat het leert onder namen. In de volgende les zie je wat namen doen in een notebook, waar elke naam bij één cel hoort.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_coffee, result_lock):
    L.wrap_up(result_coffee, result_lock, choice_divide)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
