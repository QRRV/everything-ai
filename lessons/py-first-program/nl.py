# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-first-program. Route: course/route/lessen/stam-1-basis.yaml. Checks en het programma in het beeld: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-first-program", "nl")
    C = L.checks   # het programma in het beeld, en wat het na elke regel laat zien


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Je voert een codecel uit. Waar verschijnt de uitkomst?",
        options={
            "start": "Op de startpagina.",
            "under": "Direct onder de code.",
            "window": "In een nieuw venster.",
        },
        correct="under",
        why={
            "under": "Direct onder de code. Daar laten ook je eigen programma's hun uitkomst zien.",
            "start": "De startpagina laat de route en je voortgang zien, niet de uitkomst van cellen. Voer de eerste cel van [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl) nog eens uit en kijk waar de uitkomst verschijnt.",
            "window": "Alles blijft in de les zelf. Voer de eerste cel van [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl) nog eens uit en kijk waar de uitkomst verschijnt.",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def quick_check_scratchpad():
    quick_scratchpad = L.choice(
        "quick-scratchpad",
        question="Je wilt een stukje code uitproberen zonder de les te veranderen. Waar doe je dat?",
        options={
            "box": "In het vak van een oefening.",
            "start": "Op de startpagina.",
            "scratchpad": "In het kladblok.",
        },
        correct="scratchpad",
        why={
            "scratchpad": "Het kladblok is er om dingen uit te proberen; de les blijft zoals hij is.",
            "box": "Wat je in het vak van een oefening typt, wordt nagekeken als je antwoord. Kijk bij *Iets uitproberen* in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
            "start": "Op de startpagina is geen plek voor code. Kijk bij *Iets uitproberen* in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
        },
    )
    quick_scratchpad
    return (quick_scratchpad,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een computer kan veel, maar uit zichzelf doet hij niets: hij voert alleen instructies uit. Wil je dat hij iets doet, dan schrijf je die instructies op. Vanaf deze les doe je dat zelf, in Python. Zo maak je alles in deze course, van een programma dat letters telt tot een programma dat leert van voorbeelden.

    **Een voorbeeld uit het dagelijks leven.** Een recept. Dat is een lijst stappen, en een kok voert ze een voor een uit, van boven naar beneden. Verwissel je twee stappen, dan krijg je iets anders. En de kok doet precies wat er in het recept staat: staat er twee keer "voeg zout toe", dan gaat het zout er twee keer in.
    """)
    return


@app.cell(hide_code=True)
def predict_quotes():
    guess_quotes = L.guess(
        "no-quotes",
        question="`print(\"Hallo\")` laat `Hallo` onder de cel zien. Wat denk je dat er gebeurt met `print(Hallo)`, zonder aanhalingstekens?",
        options={
            "same": "Dat laat ook `Hallo` zien.",
            "error": "Python stopt met een foutmelding.",
            "nothing": "Er verschijnt niets.",
        },
        reveal="**Python stopt met een foutmelding.** Zonder aanhalingstekens leest Python `Hallo` als een woord dat Python al moet kennen, zoals `print`, en dat woord kent het niet. Met aanhalingstekens is `\"Hallo\"` gewoon tekst, die Python precies zo laat zien.",
    )
    guess_quotes
    return (guess_quotes,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **programma** | een lijst instructies voor de computer, die ze een voor een uitvoert, van boven naar beneden | de code in een cel |
    | **print** | een instructie die laat zien wat tussen de haakjes staat, op een nieuwe regel onder de cel | `print("Hallo")` |
    | **string** | een stuk tekst tussen aanhalingstekens; Python laat het precies zo zien, en probeert het niet te begrijpen | `"Hallo"` |
    | **commentaar** | een regel die met `#` begint; Python slaat die over, en zo zet je aantekeningen voor mensen bij de code | `# een aantekening` |

    Hieronder staat een programma met drie instructies. Python voert eerst de eerste regel uit, dan de tweede, dan de derde.
    """)
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        draw.program_trace(C.STEPS["nl"], C.SHOWN["nl"], code_label="Code",
                           output_label="Wat de regel onder de cel laat zien", title="Een programma, regel voor regel"),
        mo.md(r"""
        Python voert de regels uit van boven naar beneden, en elke `print` zet zijn tekst op een nieuwe regel onder de cel, in de volgorde van de code. De aanhalingstekens verschijnen niet: die vertellen Python alleen waar de tekst begint en eindigt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Python let heel precies op wat je schrijft. `Print` met een hoofdletter is iets anders dan `print`, en aanhalingstekens moeten recht zijn, zoals `\"`, niet gekruld, zoals `“`. Typ je in een vak, dan krijg je rechte; gekrulde komen uit tekst die je uit een document of webpagina kopieert."),
        mo.md(r"""
        ### Recept

        Zo laat je regels tekst zien:

        1. Schrijf `print`, met ronde haakjes erachter: `print()`.
        2. Zet de tekst tussen aanhalingstekens binnen de haakjes: `print("Hallo")`.
        3. Schrijf één `print` voor elke regel, in de volgorde waarin de regels moeten verschijnen.
        4. Voer de cel uit, en vergelijk wat er verschijnt met wat je bedoelde.

        ### Uitgewerkt voorbeeld

        Een boodschappenlijstje van drie regels. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
        """),
    ])
    return


@app.cell
def example_shopping():
    # 1 en 2. print, en de tekst tussen aanhalingstekens binnen de haakjes
    print("Boodschappen")
    # 3. één print voor elke regel, in de volgorde van de regels
    print("brood")
    print("kaas")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        4. Onder de cel verschijnen `Boodschappen`, `brood` en `kaas`, elk op een eigen regel, in de volgorde van de code. Het commentaar verschijnt niet: Python slaat het over.
        """),
        L.task("Zet `print(\"kaas\")` in de cel hierboven boven `print(\"brood\")`, en voer de cel opnieuw uit met **Shift + Enter**. Zo verplaats je een regel: klik erin, en druk op **Alt + ↑** (op een Mac **Option + ↑**)."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_tools():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Drie regels

        Deze course gebruikt drie tools, die je een voor een leert kennen: `Python`, `NumPy` en `PyTorch`. Schrijf in het vak een programma dat die drie namen laat zien, elk op een eigen regel, in deze volgorde. De check leest wat je programma laat zien.
        """),
        L.task("Laat `Python`, `NumPy` en `PyTorch` zien, elk op een eigen regel, in deze volgorde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tools():
    box_tools = L.code_box("tools", lines=4)
    box_tools
    return (box_tools,)


@app.cell(hide_code=True)
def check_tools(box_tools):
    result_tools = L.check("tools", box_tools.value)
    result_tools
    return (result_tools,)


@app.cell(hide_code=True)
def hints_tools():
    L.hints(
        "tools",
        [
            "Kijk naar stap 3 van het recept: één `print` voor elke regel die je wilt zien.",
            "De eerste regel van je programma kan `print(\"Python\")` zijn. De andere twee zien er net zo uit, met andere tekst.",
            "`print(\"Python\")`, dan `print(\"NumPy\")`, dan een derde regel met ___.",
        ],
        why="Drie instructies, een voor elke regel, in de volgorde waarin de regels moeten verschijnen. De hoofdletters tellen mee: Python laat tekst precies zo zien als je hem schrijft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_countdown():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Zet het op volgorde

        Het programma in het vak moet aftellen: `3`, `2`, `1` en dan `Go!`. De vier regels staan er allemaal, maar in de verkeerde volgorde. Verander de regels zelf niet; verplaats ze alleen, zoals je in het uitgewerkte voorbeeld een regel verplaatste.
        """),
        L.task("Zet de vier regels in de volgorde van een countdown."),
    ])
    return


@app.cell(hide_code=True)
def exercise_countdown():
    box_countdown = L.code_box("countdown", lines=5)
    box_countdown
    return (box_countdown,)


@app.cell(hide_code=True)
def check_countdown(box_countdown):
    result_countdown = L.check("countdown", box_countdown.value)
    result_countdown
    return (result_countdown,)


@app.cell(hide_code=True)
def hints_countdown():
    L.hints(
        "countdown",
        [
            "Python werkt van boven naar beneden. Welke regel moet als eerste uitgevoerd worden?",
            "Een countdown begint bij 3, dus `print(\"3\")` komt bovenaan.",
            "`print(\"Go!\")` komt als laatste.",
        ],
        why="Dezelfde vier instructies, nu in de volgorde waarin de regels moeten verschijnen: 3, 2, 1, Go!",
    )
    return


@app.cell(hide_code=True)
def text_quotes():
    mo.md(r"""
    ### Oefening 3. Welk programma?
    """)
    return


@app.cell(hide_code=True)
def choice_quotes_cell():
    choice_quotes = L.choice(
        "quotes",
        question="Welk programma laat `Hallo` op de ene regel zien, en `wereld` op de regel eronder?",
        options={
            "one": "`print(\"Hallo wereld\")`",
            "two": "`print(\"Hallo\")`, en daaronder `print(\"wereld\")`",
            "bare": "`print(Hallo)`, en daaronder `print(wereld)`",
        },
        correct="two",
        why={
            "two": "Twee instructies geven twee regels, en de tekst tussen aanhalingstekens verschijnt precies zoals hij er staat.",
            "one": "Eén `print` geeft één regel. Kijk nog eens naar stap 3 van het recept.",
            "bare": "Zonder aanhalingstekens leest Python `Hallo` als een woord dat Python al moet kennen, zoals `print`, en stopt het met een foutmelding. Kijk nog eens naar stap 2 van het recept.",
        },
    )
    choice_quotes
    return (choice_quotes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl) voerde je een cel met `print` erin uit zonder te weten wat dat was; nu weet je het. Elk programma in deze course werkt zo: regel voor regel, van boven naar beneden. In de volgende les geef je een waarde een naam, en reken je met getallen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_quotes, result_countdown, result_tools):
    L.wrap_up(result_tools, result_countdown, choice_quotes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
