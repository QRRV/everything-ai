# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-terminal. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, paths
    from everything_ai.lesson import lesson

    L = lesson("py-terminal", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl), [Code uit modules gebruiken](/course/open/py-imports?lang=nl) en [Een tekstbestand lezen](/course/open/py-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Waar vindt `from tools import stats` in de course de module `stats`?",
        options={
            "lesson": "In de map van de les die de regel uitvoert",
            "project": "In `work/projects/tools/stats.py`",
            "python": "Tussen de modules die bij Python horen, zoals `math`",
        },
        correct="project",
        why={
            "lesson": "De eigen map van een les is niet waar je packages staan. Zie de tabel met nieuwe woorden in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
            "project": "De course zet je projectmap, `work/projects/`, op de lijst met mappen van Python; daar vindt hij het package `tools`, en daarin `stats.py`.",
            "python": "`stats` is een module die je zelf hebt geschreven. Waar bewaart de course die? Zie de figuur in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="Een programma zet `random.seed(7)` en gooit daarna drie keer een dobbelsteen. Je voert het opnieuw uit. Wat gebeurt er?",
        options={
            "same": "Het gooit weer dezelfde drie getallen",
            "new": "Het gooit drie nieuwe getallen",
            "sevens": "Het gooit drie zevens",
        },
        correct="same",
        why={
            "same": "Dezelfde seed laat `random` zijn vaste berekening op dezelfde plek beginnen, dus komen dezelfde getallen eruit.",
            "new": "Zonder seed zou dat meestal gebeuren. Wat doet een seed? Zie het idee in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "sevens": "De seed is waar de berekening begint, niet het getal dat eruit komt. Zie het recept in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="In de course opent een programma het pad `\"lessons/py-files/poem.txt\"`. Waar begint dat pad?",
        options={
            "home": "Bij je thuismap",
            "file": "Bij het bestand `poem.txt`",
            "course": "Bij de map van de course, waar je de course startte",
        },
        correct="course",
        why={
            "home": "Zo'n pad begint bij de map waarin Python draait, en dat is niet je thuismap. Zie het idee in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "file": "Bij het bestand eindigt het pad; daar begint het niet. Zie de tabel met nieuwe woorden in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "course": "Python draait in de map van de course, dus daar begint een pad: daarin staat de map `lessons`, daarin `py-files`, en daarin het bestand.",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Zolang een cel draait, wacht de les erop. Voor een seconde is dat prima, maar sommige klussen duren minuten, of uren, zoals een groot model trainen. Zo'n lange klus kan draaien als script: een `.py`-bestand dat je vanuit de terminal start. Het draait vanzelf, print hoe ver het is, en de les blijft vrij.

    **Een voorbeeld uit het dagelijks leven.** Een wasmachine. Je kiest een programma, drukt op start, en hij draait vanzelf; ondertussen doe je iets anders, en het schermpje zegt hoe ver hij is.
    """)
    return


@app.cell(hide_code=True)
def predict_wait():
    guess_wait = L.guess(
        "wait",
        question="Een cel in een les draait een loop die een uur duurt. Wat denk je dat er gebeurt als je ondertussen een andere cel uitvoert?",
        options={
            "parallel": "Ze draaien allebei tegelijk",
            "waits": "De andere cel wacht tot de eerste klaar is",
            "stops": "De eerste cel stopt, en de andere draait",
        },
        reveal="**De andere cel wacht.** Een les voert één cel tegelijk uit. Zolang de lange cel draait, wacht al het andere. Een script in een terminal heeft een eigen Python, dus blijft de les vrij.",
    )
    guess_wait
    return (guess_wait,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een terminal is een venster waarin je commando's typt. Je gebruikte er al een: je startte de course met `uv run everything-ai`. In een tweede terminal voer je je eigen scripts uit.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **terminal** | een venster waarin je een commando typt en op Enter drukt om het uit te voeren | |
    | **commando** | één regel in de terminal: een programma, en wat het moet doen | `uv run everything-ai` |
    | **de map waarin je zit** | de terminal werkt in één map tegelijk; een pad in een commando begint daar | `cd` gaat naar een map |
    | **script** | een `.py`-bestand dat je als één programma uitvoert, van de eerste tot de laatste regel | `dice.py` |
    | **`uv run python`** | voert een Python-bestand uit met de eigen Python en modules van de course | `uv run python work/projects/scripts/dice.py` |
    | **Ctrl+C** | stopt het commando dat in de terminal draait; ook op een Mac, met de Control-toets, niet Cmd | |

    **Een tweede terminal, op jouw systeem:**

    - **Mac:** de terminal is de app *Terminal*. Staat Terminal voorop, dan opent Cmd+N een tweede venster.
    - **Linux:** de terminal-app van je systeem. In de meeste opent Ctrl+Shift+N een tweede venster; of open de app nog een keer vanuit het menu.
    - **Windows:** de course draait in WSL. Open *Ubuntu* nog een keer vanuit het Startmenu, net als toen je de course startte. Klik in Windows Terminal op het pijltje naast de +, en kies Ubuntu: de + zelf opent vaak PowerShell, en daar werken deze commando's niet.
    """)
    return


@app.cell(hide_code=True)
def picture_terminal():
    mo.vstack([
        draw.marked_lines([("Terminal", [(f'$ cd "{paths.ROOT}"', "1"), ("$ uv run python work/projects/scripts/dice.py", "2"),
                                         ("part 1 of 5 done", "3"), ("part 2 of 5 done", ""), ("...", ""),
                                         ("… sixes in ten million rolls", ""), ("$", "4")])],
                          {"1": "ga naar de map van de course",
                           "2": "voer het script uit, met zijn pad vanaf de map van de course",
                           "3": "wat het script print, verschijnt hier, terwijl het draait",
                           "4": "de $ is terug: de terminal is weer vrij"},
                          title="Een script in de terminal"),
        mo.md(r"""
        Jij typt wat na de `$` komt; de rest laat de terminal zien. De `$` is het teken van de terminal dat hij op een commando wacht: die typ je nooit zelf. Op een Mac is dat teken vaak `%`. Je terminal laat er misschien iets langers voor zien, zoals je naam en de map waarin je zit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Laat de terminal waarin de course draait open: sluit je hem, dan stopt de course. Ctrl+C vraagt daar of je de course wilt stoppen; druk op Enter om hem te laten draaien. Per ongeluk gesloten? Ga naar de map van de course en typ opnieuw `uv run everything-ai`."),
        mo.md(r"""
        ### Recept

        Zo voer je een script uit vanuit de terminal:

        1. Open een tweede terminal, naast die waarin de course draait.
        2. Ga naar de map van de course: `cd`, een spatie, en de map tussen dubbele aanhalingstekens.
        3. Voer het script uit: `uv run python`, een spatie, en zijn pad vanaf de map van de course.
        4. Lees wat het print. Is de `$` terug, dan is het klaar; Ctrl+C stopt het eerder.

        Zegt de terminal `No such file or directory`? Dan zit hij nog niet in de map van de course: doe stap 2 opnieuw.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Toen je deze les opende, zette de course een script in je projectmap: `work/projects/scripts/dice.py`. Het gooit tien miljoen keer een dobbelsteen, in vijf delen, en telt de zessen:
        """),
        mo.md(f"```python\n{C.DICE}\n```"),
        mo.md(f"""
        De map van je course is `{paths.ROOT}`. Dit doe je, stap voor stap.

        **Stap 1.** Open een tweede terminal.

        **Stap 2.** Ga naar de map van de course, met precies deze regel:

        ```text
        cd "{paths.ROOT}"
        ```

        **Stap 3.** Voer het script uit:

        ```text
        uv run python work/projects/scripts/dice.py
        ```

        **Stap 4.** Lees wat het print, tot de `$` terug is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het script print na elk deel een regel, terwijl het draait, en aan het eind hoeveel zessen het gooide. Er zit een loop in een loop: de buitenste loop gaat door de 5 delen, en per deel gooit de binnenste loop 2 miljoen keer. Het gebruikt `random.seed(7)`, dus krijg je elke keer dezelfde telling.

    Probeer nu de Ctrl+C van stap 4: voer het script opnieuw uit, en druk op Ctrl+C terwijl het draait. Het stopt eerder, en Python print een paar regels die eindigen op `KeyboardInterrupt`. Dat betekent alleen dat jij het stopte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_command():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Het goede commando
    """)
    return


@app.cell(hide_code=True)
def choice_command_cell():
    choice_command = L.choice(
        "which-command",
        question="Je terminal zit in de map van de course. Welk commando voert het script `work/projects/scripts/dice.py` uit?",
        options={
            "uv": "`uv run python work/projects/scripts/dice.py`",
            "short": "`uv run python dice.py`",
            "bare": "`python work/projects/scripts/dice.py`",
        },
        correct="uv",
        why={
            "uv": "`uv run python` voert het bestand uit met de eigen Python van de course, en het pad begint bij de map van de course, waar de terminal zit.",
            "short": "De terminal zit in de map van de course, en daar staat `dice.py` niet: een pad begint bij de map waarin je zit. Kijk nog eens naar stap 3 van het recept.",
            "bare": "Zonder `uv run` is dit niet de eigen Python van de course, als er al een `python` is: op veel computers zegt de terminal `command not found`. Kijk nog eens naar de tabel met nieuwe woorden.",
        },
    )
    choice_command
    return (choice_command,)


@app.cell(hide_code=True)
def text_exercise_dice():
    mo.md(r"""
    ### Oefening 2. Voer het zelf uit

    Volg het uitgewerkte voorbeeld: open een tweede terminal, ga naar de map van de course, en voer `dice.py` uit. Beantwoord dan de vraag.
    """)
    return


@app.cell(hide_code=True)
def choice_dice_cell():
    choice_dice = L.choice(
        "dice-result",
        question="Wat is de laatste regel die `dice.py` in je terminal print?",
        options={
            "low": "`1665730 sixes in ten million rolls`",
            "real": "`1666892 sixes in ten million rolls`",
            "high": "`1667104 sixes in ten million rolls`",
        },
        correct="real",
        why={
            "low": "Dat print het script niet. Voer het uit, en lees de laatste regel.",
            "real": "Dat printte het script: dicht bij één op zes, wat 1.666.667 zou zijn, maar niet precies, omdat elke worp toeval is.",
            "high": "Dat print het script niet. Voer het uit, en lees de laatste regel.",
        },
    )
    choice_dice
    return (choice_dice,)


@app.cell(hide_code=True)
def text_exercise_countdown():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je eigen script

        Schrijf een script dat aftelt: het print `3`, `2` en `1`, één per regel, en daarna `go`. Vervang de `...`. De check voert het uit en kijkt wat het print. Klopt het, dan bewaart de check het in zijn geheel, als `work/projects/scripts/countdown.py`. Voer het daarna uit in je terminal, zoals in het uitgewerkte voorbeeld.
        """),
        L.task("Laat het script 3, 2, 1 en go printen, één per regel; voer het daarna uit in je terminal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_countdown():
    box_countdown = L.code_box("countdown", lines=6)
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
            "Een loop over `range(3)` geeft 0, 1 en 2. Welk getal print je, zodat het 3, 2 en 1 geeft?",
            "Print in de loop 3 min het getal van de loop; print na de loop, helemaal links, één keer `go`.",
            "`for i in range(3):`, ingesprongen `print(___)`, en dan helemaal links `print(\"go\")`.",
        ],
        why="`3 - i` is 3, 2 en 1 voor `i` = 0, 1 en 2. `print(\"go\")` staat na de loop, dus draait hij één keer. In de terminal print `uv run python work/projects/scripts/countdown.py` dezelfde vier regels.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl) bewaarde de check een functie in je project, zodat lessen hem kunnen binnenhalen; nu bewaarde de check een heel script, zodat jij het kunt uitvoeren. In [Een tekstbestand lezen](/course/open/py-files?lang=nl) begon een pad bij de map van de course; in de terminal ga je daar zelf heen, met `cd`. In [Code uit modules gebruiken](/course/open/py-imports?lang=nl) maakte `random.seed` toeval herhaalbaar; `dice.py` gebruikt dat, dus geeft het elke keer dezelfde telling. Later in de course kan een lange klus, zoals een model trainen, als zo'n script draaien: het print zijn loss terwijl het draait, en draait zo lang als nodig is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_command, choice_dice, result_countdown):
    L.wrap_up(choice_command, choice_dice, result_countdown)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
