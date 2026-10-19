# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-plotting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import plots
    from everything_ai.lesson import lesson

    L = lesson("py-plotting", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lijsten](/course/open/py-lists?lang=nl) en [Code uit modules gebruiken](/course/open/py-imports?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="Wat laat dit programma zien?\n\n```python\ntemps = [12, 15, 9, 14]\nprint(len(temps))\n```",
        options={
            "last": "`14`",
            "count": "`4`",
            "index": "`3`",
        },
        correct="count",
        why={
            "last": "Dat is het laatste item. `len` telt de items. Zie de tabel met nieuwe woorden in [Lijsten](/course/open/py-lists?lang=nl).",
            "count": "`len` telt de items in de lijst: vier temperaturen.",
            "index": "3 is de index van het laatste item, omdat indexen bij 0 beginnen. `len` telt de items zelf. Zie [Lijsten](/course/open/py-lists?lang=nl).",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="Het programma begint met `import random`. Welke regel gooit een dobbelsteen?",
        options={
            "bare": "`randint(1, 6)`",
            "module": "`random(1, 6)`",
            "dot": "`random.randint(1, 6)`",
        },
        correct="dot",
        why={
            "bare": "Na `import random` zet je `random.` voor elke naam uit de module. Zie het blok *Let op* in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "module": "`random` is de module, een kist met gereedschap; je roept één stuk gereedschap eruit aan, met een punt. Zie het recept in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "dot": "De naam van de module, een punt, en de naam van de functie erin: `random.randint(1, 6)`.",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een lijst van honderd getallen lees je niet makkelijk. Teken hem, en je ziet in één oogopslag of hij stijgt, daalt of alle kanten op springt. Later in de course zie je zo hoe een model leert: een lijn van zijn loss, één getal voor hoe fout het nog is, stap na stap, laat zien of het nog beter wordt.

    **Een voorbeeld uit het dagelijks leven.** Een weerapp geeft je geen tabel met de temperaturen van volgende week: hij tekent een lijn. Je ziet meteen dat het donderdag warmer wordt.
    """)
    return


@app.cell(hide_code=True)
def predict_line():
    guess_line = L.guess(
        "line",
        question="Wat denk je dat dit programma tekent?\n\n```python\nimport matplotlib.pyplot as plt\n\nplt.plot([1, 2, 3], [2, 4, 6])\nplt.show()\n```",
        options={
            "line": "Een rechte lijn die omhooggaat",
            "bars": "Drie staven, van 2, 4 en 6 hoog",
            "numbers": "De getallen 2, 4 en 6, één per regel",
        },
        reveal="**Een rechte lijn die omhooggaat.** `plt.plot` zet de eerste lijst, `[1, 2, 3]`, langs de onderkant en de tweede, `[2, 4, 6]`, langs de zijkant, zet bij elk paar een punt, en trekt een lijn door de punten. Bij elke stap naar rechts gaat de lijn 2 omhoog.",
    )
    guess_line
    return (guess_line,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Matplotlib tekent grafieken van lijsten met getallen. De module `pyplot` heeft een functie voor elke soort grafiek, en je haalt het binnen onder de korte naam `plt`.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **Matplotlib** | een library: een verzameling modules die iemand anders schreef, geïnstalleerd met de course; hij tekent grafieken, en `pyplot` is de module die je gebruikt | `import matplotlib.pyplot as plt` |
    | **`as`** | geeft een module een eigen, korte naam | `as plt` |
    | **x-as, y-as** | de lijn langs de onderkant, en de lijn langs de zijkant omhoog; elk punt heeft een x-waarde en een y-waarde | `plt.xlabel("dag")` |
    | **lijngrafiek** | punten verbonden door een lijn, voor waarden in een volgorde, zoals in de tijd | `plt.plot(x, y)` |
    | **scatter plot** | losse punten, voor paren metingen zonder volgorde | `plt.scatter(x, y)` |
    | **staafdiagram** | een staaf per naam, zo hoog als de waarde ervan, om losse dingen te vergelijken | `plt.bar(names, heights)` |

    Elke functie krijgt eerst de x-waarden en dan de y-waarden, als twee even lange lijsten: de eerste x hoort bij de eerste y, enzovoort. `plt.title`, `plt.xlabel` en `plt.ylabel` zeggen wat de lezer ziet. `plt.show()` laat de grafiek onder de cel zien.
    """)
    return


@app.cell(hide_code=True)
def picture_kinds():
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 3, figsize=(9.6, 2.6))
    _axes[0].plot([1, 2, 3, 4, 5], [12, 15, 9, 14, 16])
    _axes[0].set_title("Lijngrafiek: op volgorde")
    _axes[1].scatter([150, 160, 165, 172, 180, 186], [48, 55, 60, 64, 75, 79])
    _axes[1].set_title("Scatter plot: paren")
    _axes[2].bar(["lezen", "coderen", "lopen"], [30, 45, 20])
    _axes[2].set_title("Staafdiagram: vergelijken")
    _fig.tight_layout()
    _picture = mo.Html(plots.render(_fig))
    _plt.close(_fig)
    mo.vstack([
        _picture,
        mo.md(r"""
        Dezelfde soort getallen kan er heel anders uitzien. Een lijn verbindt punten in hun volgorde, zoals temperaturen dag na dag. Een scatter plot laat de punten los, zoals de lengte en het gewicht van zes mensen. Een staafdiagram zet losse dingen naast elkaar, zoals minuten per activiteit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("De x-waarden en de y-waarden moeten twee even lange lijsten zijn: één x voor elke y. En `plt.show()` komt als laatste: die laat de grafiek zien, en de volgende `plt.plot` begint een nieuwe."),
        mo.md(r"""
        ### Recept

        Zo teken je een grafiek:

        1. Haal bovenaan pyplot binnen onder de korte naam `plt`: `import matplotlib.pyplot as plt`.
        2. Teken met één functie: `plt.plot(x, y)` voor een lijn, `plt.scatter(x, y)` voor losse punten, `plt.bar(names, heights)` voor staven.
        3. Zeg wat de lezer ziet: `plt.title(...)`, `plt.xlabel(...)` en `plt.ylabel(...)`.
        4. Laat de grafiek zien met `plt.show()`, als laatste regel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hoeveel minuten je elke dag van een week naar school liep, en drie manieren om er te komen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_plot():
    # 1. haal pyplot binnen onder de korte naam plt
    import matplotlib.pyplot as plt

    _days = [1, 2, 3, 4, 5, 6, 7]
    _walk_minutes = [14, 12, 15, 11, 13, 16, 12]
    # 2. teken: eerst de x-waarden, dan de y-waarden
    plt.plot(_days, _walk_minutes)
    # 3. zeg wat de lezer ziet
    plt.title("Minuten naar school per dag")
    plt.xlabel("dag")
    plt.ylabel("minuten")
    # 4. laat hem zien; de volgende tekening begint een nieuwe grafiek
    plt.show()

    # 2. staven: eerst de namen, dan de hoogtes
    plt.bar(["fiets", "bus", "lopen"], [9, 14, 25])
    # 3. wat de lezer ziet
    plt.title("Minuten naar school, op drie manieren")
    plt.ylabel("minuten")
    # 4. laat hem zien
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen twee grafieken. De eerste is een lijn door de zeven dagen: dag 1 bij 14 minuten, dag 2 bij 12, enzovoort. `plt.show()` maakt hem af, dus beginnen de staven op een nieuwe grafiek: één staaf per manier om naar school te gaan. Verander `plt.plot` in `plt.scatter` en voer de cel opnieuw uit: nu zijn de dagen losse punten. De naam `plt` wordt in deze cel gemaakt, dus andere cellen kunnen hem meteen gebruiken; schrijf `import matplotlib.pyplot as plt` niet in een andere cel, want dat botst met *één naam, één cel*. De oefenvakken draaien los, dus elk vak haalt hem zelf binnen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_temps():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een week temperaturen

        Teken `temps` als lijngrafiek, met `days` langs de onderkant, een titel, en een label bij beide assen. Vervang de `...`. De check kijkt naar wat je grafiek laat zien: een lijn door de zeven temperaturen, dag 1 tot en met 7, met een titel en twee aslabels. Elke titel en elk label dat zegt wat de grafiek laat zien, is goed. In een oefenvak laat `plt.show()` de grafiek niet zien; de check laat hem onder zijn melding zien.
        """),
        L.task("Teken `temps` als lijn met `days` langs de onderkant, met `plt.title`, `plt.xlabel` en `plt.ylabel`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_temps():
    box_temps = L.code_box("temps-line", lines=10)
    box_temps
    return (box_temps,)


@app.cell(hide_code=True)
def check_temps(box_temps):
    result_temps = L.check("temps-line", box_temps.value)
    result_temps
    return (result_temps,)


@app.cell(hide_code=True)
def hints_temps():
    L.hints(
        "temps-line",
        [
            "Stap 2 en 3 van het recept. Temperaturen dag na dag hebben een volgorde: welke soort grafiek verbindt punten in hun volgorde?",
            "Teken de lijn met de dagen als x-waarden en de temperaturen als y-waarden; voeg dan een titel en de twee aslabels toe.",
            "`plt.plot(___, ___)`, dan `plt.title(\"...\")`, `plt.xlabel(\"...\")` en `plt.ylabel(\"...\")`.",
        ],
        why="`plt.plot(days, temps)` zet een punt bij elke dag en zijn temperatuur, en verbindt ze op volgorde. `plt.title(\"Temperatuur per dag\")`, `plt.xlabel(\"dag\")` en `plt.ylabel(\"graden\")` zeggen wat de lezer ziet.",
    )
    return


@app.cell(hide_code=True)
def text_which_chart():
    mo.md(r"""
    ### Oefening 2. Welke grafiek?
    """)
    return


@app.cell(hide_code=True)
def choice_which_chart_cell():
    choice_which_chart = L.choice(
        "which-chart",
        question="Je hebt de lengte en het gewicht van dertig mensen gemeten, en je wilt zien of langere mensen meer wegen. Welke grafiek past het best?",
        options={
            "scatter": "Een scatter plot, met de lengte langs de onderkant en het gewicht langs de zijkant",
            "line": "Een lijngrafiek, met de lengte langs de onderkant en het gewicht langs de zijkant",
            "bar": "Een staafdiagram, met een staaf per persoon, zo hoog als het gewicht",
        },
        correct="scatter",
        why={
            "scatter": "Elke persoon is een paar metingen, en de mensen hebben geen volgorde. Losse punten laten zien of de punten naar rechts omhooggaan.",
            "line": "Een lijn verbindt punten in een volgorde, zoals dag na dag. Hebben dertig mensen zo'n volgorde? Kijk nog eens naar de figuur met de drie soorten.",
            "bar": "Dertig staven vergelijken de mensen één voor één, maar waar is de lengte dan? Welke grafiek laat twee metingen van elke persoon zien?",
        },
    )
    choice_which_chart
    return (choice_which_chart,)


@app.cell(hide_code=True)
def text_exercise_minutes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Minuten per activiteit

        Teken `minutes` als staafdiagram, met een staaf voor elke naam in `names`, een titel, en een label bij de y-as. Vervang de `...`. De check kijkt naar wat je grafiek laat zien: vier staven, `read`, `code`, `walk` en `sleep`, zo hoog als het aantal minuten, met een titel en een label bij de y-as. Je grafiek verschijnt onder de melding van de check.
        """),
        L.task("Teken een staaf per naam, zo hoog als het aantal minuten, met een titel en een label bij de y-as."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minutes():
    box_minutes = L.code_box("minutes-bar", lines=10)
    box_minutes
    return (box_minutes,)


@app.cell(hide_code=True)
def check_minutes(box_minutes):
    result_minutes = L.check("minutes-bar", box_minutes.value)
    result_minutes
    return (result_minutes,)


@app.cell(hide_code=True)
def hints_minutes():
    L.hints(
        "minutes-bar",
        [
            "Stap 2 van het recept: welke functie tekent een staaf voor elke naam?",
            "De namen gaan langs de onderkant en de minuten geven de hoogtes, in die volgorde; dan een titel en een label bij de y-as.",
            "`plt.bar(___, ___)`, dan `plt.title(\"...\")` en `plt.ylabel(\"...\")`.",
        ],
        why="`plt.bar(names, minutes)` tekent een staaf voor elke naam, zo hoog als het aantal minuten. De staaf van `sleep` steekt ver boven de rest uit: een grafiek laat dat meteen zien. `plt.title` en `plt.ylabel` zeggen wat de staven betekenen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) bewaarde je getallen in een lijst; nu teken je ze. In [Code uit modules gebruiken](/course/open/py-imports?lang=nl) haalde je `math` en `random` binnen; Matplotlib haal je op dezelfde manier binnen, met een eigen, korte naam. Later in de course teken je de loss van een model, hoe fout het nog is, als lijngrafiek, stap na stap. In de volgende les tel je de letters in een tekst en teken je ze als staafdiagram.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which_chart, result_minutes, result_temps):
    L.wrap_up(result_temps, choice_which_chart, result_minutes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
