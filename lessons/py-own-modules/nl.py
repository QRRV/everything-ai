# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-own-modules. Route: course/route/lessen/stam-1-basis.yaml. Checks, en de module die de les laat zien: checks.py.
# Het package dat deze les je geeft, staat in project/demo/; de course kopieert het naar work/projects/demo/.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-own-modules", "nl")
    C = L.checks   # de module die de les je geeft, als tekst


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Code uit modules gebruiken](/course/open/py-imports?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Wat laat dit programma zien?\n\n```python\nimport math\nprint(math.floor(3.7))\n```",
        options={
            "four": "`4`",
            "three": "`3`",
            "same": "`3.7`",
        },
        correct="three",
        why={
            "four": "`floor` rondt altijd naar beneden af, ook bij 3,7. Zie het idee in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "three": "`math.floor` rondt naar beneden af, tot het gehele getal eronder: 3.",
            "same": "`math.floor` verandert het getal: hij rondt het af. Zie het idee in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_from():
    quick_from = L.choice(
        "quick-from",
        question="Wat laat dit programma zien?\n\n```python\nfrom math import sqrt\nprint(sqrt(25))\n```",
        options={
            "error": "Het stopt: Python kent `sqrt` niet",
            "whole": "`5`",
            "float": "`5.0`",
        },
        correct="float",
        why={
            "error": "`from math import sqrt` haalt de naam `sqrt` zelf binnen, dus het werkt zonder `math.`. Zie het recept in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "whole": "Een wortel uit `math` is altijd een float. Zie de gok in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "float": "`from math import sqrt` haalt `sqrt` zelf binnen, en de wortel uit 25 is de float 5,0.",
        },
    )
    quick_from
    return (quick_from,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De code in de cellen van een les bestaat alleen in die les. Code die je opnieuw wilt gebruiken, zoals een functie die een gemiddelde uitrekent, hoort in een eigen bestand: een module, net als `math`, maar dan van jou. Latere lessen halen je module binnen, en zo groeit je project, bestand voor bestand, uit tot je eigen language model.

    **Een voorbeeld uit het dagelijks leven.** Je eigen gereedschapskist. Gereedschap dat je zelf hebt gemaakt, gaat in de kist, en de volgende keer dat je het nodig hebt, pak je het eruit in plaats van het opnieuw te maken. `math` is een kist die iemand anders heeft gevuld; je projectmap is van jou.
    """)
    return


@app.cell(hide_code=True)
def predict_import():
    guess_import = L.guess(
        "import",
        question="Je projectmap, `work/projects/`, heeft een bestand `demo/shapes.py` met de functie `square_area(side)`, die `side * side` teruggeeft. Wat denk je dat dit programma laat zien?\n\n```python\nfrom demo import shapes\nprint(shapes.square_area(4))\n```",
        options={
            "sixteen": "`16`",
            "error": "Het stopt: Python kan `demo` niet vinden",
            "text": "`square_area(4)`",
        },
        reveal="**`16`.** In de course kijkt Python in je projectmap, `work/projects/`, als je iets binnenhaalt. Daar vindt het de map `demo` met het bestand `shapes.py`, en `shapes.square_area(4)` roept de functie daarin aan.",
    )
    guess_import
    return (guess_import,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een eigen module is gewoon een `.py`-bestand met jouw code erin. Een map met zulke bestanden is een package. Je haalt ze binnen op dezelfde manier als `math`.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **module** (van jezelf) | een `.py`-bestand met jouw code; de naam zonder `.py` is de naam van de module | `stats.py` is de module `stats` |
    | **package** | een map met modules, met een bestand `__init__.py` erin; je haalt er een module uit met `from` | `from tools import stats` |
    | **projectmap** | in de course `work/projects/`: daar staan je packages, en daar kijkt Python als je iets binnenhaalt | `work/projects/tools/stats.py` |

    Python zoekt een module in een lijst met mappen. In elke les zet de course je projectmap op die lijst, dus vindt `from tools import stats` het bestand `work/projects/tools/stats.py`. Het bestand `__init__.py` mag leeg zijn: het maakt van de map een package. Dat is een vereenvoudiging: Python kan een map vaak ook zonder dat bestand binnenhalen, maar de course maakt het altijd, zodat je meteen ziet welke mappen packages zijn. Een module bevat vooral functies: de regels helemaal links draaien als een programma de module binnenhaalt.
    """)
    return


@app.cell(hide_code=True)
def picture_folders():
    mo.vstack([
        draw.marked_lines([("Je projectmap", [("work/", ""), ("  projects/", ""), ("    demo/", "1"), ("      __init__.py", ""),
                                             ("      shapes.py", "2")]),
                           ("In een cel", [("from demo import shapes", "1"), ("shapes.square_area(3)", "2")])],
                          {"1": "het package: een map, gevonden in de projectmap",
                           "2": "de module erin, en een functie uit die module"},
                          title="Waar een import zoekt"),
        mo.md(r"""
        `from demo import shapes` zoekt in de projectmap een map `demo` (1), en daarin het bestand `shapes.py` (2). Daarna is `shapes.square_area` de functie in dat bestand.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Geef je package een eigen naam, niet de naam van een module die je gebruikt, zoals `random` of `numpy`: Python kijkt als laatste in je projectmap, dus vindt het de andere, nooit de jouwe. En zet in een module alleen functies: een `print` helemaal links draait elke keer als een programma de module binnenhaalt, en verschijnt dan daar."),
        mo.md(r"""
        ### Recept

        Zo bewaar je code en gebruik je hem opnieuw:

        1. Zet de code die je wilt bewaren in een functie.
        2. Bewaar hem in een `.py`-bestand, in een packagemap onder `work/projects/`. In deze course doet een check dat voor je als je functie klopt.
        3. Haal de module binnen waar je hem nodig hebt: `from tools import stats`.
        4. Gebruik hem met de naam van de module en een punt: `stats.mean(temps)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Toen je deze les opende, zette de course een klein package in je projectmap: `work/projects/demo/`. Daarin staat het bestand `shapes.py`, de module `shapes`, met deze twee functies; stap 1 en 2 zijn dus al voor je gedaan. Het commentaar met `#` in de cel hieronder zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
        """),
        mo.md(f"```python\n{C.DEMO}\n```"),
    ])
    return


@app.cell
def example_own_module():
    # 3. haal de module binnen uit zijn package in je projectmap
    from demo import shapes
    # 4. gebruik hem met de naam van de module en een punt
    print(shapes.square_area(3))
    print(shapes.rectangle_area(2, 5))
    return (shapes,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen `9` en `10`. De functies staan helemaal niet in deze les: ze staan in een bestand in je projectmap, en elke les kan ze op dezelfde manier binnenhalen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een functie om te bewaren

        Schrijf `mean(numbers)`: het gemiddelde van een lijst getallen. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/tools/stats.py`; oefening 3 haalt hem daar binnen. De check roept hem aan met `[4, 8, 6]`, en met andere lijsten.
        """),
        L.task("Laat `mean(numbers)` het gemiddelde teruggeven; de check bewaart hem dan in je project."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean", lines=6)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean",
        [
            "Het gemiddelde is het totaal van de getallen, gedeeld door hoeveel het er zijn.",
            "Een loop telt de getallen op in `total`; deel na de loop door het aantal items.",
            "`return total / ___`, na de loop.",
        ],
        why="De loop telt elk getal op bij `total`; `total / len(numbers)` deelt door hoeveel het er zijn. Bij `[4, 8, 6]` is dat 18 / 3, dus `6.0`.",
    )
    return


@app.cell(hide_code=True)
def text_import_line():
    mo.md(r"""
    ### Oefening 2. De goede import
    """)
    return


@app.cell(hide_code=True)
def choice_import_line_cell():
    choice_import_line = L.choice(
        "import-line",
        question="Je functie `count_words` staat in het bestand `work/projects/text/words.py`. Welke regel haalt zijn module binnen, zodat `words.count_words(...)` werkt?",
        options={
            "file": "`import words.py`",
            "package": "`from text import words`",
            "path": "`import work.projects.text.words`",
        },
        correct="package",
        why={
            "file": "De naam van een module heeft geen `.py`, en `words` zit in een package. Kijk nog eens naar de tabel met nieuwe woorden.",
            "package": "Python kijkt in `work/projects/`, vindt het package `text`, en daarin de module `words`.",
            "path": "Python kijkt al in `work/projects/`, dus dat deel hoort niet bij de naam. Kijk nog eens naar de figuur.",
        },
    )
    choice_import_line
    return (choice_import_line,)


@app.cell(hide_code=True)
def text_exercise_use():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Gebruik je eigen module

        Het vak haalt je module `stats` binnen uit het package `tools`. Vervang de `...` door een regel die het gemiddelde van `temps` laat zien, met je eigen `stats.mean`. De check probeert ook andere temperaturen.
        """),
        L.task("Laat met `stats.mean` het gemiddelde van `temps` zien: `12.0`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_use():
    box_use = L.code_box("use-stats", lines=5)
    box_use
    return (box_use,)


@app.cell(hide_code=True)
def check_use(box_use, result_mean):
    result_mean  # check again when exercise 1 saves your module
    result_use = L.check("use-stats", box_use.value)
    result_use
    return (result_use,)


@app.cell(hide_code=True)
def hints_use():
    L.hints(
        "use-stats",
        [
            "Stap 4 van het recept: een functie uit een module gebruik je met de naam van de module en een punt.",
            "Roep de functie `mean` uit de module `stats` aan op `temps`, en laat met `print` zien wat hij teruggeeft.",
            "`print(stats.___(temps))`",
        ],
        why="`stats.mean(temps)` roept de functie aan die je in oefening 1 schreef, uit het bestand dat de check bewaarde. `print` laat de uitkomst zien: `12.0`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Code uit modules gebruiken](/course/open/py-imports?lang=nl) haalde je modules binnen die anderen schreven; nu haal je je eigen modules binnen, uit je projectmap. Vanaf nu bewaren sommige checks je code daar, en halen latere lessen hem binnen: aan het eind van de route staat in `work/projects/` je eigen language model. In de volgende les maak je je eigen soort waarde, met een class.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import_line, result_mean, result_use):
    L.wrap_up(result_mean, choice_import_line, result_use)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
