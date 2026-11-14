# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-json. Route: course/route/lessen/bouwen.yaml. Checks and the files of this lesson: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-json", "nl")
    C = L.checks   # de paden van de bestanden van de les, en de dict in het plaatje


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Dictionaries](/course/open/py-dicts?lang=nl) en [Een tekstbestand lezen](/course/open/py-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="Wat laat dit programma zien?\n\n```python\nages = {\"Ada\": 36, \"Alan\": 41}\nprint(ages[\"Alan\"])\n```",
        options={
            "key": "`Alan`",
            "value": "`41`",
            "pair": "`\"Alan\": 41`",
        },
        correct="value",
        why={
            "key": "`ages[\"Alan\"]` zoekt de key `\"Alan\"` op en geeft de waarde ervan. Zie het idee in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "value": "De key `\"Alan\"` geeft zijn waarde, 41.",
            "pair": "Vierkante haken met een key geven alleen de waarde. Zie het idee in [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_read():
    quick_read = L.choice(
        "quick-read",
        question="`notes.txt` heeft drie regels. Wat is `text` hierna?\n\n```python\nwith open(\"notes.txt\") as f:\n    text = f.read()\n```",
        options={
            "string": "Eén string met het hele bestand, regeleindes erbij",
            "lines": "Een lijst met de drie regels",
            "first": "Alleen de eerste regel",
        },
        correct="string",
        why={
            "lines": "`f.read()` deelt het bestand niet op in regels. Zie het recept in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "first": "`f.read()` leest het bestand tot het eind. Zie het recept in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "string": "`f.read()` geeft het hele bestand als één string, met een `\\n` tussen de regels.",
        },
    )
    quick_read
    return (quick_read,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Programma's geven elkaar data door als tekst: een bestand met instellingen, een bewaarde lijst met scores, het antwoord dat een webdienst terugstuurt. Een dict of een lijst bestaat alleen in je draaiende programma; om hem te bewaren of te versturen moet hij tekst worden, en het programma dat hem krijgt, moet er weer een dict of lijst van maken. JSON is de schrijfwijze die bijna elk programma begrijpt. Later in de course antwoordt een language model je programma in JSON, en wordt elke run van een experiment bewaard als een regel JSON.

    **Een voorbeeld uit het dagelijks leven.** Een formulier met vaste velden: *naam*, *geboortedatum*, *allergieën*. Iedereen die het krijgt, kan het lezen, omdat de velden altijd dezelfde namen hebben. JSON is zo'n formulier voor programma's.
    """)
    return


@app.cell(hide_code=True)
def predict_dumps():
    guess_dumps = L.guess(
        "dumps",
        question="Wat denk je dat dit programma laat zien?\n\n```python\nimport json\nprint(json.dumps({\"name\": \"Ada\", \"done\": True}))\n```",
        options={
            "python": "`{'name': 'Ada', 'done': True}`",
            "json": "`{\"name\": \"Ada\", \"done\": true}`",
            "error": "Een fout: `True` is geen tekst",
        },
        reveal="**`{\"name\": \"Ada\", \"done\": true}`.** `json.dumps` schrijft de dict als JSON-tekst: met dubbele aanhalingstekens, en `true` met kleine letters, zoals JSON dat schrijft.",
    )
    guess_dumps
    return (guess_dumps,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    JSON schrijft data als tekst, met dezelfde bouwstenen als Python: dicts worden `{...}` met paren van een key en een waarde, lijsten worden `[...]`, en daarin staan strings, getallen, true of false, en null voor *niets*. De module `json` vertaalt beide kanten op.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **JSON** | een manier om dicts, lijsten, strings, getallen, true, false en null als tekst te schrijven, die bijna elke programmeertaal kan lezen en schrijven | `{"name": "Ada", "age": 36}` |
    | **json** | de module die vertaalt tussen JSON-tekst en Python-waarden | `import json` |
    | **dumps** | geeft de JSON-tekst van een waarde, als string: *dump to a string* | `json.dumps(value)` |
    | **loads** | zet JSON-tekst uit een string om in Python-waarden: *load from a string* | `json.loads(text)` |
    | **load** | leest JSON uit een open bestand | `json.load(f)` |

    JSON heeft eigen woorden voor drie waarden: `true`, `false` en `null` zijn in Python `True`, `False` en `None`. Een string heeft in JSON altijd dubbele aanhalingstekens, `"Ada"`, nooit enkele. Een key is altijd een string. Een bestand met JSON heeft meestal een naam die eindigt op `.json`.
    """)
    return


@app.cell(hide_code=True)
def picture_json():
    _py, _js = C.picture_lines()
    mo.vstack([
        draw.marked_lines([("Een dict, zoals Python hem laat zien", [(_line, _m) for _line, _m in zip(_py, ["3", "1", "2", ""])]),
                           ("Dezelfde dict als JSON-tekst, uit json.dumps", [(_line, _m) for _line, _m in zip(_js, ["3", "1", "2", ""])])],
                          {"1": "True wordt true", "2": "None wordt null", "3": "enkele aanhalingstekens worden dubbele"},
                          title="Eén dict, in Python en als JSON"),
        mo.md(r"""
        De twee lijken bijna hetzelfde, en dat is met opzet: JSON is gemaakt om op de dicts en lijsten van programmeertalen te lijken. Maar de onderste is tekst, een string die je kunt bewaren of versturen, en de bovenste is een dict in je draaiende programma. `json.loads` maakt van de onderste weer de bovenste.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`json.load` en `json.loads` verschillen maar in één letter. Zonder de s leest hij uit een open bestand `f`; met de s uit een string. Geef je `json.loads` een pad, dan probeert hij het pad zelf als JSON te lezen, en stopt hij met een `JSONDecodeError`. En op tekst die alleen op JSON lijkt, met enkele aanhalingstekens of `True`, stopt `json.loads` ook met die fout: de fout zegt op welke regel en kolom de tekst misging."),
        mo.md(r"""
        ### Recept

        Zo maak je van data JSON-tekst, en weer terug:

        1. Schrijf bovenaan `import json`.
        2. Uit een bestand: open het met `with open(path) as f:`, en lees het met `json.load(f)`.
        3. Pak een waarde eruit met vierkante haken: `[key]` voor een dict, `[index]` voor een lijst, na elkaar.
        4. Van Python naar tekst: `json.dumps(value)`.
        5. Van tekst naar Python: `json.loads(text)` geeft dicts, lijsten, strings, getallen, `True`, `False` en `None`.
        6. Schrijf JSON nooit zelf met `+`: zet je waarden in een dict, en laat `json.dumps` hem schrijven, met alle aanhalingstekens.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het bestand `pancakes.json` bevat een recept: de naam, voor hoeveel mensen het is, of het vegetarisch is, een lijst met ingrediënten, en opmerkingen. Elk ingrediënt is een eigen dict. Dit is de tekst in het bestand:

    ```json
    {
      "name": "pancakes",
      "servings": 4,
      "vegetarian": true,
      "ingredients": [
        {"item": "flour", "grams": 250},
        {"item": "milk", "grams": 450},
        {"item": "eggs", "grams": 110},
        {"item": "butter", "grams": 25}
      ],
      "notes": null
    }
    ```

    Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_recipe():
    # 1. de module
    import json

    # 2. lees het bestand: json.load maakt van de JSON Python-waarden
    with open("lessons/py-json/pancakes.json") as _f:
        _recipe = json.load(_f)
    print(_recipe["name"], _recipe["servings"], _recipe["vegetarian"], _recipe["notes"])
    # 3. een lijst met dicts in de dict: eerst [key], dan [index], dan [key]
    print(_recipe["ingredients"][0]["item"], _recipe["ingredients"][0]["grams"])
    # 4. terug naar tekst: json.dumps schrijft true en null, met dubbele aanhalingstekens
    _text = json.dumps({"servings": 8, "vegetarian": True, "notes": None})
    print(_text)
    # 5. en van tekst weer naar Python
    print(json.loads(_text)["servings"] * 2)
    return (json,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen vier regels. `pancakes 4 True None`: in het bestand staan `true` en `null`, en `json.load` maakte er `True` en `None` van. `flour 250`: het eerste ingrediënt, een dict in de lijst in de dict. `{"servings": 8, "vegetarian": true, "notes": null}`: weer tekst, op de manier van JSON. En `16`: `json.loads` maakte van die tekst weer een dict, waarin `"servings"` weer het getal 8 is, niet de tekst `8`. Een `import` maakt een naam, net als `=`: deze cel maakt `json`, dus `import json` in een andere cel breekt *één naam, één cel*. De oefenvakken draaien los, dus elk vak heeft zijn eigen regel met `import`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Alle grammen

        Schrijf een functie `total_grams(path)` die het recept in het JSON-bestand op `path` leest en de grammen van al zijn ingrediënten opgeteld teruggeeft. De check probeert hem op `pancakes.json`, en op twee andere recepten in dezelfde map.
        """),
        L.task("Zorg dat `total_grams(\"lessons/py-json/pancakes.json\")` `835` geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total-grams", lines=9)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total-grams", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total-grams",
        [
            "Stap 2 van het recept leest het bestand; het uitgewerkte voorbeeld laat zien hoe je bij de ingrediënten komt.",
            "`recipe[\"ingredients\"]` is een lijst met dicts. Ga hem langs met een `for`-loop, en tel uit elke dict iets op.",
            "`with open(path) as f:` en `recipe = json.load(f)`, dan `total = 0`, een loop over `recipe[\"ingredients\"]` die ___ optelt, en `return total`.",
        ],
        why="`json.load(f)` maakt van het bestand een dict. Zijn `\"ingredients\"` is een lijst met een dict voor elk ingrediënt, dus de loop telt bij elk ingrediënt `ingredient[\"grams\"]` op: 250 + 450 + 110 + 25 = 835.",
    )
    return


@app.cell(hide_code=True)
def text_true_null():
    mo.md(r"""
    ### Oefening 2. De eigen woorden van JSON
    """)
    return


@app.cell(hide_code=True)
def choice_true_null_cell():
    choice_true_null = L.choice(
        "true-null",
        question="Wat geeft dit?\n\n```python\nimport json\njson.loads('{\"done\": true, \"left\": null}')\n```",
        options={
            "values": "`{'done': True, 'left': None}`",
            "strings": "`{'done': 'true', 'left': 'null'}`",
            "error": "Een fout: `true` en `null` zijn geen Python",
        },
        correct="values",
        why={
            "strings": "Zonder aanhalingstekens eromheen zijn `true` en `null` in JSON geen strings: het zijn de eigen woorden van JSON voor twee waarden. Kijk nog eens naar de tabel met nieuwe woorden.",
            "error": "De tekst is JSON, geen Python, en in JSON zijn `true` en `null` in orde. `json.loads` vertaalt ze. Kijk nog eens naar de tabel met nieuwe woorden.",
            "values": "`json.loads` vertaalt `true` van JSON in `True` van Python, en `null` in `None`.",
        },
    )
    choice_true_null
    return (choice_true_null,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een weerbericht als JSON

        Schrijf een functie `report(city, mm)` die JSON-tekst teruggeeft met drie keys: `"city"`, `"mm"` (de regen in millimeter) en `"rain"`, die true is als `mm` meer dan 0 is. Een ander programma leest je tekst met `json.loads`, dus hij moet geldige JSON zijn.
        """),
        L.task("Zorg dat `report(\"Utrecht\", 2.5)` JSON-tekst geeft met `\"city\": \"Utrecht\"`, `\"mm\": 2.5` en `\"rain\": true`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=7)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Stap 6 van het recept: plak de tekst niet zelf aan elkaar.",
            "Maak eerst een dict met de drie keys. Wat zet je onder `\"rain\"`, zodat `json.dumps` `true` of `false` schrijft?",
            "`return json.dumps({\"city\": city, \"mm\": mm, \"rain\": ___})`",
        ],
        why="`mm > 0` is een bool, `True` of `False`, en `json.dumps` schrijft die als `true` of `false`. Hij zet ook dubbele aanhalingstekens om de stad, wat er ook in de naam staat, dus de tekst is elke keer geldige JSON.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Dictionaries](/course/open/py-dicts?lang=nl) zocht je waarden op met hun key; JSON bewaart zulke dicts, en lijsten ervan, als tekst. In [Een tekstbestand lezen](/course/open/py-files?lang=nl) las je een bestand als string; `json.load` leest het en maakt er meteen Python-waarden van. En in [Code uit modules gebruiken](/course/open/py-imports?lang=nl) haalde je modules binnen; `json` is er nog een. Later in de course stuurt een language model zijn antwoorden in deze vorm naar je programma, en controleer je of zijn JSON geldig is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_true_null, result_report, result_total):
    L.wrap_up(result_total, choice_true_null, result_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
