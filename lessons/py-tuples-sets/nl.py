# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-tuples-sets. Route: course/route/lessen/stam-1-basis.yaml. Checks en de rijen in de figuur: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-tuples-sets", "nl")
    C = L.checks   # de getallen in de figuur, op vier manieren bewaard


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lijsten](/course/open/py-lists?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="Wat laat dit programma zien?\n\n```python\nletters = [\"a\", \"b\", \"c\", \"d\"]\nprint(letters[:2])\n```",
        options={
            "three": "`['a', 'b', 'c']`",
            "end": "`['c', 'd']`",
            "two": "`['a', 'b']`",
        },
        correct="two",
        why={
            "three": "Een slice stopt net voor zijn tweede getal, dus index 2 hoort er niet bij. Zie het blok *Let op* in [Lijsten](/course/open/py-lists?lang=nl).",
            "end": "Staat er niets voor de dubbele punt, dan begint een slice bij het begin. Zie het idee in [Lijsten](/course/open/py-lists?lang=nl).",
            "two": "`[:2]` begint bij het begin en stopt net voor index 2: de eerste twee items.",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_append():
    quick_append = L.choice(
        "quick-append",
        question="Wat laat dit programma zien?\n\n```python\nnums = [1, 2]\nnums.append(3)\nprint(len(nums))\n```",
        options={
            "three": "`3`",
            "two": "`2`",
            "list": "`[1, 2, 3]`",
        },
        correct="three",
        why={
            "three": "`append` zet 3 achteraan, dus de lijst heeft drie items, en `len` geeft dat aantal.",
            "two": "`append` verandert de lijst zelf, voordat `len` telt. Zie het recept in [Lijsten](/course/open/py-lists?lang=nl).",
            "list": "`len` laat de lijst niet zien: het geeft het aantal items. Zie het idee in [Lijsten](/course/open/py-lists?lang=nl).",
        },
    )
    quick_append
    return (quick_append,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een lijst bewaart allerlei waarden op volgorde, en je kunt hem veranderen. Soms heb je iets anders nodig: waarden die bij elkaar horen en moeten blijven zoals ze zijn, elke waarde maar één keer, of alle waarden op volgorde. Voor alle drie heeft Python een manier. Later in de course bewaar je de afmetingen van een tabel met getallen als tuple, de verschillende woorden van een tekst als set, en vind je met sorteren de beste antwoorden.

    **Een voorbeeld uit het dagelijks leven.** Een datum: dag, maand en jaar horen bij elkaar, en je verandert er later niet één van. Een gastenlijst: elke naam staat er één keer op, hoe vaak iemand ook zegt te komen. En een uitslagenlijst na een hardloopwedstrijd: de tijden op volgorde, de snelste eerst.
    """)
    return


@app.cell(hide_code=True)
def predict_set():
    guess_set = L.guess(
        "set",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ncolors = [\"red\", \"blue\", \"red\", \"red\"]\nprint(len(set(colors)))\n```",
        options={
            "four": "`4`",
            "two": "`2`",
            "one": "`1`",
        },
        reveal="**`2`.** In `set(colors)` staat elke waarde één keer: `\"red\"` en `\"blue\"`. `len` telt wat er in de set zit, dus 2.",
    )
    guess_set
    return (guess_set,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een tuple is als een lijst die niet kan veranderen, in een set staat elke waarde maar één keer, en `sorted` zet waarden op volgorde.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **tuple** | waarden op volgorde, tussen ronde haakjes; als hij eenmaal gemaakt is, kan hij niet meer veranderen | `point = (3, 4)` |
    | **set** | waarden die er elk één keer in staan, zonder vaste volgorde; `set(...)` maakt een set van een lijst | `set(["a", "b", "a"])` geeft een set met `"a"` en `"b"` |
    | **sorted** | geeft een nieuwe lijst met de waarden op volgorde, van klein naar groot | `sorted([3, 1, 2])` geeft `[1, 2, 3]` |

    Lezen gaat bij een tuple net als bij een lijst: `point[0]`, een slice en `len(point)` werken allemaal. Maar `point[0] = 5` stopt het programma met een fout, en een tuple heeft geen `append`. `x, y = point` geeft elke waarde van de tuple een eigen naam, op volgorde. Een set heeft geen plekken, dus ook geen index, en Python laat zijn waarden zien tussen accolades, in willekeurige volgorde. Met kleine gehele getallen staan ze vaak toevallig van klein naar groot; met woorden kan de volgorde elke keer anders zijn. `sorted` werkt op een lijst, een tuple en een set, en geeft altijd een lijst. Woorden worden gesorteerd van A tot Z, en elk woord met een hoofdletter komt vóór elk woord met een kleine letter: `"Zoe"` vóór `"ada"`.
    """)
    return


@app.cell(hide_code=True)
def picture_rows():
    mo.vstack([
        draw.value_rows(C.picture_rows({"list": "op volgorde, kan veranderen", "tuple": "op volgorde, kan niet veranderen",
                                        "set": "elke waarde één keer, geen volgorde", "sorted": "een nieuwe lijst, op volgorde"}),
                        title="Dezelfde getallen, op vier manieren bewaard"),
        mo.md(r"""
        Een lijst en een tuple houden de volgorde, en allebei de drieën; alleen de lijst kan veranderen. In de set staat elke waarde maar één keer, en hij heeft geen volgorde: daarom zitten zijn waarden in een zak, zonder plekken. `sorted` geeft een nieuwe lijst, van klein naar groot.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`sorted(times)` geeft een nieuwe lijst en laat `times` zoals hij was. Er is ook `times.sort()`: die verandert de lijst zelf en geeft niets terug, net als `append`. Schrijf dus nooit `times = times.sort()`: dan wordt `times` `None`, het woord van Python voor niets."),
        mo.md(r"""
        ### Recept

        Zo kies je de goede manier om waarden te bewaren:

        1. Waarden die bij elkaar horen en blijven zoals ze zijn: een tuple, `point = (3, 4)`. Pak een waarde met `point[0]`, of geef elke waarde een eigen naam: `x, y = point`.
        2. Elke waarde maar één keer: maak een set van de lijst, `set(words)`. `len(set(words))` geeft het aantal verschillende waarden.
        3. Alle waarden op volgorde: `sorted(...)` geeft een nieuwe lijst, van klein naar groot.
        4. De paar kleinste of grootste waarden: sorteer eerst, en pak dan een slice, zoals `sorted(scores)[:3]` of `sorted(scores)[-3:]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een beeldscherm, de bezoekers van een website, en de leeftijden in een groep. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_tuples_sets():
    # 1. een tuple: de breedte en hoogte van een scherm horen bij elkaar
    _screen = (1920, 1080)
    print(_screen)
    print(_screen[0])
    # of geef elke waarde een eigen naam
    _width, _height = _screen
    print(f"{_width} bij {_height}")
    # 2. een set: elke bezoeker één keer
    _visitors = ["ana", "ben", "ana", "cas", "ben"]
    print(len(set(_visitors)))
    # 3. sorted: een nieuwe lijst, op volgorde, hier van de set
    _names = sorted(set(_visitors))
    print(_names)
    _ages = [34, 19, 52, 27]
    print(sorted(_ages))
    print(_ages)
    # 4. sorteer, en pak dan een slice: de twee jongsten, en de twee oudsten
    _youngest = sorted(_ages)[:2]
    print(_youngest)
    _oldest = sorted(_ages)[-2:]
    print(_oldest)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen negen regels. Python laat een tuple zien tussen ronde haakjes: `(1920, 1080)`. Vijf bezoeken, maar slechts drie verschillende bezoekers: in de set staan `ana` en `ben` maar één keer. `sorted(set(_visitors))` geeft ze als lijst, van A tot Z. `sorted(_ages)` geeft een nieuwe lijst op volgorde, en de regel daarna laat zien dat `_ages` zelf niet veranderd is. De laatste twee regels pakken de eerste twee en de laatste twee van de gesorteerde lijst: de twee jongsten en de twee oudsten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unique():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elk woord één keer

        Maak `unique`: elk woord uit `words` één keer, als lijst van A tot Z. Vervang de `...`. De check probeert ook een andere lijst.
        """),
        L.task("Maak `unique`, zodat het programma `['fog', 'rain', 'sun']` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unique():
    box_unique = L.code_box("unique", lines=4)
    box_unique
    return (box_unique,)


@app.cell(hide_code=True)
def check_unique(box_unique):
    result_unique = L.check("unique", box_unique.value)
    result_unique
    return (result_unique,)


@app.cell(hide_code=True)
def hints_unique():
    L.hints(
        "unique",
        [
            "Twee stappen: eerst elk woord maar één keer, dan op volgorde. Welke stappen van het recept zijn dat?",
            "In een set staat elk woord maar één keer, maar hij heeft geen volgorde. Welke instructie uit het recept maakt er een lijst van, op volgorde?",
            "`unique = ___(set(words))`",
        ],
        why="In `set(words)` staan `rain`, `sun` en `fog` elk één keer. `sorted(...)` maakt van die set een lijst van A tot Z: `['fog', 'rain', 'sun']`.",
    )
    return


@app.cell(hide_code=True)
def text_tuple_change():
    mo.md(r"""
    ### Oefening 2. Een tuple veranderen
    """)
    return


@app.cell(hide_code=True)
def choice_tuple_change_cell():
    choice_tuple_change = L.choice(
        "tuple-change",
        question="Wat gebeurt er als dit programma draait?\n\n```python\npoint = (3, 4)\npoint[0] = 5\nprint(point)\n```",
        options={
            "five": "Het laat `(5, 4)` zien",
            "same": "Het laat `(3, 4)` zien",
            "error": "Python stopt met een fout op de tweede regel",
        },
        correct="error",
        why={
            "error": "Een tuple kan niet veranderen als hij eenmaal gemaakt is. Dus stopt Python bij `point[0] = 5`, en de `print` draait nooit.",
            "five": "Dat zou een lijst doen. Kijk nog eens naar het woord *tuple* in de tabel.",
            "same": "Python slaat nooit stilletjes een regel over. Kijk nog eens naar het woord *tuple* in de tabel: wat gebeurt er als je er een probeert te veranderen?",
        },
    )
    choice_tuple_change
    return (choice_tuple_change,)


@app.cell(hide_code=True)
def text_exercise_podium():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het podium

        `times` bevat de tijden van vijf lopers, in seconden. Hoe lager de tijd, hoe sneller. Maak `podium`: de drie snelste tijden, de snelste eerst. Vervang de `...`. De check probeert ook een andere wedstrijd.
        """),
        L.task("Maak `podium`, zodat het programma `[11.8, 11.9, 12.0]` laat zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_podium():
    box_podium = L.code_box("podium", lines=4)
    box_podium
    return (box_podium,)


@app.cell(hide_code=True)
def check_podium(box_podium):
    result_podium = L.check("podium", box_podium.value)
    result_podium
    return (result_podium,)


@app.cell(hide_code=True)
def hints_podium():
    L.hints(
        "podium",
        [
            "Stap 4 van het recept: eerst sorteren, dan een slice.",
            "`sorted(times)` zet de laagste tijd vooraan. Welke slice pakt de eerste drie?",
            "`podium = sorted(times)[___]`",
        ],
        why="`sorted(times)` geeft `[11.8, 11.9, 12.0, 12.4, 13.1]`, en `[:3]` pakt daarvan de eerste drie: de drie snelste.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lijsten](/course/open/py-lists?lang=nl) bewaarde je waarden op volgorde en veranderde je ze. Een tuple bewaart ze op volgorde en laat ze niet veranderen, in een set staat elke waarde maar één keer, en `sorted` zet ze op volgorde. Later in de course zijn de afmetingen van een tabel met getallen een tuple, zoals `(3, 4)` voor 3 rijen en 4 kolommen. In de volgende les maak je een nieuwe lijst van een oude, in één regel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_tuple_change, result_podium, result_unique):
    L.wrap_up(result_unique, choice_tuple_change, result_podium)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
