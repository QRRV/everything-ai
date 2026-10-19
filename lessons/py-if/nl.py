# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-if. Route: course/route/lessen/stam-1-basis.yaml. Checks en het uitgewerkte voorbeeld: checks.py.
# termen:allow variables, program

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-if", "nl")
    C = L.checks   # het uitgewerkte voorbeeld, en wat het laat zien bij twee bedragen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Variabelen en getallen](/course/open/py-variables?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_assign():
    quick_assign = L.choice(
        "quick-assign",
        question="Wat doet de regel `price = 5` in Python?",
        options={
            "give": "Hij geeft de naam `price` de waarde 5.",
            "ask": "Hij controleert of `price` 5 is.",
            "show": "Hij laat 5 onder de cel zien.",
        },
        correct="give",
        why={
            "give": "Eén `=` geeft de naam links de waarde rechts.",
            "ask": "In Python betekent `=` niet \"is gelijk aan\". Kijk nog eens naar *Let op* in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "show": "Alleen `print` laat iets onder de cel zien. Kijk nog eens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
        },
    )
    quick_assign
    return (quick_assign,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="Wat laat dit programma zien?\n\n```python\na = 3\nb = a * 2\nprint(b + 1)\n```",
        options={
            "six": "`6`",
            "word": "`b + 1`",
            "seven": "`7`",
        },
        correct="seven",
        why={
            "six": "`b` is 6, maar `print` laat `b + 1` zien. Kijk nog eens naar het recept in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "word": "Zonder aanhalingstekens rekent Python uit wat tussen de haakjes staat. Kijk nog eens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "seven": "`b` krijgt 3 keer 2, dus 6, en `print` laat 6 plus 1 zien.",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe deden je programma's elke keer hetzelfde. Maar een bruikbaar programma maakt keuzes: een spamfilter zet een bericht alleen in de spammap als het op spam lijkt, en een spel zegt alleen "Gewonnen!" als je score hoog genoeg is. Ook een programma dat van voorbeelden leert, maakt later in de course zulke keuzes. In deze les laat je je programma kiezen, met `if` en `else`.

    **Een voorbeeld uit het dagelijks leven.** "Als het regent, neem je een paraplu mee; anders je zonnebril." Je kijkt naar buiten, en het antwoord op één vraag, *regent het?*, bepaalt wat je doet. Een programma doet hetzelfde: het stelt een vraag die waar of niet waar is, en voert bij elk antwoord andere regels uit.
    """)
    return


@app.cell(hide_code=True)
def predict_compare():
    guess_compare = L.guess(
        "compare",
        question="Wat denk je dat `print(3 > 5)` laat zien?",
        options={
            "false": "`False`",
            "three": "`3`",
            "error": "Een foutmelding",
        },
        reveal="**`False`.** `3 > 5` vraagt: is 3 groter dan 5? Het antwoord is nee, en dat schrijft Python als `False`. Een vergelijking geeft altijd `True` of `False`.",
    )
    guess_compare
    return (guess_compare,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een vergelijking is een vraag met twee mogelijke antwoorden: `True` of `False`. Na `if` bepaalt dat antwoord welke regels draaien.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **bool** | een waarde die `True` of `False` is, met een hoofdletter | `True` |
    | **vergelijking** | vergelijkt twee waarden en geeft een bool | `3 < 5` geeft `True` |
    | **voorwaarde** | de vergelijking na `if`; het antwoord bepaalt welke regels draaien | `spent >= 10` |
    | **inspringen** | een regel met spaties laten beginnen; de ingesprongen regels horen bij de `if` of `else` erboven | twee of vier spaties, dan `print("Hoi")` |
    | **if** | voert de ingesprongen regels eronder alleen uit als de voorwaarde erna `True` is; de regel eindigt op een dubbele punt | `if spent >= 10:` |
    | **else** | voert zijn ingesprongen regels uit als de voorwaarde `False` was | `else:` |

    De tekens om mee te vergelijken:

    | Teken | Je zegt | Voorbeeld | Geeft |
    |---|---|---|---|
    | `<` | kleiner dan | `3 < 5` | `True` |
    | `>` | groter dan | `3 > 5` | `False` |
    | `<=` | kleiner dan of gelijk aan | `5 <= 5` | `True` |
    | `>=` | groter dan of gelijk aan | `4 >= 5` | `False` |
    | `==` | gelijk aan | `5 == 5` | `True` |
    | `!=` | niet gelijk aan | `5 != 5` | `False` |

    Met hoeveel spaties je inspringt, maakt niet uit, zolang alle regels onder dezelfde `if` er evenveel krijgen. Python-programma's gebruiken er meestal vier; het vak in een les gebruikt er twee.
    """)
    return


@app.cell(hide_code=True)
def picture_if():
    mo.vstack([
        draw.if_else("if spent >= 10:", ['print("Gratis koekje!")', "cookies = 1"], ['print("Deze keer geen koekje.")', "cookies = 0"],
                     ["print(cookies)"], yes_label="True", no_label="False", title="Twee paden, waarvan er één draait"),
        mo.md(r"""
        Is de voorwaarde `True`, dan voert Python de regels links uit; is hij `False`, dan de regels rechts. Er draait maar één van de twee paden, en daarna gaat het programma verder met de eerstvolgende regel die niet inspringt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Eén `=` geeft een naam een waarde; twee, `==`, vragen of twee waarden gelijk zijn. En vergeet de dubbele punt niet aan het eind van de regels met `if` en `else`. Druk na een dubbele punt op Enter: het vak laat de volgende regel vanzelf inspringen. Terug naar links, voor `else:`, ga je met Backspace."),
        mo.md(r"""
        ### Recept

        Zo laat je een programma kiezen:

        1. Bedenk de voorwaarde: een vergelijking die in het ene geval `True` is en in het andere `False`.
        2. Schrijf `if`, de voorwaarde en een dubbele punt.
        3. Schrijf eronder, ingesprongen, de regels die alleen moeten draaien als de voorwaarde `True` is.
        4. Schrijf `else:` helemaal links, recht onder `if`, en eronder, ingesprongen, de regels voor als de voorwaarde `False` is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Een bakker geeft je een gratis koekje als je voor € 10 of meer koopt. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort.
        """),
        mo.md(f"```python\n{C.EXAMPLE['nl']}\n```"),
        mo.md(r"""
        Dit programma laat zien:
        """),
        mo.md(f"```\n{C.EXAMPLE_SHOWS['nl']}```"),
        mo.md(f"""
        `12.5 >= 10` is `True`, dus de twee regels onder `if` draaien, en de regels onder `else` worden overgeslagen. Met `spent = {C.OTHER_SPENT}` is de voorwaarde `False`, en dan laat het programma dit zien:
        """),
        mo.md(f"```\n{C.EXAMPLE_OTHER['nl']}```"),
        mo.md(r"""
        De laatste regel, `print(cookies)`, springt niet in, dus die draait in beide gevallen. Zelf proberen? Kopieer de code naar het kladblok en verander het bedrag.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_below():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Onder nul?

        Vervang de `...` door een `if` en een `else`: als `temperature` kleiner is dan 0, laat het programma `Below zero` zien, en anders `Zero or above` (de tekst blijft Engels, net als de namen in de code). De check probeert ook andere temperaturen, zoals 0 en 5.
        """),
        L.task("Laat `Below zero` of `Zero or above` zien, afhankelijk van `temperature`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_below():
    box_below = L.code_box("below-zero", lines=6)
    box_below
    return (box_below,)


@app.cell(hide_code=True)
def check_below(box_below):
    result_below = L.check("below-zero", box_below.value)
    result_below
    return (result_below,)


@app.cell(hide_code=True)
def hints_below():
    L.hints(
        "below-zero",
        [
            "Stap 1 van het recept: welke vergelijking is `True` als de temperatuur onder nul is? Kijk in de tabel met tekens.",
            "De voorwaarde is `temperature < 0`. Daarna komt een dubbele punt, en eronder, ingesprongen, de regel met `Below zero`.",
            "`if temperature ___ 0:`, met `print(\"Below zero\")` ingesprongen eronder, dan `else:` helemaal links, met `print(\"Zero or above\")` ingesprongen eronder.",
        ],
        why="`temperature < 0` is `True` bij -3 en -0,5, en `False` bij 0 en 5. Onder `if` komt de regel met `Below zero`, onder `else` de andere; de dubbele punt en de spaties ervoor vertellen Python welke regels bij elkaar horen.",
    )
    return


@app.cell(hide_code=True)
def text_boundary():
    mo.md(r"""
    ### Oefening 2. Precies op de grens
    """)
    return


@app.cell(hide_code=True)
def choice_boundary_cell():
    choice_boundary = L.choice(
        "boundary",
        question="Wat laat `print(7 > 7)` zien?",
        options={
            "true": "`True`",
            "false": "`False`",
            "seven": "`7`",
        },
        correct="false",
        why={
            "true": "`>` vraagt of de waarde links groter is, en 7 is niet groter dan zichzelf. Kijk nog eens in de tabel met tekens: bij welk teken hoort ook \"gelijk aan\"?",
            "false": "7 is niet groter dan 7, dus het antwoord is `False`. `7 >= 7` zou `True` zijn.",
            "seven": "Een vergelijking geeft `True` of `False`, nooit een getal. Kijk nog eens naar het woord *vergelijking* in de tabel.",
        },
    )
    choice_boundary
    return (choice_boundary,)


@app.cell(hide_code=True)
def text_exercise_ticket():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een kaartje

        Een museumkaartje kost € 5 voor kinderen onder de 12, en € 9 voor alle anderen. Vervang de `...` door een `if` en een `else` die `price` de goede waarde geven; de laatste regel laat hem zien. De check probeert ook andere leeftijden, zoals 8 en 12.
        """),
        L.task("Geef `price` de waarde 5 of 9, afhankelijk van `age`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ticket():
    box_ticket = L.code_box("ticket", lines=8)
    box_ticket
    return (box_ticket,)


@app.cell(hide_code=True)
def check_ticket(box_ticket):
    result_ticket = L.check("ticket", box_ticket.value)
    result_ticket
    return (result_ticket,)


@app.cell(hide_code=True)
def hints_ticket():
    L.hints(
        "ticket",
        [
            "Stap 1 van het recept: welke vergelijking is `True` voor kinderen onder de 12? Valt 12 zelf onder de 12?",
            "In het uitgewerkte voorbeeld gaven de regels onder `if` en `else` `cookies` een waarde. Hier geven ze `price` een waarde.",
            "`if age ___ 12:`, met `price = 5` ingesprongen eronder, dan `else:`, met `price = 9` ingesprongen eronder.",
        ],
        why="`age < 12` is `True` bij 8 en 11, en `False` bij 12 en 15. Onder `if` krijgt `price` 5, onder `else` 9. `print(price)` springt niet in, dus die draait in beide gevallen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Variabelen en getallen](/course/open/py-variables?lang=nl) draaide elke regel van een programma; nu bepaalt een voorwaarde welke regels draaien. Vergelijkingen komen overal in de course terug: een programma dat van voorbeelden leert, kiest later tussen twee antwoorden door een getal met een grens te vergelijken. In de volgende les bewaar je veel waarden samen, in een lijst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_boundary, result_below, result_ticket):
    L.wrap_up(result_below, choice_boundary, result_ticket)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
