# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-names-in-cells. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow variables, function

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-names-in-cells", "nl")


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
def quick_check_area():
    quick_area = L.choice(
        "quick-area",
        question="Wat laat dit programma zien?\n\n```python\nwidth = 4\nheight = 3\narea = width * height\nprint(area)\n```",
        options={
            "seven": "`7`",
            "twelve": "`12`",
            "word": "`area`",
        },
        correct="twelve",
        why={
            "seven": "`*` is keer, niet plus. Kijk nog eens naar de tabel met tekens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "twelve": "`area` krijgt de waarde van 4 keer 3, en `print(area)` laat die waarde zien.",
            "word": "Zonder aanhalingstekens laat `print(area)` de waarde van de naam zien, niet het woord. Kijk nog eens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
        },
    )
    quick_area
    return (quick_area,)


@app.cell(hide_code=True)
def quick_check_power():
    quick_power = L.choice(
        "quick-power",
        question="Wat laat `print(2 ** 3)` zien?",
        options={
            "eight": "`8`",
            "six": "`6`",
            "nine": "`9`",
        },
        correct="eight",
        why={
            "eight": "`**` is tot de macht: 2 keer 2 keer 2 is 8.",
            "six": "`**` is geen keer. Kijk nog eens naar de tabel met tekens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "nine": "Dat is 3 tot de macht 2, precies andersom. Kijk nog eens naar de tabel met tekens in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
        },
    )
    quick_power
    return (quick_power,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe stond elk programma in een eigen vak. Een notebook heeft veel cellen, en die werken samen: een naam die je in de ene cel maakt, kun je in een andere cel gebruiken. Dat is handig. De getallen staan in de ene cel en de berekening in de volgende, en verander je een getal, dan rekent de volgende cel vanzelf opnieuw. Maar daarvoor geldt één spelregel, en daar gaat deze les over.

    **Een voorbeeld uit het dagelijks leven.** Een gedeelde keuken, met twee potten op de plank waar allebei *suiker* op staat, en in elke pot andere suiker. Een recept zegt "een lepel uit de suikerpot". Welke pot? Dat kan niemand zeggen. Een notebook heeft hetzelfde probleem als twee cellen dezelfde naam maken, en het gaat niet gokken.
    """)
    return


@app.cell(hide_code=True)
def predict_two():
    guess_two = L.guess(
        "two-cells",
        question="In de ene cel staat `total = 3`, en in een andere cel `print(total)`. In een derde cel typ je `total = 5`, en die voer je uit. Wat denk je dat er gebeurt?",
        options={
            "five": "`print(total)` laat nu `5` zien.",
            "both": "`total` is 3 in de ene cel en 5 in de andere.",
            "stops": "Je nieuwe cel draait niet, met een melding dat `total` ook in een andere cel wordt gemaakt.",
        },
        reveal="**Je nieuwe cel draait niet.** Het notebook gokt niet welke `total` je bedoelt. Onder je cel staat dat `total` ook in een andere cel wordt gemaakt, en `print(total)` laat gewoon 3 blijven zien. Dat is de regel van deze les: één naam, één cel.",
    )
    guess_two
    return (guess_two,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een cel *maakt* een naam als hij die een waarde geeft met `=`. Andere cellen kunnen die naam dan gebruiken. De notebooks van deze course draaien op *marimo*, een programma dat bijhoudt welke cel welke naam maakt. Zo weet marimo wat er moet gebeuren als je een cel verandert: elke cel die een naam uit die cel gebruikt, draait opnieuw. Waar een cel op de pagina staat, maakt niet uit.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **marimo** | het programma waarop de notebooks van deze course draaien | |
    | **één naam, één cel** | de regel van een notebook: elke naam wordt in maar één cel gemaakt; andere cellen mogen hem gebruiken, maar niet opnieuw maken | `price = 2.5` in de ene cel, `total = price * 3` in een andere |
    | **`_`-naam** | een naam die met `_` begint; hij bestaat alleen in zijn eigen cel, dus andere cellen zien hem niet, en een andere cel kan een eigen `_`-naam hebben die precies zo heet | `_months = 12` |

    Maakt een cel een naam die een andere cel al maakt, dan voert marimo hem niet uit. Onder de cel staat *This cell redefines variables from other cells*, met de naam en de andere cel die hem ook maakt. Engels voor: deze cel maakt variabelen die andere cellen ook al maken. Variabelen is het woord uit de vorige les voor namen. De volgende keer dat het notebook opent, draait geen van beide cellen, en staat de melding bij allebei. Misschien zie je ook een knop *Fix: Wrap in a function*; klik er niet op: hier los je het op met een andere naam. En een cel die een `_`-naam uit een andere cel gebruikt, stopt met een foutmelding: Python kent die naam daar niet.
    """)
    return


@app.cell(hide_code=True)
def picture_cells():
    mo.vstack([
        draw.notebook_cells([
            ("Eén naam, één cel", ["price = 2.5", "total = price * 3", "print(total)"], ["price", "total"],
             "✓ alle drie de cellen draaien"),
            ("Dezelfde naam in twee cellen", ["total = 2.5 * 3", "total = 4 * 2"], [None],
             "✗ marimo stopt: total wordt twee keer gemaakt"),
            ("Namen die met _ beginnen", ["_total = 2.5 * 3\nprint(_total)", "_total = 4 * 2\nprint(_total)"], [None],
             "✓ elke cel heeft een eigen _total"),
        ], title="Namen tussen cellen"),
        mo.md(r"""
        Links komen `price` en `total` elk uit één cel, en een pijl laat zien welke cel ze gebruikt. In het midden maken twee cellen `total`, dus marimo kan niet weten welke telt; rechts blijft elke `_total` in zijn eigen cel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Wil je een naam een nieuwe waarde geven, zoals in de vorige les? Doe dat in de cel die hem maakt. Een tweede cel met `price = price * 2` maakt `price` opnieuw, en dan voert marimo die cel niet uit."),
        mo.md(r"""
        ### Recept

        Zo gebruik je namen in meer cellen:

        1. Maak elke naam in maar één cel. Andere cellen mogen hem gebruiken; ze maken hem niet opnieuw.
        2. Heeft alleen deze cel een naam nodig, voor een tussenstap? Laat hem met `_` beginnen.
        3. Zegt marimo *This cell redefines variables from other cells*? Zoek de andere cel die dezelfde naam maakt. Bedoelde je de naam die die cel maakt? Haal dan de regel in jouw cel weg. Anders kies je in jouw cel een andere naam, of zet je er een `_` voor als alleen jouw cel hem nodig heeft.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een kamer kost € 750 per maand. Wat kost hij per jaar? Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort.
    """)
    return


@app.cell
def example_rent():
    # 1. deze cel maakt de naam monthly_rent; andere cellen mogen hem gebruiken
    monthly_rent = 750
    return (monthly_rent,)


@app.cell
def example_year(monthly_rent):
    # 2. alleen deze cel heeft deze namen nodig, dus ze beginnen met _
    _months = 12
    # monthly_rent komt uit de cel hierboven; deze cel maakt hem niet opnieuw
    _year_cost = monthly_rent * _months
    print(_year_cost)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        De tweede cel laat 9000 zien. `monthly_rent` komt uit de eerste cel, en `_months` en `_year_cost` blijven in de tweede, dus een andere cel zou een eigen `_months` kunnen hebben.

        3. Stel dat de tweede cel ook de regel `monthly_rent = 800` had. Dan maken twee cellen `monthly_rent`, en krijg je de melding uit stap 3. De oplossing: laat die regel weg, of geef de nieuwe huur een andere naam, zoals `new_rent`.
        """),
        L.task("Verander `750` in de eerste cel in `800`, en voer hem uit met **Shift + Enter**. Kijk naar de tweede cel: die draait vanzelf opnieuw, omdat hij `monthly_rent` gebruikt."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_trip():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Met de trein

        Vier vrienden gaan met de trein naar het strand en weer terug. Een enkeltje kost € 12,50; in Python is dat `12.5`. De cel hieronder maakt `ticket_price`. Maak in het vak daaronder `trip_cost`: wat de vier vrienden samen betalen, heen en terug, en laat het zien. Gebruik `ticket_price` in je berekening; maak hem niet opnieuw. De check voert je vak uit met de `ticket_price` uit de cel erboven, net zoals een cel dat zou doen.

        Zegt de check *Goed*? Verander dan `12.5` in die cel in `15` en voer hem uit: je vak volgt de nieuwe prijs.
        """),
        L.task("Maak `trip_cost` in het vak, met `ticket_price` uit de cel erboven."),
    ])
    return


@app.cell
def data_trip():
    ticket_price = 12.5
    return (ticket_price,)


@app.cell(hide_code=True)
def exercise_trip():
    box_trip = L.code_box("trip_cost", lines=4)
    box_trip
    return (box_trip,)


@app.cell(hide_code=True)
def check_trip(box_trip, ticket_price):
    try:
        _inputs = {"ticket_price": ticket_price}   # die mag de leerling veranderen, dus de check rekent met de huidige prijs
    except NameError:                              # de cel met ticket_price is weg: check met de prijs van de les zelf
        _inputs = None
    result_trip = L.check("trip_cost", box_trip.value, inputs=_inputs)
    result_trip
    return (result_trip,)


@app.cell(hide_code=True)
def hints_trip():
    L.hints(
        "trip_cost",
        [
            "Stap 1 van het recept: `ticket_price` wordt al gemaakt in de cel hierboven. Gebruik hem in het vak; maak hem niet opnieuw.",
            "Elke vriend heeft twee kaartjes nodig: een voor de heenweg en een voor de terugweg. En het zijn vier vrienden.",
            "In het uitgewerkte voorbeeld had de tweede cel `_year_cost = monthly_rent * _months`. Hier: `trip_cost = ticket_price * ___ * ___`, met `print(trip_cost)` eronder.",
        ],
        why="`trip_cost = ticket_price * 4 * 2` geeft 100, en bij een andere prijs rekent hij vanzelf mee. Met namen voor de tussenstappen kan het net zo goed: `_friends = 4`, `_tickets = 2`, en dan `trip_cost = ticket_price * _friends * _tickets`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Welke cellen passen samen in een notebook?

        Zes paren cellen; elk paar staat in een eigen notebook. Wat gebeurt er bij elk paar? Kijk naar het beeld en naar de nieuwe woorden.
        """),
        L.task("Zet elk paar cellen in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    sort_pairs = L.sort("cell-pairs")
    sort_pairs
    return (sort_pairs,)


@app.cell(hide_code=True)
def check_pairs(sort_pairs):
    result_pairs = L.check("cell-pairs", sort_pairs)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "cell-pairs",
        [
            "Kijk bij elk paar: welke namen maakt elke cel met `=`, en welke namen gebruikt hij alleen?",
            "Een naam zonder `_` mag maar in één cel gemaakt worden. Een naam met `_` blijft in de cel die hem maakt.",
            "`price = price * 2` in cel 2 maakt `price` opnieuw, net als cel 1.",
        ],
        why="**Beide cellen draaien:** `price` en `cost` (cel 2 gebruikt `price` en maakt `cost`), en de twee `_total` (elke cel heeft zijn eigen `_total`). **marimo stopt:** de twee `total`, en de twee `price` (cel 2 maakt `price` opnieuw). **Cel 2 kent de naam niet:** `_n` en `_rate` blijven in cel 1.",
    )
    return


@app.cell(hide_code=True)
def text_fix():
    mo.md(r"""
    ### Oefening 3. De melding
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="In een notebook staat in jouw cel `total = 10`. Onder je cel zegt marimo: *This cell redefines variables from other cells.* Wat doe je?",
        options={
            "move": "Je zet je cel bovenaan in het notebook.",
            "again": "Je voert je cel nog een keer uit.",
            "rename": "Je kiest in je cel een andere naam, zoals `my_total`, of `_total` als alleen jouw cel hem nodig heeft.",
        },
        correct="rename",
        why={
            "move": "Waar een cel op de pagina staat, maakt niet uit: er maken nog steeds twee cellen `total`. Kijk nog eens naar stap 3 van het recept.",
            "again": "Opnieuw uitvoeren verandert niets: er maken nog steeds twee cellen `total`. Kijk nog eens naar stap 3 van het recept.",
            "rename": "Dan komt elke naam weer uit één cel, en draaien beide cellen.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Variabelen en getallen](/course/open/py-variables?lang=nl) stond elk programma in één vak; nu gaan namen van cel naar cel. Vanaf nu gebruiken de uitgewerkte voorbeelden `_`-namen voor hun tussenstappen, zodat je hun vorm kunt overnemen zonder dat namen botsen. In de volgende les doe je meer met tekst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, result_pairs, result_trip):
    L.wrap_up(result_trip, result_pairs, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
