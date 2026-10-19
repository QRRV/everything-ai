# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-pandas. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-pandas", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Delen van een array kiezen](/course/open/np-indexing?lang=nl) en [Een tekstbestand lezen](/course/open/py-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="`a = np.array([3, 8, 1, 6])`. Wat geeft `a[a > 4]`?",
        options={
            "mask": "`array([False, True, False, True])`",
            "places": "`array([1, 3])`",
            "values": "`array([8, 6])`",
        },
        correct="values",
        why={
            "mask": "Dat is het mask zelf, `a > 4`. Wat gebeurt er als het tussen de vierkante haken van `a` komt? Zie het recept in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
            "places": "Dat zijn de plekken van de getallen boven 4. Geeft een mask tussen de vierkante haken van `a` plekken of getallen? Zie de tabel met nieuwe woorden in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
            "values": "Het mask is `True` voor 8 en 6, dus kiest `a[a > 4]` die twee getallen.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="In de map van de course staat een map `lessons`; daarin `data-pandas`; en daarin het bestand `weather.csv`. Welk pad leidt naar dat bestand?",
        options={
            "path": "`\"lessons/data-pandas/weather.csv\"`",
            "name": "`\"weather.csv\"`",
            "back": "`\"weather.csv/data-pandas/lessons\"`",
        },
        correct="path",
        why={
            "path": "Een pad begint bij de map van de course en noemt elke map, met `/` ertussen, tot aan het bestand.",
            "name": "Een pad begint bij de map van de course, en daar staat het bestand niet. Zie de tabel met nieuwe woorden in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "back": "Een pad gaat van buiten naar binnen: eerst de map waar je begint, het bestand als laatste. Zie het recept in [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Niet alle data past in één rooster van getallen van één type. In een tabel met huizen heeft elk huis een adres (tekst), een prijs en een aantal kamers, en elke kolom heeft een naam; een array heeft één type en geen namen. pandas, een library voor tabellen, bewaart zulke data als DataFrame: een naam en een type voor elke kolom, en een rij per voorbeeld. Zulke tabellen komen vaak als CSV-bestand, een tekstbestand met een tabel erin.

    **Een voorbeeld uit het dagelijks leven.** Een klassenlijst in een spreadsheet: een kolom met namen, een kolom met leeftijden en een kolom met cijfers, elk met een kopje bovenaan.
    """)
    return


@app.cell(hide_code=True)
def predict_filter():
    guess_filter = L.guess(
        "filter",
        question="`df` is een tabel met het weer van een week, met de kolommen `day`, `temp`, `rain` en `wind`. Wat denk je dat `df[df[\"rain\"] > 0]` geeft?",
        options={
            "rows": "De regendagen, met alle vier de kolommen",
            "column": "Alleen de kolom `rain`, van de regendagen",
            "mask": "`True` of `False` voor elke dag",
        },
        reveal="**De regendagen, met alle vier de kolommen.** `df[\"rain\"] > 0` is een mask, `True` voor elke dag met regen, net als een mask op een array. Tussen de vierkante haken van `df` kiest het die rijen, in hun geheel.",
    )
    guess_filter
    return (guess_filter,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    pandas is een library voor tabellen. De tabel van pandas heet DataFrame: kolommen met een naam, elk van één type, en een rij per voorbeeld. Je leest er een uit een CSV-bestand, pakt zijn kolommen bij hun naam, en houdt met een mask de rijen over die je wilt, net als bij een array.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **library** | een verzameling modules die iemand anders schreef, met de course meegeïnstalleerd; NumPy is er ook een | |
    | **pandas** | een library voor tabellen; je importeert hem altijd als `pd` | `import pandas as pd` |
    | **CSV-bestand** | een tekstbestand met een tabel: een rij per regel, de waarden gescheiden door komma's, en de kolomnamen op de eerste regel | `day,temp,rain,wind` |
    | **DataFrame** | de tabel van pandas: kolommen met een naam, elk van één type, en een rij per voorbeeld | `df = pd.read_csv(path)` |
    | **rijlabel** | het getal dat pandas links van elke rij zet; geselecteerde rijen behouden hun oude rijlabels; pandas noemt ze de index | `0`, `1`, `2` … |
    | **één kolom** | zijn naam tussen de vierkante haken; je krijgt een Series: de waarden, met hun rijlabels | `df["temp"]` |
    | **een paar kolommen** | een lijst namen tussen de vierkante haken, dus twee paar haken; je krijgt een nieuw DataFrame | `df[["day", "temp"]]` |
    | **rijen selecteren** | een mask op een kolom, tussen de vierkante haken van het DataFrame, houdt de rijen over waar het `True` is | `df[df["rain"] > 0]` |

    `df.shape` geeft het aantal rijen en kolommen, net als bij een array, en `df.head()` geeft de eerste vijf rijen. pandas zelf is gebouwd op NumPy.
    """)
    return


@app.cell(hide_code=True)
def picture_select():
    mo.vstack([
        draw.data_tables(C.picture(["één kolom", "geselecteerde rijen"]), title="Kiezen uit een DataFrame",
                         alt="Kiezen uit een DataFrame. Twee keer de weertabel: links is de kolom rain gepakt; rechts zijn de rijen van dinsdag, woensdag, donderdag en zondag, de dagen met regen, in hun geheel overgehouden."),
        mo.md(r"""
        Twee keer dezelfde tabel. Links pakt `df["rain"]` één kolom bij zijn naam. Rechts houdt `df[df["rain"] > 0]` de rijen van de vier dagen met regen over, met al hun kolommen. De kleine getallen links zijn de rijlabels: de overgehouden rijen behouden de hunne.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Voor meer dan één kolom staan de namen in een lijst: twee paar vierkante haken, `df[[\"day\", \"temp\"]]`. Met één paar, `df[\"day\", \"temp\"]`, zoekt pandas één kolom met die hele naam, en stopt met een fout."),
        mo.md(r"""
        ### Recept

        Zo werk je met een tabel:

        1. Importeer pandas en lees de tabel: `df = pd.read_csv(path)`.
        2. Bekijk hem: `df.shape` voor zijn grootte, en `df.head()` voor zijn eerste rijen.
        3. Pak kolommen bij hun naam: `df["temp"]`, of `df[["day", "temp"]]`.
        4. Houd met een mask op een kolom de rijen over die je wilt: `df[df["rain"] > 0]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het weer van een verzonnen week, in `lessons/data-pandas/weather.csv`: voor elke dag de temperatuur in graden Celsius, de regen in millimeters, en de wind in kilometers per uur. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_weather():
    # 1. importeer pandas en lees de tabel
    import pandas as pd

    _df = pd.read_csv("lessons/data-pandas/weather.csv")
    # 2. zijn grootte, en zijn eerste vijf rijen
    print(_df.shape)
    print(_df.head())
    # 3. één kolom, en twee kolommen
    print(_df["temp"])
    print(_df[["day", "rain"]])
    # 4. het mask van de regendagen, de rijen die het overhoudt, en de temperaturen van de droge dagen
    print(_df["rain"] > 0)
    print(_df[_df["rain"] > 0])
    print(_df[_df["rain"] == 0]["temp"])
    return (pd,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tabel heeft 7 rijen en 4 kolommen, `(7, 4)`, en `head` laat maandag tot en met vrijdag zien. De kolom `temp` komt eruit als Series: elke temperatuur met zijn rijlabel, en daaronder zijn naam en dtype. Twee kolommen komen eruit als een kleiner DataFrame. Het mask is `True` of `False` voor elke dag, en de rijen die het overhoudt, zijn rij 1, 2, 3 en 6: geselecteerde rijen behouden hun oude rijlabels. De laatste regel houdt eerst de droge dagen over, en pakt dan hun temperaturen: `12.5`, `13.0` en `16.5`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_columns():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee kolommen
    """)
    return


@app.cell(hide_code=True)
def choice_columns_cell():
    choice_columns = L.choice(
        "two-columns",
        question="Welke regel geeft een DataFrame met alleen de kolommen `day` en `wind`?",
        options={
            "one": "`df[\"day\", \"wind\"]`",
            "list": "`df[[\"day\", \"wind\"]]`",
            "chain": "`df[\"day\"][\"wind\"]`",
        },
        correct="list",
        why={
            "one": "Met één paar haken zoekt pandas één kolom met de naam `\"day\", \"wind\"`, en stopt met een fout. Kijk nog eens naar het blok *Let op* boven het recept.",
            "chain": "`df[\"day\"]` is al één kolom, en in die kolom zit geen kolom `wind`. Hoe vraag je om meer kolommen tegelijk?",
            "list": "De binnenste haken maken een lijst van twee namen, en de buitenste pakken die kolommen uit `df`.",
        },
    )
    choice_columns
    return (choice_columns,)


@app.cell(hide_code=True)
def text_exercise_windy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De winderige dagen

        Het vak leest het weer van die week als `df`. Maak `windy`: de rijen van de dagen met meer wind dan 20 kilometer per uur, met al hun kolommen. Vervang de `...`. De laatste regel is `windy`, zodat de check het ziet. De check probeert je code ook op een andere week.
        """),
        L.task("Maak `windy`: de rijen van `df` waar de wind meer dan 20 is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_windy():
    box_windy = L.code_box("windy", lines=6)
    box_windy
    return (box_windy,)


@app.cell(hide_code=True)
def check_windy(box_windy):
    result_windy = L.check("windy", box_windy.value)
    result_windy
    return (result_windy,)


@app.cell(hide_code=True)
def hints_windy():
    L.hints(
        "windy",
        [
            "Stap 4 van het recept: een mask op één kolom houdt rijen over.",
            "Het mask vergelijkt de kolom `wind` met 20. Daarna komt het tussen de vierkante haken van `df`.",
            "`windy = df[___]`, met het mask tussen de haken.",
        ],
        why="`df[\"wind\"] > 20` is `True` voor dinsdag, woensdag en zondag. Tussen de vierkante haken van `df` houdt dat mask die drie rijen over, met alle vier de kolommen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_warm():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De warme dagen van elke week

        Maak `warm_days(path, limit)` af. Die leest het CSV-bestand op `path`, het weer van een week met dezelfde vier kolommen, als `df`. Hij geeft de namen terug van de dagen die warmer zijn dan `limit`: de kolom `day` van die rijen. Vervang de `...`. De check roept `warm_days("lessons/data-pandas/weather.csv", 13)` aan: dat moet de dagen `tue`, `sat` en `sun` geven. Hij probeert ook een andere week, in `lessons/data-pandas/weather-2.csv`.
        """),
        L.task("Laat `warm_days(path, limit)` de namen teruggeven van de dagen die warmer zijn dan `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_warm():
    box_warm = L.code_box("warm-days", lines=8)
    box_warm
    return (box_warm,)


@app.cell(hide_code=True)
def check_warm(box_warm):
    result_warm = L.check("warm-days", box_warm.value)
    result_warm
    return (result_warm,)


@app.cell(hide_code=True)
def hints_warm():
    L.hints(
        "warm-days",
        [
            "Twee stappen: houd eerst de warme rijen over, met een mask; pak daarvan dan één kolom.",
            "Het mask vergelijkt de kolom `temp` met `limit`. Van de rijen die het overhoudt, wil je de kolom `day`.",
            "`warm = df[___]`, en dan `return warm[___]`.",
        ],
        why="`df[df[\"temp\"] > limit]` houdt de rijen over die warmer zijn dan `limit`, en `[\"day\"]` pakt hun namen. De functie leest het bestand op `path`, dus werkt hij voor elke week in dezelfde vorm.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Delen van een array kiezen](/course/open/np-indexing?lang=nl) koos een mask getallen uit een array; in een DataFrame selecteert een mask op één kolom hele rijen. In [Een tekstbestand lezen](/course/open/py-files?lang=nl) las je een bestand regel voor regel; `pd.read_csv` leest een hele tabel tegelijk, met zijn kolomnamen. In de volgende les, *Data beschrijven: gemiddelde, mediaan en spreiding*, vat je een kolom samen met een paar getallen en een figuur.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_columns, result_warm, result_windy):
    L.wrap_up(choice_columns, result_windy, result_warm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
