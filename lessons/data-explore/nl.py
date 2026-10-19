# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-explore. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-explore", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tabellen met pandas](/course/open/data-pandas?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`df` heeft een kolom `wind`. Welke regel houdt de rijen over met meer wind dan 20?",
        options={
            "mask": "`df[df[\"wind\"] > 20]`",
            "column": "`df[\"wind\"] > 20`",
            "name": "`df[\"wind\" > 20]`",
        },
        correct="mask",
        why={
            "column": "Dat geeft `True` of `False` voor elke rij. Waar moet dat mask staan om de rijen over te houden? Zie het recept in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "name": "Dat vergelijkt de tekst `\"wind\"` met 20. Wat moet het mask vergelijken? Zie de tabel met nieuwe woorden in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "mask": "`df[\"wind\"] > 20` is `True` voor de rijen met meer wind dan 20; tussen de vierkante haken van `df` houdt het die rijen over.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_middle():
    quick_middle = L.choice(
        "quick-middle",
        question="In een kolom getallen is het gemiddelde 30 en de mediaan 25. Waar wijst dat op?",
        options={
            "low": "Een paar heel kleine waarden",
            "same": "Nergens op: die twee verschillen altijd zoveel",
            "high": "Een paar heel grote waarden",
        },
        correct="high",
        why={
            "low": "Heel kleine waarden trekken het gemiddelde omlaag, onder de mediaan. Naar welke kant wordt het gemiddelde hier getrokken? Zie de figuur in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "same": "Als de waarden gelijkmatig rond het midden liggen, zijn het gemiddelde en de mediaan ongeveer gelijk. Wat trekt ze uit elkaar? Zie de tabel met nieuwe woorden in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "high": "Het gemiddelde wordt omhoog getrokken, boven de mediaan. Dat doen een paar heel grote waarden, zoals een uitschieter.",
        },
    )
    quick_middle
    return (quick_middle,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leert van alle data die je het geeft, fouten inbegrepen. Een ontbrekende waarde, een sensor die -999 schrijft als hij het niet doet, of `Sunny` naast `sunny`: een model ziet niet dat die fout zijn, en leert ze. Daarom kijk je naar de data voordat je iets bouwt: wat zit erin, wat ontbreekt er, en wat is vreemd?

    **Een voorbeeld uit het dagelijks leven.** Een tweedehands fiets kopen: voordat je betaalt, bekijk je hem van alle kanten, probeer je de remmen, en vraag je naar alles wat er raar uitziet.
    """)
    return


@app.cell(hide_code=True)
def predict_skip():
    guess_skip = L.guess(
        "skip",
        question="Een weerstation mat 30 dagen lang elke dag de temperatuur, maar op twee dagen werd niets vastgelegd. Wat denk je dat `df[\"temp\"].mean()` met die twee dagen doet?",
        options={
            "skip": "Hij laat ze weg, en geeft het gemiddelde van de andere 28",
            "zero": "Hij telt ze als 0 graden",
            "error": "Hij stopt met een fout",
        },
        reveal="**Hij laat ze weg.** pandas slaat ontbrekende waarden over in `.mean()`, `.sum()` en dergelijke, zonder iets te zeggen: het gemiddelde gaat over 28 dagen. Dat is handig, maar je weet het alleen als je kijkt.",
    )
    guess_skip
    return (guess_skip,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Exploratory data analysis is een reeks vragen die je aan elke nieuwe dataset stelt, voordat je er iets mee bouwt. Hoe groot is hij? Wat staat er in elke kolom? Wat ontbreekt er? Wat kan niet kloppen? pandas beantwoordt de meeste in één regel per vraag. Een ontbrekende waarde is een plek waar niets is vastgelegd: pandas laat die zien als `NaN`, en slaat hem over in sommen en gemiddelden. Er zijn een paar soorten vreemde waarden: onmogelijke getallen, zoals een negatieve hoeveelheid regen; codes die "geen meting" betekenen, zoals -999; en één ding op twee manieren geschreven, zoals `sunny` en `Sunny`.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **exploratory data analysis** | een nieuwe dataset bekijken met samenvattingen en grafieken, vóór elk model, om te leren wat erin zit en problemen te vinden | |
    | **ontbrekende waarde** | een plek waar niets is vastgelegd; pandas laat die zien als `NaN` | `NaN` |
    | **describe** | per kolom met getallen: hoeveel waarden er zijn (`count`), het gemiddelde, de standaardafwijking, de kleinste waarde, het 0,25-, 0,5- en 0,75-kwantiel (te zien als `25%`, `50%` en `75%`), en de grootste waarde | `df.describe()` |
    | **isna** | `True` voor elke ontbrekende waarde en `False` voor de rest: een tabel zo groot als `df` | `df.isna()` |
    | **`.sum()` en `.mean()` van een mask** | `.sum()` telt de `True`-waarden per kolom, want `True` telt als 1 en `False` als 0; `.mean()` geeft het deel `True`-waarden, van 0 tot 1 | `df.isna().sum()` |
    | **value_counts** | hoe vaak elke waarde van een kolom voorkomt, de meest voorkomende eerst | `df["sky"].value_counts()` |
    """)
    return


@app.cell(hide_code=True)
def picture_month():
    mo.vstack([
        mo.Html(C.picture({"title": "Een maand van het weerstation", "temp": "temperatuur (°C)", "wind": "wind (km/u)", "day": "dag",
                           "alt": "Twee lijngrafieken van de 30 dagen. De temperatuur ligt tussen 5,5 en 12 graden, met gaten op dag 9 en dag 22. De wind ligt tussen 7 en 33 kilometer per uur, maar op dag 17 duikt hij naar -999, zodat alle andere dagen plat lijken."})),
        mo.md(r"""
        De maand van het station, dag voor dag. In de temperatuur ontbreken twee dagen: de lijn heeft daar gaten. De wind blijft tussen 7 en 33 kilometer per uur, maar op dag 17 duikt hij naar -999, zo ver omlaag dat alle andere dagen plat lijken. Eén blik op een grafiek laat allebei zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("pandas slaat ontbrekende waarden over in `.mean()`, `.sum()` en `describe`, zonder iets te zeggen. Een gemiddelde kan over minder waarden gaan dan je denkt: kijk naar `count`."),
        mo.md(r"""
        ### Recept

        Zo verken je een nieuwe tabel:

        1. Bekijk de grootte en de eerste rijen: `df.shape` en `df.head()`.
        2. Vat de kolommen met getallen samen: `df.describe()`. Kijk naar `count` (ontbreekt er iets?), naar `min` en `max` (kan dat?), en naar het gemiddelde tegenover de mediaan, `50%`.
        3. Tel de ontbrekende waarden per kolom: `df.isna().sum()`.
        4. Tel de waarden van elke kolom met tekst, zoals `df["sky"].value_counts()`. Is één ding op twee manieren geschreven?
        5. Schrijf op wat vreemd is, als vragen voor wie de data maakte, voordat je er iets mee bouwt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen weerstation mat een maand lang elke dag: de temperatuur in graden Celsius, de regen in millimeters, de wind in kilometers per uur, de richting van de wind, en hoe de lucht eruitzag. Het bestand is `lessons/data-explore/station.csv`. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_station():
    import pandas as pd

    # 1. de grootte en de eerste rijen
    _df = pd.read_csv("lessons/data-explore/station.csv")
    print(_df.shape)
    print(_df.head())
    # 2. een samenvatting van elke kolom met getallen
    print("describe:")
    print(_df.describe())
    # 3. de ontbrekende waarden per kolom
    print("missing:")
    print(_df.isna().sum())
    # 4. hoe vaak elke soort lucht voorkomt
    print("sky:")
    print(_df["sky"].value_counts())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tabel heeft 30 rijen en 6 kolommen. In `describe` is `count` 30 voor elke kolom met getallen, behalve `temp`, die er 28 heeft: twee temperaturen ontbreken, en `isna().sum()` zegt hetzelfde. De kleinste wind is -999; daardoor is de gemiddelde wind -12,1 kilometer per uur, en dat kan niet: wind is nooit lager dan 0. De mediaan van de wind, `50%`, is 21: één waarde trekt het gemiddelde er ver vandaan. `describe` laat alleen de kolommen met getallen zien; `value_counts` laat zien dat de lucht op 12 dagen `sunny` was en op 1 dag `Sunny`: dezelfde lucht, anders geschreven. Stap 5, de vragen om op te schrijven: waarom ontbreken dag 9 en 22? Is -999 de code van het station voor "geen meting"? Moet `Sunny` tellen als `sunny`?
    """)
    return


@app.cell(hide_code=True)
def text_exercise_code():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een vreemd maximum
    """)
    return


@app.cell(hide_code=True)
def choice_code_cell():
    choice_code = L.choice(
        "error-code",
        question="Een sporthorloge schrijft de hartslag, in slagen per minuut, in een kolom `heart_rate`. In `df.describe()` is de grootste waarde van `heart_rate` `999`. Wat is het waarschijnlijkst?",
        options={
            "workout": "Een heel zware training",
            "code": "Een code die het horloge schrijft als het niets meet",
            "typo": "Een tikfout voor 99",
        },
        correct="code",
        why={
            "workout": "Kan een hart 999 keer per minuut slaan, ook bij de zwaarste training? Wat kan een getal als 999 nog meer zijn?",
            "typo": "Een hartslag van 99 kan, maar geeft een tikfout precies 999, allemaal negens? Wat kan zo'n getal nog meer zijn?",
            "code": "Geen hart slaat 999 keer per minuut, en een getal van allemaal negens schrijven veel apparaten als ze niets meten, net als de -999 van het station. Behandel het als een ontbrekende waarde, en vraag het aan wie de data maakte.",
        },
    )
    choice_code
    return (choice_code,)


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat ontbreekt er?

        Het vak leest de maand van het station als `df`. Maak `missing`: het deel ontbrekende waarden in elke kolom, van 0 tot 1. Vervang de `...`. De laatste regel is `missing`, zodat de check het ziet. De check probeert ook een andere maand.
        """),
        L.task("Maak `missing`: het deel ontbrekende waarden in elke kolom van `df`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=6)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "Twee stappen: markeer eerst elke ontbrekende waarde, en maak dan van elke kolom markeringen één getal.",
            "`df.isna()` markeert de ontbrekende waarden met `True`. Wat in de tabel met nieuwe woorden geeft het deel `True`-waarden in elke kolom?",
            "`missing = df.___().___()`.",
        ],
        why="`df.isna()` is `True` waar een waarde ontbreekt, en `.mean()` van dat mask geeft het deel `True`-waarden in elke kolom: 2 van de 30, ongeveer 0,067, voor `temp`, en 0 voor de andere kolommen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoe vaak komt elke waarde voor?

        Maak `counts(path, column)` af. Hij leest het CSV-bestand op `path` als `df`, en geeft terug hoe vaak elke waarde van de kolom met de naam `column` voorkomt. Vervang de `...`. De check roept `counts("lessons/data-explore/station.csv", "sky")` aan, en probeert ook een andere maand en een andere kolom.
        """),
        L.task("Laat `counts(path, column)` teruggeven hoe vaak elke waarde van de kolom `column` voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=7)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "Stap 4 van het recept: neem eerst de kolom, en tel dan zijn waarden.",
            "In `column` zit de naam van een kolom, zoals `\"sky\"`: gebruik hem tussen de vierkante haken van `df`, zonder aanhalingstekens.",
            "`return df[___].___()`.",
        ],
        why="`df[column]` neemt de kolom waarvan de naam in `column` zit, en `.value_counts()` telt hoe vaak elk van zijn waarden voorkomt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tabellen met pandas](/course/open/data-pandas?lang=nl) las je een tabel en selecteerde je er rijen uit; nu stel je hem vragen. In [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) vatte je één kolom samen; `describe` doet dat voor elke kolom met getallen tegelijk. Later in de course wordt de data waarop een model traint eerst zo bekeken, en worden ontbrekende waarden en codes zoals -999 aangepakt voordat een model ze ziet. In de volgende les, *Waar komt deze data vandaan?*, stel je de vragen die geen tabel beantwoordt: wie maakte de data, hoe, en wat mag je ermee?
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code, result_counts, result_missing):
    L.wrap_up(choice_code, result_missing, result_counts)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
