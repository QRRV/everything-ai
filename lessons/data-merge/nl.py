# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-merge. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-merge", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl) en [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`df` is de maand van het weerstation, een rij per dag. Wat geeft `df.groupby(\"sky\").size()`?",
        options={
            "kinds": "Hoeveel soorten lucht er zijn",
            "right": "Voor elke soort lucht: hoeveel dagen die heeft",
            "temps": "Voor elke soort lucht: hoeveel temperaturen die heeft, zonder de ontbrekende",
        },
        correct="right",
        why={
            "kinds": "Dat zou één getal zijn. `groupby` geeft een getal per groep: wat telt `size` in elke groep? Zie de tabel met nieuwe woorden in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "temps": "Dat doet `count`, op de kolom `temp`. `size` kijkt naar geen enkele kolom: wat telt hij? Zie de tabel met nieuwe woorden in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "right": "`size` telt de rijen van elke groep, en hier is elke rij een dag.",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="`sold = {\"sat\": 95}`, en daarna `sold[\"sat\"] = 80`. Wat is `sold` nu?",
        options={
            "both": "`{\"sat\": 95, \"sat\": 80}`",
            "right": "`{\"sat\": 80}`",
            "old": "`{\"sat\": 95}`",
        },
        correct="right",
        why={
            "both": "Kan een key twee keer in een dictionary staan? Zie de tabel met nieuwe woorden in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "old": "De tweede regel geeft de key `\"sat\"` een waarde. Wat gebeurt er met de oude? Zie de gok in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "right": "Een key staat maar één keer in een dictionary, dus de tweede regel vervangt zijn waarde: je zoekt `\"sat\"` op en vindt één antwoord.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het weer van een week staat in de ene tabel; de ijsjes die een kraampje op het strand elke dag verkocht, staan in een andere. Om te zien of warme dagen meer verkopen, moeten het weer en de verkoop van elke dag op één rij staan: met de hand zou je elke dag van de ene tabel opzoeken in de andere. Een merge doet dat voor alle rijen tegelijk.

    **Een voorbeeld uit het dagelijks leven.** Een klassenlijst heeft leerlingnummers en namen; de toetsuitslagen hebben leerlingnummers en cijfers. Om elke naam naast zijn cijfer te zetten, koppel je ze op het leerlingnummer.
    """)
    return


@app.cell(hide_code=True)
def predict_rows():
    guess_rows = L.guess(
        "rows",
        question="`weather` heeft 7 dagen, maandag tot en met zondag. Het kraampje was maandag dicht, dus `sales` heeft 6 rijen, dinsdag tot en met zondag. Hoeveel rijen denk je dat `pd.merge(weather, sales, on=\"day\")` heeft?",
        options={
            "six": "6",
            "seven": "7",
            "thirteen": "13",
        },
        reveal="**6.** merge zet rijen naast elkaar waar de dag gelijk is. Maandag staat alleen in `weather`, dus die valt weg, zonder iets te zeggen: een inner join, wat merge doet als je niet om iets anders vraagt. 13 zouden de tabellen onder elkaar zijn, niet samengevoegd.",
    )
    guess_rows
    return (guess_rows,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        `pd.merge` zet de rijen van twee tabellen naast elkaar waar ze dezelfde waarde hebben in een kolom die ze allebei hebben: de key, hier `day`. Elk paar rijen met dezelfde key wordt één rij, met de kolommen van allebei; dat heet een join. Wat er met een rij zonder match gebeurt, hangt af van `how`: een inner join laat hem weg; een left join houdt elke rij van de linkertabel, met `NaN` in de kolommen van de andere tabel. Een key die twee keer in de rechtertabel staat, vindt twee keer een match, dus zijn rij komt er twee keer uit.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **join** | twee tabellen samenvoegen tot één door rijen te koppelen op een kolom die ze allebei hebben; pandas noemt het merge | `pd.merge(weather, sales, on="day")` |
        | **key** | de kolom waarop de rijen gekoppeld worden; net als in een dictionary zegt zijn waarde welke rijen bij elkaar horen | `on="day"` |
        | **linker- en rechtertabel** | de eerste en de tweede tabel die je aan merge geeft | `pd.merge(left, right, on=...)` |
        | **inner join** | houdt alleen rijen waarvan de key in beide tabellen staat; wat merge vanzelf doet | `how="inner"` |
        | **left join** | houdt elke rij van de linkertabel, met `NaN` waar de rechtertabel geen match heeft | `how="left"` |

        Hieronder staat de week van het weer naast een lijst van het kraampje waarin de regel van zaterdag twee keer is ingetikt. De slider zoekt de dagen van `weather` één voor één op: een lijn verbindt een dag met elke rij van de lijst met die dag, en elke lijn wordt een rij van beide joins; een dag zonder lijn krijgt alleen in de left join een rij. Het blok begint met alle 7 dagen opgezocht.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lookup_cell():
    _once = "`{day}` staat één keer in de lijst: elke join krijgt één rij."
    _say = [
        "Nog geen dag opgezocht: beide joins zijn leeg.",
        "`mon` staat niet in de lijst: de inner join krijgt geen rij, de left join één met `NaN`.",
        *[_once.format(day=_day) for _day in ("tue", "wed", "thu", "fri")],
        "`sat` staat twee keer in de lijst: elke join krijgt twee rijen.",
        "`sun` staat één keer in de lijst. Alle 7 dagen opgezocht: de inner join heeft 7 rijen, de left join 8.",
    ]

    def _frame(days):
        days = int(days)
        return (C.lookup(days, {"title": "Zaterdag twee keer ingetikt", "inner": "inner join", "left": "left join", "row": "rij", "rows": "rijen",
                                "alt": "De week van het weer en de lijst van het kraampje, met zaterdag twee keer; een lijn verbindt elke opgezochte dag met elke rij van de lijst met die dag. Daarna de inner en de left join van die dagen. Opgezochte dagen: {k} van de 7. Rijen in de inner join: {n}; in de left join: {m}."}),
                _say[days])

    explore_lookup = L.explore("lookup", "dagen", _frame, start=7, values=list(range(8)), role="",
                               task="Schuif naar 0 en loop de dagen langs. Welke dag geeft de inner join geen rij, en welke geeft er twee?")
    explore_lookup
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Maandag geeft de inner join geen rij en de left join er één met `NaN`; zaterdag geeft elke join er twee. Zo heeft de inner join 7 rijen, evenveel als `weather`, en toch ontbreekt maandag en staat zaterdag er twee keer in. Alleen de left join, met 8, laat zien dat er een rij bij kwam.
        """),
        L.watch_out("merge zegt niets als een rij geen match vindt, of er twee vindt. Tel de rijen voor en na elke merge."),
        mo.md(r"""
        ### Recept

        Zo voeg je twee tabellen samen:

        1. Zoek de key: de kolom die beide tabellen hebben. Zijn waarden moeten in allebei op dezelfde manier geschreven zijn.
        2. Tel de rijen van beide tabellen: `len(weather)` en `len(sales)`. Bij een tabel geeft `len` het aantal rijen, zoals bij een lijst het aantal items.
        3. Voeg samen, en kies `how`: een inner join, wat merge vanzelf doet, of `how="left"` om elke rij van de linkertabel te houden.
        4. Tel de rijen opnieuw: een left join hoort er evenveel te hebben als de linkertabel. Meer betekent dat een key meer dan één keer in de rechtertabel staat; `sales.groupby("day").size()` laat zien welke.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De week van het weer uit [Tabellen met pandas](/course/open/data-pandas?lang=nl) en de ijsjes die het kraampje die week verkocht: allebei verzonnen door de course. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_merge():
    import pandas as pd

    # 1. de twee tabellen; allebei hebben ze de kolom day
    _weather = pd.read_csv("lessons/data-pandas/weather.csv")
    _sales = pd.read_csv("lessons/data-merge/sales.csv")
    print(_sales)
    # 2. het aantal rijen van allebei, voor de merge
    print(len(_weather), len(_sales))
    # 3. een inner join, en een left join
    _both = pd.merge(_weather, _sales, on="day")
    print(_both)
    _every = pd.merge(_weather, _sales, on="day", how="left")
    print(_every)
    # 4. het aantal rijen erna; dan de lijst met zaterdag twee keer ingetikt
    print(len(_both), len(_every))
    _twice = pd.read_csv("lessons/data-merge/sales-twice.csv")
    print(len(pd.merge(_weather, _twice, on="day", how="left")))
    print(_twice.groupby("day").size())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De inner join heeft 6 rijen, zonder maandag; de left join 7, met `NaN` voor de ijsjes van maandag, dus schrijft pandas `sold` als kommagetallen: 61.0 in plaats van 61. Weer en verkoop staan nu per dag op één rij: de warmste, zaterdag met 16,5 graden, verkocht de meeste, 95; de natste, woensdag met 8 millimeter regen, de minste, 18. Met zaterdag twee keer ingetikt heeft de left join 8 rijen, en `size` laat zien waarom: `sat` heeft er 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke dag van week 2

        Het vak leest de tweede week van het weer als `weather`, en wat het kraampje die week verkocht als `sales`. Het kraampje was maandag en vrijdag dicht. Maak `week`: een left join van `weather` en `sales` op `day`. Vervang de `...`. De laatste regel is `week`, zodat de check het ziet. De check probeert ook week 1.
        """),
        L.task("Maak `week`: een left join van `weather` en `sales` op `day`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week():
    box_week = L.code_box("week-2", lines=7)
    box_week
    return (box_week,)


@app.cell(hide_code=True)
def check_week(box_week):
    result_week = L.check("week-2", box_week.value)
    result_week
    return (result_week,)


@app.cell(hide_code=True)
def hints_week():
    L.hints(
        "week-2",
        [
            "Stap 3 van het recept, zoals de left join in het uitgewerkte voorbeeld.",
            "`weather` komt eerst, als linkertabel; de key na `on=`, de soort join na `how=`.",
            "`week = pd.merge(___, ___, on=___, how=___)`.",
        ],
        why="`pd.merge(weather, sales, on=\"day\", how=\"left\")` houdt alle 7 dagen van `weather`, met de ijsjes van `sales` ernaast: 33, 58 en 71 van dinsdag tot en met donderdag, 30 en 49 in het weekend, `NaN` op maandag en vrijdag.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_extra():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tel de rijen

        Maak `extra_rows(left_path, right_path, key)` af. Hij leest twee CSV-bestanden als `left` en `right`. Hij geeft terug hoeveel rijen een left join van `left` en `right` op `key` erbij doet: de rijen na de merge min de rijen ervoor. Vervang de `...`. De check roept `extra_rows("lessons/data-pandas/weather.csv", "lessons/data-merge/sales-twice.csv", "day")` aan: dat moet 1 geven. Hij probeert ook andere tabellen.
        """),
        L.task("Laat `extra_rows(left_path, right_path, key)` teruggeven hoeveel rijen een left join erbij doet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_extra():
    box_extra = L.code_box("extra-rows", lines=8)
    box_extra
    return (box_extra,)


@app.cell(hide_code=True)
def check_extra(box_extra):
    result_extra = L.check("extra-rows", box_extra.value)
    result_extra
    return (result_extra,)


@app.cell(hide_code=True)
def hints_extra():
    L.hints(
        "extra-rows",
        [
            "Stap 3 en 4 van het recept: de left join, dan de rijen erna min de rijen ervoor.",
            "In `key` zit de naam van de kolom, zoals `\"day\"`: gebruik hem na `on=` zonder aanhalingstekens.",
            "`merged = pd.merge(left, right, on=___, how=___)`, en `return len(___) - len(___)`.",
        ],
        why="`pd.merge(left, right, on=key, how=\"left\")` houdt elke rij van `left`, en een rij waarvan de key twee keer in `right` staat, komt er twee keer uit. Dus `len(merged) - len(left)` telt de rijen die erbij kwamen: 1 voor zaterdag twee keer ingetikt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fewer():
    mo.md(r"""
    ### Oefening 3. Minder rijen dan je verwachtte
    """)
    return


@app.cell(hide_code=True)
def choice_fewer_cell():
    choice_fewer = L.choice(
        "fewer-rows",
        question="`weather` heeft 7 rijen, maandag tot en met zondag. Een ander kraampje was elke dag open: zijn `sales` heeft 7 rijen. Toch heeft `pd.merge(weather, sales, on=\"day\")` maar 6 rijen. Wat is de waarschijnlijkste oorzaak?",
        options={
            "spelled": "Eén dag is in `sales` anders geschreven, zoals `Sat` voor `sat`, dus die vindt geen match",
            "missing": "Eén dag in `sales` heeft `NaN` voor de ijsjes",
            "twice": "Eén dag staat twee keer in `sales`",
        },
        correct="spelled",
        why={
            "missing": "merge kijkt alleen naar de key: een ontbrekende waarde in een andere kolom laat geen rij wegvallen. Welke rijen laat een inner join weg? Zie *Het idee*.",
            "twice": "Een key die twee keer in de rechtertabel staat, geeft meer rijen, niet minder: kijk naar `sat` in *Het idee*. Welke rijen laat een inner join weg?",
            "spelled": "Een inner join laat een rij weg waarvan de key niet in de andere tabel staat, en voor merge zijn `Sat` en `sat` twee keys. `sales[\"day\"].value_counts()` zou de vreemde eend laten zien, zoals in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
        },
    )
    choice_fewer
    return (choice_fewer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Dictionaries](/course/open/py-dicts?lang=nl) zocht je een waarde op via zijn key; een merge zoekt elke rij van de linkertabel op in de rechter, en vindt één rij, geen enkele of meerdere. In [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl) telde `size` de rijen per groep; hier vindt hij een key die twee keer in een tabel staat. In [Vragen stellen aan data](/course/open/data-explore?lang=nl) waren `Sunny` en `sunny` één lucht op twee manieren geschreven; in een merge kost dat een rij, zonder iets te zeggen. In de volgende les, *Gaten in een tabel: dropna en fillna*, beslis je wat je doet met de `NaN` die een left join achterlaat: weglaten, of iets invullen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_week, result_extra, choice_fewer):
    L.wrap_up(result_week, result_extra, choice_fewer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
