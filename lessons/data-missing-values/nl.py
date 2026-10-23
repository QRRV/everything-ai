# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-missing-values. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-missing-values", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_left():
    quick_left = L.choice(
        "quick-left",
        question="`weather` heeft de 7 dagen van een week. Het kraampje was maandag dicht, dus `sales` heeft 6 rijen. Wat staat er in `pd.merge(weather, sales, on=\"day\", how=\"left\")` in de kolom `sold`, op de rij van maandag?",
        options={
            "zero": "`0`",
            "right": "`NaN`",
            "none": "Niets: er is geen rij voor maandag",
        },
        correct="right",
        why={
            "zero": "merge weet niet dat het kraampje dicht was. Wat zet een left join waar de rechtertabel geen match heeft? Zie de tabel met nieuwe woorden in [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
            "none": "Dat doet een inner join. Welke rijen houdt een left join? Zie de tabel met nieuwe woorden in [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
            "right": "Een left join houdt elke rij van `weather`, en zet `NaN` waar `sales` geen match heeft: een ontbrekende waarde die de merge zelf maakte.",
        },
    )
    quick_left
    return (quick_left,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="Twee kolommen hebben allebei een gemiddelde van 10. De standaardafwijking van de eerste is 1, die van de tweede 4. Wat zegt dat?",
        options={
            "larger": "De tweede kolom heeft de grotere waarden",
            "right": "De waarden van de tweede kolom liggen verder van hun gemiddelde",
            "more": "De tweede kolom heeft meer waarden",
        },
        correct="right",
        why={
            "larger": "Ze hebben hetzelfde gemiddelde. Wat meet de standaardafwijking? Zie de tabel met nieuwe woorden in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "more": "De standaardafwijking telt geen waarden. Wat meet hij wel? Zie de tabel met nieuwe woorden in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "right": "De standaardafwijking zegt hoe ver de waarden meestal van het gemiddelde liggen: 1 bij de eerste kolom, 4 bij de tweede.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De left join van [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl) liet op maandag `NaN` achter, en het station van [Vragen stellen aan data](/course/open/data-explore?lang=nl) miste twee temperaturen en schreef -999 voor de wind van één dag. pandas slaat zulke gaten in een gemiddelde over, maar veel modellen stoppen met een fout bij een `NaN`. Dus voordat een model de data ziet, beslis je per gat: weglaten, of iets invullen.

    **Een voorbeeld uit het dagelijks leven.** Twee leerlingen misten een toets. Laat je ze weg uit het klassengemiddelde, geef je ze het gemiddelde, of geef je ze een 1? Elke keuze geeft een ander gemiddelde of een andere spreiding, en welke eerlijk is, hangt af van waarom ze de toets misten.
    """)
    return


@app.cell(hide_code=True)
def predict_fill():
    guess_fill = L.guess(
        "fill",
        question="Het station mat 28 van de 30 temperaturen van de maand: hun gemiddelde is 9,66 graden, hun standaardafwijking 2,18. Je vult de 2 ontbrekende dagen in met dat gemiddelde. Wat gebeurt er met het gemiddelde en de standaardafwijking van alle 30?",
        options={
            "same": "Ze blijven allebei gelijk",
            "smaller": "Het gemiddelde blijft gelijk; de standaardafwijking wordt kleiner",
            "larger": "Het gemiddelde blijft gelijk; de standaardafwijking wordt groter",
        },
        reveal="**Het gemiddelde blijft 9,66, en de standaardafwijking zakt naar 2,10.** Twee dagen op het gemiddelde verschuiven het niet, maar ze voegen ook geen afstand toe, dus liggen de 30 waarden er meestal dichter bij: de maand lijkt rustiger dan de metingen laten zien.",
    )
    guess_fill
    return (guess_fill,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een gat kun je weglaten of invullen. `dropna` laat elke rij met een gat weg, of elke kolom met een gat. `fillna` vult een waarde in die je weet, zoals 0 ijsjes op een dag dat het kraampje dicht was, of, als je hem niet weet, een gok uit de kolom, zoals het gemiddelde of de mediaan: dat heet imputatie. Een gok ziet er daarna precies uit als een meting, dus je markeert de gaten eerst, in een eigen kolom.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **na_values** | de codes die `read_csv` als ontbrekend moet lezen, zoals -999 | `pd.read_csv(path, na_values=[-999])` |
        | **dropna** | laat elke rij weg die een gat heeft; met `axis="columns"` elke kolom met een gat | `df.dropna()` |
        | **fillna** | zet een waarde in elk gat, en geeft de ingevulde kolom of tabel terug | `df["temp"].fillna(0)` |
        | **imputatie** | een gat invullen met een redelijke gok, zoals het gemiddelde of de mediaan van zijn kolom | `fillna(df["temp"].mean())` |
        | **een nieuwe kolom** | `df["naam"] = ...` voegt een kolom toe aan een tabel, zoals `d[key] = value` een key toevoegt aan een dictionary | `df["temp_missing"] = ...` |
        | **missing indicator** | een extra kolom die `True` is waar een waarde ontbrak, zodat je een gok nog altijd van een meting kunt onderscheiden, en een model kan leren of ontbreken ertoe doet | `df["temp_missing"] = df["temp"].isna()` |

        Hieronder de maand van het station: een stip per dag op zijn temperatuur. Stel dat de thermometer op meer dagen was uitgevallen: met de slider kies je hoeveel van de 30 dagen er ontbreken, vanaf de 2 die echt ontbreken. De open stippen zijn de ontbrekende dagen, ingevuld met het gemiddelde van de gemeten dagen, op de gestreepte lijn.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lost_cell():
    def _frame(k):
        k = int(k)
        n = C.numbers(k)
        _mean, _drop, _fill = L.number(n["mean"]), L.number(n["drop"]), L.number(n["fill"])
        return (C.lost_picture(k, {"title": "Een maand met gaten", "day": "dag", "temp": "temperatuur (°C)",
                                   "alt": f"De temperatuur van de 30 dagen van de maand, een stip per dag. {30 - k} dagen zijn gemeten, als dichte stippen; de {k} ontbrekende dagen zijn open stippen op een gestreepte lijn op het gemiddelde van de gemeten dagen, {_mean} graden."}),
                f"{k} van de 30 dagen ontbreken. Weggelaten: {n['left']} dagen over, een standaardafwijking van {_drop}. Ingevuld met het gemiddelde van de gemeten dagen, {_mean} graden: 30 dagen, een standaardafwijking van {_fill}.")

    explore_lost = L.explore("lost", "ontbrekende dagen", _frame, start=2, values=C.LOST, role="",
                             task="Schuif van 2 naar 20 ontbrekende dagen. Wat gebeurt er met de standaardafwijking als je ze weglaat, en als je ze invult met het gemiddelde?")
    explore_lost
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Weglaten houdt ongeveer de spreiding van de dagen die overblijven, hier bij elke stand tussen 2,1 en 2,5 graden, maar wel met minder dagen: waren er andere dagen uitgevallen, dan had de spreiding verder van 2,18 kunnen liggen. Invullen met het gemiddelde houdt alle 30 dagen en het gemiddelde, maar elke ingevulde dag ligt op het gemiddelde en voegt geen spreiding toe: met 20 ontbrekende dagen is de standaardafwijking 1,37, tegen 2,46 voor de 10 gemeten dagen.
        """),
        L.watch_out("`dropna` en `fillna` geven een nieuwe tabel of kolom terug; `df` zelf blijft zoals hij was. Wil je een ingevulde kolom houden, zet hem dan terug: `df[\"temp\"] = df[\"temp\"].fillna(...)`."),
        mo.md(r"""
        ### Recept

        Zo pak je de gaten in een tabel aan:

        1. Maak van elk gat een `NaN`, ook van een code zoals -999: `pd.read_csv(path, na_values=[-999])`.
        2. Tel de gaten per kolom, `df.isna().sum()`, en vraag je bij elk gat af waarom het er is.
        3. Weet je de waarde, zoals 0 ijsjes op een dag dat het kraampje dicht was? Vul die in: `fillna(0)`.
        4. Zo niet: laat de rijen met een gat weg, `df.dropna()`, als het er weinig zijn; laat de kolommen met een gat weg, `df.dropna(axis="columns")`, als ze grotendeels leeg zijn; of markeer de gaten, `df["temp_missing"] = df["temp"].isna()`, en vul het gemiddelde of de mediaan van de kolom in.
        5. Vergelijk `describe()` ervoor en erna: `count`, `mean`, `std` en `50%`, de mediaan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De maand van het weerstation uit [Vragen stellen aan data](/course/open/data-explore?lang=nl), en de week van het kraampje uit [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl): allebei verzonnen door de course. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_missing():
    import pandas as pd

    # 1. de maand van het station, met de code -999 gelezen als ontbrekend
    _df = pd.read_csv("lessons/data-explore/station.csv", na_values=[-999])
    # 2. de gaten per kolom
    print(_df.isna().sum())
    # 3. de week van het kraampje: het was maandag dicht, dus het verkocht 0 ijsjes
    _weather = pd.read_csv("lessons/data-pandas/weather.csv")
    _sales = pd.read_csv("lessons/data-merge/sales.csv")
    _week = pd.merge(_weather, _sales, on="day", how="left")
    print(_week["sold"].fillna(0))
    # 4. laat de rijen met een gat weg, of de kolommen met een gat: de rijen en kolommen die overblijven
    print(_df.dropna().shape, _df.dropna(axis="columns").shape)
    # 4. of markeer de ontbrekende temperaturen, en vul ze dan in met het gemiddelde
    _df["temp_missing"] = _df["temp"].isna()
    _filled = _df["temp"].fillna(_df["temp"].mean())
    print(_df[_df["temp_missing"]])
    # 5. de temperaturen ervoor en erna
    print(_df["temp"].describe())
    print(_filled.describe())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met -999 gelezen als ontbrekend heeft de wind ook een gat, op dag 17. Op de maandag van het kraampje zet `fillna(0)` een 0 waar de left join `NaN` had: het kraampje was dicht, dus 0 is geen gok. `dropna()` houdt 27 van de 30 dagen over, en de drie dagen met een gat nemen hun andere metingen mee; `dropna(axis="columns")` houdt alle 30 dagen, maar zonder `temp` en `wind`. Ingevuld met het gemiddelde heeft de temperatuur een `count` van 30 en hetzelfde gemiddelde, 9,66, maar zijn `std` zakt van 2,18 naar 2,10 en zijn mediaan van 10,75 naar 10,25. pandas deelt voor `std` door één minder dan het aantal waarden, zoals de melding *Let op* in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) zei dat sommige tools doen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_filled():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De wind van dag 17

        Het vak leest de maand van het station als `df`, met -999 gelezen als ontbrekend: de wind heeft een gat op dag 17. Maak `filled`: de kolom `wind`, met zijn gat ingevuld met de mediaan van de wind. Vervang de `...`; de laatste regel is `filled`, zodat de check het ziet. De check probeert ook de andere maand.
        """),
        L.task("Maak `filled`: de wind van `df`, met het gat ingevuld met de mediaan van de wind."),
    ])
    return


@app.cell(hide_code=True)
def exercise_filled():
    box_filled = L.code_box("filled", lines=6)
    box_filled
    return (box_filled,)


@app.cell(hide_code=True)
def check_filled(box_filled):
    result_filled = L.check("filled", box_filled.value)
    result_filled
    return (result_filled,)


@app.cell(hide_code=True)
def hints_filled():
    L.hints(
        "filled",
        [
            "Stap 4 van het recept, zoals het gemiddelde in het uitgewerkte voorbeeld, maar met de mediaan en de kolom `wind`.",
            "`fillna` zet de waarde tussen zijn haakjes in het gat. pandas heeft een methode voor de mediaan, net als `.mean()` voor het gemiddelde.",
            "`filled = df[\"wind\"].fillna(df[\"wind\"].___())`.",
        ],
        why="`df[\"wind\"].median()` is de mediaan van de 29 gemeten windsnelheden, 21, en `df[\"wind\"].fillna(...)` zet hem op de plek van het gat van dag 17.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eerst markeren, dan invullen

        Maak `fill_flag(path, column)` af: hij leest het CSV-bestand in `path` als `df`, met -999 gelezen als ontbrekend, en geeft `df` terug. Vervang de `...` door twee regels: één die een missing-indicatorkolom maakt met de naam `f"{column}_missing"`, en dan één die de gaten van de kolom `column` invult met zijn mediaan. De check roept `fill_flag("lessons/data-explore/station.csv", "temp")` aan, en probeert ook andere bestanden en kolommen.
        """),
        L.task("Laat `fill_flag(path, column)` een missing-indicatorkolom maken en dan de kolom invullen met zijn mediaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("fill-flag", lines=8)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("fill-flag", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "fill-flag",
        [
            "Stap 4 van het recept, in zijn volgorde: eerst de gaten markeren, dan invullen.",
            "Twee regels in de vorm `df[naam] = ...`: een nieuwe kolom `f\"{column}_missing\"` die `True` is op elk gat, en dan de ingevulde kolom, terug in `df[column]`.",
            "`df[f\"{column}_missing\"] = df[column].isna()`, en dan `df[column] = df[column].fillna(df[column].___())`.",
        ],
        why="`df[column].isna()` is `True` op elk gat, en als de kolom `f\"{column}_missing\"` onthoudt hij waar ze zaten, ook nadat `df[column] = df[column].fillna(df[column].median())` ze heeft ingevuld.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_discount():
    mo.md(r"""
    ### Oefening 3. Waarom ontbreekt het?
    """)
    return


@app.cell(hide_code=True)
def choice_discount_cell():
    choice_discount = L.choice(
        "discount",
        question="Een webwinkel heeft een tabel met bestellingen, met een kolom `discount` in euro's. Die is leeg bij elke bestelling zonder kortingscode. Wat doe je met die gaten?",
        options={
            "mean": "Ze invullen met de gemiddelde korting van de andere bestellingen",
            "right": "Ze invullen met 0",
            "drop": "De bestellingen zonder korting weglaten",
        },
        correct="right",
        why={
            "mean": "Dan geef je bestellingen zonder code een korting die ze nooit kregen. Wat was de korting van een bestelling zonder code?",
            "drop": "Dan staan er alleen bestellingen met korting in de tabel, en hun gemiddelde korting ziet eruit alsof elke bestelling er een had. Wat was de korting van een bestelling zonder code?",
            "right": "Geen code, dus geen korting: als je weet waarom een waarde ontbreekt, is hij niet onbekend. Vul in wat hij was, 0, net als bij het kraampje op de dag dat het dicht was.",
        },
    )
    choice_discount
    return (choice_discount,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Vragen stellen aan data](/course/open/data-explore?lang=nl) vond gaten en beloofde dat je ze zou aanpakken; de left join van [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl) maakte zelf gaten, en één ervan kreeg de 0 waar hij voor stond. Invullen met het gemiddelde maakt de standaardafwijking uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) kleiner. Later leert een model van het ene deel van de data en wordt het getest op een ander deel; het gemiddelde of de mediaan die je invult, moet dan alleen uit het deel komen waarvan het leert, en daar komt een latere les op terug. In de volgende les, *Twee kolommen samen: een verband is geen oorzaak*, kijk je naar twee kolommen tegelijk, en naar wat een derde met ze kan doen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_filled, result_flag, choice_discount):
    L.wrap_up(result_filled, result_flag, choice_discount)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
