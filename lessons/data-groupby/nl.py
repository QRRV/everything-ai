# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-groupby. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-groupby", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vragen stellen aan data](/course/open/data-explore?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_skip():
    quick_skip = L.choice(
        "quick-skip",
        question="De kolom `df[\"temp\"]` heeft 30 rijen, en 2 daarvan zijn `NaN`. Over hoeveel waarden neemt `df[\"temp\"].mean()` het gemiddelde?",
        options={
            "all": "30: de ontbrekende tellen als 0",
            "right": "28",
            "error": "Geen: hij stopt met een fout",
        },
        correct="right",
        why={
            "all": "pandas vult geen 0 in. Wat doet hij in een gemiddelde met een ontbrekende waarde? Zie de melding *Let op* in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "error": "pandas stopt niet. Wat doet hij in een gemiddelde met een ontbrekende waarde? Zie de melding *Let op* in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "right": "pandas slaat ontbrekende waarden over in een gemiddelde, zonder iets te zeggen: het gemiddelde gaat over de 28 die er zijn.",
        },
    )
    quick_skip
    return (quick_skip,)


@app.cell(hide_code=True)
def quick_check_counts():
    quick_counts = L.choice(
        "quick-counts",
        question="Wat geeft `df[\"sky\"].value_counts()`?",
        options={
            "kinds": "Hoeveel verschillende waarden de kolom `sky` heeft",
            "right": "Hoe vaak elke waarde van `sky` voorkomt",
            "all": "Hoeveel waarden de kolom `sky` in totaal heeft",
        },
        correct="right",
        why={
            "kinds": "Dat is één getal; `value_counts` geeft er een voor elke waarde. Welk getal? Zie de tabel met nieuwe woorden in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "all": "Dat is één getal; `value_counts` geeft er een voor elke waarde. Welk getal? Zie de tabel met nieuwe woorden in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "right": "Eén telling per waarde, de meest voorkomende eerst: in de maand van het weerstation `sunny` 12 keer, `rainy` 9, `cloudy` 8 en `Sunny` één keer.",
        },
    )
    quick_counts
    return (quick_counts,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De 342 pinguïns met een lichaamsgewicht wegen gemiddeld 4202 gram. Maar dat ene getal gooit drie soorten op één hoop: hoe zwaar is een Adelie, en hoe zwaar een Gentoo? Met een mask kun je één soort overhouden en daarvan het gemiddelde nemen, en dat herhalen voor elke soort: hier drie keer, honderd keer voor honderd groepen. Een samenvatting per groep kost één regel.

    **Een voorbeeld uit het dagelijks leven.** Een hardloopclub wil haar leeftijdsgroepen vergelijken: ze legt de tijden van haar leden op stapels, één per leeftijdsgroep, neemt de gemiddelde tijd van elke stapel, en zet de gemiddelden in een lijst, één regel per leeftijdsgroep.
    """)
    return


@app.cell(hide_code=True)
def predict_per_species():
    guess_per_species = L.guess(
        "per-species",
        question="`df` is de tabel met 344 pinguïns, met onder meer de kolommen `species` en `body_mass_g`. Wat denk je dat `df.groupby(\"species\")[\"body_mass_g\"].mean()` geeft?",
        options={
            "one": "Eén getal: het gemiddelde lichaamsgewicht van alle pinguïns",
            "per": "Drie getallen: het gemiddelde lichaamsgewicht van elke soort",
            "every": "Een getal voor elke pinguïn: het gemiddelde van zijn soort",
        },
        reveal="**Drie getallen, één per soort:** 3701 gram voor een Adelie, 3733 voor een Chinstrap en 5076 voor een Gentoo.",
    )
    guess_per_species
    return (guess_per_species,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        `groupby` werkt in drie stappen, vaak split-apply-combine genoemd. Hij splitst de rijen in groepen op hun waarde in één kolom; hij doet in elke groep dezelfde berekening, zoals een gemiddelde van een andere kolom; en hij voegt de uitkomsten samen tot één Series, met een rij per groep en de groep als rijlabel. De groepen staan op volgorde van hun naam.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **groupby** | de rijen in groepen splitsen op de waarden van een kolom, en per groep één getal uitrekenen | `df.groupby("species")` |
        | **groep** | de rijen met dezelfde waarde in de kolom waarop je groepeert, zoals alle Gentoos | |
        | **één getal per groep** | een kolom na `groupby`, en dan wat je uitrekent: `.mean()`, `.median()`, `.min()`, `.max()` | `df.groupby("species")["body_mass_g"].mean()` |
        | **count** | per groep hoeveel waarden de kolom heeft; ontbrekende waarden tellen niet mee | `.count()` |
        | **size** | per groep hoeveel rijen hij heeft, met of zonder ontbrekende waarden | `df.groupby("species").size()` |
        | **agg** | meerdere getallen per groep tegelijk, in een tabel met een kolom voor elk; je geeft hun namen in een lijst | `.agg(["mean", "count"])` |

        Elke stip in het plaatje is een pinguïn, op de hoogte van zijn lichaamsgewicht, in de kleur en vorm van zijn soort: een rondje is een Adelie, een driehoek een Chinstrap en een vierkantje een Gentoo. Het korte streepje dwars over elke groep is het gemiddelde ervan, het ene getal dat `groupby` geeft.
        """),
        mo.Html(C.picture({"title": "Per soort", "x": "soort", "mass": "lichaamsgewicht (g)",
                           "alt": "Het lichaamsgewicht van 342 pinguïns per soort, een stip per pinguïn en een streepje op elk gemiddelde. De Adelies en de Chinstraps liggen rond 3700 gram, van ongeveer 2700 tot 4800; de Gentoos hoger, van 3950 tot 6300, met een gemiddelde van 5076."})),
        mo.md(r"""
        Per eiland heeft Biscoe het zwaarste gemiddelde: 4716 gram, een kilo meer dan Dream en Torgersen. Op Biscoe wonen twee soorten. De slider bepaalt hoeveel van zijn Gentoos meetellen, te beginnen bij alle 123 met een lichaamsgewicht.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_gentoos_cell():
    def _frame(k):
        k = int(k)
        m = C.island_means(k)
        return (C.island_picture(k, {"title": "Per eiland", "x": "eiland", "mass": "lichaamsgewicht (g)",
                                     "alt": f"Het lichaamsgewicht van de pinguïns per eiland, een stip per pinguïn en een streepje op elk gemiddelde. Biscoe, met zijn 44 Adelies en {k} van zijn 123 Gentoos: {m['Biscoe']} gram. Dream, Adelies en Chinstraps: {m['Dream']}. Torgersen, alleen Adelies: {m['Torgersen']}."}),
                f"Biscoe: 44 Adelies en {k} Gentoos, gemiddeld {m['Biscoe']} gram. Dream: {m['Dream']}. Torgersen: {m['Torgersen']}.")

    explore_gentoos = L.explore("gentoos", "Gentoos", _frame, start=123, values=C.GENTOOS, role="",
                                task="Haal de Gentoos van Biscoe weg, tot 0. Hoe zwaar is Biscoe dan, naast Dream en Torgersen?")
    explore_gentoos
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Zonder Gentoos weegt Biscoe 3710 gram, naast 3713 voor Dream en 3706 voor Torgersen: de Adelies van Biscoe wegen wat de pinguïns van de andere eilanden wegen. Het gemiddelde is alleen hoger door wie er wonen: de Gentoos, die in deze tabel alleen op Biscoe wonen.
        """),
        L.watch_out("Een gemiddelde per groep zegt waar het midden van elke groep ligt, niet op hoeveel waarden het rust of wat er verder verschilt tussen de groepen: vraag om de telling ernaast. En een rij zonder waarde in de kolom waarop je groepeert, zit in geen enkele groep: `groupby` laat hem weg, zonder iets te zeggen."),
        mo.md(r"""
        ### Recept

        Zo vat je een tabel per groep samen:

        1. Kies de kolom waarvan de waarden de groepen maken, en de kolom die je samenvat: `df.groupby("species")["body_mass_g"]`.
        2. Vraag om één getal per groep: `.mean()`, of om meerdere tegelijk: `.agg(["mean", "count"])`.
        3. Lees de telling naast elk gemiddelde: een gemiddelde van een paar waarden zegt weinig. `.size()` geeft de rijen per groep.
        4. Voordat je een verschil tussen groepen verklaart: vraag je af wat er verder verschilt, en groepeer ook op die kolom: `df.groupby(["island", "species"])`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns van het plaatje zijn de Palmer-pinguïns, in `data/penguins.csv`: 344 pinguïns van drie soorten, op drie eilanden, van 2007 tot 2009 gemeten door Kristen Gorman en het Palmer Station Long Term Ecological Research-programma, licentie CC0. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_groupby():
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    # 1. en 2. het gemiddelde lichaamsgewicht van elke soort
    print(_df.groupby("species")["body_mass_g"].mean())
    # 3. naast het gemiddelde: hoeveel waarden (count) en hoeveel rijen (size) elke soort heeft
    print(_df.groupby("species")["body_mass_g"].agg(["mean", "count", "size"]))
    # 3. en de rijen per geslacht: een pinguïn waarvan het geslacht niet is vastgelegd, zit in geen groep
    print(_df.groupby("sex").size())
    # 4. per eiland; dan per eiland en soort samen
    print(_df.groupby("island")["body_mass_g"].mean())
    print(_df.groupby(["island", "species"])["body_mass_g"].agg(["mean", "count"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij de Adelies en de Gentoos is `count` één lager dan `size`: van elk heeft één pinguïn geen lichaamsgewicht. Per geslacht tellen de groepen 165 en 168 pinguïns, 333 van de 344: de 11 waarvan het geslacht niet is vastgelegd, zitten in geen groep. Per eiland en soort, met een lijst van twee namen, wegen de Adelies overal ongeveer hetzelfde: 3710, 3688 en 3706 gram. Er zijn vijf groepen, geen negen: een groep komt er alleen voor een paar dat in de tabel staat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sky():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De temperatuur per soort lucht

        Het vak leest de maand van het weerstation uit [Vragen stellen aan data](/course/open/data-explore?lang=nl) als `df`, met de kolommen `day`, `temp`, `rain`, `wind`, `dir` en `sky`. Maak `per_sky`: de gemiddelde temperatuur, `temp`, voor elke soort lucht, `sky`. Vervang de `...`. De laatste regel is `per_sky`, zodat de check het ziet. De check probeert ook een andere maand.
        """),
        L.task("Maak `per_sky`: de gemiddelde temperatuur voor elke soort lucht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sky():
    box_sky = L.code_box("per-sky", lines=6)
    box_sky
    return (box_sky,)


@app.cell(hide_code=True)
def check_sky(box_sky):
    result_sky = L.check("per-sky", box_sky.value)
    result_sky
    return (result_sky,)


@app.cell(hide_code=True)
def hints_sky():
    L.hints(
        "per-sky",
        [
            "Stap 1 en 2 van het recept, zoals in het uitgewerkte voorbeeld, met andere kolommen.",
            "De groepen komen uit de kolom `sky`; het gemiddelde uit de kolom `temp`.",
            "`per_sky = df.groupby(___)[___].mean()`.",
        ],
        why="`df.groupby(\"sky\")` splitst de 30 dagen op hun lucht, `[\"temp\"]` neemt hun temperaturen, en `.mean()` geeft één gemiddelde per soort lucht: 7,5 graden voor `Sunny`, 10,79 voor `cloudy`, 9,94 voor `rainy` en 9,0 voor `sunny`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_summary():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het gemiddelde en waarop het rust

        Maak `summary(path, by, column)` af. Hij leest het CSV-bestand op `path` als `df`. Hij geeft voor elke groep van de kolom `by` twee getallen terug: het gemiddelde van de kolom `column`, en op hoeveel waarden dat gemiddelde rust. Samen een tabel met de kolommen `mean` en `count`. Vervang de `...`. De check roept `summary("data/penguins.csv", "species", "body_mass_g")` aan, en probeert ook andere bestanden en kolommen.
        """),
        L.task("Laat `summary(path, by, column)` het gemiddelde en de telling van `column` teruggeven voor elke groep van `by`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_summary():
    box_summary = L.code_box("summary", lines=7)
    box_summary
    return (box_summary,)


@app.cell(hide_code=True)
def check_summary(box_summary):
    result_summary = L.check("summary", box_summary.value)
    result_summary
    return (result_summary,)


@app.cell(hide_code=True)
def hints_summary():
    L.hints(
        "summary",
        [
            "Stap 2 van het recept: meerdere getallen per groep tegelijk.",
            "In `by` en `column` zitten de namen van twee kolommen: gebruik ze zonder aanhalingstekens, `by` in `groupby` en `column` tussen de vierkante haken. Dan `agg`, met een lijst van twee namen.",
            "`return df.groupby(___)[___].agg([___, ___])`.",
        ],
        why="`df.groupby(by)` splitst de rijen op de kolom waarvan de naam in `by` zit, `[column]` neemt de kolom die je samenvat, en `.agg([\"mean\", \"count\"])` geeft een tabel met het gemiddelde en de telling voor elke groep.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_one_day():
    mo.md(r"""
    ### Oefening 3. Een samenvatting per groep lezen
    """)
    return


@app.cell(hide_code=True)
def choice_one_day_cell():
    choice_one_day = L.choice(
        "one-day",
        question="In je `per_sky` van oefening 1 is de gemiddelde temperatuur 9,0 graden voor `sunny` en 7,5 graden voor `Sunny`. Wat kun je daaruit concluderen?",
        options={
            "colder": "Dagen met de lucht `Sunny` zijn kouder dan dagen met `sunny`",
            "right": "Niets over het weer: `Sunny` is één dag, dezelfde lucht als `sunny`, anders geschreven",
            "skipped": "Het gemiddelde van `Sunny` klopt niet, omdat pandas zijn ontbrekende temperaturen oversloeg",
        },
        correct="right",
        why={
            "colder": "Over hoeveel dagen gaat het gemiddelde van `Sunny`? Vraag om de telling ernaast, zoals in stap 3 van het recept. En wat was er vreemd aan `Sunny` in [Vragen stellen aan data](/course/open/data-explore?lang=nl)?",
            "skipped": "De ene dag met `Sunny` heeft een temperatuur: `count` en `size` zijn allebei 1. Over hoeveel dagen gaat dat gemiddelde, en wat zegt dat over een soort lucht?",
            "right": "`Sunny` is een groep van één dag: een gemiddelde van één waarde zegt niets over een soort lucht. Het is `sunny` met een hoofdletter, en hoort bij de andere 12 zonnige dagen. Tel voordat je vergelijkt, en verbeter zo'n spelling eerst.",
        },
    )
    choice_one_day
    return (choice_one_day,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Vragen stellen aan data](/course/open/data-explore?lang=nl) telde `value_counts` hoe vaak elke waarde voorkomt: een telling per groep, en `df.groupby("sky").size()` geeft dezelfde getallen, op volgorde van naam. `describe` vatte elke kolom samen; `groupby` vat elke groep samen. Later in de course vergelijk je zo weer groepen, bijvoorbeeld hoe goed een model het doet voor elke klasse, en ook daar zegt een kleine groep weinig. In de volgende les, *Twee tabellen samenvoegen: merge*, zet je twee tabellen naast elkaar op een kolom die ze delen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_sky, result_summary, choice_one_day):
    L.wrap_up(result_sky, result_summary, choice_one_day)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
