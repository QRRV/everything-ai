# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-data-validation. Route: course/route/lessen/systemen.yaml. Checks, the bike data, the gate of the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-data-validation", "nl")
    C = L.checks   # het schema, de dagen van de fietsdata, de poort van de demo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_warm():
    quick_warm = L.choice(
        "quick-warm",
        question="Vier dagen lang meet de thermometer van één station 3 graden te warm, en elke cel blijft gevuld. Welke score van de check zakt?",
        options={
            "complete": "Volledigheid, want een foute waarde telt als een ontbrekende",
            "spelling": "Consistentie, want het station schrijft de waarden op een nieuwe manier",
            "right": "Juistheid, want de waarden kloppen niet meer met de weerdienst",
            "time": "Actualiteit, want metingen die 3 graden ernaast zitten tellen als te oud om te gebruiken tot het station gerepareerd is",
        },
        correct="right",
        why={
            "right": "Juistheid vergelijkt elke waarde met een bron die je vertrouwt, binnen een tolerantie, en 3 graden valt buiten een tolerantie van 1. De cellen zijn gevuld, op de afgesproken manier geschreven, en nieuw. In deze les worden zulke checks regels die op elke batch draaien.",
            "complete": "Een foute waarde is nog steeds een waarde: de cel is gevuld. Welke dimensie vergelijkt de waarde met een bron die je vertrouwt? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
            "spelling": "Het station schrijft de waarden zoals eerst, alleen 3 graden te hoog. Welke dimensie vraagt of een waarde waar is? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
            "time": "De metingen komen op tijd binnen; alleen kloppen de waarden niet. Welke dimensie vraagt of een waarde waar is? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
        },
    )
    quick_warm
    return (quick_warm,)


@app.cell(hide_code=True)
def quick_check_fill():
    quick_fill = L.choice(
        "quick-fill",
        question="Een tabel heeft 30 lege cellen van de 600. Iemand vult de lege cellen met de mediaan, en meet daarna de volledigheid. Wat laat de score zien?",
        options={
            "zero": "0,0, want gevulde cellen tellen niet meer mee",
            "right": "1,0, en dat verbergt de 30 gaten die er eerst waren",
            "low": "0,95, het aandeel van de cellen dat voor het vullen een waarde had",
            "none": "Helemaal niets, want volledigheid is alleen te meten op een kolom waarin nog elk gat zit dat er ooit in zat",
        },
        correct="right",
        why={
            "right": "Na het vullen heeft elke cel een waarde, dus zegt de volledigheid 1,0 en zijn de 30 gaten uit beeld. Gemeten voor het vullen: 570 van de 600, 0,95. Een check hoort waar de data binnenkomt, voordat iets de data verandert.",
            "zero": "Een gevulde cel telt als gevuld, wat de waarde er ook in zette. Wat laat de score zien als elke cel een waarde heeft? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
            "low": "0,95 is wat de tabel had voor het vullen: 570 van de 600. Wat laat de score erna zien? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
            "none": "Volledigheid is op elke kolom te meten; de vraag is wanneer. Wat laat de score na het vullen zien? Zie [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl).",
        },
    )
    quick_fill
    return (quick_fill,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elke nacht komt er een nieuwe dag fietsdata binnen, en een model van de vraag leert van de nieuwste dagen. Niemand leest elke nacht een tabel van 24 rijen, dus gaat alles wat erin staat meteen de training in. Op 10 maart 2011 is de luchtvochtigheid in de tabel 0 in elk uur, op een dag met lichte regen: een kapotte sensor. In [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl) mat je hoe goed een tabel is. Hier wordt het meten een poort: regels die je één keer schrijft en op elke nieuwe batch laat draaien, voordat de batch bij de training komt. Een batch is de data die in één keer binnenkomt. Een batch die een regel breekt stopt bij de poort.

    **Een voorbeeld uit het dagelijks leven.** Een formulier op een website vraagt je geboortedatum. Typ 31 februari, en het formulier zegt het meteen, voordat er iets verstuurd is: een regel over één veld. Bestel 40 zakken meel waar mensen er 1 of 2 bestellen, en een goede winkel vraagt of je het zeker weet: een regel over de bestelling als geheel, want geen enkel veld apart breekt de regel.
    """)
    return


@app.cell(hide_code=True)
def predict_schema():
    guess_schema = L.guess(
        "schema",
        question="Op 17 augustus 2012 zegt atemp, de gevoelstemperatuur, 0,2424 in elk uur van de dag, terwijl temp van 0,62 naar 0,86 gaat. Het schema zegt dat atemp van 0 tot 1 loopt. Hoeveel van de 24 uren van atemp houdt het schema tegen, denk je?",
        options={
            "right": "Geen enkel, want elke waarde ligt tussen 0 en 1",
            "few": "Een paar: de uren waarin atemp ver onder temp ligt",
            "half": "De helft: de uren van de middag, als temp hoog is",
            "all": "Alle 24, want een waarde die nooit verandert valt buiten het bereik van de kolom",
        },
        reveal="**Geen enkel uur.** Elke 0,2424 ligt tussen 0 en 1, dus komt elke waarde apart erdoor. Het probleem zit in de kolom als geheel: een temperatuur die 24 uur niet beweegt, terwijl temp 0,24 beweegt. Een schema kijkt naar één waarde tegelijk; om dit te zien is een regel over veel waarden samen nodig. Zo'n regel is een expectation.",
    )
    guess_schema
    return (guess_schema,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een schema.** Een **schema** zegt per kolom wat mag: het type, een bereik, een set waarden, en of de kolom leeg mag zijn. Een schema kijkt naar één waarde tegelijk: een uur van 25 is fout, wat de andere rijen ook zeggen. In deze les is een schema een dict met een bereik per kolom, van low tot high, met beide grenzen erbij:

        ```python
        C.SCHEMA = {"hr": (0, 23), "weekday": (0, 6), "weathersit": (1, 4), "temp": (0.0, 1.0),
                    "atemp": (0.0, 1.0), "hum": (0.01, 1.0), "windspeed": (0.0, 1.0)}
        ```

        De data van Capital Bikeshare in Washington, D.C. (Fanaee-T, 2013) deelt temp, atemp, hum en windspeed door een vast getal, zodat elk van 0 tot 1 loopt. hum heeft een ondergrens van 0,01, want buiten is de lucht nooit helemaal droog.

        **Een expectation.** Een **expectation** is een regel over veel waarden samen: het aantal rijen van een dag, geen uur twee keer, een kolom die over de dag moet bewegen, een gemiddelde binnen een bereik. Geen enkele waarde apart breekt zo'n regel. Het woord komt van Great Expectations, een tool voor precies deze regels. Deze les gebruikt er drie: een dag heeft 20 tot en met 24 rijen, geen uur komt twee keer voor, en temp, atemp en hum bewegen elk minstens 0,01 over de dag.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **De poort.** **Datavalidatie** controleert elke nieuwe batch tegen het schema en de expectations, voordat de batch bij de training of het model komt. Als elke regel klopt gaat de batch door. Als een regel breekt wordt de batch als geheel tegengehouden: de training houdt de data van eerder, en een mens krijgt een bericht met wat er brak. Een mens beslist: de data kan kapot zijn, en wordt dan in een eigen stap hersteld, of de dag kan echt maar ongewoon zijn. Hoe dan ook deed de regel het werk: de regel stelde een vraag op het goede moment.

        **De tools.** Twee bekende tools schrijven dezelfde regels. Met Pandera ziet het schema van een pandas-tabel er zo uit:

        ```python
        import pandera.pandas as pa

        schema = pa.DataFrameSchema({
            "hr": pa.Column(int, pa.Check.in_range(0, 23)),
            "hum": pa.Column(float, pa.Check.in_range(0.01, 1.0)),
        })
        schema.validate(day, lazy=True)
        ```

        `in_range` laat beide grenzen toe, net als de dict hier, en `lazy=True` voert elke regel uit en meldt alle fouten samen. Met Great Expectations zien hetzelfde bereik en een regel over de rijen er zo uit:

        ```python
        import great_expectations as gx

        gx.expectations.ExpectColumnValuesToBeBetween(column="hum", min_value=0.01, max_value=1.0)
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=20, max_value=24)
        ```

        Great Expectations kan een regel laten slagen als de meeste waarden kloppen: met `mostly=0.95` mag 5 procent van de waarden de regel breken. Gelezen op 28 november 2026, in Pandera 0.36.0 en Great Expectations 1.25.0. De course installeert geen van beide: de dict en de twee functies van deze les doen hetzelfde werk in een paar regels.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **schema** | | per kolom wat mag: het type, een bereik, een set waarden, leeg of niet; kijkt naar één waarde tegelijk | `C.SCHEMA` |
        | **expectation** | | een regel over veel waarden samen, zoals het aantal rijen van een dag | `len(rows) >= 20` |
        | **datavalidatie** | | elke nieuwe batch tegen het schema en de expectations controleren voor het gebruik, en een batch tegenhouden die een regel breekt | |
        | **batch** | | de data die in één keer binnenkomt: hier één dag aan uren | |
        | `or` | "or" | `True` als minstens één van de twee kanten `True` is | `x < 0 or x > 1` |
        """),
        L.watch_out("Houd een batch als geheel tegen, in plaats van de rijen die een regel breken weg te gooien. Op 10 maart 2011 breekt elke rij het schema: wegdoen laat een dag van 0 uren over, en een dag van 22 uren met een paar rijen minder leert het model een dag die nooit bestond."),
        L.watch_out("Kies een bereik uit wat kan gebeuren, niet uit de data van vorige week. Een regel die elke nacht afgaat leert mensen de regel te negeren, en een bereik dat alles toelaat houdt niets tegen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een poort voor de data:

    1. Schrijf het schema: per kolom het type, het bereik of de toegestane waarden, en of de kolom leeg mag zijn.
    2. Schrijf de expectations: wat moet gelden voor de batch als geheel, zoals het aantal rijen, geen dubbele rijen, en kolommen die bewegen.
    3. Als er een batch binnenkomt controleer je elke waarde tegen het schema en de batch tegen elke expectation, en verzamel je wat breekt.
    4. Er breekt niets: laat de batch door naar de training.
    5. Er breekt iets: houd de batch tegen, train verder op de data van eerder, en vertel een mens wat er brak.
    6. De mens kijkt: kapotte data wordt in een eigen stap hersteld, en een echte maar ongewone dag gaat door of blijft apart, met een notitie waarom.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.MARCH_10` bevat de 22 rijen van 10 maart 2011, elk een dict. De cel neemt één rij en print per kolom van `C.SCHEMA` de ondergrens, de waarde en de bovengrens. `_low, _high = C.SCHEMA[_column]` geeft de twee grenzen van het bereik elk een eigen naam. Voer de cel uit.
    """)
    return


@app.cell
def example_row():
    _row = C.MARCH_10[8]
    print("hour", _row["hr"])
    for _column in C.SCHEMA:
        _low, _high = C.SCHEMA[_column]
        print(_column, _low, _row[_column], _high)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Lees elke regel van links naar rechts: ligt de waarde tussen de ondergrens en de bovengrens? Voor elke kolom wel, behalve voor hum: `0.0` ligt onder de ondergrens van `0.01`. Een weathersit van 3 is lichte regen, en in de regen is de lucht verre van droog: de sensor is kapot. Oefening 1 stelt dezelfde vraag aan elke waarde van elke rij.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_broken():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Buiten het bereik?

        `broken(rows, schema)` krijgt de rijen van een batch, elk een dict, en een schema, een dict met een paar `(low, high)` per kolom. De twee `for`-loops lopen langs elke `row` en elke `column` van het schema, en `low, high = schema[column]` geeft de twee grenzen elk een eigen naam. `found` verzamelt met `found.append((row["hr"], column))` een paar voor elke waarde die het schema breekt: het uur van de rij en de naam van de kolom. Vervang de `...` door de vraag of de waarde van `row` onder `column` onder `low` of boven `high` ligt. `a or b` is `True` als minstens één van de twee `True` is: `5 < 0 or 5 > 3` geeft `True`.

        De check roept `broken(C.MARCH_10, C.SCHEMA)` aan. Daarna een dag van juni 2012 zonder probleem, een schema van één kolom, en een paar eigen rijen met waarden net binnen en net buiten een bereik.
        """),
        L.task("Noem elke waarde buiten het bereik, met het uur en de kolom."),
    ])
    return


@app.cell(hide_code=True)
def exercise_broken():
    box_broken = L.code_box("broken", lines=9)
    box_broken
    return (box_broken,)


@app.cell(hide_code=True)
def check_broken(box_broken):
    result_broken = L.check("broken", box_broken.value)
    result_broken
    return (result_broken,)


@app.cell(hide_code=True)
def hints_broken():
    L.hints(
        "broken",
        [
            "`row[column]` is de waarde van deze rij onder deze kolom, en `low` en `high` zijn de twee grenzen van het bereik. Een waarde breekt het schema aan elke kant: onder `low`, of boven `high`. Welke twee vergelijkingen heb je nodig, en welk woord verbindt de twee zo dat één van de twee genoeg is?",
            "`0.0 < 0.01 or 0.0 > 1.0` geeft `True`: de eerste kant is `True`, en één is genoeg. `0.5 < 0.01 or 0.5 > 1.0` geeft `False`: de waarde ligt erbinnen.",
            "`            if row[column] < low or ___:`",
        ],
        why="`if row[column] < low or row[column] > high:` is `True` voor een waarde buiten het bereik, aan welke kant ook. Op 10 maart 2011 is dat hum in alle 22 rijen: `0.0`, onder de ondergrens van `0.01`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Beweegt een kolom?

        Nu een expectation, met NumPy. `stuck(hours, columns, least)` krijgt de uren van een dag als kolommen: een dict met een NumPy-array per kolom, zodat `hours["temp"]` de 24 waarden van temp bevat. `columns` noemt de namen om te controleren, en `least` is de kleinste spreiding die als bewegen telt. De spreiding van een kolom is de grootste waarde min de kleinste, en een NumPy-array heeft voor elk een methode, zoals `values.max()` voor de grootste. De loop loopt langs de namen, `values` bevat de array van één `column`, en `found.append(column)` verzamelt elke kolom die niet genoeg beweegt. Vervang de `...` door de vraag of de spreiding van `values` kleiner is dan `least`.

        `C.AUGUST_17` bevat temp, atemp, hum en windspeed van de 24 uren van 17 augustus 2012. De check roept `stuck(C.AUGUST_17, ["temp", "atemp", "hum"], 0.01)` aan. Daarna 10 maart 2011, een grotere `least`, andere kolommen, en kleine eigen arrays.
        """),
        L.task("Noem de kolommen waarvan de spreiding over de dag kleiner is dan least."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stuck():
    box_stuck = L.code_box("stuck", lines=8)
    box_stuck
    return (box_stuck,)


@app.cell(hide_code=True)
def check_stuck(box_stuck):
    result_stuck = L.check("stuck", box_stuck.value)
    result_stuck
    return (result_stuck,)


@app.cell(hide_code=True)
def hints_stuck():
    L.hints(
        "stuck",
        [
            "De spreiding is de grootste waarde min de kleinste. `values.max()` geeft de grootste; de vraag na `if` haalt daar de kleinste af, en vraagt of wat overblijft kleiner is dan `least`. Welke methode geeft de kleinste waarde van een array?",
            "`np.array([0.3, 0.8, 0.5]).max()` geeft `0.8`, en `.min()` op dezelfde array geeft `0.3`: een spreiding van 0,5. Een array van 24 keer 0,2424 heeft een spreiding van 0.",
            "`        if values.max() - ___ < least:`",
        ],
        why="`if values.max() - values.min() < least:` is `True` voor een kolom met een spreiding kleiner dan `least`. Op 17 augustus 2012 heeft atemp een spreiding van 0: 0,2424 in elk uur, terwijl temp 0,24 beweegt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rules():
    mo.md(r"""
    ### Oefening 3. Schema of expectation?

    Acht regels voor de fietsdata. Zet elke regel onder schema, een regel die één waarde op zichzelf kan breken, of onder expectation, een regel over veel waarden samen.
    """)
    return


@app.cell(hide_code=True)
def sort_rules_cell():
    sort_rules = L.sort("rules")
    sort_rules
    return (sort_rules,)


@app.cell(hide_code=True)
def check_rules(sort_rules):
    result_rules = L.check("rules", sort_rules)
    result_rules
    return (result_rules,)


@app.cell(hide_code=True)
def hints_rules():
    L.hints(
        "rules",
        [
            "Vraag bij elke regel: kan ik de regel controleren door naar één waarde te kijken, en verder niets?",
            "Aan elke kant horen vier regels.",
            "Schema: hr van 0 tot en met 23, weathersit, \\_\\_\\_ en \\_\\_\\_. Expectation: 20 tot 24 rijen, geen uur twee keer, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Schema:** hr van 0 tot en met 23, weathersit is 1, 2, 3 of 4, hum boven 0, en geen lege temp. **Expectation:** 20 tot 24 rijen, geen uur twee keer, temp beweegt, en ongeveer evenveel fietsen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_storm():
    mo.md(r"""
    ### Oefening 4. Een korte dag

    In de nacht na 29 oktober 2012 heeft de batch van 29 oktober 1 rij, en de regel van 20 tot 24 rijen houdt de batch tegen. Volgens het nieuws van de dag stopte Capital Bikeshare om 1 uur 's nachts, voor orkaan Sandy, en bleef het systeem dicht tot de volgende middag.
    """)
    return


@app.cell(hide_code=True)
def choice_storm_cell():
    choice_storm = L.choice(
        "storm",
        question="Wat doe je met de batch?",
        options={
            "rule": "De regel over rijen weghalen, want de regel hield een dag tegen die echt was",
            "lower": "De regel veranderen in minstens 1 rij, zodat zo'n dag de volgende keer doorgaat",
            "right": "De batch doorlaten als echte dag, of als aparte dag buiten de training houden, en opschrijven waarom",
            "fill": "De 23 ontbrekende uren invullen met dezelfde uren van de maandag een week eerder, en gewoon op de dag trainen",
        },
        correct="right",
        why={
            "right": "De regel deed het werk: de regel vroeg een mens om te kijken, en de mens weet waarom de dag kort is. De ene rij is echt, want het systeem was dicht. Een model kan van zo'n gesloten dag leren of niet: een eigen keuze, opgeschreven, en de regel blijft voor de volgende nacht.",
            "rule": "Zonder de regel komt ook een dag erdoor waarop de feed na 3 uur stopt, zonder dat iemand het ziet. Deed de regel iets fout, of stelde de regel een goede vraag?",
            "lower": "Met minstens 1 rij komt ook een feed met 3 van de 24 uren erdoor. Is de regel fout, of verdient deze dag een blik?",
            "fill": "Verzonnen uren maken van een echte dag een valse: het model zou leren dat er fietsen verhuurd werden terwijl het systeem dicht was. Wat zegt de ene rij over de dag?",
        },
    )
    choice_storm
    return (choice_storm,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Twee jaar aan batches

    Elke dag van 2011 en 2012 als de batch van een nacht, gecontroleerd met `C.SCHEMA` en de drie expectations: 20 tot 24 rijen, geen uur twee keer, en temp, atemp en hum bewegen elk minstens 0,01. De stroken bovenaan tonen alle 731 dagen, met een markering bij elke dag die de poort tegenhoudt. De schuif stopt bij elke tegengehouden dag, en bij vier dagen zonder probleem. Kies een dag.
    """)
    return


@app.cell(hide_code=True)
def demo_day():
    L.explore("day", "dag van 2011 en 2012", lambda _v: C.frame(_v, L.lang), start=69, values=C.DEMO_DAYS, role="",
              task="Ga langs de tegengehouden dagen: welke regel breekt, en is de data kapot, of de dag ongewoon?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De poort houdt 12 van de 731 dagen tegen. Acht zijn kort: 18 januari en 22 februari 2011, waarop de ochtend ontbreekt zonder reden in de data; 26 en 27 januari 2011, rond zware sneeuw; 27 en 28 augustus 2011, toen Capital Bikeshare dichtging voor orkaan Irene; en 29 en 30 oktober 2012, toen het systeem stopte voor orkaan Sandy. Het schema houdt er één tegen: 10 maart 2011, met een luchtvochtigheid van 0 in elk uur. En de expectation dat temp, atemp en hum bewegen houdt er nog drie tegen: 6 september 2011, en 17 augustus en 29 december 2012. Zonder expectations had het schema alleen 10 maart gevangen: de atemp van 17 augustus komt 24 keer door elk bereik. Op een paar van de 12 dagen is de data kapot, op de meeste was de dag ongewoon, en over 6 september kan niemand iets zeggen zonder te kijken. Dat kijken is waar de poort voor is.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Is de data goed? Vier manieren om te kijken](/course/open/sys-data-quality?lang=nl) mat je een tabel op vier dimensies; hier werd het meten regels die op elke batch draaien, en een poort die tegenhoudt wat de regels breekt. Validiteit en uniciteit, twee dimensies van DAMA UK, kwamen terug als regels: een waarde binnen het bereik, en geen uur twee keer. De volgende les is *Waarom trainen miljoenen kost*: daar gaat het domein van de data naar de machines, en naar wat het kost om een groot model op veel computers te trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_storm, result_broken, result_rules, result_stuck):
    L.wrap_up(result_broken, result_stuck, result_rules, choice_storm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
