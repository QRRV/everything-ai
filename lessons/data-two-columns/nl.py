# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-two-columns. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-two-columns", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) en [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scatter():
    quick_scatter = L.choice(
        "quick-scatter",
        question="Wat tekent `plt.scatter(x, y)`?",
        options={
            "line": "Een lijn door de punten, op hun volgorde",
            "right": "Een los punt voor elk paar van een x en een y",
            "bars": "Een staaf voor elke x, zo hoog als zijn y",
        },
        correct="right",
        why={
            "line": "Dat is `plt.plot`. Wat laat een scatter plot weg? Zie [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "bars": "Dat is `plt.bar`. Wat tekent een scatter plot voor elk paar? Zie [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "right": "Eén los punt per paar: de eerste x met de eerste y, enzovoort.",
        },
    )
    quick_scatter
    return (quick_scatter,)


@app.cell(hide_code=True)
def quick_check_biscoe():
    quick_biscoe = L.choice(
        "quick-biscoe",
        question="Gegroepeerd per eiland wegen de pinguïns van Biscoe gemiddeld het meest. Waarom?",
        options={
            "food": "Pinguïns op Biscoe vinden meer eten",
            "right": "Op Biscoe wonen de Gentoos, de zwaarste soort",
            "many": "Biscoe heeft de meeste pinguïns",
        },
        correct="right",
        why={
            "food": "Over eten zegt de tabel niets. Hoeveel woog Biscoe zonder zijn Gentoos? Zie het speelblok in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "many": "Een gemiddelde groeit niet met het aantal waarden. Hoeveel woog Biscoe zonder zijn Gentoos? Zie het speelblok in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "right": "Zonder zijn Gentoos weegt Biscoe wat de andere eilanden wegen: wie er woont, maakt zijn gemiddelde hoger.",
        },
    )
    quick_biscoe
    return (quick_biscoe,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl) kwamen het weer en de ijsjes van het kraampje op één rij per dag: op de warmste dag verkocht het kraampje het meest. Bij twee kolommen met getallen laat een scatter plot in één oogopslag zien of ze samen bewegen. Maar zo'n patroon lees je snel als oorzaak en gevolg, terwijl vaak een derde kolom ze allebei stuurt.

    **Een voorbeeld uit het dagelijks leven.** Op een basisschool lezen de kinderen met de grotere voeten beter. Helpen grote voeten bij het lezen? Nee: de oudere kinderen hebben grotere voeten en lezen al langer. De leeftijd stuurt allebei.
    """)
    return


@app.cell(hide_code=True)
def predict_bills():
    guess_bills = L.guess(
        "bills",
        question="De Palmer-pinguïns uit [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl): 342 ervan hebben een snavellengte en een snavelhoogte. Hebben pinguïns met een langere snavel meestal een hogere snavel, of een lagere?",
        options={
            "deeper": "Een hogere snavel",
            "shallower": "Een lagere snavel",
            "neither": "Geen van beide: er is geen richting",
        },
        reveal="**Een lagere.** Splits de pinguïns op de mediaan van de snavellengte: de helft met de langere snavels heeft gemiddeld 1,41 millimeter lagere snavels. Binnen elke soort is het precies andersom, zoals het blok hieronder laat zien.",
    )
    guess_bills
    return (guess_bills,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Er is een verband tussen twee kolommen als ze samen bewegen; een scatter plot laat dat zien als de helling van de wolk stippen. Daarmee is de ene nog geen oorzaak van de andere: een derde kolom kan ze allebei sturen, en kan de richting zelfs omdraaien.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **verband** | twee kolommen bewegen samen: waar de ene hoger is, is de andere meestal hoger (positief) of lager (negatief) | `plt.scatter(df[x], df[y])` |
        | **oorzaak**, **causaliteit** | X is een oorzaak van Y als het veranderen van X, terwijl al het andere gelijk blijft, Y verandert; causaliteit is die relatie | |
        | **confounder** | een derde kolom die twee andere kolommen allebei stuurt, zodat die samen bewegen, ook als geen van beide de ander veroorzaakt | |
        | **paradox van Simpson** | een richting die binnen elke groep geldt, maar omdraait als je de groepen samenvoegt | |
        """),
        mo.Html(C.shapes_picture({"up": "Omhoog", "down": "Omlaag", "none": "Geen", "curve": "Een bocht",
                                  "flipper": "vleugellengte (mm)", "mass": "lichaamsgewicht (g)", "day": "dag", "temp": "temperatuur (°C)", "wind": "wind (km/u)",
                                  "alt_1": "Twee scatter plots. Omhoog: de vleugellengte en het lichaamsgewicht van de pinguïns, een wolk die naar rechts stijgt. Omlaag: de dagen van de maand van het station en de temperatuur, van zo'n 11 graden aan het begin tot 6 aan het eind.",
                                  "alt_2": "Twee scatter plots. Geen: de temperatuur en de wind van de maand van het station, verspreid zonder richting. Een bocht: de dagen van een andere maand van het station en de temperatuur, die stijgt van 11 naar 16,5 graden en weer zakt naar 11,5."})),
        mo.md(r"""
        Vier scatter plots van de eigen tabellen van de course; in twee ervan is de dag ook een kolom. Omhoog: hoe langer de vleugels, hoe zwaarder de pinguïn, een positief verband. Omlaag: de maand van het station koelt af, een negatief verband. Geen: in die maand gaan de wind en de temperatuur geen kant samen op. Een bocht: een andere maand van het station warmt op en koelt weer af, en één richting zou dat verbergen.

        In het blok hieronder vat een pijl elke wolk samen, een eenvoudige samenvatting van de course zelf: splits de pinguïns op de mediaan van de snavellengte, en hij loopt van de gemiddelde lengte en hoogte van de helft met de kortere snavels naar die van de helft met de langere snavels. Het blok begint met alle drie de soorten, elk in zijn kleur en vorm en met een eigen pijl, een rondje voor een Adelie, een driehoekje voor een Chinstrap en een vierkantje voor een Gentoo, en de grijze pijl van alle pinguïns.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_species_cell():
    def _frame(step):
        step = int(step)
        n = C.step_numbers(step)
        if step == 0:
            _say = f"Alle {n['n']} pinguïns: de pijl wijst omlaag. De helft met de langere snavels heeft gemiddeld {L.number(-n['gap'])} millimeter lagere snavels."
            _alt = f"De snavelhoogte tegen de snavellengte van alle {n['n']} pinguïns, in grijs, met een pijl van het midden van de helft met de kortere snavels naar het midden van de helft met de langere snavels. Hij wijst omlaag: {L.number(-n['gap'])} millimeter lager."
        elif step < 4:
            _say = f"De {n['n']} {n['species']}s: hun pijl wijst omhoog. De helft met de langere snavels heeft gemiddeld {L.number(n['gap'])} millimeter hogere snavels."
            _alt = f"De snavelhoogte tegen de snavellengte van de pinguïns, met de {n['n']} {n['species']}s in hun kleur en vorm en de andere licht. De pijl van de {n['species']}s wijst omhoog: {L.number(n['gap'])} millimeter hoger."
        else:
            _g = n["gaps"]
            _say = f"Alle drie de soorten: elke pijl wijst omhoog, met {L.number(_g['Adelie'])}, {L.number(_g['Chinstrap'])} en {L.number(_g['Gentoo'])} millimeter. De grijze pijl van alle pinguïns wijst omlaag."
            _alt = f"De snavelhoogte tegen de snavellengte van de pinguïns, elke soort in zijn kleur en vorm, elk met een pijl die omhoog wijst: {L.number(_g['Adelie'])}, {L.number(_g['Chinstrap'])} en {L.number(_g['Gentoo'])} millimeter hoger. Een grijze pijl voor alle pinguïns samen wijst omlaag: {L.number(-n['gap'])} millimeter lager."
        return (C.species_picture(step, {"title": "Snavelhoogte tegen snavellengte", "x": "snavellengte (mm)", "y": "snavelhoogte (mm)", "alt": _alt}), _say)

    explore_species = L.explore("species", "stap", _frame, start=4, values=C.STEPS, role="",
                                task="Loop de soorten langs. Welke kant wijst de pijl van elke soort op, en welke kant de pijl van alle pinguïns?")
    explore_species
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Binnen elke soort geldt: hoe langer de snavel, hoe hoger; alle pinguïns samen: hoe langer, hoe lager. Dat is de paradox van Simpson, en de soort is de confounder: die bepaalt zowel de lengte als de hoogte van een snavel. De Adelies, kort en hoog, liggen linksboven en de Gentoos, lang en laag, rechtsonder, dus samen laten ze de hele wolk omlaag hellen.
        """),
        L.watch_out("Een scatter plot laat zien dat twee kolommen samen bewegen, niet waarom. Op zichzelf kan hij geen oorzaak laten zien; een experiment wel: verander één kolom, houd al het andere gelijk, en kijk of de andere verandert."),
        mo.md(r"""
        ### Recept

        Zo bekijk je twee kolommen samen:

        1. Teken ze als scatter plot, de ene langs de onderkant en de andere langs de zijkant: `plt.scatter(df[x], df[y])`, met een titel en de aslabels.
        2. Lees de wolk: gaat hij omhoog naar rechts, omlaag, geen kant op, of maakt hij een bocht?
        3. Bedenk een derde kolom die ze allebei kan sturen, zoals de soort, en kijk per groep: `df.groupby("species")[[x, y]].mean()` zegt waar elke groep ligt, en met een masker teken je één groep apart.
        4. Schrijf wat je ziet als "hoort bij", niet als "veroorzaakt": de grafiek alleen kan geen oorzaak laten zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Weer de Palmer-pinguïns, in `data/penguins.csv`: 344 pinguïns van drie soorten. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de twee cellen uit.
    """)
    return


@app.cell
def example_two_columns():
    import matplotlib.pyplot as plt
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    # 1. alle pinguïns: de snavellengte langs de onderkant, de snavelhoogte langs de zijkant
    plt.scatter(_df["bill_length_mm"], _df["bill_depth_mm"])
    plt.title("Alle pinguïns")
    plt.xlabel("snavellengte (mm)")
    plt.ylabel("snavelhoogte (mm)")
    plt.show()
    return pd, plt


@app.cell
def example_per_species(pd, plt):
    _df = pd.read_csv("data/penguins.csv")
    # 3. een derde kolom, de soort: waar elke soort ligt, en dan de Adelies apart
    print(_df.groupby("species")[["bill_length_mm", "bill_depth_mm"]].mean())
    _adelie = _df[_df["species"] == "Adelie"]
    plt.scatter(_adelie["bill_length_mm"], _adelie["bill_depth_mm"])
    plt.title("Alleen de Adelies")
    plt.xlabel("snavellengte (mm)")
    plt.ylabel("snavelhoogte (mm)")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Stap 2: over alle pinguïns helt de wolk een beetje omlaag. Volgens de gemiddelden zijn de Adelies kort en hoog, 38,8 en 18,3 millimeter, de Chinstraps lang en hoog, 48,8 en 18,4, en de Gentoos lang en laag, 47,5 en 15,0; apart gaan de Adelies omhoog naar rechts. Stap 4: dat over alle pinguïns een langere snavel bij een lagere hoort, komt door de soort, niet door de lengte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chinstraps():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De Chinstraps apart

        Het vak leest de pinguïns als `df`. Teken de snavelhoogte, `bill_depth_mm`, tegen de snavellengte, `bill_length_mm`, van alleen de Chinstraps, met een titel en een label bij beide assen. Vervang de `...`. De check kijkt naar de punten van je grafiek, en de grafiek verschijnt onder zijn melding.
        """),
        L.task("Houd de Chinstraps, en teken hun snavelhoogte tegen hun snavellengte, met een titel en twee aslabels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chinstraps():
    box_chinstraps = L.code_box("chinstraps", lines=9)
    box_chinstraps
    return (box_chinstraps,)


@app.cell(hide_code=True)
def check_chinstraps(box_chinstraps):
    result_chinstraps = L.check("chinstraps", box_chinstraps.value)
    result_chinstraps
    return (result_chinstraps,)


@app.cell(hide_code=True)
def hints_chinstraps():
    L.hints(
        "chinstraps",
        [
            "Stap 1 en 3 van het recept, zoals de Adelies in het uitgewerkte voorbeeld.",
            "Een masker houdt de Chinstraps: `df[df[\"species\"] == \"Chinstrap\"]`. Dan `plt.scatter` met hun snavellengte eerst, en `plt.title`, `plt.xlabel` en `plt.ylabel`.",
            "`chinstrap = df[df[\"species\"] == ___]`, en dan `plt.scatter(chinstrap[___], chinstrap[___])`.",
        ],
        why="`df[\"species\"] == \"Chinstrap\"` is `True` voor de 68 Chinstraps, en tussen de vierkante haken van `df` houdt het hun rijen. `plt.scatter` zet voor elk een punt op zijn snavellengte en snavelhoogte; de titel en de labels zeggen wat de lezer ziet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rescues():
    mo.md(r"""
    ### Oefening 2. IJsjes en reddingen
    """)
    return


@app.cell(hide_code=True)
def choice_rescues_cell():
    choice_rescues = L.choice(
        "rescues",
        question="Een hele zomer lang redden de strandwachten meer zwemmers op de dagen dat het kraampje op het strand meer ijsjes verkoopt. Wat is de waarschijnlijkste reden?",
        options={
            "ice": "IJs maakt zwemmen gevaarlijker",
            "right": "Warme dagen: dan kopen meer mensen ijs, en zwemmen er meer",
            "chance": "Toeval: de twee hebben niets met elkaar te maken",
        },
        correct="right",
        why={
            "ice": "Zou de zee veiliger worden als het kraampje dichtging? Samen bewegen zegt niet wat wat stuurt. Wat is er verder anders op de dagen met veel ijsjes?",
            "chance": "Een patroon over een hele zomer krijg je niet snel door toeval, en die dagen hebben iets gemeen. Wat zou ze allebei kunnen sturen?",
            "right": "De temperatuur is een confounder: op warme dagen kopen meer mensen ijs en zwemmen er meer, dus komen er ook meer in de problemen. Het kraampje sluiten maakt de zee niet veiliger.",
        },
    )
    choice_rescues
    return (choice_rescues,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Waar elke soort ligt

        Over alle pinguïns samen hoort zwaarder bij lagere snavels; binnen elke soort bij hogere. Het vak leest de pinguïns als `df`. Maak `per_species`: het gemiddelde lichaamsgewicht, `body_mass_g`, en de gemiddelde snavelhoogte, `bill_depth_mm`, van elke soort, in één tabel. Vervang de `...`. De laatste regel is `per_species`, zodat de check het ziet.
        """),
        L.task("Maak `per_species`: het gemiddelde lichaamsgewicht en de gemiddelde snavelhoogte van elke soort."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("per-species", lines=6)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("per-species", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "per-species",
        [
            "Stap 3 van het recept, zoals de gemiddelden per soort in het uitgewerkte voorbeeld, met andere kolommen.",
            "Twee kolommen gaan in een lijst: twee paar vierkante haken na `groupby(...)`.",
            "`per_species = df.groupby(___)[[___, ___]].mean()`.",
        ],
        why="`df.groupby(\"species\")` splitst de pinguïns per soort, `[[\"body_mass_g\", \"bill_depth_mm\"]]` neemt de twee kolommen, en `.mean()` geeft een tabel met het gemiddelde van allebei voor elke soort: de Gentoos zijn het zwaarst en hebben de laagste snavels.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je een scatter plot; nu lees je wat hij zegt, en wat niet. In [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl) maakten de Gentoos Biscoe het zwaarste eiland: ook daar was de soort een confounder. In [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl) is warm weer een waarschijnlijke oorzaak van de verkoop, maar de tabel alleen kan dat niet laten zien. Later in de course leert een model van hoe de kolommen van zijn data samen bewegen: verbanden, geen oorzaken. In de volgende les, *Waar komt deze data vandaan?*, vraag je wie een dataset maakte en wat je ermee mag.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chinstraps, choice_rescues, result_means):
    L.wrap_up(result_chinstraps, choice_rescues, result_means)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
