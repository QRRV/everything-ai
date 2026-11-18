# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-guidance-ethics-deda. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-guidance-ethics-deda", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hierarchy():
    quick_hierarchy = L.choice(
        "quick-hierarchy",
        question="Wat komt er in een waardenhiërarchie direct onder een waarde?",
        options={
            "requirement": "Een eis met een getal dat je kunt testen",
            "right": "Normen die zeggen wat ze hier vraagt",
            "code": "De code die het team schrijft",
        },
        correct="right",
        why={
            "requirement": "Eisen komen onderaan, onder de normen. Zie *Het idee* in [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl).",
            "right": "Normen zeggen wat de waarde van dit systeem vraagt; de eisen daaronder maken de normen controleerbaar.",
            "code": "Code hoort niet bij de hiërarchie: de hiërarchie zegt wat de code moet doen. Zie *Het idee* in [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl).",
        },
    )
    quick_hierarchy
    return (quick_hierarchy,)


@app.cell(hide_code=True)
def quick_check_requirement():
    quick_requirement = L.choice(
        "quick-requirement",
        question="Welke hiervan is een eis die een engineer kan controleren?",
        options={
            "right": "Een cv wordt na hoogstens 30 dagen verwijderd",
            "respect": "De tool behandelt elke gebruiker met respect en zorg",
            "privacy": "De privacy van sollicitanten is beschermd",
        },
        correct="right",
        why={
            "respect": "Respect is een waarde, en mensen met zorg behandelen is hoogstens een norm: hoe zou je dat testen? Zie *Het idee* in [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl).",
            "right": "Er staat een getal in, dus een test kan slagen of zakken: een eis.",
            "privacy": "Privacy is de waarde zelf. Wat zou je meten? Zie *Het idee* in [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl).",
        },
    )
    quick_requirement
    return (quick_requirement,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel gemeenten, scholen en overheidsinstellingen willen data en AI gebruiken, en weinig hebben een ethicus in dienst. Twee Nederlandse methoden geven een team een manier om het samen te doordenken, in een middag, met de betrokkenen. Beide worden gebruikt door Nederlandse gemeenten en overheidsinstellingen. Deze les loopt ze allebei door.

    **Een voorbeeld uit het dagelijks leven.** Voordat een gezin een hond neemt, gaan ze aan tafel en praten ze het door: wie laat hem uit, wat als iemand allergisch is, wat kost het? De vraag is niet of honden goed zijn, maar hoe een hond in dit gezin zou passen.
    """)
    return


@app.cell(hide_code=True)
def predict_table():
    guess_table = L.guess(
        "table",
        question="Een Nederlandse gemeente wil een camera op een plein die de politie waarschuwt als er een vechtpartij begint. Ze houdt er een workshop begeleidingsethiek over. Wie zitten er aan tafel, denk je?",
        options={
            "ethicists": "Vooral ethici van de dichtstbijzijnde universiteit",
            "lawyers": "De eigen juristen en ambtenaren van de gemeente",
            "right": "Alle betrokkenen, ook bewoners",
        },
        reveal="**Alle betrokkenen**: bewoners, agenten, winkeliers, de makers van de camera en de gemeente. De methode leunt meer op de ervaring van de mensen die met de technologie gaan leven dan op experts van buiten.",
    )
    guess_table
    return (guess_table,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Beide methoden zijn voor een team dat aan een concreet project begint, niet voor filosofen.

        **Begeleidingsethiek**, van de filosoof Peter-Paul Verbeek en ECP, een Nederlands platform voor de informatiesamenleving, vraagt niet *of* een technologie toegestaan moet worden, maar *hoe* ze goed gebruikt kan worden. Een workshop gaat in drie stappen, die het plaatje laat zien. Eerst de casus: wat de technologie is en waar ze gebruikt gaat worden. Dan de dialoog: wie de actoren zijn, welke goede en slechte effecten ze voor elk van hen heeft, en welke waarden er op het spel staan. Als laatste de handelingsopties, op drie plekken: in de technologie zelf, in haar omgeving zoals regels en de organisatie, en bij de mensen die haar gebruiken.

        **DEDA**, De Ethische Data Assistent (in het Engels de *Data Ethics Decision Aid*), komt van de Utrecht Data School en is gemaakt samen met de gemeente Utrecht. Het is een reeks vragen in thema's, zoals waar de data vandaan komen, het algoritme, verantwoordelijkheid en communicatie. Een team loopt ze samen door en schrijft de antwoorden op, zodat het later kan laten zien waarom het besloot wat het besloot.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **begeleidingsethiek** | een Nederlandse aanpak waarin gebruikers, makers en andere betrokkenen samen een concrete technologie bespreken, om bij te sturen hoe ze gebruikt wordt in plaats van er ja of nee tegen te zeggen |
        | **DEDA** | De Ethische Data Assistent van de Utrecht Data School: vragen in thema's die een team, vaak bij een overheid, samen doorloopt voor een data- of algoritmeproject |
        """),
        L.watch_out("Een workshop is geen controle. Beide methoden helpen een team goede vragen te stellen en zijn keuzes op te schrijven; of die keuzes worden uitgevoerd, vraagt nog steeds om eisen die iemand test, zoals in [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl)."),
        mo.Html(C.picture({"title": "De drie stappen van een workshop begeleidingsethiek",
                           "steps": ["De casus", "De dialoog", "Handelingsopties"],
                           "texts": ["Wat is de technologie, en waar wordt ze gebruikt?",
                                     "Wie zijn de actoren? Welke goede en slechte effecten heeft ze voor elk van hen? Welke waarden staan er op het spel?"],
                           "places": ["in de technologie", "in de omgeving: regels, organisatie", "bij de mensen die haar gebruiken"],
                           "alt": "Drie stappen naast elkaar. 1, de casus: wat is de technologie, en waar wordt ze gebruikt? 2, de dialoog: wie zijn de actoren, welke goede en slechte effecten heeft ze voor elk van hen, welke waarden staan er op het spel? 3, handelingsopties: in de technologie, in de omgeving, en bij de mensen die haar gebruiken."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo doordenk je als team een data- of AI-project:

    1. Beschrijf de casus: de technologie en de plek waar ze gebruikt wordt.
    2. Nodig de actoren uit: de mensen die haar maken, gebruiken en ermee leven.
    3. Loop de effecten en de waarden door, of de thema's van DEDA, één voor één.
    4. Schrijf elk antwoord op, en markeer de vragen die nog open staan.
    5. Kies handelingsopties in de technologie, de omgeving en bij de mensen, en spreek af wie elke optie uitvoert.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een klein team, verzonnen door de course, begint aan twee thema's van DEDA. `questions` zegt hoeveel vragen elk thema voor dit project heeft, `answered` hoeveel het team er beantwoord heeft. Het aandeel beantwoord is het ene gedeeld door het andere. Voer de cel uit.
    """)
    return


@app.cell
def example_two_themes():
    _questions = {"source": 10, "algorithm": 4}
    _answered = {"source": 7, "algorithm": 1}

    for _theme in _questions:
        print(_theme, _answered[_theme] / _questions[_theme])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het team heeft 0,7 van zijn vragen over de herkomst van de data beantwoord, en maar 0,25 van die over het algoritme: stap 4 van het recept laat zien waar het verder moet. Oefening 1 doet hetzelfde voor een team met vier thema's.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_done():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Waar staat het team?

        Een ander team, ook verzonnen, is verder. `questions` bevat hoeveel vragen elk thema heeft, `answered` hoeveel het team er beantwoord heeft, en de loop gaat met `theme` langs de thema's. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld, maar zet elk aandeel in `done` in plaats van het te printen. De laatste regel is `done`, zodat de check het ziet. De check probeert ook andere antwoorden.
        """),
        L.task("Zet in `done`, onder elk thema, het aandeel van zijn vragen dat het team beantwoord heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_done():
    box_done = L.code_box("done", lines=7)
    box_done
    return (box_done,)


@app.cell(hide_code=True)
def check_done(box_done):
    result_done = L.check("done", box_done.value)
    result_done
    return (result_done,)


@app.cell(hide_code=True)
def hints_done():
    L.hints(
        "done",
        [
            "De deling van het uitgewerkte voorbeeld, met het antwoord bewaard onder het thema.",
            "Welk van de twee getallen staat boven de deelstreep? En hoe zet je een waarde onder een key in een dictionary?",
            "`done[theme] = ___[theme] / ___[theme]`",
        ],
        why="`done[theme] = answered[theme] / questions[theme]` geeft `{'source': 1.0, 'algorithm': 0.5, 'responsibility': 0.25, 'communication': 1.0}`: maar een kwart van de vragen over verantwoordelijkheid is beantwoord.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_places():
    mo.md(r"""
    ### Oefening 2. Waar werkt het?

    De workshop over de camera op het plein leverde zes handelingsopties op. Zet elke optie bij de plek waar ze werkt.
    """)
    return


@app.cell(hide_code=True)
def sort_places_cell():
    sort_places = L.sort("places")
    sort_places
    return (sort_places,)


@app.cell(hide_code=True)
def check_places(sort_places):
    result_places = L.check("places", sort_places)
    result_places
    return (result_places,)


@app.cell(hide_code=True)
def hints_places():
    L.hints(
        "places",
        [
            "Vraag bij elke optie: wat verandert er, de camera, de regels en verslagen eromheen, of wat een mens weet?",
            "Twee opties veranderen de camera zelf. Twee andere veranderen niets aan de camera of de regels, alleen aan wat mensen weten.",
            "In de technologie: vervaagd en \\_\\_\\_. In de omgeving: een regel en \\_\\_\\_. Bij de mensen: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**In de technologie:** vervaagd en alleen het plein. **In de omgeving:** een regel en een verslag. **Bij de mensen die het gebruiken:** training en terugkoppeling.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_how():
    mo.md(r"""
    ### Oefening 3. Niet of, maar hoe
    """)
    return


@app.cell(hide_code=True)
def choice_how_cell():
    choice_how = L.choice(
        "how",
        question="Een ethische commissie moet ja of nee zeggen tegen de camera op het plein. Welke vraag stelt begeleidingsethiek in plaats daarvan?",
        options={
            "allowed": "Of deze technologie überhaupt toegestaan moet worden, ja of nee",
            "law": "Aan welke wet de technologie moet voldoen",
            "right": "Hoe deze technologie hier en nu goed gebruikt kan worden",
        },
        correct="right",
        why={
            "allowed": "Dat is de ja-of-neevraag van de commissie. Begeleidingsethiek gaat uit van de technologie die eraan komt, en vraagt iets anders. Zie *Het idee*.",
            "law": "De wet doet ertoe, maar begeleidingsethiek vraagt naar waarden en gebruik, niet alleen naar wat mag van de wet. Zie *Het idee*.",
            "right": "Begeleidingsethiek vraagt hoe de technologie goed gebruikt kan worden: ze zoekt opties in de technologie, de omgeving en bij de mensen, in plaats van één oordeel.",
        },
    )
    choice_how
    return (choice_how,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ontwerpen met waarden](/course/open/rai-design-with-values?lang=nl) ging je van een waarde naar eisen voor één systeem. Begeleidingsethiek en DEDA zijn manieren om die keuzes samen met de betrokkenen te maken en ze op te schrijven. In de volgende les, *Wie is verantwoordelijk?*, vraag je wie er voor een systeem instaat als het toch misgaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_how, result_done, result_places):
    L.wrap_up(result_done, result_places, choice_how)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
