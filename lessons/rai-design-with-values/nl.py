# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-design-with-values. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-design-with-values", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Principes en richtlijnen](/course/open/rai-principles?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_trustworthy():
    quick_trustworthy = L.choice(
        "quick-trustworthy",
        question="De Europese expertgroep beschreef betrouwbare AI. Welke drie dingen moet die zijn?",
        options={
            "fast": "Snel, goedkoop en nauwkeurig",
            "open": "Open, uitlegbaar en vrij van elke bias",
            "right": "Wettig, ethisch en robuust",
        },
        correct="right",
        why={
            "fast": "Dat wil een koper misschien, maar het is niet wat de expertgroep vroeg. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "open": "Transparantie is een van de zeven eisen, maar niet een van de drie woorden. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "right": "Wettig, ethisch en robuust, met zeven eisen zoals menselijk toezicht, transparantie en fairness.",
        },
    )
    quick_trustworthy
    return (quick_trustworthy,)


@app.cell(hide_code=True)
def quick_check_altai():
    quick_altai = L.choice(
        "quick-altai",
        question="Wat is ALTAI?",
        options={
            "right": "Vragen om een systeem aan zeven eisen te toetsen",
            "law": "Een Europese wet die de riskantste toepassingen van AI helemaal verbiedt",
            "oecd": "De groep landen die de OESO-principes voor AI schreef",
        },
        correct="right",
        why={
            "law": "ALTAI is geen wet: niemand is verplicht hem te gebruiken. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "right": "De Assessment List for Trustworthy AI maakt van de zeven eisen van de expertgroep vragen die een organisatie over haar eigen systeem beantwoordt.",
            "oecd": "ALTAI komt van de Europese expertgroep, niet van de OESO. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
        },
    )
    quick_altai
    return (quick_altai,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een team wil dat zijn sollicitatietool privacy respecteert. Mooi, maar wat doet de engineer maandagochtend? Een waarde als privacy is te breed om te bouwen of te testen. Ergens tussen de waarde en de code moet iemand beslissen wat ze voor deze tool betekent, en bij die beslissingen staat of valt de waarde. Deze les maakt van een waarde eisen die je kunt controleren.

    **Een voorbeeld uit het dagelijks leven.** Een gezin wil een gezonde keuken. Dat worden regels, zoals elke dag groente en weinig suiker, en daarna dingen die je kunt controleren: groente op elk bord, hoogstens één frisdrank per week.
    """)
    return


@app.cell(hide_code=True)
def predict_origin():
    guess_origin = L.guess(
        "origin",
        question="Bij participatief ontwerp ontwerpen de mensen die een systeem gaan gebruiken mee. Waar en wanneer begon dat, denk je?",
        options={
            "valley": "Silicon Valley, jaren 2000",
            "right": "Scandinavië, jaren 70",
            "japan": "Japan, jaren 50",
        },
        reveal="**In Scandinavië in de jaren 70.** Onderzoekers werkten samen met vakbonden, zodat werknemers iets te zeggen kregen over de computersystemen die hun werk zouden veranderen.",
    )
    guess_origin
    return (guess_origin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Value-sensitive design**, van Batya Friedman en collega's sinds de jaren 90, neemt de waarden van iedereen die met een technologie te maken krijgt vanaf het begin mee, niet pas aan het eind. Het stelt drie soorten vragen: welke waarden staan er op het spel, en van wie (conceptueel); wat denken en doen de betrokkenen (empirisch); en hoe ondersteunt of hindert het ontwerp die waarden (technisch).

        *Design for Values*, een aanpak uit Delft uit dezelfde familie, geeft een manier om een waarde concreet te maken: een *waardenhiërarchie*. Bovenaan staat een waarde, zoals privacy. Daaronder staan normen, regels die zeggen wat de waarde van dit systeem vraagt: bewaar alleen de data die je nodig hebt, en alleen zo lang als nodig. Onderaan staan eisen die een engineer kan controleren: een cv wordt na hoogstens 30 dagen verwijderd. Elke stap omlaag is een keuze waarover je kunt discussiëren. Een eis die niemand naar een waarde kan terugvoeren, is moeilijk te verdedigen; een waarde zonder eisen is alleen een woord, zoals de principes in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).

        Bij *participatief ontwerp* doen de mensen om wie het gaat mee aan die keuzes: ze worden niet alleen gevraagd wat ze ervan vinden, maar beslissen mee.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **value-sensitive design** | een ontwerpmethode die de waarden van iedereen die met een technologie te maken krijgt vanaf het begin en tijdens het hele ontwerp meeneemt, door te vragen welke waarden op het spel staan, wat de betrokkenen denken en hoe het ontwerp die waarden raakt |
        """),
        L.watch_out("Een eis kan gehaald worden terwijl de waarde toch verloren gaat. Een tool die elk cv na 30 dagen verwijdert, kan er daarvoor nog steeds te veel uit halen: controleer de eis, en vraag af en toe opnieuw wat de waarde vraagt."),
        mo.Html(C.picture({"title": "Van een waarde, via een norm, naar een eis",
                   "levels": ["waarde", "norm", "eis"],
                   "columns": [["privacy", "bewaar alleen de data die je nodig hebt, en alleen zo lang als nodig", "een cv wordt na hoogstens 30 dagen verwijderd"],
                               ["fairness", "de tool is voor elke groep sollicitanten even nauwkeurig", "het aandeel fouten verschilt tussen de groepen hoogstens 0,05"],
                               ["transparantie", "een sollicitant kan horen waarom", "hoogstens 0,05 van de afwijzingen komt zonder reden"]],
                   "alt": "Drie waarden van een sollicitatietool, elk met een norm en een eis. Privacy: bewaar alleen de data die je nodig hebt, en alleen zo lang als nodig; een cv wordt na hoogstens 30 dagen verwijderd. Fairness: de tool is voor elke groep sollicitanten even nauwkeurig; het aandeel fouten verschilt tussen de groepen hoogstens 0,05. Transparantie: een sollicitant kan horen waarom; hoogstens 0,05 van de afwijzingen komt zonder reden."})),
        mo.md(r"""
        Een sollicitatietool die de course verzon: drie waarden, elk met één norm en één eis. Een echte hiërarchie heeft van allebei meer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo ontwerp je met een waarde:

    1. Noem de waarde, en van wie ze is.
    2. Schrijf de normen op: wat de waarde van dit systeem vraagt.
    3. Maak van elke norm een eis met een getal of een ja of nee.
    4. Meet het systeem en vergelijk elke meting met haar eis.
    5. Voer elke eis die niet gehaald wordt terug op haar waarde, en beslis met de betrokkenen wat er moet veranderen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De chatbot van een gemeente, verzonnen door de course. De waarde *bereikbaarheid* geeft de norm *inwoners krijgen snel antwoord*, en de eis *een antwoord binnen 5 seconden*. De waarde *betrouwbaarheid* geeft de norm *de chatbot vertelt inwoners niets verkeerds over hun rechten*, en de eis *hoogstens 0,02 van de antwoorden fout in een maandelijkse test*. `limit` bevat het hoogste dat elke eis toestaat, `measured` wat het team vond. Voer de cel uit.
    """)
    return


@app.cell
def example_chatbot():
    _limit = {"seconds": 5, "wrong answers": 0.02}
    _measured = {"seconds": 3.2, "wrong answers": 0.04}

    for _requirement in _limit:
        print(_requirement, _measured[_requirement] <= _limit[_requirement])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De chatbot is snel genoeg, maar heeft twee keer zoveel antwoorden fout als de eis toestaat. Stap 5 van het recept: die mislukte eis gaat terug op betrouwbaarheid, dus daar praat het team nu over. Nu de sollicitatietool van het plaatje: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_met():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Welke eisen worden gehaald?

        De sollicitatietool van het plaatje. `limit` bevat het hoogste dat elke eis toestaat, `measured` wat het team vond, en de loop gaat met `requirement` langs de eisen. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld, maar zet het antwoord in `met` in plaats van het te printen. De laatste regel is `met`, zodat de check het ziet. De check probeert ook andere metingen.
        """),
        L.task("Zet in `met`, onder elke eis, True als de gemeten waarde hoogstens de grens is, en False als dat niet zo is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_met():
    box_met = L.code_box("met", lines=7)
    box_met
    return (box_met,)


@app.cell(hide_code=True)
def check_met(box_met):
    result_met = L.check("met", box_met.value)
    result_met
    return (result_met,)


@app.cell(hide_code=True)
def hints_met():
    L.hints(
        "met",
        [
            "Stap 4 van het recept: de vergelijking van het uitgewerkte voorbeeld, met het antwoord bewaard onder de eis.",
            "Welke vergelijking zegt *hoogstens*? En hoe zet je een waarde onder een key in een dictionary?",
            "`met[requirement] = measured[requirement] ___ limit[requirement]`",
        ],
        why="`met[requirement] = measured[requirement] <= limit[requirement]` geeft `{'days kept': False, 'error gap': True, 'no reason': False}`: de tool bewaart cv's te lang en wijst te veel sollicitanten af zonder reden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_levels():
    mo.md(r"""
    ### Oefening 2. Waarde, norm of eis?

    Zes uitspraken over een gezondheidsapp, verzonnen door de course. Zet elke uitspraak op haar niveau van de waardenhiërarchie.
    """)
    return


@app.cell(hide_code=True)
def sort_levels_cell():
    sort_levels = L.sort("levels")
    sort_levels
    return (sort_levels,)


@app.cell(hide_code=True)
def check_levels(sort_levels):
    result_levels = L.check("levels", sort_levels)
    result_levels
    return (result_levels,)


@app.cell(hide_code=True)
def hints_levels():
    L.hints(
        "levels",
        [
            "Vraag bij elke uitspraak: is het één breed woord, een regel voor deze app, of iets wat je vandaag zou kunnen testen?",
            "Twee uitspraken zijn losse woorden. Van de vier zinnen kunnen er twee zoals ze er staan voor een test slagen of zakken.",
            "Waarde: autonomie en \\_\\_\\_. Norm: \\_\\_\\_ en geen schade. Eis: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Waarde:** autonomie en veiligheid. **Norm:** kiezen en geen schade. **Eis:** uit en een lijst.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_participation():
    mo.md(r"""
    ### Oefening 3. Meedoen
    """)
    return


@app.cell(hide_code=True)
def choice_participation_cell():
    choice_participation = L.choice(
        "participation",
        question="Een stad bouwt een AI-tool die haar busroutes plant, en vraagt reizigers een enquête in te vullen. Wat zou participatief ontwerp daaraan toevoegen?",
        options={
            "survey": "Een langere enquête, met meer vragen over de routes",
            "experts": "Een panel van experts dat namens de reizigers beslist",
            "right": "Reizigers beslissen mee wat de tool optimaliseert",
        },
        correct="right",
        why={
            "survey": "Een enquête vraagt reizigers wat ze vinden; de stad beslist nog steeds alleen. Wat voegt participatief ontwerp daaraan toe? Zie *Het idee*.",
            "experts": "Experts die voor reizigers beslissen, zijn het tegendeel: de reizigers hebben nog steeds niets te zeggen. Zie *Het idee*.",
            "right": "De reizigers kiezen mee welke waarden en normen gelden, zoals korte ritten voor de meeste mensen of een bus bij elk huis, en beantwoorden niet alleen achteraf vragen.",
        },
    )
    choice_participation
    return (choice_participation,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Principes en richtlijnen](/course/open/rai-principles?lang=nl) zag je dat principes korte woorden zijn die elk document op zijn eigen manier invult. Met een waardenhiërarchie vul je er een in voor één systeem, stap voor stap, zodat je elke eis kunt controleren en terugvoeren. In de volgende les, *Begeleidingsethiek en DEDA*, leer je twee Nederlandse methoden kennen die een team samen door deze keuzes leiden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_participation, result_levels, result_met):
    L.wrap_up(result_met, result_levels, choice_participation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
