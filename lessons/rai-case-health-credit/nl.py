# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-health-credit. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-health-credit", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_questions():
    quick_questions = L.choice(
        "quick-questions",
        question="ProPublica noemde de scores van COMPAS bevooroordeeld, en Northpointe noemde ze eerlijk. Hoe kan allebei kloppen?",
        options={
            "right": "Ze stelden andere vragen aan dezelfde scores",
            "data": "Ze hadden elk andere gegevens over de verdachten",
            "error": "Een van beide maakte een rekenfout",
        },
        correct="right",
        why={
            "right": "Northpointe vroeg welk deel van de gemarkeerde mensen opnieuw werd opgepakt; ProPublica vroeg welk deel van de mensen die niet opnieuw werden opgepakt toch gemarkeerd was.",
            "data": "Beide keken naar dezelfde scores uit Broward County. Zie *Het idee* in [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl).",
            "error": "Beide sommen klopten. Zie *Het idee* in [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl).",
        },
    )
    quick_questions
    return (quick_questions,)


@app.cell(hide_code=True)
def quick_check_amazon():
    quick_amazon = L.choice(
        "quick-amazon",
        question="Waar leerde het model van Amazon cv's met het woord *women's* lager te waarderen?",
        options={
            "rule": "Uit een regel die het team erin had geschreven",
            "right": "Uit tien jaar cv's, vooral van mannen",
            "ads": "Uit de woorden in de vacatures van Amazon",
        },
        correct="right",
        why={
            "rule": "Niemand schreef zo'n regel; het team probeerde later juist het model dat woord te laten negeren. Zie *Het idee* in [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl).",
            "right": "Het leerde uit patronen in de cv's die in tien jaar naar het bedrijf waren gestuurd, en de meeste daarvan kwamen van mannen.",
            "ads": "Het leerde uit de cv's, niet uit de vacatures. Zie *Het idee* in [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl).",
        },
    )
    quick_amazon
    return (quick_amazon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2019 vond een onderzoek in *Science* dat een algoritme dat in de Verenigde Staten voor miljoenen patiënten werd gebruikt, en dat ziekenhuizen helpt kiezen wie extra zorg krijgt, zwarte patiënten dezelfde scores gaf als witte patiënten die gezonder waren. Het algoritme gebruikte geen afkomst. Het probleem was wat het had leren voorspellen: niet hoe ziek iemand was, maar hoeveel de zorg zou kosten. Een model leert zijn label, wat je er ook mee bedoelde.

    **Een voorbeeld uit het dagelijks leven.** Een school wil de leerlingen vinden die hulp nodig hebben met lezen, en kiest de leerlingen van wie de ouders vorig jaar de minste boeken kochten. Gekochte boeken zijn makkelijk te tellen, maar ze meten net zo goed geld als lezen.
    """)
    return


@app.cell(hide_code=True)
def predict_sicker():
    guess_sicker = L.guess(
        "sicker",
        question="Bij de scores waarop patiënten voor het programma werden gekozen: hoeveel meer chronische ziektes hadden zwarte patiënten volgens jou dan witte patiënten?",
        options={
            "same": "Ongeveer evenveel als witte patiënten",
            "few": "Ongeveer 5 procent meer",
            "right": "Ongeveer 26 procent meer",
        },
        reveal="**Ongeveer 26 procent meer**: 4,8 chronische ziektes tegen 3,8, bij de score vanaf waar patiënten automatisch werden gekozen.",
    )
    guess_sicker
    return (guess_sicker,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **proxy label** is iets wat je kunt meten en dat de plaats inneemt van wat je eigenlijk wilt weten, zoals de kosten van zorg de plaats innemen van de behoefte aan zorg. Een model leert de proxy. Waar de proxy en het echte doel uit elkaar lopen, zit het model ernaast op een manier die de gewone getallen niet laten zien.

        **Het zorgalgoritme.** Ziekenhuizen in de Verenigde Staten gebruiken commerciële algoritmes om patiënten met een complexe zorgvraag te kiezen voor een programma met extra zorg. Ziad Obermeyer en collega's onderzochten er één, typerend voor de branche, in één academisch ziekenhuis, met 6.079 zwarte en 43.539 witte patiënten van 2013 tot 2015. De 3 procent patiënten met de hoogste scores werden automatisch voor het programma gekozen. Het algoritme liet afkomst bewust weg, en het voorspelde de kosten van het volgende jaar voor beide groepen ongeveer even goed. Maar het label was kosten, niet zorgbehoefte: bij hetzelfde aantal chronische ziektes werd er per jaar 1.801 dollar minder uitgegeven aan zwarte patiënten dan aan witte patiënten, door ongelijke toegang tot zorg. Bij dezelfde score waren zwarte patiënten dus zieker. Van de patiënten die automatisch werden gekozen, was 17,7 procent zwart. De onderzoekers simuleerden een score die in beide groepen dezelfde gezondheid betekende: dan was het 46,5 procent geweest. Een nieuw label dat gezondheid en kosten combineerde, haalde 84 procent van de bias weg. De maker van het algoritme herhaalde de analyse op ongeveer 3,7 miljoen patiënten in het hele land.

        **De Apple Card.** In november 2019 schreef de softwareontwikkelaar David Heinemeier Hansson op Twitter dat de Apple Card, uitgegeven door de bank Goldman Sachs, Heinemeier Hansson een kredietlimiet had gegeven van 20 keer die van de vrouw van Heinemeier Hansson. Ook Steve Wozniak, een van de oprichters van Apple, schreef dat de voorwaarden veel beter waren dan die voor de vrouw van Wozniak. De bank zei dat geslacht geen rol speelde. De toezichthouder van de staat New York, het Department of Financial Services, bekeek de gegevens van bijna 400.000 aanvragers in New York, en meldde in maart 2021 dat het geen bewijs vond dat vrouwen slechter werden behandeld, met opzet of in de praktijk. Echtgenoten hadden vaak een verschillende kredietgeschiedenis, bijvoorbeeld als maar een van beiden op een hypotheek stond. Maar de klantenservice kon de limieten niet uitleggen aan de mensen die klaagden, en de toezichthouder schreef dat dit gebrek aan transparantie verwarring leek te veroorzaken die voorkomen had kunnen worden.

        Bij het zorgalgoritme mat het label iets anders dan de vraag. Bij de Apple Card vond de toezichthouder geen onrechtmatige discriminatie, maar konden de mensen om wie het ging niet zien waarom hun limieten verschilden, en het dus ook niet zelf controleren.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **proxy label** | een label dat je wel kunt meten en dat de plaats inneemt van wat je eigenlijk wilt weten, zoals de kosten van zorg in plaats van de behoefte aan zorg; het model leert de proxy, niet het echte doel |
        """),
        L.watch_out("Een proxy label kan er met de gewone getallen perfect uitzien: dit algoritme voorspelde de kosten voor zwarte en witte patiënten ongeveer even goed. Vergelijk de proxy met wat je eigenlijk wilt weten, voor elke groep."),
        mo.Html(C.picture({"title": "Wat het kostenlabel deed met de patiënten die automatisch werden gekozen",
                           "groups": ["zwart", "wit"], "illnesses": "chronische ziektes",
                           "share": "zwarte patiënten gekozen (%)", "cost": "kostenlabel", "simulated": "zelfde gezondheid",
                           "decimal": ",",
                           "alt": "Twee staafdiagrammen. Links: bij de score vanaf waar patiënten automatisch werden gekozen, hadden zwarte patiënten 4,8 chronische ziektes en witte patiënten 3,8. Rechts: van de patiënten die automatisch werden gekozen, was met het kostenlabel 17,7 procent zwart, en met een score die in beide groepen dezelfde gezondheid betekende was dat 46,5 procent geweest."})),
        mo.md(r"""
        De getallen van Obermeyer en collega's. De open staaf is hun simulatie, niet iets wat gebeurde.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om het label te controleren van een model dat over mensen beslist:

    1. Schrijf op wat je eigenlijk wilt weten, zoals de behoefte aan zorg.
    2. Schrijf op wat het model in plaats daarvan leert: het label, zoals de kosten van zorg.
    3. Vraag wat het label nog meer beïnvloedt, naast wat je wilt weten, zoals de toegang tot zorg.
    4. Vergelijk voor elke groep wie het label kiest met wie het doel zou kiezen.
    5. Controleer of de mensen over wie een besluit gaat kunnen horen waarom, zodat ze het ook zelf kunnen controleren.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Acht patiënten, verzonnen door de course: vier zwarte patiënten, B1 tot en met B4, en vier witte patiënten, W1 tot en met W4. `_illnesses` bevat het aantal chronische ziektes van elke patiënt, en `_costs` de kosten van hun zorg in het volgende jaar, in dollar. In deze verzonnen data staat een chronische ziekte voor 2.000 dollar zorg per jaar bij een zwarte patiënt en 3.000 bij een witte patiënt. De loop gaat met `_patient` langs de patiënten en zet in `_picked` elke patiënt met minstens `_threshold` chronische ziektes: kiezen op zorgbehoefte. Voer de cel uit.
    """)
    return


@app.cell
def example_by_need():
    _illnesses = {"B1": 5, "B2": 4, "B3": 2, "B4": 1, "W1": 4, "W2": 3, "W3": 2, "W4": 1}
    _costs = {"B1": 10000, "B2": 8000, "B3": 4000, "B4": 2000,
              "W1": 12000, "W2": 9000, "W3": 6000, "W4": 3000}

    _threshold = 4
    _picked = []
    for _patient in _illnesses:
        if _illnesses[_patient] >= _threshold:
            _picked.append(_patient)
    print(_picked)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op zorgbehoefte worden B1, B2 en W1 gekozen. Het programma in het onderzoek koos niet op zorgbehoefte, maar met een score die kosten leerde. Stap 4 van het recept vraagt wie het label kiest: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_picked():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie kiest het kostenlabel?

        Dezelfde acht patiënten, in `illnesses` en `costs`. Deze keer kiest het programma elke patiënt van wie de kosten minstens `threshold` dollar zijn, en `threshold` is `9000`. Vervang de `...` door de regels van het uitgewerkte voorbeeld, met de namen zonder hun liggende streepje, `picked` en `patient`, en kies op `costs` in plaats van op `illnesses`. De loop gaat met `patient` langs `costs`. Laat de `print` weg: de laatste regel is `picked`, zodat de check het ziet. De check probeert ook een andere drempel.
        """),
        L.task("Maak van `picked` de lijst van de patiënten van wie de kosten minstens `threshold` zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_picked():
    box_picked = L.code_box("picked", lines=12)
    box_picked
    return (box_picked,)


@app.cell(hide_code=True)
def check_picked(box_picked):
    result_picked = L.check("picked", box_picked.value)
    result_picked
    return (result_picked,)


@app.cell(hide_code=True)
def hints_picked():
    L.hints(
        "picked",
        [
            "Het uitgewerkte voorbeeld kiest op ziektes. Doe hetzelfde met `costs` en `threshold`, zonder de `print`.",
            "Begin met een lege lijst. In de loop vergelijkt een `if` de kosten van `patient` met `threshold`. Wat gaat er in `picked` als de kosten hoog genoeg zijn?",
            "`if costs[patient] >= threshold:` en daaronder `picked.___(___)`.",
        ],
        why="`picked = []`, dan `for patient in costs:` met daarin `if costs[patient] >= threshold:` en `picked.append(patient)`. Dat geeft `['B1', 'W1', 'W2']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxies():
    mo.md(r"""
    ### Oefening 2. Het doel of de proxy?

    Zes dingen: twee uit deze les, twee uit *COMPAS en een sollicitatietool*, en twee over een nieuwssite, verzonnen door de course. Zet elk ding bij wat het is.
    """)
    return


@app.cell(hide_code=True)
def sort_proxies_cell():
    sort_proxies = L.sort("proxies")
    sort_proxies
    return (sort_proxies,)


@app.cell(hide_code=True)
def check_proxies(sort_proxies):
    result_proxies = L.check("proxies", sort_proxies)
    result_proxies
    return (result_proxies,)


@app.cell(hide_code=True)
def hints_proxies():
    L.hints(
        "proxies",
        [
            "Vraag bij elk ding: is het wat iemand eigenlijk wil weten, of wat makkelijk te tellen is in de gegevens?",
            "De zes horen in paren bij elkaar: één over patiënten, één over verdachten en één over lezers. In elk paar neemt het ene de plaats in van het andere.",
            "De doelen: zorgbehoefte, \\_\\_\\_ en interesse. De proxy's: kosten, opnieuw opgepakt en \\_\\_\\_.",
        ],
        why="**Wat je eigenlijk wilt weten:** zorgbehoefte, een nieuw misdrijf en interesse. **Wat er in plaats daarvan gemeten wordt:** kosten, opnieuw opgepakt en klikken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_regulator():
    mo.md(r"""
    ### Oefening 3. Wat vond de toezichthouder?
    """)
    return


@app.cell(hide_code=True)
def choice_regulator_cell():
    choice_regulator = L.choice(
        "regulator",
        question="Wat meldde de toezichthouder in New York in maart 2021 over de Apple Card?",
        options={
            "women": "Dat de bank vrouwen met een regel bewust lagere limieten gaf",
            "right": "Geen onrechtmatige discriminatie, wel te weinig uitleg",
            "fine": "Alles was in orde, ook de klantenservice",
        },
        correct="right",
        why={
            "women": "De toezichthouder vond geen regel die vrouwen lagere limieten gaf, en geen bewijs van discriminatie met opzet of in de praktijk. Zie *Het idee*.",
            "right": "Ze vond geen bewijs dat vrouwen slechter werden behandeld, maar de klantenservice kon de limieten niet uitleggen aan de mensen die klaagden, en die bleven verward achter.",
            "fine": "De toezichthouder vond geen onrechtmatige discriminatie, maar vond wel iets mis. Kon de klantenservice de limieten uitleggen? Zie *Het idee*.",
        },
    )
    choice_regulator
    return (choice_regulator,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl) zag je dat schade uit de data kan komen en uit wat er gemeten werd, en dat COMPAS leerde wie er werd opgepakt, niet wie een misdrijf pleegde. Bij het zorgalgoritme was het label zelf het probleem: kosten namen de plaats in van zorgbehoefte, en de gewone getallen zagen er goed uit. Bij de Apple Card vond de toezichthouder geen onrechtmatige discriminatie, maar konden mensen niet zien waarom hun limieten verschilden. In de volgende les, *Labels, een chatbot en microtargeting*, bekijk je schade in wat een systeem zegt en in wat het met data doet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_regulator, result_picked, result_proxies):
    L.wrap_up(result_picked, result_proxies, choice_regulator)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
