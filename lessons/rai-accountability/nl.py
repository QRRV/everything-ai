# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-accountability. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-accountability", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_allocative():
    quick_allocative = L.choice(
        "quick-allocative",
        question="Een model geeft minder leningen aan mensen uit één groep. Wat voor schade is dat?",
        options={
            "right": "Allocatieve schade: het deelt minder uit",
            "representational": "Representatieve schade: het toont een groep slecht",
            "none": "Geen schade, als het klopt",
        },
        correct="right",
        why={
            "representational": "Representatieve schade gaat over hoe een groep getoond wordt, zoals een stereotype. Een lening is iets wat uitgedeeld wordt. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Een lening is iets wat een model uitdeelt, dus een verschil tussen groepen is allocatieve schade, en dat kun je per groep meten.",
            "none": "Gemiddeld nauwkeurig kan nog steeds minder leningen voor één groep betekenen. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_allocative
    return (quick_allocative,)


@app.cell(hide_code=True)
def quick_check_stereotype():
    quick_stereotype = L.choice(
        "quick-stereotype",
        question="Een zoekopdracht naar foto's van een beroep toont minder vrouwen dan er echt in dat beroep werken. Welke schade is dat?",
        options={
            "allocative": "Allocatieve schade, omdat iemand de baan niet krijgt",
            "real": "Geen: de foto's tonen de wereld zoals die is",
            "right": "Representatieve schade, een hardnekkig stereotype",
        },
        correct="right",
        why={
            "allocative": "De zoekopdracht zelf weigert niemand een baan; wat misgaat, is hoe een groep getoond wordt. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "real": "De zoekopdracht toont minder vrouwen dan er echt in het beroep werken, dus ze toont de wereld niet zoals die is. Zie het voorbeeld van de CEO's in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Een groep minder tonen dan ze er is, of als stereotype, is representatieve schade: daar moet je naar zoeken.",
        },
    )
    quick_stereotype
    return (quick_stereotype,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tussen 1985 en 1987 gaf een bestralingsapparaat, de Therac-25, minstens zes patiënten een enorme overdosis, en meerdere van hen overleden. De software had bugs; de maker zei dat een overdosis onmogelijk was; ziekenhuizen vertrouwden het scherm; operators hadden geleerd de foutmeldingen weg te klikken. Veel mensen deden hun deel, en niemand had geregeld wie er voor het geheel instond. AI-systemen worden net zo gebouwd, door veel teams en bedrijven, en daardoor is de vraag *wie is verantwoordelijk?* moeilijk te beantwoorden als het al is misgegaan.

    **Een voorbeeld uit het dagelijks leven.** In een gedeelde keuken op het werk gebruikt iedereen het koffieapparaat en ontkalkt niemand het: iedereen denkt dat een ander het doet.
    """)
    return


@app.cell(hide_code=True)
def predict_months():
    guess_months = L.guess(
        "months",
        question="Hoe lang duurde het, denk je, van de eerste bekende overdosis van de Therac-25 tot toezichthouders ziekenhuizen opdroegen ermee te stoppen?",
        options={
            "week": "Een paar weken",
            "months": "Ongeveer twee à drie maanden",
            "right": "Meer dan anderhalf jaar",
        },
        reveal="**Meer dan anderhalf jaar**: van juni 1985 tot februari 1987. De maker zei eerst dat het apparaat de overdoses niet veroorzaakt kon hebben. De Amerikaanse toezichthouder FDA noemde het in mei 1986 ondeugdelijk, maar het bleef in gebruik terwijl er reparaties kwamen, tot een andere bug in januari 1987 opnieuw een overdosis gaf (Leveson en Turner, 1993).",
    )
    guess_months
    return (guess_months,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Verantwoording** betekent dat iemand aanspreekbaar is op wat een systeem doet: die persoon kan de beslissingen uitleggen, kan worden gecorrigeerd en draagt de gevolgen als het misgaat. Drie dingen maken dat moeilijk.

        - *Het probleem van de vele handen.* Als veel mensen elk een deel bouwen, deed ieder het eigen deel en lijkt niemand verantwoordelijk voor het geheel. De filosoof Dennis Thompson gaf het in 1980 een naam voor overheden; Helen Nissenbaum liet in 1996 zien hoe het werkt bij computersystemen.
        - *Een morele kreukelzone.* De kreukelzone van een auto vangt de klap van een botsing op. Zo vangt de persoon die het dichtst bij de machine staat de schuld op, ook als het ontwerp van het systeem de fout waarschijnlijk maakte (Madeleine Clare Elish, 2019).
        - Een **responsibility gap**. Een systeem dat leert, kan dingen doen die de makers niet konden voorzien en de gebruikers niet konden sturen. Dan lijkt niemand de schuld te verdienen (Andreas Matthias, 2004).

        Het antwoord is verantwoording regelen voordat er iets misgaat: voor elke stap een naam, gaten en overlap opgespoord en opgelost, en een manier voor mensen om bezwaar te maken.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **verantwoording** | aanspreekbaar zijn op wat een AI-systeem doet: iemand kan de beslissingen uitleggen, kan worden gecorrigeerd en draagt de gevolgen als het misgaat |
        | **responsibility gap** | de zorg dat niemand redelijkerwijs de schuld kan krijgen als een lerend systeem schade veroorzaakt: de makers konden het niet voorzien en de gebruikers konden het niet sturen |
        """),
        L.watch_out("Twee namen bij een stap is niet twee keer zo veilig. Als twee teams allebei een stap claimen, kan elk op het andere rekenen: overlap kan een gat laten dat even groot is als helemaal geen naam."),
        mo.Html(C.picture({"title": "Welk team welke stap van een leningmodel claimt", "teams": ["data", "model", "product", "service"],
                           "nobody": "niemand",
                           "alt": "Zes stappen van een verzonnen leningmodel tegenover vier teams. Het datateam claimt collect data; het modelteam claimt train model en test model; het productteam claimt test model en launch; de servicedesk claimt niets. Niemand claimt set threshold of handle complaints."})),
        mo.md(r"""
        Een bank die de course verzon. Een gevulde stip betekent dat het team zegt dat het voor die stap instaat.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo regel je verantwoording voor een systeem:

    1. Zet de stappen op een rij, van data verzamelen tot klachten afhandelen.
    2. Vraag elk team voor welke stappen het instaat.
    3. Tel voor elke stap hoeveel teams hem claimen.
    4. Geef elke stap zonder naam, of met meer dan één, één eigenaar.
    5. Schrijf op hoe iemand die geraakt wordt bezwaar kan maken, en wie antwoordt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een systeem van een ziekenhuis dat een dosis medicijn voorstelt, verzonnen door de course. Drie teams kregen de vraag voor welke stappen ze instaan. `_count` begint voor elk van de drie stappen op 0. De buitenste loop gaat langs de teams, de binnenste loop langs de stappen die elk team claimt, en elke claim telt 1 op bij de telling van die stap. Voer de cel uit.
    """)
    return


@app.cell
def example_hospital():
    _claims = {"doctors": ["prescribe"], "pharmacy": ["prescribe"], "nurses": ["give medicine"]}
    _count = {"prescribe": 0, "check dose": 0, "give medicine": 0}

    for _team in _claims:
        for _step in _claims[_team]:
            _count[_step] = _count[_step] + 1

    for _step in _count:
        print(_step, _count[_step])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee teams claimen *prescribe* (voorschrijven), en niemand claimt *check dose* (de dosis controleren), de stap tussen voorschrijven en toedienen: die telling blijft 0. Zulke gaten zie je met het oog makkelijk over het hoofd, omdat een stap die niemand claimt in niemands antwoord voorkomt. Nu de bank van het plaatje, met zes stappen en vier teams: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie claimt wat?

        De bank van het plaatje. `claims` bevat voor elk team de stappen waarvoor het zegt in te staan, en `count` begint voor elke stap op 0. De buitenste loop gaat met `team` langs de teams, de binnenste met `step` langs de stappen van een team. Vervang de `...` in de binnenste loop door één regel die 1 optelt bij de telling van de stap die een team claimt, zoals in het uitgewerkte voorbeeld. De laatste regel is `count`, zodat de check het ziet. De check probeert ook andere antwoorden van de teams.
        """),
        L.task("Zorg dat `count` voor elke stap zegt hoeveel teams hem claimen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count",
        [
            "Elke keer dat de binnenste loop bij een stap komt, claimt één team meer die stap: de telling gaat één omhoog.",
            "Waar staat de telling van deze stap? En hoe zet je daar een nieuwe waarde neer die één meer is dan de oude?",
            "`count[step] = ___ + 1`",
        ],
        why="`count[step] = count[step] + 1` geeft `{'collect data': 1, 'train model': 1, 'test model': 2, 'set threshold': 0, 'launch': 1, 'handle complaints': 0}`: niemand claimt de drempel of de klachten, en twee teams claimen de test.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_who():
    mo.md(r"""
    ### Oefening 2. Welk idee past?

    Zes gevallen, verzonnen door de course. Zet elk geval bij het idee dat erbij past.
    """)
    return


@app.cell(hide_code=True)
def sort_who_cell():
    sort_who = L.sort("who")
    sort_who
    return (sort_who,)


@app.cell(hide_code=True)
def check_who(sort_who):
    result_who = L.check("who", sort_who)
    result_who
    return (result_who,)


@app.cell(hide_code=True)
def hints_who():
    L.hints(
        "who",
        [
            "Vraag bij elk geval: wordt de schuld over velen verspreid, komt hij terecht bij de persoon die het dichtst bij de machine staat, of lijkt niemand hem te verdienen?",
            "In twee gevallen krijgt één persoon de schuld. In twee andere deed het systeem iets wat niemand koos.",
            "Vele handen: de kredietscore en \\_\\_\\_. Kreukelzone: de veiligheidsbestuurder en \\_\\_\\_. Responsibility gap: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Vele handen:** de kredietscore en de chatbot. **Een morele kreukelzone:** de veiligheidsbestuurder en de operator. **Een responsibility gap:** zelf geleerd, en niemand kon het weten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Oefening 3. Wanneer je het regelt
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when",
        question="Een gemeente gaat binnenkort een AI-systeem gebruiken om te beslissen wie hulp krijgt bij de huur. Wanneer moet ze regelen wie waarvoor instaat?",
        options={
            "complaint": "Na de eerste klacht, als je weet wat er misging",
            "right": "Vóór de lancering, met een naam bij elke stap",
            "court": "Als een rechter of een journalist erom vraagt",
        },
        correct="right",
        why={
            "complaint": "Dan hebben de vele handen elk hun deel gedaan en is niemand eigenaar van de fout, en is er al iemand geschaad. Zie *Het idee*.",
            "right": "Vóór de lancering kun je elke stap nog één eigenaar geven en een manier om bezwaar te maken inrichten; achteraf komt de schuld meestal terecht bij wie het dichtstbij staat.",
            "court": "Dan is het te laat om de schade te voorkomen, en komt de schuld terecht waar dat het makkelijkst is, niet waar hij hoort. Zie *Het idee*.",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) zag je wat een model mensen kan aandoen. Hier vroeg je wie daarvoor instaat, en zag je dat dat vóór de schade geregeld moet worden, niet achteraf gezocht. In de volgende les, *Een mens in de loop*, bekijk je één veelgebruikt antwoord, een mens de beslissingen van het systeem laten controleren, en waarom dat vaak minder goed werkt dan het klinkt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_count, result_who):
    L.wrap_up(result_count, result_who, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
