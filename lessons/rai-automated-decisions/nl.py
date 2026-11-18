# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-automated-decisions. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-automated-decisions", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bias():
    quick_bias = L.choice(
        "quick-bias",
        question="Een beoordelaar mist een probleem omdat het systeem er niet op wees. Hoe heet dat?",
        options={
            "right": "Automation bias",
            "oversight": "Menselijk toezicht, dat werkt zoals bedoeld",
            "power": "Te weinig macht om nee te zeggen",
        },
        correct="right",
        why={
            "oversight": "Menselijk toezicht is dat een mens meekijkt. Hier faalde dat meekijken: hoe heet dat? Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
            "right": "Missen wat het systeem niet aanwees, en de verkeerde voorstellen ervan volgen, zijn de twee kanten van automation bias.",
            "power": "Macht gaat over nee kunnen zeggen zonder ervoor te boeten. Hier zag de beoordelaar het probleem niet eens. Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
        },
    )
    quick_bias
    return (quick_bias,)


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="Wat zegt het best of een mens in de loop werkt?",
        options={
            "agree": "Hoe vaak de persoon het eens is met wat het systeem zegt",
            "speed": "Hoe snel de persoon door elk geval heen gaat",
            "right": "Hoeveel fouten van het systeem de persoon vangt",
        },
        correct="right",
        why={
            "agree": "Iemand die het 9 van de 10 keer eens is, vangt misschien niets. Zie oefening 1 in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
            "speed": "Snelheid zegt iets over tijd, niet over of de controle werkt. Zie oefening 1 in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
            "right": "Alleen de gevallen waarin het systeem het mis heeft, laten zien of de persoon iets toevoegt.",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Steeds meer besluiten over mensen worden door software genomen, of erdoor voorbereid: wie een uitkering krijgt, een boete, een vergunning of een controle door een inspecteur. Dat kan eerlijker zijn dan wanneer een mens beslist. Het kan ook het enige wegnemen dat iemand in een ongewone situatie hielp: iemand die luistert. Of een geautomatiseerd besluit aanvaardbaar is, hangt af van wat het beslist, en van wat iemand kan doen als het fout is.

    **Een voorbeeld uit het dagelijks leven.** Een kaartjesautomaat die alleen pinpassen aanneemt, is voor iedereen hetzelfde en nooit chagrijnig. Wie alleen contant geld heeft, kan het aan niemand vragen.
    """)
    return


@app.cell(hide_code=True)
def predict_judges():
    guess_judges = L.guess(
        "judges",
        question="Bij een Amerikaanse immigratierechtbank kwamen asielzaken uit hetzelfde land bij toeval bij verschillende rechters terecht. Hoe ver denk je dat de toekenningen van de rechters uiteenliepen?",
        options={
            "close": "Binnen ongeveer 5 procentpunt van elkaar",
            "some": "Zo'n 20 procentpunt uit elkaar",
            "right": "Van ongeveer 5 tot ongeveer 88 procent",
        },
        reveal="**Van ongeveer 5 tot ongeveer 88 procent.** Bij de immigratierechtbank in Miami hadden asielzoekers uit Colombia ongeveer 5 procent kans bij de ene rechter en ongeveer 88 procent bij een andere rechter in hetzelfde gebouw (Ramji-Nogales, Schoenholtz en Schrag, 2007).",
    )
    guess_judges
    return (guess_judges,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Mensen die beslissen zijn niet allemaal hetzelfde: sommigen zijn streng, anderen mild, en sommigen beslissen 's ochtends anders dan aan het eind van de dag. Een regel in software beslist elk geval op dezelfde manier, en dat kan eerlijker zijn. Maar het neemt ook ruimte weg. Een ambtenaar aan de balie kon zien dat een geval ongewoon was en een uitzondering maken. In 2002 beschreven Mark Bovens en Stavros Zouridis hoe die ruimte verschuift van de ambtenaar aan de balie naar de mensen die het systeem ontwerpen, buiten het zicht van wie het besluit treft.

        Of een geautomatiseerd besluit aanvaardbaar is, hangt eerst af van *wat* het beslist. Een feit controleren, zoals een leeftijd of of er betaald is, kan elke keer op dezelfde manier. Omstandigheden wegen, zoals of iemand wilde misleiden of of een schuld te zwaar is om te dragen, vraagt iemand die dit geval kan zien.

        En het hangt af van wat er gebeurt als het besluit fout is. **Contestability** is hoe goed mensen een besluit kunnen aanvechten en het opnieuw laten bekijken en veranderen. Daarvoor moet iemand weten dat er een besluit is genomen en waarom, waar bezwaar kan, genoeg tijd hebben, en iemand vinden die luistert en het kan veranderen. Waar de AVG een volledig automatisch besluit met rechtsgevolgen of andere ingrijpende gevolgen al toestaat, geeft ze mensen het recht om het aan te vechten. In Nederland heeft wie het oneens is met een besluit van de overheid zes weken om bezwaar te maken.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **contestability** | hoe goed mensen een besluit dat met een AI-systeem genomen is ter discussie kunnen stellen, en het opnieuw laten bekijken en veranderen |
        """),
        L.watch_out("Een reden is niet altijd een uitleg. *Score above threshold* (score boven de drempel) is een reden, maar geeft de persoon niets om tegenin te gaan: waardoor was de score hoog, en klopt dat?"),
        mo.Html(C.picture({"title": "Vijf besluitbrieven, langs drie toetsen", "letter": "brief",
                           "tests": ["reden", "waar bezwaar", "42 dagen"], "yes": "bezwaar kan", "no": "bezwaar is lastig",
                           "alt": "Vijf besluitbrieven langs drie toetsen: een reden, waar je bezwaar maakt, en minstens 42 dagen. Brief 1 en 5 halen alle drie. Brief 2 heeft geen reden, brief 3 zegt niets over waar je bezwaar maakt, en brief 4 geeft maar 14 dagen."})),
        mo.md(r"""
        Een uitkeringsinstantie die de course verzon. Een gevulde stip betekent dat de brief de toets haalt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bepaal je of je een besluit over mensen automatiseert:

    1. Schrijf op wat het systeem beslist, en of het een feit controleert of omstandigheden weegt.
    2. Laat het wegen van omstandigheden aan een mens, met het systeem hooguit als hulp.
    3. Geef elk besluit een reden waar de persoon tegenin kan gaan.
    4. Zeg waar bezwaar kan, en geef genoeg tijd.
    5. Zorg dat een bezwaar terechtkomt bij iemand die het besluit kan veranderen, en tel hoe vaak dat gebeurt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    # in mo.vstack, zodat de editor van marimo de "" in de tekst laat zoals hij is
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Een functie die één ding van een besluitbrief toetst: geeft hij een reden? Een brief is een dictionary; een lege reden is `""`. Voer de cel uit.
        """),
    ])
    return


@app.cell
def example_reason():
    def _has_reason(letter):
        if letter["reason"] == "":
            return False
        return True

    print(_has_reason({"reason": "income above the limit", "contact": "objections desk", "days": 42}))
    print(_has_reason({"reason": "", "contact": "objections desk", "days": 42}))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste brief geeft een reden, de tweede niet. Een `return` beëindigt de functie meteen: zodra een toets faalt, doet de rest er niet meer toe. Stap 3 en 4 van het recept vragen meer dan een reden: oefening 1 toetst alle drie.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contestable():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Kan iemand bezwaar maken?

        De brieven van het plaatje. `letters` bevat de vijf brieven; elke brief is een dictionary met de keys `"reason"`, `"contact"` en `"days"`. Vervang de `...` door de body van `contestable`: geef `False` terug als de brief geen reden heeft of niets zegt over waar je bezwaar maakt, en anders of hij de zes weken geeft, 42 dagen, die de Nederlandse wet voor een bezwaar geeft. De loop eronder probeert elke brief, en de laatste regel is `results`, zodat de check het ziet. De check probeert ook andere brieven.
        """),
        L.task("Laat `contestable` `True` geven bij een brief waartegen iemand bezwaar kan maken, en `False` bij een brief waartegen dat niet kan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contestable():
    box_contestable = L.code_box("contestable", lines=18)
    box_contestable
    return (box_contestable,)


@app.cell(hide_code=True)
def check_contestable(box_contestable):
    result_contestable = L.check("contestable", box_contestable.value)
    result_contestable
    return (result_contestable,)


@app.cell(hide_code=True)
def hints_contestable():
    L.hints(
        "contestable",
        [
            "Drie toetsen na elkaar, zoals in het uitgewerkte voorbeeld: elke toets die faalt, beëindigt de functie met `False`.",
            "Het uitgewerkte voorbeeld doet de reden al. Welk ander veld kan leeg zijn? En wat vergelijkt de laatste `return`?",
            "Na de twee `if`-toetsen: `return letter[\"days\"] ___ 42`",
        ],
        why="Twee toetsen zoals in het uitgewerkte voorbeeld, voor `\"reason\"` en `\"contact\"`, en dan `return letter[\"days\"] >= 42`, geven `[True, False, False, False, True]`: brief 1 geeft precies 42 dagen, en dat is genoeg.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fact():
    mo.md(r"""
    ### Oefening 2. Een feit of een oordeel?

    Zes besluiten over mensen, verzonnen door de course. Zet elk besluit bij wat het doet.
    """)
    return


@app.cell(hide_code=True)
def sort_fact_cell():
    sort_fact = L.sort("fact")
    sort_fact
    return (sort_fact,)


@app.cell(hide_code=True)
def check_fact(sort_fact):
    result_fact = L.check("fact", sort_fact)
    result_fact
    return (result_fact,)


@app.cell(hide_code=True)
def hints_fact():
    L.hints(
        "fact",
        [
            "Vraag bij elk besluit: kun je het beslissen door iets op te zoeken, of hangt het af van de situatie van deze persoon?",
            "Drie besluiten vergelijken een bekende waarde met een vaste. De andere drie vragen naar opzet, een last of een kind.",
            "Controleert een feit: oud genoeg, \\_\\_\\_ en \\_\\_\\_. Weegt omstandigheden: \\_\\_\\_, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Controleert een feit:** oud genoeg, betaald en boven de grens. **Weegt omstandigheden:** fraude, schrijnend en het beste voor het kind.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.md(r"""
    ### Oefening 3. Wat ontbreekt er?
    """)
    return


@app.cell(hide_code=True)
def choice_score_cell():
    choice_score = L.choice(
        "score",
        question="Een brief zegt: *Uw aanvraag is afgewezen omdat uw risicoscore boven de drempel lag. U kunt binnen zes weken bezwaar maken bij de bezwaarbalie.* Wat heeft iemand het meest nodig om bezwaar te maken?",
        options={
            "phone": "Een telefoonnummer, naast de balie",
            "right": "Waardoor de score hoog was",
            "time": "Meer tijd: zes weken is voor iedereen te kort",
        },
        correct="right",
        why={
            "phone": "Een telefoonnummer kan helpen, maar de brief zegt al waar bezwaar kan. Wat zou de persoon schrijven? Zie het blok *Let op* in *Het idee*.",
            "right": "Zonder te weten waardoor de score hoog was, kan de persoon niet zeggen wat er niet aan klopt: de brief haalt de toetsen, maar geeft niets om tegenin te gaan.",
            "time": "Zes weken is wat de Nederlandse wet geeft, en de brief geeft dat. Het probleem is wat je schrijft. Zie het blok *Let op* in *Het idee*.",
        },
    )
    choice_score
    return (choice_score,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een mens in de loop](/course/open/rai-human-oversight?lang=nl) zag je dat een mens die een systeem controleert vaak weinig toevoegt. Hier zag je dat wat geautomatiseerd mag worden, afhangt van wat er beslist wordt, en dat mensen een manier nodig hebben om het besluit achteraf aan te vechten. In de volgende les, *Hoe ver kun je een leven voorspellen?*, vraag je hoe goed een systeem uitkomsten in het leven van mensen überhaupt kan voorspellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_score, result_contestable, result_fact):
    L.wrap_up(result_contestable, result_fact, choice_score)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
