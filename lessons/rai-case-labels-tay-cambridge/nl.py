# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-labels-tay-cambridge. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-labels-tay-cambridge", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl), en een over [Dark patterns, nudges en manipulatie](/course/open/rai-dark-patterns?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Een zoekopdracht naar afbeeldingen van *ingenieur* laat bijna alleen mannen zien. Wat voor schade is dat?",
        options={
            "allocative": "Allocatieve schade: het deelt minder banen uit aan vrouwen",
            "right": "Representatieve schade: in hoe het een groep laat zien",
            "none": "Geen schade, want niemand wordt iets geweigerd",
        },
        correct="right",
        why={
            "allocative": "De zoekopdracht deelt niets uit: geen baan, geen lening, geen controle. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Het beslist over niemand iets, maar het laat een groep scheef zien, en dat is ook schade.",
            "none": "Voor schade is geen weigering nodig. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_hypernudge():
    quick_hypernudge = L.choice(
        "quick-hypernudge",
        question="Wat maakt een hypernudge anders dan een gewone nudge?",
        options={
            "law": "Die is altijd verboden, door de AI Act en de DSA",
            "right": "Die past zich met data aan elke persoon aan",
            "price": "Die maakt een van de keuzes duurder",
        },
        correct="right",
        why={
            "law": "De wet verbiedt manipulatie die schade doet, niet elke hypernudge. Zie *Het idee* in [Dark patterns, nudges en manipulatie](/course/open/rai-dark-patterns?lang=nl).",
            "right": "Die verandert steeds en verschilt per persoon, dus anderen zien misschien niet wat jij ziet, en dat maakt hem moeilijk op te merken.",
            "price": "Een nudge, hyper of niet, verandert de kosten nauwelijks. Zie *Het idee* in [Dark patterns, nudges en manipulatie](/course/open/rai-dark-patterns?lang=nl).",
        },
    )
    quick_hypernudge
    return (quick_hypernudge,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2015 gaf een foto-app twee zwarte vrienden de naam van een aap. In 2016 plaatste een chatbot binnen een paar uur na de start racistische berichten. In 2018 kwam uit dat de data van maximaal 87 miljoen gebruikers van Facebook mogelijk was gedeeld met een bedrijf dat politieke advertenties op maat richtte. De eerste twee gevallen deden schade met wat een systeem zei; het derde met wat er gebeurde met data over mensen, van wie de meesten de app nooit hadden gebruikt.

    **Een voorbeeld uit het dagelijks leven.** Op een feest ligt een gastenboek de hele nacht open, en iedereen kan erin schrijven. 's Ochtends zegt de bladzijde die iedereen het eerst leest misschien meer over de slechtste gasten dan over het feest.
    """)
    return


@app.cell(hide_code=True)
def predict_tay():
    guess_tay = L.guess(
        "tay",
        question="Microsoft zette zijn chatbot Tay op 23 maart 2016 op Twitter, om te leren van de mensen met wie hij praatte. Hoe lang bleef Tay volgens jou online?",
        options={
            "week": "Ongeveer een hele week",
            "month": "Ongeveer een maand",
            "right": "Ongeveer 16 uur",
        },
        reveal="**Ongeveer 16 uur**, en in die tijd plaatste Tay meer dan 95.000 tweets. Microsoft schreef dat het iets cruciaals over het hoofd had gezien voor juist deze aanval.",
    )
    guess_tay
    return (guess_tay,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De labels bij foto's.** In juni 2015 ontdekte de programmeur Jacky Alciné dat Google Photos foto's van Alciné en een vriend, allebei zwart, het label van een aap had gegeven, een label met een lange racistische geschiedenis dat deze course niet herhaalt. Google zei geschokt te zijn en het oprecht te betreuren. In 2018 testte het tijdschrift *Wired* de app met 40.000 foto's vol dieren: zoeken naar apen vond niets. Google had het model niet verbeterd; het had de labels weggehaald. In 2023 vond *The New York Times* hetzelfde, en Google zei dat het nut van zulke labels niet opweegt tegen het risico op schade.

        **Tay.** Op 23 maart 2016 zette Microsoft Tay op Twitter, een chatbot die leerde van de mensen met wie hij praatte. Een groep gebruikers misbruikte een functie *repeat after me* om hem racistische en haatdragende berichten te laten plaatsen, en Tay nam zulke taal ook zelf over. Na 16 uur en meer dan 95.000 tweets haalde Microsoft Tay offline. Het bedrijf bood excuses aan, en schreef dat het de volle verantwoordelijkheid nam omdat het deze mogelijkheid niet van tevoren had gezien.

        Allebei zijn het gevallen van representatieve schade: wat het systeem zei, besliste over niemand iets, en deed mensen toch pijn.

        **Microtargeting** is op maat gemaakte politieke of commerciële boodschappen tonen aan kleine groepen mensen, geselecteerd met data over hun eigenschappen en gedrag. Anderen zien die boodschappen niet, net als bij een hypernudge.

        **Cambridge Analytica.** In 2014 bouwde de onderzoeker Aleksandr Kogan een app voor Facebook met een persoonlijkheidsquiz. Zo'n 300.000 mensen wereldwijd gebruikten hem, en via de regels die Facebook toen had, verzamelde de app ook data van hun vrienden: maximaal 87 miljoen mensen, de meesten in de Verenigde Staten, schatte Facebook in 2018. Gebruikers kregen te horen dat de app hun naam of andere gegevens waarmee je hen kon herkennen niet zou verzamelen; volgens de Amerikaanse Federal Trade Commission, de FTC, was dat niet waar. Het bedrijf van Kogan verkocht data uit de app aan SCL Elections en Cambridge Analytica: volgens de Britse toezichthouder data over ongeveer 30 miljoen Amerikaanse kiezers. Die bedrijven gaven kiezers persoonlijkheidsscores en richtten politieke advertenties op groepen van hen. *The Observer* en *The Guardian* maakten het in maart 2018 bekend. De Britse privacytoezichthouder gaf Facebook een boete van 500.000 pond, het hoogste wat de wet van toen toestond, en in 2019 gaf de FTC Facebook een boete van 5 miljard dollar voor het breken van een eerder bevel over privacy. In 2020 concludeerde de Britse toezichthouder dat de beweringen van het bedrijf, zoals 5.000 gegevens over elk van 230 miljoen Amerikaanse volwassenen, overdreven konden zijn: de methodes waren gewoon, en de voorspellingen voor nieuwe mensen waren waarschijnlijk veel minder goed. Dat de advertenties ook maar één stem veranderden, is niet aangetoond.

        Dat is schade in wat er met data gebeurde: de data van mensen werd gebruikt voor een doel waar ze niet mee hadden ingestemd, en de meesten hadden de app nooit gebruikt.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **microtargeting** | op maat gemaakte politieke of commerciële boodschappen tonen aan kleine groepen mensen, geselecteerd met data over hun eigenschappen en gedrag |
        """),
        L.watch_out("Grote beweringen over wat een model kan, zijn geen bewijs dat het werkt. Cambridge Analytica zei veel over zijn eigen kracht; de onderzoekers vonden gewone methodes en voorspellingen die bij nieuwe mensen waarschijnlijk veel zwakker waren."),
        mo.Html(C.picture({"title": "Van elke 1.000 mensen van wie data werd verzameld: hoeveel gebruikten de app",
                           "installed": "gebruikte de quizapp", "others": "data verzameld via een vriend",
                           "alt": "1.000 kleine vakjes, waarvan 3 gemarkeerd: van elke 1.000 mensen van wie data werd verzameld, gebruikten er ongeveer 3 de quizapp zelf."})),
        mo.md(r"""
        Ongeveer 300.000 mensen gebruikten de app, en de data van maximaal 87 miljoen mensen werd verzameld: ongeveer 3 op de 1.000. De andere 997 hadden de app nooit gebruikt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om naar deze schade te zoeken voordat een systeem live gaat:

    1. Verzamel wat het systeem over mensen en groepen zegt, en zoek naar beledigingen en stereotypen, zoals de les *Twee soorten schade* liet zien.
    2. Vraag wat er gebeurt als mensen het iets haatdragends proberen te laten zeggen, zoals de gebruikers van Tay deden.
    3. Zet op een rij van wie het systeem data verzamelt, en tel wie van hen daarmee instemde.
    4. Vergelijk wat de mensen in de data te horen kregen met wat er met hun data gebeurt.
    5. Wees voorzichtig met beweringen over wat een model kan, tot iemand ze op nieuwe mensen heeft getest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een quizapp, verzonnen door de course, geïnstalleerd door drie mensen, A, B en C. `_friends` bevat hoeveel vrienden elk van hen heeft; de course telt ze alsof geen twee van de drie een vriend gemeen hebben. De app verzamelt data van elke persoon die hem installeert en van al diens vrienden. De loop gaat met `_person` langs de mensen en telt in `_reached` iedereen op van wie de app data verzamelt: de persoon, 1, en de vrienden. `_share` is het aantal mensen dat de app installeerde, `len(_friends)`, gedeeld door `_reached`. Voer de cel uit.
    """)
    return


@app.cell
def example_friends():
    _friends = {"A": 200, "B": 350, "C": 150}

    _reached = 0
    for _person in _friends:
        _reached = _reached + 1 + _friends[_person]
    _share = len(_friends) / _reached
    print(_reached, round(_share, 5))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De app verzamelt de data van 703 mensen, en 3 van hen installeerden hem: een aandeel van 0,00427, of ongeveer 4 op de 1.000. Stap 3 van het recept voor een andere app: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_friends():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie stemde in?

        Een andere quizapp, verzonnen door de course, geïnstalleerd door vier mensen. `friends` bevat hoeveel vrienden elk van hen heeft. Vervang de `...` door de regels van het uitgewerkte voorbeeld, met de namen zonder hun liggende streepje: `reached`, `person` en `share`. De loop gaat met `person` langs `friends`. Laat de `print` weg: de laatste regel is `share`, zodat de check het ziet. De check probeert ook andere mensen.
        """),
        L.task("Reken `share` uit: het aantal mensen dat de app installeerde, gedeeld door iedereen van wie hij data verzamelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_friends():
    box_friends = L.code_box("friends", lines=10)
    box_friends
    return (box_friends,)


@app.cell(hide_code=True)
def check_friends(box_friends):
    result_friends = L.check("friends", box_friends.value)
    result_friends
    return (result_friends,)


@app.cell(hide_code=True)
def hints_friends():
    L.hints(
        "friends",
        [
            "Het uitgewerkte voorbeeld doet dit voor drie mensen. Neem de regels over, zonder de `print`, en haal bij elke naam het liggende streepje weg.",
            "Begin `reached` bij 0. Voor elke persoon die de app installeerde, wordt de data van die persoon en van de vrienden verzameld. Deel daarna.",
            "In de loop: `reached = reached + 1 + ___[person]`; erna: `share = len(friends) / ___`.",
        ],
        why="`reached = 0`, dan `for person in friends:` met daarin `reached = reached + 1 + friends[person]`, en na de loop `share = len(friends) / reached`. Dat geeft `0.00286`: 4 van 1.399 mensen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Schade en antwoord

    Zes feiten uit de drie gevallen. Zet elk feit bij waar het over gaat.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Vraag bij elk feit: is het iets wat een systeem zei, iets wat met data gebeurde, of iets wat daarna gedaan werd als antwoord op de schade?",
            "De labels bij foto's leveren twee feiten, Tay één en Cambridge Analytica drie.",
            "Gezegd: het label en \\_\\_\\_. Data: de vrienden en \\_\\_\\_. Antwoord: \\_\\_\\_ en de boete.",
        ],
        why="**Schade in wat een systeem over mensen zegt of laat zien:** het label en repeat after me. **Schade in wat er met data gebeurt:** de vrienden en de belofte. **Wat eraan gedaan werd:** de verdwenen labels en de boete.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claims():
    mo.md(r"""
    ### Oefening 3. Wat vonden de onderzoekers?
    """)
    return


@app.cell(hide_code=True)
def choice_claims_cell():
    choice_claims = L.choice(
        "claims",
        question="Wat concludeerde de Britse privacytoezichthouder in 2020 over Cambridge Analytica?",
        options={
            "brexit": "Zijn advertenties beslisten de uitslag van het Brexit-referendum",
            "right": "Zijn beweringen over zijn kracht waren mogelijk overdreven",
            "votes": "Zijn advertenties veranderden de stemmen van miljoenen Amerikanen",
        },
        correct="right",
        why={
            "brexit": "De toezichthouder vond geen rol in het EU-referendum buiten een paar eerste verkenningen, en de data van Facebook ging over Amerikaanse kiezers. Zie *Het idee*.",
            "right": "Het bedrijf beweerde 5.000 gegevens te hebben over elk van 230 miljoen Amerikaanse volwassenen; de toezichthouder vond dat dat overdreven kon zijn, dat de methodes gewoon waren, en dat de voorspellingen voor nieuwe mensen waarschijnlijk veel minder goed waren.",
            "votes": "Dat de advertenties stemmen veranderden, is niet aangetoond. Wat zei de toezichthouder over de beweringen van het bedrijf zelf? Zie *Het idee*.",
        },
    )
    choice_claims
    return (choice_claims,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) zag je dat schade in hoe een systeem mensen laat zien over niemand iets beslist en toch schade is: de labels bij foto's en Tay zijn zulke gevallen. In [Dark patterns, nudges en manipulatie](/course/open/rai-dark-patterns?lang=nl) leerde je de hypernudge kennen, die anderen niet zien; microtargeting werkt net zo, en Cambridge Analytica bouwde het op data waarvoor mensen geen toestemming hadden gegeven. In de volgende les, *Het ongeluk met een zelfrijdende testauto*, bekijk je een ongeluk dat niet door één enkele fout kwam.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claims, result_friends, result_kinds):
    L.wrap_up(result_friends, result_kinds, choice_claims)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
