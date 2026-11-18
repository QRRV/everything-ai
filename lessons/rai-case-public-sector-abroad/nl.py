# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-public-sector-abroad. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-public-sector-abroad", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_room():
    quick_room = L.choice(
        "quick-room",
        question="Als een regel van het bureau van een ambtenaar naar software gaat, waar gaat de ruimte om een uitzondering te maken dan naartoe?",
        options={
            "desk": "Die blijft bij de ambtenaar aan het loket",
            "right": "Naar de mensen die het systeem ontwerpen",
            "person": "Naar de persoon over wie het besluit gaat",
        },
        correct="right",
        why={
            "desk": "In software beslist de regel elk geval op dezelfde manier, dus die ruimte heeft de ambtenaar niet meer. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
            "right": "Mark Bovens en Stavros Zouridis beschreven dit in 2002: de ruimte gaat naar de ontwerpers, buiten het zicht van de persoon over wie het besluit gaat.",
            "person": "De persoon over wie het besluit gaat, ziet nog minder van de regel dan eerst. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
        },
    )
    quick_room
    return (quick_room,)


@app.cell(hide_code=True)
def quick_check_reason():
    quick_reason = L.choice(
        "quick-reason",
        question="Een besluitbrief geeft als enige reden: *Score boven de drempel*. Wat is daar mis mee?",
        options={
            "fine": "Er is niets mis mee: het geeft een reden",
            "right": "Het geeft de persoon niets om tegenin te gaan",
            "law": "Drempels zijn niet toegestaan in besluiten over mensen",
        },
        correct="right",
        why={
            "fine": "Een reden is niet altijd een uitleg. Zie het blok *Let op* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
            "right": "Wat maakte de score hoog, en klopt dat? Zonder dat kan de persoon nergens bezwaar tegen maken.",
            "law": "Een drempel kan deel zijn van een rechtmatig besluit. Zie het blok *Let op* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
        },
    )
    quick_reason
    return (quick_reason,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Alleen al tussen november 2016 en maart 2017 stuurde de Australische overheid elke week ongeveer 20.000 brieven naar mensen die volgens haar te veel uitkering hadden gekregen, soms jaren eerder. De schulden kwamen uit een berekening die het inkomen van een jaar gelijk verdeelde over periodes van twee weken, en veel daarvan klopten niet. Robodebt, zoals de regeling ging heten, is een van de vier gevallen in deze les van buiten Nederland. Elk ging op een andere manier mis, en bij elke manier hoort een eigen controle.

    **Een voorbeeld uit het dagelijks leven.** Een docent hoort dat een klas gemiddeld een 7 haalde, en geeft elke leerling een 7. Voor de klas als geheel klopt dat; voor bijna elke leerling niet.
    """)
    return


@app.cell(hide_code=True)
def predict_grades():
    guess_grades = L.guess(
        "grades",
        question="In 2020 gingen de examens in Engeland niet door, en een model bepaalde de cijfers voor de A-levels. Hoeveel van de 718.276 cijfers vielen volgens jou lager uit dan het cijfer dat de docent had gegeven?",
        options={
            "few": "Ongeveer 1 op de 20",
            "quarter": "Ongeveer 1 op de 4",
            "right": "Ongeveer 2 op de 5",
        },
        reveal="**Ongeveer 2 op de 5**: 39,1 van elke 100 cijfers, de meeste één cijfer lager. Vier dagen na de uitslag kregen de leerlingen alsnog de cijfers van hun docenten.",
    )
    guess_grades
    return (guess_grades,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Inkomensmiddeling** betekent dat je wat iemand over een lange periode verdiende gelijk verdeelt over kortere periodes, alsof het steeds in hetzelfde tempo binnenkwam. Voor iemand met een vast loon komt dat dicht bij de waarheid. Voor iemand die een half jaar werkte en daarna geen baan had, niet.

        **Robodebt.** In Australië wordt een uitkering per twee weken betaald, en mensen geven op wat ze in elke periode van twee weken verdienden. Vanaf 2015 vergeleek de overheid die opgaven met het inkomen dat werkgevers aan de belastingdienst hadden doorgegeven, gelijk verdeeld over de periodes van twee weken in de gegevens. Waar het gemiddelde inkomen hoger was dan wat iemand had opgegeven, nam het systeem aan dat die persoon te veel had gekregen en maakte het een schuld aan, soms met 10 procent kosten erbovenop, tenzij de persoon het tegendeel kon bewijzen, met bewijs van jaren eerder. Deanna Amato kreeg een schuld van 3.215,38 dollar: het middelen had het inkomen van Amato in zes periodes op bijna vier keer het echte bedrag gezet. Toen Centrelink, de dienst die de uitkeringen betaalt, opnieuw keek, was er maar 1,48 dollar te veel betaald, en later bleek Amato ongeveer 480 dollar te weinig te hebben gekregen. In november 2019 verklaarde de federale rechter de vordering van de schuld ongeldig, in een uitspraak waarmee de overheid instemde. In 2021 keurde de rechter een schikking goed: de overheid zou schulden van minstens 1,763 miljard dollar bij ongeveer 433.000 mensen laten vallen, ongeveer 751 miljoen dollar terugbetalen die al was betaald, en 112 miljoen dollar schadevergoeding betalen, inclusief de proceskosten. Een koninklijke onderzoekscommissie concludeerde in 2023: *Robodebt was een grof en wreed mechanisme, eerlijk noch legaal, en het gaf veel mensen het gevoel dat ze criminelen waren.*

        **De A-levels.** In 2020 gingen de examens in Engeland niet door. Docenten gaven elke leerling het cijfer dat die het waarschijnlijkst zou hebben gehaald, en zetten hun leerlingen per vak op volgorde. Ofqual, de toezichthouder op de examens, vond die cijfers veel hoger dan in eerdere jaren: het aandeel A* zou van 7,7 naar 13,9 procent zijn gegaan. Daarom gebruikte Ofqual in plaats daarvan een model. Voor elke school voorspelde het hoe de cijfers verdeeld zouden zijn, uit de resultaten van de school in dat vak in de drie jaar ervoor, en daarna deelde het die cijfers uit in de volgorde van de docent. Het cijfer van een leerling hing zo af van het verleden van de school en van de plek op de volgorde, niet van het cijfer dat de docent gaf. Voor de kleinste klassen, tot ongeveer 5 leerlingen, gebruikte het model de cijfers van de docenten, omdat er te weinig leerlingen waren om mee te vergelijken; van ongeveer 5 tot 15 mengde het die twee. In heel Engeland steeg het aandeel A- en A*-cijfers maar met 2,4 punten, maar van elke 100 cijfers viel 39,1 lager uit dan dat van de docent. Op 17 augustus 2020, vier dagen na de uitslag, gaf Ofqual de leerlingen alsnog de cijfers van hun docenten, en bood excuses aan.

        **MiDAS.** Van oktober 2013 tot augustus 2015 besliste een computersysteem van de werkloosheidsdienst van Michigan, in de Verenigde Staten, zelf dat mensen fraude hadden gepleegd. Een vragenlijst niet terugsturen leidde bijvoorbeeld tot een vaststelling van fraude, met een boete van vier keer de uitkering. Toen 22.427 van die vaststellingen werden nagekeken, bleek bij 93 van elke 100 geen sprake van fraude, al merkte een federale rechter later op dat dat getal ter discussie is gesteld. In 2024 keurde een rechter een schikking van 20 miljoen dollar goed.

        **Allegheny.** In Allegheny County, in Pennsylvania, zien de medewerkers die meldingen aannemen over kinderen die misschien verwaarloosd worden een score van 1 tot 20, voordat ze beslissen of er iemand gaat onderzoeken. De score voorspelt of het kind binnen twee jaar uit huis geplaatst wordt, uit gegevens van 21 overheidsbronnen, zoals de geestelijke gezondheidszorg en registraties van de gevangenis. De schrijver Virginia Eubanks wees erop dat gezinnen die overheidsdiensten gebruiken veel meer gegevens achterlaten dan gezinnen die dat niet doen. Boven sommige scores zegt het hulpmiddel dat de melding onderzocht moet worden, tenzij een leidinggevende anders beslist. Onderzoekers van Carnegie Mellon University vonden in 2022 dat, bij de meldingen van 2016 tot 2018 waarin de medewerkers konden kiezen, de score alleen 67,9 van elke 100 zwarte kinderen en 50,0 van elke 100 witte kinderen voor een onderzoek zou hebben aangewezen; met de eigen beslissingen van de medewerkers was dat 50,2 en 43,1.

        De vier gevallen gingen op verschillende manieren mis. Bij Robodebt klopte een berekening voor veel mensen niet. Bij de A-levels klopte een regel gemiddeld, en voor veel leerlingen niet. Bij MiDAS nam een regel de plaats in van iemand die de zaak afweegt; bij Allegheny had de score dat kunnen doen, en de medewerkers die de zaken afwogen maakten kleinere verschillen tussen zwarte en witte kinderen dan de score alleen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **inkomensmiddeling** | wat iemand over een lange periode verdiende gelijk verdelen over kortere periodes, alsof het steeds in hetzelfde tempo binnenkwam; Robodebt rekende er schulden mee uit |
        """),
        L.watch_out("Getallen uit deze gevallen tellen verschillende dingen: schulden, mensen, cijfers of nagekeken vaststellingen. Alleen bij Robodebt al gaat het om ongeveer 433.000 mensen met een onterechte schuld, 470.000 schulden die de overheid zei terug te betalen, en 648.000 mensen in de groepsvordering. Kijk voordat je twee getallen vergelijkt wat elk ervan telt."),
        mo.Html(C.picture({"title": "Wat het middelen ziet in elke periode",
                           "axis": "dollar", "fortnight": "periode van twee weken",
                           "earned": "verdiend", "averaged": "gemiddeld inkomen", "invented": "verzonnen door het middelen",
                           "alt": "Staven van wat iemand verdiende in elk van de 26 periodes van twee weken in een jaar: 1.300 dollar in elk van de eerste 10, niets in de andere 16. Een lijn op 500 dollar is het gemiddelde inkomen. In de 16 periodes zonder werk is de ruimte tussen niets en de lijn gearceerd: inkomen dat het middelen verzon."})),
        mo.md(r"""
        Verzonnen door de course: iemand die 1.300 dollar verdiende in elk van de eerste 10 periodes van twee weken van een jaar, en daarna niets. Het middelen zet 500 dollar in elke periode, ook in de 16 periodes zonder werk.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een berekening of een regel te controleren die over mensen beslist:

    1. Schrijf op wat die aanneemt over het leven van iemand, zoals een inkomen dat het hele jaar in hetzelfde tempo binnenkomt.
    2. Verzin een paar mensen voor wie die aanname niet klopt, zoals iemand die een half jaar werkte.
    3. Reken uit wat de berekening voor hen zegt, en vergelijk dat met wat er echt gebeurde.
    4. Kijk bij een regel die gemiddeld klopt naar de mensen voor wie hij niet klopt, niet alleen naar het totaal.
    5. Vraag wie moet bewijzen dat de uitkomst niet klopt, of die persoon dat kan, en wie de uitkomst kan overrulen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Iemand met losse baantjes, verzonnen door de course, verdiende om en om 400 en 600 dollar in de 26 periodes van twee weken van een jaar. `_earned` bevat wat deze persoon in elke periode verdiende. De eerste loop telt het jaar op in `_total`, en `_average` is dat totaal gedeeld door het aantal periodes. De tweede loop gaat de periodes nog eens langs en telt, waar `_average` hoger is dan wat er verdiend werd, het verschil op bij `_invented`. Voer de cel uit.
    """)
    return


@app.cell
def example_averaging():
    _earned = [400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400,
               600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600, 400, 600]

    _total = 0
    for _amount in _earned:
        _total = _total + _amount
    _average = _total / len(_earned)

    _invented = 0
    for _amount in _earned:
        if _average > _amount:
            _invented = _invented + (_average - _amount)
    print(_average, _invented)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het middelen zet 500 dollar in elke periode. In de 13 periodes waarin deze persoon 400 verdiende, ziet het elke keer 100 dollar te veel: 1.300 dollar die in die periodes niet verdiend werd. Kreeg iemand in zo'n periode ook een uitkering, dan telde Robodebt dat inkomen daartegen mee. Stap 2 van het recept vraagt om iemand voor wie de aanname helemaal niet klopt: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_averaging():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een half jaar werk

        Iemand, verzonnen door de course, verdiende 1.000 dollar in elk van de eerste 13 periodes van twee weken van een jaar, raakte toen het werk kwijt en leefde van een uitkering. `earned` bevat wat deze persoon in elke periode verdiende. Vervang de `...` door de regels van het uitgewerkte voorbeeld, met de namen zonder hun liggende streepje: `total`, `amount`, `average` en `invented`. Beide loops gaan met `amount` langs `earned`. Laat de `print` weg: de laatste regel is `invented`, zodat de check het ziet. De check probeert ook iemand van wie het inkomen op en neer gaat.
        """),
        L.task("Reken `average` uit, het gemiddelde inkomen per periode, en `invented`, het inkomen dat het middelen in periodes zet waarin het niet verdiend werd."),
    ])
    return


@app.cell(hide_code=True)
def exercise_averaging():
    box_averaging = L.code_box("averaging", lines=16)
    box_averaging
    return (box_averaging,)


@app.cell(hide_code=True)
def check_averaging(box_averaging):
    result_averaging = L.check("averaging", box_averaging.value)
    result_averaging
    return (result_averaging,)


@app.cell(hide_code=True)
def hints_averaging():
    L.hints(
        "averaging",
        [
            "Het uitgewerkte voorbeeld doet hetzelfde voor iemand met losse baantjes. Neem de regels over, zonder de `print`, en haal bij elke naam het liggende streepje weg.",
            "Tel eerst `earned` op in `total`, en deel door `len(earned)` voor `average`. Dan een tweede loop: in welke periodes ziet het middelen meer dan er verdiend werd?",
            "`if average > amount:` en daaronder `invented = invented + (___ - ___)`.",
        ],
        why="`total = 0` en `for amount in earned:` met `total = total + amount`, dan `average = total / len(earned)`. Daarna `invented = 0` en een tweede `for amount in earned:` met daarin `if average > amount:` en `invented = invented + (average - amount)`. Dat geeft een `average` van `500.0` en een `invented` van `6500.0`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Oefening 2. Drie manieren om mis te gaan

    Zes feiten uit de vier gevallen. Zet elk feit bij de manier waarop het misging.
    """)
    return


@app.cell(hide_code=True)
def sort_failures_cell():
    sort_failures = L.sort("failures")
    sort_failures
    return (sort_failures,)


@app.cell(hide_code=True)
def check_failures(sort_failures):
    result_failures = L.check("failures", sort_failures)
    result_failures
    return (result_failures,)


@app.cell(hide_code=True)
def hints_failures():
    L.hints(
        "failures",
        [
            "Vraag bij elk feit: klopte de rekensom zelf niet, klopte de uitkomst voor de groep maar niet voor de persoon, of besliste een getal of een regel waar iemand de zaak had moeten afwegen?",
            "Twee feiten gaan over Robodebt, twee over de A-levels, één over Michigan en één over Allegheny.",
            "Een berekening: de periodes en \\_\\_\\_. Gemiddeld goed: het verleden van de school en \\_\\_\\_. In plaats van een oordeel: \\_\\_\\_ en het verplichte onderzoek.",
        ],
        why="**Een berekening die voor veel mensen niet klopte:** de periodes en de 3.215,38 dollar. **Een regel die gemiddeld klopte:** het verleden van de school en de totalen. **Een getal of een regel in plaats van een oordeel:** de vragenlijst en het verplichte onderzoek.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.md(r"""
    ### Oefening 3. Waarom hielden kleine klassen de cijfers van hun docenten?
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small",
        question="Waarom gaf het A-levelmodel de kleinste klassen, tot ongeveer 5 leerlingen, de cijfers van hun docenten?",
        options={
            "right": "Te weinig leerlingen om mee te vergelijken",
            "trust": "Ofqual vertrouwde docenten van kleine klassen meer",
            "private": "De meeste kleine klassen zaten op privéscholen",
        },
        correct="right",
        why={
            "right": "Met zo weinig leerlingen zegt het verleden van de school te weinig over dit jaar: Ofqual wilde aanpassingen vermijden die het niet kon verdedigen. Kleine klassen komen in sommige vakken veel vaker voor dan in andere, dus dezelfde regel pakte niet voor elke leerling hetzelfde uit.",
            "trust": "Het model vertrouwde de cijfers van docenten juist niet: Ofqual vond ze veel hoger dan in eerdere jaren. Wat veranderde de grootte van de klas? Zie *Het idee*.",
            "private": "De regel ging over de grootte van de klas, niet over het soort school. Waarom zou grootte uitmaken voor een model dat met het verleden vergelijkt? Zie *Het idee*.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl) zag je dat het afwegen van omstandigheden iemand vraagt die het geval kan zien, en dat mensen een echte manier nodig hebben om bezwaar te maken. Robodebt zette een berekening op de plek van dat afwegen en liet mensen met bewijs van jaren eerder bewijzen dat die niet klopte; MiDAS liet een regel over fraude beslissen; de A-levels lieten zien dat een regel gemiddeld kan kloppen en voor veel leerlingen niet. In de volgende les, *Als het label het verkeerde meet*, bekijk je een algoritme in de zorg dat van het verkeerde label leerde, en een creditcard waarvan niemand van buiten de beslissingen kon controleren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_averaging, result_failures):
    L.wrap_up(result_averaging, result_failures, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
