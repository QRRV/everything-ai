# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-disability-screen-out. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-disability-screen-out", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Een online test is voor iedereen gelijk en noemt geen grond, maar mensen met een handicap zakken er veel vaker voor. Wat is het op het eerste gezicht?",
        options={
            "direct": "Directe discriminatie, want het effect is groot",
            "right": "Mogelijk indirecte discriminatie",
            "none": "Geen onderscheid",
        },
        correct="right",
        why={
            "direct": "De test noemt geen grond, dus het is niet direct. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
            "right": "Een neutrale regel die een groep in het bijzonder treft, is indirect, tenzij hij objectief gerechtvaardigd is.",
            "none": "Voor iedereen gelijk zijn is niet genoeg: het effect telt. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_college():
    quick_college = L.choice(
        "quick-college",
        question="Wat kan het College voor de Rechten van de Mens doen als het discriminatie vaststelt?",
        options={
            "fine": "Een boete opleggen",
            "right": "Een oordeel geven dat gezag heeft",
            "ban": "De organisatie verbieden nog iemand aan te nemen",
        },
        correct="right",
        why={
            "fine": "Het College kan geen boete of schadevergoeding opleggen; daarvoor ga je naar de rechter. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
            "right": "Zijn oordelen zijn niet bindend, maar in ruim 80 procent van de gevallen waarin het discriminatie vaststelt, volgen maatregelen.",
            "ban": "Het College geeft een oordeel; het kan niets verbieden. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
        },
    )
    quick_college
    return (quick_college,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Automatische filters zijn goedkoop: wijs elk cv met een lang gat af, eis een score op een spel met een klok, laat software beoordelen hoe iemand praat in een video. Elk ervan kan iemand eruit halen die het werk zou kunnen doen, vanwege een handicap of een chronische ziekte. Groepen vergelijken, zoals in de vorige lessen, vindt dit slecht, omdat geen twee handicaps hetzelfde zijn. Je moet kijken wie een regel eruit haalt, en waarom.

    **Een voorbeeld uit het dagelijks leven.** Een restaurant neemt alleen telefonisch reserveringen aan. Iemand die doof is, wil er net zo graag eten, maar de regel houdt die persoon buiten, al was dat door niemand zo bedoeld. Een e-mailadres zou genoeg zijn.
    """)
    return


@app.cell(hide_code=True)
def predict_vendor():
    guess_vendor = L.guess(
        "vendor",
        question="Een leverancier controleerde zijn sollicitatietest: mensen met een handicap, als één groep, halen hem even vaak als anderen. Kan de test toch onrechtmatig iemand met een handicap uitfilteren?",
        options={
            "no": "Nee, de controle op de groep laat zien dat hij veilig is",
            "right": "Ja, want elke handicap is anders",
            "intent": "Alleen als de leverancier dat zo bedoelde",
        },
        reveal="**Ja.** De Amerikaanse Equal Employment Opportunity Commission (EEOC) schreef in 2022 dat elke handicap uniek is: iemand kan slecht scoren op een test door een handicap, en daardoor worden uitgefilterd, hoe goed andere mensen met een handicap het ook doen. Een test die werkt voor mensen in een rolstoel, kan een blinde sollicitant nog steeds laten zakken.",
    )
    guess_vendor
    return (guess_vendor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Uitfilteren.** In mei 2022 publiceerden de EEOC en het Amerikaanse ministerie van Justitie elk een handleiding over sollicitatiesoftware en handicap. Uitfilteren, in het Engels *screen out*, gebeurt volgens de EEOC als een handicap een sollicitant of werknemer belet om aan een selectie-eis te voldoen, of het resultaat erop verlaagt, en die persoon daardoor een kans op werk misloopt. De voorbeelden: een chatbot die sollicitanten met lange gaten in hun werkgeschiedenis afwijst, terwijl een gat kan komen door een behandeling voor een ziekte; een verplichte score van 90 procent op een geheugenspel, dat een blinde sollicitant niet kan spelen, al heeft die misschien een heel goed geheugen; software die spraakpatronen in een video-interview beoordeelt, en iemand met een spraakstoornis waarschijnlijk niet goed beoordeelt; en aanslagen per minuut tellen, wat nadelig uitvalt voor wie spraakherkenning gebruikt in plaats van een toetsenbord.

        **Wanneer het onrechtmatig is.** Volgens de Amerikaanse wet over handicap, de ADA, is uitfilteren onrechtmatig als de persoon de wezenlijke taken van de baan zou kunnen doen met een redelijke aanpassing: een verandering in hoe dingen gaan, waardoor iemand met een handicap kan solliciteren of het werk kan doen. De werkgever is meestal verantwoordelijk, ook als een leverancier het programma maakte of draait. De EEOC raadde aan om sollicitanten te vertellen wat een programma meet en dat andere vormen en andere tests mogelijk zijn, en om alleen vaardigheden te meten die het werk echt nodig heeft. De handleiding was nooit bindend, en in januari 2025 is ze van de website van de EEOC gehaald; de ADA zelf geldt nog steeds.

        **Elke handicap is anders.** Een leverancier die zijn programma vrij van bias noemt, heeft meestal de gemiddelden voor ras en geslacht gecontroleerd. Maar iemand kan slecht scoren door een handicap, schreef de EEOC, hoe goed andere mensen met een handicap het ook doen. De Amerikaanse richtlijnen met de vuistregel uit [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) gelden niet voor handicap. Dus kijk je naar de regel zelf: tel wie hij eruit haalt, en vraag je af of die mensen het werk zouden kunnen doen, misschien met een aanpassing.

        **In Nederland.** De Wet gelijke behandeling op grond van handicap of chronische ziekte uit 2003, de WGBH/CZ, verbiedt direct en indirect onderscheid op deze gronden, ook bij het aanbieden van een baan en bij werving. De wet voegt een plicht toe: wie onder het verbod valt, moet naar gelang de behoefte doeltreffende aanpassingen verrichten, tenzij die een onevenredige belasting vormen. Het College voor de Rechten van de Mens kijkt of de aanpassing geschikt en noodzakelijk is, en de plicht begint zodra de persoon de behoefte kenbaar heeft gemaakt. Het College waarschuwt werkgevers dat mensen die slechtziend of slechthorend zijn een game-assessment of een online sollicitatieformulier misschien niet kunnen gebruiken: zorg dat digitale formulieren, tests en spellen toegankelijk zijn. De Europese richtlijn uit 2000 over werk vraagt hetzelfde, en volgens het VN-verdrag inzake de rechten van personen met een handicap, dat sinds 14 juli 2016 voor Nederland geldt, is het weigeren van een redelijke aanpassing zelf discriminatie.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **uitfilteren** | een handicap belet iemand om aan een selectie-eis te voldoen, of verlaagt het resultaat, en die persoon loopt de kans op werk mis; in het Engels *screen out* |
        | **doeltreffende aanpassing** | een verandering in hoe dingen gaan, waardoor iemand met een handicap op gelijke voet kan solliciteren, werken of leren; in het Engels *reasonable accommodation* |
        | **onevenredige belasting** | de grens van de plicht: een aanpassing die voor deze organisatie te duur of te zwaar is, hoeft niet |
        """),
        L.watch_out("Een controle op groepen halen betekent niet dat niemand wordt uitgefilterd. Handicap is geen groep met één gemiddelde: wie discriminatie van blinde sollicitanten voorkomt, voorkomt die daarmee niet bij sollicitanten met autisme of epilepsie, schreef het ministerie van Justitie."),
        mo.Html(C.picture({"title": "Een filter op gaten in het cv",
                           "xlabel": "maanden zonder werk in het cv", "yes": "kan het werk doen",
                           "no": "kan het werk niet doen", "limit": "eruit: meer dan 6 maanden", "months": "maanden",
                           "key": "gevuld: eruit gefilterd, open: blijft"})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je uit of een filter of een test mensen uitfiltert:

    1. Zet op een rij waar het naar kijkt: gaten, snelheid, spraak, aanslagen, een spel, een formulier.
    2. Vraag je bij elk punt af welke handicaps of ziektes het resultaat kunnen verlagen, terwijl de persoon het werk nog steeds kan doen, misschien met een aanpassing.
    3. Tel wie de regel eruit haalt, en kijk naar die mensen, niet alleen naar het percentage van een groep.
    4. Vertel sollicitanten wat het programma meet en hoe ze om een aanpassing kunnen vragen, en bied een andere vorm of een andere test aan.
    5. Houd alleen de eisen over die het werk echt nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een filter op gaten, met sollicitanten die de course verzon. Elke sollicitant in `_applicants` heeft een naam, de maanden zonder werk in het cv, en of die het werk kan doen, misschien met een aanpassing. Het filter haalt iedereen eruit met een gat van meer dan `_limit` maanden. De loop verzamelt hun namen in `_removed`. Voer de cel uit.
    """)
    return


@app.cell
def example_removed():
    # verzonnen sollicitanten: maanden zonder werk in het cv, en of ze het werk kunnen doen
    _applicants = [
        {"name": "Ada", "gap": 0, "can_do_job": "yes"},
        {"name": "Bram", "gap": 14, "can_do_job": "yes"},
        {"name": "Chen", "gap": 3, "can_do_job": "no"},
        {"name": "Dewi", "gap": 9, "can_do_job": "yes"},
        {"name": "Emre", "gap": 1, "can_do_job": "no"},
        {"name": "Fleur", "gap": 24, "can_do_job": "no"},
        {"name": "Gijs", "gap": 7, "can_do_job": "yes"},
        {"name": "Jesse", "gap": 6, "can_do_job": "yes"},
    ]
    # het filter: niemand met een gat van meer dan zoveel maanden
    _limit = 6

    _removed = []
    for _applicant in _applicants:
        if _applicant["gap"] > _limit:
            _removed.append(_applicant["name"])
    print(_removed)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vier gaan eruit: Bram, Dewi, Fleur en Gijs. In deze verzonnen lijst kwam het gat van Bram door een behandeling voor een ziekte. Maar hoeveel van de vier zouden het werk kunnen doen? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wrongly():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eruit, maar wel geschikt

        Tel de sollicitanten die het filter eruit haalt terwijl ze het werk kunnen doen. Vervang de `...` door een loop over de sollicitanten, zoals in het uitgewerkte voorbeeld, met de naam zonder liggend streepje: `applicant`. Begin met `wrongly = 0`, en tel er 1 bij op als het gat groter is dan `limit` en `can_do_job` `"yes"` is: zet een tweede `if` binnen de eerste. De laatste regel is `wrongly`, zodat de check het ziet. De check probeert ook een andere grens.
        """),
        L.task("Reken `wrongly` uit: hoeveel sollicitanten het filter eruit haalt terwijl ze het werk kunnen doen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wrongly():
    box_wrongly = L.code_box("wrongly", lines=20)
    box_wrongly
    return (box_wrongly,)


@app.cell(hide_code=True)
def check_wrongly(box_wrongly):
    result_wrongly = L.check("wrongly", box_wrongly.value)
    result_wrongly
    return (result_wrongly,)


@app.cell(hide_code=True)
def hints_wrongly():
    L.hints(
        "wrongly",
        [
            "Gebruik de loop en de eerste `if` van het uitgewerkte voorbeeld. In plaats van een naam toe te voegen, tel je: een getal dat bij 0 begint en telkens 1 omhooggaat.",
            "Binnen de eerste `if` vraagt een tweede `if` of `applicant[\"can_do_job\"] == \"yes\"`. Alleen dan gaat de telling omhoog: `wrongly = wrongly + 1`.",
            "Binnen de tweede `if`: `wrongly = ___`.",
        ],
        why="`wrongly = 0`, dan `for applicant in applicants:` met daaronder `if applicant[\"gap\"] > limit:`, daarbinnen `if applicant[\"can_do_job\"] == \"yes\":`, en daarbinnen `wrongly = wrongly + 1`. Dat geeft `3`: Bram, Dewi en Gijs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_concerns():
    mo.md(r"""
    ### Oefening 2. Waar waarschuwde de EEOC voor?

    De EEOC waarschuwde voor drie dingen: uitfilteren, aanpassingen die aangeboden moeten worden, en vragen naar gezondheid die een programma sollicitanten laat beantwoorden. Deel elk geval in bij de waarschuwing waar het bij hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_concerns_cell():
    sort_concerns = L.sort("concerns")
    sort_concerns
    return (sort_concerns,)


@app.cell(hide_code=True)
def check_concerns(sort_concerns):
    result_concerns = L.check("concerns", sort_concerns)
    result_concerns
    return (result_concerns,)


@app.cell(hide_code=True)
def hints_concerns():
    L.hints(
        "concerns",
        [
            "Vraag je bij elk geval af: haalt een regel iemand eruit, laat een verandering iemand meedoen, of laat een programma iemand over de eigen gezondheid vertellen?",
            "Het gat en het geheugenspel zijn voorbeelden van de EEOC van uitfilteren; de lagere kassa is haar voorbeeld van een aanpassing.",
            "Uitfilteren: het gat en \\_\\_\\_. Een aanpassing: hardop en \\_\\_\\_.",
        ],
        why="**Uitfilteren:** het gat en het geheugenspel. **Een aanpassing:** hardop en de lagere kassa. **Een vraag naar gezondheid:** het formulier en medicijnen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_adjust():
    mo.md(r"""
    ### Oefening 3. Ondertitels bij een gesprek
    """)
    return


@app.cell(hide_code=True)
def choice_adjust_cell():
    choice_adjust = L.choice(
        "adjust",
        question="Een sollicitant die slechthorend is, vraagt een Nederlandse werkgever om ondertitels bij een online videogesprek. Wat vraagt de Nederlandse wet van de werkgever?",
        options={
            "nothing": "Niets, want het gesprek is voor iedereen gelijk",
            "right": "Aanpassen, tenzij dat onevenredig belastend is",
            "any": "Alles wat de sollicitant vraagt, wat het ook kost",
        },
        correct="right",
        why={
            "nothing": "Hetzelfde gesprek voor iedereen kan toch iemand buitensluiten. De wet vraagt naar gelang de behoefte doeltreffende aanpassingen. Zie *In Nederland* in *Het idee*.",
            "right": "De sollicitant heeft de behoefte kenbaar gemaakt, en ondertitels zijn geschikt en noodzakelijk om mee te doen; alleen een onevenredige belasting heft de plicht op.",
            "any": "De plicht heeft een grens: een aanpassing die een onevenredige belasting vormt, hoeft niet, en ze moet geschikt en noodzakelijk zijn. Zie *In Nederland* in *Het idee*.",
        },
    )
    choice_adjust
    return (choice_adjust,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) zag je dat een neutrale regel kan discrimineren door zijn effect. Een filter op gaten is zo'n regel, en handicap voegt een plicht toe: niet alleen een regel stoppen die mensen buitensluit, maar ook aanpassingen doen zodat ze kunnen meedoen. Percentages per groep tellen, zoals in [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl), is hier niet genoeg; jouw telling van wie een regel eruit haalt, en waarom, wel. De volgende les, *Bias in elke stap*, volgt waar bias binnen kan komen, van de wereld tot een model dat in gebruik is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_adjust, result_concerns, result_wrongly):
    L.wrap_up(result_wrongly, result_concerns, choice_adjust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
