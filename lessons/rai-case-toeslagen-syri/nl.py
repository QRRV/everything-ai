# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-toeslagen-syri. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-toeslagen-syri", "nl")
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
def quick_check_weigh():
    quick_weigh = L.choice(
        "quick-weigh",
        question="Bij welk van deze besluiten weeg je omstandigheden af, in plaats van een feit te controleren?",
        options={
            "fee": "Of een bedrag op tijd is betaald",
            "right": "Of iemand opzettelijk misleidde",
            "age": "Of een aanvrager 18 jaar of ouder is",
        },
        correct="right",
        why={
            "fee": "Of een bedrag betaald is, is een feit dat een systeem elke keer op dezelfde manier kan controleren. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
            "right": "Of iemand opzettelijk misleidde, hangt af van deze persoon en dit geval: daarvoor is iemand nodig die de omstandigheden kan zien.",
            "age": "Een leeftijd is een feit dat een systeem elke keer op dezelfde manier kan controleren. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
        },
    )
    quick_weigh
    return (quick_weigh,)


@app.cell(hide_code=True)
def quick_check_contest():
    quick_contest = L.choice(
        "quick-contest",
        question="Wat betekent *contestability*?",
        options={
            "accuracy": "Hoe vaak de besluiten van een systeem achteraf goed blijken te zijn",
            "right": "Hoe goed mensen een besluit kunnen aanvechten en laten veranderen",
            "speed": "Hoe snel een besluit genomen wordt",
        },
        correct="right",
        why={
            "accuracy": "Een systeem dat vaak gelijk heeft, kan mensen toch geen enkele manier geven om bezwaar te maken. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
            "right": "Daarvoor moeten mensen weten dat er een besluit is genomen en waarom, waar ze bezwaar kunnen maken, genoeg tijd hebben, en iemand vinden die het besluit kan veranderen.",
            "speed": "Snelheid zegt niets over de vraag of mensen bezwaar kunnen maken. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
        },
    )
    quick_contest
    return (quick_contest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In januari 2021 trad het kabinet af om de manier waarop de Belastingdienst ouders met kinderopvangtoeslag had behandeld. Een risicomodel speelde daarbij een rol, en de regels en de mensen eromheen ook. Een jaar eerder had de rechter de wet achter SyRI buiten werking gesteld. SyRI zocht fraude door data van veel overheidsinstanties aan elkaar te koppelen. Samen laten de twee gevallen zien wat er misgaat als niemand kan zien hoe mensen worden uitgekozen, of wat het ze kost.

    **Een voorbeeld uit het dagelijks leven.** Een winkel controleert de tas van elke klant met een grote jas. De meeste klanten met een grote jas stelen niets, en de winkel zegt nooit waarom hij ze aanhoudt. Kost een ontbrekend bonnetje dan ook nog een boete ter waarde van een jaar boodschappen, dan is de regel over jassen niet meer het enige probleem.
    """)
    return


@app.cell(hide_code=True)
def predict_label():
    guess_label = L.guess(
        "label",
        question="Hoeveel mensen kregen volgens jou van 2012 tot begin 2020 van de Belastingdienst het label *opzet/grove schuld*, waardoor ze geen betalingsregeling konden krijgen?",
        options={
            "hundreds": "Een paar honderd mensen",
            "thousands": "Ongeveer 3.000 mensen",
            "right": "25.000 tot 35.000",
        },
        reveal="**25.000 tot 35.000**, meldde de staatssecretaris in maart 2020 aan de Tweede Kamer. Op 4 december 2020 hoorde de Kamer dat in een steekproef 94 van elke 100 van die labels niet meer als terecht werden beoordeeld.",
    )
    guess_label
    return (guess_label,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Risicoprofilering** is bepalen wie je controleert aan de hand van eigenschappen die een groep mensen deelt, in plaats van op grond van iets wat één persoon deed. De Belastingdienst, banken en gemeenten doen het allemaal. Waar het om gaat: welke eigenschappen meetellen, of iemand ze kan zien, en wat er gebeurt met de mensen die worden uitgekozen.

        **De kinderopvangtoeslag.** Van 2016 tot 2018 kregen per jaar meer dan een half miljoen huishoudens kinderopvangtoeslag van de Belastingdienst. Vanaf 2013 gaf een zelflerend model elke aanvraag een risicoscore tussen 0 en 1, op basis van enkele tientallen indicatoren, en medewerkers controleerden de aanvragen met de hoogste scores. Ze wisten niet waarom het model een aanvraag had gekozen. Een van de indicatoren was of de aanvrager de Nederlandse nationaliteit had of niet. De Autoriteit Persoonsgegevens vond die indicator in het model van in ieder geval maart 2016 tot oktober 2018, noemde dat in 2020 onrechtmatig en discriminerend, en gaf de Belastingdienst in 2021 een boete van 2,75 miljoen euro voor dit en twee andere onrechtmatige verwerkingen van nationaliteit.

        Om het model heen stonden harde regels. Een kleine fout, zoals een ontbrekende handtekening, kon betekenen dat ouders de toeslag van een heel jaar moesten terugbetalen. Tussen 2012 en 2020 kregen 25.000 tot 35.000 mensen het label *opzet/grove schuld*: ze konden geen betalingsregeling krijgen, en andere overheidsinstanties zagen hen als moedwillige fraudeurs. In december 2020 noemde een parlementaire ondervragingscommissie het *ongekend onrecht*. Halverwege 2026 waren meer dan 44.000 ouders erkend als gedupeerd.

        **SyRI**, Systeem Risico Indicatie, koppelde data over inkomen, wonen, schulden, uitkeringen en meer, van gemeenten, het UWV, de SVB, de Belastingdienst en anderen, en een risicomodel met punten per indicator wees mensen en adressen aan die het onderzoeken waard zouden zijn. Vanaf 2015 werd het in vijf projecten gebruikt, allemaal in wijken die de overheid probleemwijken noemde. Op 5 februari 2020 oordeelde de rechtbank Den Haag dat de wet achter SyRI in strijd was met het recht op respect voor het privéleven uit het Europees Verdrag voor de Rechten van de Mens. Het risicomodel en de indicatoren waren geheim, dus niemand, zelfs de rechtbank niet, kon controleren hoe SyRI mensen koos, en wie werd aangewezen, kon zich nauwelijks verweren. Of SyRI discrimineerde, kon de rechtbank niet zeggen, omdat ze er niet in kon kijken. De Staat ging niet in hoger beroep.

        Beide gevallen laten zien wat er misgaat zonder transparantie en zonder een echte manier om bezwaar te maken.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **risicoprofilering** | bepalen wie je controleert, bijvoorbeeld op fraude, aan de hand van eigenschappen die een groep mensen deelt, zoals inkomen, adres of nationaliteit, in plaats van op grond van wat één persoon deed |
        """),
        L.watch_out("Nationaliteit eruit halen is niet genoeg. Andere indicatoren, zoals een adres of een soort opleiding, kunnen ervoor in de plaats komen. De volgende les laat zo'n geval zien."),
        mo.Html(C.picture({"title": "Van een aanvraag naar een label",
                           "steps": ["Een ouder vraagt kinderopvangtoeslag aan.",
                                     "Een zelflerend model geeft de aanvraag een risicoscore tussen 0 en 1.",
                                     "Medewerkers controleren de aanvragen met de hoogste scores, zonder te weten waarom die gekozen zijn.",
                                     "Een kleine fout, zoals een ontbrekende handtekening, kan betekenen dat het hele jaar terug moet.",
                                     "Het label opzet/grove schuld: geen betalingsregeling."],
                           "model": "het model", "rules": "regels en mensen",
                           "alt": "Vijf stappen. 1, een ouder vraagt kinderopvangtoeslag aan. 2, een zelflerend model geeft de aanvraag een risicoscore tussen 0 en 1. 3, medewerkers controleren de aanvragen met de hoogste scores, zonder te weten waarom die gekozen zijn. 4, een kleine fout, zoals een ontbrekende handtekening, kan betekenen dat het hele jaar terug moet. 5, het label opzet/grove schuld: geen betalingsregeling. Stap 2 is het model; stap 3 tot en met 5 zijn regels en mensen."})),
        mo.md(r"""
        Alleen stap 2 is het model. De schade kwam ook uit de regels en de mensen eromheen. Het beeld laat één manier zien waarop een aanvraag bij het label kon uitkomen; de Autoriteit Persoonsgegevens kon niet zien wat de indicator voor nationaliteit met individuele aanvragers deed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bekijk je een risicoprofiel voordat het gebruikt wordt:

    1. Zet de indicatoren op een rij, met de punten die elke indicator geeft.
    2. Vraag bij elke indicator of die iets zegt over wat iemand deed, of alleen over wie iemand is.
    3. Bereken de score van een paar aanvragen die maar in één indicator verschillen, en kijk wie over de grens gaat.
    4. Controleer of de mensen die worden uitgekozen kunnen horen waarom, en bezwaar kunnen maken.
    5. Schrijf op wat uitgekozen worden iemand kost, ook als de controle niets oplevert.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het echte model had enkele tientallen indicatoren; er zijn er maar twee openbaar, de nationaliteit en de afstand tussen woning en opvang. De scorekaart hier is verzonnen door de course, met drie indicatoren, en een aanvraag met 6 punten of meer gaat naar een mens voor controle. `_points` geeft de punten van elke indicator, `_application` of elke indicator geldt voor één aanvraag. Voer de cel uit.
    """)
    return


@app.cell
def example_score():
    _points = {"distance": 2, "income change": 3, "not Dutch": 4}
    _application = {"distance": True, "income change": False, "not Dutch": True}

    _score = 0
    for _indicator in _points:
        if _application[_indicator]:
            _score = _score + _points[_indicator]
    print(_score)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Deze aanvraag heeft 6 punten: genoeg voor een controle. Stap 3 van het recept vraagt om meer aanvragen die maar in één indicator verschillen: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie gaat over de grens?

        Vier aanvragen, verzonnen door de course. `points` geeft de punten van elke indicator, en `applications` geeft per aanvraag of elke indicator geldt. Vervang de `...` door de body van `risk_score`, die één `application` krijgt: de regels van het uitgewerkte voorbeeld, met de namen zonder liggend streepje, zoals `score` en `indicator`, en geef aan het eind de score terug in plaats van die te printen. In de functie kun je `points` gewoon gebruiken. De loop eronder gaat met `name` langs de aanvragen en zet elke score in `scores`, en de laatste regel is `scores`, zodat de check het ziet. De check probeert ook andere punten.
        """),
        L.task("Laat `risk_score` de som teruggeven van de punten van de indicatoren die voor de aanvraag gelden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=24)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "Het uitgewerkte voorbeeld telt de punten van één aanvraag op. Zet dezelfde regels in de functie, en geef aan het eind de score terug.",
            "Begin `score` bij 0, vóór de loop. In de loop vraagt een `if` of de indicator voor `application` geldt. Waar hoort `return`: in de loop, of erna?",
            "`if application[indicator]:` en daaronder `score = score + ___[indicator]`; na de loop `return ___`.",
        ],
        why="`score = 0`, dan `for indicator in points:` met daarin `if application[indicator]:` en `score = score + points[indicator]`, en na de loop `return score`. Dat geeft `{'A': 3, 'B': 5, 'C': 9, 'D': 7}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Oefening 2. Wat ging er mis?

    Zes feiten uit de twee gevallen. Zet elk feit bij wat er misging.
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
            "Vraag bij elk feit: gaat het over wat niemand kon zien, over wie er gekozen kon worden, of over wat er na een fout gebeurde?",
            "Twee feiten gaan over SyRI, vier over de kinderopvangtoeslag. Twee van de feiten over de toeslag gaan over geld en wat daaruit volgde.",
            "Niemand kon zien: de medewerkers en \\_\\_\\_. Wie er werd gekozen: \\_\\_\\_ en de wijken. Wat een fout kostte: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Niemand kon zien hoe het werkte:** de medewerkers en de rechtbank. **Wie er werd gekozen:** de nationaliteit en de wijken. **Wat een kleine fout kostte:** de handtekening en het label.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ruling():
    mo.md(r"""
    ### Oefening 3. Waarom zette de rechter de wet achter SyRI buiten werking?
    """)
    return


@app.cell(hide_code=True)
def choice_ruling_cell():
    choice_ruling = L.choice(
        "ruling",
        question="Op welke grond zette de rechtbank Den Haag in 2020 de wet achter SyRI buiten werking?",
        options={
            "errors": "Het was bewezen dat SyRI vooral onschuldige mensen aanwees",
            "gdpr": "De rechtbank vond dat SyRI de regels van de AVG over geautomatiseerde besluiten overtrad",
            "right": "Ze schond het recht op privéleven: niemand kon SyRI controleren",
        },
        correct="right",
        why={
            "errors": "De rechtbank kon niet in SyRI kijken, en kon dus ook niet zeggen hoe vaak het zich vergiste. Zie *Het idee*.",
            "gdpr": "De rechtbank deed geen uitspraak over de AVG. Welk recht noemde ze wel? Zie *Het idee*.",
            "right": "Artikel 8 van het Europees Verdrag voor de Rechten van de Mens: de wet liet niemand zien of controleren hoe SyRI mensen koos, dus wie werd aangewezen, kon zich nauwelijks verweren. De rechtbank vond ook dat de wet niet beperkte welke data gekoppeld mochten worden.",
        },
    )
    choice_ruling
    return (choice_ruling,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl) zag je dat het afwegen van omstandigheden iemand vraagt die het geval kan zien, en dat mensen een manier nodig hebben om bezwaar te maken. In de toeslagenaffaire ging een label waarvoor iemand opzet had moeten afwegen naar tienduizenden mensen, en bleek het grootste deel van die labels later onterecht; bij SyRI kon niemand zien waartegen je bezwaar moest maken. In de volgende les, *Meer algoritmes die over mensen oordeelden*, bekijk je meer Nederlandse gevallen, en een register waarin overheidsorganisaties de algoritmes die ze gebruiken kunnen beschrijven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ruling, result_failures, result_score):
    L.wrap_up(result_score, result_failures, choice_ruling)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
