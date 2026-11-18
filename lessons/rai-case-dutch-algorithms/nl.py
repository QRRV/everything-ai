# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-dutch-algorithms. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-dutch-algorithms", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_profiling():
    quick_profiling = L.choice(
        "quick-profiling",
        question="Een gemeente controleert mensen met een laag inkomen vaker op fraude dan anderen, wat ieder van hen ook deed. Hoe heet dat?",
        options={
            "right": "Risicoprofilering",
            "contest": "Zorgen voor contestability",
            "oversight": "Een mens in de loop die een model controleert",
        },
        correct="right",
        why={
            "right": "Bepalen wie je controleert op grond van iets wat een groep deelt, hier een laag inkomen, in plaats van op grond van wat één persoon deed.",
            "contest": "Contestability gaat over de vraag of mensen een besluit kunnen aanvechten, niet over wie je controleert. Zie *Het idee* in [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl).",
            "oversight": "Een mens in de loop controleert wat een model beslist; hier gaat het erom wie de gemeente überhaupt controleert. Zie *Het idee* in [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl).",
        },
    )
    quick_profiling
    return (quick_profiling,)


@app.cell(hide_code=True)
def quick_check_nationality():
    quick_nationality = L.choice(
        "quick-nationality",
        question="Op de verzonnen scorekaart van de vorige les deden aanvragen A en D precies hetzelfde, maar alleen D werd gecontroleerd. Waarom?",
        options={
            "far": "D had een grotere afstand tussen woning en opvang",
            "right": "D had niet de Nederlandse nationaliteit",
            "staff": "Een medewerker bekeek D eerder dan A",
        },
        correct="right",
        why={
            "far": "Geen van beide kreeg punten voor de afstand: ze verschilden maar in één indicator. Welke? Zie Oefening 1 in [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl).",
            "right": "Met 4 punten voor *not Dutch* kwam D op 7 punten, boven de grens van 6, en A bleef op 3.",
            "staff": "De scorekaart besliste alleen op grond van de punten, voordat er een mens naar keek. Zie Oefening 1 in [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl).",
        },
    )
    quick_nationality
    return (quick_nationality,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De toeslagenaffaire was niet de enige keer dat een Nederlandse organisatie een algoritme liet meebeslissen over mensen. Ook studenten, mensen in de bijstand, taxichauffeurs en hele wijken kregen een score. Telkens konden de mensen om wie het ging de keuzes erachter niet zien: welke features meetelden, of er echt een mens naar keek, en of het systeem überhaupt werkte.

    **Een voorbeeld uit het dagelijks leven.** Een docent kijkt het huiswerk van de leerlingen achter in de klas vaker na dan dat van de leerlingen voorin. Achterin wordt dan vaker overschrijven ontdekt, ook als er voorin net zo vaak wordt overgeschreven: je vindt alleen iets waar je zoekt.
    """)
    return


@app.cell(hide_code=True)
def predict_lawyers():
    guess_lawyers = L.guess(
        "lawyers",
        question="In tien jaar hielpen advocaten 376 studenten die DUO van fraude met hun studiefinanciering beschuldigde. Hoeveel van hen hadden volgens jou een migratieachtergrond?",
        options={
            "half": "Ongeveer de helft",
            "most": "Ongeveer drie op de vier",
            "right": "Bijna allemaal",
        },
        reveal="**367 van de 376**, vonden journalisten van Investico en NOS op 3 in 2023, in de zaken van 32 advocaten. Die zaken hoeven niet typisch te zijn voor alle controles, dus *Het idee* gebruikt aantallen van alle studenten, van het CBS.",
    )
    guess_lawyers
    return (guess_lawyers,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **DUO.** Vanaf 2012 gebruikte DUO, de organisatie die studiefinanciering betaalt, een risicoprofiel om te kiezen welke studenten met een uitwonendenbeurs het met een huisbezoek zou controleren. Drie features maakten de score hoger: een mbo-opleiding in plaats van hbo of wo, een korte afstand tot het adres van de ouders, en een jonge leeftijd. Medewerkers kozen daarna met de hand studenten, vooral uit de hoge scores. Geen van de drie features is iemands afkomst, maar ze stonden er wel voor in de plaats: een mbo-opleiding en een korte afstand tot de ouders kwamen veel vaker voor bij studenten met een niet-Europese migratieachtergrond. In 2024 vond de organisatie Algorithm Audit, met data van het CBS, dat het profiel deze studenten 2,0 keer zo vaak een hoog risico gaf als studenten van Nederlandse herkomst, en dat zij 6,2 keer zo vaak voor een huisbezoek werden gekozen. De Autoriteit Persoonsgegevens noemde het profiel in 2024 discriminerend en onrechtmatig, en merkte op dat DUO het nooit had geëvalueerd. De minister bood in maart 2024 excuses aan, en in november 2024 maakte de volgende minister bekend dat ruim 10.000 oud-studenten hun boetes en terugvorderingen terugkrijgen; daarvoor is 61 miljoen euro gereserveerd.

        **Rotterdam.** Van 2017 tot 2021 zette de gemeente Rotterdam zo'n 30.000 mensen in de bijstand op volgorde met een model dat ieder van hen een score tussen 0 en 1 gaf op grond van 315 features, en de mensen met de hoogste scores werden onderzocht. Ongeveer twintig van die features gingen over taal, zoals hoe goed iemand Nederlands sprak of schreef. Journalisten van Lighthouse Reports kregen het model in handen en publiceerden in 2023 dat mensen die moeite hadden met Nederlands bij de hoogste scores ongeveer 2,2 keer zo sterk vertegenwoordigd waren als onder alle mensen in de bijstand. Rotterdam stopte in 2021 uit voorzorg met het model.

        **Uber en Ola.** Chauffeurs van Uber en Ola uit het Verenigd Koninkrijk stapten in Amsterdam naar de rechter, met een beroep op de AVG. In 2023 oordeelde het gerechtshof dat Uber drie chauffeurs met een volledig automatisch besluit van zijn platform had gehaald: de medewerkers ertussen deden niet meer dan een louter symbolische handeling, en de chauffeurs werden niet gehoord. Uber moest de logica achter de besluiten uitleggen, op straffe van 4.000 euro per dag. Ook de fraudescore van Ola raakte chauffeurs volgens het hof in aanmerkelijke mate, dus ook Ola moest die uitleggen. In [Een mens in de loop](/course/open/rai-human-oversight?lang=nl) zag je dat iemand die vooral ja zegt weinig fouten vangt; hier telde de rechter wat zo iemand deed niet eens als een besluit door een mens.

        **CAS.** Vanaf 2013 in Amsterdam, en later ook in andere delen van het land, gebruikte de politie CAS, het Criminaliteits Anticipatie Systeem. Het deelde een gebied op in vakken van 125 bij 125 meter en voorspelde in welke vakken misdaad zoals woninginbraak te verwachten was, uit politiegegevens zoals eerdere incidenten en bekende verdachten die in de buurt woonden. Een evaluatie van de Politieacademie vond in 2017 geen aanwijzingen dat het de effecten had waarop de politie hoopte, en de teams konden niet zien welke gegevens een voorspelling bepaalden. De politie stopte op 1 december 2025 met CAS, omdat onduidelijk was wat het echt opleverde.

        **Het Algoritmeregister.** Sinds december 2022 kunnen Nederlandse overheidsorganisaties de algoritmes die ze gebruiken beschrijven in het **Algoritmeregister**, op algoritmes.overheid.nl: waar elk algoritme voor is, hoe het werkt, welke data het gebruikt, hoe mensen er toezicht op houden en wat de risico's zijn. Het begon met 109 algoritmes; op 18 november 2026 stonden er 1.606 in. Registreren is nog niet wettelijk verplicht, dus een algoritme dat er niet in staat, kan toch in gebruik zijn. Ook CAS staat erin, nu als buiten gebruik.

        Elk geval draaide om keuzes die de mensen om wie het ging niet konden zien: welke features meetelden, of er echt een mens besliste, en of het systeem werkte. Een register maakt een deel van die keuzes zichtbaar, voordat iemand er voor naar de rechter moet.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **Algoritmeregister** | de openbare lijst waarop Nederlandse overheidsorganisaties de algoritmes die ze gebruiken kunnen beschrijven: waarvoor, hoe ze werken, met welke data, en met welk toezicht en welke risico's |
        """),
        L.watch_out("Een groter deel onder de mensen die gecontroleerd werden, laat niet zien dat een groep vaker fraudeert. Je vindt alleen misbruik waar je zoekt: een groep die vaker gecontroleerd wordt, wordt ook vaker betrapt."),
        mo.Html(C.picture({"title": "Wie er in de groep zat, stap voor stap",
                           "rows": ["Alle studenten met de beurs (248.650)",
                                    "Hoog risico volgens het profiel (42.800)",
                                    "Gekozen voor een huisbezoek (2.810)"],
                           "axis": "deel met een niet-Europese migratieachtergrond (%)", "decimal": ",",
                           "alt": "Drie balken voor de studenten van DUO in 2014. Van alle 248.650 studenten met de beurs had 22,0 procent een niet-Europese migratieachtergrond; van de 42.800 die het profiel een hoog risico gaf 36,0 procent; van de 2.810 die gekozen werden voor een huisbezoek 63,2 procent. Een gestippelde lijn markeert de 22,0 procent van alle studenten."})),
        mo.md(r"""
        De gevulde balk is wat het profiel zelf gaf: het maakte het deel groter, en de selectie met de hand maakte het nog groter. De delen komen uit het rapport van Algorithm Audit uit 2024, voor de studenten die op 1 februari 2014 een uitwonendenbeurs hadden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te vergelijken hoe vaak twee groepen gekozen worden:

    1. Tel de mensen in elke groep.
    2. Tel hoeveel mensen uit elke groep gekozen werden.
    3. Deel per groep de gekozen mensen door alle mensen in de groep: het deel dat gekozen werd.
    4. Deel het ene deel door het andere: hoeveel keer zo vaak de ene groep gekozen wordt.
    5. Doe dit voor elke stap, van de score tot de controle, en zie waar het verschil groeit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De aantallen hier zijn door de course uitgerekend uit de delen in het rapport van Algorithm Audit, voor de 248.650 studenten die op 1 februari 2014 een uitwonendenbeurs hadden. `non-European` staat voor de studenten met een niet-Europese migratieachtergrond, en `everyone else` voor alle andere studenten, dus de uitkomsten hoeven niet precies die van het rapport te zijn: het rapport vergelijkt alleen met studenten van Nederlandse herkomst. `_students` geeft hoeveel studenten uit elke groep de beurs hadden, `_high_risk` hoeveel van hen het profiel een hoog risico gaf. Voer de cel uit.
    """)
    return


@app.cell
def example_high_risk():
    _students = {"non-European": 54703, "everyone else": 193947}
    _high_risk = {"non-European": 15408, "everyone else": 27392}

    _share = {}
    for _group in _students:
        _share[_group] = _high_risk[_group] / _students[_group]
    print(round(_share["non-European"], 2), round(_share["everyone else"], 2))
    print(round(_share["non-European"] / _share["everyone else"], 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het profiel gaf 0,28 van de studenten met een niet-Europese migratieachtergrond een hoog risico, en 0,14 van alle anderen: een twee keer zo groot deel. Python schrijft een punt waar wij een komma schrijven. Stap 5 van het recept vraagt ook om de volgende stap, het huisbezoek: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_visited():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie kreeg het huisbezoek?

        `students` geeft weer hoeveel studenten uit elke groep de beurs hadden, en `visited` hoeveel van hen DUO voor een huisbezoek koos. Vervang de eerste `...` door één regel, zoals in het uitgewerkte voorbeeld, die onder `picked[group]` het deel zet van de studenten uit deze groep die een huisbezoek kregen; de loop gaat met `group` langs de groepen. Vervang de tweede `...` door hoeveel keer zo groot het niet-Europese deel is als het deel van alle anderen, zonder `round`. De laatste regel is `times`, zodat de check het ziet. De check probeert ook andere aantallen.
        """),
        L.task("Zet in `times` hoeveel keer zo vaak een student met een niet-Europese migratieachtergrond een huisbezoek kreeg."),
    ])
    return


@app.cell(hide_code=True)
def exercise_visited():
    box_visited = L.code_box("visited", lines=10)
    box_visited
    return (box_visited,)


@app.cell(hide_code=True)
def check_visited(box_visited):
    result_visited = L.check("visited", box_visited.value)
    result_visited
    return (result_visited,)


@app.cell(hide_code=True)
def hints_visited():
    L.hints(
        "visited",
        [
            "Het uitgewerkte voorbeeld doet hetzelfde voor wie het profiel een hoog risico gaf. Welk aantal komt hier in de plaats van `_high_risk`?",
            "Onder de streep komen alle studenten van de groep, `students[group]`. Voor `times` haal je de twee delen met hun keys uit `picked`.",
            "`picked[group] = ___[group] / students[group]`, en na de loop `times = picked[\"non-European\"] / ___`.",
        ],
        why="`picked[group] = visited[group] / students[group]` in de loop, en `times = picked[\"non-European\"] / picked[\"everyone else\"]` erna. Dat geeft `6.09`, ongeveer 6,1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ### Oefening 2. Drie vragen

    Zes feiten uit de vier gevallen, elk over een van drie vragen. Zet elk feit bij de vraag waar het over gaat.
    """)
    return


@app.cell(hide_code=True)
def sort_unseen_cell():
    sort_unseen = L.sort("unseen")
    sort_unseen
    return (sort_unseen,)


@app.cell(hide_code=True)
def check_unseen(sort_unseen):
    result_unseen = L.check("unseen", sort_unseen)
    result_unseen
    return (result_unseen,)


@app.cell(hide_code=True)
def hints_unseen():
    L.hints(
        "unseen",
        [
            "Vraag bij elk feit: gaat het over wat er in de score ging, over wie het besluit nam, of over wat het systeem opleverde?",
            "Twee feiten gaan over wat een model binnenkreeg, twee over Uber en Ola, en twee over een evaluatie.",
            "Wat meetelde: de afstand en \\_\\_\\_. Of een mens besliste: \\_\\_\\_ en de kortingen. Of het werkte: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Wat er meetelde in de score:** de afstand en de taal. **Of er echt een mens besliste:** het akkoord en de kortingen. **Of het systeem werkte:** de evaluatie en de effecten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_register():
    mo.md(r"""
    ### Oefening 3. Wat staat er in het register?
    """)
    return


@app.cell(hide_code=True)
def choice_register_cell():
    choice_register = L.choice(
        "register",
        question="Een gemeente gaat een model gebruiken dat kiest wie ze op bijstandsfraude controleert, en beschrijft het in het Algoritmeregister. Wat kun je daar lezen?",
        options={
            "score": "De score die het model jou gaf, en waar je bezwaar kunt maken",
            "proof": "Een bewijs dat het model niet discrimineert, gecontroleerd en goedgekeurd door de overheid",
            "right": "Waar het model voor is, welke data het gebruikt en hoe er toezicht op is",
        },
        correct="right",
        why={
            "score": "Het register beschrijft het algoritme zelf, niet wat het over één persoon besliste. Zie *Het idee*.",
            "proof": "Een beschrijving komt van de organisatie die het algoritme gebruikt, en is geen test ervan. Bij DUO was er een apart onderzoek met data van het CBS nodig om het verschil te laten zien. Zie *Het idee*.",
            "right": "En wat de risico's zijn. Registreren is nog niet wettelijk verplicht, dus niet elk algoritme staat erin.",
        },
    )
    choice_register
    return (choice_register,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De toeslagenaffaire en SyRI](/course/open/rai-case-toeslagen-syri?lang=nl) zag je op een verzonnen scorekaart hoe één indicator over wie iemand is kan bepalen wie er gecontroleerd wordt. Het profiel van DUO keek nooit naar afkomst, maar met features die ervoor in de plaats stonden, en een selectie met de hand na de score, kregen studenten met een niet-Europese migratieachtergrond ongeveer zes keer zo vaak een huisbezoek; een latere les, *Een kolom weglaten is niet genoeg*, laat zien waarom een feature weglaten niet genoeg is. In de volgende les, *Robodebt, de A-levels en meer*, bekijk je gevallen in het buitenland.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_register, result_unseen, result_visited):
    L.wrap_up(result_visited, result_unseen, choice_register)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
