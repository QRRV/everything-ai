# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-fairness-in-organizations. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fairness-in-organizations", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="De selectiepercentages van een model zijn 0,30 voor vrouwen, 0,50 voor mannen en 0,40 voor een derde groep. De demographic parity ratio van Fairlearn deelt het kleinste selectiepercentage door het grootste. Wat is die?",
        options={
            "right": "0,6",
            "third": "0,75",
            "top": "0,8",
        },
        correct="right",
        why={
            "right": "0,30 / 0,50 = 0,6: het kleinste selectiepercentage gedeeld door het grootste, over alle groepen.",
            "third": "Dat is 0,30 / 0,40, de vrouwen tegenover de derde groep. De ratio neemt het kleinste en het grootste percentage van alle groepen: 0,30 / 0,50. Zie [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
            "top": "Dat is 0,40 / 0,50, de derde groep tegenover de mannen. De ratio neemt het kleinste en het grootste percentage van alle groepen: 0,30 / 0,50. Zie [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="Over tien steekproeven van 1000 mensen uit UCI Adult bracht reweighing het deel met een ja van vrouwen gedeeld door dat van mannen 0,07 tot 0,51 omhoog. Een paper geeft één zo'n steekproef, met een stijging van 0,45. Wat zegt dat ene getal je?",
        options={
            "exact": "Precies hoeveel de methode voor iedereen helpt",
            "right": "Weinig, een andere splitsing verschilt sterk",
            "fails": "Dat de methode helemaal niet werkt",
        },
        correct="right",
        why={
            "exact": "Eén splitsing van een kleine tabel zegt weinig: over tien splitsingen liep de stijging van 0,07 tot 0,51. Zie [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
            "right": "Een andere steekproef had 0,07 kunnen geven. Rapporteer hoe ver de uitkomsten over meerdere splitsingen uit elkaar liggen, en gebruik meer dan één tabel.",
            "fails": "Een stijging van 0,45 is geen teken dat het mislukt. Maar één splitsing van een kleine tabel zegt hoe dan ook weinig. Zie [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Deze module ging over modellen meten en verbeteren. In de praktijk loopt werk aan fairness vaak al vast voordat iemand een notebook opent. Kenneth Holstein en collega's (2019) interviewden 35 mensen die in het bedrijfsleven systemen voor machine learning bouwen, en hielden een enquête onder 267 mensen uit het vak. Maar 21 procent zei dat hun team fairness 'veel' of 'heel veel' voorrang geeft; 36 procent zei 'helemaal niet'. Veel teams keken pas naar fairness nadat een klant had geklaagd, terwijl ze naar beveiligingsrisico's keken voordat er iets misging. Een van de mensen die Michael Madaio en collega's (2020) spraken, vertelde wat er gebeurt als er geen vast moment voor is: er is geen controlepunt waarop iemand iets hoort te zeggen, dus je kunt het alleen doen door steeds weer de lastpost te zijn.

    **Een voorbeeld uit het dagelijks leven.** In een restaurantkeuken moet iemand de temperatuur van de koelkasten controleren. Als niemand die controle als taak heeft en er geen vaste tijd voor is, gebeurt het de dag nadat een gast ziek is geworden. Met een lijst op de deur die zegt wie controleert, wanneer, en wat er gebeurt als een koelkast te warm is, gebeurt het elke ochtend.
    """)
    return


@app.cell(hide_code=True)
def predict_suspect():
    guess_suspect = L.guess(
        "suspect",
        question="Holstein en collega's vroegen mensen uit de praktijk of hun team problemen met fairness in zijn producten had gevonden. Hoeveel van wie nee zei, vermoedde dat er problemen konden zijn die ze nog niet hadden ontdekt?",
        options={
            "few": "Minder dan een op de vijf",
            "half": "Ongeveer de helft",
            "right": "Vier op de vijf",
        },
        reveal="**Vier op de vijf.** Van de mensen die de vraag kregen, zei bijna de helft, 49 procent, dat hun team mogelijke problemen met fairness had gevonden. Van de 51 procent bij wie dat niet zo was, vermoedde het grootste deel, 80 procent, problemen die ze nog niet hadden ontdekt. Let op de noemer: maar 63 procent van de ondervraagden kreeg deze vraag, en elk percentage telt alleen wie hem kreeg.",
    )
    guess_suspect
    return (guess_suspect,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Waar het vastloopt.** De meeste mensen die Holstein en collega's interviewden, zagen het verzamelen van data, niet het bouwen van het model, als de belangrijkste plek om iets te verbeteren; een engineer noemde het verzamelen van data in het eigen team 'bijna het Wilde Westen'. Veel teams konden gegevens over groepen alleen grof verzamelen, of helemaal niet, en 70 procent van de 69 procent die die vraag kreeg, vond tools om fairness te controleren zonder gegevens over groepen per persoon minstens 'erg' nuttig. Madaio en collega's (2022) werkten met 33 mensen uit tien teams bij drie techbedrijven. Die vonden het moeilijk om de maten te kiezen, om te weten op welke mensen en groepen ze moesten letten, en om data te verzamelen om per groep te evalueren, en hun keuzes volgden de bedrijfsbelangen: als de organisatie alleen over bedrijfsresultaten praat, geeft het project voorrang aan de groepen die voor het bedrijf het belangrijkst zijn. McKane Andrus en collega's (2021) interviewden 38 mensen uit 26 organisaties. Bijna iedereen noemde toegang tot gegevens over groepen een grote drempel. Bedrijven die zelf data verzamelden, hadden bijna allemaal gender en leeftijd; buiten werk, zorg en sommige financiële diensten hadden weinigen gegevens over ras, dus de meesten controleerden hun systemen niet op bias naar ras. Sommigen moesten een analyse van bias rechtvaardigen met verwachte winst op prestatie-indicatoren, en als dat mislukte, kon het werk gewoon stoppen. Andrus en collega's concluderen niet dat teams gewoon meer gegevens over mensen moeten verzamelen: dat roept eigen vragen op, zoals privacy, toestemming, welke indeling je gebruikt, en of een bedrijf zulke gegevens wel mag raden.

        **Reactief, proactief, anticiperend.** Bogdana Rakova en collega's (2021) interviewden eind 2019 26 mensen uit 19 organisaties op vier continenten: veel organisaties werken reactief, minder zijn inmiddels proactief, en de geïnterviewden hopen dat hun organisatie in de toekomst anticiperend wordt. In reactieve organisaties was de meest gehoorde aanleiding om iets te doen rampzalige aandacht in de media, en iemand vroeg in vergaderingen: 'Wat als ProPublica hierachter kwam?' Ongeveer 40 procent, 11 van de 26, stak vrijwillig tijd in werk aan responsible AI, buiten de eigen functie. Een paar organisaties zijn proactief: ze hebben vaste processen om hun systemen te evalueren en te beoordelen, met het werk en de verantwoordelijkheid vaak verdeeld over meerdere teams, en sommige hebben een interne commissie die erop toeziet. Elke organisatie, schrijven ze, heeft een manier nodig om vier vragen te beantwoorden: wanneer en hoe handelen we, hoe meten we succes, op welke interne structuren leunen we, en hoe lossen we spanningen op? Het is cruciaal, schrijven ze, om eerst te vragen of een systeem wel gepast is, en een besluit om te stoppen met ontwikkelen moet gerespecteerd worden en mag niet door andere overwegingen opzij worden gezet.

        **Pauzemomenten.** Madaio en collega's (2020) ontwierpen een checklist voor fairness samen met 48 mensen uit 12 techbedrijven. Die volgt de fasen van een product: envision, define, prototype, build, launch en evolve. Uit de luchtvaart en de geneeskunde namen ze de les over dat een checklist gebruikt wordt als hij aan de goede momenten vastzit, pauzemomenten (*pause points*). Elke fase noemt dus momenten die teams al hebben: bijeenkomsten om het idee te verkennen en pre-mortems, waarin een team vooraf bedenkt hoe het product mis kan gaan, besprekingen van de specificaties en van het ontwerp, go/no-go-besprekingen en code reviews, de ship review vóór de lancering, en vaste productbesprekingen erna. Een van de punten is: overweeg de ontwikkeling af te breken. De checklist waarschuwt zelf dat het afwerken van de punten geen fairness garandeert: ze zijn bedoeld om gesprek en reflectie op gang te brengen. De auteurs voegen toe dat een checklist de indruk kan wekken dat een eenvoudig nalevingsproces fairness garandeert.

        **Een fairness review.** Geen van deze papers definieert het begrip; de course gebruikt het voor wat ze samen beschrijven. Een *fairness review* is een gepland moment in het werk aan een product waarop een team met verschillende rollen het systeem per groep meet, opschrijft wie schade kan oplopen en wat eraan gedaan wordt, en iemand met de bevoegdheid daartoe besluit om door te gaan, het te veranderen of te stoppen. Gepland: het staat in de planning, net als een code review, en wacht niet op een klacht. Verschillende rollen: de mensen die het systeem bouwen, iemand die de wet kent, iemand die de mensen kent voor wie het gevolgen heeft. Een besluit: en een besluit om te stoppen houdt stand.

        **In Nederland en de EU.** De Nederlandse overheid heeft een eigen manier voor zo'n review: het IAMA, het Impact Assessment Mensenrechten en Algoritmes, gemaakt door de Universiteit Utrecht in opdracht van het ministerie van Binnenlandse Zaken. Versie 2 verscheen op 16 februari 2026; versie 1 is van 2021. Het noemt zichzelf 'een instrument voor discussie en besluitvorming' en 'geen afvinklijstje'. Het raadt een team van 4 tot 7 mensen aan, met minstens een projectleider, een data scientist en een jurist. De vijf delen vragen waarom, wat en hoe, lopen dan de mensenrechten langs, en sluiten af; onderweg vraagt het welke aannames en bias in de data zitten die het algoritme gebruikt, en of het algoritme een direct of indirect onderscheid maakt dat onder de gelijkebehandelingswetgeving valt. Het team schrijft een advies over het al dan niet inzetten van het algoritme en over de voorwaarden; wie er inhoudelijk of politiek verantwoordelijk voor is, beslist. Houdt het deel over mensenrechten geen stand, dan is het advies terug te gaan naar de tekentafel. Versie 2 bevat wat de AI Act vraagt voor de beoordeling van de gevolgen voor de grondrechten (FRIA). Die geldt voor overheidsorganisaties, private partijen die openbare diensten verlenen, en gebruiksverantwoordelijken van systemen die de kredietwaardigheid van mensen scoren of het risico en de prijs van levens- en ziektekostenverzekeringen bepalen. Voordat ze zo'n AI-systeem met een hoog risico voor het eerst gebruiken, beschrijven ze hoe en hoe vaak ze het gebruiken, welke groepen er gevolgen van kunnen merken, welke risico's op schade er zijn, welk menselijk toezicht er is, en wat ze doen als de risico's zich voordoen, met klachtenregelingen. De uitkomsten melden ze bij de markttoezichthouder. Kritieke infrastructuur valt erbuiten. Voor deze systemen gelden de regels vanaf 2 december 2027. En sinds december 2022 kunnen Nederlandse overheden hun algoritmes beschrijven in het openbare Algoritmeregister, algoritmes.overheid.nl; publiceren is daar nog niet verplicht.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **fairness review** | een gepland moment in het werk aan een product waarop een team met verschillende rollen het systeem per groep meet, opschrijft wie schade kan oplopen en wat eraan gedaan wordt, en iemand met bevoegdheid besluit door te gaan, te veranderen of te stoppen; de werkdefinitie van de course |
        | **pauzemoment** | een moment dat een team al heeft, zoals een ontwerpbespreking of de ship review, waarop een checklist gebruikt wordt |
        | **reactief** en **proactief** | handelen na klachten of slechte publiciteit, of met vaste processen om een systeem vooraf te evalueren en te beoordelen (Rakova en collega's) |
        | **IAMA** | het Nederlandse Impact Assessment Mensenrechten en Algoritmes: een gestructureerd gesprek dat eindigt in een advies over het al dan niet inzetten van een algoritme |
        | **FRIA** | de beoordeling van de gevolgen voor de grondrechten die de AI Act vraagt van sommige gebruiksverantwoordelijken van AI-systemen met een hoog risico |
        """),
        L.watch_out("Een review is geen vakje om af te vinken. De checklist van Madaio en collega's zegt dat het afwerken van de punten geen fairness garandeert, en het IAMA noemt zichzelf geen afvinklijstje: allebei zijn ze er om een team te helpen praten en besluiten. Een review die nooit iets verandert of stopt, is een ritueel."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zet je fairness reviews op in een team:

    1. Zet ze in de planning: kies momenten die het team al heeft, zoals een ontwerpbespreking, een go/no-go-bespreking en de ship review, en één na de lancering.
    2. Noem per moment wie het leidt en wie meedoet: de mensen die het systeem bouwen, iemand die de wet kent, iemand die de mensen kent voor wie het gevolgen heeft.
    3. Spreek vóór je kijkt af wat je meet, voor welke groepen, en hoe groot een verschil mag zijn; zeg waar de gegevens over groepen vandaan komen, of waarom je ze niet hebt.
    4. Schrijf op wie schade kan oplopen en wat er gebeurt: het systeem veranderen, menselijk toezicht toevoegen, of stoppen.
    5. Laat iemand met de bevoegdheid beslissen, en laat een besluit om te stoppen standhouden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het plan van een team, verzonnen door de course: de momenten waarop het naar fairness kijkt, elk met een fase van de checklist en wie het leidt. De loop telt per fase hoeveel momenten erin vallen. Voer de cel uit.
    """)
    return


@app.cell
def example_plan():
    # de fasen van de checklist, op volgorde, en het plan van een team
    _stages = ["envision", "define", "prototype", "build", "launch", "evolve"]
    _moments = [
        {"moment": "design review", "stage": "define", "lead": "designer"},
        {"moment": "code review", "stage": "build", "lead": "engineer"},
        {"moment": "go/no-go discussion", "stage": "build", "lead": "product manager"},
        {"moment": "ship review", "stage": "launch", "lead": "director"},
    ]

    for _stage in _stages:
        _count = 0
        for _moment in _moments:
            if _moment["stage"] == _stage:
                _count = _count + 1
        print(_stage, _count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie fasen krijgen 0: envision, prototype en evolve. Niemand vraagt of het systeem gebouwd moet worden, en niemand kijkt na de lancering opnieuw, dus een probleem daar komt pas boven als iemand klaagt. Welke fasen slaat een ander plan over? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Fasen zonder moment

        Reken `missing` uit: de fasen, in de volgorde van `stages`, waarin geen moment van `moments` valt. Begin met `missing = []` en loop met `stage` over `stages`. Tel voor elke fase de momenten erin, zoals in het uitgewerkte voorbeeld: `count = 0`, een loop met `moment` over `moments`, en `count = count + 1` als `moment["stage"] == stage`. Is na die binnenste loop `count == 0`, doe dan `missing.append(stage)`. De check probeert ook een ander plan.
        """),
        L.task("Reken `missing` uit: de fasen waarin geen moment van het plan valt, in de volgorde van `stages`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=20)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "Net als in het uitgewerkte voorbeeld: een loop over de fasen, en daarbinnen een loop over de momenten die telt. Deze keer print je de telling niet; je houdt de fasen bij waar die 0 blijft.",
            "Vóór de binnenste loop `count = 0`; daarin `if moment[\"stage\"] == stage:` met `count = count + 1`. Na de binnenste loop, op de hoogte van `for moment`, kijk je `if count == 0:`.",
            "Onder `if count == 0:`: `missing.append(___)`.",
        ],
        why="`missing = []`, dan `for stage in stages:` met daaronder `count = 0`, en `for moment in moments:` met `if moment[\"stage\"] == stage:` en `count = count + 1`. Na de binnenste loop `if count == 0:` met `missing.append(stage)`. Dat geeft `['define', 'prototype', 'evolve']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proactive():
    mo.md(r"""
    ### Oefening 2. Vooraf of achteraf?
    """)
    return


@app.cell(hide_code=True)
def choice_proactive_cell():
    choice_proactive = L.choice(
        "proactive",
        question="Welke van deze maakt het werk van een team aan fairness proactief in plaats van reactief, in de woorden van Rakova en collega's?",
        options={
            "complaint": "Elk probleem oplossen dat een klant meldt",
            "right": "Een review op vaste momenten vooraf",
            "values": "Een verklaring over waarden op de website van het bedrijf",
        },
        correct="right",
        why={
            "complaint": "Handelen op klachten is de reactieve manier: het probleem heeft mensen al bereikt. Zie *Reactief, proactief, anticiperend* bij *Het idee*.",
            "right": "Proactieve organisaties hebben vaste processen om hun systemen vooraf te evalueren en te beoordelen, met het werk en de verantwoordelijkheid verdeeld over meerdere teams. Reactieve organisaties handelen na klachten of slechte publiciteit.",
            "values": "Een verklaring zegt wat een bedrijf wil, niet wanneer iemand controleert. Proactief werk heeft vaste momenten en mensen bij wie ze horen. Zie *Reactief, proactief, anticiperend* bij *Het idee*.",
        },
    )
    choice_proactive
    return (choice_proactive,)


@app.cell(hide_code=True)
def text_exercise_behind():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Welke groepen hebben werk nodig?

        In een review vóór de lancering kijkt een team per groep naar de eigen spraakherkenner. Stel dat die de word error rates in `wer` had, die Allison Koenecke en collega's (2020) gemiddeld maten voor vijf systemen voor spraakherkenning: de fouten in een transcriptie per gezegd woord, dus lager is beter. Vóór het kijken spraken ze `limit` af: hoe ver een groep boven de WER van de beste groep mag liggen. Reken `behind` uit: de groepen, in de volgorde van `wer`, waarvan de WER boven `best + limit` ligt, waarbij `best` de laagste WER is. Verzamel eerst de waarden in een lijst met een loop over `wer` en neem `best = min(values)`; begin dan met `behind = []` en loop opnieuw over `wer`. De check probeert ook een kleinere grens.
        """),
        L.task("Reken `behind` uit: de groepen waarvan de WER meer dan `limit` boven de laagste WER ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_behind():
    box_behind = L.code_box("behind", lines=16)
    box_behind
    return (box_behind,)


@app.cell(hide_code=True)
def check_behind(box_behind):
    result_behind = L.check("behind", box_behind.value)
    result_behind
    return (result_behind,)


@app.cell(hide_code=True)
def hints_behind():
    L.hints(
        "behind",
        [
            "Twee stappen: eerst de WER van de beste groep, de laagste; dan elke groep die meer dan `limit` daarboven ligt.",
            "Een loop `for group in wer:` geeft de namen; `wer[group]` geeft de WER. Verzamel de WER's met `values.append(wer[group])`, en neem `best = min(values)`.",
            "In de tweede loop: `if wer[group] > ___:` met daaronder `behind.append(group)`.",
        ],
        why="`values = []`, `for group in wer:` met `values.append(wer[group])`, en `best = min(values)`. Dan `behind = []` en `for group in wer:` met `if wer[group] > best + limit:` en `behind.append(group)`. Dat geeft `['black men', 'black women']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fria():
    mo.md(r"""
    ### Oefening 4. Wie moet een FRIA doen?
    """)
    return


@app.cell(hide_code=True)
def choice_fria_cell():
    choice_fria = L.choice(
        "fria",
        question="Welke van deze gebruiksverantwoordelijken moet volgens de AI Act de gevolgen voor de grondrechten beoordelen vóór het eerste gebruik van het eigen AI-systeem?",
        options={
            "shop": "Een winkel die met AI de binnenkomende post sorteert",
            "right": "Een bank die aanvragers van een lening scoort met AI",
            "grid": "Een netbeheerder die met AI het stroomnet stabiel houdt",
        },
        correct="right",
        why={
            "shop": "Post sorteren is geen gebruik met een hoog risico. De AI Act noemt overheidsorganisaties, private partijen die openbare diensten verlenen, en gebruiksverantwoordelijken van kredietscores en van prijzen voor levens- en ziektekostenverzekeringen. Zie *In Nederland en de EU* bij *Het idee*.",
            "right": "De kredietwaardigheid van mensen scoren is een gebruik met een hoog risico waarvoor de AI Act van elke gebruiksverantwoordelijke, ook een private bank, een beoordeling van de gevolgen voor de grondrechten vraagt vóór het eerste gebruik. Voor deze systemen geldt dat vanaf 2 december 2027.",
            "grid": "Kritieke infrastructuur telt als hoog risico, maar de AI Act laat die hier buiten: daar wordt geen beoordeling van de gevolgen voor de grondrechten gevraagd. Zie *In Nederland en de EU* bij *Het idee*.",
        },
    )
    choice_fria
    return (choice_fria,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl) was de vraag wat een getal over fairness kan laten zien. Hier is de vraag wie het uitrekent, wanneer, en wat er daarna gebeurt. De metingen van deze module helpen pas als iemand het moment, de data en de bevoegdheid heeft om ernaar te handelen: een fairness review zet die in de planning. Het IAMA en de FRIA zijn zulke reviews, uitgeschreven voor overheden en sommige bedrijven; ze vragen om een gesprek en een besluit, niet om afgevinkte vakjes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fria, choice_proactive, result_behind, result_missing):
    L.wrap_up(result_missing, choice_proactive, result_behind, choice_fria)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
