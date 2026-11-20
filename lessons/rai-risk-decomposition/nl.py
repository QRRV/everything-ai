# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-risk-decomposition. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Risk, Risk Management

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-risk-decomposition", "nl")
    C = L.checks   # een verzonnen AI-assistent op de website van een apotheek: de gevaren, per antwoord en per dag


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Robuust, veerkrachtig, beschikbaar, voorspelbaar](/course/open/rai-reliability?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_robust():
    quick_robust = L.choice(
        "quick-robust",
        question="Het cijfermodel van de course leest 890 van de 1000 testcijfers goed, en 452 als de cijfers drie pixels naar rechts schuiven. Over welk van de vier woorden gaat dat verschil?",
        options={
            "right": "Robuustheid: zijn niveau houden bij input die een beetje anders is",
            "available": "Beschikbaarheid: het deel van de requests dat niet met een fout mislukt",
            "predictable": "Voorspelbaarheid: elke keer hetzelfde antwoord geven als het hetzelfde cijfer ziet",
        },
        correct="right",
        why={
            "right": "Het zijn dezelfde cijfers, een beetje opzij, en het model zakt in niveau: dat is robuustheid, gemeten zoals Hendrycks en Dietterich dat testen.",
            "available": "Het model gaf bij elk cijfer een antwoord; het zat er alleen vaker naast. Beschikbaarheid telt antwoorden die niet aankomen. Zie [Robuust, veerkrachtig, beschikbaar, voorspelbaar](/course/open/rai-reliability?lang=nl).",
            "predictable": "Het model geeft bij hetzelfde beeld elke keer hetzelfde antwoord; de opgeschoven cijfers zijn andere beelden. Zie [Robuust, veerkrachtig, beschikbaar, voorspelbaar](/course/open/rai-reliability?lang=nl).",
        },
    )
    quick_robust
    return (quick_robust,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="Een dobbelspel: je wint €6 als je een zes gooit, en anders niets. Wat is de verwachtingswaarde van wat je wint?",
        options={
            "half": "€3: precies halverwege tussen niets en €6",
            "right": "€1: €6 maal 1/6, plus €0 maal 5/6",
            "prize": "€6: wat je wint als je een zes gooit",
        },
        correct="right",
        why={
            "right": "Elke waarde maal de kans erop, opgeteld: 6 × 1/6 + 0 × 5/6 = 1. Gemiddeld wint een worp €1.",
            "half": "Dat is het gewone gemiddelde van €0 en €6, alsof ze even waarschijnlijk zijn; niets komt vijf keer zo vaak voor. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "prize": "Dat is het beste geval, en dat komt één worp op de zes voor. De verwachtingswaarde weegt elke waarde met de kans erop. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een apotheek zet een AI-assistent op de website die vragen over medicijnen beantwoordt. Er kunnen drie dingen misgaan: de assistent kan onbeleefd antwoorden, een verkeerde dosis geven, en missen dat iemand een noodsituatie beschrijft. Onbeleefde antwoorden komen het vaakst voor; een gemiste noodsituatie doet de meeste schade. Waar moet het team als eerste aan werken?

    Hoe vaak iets misgaat, beantwoordt dat niet, en hoe erg het is evenmin. Deze les combineert de twee tot risico, telt risico op over alles wat mis kan gaan, en splitst het dan in vier factoren, die elk een plek zijn om het risico te verlagen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een verzonnen chatbot beantwoordt 1.000.000 vragen per dag. 1 vraag op de 100 vraagt om iets gevaarlijks, en de chatbot weigert 99,9% daarvan. Hoeveel gevaarlijke antwoorden geeft hij per dag?",
        options={
            "one": "Ongeveer 1 per dag",
            "right": "Ongeveer 10 per dag",
            "thousand": "Ongeveer 1000 per dag",
        },
        reveal="**Ongeveer 10 per dag.** 1 op de 100 van 1.000.000 vragen is 10.000 gevaarlijke vragen, en 0,1% daarvan is 10: 3650 per jaar. Een kans die voor één vraag piepklein lijkt, wordt een vast aantal als er veel vragen komen: dat is blootstelling, die *Blootstelling en kwetsbaarheid* in *Het idee* in de formule zet.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Gevaar en risico

    In het hoofdstuk over veiligheidstechniek definieert het leerboek *Introduction to AI Safety, Ethics, and Society* (2024) van Dan Hendrycks het gevaar: een bron van mogelijke schade. Een brandbare vloeistof is er een, en glasscherven ook. Een **gevaar** is nog geen risico: een haai is een gevaar, maar vormt geen risico zolang niemand het water in gaat. Een dreiging is een gevaar met kwade opzet erachter, zoals een hacker.

    ### De klassieke risicoformule

    Risico valt uiteen in kans en ernst, gaat het leerboek verder: de kans dat er een ongeluk gebeurt, en hoeveel schade het aanricht als het gebeurt. Als formule: risico(gevaar) = P(gevaar) × ernst(gevaar), volgens het boek de klassieke manier om risico op te schrijven. Een vulkaan die waarschijnlijk uitbarst, vormt weinig risico als er niemand in de buurt woont, want de schade is klein; een vulkaan die waarschijnlijk blijft slapen, vormt ook weinig risico, zelfs met veel mensen eromheen, want de kans is klein. Het AI Risk Management Framework van NIST uit 2023 definieert risico langs dezelfde lijn, als combinatie van beide, zonder formule, in een formulering die het ontleende aan de norm ISO 31000:2018 en aanpaste: een samengestelde maat van de kans dat een gebeurtenis plaatsvindt en de omvang van de gevolgen ervan. NIST voegt eraan toe dat gevolgen positief, negatief of allebei kunnen zijn; deze les telt, net als het leerboek, alleen schade.

    Om het risico van een AI-systeem te beoordelen, zegt het leerboek, moet je twee dingen weten: de kans dat het iets onbedoelds doet, en hoe erg de gevolgen dan zijn. Dat een mogelijk catastrofaal gevaar is gedemonstreerd, betekent volgens het boek niet per se dat het risico hoog is: dat een model in een test iets schadelijks kan doen, zegt nog niet hoe waarschijnlijk het dat in gebruik doet. Het leerboek vraagt om allebei: wat een systeem kan, en wat het geneigd is te doen.

    ### De gevaren optellen

    Het totale risico van een systeem is de som van de risico's van al zijn gevaren: risico = Σ P(gevaar) × ernst(gevaar), waarbij Σ voor elk gevaar één term optelt. Dat is een verwachtingswaarde, zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl): elke schade maal de kans erop, opgeteld. Risico is de verwachte schade. Het leerboek telt de risico's van onafhankelijke gevaren op: de som behandelt de gevaren als losse gebeurtenissen.

    Nul risico is niet het doel, want risico tot nul terugbrengen is volgens het boek waarschijnlijk onmogelijk. Een beter doel, stelt het leerboek voor, is catastrofale risico's verwaarloosbaar maken, bijvoorbeeld minder dan 0,01% kans op een existentiële catastrofe per eeuw.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Blootstelling en kwetsbaarheid

    Wie zich met rampen bezighoudt, splitst risico verder. Op 1 december 2016 zette een expertgroep die de Algemene Vergadering van de Verenigde Naties had ingesteld, de begrippen voor het verkleinen van rampenrisico's op een rij. Rampenrisico gaat over de doden, de gewonden en de schade die in een bepaalde periode kunnen vallen, probabilistisch bepaald als een functie van gevaar, blootstelling, kwetsbaarheid en capaciteit. Blootstelling gaat over de mensen, gebouwen en andere bezittingen in een gebied met gevaar; kwetsbaarheid over de omstandigheden die mensen, gemeenschappen, bezittingen of systemen vatbaarder maken voor wat een gevaar aanricht; capaciteit over de sterke punten en middelen om het risico te beheersen. Het verslag geeft geen formule: het spreekt alleen van een functie. Het noemt ook het restrisico: het rampenrisico dat overblijft, zelfs als er goede maatregelen zijn genomen.

    Het leerboek gebruikt een vergelijkbare splitsing, met eigen betekenissen en zonder capaciteit, en vermenigvuldigt de factoren: de **disaster risk equation**. Risico valt uiteen in blootstelling, kans, ernst en kwetsbaarheid. Bij een gladde vloer is blootstelling hoeveel mensen eroverheen lopen en hoelang hij nat blijft; kans, hoe waarschijnlijk het is dat iemand die erover loopt, valt; ernst, hoe hard de vloer is, marmer of tapijt; kwetsbaarheid, hoe makkelijk wie valt iets breekt, wat bijvoorbeeld van de leeftijd afhangt. Die blootstelling lijkt op die van de VN, maar is niet hetzelfde: niet waar mensen zijn, maar hoeveel ze aan een gevaar blootstaan. Kans en ernst samen vormen het gevaarsniveau van het gevaar zelf. Voor elke gevaarlijke gebeurtenis h is risico = P(h) × ernst(h) × blootstelling(h) × kwetsbaarheid(h), en het totaal telt de gebeurtenissen op.

    Het leerboek waarschuwt dat de formule niet wiskundig exact is: ze moet laten zien dat het risico stijgt als een van de factoren stijgt, en daalt als een van de factoren daalt. Eén factor op nul maakt het risico van dat gevaar nul, hoe groot de andere factoren ook zijn; de andere gevaren houden hun risico. Een gangbare uitbreiding deelt door het vermogen om ermee om te gaan: hoe goed mensen herstellen. Voor schade waarvan niemand kan herstellen, zoals een existentiële catastrofe, zou de formule een risico geven dat volgens het boek astronomisch groot of oneindig is.

    ### Drie plekken om AI-risico te verlagen

    Elke factor is een plek om risico te verlagen, en het leerboek koppelt er drie onderzoeksgebieden aan. Robuustheid, bijvoorbeeld tegen aanvallen, verlaagt de kwetsbaarheid van AI-systemen zelf: hoe makkelijk ze tot schade te brengen zijn; monitoring verlaagt de blootstelling, door gevaren op te merken en te vermijden; control verlaagt het gevaarsniveau zelf, door AI-systemen bestuurbaar en minder gevaarlijk te houden. Om AI-risico te verlagen moet je volgens het boek aan alle drie werken, en niet op één vertrouwen. Voor proxy gaming, waarbij een AI-systeem een onvolmaakte maatstaf voor de bedoeling uitbuit (zie [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl)), geeft het leerboek bij elk een voorbeeld: voor minder blootstelling monitoring die vreemd gedrag vroeg opmerkt; voor een lager gevaarsniveau een doel dat dichter bij de bedoeling ligt en minder macht voor het systeem; voor minder kwetsbaarheid betere proxy's, robuustere AI-systemen of minder afhankelijkheid ervan. In een paper uit 2021 gaven Hendrycks, Nicholas Carlini, John Schulman en Jacob Steinhardt die rol, het gevaar zelf verlagen, aan een ander gebied: alignment, die systemen uit zichzelf minder gevaarlijk maakt.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **gevaar** | een bron van mogelijke schade; het vormt pas een risico als iemand of iets eraan blootstaat |
        | dreiging | een gevaar met iemand erachter die schade wil aanrichten |
        | risico | de kans op een schadelijke gebeurtenis, gecombineerd met hoeveel schade die aanricht; in de klassieke risicoformule kans maal ernst |
        | ernst | hoeveel schade een gebeurtenis aanricht als die plaatsvindt; in deze les in verzonnen schadepunten |
        | **disaster risk equation** | risico als kans × ernst × blootstelling × kwetsbaarheid, opgeteld over de gevaarlijke gebeurtenissen: een vuistregel, geen exacte wiskunde |
        | blootstelling | hoeveel iemand aan een gevaar blootstaat: hoeveel mensen, hoe vaak, hoelang |
        | kwetsbaarheid | hoe vatbaar wie blootstaat is voor schade door het gevaar |
        | restrisico | het risico dat overblijft na de maatregelen ertegen |
        | Σ | de som: Σ over de gevaren telt voor elk gevaar één term op |
        | `C.HAZARDS`, `C.EVENTS` | de verzonnen assistent van de apotheek: de drie gevaren met een kans per antwoord en een ernst, en dezelfde drie voor een hele dag, met vier factoren |
        """),
        L.watch_out("Risico is niet de kans alleen en ook niet de schade alleen: een zeldzaam gevaar met grote schade kan meer risico dragen dan een vaak voorkomend gevaar met weinig schade. En een factor op nul haalt het risico van één gevaar weg, niet dat van het hele systeem."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zet de gevaren op een rij: wat kan er misgaan, en voor wie.
    2. Schat voor elk gevaar de kans en de ernst, met één eenheid van schade voor allemaal.
    3. Vermenigvuldig ze, en tel op over de gevaren: de verwachte schade.
    4. Weet je meer, deel het risico dan verder op: hoeveel blootstelling er is, en hoe kwetsbaar wie blootstaat is.
    5. Zoek bij de grootste risico's de factor die het makkelijkst omlaag kan.
    6. Behandel de getallen als schattingen: een factor is zelden precies nul, en wat na de maatregelen overblijft, is restrisico.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent het risico uit van één verzonnen gevaar, een schadelijk antwoord: eerst met de klassieke risicoformule, voor één antwoord, dan met de disaster risk equation, voor een dag, en dan voor een dag met de helft van de blootstelling.
    """)
    return


@app.cell
def example_harm():
    p, severity = 0.002, 3                                # verzonnen: een schadelijk antwoord, 3 schadepunten
    print(p * severity)                                   # de klassieke risicoformule: per antwoord
    exposure, vulnerability = 5000, 0.5                   # 5000 antwoorden per dag; halve schade komt aan
    print(p * severity * exposure * vulnerability)        # schadepunten per dag
    print(p * severity * (exposure / 2) * vulnerability)  # de helft van de blootstelling
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Voor één antwoord is het risico 0,006 schadepunten; voor een dag van 5000 antwoorden, waarbij de helft van de schade de vragensteller bereikt, is het 15. De helft van de blootstelling geeft 7,5, de helft van het risico, en de helft van elke andere factor ook. In oefening 1 tel je de risico's van meer gevaren op.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. total_risk: alle gevaren samen

        `C.HAZARDS` is een lijst van drie dicts, één voor elk gevaar van de assistent van de apotheek, elk met een `"name"`, een kans per antwoord `"p"` en een ernst in schadepunten `"severity"`. Schrijf `total_risk(hazards)`. Begin `total` bij 0; tel voor elke dict `h` in `hazards` de kans maal de ernst op bij `total`; geef `total` terug. Vervang de `...`. De check roept `total_risk(HAZARDS)` aan, en daarna andere lijsten van gevaren, ook een lege.
        """),
        L.task("Het risico van alle gevaren samen, per antwoord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total_risk", lines=5)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total_risk", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total_risk",
        [
            "De klassieke risicoformule voor elk gevaar, en dan de som over de gevaren, zoals bij een verwachtingswaarde.",
            "Een `for`-loop over `hazards` geeft telkens één dict, en `h[\"p\"]` is de kans ervan. Tel het risico van elk gevaar op bij `total`, en geef `total` na de loop terug.",
            "`        total = total + ___`",
        ],
        why="`total = 0`, `for h in hazards:` met daarin `total = total + h[\"p\"] * h[\"severity\"]`, en dan `return total`. Voor HAZARDS: 0,65 schadepunten per antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_after_total():
    mo.md(r"""
    Voor één antwoord brengen de drie gevaren 0,65 schadepunten: 0,05 van onbeleefde antwoorden, 0,1 van verkeerde doses en 0,5 van gemiste noodsituaties. Het zeldzaamste gevaar, één gemiste noodsituatie op de 10.000 antwoorden, draagt het grootste deel van het risico, want het richt 5000 keer zoveel schade aan als een onbeleefd antwoord, dat 500 keer zo vaak voorkomt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_disaster():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. disaster_risk: vier factoren

        `C.EVENTS` bevat dezelfde drie gevaren voor een hele dag, nu met vier factoren: `"p"`, de kans per vraag van de soort waar het om gaat; `"severity"`; `"exposure"`, hoeveel van zulke vragen er per dag komen; en `"vulnerability"`, het deel van de schade dat de vragenstellers bereikt. Ze zijn allemaal verzonnen. Van de 2000 vragen per dag gaan er 500 over een dosis en beschrijven er 200 een noodsituatie; een onbeleefd antwoord kan bij elke vraag komen, dus de blootstelling is alle 2000, en het bereikt iedereen die het krijgt, kwetsbaarheid 1,0. De kansen zijn die van `C.HAZARDS`, per vraag van de soort: 10 gemiste noodsituaties op 10.000 vragen over een noodsituatie zijn 1 op de 10.000 antwoorden, omdat 1 vraag op de 10 een noodsituatie beschrijft. Een verkeerde dosis heeft een kwetsbaarheid van 0,25, omdat drie op de vier de bijsluiter in de doos nalezen; een gemiste noodsituatie 0,5, omdat ongeveer de helft toch zelf het alarmnummer zou bellen. Schrijf `disaster_risk(events)`: maak een lege dict `risks`; zet voor elke dict `e` in `events` het product van de vier factoren in `risks` onder `e["name"]`; geef `risks` terug. Vervang de `...`. De check roept `disaster_risk(EVENTS)` aan, en daarna andere lijsten.
        """),
        L.task("De schadepunten per dag van elke gevaarlijke gebeurtenis, in een dict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_disaster():
    box_disaster = L.code_box("disaster_risk", lines=5)
    box_disaster
    return (box_disaster,)


@app.cell(hide_code=True)
def check_disaster(box_disaster):
    result_disaster = L.check("disaster_risk", box_disaster.value)
    result_disaster
    return (result_disaster,)


@app.cell(hide_code=True)
def hints_disaster():
    L.hints(
        "disaster_risk",
        [
            "Dezelfde loop als in oefening 1, maar elke gebeurtenis krijgt een eigen plek in een dict in plaats van een plek in één som.",
            "Je zet een waarde in een dict met `risks[key] = value`; de key is hier de naam van de gebeurtenis. De vier factoren staan onder de keys `\"p\"`, `\"severity\"`, `\"exposure\"` en `\"vulnerability\"`.",
            "`        risks[e[\"name\"]] = ___`",
        ],
        why="`risks = {}`, `for e in events:` met daarin `risks[e[\"name\"]] = e[\"p\"] * e[\"severity\"] * e[\"exposure\"] * e[\"vulnerability\"]`, en dan `return risks`. Voor EVENTS: 100 schadepunten per dag van onbeleefde antwoorden, 50 van verkeerde doses en 500 van gemiste noodsituaties.",
    )
    return


@app.cell(hide_code=True)
def text_after_disaster():
    mo.md(r"""
    Een dag brengt 650 schadepunten: 100 van onbeleefde antwoorden, 50 van verkeerde doses en 500 van gemiste noodsituaties. Zonder de kwetsbaarheid was het 1300: de 0,65 per antwoord uit oefening 1 maal 2000 antwoorden. Elke factor van de gemiste noodsituatie halveren scheelt 250 schadepunten, meer dan alle onbeleefde antwoorden samen. Het team kan dus de factor kiezen die het makkelijkst te veranderen is: het alarmnummer boven het chatvenster zetten, zodat minder mensen er een noodsituatie in typen, wat de blootstelling verlaagt; of onder elk antwoord een regel zetten over wat je in een noodsituatie doet, wat de kwetsbaarheid verlaagt.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: meer gemiste noodsituaties

    De schuif stelt in hoeveel van 10.000 vragen die een noodsituatie beschrijven de assistent mist, van 1 tot 20; `C.EVENTS` heeft er 10. Het plaatje toont de schadepunten per dag van elke gevaarlijke gebeurtenis, en het totaal.
    """)
    return


@app.cell(hide_code=True)
def demo_missed():
    missed = L.explore("missed", "gemist per 10.000 vragen over een noodsituatie", lambda _v: C.demo_frame(_v, L.lang),
                       start=1, values=C.MISSED, role="",
                       task="Zoek het kleinste getal op de schuif waarbij de gemiste noodsituatie meer dan de helft van de schadepunten per dag draagt.")
    missed
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 4 draagt de gemiste noodsituatie meer dan de helft: 200 van de 350 schadepunten per dag. Bij 1 draagt de gemiste noodsituatie 50 van de 200. Elke extra misser op de 10.000 voegt 50 schadepunten per dag toe, terwijl de andere twee samen op 150 blijven. Een zeldzame gebeurtenis met grote schade kan door een kleine verandering in de kans erop van een klein deel van het risico naar het grootste deel gaan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_demonstration():
    mo.md(r"""
    ### Oefening 3. Eén demonstratie

    Een team schrijft 500 prompts om een model voor klantenservice te misleiden. Eén ervan krijgt het model zover dat het een klant een terugbetaling van €10.000 belooft.
    """)
    return


@app.cell(hide_code=True)
def choice_demonstration_cell():
    choice_demonstration = L.choice(
        "demonstration",
        question="Wat zegt deze demonstratie over het risico van het model in gebruik?",
        options={
            "high": "Het risico is hoog: wat een model één keer in een test deed, kan het in gebruik weer doen",
            "right": "Op zich weinig: het toont het gevaar, niet hoe vaak het in gebruik gebeurt",
            "euro": "€20 per klant: de 0,2% van de test maal de terugbetaling van €10.000",
        },
        correct="right",
        why={
            "right": "De test laat zien dat het gevaar bestaat, en wat het kost als het gebeurt. Hoe vaak het gebeurt bij echte klanten, die geen prompts schrijven om het model te misleiden, is nog onbekend, en de klassieke risicoformule heeft allebei nodig.",
            "high": "Dat een mogelijk catastrofaal gevaar is gedemonstreerd, betekent volgens het leerboek niet per se dat het risico hoog is, en dat geldt hier ook: de kans in gebruik is nog onbekend. Zie *De klassieke risicoformule* in *Het idee*.",
            "euro": "0,2% is het deel van de misleidende prompts dat werkte, niet de kans bij echte klanten; die kans is nog onbekend. Zie *De klassieke risicoformule* in *Het idee*.",
        },
    )
    choice_demonstration
    return (choice_demonstration,)


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Oefening 4. Eén factor op nul

    De assistent van een bank heeft drie gevaren: verkeerd advies over leningen, een onbeleefd antwoord en een gelekt rekeningnummer. De bank laat hem geen vragen over leningen meer beantwoorden, dus de blootstelling van het eerste gevaar wordt 0.
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "zero",
        question="Wat gebeurt er volgens de disaster risk equation met het totale risico van de assistent?",
        options={
            "zero": "Het wordt 0, want één factor van het product is nu 0",
            "right": "Het daalt met het risico van dat gevaar; de andere twee tellen nog mee",
            "same": "Het blijft gelijk, want de kans op verkeerd leenadvies is niet veranderd",
        },
        correct="right",
        why={
            "right": "Eén factor op nul maakt het risico van dat gevaar nul, hoe groot de andere factoren ook zijn. Het totaal telt de gevaren op, en de andere twee houden hun risico.",
            "zero": "Het product geldt per gevaar; het totaal is een som over de gevaren, en het onbeleefde antwoord en het gelekte rekeningnummer tellen nog mee. Zie *Blootstelling en kwetsbaarheid* in *Het idee*.",
            "same": "De kans per vraag veranderde niet, maar er komt geen vraag over leningen meer binnen: met blootstelling 0 is het product voor dat gevaar 0. Zie *Blootstelling en kwetsbaarheid* in *Het idee*.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Robuust, veerkrachtig, beschikbaar, voorspelbaar](/course/open/rai-reliability?lang=nl) mat hoe vaak een model faalt. Deze les woog falen naar de schade: risico is kans maal ernst, opgeteld over de gevaren, de verwachtingswaarde uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) met schade als waarde; de disaster risk equation voegt blootstelling en kwetsbaarheid toe. De volgende les, *Negens van betrouwbaarheid*, draait de kans op falen om tot betrouwbaarheid, en telt die in negens.

    **Een voorbeeld uit het dagelijks leven.** In het donker naar huis fietsen: licht verlaagt de kans op een ongeluk, langzamer rijden de ernst, minder vaak fietsen de blootstelling, en een helm de kwetsbaarheid van je hoofd.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_demonstration, choice_zero, result_disaster, result_total):
    L.wrap_up(result_total, result_disaster, choice_demonstration, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
