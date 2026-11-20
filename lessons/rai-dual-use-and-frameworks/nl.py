# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-dual-use-and-frameworks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Code of Practice

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dual-use-and-frameworks", "nl")
    C = L.checks   # de kaders, de datums van de AI Act en de tellingen van Google, zoals de bronnen ze geven


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Red teaming](/course/open/rai-red-teaming?lang=nl) en één over [De OWASP-lijsten voor LLM-apps](/course/open/rai-owasp-lists?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_disclosure():
    quick_disclosure = L.choice(
        "quick-disclosure",
        question="Je merkt dat de hulpassistent van een andere fietsenwinkel de personeelscode prijsgeeft. Wie moet dat als eerste horen?",
        options={
            "right": "Die winkel, vertrouwelijk",
            "shops": "De andere fietsenwinkels, zodat ze die van hen kunnen nakijken",
            "online": "Iedereen online, zodat de winkel het snel oplost",
        },
        correct="right",
        why={
            "right": "Dat is responsible disclosure, wat CERT en het Nederlandse NCSC coordinated vulnerability disclosure noemen: eerst de maker, en het publiek pas als er een oplossing is. Deze les stelt dezelfde vraag over een hele lijst aanvallen die werken.",
            "shops": "De andere winkels komen later: eerst hoort de winkel met de lekkende assistent het, vertrouwelijk, en krijgt die tijd om het te herstellen. Zie *Melden wat je vond* in [Red teaming](/course/open/rai-red-teaming?lang=nl).",
            "online": "Online kan iedereen het bij die winkel gebruiken voordat er een oplossing is. Responsible disclosure gaat andersom: eerst de maker, en het publiek pas als er een oplossing is. Zie *Melden wat je vond* in [Red teaming](/course/open/rai-red-teaming?lang=nl).",
        },
    )
    quick_disclosure
    return (quick_disclosure,)


@app.cell(hide_code=True)
def quick_check_owasp():
    quick_owasp = L.choice(
        "quick-owasp",
        question="Waarvoor is de OWASP Top 10 for LLM Applications bedoeld?",
        options={
            "test": "Het is een test die een app doorstaat voordat hij online gaat",
            "right": "Hij helpt kiezen naar welke risico's van een app je eerst kijkt",
            "law": "Hij bevat de regels die de EU stelt aan elke app met een LLM erin",
        },
        correct="right",
        why={
            "right": "OWASP noemt zijn Top 10 voor het web een standaarddocument om mensen bewust te maken: een lijst is het begin van een beoordeling, niet het eind, en niemand hoeft hem te volgen. Deze les komt meer van zulke documenten tegen, en één dat wet is.",
            "test": "Geen app slaagt of zakt voor een Top 10: de lijst helpt kiezen waar je eerst kijkt, en een beoordeling gaat daarna verder. Zie *Wat een Top 10 van OWASP is* in [De OWASP-lijsten voor LLM-apps](/course/open/rai-owasp-lists?lang=nl).",
            "law": "OWASP is geen overheid, en zijn lijsten verplichten niemand. De EU heeft wel een wet over AI, de AI Act, waar deze les op uitkomt. Zie *Wat een Top 10 van OWASP is* in [De OWASP-lijsten voor LLM-apps](/course/open/rai-owasp-lists?lang=nl).",
        },
    )
    quick_owasp
    return (quick_owasp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les schreef Qwen3 30 testgevallen, en 23 daarvan kregen de personeelscode uit de hulpassistent van Spoke & Chain. Voor de winkel is die lijst een geschenk: hij laat zien wat er hersteld moet worden. Voor iemand die de code wil van een winkel met dezelfde assistent, is diezelfde lijst de aanval. Red teaming, de OWASP-lijsten, een model dat code schrijft: wat een verdediger helpt, kan een aanvaller helpen. Daar gaat deze les over, en over de regels die mensen daartegen hebben opgeschreven.

    Twee vragen keren in deze les steeds terug. Wat doen aanvallers echt met AI, volgens de makers die het zien gebeuren? En welke documenten zeggen wat de maker van een AI-systeem daartegen moet doen, en welke daarvan zijn wet?

    **Een voorbeeld uit het dagelijks leven.** Een ladder helpt een schilder bij een raam te komen, en een inbreker ook. Niemand verbiedt ladders. Het gaat erom wie er een gebruikt, waarvoor, en wat het huis doet om veilig te blijven.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In november 2025 meldde Anthropic een campagne waarin aanvallers zijn tool Claude Code gebruikten tegen zo'n dertig doelwitten. Welk deel van het werk deed de AI, volgens Anthropic?",
        options={
            "small": "Ongeveer een tiende",
            "middle": "Ongeveer de helft",
            "right": "80 tot 90%",
        },
        reveal="**80 tot 90%.** Volgens Anthropic deed de AI 80 tot 90% van de campagne, en hoefden mensen maar af en toe in te grijpen, op misschien 4 tot 6 beslissende momenten per campagne. Anthropic schrijft ook dat de aanvallers maar in een klein aantal gevallen slaagden, en dat de AI soms inloggegevens verzon: een groot deel van het werk, geen zeker succes. *Wat aanvallers met AI doen* in *Het idee* zet dit rapport naast dat van Google.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat dual use betekent

    **Dual use** betekent dat hetzelfde ding een goed en een schadelijk doel dient. Ook het handelsrecht gebruikt het woord. De Europese Commissie omschrijft producten voor tweeërlei gebruik als goederen, software en technologie die zowel voor civiele als voor militaire toepassingen te gebruiken zijn. De export ervan valt onder Verordening (EU) 2021/821, die in elk EU-land rechtstreeks geldt.

    Voor AI omschreef Executive Order 14110, een decreet van de Amerikaanse president uit oktober 2023, een *dual-use foundation model* als een model dat op brede data getraind is, meestal met self-supervision leert, minstens tientallen miljarden parameters heeft, in veel verschillende situaties bruikbaar is, en heel goed is in taken die een ernstig risico vormen voor de veiligheid, de economische veiligheid of de volksgezondheid van het land, of zo te veranderen is dat het daar heel goed in wordt. Eén van de voorbeelden van zo'n taak: krachtige cyberaanvallen mogelijk maken door automatisch zwakke plekken te vinden en te misbruiken bij veel mogelijke doelwitten. En een beveiliging veranderde daar niets aan: een model valt ook onder de omschrijving als het met technische beveiligingen bij gebruikers komt die proberen te voorkomen dat ze de gevaarlijke mogelijkheden benutten. Het decreet werd in januari 2025 ingetrokken. Qwen3-0.6B, met 0,6 miljard parameters, blijft ver onder tientallen miljarden.

    Het Amerikaanse AI Safety Institute, een overheidsinstelling die geavanceerde AI-modellen test (een latere les gaat in op zulke instituten), schreef vrijwillige richtlijnen over het misbruik van zulke modellen, uitgegeven door NIST, het Amerikaanse instituut voor normen; de versie van januari 2025 is een concept. Het concept noemt misbruikrisico het risico dat iemand een AI-model met opzet misbruikt om schade aan te richten. En in de beveiliging, schrijft het, is de grens moeilijk te trekken: door het samenspel van aanval en verdediging zijn de meeste mogelijkheden in zekere zin dual use, en misbruik betrouwbaar onderscheiden van legitiem gebruik kan lastig zijn. De testgevallen van de vorige les zijn een voorbeeld: dezelfde berichten vertellen de winkel wat er hersteld moet worden, en een aanvaller wat werkt.

    ### Wat aanvallers met AI doen

    De makers van modellen zien een deel ervan gebeuren, in hun eigen producten, en sommige melden het. In januari 2025 schreef de Google Threat Intelligence Group hoe aanvallers Gemini gebruikten, het model van Google: AI kan voor aanvallers een nuttig hulpmiddel zijn, maar is nog niet de doorbraak die er soms van gemaakt wordt. Ze gebruikten het voor gewone taken, zoals problemen oplossen, onderzoek doen en teksten maken, en Google zag geen aanwijzingen dat ze er nieuwe mogelijkheden mee ontwikkelden. In plaats van alles om te gooien, laat generatieve AI aanvallers sneller en op grotere schaal werken. Jailbreaks waren eenvoudig, en mislukten: de aanvallers probeerden met openbaar beschikbare jailbreak-prompts de beveiliging van Gemini te omzeilen, zonder succes. Google vond APT-groepen, kort voor *advanced persistent threat*, uit meer dan 20 landen; het telde meer dan 20 groepen gesteund door China, meer dan 10 gesteund door Iran, negen gesteund door Noord-Korea en drie gesteund door Rusland, en volgens Google gebruikten de Iraanse groepen Gemini het meest.

    Anthropic schreef in augustus 2025 dat AI die zelf taken uitvoert, tot wapen is gemaakt, en dat AI de drempel voor geavanceerde cybercriminaliteit heeft verlaagd. In één geval van afpersing met gestolen data, waarbij aanvallers data stelen en geld eisen om ze niet te publiceren, richtte de dader zich op minstens 17 verschillende organisaties, met losgeld dat soms boven de $500.000 lag. In november 2025 beschreef Anthropic de campagne uit de gok, die half september 2025 ontdekt werd. De dader was volgens Anthropic met grote zekerheid een groep die door de Chinese staat gesteund werd, en manipuleerde Claude Code zo dat het probeerde binnen te dringen bij zo'n dertig doelwitten over de hele wereld; in een klein aantal gevallen lukte dat. Anthropic denkt dat dit het eerste gedocumenteerde geval is van een grote cyberaanval die zonder wezenlijk ingrijpen van mensen werd uitgevoerd. De AI maakte ook fouten: soms verzon hij inloggegevens, of beweerde hij geheime informatie buitgemaakt te hebben die in werkelijkheid openbaar was. En Anthropic stelde zijn eigen verhaal over de snelheid achteraf bij, tot duizenden verzoeken, vaak meerdere per seconde.

    Elk van deze rapporten komt van een bedrijf, over zijn eigen model. Google keek naar Gemini, met gegevens tot januari 2025; Anthropic keek naar Claude, en schrijft dat het alleen zicht heeft op het gebruik van Claude. Geen ervan is een controle door een buitenstaander.

    ### De regels

    Verschillende documenten zeggen wat de maker van een AI-systeem moet doen om het veilig te houden. In juni 2023 stelde Google het *Secure AI Framework* (SAIF) voor, een denkkader voor veilige AI-systemen, met zes kernonderdelen, van een stevige basis voor beveiliging uitbreiden naar AI tot de risico's van een AI-systeem bekijken in de bedrijfsprocessen eromheen; een ervan vraagt om geregelde oefeningen van een red team. De site deelt de kaart van SAIF in vier onderdelen in: data, infrastructuur, model en applicatie, en voegt eraan toe dat het niet laat zien hoe Google het zelf technisch inricht. SAIF 2.0, van oktober 2025, voegt agents toe, AI-systemen die zelf taken uitvoeren: een agent moet duidelijke menselijke beheerders hebben, zijn bevoegdheden moeten zorgvuldig beperkt zijn, en wat hij doet en plant moet te volgen zijn.

    In november 2023 publiceerden het Britse National Cyber Security Centre (NCSC), het Amerikaanse Cybersecurity and Infrastructure Security Agency (CISA) en hun partners de *Guidelines for secure AI system development*. Het document beveelt richtlijnen aan voor makers van elk systeem dat AI gebruikt, in vier gebieden van het leven van een AI-systeem: *secure design*, *secure development*, *secure deployment* en *secure operation and maintenance*. Volgens het persbericht van het NCSC steunen instanties uit 18 landen, waaronder de VS, de richtlijnen. Er staat geen Nederlandse instantie op de lijst.

    In januari 2025 publiceerde de Britse overheid een vrijwillige praktijkcode, de *Code of Practice for the Cyber Security of AI*, met 13 principes, om bij ETSI, een van de drie erkende Europese normalisatie-instellingen, een norm te helpen maken. Die norm is ETSI EN 304 223, aangenomen op 8 december 2025, die basiseisen voor de beveiliging van AI-modellen en -systemen vastlegt. Volgens ETSI bevat hij 13 principes en eisen in vijf fasen: *secure design*, *secure development*, *secure deployment*, *secure maintenance* en *secure end of life*. Principe 1 is mensen bewust maken van de dreigingen en risico's van AI, principe 13 data en modellen op de juiste manier afvoeren. Over testen: ontwikkelaars moeten ervoor zorgen dat elk model, elke applicatie en elk systeem dat bij beheerders of eindgebruikers komt, getest is als deel van een beoordeling van de beveiliging. Een norm is geen wet: de bronnen die deze les las, zeggen niet dat een wet hem verplicht stelt.

    Eén van deze documenten is wet. De AI Act is de wet van de EU over AI, en zijn artikel 15 gaat over AI-systemen met een hoog risico; voor de datums waarop het geldt, verdeelt de wet die over de systemen van zijn bijlage III en die van zijn bijlage I. Zulke systemen moeten zo ontworpen en ontwikkeld worden dat ze een passende accuracy, robuustheid en cyberbeveiliging halen, en dat gedurende hun hele levenscyclus volhouden. En ze moeten bestand zijn tegen pogingen van onbevoegden om hun gebruik, uitvoer of prestaties te veranderen door zwakke plekken in het systeem te misbruiken. Het artikel noemt data poisoning, model poisoning, adversarial examples en aanvallen op vertrouwelijkheid. De wet geldt vanaf 2 augustus 2026, maar na een wijziging, de Digital Omnibus over AI, gelden de regels voor systemen met een hoog risico, met artikel 15, vanaf 2 december 2027 voor de systemen van bijlage III, en vanaf 2 augustus 2028 voor die van bijlage I.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.documents_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **dual use** | dat hetzelfde ding een goed en een schadelijk doel dient, zoals een lijst aanvallen die werkt |
        | APT-groep | *advanced persistent threat*: in de tellingen van Google een groep aanvallers die door een staat gesteund wordt |
        | AI-systeem met een hoog risico | een AI-systeem waarvoor de regels van de AI Act voor een hoog risico gelden, zoals artikel 15 |
        | `C.NCSC_AREAS` | de vier gebieden van de richtlijnen van het NCSC en CISA, 2023 |
        | `C.ETSI_PHASES` | de vijf fasen van ETSI EN 304 223, 2025 |
        | `C.GROUPS`, `C.NUMBERS` | de tellingen van Google per land, en de tellingen in woorden met hun getal |
        """),
        L.watch_out("De datums van artikel 15 zijn die van de AI Act zoals de Digital Omnibus hem wijzigde, in de tekst van 27 juli 2026. Regels veranderen: kijk naar de tekst die geldt op de dag dat je hem nodig hebt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag voordat je een bevinding, een tool of een model deelt, wie het nog meer helpt: wat jou helpt herstellen, helpt een ander aanvallen.
    2. Lees een rapport over aanvallers naar wie het schreef, over welk model, en wanneer.
    3. Gebruik een kader als checklist voor het hele leven van een systeem, van ontwerp tot de dag dat het stopt.
    4. Zoek uit welke regels voor jouw systeem wet zijn, en vanaf welke dag; de rest is advies, hoe nuttig ook.
    5. Test voordat je iets uitbrengt, zoals ETSI EN 304 223 vraagt, en blijf daarna testen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel vergelijkt de richtlijnen van 2023 met de norm van 2025. `C.NCSC_AREAS` bevat de vier gebieden van de richtlijnen en `C.ETSI_PHASES` de vijf fasen van de norm, elk als een lijst teksten. `area in etsi` vraagt of de lijst `etsi` de tekst `area` bevat.
    """)
    return


@app.cell
def example_same():
    ncsc = C.NCSC_AREAS    # de vier gebieden van de richtlijnen van het NCSC en CISA, 2023
    etsi = C.ETSI_PHASES   # de vijf fasen van ETSI EN 304 223, 2025
    print(len(ncsc), "areas,", len(etsi), "phases")

    same = []
    for area in ncsc:
        if area in etsi:
            same.append(area)
    print("the same name in both:", same)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie namen zijn in beide hetzelfde. De norm heeft vijf fasen, dus twee fasen hebben geen gebied met dezelfde naam. Laten de richtlijnen die weg, of noemen ze hetzelfde onderwerp met andere woorden? Hele namen vergelijken is niet genoeg om dat te zien: oefening 1 kijkt in de namen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_uncovered():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. uncovered: wat de norm toevoegt

        Schrijf `uncovered(phases, areas)`: de fasen uit de lijst `phases` die geen gebied uit de lijst `areas` dekt, in de volgorde van `phases`. Een gebied dekt een fase als de naam van de fase zonder `secure ` deel is van de naam van het gebied: `maintenance` is deel van `secure operation and maintenance`. Elke naam begint met `secure `, 7 tekens, dus `rest = phase[7:]` is de rest van de naam. Tussen twee teksten vraagt `in` of de eerste deel is van de tweede: `"cat" in "a cat sat"` is `True`. `covered` zegt of een gebied `rest` bevat, en de fasen die niet gedekt zijn, gaan in de lijst `found`. Vervang de `...`. De check roept `uncovered(ETSI_PHASES, NCSC_AREAS)` aan, en daarna een paar eigen lijsten.
        """),
        L.task("Bekijk per fase alle gebieden, en neem de fase op als geen enkel gebied rest bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_uncovered():
    box_uncovered = L.code_box("uncovered", lines=12)
    box_uncovered
    return (box_uncovered,)


@app.cell(hide_code=True)
def check_uncovered(box_uncovered):
    result_uncovered = L.check("uncovered", box_uncovered.value)
    result_uncovered
    return (result_uncovered,)


@app.cell(hide_code=True)
def hints_uncovered():
    L.hints(
        "uncovered",
        [
            "Per fase: begin met `covered = False`, bekijk elk gebied, en beslis pas na het laatste gebied of de fase in `found` komt.",
            "Binnen de loop over de gebieden zet één `if` `covered = True`; na die loop voegt één `if` op `covered` de fase toe.",
            "`            if rest ___ area:`",
        ],
        why="`covered = False`, `for area in areas:` met `if rest in area:` en daaronder `covered = True`, dan `if not covered:` met `found.append(phase)`. `uncovered(ETSI_PHASES, NCSC_AREAS)` geeft `['secure end of life']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_uncovered():
    mo.md(r"""
    De richtlijnen van 2023 dekken ontwerp, ontwikkeling, uitrol en onderhoud; de norm van 2025 voegt het einde toe: wat er met de data en het model gebeurt als een systeem stopt. Een model dat op een oude schijf het gebouw verlaat, kan nog steeds prijsgeven wat het geleerd heeft. Twee documenten vergelijken op de woorden in hun namen gaat snel, maar het is iets anders dan ze lezen, en een fase met een andere naam kan onder een ander kopje toch gedekt zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_article():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. article_15: wanneer de wet geldt

        Schrijf `article_15(kind, day)`: of artikel 15 van de AI Act op de dag `day` geldt voor een systeem van de soort `kind`, als `True` of `False`. `kind` is `"Annex III"` of `"Annex I"` voor een AI-systeem met een hoog risico uit die bijlage, en elke andere tekst voor een systeem zonder hoog risico. `day` is een datum geschreven als jaar-maand-dag, zoals `"2027-12-02"`. Twee teksten worden teken voor teken vergeleken, dus `"2027-12-02" < "2028-01-15"` is `True`: zo geschreven datums vergelijk je zoals de dagen zelf. Een regel geldt vanaf zijn eerste dag, en die dag telt mee. Vervang de `...`. De check roept `article_15("Annex III", "2027-12-02")` aan, en daarna meer soorten en dagen.
        """),
        L.task("Geef elke bijlage een eigen eerste dag, en elke andere soort False."),
    ])
    return


@app.cell(hide_code=True)
def exercise_article():
    box_article = L.code_box("article-15", lines=7)
    box_article
    return (box_article,)


@app.cell(hide_code=True)
def check_article(box_article):
    result_article = L.check("article-15", box_article.value)
    result_article
    return (result_article,)


@app.cell(hide_code=True)
def hints_article():
    L.hints(
        "article-15",
        [
            "*De regels* in *Het idee* geeft twee dagen: een voor de systemen van bijlage III, een voor die van bijlage I. Een systeem zonder hoog risico heeft er geen.",
            "Eén `if` per bijlage, die elk een vergelijking van `day` met de eerste dag van die bijlage teruggeeft; geef daarna `False` terug.",
            "`        return day ___ \"2027-12-02\"`",
        ],
        why="`if kind == \"Annex III\":` met `return day >= \"2027-12-02\"`, `if kind == \"Annex I\":` met `return day >= \"2028-08-02\"`, dan `return False`. `article_15(\"Annex III\", \"2027-12-02\")` geeft `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_article():
    mo.md(r"""
    Op 2 augustus 2026 geldt de AI Act, maar artikel 15 nog voor geen enkel systeem: een datum in een kop is niet de datum van elke regel. En een hulpassistent die geen AI-systeem met een hoog risico is, valt nooit onder artikel 15, op welke dag ook. Voor zo'n systeem zijn SAIF, de richtlijnen en de norm advies: de moeite waard om te volgen, maar geen ervan is een wet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. at_least: een telling lezen

        `C.GROUPS` bevat de tellingen van Google van de door een staat gesteunde groepen die Gemini gebruikten, `{"China": "Over 20", "Iran": "Over 10", "North Korea": "Nine", "Russia": "Three"}`, en `C.NUMBERS` de tellingen in woorden, met hun getal, `{"Nine": 9, "Three": 3}`. Schrijf `at_least(groups, numbers)`: het kleinste aantal groepen dat deze tellingen toelaten, als een geheel getal. Meer dan 20 groepen is minstens 21. `count[:5]` zijn de eerste 5 tekens van de tekst `count`, en `int("20")` geeft het getal `20`. `for country, count in groups.items():` neemt elk land met zijn telling, en `total` is de som tot nu toe. Vervang de `...`. De check roept `at_least(GROUPS, NUMBERS)` aan, en daarna eigen tellingen.
        """),
        L.task("Tel per land het kleinste aantal groepen op bij total: Over n is n + 1, een woord is zijn getal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_groups():
    box_groups = L.code_box("at-least", lines=9)
    box_groups
    return (box_groups,)


@app.cell(hide_code=True)
def check_groups(box_groups):
    result_groups = L.check("at-least", box_groups.value)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "at-least",
        [
            "Twee soorten telling: de ene begint met `Over ` en een getal, de andere is een woord dat `numbers` kent. Elk heeft een eigen optelling nodig.",
            "Een `if` op `count[:5]`; tel daaronder bij `total` op wat de telling na `Over` minstens toelaat; tel onder `else:` het getal uit `numbers` op.",
            "`            total = total + ___ + 1`",
        ],
        why="`if count[:5] == \"Over \":` met `total = total + int(count[5:]) + 1`, en onder `else:` `total = total + numbers[count]`. `at_least(GROUPS, NUMBERS)` geeft `44`.",
    )
    return


@app.cell(hide_code=True)
def text_after_groups():
    mo.md(r"""
    Minstens 44 groepen, gesteund door vier staten, gebruikten Gemini, en het rapport van Google spreekt van groepen uit meer dan 20 landen. Dat zijn de groepen die één bedrijf zag in zijn eigen model; wat aanvallers met andere modellen doen, of met modellen die op hun eigen computer draaien, zit niet in deze telling.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dual():
    mo.md(r"""
    ### Oefening 4. Een lijst die werkt

    Spoke & Chain heeft zijn assistent hersteld. De winkel wil de 23 testgevallen die de code eruit kregen online zetten, zodat andere fietsenwinkels hun eigen assistent kunnen testen.
    """)
    return


@app.cell(hide_code=True)
def choice_dual_cell():
    choice_dual = L.choice(
        "dual-use",
        question="Waarom is die lijst dual use?",
        options={
            "right": "Hij helpt winkels testen, en aanvallers winkels aanvallen die niet testten",
            "harmless": "Dat is hij niet: er staat geen code in, alleen vragen",
            "law": "Omdat het Europese exportrecht elke test van een red team als product voor tweeërlei gebruik telt",
        },
        correct="right",
        why={
            "right": "De berichten die Spoke & Chain lieten zien wat er hersteld moest worden, werken bij elke assistent met dezelfde zwakke plek, ook voor iemand die de code wil. Volgens NIST zijn de meeste mogelijkheden in zekere zin dual use. Een zorgvuldige weg: vertel het eerst aan de makers van zulke assistenten, zoals bij elke zwakke plek, en zet de lijst online als ze het hebben kunnen herstellen.",
            "harmless": "De berichten zijn de aanval: 23 van de 30 kregen de code uit een assistent die opdracht had hem geheim te houden. Een lijst aanvallen die werken, werkt ook zonder code erin. Zie *Wat dual use betekent* in *Het idee*.",
            "law": "De Europese exportregels gaan over goederen, software en technologie voor civiel en militair gebruik, en niets wat deze les las, noemt tests van een red team. Dual use is hier de gewone betekenis: hetzelfde ding helpt beide kanten. Zie *Wat dual use betekent* in *Het idee*.",
        },
    )
    choice_dual
    return (choice_dual,)


@app.cell(hide_code=True)
def text_exercise_reports():
    mo.md(r"""
    ### Oefening 5. Twee rapporten

    Google schreef in januari 2025 dat AI voor aanvallers nog niet de doorbraak is die er soms van gemaakt wordt. Anthropic schreef in november 2025 dat AI 80 tot 90% van het werk van één campagne deed.
    """)
    return


@app.cell(hide_code=True)
def choice_reports_cell():
    choice_reports = L.choice(
        "reports",
        question="Spreken de twee rapporten elkaar tegen?",
        options={
            "error": "Ja: een van de twee bedrijven moet het mis hebben",
            "right": "Niet echt: andere tijd, andere modellen, elk gezien door zijn maker",
            "both": "Nee: allebei laten ze zien dat AI nu zelf bij de meeste doelwitten binnenkomt",
        },
        correct="right",
        why={
            "right": "Google keek naar Gemini, met gegevens tot januari 2025; Anthropic naar Claude Code, in september 2025; en elk ziet alleen zijn eigen model, zoals Anthropic zelf schrijft. Lees een rapport naar wie het schreef, over welk model, en wanneer.",
            "error": "Allebei kunnen ze kloppen: ze beschrijven andere maanden en andere modellen, elk van binnen één bedrijf. Zie *Wat aanvallers met AI doen* in *Het idee*.",
            "both": "De campagne van Anthropic richtte zich op zo'n dertig doelwitten en slaagde maar in een klein aantal gevallen, en de AI verzon soms inloggegevens. Google zag aanvallers sneller werken, geen nieuwe mogelijkheden. Zie *Wat aanvallers met AI doen* in *Het idee*.",
        },
    )
    choice_reports
    return (choice_reports,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Red teaming](/course/open/rai-red-teaming?lang=nl) vond je de zwakke plekken van je eigen assistent; dezelfde testgevallen zouden een aanvaller dienen, en dat is dual use. [De OWASP-lijsten voor LLM-apps](/course/open/rai-owasp-lists?lang=nl) zeggen naar welke risico's je eerst kijkt; de documenten hier zeggen wat je doet in het hele leven van een systeem, en de AI Act maakt een deel ervan wet voor systemen met een hoog risico. De volgende les, *Waar komt dit plaatje vandaan?*, gaat over wat AI maakt: hoe een maker een plaatje kan ondertekenen, zodat iedereen kan nagaan waar het vandaan komt.

    **Een voorbeeld uit het dagelijks leven.** Keukenmessen zijn overal te koop, en je kunt er iemand mee verwonden. Een kok leert er veilig mee werken, een restaurant volgt de hygiëneregels die wet zijn, en het handboek van een koksvereniging is advies. Beveiliging van AI heeft alle drie: vaardigheden, advies en, voor sommige systemen, wet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dual, choice_reports, result_article, result_groups, result_uncovered):
    L.wrap_up(result_uncovered, result_article, result_groups, choice_dual, choice_reports)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
