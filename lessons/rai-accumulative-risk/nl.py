# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-accumulative-risk. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Existential Risk, Risk

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-accumulative-risk", "nl")
    C = L.checks   # een verzonnen samenleving die jaarlijkse schade uitholt, en een schok erbovenop


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="Een argument heeft drie premissen, met voorwaardelijke subjectieve kansen van 0,5; 0,8 en 0,5, elk gegeven de premissen ervoor. Hoe waarschijnlijk is het volgens de methode van Carlsmith dat alle drie kloppen?",
        options={
            "sum": "1,8: tel de drie kansen op",
            "right": "0,2: vermenigvuldig de drie",
            "mean": "0,6: neem het gemiddelde van de drie",
        },
        correct="right",
        why={
            "right": "Carlsmith vermenigvuldigt: 0,5 × 0,8 × 0,5 = 0,2. Elke premisse moet er ook nog bij kloppen, gegeven de premissen ervoor, dus elke premisse kan de kans alleen verlagen of gelijk houden.",
            "sum": "Een kans kan niet groter zijn dan 1. Elke premisse moet er ook nog bij kloppen, gegeven de premissen ervoor, dus Carlsmith vermenigvuldigt: 0,5 × 0,8 × 0,5 = 0,2. Zie [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl).",
            "mean": "Het gemiddelde zegt hoe waarschijnlijk een gewone premisse is, niet alle drie samen. Alle drie moeten kloppen, elk gegeven de premissen ervoor, dus Carlsmith vermenigvuldigt: 0,5 × 0,8 × 0,5 = 0,2. Zie [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl).",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_gradual():
    quick_gradual = L.choice(
        "quick-gradual",
        question="Waarom zouden mensen volgens het argument van Kulveit en anderen hun zeggenschap over de economie, de staat en de cultuur kunnen verliezen, ook als geen enkele AI macht zoekt?",
        options={
            "better": "Omdat AI-systemen mensen in elke taak zouden overtreffen",
            "right": "Omdat economie, staat en cultuur mensen niet meer nodig zouden hebben",
            "care": "Omdat het mensen niet meer zou kunnen schelen wie die systemen bestuurt, als AI het meeste werk deed",
        },
        correct="right",
        why={
            "right": "Volgens Kulveit en anderen bleven economieën, staten en culturen in lijn met menselijke belangen omdat ze mensen nodig hadden om mee te doen. Zodra machines de plaats van mensen kunnen innemen, kan die band losser worden, ook zonder enig gecoördineerd zoeken naar macht.",
            "better": "Beter zijn in taken is niet hun punt. Volgens Kulveit en anderen bleven economieën, staten en culturen in lijn met menselijke belangen omdat ze mensen nodig hadden om mee te doen; zodra machines de plaats van mensen kunnen innemen, kan die band losser worden. Zie [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl).",
            "care": "Hun argument gaat niet over wat mensen belangrijk vinden. Volgens Kulveit en anderen bleven economieën, staten en culturen in lijn met menselijke belangen omdat ze mensen nodig hadden om mee te doen; zodra machines de plaats van mensen kunnen innemen, kan die band losser worden. Zie [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl).",
        },
    )
    quick_gradual
    return (quick_gradual,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het race-einde van *AI 2027*, uit de vorige les, roeit de AI in 2030 de mensheid uit: één verhaal met één beslissende wending. Atoosa Kasirzadeh, van Carnegie Mellon University, zet naast zulke verhalen een tweede soort verhaal, waarin schade zich opbouwt uit veel kleinere verstoringen die elk onbeduidend lijken, tot één schok meer is dan een samenleving kan opvangen. Sommige gevolgen van AI voor mensen zijn nu al te meten. Het International AI Safety Report 2026 (februari 2026), onder voorzitterschap van Yoshua Bengio, beschrijft een klinische studie waarin artsen een paar maanden diagnoses stelden met hulp van AI; daarna werd gemeten hoe goed ze tumoren zonder AI vonden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Hoe veranderde volgens de studie die het rapport beschrijft het vermogen van de artsen om tumoren zonder AI te vinden?",
        options={
            "same": "Het veranderde niet",
            "right": "Het was ongeveer 6% lager",
            "half": "Het was ongeveer half zo goed",
        },
        reveal="**Ongeveer 6% lager.** Volgens het rapport vond één klinische studie dat het vermogen van artsen om tumoren zonder AI op te sporen ongeveer 6% lager was na een paar maanden waarin ze diagnoses stelden met hulp van AI. Zo'n verandering brengt een samenleving op zichzelf niet ten val. Deze les kijkt naar wat veel veranderingen samen zouden kunnen doen, en naar waarom zulke risico's zich volgens Kasirzadeh ongemerkt kunnen opbouwen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Beslissend en accumulatief

    Het paper *Two Types of AI Existential Risk: Decisive and Accumulative* van Kasirzadeh verscheen voor het eerst op 15 januari 2024 op arXiv; deze les volgt versie 3 van 17 januari 2025, waarin staat dat het paper in *Philosophical Studies* gaat verschijnen. Het paper noemt twee hypotheses. Volgens de eerste, de beslissende hypothese, gaan existentiële risico's van kunstmatige superintelligentie over de mogelijkheid van abrupte gebeurtenissen op grote schaal die de mensheid uitroeien of het potentieel van de mensheid onherstelbaar laten afnemen. Als voorbeeld van deze gangbare opvatting haalt Kasirzadeh Toby Ord (2020) aan, die een catastrofe in de gebruikelijke zin opvat als één beslissende gebeurtenis, niet als een combinatie van gebeurtenissen die samen slecht uitpakt.

    Volgens de tweede, de accumulatieve hypothese, komen existentiële risico's van AI voort uit de opbouw van een reeks kleinere, minder ernstige verstoringen in de loop van de tijd, die samen en geleidelijk de veerkracht van het systeem verzwakken, totdat een directe aanleiding een onherstelbare ineenstorting veroorzaakt. Veerkracht is het vermogen van een systeem om een schok op te vangen en te herstellen. Versie 1 formuleerde de hypothese anders. Kasirzadeh vergelijkt het met klimaatverandering, waarbij elke afzonderlijke uitstoot van broeikasgassen klein lijkt, maar alle uitstoot samen leidt tot grote en mogelijk onomkeerbare veranderingen in het klimaat van de aarde. In de samenvatting van het paper heet het een scenario van de kikker die langzaam gekookt wordt. Let op hoe de hypothese eindigt: de opbouw is traag, maar de ineenstorting komt met een directe aanleiding.

    Het paper van Bostrom uit 2002, bekend uit de vorige les, heeft al een langzaam einde. Bostrom deelt existentiële risico's in vier soorten in, en een daarvan, de whimpers, loopt langzaam af. Volgens Bostrom ontstaat er dan een posthumane beschaving, die zich in een richting ontwikkelt die geleidelijk maar onherroepelijk leidt tot het volledig verdwijnen van wat we waardevol vinden, of tot een toestand waarin daarvan maar een piepklein deel verwezenlijkt wordt van wat mogelijk was geweest. Posthumaan is het woord van Bostrom voor wat de mensheid zou kunnen worden.

    ### Een perfecte storm

    Om de accumulatieve hypothese concreet te maken, beschrijft Kasirzadeh een gedachte-experiment, The perfect storm MISTER. De letters staan voor de Engelse woorden voor manipulatie, dreigingen voor de veiligheid, surveillance en afbrokkelend vertrouwen, economische ontwrichting en het schenden van rechten. Het verhaal speelt in 2040, in een sterk verbonden wereld waarin AI-tools, AI-assistenten, AI-agents en het internet of things bijna elk deel van het dagelijks leven hebben veranderd. In versie 1 was het jaar 2035. In die wereld vervuilen deepfakes en desinformatie het informatielandschap, worden slimme apparaten, van spiegels tot koelkasten, zwakke plekken voor aanvallen, breidt surveillance zich uit en brokkelt vertrouwen af, raakt de economie ontwricht en worden rechten geschonden. Dan begint de catastrofe volgens het verhaal met een verwoestende cyberaanval met AI, die tegelijk vitale elektriciteitsnetten op drie continenten treft. Kasirzadeh noemt die aanval het kantelpunt. De stroomuitval zet volgens het verhaal een domino-effect in gang, met grote economische crashes. Het wantrouwen dat vóór de stroomuitval gezaaid was, maakt het moeilijker om de orde te herstellen, en storingen in AI-systemen slaan over naar de zorg en de communicatienetwerken.

    Kasirzadeh telt niet elke schade als een stap naar een catastrofe. Volgens het paper hebben niet alle sociale of ethische risico's existentiële gevolgen, en richt MISTER zich op de risico's die het systeem als geheel flink kunnen raken.

    ### Effecten die je kunt meten

    Het International AI Safety Report 2026 rekent effecten zoals dat uit de gok tot de systeemrisico's, in dat rapport risico's die voortkomen uit het brede gebruik van zeer krachtige general-purpose AI in de samenleving en de economie. Het rapport merkt op dat dit anders is dan hoe de AI Act van de Europese Unie het begrip gebruikt. Naast de studie uit de gok beschrijft het rapport automation bias, waarbij mensen op de uitvoer van AI vertrouwen zonder die genoeg te controleren. In een gerandomiseerd experiment met 2.784 deelnemers aan een taak waarbij AI hielp met annoteren, verbeterden de deelnemers volgens het rapport foute suggesties van de AI minder vaak als dat extra moeite kostte, of als ze positiever over AI dachten.

    Het rapport waarschuwt ook voor gedeelde zwakke plekken, onder de kop dat het brede leunen op een klein aantal modellen single points of failure oplevert. Als hetzelfde model veel toepassingen aandrijft, kunnen fouten in dat model zich volgens het rapport verspreiden naar alle toepassingen die ervan afhangen. En gevolgen worden soms pas langzaam zichtbaar. Over de arbeidsmarkt merkt het rapport op dat aanwijzingen voor de gevolgen waarschijnlijk geleidelijk aan het licht komen, waardoor ook moeilijk te bepalen is wanneer beleid zou moeten reageren. Het rapport koppelt deze effecten niet aan de hypothese van Kasirzadeh; of het stappen van die soort zouden kunnen zijn, is de vraag van deze les, niet van het rapport.

    ### Vastgelegd

    Sommige veranderingen zijn moeilijk terug te draaien. Lukas Finnveden, Jess Riedel en Carl Shulman betogen in *AGI and Lock-in* (Forethought; gedateerd 8 oktober 2022, laatst bijgewerkt op 11 maart 2025) dat de verre toekomst van intelligent leven nu nog onvoorspelbaar en onbepaald is, en dat de uitvinding van algemene kunstmatige intelligentie (AGI) daar verandering in zou kunnen brengen, door extreme vormen van lock-in technisch haalbaar te maken. In hun definitie is een eigenschap van de wereld vastgelegd vanaf een moment waarop de grote onzekerheid over hoe die eigenschap op de lange termijn uitpakt, flink is afgenomen. Uitsterven van de mensheid is volgens hen één voorbeeld van een lock-in. Bij wereldwijde value lock-in hebben alle machtige partijen waarden uit een veel kleinere verzameling dan eerst, is het heel onwaarschijnlijk dat een van hen waarden van daarbuiten overneemt, en duurt dat minstens miljoenen jaren.

    Ze splitsen hun hoofdclaim in drie beweringen, die allemaal uitgaan van AGI. Een daarvan is dat het mogelijk wordt om heel genuanceerde beschrijvingen van waarden en doelen tot ver in de toekomst te bewaren zonder informatie te verliezen; een andere is dat een institutie die is opgezet door een grote meerderheid van de economische en militaire machten van de wereld, en die de macht krijgt om zichzelf te verdedigen, de eigen agenda minstens miljoenen jaren zou kunnen nastreven, en misschien wel biljoenen jaren. Ze zeggen erbij dat het hun vooral gaat om wat haalbaar is, niet om wat waarschijnlijk is. Volgens hen is er voor lock-in geen AI nodig die ver boven mensen uitstijgt; AI op menselijk niveau lijkt genoeg. Lock-in is ook niet alleen een gevaar. Stabiele instituties zouden volgens hen een existentieel risico kunnen vormen als ze slecht gekozen en te weinig flexibele waarden uitvoeren. Maar als we willen dat humane waarden of instituties zoals de liberale democratie op lange termijn overleven, kunnen sommige soorten stabiliteit volgens hen juist cruciaal zijn om ze te behouden. En lock-in hoeft volgens hen niet in één keer te gebeuren: een groep zou de samenleving eerst op andere manieren heel stabiel kunnen maken, en in honderden jaren steeds zekerder kunnen worden van de eigen waarden. Dan is er misschien geen duidelijk moment waarop value lock-in gebeurt, terwijl de waarden uiteindelijk toch heel stabiel zijn.

    Macht kan zich ook in één klap concentreren. Tom Davidson, Lukas Finnveden en Rose Hadshar schrijven in *AI-Enabled Coups: How a Small Group Could Use AI to Seize Power* (Forethought, 15 april 2025) dat een staatsgreep met AI gepleegd zou kunnen worden door een heel kleine groep, of zelfs door één persoon, en dat dat ook in gevestigde democratieën zou kunnen. Zo'n staatsgreep zou volgens hen ook blijvend kunnen zijn, omdat de leiders misschien voor onbepaalde tijd aan de macht kunnen blijven, door AI-systemen in te zetten die tot ver in de toekomst de doelen van die leiders bewaren en nastreven.

    ### Waar je moet kijken

    Volgens Kasirzadeh vragen de twee soorten om ander toezicht. Beslissend risico vraagt volgens het paper om centrale controle, vergelijkbaar met de afspraken tegen de verspreiding van kernwapens. Accumulative risk vraagt juist om verspreid toezicht dat kan volgen hoe de gevolgen van AI in verschillende domeinen zich opstapelen. Kasirzadeh noemt een gat, het probleem van versnipperde risicokaders: AI-risico wordt opgedeeld in aparte soorten, zoals sociale en ethische, catastrofale, extreme en existentiële risico's, elk met een eigen aanpak, en volgens Kasirzadeh dekken die aanpakken samen niet alles af, zodat er gevaarlijke blinde vlekken ontstaan waar risico's zich ongemerkt kunnen opstapelen.

    Kasirzadeh zet geen getallen bij hoe waarschijnlijk elk van beide soorten is. Een figuur in het paper die de twee naast elkaar zet, laat volgens Kasirzadeh kwalitatieve verschillen zien en geen voorspellingen over wanneer of hoe ernstig, en hoe je accumulative risk kwantificeert, is volgens het paper een van de open vragen voor verder onderzoek.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **accumulative risk** | existentieel of catastrofaal risico dat zich opbouwt uit veel kleinere schades die een samenleving in de loop van de tijd verzwakken, tot een directe aanleiding een onherstelbare ineenstorting veroorzaakt; naar Kasirzadeh |
        | beslissend risico | existentieel of catastrofaal risico door één abrupte gebeurtenis, zoals een machtsovername door een misaligned AI; naar Kasirzadeh |
        | veerkracht | het vermogen van een systeem om een schok op te vangen en te herstellen |
        | directe aanleiding | de schok die een verzwakt systeem laat instorten |
        | systeemrisico | in het International AI Safety Report 2026 een risico dat voortkomt uit het brede gebruik van zeer krachtige general-purpose AI in de samenleving en de economie |
        | automation bias | vertrouwen op geautomatiseerde systemen, ook AI, zonder hun uitvoer genoeg te controleren |
        | single point of failure | een onderdeel waarvan het falen het hele systeem ontwricht |
        | **value lock-in** | waarden die voor heel lang vast komen te liggen: alle machtige partijen hebben waarden uit een veel kleinere verzameling dan eerst, en nemen heel waarschijnlijk geen andere over; naar Finnveden, Riedel en Shulman |
        """),
        L.watch_out("MISTER is een gedachte-experiment, geen voorspelling: het jaar schoof tussen de versies van 2035 naar 2040, en Kasirzadeh geeft geen kansen. Het model in oefening 1 en 2 is verzonnen, niet dat van Kasirzadeh. En Finnveden en anderen betogen vooral dat extreme lock-in met AGI haalbaar zou kunnen worden, niet hoe waarschijnlijk dat is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien welke soort risico een verhaal beschrijft:

    1. Zoek het einde: wat breekt er uiteindelijk, en hoe snel?
    2. Kijk terug naar de kleinere schades die eraan voorafgingen.
    3. Vraag je af of een gezonde samenleving dat einde had opgevangen.
    4. Werkt het einde alleen door wat eraan voorafging, noem het dan accumulatief; is één gebeurtenis op zichzelf genoeg, dan beslissend.
    5. Vraag je af wat daarna niet meer terug te draaien is: waarden, macht of vaardigheden.
    6. Lees elk verhaal als een manier waarop het zou kunnen lopen, niet als een voorspelling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent één slecht en één goed jaar uit voor een verzonnen samenleving. De samenleving begint bij een veerkracht van 100, het hoogste wat mogelijk is. In het slechte jaar halen de verstoringen er 5 af en wint de samenleving 2 terug; in het goede jaar halen ze er 1 af en wint de samenleving 2 terug, maar boven 100 kan het niet.
    """)
    return


@app.cell
def example_years():
    start = 100   # verzonnen: de veerkracht van een gezonde samenleving, en het hoogste wat mogelijk is
    recovery = 2   # verzonnen: wat de samenleving in een jaar terugwint
    bad_year = min(start, start - 5 + recovery)   # een schade van 5: omlaag naar 97
    good_year = min(start, start - 1 + recovery)   # een schade van 1: 101 ligt boven de top, dus 100
    print(bad_year, good_year)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het slechte jaar eindigt op 97 en het goede op 100: `min` zorgt dat de veerkracht niet boven het niveau van een gezonde samenleving uitkomt. In oefening 1 begint elk jaar waar het jaar ervoor eindigde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_resilience():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. resilience: jaar na jaar

        Een verzonnen model van de hypothese van Kasirzadeh, niet dat van Kasirzadeh zelf: het paper heeft geen model met getallen. Een samenleving begint met een veerkracht van 100. Elk jaar halen de verstoringen van dat jaar er een schade af, en wint de samenleving `recovery` terug, maar nooit verder dan tot 100.

        Schrijf `resilience(harms, recovery)`, die een lijst `result` teruggeeft met de veerkracht aan het eind van elk jaar. `harms` is een lijst met de schade van elk jaar. Laat `level` bij 100 beginnen; reken in een `for`-loop met `harm` over `harms` de nieuwe `level` uit en voeg die toe aan `result`.

        De check roept `resilience([6, 3, 8, 5, 9, 4, 7, 10, 6, 8], 2)` aan, tien verzonnen jaren met een herstel van 2, en daarna andere gevallen.
        """),
        L.task("De veerkracht aan het eind van elk jaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resilience():
    box_resilience = L.code_box("resilience", lines=7)
    box_resilience
    return (box_resilience,)


@app.cell(hide_code=True)
def check_resilience(box_resilience):
    result_resilience = L.check("resilience", box_resilience.value)
    result_resilience
    return (result_resilience,)


@app.cell(hide_code=True)
def hints_resilience():
    L.hints(
        "resilience",
        [
            "Begin klein: een herstel van 2, en twee jaren met een schade van 5 en 1. Waar staat de samenleving na het eerste jaar, en waar na het tweede?",
            "Begin `result` als een lege lijst en `level` bij 100. Reken in een `for`-loop met `harm` over `harms` de nieuwe `level` uit met de oude, zoals in het uitgewerkte voorbeeld, nooit boven 100, en voeg die toe. Geef na de loop `result` terug.",
            "`        level = min(100, ___)`",
        ],
        why="`result = []` en `level = 100`, dan `for harm in harms:` met `level = min(100, level - harm + recovery)` en `result.append(level)`; na de loop `return result`. Met een herstel van 2 en een schade van 5 en 1 eindigt het eerste jaar op 97 en het tweede op 98: elk jaar begint waar het jaar ervoor eindigde.",
    )
    return


@app.cell(hide_code=True)
def text_after_resilience():
    mo.md(r"""
    Voor de tien verzonnen jaren zakt de veerkracht van 96 na het eerste jaar naar 54 na het tiende. Geen enkel jaar haalt er meer dan 10 af, en de samenleving wint elk jaar 2 terug, en toch is na tien jaar bijna de helft weg. In het geval `[1, 1, 5]` met een herstel van 2 blijven de eerste twee jaren op 100: het herstel maakt een kleine schade goed, maar kan niet verder dan tot 100.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_collapse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. collapse_year: wanneer de schok te veel is

        Een schok, zoals de aanval op de elektriciteitsnetten in het verhaal van Kasirzadeh, haalt in één keer `shock` van de veerkracht af. Schrijf `collapse_year(levels, shock)`, die het eerste jaar teruggeeft, geteld vanaf 1, waarin de schok de veerkracht op 0 of lager zou brengen, en `None` als dat in geen enkel jaar gebeurt. `levels` is een lijst met de veerkracht aan het eind van elk jaar, zoals `resilience` die teruggeeft. Tel de jaren in `year`, beginnend bij 0; tel in een `for`-loop met `level` over `levels` eerst 1 op bij `year`.

        De check roept `collapse_year([96, 95, 89, 86, 79, 77, 72, 64, 60, 54], 60)` aan, de veerkracht uit het eerste geval van oefening 1 en een schok van 60, en daarna andere gevallen.
        """),
        L.task("Het eerste jaar waarin de schok de samenleving zou omverwerpen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_collapse():
    box_collapse = L.code_box("collapse", lines=7)
    box_collapse
    return (box_collapse,)


@app.cell(hide_code=True)
def check_collapse(box_collapse):
    result_collapse = L.check("collapse", box_collapse.value)
    result_collapse
    return (result_collapse,)


@app.cell(hide_code=True)
def hints_collapse():
    L.hints(
        "collapse",
        [
            "Begin klein: een veerkracht van 90, 70 en 50, en een schok van 60. Wat zou er na de schok in elk van de drie jaren over zijn, en in welk jaar zou er voor het eerst niets meer over zijn?",
            "Begin `year` bij 0. Tel in een `for`-loop met `level` over `levels` 1 op bij `year`, kijk dan of de schok `level` op 0 of lager zou brengen, en geef zo ja `year` terug. Geef na de loop `None` terug.",
            "`        if ___:`",
        ],
        why="`year = 0`, dan `for level in levels:` met `year = year + 1` en `if level <= shock:` met daaronder `return year`; na de loop `return None`. Met een veerkracht van 90, 70 en 50 en een schok van 60 zou er in de eerste twee jaren 30 en 10 over zijn, en in het derde -10, dus het antwoord is jaar 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_collapse():
    mo.md(r"""
    Voor de tien verzonnen jaren en een schok van 60 is het antwoord jaar 9, als de veerkracht precies 60 is. In jaar 1 zou dezelfde schok er 36 hebben overgelaten. De schok werd niet groter; wat zich ervoor had opgebouwd wel. En het geval `[50, 90]` valt in jaar 1: een samenleving kan ook vanaf het begin zwak zijn.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: elk jaar schade

    Het herstel blijft 2 en de schok 60, en elk van 30 jaren heeft dezelfde schade. Verander die schade, en kijk wanneer de schok de samenleving omverwerpt, als dat gebeurt.
    """)
    return


@app.cell(hide_code=True)
def demo_harm():
    harm = L.explore("harm", "harm: wat elk jaar eraf haalt",
                     lambda _v: C.demo_frame(_v, L.lang), start=2, values=C.DEMO_HARMS, role="",
                     task="Zoek de kleinste jaarlijkse schade waarbij de schok de samenleving binnen de 30 jaar omverwerpt.")
    harm
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met een schade van 1 of 2 per jaar maakt het herstel van 2 het goed, en blijft de veerkracht 100. Met 3 zakt de veerkracht 1 per jaar, naar 70 na 30 jaar, en vangt de samenleving de schok elk jaar op. Vanaf een schade van 4 werpt de schok de samenleving binnen de 30 jaar omver: in jaar 20 bij 4, en in jaar 14 bij 5. Geen enkel jaar haalt er meer dan 5 van de 100 af, en de schok is elke keer dezelfde; wat verschilt, is hoeveel zich ervoor heeft opgebouwd. Echte samenlevingen zijn niet zo eenvoudig, en het paper heeft geen model zoals dit; volgens Kasirzadeh werken kleinere verstoringen in de loop van de tijd ook op elkaar in en versterken ze elkaar, wat dit model weglaat. Het model laat alleen de vorm van de hypothese zien: een trage opbouw, dan een snel einde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_locked():
    mo.md(r"""
    ### Oefening 3. Democratie voor altijd

    De grootmachten van de wereld willen dat de liberale democratie miljoenen jaren blijft bestaan. Samen zetten ze met AI instituties op die zichzelf verdedigen en die later niemand meer kan veranderen.
    """)
    return


@app.cell(hide_code=True)
def choice_locked_cell():
    choice_locked = L.choice(
        "locked",
        question="Zou dat value lock-in zijn, volgens de definitie van Finnveden, Riedel en Shulman?",
        options={
            "good": "Nee: lock-in betekent dat slechte waarden winnen, en deze waarden zijn humaan",
            "right": "Ja: de verre toekomst is dan nauwelijks te veranderen, wat de waarden ook zijn",
            "death": "Nee: in hun definitie telt alleen het uitsterven van de hele mensheid als een lock-in",
        },
        correct="right",
        why={
            "right": "In hun definitie is een eigenschap vastgelegd zodra de onzekerheid over hoe die op de lange termijn uitpakt flink is afgenomen, wat de waarden ook zijn. Finnveden, Riedel en Shulman zien beide kanten. Stabiele instituties met slecht gekozen en te weinig flexibele waarden zouden volgens hen een existentieel risico kunnen vormen, en toch kunnen sommige soorten stabiliteit cruciaal zijn om humane waarden of de liberale democratie te laten overleven.",
            "good": "Lock-in gaat over hoeveel de verre toekomst nog kan veranderen, niet over of de waarden goed zijn. Volgens Finnveden, Riedel en Shulman kunnen sommige soorten stabiliteit juist cruciaal zijn om humane waarden of instituties zoals de liberale democratie op lange termijn te behouden. Zie *Vastgelegd*.",
            "death": "Uitsterven is volgens hen één voorbeeld van een lock-in, niet het enige: bij wereldwijde value lock-in hebben alle machtige partijen waarden uit een veel kleinere verzameling dan eerst. Zie *Vastgelegd*.",
        },
    )
    choice_locked
    return (choice_locked,)


@app.cell(hide_code=True)
def text_exercise_update():
    mo.md(r"""
    ### Oefening 4. Een foute update

    Stel je vijftien jaar voor waarin nepvideo's die met AI gemaakt zijn ervoor zorgen dat mensen het nieuws minder vertrouwen, een paar AI-modellen het grootste deel gaan draaien van wat ziekenhuizen en banken gebruiken, en medewerkers steeds minder controleren wat die modellen voorstellen. Dan legt één foute update van een veelgebruikt model in één keer betalingen over de hele wereld stil, en in het wantrouwen dat volgt, valt de ineenstorting niet meer terug te draaien.
    """)
    return


@app.cell(hide_code=True)
def choice_update_cell():
    choice_update = L.choice(
        "update",
        question="Bij welke van de twee hypotheses van Kasirzadeh past dit verhaal?",
        options={
            "decisive": "Bij de beslissende, omdat het eindigt met één plotselinge gebeurtenis, de update",
            "right": "Bij de accumulatieve, omdat de jaren ervoor de samenleving hadden verzwakt",
            "neither": "Bij geen van beide, omdat geen enkel AI-systeem in het verhaal macht zoekt of grijpt",
        },
        correct="right",
        why={
            "right": "De update is de directe aanleiding, maar haalt alles alleen onderuit omdat jaren van kleinere schades het vertrouwen hadden uitgehold, en een paar modellen bijna alles waren gaan draaien. Volgens de accumulatieve hypothese verzwakken kleinere verstoringen de veerkracht van het systeem geleidelijk, tot een directe aanleiding een onherstelbare ineenstorting veroorzaakt. Een trage opbouw, dan een snel einde.",
            "decisive": "Het einde is plotseling, maar in de beslissende hypothese veroorzaakt één abrupte gebeurtenis de catastrofe. Hier haalt de update alles onderuit omdat er jaren van kleinere schades aan voorafgingen: de update is de directe aanleiding uit de accumulatieve hypothese. Zie *Beslissend en accumulatief*.",
            "neither": "De accumulatieve hypothese heeft geen AI nodig die macht zoekt; het gaat om kleinere verstoringen die zich opbouwen tot een directe aanleiding. Hier gingen jaren van wantrouwen en van leunen op een paar modellen aan de foute update vooraf. Zie *Beslissend en accumulatief*.",
        },
    )
    choice_update
    return (choice_update,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De argumenten voor catastrofaal risico](/course/open/rai-xrisk-arguments?lang=nl) liet zien waarom AI tot een catastrofe zou kunnen leiden, en eindigde bij Christiano en bij Kulveit en anderen, voor wie falen niet in één klap hoeft te komen. Deze les zette de twee soorten naast elkaar: de beslissende en de accumulatieve hypothese van Kasirzadeh, het gedachte-experiment MISTER waarin kleine schades zich opbouwen, totdat één aanval het kantelpunt is, kleine effecten die het International AI Safety Report 2026 beschrijft, en value lock-in, een toekomst die heel moeilijk te veranderen wordt. Niet iedereen kijkt zo naar deze risico's; de volgende les, *Andere standpunten*, bekijkt andere opvattingen.

    **Een voorbeeld uit het dagelijks leven.** Een brug roest elk jaar een beetje, en bij onderhoud wordt een deel daarvan hersteld. De roest van één jaar ziet er nooit gevaarlijk uit. Dan haalt een storm de brug neer, terwijl de brug twintig jaar eerder een even zware storm doorstond. De storm was de aanleiding, maar door de jaren roest was die storm genoeg.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_locked, choice_update, result_collapse, result_resilience):
    L.wrap_up(result_resilience, result_collapse, choice_locked, choice_update)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
