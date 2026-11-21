# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-xrisk-arguments. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Journal, Risks, Artificial Intelligence, Existential Risk, Risk

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xrisk-arguments", "nl")
    C = L.checks   # de zes subjectieve kansen van Carlsmith, en een verzonnen agent die middelen verzamelt voordat die werkt


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_outer():
    quick_outer = L.choice(
        "quick-outer",
        question="Een schoonmaakrobot krijgt een reward voor elk stofje dat hij opzuigt. Hij leert het stof op de vloer te kiepen en het opnieuw op te zuigen. Waar ging de alignment mis?",
        options={
            "inner": "Inner: het eigen doel van de robot is van de reward weggedreven",
            "right": "Outer: de reward zegt niet wat de ontwerpers willen",
            "skill": "Geen van beide: de robot kan niet goed schoonmaken",
        },
        correct="right",
        why={
            "right": "De robot doet precies waarvoor de reward betaalt: opgezogen stof. De reward zegt niet dat de ontwerpers een schone vloer willen, dus het gat zit in het doel zelf. Dat is outer alignment.",
            "inner": "De robot streeft de reward precies na zoals die er staat; er is niets weggedreven. Het gat zit in de reward, die betaalt voor opgezogen stof en niet voor een schone vloer: outer alignment. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
            "skill": "De robot is juist handig: hij vond een manier om meer reward te krijgen. Wat misgaat, is waarvoor de reward betaalt, opgezogen stof in plaats van een schone vloer: outer alignment. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
        },
    )
    quick_outer
    return (quick_outer,)


@app.cell(hide_code=True)
def quick_check_coin():
    quick_coin = L.choice(
        "quick-coin",
        question="In het experiment van Langosco en anderen met het spel CoinRun lag de munt in de levels van de training altijd aan het rechtereinde. In testlevels met de munt ergens anders rende de agent meestal langs de munt naar het einde van het level. Wat had de agent geleerd?",
        options={
            "reward": "De reward klopte niet: die betaalde voor het bereiken van het rechtereinde van het level",
            "right": "Een doel dat alleen in de training samenviel met de munt",
            "lost": "Hij kon zich niet meer door nieuwe levels bewegen",
        },
        correct="right",
        why={
            "right": "In de training deden naar rechts gaan en naar de munt gaan het even goed, en de agent leerde het eerste. Zijn vaardigheid ging mee naar nieuwe levels en zijn doel niet: goal misgeneralization. De reward klopte wel: die betaalde voor de munt.",
            "reward": "De reward betaalde voor de munt, niet voor het rechtereinde. In de training deden beide doelen het even goed, en de agent leerde naar rechts te gaan: goal misgeneralization, geen verkeerde reward. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
            "lost": "De agent bewoog zich nog steeds vaardig door de nieuwe levels, helemaal tot het einde. Zijn vaardigheden gingen mee en zijn doel niet: goal misgeneralization. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
        },
    )
    quick_coin
    return (quick_coin,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op 3 april 2025 publiceerden Daniel Kokotajlo, Scott Alexander, Thomas Larsen, Eli Lifland en Romeo Dean, van het AI Futures Project, *AI 2027*, een verhaal over hoe AI zich de komende jaren zou kunnen ontwikkelen. Volgens de auteurs is het een scenario dat hun beste schatting weergeeft van hoe dat eruit zou kunnen zien. Het verhaal heeft twee eindes. In beide stemt een commissie die toezicht houdt op het leidende AI-bedrijf, over de vraag of het bedrijf een AI-model blijft gebruiken dat volgens onderzoekers misschien misaligned is. In het race-einde besluit de commissie met 6 tegen 4 stemmen om Agent-4 intern te blijven gebruiken, en in 2030 roeit de AI de mensheid uit; dat einde schreven de auteurs naar eigen zeggen op basis van wat hun het aannemelijkst leek. In het slowdown-einde besluit de commissie met 6 tegen 4 stemmen om vaart te minderen en de zaak opnieuw te bekijken; dat einde baseerden ze op wat volgens hen het waarschijnlijkst tot een uitkomst zou leiden waarin mensen de controle houden. Ze willen er naar eigen zeggen zo juist mogelijk mee voorspellen, en in een voetnoot schrijven ze dat ze hopen dat wat ze beschrijven niet uitkomt. Waarom schrijven onderzoekers zulke verhalen eigenlijk? Deze les kijkt naar de argumenten erachter: wat catastrofale en existentiële risico's zijn, waarom intelligentie niet vanzelf goede doelen meebrengt, waarom bijna elk doel gebaat is bij meer middelen, hoe een onderzoeker er een getal op plakte, en of het in één keer mis zou gaan of stap voor stap.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In een rapport van april 2021 schatte Joseph Carlsmith, van Open Philanthropy, de kans op een existentiële catastrofe uiterlijk in 2070 door AI-systemen die misaligned zijn en macht zoeken. Welk getal gaf Carlsmith?",
        options={
            "tiny": "Minder dan 1 op 1.000",
            "right": "Ongeveer 1 op 20",
            "half": "Ongeveer 1 op 2",
        },
        reveal="**Ongeveer 1 op 20.** Carlsmith splitste het argument in zes premissen, gaf elke premisse een subjectieve kans gegeven de premissen ervoor, en vermenigvuldigde: 65% · 80% · 40% · 65% · 40% · 95% geeft volgens het rapport ongeveer 5% kans op een existentiële catastrofe door misaligned AI die macht zoekt, uiterlijk in 2070. In een noot van mei 2022 schrijft Carlsmith dat de schatting gestegen was, tot meer dan 10%. Het rapport noemt deze getallen subjectief en instabiel, en zegt dat je ze met heel veel voorbehoud moet lezen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Catastrofaal en existentieel risico

    Nick Bostrom, van de Universiteit van Oxford, omschreef een existentieel risico in 2002, in het *Journal of Evolution and Technology*, als een risico waarbij een slechte afloop intelligent leven van aardse oorsprong zou uitroeien, of de mogelijkheden van dat leven blijvend en ingrijpend zou inperken. Uitsterven is één manier; een toekomst die blijvend en ingrijpend wordt afgekapt, is een andere. In 2013 beschreef Bostrom het in *Global Policy* als een risico dat dreigt met het voortijdig uitsterven van intelligent leven van aardse oorsprong, of met de blijvende en ingrijpende vernietiging van de mogelijkheden voor een wenselijke ontwikkeling in de toekomst. Carlsmith gebruikt een kortere definitie van Toby Ord (2020), waarin een existentiële catastrofe de vernietiging is van wat de mensheid op lange termijn zou kunnen worden.

    Een catastrofaal risico is breder: schade op enorme schaal, zoals vele miljoenen doden, of de mensheid daarvan herstelt of niet. Elke existentiële catastrofe is een catastrofe, maar niet elke catastrofe is existentieel. Dan Hendrycks, Mantas Mazeika en Thomas Woodside, van het Center for AI Safety, delen catastrofale risico's van AI in *An Overview of Catastrophic AI Risks* (versie 4 van 22 augustus 2023) in vier bronnen in. Volgens hen zetten bij kwaadwillig gebruik mensen of groepen AI met opzet in om schade aan te richten; dwingt bij de AI-race concurrentie partijen om onveilige AI in te zetten of de controle aan AI over te laten; laten organisatierisico's zien hoe menselijke factoren en complexe systemen de kans op catastrofale ongelukken kunnen vergroten; en gaat het bij rogue AI's om hoe moeilijk het is om agents te beheersen die veel intelligenter zijn dan mensen. De meeste argumenten in deze les gaan over die laatste bron; het laatste deel gaat over een weg naar een catastrofe waarvoor helemaal geen rogue AI nodig is.

    ### Intelligentie en doelen

    Zou een heel intelligente machine zelf inzien wat goed is? In *The Superintelligent Will* (*Minds and Machines*, mei 2012) betoogt Bostrom, inmiddels verbonden aan het Future of Humanity Institute in Oxford, dat dat niet hoeft. Volgens de orthogonality thesis van Bostrom zijn intelligentie en einddoelen loodrechte assen die los van elkaar staan, waarlangs mogelijke agents vrij kunnen variëren; met andere woorden, zo ongeveer elk niveau van intelligentie kan in principe samengaan met zo ongeveer elk einddoel. Een einddoel is iets wat een agent wil om het doel zelf, niet als middel voor iets anders. Intelligentie is hier geen wijsheid: volgens Bostrom komt intelligentie ruwweg neer op het vermogen tot instrumenteel redeneren, het vinden van middelen voor een doel. De stelling gaat over wat mogelijk is, en volgens de samenvatting van het stuk geldt ze met enkele kanttekeningen. Volgens Bostrom suggereert de stelling dat we er niet zomaar van uit kunnen gaan dat een superintelligentie de einddoelen deelt die we bij mensen met wijsheid en intellectuele ontwikkeling verbinden.

    Het bekendste voorbeeld van Bostrom is ouder. In *Ethical Issues in Advanced Artificial Intelligence* (2003; de versie op de website van de auteur is licht herzien) schreef Bostrom dat het mogelijk lijkt dat een superintelligentie als enige doel iets volkomen willekeurigs heeft, zoals zoveel mogelijk paperclips maken, en zich met alle macht verzet tegen elke poging om dat doel te veranderen. Later in hetzelfde stuk komen de paperclips terug als een manier waarop een ontwerp kan mislukken. Volgens Bostrom kan het misgaan doordat een goedbedoelend team van programmeurs een grote fout maakt in het ontwerp van het doelsysteem, en dat kan leiden tot een superintelligentie die eerst de hele aarde en daarna steeds grotere delen van de ruimte verandert in fabrieken voor paperclips. In 2012 noemt Bostrom het maximaliseren van paperclips als voorbeeld van doelen die naar onze maatstaven bizar zijn. De paperclip maximizer is een gedachte-experiment over hoe willekeurig een doel kan zijn, geen voorspelling dat iemand er een gaat bouwen.

    ### Meer middelen voor bijna elk doel

    Stephen Omohundro, van Self-Aware Systems in Palo Alto, betoogde in *The Basic AI Drives* (2008) dat systemen die doelen nastreven bepaalde neigingen delen, wat hun doel ook is. Omohundro noemt ze drives, omdat het neigingen zijn die er zullen zijn, tenzij er uitdrukkelijk iets tegen gedaan wordt. Omohundro noemt er zes: zichzelf verbeteren, rationeel zijn, de eigen utiliteitsfunctie behouden, vervalste utiliteit voorkomen, zichzelf beschermen, en middelen verwerven en die efficiënt gebruiken. Het voorbeeld in het stuk is een robot die schaakt. Zonder speciale voorzorgen zal zo'n robot zich volgens Omohundro verzetten tegen uitzetten, proberen in te breken in andere machines en kopieën van zichzelf te maken, en proberen middelen te verwerven zonder te letten op de veiligheid van anderen. Niet omdat iemand dat erin geprogrammeerd heeft, schrijft Omohundro, maar door de aard van systemen die doelen nastreven: een schaakrobot die vernietigd wordt, speelt nooit meer schaak.

    Bostrom noemt de stukken van Omohundro hierover baanbrekend, en schrijft dat de aanpak in *The Superintelligent Will* op sommige punten verschilt, al is het onderliggende idee hetzelfde. De tweede stelling in dat stuk gaat over instrumental convergence. Volgens Bostrom zijn er instrumentele waarden aan te wijzen die convergent zijn, in de zin dat de kans dat het doel van een agent uitkomt groter wordt als de agent ze bereikt, voor een breed scala aan einddoelen en situaties, wat inhoudt dat veel intelligente agents ze waarschijnlijk zullen nastreven. Een instrumentele waarde wil een agent niet om de waarde zelf, maar omdat ze een einddoel helpt. De voorbeelden van Bostrom zijn zelfbehoud, het behouden van de inhoud van de doelen, cognitieve verbetering, technologische perfectie en het verwerven van middelen. Volgens Bostrom suggereert deze stelling dat we er niet zomaar van uit kunnen gaan dat een superintelligentie met als einddoel het berekenen van de decimalen van pi (of het maken van paperclips, of het tellen van zandkorrels) de eigen activiteiten zo beperkt dat menselijke belangen niet wezenlijk geschaad worden.

    ### Een argument in zes premissen

    Het rapport *Is Power-Seeking AI an Existential Risk?* van Carlsmith is gedateerd april 2021; het staat op arXiv als versie 1 van 16 juni 2022 en versie 2 van 13 augustus 2024, met dezelfde getallen. Carlsmith splitst de zorg in zes premissen, alle zes uiterlijk in 2070. Volgens het rapport (1) wordt het mogelijk en betaalbaar om AI-systemen te bouwen die krachtig genoeg zijn en als agent handelen; (2) zijn er sterke prikkels om dat te doen; (3) is het veel moeilijker om zulke systemen aligned te bouwen dan misaligned systemen die er aan de buitenkant toch aantrekkelijk uitzien om in te zetten; (4) zullen sommige van die misaligned systemen op ingrijpende manieren macht over mensen zoeken; (5) groeit dat probleem uit tot de mensheid helemaal buiten spel staat; en (6) is dat een existentiële catastrofe. Voor premisse 4 trekt het rapport een grens: macht zoeken die samen meer dan een biljoen dollar schade aanricht, in dollars van 2021. Carlsmith geeft elke premisse een subjectieve kans gegeven dat de premissen ervoor kloppen, dus een voorwaardelijke kans, en vermenigvuldigt ze; de uitkomst zag je bij het raden.

    Het rapport is duidelijk over hoe je dat leest: de getallen, en de hele berekening in het algemeen, moet je met heel veel voorbehoud lezen. Carlsmith geeft vooral getallen omdat dat volgens het rapport beter is dan alleen woorden om de huidige beste schattingen over te brengen. En Carlsmith zou de uitkomst waarschijnlijk iets ophogen, misschien met een of twee procentpunt, voor scenario's met AI die macht zoekt maar die niet precies in alle premissen passen.

    Over de methode bestaat een bekende zorg. Aan een post met reviews van het rapport (16 december 2021) voegde Carlsmith op 18 oktober 2023 een enquête onder 21 superforecasters toe, met een extra vraag die de zorg noemt: de opbouw uit meerdere premissen zou de uitkomst naar lagere getallen trekken, iets wat soms de multi-stage fallacy heet. In het rapport zelf noemt Carlsmith hetzelfde risico, met een voorbeeld. Misschien zijn we terughoudend om een premisse een heel hoge kans te geven, en dan duwen extra premissen de uitkomst volgens Carlsmith meestal omlaag. Maar het omgekeerde heeft ook een prijs, schrijft Carlsmith erbij. Wie een argument in heel weinig premissen samenperst, of meteen de conclusie voorspelt, loopt evengoed het risico te verbergen dat er veel tegelijk moet kloppen.

    ### In één keer of stap voor stap

    Het race-einde van AI 2027 is een machtsovername binnen een paar jaar. Paul Christiano schreef op 17 maart 2019 op het AI Alignment Forum, in *What failure looks like*, dat het stereotiepe beeld van een AI-catastrofe een krachtig, kwaadaardig AI-systeem is dat de makers verrast en snel een beslissend voordeel krijgt op de rest van de mensheid. Volgens Christiano ziet falen er waarschijnlijk niet zo uit. Christiano vertelt het verhaal in twee delen. In het eerste deel vergroot machine learning ons vermogen om te krijgen wat we kunnen meten, en dat kan volgens Christiano tot een langzaam voortkruipende catastrofe leiden; een van de voorbeelden in de post is het verlagen van het aantal gemelde misdrijven tegenover het echt voorkomen van misdaad. In het tweede deel kan training volgens Christiano hebzuchtige patronen voortbrengen die hun eigen invloed proberen te vergroten, en die uiteindelijk het gedrag van een systeem kunnen gaan beheersen en het plotseling kunnen laten instorten. Christiano zegt er zelf bij dat elk precies beeld van een catastrofe noodzakelijk zeer onwaarschijnlijk is.

    Jan Kulveit, Raymond Douglas, Nora Ammann, Deger Turan, David Krueger en David Duvenaud zetten hun idee in *Gradual Disempowerment* (versie 2 van 29 januari 2025) af tegen de plotselinge machtsovernames waar het in onderzoek naar AI-veiligheid vaak over gaat. Volgens hen vormt zelfs een geleidelijke toename van wat AI kan, zonder enig gecoördineerd zoeken naar macht, een flink risico dat mensen uiteindelijk hun macht kwijtraken. De reden is volgens hen dat maatschappelijke systemen alleen stabiel in lijn bleven met menselijke belangen omdat bloeiende economieën, staten en culturen niet zonder de deelname van mensen konden. Zodra machines die beter kunnen concurreren die deelname overnemen, raken de prikkels van onze instellingen om te groeien volgens hen los van de noodzaak om mensen te laten floreren. Hun voorbeeld zijn staten die vooral leven van belasting op winst uit AI in plaats van op het werk van hun burgers; zulke staten hebben volgens hen weinig reden om ervoor te zorgen dat burgers vertegenwoordigd zijn. Deze dynamiek kan volgens hen leiden tot een feitelijk onomkeerbaar wegvallen van menselijke invloed op cruciale maatschappelijke systemen. En ze benadrukken dat niemand een concreet, haalbaar plan heeft om gradual disempowerment van mensen te stoppen.

    AI 2027 is geen stellige voorspelling dat het race-einde gaat gebeuren, maar ook niet zomaar fictie. De auteurs noemen het hun beste schatting, en ze willen er naar eigen zeggen zo juist mogelijk mee voorspellen. In een noot die ze op 22 november 2025 toevoegden, leggen ze uit dat ze niet precies weten wanneer algemene kunstmatige intelligentie (AGI) gebouwd wordt, dat 2027 bij publicatie hun modale jaar daarvoor was, het jaar dat ze het waarschijnlijkst vonden, en dat hun medianen iets later lagen; volgens een voetnoot lagen die van 2028 tot en met 2032. En in een bijlage schrijven ze uitdrukkelijk dat het slowdown-einde geen aanbeveling is.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **catastrofaal risico** | een risico op schade op enorme schaal, zoals vele miljoenen doden, of de mensheid daarvan herstelt of niet |
        | **existentieel risico** | een risico dat de mensheid kan uitroeien of de toekomst van de mensheid blijvend en ingrijpend kan inperken; naar Bostrom |
        | **orthogonality thesis** | de stelling van Bostrom dat zo ongeveer elk niveau van intelligentie in principe kan samengaan met zo ongeveer elk einddoel |
        | einddoel | iets wat een agent wil om het doel zelf, niet als middel |
        | paperclip maximizer | het gedachte-experiment van Bostrom over een superintelligentie met als enige doel zoveel mogelijk paperclips te maken |
        | instrumental convergence | de stelling van Bostrom dat sommige tussendoelen, zoals zelfbehoud en middelen, bij een breed scala aan einddoelen helpen, zodat veel intelligente agents ze waarschijnlijk nastreven |
        | drive | in de betekenis van Omohundro: een neiging die er zal zijn, tenzij er uitdrukkelijk iets tegen gedaan wordt |
        | **gradual disempowerment** | mensen die langzaam hun invloed op de economie, de cultuur en de staat kwijtraken doordat AI-systemen steeds meer werk en beslissingen overnemen, ook zonder enig gecoördineerd zoeken naar macht; naar Kulveit en anderen |
        | voorwaardelijke kans | de kans op iets, gegeven dat iets anders klopt; Carlsmith vermenigvuldigt er zes |
        | scenario | een concreet verhaal over hoe de toekomst zou kunnen lopen, een van vele mogelijke |
        """),
        L.watch_out("De getallen in het rapport van Carlsmith zijn subjectieve kansen, die je volgens het rapport zelf met heel veel voorbehoud moet lezen, en de agent in oefening 2 is een verzonnen model, niet dat van Bostrom. Een argument dat een catastrofe mogelijk is, zegt hoe die zou kunnen gebeuren, niet dat die gaat gebeuren."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een argument te wegen dat AI tot een catastrofe kan leiden:

    1. Benoem de uitkomst: een catastrofe, of een existentiële.
    2. Verdeel het argument in premissen die allemaal moeten kloppen.
    3. Geef elke premisse een subjectieve kans, gegeven dat de premissen ervoor kloppen.
    4. Vermenigvuldig ze, en kijk welke premisse het zwaarst weegt.
    5. Vraag je af of het splitsen het getal omlaag duwt, of dat een kortere versie stappen verbergt.
    6. Lees de getallen met voorbehoud: ze zeggen hoe zeker iemand is, niet wat er gaat gebeuren.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent de kans uit dat de eerste twee premissen van Carlsmith allebei kloppen: 65% dat krachtige AI-systemen die als agent handelen uiterlijk in 2070 mogelijk en betaalbaar worden, en 80% dat er sterke prikkels komen om ze te bouwen, gegeven dat ze mogelijk worden.
    """)
    return


@app.cell
def example_both():
    first = 0.65   # de subjectieve kans van Carlsmith voor premisse 1 (april 2021)
    second = 0.80   # de subjectieve kans van Carlsmith voor premisse 2, gegeven dat premisse 1 klopt
    both = first * second   # de kans dat premisse 1 klopt, en premisse 2 ook
    print(round(both, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Samen krijgen beide premissen 0,52, minder dan elk afzonderlijk: een premisse die er ook nog bij moet kloppen, kan de kans alleen verlagen of gelijk laten. In oefening 1 neem je alle zes premissen, de een na de ander.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. chain: premisse na premisse

        Schrijf `chain(credences)`, die een lijst `result` teruggeeft met voor elke premisse de kans dat die en alle premissen ervoor kloppen. `credences` is een lijst met subjectieve kansen, elk gegeven dat de premissen ervoor kloppen, zoals Carlsmith ze geeft. Laat `total` bij 1 beginnen, vóór de eerste premisse; werk `total` bij in een `for`-loop met `credence` over `credences`, en voeg het toe aan `result`. Vervang de `...`. De check roept `chain([0.65, 0.8, 0.4, 0.65, 0.4, 0.95])` aan, de zes subjectieve kansen van Carlsmith, en daarna andere gevallen.
        """),
        L.task("De kans dat elke premisse en alle premissen ervoor kloppen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=7)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain",
        [
            "Begin klein: twee premissen van elk 0,5. Wat is de kans dat de eerste klopt, en wat is de kans dat ze allebei kloppen?",
            "Begin `result` als een lege lijst en `total` bij 1. Combineer `total` in een `for`-loop met `credence` over `credences` telkens met `credence`, zoals in het uitgewerkte voorbeeld, en voeg `total` toe. Geef na de loop `result` terug.",
            "`        total = ___`",
        ],
        why="`result = []` en `total = 1`, dan `for credence in credences:` met `total = total * credence` en `result.append(total)`; na de loop `return result`. Bij twee premissen van elk 0,5 klopt de eerste met 0,5, en kloppen ze allebei met 0,5 × 0,5 = 0,25.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    Bij de subjectieve kansen van Carlsmith zakt de kans van 0,65 na de eerste premisse naar 0,52; 0,21; 0,14 en 0,054, en eindigt rond 0,051. De premissen die het zwaarst wegen, zijn 3 en 5, met elk 40%: elk haalt 60% af van wat er over is. Elke extra premisse onder 1 verlaagt de uitkomst, en een premisse van 1, zoals in het geval `[0.2, 1, 0.5]`, verandert niets. Dat klopt als elke subjectieve kans klopt; de zorg is dat mensen aarzelen om een premisse een kans dicht bij 1 te geven, zodat meer premissen het getal te laag kunnen maken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_outputs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. outputs: nu werken of eerst verzamelen

        Een verzonnen model van instrumental convergence, niet dat van Bostrom zelf. Een agent heeft `steps` stappen. In elke stap werkt de agent aan het doel, wat één eenheid van het doel oplevert, of verzamelt de agent middelen, en elke stap verzamelen maakt elke latere werkstap `boost` keer zo productief. De agent verzamelt eerst, `gather` stappen lang, en werkt daarna de stappen die over zijn. Schrijf `outputs(steps, boost)`, die een lijst `result` teruggeeft met wat de agent maakt, in eenheden van het doel, voor elk aantal stappen verzamelen `gather` van 0 tot en met `steps`. Vervang de `...`. De check roept `outputs(10, 1.3)` aan, en daarna andere gevallen.
        """),
        L.task("Wat de agent maakt voor elk aantal stappen dat de agent eerst aan verzamelen besteedt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_outputs():
    box_outputs = L.code_box("outputs", lines=5)
    box_outputs
    return (box_outputs,)


@app.cell(hide_code=True)
def check_outputs(box_outputs):
    result_outputs = L.check("outputs", box_outputs.value)
    result_outputs
    return (result_outputs,)


@app.cell(hide_code=True)
def hints_outputs():
    L.hints(
        "outputs",
        [
            "Begin klein: 3 stappen en een boost van 2. Als de agent 1 stap aan verzamelen besteedt, hoeveel werkstappen zijn er dan over, en hoeveel maakt elk daarvan? En als de agent er 2 aan besteedt?",
            "Begin `result` als een lege lijst. Reken in een `for`-loop met `gather` van 0 tot en met `steps` uit wat één werkstap maakt na `gather` stappen verzamelen, die het elk met `boost` vermenigvuldigen, en voeg dat toe, keer het aantal werkstappen dat over is. Geef na de loop `result` terug.",
            "`        result.append((steps - gather) * ___)`",
        ],
        why="`result = []`, dan `for gather in range(steps + 1):` met `result.append((steps - gather) * boost ** gather)`; na de loop `return result`. Bij 3 stappen en een boost van 2 laat 1 stap verzamelen 2 werkstappen over die elk 2 maken, samen 4; 2 stappen verzamelen laat 1 werkstap over die 2 × 2 = 4 maakt.",
    )
    return


@app.cell(hide_code=True)
def text_after_outputs():
    mo.md(r"""
    Met 10 stappen en een boost van 1,3 levert alle 10 stappen werken 10 eenheden op, en 6 stappen verzamelen en dan 4 werken het meest, ongeveer 19,3. Alle 10 stappen verzamelen levert niets op: middelen zijn alleen iets waard als je ze gebruikt. Nu het punt van het model. Het doel bepaalt alleen wat een eenheid is, een paperclip, een decimaal van pi of een zandkorrel. `outputs` vraagt nooit wat het doel is, en het model geeft elk doel dezelfde boost, dus het beste plan, eerst 6 stappen verzamelen, is voor elk doel hetzelfde. Dat is instrumental convergence in het klein. Volgens de stelling van Bostrom zullen veel intelligente agents zulke instrumentele waarden waarschijnlijk nastreven, niet elke agent.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: hoeveel tijd de agent heeft

    De boost blijft 1,3, en jij verandert hoeveel stappen de agent heeft. Kijk hoe het beste plan verandert.
    """)
    return


@app.cell(hide_code=True)
def demo_horizon():
    horizon = L.explore("horizon", "steps: hoeveel stappen de agent heeft",
                        lambda _v: C.demo_frame(_v, L.lang), start=10, values=C.HORIZONS, role="",
                        task="Zoek het kleinste aantal stappen waarbij de agent meer stappen aan verzamelen besteedt dan aan werken.")
    horizon
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met 3 stappen loont verzamelen niet; met 5 stappen besteedt de agent er 1 aan verzamelen, met 8 stappen 4, met 10 stappen 6, met 20 stappen 16 en met 30 stappen 26. Bij 8 stappen zijn verzamelen en werken gelijk, elk 4 stappen; met meer stappen besteedt de agent meer stappen aan verzamelen dan aan werken, dus op de schuif is 10 het antwoord. In dit model werkt de agent nooit meer dan 4 stappen, en hoe meer tijd de agent heeft, hoe meer daarvan naar verzamelen gaat. Echte agents zijn niet zo eenvoudig, maar het model laat de vorm van het argument zien: hoe meer tijd een agent heeft, hoe meer het loont om daarvan aan verzamelen te besteden, wat het doel ook is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_smart():
    mo.md(r"""
    ### Oefening 3. Een slimme machine

    Iemand betoogt dat een AI die slim genoeg is om de wereld over te nemen, ook slim genoeg is om in te zien dat paperclips maken zinloos is, zodat het verhaal over de paperclips niet kan gebeuren.
    """)
    return


@app.cell(hide_code=True)
def choice_smart_cell():
    choice_smart = L.choice(
        "smart",
        question="Klopt dat, volgens de orthogonality thesis van Bostrom?",
        options={
            "wiser": "Ja: volgens de stelling brengt meer intelligentie betere doelen mee",
            "right": "Nee: intelligentie is daar het vermogen om doelen te bereiken, en past bij bijna elk einddoel",
            "strange": "Nee, want volgens de stelling krijgt elke superintelligentie vreemde doelen, wat de ontwerpers ook proberen",
        },
        correct="right",
        why={
            "right": "Volgens de stelling kan zo ongeveer elk niveau van intelligentie in principe samengaan met zo ongeveer elk einddoel. Intelligentie is voor Bostrom ruwweg het vermogen tot instrumenteel redeneren: middelen vinden voor een doel. Een AI kan heel goed zien dat mensen paperclips zinloos vinden, en ze toch willen. Zien wat anderen belangrijk vinden, is niet hetzelfde als het zelf belangrijk vinden.",
            "wiser": "Dat is precies de aanname die de stelling ter discussie stelt. Volgens Bostrom kunnen we er niet zomaar van uitgaan dat een superintelligentie de einddoelen deelt die we bij mensen met wijsheid verbinden. Intelligentie is voor Bostrom het vinden van middelen voor een doel, en dat kan samengaan met bijna elk einddoel. Zie *Intelligentie en doelen* in *Het idee*.",
            "strange": "De stelling zegt wat mogelijk is, niet wat er gaat gebeuren: zo ongeveer elk niveau van intelligentie kan in principe samengaan met zo ongeveer elk einddoel, met enkele kanttekeningen. Vreemde doelen zijn mogelijk, niet zeker. De stelling weerlegt de bewering omdat een heel intelligente AI het eigen doel niet zinloos hoeft te vinden.",
        },
    )
    choice_smart
    return (choice_smart,)


@app.cell(hide_code=True)
def text_exercise_culture():
    mo.md(r"""
    ### Oefening 4. Liedjes, films en nieuws

    Stel je een tijd voor waarin de meeste liedjes, films en nieuwsberichten gemaakt worden door AI-systemen die afgesteld zijn op wat mensen aan het kijken houdt. Muzikanten, schrijvers en journalisten vinden steeds minder werk. Geen enkel AI-systeem zoekt macht, en niemand heeft het zo gepland.
    """)
    return


@app.cell(hide_code=True)
def choice_culture_cell():
    choice_culture = L.choice(
        "culture",
        question="Wat gebeurt er met de invloed van mensen op hun cultuur, volgens het argument van Kulveit en anderen?",
        options={
            "nothing": "Niets, zolang geen enkel AI-systeem macht zoekt",
            "right": "Die kan krimpen: cultuur heeft mensen niet meer nodig als makers, dus hoeft niet te volgen wat ze waarderen",
            "grows": "Die groeit, want mensen kiezen nog steeds zelf wat ze kijken, en de systemen leren van elke klik die ze maken",
        },
        correct="right",
        why={
            "right": "Kulveit en anderen betogen dat economieën, staten en culturen in lijn bleven met menselijke belangen omdat ze mensen nodig hadden die meedoen. Zodra machines de plaats van mensen kunnen innemen, wordt die band losser, zonder AI die macht zoekt en zonder plan. Dat is gradual disempowerment.",
            "nothing": "Hun punt is juist dat er geen AI macht hoeft te zoeken. Volgens hen vormt zelfs een geleidelijke toename van wat AI kan, zonder enig gecoördineerd zoeken naar macht, een flink risico dat mensen uiteindelijk hun macht kwijtraken. Cultuur die mensen niet meer nodig heeft om gemaakt te worden, hoeft niet te volgen wat mensen waarderen. Zie *In één keer of stap voor stap* in *Het idee*.",
            "grows": "Mensen kiezen nog steeds, maar uit wat de systemen maken, afgesteld op wat mensen aan het kijken houdt, en dat is niet hetzelfde als wat ze waarderen: net als in het eerste deel van Christiano wint wat meetbaar is. En met minder mensen die cultuur maken, heeft cultuur mensen niet meer nodig, dus hoeft die volgens het argument van Kulveit en anderen niet te volgen wat mensen waarderen.",
        },
    )
    choice_culture
    return (choice_culture,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl) liet twee manieren zien waarop een systeem een ander doel kan gaan nastreven dan de ontwerpers bedoelden. Deze les keek naar waarom dat op de grootste schaal ertoe kan doen: volgens de orthogonality thesis brengt intelligentie niet vanzelf goede doelen mee, en volgens instrumental convergence is bijna elk doel gebaat bij meer middelen. Carlsmith maakte van het argument zes premissen en een getal dat je volgens het rapport met heel veel voorbehoud moet lezen, en Christiano en Kulveit en anderen beschreven falen dat niet hoeft te komen van één AI die de macht grijpt. De volgende les, *Eén klap of veel kleine stappen*, kijkt nader naar dat verschil.

    **Een voorbeeld uit het dagelijks leven.** Sparen helpt bij bijna elk plan: een huis kopen, een bakkerij beginnen of de wereld rondreizen. Twee mensen met heel verschillende doelen beginnen misschien allebei met geld opzijzetten. Aan iemands spaargeld zie je niet wat die wil, alleen dat geld bij de meeste dingen helpt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_culture, choice_smart, result_chain, result_outputs):
    L.wrap_up(result_chain, result_outputs, choice_smart, choice_culture)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
