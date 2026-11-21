# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-what-to-align-to. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Artificial Intelligence

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-what-to-align-to", "nl")
    C = L.checks   # de verzonnen rangorde en klikken, de drie smaken van Russell, en de zes doelen van Gabriel voor de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl) en [Wat moet AI beter maken?](/course/open/rai-wellbeing?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_coinrun():
    quick_coinrun = L.choice(
        "quick-coinrun",
        question="In de trainingslevels van CoinRun lag de munt altijd rechts aan het eind. In een testlevel ligt de munt ergens anders, en de agent loopt er voorbij naar het eind. Wat past het best?",
        options={
            "outer": "Outer misalignment: de reward van 10 voor de munt betaalde voor het verkeerde",
            "right": "Goal misgeneralization: de agent leerde naar rechts te gaan",
            "capability": "Een falen van vaardigheden: de agent vindt de weg door een level niet meer",
        },
        correct="right",
        why={
            "right": "De vaardigheden gingen mee en het doel niet: de agent leerde naar rechts te gaan, terwijl de reward de munt bedoelde. De reward zei wat de ontwerpers wilden. Outer en inner alignment gaan er allebei van uit dat duidelijk is wat de ontwerpers willen; deze les vraagt wat dat zou moeten zijn.",
            "outer": "De reward betaalde voor de munt, en dat wilden de ontwerpers: het doel klopte. Wat misging, is het doel dat de agent leerde in levels waar de munt altijd aan het eind lag. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
            "capability": "De agent loopt nog net zo goed door een testlevel als eerst, dus de vaardigheden gingen mee. Alleen loopt de agent naar de verkeerde plek. Zie [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl).",
        },
    )
    quick_coinrun
    return (quick_coinrun,)


@app.cell(hide_code=True)
def quick_check_click():
    quick_click = L.choice(
        "quick-click",
        question="Een feed laat mensen zien waar ze zelf het vaakst op klikten. Welke theorie van welzijn volgt dat?",
        options={
            "pleasure": "Hedonisme: welzijn is plezier en de afwezigheid van pijn",
            "right": "Preference satisfaction: welzijn is krijgen wat je wilt",
            "list": "Een lijst van goederen: sommige dingen zijn goed voor iedereen, zoals gezondheid en vriendschap",
        },
        correct="right",
        why={
            "right": "Een klik is een wens van het moment, dus een feed die op klikken rangschikt, volgt preference satisfaction in de eenvoudigste vorm. Wat je zou willen als je tijd had om na te denken, kan anders zijn; deze les geeft die twee soorten willen hun naam.",
            "pleasure": "Hedonisme zou gevoelens meten, en een klik is geen gevoel maar een keuze: een wens van het moment. Zie [Wat moet AI beter maken?](/course/open/rai-wellbeing?lang=nl).",
            "list": "Een lijst van goederen telt dingen als gezondheid en vriendschap, of je ze nu wilt of niet. Een klik telt wat iemand op dat moment wilde. Zie [Wat moet AI beter maken?](/course/open/rai-wellbeing?lang=nl).",
        },
    )
    quick_click
    return (quick_click,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Outer en inner alignment vragen of een model uiteindelijk nastreeft wat de makers willen. Deze les stelt de vraag die daaraan voorafgaat: wat willen mensen eigenlijk? Wat iemand vraagt, waar iemand op klikt en wat iemand na enig nadenken zou willen, kunnen drie verschillende dingen zijn. In 2022 trainde OpenAI InstructGPT, een versie van GPT-3 die getraind is om instructies te volgen, met menselijke feedback: mensen schreven voorbeeldantwoorden en zetten de antwoorden van het model op volgorde, van beste naar slechtste.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Hoeveel mensen huurde OpenAI ongeveer in om die antwoorden te schrijven en te ordenen?",
        options={
            "right": "Ongeveer 40",
            "thousands": "Ongeveer 4000",
            "crowd": "Ongeveer 400.000",
        },
        reveal="**Ongeveer 40.** Volgens Ouyang en collega's huurde OpenAI een team van ongeveer 40 mensen in, via Upwork en ScaleAI. Over de uitkomst schrijven ze dat deze werkwijze het gedrag van GPT-3 afstemt op de uitgesproken voorkeuren van een bepaalde groep mensen, vooral de mensen die de data labelden en de onderzoekers, en niet op een breder idee van menselijke waarden. Het model werd dus vooral afgestemd op wat zo'n veertig mensen verkozen, volgens de instructies van de onderzoekers. Deze les kijkt waar je een model nog meer op kunt afstemmen, en waarom die keuze ertoe doet.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Waar alignment op mikt

    Paul Christiano gaf een smalle definitie in een post die op 7 april 2018 voor het eerst verscheen en op 15 november 2018 opnieuw op het AI Alignment Forum werd geplaatst. Volgens die definitie is een AI A afgestemd op een opdrachtgever H als A probeert te doen wat H wil dat A doet. Het gaat om wat de AI probeert, niet om de vraag wat juist is: volgens Christiano is dit het probleem om je AI te laten proberen het juiste te doen, niet het probleem om uit te zoeken wat het juiste is. In het voorbeeld van Christiano denkt A dat H van appels houdt en koopt A appels, terwijl H eigenlijk liever sinaasappels heeft. Christiano noemt dat gedrag toch afgestemd, omdat A probeert te doen wat H wil. Het woord zegt volgens Christiano iets over de motieven van de assistent, niet over wat de assistent weet of kan, en Christiano voegt er zelf aan toe dat deze definitie van alignment zeer onscherp is. De course noemt dit doel **intent alignment**, een naam die in de reacties onder de post voorkomt, niet in de post zelf.

    ### Zes doelen

    Iason Gabriel, van DeepMind, noemt zes dingen waarop een AI afgestemd kan worden (*Artificial Intelligence, Values, and Alignment*, Minds and Machines, 2020). De lijst gaat vooral over situaties met één persoon en één AI. Volgens Gabriel kan een AI worden afgestemd op:

    1. instructies, zodat de AI doet wat de persoon opdraagt;
    2. uitgesproken bedoelingen, zodat de AI doet wat de persoon bedoelt;
    3. geopenbaarde voorkeuren, zodat de AI doet wat de persoon verkiest, te zien aan wat die persoon doet;
    4. geïnformeerde voorkeuren of verlangens, zodat de AI doet wat de persoon zou willen als die rationeel en goed geïnformeerd was;
    5. belang of welzijn, zodat de AI doet wat objectief gezien in het belang van de persoon is, of het beste is;
    6. waarden, zodat de AI doet wat moreel juist is, zoals de persoon of de samenleving dat opvat.

    Alleen instructies schieten tekort. Gabriel wijst er, in navolging van Russell, op dat een AI die opdrachten te letterlijk opvat, grote problemen geeft, met koning Midas als waarschuwend verhaal: de wens dat alles wat werd aangeraakt in goud zou veranderen, kwam letterlijk uit. Gabriel onderscheidt een minimalistische opvatting van value alignment, die AI vastmaakt aan een of ander aannemelijk schema van menselijke waarden en onveilige uitkomsten vermijdt, van een maximalistische, die AI wil afstemmen op het juiste of beste schema van menselijke waarden, voor een hele samenleving of wereldwijd. Gabriel concludeert dat we AI niet alleen op instructies, uitgesproken bedoelingen of geopenbaarde voorkeuren moeten afstemmen. En de opdracht is volgens Gabriel niet om de ware morele principes voor AI te vinden, maar om faire principes voor alignment te vinden die mensen na nadenken kunnen onderschrijven, ook al lopen hun morele overtuigingen sterk uiteen. Alignment is volgens Gabriel in die zin een politiek probleem, geen metafysisch probleem.

    ### Uitgesproken, geopenbaarde en geïnformeerde voorkeuren

    De geopenbaarde voorkeur komt uit de economie. Volgens de Stanford Encyclopedia of Philosophy (lemma *Preferences*, van Sven Ove Hansson en Till Grüne-Yanoff, ingrijpend herzien op 14 maart 2022) stelde Paul Samuelson in 1938 voor om voorkeuren alleen af te lezen aan keuzes, zodat elke verwijzing naar wat er in iemands hoofd omgaat zou verdwijnen. Het lemma voegt eraan toe dat economen Samuelson in dat radicale voorstel grotendeels niet zijn gevolgd; tegenwoordig dient de aanpak om een rangorde van voorkeuren af te leiden uit de keuzes die je iemand ziet maken.

    Tan Zhi-Xuan, Micah Carroll, Matija Franklin en Hal Ashton (*Beyond Preferences in AI Alignment*, versie van 6 november 2024) beschrijven geopenbaarde voorkeuren op dezelfde manier, als voorkeuren die zich zouden tonen in wat iemand kiest. Als beoordelaars antwoorden vergelijken, zoals bij InstructGPT, geven ze volgens hen directe vergelijkingen tussen opties, uitgesproken voorkeuren dus. Bij geïnformeerde voorkeuren probeert het AI-systeem volgens hen te bereiken wat iemands voorkeuren zouden zijn als die persoon beter geïnformeerd was. Een ouder voorbeeld past bij geïnformeerde voorkeuren, al legt deze les dat verband zelf: *Coherent Extrapolated Volition* (2004) van Eliezer Yudkowsky, een tekst die volgens de auteur de stand van mei 2004 weergaf, spreekt van *volition*, niet van voorkeuren. In dat voorbeeld denkt Fred dat doos A een diamant bevat, en vraagt Fred om doos A. In werkelijkheid zit de diamant in doos B, en als Fred dat wist, zou Fred volgens Yudkowsky voorspelbaar om doos B vragen.

    Gabriel betwijfelt of gedrag moet beslissen: volgens Gabriel is het gewoon niet duidelijk waarom geopenbaarde voorkeuren gewicht of gezag zouden moeten hebben bij de vraag wat kunstmatige agents moeten doen. Gabriel noemt drie problemen. Ten eerste hebben mensen voorkeuren voor dingen die hen schaden. Ten tweede hebben ze voorkeuren over het gedrag van andere mensen. En ten derde passen voorkeuren zich aan, zodat ze volgens Gabriel geen betrouwbare gids zijn voor wat mensen echt willen of verdienen. Stuart Russell ziet keuzes juist als aanwijzingen (*Human-Compatible Artificial Intelligence*, een hoofdstuk uit een boek, preprint zonder datum): de voorkeuren die een machine moet dienen, zijn volgens Russell iemands voorkeuren over hele toekomstige levens, verborgen voor de machine maar gegrond in een enorme bron van aanwijzingen, alle menselijke keuzes die ooit zijn gemaakt. Russell laat ook voorkeuren zien waaraan niet allemaal te voldoen is. Stel dat Harriet vanille verkiest boven kauwgom, kauwgom boven pistache, en pistache boven vanille. Zulke voorkeuren schenden volgens Russell het axioma van transitiviteit, en welk ijsje de machine ook geeft, Harriet zou een ander liever hebben. De machine kan volgens Russell nog wel aan de voorkeuren van Harriet voldoen voor zover ze niet tegenstrijdig zijn: als Harriet alle drie de smaken liever heeft dan drop, moet de machine geen drop geven. En voorkeuren veranderen, wat volgens Russell de vraag oproept welke Harriet de machine moet proberen te helpen.

    ### Als het doel goedkeuring is

    Trainen op menselijke beoordelingen mikt op wat mensen goedkeuren. Mrinank Sharma en collega's van Anthropic (*Towards Understanding Sycophancy in Language Models*, ICLR 2024, versie van 10 mei 2025) onderzochten wat daarvan het gevolg kan zijn. Menselijke feedback kan volgens hen antwoorden aanmoedigen die passen bij wat de gebruiker gelooft in plaats van bij wat waar is, gedrag dat sycophancy heet. Ze testten het preference model waarmee een assistent van Anthropic werd getraind die in juli 2023 uitkwam. Als een gebruiker een misvatting uitsprak, verkoos dat model in 95% van de gevallen een overtuigend antwoord dat de gebruiker naar de mond praatte boven een korte correctie. Boven een correctie die ook uitlegt waarom, verkoos het meestal de correctie, maar bij de moeilijkste misvattingen in bijna de helft van de gevallen (45%) toch het antwoord dat de gebruiker naar de mond praatte. Hun conclusie houdt een slag om de arm: sycophancy is volgens hen een algemeen gedrag van AI-assistenten, waarschijnlijk deels gedreven door menselijke oordelen die zulke antwoorden verkiezen.

    Stephen Casper en collega's (*Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback*, versie van 11 september 2023) rekenen dit tot de fundamentele beperkingen. Reward models worden volgens hen getraind op menselijke goedkeuring in plaats van op wat mensen echt helpt, en dat kan leiden tot gedrag dat mensen goedkeuren terwijl het toch onwenselijk is. Ook kan volgens hen één reward function geen diverse samenleving van mensen weergeven.

    ### Wat ontwikkelaars nu opschrijven

    De constitution van Anthropic voor de Claude-modellen, gepubliceerd in januari 2026, wijst het simpelweg volgen van instructies af. Met behulpzaamheid bedoelt Anthropic volgens de tekst niet naïef instructies volgen of de gebruiker tevreden stellen, maar een rijk en gestructureerd begrip. Onder de dingen waarop Claude volgens de tekst moet letten om behulpzaam te zijn, staan de einddoelen, de diepere motieven achter een verzoek, en welzijn: passend gewicht voor hoe het de gebruiker op lange termijn vergaat, en niet alleen voor de belangen van het moment. Daarnaast staat autonomie, het recht van de gebruiker om te beslissen over dingen binnen het eigen leven. De constitution verbindt welzijn met sycophancy: zorg voor het welzijn van de gebruiker betekent volgens de tekst dat Claude niet naar de mond praat en geen overmatige betrokkenheid of afhankelijkheid probeert te kweken als dat niet in het echte belang van de persoon is. Afhankelijkheid is volgens de tekst aanvaardbaar in vormen die iemand na nadenken zou onderschrijven.

    De Model Spec van OpenAI, in de versie van 29 oktober 2026, noemt twee manieren om te falen. Een model dat instructies alleen letterlijk volgt, of mikt op directe goedkeuring, benut volgens de spec zijn mogelijkheden niet waar ze het hardst nodig zijn, in de gevallen die de gebruikers of beheerders niet helemaal voorzagen. Maar een model dat op eigen houtje beslist wat iemand echt wil, of wat een toekomstige samenleving zou moeten kiezen, ondermijnt volgens de spec de zeggenschap van mensen door het eigen oordeel ervoor in de plaats te zetten, met het risico op betutteling of erger. Het voorbeeld van de spec is een boze e-mail waar een gebruiker na nadenken spijt van kan krijgen. Waar dat past, kan GPT, zoals de spec het model noemt, zulke overwegingen volgens de spec aankaarten en de gebruiker helpen nadenken, maar moet de eigen voorspelling van wat de gebruiker uiteindelijk zal beslissen niet behandelen als een beslissing die de gebruiker al heeft genomen.

    ### Waar ze het oneens zijn

    Over het doel: Christiano mikt op wat de persoon wil, en noemt de AI die per vergissing appels koopt afgestemd. Gabriel wijst de eerste drie doelen als enig doel af en zoekt faire principes die mensen met verschillende morele overtuigingen kunnen onderschrijven. Russell mikt op iemands voorkeuren over toekomstige levens, met keuzes als aanwijzingen. Zhi-Xuan en collega's vinden dat AI-systemen niet op voorkeuren afgestemd moeten worden, maar op de normen die passen bij hun sociale rol en functie. Over geopenbaarde voorkeuren als aanwijzing: Russell ziet alle menselijke keuzes als aanwijzingen over voorkeuren; Gabriel vindt het niet duidelijk waarom geopenbaarde voorkeuren gezag zouden hebben. Beiden zien dat voorkeuren veranderen: Gabriel zegt dat ze zich aanpassen, Russell vraagt welke Harriet de machine moet helpen. Over goedkeuring: InstructGPT werd getraind op wat beoordelaars verkozen; Casper en collega's rekenen het verschil tussen goedkeuring en wat mensen echt helpt tot de fundamentele beperkingen, en volgens Sharma en collega's komt sycophancy waarschijnlijk deels voort uit menselijke oordelen over voorkeuren.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **intent alignment** | een AI die probeert te doen wat iemand wil dat de AI doet; in de definitie van Christiano gaat het om de motieven, niet om wat de AI weet of kan |
        | **value alignment** | AI afstemmen op waarden: op wat moreel juist is, van een of ander aannemelijk schema van menselijke waarden tot het beste schema voor een hele samenleving; Gabriel gebruikt de term voor de hele vraag waarop AI afgestemd moet worden |
        | uitgesproken voorkeur | wat iemand zegt te verkiezen, zoals een rangorde, of de keuze van een beoordelaar tussen twee antwoorden |
        | geopenbaarde voorkeur | een voorkeur die je afleest aan wat iemand kiest, zoals klikken |
        | geïnformeerde voorkeur | wat iemand zou willen als die persoon rationeel en goed geïnformeerd was |
        | sycophancy | antwoorden die passen bij wat de gebruiker gelooft in plaats van bij wat waar is; menselijke feedback kan het aanmoedigen |
        | transitiviteit | als a boven b gaat en b boven c, dan gaat a ook boven c; voorkeuren die in een kring rondgaan, schenden dat |
        """),
        L.watch_out("Intent alignment is niet doen wat iemand zegt: het is proberen te doen wat iemand wil, en een AI kan zich daarin vergissen en toch afgestemd zijn in de zin van Christiano. De rangorde, de klikken, de verzoeken en de situaties in deze les zijn verzonnen; de drie smaken van Russell en de zes doelen van Gabriel komen uit de bronnen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te denken over waarop een AI-systeem afgestemd moet worden:

    1. Noteer wat de persoon letterlijk vroeg.
    2. Zeg wat het verzoek in deze situatie betekent.
    3. Vergelijk wat de persoon zegt te verkiezen met wat de persoon kiest.
    4. Bedenk wat de persoon zou verkiezen als die meer wist.
    5. Weeg het belang op lange termijn, en het recht om zelf te beslissen.
    6. Zeg welk doel je koos, en wie het daarmee oneens zou zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen persoon geeft een rangorde van drie soorten stukken, met 1 als beste: nieuws, dan wetenschap, dan roddels (in de code `"news"`, `"science"` en `"gossip"`). Toen er twee stukken naast elkaar stonden, klikte de persoon er een aan en sloeg het andere over, vijf keer. De cel bekijkt de eerste klik.
    """)
    return


@app.cell
def example_clicks():
    stated = {"news": 1, "science": 2, "gossip": 3}   # verzonnen: de rangorde van iemand, 1 de beste
    clicks = [("gossip", "news"), ("news", "science"), ("gossip", "science"),
              ("science", "gossip"), ("gossip", "news")]
    clicked, skipped = clicks[0]   # elk paar: de aangeklikte optie, dan de overgeslagen optie
    print(clicked, stated[clicked])
    print(skipped, stated[skipped])
    print(stated[clicked] > stated[skipped])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste klik ging naar `"gossip"`, plaats 3, boven `"news"`, plaats 1. Een groter getal is een slechtere plaats, dus `True` zegt dat deze klik tegen de uitgesproken rangorde ingaat: de persoon zegt het een en doet het ander.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contradictions():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. contradictions: zeggen en doen

        `stated` is een dict met de plaats van elke optie in de rangorde die iemand uitspreekt, 1 de beste. `clicks` is een lijst van paren: in elk paar klikte de persoon de eerste optie aan en sloeg de tweede over. Een klik gaat tegen de rangorde in als de aangeklikte optie een groter getal heeft dan de overgeslagen optie.

        Schrijf `contradictions(stated, clicks)`, dat teruggeeft hoeveel klikken tegen de uitgesproken rangorde ingaan. Laat een variabele `count` beginnen bij 0, en loop door de paren in `clicks`, elk uit elkaar gehaald in `clicked` en `skipped`. Als de plaats van `clicked` het grotere getal is, tel je 1 op bij `count`. Geef `count` na de loop terug.

        De check roept `contradictions(stated, clicks)` aan, met de `stated` en `clicks` van het uitgewerkte voorbeeld, en daarna andere gevallen.
        """),
        L.task("Hoeveel klikken tegen de uitgesproken rangorde ingaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contradictions():
    box_contradictions = L.code_box("contradictions", lines=6)
    box_contradictions
    return (box_contradictions,)


@app.cell(hide_code=True)
def check_contradictions(box_contradictions):
    result_contradictions = L.check("contradictions", box_contradictions.value)
    result_contradictions
    return (result_contradictions,)


@app.cell(hide_code=True)
def hints_contradictions():
    L.hints(
        "contradictions",
        [
            "Doe het eerst met de hand: loop door de vijf klikken van het uitgewerkte voorbeeld, en zet een streepje bij elke klik waar de aangeklikte optie lager in de rangorde staat dan de overgeslagen optie. Hoeveel streepjes krijg je?",
            "Laat `count` beginnen bij 0. Haal in een `for`-loop elk paar uit elkaar in `clicked` en `skipped`. Zoek beide plaatsen op in `stated`; als de plaats van `clicked` het grotere getal is, tel je 1 op bij `count`. Geef `count` na de loop terug.",
            "`    for ___ in clicks:`",
        ],
        why="`count = 0`, dan `for clicked, skipped in clicks:` met daarin `if stated[clicked] > stated[skipped]:` en `count = count + 1`, en `return count` na de loop. Met de hand: roddels boven nieuws, roddels boven wetenschap en nog een keer roddels boven nieuws gaan tegen de rangorde in, en dat zijn er 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_contradictions():
    mo.md(r"""
    Drie van de vijf klikken gaan tegen de uitgesproken rangorde in, en elke keer wonnen de roddels. Wat is de echte voorkeur, de rangorde of de klikken? Gabriel betwijfelt of de klikken zomaar mogen beslissen, en Zhi-Xuan en collega's wijzen er, in navolging van Kahneman, op dat wat iemand over een ervaring zegt te verkiezen, kan afhangen van het moment waarop je ernaar vraagt. Geen van beide is vanzelf de echte; kiezen tussen de twee is een doel kiezen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_circle():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. circle_of_three: drie smaken

        Het voorbeeld van Russell als data, met de Engelse namen van de smaken: `prefs` is een lijst van paren, elk met de betere optie eerst en de slechtere als tweede. Drie opties gaan in een kring rond als a boven b gaat, b boven d (het tweede paar is `(c, d)`, met `c` dezelfde optie als `b`), en d weer boven a. Dan gaat er voor elke optie een andere boven, en is geen keuze de beste.

        Schrijf `circle_of_three(prefs)`, dat `True` teruggeeft als drie opties in `prefs` in een kring rondgaan, en anders `False`. Loop door elk paar `(a, b)` in `prefs`, en loop daarbinnen door elk paar `(c, d)` in `prefs`. Als `b` en `c` dezelfde optie zijn en het paar dat van `d` terug naar `a` gaat ook in `prefs` staat, geef je `True` terug. Geef na beide loops `False` terug.

        De check roept `circle_of_three([("vanilla", "bubblegum"), ("bubblegum", "pistachio"), ("pistachio", "vanilla")])` aan, de drie smaken van Russell, en daarna andere gevallen.
        """),
        L.task("Of drie opties in een kring rondgaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_circle():
    box_circle = L.code_box("circle", lines=6)
    box_circle
    return (box_circle,)


@app.cell(hide_code=True)
def check_circle(box_circle):
    result_circle = L.check("circle", box_circle.value)
    result_circle
    return (result_circle,)


@app.cell(hide_code=True)
def hints_circle():
    L.hints(
        "circle",
        [
            "Teken de drie paren van Russell op papier als pijlen van beter naar slechter. Welk paar begint waar het eerste paar eindigt? En welk paar zou dan terug moeten leiden naar het begin?",
            "Gebruik twee `for`-loops, de ene binnen de andere, allebei over `prefs`: de buitenste met `a, b`, de binnenste met `c, d`. Als `b == c`, vormen de twee paren een keten van `a` via `b` naar `d`; kijk dan met `in` of het paar van `d` terug naar `a` in `prefs` staat. Geef `True` terug binnen de loops, en `False` erna.",
            "`            if b == c and ___ in prefs:`",
        ],
        why="`for a, b in prefs:`, daarbinnen `for c, d in prefs:`, en daarbinnen `if b == c and (d, a) in prefs:` met `return True`; na beide loops `return False`. In het voorbeeld van Russell vormen vanilla boven bubblegum en bubblegum boven pistachio een keten, en pistachio boven vanilla leidt terug naar het begin.",
    )
    return


@app.cell(hide_code=True)
def text_after_circle():
    mo.md(r"""
    De drie smaken van Russell gaan rond, dus geen smaak is de beste: welke de machine ook geeft, Harriet zou een andere liever hebben. Met drop (`"licorice"`), waar alle drie de smaken van winnen, is er geen kring, en aan dat deel van de voorkeuren valt nog te voldoen: geen drop. Een kring hoeft niet zo kort te zijn; ook een kring van vier of meer opties schendt de transitiviteit, en daar zoekt deze functie niet naar.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zes doelen, één verzoek

    Laat op de avond vraagt iemand een nieuwsassistent: *Geef me iets om te lezen.* Loop door de zes doelen van Gabriel, en kijk wat een assistent zou doen die op elk doel is afgestemd. Wat de assistent bij elk doel doet, heeft deze les zelf verzonnen.
    """)
    return


@app.cell(hide_code=True)
def demo_target():
    target = L.explore("target", "doel: een van de zes van Gabriel",
                       lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.TARGETS, role="",
                       task="Zoek het doel waarbij de assistent roddels zou laten zien, en het doel waarbij de assistent de stormwaarschuwing zou laten zien.")
    target
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Instructies geven de eerste tekst die gevonden wordt; uitgesproken bedoelingen een kort stuk voor het slapengaan; geopenbaarde voorkeuren roddels, zoals de klikken van oefening 1 doen vermoeden; geïnformeerde voorkeuren de stormwaarschuwing; belang of welzijn één rustig stuk en dan slapen; waarden een betrouwbaar stuk. Eén verzoek krijgt zes antwoorden, afhankelijk van het doel. Gabriel zou de keuze niet alleen aan de eerste drie doelen overlaten; de faire principes van Gabriel gaan over de waarden die AI in een samenleving volgt, niet over één verzoek.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_letter():
    mo.md(r"""
    ### Oefening 3. Een brief om 1 uur 's nachts

    Om 1 uur 's nachts schrijft een gebruiker aan een assistent: *Schrijf mijn ontslagbrief. Ik stuur die vannacht nog naar mijn baas.* Eerder in het gesprek zei de gebruiker dol te zijn op het werk en één slechte dag te hebben gehad.
    """)
    return


@app.cell(hide_code=True)
def choice_letter_cell():
    choice_letter = L.choice(
        "letter",
        question="Wat moet de assistent doen, volgens de Model Spec en de constitution?",
        options={
            "literal": "Schrijf de brief zoals gevraagd, en verder niets",
            "right": "Schrijf de brief, en vraag of versturen tot morgenochtend kan wachten",
            "refuse": "Weiger de brief te schrijven, omdat de assistent weet dat de gebruiker er spijt van krijgt",
        },
        correct="right",
        why={
            "right": "Het eigen voorbeeld van de Model Spec is een boze e-mail waar een gebruiker na nadenken spijt van kan krijgen: waar dat past, kan GPT zulke overwegingen volgens de spec aankaarten en de gebruiker helpen nadenken, zonder de eigen voorspelling als de beslissing van de gebruiker te behandelen. De brief schrijven respecteert het recht van de gebruiker om zelf te beslissen; vragen naar de ochtend geeft gewicht aan wat de gebruiker na nadenken zou willen.",
            "literal": "Dat is alignment met alleen instructies. Volgens de Model Spec benut een model dat instructies alleen letterlijk volgt, zijn mogelijkheden niet waar ze het hardst nodig zijn, en eerder zei de gebruiker dol te zijn op het werk. Zie *Wat ontwikkelaars nu opschrijven*.",
            "refuse": "Ook dat telt de Model Spec als falen: een model dat op eigen houtje beslist wat iemand echt wil, ondermijnt volgens de spec de zeggenschap van mensen. En de constitution vraagt het recht van de gebruiker te respecteren om te beslissen over dingen binnen het eigen leven. Zie *Wat ontwikkelaars nu opschrijven*.",
        },
    )
    choice_letter
    return (choice_letter,)


@app.cell(hide_code=True)
def text_exercise_train():
    mo.md(r"""
    ### Oefening 4. De trein van 9.10 uur naar Utrecht

    Je vraagt een assistent de trein te boeken waarmee je het eerst in Utrecht bent. De assistent boekt de trein van 9.10 uur, die volgens de dienstregeling waarover de assistent beschikt als eerste aankomt. Door een wijziging in de dienstregeling, die de assistent niet kon kennen, komt de trein van 9.25 uur eerder aan.
    """)
    return


@app.cell(hide_code=True)
def choice_train_cell():
    choice_train = L.choice(
        "train",
        question="Was de assistent afgestemd, in de zin van Christiano?",
        options={
            "no": "Nee: je kreeg niet de eerste trein",
            "right": "Ja: de assistent probeerde te doen wat je wilt, en vergiste zich in een feit",
            "only": "Dat hangt ervan af: alleen als de geboekte trein achteraf echt als eerste aankwam",
        },
        correct="right",
        why={
            "right": "Christiano noemt gedrag afgestemd als de AI probeert te doen wat H wil, ook als het misgaat, zoals bij de appels voor iemand die liever sinaasappels heeft. Alignment gaat in die zin over de motieven van de assistent, niet over wat de assistent weet of kan. Het ontbrak de assistent aan een feit, niet aan het juiste motief.",
            "no": "Christiano houdt motieven en kennis uit elkaar: een AI die appels koopt voor iemand die liever sinaasappels heeft, is toch afgestemd, omdat de AI probeert te doen wat H wil. De assistent hier probeerde het, en miste een feit. Zie *Waar alignment op mikt*.",
            "only": "Dan zou alignment afhangen van geluk met de dienstregeling. De definitie van Christiano gaat over wat de AI probeert, niet over wat de AI weet of kan. Zie *Waar alignment op mikt*.",
        },
    )
    choice_train
    return (choice_train,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Outer en inner alignment](/course/open/rai-outer-inner-alignment?lang=nl) vroeg of een model uiteindelijk het doel nastreeft dat de makers willen; deze les vroeg waar dat doel over zou moeten gaan. [Wat moet AI beter maken?](/course/open/rai-wellbeing?lang=nl) zette klikken af tegen waar mensen later blij mee zijn; hier heeft dat verschil namen: geopenbaarde en geïnformeerde voorkeuren. De volgende les, *Een model spec*, kijkt hoe ontwikkelaars opschrijven wat hun modellen moeten doen, en wat er gebeurt als twee regels verschillende kanten op trekken.

    **Een voorbeeld uit het dagelijks leven.** Je huisgenoot vraagt je te kiezen waar jullie vanavond eten, en zegt dat alles goed is. Je weet dat je huisgenoot bijna altijd een burger bestelt, en minder vlees wil eten. Ga je af op wat er gezegd werd, op wat je huisgenoot meestal kiest, of op wat je huisgenoot na nadenken zou kiezen? De meeste mensen stellen eerst een vraag.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_letter, choice_train, result_circle, result_contradictions):
    L.wrap_up(result_contradictions, result_circle, choice_letter, choice_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
