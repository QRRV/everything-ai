# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-xrisk-counterpositions. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xrisk-counterpositions", "nl")
    C = L.checks   # verzonnen gevaren met elk een bescherming ertegen, en een verzonnen zachte pauze


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="In het verzonnen model van de vorige les is de veerkracht van een samenleving 80. Elk jaar haalt er een schade van 5 af, en de samenleving wint 2 terug, maar nooit verder dan tot 100. Wat is de veerkracht na twee jaar?",
        options={
            "again": "77: elk nieuw jaar begint weer bij 80",
            "right": "74: het tweede jaar begint bij 77",
            "gain": "84: de samenleving wint elk jaar 2 terug",
        },
        correct="right",
        why={
            "right": "Elk jaar begint waar het jaar ervoor eindigde: 80 − 5 + 2 = 77 na het eerste jaar, en 77 − 5 + 2 = 74 na het tweede. Geen van beide ligt boven 100, dus de top van 100 verandert hier niets.",
            "again": "Dan zouden de schades zich nooit opbouwen. Elk jaar begint waar het jaar ervoor eindigde: 77 na het eerste jaar, daarna 77 − 5 + 2 = 74. Zie [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl).",
            "gain": "De schade gaat er ook af: elk jaar gaat er 5 af en komt er 2 terug, dus de veerkracht daalt met 3 per jaar, van 80 naar 77 en dan naar 74. Zie [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_lockin():
    quick_lockin = L.choice(
        "quick-lockin",
        question="Wanneer is een eigenschap van de wereld vastgelegd, volgens de definitie van Finnveden, Riedel en Shulman?",
        options={
            "bad": "Zodra slechte waarden het van alle goede hebben gewonnen",
            "right": "Zodra veel minder onzeker is hoe die op lange termijn uitpakt",
            "extinct": "Pas als de mensheid is uitgestorven en er niets meer kan veranderen",
        },
        correct="right",
        why={
            "right": "In hun definitie is een eigenschap vastgelegd vanaf een moment waarvóór grote onzekerheid bestond over hoe die eigenschap op lange termijn uitpakt, en waarna die onzekerheid veel kleiner is. Goede waarden kunnen net zo goed vastgelegd worden als slechte.",
            "bad": "Lock-in gaat over hoeveel de verre toekomst nog kan veranderen, niet over of de waarden goed zijn: ook goede waarden kunnen vastgelegd worden. Zie [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl).",
            "extinct": "Uitsterven is volgens hen één voorbeeld van een lock-in, niet het enige: bij wereldwijde value lock-in hebben alle machtige partijen waarden uit een veel kleinere verzameling dan eerst. Zie [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl).",
        },
    )
    quick_lockin
    return (quick_lockin,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De vorige lessen lieten zien waarom sommige onderzoekers een catastrofe door AI vrezen, in één klap of in veel kleine stappen. Niet iedereen kijkt er zo naar. Sommigen zien AI als een normale technologie, hoe ingrijpend ook; sommigen willen de aandacht bij de schade die AI nu al doet; sommigen willen AI sneller bouwen; en sommigen willen AI ook sneller bouwen, maar met bescherming voorop. Achter dat laatste zit een idee: versnel de technologieën die tegen andere beschermen, en rem de gevaarlijke af, zodat elke bescherming er is voordat het gevaar komt waartegen ze beschermt. Nick Bostrom gaf het idee een naam: differential technological development.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In welke tekst gaf Bostrom het idee deze naam?",
        options={
            "book": "In het boek Superintelligence, uit 2014",
            "right": "In een paper over existentiële risico's, uit 2002",
            "preprint": "In een gezamenlijke preprint over het concept, uit 2022",
        },
        reveal="**In een paper over existentiële risico's, uit 2002**, waarvan een eerste versie uit 2001 is. Daarin wil Bostrom de aandacht richten op wat Bostrom differential technological development wil noemen, namelijk de invoering van gevaarlijke technologieën vertragen en die van nuttige versnellen, vooral van technologieën die de gevaren van andere technologieën verkleinen. Latere teksten wijzen ergens anders naar: een hoofdstuk van Anders Sandberg en een post uit 2022 schrijven het idee toe aan het boek *Superintelligence* (2014) van Bostrom. Beide komen van de auteurs van een preprint van Jonas Sandbrink en anderen, en volgens de post is die preprint, voor zover de auteurs weten, het eerste wetenschappelijke stuk dat een principe van differential technology development volledig probeert uit te werken. Deze les kijkt naar dat idee, en naar de andere standpunten ernaast.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### AI als normale technologie

    Arvind Narayanan en Sayash Kapoor publiceerden *AI as Normal Technology* in april 2025 bij het Knight First Amendment Institute. De naam betekent niet dat AI weinig uitmaakt. Wie AI normaal noemt, doet volgens hen de impact niet tekort, want ook ingrijpende technologieën voor algemeen gebruik, zoals elektriciteit en internet, zijn in hun opvatting normaal. De uitspraak dat AI normale technologie is, is volgens hen drie dingen tegelijk: een beschrijving van de AI van nu, een voorspelling over de AI van de afzienbare toekomst, en een voorschrift voor hoe we ermee om moeten gaan. Ze verwachten dat de ingrijpende economische en maatschappelijke gevolgen van AI langzaam komen, over tientallen jaren. Ze zien AI als een gereedschap dat we onder controle kunnen en moeten houden, en ze betogen dat daar geen drastische beleidsingrepen of technische doorbraken voor nodig zijn. Het essay wil volgens de auteurs een wereldbeeld neerzetten, en niet één stelling verdedigen. Ze verwachten ook niet dat de vooruitgang stilvalt. In de wereld die ze beschrijven, is AI veel verder dan nu, en ze zeggen er uitdrukkelijk bij dat ze niet beweren dat de vooruitgang van AI, of van de mensheid, op dat punt stopt.

    Catastrofale misalignment is volgens hen veruit het meest speculatieve risico dat ze bespreken. Speculatief betekent bij hen dat onzeker is of het werkelijke risico nul is, een onzekerheid die nieuwe observaties of verder onderzoek misschien kunnen wegnemen. De belangrijkste verdediging ertegen ligt volgens hen niet bij de modellen zelf, maar verderop, waar AI wordt ingezet. Dezelfde verdediging die ook tegen misbruik nodig is, van het beter beveiligen van kritieke infrastructuur tot betere cybersecurity, beschermt volgens hen ook tegen mogelijke risico's van misalignment.

    Systeemrisico's vinden ze belangrijker, zoals machtsconcentratie, afnemend onderling vertrouwen, vervuiling van de informatievoorziening en massasurveillance. Als AI normale technologie is, worden die risico's volgens hen veel belangrijker dan de catastrofale. Ze pleiten ervoor om het verkleinen van onzekerheid tot een beleidsdoel van de eerste rang te maken, en veerkracht tot de overkoepelende aanpak van catastrofale risico's. En ze zien een afweging. Verdediging tegen superintelligentie vraagt volgens hen dat de mensheid zich verenigt tegen een gezamenlijke vijand, en dus macht concentreert en AI-technologie centraal controleert. Zelf maken ze zich meer zorgen over risico's van mensen die AI voor hun eigen doelen gebruiken.

    ### Eerst de schade van nu

    Anderen zeggen dat het praten over uitsterven de aandacht afleidt van de schade die er al is. Volgens een ongetekend redactioneel stuk in *Nature* van 27 juni 2023 leidt het idee dat AI de mensheid zou kunnen uitroeien, als de truc van een goochelaar de aandacht af van het echte probleem, de schade die AI-systemen en -tools nu al aan de samenleving toebrengen, of in de toekomst kunnen toebrengen. Volgens het stuk zijn veel AI-onderzoekers en ethici die *Nature* sprak gefrustreerd over het gepraat over de ondergang, en is dat gepraat op minstens twee manieren een probleem. De eerste is dat het spookbeeld van AI als almachtige machine landen aanzet tot een wedstrijd om AI te ontwikkelen, zodat ze er profijt van hebben en er controle over krijgen. Dat werkt volgens het stuk in het voordeel van techbedrijven, omdat het investeringen aanmoedigt en argumenten voor regulering van de sector verzwakt. Verderop staat dat bangmakerij over existentiële risico's niets oplevert, en dat een serieus gesprek over echte risico's, en maatregelen om ze in te dammen, dat wel doen.

    Alex Hanna, onderzoeksdirecteur bij het Distributed AI Research Institute, en Emily M. Bender, hoogleraar taalkunde aan de University of Washington, schreven op 12 augustus 2023 in *Scientific American* over onterechte arrestaties, surveillance, smaad en deepfakeporno. Volgens hen zijn dat de echte bedreiging van AI, en niet het vermogen om de mensheid uit te roeien, dat ze denkbeeldig noemen. Dat woord is van hen; het stuk is een opiniestuk, over waar de aandacht heen moet.

    ### Sneller

    Een post van 10 juli 2022, *Notes on e/acc principles and tenets*, geschreven door Beff Jezos en bayes, zoals de byline zegt, is een van de vroege teksten van effective accelerationism, kortweg e/acc; de post verwijst naar een oudere, eerste post van Swarthy. Volgens de post zijn markten met meer variatie en concurrentie beter in het vinden en benutten van nut dan andere methoden, zoals optimale sturing van bovenaf, en remt te veel regulering van technologie de variatie en daarmee de vooruitgang af. Ook wil effective accelerationism volgens de post de wil van het universum volgen. De korte samenvatting in de post zegt onder meer dat je moet ophouden met vechten tegen de thermodynamische wil van het universum, en dat je de versnelling niet kunt tegenhouden.

    *The Techno-Optimist Manifesto* van Marc Andreessen (16 oktober 2023) noemt de eigen opvatting techno-optimisme. Het manifest zegt te geloven in accelerationism, het bewust en weloverwogen vooruitstuwen van technologische ontwikkeling, om de Law of Accelerating Returns waar te maken. Over AI zegt het dat elke vertraging van AI levens zal kosten, en dat doden die voorkomen hadden kunnen worden door AI die niet mocht bestaan, een vorm van moord zijn. Onder de kop *The Enemy* noemt het onder meer existentieel risico, trust and safety en techethiek als namen van wat het een massale demoralisatiecampagne noemt. De lijst van beschermheiligen van het techno-optimisme begint met de accounts @BasedBeffJezos en @bayeslord. Het manifest kwam meer dan een jaar na de post over e/acc.

    ### d/acc

    In *My techno-optimism* (27 november 2023) schrijft Vitalik Buterin in veel situaties sympathie te voelen voor het perspectief van e/acc, maar volgens Buterin kun je dat perspectief ook te ver doortrekken, met zorgen over AI en over het enthousiasme van e/acc voor militaire technologie. Buterin wantrouwt ook een gangbaar antwoord op zulke risico's. Volgens Buterin geven veel te veel plannen om de wereld te redden een kleine groep mensen extreme en ondoorzichtige macht, in de hoop dat die groep er verstandig mee omgaat. Dat brengt Buterin bij een andere filosofie, met de naam d/acc. De d kan volgens Buterin voor veel dingen staan, vooral voor defense, decentralization, democracy en differential, ofwel verdediging, decentralisatie, democratie en onderscheid tussen technologieën. Buterin verbindt het met een ouder idee. Het kernidee, dat sommige technologieën de verdediging helpen en het waard zijn om te bevorderen, terwijl andere de aanval helpen en afgeremd moeten worden, heeft volgens Buterin wortels in de literatuur van effective altruism, onder een andere naam: differential technology development.

    Een jaar later, in *d/acc: one year later* (5 januari 2025), beschrijft Buterin het kernidee van d/acc als decentrale en democratische, differentiële en defensieve versnelling. Je bouwt dan technologie die de verhouding tussen aanval en verdediging naar de verdediging laat doorslaan, op een manier die niet afhangt van meer macht voor centrale autoriteiten. De eerste strategie van Buterin tegen de risico's van AI is aansprakelijkheid: wie schade veroorzaakt waarvoor je naar de rechter kunt, kan worden aangeklaagd. Pas als tweede strategie, onder een kop over een wereldwijde knop voor een zachte pauze op hardware op industriële schaal, schrijft Buterin wat de keuze zou zijn bij de overtuiging dat er iets krachtigers nodig is dan regels voor aansprakelijkheid. Het doel is dan het vermogen om de wereldwijd beschikbare compute op een kritiek moment 1 tot 2 jaar met zo'n 90 tot 99% te verminderen, om de mensheid meer tijd te geven om zich voor te bereiden.

    ### Differential technology development

    De paragraaf van Bostrom over het idee begint bij een grens. Verboden op technologie zijn volgens Bostrom maar beperkt af te dwingen, en dat beperkt welk beleid haalbaar is. Wat we wel kunnen beïnvloeden, is volgens Bostrom het tempo waarin verschillende technologieën zich ontwikkelen, en misschien de volgorde waarin haalbare technologieën worden ontwikkeld en ingevoerd. In de biotechnologie zou Bostrom bijvoorbeeld vaccins, medicijnen, beschermende uitrusting, sensoren en diagnostiek bevorderen, en biologische strijdmiddelen zo lang mogelijk uitstellen. Voor nanotechnologie is volgens Bostrom de gewenste volgorde dat verdedigingssystemen er zijn voordat veel onafhankelijke machten over aanvalsmogelijkheden beschikken. Voor AI pleitte Bostrom niet voor afremmen. Superintelligentie heeft volgens het paper eigen gevaren, maar dat zijn gevaren waar we hoe dan ook ooit mee te maken krijgen, en superintelligentie vroeg krijgen is volgens Bostrom juist wenselijk, omdat het andere risico's zou helpen verkleinen. Superintelligentie zou volgens het paper nanotechnologie sneller vooruit helpen, en zo de periode van kwetsbaarheid inkorten tussen de ontwikkeling van gevaarlijke nanoreplicators en de invoering van afdoende verdediging.

    In 2022 brachten Jonas B. Sandbrink, Hamish Hobbs, Jacob L. Swett, Allan Dafoe en Anders Sandberg het idee in een preprint naar voren als principe, met een iets andere naam. Sandberg citeert het principe in een boekhoofdstuk dat deels op die preprint is gebaseerd; de pdf is van maart 2024 en noemt Reuben College en het Future of Humanity Institute van de University of Oxford als de plekken waar Sandberg werkt. Volgens het principe, differential technology development (DTD), maak je gebruik van de manieren waarop de ene technologie de risico's van een andere verkleint, door hun onderlinge timing te beïnvloeden. Volgens Sandberg kan dat op drie manieren, doordat technologieën voor veiligheid, voor verdediging of als vervanging eerder komen dan technologieën die het risico vergroten, of in plaats daarvan. Het hoofdstuk wijst ook op de mohisten uit de tijd van de Strijdende Staten in China, die kennis over verdediging in oorlog verspreidden, om aanvalsoorlogen onpraktisch te maken. Volgens Sandberg is dat misschien ook de vroegste poging om bepaalde technologieën te versnellen om de schade van andere technologieën te verkleinen.

    ### Een bezwaar van een onderzoeker

    Yann LeCun noemt in aflevering #416 van de Lex Fridman Podcast (uitgeschreven tekst geplaatst op 7 maart 2024) de meeste aannames achter catastrofescenario's onjuist. De eerste aanname is volgens LeCun dat superintelligentie als één gebeurtenis komt, waarbij we op een dag het geheim ontdekken en een superintelligente machine aanzetten, die de wereld overneemt en ons allemaal doodt, omdat we dat nooit eerder deden. Dat klopt volgens LeCun niet, en het wordt geen gebeurtenis. Veel mensen zouden zulke systemen bouwen, en als er één ontspoort, kun je volgens LeCun de goede inzetten tegen de ontspoorde, zodat het de slimme AI-politie van de een wordt tegen de ontspoorde AI van de ander. En zulke systemen zullen volgens LeCun niet willen overheersen, omdat dat verlangen in een intelligent systeem ingebouwd moet worden. Het bezwaar van LeCun richt zich op het beeld van één beslissende gebeurtenis; of het ook een antwoord is op het accumulatieve beeld uit de vorige les, zegt LeCun daar niet. Die lezing is van deze les.

    ### Waar ze het oneens zijn

    Ze verschillen over controle. Kasirzadeh vraagt in de vorige les om centrale controle tegen beslissend risico. Buterin schrijft dat centrale controle omwille van veiligheid altijd aantrekkelijk blijft voor een deel van de mensen, en dat lezers daar veel voorbeelden van kennen, met de nadelen ervan. Een paragraaf van Narayanan en Kapoor stelt al in de kop dat nonproliferatie niet af te dwingen is en tot single points of failure leidt. En volgens de post over e/acc werkt concurrentie beter dan optimale sturing van bovenaf. Ze verschillen ook over tempo. Volgens Andreessen kost elke vertraging van AI levens; Buterin houdt een zachte pauze alleen achter de hand; en Bostrom wijst niet alleen op het tempo, maar op tempo en volgorde samen.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **differential technology development** | technologieën versnellen die de risico's van andere verkleinen, en technologieën afremmen die risico vergroten, zodat de bescherming er is voordat het gevaar komt waartegen ze beschermt; Bostrom (2002) noemde het differential technological development |
        | effective accelerationism (e/acc) | een stroming die technologie zo snel mogelijk vooruit wil helpen; een van de vroege teksten is een post van juli 2022 |
        | defensive accelerationism (d/acc) | de d/acc van Buterin (2023), in de woorden van Buterin defensive (of decentralization, of differential) acceleration: technologie versnellen die de verdediging helpt, op een decentrale en democratische manier |
        | AI als normale technologie | het wereldbeeld van Narayanan en Kapoor (2025): AI als gereedschap dat mensen onder controle kunnen en moeten houden, ingrijpend zoals elektriciteit en internet |
        | speculatief risico | naar Narayanan en Kapoor een risico waarvan onzeker is of het wel groter is dan nul, een onzekerheid die nieuwe observaties of verder onderzoek kunnen wegnemen |
        | zachte pauze | in het voorstel van Buterin het vermogen om de wereldwijd beschikbare compute op een kritiek moment kort en sterk te verminderen |
        """),
        L.watch_out("Dit zijn standpunten, geen onderzoeksresultaten: Narayanan en Kapoor zetten een wereldbeeld neer, het redactionele stuk in Nature en het stuk van Hanna en Bender zijn opiniestukken, en het manifest en de posts van Buterin zijn essays. Geen van hen geeft een eigen schatting van hoe waarschijnlijk een catastrofe is; Buterin noemt getallen van anderen, waar de volgende les naar kijkt. De modellen in oefening 1 en 2 zijn verzonnen, niet die van Bostrom, van Sandbrink en anderen of van Buterin."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo plaats je een standpunt over AI-risico:

    1. Zoek waar het zich het meest zorgen over maakt: een catastrofe, de schade van nu, of vooruitgang die wordt tegengehouden.
    2. Vraag je af wat het met het tempo wil: sneller, langzamer of in een andere volgorde.
    3. Vraag je af wie de controle zou hebben: een paar centrale partijen, of veel.
    4. Zoek wat het zegt over de volgorde waarin technologieën komen.
    5. Kijk wat voor tekst het is: een wereldbeeld, een opiniestuk, een manifest, een essay of een paper.
    6. Geef het weer zoals de bron het zegt, met alle voorbehouden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent één verzonnen paar uit: een gevaarlijke technologie die in jaar 4 komt, en de bescherming ertegen, die in jaar 7 komt, of in jaar 3 als die naar voren wordt gehaald.
    """)
    return


@app.cell
def example_pair():
    risk = 4   # verzonnen: het jaar waarin de gevaarlijke technologie komt
    guard = 7   # verzonnen: het jaar waarin de bescherming ertegen komt
    early_guard = 3   # verzonnen: dezelfde bescherming, naar voren gehaald
    print(max(0, guard - risk), max(0, early_guard - risk))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de bescherming in jaar 7 is het gevaar 3 jaar onbeschermd. Met de bescherming in jaar 3 is die er eerst: 3 − 4 is −1, en `max` maakt daar 0 van, want een bescherming die vroeg komt, laat geen onbeschermde jaren over, en ook geen negatief aantal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_exposure():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. exposure: jaren zonder bescherming

        Een verzonnen model van differential technology development, niet dat van Bostrom of van Sandbrink en anderen: vijf gevaarlijke technologieën, elk met het jaar waarin die komt, en een bescherming tegen elk ervan, met het jaar waarin de bescherming komt. Bostrom schrijft over de periode van kwetsbaarheid tussen de ontwikkeling van gevaarlijke nanoreplicators en de invoering van afdoende verdediging; dit model telt zulke perioden in jaren.

        Schrijf `exposure(risks, guards)`, die een lijst `result` teruggeeft met voor elk paar het aantal jaren dat de gevaarlijke technologie er is voordat de bescherming komt, en 0 als de bescherming er eerst is. `risks` en `guards` zijn lijsten van dezelfde lengte: `risks[k]` is het jaar waarin het gevaar van paar `k` komt, en `guards[k]` het jaar waarin de bescherming ervan komt. Voeg in een `for`-loop met `k` over `range(len(risks))` het getal van elk paar toe aan `result`.

        De check roept `exposure([3, 6, 7, 9, 11], [6, 5, 12, 10, 11])` aan, vijf verzonnen paren, en daarna andere gevallen.
        """),
        L.task("De onbeschermde jaren van elk paar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exposure():
    box_exposure = L.code_box("exposure", lines=5)
    box_exposure
    return (box_exposure,)


@app.cell(hide_code=True)
def check_exposure(box_exposure):
    result_exposure = L.check("exposure", box_exposure.value)
    result_exposure
    return (result_exposure,)


@app.cell(hide_code=True)
def hints_exposure():
    L.hints(
        "exposure",
        [
            "Begin klein: één paar met het gevaar in jaar 4 en de bescherming in jaar 7, en één met het gevaar in jaar 6 en de bescherming in jaar 5. Hoeveel jaar is elk gevaar onbeschermd?",
            "Begin `result` als een lege lijst. Reken in een `for`-loop met `k` over `range(len(risks))` de onbeschermde jaren van paar `k` uit zoals in het uitgewerkte voorbeeld, nooit minder dan 0, en voeg die toe. Geef na de loop `result` terug.",
            "`        result.append(max(0, ___))`",
        ],
        why="`result = []`, dan `for k in range(len(risks)):` met `result.append(max(0, guards[k] - risks[k]))`; na de loop `return result`. Met het gevaar in jaar 4 en de bescherming in jaar 7 is het gevaar 3 jaar onbeschermd; met het gevaar in jaar 6 en de bescherming in jaar 5 is de bescherming er eerst, dus 0.",
    )
    return


@app.cell(hide_code=True)
def text_after_exposure():
    mo.md(r"""
    Voor de vijf verzonnen paren is het antwoord `[3, 0, 5, 1, 0]`: samen 9 onbeschermde jaren. Bij paar 2 komt de bescherming een jaar vóór het gevaar, en bij paar 5 in hetzelfde jaar, dus geen van beide laat een onbeschermd jaar over. Een bescherming die vroeg komt, maakt een bescherming die elders te laat komt niet goed: het jaar over bij paar 2 helpt paar 3 niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_progress():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. progress: een zachte pauze

        Een verzonnen model van de zachte pauze van Buterin, niet dat van Buterin zelf: elk jaar levert evenveel vooruitgang op als het deel van de compute in de wereld dat er dat jaar beschikbaar is, 1 in een gewoon jaar. Schrijf `progress(shares)`, die een lijst `result` teruggeeft met de vooruitgang aan het eind van elk jaar. `shares` is een lijst met het aandeel van elk jaar. Laat `total` bij 0 beginnen; reken in een `for`-loop met `share` over `shares` de nieuwe `total` uit en voeg die toe aan `result`.

        De check roept `progress([1, 1, 0.05, 0.05, 1, 1, 1, 1])` aan, acht verzonnen jaren met een pauze van 2 jaar met 95% minder compute, binnen de zo'n 90 tot 99% en de 1 tot 2 jaar uit het voorstel van Buterin, en daarna andere gevallen.
        """),
        L.task("De vooruitgang aan het eind van elk jaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_progress():
    box_progress = L.code_box("progress", lines=7)
    box_progress
    return (box_progress,)


@app.cell(hide_code=True)
def check_progress(box_progress):
    result_progress = L.check("progress", box_progress.value)
    result_progress
    return (result_progress,)


@app.cell(hide_code=True)
def hints_progress():
    L.hints(
        "progress",
        [
            "Begin klein: drie jaar met aandelen van 1; 0,5 en 1. Hoeveel vooruitgang is er aan het eind van elk van de drie jaren?",
            "Begin `result` als een lege lijst en `total` bij 0. Reken in een `for`-loop met `share` over `shares` de nieuwe `total` uit met de oude en het aandeel van dit jaar, en voeg die toe. Geef na de loop `result` terug.",
            "`        total = ___`",
        ],
        why="`result = []` en `total = 0`, dan `for share in shares:` met `total = total + share` en `result.append(total)`; na de loop `return result`. Met aandelen van 1; 0,5 en 1 is de vooruitgang 1 aan het eind van het eerste jaar, 1,5 aan het eind van het tweede en 2,5 aan het eind van het derde.",
    )
    return


@app.cell(hide_code=True)
def text_after_progress():
    mo.md(r"""
    Voor de acht verzonnen jaren is de vooruitgang aan het eind van jaar 8 gelijk aan 6,1, tegen 8 zonder pauze. Zonder pauze is een vooruitgang van 6,1 na 6,1 jaar bereikt; met deze pauze pas aan het eind van jaar 8, 1,9 jaar later. Dat is de tijd die deze pauze oplevert: 2 × 0,95 jaar. Buterin wil met de pauze de mensheid meer tijd geven om zich voor te bereiden. Het model laat veel weg: vooruitgang hoeft compute niet zo nauw te volgen, de pauze in het voorstel van Buterin geldt alleen voor hardware op industriële schaal, en verder verandert er in het model niets tijdens de pauze. Buterin noemt beide strategieën, aansprakelijkheid en de pauzeknop, ook maar tijdelijke noodoplossingen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: bescherming eerder

    De vijf verzonnen paren uit oefening 1 blijven hetzelfde. Laat elke bescherming hetzelfde aantal jaren eerder komen, en kijk hoeveel onbeschermde jaren er overblijven.
    """)
    return


@app.cell(hide_code=True)
def demo_sooner():
    sooner = L.explore("sooner", "sooner: hoeveel jaar eerder elke bescherming komt",
                       lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_SOONER, role="",
                       task="Zoek het kleinste aantal jaren eerder waarbij geen enkele gevaarlijke technologie meer onbeschermd is.")
    sooner
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Als elke bescherming volgens de eigen planning komt, zijn de vijf gevaren samen 9 jaar onbeschermd; als elke bescherming 1 jaar eerder komt, 6 jaar; 2 jaar eerder, 4 jaar; 3 jaar eerder, 2 jaar; 4 jaar eerder, 1 jaar; en vanaf 5 jaar eerder helemaal niet. Het laatste onbeschermde jaar hoort bij paar 3, waarvan de bescherming 5 jaar na het gevaar komt. Als alles evenveel jaren eerder kwam, gevaren en beschermingen allebei, veranderde geen enkel gat: in dit model telt wanneer elke bescherming komt vergeleken met het gevaar. Een hoger tempo dat elk gat kleiner maakt, zoals Bostrom van superintelligentie voor nanotechnologie verwachtte, zit niet in dit model. Ontwikkelingen die aanval en verdediging evenveel vooruit helpen, noemt Bostrom neutraal voor de veiligheid, tenzij landen die we als verantwoordelijk zien ze uitvoeren; dan zijn ze volgens Bostrom een voordeel, voor zover ze onze technologische voorsprong op mogelijke vijanden vergroten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_normal():
    mo.md(r"""
    ### Oefening 3. Een column in de krant

    In een column staat dat Narayanan en Kapoor AI een normale technologie noemen, dat AI volgens hen dus niet veel zal veranderen, en dat er geen serieuze risico's zijn om je zorgen over te maken.
    """)
    return


@app.cell(hide_code=True)
def choice_normal_cell():
    choice_normal = L.choice(
        "normal",
        question="Geeft de column Narayanan en Kapoor goed weer?",
        options={
            "speculative": "Ja: ze noemen catastrofale misalignment een speculatief risico",
            "right": "Nee: normaal is niet klein, en ze noemen serieuze risico's",
            "half": "Deels: AI verandert veel, maar ze zien er geen serieuze risico's in",
        },
        correct="right",
        why={
            "right": "Wie AI normaal noemt, doet volgens Narayanan en Kapoor de impact niet tekort, want ook ingrijpende technologieën voor algemeen gebruik, zoals elektriciteit en internet, zijn in hun opvatting normaal. Ze noemen ook systeemrisico's, zoals machtsconcentratie en massasurveillance, die volgens hen veel belangrijker worden als AI normale technologie is.",
            "speculative": "Met speculatief bedoelen ze dat onzeker is of het risico nul is, een onzekerheid die nieuwe observaties of verder onderzoek kunnen wegnemen; dat is iets anders dan geen risico. En ze noemen systeemrisico's, zoals machtsconcentratie en massasurveillance. Zie *AI als normale technologie*.",
            "half": "Ze zien wel serieuze risico's: systeemrisico's, zoals machtsconcentratie, afnemend onderling vertrouwen en massasurveillance, die volgens hen veel belangrijker worden dan de catastrofale als AI normale technologie is. Zie *AI als normale technologie*.",
        },
    )
    choice_normal
    return (choice_normal,)


@app.cell(hide_code=True)
def text_exercise_dacc():
    mo.md(r"""
    ### Oefening 4. Snel en verspreid

    Een groep wil alle AI zo snel mogelijk laten bouwen, door veel kleine bedrijven in plaats van een paar grote. Welke technologieën eerst komen, maakt de groep niet uit. De groep noemt de eigen opvatting d/acc.
    """)
    return


@app.cell(hide_code=True)
def choice_dacc_cell():
    choice_dacc = L.choice(
        "dacc",
        question="Wat laat de opvatting van de groep weg uit d/acc, zoals Buterin het beschrijft?",
        options={
            "nothing": "Niets, want bij d/acc draait alles om macht verdelen over veel handen",
            "right": "Voorrang voor technologie die de verdediging sterker maakt",
            "slow": "Alle AI afremmen tot die veilig is, zoals d/acc eist",
        },
        correct="right",
        why={
            "right": "Volgens Buterin draait d/acc om decentrale en democratische, differentiële en defensieve versnelling, en om technologie die de verhouding tussen aanval en verdediging naar de verdediging laat doorslaan. Buterin zet het af tegen decentrale versnelling zonder dat differentiële, defensieve deel, wat volgens Buterin neerkomt op e/acc, maar dan decentraal. De groep is snel en verspreid, maar niet differentieel.",
            "nothing": "Macht verdelen is de ene helft. De andere helft is het differentiële, defensieve deel, technologie die de verhouding tussen aanval en verdediging naar de verdediging laat doorslaan. Zonder dat deel is een opvatting volgens Buterin e/acc, maar dan decentraal. Zie *d/acc*.",
            "slow": "d/acc versnelt technologie; het geeft sommige technologieën voorrang boven andere. Buterin houdt een zachte pauze alleen achter de hand, bij de overtuiging dat er iets krachtigers dan regels voor aansprakelijkheid nodig is. Wat de groep weglaat, is het differentiële, defensieve deel. Zie *d/acc*.",
        },
    )
    choice_dacc
    return (choice_dacc,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Eén klap of veel kleine stappen](/course/open/rai-accumulative-risk?lang=nl) zette twee soorten catastrofaal risico naast elkaar, beslissend en accumulatief. Deze les keek naar standpunten naast die kijk: Narayanan en Kapoor, voor wie AI een normale technologie is, hoe ingrijpend ook; schrijvers die de aandacht bij de schade van nu willen; e/acc en Andreessen, die technologie sneller willen; d/acc van Buterin, sneller maar met verdediging voorop; en differential technology development, het idee dat de volgorde telt waarin technologieën komen. De standpunten verschillen het meest over tempo en over wie de controle moet hebben. De volgende les, *Wat experts verwachten*, kijkt naar wat onderzoekers verwachten, in getallen.

    **Een voorbeeld uit het dagelijks leven.** Een gemeente opent een nieuwe zwemplas. Dat kan eerder of later, maar belangrijker is of de strandwacht begint voordat de eerste zwemmers komen, of pas een maand erna. Eerder of later opengaan verandert niets aan die maand zonder strandwacht; alleen de strandwacht eerst laten beginnen doet dat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dacc, choice_normal, result_exposure, result_progress):
    L.wrap_up(result_exposure, result_progress, choice_normal, choice_dacc)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
