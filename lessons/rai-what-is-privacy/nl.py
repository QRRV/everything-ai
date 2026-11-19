# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-what-is-privacy. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-what-is-privacy", "nl")
    C = L.checks   # de taxonomie van Solove, drie verzonnen tabellen over dezelfde verzonnen mensen, en een verzonnen enquête


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) en [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_consent():
    quick_consent = L.choice(
        "quick-consent",
        question="Een dataset met berichten heeft de licentie CC BY. In de datasheet is het veld `consent` leeg. Wat vertelt de licentie je niet?",
        options={
            "right": "Of de schrijvers instemden",
            "credit": "Of je de maker moet vermelden",
            "copy": "Of je er kopieën van mag delen",
        },
        correct="right",
        why={
            "right": "Een licentie komt van wie de rechten heeft, en zegt wat je met de data mag doen. Of de mensen die de berichten schreven ermee instemden, is een aparte vraag: toestemming.",
            "credit": "Dat zegt CC BY wel: je mag de data gebruiken en delen als je de maker noemt. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "copy": "CC BY staat toe dat je kopieën deelt, zolang je de maker noemt. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
        },
    )
    quick_consent
    return (quick_consent,)


@app.cell(hide_code=True)
def quick_check_stakeholders():
    quick_stakeholders = L.choice(
        "quick-stakeholders",
        question="Een model beslist welke huurders een herinnering krijgen over achterstallige huur. Is een huurder die nooit een herinnering krijgt een van de stakeholders?",
        options={
            "right": "Ja: het beslist ook over hen",
            "flagged": "Nee: alleen wie het aanwijst telt",
            "landlord": "Nee: alleen de verhuurder die het gebruikt",
        },
        correct="right",
        why={
            "right": "Stakeholders zijn alle mensen die belang hebben bij het systeem of erdoor geraakt worden, dus ook elke persoon over wie het beslist, zelfs als het besluit niets te doen.",
            "flagged": "Iemand niet aanwijzen is ook een beslissing over die persoon. Zie *Het idee* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
            "landlord": "De verhuurder is één stakeholder, maar niet de enige: de mensen over wie het model beslist tellen ook mee. Zie *Het idee* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
        },
    )
    quick_stakeholders
    return (quick_stakeholders,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    AI-systemen worden gebouwd uit informatie over mensen: wat ze schrijven, kopen en zoeken, en waar ze komen. Een model kan zulke informatie ook combineren, en dingen afleiden die niemand het verteld heeft. Of dat een privacyprobleem is, hangt af van wat je onder privacy verstaat, en daar is geen eenduidig antwoord op. Deze les geeft je drie opvattingen van privacy, een lijst van zestien soorten privacyproblemen om te benoemen wat er misgaat, en een vraag die door het hele debat loopt: als mensen zeggen dat ze erom geven, waarom geven ze hun data dan toch weg?

    **Een voorbeeld uit het dagelijks leven.** Je vertelt je huisarts dat je slecht slaapt. Je zou het vervelend vinden als je werkgever het hoorde. Het is dezelfde informatie, en dat de huisarts het weet is geen probleem; waar het daarna heen gaat wel.
    """)
    return


@app.cell(hide_code=True)
def predict_card():
    guess_card = L.guess(
        "card",
        question="In een onderzoek van Acquisti en Grossklags hadden sommige mensen veel zorgen over het verzamelen van offline identificerende gegevens, zoals naam en adres. Welk deel van hen meldde zich volgens jou aan voor een klantenkaart met hun echte identificerende gegevens?",
        options={
            "quarter": "Ongeveer 1 op de 4",
            "half": "Ongeveer de helft",
            "right": "Bijna 9 op de 10",
        },
        reveal="**Bijna 9 op de 10: 87,5%**, zoals Solove het onderzoek beschrijft. Zo'n kloof tussen wat mensen zeggen en wat ze doen heet de privacyparadox. Of die laat zien dat mensen er niet echt om geven, is een andere vraag: daar komt deze les op terug.",
    )
    guess_card
    return (guess_card,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Iedereen gebruikt het woord privacy, en bijna niemand bedoelt er hetzelfde mee. De rechtsgeleerde Daniel Solove schreef in 2006 dat privacy een begrip in wanorde is, waarvan niemand kan zeggen wat het betekent. In plaats van één definitie geeft deze les je drie opvattingen, en een lijst problemen die je kunt benoemen.

        ### Drie opvattingen van privacy

        In een eerdere publicatie, uit 2002, verdeelt Solove wat wetenschappers over privacy schreven over zes kopjes, en merkt op dat die alleen volgen hoe er in de wetenschap over is nagedacht. De course neemt er drie van, voegt er twee samen en zet er één later idee bij: zo ontstaan drie opvattingen.

        **Met rust gelaten worden.** In 1890 schreven Samuel Warren en Louis Brandeis over een "right to be let alone", een recht om met rust gelaten te worden; die woorden namen ze over uit een boek uit 1880 van rechter Thomas Cooley over onrechtmatige daad, het recht over schade die de een de ander toebrengt. Dicht daarbij ligt privacy als geheimhouding: privacy wordt geschonden als iets wat verborgen was bekend wordt.

        **Zeggenschap.** Alan Westin schreef in 1967, in woorden die Solove citeert, dat privacy de aanspraak is van personen, groepen of instellingen om zelf te bepalen wanneer, hoe en in welke mate informatie over hen aan anderen wordt doorgegeven.

        **Gepaste stroom.** Helen Nissenbaum noemt privacy de gepaste stroom van persoonlijke informatie. Wat telt, is of een stroom informatie past bij de context waar hij vandaan komt. Nissenbaum schrijft dat het er alleen om gaat of de stroom gepast is, niet of hij überhaupt plaatsvindt. Dat je huisarts weet dat je slecht slaapt, past bij de zorg; dat je werkgever het weet niet. Nissenbaum noemt deze opvatting contextual integrity, het onderwerp van *Informatie op de verkeerde plek*, twee lessen verder.

        De opvattingen kunnen het over hetzelfde geval oneens zijn. Iets wat je zelf hebt gepost, is niet meer geheim, dus geheimhouding ziet geen probleem als anderen het doorgeven. Zeggenschap vraagt of jij het besliste. Gepaste stroom vraagt waar het daarna heen ging, en of dat past bij waar het vandaan kwam.

        ### De taxonomie van Solove

        Een taxonomie is een indeling in groepen en soorten. Die van Solove uit 2006 definieert privacy niet, maar deelt de activiteiten in die privacyproblemen kunnen veroorzaken. Solove vindt vier groepen en zestien soorten. De eerste drie groepen volgen informatie die steeds verder buiten de zeggenschap komt van de persoon over wie ze gaat: ze wordt verzameld, verwerkt en verspreid. De vierde, binnendringen, komt naar de persoon toe, en hoeft niet over informatie te gaan. Verwerken is bij Solove smaller dan verwerking in de AVG van de volgende les, waar ook verzamelen al verwerking is.
        """),
        mo.Html(C.taxonomy_picture("nl")),
        mo.md(r"""
        Veel van de soorten passen bij AI-systemen. Een camera in een deurbel die de straat filmt, is observeren (surveillance). Verzamelt een bedrijf chats om vragen te beantwoorden en traint het er later zonder te vragen een model mee, dan is dat hergebruik voor een ander doel (secondary use). Een model dat een account koppelt aan de persoon erachter, is identificeren. En een databroker die aankopen, locaties en berichten samenvoegt tot één profiel, is aggregatie. Solove schrijft daarover dat een stukje informatie hier of daar weinig zegt, maar dat stukjes data samen een portret van iemand gaan vormen: het geheel wordt meer dan de delen.

        ### De privacyparadox

        De privacyparadox is het verschijnsel dat mensen zeggen dat ze privacy heel belangrijk vinden, en hun persoonsgegevens toch voor heel weinig weggeven of niet beschermen. De naam raakte halverwege de jaren 2000 in gebruik. Zoals Solove de onderzoeken beschrijft, meldde in het onderzoek van Acquisti en Grossklags uit de gok 87,5% van de mensen met veel zorgen zich aan voor een klantenkaart met hun echte identificerende gegevens; in een ander zei 95% belang te hechten aan de bescherming van hun persoonlijke informatie, en koos daarna bijna iedereen de winkel met de lagere prijs die meer persoonsgegevens vroeg.

        Niet iedereen is het eens over wat dat laat zien. In 2017 brachten Susanne Barth en Menno de Jong 32 publicaties samen in een overzicht, en vonden dat mensen afwegen wat ze riskeren tegen wat ze krijgen, of het risico verwaarloosbaar inschatten. Ze waarschuwen ook dat wat mensen vinden en van plan zijn nog geen gedrag is, en dat de onderzoeken zelden meten wat mensen echt doen. Solove noemt de paradox in 2021 een mythe die voortkomt uit gebrekkige logica: een keuze om één stukje data te delen in één context, voor één voordeel, zegt weinig over hoeveel iemand in het algemeen om privacy geeft. Wie data met anderen deelt, geeft daarmee volgens Solove niet alle privacy op. Barth en de Jong, die vier jaar eerder schreven, gaan ervan uit dat de paradox bestaat en zoeken er een verklaring voor: de bronnen zijn het oneens over de vraag of hij bestaat.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **privacy** | wat iemand voor anderen kan afschermen, en hoe informatie over die persoon verzameld, gebruikt en doorgegeven mag worden; er is niet één definitie |
        | **taxonomie** | een indeling van dingen in groepen en soorten; die van Solove deelt de activiteiten in die privacyproblemen kunnen veroorzaken |
        | **aggregatie** | stukjes informatie over één persoon samenvoegen, zodat ze samen meer zeggen dan elk stukje apart |
        | **privacyparadox** | het verschijnsel dat mensen zeggen dat ze privacy belangrijk vinden, en hun persoonsgegevens toch voor heel weinig weggeven |
        """),
        L.watch_out("De taxonomie van Solove is geen definitie van privacy. Ze deelt activiteiten in die privacyproblemen kunnen veroorzaken; privacy zelf heeft volgens Solove geen eenduidige kern."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo benoem je een privacyprobleem in een AI-systeem:

    1. Vraag wat het systeem met informatie over mensen doet: verzamelt het die, verwerkt het die, verspreidt het die, of dringt het binnen bij de mensen zelf?
    2. Zoek de groep en de soort in de taxonomie van Solove. Dat is `group_of`.
    3. Kijk wat stukjes informatie samen zeggen, niet alleen elk apart: aggregatie. Dat is `facts`.
    4. Leg het langs de drie opvattingen: wordt iets verborgens bekend, beslisten de mensen zelf, en past de stroom bij de context waar de informatie vandaan kwam?
    5. Als mensen zeggen dat ze zich zorgen maken en toch delen, meet dan de kloof, en vraag wat ze ervoor terugkregen voordat je concludeert dat ze er niet om geven. Dat is `share_who_shared`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier gebeurtenissen in AI-systemen, verzonnen door de course, elk met de soort privacyprobleem waar het om gaat. `_group` geeft de groep van elk van die soorten. De cel print voor elke gebeurtenis de groep, de soort en de gebeurtenis.
    """)
    return


@app.cell
def example_events():
    # vier verzonnen gebeurtenissen in AI-systemen, elk met de soort privacyprobleem waar het om gaat
    _events = [("een deurbelcamera filmt de hele dag de straat", "surveillance"),
               ("een chatapp traint zonder te vragen een model op de chats", "secondary use"),
               ("een databroker voegt aankoop- en locatiedata samen", "aggregation"),
               ("een chatbot herhaalt een adres uit een openbaar register", "increased accessibility")]
    _group = {"surveillance": "collection", "secondary use": "processing", "aggregation": "processing",
              "increased accessibility": "dissemination"}

    for _event, _kind in _events:
        print(_group[_kind], "|", _kind, "|", _event)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee van de vier gebeurtenissen vallen onder verwerken (processing). Het adres in een openbaar register was al te vinden; de chatbot maakt het makkelijker te vinden, wat Solove toegankelijker maken (increased accessibility) noemt. `_group` gaat van een soort meteen naar de groep. De taxonomie van Solove gaat de andere kant op, van elke groep naar de lijst met soorten: om daar de groep van een soort te vinden, ga je de groepen langs. Dat is oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_group():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. group_of: bij welke groep hoort een soort?

        `group_of(kind, taxonomy)` krijgt een soort privacyprobleem `kind`, zoals `"aggregation"`, en een taxonomie `taxonomy`: een dictionary van elke groep naar de lijst met soorten in die groep. `TAXONOMY` is die van Solove, met de vier groepen en zestien soorten van het plaatje, zoals `"processing": ["aggregation", "identification", "insecurity", "secondary use", "exclusion"]`. De `for`-loop gaat `taxonomy.items()` langs: elk paar van de naam van een groep `group` en de lijst `kinds`. Vervang de `...` door twee regels: als `kind` in `kinds` staat, geef dan `group` terug. Na de loop geeft de functie `None` terug: geen enkele groep heeft de soort. De check roept `group_of("aggregation", TAXONOMY)` aan, daarna andere soorten, een soort die niet in de taxonomie staat, en een kleinere taxonomie.
        """),
        L.task("Geef de groep terug waarvan de lijst met soorten `kind` bevat, of `None`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_group():
    box_group = L.code_box("group", lines=6)
    box_group
    return (box_group,)


@app.cell(hide_code=True)
def check_group(box_group):
    result_group = L.check("group", box_group.value)
    result_group
    return (result_group,)


@app.cell(hide_code=True)
def hints_group():
    L.hints(
        "group",
        [
            "Stap 2 van het recept. Elke keer door de loop is `kinds` de lijst van één groep: vraag of `kind` erin staat.",
            "Een `if` met `in`, en daaronder een `return`. Welke van de twee loopvariabelen is de naam van een groep, en welke de lijst?",
            "`if kind in kinds:` en daaronder `return ___`",
        ],
        why="`if kind in kinds:` en daaronder `return group`. `group_of(\"aggregation\", TAXONOMY)` is `\"processing\"`, en een soort die geen groep heeft, zoals `\"profiling\"`, geeft `None`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_facts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. facts: wat zeggen de tabellen samen?

        Drie verzonnen apps houden elk een tabel bij over dezelfde verzonnen mensen, per e-mailadres: `RUNS`, wat een hardloopapp weet van hun hardlooprondes, `PHARMACY`, wat een apotheekpas weet van hun aankopen, en `FORUM`, wat een forum weet uit hun berichten. `TABLES` is de lijst van de drie dictionaries. Vier van de zes mensen, met de feiten zoals ze in het Engels in de tabellen staan:

        | persoon | hardloopapp | apotheekpas | forum |
        |---|---|---|---|
        | sam@example.org | runs from Harbor Street at 6:30 on weekdays | buys allergy tablets every April | writes about night shifts at the hospital |
        | noor@example.org | runs in the dunes on Sunday mornings | | writes about moving house in May |
        | lior@example.org | | buys reading glasses every two years | |
        | ash@example.org | runs to the station at 7:15 on weekdays | buys sleeping tablets every week | writes about looking for a new job |

        `facts(key, tables)` krijgt een e-mailadres `key` en een lijst `tables` met dictionaries, en geeft een lijst terug met wat elke tabel die `key` heeft erover zegt. `found` begint als lege lijst, en de `for`-loop gaat elke `table` langs. Vervang de `...` door twee regels: als `key` in `table` staat, voeg dan `table[key]` toe aan `found`. De functie geeft `found` terug. De check roept `facts("sam@example.org", TABLES)` aan, daarna andere mensen, een adres dat in geen enkele tabel staat, en een korte lijst.
        """),
        L.task("Geef een lijst terug met wat elke tabel die `key` heeft erover zegt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_facts():
    box_facts = L.code_box("facts", lines=7)
    box_facts
    return (box_facts,)


@app.cell(hide_code=True)
def check_facts(box_facts):
    result_facts = L.check("facts", box_facts.value)
    result_facts
    return (result_facts,)


@app.cell(hide_code=True)
def hints_facts():
    L.hints(
        "facts",
        [
            "Stap 3 van het recept. Niet elke tabel heeft elke persoon, dus vraag eerst of `key` in `table` staat.",
            "Een `if` met `in`, en daaronder `found.append(...)`. Hoe zoek je in een dictionary op wat die bij een key bewaart?",
            "`if key in table:` en daaronder `found.append(___)`",
        ],
        why="`if key in table:` en daaronder `found.append(table[key])`. `facts(\"sam@example.org\", TABLES)` geeft drie feiten, één uit elke app.",
    )
    return


@app.cell(hide_code=True)
def text_after_facts():
    mo.md(r"""
    Elke app apart weet één ding over sam. Samen vertellen ze waar sam op doordeweekse ochtenden begint met hardlopen, en dus misschien waar sam woont, dat sam een allergie heeft, en dat sam nachtdiensten draait in een ziekenhuis. Elke app apart weet weinig, en toch zegt het geheel meer dan de delen: dat is aggregatie. Een systeem dat zulke tabellen samenvoegt, weet het ook, en kan zich ook vergissen: misschien begint sam bij het ziekenhuis en niet thuis.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_views():
    mo.md(r"""
    ### Oefening 3. Drie opvattingen, één geval
    """)
    return


@app.cell(hide_code=True)
def choice_views_cell():
    choice_views = L.choice(
        "views",
        question="Jules ging in de voorwaarden van een app akkoord dat die de stappendata van Jules mag doorgeven aan partners. De app stuurt ze naar de huisarts van Jules, en ook naar een verzekeraar, die de premie van Jules verhoogt. Welke opvatting ziet een probleem in de stroom naar de verzekeraar, hoewel Jules akkoord ging?",
        options={
            "secret": "Geheimhouding: de data waren eerst geheim",
            "control": "Zeggenschap: Jules had niets te zeggen",
            "right": "Gepaste stroom: past niet bij de context",
        },
        correct="right",
        why={
            "secret": "De stappendata waren niet verborgen: Jules deelde ze met de app en ging akkoord dat die ze mag doorgeven. Als informatie gedeeld is, ziet geheimhouding geen probleem meer. Zie *Drie opvattingen van privacy* in *Het idee*.",
            "control": "Jules had wel iets te zeggen, in elk geval op papier, en ging akkoord in de voorwaarden. In de opvatting van zeggenschap beslist de persoon zelf, dus ziet die opvatting geen probleem in een stroom waarmee de persoon akkoord ging. Zie *Drie opvattingen van privacy* in *Het idee*.",
            "right": "Stappendata die naar een huisarts gaan, passen bij de context van de zorg; dezelfde getallen gebruiken om bij een verzekeraar een premie te bepalen niet, en een vinkje in de voorwaarden maakt dat niet vanzelf goed. In de opvatting van Nissenbaum is toestemming niet het enige dat bepaalt of een stroom gepast is.",
        },
    )
    choice_views
    return (choice_views,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. share_who_shared: het een zeggen, het ander doen?

        `SURVEY` is een enquête onder 20 mensen, verzonnen door de course. Elke persoon is een dictionary met `"concern"`, hoeveel zorgen die persoon zegt zich te maken over privacy, van 1 (helemaal niet) tot 5 (heel veel), en `"shared"`, `True` als die persoon zich met naam en adres aanmeldde voor een klantenkaart, voor korting. `share_who_shared(people, least)` krijgt zo'n lijst `people` en een getal `least`, en geeft van de mensen van wie de zorgscore `least` of meer is het aandeel terug dat zich aanmeldde. `count` en `shared` beginnen op 0, en de `for`-loop gaat elke `person` langs. Vervang de `...` door vier regels: als `person["concern"]` minstens `least` is, tel dan 1 op bij `count`, en als `person["shared"]` ook `True` is, tel dan 1 op bij `shared`. De functie geeft `shared / count` terug. De check roept `share_who_shared(SURVEY, 4)` aan, daarna andere waarden van `least`, en een korte lijst.
        """),
        L.task("Geef van de mensen van wie de zorgscore `least` of meer is het aandeel terug dat zich aanmeldde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=10)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Stap 5 van het recept. Tel twee dingen: in `count` de mensen die zich genoeg zorgen maken, en in `shared` wie van hen deelde.",
            "Twee `if`s, de ene in de andere: de buitenste over de zorgscore van de persoon, de binnenste over `person[\"shared\"]`. Minstens is `>=`.",
            "`if person[\"concern\"] >= ___:`",
        ],
        why="`if person[\"concern\"] >= least:` met daaronder `count = count + 1` en `if person[\"shared\"]:` met daaronder `shared = shared + 1`. `share_who_shared(SURVEY, 4)` is 0,875: 7 van de 8 mensen met zorgscore 4 of meer meldden zich aan.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    In de verzonnen enquête meldden 7 van de 8 mensen met zorgscore 4 of meer zich aan voor de kaart: 0,875, hetzelfde aandeel als in het echte onderzoek uit de gok; de course koos het zo. Van de 12 anderen deden 10 dat: 0,833. Gelezen als de privacyparadox doen de bezorgde mensen alsof ze er niet om geven. Gelezen zoals Solove zulke onderzoeken leest, maakte elk van hen één keuze, over een naam en een adres, voor korting, bij één winkel; dat zegt weinig over hoeveel ze in het algemeen om privacy geven. Een enquête als deze meet wat mensen zeggen, en maar één ding dat ze doen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Oefening 5. Een kop
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Stel dat een nieuwssite schrijft over het onderzoek uit de gok: \"Niemand geeft om privacy: 87,5% ruilt adres in voor korting.\" Wat ondersteunt het onderzoek, zoals Solove het beschrijft?",
        options={
            "exact": "De kop zoals hij er staat",
            "right": "Minder: één keuze, niet wat ze waarderen",
            "more": "Meer: bezorgde mensen liegen ook in enquêtes",
        },
        correct="right",
        why={
            "exact": "De 87,5% gaat over de mensen met veel zorgen, niet over iedereen, en ze meldden zich aan voor een klantenkaart. Solove noemt het een logische sprong om uit zo'n keuze in één context af te leiden hoe mensen privacy waarderen. Zie *De privacyparadox* in *Het idee*.",
            "right": "Het onderzoek zag mensen met veel zorgen zich aanmelden voor een klantenkaart met hun echte identificerende gegevens: wat ze deden in één geval, voor één voordeel. Solove noemt het een logische sprong om daaruit af te leiden hoe mensen privacy waarderen.",
            "more": "Het onderzoek zette wat mensen zeiden naast wat ze deden; het liet niet zien dat ze logen. Zie *De privacyparadox* in *Het idee*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) schreef je op met wiens toestemming een dataset gemaakt is, en in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl) vroeg je wie door een systeem schade kan oplopen. Privacy vraagt naar de mensen in de data zelf: welke informatie over hen gaat waarheen, en wat kan dat met hen doen. De taxonomie van Solove geeft elk probleem een naam, en de drie opvattingen zeggen waarom het er een is. In de volgende les, *De AVG voor AI*, lees je wat de wet in Europa zegt over informatie over mensen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_headline, choice_views, result_facts, result_group, result_share):
    L.wrap_up(result_group, result_facts, choice_views, result_share, choice_headline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
