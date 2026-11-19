# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-gdpr. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-gdpr", "nl")
    C = L.checks   # the six lawful bases, and a register, uses and forum posts of a made-up language-learning app


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="In welke groep van de taxonomie van privacyproblemen van Solove hoort aggregatie?",
        options={
            "right": "Verwerken",
            "collection": "Verzamelen",
            "dissemination": "Verspreiden",
        },
        correct="right",
        why={
            "right": "Aggregatie voegt stukjes informatie over één persoon samen die al verzameld waren: het is een van de vijf soorten verwerken.",
            "collection": "Onder verzamelen vallen observeren en ondervragen. Aggregatie voegt samen wat al verzameld was: verwerken. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
            "dissemination": "Verspreiden brengt informatie verder weg. Aggregatie voegt stukjes informatie over één persoon samen: verwerken. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def quick_check_flow():
    quick_flow = L.choice(
        "quick-flow",
        question="Welke opvatting van privacy vraagt of een stroom informatie past bij de context waar hij vandaan komt?",
        options={
            "right": "Gepaste stroom",
            "alone": "Met rust gelaten worden",
            "control": "Zeggenschap over informatie",
        },
        correct="right",
        why={
            "right": "De opvatting van Helen Nissenbaum: wat telt, is of een stroom past bij de context, niet of hij überhaupt plaatsvindt.",
            "alone": "Met rust gelaten worden, of iets geheimhouden, ziet een probleem als iets wat verborgen was bekend wordt. Zie *Drie opvattingen van privacy* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
            "control": "Zeggenschap vraagt of de persoon zelf besliste. Zie *Drie opvattingen van privacy* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
        },
    )
    quick_flow
    return (quick_flow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Modellen worden vaak getraind op informatie over mensen: wat ze schreven, zeiden, kochten of zochten. In de Europese Unie legt de AVG daar de regels voor vast. Die verbiedt niet dat je traint op persoonsgegevens, en staat ook niet alles toe: ze vraagt een reden die de wet aanvaardt, een duidelijk doel, en ruimte voor de mensen in de data om bezwaar te maken of hun gegevens te laten wissen. Deze les geeft je de belangrijkste woorden van de AVG, de zes grondslagen, de regel over doelen, twee van de rechten die de AVG mensen geeft, en wat toezichthouders tot nu toe over AI hebben gezegd. De les legt de wet uit; het is geen juridisch advies.

    **Een voorbeeld uit het dagelijks leven.** Een winkel vraagt je e-mailadres om je een bon te sturen. Een maand later stuurt de winkel je de nieuwsbrief. Je gaf het adres voor het een, en het wordt gebruikt voor iets anders. De AVG heeft daar een regel voor: doelbinding.
    """)
    return


@app.cell(hide_code=True)
def predict_bases():
    guess_bases = L.guess(
        "bases",
        question="Onder de AVG heb je voor het verwerken van persoonsgegevens een grondslag nodig: een reden die de wet aanvaardt. Toestemming van de persoon is er een. Hoeveel zijn er in totaal?",
        options={
            "two": "Twee",
            "four": "Vier",
            "right": "Zes",
        },
        reveal="**Zes**, in artikel 6, lid 1: toestemming, een overeenkomst, een wettelijke verplichting, een vitaal belang, een taak van algemeen belang, en een gerechtvaardigd belang. Het Europees Comité voor gegevensbescherming (EDPB) schrijft dat er geen rangorde is tussen de grondslagen: toestemming is er een van de zes, niet de standaard.",
    )
    guess_bases
    return (guess_bases,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De algemene verordening gegevensbescherming, de AVG, is een wet van de Europese Unie: Verordening (EU) 2016/679. Ze geldt vanaf 25 mei 2018, is volledig bindend en is rechtstreeks van toepassing in alle lidstaten: ze geldt in elk land van de EU zoals ze er staat, zonder dat elk land eerst een eigen wet maakt. In het Engels heet ze de General Data Protection Regulation, de GDPR.

        ### Persoonsgegevens, en wie verantwoordelijk is

        De AVG gaat over **persoonsgegevens**: volgens artikel 4 "alle informatie over een geïdentificeerde of identificeerbare natuurlijke persoon". Een persoon is identificeerbaar als die direct of indirect te identificeren is, bijvoorbeeld aan de hand van een naam, een identificatienummer, locatiegegevens of een online identificator. De persoon over wie de gegevens gaan, is de **betrokkene**. Verwerken is bijna alles wat je met persoonsgegevens kunt doen: de AVG noemt onder meer verzamelen, opslaan, gebruiken en wissen. Teksten verzamelen voor een dataset is verwerken, en een model erop trainen ook, als er persoonsgegevens in staan.

        De **verwerkingsverantwoordelijke** is wie volgens artikel 4 "het doel van en de middelen voor de verwerking van persoonsgegevens vaststelt": wie beslist waarom en hoe. Dat kan een bedrijf zijn, een overheidsinstantie of één persoon, alleen of samen met anderen. De verwerkingsverantwoordelijke is ervoor verantwoordelijk dat de AVG wordt nageleefd. Een organisatie die de gegevens in opdracht van de verwerkingsverantwoordelijke verwerkt, zoals een bedrijf dat volgens de instructies van een klant een model traint, is een verwerker.

        ### Zes grondslagen

        De verwerking is volgens artikel 6, lid 1 "alleen rechtmatig indien en voor zover aan ten minste een van de onderstaande voorwaarden is voldaan", en dan volgen zes voorwaarden. De Autoriteit Persoonsgegevens (AP), de Nederlandse toezichthouder, schrijft: "In de Algemene verordening gegevensbescherming (AVG) staan 6 redenen genoemd. De juridische naam voor die redenen is 'grondslagen'." De verordening zelf spreekt van een rechtsgrond; deze course zegt **grondslag**, net als de AP.
        """),
        mo.Html(C.bases_picture("nl")),
        mo.md(r"""
        De laatste grondslag heeft een grens in de wet zelf: hij geldt niet als de belangen of de grondrechten en fundamentele vrijheden van de betrokkene zwaarder wegen, vooral als de betrokkene een kind is. De EDPB, waarin de toezichthouders van de EU samenwerken, beschrijft de afweging als een toets in drie stappen: is er een gerechtvaardigd belang, is de verwerking daarvoor nodig, en houdt het stand tegenover de belangen en rechten van de mensen in de data? Het belang moet echt en actueel zijn, niet speculatief; een van de voorbeelden die de EDPB geeft van een belang dat gerechtvaardigd kan zijn, is het ontwikkelen van een conversational agent die gebruikers helpt. Welke grondslag past, moet de verwerkingsverantwoordelijke zelf bepalen, schrijft de EDPB. De AP zegt het zo: "U moet zelf beoordelen welke grondslag voor u van toepassing is. Dat is uw eigen verantwoordelijkheid."

        Twee misverstanden komen vaak voor. Je hebt niet altijd toestemming nodig: dat is één grondslag van de zes, en er is geen rangorde tussen de grondslagen. En openbare data zijn niet vrij te gebruiken: openbaar is geen grondslag. Of gegevens openbaar waren, is een van de dingen die bepalen wat mensen redelijkerwijs kunnen verwachten, en dat telt mee in de derde stap van de toets; of mensen überhaupt weten dat hun persoonsgegevens online staan, telt ook mee. Een regel in een privacyverklaring beslist het ook niet: dat iets daarin staat, betekent volgens de EDPB niet vanzelf dat mensen het redelijkerwijs kunnen verwachten.

        Sommige persoonsgegevens krijgen extra bescherming: gegevens over gezondheid, religieuze overtuiging, politieke opvattingen, seksuele gerichtheid en nog een paar andere soorten, de bijzondere persoonsgegevens. Volgens artikel 9, lid 1 is het verwerken daarvan verboden, tenzij een van de uitzonderingen in artikel 9, lid 2 van toepassing is.

        ### Doelbinding

        Persoonsgegevens moeten volgens artikel 5, lid 1, punt b "voor welbepaalde, uitdrukkelijk omschreven en gerechtvaardigde doeleinden worden verzameld en mogen vervolgens niet verder op een met die doeleinden onverenigbare wijze worden verwerkt". Dat is **doelbinding**. Een nieuw doel is niet vanzelf verboden, maar het mag niet onverenigbaar zijn met het eerste. Om dat te beoordelen noemt artikel 6, lid 4 waar je rekening mee houdt, onder meer de context waarin de persoonsgegevens verzameld zijn en passende waarborgen, zoals versleuteling of pseudonimisering. Een model trainen op chats die verzameld zijn om vragen te beantwoorden, is zo'n verder doel: wat Solove hergebruik voor een ander doel noemt.

        ### Wissen en bezwaar

        De AVG geeft betrokkenen rechten over gegevens over henzelf. Twee daarvan doen er voor AI veel toe. Het **recht op gegevenswissing** (artikel 17, dat er "recht op vergetelheid" tussen haakjes bij zet) houdt in dat iemand persoonsgegevens over zichzelf zonder onredelijke vertraging kan laten wissen, maar alleen op een van zes gronden: bijvoorbeeld als die persoon de toestemming intrekt waarop de verwerking rustte en er geen andere grondslag voor is, of als de gegevens onrechtmatig verwerkt zijn. En er zijn uitzonderingen, onder meer de vrijheid van meningsuiting en informatie, en archivering in het algemeen belang, wetenschappelijk of historisch onderzoek en statistiek, voor zover wissen de doelen van die verwerking waarschijnlijk onmogelijk maakt of ernstig in gevaar brengt. Met het recht van bezwaar (artikel 21) kan iemand, om redenen die met de eigen situatie te maken hebben, bezwaar maken tegen verwerking op grond van een taak van algemeen belang of een gerechtvaardigd belang. De verwerkingsverantwoordelijke moet dan stoppen, tenzij die dwingende gerechtvaardigde gronden voor de verwerking aantoont die zwaarder wegen dan de belangen, rechten en vrijheden van de persoon, of de gegevens nodig heeft voor een rechtszaak.

        ### De AVG en AI

        Op 17 december 2024 nam de EDPB, op verzoek van de Ierse toezichthouder, Advies 28/2024 over AI-modellen aan. Het is een advies, geen uitspraak van een rechter: de toezichthouders moeten er bij hun onderzoek zo veel mogelijk rekening mee houden. Een paar dingen die erin staan:

        - AI-modellen die met persoonsgegevens getraind zijn, kun je niet in alle gevallen als anoniem beschouwen. Een model geldt pas als anoniem als de kans verwaarloosbaar is dat je de persoonsgegevens waarop het getraind is er weer uit krijgt, door ze te extraheren of door het model vragen te stellen.
        - Informatie die gescrapet is van bijvoorbeeld nieuwssites, sociale media, forums en persoonlijke websites, kan persoonsgegevens bevatten. De Autoriteit Persoonsgegevens schrijft: "Met scraping worden bijna altijd persoonsgegevens verzameld. Daardoor ontstaan privacyrisico's."
        - Voorbeelden van maatregelen die de gevolgen voor mensen beperken: vanaf het begin een onvoorwaardelijke opt-out aanbieden, mensen hun gegevens laten wissen ook als geen van de zes gronden van artikel 17, lid 1 geldt, en bij scraping robots.txt- of ai.txt-bestanden respecteren.
        - Als een model onrechtmatig getraind is, kan een toezichthouder bevelen om een deel van de dataset te wissen of, als dat niet kan, om de hele dataset waarmee het AI-model ontwikkeld is en/of het AI-model zelf te wissen.

        Nationale toezichthouders treden ook zelf op. Op 30 maart 2023 beval de Italiaanse toezichthouder, de Garante, OpenAI, de maker van ChatGPT, om de verwerking van de gegevens van mensen in Italië tijdelijk te beperken; een van de redenen was dat er geen geschikte grondslag was voor het verzamelen van persoonsgegevens en het verwerken ervan om de algoritmes achter ChatGPT te trainen. Op 28 april 2023 maakte OpenAI ChatGPT na aanpassingen weer beschikbaar in Italië; een van de aanpassingen was dat iedereen in Europa, ook wie ChatGPT niet gebruikt, zich kon afmelden voor het gebruik van de eigen gegevens voor training. Op 20 december 2024 maakte de Garante een boete van 15 miljoen euro bekend, voor meerdere bevindingen: onder meer dat OpenAI persoonsgegevens van gebruikers verwerkte om ChatGPT te trainen zonder eerst een passende grondslag vast te stellen, een datalek van maart 2023 niet bij de toezichthouder meldde, en geen leeftijdscontrole had. In 2026 verklaarde de rechtbank van Rome het beroep tegen dat besluit gegrond, in een vonnis dat op 18 maart 2026 gepubliceerd is: wie in beroep ging, kreeg gelijk. De course heeft het vonnis niet gelezen, en weet dus niet op welke gronden, en of het vonnis definitief is. De AVG staat boetes toe tot 20 miljoen euro of, voor een onderneming, 4% van de wereldwijde jaaromzet van het voorgaande boekjaar, als dat hoger is.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **AVG** | de Europese wet die persoonsgegevens beschermt: wanneer organisaties ze mogen verwerken, en welke rechten mensen hebben over gegevens over henzelf |
        | **persoonsgegevens** | alle informatie over een persoon die geïdentificeerd is of direct of indirect te identificeren is |
        | **betrokkene** | de persoon over wie de persoonsgegevens gaan |
        | **verwerkingsverantwoordelijke** | wie beslist waarom en hoe persoonsgegevens worden verwerkt, en daarvoor onder de AVG verantwoordelijk is |
        | **grondslag** | een van de zes redenen in artikel 6, lid 1, zoals toestemming of een gerechtvaardigd belang: verwerken is alleen rechtmatig als er ten minste één van toepassing is |
        | **doelbinding** | de regel dat persoonsgegevens voor een duidelijk doel verzameld worden en later niet gebruikt worden op een manier die daarmee onverenigbaar is |
        | **recht op gegevenswissing** | het recht van iemand om persoonsgegevens over zichzelf te laten wissen, op een van zes gronden en met uitzonderingen (artikel 17) |
        """),
        L.watch_out("De AVG verbiedt niet dat je een model traint op persoonsgegevens, en staat het ook niet toe omdat de data openbaar zijn. Trainen is verwerken: het heeft een van de zes grondslagen en een doel nodig, en bij gezondheidsgegevens en de andere bijzondere persoonsgegevens ook een uitzondering uit artikel 9, lid 2."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan wat een AI-systeem met gegevens over mensen mag doen:

    1. Vraag of de gegevens gaan over een persoon die direct of indirect te identificeren is. Dan zijn het persoonsgegevens, waar de AVG over gaat. Let op bijzondere persoonsgegevens, zoals gegevens over gezondheid.
    2. Zoek de verwerkingsverantwoordelijke: wie beslist waarom en hoe de gegevens worden verwerkt.
    3. Geef elke activiteit een van de zes grondslagen, en ga na of die klopt; bij een gerechtvaardigd belang met de toets in drie stappen. Dat is `has_basis`.
    4. Vergelijk waarvoor de gegevens gebruikt worden met waarvoor ze verzameld zijn. Een nieuw doel mag niet onverenigbaar zijn met het eerste. Dat is `new_purposes`.
    5. Als iemand om wissing vraagt en een van de gronden geldt, zoek dan alle gegevens over die persoon, ook waar anderen over die persoon schrijven, ga de uitzonderingen na, en denk aan de modellen die er al op getraind zijn. Dat is `erase`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier velden van een dataset die de course verzonnen heeft, elk met wat het over een persoon zegt. `_special` bevat vier soorten bijzondere persoonsgegevens uit artikel 9, lid 1. De cel print voor elk veld de naam, waar het over gaat, en of dat bijzondere persoonsgegevens zijn.
    """)
    return


@app.cell
def example_fields():
    # vier verzonnen velden van een verzonnen dataset, elk met wat het over een persoon zegt
    _fields = [{"field": "email", "about": "contact"},
               {"field": "diagnosis", "about": "health"},
               {"field": "church", "about": "religious beliefs"},
               {"field": "streak", "about": "use of the app"}]
    # vier soorten bijzondere persoonsgegevens uit artikel 9, lid 1
    _special = ["health", "religious beliefs", "political opinions", "sexual orientation"]

    for _row in _fields:
        print(_row["field"], "|", _row["about"], "|", _row["about"] in _special)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee van de vier velden zijn bijzondere persoonsgegevens: een diagnose gaat over gezondheid, een kerk over religieuze overtuiging. Het verwerken daarvan is verboden, tenzij een van de uitzonderingen in artikel 9, lid 2 van toepassing is. Artikel 9, lid 1 noemt er meer, zoals het lidmaatschap van een vakbond en genetische gegevens. Alle vier de velden zijn persoonsgegevens, dus voor het verwerken van elk ervan is een van de zes grondslagen nodig; de twee soorten bijzondere persoonsgegevens hebben ook een uitzondering nodig. `in` geeft `True` of `False`; in oefening 1 bewaar je zo'n antwoord voor elke activiteit in een dictionary.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_basis():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. has_basis: heeft elke activiteit een grondslag?

        Een verzonnen taalleerapp houdt een register bij van wat de app met persoonsgegevens doet, met de grondslag die het register voor elke activiteit noemt. `REGISTER` is een lijst met dictionaries, elk met `"activity"` (de activiteit) en `"basis"` (de grondslag), zoals ze in het Engels in het register staan:

        | activity | basis |
        |---|---|
        | send lesson reminders | consent |
        | bill subscriptions | contract |
        | keep invoices for the tax office | legal obligation |
        | detect fraud in payments | legitimate interests |
        | train a model on posts scraped from other sites | public data |
        | share progress with advertisers | none |
        | train the tutor bot on users' chats | unknown |

        Dat is: lesherinneringen sturen (toestemming), abonnementen factureren (overeenkomst), facturen bewaren voor de Belastingdienst (wettelijke verplichting), fraude bij betalingen opsporen (gerechtvaardigd belang), een model trainen op berichten die van andere sites gescrapet zijn (*openbare data*), voortgang delen met adverteerders (*geen*) en de tutorbot trainen op de chats van gebruikers (*onbekend*). `BASES` is de lijst met de zes grondslagen zoals de course ze in het Engels schrijft: `["consent", "contract", "legal obligation", "vital interests", "public task", "legitimate interests"]`. `has_basis(register, bases)` krijgt zo'n lijst `register` en een lijst `bases`, en geeft een dictionary terug met voor elke activiteit `True` als de grondslag in `bases` staat, en `False` als dat niet zo is. `result` begint als lege dictionary, en de `for`-loop gaat elke `row` langs. Vervang de `...` door één regel die in `result`, onder `row["activity"]`, bewaart of `row["basis"]` in `bases` staat. De functie geeft `result` terug. De check roept `has_basis(REGISTER, BASES)` aan, daarna met minder grondslagen, een kort register en een leeg register.
        """),
        L.task("Geef een dictionary terug met voor elke activiteit of de grondslag een van de `bases` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_basis():
    box_basis = L.code_box("basis", lines=6)
    box_basis
    return (box_basis,)


@app.cell(hide_code=True)
def check_basis(box_basis):
    result_basis = L.check("basis", box_basis.value)
    result_basis
    return (result_basis,)


@app.cell(hide_code=True)
def hints_basis():
    L.hints(
        "basis",
        [
            "Stap 3 van het recept. Een dictionary krijgt een waarde met `result[key] = waarde`. Welke key heb je nodig, en welke waarde?",
            "De key is de activiteit van de rij. De waarde is `True` of `False`, en een test met `in` geeft precies een van die twee.",
            "`result[row[\"activity\"]] = ___`",
        ],
        why="`result[row[\"activity\"]] = row[\"basis\"] in bases`. `has_basis(REGISTER, BASES)` geeft `True` voor de eerste vier activiteiten en `False` voor de laatste drie.",
    )
    return


@app.cell(hide_code=True)
def text_after_basis():
    mo.md(r"""
    Drie activiteiten hebben geen grondslag in het register. `"public data"` is er geen van de zes: dat de berichten openbaar waren, bepaalt mee wat mensen kunnen verwachten, en is op zich geen grondslag. `"none"` en `"unknown"` zeggen het tenminste openlijk, en betekenen toch dat de app het niet mag doen zolang er geen grondslag is vastgesteld. Een `True` is ook niet het eind: je functie kijkt alleen of de naam van een grondslag is opgeschreven, niet of die klopt. Fraude opsporen rust op een gerechtvaardigd belang, dus de app moet nog door de toets in drie stappen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_purpose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. new_purposes: voor iets anders gebruikt?

        Dezelfde app houdt ook bij waarvoor elk soort gegevens verzameld is en waarvoor het nu gebruikt wordt. `USES` is een lijst met dictionaries, elk met `"data"` (welke gegevens), `"collected_for"` (verzameld voor) en `"used_for"` (gebruikt voor), in het Engels:

        | data | collected for | used for |
        |---|---|---|
        | email addresses | lesson reminders | lesson reminders |
        | voice recordings | pronunciation feedback | training a speech model |
        | lesson scores | showing progress | showing progress |
        | chats with the tutor bot | answering questions | training the tutor bot |
        | payment details | billing | detecting fraud |

        Dat is: e-mailadressen voor lesherinneringen; spraakopnames, verzameld voor feedback op de uitspraak en nu gebruikt om een spraakmodel te trainen; lesscores om voortgang te tonen; chats met de tutorbot, verzameld om vragen te beantwoorden en nu gebruikt om de tutorbot te trainen; en betaalgegevens, verzameld om te factureren en nu gebruikt om fraude op te sporen. `new_purposes(uses)` krijgt zo'n lijst `uses`, en geeft een lijst terug met de gegevens waarvan `"used_for"` verschilt van `"collected_for"`. `found` begint als lege lijst, en de `for`-loop gaat elke `use` langs. Vervang de `...` door twee regels: als `use["used_for"]` verschilt van `use["collected_for"]`, voeg dan `use["data"]` toe aan `found`. De functie geeft `found` terug. De check roept `new_purposes(USES)` aan, daarna een korte lijst en een lege lijst.
        """),
        L.task("Geef een lijst terug met de gegevens die gebruikt worden voor een ander doel dan waarvoor ze verzameld zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_purpose():
    box_purpose = L.code_box("purpose", lines=7)
    box_purpose
    return (box_purpose,)


@app.cell(hide_code=True)
def check_purpose(box_purpose):
    result_purpose = L.check("purpose", box_purpose.value)
    result_purpose
    return (result_purpose,)


@app.cell(hide_code=True)
def hints_purpose():
    L.hints(
        "purpose",
        [
            "Stap 4 van het recept. Vergelijk twee waarden van dezelfde rij: waarvoor de gegevens verzameld zijn, en waarvoor ze gebruikt worden.",
            "Een `if` die vraagt of de twee verschillen, en daaronder `found.append(...)`. Moet de lijst de hele rij bevatten, of om welke gegevens het gaat?",
            "`if use[\"used_for\"] != use[\"collected_for\"]:` en daaronder `found.append(___)`",
        ],
        why="`if use[\"used_for\"] != use[\"collected_for\"]:` en daaronder `found.append(use[\"data\"])`. `new_purposes(USES)` geeft `[\"voice recordings\", \"chats with the tutor bot\", \"payment details\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_purpose():
    mo.md(r"""
    Drie soorten gegevens hebben een nieuw doel. Daarmee is het nieuwe gebruik niet verboden: het mag alleen niet onverenigbaar zijn met het doel waarvoor de gegevens verzameld zijn. Dat kan je functie niet beoordelen. Twee teksten vergelijken weegt niet de context waarin de gegevens verzameld zijn, of de waarborgen, zoals artikel 6, lid 4 vraagt; het laat zien waar iemand moet kijken. De chats met de tutorbot zijn het geval uit het register: verzameld om vragen te beantwoorden, nu gebruikt om te trainen, en zonder grondslag in het register.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_controller():
    mo.md(r"""
    ### Oefening 3. Wie beslist?
    """)
    return


@app.cell(hide_code=True)
def choice_controller_cell():
    choice_controller = L.choice(
        "controller",
        question="Een ziekenhuis wil een model dat de brieven sorteert die het van patiënten krijgt. Het ziekenhuis beslist waar het model voor is en hoe het gemaakt wordt, en betaalt een bedrijf om het op de brieven te trainen, volgens de instructies van het ziekenhuis. Wie is de verwerkingsverantwoordelijke?",
        options={
            "right": "Het ziekenhuis: dat beslist waarom en hoe",
            "company": "Het bedrijf: dat doet de training",
            "patients": "De patiënten: de brieven gaan over henzelf",
        },
        correct="right",
        why={
            "right": "De verwerkingsverantwoordelijke is wie \"het doel van en de middelen voor de verwerking van persoonsgegevens vaststelt\". Dat is het ziekenhuis; het bedrijf verwerkt de gegevens in opdracht van het ziekenhuis, en is een verwerker. Brieven over de gezondheid van patiënten bevatten ook bijzondere persoonsgegevens: die verwerken is verboden, tenzij een van de uitzonderingen in artikel 9, lid 2 van toepassing is.",
            "company": "Het werk doen maakt het bedrijf niet de verwerkingsverantwoordelijke: het traint het model in opdracht van het ziekenhuis, volgens de instructies van het ziekenhuis, en is daarmee een verwerker. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee*.",
            "patients": "De patiënten zijn de betrokkenen: de mensen over wie de gegevens gaan. De verwerkingsverantwoordelijke is wie beslist waarom en hoe de gegevens worden verwerkt. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee*.",
        },
    )
    choice_controller
    return (choice_controller,)


@app.cell(hide_code=True)
def text_exercise_erase():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. erase: een persoon uit de dataset halen

        De app heeft een forum, en traint een model op de berichten. `POSTS` is een lijst met zeven verzonnen berichten, elk een dictionary met `"id"`, `"author"` (wie het schreef), `"mentions"` (een lijst met de mensen die het bericht noemt) en `"text"`, in het Engels:

        | id | author | mentions | text |
        |---|---|---|---|
        | 1 | sam | | Does anyone know a good podcast in Spanish? |
        | 2 | noor | sam | Try the one about cooking, sam. |
        | 3 | kai | | Day 30 of my streak! |
        | 4 | sam | kai | Well done, kai! |
        | 5 | ash | | I keep mixing up ser and estar. |
        | 6 | lior | sam, ash | sam and ash, shall we practice on Friday? |
        | 7 | remi | noor | Thanks for the tip, noor. |

        Stel dat de app op de berichten traint met toestemming van de schrijvers, en dat sam die toestemming intrekt. Dan kan sam om wissing vragen: ingetrokken toestemming, zonder andere grondslag, is een van de zes gronden van artikel 17, lid 1. Een bericht van iemand anders dat sam noemt, is ook informatie over sam, maar de toestemming van sam gold er nooit voor, en het zijn ook de woorden van de schrijver. De app kiest er hier voor om verder te gaan, en haalt elk bericht van sam en elk bericht dat sam noemt eruit. `erase(posts, name)` krijgt zo'n lijst `posts` en een naam `name`, en geeft een lijst terug met de berichten die niet van `name` zijn en `name` niet noemen. `kept` begint als lege lijst, en de `for`-loop gaat elke `post` langs. Vervang de `...` door twee regels: als `post["author"]` niet `name` is en `name` niet in `post["mentions"]` staat, voeg dan `post` toe aan `kept`. De functie geeft `kept` terug. De check roept `erase(POSTS, "sam")` aan, daarna andere namen, een naam zonder berichten, en een korte lijst.
        """),
        L.task("Geef een lijst terug met de berichten die niet van `name` zijn en `name` niet noemen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_erase():
    box_erase = L.code_box("erase", lines=7)
    box_erase
    return (box_erase,)


@app.cell(hide_code=True)
def check_erase(box_erase):
    result_erase = L.check("erase", box_erase.value)
    result_erase
    return (result_erase,)


@app.cell(hide_code=True)
def hints_erase():
    L.hints(
        "erase",
        [
            "Stap 5 van het recept. Een bericht blijft alleen als twee dingen tegelijk waar zijn: het is niet van `name`, en het noemt `name` niet.",
            "Twee voorwaarden in één `if`, verbonden met `and`. Hoe vraag je of een naam niet in een lijst staat?",
            "`if post[\"author\"] != name and ___:`",
        ],
        why="`if post[\"author\"] != name and name not in post[\"mentions\"]:` en daaronder `kept.append(post)`. `erase(POSTS, \"sam\")` bewaart de berichten met id 3, 5 en 7.",
    )
    return


@app.cell(hide_code=True)
def text_after_erase():
    mo.md(r"""
    Vier van de zeven berichten zijn weg: twee van sam, en twee die sam noemen. De dataset bevat ze niet meer, maar een model dat al op de oude dataset getraind is, verandert niet als er rijen uit de dataset verdwijnen. De EDPB schrijft dat modellen die op persoonsgegevens getraind zijn niet in alle gevallen als anoniem te beschouwen zijn: een model kan soms teruggeven waarop het getraind is, het onderwerp van *Wat een model onthoudt*, later in deze module. En als een model onrechtmatig getraind is, kan een toezichthouder bevelen om het model zelf te wissen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_blog():
    mo.md(r"""
    ### Oefening 5. Een blogpost
    """)
    return


@app.cell(hide_code=True)
def choice_blog_cell():
    choice_blog = L.choice(
        "blog",
        question="Een blog schrijft: \"In 2024 deed de EDPB de uitspraak dat AI trainen op gescrapete data illegaal is.\" Wat ondersteunt Advies 28/2024?",
        options={
            "exact": "De blogpost zoals hij er staat",
            "right": "Minder: een advies, geen verbod",
            "more": "Meer: de uitspraak bindt elke rechter",
        },
        correct="right",
        why={
            "exact": "De EDPB deed geen uitspraak: het gaf een advies, waar de toezichthouders zo veel mogelijk rekening mee moeten houden. En het noemt scrapen niet illegaal: het zegt dat gescrapete informatie persoonsgegevens kan bevatten, en bespreekt hoe een grondslag als gerechtvaardigd belang van toepassing kan zijn. Zie *De AVG en AI* in *Het idee*.",
            "right": "Advies 28/2024 is een advies, geen uitspraak van een rechter. Het verbiedt niet dat je traint op gescrapete data: het zegt dat zulke data persoonsgegevens kunnen bevatten, zodat het verwerken ervan een grondslag nodig heeft, en het noemt waarborgen, zoals robots.txt- of ai.txt-bestanden respecteren.",
            "more": "Uitspraken komen van rechters, niet van de EDPB. Het advies is een leidraad voor de toezichthouders, die er zo veel mogelijk rekening mee moeten houden. Zie *De AVG en AI* in *Het idee*.",
        },
    )
    choice_blog
    return (choice_blog,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) benoemde je privacyproblemen met de taxonomie van Solove. Een paar daarvan hebben een tegenhanger in de AVG: tegenover hergebruik voor een ander doel staat doelbinding, en de mensen in de data krijgen rechten zoals wissing en bezwaar. Wie beslist waarom en hoe de gegevens worden verwerkt, is verantwoordelijk, ook als er een model op getraind wordt. In de volgende les, *Informatie op de verkeerde plek*, kijk je nader naar de derde opvatting van privacy: of een stroom informatie past bij de context.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_blog, choice_controller, result_basis, result_erase, result_purpose):
    L.wrap_up(result_basis, result_purpose, choice_controller, result_erase, choice_blog)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
