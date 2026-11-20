# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-kinds-of-ai-risk. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow categorie, categorieën, risk

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-kinds-of-ai-risk", "nl")
    C = L.checks   # negen verzonnen gevallen van risico's van AI, en de zeven domeinen van de MIT AI Risk Repository


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_allocative():
    quick_allocative = L.choice(
        "quick-allocative",
        question="Een model beslist wie er aan de grens extra gecontroleerd wordt, en mensen uit één groep krijgen veel meer controles dan anderen. Welke soort schade is dat?",
        options={
            "representational": "Representatief: het zet die groep in een slecht daglicht",
            "right": "Allocatief: het gaat om wat het systeem aan mensen uitdeelt",
            "none": "Nog geen schade, want een controle is over niemand een eindbeslissing",
        },
        correct="right",
        why={
            "right": "Een extra controle is iets wat het systeem uitdeelt, en één groep krijgt er meer van: allocatieve schade. Deze les deelt risico's op een andere manier in: naar waar ze vandaan komen.",
            "representational": "Representatieve schade gaat over hoe een systeem groepen laat zien of beschrijft. Hier deelt het systeem controles uit. Zie [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "none": "Extra controles kosten tijd en vertrouwen, en ze worden ongelijk uitgedeeld: dat is allocatieve schade. Zie [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_allocative
    return (quick_allocative,)


@app.cell(hide_code=True)
def quick_check_representational():
    quick_representational = L.choice(
        "quick-representational",
        question="Hoe vind je representatieve schade in wat een model schrijft?",
        options={
            "right": "Verzamel veel uitvoer, en tel wie hoe wordt getoond",
            "yes": "Vergelijk het aandeel van elke groep dat een ja krijgt",
            "ask": "Vraag het model of zijn eigen uitvoer stereotypen bevat",
        },
        correct="right",
        why={
            "right": "Voor representatieve schade hoeft over niemand iets beslist te worden, dus je moet ernaar zoeken in veel uitvoer. Deze les vraagt welke andere soorten risico er zijn.",
            "yes": "Dat meet allocatieve schade: wat een systeem uitdeelt. Representatieve schade zit in hoe groepen getoond worden. Zie [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "ask": "Een model dat een stereotype laat zien, hoeft dat niet te merken. Je verzamelt veel uitvoer en telt wie hoe getoond wordt. Zie [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_representational
    return (quick_representational,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een chatbot die geld terug belooft dat de winkel niet geeft, een gekloonde stem van een manager waarmee iemand personeel geld laat overmaken, vertalers die minder opdrachten krijgen: alle drie zijn het risico's van AI, en elk vraagt om een ander antwoord. Het eerste los je op in het systeem. Bij het tweede moet iemand de persoon tegenhouden die het misbruikt. Het derde los je met geen van beide op, want niemand wilde schade doen en elk systeem deed wat het moest doen.

    Met deze les begint de module over AI-veiligheid. Ze deelt risico's op twee manieren in, een uit 2019 en een uit het International AI Safety Report, schrijft die indeling als functie, en laat zien waarom een lijst van soorten helpt: een lijst uit je hoofd laat dingen weg.

    **Een voorbeeld uit het dagelijks leven.** Een paklijst die is ingedeeld in kleren, papieren en medicijnen laat in één oogopslag zien dat er geen paspoort op staat. Een lijst uit je hoofd doet dat niet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Onderzoekers van MIT verzamelden 74 documenten die elk risico's van AI opsommen, en deelden de risico's die ze noemen in 24 subdomeinen in. Hoeveel van de 24 dekt één document gemiddeld?",
        options={
            "few": "Ongeveer 3",
            "right": "Ongeveer 8",
            "most": "Ongeveer 20",
        },
        reveal="**Ongeveer 8.** Volgens Slattery et al. (2026) behandelden de documenten maar een deel van de risico's: gemiddeld 8 van de 24 subdomeinen, en per document van 1 tot 20. *Waarom indelen?* in *Het idee* zegt wat daaruit volgt.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Waar AI-veiligheid over gaat

    Het International AI Safety Report, waar deze les op terugkomt, noemt in de editie van 2026 een AI-systeem veilig als het waarschijnlijk geen schade doet, niet door kwaadwillig misbruik en niet door falen. **AI-veiligheid** is het vakgebied dat daaraan werkt: voorkomen dat AI-systemen schade doen, of mensen ze nu misbruiken of ze falen. Dat gaat over meer dan één zorg: het rapport van 2025 rekent onbetrouwbare antwoorden en bias tot de risico's, naast verlies van controle; de editie van 2026 heeft een kleiner bereik, en behandelt onderwerpen als bias, milieu, privacy en auteursrecht niet meer. Amodei et al. gebruikten de naam in 2016 in engere zin: voor het beperken van het risico op ongelukken, wat volgens hen in het publieke debat vaak AI-veiligheid heette.

    Een verwant woord: **alignment** is, in de woorden van de editie van 2026, de neiging van een AI-model of -systeem om wat het kan in te zetten in lijn met wat mensen bedoelen, met hun waarden of normen. Eenvoudig gezegd: een model met goede alignment doet meestal wat mensen bedoelen. Latere lessen van deze module laten zien hoe dat misgaat.

    ### Ongelukken, misbruik en structuur

    In 2019 schreven Remco Zwetsloot en Allan Dafoe dat het in het vakgebied gangbaar was geworden om de risico's van AI in twee soorten te verdelen: misbruik en ongelukken. **Misbruik** is iemand die AI gebruikt om schade te doen: mensen die AI op een onethische manier inzetten, het duidelijkst als ze kwade bedoelingen hebben. Een **ongeluk** is schade door een systeem dat niet doet wat bedoeld was. Hun typische voorbeeld: een zelfrijdende auto die botst omdat de AI zijn omgeving verkeerd begrijpt.

    Ze voegen een derde perspectief toe. Misbruik en ongelukken kijken volgens hen meestal alleen naar de laatste stap in de keten die tot schade leidt: de persoon die de technologie misbruikte, of het systeem dat iets onbedoelds deed. Maar technologie verandert ook de omgeving en de prikkels, de structuur van een situatie. Een **structureel risico** komt voort uit die structuur, niet van één persoon of één falend systeem: in veel situaties blijft het risico ongeveer even groot, ook als één partij zich anders gaat gedragen. Net als bij een lawine is het misschien zinvoller te vragen waardoor de helling zo steil werd dan wat hem precies in gang zette. Hun voorbeelden voor AI zijn nucleaire afschrikking en cyberoperaties, en ze wijzen op wat anderen noemden: markten met een monopolie, banen die verdwijnen en privacy die afbrokkelt. De spoorwegen vóór de Eerste Wereldoorlog, waarvan historici de rol nog altijd bespreken, maken volgens hen duidelijk dat technologie grote schade kan doen, ook als niemand de technologie misbruikt en die precies werkt zoals bedoeld. De naam structureel risico is een verkorting: de auteurs spreken van een structureel perspectief, en schrijven dat structurele risico's ook collectieve risico's zijn.

    ### Het International AI Safety Report

    Het **International AI Safety Report** zet op een rij wat de wetenschap weet over wat general-purpose AI kan en welke risico's het heeft. General-purpose AI zijn modellen zoals chat models, die veel soorten taken kunnen doen. Yoshua Bengio is de voorzitter. Aan het eerste volledige rapport, van januari 2025, werkten 96 AI-experts mee, onder wie een adviespanel met leden die 30 landen, de OESO, de EU en de VN voordroegen. Dat rapport zegt van zichzelf dat het geen aanbevelingen doet voor bepaald beleid. De editie van februari 2026 kwam tot stand met hulp van meer dan 100 experts. Beide delen de risico's in drie categorieën in: kwaadwillig gebruik, falen en systeemrisico's.

    - Kwaadwillig gebruik, in de woordenlijst van 2025: AI inzetten om met opzet schade te doen.
    - Falen (*malfunction*), in dezelfde woordenlijst: een general-purpose AI-systeem dat niet werkt zoals de ontwikkelaar of de gebruiker bedoelde, met verkeerde of schadelijke uitvoer of verstoringen als gevolg.
    - Systeemrisico's, in de editie van 2026: risico's die ontstaan doordat het ontwikkelen en inzetten van AI het gedrag van mensen, de manier van werken van organisaties of de structuren van de samenleving verandert, en niet rechtstreeks door wat AI kan.

    Ze lijken op misbruik, ongelukken en structuur, maar ze zijn niet hetzelfde. Voor Zwetsloot en Dafoe is misbruik elk onethisch gebruik, met kwade bedoelingen als het duidelijkste geval; kwaadwillig gebruik is in het rapport schade met opzet. En structuur is voor hen minder een derde vakje dan een perspectief dat de andere twee aanvult. Het rapport van 2025 rekent bias tot falen, naast onbetrouwbaarheid en verlies van controle. En de editie van 2026 waarschuwt dat de categorieën niet alles omvatten en elkaar niet uitsluiten: een risico kan in meer categorieën vallen. Dat een risico in de lijst staat, betekent niet per se dat het waarschijnlijk of ernstig is, of dat er beleid voor nodig is.
    """)
    return


@app.cell(hide_code=True)
def report_cell():
    mo.Html(C.report_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Verlies van controle

    **Verlies van controle** is volgens de woordenlijst van 2025 een scenario waarin een of meer general-purpose AI-systemen buiten ieders controle gaan werken, zonder duidelijke weg om die controle terug te krijgen. Dat hoeft niet spectaculair te zijn. Bij passief verlies van controle worden belangrijke beslissingen aan AI-systemen overgelaten, maar zijn die beslissingen te ondoorzichtig, te ingewikkeld of te snel om er nog echt toezicht op te houden. Hoe waarschijnlijk is het? Volgens hetzelfde rapport lopen de meningen van experts over de kans op verlies van controle in de komende jaren sterk uiteen: sommigen vinden het onwaarschijnlijk, sommigen verwachten het, en sommigen zien het als een risico met een bescheiden kans dat aandacht verdient omdat de gevolgen zo ernstig kunnen zijn. Over één ding zijn ze het grotendeels eens: de general-purpose AI van nu kan dit risico nog niet veroorzaken.

    ### Waarom indelen?

    De MIT AI Risk Repository van Slattery et al. verzamelt wat anderen schreven: een database van 1725 risico's uit 74 bestaande indelingen en kaders, in de versie van maart 2026. Een taxonomie is een indeling in soorten. De repository deelt de risico's twee keer in. De *Causal Taxonomy* vraagt wie het risico veroorzaakt (een mens of AI), of het met opzet gebeurt, en wanneer (voor of na het in gebruik nemen): 42% van de risico's wordt toegeschreven aan AI-systemen en 38% aan beslissingen van mensen, en 35% gebeurt met opzet en 35% zonder. De *Domain Taxonomy* heeft zeven domeinen en 24 subdomeinen, van discriminatie en toxiciteit tot de veiligheid van AI-systemen zelf. En geen document dekt alle 24 subdomeinen: één document dekt er gemiddeld 8, van 1 tot 20. Daar dient een taxonomie voor: een lijst van soorten laat zien wat een lijst uit je hoofd weglaat.
    """)
    return


@app.cell(hide_code=True)
def mit_cell():
    mo.Html(C.mit_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **AI-veiligheid** | het vakgebied dat werkt aan voorkomen dat AI-systemen schade doen, of mensen ze nu misbruiken of ze falen |
        | **alignment** | hoe sterk een AI-systeem geneigd is om wat het kan in te zetten in lijn met wat mensen bedoelen |
        | **misbruik** | iemand die AI gebruikt om schade te doen, het duidelijkst met opzet |
        | **ongeluk** | schade door een AI-systeem dat niet doet wat bedoeld was |
        | **structureel risico** | risico door hoe AI de omgeving en de prikkels verandert, niet door één persoon of één falend systeem |
        | **International AI Safety Report** | een overzicht, met Yoshua Bengio als voorzitter, van wat de wetenschap weet over wat general-purpose AI kan en welke risico's het heeft |
        | kwaadwillig gebruik, falen, systeemrisico | de drie categorieën van het rapport |
        | **verlies van controle** | AI-systemen die buiten ieders controle werken, zonder duidelijke weg om die terug te krijgen |
        | `C.RISKS` | negen verzonnen gevallen: dicts met `"text"`, `"harm_meant"` en `"worked"` |
        | `C.DOMAINS`, `C.CHECKLIST` | de zeven domeinen van de MIT-repository, en de drie die een verzonnen team opschreef |
        | `elif` | anders als: Python bekijkt deze voorwaarde alleen als de `if` ervoor niet waar was |
        | `x not in a_list` | `True` als de lijst `x` niet bevat |
        """),
        L.watch_out("Een risico indelen zegt wat voor soort het is, niet hoe waarschijnlijk of hoe ernstig het is. Het rapport zegt zelf dat een risico in zijn lijst niet per se waarschijnlijk of ernstig is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        1. Vraag of iemand met AI schade wilde doen. Zo ja, dan is het kwaadwillig gebruik, het dichtst bij wat Zwetsloot en Dafoe misbruik noemen.
        2. Zo nee, vraag of het systeem deed wat de ontwikkelaar of de gebruiker bedoelde. Zo nee, dan is het falen, het dichtst bij een ongeluk.
        3. Deed het dat wel, en komt er toch schade van hoe AI in de samenleving gebruikt wordt, dan is het een systeemrisico, het dichtst bij structuur.
        4. Onthoud dat één risico in meer dan één categorie kan horen.
        5. Leg een lijst van risico's naast een taxonomie, om te zien wat erop ontbreekt.
        """),
        mo.Html(C.tree_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt het tweede geval van de course en print de tekst en de twee antwoorden. `C.RISKS[1]` is een dict; `case["worked"]` zoekt de waarde op onder de key `"worked"`. De tekst van een geval is Engels; onder de uitvoer staat hij in het Nederlands.
    """)
    return


@app.cell
def example_case():
    case = C.RISKS[1]   # het tweede geval
    print(case["text"])
    print("harm meant:", case["harm_meant"])
    print("worked as intended:", case["worked"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Een chatbot op de website van een fietsenwinkel belooft geld terug dat de winkel niet geeft. Niemand wilde schade doen, en de chatbot deed niet wat de winkel bedoelde: volgens het recept falen. Het rapport zou het onbetrouwbaarheid noemen, Zwetsloot en Dafoe een ongeluk. De oplossing zit in het systeem: een chatbot die alleen de echte regels van de winkel noemt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_category():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. iasr_category: drie categorieën

        `C.RISKS` bevat negen verzonnen gevallen, elk een dict met de keys `"text"`, `"harm_meant"` (wilde iemand schade doen?) en `"worked"` (deed het systeem wat de ontwikkelaar of de gebruiker bedoelde?). Schrijf `iasr_category(case)`: de categorie van de dict `case` van de drie van het rapport, als een van de strings `"malicious use"` (kwaadwillig gebruik), `"malfunction"` (falen) of `"systemic"` (systeemrisico). Stel de vragen van het recept in hun volgorde, met `if` en daarna `elif`, dat Python alleen bekijkt als de `if` ervoor niet waar was. Vervang de `...`. De check roept `iasr_category(RISKS[0])` aan, daarna de andere gevallen en een eigen geval.
        """),
        L.task("Vraag of iemand schade wilde doen, en anders of het systeem deed wat bedoeld was; geef de categorie terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_category():
    box_category = L.code_box("iasr-category", lines=6)
    box_category
    return (box_category,)


@app.cell(hide_code=True)
def check_category(box_category):
    result_category = L.check("iasr-category", box_category.value)
    result_category
    return (result_category,)


@app.cell(hide_code=True)
def hints_category():
    L.hints(
        "iasr-category",
        [
            "*Recept* geeft de twee vragen en hun volgorde.",
            "De eerste vraag geeft de eerste categorie terug als iemand schade wilde doen; de tweede vraag kijkt daarna of het systeem deed wat bedoeld was; de laatste regel geeft de categorie die overblijft.",
            "`    elif ___:`",
        ],
        why="`if case[\"harm_meant\"]:` met `return \"malicious use\"` eronder, dan `elif case[\"worked\"]:` met `return \"systemic\"`, en als laatste `return \"malfunction\"`. `iasr_category(RISKS[0])` geeft `\"malicious use\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_category():
    mo.md(r"""
    Van elk drie. De valse reviews, de gekloonde stem en de berichten voor een verkiezing zijn kwaadwillig gebruik, hoewel elk model precies deed wat gevraagd werd: misbruik heeft geen kapot systeem nodig. In het rapport van 2025 is het model dat vrouwen lager zet falen, en het stroomverbruik van datacenters een systeemrisico, hoewel geen enkel systeem iets verkeerd doet.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: volg een geval

    Kies een geval, en zie welke weg het neemt langs de twee vragen: de categorie in het rapport, en wat er bij Zwetsloot en Dafoe het dichtst bij komt.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_case = mo.ui.dropdown(options=C.case_options(L.lang), value=list(C.case_options(L.lang))[0], label="Een geval")
    demo_case
    return (demo_case,)


@app.cell(hide_code=True)
def demo_look(demo_case):
    mo.Html(C.demo_picture(demo_case.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. missing: wat een lijst weglaat

        De zeven domeinen van de MIT-repository staan in `C.DOMAINS`. Een verzonnen team dat een chatbot bouwt voor een fietsenwinkel, schreef risico's op onder drie ervan, in `C.CHECKLIST`. Schrijf `missing(covered, domains)`: de domeinen van de lijst `domains`, in hun volgorde, die niet in de lijst `covered` staan. Verzamel ze in de lijst `left`, met een `for`-loop die elk domein als `domain` neemt; `x not in a_list` is `True` als de lijst `x` niet bevat. Vervang de `...`. De check roept `missing(CHECKLIST, DOMAINS)` aan, en daarna eigen lijsten.
        """),
        L.task("Ga de domeinen langs, en houd elk domein dat niet in de lijst covered staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=6)
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
            "*Waarom indelen?* in *Het idee* zegt waar een taxonomie voor dient: hier leg je een lijst ernaast.",
            "`left` begint als een lege lijst; in de loop beslist een `if` of het domein eraan wordt toegevoegd; na de loop geeft de functie `left` terug.",
            "`        if ___:`",
        ],
        why="`left = []`, `for domain in domains:`, dan `if domain not in covered:` met `left.append(domain)` eronder, en `return left`. `missing(CHECKLIST, DOMAINS)` geeft de vier andere domeinen, van `\"Malicious actors & misuse\"` tot `\"AI system safety, failures & limitations\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_missing():
    mo.md(r"""
    Het team dacht aan wat de chatbot mensen laat zien en wat hij over hen weet, en liet misbruik door anderen weg, hoe mensen met de chatbot omgaan, de gevolgen voor banen en het milieu, en het falen van het systeem zelf. Ook documenten laten dingen weg: in 74 documenten vonden de auteurs van MIT dat één document gemiddeld 8 van de 24 subdomeinen dekt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_uber():
    mo.md(r"""
    ### Oefening 3. Een rem die uit stond

    In 2018 reed een zelfrijdende testauto van Uber een voetganger dood. Commentatoren gaven eerst het zicht van de auto de schuld. Maar volgens Zwetsloot en Dafoe had de auto het slachtoffer op tijd gezien om met de noodrem een botsing te voorkomen. Het probleem: technici hadden de noodrem met opzet uitgezet, uit angst dat een te gevoelige rem hun auto slecht zou laten uitkomen tegenover die van concurrenten.
    """)
    return


@app.cell(hide_code=True)
def choice_uber_cell():
    choice_uber = L.choice(
        "uber",
        question="Wat voegt het structurele perspectief aan deze botsing toe?",
        options={
            "accident": "Niets: de auto deed niet wat bedoeld was, dus het blijft een ongeluk",
            "right": "Concurrentie met andere bedrijven liet het uitzetten van de rem verstandig lijken",
            "misuse": "Het maakt de botsing alsnog tot misbruik, want technici zetten de noodrem met opzet uit",
        },
        correct="right",
        why={
            "right": "Niemand wilde schade doen, maar concurrentie dreef de technici ertoe de rem uit te zetten: de helling was al steil voordat iets de lawine in gang zette. Zwetsloot en Dafoe vatten het zo samen: veiligheid is niet alleen een technisch probleem, maar ook een politiek probleem.",
            "accident": "Dat is de laatste stap van de keten. Het structurele perspectief vraagt wat eraan voorafging: technici die vreesden dat hun auto slecht zou uitkomen tegenover concurrenten. Zie *Ongelukken, misbruik en structuur* in *Het idee*.",
            "misuse": "De rem werd met opzet uitgezet, maar niemand wilde schade doen; de duidelijkste gevallen van misbruik gaan over kwade bedoelingen. Zie *Ongelukken, misbruik en structuur* in *Het idee*.",
        },
    )
    choice_uber
    return (choice_uber,)


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ### Oefening 4. Eén fout, veel plekken

    Veel apps bouwen op de modellen van een paar aanbieders. Stel dat een van die modellen een fout heeft, en dat elke app die erop bouwt op hetzelfde moment op dezelfde manier faalt.
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="Waar hoort dat bij de drie categorieën van het rapport?",
        options={
            "malfunction": "Alleen bij falen: het model doet niet wat bedoeld was",
            "right": "Bij twee: falen dat uitgroeit tot een systeemrisico",
            "misuse": "Bij kwaadwillig gebruik, zodra iemand de fout vindt en misbruikt",
        },
        correct="right",
        why={
            "right": "De fout is falen, en dat zoveel apps tegelijk falen, komt door marktconcentratie: het rapport van 2025 rekent kwetsbare knooppunten waar veel van afhangt tot de systeemrisico's, en de editie van 2026 waarschuwt dat de categorieën niet alles omvatten en elkaar niet uitsluiten.",
            "malfunction": "Dat klopt voor de fout, maar niet voor waarom zoveel apps tegelijk falen: het rapport van 2025 rekent kwetsbare knooppunten waar veel van afhangt tot de systeemrisico's. Zie *Het International AI Safety Report* in *Het idee*.",
            "misuse": "In dit geval misbruikt niemand de fout, en wil niemand schade doen. Het is falen, en omdat zoveel apps van één model afhangen, ook een systeemrisico.",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) deelde schade in naar wat ze mensen aandoet: wat een systeem uitdeelt, en hoe het mensen laat zien. Deze les deelde risico's in naar waar ze vandaan komen: iemand die schade wil doen, een systeem dat faalt, of een structuur waarin niemand iets verkeerd doet. Het International AI Safety Report noemt ze kwaadwillig gebruik, falen en systeemrisico's; Zwetsloot en Dafoe spreken van misbruik, ongelukken en structuur. Een taxonomie zoals die van MIT laat zien wat een lijst uit je hoofd weglaat. De volgende les, *Concrete problemen in AI-veiligheid*, bekijkt vijf problemen van systemen die leren, benoemd in 2016.

    **Een voorbeeld uit het dagelijks leven.** Bij een controle loopt een huisarts elke keer dezelfde lijst vragen langs: niet omdat elk probleem waarschijnlijk is, maar zodat er geen wordt overgeslagen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overlap, choice_uber, result_category, result_missing):
    L.wrap_up(result_category, result_missing, choice_uber, choice_overlap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
