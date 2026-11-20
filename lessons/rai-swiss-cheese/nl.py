# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-swiss-cheese. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Accidents

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-swiss-cheese", "nl")
    C = L.checks   # gaten op één lijn, een foutenboom met EN en OF, en safeguards achter elkaar met een gedeelde truc


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Ontwerpen voor veiligheid](/course/open/rai-safe-design-principles?lang=nl), en een over [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_copies():
    quick_copies = L.choice(
        "quick-copies",
        question="Drie kopieën van hetzelfde model controleren elke factuur en stemmen. Waarom beschermt dat weinig tegen een fout van het model?",
        options={
            "five": "Een stemming werkt pas met minstens vijf stemmers",
            "right": "De kopieën maken dezelfde fout bij dezelfde factuur",
            "reverse": "Stemmen helpt alleen als de modellen vaker fout dan goed zitten",
        },
        correct="right",
        why={
            "right": "Kopieën hebben dezelfde zwakke plekken, dus als er één fout zit, zitten ze alle drie fout: de stemming overstemt niets. Redundantie helpt als de onderdelen niet om dezelfde reden falen.",
            "five": "Voor een stemming zijn er minstens drie nodig: twee overstemmen een derde. Het probleem is dat kopieën dezelfde fouten maken. Zie [Ontwerpen voor veiligheid](/course/open/rai-safe-design-principles?lang=nl).",
            "reverse": "Het is andersom: stemmen helpt alleen als elke stemmer vaker goed dan fout zit. En kopieën helpen zelfs dan niet, omdat ze dezelfde fouten maken. Zie [Ontwerpen voor veiligheid](/course/open/rai-safe-design-principles?lang=nl).",
        },
    )
    quick_copies
    return (quick_copies,)


@app.cell(hide_code=True)
def quick_check_given():
    quick_given = L.choice(
        "quick-given",
        question="Van 1000 schadelijke verzoeken mist een eerste safeguard er 100. Een tweede safeguard mist 30 van die 100. Wat is P(tweede mist | eerste mist)?",
        options={
            "right": "0,3: 30 van de 100",
            "all": "0,03: 30 van alle 1000 verzoeken",
            "first": "0,1: de 100 van de eerste safeguard op 1000",
        },
        correct="right",
        why={
            "right": "Gegeven dat de eerste safeguard miste, tel je alleen onder de 100 verzoeken die hij miste: 30 van 100 is 0,3.",
            "all": "De voorwaarde bepaalt onder welke gevallen je telt: alleen de 100 verzoeken die de eerste safeguard miste. 30 van die 100 is 0,3. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "first": "0,1 is de kans dat de eerste safeguard mist, P(eerste mist). De vraag gaat over de tweede, onder de 100 die de eerste miste: 30 van 100 is 0,3. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
        },
    )
    quick_given
    return (quick_given,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzonnen bedrijf heeft een chatbot. Om verzoeken om gevaarlijke hulp tegen te houden, zet het drie safeguards achter elkaar: een filter op de binnenkomende vraag, een model dat getraind is om te weigeren, en een controle die elk antwoord leest voordat het de deur uit gaat. Geen ervan is perfect; elk mist een deel van de schadelijke verzoeken. Is een rij van drie safeguards die elk 90% tegenhouden dan net zo goed als één safeguard die 99,9% tegenhoudt? Alleen als hun missers niets met elkaar te maken hebben.

    In kerncentrales, de luchtvaart en de ruimtevaart staan al tientallen jaren safeguards achter elkaar. Deze les geeft het beeld dat de veiligheidskunde daarvoor gebruikt, het Swiss cheese model, en de manier van ingenieurs om het door te rekenen, de foutenboom, en laat zien waar allebei misgaan als safeguards om dezelfde reden missen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Drie safeguards missen elk 10% van de gewone schadelijke verzoeken, onafhankelijk van elkaar. Maar 1% van de schadelijke verzoeken gebruikt een truc die alle drie om de tuin leidt. Hoeveel van de 1.000.000 schadelijke verzoeken komen ongeveer door alle drie?",
        options={
            "independent": "Ongeveer 1000",
            "right": "Ongeveer 11.000",
            "one": "Ongeveer 100.000",
        },
        reveal="**Ongeveer 11.000.** De 990.000 gewone verzoeken komen langs drie safeguards die elk onafhankelijk 10% missen: 990.000 × 0,1 × 0,1 × 0,1 = 990 komen erdoor. De 10.000 verzoeken met de truc komen er allemaal door. Samen is dat 10.990, bijna elf keer de 1000 die er zonder de truc door zouden komen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Plakken kaas met gaten

    Zie elke safeguard als een plak Zwitserse kaas. Elke plak heeft gaten, de plekken waar hij faalt, en schade komt er alleen door waar gaten op één lijn liggen. Het leerboek *Introduction to AI Safety, Ethics, and Society* (2024) van Dan Hendrycks zegt het zo: waar de gaten in alle plakken op één lijn liggen en samen één doorlopend gat vormen, ligt een mogelijke route naar een ongeluk. Dat is het **Swiss cheese model**.

    Het model komt van James Reason, die ermee begon in 1987 en 1988, tijdens het schrijven van het boek *Human Error* (1990). In een notitie uit 2006 schrijven Reason, Erik Hollnagel en Jean Paries dat de naam niet van Reason zelf kwam, maar waarschijnlijk van Rob Lee, toen directeur van het Bureau of Air Safety Investigation in Canberra. Wat het model zegt: geen enkele storing, van een mens of van de techniek, is op zichzelf genoeg voor een ongeluk. En waar de gaten vandaan komen: kortstondige gaten kunnen ontstaan door fouten en overtredingen van de mensen die het werk direct doen; gaten die langer blijven en gevaarlijker zijn, ontstaan door beslissingen van ontwerpers, bouwers, schrijvers van procedures, topmanagers en onderhoudsmensen.

    Het leerboek trekt de conclusie: het model laat zien waarom defense in depth belangrijk is, want met meer verdedigingslagen wordt de kans kleiner dat er een route naar een ongeluk is die ze allemaal omzeilt. Een systeem veiliger maken kan op twee manieren: met meer plakken, of met kleinere gaten.

    ### Defense in depth

    Het idee van lagen is ouder dan de kaas, en de twijfel erover ook. De nucleaire veiligheid werkte al vroeg met lagen, en Reason, Hollnagel en Paries merken op dat Perrow in het boek *Normal Accidents* (1984) er al op wees dat defense in depth een systeem complexer, strakker gekoppeld en ondoorzichtiger maakt: lagen maken een systeem ook moeilijker te doorzien. De Amerikaanse Nuclear Regulatory Commission omschrijft **defense in depth** als een manier om kerninstallaties te ontwerpen en te bedienen die ongelukken voorkomt waarbij straling of gevaarlijke stoffen vrijkomen, en de gevolgen ervan beperkt: met meerdere onafhankelijke en redundante verdedigingslagen die fouten van mensen en machines opvangen, zodat je nooit op één laag alleen vertrouwt, hoe sterk die ook is. In 1996 beschreef INSAG, de groep van adviseurs voor nucleaire veiligheid van het Internationaal Atoomenergieagentschap, hoe het idee groeide van drie niveaus naar vijf: faalt één niveau, dan treedt het niveau erna in werking. Ze voegden er een waarschuwing aan toe: dat er meerdere elementen van defense in depth zijn, is geen reden om door te gaan als er één ontbreekt.

    AI-bedrijven gebruiken dezelfde woorden. In mei 2025, toen Anthropic strengere beschermingen inschakelde voor een van zijn modellen, citeerde het zijn eigen beleid: kies een aanpak van defense in depth en bouw een reeks verdedigingslagen, die elk pogingen tot misbruik moeten opvangen die door de lagen ervoor kunnen glippen. De lagen hebben een prijs: de maatregelen die erbij kwamen, waaronder classifiers, kunnen soms ook gewone vragen raken, met andere woorden false positives geven.

    ### Lagen als negens

    Het leerboek koppelt de lagen aan [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl): je kunt verdedigingslagen zien als extra negens van betrouwbaarheid. Vaak lijkt het redelijk om aan te nemen dat elke extra laag helpt het risico dat er nog over is ongeveer tien keer kleiner te maken, door 90% daarvan weg te nemen. Let op het woord aannemen: die 90% is een aanname, geen meting. Het voorbeeld uit het leerboek heeft drie lagen die op verschillende dingen mikken: adversarial fine-tuning die 90% van de schadelijke verzoeken eruit filtert; een kunstmatig geweten dat in 10% van de gevallen toch tegen het welzijn van mensen in kan handelen; en AI-waakhonden die AI's die anders handelen dan we willen negen van de tien keer betrappen. Als elke laag 10% doorlaat en ze onafhankelijk van elkaar falen, laten ze samen 0,1 × 0,1 × 0,1 = 0,001 door: drie negens. Het leerboek rekent dat niet uit, en zegt niet of de lagen onafhankelijk falen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Als safeguards niet onafhankelijk zijn

    Vermenigvuldigen mag alleen bij onafhankelijke gebeurtenissen. De handleiding van NASA voor probabilistische risicoanalyse (2011) omschrijft ze zo: twee gebeurtenissen A en B zijn onafhankelijk als weten dat B waar is, je kans op A niet verandert. Met een voorwaardelijke kans: P(A | B) = P(A). De kans dat allebei gebeuren, P(A en B), is P(B) maal P(A | B), het deel van die gevallen waarin ook A gebeurt; bij onafhankelijkheid wordt P(A en B) dus P(A) × P(B). Daarom proberen foutenbomen onafhankelijke gebeurtenissen te gebruiken: dan is de kans dat twee ervan samen gebeuren gewoon het product van hun kansen.

    Echte safeguards missen vaak samen. NASA noemt dat een common cause failure: meer dan één onderdeel dat faalt of niet beschikbaar is door een gedeelde oorzaak. Een van de voorbeelden: een medewerker in een kerncentrale die een ladder bij de motoren van twee redundante pompen zet om het plafond te verven, en zo per ongeluk beide pompen uitschakelt. En NASA waarschuwt: wie bij een systeem met meerdere verdedigingslagen de afhankelijkheden negeert, kan de betrouwbaarheid overschatten. INSAG noemt brand, overstromingen en aardbevingen als gevaren die meerdere niveaus tegelijk kunnen treffen. En Dekker, in *The Field Guide to Human Error Investigations* (2002), zoals aangehaald in de notitie uit 2006: de verdedigingslagen staan niet stil en blijven niet gelijk, en ze zijn ook niet onafhankelijk van elkaar; ze kunnen op elkaar inwerken, elkaar steunen of elkaar uithollen. Bij AI kan hetzelfde verzoek langs het filter en langs het model erachter glippen, als ze dezelfde blinde vlek hebben, zoals verzoeken in een taal die geen van beide goed beheerst: dan liggen hun gaten op één lijn.

    ### Foutenbomen

    Een foutenboom begint volgens het leerboek bij een bepaalde ongewenste uitkomst en werkt van daaruit terug naar de oorzaken. Dat is **foutenboomanalyse**. Bovenaan staat de ongewenste gebeurtenis, de topgebeurtenis. Daaronder verbinden poorten de oorzaken. NASA legt het zo uit: bij een OF-poort gebeurt de gebeurtenis erboven als een van de gebeurtenissen eronder gebeurt, en bij een EN-poort moeten ze allebei gebeuren.

    Bij onafhankelijke gebeurtenissen hebben de poorten eenvoudige regels. Onder een EN-poort vermenigvuldigen de kansen. Bij een OF-poort gebeurt de gebeurtenis erboven alleen niet als geen van de gebeurtenissen eronder gebeurt: voor twee onafhankelijke gebeurtenissen is P(A of B) = 1 − (1 − P(A)) × (1 − P(B)). De kansen gewoon optellen heet de benadering voor zeldzame gebeurtenissen: die zit er dichtbij als de kansen klein zijn, maar overschat volgens NASA de kans op de topgebeurtenis. In het voorbeeld van NASA faalt een systeem als minstens twee van zijn drie onderdelen falen, elk met kans 0,1: het precieze antwoord is 0,028 en de benadering 0,030. Die 0,028 is dezelfde kans als bij drie verschillende modellen die stemmen in de vorige les, elk in 10% van de gevallen fout: een stemming is een kleine foutenboom.

    ### Wat de kaas weglaat

    Het leerboek is open over de zwakte van deze methoden: in de veiligheidskunde gelden ze grotendeels als te simpel en verouderd. Ze volgen een keten van storingen terug vanaf een ongeluk, en sommige ongelukken hebben geen kapot onderdeel. De Mars Polar Lander zette zijn stuwraketten uit voordat hij geland was, omdat de sensoren in zijn poten het uitklappen van die poten voor een landing aanzagen, en hij stortte neer. Geen enkel onderdeel faalde. Reason, Hollnagel en Paries zeggen over het model dat het laat zien hoe een ongeluk kan gebeuren, maar niet hoe het moet gebeuren.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **Swiss cheese model** | safeguards voorgesteld als plakken kaas met gaten: schade komt erdoor waar de gaten van alle plakken op één lijn liggen |
        | **defense in depth** | meerdere lagen safeguards, zodat je nooit op één laag alleen vertrouwt |
        | **foutenboomanalyse** | terugwerken van één ongewenste gebeurtenis naar de storingen die ertoe kunnen leiden, verbonden door EN- en OF-poorten |
        | onafhankelijk | weten dat de ene gebeurtenis plaatsvond, verandert de kans op de andere niet: P(A \| B) = P(A) |
        | common cause failure | meerdere onderdelen die samen falen door één gedeelde oorzaak |
        | topgebeurtenis | de ongewenste gebeurtenis bovenaan een foutenboom |
        | EN-poort | de gebeurtenis erboven gebeurt alleen als alle gebeurtenissen eronder gebeuren |
        | OF-poort | de gebeurtenis erboven gebeurt als minstens één gebeurtenis eronder gebeurt |
        | `and` | de test in Python of twee dingen allebei waar zijn: `"x" in a and "x" in b` |
        """),
        L.watch_out("Vermenigvuldig de kansen van safeguards alleen als ze onafhankelijk van elkaar missen. Een gedeelde oorzaak, dezelfde truc of dezelfde blinde vlek, legt hun gaten op één lijn, en dan zijn drie lagen soms weinig meer waard dan één."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Benoem de topgebeurtenis: de schade die je buiten wilt houden, zoals een schadelijk antwoord dat bij een gebruiker komt.
    2. Zet de safeguards op een rij die daartussen staan, met de kans dat elk ervan mist.
    3. Vraag je af of twee safeguards om dezelfde reden kunnen missen: dezelfde voorbeelden, dezelfde blinde vlek, dezelfde truc.
    4. Teken de foutenboom: een EN-poort waar elke safeguard moet missen, een OF-poort waar één oorzaak genoeg is.
    5. Vermenigvuldig onder een EN-poort; neem bij een OF-poort één min de kans dat er niets onder gebeurt. Allebei alleen bij onafhankelijke gebeurtenissen: zet een gedeelde oorzaak in een eigen tak.
    6. Dicht gaten en voeg niet alleen plakken toe: een gedeelde oorzaak legt een bodem die meer lagen niet verlagen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De chatbot uit *Waarom*, met zijn drie safeguards achter elkaar, krijgt 1.000.000 schadelijke verzoeken. Elke safeguard mist 10% van wat er bij die safeguard aankomt, onafhankelijk van de safeguards ervoor. De cel volgt de verzoeken langs de rij.
    """)
    return


@app.cell
def example_layers():
    safeguards = ["invoerfilter", "weigerend model", "controle van antwoorden"]
    left = 1000000                                   # schadelijke verzoeken die binnenkomen
    for safeguard in safeguards:
        left = left * 0.1                            # alleen de 10% die hij mist, gaat door
        print(safeguard + ":", round(left), "komen erdoor")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke safeguard deelt wat er aankomt door tien: van de 1.000.000 komen er 1000 door: drie negens, één extra negen per laag, zoals het leerboek voorstelt. Dat klopt alleen omdat elke safeguard onafhankelijk van de safeguards ervoor mist. In de oefeningen zoek je de gaten die in drie safeguards op één lijn liggen, en reken je een foutenboom met een gedeelde truc door.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holes():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. gets_through: gaten op één lijn

        De gaten van een safeguard zijn de soorten trucs die hij mist, zoals `"role play"` (rollenspel) of `"base64"`. Schrijf `gets_through(first, second, third)`. De functie krijgt de gaten van drie safeguards achter elkaar als drie lijsten, en geeft een lijst `through` terug met de gaten die in alle drie de lijsten staan, in de volgorde van `first`: een verzoek met zo'n truc komt door elke safeguard. Loop met een `for`-loop door `first`, met `hole` voor elk gat. Vervang de `...`. De check roept `gets_through(["role play", "base64", "prefix"], ["base64", "role play", "many-shot"], ["role play", "other language"])` aan, daarna andere safeguards.
        """),
        L.task("De gaten die door drie plakken op één lijn liggen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holes():
    box_holes = L.code_box("gets_through", lines=6)
    box_holes
    return (box_holes,)


@app.cell(hide_code=True)
def check_holes(box_holes):
    result_holes = L.check("gets_through", box_holes.value)
    result_holes
    return (result_holes,)


@app.cell(hide_code=True)
def hints_holes():
    L.hints(
        "gets_through",
        [
            "Het Swiss cheese model: schade komt er alleen door waar de gaten van alle plakken op één lijn liggen. Een gat van de eerste safeguard telt alleen als de tweede en de derde safeguard het ook hebben.",
            "Begin `through` als een lege lijst. Vraag in een `for`-loop over `first` voor elk `hole` of `second` het heeft en `third` het heeft; zo ja, voeg het toe met `append`. Geef na de loop `through` terug.",
            "`        if hole in second ___ hole in third:`",
        ],
        why="`through = []`, dan `for hole in first:` met daarin `if hole in second and hole in third:` en `through.append(hole)`; na de loop `return through`. In het eerste geval staat alleen `\"role play\"` in alle drie de lijsten.",
    )
    return


@app.cell(hide_code=True)
def text_after_holes():
    mo.md(r"""
    In het eerste geval komt alleen rollenspel (`"role play"`) erdoor. Base64 is een gat in twee plakken, maar de derde safeguard vangt het op. Een safeguard met gaten is niet het probleem; een gat dat op één lijn ligt met gaten in alle andere wel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tree():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. top_event: een foutenboom met EN en OF

        De topgebeurtenis is dat een schadelijk verzoek erdoor komt. Dat gebeurt als elke safeguard het mist, een EN-poort over de safeguards, of als het verzoek een gedeelde truc gebruikt die ze allemaal om de tuin leidt; een OF-poort verbindt die twee. Schrijf `top_event(misses, shared)`. `misses` is een lijst met voor elke safeguard de kans dat hij een gewoon schadelijk verzoek mist, onafhankelijk van de andere; `shared` is de kans dat een verzoek de gedeelde truc gebruikt, onafhankelijk van de rest. De lijst kan leeg zijn: een boom zonder safeguards. Reken de kans dat elke safeguard mist uit in een variabele `all_miss`, met een `for`-loop en `miss` voor elke kans, en geef de kans op de topgebeurtenis terug. Vervang de `...`. De check roept `top_event([0.1, 0.1, 0.1], 0.01)` aan, daarna andere bomen.
        """),
        L.task("De kans op de topgebeurtenis van een foutenboom."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tree():
    box_tree = L.code_box("top_event", lines=5)
    box_tree
    return (box_tree,)


@app.cell(hide_code=True)
def check_tree(box_tree):
    result_tree = L.check("top_event", box_tree.value)
    result_tree
    return (result_tree,)


@app.cell(hide_code=True)
def hints_tree():
    L.hints(
        "top_event",
        [
            "Twee poorten. Onder de EN-poort moet elke safeguard missen, dus bij onafhankelijke safeguards vermenigvuldigen de kansen. Bij de OF-poort gebeurt de topgebeurtenis alleen niet als geen van beide takken gebeurt.",
            "Begin `all_miss` op 1 en vermenigvuldig het in een `for`-loop met elke `miss`. Voor de OF-poort: de kans dat geen van beide takken gebeurt, is de kans op geen gedeelde truc maal de kans dat niet elke safeguard mist; geef één min dat terug.",
            "`    return 1 - (1 - shared) * ___`",
        ],
        why="`all_miss = 1`, dan `for miss in misses:` met `all_miss = all_miss * miss`; dan `return 1 - (1 - shared) * (1 - all_miss)`. Voor `[0.1, 0.1, 0.1]` en `0.01`: `all_miss` is 0,001, en de topgebeurtenis 1 − 0,99 × 0,999 = 0,01099.",
    )
    return


@app.cell(hide_code=True)
def text_after_tree():
    mo.md(r"""
    Met drie safeguards die elk 10% missen en zonder gedeelde truc komt 0,001 erdoor; met een gedeelde truc in 1% van de verzoeken 0,01099, bijna elf keer zoveel. De gedeelde truc is een common cause: één reden waardoor elke safeguard tegelijk mist.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: meer safeguards achter elkaar

    Elke safeguard mist 10% van de gewone schadelijke verzoeken, onafhankelijk van de andere. Met de gedeelde truc leidt 1% van de verzoeken alle safeguards tegelijk om de tuin. Kies hoeveel safeguards er achter elkaar staan, en vergelijk hoeveel van de 1.000.000 schadelijke verzoeken erdoor komen, en hoeveel negens dat is, met en zonder de gedeelde truc.
    """)
    return


@app.cell(hide_code=True)
def demo_layers():
    layers = L.explore("layers", "safeguards achter elkaar", lambda _v: C.demo_frame(_v, L.lang),
                       start=1, values=C.LAYERS, role="",
                       task="Zoek het kleinste aantal safeguards op de schuif waarbij de gedeelde truc meer dan tien keer zoveel schadelijke verzoeken doorlaat als de safeguards zonder die truc doorlaten.")
    layers
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf drie safeguards: 10.990 tegen 1000. Zonder de gedeelde truc levert elke extra safeguard een negen op. Met de truc blijven de negens net onder de twee steken, want de 10.000 verzoeken met de truc komen erdoor, hoeveel safeguards er ook achter elkaar staan; vijf safeguards laten er nog 10.010 door. Dan helpt geen zesde plak, maar de truc vinden en dat gat in minstens één plak dichten: kleinere gaten, niet alleen meer plakken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_common_cause():
    mo.md(r"""
    ### Oefening 3. Drie classifiers

    Een bedrijf stapelt drie safeguards tegen schadelijke verzoeken: drie classifiers, allemaal getraind op dezelfde verzameling voorbeelden van schadelijke verzoeken. Elk houdt in een test 90% van de schadelijke verzoeken tegen.
    """)
    return


@app.cell(hide_code=True)
def choice_common_cause_cell():
    choice_common_cause = L.choice(
        "common_cause",
        question="Waarom kunnen de drie samen veel minder tegenhouden dan de 99,9% die vermenigvuldigen belooft?",
        options={
            "slow": "Drie classifiers maken elk antwoord trager, dus gebruikers zetten ze uit",
            "right": "Een soort verzoek dat niet in hun voorbeelden zat, glipt langs alle drie tegelijk",
            "two": "Kansen vermenigvuldigen werkt alleen bij precies twee safeguards achter elkaar, nooit bij drie",
        },
        correct="right",
        why={
            "right": "De drie leerden van dezelfde voorbeelden, dus ze delen blinde vlekken: hun gaten liggen op één lijn. Ze missen niet onafhankelijk van elkaar, en dan overschat het product hoeveel ze tegenhouden.",
            "slow": "Snelheid kan een echte prijs zijn, maar daarom vermenigvuldigen de kansen niet. De classifiers leerden van dezelfde voorbeelden, dus ze missen dezelfde verzoeken. Zie *Als safeguards niet onafhankelijk zijn* in *Het idee*.",
            "two": "Vermenigvuldigen werkt bij elk aantal onafhankelijke safeguards: 0,1 × 0,1 × 0,1 = 0,001. Deze drie zijn niet onafhankelijk, omdat ze van dezelfde voorbeelden leerden. Zie *Als safeguards niet onafhankelijk zijn* in *Het idee*.",
        },
    )
    choice_common_cause
    return (choice_common_cause,)


@app.cell(hide_code=True)
def text_exercise_missing_layer():
    mo.md(r"""
    ### Oefening 4. Eén laag weg

    Een chatbot heeft vier safeguards achter elkaar. Op een ochtend werkt de controle van de uitgaande antwoorden niet meer. De teamleider zegt: we hebben nog drie safeguards, dus de chatbot blijft draaien tot de controle weer werkt.
    """)
    return


@app.cell(hide_code=True)
def choice_missing_layer_cell():
    choice_missing_layer = L.choice(
        "missing_layer",
        question="Wat zegt defense in depth hierover?",
        options={
            "nines": "Doordraaien is veilig: drie safeguards die elk 90% tegenhouden, geven nog drie negens",
            "right": "De andere lagen zijn geen reden om zonder de ontbrekende door te draaien",
            "logs": "Doordraaien mag, als iemand achteraf de logboeken nakijkt",
        },
        correct="right",
        why={
            "right": "INSAG, de adviseurs voor nucleaire veiligheid, zegt het zo: dat er meerdere elementen van defense in depth zijn, is geen reden om door te gaan als er één ontbreekt. Het ontwerp rekende op alle vier de lagen; met één weg bereikt een verzoek dat door de andere drie glipt nu de gebruiker.",
            "nines": "Dat rekent erop dat de andere drie onafhankelijk missen en dat die 90% klopt; en het ontwerp rekende op vier lagen, niet op drie. Zie *Defense in depth* in *Het idee*.",
            "logs": "Logboeken helpen om achteraf uit te zoeken wat er misging, maar ze houden niets tegen terwijl de chatbot draait. Defense in depth zegt dat de andere lagen geen reden zijn om zonder de ontbrekende door te draaien. Zie *Defense in depth* in *Het idee*.",
        },
    )
    choice_missing_layer
    return (choice_missing_layer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Ontwerpen voor veiligheid](/course/open/rai-safe-design-principles?lang=nl) gaf redundantie, met de waarschuwing dat kopieën samen falen, en [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl) gaf P(A | B). Deze les zette safeguards achter elkaar: het Swiss cheese model, defense in depth, en foutenbomen die vermenigvuldigen onder EN en combineren bij OF, maar alleen als de safeguards onafhankelijk van elkaar missen. De volgende les, *Afglijden naar falen, en zwarte zwanen*, kijkt naar wat deze modellen weglaten: ongelukken zonder kapot onderdeel, en zeldzame gebeurtenissen die geen boom voorzag.

    **Een voorbeeld uit het dagelijks leven.** Een kantoor heeft een pasjeslezer bij de voordeur, een receptionist en een afgesloten serverruimte: drie lagen. Maar als één gestolen pasje zowel de voordeur als de serverruimte opent, delen twee van de drie een gat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_common_cause, choice_missing_layer, result_holes, result_tree):
    L.wrap_up(result_holes, result_tree, choice_common_cause, choice_missing_layer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
