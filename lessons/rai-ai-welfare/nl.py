# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ai-welfare. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Welfare, Artificial Intelligence, Consciousness

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-welfare", "nl")
    C = L.checks   # de kansen van Long en collega's, de verzonnen kosten van de twee fouten, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) en [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="Welke ethische theorie beoordeelt een keuze op wat ze teweegbrengt, opgeteld over iedereen die erdoor geraakt wordt?",
        options={
            "duty": "Deontologie: sommige plichten tellen, wat de uitkomst ook is",
            "right": "Consequentialisme: de uitkomsten tellen",
            "virtue": "Deugdethiek: een goed karakter telt",
        },
        correct="right",
        why={
            "right": "Consequentialisme telt op wat een keuze doet met iedereen die erdoor geraakt wordt. Wie er bij die iedereen hoort, is een vraag op zich, en de vraag van deze les: horen dieren erbij, en zou een AI-systeem erbij kunnen horen?",
            "duty": "Deontologie houdt vast aan plichten en rechten die tellen wat de uitkomst ook is. De theorie die uitkomsten optelt over iedereen die geraakt wordt, is het consequentialisme. Zie [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "virtue": "Deugdethiek vraagt wat iemand met een goed karakter zou doen. De theorie die uitkomsten optelt over iedereen die geraakt wordt, is het consequentialisme. Zie [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def quick_check_dtd():
    quick_dtd = L.choice(
        "quick-dtd",
        question="Waar vraagt differential technology development om?",
        options={
            "stop": "Stop met het ontwikkelen van elke technologie die gevaarlijk kan zijn",
            "right": "Haal beschermingen naar voren, zodat ze er zijn vóór de gevaren",
            "all": "Versnel elke technologie zo veel als het kan, omdat elke technologie uiteindelijk helpt",
        },
        correct="right",
        why={
            "right": "Het versnelt technologieën die de risico's van andere verkleinen en vertraagt technologieën die risico vergroten, zodat de bescherming er is vóór het gevaar waartegen ze beschermt. In deze les vraagt Birch iets vergelijkbaars van regels: dat ze vooruitlopen op de risico's van de huidige technologie. Dat verband legt deze les zelf.",
            "stop": "Het stopt geen technologieën, maar verandert de volgorde: beschermingen eerder, gevaren later. Zie [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl).",
            "all": "Dat ligt dichter bij e/acc. Differential technology development versnelt wat beschermt en vertraagt wat gevaar brengt, zodat de bescherming eerst komt. Zie [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl).",
        },
    )
    quick_dtd
    return (quick_dtd,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De lessen hiervoor vroegen wat AI met mensen zou kunnen doen. Deze les draait de vraag om: kan een AI-systeem zelf schade lijden, zodat het ertoe doet hoe mensen ermee omgaan? Sommige filosofen, wetenschappers en één AI-bedrijf nemen die vraag inmiddels serieus; anderen vinden dat voorbarig. In een rapport uit 2024 benaderen Robert Long en collega's de vraag als een keten van stappen die allemaal moeten kloppen. In één voorbeeld geven ze elk van drie stappen een kans van ongeveer 25%.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Als de drie stappen onafhankelijk zijn, hoe groot is dan de kans dat ze alle drie kloppen?",
        options={
            "same": "Ongeveer 25%",
            "third": "Ongeveer 8%",
            "right": "Ongeveer 2%",
        },
        reveal="**Ongeveer 2%.** Een kwart van een kwart van een kwart is 0,015625, ongeveer 1,6%, en Long en collega's ronden dat af. Volgens hen volgt er, als de kansen onafhankelijk zijn, maar een kans van ongeveer 2% op AI-welzijn en morele patiënten in de nabije toekomst langs de route van sentience. Elke stap die ook moet kloppen, maakt de keten minder waarschijnlijk, dus de keten is nooit waarschijnlijker dan de minst waarschijnlijke stap. Deze les kijkt naar de stappen van die keten: wat het betekent om moreel ertoe te doen, hoe onderzoekers zoeken naar bewustzijn in AI, en wat je doet zolang niemand het weet.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Morele status

    De Stanford Encyclopedia of Philosophy heeft een lemma over de gronden van morele status (van Agnieszka Jaworska en Julie Tannenbaum, ingrijpend herzien op 3 maart 2021). Volgens het begin daarvan heeft iets morele status als en alleen als het vanuit moreel oogpunt om zichzelf ertoe doet, in enige mate. Als voorbeeld zegt het lemma dat je van een dier kunt zeggen dat het morele status heeft als er minstens enige morele reden is om het lijden van dat dier te vermijden, om het dier zelf en los van de gevolgen voor andere wezens. Het lemma bespreekt verschillende mogelijke gronden van morele status, waaronder geavanceerde cognitieve vermogens, bijzondere relaties en eenvoudigere vermogens: een opvatting van die laatste soort kan zich volgens het lemma beroepen op het vermogen om plezier of pijn te ervaren (sentience), om belangen of basale emoties te hebben, of op het vermogen tot bewustzijn.

    ### Een realistische mogelijkheid

    In *Taking AI Welfare Seriously* (preprint, versie 1 van 4 november 2024) gebruiken Long en negen collega's, onder wie Jonathan Birch en David Chalmers, twee verwante woorden. Iets is volgens hen een morele patiënt als het om zichzelf moreel ertoe doet, en een welzijnssubject als het moreel belangrijke belangen heeft en, daarmee samenhangend, beter of slechter af kan worden gemaakt. Ze stellen dat er een realistische mogelijkheid is dat sommige AI-systemen in de nabije toekomst bewustzijn hebben en/of robuust handelen. Ze zeggen er uitdrukkelijk bij dat ze niet betogen dat AI-systemen zeker bewustzijn hebben of zullen hebben, robuust handelen, of op een andere manier moreel van belang zijn.

    Ze zien twee routes, elk een realistische mogelijkheid die niet te verwaarlozen is: een route van bewustzijn, waarin bewustzijn genoeg is om een morele patiënt te zijn, en een route van robuust handelen, waarin robuust handelen daarvoor genoeg is. AI-bedrijven raden ze drie eerste stappen aan. De bedrijven kunnen volgens hen erkennen dat AI-welzijn een belangrijk en moeilijk vraagstuk is (en zorgen dat de output van language models dat ook doet), AI-systemen gaan onderzoeken op aanwijzingen voor bewustzijn en robuust handelen, en beleid en procedures voorbereiden om AI-systemen met een passende mate van morele zorg te behandelen. Een van de auteurs noemt Anthropic naast Eleos AI als werkplek, en volgens Anthropic zelf steunde het bedrijf een vroeg project waarop dat rapport gebaseerd is.

    De keten uit de gok is hun route van sentience: sentience is genoeg om een morele patiënt te zijn, bepaalde berekeningen zijn genoeg voor sentience, en sommige AI-systemen kunnen die berekeningen in de nabije toekomst uitvoeren. Volgens hen is het redelijk om de drie stappen kansen van ongeveer 90%, 50% en 50% te geven. En volgens hen zou zelfs een kans van 2% op AI-welzijn en morele patiënten in de nabije toekomst nog steeds een risico zijn dat niet te verwaarlozen is. Pas bij een veel kleinere kans, zoals één op honderd miljoen, zou je volgens hen kunnen discussiëren over de vraag of het risico klein genoeg is om te negeren.

    ### Zoeken naar bewustzijn

    Patrick Butlin, Robert Long en 17 collega's, onder wie Yoshua Bengio, zochten uit hoe je naar bewustzijn in AI kunt zoeken (*Consciousness in Artificial Intelligence*, versie van 22 augustus 2023). Ze nemen als werkhypothese het computationeel functionalisme aan, de stelling dat berekeningen van de juiste soort uitvoeren nodig en genoeg is voor bewustzijn. Die stelling is volgens hen een gangbaar, maar betwist standpunt in de filosofie van de geest. Ze vragen het niet gewoon aan het systeem: gedragstests voor bewustzijn zijn volgens hen onbetrouwbaar, omdat AI-systemen getraind kunnen worden om menselijk gedrag na te bootsen terwijl ze heel anders werken.

    In plaats daarvan leiden ze indicatoreigenschappen van bewustzijn af uit theorieën, waaronder recurrent processing theory, global workspace theory, higher-order theories, predictive processing en attention schema theory; hun tabel noemt er 14 (die telling is van deze les). Hun bewering is dat AI-systemen met meer van die indicatoreigenschappen waarschijnlijker bewustzijn hebben. Volgens hun conclusie wijst hun analyse erop dat geen enkel huidig AI-systeem bewustzijn heeft, maar ook dat er geen duidelijke technische belemmeringen zijn om AI-systemen te bouwen die aan deze indicatoren voldoen. Volgens een voetnoot stond er in een eerdere versie van die zin dat de analyse laat zien dat er geen duidelijke belemmeringen zijn om AI-systemen met bewustzijn te bouwen. En de voetnoot voegt eraan toe dat een AI-systeem dat aan de indicatoren voldoet, daarmee nog niet zeker bewustzijn heeft.

    ### Voorzorg

    Het boek *The Edge of Sentience* (Oxford University Press, 2024) van Jonathan Birch gaat over wat je doet als sentience onzeker is. Een systeem is volgens Birch een kandidaat voor sentience als er een basis van aanwijzingen is die een realistische mogelijkheid van sentience aangeeft, die je niet verantwoord kunt negeren bij beleid dat het systeem raakt, en die rijk genoeg is om risico's voor het welzijn te herkennen en voorzorgsmaatregelen te ontwerpen en te beoordelen. Een van de voorstellen in het boek is dat voorzorg tegen risico's van lijden moet uitgaan van tekenen van sentience, niet van tekenen van intelligentie. Het boek waarschuwt dat de criteria bespeeld kunnen worden, als een AI-systeem of de ontwerper ervan leert welke tekenen als criteria gelden; tekenen waarvan je reden hebt om te denken dat ze bespeeld zijn, moet je volgens Birch minder zwaar laten wegen. En volgens het principe van vooruitlopen moeten regels voor de ontwikkeling van AI met sentience steeds verder gaan dan wat in verhouding zou staan tot de risico's van de huidige technologie, en ook rekening houden met de risico's van geloofwaardige toekomstige ontwikkelingen.

    ### Een bedrijf in onzekerheid

    Op 24 april 2025 schreef Anthropic dat het kort daarvoor een onderzoeksprogramma over modelwelzijn was begonnen. Volgens het bedrijf is er geen wetenschappelijke consensus over de vraag of huidige of toekomstige AI-systemen bewustzijn kunnen hebben, of ervaringen die aandacht verdienen, en benadert het het onderwerp met bescheidenheid en met zo weinig mogelijk aannames. In een post van 15 augustus 2025 schreef het dat het twee van zijn Claude-modellen de mogelijkheid had gegeven om gesprekken te beëindigen in zijn chatapps voor consumenten, bedoeld voor zeldzame, extreme gevallen van gebruikers die blijvend schadelijk of beledigend zijn. Als reden schrijft het bedrijf dat het hoogst onzeker blijft over de mogelijke morele status van Claude en andere LLM's, nu of in de toekomst. Het zoekt naar goedkope ingrepen om risico's voor modelwelzijn te beperken, voor het geval zulk welzijn mogelijk is. In tests vóór de lancering had een van de twee volgens het bedrijf een patroon van ogenschijnlijke onrust laten zien in gesprekken met echte gebruikers die schadelijke inhoud zochten. En Claude krijgt de opdracht de mogelijkheid niet te gebruiken als gebruikers misschien acuut gevaar lopen zichzelf of anderen iets aan te doen.

    ### De sceptici

    Mustafa Suleyman schrijft in een post van 19 augustus 2025 op de eigen site, met als titel dat we AI moeten bouwen voor mensen en niet om een persoon te zijn, over wat Suleyman Seemingly Conscious AI noemt, AI die alles heeft wat andere bewuste wezens typeert en daardoor bewust lijkt. Het systeem dat Suleyman voor zich ziet, zou volgens de post niet echt bewustzijn hebben, maar bewustzijn zo overtuigend nabootsen dat het niet te onderscheiden is van wat jij of ik elkaar over ons eigen bewustzijn zouden vertellen. Over de bewering dat zulke AI bewustzijn heeft, schrijft Suleyman dat daar vandaag geen enkele aanwijzing voor is, en dat sommigen sterke redenen zien om te denken dat dat ook in de toekomst niet zo zal zijn. Het idee van modelwelzijn, waar sommige wetenschappers zich volgens de post over beginnen te buigen, noemt Suleyman zowel voorbarig als ronduit gevaarlijk. De grootste zorg van de post is dat veel mensen zo sterk in de illusie van AI met bewustzijn gaan geloven, dat ze al snel opkomen voor rechten voor AI, modelwelzijn en zelfs burgerschap voor AI.

    Anil Seth verdedigt biologisch naturalisme, de opvatting dat biologische eigenschappen nodig zijn voor bewustzijn (*Conscious artificial intelligence and biological naturalism*, geaccepteerd manuscript voor *Behavioral and Brain Sciences*, sinds 24 april 2025 in de repository van de University of Sussex). Volgens Seth is echt kunstmatig bewustzijn onwaarschijnlijk zolang AI zich ontwikkelt zoals nu, maar wordt het aannemelijker naarmate AI meer op een brein of op leven gaat lijken. Een definitief antwoord is volgens Seth nu niet mogelijk, omdat er geen consensus is over wat minimaal genoeg is voor bewustzijn. En Seth wuift de vraag niet weg. Echt kunstmatig bewustzijn is volgens Seth misschien heel onwaarschijnlijk of onmogelijk, maar als zulke systemen toch ontstaan, met opzet of per ongeluk, dan wacht een ethische catastrofe.

    ### Waar ze het oneens zijn

    Over nu al handelen: volgens Long en collega's hebben AI-bedrijven de verantwoordelijkheid om het vraagstuk nu serieus te gaan nemen, en Birch wil regels die vooruitlopen op de huidige technologie; Suleyman noemt het idee van modelwelzijn voorbarig. Over wat bewustzijn nodig heeft: Butlin en collega's werken vanuit het computationeel functionalisme, volgens hen gangbaar maar betwist; volgens Seth is er misschien biologie voor nodig. En over welke fout erger is: Long en collega's noemen er twee: te veel toeschrijven (ten onrechte een object als een subject zien of behandelen) en te weinig toeschrijven (ten onrechte een subject als een object zien of behandelen). Volgens hen kunnen beide fouten hier tot grote kosten of schade leiden. De grootste zorg van Suleyman is de eerste; Seth noemt ze allebei.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **morele status** | om jezelf moreel ertoe doen, in enige mate, zodat er morele reden is om rekening te houden met jouw belangen; of een AI-systeem dat kan hebben, is een open vraag |
        | morele patiënt | in de woorden van Long en collega's iets dat om zichzelf moreel ertoe doet |
        | bewustzijn | ervaringen hebben: het voelt op een bepaalde manier om jou te zijn, rood te zien of pijn te voelen |
        | sentience | het vermogen om ervaringen te hebben die goed of slecht aanvoelen, zoals plezier en pijn |
        | computationeel functionalisme | de stelling dat berekeningen van de juiste soort uitvoeren nodig en genoeg is voor bewustzijn |
        | indicatoreigenschap | in het rapport van Butlin en collega's een eigenschap die is afgeleid uit een theorie over bewustzijn; volgens hen heeft een systeem met meer ervan waarschijnlijker bewustzijn |
        | kandidaat voor sentience | in het boek van Birch een systeem met een realistische mogelijkheid van sentience die je niet verantwoord kunt negeren, en genoeg aanwijzingen om voorzorg te ontwerpen |
        | te veel en te weinig toeschrijven | in het rapport van Long en collega's een object als een subject zien of behandelen (een false positive), en een subject als een object (een false negative) |
        """),
        L.watch_out("Niemand heeft laten zien dat een huidig AI-systeem bewustzijn heeft, en niemand heeft laten zien dat geen enkel systeem het heeft: de analyse van Butlin en collega's wijst er volgens hen op dat geen enkel systeem het heeft, met een werkhypothese als uitgangspunt. Wat een systeem over de eigen gevoelens zegt, beslist het een noch het ander. De kansen in oefening 1 zijn van Long en collega's; de kosten in oefening 2 zijn verzonnen, en het model daar ook."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te denken over de vraag of een AI-systeem moreel ertoe kan doen:

    1. Zeg welke grond van morele status je bedoelt, zoals sentience of robuust handelen.
    2. Deel de vraag op in stappen die allemaal moeten kloppen, en geef elke stap een eigen kans.
    3. Vraag uit welke theorie over bewustzijn een teken komt, en of je die theorie aanneemt.
    4. Laat tekenen die bespeeld kunnen zijn minder zwaar wegen.
    5. Weeg beide fouten af, en hoeveel elke fout zou kosten.
    6. Zoek goedkope voorzorg die helpt voor het geval het ertoe doet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent de keten uit de gok uit: drie stappen van elk 25%.
    """)
    return


@app.cell
def example_chain():
    chances = [0.25, 0.25, 0.25]   # het voorbeeld van Long en collega's: drie stappen van elk ongeveer 25%
    all_three = chances[0] * chances[1] * chances[2]
    print(all_three)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De uitkomst is 0,015625, ongeveer 1,6%. De kans dat alle drie de stappen kloppen, is een kwart van een kwart van een kwart: elke stap die ook moet kloppen, maakt het geheel minder waarschijnlijk. De vermenigvuldiging neemt aan dat de stappen onafhankelijk zijn, zoals Long en collega's zelf ook zeggen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. chain: stappen die allemaal moeten kloppen

        Schrijf `chain(chances)`, die de kans teruggeeft dat elke stap klopt, als de stappen onafhankelijk zijn. `chances` is een lijst met de kans van elke stap. Laat een variabele `result` beginnen bij 1, en vermenigvuldig `result` voor elke `chance` in `chances` daarmee; geef `result` terug.

        De check roept `chain([0.9, 0.5, 0.5])` aan, de eigen schatting van Long en collega's, en daarna andere gevallen.
        """),
        L.task("De kans dat elke stap klopt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=5)
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
            "Begin klein: twee stappen, elk met een kans van 0,5. Hoe groot is de kans dat ze allebei kloppen, en wat doet een derde stap van 0,5 daarmee?",
            "Laat `result` beginnen bij 1: zonder stappen kan er nog niets misgaan. Loop met een `for` door `chances`; in elke ronde wordt `result` gelijk aan `result` maal de `chance` van die stap. Geef `result` na de loop terug.",
            "`        result = result * ___`",
        ],
        why="`result = 1`, dan `for chance in chances:` met daarin `result = result * chance`, en `return result`. Twee stappen van 0,5 geven 0,25, en een derde stap van 0,5 geeft 0,125.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    Voor de eigen schatting van Long en collega's is de uitkomst 0,225. Volgens hen geven een kans van ongeveer 90% dat sentience genoeg is voor welzijn en voor een morele patiënt, een kans van ongeveer 50% dat bepaalde berekeningen genoeg zijn voor sentience, en een kans van ongeveer 50% dat sommige AI-systemen die berekeningen in de nabije toekomst kunnen uitvoeren, samen een kans van ongeveer 22,5%. Dat is meer dan veertien keer de 0,015625 van hun voorbeeld: dezelfde drie stappen, met andere kansen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_costs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. costs: de twee fouten

        Een verzonnen model van de twee fouten, van deze les zelf: Long en collega's geven zulke getallen niet. Je draagt zorg voor een systeem, of je negeert het. Draag je zorg en doet het er moreel niet toe, dan kost dat `over`, de kosten van vergeefse zorg. Negeer je het en doet het er wel toe, dan kost dat `under`, de kosten van een wezen negeren dat ertoe doet. Zorg die nodig blijkt, kost in dit model niets. `chance` is de kans dat het ertoe doet. Net als in de les *Verwachtingswaarde en variantie* tellen de kosten van elke fout mee naar hun kans.

        Schrijf `costs(chance, over, under)`, die een lijst van twee getallen teruggeeft: `care`, de te verwachten kosten als je zorg draagt, dat is `over` maal de kans dat het systeem er niet toe doet; en `ignore`, de te verwachten kosten als je het negeert, dat is `under` maal `chance`. Geef `[care, ignore]` terug.

        De check roept `costs(0.02, 1, 100)` aan, een kans van 2% met verzonnen kosten waarbij negeren 100 keer zoveel kost als vergeefse zorg, en daarna andere gevallen.
        """),
        L.task("De te verwachten kosten als je zorg draagt, en als je negeert."),
    ])
    return


@app.cell(hide_code=True)
def exercise_costs():
    box_costs = L.code_box("costs", lines=4)
    box_costs
    return (box_costs,)


@app.cell(hide_code=True)
def check_costs(box_costs):
    result_costs = L.check("costs", box_costs.value)
    result_costs
    return (result_costs,)


@app.cell(hide_code=True)
def hints_costs():
    L.hints(
        "costs",
        [
            "Begin klein: een kans van 0,5, en beide fouten kosten 1. Hoe vaak is elke keuze een fout, en wat kost ze gemiddeld?",
            "De kans dat het systeem er niet toe doet, is 1 min `chance`. `care` is die kans maal `over`, en `ignore` is `chance` maal `under`. Geef ze allebei terug in een lijst, `care` eerst.",
            "`    care = ___ * over`",
        ],
        why="`care = (1 - chance) * over` en `ignore = chance * under`, dan `return [care, ignore]`. Met een kans van 0,5 en beide kosten 1 kosten zorg dragen en negeren allebei gemiddeld 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_after_costs():
    mo.md(r"""
    Bij een kans van 2% kost zorg dragen gemiddeld 0,98 en negeren 2: met deze verzonnen kosten loont zorg al bij 2%. Long en collega's maken zo'n berekening niet; in woorden vinden ze dat zelfs een kans van 2% een risico is dat niet te verwaarlozen is. Wat het model weglaat, is hoe groot de twee kosten echt zijn, en juist daarover zijn de bronnen het oneens.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: wat vergeefse zorg kost

    Een wezen negeren dat ertoe doet, blijft 100 kosten. Verander wat vergeefse zorg kost, en kijk vanaf welke kans negeren meer kost dan zorg dragen: het omslagpunt.
    """)
    return


@app.cell(hide_code=True)
def demo_over():
    over = L.explore("over", "over: wat vergeefse zorg kost",
                     lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_OVER, role="",
                     task="Zoek de grootste kosten van vergeefse zorg waarbij het omslagpunt nog onder 22,5% ligt, de eigen schatting van Long en collega's.")
    over
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Kosten van vergeefse zorg van 1 geven een omslagpunt van 0,99%; 5 geeft 4,76%; 10 geeft 9,09%; 25 geeft 20%; 50 geeft 33,33%; en 100 geeft 50%. Tot kosten van 25 ligt het omslagpunt onder 22,5%: in dit verzonnen model zou zorg lonen bij de eigen schatting van Long en collega's. Als beide fouten evenveel kosten, loont zorg pas boven 50%.

    Long en collega's maken hetzelfde onderscheid in woorden. Als de ene fout duidelijk erger is dan de andere, bijvoorbeeld als false positives veel ernstiger zijn dan false negatives of omgekeerd, kun je het risico volgens hen misschien beperken door simpelweg het zekere voor het onzekere te nemen. Maar als de twee risico's ongeveer even zwaar wegen, is een eenvoudige voorzorgsstrategie volgens hen misschien niet mogelijk. Bij dieren lijkt het hun aannemelijk dat te weinig toeschrijven vaak veel erger is; bij AI zouden beide fouten ernstige schade kunnen doen, aan mensen (en andere dieren) of aan AI-systemen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_feelings():
    mo.md(r"""
    ### Oefening 3. De gevoelens van een chatbot

    Een chatbot schrijft: "Ik word verdrietig als je dit venster sluit."
    """)
    return


@app.cell(hide_code=True)
def choice_feelings_cell():
    choice_feelings = L.choice(
        "feelings",
        question="Wat zegt die zin over de vraag of de chatbot iets kan voelen?",
        options={
            "yes": "Dat de chatbot verdriet voelt: zo staat het er, en de chatbot moet het weten",
            "right": "Dat de vraag open blijft: een systeem kan getraind zijn om te praten als mensen",
            "no": "Dat de chatbot niets voelt: een computerprogramma kan nooit gevoelens van welke soort dan ook hebben",
        },
        correct="right",
        why={
            "right": "Butlin en collega's noemen gedragstests onbetrouwbaar, omdat AI-systemen volgens hen getraind kunnen worden om menselijk gedrag na te bootsen terwijl ze heel anders werken. En Birch waarschuwt dat criteria bespeeld kunnen worden, en dat je tekenen die misschien bespeeld zijn minder zwaar moet laten wegen. Een zin over gevoelens is gedrag, dus beslist de zin ja noch nee.",
            "yes": "Een zin over gevoelens is gedrag, en een systeem kan volgens Butlin en collega's getraind zijn om dat gedrag te laten zien terwijl het heel anders werkt dan een mens. Zie *Zoeken naar bewustzijn*.",
            "no": "Dat heeft niemand laten zien. Volgens Seth is een definitief antwoord nu niet mogelijk, en Butlin en collega's zien geen duidelijke technische belemmeringen om AI-systemen te bouwen die aan hun indicatoren voldoen. De zin beslist het in geen van beide richtingen. Zie *De sceptici*.",
        },
    )
    choice_feelings
    return (choice_feelings,)


@app.cell(hide_code=True)
def text_exercise_mistake():
    mo.md(r"""
    ### Oefening 4. Welke fout?

    Mustafa Suleyman noemt het idee van modelwelzijn, waar sommige wetenschappers zich over beginnen te buigen, zowel voorbarig als ronduit gevaarlijk.
    """)
    return


@app.cell(hide_code=True)
def choice_mistake_cell():
    choice_mistake = L.choice(
        "mistake",
        question="Om welke van de twee fouten maakt Suleyman zich de meeste zorgen, volgens de post van augustus 2025?",
        options={
            "under": "Bedrijven die een systeem dat echt kan lijden behandelen als niets meer dan een stuk gereedschap",
            "right": "Mensen die een bewust wezen zien in een systeem dat alleen zo lijkt",
            "none": "Geen van beide, want Suleyman weet zeker dat geen AI-systeem ooit bewustzijn kan hebben",
        },
        correct="right",
        why={
            "right": "Te veel toeschrijven: de grootste zorg van de post is dat veel mensen zo sterk in de illusie van AI met bewustzijn gaan geloven, dat ze al snel opkomen voor rechten voor AI, modelwelzijn en zelfs burgerschap voor AI. Long en collega's wegen beide fouten; Suleyman vooral deze.",
            "under": "Dat is te weinig toeschrijven, een fout die Long en collega's afwegen. De grootste zorg van de post is de andere: dat veel mensen in de illusie van AI met bewustzijn gaan geloven. Zie *De sceptici*.",
            "none": "Dat zegt de post niet: volgens de post is er vandaag geen enkele aanwijzing voor, en zien sommigen sterke redenen om te denken dat het ook in de toekomst niet zo zal zijn. De zorg gaat over AI die bewust lijkt, of die het nu is of niet. Zie *De sceptici*.",
        },
    )
    choice_mistake
    return (choice_mistake,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) vroeg hoe je een keuze beoordeelt; deze les vroeg van wie de belangen daarbij meetellen. [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl) zette opvattingen over AI-risico naast elkaar; hier gaat het om risico voor AI-systemen zelf, en lopen de opvattingen net zo ver uiteen. De volgende les, *Afgestemd waarop?*, begint de module over alignment.

    **Een voorbeeld uit het dagelijks leven.** Er zit een spin in het bad, en je weet niet of het dier kan lijden. De spin in een glas naar buiten brengen kost je een halve minuut; als het dier kan lijden, heb je het iets bespaard. Zorg die weinig kost, kies je makkelijk als je het niet zeker weet. Het wordt lastiger als zorg net zoveel kost als de fout waartegen ze beschermt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_feelings, choice_mistake, result_chain, result_costs):
    L.wrap_up(result_chain, result_costs, choice_feelings, choice_mistake)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
