# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-discourse. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-discourse", "nl")
    C = L.checks   # de tekst van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_relevant():
    quick_relevant = L.choice(
        "quick-relevant",
        question="Kim: *De bus is weer te laat.* Sam: *Mijn fiets staat in de schuur.* Wat haalt Kim uit het antwoord van Sam?",
        options={
            "right": "Een aanbod om de fiets te lenen",
            "where": "Waar Sam de fiets bewaart, als feit",
            "bus": "Dat Sam weet hoe laat de bus komt",
        },
        correct="right",
        why={
            "right": "De woorden van Sam zeggen alleen waar de fiets staat. Kim neemt aan dat het antwoord relevant is, de maxime van relatie, en werkt uit wat Sam moet bedoelen: een implicatuur. Een lezer van een tekst doet iets vergelijkbaars, zoals deze les laat zien: een lezer gaat ervan uit dat elke zin iets te maken heeft met de zin ervoor.",
            "where": "Dat is wat de woorden zeggen. Maar waarom zou Sam dat zeggen, vlak na de klacht van Kim? Een luisteraar die aanneemt dat het antwoord relevant is, haalt er meer uit. Zie *Samenwerken* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
            "bus": "Sam zegt niets over de bus. Wat Kim uit het antwoord haalt, komt voort uit de aanname dat het relevant is voor de late bus: Sam biedt een uitweg. Zie *Samenwerken* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
        },
    )
    quick_relevant
    return (quick_relevant,)


@app.cell(hide_code=True)
def quick_check_the():
    quick_the = L.choice(
        "quick-the",
        question="*De fiets in de schuur is rood.* Wat neemt deze zin als vanzelfsprekend aan?",
        options={
            "right": "Dat er één fiets in de schuur staat",
            "likes": "Dat de spreker rode fietsen mooi vindt",
            "painted": "Dat iemand de fiets vorige week rood heeft geverfd",
        },
        correct="right",
        why={
            "right": "*De* neemt aan dat er precies één zo'n ding is: een presuppositie. Een luisteraar die dat nog niet wist, leert het terloops. Deze les komt terug op *de*: dat lidwoord wijst vaak op iets wat de lezer al is tegengekomen.",
            "likes": "De zin zegt niets over wat de spreker mooi vindt. Wat een zin als vanzelfsprekend aanneemt, blijft staan als je *niet* toevoegt: *De fiets in de schuur is niet rood* heeft nog steeds een fiets in de schuur. Zie *Als vanzelfsprekend aangenomen* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
            "painted": "Niets in de zin gaat over verven. Wat een zin als vanzelfsprekend aanneemt, blijft staan als je *niet* toevoegt: *De fiets in de schuur is niet rood* heeft nog steeds een fiets in de schuur. Zie *Als vanzelfsprekend aangenomen* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
        },
    )
    quick_the
    return (quick_the,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een programma dat samenvat, kiest zinnen uit een lange tekst; een zoekmachine toont een paar zinnen van een pagina; een language model schrijft een pagina woord voor woord. Elke zin kan op zich in orde zijn, en het geheel toch geen tekst. Vroeg in hun hoofdstuk hierover geven Jurafsky en Martin twee zinnen van de informaticus Jerry Hobbs (1979): een zin over John die met de trein van Parijs naar Istanbul gaat, en dan een zin die zegt dat John van spinazie houdt. Dat paar hangt niet samen, schrijven ze, omdat een lezer niet ziet waarom de tweede zin op de eerste volgt: wat heeft spinazie met een treinreis te maken? Meten hoe goed zinnen samenhangen is nuttig, voegen ze eraan toe: bij het beoordelen van opstellen, bij samenvatten, en misschien zelfs bij het herkennen van tekenen van verstoorde taal. Deze les kijkt naar wat een tekst bij elkaar houdt: banden in de woorden, een samenhang in betekenis eronder, en een volgorde binnen elke zin.

    **Een voorbeeld uit het dagelijks leven.** Je plakt alinea's van drie webpagina's in één document. Elke alinea leest goed, maar de tweede begint met *Dit plan*, en de lezer heeft nog geen plan gezien.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Regina Barzilay en Mirella Lapata (2008) gaven een programma paren teksten: een nieuwsbericht of een ongevalsrapport, en dezelfde tekst met de zinnen in willekeurige volgorde. Het programma begreep de teksten niet. Het noteerde alleen, per zin, welke personen en dingen terugkwamen, en in welke rol. In hoeveel van de paren gaf het de oorspronkelijke tekst de hoogste score?",
        options={
            "right": "In ongeveer 9 van de 10 paren",
            "half": "In ongeveer de helft, zoals bij kop of munt",
            "six": "In ongeveer 6 van de 10 paren",
        },
        reveal="**In ongeveer 9 van de 10 paren**: 87,2 op de 100 bij nieuwsberichten van het persbureau Associated Press over aardbevingen, en 90,4 op de 100 bij rapporten over vliegtuigongevallen van de National Transportation Safety Board, gemeten op 2.056 en 2.087 testparen. De teksten waren kort: gemiddeld 10,4 en 11,5 zinnen. De test gaat ervan uit dat de oorspronkelijke volgorde coherent is, merken de auteurs op. En bij de aardbevingen scoorde een versie van het programma zonder een van de onderdelen iets hoger, met 88,3 op de 100.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat een tekst maakt

    Jurafsky en Martin noemen een samenhangende, gestructureerde groep zinnen een **discourse**. Het verband tussen de zinnen dat een echte discourse onderscheidt van een willekeurige verzameling zinnen, noemen ze **coherentie**. Een discourse kan een verhaal zijn, een verslag of een gesprek. Coherentie is dat verband in betekenis: een lezer ziet waarom elke zin op de vorige volgt. Het paar van Hobbs mist dat. In de coherente versie van Jurafsky en Martin moet de reiziger naar een congres: de tweede zin geeft dan de reden voor de reis.

    Rhetorical Structure Theory, van William Mann en Sandra Thompson (1988), geeft zulke verbanden tussen naburige delen van een tekst een naam: het ene deel onderbouwt het andere, werkt het uit, of staat er tegenover. Een voorbeeld, vrij naar Jurafsky en Martin: *Kim moet thuis zijn. De fiets staat in de gang.* De tweede zin onderbouwt de eerste. Vaak zegt geen enkel woord welk verband er is. De Penn Discourse Treebank 2.0, van Rashmi Prasad en collega's (2008), markeert de verbanden in ongeveer een miljoen woorden uit de Wall Street Journal: 40.600 in totaal. Zo'n 18.000 hebben een **signaalwoord** in de tekst, een woord zoals *omdat* of *daardoor*; zo'n 16.000 hebben er geen, en de lezer vult het verband zelf in; de rest is van andere soorten. De maxime van relatie uit *Wat bedoeld wordt, niet gezegd* werkt hier ook, in de lezing van de course: een lezer neemt aan dat de volgende zin relevant is, en zoekt uit hoe.
    """)
    return


@app.cell(hide_code=True)
def text_idea_cohesion():
    mo.md(r"""
    ### Banden in de woorden

    Een deel van wat een tekst bij elkaar houdt, zie je aan de woorden. M.A.K. Halliday en Ruqaiya Hasan (1976) noemden dat **cohesie**. De hoofdstukken van hun boek *Cohesion in English* gaan over vijf soorten banden: verwijzing, substitutie, ellips, conjunctie en lexicale cohesie. Met voorbeelden van de course zelf: *het* verwijst terug naar iets wat eerder genoemd is; *er een* neemt de plaats in van een woord (*Ik heb een pen nodig. Heb jij er een?*); woorden worden weggelaten en toch begrepen (*Sam kan zwemmen, en Kim ook.*); een signaalwoord zegt hoe twee zinnen samenhangen; en een woord komt terug, of een verwant woord. Die laatste soort, **lexicale cohesie**, beschrijven Jurafsky en Martin als het delen van dezelfde of verwante woorden in zinnen die dicht bij elkaar staan.

    Cohesie is geen coherentie. In het paar van Hobbs zoals het in het boek staat, wijst de tweede zin met een voornaamwoord terug naar John: een band, en toch geen samenhang in betekenis. En twee zinnen kunnen samenhangen zonder enige band: *Kim miste de bus. De wekker was niet afgegaan.*

    Banden zijn te tellen. Marti Hearst (1997) gebruikte een heel eenvoudige vorm van lexicale cohesie om te vinden waar een lange tekst overgaat op een ander deelonderwerp. Het programma van Hearst vergelijkt naburige blokken zinnen: hoe meer woorden twee blokken gemeen hebben, hoe hoger de score bij de overgang ertussen. Een lage score met hoge scores ervoor en erna wijst, zo neemt Hearst aan, op een andere woordkeus, en dus op een nieuw deelonderwerp. Oefening 1 doet hetzelfde voor losse zinnen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_entities():
    mo.md(r"""
    ### Steeds dezelfde dingen

    Een coherente tekst blijft ook vaak bij waar de tekst over gaat. Barbara Grosz, Aravind Joshi en Scott Weinstein (1995) bouwden hier een theorie van centering op. In de samenvatting van Jurafsky en Martin gelden overgangen tussen zinnen die bij dezelfde opvallende entiteit blijven, als coherenter dan overgangen die steeds van de ene entiteit naar de andere springen; opvallend wil hier zeggen: op dat moment het meest in beeld. Het programma uit de gok schreef een tekst als een tabel, het entity grid, met een rij per zin, een kolom per ding, en in elke cel de rol van het ding. De afbeelding toont de tekst van de course zelf op die manier, gekanteld en zonder de rollen; het uitgewerkte voorbeeld schrijft de tekst uit als data. Elke keer dat een tekst een ding noemt of ernaar verwijst, is een **mention** van dat ding.
    """)
    return


@app.cell(hide_code=True)
def grid_picture():
    mo.Html(C.grid_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_order():
    mo.md(r"""
    ### Bekend eerst, nieuw achteraan

    Een tekst houdt bij wat de lezer al heeft. Jurafsky en Martin noemen in hun hoofdstuk over coreferentie de manier waarop verwijzingen nieuwe dingen in de tekst brengen (nieuwe informatie), of dingen oppakken die er al zijn (oude informatie), de informatiestatus of informatiestructuur. *Een storm* brengt een storm de tekst in; *de storm*, een zin later, pakt die storm weer op. Sommige dingen zijn nieuw en toch verwacht. Hun voorbeeld, hier vertaald: *Ik was gisteren in een prachtig restaurant. De chef had het net geopend.* De chef is nieuw, en een lezer kan afleiden dat een restaurant er een heeft.

    Susan Haviland en Herbert Clark (1974) maten wat dat kost. Lezers zagen paren zinnen en drukten op een knop zodra ze de tweede begrepen, *Het bier was lauw.* De zinnen waren Engels; hier staan de zinnen vertaald. Na *We haalden wat bier uit de kofferbak.* duurde dat gemiddeld 835 milliseconden; na *We bekeken de picknickspullen.*, waar de lezer het bier moet afleiden, 1.016 milliseconden, 181 meer. Een tweede experiment liet zien dat het herhaalde woord alleen het verschil niet verklaarde. De auteurs voegen eraan toe dat ze maten hoe lang het duurde tot iemand het gevoel had de zin te begrijpen, en dat je de absolute verschillen tussen de condities niet al te serieus moet nemen.

    Binnen een zin loopt de volgorde vaak van bekend naar nieuw: **informatiestructuur**. Clark en Haviland (1977) schrijven over het Engels dat gegeven informatie in het algemeen vóór nieuwe komt. Engelse zinnen beginnen daardoor ook vaak met een woordgroep met een bepaald lidwoord, en schuiven woordgroepen met een onbepaald lidwoord naar later in de zin. Gregory Ward en Betty Birner (2001) schrijven dat het Engels, net als veel andere talen, de neiging heeft om gegeven informatie vóór nieuwe te zetten. De naslaggrammatica van het Nederlands, de *Algemene Nederlandse Spraakkunst*, noemt het verschijnsel het *links-rechts-principe*: wat informatief belangrijker is, staat vaak verder achteraan in de zin. Zinnen die daarvan afwijken, zijn volgens de ANS lang niet altijd ongewoon. En de taalkundige Randy LaPolla (2019) wijst erop dat sommige talen de nieuwe informatie juist vooraan in de zin zetten.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **discourse** | een samenhangende, gestructureerde groep zinnen, zoals een verhaal, een verslag of een gesprek |
        | **coherentie** | hoe de zinnen van een tekst in betekenis samenhangen, zodat een lezer ziet waarom elke zin volgt |
        | **signaalwoord** | een woord zoals *omdat*, *dus* of *maar* dat zegt hoe twee delen van een tekst samenhangen |
        | **cohesie** | de banden die je aan de woorden ziet: woorden die terugverwijzen, signaalwoorden, herhaalde of verwante woorden |
        | **lexicale cohesie** | banden door dezelfde of verwante woorden in zinnen die dicht bij elkaar staan |
        | **mention** | elke keer dat een tekst een ding noemt of ernaar verwijst: *een storm* en *de storm* zijn twee mentions van één storm |
        | **informatiestructuur** | de verdeling in een zin tussen wat de lezer al is tegengekomen (oud) en wat de zin inbrengt (nieuw), en de volgorde ervan |
        | `len(set(words))` | het aantal verschillende woorden in een lijst: de set bevat elk woord maar één keer |
        """),
        L.watch_out("*Nieuw* betekent hier nieuw in deze tekst, niet onbekend bij de lezer. Ward en Birner geven *de maan* in een zin over een mooie nacht: nieuw in de tekst als de maan voor het eerst opduikt, en bij iedereen bekend."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Deel de tekst op in zinnen, en zet bij elke zin de woorden die inhoud dragen.
    2. Tel de woorden die naburige zinnen delen: waar het aantal zakt, verandert het onderwerp misschien.
    3. Zet bij elke zin de dingen die de zin noemt, en markeer elke mention als nieuw of oud: heeft de tekst dat ding al eerder genoemd?
    4. Zoek het verband tussen buren: een reden, een onderbouwing, een tegenstelling? Staat er een signaalwoord dat het zegt?
    5. Kijk naar de volgorde: begint een zin met iets ouds, en eindigt de zin met iets nieuws?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel schrijft de tekst van de course zelf uit, de acht zinnen van de afbeelding. `WORDS` bevat per zin de woorden die inhoud dragen, in kleine letters en in hun grondvorm, met de hand gekozen; `MENTIONS` bevat de dingen die elke zin noemt, op volgorde. In de derde zin is *ermee* weer de fiets. De tweede cel definieert twee functies, probeert beide uit en kijkt naar de eerste twee zinnen.
    """)
    return


@app.cell
def text_cell():
    TEXT = [
        "Kim wilde een goedkope fiets voor de rit naar het werk.", "De fiets in de etalage was goedkoop.",
        "Kim kocht de fiets en reed ermee naar huis.", "Vlak bij huis liep de ketting eraf.",
        "Kim zette de ketting terug met gereedschap uit de schuur.", "Die nacht trok een storm over de stad.",
        "De storm blies het dak van de schuur.", "De volgende ochtend reed Kim langs het dak dat op straat lag.",
    ]
    WORDS = [
        ["kim", "willen", "goedkoop", "fiets", "rit", "werk"], ["fiets", "etalage", "goedkoop"],
        ["kim", "kopen", "fiets", "rijden", "huis"], ["vlak", "huis", "ketting", "lopen"],
        ["kim", "terugzetten", "ketting", "gereedschap", "schuur"], ["nacht", "storm", "trekken", "stad"],
        ["storm", "blazen", "dak", "schuur"], ["volgend", "ochtend", "rijden", "kim", "dak", "liggen", "straat"],
    ]
    MENTIONS = [
        ["Kim", "fiets", "werk"], ["fiets", "etalage"], ["Kim", "fiets", "fiets", "huis"], ["huis", "ketting"],
        ["Kim", "ketting", "gereedschap", "schuur"], ["nacht", "storm", "stad"], ["storm", "dak", "schuur"],
        ["ochtend", "Kim", "dak", "straat"],
    ]
    return MENTIONS, TEXT, WORDS


@app.cell
def example_look(MENTIONS, WORDS):
    def flat(text):
        out = []
        for sentence in text:
            for m in sentence:
                out.append(m)
        return out

    def mentions_of(thing, text):
        return [i for i in range(len(text)) if thing in text[i]]

    print(flat(MENTIONS)[:7])
    print(mentions_of("Kim", MENTIONS), mentions_of("storm", MENTIONS))
    print(WORDS[0])
    print(WORDS[1])
    print(len(set(["fiets", "fiets", "goedkoop"])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `flat` zet de mentions van alle zinnen in één lijst, in de volgorde van de tekst: eerst de drie van de eerste zin, dan die van de tweede. `mentions_of` geeft de plaatsen van de zinnen die een ding noemen, geteld vanaf 0: Kim staat in vier ervan, de storm in twee na elkaar, zoals in de afbeelding. De eerste twee zinnen delen *fiets* en *goedkoop*. De laatste regel telt verschillende woorden: de set houdt *fiets* één keer over, dus komt er 2 uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. overlap: de woorden die buren delen

        Schrijf `overlap(a, b)`: geef terug hoeveel verschillende woorden de lijsten `a` en `b` gemeen hebben. Een woord dat twee keer in een lijst staat, telt één keer. Voor `WORDS[0]` en `WORDS[1]` is dat `2`: *fiets* en *goedkoop*. Vervang de `...`. De check roept `overlap(WORDS[0], WORDS[1])` aan, en daarna eigen gevallen, met een woord dat twee keer voorkomt, zonder gedeelde woorden en met een lege lijst.
        """),
        L.task("Tel de verschillende woorden die twee zinnen delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overlap():
    box_overlap = L.code_box("overlap", lines=3)
    box_overlap
    return (box_overlap,)


@app.cell(hide_code=True)
def check_overlap(box_overlap):
    result_overlap = L.check("overlap", box_overlap.value)
    result_overlap
    return (result_overlap,)


@app.cell(hide_code=True)
def hints_overlap():
    L.hints(
        "overlap",
        [
            "Verzamel in een lijst `both` de woorden die de twee lijsten gemeen hebben, met een list comprehension over `a`. Tel daarna de verschillende, zoals de laatste regel van het uitgewerkte voorbeeld doet.",
            "Een list comprehension kan alleen sommige waarden houden, met een `if` na de `for`. Aan welke voorwaarde moet een woord uit `a` voldoen om in `both` te komen?",
            "`    both = [w for w in a if ___]`",
        ],
        why="`both = [w for w in a if w in b]`, en daarna `return len(set(both))`. `overlap(WORDS[0], WORDS[1])` geeft `2`.",
    )
    return


@app.cell(hide_code=True)
def text_after_overlap():
    mo.md(r"""
    De afbeelding onder de check toont de tekst met jouw aantal bij elke overgang tussen twee zinnen, en de gedeelde woorden ernaast. De eerste zinnen delen steeds een woord: de fiets, huis, de ketting. Waar de storm opduikt, zakt het aantal naar 0, en de laatste zin pakt het dak weer op. Een aantal van 1 of 2 is klein; het programma van Hearst vergeleek blokken van meerdere zinnen, geen losse zinnen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_status():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. status: nieuw of oud

        Schrijf `status(mentions)`: geef een lijst terug met een label voor elke mention in `mentions`: `"new"` als de tekst dat ding voor het eerst noemt, en `"old"` als het al eerder voorkwam. De labels blijven Engels, net als de namen in de code. Voor `flat(MENTIONS)` begint de lijst met `"new"`, `"new"`, `"new"`, `"old"`: de fiets van de tweede zin is oud. Vervang de `...`. De check roept `status(flat(MENTIONS))` aan, en daarna eigen gevallen, met een ding dat drie keer genoemd wordt en met een lege lijst.
        """),
        L.task("Markeer elke mention in een tekst als nieuw of oud."),
    ])
    return


@app.cell(hide_code=True)
def exercise_status():
    box_status = L.code_box("status", lines=10)
    box_status
    return (box_status,)


@app.cell(hide_code=True)
def check_status(box_status):
    result_status = L.check("status", box_status.value)
    result_status
    return (result_status,)


@app.cell(hide_code=True)
def hints_status():
    L.hints(
        "status",
        [
            "Houd twee lijsten bij: `seen` voor de dingen die de tekst tot nu toe noemde, en `out` voor de labels. Loop met een `for` over `mentions`, en geef `out` terug na de loop.",
            "Wanneer is een mention oud? En wat moet er met `seen` gebeuren na een nieuwe?",
            "`        if ___:`",
        ],
        why="`seen = []` en `out = []`; `for m in mentions:` met daaronder `if m in seen:` en daaronder `out.append(\"old\")`; onder `else:` `out.append(\"new\")` en `seen.append(m)`; na de loop `return out`. `status(flat(MENTIONS))` begint met `['new', 'new', 'new', 'old']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_status():
    mo.md(r"""
    De afbeelding onder de check toont elke mention als nieuw of oud, in de tekst zoals geschreven en in een volgorde door elkaar. Zoals geschreven beginnen de meeste zinnen met iets ouds, en eindigen de meeste met iets nieuws, zoals Clark en Haviland beschrijven. *Die nacht* en *De volgende ochtend* zijn nieuw, en zetten eerst de tijd neer. Volgens deze telling is *de ketting* ook nieuw, al verwacht een lezer dat een fiets er een heeft: iets wat een lezer kan afleiden, zoals de chef bij het restaurant. Door elkaar beginnen meer zinnen met iets nieuws, en komt *de storm* vóór *een storm*: de zin neemt een storm als vanzelfsprekend aan die de lezer nog niet kent.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de tekst door elkaar

    Het programma uit de gok vergeleek teksten met versies waarin de zinnen door elkaar stonden. Doe hetzelfde met twee eenvoudige tellingen: de woorden die buren delen, zoals in oefening 1, en de zinnen die met iets ouds beginnen, zoals in oefening 2. Kies een volgorde, en vergelijk die met de tekst zoals geschreven.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_shuffle = mo.ui.slider(1, 12, step=1, value=1, label="Volgorde", show_value=True)
    demo_shuffle
    return (demo_shuffle,)


@app.cell(hide_code=True)
def demo_look(demo_shuffle):
    mo.Html(C.demo_picture(demo_shuffle.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De acht zinnen kunnen in 40.320 volgordes staan. De geschreven volgorde heeft meer gedeelde woorden tussen buren dan 38.434 daarvan, en laat meer zinnen met iets ouds beginnen dan 29.316. Geen van beide tellingen is sluitend: sommige volgordes komen even hoog of hoger uit, en twee zinnen kunnen een woord delen zonder samen te hangen, zoals oefening 4 laat zien. Het programma van Barzilay en Lapata leerde uit paren van oorspronkelijke teksten en teksten door elkaar welke patronen van mentions en rollen bij een oorspronkelijke volgorde horen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ties():
    mo.md(r"""
    ### Oefening 3. Wat verbindt de twee zinnen?

    Elk vetgedrukt paar hieronder hangt samen door één soort band. Zet elk paar bij die soort.
    """)
    return


@app.cell(hide_code=True)
def exercise_ties():
    sort_ties = L.sort("ties")
    sort_ties
    return (sort_ties,)


@app.cell(hide_code=True)
def check_ties(sort_ties):
    result_ties = L.check("ties", sort_ties)
    result_ties
    return (result_ties,)


@app.cell(hide_code=True)
def text_exercise_bank():
    mo.md(r"""
    ### Oefening 4. Gedeeld, en toch los

    Een programma dat gedeelde woorden telt, zoals `overlap`, ziet een band waar een woord terugkomt.
    """)
    return


@app.cell(hide_code=True)
def choice_bank_cell():
    choice_bank = L.choice(
        "bank",
        question="Welk paar deelt een woord, maar hangt niet samen?",
        options={
            "right": "*Sam zat op de bank bij het raam. De bank verhoogde de rente.*",
            "key": "*Sam verloor de sleutel van de schuur. De sleutel lag een week later onder de deurmat.*",
            "cake": "*Sam kocht een taart voor Kim. De taart was nog warm uit de oven.*",
        },
        correct="right",
        why={
            "right": "*Bank* komt terug, dus `overlap` telt een gedeeld woord. Maar de bank bij het raam en de bank die de rente verhoogt, zijn twee verschillende dingen, en niets verbindt de twee zinnen: een band in de woorden, zonder coherentie. Lexicale cohesie is een aanwijzing voor coherentie, geen garantie.",
            "key": "Hier komt de sleutel terug als dezelfde sleutel, en de tweede zin vertelt wat ermee gebeurde: het paar hangt samen. Zoek een woord dat terugkomt met een andere betekenis.",
            "cake": "Hier komt de taart terug als dezelfde taart, en de tweede zin zegt er meer over: het paar hangt samen. Zoek een woord dat terugkomt met een andere betekenis.",
        },
    )
    choice_bank
    return (choice_bank,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl) keek naar wat een spreker bedoelt voorbij de woorden van één uiting. Deze les ging voorbij de losse zin: banden in de woorden (cohesie), een samenhang in betekenis tussen de zinnen (coherentie), en een volgorde binnen elke zin van oud naar nieuw (informatiestructuur). Je telde gedeelde woorden, markeerde mentions als nieuw of oud, deelde soorten banden in en vond een paar dat een woord deelt zonder samen te hangen. Een latere les volgt alle mentions van één ding door een tekst.

    **Een voorbeeld uit het dagelijks leven.** Als een verhaal begint met *een hond*, zegt de volgende zin *de hond*. Zeg je opnieuw *een hond*, dan denkt je luisteraar aan een tweede hond.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bank, result_overlap, result_status, result_ties):
    L.wrap_up(result_overlap, result_status, result_ties, choice_bank)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
