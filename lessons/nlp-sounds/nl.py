# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-sounds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sounds", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_plural():
    quick_plural = L.choice(
        "quick-plural",
        question="Wat is *-en* in *katten*?",
        options={
            "right": "Een morfeem dat het meervoud aangeeft",
            "letter": "Alleen letters, zonder eigen taak",
            "word": "Een eigen woord, zoals in een samenstelling",
        },
        correct="right",
        why={
            "right": "*-en* is het kleinste deel van *katten* met een taak in de grammatica: het maakt het meervoud, door inflectie. Deze les gebruikt zo'n meervoudsuitgang: de *-en* van *honden* laat een klank horen die het enkelvoud *hond* verbergt.",
            "letter": "*-en* heeft een taak: *kat* is er één, *katten* meer dan één. Een deel met een betekenis of een taak is een morfeem. Zie *Delen van woorden* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "word": "*-en* kan niet als woord op zichzelf staan; een samenstelling voegt hele woorden samen, zoals *voordeur*. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_plural
    return (quick_plural,)


@app.cell(hide_code=True)
def quick_check_derive():
    quick_derive = L.choice(
        "quick-derive",
        question="Wat doet *-heid* in *schoonheid*?",
        options={
            "form": "Het maakt een andere vorm van hetzelfde woord",
            "join": "Het voegt twee hele woorden samen tot één nieuw woord",
            "right": "Het maakt een nieuw woord van een bestaand woord",
        },
        correct="right",
        why={
            "right": "*-heid* maakt van het bijvoeglijk naamwoord *schoon* het zelfstandig naamwoord *schoonheid*: derivatie. Morfemen en letters lopen niet altijd gelijk op, zoals de dubbele t van *katten* laat zien. Deze les laat zien dat letters en klanken dat ook niet doen.",
            "form": "Dat doet inflectie, zoals bij *lopen* en *liep*. *Schoonheid* is een nieuw woord, een zelfstandig naamwoord, waar *schoon* een bijvoeglijk naamwoord is. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "join": "*-heid* is geen woord op zichzelf: het is een achtervoegsel. Een samenstelling voegt hele woorden samen, zoals *voordeur*. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_derive
    return (quick_derive,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een programma dat spreekt, heeft letters en moet klanken maken: een navigatie-app die een straatnaam voorleest, of een schermlezer. Een programma dat luistert, heeft klanken en moet letters schrijven. Letters zijn een slechte gids voor klanken. Jurafsky en Martin schrijven in hun hoofdstuk over fonetiek dat de koppeling tussen de letters van het Engels en de spraakklanken vrij ondoorzichtig is: één letter kan in verschillende omgevingen heel verschillende klanken weergeven. Hun voorbeeld: de letter *c* is een k-klank in *cougar* (poema) en een s-klank in *cell* (cel). Dus houdt spraaktechnologie lijsten bij van hoe woorden klinken. Jackson Lee en collega's (2020), die zulke lijsten uit een vrij woordenboek haalden, schrijven dat bijna alle spraaktechnologie steunt op een koppeling tussen de geschreven vorm van woorden en de uitspraak, als een reeks spraakklanken. Voor een woord dat niet in de lijst staat, leert een model letters om te zetten in klanken: grapheme-to-phoneme-conversie. Voor allebei heb je een manier nodig om klanken op te schrijven, en een idee van welke verschillen tussen klanken in een taal ertoe doen.

    **Een voorbeeld uit het dagelijks leven.** Je navigatie-app leest de naam van een Nederlandse straat voor alsof het Engels is. Of je schrijft *hond* met een d, en zegt het woord met een t.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De taalkundige Ian Maddieson telde voor de World Atlas of Language Structures de medeklinkers van honderden talen. Het Engels heeft er 24; daarover zijn taalkundigen het in grote lijnen eens. Hoeveel heeft de taal in het midden van de telling, met de helft van de talen eronder en de helft erboven?",
        options={
            "few": "Ongeveer 10, nog niet de helft van het Engels",
            "right": "Ongeveer 21, iets minder dan het Engels",
            "many": "Ongeveer 45, bijna twee keer zoveel als het Engels",
        },
        reveal="**Ongeveer 21, iets minder dan het Engels**: de mediaan van 562 talen in het hoofdstuk van Maddieson in de World Atlas (2013); het vaakst komt 22 voor, en het gemiddelde is 22,7. De spreiding loopt van 6 medeklinkers in het Rotokas, een taal van Papoea-Nieuw-Guinea, tot 122 in het !Xóõ, een taal van Botswana, vooral omdat die taal heel veel verschillende klikklanken heeft. Over de 3.020 klankinventarissen van de database PHOIBLE (versie 2.0, 2019) telt de course een mediaan van 22 medeklinkers, en 130 voor het !Xóõ: een telling hangt af van hoe een taalkundige een taal analyseert. Zelfs voor het Engels, voegt Maddieson toe, zouden sommige taalkundigen op een of twee meer of minder uitkomen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Klanken, geen letters

    **Fonetiek** bestudeert de klanken van spraak. Jurafsky en Martin omschrijven het vak als de studie van de spraakklanken in de talen van de wereld: hoe een spraakklank in mond en keel gemaakt wordt, hoe die als geluid klinkt, en hoe je die kunt digitaliseren en verwerken. Fonetici schrijven een **spraakklank** tussen rechte haken: [k].

    Letters en klanken gaan elk hun eigen weg, in het Engels meer dan in veel andere talen: het Spaans is volgens Jurafsky en Martin veel doorzichtiger. De Engelse Wikipedia telt minstens acht uitspraken van *ough* in het Noord-Amerikaans Engels en negen in het Brits Engels, zoals in *tough*, *cough*, *though* en *through*. Ook het Nederlands kent geen een-op-eenrelatie tussen klanken en letters, zo merkt de *Algemene Nederlandse Spraakkunst* (e-ANS), de naslaggrammatica van het Nederlands, op. Een bekende grap spelt *fish* (vis) als *ghoti*: *gh* zoals in *tough*, *o* zoals in *women*, *ti* zoals in *nation*. Die grap wordt vaak aan George Bernard Shaw toegeschreven, maar staat nergens in het werk van Shaw; het eerste bevestigde gebruik staat in een brief uit 1855, het jaar vóór de geboorte van Shaw.
    """)
    return


@app.cell(hide_code=True)
def text_idea_phonemes():
    mo.md(r"""
    ### Klanken die een woord veranderen

    Sommige verschillen tussen klanken veranderen een woord, en andere niet. De International Phonetic Association schreef in de beginselen van 1912 dat je moet vaststellen welke klanken in een taal onderscheidend zijn: de klanken die de betekenis van woorden kunnen veranderen als je de ene voor de andere gebruikt. Zo'n klank is een **foneem**, geschreven tussen schuine strepen: /p/. Zet /b/ op de plaats van /p/, en het Engelse *pat* (tikje) wordt *bat* (knuppel). Twee woorden die in één klank verschillen, op dezelfde plaats, vormen een **minimaal paar**: het Nederlands heeft *pak* en *bak*, en *tak* en *dak*. *Kat* en *kast* zijn geen minimaal paar, want *kast* heeft een klank meer.

    Andere verschillen veranderen nooit een woord. De e-ANS geeft een Nederlands voorbeeld: de i van *bier* is langer dan de i van *biet*, omdat er een r volgt. Het zijn allebei vormen van het foneem /i/: twee **allofonen** ervan, en de klank ernaast bepaalt welke je hoort. De beginselen van de Association noemen fonemen in de versie van 1949 families van verwante klanken die in een taal tellen alsof het er één is.

    **Fonologie** bestudeert wat een taal met klanken doet: welke verschillen tellen, en welke regels klanken volgen. De e-ANS trekt de grens zo: fonologie gaat over de rol van klanken in het systeem van een taal, en fonetiek over hoe klanken gemaakt en waargenomen worden, en welke natuurkundige eigenschappen spraakklanken hebben. Welke verschillen tellen, hangt af van de taal, zoals het laatste stuk van *Het idee* laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_idea_ipa():
    mo.md(r"""
    ### Eén teken per klank

    De International Phonetic Association werd in 1886 in Parijs opgericht, door Paul Passy en andere taalkundigen en taaldocenten. In 1888 stelde de vereniging een alfabet op voor de uitspraak van elke taal: het **IPA**, kort voor International Phonetic Alphabet, in het Nederlands het Internationaal Fonetisch Alfabet. Een van de beginselen van 1912: een typische enkele klank krijgt zo veel mogelijk één letter, zonder diakritische tekens. De kaart van het alfabet is in 2015 voor het laatst herzien, en verschijnt sinds 2025 elk jaar opnieuw, vrij te delen onder CC BY-SA 4.0. Op de uitgave van 2026 telt de course 59 tekens in de tabel van medeklinkers die met lucht uit de longen gemaakt worden, en 28 klinkers. Er komen meer tekens en tekentjes bij, bijvoorbeeld voor klikklanken, of de kleine ʰ voor een pufje lucht.

    Computers gebruikten lang eenvoudiger alfabetten. Het hoofdstuk van Jurafsky en Martin schrijft klanken in het ARPAbet, een eenvoudig fonetisch alfabet met alleen ASCII-tekens, voor een deel van het IPA: de klanken van het Amerikaans Engels. Het CMU Pronouncing Dictionary van Carnegie Mellon University gebruikt een versie ervan met 39 fonemen, voor zo'n 134.000 woordvormen, schrijven Jurafsky en Martin. In het woordenboek is *pat* P AE1 T, waarbij de 1 de klinker met de klemtoon aangeeft. CELEX, een lexicon van het Engels, het Nederlands en het Duits, is niet vrij te delen. WikiPron, het werk van Lee en collega's, verzamelt de uitspraken uit Wiktionary, het vrije woordenboek waar iedereen aan kan schrijven: versie 2.2.0, van juli 2026, bevat 5.225.170 uitspraken in 348 talen, en de Nederlandse hoofdlijst telt 58.539 regels. Ook de woorden van deze les komen uit Wiktionary.

    Ook in spraakmodellen duiken klanken op. Jurafsky en Martin beschrijven in hun hoofdstuk over spraakherkenning de taak als het omzetten van geluidsgolven in reeksen grafemen: van klank rechtstreeks naar letters. Toch kan een model dat zonder labels van audio leert, eigen eenheden vinden die op klanken lijken. Over wav2vec 2.0 schrijven Alexei Baevski en collega's (2020) dat veel van de geleerde discrete eenheden zich lijken te specialiseren in bepaalde spraakklanken.
    """)
    return


@app.cell(hide_code=True)
def text_idea_devoicing():
    mo.md(r"""
    ### Klanken die veranderen met hun plaats

    Een klank kan veranderen met de plaats in een woord. In het Nederlands zijn plofklanken zoals *d* en wrijfklanken zoals *z*, samen de **obstruenten**, aan het eind van een lettergreep in principe stemloos, zegt de e-ANS: **finale verscherping**, ook wel eindklankverscherping genoemd. Het Taalportaal, een beschrijving van de Nederlandse grammatica in het Engels, zegt het zo: stemhebbende obstruenten komen in het Nederlands meestal niet aan het eind van een lettergreep voor, en een obstruent die daar fonologisch stemhebbend is, wordt stemloos. Fonologisch stemhebbend wil zeggen: stemhebbend in het systeem van de taal, zoals de d van *hond*, die je in *honden* hoort. Een **stemhebbende** klank maak je met trillende stembanden, een **stemloze** zonder: leg je vingers op je keel en zeg *zzz*, en dan *sss*. De afbeelding toont de vijf paren uit de tabel die de e-ANS bij de regel geeft.

    Het morfeem *hond* heeft volgens de e-ANS twee klankvormen: [hɔnt] in *hond*, en [hɔnd] in *honden*. In *honden* begint de d de tweede lettergreep, dus blijft die stemhebbend. De regel werkt aan het eind van elke lettergreep, ook midden in een woord: *handrem* heeft een t-klank in het midden. En in verbonden spraak kan een andere regel voorgaan: in *huisdeur* kan de s van *huis* vóór de stemhebbende d toch weer als z klinken, merkt de e-ANS op. Het Nederlands, voegt de e-ANS toe, heeft wel woorden als *kat* [kɑt], maar geen woord als het Engelse *bed* [bɛd]. Het Engels houdt *bad* (slecht) en *bat* (knuppel) uit elkaar, al worden stemhebbende obstruenten aan het eind van een woord daar tot op zekere hoogte stemloos, volgens de Engelse Wikipedia. Het Duits in Duitsland kent de regel ook: *Rad* (wiel) en *Rat* (raad) klinken hetzelfde.

    De Nederlandse spelling verbergt de regel bij b, d en g, de letter voor ɣ. De *Woordenlijst*, de officiële woordenlijst van het Nederlands, verklaart dat met het beginsel van de gelijkvormigheid: je spelt een woord zo veel mogelijk op dezelfde manier, dus *bloed* altijd met een d, al hoor je een t. Maar het Nederlands schrijft geen v of z aan het eind van een lettergreep: *raaf* naast *raven*, en *prijs* naast *prijzen*.
    """)
    return


@app.cell(hide_code=True)
def rule_picture():
    mo.Html(C.rule_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **fonetiek** | de studie van spraakklanken: hoe een klank gemaakt wordt, als geluid reist en gehoord wordt |
        | **spraakklank** | een klank van gesproken taal, tussen rechte haken geschreven: [t] |
        | **foneem** | een klank die in een taal een woord kan veranderen, tussen schuine strepen geschreven: /t/ |
        | **minimaal paar** | twee woorden die in één klank verschillen, op dezelfde plaats, zoals *pak* en *bak* |
        | **allofoon** | een van de klanken die als hetzelfde foneem tellen, zoals de twee i's van *bier* en *biet* |
        | **fonologie** | de studie van wat een taal met klanken doet: welke verschillen tellen, en welke regels klanken volgen |
        | **IPA** | het Internationaal Fonetisch Alfabet: zo veel mogelijk één teken per klank, voor elke taal |
        | **stemhebbend, stemloos** | gemaakt met of zonder trillende stembanden: *z* is stemhebbend, *s* stemloos |
        | **obstruent** | een klank waarbij de weg voor de lucht dichtgaat of nauw wordt: een plofklank zoals *p* of *d*, of een wrijfklank zoals *f* of *z* |
        | **finale verscherping** | de Nederlandse regel die een stemhebbende obstruent aan het eind van een lettergreep stemloos maakt: *hond* eindigt op een t-klank |
        | **ɔ ɑ ɪ ɛ u ə** | klinkers in het IPA: de klinker van *hond*, *kat*, *rib*, *pet* en *hoed*, en de e van de uitgang in *honden* |
        | **aː, œy̯** | de lange aa van *zaag*, waarbij ː een lange klinker aangeeft, en de ui van *druif* |
        | **ɣ, x** | de g van *zagen*, en de stemloze g aan het eind van *zaag*, zoals de ch van *lach* |
        | **æ ʃ θ ŋ** | in het Engels: de a van *pat*, de sh van *ship*, de th van *think* en de n van *think* |
        | `s[len(s) - 1]` | de laatste waarde van een lijst `s`: de plaatsen lopen van 0 tot en met `len(s) - 1` |
        """),
        L.watch_out("Finale verscherping gaat over klank, niet over spelling: *hond* houdt de d op papier. En de regel geldt voor het Standaardnederlands, niet voor elk dialect: het dialect van het eiland Ameland kent volgens de Engelse Wikipedia geen finale verscherping."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf elk woord als een lijst van de klanken, één IPA-teken per klank, naar een uitspraakwoordenboek zoals Wiktionary.
    2. Wil je weten of twee klanken verschillende fonemen zijn, zoek dan een minimaal paar: twee even lange woorden die alleen in die twee klanken verschillen.
    3. Voor een regel zoals finale verscherping neem je een vorm van het woord waarin de klank niet aan het eind staat, zoals het meervoud, en haal je de uitgang weg.
    4. Pas de regel toe op de laatste klank, en vergelijk de uitkomst met het woordenboek.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat de woorden van de les als lijsten klanken, `DUTCH` en `ENGLISH`, naar de Engelse Wiktionary van 22 november 2026. Wiktionary schrijft *krabben* als /ˈkrɑ.bə(n)/; de lijsten laten de schuine strepen, het klemtoonteken ˈ, de punt tussen de lettergrepen en de (n) aan het eind weg. Die n spreken veel sprekers niet uit. Een lange klinker houdt het lengteteken ː, zoals in *zagen*, en de klinker van *druif*, œy̯, is één klank en één waarde. Wiktionary schrijft de Nederlandse h als ɦ, een stemhebbende h, waar de e-ANS h schrijft. `PAIRS` bevat tien enkelvouden met hun meervoud, en `VOICELESS` de vijf stemhebbende obstruenten met hun stemloze partners. De tweede cel definieert `stem`, een functie die de laatste klank van een meervoud weghaalt, de uitgang ə, en bekijkt een paar waarden.
    """)
    return


@app.cell
def words_cell():
    DUTCH = {
        "hond": ["ɦ", "ɔ", "n", "t"], "honden": ["ɦ", "ɔ", "n", "d", "ə"],
        "hand": ["ɦ", "ɑ", "n", "t"], "handen": ["ɦ", "ɑ", "n", "d", "ə"],
        "hoed": ["ɦ", "u", "t"], "hoeden": ["ɦ", "u", "d", "ə"],
        "krab": ["k", "r", "ɑ", "p"], "krabben": ["k", "r", "ɑ", "b", "ə"],
        "rib": ["r", "ɪ", "p"], "ribben": ["r", "ɪ", "b", "ə"],
        "druif": ["d", "r", "œy̯", "f"], "druiven": ["d", "r", "œy̯", "v", "ə"],
        "zaag": ["z", "aː", "x"], "zagen": ["z", "aː", "ɣ", "ə"],
        "kat": ["k", "ɑ", "t"], "katten": ["k", "ɑ", "t", "ə"],
        "boek": ["b", "u", "k"], "boeken": ["b", "u", "k", "ə"],
        "kant": ["k", "ɑ", "n", "t"], "kanten": ["k", "ɑ", "n", "t", "ə"],
        "bed": ["b", "ɛ", "t"], "pet": ["p", "ɛ", "t"],
    }
    ENGLISH = {
        "pat": ["p", "æ", "t"], "bat": ["b", "æ", "t"], "pin": ["p", "ɪ", "n"], "bin": ["b", "ɪ", "n"],
        "spin": ["s", "p", "ɪ", "n"], "ship": ["ʃ", "ɪ", "p"], "sheep": ["ʃ", "iː", "p"],
        "think": ["θ", "ɪ", "ŋ", "k"], "sink": ["s", "ɪ", "ŋ", "k"],
    }
    PAIRS = [("hond", "honden"), ("hand", "handen"), ("hoed", "hoeden"), ("krab", "krabben"), ("rib", "ribben"),
             ("druif", "druiven"), ("zaag", "zagen"), ("kat", "katten"), ("boek", "boeken"), ("kant", "kanten")]
    VOICELESS = {"b": "p", "d": "t", "v": "f", "z": "s", "ɣ": "x"}
    return DUTCH, ENGLISH, PAIRS, VOICELESS


@app.cell
def example_look(DUTCH, VOICELESS):
    def stem(plural):
        return plural[:len(plural) - 1]

    print(DUTCH["honden"], stem(DUTCH["honden"]))
    print(DUTCH["hond"], len(DUTCH["hond"]))
    print(DUTCH["kat"][len(DUTCH["kat"]) - 1])
    print(VOICELESS["d"], "d" in VOICELESS, "t" in VOICELESS)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Honden* zonder uitgang geeft de klanken ɦ ɔ n d: de d die je in het meervoud hoort. Het enkelvoud *hond* heeft op die plaats een t. `len` telt waarden, dus hier klanken, geen letters. `DUTCH["kat"][len(DUTCH["kat"]) - 1]` is de laatste klank van *kat*, een t. `VOICELESS["d"]` geeft `t`, en `in` vraagt of een klank een sleutel van de dictionary is: *t* is dat niet, want die klank is al stemloos.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimal():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. minimal_pair: één klank verschil

        Schrijf `minimal_pair(a, b)`: `a` en `b` zijn twee woorden als lijsten klanken. Geef `True` terug als de twee woorden een minimaal paar vormen: evenveel klanken, en op één plaats een andere klank. Geef anders `False` terug. Voor `DUTCH["hond"]` en `DUTCH["hand"]` is dat `True`: alleen de klinker verschilt. Vervang de `...`. De check roept `minimal_pair(DUTCH["hond"], DUTCH["hand"])` aan, en daarna eigen gevallen: woorden van verschillende lengte, woorden die op twee plaatsen verschillen, en een woord met zichzelf.
        """),
        L.task("Bepaal of twee woorden, als lijsten klanken, een minimaal paar vormen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimal():
    box_minimal = L.code_box("minimal_pair", lines=5)
    box_minimal
    return (box_minimal,)


@app.cell(hide_code=True)
def check_minimal(box_minimal):
    result_minimal = L.check("minimal_pair", box_minimal.value)
    result_minimal
    return (result_minimal,)


@app.cell(hide_code=True)
def hints_minimal():
    L.hints(
        "minimal_pair",
        [
            "Twee woorden van verschillende lengte zijn geen minimaal paar: geef daarvoor eerst `False` terug, met een `if`. Maak daarna een lijst `diff` met een list comprehension over de plaatsen `i` in `range(len(a))`, en kijk naar de lengte van `diff`.",
            "Welke plaatsen horen in `diff`? En hoe lang is `diff` bij een minimaal paar?",
            "`    diff = [i for i in range(len(a)) if ___]`",
        ],
        why="`if len(a) != len(b):` met `return False` eronder, dan `diff = [i for i in range(len(a)) if a[i] != b[i]]` en `return len(diff) == 1`. `minimal_pair(DUTCH[\"hond\"], DUTCH[\"hand\"])` geeft `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_minimal():
    mo.md(r"""
    De afbeelding onder de check geeft jouw antwoorden voor tien paren. Zes zijn minimale paren, zoals *hond* en *hand*, of het Engelse *ship* en *sheep*. *Kat* en *kant*, *hond* en *honden*, en *pin* en *spin* verschillen in lengte; *kat* en *pet* verschillen op twee plaatsen. *Bed* en *pet* verschillen alleen in de eerste klank: *bed* eindigt op een t-klank, en oefening 2 laat zien waarom.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_devoice():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. devoice: de regel in code

        Schrijf `devoice(sounds, voiceless)`: geef een lijst terug met dezelfde klanken als `sounds`, maar met de laatste stemloos gemaakt als die klank een sleutel is van de dictionary `voiceless`. Een laatste klank die geen sleutel is, blijft zoals die is. Voor `stem(DUTCH["honden"])`, de klanken van *honden* zonder de uitgang, is dat `['ɦ', 'ɔ', 'n', 't']`: de klanken van *hond*. Vervang de `...`. De check roept `devoice(stem(DUTCH["honden"]), VOICELESS)` aan, en daarna eigen gevallen: een stemhebbende klank aan het begin, een laatste klank die al stemloos is, een woord van één klank, en een andere dictionary.
        """),
        L.task("Maak de laatste klank van een woord stemloos, zoals de Nederlandse regel doet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_devoice():
    box_devoice = L.code_box("devoice", lines=7)
    box_devoice
    return (box_devoice,)


@app.cell(hide_code=True)
def check_devoice(box_devoice):
    result_devoice = L.check("devoice", box_devoice.value)
    result_devoice
    return (result_devoice,)


@app.cell(hide_code=True)
def hints_devoice():
    L.hints(
        "devoice",
        [
            "Maak `out`, alle klanken behalve de laatste, met een slice, en `last`, de laatste klank, zoals in het uitgewerkte voorbeeld. Is `last` een sleutel van `voiceless`, geef `last` dan een nieuwe waarde. Voeg daarna `last` toe aan `out`, en geef `out` terug.",
            "Welke klank moet de plaats van `last` innemen als `last` een sleutel is?",
            "`        last = ___`",
        ],
        why="`out = sounds[:len(sounds) - 1]` en `last = sounds[len(sounds) - 1]`; `if last in voiceless:` met `last = voiceless[last]` eronder; dan `out.append(last)` en `return out`. `devoice(stem(DUTCH[\"honden\"]), VOICELESS)` geeft `['ɦ', 'ɔ', 'n', 't']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_devoice():
    mo.md(r"""
    De afbeelding onder de check neemt elk meervoud uit `PAIRS`, haalt de uitgang weg, en laat jouw `devoice` los op wat overblijft. Bij de zeven woorden die vóór de uitgang op d, b, v of ɣ eindigen, geeft jouw regel het enkelvoud uit het woordenboek; de andere drie eindigen al op een stemloze klank. Je functie kijkt alleen naar het eind van het woord. De regel zelf werkt aan het eind van elke lettergreep, zoals in *handrem*, en een stemhebbende klank erna kan de regel opzijzetten, zoals in *huisdeur*.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: letters en klanken

    Kies een woord uit de twee lijsten van de les, en vergelijk de letters met de klanken, en met de vorm zoals Wiktionary het woord schrijft.
    """)
    return


@app.cell(hide_code=True)
def demo_controls(DUTCH, ENGLISH):
    demo_word = mo.ui.dropdown(options=list(DUTCH) + list(ENGLISH), value="krabben", label="Een woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Letters en klanken komen niet één op één overeen. *Ship* heeft vier letters en drie klanken, want *sh* is één klank. In *krabben* staan twee letters b voor één b-klank, en in *druiven* staan twee letters, *ui*, voor één klinker. *Hond* heeft evenveel letters als klanken, en toch is de laatste letter niet de laatste klank.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fields():
    mo.md(r"""
    ### Oefening 3. Klank, systeem of spelling?

    Elke vetgedrukte zin hieronder gaat over hoe een klank gemaakt wordt, over de klanken die een taal gebruikt om woorden uit elkaar te houden, of over hoe een taal de klanken opschrijft. Zet elke zin bij de groep waar die zin over gaat.
    """)
    return


@app.cell(hide_code=True)
def exercise_fields():
    sort_fields = L.sort("fields")
    sort_fields
    return (sort_fields,)


@app.cell(hide_code=True)
def check_fields(sort_fields):
    result_fields = L.check("fields", sort_fields)
    result_fields
    return (result_fields,)


@app.cell(hide_code=True)
def text_exercise_allophone():
    mo.md(r"""
    ### Oefening 4. Eén foneem of twee?

    In het Engels begint *pin* (speld) met [pʰ]: een p met een klein pufje lucht erachter. In *spin* (draaien) komt de p zonder pufje: [p].
    """)
    return


@app.cell(hide_code=True)
def choice_allophone_cell():
    choice_allophone = L.choice(
        "allophone",
        question="Waarom tellen de twee in het Engels als één foneem /p/?",
        options={
            "letter": "Allebei geschreven met de letter *p*",
            "right": "Verwisselen verandert geen Engels woord",
            "puff": "Het pufje lucht is te kort om het te kunnen meten",
        },
        correct="right",
        why={
            "right": "In het Engels onderscheiden de twee nooit woorden: met de ene of de andere uitgesproken, blijft een woord hetzelfde woord. Het zijn allofonen van /p/, en de plaats bepaalt welke je hoort, net als bij de twee i's van *bier* en *biet*. De Engelse Wikipedia geeft dit voorbeeld, en de Engelse Wiktionary schrijft *pin* met dat detail als [ˈpʰɪn].",
            "letter": "Letters zijn hier geen gids: één letter kan voor twee fonemen staan, zoals de *c* van *cell* en *cougar* laat zien. Kijk wat er met een woord gebeurt als je de twee klanken verwisselt.",
            "puff": "Het pufje is te horen en te meten; het IPA schrijft het met een kleine ʰ. Of twee klanken één foneem zijn, hangt af van wat de taal ermee doet. Kijk wat er met een woord gebeurt als je de twee verwisselt.",
        },
    )
    choice_allophone
    return (choice_allophone,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl) knipte woorden in morfemen, langs de letters. Deze les keek naar de klanken onder de letters: fonetiek bestudeert de klanken zelf, en fonologie wat een taal ermee doet. Een foneem is een klank die een woord kan veranderen, zoals een minimaal paar laat zien; het IPA schrijft elke klank zo veel mogelijk met één teken; en een regel zoals finale verscherping verandert een klank door de plaats in het woord, zodat het morfeem *hond* twee klankvormen heeft. Je testte minimale paren, schreef de regel in code, vergeleek letters met klanken, en hield fonetiek, fonologie en spelling uit elkaar. Spraakherkenning en tekst-naar-spraak beginnen bij klanken als deze: het domein van de course over spraak en audio komt erop terug.

    **Een voorbeeld uit het dagelijks leven.** *Hart* en *hard* klinken in het Nederlands hetzelfde: zeg de twee woorden maar hardop. Welke van de twee bedoeld is, hoor je aan de zin, en lees je aan de spelling.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_allophone, result_devoice, result_fields, result_minimal):
    L.wrap_up(result_minimal, result_devoice, result_fields, choice_allophone)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
