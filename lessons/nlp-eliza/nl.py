# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-eliza. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-eliza", "nl")
    C = L.checks   # ELIZA met de scripts, het gesprek, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl), en één over [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_findall():
    quick_findall = L.choice(
        "quick-findall",
        question='In *Patronen vinden: reguliere expressies* zocht je met patronen die een groep hebben. Wat geeft `re.findall(r"ik ben ([a-z]+)", "ik ben moe en ik ben verdrietig")`?',
        options={
            "whole": '`["ik ben moe", "ik ben verdrietig"]`: elke match, van de *i* tot de laatste letter',
            "right": '`["moe", "verdrietig"]`: wat de groep matchte, elke keer',
            "first": '`["moe"]`: het zoeken stopt bij de eerste match',
            "none": "`[]`: door de spatie na *ben* matcht `[a-z]+` helemaal niets",
        },
        correct="right",
        why={
            "right": "Met een groep in het patroon geeft `re.findall` wat de groep matchte, één keer voor elke match: de letters na *ik ben*. De regels van ELIZA werken in dezelfde geest: een patroon rond een woord, en de woorden die het patroon vangt, komen terug in het antwoord.",
            "whole": "Het patroon heeft een groep rond `[a-z]+`. Wat geeft `re.findall` als het patroon een groep heeft? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "first": "`re.search` geeft de eerste match. Wat geeft `re.findall`, en hoe vaak staat *ik ben* in de tekst? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "none": "De spatie in het patroon matcht de spatie in de tekst, en `[a-z]+` de letters erna. Is er een match? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
        },
    )
    quick_findall
    return (quick_findall,)


@app.cell(hide_code=True)
def quick_check_pair():
    quick_pair = L.choice(
        "quick-pair",
        question="In *Hoe een gesprek werkt* vroeg Sam aan Kim naar de laatste trein. Sam vraagt: *Hoe laat gaat de laatste trein?* Welke beurt van Kim is het tweede deel van het paar dat de vraag van Sam begint?",
        options={
            "back": "*Mhm.*: het laat zien dat Kim de vraag hoorde",
            "right": "*Om 23.40 uur.*: het geeft Sam waar de vraag om vroeg",
            "check": "*De laatste trein?*: het controleert wat Sam bedoelde, voordat Kim verdergaat",
            "new": "*Ga je mee?*",
        },
        correct="right",
        why={
            "right": "Een vraag maakt een antwoord verwacht als volgende beurt, en *Om 23.40 uur.* is dat antwoord. ELIZA neemt altijd de beurt na de gebruiker, en veel antwoorden van ELIZA zijn vragen, zodat de gebruiker de volgende beurt moet nemen.",
            "back": "*Mhm* laat de ander verder praten, maar Sam vroeg iets. Wat maakt een vraag verwacht? Zie [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl).",
            "check": "*De laatste trein?* vraagt Sam om een bevestiging, voordat Kim antwoordt. Beantwoordt deze beurt de vraag? Zie [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl).",
            "new": "Een vraag als reactie begint een eigen paar. Welke beurt beantwoordt de vraag van Sam? Zie [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl).",
        },
    )
    quick_pair
    return (quick_pair,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In januari 1966 beschreef Joseph Weizenbaum van MIT een programma waarmee mensen konden praten door te typen op een schrijfmachine die met een computer verbonden was: ELIZA. Een gesprek in het paper begint zo, hier in het Nederlands vertaald; de zinnen in hoofdletters zijn van ELIZA:

    > Mannen zijn allemaal hetzelfde.
    > OP WELKE MANIER
    > Mannen vallen ons altijd lastig over het een of ander.
    > KUN JE EEN BEPAALD VOORBEELD BEDENKEN
    > Nou, mijn vriend liet me hierheen komen.
    > JOUW VRIEND LIET JOU HIERHEEN KOMEN

    ELIZA begreep er niets van. In de eerste zinnen van het paper legde Weizenbaum uit hoe het werkt: het programma ontleedt elke getypte zin met regels die in werking treden door sleutelwoorden in de tekst, en bouwt een antwoord met regels die bij de gekozen ontledingsregel horen. Toch schreef Weizenbaum dat sommige proefpersonen er nauwelijks van te overtuigen waren dat ELIZA *geen* mens is. In 1976 vertelde Weizenbaum in het boek *Computer Power and Human Reason* hoe snel en hoe diep mensen die met het programma praatten, er emotioneel bij betrokken raakten. Weizenbaums eigen secretaris had maandenlang gezien hoe het programma geschreven werd, en vroeg Weizenbaum na een paar zinnen met het programma om de kamer uit te gaan.

    De chatmodellen van nu schrijven een antwoord met een neural network dat getraind is op enorme hoeveelheden tekst, niet met een paar honderd regels. Wat ELIZA liet zien, geldt nog steeds: een programma dat in vloeiende zinnen antwoordt, lijkt te begrijpen. In deze les bouw je de stappen van ELIZA zelf, en laat je ELIZA praten met een Nederlands script van de course, gebouwd zoals de originele regels van 1965, die in 2021 zijn teruggevonden.

    **Een voorbeeld uit het dagelijks leven.** Het chatvenster van een webshop antwoordt *Vervelend dat er iets mis is met je bestelling* op elk bericht met het woord *bestelling* erin, ook op *Ik wil graag nog een bestelling plaatsen*.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_mirror = L.guess(
        "guess",
        question="Verderop beantwoordt ELIZA alle 15 regels van het gesprek, met het Nederlandse script van deze les. Sommige antwoorden herhalen een stuk van wat de gebruiker typte, zoals *ALLEMAAL HETZELFDE* in *DACHT JE DAT MANNEN MISSCHIEN NIET ALLEMAAL HETZELFDE ZIJN*; andere zijn vaste zinnen uit het script, zoals *KUN JE EEN BEPAALD VOORBEELD BEDENKEN*. Hoeveel van de 15 antwoorden herhalen de gebruiker?",
        options={
            "few": "3: ELIZA antwoordt vooral met zinnen die klaarstaan",
            "half": "7, en 8 zijn vaste zinnen",
            "right": "10, en 5 zijn vaste zinnen",
            "all": "Alle 15",
        },
        reveal="10 van de 15. De antwoorden 1, 3 tot en met 6 en 10 tot en met 13 zetten een deel van de zin van de gebruiker in het antwoord, en antwoord 15 haalt regel 3 van de gebruiker terug. De andere vijf zijn vaste zinnen: *KUN JE EEN BEPAALD VOORBEELD BEDENKEN*, *JE KLINKT NIET HELEMAAL ZEKER*, *VERTEL ME MEER OVER JE FAMILIE*, *WIE ANDERS IN JE FAMILIE* en *ECHT , IEDEREEN*. Het programma van 1965 komt met het Engelse gesprek op hetzelfde aantal uit. Je eigen woorden terughoren, omgedraaid, is voor een groot deel waarom ELIZA lijkt te luisteren.",
    )
    guess_mirror
    return (guess_mirror,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **chatbot** is een programma waarmee je in gewone taal een gesprek kunt voeren. De regels van ELIZA staan niet in het programma, maar in een **script**; Weizenbaum benadrukte dat een script uit gegevens bestaat, en geen deel is van het programma zelf. Het bekendste script, DOCTOR, laat ELIZA antwoorden als een psychotherapeut, die naar alles kan vragen zonder er iets van te weten. Deze les gebruikt een Nederlands script van de course, met dezelfde opbouw als DOCTOR. Voor elke regel die de gebruiker typt, doorloopt het programma deze stappen:

        1. **Woorden.** De invoer wordt een lijst woorden in hoofdletters; een punt en een komma worden elk een eigen woord.
        2. **Eén deel.** Bij een punt, een komma of *maar* laat ELIZA het deel dat eraan voorafgaat weg als daar geen sleutelwoord in staat, en anders de rest. Wat overblijft, is het eerste deel met een sleutelwoord, of anders het laatste deel. Het programma van 1965 knipt bij het Engelse *but*; de course voegt *maar* toe voor het Nederlands.
        3. **Omwisselen.** Een woord van het script kan een vervanger hebben: `(MIJ = JOU)` wisselt *mij* om voor *jou*. Het Nederlandse script wisselt *ik* en *je* om, *mijn* en *jouw*, *ben* en *bent*, en *heb* en *hebt*, zodat ELIZA een zin kan teruggeven.
        4. **Het sleutelwoord.** Een woord van het script met eigen regels voor een antwoord is een **sleutelwoord**. Een paar sleutelwoorden hebben een **rang**: *altijd* 1, *iedereen* en *niemand* 2, *als* 3; de andere hebben rang 0. Het sleutelwoord met de hoogste rang wint, bij gelijke rang het eerste.
        5. **Ontleden.** Elk sleutelwoord heeft **ontledingsregels**, die ELIZA op volgorde probeert, geschreven met de omgewisselde woorden. De 0 in een ontledingsregel staat volgens Weizenbaum voor een onbepaald aantal woorden. Zo past `(0 JOUW 0)` op elke tekst met *JOUW* erin, en knipt de regel de tekst in drie delen: de woorden tot *JOUW*, *JOUW* zelf, en de woorden erna. Een `0` die niet aan het einde staat, neemt zo weinig mogelijk woorden; de laatste neemt de rest. Een schuine streep tussen haakjes, `(/FAMILIE)`, past op elk woord dat het script als familie markeert, zoals *MOEDER*, en `(=WAT)` stuurt ELIZA door naar de regels van *WAT*.
        6. **Opbouwen.** Elke ontledingsregel heeft **opbouwregels**: `(JOUW 3)` geeft *JOUW*, gevolgd door de woorden van deel 3. De opbouwregels van één ontledingsregel wisselen elkaar af; Weizenbaum schreef dat ELIZA zo alle opbouwregels van een ontledingsregel langsgaat, voordat er één herhaald wordt.
        7. **Geen sleutelwoord.** ELIZA zegt dan een zin van het speciale sleutelwoord NONE, zoals *GA VERDER*, of een herinnering. Elke keer dat *MIJN* het sleutelwoord is, maakt een van vier geheugenregels van de tekst een zin voor later, zoals *EERDER ZEI JE: JOUW ...*; een hash van het laatste woord bepaalt welke regel. Heeft de invoer geen sleutelwoord en staat een teller in het programma op 4 (de teller telt 1, 2, 3, 4 en begint dan opnieuw, één stap per getypte regel), dan zegt ELIZA de oudste herinnering.

        **Het programma van 1965.** Tientallen jaren was alleen het paper bekend. In 2021 vonden Jeff Shrager en archivaris Myles Crowley een uitdraai van het programma in de archieven van Weizenbaum bij MIT, met het script DOCTOR erbij, en de erven van Weizenbaum gaven de uitdraai vrij onder CC0, zodat iedereen het programma mag gebruiken. Anthony Hay en Arthur Schwarz typten de code over, en kregen het programma met Rupert Lane weer aan de praat op CTSS, het timesharingsysteem van MIT in de jaren zestig, in een emulator van de IBM 7094-computer van toen. De run van dit drietal met het gesprek uit het paper is de maat voor de Engelse les: de Python-versie van de course geeft dezelfde 15 antwoorden. Het script van 1965 wijkt een beetje af van het paper: op *Men are all alike.* antwoordt het *DID YOU THINK THEY MIGHT NOT BE ALL ALIKE*, niet *IN WHAT WAY*.

        **Andere scripts, andere chatbots.** Weizenbaum schreef dat er naast Engelse ook Welshe en Duitse scripts waren. Het Nederlandse script van deze les heeft de course zelf geschreven, en het Nederlands maakt het lastiger dan het Engels. In een bijzin staat het werkwoord achteraan. Een opbouwregel zet het werkwoord dan achter het deel van de gebruiker, zoals `(IS HET OMDAT JE 4 BENT DAT JE NAAR MIJ TOE KOMT)`, of zet de woorden van de gebruiker na een dubbele punt, zoals `(JE ZEGT: 1)`. En *je* is zowel *jij* als *jouw*: *Je moeder is lief.* wordt *IK MOEDER IS LIEF*, en ELIZA antwoordt *WE HADDEN HET OVER JOU , NIET OVER MIJ*. Chatbots met regels bleven na ELIZA bestaan. A.L.I.C.E. van Richard Wallace, voor het eerst geschreven in 1995, won in 2000 en 2001 de Loebner Prize, een jaarlijkse wedstrijd naar het voorbeeld van de Turingtest. De regels staan in AIML, en de basiseenheid van kennis in AIML noemde Wallace een *category*. Een *category* heeft een patroon met jokers, zoals `I AM *`, en een *template* voor het antwoord: een ontledingsregel en een opbouwregel met andere namen.

        **Het ELIZA effect.** Douglas Hofstadter (1995) gebruikte en omschreef een naam voor wat ELIZA liet zien, het **ELIZA effect**: de neiging van mensen om in reeksen symbolen, vooral woorden, die een computer aan elkaar rijgt, veel meer begrip te lezen dan terecht is.

        Het plaatje volgt regel 3 van het gesprek door de stappen.
        """),
    ])
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **chatbot** | een programma waarmee je in gewone taal een gesprek kunt voeren |
    | **ELIZA** | de chatbot van Joseph Weizenbaum uit 1966, die antwoordt volgens de regels van een script |
    | **script** | de regels van ELIZA, los van het programma: omwisselingen, sleutelwoorden met een rang, ontledingsregels en opbouwregels |
    | **sleutelwoord** | een woord van het script met eigen regels; ELIZA reageert op het sleutelwoord met de hoogste rang |
    | **rang** | een getal dat zegt welk sleutelwoord wint als een tekst er meer dan één bevat; 0 als het script geen rang geeft |
    | **ontledingsregel** | een patroon zoals `(0 JOUW 0)` dat een tekst in genummerde delen knipt; `0` staat voor een willekeurig aantal woorden |
    | **opbouwregel** | een antwoord met nummers erin, zoals `(JOUW 3)`; elk nummer wordt de woorden van dat deel |
    | **ELIZA effect** | de neiging om te geloven dat een chatbot je begrijpt, omdat de chatbot in vloeiende taal antwoordt |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om op een getypte regel te antwoorden zoals ELIZA:

    1. Schrijf de invoer als woorden in hoofdletters, met een punt en een komma als eigen woorden.
    2. Houd het eerste deel tussen punten, komma's en *maar* met een sleutelwoord erin.
    3. Wissel elk woord om dat het script omwisselt.
    4. Kies het sleutelwoord met de hoogste rang, bij gelijke rang het eerste.
    5. Probeer de ontledingsregels op volgorde, en knip de omgewisselde woorden in delen met de eerste regel die past.
    6. Vul de delen in de volgende opbouwregel van deze ontledingsregel.
    7. Zonder sleutelwoord: de oudste herinnering als de teller op 4 staat, en anders een zin van NONE.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem regel 1 van het gesprek. `C.CONVERSATION` bevat de 15 regels in de taal van deze les. `C.words` maakt van een getypte regel woorden in hoofdletters, en `C.clause` houdt het eerste deel met een sleutelwoord. `C.swap` wisselt woorden om volgens `C.SWAPS`, de omwisselingen van het script, en `C.keyword` kiest het sleutelwoord volgens `C.RANKS`, de sleutelwoorden van het script met de rang van elk. `C.RULES` bevat de regels van elk sleutelwoord, en `C.decompose` knipt de woorden in delen met één ontledingsregel, of geeft `None` als de regel niet past:
    """)
    return


@app.cell
def example_cell():
    _kept = C.clause(C.words(C.CONVERSATION[L.lang][0]), C.RANKS[L.lang])
    _swapped = C.swap(_kept, C.SWAPS[L.lang])
    _key = C.keyword(_kept, C.RANKS[L.lang])
    print(_kept)
    print(_swapped)
    print(_key)
    for _rule in C.RULES[L.lang][_key]:
        print(_rule.decomposition, C.decompose(_rule.decomposition, _swapped))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De punt aan het einde is weg: de punt kwam na het sleutelwoord. *ZIJN* is het enige sleutelwoord, en het script wisselt geen van de vier woorden om. *ZIJN* heeft één ontledingsregel, `(0 ZIJN 0)`, met *MANNEN* als deel 1 en *ALLEMAAL HETZELFDE* als deel 3. De eerste opbouwregel is `(DACHT JE DAT 1 MISSCHIEN NIET 3 ZIJN)`, en dat geeft *DACHT JE DAT MANNEN MISSCHIEN NIET ALLEMAAL HETZELFDE ZIJN*. In een bijzin staat het werkwoord achteraan, dus de opbouwregel zet *ZIJN* aan het einde, na deel 3. De volgende keer dat deze ontledingsregel past, neemt ELIZA de tweede opbouwregel, `(ZOU JE WILLEN DAT 1 NIET 3 WAREN)`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_swap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. swap: woorden omdraaien

        Schrijf `swap`, die `words` krijgt, een lijst woorden in hoofdletters, en `swaps`, een dictionary van een woord naar de vervanger, zoals `C.SWAPS["nl"]`, en een nieuwe lijst teruggeeft: elk woord dat een sleutel van `swaps` is, vervangen door het woord dat erbij hoort, en elk ander woord zoals het is. In de startcode begint `new` als lege lijst, en gaat de `for` langs elk `word`. Vervang de `...`.

        De check roept `swap(["MY", "MOTHER", "TAKES", "CARE", "OF", "ME"], {"I": "YOU", "YOU": "I", "MY": "YOUR", "ME": "YOU"})` aan, met woorden van de Engelse les, wat `["YOUR", "MOTHER", "TAKES", "CARE", "OF", "YOU"]` geeft, en `swap(["IK", "BEN", "ONGELUKKIG"], {"IK": "JE", "JE": "IK", "BEN": "BENT", "BENT": "BEN"})`, wat `["JE", "BENT", "ONGELUKKIG"]` geeft. Daarna eigen gevallen: geen woorden, geen woord om te wisselen, *YOU* en *I* in één zin, drie keer hetzelfde woord, en regels van het gesprek met de omwisselingen van beide scripts.
        """),
        L.task("Wissel elk woord van het script om voor de vervanger, en houd de andere woorden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_swap():
    box_swap = L.code_box("swap", lines=8)
    box_swap
    return (box_swap,)


@app.cell(hide_code=True)
def check_swap(box_swap):
    result_swap = L.check("swap", box_swap.value)
    result_swap
    return (result_swap,)


@app.cell(hide_code=True)
def hints_swap():
    L.hints(
        "swap",
        [
            '`new.append(x)` zet `x` achteraan in de lijst `new`. `word in swaps` is `True` als `word` een sleutel van de dictionary `swaps` is, en `swaps["MIJN"]` geeft het woord dat bij de sleutel `"MIJN"` hoort.',
            "Binnen de `for`: is het woord een sleutel van `swaps`, voeg dan het woord dat erbij hoort toe aan `new`; zo niet, voeg dan het woord zelf toe. Zoek elk woord één keer op, zoals het in `words` staat.",
            "`            new.append(___)`",
        ],
        why="`if word in swaps:` met `new.append(swaps[word])`, en `else:` met `new.append(word)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_swap():
    mo.md(r"""
    De omwisselingen gaan woord voor woord, zonder naar de zin te kijken. Voor *mijn vriend liet me hierheen komen* is dat genoeg, maar het gaat mis zodra de grammatica verandert. In regel 13 wordt *Je bent bang voor me* eerst *IK BEN BANG VOOR JOU*, en de opbouwregel `(HEBBEN JE OUDERS 3 JOU)` maakt er *HEBBEN JE OUDERS BEN BANG VOOR JOU* van. Het Engelse script van 1965 maakt dezelfde fout. Een programma dat elk paar woorden na elkaar omwisselde, eerst alle *IK*'s en dan alle *JE*'s, zou elke *IK* in *JE* veranderen en meteen weer terug.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keyword():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. keyword: het woord met de hoogste rang

        Schrijf `keyword`, die `words` krijgt, een lijst woorden, en `ranks`, een dictionary van elk sleutelwoord naar de rang, zoals `C.RANKS["nl"]`, en het sleutelwoord met de hoogste rang teruggeeft, bij gelijke rang het eerste, of `None` als `words` geen sleutelwoord bevat. In de startcode is `best` het beste sleutelwoord tot nu toe en `best_rank` de rang ervan, die begint bij -1, onder elke rang; de `for` gaat langs elk `word`. Vervang de `...` in de `if`.

        De check roept `keyword(["MY", "FATHER", "IS", "AFRAID", "OF", "EVERYBODY"], {"MY": 0, "EVERYBODY": 2, "I": 0})` aan, met woorden van de Engelse les, wat `"EVERYBODY"` geeft, en `keyword(["MIJN", "VADER", "IS", "BANG", "VOOR", "IEDEREEN"], {"MIJN": 0, "IEDEREEN": 2, "IK": 0})`, wat `"IEDEREEN"` geeft. Daarna eigen gevallen: geen woorden, geen sleutelwoord, drie sleutelwoorden met rang 0, een later sleutelwoord met een hogere rang, twee sleutelwoorden met dezelfde rang, en regels van het gesprek met de rangen van beide scripts.
        """),
        L.task("Kies het sleutelwoord met de hoogste rang, bij gelijke rang het eerste, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_keyword():
    box_keyword = L.code_box("keyword", lines=8)
    box_keyword
    return (box_keyword,)


@app.cell(hide_code=True)
def check_keyword(box_keyword):
    result_keyword = L.check("keyword", box_keyword.value)
    result_keyword
    return (result_keyword,)


@app.cell(hide_code=True)
def hints_keyword():
    L.hints(
        "keyword",
        [
            '`word in ranks` is `True` als `word` een sleutelwoord is, een sleutel van de dictionary `ranks`, en `ranks["IEDEREEN"]` is de rang van *IEDEREEN*. Een test met `and` is alleen `True` als de tests aan beide kanten `True` zijn.',
            "Een sleutelwoord neemt alleen de plaats van `best` in als de rang hoger is dan `best_rank`, niet als de rang gelijk is: bij gelijke rang blijft het eerste sleutelwoord staan. Test eerst of het woord een sleutelwoord is, en zoek pas dan de rang op.",
            "`        if word in ranks and ___:`",
        ],
        why="`if word in ranks and ranks[word] > best_rank:`.",
    )
    return


@app.cell(hide_code=True)
def text_after_keyword():
    mo.md(r"""
    Met een rang zegt de schrijver van een script welk woord het belangrijkst is. In regel 14 komt *mijn* eerst, maar *iedereen* heeft rang 2, en ELIZA antwoordt *ECHT , IEDEREEN*. In regel 10 staan *jij*, *bent* en *mijn*, maar *als*, uit *net als*, heeft rang 3: ELIZA antwoordt *DENK JE DAT HET WAARSCHIJNLIJK IS DAT JOUW VADER*, alsof de gebruiker *als* in de betekenis van *indien* gebruikte. De meeste sleutelwoorden hebben geen rang, en dan wint het eerste: in regel 7 komt *misschien* eerder dan *ik* en *mijn*, dus reageert ELIZA op de twijfel en niet op de moeder. Weizenbaum gaf zelf een voorbeeld: krijgt *everybody* een hogere rang dan *I*, dan kan ELIZA vragen aan wie de gebruiker in het bijzonder denkt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reassemble():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. reassemble: het antwoord bouwen

        Schrijf `reassemble`, die `template` krijgt, een opbouwregel als lijst woorden, en `parts`, de delen van de ontleding als dictionary van de nummers, geschreven als strings zoals `"3"`, naar lijsten woorden. De functie geeft het antwoord terug als lijst woorden: elk nummer van de opbouwregel vervangen door de woorden van dat deel, en elk ander woord zoals het is. In de startcode begint `reply` als lege lijst, en gaat de `for` langs elk `word` van de opbouwregel. Vervang de `...`.

        De check roept `reassemble(["WHY", "DO", "YOU", "THINK", "I", "3", "YOU"], {"1": [], "2": ["I"], "3": ["DON'T", "ARGUE", "WITH"], "4": ["YOU"], "5": []})` aan, met de delen die `(0 I 0 YOU 0)` uit regel 12 van de Engelse les knipt, wat `["WHY", "DO", "YOU", "THINK", "I", "DON'T", "ARGUE", "WITH", "YOU"]` geeft, en `reassemble(["HOE", "LANG", "BEN", "JE", "AL", "4"], {"1": [], "2": ["JE"], "3": ["BENT"], "4": ["ONGELUKKIG"]})`, uit regel 5 van het gesprek, wat `["HOE", "LANG", "BEN", "JE", "AL", "ONGELUKKIG"]` geeft. Daarna eigen gevallen: een leeg deel, een deel dat twee keer gebruikt wordt, een opbouwregel zonder nummers, een lege opbouwregel, en nog twee antwoorden, uit elk script één.
        """),
        L.task("Zet de woorden van elk deel op de plaats van het nummer in de opbouwregel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reassemble():
    box_reassemble = L.code_box("reassemble", lines=9)
    box_reassemble
    return (box_reassemble,)


@app.cell(hide_code=True)
def check_reassemble(box_reassemble):
    result_reassemble = L.check("reassemble", box_reassemble.value)
    result_reassemble
    return (result_reassemble,)


@app.cell(hide_code=True)
def hints_reassemble():
    L.hints(
        "reassemble",
        [
            "Een deel is een lijst woorden, dus een tweede `for`, met een naam zoals `part_word` voor elk woord van het deel, kan de woorden één voor één aan `reply` toevoegen met `reply.append`.",
            '`parts["3"]` is de lijst woorden van deel 3: het nummer in de opbouwregel is de sleutel. De tweede `for` gaat langs deze lijst.',
            "`            for part_word in ___:`",
        ],
        why="`for part_word in parts[word]:` met `reply.append(part_word)` erin.",
    )
    return


@app.cell(hide_code=True)
def text_after_reassemble():
    mo.md(r"""
    Antwoord 9, *WIE ANDERS IN JE FAMILIE*, komt uit `(WIE ANDERS IN JE FAMILIE 5)` met een leeg deel 5. In het paper gaf de Engelse versie van deze opbouwregel al het antwoord op regel 8 van het gesprek: *WHO ELSE IN YOUR FAMILY TAKES CARE OF YOU*. Antwoord 15 komt helemaal niet uit een opbouwregel: *Pestkoppen.* bevat geen sleutelwoord, de teller stond op 4, en ELIZA zei de herinnering die het programma bij regel 3 had gemaakt. Wie dat niet weet, ziet een programma dat de pestkoppen met de vriend verbindt.

    ### Probeer het: praat met ELIZA

    Typ in het vak één regel per beurt. ELIZA beantwoordt elke regel met het Nederlandse script van deze les, en het plaatje laat zien hoe ELIZA aan elk antwoord kwam. De gebruikers van 1966 konden geen vraagteken typen: het MAC-systeem las een vraagteken als teken om de regel te wissen, schreef Weizenbaum. De course leest ? en ! als een punt. Probeer ELIZA eerst slim te laten lijken, en daarna iets geks te laten zeggen. Op welke woorden reageert het programma, en wat doet het met een regel zonder sleutelwoord?
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.TRY_START[L.lang], label="Je regels", rows=6, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_does():
    mo.md(r"""
    ### Oefening 4. Wat doet ELIZA?

    Deel elke uitspraak in: doet het programma van 1965 dit met het Nederlandse script van deze les, of niet?
    """)
    return


@app.cell(hide_code=True)
def exercise_does():
    sort_does = L.sort("does")
    sort_does
    return (sort_does,)


@app.cell(hide_code=True)
def check_does(sort_does):
    result_does = L.check("does", sort_does)
    result_does
    return (result_does,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl) zocht patronen in tekst; een ontledingsregel is zo'n patroon, met woorden waar een regex tekens heeft. [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl) liet beurten, paren en grounding zien. ELIZA neemt elke beurt en antwoordt vaak met een vraag, waarmee de volgende beurt terug naar de gebruiker gaat, maar aan grounding doet ELIZA niets: het programma controleert nooit wat het begreep, want het begrijpt niets. De volgende les bouwt een programma dat wel bijhoudt wat de gebruiker zei, om iets voor elkaar te krijgen: een trein boeken.

    **Een voorbeeld uit het dagelijks leven.** Mensen vertellen een chatbot nog steeds dingen die niemand anders te horen krijgt, en lezen zorg in vloeiende zinnen. Weizenbaum zag het al in 1966: ELIZA laat in elk geval zien hoe makkelijk je de illusie van begrip kunt wekken en vasthouden, en daarmee misschien de illusie van een oordeel dat vertrouwen verdient. Daar schuilt een gevaar.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_swap, result_keyword, result_reassemble, result_does):
    L.wrap_up(result_swap, result_keyword, result_reassemble, result_does)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
