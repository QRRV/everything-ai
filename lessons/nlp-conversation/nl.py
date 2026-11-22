# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-conversation. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-conversation", "nl")
    C = L.checks   # het gesprek van de course, de afbeeldingen en de checks


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
def quick_check_salt():
    quick_salt = L.choice(
        "quick-salt",
        question="Wat hoort een luisteraar meestal in *Kun je het zout even aangeven?*",
        options={
            "ability": "Een vraag of je dat kunt",
            "right": "Een verzoek om het zout aan te geven",
            "promise": "Een belofte dat de spreker het zout zo aangeeft",
        },
        correct="right",
        why={
            "right": "De vorm is een vraag naar wat je kunt, en de handeling is een verzoek: een indirecte taalhandeling. Deze les geeft zulke handelingen een label, beurt voor beurt, in een heel gesprek.",
            "ability": "Dat vragen de woorden, letterlijk genomen. Een luisteraar die alleen *ja* antwoordt, heeft gemist wat bedoeld werd: de spreker wil het zout. Zie *Zeggen is doen* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
            "promise": "Met een belofte legt de spreker zichzelf iets op, en hier wil de spreker juist iets van de luisteraar. Zie *Zeggen is doen* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
        },
    )
    quick_salt
    return (quick_salt,)


@app.cell(hide_code=True)
def quick_check_quantity():
    quick_quantity = L.choice(
        "quick-quantity",
        question="*Sommige gasten kwamen.* Waarom maakt een luisteraar eruit op dat niet alle gasten kwamen?",
        options={
            "logic": "In de logica betekent *sommige* niet alle",
            "presupposition": "Het is een presuppositie: die blijft staan als je *niet* toevoegt",
            "right": "De spreker had *alle* kunnen zeggen, en deed dat niet",
        },
        correct="right",
        why={
            "right": "Dat is de maxime van kwantiteit: een coöperatieve spreker die wist dat iedereen kwam, had dat gezegd. Het *niet alle* is een implicatuur, en ook deze les gaat ervan uit dat sprekers samenwerken.",
            "logic": "In de logica betekent *sommige* minstens één, ook als iedereen kwam. Het *niet alle* werkt een luisteraar daar bovenop uit. Zie *Sommige, maar niet alle* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
            "presupposition": "Een presuppositie is wat een zin als vanzelfsprekend aanneemt. Het *niet alle* werkt een luisteraar uit, en de spreker kan het terugnemen: *sommige gasten kwamen; sterker nog, iedereen kwam*. Het is een implicatuur. Zie *Sommige, maar niet alle* in [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl).",
        },
    )
    quick_quantity
    return (quick_quantity,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een spraakassistent, een programma van een klantenservice of een language model waarmee je chat: elk van die programma's doet mee aan een gesprek, en een gesprek heeft een eigen orde. Jurafsky en Martin noemen zes eigenschappen van gesprekken die het onder meer moeilijk maken voor een programma om mee te praten zoals mensen dat doen: beurten, taalhandelingen, grounding, hoe een gesprek is opgebouwd, wie het initiatief neemt, en implicatuur. Een programma dat één zin goed beantwoordt, kan in het gesprek toch de mist in gaan: het praat terwijl jij nadenkt, of heeft niet door dat jouw *ja* de vraag van het programma beantwoordde.

    **Een voorbeeld uit het dagelijks leven.** Je vraagt een spraakassistent om een trein naar ... en denkt even na over de naam van de stad. De assistent antwoordt al, voordat je hebt gezegd waarheen. Of je bent klaar, en het blijft zo lang stil dat je vraagt of de assistent je wel gehoord heeft.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Tanya Stivers en collega's (2009) maten in video-opnames van alledaagse gesprekken in tien talen, waaronder het Nederlands, hoe snel mensen op een ja-neevraag begonnen te reageren, gerekend vanaf het einde van de vraag. Hoe lang was dat gemiddeld ongeveer?",
        options={
            "right": "Ongeveer 0,2 seconde",
            "second": "Ongeveer 1 seconde",
            "think": "Ongeveer 2 seconden: tijd om het antwoord te bedenken",
        },
        reveal="**Ongeveer 0,2 seconde**: gemiddeld 208 milliseconden over alle tien talen, en de waarde die het vaakst voorkwam was 0, een beurt die begint precies als de vraag eindigt. Deense sprekers waren gemiddeld het traagst (469 ms), Japanse sprekers het snelst (7 ms). De gemiddelden van de tien talen lagen binnen ongeveer 250 ms van het totale gemiddelde: volgens de onderzoekers ongeveer de tijd die het kost om één Engelse lettergreep uit te spreken. Ze keken alleen naar wat er volgde op ja-neevragen, 350 vragen per taal, en rondden elke tijd af op 100 ms.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Om de beurt

    Een gesprek is geen tekst met één schrijver: twee of meer mensen maken het samen. Volgens Jurafsky en Martin is een dialoog een reeks beurten, elk een bijdrage van één spreker. Zo'n bijdrage is een **beurt**. De sociologen Harvey Sacks, Emanuel Schegloff en Gail Jefferson (1974) zetten feiten over gesprekken op een rij. Hun lijst, hier gelezen in een herdruk die zichzelf een andere versie van de publicatie uit 1974 noemt, begint zo: de spreker wisselt steeds weer, of in elk geval minstens één keer; verreweg het grootste deel van de tijd praat er één tegelijk; en dat er meer dan één spreker tegelijk praat, komt vaak voor, maar duurt kort. De meeste sprekerswisselingen gaan zonder stilte en zonder overlap, of met heel weinig stilte of overlap. Ze schreven dat die feiten al bij een tamelijk ongerichte blik op gespreksmateriaal duidelijk leken.

    Hoe sprekers zo het praten verdelen, de een na de ander, heet **beurtwisseling**. Jurafsky en Martin merken op dat sprekers in een gesprek meestal vrijwel meteen aan hun beurt beginnen als de ander klaar is, zonder lange pauze. De gok liet zien hoe kort dat is. Om het te meten, schrijf je bij elk stuk gepraat op wie er spreekt, van wanneer tot wanneer. Bij een sprekerswisseling is het begin van de nieuwe beurt min het einde van de vorige de **overgangstijd**. Een positieve overgangstijd is een stilte; een negatieve is **overlap**: de nieuwe spreker begint voordat de oude klaar is. Stivers en collega's maten op dezelfde manier. Een pauze tussen twee stukken van dezelfde spreker is geen overgangstijd tussen beurten: die beurt loopt gewoon door.
    """)
    return


@app.cell(hide_code=True)
def text_idea_pairs():
    mo.md(r"""
    ### Beurten in paren

    Sommige beurten komen in paren. Emanuel Schegloff en Harvey Sacks (1973) beschreven het **aangrenzend paar** (adjacency pair): twee uitingen, direct na elkaar, van verschillende sprekers, zoals vraag en antwoord, een groet en een groet terug, of een aanbod en het aannemen of afslaan ervan. Volgens hen hóren paren die eigenschappen te hebben: het is waar sprekers zich naar richten, geen wet zonder uitzonderingen. Is het eerste deel gezegd, dan hoort de spreker volgens hen te stoppen, en hoort een volgende spreker het bijpassende tweede deel te geven.

    Jurafsky en Martin voegen toe dat er tussen de twee delen van een paar een zijsprong kan zitten. Een voorbeeld van de course zelf: *Wil je koffie?* *Is er havermelk?* *Ja.* *Dan graag.* De vraag over de melk en het antwoord daarop zitten binnen het paar van het aanbod en het aannemen ervan.
    """)
    return


@app.cell(hide_code=True)
def text_idea_grounding():
    mo.md(r"""
    ### Is het overgekomen?

    Een spreker kan niet in het hoofd van de luisteraar kijken. De psycholoog Herbert Clark en Edward Schaefer (1989) knipten een bijdrage in tweeën: de spreker brengt een uiting naar voren, en de luisteraar aanvaardt die door te laten blijken dat de uiting, voor zover de luisteraar weet, begrepen is. Ze noemen vijf tekens van begrip, ruwweg van zwakst naar sterkst: blijvende aandacht; beginnen aan de passende volgende bijdrage; een ontvangstbevestiging, zoals knikken of *hm-hm* of *ja*; een demonstratie van wat er begrepen is; en een letterlijke herhaling van een deel van wat er gezegd is, of van alles.

    Clark en Susan Brennan (1991) noemen dat hele proces **grounding**: spreker en luisteraar werken samen toe naar de gedeelde overtuiging dat de luisteraar heeft begrepen wat er bedoeld werd, goed genoeg voor het doel van dat moment. Een vertrektijd vraagt meer zorg dan een praatje over het weer. Ontvangstbevestigingen beslaan volgens hen een groot deel van wat in het Engels back-channel responses heet. Een **luistersignaal** (backchannel) is een kort signaal zoals *mhm* of *ja* waarmee de ander kan doorpraten.
    """)
    return


@app.cell(hide_code=True)
def evidence_picture():
    mo.Html(C.evidence_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_acts():
    mo.md(r"""
    ### Wat een beurt doet

    *Wat bedoeld wordt, niet gezegd* noemde een uiting, gezien als handeling, een taalhandeling. Een programma dat aan een gesprek meedoet, of een gesprek onderzoekt, geeft elke beurt zo'n handeling uit een vaste lijst labels: een **dialogue act**. Jurafsky en Martin omschrijven die als een label dat de functie van de zin in het gesprek weergeeft. Het werkt zoals de woordsoorten in *Woordsoorten en woordgroepen*, met een label voor elke beurt in plaats van voor elk woord.

    In 1997 gaven acht studenten taalkunde met de hand een label aan 1.155 telefoongesprekken uit de Switchboard-verzameling: 205.000 uitingen en 1,4 miljoen woorden. Ze gebruikten 220 labels, die de onderzoekers voor hun experimenten in 42 groepen samenvoegden. De twee grootste: mededelingen, 36 van de 100 uitingen, en luistersignalen en ontvangstbevestigingen, 19 van de 100; meningen waren een groep apart, met 13 van de 100. Ja-neevragen waren 2 van de 100. Andreas Stolcke en collega's (2000) trainden daarna een programma dat elke uiting een label geeft. Op hun testgesprekken, uitgeschreven, had het er 71 van de 100 goed, terwijl altijd *mededeling* zeggen er 35 van de 100 goed zou hebben. Twee mensen die dezelfde uitingen labelden, waren het bij 84 van de 100 eens.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **beurt** | één bijdrage van één spreker aan een gesprek |
        | **beurtwisseling** | hoe sprekers het praten verdelen, de een na de ander |
        | **overgangstijd** | bij een sprekerswisseling het begin van de nieuwe beurt min het einde van de vorige: positief is een **stilte**, negatief is **overlap**, tijd waarin beide spreken |
        | **aangrenzend paar** | twee beurten van verschillende sprekers waarbij de eerste de tweede verwacht maakt, zoals vraag en antwoord |
        | **grounding** | wat spreker en luisteraar doen om tot de gedeelde overtuiging te komen dat begrepen is wat er bedoeld werd |
        | **luistersignaal** | een kort signaal zoals *mhm* of *ja* waarmee de ander kan doorpraten; in het Engels backchannel |
        | **dialogue act** | wat een beurt doet, als label uit een vaste lijst, zoals mededeling, vraag of luistersignaal |
        | `"?" in text` | `True` als de tekst een vraagteken bevat: `in` kijkt in een tekst zoals in een lijst |
        """),
        L.watch_out("In lessen over language models die antwoorden uit documenten halen, betekent **grounding** iets anders: een model het antwoord laten baseren op bronnen die je meegeeft. In deze les is het wat twee mensen in een gesprek doen om zeker te weten dat ze elkaar begrepen hebben."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zet de stukken gepraat op volgorde van tijd: wie spreekt, van wanneer, tot wanneer. Stukken van dezelfde spreker achter elkaar zijn één beurt.
    2. Meet bij elke sprekerswisseling de overgangstijd: het begin van de nieuwe beurt min het einde van de vorige. Onder nul is overlap.
    3. Zoek paren: is een beurt een antwoord op de beurt ervoor, het aannemen van een aanbod of een groet terug?
    4. Zoek grounding: hoe toont de luisteraar wat er begrepen is, van *mhm* tot het herhalen van de woorden?
    5. Geef elke beurt een dialogue act: wat doet de beurt in het gesprek?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel schrijft een gesprek van de course zelf: Sam vraagt Kim naar de laatste trein. `TALK` bevat elk stuk gepraat als wie er spreekt, van wanneer en tot wanneer, in milliseconden; `SAID` bevat de woorden van elk stuk, in dezelfde volgorde. De tweede cel schrijft een lijst luistersignalen. De derde definieert twee functies, probeert beide uit op `TALK` en bekijkt de eerste twee stukken.
    """)
    return


@app.cell
def talk_cell():
    TALK = [
        ("Sam", 0, 1500), ("Kim", 1700, 3000), ("Sam", 2950, 3500), ("Kim", 3600, 3900), ("Sam", 3900, 5200),
        ("Sam", 5800, 6500), ("Kim", 6600, 6900), ("Sam", 6950, 8400), ("Kim", 8700, 9000), ("Sam", 9100, 9600),
    ]
    SAID = [
        "Hoe laat gaat de laatste trein naar Utrecht?", "Om 23.40 uur, van spoor 4", "Spoor 4?", "Ja",
        "Dus ik moet om elf uur van huis", "of iets eerder", "Mhm", "Kun je me de tijden sturen?", "Oké",
        "Bedankt",
    ]
    return SAID, TALK


@app.cell
def backchannels_cell():
    BACKCHANNELS = ["mhm", "hm-hm", "ja", "oké", "precies"]
    return (BACKCHANNELS,)


@app.cell
def example_look(BACKCHANNELS, SAID, TALK):
    def speakers(talk):
        return [stretch[0] for stretch in talk]

    def lengths(talk):
        out = []
        for i in range(len(talk)):
            out.append(talk[i][2] - talk[i][1])
        return out

    print(speakers(TALK))
    print(lengths(TALK))
    # van het einde van het eerste stuk tot het begin van het tweede, en de woorden van het tweede
    print(TALK[1][1] - TALK[0][2], SAID[1])
    print("?" in SAID[0], SAID[3] in BACKCHANNELS)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `speakers` neemt plaats 0 van elk stuk, en `lengths` geeft hoe lang elk stuk duurt: het einde min het begin. Kim begint 200 milliseconden nadat Sam stopt. `TALK[4]` en `TALK[5]` zijn allebei van Sam: één beurt, met een pauze erin. De laatste regel vraagt of de eerste beurt een vraagteken heeft, en dat klopt, en of *Ja* in `BACKCHANNELS` staat: `False`, want de lijst schrijft de woorden met kleine letters. Oefening 2 moet daar iets mee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. gaps: de tijd tussen beurten

        Schrijf `gaps(talk)`: geef bij elke sprekerswisseling in `talk` de overgangstijd in milliseconden, het begin van het nieuwe stuk min het einde van het stuk ervoor, in de volgorde van het gesprek. Een pauze tussen twee stukken van dezelfde spreker is geen overgangstijd: laat die weg. Bij `TALK` geeft de eerste wisseling `200`, en de tweede `-50`: Sam begint aan *Spoor 4?* voordat Kim klaar is. Vervang de `...`. De check roept `gaps(TALK)` aan, en daarna eigen gevallen, met één stuk, met overlap en met een pauze binnen een beurt.
        """),
        L.task("Meet de stilte of overlap bij elke sprekerswisseling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gaps", lines=6)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gaps", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Begin met een lege lijst `out`, en loop de plaatsen af vanaf 1 met `for i in range(1, len(talk)):`, zoals `lengths` ze afloopt vanaf 0. Voeg alleen iets toe aan `out` waar de spreker wisselt.",
            "Op plaats 0 van een stuk staat de spreker. Bij welke paren stukken achter elkaar begint er een nieuwe beurt?",
            "`        if ___:`",
        ],
        why="`out = []`; `for i in range(1, len(talk)):` met daaronder `if talk[i][0] != talk[i - 1][0]:` en daaronder `out.append(talk[i][1] - talk[i - 1][2])`; na de loop `return out`. `gaps(TALK)` geeft `[200, -50, 100, 0, 100, 50, 300, 100]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_gaps():
    mo.md(r"""
    De afbeelding onder de check tekent het gesprek als twee rijen stukken, van Sam en van Kim, met jouw getallen bij elke sprekerswisseling. De meeste overgangstijden zijn een paar honderd milliseconden of minder, en één keer is er overlap. De langste stilte, de 600 milliseconden tussen *om elf uur van huis* en *of iets eerder*, telt niet mee: Sam houdt de beurt. Een luisteraar die die pauze voor het einde aanzag, was Sam in de rede gevallen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: wanneer is Sam klaar?

    Een programma dat luistert, moet bepalen wanneer jouw beurt voorbij is. De eenvoudigste manier: wachten op een stilte van een vaste lengte, en dan beginnen te praten. Hieronder vraagt Sam een programma om een trein, met twee pauzes om na te denken. Kies hoe lang het programma op stilte wacht, en kijk waar het begint te praten. Zoek een wachttijd waarbij het programma Sam niet onderbreekt, en kijk hoe lang Sam dan op een antwoord wacht.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_wait = mo.ui.slider(100, 1500, step=50, value=300, label="Wachten op stilte (ms)", show_value=True)
    demo_wait
    return (demo_wait,)


@app.cell(hide_code=True)
def demo_look(demo_wait):
    mo.Html(C.demo_picture(demo_wait.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Om Sam 900 milliseconden te laten nadenken, moet het programma na elk stuk langer dan dat wachten, ook na het laatste. Mensen doen het beter: ze reageren op een vraag ongeveer zo snel als de gok liet zien, en Kim laat de pauze van 600 milliseconden van Sam voorbijgaan. Stivers en collega's vatten eerder onderzoek naar het Engels samen: sprekers wachten niet op een pauze om aan hun beurt te beginnen, maar vermijden stiltes en overlap; daarvoor voorspellen ze met grammatica, prosodie en pragmatiek wanneer ze aan een volgende beurt kunnen beginnen. Prosodie is de melodie en het ritme van spraak. Een programma dat alleen op stilte let, heeft daar niets van, en moet één wachttijd kiezen voor beide gevallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. label: een dialogue act bij elke beurt

        Schrijf `label(text, backchannels)`: geef een dialogue act terug voor de beurt `text`, volgens drie regels, in deze volgorde. Een beurt die in de lijst `backchannels` staat, met of zonder hoofdletters, is een `"backchannel"`; een beurt met een vraagteken erin is een `"question"`; elke andere beurt is een `"statement"`. De labels blijven Engels, zoals in Switchboard. Vervang de `...`. De check roept `label("Mhm", BACKCHANNELS)` aan, met een eigen Engelse lijst waar *mhm* ook in staat, en daarna eigen gevallen, elk met een eigen lijst.
        """),
        L.task("Geef een beurt het label backchannel, question of statement."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=6)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label):
    result_label = L.check("label", box_label.value)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Schrijf de drie regels op volgorde, elk als een `if` met een eigen `return`: eerst het luistersignaal, dan de vraag. De laatste regel, na beide, geeft `\"statement\"` terug.",
            "De lijst bevat de woorden in kleine letters, en *Mhm* begint met een hoofdletter. Wat kun je met de beurt doen voordat je die in de lijst opzoekt?",
            "`    if ___ in backchannels:`",
        ],
        why="`if text.lower() in backchannels:` met daaronder `return \"backchannel\"`; `if \"?\" in text:` met daaronder `return \"question\"`; aan het eind `return \"statement\"`. `label(\"Mhm\", BACKCHANNELS)` geeft `'backchannel'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_label():
    mo.md(r"""
    De afbeelding onder de check geeft elke beurt van het gesprek jouw label, het label van de drie regels en een label dat de course met de hand gaf, in de stijl van Switchboard. De regels zien alleen de vorm van een beurt. *Ja* na *Spoor 4?* is een antwoord, *Kun je me de tijden sturen?* is een verzoek, en *Oké* neemt dat verzoek aan. Om dat te zien, heeft een programma de beurt ervoor nodig, en wat een spreker met woorden doet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_evidence():
    mo.md(r"""
    ### Oefening 3. Hoe toont Sam dat het is overgekomen?

    Het is acht uur op het station. Kim zegt tegen Sam: *Je trein vertrekt om acht uur vijftien, van spoor 7.* Zet elk vetgedrukt antwoord van Sam bij wat het laat zien van wat Sam begreep.
    """)
    return


@app.cell(hide_code=True)
def exercise_evidence():
    sort_evidence = L.sort("evidence")
    sort_evidence
    return (sort_evidence,)


@app.cell(hide_code=True)
def check_evidence(sort_evidence):
    result_evidence = L.check("evidence", sort_evidence)
    result_evidence
    return (result_evidence,)


@app.cell(hide_code=True)
def text_exercise_yeah():
    mo.md(r"""
    ### Oefening 4. Antwoord of luistersignaal?

    Eén woord kan in een gesprek verschillende dingen doen. De regels van oefening 2 noemen elk *ja* een luistersignaal.
    """)
    return


@app.cell(hide_code=True)
def choice_yeah_cell():
    choice_yeah = L.choice(
        "yeah",
        question="In welk gesprekje is *Ja* een antwoord, en geen luistersignaal?",
        options={
            "right": "*Sam: Is het café nog open? Kim: Ja.*",
            "story": "*Sam: Dus ik kom op het station, Kim: Ja, Sam: en de trein was al weg.*",
            "plan": "*Sam: Ik neem de bus naar het centrum, Kim: Ja, Sam: en loop het laatste stuk vanaf de brug.*",
        },
        correct="right",
        why={
            "right": "*Is het café nog open?* is een vraag: het eerste deel van een paar, dat een antwoord de verwachte volgende beurt maakt. Het *Ja* van Kim is dat antwoord. De regels van oefening 2 noemen het toch een luistersignaal: de regels kijken naar de beurt alleen, niet naar de beurt ervoor.",
            "story": "Sam is midden in een verhaal, en het *Ja* van Kim zegt: ga door, ik luister. Dat is een luistersignaal. Zoek een *Ja* direct na een vraag.",
            "plan": "Sam vertelt een plan in twee delen, en het *Ja* van Kim ertussen zegt: ga door. Dat is een luistersignaal. Zoek een *Ja* direct na een vraag.",
        },
    )
    choice_yeah
    return (choice_yeah,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat bedoeld wordt, niet gezegd](/course/open/nlp-pragmatics?lang=nl) keek naar losse uitingen: wat een spreker met woorden bedoelt en doet. Deze les zette die uitingen in een gesprek: beurten die elkaar met korte overgangstijden opvolgen, paren zoals vraag en antwoord, grounding die laat zien wat er begrepen is, en een dialogue act bij elke beurt. Je mat overgangstijden en overlap, gaf beurten een label volgens drie regels, deelde tekens van begrip in en hield een antwoord en een luistersignaal uit elkaar. Latere lessen bouwen programma's die zelf om de beurt praten, te beginnen met ELIZA, uit 1966. De les *Een tekst is meer dan losse zinnen* kijkt hoe de zinnen van een tekst samenhangen.

    **Een voorbeeld uit het dagelijks leven.** Aan de telefoon zeg je af en toe *mhm*, zonder erbij na te denken. Zeg een tijdje niets, en de ander vraagt of je er nog bent.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_yeah, result_evidence, result_gaps, result_label):
    L.wrap_up(result_gaps, result_label, result_evidence, choice_yeah)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
