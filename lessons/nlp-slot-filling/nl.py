# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-slot-filling. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-slot-filling", "nl")
    C = L.checks   # het boekingssysteem met het frame, het gesprek, de kaartjesopdrachten van MASSIVE en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl), en één over [ELIZA: een chatbot met regels](/course/open/nlp-eliza?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question=r'In *Van tekst naar een tabel* haalde je datums uit tekst met groepen. Wat geeft `re.findall(r"(\d+) (mei|juni)", "op 28 mei of 4 juni")`?',
        options={
            "whole": '`["28 mei", "4 juni"]`: elke hele match, dag en maand samen',
            "first": '`("28", "mei")`: wat de twee groepen de eerste keer matchten',
            "right": '`[("28", "mei"), ("4", "juni")]`: een paar voor elke match',
            "days": '`["28", "4"]`: wat de eerste groep matchte',
        },
        correct="right",
        why={
            "right": "Met twee groepen in het patroon geeft `re.findall` voor elke match een paar, met één string per groep. In deze les heeft elk slot een patroon met één groep, en heeft het systeem alleen de eerste match nodig.",
            "whole": "Het patroon heeft groepen. Wat geeft `re.findall` als een patroon meer dan één groep heeft? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
            "first": "Dat zijn de groepen van de eerste match. Stopt `re.findall` bij de eerste match? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
            "days": "Het patroon heeft twee groepen, een rond de dag en een rond de maand. Wat geeft `re.findall` voor elke match? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_eliza():
    quick_eliza = L.choice(
        "quick-eliza",
        question="In *ELIZA: een chatbot met regels* antwoordde ELIZA met het Nederlandse script van de course. Een gebruiker typt *Mijn trein vertrekt om negen uur.*, en ELIZA antwoordt *JOUW TREIN VERTREKT OM NEGEN UUR*. Daarna typt de gebruiker *Hoe laat vertrekt mijn trein?* Wat antwoordt ELIZA?",
        options={
            "nine": "*OM NEGEN UUR*: ELIZA zoekt op wat de gebruiker over de trein zei",
            "memory": "*JOUW TREIN VERTREKT OM NEGEN UUR*, de herinnering aan zin 1",
            "right": "*WAAROM VRAAG JE DAT*: een opbouwregel van *hoe*",
            "none": "*GA VERDER*: de zin heeft geen sleutelwoord",
        },
        correct="right",
        why={
            "right": "*Hoe* en *mijn* zijn allebei sleutelwoorden met rang 0, dus ELIZA neemt het eerste, *hoe*, en een opbouwregel van *hoe* geeft *WAAROM VRAAG JE DAT*. ELIZA bewaarde geen vertrektijd: het geheugen bevat zinnen om terug te zeggen, geen feiten om op te zoeken. Het systeem van deze les bewaart wat de gebruiker zei, in slots.",
            "nine": "ELIZA antwoordt met de ontledingsregels en opbouwregels van een sleutelwoord in de nieuwe zin. Bewaart ELIZA feiten over wat de gebruiker zei? Zie [ELIZA: een chatbot met regels](/course/open/nlp-eliza?lang=nl).",
            "memory": "ELIZA zegt een herinnering alleen als een zin geen sleutelwoord heeft en de teller op 4 staat. Heeft deze zin een sleutelwoord? Zie [ELIZA: een chatbot met regels](/course/open/nlp-eliza?lang=nl).",
            "none": "*Hoe* en *mijn* zijn sleutelwoorden van het script. Welke kiest ELIZA, en wat geven de opbouwregels ervan? Zie [ELIZA: een chatbot met regels](/course/open/nlp-eliza?lang=nl).",
        },
    )
    quick_eliza
    return (quick_eliza,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 1977 beschreven zes onderzoekers van het Xerox Palo Alto Research Center, Daniel Bobrow, Ronald Kaplan, Martin Kay, Donald Norman, Henry Thompson en Terry Winograd, het programma GUS, dat voor een eenvoudige vliegreis de rol van reisagent speelde. GUS staat voor Genial Understander System, en volgens de makers moest het programma in het Engels een gesprek voeren met een welwillende mens die goed meewerkt. Dan Jurafsky en James Martin namen in een eerdere versie van het boek *Speech and Language Processing* het verslag van een gesprek met GUS op. Vertaald begint het zo:

    > GUS: Hallo. Ik heet GUS. Ik kan je helpen om een eenvoudige vliegreis te plannen. Waar wil je naartoe?
    > Klant: Ik wil op 28 mei naar San Diego
    > GUS: Hoe laat wil je vertrekken?
    > Klant: Ik moet uiterlijk om 10 uur 's ochtends in San Diego zijn
    > GUS: Wil je vlucht 102 van P.S.A., met aankomst om 9.15 uur?

    De klant gaf twee dingen in één zin, een stad en een datum, en GUS vroeg niet opnieuw naar de datum: het programma vroeg naar wat nog ontbrak. ELIZA beantwoordde elke zin los van de rest; GUS hield bij wat de klant had gezegd, in een formulier met een lege plaats voor elk gegeven dat het programma nodig had. De klant noemde nooit een stad van vertrek, en toch zegt een bevestiging verderop in het gesprek *van San Jose naar San Diego*: GUS vulde deze plaats zelf in.

    Volgens Jurafsky en Martin legde GUS eind jaren zeventig de aanpak met frames vast, en bleef deze aanpak ruim dertig jaar de standaard voor dialogue systems in de industrie. Rond 2010 kwam de architectuur van GUS eindelijk op grote schaal in gebruik, in dialogue systems op telefoons, zoals SIRI van Apple, en in andere digitale assistenten. In deze les bouw je zo'n systeem voor treinkaartjes, en laat je de eerste stap ervan los op echte opdrachten aan een assistent.

    **Een voorbeeld uit het dagelijks leven.** Zeg *Zet een wekker* tegen een telefoon, en de telefoon vraagt *Hoe laat?*: een wekker heeft een tijd nodig, en de telefoon vraagt naar wat je wegliet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_frame = L.guess(
        "guess",
        question="MASSIVE is een dataset met opdrachten aan een assistent, in 2022 gepubliceerd door Jack FitzGerald en collega's van Amazon, in 51 talen, waaronder het Nederlands; in elke opdracht zijn de woorden van de slots gemarkeerd. 187 opdrachten vragen om een kaartje, zoals *boek een treinkaartje voor me*. Een opdracht die het hele frame van deze les in één keer vult, noemt twee plaatsen, waarvandaan en waarheen, en een datum. Bij hoeveel van de 187 markeert MASSIVE minstens twee plaatsen en een datum?",
        options={
            "most": "140, drie op de vier",
            "half": "93, ongeveer de helft",
            "quarter": "47, een kwart",
            "right": "8",
        },
        reveal="8 van de 187, zoals *koop een trein kaartje van amsterdam naar maastricht limburg voor over een week*. MASSIVE markeert minstens twee plaatsen zonder datum in nog 16 opdrachten, één plaats in 98, en geen plaats in 65; in 55 van deze 65 markeert MASSIVE geen plaats en geen datum, zoals in *boek een treinkaartje voor me*. De meeste opdrachten laten iets open, dus een systeem dat drie waarden nodig heeft, moet naar de rest vragen.",
    )
    guess_frame
    return (guess_frame,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **dialogue system** is een programma dat in meerdere beurten een gesprek met een mens voert. ELIZA was er een zonder taak. Een systeem voor **task-oriented dialogue** helpt de gebruiker met één concrete taak, zoals een kaartje boeken, en is klaar als de taak klaar is. Jurafsky en Martin, in de bijlage *Frame-Based Dialogue Systems* van het boek: frames zijn kennisstructuren die de details vastleggen van de taak die de gebruiker opgeeft; elk frame bestaat uit een verzameling slots, en elk slot kan een waarde uit een bepaalde verzameling krijgen.

    Een **frame** is een formulier voor een taak, en een **slot** is een plaats daarin voor één gegeven dat de taak nodig heeft. Een wekker heeft een tijd nodig, staat in de bijlage. Het frame van deze les heeft drie slots, in deze volgorde: *naar*, het station van bestemming; *van*, het station van vertrek; en *datum*, de dag van de reis. In de code heten de slots `"to"`, `"from"` en `"date"`, in beide lessen. Elk slot heeft een vraag, naar het frame voor vliegreizen in de bijlage: *Waar ga je naartoe?*, *Uit welke stad vertrek je?* en *Op welke dag wil je vertrekken?*. Het doel van het systeem is volgens Jurafsky en Martin om de slots te vullen met de waarden die de gebruiker bedoelt, en daarna te doen wat de gebruiker wil, zoals een vraag beantwoorden of een vlucht boeken.

    Voor elke zin die de gebruiker typt, doorloopt het systeem van deze les deze stappen:

    1. **Slot filling.** De waarden voor de slots vinden in wat de gebruiker zegt, heet **slot filling**. De eenvoudigste dialogue systems doen dat met regels die met de hand geschreven zijn, zoals deze reguliere expressie uit de bijlage voor een verzoek om een wekker te zetten: `wake me (up) | set (the|an) alarm | get me up`. Het systeem van deze les schrijft de zin in kleine letters en heeft één reguliere expressie voor elk slot, met één groep: `\bnaar (alkmaar|amsterdam|…|zwolle)\b` voor *naar*, hetzelfde met *van* voor *van*, en de dagen van de week met *vandaag* en *morgen* voor *datum*. De stations staan in het patroon, dus *ik wil naar huis en daarna naar utrecht* geeft *utrecht*, niet *huis*. Een slot met een passende expressie krijgt wat de groep matchte, de eerste keer.
    2. **Het frame.** Elke gevonden waarde gaat in het eigen slot, en de andere slots houden wat erin stond. Volgens Jurafsky en Martin is de toestand van het gesprek meer dan de waarden in de laatste zin: het is het hele frame op dat moment, met alles wat de gebruiker tot dan toe wil. Deze toestand na elke beurt bijwerken heet dialogue state tracking.
    3. **De volgende vraag.** In de eerste commerciële systemen met frames was de werkwijze eenvoudig, schrijven Jurafsky en Martin: vragen stellen tot alle slots vol zijn, een database doorzoeken, en dan de gebruiker het resultaat melden. Het systeem stelt de vraag van het eerste slot dat nog leeg is.
    4. **Een kaal antwoord.** Als een zin helemaal geen waarde bevat, leest het systeem de zin opnieuw, met het woord van het slot waar het net naar vroeg aan het begin: na *Uit welke stad vertrek je?* wordt *Groningen.* gelezen als *van groningen.*
    5. **Bevestiging.** Na een beurt die een slot vult, begint de vraag met wat het systeem tot nu toe begreep, zoals in *Op vrijdag een trein naar Utrecht. Uit welke stad vertrek je?*: een **impliciete bevestiging**, die volgens de bijlage minder direct voor grounding zorgt, door bij de volgende vraag te herhalen wat het systeem begreep. Als elk slot gevuld is, zegt het systeem de hele reis en vraagt *Zal ik het kaartje boeken?*: een **expliciete bevestiging**. Alleen een ja boekt het kaartje; na een nee zegt de gebruiker wat er anders moet. De bijlage weegt de twee af: met een expliciete bevestiging kan de gebruiker een misverstand makkelijk rechtzetten, gewoon door *nee* te zeggen, maar zo'n bevestiging kost tijd en voelt onhandig.

    **Intents.** Een assistent doet meer dan één taak, dus bepaalt de assistent eerst om welke taak een opdracht vraagt, zoals een kaartje kopen of een wekker zetten. Zo'n soort verzoek is de **intent** van de opdracht, en de intent vinden heet **intent detection**: een classifier met één label voor elke intent. Het systeem van deze les doet één enkele taak, dus elke zin heeft dezelfde intent. In MASSIVE heeft elke opdracht een van 60 intents; de 187 kaartjesopdrachten hebben de intent `transport_ticket`, met de slots gemarkeerd, zoals in `boek een [transport_type : trein] kaartje naar [place_name : amsterdam]`.

    Het plaatje volgt beurt 1 van het gesprek van de course door de stappen.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **dialogue system** | een programma dat in meerdere beurten een gesprek met een mens voert |
    | **task-oriented dialogue** | een gesprek waarin het systeem de gebruiker helpt met één concrete taak, zoals een kaartje boeken |
    | **frame** | een formulier voor een taak, met een slot voor elk gegeven dat de taak nodig heeft |
    | **slot** | een plaats in een frame voor één waarde, zoals het station van bestemming |
    | **slot filling** | de waarden voor de slots vinden in wat de gebruiker zegt |
    | **intent** | het soort verzoek in een opdracht, zoals een kaartje kopen; de intent vinden heet **intent detection** |
    | **impliciete bevestiging** | zeggen wat het systeem begreep, als deel van de volgende vraag |
    | **expliciete bevestiging** | de gebruiker vragen of klopt wat het systeem begreep, zoals *Zal ik het kaartje boeken?* |
    | `.group(1)` | bij een match van `re.search`: de tekst die de eerste groep van het patroon matchte |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo boek je een kaartje met een frame:

    1. Begin met een frame waarin elk slot leeg is, en stel de vraag van het eerste slot.
    2. Zoek in elke zin de waarde van elk slot met de reguliere expressie ervan: wat de groep de eerste keer matcht.
    3. Niets gevonden? Lees de zin opnieuw, met het woord van het slot waar je naar vroeg aan het begin.
    4. Zet de gevonden waarden in het frame; elk ander slot houdt de waarde.
    5. Stel de vraag van het eerste lege slot, met aan het begin wat je begreep.
    6. Is geen slot meer leeg, zeg dan de hele reis en vraag of je moet boeken; boek alleen na een ja.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem beurt 1 van het gesprek. `C.CONVERSATION` bevat de vier zinnen van de gebruiker in de taal van deze les, en het systeem schrijft een zin eerst in kleine letters met `.lower()`. `C.SLOTS` bevat de namen van de slots in de volgorde van het frame, en `C.PATTERNS` de reguliere expressie van elk slot, voor elke taal. `re.search` zoekt de eerste match van elke expressie. Daarna geeft `C.find_slots` de gevonden waarden, `C.new_frame()` een frame waarin elk slot leeg is, `C.fill` het frame na de beurt, en `C.next_slot` het eerste slot dat nog leeg is:
    """)
    return


@app.cell
def example_cell():
    import re

    _text = C.CONVERSATION[L.lang][0].lower()
    print(_text)
    for _slot in C.SLOTS:
        print(_slot, re.search(C.PATTERNS[L.lang][_slot], _text))
    _found = C.find_slots(_text, C.PATTERNS[L.lang])
    _frame = C.fill(C.new_frame(), _found)
    print(_found)
    print(_frame)
    print(C.next_slot(_frame))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De expressie van *naar* matcht *naar utrecht*: in de geprinte match zegt `span=(15, 27)` dat de match loopt van plaats 15 tot plaats 27 van de zin, zonder plaats 27 zelf, en `match='naar utrecht'` geeft de tekst. De groep bevat *utrecht*. De expressie van *van* vindt niets, en `re.search` geeft `None`. De expressie van *datum* matcht *vrijdag*, eerder in de zin: elk slot heeft een eigen expressie, dus de volgorde in de zin doet er niet toe. Na de beurt bevat het frame *utrecht* en *vrijdag*, en is *van* nog leeg, dus het systeem antwoordt *Op vrijdag een trein naar Utrecht. Uit welke stad vertrek je?*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_find_slots():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. find_slots: slots vullen uit een zin

        Schrijf `find_slots`, die `text` krijgt, een zin in kleine letters, en `patterns`, een dictionary van elk slot naar de reguliere expressie ervan met één groep, zoals `C.PATTERNS["nl"]`. De functie geeft een dictionary terug met, voor elk slot met een passende expressie, wat de groep de eerste keer matchte; een slot zonder match blijft erbuiten. In de startcode maakt `import re` de module `re` beschikbaar in het vak, begint `found` als lege dictionary, en gaat de `for` langs elk `slot`; `m = re.search(patterns[slot], text)` is de eerste match, of `None` als de expressie niet past. Vervang de `...`: als `m` een match is, gaat de waarde in `found`.

        De check roept `find_slots("i want to go to boston on friday", C.PATTERNS["en"])` aan, met de expressies van de Engelse les, wat `{"to": "boston", "date": "friday"}` geeft, en `find_slots("ik wil vrijdag naar utrecht", C.PATTERNS["nl"])`, wat `{"to": "utrecht", "date": "vrijdag"}` geeft. Daarna eigen gevallen: een lege zin, alle drie de slots in één zin, een station van twee woorden, twee stations na *to*, een Nederlandse zin met alle drie de slots, twee slots van een ander frame (een kleur en een ding), en helemaal geen expressies.
        """),
        L.task("Geef elk slot met een passende reguliere expressie de tekst van de groep, de eerste keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_find_slots():
    box_find_slots = L.code_box("find_slots", lines=10)
    box_find_slots
    return (box_find_slots,)


@app.cell(hide_code=True)
def check_find_slots(box_find_slots):
    result_find_slots = L.check("find_slots", box_find_slots.value)
    result_find_slots
    return (result_find_slots,)


@app.cell(hide_code=True)
def hints_find_slots():
    L.hints(
        "find_slots",
        [
            "`found[slot] = x` zet `x` in de dictionary `found`, onder de naam van het slot. Bij een match geeft `.group()` de hele tekst van de match.",
            "Als de reguliere expressie niet past, is `m` gelijk aan `None`, zonder groepen: test eerst met `m is not None` of er een match is. De waarde van een slot is alleen het station of de dag, zonder *naar* of *van* aan het begin: `.group(1)` geeft de tekst van de eerste groep.",
            "`            found[slot] = ___`",
        ],
        why="`if m is not None:` met daarin `found[slot] = m.group(1)`: de tekst van de groep, onder de naam van het slot, alleen als er een match is.",
    )
    return


@app.cell(hide_code=True)
def text_after_find_slots():
    mo.md(r"""
    Het plaatje laat `find_slots` los op de kaartjesopdrachten van MASSIVE met alleen stations van de les als plaats. Met de versie van de course vindt de functie 77 van de 88 plaatsen en datums die MASSIVE markeert in de 58 Nederlandse opdrachten, en 55 van de 65 in de 43 Engelse. De functie mist wat de expressies niet kennen: *kaartje voor delft* zonder *naar* of *van*, *vanaf rotterdam* en *vanuit rotterdam* met een ander woord dan *van*, en datums zonder dag van de week, zoals *veertien april* en *volgende week*. Regels die met de hand geschreven zijn, werken goed voor wat de schrijver bedacht, en missen de rest. In de bijlage staat direct na de reguliere expressie voor de wekker dat de meeste systemen supervised learning gebruiken: deze systemen leren slot filling uit gemarkeerde voorbeelden, als een label voor elk woord; de les *Een label per woord: sequence labeling en BIO* laat zien hoe zulke labels werken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fill():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. fill: het frame bijhouden

        Schrijf `fill`, die `frame` krijgt, het frame tot nu toe als dictionary van elk slot naar de waarde (`None` zolang het slot leeg is), en `found`, de waarden die `find_slots` in deze beurt vond. De functie geeft een nieuw frame terug met dezelfde slots, in dezelfde volgorde: een slot dat deze beurt vulde, krijgt de nieuwe waarde, en elk ander slot houdt de waarde die het had. In de startcode begint `new` als lege dictionary en gaat de `for` langs elk `slot` van `frame`; een slot dat in `found` staat, krijgt al de nieuwe waarde. Vervang de `...`.

        De check roept `fill({"to": None, "from": None, "date": None}, {"to": "boston", "date": "friday"})` aan, uit de Engelse les, wat `{"to": "boston", "from": None, "date": "friday"}` geeft, en `fill({"to": "utrecht", "from": None, "date": "vrijdag"}, {"from": "groningen"})`, wat `{"to": "utrecht", "from": "groningen", "date": "vrijdag"}` geeft. Daarna eigen gevallen: niets gevonden, een nieuwe waarde voor een slot dat al gevuld was, drie nieuwe waarden tegelijk, het tegelijk vullen van twee lege slots in een Nederlands frame, twee slots van een ander frame, en een frame zonder slots.
        """),
        L.task("Geef de nieuwe waarden van deze beurt, en houd de waarde van elk ander slot."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fill():
    box_fill = L.code_box("fill", lines=9)
    box_fill
    return (box_fill,)


@app.cell(hide_code=True)
def check_fill(box_fill):
    result_fill = L.check("fill", box_fill.value)
    result_fill
    return (result_fill,)


@app.cell(hide_code=True)
def hints_fill():
    L.hints(
        "fill",
        [
            "`new[slot] = x` zet `x` in het nieuwe frame onder de naam van het slot.",
            "Een slot dat deze beurt niet vulde, houdt de oude waarde: de waarde van hetzelfde slot in `frame`, het frame van het begin van de beurt.",
            "`            new[slot] = ___`",
        ],
        why="`new[slot] = frame[slot]`: de waarde die het slot al had.",
    )
    return


@app.cell(hide_code=True)
def text_after_fill():
    mo.md(r"""
    Omdat een nieuwe waarde de plaats van een oude inneemt, kan de gebruiker het systeem verbeteren: *Nee, op zaterdag.* verandert de datum en verder niets. Omdat elk ander slot de waarde houdt, hoeft de gebruiker nooit iets twee keer te zeggen. Het frame na elke beurt is wat de bijlage de toestand van het gesprek noemt; grotere systemen houden er meer in bij, zoals de dialogue act van de laatste beurt van de gebruiker: een vraag, een antwoord of een verbetering.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next_slot():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. next_slot: de volgende vraag

        Schrijf `next_slot`, die `frame` krijgt en het eerste slot teruggeeft, in de volgorde van het frame, dat nog leeg is, of `None` als elk slot gevuld is. In de startcode gaat de `for` langs elk `slot` van `frame` en geeft het slot meteen terug als de test in de `if` klopt; na de `for` zegt `return None` dat geen slot leeg is. Vervang de `...` door deze test.

        De check roept `next_slot({"to": "boston", "from": None, "date": "friday"})` aan, wat `"from"` geeft, en `next_slot({"to": "utrecht", "from": "groningen", "date": "vrijdag"})`, wat `None` geeft. Daarna eigen gevallen: een nieuw frame waarin elk slot leeg is, twee lege slots, alleen het laatste slot leeg, twee slots van een ander frame, en een frame zonder slots.
        """),
        L.task("Geef het eerste slot van het frame dat nog leeg is, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next_slot():
    box_next_slot = L.code_box("next_slot", lines=6)
    box_next_slot
    return (box_next_slot,)


@app.cell(hide_code=True)
def check_next_slot(box_next_slot):
    result_next_slot = L.check("next_slot", box_next_slot.value)
    result_next_slot
    return (result_next_slot,)


@app.cell(hide_code=True)
def hints_next_slot():
    L.hints(
        "next_slot",
        [
            "`frame[slot]` is de waarde van het slot `slot` in het frame: een station of een dag, of `None` zolang het slot leeg is.",
            "Het slot om naar te vragen heeft als waarde nog de `None` van een nieuw frame. Test de waarde van het slot, niet de naam, zoals je find_slots `m` testte op `None`.",
            "`        if frame[slot] ___:`",
        ],
        why="`if frame[slot] is None:`: het slot heeft nog geen waarde.",
    )
    return


@app.cell(hide_code=True)
def text_after_next_slot():
    mo.md(r"""
    De volgorde van het frame is de volgorde van de vragen: GUS vroeg eerst waar de reis naartoe ging, en dit systeem ook. Een gebruiker die meer zegt dan gevraagd werd, slaat vragen over: beurt 1 van het gesprek van de course vult twee slots tegelijk. Na beurt 2 is geen slot meer leeg, geeft `next_slot` dus `None`, en zegt het systeem de hele reis en vraagt *Zal ik het kaartje boeken?*. *Nee, op zaterdag.* bevat een waarde, dus het systeem vult het slot en vraagt het opnieuw; *Ja, graag.* bevat geen waarde en begint met *ja*, dus het systeem boekt.

    ### Probeer het: boek een kaartje

    Typ in het vak één zin per beurt. Het systeem beantwoordt elke zin, en het plaatje laat het frame na elke beurt zien. Probeer een zin met twee slots tegelijk, een kaal antwoord, een verbetering na een *nee*, en een station dat het systeem niet kent. Welke zinnen begrijpt het systeem verkeerd, en vang je de fout op dankzij een bevestiging?
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.TRY_START[L.lang], label="Je zinnen", rows=6, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_what():
    mo.md(r"""
    ### Oefening 4. Wat vult het systeem in?

    Het systeem heeft net gegroet en gevraagd *Waar ga je naartoe?*. Deel elk antwoord in naar wat het systeem van deze les invult: alleen goede waarden, een slot met een verkeerde waarde, of helemaal geen slot. Een waarde die de gebruiker gaf maar het systeem mist, maakt een gevuld slot niet verkeerd.
    """)
    return


@app.cell(hide_code=True)
def exercise_what():
    sort_what = L.sort("what")
    sort_what
    return (sort_what,)


@app.cell(hide_code=True)
def check_what(sort_what):
    result_what = L.check("what", sort_what)
    result_what
    return (result_what,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl) haalde velden uit tekst met reguliere expressies met groepen; slot filling doet hetzelfde in een gesprek, met één veld voor elk slot. [ELIZA: een chatbot met regels](/course/open/nlp-eliza?lang=nl) beantwoordde elke zin los van de rest; het systeem van deze les houdt een frame bij over de beurten heen, vraagt naar wat ontbreekt en weet wanneer de taak klaar is. De bevestigingen zijn de grounding van [Hoe een gesprek werkt](/course/open/nlp-conversation?lang=nl): zeggen wat je begreep, zodat de ander het kan verbeteren. De volgende lessen van het domein gaan over zoeken: de documenten vinden die een vraag beantwoorden.

    **Een voorbeeld uit het dagelijks leven.** Een kaartjesautomaat op het station vraagt op het scherm dezelfde dingen, de een na de ander: waarheen, welke dag, en of je wilt betalen. Waarvandaan is het station waar de automaat staat, tenzij je het verandert: een slot dat al gevuld is als je begint, zoals GUS San Jose invulde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_find_slots, result_fill, result_next_slot, result_what):
    L.wrap_up(result_find_slots, result_fill, result_next_slot, result_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
