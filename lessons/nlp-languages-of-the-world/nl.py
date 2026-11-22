# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-languages-of-the-world. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-languages-of-the-world", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl), en één over [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pronoun():
    quick_pronoun = L.choice(
        "quick-pronoun",
        question="Universal Dependencies geeft elk woord een van 17 tags voor de woordsoort. Welke tag krijgt *het* in *de hond vond het*?",
        options={
            "noun": "NOUN, een zelfstandig naamwoord",
            "right": "PRON, een voornaamwoord",
            "det": "DET, een lidwoord",
        },
        correct="right",
        why={
            "right": "*Het* staat hier in de plaats van een woordgroep, zoals *het bot*: een voornaamwoord. Deze les telt zinnen met een onderwerp en een lijdend voorwerp, en de atlas van talen die de les gebruikt, telt alleen de zinnen waarin dat allebei zelfstandige naamwoorden zijn, geen voornaamwoorden.",
            "noun": "*Het* neemt hier de plaats in van een woordgroep, zoals *het bot*, maar is zelf geen zelfstandig naamwoord. Welke tag krijgen *ik*, *het* en *jij*? Zie *Zeventien tags voor elke taal* in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
            "det": "Als lidwoord staat *het* vóór een zelfstandig naamwoord, zoals in *het bot*, en hier staat *het* alleen. Welke tag krijgen *ik*, *het* en *jij*? Zie *Zeventien tags voor elke taal* in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
        },
    )
    quick_pronoun
    return (quick_pronoun,)


@app.cell(hide_code=True)
def quick_check_object():
    quick_object = L.choice(
        "quick-object",
        question="In *de kat zag de vogel* hangt Universal Dependencies *vogel* aan *zag*. Met welke relatie?",
        options={
            "nsubj": "nsubj, het onderwerp",
            "right": "obj, het lijdend voorwerp",
            "amod": "amod, een bijvoeglijk naamwoord bij een zelfstandig naamwoord",
        },
        correct="right",
        why={
            "right": "Met *de vogel* gebeurt het zien: het lijdend voorwerp van *zag*, `obj(zag, vogel)`. Deze les leest het onderwerp en het lijdend voorwerp van een werkwoord af aan deze labels, en kijkt dan in welke volgorde die staan.",
            "nsubj": "Het onderwerp is wie of wat iets doet: hier *de kat*. Wat is *de vogel* voor *zag*? Zie de tabel in *Universal Dependencies* in [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
            "amod": "*Vogel* is een zelfstandig naamwoord, geen bijvoeglijk naamwoord, en het hangt aan een werkwoord. Wat is *de vogel* voor *zag*? Zie de tabel in *Universal Dependencies* in [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
        },
    )
    quick_object
    return (quick_object,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Er zijn zo'n 7.000 talen op de wereld, schrijven Jurafsky en Martin in het hoofdstuk *Machine Translation* van *Speech and Language Processing*. Die talen verschillen in klanken, in woorden en in de manier waarop een zin gebouwd wordt, en de tools van NLP werken maar voor weinig ervan. In 2020 deelden Pratik Joshi en vier anderen 2.485 talen in zes groepen in, naar de data en hulpmiddelen die er voor elke taal zijn, van de *Left-Behinds* tot de *Winners*. Maar 7 talen haalden de hoogste groep, waaronder het Engels, het Spaans, het Duits, het Japans en het Frans; 2.191 zitten in de laagste. Het web is net zo scheef verdeeld: in de crawl CC-MAIN-2026-43 van Common Crawl, een grote verzameling webpagina's, is 42,6% van de pagina's Engels en 1,8% Nederlands, volgens de taalherkenner die Common Crawl gebruikt. In 2019 vroeg Emily M. Bender iedereen die een tool voor taal bouwt of test: noem altijd de taal of talen waaraan je werkt. Deze les kijkt naar wat talen gemeen hebben en waarin talen verschillen: in families, en in de volgorde van onderwerp, werkwoord en lijdend voorwerp.

    **Een voorbeeld uit het dagelijks leven.** Tel de talen die het toetsenbord van je telefoon of een vertaalapp aanbiedt. Hoe lang de lijst er ook uitziet, zet die naast de 7.000.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een atlas van talen legt voor 1.376 talen vast in welke volgorde die talen meestal het onderwerp (S, van subject), het lijdend voorwerp (O, van object) en het werkwoord (V, van verb) van een zin zetten. Welke volgorde hebben de meeste talen?",
        options={
            "svo": "SVO, zoals het Engels en het Mandarijn",
            "right": "SOV, zoals het Japans",
            "vso": "VSO, zoals het Welsh",
        },
        reveal="**SOV, zoals het Japans**: 564 van de 1.376 talen, tegen 488 met SVO, de volgorde van het Engels. Samen zijn die twee goed voor meer dan drie op de vier talen. Het werkwoord vooraan, VSO of VOS, komt voor in 120 talen, en het lijdend voorwerp vooraan, OVS of OSV, in maar 15. 189 talen hebben geen dominante volgorde, waaronder het Nederlands en het Duits.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Hoeveel talen?

    Het aantal hangt af van wat je telt. *Ethnologue*, een catalogus van SIL International, telde 7.170 levende talen in de editie van 2026, zoals Wikipedia meldt. *Glottolog*, een catalogus van het Max Planck-instituut voor evolutionaire antropologie in Leipzig, telt 7.685 gesproken eerste talen in versie 5.4, van 10 november 2026, en in dat aantal tellen ook de talen mee die niet meer gesproken worden: 972, volgens Glottologs eigen maat voor hoe bedreigd een taal is, in de telling van de course. En waar de ene taal ophoudt en de volgende begint, is niet altijd duidelijk. Glottolog telt twee manieren van spreken als aparte talen wanneer de sprekers van de ene de andere niet verstaan, en als dialecten van één taal wanneer dat wel zo is; aan een consequente indeling van de dialecten heeft Glottolog naar eigen zeggen weinig moeite besteed. Een latere les gaat over taal en dialect.

    ### Families

    Talen die zijn voortgekomen uit één oudere taal, vormen een **taalfamilie**. Glottolog zet talen in één familie als hun overeenkomsten het best te verklaren zijn uit een gemeenschappelijke vooroudertaal. Het Nederlands en het Engels zijn allebei West-Germaans, een tak van het Germaans, dat een tak is van het Indo-Europees: een familie van 587 talen in Glottolog, uitgestorven talen meegeteld, met onder meer het Hindi, het Perzisch, het Russisch en het Welsh. In de indeling van Glottolog staat het Engels zelfs dichter bij het Fries dan bij het Nederlands: het Engels en het Fries delen de tak Anglo-Fries, en het Afrikaans hoort bij de tak van het Nederlands. Een taal zonder bekende verwanten is een **isolaat**, een familie van één, zoals het Baskisch. In de telling van de course in de data van Glottolog 5.4 vallen de 7.685 talen in 237 families en 183 isolaten (de eigen pagina van Glottolog zegt 245 families, kennelijk met 8 groepen erbij die geen families van verwante talen zijn, zoals gebarentalen en pidgins); de grootste families zijn de familie die Glottolog Atlantic-Congo noemt, met 1.409 talen, en het Austronesisch, met 1.280.

    ### Typen: hoe talen gebouwd zijn

    Talen kun je ook vergelijken zonder te vragen waar die talen vandaan komen. Jurafsky en Martin noemen de studie van zulke systematische overeenkomsten en verschillen tussen talen **taaltypologie** (linguistic typology). Taaltypologie deelt talen in typen in naar hun bouw, verwant of niet. Een van de bekendste eigenschappen is de volgorde van het onderwerp (S), het werkwoord (V) en het lijdend voorwerp (O) in een zin als *de kat zag de vogel*: zes volgordes zijn mogelijk, SVO, SOV, VSO, VOS, OVS en OSV. In 1963 vergeleek Joseph Greenberg 30 talen; de eerste universale in dat werk luidt dat in mededelende zinnen met een zelfstandig naamwoord als onderwerp en als lijdend voorwerp de dominante volgorde bijna altijd een volgorde is waarin het onderwerp vóór het lijdend voorwerp komt.

    De **World Atlas of Language Structures** (WALS), samengesteld door Matthew S. Dryer en Martin Haspelmath, beschrijft 192 van zulke eigenschappen voor 2.662 talen, verzameld door 55 auteurs uit grammatica's en andere beschrijvingen. De atlas verscheen in 2005 als boek en in 2008 online; de course gebruikt de data van versie v2020.6, vrijgegeven onder CC BY 4.0. Voor eigenschap 81A legde Dryer de **dominante volgorde** van 1.376 talen vast: de enige volgorde die een taal toestaat, of de volgorde die de taal het meest gebruikt. Een volgorde van drie delen telt als dominant als tellingen in teksten laten zien dat die volgorde meer dan twee keer zo vaak voorkomt als de volgende; is geen enkele volgorde dat, dan heeft de taal geen dominante volgorde. Dryer kijkt alleen naar zinnen waarin het onderwerp en het lijdend voorwerp zelfstandige naamwoorden zijn, geen voornaamwoorden, en gebruikt onderwerp en lijdend voorwerp in een losse betekenis: het deel dat handelt en het deel dat iets ondergaat. De afbeelding laat de aantallen zien.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_dutch():
    mo.md(r"""
    SOV en SVO zijn samen de dominante volgorde van 1.052 van de 1.376 talen, 76,5%, en de volgordes met het lijdend voorwerp vooraan zijn zeldzaam, zoals Greenberg schreef: 11 talen met OVS en 4 met OSV. Het Engels is SVO. Het Nederlands, het Duits en het Fries hebben in WALS geen dominante volgorde. Voor het Duits en het Nederlands geeft Dryer de reden, de plaats van het werkwoord, in het hoofdstuk van WALS over deze eigenschap: die talen hebben SVO in hoofdzinnen zonder hulpwerkwoord, en SOV in bijzinnen en in zinnen met een hulpwerkwoord. Beide volgordes komen veel voor, dus geen van beide telt als dominant; WALS legt het paar, SOV of SVO, vast in een eigen eigenschap.

    Eigenschappen gaan ook samen. Neem de talen in WALS met een waarde voor de volgorde van lijdend voorwerp en werkwoord, en voor de plaats van kleine woorden zoals *in* en *van* ten opzichte van hun naamwoordgroep. Daarvan zetten er 472 het lijdend voorwerp vóór het werkwoord en hebben **achterzetsels**, die na hun naamwoordgroep komen, zoals in het Japans, het Turks en het Hindi; maar 14 zetten het lijdend voorwerp vóór het werkwoord en hebben voorzetsels. Met het lijdend voorwerp na het werkwoord hebben 454 talen voorzetsels en 41 achterzetsels. Dryer noemt het een neiging in twee richtingen: een taal is meestal OV als die taal achterzetsels heeft, en heeft meestal achterzetsels als die taal OV is. Greenbergs vierde universale, uit 1963, verbond SOV al met achterzetsels.

    ### Weinig talen, weinig tools

    Terug naar Joshi en de anderen. In hun groep 0, de *Left-Behinds*, zitten 2.191 van de 2.485 talen die ze indeelden, zoals het Warlpiri en het Dahalo; over die talen schrijven ze dat die genegeerd werden en nog steeds worden waar het om taaltechnologie gaat. Het Nederlands zit in groep 4, de *Underdogs*, met het Russisch, het Hongaars, het Vietnamees en het Koreaans. Grote modellen komen verder, maar niet ver: NLLB-200, een model uit 2022 dat vertaalt, ondersteunt 202 talen, en de modellen van Qwen3, uit 2025, zijn getraind op 119 talen en dialecten. Bender zette de regel in juni 2019 in een tweet; in Benders stuk erover gaat een deel over het Engels, dat geen synoniem is van taal in het algemeen, en ook geen goede vertegenwoordiger ervan.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **taalfamilie** (language family) | talen die zijn voortgekomen uit één vooroudertaal, zoals het Indo-Europees |
        | **isolaat** (isolate) | een taal zonder bekende verwanten: een familie van één, zoals het Baskisch |
        | **taaltypologie** (linguistic typology) | talen vergelijken op hun bouw, verwant of niet |
        | **WALS** | de World Atlas of Language Structures: 192 eigenschappen van 2.662 talen |
        | **dominante volgorde** (dominant order) | de enige volgorde die een taal toestaat, of de volgorde die de taal het meest gebruikt: volgens de regel van WALS meer dan twee keer zo vaak als de volgende |
        | S, V, O | onderwerp, werkwoord en lijdend voorwerp, naar de Engelse woorden subject, verb en object; SOV is de volgorde onderwerp, lijdend voorwerp, werkwoord |
        | **achterzetsel** (postposition) | een woord als een voorzetsel dat na de naamwoordgroep komt in plaats van ervoor |
        | `letters + "V"` | een nieuwe string: de string `letters` met `"V"` erachter |
        | `for letters in counts:` | gaat de sleutels van de dictionary `counts` een voor een langs |
        | `d.values()` | de waarden van de dictionary `d`, zonder de sleutels |
        | `sorted(x, reverse=True)` | een nieuwe lijst met de elementen van `x`, van hoog naar laag |
        """),
        L.watch_out("WALS geeft elke taal één waarde voor een eigenschap, uit grammatica's en tellingen in teksten, en een taal kan gevarieerder zijn dan één waarde. Dryer schrijft zelf dat verdere tellingen sommige talen waarschijnlijk een andere waarde zouden geven. En een familie en een type zijn verschillende dingen: verwante talen kunnen van type verschillen, zoals het Nederlands en het Engels, en talen uit verschillende families kunnen een type delen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zoek het werkwoord van een zin, en het onderwerp en het lijdend voorwerp: de woorden die eraan hangen met de relaties `nsubj` en `obj`.
    2. Ga de woorden van links naar rechts langs, en schrijf voor elk van de drie S, V of O op: dat geeft de volgorde, zoals SVO.
    3. Doe dat voor elke zin van een treebank, en tel hoe vaak elke volgorde terugkomt.
    4. Vergelijk de volgorde die het vaakst voorkomt met de volgende: komt die meer dan twee keer zo vaak voor, dan is het de dominante volgorde, volgens de regel van WALS.
    5. Zoek voor veel talen tegelijk de waarden op in WALS, en vergelijk families en delen van de wereld.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat twee Nederlandse zinnen van de course, met de relaties van Universal Dependencies, als lijsten van (vorm, hoofdwoord, relatie): `SENTENCE`, *gisteren zag de kat de vogel*, en `CLAUSE`, *omdat de kat de vogel ziet*. De tweede cel gaat de woorden van een zin langs, genummerd vanaf 1, en print elk onderwerp, het woord met de relatie `nsubj`: het nummer, en het nummer van het woord waar het onderwerp aan hangt.
    """)
    return


@app.cell
def sentences_cell():
    SENTENCE = [
        ("gisteren", 2, "advmod"),   # bijwoordelijke bepaling
        ("zag", 0, "root"),          # het werkwoord, de root
        ("de", 4, "det"),            # lidwoord
        ("kat", 2, "nsubj"),         # onderwerp van zag
        ("de", 6, "det"),            # lidwoord
        ("vogel", 2, "obj"),         # lijdend voorwerp van zag
    ]
    CLAUSE = [
        ("omdat", 6, "mark"),        # voegwoord
        ("de", 3, "det"),            # lidwoord
        ("kat", 6, "nsubj"),         # onderwerp van ziet
        ("de", 5, "det"),            # lidwoord
        ("vogel", 6, "obj"),         # lijdend voorwerp van ziet
        ("ziet", 0, "root"),         # het werkwoord, de root
    ]
    return CLAUSE, SENTENCE


@app.cell
def example_subjects(CLAUSE, SENTENCE):
    def subjects(sentence):
        for i in range(1, len(sentence) + 1):
            form, head, relation = sentence[i - 1]
            if relation == "nsubj":
                print(form, "is woord", i, "en hangt aan woord", head)

    subjects(SENTENCE)
    subjects(CLAUSE)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In `SENTENCE` is het onderwerp *kat* woord 4, en het hangt aan woord 2, het werkwoord *zag*: het onderwerp komt na het werkwoord, want *gisteren* nam de eerste plaats in, en een Nederlandse hoofdzin houdt het werkwoord op de tweede plaats. Het Taalportaal, een grammatica van het Nederlands online, zegt dat de persoonsvorm meestal op de zogenoemde tweede plaats van de hoofdzin staat. In `CLAUSE` is het onderwerp woord 3, en het hangt aan woord 6, het werkwoord *ziet*, aan het eind, zoals gewoonlijk na *omdat*. Eén Nederlands werkwoord, twee plaatsen: de oefeningen tellen hoe vaak elke volgorde voorkomt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. order: onderwerp, werkwoord en lijdend voorwerp op een rij

        Schrijf `order(sentence, h)`: ga de woorden van `sentence` van links naar rechts langs, en bouw een string van letters: `"V"` voor woord `h`, het werkwoord, `"S"` voor het onderwerp van dat werkwoord (relatie `nsubj`) en `"O"` voor het lijdend voorwerp (relatie `obj`). Geef de string terug. Een werkwoord zonder lijdend voorwerp geeft twee letters, zoals `"SV"`. Nummer de woorden vanaf 1, net als de hoofdwoorden. Voor `order(SENTENCE, 2)` is de uitkomst `"VSO"`. Vervang de `...`. De check roept `order(SENTENCE, 2)` aan, en daarna eigen gevallen: `CLAUSE`, andere korte zinnen, een werkwoord zonder lijdend voorwerp, een betrekkelijke bijzin, een werkwoord dat het eerste woord is, en een zin met twee werkwoorden, elk met een eigen onderwerp.
        """),
        L.task("Lees de volgorde van onderwerp, werkwoord en lijdend voorwerp af uit een dependency tree."),
    ])
    return


@app.cell(hide_code=True)
def exercise_order():
    box_order = L.code_box("order", lines=11)
    box_order
    return (box_order,)


@app.cell(hide_code=True)
def check_order(box_order):
    result_order = L.check("order", box_order.value)
    result_order
    return (result_order,)


@app.cell(hide_code=True)
def hints_order():
    L.hints(
        "order",
        [
            "Begin met een lege string `letters`. Ga de nummers `i` langs van 1 tot en met het aantal woorden, zoals in het uitgewerkte voorbeeld, en neem woord `i` van plaats `i - 1`. Voeg bij elk woord de letter toe als het woord er een heeft: V voor woord `h` zelf, S voor het onderwerp van `h`, O voor het lijdend voorwerp. Geef na de loop `letters` terug.",
            "Woord `i` heeft de relatie `nsubj`. Hoe weet je dat het het onderwerp van werkwoord `h` is, en niet van een ander werkwoord in de zin?",
            '`        if ___ and relation == "nsubj":`',
        ],
        why='`letters = ""`, dan `for i in range(1, len(sentence) + 1):` met daaronder `form, head, relation = sentence[i - 1]`, en drie keer `if`: `if i == h:` met `letters = letters + "V"`, `if head == h and relation == "nsubj":` met `letters = letters + "S"`, en `if head == h and relation == "obj":` met `letters = letters + "O"`. Na de loop: `return letters`.',
    )
    return


@app.cell(hide_code=True)
def text_after_order():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op zeven zinnen uit de treebanks. De vijf Nederlandse zijn één zin, *ik mis de trein*, in vijf volgordes, uit de Nederlandse treebank: het onderwerp vooraan en het werkwoord op de tweede plaats; *vandaag* vooraan, zodat het onderwerp na het werkwoord komt; met het hulpwerkwoord *heb* het hoofdwerkwoord *gemist* aan het eind; en het lijdend voorwerp *de trein* vooraan, met en zonder *heb*. Universal Dependencies hangt het onderwerp en het lijdend voorwerp aan het hoofdwerkwoord, niet aan het hulpwerkwoord, dus *ik heb de trein gemist* telt als SOV: de volgorde die Dryer noemt voor Nederlandse zinnen met een hulpwerkwoord.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. count_orders: elke zin van een treebank

        `ENGLISH` hieronder bevat de 11.723 zinnen van de Engelse treebank uit [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl), en `DUTCH` de 13.603 van de Nederlandse, elke zin een lijst woorden zoals hierboven. Schrijf `count_orders(sentences)`: vraag voor elk woord `h` van elke zin `order(sentence, h)` om de letters, en tel in een dictionary van volgorde naar aantal hoe vaak elke volgorde van alle drie de letters terugkomt. Woorden met minder letters, zonder onderwerp of lijdend voorwerp, tellen niet mee. `order` staat klaar voor je functie: de check geeft je functie de `order` van de course. Voor `count_orders(DUTCH)` komt SOV 2.319 keer voor. Vervang de `...`. De check roept `count_orders(DUTCH)` aan, en daarna eigen gevallen: korte lijsten met zinnen van de course, en een lege lijst.
        """),
        L.task("Tel hoe vaak elke volgorde van onderwerp, werkwoord en lijdend voorwerp voorkomt in een treebank."),
    ])
    return


@app.cell
def treebanks_cell():
    ENGLISH = C.treebank("en")
    DUTCH = C.treebank("nl")
    print(len(ENGLISH), len(DUTCH))
    return (DUTCH,)


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("count-orders", lines=11)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("count-orders", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "count-orders",
        [
            "Begin met een lege dictionary `counts`. Ga de zinnen langs, en daarbinnen de nummers `h` van de woorden van de zin. Vraag `order(sentence, h)` om de letters; zijn het er drie, tel dan 1 op bij het aantal van die volgorde, of geef de volgorde het aantal 1 als die nog niet in de dictionary staat. Geef na beide loops de dictionary terug.",
            "De woorden van een zin zijn genummerd vanaf 1. Waar moet `range` beginnen, en waar moet `range` stoppen om bij het laatste woord te komen?",
            "`        for h in range(___):`",
        ],
        why="`counts = {}`, dan `for sentence in sentences:` en daarin `for h in range(1, len(sentence) + 1):`, met daaronder `letters = order(sentence, h)` en `if len(letters) == 3:`; daarbinnen `if letters in counts:` met `counts[letters] = counts[letters] + 1`, en `else:` met `counts[letters] = 1`. Na beide loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    In het Engels zijn 5.001 van de 5.273 zinnen met een onderwerp, een werkwoord en een lijdend voorwerp SVO, en bijna alle andere OSV: hun lijdend voorwerp is een vraagwoord of een woord als *that* of *which* aan het begin van een betrekkelijke bijzin, zoals in *What do I do?* In het Nederlands komt geen volgorde daar in de buurt: SOV 2.319 keer, SVO 1.564, VSO 716, OSV 359 en OVS 250. WALS telt alleen zinnen waarin onderwerp en lijdend voorwerp zelfstandige naamwoorden zijn; met alleen die zinnen veranderen de aantallen weinig: 1.208 van de 1.215 Engelse zinnen zijn SVO, en in het Nederlands gaat SOV voorop met 820, vóór SVO met 762.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dominant():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. dominant: de regel van WALS

        `COUNTS` hieronder bevat de aantallen van de Nederlandse treebank: de uitkomst van `count_orders(DUTCH)`. Schrijf `dominant(counts)`: geef de volgorde die het vaakst voorkomt terug als die meer dan twee keer zo vaak voorkomt als de volgende, volgens de regel van WALS, en anders de lege string `""`, voor geen dominante volgorde. Sorteer de aantallen met `sorted(..., reverse=True)` om de twee hoogste te vinden, en zoek dan op welke volgorde het hoogste aantal heeft. Een dictionary met maar één volgorde maakt die volgorde dominant. Voor `dominant(COUNTS)` is de uitkomst `""`. Vervang de `...`. De check roept `dominant(COUNTS)` aan, en daarna eigen gevallen: de aantallen van het Engels, één volgorde alleen, een volgorde die precies twee keer zo vaak voorkomt als de volgende en een die iets vaker voorkomt, de volgorde die het vaakst voorkomt niet vooraan in de dictionary, en twee volgordes die even vaak voorkomen.
        """),
        L.task("Beslis met de regel van WALS of één volgorde dominant is."),
    ])
    return


@app.cell
def counts_cell(DUTCH):
    COUNTS = C.count_orders(DUTCH)
    print(COUNTS)
    return


@app.cell(hide_code=True)
def exercise_dominant():
    box_dominant = L.code_box("dominant", lines=7)
    box_dominant
    return (box_dominant,)


@app.cell(hide_code=True)
def check_dominant(box_dominant):
    result_dominant = L.check("dominant", box_dominant.value)
    result_dominant
    return (result_dominant,)


@app.cell(hide_code=True)
def hints_dominant():
    L.hints(
        "dominant",
        [
            "Zet de aantallen van de dictionary in een lijst van hoog naar laag, `ranked`. Gebruik de regel om te beslissen of je de lege string teruggeeft. Zo niet, ga dan de volgordes van de dictionary langs, en geef de volgorde terug waarvan het aantal het eerste van `ranked` is.",
            "Welke twee aantallen van `ranked` vergelijkt de regel, en welke uitkomst betekent dat er geen dominante volgorde is?",
            "`    if len(ranked) > 1 and ___:`",
        ],
        why='`ranked = sorted(counts.values(), reverse=True)`, dan `if len(ranked) > 1 and ranked[0] <= 2 * ranked[1]:` met daaronder `return ""`. Daarna `for letters in counts:` met daaronder `if counts[letters] == ranked[0]:` en `return letters`.',
    )
    return


@app.cell(hide_code=True)
def text_after_dominant():
    mo.md(r"""
    De regel geeft het Engels in elke rij van de afbeelding SVO, en het Nederlands, over alle zinnen, geen dominante volgorde, net als WALS: SOV komt maar 1,5 keer zo vaak voor als SVO. Splits de Nederlandse zinnen naar soort, en de twee volgordes uit Dryers beschrijving komen terug. Met een hulpwerkwoord wint SOV 984 van de 1.090 zinnen, en na een onderschikkend voegwoord zoals *omdat* 657 van de 732. In de overige zinnen, zonder hulpwerkwoord of onderschikkend voegwoord en geen betrekkelijke bijzin, komt SVO 2,2 keer zo vaak voor als VSO, en is SVO dus net dominant; daar zijn 2.458 van de 2.632 zinnen SVO, VSO of OVS: de volgordes die een werkwoord op de tweede plaats geeft, na het onderwerp, na een ander woord of na het lijdend voorwerp.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de kaart van WALS

    Kies een eigenschap en een familie. De kaart tekent elke taal van WALS met een waarde voor die eigenschap, en de staven en de tabel tellen de waarden, met voorbeelden. Vergelijk Niger-Congo met het Sino-Tibetaans, kijk waar de talen met het werkwoord vooraan gesproken worden, en wissel tussen 83A en 85A: de talen met het lijdend voorwerp vóór het werkwoord en de talen met achterzetsels liggen grotendeels op dezelfde plekken.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_feature = mo.ui.dropdown(options=C.feature_names(L.lang), value=list(C.feature_names(L.lang))[0], label="Eigenschap")
    demo_family = mo.ui.dropdown(options=C.family_names(L.lang), value=list(C.family_names(L.lang))[0], label="Familie")
    mo.hstack([demo_feature, demo_family], justify="start", gap=1.5)
    return demo_family, demo_feature


@app.cell(hide_code=True)
def demo_look(demo_family, demo_feature):
    mo.Html(C.demo_picture(demo_feature.value, demo_family.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Families verschillen sterk. In WALS zijn 173 van de 187 Niger-Congotalen met een waarde voor 81A SVO, en alle 68 van Trans-Nieuw-Guinea SOV, terwijl het Austronesisch elke volgorde heeft. Het Indo-Europees is verdeeld: 30 SOV, zoals het Hindi en het Perzisch, en 27 SVO, zoals het Engels en het Russisch. WALS gebruikt een eigen lijst van families, die op plaatsen afwijkt van die van Glottolog: WALS heeft een Altaïsche familie, met het Turks erin, en Glottolog niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_family():
    mo.md(r"""
    ### Oefening 4. Familie of type?

    Elke vetgedrukte uitspraak gaat over een familie of over een type. Zet de uitspraak bij waar die over gaat.
    """)
    return


@app.cell(hide_code=True)
def exercise_family():
    sort_family = L.sort("family-or-type")
    sort_family
    return (sort_family,)


@app.cell(hide_code=True)
def check_family(sort_family):
    result_family = L.check("family-or-type", sort_family)
    result_family
    return (result_family,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl) gaf elk woord een hoofdwoord en een relatie, en [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) een tag; deze les gebruikte die om talen te vergelijken. Er worden zo'n 7.000 talen gesproken, in families die zijn voortgekomen uit gemeenschappelijke voorouders, en taaltypologie vergelijkt talen op hun bouw, wat hun familie ook is. Je las de volgorde van onderwerp, werkwoord en lijdend voorwerp af uit een boom, telde de volgordes van twee treebanks, paste de regel van WALS toe, en hield families en typen uit elkaar. Het Engels is SVO; het Nederlands houdt het werkwoord in een hoofdzin op de tweede plaats, en zet het werkwoord aan het eind na een onderschikkend voegwoord zoals *omdat*, of bij een hulpwerkwoord. De meeste tools van NLP werken voor weinig van de 7.000 talen, dus zeg voor welke taal jouw tool is, zoals Bender vraagt. De volgende les van het domein gaat over taal in de samenleving: *Taal, groepen en dialecten*.

    **Een voorbeeld uit het dagelijks leven.** Als een app of een model de volgende keer zegt dat het jouw taal spreekt, vraag dan hoe goed, en hoeveel van de 7.000 talen het spreekt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_counts, result_dominant, result_family, result_order):
    L.wrap_up(result_order, result_counts, result_dominant, result_family)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
