# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-phrase-structure. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-phrase-structure", "nl")
    C = L.checks   # de grammatica van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tag():
    quick_tag = L.choice(
        "quick-tag",
        question="Welke tag van Universal Dependencies krijgt *after* in *the train left after midnight* (de trein vertrok na middernacht)?",
        options={
            "adv": "`ADV`, omdat het zegt wanneer",
            "right": "`ADP`, omdat er een naamwoord volgt",
            "sconj": "`SCONJ`, zoals in *after the train left*",
        },
        correct="right",
        why={
            "right": "*after* staat vóór het zelfstandig naamwoord *midnight*: een voorzetsel, dat in UD de tag `ADP` krijgt, net als *in* en *of*. In deze les wordt `ADP` een symbool van een grammatica.",
            "adv": "*after* zegt wel wanneer, maar het staat vóór een zelfstandig naamwoord, *midnight*, en dan is het een voorzetsel: `ADP`. Zie de tabel met de 17 tags in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
            "sconj": "Met een hele bijzin erachter, zoals in *after the train left*, is *after* een `SCONJ`. Hier volgt alleen het zelfstandig naamwoord *midnight*: een voorzetsel, `ADP`. Zie de tabel met de 17 tags in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
        },
    )
    quick_tag
    return (quick_tag,)


@app.cell(hide_code=True)
def quick_check_tags():
    quick_tags = L.choice(
        "quick-tags",
        question="In *Woordsoorten en woordgroepen* geeft `tags_per_word` elk woord in kleine letters de lijst met de tags ervan, elke tag één keer, in de volgorde waarin ze voor het eerst komen. Wat geeft `tags_per_word([[(\"Book\", \"VERB\"), (\"it\", \"PRON\")], [(\"a\", \"DET\"), (\"book\", \"NOUN\")]])`?",
        options={
            "upper": "`{'Book': ['VERB'], 'it': ['PRON'], 'a': ['DET'], 'book': ['NOUN']}`",
            "last": "`{'book': ['NOUN'], 'it': ['PRON'], 'a': ['DET']}`",
            "right": "`{'book': ['VERB', 'NOUN'], 'it': ['PRON'], 'a': ['DET']}`",
        },
        correct="right",
        why={
            "right": "*Book* en *book* delen de key `'book'`, en de lijst ervan bevat beide tags, in de volgorde waarin ze kwamen. Dezelfde letters, twee woordsoorten: de woorden eromheen beslissen.",
            "upper": "`tags_per_word` maakt elk woord klein met `lower()`, dus *Book* en *book* delen één key. Zie Oefening 2 in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
            "last": "De lijst houdt elke tag van een woord één keer, in de volgorde waarin ze kwamen, dus `'book'` krijgt zowel `VERB` als `NOUN`. Zie Oefening 2 in [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
        },
    )
    quick_tags
    return (quick_tags,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Woordsoorten en woordgroepen* gaf elk woord een tag, en markeerde één naamwoordgroep, *de eerste helft*. Woordgroepen zitten ook in grotere woordgroepen. In *de trein verliet de stad na middernacht* is *de stad* een naamwoordgroep, en *verliet de stad* een grotere groep rond het werkwoord. Een programma dat de zin leest, moet weten welke woorden bij elkaar horen. Jurafsky en Martin noemen toepassingen zoals grammaticacontrole (een zin die geen structuur kan krijgen, bevat misschien een fout) en uitzoeken wat een zin betekent.

    De studie van grammatica is oud: Jurafsky en Martin openen hun hoofdstuk erover met Pāṇini, die ergens tussen de 7e en de 4e eeuw voor Christus de grammatica van het Sanskriet beschreef. In deze les schrijf je een grammatica als een paar regels in Python, en laat je de regels zinnen bouwen. Een handvol regels blijkt eindeloos veel zinnen te maken, meer dan er in welke lijst ook passen.

    **Een voorbeeld uit het dagelijks leven.** Een map op een computer kan bestanden en andere mappen bevatten, die weer bestanden en mappen kunnen bevatten. Eén eenvoudige regel, en niets in die regel zegt hoe diep de mappen mogen gaan.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De Penn Treebank, uit *Woordsoorten en woordgroepen*, heeft voor elke zin een boom, en uit die bomen kun je regels aflezen. Eén regel zegt dat een werkwoordgroep (VP, van *verb phrase*), een werkwoord met de woorden die erbij horen, kan bestaan uit een werkwoord gevolgd door een naamwoordgroep, zoals *left the city* (verliet de stad). Jurafsky en Martin geven het aantal verschillende regels voor een VP in de grammatica van de Penn Treebank. Ongeveer hoeveel zijn het er?",
        options={
            "few": "Minder dan vijftig",
            "some": "Ongeveer 500",
            "right": "Ongeveer 4.500",
        },
        reveal="**Ongeveer 4.500.** Die grammatica is heel plat, schrijven Jurafsky en Martin: er zijn aparte regels voor elk aantal voorzetselgroepen achter elkaar (PP, van *prepositional phrase*), zoals *after midnight* (na middernacht), en voor elke volgorde van de woorden rond het werkwoord. Zo zijn er VP → VBD PP, VP → VBD PP PP, VP → VBD PP PP PP enzovoort, waarbij VBD de tag van de Penn Treebank is voor een werkwoord in de verleden tijd. De course voegt eraan toe dat twee regels die allemaal kunnen maken: VP → VBD PP, en VP → VP PP, die je opnieuw kunt toepassen op wat deze regel zelf maakte. Deze les laat zien hoe zulke regels werken.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Constituenten

    In *de late trein verliet de stad na middernacht* horen sommige woorden nauwer bij elkaar dan andere. In plaats van *na middernacht* kun je één woord zetten, *toen*: *de late trein verliet de stad toen*. In plaats van *stad na* past geen enkel woord, en die twee woorden kunnen ook niet samen naar een andere plek. Een groep woorden die in een zin als één geheel werkt, heet een **constituent**. Jurafsky en Martin omschrijven het zo: groepen woorden kunnen zich gedragen als één eenheid.

    Hoe zie je dat? Het leerboek *Essentials of Linguistics* (Catherine Anderson en anderen, tweede druk, 2022) geeft vier soorten tests:

    - **Vervangen** door één woord. Een naamwoordgroep kan een voornaamwoord worden, zoals *die* of *ze*, en een groep die zegt wanneer, zoals *na middernacht*, kan *toen* worden.
    - **Verplaatsen**: *Na middernacht verliet de late trein de stad.*
    - **Een gespleten zin** maken, met de groep in *het was ... dat*: *Het was na middernacht dat de late trein de stad verliet.*
    - **Antwoorden** op een vraag met alleen de groep: *Wanneer verliet de late trein de stad? Na middernacht.*

    Veel leerboeken voegen een test met *en* toe, maar volgens de auteurs is die niet altijd betrouwbaar. En een test zegt alleen iets over de zin die je test, schrijven de auteurs: dezelfde woorden kunnen in de ene zin een constituent zijn en in de andere niet.

    De Nederlandse schoolgrammatica heeft een eigen test. In *Kleine ABN-syntaxis* (1961) zegt P.C. Paardekooper het zo: de grootste groep die vóór de persoonsvorm kan staan, is één zinsdeel. De persoonsvorm is de vorm van het werkwoord die verandert met het onderwerp (*ik verliet*, *wij verlieten*). In *Na middernacht verliet de late trein de stad* staat *na middernacht* vóór *verliet*: één zinsdeel. De test heeft uitzonderingen, voor werkwoorden en voor een paar korte woorden, zoals *niet*.

    ### Woordgroepen in woordgroepen

    Constituenten zitten in elkaar. In de grammatica van deze les bevat *de late trein* een kleinere groep, *late trein*, en die bevat weer *trein*. Teken je de groepen met de hele zin bovenaan en de woorden onderaan, dan krijg je een boom: een **ontleedboom**, zoals in de afbeelding hieronder. De Penn Treebank heeft zo'n boom voor elke zin; Universal Dependencies tekent een ander soort boom, en dat laat de volgende les zien.

    ### Regels die zinnen bouwen

    Een **contextvrije grammatica** beschrijft constituenten met regels. Elke regel zegt dat je het symbool links van de pijl mag herschrijven als de rij symbolen rechts ervan, de **rechterkant**. Dit zijn de regels van deze les:

    ```
    S   → NP VP
    NP  → DET NOM  |  PROPN
    NOM → NOUN  |  ADJ NOM
    VP  → VERB NP  |  VERB NP PP
    PP  → ADP NP
    ```

    Lees de pijl, schrijven Jurafsky en Martin, als: herschrijf het symbool links als de rij symbolen rechts. De verticale streep `|` scheidt keuzes: een NP is een DET en een NOM, of een PROPN. S staat voor een zin (*sentence*), NP voor een naamwoordgroep, VP voor een werkwoordgroep en PP voor een voorzetselgroep, zoals *na middernacht*. NOM is het deel van een naamwoordgroep na het lidwoord of een ander woord met de tag DET: een zelfstandig naamwoord, met eventuele bijvoeglijke naamwoorden ervoor. De symbolen vlak boven de woorden zijn de tags van *Woordsoorten en woordgroepen*. Een tweede set regels, het **lexicon**, geeft elke tag de woorden: DET → *de* | *een*, NOUN → *trein* | *brug* | *stad*, enzovoort.

    De symbolen voor groepen, zoals NP, heten **non-terminals**. De woorden zijn **terminals**: daar houdt het herschrijven op. Elke zin begint bij één non-terminal, het **startsymbool**, hier S. In de formele definitie van Jurafsky en Martin heeft een contextvrije grammatica deze vier delen: de non-terminals, de terminals, de regels en het startsymbool. De grammatica heet contextvrij omdat een regel één symbool herschrijft, wat er ook omheen staat. De regels zijn van de course zelf, naar de kleine grammatica L0 van Jurafsky en Martin, met woorden van de course.

    ### Van S naar een zin

    Begin bij S, en herschrijf telkens één non-terminal, tot er alleen woorden over zijn. Deze les herschrijft altijd de eerste non-terminal van links:

    ```
    S
    NP VP
    DET NOM VP
    de NOM VP
    de ADJ NOM VP
    de late NOM VP
    ...
    de late trein verliet de stad
    ```

    De rij stappen van S naar de zin is een **afleiding**. Bij veel stappen is er een keuze: NOM kan NOUN worden, of ADJ NOM. Nummer de rechterkanten van elk symbool 0, 1, 2 enzovoort, en een lijst met getallen zegt welke regel je bij elke stap neemt: deze zin kost 13 stappen. De afbeelding toont dezelfde afleiding als boom: elke herschrijfstap hangt kinderen onder een symbool.

    ### Eindeloze zinnen

    Kijk naar NOM → ADJ NOM. NOM staat aan beide kanten van de pijl, dus je kunt de regel opnieuw toepassen op wat deze regel net maakte: NOM, ADJ NOM, ADJ ADJ NOM, enzovoort, zoals in *de late lange late trein*. Er is geen langste zin: één bijvoeglijk naamwoord meer maakt altijd een langere. Een regel kan ook via andere regels terugkomen bij het eigen symbool. Voeg VP → VERB *that* S toe, en een Engelse zin bevat een werkwoordgroep die weer een zin bevat: *you know that I think that the train left* (je weet dat ik denk dat de trein vertrok). Zo maakt een grammatica met een paar regels en een paar woorden eindeloos veel zinnen.

    In het Nederlands werkt die laatste regel niet zomaar: na *dat* komt het werkwoord achteraan, zoals in *ik weet dat de trein de stad verliet*. Een grammatica van het Nederlands heeft voor zo'n bijzin eigen regels nodig.

    ### Wie het bedacht

    Het idee dat een zin uiteenvalt in constituenten gaat terug op de psycholoog Wilhelm Wundt (1900), schrijven Jurafsky en Martin. Noam Chomsky maakte het in 1956 formeel, als frasestructuurgrammatica. Kort daarna kwamen volgens Jurafsky en Martin John Backus (1959) en, los daarvan, Peter Naur en anderen (1960) op hetzelfde soort grammatica uit, om de programmeertaal ALGOL te beschrijven. De schrijfwijze van die regels heet *Backus-Naur form*, of BNF, en volgens Jurafsky en Martin is die gelijkwaardig aan een contextvrije grammatica. Ook noemen Jurafsky en Martin contextvrije grammatica's de ruggengraat van veel formele modellen van de syntaxis van menselijke talen, en van computertalen.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **constituent** | een groep woorden die in een zin als één geheel werkt, zoals *de late trein* |
        | **ontleedboom** | de constituenten van een zin, getekend als boom, met S bovenaan en de woorden onderaan |
        | **contextvrije grammatica** | regels die elk één symbool herschrijven als een rij symbolen, wat er ook omheen staat; afgekort CFG, van *context-free grammar* |
        | **rechterkant** | de rij symbolen rechts van de pijl van een regel |
        | **non-terminal**, **terminal** | een symbool voor een groep, zoals NP, en een woord, waar het herschrijven ophoudt |
        | **startsymbool** | de non-terminal waar elke zin bij begint, hier S |
        | **lexicon** | de regels die elke tag de woorden geven |
        | **afleiding** | de stappen van S naar een zin, waarbij elke stap één symbool herschrijft |
        | `GRAMMAR` | de grammatica van de les als dict: elke non-terminal is een key, met de lijst van de rechterkanten ervan, elk een lijst met symbolen |
        | `a + b` | met twee lijsten: een nieuwe lijst met de items van `a`, en dan die van `b` |
        | `return` in een loop | beëindigt de functie meteen, midden in de loop |
        """),
        L.watch_out("*Woorden en hun delen* noemde het maken van een nieuw woord uit een ander *derivatie*, of ook *afleiding*, zoals *schoonheid* uit *schoon*. In een grammatica betekent *afleiding* de stappen van S naar een zin: hetzelfde woord, een andere betekenis."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de grammatica als dict: elke non-terminal is een key, met de lijst van de rechterkanten ervan.
    2. Voor een stap zoek je de eerste non-terminal, neem je een van de rechterkanten ervan, en zet je die op de plaats van het symbool.
    3. Begin bij `["S"]`, doe voor elke keuze uit de lijst een stap, en bewaar elke rij.
    4. Teken de rijen als boom, en zoek een regel die terugkomt bij het eigen symbool.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel is de grammatica van de les als dict. De tweede definieert `first_open`: deze functie geeft de plaats van de eerste non-terminal terug, of `None` als er geen is. Daarna herschrijft de cel één stap met de hand.
    """)
    return


@app.cell
def grammar_cell():
    GRAMMAR = {
        "S": [["NP", "VP"]],
        "NP": [["DET", "NOM"], ["PROPN"]],
        "NOM": [["NOUN"], ["ADJ", "NOM"]],
        "VP": [["VERB", "NP"], ["VERB", "NP", "PP"]],
        "PP": [["ADP", "NP"]],
        "DET": [["de"], ["een"]],
        "ADJ": [["late"], ["lange"]],
        "NOUN": [["trein"], ["brug"], ["stad"]],
        "PROPN": [["Utrecht"], ["Breda"]],
        "VERB": [["passeerde"], ["verliet"]],
        "ADP": [["bij"], ["na"]],
    }
    return (GRAMMAR,)


@app.cell
def example_first(GRAMMAR):
    def first_open(symbols, grammar):
        for i in range(len(symbols)):
            if symbols[i] in grammar:
                return i
        return None

    print(first_open(["de", "NOM", "VP"], GRAMMAR))
    print(first_open(["de", "late", "trein"], GRAMMAR))
    print(["de"] + GRAMMAR["NOM"][1] + ["VP"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *de* is een woord, dus de eerste non-terminal van `["de", "NOM", "VP"]` staat op plaats 1. In `["de", "late", "trein"]` is elk symbool een woord: er valt niets meer te herschrijven, en `first_open` geeft `None` terug. De `return` in de loop stopt de functie bij de eerste non-terminal. De laatste coderegel zette de tweede rechterkant van NOM, `["ADJ", "NOM"]`, met de hand op de plaats van NOM. Oefening 1 doet dat op elke plaats.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rewrite():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. rewrite_first: één stap

        Schrijf `rewrite_first(symbols, grammar, k)`: zoek met `first_open` de plaats `i` van de eerste non-terminal in `symbols`, neem rechterkant nummer `k` ervan uit `grammar`, en geef een nieuwe lijst terug: de symbolen vóór plaats `i`, dan die rechterkant, dan de symbolen na plaats `i`. `symbols` heeft hier altijd een non-terminal. De check geeft je functie `first_open` mee, de functie uit het uitgewerkte voorbeeld. Vervang de `...`. De check roept `rewrite_first(["NP", "VP"], GRAMMAR, 0)` aan; daarna eigen gevallen, ook met kleine grammatica's van letters.
        """),
        L.task("Zet rechterkant nummer `k` op de plaats van de eerste non-terminal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rewrite():
    box_rewrite = L.code_box("rewrite_first", lines=4)
    box_rewrite
    return (box_rewrite,)


@app.cell(hide_code=True)
def check_rewrite(box_rewrite):
    result_rewrite = L.check("rewrite_first", box_rewrite.value)
    result_rewrite
    return (result_rewrite,)


@app.cell(hide_code=True)
def hints_rewrite():
    L.hints(
        "rewrite_first",
        [
            "Drie coderegels: zoek de plaats met `first_open`, zoek de rechterkant op in `grammar`, en plak drie lijsten aan elkaar met `+`.",
            "Welk symbool mag niet terugkomen in de nieuwe lijst? En welk deel van `symbols` komt na de rechterkant?",
            "`    return symbols[:i] + right + ___`",
        ],
        why="`i = first_open(symbols, grammar)`, dan `right = grammar[symbols[i]][k]`, en `return symbols[:i] + right + symbols[i + 1:]`. `rewrite_first([\"NP\", \"VP\"], GRAMMAR, 0)` geeft `['DET', 'NOM', 'VP']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rewrite():
    mo.md(r"""
    In de afbeelding onder de check zet je functie de 13 stappen van de zin uit *Het idee*. De woorden verschijnen van links naar rechts, omdat elke stap de eerste non-terminal herschrijft: *de late trein* is klaar voordat de werkwoordgroep begint.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_derive():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. derive: van S naar een zin

        Schrijf `derive(grammar, choices)`: begin met `symbols`, de lijst `["S"]`, en een lijst `steps` met deze eerste rij erin. Herschrijf `symbols` voor elk getal `k` in `choices` met `rewrite_first`, en voeg de nieuwe rij toe aan `steps`. Geef aan het eind `steps` terug. De check geeft je functie de `rewrite_first` van de course mee. Vervang de `...`. De check roept eerst `derive({"S": [["a", "S", "b"], ["c"]]}, [0, 0, 1])` aan: een grammatica met een regel die terugkomt bij S, en die *c*, *a c b*, *a a c b b* enzovoort maakt. Daarna probeert de check een grammatica voor sommen, zoals *x + ( y )*, helemaal geen keuzes, en de grammatica van de les.
        """),
        L.task("Herschrijf `[\"S\"]` één keer voor elke keuze, en bewaar elke rij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_derive():
    box_derive = L.code_box("derive", lines=7)
    box_derive
    return (box_derive,)


@app.cell(hide_code=True)
def check_derive(box_derive):
    result_derive = L.check("derive", box_derive.value)
    result_derive
    return (result_derive,)


@app.cell(hide_code=True)
def hints_derive():
    L.hints(
        "derive",
        [
            "Maak de twee lijsten vóór de loop. In de loop staan twee coderegels: de ene herschrijft, de andere voegt toe.",
            "Wat moet `derive(grammar, [])` teruggeven, zonder keuzes?",
            "`    steps = ___`",
        ],
        why="`symbols = [\"S\"]` en `steps = [symbols]`, dan `for k in choices:` met daaronder `symbols = rewrite_first(symbols, grammar, k)` en `steps.append(symbols)`; na de loop `return steps`. `derive({\"S\": [[\"a\", \"S\", \"b\"], [\"c\"]]}, [0, 0, 1])` geeft `[['S'], ['a', 'S', 'b'], ['a', 'a', 'S', 'b', 'b'], ['a', 'a', 'c', 'b', 'b']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_derive():
    mo.md(r"""
    De boom onder de check hoort bij een langere zin van de grammatica, met 17 keuzes. Een naamwoordgroep zit in de werkwoordgroep, en een andere in de voorzetselgroep *bij Breda*: woordgroepen in woordgroepen. De grammatica van letters deed hetzelfde met S zelf: *a a c b b* heeft een S in een S in een S.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: eindeloze zinnen

    De tabel telt de zinnen die de grammatica van de les maakt, naar het aantal woorden. Zet de regel NOM → ADJ NOM uit, en kijk hoe de tellingen ophouden. Je kunt `GRAMMAR` ook veranderen in de cel van het uitgewerkte voorbeeld, bijvoorbeeld door een woord aan het lexicon toe te voegen; de tabel volgt.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_rule = mo.ui.checkbox(value=True, label="Met de regel NOM → ADJ NOM")
    demo_words = mo.ui.slider(3, 14, value=10, label="Woorden, hoogstens", show_value=True)
    demo_new = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Andere voorbeelden")
    mo.hstack([demo_rule, demo_words, demo_new], justify="start", gap=1.5, wrap=True)
    return demo_new, demo_rule, demo_words


@app.cell(hide_code=True)
def demo_look(GRAMMAR, demo_new, demo_rule, demo_words):
    mo.Html(C.demo_picture(GRAMMAR, demo_rule.value, demo_words.value, demo_new.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_constituents():
    mo.md(r"""
    ### Oefening 3. Welke groepen zijn constituenten?

    Neem de zin *De oude veerboot verliet de haven vóór zonsopgang.* Zet elke groep woorden onder constituent of geen constituent. Probeer op elke groep de tests uit *Het idee*.
    """)
    return


@app.cell(hide_code=True)
def exercise_constituents():
    sort_constituents = L.sort("constituents")
    sort_constituents
    return (sort_constituents,)


@app.cell(hide_code=True)
def check_constituents(sort_constituents):
    result_constituents = L.check("constituents", sort_constituents)
    result_constituents
    return (result_constituents,)


@app.cell(hide_code=True)
def text_exercise_endless():
    mo.md(r"""
    ### Oefening 4. Eén regel erbij

    Neem de grammatica van de les zonder NOM → ADJ NOM. Dan komt geen regel terug bij het eigen symbool, en maakt de grammatica een beperkt aantal zinnen, zoals de demo liet zien.
    """)
    return


@app.cell(hide_code=True)
def choice_endless_cell():
    choice_endless = L.choice(
        "endless",
        question="Welke ene regel maakt, toegevoegd aan die grammatica, het aantal zinnen weer eindeloos?",
        options={
            "right": "NP → DET NOM PP",
            "vp": "VP → VERB NP PP PP",
            "pp": "PP → ADP DET NOUN",
            "s": "S → PROPN VP",
        },
        correct="right",
        why={
            "right": "Een naamwoordgroep bevat nu een PP, en PP → ADP NP bevat weer een naamwoordgroep: NP komt via PP terug bij NP, zoals in *de brug bij de stad bij de brug bij ...*. Een kring via twee regels is net zo eindeloos als een kring in één regel.",
            "vp": "VP → VERB NP PP PP maakt langere zinnen, maar NP en PP leiden geen van beide terug naar VP, dus er is nog steeds een langste zin. Zoek een symbool dat bij zichzelf terugkomt. Zie *Eindeloze zinnen* in *Het idee*.",
            "pp": "PP → ADP DET NOUN eindigt in tags die elk één woord worden, dus via deze regel komt geen symbool bij zichzelf terug. Zie *Eindeloze zinnen* in *Het idee*.",
            "s": "PROPN en VP leiden niet terug naar S, dus S → PROPN VP maakt nieuwe zinnen, maar geen eindeloze. Zie *Eindeloze zinnen* in *Het idee*.",
        },
    )
    choice_endless
    return (choice_endless,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) gaf elk woord een tag, en vond een eerste woordgroep. Hier werden de tags de symbolen vlak boven de woorden van een boom. Constituenten zijn groepen woorden die als één geheel werken, zoals tests met vervangen, verplaatsen en antwoorden laten zien; ze zitten in elkaar, en een ontleedboom tekent ze. Een contextvrije grammatica beschrijft ze met regels die telkens één symbool herschrijven, van S naar een zin. Je schreef één stap en een hele afleiding, en zag dat een regel die terugkomt bij het eigen symbool, direct of via andere regels, eindeloos veel zinnen maakt. De volgende les tekent een ander soort boom, waarin elk woord hangt aan het woord waar het van afhangt.

    **Een voorbeeld uit het dagelijks leven.** Je voorouders zijn je ouders, en de voorouders van je ouders: één regel die zichzelf gebruikt, en toch reikt die zo ver terug als je wilt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_endless, result_constituents, result_derive, result_rewrite):
    L.wrap_up(result_rewrite, result_derive, result_constituents, choice_endless)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
