# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-coreference. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-coreference", "nl")
    C = L.checks   # de tekst over het museum, de mentions van LitBank, de getallen van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl) en [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_old():
    quick_old = L.choice(
        "quick-old",
        question="In *Een tekst is meer dan losse zinnen* is een mention oud als de tekst dat ding al eerder noemde. Welke mention van de tweede zin van *Een storm trof de stad. De storm blies het dak van een schuur.* is oud?",
        options={
            "roof": "*het dak*, want elke mention met *het* of *de* is oud",
            "shed": "*een schuur*, want een stad heeft schuren",
            "right": "*De storm*: de tekst noemde de storm al",
            "none": "Geen: elke zin noemt eigen dingen",
        },
        correct="right",
        why={
            "right": "De eerste zin bracht de storm binnen als *een storm*, en *de storm* pakt dezelfde storm weer op: twee mentions van één storm. Deze les zoekt zulke mentions in een hele tekst, ook als de latere een voornaamwoord is, zoals *het*.",
            "roof": "*De* en *het* horen vaak bij iets ouds, maar de tekst noemde nog geen dak. Welk ding noemde de eerste zin? Zie [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl).",
            "shed": "*Een* brengt de schuur binnen als iets nieuws. Welk ding noemde de eerste zin? Zie [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl).",
            "none": "Een mention is oud als de tekst het ding al eerder noemde, ook in een eerdere zin. Welk ding van de tweede zin noemde de eerste zin al? Zie [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl).",
        },
    )
    quick_old
    return (quick_old,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="Een tagger voor namen zoals in *Namen vinden* leest *Emma Woodhouse lived in Highbury. Emma rarely left it.* Welke spans markeert de tagger, elk met een type?",
        options={
            "it": "De namen, en *it* ook als plaats",
            "right": "*Emma Woodhouse*, *Highbury* en *Emma*",
            "first": "Alleen *Emma Woodhouse* en *Highbury*, want een naam telt één keer",
            "capitals": "Elk woord met een hoofdletter, woord voor woord",
        },
        correct="right",
        why={
            "right": "NER markeert de spans van namen: *Emma Woodhouse* en *Emma* als persoon, *Highbury* als plaats. NER zegt niet dat *Emma* dezelfde persoon is als *Emma Woodhouse*, of dat *it* Highbury is. Dat is coreferentie, het onderwerp van deze les.",
            "it": "*It* is een voornaamwoord, geen naam, en een tagger voor namen laat het weg. Welke spans zijn namen? Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
            "first": "De tagger markeert elke span die een naam is, ook een naam die al eerder kwam. Hoeveel namen bevat de tekst? Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
            "capitals": "De tagger markeert spans, en één span kan meer woorden bevatten: een voornaam en een achternaam samen. Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tekst noemt hetzelfde ding steeds opnieuw, en zelden met dezelfde woorden: eerst een naam, dan een beschrijving, dan een voornaamwoord als *het*. Daniel Jurafsky en James Martin beginnen het hoofdstuk over coreferentie met de gedachte dat een belangrijk deel van het werken met taal is: weten over wie een tekst het heeft. Een programma dat vragen over een tekst beantwoordt, of een tekst samenvat, moet weten welke woorden over dezelfde persoon of hetzelfde ding gaan. Deze les groepeert de mentions van een tekst naar waar de mentions naar verwijzen, en probeert een eenvoudige regel voor waar *it* naar verwijst.

    **Een voorbeeld uit het dagelijks leven.** Een briefje zegt *Het boek ligt op het bureau. Wil je het meenemen?* Je leest *het* meteen als het boek, hoewel het bureau later kwam en ook een ding is: niemand neemt even een bureau mee.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="In 100 romans en verhalen markeerden mensen welke mentions naar dezelfde persoon of hetzelfde ding verwijzen. De course nam elke *it*, *its* en *itself* die de makers aan een eerdere mention koppelden: 318 gevallen. Een regel kiest de dichtstbijzijnde eerdere naam, beschrijving of voornaamwoord dat een ding is en geen persoon. In hoeveel van elke 100 gevallen kiest de regel iets wat naar hetzelfde verwijst?",
        options={
            "low": "Ongeveer 25 van elke 100",
            "half": "Ongeveer 45 van elke 100",
            "right": "Ongeveer 68 van elke 100",
            "high": "Ongeveer 95 van elke 100: bijna altijd",
        },
        reveal="**Ongeveer 68:** 216 van de 318, 0,679, met niets dan de volgorde van de tekst en twee soorten. Zonder te vragen om een ding heeft de dichtstbijzijnde eerdere mention er 137 goed, 0,431.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Mentions, entiteiten, ketens

    [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl) noemde elke keer dat een tekst een ding noemt of ernaar verwijst een mention. Volgens Jurafsky en Martin corefereren twee of meer verwijzende uitdrukkingen als elke uitdrukking naar dezelfde entiteit in het discours verwijst. Een verwijzende uitdrukking is een mention; een **entiteit** is de persoon of het ding waar de tekst over spreekt, zoals die persoon of dat ding in de tekst vorm krijgt. De verzameling mentions die corefereren heet volgens Jurafsky en Martin vaak een coreference chain of cluster; in deze les een **keten**. De afbeelding toont de tekst van de course, met elke mention in een kader en de letter van de keten eronder.
    """)
    return


@app.cell(hide_code=True)
def museum_cell():
    mo.Html(C.museum_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_anaphor():
    mo.md(r"""
    Acht mentions, vijf entiteiten. Vier mentions zijn het schilderij: een beschrijving, dan *It*, dan twee keer *it*. Een entiteit met maar één mention, zoals hier het museum, heet een **singleton**. De ketens vinden heet **coreference resolution**: volgens Jurafsky en Martin de taak om te bepalen of twee mentions corefereren, dus of de twee naar dezelfde entiteit in het discoursmodel verwijzen. Het begint met de mentions vinden; deze les begint bij mentions die mensen markeerden.

    ### Anafoor en antecedent

    Jurafsky en Martin noemen een verwijzing in een tekst naar een entiteit die al eerder in het discours voorkwam anafora, en de uitdrukking die verwijst een **anafoor**. De eerdere mention van dezelfde entiteit is het **antecedent**. In de tekst zijn de drie keer *it* anaforen; een naam die terugkomt, is er ook een. In een voetnoot schrijven Jurafsky en Martin dat het boek het gangbare gebruik in NLP volgt: een anafoor is elke mention met een antecedent. In het smallere gebruik dat het boek opzijzet, zijn alleen mentions als voornaamwoorden, waarvan de betekenis van het antecedent afhangt, anaforen.

    ### LitBank

    De data van deze les is LitBank, van David Bamman, Olivia Lewke en Anya Mansoor (2020): ongeveer 2.000 woorden uit elk van 100 Engelstalige romans en verhalen, van 1719 tot 1922. Mensen markeerden met de hand elke mention van zes typen entiteiten (personen, bouwwerken, locaties, geopolitieke entiteiten, organisaties en voertuigen) en groepeerden de mentions in ketens. Volgens de makers gaat de grote meerderheid van de mentions, 83,1 procent, over personen. De kopie van de course bevat 28.411 mentions in 7.235 ketens; 5.071 van die ketens zijn singletons. Bamman en anderen tellen 17,4 procent van alle mentions als singleton, tegen 56 procent die Recasens en anderen (2013) vonden in OntoNotes, een dataset met vooral nieuws, gesprekken, de Bijbel en teksten van het web. De course telt 0,178.

    ### De dichtstbijzijnde eerdere mention die past

    Welk antecedent heeft een voornaamwoord? Jurafsky en Martin noemen wat een lezer helpt kiezen. Een ervan is recentheid: entiteiten die net genoemd zijn, staan meer in beeld dan entiteiten van verder terug. Het voorbeeld in het boek: *The doctor found an old map in the captain's chest. Jim found an even older map hidden on the shelf. It described an island.* Daar is *it* waarschijnlijker de kaart van Jim. Een ander is dat *it* naar een ding verwijst en niet naar een persoon: in *Maryam has a theorem. It is exciting.* is *it* de stelling, niet Maryam.

    Samen geeft dat een regel: neem de dichtstbijzijnde eerdere mention die past. In deze les heeft elke mention een van twee **soorten**: `"person"` of `"thing"`. *It*, *its* en *itself* zijn dingen; elk ander persoonlijk voornaamwoord, zoals *I*, *you* of *they*, is een persoon. Een naam of beschrijving is een persoon als LitBank de mention zo markeert, en anders een ding. De afbeelding past de regel toe op de drie keer *it* in de tekst, een keer zonder de soort en een keer met de soort.
    """)
    return


@app.cell(hide_code=True)
def rule_cell():
    mo.Html(C.rule_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_back():
    mo.md(r"""
    Met de soort heeft de regel er twee van de drie goed. Bij de tweede *it* is het dichtstbijzijnde eerdere ding de hal, en een lezer neemt het schilderij: een bewaker hangt een scheef schilderij recht, geen hal. De laatste *it* is met de soort goed, al kiest de regel een andere *it*: het telt dat de keuze naar dezelfde entiteit verwijst.

    Hoe ver kijkt een voornaamwoord terug? Volgens een voetnoot van Jurafsky en Martin verwijzen voornaamwoorden meestal, maar niet altijd, naar entiteiten die hooguit een of twee zinnen eerder in het discours kwamen, terwijl naamwoordgroepen met een bepaald lidwoord, zoals *het boek*, vaak verder terug verwijzen. De afbeelding telt dat voor *it*, *its* en *itself* in LitBank.
    """)
    return


@app.cell(hide_code=True)
def back_cell():
    mo.Html(C.back_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        Niet elke *it* verwijst ergens naar. Jurafsky en Martin schrijven dat voornaamwoorden als *it* in het Engels, en vergelijkbare voornaamwoorden in andere talen, vaak niet verwijzen. Het eerste voorbeeld is *it is raining*, in het Nederlands *het regent*. *Pride and Prejudice* begint met zo'n *it*: *It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.* En LitBank markeert maar zes typen entiteiten. De teksten bevatten 2.306 keer *it*, *its* of *itself*; LitBank markeert er 333 als mention, en 318 daarvan hebben een eerdere mention in de keten.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **entiteit** | de persoon of het ding waar een tekst over spreekt, zoals die persoon of dat ding in de tekst vorm krijgt; in het Engels entity |
        | **coreferentie**, **corefereren** | twee mentions corefereren als de twee naar dezelfde entiteit verwijzen; in het Engels coreference, corefer |
        | **keten** | de mentions van één entiteit, in de volgorde van de tekst; in het Engels coreference chain of cluster |
        | **singleton** | een entiteit met maar één mention |
        | **coreference resolution** | vinden welke mentions van een tekst corefereren, en zo de ketens |
        | **anafoor** | een mention met een eerdere mention van dezelfde entiteit; in het Engels anaphor |
        | **antecedent** | de eerdere mention van dezelfde entiteit; in deze les de dichtstbijzijnde |
        | **soort** | in deze les `"person"` of `"thing"`: waar een voornaamwoord bij moet passen; in het Engels kind |
        | `C.MUSEUM` | de acht mentions van de tekst, op volgorde, elk een paar `(text, kind)`, zoals `("a painting", "thing")` |
        | `C.MUSEUM_IDS` | de keten van elke mention van `C.MUSEUM`, op dezelfde plaats: `"museum"`, `"painting"`, `"painting"`, enzovoort |
        """),
        L.watch_out("Een keten gaat over waar de woorden naar verwijzen, niet over de woorden zelf: *a painting* en *It* zitten in één keten, en hetzelfde woord kan in twee ketens zitten, zoals in *The truck pulled a trailer. It towed it up the hill.* En in NLP is een anafoor elke mention met een antecedent, ook een naam die terugkomt, niet alleen een voornaamwoord."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Ketens: loop langs de mentions met de keten van elke mention. Een nieuwe keten krijgt een lijst met de tekst van de mention; bij een keten die al eerder kwam, komt de tekst achteraan in de lijst.
    2. Antecedenten: loop langs de ketens van de mentions, en houd in een dict voor elke keten de plaats van de laatste mention tot nu toe bij. Een mention waarvan de keten in de dict staat, krijgt die plaats; een mention met een nieuwe keten krijgt `None`. Daarna leert de dict de plaats van deze mention.
    3. De regel: loop langs de plaatsen vóór mention `i`, en onthoud elke plaats waarvan de mention de soort van mention `i` heeft. De laatst onthouden plaats is de dichtstbijzijnde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de soorten van de mentions van de tekst en print de aantallen. Daarna print de cel de mention op plaats 5 en de keten ervan.
    """)
    return


@app.cell
def example_kinds():
    counts = {"person": 0, "thing": 0}
    for text, kind in C.MUSEUM:
        counts[kind] = counts[kind] + 1
    print(counts)
    print(C.MUSEUM[5], C.MUSEUM_IDS[5])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zes van de acht mentions zijn dingen, dus een *it* heeft meerdere kandidaten. Op plaats 5 staat de tweede *it*: een ding, in de keten van het schilderij. Oefening 1 verzamelt de mentions van elke keten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chains():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. chains: de mentions van elke entiteit

        Schrijf `chains(mentions, ids)`: `mentions` bevat de mentions van een tekst op volgorde, elk een paar `(text, kind)`, en `ids` bevat de keten van elke mention, op dezelfde plaats. Geef een dict terug met voor elke keten de lijst met de teksten van de mentions in die keten, in de volgorde van de tekst. Vervang de `...`. De check roept `chains(C.MUSEUM, C.MUSEUM_IDS)` aan, en dat geeft `{"museum": ["The museum"], "painting": ["a painting", "It", "it", "it"], "hall": ["the hall"], "guard": ["A guard"], "visitors": ["visitors"]}`. Daarna eigen gevallen: geen mentions, elke mention in een eigen keten, ketens met de nummers `0` en `1` zoals in LitBank met één tekst twee keer in een keten, het woord *it* in twee ketens, en één keten voor alle mentions.
        """),
        L.task("Verzamel de teksten van de mentions van elke keten, op volgorde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chains():
    box_chains = L.code_box("chains", lines=10)
    box_chains
    return (box_chains,)


@app.cell(hide_code=True)
def check_chains(box_chains):
    result_chains = L.check("chains", box_chains.value)
    result_chains
    return (result_chains,)


@app.cell(hide_code=True)
def hints_chains():
    L.hints(
        "chains",
        [
            "Bij elke mention heeft `found` al een lijst voor de keten `c`, of nog niet: een `if` met een `else`.",
            "`c in found` vraagt of `found` al een lijst voor `c` heeft. Zo ja, dan zet `found[c].append(text)` de tekst achteraan. Zo nee, dan krijgt `found[c]` een nieuwe lijst: wat moet er meteen in die lijst staan?",
            "De regel onder `else:`: `            found[c] = ___`",
        ],
        why="Vervang de `...` door `if c in found:` met `found[c].append(text)` eronder, en dan `else:` met `found[c] = [text]` eronder. Het voorbeeld geeft `{\"museum\": [\"The museum\"], \"painting\": [\"a painting\", \"It\", \"it\", \"it\"], \"hall\": [\"the hall\"], \"guard\": [\"A guard\"], \"visitors\": [\"visitors\"]}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_chains():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op de 366 mentions van het begin van *Pride and Prejudice*, van Jane Austen (1813), en toont de vijf langste ketens. De langste, Mr. Bennet, heeft 103 mentions, en de meeste daarvan zijn voornaamwoorden. In heel LitBank zijn 15.815 van de 28.411 mentions voornaamwoorden. Oefening 2 zoekt bij elke mention de dichtstbijzijnde eerdere mention van dezelfde keten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_antecedents():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. antecedents: de dichtstbijzijnde eerdere mention van dezelfde keten

        Schrijf `antecedents(ids)`: `ids` bevat de keten van elke mention van een tekst, op volgorde. Geef een lijst terug met voor elke mention de plaats van de dichtstbijzijnde eerdere mention van dezelfde keten, of `None` als de keten daar nieuw is. `last` bevat voor elke keten die al langskwam de plaats van de laatste mention; de regel onder de `...` houdt `last` bij. Vervang de `...`. De check roept `antecedents(C.MUSEUM_IDS)` aan, en dat geeft `[None, None, 1, None, None, 2, None, 5]`: *It* op plaats 2 gaat terug naar *a painting* op plaats 1, en de laatste *it* naar de *it* op plaats 5. Daarna eigen gevallen: geen mentions, elke mention in een eigen keten, één keten drie keer, ketens met nummers zoals in LitBank, een keten binnen een andere, en een keten waarvan de eerdere mention op plaats `0` staat.
        """),
        L.task("Geef de plaats van de dichtstbijzijnde eerdere mention van dezelfde keten, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_antecedents():
    box_antecedents = L.code_box("antecedents", lines=10)
    box_antecedents
    return (box_antecedents,)


@app.cell(hide_code=True)
def check_antecedents(box_antecedents):
    result_antecedents = L.check("antecedents", box_antecedents.value)
    result_antecedents
    return (result_antecedents,)


@app.cell(hide_code=True)
def hints_antecedents():
    L.hints(
        "antecedents",
        [
            "Zet een `if` met een `else` boven de regel die `last` bijhoudt: de ene tak voegt aan `out` de plaats toe die `last` voor de keten `ids[k]` bevat, de andere voegt `None` toe.",
            "Onder de `if`: `out.append(last[ids[k]])`. Onder de `else`: `out.append(None)`. De `if` kijkt in `last` voordat `last` de plaats van mention `k` leert.",
            "`        if ___:`",
        ],
        why="Vervang de `...` door `if ids[k] in last:` met `out.append(last[ids[k]])` eronder, en dan `else:` met `out.append(None)` eronder. De regel `last[ids[k]] = k` blijft eronder staan. Het voorbeeld geeft `[None, None, 1, None, None, 2, None, 5]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_antecedents():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op de ketens van alle 100 werken. Van de voornaamwoorden heeft 0,956 een eerdere mention in de keten, tegen 0,712 van de namen en 0,393 van de beschrijvingen: een beschrijving brengt vaak iemand of iets nieuws binnen. Volgens Bamman en anderen ligt het dichtstbijzijnde antecedent van een voornaamwoord meestal dichtbij: de mediaan is 2, 90 procent ligt binnen 5 mentions en 95 procent binnen 9. In de kopie van de course, voor de voornaamwoorden met een antecedent: een mediaan van 2 mentions terug, 0,912 binnen 5 en 0,956 binnen 9. Oefening 3 raadt een antecedent zoals een programma zonder de ketens dat moet doen: met de regel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. pick: de dichtstbijzijnde eerdere mention van dezelfde soort

        Schrijf `pick(mentions, i)`: `mentions` bevat mentions als paren `(text, kind)`, en `i` is een plaats in die lijst. Geef de plaats terug van de dichtstbijzijnde mention vóór plaats `i` met dezelfde soort als mention `i`, of `None` als geen enkele mention vóór `i` die soort heeft. Vervang de `...`. De check roept `pick(C.MUSEUM, 5)` aan: de tweede *it*, een ding. Het dichtstbijzijnde eerdere ding is *the hall*, op plaats 3, dus dat geeft `3`: wat de regel kiest, al neemt een lezer het schilderij. Daarna eigen gevallen: plaats `0`, de eerste *it*, *visitors*, de laatste *it*, een persoon met alleen een ding ervoor, een persoon waarvan de dichtstbijzijnde eerdere persoon vier plaatsen terug staat, en een ding met een mention van dezelfde soort erna.
        """),
        L.task("Geef de plaats van de dichtstbijzijnde eerdere mention van dezelfde soort, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=8)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "Loop langs de plaatsen `j` vóór `i`. Onthoud bij elke mention met de soort `kind` de plaats `j` in `best`. De loop gaat van voor naar achter, dus de laatst onthouden plaats is de dichtstbijzijnde.",
            "`mentions[j][1]` is de soort van mention `j`: in de loop `if mentions[j][1] == kind:` met `best = j` eronder. De loop moet stoppen vóór plaats `i` zelf.",
            "`    for j in ___:`",
        ],
        why="Vervang de `...` door `for j in range(i):`, met daaronder `if mentions[j][1] == kind:` en `best = j` daar weer onder. Het voorbeeld geeft `3`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Op de 318 gevallen van LitBank heeft je functie 0,679 goed: het antwoord op de gok aan het begin. De dichtstbijzijnde eerdere mention van welke soort ook haalt 0,431. Voor een deel van wat de regel mist, is kennis van de betekenis van woorden nodig. Een voorbeeld van Jurafsky en Martin: *I ate the soup in my new bowl after cooking it for hours*. Het dichtstbijzijnde eerdere ding is de kom, en een lezer weet dat je soep eet, geen kom. Jurafsky en Martin noemen zulke kennis een selectional restriction: een werkwoord als *eat* neemt alleen bepaalde dingen als lijdend voorwerp, hier iets eetbaars. En 28 van de 318 zitten in een keten die LitBank als persoon markeert, zoals een baby of een kind waar de tekst *it* bij gebruikt; daarvan heeft de regel er 10 goed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_trophy():
    mo.md(r"""
    ### Oefening 4. Eén woord na *it*

    Terry Winograd liet in 1972 zien dat sommige voornaamwoorden kennis van de wereld of slim redeneren lijken te vragen. In 2011 stelde Hector Levesque een test voor die op zulke gevallen bouwt: de Winograd Schema Challenge. Jurafsky en Martin beschrijven de opgaven als een paar zinnen die in één woord of woordgroep verschillen, met een vraag naar coreferentie. De vraag hieronder gebruikt het voorbeeld uit het boek.
    """)
    return


@app.cell(hide_code=True)
def choice_trophy_cell():
    choice_trophy = L.choice(
        "trophy",
        question="*The trophy didn't fit into the suitcase because it was too large.* en *The trophy didn't fit into the suitcase because it was too small.* Wat kiest de regel van oefening 3 voor *it* in elke zin, en hoe doet de regel het?",
        options={
            "subject": "*The trophy* in allebei, want het onderwerp van een zin wint",
            "right": "*The suitcase* in allebei, dus één keer fout",
            "after": "*The trophy*, dan *the suitcase*: het woord na *it* beslist",
            "nothing": "Geen van beide: niets vóór *it* heeft dezelfde soort",
        },
        correct="right",
        why={
            "right": "Beide zinnen zetten *the trophy* en dan *the suitcase* vóór *it*, en alle drie zijn dingen. Dus kiest de regel twee keer *the suitcase*: goed bij *too small*, fout bij *too large*. Het woord dat beslist komt na *it*, en de doorslag geeft kennis die Jurafsky en Martin zo samenvatten: een groter voorwerp past niet in een kleiner. Zulke opgaven zijn gemaakt om makkelijk te zijn voor een menselijke lezer, maar hopelijk niet op te lossen met eenvoudige trucs, zoals selectional restrictions of woorden die vaak samen voorkomen. Jurafsky en Martin voegen eraan toe dat een groot deel van de oorspronkelijke opgaven toch op te lossen is met pretrained language models die verder getraind zijn op Winograd-zinnen.",
            "subject": "De regel kijkt naar de volgorde en de soort, niet naar de rol van een woord in de zin. Welk ding komt als laatste vóór *it*?",
            "after": "De regel kijkt alleen naar wat vóór *it* staat; de woorden na *it* tellen niet mee. Welk ding komt in elke zin als laatste vóór *it*?",
            "nothing": "Beide dingen in de zin staan vóór *it*, en *it* is ook een ding. Welke van de twee komt later?",
        },
    )
    choice_trophy
    return (choice_trophy,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de regel op de verhalen

    Schuif door 246 van de 318 gevallen van *it*, *its* en *itself* uit LitBank: de gevallen waarin het voornaamwoord, het antecedent en wat de regel kiest in hooguit drie korte zinnen passen, en de course elk van die zinnen als voorbeeld toont. Zet *Kijk naar de soort* uit, en de regel neemt de dichtstbijzijnde eerdere mention van welke soort ook. De slider begint in *Pride and Prejudice*: Mrs. Bennet vraagt *have you heard that Netherfield Park is let at last?*, en de *it* twee zinnen later is Netherfield Park.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_case = mo.ui.slider(1, len(C.DEMO_CASES), value=C.DEMO_START, label="Geval", full_width=True)
    demo_kind = mo.ui.checkbox(value=True, label="Kijk naar de soort")
    mo.vstack([demo_case, demo_kind])
    return demo_case, demo_kind


@app.cell(hide_code=True)
def demo_look(demo_case, demo_kind):
    mo.Html(C.demo_picture(demo_case.value, demo_kind.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een tekst is meer dan losse zinnen](/course/open/nlp-discourse?lang=nl) markeerde elke mention als nieuw of oud; deze les groepeerde de mentions van een tekst in ketens, vond het antecedent van elke anafoor, en probeerde een regel voor waar *it* naar verwijst. [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) vond de namen; coreferentie verbindt een naam met de beschrijvingen en voornaamwoorden van dezelfde entiteit. De regel van de dichtstbijzijnde eerdere mention van dezelfde soort heeft 0,679 van de 318 gevallen van *it* in LitBank goed, tegen 0,431 zonder de soort, met niets dan de volgorde van de tekst en twee soorten. Wat de regel mist, vraagt om kennis van de wereld, zoals bij *the trophy* en *the suitcase*. De volgende les, *Encoders, decoders en de GPT-lijn*, gaat over de neurale modellen waar latere systemen voor coreferentie op bouwen.

    **Een voorbeeld uit het dagelijks leven.** Een briefje zegt *Vraag Kim naar het rapport. Het moet vrijdag af zijn.* Je verbindt *Het* zonder nadenken met het rapport en niet met Kim: de dichtstbijzijnde eerdere mention van de goede soort.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_trophy, result_antecedents, result_chains, result_pick):
    L.wrap_up(result_chains, result_antecedents, result_pick, choice_trophy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
