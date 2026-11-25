# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-classic-mt. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-classic-mt", "nl")
    C = L.checks   # de Nederlandse zinnen van Tatoeba met de Engelse vertalingen, het woordenboek, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl), en één over [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_root():
    quick_root = L.choice(
        "quick-root",
        question="In *Wie hangt van wie af: dependency trees* hangt elk woord van een zin aan één hoofdwoord, behalve de root. Welk woord is de root in *de late trein verliet de stad na middernacht*?",
        options={
            "subject": "*trein*, het onderwerp, want het onderwerp staat vooraan en de rest van de zin hangt eraan",
            "right": "*verliet*, het werkwoord: onderwerp en lijdend voorwerp hangen eraan",
            "object": "*stad*, het lijdend voorwerp",
            "first": "*de*, het eerste woord",
        },
        correct="right",
        why={
            "right": "De root is vaak het hoofdwerkwoord: *trein* hangt aan *verliet* met `nsubj`, *stad* met `obj` en *middernacht* met `obl`. Deze les gebruikt zo'n relatie: in *ik zal de volgende bus nemen* hangt het hulpwerkwoord *zal* aan het hoofdwerkwoord *nemen*.",
            "subject": "*trein* hangt aan een ander woord, met `nsubj`. Aan welk woord hangen het onderwerp en het lijdend voorwerp allebei? Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
            "object": "*stad* hangt aan een ander woord, met `obj`. Aan welk woord hangen het onderwerp en het lijdend voorwerp allebei? Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
            "first": "*de* is een bepalend woord: het hangt aan *trein*, met `det`. Welk woord heeft geen hoofdwoord? Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
        },
    )
    quick_root
    return (quick_root,)


@app.cell(hide_code=True)
def quick_check_in():
    quick_in = L.choice(
        "quick-in",
        question="In *Dictionaries* kijkt `in` of een key in een dictionary staat. Wat geeft `\"cat\" in {\"kat\": \"cat\"}`?",
        options={
            "true": "`True`, want *cat* staat in de dictionary",
            "right": "`False`",
            "error": "Een fout: `in` werkt op een lijst of een tekst, en niet op een dictionary",
            "key": "`\"kat\"`, de key die bij *cat* hoort",
        },
        correct="right",
        why={
            "right": "`in` kijkt naar de keys, en *cat* is een waarde: `\"kat\" in {\"kat\": \"cat\"}` geeft `True`. In deze les beantwoordt een dictionary van Nederlandse naar Engelse woorden `in` voor een Nederlands woord.",
            "true": "*cat* staat in de dictionary, maar als waarde. Waar kijkt `in` naar, de keys of de waarden? Zie [Dictionaries](/course/open/py-dicts?lang=nl).",
            "error": "`in` werkt ook op een dictionary, en geeft `True` of `False`. Waar kijkt `in` naar? Zie [Dictionaries](/course/open/py-dicts?lang=nl).",
            "key": "`in` geeft `True` of `False`, geen key. Is *cat* een key van deze dictionary? Zie [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_in
    return (quick_in,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 1947 schreef Warren Weaver van de Rockefeller Foundation aan de wiskundige Norbert Wiener dat de nieuwe computers misschien konden vertalen. Weaver wist niets officieels over de nieuwe machinale methoden om codes te kraken, maar had er veel over geraden, en vroeg zich af of vertalen het kraken van een code was: een Russische tekst zag Weaver als Engels, gecodeerd in vreemde tekens, dat je alleen nog hoefde te ontcijferen. In 1949 citeerde Weaver uit de brief in een memorandum, *Translation*, in de hoop dat het iemand anders een zetje zou geven.

    In januari 1954 lieten IBM en de Georgetown University in New York een computer zien die Russische zinnen in het Engels vertaalde. Hutchins, een historicus van de machinevertaling, schreef erover dat het een klein experiment was, met slechts 250 woorden en zes 'grammaticaregels', maar dat het de verwachting wekte dat systemen binnenkort automatisch en goed zouden vertalen. De regels konden kiezen tussen twee Engelse woorden voor een Russisch woord, en twee woorden naast elkaar omwisselen.

    In deze les vertaal je Nederlands in het Engels met een woordenboek van 478 woorden en een eigen regel, en meet je op de zinnen van Tatoeba hoe ver je daarmee komt.

    **Een voorbeeld uit het dagelijks leven.** Een toerist met een zakwoordenboek zoekt elk woord van een bord op en leest de vertalingen achter elkaar. *Verboden toegang* wordt *forbidden access*: je begrijpt het, maar in het Engels staat er *No entry*.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_exact = L.guess(
        "guess",
        question="Het woordenboek van de course heeft 478 gewone Nederlandse woorden, elk met één Engels woord. De les meet op 21.636 Nederlandse zinnen van Tatoeba waarvan alle woorden erin staan. Vertaal elk van deze zinnen woord voor woord, en vergelijk de uitkomst met de Engelse vertalingen die mensen van Tatoeba schreven, zonder hoofdletters en leestekens, en met *I'll* uitgeschreven als *i will*. Bij hoeveel zinnen geeft woord voor woord precies een van deze vertalingen?",
        options={
            "all": "Bijna allemaal",
            "half": "Ongeveer de helft",
            "right": "Ongeveer één op de zeven",
            "hundred": "Bijna geen, ongeveer één op de honderd",
        },
        reveal="Ongeveer één op de zeven: 2.979 van de 21.636, of 13,8%. Woord voor woord vertaalt korte zinnen goed waarvan de woorden één op één overeenkomen, zoals *Ik ben moe*, *i am tired*. Nog 1.037 (4,8%) hebben de goede woorden in de verkeerde volgorde: *Ik zal de volgende bus nemen* wordt *i will the next bus take*. De andere 17.620 (81,4%) hebben andere woorden: *Ik heb honger* wordt *i have hunger*, waar mensen *I'm hungry* schrijven.",
    )
    guess_exact
    return (guess_exact,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Machinevertaling** zet een tekst in de ene taal, de **brontaal**, om in een tekst in een andere taal, de **doeltaal**. Tientallen jaren schreven mensen de regels daarvoor met de hand. Bernard Vauquois en Christian Boitet, die zulke systemen bouwden aan de universiteit van Grenoble, verdeelden het vertalen in drie fasen: analyse, transfer en generatie. De analyse leest de brontekst, de transfer zet wat de analyse vond om naar de doeltaal, en de generatie schrijft de doeltekst.

    Hoe ver de analyse gaat, bepaalt het soort systeem. De afbeelding, bekend als de **driehoek van Vauquois**, toont drie manieren:

    - **Directe vertaling** blijft onderaan: het systeem zoekt elk woord op in een woordenboek en zet de vertalingen achter elkaar, met misschien een paar regels om woorden te verplaatsen. Het systeem van Georgetown en IBM was van dit soort.
    - **Transfer** ontleedt de bronzin eerst in een structuur, zoals een dependency tree, zet deze structuur om in een boom van de doeltaal, en maakt daaruit de doelzin.
    - Een **interlingua** gaat helemaal naar boven: de analyse schrijft de betekenis op in een vorm die bij geen enkele taal hoort, en de generatie schrijft deze betekenis in de doeltaal.

    Hoe hoger een systeem klimt, hoe meer werk de analyse en de generatie doen, en hoe minder er overblijft voor de transfer. Woord voor woord gaat mis waar twee talen verschillen: in de volgorde van de woorden, in woorden die de ene taal nodig heeft en de andere niet, en in **idiomen**, vaste uitdrukkingen met een betekenis die je niet uit de losse woorden haalt. Met een groter woordenboek wordt *Het regent pijpenstelen* woord voor woord *it rains pipe stems*; in het Engels zeg je *It's raining cats and dogs*.
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

    | Woord | Wat het betekent |
    |---|---|
    | **machinevertaling** (machine translation, MT) | een programma dat een tekst van de ene taal in een andere vertaalt |
    | **brontaal**, **doeltaal** | de taal waaruit een tekst vertaald wordt, en de taal waarin |
    | **directe vertaling** | woord voor woord vertalen met een woordenboek, met hooguit een paar regels om woorden te verplaatsen |
    | **transfer** | vertalen via een structuur: ontleed de bronzin in een boom, zet deze boom om in een boom van de doeltaal, en maak daaruit de doelzin |
    | **interlingua** | een manier om de betekenis van een tekst op te schrijven die bij geen enkele taal hoort; een systeem met een interlingua vertaalt via deze vorm |
    | **driehoek van Vauquois** (Vauquois triangle) | de afbeelding van de drie manieren: hoe hoger de analyse gaat, hoe minder er overblijft voor de transfer |
    | **idioom** | een vaste uitdrukking met een betekenis die je niet uit de losse woorden haalt, zoals *het regent pijpenstelen* |
    | **hulpwerkwoord** | een werkwoord bij een hoofdwerkwoord, zoals *zal* of *heb*; `AUX` in *Woordsoorten en woordgroepen*, en in UD hangt het aan het hoofdwerkwoord met de relatie `aux` |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vertaal je een Nederlandse zin met regels:

    1. Knip de zin in woorden, in kleine letters.
    2. Zoek het eerste hulpwerkwoord, en kijk of het laatste woord een werkwoordsvorm is: een vorm die aan het eind kan staan, zoals *nemen* of *gelezen*.
    3. Als beide er zijn, zet dat werkwoord direct na het hulpwerkwoord, waar het Engels het wil hebben.
    4. Zoek elk woord op in het woordenboek. Een woord dat er niet in staat, zoals een naam, blijft zoals het is.
    5. Vergelijk de uitkomst met de vertalingen door mensen: precies een ervan, dezelfde woorden in een andere volgorde, of andere woorden.

    Stap 2 en 3 zijn een kleine versie van analyse en transfer. Een systeem met transfer vindt het hulpwerkwoord en het hoofdwerkwoord in een boom, via de relatie `aux`, en verplaatst het werkwoord daar. Deze les verplaatst het in de lijst met woorden, met een set hulpwerkwoorden in plaats van de boom: directe vertaling met één regel om woorden te verplaatsen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem *Ik zal de volgende bus nemen.* uit Tatoeba, daar vertaald als *I'll take the next bus.* Alle zes woorden staan in het woordenboek, dat de les heeft als `C.DICTIONARY`; `C.words` knipt een Nederlandse tekst in woorden, in kleine letters. `C.pair` geeft een zin van Tatoeba met de vertalingen, op het nummer van de zin daar (`C.WORKED` is het nummer van deze), en `C.english_words` geeft de woorden van een Engelse tekst, met samentrekkingen als *I'll* uitgeschreven:
    """)
    return


@app.cell
def example_cell():
    _dutch = C.words("Ik zal de volgende bus nemen.")
    print(_dutch, len(C.DICTIONARY))
    print(C.DICTIONARY["zal"], C.DICTIONARY["nemen"])
    _moved = _dutch[:2] + [_dutch[-1]] + _dutch[2:-1]
    print(_moved)
    print(C.pair(C.WORKED))
    print(C.english_words("I'll take the next bus."))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Woord voor woord geven de zes woorden *i will the next bus take*: de woorden van de vertaling door mensen, in een andere volgorde. In de boom is *nemen* de root, en *zal* hangt eraan met `aux`; het Nederlands zet het hoofdwerkwoord aan het eind, het Engels direct na het hulpwerkwoord. Dryer schrijft in de *World Atlas of Language Structures* dat de dominante volgorde in het Duits en het Nederlands SVO is in een hoofdzin zonder hulpwerkwoord, en SOV in een bijzin en in een zin met een hulpwerkwoord. SVO is onderwerp, werkwoord, lijdend voorwerp; SOV zet het werkwoord na het lijdend voorwerp. De slices zetten *nemen* direct na *zal*: *ik zal nemen de volgende bus* is slecht Nederlands, maar woord voor woord geeft het *i will take the next bus*, precies de vertaling door mensen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. word_for_word: vertalen met een woordenboek

        Schrijf `word_for_word`, dat `words` krijgt, een lijst met Nederlandse woorden, en `dictionary`, een dictionary van Nederlandse naar Engelse woorden, zoals `C.DICTIONARY`. Geef een nieuwe lijst terug met het Engelse woord van elk woord, in dezelfde volgorde. Een woord dat niet in `dictionary` staat, blijft zoals het is. In de startcode begint `english` als een lege lijst, en de loop neemt steeds één woord `w`. Vervang de twee `...`.

        De check roept `word_for_word(["ik", "zie", "een", "kat"], {"ik": "i", "zie": "see", "een": "a"})` aan, en dat geeft `["i", "see", "a", "kat"]`: *kat* staat niet in de dictionary, dus het blijft staan. Daarna `word_for_word(["de", "hond", "slaapt"], {"de": "the", "hond": "dog"})`, en eigen gevallen: een lege lijst, een woord dat twee keer staat, en zinnen van Tatoeba met `C.DICTIONARY`.
        """),
        L.task("Vertaal een lijst met Nederlandse woorden woord voor woord met een woordenboek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("word_for_word", lines=8)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("word_for_word", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "word_for_word",
        [
            "Zoek voor een woord `w` dat een key van `dictionary` is de waarde op met vierkante haken, zoals in *Dictionaries*, en voeg de waarde achteraan de lijst toe met `english.append`. Voor elk ander woord voegt `english.append` het woord `w` zelf toe.",
            "Zonder de `if` stopt het opzoeken met een `KeyError` bij het eerste woord dat niet in `dictionary` staat. En houd dat woord: een naam als *Tom*, of een woord dat het woordenboek niet heeft, moet in de lijst blijven, anders valt er een woord uit de vertaling weg.",
            "`            english.append(___)`",
        ],
        why="In de `if`: `english.append(dictionary[w])`; in de `else`: `english.append(w)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_words():
    mo.md(r"""
    Het woordenboek dekt korte zinnen met gewone woorden. De kopie van Tatoeba in de course heeft 147.001 Nederlandse zinnen met een Engelse vertaling, samen 169.641 vertalingen. Van deze zinnen hebben er 21.923 elk woord in het woordenboek; de les meet op 21.636 daarvan, en laat er 287 weg omdat de vertalingen een voornaamwoord met een geslacht gebruiken, en zulke voornaamwoorden toont de course niet. 3.707 van de 21.636 hebben meer dan één vertaling: mensen vertalen één zin op verschillende manieren, zoals *I'm tired* en *I am tired*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_outcome():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. outcome: hoe dicht zit een vertaling erbij?

        Schrijf `outcome`, dat `translation` krijgt, een lijst met Engelse woorden, en `references`, een lijst met de vertalingen door mensen, elk een lijst met woorden. Geef `"same"` terug als `translation` een ervan is; anders `"order"` als een ervan dezelfde woorden heeft, elk even vaak, in een andere volgorde; en anders `"words"`. De startcode stelt de twee vragen in deze volgorde. Vervang de twee `...`.

        De check roept `outcome(["the", "dog", "sleeps"], [["the", "dog", "is", "sleeping"], ["the", "dog", "sleeps"]])` aan, en dat geeft `"same"`: het is de tweede vertaling. Daarna `outcome(["i", "will", "the", "bus", "take"], [["i", "will", "take", "the", "bus"]])`, dat `"order"` geeft, en eigen gevallen, met de vertalingen woord voor woord van de course (in de les: `C.direct()`, één voor elk van de 21.636 zinnen) en de vertalingen door mensen (`C.references()`, in dezelfde volgorde).
        """),
        L.task("Zeg of een vertaling precies een van de vertalingen door mensen is, de woorden ervan in een andere volgorde heeft, of andere woorden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_outcome():
    box_outcome = L.code_box("outcome", lines=7)
    box_outcome
    return (box_outcome,)


@app.cell(hide_code=True)
def check_outcome(box_outcome):
    result_outcome = L.check("outcome", box_outcome.value)
    result_outcome
    return (result_outcome,)


@app.cell(hide_code=True)
def hints_outcome():
    L.hints(
        "outcome",
        [
            "`in` vraagt of een lijst een van de items van een lijst met lijsten is, zoals bij elk ander item. Voor de tweede vraag geeft `sorted` een nieuwe lijst met de woorden van `r` op volgorde, zoals in *Tuples, sets en sorteren*; vergelijk deze lijst met `==` met de woorden van `translation`, op dezelfde manier gesorteerd.",
            "Sorteer, en maak geen sets: een set houdt een woord één keer, dus *the dog sees the cat* zou gelijk zijn aan *the cat sees dog*. En sorteer met `sorted`, niet met `sort`: `sort` verandert een lijst en geeft `None` terug.",
            "`        if sorted(r) == ___:`",
        ],
        why="`if translation in references:` en `if sorted(r) == sorted(translation):`.",
    )
    return


@app.cell(hide_code=True)
def text_after_outcome():
    mo.md(r"""
    Woord voor woord vertaalt 2.979 van de 21.636 zinnen precies goed. Bij 1.037 kloppen de woorden en niet de volgorde: daar kan een regel helpen om woorden te verplaatsen. De 17.620 met andere woorden hebben meer nodig dan een regel voor de volgorde. Soms ontbreekt een woord, want het Engels heeft in een vraag *do* nodig waar het Nederlands geen woord heeft: *Waar woon je?* wordt *where live you*, en mensen schrijven *Where do you live?* Soms betekent een woord iets anders in een vaste uitdrukking: *Ik hou van je* wordt *i love of you*, want het woordenboek heeft één Engels woord voor *van*, *of*. En soms zeg je het anders: *Ik heb honger* is *I'm hungry*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. verb_after_helper: een regel om een werkwoord te verplaatsen

        Schrijf `verb_after_helper`, dat de Nederlandse woorden `words` van een zin krijgt, een set `helpers` met hulpwerkwoorden en een set `verbs` met werkwoordsvormen die aan het eind kunnen staan, zoals *nemen* en *gelezen*. Als het laatste woord in `verbs` staat en er eerder een woord uit `helpers` staat, zet dan het laatste woord direct na het eerste hulpwerkwoord, en geef de nieuwe lijst terug. Geef anders `words` onveranderd terug. In de startcode is `last` het laatste woord, gaat de loop langs de plaatsen `i` van de andere woorden, en vindt de `if` erin het eerste hulpwerkwoord. Vervang de twee `...`.

        De check roept `verb_after_helper(["ik", "heb", "het", "boek", "gelezen"], {"heb", "kan"}, {"gelezen", "zwemmen"})` aan, en dat geeft `["ik", "heb", "gelezen", "het", "boek"]`, de volgorde van *I have read the book*. Daarna `verb_after_helper(["ik", "kan", "snel", "zwemmen"], {"heb", "kan"}, {"gelezen", "zwemmen"})`, en eigen gevallen: een zin zonder hulpwerkwoord, een zin waarvan het laatste woord geen werkwoord is, een vraag, en zinnen van Tatoeba met de sets van de course, `C.HELPERS` (33 hulpwerkwoorden) en `C.VERBS` (66 werkwoordsvormen).
        """),
        L.task("Zet een werkwoord aan het eind van een Nederlandse zin direct na het hulpwerkwoord, zoals het Engels het wil."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("verb_after_helper", lines=8)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("verb_after_helper", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "verb_after_helper",
        [
            "De `if` geeft `words` terug als `last` niet in `verbs` staat. In de loop bestaat de nieuwe lijst uit drie stukken, aan elkaar gezet met `+`: de woorden tot en met het hulpwerkwoord, `words[:i + 1]`; dan `[last]`, een lijst met alleen het laatste woord; en dan de woorden tussen het hulpwerkwoord en het laatste woord, een slice van `words`.",
            "Laat het laatste woord weg uit het derde stuk, anders staat het twee keer in de lijst; en begin dat stuk na het hulpwerkwoord, anders staat het hulpwerkwoord twee keer in de lijst. Een negatief getal in een slice telt vanaf het eind, zoals in *Lijsten*.",
            "`            return words[:i + 1] + [last] + ___`",
        ],
        why="`if last not in verbs:` en `return words[:i + 1] + [last] + words[i + 1:-1]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rule():
    mo.md(r"""
    De regel verandert de volgorde van 5.361 van de 21.636 zinnen. De regel zet er 160 goed die woord voor woord fout gingen, zoals *Ik kan snel zwemmen*, nu *i can swim fast*. Maar de regel maakt er 133 fout die woord voor woord goed gingen. 72 daarvan zijn vragen: het hulpwerkwoord staat vooraan, het Engels zet het werkwoord direct na het onderwerp, en *Kan ik helpen?* gaat van *can i help* naar *can help i*. Bij 38 staat *niet* eerder dan het werkwoord: *Ik kan niet zwemmen* gaat van *i can not swim* naar *i can swim not*. En de regel ziet *leven* als werkwoord, ook waar het een zelfstandig naamwoord is, *life*: *Ik heb een geweldig leven* wordt *i have life a great*. Per saldo wint de regel er 27: 3.006 zinnen komen er precies uit, 13,9%. Een regel voor vragen zou een vraag moeten herkennen, een regel voor *niet* zou moeten weten waar *not* hoort, en de uitzonderingen daarop zouden weer eigen regels nodig hebben. Vauquois en Boitet maten de omvang van zulke systemen in programmaregels, niet in grammaticaregels, want een grammaticaregel kan heel eenvoudig zijn of heel ingewikkeld. Het kleine systeem van Vauquois en Boitet voor het onderwijs, BEX-FEX, met woordenboeken die met opzet heel klein zijn (500 lexicale eenheden per taal), had 4.320 programmaregels grammatica: 420 voor de vormen van woorden, 2.500 voor de structuur van zinnen, 300 voor de transfer en 1.100 voor de generatie.

    ### Probeer het: vertaal een zin

    Typ een Nederlandse zin en zie de zin woord voor woord, en na de regel van oefening 3. Een woord dat niet in het woordenboek staat, blijft zoals het is, met een gestippelde rand. Bij een van de 21.636 zinnen waarop de les meet, toont het vak ook de vertalingen door mensen. Probeer *Ik kan snel zwemmen.*, *Kan ik helpen?* of *Wat wil je doen?*
    """)
    return


@app.cell(hide_code=True)
def box_sentence():
    try_sentence = mo.ui.text(value="Ik zal de volgende bus nemen.", label="Een Nederlandse zin", full_width=True)
    try_sentence
    return (try_sentence,)


@app.cell(hide_code=True)
def box_cell(try_sentence):
    mo.Html(C.box_picture(L.lang, try_sentence.value))
    return


@app.cell(hide_code=True)
def text_exercise_transfers():
    mo.md(r"""
    ### Oefening 4. Waarom een interlingua?

    Een bedrijf wil een systeem dat tussen 10 talen vertaalt, in elke richting. Met transfer heeft elke taal een analyse en een generatie nodig, en elke richting een eigen transfer.
    """)
    return


@app.cell(hide_code=True)
def choice_transfers_cell():
    transfers = L.choice(
        "transfers",
        question="Hoeveel transfers heeft dat systeem nodig, elk van de ene taal naar een andere?",
        options={
            "ten": "10, één voor elke taal",
            "hundred": "100, tien voor elke taal",
            "right": "90, één voor elke richting",
            "pairs": "45, één voor elk paar talen, in beide richtingen gebruikt",
        },
        correct="right",
        why={
            "right": "Elk van de 10 talen kan naar 9 andere: 10 × 9 = 90 richtingen, en een transfer van Nederlands naar Engels vertaalt geen Engels naar Nederlands. Met een interlingua heeft het systeem alleen de analyse en de generatie van elke taal nodig: 20 delen. Een elfde taal voegt 2 delen toe, waar transfer er 20 transfers bij krijgt. Dat is de belofte van de top van de driehoek; de prijs is een manier om betekenis op te schrijven die bij elke taal past.",
            "ten": "10 is het aantal analyses. Hoeveel paren van een brontaal en een doeltaal zijn er?",
            "hundred": "Dan telt ook een transfer van elke taal naar dezelfde taal mee. Is dat een richting die vertaald moet worden?",
            "pairs": "Een transfer werkt één kant op: van een boom van de brontaal naar een boom van de doeltaal. Vertaalt een transfer van Nederlands naar Engels ook Engels naar Nederlands?",
        },
    )
    transfers
    return (transfers,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl) las je een zin als boom; hier liet één relatie daarin, `aux`, zien welk werkwoord moest verschuiven, en een regel verplaatste het in de lijst met woorden. [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) zette regels schrijven tegenover leren van voorbeelden: een woordenboek van 478 woorden en één regel vertalen 3.006 van de 21.636 zinnen precies goed, en elke nieuwe regel zet sommige zinnen goed en maakt andere fout. Systemen die leren van vertaalde zinnen namen het over van de regels, zoals latere lessen laten zien. De volgende les, *Een vertaling scoren: BLEU en chrF*, geeft ook punten aan een vertaling die deels klopt, in plaats van alleen goed of fout.

    **Een voorbeeld uit het dagelijks leven.** Een taalgids voor op reis bevat hele zinnen, zoals *Waar is het station?*, en niet alleen woorden: een manier om de problemen van woord voor woord te omzeilen, die mensen al kenden lang voordat er computers waren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_words, result_outcome, result_rule, transfers):
    L.wrap_up(result_words, result_outcome, result_rule, transfers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
