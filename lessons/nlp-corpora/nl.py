# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-corpora. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-corpora", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="Welk woord is in *de late trein verliet de stad* het hoofdwoord van *late*?",
        options={
            "root": "*verliet*, de root van de hele zin",
            "right": "*trein*, waar het iets over zegt",
            "next": "*de*, het woord er vlak voor",
        },
        correct="right",
        why={
            "right": "*Late* zegt iets over *trein*, dus *late* hangt aan *trein*, met de relatie amod. Deze les laat zien waar zulke bomen vandaan komen: mensen labelen zinnen volgens geschreven richtlijnen, en de gelabelde zinnen vormen een treebank.",
            "root": "*Verliet* is de root: de hele zin hangt eraan, maar *late* hangt er niet rechtstreeks aan. Over welk woord zegt *late* iets? Zie *Hoofdwoorden en afhankelijke woorden* in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl).",
            "next": "Naast een woord staan maakt het nog geen hoofdwoord; *de* hangt zelf ook aan *trein*. Over welk woord zegt *late* iets? Zie *Hoofdwoorden en afhankelijke woorden* in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl).",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_root():
    quick_root = L.choice(
        "quick-root",
        question="In een zin als lijst van `(form, head, relation)` heeft een woord hoofdwoord `0`. Wat betekent dat?",
        options={
            "first": "Het hoofdwoord is het eerste woord van de zin",
            "out": "Het woord valt buiten de boom",
            "right": "Geen ander woord is het hoofdwoord: de root",
        },
        correct="right",
        why={
            "right": "Hoofdwoord 0 wijst de root aan, het woord waar elk ander woord aan hangt, rechtstreeks of via andere woorden. De woorden tellen vanaf 1, dus 0 is geen woord van de zin. In deze les zoek je hoofdwoorden weer op met hun nummer.",
            "first": "Het eerste woord heeft nummer 1: de woorden tellen vanaf 1. Hoofdwoord 0 is geen woord van de zin. Zie *De bestanden: CoNLL-U* in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl).",
            "out": "Elk woord van een zin staat in de boom, ook de root. Zie *Een boom van woorden* in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl).",
        },
    )
    quick_root
    return (quick_root,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elk model dat in deze course met taal werkt, leert van tekst die iemand verzamelde, en wordt getest op tekst die iemand labelde. Jurafsky en Martin wijzen er in hun hoofdstuk over woorden en tokens op dat woorden niet uit het niets komen: elk stuk tekst is gemaakt door bepaalde sprekers of schrijvers, in een bepaald dialect van een bepaalde taal, op een bepaald moment, op een bepaalde plek en met een bepaald doel. Wat een model van een taal weet, hangt af van de teksten die erin gingen. De bomen van [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl) komen uit treebanks, zinnen die mensen een voor een labelden, en large language models leren van verzamelingen van biljoenen tokens. Ook taalkundigen kijken naar wat mensen echt schrijven en zeggen, in plaats van naar wat ze denken dat mensen schrijven en zeggen. Daarvoor verzamelen ze corpora, en ze leggen vast wat erin gaat en hoe het gelabeld wordt.

    **Een voorbeeld uit het dagelijks leven.** Je twijfelt tussen *groter dan* en *groter als*, dus je zoekt allebei op en vergelijkt hoe vaak elk voorkomt. Je gebruikte zojuist het web als verzameling teksten.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Het Brown Corpus bevat ongeveer een miljoen woorden Amerikaans Engels, gedrukt in 1961. Het woord dat er het vaakst in staat, is *the*. Hoe vaak komt het voor?",
        options={
            "often": "Ongeveer 1 op de 4 woorden",
            "right": "Ongeveer 1 op de 14 woorden",
            "rare": "Ongeveer 1 op de 100 woorden",
        },
        reveal="**Ongeveer 1 op de 14 woorden**: in de kopie van het Brown Corpus die bij de Python-toolkit NLTK hoort, telt de course 69.971 keer *the* op 1.013.319 woorden, dus ongeveer 69.000 per miljoen woorden. *Of* staat op de tweede plaats, met 36.412. John Sinclair (2004) komt op 69.970. In het British National Corpus neemt *the* 61.847 plaatsen per miljoen woorden in (Geoffrey Leech, Paul Rayson en Andrew Wilson, 2001): 64.420 in geschreven en 39.605 in gesproken taal. Hoeveel woorden een corpus heeft, hangt af van wat je als woord telt, en in de boeken van de course verschilt het aandeel per boek, zoals de demo van deze les laat zien.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een verzameling met een doel

    Een **corpus** (meervoud *corpora*) is een verzameling teksten, of opgenomen spraak, bijeengebracht om een taal te bestuderen. De taalkundige John Sinclair omschreef het in 2004 als een verzameling stukken tekst in elektronische vorm, gekozen volgens criteria van buiten de tekst, om zo goed als het gaat een taal of een variant van een taal te vertegenwoordigen, als bron van gegevens voor taalkundig onderzoek. Criteria van buiten de tekst wil zeggen: gekozen op waar een tekst vandaan komt en waar die voor dient, niet op de woorden erin. Volgens Sinclair kies je de inhoud zonder op de taal ervan te letten, alleen op de rol die een tekst heeft in de gemeenschap waar die tekst ontstond.

    Het Brown Corpus, samengesteld door W. Nelson Francis en Henry Kučera aan Brown University in de jaren zestig, was volgens de Corpus Resource Database van de Universiteit van Helsinki het eerste algemene corpus op de computer voor taalkundig onderzoek naar modern Engels: 500 stukken van ongeveer 2.000 woorden, uit Amerikaans Engels dat in 1961 gedrukt werd, van nieuws tot sciencefiction. Het British National Corpus (BNC), gebouwd van 1991 tot en met 1994, bevat 100 miljoen woorden Brits Engels, voor 90% geschreven en voor 10% gesproken. Voor het Nederlands bevat het Corpus Gesproken Nederlands (CGN) bijna 9 miljoen woorden opgenomen spraak uit Nederland en Vlaanderen, verzameld van 1998 tot en met 2004, en SoNaR-500 meer dan 500 miljoen woorden geschreven Nederlands.

    Groter is niet hetzelfde als beter. Het web is volgens Sinclair geen corpus: niemand weet hoe groot het is, het verandert steeds, en het is niet met het oog op taalonderzoek opgezet. De makers van de Penn Treebank schreven in 1993 dat hun ruwe teksten een verzameling zijn, en geen corpus, als je streng scheidt tussen een corpus en een verzameling die op goed geluk is bijeengebracht. De tekst waar large language models van leren, is meestal zo'n verzameling: FineWeb, een dataset uit 2024, bevat 15 biljoen tokens uit 96 momentopnamen van het web door Common Crawl. Jurafsky en Martin raden aan om bij elk corpus een datasheet te schrijven, die onder meer zegt waarom het corpus verzameld is, door wie, en wie het betaalde.
    """)
    return


@app.cell(hide_code=True)
def text_idea_labels():
    mo.md(r"""
    ### Labels erbovenop: treebanks en lexicale databases

    Veel corpora hebben labels die mensen toevoegden. Een **treebank** is een corpus waarin elke zin een boom heeft. Jurafsky en Martin noemen een corpus waarin elke zin een parse tree heeft gekregen, een treebank. Meestal maakt een programma een eerste boom en verbeteren mensen die: een parser gaat over elke zin, en taalkundigen corrigeren de uitkomst met de hand. De Penn Treebank van Mitchell Marcus, Beatrice Santorini en Mary Ann Marcinkiewicz (1993) bevat ruim 4,5 miljoen woorden Amerikaans Engels, getagd met 36 tags voor woordsoorten en 12 voor leestekens en valutatekens; meer dan de helft kreeg ook een geraamte van de zinsbouw. Universal Dependencies, waarvan je de bomen las in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl), had bij de uitgave van 15 november 2026 402 treebanks in 226 talen; twee daarvan zitten in deze les.

    Een **lexicale database** bevat feiten over woorden in plaats van doorlopende tekst. De bekendste is WordNet, van Princeton University; volgens Jurafsky en Martin de bron die het meest gebruikt wordt voor relaties tussen betekenissen, in het Engels en in veel andere talen. WordNet zet woorden met dezelfde betekenis bij elkaar in een **synset**, kort voor *synonym set*, en een woord met meer betekenissen staat in meer synsets. Versie 3.0 heeft 117.659 synsets. Open Dutch WordNet (2016) koppelt Nederlandse woorden eraan. De afbeelding laat één zin van de course drie keer zien: als tekst, met de labels van een treebank, en met een van de woorden in een lexicale database.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_concordance():
    mo.md(r"""
    ### Elke plek van een woord: de concordantie

    Vóór de computer was een **concordantie** een boek. De Catholic Encyclopedia van 1908 beschrijft concordanties van de Bijbel als alfabetische lijsten van de woorden van de Bijbel, met aanwijzingen om de passages te vinden waar elk woord staat. Volgens die encyclopedie werd de eerste voltooid in 1230, onder leiding van Hugo van Saint-Cher, die naar men zegt door 500 medebroeders van de dominicanen geholpen werd. Een computer maakt er een in een oogwenk. Sinclair noemt een concordantie een reeks korte stukjes uit een corpus die een programma uitkiest. De gewone vorm is **KWIC**, kort voor *key word in context*, trefwoord in context: een naam die Hans Peter Luhn rond 1960 gaf. Elke plek waar het woord staat, krijgt een regel, met het woord in het midden en de buren links en rechts. De horizontale regel lees je als tekst, legt Sinclair uit, en van boven naar beneden zie je waarin een regel lijkt op de regels eromheen, en waarin niet. Wat een lezer in één keer overziet, schat Sinclair op een scherm vol: zo'n 23 regels.
    """)
    return


@app.cell(hide_code=True)
def text_idea_guidelines():
    mo.md(r"""
    ### Opschrijven hoe je labelt

    Labels helpen alleen als iedereen op dezelfde manier labelt, dus de makers van een corpus schrijven **richtlijnen**: regels met voorbeelden van de lastige gevallen. Beatrice Santorini schrijft in de richtlijnen voor de tags van de Penn Treebank (1990) dat het vaak heel moeilijk is om te beslissen welke tag in een bepaalde zin past. Een lastig geval is een woord als *off* na een werkwoord: een voorzetsel, tag IN, of een **partikel** dat bij het werkwoord hoort, tag RP? Die twee zijn volgens de richtlijnen vaak moeilijk uit elkaar te houden, en er staan tests in. Een woord is een partikel als het vóór én na het lijdend voorwerp kan staan: Engels *told off the team* en *told the team off* kunnen allebei. Een voorzetsel moet vóór het zelfstandig naamwoord staan: *stepped off the train*, maar niet *stepped the train off*. (De zinnen zijn van de course, naar de voorbeelden van Santorini.) Ook het Nederlands heeft partikels: het losse deel van een scheidbaar werkwoord, zoals *op* in *we haalden de kinderen op*. Oefening 3 geeft daar een test voor.

    Ook met richtlijnen zijn mensen het niet altijd eens. In het onderzoek van Marcus en collega's tagden vier mensen met een opleiding in de taalkunde stukken uit het Brown Corpus. Gemiddeld verschilden ze bij 7,2% van de woorden als ze zelf tagden, en bij 4,1% als ze de tags van een programma verbeterden. Eén tekst zorgde voor een groot deel daarvan: er stonden veel tekens voor scheikundige formules in, en omdat er geen richtlijn voor dat geval was, kozen de mensen verschillend. Jurafsky en Martin schatten wat mensen bij het taggen van Engels halen op ongeveer 97%. Hoe je overeenstemming meet, en hoe Cohen's kappa rekening houdt met overeenstemming door toeval, staat in de les *Goed labelen: richtlijnen en overeenstemming* van de course.

    Twee richtlijnen kunnen ook verschillen. De Penn Treebank geeft een partikel de tag RP; Universal Dependencies geeft het de woordsoort ADP, net als een voorzetsel, en de relatie `compound:prt` naar het werkwoord. Allebei zijn de richtlijnen consequent; alleen de regels verschillen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **corpus** | een verzameling teksten of opgenomen spraak, gekozen om een taal of een soort taal te vertegenwoordigen; meervoud *corpora* |
        | **treebank** | een corpus waarin elke zin een boom heeft, door mensen gecontroleerd |
        | **lexicale database** | een verzameling feiten over woorden, zoals welke woorden dezelfde betekenis hebben: WordNet is er een |
        | **synset** | in WordNet: een verzameling woorden met één gedeelde betekenis |
        | **concordantie** | elke plek waar een woord in een corpus staat, telkens met de woorden eromheen |
        | **KWIC** | *key word in context*, trefwoord in context: een concordantie met een regel per plek en het woord in het midden |
        | **richtlijn** | een geschreven regel voor de mensen die een corpus labelen, met voorbeelden van de lastige gevallen |
        | **partikel** | een woord zoals *op* in *de kinderen ophalen*, dat bij het werkwoord hoort: tag RP in de Penn Treebank, relatie `compound:prt` in Universal Dependencies |
        | `" ".join(words[a:b])` | de woorden van plaats `a` tot plaats `b`, zonder `b` zelf, als één tekst met spaties ertussen |
        """),
        L.watch_out("Een telling uit een corpus is een telling van dat corpus. *Whale* is een gewoon woord in *Moby-Dick* en een zeldzaam woord in het Engels. Vraag je af wat er in het corpus ging, voordat je iets over een taal concludeert."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Kies een corpus dat past bij je vraag, en lees wat erin ging.
    2. Voor een concordantie loop je de woorden langs, en op elke plek waar je woord staat, neem je een paar woorden links en rechts ervan.
    3. Vlak bij het begin en het eind van de tekst neem je de woorden die er zijn.
    4. Lees de regels van boven naar beneden, en zoek wat terugkomt.
    5. Voor een vraag over zinsbouw, zoals welk partikel bij welk werkwoord hoort, gebruik je een treebank: de labels houden uit elkaar wat de woorden alleen niet uit elkaar houden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel geeft je de woorden van *Alice's Adventures in Wonderland* van Lewis Carroll (1865), een van de acht boeken van de course, als lijst. Het boek is Engels; drie van de acht boeken zijn Nederlands, en die kun je in de demo kiezen. `C.book_words("alice")` haalt de tekst uit de kopie van de course en knipt die in woorden, zonder leestekens en met de hoofdletters van het boek. De tweede cel zoekt elke plek waar *queen* (koningin) staat, met of zonder hoofdletter, en print de eerste met vier woorden aan elke kant.
    """)
    return


@app.cell
def book_cell():
    ALICE = C.book_words("alice")
    print(len(ALICE))
    print(ALICE[:12])
    return (ALICE,)


@app.cell
def example_look(ALICE):
    places = [i for i in range(len(ALICE)) if ALICE[i].lower() == "queen"]
    print(len(places), places[:5])
    i = places[0]
    print(ALICE[i - 4:i], ALICE[i], ALICE[i + 1:i + 5])
    print(" ".join(ALICE[i - 4:i]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het boek heeft 26.776 woorden, volgens de telling van de course. De eerste twaalf zijn niet het verhaal: *Illustration* geeft een plaatje aan, en dan volgen de titel, de schrijver en de naam van de uitgave. Een corpus bevat wat in de bestanden staat, tenzij iemand het eruit haalt. *Queen* staat op 68 plaatsen, de eerste op plaats 11.101, en vier woorden aan elke kant geven *An invitation from the Queen to play croquet The*: een uitnodiging van de koningin om te croquetten. Het laatste *The* begint al aan de volgende zin, want de lijst woorden heeft geen leestekens. `" ".join(...)` maakt van een lijst woorden één tekst, met een spatie tussen elke twee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kwic():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. kwic: een concordantie

        Schrijf `kwic(words, target, width)`: geef een lijst terug met een regel voor elke plek waar `target` in `words` staat, met of zonder hoofdletters. `target` is in kleine letters. Een regel is een tuple van drie teksten: de `width` woorden vóór het woord, met spaties ertussen; het woord zoals de tekst het schrijft; en de `width` woorden erna, met spaties ertussen. Vlak bij het begin van de lijst heeft de linkerkant minder woorden, en vlak bij het eind de rechterkant. Bij `kwic(ALICE, "rabbit", 3)` is de vierde regel `("suddenly a White", "Rabbit", "with pink eyes")`. Vervang de `...`. De check roept `kwic(ALICE, "rabbit", 3)` aan, en daarna eigen gevallen: een woord vlak na het begin, een woord op de eerste en op de laatste plaats, een woord met een hoofdletter, een breedte van 0, een woord dat er niet in staat, en *hatter* in *Alice*.
        """),
        L.task("Vind elke plek van een woord in een tekst, met de buren aan beide kanten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kwic():
    box_kwic = L.code_box("kwic", lines=11)
    box_kwic
    return (box_kwic,)


@app.cell(hide_code=True)
def check_kwic(box_kwic):
    result_kwic = L.check("kwic", box_kwic.value)
    result_kwic
    return (result_kwic,)


@app.cell(hide_code=True)
def hints_kwic():
    L.hints(
        "kwic",
        [
            "Begin met een lege lijst `lines`, en loop de plaatsen `i` langs met `range(len(words))`. Waar `words[i].lower()` gelijk is aan `target`, begint de linkerkant bij `i - width`; maak dat 0 als het onder 0 ligt. Maak dan van de twee slices tekst met `\" \".join(...)`, en voeg de tuple toe aan `lines`.",
            "Op welke plaats begint de rechterkant, en op welke plaats stopt die?",
            "`            right = \" \".join(words[i + 1:___])`",
        ],
        why="`lines = []`, dan `for i in range(len(words)):` met `if words[i].lower() == target:` eronder. Daarbinnen: `start = i - width`, `if start < 0:` met `start = 0` eronder, `left = \" \".join(words[start:i])`, `right = \" \".join(words[i + 1:i + 1 + width])` en `lines.append((left, words[i], right))`. Na de loop: `return lines`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kwic():
    mo.md(r"""
    De afbeelding onder de check toont de eerste regels van je concordantie van *rabbit* (konijn). De eerste twee komen uit de inhoudsopgave van het boek, niet uit het verhaal. Lees de regels van boven naar beneden: *White* komt telkens terug vóór *Rabbit*, wat je minder snel ziet als je het boek van begin tot eind leest. Een concordantie van *op* zou laten zien waar gewone tekst tekortschiet: *op* in *we lopen op de stoep* en in *we haalden de kinderen op* ziet er hetzelfde uit. Oefening 2 gebruikt een treebank om de twee uit elkaar te houden.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een eigen concordantie

    Kies een van de acht boeken van de course en typ een woord. De afbeelding geeft hoe vaak het woord voorkomt, per miljoen woorden, en de eerste 23 regels van de concordantie, een scherm vol naar de maat van Sinclair, gemaakt met de eigen `kwic` van de course. Probeer *whale* (walvis) in *Moby-Dick* en in *Frankenstein*, *I* (ik) in *The Adventures of Sherlock Holmes* en in *Moby-Dick*, of in de drie Nederlandse boeken *zoo*: de oude spelling van *zo*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_book = mo.ui.dropdown(options=C.book_names(), value="Moby-Dick; or, The Whale (1851)", label="Boek")
    demo_word = mo.ui.text(value="whale", label="Woord")
    demo_width = mo.ui.slider(2, 8, value=5, label="Woorden aan elke kant", show_value=True)
    mo.hstack([demo_book, demo_word, demo_width], justify="start", gap=1.5, wrap=True)
    return demo_book, demo_width, demo_word


@app.cell(hide_code=True)
def demo_look(demo_book, demo_width, demo_word):
    mo.Html(C.demo_picture(demo_book.value, demo_word.value, demo_width.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Hetzelfde woord kan in het ene boek veel voorkomen en in het volgende ontbreken. *Whale* staat 1.110 keer in *Moby-Dick*, ongeveer 5.100 keer per miljoen woorden, en 3 keer in *Frankenstein*. *I* neemt in *The Adventures of Sherlock Holmes*, dat dokter Watson in de ik-vorm vertelt, ongeveer 28.500 plaatsen per miljoen in, en in *Moby-Dick* ongeveer 9.200. In *Eline Vere* (1889) staat *zoo* 1.221 keer en *zo* geen enkele keer. Acht boeken zijn een verzameling, geen evenwichtig corpus: een telling eruit zegt iets over deze boeken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_particles():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. particles: wat een treebank toevoegt

        `SENTENCE` hieronder bevat de zin van de course *We haalden de kinderen op en zetten ze af*, gelabeld volgens de richtlijnen van Universal Dependencies, woord voor woord als `(form, head, relation)`, zoals in [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl). Woord nummer `i` staat op plaats `i - 1` van de lijst, en hoofdwoord 0 is de root. *Op* en *af* hebben de relatie `compound:prt`: het zijn partikels van *haalden* en *zetten*.

        Schrijf `particles(sentence)`: geef een lijst paren `(head, i)` terug, een voor elk woord met de relatie `compound:prt`: het nummer van het hoofdwoord, het werkwoord, en het eigen nummer van het woord, in de volgorde van de zin. Voor `SENTENCE` is dat `[(2, 5), (7, 9)]`. Vervang de `...`. De check roept `particles(SENTENCE)` aan, en daarna eigen gevallen: een zin zonder partikel, een woord met de relatie `compound`, een voorzetsel, een partikel vóór het werkwoord, en een zin van één woord.
        """),
        L.task("Vind de partikels van een zin aan de labels, met het werkwoord waar elk partikel bij hoort."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = [
        ("We", 2, "nsubj"),
        ("haalden", 0, "root"),
        ("de", 4, "det"),
        ("kinderen", 2, "obj"),
        ("op", 2, "compound:prt"),
        ("en", 7, "cc"),
        ("zetten", 2, "conj"),
        ("ze", 7, "obj"),
        ("af", 7, "compound:prt"),
    ]
    return (SENTENCE,)


@app.cell(hide_code=True)
def exercise_particles():
    box_particles = L.code_box("particles", lines=7)
    box_particles
    return (box_particles,)


@app.cell(hide_code=True)
def check_particles(box_particles):
    result_particles = L.check("particles", box_particles.value)
    result_particles
    return (result_particles,)


@app.cell(hide_code=True)
def hints_particles():
    L.hints(
        "particles",
        [
            "Begin met een lege lijst `found`, en loop de woordnummers `i` langs van 1 tot en met `len(sentence)`. Haal de drie delen van woord `i` uit elkaar in `form`, `head` en `relation`. Als `relation` gelijk is aan `\"compound:prt\"`, voeg je het paar `(head, i)` toe aan `found`.",
            "Op welke plaats van de lijst staat woord nummer `i`?",
            "`        form, head, relation = ___`",
        ],
        why="`found = []`, dan `for i in range(1, len(sentence) + 1):` met `form, head, relation = sentence[i - 1]` eronder, en `if relation == \"compound:prt\":` met `found.append((head, i))`. Na de loop: `return found`. `particles(SENTENCE)` geeft `[(2, 5), (7, 9)]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_particles():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op de zinnen van UD_Dutch-Alpino, bijna 12.000 zinnen uit kranten en ander geschreven Nederlands, en telt de paren van werkwoord en partikel die je functie vindt. *Vond plaats* staat bovenaan: een partikel lijkt niet altijd op een voorzetsel, want in *plaatsvinden* is het partikel *plaats* een zelfstandig naamwoord. Van de 1.640 partikels staan er 690 minstens twee woorden na het werkwoord, achteraan in de zin, en 650 ervoor, zoals *op* in *om ze op te halen*. Een concordantie van *plaats* vindt elke *plaats*, ook het gewone zelfstandig naamwoord; de labels, die mensen volgens een richtlijn toevoegden, vinden alleen de partikels.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Oefening 3. Labelen zoals de makers van een treebank

    Voor het Nederlands werkt deze test: zet *omdat* voor de zin, zodat het werkwoord achteraan komt. Een partikel plakt dan aan het werkwoord vast, zoals in *omdat we de kinderen ophaalden*; een voorzetsel blijft bij het zelfstandig naamwoord staan. Doe de test met het schuingedrukte woord in elke vetgedrukte zin, en zet elke zin bij de groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_label():
    sort_label = L.sort("label")
    sort_label
    return (sort_label,)


@app.cell(hide_code=True)
def check_label(sort_label):
    result_label = L.check("label", sort_label)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def text_exercise_purpose():
    mo.md(r"""
    ### Oefening 4. Welk corpus?

    Je wilt weten hoe mensen in het dagelijks leven Nederlands spreken, en je kunt kiezen uit drie verzamelingen.
    """)
    return


@app.cell(hide_code=True)
def choice_purpose_cell():
    choice_purpose = L.choice(
        "purpose",
        question="Welke verzameling past het best bij die vraag?",
        options={
            "right": "Het CGN: bijna 9 miljoen woorden opgenomen Nederlandse spraak",
            "size": "SoNaR-500: meer dan 500 miljoen woorden geschreven Nederlands, uit veel genres",
            "wiki": "De Nederlandse Wikipedia, met alle pagina's",
        },
        correct="right",
        why={
            "right": "Een corpus kies je voor een doel, en het CGN werd gemaakt om gesproken Nederlands te vertegenwoordigen, uit Nederland en uit Vlaanderen. Het is veel kleiner dan SoNaR-500, en toch past het beter: geen enkel aantal geschreven woorden vertelt je hoe mensen spreken.",
            "size": "SoNaR-500 is veel groter, maar het is geschreven Nederlands: geen enkel aantal geschreven woorden vertelt je hoe mensen spreken. Welke verzameling werd gekozen om spraak te vertegenwoordigen?",
            "wiki": "Wikipedia is geschreven, in de stijl van een encyclopedie, en niemand koos de pagina's om een taal te vertegenwoordigen. Welke verzameling werd gekozen om spraak te vertegenwoordigen?",
        },
    )
    choice_purpose
    return (choice_purpose,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wie hangt van wie af](/course/open/nlp-dependencies?lang=nl) las dependency trees; deze les liet zien waar zulke bomen vandaan komen. Een corpus is een verzameling teksten die voor een doel gekozen is, en wat het over een taal zegt, hangt af van wat erin ging. Een treebank voegt aan elke zin een boom toe en een lexicale database feiten over woorden, en mensen voegen die labels toe volgens geschreven richtlijnen, die niet elk geval beslissen. Je maakte een concordantie, vond de partikels van een treebank aan de labels, labelde zelf zinnen volgens een richtlijn, en koos een corpus bij een vraag. De volgende les van het domein telt de woorden van de boeken van de course: de wet van Zipf en de wet van Heaps.

    **Een voorbeeld uit het dagelijks leven.** Zoek je een volgende keer een uitdrukking op om te zien hoe anderen die gebruiken, denk dan aan Sinclair: het web is enorm, maar niemand koos de pagina's om een taal te vertegenwoordigen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_purpose, result_kwic, result_label, result_particles):
    L.wrap_up(result_kwic, result_particles, result_label, choice_purpose)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
