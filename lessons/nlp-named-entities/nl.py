# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-named-entities. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-named-entities", "nl")
    C = L.checks   # de zinnen van Universal NER, de gazetteer, de tabellen van de tagger, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl) en [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spans():
    quick_spans = L.choice(
        "quick-spans",
        question="Een tagger geeft de vier tokens van *near the Mississippi River* de tags `O O I-LOC I-LOC`. Welke spans leest de course uit deze tags, zoals het script van CoNLL dat doet?",
        options={
            "none": "Geen: een span moet met een `B-` beginnen",
            "two": "Twee plaatsen van elk één token, `(2, 3, \"LOC\")` en `(3, 4, \"LOC\")`",
            "right": "Eén plaats, `(2, 4, \"LOC\")`",
            "drop": "Eén plaats, `(3, 4, \"LOC\")`: de eerste `I-` zet niets voort, dus de course laat de tag weg",
        },
        correct="right",
        why={
            "right": "Een `I-` na een `O` begint een nieuwe span, zoals het script van CoNLL het leest, en de volgende `I-LOC` maakt de span één token langer: *Mississippi River*, van positie 2 tot 4. Deze les leest de tags van de tagger op dezelfde manier.",
            "none": "Het script van CoNLL, en de course ook, leest een `I-` die niets voortzet als het begin van een nieuwe span. Welke span geeft dat hier? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "two": "De tweede `I-LOC` staat direct na een span van hetzelfde type, en maakt de span dus één token langer. Alleen een `B-` op die plek zou een tweede plaats beginnen. Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "drop": "Er valt niets weg: het script van CoNLL leest een `I-` na een `O` als het begin van een span. Waar eindigt de span? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
        },
    )
    quick_spans
    return (quick_spans,)


@app.cell(hide_code=True)
def quick_check_unknown():
    quick_unknown = L.choice(
        "quick-unknown",
        question="In *Woordsoorten taggen* kwam de eerste tagger een woord tegen dat niet in de training set stond. Welke tag kreeg dat woord?",
        options={
            "propn": "`PROPN`, omdat een woord dat de tagger nooit zag meestal een naam is",
            "right": "`NOUN`, de vaakste tag van allemaal",
            "before": "De tag van het woord ervoor",
            "skip": "Geen: de tagger sloeg het woord over",
        },
        correct="right",
        why={
            "right": "`NOUN` is de vaakste tag van de training set, dus elk onbekend woord kreeg `NOUN`, waar het woord ook stond. Een tagger voor namen kan meer bekijken van een woord dat de tagger niet kent, en ook de tag direct ervoor: deze les bouwt zo'n tagger.",
            "propn": "De eerste tagger wist niets van een onbekend woord, en gaf elk onbekend woord dus één en dezelfde tag, en in EWT is dat niet `PROPN`. Op welke tag viel de tagger terug? Zie [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl).",
            "before": "De eerste tagger keek naar elk woord alleen, niet naar de woorden eromheen. Wat kreeg een woord waar de tagger niets van wist? Zie [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl).",
            "skip": "Een tagger geeft elk token een tag, ook een woord dat de tagger nooit zag: sequence labeling heeft evenveel labels als tokens. Welke tag kreeg het woord? Zie [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl).",
        },
    )
    quick_unknown
    return (quick_unknown,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Over wie gaat een tekst, waar speelt de tekst zich af, en welke bedrijven komen erin voor? Een zoekmachine, een nieuwssite en een helpdesk stellen die vragen bij elke tekst die binnenkomt. Volgens Jurafsky en Martin is het taggen van named entities een nuttige eerste stap bij veel taken met taal. Je wilt bijvoorbeeld weten wat een klant van één bepaald bedrijf of product vindt, en named entities zijn ook een nuttig begin bij het beantwoorden van vragen en bij het koppelen van tekst aan gestructureerde kennis, zoals Wikipedia. Een programma dat de namen van mensen in een tekst onherkenbaar maakt voordat iemand de tekst deelt, moet die namen eerst vinden. Deze les vindt namen op twee manieren: met een lijst van namen die het programma al kent, en met een tagger die leert van zinnen waarin mensen de namen markeerden.

    **Een voorbeeld uit het dagelijks leven.** Je leest *een brief van dr. Okonkwo uit Tilburg*. Je weet dat *Tilburg* een stad is, omdat je ervan gehoord hebt. Van *Okonkwo* heb je nooit gehoord, en toch weet je dat het om een persoon gaat: de *dr.* ervoor en de hoofdletter vertellen het je.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="De Engelse zinnen van Universal NER bestaan uit delen, net als de treebank van *Woordsoorten taggen*: een training set van 8.773 zinnen om van te leren, en een test set van 1.471 andere zinnen om op te meten. De test set heeft 802 namen. Hoeveel daarvan staan ook in de training set, met precies dezelfde woorden?",
        options={
            "ten": "Ongeveer 10 van elke 100",
            "quarter": "Ongeveer 25 van elke 100",
            "right": "Ongeveer 40 van elke 100",
            "most": "Ongeveer 70 van elke 100: de meeste namen komen terug",
        },
        reveal="**Ongeveer 40:** 324 van de 802 namen van de test set staan met dezelfde woorden in de training set, 40 van elke 100. De andere 478 zijn nieuw: mensen, winkels en plaatsen die de training set nergens noemt. Een lijst met de namen van de training set kan alleen die 324 vinden.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Named entities

    Volgens Jurafsky en Martin is een **named entity**, grofweg, alles waar je met een eigennaam naar kunt verwijzen: een persoon, een plaats, een organisatie. De taak om named entities te vinden heet **named entity recognition** (NER): de spans van een tekst vinden die eigennamen zijn, en elke span het type van de named entity geven. Spans en types werken zoals in [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl): Universal NER markeert drie types, `PER`, `LOC` en `ORG`. Jurafsky en Martin noemen ook een vierde type dat veel datasets gebruiken, `GPE`, een geopolitieke eenheid zoals een land of een staat; Universal NER telt zulke namen als plaatsen. Volgens de twee auteurs rekent men ook vaak datums, tijden en prijzen tot de named entities. Universal NER doet dat niet.

    ### Named entity of eigennaam

    Een tagger voor woordsoorten kan al zien dat *Janet*, *Stanford University* en *Colorado* eigennamen zijn, `PROPN`. Volgens Jurafsky en Martin gaat het daarbij om een grammaticale eigenschap van de woorden. Een named entity gaat over waar de woorden naar verwijzen: een persoon, een organisatie, een plaats. De twee overlappen veel, maar niet helemaal. In de 11.702 zinnen die Universal NER en de treebank van [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl) in dezelfde tokens knippen, zijn 8.528 van de 9.634 tokens binnen een naam `PROPN`, ongeveer 9 van de 10. De andere zijn woorden zoals *United* in *United States* en *of* in *Bay of Plenty*. Andersom staan 3.068 van de 11.596 tokens met `PROPN` buiten elke naam: maanden zoals *April*, dagen zoals *Friday*, titels zoals *Dr.* en *President*, en *Americans*.

    ### Twee moeilijke delen

    Jurafsky en Martin noemen twee dingen die het vinden van namen onder meer moeilijk maken. Het eerste is de segmentatie: je moet beslissen wat een named entity is en wat niet, en waar de grenzen liggen. *Mississippi* is een naam, en *Mississippi River* ook. Het tweede is het type. Dezelfde naam kan een persoon, een plaats of een organisatie zijn, zoals *Washington* in de demo van [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl). Het voorbeeld van de twee auteurs: *JFK* kan een persoon zijn, het vliegveld in New York, of een van de vele scholen, bruggen en straten in de Verenigde Staten met die naam.
    """)
    return


@app.cell(hide_code=True)
def text_idea_list():
    mo.md(r"""
    ### Lijsten: een gazetteer

    De eenvoudigste manier om een naam te vinden is opzoeken. Jurafsky en Martin omschrijven een **gazetteer** als een lijst van plaatsnamen, vaak met miljoenen plaatsen en veel geografische en politieke informatie erbij. Er bestaan ook lijsten van andere namen, zoals van voornamen of van bedrijven. Deze les maakt een eigen lijst uit de training set: elke naam die mensen daar markeerden, als tuple van de woorden, met het type dat de naam het vaakst heeft. Dat geeft 2.067 namen, en de langste heeft 16 woorden. In de course is de lijst een dict, `C.GAZETTEER`: `C.GAZETTEER[("New", "Orleans")]` is `"LOC"`. *Washington* is 10 keer een plaats in de training set en één keer een organisatie, dus de lijst zegt `LOC`, elke keer.

    Twee namen van de lijst kunnen bij hetzelfde woord beginnen: *United Nations* staat in de lijst, en *United Nations Security Council* ook. De **langste match** beslist: neem op elke positie de langste naam van de lijst die daar begint, en ga verder na het eind ervan, zodat namen niet overlappen.

    ### Regels

    Veel systemen in de praktijk voegen regels toe die mensen met de hand schrijven, zoals de regel dat een woord met een hoofdletter direct na *Dr.* een persoon is. Volgens Jurafsky en Martin werken commerciële systemen voor NER vaak met een praktische combinatie van lijsten en regels, met wat minder supervised machine learning. Een lijst en de regels erbij vinden alleen wat iemand erin stopte.
    """)
    return


@app.cell(hide_code=True)
def text_idea_tagger():
    mo.md(r"""
    ### Een tagger die leert

    Een tagger leert namen uit de zinnen van de training set, met de BIO-tags van [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl): `B-PER` op het eerste woord van een persoon, `I-PER` op de woorden erna, `O` buiten elke naam. De tagger telt, zoals de eerste tagger van [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl), maar met twee aanwijzingen.

    De eerste aanwijzing is de **tag ervoor**: de tag die de tagger zelf net aan het woord ervoor gaf. Na `B-PER` is een woord vaak het volgende woord van dezelfde naam; na `O` staan de meeste woorden buiten elke naam. Vóór het eerste woord van een zin telt `O` als de tag ervoor.

    De tweede aanwijzing helpt bij een woord dat de tagger nooit na die tag zag: de **word shape**. Jurafsky en Martin omschrijven de word shape als het abstracte patroon van de letters van een woord: een hoofdletter wordt X, een kleine letter x, een cijfer d, en leestekens blijven staan. In de **short word shape** valt een herhaling van dezelfde soort teken weg: *Villanueva* wordt `Xx`, *U.S.* wordt `X.X.` en *DC10-30* wordt `Xd-d`.

    De tagger heeft twee tabellen, gemaakt door te tellen in de training set. `C.BY_WORD` geeft de vaakste tag van elk paar `(woord, tag ervoor)`, en `C.BY_SHAPE` de vaakste tag van elk paar `(short word shape, tag ervoor)`. Elk woord krijgt, van links naar rechts, de tag van het paar in `C.BY_WORD` als die tabel het paar heeft; anders de tag van de short word shape in `C.BY_SHAPE`; anders `O`. *Woodward* staat niet in de training set, maar na `B-PER` krijgt de shape `Xx` de tag `I-PER`, dus *Bob Woodward* komt eruit als één persoon: een naam die niet in de lijst staat.
    """)
    return


@app.cell(hide_code=True)
def shapes_cell():
    mo.Html(C.shapes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_both():
    mo.md(r"""
    ### Beide op één zin

    De afbeelding hieronder laat de namen van één zin van de test set zien: de namen die mensen markeerden, de namen van de gazetteer met de langste match, en de namen van de tagger.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        De lijst heeft *Mississippi* als plaats, maar niet *Mississippi River*, en vindt dus maar een deel van die naam. De tagger kent *River* na `B-LOC` als `I-LOC`, en zet de naam voort.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **named entity** | iets waar je met een eigennaam naar kunt verwijzen: een persoon, een plaats of een organisatie |
        | **named entity recognition (NER)** | de spans van een tekst vinden die namen zijn, elk met het type |
        | **gazetteer** | een lijst van namen van één soort, zoals plaatsen; hier een dict van namen van drie types, elk met het type |
        | **langste match** | op elke positie de langste naam van de lijst die daar begint |
        | **tag ervoor** | de tag die de tagger aan het woord ervoor gaf; vóór het eerste woord is dat `O` |
        | **word shape**, **short word shape** | het patroon van een woord: `X` voor een hoofdletter, `x` voor een kleine letter, `d` voor een cijfer, en leestekens ongewijzigd; in de short word shape wordt een reeks van dezelfde soort één teken |
        | **training set**, **test set** | het deel van de zinnen waar een tagger van leert, en het deel waarop de tagger gemeten wordt, dat de tagger nooit zag |
        | `C.TRAIN`, `C.TEST` | de training set en de test set van de Engelse zinnen van Universal NER: tuples van zinnen, elk een tuple van paren `(woord, tag)` |
        | `C.EXAMPLE` | de zin van het uitgewerkte voorbeeld, *Alteration of the Mississippi River ...*, als paren `(woord, tag)` |
        | `C.GAZETTEER` | de gazetteer van de course: een dict van elke naam van de training set, een tuple van woorden, naar het vaakste type; je kunt de dict lezen, niet veranderen |
        | `C.BY_WORD`, `C.BY_SHAPE` | de tabellen van de tagger: dicts van elk paar `(woord, tag ervoor)` en `(short word shape, tag ervoor)` naar de vaakste tag; je kunt de tabellen lezen, niet veranderen |
        | `C.spans(tags)` | de spans van een lijst BIO-tags, zoals je `to_spans` uit *Een label per woord* de spans gaf |
        | `(word,)` | een tuple met één item: de komma maakt er een tuple van |
        | `tuple(a_list)` | een tuple met de items van de lijst, in dezelfde volgorde; een lijst kan geen key van een dict zijn, een tuple wel |
        | `range(a, b, -1)` | de gehele getallen van `a` omlaag tot `b`, zonder `b`: `range(5, 2, -1)` geeft 5, 4 en 3 |
        | `ch.isupper()`, `ch.islower()`, `ch.isdigit()` | `True` als het teken `ch` een hoofdletter, een kleine letter of een cijfer is |
        """),
        L.watch_out("Een gazetteer geeft een naam overal hetzelfde type: *Washington* is een plaats in de lijst, ook in *the Washington Post*. En de lijst markeert een woord dat maar af en toe een naam is. De training set heeft *us* 4 keer als plaats, klein geschreven voor *US*, en 101 keer buiten elke naam; de lijst markeert elke *us*."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. De gazetteer, met de langste match: ga de posities van links naar rechts af. Probeer op een positie die niet binnen een al gevonden naam ligt het stuk woorden van daar tot het eind van de zin, dan één woord korter, en zo verder tot één woord. Het eerste stuk dat in de lijst staat, is de naam; ga verder na het eind ervan.
    2. De tagger: ga de woorden van links naar rechts af, met `O` als tag vóór het eerste woord. Zoek `(woord, tag ervoor)` op in `C.BY_WORD`; staat het paar daar niet in, dan `(short word shape, tag ervoor)` in `C.BY_SHAPE`; staat ook dat paar er niet in, dan is de tag `O`. De tag die je geeft, is de tag vóór het volgende woord.
    3. Lees de namen uit de tags met `C.spans`, zoals in [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel zoekt één woord tegelijk op: `one_word_at` geeft de span van een naam van één woord op positie `i`, of `None`, en `find_names` gaat de posities van links naar rechts af. De cel print de namen die `find_names` in `C.EXAMPLE` vindt, en de namen die mensen daar markeerden.
    """)
    return


@app.cell
def example_find():
    def one_word_at(words, i, gazetteer):
        name = (words[i],)
        if name in gazetteer:
            return (i, i + 1, gazetteer[name])
        return None


    def find_names(words, gazetteer):
        found = []
        free = 0   # de eerste positie die niet binnen een al gevonden naam ligt
        for i in range(len(words)):
            if i >= free:
                span = one_word_at(words, i, gazetteer)
                if span is not None:
                    found.append(span)
                    free = span[1]
        return found


    words = [word for word, tag in C.EXAMPLE]
    tags = [tag for word, tag in C.EXAMPLE]
    print(find_names(words, C.GAZETTEER))
    print(C.spans(tags))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Eén woord tegelijk vindt *Mississippi*, een deel van de naam *Mississippi River*, en mist *New Orleans*: de lijst heeft *New Orleans*, maar *New* en *Orleans* niet los. Op de hele test set heeft één woord tegelijk 257 van de 802 namen goed. Oefening 1 zoekt in plaats daarvan de langste naam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_longest():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. longest_at: de langste naam die hier begint

        Schrijf `longest_at(words, i, gazetteer)`, met dezelfde argumenten als `one_word_at`: geef de span `(i, end, kind)` terug van de langste naam van `gazetteer` die op positie `i` begint, of `None` als daar geen naam begint. Probeer de eindes van het eind van de zin omlaag tot `i + 1`: het eerste stuk `words[i:end]` dat in `gazetteer` staat, als tuple, is het langste. Vervang de `...`. De check roept `longest_at(["the", "United", "Nations", "Security", "Council", "met"], 1, gazetteer)` aan met een kleine gazetteer met *United Nations*, *United Nations Security Council* en *Council*, wat `(1, 5, 'ORG')` geeft; daarna eigen gevallen: een positie waar geen naam begint, een naam van één woord, een naam tot het laatste woord, een positie binnen een naam, een naam die langer is dan de rest van de zin, de langste van twee namen die allebei tot het laatste woord lopen, en de `words` van het uitgewerkte voorbeeld met `C.GAZETTEER`.
        """),
        L.task("Zoek de langste naam van de gazetteer die op positie i begint."),
    ])
    return


@app.cell(hide_code=True)
def exercise_longest():
    box_longest = L.code_box("longest_at", lines=7)
    box_longest
    return (box_longest,)


@app.cell(hide_code=True)
def check_longest(box_longest):
    result_longest = L.check("longest_at", box_longest.value)
    result_longest
    return (result_longest,)


@app.cell(hide_code=True)
def hints_longest():
    L.hints(
        "longest_at",
        [
            "Eén loop over `end`, met `range`, vanaf `len(words)` omlaag met een stap van `-1`. Daarin `name = tuple(words[i:end])` en een `if` die vraagt of `name` in `gazetteer` staat. Een `return` in de loop stopt de functie bij de eerste naam die de functie vindt, en dat is de langste.",
            "`range` stopt vlak vóór het tweede getal. Welke `end` probeer je als laatste, voor het kortste stuk van één woord, en welk getal komt daarna als je omlaag telt?",
            "`    for end in range(len(words), ___, -1):`",
        ],
        why="`for end in range(len(words), i, -1):` met daaronder `name = tuple(words[i:end])`, dan `if name in gazetteer:` met daaronder `return (i, end, gazetteer[name])`; na de loop `return None`. `longest_at([\"the\", \"United\", \"Nations\", \"Security\", \"Council\", \"met\"], 1, gazetteer)` geeft `(1, 5, 'ORG')`.",
    )
    return


@app.cell(hide_code=True)
def text_after_longest():
    mo.md(r"""
    De afbeelding onder de check zet je `longest_at` op de plaats van `one_word_at` in `find_names`, en laat het geheel los op de hele test set. De langste match heeft 317 van de 802 namen goed, 60 meer dan één woord tegelijk. Een naam die niet in de lijst staat, vindt de langste match nooit, en sommige vondsten zijn fout: een deel van een langere naam, zoals *Bob* in *Bob Woodward*, of helemaal geen naam, zoals *us*. Oefening 2 maakt de word shape die de tagger nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. shape: de short word shape

        Schrijf `shape(word)`: geef de short word shape van `word` terug. Ga de tekens af. Een hoofdletter krijgt de soort `X`, een kleine letter `x`, een cijfer `d`, en elk ander teken is een eigen soort. Voeg een soort alleen aan de shape toe als de soort verschilt van de soort ervoor, zodat een reeks van dezelfde soort één teken wordt, en drie punten één punt. Vervang de `...`. De check roept `shape("Villanueva")` aan, wat `'Xx'` geeft; daarna eigen gevallen: *L'Occitane*, *I.M.F.*, *DC10-30*, *iPhone*, *us*, *US*, *2026*, een leeg woord, drie punten, *café* en *McDonald's*.
        """),
        L.task("Maak van een woord de short word shape."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shape():
    box_shape = L.code_box("shape", lines=14)
    box_shape
    return (box_shape,)


@app.cell(hide_code=True)
def check_shape(box_shape):
    result_shape = L.check("shape", box_shape.value)
    result_shape
    return (result_shape,)


@app.cell(hide_code=True)
def hints_shape():
    L.hints(
        "shape",
        [
            "Eén loop `for ch in word:`, met `if`, `elif` en `else` die `kind` een waarde geven: `ch.isupper()`, `ch.islower()`, `ch.isdigit()`, en anders het teken zelf. Bewaar de shape in een string `out` die begint als `\"\"`, en houd een tweede string bij voor de soort die je het laatst toevoegde, ook `\"\"` aan het begin.",
            "Wanneer voeg je `kind` toe aan `out`, en wat komt er direct daarna in je tweede string?",
            "`        if kind != ___:`",
        ],
        why="`out = \"\"` en `last = \"\"`, dan `for ch in word:` met `if ch.isupper():` en `kind = \"X\"`, `elif ch.islower():` en `kind = \"x\"`, `elif ch.isdigit():` en `kind = \"d\"`, `else:` en `kind = ch`; dan `if kind != last:` met daaronder `out = out + kind` en `last = kind`; na de loop `return out`. `shape(\"Villanueva\")` geeft `'Xx'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shape():
    mo.md(r"""
    De 16.600 verschillende woorden van de training set hebben maar 267 verschillende short word shapes, dus een woord dat de tagger nooit zag, heeft toch een shape die de tagger vaak zag. Oefening 3 bouwt de tagger.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. tag_sentence: taggen met de tag ervoor

        Schrijf `tag_sentence(words, by_word, by_shape)`: geef een lijst terug met één BIO-tag voor elk woord van `words`. `by_word` is een tabel met een tag voor elk paar `(woord, tag ervoor)`, en `by_shape` een tabel met een tag voor elk paar `(short word shape, tag ervoor)`, zoals `C.BY_WORD` en `C.BY_SHAPE`. Ga de woorden van links naar rechts af, met `"O"` als tag vóór het eerste woord. Geef elk woord de tag van `(word, before)` in `by_word` als die tabel het paar heeft; anders de tag van `(shape(word), before)` in `by_shape`; anders `"O"`. De tag die je geeft, is de tag vóór het volgende woord. In dit vak is `shape` de versie van de course van Oefening 2. Vervang de `...`. De check roept `tag_sentence(["Jane", "Villanueva", "said", "so"], by_word, by_shape)` aan met twee kleine tabellen, wat `['B-PER', 'I-PER', 'O', 'O']` geeft; daarna eigen gevallen: geen woorden, een woord dat geen van beide tabellen heeft, een eerste woord dat alleen `by_shape` kan taggen, een bekend woord waarvan de tag afhangt van de tag ervoor, een naam van drie woorden, en drie zinnen van de test set met `C.BY_WORD` en `C.BY_SHAPE`.
        """),
        L.task("Tag de woorden van links naar rechts, elk met de tag ervoor."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tag():
    box_tag = L.code_box("tag_sentence", lines=12)
    box_tag
    return (box_tag,)


@app.cell(hide_code=True)
def check_tag(box_tag):
    result_tag = L.check("tag_sentence", box_tag.value)
    result_tag
    return (result_tag,)


@app.cell(hide_code=True)
def hints_tag():
    L.hints(
        "tag_sentence",
        [
            "Begin met een lege lijst `tags`, `before = \"O\"` en een loop `for word in words:`. Daarin kiezen `if`, `elif` en `else` de `tag`: eerst `(word, before) in by_word`, dan `(shape(word), before) in by_shape`. Daarna komt `tags.append(tag)`.",
            "Als de loop naar het volgende woord gaat, welke tag staat er dan vóór dat woord?",
            "`        before = ___`",
        ],
        why="`tags = []` en `before = \"O\"`, dan `for word in words:` met `if (word, before) in by_word:` en `tag = by_word[(word, before)]`, `elif (shape(word), before) in by_shape:` en `tag = by_shape[(shape(word), before)]`, `else:` en `tag = \"O\"`; dan `tags.append(tag)` en `before = tag`; na de loop `return tags`. `tag_sentence([\"Jane\", \"Villanueva\", \"said\", \"so\"], by_word, by_shape)` geeft `['B-PER', 'I-PER', 'O', 'O']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tag():
    mo.md(r"""
    Op de hele test set heeft de tagger 343 van de 802 namen goed, 26 meer dan de gazetteer, met ongeveer evenveel foute vondsten: 100, tegen 97. Van de goede namen staan er 58 helemaal niet in de lijst, zoals *Bob Woodward* en *Bay of Plenty*: de tag ervoor zette een naam voort in woorden die de tagger daar nooit zag. Beide missen nog meer dan de helft van de namen. Een woord dat de tagger niet kent, krijgt na `O` bijna altijd `O`: `Xx` na `O` is meestal geen naam.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen zin

    Typ een Engelse zin, met spaties tussen de woorden en vóór leestekens, en bekijk de namen die de gazetteer en de tagger vinden. Probeer *Tell us about it .*, *I read the Washington Post .*, of een naam die niet in de training set staat, zoals *Dr. Okonkwo visited Tilburg .*
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value=C.DEMO_TEXT, label="Een zin", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_streets():
    mo.md(r"""
    ### Oefening 4. Een lijst of een tagger?

    Een gemeente krijgt brieven van inwoners, en wil daarin de namen van de 30 straten van de gemeente vinden. Niemand heeft de namen in de brieven gemarkeerd.
    """)
    return


@app.cell(hide_code=True)
def choice_streets_cell():
    choice_streets = L.choice(
        "streets",
        question="Welke manier past het best?",
        options={
            "tagger": "De tagger van Oefening 3, met de tabellen van Universal NER",
            "right": "Een gazetteer met de 30 straatnamen",
            "capitals": "Elk woord met een hoofdletter dat niet aan het begin van een zin staat",
            "wait": "Nog geen: eerst moeten mensen de namen in duizend brieven markeren",
        },
        correct="right",
        why={
            "right": "De 30 namen zijn bekend en veranderen niet, dus een lijst vindt elke straat, met het type, zonder één gemarkeerd voorbeeld. Veel systemen in de praktijk beginnen met lijsten en regels. Een lijst vraagt wel zorg: de lijst mist een straat die anders geschreven is, zoals met een tikfout, en markeert een straat met de naam van een persoon ook waar de persoon bedoeld is.",
            "tagger": "De tabellen komen uit reviews, blogs, nieuwsgroepen en antwoorden op vragen op het web, en de meeste van de 30 straten staan er niet in. Op de test set vond de tagger 58 namen die de tagger nooit als naam zag, vooral door een bekende naam voort te zetten. Wat kan een bekende naam vinden zonder gemarkeerde voorbeelden?",
            "capitals": "Dan markeer je ook elke persoon, elke organisatie en de namen van andere plaatsen. En een straat die met kleine letters geschreven is, zoals in veel teksten op het web, glipt erdoor. Wat weet precies naar welke 30 namen het moet zoeken?",
            "wait": "Gemarkeerde brieven zouden een tagger helpen, maar de 30 namen zijn al bekend. Wat kan een bekende naam vinden zonder voorbeelden?",
        },
    )
    choice_streets
    return (choice_streets,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl) schreef namen als spans en als BIO-tags, en [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl) bouwde een tagger door te tellen. Hier kwamen beide samen in named entity recognition. Een gazetteer uit de training set, gelezen met de langste match, heeft 317 van de 802 namen van de test set goed; de gazetteer vindt alleen namen die erin staan, en geeft een naam overal hetzelfde type. Een tagger die tags telt na de tag ervoor, met de word shape voor woorden die de tagger daar nooit zag, heeft er 343 goed, waarvan 58 namen die niet in de lijst staan. De volgende les, *Een tagger beoordelen: accuracy en span F1*, maakt van de goede namen, de foute vondsten en de gemiste namen één score.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe collega leert de namen op het werk op twee manieren: van de lijst met personeel aan de muur, en door te luisteren. *Kun je het even aan Okonkwo vragen?* vertelt de nieuwkomer dat er een persoon bedoeld is, voordat die naam op een lijst staat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_streets, result_longest, result_shape, result_tag):
    L.wrap_up(result_longest, result_shape, result_tag, choice_streets)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
