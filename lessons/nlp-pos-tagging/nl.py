# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-pos-tagging. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pos-tagging", "nl")
    C = L.checks   # de twee treebanks met de training sets en test sets, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl) en [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spans():
    quick_spans = L.choice(
        "quick-spans",
        question="Tags voor namen hebben een `B-` en een `I-`, zoals `B-LOC` en `I-LOC`. De tags voor woordsoorten hebben geen van beide. Waarom kunnen de tags voor woordsoorten zonder?",
        options={
            "fewer": "Er zijn minder woordsoorten dan types namen",
            "right": "Een woordsoort hoort bij één token, dus er is geen span om te markeren",
            "neighbors": "Twee woorden van dezelfde woordsoort staan nooit naast elkaar, dus niets hoeft uit elkaar gehouden te worden",
            "english": "B en I werken alleen voor het Engels",
        },
        correct="right",
        why={
            "right": "Elk token krijgt een eigen woordsoort, en geen tag loopt over meer tokens. Jurafsky en Martin maken hetzelfde onderscheid: bij woordsoorten krijgt elk woord één tag, dus er valt niets op te delen, terwijl je bij namen spans van tekst moet vinden en labelen. Deze les bouwt een tagger voor tags van één token.",
            "fewer": "Universal Dependencies heeft 17 tags voor woordsoorten, en Universal NER drie types namen: meer woordsoorten dus, niet minder. Wat markeert een B in een naam van twee tokens, en loopt een woordsoort ooit over twee tokens? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "neighbors": "Dat gebeurt vaak: in *een grote rode auto* zijn *grote* en *rode* allebei `ADJ`. Maar elk woord houdt een eigen tag, dus niets hoeft te zeggen waar een woordsoort ophoudt. Wat markeert een B in een naam? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "english": "Universal NER markeert namen met B en I in 13 talen. Het verschil zit in waar een tag bij hoort: bij een hele naam, of bij één token. Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
        },
    )
    quick_spans
    return (quick_spans,)


@app.cell(hide_code=True)
def quick_check_upos():
    quick_upos = L.choice(
        "quick-upos",
        question="In een CoNLL-U-bestand van Universal Dependencies heeft elk woord een regel met kolommen. In welke kolom staat de woordsoort?",
        options={
            "form": "`FORM`",
            "right": "`UPOS`",
            "head": "`HEAD`",
            "deprel": "`DEPREL`, de relatie",
        },
        correct="right",
        why={
            "right": "`UPOS`, de universele tag voor de woordsoort, zoals `NOUN` of `VERB`. Mensen vulden die kolom in voor elk woord van een treebank; een tagger moet de kolom invullen voor een zin waar nog niemand tags bij zette.",
            "form": "`FORM` bevat het woord zelf, zoals het in de zin staat. De tag van het woord staat in een andere kolom. Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
            "head": "`HEAD` bevat het nummer van het hoofdwoord, met 0 voor de root. In welke kolom staan tags zoals `NOUN` en `VERB`? Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
            "deprel": "`DEPREL` bevat de relatie met het hoofdwoord, zoals `nsubj` of `det`. In welke kolom staan tags zoals `NOUN` en `VERB`? Zie [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl).",
        },
    )
    quick_upos
    return (quick_upos,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De dependency trees van [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl) komen uit een treebank, waar mensen elk woord een tag en een hoofdwoord gaven. Voor een nieuwe zin heeft niemand dat gedaan, en een programma dat de boom tekent, gebruikt vaak de woordsoorten. Jurafsky en Martin leggen uit waarom: weet je of een woord een zelfstandig naamwoord of een werkwoord is, dan weet je iets over de woorden ernaast en over de structuur van de zin, en dat maakt woordsoorten taggen een belangrijk deel van parsing. Ook namen vinden, de volgende les, begint bij de woorden en de tags. Deze les bouwt een tagger door te tellen, en meet hoe goed de tagger is.

    **Een voorbeeld uit het dagelijks leven.** In *wij eten om zes uur* is *eten* een werkwoord, in *het eten is klaar* een zelfstandig naamwoord. Wie de zin naar het Engels vertaalt, kiest tussen *eat* en *food*, en moet dus eerst weten welke woordsoort het is. Het woord ervoor helpt: na *het* komt veel vaker een zelfstandig naamwoord dan een werkwoord.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="De Nederlandse treebank Alpino heeft een training set van 12.289 zinnen en een test set van 596 andere zinnen. Een tagger telt voor elk woord van de training set welke tag het woord het vaakst heeft, en geeft het woord elke keer die tag; een woord dat de tagger nooit zag, krijgt `NOUN`. Hoeveel van elke 100 woorden van de test set krijgen de tag die mensen eraan gaven?",
        options={
            "half": "Ongeveer 55: de helft van de woorden kan meer dan één tag hebben",
            "seventy": "Ongeveer 70",
            "right": "Ongeveer 88",
            "all": "Ongeveer 99: de meeste woorden hebben maar één tag",
        },
        reveal="**Ongeveer 88:** 9.729 van de 11.046 woorden van de test set, 88,1 van elke 100. Met alleen tellen heb je de meeste woorden dus al goed. De rest van de les kijkt waar het misgaat: bij woorden die niet in de training set staan, en bij woorden met meer dan één tag. Voor het Engels, met de treebank EWT, is het 86,2 van elke 100.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Woordsoorten taggen

    Part-of-speech tagging, of woordsoorten taggen, is volgens Jurafsky en Martin het geven van een woordsoort aan elk woord van een tekst. Het is sequence labeling met één tag per token, zoals in [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl), en de tags zijn de 17 van Universal Dependencies uit [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl). Het moeilijke deel: hetzelfde woord kan meer dan één tag hebben, en het doel is volgens de twee auteurs de juiste tag voor de situatie te vinden. Een **tagger** is een programma dat dit doet.

    ### Leren van het ene deel, meten op het andere

    Een treebank van Universal Dependencies bestaat uit delen. Een tagger leert alleen van de **training set**, en wordt gemeten op de **test set**: zinnen die de tagger nooit zag, zoals bij een nieuwe tekst. De Nederlandse treebank Alpino heeft een training set van 12.289 zinnen met 186.026 woorden en een test set van 596 zinnen met 11.046 woorden; de Engelse treebank EWT heeft 8.774 zinnen met 158.323 woorden, en 1.471 zinnen met 18.987 woorden. Elke treebank heeft ook een derde deel, waar deze les niets mee doet.

    De maat is **accuracy**: het aandeel van de woorden van de test set dat de tag krijgt die mensen het woord in de treebank gaven. Jurafsky en Martin omschrijven accuracy als het percentage tags van de test set dat overeenkomt met de labels van mensen. Deze les schrijft accuracy als een getal tussen 0 en 1, en in de tekst vaak als een aantal van elke 100.

    ### De eerste tagger: de vaakste tag

    Tel voor elk woord in de training set hoe vaak het woord elke tag heeft, en houd de tag die het woord het vaakst heeft. De tagger geeft het woord elke keer die tag, waar het woord ook staat. Een woord dat niet in de training set staat, een **onbekend woord**, krijgt `NOUN`, de vaakste tag van allemaal: 30.867 van de 186.026 Nederlandse woorden, ongeveer 1 op de 6. De woorden blijven zoals in de tekst, met de hoofdletters erbij.

    Zo'n eenvoudige methode is een **baseline**: een betere methode moet de baseline verslaan voordat je de score ervan vertrouwt. Jurafsky en Martin maken er een regel van voor elke classifier, een programma dat elk item een van een vast aantal labels geeft, zoals een tagger met woorden doet: vergelijk een classifier altijd met een baseline die minstens zo goed is als de most frequent class baseline, de methode die elk token de tag geeft die het token in de training set het vaakst had.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_hard():
    mo.md(r"""
    *onderzochte* en *liegt* staan niet in de training set en krijgen dus `NOUN`. *of* staat er wel in, vaker als `CCONJ`, zoals in *koffie of thee*, dan als `SCONJ`, zoals in *ik weet niet of het kan*; de eerste tagger geeft *of* dus `CCONJ`, ook hier, na *weten*.

    ### Waar de eerste tagger foutgaat

    Van de 26.665 verschillende woorden in de Nederlandse training set hebben er maar 765 meer dan één tag: ongeveer 1 op de 35. Maar het gaat om veelvoorkomende woorden, zoals *dat*, *het* en *of*, en samen vormen deze woorden 91.055 van de 186.026 woorden van de lopende tekst: de helft. Jurafsky en Martin vonden hetzelfde voor een oudere Engelse tagset met 45 tags: de meeste verschillende woorden, 85 tot 86 procent, hebben maar één tag, en toch heeft 55 tot 67 procent van de woorden in lopende tekst er meer dan één. In de Engelse training set hebben 1.596 van de 16.599 verschillende woorden meer dan één tag, ongeveer 1 op de 10, en ook die woorden vormen de helft van de tekst.

    ### De woorden eromheen beslissen

    Jurafsky en Martin noemen ook de woorden ervoor en erna als nuttige aanwijzing: staat er *the* voor een woord, dan is het woord waarschijnlijk geen werkwoord. Neem *dat*. Na *zei* is *dat* meestal een `SCONJ`, zoals in *de buurman zei dat het ging regenen*; na *op* is *dat* het vaakst een `DET`, zoals in *op dat moment*. Een tagger kan tags tellen per **paar** van het woord ervoor en het woord, net zoals [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) paren van tekens telde, en een woord de vaakste tag van het paar geven.
    """)
    return


@app.cell(hide_code=True)
def pairs_cell():
    mo.Html(C.pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        Veel paren komen niet voor in de training set: de test set heeft nieuwe combinaties van woorden. Dan valt de tagger terug op het woord alleen, en bij een onbekend woord op `NOUN`. Taggers die naar veel meer kijken, zoals naar de tags die zo'n tagger zelf aan de woorden ervoor gaf en naar de letters aan het eind van een woord, hebben ongeveer 97 van elke 100 woorden goed. Volgens Jurafsky en Martin halen allerlei methoden op Engelse treebanks ongeveer 97 procent, en is dat ook ongeveer wat mensen halen, in elk geval voor het Engels. Op nieuwsteksten van de *Wall Street Journal* geven de twee auteurs de eerste tagger een accuracy van ongeveer 92 procent. Latere lessen bouwen zulke taggers.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **part-of-speech tagging** | elk woord van een tekst de woordsoort geven, als één tag per token |
        | **tagger** | een programma dat elk token een tag geeft |
        | **training set** | het deel van een treebank waar een tagger van leert |
        | **test set** | het deel waarop een tagger gemeten wordt: zinnen die de tagger nooit zag |
        | **accuracy** | het aandeel woorden dat de tag van de treebank krijgt, tussen 0 en 1 |
        | **baseline** | een eenvoudige methode die een betere methode moet verslaan, zoals de vaakste tag van elk woord |
        | **onbekend woord** | een woord dat niet in de training set staat |
        | `C.TRAIN_NL`, `C.TEST_NL` | de training set en de test set van Alpino: tuples van zinnen, elk een tuple van paren `(woord, tag)` |
        | `C.TRAIN_EN`, `C.TEST_EN` | hetzelfde voor EWT |
        | `C.BEST_NL`, `C.BEST_EN` | de tabel van de course met de vaakste tag van elk woord van een training set; je kunt de tabel lezen, niet veranderen |
        | `C.PAIRS_NL`, `C.PAIRS_EN` | de tabel van de course met de vaakste tag van elk paar `(woord ervoor, woord)` van een training set; je kunt de tabel lezen, niet veranderen |
        | `C.SAMPLE`, `C.SAMPLE_BEST` | twee korte Engelse zinnen met de tags, en een kleine tabel met vaakste tags, voor Oefening 2 |
        | een tuple als key | een paar woorden kan een key van een dict zijn, net als een string: `pairs[("go", "to")]` |
        """),
        L.watch_out("De les over woordsoorten zette elk woord in kleine letters. De tagger hier houdt de hoofdletters, en dat maakt uit: in de Nederlandse training set is *Van* 125 keer `PROPN`, zoals in *Van Gogh*, en *van* 5.298 keer `ADP`; *Tweede* is 32 keer `PROPN`, zoals in *Tweede Kamer*, en *tweede* 91 keer `ADJ`. Een woord met een hoofdletter aan het begin van een zin kan onbekend zijn, ook als de training set het woord met kleine letters heeft."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Tel: voor elk woord van de training set, hoe vaak het woord elke tag heeft.
    2. Kies: houd voor elk woord de tag met het hoogste aantal.
    3. Meet: tag elk woord van de test set, met `NOUN` voor een onbekend woord, en geef het aandeel terug dat de tag van de treebank krijgt.
    4. Kijk naar het woord ervoor: heeft de training set het paar van het woord ervoor en het woord, dan beslist het paar; anders het woord alleen; anders `NOUN`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel doet stap 1 van het recept voor beide training sets, en print de aantallen van *that* in EWT en van *dat* in Alpino.
    """)
    return


@app.cell
def example_counts():
    def tag_counts(sentences):
        counts = {}
        for sentence in sentences:
            for word, tag in sentence:
                if word not in counts:
                    counts[word] = {}
                if tag not in counts[word]:
                    counts[word][tag] = 0
                counts[word][tag] = counts[word][tag] + 1
        return counts


    counts_en = tag_counts(C.TRAIN_EN)
    counts_nl = tag_counts(C.TRAIN_NL)
    print(counts_en["that"])
    print(counts_nl["dat"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *dat* heeft in de training set drie tags: 1.228 keer `SCONJ`, zoals in *ik denk dat het werkt*, 577 keer `PRON`, zoals in *dat klopt*, en 74 keer `DET`, zoals in *dat boek*. Engels *that* heeft er vijf, en ook daar is `SCONJ` de vaakste. Oefening 1 doet stap 2 voor elk woord.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. best_tags: de vaakste tag van elk woord

        Schrijf `best_tags(counts)`. `counts` is een tabel van het soort dat `tag_counts` van het uitgewerkte voorbeeld maakt: een dict met voor elk woord een dict van de tags en hoe vaak het woord elke tag heeft. Geef een dict terug met voor elk woord de tag die het woord het vaakst heeft. Staan twee tags van een woord gelijk, dan is elk van beide goed. Vervang de `...`. De check roept `best_tags({"that": {"PRON": 641, "SCONJ": 733, "DET": 123}, "book": {"VERB": 6, "NOUN": 18}})` aan, wat `{'that': 'SCONJ', 'book': 'NOUN'}` geeft; daarna eigen gevallen: geen woorden, één woord met één tag, de vaakste tag eerst, en twee woorden waarvan het tweede lagere aantallen heeft dan het eerste.
        """),
        L.task("Zoek voor elk woord de tag met het hoogste aantal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_tags", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best_tags", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_tags",
        [
            "Twee loops, zoals in het uitgewerkte voorbeeld: een over de woorden van `counts`, en daarin een over de tags van `counts[word]`. Houd voor elk woord de beste tag tot nu toe en het aantal ervan bij in twee eigen namen, gezet voor de binnenste loop; na die loop krijgt de tabel het woord met de beste tag.",
            "Een tag wordt de nieuwe beste tag van het woord als het aantal hoger is dan iets. Hoger dan wat, en wat verandert er dan?",
            "`            if counts[word][tag] > ___:`",
        ],
        why="`best = {}`, dan `for word in counts:` met daaronder `top = \"\"` en `most = 0`, en `for tag in counts[word]:` met daaronder `if counts[word][tag] > most:`; onder de `if` komen `most = counts[word][tag]` en `top = tag`. Na de binnenste loop `best[word] = top`, en na de buitenste loop `return best`. Het voorbeeld geeft `{'that': 'SCONJ', 'book': 'NOUN'}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op de aantallen van beide training sets. Kijk naar *dat*: `SCONJ` wint met 1.228, dus de eerste tagger noemt *dat* elke keer een voegwoord, ook in *dat boek*, waar het woord bij *boek* hoort en `DET` is. Oefening 2 meet hoe vaak de vaakste tag goed is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. accuracy: hoeveel woorden zijn goed?

        Schrijf `accuracy(sentences, best)`. `sentences` bevat zinnen, elk een tuple van paren `(woord, tag)` met de tag van de treebank, en `best` is een tabel met vaakste tags van het soort dat `best_tags` maakt. Voor elk woord is de gok van de eerste tagger de tag in `best`, of `"NOUN"` als `best` het woord niet heeft. Geef het aandeel terug van alle woorden waarvan de gok de tag van de treebank is: een getal tussen 0 en 1. Vervang de `...`. De check roept `accuracy(C.SAMPLE, C.SAMPLE_BEST)` aan: de twee Engelse zinnen *Book that flight.* en *Hand me that book.* (boek die vlucht, en geef me dat boek), en een kleine tabel die *Hand* niet heeft en *that* de tag `SCONJ` geeft. Dat geeft `0.6666666666666666`, 6 van de 9 tokens; daarna eigen gevallen: een lege tabel, een woord in de tabel, een onbekend woord dat een zelfstandig naamwoord is, twee zinnen van verschillende lengte, en beide hele test sets met de tabellen van de course.
        """),
        L.task("Gok de tag van elk woord met de tabel, en geef het aandeel goede gokken terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=12)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Tel in `right` de woorden waarvan de gok de tag van de treebank is, en tel elk woord in een tweede naam. Twee loops, zoals in het uitgewerkte voorbeeld: over de zinnen, en over de paren `(word, tag)` van elke zin. Aan het eind geeft de functie een aandeel terug.",
            "Wat is de gok voor een woord dat `best` niet heeft? En welk getal telt alle woorden, ook de foute?",
            "`    return right / ___`",
        ],
        why="`right = 0` en `total = 0`, dan `for sentence in sentences:` en `for word, tag in sentence:`; daaronder `if word in best:` met `guess = best[word]`, en `else:` met `guess = \"NOUN\"`; dan `if guess == tag:` met `right = right + 1`, en `total = total + 1` voor elk woord. Na de loops `return right / total`. Het voorbeeld geeft `0.6666666666666666`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    In de afbeelding heeft de eerste tagger 88,1 van elke 100 woorden van de Nederlandse test set goed, en 86,2 van de Engelse. De tabel verdeelt de Nederlandse woorden in drie groepen. Woorden met één tag in de training set zijn bijna altijd goed. Woorden met meer dan één tag gaan fout waar de vaakste tag niet past. En van de onbekende woorden krijgt minder dan de helft de goede tag: 1 op de 7 woorden is onbekend, maar de onbekende woorden zijn goed voor 821 van de 1.317 fouten. De grootste groep daarvan, 342 woorden, bestaat uit namen, `PROPN`, die de tagger `NOUN` noemt. Jurafsky en Martin noemen een aanwijzing die een tagger daar kan gebruiken: woorden met een hoofdletter aan het begin zijn waarschijnlijk eigennamen.

    Accuracy telt woorden, geen zinnen. Maar 104 van de 596 zinnen van de Nederlandse test set hebben elk woord goed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. tag_words: het woord ervoor beslist

        Schrijf `tag_words(words, best, pairs)`: geef een lijst terug met één tag voor elk woord van de lijst `words`. `best` is een tabel met vaakste tags zoals hiervoor, en `pairs` is een tabel met een tag voor elk paar `(woord ervoor, woord)`, op dezelfde manier geteld: de keys zijn tuples van twee woorden. Neem voor elk woord de tag van het paar met het woord ervoor als `pairs` dat paar heeft; anders de tag in `best`; en anders `"NOUN"`. Het eerste woord heeft geen woord ervoor. Vervang de `...`. De check roept `tag_words(["go", "to", "school"], {"go": "VERB", "to": "PART"}, {("go", "to"): "ADP"})` aan, wat `['VERB', 'ADP', 'NOUN']` geeft; daarna eigen gevallen: *to* als eerste woord, een paar voor het woord na *to*, geen woorden, een onbekend woord, een paar voor een woord dat `best` niet heeft, en een zin van elke treebank met de tabellen van de course, `C.BEST_EN` met `C.PAIRS_EN` en `C.BEST_NL` met `C.PAIRS_NL`.
        """),
        L.task("Tag elk woord met het paar als dat kan, anders met het woord alleen, anders met NOUN."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tag():
    box_tag = L.code_box("tag_words", lines=10)
    box_tag
    return (box_tag,)


@app.cell(hide_code=True)
def check_tag(box_tag):
    result_tag = L.check("tag_words", box_tag.value)
    result_tag
    return (result_tag,)


@app.cell(hide_code=True)
def hints_tag():
    L.hints(
        "tag_words",
        [
            "Eén loop over de posities `i` van `words`, met `word = words[i]`, en drie takken: `if`, `elif` en `else`. De eerste tak vraagt of het paar `(words[i - 1], word)` in `pairs` staat, de tweede of `word` in `best` staat.",
            "Welk woord heeft geen woord ervoor? En wat geeft `words[i - 1]` voor dat woord in Python?",
            "`        if ___ and (words[i - 1], word) in pairs:`",
        ],
        why="`tags = []`, dan `for i in range(len(words)):` met `word = words[i]`; onder `if i > 0 and (words[i - 1], word) in pairs:` komt `tags.append(pairs[(words[i - 1], word)])`, onder `elif word in best:` komt `tags.append(best[word])`, en onder `else:` komt `tags.append(\"NOUN\")`. Na de loop `return tags`. Het voorbeeld geeft `['VERB', 'ADP', 'NOUN']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tag():
    mo.md(r"""
    In de afbeelding maakt het woord ervoor van *dat* in *In dat platform zullen alle JGZ-disciplines vertegenwoordigd zijn* een `DET`, en van *dat* in *zoals dat ook eigenlijk moet* een `PRON`. Op de hele Nederlandse test set heeft de tagger met paren 88,7 van elke 100 woorden goed, en van de woorden met meer dan één tag 93,2 in plaats van 92,0. Bij de onbekende woorden helpt het paar niet: ook de paren met die woorden staan niet in de training set.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen zin

    Typ een zin en bekijk de tags van beide taggers, met de tabellen van Alpino. Zet een spatie tussen de woorden en voor leestekens, zoals de treebank dat doet. Probeer *Op dat moment regende het .* en *Het eten is klaar .*, en daarna *Dat boek is mooi .*: welke tag gaat daar fout, en wat in de training set kan dat veroorzaken?
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value=C.DEMO_TEXT[L.lang], label="Een zin", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Oefening 4. Welke buur beslist?

    *dat* is een `SCONJ` in *ik denk dat het werkt*, een `PRON` in *dat klopt*, en een `DET` in *dat boek*. In de Nederlandse test set komt *dat* 117 keer voor. De eerste tagger heeft het woord 76 keer goed, en je tagger met het woord ervoor 84 keer.
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after",
        question="Welk ander woord zou een tagger waarschijnlijk meer helpen om voor *dat* te kiezen tussen `SCONJ`, `PRON` en `DET`?",
        options={
            "first": "Het eerste woord van de zin",
            "right": "Het woord na *dat*",
            "two_before": "Het woord twee plaatsen voor *dat*, want het woord direct ervoor zegt te weinig",
            "capital": "Of *dat* met een hoofdletter begint",
        },
        correct="right",
        why={
            "right": "Na *dat* als `SCONJ` komt een zin, zoals in *dat het werkt*; na *dat* als `DET` een zelfstandig naamwoord, zoals in *dat boek*; en na *dat* als `PRON` van alles: een werkwoord, zoals in *dat klopt*, maar ook een voorzetsel of een bijwoord. Een tagger die paren van een woord en het woord erna telt, heeft *dat* 89 van de 117 keer goed. Jurafsky en Martin noemen beide buren: de woorden ervoor en erna. Taggers die naar beide kanten kijken, en naar de tags die de tagger zelf kiest, komen in latere lessen.",
            "first": "Het eerste woord van een zin staat vaak ver van *dat*, en zegt er weinig over. *Ik weet dat het werkt* en *Ik weet dat antwoord niet* beginnen hetzelfde. Welk woord direct naast *dat* verschilt tussen de twee?",
            "two_before": "Twee plaatsen terug hebben *Ik weet dat het werkt* en *Ik weet dat antwoord niet* allebei *Ik*. Wat komt in elke zin direct na *dat*?",
            "capital": "*Dat* met een hoofdletter staat aan het begin van een zin, en is daar vaak een `PRON`, zoals in *Dat klopt*. Maar 9 van de 10 keer heeft *dat* geen hoofdletter. Welk woord direct naast *dat* verschilt tussen *dat het werkt* en *dat boek*?",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) liet zien dat hetzelfde woord meer dan één tag kan hebben, en [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl) noemde het geven van een label aan elk token sequence labeling. Hier bouwde je een tagger door te tellen: de vaakste tag van elk woord heeft 88 van elke 100 Nederlandse woorden goed, gemeten op een test set die de tagger nooit zag. De fouten vallen op onbekende woorden en op woorden met meer dan één tag, en bij de tweede soort helpt het woord ervoor, zoals de aantallen van paren in [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl). De volgende les, *Namen vinden: named entity recognition*, tagt namen met de B, I en O van de les hiervoor, met regels, lijsten en een tagger die leert.

    **Een voorbeeld uit het dagelijks leven.** Wie *eten* vooral kent als werkwoord, leest *het eten is klaar* even verkeerd. Het woord *het* ervoor is de aanwijzing dat het deze keer een zelfstandig naamwoord is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after, result_accuracy, result_best, result_tag):
    L.wrap_up(result_best, result_accuracy, result_tag, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
