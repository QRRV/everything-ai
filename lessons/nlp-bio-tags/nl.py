# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-bio-tags. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bio-tags", "nl")
    C = L.checks   # de zinnen van Universal NER, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) en [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_book():
    quick_book = L.choice(
        "quick-book",
        question="In *Woordsoorten en woordgroepen* was *Book* een `VERB` in *Book that flight.* (boek die vlucht), en *book* een `NOUN` in *Hand me that book.* (geef me dat boek). Wat bepaalde elke keer de tag?",
        options={
            "capital": "De hoofdletter: *Book* met een hoofdletter is een werkwoord",
            "right": "De woorden eromheen",
            "first": "De plek in de zin: het eerste woord van een zin is altijd een werkwoord",
            "list": "Een woordenboek dat elk woord één tag geeft",
        },
        correct="right",
        why={
            "right": "Alleen de woorden eromheen onderscheiden *book* het werkwoord van *book* het zelfstandig naamwoord. Deze les geeft elk woord een label, zoals de tags van woordsoorten dat doen.",
            "capital": "*Book* heeft een hoofdletter omdat het het eerste woord van de zin is. Wat er een werkwoord van maakt, is *that flight* erachter. Zie [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
            "first": "Veel zinnen beginnen met een zelfstandig naamwoord of een voornaamwoord, zoals *We had a fantastic time.* Hier maakt *that flight* na *Book* er een werkwoord van. Zie [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
            "list": "Dan had *book* in beide zinnen dezelfde tag. Dezelfde letters hadden twee tags, en de woorden eromheen bepaalden welke. Zie [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl).",
        },
    )
    quick_book
    return (quick_book,)


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="Een programma geeft één label aan elk token van *We don't know.* (we weten het niet), in tokens geknipt zoals de Penn Treebank dat doet. Hoeveel labels geeft het?",
        options={
            "spaces": "3: één voor elk stuk tussen de spaties",
            "no_nt": "4: *n't* is geen woord, dus het krijgt geen label",
            "right": "5: *don't* geeft twee tokens",
            "whole": "4: *We*, *don't*, *know* en de punt",
        },
        correct="right",
        why={
            "right": "*We*, *do*, *n't*, *know* en de punt. De Penn Treebank knipt *don't* in *do* en *n't*, en de treebank van deze les doet hetzelfde. Eén label per token betekent evenveel labels als tokens, dus de standaard die de tokens knipt, bepaalt ook hoeveel labels er zijn.",
            "spaces": "Knippen bij de spaties geeft *We*, *don't* en *know.*, met de punt vast aan *know*. De Penn Treebank knipt de punt los, en knipt *don't* in *do* en *n't*. Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "no_nt": "*n't* is een eigen token, en elk token krijgt een label; als de labels namen markeren, krijgt *n't* het label voor geen naam. Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "whole": "De Penn Treebank knipt *don't* in twee tokens, *do* en *n't*, en elk token krijgt een eigen label. Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel vragen over een tekst gaan over stukken ervan, niet over de tekst als geheel. Welke woorden zijn namen van personen, plaatsen en organisaties? Welke woorden in *zet de wekker morgen om zeven uur* zeggen wanneer? Welke woorden in een regel van een adres zijn de straat, en welke de woonplaats? Een spamfilter geeft een heel bericht één label; deze taken geven elk woord een eigen label. De tools voor woordsoorten geven al één label per woord, maar een naam kan uit meer woorden bestaan. Deze les laat zien hoe een stuk tekst van meer woorden één label per woord wordt, en weer terug.

    **Een voorbeeld uit het dagelijks leven.** Je markeert de namen in een brief met een markeerstift. Staan twee namen naast elkaar, zoals een stad en de provincie erachter, dan lijkt één lange gele streep één naam. Een kleine onderbreking aan het begin van de tweede streep laat zien waar de tweede naam begint.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="Universal NER is een dataset waarin mensen elke naam van een persoon, een plaats of een organisatie hebben gemarkeerd. Deze les gebruikt 11.722 van de Engelse zinnen, van het web: reviews, vragen met antwoorden, blogs en nieuwsgroepen. Hoeveel van elke 100 tokens daarin staan buiten elke naam?",
        options={
            "seventy": "Ongeveer 70: het web staat vol namen",
            "eighty-five": "Ongeveer 85",
            "right": "Ongeveer 95",
            "ninety-nine": "Ongeveer 99",
        },
        reveal="**Ongeveer 95:** 187.363 van de 197.019 tokens staan buiten elke naam, 95,1 van elke 100. De andere 9.656 tokens vormen samen 6.429 namen, en 8.089 van de 11.722 zinnen hebben helemaal geen naam. Bijna elk token krijgt hetzelfde label; de paar labels die ertoe doen, markeren waar een naam begint en wat voor naam het is.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén label per token: sequence labeling

    In [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) had elk woord van een treebank een tag, dus een zin van tien tokens had tien tags. Jurafsky en Martin geven zo'n taak een naam: een taak die elk woord $x_i$ van een rij woorden een label $y_i$ geeft, zodat de rij labels $Y$ even lang is als de rij woorden $X$, heet **sequence labeling**. Woordsoorten taggen is sequence labeling, en namen vinden ook, zodra elke naam als labels op de tokens ervan staat. Een label hoort bij een token, zoals [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl) tokens knipte: *don't* is twee tokens, *do* en *n't*, met elk een label.

    ### Namen zijn spans

    Een naam kan één token zijn, zoals *India*, of meer, zoals *San Diego* of *Byron York*. Een **span** is een reeks tokens die in een zin naast elkaar staan. **Universal NER** (Stephen Mayhew en anderen, 2024) markeert namen in 19 datasets van 13 talen, allemaal op zinnen van Universal Dependencies. Voor het Engels markeert Universal NER onder meer de zinnen van EWT, de treebank van de les over woordsoorten, in een eerdere versie. Deze les gebruikt die zinnen zonder de e-mails, want daarin staan de namen en adressen van echte mensen. Elke span heeft een van drie **types**: Mayhew en de anderen kiezen bewust voor drie grove types, persoon (`PER`), organisatie (`ORG`) en plaats (`LOC`). Titels blijven buiten een naam: volgens de richtlijnen van Universal NER hoort een titel nooit bij een naam, dus in *Mr. Robinson* is alleen *Robinson* een persoon.

    Van de 6.429 namen in de 11.722 zinnen zijn er 2.222 langer dan één token, ongeveer 1 op 3; de langste heeft er 16. In Python schrijft de course een span als een tuple `(start, end, kind)`. `start` is de positie van het eerste token, geteld vanaf 0, en `end` de positie net na het laatste, zoals bij een slice: `words[7:9]` is `["San", "Diego"]`, de span `(7, 9, "LOC")`. In een tekst staat een span vaak tussen haken met het type erbij, zoals [LOC San Diego].
    """)
    return


@app.cell(hide_code=True)
def spans_cell():
    mo.Html(C.spans_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_bio():
    mo.md(r"""
    ### BIO: een span als labels

    Om spans als één label per token te schrijven, krijgt elk token een van drie soorten labels: **B** op het eerste token van een span (begin), **I** op de tokens erna in de span (inside, binnen) en **O** op elk token buiten alle spans (outside, buiten). B en I dragen het type mee, zoals in `B-LOC` en `I-LOC`. Jurafsky en Martin beschrijven het zo: een B voor elk token waarmee een span begint, een I voor de tokens binnen een span, en een O voor elk token buiten elke span. Met drie types zijn dat 2 × 3 + 1 = 7 labels. Dit is **BIO tagging**.

    ### Waarom B? Twee namen die elkaar raken

    Eenvoudiger is **IO tagging**: `I-` op elk token van een span, en `O` op de rest. Jurafsky en Martin merken op dat daarbij wat informatie verloren gaat, doordat de B wegvalt. Wat er verloren gaat: twee spans van hetzelfde type die elkaar raken. *Miramar California* wordt `I-LOC I-LOC`, en teruggelezen is dat één plaats van twee tokens, net als *San Diego*. De richtlijnen van Universal NER vragen hier om twee spans: twee plaatsen direct na elkaar krijgen elk een eigen span. Met BIO zegt `B-LOC B-LOC` precies dat.
    """)
    return


@app.cell(hide_code=True)
def bio_cell():
    mo.Html(C.bio_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_back():
    mo.md(r"""
    In de 11.722 zinnen staat een `B-` 33 keer direct na een span van hetzelfde type: 24 keer twee namen, meestal een plaats direct voor de grotere plaats eromheen, zoals [LOC san francisco] [LOC california], en 9 keer iets wat op een vergissing lijkt, zoals *George Bush* getagd als twee personen, *George* en *Bush*. Ook mensen die labels geven, maken fouten.

    ### Andere schema's

    Lance Ramshaw en Mitchell Marcus voerden in 1995 de drie tags in, met een B alleen waar twee spans elkaar raken. Erik Tjong Kim Sang en Jorn Veenstra (1999) noemden dat schema IOB1, en het schema met een B op elk eerste token IOB2; Universal NER gebruikt IOB2, en deze les ook. **BIOES** voegt E toe voor het laatste token van een span en S voor een span van één token. Al deze schema's bevatten dezelfde spans, en een programma kan het ene in het andere omzetten.

    ### Van labels terug naar spans

    Tags teruglezen gaat van links naar rechts. Een `B-` begint een span. Een `I-` direct na een span van hetzelfde type maakt die span één token langer. Een `O` hoort bij geen span. Een **tagger**, een programma dat de labels geeft, kan ook een `I-` geven die nergens op aansluit: na een `O`, of na een span van een ander type. Het evaluatiescript van de CoNLL shared tasks leest zo'n `I-` als het begin van een nieuwe span, en deze les doet dat ook.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **sequence labeling** | een taak die elk token van een rij één label geeft, dus er zijn evenveel labels als tokens |
        | **span** | een reeks tokens die naast elkaar staan, zoals *San Diego*; in de course een tuple `(start, end, kind)` |
        | **Universal NER** | een dataset met namen in 13 talen, op zinnen van Universal Dependencies |
        | **type** | de soort naam van een span: `PER` (een persoon), `LOC` (een plaats) of `ORG` (een organisatie) |
        | **BIO tagging** | labels voor spans: `B-` en het type op het eerste token van een span, `I-` en het type op de tokens erna, `O` op alle andere tokens |
        | **IO tagging** | labels met alleen `I-` en `O`: twee spans van hetzelfde type die elkaar raken, worden één |
        | **IOB1**, **IOB2**, **BIOES** | andere schema's: een B alleen waar twee spans elkaar raken; een B op elk eerste token, zoals hier; en E en S voor het laatste token en voor een span van één token |
        | **tagger** | een programma dat elk token een label geeft |
        | `C.UNER` | de 11.722 Engelse zinnen van Universal NER uit EWT die deze les gebruikt: een tuple van zinnen, elk een tuple van paren `(woord, tag)` |
        | `C.EXAMPLE` | de zin van de afbeeldingen, *Miramar California is a bit north of San Diego.* (Miramar in Californië ligt iets ten noorden van San Diego), als paren `(woord, tag)` |
        | `["O"] * n` | een lijst met `n` keer `"O"`: `["O"] * 3` is `["O", "O", "O"]` |
        | `spans[-1]` | het laatste item van de lijst `spans`; `spans[-1] = x` zet `x` op die plek |
        """),
        L.watch_out("In andere lessen van de course, over hoe je stap voor stap volgt wat een AI-toepassing doet, is een *span* één stap van een trace, zoals één aanroep van een model. Hier is een span een reeks tokens in een zin: hetzelfde woord voor iets anders."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Van spans naar tags: begin met `O` voor elk token. Bij elke span `(start, end, kind)` krijgt het token op `start` `B-` en het type, en krijgen de tokens erna, tot `end` en zonder `end`, `I-` en het type.
    2. Van tags naar spans: ga de tags van links naar rechts langs. Bij een `B-` begin je een nieuwe span van één token. Bij een `I-` kijk je naar de laatste span. Eindigt de laatste span direct voor dit token, met hetzelfde type, dan wordt de span één token langer; anders begin je een nieuwe span. Bij een `O` doe je niets.
    3. Controleer beide richtingen tegelijk: de spans die je uit je tags terugleest, zijn de spans waarmee je begon.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt IO-tags voor de spans van `C.EXAMPLE`, en print elk woord met de tag in Universal NER en de IO-tag.
    """)
    return


@app.cell
def example_io():
    def io_tags(n, spans):
        tags = ["O"] * n
        for start, end, kind in spans:
            for i in range(start, end):
                tags[i] = "I-" + kind
        return tags


    example_spans = [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")]
    tags_io = io_tags(len(C.EXAMPLE), example_spans)
    for i in range(len(C.EXAMPLE)):
        word, tag = C.EXAMPLE[i]
        print(word, tag, tags_io[i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Universal NER heeft `B-LOC` op *Miramar* en op *California*: twee plaatsen. De IO-tags geven beide `I-LOC`, net als *San* en *Diego*, samen één plaats. Oefening 1 maakt de BIO-tags.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. to_tags: van spans naar BIO-tags

        Schrijf `to_tags(n, spans)`: geef de lijst van de `n` tags terug van een zin van `n` tokens met deze `spans`, elk een tuple `(start, end, kind)`. Begin met `n` keer `"O"`. Bij elke span krijgt het token op `start` `"B-"` en het type; de tokens erna, tot `end` en zonder `end`, krijgen `"I-"` en het type. Vervang de `...`. De check roept `to_tags(10, [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")])` aan, de spans van het voorbeeld, wat `['B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']` geeft; daarna eigen gevallen: nog twee zinnen van Universal NER, geen spans, een span van drie tokens, een span tot het laatste token, spans in een andere volgorde, en één token.
        """),
        L.task("Geef het eerste token van elke span een B-, de andere tokens van de span een I-, en elk ander token een O."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("to_tags", lines=7)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("to_tags", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "to_tags",
        [
            "Begin zoals het uitgewerkte voorbeeld, met `[\"O\"] * n` en een loop over de spans. Daarin krijgt het token op `start` een eigen regel, en doet een tweede loop over `i` in een `range` tot `end` de tokens erna.",
            "Op welke positie begint de tweede loop: op `start`, of verder? Welk token heeft al een tag?",
            "`        for i in range(___, end):`",
        ],
        why="`tags = [\"O\"] * n`, dan `for start, end, kind in spans:` met daaronder `tags[start] = \"B-\" + kind`, en `for i in range(start + 1, end):` met daaronder `tags[i] = \"I-\" + kind`; na de loop `return tags`. `to_tags(10, [(0, 1, \"LOC\"), (1, 2, \"LOC\"), (7, 9, \"LOC\")])` geeft `['B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op alle 11.722 zinnen van Universal NER, met de spans die de course uit de tags leest. Komt elke zin goed uit, dan bevatten de spans en de tags dezelfde informatie: onderweg gaat niets verloren. Oefening 2 gaat de andere kant op.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spans():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. to_spans: van tags terug naar spans

        Schrijf `to_spans(tags)`: geef de lijst terug van de spans `(start, end, kind)` die `tags` markeert, van links naar rechts. Begin met een lege lijst `spans`, en ga de posities `i` van `tags` langs. Bij een tag die met `B` begint, voeg je een nieuwe span van één token toe, met het type `tag[2:]`: de tag zonder de eerste twee tekens. Bij een tag die met `I` begint, kijk je naar de laatste span, `spans[-1]`: is er een laatste span die direct voor dit token eindigt en hetzelfde type heeft, zet dan op die plek een span die één token langer is; anders voeg je een nieuwe span van één token toe, zoals het CoNLL-script doet. Bij een `O` gebeurt er niets. Vervang de `...`. De check roept `to_spans(["B-LOC", "B-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"])` aan, de tags van het voorbeeld, wat `[(0, 1, 'LOC'), (1, 2, 'LOC'), (7, 9, 'LOC')]` geeft; daarna eigen gevallen: nog twee zinnen, de tags van een tagger met een `I-` na een `O`, een `I-` van een ander type, geen namen, geen tokens, een `I-` aan het begin, een span van vier tokens, en twee personen die elkaar raken.
        """),
        L.task("Lees de tags van links naar rechts, en geef de spans terug die de tags markeren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spans():
    box_spans = L.code_box("to_spans", lines=11)
    box_spans
    return (box_spans,)


@app.cell(hide_code=True)
def check_spans(box_spans):
    result_spans = L.check("to_spans", box_spans.value)
    result_spans
    return (result_spans,)


@app.cell(hide_code=True)
def hints_spans():
    L.hints(
        "to_spans",
        [
            "Eén loop over de posities `i`, met `if` en `elif` op de eerste letter van de tag. In de `I`-tak stelt een tweede `if` drie vragen over de laatste span, verbonden met `and`: is er een (`len(spans) > 0`), waar eindigt de span (`spans[-1][1]`), en welk type heeft de span (`spans[-1][2]`)?",
            "Een span `(start, end, kind)` die direct voor token `i` eindigt: welk getal is dan `end`?",
            "`            if len(spans) > 0 and spans[-1][1] == ___ and spans[-1][2] == tag[2:]:`",
        ],
        why="`spans = []`, dan `for i in range(len(tags)):` met `tag = tags[i]`; onder `if tag[0] == \"B\":` komt `spans.append((i, i + 1, tag[2:]))`, en onder `elif tag[0] == \"I\":` komt `if len(spans) > 0 and spans[-1][1] == i and spans[-1][2] == tag[2:]:` met daaronder `spans[-1] = (spans[-1][0], i + 1, tag[2:])`, en `else:` met `spans.append((i, i + 1, tag[2:]))`; na de loop `return spans`. `to_spans` van het voorbeeld geeft `[(0, 1, 'LOC'), (1, 2, 'LOC'), (7, 9, 'LOC')]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_spans():
    mo.md(r"""
    In de afbeelding leest je functie de tags van alle 11.722 zinnen: 6.429 namen, waarvan 2.222 langer dan één token. Op de tags van de tagger komt *Miramar California* terug als één plaats, en *Diego* als een eigen plaats: de tagger maakte twee fouten, en de spans laten beide zien. Een latere les geeft een tagger een score door zulke spans te tellen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: één woord, meer dan één type

    Typ een Engels woord, en zie hoe vaak het in Universal NER deel is van een persoon, een plaats of een organisatie, met voorbeelden. Probeer *Washington*, *US*, *Kennedy*, *Apple* of *Chicago*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="Washington", label="Een woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 3. Wanneer raakt IO een span kwijt?

    Elke zin hieronder komt uit Universal NER, met de spans tussen haken. Bedenk de IO-tags van elke zin, en lees die tags terug: krijg je dezelfde spans?
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_io = L.sort("io-or-bio")
    sort_io
    return (sort_io,)


@app.cell(hide_code=True)
def check_sort(sort_io):
    result_sort = L.check("io-or-bio", sort_io)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 4. Eén naam, meer dan één type

    In de demo staat *Washington* 15 keer in de span van een plaats, en 5 keer in de span van een organisatie, meestal een krant zoals de *Washington Post*.
    """)
    return


@app.cell(hide_code=True)
def choice_kinds_cell():
    choice_kinds = L.choice(
        "kinds",
        question="Wat betekent dat voor een programma dat namen moet vinden?",
        options={
            "list": "Een lijst met namen, elk met een type, is genoeg, als de lijst maar lang is",
            "right": "De woorden rond een naam moeten helpen het type te bepalen",
            "rare": "Dat komt zelden voor, dus een programma kan zulke namen overslaan",
            "slips": "De mensen die de zinnen labelden, maakten fouten, want een naam heeft altijd hetzelfde type",
        },
        correct="right",
        why={
            "right": "Net als bij *book* in *Woordsoorten en woordgroepen* kunnen dezelfde letters voor verschillende dingen staan, en de woorden eromheen beslissen: *in Washington* is een plaats, *the Washington Post* een krant. Een lijst met namen helpt, maar kan het type niet zonder de woorden eromheen kiezen. Latere lessen bouwen programma's die leren van de woorden rond een naam.",
            "list": "Een lijst zou *Washington* één type moeten geven, en in deze zinnen is het 15 keer een plaats en 5 keer deel van een organisatie. *US* is 161 keer een plaats en 22 keer een organisatie. Welk woord in *the Washington Post* zegt je dat het niet om de stad gaat?",
            "rare": "*US* alleen al is 161 keer een plaats en 22 keer een organisatie, en ook *Chicago*, *Apple* en *Kennedy* hebben meer dan één type. Probeer deze woorden in de demo: wat zeggen de woorden eromheen?",
            "slips": "Sommige labels zijn vergissingen, zoals *George Bush* getagd als twee personen. Maar de stad *Washington* en de krant de *Washington Post* zijn verschillende dingen met een woord gemeen. Wat onderscheidt de twee in een zin?",
        },
    )
    choice_kinds
    return (choice_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woordsoorten en woordgroepen](/course/open/nlp-word-classes?lang=nl) gaf elk token een tag: dat is sequence labeling. Een naam kan uit meer tokens bestaan, dus schreef de course namen als spans `(start, end, kind)`, en BIO tagging maakt van spans één label per token: `B-` op het eerste token, `I-` op de tokens erna, `O` erbuiten. Je schreef beide richtingen, en op alle 11.722 zinnen van Universal NER gaven de twee functies terug waarmee je begon. IO tagging is eenvoudiger, maar maakt van twee spans van hetzelfde type die elkaar raken er één, zoals bij *Miramar California*. De volgende les, *Woordsoorten taggen*, bouwt een programma dat elk woord een woordsoort geeft, en laat zien dat je met de vaakst voorkomende tag van elk woord de meeste woorden al goed hebt.

    **Een voorbeeld uit het dagelijks leven.** In een theater is elke stoel bezet of vrij. Twee gezinnen die vlak naast elkaar zitten, lijken één groot gezin, tenzij elke reservering de eerste stoel markeert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_kinds, result_sort, result_spans, result_tags):
    L.wrap_up(result_tags, result_spans, result_sort, choice_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
