# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-spelling. Route: course/route/lessen/taal-en-zoeken.yaml. Controles: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-spelling", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de controles


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl), en een over [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="Als invoegen, weghalen en vervangen elk 1 kosten, wat is dan de edit distance van *hte* naar *het*?",
        options={
            "one": "1: een verwisseling",
            "right": "2",
            "three": "3: een bewerking voor elke letter",
        },
        correct="right",
        why={
            "right": "Vervang de *t* door een *e* en de *e* door een *t*, of haal de *t* weg en voeg na de *e* een *t* in: hoe dan ook twee bewerkingen. Een verwisseling van twee buurletters kost twee van deze bewerkingen. Deze les voegt de verwisseling toe als vierde soort bewerking, zodat *hte* op één bewerking van *het* ligt.",
            "one": "Een verwisseling is geen van de drie bewerkingen: met invoegen, weghalen en vervangen kost het verwisselen van twee buurletters er twee. Deze les voegt de verwisseling toe als vierde soort bewerking; zie *Andere kosten en verwisselingen* in [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
            "three": "De *h* blijft staan, dus alleen de *t* en de *e* hebben een bewerking nodig: twee. Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def quick_check_corpus():
    quick_corpus = L.choice(
        "quick-corpus",
        question="Wat is een corpus?",
        options={
            "web": "Alle pagina's van het web, zonder keuze",
            "dictionary": "Een lijst van de woorden van een taal, met de betekenis van elk woord",
            "right": "Teksten gekozen om een taal te vertegenwoordigen",
        },
        correct="right",
        why={
            "right": "Een corpus is gekozen: het CGN bevat bijvoorbeeld bijna 9 miljoen woorden gesproken Nederlands. Deze les telt hoe vaak elk woord voorkomt in een verzameling Nederlandse zinnen, en laat die aantallen beslissen voor welk woord een tikfout waarschijnlijk staat.",
            "web": "Niemand koos de pagina's van het web om een taal te vertegenwoordigen, dus het web is in die zin geen corpus. Zie [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
            "dictionary": "Een lijst woorden met de betekenissen is een woordenboek, of een lexicale database zoals WordNet. Een corpus bevat doorlopende tekst, zodat je kunt tellen hoe vaak elk woord erin voorkomt. Zie [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
        },
    )
    quick_corpus
    return (quick_corpus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Mensen maken tikfouten, en vaker dan je denkt. Daniel Jurafsky en James Martin noemen in de bijlage *Spelling Correction and the Noisy Channel* van *Speech and Language Processing* schattingen van 1 tot 2% fouten als mensen gedrukte tekst zorgvuldig overtikken, tot 10 tot 15% in zoekvragen op het web. Een zoekmachine, het toetsenbord van je telefoon en een tekstverwerker stellen allemaal het woord voor dat je bedoelde. Peter Norvig schreef het essay *How to Write a Spelling Corrector* na een week in 2007 waarin twee vrienden, Dean en Bill, los van elkaar vertelden hoe verbaasd ze waren over de spellingcorrectie van Google. Deze les bouwt in eigen code een **spellingcorrector** naar het voorbeeld van Norvig: de corrector zoekt de woorden dicht bij een tikfout met de bewerkingen van [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl), en laat de aantallen van een corpus, zoals in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl), het woord kiezen dat het waarschijnlijkst bedoeld was.

    **Een voorbeeld uit het dagelijks leven.** Een tekstverwerker die *eigelijk* onderstreept en *eigenlijk* voorstelt, heeft een woord gevonden dat op één bewerking ligt van wat je typte.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De course nam 500 woorden van minstens vier letters, op willekeurige plekken in de Nederlandse zinnen van Tatoeba, en maakte in elk woord één tikfout, op een willekeurige plek: een letter weggehaald, ingevoegd of vervangen, of twee buurletters verwisseld. Hoeveel van de 500 tikfouten verbetert de corrector van deze les?",
        options={
            "few": "Zo'n 20%",
            "half": "Zo'n 50%",
            "right": "Zo'n 80%",
            "all": "Bijna 100%",
        },
        reveal="**414 van de 500**: 82,8%. Van de rest werden er 44 een ander woord, zoals *dze*, bedoeld als *deze*, dat *de* werd; en 42 tikfouten waren al een woord, zoals *vel* voor *veel*, dat de corrector niet verandert. In oefening 3 gaat je eigen corrector aan de slag met dezelfde 500 tikfouten.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Elke string op één bewerking

    De corrector van Norvig begint bij de tikfout, niet bij een woordenlijst: de corrector maakt elke string die op één bewerking van de tikfout ligt, en kijkt dan welke daarvan een woord zijn. In het deel *How It Works: Some Python* noemt het essay vier soorten eenvoudige bewerkingen: een letter weghalen, twee buurletters verwisselen, een letter vervangen door een andere, en een letter invoegen. Dat zijn de drie bewerkingen van de edit distance, plus de verwisseling, die hier als één bewerking telt. Deze vier bewerkingen zijn oud: de handleiding van SPELL, het programma van Ralph Gorin op Stanford, noemt in de versie van december 1974 dezelfde vier soorten fouten: één verkeerde letter, één ontbrekende letter, één letter te veel, en twee verwisselde letters.

    Knip het woord op elke plek, van vóór de eerste letter tot na de laatste, en maak op die plek elke bewerking. Met de 26 letters *a* tot en met *z* telt Norvig voor een woord van n letters: n keer weghalen, n − 1 verwisselingen, 26n keer vervangen en 26(n + 1) keer invoegen, samen 54n + 25 strings, waarvan er meestal een paar dubbel zijn. *hius* heeft vier letters, dus 4 + 3 + 104 + 130 = 241 strings, waarvan 234 verschillend: een letter vervangen door dezelfde letter geeft vier keer *hius* terug, en een letter invoegen naast dezelfde letter geeft van beide kanten dezelfde string: *hhius*, *hiius*, enzovoort. Twee bewerkingen bereiken veel meer strings: één bewerking vanaf elk van die 234 strings geeft 24.254 verschillende strings. Volgens Norvig zitten daar meestal maar een paar bekende woorden tussen.

    ### Houd de woorden over, kies het vaakste

    De meeste van de 234 strings zijn geen woord. Vier zijn een woord in de Nederlandse zinnen van Tatoeba: *huis*, 1.800 keer, *heus*, 11 keer, en *hijs* en *hits*, elk 1 keer. Een **bekend woord** is een woord dat in het corpus staat, en deze vier bekende woorden zijn de **kandidaten**: de woorden die de schrijver bedoeld kan hebben. Norvig kiest de kandidaat met de grootste kans, en splitst die kans in twee delen: een language model P(c), de kans dat c als woord in een tekst voorkomt, en een **error model** P(w|c), de kans dat iemand w typt terwijl c bedoeld was. Hier is w het getypte woord en c een kandidaat. Hoe de twee delen samengaan, is het onderwerp van de les *De regel van Bayes*, in de kern van de course.

    Het language model van deze les is het eenvoudigste dat er is: hoe vaak elk woord in het corpus voorkomt. De zinnen van Tatoeba bevatten 1.301.039 woorden, waarvan 38.682 verschillend als je in kleine letters telt. Hoe vaker een woord voorkomt, hoe waarschijnlijker een schrijver het bedoelde, dus *huis* wint. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze geven dezelfde regel in paragraaf 3.3.1 van *Introduction to Information Retrieval*: de eenvoudigste manier om te zeggen welk woord gewoner is, is tellen hoe vaak een term in de verzameling voorkomt, dus als *grunt* vaker voorkomt dan *grant*, wordt *grunt* de correctie.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Eén bewerking vóór twee

    Norvig begon in 2007 in een vliegtuig aan het programma en had geen gegevens over tikfouten voor een error model. Het essay neemt een kortere weg, met wat Norvig zelf een triviaal, gebrekkig error model noemt: elk bekend woord op één bewerking is oneindig veel waarschijnlijker dan elk bekend woord op twee bewerkingen, en oneindig veel minder waarschijnlijk dan het getypte woord zelf, als dat een bekend woord is. Met andere woorden: een getypt woord dat een bekend woord is, blijft staan; elk bekend woord op één bewerking wint van elk woord op twee bewerkingen; en alleen tussen woorden op dezelfde afstand beslist het aantal. Binnen twee bewerkingen van *hius* liggen 105 woorden uit de zinnen, waaronder *is*, dat 36.467 keer voorkomt. Als alle 105 meededen, won *is*; eerst op één bewerking kijken houdt *huis* over.

    ### Het noisy channel

    Jurafsky en Martin beschrijven dezelfde twee delen als het **noisy channel model**: je behandelt een verkeerd gespeld woord alsof een goed gespeld woord vervormd is doordat het door een kanaal met ruis ging. De kans dat een bedoeld woord als het getypte woord uit het kanaal komt, noemen Jurafsky en Martin het channel model, en de kans op het bedoelde woord zelf de prior; Jurafsky en Martin schrijven x voor het getypte woord en w voor het bedoelde, waar Norvig w en c schrijft. Mark Kernighan, Kenneth Church en William Gale beschreven in 1990 zo'n corrector, gebouwd bij AT&T Bell Laboratories met aantallen uit de nieuwsberichten van het Associated Press van 1988: 44 miljoen woorden. Op 329 tikfouten met elk twee kandidaten kwam alleen het vaakste woord in 75% van de gevallen overeen met de meerderheid van drie beoordelaars, de aantallen en het channel model samen in 87%, en gewoon de eerste kandidaat in 52%.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **spellingcorrector** | een programma dat bij een getypt woord het woord geeft dat het waarschijnlijkst bedoeld was |
        | **bewerking** | in deze les: weghalen, invoegen, vervangen, of twee buurletters verwisselen (een verwisseling), elk één bewerking |
        | **bekend woord** | een woord dat in het corpus staat: een key van de dictionary met aantallen |
        | **kandidaat** | een bekend woord dat de schrijver bedoeld kan hebben, zoals *huis* en *heus* bij *hius* |
        | **error model** | P(w\|c): hoe waarschijnlijk iemand die het woord c bedoelt, w typt |
        | **noisy channel model** | een tikfout gezien als een bedoeld woord dat onderweg door een kanaal veranderd is: de kans op de verandering, maal de kans op het woord |
        | **non-word error** | een tikfout die geen woord is, zoals *boec* voor *boek* |
        | **real-word error** | een tikfout die zelf een woord is, zoals *kat* voor *kast* |
        | `for c in letters` | gaat de letters van de string `letters` één voor één langs, zoals `for` een lijst langsgaat |
        """),
        L.watch_out("De corrector kent alleen het corpus. Een woord dat niet in het corpus staat, hoeft geen tikfout te zijn: namen en zeldzame woorden ontbreken, en de corrector maakt er een woord van dat vaker voorkomt. En een woord dat in het corpus staat, is niet altijd goed gespeld: oude boeken gebruiken oude spellingen, en de zinnen van Tatoeba bevatten naast *bibliotheek* ook *biblotheek*. Norvig schreef in 2009 dat het internet vol slechte spelling staat."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Staat het getypte woord in het corpus, geef het dan terug zoals het is.
    2. Maak elke string op één bewerking: op elke plek weghalen en verwisselen, en voor elke letter invoegen en vervangen.
    3. Houd de strings over die een woord in het corpus zijn. Zijn die er, geef dan het woord terug dat het vaakst voorkomt.
    4. Zo niet, doe hetzelfde met de strings op twee bewerkingen.
    5. Vindt ook dat niets, geef dan het getypte woord terug.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder knipt *thier*, een tikfout voor het Engelse *their*, op plek 2, in de eerste twee letters en de rest, en maakt daar de vier soorten bewerkingen, met de letter *a* voor invoegen en vervangen.
    """)
    return


@app.cell
def example_cell():
    word = "thier"
    start = word[:2]
    rest = word[2:]
    print(start, rest)
    print(start + rest[1:])                       # weghalen
    print(start + rest[1] + rest[0] + rest[2:])   # verwisselen
    print(start + "a" + rest)                     # invoegen
    print(start + "a" + rest[1:])                 # vervangen
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `rest[1:]` is `rest` zonder de eerste letter, dus weghalen haalt de *i* eruit. Verwisselen zet de tweede letter van `rest` vóór de eerste, en geeft *their*: het bedoelde woord ligt één verwisseling verderop. Invoegen zet de *a* tussen `start` en `rest`, en vervangen zet de *a* op de plaats van de eerste letter van `rest`. Op de laatste plek, na de *r*, is `rest` leeg: dan kan alleen invoegen, en verwisselen heeft minstens twee letters in `rest` nodig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_near():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. near_words: de woorden op één bewerking

        Schrijf `near_words(word, counts, letters)`. Knip `word` op elke plek, van vóór de eerste letter tot na de laatste, in `start` en `rest`, zoals in het uitgewerkte voorbeeld, en bewaar elke string die je maakt in een lijst: op elke plek weghalen en verwisselen, als `rest` genoeg letters heeft, en voor elke letter van `letters` invoegen en vervangen. Geef daarna de strings terug die een key van `counts` zijn, gesorteerd, elk één keer. Voor `near_words("hius", dutch, LETTERS)` is de uitkomst `["heus", "hijs", "hits", "huis"]`. Vervang de `...`. De check roept dat aan en `near_words("thier", english, LETTERS)`, en daarna eigen gevallen: *teh*; *acress*, met woorden bij alle vier soorten bewerkingen; *whal*, dat een letter aan het eind nodig heeft; een letter te veel vooraan; *korrectud*, zonder woord op één bewerking; Nederlandse tikfouten; en een kleine dictionary met alleen de letters `"ab"`, ook voor een leeg woord.
        """),
        L.task("Maak elke string op één bewerking van een woord, en houd de strings over die een woord in het corpus zijn."),
    ])
    return


@app.cell
def counts_cell():
    english = C.word_counts("english")
    dutch = C.word_counts("tatoeba")
    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    print(len(english), english["their"], english["thief"])
    print(len(dutch), dutch["huis"])
    return LETTERS, dutch, english


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    `dutch` bevat hoe vaak elk woord voorkomt in de Nederlandse zinnen van Tatoeba, een verzameling zinnen die vrijwilligers schrijven en vertalen, in kleine letters: 38.682 verschillende woorden, met *huis* 1.800 keer. `english` doet hetzelfde voor de vijf Engelse boeken van de course: 23.131 verschillende woorden, met *their* 1.411 keer.
    """)
    return


@app.cell(hide_code=True)
def exercise_near():
    box_near = L.code_box("near-words", lines=14)
    box_near
    return (box_near,)


@app.cell(hide_code=True)
def check_near(box_near):
    result_near = L.check("near-words", box_near.value)
    result_near
    return (result_near,)


@app.cell(hide_code=True)
def hints_near():
    L.hints(
        "near-words",
        [
            "Begin met `found = []` en ga de plekken langs met `for i in range(len(word) + 1)`. Zet op elke plek `start = word[:i]` en `rest = word[i:]`, voeg weghalen toe als `rest` een letter heeft en verwisselen als `rest` er twee heeft, en ga met `for c in letters` de letters langs om invoegen toe te voegen en, als `rest` een letter heeft, vervangen. Maak na de loop een lijst van de strings `w` uit `found` die een woord zijn, en geef `sorted(set(...))` van die lijst terug.",
            "Met welk woord vraag je of een key in een dictionary staat?",
            r"`    return sorted(set([w for w in found if ___]))`",
        ],
        why=r"`found = []`; `for i in range(len(word) + 1):` met `start = word[:i]` en `rest = word[i:]`; als `len(rest) > 0`, voeg `start + rest[1:]` toe; als `len(rest) > 1`, voeg `start + rest[1] + rest[0] + rest[2:]` toe; `for c in letters:` voeg `start + c + rest` toe, en als `len(rest) > 0` ook `start + c + rest[1:]`; na de loop `return sorted(set([w for w in found if w in counts]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_near():
    mo.md(r"""
    Bij *keuke* vindt je functie *keuken*, met een *n* aan het eind: dus de plek na de laatste letter telt ook mee. Bij *scool* vindt je functie *school*, maar ook *cool*, dat als Engels leenwoord in de zinnen staat. En bij *acress*, het voorbeeld van Jurafsky en Martin, vindt je functie in de Engelse boeken vijf woorden, met alle vier soorten bewerkingen: *access*, *acres*, *across*, *actress* en *caress*. Welk woord bedoelde de schrijver? Oefening 2 kiest er een.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. most_common: het woord dat het vaakst voorkomt

        Schrijf `most_common(words, counts)`: ga `words` langs, bewaar het woord met het hoogste aantal tot nu toe samen met dat aantal, en geef het woord terug. Zonder woorden geef je `None` terug. Voor `most_common(["heus", "hijs", "hits", "huis"], dutch)` is de uitkomst `"huis"`. Vervang de `...`. De check roept dat aan en `most_common(["their", "thief", "tier"], english)`, en daarna eigen gevallen: de woorden op één bewerking van andere tikfouten, met het woord dat het vaakst voorkomt vooraan, in het midden of achteraan; een lijst van één woord; een lege lijst; en de kleine dictionary. Geen lijst heeft twee woorden met hetzelfde hoogste aantal.
        """),
        L.task("Geef het woord uit een lijst terug dat het vaakst in het corpus voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("most-common", lines=8)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("most-common", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "most-common",
        [
            "Begin met `best = None` en `best_count = 0`, en ga de woorden langs met `for w in words:`. Is het aantal van `w` hoger dan `best_count`, zet dan `best` op `w` en `best_count` op dat aantal. Geef na de loop `best` terug.",
            "Waar vind je hoe vaak het woord `w` voorkomt?",
            r"`        if ___ > best_count:`",
        ],
        why=r"`best = None` en `best_count = 0`; `for w in words:` en daarin `if counts[w] > best_count:` met `best = w` en `best_count = counts[w]`; na de loop `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Bij *wear* kiest de corrector *waar*, 3.231 keer in de zinnen, en niet *weer*, 895 keer: ook als je *Het wear is mooi* typte. In de Engelse boeken kiezen de aantallen bij *acress* het woord *across*: 96 keer, tegen 5 keer *acres*. Het noisy channel model van Jurafsky en Martin, dat ook telt hoe vaak mensen letters verwarren, kiest ook *across*, met *actress* als tweede. In hun voorbeeldzin gaat het om een actrice, en de woorden eromheen maken volgens Jurafsky en Martin duidelijk dat *actress* bedoeld was en niet *across*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_correct():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. correct: de corrector

        Schrijf `correct(word, counts)` met de stappen van het recept: geef `word` zelf terug als het in `counts` staat; anders het woord op één bewerking dat het vaakst voorkomt, als er zo'n woord is; anders het woord op twee bewerkingen dat het vaakst voorkomt, als er zo'n woord is; anders `word` zelf. De check geeft je functie de `near_words` en `most_common` van de course, de letters `LETTERS`, en `far_words`, die een woord en de aantallen krijgt en de bekende woorden binnen twee bewerkingen teruggeeft, gesorteerd, elk één keer. Twee bewerkingen bereiken ook elk woord op één bewerking, en het woord zelf, dus `far_words` geeft ook die woorden. Voor `correct("hius", dutch)` is de uitkomst `"huis"`. Vervang de `...`. De check roept dat aan en `correct("thier", english)`, en daarna eigen gevallen: Engelse tikfouten van elke soort; *kat*, dat al een woord is; *gegan*, op één bewerking van een woord; *mensch*, op twee bewerkingen; en *fietsenstaling*, waarvan het woord *fietsenstalling* niet in de zinnen staat.
        """),
        L.task("Zet de stappen samen: houd een bekend woord, kijk dan op één bewerking, dan op twee, en geef anders op."),
    ])
    return


@app.cell
def course_cell(LETTERS, english):
    near_words = C.near_words
    most_common = C.most_common
    far_words = C.far_words
    print(near_words("korrectud", english, LETTERS))
    print(far_words("korrectud", english))
    return


@app.cell(hide_code=True)
def exercise_correct():
    box_correct = L.code_box("correct", lines=10)
    box_correct
    return (box_correct,)


@app.cell(hide_code=True)
def check_correct(box_correct):
    result_correct = L.check("correct", box_correct.value)
    result_correct
    return (result_correct,)


@app.cell(hide_code=True)
def hints_correct():
    L.hints(
        "correct",
        [
            "Vier stappen in de volgorde van het recept, elk met een eigen `return`. Bewaar de woorden op één bewerking in een lijst `near`, met `near_words`, en de woorden op twee bewerkingen in een lijst `far`, met `far_words`, die één waarde minder krijgt dan `near_words`.",
            "*mensch* heeft in de zinnen niets op één bewerking. Wat geeft `most_common` terug voor een lege lijst, en hoe vraag je of er woorden in `near` staan?",
            r"`    if ___:`",
        ],
        why=r"`if word in counts:` `return word`; `near = near_words(word, counts, LETTERS)`; `if len(near) > 0:` `return most_common(near, counts)`; `far = far_words(word, counts)`; `if len(far) > 0:` `return most_common(far, counts)`; en aan het eind `return word`.",
    )
    return


@app.cell(hide_code=True)
def text_after_correct():
    mo.md(r"""
    ### Hoe goed is de corrector?

    In de afbeelding verbetert je corrector bijna de hele zin, maar maakt *wekend* tot *bekend*, niet *weekend*: allebei liggen op één bewerking, en *bekend* staat 161 keer in de zinnen, *weekend* 109 keer. Norvig nam zich voor om een eenvoudige corrector te schrijven die 80 of 90% goed doet, minstens 10 woorden per seconde verwerkt, en in ongeveer een halve pagina code past. Norvig testte de corrector van het essay op tikfouten uit het Birkbeck-corpus van spelfouten van Roger Mitton: 75% goed op 270 tikfouten die Norvig tijdens het schrijven bekeek, en 68% goed op 400 die tot het eind apart bleven, met 41 en 35 woorden per seconde. Het deel *Evaluation* concludeert dat Norvig de doelen haalde voor korte code, ontwikkeltijd en snelheid, maar niet voor het percentage goed.

    Het deel *Future Work* noemt de oorzaken. Een deel van de bedoelde woorden staat helemaal niet in het corpus, 15 van de 270 en 43 van de 400, dus de corrector kan die woorden niet vinden. Het error model is te simpel: *adres* wordt *acres*, op één bewerking, terwijl het Engelse *address* bedoeld was, en de corrector van de course doet hetzelfde met de vijf Engelse boeken. Een deel van de tikfouten ligt verder dan twee bewerkingen van het woord: 3 van de 270 en 23 van de 400. Als vierde punt noemt Norvig de beste verbetering: meer context bekijken. Norvig verwacht dat een spellingcorrector die 90% goed doet, de woorden eromheen moet gebruiken om te kiezen.

    ### Tikfouten die een woord zijn

    Jurafsky en Martin onderscheiden twee taken. Bij non-word errors levert de fout een niet-bestaand woord op, zoals *graffe* voor *giraffe*. Bij real-word errors levert de fout per ongeluk een echt woord op. Volgens een aantal onderzoeken, die Jurafsky en Martin aanhalen via Kukich (1992), is 25 tot 40% van de spelfouten een geldig Engels woord. Manning, Raghavan en Schütze geven in paragraaf 3.3.2 een voorbeeld uit zoekmachines: in de zoekvraag *flew form Heathrow* staat *form* voor *from*, maar een corrector die elk woord los bekijkt, ziet dat niet, want elk woord is op zichzelf goed gespeld.

    Een bekende real-word error in het Nederlands is de d/t-fout. Frans Daems van de Universiteit Antwerpen noemt in 2003 vormen die hetzelfde klinken maar anders gespeld worden, zoals *houd* en *houdt*, of *betaalt* en *betaald*, en schrijft dat de kans dat we een vorm opschrijven groter wordt naarmate we de vorm vaker gebruikt hebben, ook als dat een fout oplevert. De Antwerpse onderzoekers Verhaert en Sandra schrijven in 2015 dat d/t-fouten bijna alleen voorkomen bij vormen die je niet kunt spellen op wat je hoort, en dat de vorm die het vaakst voorkomt, het vaakst in de plaats van de goede vorm terechtkomt. Een corrector die één woord tegelijk bekijkt, ziet zo'n fout niet: beide vormen zijn een woord. Om een real-word error te vinden, moet een corrector naar de woorden eromheen kijken.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen tikfout

    Typ een woord en kies een corpus: de vijf Engelse boeken, de Nederlandse zinnen van Tatoeba, of de drie Nederlandse boeken, die gedrukt zijn in een spelling van vóór 1934. De afbeelding laat de bekende woorden op één en op twee bewerkingen zien, met hoe vaak elk voorkomt, en het woord dat de corrector kiest. Probeer een tikfout van jezelf, een naam, of een woord in een oude spelling.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="spellnig", label="Je woord")
    demo_corpus = mo.ui.dropdown(options=C.corpus_menu(L.lang), value="De Nederlandse zinnen van Tatoeba", label="Corpus")
    mo.vstack([demo_word, demo_corpus])
    return demo_corpus, demo_word


@app.cell(hide_code=True)
def demo_look(demo_corpus, demo_word):
    mo.Html(C.demo_picture(demo_word.value, demo_corpus.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Norvig schreef in 2009, in het hoofdstuk *Natural Language Corpus Data* van het boek *Beautiful Data*, dat je voor een andere taal alleen een groot corpus in die taal nodig hebt: de code blijft hetzelfde. Dat zie je hier: dezelfde corrector werkt voor het Nederlands met de zinnen van Tatoeba, en maakt van *mensch* het woord *mensen*. De drie Nederlandse boeken geven juist oude spellingen. Kies de boeken en typ *mens*: de boeken schrijven *mensch*, twee bewerkingen verderop, en *mens* zelf staat er niet in, dus de corrector kiest *men*, op één bewerking. Volgens Onze Taal maakte minister van Onderwijs Marchant in 1934 een eenvoudiger spelling verplicht in het onderwijs, waarin de *-ch* aan het eind van een woord als *mensch* verdween. Daarna stelde een Vlaams-Nederlandse commissie een nieuwe spelling op, die in Vlaanderen in 1946 werd ingevoerd en in Nederland in 1947, en in 1954 verscheen het eerste Groene Boekje, de woordenlijst die veertig jaar lang de officiële spelling voorschreef.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Wat doet de corrector?

    Elke vetgedrukte regel is een tikfout en het woord dat bedoeld was. Zet de regel bij wat de corrector van deze les ermee doet, met de Nederlandse zinnen van Tatoeba als corpus.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_happens = L.sort("what-happens")
    sort_happens
    return (sort_happens,)


@app.cell(hide_code=True)
def check_sort(sort_happens):
    result_sort = L.check("what-happens", sort_happens)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl) mat hoe ver twee strings uit elkaar liggen; deze les draaide dat om en maakte elke string op één of twee bewerkingen van een tikfout, met de verwisseling als vierde soort bewerking. De aantallen van een corpus, zoals in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl), dienden als het language model dat het waarschijnlijkste woord kiest, en een kortere weg voor het error model zette één bewerking vóór twee. De corrector bekijkt één woord tegelijk, dus tikfouten die een woord zijn, glippen erdoor: daarvoor is een language model over meer woorden nodig. De volgende les van het domein, *Welke taal is dit?*, telt letters in plaats van woorden: elke taal heeft eigen reeksen letters.

    **Een voorbeeld uit het dagelijks leven.** Een toetsenbord op je telefoon dat *gistren* tijdens het typen in *gisteren* verandert, doet wat jouw functie `correct` doet: het toetsenbord vond een bekend woord dicht bij wat je typte, en koos een waarschijnlijk woord.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_correct, result_near, result_pick, result_sort):
    L.wrap_up(result_near, result_pick, result_correct, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
