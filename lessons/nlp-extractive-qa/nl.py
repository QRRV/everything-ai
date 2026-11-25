# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-extractive-qa. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-extractive-qa", "nl")
    C = L.checks   # de alinea's en vragen van SQuAD en SQuAD-NL, in zinnen geknipt, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Soorten question answering](/course/open/nlp-qa-kinds?lang=nl), en één over [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_span():
    quick_span = L.choice(
        "quick-span",
        question='In *Soorten question answering* is het antwoord van extractive question answering een span `(start, end)`, zodat `text[start:end]` het antwoord is. Welke span geeft *21.00*, bij `text = "Geopend tot 21.00"`?',
        options={
            "short": "`(12, 16)`",
            "right": "`(12, 17)`",
            "length": "`(12, 5)`",
            "text": '`"21.00"`, het antwoord zelf',
        },
        correct="right",
        why={
            "right": "*21.00* begint op plaats 12, na 12 tekens, spaties meegeteld, en het einde is de plaats net na het laatste teken: 12 + 5 = 17. In deze les wezen mensen zulke spans aan in alinea's van Wikipedia.",
            "short": "Een slice neemt het teken op de plaats van het tweede getal niet mee. Welk teken laat `text[12:16]` weg? Zie [Soorten question answering](/course/open/nlp-qa-kinds?lang=nl).",
            "length": "Het tweede getal zegt waar het antwoord eindigt, niet hoe lang het is. Waar eindigt *21.00*? Zie [Soorten question answering](/course/open/nlp-qa-kinds?lang=nl).",
            "text": "Het antwoord zelf zegt wat het is, niet waar het staat. Welke twee getallen zeggen waar het begint en eindigt? Zie [Soorten question answering](/course/open/nlp-qa-kinds?lang=nl).",
        },
    )
    quick_span
    return (quick_span,)


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="In *Normaliseren: hoofdletters, accenten en stop words* liet je de stop words weg uit een lijst woorden. Welke woorden zijn stop words?",
        options={
            "end": "Woorden aan het eind van een zin, zoals het laatste woord van de zin",
            "rare": "Zeldzame woorden die slechts één keer in een tekst staan",
            "right": "De woorden die het vaakst voorkomen, zoals *de*, *van* en *is*",
            "unknown": "Woorden die een tokenizer niet kan knippen",
        },
        correct="right",
        why={
            "right": "Woorden zoals *de*, *van* en *is* staan in bijna elke tekst, dus zeggen deze woorden weinig over waar een tekst over gaat. Deze les laat de stop words weg als de les telt welke woorden een vraag en een zin delen.",
            "end": "Stop words hebben niets met het eind van een zin te maken. Welke woorden staan in bijna elke tekst? Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
            "rare": "Zeldzame woorden zeggen vaak het meest over een tekst. Welke woorden zeggen het minst? Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
            "unknown": "Een lijst stop words gaat over hoe vaak een woord voorkomt, niet over knippen. Welke woorden staan erop? Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op school krijg je bij begrijpend lezen een tekst met vragen, en het antwoord op elke vraag staat ergens in de tekst. In 2016 gaven Pranav Rajpurkar en anderen van Stanford computers zo'n toets op grote schaal: SQuAD, meer dan 100.000 vragen die mensen stelden over alinea's van Wikipedia, met elk antwoord aangewezen in de alinea. Omdat elk antwoord een span is, kun je het antwoord van een systeem ermee vergelijken zonder dat er een mens meeleest, en sindsdien zijn veel systemen voor reading comprehension erop gemeten.

    Deze les neemt 1.183 van deze vragen, in het Nederlands, en probeert de eenvoudigste reader die er is: kies de zin die de meeste woorden met de vraag deelt.

    **Een voorbeeld uit het dagelijks leven.** In een lang recept zoek je de temperatuur van de oven door te speuren naar *oven* en *graden*. Zo vind je de goede zin, tot het recept *zet op 200* zegt, in een zin zonder een van beide woorden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_reader = L.guess(
        "guess",
        question="Bij elke vraag van SQuAD hoort een alinea van Wikipedia. Neem de zin van de alinea die de meeste woorden met de vraag deelt, zonder stop words zoals *de* en *in*. Bij hoeveel van de 1.183 vragen staat het antwoord in deze zin?",
        options={
            "quarter": "Bij ongeveer een kwart",
            "half": "Bij ongeveer de helft",
            "most": "Bij ongeveer drie op de vier",
            "all": "Bij bijna allemaal",
        },
        reveal="Bij ongeveer drie op de vier: 877 van de 1.183, of 74%. Een willekeurige zin heeft het antwoord bij ongeveer 26% van de vragen, en steeds de eerste zin van de alinea bij 32%. Woorden tellen doet het veel beter, maar een kwart van de vragen gaat nog mis, en je ziet waarom.",
    )
    guess_reader
    return (guess_reader,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **SQuAD**, de Stanford Question Answering Dataset, is een dataset voor reading comprehension: bij elke vraag hoort een alinea, en het antwoord is een span van deze alinea. Rajpurkar en anderen namen 536 pagina's van Wikipedia, knipten de pagina's in alinea's, en lieten mensen op een platform voor crowdsourcing de vragen schrijven. Bij elke alinea stelden crowdworkers tot vijf vragen over de inhoud, met het antwoord erbij: elke vraag in een tekstveld, elk antwoord gemarkeerd in de alinea. Het verzoek was om de vragen in eigen woorden te stellen, zonder stukken tekst uit de alinea over te nemen, en kopiëren en plakken uit de alinea werkte niet. SQuAD verdeelde de vragen over een training set, een development set (zo noemt SQuAD de validation set) en een test set; bij de vragen van de laatste twee markeerden nog minstens twee andere mensen elk ook een antwoord. In totaal heeft SQuAD 107.785 vragen met antwoorden.

    In de bestanden van SQuAD is elk antwoord een tekst met `answer_start`: de plaats van het eerste teken in de alinea. XQuAD, van Mikel Artetxe, Sebastian Ruder en Dani Yogatama, nam 240 alinea's van de development set van SQuAD met de 1.190 vragen erbij. SQuAD-NL, van Wietse de Vries, Martijn Wieling en Malvina Nissim, heeft dezelfde alinea's en vragen in het Nederlands: vertaald door een machine, en verbeterd door mensen. In het Nederlands zijn het 1.183 vragen. Het plaatje laat een van de 240 alinea's zien, van de pagina over de Rijn, met de drie vragen erbij.
    """)
    return


@app.cell(hide_code=True)
def squad_cell():
    mo.Html(C.squad_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_reader():
    mo.md(r"""
    Een reader die de precieze span aanwijst, is moeilijk te bouwen. Een eerste stap is makkelijker: vind de zin met het antwoord. De mensen die de vragen schreven, vroegen naar de alinea, dus de vraag en de zin met het antwoord delen vaak woorden: *Waar begint de Delta in Nederland?* deelt *begint* en *Nederland* met de eerste zin, waar de delta bij Millingen aan de Rijn begint. Een reader kan voor elke zin de gedeelde woorden tellen, en de zin met de meeste kiezen. Stop words zoals *de* en *in* staan in bijna elke zin, dus laat de reader de stop words weg.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **SQuAD** | de Stanford Question Answering Dataset (2016): meer dan 100.000 vragen die mensen stelden over alinea's van Wikipedia, elk met het antwoord aangewezen als span van de alinea |
        | **passage** | de tekst die bij reading comprehension met een vraag meekomt; bij SQuAD een alinea van Wikipedia |
        | `answer_start` | in de bestanden van SQuAD de plaats van het eerste teken van een antwoord in de alinea |
        | **crowdworker** | iemand die per kleine taak betaald wordt op een website, bijvoorbeeld om vragen over een alinea te schrijven |
        | **XQuAD**, **SQuAD-NL** | 240 alinea's van SQuAD met de vragen erbij, in het Engels en in andere talen; SQuAD-NL heeft de alinea's en vragen in het Nederlands |
        | `text.index(part)` | de eerste plaats waar `part` in `text` staat |
        """),
        L.watch_out("In andere lessen van de course, over stap voor stap volgen wat een AI-toepassing doet, is een *span* één stap van een trace, zoals één aanroep van een model. Hier is een span een stuk van een tekst, zoals in *Soorten question answering*: hetzelfde woord voor iets anders."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vind je de zin met het antwoord door woorden te tellen:

    1. Knip de alinea in zinnen, elk met de spatie erachter, zodat de zinnen samen weer de alinea zijn.
    2. Neem de verschillende woorden van de vraag, zonder de stop words.
    3. Tel bij elke zin hoeveel van deze woorden er ook in staan.
    4. Kies de zin met de meeste; bij gelijkspel de eerste daarvan.

    Zo controleer je de keuze: tel de lengtes van de zinnen op, vanaf de eerste, tot het totaal groter is dan `answer_start`. De zin die het totaal groter maakt, heeft het antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem de derde vraag over de Rijn: *Wat is de bijnaam voor de Delta in Nederland?* Zonder de stop words *wat*, *is*, *de*, *voor* en *in* blijven drie woorden over: *bijnaam*, *delta* en *nederland*. De eerste zin deelt er één, *nederland*: in *Rijn-Maasdelta* staat *delta* niet los, maar zit het vast aan *Maas*. De tweede zin deelt er geen, en de derde één, *delta*. Bij dit gelijkspel kiest de reader de eerste zin. Waar begint het antwoord, *Rijndelta*, en in welke zin is dat? Een paar regels rekenen het uit. `C.rhine("nl")` geeft de drie zinnen:
    """)
    return


@app.cell
def example_cell():
    _sentences = C.rhine("nl")
    _paragraph = "".join(_sentences)
    _start = _paragraph.index("Rijndelta")
    print(_start)
    print(len(_sentences[0]), len(_sentences[0]) + len(_sentences[1]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het antwoord begint op plaats 272. De eerste zin heeft 185 tekens, spaties meegeteld, dus beslaat de zin de plaatsen 0 tot en met 184, en de tweede eindigt bij 185 + 107 = 292. Plaats 272 ligt ertussen: het antwoord staat in de tweede zin, *Aangezien de Rijn het grootste deel van het water aanvoert, wordt vaak de kortere term Rijndelta gebruikt.* De reader zit er hier naast. De tweede zin zegt *term* waar de vraag *bijnaam* zegt, noemt Nederland niet, en heeft *delta* alleen in *Rijndelta*: de zin deelt geen enkel woord met de vraag.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_of():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. sentence_of: welke zin het antwoord heeft

        Schrijf `sentence_of`, dat `sentences` krijgt, de lijst met de zinnen van een alinea, elk met de spatie erachter, zodat de zinnen samen de alinea zijn, en `start`, een plaats in de alinea, zoals de `answer_start` van een antwoord. Geef de plaats in de lijst terug van de zin waarin dat teken staat. In de startcode bevat `end` waar de zinnen tot nu toe eindigen. Vervang de `...`.

        De check roept `sentence_of(["Het museum gaat om 10.00 uur open. ", "Om 17.00 uur gaat het weer dicht."], 38)` aan, wat `1` geeft: *17.00 uur* begint op plaats 38, in de tweede zin. Hetzelfde in het Engels, en daarna eigen gevallen: een start bij het allereerste teken, bij het eerste en bij het laatste teken van een zin, in de laatste zin, en antwoorden in de alinea over de Rijn (in de les: `C.rhine("nl")`).
        """),
        L.task("Vind de zin waarin een antwoord begint."),
    ])
    return


@app.cell(hide_code=True)
def exercise_of():
    box_of = L.code_box("sentence_of", lines=6)
    box_of
    return (box_of,)


@app.cell(hide_code=True)
def check_of(box_of):
    result_of = L.check("sentence_of", box_of.value)
    result_of
    return (result_of,)


@app.cell(hide_code=True)
def hints_of():
    L.hints(
        "sentence_of",
        [
            "In de loop groeit `end` met de lengte van zin `k`, met `len` en `sentences[k]`, zodat `end` aangeeft waar deze zin in de alinea eindigt. Als `start` kleiner is dan `end`, begint het antwoord in zin `k`, en geeft `return` de plaats `k` terug.",
            "Een eerste zin van 35 tekens beslaat de plaatsen 0 tot en met 34; plaats 35 is het eerste teken van de volgende zin. Welke vergelijking, `<` of `<=`, houdt plaats 35 buiten de eerste zin?",
            "`        end = end + ___`",
        ],
        why="In de loop: `end = end + len(sentences[k])`, dan `if start < end:` met `return k` eronder.",
    )
    return


@app.cell(hide_code=True)
def text_after_of():
    mo.md(r"""
    Met je functie weet de course bij elk van de 1.183 vragen welke zin het antwoord heeft. De 240 alinea's hebben samen 1.166 zinnen, ongeveer vijf per alinea. In een alinea van vijf zinnen heeft een willekeurige zin het antwoord één keer op de vijf; in kortere alinea's is de kans groter, dus over alle vragen komt het uit op ongeveer een kwart. De meeste antwoorden van SQuAD zijn kort: gemiddeld 3,2 woorden, in een zin van 31,5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. best_sentence: de reader

        Schrijf `best_sentence(question, sentences, stop_words)`. Geef de plaats terug van de zin die de meeste verschillende woorden met de vraag deelt, zonder de woorden in de set `stop_words`; bij gelijkspel de eerste daarvan, en de eerste zin als geen enkele zin een woord met de vraag deelt. Het vak kent `words(text)` uit *Soorten question answering*, dat de woorden van een tekst geeft in kleine letters, zonder leestekens. In de startcode moet `asked` de verschillende woorden van de vraag bevatten die geen stop words zijn, en `shared` hoeveel daarvan in zin `k` staan; de regels erna houden de beste zin tot nu toe bij. Vervang de twee `...`.

        De check roept `best_sentence("Wanneer gaat het zwembad open?", ["De bibliotheek gaat om 10.00 uur open. ", "Het zwembad gaat om 9.00 uur open."], C.STOP_WORDS["nl"])` aan, wat `1` geeft, met de stop words van Snowball. Hetzelfde in het Engels, en daarna eigen gevallen: een vraag met stop words in de verkeerde zin, een woord van de vraag binnen een langer woord, een zin die een woord van de vraag herhaalt, een vraag die een woord herhaalt, een gelijkspel, en vragen van SQuAD.
        """),
        L.task("Kies de zin die de meeste woorden met de vraag deelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("best_sentence", lines=10)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("best_sentence", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "best_sentence",
        [
            "Maak `asked` met `set` van een list comprehension over `words(question)` die elk woord `w` houdt dat niet in `stop_words` staat. In de loop is `shared` de `len` van een tweede list comprehension: elk woord `w` van `asked` dat bij de woorden van `sentences[k]` hoort, uit `words`.",
            "Een set bevat elk verschillend woord één keer, dus een woord dat de vraag herhaalt, telt één keer, en als je langs `asked` gaat, tel je elk woord één keer, ook als de zin het woord herhaalt. Zoek elk woord tussen de woorden van de zin, niet in de zin zelf: `in` op een tekst zoekt naar letters, dus *open* staat ook in *reopens*.",
            "`        shared = len([w for w in asked if w in ___])`",
        ],
        why="`asked = set([w for w in words(question) if w not in stop_words])`, en in de loop `shared = len([w for w in asked if w in words(sentences[k])])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Bij deze vier vragen heeft je reader alleen de eerste goed, en de eerste alleen dankzij een gelijkspel: de eerste en de laatste zin delen elk twee woorden met de vraag, en de eerste zin wint. Bij de kostuumcollectie van de V&A deelt de laatste zin drie woorden met de vraag, *items* en de *v* en de *a* van *V&A*, en gaat over een geschenk van 1.442 kostuums; het antwoord, *meer dan 14.000*, staat in de eerste zin, die alleen *kostuumcollectie* deelt. Bij Jacksonville delen drie zinnen elk één woord, en wint de eerste. En bij de beurs van Warschau zegt de zin met het antwoord *Het werd hersteld in april 1991*: de zin deelt geen woord met *weer tot stand gebracht*, en *Het* wijst terug naar de vorige zin.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. share_right: meet de reader

        Schrijf `share_right(items, stop_words)`. Elk item is een tuple `(question, sentences, start)`: een vraag, de zinnen van de alinea, en waar het eerste antwoord begint. Geef het aandeel van de items terug waarbij `best_sentence` dezelfde zin kiest als de zin die `sentence_of` geeft: de zin waarin het antwoord begint. Het vak kent de `best_sentence` en de `sentence_of` van de course, dus je antwoorden op oefening 1 en 2 doen hier niet mee. Vervang de `...`.

        De check roept `share_right(C.rhine_items("nl"), C.STOP_WORDS["nl"])` aan, de drie vragen over de Rijn, wat `0.6666666666666666` geeft: twee van de drie. Hetzelfde in het Engels, en daarna alle vragen, in het Nederlands en in het Engels (in de les: `C.items("nl")`), en delen ervan.
        """),
        L.task("Meet hoe vaak je reader de zin met het antwoord kiest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share_right", lines=6)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share_right", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share_right",
        [
            "`best_sentence` krijgt `question`, `sentences` en `stop_words`, en geeft de plaats van de zin die je reader kiest; `sentence_of` krijgt `sentences` en `start`, en geeft de plaats van de zin met het antwoord. Als de twee plaatsen gelijk zijn, gaat `right` één omhoog.",
            "Vergelijk de twee plaatsen met `==`, in een `if`. Welke regel onder de `if` telt één op bij `right`?",
            "`        if best_sentence(question, sentences, stop_words) == ___:`",
        ],
        why="In de loop: `if best_sentence(question, sentences, stop_words) == sentence_of(sentences, start):` met `right = right + 1` eronder.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    Je reader kiest de zin met het antwoord bij 877 van de 1.183 vragen: 74,1%. [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) zegt dat je een score eerst met een eenvoudige baseline vergelijkt. Een willekeurige zin geeft gemiddeld 26,1%, en steeds de eerste zin van de alinea 32,4%: woorden tellen doet het veel beter dan allebei. Rajpurkar en anderen bouwden voor SQuAD een logistic regression met 180 miljoen features, en dit model vond de zin met het antwoord bij 79,3% van de vragen, op de eerste versie van SQuAD. Op de Engelse vragen van versie 1.1 haalt dezelfde reader als de jouwe 78,1%, dicht daarbij.

    Het weglaten van de stop words helpt in het Nederlands zelfs niet: met de stop words erbij heeft dezelfde reader er 882 goed, 5 meer. Zonder stop words tellen minder woorden mee, dus eindigen meer zinnen gelijk, en een gelijkspel gaat naar de eerste zin, ook als het antwoord in een latere zin staat. In het Engels helpt het weglaten een beetje: 929 goed, tegen 912 met de stop words erbij.
    """)
    return


@app.cell(hide_code=True)
def text_fails():
    mo.md(r"""
    ### Waar de reader misgaat

    Je reader mist 306 van de 1.183 vragen. Bij 124 daarvan eindigt de zin met het antwoord gelijk met een eerdere zin, en wint de eerdere. Bij 88 deelt een andere zin meer woorden met de vraag. En bij 94 deelt de zin met het antwoord helemaal geen woord met de vraag: bijna twee keer zo vaak als in het Engels, waar het 50 keer gebeurt, onder meer door samenstellingen zoals *Rijndelta*, met *delta* erin verstopt. Rajpurkar en anderen gingen 192 vragen van de development set met de hand na, en elke vraag verschilt van de zin met het antwoord in de woorden of in de volgorde van de woorden. Bij 33,3% zijn de belangrijkste woorden synoniemen, zoals bij *What is the Rankine cycle sometimes called?* en *The Rankine cycle is sometimes referred to as a practical Carnot cycle.* Bij 13,6% is er meer dan één zin nodig, bijvoorbeeld omdat een woord zoals *They* terugwijst naar een eerdere zin.

    ### Probeer het: wat de reader kiest

    Kies wat de reader doet en een vraag, en zie de zinnen van de alinea, de woorden die elke zin met de vraag deelt, en wat de reader kiest. Het vak laat de alinea's weg die de course niet laat zien: het heeft 260 van de 1.183 vragen.
    """)
    return


@app.cell(hide_code=True)
def look_kind():
    look_which = mo.ui.dropdown(options=C.kind_options(L.lang), value=list(C.kind_options(L.lang))[4], label="Uitkomst")
    look_which
    return (look_which,)


@app.cell(hide_code=True)
def look_number(look_which):
    look_n = mo.ui.slider(1, C.kind_count(L.lang, look_which.value), value=1, label="Vraag", show_value=True)
    look_n
    return (look_n,)


@app.cell(hide_code=True)
def look_cell(look_n, look_which):
    mo.Html(C.look_picture(L.lang, look_which.value, look_n.value))
    return


@app.cell(hide_code=True)
def text_exercise_left():
    mo.md(r"""
    ### Oefening 4. Wat ontbreekt er nog?

    Je reader kiest de zin met het antwoord bij 74% van de vragen. Maar SQuAD telt een antwoord als goed wanneer het antwoord gelijk is aan een span die mensen aanwezen.
    """)
    return


@app.cell(hide_code=True)
def choice_left_cell():
    what_left = L.choice(
        "what_left",
        question="Wat moet je reader nog doen voordat de reader een antwoord geeft zoals SQuAD het telt?",
        options={
            "nothing": "Niets: de zin met het antwoord is het antwoord",
            "right": "De span van het antwoord in de zin aanwijzen",
            "search": "Eerst de goede alinea vinden tussen alle pagina's van Wikipedia, zoals DrQA doet",
            "write": "Het antwoord opnieuw schrijven in eigen woorden",
        },
        correct="right",
        why={
            "right": "Het antwoord is een span van ongeveer 3 woorden, in een zin van ongeveer 31. Rajpurkar en anderen zagen hetzelfde bij de logistic regression, die de goede zin koos bij 79,3% van de vragen: volgens het paper zit het moeilijke vooral in het vinden van de precieze span binnen de zin. De volgende les meet hoe dicht een antwoord bij de spans van de mensen komt.",
            "nothing": "Een zin heeft ongeveer 31 woorden, het antwoord ongeveer 3. Telt SQuAD een hele zin als het antwoord?",
            "search": "Bij reading comprehension komt de alinea mee met de vraag. Wat heeft de reader nog niet, als de reader de goede zin heeft?",
            "write": "De antwoorden van SQuAD zijn spans van de alinea. Schrijft een extractive reader zelf iets?",
        },
    )
    what_left
    return (what_left,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Soorten question answering](/course/open/nlp-qa-kinds?lang=nl) wees je antwoorden aan als spans, en schreef je een retriever die gedeelde woorden telt. Hier werd hetzelfde tellen, zonder de stop words van [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl), een reader, en mat je de reader tegen de baselines van [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl). De volgende les, *Antwoorden scoren: exact match en F1*, scoort antwoorden zoals SQuAD dat doet: een antwoord dat precies gelijk is aan een span van de mensen, en een antwoord dat slechts een deel van de woorden deelt.

    **Een voorbeeld uit het dagelijks leven.** Het zoekvak van een webpagina springt naar de eerste plaats waar je woorden staan. Zo vind je het antwoord vaak, maar niet op een pagina die hetzelfde in andere woorden zegt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_of, result_pick, result_share, what_left):
    L.wrap_up(result_of, result_pick, result_share, what_left)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
