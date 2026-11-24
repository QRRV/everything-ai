# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-topic-models. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-topic-models", "nl")
    C = L.checks   # de alinea's van 48 Wikipedia-pagina's in het Engels en het Nederlands, de topic models van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Een document als vector](/course/open/nlp-document-vectors?lang=nl), en een over [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_matrix():
    quick_matrix = L.choice(
        "quick-matrix",
        question="In *Een document als vector* wordt een verzameling documenten een document-term matrix. Wat zegt het getal in rij 3, kolom 7? Rijen en kolommen tellen vanaf 0.",
        options={
            "swapped": "Hoe vaak woord 3 in document 7 voorkomt",
            "right": "Hoe vaak woord 7 in document 3 voorkomt",
            "documents": "In hoeveel documenten woord 7 voorkomt",
            "share": "Hoeveel van document 3 over topic nummer 7 gaat",
        },
        correct="right",
        why={
            "right": "Elke rij is een document en elke kolom een woord van de vocabulary, dus het getal telt hoe vaak woord 7 in document 3 voorkomt. Een topic model begint met zo'n matrix en met niets anders: niet de volgorde van de woorden, en niet de namen van de documenten.",
            "swapped": "Rij en kolom andersom: welke van de twee staat voor de documenten? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "documents": "Dat is één getal per woord voor de hele verzameling, en de matrix heeft een getal voor elk document en elk woord. Wat telt één cel? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "share": "Aandelen van topics bouwt deze les; de matrix bevat iets eenvoudigers. Wat telt één cel ervan? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_matrix
    return (quick_matrix,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="Een lijst geeft de woorden *oven*, *pasta* en *wind* de weights 6, 3 en 1. Maak van de weights een kansverdeling over de drie woorden: welke kans krijgt *oven*?",
        options={
            "weight": "6, de weight zelf",
            "equal": "1/3, een van drie woorden",
            "right": "0,6",
            "half": "0,5, twee keer zoveel als *pasta*",
        },
        correct="right",
        why={
            "right": "De kansen van een verdeling tellen op tot 1, dus je deelt elke weight door de som van de weights, 6 + 3 + 1 = 10: *oven* krijgt 0,6, *pasta* 0,3 en *wind* 0,1. Een topic model houdt zulke weights bij voor elk topic.",
            "weight": "Een kans ligt tussen 0 en 1, en de drie tellen op tot 1. Waardoor deel je elke weight? Zie [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "equal": "Gelijke kansen laten de weights weg. Hoe worden 6, 3 en 1 drie getallen die optellen tot 1? Zie [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "half": "Met 0,5 voor *oven* krijgt *pasta* 0,25 en *wind* ongeveer 0,08: samen minder dan 1. Waar horen de drie samen op uit te komen? Zie [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzameling kan te groot zijn om helemaal te lezen, als je alleen wilt weten waar de verzameling over gaat: de brieven in een archief, de publicaties van een tijdschrift, de antwoorden op een open vraag in een enquête. Blei (2012) beschrijft topic models als algoritmes die de belangrijkste thema's ontdekken in een grote verzameling documenten zonder verdere ordening. Blei trainde er een met 100 topics op 17.000 publicaties uit het tijdschrift *Science*. Eén topic had als waarschijnlijkste woorden *human*, *genome*, *dna*, *genetic* en *genes*, een ander *evolution*, *evolutionary*, *species*, *organisms* en *life*. Blei benadrukt dat het algoritme niets over deze onderwerpen weet, en dat de publicaties geen labels met topics of trefwoorden hebben.

    Deze les gebruikt 2.067 alinea's van 48 pagina's van de Engelse Wikipedia: de pagina's van de development set van SQuAD (Rajpurkar en anderen, 2016), een verzameling vragen over Wikipedia, met het Nederlands uit SQuAD-NL (de Vries en anderen, 2023). Het meeste Nederlands is machinevertaling: mensen verbeterden 240 van de alinea's. De pagina's gaan over heel verschillende dingen, van *Super Bowl 50* en *Warschau* tot *Zuurstof* en *Priemgetal*. Je weet van elke alinea van welke pagina de alinea komt, dus je kunt nagaan wat een topic model vindt. Het model zelf ziet de namen van de pagina's nooit.

    **Een voorbeeld uit het dagelijks leven.** Je gaat door een stapel oude mails zonder mappen. Sommige gaan over werk, sommige over familie, en een paar over allebei, zoals de mail waarin je een dag vrij vraagt voor een bruiloft. Niemand vertelde je welke thema's er zijn: al lezend vond je de thema's, en de namen bedacht je zelf.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_mix = L.guess(
        "guess",
        question="Het topic model van deze les geeft elk van de 2.067 Nederlandse alinea's een mix van 12 topics: 12 aandelen die samen 1 zijn. In hoeveel alinea's heeft één topic een aandeel van minstens de helft?",
        options={
            "few": "Ongeveer één op de tien: de meeste alinea's spreiden zich over veel topics",
            "third": "Ongeveer drie op de tien",
            "right": "Ongeveer zeven op de tien",
            "all": "Bijna allemaal, want elke alinea komt van één pagina",
        },
        reveal="**1.419 van de 2.067: ongeveer zeven op de tien.** Het grootste aandeel van een alinea is gemiddeld 0,62. Een alinea van een pagina gaat meestal over één ding, en LDA, het model van deze les, geeft de voorkeur aan mixen met weinig topics. De andere drie op de tien mengen: vaak een pagina die een topic deelt met andere pagina's, of een alinea die tussen twee onderwerpen in staat.",
    )
    guess_mix
    return (guess_mix,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **topic** is een kansverdeling over de woorden van een vocabulary: elk woord heeft een kans in elk topic, en een topic over koken geeft *oven* en *pasta* een hoge kans. Blei definieert een topic precies zo: als een verdeling over een vaste vocabulary. Een **topic model** beschrijft een verzameling met een vast aantal topics, $k$, en geeft elk document een **mix**: een aandeel voor elk topic, $k$ getallen die samen 1 zijn. Alle documenten delen dezelfde topics, schrijft Blei, maar elk document heeft de topics in een eigen verhouding.

    Het bekendste topic model is **latent Dirichlet allocation**, **LDA**, van Blei, Ng en Jordan (2003). Het basisidee: een document is een willekeurige mix van verborgen topics, en elk topic is een verdeling over woorden. LDA stelt zich voor hoe elk document geschreven is, in twee stappen:

    1. Kies een mix voor het document.
    2. Voor elk woord van het document: kies een topic volgens de mix, en dan een woord volgens dat topic.

    Niemand schrijft zo, en LDA beweert dat ook niet: het verhaal zegt alleen welke aantallen waarschijnlijk zijn. LDA werkt het verhaal achterstevoren af. Het krijgt de document-term matrix, de aantallen van de woorden in elk document en verder niets, en zoekt topics en mixen waarbij deze aantallen waarschijnlijk zijn. Blei noemt dat het omdraaien van het verhaal: welke verborgen structuur heeft de verzameling waarschijnlijk voortgebracht? De topics zijn **latent**: verborgen, want niemand heeft de topics opgeschreven. De mixen komen uit een **Dirichlet**-verdeling, met een parameter **α** (alfa): hoe kleiner α, hoe minder topics een mix meestal gebruikt. scikit-learn zet α op $1/k$, tenzij je zelf iets anders kiest.

    De volgorde van de woorden maakt voor LDA niet uit, alleen hoe vaak elk woord voorkomt, net als in de bag of words van *Een document als vector*. En LDA nummert de topics, van 0 tot $k - 1$; de namen zijn aan jou. Blei, Ng en Jordan laten topics zien van een model met 100 topics van 16.000 nieuwsberichten. De verdelingen lijken, zoals de auteurs hoopten, een deel van de onderliggende topics te vangen, en de auteurs gaven deze topics zelf een naam.

    De afbeelding laat zes korte documenten zien en wat LDA ervan maakt met twee topics: de kans op elk woord in elk topic, en de mix van elk document.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    LDA kreeg alleen de aantallen. Het vond een keukentopic, topic 0, met *oven*, *pasta* en *pizza*, en een weertopic, topic 1, met *wind*, *storm* en *mist*. Documenten 0 tot en met 3 gebruiken elk de woorden van één topic, en de mix van elk van deze vier is 0,90 voor dat topic: niet 1, want α houdt een beetje over voor het andere topic. Document 4 heeft twee woorden van elk, en een mix van half om half. Document 5, *mist wind pasta*, neigt naar het weer, 0,64; toch krijgt *pasta* erin topic 0, want *pasta* is alleen in het keukentopic waarschijnlijk.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **topic** | een kansverdeling over de woorden van een vocabulary |
    | **topic model** | een model dat een verzameling beschrijft met $k$ topics, en elk document met een mix ervan |
    | **mix** | de aandelen van de $k$ topics in één document: $k$ getallen die samen 1 zijn |
    | $k$ | het aantal topics, dat je zelf kiest |
    | **latent Dirichlet allocation** (LDA) | het bekendste topic model: elk document een mix uit een Dirichlet-verdeling, elk woord uit een topic dat de mix kiest |
    | **latent** | verborgen: niet in de data, gevonden door het model |
    | **α** (alfa) | de parameter van de Dirichlet-verdeling van de mixen: hoe kleiner, hoe minder topics een mix meestal gebruikt |
    | `LatentDirichletAllocation(n_components=k, random_state=0)` | LDA van scikit-learn met `k` topics; `random_state` legt vast waar het model begint, zodat een run dezelfde topics opnieuw geeft |
    | `lda.fit_transform(counts)` | traint het model op een document-term matrix en geeft de mix van elk document: een rij per document, een kolom per topic |
    | `lda.components_` | de weights van de topics: een rij per topic, een kolom per woord; elke rij gedeeld door de eigen som geeft het topic als kansen |
    | `np.argsort(row)` | de plaatsen van de waarden van `row`, van de kleinste waarde naar de grootste, vanaf 0 geteld: `np.argsort(np.array([0.2, 0.5, 0.1]))` geeft `array([2, 0, 1])` |
    | `C.count_matrix(docs)` | de document-term matrix van korte documenten met woorden gescheiden door spaties, als array met aantallen, en de woorden op alfabet |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Tel de woorden van elk document: de document-term matrix. Laat stop words weg, en woorden die in heel weinig of in heel veel documenten voorkomen.
    2. Kies $k$, en train LDA op de aantallen.
    3. Lees elk topic af aan de woorden met de hoogste kans.
    4. Lees elk document af aan de mix, en elk woord aan het topic waar het woord het waarschijnlijkst uit komt.
    5. Geef de topics zelf een naam, en leg de namen naast waar de documenten over gaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code telt de zes documenten van de afbeelding, traint LDA met twee topics, en print de woorden van de kolommen, de weights van de twee topics en de mix van elk document, afgerond op twee decimalen.
    """)
    return


@app.cell
def example_cell():
    from sklearn.decomposition import LatentDirichletAllocation

    _docs = ["oven pizza pasta oven", "wind storm mist wind", "pizza pasta oven pasta",
             "storm mist wind storm", "oven pizza wind storm", "mist wind pasta"]
    _counts, _words = C.count_matrix(_docs)
    _lda = LatentDirichletAllocation(n_components=2, random_state=0)
    _mix = _lda.fit_transform(_counts)
    print(_words)
    print(_lda.components_.round(2))
    print(_mix.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De weights zijn nog geen kansen: de eerste rij telt op tot 13,94, de tweede tot 15,06. De documentatie van scikit-learn zegt wat deze getallen zijn: `components_[i, j]` kun je zien als een pseudo-aantal, het aantal keer dat woord j aan topic i is toegekend. De zes documenten hebben 23 woorden, en LDA telt er voor elk woord in elk topic een klein aantal bij, de `topic_word_prior` van scikit-learn, 1/k tenzij je een ander getal kiest, dus hier 0,5: 23 + 12 × 0,5 = 29 = 13,94 + 15,06. Gedeeld door de som van de rij wordt een weight een kans: *oven* krijgt 4,46 / 13,94 = 0,32 in topic 0, zoals in de afbeelding.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: LDA op 2.067 alinea's

    De course telt de woorden van de 2.067 Nederlandse alinea's: 4.761 woorden van drie letters of meer die in minstens 5 alinea's voorkomen en in hoogstens een tiende ervan, zonder stop words: 98.127 getelde woorden in totaal. Daarna traint de course LDA met 12 topics, 20 stappen lang. In elke stap gaat LDA één keer over alle alinea's en verbetert de topics en de mixen.

    De grafiek laat zien hoe verrast het model is door de woorden van de alinea's. Net als in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) is de verrassing van een woord min de log van de kans op het woord, en de grafiek geeft het gemiddelde per woord, zoals scikit-learn dat schat. De gestippelde lijn is een model zonder topics dat elk van de 4.761 woorden dezelfde kans geeft: log 4.761 = 8,468. Naast de grafiek zie je de 6 woorden met de hoogste kans in elk topic, zoals de topics op dat moment zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train LDA op de alinea's, 20 stappen")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button):
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om te trainen.*")
    else:
        _now = {}

        def _progress(_step, _surprise, _tops):
            if "chart" not in _now:
                _words = C.live_words(L.lang)
                _now["chart"] = L.live(_words, C.STEPS, low=C.LOW, high=C.HIGH)
                _now["chart"].reference(C.uniform_line(L.lang), _words["uniform"])
                _now["words"] = _words
            _now["chart"].watch(lambda: C.now_parts(_tops, L.lang))
            _now["chart"].add(_step, _surprise, _now["words"]["progress"].format(s=_step, n=C.STEPS, v=L.number(_surprise, 3)))

        mo.output.replace(mo.md("*De woorden van de alinea's tellen …*"))
        _run = C.demo(L.lang, _progress)
        if isinstance(_run, str):
            _out = _now["chart"].stopped(C.slow_note(L.lang)) if "chart" in _now else mo.md(C.slow_note(L.lang))
        else:
            _out = mo.vstack([
                mo.md(f"""Na {C.STEPS} stappen is de gemiddelde verrassing per woord {L.number(_run["curve"][-1][1], 3)}, tegen {L.number(_run["uniform"], 3)} zonder topics. Dit zijn de topics van de course: de oefeningen hieronder gebruiken precies dit model."""),
                L.explore("rewind", "stap", lambda _s: (C.rewind_frame(_run, int(_s), L.lang), C.rewind_said(_run, int(_s), L.lang)),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar stap 1 en zoek *abc*: in hoeveel topics staat het bij de 6 woorden? Loop dan vooruit: vanaf welke stap staat het in nog maar één topic, en welke woorden staan daar ernaast?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Na de eerste stap is LDA nauwelijks begonnen: *abc*, van de pagina *American Broadcasting Company*, staat in zes topics tegelijk, en de verrassing per woord, 8,526, ligt zelfs boven de gestippelde lijn. Vanaf stap 9 staat *abc* in één topic, met *netwerk* en *seizoen*. De verrassing daalt eerst snel, dan steeds langzamer: van stap 10 tot stap 20 nog slechts 0,062. De meeste topics veranderen na stap 10 nauwelijks. Langer trainen zou de verrassing nog wat verlagen, maar verrassing is niet hetzelfde als betekenis; na oefening 4 komt de les daarop terug. Dit is één run, met `random_state` 0. Met `random_state` 1 vindt LDA andere topics: geen van de 12 heeft dezelfde vijf woorden als hier.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. topic_words: de woorden van elk topic

        Schrijf `topic_words(weights, words, n)`, waarbij `weights` een array is met een rij per topic en een kolom per woord, zoals `components_`, en `words` de lijst met de woorden van de kolommen. Geef een lijst terug met voor elk topic een lijst van de `n` woorden met de hoogste kans, de hoogste eerst, elk als paar `(woord, kans)`. De kans op een woord in een topic is de weight gedeeld door de som van de weights in de rij van het topic. Vervang de twee keer `...`.

        De check roept `topic_words(small_weights, small_words, 3)` aan, met twee topics over de zes woorden `["mist", "oven", "pasta", "pizza", "storm", "wind"]` en de weights `[[0.5, 4.0, 3.0, 2.0, 0.4, 0.1], [2.0, 0.2, 0.3, 0.1, 1.8, 3.6]]` (in de les: `C.SMALL_WORDS` en `C.SMALL_WEIGHTS`). Dat geeft `[[("oven", 0.4), ("pasta", 0.3), ("pizza", 0.2)], [("wind", 0.45), ("mist", 0.25), ("storm", 0.225)]]`. Daarna eigen gevallen: hetzelfde met `n` 1; drie topics over vier woorden met `n` 2; één topic over drie woorden met `n` 3; twee topics over twee woorden met `n` 1; en het model van de course voor de Engelse en voor de Nederlandse alinea's met `n` 5.
        """),
        L.task("Zoek de woorden met de hoogste kans in elk topic."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("topic_words", lines=11)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("topic_words", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "topic_words",
        [
            "`probs` is `weights` gedeeld door de som van elke rij: `weights.sum` langs `axis=1`. `best` bevat de plaatsen van de `n` grootste waarden van `row`: `np.argsort` sorteert van de kleinste waarde naar de grootste, dus sorteer `-row` en houd de eerste `n`.",
            "`weights.sum(axis=1)` geeft één som per topic, als platte array, en `weights` heeft een rij per topic en een kolom per woord. Welk argument van `sum` uit *Broadcasting* houdt de sommen als kolom, zodat elke rij door de eigen som wordt gedeeld?",
            "`    probs = weights / weights.sum(axis=1, ___)`",
        ],
        why="`probs = weights / weights.sum(axis=1, keepdims=True)` en, in de loop, `best = np.argsort(-row)[:n]`. Het voorbeeld geeft `[[(\"oven\", 0.4), (\"pasta\", 0.3), (\"pizza\", 0.2)], [(\"wind\", 0.45), (\"mist\", 0.25), (\"storm\", 0.225)]]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_words():
    mo.md(r"""
    Sommige topics herken je meteen: topic 11 met *kracht*, *krachten* en *snelheid*, topic 3 met *scholen*, *school* en *studenten*. Andere vragen een tweede blik. Topic 6 heeft *khan*, *mongoolse*, *zuurstof*, *genghis* en *dynastie*: de Mongolen en zuurstof in één topic. En *the* staat in topic 4 en in topic 5: een Engels woord uit namen en titels in de Nederlandse tekst, en het staat niet op de Nederlandse lijst met stop words. De kansen zijn klein, hoogstens 0,043, want elk topic verdeelt de kans over 4.761 woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_topics():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. word_topics: het topic van elk woord

        In het verhaal van LDA komt elk woord van een alinea uit één topic. Uit welk topic? Volgens [De regel van Bayes](/course/open/prob-bayes?lang=nl) is de kans dat een woord uit topic $t$ komt evenredig met het aandeel van $t$ in de mix van de alinea maal de kans op het woord in $t$. Schrijf `word_topics(mix, topics, ids)`, waarbij `mix` de mix van een alinea is, een array met $k$ aandelen; `topics` een array met kansen, met een rij per topic en een kolom per woord; en `ids` de lijst met de kolommen van de woorden van de alinea, in de volgorde van de tekst, vanaf 0 geteld. Geef een lijst terug met voor elk woord het nummer van het topic met het grootste product. Topics zijn genummerd vanaf 0. Vervang de `...`.

        De check roept `word_topics(small_mix, small_topics, [1, 2, 0, 5])` aan, met de mix `[0.9, 0.1]`, de topics van oefening 1 als kansen (in de les: `C.SMALL_MIX` en `C.SMALL_TOPICS`), en de woorden *oven*, *pasta*, *mist* en *wind*. Dat geeft `[0, 0, 0, 1]`. *mist* is waarschijnlijker in het weertopic, 0,25 tegen 0,05, maar deze alinea is voor 90% keuken: 0,9 × 0,05 = 0,045 tegen 0,1 × 0,25 = 0,025. Daarna eigen gevallen: een mix van half om half; de mix `[0.3, 0.7]` met een woord twee keer; helemaal geen woorden; drie topics over vier woorden; en alinea 3 van de Engelse en van de Nederlandse alinea's, vanaf 0 geteld, met het model van de course.
        """),
        L.task("Geef elk woord van een alinea het topic waar het woord het waarschijnlijkst uit komt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topics():
    box_topics = L.code_box("word_topics", lines=9)
    box_topics
    return (box_topics,)


@app.cell(hide_code=True)
def check_topics(box_topics):
    result_topics = L.check("word_topics", box_topics.value)
    result_topics
    return (result_topics,)


@app.cell(hide_code=True)
def hints_topics():
    L.hints(
        "word_topics",
        [
            "Vermenigvuldig voor woord `j` de `mix` met de kansen op woord `j` in elk topic. `np.argmax` geeft de plaats van het grootste product, `int` maakt er een gewoon getal van, en `found.append` voegt het getal toe aan de lijst.",
            "`topics` heeft een rij per topic en een kolom per woord, dus `topics[j]` is een rij: de kansen op alle woorden in topic `j`. Je hebt de kansen op woord `j` in elk topic nodig. Hoe neem je kolom `j` van een array, uit elke rij?",
            "`        found.append(int(np.argmax(mix * ___)))`",
        ],
        why="In de loop: `found.append(int(np.argmax(mix * topics[:, j])))`. Het voorbeeld geeft `[0, 0, 0, 1]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_topics():
    mo.md(r"""
    Alinea 3 gaat over hoe Super Bowl 50 op televisie kwam, en de mix is 0,56 topic 4 en 0,37 topic 5. *CBS*, *zond* en *uitzending* komen uit topic 5, het topic van *abc* en *netwerk*; *Super*, *Bowl* en *wedstrijd* uit topic 4, het topic van *bowl* en *super*. Een alinea gaat niet over één topic: elk woord ervan neigt de ene of de andere kant op. *Britse* gaat ook naar topic 4, al beschrijft het woord hier de band Coldplay. LDA weet alleen dat *britse* en *bowl* in hetzelfde topic voorkomen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pages():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. topic_pages: de pagina's van een topic

        Jij weet van welke pagina elke alinea komt; LDA niet. Schrijf `topic_pages(mix, pages, topic)`, waarbij `mix` een array is met de mix van elke alinea, een rij per alinea; `pages` de lijst met de namen van de pagina's van de alinea's; en `topic` het nummer van een topic. Geef een dictionary terug met voor elke pagina hoeveel alinea's ervan `topic` als grootste aandeel hebben. Een pagina zonder zulke alinea's laat je weg. Vervang de `...`.

        De check roept `topic_pages(small_mixes, small_pages, 0)` aan, met vijf alinea's met de mixen `[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.7, 0.3]]` en de pagina's `["Pizza", "Storm", "Pizza", "Mist", "Pasta"]` (in de les: `C.SMALL_MIXES` en `C.SMALL_PAGES`). Dat geeft `{"Pizza": 1, "Pasta": 1}`. Daarna eigen gevallen: topic 1 van hetzelfde; drie topics over vier alinea's van de pagina's *Radio* en *Film*, topic 0; een topic dat geen enkele alinea als grootste aandeel heeft; topic 9 van de Engelse en topic 5 van de Nederlandse alinea's met het model van de course.
        """),
        L.task("Tel voor elke pagina de alinea's met één topic als grootste aandeel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pages():
    box_pages = L.code_box("topic_pages", lines=12)
    box_pages
    return (box_pages,)


@app.cell(hide_code=True)
def check_pages(box_pages):
    result_pages = L.check("topic_pages", box_pages.value)
    result_pages
    return (result_pages,)


@app.cell(hide_code=True)
def hints_pages():
    L.hints(
        "topic_pages",
        [
            "In de loop: een `if` die alleen de alinea's houdt met `topic` als grootste aandeel. Daaronder het tellen met een dictionary: `if pages[i] in counts:` telt 1 op bij `counts[pages[i]]`, en `else:` geeft de pagina 1.",
            "Welk topic heeft het grootste aandeel in alinea `i`? Dat is een plaats in de rij `mix[i]`: de plaats van de grootste waarde. Welke functie van NumPy uit oefening 2 geeft deze plaats?",
            "`        if ___ == topic:`",
        ],
        why="In de loop: `if np.argmax(mix[i]) == topic:` met daaronder `if pages[i] in counts:` en `counts[pages[i]] = counts[pages[i]] + 1`, en `else:` met `counts[pages[i]] = 1`. Het voorbeeld geeft `{\"Pizza\": 1, \"Pasta\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pages():
    mo.md(r"""
    Topic 5 bevat 85 alinea's van *American Broadcasting Company*, 23 van *Doctor Who* en 21 van *Sky (Verenigd Koninkrijk)*: televisie, alle drie. Niet elk topic is zo duidelijk; oefening 4 gaat over topic 2.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: geef de topics een naam

    LDA geeft nummers; de namen zijn van jou. Schrijf een naam achter elk nummer, één per regel, zoals `11: krachten en beweging`. De afbeelding zet je namen naast de woorden van elk topic en de pagina's met de meeste alinea's erin. Welke topics kun je in twee woorden noemen, en voor welke heb je *en* nodig?
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    names_box = mo.ui.text_area(value=C.NAMES_START, label="Jouw namen, één regel per topic", rows=12, full_width=True)
    names_box
    return (names_box,)


@app.cell(hide_code=True)
def try_look(names_box):
    mo.Html(C.names_picture(names_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_mixed():
    mo.md(r"""
    ### Oefening 4. Een pagina in het verkeerde topic

    Topic 2 heeft de woorden *luther*, *cellen*, *chloroplast*, *chloroplasten* en *luthers*. De meeste alinea's ervan komen van *Maarten Luther*, *Bladgroenkorrel* en *Immuunsysteem*.
    """)
    return


@app.cell(hide_code=True)
def choice_mixed_cell():
    choice_mixed = L.choice(
        "mixed-topic",
        question="Waarom zet LDA alinea's over Maarten Luther in een topic over cellen?",
        options={
            "bug": "Een fout in scikit-learn: een LDA zonder fouten geeft elke pagina een eigen topic",
            "word": "De alinea's over Luther gebruiken het woord *cellen*",
            "right": "12 topics, 48 pagina's: woorden die nooit samen staan, kunnen een topic delen",
            "names": "LDA las de namen van de pagina's, en deze lijken op elkaar",
        },
        correct="right",
        why={
            "right": "Twaalf topics voor 48 pagina's: sommige pagina's moeten een topic delen. *luther* staat in geen enkele alinea van *Immuunsysteem*, en *cellen* in geen enkele van *Maarten Luther*, dus één topic kan allebei bedienen: een alinea over Luther gebruikt de ene helft van het topic, een alinea over cellen de andere. Een topic is een groep woorden die LDA samen handig vindt, geen betekenis. Met 20 topics krijgt *chloroplast* een topic met *zuurstof* en *water*, maar *luther* en *cellen* blijven samen: meer topics verschuiven het delen, maar maken er geen einde aan.",
            "bug": "LDA heeft geen regel dat een pagina een eigen topic krijgt, en ziet de pagina's niet eens. Hoeveel topics zijn er hier, en hoeveel pagina's?",
            "word": "*cellen* komt in geen enkele alinea van *Maarten Luther* voor. Welke woorden hebben de alinea's over Luther en de alinea's over cellen gemeen?",
            "names": "LDA krijgt alleen de aantallen van de woorden in elke alinea, nooit de naam van de pagina. Waarop baseert LDA de topics?",
        },
    )
    choice_mixed
    return (choice_mixed,)


@app.cell(hide_code=True)
def text_after_choice():
    mo.md(r"""
    Hoe weet je of een topic model goed is? De grafiek hierboven mat verrassing, op dezelfde alinea's waarop LDA trainde. Chang en anderen (2009) vroegen het aan mensen. In de test met een indringer kreeg iemand zes woorden te zien: de vijf waarschijnlijkste woorden van een topic, en één woord dat in dat topic onwaarschijnlijk is maar in een ander topic waarschijnlijk. Als het topic klopt, vind je de vreemde eend snel: de meeste mensen wijzen *appel* aan in *hond, kat, paard, appel, varken, koe*. Tot verrassing van de onderzoekers vonden topic models die beter scoren op de waarschijnlijkheid van nieuwe teksten soms topics met minder betekenis. Blei (2012) schrijft ook dat er geen technische reden is om aan te nemen dat een betere score op nieuwe teksten een betere ordening of een makkelijkere uitleg oplevert. Of een topic iets betekent, is een vraag voor mensen die het topic lezen, zoals jij deed bij het geven van namen.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een document als vector](/course/open/nlp-document-vectors?lang=nl) maakte van elke tekst een rij met aantallen. LDA begint met zulke rijen en verder niets, en vindt $k$ topics, elk een kansverdeling over de woorden, en een mix van topics voor elk document: leren zonder labels, zoals in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl). Je `topic_words` las de topics af aan de weights, je `word_topics` gaf elk woord van een alinea het waarschijnlijkste topic, en je `topic_pages` legde de topics naast de pagina's, die LDA nooit zag. Sommige topics passen bij een pagina, andere voegen pagina's samen waarvan de woorden elkaar nooit tegenkomen; de namen, en het oordeel of een topic iets betekent, zijn van jou. De volgende les, *Van tekst naar een tabel*, haalt feiten uit een tekst, zoals namen en datums, en zet de feiten in de kolommen van een tabel.

    **Een voorbeeld uit het dagelijks leven.** Een receptensite ordent de recepten op de woorden erin: *bloem*, *oven* en *gist* wijzen op bakken, *wok* en *sojasaus* op roerbakken. Een recept voor pizza van de barbecue krijgt een mix van bakken en barbecueën, en een mens bepaalt nog steeds hoe elke groep heet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mixed, result_pages, result_topics, result_words):
    L.wrap_up(result_words, result_topics, result_pages, choice_mixed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
