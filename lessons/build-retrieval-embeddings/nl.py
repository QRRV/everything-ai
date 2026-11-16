# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-retrieval-embeddings. Route: course/route/lessen/bouwen.yaml. Checks, de embeddings en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-retrieval-embeddings", "nl")
    C = L.checks   # de knowledge base, de embeddings die Granite ervan maakte, de checks, en de demo met je retrieve


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_embed():
    quick_embed = L.choice(
        "quick-embed",
        question="Granite Embedding geeft elke embedding lengte 1. Wat is het inproduct van twee van zijn embeddings?",
        options={
            "distance": "De afstand tussen de twee teksten: hoe kleiner, hoe dichterbij",
            "words": "Het aantal woorden dat de twee teksten delen, gedeeld door hun lengte",
            "cosine": "Hun cosine similarity: hoe hoger, hoe dichter in betekenis",
        },
        correct="cosine",
        why={
            "cosine": "De cosine similarity is het inproduct gedeeld door de twee lengtes; als beide lengtes 1 zijn, verandert er niets. Hoe hoger hij is, hoe meer de twee embeddings dezelfde kant op wijzen.",
            "distance": "Een afstand is het kleinst voor de teksten die het dichtst bij elkaar liggen; het inproduct van twee embeddings met lengte 1 is voor die teksten juist het grootst. Zie [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl).",
            "words": "Een embedding is een lijst van 384 getallen, niet van woorden. Zie [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl).",
        },
    )
    quick_embed
    return (quick_embed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De zoekfunctie van [Je eerste RAG](/course/open/build-rag-first?lang=nl) vond het goede punt voor maar {words} van de {meaning} vragen die het in andere woorden vragen, en BM25, uit [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl), ook voor maar {bm25}. Mensen vragen in hun eigen woorden, niet in die van je documenten, en soms in een andere taal. In [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl) vond Granite een Nederlandse hulppagina bij een Engelse vraag. In deze les doorzoek je zo alle {points} punten van de course: op betekenis.

    **Een voorbeeld uit het dagelijks leven.** Je vraagt een bibliothecaris om *een boek over waarom de lucht blauw is*, en krijgt er een met de titel *Licht in de atmosfeer*. Geen van je woorden staat in de titel, maar de bibliothecaris snapt wat je bedoelt; het register achter in een boek kan alleen naar woorden wijzen. Dat is maar een vergelijking: de zoekfunctie hieronder vergelijkt lijsten getallen.
    """.format(words=C.findings()["search_meaning"], bm25=C.findings()["bm25_meaning"], meaning=C.findings()["meaning_questions"],
               points=C.number(len(C.POINTS), L.lang, 0)))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **dense retrieval** | retrieval op betekenis: de embedding van de vraag vergelijken met de embedding van elke tekst, en de teksten met de hoogste cosine similarity houden |
    | **cross-lingual retrieval** | retrieval over talen heen: een vraag in de ene taal vindt teksten in een andere |
    | `vectors @ q` | bij een array `vectors` met één embedding per rij en een embedding `q`: het inproduct van elke rij met `q`, één getal per rij |

    *Dense*, omdat elk van de 384 getallen van een embedding voor elke tekst een waarde heeft. Een lijst die elk van de {vocab} verschillende woorden in de punten telt, zou voor één punt bijna overal nul zijn: geen punt heeft meer dan {most} verschillende woorden.

    De embeddings van de punten hangen niet van de vraag af, dus de course maakte ze één keer: {points} embeddings van 384 getallen, de rijen van één array. Dat duurde {all} seconden op de computer van de course. Een vraag kost daarna één request met één tekst: {one} seconden. Beide komen van hetzelfde model, Granite Embedding, omdat de embeddings van twee modellen niet bij elkaar passen. En omdat Granite een tekst en zijn vertaling embeddings geeft die dicht bij elkaar liggen, vindt een Nederlandse vraag de punten ook, al zijn ze in het Engels.

    `np.array([[0.6, 0.8], [1.0, 0.0]]) @ np.array([0.8, 0.6])` geeft een array met 0,96 en 0,8: het inproduct van elke rij met de embedding.
    """.format(vocab=C.number(C.vocabulary(), L.lang, 0), most=C.most_words(), points=C.number(len(C.POINTS), L.lang, 0),
               all=C.number(C.SECONDS["points"], L.lang, 1), one=C.number(C.SECONDS["question"], L.lang, 3)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: zoeken op betekenis

    1. **Eén keer:** maak van elk punt de embedding, en bewaar de embeddings als de rijen van één array.
    2. **Vraag:** maak de embedding van de vraag, met hetzelfde model: één request met één tekst.
    3. **Score:** neem het inproduct van elke rij met de embedding van de vraag; bij lengte 1 is dat de cosine similarity.
    4. **Houden:** de `k` punten met de hoogste score, de hoogste eerst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De Engelse vraag *{question}* (Hoe kiest een classifier die naar de dichtstbijzijnde bewaarde voorbeelden kijkt een klasse voor iets nieuws?)

    1. **Eén keer:** de course maakte de embeddings van alle {points} punten: een array van {points} rijen van 384 getallen.
    2. **Vraag:** één request geeft de 384 getallen van de vraag.
    3. **Score:** `vectors @ q` geeft {points} scores, van {low} tot {s0}.
    4. **Houden:** de hoogste, {s0}, is een punt van *{t0}*: *{p0}* Het beantwoordt de vraag: de dichtstbijzijnde voorbeelden stemmen. Daarna {s1}, een punt van *{t1}*, en {s2}, van *{t2}*.

    Dat punt deelt alleen {shared} met de vraag, dus BM25 zette het op plaats {bm25}. De eerste van BM25 was het punt van *{t1}*, hier tweede.
    """.format(question=_w["question"], points=C.number(len(C.POINTS), L.lang, 0), low=C.number(_w["low"], L.lang, 3),
               s0=C.number(_w["scores"][0], L.lang, 3), s1=C.number(_w["scores"][1], L.lang, 3), s2=C.number(_w["scores"][2], L.lang, 3),
               t0=C.title_of(_w["top"][0], L.lang), t1=C.title_of(_w["top"][1], L.lang), t2=C.title_of(_w["top"][2], L.lang), p0=_w["top"][0],
               shared=", ".join(["*" + w + "*" for w in _w["shared"][:-1]]) + " en *" + _w["shared"][-1] + "*", bm25=_w["bm25_rank"]))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    Voor elk van de {questions} vragen die een punt beantwoordt, doorzocht de course alle {points} punten op twee manieren, met BM25 en op betekenis, en noteerde de plaats van het goede punt. Daarna deed hij hetzelfde met de vragen in het Nederlands, tegen dezelfde Engelse punten. Het plaatje markeert de {meaning} vragen die het in andere woorden vragen:
    """.format(questions=C.findings()["questions"], points=C.number(len(C.POINTS), L.lang, 0), meaning=C.findings()["meaning_questions"]))
    return


@app.cell(hide_code=True)
def picture_ranks():
    mo.Html(C.ranks_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Op betekenis stond het goede punt bij de bovenste drie voor {meaning} van de {questions} vragen; met BM25 voor {bm25}. Het verschil zit in de vragen in andere woorden: {mm} van de {nm} op betekenis, {bm} met BM25. Op de vragen die woorden met hun punt delen, was BM25 iets beter: {bw} van de {nw} bij de bovenste drie tegen {mw}, en {bwf} keer eerst tegen {mwf}.

        Met de vragen in het Nederlands vond betekenis er {mnl} van de {questions}, BM25 {bnl}. BM25 kan alleen afgaan op de woorden die een Nederlandse vraag met een Engels punt deelt: getallen, namen, en Engelse woorden die het Nederlands ook gebruikt, zoals *cache* en *request*. *{sha}* deelt er geen, omdat het Nederlands *SHA-256-checksum* als één woord schrijft: BM25 zette het punt op plaats {sha_bm25}, zoeken op betekenis op de eerste.
        """.format(meaning=_f["meaning"], questions=_f["questions"], bm25=_f["bm25"], mm=_f["meaning_meaning"], nm=_f["meaning_questions"],
                   bm=_f["bm25_meaning"], bw=_f["bm25_words"], nw=_f["words_questions"], mw=_f["meaning_words"], bwf=_f["bm25_words_first"],
                   mwf=_f["meaning_words_first"], mnl=_f["meaning_nl"], bnl=_f["bm25_nl"], sha=C.question("q17")["text"]["nl"],
                   sha_bm25=C.number(C.ranks()["q17"][("bm25", "nl")], L.lang, 0))),
        L.watch_out("Betekenis is niet altijd beter. *{agent}* (Wat vertelt een agent hoe goed de stap was die hij net zette?) gaat over reinforcement learning, maar op betekenis waren de drie beste punten uit de twee lessen over coding agents, waar een agent iets anders is; BM25 zette het goede punt op de eerste plaats. Een latere les combineert zoeken op woorden en op betekenis, omdat elk punten vindt die de ander mist.".format(agent=C.question("q20")["text"]["en"])),
    ])
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De punten waarvan de embedding het best past

        `meaning_search(q, vectors, points, k)` moet de `k` punten geven waarvan de embedding het hoogste inproduct met `q` heeft, de hoogste eerst. `q` is de embedding van de vraag, een NumPy-array; `vectors` heeft één embedding per rij, in de volgorde van `points`, dus `vectors[i]` is de embedding van `points[i]`. Het vak is de zoekfunctie van [Je eerste RAG](/course/open/build-rag-first?lang=nl), met paren van een score en een punt, maar elke score is 0. Als hij goed is, bewaart de check hem ook als bestand, `work/projects/app/meaning.py`: [Project: je assistent leest je documenten](/course/open/build-project-rag?lang=nl) gebruikt hem.
        """),
        L.task("Geef elk punt het inproduct van zijn embedding met `q` als score, in plaats van 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("meaning_search", lines=5)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("meaning_search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "meaning_search",
        [
            "De score van punt `i` vergelijkt zijn eigen embedding, rij `i` van `vectors`, met de embedding van de vraag.",
            "`vectors[i] @ q` is het inproduct van rij `i` met `q`: één getal, de score van punt `i`.",
            "In de loop: `found.append((vectors[i] @ q, points[i]))`.",
        ],
        why="""```python
def meaning_search(q, vectors, points, k):
    found = []
    for i in range(len(points)):
        found.append((vectors[i] @ q, points[i]))
    return [point for score, point in sorted(found, reverse=True)[:k]]
```

`scores = vectors @ q` vóór de loop geeft alle scores tegelijk, één per rij, en `scores[i]` in de loop geeft dezelfde paren. Alleen de score is anders dan bij zoeken op woorden: de rest van de zoekfunctie blijft zoals hij was.""",
    )
    return


@app.cell(hide_code=True)
def text_retrieve():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De embeddings van de punten één keer maken

        `retrieve(url, question, k)` moet de `k` punten geven die op betekenis het best bij een vraag passen. Het vak doet het zoals `best_match` uit [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl): het stuurt de vraag samen met alle {points} punten, in één request, bij elke vraag. Dat werkt, maar zo maakt het bij elke vraag opnieuw de embeddings van de hele knowledge base: alle punten duurden {all} seconden op de computer van de course, één vraag {one}. De course maakte de embeddings van de punten één keer: `VECTORS` bevat ze, een rij per punt, in de volgorde van `POINTS`. `embed` uit die les, `meaning_search`, `VECTORS` en `POINTS` zijn gegeven. De check draait je functie tegen een stand-in.
        """.format(points=C.number(len(C.POINTS), L.lang, 0), all=C.number(C.SECONDS["points"], L.lang, 1), one=C.number(C.SECONDS["question"], L.lang, 3))),
        L.task("Haal alleen de embedding van de vraag op, en zoek daarmee in `VECTORS`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_retrieve():
    box_retrieve = L.code_box("retrieve", lines=7)
    box_retrieve
    return (box_retrieve,)


@app.cell(hide_code=True)
def check_retrieve(box_retrieve):
    result_retrieve = L.check("retrieve", box_retrieve.value)
    result_retrieve
    return (result_retrieve,)


@app.cell(hide_code=True)
def hints_retrieve():
    L.hints(
        "retrieve",
        [
            "De embeddings van de punten staan klaar in `VECTORS`; bij elke vraag heeft alleen de vraag zelf er nog een nodig.",
            "`embed(url, [question])` geeft een lijst met één embedding; `[0]` haalt hem eruit, en `np.array(...)` maakt er een array van.",
            "`q = np.array(embed(url, [question])[0])`, en dan `return meaning_search(q, VECTORS, POINTS, k)`.",
        ],
        why="""```python
import numpy as np


def retrieve(url, question, k):
    q = np.array(embed(url, [question])[0])
    return meaning_search(q, VECTORS, POINTS, k)
```

Een RAG-systeem maakt de embeddings van zijn documenten één keer, als ze binnenkomen, en bewaart ze; elke vraag kost daarna één kort request. Verandert een document, dan worden alleen zijn eigen embeddings opnieuw gemaakt.""",
    )
    return


@app.cell(hide_code=True)
def text_high_score():
    mo.md(r"""
    ### Oefening 3. Een hoge score

    Geen punt van de course zegt hoeveel mensen hem hebben afgemaakt.
    """)
    return


@app.cell(hide_code=True)
def choice_high_score_cell():
    _f = C.findings()
    choice_high_score = L.choice(
        "high-score",
        question="Bij `{question}` (Hoeveel mensen hebben de course afgemaakt?) vindt zoeken op betekenis toch drie punten, het beste met een score van {high}. Wat zegt die score je?".format(
            question=C.question(_f["none_question"])["text"]["en"], high=C.number(_f["none_high"], L.lang, 3)),
        options={
            "answers": "Dat het beste punt de vraag beantwoordt, omdat {high} een hoge score is".format(high=C.number(_f["none_high"], L.lang, 3)),
            "nothing": "Op zichzelf niets: goede punten scoren vaak lager",
            "below": "Dat Granite de vraag slecht begreep, omdat de score onder de 0,8 ligt",
        },
        correct="nothing",
        why={
            "nothing": "Zoeken op betekenis geeft altijd de `k` hoogste, ook als geen punt de vraag beantwoordt. Het goede punt van {below} van de {questions} vragen scoorde lager dan {high}, waaronder {mbelow} van de {mn} die het in andere woorden vragen. Het model moet zelf zien of de bronnen de vraag beantwoorden, in [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).".format(
                below=_f["gold_below"], questions=_f["questions"], high=C.number(_f["none_high"], L.lang, 3), mbelow=_f["meaning_below"], mn=_f["meaning_questions"]),
            "answers": "Het beste punt gaat over de training van de course op zijn CPU, niet over mensen. En scoren goede punten altijd hoger dan {high}?".format(high=C.number(_f["none_high"], L.lang, 3)),
            "below": "Is er een score waarboven een punt goed is? Het goede punt van de vraag over het geheugen terwijl een netwerk leert, scoorde {low}, en stond tweede.".format(low=C.number(C.gold_score("q19"), L.lang, 3)),
        },
    )
    choice_high_score
    return (choice_high_score,)


@app.cell(hide_code=True)
def text_dutch():
    mo.md(r"""
    ### Oefening 4. Een vraag in het Nederlands

    Kijk nog eens naar de vraag over de checksum, in het plaatje van wat de course mat.
    """)
    return


@app.cell(hide_code=True)
def choice_dutch_cell():
    _f = C.findings()
    choice_dutch = L.choice(
        "dutch-question",
        question="In het Nederlands is de vraag over de checksum `{sha}`. BM25 zet het Engelse punt over de checksum op plaats {place}, zoeken op betekenis op de eerste. Waarom mist BM25 het?".format(
            sha=C.question("q17")["text"]["nl"], place=C.number(C.ranks()["q17"][("bm25", "nl")], L.lang, 0)),
        options={
            "rare": "SHA-256 is een te zeldzaam woord om voor BM25 mee te tellen",
            "translate": "BM25 vertaalt de vraag eerst naar het Engels, en vertaalt hextekens verkeerd",
            "words": "De vraag deelt geen woord met het punt, dus elk punt scoort 0",
        },
        correct="words",
        why={
            "words": "BM25 telt alleen de woorden die een punt met de vraag deelt, zoals ze geschreven staan. *SHA-256-checksum?* is in het Nederlands één woord, en *hoeveel*, *hextekens*, *heeft* en *een* staan in geen enkel Engels punt, dus alle {points} punten scoren 0 en hun volgorde zegt niets. Granite geeft een tekst en zijn vertaling embeddings die dicht bij elkaar liggen, dus zoeken op betekenis vond het punt als eerste. Over alle {questions} vragen in het Nederlands: {mnl} op betekenis, {bnl} met BM25.".format(
                points=C.number(len(C.POINTS), L.lang, 0), questions=_f["questions"], mnl=_f["meaning_nl"], bnl=_f["bm25_nl"]),
            "rare": "BM25 geeft een zeldzaam woord juist meer gewicht: dat doet de idf. Maar is *SHA-256-checksum?* een woord van het Engelse punt?",
            "translate": "BM25 kent geen taal: het telt woorden zoals ze geschreven staan. Welke woorden deelt de Nederlandse vraag met het Engelse punt?",
        },
    )
    choice_dutch
    return (choice_dutch,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Doorzoek de course met je eigen retrieve

    Zodra oefening 1 en 2 goed zijn, kun je de course doorzoeken met je eigen functies: de course start zijn server op je computer, met Granite erachter, en laat naast wat je `retrieve` vindt zien wat BM25 bij dezelfde vraag vindt. Vraag in het Nederlands, in het Engels, of in een andere van de twaalf talen van Granite, en probeer een vraag in je eigen woorden.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*De demo heeft Granite Embedding nodig, en dat staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, in een vaste versie: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, onder de licentie Apache-2.0.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte vrij, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Granite Embedding is gedownload.*")
    return


@app.cell(hide_code=True)
def settings_cell():
    question_text = mo.ui.text(value=C.question("q22")["text"][L.lang], label="Je vraag", full_width=True)
    search_button = mo.ui.run_button(label="Zoek met je retrieve")
    mo.vstack([question_text, search_button])
    return question_text, search_button


@app.cell(hide_code=True)
def demo_answer(box_retrieve, box_search, question_text, search_button):
    _fn = C.learner_retrieve(box_search.value, box_retrieve.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Granite Embedding, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo zoekt met je eigen `meaning_search` en `retrieve`.*")
    elif not search_button.value:
        _out = mo.md("*Druk op de knop om te zoeken.*")
    else:
        mo.output.replace(mo.md("*Je `retrieve` stuurt de vraag; de server laadt Granite en rekent de embedding ervan uit …*"))
        _words = {"title": "Je retrieve: {question}", "mine": "Op betekenis, met je retrieve:", "bm25": "Op woorden, met BM25:",
                  "done": "{seconds} seconden voor je retrieve, met het laden van Granite de eerste keer.", "empty": "Typ eerst een vraag.",
                  "failed": "Je retrieve stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De punten die je `retrieve` vindt, gaan in de prompt zoals in [Je eerste RAG](/course/open/build-rag-first?lang=nl). [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) vraagt Qwen3 om alleen daaruit te antwoorden, het punt te noemen dat het gebruikte, en het te zeggen als ze de vraag niet beantwoorden, wat de score niet kan vertellen. De punten hier zijn kort; een langer document wordt eerst in stukken geknipt, zoals in [Documenten in stukken knippen](/course/open/build-chunking?lang=nl), en elk stuk krijgt zijn eigen embedding.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dutch, choice_high_score, result_retrieve, result_search):
    L.wrap_up(result_search, result_retrieve, choice_high_score, choice_dutch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
