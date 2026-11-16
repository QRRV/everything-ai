# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-hybrid-search. Route: course/route/lessen/bouwen.yaml. Checks, de twee zoekfuncties en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-hybrid-search", "nl")
    C = L.checks   # de punten van de course met hun embeddings, BM25, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_meaning():
    _f = C.findings()
    quick_meaning = L.choice(
        "quick-meaning",
        question="Bij welke vraag vindt zoeken op betekenis een punt beter dan BM25?",
        options={
            "names": "Een vraag naar een zeldzame naam of een getal uit het punt",
            "long": "Een vraag die veel langer is dan het punt dat hij moet vinden",
            "other-words": "Een vraag die zegt wat het punt zegt, in andere woorden",
        },
        correct="other-words",
        why={
            "other-words": "Embeddings van teksten die hetzelfde betekenen liggen dicht bij elkaar, met welke woorden ook. BM25 telt alleen de woorden die een vraag met een punt deelt: van de {n} vragen in andere woorden vond zoeken op betekenis er {m}, BM25 {b}.".format(
                n=_f["meaning_questions"], m=_f[("en", "meaning", "meaning")], b=_f[("en", "bm25", "meaning")]),
            "names": "Daar is BM25 juist sterk: een zeldzaam woord dat in allebei staat, geeft een hoge score. Op de vragen die woorden met hun punt delen, was BM25 iets beter. Zie [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).",
            "long": "De lengte beslist het niet: de embedding van elke tekst heeft 384 getallen. Het gaat erom of de vraag en het punt hetzelfde betekenen. Zie [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).",
        },
    )
    quick_meaning
    return (quick_meaning,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    In [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl) zette zoeken op betekenis het goede punt bij de drie beste voor {m} van de {n} vragen, BM25 voor {b}. Maar ze misten niet dezelfde. *{q20}* ({q20_nl}) BM25 zette het punt op de eerste plaats, zoeken op betekenis pas op plaats {q20_m}. *{q18}* ({q18_nl}) Zoeken op betekenis zette het punt op de eerste plaats, BM25 op plaats {q18_b}. Namen, codes en getallen vind je aan hun woorden; hetzelfde idee in andere woorden vind je aan de betekenis. Een zoekfunctie die allebei gebruikt, kan allebei vinden.

    **Een voorbeeld uit het dagelijks leven.** Twee vrienden geven je elk hun tien favoriete restaurants, op volgorde. De een geeft sterren, de ander gaat op gevoel, dus wat ze zeggen kun je niet optellen; maar je ziet wel waar een restaurant op elke lijst staat, en daar kun je op afgaan. Dat is maar een vergelijking: de zoekfunctie hieronder voegt twee ranglijsten van punten samen.
    """.format(m=_f[("en", "meaning")], n=_f["questions"], b=_f[("en", "bm25")], q20=C.question(C.WORKED)["text"]["en"],
               q20_m=C.places(C.WORKED)["meaning"], q18=C.question(C.LOST)["text"]["en"], q18_b=C.places(C.LOST)["bm25"],
               q20_nl=C.question(C.WORKED)["text"]["nl"], q18_nl=C.question(C.LOST)["text"]["nl"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.worked()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **hybrid search** | zoeken op woorden, zoals met BM25, en op betekenis, en de twee ranglijsten samenvoegen tot één |
    | **rang** | de plaats van een tekst in een ranglijst: 1 voor de eerste, 2 voor de tweede |
    | **reciprocal rank fusion** (RRF) | ranglijsten samenvoegen door elke tekst de som van `1 / (k + rang)` over de ranglijsten te geven, en op die som te sorteren |
    | `k` | het getal dat RRF bij elke rang optelt; 60 is de gewone waarde |

    Waarom tel je de twee scores niet gewoon op? Ze liggen niet op dezelfde schaal. Voor de vraag van het uitgewerkte voorbeeld gaf BM25 de punten scores van {bl} tot {bh}, het inproduct van de embeddings van {ml} tot {mh}.

    Reciprocal rank fusion, in 2009 voorgesteld door Cormack, Clarke en Büttcher, kijkt alleen naar de plaatsen. Elke ranglijst geeft elke tekst `1 / (k + rang)`, en de delen tellen op. Met `k = 60` geeft de eerste plaats 1/61 = {p1}, de tiende 1/70 = {p10}. Een tekst die maar in één ranglijst staat, krijgt alleen dat ene deel. Elasticsearch, OpenSearch, Weaviate en Qdrant hebben hybrid search met RRF ingebouwd.
    """.format(bl=C.number(_w["bm25_low"], L.lang, 0), bh=C.number(_w["bm25_high"], L.lang, 2), ml=C.number(_w["meaning_low"], L.lang, 3),
               mh=C.number(_w["meaning_high"], L.lang, 3), p1=C.number(1 / 61, L.lang, 4), p10=C.number(1 / 70, L.lang, 4)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: hybrid search

    1. **Twee keer zoeken:** rangschik de punten met BM25, en op betekenis.
    2. **Score:** geef in elke ranglijst elk punt `1 / (k + rang)`, en tel de delen op.
    3. **Houden:** de `n` punten met de hoogste som, de hoogste eerst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{question}* ({question_nl}) De punten zijn in het Engels, en de course stelde de vraag ook in het Engels.

    1. **Twee keer zoeken:** BM25 zet als eerste een punt van *{w0}*: *{gold}* Het beantwoordt de vraag. Op betekenis zijn de drie beste punten uit de twee lessen over coding agents, modellen die voor je aan je code werken: *{m0}* en *{m1}*; het goede punt komt op plaats {gm}.
    2. **Score:** het goede punt krijgt 1/(60 + {gb}) + 1/(60 + {gm}) = {gs}. De eerste op betekenis staat bij BM25 op plaats {fb}: *{first}* Die krijgt 1/(60 + {fb}) + 1/(60 + {fm}) = {fs}. Een punt van *{h1}*, op plaats {hb} bij BM25 en {hm} op betekenis, krijgt {hs}.
    3. **Houden:** het goede punt eerst, met {gs}, dan het punt van *{h1}* en de eerste op betekenis.
    """.format(question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], w0=C.title_of(_w["words"][0], L.lang), gold=_w["gold"],
               m0=C.title_of(_w["meaning"][0], L.lang), m1=C.title_of(_w["meaning"][1], L.lang), gb=_w["gold_places"][0],
               gm=_w["gold_places"][1], gs=C.number(_w["gold_score"], L.lang, 4), first=_w["first"], fb=_w["first_places"][0],
               fm=_w["first_places"][1], fs=C.number(_w["first_score"], L.lang, 4), h1=C.title_of(_w["hybrid"][1], L.lang),
               hb=_w["hybrid_places"][1][0], hm=_w["hybrid_places"][1][1], hs=C.number(_w["hybrid_scores"][1], L.lang, 4)))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    Voor elk van de {n} vragen die een punt beantwoordt, voegde de course de ranglijst van BM25 van alle {points} punten en de ranglijst op betekenis samen, met `k = 60`, en keek of het goede punt bij de drie beste zat. Daarna deed hij hetzelfde met de vragen in het Nederlands, tegen dezelfde Engelse punten:
    """.format(n=C.findings()["questions"], points=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def picture_bars():
    mo.Html(C.bars_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        In het Engels gesteld won de samengevoegde zoekfunctie {won} vragen van zoeken op betekenis, zoals die van het uitgewerkte voorbeeld, en verloor er {lost}: twee vragen waarbij zoeken op betekenis het goede punt eerst zette en BM25 ver naar beneden, op plaats {b18} en {b21}.

        In het Nederlands gesteld vond BM25 het goede punt voor maar {b_nl} van de {n}: het kan alleen afgaan op de woorden die een Nederlandse vraag met een Engels punt deelt. De samengevoegde zoekfunctie vond er {h_nl}, minder dan de {m_nl} van zoeken op betekenis: de zwakke plaatsen van BM25 trokken goede punten omlaag. Met `k = 1` geeft de eerste plaats 1/2 en de tiende 1/11, dus telt een eerste plaats veel zwaarder. Dan vond de samengevoegde zoekfunctie er {s_nl} in het Nederlands, en {s_en} in het Engels, net zoveel als met `k = 60`.
        """.format(won=len(_f["won"]), lost=len(_f["lost"]), b18=C.places(_f["lost"][0])["bm25"], b21=C.places(_f["lost"][1])["bm25"],
                   b_nl=_f[("nl", "bm25")], n=_f["questions"], h_nl=_f[("nl", "hybrid")], m_nl=_f[("nl", "meaning")],
                   s_nl=_f[("nl", "small")], s_en=_f[("en", "small")])),
        L.watch_out("Samenvoegen is niet vanzelf beter: op dezelfde punten hielp het de vragen in het Engels en schaadde het die in het Nederlands. Meet het op je eigen vragen, gesteld zoals je gebruikers ze stellen. Als één zoekfunctie veel zwakker is, kan een kleinere `k`, of die zoekfunctie weglaten, beter werken."),
    ])
    return


@app.cell(hide_code=True)
def text_rrf():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Ranglijsten samenvoegen

        `rrf(rankings, k)` moet een dict geven met voor elke tekst in de ranglijsten de som van `1 / (k + rang)` over de ranglijsten waarin hij staat. `rankings` is een lijst van ranglijsten, elk een lijst teksten, de beste eerst. Het vak geeft elke tekst al de score 0. `rrf([["A", "B", "C"], ["C", "A", "D"]], 60)` moet, afgerond, `{"A": 0.0325, "B": 0.0161, "C": 0.0323, "D": 0.0159}` geven: `"A"` staat in de ene ranglijst eerst en in de andere tweede, dus krijgt hij 1/61 + 1/62. De check laat ook zien wat je functie de punten geeft voor de vraag van het uitgewerkte voorbeeld.
        """),
        L.task("Tel `1 / (k + rang)` op bij de score van een tekst, elke keer dat hij in een ranglijst staat, met rang 1 voor de eerste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rrf():
    box_rrf = L.code_box("rrf", lines=9)
    box_rrf
    return (box_rrf,)


@app.cell(hide_code=True)
def check_rrf(box_rrf):
    result_rrf = L.check("rrf", box_rrf.value)
    result_rrf
    return (result_rrf,)


@app.cell(hide_code=True)
def hints_rrf():
    L.hints(
        "rrf",
        [
            "Elke keer dat een tekst in een ranglijst staat, groeit zijn score met `1 / (k + rang)`. De rang van `ranking[place]` is één meer dan `place`: de eerste tekst heeft `place` 0 en rang 1.",
            "Zet één regel in de binnenste loop, na de `if` en even ver ingesprongen als de `if`, zodat hij voor elke tekst draait: `scores[text] = scores[text] + ...`.",
            "De nieuwe regel is `scores[text] = scores[text] + 1 / (k + ...)`, met `place + 1` als rang.",
        ],
        why="""```python
def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for place in range(len(ranking)):
            text = ranking[place]
            if text not in scores:
                scores[text] = 0
            scores[text] = scores[text] + 1 / (k + place + 1)
    return scores
```

Een tekst krijgt een deel van elke ranglijst waarin hij staat; een tekst in maar één ranglijst krijgt alleen dat deel.""",
    )
    return


@app.cell(hide_code=True)
def text_hybrid():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Met allebei zoeken

        `hybrid_search(question, q, n)` moet de `n` punten geven met de hoogste score van `rrf` over de ranglijst van BM25 en de ranglijst op betekenis, de hoogste eerst. `question` is de tekst van de vraag en `q` de embedding ervan, gemaakt door Granite; een checkbericht schrijft die als `QVECTORS[i]`, de embedding van vraag `i` uit de lijst van de course. Gegeven zijn:

        - `bm25_search(question, texts, k)`: de `k` teksten met de hoogste BM25-score, de hoogste eerst;
        - `meaning_search(q, vectors, points, k)`: de `k` punten waarvan de embeddings het hoogste inproduct met `q` hebben, zoals in [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl);
        - `rrf`: die van oefening 1;
        - `POINTS` en `VECTORS`: de {points} punten van de course en hun embeddings.

        Het vak rangschikt alle punten op beide manieren, en geeft de eerste van alleen zoeken op betekenis. Voor de vraag van het uitgewerkte voorbeeld moet `hybrid_search` de drie punten van stap 3 geven, het goede punt eerst. De check probeert ook andere vragen, en andere `n`.
        """.format(points=C.number(len(C.POINTS), L.lang))),
        L.task("Voeg `words` en `meaning` samen met `rrf`, `k = 60`, en geef de `n` punten met de hoogste score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hybrid():
    box_hybrid = L.code_box("hybrid_search", lines=7)
    box_hybrid
    return (box_hybrid,)


@app.cell(hide_code=True)
def check_hybrid(box_hybrid):
    result_hybrid = L.check("hybrid_search", box_hybrid.value)
    result_hybrid
    return (result_hybrid,)


@app.cell(hide_code=True)
def hints_hybrid():
    L.hints(
        "hybrid_search",
        [
            "Geef `rrf` beide ranglijsten, en neem dan de `n` teksten uit zijn dict met de hoogste score.",
            "`rrf([words, meaning], 60)` geeft een dict van tekst naar score. Een lijst paren `(score, text)`, gesorteerd met `reverse=True`, heeft de hoogste score eerst.",
            "Vervang de laatste regel door drie: `scores = rrf(...)`, dan `best = sorted([(score, text) for text, score in scores.items()], reverse=True)`, en geef dan de teksten van `best[:n]` terug.",
        ],
        why="""```python
def hybrid_search(question, q, n):
    words = bm25_search(question, POINTS, len(POINTS))
    meaning = meaning_search(q, VECTORS, POINTS, len(POINTS))
    scores = rrf([words, meaning], 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
```

Beide zoekfuncties rangschikken alle punten, dus elk punt krijgt een deel van allebei. Een dict houdt zijn teksten in de volgorde waarin ze erin kwamen, hier die van BM25; pas sorteren op de score geeft de samengevoegde volgorde.""",
    )
    return


@app.cell(hide_code=True)
def text_add():
    mo.md(r"""
    ### Oefening 3. De scores optellen?

    Een collega voegt de twee zoekfuncties anders samen: tel van elk punt de BM25-score en het inproduct op, en sorteer op de som.
    """)
    return


@app.cell(hide_code=True)
def choice_add_cell():
    _w = C.worked()
    _s = C.summed()
    _n = C.findings()["questions"]
    choice_add = L.choice(
        "add-scores",
        question="Welke volgorde geeft die som?",
        options={
            "bm25": "Bijna de eigen volgorde van BM25, want zijn scores zijn veel groter",
            "meaning": "Bijna de volgorde op betekenis, want zijn scores liggen dicht bij elkaar",
            "same": "Dezelfde volgorde als reciprocal rank fusion op beide lijsten geeft",
        },
        correct="bm25",
        why={
            "bm25": "Voor de vraag van het uitgewerkte voorbeeld gaf BM25 scores van 0 tot {bh}, het inproduct van {ml} tot {mh}: één kleine stap bij BM25 weegt zwaarder dan alles van betekenis. De course probeerde het: voor {same} van de {n} vragen waren de drie beste van de som precies de drie beste van BM25. Plaatsen liggen voor beide zoekfuncties op dezelfde schaal; daarom gebruikt RRF die.".format(
                bh=C.number(_w["bm25_high"], L.lang, 2), ml=C.number(_w["meaning_low"], L.lang, 3), mh=C.number(_w["meaning_high"], L.lang, 3),
                same=_s["same"], n=_n),
            "meaning": "Het is andersom: scores die dicht bij elkaar liggen, veranderen een som nauwelijks. De scores van BM25 liepen van 0 tot {bh}, dus die beslissen: voor {same} van de {n} vragen waren de drie beste van de som precies de drie beste van BM25.".format(
                bh=C.number(_w["bm25_high"], L.lang, 2), same=_s["same"], n=_n),
            "same": "RRF kijkt alleen naar de plaatsen, en die liggen voor beide zoekfuncties op dezelfde schaal. Een som wordt beslist door de grootste getallen, die van BM25: voor {same} van de {n} vragen waren de drie beste van de som precies de drie beste van BM25.".format(
                same=_s["same"], n=_n),
        },
    )
    choice_add
    return (choice_add,)


@app.cell(hide_code=True)
def text_first():
    _l = C.lost()
    mo.md(r"""
    ### Oefening 4. Niet op de eerste plaats

    *{question}* ({question_nl}) Zoeken op betekenis zette het goede punt eerst: *{gold}* BM25 zette het op plaats {b}, en de samengevoegde zoekfunctie, met `k = 60`, pas op plaats {h}. De samengevoegde eerste was een punt van *{t}*, op plaats {fb} bij BM25 en {fm} op betekenis: *{first}*
    """.format(question=C.question(C.LOST)["text"]["en"], question_nl=C.question(C.LOST)["text"]["nl"], gold=C.gold(C.question(C.LOST)), b=_l["places"]["bm25"], h=_l["places"]["hybrid"],
               t=C.title_of(_l["first"], L.lang), fb=_l["first_places"][0], fm=_l["first_places"][1], first=_l["first"]))
    return


@app.cell(hide_code=True)
def choice_first_cell():
    _l = C.lost()
    choice_first = L.choice(
        "first-place",
        question="Waarom zette de samengevoegde zoekfunctie dat punt boven het goede?",
        options={
            "bm25": "BM25 weegt zwaarder in de som, omdat zijn scores vele malen hoger zijn",
            "both": "Beide lijsten zetten het vrij hoog, en met k = 60 telt dat zwaarder",
            "count": "RRF telt hoe vaak een punt in de lijsten staat, en telt dat op",
        },
        correct="both",
        why={
            "both": "Met `k = 60` geeft een eerste plaats 1/61 en een achtste 1/68: niet veel minder. Plaats {fb} en {fm} gaven {fs}, plaats 1 en {b} gaven {s}. Met `k = 1` geeft een eerste plaats 1/2 en een achtste 1/9: dan kwam het goede punt op plaats {small}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4), small=_l["places"]["small"]),
            "bm25": "RRF ziet de scores van BM25 nooit, alleen de plaatsen: elke ranglijst telt `1 / (k + rang)` op. Plaats {fb} en {fm} gaven {fs}, plaats 1 en {b} gaven {s}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4)),
            "count": "Elk punt staat in beide ranglijsten, dus tellen zou ze allemaal hetzelfde geven. RRF telt `1 / (k + rang)` op: plaats {fb} en {fm} gaven {fs}, plaats 1 en {b} gaven {s}.".format(
                fb=_l["first_places"][0], fm=_l["first_places"][1], fs=C.number(_l["first_score"], L.lang, 4), b=_l["places"]["bm25"],
                s=C.number(_l["score"], L.lang, 4)),
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Doorzoek de course met je eigen hybrid_search

    Als oefening 1 en 2 goed zijn, kies je een vraag. De course doorzoekt zijn punten met je `hybrid_search` en je `rrf`, naast de drie beste van BM25 en van zoeken op betekenis, elk punt met zijn twee plaatsen. De vragen zijn in de taal van de les.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q23")["text"][L.lang], label="Vraag", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_search(box_hybrid, box_rrf, question_pick):
    _fn = C.learner_search(box_rrf.value, box_hybrid.value)
    if _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de zoekfunctie gebruikt je eigen `rrf` en `hybrid_search`.*")
    else:
        _out = mo.Html(C.searched(_fn, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De zoekfunctie in [Je eerste RAG](/course/open/build-rag-first?lang=nl) kan nu een hybrid search zijn: de samengevoegde punten gaan in de prompt, zoals in [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl). Een snelle zoekfunctie, op woorden, op betekenis of allebei, geeft een paar kandidaten; een latere les laat een model elk ervan samen met de vraag lezen, en ze in een betere volgorde zetten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_add, choice_first, result_hybrid, result_rrf):
    L.wrap_up(result_rrf, result_hybrid, choice_add, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
