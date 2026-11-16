# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-query-rewriting. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen herschrijvingen en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-query-rewriting", "nl")
    C = L.checks   # de punten van de course, de opgenomen herschrijvingen, passages en versies van Qwen3, hun embeddings, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    _h = C.hybrid_worked()
    quick_scale = L.choice(
        "quick-scale",
        question="Waarom voegt de samengevoegde zoekfunctie de twee ranglijsten samen, in plaats van de BM25-score van een punt bij het inproduct van de embeddings op te tellen?",
        options={
            "scale": "De twee scores hebben een andere schaal",
            "slow": "Scores optellen is trager dan ranglijsten samenvoegen",
            "sign": "Een BM25-score kan negatief zijn, een inproduct niet",
        },
        correct="scale",
        why={
            "scale": "Bij het uitgewerkte voorbeeld van die les gaf BM25 de punten van {bl} tot {bh}, het inproduct van {ml} tot {mh}: in een som beslist BM25 bijna alleen. RRF kijkt alleen naar de plaatsen.".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
            "slow": "Allebei zijn ze snel: een som is één optelling per punt. Het probleem is de schaal: BM25 gaf de punten van {bl} tot {bh}, het inproduct van {ml} tot {mh}. Zie [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl).".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
            "sign": "Andersom: een BM25-score komt nooit onder de 0, en een inproduct wel. Het probleem is de schaal: BM25 gaf de punten van {bl} tot {bh}, het inproduct van {ml} tot {mh}. Zie [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl).".format(
                bl=C.number(_h["bm25_low"], L.lang, 0), bh=C.number(_h["bm25_high"], L.lang, 0), ml=C.number(_h["meaning_low"], L.lang, 2), mh=C.number(_h["meaning_high"], L.lang, 2)),
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _w = C.worked()
    mo.md(r"""
    ## Waarom

    In een chat leunt de volgende vraag vaak op de vorige. *{earlier}* ({earlier_nl}) En dan: *{follow}* ({follow_nl}) Als je er alleen mee zoekt, zegt de vervolgvraag niet waar "its" naar verwijst. Zoeken op betekenis zette het goede punt bij de drie beste voor {full} van de {n} vragen over de course als ze helemaal gesteld werden, en voor {alone} als alleen de vervolgvraag gesteld werd. Een korte vraag, zoals mensen die in een zoekvak typen, deed het niet veel beter: {short}.

    **Een voorbeeld uit het dagelijks leven.** Je loopt een bibliotheek in en vraagt: "Hoeveel bladzijden heeft hij?" De bibliothecaris vraagt eerst welk boek je bedoelt, en gaat pas dan zoeken. Een zoekmachine kan niets terugvragen; iets moet het boek eerst in de vraag zetten.
    """.format(earlier=_w["earlier"], follow=_w["follow"], earlier_nl=_w["earlier_nl"], follow_nl=_w["follow_nl"], full=_f["full"], n=_f["questions"], alone=_f["follow"], short=_f["short"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **query transformation** | de vraag veranderen vóór het zoeken, vaak door een model de tekst te laten schrijven waarmee je zoekt |
    | **standalone question** | een vervolgvraag die opnieuw geschreven is zodat hij zelf zegt waar hij over gaat, met de namen waar hij naar verwijst |
    | **HyDE** | hypothetical document embeddings: laat het model een passage schrijven die de vraag zou kunnen beantwoorden, en zoek met de embedding van die passage |
    | **multi-query retrieval** | laat het model andere versies van de vraag schrijven, zoek met elke versie, en voeg de ranglijsten samen, hier met RRF |

    Een zoekfunctie vergelijkt de vraag met de punten. Een vervolgvraag laat weg waar hij over gaat, en een korte vraag zegt weinig, dus schrijft het model eerst een betere tekst om mee te zoeken. Bij een vervolgvraag is dat de vraag op zichzelf: het model krijgt de chat, en schrijft de laatste vraag opnieuw. HyDE, in 2022 voorgesteld door Gao, Ma, Lin en Callan, begint aan de andere kant: een passage die antwoordt, lijkt meer op het goede punt dan de vraag, ook als de details verzonnen zijn. Multi-query retrieval vraagt hetzelfde op meer manieren, zodat een punt dat bij de meeste hoog komt, stijgt; RAG-Fusion voegt zulke ranglijsten samen met RRF, zoals deze les doet. Een vraag met twee delen kun je ook splitsen, met een zoekfunctie per deel.

    Elk hiervan kost een aanroep van het model voordat het zoeken kan beginnen, en het model schrijft vanuit wat het weet. Of het zoeken beter wordt, zie je alleen door te meten.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: vraag het anders

    1. **Kijken:** wat is er mis met de vraag zoals hij binnenkomt: leunt hij op de chat, of is hij kort?
    2. **Schrijven:** laat het model de tekst schrijven waarmee je zoekt: de vraag op zichzelf, een passage die antwoordt, of versies van de vraag.
    3. **Zoeken:** zoek met wat het model schreef; voeg bij meer versies de ranglijsten samen met RRF.
    4. **Meten:** vergelijk met zoeken op de vraag zoals hij binnenkwam, op vragen waarvan je het goede punt kent.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        1. **Kijken:** *{follow}* ({follow_nl}) leunt op de vraag ervoor: *{earlier}* ({earlier_nl}) De punten zijn in het Engels, en de course stelde de vragen ook in het Engels. Als je alleen met de vervolgvraag zoekt, staat het goede punt op plaats {alone}; eerst komt een punt uit *{first}*: *{first_text}*
        2. **Schrijven:** de course stuurt Qwen3 de chat, met een system prompt die om de laatste vraag op zichzelf vraagt.
        """.format(follow=_w["follow"], earlier=_w["earlier"], follow_nl=_w["follow_nl"], earlier_nl=_w["earlier_nl"], alone=_w["places"]["follow"], first=C.title_of(_w["first"], L.lang), first_text=_w["first"])),
        mo.md(C.request_code(C.WORKED)),
        mo.md(r"""
        Met `"temperature": 0` neemt het model elke keer het waarschijnlijkste token, dus het schrijft bij elke run hetzelfde. Qwen3 schreef: *{rewrite}*

        3. **Zoeken:** daarmee zet zoeken op betekenis het goede punt op plaats {rewritten}: *{gold}*
        4. **Meten:** één vraag bewijst weinig. De course deed hetzelfde voor alle {n} vragen die een punt beantwoordt.
        """.format(rewrite=_w["rewrite"], rewritten=_w["places"]["rewrite"], gold=_w["gold"], n=C.findings()["questions"])),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    Voor elk van de {n} vragen die een punt beantwoordt, schreef de course een vervolgvraag na een eerdere vraag, en een korte vraag zoals mensen die in een zoekvak typen. Qwen3 herschreef elke vervolgvraag, en schreef bij elke korte vraag een passage (HyDE) en drie versies. Zoeken op betekenis, met de embeddings van Granite, zocht met elk ervan:
    """.format(n=_f["questions"]))
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
        De herschrijving van Qwen3 tilde de vervolgvragen van {follow} naar {rewrite}. Maar de vervolgvraag gewoon na de eerdere vraag, zonder het model, haalde er {joined}. Bij {copied} van de {n} herschrijvingen kopieerde Qwen3 alleen de twee vragen achter elkaar, en waar het echt een nieuwe vraag schreef, liet het soms weg waar de vraag over ging.

        Bij de korte vragen maakte Qwen3 het slechter: zijn passages haalden er {hyde}, de korte vraag met zijn drie versies {multi}, tegen {short} voor de korte vraag alleen. Qwen3 weet weinig van deze course, dus wat het schreef, dreef af naar AI in het algemeen.

        Elke manier kostte ook een aanroep van het model vóór het zoeken: op de computer van de course duurde een herschrijving ongeveer {s_rewrite} seconden, een passage {s_hyde}, drie versies {s_versions}.
        """.format(follow=_f["follow"], rewrite=_f["rewrite"], joined=_f["joined"], copied=len(_f["copied"]), n=_f["questions"], hyde=_f["hyde"],
                   multi=_f["multi"], short=_f["short"], s_rewrite=C.number(_f["seconds_rewrite"], L.lang, 1),
                   s_hyde=C.number(_f["seconds_hyde"], L.lang, 1), s_versions=C.number(_f["seconds_versions"], L.lang, 1))),
        L.watch_out("Query transformation kan het zoeken slechter maken, en een klein model dat je onderwerp niet kent, doet dat makkelijk. Meet het op je eigen vragen tegen de vraag zoals hij binnenkwam, en probeer eerst het goedkoopste: hier was dat de eerdere vraag vóór de vervolgvraag."),
    ])
    return


@app.cell(hide_code=True)
def text_versions():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De versies lezen

        `versions(text)` krijgt wat het model schreef toen het om andere versies van een vraag gevraagd werd, en moet elke versie als een eigen tekst geven. Modellen schrijven zo'n lijst als regels die met `- ` beginnen, of genummerd als `1.`, `2.` en `3.`, vaak met spaties eromheen, en soms met lege regels ertussen. Qwen3 schreef dit bij de korte vraag *{short}*:

        `{raw}`

        Daarvoor moet `versions` drie vragen geven, zonder de `- ` en de spaties. De check probeert ook een genummerde lijst, lege regels en regels zonder `- `.
        """.format(short=C.SHORT[C.VERSIONS_SHOWN], raw=repr(C.written(C.VERSIONS_SHOWN, "versions")))),
        L.task("Voor elke regel: strip hem, knip een `- ` vooraan of een cijfer met een punt eraf, strip hem opnieuw, en voeg hem alleen aan `found` toe als hij niet leeg is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_versions():
    box_versions = L.code_box("versions", lines=12)
    box_versions
    return (box_versions,)


@app.cell(hide_code=True)
def check_versions(box_versions):
    result_versions = L.check("versions", box_versions.value)
    result_versions
    return (result_versions,)


@app.cell(hide_code=True)
def hints_versions():
    L.hints(
        "versions",
        [
            "Maak elke regel in drie stappen schoon: eerst de spaties eromheen, dan wat de lijst ervoor zet, dan de spaties daarna. Een lege regel eindigt als `\"\"`.",
            "`line.startswith(\"- \")` zegt of de regel met een streepje begint, en `line[2:]` knipt de eerste twee tekens eraf. Voor een nummer: `line[0].isdigit() and line[1] == \".\"`, als je weet dat de regel minstens twee tekens heeft.",
            "In de loop: `line = line.strip()`, dan `if line.startswith(\"- \"):` met `line = line[2:]`, een `elif len(line) > 1 and line[0].isdigit() and line[1] == \".\":` met hetzelfde, dan opnieuw `line = line.strip()`, en `if line != \"\":` met `found.append(line)`.",
        ],
        why="""```python
def versions(text):
    found = []
    for line in text.split("\\n"):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        elif len(line) > 1 and line[0].isdigit() and line[1] == ".":
            line = line[2:]
        line = line.strip()
        if line != "":
            found.append(line)
    return found
```

Wat een model schrijft, is tekst, geen lijst. Wat het voor een regel zet en de spaties eromheen verschillen per model en per antwoord, dus de code die het leest, moet daar tegen kunnen.""",
    )
    return


@app.cell(hide_code=True)
def text_multi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Zoeken met elke versie

        `multi_search(texts, n)` krijgt meer teksten die hetzelfde vragen, en moet de `n` punten geven met de hoogste RRF-score over hun ranglijsten, de hoogste eerst. Gegeven zijn:

        - `embed(text)`: de embedding van Granite van een tekst, uit de opgenomen embeddings van de course;
        - `meaning_search(q, vectors, points, k)`: die van [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl);
        - `rrf(rankings, k)`: die van [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl);
        - `POINTS` en `VECTORS`: de punten van de course en hun embeddings.

        De code maakt voor elke tekst een ranglijst van alle punten, maar geeft de eerste `n` van de eerste ranglijst. `QUERIES["{qid}"]` is de korte vraag *{short}* met de drie versies van Qwen3; daarvoor moet `multi_search` het goede punt eerst geven. De check probeert ook andere vragen, een andere `n`, twee teksten en één.
        """.format(qid=C.MULTI_SHOWN, short=C.SHORT[C.MULTI_SHOWN])),
        L.task("Voeg de ranglijsten samen met `rrf(rankings, 60)`, sorteer de punten op hun score met de hoogste eerst, en geef de eerste `n`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_multi():
    box_multi = L.code_box("multi_search", lines=8)
    box_multi
    return (box_multi,)


@app.cell(hide_code=True)
def check_multi(box_multi):
    result_multi = L.check("multi_search", box_multi.value)
    result_multi
    return (result_multi,)


@app.cell(hide_code=True)
def hints_multi():
    _f = C.findings()
    L.hints(
        "multi_search",
        [
            "Het is weer het samenvoegen van [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl), nu met een ranglijst voor elke tekst in plaats van een voor woorden en een voor betekenis.",
            "`scores = rrf(rankings, 60)` geeft een dict van punt en score. Paren `(score, text)` sorteren op de score.",
            "Vervang de laatste regel door drie: `scores = rrf(rankings, 60)`, dan `best = sorted([(score, text) for text, score in scores.items()], reverse=True)`, en geef `[text for score, text in best[:n]]` terug.",
        ],
        why="""```python
def multi_search(texts, n):
    rankings = []
    for text in texts:
        rankings.append(meaning_search(embed(text), VECTORS, POINTS, len(POINTS)))
    scores = rrf(rankings, 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
```

Een punt dat bij meer versies hoog komt, krijgt een hoge score, ook als geen enkele versie het eerst zet. Of dat helpt, hangt af van de versies: met die van Qwen3 kwam het goede punt hoger dan met de korte vraag alleen bij {better} vragen, en lager bij {worse}.""".format(better=_f["multi_better"], worse=_f["multi_worse"]),
    )
    return


@app.cell(hide_code=True)
def text_drift():
    _d = C.places(C.DRIFT)
    mo.md(r"""
    ### Oefening 3. Een passage die op het verkeerde spoor zet

    Bij de korte vraag *{short}* schreef Qwen3 deze passage om mee te zoeken: *{passage}* De zoekfunctie zette het goede punt op plaats {hyde}; met de korte vraag zelf kwam het op plaats {alone}: *{gold}*
    """.format(short=C.SHORT[C.DRIFT], passage=C.written(C.DRIFT, "hyde"), hyde=C.number(_d["hyde"], L.lang), alone=_d["short"],
               gold=C.gold(C.question(C.DRIFT))))
    return


@app.cell(hide_code=True)
def choice_drift_cell():
    _d = C.places(C.DRIFT)
    choice_drift = L.choice(
        "hyde-drift",
        question="Waarom stuurde de passage de zoekfunctie zo ver weg?",
        options={
            "long": "Twee zinnen zijn te lang: de embedding van een lange tekst zegt minder",
            "unknown": "Qwen3 kende deze course niet en schreef over iets anders",
            "words": "Zoek met de passage via BM25, want de woorden zijn zeldzaam",
        },
        correct="unknown",
        why={
            "unknown": "Niets in de passage gaat over een opening of over spikkels: Qwen3 schreef over AI in het algemeen. HyDE schrijft vanuit wat het model weet, dus het helpt alleen als het model het onderwerp goed genoeg kent om iets te schrijven dat dicht bij het antwoord ligt.",
            "long": "De punten van de course zijn ongeveer even lang, en Granite maakt er goede embeddings van. Lees wat Qwen3 schreef: niets erin gaat over een opening of over spikkels. Qwen3 kent deze course niet, dus het schreef over AI in het algemeen.",
            "words": "BM25 zou het ook niet vinden: de passage deelt bijna geen woord met het goede punt. Niets erin gaat over een opening of over spikkels. Qwen3 kent deze course niet, dus het schreef over AI in het algemeen.",
        },
    )
    choice_drift
    return (choice_drift,)


@app.cell(hide_code=True)
def text_lost():
    _l = C.places(C.LOST)
    mo.md(r"""
    ### Oefening 4. Een herschrijving die iets kwijtraakte

    *{earlier}* ({earlier_nl}) En dan: *{follow}* ({follow_nl}) Qwen3 herschreef de vervolgvraag als *{rewrite}* Daarmee zette de zoekfunctie het goede punt op plaats {rewritten}; de vervolgvraag na de eerdere vraag vond het op plaats {joined}.
    """.format(earlier=C.FOLLOW[C.LOST][0], follow=C.FOLLOW[C.LOST][1], earlier_nl=C.FOLLOW_NL[C.LOST][0], follow_nl=C.FOLLOW_NL[C.LOST][1], rewrite=C.written(C.LOST, "rewrite"), rewritten=_l["rewrite"], joined=_l["joined"]))
    return


@app.cell(hide_code=True)
def choice_lost_cell():
    _l = C.places(C.LOST)
    choice_lost = L.choice(
        "lost-subject",
        question="Wat ging er mis in de herschrijving?",
        options={
            "rare": "De woorden log scale zijn te zeldzaam voor zoeken op betekenis",
            "answer": "Hij had ook het antwoord op de eerdere vraag moeten bevatten",
            "lost": "Hij liet weg wat de kromme is: loss tegen grootte",
        },
        correct="lost",
        why={
            "lost": "De herschrijving zegt the curve, maar niet waarvan. De vervolgvraag na de eerdere vraag had de loss en de grootte nog, en vond het goede punt op plaats {joined}. Een herschrijving moet binnenhalen waar de vervolgvraag naar terugwijst: kijk na of hij dat doet, op je eigen chats.".format(joined=_l["joined"]),
            "rare": "De vervolgvraag na de eerdere vraag heeft dezelfde woorden, en vond het goede punt op plaats {joined}. Vergelijk de herschrijving met de eerdere vraag: de loss en de grootte zijn weg.".format(joined=_l["joined"]),
            "answer": "De zoekfunctie moet weten waar de vraag over gaat, niet wat het antwoord is: de vervolgvraag na de eerdere vraag vond het goede punt op plaats {joined}. Vergelijk de herschrijving met de eerdere vraag: de loss en de grootte zijn weg.".format(joined=_l["joined"]),
        },
    )
    choice_lost
    return (choice_lost,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Zes manieren van vragen

    Zodra oefening 1 en 2 goed zijn, kies je een vraag. Je ziet wat de zoekfunctie kreeg bij elke manier van vragen, en waar hij het goede punt zette; de laatste rij leest de versies van Qwen3 met jouw `versions` en zoekt met jouw `multi_search`. De course vroeg alles in het Engels.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q18")["text"][L.lang], label="Vraag", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_ways(box_multi, box_versions, question_pick):
    _fns = C.learner_multi(box_versions.value, box_multi.value)
    if _fns is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de laatste rij gebruikt je eigen `versions` en `multi_search`.*")
    else:
        _out = mo.Html(C.tried(_fns, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een RAG-systeem in een chat heeft de eerdere vragen nodig bij het zoeken, zoals hier, en de reranker van [Een tweede blik: reranking](/course/open/build-reranking?lang=nl) kan daarna ordenen wat terugkomt. Of een verandering helpt, weet je alleen door te meten; een latere les meet een RAG-systeem in twee delen, het zoeken en het antwoord.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_drift, choice_lost, result_multi, result_versions):
    L.wrap_up(result_versions, result_multi, choice_drift, choice_lost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
