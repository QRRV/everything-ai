# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-rag-eval. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen oordelen en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-eval", "nl")
    C = L.checks   # het RAG-systeem van de course, zijn opgenomen antwoorden en de oordelen van Qwen3 erover, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="Wanneer geeft supported, de check van die les, True voor een antwoord?",
        options={
            "right": "Als er het getal in staat waar de vraag om vraagt, uit het goede punt",
            "cited": "Als het bronnen noemt en elk getal erin in die bronnen staat",
            "words": "Als de meeste woorden uit de bronnen komen die het noemt",
        },
        correct="cited",
        why={
            "cited": "Alleen dat kijkt hij na: minstens één genoemde bron, alleen bronnen die er zijn, en geen getal dat er niet in staat. Of een getal de vraag beantwoordt, ziet hij niet.",
            "right": "De check kent het goede antwoord niet: hij kijkt alleen naar de bronnen die het antwoord noemt, en of elk getal van het antwoord erin staat. Zie [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).",
            "words": "Hij telt geen woorden: hij kijkt naar de bronnen die het antwoord noemt, en of elk getal van het antwoord erin staat. Zie [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een RAG-systeem geeft een fout antwoord. Dan weet je dat er iets mis is, maar niet wat je moet verbeteren. Misschien bracht de zoekfunctie het punt dat de vraag beantwoordt niet terug; dan helpt geen enkele prompt. Misschien was het punt er wel en haalde het antwoord er iets anders uit; dan verandert een betere zoekfunctie niets. Een RAG-systeem heeft twee delen, en elk deel heeft een eigen meting nodig.

    **Een voorbeeld uit het dagelijks leven.** Een gerecht smaakt niet goed. Voordat je het recept verandert, proef je wat erin ging: als de tomaten zuur waren, helpt anders koken niet. Waren ze goed, dan zit de fout in het koken.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **context recall** | het deel van de relevante punten dat de zoekfunctie terugbracht; hier heeft elke vraag één goed punt, dus een vraag scoort 1 of 0 |
    | **context precision** | hoe hoog de relevante punten kwamen tussen wat de zoekfunctie terugbracht: bij elk relevant punt het deel relevante punten tot die plaats, gemiddeld; één goed punt op plaats r geeft 1/r |
    | **groundedness** | of de bronnen zeggen wat het antwoord zegt; een grounded antwoord kan toch fout zijn, als de bronnen fout zijn |
    | **answer relevance** | of het antwoord ingaat op de vraag, goed of fout |

    De eerste twee meten het zoeken. Ze moeten weten welke punten relevant zijn, dus je hebt een set vragen met hun goede punten nodig, zoals de course die heeft voor zijn 24 vragen. De andere twee meten het antwoord. Code kan ze een score geven, zoals `supported` van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) dat doet voor groundedness, of een model kan het: een LLM-as-a-judge, die je vraagt of de bronnen zeggen wat het antwoord zegt, en of het de vraag beantwoordt. Een latere les gaat dieper in op judges.

    Ragas, een library om een RAG-systeem te meten, berekent deze vier, de meeste met een LLM als judge. Het noemt groundedness faithfulness; deze course gebruikt dat woord alleen voor de vraag of de tussenstappen van een model de echte reden voor zijn antwoord zijn, zoals in [Hoe een LLM redeneert](/course/open/llm-reasoning-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een RAG-systeem meten

    1. **Verzamelen:** vragen met de punten die ze beantwoorden, en een manier om een antwoord na te kijken.
    2. **Zoeken:** context recall en context precision voor elke vraag, gemiddeld over de vragen.
    3. **Antwoord:** voor elk antwoord: is het goed, grounded en relevant?
    4. **Splitsen:** een fout antwoord met het goede punt bij zijn bronnen is een fout van de stap die antwoordt; een zonder, van het zoeken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _q = C.question(C.WORKED)
    _found = C.found(_q)
    _a = C.answers(C.WORKED)[0]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    1. **Verzamelen:** *{question}* ({question_nl}) Het goede punt: *{gold}*
    2. **Zoeken:** zoeken op betekenis bracht een punt van *{t1}* terug, een van *{t2}*, en het goede punt als derde. Context recall: 1 van de 1 relevante punten kwam terug, dus 1. Context precision: bij het goede punt, plaats 3, is 1 van de 3 punten relevant: 1/3 = {p}. De reranker van [Een tweede blik: reranking](/course/open/build-reranking?lang=nl) zet het goede punt eerst: precision 1, recall nog steeds 1.
    3. **Antwoord:** Qwen3 schreef *{answer}* De check van de course, die het goede getal kent, zegt dat het goed is; `supported` geeft `True`, want {n1} en {n2} staan in bron 3. Qwen3 als judge gaf het een `P(yes)` van {g} voor groundedness en {r} voor relevance.
    4. **Splitsen:** niets te splitsen: de zoekfunctie vond het punt, en het antwoord gebruikte het.
    """.format(question=_q["text"]["en"], question_nl=_q["text"]["nl"], gold=C.gold(_q), t1=C.title_of(_found[0], L.lang), t2=C.title_of(_found[1], L.lang),
               p=C.number(1 / 3, L.lang, 2), answer=_a, n1=C.number(C.worked_numbers()[0], L.lang), n2=C.number(C.worked_numbers()[1], L.lang),
               g=C.number(C.p_yes(C.WORKED, 0, "grounded"), L.lang, 3), r=C.number(C.p_yes(C.WORKED, 0, "relevant"), L.lang, 3)))
    return


@app.cell(hide_code=True)
def text_measured():
    _m, _r = C.scores("meaning"), C.scores("reranked")
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    De course mat het RAG-systeem van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl): zoeken op betekenis geeft drie punten, en Qwen3 antwoordt ermee na een grounded system prompt, twee keer voor elk van de {n} vragen die een punt beantwoordt. Gemiddeld over de vragen had de zoekfunctie een context recall van {rm} en een context precision van {pm}; met de reranker in de plaats daarvan {rr} en {pr}. Dan de {answers} antwoorden:
    """.format(n=len(C.ANSWERABLE), rm=C.number(_m["recall"], L.lang, 2), pm=C.number(_m["precision"], L.lang, 2),
               rr=C.number(_r["recall"], L.lang, 2), pr=C.number(_r["precision"], L.lang, 2), answers=_f["answers"]))
    return


@app.cell(hide_code=True)
def picture_answers():
    mo.Html(C.answers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_split():
    _f = C.findings()
    mo.md(r"""
    {right} van de {answers} antwoorden waren goed. Van de {wrong} foute hadden er {ws} geen goed punt bij hun bronnen: die verloor de zoekfunctie. De andere {wa} hadden het wel, en gingen toch mis: die zijn voor de stap die antwoordt. De {mr} goede antwoorden zonder het goede punt haalden hun getal uit een ander punt dat het ook zegt. De course kent één goed punt per vraag, dus context recall telt die zoekopdracht als een misser.

    `supported` gaf `True` voor {supported} van de {answers} antwoorden, en alle {wsup} foute zaten daarbij: hij ziet alleen of de getallen van een antwoord in zijn bronnen staan. Qwen3 als judge, gevraagd zoals de reranker, om één token met logprobs, schreef yes bij alle {judged} oordelen. Zijn `P(yes)`:
    """.format(right=_f["right"], answers=_f["answers"], wrong=_f["wrong"], ws=_f["wrong_search"], wa=_f["wrong_answer"], mr=_f["miss_right"],
               supported=_f["supported"], wsup=_f["wrong_supported"], judged=_f["judged"]))
    return


@app.cell(hide_code=True)
def picture_judge():
    mo.Html(C.judge_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_judge():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Voor groundedness kreeg elk antwoord minstens {gl}. Bij een goed en een fout antwoord kreeg het goede de hoogste score in {gp}% van de paren: ongeveer zo vaak als kop of munt. Voor relevance was het {rp}%, en de laagste score ging naar een antwoord dat alleen *{low}* zei.
        """.format(gl=C.number(_f["grounded_low"], L.lang, 2), gp=round(100 * _f["grounded_pairs"]), rp=round(100 * _f["relevant_pairs"]), low=_f["relevant_lowest"])),
        L.watch_out("Ook een judge is een meting. Voordat je zijn scores vertrouwt, vergelijk je ze met antwoorden waarvan je het oordeel kent, zoals de course hier deed. Ragas geeft ook scores met een LLM: zijn getallen zijn zo goed als zijn judge."),
    ])
    return


@app.cell(hide_code=True)
def text_recall():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Context recall

        `context_recall(retrieved, relevant)` krijgt de punten die de zoekfunctie terugbracht en de relevante punten, en moet het deel van de relevante punten geven dat terugkwam. `FOUND["q01"]` zijn de drie punten die zoeken op betekenis vond voor *{q}*, en `RELEVANT["q01"]` een lijst met het goede punt. Daar miste de zoekfunctie het, dus `context_recall` moet `0.0` geven. De check probeert ook andere vragen, en lijsten van letters met meer dan één relevant punt.
        """.format(q=C.question("q01")["text"]["en"])),
        L.task("Tel een punt van `relevant` alleen als het in `retrieved` staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recall():
    box_recall = L.code_box("context_recall", lines=7)
    box_recall
    return (box_recall,)


@app.cell(hide_code=True)
def check_recall(box_recall):
    result_recall = L.check("context_recall", box_recall.value)
    result_recall
    return (result_recall,)


@app.cell(hide_code=True)
def hints_recall():
    L.hints(
        "context_recall",
        [
            "Context recall kijkt vanaf de kant van de relevante punten: welke ervan zitten bij de punten die terugkwamen?",
            "`in` zegt of een lijst iets bevat: `x in items` is `True` als `x` een van de items is. Tel alleen de punten die terugkwamen.",
            "Zet de regel `found = found + 1` onder een `if ___:`, één stap verder ingesprongen.",
        ],
        why="""```python
def context_recall(retrieved, relevant):
    found = 0
    for point in relevant:
        if point in retrieved:
            found = found + 1
    return found / len(relevant)
```

Context recall zegt of de stap die antwoordt een kans had: een punt dat niet terugkwam, kan niet in het antwoord terechtkomen.""",
    )
    return


@app.cell(hide_code=True)
def text_precision():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Context precision

        `context_precision(retrieved, relevant)` moet zeggen hoe hoog de relevante punten kwamen. Loop `retrieved` op volgorde langs. Neem bij elk relevant punt de precision tot die plaats: het aantal relevante punten tot nu toe, gedeeld door de plaats. Geef het gemiddelde daarvan, of `0.0` als er geen relevant punt terugkwam. In `FOUND["q05"]` staat het goede punt tweede: 1/2 = 0,5. Met de relevante punten `["c", "a"]` in `["a", "b", "c"]`: (1/1 + 2/3) / 2 = {two}. De check probeert ook een vraag waarbij de zoekfunctie miste, en andere lijsten.
        """.format(two=C.number((1 + 2 / 3) / 2, L.lang, 2))),
        L.task("Tel bij elk relevant punt de precision tot die plaats op bij `total`; geef aan het eind het gemiddelde over de relevante punten, of `0.0` als er geen terugkwam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("context_precision", lines=11)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("context_precision", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "context_precision",
        [
            "De plaats van `retrieved[i]` is `i + 1`. Bij een relevant punt zijn er tot die plaats `hits` relevante punten teruggekomen.",
            "Binnen de `if`, na `hits = hits + 1`: `total = total + hits / (i + 1)`.",
            "Vervang de laatste regel door een `if hits == 0:` met `return 0.0` eronder, en daarna `return total / ___`.",
        ],
        why="""```python
def context_precision(retrieved, relevant):
    total = 0
    hits = 0
    for i in range(len(retrieved)):
        if retrieved[i] in relevant:
            hits = hits + 1
            total = total + hits / (i + 1)
    if hits == 0:
        return 0.0
    return total / hits
```

Met één goed punt op plaats r is context precision 1/r. De reranker tilde het goede punt van het uitgewerkte voorbeeld van plaats 3 naar plaats 1: de precision ging van {third} naar 1, en de recall bleef 1.""".format(third=C.number(1 / 3, L.lang, 2)),
    )
    return


@app.cell(hide_code=True)
def text_diagnose():
    _qid, _s = C.DIAGNOSED
    _q = C.question(_qid)
    mo.md(r"""
    ### Oefening 3. Welk deel verbeter je?

    *{question}* ({question_nl}) De zoekfunctie zette het goede punt eerst: *{gold}* Qwen3 antwoordde: *{answer}* `supported` gaf er `True` voor.
    """.format(question=_q["text"]["en"], question_nl=_q["text"]["nl"], gold=C.gold(_q), answer=C.answers(_qid)[_s]))
    return


@app.cell(hide_code=True)
def choice_diagnose_cell():
    _f = C.findings()
    choice_diagnose = L.choice(
        "diagnose",
        question="Wat verbeter je voor deze vraag?",
        options={
            "answer": "De stap die antwoordt: verkeerd getal gekozen",
            "search": "Het zoeken: een beter punt had eerst moeten komen",
            "check": "Niets: supported gaf True, dus het antwoord is grounded",
        },
        correct="answer",
        why={
            "answer": "Context recall en context precision waren hier allebei 1: de zoekfunctie deed zijn deel. De 14 staat ook in het punt, maar als de pixels die een closing fout laat. Van de {wrong} foute antwoorden hadden er {wa} het goede punt bij hun bronnen, zoals dit.".format(
                wrong=_f["wrong"], wa=_f["wrong_answer"]),
            "search": "De zoekfunctie zette het goede punt eerst: context recall 1, context precision 1. De stap die antwoordt nam er het verkeerde getal uit: 14 is het aantal pixels dat een closing fout laat.",
            "check": "`supported` ziet alleen dat elk getal van het antwoord in een genoemde bron staat, en 14 staat erin, als de pixels die een closing fout laat. Hij gaf `True` voor alle {wsup} foute antwoorden.".format(
                wsup=_f["wrong_supported"]),
        },
    )
    choice_diagnose
    return (choice_diagnose,)


@app.cell(hide_code=True)
def text_echo():
    _qid, _s = C.ECHO
    mo.md(r"""
    ### Oefening 4. Een judge die ja zegt

    *{question}* ({question_nl}) Eén antwoord van het RAG-systeem van de course was: *{answer}* Het herhaalt alleen de vraag. Qwen3 als judge gaf het een `P(yes)` van {r} voor "beantwoordt het antwoord de vraag?"
    """.format(question=C.question(_qid)["text"]["en"], question_nl=C.question(_qid)["text"]["nl"], answer=C.answers(_qid)[_s], r=C.number(C.p_yes(_qid, _s, "relevant"), L.lang, 3)))
    return


@app.cell(hide_code=True)
def choice_judge_cell():
    _f = C.findings()
    choice_judge = L.choice(
        "judge",
        question="Wat leer je daaruit?",
        options={
            "sure": "De antwoorden zijn goed: de judge is van bijna allemaal zeker",
            "words": "Vraag de judge om een geschreven yes of no in plaats van de kans",
            "measure": "Ook een judge moet je meten, op nagekeken antwoorden",
        },
        correct="measure",
        why={
            "measure": "Op de {n} antwoorden die de course nakeek, kreeg een goed antwoord een hogere score dan een fout in {gp}% van de paren voor groundedness en {rp}% voor relevance. Kijk een judge na op antwoorden waarvan je het oordeel kent voordat je hem vertrouwt; een groter model oordeelt misschien beter, maar ook dat meet je.".format(
                n=_f["answers"], gp=round(100 * _f["grounded_pairs"]), rp=round(100 * _f["relevant_pairs"])),
            "sure": "De judge was net zo zeker van de foute antwoorden: voor groundedness kreeg elk antwoord minstens {gl}, en een goed antwoord won maar in {gp}% van de paren van een fout.".format(
                gl=C.number(_f["grounded_low"], L.lang, 2), gp=round(100 * _f["grounded_pairs"])),
            "words": "Qwen3 schreef yes bij alle {judged} oordelen: het geschreven woord zegt nog minder dan de kans.".format(judged=_f["judged"]),
        },
    )
    choice_judge
    return (choice_judge,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Meet het RAG-systeem zelf

    Zodra oefening 1 en 2 goed zijn, kies je een vraag. Je ziet de drie punten van beide zoekfuncties met jouw context recall en precision, en de twee antwoorden van het RAG-systeem van de course met wat de checks en de judge zeiden. De course vroeg alles in het Engels.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q20")["text"][L.lang], label="Vraag", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_measure(box_precision, box_recall, question_pick):
    _fns = C.learner_metrics(box_recall.value, box_precision.value)
    if _fns is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de getallen van het zoeken zijn die van je eigen `context_recall` en `context_precision`.*")
    else:
        _out = mo.Html(C.looked_into(_fns, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Met deze getallen kun je elke verandering aan een RAG-systeem wegen, zoals een andere zoekfunctie, de reranker van [Een tweede blik: reranking](/course/open/build-reranking?lang=nl) of een andere prompt, en zien welk deel ze verbetert. `supported` van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) is de goedkoopste judge; een model als judge kan naar meer kijken dan getallen, als je het eerst gemeten hebt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_diagnose, choice_judge, result_precision, result_recall):
    L.wrap_up(result_recall, result_precision, choice_diagnose, choice_judge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
