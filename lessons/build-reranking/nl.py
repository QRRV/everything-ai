# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-reranking. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen oordelen en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-reranking", "nl")
    C = L.checks   # de punten van de course, de samengevoegde zoekfunctie, de opgenomen oordelen van Qwen3, de checks en de plaatjes


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
def quick_check_rrf():
    quick_rrf = L.choice(
        "quick-rrf",
        question="Reciprocal rank fusion met k = 60 voegt twee ranglijsten samen. Punt A staat in de ene eerste en in de andere tiende; punt B staat in allebei derde. Welk punt krijgt de hoogste score?",
        options={
            "a": "A, want een eerste plaats telt het zwaarst",
            "b": "B, want twee derde plaatsen leveren samen meer op",
            "same": "Geen van beide: ze staan allebei gemiddeld rond plaats drie",
        },
        correct="b",
        why={
            "b": "A krijgt 1/61 + 1/70 = {a}, B krijgt 1/63 + 1/63 = {b}. Met k = 60 is een eerste plaats weinig meer waard dan een derde, dus wie in allebei de ranglijsten hoog staat, wint.".format(
                a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
            "a": "Met k = 60 geeft een eerste plaats 1/61 = {p1}, een derde 1/63 = {p3}: nauwelijks meer. A krijgt in totaal {a}, B krijgt {b}. Zie [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl).".format(
                p1=C.number(1 / 61, L.lang, 4), p3=C.number(1 / 63, L.lang, 4), a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
            "same": "RRF neemt geen gemiddelde van de plaatsen: elke ranglijst geeft `1 / (k + rang)`, en de delen tellen op. A krijgt {a}, B krijgt {b}. Zie [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl).".format(
                a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
        },
    )
    quick_rrf
    return (quick_rrf,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _w = C.worked()
    mo.md(r"""
    ## Waarom

    In [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl) zette de samengevoegde zoekfunctie het goede punt bij de drie beste voor {top3} van de {n} vragen. *{question}* ({question_nl}) Daar kwam het goede punt op plaats {before}. Zoeken op betekenis maakt de embedding van de vraag en die van elk punt los van elkaar, en vergelijkt dus twee samenvattingen van {dims} getallen. Het leest de vraag en het punt nooit samen.

    **Een voorbeeld uit het dagelijks leven.** Een recruiter sorteert in een uur een stapel van 500 brieven op een paar woorden: een diploma, een jaartal, een stad. De 20 beste gaan daarna naar iemand die elke brief helemaal leest, met de baan in het achterhoofd, en ze opnieuw ordent. Trager per brief, maar het zijn er maar 20.
    """.format(top3=_f["hybrid_top3"], n=_f["questions"], question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], before=_w["before"], dims=C.DIMS))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **reranker** | een tweede, trager model dat de vraag samen met elke kandidaat leest en die een score geeft, om de kandidaten in een betere volgorde te zetten |
    | **kandidaten** | de teksten die de snelle zoekfunctie aan de reranker geeft: hier de {cands} beste |
    | **bi-encoder** | een model dat van elke tekst apart een embedding maakt, zoals Granite doet bij zoeken op betekenis: de punten kun je één keer van tevoren embedden |
    | **cross-encoder** | een model dat twee teksten samen leest en één score voor het paar geeft: preciezer, maar het moet voor elk paar opnieuw draaien |
    | **logprob** | de natuurlijke logaritme van de kans op een token; een API met `"logprobs": true` geeft die voor elk geschreven token, en met `"top_logprobs": 5` ook voor de vijf waarschijnlijkste tokens op die plek |
    | `P(yes)` | de kans dat het antwoord van het model met yes begint: `math.exp` van de logprob geeft die kans terug |

    Een reranker is een cross-encoder. Elk chat model kan er een zijn: vraag of de passage de vraag beantwoordt, laat het model één token schrijven, en neem de kans op yes als score. Het geschreven token alleen is niet genoeg: Qwen3 schreef yes bij alle {judged} kandidaten die de course gaf. De kans zegt hoe zeker het model was, en die ordent de kandidaten.

    Modellen die hiervoor gemaakt zijn, werken net zo. Qwen3-Reranker is een Qwen3 die getraind is op precies de prompt van deze les, en zijn score is de kans op yes. Cohere Rerank, bge-reranker en Jina Reranker zijn andere, vaak achter een API die de vraag en de kandidaten krijgt en voor elke kandidaat een score teruggeeft.
    """.format(cands=C.CANDIDATES, judged=_f["judged"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: reranking

    1. **Zoeken:** de snelle zoekfunctie geeft de 20 beste kandidaten.
    2. **Oordelen:** vraag het model voor elke kandidaat om één token over de vraag en de kandidaat, met logprobs, en haal `P(yes)` uit de waarschijnlijkste tokens.
    3. **Sorteren:** zet de kandidaten op volgorde van `P(yes)`, de hoogste eerst, en houd de paar beste over voor het antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        *{question}* ({question_nl}) De punten zijn in het Engels, en de course stelde de vraag ook in het Engels.

        1. **Zoeken:** de samengevoegde zoekfunctie geeft 20 kandidaten. Bovenaan staat een punt uit *{t0}*; het goede punt staat op plaats {before}: *{gold}*
        2. **Oordelen:** voor elke kandidaat stuurt de course deze request naar zijn server, met de kandidaat na `<Document>:`.
        """.format(question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], t0=C.title_of(_w["candidates"][0], L.lang), before=_w["before"], gold=_w["gold"])),
        mo.md(C.request_code(_w["question"]["text"]["en"])),
        mo.md(r"""
        Met `"max_tokens": 1` schrijft het model één token, en met `"temperature": 0` neemt het het waarschijnlijkste, dus een oordeel is bij elke run hetzelfde. Voor het goede punt waren de vijf waarschijnlijkste eerste tokens {tokens}. `P(yes)` is de som van de kansen op de tokens die yes zijn: {sum} = {p}. De eerste kandidaat van de zoekfunctie kreeg {p0}.

        3. **Sorteren:** het goede punt gaat naar plaats {after}, met {p}. Tweede is een punt van *{t1}*, met {p1}.
        """.format(tokens=C.tokens_text(_w["gold_response"], L.lang), sum=C.sum_text(_w["gold_response"], L.lang), p=C.number(_w["gold_p"], L.lang, 3),
                   p0=C.number(_w["first_p"], L.lang, 3), after=_w["after"], t1=C.title_of(_w["reranked"][1], L.lang), p1=C.number(_w["second_p"], L.lang, 3))),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    Voor elk van de {n} vragen die een punt beantwoordt, gaf de course Qwen3 de {cands} kandidaten van de samengevoegde zoekfunctie, elk in een eigen request, en sorteerde ze op `P(yes)`. Waar kwam het goede punt terecht?
    """.format(n=_f["questions"], cands=C.CANDIDATES))
    return


@app.cell(hide_code=True)
def picture_places():
    mo.Html(C.places_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Bij de drie beste: {before3} vragen na de samengevoegde zoekfunctie, {after3} na reranking. Op de eerste plaats: {before1} ervoor, {after1} erna. De reranker tilde het goede punt van {rescued} vragen van ver achteraan naar de drie beste, en liet het bij {lost} vraag uit de drie beste vallen: *{lost_q}* ({lost_q_nl}) Daar had de zoekfunctie het goede punt eerst, en zette de reranker het op plaats {lost_place}.

        Het kostte tijd. Eén oordeel duurde {one} seconden op de computer van de course, dus 20 kandidaten duurden zo'n {twenty} seconden per vraag, tegen een zoekfunctie die het model helemaal niet aanroept. Voor elk van de {none} vragen die geen punt beantwoordt, kreeg de beste kandidaat toch een `P(yes)` van minstens {none_p}: de score ordent de kandidaten, maar zegt niet of er een de vraag beantwoordt.
        """.format(before3=_f["hybrid_top3"], after3=_f["reranked_top3"], before1=_f["hybrid_first"], after1=_f["reranked_first"],
                   rescued=len(_f["rescued"]), lost=len(_f["lost"]),
                   lost_q=C.question(_f["lost"][0])["text"]["en"], lost_q_nl=C.question(_f["lost"][0])["text"]["nl"], lost_place=_f["lost_place"], one=C.number(_f["judgment"], L.lang, 2),
                   twenty=C.number(_f["seconds"], L.lang, 0), none=len(C.NONE), none_p=C.number(_f["none_p"], L.lang, 2))),
        L.watch_out("Een reranker kan een volgorde ook slechter maken, zoals bij de vraag hierboven; Qwen3 is een klein model dat er niet voor gemaakt is. Meet het op je eigen vragen, en weeg de winst af tegen de seconden die het per vraag kost."),
    ])
    return


@app.cell(hide_code=True)
def text_p_yes():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De kans op yes

        `p_yes(response)` krijgt de JSON van een antwoord van de server, als dict, en moet de kans geven dat het antwoord met yes begint. `response["choices"][0]["logprobs"]["content"][0]` is het eerste token, en zijn `"top_logprobs"` een lijst met de vijf waarschijnlijkste tokens, elk een dict met `"token"` en `"logprob"`. Tel elk token dat yes is, met welke hoofdletters en spaties eromheen ook: `"yes"`, `"Yes"` en `" yes"` tellen allemaal. Voor het goede punt van het uitgewerkte voorbeeld moet `p_yes` {p} geven (afgerond). De check laat ook zien wat je functie de 20 kandidaten van die vraag geeft.
        """.format(p=C.number(C.worked()["gold_p"], L.lang, 3))),
        L.task("Tel de kans op van elk token in `top_logprobs` dat yes is: `math.exp` van zijn logprob."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p_yes():
    box_p_yes = L.code_box("p_yes", lines=10)
    box_p_yes
    return (box_p_yes,)


@app.cell(hide_code=True)
def check_p_yes(box_p_yes):
    result_p_yes = L.check("p_yes", box_p_yes.value)
    result_p_yes
    return (result_p_yes,)


@app.cell(hide_code=True)
def hints_p_yes():
    L.hints(
        "p_yes",
        [
            "Een logprob is de logaritme van een kans, dus `math.exp` ervan geeft de kans terug. Tel die van de tokens die yes betekenen op.",
            "In de loop maakt `token.strip().lower()` van `\" Yes\"` een `\"yes\"`. Alleen als het `\"yes\"` is, telt het token mee.",
            "In de loop, na `token = ...`: `if token.strip().lower() == \"yes\":`, en daaronder `total = total + math.exp(entry[\"logprob\"])`.",
        ],
        why="""```python
import math


def p_yes(response):
    first = response["choices"][0]["logprobs"]["content"][0]
    total = 0.0
    for entry in first["top_logprobs"]:
        token = entry["token"]
        if token.strip().lower() == "yes":
            total = total + math.exp(entry["logprob"])
    return total
```

Het model kan yes op meer dan één manier schrijven, en elke manier heeft een eigen kans; samen zijn ze de kans op yes. Een token dat niet bij de vijf waarschijnlijkste hoort, voegt bijna niets toe.""",
    )
    return


@app.cell(hide_code=True)
def text_rerank():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Reranken

        `rerank(question, candidates, n)` moet de `n` kandidaten met de hoogste `P(yes)` geven, de hoogste eerst. Gegeven zijn:

        - `judge(question, passage)`: het antwoord van de server voor die vraag en passage, als dict, uit de opgenomen oordelen van de course;
        - `p_yes(response)`: die van oefening 1.

        De code vraagt voor elke kandidaat een oordeel, en geeft de eerste `n` in de volgorde van de zoekfunctie. Voor de vraag van het uitgewerkte voorbeeld moet `rerank` het goede punt eerst geven. De check probeert ook andere vragen, en een andere `n`.
        """),
        L.task("Bewaar voor elke kandidaat een paar `(p_yes(response), passage)`, sorteer de paren met de hoogste eerst, en geef de passages van de eerste `n`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rerank():
    box_rerank = L.code_box("rerank", lines=7)
    box_rerank
    return (box_rerank,)


@app.cell(hide_code=True)
def check_rerank(box_rerank):
    result_rerank = L.check("rerank", box_rerank.value)
    result_rerank
    return (result_rerank,)


@app.cell(hide_code=True)
def hints_rerank():
    L.hints(
        "rerank",
        [
            "Elke kandidaat krijgt een score: `p_yes` van zijn antwoord. Als je paren van score en passage sorteert, komen ze op volgorde van de score.",
            "Zet in de loop `scored.append((p_yes(response), passage))`. Na de loop heeft `sorted(scored, reverse=True)` de hoogste score eerst.",
            "Vervang de laatste regel door twee: `best = sorted(scored, reverse=True)`, en geef dan de passages van `best[:n]` terug: `[passage for score, passage in best[:n]]`.",
        ],
        why="""```python
def rerank(question, candidates, n):
    scored = []
    for passage in candidates:
        response = judge(question, passage)
        scored.append((p_yes(response), passage))
    best = sorted(scored, reverse=True)
    return [passage for score, passage in best[:n]]
```

De reranker ziet de volgorde van de zoekfunctie nooit: hij beoordeelt elke kandidaat op zichzelf, en alleen zijn scores beslissen.""",
    )
    return


@app.cell(hide_code=True)
def text_all():
    mo.md(r"""
    ### Oefening 3. Waarom niet alle punten?

    De reranker las het goede punt van het uitgewerkte voorbeeld beter dan de zoekfunctie. Een collega wil de zoekfunctie overslaan en Qwen3 voor elke vraag alle {points} punten laten beoordelen.
    """.format(points=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def choice_all_cell():
    _f = C.findings()
    _all = len(C.POINTS) * _f["judgment"] / 60
    choice_all = L.choice(
        "all-points",
        question="Wat is daar het grootste probleem mee?",
        options={
            "worse": "De reranker wordt slechter als hij uit meer punten moet kiezen",
            "embed": "De punten hebben dan eerst een eigen embedding nodig",
            "slow": "Het duurt dan minuten voor elke vraag",
        },
        correct="slow",
        why={
            "slow": "Met {one} seconden per oordeel duren {points} punten zo'n {minutes} minuten voor één vraag. De snelle zoekfunctie kan alle punten rangschikken omdat hun embeddings één keer van tevoren gemaakt zijn; een cross-encoder moet voor elk paar van vraag en punt opnieuw draaien.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
            "worse": "Elk oordeel kijkt naar één punt op zichzelf, dus de andere punten veranderen de score niet. Het probleem is tijd: met {one} seconden per oordeel duren {points} punten zo'n {minutes} minuten voor één vraag.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
            "embed": "De reranker heeft geen embeddings nodig: hij leest de vraag en het punt als tekst. Het probleem is tijd: met {one} seconden per oordeel duren {points} punten zo'n {minutes} minuten voor één vraag.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
        },
    )
    choice_all
    return (choice_all,)


@app.cell(hide_code=True)
def text_none():
    _f = C.findings()
    _n = C.none_example()
    mo.md(r"""
    ### Oefening 4. Een vraag die geen punt beantwoordt

    *{question}* ({question_nl}) Geen punt van de course beantwoordt die vraag. Toch gaf de reranker zijn beste kandidaat een `P(yes)` van {p}: *{best}*
    """.format(question=_n["question"]["text"]["en"], question_nl=_n["question"]["text"]["nl"], p=C.number(_n["p"], L.lang, 3), best=_n["best"]))
    return


@app.cell(hide_code=True)
def choice_none_cell():
    _f = C.findings()
    choice_none = L.choice(
        "no-answer",
        question="Wat zegt dat over de score van de reranker?",
        options={
            "order": "Hij ordent kandidaten goed, maar zegt niet of er een antwoordt",
            "broken": "De reranker is stuk: bij zo'n vraag hoort elke score dicht bij 0 te liggen",
            "true": "Die kandidaat beantwoordt de vraag toch, anders was de score laag",
        },
        correct="order",
        why={
            "order": "De course mat het: voor elk van de {none} vragen die geen punt beantwoordt, kreeg de beste kandidaat minstens {p}. Gebruik de volgorde om de paar beste te kiezen, en laat de stap die het antwoord schrijft zeggen wanneer de bronnen geen antwoord geven, zoals de grounded prompt van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) doet.".format(
                none=len(C.NONE), p=C.number(_f["none_p"], L.lang, 2)),
            "broken": "Een klein model dat niet getraind is om te oordelen, zegt makkelijk yes: het schreef yes bij alle {judged} kandidaten. Toch zetten zijn scores het goede punt hoger dan andere, dus de volgorde is nuttig, ook als elke score hoog is.".format(
                judged=_f["judged"]),
            "true": "Geen punt beantwoordt hem: de course schreef hem als een van zijn vragen zonder antwoord in de course. Een hoge score betekent alleen dat het model vrij zeker was van yes. Voor elk van de {none} vragen die geen punt beantwoordt, kreeg de beste kandidaat minstens {p}.".format(
                none=len(C.NONE), p=C.number(_f["none_p"], L.lang, 2)),
        },
    )
    choice_none
    return (choice_none,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Reranken met je eigen functies

    Zodra oefening 1 en 2 goed zijn, kies je een vraag. De course rerankt de 20 kandidaten met jouw `rerank` en jouw `p_yes`, op de opgenomen oordelen van Qwen3, en zet de vijf beste naast die van de samengevoegde zoekfunctie. Qwen3 kreeg de vraag in het Engels.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q21")["text"][L.lang], label="Vraag", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_rerank(box_p_yes, box_rerank, question_pick):
    _fn = C.learner_rerank(box_p_yes.value, box_rerank.value)
    if _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: het reranken gebruikt je eigen `p_yes` en `rerank`.*")
    else:
        _out = mo.Html(C.reranked(_fn, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De paar beste na reranking gaan de prompt in, zoals in [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl). Een zoekfunctie die het goede punt bij de 20 kandidaten heeft, maar niet bij de drie beste, is precies waar een reranker helpt; een latere les meet een RAG-systeem in twee delen, het zoeken en het antwoord, zodat je ziet welk van de twee je moet verbeteren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_all, choice_none, result_p_yes, result_rerank):
    L.wrap_up(result_p_yes, result_rerank, choice_all, choice_none)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
