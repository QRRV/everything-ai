# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-rag-first. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen runs en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-first", "nl")
    C = L.checks   # de knowledge base, de opgenomen runs van Qwen3, de checks, en de demo met jouw rag


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="Je app heeft een system prompt en stuurt de vragen van de gebruiker over HTTP naar een model. Waar gaat de system prompt heen?",
        options={
            "every": "In elk request, als eerste bericht, met de rol system",
            "once": "Alleen in het eerste request; daarna onthoudt de server hem",
            "end": "Aan het eind van het bericht van de gebruiker, zodat het model hem als laatste leest",
        },
        correct="every",
        why={
            "every": "De server onthoudt niets tussen requests, dus de system prompt gaat met elk request mee, als eerste bericht met de rol system. Daarom kost hij ook elke keer input tokens.",
            "once": "De server onthoudt niets tussen requests: de system prompt gaat met elk request mee, als eerste bericht met de rol system. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
            "end": "De system prompt is een eigen bericht, met de rol system, aan het begin van het gesprek, in elk request. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl) gaf de course Qwen3 de goede bron. Een assistent moet hem zelf vinden: tussen duizenden teksten, bij vragen waar niemand van tevoren aan dacht. In deze les bouw je het geheel, in drie korte functies: een die de woorden van een tekst vindt, een die zoekt, en een die het model vraagt met wat de zoekfunctie vond.

    **Een voorbeeld uit het dagelijks leven.** Om iets te vinden in een dik handboek, lees je het niet van voren af: je zoekt een woord uit je vraag op in het register achterin, en leest de bladzijden waar het naar wijst. Zoek je een woord op dat het boek niet gebruikt, zeg *prijs* waar het boek *kosten* zegt, dan vind je niets, ook al staat het antwoord in het boek. Dat is maar een vergelijking; hieronder zie je hetzelfde met een zoekfunctie die woorden telt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **knowledge base** | de verzameling teksten die een assistent kan doorzoeken voor zijn antwoorden |
    | **RAG-pipeline** | de stappen van vraag naar antwoord, na elkaar: de knowledge base doorzoeken, de beste teksten in de prompt zetten, het model vragen |
    | `a & b` | wat twee sets `a` en `b` gemeen hebben, als nieuwe set |

    De knowledge base van deze les is de course zelf: elke les eindigt met een samenvatting van een paar punten die zeggen wat hij mat, en samen zijn dat {points} punten uit {lessons} lessen. Elk punt is een paar zinnen in het Engels.

    De zoekfunctie telt voor elk punt hoeveel woorden het met de vraag deelt, en houdt de punten met de meeste. Voordat hij telt, maakt hij elk woord klein en haalt hij de leestekens van de uiteinden, zodat *Cache?* en *cache* als hetzelfde woord tellen. Een set bevat elk woord één keer, en `&` geeft wat twee sets gemeen hebben: `{{"a", "cache", "the"}} & {{"the", "cache", "shop"}}` geeft `{{"cache", "the"}}`, twee woorden.
    """.format(points=C.number(len(C.POINTS), L.lang, 0), lessons=C.lesson_count()))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een eerste RAG

    1. **Woorden:** maak de set van de woorden van de vraag, klein geschreven en zonder leestekens aan de uiteinden.
    2. **Zoeken:** tel voor elk punt hoeveel van die woorden het heeft, en houd de drie met de meeste.
    3. **Prompt:** zet die drie genummerd voor de vraag, na de system prompt.
    4. **Vragen:** stuur het naar het model, en geef zijn antwoord terug.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course vraagt in het Engels, omdat de punten Engels zijn: *{question}*

    1. **Woorden:** de vraag heeft {n} woorden: {asked}.
    2. **Zoeken:** een punt van de les over prompt caching deelt er {gold}: {shared}. Geen ander punt deelt er meer dan {second}; van alle {points} punten delen er {many} precies {common} woorden met de vraag, meestal *a* en *the*, en {zero} delen er geen. De drie met de meeste delen er {top0}, {top1} en {top2}.
    3. **Prompt:** de system prompt van deze les, *{system}*, en dan de drie punten met hun nummers voor de vraag, zoals `with_sources` hem maakt.
    4. **Vragen:** in beide runs antwoordde Qwen3: *{answer}* In het punt staat *4.78 seconds*, dus het antwoord is goed. Zonder de bronnen zei Qwen3 {alone0} en {alone1}.
    """.format(question=_w["question"], n=len(_w["asked"]), asked=", ".join(["*" + w + "*" for w in _w["asked"]]),
               gold=_w["shares"][0], shared=", ".join(["*" + w + "*" for w in _w["shared"]]), second=_w["shares"][1],
               points=C.number(len(C.POINTS), L.lang, 0), many=C.number(_w["many"], L.lang, 0), common=_w["common"], zero=_w["zero"],
               top0=_w["shares"][0], top1=_w["shares"][1], top2=_w["shares"][2],
               system=C.SYSTEM, answer=C.answers("q11", "rag")[0]["text"].strip(), alone0="10 seconden", alone1="12 seconden"))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    De course stelde Qwen3 {all} vragen over de course, in het Engels, elk {seeds} keer: zonder bronnen, met alleen de system prompt; en met RAG, met de drie punten die het zoeken op woorden vond tussen alle {points}. Een punt beantwoordt er {questions}: {words} delen woorden met dat punt, en {meaning}, gemarkeerd in het plaatje, vragen hetzelfde in andere woorden. Het plaatje laat die {questions} zien. Elk vierkantje is één antwoord; wijs een vierkantje aan om het te lezen:
    """.format(all=len(C.QUESTIONS), questions=_f["questions"], seeds=C.SEEDS, points=C.number(len(C.POINTS), L.lang, 0), words=_f["words"], meaning=_f["meaning"]))
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        De zoekfunctie vond het punt dat de vraag beantwoordt bij {hits} van de {questions}: bij {words_hits} van de {words} vragen die er woorden mee delen, maar bij maar {meaning_hits} van de {meaning} die het in andere woorden vragen. Zonder bronnen had Qwen3 {alone} van de {runs} keer gelijk, beide keren over iets wat het van buiten de course wist: dat een SHA-256-checksum 64 hextekens heeft, en dat je voor de halve onzekerheid van een gemiddelde vier keer zoveel data nodig hebt. Met RAG waren {rag} van de {runs} antwoorden goed, allemaal waar de zoekfunctie het punt had gevonden: daar {rag_hit} van de {hit_runs}, tegen {rag_miss} van de {miss_runs} waar hij het niet had gevonden.

        Een gevonden punt is nog geen goed antwoord. Het punt over het aandeel klanten zegt *21.7%*, en voegt eraan toe dat het met alle bestellingen meegeteld *26.1%* zou zijn; Qwen3 antwoordde in beide runs *26.1%*. Waar de zoekfunctie miste, nam Qwen3 een getal uit wat het kreeg: het punt over het kleine CNN zat niet bij de drie, en één antwoord gaf het *786 parameters*, uit een punt over 786 vormen.

        Dan de {none} vragen die geen punt beantwoordt, zoals de stad van de fietsenwinkel. Zonder bronnen zei geen van hun {none_runs} antwoorden dat Qwen3 het niet wist: het gaf *$25 per month* en *Paris*, of zei alleen dat de course zijn runs opnam *on a graphics card*. Met RAG zeiden er {unknown} van de {none_runs} dat de bronnen het niet zeggen; de andere verzonnen toch iets, zoals een grafische kaart *that uses HistGradientBoostingRegressor*.
        """.format(hits=_f["hits"], questions=_f["questions"], words_hits=_f["words_hits"], words=_f["words"], meaning_hits=_f["meaning_hits"],
                   meaning=_f["meaning"], alone=_f["alone"], runs=_f["runs"], rag=_f["rag"], rag_hit=_f["rag_hit"], hit_runs=_f["hit_runs"],
                   rag_miss=_f["rag_miss"], miss_runs=_f["miss_runs"], none=_f["none"], unknown=_f["none_unknown_rag"], none_runs=_f["none_runs"])),
        L.watch_out("Zoeken op woorden telt elk woord even zwaar: *the* en *a* tellen net zoveel als *cache*, dus de meeste punten delen een woord of twee met elke vraag, zonder erover te gaan. En een vraag in andere woorden deelt nauwelijks een woord met zijn punt. Verkeerde bronnen kunnen zelfs verdringen wat het model wist: zonder bronnen antwoordde Qwen3 één keer dat je voor de halve onzekerheid *quadruple* de data nodig hebt; met drie punten over iets anders erbij ging het in beide runs over die punten. [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl) laat zeldzame woorden zwaarder tellen; een latere les zoekt op betekenis in plaats van op woorden."),
    ])
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De woorden van een tekst

        `words(text)` moet de set van de woorden van een tekst geven, zoals het uitgewerkte voorbeeld ze telde: klein geschreven, en zonder leestekens aan de uiteinden. Het vak splitst de tekst bij de spaties en maakt er een set van, maar houdt hoofdletters en leestekens, dus *Cache?* en *cache* blijven twee woorden. `text.lower()` geeft de tekst klein geschreven. `strip` haalt meer weg dan spaties als je hem tekens geeft: `"(cache?)".strip(".,:;!?()")` haalt elk van die tekens van beide uiteinden, en geeft `"cache"`. Leestekens binnen een woord blijven staan, zoals in *4.78*.
        """),
        L.task("Maak de tekst klein, en haal `.,:;!?()` van beide uiteinden van elk woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("words", lines=3)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("words", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "words",
        [
            "Er veranderen twee dingen voordat de set gemaakt wordt: de hele tekst wordt klein, en elk woord verliest de leestekens aan zijn uiteinden.",
            "`text.lower().split()` geeft de woorden klein geschreven. Een list comprehension, `[... for word in ...]`, verandert onderweg elk woord.",
            "Maak van de laatste regel `return set([___ for word in text.lower().split()])`.",
        ],
        why="""```python
def words(text):
    return set([word.strip(".,:;!?()") for word in text.lower().split()])
```

Een set bevat elk woord één keer, dus een woord dat twee keer in een punt staat, telt één keer. `strip` en niet `replace`, omdat leestekens binnen een woord erbij horen: *4.78* en *1,038* blijven heel.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De punten die de meeste woorden delen

        `search(question, points, k)` moet de `k` punten geven die de meeste woorden met de vraag delen, de meeste eerst. Het vak maakt een lijst van paren, een getal en een punt, sorteert ze van groot naar klein met `sorted(found, reverse=True)`, en houdt de punten van de eerste `k`. Paren sorteren op hun eerste deel, het getal; alleen als twee getallen gelijk zijn, beslist het punt. Maar het vak geeft elk punt het getal 0. `words` is gegeven, zoals hij is na oefening 1.
        """),
        L.task("Geef elk punt het aantal woorden dat het met de vraag deelt, in plaats van 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=7)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Maak de set van de woorden van de vraag één keer, vóór de loop, en vergelijk hem in de loop met de woorden van elk punt.",
            "`asked = words(question)` vóór de loop. `&` tussen twee sets geeft de set van wat in allebei zit, en `len(...)` telt die.",
            "In de loop: `found.append((___, point))`, met het aantal gedeelde woorden in het gat.",
        ],
        why="""```python
def search(question, points, k):
    asked = words(question)
    found = []
    for point in points:
        found.append((len(asked & words(point)), point))
    return [point for shared, point in sorted(found, reverse=True)[:k]]
```

De woorden van de vraag zijn voor elk punt hetzelfde, dus je maakt ze één keer. Punten die evenveel woorden delen, komen in omgekeerde alfabetische volgorde: zoeken op woorden heeft niets beters om tussen ze te kiezen.""",
    )
    return


@app.cell(hide_code=True)
def text_rag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een hele RAG

        Het vak is de `ask` van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met de system prompt van deze les, `SYSTEM`, voor de vraag en ruimte voor 60 tokens. Verander `rag(url, question)` zo dat hij tussen alle punten, `POINTS`, de drie vindt die de meeste woorden met de vraag delen, en de prompt die `with_sources` maakt van de vraag en die drie als bericht van de gebruiker stuurt. `SYSTEM`, `POINTS`, `search` en de `with_sources` van de vorige les zijn gegeven. De check draait je functie tegen een stand-in.
        """),
        L.task("Vind drie punten met `search`, en stuur de vraag met die bronnen als bericht van de gebruiker."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rag():
    box_rag = L.code_box("rag", lines=15)
    box_rag
    return (box_rag,)


@app.cell(hide_code=True)
def check_rag(box_rag):
    result_rag = L.check("rag", box_rag.value)
    result_rag
    return (result_rag,)


@app.cell(hide_code=True)
def hints_rag():
    L.hints(
        "rag",
        [
            "Er komen twee stappen vóór de berichten: vind de passages, en maak er dan de prompt van.",
            "`search` met `POINTS` en 3 geeft de drie punten, en `with_sources` maakt de prompt van de vraag en die punten.",
            "`passages = search(question, POINTS, 3)` vóór `messages`, en als bericht van de gebruiker `{\"role\": \"user\", \"content\": ___}`.",
        ],
        why="""```python
import json
import urllib.request


def rag(url, question):
    passages = search(question, POINTS, 3)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": with_sources(question, passages)}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 60}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

Dat is de hele pipeline: zoeken, prompt, model. Elke stap kan door een betere worden vervangen zonder de andere aan te raken, en de volgende lessen doen dat met het zoeken.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra de drie oefeningen goed zijn, kun je je eigen vragen over de course stellen, met je eigen `rag`: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. Vraag in het Engels, omdat de punten Engels zijn en omdat Qwen3 Engels het best schrijft. Probeer een vraag die woorden deelt met een punt, en een die hetzelfde in andere woorden vraagt, en kijk wat je zoekfunctie vindt.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de licentie Apache-2.0, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het binnengekomen bestand is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def settings_cell():
    question_text = mo.ui.text(value=C.question("q11")["text"]["en"], label="Vraag", full_width=True)
    ask_button = mo.ui.run_button(label="Vraag met je rag")
    mo.vstack([question_text, ask_button])
    return ask_button, question_text


@app.cell(hide_code=True)
def demo_answer(ask_button, box_rag, box_search, box_words, question_text):
    _fn = C.learner_rag(box_words.value, box_search.value, box_rag.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst de drie oefeningen af: de demo draait je eigen `words`, `search` en `rag`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    else:
        mo.output.replace(mo.md("*Je zoekfunctie kijkt de punten door, de server laadt Qwen3, en Qwen3 schrijft …*"))
        _words = {"title": "Je rag: {question}", "sources": "De drie punten die je zoekfunctie vond, met de les van elk:",
                  "answer": "Het antwoord van Qwen3, na {seconds} seconden:", "empty": "Typ eerst een vraag.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De knowledge base was hier al schone tekst. De meeste documenten zijn pdf's en webpagina's, en [Van bestand naar tekst](/course/open/build-documents-to-text?lang=nl) haalt de tekst eruit. Elk punt was hier een paar zinnen; een handboek van honderd bladzijden moet eerst in stukken, in [Documenten in stukken knippen](/course/open/build-chunking?lang=nl). En [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl) laat de zoekfunctie zeldzame woorden zwaarder tellen dan *the* en *a*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rag, result_search, result_words):
    L.wrap_up(result_words, result_search, result_rag)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
