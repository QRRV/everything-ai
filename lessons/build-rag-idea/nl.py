# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-rag-idea. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen runs en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-idea", "nl")
    C = L.checks   # de vragen, de opgenomen runs van Qwen3, de checks, en de demo met jouw ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Waarom language models dingen verzinnen](/course/open/lm-hallucination?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cutoff():
    quick_cutoff = L.choice(
        "quick-cutoff",
        question="Een model is een jaar geleden getraind. Je vraagt het naar een regel die je bedrijf vorige maand maakte. Wat gebeurt er het waarschijnlijkst?",
        options={
            "refuse": "Het zegt dat het dat niet kan weten, omdat de regel nieuwer is dan zijn training",
            "search": "Het zoekt de regel op de website van het bedrijf op voordat het antwoordt",
            "made-up": "Het kan een antwoord geven dat zeker klinkt maar verzonnen is",
        },
        correct="made-up",
        why={
            "made-up": "Het model schrijft waarschijnlijke tekst, geen gecontroleerde feiten, en het weet niets van na zijn knowledge cutoff. Toch kan het zeker klinken, dus een antwoord over de nieuwe regel kan gewoon verzonnen zijn.",
            "refuse": "Een model weet niet wat het niet weet: het schrijft waarschijnlijke tekst, en die kan zeker klinken over een regel die het nooit zag. Zie [Waarom language models dingen verzinnen](/course/open/lm-hallucination?lang=nl).",
            "search": "Een model zoekt uit zichzelf niets op: het schrijft vanuit wat het leerde, tot aan zijn knowledge cutoff. Eerst opzoeken is wat deze les toevoegt. Zie [Waarom language models dingen verzinnen](/course/open/lm-hallucination?lang=nl).",
        },
    )
    quick_cutoff
    return (quick_cutoff,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Qwen3-0.6B leerde van heel veel teksten, tot het moment waarop het getraind werd. Het weet niet wat er daarna gebeurde, en ook niet wat nooit in die teksten stond: het handboek van een bedrijf, de notities van een team, de getallen die deze course mat. Vraag het toch, en het antwoordt alsof het het wist. Een bedrijf dat een model vragen over zijn eigen documenten wil laten beantwoorden, kan het model niet elke keer opnieuw trainen als een document verandert. Het kan eenvoudiger: zoek eerst de tekst op die de vraag beantwoordt, en geef die samen met de vraag aan het model.

    **Een voorbeeld uit het dagelijks leven.** Bij een tentamen met open boek hoef je niet elk getal uit je hoofd te kennen: je zoekt het op en antwoordt vanuit wat het boek zegt. Maar je moet de goede bladzijde vinden, en lezen wat er staat in plaats van wat je denkt dat er zou moeten staan. Dat is alleen een vergelijking; hieronder zie je wat Qwen3 doet met en zonder de goede bladzijde.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **retrieval** | uit heel veel teksten de teksten vinden die bij een vraag passen |
    | **retrieval-augmented generation** (RAG) | eerst teksten opzoeken die bij de vraag passen, ze dan met de vraag in de prompt zetten, zodat het model zijn antwoord eruit kan halen |

    De naam komt uit een paper van Patrick Lewis en anderen bij Facebook AI uit 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. In hetzelfde jaar liet REALM van Google een model passages uit Wikipedia ophalen terwijl het leerde, en in 2021 haalde RETRO van DeepMind ze uit een verzameling van zo'n 2 biljoen tokens. De meeste RAG-systemen doen het nu op de eenvoudige manier van deze les: een zoekfunctie vindt de teksten, en de prompt brengt ze naar een model dat er nooit op getraind is.

    ### Wat de course mat

    De course stelde Qwen3 {n} vragen, elk {seeds} keer, op drie manieren: zonder bron; met het punt uit de samenvattingen van de course dat de vraag beantwoordt; en met het punt van een andere vraag, een bron die de vraag niet beantwoordt. Twee vragen gaan over wat al lang vóór de training van Qwen3 bekend was: de top-5 error van AlexNet in 2012, en het aantal hextekens van een SHA-256-checksum. De andere zes gaan over wat alleen deze course mat. Elk vierkantje is één antwoord, en de legenda onder het plaatje zegt of het goed is; wijs een vierkantje aan om het antwoord te lezen:
    """.format(n=len(C.QUESTIONS), seeds=C.SEEDS))
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ways():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Zonder bron waren {none} van de {runs} antwoorden goed. Qwen3 kende geen enkel getal van de course: de antwoorden zeiden dat het ervan afhangt, of legden uit hoe je het zou kunnen uitzoeken. Zelfs op de twee vragen van buiten de course had het het maar één keer goed: het gaf AlexNet in de ene run een top-5 error van *{alex1}* en in een andere een accuracy van *{alex2}*, en een SHA-256-checksum *{sha0}*, *{sha1}*, *{sha2}* en één keer de goede *{sha3}*. Met de goede bron waren {right} van de {runs} goed. Met de bron van een andere vraag was er {wrong} goed. Bij de vragen over de course zeiden alle {unknown} antwoorden op een of andere manier dat de bronnen de vraag niet beantwoorden, al las er één eerst de 929 uit de verkeerde bron als de perplexity. Bij SHA-256 antwoordde Qwen3 uit zijn geheugen, en had het drie keer fout.
        """.format(none=_f["none"]["right"], runs=_f["none"]["runs"], right=_f["right"]["right"], wrong=_f["wrong"]["right"], unknown=_f["wrong"]["unknown"],
                   alex1="1.14%", alex2="99.9%", sha0="32 hex digits", sha1="64-bit", sha2="64 bits", sha3="64 hex digits")),
        L.watch_out("Met RAG is een antwoord zo goed als zijn bron. Met de goede tekst had Qwen3 het elke keer goed. Met een tekst die niet past, zei het dat meestal, maar het nam er ook één keer een getal uit over, en bij een vraag waarvan het dacht het antwoord te weten, negeerde het de bron. De zoekfunctie die de teksten vindt, telt dus net zo zwaar als het model, en daar gaan de volgende lessen over."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: RAG stap voor stap

    1. **Zoek op:** vind de teksten die bij de vraag passen.
    2. **Zet ze in de prompt:** genummerd, met een opdracht om ze te gebruiken, en de vraag erachter.
    3. **Laat het model schrijven:** stuur de prompt naar het model, dat het antwoord schrijft.
    4. **Controleer:** zegt het antwoord wat de bron zegt?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{question}*

    1. **Zoek op:** hier geeft de course de goede tekst, een punt uit de samenvatting van de les over een klein CNN: *{passage}*
    2. **Zet hem in de prompt:** de opdracht, de bron met zijn nummer, en de vraag:

        ```text
        {prompt}
        ```

    3. **Laat het model schrijven:** in alle {seeds} runs schreef Qwen3: *{answer}* Zonder de bron begon het met: *{alone}*
    4. **Controleer:** de bron zegt 9.098 parameters, en het antwoord ook.
    """.format(question=C.question("q01")["text"]["en"], passage=C.passages_for("q01", "right")[0], seeds=C.SEEDS,
               prompt=C.with_sources(C.question("q01")["text"]["en"], C.passages_for("q01", "right")).replace("\n", "\n    "),
               answer=C.answers("q01", "right")[0]["text"].strip(), alone=C.answers("q01", "none")[0]["text"].split(". ")[0] + "."))
    return


@app.cell(hide_code=True)
def text_sources():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Bronnen vóór de vraag

        Schrijf `with_sources(question, passages)`, die de prompt van het uitgewerkte voorbeeld maakt van een vraag en een lijst passages:

        ```text
        Use these sources to answer the question.

        [1] de eerste passage
        [2] de tweede passage

        Question: de vraag
        ```

        Elke passage staat op een eigen regel, na zijn nummer tussen vierkante haken en een spatie. Zonder passages geeft de functie alleen de vraag terug. Het vak zet de passages zonder nummers vóór de vraag, en ook als er geen zijn. De opdracht en `Question:` blijven Engels, omdat Qwen3 Engels het best schrijft.
        """),
        L.task("Nummer de passages vanaf 1, en geef alleen de vraag terug als er geen zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sources():
    box_sources = L.code_box("with_sources", lines=11)
    box_sources
    return (box_sources,)


@app.cell(hide_code=True)
def check_sources(box_sources):
    result_sources = L.check("with_sources", box_sources.value)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "with_sources",
        [
            "Twee veranderingen: een nummer dat bij elke passage één omhooggaat, en aan het begin een test op een lijst zonder passages.",
            "Begin vóór de loop een teller bij 1, zet hem met een f-string vóór elke passage, en tel er na elke passage 1 bij op. `len(passages) == 0` is waar voor een lege lijst.",
            "Vóór de loop: `n = 1`. In de loop: `text = text + f\"[{n}] {passage}\\n\"` en `n = n + 1`. Helemaal aan het begin: `if len(passages) == 0:` met `return question` eronder.",
        ],
        why="""```python
def with_sources(question, passages):
    if len(passages) == 0:
        return question
    text = "Use these sources to answer the question.\\n\\n"
    n = 1
    for passage in passages:
        text = text + f"[{n}] {passage}\\n"
        n = n + 1
    return text + "\\nQuestion: " + question
```

Met de nummers kunnen het model en jij zeggen uit welke bron een antwoord komt: *volgens [2]*. Zonder passages zou een opdracht om bronnen te gebruiken het model alleen in de war brengen.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vragen met bronnen

        Het vak is de `ask` van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met ruimte voor 80 tokens en een derde argument, `passages`; hij stuurt nog steeds alleen de vraag. Verander `ask(url, question, passages)` zo dat de content van het bericht van de user de prompt is die `with_sources` maakt van de vraag en de passages. `with_sources` is gegeven. De check draait je functie tegen een stand-in.
        """),
        L.task("Stuur de prompt met de bronnen als de content van het bericht van de user."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=14)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "Alleen wat er in het bericht gaat, verandert; het request en het antwoord blijven zoals ze zijn.",
            "Maak eerst de prompt met `with_sources(question, passages)`, en zet die in het bericht in plaats van `question`.",
            "`prompt = with_sources(question, passages)` vóór `body`, en in `body` het bericht `{\"role\": \"user\", \"content\": prompt}`.",
        ],
        why="""```python
import json
import urllib.request


def ask(url, question, passages):
    prompt = with_sources(question, passages)
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 80}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

De server en het model veranderen helemaal niet: RAG is alleen een andere prompt.""",
    )
    return


@app.cell(hide_code=True)
def text_needs():
    mo.md(r"""
    ### Oefening 3. Wanneer RAG helpt
    """)
    return


@app.cell(hide_code=True)
def choice_needs_cell():
    choice_needs = L.choice(
        "needs",
        question="Je bouwt een assistent op Qwen3-0.6B. Bij welke van deze vragen helpt RAG het meest?",
        options={
            "capital": "Wat is de hoofdstad van Frankrijk, en welke rivier stroomt erdoor?",
            "handbook": "Wat rekent Spoke & Chain voor een volledige servicebeurt?",
            "poem": "Schrijf een kort gedicht over herfstbladeren voor een verjaardagskaart.",
        },
        correct="handbook",
        why={
            "handbook": "De prijs staat in het eigen handboek van de winkel, dat Qwen3 nooit zag, dus uit zichzelf kan het alleen gokken. Met het goede deel van het handboek in de prompt kan het daaruit antwoorden.",
            "capital": "Parijs en de Seine stonden in talloze teksten waar Qwen3 van leerde, dus het kan zonder bron antwoorden. RAG helpt het meest bij wat het model niet kan weten, zoals de prijs in het eigen handboek van de winkel.",
            "poem": "Een gedicht vraagt om geen feit dat een bron kan geven. RAG helpt het meest bij wat het model niet kan weten, zoals de prijs in het eigen handboek van de winkel.",
        },
    )
    choice_needs
    return (choice_needs,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 en 2 goed zijn, kun je Qwen3 zelf de vragen van de course stellen, met je eigen `ask`: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. Kies een vraag en een bron, en vraag een paar keer: het antwoord kan van keer tot keer verschillen, en jouw computer is niet die van de course.
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
    question_choice = mo.ui.dropdown(options={C.question(q)["text"][L.lang]: q for q in C.QUESTIONS},
                                     value=C.question("q01")["text"][L.lang], label="Vraag")
    way_choice = mo.ui.radio(options={"geen bron": "none", "de goede bron": "right", "de bron van een andere vraag": "wrong"},
                             value="de goede bron", label="Bron")
    ask_button = mo.ui.run_button(label="Vraag met je ask")
    mo.vstack([question_choice, way_choice, ask_button])
    return ask_button, question_choice, way_choice


@app.cell(hide_code=True)
def demo_answer(ask_button, box_ask, box_sources, question_choice, way_choice):
    _fn = C.learner_ask(box_sources.value, box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo draait je eigen `with_sources` en `ask`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    else:
        mo.output.replace(mo.md("*De server laadt Qwen3, en Qwen3 schrijft …*"))
        _words = {"title": "Je ask: {question}", "sources": "De bronnen:", "no_sources": "Geen bronnen: alleen de vraag.",
                  "answer": "Het antwoord van Qwen3:", "right": "Het antwoord is goed; het duurde {seconds} seconden.",
                  "not": "Het antwoord is niet goed; het duurde {seconds} seconden.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, question_choice.value, way_choice.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Hier gaf de course Qwen3 de goede bron. In [Je eerste RAG](/course/open/build-rag-first?lang=nl) vindt een zoekfunctie de bronnen zelf, tussen alle {n} punten van de samenvattingen van de course, en zegt een system prompt waar de assistent voor is.
    """.format(n=C.number(C.point_count(), L.lang, 0)))
    return


@app.cell(hide_code=True)
def wrap_up(choice_needs, result_ask, result_sources):
    L.wrap_up(result_sources, result_ask, choice_needs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
