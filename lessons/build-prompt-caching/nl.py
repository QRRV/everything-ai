# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-prompt-caching. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-caching", "nl")
    C = L.checks   # de opgenomen runs van de course, de checks, en de demo met je messages_for


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="Wat weet een model over je gesprek als het een request beantwoordt?",
        options={
            "earlier": "Wat het hoorde in de eerdere requests van je app",
            "window": "Alleen wat in dat request staat, in zijn context window",
            "training": "Wat het leerde bij de training, en elk gesprek sindsdien",
        },
        correct="window",
        why={
            "window": "Het model onthoudt niets tussen requests: de app stuurt alles wat het nodig heeft opnieuw mee, elke keer. Daarom gaat hetzelfde lange begin er steeds weer in.",
            "earlier": "Het model onthoudt niets tussen requests; de app stuurt het gesprek elke keer opnieuw. Zie [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).",
            "training": "De training was klaar voordat je app draaide, en het model leert niets van je requests. Zie [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een app stuurt zijn lange instructies met elk request mee: een system prompt met de regels van een winkel, een document waar een gebruiker iets over vraagt, een lijst tools. Het model onthoudt niets tussen requests, dus de server leest datzelfde lange begin elke keer opnieuw, voordat hij een woord kan schrijven. Het meeste wachten, en bij een gehoste API een groot deel van de rekening, is dan steeds dezelfde tekst. Met prompt caching bewaart de server zijn werk aan dat begin, en gaat het volgende request dat net zo begint vanaf daar verder.

    **Een voorbeeld uit het dagelijks leven.** Een docent die dertig toetsen nakijkt met hetzelfde antwoordblad, leest het blad één keer en laat het open op het bureau liggen; het vóór elke toets opnieuw lezen zou het grootste deel van de avond kosten. Maar als het antwoordblad op bladzijde één verandert, moet alles erna opnieuw gelezen worden. Dat is maar een vergelijking; hieronder meet je wat de server van de course bespaart, en waardoor hij niets bespaarde.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **prompt caching** | de server bewaart wat hij voor het begin van een prompt berekende, en een latere prompt die met precies dezelfde tokens begint, gaat vanaf daar verder in plaats van ze opnieuw te berekenen |
    | **KV cache** | wat het model berekent voor elk token dat het gelezen heeft en voor elk volgend token weer nodig heeft: een key en een value per token in elke layer; een server bewaart hem terwijl hij een antwoord schrijft, en met prompt caching ook daarna |
    | **cached tokens** | de tokens van een prompt die uit de bewaarde cache kwamen: in de vorm van OpenAI `usage.prompt_tokens_details.cached_tokens` |

    ### Wat de server bewaart

    Om een antwoord te schrijven, leest het model eerst de hele prompt: elk token gaat door elke layer, en voor elk token bewaart elke layer een key en een value, waar de tokens erna naar terugkijken. Dat is de KV cache. De server van de course bewaart de KV cache van zijn laatste request, zoals de server van llama.cpp dat doet. Begint de volgende prompt met dezelfde tokens, dan haalt hij hun keys en values uit de cache en berekent hij alleen de rest. De tokens moeten gelijk zijn vanaf het allereerste: zodra er één verschilt, ziet elk token erna iets anders, dus vanaf daar kan niets meer hergebruikt worden.

    De course gaf Qwen3 het handboek van een fietsenwinkel als system prompt, {handbook} tokens aan regels, en stuurde zeven requests na elkaar. `max_tokens` was 1, dus de tijd is bijna helemaal het lezen van de prompt:
    """.format(handbook=C.number(C.findings()["handbook"], "nl", 0)))
    return


@app.cell(hide_code=True)
def picture_sequence():
    mo.Html(C.sequence_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_sequence():
    _f = C.findings()
    _s = _f["steps"]
    mo.vstack([
        mo.md(r"""
        Het eerste request las alle {cold_tokens} tokens zelf, in {cold} seconden. Het tweede stelde een andere vraag na hetzelfde handboek: {same_cached} van zijn {same_tokens} tokens kwamen uit de cache, en het duurde {same} seconden, {times} keer zo snel. Daarna kwam de tijd vóór het handboek, zoals een app hem aan de system prompt zou kunnen toevoegen: de twee prompts met een minuut ertussen hadden aan het begin maar {first_cached} tokens gemeen, tot aan de minuut, dus las de server het hele handboek opnieuw, {first} seconden. Met de tijd na de vraag, aan het eind, las de eerste van de twee nog bijna alles zelf, omdat het request ervoor met de tijd begon; een minuut later gaf de tweede {last_cached} cached tokens, en {last} seconden.

        De cache verandert niet wat het model leest, alleen wat de server opnieuw berekent. De derde vraag, één keer gesteld zonder de cache en één keer met {again_cached} van zijn tokens eruit, kreeg {same_answer}.
        """.format(cold_tokens=C.number(_s["cold"]["prompt"], "nl", 0), cold=C.number(_s["cold"]["seconds"], "nl", 2),
                   same_cached=C.number(_s["same-1"]["cached"], "nl", 0), same_tokens=C.number(_s["same-1"]["prompt"], "nl", 0),
                   same=C.number(_s["same-1"]["seconds"], "nl", 2), times=C.number(_s["cold"]["seconds"] / _s["same-1"]["seconds"], "nl", 0),
                   first_cached=_s["first-2"]["cached"], first=C.number(_s["first-2"]["seconds"], "nl", 2),
                   last_cached=C.number(_s["last-2"]["cached"], "nl", 0), last=C.number(_s["last-2"]["seconds"], "nl", 2),
                   again_cached=C.number(_f["same"][1]["cached_tokens"], "nl", 0),
                   same_answer="woord voor woord hetzelfde antwoord" if _f["same"][0]["text"] == _f["same"][1]["text"] else "antwoorden die verschillen")),
        L.watch_out("Gehoste API's cachen ook, elk op hun eigen manier (voorbeelden in november 2026). OpenAI cachet een prompt van minstens 1.024 tokens uit zichzelf en rekent de cached tokens voor een fractie van de gewone prijs. Anthropic cachet waar het request het aangeeft, met `cache_control`, en rekent het schrijven naar de cache iets duurder en het lezen eruit veel goedkoper. Standaard blijft een cache een paar minuten na zijn laatste gebruik bestaan, dus hij helpt vooral een app met veel requests kort na elkaar."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een prompt die gecachet kan worden

    1. **Wat hetzelfde blijft, komt eerst:** de system prompt, de documenten, de tools; wat per request verandert, zoals de vraag, de tijd of de naam van de gebruiker, komt als laatste.
    2. **Houd het begin precies hetzelfde:** geen tijd, geen teller en geen lijst in willekeurige volgorde erin; één token dat vroeg verschilt, betekent dat de rest opnieuw berekend wordt.
    3. **Tel wat uit de cache kwam:** lees `cached_tokens` in `usage`, en vergelijk het met `prompt_tokens`.
    4. **Houd de requests dicht bij elkaar:** standaard wordt een cache een paar minuten bewaard.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course stelde de drie vragen na het handboek na elkaar, met `max_tokens` 60, de eerste met de cache uitgezet: *{q0}*, *{q1}* en *{q2}*. Hoeveel bespaarde de cache bij de derde?

    1. **Wat hetzelfde blijft, komt eerst:** het handboek is de system prompt, en de vraag is het bericht van de gebruiker erna.
    2. **Het begin precies hetzelfde:** de drie prompts verschillen pas vanaf de vraag.
    3. **Tel wat uit de cache kwam:** het derde request had {tokens} prompt tokens, en `cached_tokens` was {cached}. De server las alleen de laatste {rest} tokens, vanaf de vraag: het handboek en de markeringen vóór de vraag kwamen uit de cache die het tweede request achterliet. Het antwoord was *{a2}*.
    4. **Dicht bij elkaar:** de drie requests kwamen direct na elkaar.
    """.format(q0=C.QUESTIONS[0], q1=C.QUESTIONS[1], q2=C.QUESTIONS[2], tokens=C.number(C.findings()["steps"]["same-2"]["prompt"], "nl", 0),
               cached=C.number(_f["answer_cached"][2], "nl", 0), rest=C.findings()["steps"]["same-2"]["prompt"] - _f["answer_cached"][2],
               a2=_f["answers"][2].strip()))
    return


@app.cell(hide_code=True)
def text_shared():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat twee prompts gemeen hebben

        De server vergelijkt de tokens van een nieuwe prompt met de tokens die hij bewaarde, vanaf het eerste token, en hergebruikt ze tot het eerste dat verschilt. Schrijf `shared_start(a, b)`, die teruggeeft hoeveel tokens de lijsten `a` en `b` aan het begin gemeen hebben. Het vak telt elke plaats waar de twee gelijk zijn, ook na een plaats waar ze verschillen.
        """),
        L.task("Stop bij de eerste plaats waar de tokens verschillen, en geef terug hoeveel er daarvóór kwamen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shared():
    box_shared = L.code_box("shared_start", lines=8)
    box_shared
    return (box_shared,)


@app.cell(hide_code=True)
def check_shared(box_shared):
    result_shared = L.check("shared_start", box_shared.value)
    result_shared
    return (result_shared,)


@app.cell(hide_code=True)
def hints_shared():
    L.hints(
        "shared_start",
        [
            "Het tellen gaat door zolang de tokens gelijk zijn; bij de eerste plaats waar ze verschillen, is de functie klaar.",
            "Draai de test om: een `if` die `True` is waar de tokens verschillen, en geef daar `n` terug met `return`. Na de `if` telt de loop er een bij.",
            "In de loop: `if a[i] ___ b[i]:` met `return n` eronder, en dan `n = n + 1`, niet onder de `if`.",
        ],
        why="""```python
def shared_start(a, b):
    n = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return n
        n = n + 1
    return n
```

De laatste `return n` is voor lijsten die tot het eind van de kortste nergens verschillen: dan is de hele kortste lijst gemeen.""",
    )
    return


@app.cell(hide_code=True)
def text_order():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De tijd na het handboek

        De assistent van de winkel hoort de tijd te weten, voor vragen als *Zijn jullie nog open?*. Het vak zet hem helemaal aan het begin van de system prompt, dus begint de prompt elke minuut anders. Verander `messages_for(question, now)` zo dat de system prompt alleen het handboek is, `HANDBOOK`, en het bericht van de gebruiker de vraag, een lege regel, en `Now: ` met de tijd. De check probeert je functie met drie vragen op drie tijden, en vergelijkt de berichten die hij maakt.
        """),
        L.task("Maak de system prompt alleen `HANDBOOK`, en zet de tijd na de vraag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_order():
    box_order = L.code_box("messages_for", lines=6)
    box_order
    return (box_order,)


@app.cell(hide_code=True)
def check_order(box_order):
    result_order = L.check("messages_for", box_order.value)
    result_order
    return (result_order,)


@app.cell(hide_code=True)
def hints_order():
    L.hints(
        "messages_for",
        [
            "Alleen de plaats van de tijd verandert: van de voorkant van de system prompt naar de achterkant van het bericht van de gebruiker.",
            "Het system-bericht krijgt `HANDBOOK` als content, en het bericht van de gebruiker de vraag, dan een witregel, dan `\"Now: \"` en `now`, aan elkaar gezet met `+`.",
            "Het tweede bericht: `{\"role\": \"user\", \"content\": question + ___ + now}`.",
        ],
        why="""```python
def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": question + "\\n\\nNow: " + now}]
```

Nu begint elke prompt met dezelfde tokens van het handboek, hoe laat het ook is.""",
    )
    return


@app.cell(hide_code=True)
def text_breaks():
    mo.md(r"""
    ### Oefening 3. Een groet met naam
    """)
    return


@app.cell(hide_code=True)
def choice_breaks_cell():
    choice_breaks = L.choice(
        "breaks",
        question="De winkel wil dat zijn assistent elke klant met naam begroet. Waar komt de naam, zodat de cache blijft werken?",
        options={
            "after": "In het bericht van de gebruiker, na het hele handboek",
            "top": "Bovenaan: *You are talking to Sam.*",
            "both": "Het maakt niet uit: de server cachet de prompt van elke klant apart",
        },
        correct="after",
        why={
            "after": "Het handboek blijft voor elke klant hetzelfde begin, dus hergebruikt de server het; alleen de tokens vanaf de naam zijn nieuw.",
            "top": "Dan begint de prompt voor elke klant anders, en leest de server het hele handboek elke keer opnieuw, net als met de tijd ervoor. Wat verandert, komt als laatste.",
            "both": "De server van de course bewaart alleen de cache van zijn laatste request, en een gehoste API vergelijkt ook het begin van de prompt: met de naam op de eerste regel delen geen twee klanten een begin. Wat verandert, komt als laatste.",
        },
    )
    choice_breaks
    return (choice_breaks,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 2 goed is, kun je vragen met je eigen `messages_for` naar het echte model sturen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo stuurt elke vraag met de tijd van je computer, na elkaar, de eerste met de cache uitgezet, en laat bij elke zien hoeveel tokens uit de cache kwamen, en hoe lang het duurde. Je computer is niet die van de course, dus je seconden verschillen. Schrijf je vragen in het Engels, omdat Qwen3 Engels het best schrijft en het handboek Engels is.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def questions_cell():
    questions_box = mo.ui.text_area(value="\n".join(C.QUESTIONS), label="Je vragen, één per regel", full_width=True, rows=4)
    send_button = mo.ui.run_button(label="Verstuur met je messages_for")
    mo.vstack([questions_box, send_button])
    return questions_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_order, questions_box, send_button):
    _fn = C.learner_messages_for(box_order.value)
    _questions = [q.strip() for q in questions_box.value.splitlines() if q.strip()]
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo maakt de berichten met je eigen `messages_for`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te versturen.*")
    elif not _questions:
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vragen gaan na elkaar; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Je vragen, na elkaar",
                  "done": "{given} tokens erin, waarvan {cached} uit de cache, in {seconds} seconden:",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, _questions, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het context window van [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl) wordt bij elk request opnieuw gelezen, en prompt caching maakt het deel dat hetzelfde blijft goedkoop om opnieuw te lezen. De system prompt van [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) is meestal dat deel, en de input tokens van [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl) zijn wat de cache bespaart. Met deze les is de module over prompts af; de volgende module zoekt teksten op die bij een vraag passen en zet ze in de prompt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_breaks, result_order, result_shared):
    L.wrap_up(result_shared, result_order, choice_breaks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
