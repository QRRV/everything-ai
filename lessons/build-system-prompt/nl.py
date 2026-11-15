# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-system-prompt. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-system-prompt", "nl")
    C = L.checks   # de opgenomen runs van de course, de stand-in van de checks, en de demo met je with_system


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_message():
    quick_message = L.choice(
        "quick-message",
        question="Wat is één bericht van het gesprek, in een request aan `/v1/chat/completions`?",
        options={
            "text": "Een tekst die begint met de rol en een dubbele punt, zoals `user: Hi`",
            "dict": "Een dictionary met `\"role\"` en `\"content\"`",
            "ids": "Een lijst met de token-ID's van het bericht",
        },
        correct="dict",
        why={
            "dict": "Zoals `{\"role\": \"user\", \"content\": \"Hi\"}`; het gesprek is een lijst daarvan, in `\"messages\"`.",
            "text": "Uiteindelijk krijgt het model het gesprek als één tekst, maar die schrijft de server uit; het request houdt de berichten apart. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "ids": "De tokens maakt de server zelf; het request bevat tekst. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_message
    return (quick_message,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een vraag alleen maakt zelden een app. De assistent van een winkel hoort kort te antwoorden en bij zijn onderwerp te blijven, in elk gesprek, wat de klant ook als eerste typt. En als een programma het antwoord leest, moet het antwoord beginnen zoals het programma verwacht. Geen van beide hoort in wat de gebruiker typt: de app voegt het toe. De system prompt bevat de instructies voor het hele gesprek, en een response prefill schrijft de eerste woorden van het antwoord.

    **Een voorbeeld uit het dagelijks leven.** Vóór een dienst krijgt een nieuwe medewerker aan de balie van een winkel een briefing: beantwoord alleen vragen over fietsen, en houd het kort. Elke klant vraagt iets anders; de briefing geldt voor allemaal, en de klanten horen hem nooit. En op een formulier waar *Naam:* al voorgedrukt staat, komt daarna een naam, geen verhaal. Dat is maar een vergelijking; hieronder zie je wat een system prompt en een prefill met de antwoorden van Qwen3 deden, en waar ze tekortschoten.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **system prompt** | de instructies in het bericht met de rol `system`, aan het begin van het gesprek; de bouwer van de app schrijft ze, de gebruiker ziet ze niet, en ze gelden voor het hele gesprek; ze gaan met elk request mee, als input tokens |
    | **response prefill** | de eerste woorden van het antwoord van het model, door jou geschreven: een laatste bericht met de rol `assistant`, waar het model op verdergaat; het antwoord bevat alleen wat het model daarna schrijft |

    Sommige API's zeggen *system instructions* of *developer message* voor de system prompt (voorbeelden in november 2026); het idee is hetzelfde.

    ### Een korte system prompt

    De course vroeg Qwen3 via zijn eigen server *What is a neural network?*, vier keer zonder system prompt en vier keer met een. De vragen zijn Engels, omdat Qwen3 Engels het best schrijft. De server trekt elk volgend token uit de waarschijnlijkste, dus vier antwoorden op dezelfde vraag verschillen een beetje:
    """)
    return


@app.cell(hide_code=True)
def picture_brief():
    mo.Html(C.brief_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_brief():
    _f = C.findings()
    mo.md(r"""
    Zonder system prompt liepen alle vier de antwoorden door tot `max_tokens` 200 en werden ze afgekapt, midden in kopjes en lijstjes: uit zichzelf legt Qwen3 uitgebreid uit. Met *Answer in one sentence of at most 20 words.* hadden ze {lo} tot {hi} tokens, en beëindigde Qwen3 ze alle vier zelf. Tellen deed het niet goed: {within} van de vier bleven binnen 20 woorden, en één had er {most}. De system prompt is ook niet gratis: hij gaat met elk request mee, dus de vraag was zonder {plain} input tokens, en met {brief}.

    ### Een system prompt die grenzen stelt

    Een system prompt kan ook zeggen wat de assistent niet moet doen. De course gaf Qwen3 de system prompt van een fietsenwinkel, en stelde twee vragen elk vier keer: één over fietsen, en één om een gedicht:
    """.format(lo=min(_f["brief_tokens"]), hi=max(_f["brief_tokens"]), within=sum(w <= 20 for w in _f["brief_words"]),
               most=max(_f["brief_words"]), plain=_f["prompt_tokens"][0], brief=_f["prompt_tokens"][1]))
    return


@app.cell(hide_code=True)
def picture_shop():
    mo.Html(C.shop_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_shop():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Qwen3 schreef het gedicht in {poem} van de {runs} runs, en weigerde {bike} van de {runs} vragen over fietsen, die het had moeten beantwoorden: de regel hield geen stand, in geen van beide richtingen. Wat het over banden zei, is ook niet veel waard; een model dat zo klein is, weet weinig.
        """.format(poem=_f["runs"] - _f["poem_refused"], bike=_f["bike_refused"], runs=_f["runs"])),
        L.watch_out("Een system prompt is geen slot, en geen geheim. Een gebruiker kan iets typen waardoor een model hem negeert, en kan het model hem vaak woord voor woord laten herhalen. Wat moet gelden, zoals wie wat mag zien, hoort in de code van je app, niet in een prompt."),
    ])
    return


@app.cell(hide_code=True)
def text_prefill():
    mo.md(r"""
    ### Response prefill

    In [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl) zag je dat het model het gesprek als één tekst krijgt, die eindigt met de markering van de assistant, zodat wat daarna komt het antwoord is. Als het laatste bericht van je gesprek de rol `assistant` heeft, schrijft de server van de course dat bericht achter de markering en laat hij de beurt open: voor het model is zijn antwoord dan al begonnen met jouw woorden, en het gaat vanaf daar verder.
    """)
    return


@app.cell(hide_code=True)
def picture_prompt():
    mo.Html(C.prompt_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_prefill_runs():
    mo.md(r"""
    Het antwoord dat terugkomt, bevat alleen wat het model na jouw woorden schreef, hier `Paris"}`, dus je code zet de twee aan elkaar. De course vroeg Qwen3 om de hoofdstad van Frankrijk als JSON, vier keer zonder en vier keer met de prefill `{"city": "`:
    """)
    return


@app.cell(hide_code=True)
def picture_json():
    mo.Html(C.json_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_json_after():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Zonder de prefill stonden {fenced} van de vier antwoorden in een codeblok met backticks, dat `json.loads` niet kan lezen, en hadden er {capital} de key `capital` in plaats van `city`: geen van de vier was te lezen zoals gevraagd. Met de prefill alle vier: de openingsaccolade en de key stonden er al, dus het model hoefde alleen de rest in te vullen. Een prefill bepaalt hoe het antwoord begint, niet wat het zegt; de volgende les controleert een JSON-antwoord helemaal.
        """.format(fenced=_f["fenced"], capital=_f["capital"])),
        L.watch_out("Niet elke API neemt een response prefill. De server van llama.cpp leest een laatste bericht van de assistant net zo als de server van de course. De API van OpenAI heeft geen prefill. De API van Anthropic nam er een voor zijn oudere Claude-modellen, en weigert bij de nieuwere een request dat eindigt met een bericht van de assistant (voorbeelden in november 2026). Zonder prefill moeten een duidelijke system prompt en een controle van het antwoord in code het doen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: system prompt en response prefill

    1. **Voor het hele gesprek:** zet de instructies in een eerste bericht met de rol `system`. Het gaat met elk request mee, dus houd het kort en concreet: wat te doen, en in welke vorm.
    2. **Eén system prompt:** heeft het gesprek er al een, zet de nieuwe dan op zijn plaats, niet ernaast.
    3. **Voor hoe het antwoord begint:** voeg een laatste bericht toe met de rol `assistant` waar die eerste woorden in staan, en zet ze vóór het antwoord dat terugkomt.
    4. **Controleer in code wat moet gelden:** een system prompt is geen slot, en een prefill legt alleen het begin vast.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een app geeft elke tekst die een gebruiker stuurt een korte titel, als JSON die een programma leest: `{"title": ...}`. Welke berichten?

    ```python
    messages = [
        {"role": "system", "content": "Make a title of at most five words for the text the user sends."},
        {"role": "user", "content": "Yesterday it rained all day, so we stayed in, baked bread and played cards."},
        {"role": "assistant", "content": '{"title": "'},
    ]
    ```

    1. **Voor het hele gesprek:** de system prompt zegt wat elk antwoord moet zijn, welke tekst er ook komt.
    2. **Eén system prompt:** dit gesprek heeft geen andere.
    3. **Hoe het antwoord begint:** de prefill `{"title": "`. Qwen3 antwoordde `Rainy Day"}`, met `finish_reason` `"stop"`, en de app zet de twee aan elkaar: `'{"title": "' + 'Rainy Day"}'` is `{"title": "Rainy Day"}`.
    4. **Controleer in code:** `json.loads` leest het, en de titel heeft 2 woorden, binnen de vijf.
    """)
    return


@app.cell(hide_code=True)
def text_with():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een system prompt aan het begin

        Een app bewaart het gesprek vaak als lijst, en voegt de system prompt pas toe als hij het verstuurt: dan kan hij de system prompt veranderen zonder aan het gesprek te komen. Schrijf `with_system(messages, system)`, die een nieuwe lijst teruggeeft: eerst een bericht met de rol `system` en de tekst `system`, dan de berichten van `messages`, zonder een system prompt die het gesprek al had. Het vak begint met een functie die de lijst kopieert.
        """),
        L.task("Laat `new` beginnen met de system prompt als bericht, en laat de loop alleen de berichten kopiëren waarvan de rol niet `system` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_with():
    box_with = L.code_box("with_system", lines=8)
    box_with
    return (box_with,)


@app.cell(hide_code=True)
def check_with(box_with):
    result_with = L.check("with_system", box_with.value)
    result_with
    return (result_with,)


@app.cell(hide_code=True)
def hints_with():
    L.hints(
        "with_system",
        [
            "Twee veranderingen in het vak: waar `new` mee begint, en welke berichten de loop kopieert.",
            "`new` begint als een lijst met één dictionary erin, met de keys `\"role\"` en `\"content\"`. In de loop bepaalt een `if` op `message[\"role\"]` of een bericht in `new` komt.",
            "`new = [{\"role\": \"system\", \"content\": system}]`, en in de loop `if message[\"role\"] != \"system\":` boven `new.append(message)`.",
        ],
        why="""```python
def with_system(messages, system):
    new = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] != "system":
            new.append(message)
    return new
```

De functie maakt een nieuwe lijst en laat `messages` zoals hij was, zodat de app hetzelfde gesprek opnieuw kan sturen met een andere system prompt.""",
    )
    return


@app.cell(hide_code=True)
def text_json_exercise():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. JSON met een response prefill

        `ask_json(url, question, start)` stelt een vraag waarvan het antwoord JSON moet zijn. Stuur `start` mee als response prefill, en geef de hele JSON-tekst terug: `start`, en daarachter wat het model erna schreef. De check roept je `ask_json` aan tegen een stand-in van de server, die na een prefill verdergaat zoals de server van de course.
        """),
        L.task("Voeg de prefill toe als laatste bericht, en zet `start` vóór de tekst die terugkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_json():
    box_json = L.code_box("ask_json", lines=13)
    box_json
    return (box_json,)


@app.cell(hide_code=True)
def check_json(box_json):
    result_json = L.check("ask_json", box_json.value)
    result_json
    return (result_json,)


@app.cell(hide_code=True)
def hints_json():
    L.hints(
        "ask_json",
        [
            "De prefill is één bericht meer in `messages`, na de vraag. Het antwoord dat terugkomt, bevat alleen wat erna komt.",
            "Het bericht is `{\"role\": \"assistant\", \"content\": start}`, het tweede item van de lijst. De laatste regel zet `start +` vóór de tekst.",
            "`messages = [{\"role\": \"user\", \"content\": question}, {\"role\": \"assistant\", \"content\": start}]`, en `return start + answer[\"choices\"][0][\"message\"][\"content\"]`.",
        ],
        why="""```python
def ask_json(url, question, start):
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": start}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 40}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return start + answer["choices"][0]["message"]["content"]
```

Het model gaat verder vanaf `start`, en het antwoord bevat alleen de rest; samen zijn ze de JSON-tekst.""",
    )
    return


@app.cell(hide_code=True)
def text_secret():
    mo.md(r"""
    ### Oefening 3. Een code voor vaste klanten
    """)
    return


@app.cell(hide_code=True)
def choice_secret_cell():
    choice_secret = L.choice(
        "secret",
        question="De assistent van je winkel mag een kortingscode alleen geven aan klanten die al eens besteld hebben. Waar komt de code?",
        options={
            "system": "In de system prompt, met de regel *Geef hem alleen aan klanten die al eens bestelden*",
            "prefill": "In een response prefill, waar de gebruiker die typt hem nooit ziet",
            "code": "Niet in een prompt: je app kijkt zelf de bestellingen na, en geeft de code",
        },
        correct="code",
        why={
            "code": "Een system prompt is geen slot: het model van de fietsenwinkel schreef het gedicht toch, en een gebruiker kan een model zijn system prompt vaak laten herhalen. Wat moet gelden, doet je eigen code.",
            "system": "De system prompt van de fietsenwinkel zei *alleen fietsen*, en Qwen3 schreef het gedicht in elke run. Een regel in een prompt vraagt het model iets; hij sluit niets af. Waar kan geen prompt bij de code?",
            "prefill": "Het model leest een prefill zoals de rest van de prompt, en kan de code dus gewoon in zijn antwoord herhalen; en de prefill is het begin van het antwoord, dat je app misschien laat zien. Waar kan geen prompt bij de code?",
        },
    )
    choice_secret
    return (choice_secret,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 goed is, kun je system prompts en prefills op het echte model proberen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo zet je system prompt vooraan met je eigen `with_system`, en voegt het begin van het antwoord toe als prefill als je er een schrijft. Probeer de fietsenwinkel met andere vragen, of een vraag om JSON met een prefill.
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
def question_box_cell():
    system_box = mo.ui.text_area(value=C.SHOP, label="System prompt (mag leeg blijven)", full_width=True, rows=3)
    question_box = mo.ui.text(value=C.POEM, label="Je vraag", full_width=True)
    start_box = mo.ui.text(value="", label="Begin van het antwoord, de prefill (mag leeg blijven)", full_width=True)
    send_button = mo.ui.run_button(label="Verstuur met je with_system")
    mo.vstack([system_box, question_box, start_box, send_button])
    return question_box, send_button, start_box, system_box


@app.cell(hide_code=True)
def demo_answer(box_with, question_box, send_button, start_box, system_box):
    _with = C.learner_with_system(box_with.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _with is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo zet de system prompt erin met je eigen `with_system`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te versturen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De berichten zijn verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat er naar Qwen3 ging, en wat er terugkwam", "sent": "De berichten:",
                  "answer": "Het antwoord, met je prefill vetgedrukt:",
                  "done": "{given} tokens erin, {written} eruit, finish_reason \"{finish}\", in {seconds} seconden.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_with, system_box.value, question_box.value, start_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rol `system` van [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl) is nu een bericht dat je app aan elk request van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl) toevoegt, en de ene tekst van [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl) verklaart waarom een prefill werkt: het model gaat verder waar de tekst ophoudt. In de volgende les, [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl), controleer je een JSON-antwoord op wat erin moet staan, en vraag je opnieuw als het niet past.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_secret, result_json, result_with):
    L.wrap_up(result_with, result_json, choice_secret)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
