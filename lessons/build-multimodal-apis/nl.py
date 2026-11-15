# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-multimodal-apis. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-multimodal-apis", "nl")
    C = L.checks   # de opgenomen vragen van de course over foto's, de stand-in van de checks, en de demo met je ask_photo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_post():
    quick_post = L.choice(
        "quick-post",
        question="Wat maakt van `urllib.request.Request(url, data=body_bytes, headers={...})` een POST?",
        options={
            "headers": "De header `Content-Type: application/json`",
            "data": "De `data=` met de body",
            "url": "Het adres, dat eindigt op `/chat/completions`",
        },
        correct="data",
        why={
            "data": "Een request met `data=` stuurt een body mee, dus urllib maakt er een POST van; zonder `data=` is het een GET.",
            "headers": "De header zegt wat de body is, JSON, maar niet of er een is. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "url": "Het adres zegt waar het request heen gaat, niet hoe. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_post
    return (quick_post,)


@app.cell(hide_code=True)
def quick_check_choices():
    quick_choices = L.choice(
        "quick-choices",
        question="Waarom staat de tekst van een antwoord in `answer[\"choices\"][0][\"message\"][\"content\"]`, met een `[0]`?",
        options={
            "token": "`[0]` is het eerste token van het antwoord",
            "sent": "`[0]` is het eerste bericht dat je stuurde",
            "list": "`choices` is een lijst, omdat een request om meer dan één antwoord kan vragen",
        },
        correct="list",
        why={
            "list": "Een request kan om meerdere antwoorden op dezelfde vraag vragen; meestal is er één, de eerste, `[0]`.",
            "token": "De hele tekst staat in `\"content\"`, als één string. Wat voor waarde is `choices`? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "sent": "De berichten die je stuurde, staan in het request, niet in het antwoord. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_choices
    return (quick_choices,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel vragen bestaan niet alleen uit woorden. Een foto van een bonnetje om te lezen, een screenshot van een foutmelding, een spraakbericht, een grafiek in een rapport. De API's van grote modellen nemen die ook aan, in hetzelfde request dat je al kent: de content van een bericht kan een lijst met delen zijn, woorden en beelden naast elkaar. En voor een gesprek met je stem is er een soort API die open blijft terwijl je praat.

    **Een voorbeeld uit het dagelijks leven.** Je stuurt een vriend een foto van een plant, met eronder: "Welke plant is dit?". Eén bericht, met twee delen: een plaatje en een vraag. Je vriend kijkt naar allebei en antwoordt. Dat is alleen een vergelijking; hieronder zie je hoe een beeld en een vraag in één bericht naar een model gaan.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **multimodale API** | een API waarvan de berichten meer dan tekst dragen: beelden, geluid, soms video; een *multimodaal model* leest ze zelf |
    | **content part** | één stuk van de content van een bericht: in plaats van één string is de content een lijst met delen, elk met zijn `"type"`: `"text"` voor woorden, `"image_url"` voor een beeld, `"input_audio"` voor geluid |
    | **base64** | een manier om willekeurige bytes als tekst te schrijven, met 64 tekens (`A` tot `Z`, `a` tot `z`, `0` tot `9`, `+` en `/`): elke 3 bytes worden 4 tekens, dus de tekst is een derde langer dan het bestand |
    | **data-URL** | een adres waarin het bestand zelf zit: `data:image/png;base64,` en dan de base64 van het bestand; een browser laat het als beeld zien, een API leest het beeld eruit |
    | `open(path, "rb")` | opent een bestand om de bytes te lezen zoals ze zijn: `"r"` om te lezen, `"b"` voor bytes; zonder `"b"` leest Python letters, en de bytes van een beeld zijn geen letters |
    | `base64.b64encode(data)` | uit de eigen module `base64` van Python: geeft de base64 van de bytes `data`, als bytes; `.decode()` maakt daar een string van |
    | **hosted tool** | een tool die bij de aanbieder draait, niet in je programma: je noemt hem in de `"tools"` van het request, de server voert hem uit, en het antwoord zegt wat hij vond |
    | **realtime API** | een API met een verbinding die in beide richtingen open blijft: je programma stuurt geluid terwijl je praat, en het gesproken antwoord komt terug terwijl het gemaakt wordt |

    ### Een beeld in een bericht

    De server van de course draait Qwen3-0.6B, dat alleen tekst leest. Om toch over een beeld te kunnen antwoorden, heeft de server een hosted tool: een objectdetector, DETR, een model dat de voorwerpen op een foto vindt en ze een naam geeft (het domein over computer vision bouwt zulke detectors). De server laat hem op elk beeld los, en zet zijn woorden waar het beeld stond. Het request is hetzelfde als wat een multimodaal model krijgt:
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_three():
    mo.md(r"""
    De course stelde de server op deze manier drie vragen, elk over een van zijn foto's. Onder elke vraag: wat de detector vond, en wat Qwen3 antwoordde.
    """)
    return


@app.cell(hide_code=True)
def picture_recorded():
    mo.Html(C.recorded_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_more():
    mo.vstack([
        mo.md(r"""
        Een groot multimodaal model bij een gehoste API heeft geen detector ertussen nodig: het maakt van het beeld eigen tokens, en leest die samen met de vraag. Zo kan het ook de tekst op een bonnetje lezen, of zeggen welk van twee kopjes voller is. Het request dat je het stuurt, is hetzelfde, met dezelfde delen.

        ### Realtime: praten terwijl het luistert

        Een request en zijn antwoord gaan na elkaar: eerst de hele vraag, dan het antwoord. Voor een gesprek met je stem is dat traag en stroef. Een realtime API houdt één verbinding open in beide richtingen, via een *WebSocket* of *WebRTC*: je programma stuurt geluid terwijl je praat, de server hoort wanneer je stopt, en het gesproken antwoord komt terug terwijl het gemaakt wordt; begin je weer te praten, dan stopt het. De Realtime API van OpenAI en de Live API van Google werken zo (voorbeelden in november 2026). De server van de course heeft er geen: Qwen3-0.6B hoort niets.

        ### Hosted tools: de aanbieder voert ze uit

        Een tool is iets waar een model om kan vragen, zoals een zoekopdracht op het web. Meestal voert je eigen programma hem uit, zoals een latere les laat zien. Een hosted tool draait in plaats daarvan bij de aanbieder: je noemt hem alleen in het request, en het antwoord komt terug met de uitkomst erin. Zoeken op het web en code uitvoeren zijn bijvoorbeeld hosted tools bij OpenAI en bij Anthropic (voorbeelden in november 2026). De server van de course heeft er één: zijn detector.
        """),
        L.watch_out("Een beeld kost ook tokens, vaak meer dan de vraag erbij: een multimodaal model maakt er tokens van, en de documentatie van de aanbieder zegt hoeveel. En base64 maakt het beeld onderweg een derde groter."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een vraag stellen over een beeld

    1. **Lees de bytes van het bestand:** `with open(path, "rb") as f:` en dan `data = f.read()`.
    2. **Maak een data-URL:** `"data:image/png;base64," + base64.b64encode(data).decode()`.
    3. **Maak van de content een lijst met delen:** `[{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": the_data_url}}]`.
    4. **Verstuur het zoals eerst**, als content van het bericht van de gebruiker; vraag bij de server van de course om zijn hosted tool: `"tools": [{"type": "detect_objects"}]`.
    5. **Lees het antwoord zoals eerst:** de tekst staat in `answer["choices"][0]["message"]["content"]`; de server van de course zegt ook wat zijn tool vond, onder `"hosted_tools"`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vraag van de course over zijn foto van een kopje koffie, `data/photos/coffee.png`: het request dat zijn code stuurde, en wat er terugkwam.
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **De content:** een lijst met twee delen, de vraag als tekstdeel en de foto als beelddeel.
    2. **Het beeld:** het bestand heeft 466.706 bytes; als data-URL zijn dat 622.298 tekens, een derde meer, en het plaatje kort hem in.
    3. **De tool:** `"tools": [{"type": "detect_objects"}]` vraagt de server om eerst zijn detector op elk beeld los te laten.
    4. **Wat het model las:** de detector vond een kopje, een eettafel en een lepel, en de server zette de regel `[Image 1. The objects the server's detector found on it: cup, dining table, spoon.]` na de vraag. Dat zijn de 40 tokens onder `"prompt_tokens"`: de foto zelf heeft Qwen3 nooit bereikt.
    5. **Het antwoord:** Qwen3 schreef 24 tokens, op basis van die woorden.
    """)
    return


@app.cell(hide_code=True)
def text_data_url():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een beeld als data-URL

        `data_url(path)` leest de bytes van het PNG-bestand op `path` al. Maak er de data-URL van, zoals in stap 2 van het recept. De check roept hem aan met `"data/photos/fruit-sticker.png"`, een tekening van vier vruchten, en met een ander beeld; het plaatje onder de check laat je data-URL zien zoals een browser hem laat zien.
        """),
        L.task("Maak `data_url` af: maak van `data` base64-tekst, en geef de data-URL van het PNG-beeld terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_data_url():
    box_url = L.code_box("data_url", lines=7)
    box_url
    return (box_url,)


@app.cell(hide_code=True)
def check_data_url(box_url):
    result_url = L.check("data_url", box_url.value)
    result_url
    return (result_url,)


@app.cell(hide_code=True)
def hints_data_url():
    L.hints(
        "data_url",
        [
            "Twee stappen: eerst de base64 van de bytes, als string; dan het begin van een data-URL ervoor.",
            "`base64.b64encode(data)` geeft bytes; `.decode()` erachter maakt er een string van.",
            "`text = base64.b64encode(data).decode()`, en `return \"data:image/png;base64,\" + ...`, met `text` op de plaats van de puntjes.",
        ],
        why="""```python
def data_url(path):
    with open(path, "rb") as f:
        data = f.read()
    text = base64.b64encode(data).decode()
    return "data:image/png;base64," + text
```

Het begin zegt wat voor bestand het is, `image/png`, en dat er base64 volgt; na de komma komt het beeld zelf, als tekst.""",
    )
    return


@app.cell(hide_code=True)
def text_ask_photo():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een vraag over een beeld

        `ask_photo(url, path, question)` verstuurt al een request en leest het antwoord, zoals je `ask` deed. Vul de content in, met de vraag en het beeld op `path` als twee delen, en vraag om de hosted tool van de server. Je `data_url` uit oefening 1 staat klaar om te gebruiken. De check roept je functie aan met een stand-in van de server, die meteen antwoordt; het plaatje onder de check laat zien wat je functie stuurde, met de data-URL ingekort.
        """),
        L.task("Maak van `content` een lijst met een tekstdeel en een beelddeel, en van `\"tools\"` een lijst met de hosted tool `detect_objects`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask_photo():
    box_photo = L.code_box("ask_photo", lines=13)
    box_photo
    return (box_photo,)


@app.cell(hide_code=True)
def check_ask_photo(box_photo, box_url):
    result_photo = L.check("ask_photo", box_photo.value, C.photo_inputs(box_url.value))
    result_photo
    return (result_photo,)


@app.cell(hide_code=True)
def hints_ask_photo():
    L.hints(
        "ask_photo",
        [
            "Stap 3 en 4 van het recept: de content wordt een lijst met twee delen, en `\"tools\"` een lijst met één tool erin.",
            "Het tekstdeel is `{\"type\": \"text\", \"text\": question}`. Het beelddeel heeft `\"type\": \"image_url\"`, en onder `\"image_url\"` een dictionary met de key `\"url\"` en de data-URL van `path`.",
            "`content = [{\"type\": \"text\", \"text\": question}, {\"type\": \"image_url\", \"image_url\": {\"url\": ...}}]`, met `data_url(path)` op de plaats van de puntjes, en `\"tools\": [{\"type\": ...}]`, met de naam van de hosted tool.",
        ],
        why="""```python
def ask_photo(url, path, question):
    content = [{"type": "text", "text": question},
               {"type": "image_url", "image_url": {"url": data_url(path)}}]
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": content}],
            "max_tokens": 60, "tools": [{"type": "detect_objects"}]}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

De vraag en het beeld gaan in één bericht, als twee delen; de tool vraagt de server om te benoemen wat er op het beeld staat, voordat het model het leest.""",
    )
    return


@app.cell(hide_code=True)
def text_fruit():
    mo.md(r"""
    ### Oefening 3. Waar komt de appel vandaan?

    Kijk nog eens naar de derde vraag van de course, over de tekening van fruit: vier rode appels, een sinaasappel, een citroen en een limoen.
    """)
    return


@app.cell(hide_code=True)
def choice_fruit_cell():
    choice_fruit = L.choice(
        "fruit",
        question="De detector vond `sports ball (7)` op de tekening, en Qwen3 antwoordde dat het fruit een appel is. Waar komt die appel vandaan?",
        options={
            "guess": "Qwen3 kreeg alleen de woorden `sports ball (7)` en het woord fruit in de vraag, en gokte een ronde vrucht",
            "saw": "Qwen3 zag de rode appels op het beeld",
            "detector": "De detector vertelde Qwen3 dat de ballen appels zijn",
        },
        correct="guess",
        why={
            "guess": "Het model ziet het beeld nooit, alleen de woorden van de detector, en die kent geen appels: een gok die toevallig klopte, geen blik. Een multimodaal model zou het beeld zelf lezen.",
            "saw": "Kijk naar het plaatje met de pijlen: wat bereikt Qwen3, het beeld of iets in zijn plaats?",
            "detector": "Kijk wat de detector vond, onder de vraag: welk woord gebruikte hij voor elk van de zeven?",
        },
    )
    choice_fruit
    return (choice_fruit,)


@app.cell(hide_code=True)
def text_realtime():
    mo.md(r"""
    ### Oefening 4. Welke API voor een spraakassistent?
    """)
    return


@app.cell(hide_code=True)
def choice_realtime_cell():
    choice_realtime = L.choice(
        "realtime",
        question="Je bouwt een spraakassistent voor in de auto: de bestuurder praat, hoort het antwoord terwijl het gemaakt wordt, en kan ertussen komen. Welke API past?",
        options={
            "chat": "Het chat-endpoint, met het geluid als `input_audio`-deel",
            "realtime": "Een realtime API, met een verbinding die in beide richtingen open blijft",
            "hosted": "Een hosted tool die naar het geluid luistert",
        },
        correct="realtime",
        why={
            "realtime": "Alleen een open verbinding laat het geluid erin gaan terwijl de bestuurder praat, het antwoord eruit komen terwijl het gemaakt wordt, en stoppen als de bestuurder ertussen komt.",
            "chat": "Een request gaat pas weg als de hele vraag binnen is, en het antwoord komt terug op dat ene request. Kan de bestuurder ertussen komen?",
            "hosted": "Een hosted tool draait tijdens één request, bij de aanbieder. Gaat het gesprek daardoor tegelijk beide kanten op?",
        },
    )
    choice_realtime
    return (choice_realtime,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag het over een foto met je eigen ask_photo

    Zodra oefening 1 en 2 goed zijn, kun je `ask_photo` met de echte server laten praten: de course start hem op je computer, met DETR als hosted tool en Qwen3-0.6B erachter. Kies een foto, stel je vraag, en vergelijk wat de detector vond met het antwoord. De vraag staat in het Engels, net als de woorden van de detector; een vraag in het Nederlands mag ook, maar een model zo klein als Qwen3-0.6B antwoordt in het Engels vaak beter.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B en DETR (1,7 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*De demo heeft twee modellen nodig die nog niet allebei op je computer staan: het chat model Qwen3-0.6B, en DETR, de detector van de hosted tool van de server. De knop downloadt wat ontbreekt één keer van Hugging Face, in vaste versies: `Qwen/Qwen3-0.6B`, 1,5 GB, met de tokenizer van Qwen3-0.6B-Base, en `facebook/detr-resnet-50`, 167 MB, allebei onder de Apache-2.0-licentie.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: een bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B en DETR zijn gedownload.*")
    return


@app.cell(hide_code=True)
def photo_box_cell():
    photo_box = mo.ui.dropdown(options=C.DEMO_PHOTOS["nl"], value="een kopje koffie", label="Foto")
    question_box = mo.ui.text(value="What is in this picture?", label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Vraag het met je ask_photo")
    mo.vstack([photo_box, question_box, send_button])
    return photo_box, question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_photo, box_url, photo_box, question_box, send_button):
    _ask = C.learner_ask_photo(box_photo.value, box_url.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst de twee modellen, met de knop hierboven.*")
    elif _ask is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo vraagt het met je eigen `ask_photo`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om je vraag te stellen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `ask_photo` heeft de foto en de vraag verstuurd; de server laadt DETR en Qwen3, benoemt wat er op de foto staat, en Qwen3 schrijft het antwoord …*"))
        _words = {"title": "Wat er terugkwam bij je ask_photo",
                  "done": "{seconds} seconden, met de detector erbij.",
                  "failed": "Je ask_photo stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, photo_box.value, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het request is dat van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met een lijst met delen als content; de tokens van een beeld tellen mee in de prijs, zoals in [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl). De volgende les, [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl), maakt via een API vectoren van teksten, en stuurt documenten mee met een request. De tools die je eigen programma uitvoert, komen in een latere les over tool calling, en het domein over computer vision bouwt detectors zoals DETR.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fruit, choice_realtime, result_photo, result_url):
    L.wrap_up(result_url, result_photo, choice_fruit, choice_realtime)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
