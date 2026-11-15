# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-streaming. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-streaming", "nl")
    C = L.checks   # de opgenomen streams van de course, de stand-in van de checks, en de demo met jouw piece


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
def quick_check_text():
    quick_text = L.choice(
        "quick-text",
        question="Waar staat de tekst van het antwoord, in de JSON `answer` die een server terugstuurt?",
        options={
            "text": "`answer[\"text\"]`",
            "choices": "`answer[\"choices\"][0][\"message\"][\"content\"]`",
            "usage": "`answer[\"usage\"][\"content\"]`",
        },
        correct="choices",
        why={
            "choices": "`choices` is een lijst, omdat een request om meer dan één antwoord kan vragen; het eerste heeft de message, en zijn `content` is de tekst.",
            "text": "Het antwoord heeft bovenaan geen key `\"text\"`. Welke lijst bevat de antwoorden? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "usage": "`usage` telt alleen de tokens erin en eruit. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_text
    return (quick_text,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="Je `ask` uit die les verstuurt een vraag. Wanneer krijgt hij het eerste woord van het antwoord?",
        options={
            "end": "Pas als het hele antwoord geschreven is",
            "start": "Zodra het model het schrijft",
            "before": "Voordat het model begint te schrijven",
        },
        correct="end",
        why={
            "end": "`response.read()` leest het hele antwoord, en de server stuurt het pas als het model klaar is: één JSON met alles erin.",
            "start": "Dan zou de server het antwoord in stukjes moeten sturen, en zou `ask` ze een voor een moeten lezen. Wat leest `response.read()`? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "before": "Het model moet de woorden eerst schrijven. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model schrijft zijn antwoord token voor token, en een lang antwoord duurt seconden, op een kleine computer meer dan tien. Je `ask` wacht al die tijd en laat niets zien, en dan alles tegelijk. Elke chat-app doet het anders: de eerste woorden verschijnen bijna meteen, en de rest volgt terwijl je leest. Dat is geen sneller model; de server stuurt elk stukje zodra het geschreven is, en de app laat het zien.

    **Een voorbeeld uit het dagelijks leven.** Een vriend vertelt je aan de telefoon een verhaal: je hoort elke zin op het moment dat hij gezegd wordt. Met een spraakbericht hoor je niets tot je vriend klaar is, en dan het hele verhaal. Het verhaal duurt even lang; aan de telefoon begin je alleen eerder. Dat is maar een vergelijking; hieronder zie je wanneer de stukjes van een echt antwoord binnenkwamen.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **streaming** | het antwoord van een model stukje voor stukje binnenkrijgen terwijl het nog geschreven wordt, in plaats van op het hele antwoord te wachten |
    | **server-sent events** | *SSE*: een manier waarop een server kan blijven sturen: de response blijft open, en elk *event* is een regel die begint met `data: `, met een lege regel erna |
    | `"stream": True` | in de body van een request: stuur het antwoord als server-sent events |
    | **delta** | in een event alleen het nieuwe stukje: `{"content": " keeper"}`, niet het hele bericht tot nu toe |
    | `data: [DONE]` | de laatste regel van de stream: het antwoord is compleet |
    | `.decode()` | maakt van bytes weer tekst, andersom dan `.encode()` |

    ### Wanneer de stukjes binnenkwamen

    De course vroeg Qwen3-0.6B via zijn eigen server om een kort verhaal, met `"stream": True`, drie keer. Elk streepje is een stukje van het eerste antwoord, op de seconde waarop het binnenkwam:
    """)
    return


@app.cell(hide_code=True)
def picture_timeline():
    mo.Html(C.timeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_timeline():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    Het eerste stukje kwam na {_n['first']} seconde, het einde na {_n['seconds']}: {_n['pieces']} stukjes, voor de {_n['tokens']} tokens van het antwoord. Zonder stream laat hetzelfde antwoord die {_n['seconds']} seconden niets zien. Over de {_n['runs']} runs van de course kwam het eerste stukje altijd binnen {_n['first_max']} seconde, en geen antwoord was in minder dan {_n['seconds_min']} seconden compleet. Dat is de winst: het wachten tot de eerste tekst, niet het hele antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_lines():
    mo.md(r"""
    ### Wat er over de lijn komt

    Dit is het begin en het einde van die stream, zoals de server hem stuurde. Elk event is één regel: `data: `, en dan een stuk JSON.
    """)
    return


@app.cell(hide_code=True)
def picture_lines():
    mo.Html(C.lines_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_parts():
    mo.vstack([
        mo.md(r"""
        Elk event heeft de vorm van een antwoord, met `choices`, maar in plaats van een `message` heeft het een `delta`: alleen wat nieuw is. De eerste delta heeft de rol en nog geen tekst; dan komen de stukjes, elk in `"content"`; de laatste delta is leeg, met de `finish_reason` ernaast. Met `"stream_options": {"include_usage": True}` in de body telt nog één event de tokens. En `data: [DONE]` sluit de stream af.

        Je code leest een stream zoals hij een bestand leest: een `for`-loop over de response geeft één regel tegelijk, elk met `"\n"` aan het eind, en een lege regel na elk event. Elke regel komt binnen als bytes, dus `.decode()` maakt er eerst tekst van.
        """),
        L.watch_out("Alleen het deel na `data: ` is JSON. `json.loads` op de hele regel, op een lege regel of op `data: [DONE]` stopt met een `JSONDecodeError`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een antwoord lezen als stream

    1. **Vraag om een stream:** zet `"stream": True` in de body.
    2. **Verstuur het zoals eerst:** hetzelfde request, dezelfde `urlopen`.
    3. **Ga door de regels:** `for line in response:`, en `line.decode()` voor de tekst.
    4. **Sla over wat geen stukje is:** een regel die niet met `data: ` begint, en `data: [DONE]`.
    5. **Lees het event:** `json.loads(text[6:])`, het deel na de zes tekens van `data: `, en dan `["choices"][0]["delta"]`.
    6. **Gebruik het stukje meteen:** als `"content"` in de delta zit, laat het zien, of bewaar het.

    In november 2026 doet het package `openai` stap 3 tot 5 voor je: met `stream=True` geeft het de events een voor een in een `for`-loop.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén regel van een stream, zoals hij uit de response komt, met de hand gelezen. De regel is ingekort: een echte heeft ook de id van het antwoord, het model en de tijd.

    ```python
    import json

    line = b'data: {"choices": [{"index": 0, "delta": {"content": " Every"}, "finish_reason": null}]}\n'
    text = line.decode()
    chunk = json.loads(text[6:])
    chunk["choices"][0]["delta"]
    ```

    1. **De bytes:** de `b` ervoor betekent dat de regel bytes is, zoals hij van het netwerk komt; hij eindigt op `\n`.
    2. **De tekst:** `line.decode()` geeft dezelfde tekens als string; `text[:6]` is `'data: '`.
    3. **De JSON:** `text[6:]` is alles na die zes tekens; `json.loads` leest het, en de `\n` aan het eind stoort niet.
    4. **Het stukje:** de eerste choice heeft de delta `{'content': ' Every'}`: de spatie ervoor hoort bij het stukje, dus de stukjes aan elkaar geven de tekst.
    """)
    return


@app.cell(hide_code=True)
def text_piece():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén regel van de stream

        Schrijf `piece(line)`: hij krijgt één regel van een stream als tekst, en geeft het stukje van het antwoord terug dat erin staat, of `""` als de regel geen stukje heeft. De regels komen zoals ze uit een response komen: met `"\n"` aan het eind, lege regels, de eerste delta met de rol en een lege `content`, de laatste met alleen `finish_reason`, en `data: [DONE]`. Voor de regel van het uitgewerkte voorbeeld geeft `piece` `' keeper'`.

        ```python
        piece('data: {"choices": [{"index": 0, "delta": {"content": " keeper"}, "finish_reason": null}]}')
        ```
        """),
        L.task("Maak `piece` af: geef `\"\"` terug voor een regel die geen event is of `data: [DONE]` is, en anders de `\"content\"` van de delta, als die er is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_piece():
    box_piece = L.code_box("piece", lines=10)
    box_piece
    return (box_piece,)


@app.cell(hide_code=True)
def check_piece(box_piece):
    result_piece = L.check("piece", box_piece.value)
    result_piece
    return (result_piece,)


@app.cell(hide_code=True)
def hints_piece():
    L.hints(
        "piece",
        [
            "Stap 4 en 5 van het recept: eerst de regels zonder stukje, dan de JSON.",
            "`line[:6]` is het begin van de regel: als dat niet `\"data: \"` is, of `line.strip()` is `\"data: [DONE]\"`, return dan `\"\"`. Daarna `chunk = json.loads(line[6:])`.",
            "`delta = chunk[\"choices\"][0][\"delta\"]`; dan `if \"content\" in delta:` return `delta[\"content\"]`, en daarna `return \"\"`.",
        ],
        why="""```python
def piece(line):
    if line[:6] != "data: " or line.strip() == "data: [DONE]":
        return ""
    chunk = json.loads(line[6:])
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        return delta["content"]
    return ""
```

Alleen het deel na `data: ` is JSON, en niet elke delta heeft een `"content"`: de eerste heeft de rol, de laatste zegt alleen waarom het antwoord stopte.""",
    )
    return


@app.cell(hide_code=True)
def text_stream():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De hele stream

        `stream(url, question)` verstuurt een vraag, en geeft de lijst van de stukjes van het antwoord terug, in de volgorde waarin ze kwamen, zonder de lege. Het request maakt hij al; vraag om een stream, en ga door de regels van de response met je `piece` uit oefening 1. De check roept je `stream` aan met een stand-in van de server, die zijn antwoord woord voor woord stuurt.
        """),
        L.task("Vraag in de body om een stream, en vul de loop in: elke regel door `piece`, en bewaar elk stukje met tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stream():
    box_stream = L.code_box("stream", lines=14)
    box_stream
    return (box_stream,)


@app.cell(hide_code=True)
def check_stream(box_piece, box_stream):
    result_stream = L.check("stream", box_stream.value, C.stream_inputs(box_piece.value))
    result_stream
    return (result_stream,)


@app.cell(hide_code=True)
def hints_stream():
    L.hints(
        "stream",
        [
            "Stap 1 en 3 van het recept: de body vraagt om de stream, de loop leest hem.",
            "In de body: `\"stream\": True`. In de `with`: `for line in response:`, en in die loop `text = piece(line.decode())`.",
            "Na `text = piece(line.decode())`, nog in de loop: `if text:` en dan `pieces.append(text)`. De `return pieces` staat er al.",
        ],
        why="""```python
def stream(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 200,
            "stream": True}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    pieces = []
    with urllib.request.urlopen(request) as response:
        for line in response:
            text = piece(line.decode())
            if text:
                pieces.append(text)
    return pieces
```

De loop krijgt elke regel op het moment dat hij binnenkomt; een app zou `text` daar meteen laten zien, in plaats van het te bewaren.""",
    )
    return


@app.cell(hide_code=True)
def text_when():
    mo.md(r"""
    ### Oefening 3. Wanneer helpt een stream niet?
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when",
        question="Bij welke van deze helpt streaming het minst?",
        options={
            "chat": "Een chatvenster, waarin iemand het antwoord leest",
            "json": "Een programma dat het hele antwoord nodig heeft, als JSON, voordat het er iets mee kan",
            "story": "Een tool die een lang verhaal schrijft voor de gebruiker",
        },
        correct="json",
        why={
            "json": "Een halve JSON is niet te lezen: het programma wacht toch op het einde, en het hele antwoord duurt met een stream even lang.",
            "chat": "Daar begint iemand na het eerste stukje al te lezen, in plaats van op het einde te wachten. Wie heeft iets aan een half antwoord?",
            "story": "Bij een lang antwoord is het wachten juist het langst; met een stream leest de gebruiker mee. Wie heeft iets aan een half antwoord?",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Kijk hoe Qwen3 schrijft, door je piece

    Zodra oefening 1 goed is, streamt de course een antwoord van Qwen3-0.6B van zijn eigen server op je computer, en stuurt elke regel door je `piece`. Wat die teruggeeft, verschijnt meteen.
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
    question_box = mo.ui.text(value="Schrijf een kort verhaal van vijf zinnen over een vuurtorenwachter.", label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Stream het door je piece")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_piece, question_box, send_button):
    _piece = C.learner_piece(box_piece.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _piece is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo leest de stream met je eigen `piece`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om het antwoord te streamen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3, en het eerste stukje volgt kort daarna …*"))
        _words = {"title": "Wat je piece uit de stream las",
                  "writing": "Qwen3 schrijft; elk stukje verschijnt op het moment dat je piece het gelezen heeft …",
                  "done": "Het eerste stukje kwam na {first} seconde, het einde na {seconds}.",
                  "failed": "Je piece stopte met een fout op een regel van de echte stream: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_piece, question_box.value, lambda _h: mo.output.replace(mo.Html(_h)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een stream is hetzelfde request als in [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met één key meer, en elk event is JSON zoals in [Data als tekst: JSON](/course/open/py-json?lang=nl), regel voor regel gelezen zoals in [Een tekstbestand lezen](/course/open/py-files?lang=nl). De stukjes zijn de tokens die het model een voor een schrijft. In de volgende les, [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl), zegt de server nee, of antwoordt hij helemaal niet, en gaat je code daarmee om.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_piece, result_stream):
    L.wrap_up(result_piece, result_stream, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
