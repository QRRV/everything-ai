# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-call-api. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-call-api", "nl")
    C = L.checks   # de opgenomen aanroep van de course, de stand-in van de checks, en de demo met jouw ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Data als tekst: JSON](/course/open/py-json?lang=nl), en één over [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loads():
    quick_loads = L.choice(
        "quick-loads",
        question="Wat geeft `json.loads('{\"n\": 3}')`?",
        options={
            "text": "De tekst `'{\"n\": 3}'`, onveranderd",
            "dict": "De dictionary `{'n': 3}`",
            "error": "Een `JSONDecodeError`",
        },
        correct="dict",
        why={
            "dict": "`json.loads` maakt van JSON-tekst Python-waarden: een JSON-object wordt een dictionary.",
            "text": "Dat zou `json.dumps` de andere kant op doen. Waar maakt `loads` JSON-tekst van? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
            "error": "De tekst is geldige JSON: dubbele aanhalingstekens om de key, een getal als waarde. Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
        },
    )
    quick_loads
    return (quick_loads,)


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="Hoe schrijf je in code een gesprek voor een chat model?",
        options={
            "string": "Als één string met het hele gesprek",
            "people": "Als een dictionary met een key per persoon",
            "list": "Als een lijst dictionaries, elk met een `\"role\"` en een `\"content\"`",
        },
        correct="list",
        why={
            "list": "Elk bericht is een dictionary met zijn rol, `\"user\"`, `\"assistant\"` of `\"system\"`, en zijn inhoud, in de volgorde waarin ze gezegd zijn.",
            "string": "Het model krijgt uiteindelijk wel één tekst, maar die plakt je code niet zelf in elkaar. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
            "people": "Een dictionary heeft geen volgorde van beurten, en één persoon zegt veel dingen. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een app met een model erin, een chat op een website of een tool in je editor, draait het model zelden zelf. Hij stuurt de vraag over het netwerk naar een server die het model draait, en krijgt het antwoord terug. Hoe je dat vraagt, ligt vast, en bijna elke server voor LLM's spreekt dezelfde vorm: die van de API van OpenAI. Dezelfde tien regels bereiken dus een model in een datacenter aan de andere kant van de wereld, of Qwen3-0.6B op je eigen computer.

    **Een voorbeeld uit het dagelijks leven.** Aan de balie van een kantine vul je een bestelbriefje in: wat je wilt, en hoeveel. De keuken maakt het, en je krijgt een dienblad terug met een bonnetje erop. Je komt nooit in de keuken, en elke keuken die hetzelfde briefje aanneemt, kan je helpen. Dat is alleen een vergelijking; hieronder zie je het echte briefje, en wat er terugkomt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **API** | *application programming interface*: een vaste manier waarop het ene programma het andere iets vraagt; bij een model stuur je berichten en krijg je een antwoord |
    | **HTTP-request** | een bericht aan een server over het netwerk: een *methode* (GET om iets op te halen, POST om iets te sturen), een adres, *headers* met gegevens over het request, zoals wat de body is, en een *body* |
    | **endpoint** | één adres van een API, voor één taak: `/v1/chat/completions` voor antwoorden in een chat, `/v1/models` voor de modellen die een server heeft |
    | **lokale server** | een server op je eigen computer, op het adres `127.0.0.1`, ook *localhost* genoemd, en een poortnummer zoals `8765`; er gaat niets over het internet |
    | **OpenAI-compatible API** | een API met de endpoints en de JSON van de API van OpenAI; Ollama, llama.cpp, vLLM, veel gehoste API's en de eigen server van de course spreken hem (voorbeelden in november 2026) |
    | `urllib.request` | de eigen module van Python voor HTTP: `Request(url, data=..., headers=...)` maakt een request, `urlopen(request)` verstuurt hem en geeft het antwoord |

    ### Wat er over de lijn gaat

    De course startte zijn eigen server op zijn computer, met Qwen3-0.6B erachter, en stuurde hem één vraag. Dit is het request, en het antwoord dat de server terugstuurde:
    """)
    return


@app.cell(hide_code=True)
def picture_call():
    mo.Html(C.call_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_parts():
    mo.vstack([
        mo.md(r"""
        Het request is een POST naar het endpoint `/v1/chat/completions`, met één header: de body is JSON. De body zegt welk model, welke berichten, en hoeveel tokens het hoogstens mag schrijven. In het antwoord is `choices` een lijst, omdat een request om meer dan één antwoord kan vragen; de eerste, `[0]`, heeft de `message` met zijn `content`, de tekst. `finish_reason` zegt waarom het model stopte: `"stop"` als het het antwoord zelf beëindigde, `"length"` als het `max_tokens` bereikte. En `usage` telt de tokens: er gingen er 19 in en er kwamen er 13 uit, waar je bij een gehoste API voor betaalt, zoals in [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl).

        Een request gaat als bytes over het netwerk, niet als Python-tekst. `.encode()` maakt van een string bytes; `response.read()` geeft het antwoord als bytes, en `json.loads` leest die meteen.
        """),
        L.watch_out("De server van de course luistert alleen op `127.0.0.1`: geen andere computer kan hem bereiken, dus hij heeft geen key nodig. Een gehoste API staat open voor het hele internet, en wil bij elk request een key; een latere les laat zien hoe je die buiten je code houdt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: de API van een model aanroepen met urllib

    1. **Maak de body:** een dictionary met `"model"`, `"messages"` en `"max_tokens"`.
    2. **Maak er JSON-bytes van:** `json.dumps(body).encode()`.
    3. **Maak het request:** `urllib.request.Request(url + "/chat/completions", data=..., headers={"Content-Type": "application/json"})`. Met `data=` is het een POST.
    4. **Verstuur het, en lees het antwoord:** `with urllib.request.urlopen(request) as response:` en dan `answer = json.loads(response.read())`.
    5. **Haal de tekst eruit:** `answer["choices"][0]["message"]["content"]`.

    Buiten de course gebruik je hiervoor vaak het package `openai` van OpenAI (een voorbeeld in november 2026), dat deze stappen voor je doet: met `OpenAI(base_url=url, api_key=...)` praat het met elke OpenAI-compatible server. Daaronder verstuurt het hetzelfde request.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Welke modellen heeft de server van de course? Dat is het endpoint `/v1/models`, en ophalen is een GET: een request zonder `data=`.

    ```python
    import json
    import urllib.request

    url = "http://127.0.0.1:8765/v1"
    with urllib.request.urlopen(url + "/models") as response:
        models = json.loads(response.read())
    models["data"][0]["id"]
    ```

    1. **Het adres:** `url` is waar de server luistert, eindigend op `/v1`; het endpoint komt erachter.
    2. **Het request:** `urlopen` neemt ook een adres, en maakt daar dan een GET van.
    3. **Het antwoord:** `{"object": "list", "data": [{"id": "qwen3-0.6b", "object": "model", "owned_by": "everything-ai"}, {"id": "granite-embedding-107m-multilingual", "object": "model", "owned_by": "everything-ai"}]}`, als bytes, waar `json.loads` een dictionary van maakt.
    4. **De waarde:** de lijst onder `"data"` heeft twee modellen; het eerste `"id"` is `'qwen3-0.6b'`, het chat model, de naam die je in de body van een request zet. Het andere maakt embeddings van teksten, zoals een latere les laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_body():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De body van een request

        Vul `body` in: het request voor `question`, aan het model `"qwen3-0.6b"`, met hoogstens 50 tokens. De check probeert ook een andere vraag.
        """),
        L.task("Geef `body` de keys `\"model\"`, `\"messages\"` en `\"max_tokens\"`, met `question` als bericht van de gebruiker."),
    ])
    return


@app.cell(hide_code=True)
def exercise_body():
    box_body = L.code_box("body", lines=3)
    box_body
    return (box_body,)


@app.cell(hide_code=True)
def check_body(box_body):
    result_body = L.check("body", box_body.value)
    result_body
    return (result_body,)


@app.cell(hide_code=True)
def hints_body():
    L.hints(
        "body",
        [
            "Kijk naar de body in het plaatje van de aanroep van de course: die heeft drie keys.",
            "`\"messages\"` is een lijst met één dictionary erin: het bericht, met de keys `\"role\"` en `\"content\"`.",
            "`body = {\"model\": \"qwen3-0.6b\", \"messages\": [___], \"max_tokens\": 50}`, met het bericht in het gat.",
        ],
        why="`body = {\"model\": \"qwen3-0.6b\", \"messages\": [{\"role\": \"user\", \"content\": question}], \"max_tokens\": 50}`: het model, het gesprek als lijst berichten, en het grootste aantal tokens dat het antwoord mag hebben.",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Verstuur het, en lees het antwoord

        `ask(url, question)` maakt de body al. Stuur hem naar `url + "/chat/completions"` en geef de tekst van het antwoord terug, zoals in het recept. De check roept je `ask` aan met een stand-in van de server, die dezelfde API heeft en meteen antwoordt; het plaatje onder de check laat zien wat je functie stuurde, en wat er terugkwam.
        """),
        L.task("Maak `ask` af: verstuur de body als POST, en geef alleen de tekst van het antwoord terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=10)
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
            "Stap 2 tot en met 5 van het recept: bytes, een request, versturen en lezen, de tekst.",
            "`data = json.dumps(body).encode()`, en dan `request = urllib.request.Request(url + \"/chat/completions\", data=data, headers={\"Content-Type\": \"application/json\"})`.",
            "Binnen het `with`-blok dat `request` verstuurt: `answer = json.loads(___)`.",
        ],
        why="""```python
def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

Met `data=` is het request een POST; de server stuurt JSON terug, en de tekst staat in de eerste choice, in zijn message.""",
    )
    return


@app.cell(hide_code=True)
def text_finish():
    mo.md(r"""
    ### Oefening 3. Waarom stopte het?
    """)
    return


@app.cell(hide_code=True)
def choice_finish_cell():
    choice_finish = L.choice(
        "finish",
        question="Je vraagt om een verhaal met `\"max_tokens\": 50`. Het antwoord eindigt midden in een zin, en zijn `finish_reason` is `\"length\"`. Wat is er gebeurd?",
        options={
            "done": "Het model schreef een lang antwoord, en maakte het af",
            "error": "De server stopte met een fout",
            "limit": "Het model kwam bij 50 tokens, en de server stopte het daar",
        },
        correct="limit",
        why={
            "limit": "`\"length\"` betekent dat het antwoord `max_tokens` bereikte; `\"stop\"` zou betekenen dat het model het zelf beëindigde. Vraag om meer tokens, of om een korter verhaal.",
            "done": "Dan zou er `\"stop\"` staan: het model beëindigde het antwoord zelf. Waar wijst `\"length\"` op?",
            "error": "Een fout komt terug als een foutstatus, zonder antwoord. Hier is er een antwoord, afgebroken. Welk getal in de body begrenst het?",
        },
    )
    choice_finish
    return (choice_finish,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag het Qwen3 met je eigen ask

    Zodra oefening 2 goed is, kan je `ask` met de echte server praten: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter, en je functie stuurt hem je vraag. Let op wat je ziet terwijl je wacht.
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
    question_box = mo.ui.text(value="Noem drie kleuren van de regenboog.", label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Verstuur hem met je ask")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_ask, question_box, send_button):
    _ask = C.learner_ask(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _ask is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo stuurt je vraag met je eigen `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om je vraag te versturen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `ask` heeft de vraag verstuurd; de server laadt Qwen3, dat het hele antwoord schrijft voordat er iets terugkomt …*"))
        _words = {"title": "Wat er terugkwam bij je ask",
                  "done": "{seconds} seconden, via {url}. Er was niets te zien tot het hele antwoord er was: de les over streaming verandert dat.",
                  "failed": "Je ask stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De `usage` in elk antwoord telt de tokens van [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl), en de berichten zijn het gesprek van [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl), in JSON zoals in [Data als tekst: JSON](/course/open/py-json?lang=nl). In de volgende les, [Project: je assistent antwoordt](/course/open/build-project-call?lang=nl), wordt je `ask` de client van je eigen assistent, die een heel gesprek verstuurt. Latere lessen laten het antwoord binnenkomen terwijl het geschreven wordt, en gaan om met een server die nee zegt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_finish, result_ask, result_body):
    L.wrap_up(result_body, result_ask, choice_finish)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
