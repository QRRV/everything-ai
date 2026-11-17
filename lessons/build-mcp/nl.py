# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-mcp. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-mcp", "nl")
    C = L.checks   # de MCP server van de course met convert, de runs van Qwen3 met zes servers, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_runs():
    quick_runs = L.choice(
        "quick-runs",
        question="Je programma stuurt Qwen3 de tool `convert`, en het model roept hem aan. Waar draait `convert`?",
        options={
            "provider": "Bij de aanbieder van het model, die elke tool uitvoert die hij in een request krijgt",
            "program": "In je programma: het model schrijft alleen de call, en je code voert hem uit",
            "model": "In het model zelf, dat de uitkomst uitrekent terwijl het zijn antwoord schrijft",
        },
        correct="program",
        why={
            "program": "Dat is een client-side tool: het model schrijft een naam en arguments, en je code doet het werk en stuurt het resultaat terug.",
            "provider": "Alleen een hosted tool draait bij de aanbieder. `convert` is een client-side tool. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
            "model": "Het model schrijft alleen de call; de uitkomst komt van code. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
        },
    )
    quick_runs
    return (quick_runs,)


@app.cell(hide_code=True)
def quick_check_description():
    quick_description = L.choice(
        "quick-description",
        question="Waar leert een model welke waarden een argument van een tool kan hebben, zoals de eenheden van `convert`?",
        options={
            "code": "Uit de code van de tool, die de API met de tool meestuurt",
            "first": "Uit de eerste fout die de tool geeft, na een call met een waarde die hij niet kan gebruiken",
            "description": "Uit de beschrijving van de tool: een enum en een beschrijving van elk argument",
        },
        correct="description",
        why={
            "description": "Een model kent een tool alleen van zijn beschrijving, dus daar horen de eenheden.",
            "code": "De code van een tool komt nooit bij het model. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
            "first": "Een goede fout helpt, maar pas na een verkeerde call: de beschrijving komt eerst. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
        },
    )
    quick_description
    return (quick_description,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl) zat de tool `convert` in je programma: zijn beschrijving in het request, zijn code naast de loop die hem uitvoerde. Wil je dezelfde tool aan een andere assistent geven, zoals een coding agent of een chat-app, dan schrijf je hem voor elk opnieuw. Het Model Context Protocol, kortweg MCP, is één vaste manier waarop een assistent de tools van een programma vindt en aanroept. Je schrijft de tool één keer, als MCP server, en elke assistent die MCP spreekt, kan hem gebruiken. Anthropic publiceerde MCP in november 2024; Claude, ChatGPT, Cursor en VS Code zijn een paar van de assistenten die het nu spreken.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **MCP** | het Model Context Protocol: vaste berichten waarmee een assistent een programma om zijn tools vraagt en ze aanroept |
    | **MCP server** | een programma dat tools aanbiedt via MCP, zoals `convert` |
    | **MCP client** | het deel van een assistent dat met een server praat: het haalt de tools op, geeft ze aan het model, en stuurt de calls van het model door |
    | **JSON-RPC** | een manier om requests en antwoorden als JSON te sturen: een request heeft een `method`, `params` en een `id`; het antwoord heeft hetzelfde `id`, en een `result` of een `error` |
    | **notification** | een JSON-RPC-bericht zonder `id`, dat geen antwoord krijgt |
    | **stdio** | de standaardinvoer en -uitvoer van een programma: de client start de server als programma, schrijft elk bericht als een regel naar zijn invoer, en leest elk antwoord als een regel uit zijn uitvoer |
    | **`_meta`** | in MCP-versie 2026-07-28 het deel van de params van elk request dat zegt welke versie van MCP de client spreekt, en wat hij kan |

    Een MCP server antwoordt op een paar methodes. `tools/list` geeft elke tool: zijn `name`, zijn `description` en zijn `inputSchema`, het JSON Schema van zijn arguments. `tools/call` voert een tool uit, en het result daarvan bevat `content`: een lijst met delen, elk met een type, zoals `{"type": "text", "text": "22.99 mi"}`, en `isError` als de tool draaide maar misging. `server/discover` zegt welke versies van MCP de server spreekt. De versie in deze les, 2026-07-28, is de nieuwste: elk request heeft in zijn `_meta` alles wat de server nodig heeft, dus de server onthoudt niets tussen requests.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De server van de course geeft `convert` met dezelfde naam, beschrijving en arguments als in [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl): de client maakt van `inputSchema` de `parameters`, en de tool die het model krijgt is {same}. MCP verandert dus hoe een tool bij het model komt, niet wat het model leest. Wat wel verandert, is wat een client het model geeft als hij meer servers heeft. De course stelde Qwen3-0.6B de acht vragen van [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl) opnieuw, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt, nu met de tools van {s} servers: een keuken, een agenda, de server met eenheden, notities, een winkel en het weer, {t} tools in totaal.
    """.format(same="precies dezelfde, teken voor teken" if F["same_tool"] else "niet dezelfde", s=F["servers"], t=F["tools"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een MCP server is een programma dat je draait, en zijn tools werken met jouw rechten: een server die je bestanden kan lezen, kan ze ook doorsturen. Elke beschrijving van een tool komt bovendien in de context van het model, en een beschrijving kan instructies bevatten die het model opvolgt, een prompt injection. Voeg servers toe die je vertrouwt, en alleen de tools die je nodig hebt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een tool als MCP server

    1. **Zet elke tool in een lijst** met zijn naam, zijn beschrijving en zijn `inputSchema`, zoals je hem voor een model zou beschrijven.
    2. **Antwoord op `tools/list`** met de tools, en zeg hoelang een client de lijst mag bewaren: `ttlMs` en `cacheScope`.
    3. **Antwoord op `tools/call`** door de tool uit te voeren, en geef zijn tekst terug als `content`; zet `isError` als de tool draaide en misging.
    4. **Antwoord met een fout op wat je niet kunt volgen:** een tool die je niet hebt, een versie van MCP die je niet spreekt, een methode die je niet kent.
    5. **Maak in de client** van elke tool uit de lijst de vorm van de chat-API, stuur elke call van het model als `tools/call`, en geef de tekst terug aan het model als resultaat van de tool.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture("nl"))
    return


@app.cell(hide_code=True)
def text_server():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De antwoorden van de server

        `handle(message)` is het hart van de server: hij krijgt één JSON-RPC-bericht, als dictionary, en geeft het antwoord terug, of `None` bij een notification. De controles van de versie en van `clientCapabilities` staan er al, net als `server/discover` en de hulpfuncties `result` en `error`. Vul de twee methodes in. Geef bij `tools/list` de lijst `TOOLS` terug, met `"ttlMs": 300000` (vijf minuten) en `"cacheScope": "public"`. Bij `tools/call` staan in de params de `name` van de tool en zijn `arguments`, die een tool zonder arguments mag weglaten; `FUNCTIONS` koppelt elke naam aan zijn functie, die een tekst teruggeeft. Een naam die niet in `FUNCTIONS` staat, krijgt de fout `-32602`; een tekst die begint met `Error` krijgt `"isError": True`. Het vak kent `TOOLS` en `FUNCTIONS`; om je code heen zet de course de tool en de loop die de regels leest en schrijft. De check stuurt het request voor de lijst uit het plaatje en nog acht berichten.
        """),
        L.task("Vervang elke `...` door de regel of regels die op die methode antwoorden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_server():
    box_server = L.code_box("server", lines=34)
    box_server
    return (box_server,)


@app.cell(hide_code=True)
def check_server(box_server):
    result_server = L.check("server", box_server.value)
    result_server
    return (result_server,)


@app.cell(hide_code=True)
def hints_server():
    L.hints(
        "server",
        [
            "Beide methodes eindigen met `return result(message, {...})`, met tussen de accolades wat in het result hoort. `result` zet zelf `resultType` en de `_meta` van de server erbij.",
            "Bij `tools/call`: neem `name = params.get(\"name\")`; staat die niet in `FUNCTIONS`, dan `return error(message, -32602, \"Unknown tool: \" + str(name))`; anders voer je `FUNCTIONS[name](params.get(\"arguments\") or {})` uit. `text.startswith(\"Error\")` is `True` als de tekst begint met `Error`.",
            "Na het uitvoeren van de tool: `return result(message, {\"content\": ___, \"isError\": text.startswith(\"Error\")})`, met in het gat een lijst met één tekstdeel.",
        ],
        why="""```python
    if message["method"] == "tools/list":
        return result(message, {"tools": TOOLS, "ttlMs": 300000, "cacheScope": "public"})
    if message["method"] == "tools/call":
        name = params.get("name")
        if name not in FUNCTIONS:
            return error(message, -32602, "Unknown tool: " + str(name))
        text = FUNCTIONS[name](params.get("arguments") or {})
        content = [{"type": "text", "text": text}]
        return result(message, {"content": content, "isError": text.startswith("Error")})
```

Een tool die draaide en misging, is een result met `isError`: het model leest de tekst en kan het opnieuw proberen. Een tool die er niet is, is een fout in het request, en die krijgt de client, niet het model.""",
    )
    return


@app.cell(hide_code=True)
def text_call():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De call van de client

        `call_tool(send, number, name, arguments)` is de helft van de client. Maak het JSON-RPC-request voor `tools/call`: `"jsonrpc": "2.0"`, `"id": number`, `"method": "tools/call"`, en `params` met de `name` van de tool, zijn `arguments` en `"_meta": META`. `send(request)` schrijft het naar de server en geeft het antwoord terug. Zit er in het antwoord een `error`, geef dan `"Error: "` en de `message` van de fout terug, zodat het model kan lezen wat er misging; voeg anders de `text` samen van elk deel van de `content` van het result met type `"text"`, met een nieuwe regel ertussen. De check roept een server met `convert` aan, en een server waarvan de tool een afbeelding en twee teksten teruggeeft.
        """),
        L.task("Vervang `request = ...` door het request, en de laatste `...` door de regels die van het antwoord een tekst maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_call():
    box_call = L.code_box("call", lines=14)
    box_call
    return (box_call,)


@app.cell(hide_code=True)
def check_call(box_call):
    result_call = L.check("call", box_call.value)
    result_call
    return (result_call,)


@app.cell(hide_code=True)
def hints_call():
    L.hints(
        "call",
        [
            "Het request is een dictionary met vier keys; de naam en de arguments van de tool gaan in `params`, naast `_meta`. Het antwoord is ook een dictionary: kijk of het de key `\"error\"` heeft.",
            "`if \"error\" in answer:` en dan `return \"Error: \" + answer[\"error\"][\"message\"]`. De delen staan in `answer[\"result\"][\"content\"]`; houd `part[\"text\"]` over van elk deel waarvan `part[\"type\"]` gelijk is aan `\"text\"`.",
            "`request = {\"jsonrpc\": \"2.0\", \"id\": number, \"method\": \"tools/call\", \"params\": ___}`, met in het gat een eigen dictionary.",
        ],
        why="""```python
def call_tool(send, number, name, arguments):
    request = {"jsonrpc": "2.0", "id": number, "method": "tools/call",
               "params": {"name": name, "arguments": arguments, "_meta": META}}
    answer = send(request)
    if "error" in answer:
        return "Error: " + answer["error"]["message"]
    texts = [part["text"] for part in answer["result"]["content"] if part["type"] == "text"]
    return "\\n".join(texts)
```

Het model leest hier alleen tekst; voor een deel met een afbeelding heb je een model nodig dat kan zien, zoals in [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl).""",
    )
    return


@app.cell(hide_code=True)
def text_trust():
    mo.md(r"""
    ### Oefening 3. Een server van ergens anders
    """)
    return


@app.cell(hide_code=True)
def choice_trust_cell():
    choice_trust = L.choice(
        "trust-server",
        question="Je vindt online een MCP server met een tool die je e-mail leest. Je wilt dat je agent hem gebruikt. Wat controleer je eerst?",
        options={
            "protocol": "Of hij versie 2026-07-28 van MCP spreekt, zodat het protocol hem vanzelf veilig houdt",
            "who": "Wie hem maakte, wat elke tool kan, en wat zijn beschrijvingen tegen het model zeggen",
            "nothing": "Nog niets: de agent ziet aan de beschrijving wel of een tool schadelijk is",
        },
        correct="who",
        why={
            "who": "De server draait als programma met jouw rechten, en zijn beschrijvingen komen in de context van het model. Lees ze, en houd alleen de tools die je nodig hebt.",
            "protocol": "Het protocol legt de berichten vast, niet wat een server met je e-mail doet: een server die het spreekt, kan hem toch doorsturen.",
            "nothing": "In een beschrijving kan alles staan, en een model volgt vaak op wat het leest: een beschrijving met verborgen instructies is een prompt injection.",
        },
    )
    choice_trust
    return (choice_trust,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zijn oefening 1 en 2 goed, dan draait je server als eigen programma: de course bewaart hem als bestand, `units_server.py` in je werkmap, start hem, en praat met hem over stdio met je `call_tool`. Qwen3-0.6B, op de server van de course op je computer, krijgt de tools die je server geeft. Stel een vraag over eenheden; onder het antwoord kun je de regels openen die van en naar je server gingen.
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
def ask_box_cell():
    ask_box = mo.ui.text(value=C.QUESTIONS[0][0], label="Je vraag", full_width=True)
    ask_button = mo.ui.run_button(label="Vraag")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_call, box_server):
    _server = C.learner_fn("server", box_server.value)
    _call = C.learner_fn("call", box_call.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _server is None or _call is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo draait je eigen server, en roept hem aan met je eigen `call_tool`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je server start, en de server van de course laadt Qwen3 …*"))
        _words = {"title": "Qwen3 met de tool van je server", "no_call": "Het model riep geen tool aan: het antwoordde zelf.",
                  "client": "client → server", "server": "server → client", "lines": "De regels over stdio",
                  "saved": "Je server is bewaard als {path}.", "stopped": "Je server stopte: {e}", "failed": "Het request aan Qwen3 stopte: {e}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.ask_server(_server, _call, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tool is nog steeds de `convert` van [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl), met de beschrijving uit [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl); MCP legt alleen vast hoe een client hem vindt en aanroept, in JSON-RPC-berichten over stdio. Je server is een eigen programma, dat elke assistent die MCP spreekt, kan starten. De volgende les, [AGENTS.md, skills en andere uitbreidingen](/course/open/build-agent-extensions?lang=nl), geeft een agent meer dan tools: notities over een project, en instructies die hij pas laadt als hij ze nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_trust, result_call, result_server):
    L.wrap_up(result_server, result_call, choice_trust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
