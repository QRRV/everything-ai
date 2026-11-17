# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-frameworks. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-frameworks", "nl")
    C = L.checks   # de tool convert op drie manieren, de runs van Qwen3 van de course, de graaf van de agent loop, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl), en een over [Wat een agent onthoudt](/course/open/build-agent-memory?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mcp():
    quick_mcp = L.choice(
        "quick-mcp",
        question="Een tool komt van een MCP server in plaats van uit je eigen code. Wat verandert er voor het model?",
        options={
            "protocol": "Het model praat zelf JSON-RPC met de server, dus het heeft geen tool definitions nodig",
            "nothing": "Niets: je code vraagt de server om zijn tools en zet ze in het request zoals eerst",
            "faster": "De tool wordt aangeroepen zonder request aan het model, omdat de server beslist wanneer hij nodig is",
        },
        correct="nothing",
        why={
            "nothing": "MCP legt vast hoe je programma de tools vindt en aanroept; het model ziet dezelfde tool definitions en schrijft dezelfde calls.",
            "protocol": "Je code is de MCP client; het model ziet alleen de tools in het request. Zie [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl).",
            "faster": "Het model beslist nog steeds wanneer het een tool aanroept; de server voert de call alleen uit. Zie [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl).",
        },
    )
    quick_mcp
    return (quick_mcp,)


@app.cell(hide_code=True)
def quick_check_memory():
    quick_memory = L.choice(
        "quick-memory",
        question="Hoe weet een nieuw gesprek wat de agent vorige week met `remember` bewaarde?",
        options={
            "model": "Het model heeft het geleerd: elke aanroep van `remember` traint het model een beetje",
            "server": "De server bewaart elk eerder gesprek en stuurt het uit zichzelf mee",
            "notes": "Je code leest de notities uit het bestand en zet ze in de system prompt",
        },
        correct="notes",
        why={
            "notes": "Het model onthoudt niets tussen requests; de notities worden buiten het model bewaard, en je code geeft ze terug.",
            "model": "Niets traint het model; `remember` schrijft een notitie in een bestand. Zie [Wat een agent onthoudt](/course/open/build-agent-memory?lang=nl).",
            "server": "De server beantwoordt elk request op zich en bewaart niets. Zie [Wat een agent onthoudt](/course/open/build-agent-memory?lang=nl).",
        },
    )
    quick_memory
    return (quick_memory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In deze module schreef je de agent loop, tool definitions, een MCP client, een geheugen, een trace en grenzen zelf. Een agent framework geeft je dat allemaal kant-en-klaar: LangGraph, LlamaIndex, AutoGen, PydanticAI, en de agent-SDK's van de aanbieders van modellen. Dat scheelt werk, maar een framework doet ook dingen die je niet ziet, zoals eigen instructies toevoegen aan elk request. Als je weet welk van je eigen onderdelen onder elke naam zit, kun je een framework kiezen, of geen, en uitzoeken wat het doet als je agent zich anders gedraagt dan je verwachtte.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **agent framework** | een library die de onderdelen van agents kant-en-klaar geeft, zoals de agent loop, tool definitions, geheugen en traces |
    | **docstring** | de tekst tussen drie aanhalingstekens direct onder de `def`-regel van een functie, die zegt wat de functie doet |
    | **graaf** | nodes die door edges verbonden zijn; hier is een node een stap die de state verandert, en de edge erna zegt welke node volgt |
    | **state** | alles wat de stappen delen, zoals de berichten tot nu toe; elke node krijgt hem en geeft een nieuwe terug |
    | **`while ...:`** | voert de regels eronder steeds opnieuw uit, zolang klopt wat er na `while` staat |
    | **`a and b`** | `True` alleen als `a` en `b` allebei `True` zijn |

    De onderdelen van een framework hebben eigen namen, maar onder elke naam zit iets wat je bouwde:

    | In een framework | Wat je bouwde |
    |---|---|
    | de agent, zijn run | de loop uit [De agent loop](/course/open/build-agent-loop?lang=nl): sturen, de calls uitvoeren, de resultaten toevoegen, tot er geen calls meer zijn of de stapgrens bereikt is |
    | een tool gemaakt van een functie | de tool definition uit [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl), met de hand geschreven |
    | MCP-tools | je client uit [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl) |
    | geheugen | de notities uit [Wat een agent onthoudt](/course/open/build-agent-memory?lang=nl) |
    | tracing | de trace uit [De agent loop](/course/open/build-agent-loop?lang=nl) |
    | guardrails, goedkeuringen | de grenzen uit [Wat een tool mag](/course/open/build-tool-permissions?lang=nl) |

    Sommige frameworks schrijven de loop als een graaf. LangGraph is het bekendst: je geeft de nodes, de edges en een begin, en het framework draait ze. De agent loop is dan twee nodes, `model` en `tools`, en de edge na `model` kijkt naar het laatste antwoord:
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

    Een framework maakt een tool van een functie: van zijn naam, zijn docstring en de types van zijn parameters. Het kan ook instructies vóór elk request zetten zonder ze te laten zien. De course stelde Qwen3-0.6B de {n} vragen uit [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl) op drie manieren, bij temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt, met hoogstens twee requests per vraag: met `convert` met de hand geschreven en geen system prompt, zoals in die les; met dezelfde tool en een system prompt van het soort dat een framework uit zichzelf toevoegt, geschreven door de course; en met `convert` gemaakt van de functie, zoals je `tool_from` uit oefening 1 hem maakt.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("De verborgen instructies van een framework en de tool definitions die het maakt, gaan net zo naar het model als je eigen. Kijk voordat je op een framework bouwt naar de requests die het echt verstuurt: de meeste kunnen ze laten zien, in hun tracing of in een log."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een agent framework beoordelen

    1. **Noem de onderdelen die je nodig hebt:** de loop, tools, MCP, geheugen, traces, grenzen.
    2. **Zoek elk onderdeel onder de namen van het framework,** en wat het anders doet dan je eigen.
    3. **Kijk naar de requests die het verstuurt:** de system prompt die het toevoegt, en de tool definitions die het van je functies maakt.
    4. **Draai dezelfde taken met en zonder,** met hetzelfde model, en vergelijk de antwoorden en de tokens.
    5. **Houd de grenzen in je eigen code,** in de tools en in de stapgrens, welk framework de loop ook draait.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(*C.worked_run(), "nl"))
    return


@app.cell(hide_code=True)
def text_tool():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een tool gemaakt van een functie

        `tool_from(name, doc, params)` maakt een tool definition van een functie, zoals een framework doet: `name` is de naam van de functie, `doc` zijn docstring, en `params` een dictionary van elke parameter met zijn type in een JSON schema, zoals `{"amount": "number", "from_unit": "string"}`. In de definition is elke parameter een eigen dictionary, met zijn type onder `"type"`. Frameworks verschillen in hoeveel van de docstring ze nemen; hier is de description de eerste regel ervan, zonder de spaties en nieuwe regels rond de docstring. De check probeert `convert` uit het plaatje, een tool met één parameter, een docstring met spaties eromheen en zonder parameters, en een tool met een aantal hele stuks, van het type `"integer"`.
        """),
        L.task("Vervang de twee `...`: de regel die elke parameter in `properties` zet, en de description."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tool():
    box_tool = L.code_box("tool-from", lines=9)
    box_tool
    return (box_tool,)


@app.cell(hide_code=True)
def check_tool(box_tool):
    result_tool = L.check("tool-from", box_tool.value)
    result_tool
    return (result_tool,)


@app.cell(hide_code=True)
def hints_tool():
    L.hints(
        "tool-from",
        [
            "Kijk hoe `convert` met de hand geschreven is in Tools als JSON: elke parameter onder `\"properties\"` is een kleine dictionary, en de description is één zin.",
            "Voor een parameter is die dictionary `{\"type\": kind}`, onder zijn naam in `properties`. Voor de description: `.strip()` haalt de spaties van de uiteinden van een tekst, en `.split(\"\\n\")` knipt hem in een lijst van zijn regels.",
            "In de definition, de eerste regel van de docstring: `\"description\": doc.___[0],`.",
        ],
        why="""```python
def tool_from(name, doc, params):
    properties = {}
    for param, kind in params.items():
        properties[param] = {"type": kind}
    return {"type": "function", "function": {
        "name": name,
        "description": doc.strip().split("\\n")[0],
        "parameters": {"type": "object", "properties": properties, "required": list(params)},
    }}
```

Een framework leest `params` uit de types die je achter elke parameter in de `def`-regel schrijft, en maakt van `float` `"number"` en van `str` `"string"`; de rest is hetzelfde.""",
    )
    return


@app.cell(hide_code=True)
def text_graph():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De agent loop als graaf

        `run_graph(nodes, edges, state, start, limit)` draait een graaf van stappen. `nodes` is een dictionary van de naam van elke node met een functie die de state krijgt en een nieuwe state teruggeeft. `edges` is een dictionary van de naam van elke node met een functie die de nieuwe state krijgt en de naam van de volgende node teruggeeft, of `"END"`. Begin bij `start`. Zolang de node niet `"END"` is en er minder dan `limit` stappen gedraaid hebben: draai de node, vraag zijn edge om de volgende node, en tel de stap. Geef de state terug. De check probeert de agent loop met een model dat `convert` één keer aanroept, een dat meteen antwoordt, een dat elke keer een tool aanroept, en een kleine graaf van getallen die 1 optelt en verdubbelt, met een grens van 10 die hij niet haalt en met een grens van 3.
        """),
        L.task("Vervang `...` door een `while`-loop van drie regels, één voor elke stap hierboven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_graph():
    box_graph = L.code_box("graph", lines=8)
    box_graph
    return (box_graph,)


@app.cell(hide_code=True)
def check_graph(box_graph):
    result_graph = L.check("graph", box_graph.value)
    result_graph
    return (result_graph,)


@app.cell(hide_code=True)
def hints_graph():
    L.hints(
        "graph",
        [
            "Twee dingen stoppen de loop: de node `\"END\"`, en de grens. Zet ze allebei in de voorwaarde van de `while`, verbonden met `and`.",
            "In elke ronde geeft de functie van de node de nieuwe state: `nodes[node](state)`. Daarna geeft de edge van die node de naam van de volgende: `edges[node](state)`, met de nieuwe state.",
            "Begin met `while node != \"END\" and ___:`, en zet de drie regels eronder.",
        ],
        why="""```python
def run_graph(nodes, edges, state, start, limit):
    node = start
    steps = 0
    while node != "END" and steps < limit:
        state = nodes[node](state)
        node = edges[node](state)
        steps = steps + 1
    return state
```

Met de nodes `model` en `tools` zijn deze regels de agent loop uit het plaatje; met andere nodes draaien ze elke workflow van stappen.""",
    )
    return


@app.cell(hide_code=True)
def text_look():
    mo.md(r"""
    ### Oefening 3. Waar je eerst kijkt
    """)
    return


@app.cell(hide_code=True)
def choice_look_cell():
    choice_look = L.choice(
        "look",
        question="Met een framework roept je agent zijn tool minder vaak aan dan met je eigen loop, met hetzelfde model en dezelfde functie. Waar kijk je eerst?",
        options={
            "temperature": "Naar de temperature, omdat een framework een hogere stuurt zodat de antwoorden meer verschillen",
            "requests": "Naar de requests die het framework verstuurt: de system prompt die het toevoegt, en de tool definition",
            "tool": "Naar de code van de tool, omdat een framework een tool op een andere manier uitvoert dan je eigen loop, en hem kan breken",
        },
        correct="requests",
        why={
            "requests": "Het model ziet alleen de requests; wat het framework eraan toevoegt, of anders schrijft, is wat er veranderde.",
            "temperature": "De temperature kan verschillen, maar kijk eerst naar wat het framework aan elk request toevoegt, zoals een system prompt.",
            "tool": "Of de tool goed draait doet er pas toe als hij aangeroepen wordt; dat beslist het model, op grond van wat er in het request staat.",
        },
    )
    choice_look
    return (choice_look,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Stel een vraag over kilometers, mijlen, kilo's of ponden, en je `run_graph` draait de agent loop uit het plaatje op Qwen3-0.6B, met `convert` gemaakt door je `tool_from`, hoogstens {k} stappen. De tabel laat elke node zien waar je graaf langs ging, zoals de trace van een framework hem laat zien. De course start een eigen server op je computer, met het model erachter.
    """.format(k=C.DEMO_LIMIT))
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Qwen3-0.6B downloaden (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen plek op je schijf voor 1,5 GB. Maak plek, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def question_cell():
    question_box = mo.ui.text(value=C.QUESTIONS[4][0], label="Vraag", full_width=True)
    run_button = mo.ui.run_button(label="Draai de graaf")
    mo.vstack([question_box, run_button])
    return question_box, run_button


@app.cell(hide_code=True)
def demo_run(box_graph, box_tool, question_box, run_button):
    _graph = C.learner_fn("graph", box_graph.value)
    _tool = C.learner_fn("tool-from", box_tool.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _graph is None or _tool is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo draait je eigen graaf, met de tool die je eigen functie maakt.*")
    elif not run_button.value:
        _out = mo.md("*Druk op de knop om de graaf te draaien.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je graaf is gestart; de server laadt Qwen3 …*"))
        _words = {"title": "Jouw run_graph op de agent loop, node voor node", "lang": "nl",
                  "head": ["stap", "node", "tokens aan prompt", "wat hij gaf"],
                  "stopped": "Het stopte met {e}.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.run_demo(_graph, _tool, question_box.value.strip(), _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elk onderdeel van een framework is er een die je in deze module bouwde: de loop uit [De agent loop](/course/open/build-agent-loop?lang=nl), de tools uit [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl), en de notities uit [Wat een agent onthoudt](/course/open/build-agent-memory?lang=nl). Daarmee is de module agents af. De volgende module gaat over wat er na het bouwen komt: uitzoeken waar een LLM-toepassing misgaat, het meten, en het draaiend houden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_look, result_graph, result_tool):
    L.wrap_up(result_tool, result_graph, choice_look)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
