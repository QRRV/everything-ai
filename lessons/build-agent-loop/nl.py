# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-loop. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-loop", "nl")
    C = L.checks   # de agent van de keuken met en zonder een gedachte voor elke call, de runs van Qwen3 door de course, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_orchestrator():
    quick_orchestrator = L.choice(
        "quick-orchestrator",
        question="Wat maakt een orchestrator met workers anders dan delen naast elkaar?",
        options={
            "decides": "Het model bepaalt welke delen er zijn en hoeveel, zodra het de invoer heeft gelezen",
            "faster": "Zijn workers draaien altijd na elkaar, zodat elk het resultaat van de vorige kan lezen",
            "model": "Hij heeft voor elke worker een groot model nodig, terwijl delen naast elkaar een klein model kunnen gebruiken",
        },
        correct="decides",
        why={
            "decides": "Bij delen naast elkaar kent je code de delen vooraf; bij een orchestrator kan een menu twee gerechten hebben of vijf.",
            "faster": "Ook de workers van een orchestrator hebben elkaar niet nodig, dus ze kunnen tegelijk draaien. Zie [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl).",
            "model": "Beide patronen werken met elk model; in de oefening was de worker gewone code. Zie [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl).",
        },
    )
    quick_orchestrator
    return (quick_orchestrator,)


@app.cell(hide_code=True)
def quick_check_evaluator():
    quick_evaluator = L.choice(
        "quick-evaluator",
        question="In `improve` vindt de controle de tip te lang. Wat leest het model in de volgende ronde?",
        options={
            "task": "Alleen de opdracht opnieuw, zodat het fris begint en zijn fout niet herhaalt",
            "problem": "Alleen wat er mis was, als nieuw bericht van de gebruiker, zonder de tip die het schreef",
            "both": "De opdracht, zijn eigen tip met de rol `assistant`, en wat daar mis mee was",
        },
        correct="both",
        why={
            "both": "Dat stuurt `improve` terug: de tip die het model schreef, met de rol `assistant`, en waarom die terugging, zodat het model weet wat het moet veranderen.",
            "task": "Met temperature 0 geven dezelfde berichten weer dezelfde tip. Zie [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl).",
            "problem": "Zonder zijn eigen tip weet het model niet wat er te lang was. Zie [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl).",
        },
    )
    quick_evaluator
    return (quick_evaluator,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vijf patronen bepaalde je code de paden. Sommige taken hebben geen vast pad: wat je daarna doet, hangt af van wat de vorige stap vond. Dan moet het model kiezen, en heb je een agent nodig. In [Wat een agent is](/course/open/agent-idea?lang=nl) koos Qwen3 slecht: het zocht het recept op, en antwoordde zonder in de voorraadkast te kijken. Een bekende manier om een model beter te laten kiezen, is het voor elke tool call te laten opschrijven wat het weet en wat het daarna nodig heeft. In deze les bouw je de hele loop van een agent via de API, voeg je dat toe, en meet je of het helpt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **ReAct** | *reason and act*: een agent waarin het model afwisselt tussen een gedachte, een actie (een tool call) en een observation, tot het kan antwoorden; van Yao en anderen (2022) |
    | **gedachte** | een regel tekst die het model bij zijn tool call schrijft: wat het tot nu toe weet, en wat het daarna nodig heeft |
    | **observation** | wat de agent na een actie te zien krijgt; hier het resultaat van de tool, dat de loop terugstuurt met de rol `tool` |

    De loop is die van [Wat een agent is](/course/open/agent-idea?lang=nl), nu via de API van [Tools als JSON](/course/open/build-tool-calling?lang=nl). Elk request bevat het hele gesprek tot nu toe, met de tools. Heeft het antwoord van het model tool calls, dan zet je code het antwoord erbij zoals het kwam, voert elke call uit, en zet elk resultaat erbij met het id van zijn call; dan vraagt hij opnieuw. Een antwoord zonder tool calls is het antwoord van de agent. Na `max_steps` requests stopt de loop hoe dan ook.

    ReAct verandert alleen de system prompt. De prompt van de course is die van de vorige lessen, met één zin erbij:

    > {react}

    De gedachte komt dan terug als de tekst van een antwoord met tool calls, en de loop stuurt hem in het volgende request mee, zodat het model zijn eigen redenering leest.
    """.format(react=C.REACT))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course stelde Qwen3-0.6B de {n} vragen van [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl) als agent, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt, en met een limiet van {steps} requests: één keer met de system prompt van die les, één keer met de zin hierboven erbij.
    """.format(n=F["questions"], steps=F["steps"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een gedachte is tekst die het model schrijft, geen bewijs dat het goed redeneert. Het kan *I have all the ingredients* schrijven en het mis hebben, en elke gedachte kost tokens in elk volgend request. Lees de traces van je eigen runs, en meet op je eigen vragen of de gedachten helpen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een agent loop via de API

    1. **Begin de berichten** met de system prompt en de opdracht.
    2. **Vraag het model,** met de tools in het request.
    3. **Geen tool calls in het antwoord?** Dan is het het antwoord: geef zijn content terug.
    4. **Zet anders het antwoord erbij zoals het kwam,** gedachte en calls samen, voer elke call uit, en zet elk resultaat erbij met de rol `tool` en het id van zijn call.
    5. **Stop na `max_steps` requests,** en geef `None` terug, zodat je code weet dat de agent niet klaar kwam.
    6. **Schrijf elke run op als trace,** en lees hem: de trace laat zien waar de agent de fout in ging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_run():
    mo.Html(C.run_picture(C.WORKED, "nl"))
    return


@app.cell(hide_code=True)
def text_run_agent():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De hele loop

        `run_agent(url, task, max_steps)` is de agent van deze les. `ask(url, messages)` stuurt de berichten met `TOOLS` naar de server, en geeft het bericht van het model terug als dictionary: zijn `"content"`, en, als het tools aanroept, zijn `"tool_calls"`, elk met een `"id"` en een `"function"` met een `"name"` en de `"arguments"` als JSON-tekst. `FUNCTIONS["pantry"]` is de tool zelf: hij krijgt de argumenten als dictionary en geeft het resultaat terug als tekst. Maak de loop af. Het vak kent `TOOLS`, `FUNCTIONS` en `REACT`. De check laat je loop draaien tegen een stand-in van de server die drie runs speelt: een op `pancakes` die vier tools tegelijk aanroept, een die nooit ophoudt met tools aanroepen, en een die meteen antwoordt. Het plaatje eronder laat je run op `pancakes` zien als trace.
        """),
        L.task("Zet in de loop het bericht van het model in `messages`; geef zijn content terug als het geen tool calls heeft; voer anders elke call uit en voeg het resultaat toe als bericht met de rol `tool`, het id van de call en het resultaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_agent():
    box_run_agent = L.code_box("run-agent", lines=26)
    box_run_agent
    return (box_run_agent,)


@app.cell(hide_code=True)
def check_run_agent(box_run_agent):
    result_run_agent = L.check("run-agent", box_run_agent.value)
    result_run_agent
    return (result_run_agent,)


@app.cell(hide_code=True)
def hints_run_agent():
    L.hints(
        "run-agent",
        [
            "Elke ronde heeft twee eindes. Een antwoord zonder tool calls beëindigt de run. Een antwoord met tool calls betekent: voer ze uit, en vraag opnieuw met hun resultaten in de berichten.",
            "`not` draait een test om: `not x` klopt als `x` `None` is of een lege lijst. Zet eerst `message` in `messages`, zoals het kwam. Lees voor elke call in `message[\"tool_calls\"]` de argumenten met `json.loads`, voer `FUNCTIONS[...]` uit met de naam van de call, en voeg een bericht toe met de rol `tool`.",
            "`messages = messages + [message]`; dan `if not message.get(\"tool_calls\"):` met `return message[\"content\"]`; dan `for call in message[\"tool_calls\"]:`, met `arguments = json.loads(call[\"function\"][\"arguments\"])`, het resultaat van `FUNCTIONS[call[\"function\"][\"name\"]](arguments)`, en `{\"role\": \"tool\", \"tool_call_id\": call[\"id\"], \"content\": result}` toegevoegd aan `messages`.",
        ],
        why="""```python
def run_agent(url, task, max_steps):
    messages = [{"role": "system", "content": REACT}, {"role": "user", "content": task}]
    for step in range(max_steps):
        message = ask(url, messages)
        messages = messages + [message]
        if not message.get("tool_calls"):
            return message["content"]
        for call in message["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = FUNCTIONS[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
    return None
```

Het bericht van het model gaat terug zoals het kwam, met zijn gedachte en zijn calls: de server controleert dat elke call zijn resultaat heeft, en het model leest in de volgende ronde zijn eigen redenering.""",
    )
    return


@app.cell(hide_code=True)
def text_trace():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De trace

        Een run leest makkelijker als regels dan als berichten. `trace(messages)` loopt door de berichten van een run en geeft een lijst regels terug: voor het bericht van een tool `"Observation: "` en zijn content; voor een antwoord met tool calls zijn gedachte zoals het model hem schreef, als het er een schreef, en dan voor elke call `"Action: "`, zijn naam en zijn argumenten tussen haakjes, zoals `Action: pantry({"item": "milk"})`; voor een antwoord zonder tool calls `"Answer: "` en zijn content. De system prompt en de opdracht krijgen geen regel. Maak de loop af. De check probeert een run met gedachten, een run zonder, en een antwoord zonder tool call.
        """),
        L.task("Voeg onder elke `if` de regels toe die de tekst hierboven voor dat soort bericht geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trace():
    box_trace = L.code_box("trace", lines=16)
    box_trace
    return (box_trace,)


@app.cell(hide_code=True)
def check_trace(box_trace):
    result_trace = L.check("trace", box_trace.value)
    result_trace
    return (result_trace,)


@app.cell(hide_code=True)
def hints_trace():
    L.hints(
        "trace",
        [
            "Het bericht van een tool is eenvoudig: één regel. Een bericht van de assistant is van twee soorten: met tool calls, of zonder. Onderscheid ze met `m.get(\"tool_calls\")`.",
            "Met tool calls: heeft `m.get(\"content\")` tekst, dan is dat de gedachte, een eigen regel; dan één regel per call. Zonder: één regel `\"Answer: \"`.",
            "`lines.append(\"Observation: \" + m[\"content\"])` voor een tool; voor een assistant `if m.get(\"tool_calls\"):`, dan `if m.get(\"content\"):` en een loop `for c in m[\"tool_calls\"]:` die `\"Action: \" + c[\"function\"][\"name\"] + \"(\" + c[\"function\"][\"arguments\"] + \")\"` toevoegt; onder `else:` de regel `\"Answer: \"`.",
        ],
        why="""```python
def trace(messages):
    lines = []
    for m in messages:
        if m["role"] == "tool":
            lines.append("Observation: " + m["content"])
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                if m.get("content"):
                    lines.append(m["content"])
                for c in m["tool_calls"]:
                    lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
            else:
                lines.append("Answer: " + m["content"])
    return lines
```

`if m.get("content"):` klopt niet voor `None`, en ook niet voor de lege tekst `""`: een model zonder gedachte kan allebei sturen.""",
    )
    return


@app.cell(hide_code=True)
def text_step_limit():
    mo.md(r"""
    ### Oefening 3. Bij de stappenlimiet
    """)
    return


@app.cell(hide_code=True)
def choice_step_limit_cell():
    choice_step_limit = L.choice(
        "step-limit",
        question="Je agent bereikt de stappenlimiet terwijl hij nog tools aanroept, en `run_agent` geeft `None` terug. Wat moet je app dan doen?",
        options={
            "raise": "De limiet verhogen naar duizend requests, zodat elke run uiteindelijk klaar komt",
            "say": "Zeggen dat de agent niet klaar kwam, en laten zien wat hij tot dan toe deed",
            "last": "De tekst van de laatste stap als antwoord laten zien, want het model schreef die ook",
        },
        correct="say",
        why={
            "say": "De gebruiker weet dan dat er geen antwoord is, en de trace laat zien waar de agent in rondjes ging.",
            "raise": "Een agent die in rondjes gaat, blijft dat doen, en kost tijd en tokens voor elk request. De limiet is er om hem te stoppen.",
            "last": "De laatste stap is een gedachte met een tool call, geen antwoord. Hem als antwoord laten zien verbergt dat de agent niet klaar kwam.",
        },
    )
    choice_step_limit
    return (choice_step_limit,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Is oefening 1 goed, dan kun je je agent op het echte model laten draaien: de course start een eigen server op je computer, met Qwen3-0.6B erachter, en je loop voert de tools van de keuken uit. Vraag in je eigen woorden naar een gerecht, en lees de trace.
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
    question_box = mo.ui.text(value=C.QUESTIONS[4][0], label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Laat je agent draaien")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_run_agent(box_run_agent, question_box, send_button):
    _run_agent = C.learner_run_agent(box_run_agent.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _run_agent is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo laat je eigen loop draaien.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te starten.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3, en je agent draait …*"))
        _words = {"title": "Je agent met Qwen3, als trace", "gave": "Je run_agent gaf terug: {v}",
                  "error": "Je run_agent stopte met {e}.", "none": "Je run_agent stuurde niets naar de server.",
                  "requests": "{k} requests",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_run_agent, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop van [Wat een agent is](/course/open/agent-idea?lang=nl) draait nu via de API van [Tools als JSON](/course/open/build-tool-calling?lang=nl), met de resultaten teruggestuurd bij de id's van hun calls. Waar de patronen van [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl) het pad in code vastlegden, laat deze loop het model het kiezen, stap voor stap, met een gedachte voor elke call en een limiet op de stappen. In de volgende les schrijft het model eerst zijn hele plan, voordat er een stap draait.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_step_limit, result_run_agent, result_trace):
    L.wrap_up(result_run_agent, result_trace, choice_step_limit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
