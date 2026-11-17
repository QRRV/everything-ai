# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tool-calling. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tool-calling", "nl")
    C = L.checks   # de tool convert, de runs van Qwen3 met en zonder die tool, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wat een agent is: een model, tools en een loop](/course/open/agent-idea?lang=nl), één over [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In de agent loop heeft het bericht van het model een tool call. Wat doet je programma nu?",
        options={
            "show": "Het laat de call aan de gebruiker zien als antwoord van het model, want het model schreef hem, en de loop stopt",
            "model": "Nog niets: het model voert de tool zelf uit en schrijft het resultaat in zijn volgende bericht",
            "run": "Het voert de tool uit, voegt het resultaat toe als bericht met de rol `tool`, en vraagt het model opnieuw",
        },
        correct="run",
        why={
            "run": "Het model schrijft alleen tekst; jouw programma voert de tool uit, en het model leest het resultaat bij zijn volgende beurt.",
            "show": "Een tool call is geen antwoord voor de gebruiker: hij vraagt je programma iets te doen. De loop stopt pas bij een bericht zonder tool call. Zie [Wat een agent is](/course/open/agent-idea?lang=nl).",
            "model": "Een model schrijft alleen tekst, ook een tool call: het kan niets uitvoeren. Zie [Wat een agent is](/course/open/agent-idea?lang=nl).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_schema():
    quick_schema = L.choice(
        "quick-schema",
        question="Welk antwoord past bij het schema `{\"type\": \"object\", \"properties\": {\"price\": {\"type\": \"number\"}}, \"required\": [\"price\"]}`?",
        options={
            "fits": "`{\"price\": 649}`",
            "text": "`{\"price\": \"649\"}`",
            "key": "`{\"cost\": 649}`",
        },
        correct="fits",
        why={
            "fits": "Het heeft de verplichte key `price`, en de waarde ervan is een getal.",
            "text": "`\"649\"` is een tekst, tussen aanhalingstekens, en het schema wil een getal. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
            "key": "Het schema vraagt de key `price`; `cost` is een andere key. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
        },
    )
    quick_schema
    return (quick_schema,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een language model schrijft het waarschijnlijkste volgende token; het rekent niet. Vraag het hoeveel mijl 37 km is, en het schrijft iets wat op een berekening lijkt. Voor een app is *lijkt op* niet genoeg. Een functie in je programma rekent precies om, elke keer. Laat het model je programma dus vragen om die functie uit te voeren: het model leest de vraag en kiest de getallen, je code doet het rekenwerk. In [Wat een agent is](/course/open/agent-idea?lang=nl) zette de `ask` van de course de tools voor je in de prompt, en haalde hij de calls uit het antwoord. Elke chat-API heeft daar zijn eigen manier voor, function calling. In deze les stuur je de tools zelf, en lees je de calls zoals de API ze geeft.
    """)
    return


@app.cell(hide_code=True)
def predict_own():
    guess_own = L.guess(
        "own",
        question="De course vroeg Qwen3-0.6B {n} omrekeningen, zoals *{q}*, zonder tools. Hoeveel antwoorden kwamen uit op het goede getal, tot op twee decimalen?".format(
            n=F["questions"], q=C.QUESTIONS[0][0]),
        options={
            "all": "Alle {n}: eenheden omrekenen is eenvoudig rekenwerk".format(n=F["questions"]),
            "some": "Een deel ervan, niet alle",
            "none": "Geen: zo'n klein model kan helemaal niet omrekenen",
        },
        reveal="**{r} van de {n}** kwamen uit op het goede getal. Het model kende de factoren, zoals 0,621371 mijl in een kilometer, maar het rekenwerk ging mis: bij 37 km schreef het 37 × 0,621371 ≈ 23,39, en 37 × 0,621371 is 22,99. Ook schreef het elke berekening uit: {t} tokens en {s} seconden per antwoord, op de computer van de course.".format(
            r=F["own_right"], n=F["questions"], t=round(F["own_tokens"]), s=round(F["own_seconds"])),
    )
    guess_own
    return (guess_own,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Met function calling gaan de tools in een eigen veld van het request, `"tools"`: voor elke tool zijn naam, een beschrijving, en een JSON-schema van zijn argumenten, zoals in [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl). De server zet ze in de prompt, in de vorm die zijn model geleerd heeft. Roept het model een tool aan, dan heeft het bericht van het antwoord geen tekst maar een lijst `"tool_calls"`, en is `"finish_reason"` `"tool_calls"`. Elke call heeft een `"id"`, de naam van de functie, en zijn argumenten als JSON-tekst. Je programma voert de functies uit, voegt elk resultaat toe als bericht met de rol `tool` en het `id` van zijn call, en vraagt opnieuw, met dezelfde tools. Nu leest het model de resultaten en antwoordt het.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **function calling** | de manier waarop een chat-API een model je functies laat aanroepen: je beschrijft de tools in het request, en het antwoord geeft de tool calls los van de tekst | `"tools": [convert_tool]` |
    | **tool choice** | een veld van het request dat zegt of het model een tool mag aanroepen (`"auto"`, de standaard), niet mag (`"none"`), er een moet aanroepen (`"required"`), of de tool moet aanroepen die jij noemt | `"tool_choice": "auto"` |

    Nog twee dingen die je hieronder tegenkomt. **`enum`** noemt in een JSON-schema de enige waarden die een argument mag hebben: `"enum": ["km", "mi", "kg", "lb"]`. En **`tool_call_id`** is het veld waarin een resultaat zijn call noemt, met het `"id"` van de call, zodat het model weet welk resultaat bij welke call hoort.

    ### De tool convert

    Zo beschrijft de course zijn functie `convert` voor de API. De functie zelf is gewone Python: hij krijgt de argumenten als dictionary, en geeft het bedrag in de andere eenheid terug als tekst, zoals `"22.99 mi"`.

    ```python
    convert_tool = {
        "type": "function",
        "function": {
            "name": "convert",
            "description": ("Convert an amount from one unit to another: "
                            "kilometres and miles, or kilograms and pounds."),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "the amount to convert, such as 37"},
                    "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                                  "description": "the unit of the amount"},
                    "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                                "description": "the unit to convert to"},
                },
                "required": ["amount", "from_unit", "to_unit"],
            },
        },
    }
    ```

    Het model ziet de code van `convert` nooit: het leest alleen deze beschrijving. Elk request met de tool draagt haar mee, als tokens: de vraag *{q}* is alleen {own} prompt tokens, en {tools} met de beschrijving van `convert`.
    """.replace("{q}", C.QUESTIONS[0][0]).replace("{own}", str(F["prompt_own"])).replace("{tools}", str(F["prompt_tools"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course stuurde dezelfde {n} vragen met de tool `convert`, op twee manieren: met `tool_choice` `"auto"`, zodat het model zelf beslist of het hem aanroept, en met `tool_choice` die `convert` noemt, zodat het moet. Elke keer voerde het programma van de course de calls uit en vroeg het opnieuw, zoals in het recept hieronder.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(r"""
        Met `"auto"` riep Qwen3 `convert` aan bij {called} van de {n} vragen, elke keer met de goede argumenten, en elk van die antwoorden klopte. De andere {missed} beantwoordde het zelf, zonder call, en daarvan klopte er geen tot op twee decimalen: één keer net ernaast, één keer afgerond op één decimaal, en één keer *{wrong}* Met `tool_choice` die `convert` noemt, riep het de tool elke keer aan, met de goede argumenten, en klopten alle {n} antwoorden. Twee requests klinken trager dan één, maar hier waren ze sneller: samen zo'n {tool_s} seconden, tegen {own_s} voor een uitgeschreven berekening.

        Een vraag die geen tool nodig heeft, zoals *{no_need}*, kreeg met `"auto"` geen call: bij {k} vragen van die soort riep Qwen3 niets aan. Daar is `"auto"` voor: het model beslist. Met `"none"` laat de server van de course de tools helemaal uit de prompt: {none_p} prompt tokens in plaats van {auto_p}.
        """.format(called=len(F["auto_called"]), n=F["questions"], missed=len(F["auto_missed"]),
                   wrong=C.run("auto-{}".format(C.WRONG))["steps"][-1]["text"], tool_s=L.number(F["tool_seconds"], 1),
                   own_s=L.number(F["own_seconds"], 1), no_need=C.NO_NEED[0], k=len(C.NO_NEED),
                   none_p=F["none_prompt"], auto_p=F["auto_prompt_no_need"])),
        L.watch_out("Een tool call is een wens van het model, geen belofte. Het model kan een tool overslaan die het zou moeten gebruiken, zoals hierboven, er een aanroepen die het niet zou moeten aanroepen, of argumenten schrijven waar de functie niets mee kan, zoals een eenheid die ze niet kent. Je code controleert wat er binnenkomt, en de functie geeft een fout terug als tekst als ze het werk niet kan doen. De volgende les gaat over tools en fouten waar een model goed mee uit de voeten kan."),
    ])
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### tool_choice

    | Waarde | Het model … |
    |---|---|
    | `"auto"` | mag een tool aanroepen, of antwoorden met tekst: het beslist zelf. De standaard, als het request tools heeft. |
    | `"none"` | antwoordt met tekst, en roept niets aan. |
    | `"required"` | moet een tool aanroepen, een van de lijst. |
    | `{"type": "function", "function": {"name": "convert"}}` | moet `convert` aanroepen. |

    Een afgedwongen call geldt voor één request. Stuur je `"required"` of een tool bij naam opnieuw mee met de resultaten, dan moet het model opnieuw een tool aanroepen: het komt nooit aan antwoorden toe. Dwing de call dus alleen af in het eerste request, zoals de course deed, en stuur het tweede met `"auto"`, of zonder `tool_choice`.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: één ronde function calling

    1. **Beschrijf elke tool**: `"type": "function"`, en onder `"function"` zijn naam, een beschrijving, en zijn argumenten als JSON-schema onder `"parameters"`.
    2. **Vraag** met de vraag en `"tools": [...]`.
    3. **Geen `tool_calls` in het antwoord?** Dan is de `"content"` ervan het antwoord: klaar.
    4. **Anders** voeg je het bericht van het antwoord toe aan de berichten, zoals het terugkwam. Lees voor elke call zijn argumenten met `json.loads`, voer de functie met zijn naam uit, en voeg `{"role": "tool", "tool_call_id": call["id"], "content": result}` toe.
    5. **Vraag opnieuw**, met alle berichten en dezelfde tools. Het model leest de resultaten en antwoordt.

    Een agent herhaalt stap 3 tot en met 5 in een loop, zoals in [Wat een agent is](/course/open/agent-idea?lang=nl). Hier is één ronde genoeg: één vraag, één tool.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vraag *{q}*, zoals de course hem stuurde met `"auto"`.

    1. **De tool**: `convert_tool`, hierboven.
    2. **Vraag**: de berichten bevatten alleen de vraag; het request heeft `"tools": [convert_tool]`.
    3. **Het antwoord** heeft geen tekst, `"finish_reason"` is `"tool_calls"`, en `"tool_calls"` bevat één call: zijn `"id"` is `"call_0"`, zijn naam `"convert"`, en zijn argumenten de tekst `'{args}'`.
    4. **Voer hem uit**: `json.loads` maakt van die tekst een dictionary, en `convert` geeft `"{result}"` terug. De berichten bevatten nu de vraag, het antwoord met zijn call, en `{{"role": "tool", "tool_call_id": "call_0", "content": "{result}"}}`.
    5. **Vraag opnieuw**: het model antwoordt *{answer}*, met `"finish_reason"` `"stop"`.
    """.format(q=C.QUESTIONS[C.WORKED][0], args=C.recorded_messages(C.WORKED)[0][1]["tool_calls"][0]["function"]["arguments"],
               result=C.run("auto-{}".format(C.WORKED))["steps"][0]["results"][0],
               answer=C.run("auto-{}".format(C.WORKED))["steps"][-1]["text"]))
    return


@app.cell(hide_code=True)
def picture_talk():
    mo.Html(C.talk_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_describe():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Beschrijf convert

        Het vak bevat de beschrijving van `convert` met alleen zijn eerste argument, `amount`. Voeg de twee eenheden toe, `from_unit` en `to_unit`: elk een string die alleen `km`, `mi`, `kg` of `lb` kan zijn, met een korte beschrijving. Maak daarna alle drie de argumenten verplicht. Het plaatje onder de check laat je argumenten zien, en de regel van de prompt waarin Qwen3 ze leest.
        """),
        L.task("Zet `from_unit` en `to_unit` in `\"properties\"`, elk met `\"type\": \"string\"` en een `\"enum\"` van de vier eenheden, en noem alle drie in `\"required\"`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=17)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "`\"properties\"` is een dictionary met één key per argument. Elke waarde is een klein schema op zich, zoals dat van `amount`.",
            "Een eenheid is een string met een vaste lijst waarden: `{\"type\": \"string\", \"enum\": [\"km\", \"mi\", \"kg\", \"lb\"], \"description\": ...}`.",
            "Zet `\"from_unit\"` en `\"to_unit\"` onder `\"amount\"`, elk met dat schema, en maak de laatste regel van `\"parameters\"` `\"required\": [\"amount\", ___],`.",
        ],
        why="""```python
"properties": {
    "amount": {"type": "number", "description": "the amount to convert, such as 37"},
    "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                  "description": "the unit of the amount"},
    "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                "description": "the unit to convert to"},
},
"required": ["amount", "from_unit", "to_unit"],
```

Met `\"enum\"` leest het model welke vier waarden er zijn, en schrijft het `\"mi\"`, niet `\"miles\"`: een waarde die `convert` kent. Je beschrijvingen mogen andere woorden zijn; de check kijkt naar de argumenten, hun types en de toegestane waarden.""",
    )
    return


@app.cell(hide_code=True)
def text_answer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voer de calls uit, en stuur de resultaten terug

        `answer(url, question, tools, functions)` vraagt het model met de tools, via `ask`, die een request stuurt zoals in [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl) en het bericht van het antwoord teruggeeft. `functions` bevat de functies zelf bij naam, zoals `{"convert": convert}`. Heeft het bericht tool calls, dan voegt `answer` het toe aan de berichten en vraagt opnieuw, maar de loop over de calls doet nog niets. Laat hem elke call uitvoeren en het resultaat toevoegen. De check roept je `answer` aan tegen een stand-in van de server, met een vraag die één call krijgt, een die er twee tegelijk krijgt, en een die er geen krijgt; het plaatje onder de check laat zien wat je functie stuurde bij de twee calls.
        """),
        L.task("Lees in de loop de argumenten van de call met `json.loads`, voer `functions[name]` ermee uit, en voeg een bericht toe met de rol `tool`, het `id` van de call als `tool_call_id`, en het resultaat als `content`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=24)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "answer",
        [
            "Een call is een dictionary: `call[\"id\"]`, en onder `call[\"function\"]` de `\"name\"` en de `\"arguments\"`. De argumenten zijn een JSON-tekst, nog geen dictionary.",
            "`json.loads(call[\"function\"][\"arguments\"])` geeft de dictionary. `functions[call[\"function\"][\"name\"]]` is de functie die je ermee uitvoert.",
            "Nadat de call gedraaid heeft: `messages = messages + [___]`, met het bericht van de tool in het gat.",
        ],
        why="""```python
def answer(url, question, tools, functions):
    messages = [{"role": "user", "content": question}]
    message = ask(url, messages, tools)
    if message.get("tool_calls"):
        messages = messages + [message]
        for call in message["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = functions[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
        message = ask(url, messages, tools)
    return message["content"]
```

De resultaten komen direct na het bericht van de assistant, één per call, in de volgorde van de calls. `tool_call_id` koppelt elk aan zijn call: bij twee calls tegelijk leest het model welk resultaat de afstand is en welk de datum.""",
    )
    return


@app.cell(hide_code=True)
def text_tool_choice_choice():
    mo.md(r"""
    ### Oefening 3. Laat het de tool aanroepen
    """)
    return


@app.cell(hide_code=True)
def choice_tool_choice_cell():
    choice_tool_choice = L.choice(
        "tool-choice",
        question="Met `\"auto\"` antwoordde Qwen3 *{wrong}* zonder `convert` aan te roepen. Hoe laat je het `convert` aanroepen, en na het resultaat toch antwoorden?".format(
            wrong=C.run("auto-{}".format(C.WRONG))["steps"][-1]["text"]),
        options={
            "every": "Stuur `\"tool_choice\": \"required\"` mee in elk request van het gesprek, het eerste en alle daarna",
            "first": "Noem `convert` in `tool_choice` in het eerste request, en laat het weg uit het tweede",
            "none": "Stuur `\"tool_choice\": \"none\"`, zodat het de vraag zelf beantwoordt, met meer zorg",
        },
        correct="first",
        why={
            "first": "Het eerste request moet `convert` aanroepen; het tweede, met het resultaat, mag antwoorden. Zo kreeg de course alle {n} antwoorden goed.".format(n=F["questions"]),
            "every": "Dan moet het tweede request weer een tool aanroepen, en het derde ook: het model komt nooit aan antwoorden toe. Welk request heeft de call nodig?",
            "none": "`\"none\"` zet de tools uit: dan rekent het model zelf, zoals zonder tools. Welke waarde laat het aanroepen?",
        },
    )
    choice_tool_choice
    return (choice_tool_choice,)


@app.cell(hide_code=True)
def text_other_apis():
    mo.vstack([
        mo.md(r"""
        ### Hetzelfde idee, in andere API's

        De vorm hierboven is die van de API van OpenAI, die de server van de course nadoet, net als de meeste servers van open modellen. Andere API's hebben dezelfde onderdelen onder andere namen. In de API van Anthropic bijvoorbeeld, in november 2026, heeft een tool `input_schema` in plaats van `parameters`; is een call een blok van het type `tool_use` in de content van het antwoord, met zijn argumenten al als dictionary, onder `input`; en gaat het resultaat terug als blok van het type `tool_result` met de `tool_use_id` van de call, in een bericht met de rol `user`. Een beschrijving, een call met een id, en een resultaat met dat id: die drie onderdelen vind je overal.
        """),
        L.watch_out("Elke tool die je meestuurt, kost tokens in elk request, ook als het model er geen aanroept: één tool met drie argumenten bracht de prompt van Qwen3 van {own} naar {tools} tokens. Stuur de tools mee die een request nodig heeft, niet elke tool die je hebt.".format(
            own=F["prompt_own"], tools=F["prompt_tools"])),
    ])
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 2 goed is, kun je je `answer` op het echte model proberen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo stelt de vraag in het vak met de tool `convert`, zoals je beschrijving van oefening 1 hem heeft als die goed is, en anders zoals die van de course. Vraag een omrekening, een vraag die geen tool nodig heeft, of een met een eenheid die `convert` niet kent, zoals feet. Schrijf hem in het Engels, omdat Qwen3 Engels het best schrijft.
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
    question_box = mo.ui.text(value=C.QUESTIONS[0][0], label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Vraag met je answer")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_answer, box_describe, question_box, send_button):
    _answer = C.learner_answer(box_answer.value)
    _tool, _mine = C.learner_tool(box_describe.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _answer is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo vraagt met je eigen `answer`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat je answer naar Qwen3 stuurde, met {}".format("je beschrijving van convert" if _mine else "de beschrijving van convert van de course"),
                  "request": "Request {n}: je answer stuurde deze berichten", "answer": "De server antwoordde:",
                  "id": "id {id}", "answers": "antwoord op {id}:", "finish": "finish_reason: {f}",
                  "status": "De server antwoordde met status {s}: {m}", "gave": "Je answer gaf terug: {v}",
                  "error": "Je answer stopte met {e}.", "none": "Je answer stuurde niets naar de server.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_answer, _tool, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De agent loop van [Wat een agent is: een model, tools en een loop](/course/open/agent-idea?lang=nl) spreekt nu de eigen vorm van de API: tools in het request, calls met een id in het antwoord, resultaten met dat id. Het JSON-schema van [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl) beschrijft nu de argumenten van een tool. In de volgende les, [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl), meet je wat een naam, een beschrijving en een foutmelding doen voor een klein model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_tool_choice, result_answer, result_describe):
    L.wrap_up(result_describe, result_answer, choice_tool_choice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
