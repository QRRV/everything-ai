# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-plan-execute. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-plan-execute", "nl")
    C = L.checks   # de drie tools van de keuken, de runs van Qwen3 als plan en als loop, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [De agent loop](/course/open/build-agent-loop?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="In `run_agent` antwoordt het model met twee tool calls. Wat gaat er in `messages` vóór het volgende request?",
        options={
            "results": "Alleen de twee resultaten, samen in één bericht van de gebruiker, zodat het request kort blijft",
            "both": "Het bericht van het model zoals het kwam, dan een bericht met de rol `tool` voor elke call",
            "last": "Alleen het resultaat van de laatste call, want het model kan later altijd opnieuw om de eerste vragen",
        },
        correct="both",
        why={
            "both": "De server controleert of elke call zijn resultaat heeft, via het id van de call, en het model leest in de volgende ronde zijn eigen calls en hun resultaten.",
            "results": "Zonder zijn eigen bericht met de calls weet het model niet waar de resultaten bij horen, en de server weigert calls zonder resultaat. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "last": "Elke call heeft zijn resultaat nodig: de server weigert een request met een call waarvan het resultaat ontbreekt. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def quick_check_limit():
    quick_limit = L.choice(
        "quick-limit",
        question="Waartegen beschermt de stappenlimiet in `run_agent` je?",
        options={
            "wrong": "Tegen een fout antwoord: zodra de limiet bereikt is, controleert de loop het antwoord met de resultaten van de tools",
            "long": "Tegen een antwoord dat te lang is: de limiet is het grootste aantal tokens dat het model mag schrijven",
            "loop": "Tegen een run die tools blijft aanroepen en nooit antwoordt, wat tijd en tokens kost",
        },
        correct="loop",
        why={
            "loop": "Na `max_steps` requests stopt de loop en geeft hij `None` terug, zodat je code weet dat de agent niet klaar kwam.",
            "wrong": "De loop controleert geen antwoorden; hij telt alleen requests. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "long": "Dat is `max_tokens`, per request. De stappenlimiet telt requests. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_limit
    return (quick_limit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [De agent loop](/course/open/build-agent-loop?lang=nl) koos het model één stap tegelijk: na elke tool call ging het hele gesprek terug naar het model, en het besliste wat het daarna deed. Voor een taak waarvan de stappen vanaf het begin duidelijk zijn, zoals uitrekenen wat je voor een gerecht moet kopen, zijn dat veel requests voor weinig keuze, en het model moet elk resultaat overnemen in zijn volgende call. Het kan ook anders: laat het model eerst het hele plan schrijven, en laat je code het uitvoeren. Dan kun je het plan ook lezen voordat er iets draait. In deze les voer je zo'n plan uit, vind je de stappen die tegelijk kunnen draaien, en meet je het tegen een agent loop.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **plan-and-execute** | een agent waarin het model eerst alle stappen als plan schrijft, waarna code ze uitvoert; aan het eind schrijft nog één request het antwoord uit de resultaten |
        | **ReWOO** | *reasoning without observation*, van Xu en anderen (2023): een vorm van plan-and-execute waarin een stap het resultaat van een eerdere stap kan gebruiken via zijn nummer, zodat het model het plan schrijft zonder een resultaat te zien |
        | **`"#1"`** | in de plannen van deze les: het resultaat van stap 1, als waarde in een latere stap |
        | **LLMCompiler** | van Kim en anderen (2023): een vorm van plan-and-execute die de stappen die elkaars resultaten niet nodig hebben, tegelijk laat draaien |
        | **golf** | stappen die tegelijk kunnen draaien, omdat elk resultaat dat ze gebruiken er al is |

        De keuken is die van [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl), met drie tools voor het plan. Elke tool krijgt zijn argumenten als dictionary:

        ```python
        recipe({"dish": "pancakes"})       # {"flour": 250, "milk": 450, "eggs": 110, "butter": 25}
        pantry({"items": ["milk", "eggs"]})   # {"milk": 200, "eggs": 0}
        missing({"need": {"milk": 450, "eggs": 110}, "have": {"milk": 200, "eggs": 0}})   # {"milk": 250, "eggs": 110}
        ```

        `pantry` neemt ook een recept als `items`: dan zoekt hij elk ingrediënt ervan op. Een plan is JSON met een lijst stappen, elk een tool met zijn argumenten. Waar een stap een eerder resultaat nodig heeft, schrijft het plan het nummer ervan. Het plan voor `pancakes`:

        ```json
        {"steps": [{"tool": "recipe", "args": {"dish": "pancakes"}},
                   {"tool": "pantry", "args": {"items": "#1"}},
                   {"tool": "missing", "args": {"need": "#1", "have": "#2"}}]}
        ```

        Je code voert stap 1 uit, bewaart het resultaat als `"#1"`, geeft het aan stap 2 als `items`, enzovoort. Het model schrijft het plan in één request, in JSON mode, en hoeft nooit een resultaat over te nemen. Een tweede request krijgt de vraag en het resultaat van elke stap, en schrijft het antwoord.
        """),
        mo.accordion({"De twee prompts van de course, als je ze wilt lezen": mo.md(
            "```text\n" + C.PLAN.format(question="...") + "\n```\n\n```text\n" + C.SOLVE + "\n```")}),
    ])
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course stelde Qwen3-0.6B de {single} vragen van [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl) en nog {double} over twee gerechten, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. Elke vraag ging op drie manieren. Als eigen plan van Qwen3: de prompt voor het plan in JSON mode, de code van de course die de stappen uitvoerde, en een tweede request voor het antwoord. Met het plan van de course: de code voerde het plan uit dat de course voor die vraag schreef, en het model schreef alleen het antwoord uit de resultaten. En als agent loop, met dezelfde drie tools als tools van de API en een limiet van {steps} requests.
    """.format(single=F["single"], double=F["double"], steps=F["steps"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een plan dat vooraf geschreven is, kan niet reageren op wat een stap vindt. Ontbreekt het recept, dan draaien de stappen erna toch, op een fout. Gebruik een plan waar de stappen vanaf het begin duidelijk zijn, en een loop waar de volgende stap afhangt van wat de vorige vond."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: eerst plannen, dan doen

    1. **Beschrijf de tools in de prompt,** met de vorm van een plan en hoe een stap een eerder resultaat noemt.
    2. **Vraag om het plan in JSON mode,** en lees het met `json.loads`. Kan je code het niet lezen? Dan is er geen plan om uit te voeren.
    3. **Lees het plan voordat het draait,** of laat een persoon het goedkeuren, zoals bij een tool die iets verandert.
    4. **Voer de stappen op volgorde uit,** en geef elke stap het resultaat van elke eerdere stap die hij via zijn nummer noemt.
    5. **Laat de stappen van een golf tegelijk draaien** als je tools traag zijn, zoals calls over een netwerk.
    6. **Vraag nog één keer om het antwoord,** met de vraag en het resultaat van elke stap.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(C.WORKED, "nl"))
    return


@app.cell(hide_code=True)
def text_run_plan():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een plan uitvoeren

        `run_plan(plan)` is de uitvoerder: hij voert de stappen van een plan op volgorde uit, en geeft het resultaat van elke stap terug onder zijn nummer, als `{"#1": ..., "#2": ..., ...}`. `FUNCTIONS["recipe"]` is de tool zelf, en krijgt de argumenten als dictionary. In de binnenste loop is `value` één argument van een stap: een tekst, zoals `"pancakes"` of `"#1"`, of een lijst, zoals `["milk", "eggs"]`. Een tekst die het nummer van een eerdere stap is, wordt het resultaat van die stap; daarna gaat elke waarde onder zijn naam in `arguments`. Maak de binnenste loop af. Het vak kent `FUNCTIONS`. De check laat je functie draaien op het plan voor `pancakes` en op zes andere plannen; het plaatje eronder laat elke stap van het plan voor `pancakes` zien met wat je functie gaf.
        """),
        L.task("Vervang in de binnenste loop een waarde die in `results` staat door dat resultaat, en zet elke waarde onder zijn naam in `arguments`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_plan():
    box_run_plan = L.code_box("run-plan", lines=12)
    box_run_plan
    return (box_run_plan,)


@app.cell(hide_code=True)
def check_run_plan(box_run_plan):
    result_run_plan = L.check("run-plan", box_run_plan.value)
    result_run_plan
    return (result_run_plan,)


@app.cell(hide_code=True)
def hints_run_plan():
    L.hints(
        "run-plan",
        [
            "Met elke waarde gebeuren twee dingen. Eerst: is het het nummer van een eerdere stap? Dan wordt het het resultaat van die stap. Daarna: hij gaat in `arguments`.",
            "`results` bevat het resultaat van elke eerdere stap onder zijn nummer, zoals `\"#1\"`. `value in results` zegt of `value` een van die nummers is. Een lijst kan geen key van een dictionary zijn, dus `value in results` stopt bij een lijst met een fout: kijk eerst of `value` een tekst is, met `isinstance(value, str)`.",
            "Onder `if isinstance(value, str) and value in results:`: `value = ___`; daarna, niet ingesprongen onder de `if`, `arguments[name] = value`.",
        ],
        why="""```python
def run_plan(plan):
    results = {}
    for step in plan["steps"]:
        arguments = {}
        for name, value in step["args"].items():
            if isinstance(value, str) and value in results:
                value = results[value]
            arguments[name] = value
        results["#" + str(len(results) + 1)] = FUNCTIONS[step["tool"]](arguments)
    return results
```

`"pancakes"` is geen key van `results`, dus die blijft zoals hij is; `"#1"` wel, en wordt het recept. Een lijst is geen tekst, dus die gaat zoals hij is in `arguments`; door `and` kijkt de code alleen in `results` als `value` een tekst is. Het nummer van een nieuw resultaat is één meer dan het aantal resultaten tot nu toe.""",
    )
    return


@app.cell(hide_code=True)
def text_waves():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stappen die tegelijk kunnen draaien

        Een stap kan beginnen zodra elk resultaat dat hij gebruikt er is. `waves(plan)` geeft de golf van elke stap onder zijn nummer: 1 voor een stap die geen eerder resultaat gebruikt, en anders één meer dan de laatste golf van de resultaten die hij gebruikt. In het plan voor twee gerechten zitten beide recepten in golf 1, beide keren opzoeken in de voorraadkast in golf 2, en beide stappen met `missing` in golf 3: zes stappen in drie golven. Maak de binnenste loop af. De check laat je functie draaien op dat plan en zes andere; het plaatje laat het plan voor twee gerechten zien in de golven die je functie vond.
        """),
        L.task("Maak in de binnenste loop `w` minstens één meer dan de golf van een eerdere stap als `value` het nummer van die stap is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_waves():
    box_waves = L.code_box("waves", lines=10)
    box_waves
    return (box_waves,)


@app.cell(hide_code=True)
def check_waves(box_waves):
    result_waves = L.check("waves", box_waves.value)
    result_waves
    return (result_waves,)


@app.cell(hide_code=True)
def hints_waves():
    L.hints(
        "waves",
        [
            "Een stap wacht op elk resultaat dat hij gebruikt. Hij kan dus pas beginnen na het laatste ervan: zijn golf is één meer dan de grootste golf daaronder.",
            "`wave` bevat de golf van elke eerdere stap onder zijn nummer. Als `value in wave`, heeft deze stap een golf nodig die minstens één meer is dan die van die stap. `max(a, b)` geeft het grootste van twee getallen. Net als in oefening 1 kan een waarde een lijst zijn: kijk eerst of hij een tekst is.",
            "`if isinstance(value, str) and value in wave:`, met `w = max(w, ___)` eronder.",
        ],
        why="""```python
def waves(plan):
    wave = {}
    for step in plan["steps"]:
        w = 1
        for name, value in step["args"].items():
            if isinstance(value, str) and value in wave:
                w = max(w, wave[value] + 1)
        wave["#" + str(len(wave) + 1)] = w
    return wave
```

Met `max` maakt de volgorde van de argumenten niet uit: een stap met `"have": "#2"` vóór `"need": "#1"` wacht toch op stap 2.""",
    )
    return


@app.cell(hide_code=True)
def text_plan_or_loop():
    mo.md(r"""
    ### Oefening 3. Een plan of een loop
    """)
    return


@app.cell(hide_code=True)
def choice_plan_or_loop_cell():
    choice_plan_or_loop = L.choice(
        "plan-or-loop",
        question="Bij welke taak past een plan dat eerst geschreven wordt beter dan een agent loop?",
        options={
            "search": "Een recept vinden dat je gasten lekker vinden, door pagina's te lezen tot er een past bij wat ze eten",
            "known": "Boodschappen voor twee gerechten: beide recepten opzoeken, dan de voorraadkast, dan wat er ontbreekt",
            "debug": "Een test repareren die faalt, waarbij elke nieuwe run van de tests laat zien wat je daarna probeert",
        },
        correct="known",
        why={
            "known": "De stappen liggen vast voordat er een resultaat binnenkomt; alleen de waarden erin hangen af van eerdere stappen, en daar zorgen de nummers zoals `\"#1\"` voor.",
            "search": "Wat je daarna leest, hangt af van wat de vorige pagina zei: de agent moet elk resultaat zien voordat hij kiest. Dat vraagt om een loop.",
            "debug": "Elke run van de tests verandert de volgende stap. Een plan dat vooraf geschreven is, kan niet weten wat de tests gaan zeggen.",
        },
    )
    choice_plan_or_loop
    return (choice_plan_or_loop,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Is oefening 1 goed, dan kun je een plan van het echte model laten draaien: de course start een eigen server op je computer, met Qwen3-0.6B erachter, vraagt het om een plan, voert het plan uit met je `run_plan`, en vraagt om het antwoord. Is oefening 2 ook goed, dan zie je ook je golven. In de demo geeft een tool die faalt zijn fout terug als tekst, zoals in de runs van de course, zodat de stappen erna toch draaien. De vraag in het vak is die van het uitgewerkte voorbeeld. Vraag in je eigen woorden naar een of twee gerechten van de keuken.
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
    question_box = mo.ui.text(value=C.TWO[0][0], label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Plan en voer uit")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_plan(box_run_plan, box_waves, question_box, send_button):
    _run_plan = C.learner_fn("run-plan", box_run_plan.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _run_plan is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo voert het plan uit met je eigen `run_plan`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om het te laten draaien.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3, en het model schrijft zijn plan …*"))
        _words = {"title": "Een plan van Qwen3, uitgevoerd door je run_plan", "wrote": "1. Het plan dat Qwen3 schreef",
                  "waves": "Je golven", "wave": "golf {k}", "ran": "2. Wat je run_plan gaf voor elke stap",
                  "answered": "3. Het antwoord dat Qwen3 uit de resultaten schreef",
                  "unread": "Je code kan dit niet als plan lezen: het is geen JSON met een lijst stappen. Er is niets om uit te voeren.",
                  "error": "Je run_plan stopte met {e}. Een plan van een model kan een waarde bevatten die je code niet verwacht: lees het plan hierboven.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_run_plan, C.learner_fn("waves", box_waves.value), question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop van [De agent loop](/course/open/build-agent-loop?lang=nl) liet het model elke stap kiezen nadat het het vorige resultaat had gezien; hier schrijft het eerst alle stappen, en voert je code ze uit, met de resultaten doorgegeven via hun nummers. Stappen die elkaar niet nodig hebben tegelijk laten draaien, konden de workers van een orchestrator in [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl) ook. De volgende les gaat over wat een agent onthoudt, binnen één gesprek en over gesprekken heen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_plan_or_loop, result_run_plan, result_waves):
    L.wrap_up(result_run_plan, result_waves, choice_plan_or_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
