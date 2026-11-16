# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-workflow-patterns. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-workflow-patterns", "nl")
    C = L.checks   # de vijf patronen, de runs van Qwen3 door de course (tips onder een woordlimiet, menu's opgesplitst in gerechten), de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl), en één over [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gate():
    quick_gate = L.choice(
        "quick-gate",
        question="In de workflow `shopping` noemt het model het gerecht. Wat doet de code met die naam voordat hij `to_buy` aanroept?",
        options={
            "check": "Hij controleert of het gerecht in `DISHES` staat, en geeft `SORRY` terug als dat niet zo is",
            "trust": "Niets: het model heeft de vraag gelezen, dus het gerecht dat het noemt, is het gerecht waarnaar gevraagd werd",
            "again": "Hij vraagt het model een tweede keer, en gaat alleen verder als beide antwoorden hetzelfde gerecht noemen",
        },
        correct="check",
        why={
            "check": "Het model vult een stap in; de code beslist of de volgende stap door mag gaan. Die controle is een poort.",
            "trust": "Een klein model kan een gerecht noemen dat de keuken niet heeft, of een woord dat helemaal geen gerecht is. Zie [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl).",
            "again": "Nog eens vragen kost een request, en kan twee keer dezelfde verkeerde naam geven. Een controle in code is precies en gratis. Zie [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl).",
        },
    )
    quick_gate
    return (quick_gate,)


@app.cell(hide_code=True)
def quick_check_router():
    quick_router = L.choice(
        "quick-router",
        question="Wat doet een router tussen een klein en een groot model?",
        options={
            "both": "Hij stuurt elke vraag naar beide modellen, en houdt het langste antwoord",
            "train": "Hij traint het kleine model op de antwoorden van het grote, zodat er uiteindelijk één model genoeg is",
            "send": "Hij stuurt elke vraag naar het model dat erbij past, zoals makkelijke vragen naar het kleine",
        },
        correct="send",
        why={
            "send": "De router beslist per vraag welk pad die neemt. In deze les kunnen de paden ook prompts of stukken code zijn.",
            "both": "Dat kost elke keer beide modellen. Een router kiest er één. Zie [Een model kiezen](/course/open/build-choose-model?lang=nl).",
            "train": "Dat is een ander idee. Een router laat beide modellen zoals ze zijn, en kiest alleen. Zie [Een model kiezen](/course/open/build-choose-model?lang=nl).",
        },
    )
    quick_router
    return (quick_router,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De workflow van de vorige les had drie stappen na elkaar: het model noemt het gerecht, je code rekent de lijst uit, het model schrijft het antwoord. Niet elke taak past in een rij. Een vraag kan van verschillende soorten zijn; een menu kan twee gerechten hebben of vijf; een tip kan te lang uitvallen. De meeste workflows die mensen bouwen, bestaan uit een paar vormen die precies op zulke gevallen antwoord geven. Ken je ze, dan zie je welke een taak nodig heeft, en bouw je die in een paar regels.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **orchestrator-workers** | een patroon waarin het model eerst bepaalt uit welke delen een taak bestaat (de orchestrator), en elk deel naar een eigen aanroep gaat, een worker; daarna voegt je code de resultaten samen |
    | **evaluator-optimizer** | een patroon waarin één aanroep een antwoord schrijft, een controle het beoordeelt, en een antwoord dat niet goed is, teruggaat naar de schrijver met wat er mis is, in een loop, tot het erdoor komt of de rondes op zijn |
    | **worker** | de aanroep, of de functie, die één deel van de taak doet |

    Anthropic beschrijft vijf patronen in *Building effective agents* (december 2024). Hier staan ze, toegepast op de keuken van de vorige les:

    1. **Stappen na elkaar** ([prompt chaining](/course/open/start-prompt-chains?lang=nl)): elke stap leest wat de vorige gaf. Tussen twee stappen kan je code het resultaat controleren voordat hij verdergaat, zoals `shopping` deed met het gerecht.
    2. **Router**: een eerste aanroep bepaalt wat voor vraag het is, en elke soort krijgt een eigen pad: een vraag over een recept, een over boodschappen, een waar de keuken geen antwoord op heeft. Het is de router van [Een model kiezen](/course/open/build-choose-model?lang=nl), maar de paden kunnen ook prompts of code zijn.
    3. **Delen naast elkaar**: delen die elkaar niet nodig hebben, draaien tegelijk, zoals een tip voor elk van de zes gerechten; of dezelfde vraag gaat drie keer weg, en je code houdt het antwoord dat het vaakst terugkomt.
    4. **Orchestrator en workers**: zoals delen naast elkaar, maar de delen zijn vooraf niet bekend. Bij *Wat koop ik voor `pancakes` en `tomato soup`?* noemt het model eerst de gerechten; je code geeft elk gerecht aan een worker, en voegt de lijsten samen.
    5. **Evaluator en optimizer**: een antwoord, een controle, en terug met wat er mis is. De controle kan een andere aanroep van een model zijn, of code als er een duidelijke regel is, zoals een limiet op het aantal woorden.
    """)
    return


@app.cell(hide_code=True)
def picture_patterns():
    mo.Html(C.patterns_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course probeerde twee van de patronen op Qwen3-0.6B, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. Voor de evaluator vroeg ze een tip voor elk van de {dishes} gerechten, in hooguit {most} woorden. Een controle in code telde de woorden, en een tip die te lang was, ging terug naar het model, tot {tries} antwoorden per gerecht. De course probeerde twee teksten voor die terugkoppeling. De eerste: *{again}* De tweede: *{new}*
    """.format(dishes=F["dishes"], most=F["limit"], tries=F["tries"],
               again=C.AGAIN.format(n="…", limit=F["limit"]), new=C.FEEDBACK.format(n="…", limit=F["limit"])))
    return


@app.cell(hide_code=True)
def picture_tips():
    mo.Html(C.tips_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_tips():
    mo.md(C.tips_text("nl"))
    return


@app.cell(hide_code=True)
def picture_plans():
    mo.Html(C.plans_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_plans():
    mo.vstack([
        mo.md(C.plans_text("nl")),
        L.watch_out("Een evaluator is zo goed als zijn controle. Een controle in code, zoals woorden tellen of JSON lezen, zegt precies wat er mis is. Is een model de beoordelaar, dan kan dat zich ook vergissen, en een loop die op zijn *ja* wacht, eindigt misschien nooit. Geef elke loop een limiet op het aantal rondes, en bepaal wat je app doet als geen antwoord erdoor komt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: welk patroon?

    1. **Zijn de stappen voor elke invoer dezelfde, de een na de ander?** Stappen na elkaar, met een controle in code tussen twee stappen waar een fout zou doorwerken.
    2. **Hebben verschillende soorten invoer een andere aanpak nodig?** Een router, met een pad voor invoer die hij niet kent.
    3. **Zijn er delen die elkaar niet nodig hebben, en ken je ze vooraf?** Delen naast elkaar.
    4. **Ken je de delen pas als je de invoer hebt gelezen?** Een orchestrator die ze noemt, en een worker per deel; controleer elk deel dat hij noemt.
    5. **Is er een duidelijke toets voor een goed antwoord?** Een evaluator, met een limiet op de rondes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def text_improve():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De loop van de evaluator

        `improve(ask, task, check, rounds)` is de evaluator en optimizer. `ask(messages)` stuurt de berichten naar het model en geeft zijn antwoord terug als tekst; `check(answer)` geeft `None` als het antwoord goed is, of een tekst die zegt wat er mis is, zoals `"Your tip has 15 words. Write it again in at most 8 words."`. Maak de loop af. Het model schrijft pas een beter antwoord als het zijn eigen antwoord leest, en wat daar mis mee was. De check probeert vijf stand-ins van het model: een die meteen goed is, een paar die twee of drie rondes nodig hebben, en een paar die het nooit op tijd goed krijgen. Het plaatje eronder volgt je loop ronde na ronde.
        """),
        L.task("Geef in de loop het antwoord terug als de controle geen probleem vindt; zet anders het antwoord en het probleem in `messages`, elk met de rol die het model nodig heeft om het te lezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_improve():
    box_improve = L.code_box("improve", lines=10)
    box_improve
    return (box_improve,)


@app.cell(hide_code=True)
def check_improve(box_improve):
    result_improve = L.check("improve", box_improve.value)
    result_improve
    return (result_improve,)


@app.cell(hide_code=True)
def hints_improve():
    L.hints(
        "improve",
        [
            "Twee gevallen na `problem = check(answer)`: geen probleem, en wel een probleem. In het eerste is de loop klaar. In het tweede moet de volgende ronde weten wat er misging.",
            "Het eigen antwoord van het model gaat in `messages` met de rol `assistant`, wat er mis mee is daarna met de rol `user`. Dan vraagt de loop opnieuw.",
            "`if problem is None:` met `return answer` eronder; dan `messages = messages + [...]` met twee dictionaries: `{\"role\": \"assistant\", \"content\": answer}` en `{\"role\": \"user\", \"content\": problem}`.",
        ],
        why="""```python
def improve(ask, task, check, rounds):
    messages = [{"role": "user", "content": task}]
    for n in range(rounds):
        answer = ask(messages)
        problem = check(answer)
        if problem is None:
            return answer
        messages = messages + [{"role": "assistant", "content": answer},
                               {"role": "user", "content": problem}]
    return None
```

De `None` na de loop doet ertoe: kwam er op tijd geen antwoord door, dan weet je code dat, en kan hij het zeggen in plaats van een tip te gebruiken die te lang is.""",
    )
    return


@app.cell(hide_code=True)
def text_orchestrate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De orchestrator

        `orchestrate(ask, question, worker)` is een orchestrator met workers. De eerste regel vraagt het model over welke gerechten de vraag gaat, met de prompt `PLAN`, en leest de JSON die terugkomt, zoals `{"dishes": ["pancakes", "tomato soup"]}`. `worker(dish)` geeft de boodschappenlijst van één gerecht als tekst, zoals `"pancakes: milk 250 g, eggs 110 g"`. Maak de functie af, zodat `results` voor elk gerecht van het plan de lijst van de worker bevat, in de volgorde van het plan. Het plan komt van een model: een gerecht dat de keuken niet kent, heeft geen lijst. Het vak kent `PLAN` en `DISHES`. De check probeert plannen van twee en drie gerechten, een met een `pizza` erin, en een leeg plan.
        """),
        L.task("Geef elk gerecht van het plan dat in `DISHES` staat aan `worker`, en voeg toe aan `results` wat die teruggeeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_orchestrate():
    box_orchestrate = L.code_box("orchestrate", lines=10)
    box_orchestrate
    return (box_orchestrate,)


@app.cell(hide_code=True)
def check_orchestrate(box_orchestrate):
    result_orchestrate = L.check("orchestrate", box_orchestrate.value)
    result_orchestrate
    return (result_orchestrate,)


@app.cell(hide_code=True)
def hints_orchestrate():
    L.hints(
        "orchestrate",
        [
            "Het plan bepaalt hoeveel workers er zijn: één per gerecht in `plan[\"dishes\"]`. Een loop over die lijst doet het, hoe lang die ook is.",
            "Sla in de loop een gerecht over dat niet in `DISHES` staat; geef elk ander gerecht aan `worker`, en voeg het resultaat toe aan `results`.",
            "`for dish in plan[\"dishes\"]:`, met `if dish in DISHES:` eronder, en daaronder `results.append(worker(dish))`.",
        ],
        why="""```python
def orchestrate(ask, question, worker):
    plan = json.loads(ask(PLAN.format(question=question)))
    results = []
    for dish in plan["dishes"]:
        if dish in DISHES:
            results.append(worker(dish))
    return results
```

Het model bepaalt de delen; je code deelt ze uit en controleert elk deel. De workers hebben elkaar niet nodig, dus ze zouden ook tegelijk kunnen draaien.""",
    )
    return


@app.cell(hide_code=True)
def text_pattern():
    mo.md(r"""
    ### Oefening 3. Het patroon voor een taak
    """)
    return


@app.cell(hide_code=True)
def choice_pattern_cell():
    choice_pattern = L.choice(
        "pattern",
        question="Een winkel laat een model elke productbeschrijving naar het Nederlands vertalen. Een script kan controleren of elk getal en elke productnaam van het origineel nog in de vertaling staat. Welk patroon past?",
        options={
            "router": "Een router die lange beschrijvingen naar een groot model stuurt, en korte naar een klein model",
            "evaluator": "Een evaluator: een vertaling die een getal of naam kwijt is, gaat terug, met wat er ontbreekt",
            "orchestrator": "Een orchestrator die elke beschrijving eerst in zinnen opsplitst, met een worker per zin",
        },
        correct="evaluator",
        why={
            "evaluator": "Er is een duidelijke toets voor een goede vertaling, en het script kan precies zeggen wat er ontbreekt. Dat heeft een evaluator nodig, met een limiet op de rondes.",
            "router": "Een router kan geld besparen, maar niets erin merkt een vertaling op die een getal kwijt is.",
            "orchestrator": "Opsplitsen in zinnen vangt een kwijtgeraakt getal ook niet, en de delen van een beschrijving zijn vooraf bekend.",
        },
    )
    choice_pattern
    return (choice_pattern,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Is oefening 1 goed, dan kun je je loop op het echte model proberen: de course start een eigen server op je computer, met Qwen3-0.6B erachter, en de controle van de course telt de woorden. Kies een gerecht, en kijk hoeveel rondes je `improve` nodig heeft, tot {tries}.
    """.format(tries=F["tries"]))
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
def dish_box_cell():
    dish_box = mo.ui.dropdown(options=list(C.DISHES), value=C.DISHES[1], label="Gerecht")
    send_button = mo.ui.run_button(label="Vraag met je improve")
    mo.vstack([dish_box, send_button])
    return dish_box, send_button


@app.cell(hide_code=True)
def demo_improve(box_improve, dish_box, send_button):
    _improve = C.learner_improve(box_improve.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _improve is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo vraagt met je eigen loop.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Je improve met Qwen3, ronde na ronde", "round": "Ronde {n}: het model schreef",
                  "ok": "goed", "gave": "Je improve gaf terug: {v}", "error": "Je improve stopte met {e}.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_improve, dish_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De workflow van [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl) bestond uit stappen na elkaar met een controle ertussen; de router van [Een model kiezen](/course/open/build-choose-model?lang=nl) werd een van de vijf patronen. In alle vijf bepaalt je code de paden. In de volgende les, [De agent loop](/course/open/build-agent-loop?lang=nl), bepaalt het model ze weer: je bouwt de hele loop van een agent via de API, met een gedachte voor elke tool call.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pattern, result_improve, result_orchestrate):
    L.wrap_up(result_improve, result_orchestrate, choice_pattern)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
