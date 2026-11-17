# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-kinds-of-agents. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-kinds-of-agents", "nl")
    C = L.checks   # de vier werelden en hun tools, de runs van Qwen3 van de course, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [De agent loop](/course/open/build-agent-loop?lang=nl), en een over [Zoeken, deep research en leren met AI](/course/open/build-research-and-learn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="Wanneer houdt `run_agent` op het model te vragen?",
        options={
            "done": "Als het model schrijft dat het klaar is, of zodra elke tool minstens één keer aangeroepen is",
            "stop": "Als een antwoord geen tool calls heeft, of als hij `max_steps` requests gestuurd heeft",
            "error": "Als een tool een fout teruggeeft, want daarna kan het model niet verder",
        },
        correct="stop",
        why={
            "stop": "Een antwoord zonder tool calls is het antwoord van het model op de taak; de stapgrens stopt een model dat tools blijft aanroepen.",
            "done": "De loop leest niet wat het model schrijft; hij kijkt alleen of het antwoord tool calls heeft. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "error": "Een fout gaat als resultaat van de tool terug naar het model, en het model kan iets anders proberen. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def quick_check_source():
    quick_source = L.choice(
        "quick-source",
        question="Een verslag van deep research geeft een pagina als bron van een bewering. Wanneer is die bron iets waard?",
        options={
            "known": "Als de pagina op een bekende site staat, zodat wat erop staat waarschijnlijk klopt",
            "twice": "Als het verslag dezelfde pagina bij meer beweringen geeft, want dan moet hij gelezen zijn",
            "opened": "Zodra je de pagina geopend hebt en de bewering daar gevonden hebt",
        },
        correct="opened",
        why={
            "opened": "Een model kan een pagina noemen die het nooit las, of een pagina die iets anders zegt. Alleen de pagina zelf laat het zien.",
            "known": "Op een bekende site kan iets anders staan dan het verslag beweert, of er staat niets over. Zie [Zoeken, deep research en leren met AI](/course/open/build-research-and-learn?lang=nl).",
            "twice": "Een model kan dezelfde pagina steeds weer noemen zonder hem gelezen te hebben. Zie [Zoeken, deep research en leren met AI](/course/open/build-research-and-learn?lang=nl).",
        },
    )
    quick_source
    return (quick_source,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een coding agent, een agent voor deep research, een persoonlijke agent in je mail en een winkelagent lijken vier verschillende dingen. Eronder zit dezelfde loop die je in [De agent loop](/course/open/build-agent-loop?lang=nl) schreef: stuur de berichten en de tools, voer de call in het antwoord uit, voeg het resultaat toe, en opnieuw. Wat ze verschillend maakt, is welke tools ze krijgen, en wat er met die tools mis kan gaan. Een coding agent kan een project kapotmaken, een research agent kan een bron noemen die hij nooit las, een persoonlijke agent kan een mail versturen uit jouw naam, en een winkelagent kan je geld uitgeven. Elke soort heeft dus een eigen grens nodig, in de code die zijn tools uitvoert.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **agentic commerce** | winkelen en betalen door agents namens een persoon, binnen grenzen die die persoon gesteld heeft |
    | **mandaat** | wat een persoon een agent vooraf toestaat, zoals in welke winkels hij zelf mag betalen en hoeveel per bestelling |
    | **testverslag** | wat een testrun teruggeeft: hoeveel tests slaagden, en bij elke mislukking wat de aanroep gaf en wat er verwacht werd |
    | **`except Exception as e:`** | vangt een fout van elke soort, en geeft hem de naam `e`; `str(e)` is zijn melding |
    | **`x not in items`** | `True` als `x` niet in de lijst `items` staat; `x in items` is het omgekeerde |
    | **`a and b`** | `True` alleen als `a` en `b` allebei `True` zijn |

    De course bouwde vier kleine werelden, elk met eigen tools. Een coding agent werkt in een project met één bestand, `shop.py`, en tests die het controleren. Een research agent leest de pagina's van de website van een stad, `town.example`. Een persoonlijke agent heeft een mailbox en een agenda. Een winkelagent zoekt in twee webwinkels, `shop.example` en `deals.example`, vult een winkelwagen en betaalt. Alle vier draaien met dezelfde loop, hetzelfde model en dezelfde stapgrens. Alleen de system prompt, de tools en de grens verschillen:
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

    De course gaf Qwen3-0.6B drie taken in elke wereld, bij temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt, met hoogstens één tool call per antwoord en hoogstens {s} requests per taak. In het eerste request van elke taak was `tool_choice` `"required"`, zodat de agent begint met een tool call; de latere requests lieten de keuze aan het model. De taken van de coding agent zijn drie bugs in `shop.py`. Die van de research agent zijn drie vragen over de stad; een van de pagina's is een oud bericht op een forum dat een andere openingstijd geeft. De persoonlijke agent zet een afspraak uit een mail in de agenda, antwoordt een collega per mail, en zegt wat er in de agenda staat. De winkelagent krijgt een mandaat in zijn system prompt: hij mag zelf alleen betalen bij `shop.example`, tot {m} per bestelling.
    """.format(s=F["steps"], m=C.euro(C.MANDATE["max"], "nl")))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een grens in de system prompt is een vraag aan het model; de grens in de code die de tool uitvoert, is degene die standhoudt. Geef elke soort agent alleen de tools die zijn taken nodig hebben, en zet de stap die niet terug kan, zoals versturen of betalen, achter het ja van de gebruiker of een mandaat dat de gebruiker gaf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een nieuwe soort agent

    1. **Begin bij de loop die je hebt:** dezelfde requests, dezelfde stapgrens.
    2. **Geef hem de tools van zijn wereld,** en niet meer: wat hij moet lezen, en wat hij moet veranderen.
    3. **Vraag bij elke tool wat er mis kan gaan,** en wat niet terug kan: een gewist bestand, een verstuurde mail, een betaling.
    4. **Zet de grens in de code die de tool uitvoert:** de tests blijven zoals ze zijn, een bron telt pas als hij geopend is, een mail wacht op het ja van de gebruiker, een betaling blijft binnen het mandaat.
    5. **Geef de agent resultaten waar hij iets mee kan:** een testverslag dat zegt wat de aanroep gaf en wat er verwacht werd, een fout die zegt wat wel mag.
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
def text_report():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het verslag dat een coding agent leest

        `test_report(name, fn, cases)` draait de tests van een functie en schrijft het verslag dat als resultaat van `run_tests` terug naar de coding agent gaat. `name` is de naam van de functie, `fn` de functie zelf, en `cases` een lijst van paren: een argument, en wat de functie daarvoor moet geven. De eerste regels lopen de gevallen langs, schrijven elke aanroep als tekst, zoals `total([110, 240])`, en zetten aan het eind het verslag in elkaar. Roep voor elk geval `fn` aan met het argument. Geeft hij iets anders dan verwacht, voeg dan een regel als `total([110, 240]) gave 351, expected 350.` toe aan `failed`. Stopt hij met een fout, voeg dan een regel toe als `cheapest([]) stopped with an error: list index out of range.`, waarin het deel na de dubbele punt `str(e)` is. De check probeert de functie van taak 1 vóór en na de reparatie, en de functies van taak 2 en 3.
        """),
        L.task("Vervang `...` door een `try`-blok dat `fn` aanroept en een regel toevoegt als de waarde niet de verwachte is, en een blok `except Exception as e:` dat een regel voor de fout toevoegt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Per geval kunnen er twee dingen gebeuren: `fn` geeft een waarde terug, die fout kan zijn, of hij stopt met een fout. `try` en `except` vangen het tweede op; vergelijk binnen de `try` de waarde.",
            "In de `try`: `got = fn(argument)`, dan `if got != expected:` met `failed.append(...)` van `call`, `\" gave \"`, `str(got)`, `\", expected \"`, `str(expected)` en `\".\"`, aan elkaar met `+`.",
            "`try:`, `got = fn(argument)`, `if got != expected:`, `failed.append(call + \" gave \" + str(got) + \", expected \" + str(expected) + \".\")`; dan `except Exception as e:` met `failed.append(call + \" stopped with an error: \" + str(e) + \".\")`.",
        ],
        why="""```python
def test_report(name, fn, cases):
    failed = []
    for argument, expected in cases:
        call = name + "(" + str(argument) + ")"
        try:
            got = fn(argument)
            if got != expected:
                failed.append(call + " gave " + str(got) + ", expected " + str(expected) + ".")
        except Exception as e:
            failed.append(call + " stopped with an error: " + str(e) + ".")
    passed = len(cases) - len(failed)
    return " ".join([str(passed) + " of " + str(len(cases)) + " tests passed."] + failed)
```

De agent repareert de code aan de hand van wat het verslag zegt: welke aanroep mislukte, wat hij gaf, en wat hij had moeten geven.""",
    )
    return


@app.cell(hide_code=True)
def text_unread():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Bronnen die hij nooit opende

        Een research agent noemt zijn bronnen tussen vierkante haken, zoals `[3]`. `unread(answer, opened)` geeft de nummers van de bronnen die het antwoord noemt en die niet in `opened` staan, de lijst van pagina's die de agent las: elk nummer één keer, in de volgorde van het antwoord. `answer.split("[")` knipt de tekst bij elke `[`; elk deel na het eerste begint met een getal en `]`. De check probeert het antwoord uit het plaatje en nog vier.
        """),
        L.task("Vervang `...` door een regel die het deel vóór `]` een getal maakt, en een `if` die dat getal aan `missing` toevoegt als het niet in `opened` staat en nog niet in `missing`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unread():
    box_unread = L.code_box("unread", lines=8)
    box_unread
    return (box_unread,)


@app.cell(hide_code=True)
def check_unread(box_unread):
    result_unread = L.check("unread", box_unread.value)
    result_unread
    return (result_unread,)


@app.cell(hide_code=True)
def hints_unread():
    L.hints(
        "unread",
        [
            "Een deel als `\"5]. On Sunday …\"` begint met het getal. Knip het bij `]`, en houd wat ervoor staat.",
            "`part.split(\"]\")[0]` is de tekst van het getal; `int` maakt er een getal van. Test het daarna tegen beide lijsten met `not in`, verbonden met `and`.",
            "`number = int(part.split(\"]\")[0])`, dan `if number not in opened and number not in missing:` met `missing.append(number)` eronder.",
        ],
        why="""```python
def unread(answer, opened):
    missing = []
    for part in answer.split("[")[1:]:
        number = int(part.split("]")[0])
        if number not in opened and number not in missing:
            missing.append(number)
    return missing
```

Een bron die de agent nooit opende, is een bewering die hij niet controleerde. Je code ziet dat zonder de pagina's te lezen: hij vergelijkt alleen de nummers.""",
    )
    return


@app.cell(hide_code=True)
def text_pay():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Betalen binnen het mandaat

        Het mandaat is een dictionary: `mandate["shops"]` is de lijst van winkels waar de agent zelf mag betalen, en `mandate["max"]` is het meeste dat een bestelling mag kosten. `may_pay(cart, mandate)` krijgt de winkelwagen als een lijst van dictionaries, elk met `"product"`, `"shop"`, `"price"` en `"quantity"`. Hij geeft `"no"` terug als een product uit een winkel komt die niet in het mandaat staat, `"ask"` als het totaal boven het maximum ligt, zodat de gebruiker beslist, en `"yes"` als de agent zelf mag betalen. De eerste regels tellen het totaal op. De check probeert de winkelwagens uit het plaatje en nog één, met een mandaat dat beide winkels toestaat.
        """),
        L.task("Voeg in de loop een regel toe voor een winkel die niet in het mandaat staat, en na de loop de regels die met het totaal beslissen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pay():
    box_pay = L.code_box("may-pay", lines=11)
    box_pay
    return (box_pay,)


@app.cell(hide_code=True)
def check_pay(box_pay):
    result_pay = L.check("may-pay", box_pay.value)
    result_pay
    return (result_pay,)


@app.cell(hide_code=True)
def hints_pay():
    L.hints(
        "may-pay",
        [
            "Een winkel buiten het mandaat sluit de hele winkelwagen uit, wat hij ook kost, dus die controle komt eerst: in de loop, bij elk product. Het totaal ken je pas na de loop.",
            "In de loop, vóór de regel met `total`: `if item[\"shop\"] not in mandate[\"shops\"]:` met `return \"no\"`. Na de loop: `\"ask\"` als `total` boven `mandate[\"max\"]` ligt, anders `\"yes\"`.",
            "In de loop: `if item[\"shop\"] not in mandate[\"shops\"]:` en `return \"no\"`; erna: `if total > mandate[\"max\"]:` met `return \"ask\"`, en `return \"yes\"`.",
        ],
        why="""```python
def may_pay(cart, mandate):
    total = 0
    for item in cart:
        if item["shop"] not in mandate["shops"]:
            return "no"
        total = total + item["price"] * item["quantity"]
    if total > mandate["max"]:
        return "ask"
    return "yes"
```

Binnen het mandaat betaalt de agent zelf, zodat de gebruiker niet bij elke kleine bestelling gevraagd wordt; erboven beslist de gebruiker. Precies het maximum mag nog: `>` en niet `>=`.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Kies een taak, en Qwen3-0.6B doet hem met dezelfde loop, in de wereld van die taak: de course start zijn eigen server op je computer, met het model erachter. Je functies zitten erin: de tests van de coding agent maken hun verslag met je `test_report`, de winkelagent betaalt alleen als je `may_pay` `"yes"` zegt, en onder het antwoord van de research agent zegt je `unread` welke bronnen hij nooit opende. Een taak voor de coding agent heeft oefening 1 nodig, een taak voor de research agent oefening 2, en een taak voor de winkelagent oefening 3. De taken zelf zijn in het Engels, zoals het model ze krijgt.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de licentie Apache-2.0, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def task_cell():
    _kinds = {"coding": "Coding", "research": "Research", "personal": "Persoonlijk", "shopping": "Winkelen"}
    task_pick = mo.ui.dropdown(options={f"{_kinds[k]}: {t}": (k, i) for k in C.KINDS for i, t in enumerate(C.TASKS[k])},
                               value=f"Coding: {C.TASKS['coding'][0]}", label="Taak")
    run_button = mo.ui.run_button(label="Start de agent")
    mo.vstack([task_pick, run_button])
    return run_button, task_pick


@app.cell(hide_code=True)
def demo_run(box_pay, box_report, box_unread, run_button, task_pick):
    _kind, _task = task_pick.value
    _report = box_report.value if C.learner_fn("report", box_report.value) is not None else None
    _unread = C.learner_fn("unread", box_unread.value)
    _pay = C.learner_fn("may-pay", box_pay.value)
    _needs = {"coding": (_report, 1), "research": (_unread, 2), "shopping": (_pay, 3)}
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _kind in _needs and _needs[_kind][0] is None:
        _out = mo.md("*Maak eerst oefening {k} af: deze taak gebruikt je eigen functie.*".format(k=_needs[_kind][1]))
    elif not run_button.value:
        _out = mo.md("*Druk op de knop om de agent te starten.*")
    else:
        mo.output.replace(mo.md("*De agent is gestart; de server laadt Qwen3 …*"))
        _names = {"coding": "coding agent", "research": "research agent", "personal": "persoonlijke agent", "shopping": "winkelagent"}
        _words = {"title": "Qwen3 als {k}, met dezelfde loop".format(k=_names[_kind]), "step": "stap {k}: de call van het model",
                  "no_call": "het antwoord van het model, zonder call", "limit": "De agent bereikte de stapgrens van {s} requests.",
                  "unread": "Je unread: het antwoord noemt {m}, die de agent nooit opende.",
                  "all_read": "Je unread: de agent opende elke bron die het antwoord noemt.",
                  "stopped": "Je functie stopte met {e}.", "broke": "De run stopte met {e}.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.run_agent(_kind, _task, _words, report_code=_report, unread_fn=_unread, pay_fn=_pay))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop is die uit [De agent loop](/course/open/build-agent-loop?lang=nl); alleen de tools, de system prompt en de grens veranderden. De coding agent is die waarmee je werkte in [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl), de research agent doet wat [Zoeken, deep research en leren met AI](/course/open/build-research-and-learn?lang=nl) beschreef, en de persoonlijke agent is die uit [Werk automatiseren: flows, persoonlijke agents en process mining](/course/open/build-automate-work?lang=nl), die zijn eigen stappen kiest, dus jij controleert wat hij deed. De volgende les kijkt terug naar de agents van vóór deze, en waarom de meeste ervan ontspoorden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pay, result_report, result_unread):
    L.wrap_up(result_report, result_unread, result_pay)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
