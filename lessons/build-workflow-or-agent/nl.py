# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-workflow-or-agent. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-workflow-or-agent", "nl")
    C = L.checks   # de keuken, de runs van Qwen3 als agent en als workflow, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_results():
    quick_results = L.choice(
        "quick-results",
        question="Het antwoord van Qwen3 bevat een tool call van `convert`. Wat zet je programma in het volgende request, na de berichten tot nu toe?",
        options={
            "user": "Alleen het resultaat, als nieuw bericht van de gebruiker, zodat het request kort en goedkoop blijft",
            "result": "Het antwoord zelf, en een bericht met de rol `tool` met het resultaat en het `id` van de call",
            "nothing": "Niets nieuws: de server van de course houdt de call vast, voert `convert` zelf uit en stuurt het later",
        },
        correct="result",
        why={
            "result": "Aan het id leest het model bij welke call elk resultaat hoort, en daar gaat het mee verder.",
            "user": "Een resultaat gaat terug met de rol `tool` en het id van zijn call, direct na het antwoord dat de call deed. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
            "nothing": "Je programma voert `convert` zelf uit, en stuurt het resultaat terug. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
        },
    )
    quick_results
    return (quick_results,)


@app.cell(hide_code=True)
def quick_check_auto():
    quick_auto = L.choice(
        "quick-auto",
        question="Wat laat `\"tool_choice\": \"auto\"` aan het model over?",
        options={
            "tools": "Welke tools in het request gaan, van alle tools die je programma heeft",
            "schema": "Welke argumenten de functie `convert` aanneemt, en van welke types",
            "decide": "Of het überhaupt een tool aanroept, en zo ja, welke en waarmee",
        },
        correct="decide",
        why={
            "decide": "Met `\"auto\"` beslist het model; met `convert` bij naam beslist je programma.",
            "tools": "Je programma zet de tools in het request; `\"auto\"` gaat over wat het model ermee doet. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
            "schema": "Dat zegt het schema in de beschrijving van de tool, en dat schrijft je programma. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
        },
    )
    quick_auto
    return (quick_auto,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Wat een agent is](/course/open/agent-idea?lang=nl) kreeg Qwen3 twee tools en een loop voor een eenvoudige vraag: wat moet ik kopen voor `pancakes`? {why_runs} Het model koos de stappen, en koos ze slecht. Toch zijn de stappen van die taak altijd dezelfde: zoek het recept op, kijk voor elk ingrediënt in de voorraadkast, en reken uit wat er ontbreekt. Die stappen kan je code zelf zetten, elke keer op dezelfde manier, en alleen wat taal nodig heeft aan het model laten. Dat is een workflow. Vraag je af of je een agent nodig hebt, voordat je er een bouwt.
    """.format(why_runs=C.agent_idea_line("nl")))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **workflow** | een programma waarin je code de stappen en hun volgorde vastlegt, en het model er maar een paar invult, zoals de vraag lezen of het antwoord schrijven; ook *LLM workflow* genoemd |
    | **`text.lower()`** | de tekst in kleine letters: `"Pancakes".lower()` is `"pancakes"` |

    In een agent kiest het model zelf de volgende stap, in een loop, tot het antwoordt. In een workflow zet je code de stappen in een vaste volgorde. Anthropic trekt de grens zo in *Building effective agents* (december 2024): een workflow volgt paden die code vooraf vastlegt; een agent bepaalt zijn eigen stappen.

    De keuken van deze les heeft {dishes} recepten en een voorraadkast. `recipe` geeft de ingrediënten van een gerecht met hun grammen, `pantry` hoeveel gram van een ingrediënt er is:

    ```python
    recipe("pancakes")   # {{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}}
    pantry("milk")       # 200
    pantry("eggs")       # 0
    ```

    De twee manieren hieronder krijgen dezelfde vraag en dezelfde twee opzoekfuncties. De agent krijgt ze als tools; de code van de workflow roept ze zelf aan.
    """.format(dishes=F["dishes"]))
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course stelde Qwen3-0.6B {n} vragen, twee voor elk gerecht, op beide manieren, via de server van de course met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. De agent kreeg de system prompt van [Wat een agent is](/course/open/agent-idea?lang=nl), die nu ook om de grammen van elk ingrediënt vraagt, de twee tools, en een limiet van {steps} requests. De workflow vroeg het model twee keer iets: over welk gerecht de vraag gaat, en om één zin te schrijven met de lijst die de code had uitgerekend.
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
        L.watch_out("Een workflow beantwoordt alleen de vragen waarvoor hij gebouwd is. Vraag hem *Wat kan ik maken met wat er in mijn voorraadkast staat?*, en zijn eerste stap heeft niets te kiezen: er staat geen gerecht in de vraag. Daarom controleert de workflow het gerecht van het model voordat hij verdergaat. Een agent kan nog andere stappen proberen; een workflow hoort duidelijk te zeggen wat hij niet kan."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een workflow of een agent?

    1. **Schrijf de stappen op.** Zijn ze hetzelfde voor elke vraag die je app krijgt, bouw dan een workflow.
    2. **Geef code wat code precies doet:** opzoeken, rekenen, controleren.
    3. **Geef het model wat taal nodig heeft:** de vraag lezen, het antwoord schrijven. Controleer wat het invult voordat je code verdergaat.
    4. **Kies alleen een agent waar de volgende stap afhangt van wat de vorige vond,** op manieren die je vooraf niet kunt opsommen, zoals uitzoeken waarom een programma faalt.
    5. **Meet beide op je eigen vragen:** goede antwoorden, requests, tokens en seconden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def text_to_buy():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De eigen stap van de workflow

        `to_buy(dish)` is de stap van de workflow waar geen model aan te pas komt. `recipe(dish)` geeft het recept als dictionary, zoals `{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}`, en `pantry(item)` hoeveel gram van een ingrediënt er in de voorraadkast is, zoals `200` voor melk. Maak de loop af, zodat `buy` elk ingrediënt bevat waarvan de voorraadkast te weinig heeft, met de grammen die ontbreken. De check probeert alle zes gerechten; het plaatje eronder laat je lijst voor elk zien.
        """),
        L.task("Zoek in de loop op hoeveel gram van `item` de voorraadkast heeft; is dat minder dan `need[item]`, zet `item` dan in `buy` met de grammen die ontbreken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_to_buy():
    box_to_buy = L.code_box("to-buy", lines=8)
    box_to_buy
    return (box_to_buy,)


@app.cell(hide_code=True)
def check_to_buy(box_to_buy):
    result_to_buy = L.check("to-buy", box_to_buy.value)
    result_to_buy
    return (result_to_buy,)


@app.cell(hide_code=True)
def hints_to_buy():
    L.hints(
        "to-buy",
        [
            "Twee hoeveelheden naast elkaar: `need[item]` is wat het recept wil, `pantry(item)` wat er is. Van een ingrediënt is er te weinig als er minder is dan het recept nodig heeft.",
            "Bewaar wat de voorraadkast heeft onder een naam, zoals `have`, en vergelijk het in een `if` met `need[item]`.",
            "Onder `if have < need[item]:` komt `buy[item] = ___`: de grammen die ontbreken.",
        ],
        why="""```python
def to_buy(dish):
    need = recipe(dish)
    buy = {}
    for item in need:
        have = pantry(item)
        if have < need[item]:
            buy[item] = need[item] - have
    return buy
```

Voor `pancakes` blijven `flour` (1000 ≥ 250) en `butter` (250 ≥ 25) van de lijst af; `milk` geeft 450 − 200 = 250, en `eggs` 110 − 0 = 110. `<` en niet `<=`: voor `tomato soup` heeft de voorraadkast precies de 150 gram `onion` die het recept nodig heeft, dus koop je er niets van.""",
    )
    return


@app.cell(hide_code=True)
def text_shopping():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De hele workflow

        `shopping(url, question)` is de workflow uit het plaatje hierboven. De eerste regel vraagt het model over welk gerecht de vraag gaat, met de prompt `PICK`, en ruimt het antwoord op met `.strip().lower()`. Schrijf de rest: geef `SORRY` terug als het gerecht niet een van `DISHES` is; reken anders uit wat er gekocht moet worden met `to_buy`, vraag het model het antwoord te schrijven met de prompt `WRITE`, en geef terug wat het schrijft. `WRITE` heeft twee plekken om in te vullen: `{dish}`, en `{items}` voor de lijst als tekst, die `as_text(buy)` geeft, zoals `"milk 250 g, eggs 110 g"`. Het vak kent `PICK`, `WRITE`, `DISHES`, `SORRY`, `as_text` en de `to_buy` van de course. De check roept je workflow aan tegen een stand-in van de server, met een vraag over `pancakes`, een over `rice pudding`, en een over `pizza`, die de keuken niet heeft.
        """),
        L.task("Geef na de eerste regel `SORRY` terug als het gerecht niet in `DISHES` staat; reken anders de lijst uit met `to_buy`, en geef het antwoord van het model op `WRITE` terug, ingevuld met het gerecht en de lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shopping():
    box_shopping = L.code_box("shopping", lines=22)
    box_shopping
    return (box_shopping,)


@app.cell(hide_code=True)
def check_shopping(box_shopping):
    result_shopping = L.check("shopping", box_shopping.value)
    result_shopping
    return (result_shopping,)


@app.cell(hide_code=True)
def hints_shopping():
    L.hints(
        "shopping",
        [
            "Het gerecht van het model is tekst van buiten je code: controleer het voordat je het gebruikt. `dish not in DISHES` is `True` als het model iets koos dat de keuken niet kent.",
            "`buy = to_buy(dish)` geeft de lijst. Vul `WRITE` in zoals de eerste regel `PICK` invult, en stuur hem met `ask(url, ...)`.",
            "De laatste regel: `return ask(url, ___)`, met `WRITE` ingevuld met het gerecht en de lijst.",
        ],
        why="""```python
def shopping(url, question):
    dish = ask(url, PICK.format(question=question)).strip().lower()
    if dish not in DISHES:
        return SORRY
    buy = to_buy(dish)
    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))
```

Twee requests, elke keer, in dezelfde volgorde: daardoor is een workflow goedkoop om te draaien en makkelijk te testen. De controle op `dish` is een poort: het model vult een stap in, en je code beslist of de volgende stap door mag.""",
    )
    return


@app.cell(hide_code=True)
def text_agent_task():
    mo.md(r"""
    ### Oefening 3. Waar een agent past
    """)
    return


@app.cell(hide_code=True)
def choice_agent_task_cell():
    choice_agent_task = L.choice(
        "agent-task",
        question="Welke van deze taken heeft een agent nodig, en geen workflow?",
        options={
            "invoice": "Elke binnenkomende factuur lezen, en de datum, het bedrag en de leverancier in een tabel zetten",
            "translate": "De beschrijving van elk nieuw product vertalen naar het Nederlands en het Frans",
            "tests": "Uitzoeken waarom de tests van een programma falen, en de code repareren",
        },
        correct="tests",
        why={
            "tests": "Wat er daarna moet gebeuren, hangt af van wat de vorige stap vond: een foutmelding, een bestand, een test die nog steeds faalt. Die stappen zijn vooraf niet op te sommen.",
            "invoice": "Elke factuur doorloopt dezelfde stappen: drie velden lezen, ze controleren, een rij toevoegen. Je code kan die volgorde vastleggen; het model leest alleen.",
            "translate": "Elke beschrijving doorloopt dezelfde stap, twee keer. Dat doet een workflow, en die is goedkoper en makkelijker te testen dan een agent.",
        },
    )
    choice_agent_task
    return (choice_agent_task,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Is oefening 2 goed, dan kun je je workflow op het echte model proberen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. Vraag naar een van de zes gerechten in je eigen woorden, of naar een gerecht dat de keuken niet heeft. Het model leest Engels het best.
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
    question_box = mo.ui.text(value=C.QUESTIONS[3][0], label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Vraag het je workflow")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_shopping(box_shopping, question_box, send_button):
    _shopping = C.learner_shopping(box_shopping.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _shopping is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo vraagt met je eigen workflow.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat je workflow met Qwen3 deed", "request": "Request {n}: je workflow vroeg",
                  "answer": "De server antwoordde:", "code": "Daarna je code: gerecht {dish!r}, te kopen {buy}",
                  "gave": "Je workflow gaf terug: {v}", "error": "Je workflow stopte met {e}.",
                  "none": "Je workflow stuurde niets naar de server.", "status": "De server antwoordde met status {s}: {m}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_shopping, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop van [Wat een agent is: een model, tools en een loop](/course/open/agent-idea?lang=nl) en de calls van [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl) laten een model zijn eigen stappen kiezen. Hier koos je code ze, en vulde het model er twee in. In de volgende les, [Vijf workflowpatronen](/course/open/build-workflow-patterns?lang=nl), zie je de vormen die de meeste workflows hebben: stappen na elkaar, een router, delen naast elkaar, een orchestrator met workers, en een evaluator die een antwoord terugstuurt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agent_task, result_shopping, result_to_buy):
    L.wrap_up(result_to_buy, result_shopping, choice_agent_task)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
