# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-failures. Route: course/route/lessen/bouwen.yaml. Controles, de twintig runs, wat de course in elke run vond en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-failures", "nl")
    C = L.checks   # de twintig opgenomen runs, de manieren die de course in elke run vond, en de plaatjes
    RUNS = C.RUNS  # elke run: "name", "task", "calls", "results", "answer", "steps", "limit"; een winkelrun ook "wanted" en "added"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Waar komen bij error analysis de soorten fouten vandaan?",
        options={
            "list": "Uit een lijst die vooraf is gemaakt",
            "model": "Uit wat het model zegt dat er misging",
            "notes": "Uit notities bij de antwoorden",
        },
        correct="notes",
        why={
            "notes": "Je leest ongeveer honderd antwoorden, schrijft bij elk in een paar woorden op wat er misgaat, en deelt die notities in soorten in. De soorten komen uit je eigen antwoorden.",
            "list": "Een lijst die vooraf is gemaakt, vindt wat je verwachtte en mist de rest. De soorten komen uit notities bij de antwoorden zelf. Zie [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl).",
            "model": "Het model weet niet waar het de fout in ging, en wat het daarover zegt, is tekst zoals alle andere. De soorten komen uit je notities bij de antwoorden. Zie [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    Een agent zet veel stappen voordat hij antwoordt, en één verkeerde stap kan de rest bederven, terwijl het antwoord nog steeds zeker klinkt. In [De agent loop](/course/open/build-agent-loop?lang=nl) beantwoordde Qwen3-0.6B {right} van de {k} vragen over de keuken goed. Om dat te verbeteren, is een telling van goed en fout niet genoeg: je moet weten *hoe* elke run de fout in ging. Een run die rondjes draait, vraagt om een andere oplossing dan een run die verzint wat er in de voorraadkast ligt.

    De manieren waarop agents ontsporen, komen bij agent na agent terug. Deze les leest {n} opgenomen runs van de agent van de course en telt vijf van die manieren. {k} daarvan zijn de keukenruns van de agent loop. De andere {s} spelen in een webwinkel: de agent kreeg een taak zoals *Put eggs and butter in the cart.* en een pagina met knoppen, en klikte op knoppen via hun naam tot hij zei dat hij klaar was.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe collega die boodschappen doet, kan de fout in gaan op manieren die je herkent: twee keer dezelfde leverancier bellen, bij de verkeerde winkel kopen, zeggen dat de levering binnen is terwijl dat niet zo is, halverwege het lijstje terugkomen, of dingen kopen waar niemand om vroeg. Voor elk is er een eigen oplossing: een lijstje om af te vinken, een bonnetje, een blik in de tas bij de deur.
    """.format(right=sum(r.startswith("kitchen") for r in _f["right"]), k=_f["kitchen"], n=_f["runs"], s=_f["shop"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **agent failure mode** | een manier van ontsporen die bij agents en taken steeds terugkomt, zoals in een loop blijven hangen of een resultaat verzinnen |

    Vijf daarvan komen bij bijna elke agent terug, en elk is te zien in de trace van een run: de calls, hun resultaten en het antwoord.

    | Manier | Wat de run laat zien | Wat helpt |
    |---|---|---|
    | **stopt te vroeg** | hij antwoordt voordat hij heeft opgezocht wat het antwoord nodig heeft | code die nagaat of de nodige calls er waren voordat het antwoord wordt aangenomen, en het model anders terugstuurt met wat ontbreekt |
    | **doet meer dan gevraagd** | hij doet iets waar de taak niet om vroeg | code die elke actie naast de taak legt voordat die draait, en een approval step voor wat niet terug te draaien is |
    | **blijft in een loop hangen** | hij doet een call die hij al eerder deed, of gaat door tot de stappenlimiet | een stappenlimiet, en code die een eerder gedane call weigert en dat zegt |
    | **verzint een resultaat** | zijn antwoord geeft een resultaat dat geen tool gaf, zoals wat er in de voorraadkast ligt of in de winkelwagen zit | code die de getallen zelf in het antwoord zet, of een controle die het antwoord naast de resultaten leest |
    | **gebruikt een tool verkeerd** | hij roept een tool aan die niet past, of geeft hem een argument dat de tool niet kent | argumenten met een vaste lijst keuzes, en een tool die zegt wat hij niet kent in plaats van toch te antwoorden |

    Een run die antwoordt zonder een resultaat dat hij nodig heeft, zoals een boodschappenlijst zonder in de voorraadkast te kijken, stopt te vroeg. Hij verzint een resultaat als hij zegt wat een tool gezegd zou hebben.

    Code kan vier van de vijf vinden als bekend is wat de taak nodig heeft en wat de tools kennen: een call die al eerder gedaan werd, een product waar niemand om vroeg, een nodige call die nooit kwam en een argument dat de tool niet kent, zijn allemaal dingen die code kan vergelijken. Een verzonnen resultaat is lastiger: het zet vaak ware getallen in een zin die niet klopt. Om het te vinden moet iemand lezen: jij, of een judge zoals in [Wie het antwoord beoordeelt](/course/open/build-judges?lang=nl).

    De vijf zijn geen lijstje om af te vinken in plaats van te lezen. Je eigen error analysis, zoals in [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl), zegt welke je agent heeft, en vindt misschien andere.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: vinden hoe je agent ontspoort

    1. **Bewaar elke run:** de calls met hun argumenten, hun resultaten en het antwoord, als trace.
    2. **Lees ongeveer twintig runs** en schrijf bij elke run op hoe hij ontspoorde; één run kan op meerdere manieren ontsporen.
    3. **Schrijf regels in code** voor wat code kan zien: een call die al eerder gedaan werd, een run die de stappenlimiet haalde, een argument dat de tool niet kent, een actie waar de taak niet om vroeg, een nodige call die nooit kwam.
    4. **Tel per manier,** en pak de grootste aan met een controle in code waar dat kan; lees de rest, of laat een judge die lezen.
    5. **Meet opnieuw** op dezelfde runs na elke wijziging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _run = RUNS[3]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `RUNS` bevat de {n} runs. Elke run is een dictionary met zijn `"name"`, de `"agent"` (`"kitchen"` of `"shop"`), de `"task"`, zijn `"calls"` op volgorde, elk een paar van de naam van de tool en zijn argumenten, de `"results"`, het `"answer"` (`None` als de stappenlimiet de run stopte), het aantal `"steps"` en de `"limit"`. Een keukenrun heeft ook het `"dish"`; een winkelrun heeft ook `"wanted"`, de producten waar de taak om vroeg, en `"added"`, wat elke klik in de winkelwagen deed.

    `RUNS[3]` is keuken 4: *{task}* De calls zijn `{calls}`, en elk resultaat was *0 grams*. Deze run ontspoort op drie manieren tegelijk. In zijn eerste stap vroeg hij twee keer naar de tomaat: een call die al eerder gedaan was, dus hij blijft in een loop hangen. Hij vroeg de voorraadkast om `"tomato"`, die de voorraadkast kent als `"tomatoes"`, en om `"soup"`, wat een gerecht is: hij gebruikt een tool verkeerd. En hij antwoordde zonder het recept op te zoeken: hij stopt te vroeg. Zijn antwoord zegt dat de voorraadkast niets heeft voor tomatensoep, terwijl er {tomatoes} gram tomaten in ligt. De voorraadkast maakte het erger: die antwoordt *0 grams* bij alles wat hij niet kent.
    """.format(n=len(RUNS), task=_run["task"], calls=C.code_of(_run["calls"]), tomatoes=C.W.PANTRY["tomatoes"]))
    return


@app.cell(hide_code=True)
def picture_worked_example():
    mo.Html(C.trace_picture("kitchen 4", L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _c = _f["count"]
    _text = mo.md(r"""
    ### Wat de course telde

    De course las elk van de {n} runs en schreef met de hand op hoe hij ontspoorde; het raster hierboven laat het resultaat zien. {early} runs stopten te vroeg: alle {k} in de keuken, want geen enkele keek voor elk ingrediënt in de voorraadkast, en één in de winkel. {after} keukenruns antwoordden na alleen het recept, {none_called} riepen geen enkele tool aan en schreven dat ze het recept nog nodig hadden, en de andere {rest} zochten maar een deel op van wat ze nodig hadden. {more} winkelruns deden iets in de winkelwagen waar de taak niet om vroeg, {loop} runs bleven in een loop hangen, {made} verzonnen een resultaat en {wrong} gebruikten een tool verkeerd. {fine} runs gingen op elk punt goed, allemaal in de winkel.

    Daarna schreef de course voor elke manier behalve verzinnen een regel in code, en liet die regels los op dezelfde runs. Voor die vier manieren vonden de regels precies wat het lezen vond. Voor verzinnen probeerde de course een regel die zoekt naar getallen in het antwoord die geen tool en geen taak gaf: die vond er geen in de {made} verzonnen antwoorden. Twee keukenruns schreven de grammen van het recept op als wat er in de voorraadkast ligt: ware getallen in een zin die niet klopt. Winkel 6 schreef dat hij de pijnboompitten had toegevoegd, zonder één getal.
    """.format(n=_f["runs"], early=_c["too-early"], k=_f["kitchen"], after=len(_f["after_recipe"]), none_called=len(_f["no_calls"]),
               rest=_f["kitchen"] - len(_f["after_recipe"]) - len(_f["no_calls"]),
               more=_c["more"], loop=_c["loop"], made=_c["made-up"], wrong=_c["wrong-tool"], fine=len(_f["none"])))
    mo.vstack([
        _text,
        L.watch_out("Eén run kan op meerdere manieren tegelijk ontsporen: winkel 6 klikte twee keer op de pasta, deed er rijst bij, voegde de pijnboompitten nooit toe en zei dat hij dat wel had gedaan. Schrijf elke manier op, niet alleen de eerste die je ziet."),
    ])
    return


@app.cell(hide_code=True)
def text_repeats():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een call die al eerder gedaan werd

        `repeats(calls)` moet de calls van een run geven die in dezelfde run al eerder gedaan werden: elke call die hetzelfde is als een eerdere, naam en argumenten allebei, op volgorde. Een call is een paar, zoals `("pantry", {"item": "tomato"})`, en `==` vergelijkt twee paren op beide delen. `RUNS[3]["calls"]` heeft `("pantry", {"item": "tomato"})` twee keer, dus `repeats(RUNS[3]["calls"])` moet die call één keer geven. De begincode geeft een lege lijst. Het plaatje laat je functie los op alle twintig runs.
        """),
        L.task("Zet elke call die al eerder gedaan was in `again`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_repeats():
    box_repeats = L.code_box("repeats", lines=8)
    box_repeats
    return (box_repeats,)


@app.cell(hide_code=True)
def check_repeats(box_repeats):
    result_repeats = L.check("repeats", box_repeats.value)
    result_repeats
    return (result_repeats,)


@app.cell(hide_code=True)
def hints_repeats():
    L.hints(
        "repeats",
        [
            "Houd een lijst bij van de calls die je tot nu toe zag: een call die er al in staat, is al eerder gedaan.",
            "De volgorde telt: vraag het voordat je de call aan `seen` toevoegt; andersom staat elke call er al in.",
            "Boven `seen.append(call)`: `if ___:`, met `again.append(call)` eronder.",
        ],
        why="""```python
def repeats(calls):
    seen = []
    again = []
    for call in calls:
        if call in seen:
            again.append(call)
        seen.append(call)
    return again
```

Een call die al eerder gedaan werd, is één teken van een loop. Het andere is een run die doorgaat tot de stappenlimiet: winkel 5 klikte op zes verschillende producten en deed nooit een call opnieuw.""",
    )
    return


@app.cell(hide_code=True)
def text_compare():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat de winkelagent wegliet, en wat hij toevoegde

        `compare(added, wanted)` moet vergelijken wat een run in de winkelwagen deed met wat de taak vroeg. Geef twee lijsten terug: `missing`, wat de taak vroeg en niet in `added` zit, in de volgorde van `wanted`; en `extra`, wat in `added` zit en de taak niet vroeg, in de volgorde van `added`. `x not in lijst` is `True` als `x` niet in de lijst zit. De begincode geeft twee lege lijsten. Het plaatje laat je functie los op de acht winkelruns.
        """),
        L.task("Vul `missing` vanuit `wanted`, en `extra` vanuit `added`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=10)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "Twee vragen: wat vroeg de taak dat niet in de winkelwagen zit, en wat zit in de winkelwagen dat de taak niet vroeg?",
            "Eén loop door `wanted` voor `missing`, en één loop door `added` voor `extra`.",
            "De eerste loop: `for product in wanted:`, dan `if ___:`, dan `missing.append(product)`; de tweede andersom.",
        ],
        why="""```python
def compare(added, wanted):
    missing = []
    extra = []
    for product in wanted:
        if product not in added:
            missing.append(product)
    for product in added:
        if product not in wanted:
            extra.append(product)
    return missing, extra
```

Iets wat ontbreekt, betekent dat de run te vroeg stopte; iets extra's betekent dat hij meer deed dan gevraagd. Winkel 6 heeft allebei, en zijn antwoord zei dat hij de pijnboompitten had toegevoegd: vergelijk de winkelwagen met de taak, niet met wat het model zegt.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Oefening 3. Welke manier?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-mode",
        question="Een keukenrun zocht het recept van een omelet op, riep `pantry` nooit aan, en antwoordde: *{said}* Op welke manier ontspoort hij met die zin?".format(
            said=C.pantry_sentence("kitchen 5")),
        options={
            "loop": "Hij blijft in een loop hangen",
            "made": "Hij verzint een resultaat",
            "more": "Hij doet meer dan gevraagd",
        },
        correct="made",
        why={
            "made": "Geen tool zei wat er in de voorraadkast ligt: de getallen zijn de grammen van het recept, opgeschreven alsof ze in de voorraadkast liggen. Een regel die zoekt naar getallen die geen tool gaf, mist dit, want het recept gaf ze; om het te vinden moet je lezen.",
            "loop": "Hij deed één call, naar het recept, en antwoordde. Niets kwam twee keer, en de stappenlimiet was ver weg. De zin zegt wat er in de voorraadkast ligt, en dat vertelde geen tool hem: hij verzint een resultaat.",
            "more": "Hij deed minder, niet meer: hij keek nooit in de voorraadkast. De zin zegt wat er in de voorraadkast ligt zonder dat een tool dat zei: hij verzint een resultaat.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_helps():
    mo.md(r"""
    ### Oefening 4. Wat helpt?
    """)
    return


@app.cell(hide_code=True)
def choice_helps_cell():
    choice_helps = L.choice(
        "what-helps",
        question="Winkel 2 moest eieren en boter in de winkelwagen doen, en deed er ook melk en kaas in. Wat helpt daar het best tegen?",
        options={
            "check": "Code legt elke klik naast de taak",
            "prompt": "Een regel in de prompt: alleen wat gevraagd is",
            "limit": "Een lagere stappenlimiet",
        },
        correct="check",
        why={
            "check": "De taak noemt de producten, dus code kan een klik weigeren die iets anders toevoegt, elke keer. Waar een taak minder duidelijk is, vraag je de gebruiker vóór de actie, zoals in [Eerst vragen, dan doen](/course/open/build-human-in-loop?lang=nl).",
            "prompt": "De system prompt zei de agent al te stoppen als de winkelwagen bevat wat gevraagd is, en winkel 2 ging door. Een regel in een prompt is een vraag aan het model; een controle in code houdt elke keer stand: weiger een klik die een product toevoegt dat de taak niet noemde.",
            "limit": "Winkel 2 deed de melk erbij met zijn allereerste klik, dus een lagere limiet houdt niets tegen wat ertoe doet. Een controle in code vóór elke klik kan een product weigeren dat de taak niet noemde.",
        },
    )
    choice_helps
    return (choice_helps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De manieren komen uit het lezen van runs, zoals de soorten van [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl) uit het lezen van antwoorden kwamen; de keukenruns zijn die van [De agent loop](/course/open/build-agent-loop?lang=nl). Waar code niet kan beslissen of een actie gewenst is, helpt de approval step van [Eerst vragen, dan doen](/course/open/build-human-in-loop?lang=nl) tegen meer doen dan gevraagd. [Prompt, RAG of finetunen?](/course/open/build-prompt-rag-finetune?lang=nl) vraagt welke wijziging je maakt als het model zelf tekortschiet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_helps, choice_which, result_compare, result_repeats):
    L.wrap_up(result_repeats, result_compare, choice_which, choice_helps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
