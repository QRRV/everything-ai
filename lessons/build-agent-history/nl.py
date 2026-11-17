# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-history. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-history", "nl")
    C = L.checks   # de kleine BabyAGI van de course, zijn runs van Qwen3, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_same():
    quick_same = L.choice(
        "quick-same",
        question="Wat maakte de coding agent anders dan de winkelagent?",
        options={
            "loop": "Een andere loop: een coding agent draait zijn tests in een eigen loop, en een winkelagent betaalt aan het eind",
            "tools": "Zijn system prompt, zijn tools en zijn grens; de loop was dezelfde",
            "model": "Een ander model, een dat op code getraind is, omdat een chat model geen functie kan repareren",
        },
        correct="tools",
        why={
            "tools": "Eén loop, hetzelfde model en dezelfde stapgrens; alleen wat de agent kan doen, en wat daarbij mis kan gaan, verschilde.",
            "loop": "Allebei draaiden ze met dezelfde loop: sturen, de aanroep uitvoeren, het resultaat toevoegen, opnieuw. Zie [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl).",
            "model": "Alle vier de werelden draaiden met Qwen3-0.6B. Zie [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl).",
        },
    )
    quick_same
    return (quick_same,)


@app.cell(hide_code=True)
def quick_check_mandate():
    quick_mandate = L.choice(
        "quick-mandate",
        question="De system prompt van de winkelagent zegt al waar hij mag betalen, en hoeveel. Waarom vraagt `pay` dan toch `may_pay`?",
        options={
            "code": "Een regel in de prompt is een vraag aan het model; alleen de controle in de code die betaalt houdt stand",
            "speed": "Zodat het model het totaal van de winkelwagen niet zelf hoeft uit te rekenen, wat het een request scheelt",
            "user": "Zodat de gebruiker het mandaat vóór elke betaling nog een keer ziet",
        },
        correct="code",
        why={
            "code": "Het model kan toch een winkel buiten het mandaat kiezen; de code die `pay` uitvoert houdt het daar tegen.",
            "speed": "`may_pay` voegt geen request toe en vervangt er geen; het beslist of de betaling door mag gaan. Zie [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl).",
            "user": "Binnen het mandaat wordt de gebruiker helemaal niets gevraagd. Zie [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl).",
        },
    )
    quick_mandate
    return (quick_mandate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De loop met tools en grenzen kwam niet als eerste. In 2022 en 2023 probeerden verschillende ontwerpen een model meer te laten doen dan antwoorden, en in 2023 trokken er twee veel aandacht: AutoGPT en BabyAGI beloofden dat een model dat alleen een doel kreeg, daar zelf naartoe zou werken. Meestal draaiden ze rondjes of dwaalden ze af. Wie weet wat ze deden, begrijpt waarom agents nu een taak krijgen in plaats van een doel, en daarbij een stapgrens en iemand die controleert.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **autonomous agent** | een agent die zelf zijn taken bedenkt richting een doel en ze uitvoert, zonder dat er iemand tussen zit |
    | **MRKL** | een vroeg ontwerp (AI21 Labs, 2022), uitgesproken als *miracle*: een router stuurt elke vraag naar de juiste module, zoals een rekenmachine |
    | **HuggingGPT** | een ontwerp uit 2023 waarin een chat model de stappen van een taak plant en elke stap aan een ander model geeft |
    | **`.lstrip(chars)`, `.rstrip(chars)`** | halen elk teken dat in `chars` staat van het begin van een tekst af, of van het eind: `"2. Buy milk.".lstrip("0123456789. ")` is `"Buy milk."`, en `.rstrip(".")` maakt daar `"Buy milk"` van |
    | **`x not in items`** | `True` als `x` niet in de lijst `items` staat; `x in items` is het omgekeerde |
    | **`a and b`** | `True` alleen als `a` en `b` allebei `True` zijn |

    De tabel hieronder zet de ontwerpen op een rij. Elk liet iets na dat er nog steeds is, behalve de autonomous agents: wat er van hen overbleef, is vooral wat agents nu doen om hun fouten te voorkomen.
    """)
    return


@app.cell(hide_code=True)
def picture_history():
    mo.Html(C.history_picture("nl"))
    return


@app.cell(hide_code=True)
def text_babyagi():
    mo.md(r"""
    BabyAGI past in een paar regels. Het houdt een lijst met taken bij richting een doel. Elke ronde voert het model de eerste taak uit de lijst uit; een tweede request vraagt het, met het resultaat, welke nieuwe taken er nog nodig zijn, en die gaan achteraan de lijst; een derde zet de lijst op volgorde. Dan begint de volgende ronde. Niets erin zegt wanneer het doel bereikt is; het origineel ging door tot iemand het stopte. De course schreef een eigen kleine versie met de eerste twee requests, en een regel in de tweede die het model vraagt DONE te schrijven als het doel bereikt is. De run stopt als de lijst leeg is, of na de laatste ronde.
    """)
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course draaide zijn kleine BabyAGI met Qwen3-0.6B op {o} doelen, {r} rondes elk, bij temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. Elk doel ging twee keer: een keer met elke nieuwe taak achteraan de lijst, zoals BabyAGI deed, en een keer met de lijst van oefening 2, die een taak overslaat die al gedaan is of al in de lijst staat, en er hoogstens {l} vasthoudt.
    """.format(o=F["objectives"], r=F["rounds"], l=F["limit"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een agent die zelf zijn taken bedenkt, heeft een einde nodig dat niet aan het model is: een stapgrens, een lijst die niet onbeperkt kan groeien, en iemand die kijkt wat hij deed. Agents beginnen nu bij een taak die iemand gaf, en stoppen als die af is of als de grens bereikt is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een lijst met taken die een agent zelf schrijft

    1. **Begin bij een taak die iemand gaf,** niet alleen een doel.
    2. **Lees wat het model als taken schrijft:** één per regel, zonder nummers of streepjes, en herken wanneer het zegt dat het klaar is.
    3. **Sla een taak over die al gedaan is of al in de lijst staat,** zodat de agent hem niet twee keer doet.
    4. **Begrens de lijst en de rondes,** zodat de agent stopt, ook als het model taken blijft schrijven.
    5. **Laat iemand de taken en de resultaten zien,** voordat er iets gebeurt dat niet terug kan.
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
def text_parse():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De regels van het model als taken

        `parse_tasks(text)` maakt van wat het model schreef een lijst met taken. Elke regel is één taak. Haal de spaties van beide kanten van een regel af, daarna, met `.lstrip("-*0123456789.) ")`, de streepjes, sterretjes en nummers waarmee een lijst kan beginnen, en met `.rstrip(".")` een punt aan het eind. Sla een regel over die daarna leeg is, en de regels `DONE` en `Done`, waarmee het model zegt dat het klaar is. De check probeert drie gewone regels, een genummerde lijst, een lijst met streepjes en een lege regel, een losse `DONE`, de tekst uit het plaatje, en twee teksten waarin het model `DONE.` en `Done.` met een punt schreef.
        """),
        L.task("Vervang `...` door een regel die `line` opschoont, en een `if` die hem aan `tasks` toevoegt als hij niet leeg is, niet `DONE` en niet `Done`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parse():
    box_parse = L.code_box("parse", lines=7)
    box_parse
    return (box_parse,)


@app.cell(hide_code=True)
def check_parse(box_parse):
    result_parse = L.check("parse", box_parse.value)
    result_parse
    return (result_parse,)


@app.cell(hide_code=True)
def hints_parse():
    L.hints(
        "parse",
        [
            "Schoon de regel eerst op, en beslis dan of het een taak is. Elk van deze methoden geeft een nieuwe tekst terug, dus je kunt ze achter elkaar zetten.",
            "`line.strip()` haalt de spaties van beide kanten af; zet de `.lstrip(...)` en `.rstrip(...)` uit de oefening er direct achter, en sla het resultaat weer op in `line`. `x not in [...]` klopt als `x` geen van de teksten in de lijst is.",
            "Na de regel die `line` opschoont: `if line not in ___:`, met `tasks.append(line)` eronder.",
        ],
        why="""```python
def parse_tasks(text):
    tasks = []
    for line in text.split("\\n"):
        line = line.strip().lstrip("-*0123456789.) ").rstrip(".")
        if line not in ["", "DONE", "Done"]:
            tasks.append(line)
    return tasks
```

Een model schrijft een lijst op zijn eigen manier, met nummers, streepjes of geen van beide; de code die hem leest moet met elk overweg kunnen.""",
    )
    return


@app.cell(hide_code=True)
def text_add():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een lijst die binnen de perken blijft

        `add_tasks(queue, new, done, limit)` voegt de nieuwe taken toe aan de lijst `queue` en geeft de lijst terug. Sla een taak over die al in `queue` staat, en een die in `done` staat, de taken die de agent deed. Voeg alleen toe zolang de lijst minder dan `limit` taken bevat. De check probeert het geval uit het plaatje, een taak die al in de lijst staat, een lijst die vol raakt, twee keer dezelfde nieuwe taak, en een lijst met plek voor twee.
        """),
        L.task("Vervang `...` door een `if` die `task` alleen aan `queue` toevoegt als de lijst plek heeft, en de taak niet in `queue` en niet in `done` staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_add():
    box_add = L.code_box("add", lines=6)
    box_add
    return (box_add,)


@app.cell(hide_code=True)
def check_add(box_add):
    result_add = L.check("add", box_add.value)
    result_add
    return (result_add,)


@app.cell(hide_code=True)
def hints_add():
    L.hints(
        "add",
        [
            "Drie dingen moeten allemaal kloppen voordat een taak erin gaat: er is plek, hij staat nog niet in de lijst, en hij is niet gedaan. Verbind ze met `and`.",
            "Plek betekent dat de lijst nog korter is dan `limit`; `len(queue)` geeft de lengte. De andere twee zijn `task not in queue` en `task not in done`.",
            "`if ___ and task not in queue and task not in done:`, met `queue.append(task)` eronder.",
        ],
        why="""```python
def add_tasks(queue, new, done, limit):
    for task in new:
        if len(queue) < limit and task not in queue and task not in done:
            queue.append(task)
    return queue
```

De test op `queue` vangt ook twee keer dezelfde taak in `new`: de eerste staat al in de lijst als de tweede komt.""",
    )
    return


@app.cell(hide_code=True)
def text_stayed():
    mo.md(r"""
    ### Oefening 3. Wat bleef
    """)
    return


@app.cell(hide_code=True)
def choice_stayed_cell():
    choice_stayed = L.choice(
        "stayed",
        question="Agents van nu hielden een deel van AutoGPT en BabyAGI, en vervingen een ander deel. Welk is welk?",
        options={
            "kept-loop": "Ze hielden de loop van een model dat handelt en het resultaat leest; het doel zonder einde werd een taak van iemand, met een stapgrens",
            "kept-list": "Ze hielden de lijst met taken die het model voor zichzelf schrijft, zonder einde; de loop met tools viel weg, omdat die rondjes draaide",
            "kept-all": "Ze hielden allebei, en maakten alleen het model groter, omdat een groter model niet meer afdwaalt",
        },
        correct="kept-loop",
        why={
            "kept-loop": "De loop werkt; wat misging was dat niets buiten het model besliste wanneer het moest stoppen of wat mocht.",
            "kept-list": "Het was andersom: de loop met tools bleef, in elke agent van deze module; de lijst zonder einde niet.",
            "kept-all": "Een groter model maakt minder fouten, maar een agent heeft nog steeds een einde nodig dat niet aan het model is.",
        },
    )
    choice_stayed
    return (choice_stayed,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Geef de kleine BabyAGI van de course een eigen doel, en hij draait {r} rondes op Qwen3-0.6B, met jouw `parse_tasks` die de taken van het model leest en jouw `add_tasks` die de lijst binnen de perken houdt. De course start een eigen server op je computer, met het model erachter.
    """.format(r=C.DEMO_ROUNDS))
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
def objective_cell():
    objective_box = mo.ui.text(value=C.OBJECTIVES[1], label="Doel", full_width=True)
    start_button = mo.ui.run_button(label="Start")
    mo.vstack([objective_box, start_button])
    return objective_box, start_button


@app.cell(hide_code=True)
def demo_run(box_add, box_parse, objective_box, start_button):
    _parse = C.learner_fn("parse", box_parse.value)
    _add = C.learner_fn("add", box_add.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _parse is None or _add is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo leest en bewaart de taken met je eigen functies.*")
    elif not start_button.value:
        _out = mo.md("*Druk op de knop om te starten.*")
    elif not objective_box.value.strip():
        _out = mo.md("*Schrijf eerst een doel in het vak.*")
    else:
        mo.output.replace(mo.md("*De agent is gestart; de server laadt Qwen3 …*"))
        _words = {"title": "Jouw kleine BabyAGI, ronde voor ronde", "first": "de eerste taken", "round": "ronde {k}: het model deed",
                  "new": "nieuwe taken die het schreef", "left": "aan het eind over in de lijst ({n})", "empty": "(geen)",
                  "stopped": "Het stopte met {e}.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.baby_agi(objective_box.value.strip(), _parse, _add, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** ReAct is de loop van [De agent loop](/course/open/build-agent-loop?lang=nl), de modules van MRKL zijn de tools van deze hele module, en de autonomous agents verklaren de grenzen uit [Coding-, research- en persoonlijke agents](/course/open/build-kinds-of-agents?lang=nl): een taak van iemand, een stapgrens, en een controle in de code. De volgende les kijkt naar agent frameworks, die je de loop, de tools en de rest kant-en-klaar geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stayed, result_add, result_parse):
    L.wrap_up(result_parse, result_add, choice_stayed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
