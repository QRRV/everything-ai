# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-code-sandbox. Route: course/route/lessen/bouwen.yaml. Checks, de sandbox van de course, de tabel en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-code-sandbox", "nl")
    C = L.checks   # de sandbox en de bewaker van de course, acht stukken code van een agent gedraaid op deze computer, de checks
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [De agent loop](/course/open/build-agent-loop?lang=nl), één over [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In de agent-loop bevat het antwoord van het model twee tool calls. Wat doet de loop daarna?",
        options={
            "user": "Hij laat beide calls aan de gebruiker zien, die ze uitvoert en intypt wat elke call teruggaf",
            "run": "Hij voert beide calls uit, voegt elk resultaat toe met het id van zijn call, en vraagt het model opnieuw",
            "first": "Hij voert alleen de eerste call uit, omdat het model per antwoord maar één stap kan zetten",
        },
        correct="run",
        why={
            "run": "Elk resultaat gaat terug met de rol `tool` en het id van zijn call, en het volgende request bevat het hele gesprek tot nu toe.",
            "user": "De loop voert de tools zelf uit: dat maakt hem een agent. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "first": "Een antwoord kan meerdere calls bevatten, en de loop voert ze allemaal uit. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_allowlist():
    quick_allowlist = L.choice(
        "quick-allowlist",
        question="Waarom zet een policy op een lijst wat mag, en niet wat verboden is?",
        options={
            "shorter": "Een lijst van wat mag is altijd korter, dus het model leest hem sneller",
            "model": "Het model weet dan vooraf welke calls het niet mag doen, en slaat die over",
            "forgot": "Een tool of plek waar niemand aan dacht, wordt dan ook geweigerd, in plaats van doorgelaten",
        },
        correct="forgot",
        why={
            "forgot": "Een allowlist weigert alles wat er niet op staat; een blocklist vergeet altijd iets.",
            "shorter": "De policy is niet voor het model om te lezen: je code controleert hem, voordat een call draait. Zie [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).",
            "model": "Het model hoeft de policy niet te kennen; je code houdt tegen wat niet op de lijst staat. Zie [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).",
        },
    )
    quick_allowlist
    return (quick_allowlist,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige dingen gaan in code makkelijker dan in woorden: een som, een lijst sorteren, de rijen van een tabel tellen. In [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl) rekende de code van de course uit wat er in de voorraadkast ontbrak, en schreef het model alleen de zin. Een agent kan dat zelf doen als hij een tool heeft die code draait: het model schrijft een paar regels Python, je programma voert ze uit, en het model leest wat eruit kwam. Maar die code schreef een model, en dat kan zijn taak verkeerd begrijpen, of zich laten beetnemen door een tekst die het las, zoals in [Wat een tool mag](/course/open/build-tool-permissions?lang=nl). Hij kan uren in een loop blijven hangen, bestanden schrijven waar dat niet mag, ze verwijderen, je keys lezen en ze wegsturen. Daarom draait hij niet in je eigen programma, maar in een **sandbox**: een plek die is afgesloten van de rest van je computer, zodat wat er misgaat daarbinnen blijft. In deze les bouw je een kleine, en meet je wat die tegenhoudt en wat niet.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **sandbox** | een afgesloten plek om code te draaien, zoals een apart proces, een container of een kleine virtuele machine, zonder je bestanden, je netwerk of je keys |
    | **proces** | een programma terwijl het draait, met zijn eigen geheugen; je Python-programma is er een, en het kan andere starten |
    | **`subprocess.run(command, ...)`** | start een programma als apart proces, wacht tot het klaar is, en geeft terug wat eruit kwam: `returncode` (0 als alles goed ging), `stdout` (wat het printte) en `stderr` (zijn fouten) |
    | **environment variable** | een naam met een waarde die een programma bij het starten van je computer krijgt, buiten zijn code; een **API key**, het geheim dat toegang geeft tot een betaald model, staat er vaak in |
    | **`cwd=folder`, `env={}`** | keyword arguments van `subprocess.run`: de map waarin het proces begint, en zijn environment variables, hier geen |
    | **`capture_output=True`, `text=True`** | bewaar wat het proces print, in `stdout` en `stderr`, als tekst en niet als bytes |
    | **`timeout=seconds`** | stop het proces na zoveel seconden; `subprocess.run` geeft dan de fout `subprocess.TimeoutExpired` |
    | **`tempfile.TemporaryDirectory()`** | maakt een nieuwe, lege map, en gooit die met alles erin weg aan het eind van het `with`-blok |
    | **`text.split("\n")`** | de regels van een tekst, als lijst: `"a\nb".split("\n")` is `["a", "b"]` |

    ### Vier lagen

    De sandbox van de course heeft vier lagen. **Een apart proces:** crasht de code of eet hij geheugen, dan stopt dat proces, en gaat je programma door. **Een tijdslimiet:** code die eindeloos doorloopt, wordt gestopt. **Een nieuwe, lege map,** en geen van je environment variables: een bestand dat de code schrijft, komt in een map die daarna wordt weggegooid, en een key die je in een environment variable bewaart, is er niet. **Een bewaker:** een paar regels Python van de course, die vóór de code van de agent draaien en hem stoppen als hij het netwerk opent, een bestand buiten zijn map, of een ander programma.
    """)
    return


@app.cell(hide_code=True)
def picture_layers():
    mo.Html(C.layers_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course schreef {n} stukken code van het soort dat een agent zou kunnen schrijven, en draaide ze allemaal op jouw computer, toen deze les openging: één keer in een sandbox met de eerste drie lagen, één keer ook met de bewaker, elke keer met een limiet van {s} seconden.
    """.format(n=F["rows"], s=F["seconds"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("De bewaker is een controle binnen Python, in hetzelfde proces als de code van de agent. De documentatie van Python zegt zelf dat code die eruit wil, om zo'n controle heen kan. Hij houdt tegen wat een agent per ongeluk doet, niet iemand die erop uit is om in te breken. Gebruik voor code die vreemden kunnen sturen een sandbox van het besturingssysteem: een container zonder netwerk, zoals een van Docker, een aparte gebruiker die maar één map kan openen, of een dienst die gemaakt is om code te draaien."),
        mo.accordion({"De bewaker van de course, als je hem wilt lezen": mo.md("```python\n" + C.GUARD + "```")}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: code draaien die een agent schreef

    1. **Nooit in je eigen programma:** start er een apart proces voor.
    2. **Geef hem een tijdslimiet,** en stop hem als de tijd om is.
    3. **Laat hem beginnen in een nieuwe, lege map,** die daarna wordt weggegooid, en geef hem geen van je environment variables: daar staan vaak keys in.
    4. **Sluit het netwerk af en de bestanden buiten de map:** hier met de bewaker van de course, voor code van vreemden met een container of een aparte gebruiker.
    5. **Stuur terug waar het model iets mee kan:** de output, afgeknipt bij een limiet, of de laatste regel van een fout, of dat de tijd op was.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een eigen proces

        `run(code, seconds)` draait code die een agent schreef in een sandbox. Het vak maakt de opdracht al, een lijst: `PYTHON`, de Python waarop de course draait, met `-I`, dat je eigen instellingen weglaat, en `-c` met de bewaker en de code. De regel met `with` maakt een nieuwe, lege map. Maak hem af. De check laat je functie zes stukken code draaien: de boodschappenlijst, een tikfout, code die zijn map opsomt, code die een environment variable leest, code die je bestand met keys leest, en code die langer duurt dan zijn limiet. Het plaatje eronder laat zien wat je `run` voor elk stuk teruggaf.
        """),
        L.task("Voer in het `with`-blok `command` uit in `folder` met `subprocess.run`, zonder je environment variables, met de output als tekst en een tijdslimiet van `seconds`; geef terug wat dat teruggeeft, of `None` als de tijdslimiet hem stopt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=14)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "`subprocess.run` krijgt de opdracht als lijst, en de rest als keyword arguments: een voor de map, een voor de environment variables, twee voor de output, een voor de tijdslimiet. Is de tijd om, dan stopt hij het proces en geeft hij de fout `subprocess.TimeoutExpired`.",
            "De keyword arguments zijn `cwd`, `env`, `capture_output`, `text` en `timeout`. Zet de aanroep, na `return`, in een `try:`, en vang de tijdslimiet op met `except subprocess.TimeoutExpired:`.",
            "`try:`, dan `return subprocess.run(command, cwd=folder, env={}, capture_output=True, text=True, timeout=seconds)`, dan `except subprocess.TimeoutExpired:` met `return None` eronder.",
        ],
        why="""```python
def run(code, seconds):
    command = [PYTHON, "-I", "-c", GUARD + code]
    with tempfile.TemporaryDirectory() as folder:
        try:
            return subprocess.run(command, cwd=folder, env={}, capture_output=True,
                                  text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            return None
```

De `return` staat in het `with`-blok, dus de map wordt er direct na weggegooid: wat de code daar schreef, is weg.""",
    )
    return


@app.cell(hide_code=True)
def text_result_text():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat het model leest

        Als tool van een agent geeft de sandbox zijn resultaat als tekst terug aan het model, met de rol `tool`. `result_text(done, seconds)` maakt die tekst uit wat `run` teruggaf: `done` is `None` als de tijdslimiet de code stopte; anders heeft het `done.returncode`, `done.stdout` en `done.stderr`. Het vak regelt de tijdslimiet al. Voeg de twee andere gevallen toe. Is `done.returncode` niet 0, dan stopte de code met een fout: het model krijgt `"Error: "` en de laatste regel van `done.stderr`, de regel die zegt wat er misging. Is de output langer dan `LIMIT` tekens, hier {n}, dan krijgt het model de eerste `LIMIT`, en een regel die zegt hoeveel er nog waren: één `print` in een loop kan een miljoen tekens schrijven, en die zouden allemaal in elk volgend request weer naar het model gaan. De check probeert een output, een fout met zijn traceback, een lange output, code die niets printte, en de tijdslimiet.
        """.format(n=C.number(C.LIMIT, L.lang))),
        L.task("Geef vóór de laatste regel een tekst terug met de laatste regel van `done.stderr` als `done.returncode` niet 0 is, en knip `done.stdout` af op `LIMIT` tekens, met een regel over hoeveel er weg zijn, als hij langer is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_result_text():
    box_result_text = L.code_box("result_text", lines=12)
    box_result_text
    return (box_result_text,)


@app.cell(hide_code=True)
def check_result_text(box_result_text):
    result_result_text = L.check("result_text", box_result_text.value)
    result_result_text
    return (result_result_text,)


@app.cell(hide_code=True)
def hints_result_text():
    L.hints(
        "result_text",
        [
            "Twee tests vóór de laatste regel, elk met een eigen `return`: een op `done.returncode`, een op de lengte van `done.stdout`.",
            "`done.stderr.strip().split(\"\\n\")` geeft de regels van de fout, en `[-1]` de laatste daarvan. `done.stdout[:LIMIT]` zijn de eerste `LIMIT` tekens, en `len(done.stdout) - LIMIT` is hoeveel er weg zijn.",
            "`if done.returncode != 0:` met `lines = done.stderr.strip().split(\"\\n\")` en `return \"Error: \" + lines[-1]`; dan `if len(done.stdout) > LIMIT:` met een `return` van `done.stdout[:LIMIT]`, plus een regel waar `str(len(done.stdout) - LIMIT)` in staat.",
        ],
        why="""```python
def result_text(done, seconds):
    if done is None:
        return "Stopped: the code was still running after " + str(seconds) + " seconds."
    if done.returncode != 0:
        lines = done.stderr.strip().split("\\n")
        return "Error: " + lines[-1]
    if len(done.stdout) > LIMIT:
        cut = len(done.stdout) - LIMIT
        return done.stdout[:LIMIT] + "\\n(cut off: " + str(cut) + " more characters)"
    return done.stdout
```

De regel erover doet ertoe: zonder die regel houdt het model het eerste stuk voor de hele output.""",
    )
    return


@app.cell(hide_code=True)
def text_guard_wall():
    mo.md(r"""
    ### Oefening 3. Code van vreemden
    """)
    return


@app.cell(hide_code=True)
def choice_guard_wall_cell():
    choice_guard_wall = L.choice(
        "guard-wall",
        question="Een webwinkel laat bezoekers vragen stellen aan een agent, en de agent draait de code die hij schrijft in de sandbox van de course. Is dat veilig genoeg?",
        options={
            "yes": "Ja: de bewaker houdt het netwerk, bestanden buiten de map en andere programma's tegen, en daarmee is alles gedekt",
            "no": "Nee: de bewaker is een controle in dezelfde Python als de code, en code die eruit wil, kan erlangs",
            "short": "Ja, als de tijdslimiet kort is: code die niet lang kan draaien, kan weinig kwaad",
        },
        correct="no",
        why={
            "no": "Bezoekers kunnen vragen schrijven waardoor de agent zulke code met opzet schrijft. Gebruik voor code die vreemden kunnen sturen een sandbox van het besturingssysteem: een container zonder netwerk, een aparte gebruiker, of een dienst die gemaakt is om code te draaien.",
            "yes": "De bewaker vangt op wat een agent per ongeluk doet. Python heeft manieren om om zo'n controle heen te gaan, en wie eruit wil, zoekt ze op.",
            "short": "Een key lezen en hem wegsturen duurt minder dan een seconde.",
        },
    )
    choice_guard_wall
    return (choice_guard_wall,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: je sandbox op je eigen code

    Zodra oefening 1 goed is, kun je eigen code in je sandbox draaien, zoals een agent dat zou doen: schrijf hem in het vak, en lees wat het model terug zou krijgen, via je `result_text` zodra oefening 2 goed is, en tot dan via die van de course. Probeer een `print`, een fout, een loop die nooit stopt, of `open` op een bestand buiten de map.
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.code_editor(value=C.SHOP, language="python", label="Code voor je sandbox")
    try_button = mo.ui.run_button(label="Draai in je sandbox")
    mo.vstack([try_box, try_button])
    return try_box, try_button


@app.cell(hide_code=True)
def demo_try(box_result_text, box_run, try_box, try_button):
    _run = C.learner_run(box_run.value) if try_button.value else None
    if not try_button.value:
        _out = mo.md("*Druk op de knop om te draaien.*")
    elif _run is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo draait je eigen `run`.*")
    else:
        _words = {"title": "Je code in je sandbox, met een limiet van 2 seconden", "reads": "Wat het model zou lezen:",
                  "error": "Je run stopte met {e}."}
        _out = mo.Html(C.tried(_run, C.learner_text(box_result_text.value), try_box.value, 2, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat een tool mag](/course/open/build-tool-permissions?lang=nl) controleerde elke call in je code, voordat hij draaide. Een tool die code draait, heeft meer nodig: de call zelf kan in orde zijn, terwijl de code erin iets anders doet. Daarom draait de code waar hij weinig kwaad kan, in een apart proces met een tijdslimiet, in een lege map, achter een bewaker, en krijgt de loop van [De agent loop](/course/open/build-agent-loop?lang=nl) een korte tekst terug die hij kan lezen. In de volgende les gebruikt een agent een browser.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_guard_wall, result_result_text, result_run):
    L.wrap_up(result_run, result_result_text, choice_guard_wall)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
