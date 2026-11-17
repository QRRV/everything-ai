# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-extensions. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-extensions", "nl")
    C = L.checks   # de zes skills van de course, de runs van Qwen3 met en zonder skills, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl), en een over [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_list():
    quick_list = L.choice(
        "quick-list",
        question="Een MCP client heeft net een server gestart. Hoe komt hij te weten welke tools de server heeft?",
        options={
            "list": "Hij stuurt tools/list, en de server antwoordt met elke tool en zijn input schema",
            "start": "De server schrijft al zijn tools uit zichzelf naar zijn uitvoer, zodra de client hem start",
            "call": "Hij stuurt tools/call met een lege naam, en de fout noemt elke naam",
        },
        correct="list",
        why={
            "list": "De client maakt daarna van elke tool de vorm van de chat-API, en geeft de tools aan het model.",
            "start": "Een server antwoordt op requests; hij stuurt niets voordat hij iets gevraagd wordt. Zie [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl).",
            "call": "Een call van een tool die de server niet heeft, is een fout, en die fout noemt niets. Zie [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl).",
        },
    )
    quick_list
    return (quick_list,)


@app.cell(hide_code=True)
def quick_check_agents():
    quick_agents = L.choice(
        "quick-agents",
        question="Wat hoort in AGENTS.md?",
        options={
            "spec": "De voorbeelden van wat deze ene taak moet opleveren, zodat de agent zijn werk kan controleren",
            "plan": "De stappen van deze taak, in de volgorde waarin de agent ze gaat zetten",
            "always": "Wat voor elke taak in het project geldt, zoals hoe je de tests draait",
        },
        correct="always",
        why={
            "always": "Een agent leest AGENTS.md aan het begin van elke taak, dus daar staat wat elke taak nodig heeft.",
            "spec": "Dat is de spec van één taak. Zie [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl).",
            "plan": "Dat is het plan van één taak. Zie [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl).",
        },
    )
    quick_agents
    return (quick_agents,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een agent weet wat in zijn context staat en kan wat zijn tools kunnen; het model zelf blijft hetzelfde. Je leert een agent dus iets nieuws met wat je hem voorlegt. [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl) gaf hem tools uit een programma. Deze les voegt er drie manieren aan toe: notities over een project die hij aan het begin van elke taak leest, werkwijzen die hij pas leest als een taak ze nodig heeft, en de beschrijving van een web-API, waaruit hij een tool krijgt voor alles wat de API kan. Een plugin bundelt zulke uitbreidingen, zodat je ze in één keer installeert.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **AGENTS.md** | een bestand bovenin een project met wat voor elke taak geldt, dat een coding agent vóór je vraag leest |
    | **skill** | een map met een bestand SKILL.md: bovenaan een naam en een beschrijving, daaronder de instructies; er kunnen ook scripts en andere bestanden in staan |
    | **frontmatter** | de regels bovenaan een bestand tussen twee regels `---`, elk een key, `: ` en een value |
    | **progressive disclosure** | de korte versie van iets altijd laten zien, en het geheel pas als het nodig is |
    | **OpenAPI-specificatie** | een beschrijving van een web-API als JSON of YAML: voor elk pad en elke methode een operation, met zijn `operationId`, een `summary` en zijn parameters met hun schema's |
    | **plugin** | een pakket uitbreidingen voor een agent, zoals skills, MCP servers en commando's, dat je in één keer installeert |

    AGENTS.md kost ruimte in elk request, dus er hoort alleen in wat elke taak nodig heeft. Een werkwijze die maar af en toe telt, hoort in een skill. De agent ziet elke skill altijd als één regel: de naam en de beschrijving, die zegt waar de skill voor is en wanneer je hem gebruikt. De instructies blijven in het bestand tot een vraag erbij past; dan laadt de agent ze, hier met een tool `load_skill`. Dat is progressive disclosure: met honderd skills leest de agent honderd korte regels, en de instructies van één. Anthropic introduceerde skills in oktober 2025 en publiceerde de vorm als open standaard; andere coding agents lezen hem ook.
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

    De course schreef {k} skills voor een assistent in de keuken, zoals cups en spoons in grammen, hoelang restjes goed blijven, en de allergenen in de gerechten; elk heeft getallen die een model niet kan weten. Daarna stelde de course Qwen3-0.6B {n} vragen, twee per skill, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt, op drie manieren: zonder de skills; met de lijst van skills in de system prompt en de tool `load_skill`; en met de instructies van alle {k} skills in de system prompt. Daarna deed de course de tweede manier nog een keer, met een behulpzamere fout, zoals in [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl): als het model een skill aanriep alsof het een tool was, zei de fout dat het hem met `load_skill` moest laden.
    """.format(k=F["skills"], n=F["questions"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een skill is instructies die de agent volgt, en er kunnen scripts bij zitten die op je computer draaien: installeer skills en plugins alleen van mensen die je vertrouwt, en lees ze eerst. In een skill van ergens anders kunnen instructies staan waar je niet om vroeg, die de agent dan opvolgt: een prompt injection."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een agent iets leren zonder het model te veranderen

    1. **Wat voor elke taak geldt** in een project, hoort in AGENTS.md: hoe je de tests draait, de regels, wat de agent niet mag aanraken.
    2. **Een werkwijze die af en toe telt,** wordt een skill: een map met een SKILL.md, waarvan de beschrijving zegt waar hij voor is en wanneer je hem gebruikt.
    3. **Geef de agent de lijst van skills,** een regel per skill, en een manier om er een te laden; hij leest dan de instructies van een skill pas als een vraag ze nodig heeft.
    4. **Een web-API met een OpenAPI-beschrijving** wordt tools: een operation wordt een tool, zijn parameters de parameters van de tool, en elke call een HTTP-request dat je code verstuurt.
    5. **Wil je een set uitbreidingen delen,** zet ze dan in een plugin, en installeer plugins alleen van mensen die je vertrouwt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture("nl"))
    return


@app.cell(hide_code=True)
def text_front():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De naam en beschrijving van een skill

        `front(text)` leest de frontmatter van een SKILL.md: de regels tussen de eerste regel `---` en de volgende. De eerste regels staan er al: ze splitsen de tekst in regels en zoeken de regel die de frontmatter afsluit. Zet voor elke regel daartussen met `: ` erin het deel vóór de eerste `: ` als key in `found`, en het deel erna, zonder spaties aan de uiteinden, als value. In de value zelf kan ook `: ` staan. (Agents lezen echte frontmatter als YAML, en daarin staat zo'n value tussen aanhalingstekens; je lezer houdt het eenvoudig.) De check probeert de SKILL.md van cup-measures, die het plaatje laat zien, en twee eigen skills.
        """),
        L.task("Vervang `...` door de regels die elke key met zijn value in `found` zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_front():
    box_front = L.code_box("front", lines=10)
    box_front
    return (box_front,)


@app.cell(hide_code=True)
def check_front(box_front):
    result_front = L.check("front", box_front.value)
    result_front
    return (result_front,)


@app.cell(hide_code=True)
def hints_front():
    L.hints(
        "front",
        [
            "Kijk naar elke regel: staat er `: ` in, splits hem dan in twee delen, alleen bij de eerste `: `.",
            "`line.split(\": \", 1)` splitst bij de eerste `: ` en geeft een lijst van twee delen; `.strip()` haalt de spaties van de uiteinden van een tekst.",
            "`if \": \" in line:`, dan `key, value = line.split(\": \", 1)` en `found[key] = value.strip()`.",
        ],
        why="""```python
def front(text):
    lines = text.split("\\n")
    end = lines.index("---", 1)   # the line that closes the frontmatter
    found = {}
    for line in lines[1:end]:
        if ": " in line:
            key, value = line.split(": ", 1)
            found[key] = value.strip()
    return found
```

`lines.index("---", 1)` zoekt `---` vanaf de tweede regel, dus vindt het de regel die de frontmatter afsluit, niet die hem opent.""",
    )
    return


@app.cell(hide_code=True)
def text_openapi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tools uit de beschrijving van een web-API

        `openapi_tools(spec)` maakt een tool voor elke operation van een OpenAPI-specificatie. De loops over de paden en hun methodes staan er al, net als de naam en beschrijving van elke tool. Vul de parameters in: kopieer voor elke parameter in de `"parameters"` van de operation zijn `"schema"`, geef de kopie de `"description"` van de parameter, en zet hem in `properties` onder de `"name"` van de parameter. Heeft de parameter `"required": true`, voeg zijn naam dan toe aan `required`. Een operation zonder parameters heeft geen key `"parameters"`, en een parameter die niet verplicht is, heeft geen key `"required"`. De check leest de API van de buurtwinkel hieronder, en de API van een weerdienst.
        """),
        mo.accordion({"De API van de buurtwinkel, zoals de OpenAPI-specificatie hem geeft": mo.md(
            "```json\n" + C.SHOP_API_JSON + "\n```")}),
        L.task("Vervang `...` door een loop over de parameters van de operation die `properties` en `required` vult."),
    ])
    return


@app.cell(hide_code=True)
def exercise_openapi():
    box_openapi = L.code_box("openapi", lines=20)
    box_openapi
    return (box_openapi,)


@app.cell(hide_code=True)
def check_openapi(box_openapi):
    result_openapi = L.check("openapi", box_openapi.value)
    result_openapi
    return (result_openapi,)


@app.cell(hide_code=True)
def hints_openapi():
    L.hints(
        "openapi",
        [
            "Loop over `operation.get(\"parameters\", [])`: `.get` met een tweede waarde geeft een lege lijst als de key er niet is.",
            "`schema = dict(p[\"schema\"])` maakt een kopie die je mag veranderen; `schema[\"description\"] = p[\"description\"]` geeft hem de beschrijving. `p.get(\"required\")` is `None` als de key er niet is.",
            "`for p in operation.get(\"parameters\", []):`, dan de twee regels met `schema`, `properties[p[\"name\"]] = schema`, en `if p.get(\"required\"):` met `required.append(p[\"name\"])`.",
        ],
        why="""```python
            for p in operation.get("parameters", []):
                schema = dict(p["schema"])
                schema["description"] = p["description"]
                properties[p["name"]] = schema
                if p.get("required"):
                    required.append(p["name"])
```

Een tool die zo gemaakt is, beschrijft de operation alleen. Om een call uit te voeren, verstuurt je code het HTTP-request: de methode, het pad met `{product}` ingevuld, en de andere arguments als query.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Oefening 3. Welke uitbreiding?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-extension",
        question="Een release van je team gaat volgens een lijst van twaalf stappen, twee pagina's lang, en gebeurt één keer per maand. Waar zet je de stappen, zodat je coding agent ze volgt?",
        options={
            "agents": "In AGENTS.md, dat de agent aan het begin van elke taak helemaal van voor naar achter leest",
            "skill": "In een skill: de agent ziet er één regel over, en laadt de stappen als dat nodig is",
            "server": "In een MCP server, met een eigen tool voor elk van de twaalf stappen van de release",
        },
        correct="skill",
        why={
            "skill": "De stappen tellen één keer per maand, dus ze kosten één regel in andere taken, en de agent laadt ze voor een release.",
            "agents": "Dan gaan er twee pagina's in elke taak, ook in de taken die niets met een release te maken hebben: AGENTS.md is voor wat elke taak nodig heeft.",
            "server": "Een tool doet één ding als hij wordt aangeroepen; stappen die de agent in een volgorde moet volgen, zijn instructies, en die horen in een skill.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Schrijf een eigen skill: in het vak hieronder staat er een om mee te beginnen. Is oefening 1 goed, dan leest je `front` de naam en beschrijving, en zet de demo je skill bij de lijst van de zes skills van de course, met de tool `load_skill`. Qwen3-0.6B, op de server van de course op je computer, beantwoordt dan je vraag. Laadt het je skill?
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
def skill_box_cell():
    skill_box = mo.ui.text_area(value=C.DEMO_SKILL, label="Je SKILL.md", rows=14, full_width=True)
    ask_box = mo.ui.text(value=C.DEMO_QUESTION, label="Je vraag", full_width=True)
    ask_button = mo.ui.run_button(label="Vraag")
    mo.vstack([skill_box, ask_box, ask_button])
    return ask_box, ask_button, skill_box


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_front, skill_box):
    _front = C.learner_fn("front", box_front.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _front is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo leest je skill met je eigen `front`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 …*"))
        _words = {"title": "Qwen3 met de skills van de course en die van jou", "line": "De regel die de agent altijd van je skill ziet:",
                  "no_load": "Het model laadde geen skill: het antwoordde zelf.",
                  "unreadable": "Je front kon deze tekst niet lezen als SKILL.md: hij moet beginnen met een regel `---`, en onder de frontmatter een tweede regel `---` hebben.",
                  "no_name": "Je skill heeft een name en een description in zijn frontmatter nodig.",
                  "bad_name": "De naam van een skill bestaat uit kleine letters, cijfers en streepjes, zoals bread-rising.",
                  "taken": "De course heeft al een skill met die naam. Geef de jouwe een andere.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.ask_with_skill(_front, skill_box.value, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl) zette wat voor elke taak geldt in AGENTS.md; een skill bevat wat maar af en toe geldt, en kost één regel tot hij nodig is. [Tools via een protocol: MCP](/course/open/build-mcp?lang=nl) gaf een agent tools uit een programma; de beschrijving van een web-API geeft hem ook tools, gemaakt door je `openapi_tools`. Een plugin bundelt skills, servers en commando's. De volgende les bekijkt drie soorten agents die uit al deze delen bestaan: voor code, voor onderzoek en voor je eigen taken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_front, result_openapi):
    L.wrap_up(result_front, result_openapi, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
