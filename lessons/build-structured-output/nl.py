# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-structured-output. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-structured-output", "nl")
    C = L.checks   # de opgenomen runs van de course, de stand-in van de checks, en de demo met je ask_json


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) en [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_prefill():
    quick_prefill = L.choice(
        "quick-prefill",
        question="Je request eindigt met de prefill `{\"city\": \"`, en het antwoord dat terugkomt bevat `Paris\"}`. Welke tekst kan `json.loads` lezen?",
        options={
            "rest": "`Paris\"}`, de tekst van het antwoord precies zoals hij terugkwam",
            "prefill": "`{\"city\": \"`, de prefill alleen",
            "joined": "`{\"city\": \"Paris\"}`, prefill en antwoord samen",
        },
        correct="joined",
        why={
            "joined": "Het antwoord bevat alleen wat het model na de prefill schreef; samen zijn ze de JSON-tekst.",
            "rest": "Dat is alleen wat het model na de prefill schreef, en dat kan `json.loads` niet in zijn eentje lezen. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
            "prefill": "Dat is het begin dat jij schreef; het deel van het model ontbreekt. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
        },
    )
    quick_prefill
    return (quick_prefill,)


@app.cell(hide_code=True)
def quick_check_except():
    quick_except = L.choice(
        "quick-except",
        question="`json.loads` stopt met een `ValueError` bij een tekst die geen JSON is. Wat laat dit programma zien?\n\n```python\nimport json\n\ntry:\n    value = json.loads(\"Sure! Here it is.\")\nexcept ValueError:\n    value = None\nprint(value)\n```",
        options={
            "none": "`None`",
            "text": "`Sure! Here it is.`",
            "stops": "Een foutmelding, en het programma stopt",
        },
        correct="none",
        why={
            "none": "`json.loads` stopt met een `ValueError`, dus springt Python naar `except ValueError:`, en `value` wordt `None`; het programma gaat verder.",
            "text": "`json.loads` geeft geen tekst terug die het niet kan lezen: het stopt met een `ValueError`. Zie [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).",
            "stops": "De `except ValueError:` vangt precies die fout op, dus het programma gaat verder. Zie [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).",
        },
    )
    quick_except
    return (quick_except,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een app die een model om gegevens vraagt, zoals de naam en de prijs van een product in een tekst, wil geen verhaal terug. Hij wil JSON die hij kan lezen, met de keys die hij nodig heeft, en een getal waar hij met een getal rekent. Een model schrijft tekst, token voor token, en doet dat vaak goed, maar niet altijd. Eén antwoord dat bijna JSON is, laat je programma stoppen; erger nog, een prijs die een tekst is, kan erdoorheen glippen. Deze les maakt van het antwoord iets waar je code op kan bouwen: je code controleert elk antwoord tegen een beschrijving van wat erin moet staan, en vraagt opnieuw met wat er mis was.

    **Een voorbeeld uit het dagelijks leven.** Het bestelformulier van een webwinkel neemt *een stuk of twintig* niet aan in het vak voor het aantal. Het zegt wat er mis is, *vul een heel getal in*, en je vult het opnieuw in. Dat is maar een vergelijking; hieronder zie je hoe vaak de JSON van Qwen3 niet was wat een app nodig heeft, en wat opnieuw vragen deed.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **structured output** | een antwoord in een vaste vorm die een programma kan lezen zonder te gokken, zoals JSON met vaste keys en types |
    | **JSON mode** | een instelling in het request, `"response_format": {"type": "json_object"}`, waarmee de server ervoor zorgt dat het antwoord geldige JSON is |
    | **JSON Schema** | een standaard om in JSON te beschrijven wat een stuk JSON moet bevatten: welke keys, welke daarvan verplicht zijn, en het type van elke waarde; één zo'n beschrijving heet *een JSON-schema* |
    | **isinstance(x, t)** | `True` als de waarde `x` van het type `t` is, zoals `isinstance("649", str)`; met een tuple van types, zoals `(int, float)`, als hij een van die types heeft |
    | **not** | maakt van `True` `False` en van `False` `True`; `key not in value` is `True` als de key ontbreekt |

    ### Bijna JSON

    De course vroeg Qwen3 acht keer om de gegevens van een fiets als JSON, via zijn eigen server. De vraag is Engels, omdat Qwen3 Engels het best schrijft:

    ```
    Give the name, color and price of this bike as JSON, with the keys name, color and price.

    Our new Cityline is a grey city bike with seven gears, for 649 euros.
    ```

    Elk antwoord hieronder staat op één regel: ↵ is een regeleinde, en een prijs die een tekst is, is gemarkeerd.
    """)
    return


@app.cell(hide_code=True)
def picture_plain():
    mo.Html(C.plain_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_plain():
    _f = C.findings()
    mo.md(r"""
    Alle {fenced} antwoorden stonden in een codeblok dat begint met drie backticks, en `json.loads` kon er {read} van lezen. Een app zou de backticks eraf kunnen halen, maar ook daarbinnen was de prijs in {text} van de {runs} de tekst `"649"`, en niet het getal 649: daar kan je code niet mee rekenen, en `"649" + 10` stopt met een fout. In [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) legde een prefill vast hoe een antwoord begint; deze les heeft er nog twee manieren bij: JSON mode, en een controle van wat er in het antwoord staat.

    ### JSON mode

    Met `"response_format": {{"type": "json_object"}}` in het request zet de server van de course JSON mode aan. Het model schrijft zoals altijd, token voor token, maar voordat de server een token houdt, kijkt hij of de tekst tot dan toe nog JSON kan worden. Een token waarna dat niet meer kan, zoals de backticks aan het begin, legt hij opzij, en hij trekt opnieuw uit de andere tokens. Zodra het JSON-object af is, eindigt het antwoord. Dezelfde vraag, met JSON mode:
    """.format(fenced=_f["plain_fenced"], read=_f["plain_json"], text=_f["plain_text_price"], runs=_f["runs"]))
    return


@app.cell(hide_code=True)
def picture_mode():
    mo.Html(C.mode_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_mode():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Deze keer kon `json.loads` ze alle {read} lezen. De antwoorden zagen eruit als de JSON in de codeblokken van daarnet, met dezelfde keys en dezelfde opmaak, en in {text} van de {runs} was de prijs weer de tekst `"649"`. JSON mode maakt het antwoord geldige JSON; welke keys en types je app nodig heeft, weet hij niet.
        """.format(read=_f["mode_json"], text=_f["runs"] - _f["mode_fit"], runs=_f["runs"])),
        L.watch_out("JSON mode kan geen antwoord afmaken dat `max_tokens` eerder stopt: `finish_reason` is dan `\"length\"`, en de JSON is niet af. Geef het antwoord genoeg ruimte, en controleer het toch."),
    ])
    return


@app.cell(hide_code=True)
def text_schema():
    mo.vstack([
        mo.md(r"""
        ### Een schema

        Wat je app nodig heeft, schrijf je één keer op, in een JSON-schema: hier dat van de fiets. `"properties"` zegt welke keys er zijn en welk type elke waarde heeft, en `"required"` welke van de keys er moeten zijn:

        ```python
        SCHEMA = {"type": "object",
                  "properties": {"name": {"type": "string"}, "color": {"type": "string"},
                                 "price": {"type": "number"}},
                  "required": ["name", "color", "price"]}
        ```

        JSON Schema heeft nog veel meer regels, zoals een laagste waarde, of een lijst van toegestane woorden. Je code controleert elk antwoord tegen het schema: kan `json.loads` het lezen, staat elke verplichte key erin, en heeft elke waarde zijn type? De antwoorden hierboven met `"649"` passen niet, want price moet een getal zijn.
        """),
        L.watch_out("De controle in deze les is met opzet klein: hij kent twee types en de verplichte keys. In Python is `isinstance(True, int)` ook `True`, dus hij zou `true` als prijs doorlaten. De library `jsonschema` controleert elke regel van de standaard, die ook; Pydantic controleert in plaats daarvan tegen een Python-class, en weigert `\"649\"` en `true` als getal alleen in zijn strikte modus (voorbeelden in november 2026)."),
    ])
    return


@app.cell(hide_code=True)
def text_again():
    mo.md(r"""
    ### Opnieuw vragen met wat er mis was

    Als een antwoord niet past, kan de app opnieuw vragen. Niet met dezelfde vraag: dan weet het model niet wat er mis was, en schrijft het misschien weer hetzelfde. In plaats daarvan zet de app het antwoord in het gesprek, als dat van de assistant, met daarachter een bericht van de user dat de problemen noemt. De course deed dat bij elke run met JSON mode, met hooguit drie pogingen:
    """)
    return


@app.cell(hide_code=True)
def picture_again():
    mo.Html(C.again_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_again_after():
    _f = C.findings()
    _first = C.answers("json-mode")[0]["tries"]
    mo.vstack([
        mo.md(r"""
        {again} van de {runs} runs hadden een tweede poging nodig, en alle {again} pasten daarna: uiteindelijk pasten {fit} van de {runs} antwoorden bij het schema. Elke poging is een heel nieuw request, en het gesprek groeit elke keer: de eerste poging was {one} input tokens, de tweede {two}. Houd het aantal pogingen dus laag, en geef `None` terug als niets past: dan kan de rest van je app dat zeggen, in plaats van door te gaan met een verkeerd antwoord.
        """.format(again=_f["again"], runs=_f["runs"], fit=_f["final_fit"], one=_first[0]["prompt_tokens"], two=_first[1]["prompt_tokens"])),
        L.watch_out("Andere API's en servers gaan verder, met *structured outputs*: je stuurt het schema zelf mee in het request, en de server zorgt ervoor dat het antwoord erbij past terwijl het model schrijft, zoals JSON mode dat doet voor geldige JSON. De API van OpenAI neemt een schema in `response_format`, de server van llama.cpp neemt er ook een, en Ollama neemt er een in `format` (voorbeelden in november 2026). De server van de course heeft alleen JSON mode: vraag je hem om een schema, dan antwoordt hij met een fout die dat zegt. Controleer ook met structured outputs het antwoord in code: een server ondersteunt maar een deel van JSON Schema, en een antwoord dat `max_tokens` afkapte, is nog steeds niet af."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: JSON waar je op kunt bouwen

    1. **Schrijf op wat je nodig hebt:** een JSON-schema met de keys, het type van elke waarde, en welke keys verplicht zijn.
    2. **Vraag om JSON:** noem de keys in de prompt, en zet JSON mode aan, of structured outputs waar de API die heeft.
    3. **Controleer elk antwoord in code:** `json.loads`, de verplichte keys, en het type van elke waarde.
    4. **Vraag opnieuw met wat er mis was:** voeg het antwoord en een bericht met de problemen toe, hooguit een paar keer; als niets past, geef je `None` terug.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.findings()["worked"]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste run van de course met JSON mode. Het eerste antwoord van Qwen3 was:

    ```json
    {first}
    ```

    1. **Wat je nodig hebt:** het schema van de fiets hierboven.
    2. **Vraag om JSON:** de vraag noemt de keys, en JSON mode staat aan.
    3. **Controleer:** `json.loads` leest het, en het is een dictionary; `name`, `color` en `price` staan er alle drie in. `name` en `color` zijn strings, zoals het schema zegt, maar `"649"` is ook een string, waar het schema `number` zegt. Eén probleem: `{found}`.
    4. **Vraag opnieuw:** het gesprek krijgt twee berichten: het antwoord hierboven als dat van de assistant, en dat van de user, `{fix}` Het tweede antwoord van Qwen3 had `"price": 649`, en dat past, dus de functie geeft `{value}` terug.
    """.format(first=_w["first"], found=_w["found"][0], fix=C.fix_message(_w["found"]),
               value=repr(_w["value"])))
    return


@app.cell(hide_code=True)
def text_problems():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een antwoord controleren tegen het schema

        `problems(text, schema)` geeft een lijst terug met wat er mis is met een antwoord, in woorden waar het model iets mee kan, en een lege lijst als het past. Het vak kijkt al of `json.loads` het antwoord kan lezen, of het een JSON-object is, en of elke verplichte key erin staat. Voeg de controle van de types toe: of elke key in `schema["properties"]` die in het antwoord staat, een waarde heeft van het type dat het schema noemt. `TYPES` geeft het Python-type bij elke naam, en `WORDS` hoe je het zegt. Als hij goed is, bewaart de check hem ook als bestand, `work/projects/app/schema.py`: een latere stap, [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl), gebruikt hem.
        """),
        L.task("Voeg na de loop over de verplichte keys een loop toe over de keys en regels in `schema[\"properties\"]`, die elke waarde van het verkeerde type in `found` zet, in woorden zoals `price must be a number`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_problems():
    box_problems = L.code_box("problems", lines=22)
    box_problems
    return (box_problems,)


@app.cell(hide_code=True)
def check_problems(box_problems):
    result_problems = L.check("problems", box_problems.value)
    result_problems
    return (result_problems,)


@app.cell(hide_code=True)
def hints_problems():
    L.hints(
        "problems",
        [
            "Loop door `schema[\"properties\"].items()`: elke key met zijn regel, zoals `\"price\"` met `{\"type\": \"number\"}`. Een waarde is een probleem als hij er is en `isinstance` zegt dat hij niet van zijn type is.",
            "`TYPES[rule[\"type\"]]` is het Python-type, zoals `(int, float)`. De voorwaarde: de key staat in `value`, en `not isinstance(value[key], ...)`. De tekst: `key + \" must be \" + WORDS[rule[\"type\"]]`.",
            "`for key, rule in schema[\"properties\"].items():`, daaronder `if key in value and not isinstance(value[key], TYPES[rule[\"type\"]]):`, en daaronder de `found.append(...)` met de tekst.",
        ],
        why="""```python
    for key, rule in schema["properties"].items():
        if key in value and not isinstance(value[key], TYPES[rule["type"]]):
            found.append(key + " must be " + WORDS[rule["type"]])
    return found
```

Een key die ontbreekt, staat al in `found`, dus de loop over de types slaat hem over met `key in value`. `(int, float)` laat zowel `649` als `649.5` door als getal.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Opnieuw vragen met wat er mis was

        `ask_json(url, messages, schema, tries)` vraagt met JSON mode, met `ask`, en controleert elk antwoord met `problems`: die van de course, zoals in oefening 1. Als een antwoord past, geeft hij terug wat `json.loads` ervan maakt. Maar na een antwoord dat niet past, stelt hij dezelfde vraag opnieuw. Laat hem vóór de volgende poging twee berichten aan het gesprek toevoegen: het antwoord, met de rol `assistant`, en een bericht met de rol `user` dat de problemen noemt, zoals *Fix these problems: price must be a number. Answer with the whole JSON again.* Maak een nieuwe lijst met `+`: `[1, 2] + [3]` geeft de nieuwe lijst `[1, 2, 3]`, en laat `[1, 2]` zoals hij was. De check roept je `ask_json` aan tegen een stand-in van de server, waarvan het eerste antwoord op de vraag over de fiets de prijs als tekst heeft.
        """),
        L.task("Maak in de loop, na de `if`, van `messages` het gesprek met het antwoord en je bericht met de problemen erbij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_json", lines=25)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask_json", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_json",
        [
            "Na de `if` paste het antwoord niet. De volgende poging stuurt `messages` opnieuw: maak er het oude gesprek van, met twee nieuwe berichten erachter.",
            "De twee berichten zijn dictionaries met `\"role\"` en `\"content\"`: het antwoord is `text`, met de rol `assistant`; je bericht bevat de problemen, samengevoegd met `\"; \".join(found)`, met de rol `user`.",
            "Eerst de tekst: `fix = \"Fix these problems: \" + \"; \".join(found) + ...`. Dan `messages = messages + [...]`: in de lijst het bericht van de assistant met `text`, en daarna dat van de user met `fix`.",
        ],
        why="""```python
def ask_json(url, messages, schema, tries):
    for attempt in range(tries):
        text = ask(url, messages)
        found = problems(text, schema)
        if found == []:
            return json.loads(text)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": fix}]
    return None
```

Met `+` wordt `messages` elke keer een nieuwe lijst, en blijft de lijst die de functie kreeg zoals hij was. Na de laatste poging stopt de loop, en geeft de functie `None` terug.""",
    )
    return


@app.cell(hide_code=True)
def text_guarantee():
    mo.md(r"""
    ### Oefening 3. Waar JSON mode voor zorgt
    """)
    return


@app.cell(hide_code=True)
def choice_guarantee_cell():
    choice_guarantee = L.choice(
        "guarantee",
        question="Je app vraagt de server van de course om de gegevens van de fiets, met JSON mode aan, en het antwoord komt terug met `finish_reason` `\"stop\"`. Welk antwoord kan dat nog zijn?",
        options={
            "fence": "De JSON in een codeblok met backticks",
            "keys": "`{\"price\": \"649\"}`, zonder name en color",
            "cut": "`{\"name\": \"Cityline\"` zonder de sluitende accolade",
        },
        correct="keys",
        why={
            "keys": "JSON mode zorgt ervoor dat het antwoord geldige JSON is, niet dat het bij je schema past: ontbrekende keys en een prijs die een tekst is, komen erdoor. Daar is je `problems` voor.",
            "fence": "Met backticks kan geldige JSON niet beginnen, dus legt JSON mode ze opzij, zoals bij de acht antwoorden hierboven. Waar weet JSON mode niets van?",
            "cut": "Met JSON mode eindigt het antwoord zodra het object af is, en niet eerder; in de praktijk kapt alleen `max_tokens` het af, en dan is `finish_reason` `\"length\"`. Waar weet JSON mode niets van?",
        },
    )
    choice_guarantee
    return (choice_guarantee,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 2 goed is, kun je het op het echte model proberen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo vraagt de naam, de kleur en de prijs als JSON, over de tekst in het vak, met je eigen `ask_json` en het schema van de fiets, hooguit drie keer. Verander de tekst: een fiets zonder prijs, of met de prijs in woorden. Schrijf hem in het Engels, omdat Qwen3 Engels het best schrijft.
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
def text_box_cell():
    text_box = mo.ui.text_area(value=C.BIKE_TEXT, label="De tekst over de fiets", full_width=True, rows=2)
    send_button = mo.ui.run_button(label="Vraag met je ask_json")
    mo.vstack([text_box, send_button])
    return send_button, text_box


@app.cell(hide_code=True)
def demo_answer(box_ask, send_button, text_box):
    _ask = C.learner_ask_json(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _ask is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo vraagt met je eigen `ask_json`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not text_box.value.strip():
        _out = mo.md("*Schrijf eerst een tekst over een fiets in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat Qwen3 antwoordde, en wat je ask_json teruggaf", "try": "Poging {n}:",
                  "found": "problemen: {p}", "fits": "past bij het schema", "gave": "Je ask_json gaf terug:",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, text_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De `json.loads` van [Data als tekst: JSON](/course/open/py-json?lang=nl) leest nu de antwoorden van een model, met de `try` en `except` van [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl) eromheen. De response prefill van [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) legde vast hoe een antwoord begint; het schema controleert het hele antwoord. In de volgende les, [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl), gaan je prompts uit je code naar templates met velden, zodat je weet welke versie welke antwoorden gaf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_guarantee, result_ask, result_problems):
    L.wrap_up(result_problems, result_ask, choice_guarantee)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
