# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-where-models-run. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-where-models-run", "nl")
    C = L.checks   # de stand-ins op andere plekken, de gemeten snelheden van de course, en de demo met jouw ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="Hoe kom je erachter welk model het best is voor je app?",
        options={
            "list": "Neem het model bovenaan een openbare ranglijst",
            "own": "Probeer de kandidaten op voorbeelden uit je eigen taak, en weeg kwaliteit, snelheid en prijs af",
            "big": "Neem het grootste model dat je kunt betalen",
        },
        correct="own",
        why={
            "own": "Een ranglijst meet andere taken; alleen je eigen voorbeelden zeggen hoe goed een model de jouwe doet, en hoe snel en duur het daar is.",
            "list": "Een ranglijst meet de taken van de ranglijst. Wat zegt die over de jouwe? Zie [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).",
            "big": "Een groter model is trager en duurder, en bij makkelijke vragen vaak niet beter. Zie [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def quick_check_router():
    quick_router = L.choice(
        "quick-router",
        question="Wat doet een router?",
        options={
            "send": "Hij stuurt elke vraag naar het model dat erbij past, zoals makkelijke naar een snel model en moeilijke naar een langzaam",
            "all": "Hij stelt elke vraag aan alle modellen, en houdt het beste antwoord",
            "network": "Hij verbindt je computer met het internet",
        },
        correct="send",
        why={
            "send": "Een router kijkt eerst naar elke vraag, en kiest er één model voor, zodat makkelijke vragen niet de prijs van de langzame weg betalen.",
            "all": "Dan kost elke vraag de prijs van elk model. Zie [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).",
            "network": "Dat is een router voor netwerken. Welke router bedoelt die les? Zie [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl).",
        },
    )
    quick_router
    return (quick_router,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Achter elke API staat een computer die het model draait. Dat kan je eigen laptop zijn, een server met een GPU in je bedrijf, of het datacenter van een bedrijf dat de antwoorden verkoopt. Je code verandert nauwelijks: het request is hetzelfde, alleen het adres verschilt. Maar de plek bepaalt wie je data ziet, wat elk antwoord kost, en hoe snel het komt.

    **Een voorbeeld uit het dagelijks leven.** Je kunt thuis koken, een kok in je keuken laten komen, of naar een restaurant gaan. Het recept is hetzelfde; wat verandert, is wie in je keuken staat, wat je betaalt, en hoe lang het duurt. Dat is maar een vergelijking; hieronder zie je de echte plekken.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **open weights** | weights die iedereen mag downloaden en draaien, onder een licentie zoals Apache-2.0, zoals die van Qwen3-0.6B; de weights van een closed model bereik je alleen via de API van de maker |
    | **local runtime** | een programma dat open weights op je eigen computer laadt en antwoordt via een API daarop, vaak OpenAI-compatible: Ollama, llama.cpp, LM Studio, en de eigen server van de course (voorbeelden in november 2026) |
    | **inference provider** | een bedrijf dat modellen voor je op zijn GPU's draait, ook modellen met open weights, achter een API met een key, betaald per token; een model zo laten draaien heet ook *hosted inference* |
    | **base URL** | het deel van het adres vóór het endpoint, zoals `http://127.0.0.1:8765/v1`; verander je het, dan verander je waar het model draait |

    ### Dezelfde aanroep, andere adressen

    | Waar | Base URL (voorbeelden in november 2026) | Wie je data ziet | Wat je betaalt |
    |---|---|---|---|
    | de server van de course, op je computer | `http://127.0.0.1:8765/v1` | alleen jij | je stroom |
    | Ollama, op je computer | `http://localhost:11434/v1` | alleen jij | je stroom |
    | `llama-server` van llama.cpp, op je computer | `http://localhost:8080/v1` | alleen jij | je stroom |
    | vLLM, op je eigen server met een GPU | `http://<je server>:8000/v1` | je organisatie | de server |
    | een inference provider, of de API van een maker | het adres van het bedrijf, met een key | het bedrijf | per token |

    `localhost` is een andere naam voor `127.0.0.1`. Wat er op elk adres draait, kan hetzelfde open model zijn: Qwen3 draait in allemaal.

    ### Past het?

    Een model is zijn weights, en elke weight neemt geheugen in: 32 bits in `float32`, 16 in `bfloat16`, en ongeveer 4 als de weights afgerond bewaard worden, zoals local runtimes vaak doen. Acht bits zijn een byte.
    """)
    return


@app.cell(hide_code=True)
def picture_sizes():
    mo.Html(C.sizes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_speed():
    _n = C.numbers(L.lang)
    mo.vstack([
        mo.md(rf"""
        Qwen3-0.6B past op elke laptop. Een model van 70 miljard weights past niet op een laptop met 16 GB, zelfs niet met 4 bits: dat heeft een of meer grote GPU's nodig, in je eigen server of bij een bedrijf. En geheugen is niet alles: op één thread van de computer van de course schreef Qwen3-0.6B {_n['float32']} tokens per seconde in `float32`, en {_n['bfloat16']} in `bfloat16`. Een GPU schrijft vele keren zo snel, en daarvoor betaal je een host.
        """),
        L.watch_out("Een local runtime houdt je data op je computer, maar een klein model antwoordt ook als een klein model. Meet op je eigen voorbeelden, zoals in de vorige les, waar het model ook draait."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: één aanroep, elke plek

    1. **Kies de plek:** je computer voor privacy en geen rekening per token, een host voor grote modellen en snelheid.
    2. **Start of zoek de server:** een local runtime, zoals `ollama serve`, of het adres en de key van een host.
    3. **Neem zijn base URL en de naam van zijn model:** elke plek noemt zijn modellen op zijn eigen manier, zoals `qwen3-0.6b` of `llama3.2:1b`.
    4. **Roep hem aan zoals altijd:** `base_url + "/chat/completions"`, met de naam in `"model"`.

    In november 2026 doet het package `openai` hetzelfde met `OpenAI(base_url=..., api_key=...)`: voor een local runtime is elke key goed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    ### Uitgewerkt voorbeeld

    Hoeveel geheugen nemen de weights van Qwen3-0.6B in `bfloat16` in?

    ```python
    params = 600_000_000
    bits = 16
    params * bits / 8 / 1_000_000_000
    ```

    1. **De weights:** ongeveer 600 miljoen; `600_000_000` is dat getal, en Python slaat de liggende streepjes over.
    2. **De bits:** in `bfloat16` neemt elke weight 16 bits in.
    3. **De bytes:** gedeeld door 8, omdat een byte 8 bits heeft: 1.200.000.000 bytes.
    4. **De gigabytes:** een gigabyte is een miljard bytes, dus `1.2`: de weights nemen 1,2 GB in, de helft van wat ze in `float32` innemen.

    De download van Qwen3-0.6B is toch 1,5 GB. Het bestand bewaart één tabel twee keer: de {_n['table']} weights die elk token aan het begin in getallen omzetten, en dezelfde tabel nog een keer als laatste layer, die het volgende token kiest. Het model gebruikt één tabel voor allebei, dus het heeft {_n['weights']} weights, maar het bestand bevat {_n['file']} getallen: {_n['file_gb']} GB in `bfloat16`.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén functie, elke plek

        Deze `ask` bereikt alleen de server van de course, en vraagt alleen om `"qwen3-0.6b"`. Laat hem elke OpenAI-compatible server bereiken: gebruik de `base_url` en het `model` die hij krijgt. De check roept hem aan op twee plekken, een stand-in met `qwen3-0.6b` en een met `llama3.2:1b`, elk op een eigen adres.
        """),
        L.task("Gebruik het `model` en de `base_url` die `ask` krijgt, in plaats van de vaste naam en het vaste adres."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=12)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "Twee dingen staan vast in de code die uit de argumenten zouden moeten komen: de naam van het model, en het adres.",
            "In de body `\"model\": model`, zonder aanhalingstekens om `model`. Het adres is de `base_url` met het endpoint erachter.",
            "`urllib.request.Request(base_url + \"/chat/completions\", data=data, ...)`, met de rest zoals het is.",
        ],
        why="""```python
def ask(base_url, model, question):
    body = {"model": model, "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(base_url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

Waar het model draait, is alleen het adres en de naam; al het andere is hetzelfde request.""",
    )
    return


@app.cell(hide_code=True)
def text_gb():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het geheugen van de weights

        Schrijf `weights_gb(params, bits)`: hoeveel gigabyte de weights van een model innemen, met `params` weights van elk `bits` bits. Voor Qwen3-0.6B in `float32` geeft `weights_gb(600_000_000, 32)` `2.4`.
        """),
        L.task("Maak `weights_gb` af: weights keer bits, dan bytes, dan gigabytes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("weights_gb", lines=3)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("weights_gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "weights_gb",
        [
            "Volg het uitgewerkte voorbeeld, met `params` en `bits` op de plek van de getallen.",
            "`params * bits` is het aantal bits; gedeeld door 8 de bytes; gedeeld door een miljard de gigabytes.",
            "`return params * bits / 8 / ...`, met een miljard op de plek van de puntjes, geschreven als `1_000_000_000`.",
        ],
        why="""```python
def weights_gb(params, bits):
    return params * bits / 8 / 1_000_000_000
```

Van bits naar bytes is delen door 8; van bytes naar gigabytes is delen door een miljard.""",
    )
    return


@app.cell(hide_code=True)
def text_where():
    mo.md(r"""
    ### Oefening 3. Waar moet het draaien?
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="Een ziekenhuis wil een model brieven over zijn patiënten laten samenvatten, en de brieven mogen de eigen computers van het ziekenhuis niet verlaten. Waar moet het model draaien?",
        options={
            "host": "Bij een bedrijf dat open modellen host, achter een API",
            "maker": "Bij de API van een grote modelmaker",
            "own": "Een open-weight model in een local runtime, op de eigen server van het ziekenhuis",
        },
        correct="own",
        why={
            "own": "Open weights in een eigen runtime: de brieven verlaten het gebouw nooit. De prijs is de hardware, en een model dat er klein genoeg voor is.",
            "host": "Dan gaat elke brief naar de computers van de host. Welke plek houdt ze binnen?",
            "maker": "Dan gaat elke brief naar de computers van de maker, en een gesloten model kan nergens anders draaien. Welke plek houdt ze binnen?",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Je ask, bij een local runtime

    Zodra oefening 1 goed is, richt de course je `ask` op zijn eigen server op je computer: een local runtime met Qwen3-0.6B. Er gaat niets van je computer af.
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
    question_box = mo.ui.text(value="Noem drie kleuren van de regenboog.", label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Verstuur hem met je ask")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_ask, question_box, send_button):
    _ask = C.learner_ask(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _ask is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo verstuurt je vraag met je eigen `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om je vraag te versturen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `ask` heeft de vraag naar de server op je computer gestuurd, die Qwen3 laadt …*"))
        _words = {"title": "Wat de local runtime antwoordde",
                  "done": "{seconds} seconden, via {url}, op je eigen computer.",
                  "failed": "Je ask stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het request is dat uit [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en het model dat je kiest door te meten, zoals in [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl), kan op elk van deze plekken draaien. Een host wil bij elk request een API key, en stuurt een rekening per token. De volgende lessen gaan verder dan tekst: beelden en geluid, via API's van dezelfde soort.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, result_ask, result_gb):
    L.wrap_up(result_ask, result_gb, choice_where)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
