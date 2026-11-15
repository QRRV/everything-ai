# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-prompt-templates. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-templates", "nl")
    C = L.checks   # de opgenomen runs van de course met twee versies, en de demo met een eigen versie


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="Je app voegt aan elk gesprek dezelfde instructies toe. Waar horen ze?",
        options={
            "first": "In een eerste bericht met de rol `system`",
            "user": "Aan het eind van elk bericht dat de gebruiker typt",
            "prefill": "In een response prefill, na de vraag",
        },
        correct="first",
        why={
            "first": "De system prompt geldt voor het hele gesprek: de app voegt hem toe, en de gebruiker ziet hem niet.",
            "user": "Dan lopen de woorden van de gebruiker en je instructies door elkaar, in elk bericht. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
            "prefill": "Een prefill schrijft de eerste woorden van het antwoord; hij geeft geen instructies voor het hele gesprek. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In een app typt niemand de prompt: code bouwt hem voor elk request, rond wat de gebruiker stuurde. En de woorden van een prompt veranderen. Je voegt een zin toe omdat een antwoord misging, en een week later zijn de antwoorden slechter, of beter: dan wil je weten welke woorden welke antwoorden gaven. Dat lukt als een prompt geen string ergens in je code is, maar een template: vaste woorden met velden die code invult, in een eigen bestand, met een versienummer dat bij elk antwoord in het log komt.

    **Een voorbeeld uit het dagelijks leven.** Een brief van je bank heeft voor iedereen dezelfde tekst, met jouw naam en jouw bedrag ingevuld. Verandert de bank een zin, dan is dat een nieuwe versie van de brief, en weet de bank wie welke versie kreeg. Dat is maar een vergelijking; hieronder zie je twee versies van één prompt, en wat elk gaf.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **prompt template** | een prompt met velden, zoals `{text}`, die code voor elk request invult, zodat elke aanroep dezelfde geteste woorden gebruikt |
    | **promptversie** | één vaste formulering van een template, met een eigen nummer; elke verandering aan de woorden krijgt een nieuw nummer, en bij elk antwoord wordt vastgelegd welke versie het maakte |
    | **.format(...)** | vult de velden van een tekst in: `"Hi {name}".format(name="Sam")` geeft `"Hi Sam"`; een accolade die in de tekst moet blijven, schrijf je dubbel, `{{` of `}}` |

    ### Een template met een veld

    De vraag van de course over een fiets, als template: de vaste woorden, en het veld `{text}` voor de tekst over de fiets. `.format(text=...)` maakt er voor elke tekst een prompt van. De vraag is Engels, omdat Qwen3 Engels het best schrijft:
    """)
    return


@app.cell(hide_code=True)
def picture_fill():
    mo.Html(C.fill_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_fill():
    mo.md(r"""
    In code:

    ```python
    TEMPLATE = "Give the name, color and price of this bike as JSON, with the keys name, color and price.\n\n{text}"
    prompt = TEMPLATE.format(text="The Trailhawk is a green mountain bike with 21 gears. Price: 1199 euros.")
    ```

    De tekst gaat erin zoals hij is. Staan er zelf accolades in, dan blijven die staan: `.format` leest alleen de accolades van de template.

    ### Twee versies

    De course stuurde deze template, versie 1, acht keer voor elk van de drie teksten, met **JSON mode**: de server laat het model dan alleen geldige JSON schrijven, wat nog niets zegt over welke keys erin staan. Een antwoord telt als passend als `json.loads` het leest, met `name`, `color` en `price` erin, en de prijs een getal. Versie 2 voegt één zin toe: *The price is a number, without quotes, such as {"price": 100}.* Het log van beide versies:
    """)
    return


@app.cell(hide_code=True)
def picture_log():
    mo.Html(C.log_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_log():
    _f = C.findings()
    mo.md(r"""
    Versie 1: {v1} van de {k} antwoorden passen, en in elk van de {wrong} andere was de prijs een tekst, zoals `"649"`. Versie 2: {v2} van de {k}. Zonder de versie in het log zouden dit {all} antwoorden van één prompt zijn, en was het verschil weg. Acht runs van drie teksten is niet veel: versie 2 is hier beter, voor deze teksten en dit model. Een ander model, of andere teksten, kan andere woorden nodig hebben: daarom krijgt elke verandering een eigen nummer, zodat je dezelfde voorbeelden opnieuw kunt draaien en vergelijken.

    ### Accolades in een template

    Versie 2 heeft eigen accolades, in het voorbeeld. Voor `.format` is elk paar accolades een veld. Enkel geschreven zou `{{"price": 100}}` een veld zijn met de naam `"price"` (`.format` leest wat na de dubbele punt komt als een opmaak), en `.format` stopt met een KeyError, de fout van Python voor een key zonder waarde: voor dat veld kwam geen waarde. Een accolade die in de prompt moet blijven, schrijf je dubbel. De template hieronder staat over twee regels: twee strings naast elkaar tussen haakjes zijn voor Python één string.

    ```python
    TEMPLATE_V2 = ("Give the name, color and price of this bike as JSON, with the keys name, color and price. "
                   'The price is a number, without quotes, such as {{{{"price": 100}}}}.\n\n{{text}}')
    ```

    Van `{{{{"price": 100}}}}` maakt `.format` `{{"price": 100}}`, en het vult `{{text}}` in.
    """.format(v1=_f["total"][1], v2=_f["total"][2], k=_f["asked"][1], wrong=_f["wrong"], all=_f["asked"][1] + _f["asked"][2]))
    return


@app.cell(hide_code=True)
def text_file():
    mo.vstack([
        mo.md(r"""
        ### In een bestand, met een versie

        Een template hoort in een eigen bestand, niet midden in je code: dan lees je al zijn woorden in één keer, en bewaart versiebeheer, zoals Git, elke versie van het bestand. Eén manier is een JSON-bestand voor elke template:

        ```json
        {"name": "bike-details", "version": 2,
         "template": "Give the name, color and price of this bike as JSON, ... such as {{\"price\": 100}}.\n\n{text}"}
        ```

        Je code leest het met `json.load`, vult de template in, en zet de naam en de versie in het log, naast elk antwoord. Verander je een woord, dan gaat de versie omhoog: de woorden van een versie die antwoorden heeft gegeven, veranderen nooit meer.
        """),
        L.watch_out("Libraries helpen met templates. Jinja2 heeft loops en voorwaarden in een template, en tools zoals Langfuse en MLflow bewaren de versies van een prompt, met labels zoals *production* (voorbeelden in november 2026). Het idee blijft hetzelfde: vaste woorden, velden, en een versie die bij elk antwoord hoort."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een prompt als code

    1. **Vaste woorden, velden voor de rest:** schrijf de prompt als template, met een veld voor elk deel dat verandert, zoals `{text}`.
    2. **Verdubbel de eigen accolades van de template:** `{{` en `}}` voor elke accolade die in de prompt moet blijven.
    3. **Een eigen bestand, met een naam en een versie:** elke verandering aan de woorden is een nieuwe versie, en versiebeheer bewaart de oude.
    4. **Leg bij elk antwoord de versie vast:** dan kun je voor elke versie tellen wat goed ging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _a = C.answers(2, "pixie")[0]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Versie 2, ingevuld met de tekst over de Pixie, en de eerste run ervan.

    1. **Vaste woorden en een veld:** versie 2 van de template, met als `text` de tekst over de Pixie: `{text}`
    2. **Accolades:** `.format` maakt van `{{{{"price": 100}}}}` `{{"price": 100}}`, en zet de tekst in het veld, dus de prompt eindigt met `such as {{"price": 100}}.`, een lege regel, en de tekst.
    3. **Naam en versie:** `bike-details`, versie 2.
    4. **Het log:** met seed 0 antwoordde Qwen3 `{answer}`. Dat past, dus de regel in het log is `{{"prompt": "bike-details", "version": 2, "text": "pixie", "seed": 0, "fits": true}}`. De naam erin is *{name}*, niet *Pixie*: `fits` controleert de keys en de types, niet of elke waarde klopt.
    """.format(text=C.TEXTS["pixie"], answer=" ".join(_a["text"].split()), name=C.value(_a["text"])["name"]))
    return


@app.cell(hide_code=True)
def text_prompt():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Accolades in de template

        In het vak staat versie 2 van de template voor fietsen, maar met de accolades van het voorbeeld enkel geschreven, dus `bike_prompt` stopt met een KeyError. Maak de template goed, zodat `bike_prompt(text)` de prompt geeft met het voorbeeld erin, en de tekst in het veld.
        """),
        L.task("Schrijf elke accolade van het voorbeeld in `TEMPLATE` dubbel, en laat het veld `{text}` zoals het is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("bike_prompt", lines=7)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("bike_prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "bike_prompt",
        [
            "`.format` leest elk paar accolades als een veld. Welke accolades in `TEMPLATE` zijn een veld, en welke moeten in de prompt blijven?",
            "`{text}` is het veld. De accolades rond `\"price\": 100` moeten blijven, dus elk ervan schrijf je dubbel.",
            "Het voorbeeld wordt `{{\"price\": 100}}`; de rest van de template blijft zoals hij is.",
        ],
        why="""```python
TEMPLATE = ("Give the name, color and price of this bike as JSON, with the keys name, color and price. "
            'The price is a number, without quotes, such as {{"price": 100}}.\\n\\n{text}')
```

`.format` maakt van elke dubbele accolade een enkele, en vult het veld met enkele accolades in.""",
    )
    return


@app.cell(hide_code=True)
def text_tally():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat paste, per versie

        Het log heeft voor elk antwoord een dictionary, met de `"version"` van de template, en `"fits"`: `True` als het antwoord past. Schrijf `fits_per_version(log)`, die teruggeeft hoeveel antwoorden van elke versie in het log passen, zoals `{1: 1, 2: 2}`. Een versie waarvan geen enkel antwoord past, krijgt 0, zodat je ziet dat hij geprobeerd is. `version not in counts` is `True` zolang `counts` nog geen key `version` heeft. Het plaatje onder de check laat zien wat je functie telt in het log van de course.
        """),
        L.task("Geef in de loop een versie die nog niet in `counts` staat de telling 0, en tel er 1 bij op voor elk antwoord waarvan `entry[\"fits\"]` `True` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tally():
    box_tally = L.code_box("fits_per_version", lines=10)
    box_tally
    return (box_tally,)


@app.cell(hide_code=True)
def check_tally(box_tally):
    result_tally = L.check("fits_per_version", box_tally.value)
    result_tally
    return (result_tally,)


@app.cell(hide_code=True)
def hints_tally():
    L.hints(
        "fits_per_version",
        [
            "Twee stappen voor elke regel: zorg dat zijn versie in `counts` staat, en tel er 1 bij op als het antwoord past.",
            "`if version not in counts:` geeft de versie zijn eerste telling, 0. Daarna telt `if entry[\"fits\"]:` er 1 bij op in `counts[version]`.",
            "`if version not in counts:` met daaronder `counts[version] = 0`, en dan `if entry[\"fits\"]:` met daaronder `counts[version] = counts[version] + 1`.",
        ],
        why="""```python
def fits_per_version(log):
    counts = {}
    for entry in log:
        version = entry["version"]
        if version not in counts:
            counts[version] = 0
        if entry["fits"]:
            counts[version] = counts[version] + 1
    return counts
```

Elke versie krijgt zijn telling de eerste keer dat hij langskomt, ook als geen van zijn antwoorden past.""",
    )
    return


@app.cell(hide_code=True)
def text_new_version():
    mo.md(r"""
    ### Oefening 3. Wanneer een nieuwe versie?
    """)
    return


@app.cell(hide_code=True)
def choice_new_version_cell():
    _t, _k = C.findings()["total"], C.findings()["asked"][1]
    choice_new_version = L.choice(
        "new-version",
        question="Welke verandering aan de template `bike-details` krijgt een nieuw versienummer?",
        options={
            "longer": "Alleen een verandering die de prompt duidelijk langer of korter maakt",
            "comma": "Elke verandering aan de woorden, zelfs één komma",
            "field": "Alleen een nieuw veld, zoals `{shop}` naast `{text}`",
        },
        correct="comma",
        why={
            "comma": "Eén zin bracht de antwoorden van de course van {v1} naar {v2} van de {k} die passen, en ook een komma kan antwoorden veranderen. Met een nieuw nummer voor elke verandering zegt het log welke woorden welke antwoorden gaven.".format(v1=_t[1], v2=_t[2], k=_k),
            "longer": "Versie 2 verschilt één zin van versie 1, en gaf andere antwoorden: een kleine verandering kan ertoe doen. Welke veranderingen kunnen de antwoorden veranderen?",
            "field": "Versie 1 en 2 hebben hetzelfde veld, en gaven andere antwoorden: alleen de woorden maakten het verschil. Welke veranderingen kunnen de antwoorden veranderen?",
        },
    )
    choice_new_version
    return (choice_new_version,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer een eigen versie op Qwen3

    Schrijf een versie 3, en stuur hem naar Qwen3-0.6B op de eigen server van de course. De demo vult je template in met elk van de drie teksten, met `.format`, en vraagt voor elk één keer, met seed 0, met JSON mode. Let op de accolades, en schrijf in het Engels, omdat Qwen3 Engels het best schrijft.
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
def template_box_cell():
    template_box = mo.ui.text_area(value=C.TEMPLATES[2], label="Je template, met het veld {text}", full_width=True, rows=4)
    send_button = mo.ui.run_button(label="Stuur voor de drie fietsen")
    mo.vstack([template_box, send_button])
    return send_button, template_box


@app.cell(hide_code=True)
def demo_answer(send_button, template_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te versturen.*")
    else:
        mo.output.replace(mo.md("*De prompts zijn verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat Qwen3 antwoordde op je versie", "fits": "past", "not": "past niet",
                  "format": ".format stopte: {why}. Schrijf elke accolade die in de prompt moet blijven dubbel.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(template_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De system prompt van [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) is ook vaste tekst, en hoort in een bestand naast de templates, met een eigen versie. Voor elke versie tellen wat past, op dezelfde voorbeelden, is wat een eval op grotere schaal doet, verderop in dit domein. In de volgende les, [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl), krijgt je assistent een eigen system prompt en templates, en antwoordt hij met JSON die hij zelf controleert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_version, result_prompt, result_tally):
    L.wrap_up(result_prompt, result_tally, choice_new_version)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
