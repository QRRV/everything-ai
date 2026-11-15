# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-choose-model. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-choose-model", "nl")
    C = L.checks   # de meting van de course van twee manieren op achttien vragen, de router, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_usage():
    quick_usage = L.choice(
        "quick-usage",
        question="Waar staat in de JSON `answer` van een server hoeveel tokens het model schreef?",
        options={
            "content": "In de lengte van `answer[\"choices\"][0][\"message\"][\"content\"]`",
            "finish": "In `answer[\"choices\"][0][\"finish_reason\"]`",
            "usage": "In `answer[\"usage\"][\"completion_tokens\"]`",
        },
        correct="usage",
        why={
            "usage": "`usage` telt de tokens: `prompt_tokens` gingen erin, `completion_tokens` kwamen eruit, en die rekent een gehoste API af.",
            "content": "De content is tekst: de lengte ervan telt tekens, geen tokens. Waar telt het antwoord tokens? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "finish": "`finish_reason` zegt waarom het model stopte, niet hoeveel het schreef. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_usage
    return (quick_usage,)


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="Wat zegt `\"model\"` in de body van een request?",
        options={
            "which": "Welk model op de server moet antwoorden",
            "long": "Hoe lang het antwoord mag zijn",
            "who": "Wie het request stuurt",
        },
        correct="which",
        why={
            "which": "Een server kan meer dan één model hebben; `\"model\"` kiest er een, met de naam die `/v1/models` noemt.",
            "long": "Dat is `\"max_tokens\"`. Wat kiest `\"model\"`? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "who": "Dat haalt de server niet uit `\"model\"`. Wat kiest het? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Honderden modellen antwoorden via een API, van klein en goedkoop tot groot en duur, en veel ervan kunnen ook eerst nadenken, wat langer duurt en meer kost. Het bekendste model is niet voor elke vraag de beste keus: een antwoord dat een klein model goed heeft, kost bij een groot model vele malen meer, en het duurt langer. Dus kies je voor je eigen taak: is het vaak genoeg goed, snel genoeg, en tegen welke prijs?

    **Een voorbeeld uit het dagelijks leven.** Je neemt geen taxi naar de bakker om de hoek, en je loopt niet naar het vliegveld. Hoe je het best reist, hangt af van de reis, niet van welke auto het snelst is. Dat is maar een vergelijking; hieronder zie je de echte metingen.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **thinking mode** | het model schrijft eerst zijn redenering uit, zoals in een chain of thought, en pas daarna het antwoord; de API geeft de redenering los van het antwoord, maar telt de tokens ervan als output tokens. Een model dat gemaakt is om zo te antwoorden, heet een *reasoning model* |
    | **LLM routing** | elke vraag naar het model sturen, of naar de manier van antwoorden, die erbij past: makkelijke naar een snel, goedkoop model, moeilijke naar een langzaam, sterk model |
    | **router** | het deel van je programma dat de routing doet: een functie die naar een vraag kijkt en teruggeeft welk model of welke manier hem krijgt |
    | **LLM gateway** | één adres voor veel modellen, bij verschillende bedrijven of op je eigen computers: je code roept altijd de gateway aan en noemt het model in `"model"`; de gateway geeft het request door, houdt de kosten bij, en kan naar een ander model overstappen als er een uitvalt |
    """)
    return


@app.cell(hide_code=True)
def text_measure():
    mo.md(C.measure_text(L.lang))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(C.measured_after(L.lang)),
        L.watch_out("Deze achttien vragen zijn die van de course, niet de jouwe: hoe een model het daarop doet, zegt weinig over jouw taak. Dat geldt ook voor een openbare ranglijst van modellen, een leaderboard: die meet de taken van die ranglijst. Meet op vragen zoals je eigen app ze gaat krijgen."),
    ])
    return


@app.cell(hide_code=True)
def text_routing():
    mo.md(C.routing_text(L.lang))
    return


@app.cell(hide_code=True)
def picture_router():
    mo.Html(C.router_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_gateway():
    mo.md(r"""
    Een router kan een regel zijn, zoals hier, of een klein model dat geleerd heeft makkelijke vragen van moeilijke te onderscheiden. Veel apps sturen hun requests via een gateway, één adres voor al hun modellen (voorbeelden in november 2026: LiteLLM, dat je zelf draait, en OpenRouter, een bedrijf dat je requests met één key doorgeeft aan de modellen van veel anderen). Dan hoeft de router alleen een naam voor `"model"` te kiezen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: kies een model

    1. **Verzamel je eigen voorbeelden:** twintig of meer vragen zoals je app ze gaat krijgen, elk met wat een goed antwoord bevat.
    2. **Beoordeel met code:** een functie die goed of fout zegt, voor elke kandidaat hetzelfde.
    3. **Stel elke kandidaat elke vraag een paar keer:** een model samplet, dus één run zegt weinig.
    4. **Tel per kandidaat:** hoe vaak hij goed was, zijn tokens en seconden, en wat een goed antwoord kost.
    5. **Kies:** de goedkoopste kandidaat die vaak genoeg goed is en snel genoeg; verschillen je vragen sterk, verdeel ze dan met een router.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text(L.lang))
    return


@app.cell(hide_code=True)
def text_cost():
    _e = C.example(L.lang)
    mo.vstack([
        mo.md(rf"""
        ## Zelf doen

        ### Oefening 1. Wat een goed antwoord kost

        Schrijf `cost_per_right(runs, price_out)`: `runs` is een lijst met runs, elk een dictionary met `"right"`, `True` of `False`, en `"tokens"`, de output tokens die hij gebruikte; `price_out` is de prijs in dollar per miljoen output tokens. Geef terug wat de tokens van alle runs kosten, gedeeld door het aantal goede runs; is geen run goed, geef dan `None` terug. Voor {_e["n"]} runs met {_e["tokens"]} tokens, waarvan die met {_e["wrong"]} fout was, tegen {_e["price"]} dollar per miljoen, is dat `{_e["want"]}`.
        """),
        L.task("Maak `cost_per_right` af: de tokens van alle runs, het aantal goede runs, en `None` als er geen goed is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost_per_right", lines=12)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost_per_right", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost_per_right",
        [
            "Tel twee dingen in één loop over `runs`: alle tokens, en hoeveel runs goed zijn.",
            "Begin met `tokens = 0` en `right = 0`. Tel in de loop `run[\"tokens\"]` op bij `tokens`, en 1 bij `right` als `run[\"right\"]` waar is. Geef na de loop `None` terug als `right == 0`.",
            "Aan het eind: `return tokens * price_out / 1_000_000 / right`.",
        ],
        why="""```python
def cost_per_right(runs, price_out):
    tokens = 0
    right = 0
    for run in runs:
        tokens = tokens + run["tokens"]
        if run["right"]:
            right = right + 1
    if right == 0:
        return None
    return tokens * price_out / 1_000_000 / right
```

Ook de foute antwoorden kosten tokens. Daarom kan een model dat goedkoop is per token, maar vaak fout, per goed antwoord meer kosten.""",
    )
    return


@app.cell(hide_code=True)
def text_route():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een router

        Schrijf `route(question)`: geef `"think"` terug als er een cijfer in de vraag staat, en `"fast"` als er geen in staat. Onder de check zie je wat je router met de achttien vragen van de course gedaan zou hebben, met de gemeten runs van elke manier.
        """),
        L.task("Maak `route` af: `\"think\"` voor een vraag met een cijfer, `\"fast\"` voor elke andere."),
    ])
    return


@app.cell(hide_code=True)
def exercise_route():
    box_route = L.code_box("route", lines=8)
    box_route
    return (box_route,)


@app.cell(hide_code=True)
def check_route(box_route):
    result_route = L.check("route", box_route.value)
    result_route
    return (result_route,)


@app.cell(hide_code=True)
def hints_route():
    L.hints(
        "route",
        [
            "Loop de vraag teken voor teken door, en stop bij het eerste cijfer.",
            "`for ch in question:` geeft de tekens een voor een, en `ch in \"0123456789\"` is `True` als `ch` een van de cijfers is. Dan `return \"think\"`.",
            "Na de loop, als er geen cijfer kwam: `return \"fast\"`.",
        ],
        why="""```python
def route(question):
    for ch in question:
        if ch in "0123456789":
            return "think"
    return "fast"
```

Een regel is de eenvoudigste router: snel, gratis, en af en toe fout, zoals bij een som die in woorden geschreven is. Of hij loont, meet je.""",
    )
    return


@app.cell(hide_code=True)
def text_support():
    mo.md(r"""
    ### Oefening 3. Welk model krijgt welke vraag?
    """)
    return


@app.cell(hide_code=True)
def choice_support_cell():
    choice_support = L.choice(
        "support",
        question="De chatbot van een webwinkel krijgt 10.000 vragen per dag. De meeste gaan over de openingstijden of waar een pakket is; een paar zijn lange klachten die zorg nodig hebben. Wat doe je?",
        options={
            "strong": "Elke vraag naar het sterkste model dat je kunt krijgen, voor de zekerheid",
            "route": "De korte, alledaagse vragen naar een klein, goedkoop model en de klachten naar een sterk model, en beide meten op echte vragen",
            "cheap": "Elke vraag naar het goedkoopste model; een klant met een fout antwoord vraagt het wel opnieuw",
        },
        correct="route",
        why={
            "route": "De alledaagse vragen zijn verreweg de meeste, en die doet een klein model voor een fractie van de prijs; de paar moeilijke krijgen het sterke model. Echte vragen laten zien of de router ze de goede kant op stuurt.",
            "strong": "Dan betalen 10.000 makkelijke vragen per dag de prijs van het sterkste model. Welke vragen hebben het nodig?",
            "cheap": "Dan krijgen de klachten de zwakste antwoorden, juist waar zorg het meest telt. Welke vragen hebben meer nodig?",
        },
    )
    choice_support
    return (choice_support,)


@app.cell(hide_code=True)
def text_try():
    mo.vstack([
        mo.md(r"""
        ### Je router voor Qwen3

        Zodra oefening 2 goed is, kiest je `route` de manier voor je vraag, en antwoordt de server van de course op je computer op die manier. Thinking kan op een laptop minuten duren: het model schrijft honderden tokens vóór het antwoord.
        """),
        mo.md(C.default_question(L.lang)),
    ])
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
    question_box = mo.ui.text(value="What is the capital of Japan?", label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Laat je router kiezen, en vraag het Qwen3")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_route, question_box, send_button):
    _route = C.learner_route(box_route.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _route is None:
        _out = mo.md("*Maak eerst oefening 2 af: de demo kiest de manier met je eigen `route`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om je vraag te sturen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `route` heeft een manier gekozen; de server laadt Qwen3, en het antwoord volgt …*"))
        _words = {"title_fast": "Qwen3, op de normale manier", "title_think": "Qwen3, in thinking mode",
                  "done_fast": "Je router koos de normale manier: {tokens} output tokens, {seconds} seconden.",
                  "done_think": "Je router koos thinking: {tokens} output tokens met de gedachte erbij, {seconds} seconden.",
                  "no_answer": "Je router koos thinking, en Qwen3 gebruikte alle {tokens} output tokens voor zijn gedachte, in {seconds} seconden: er kwam geen antwoord. Hierboven staat het eind van de gedachte.",
                  "other": "Je route gaf {way}, en de demo kent alleen \"fast\" en \"think\".",
                  "failed": "Je route stopte met een fout: {why}",
                  "server": "De server antwoordde niet ({why}). Druk opnieuw op de knop.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_route, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De aanroepen zijn die uit [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met één instelling meer in de body voor thinking, en de tokens die ze tellen, betaal je, zoals in [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl). Waar het model dat je kiest draait, op je eigen computer of bij een bedrijf, is de volgende les: [Waar een model draait](/course/open/build-where-models-run?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_support, result_cost, result_route):
    L.wrap_up(result_cost, result_route, choice_support)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
