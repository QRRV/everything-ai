# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-model-specific. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-model-specific", "nl")
    C = L.checks   # de opgenomen runs van de course, de checks, en de demo met je run_examples en worse


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_versions():
    quick_versions = L.choice(
        "quick-versions",
        question="Je hebt één zin in een prompt template veranderd. Hoe kom je erachter of de nieuwe versie beter is?",
        options={
            "examples": "Draai dezelfde voorbeelden met beide versies, en vergelijk wat er terugkomt",
            "longer": "De nieuwe versie zegt meer, dus het model heeft meer houvast en antwoordt beter",
            "ask": "Vraag het model welke versie het het beste vindt",
        },
        correct="examples",
        why={
            "examples": "Alleen de antwoorden op je eigen voorbeelden laten zien wat een versie doet, met dit model.",
            "longer": "Meer woorden kunnen helpen of schaden; bij de template voor fietsen van de course hielp één zin meer, maar dat lieten pas de runs zien. Zie [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl).",
            "ask": "Wat een model over een prompt zegt, is geen meting van wat het ermee doet. Zie [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl).",
        },
    )
    quick_versions
    return (quick_versions,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een prompt groeit op één model: je probeert woorden tot de antwoorden op je voorbeelden kloppen. Dan verandert het model. Er komt een goedkoper model uit, de aanbieder zet een nieuwe versie achter dezelfde naam, of je verhuist hetzelfde model naar een andere server. De prompt blijft hetzelfde, en toch gaan antwoorden die goed waren nu fout, omdat elk model zijn eigen gewoontes en zijn eigen instellingen heeft. Verhuizen naar een nieuw model hoort bij het werk, en het gaat zoals elke verandering aan een prompt: met je voorbeelden.

    **Een voorbeeld uit het dagelijks leven.** Een recept voor een taart die in je eigen oven altijd lukt, kan in de oven van een vriend verbranden: hetzelfde recept, dezelfde stappen, een andere oven. Een ervaren bakker leest eerst de oven van de vriend, en bakt één proeftaart. Dat is maar een vergelijking; hieronder meet je hoe één prompt het doet bij vier manieren om hetzelfde model te draaien.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **prompt migration** | een prompt naar een ander model, een andere versie of andere instellingen verhuizen, en hem aanpassen tot je voorbeelden weer kloppen |
    | **prompting guide** | wat de maker van een model ervoor aanraadt: hoe je prompts schrijft, welke instellingen je meestuurt, en welke schakelaars het kent |
    | **/no_think** | een schakelaar van Qwen3 aan het eind van een bericht: niet denken voor dit antwoord, ook als de server thinking heeft aangezet |

    ### Eén prompt, vier manieren om het model te draaien

    Qwen3 heeft twee modes, zoals je zag in [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl): het antwoordt meteen, of het denkt eerst. De server van de course antwoordt meteen, tenzij een request om thinking vraagt. De eigen chat template van Qwen3, en de servers vLLM en SGLang, denken tenzij je zegt dat het niet moet. Een app die tegen de server van de course gebouwd is en naar zo'n server verhuist, krijgt dus hetzelfde model met dezelfde prompt, en toch een andere manier van antwoorden.

    De course vroeg Qwen3 om de afspraak in drie berichten als JSON, met de system prompt en template hieronder, elk {seeds} keer, op vier manieren: op de gewone manier met `max_tokens` 80, met thinking aan en dezelfde 80 tokens, met thinking en 1000 tokens, en met thinking en `/no_think` aan het eind van elk bericht.

    ```text
    {system}

    {event}
    ```
    """.format(seeds=_f["seeds"], system=C.SYSTEM, event=C.EVENT))
    return


@app.cell(hide_code=True)
def picture_modes():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_modes():
    _f = C.findings()
    _n = _f["seeds"] * _f["examples"]
    mo.vstack([
        mo.md(r"""
        Op de gewone manier waren {normal} van de {n} antwoorden goed. Met thinking aan en dezelfde instellingen was er geen enkel goed: alle {cut} hadden hun 80 tokens op terwijl Qwen3 nog aan het denken was, dus het antwoord zelf bleef leeg. De gedachte hoort niet bij het antwoord, dus een app die alleen het antwoord leest, krijgt helemaal niets:
        """.format(normal=_f["normal"]["all"], n=_n, cut=_f["think-short"]["cut"])),
        mo.Html(C.answer_picture(L.lang, "think-short")),
        mo.md(r"""
        Met 1000 tokens was er ruimte, en toch waren maar {think} van de {n} goed. Voor *7 pm* in het bericht over voetbal gaf elke run met thinking het uur {think_hour}, waar de gewone manier elke keer {normal_hour} gaf; en een antwoord kostte tot {think_tokens} tokens, de gedachte meegerekend, tegen {normal_tokens} op de gewone manier. De prompting guide van Qwen3, zijn model card op Hugging Face, de pagina waarop de maker het model beschrijft, noemt de andere uitweg: `/no_think` aan het eind van een bericht. Daarmee waren {no_think} van de {n} goed, in tot {no_think_tokens} tokens: evenveel als vóór de verhuizing.

        De leesclub ging op elke manier fout: waar een antwoord een uur gaf, was het {book}, voor *3 in the afternoon*. Dat heeft de verhuizing niet kapotgemaakt. Het was eerst al fout, dus het hoort niet bij wat slechter werd; het vraagt om een eigen verandering in de template.
        """.format(think=_f["think"]["all"], n=_n, think_hour=_f["think"]["hours"]["football"][0], normal_hour=_f["normal"]["hours"]["football"][0],
                   think_tokens=_f["think"]["tokens"], normal_tokens=_f["normal"]["tokens"],
                   no_think=_f["no-think"]["all"], no_think_tokens=_f["no-think"]["tokens"], book=_f["normal"]["hours"]["book-club"][0])),
        L.watch_out("Elke maker schrijft zijn eigen prompting guide (voorbeelden in november 2026). OpenAI en Anthropic publiceren er een voor hun modellen, met adviezen die per generatie veranderen, en de model card van Qwen3 noemt de instellingen voor sampling en de schakelaars `/think` en `/no_think`. Wat bij het ene model werkte, is een begin voor het volgende, geen belofte."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een prompt naar een ander model verhuizen

    1. **Houd je voorbeelden:** inputs met wat een goed antwoord is, dezelfde waarmee je elke versie van een template test.
    2. **Lees de guide van het nieuwe model:** welke instellingen het wil, welke schakelaars het kent, en in welke vorm het instructies verwacht.
    3. **Draai elk voorbeeld met het oude en het nieuwe,** en maak een lijst van de voorbeelden die eerst goed waren en nu fout zijn.
    4. **Pas de prompt of de instellingen aan** voor het nieuwe model, en draai de voorbeelden opnieuw, tot die lijst leeg is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _n = _f["seeds"] * _f["examples"]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De app verhuist naar een server die denkt, tenzij je zegt dat het niet moet. Wat moet er veranderen?

    1. **De voorbeelden:** de drie berichten, elk met zijn dag en uur; een run telt als goed als de JSON in het antwoord die dag en dat uur heeft.
    2. **De guide:** in thinking mode schrijft Qwen3 eerst zijn redenering uit, binnen de tokens van het antwoord, en `/no_think` aan het eind van een bericht zet dat uit.
    3. **Oud en nieuw:** op de gewone manier {normal} van de {n} goed, met thinking en 80 tokens {short}: elk voorbeeld dat goed was, werd slechter.
    4. **Aanpassen:** een grotere `max_tokens` gaf ruimte, maar er waren er maar {think} van de {n} goed, voor tot {think_tokens} tokens per antwoord. Met `/no_think` in de template waren {no_think} van de {n} goed, voor tot {no_think_tokens}: de lijst van wat slechter werd, is weer leeg, voor één woord in de template. De leesclub blijft fout, zoals hij vóór de verhuizing was.
    """.format(normal=_f["normal"]["all"], n=_n, short=_f["think-short"]["all"], think=_f["think"]["all"],
               think_tokens=_f["think"]["tokens"], no_think=_f["no-think"]["all"], no_think_tokens=_f["no-think"]["tokens"]))
    return


@app.cell(hide_code=True)
def text_worse():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat slechter werd

        Na een run met het oude en met het nieuwe model heb je twee dictionaries met per voorbeeld of zijn antwoord goed was: `True` of `False`. Schrijf `worse(old, new)`, die een lijst teruggeeft van de namen die in `old` goed waren en in `new` fout zijn, in de volgorde van `old`. Het vak geeft elke naam terug. Twee woorden Python helpen: `not x` is `True` als `x` `False` is, en andersom; `a and b` is alleen `True` als `a` en `b` allebei `True` zijn.
        """),
        L.task("Houd alleen de namen over die in `old` goed waren en in `new` fout zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worse():
    box_worse = L.code_box("worse", lines=7)
    box_worse
    return (box_worse,)


@app.cell(hide_code=True)
def check_worse(box_worse):
    result_worse = L.check("worse", box_worse.value)
    result_worse
    return (result_worse,)


@app.cell(hide_code=True)
def hints_worse():
    L.hints(
        "worse",
        [
            "Een naam hoort in de lijst als twee dingen tegelijk gelden: hij was eerst goed, en hij is nu fout.",
            "Zet een `if` om de `append`, met twee tests verbonden door `and`: een op `old[name]` en een op `new[name]`.",
            "In de loop: `if old[name] and ___:`, met `names.append(name)` een stap verder in.",
        ],
        why="""```python
def worse(old, new):
    names = []
    for name in old:
        if old[name] and not new[name]:
            names.append(name)
    return names
```

Een voorbeeld dat in `old` al fout was, werd niet slechter, en een dat van fout naar goed ging, werd beter; geen van beide hoort in de lijst.""",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Elk voorbeeld, met gegeven settings

        Om te vergelijken moet elk voorbeeld op dezelfde manier de deur uit, met alleen de settings anders. Het vak stuurt elke tekst in `texts`, een dictionary van een naam naar een bericht, met `messages_for(text)`, en bewaart het antwoord zelf. Verander `run_examples(url, settings, texts)` op twee plaatsen. Zet eerst elke key van `settings` met zijn waarde in `body`, na de regel die de body maakt, zodat een setting als `max_tokens` de 80 vervangt. Bewaar daarna `ok(name, ...)` van het antwoord in plaats van het antwoord zelf: `True` als de JSON erin de goede dag en het goede uur heeft. `messages_for` en `ok` zijn gegeven. De check draait je functie tegen een stand-in, ook met thinking aan.
        """),
        L.task("Zet elke setting in de body, en bewaar `ok(name, ...)` van elk antwoord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run_examples", lines=18)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run_examples", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run_examples",
        [
            "Een dictionary krijgt een nieuwe key, of een nieuwe waarde voor een key die hij al heeft, met `body[key] = value`. Doe dat voor elke key in `settings`.",
            "Direct na de regel met `body = ...`: een `for`-loop over `settings` die elke key met zijn waarde in `body` zet. Zet aan het eind `ok(name, ...)` om de tekst van het antwoord die het vak nu bewaart.",
            "Onder `body = ...`: `for key in settings:`, met `body[key] = ___` een stap verder in.",
        ],
        why="""```python
import json
import urllib.request


def run_examples(url, settings, texts):
    results = {}
    for name in texts:
        body = {"model": "qwen3-0.6b", "messages": messages_for(texts[name]), "max_tokens": 80}
        for key in settings:
            body[key] = settings[key]
        data = json.dumps(body).encode()
        request = urllib.request.Request(url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            answer = json.loads(response.read())
        results[name] = ok(name, answer["choices"][0]["message"]["content"])
    return results
```

De settings komen erin nadat de body gemaakt is, dus ze winnen van wat de body had: `{"max_tokens": 1000}` vervangt de 80.""",
    )
    return


@app.cell(hide_code=True)
def text_migrate():
    mo.md(r"""
    ### Oefening 3. Een nieuwe versie achter dezelfde naam
    """)
    return


@app.cell(hide_code=True)
def choice_migrate_cell():
    choice_migrate = L.choice(
        "migrate",
        question="Je aanbieder zet volgende maand een nieuwere versie van het model achter de naam die je app gebruikt. Wat doe je eerst?",
        options={
            "nothing": "Niets: een nieuwere versie van hetzelfde model is beter in alles wat de oude kon",
            "longer": "Elke prompt langer maken, zodat geen versie hem verkeerd leest",
            "run": "Je voorbeelden tegen de nieuwe versie draaien, en aanpassen wat slechter werd",
        },
        correct="run",
        why={
            "run": "Alleen je voorbeelden laten zien wat de nieuwe versie met je prompts doet; de lijst van wat slechter werd, zegt wat je moet aanpassen.",
            "nothing": "Een nieuwe versie kan gemiddeld beter zijn en toch slechter op jouw voorbeelden: dezelfde prompt kan andere antwoorden krijgen. Draai ze eerst.",
            "longer": "Meer woorden kunnen helpen of schaden, en zonder run weet je niet welke van de twee. Draai eerst je voorbeelden tegen de nieuwe versie.",
        },
    )
    choice_migrate
    return (choice_migrate,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 en 2 goed zijn, kun je de drie voorbeelden op het echte model draaien: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo draait ze met je `run_examples`, eerst op de gewone manier en dan op de manier die je kiest, en laat met je `worse` zien wat slechter werd. Je computer is niet die van de course, dus je antwoorden kunnen verschillen.
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
def mode_cell():
    mode_box = mo.ui.dropdown(options={"thinking aan, max_tokens 80": "think-short", "thinking aan, met /no_think in elk bericht": "no-think"},
                              value="thinking aan, max_tokens 80", label="De nieuwe manier")
    run_button = mo.ui.run_button(label="Draai met je run_examples")
    mo.vstack([mode_box, run_button])
    return mode_box, run_button


@app.cell(hide_code=True)
def demo_answer(box_run, box_worse, mode_box, run_button):
    _fns = C.learner_functions(box_worse.value, box_run.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fns is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo draait je eigen `run_examples` en `worse`.*")
    elif not run_button.value:
        _out = mo.md("*Druk op de knop om te draaien.*")
    else:
        mo.output.replace(mo.md("*De voorbeelden gaan twee keer de deur uit, op de gewone en op de nieuwe manier; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Je voorbeelden, op de gewone en op de nieuwe manier", "old": "gewoon", "new": "nieuw",
                  "right": "goed", "wrong": "fout",
                  "lost": "Je worse: {names}, in {seconds} seconden.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fns[0], _fns[1], mode_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De voorbeelden en versies van [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl) zijn waarmee je een prompt verhuist, en de thinking mode van [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl) is een van de instellingen waardoor een model anders antwoordt. [Prompts die standhouden](/course/open/build-defensive-prompts?lang=nl) kijkt naar tekst in een prompt die eigen instructies probeert te geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_migrate, result_run, result_worse):
    L.wrap_up(result_worse, result_run, choice_migrate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
