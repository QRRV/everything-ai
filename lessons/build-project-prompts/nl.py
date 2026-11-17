# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-project-prompts. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-prompts", "nl")
    C = L.checks   # the course's runs with and without the rule about JSON, the stand-ins, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl), en één over [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="Waarom krijgt een template in een programma een versie, die bij elk antwoord in het log staat?",
        options={
            "answers": "Om te weten welke woorden welke antwoorden gaven",
            "model": "Zodat het model weet welke versie het leest",
            "format": "Omdat `.format` alleen een template met een versie invult",
        },
        correct="answers",
        why={
            "answers": "Verander een woord en de versie gaat omhoog, dus het log laat per versie zien wat er goed ging.",
            "model": "Het model ziet de versie nooit: die staat niet in de prompt. Zie [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl).",
            "format": "`.format` vult de velden van elke string in. Zie [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl).",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def quick_check_again():
    quick_again = L.choice(
        "quick-again",
        question="Een antwoord past niet bij het schema. Wat zit er in het volgende request van `ask_json`?",
        options={
            "same": "Dezelfde berichten als daarvoor, zodat het model het nog eens probeert",
            "both": "De berichten tot nu toe, het antwoord, en de problemen",
            "problems": "Alleen een bericht dat de problemen noemt",
        },
        correct="both",
        why={
            "both": "Het model onthoudt niets tussen requests: het heeft zijn eigen antwoord nodig om te weten waar de problemen over gaan.",
            "same": "Dan hoort het model niets over wat er mis was. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
            "problems": "Zonder zijn eigen antwoord weet het model niet waar de problemen over gaan. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
        },
    )
    quick_again
    return (quick_again,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je assistent uit [Project: je assistent antwoordt](/course/open/build-project-call?lang=nl) stuurt wat de gebruiker typt, en geeft tekst terug. Voor een chat is dat genoeg. Maar een app doet meer met een antwoord dan het laten zien: een agenda heeft de dag en het uur van een afspraak nodig als waarden waarmee hij kan rekenen, niet als zin. Sindsdien heb je geleerd wat daarvoor nodig is: een system prompt met regels voor elke taak, uit [System prompts en response prefill](/course/open/build-system-prompt?lang=nl), templates met een versie, en JSON die je code controleert. Vandaag gaat het allemaal je project in: je assistent krijgt zijn prompts in een eigen bestand, en een functie die JSON teruggeeft die hij zelf heeft gecontroleerd.

        **Een voorbeeld uit het dagelijks leven.** Een nieuwe collega krijgt op de eerste dag een handboek, met de regels voor elke brief. Voor elk soort brief is er een formulier met lege plekken om in te vullen. En voordat een brief de deur uitgaat, controleert iemand hem aan de hand van een lijst. Je assistent krijgt dezelfde drie dingen: de system prompt is het handboek, de templates zijn de formulieren, en het schema is de lijst. Dat is maar een vergelijking; hieronder zie je wat elk ervan doet bij Qwen3.
        """),
        draw.project_steps(["Je assistent antwoordt", "Je assistent, goed geprompt", "Je assistent leest je documenten"], 1,
                           here_label="← vandaag", title="Waar je project staat: je assistent"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_rule():
    _f = C.findings()
    guess_rule = L.guess(
        "rule",
        question="De course vroeg Qwen3-0.6B om de afspraak in een bericht als JSON, acht keer voor elk van drie berichten, met de system prompt *You are a friendly assistant. Answer briefly.* Daarna nog 24 keer, met één zin erbij in de system prompt: *When a task asks for JSON, answer with the JSON only, without a code block.* Wat deed die zin?",
        options={
            "same": "Weinig: de template vraagt al om JSON",
            "more": "Meer antwoorden die `json.loads` kan lezen",
            "fewer": "Minder: een langere system prompt brengt een klein model in de war",
        },
        reveal="**Meer antwoorden die `json.loads` kan lezen: {rule} van de {k} bij de eerste poging, tegen {plain} van de {k} zonder.** Zonder de zin stonden de {fenced} andere allemaal in een codeblok, dat `json.loads` niet kan lezen. Opnieuw vragen met het probleem hielp ze niet: in elk van die {fenced} runs stonden de tweede en de derde poging ook in een codeblok. Een regel in de system prompt werkte waar opnieuw vragen dat niet deed.".format(
            rule=_f["rule"]["first_all"], plain=_f["plain"]["first_all"], k=_f["rule"]["runs"], fenced=_f["plain"]["fenced"]),
    )
    guess_rule
    return (guess_rule,)


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Vier bestanden, elk met één taak

        Na vandaag heeft je package `app` vier bestanden. Elk doet één ding, en alleen dat:

        - `app/client.py`, uit [Project: je assistent antwoordt](/course/open/build-project-call?lang=nl), praat met de server: `chat` stuurt een gesprek en geeft de tekst terug.
        - `app/prompts.py`, nieuw vandaag, bevat de woorden: `SYSTEM`, de system prompt met de regels voor elke taak, en `TEMPLATES`, een ingang voor elke taak, met de `"text"` van de template, zijn `"version"` en het `"schema"` van het antwoord waar hij om vraagt. Zijn `messages_for(name, text)` maakt de eerste berichten van een taak.
        - `app/schema.py` bevat je `problems` uit [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl): de check van die les heeft hem als bestand bewaard.
        - `app/answers.py`, nieuw vandaag, heeft `ask_json(url, name, text)`: die vraagt met de prompts via de client, controleert het antwoord, en vraagt opnieuw met wat er mis was.
        """),
        mo.Html(C.files_picture(L.lang)),
        mo.md(r"""
        De rest van een app roept alleen `ask_json` aan, en krijgt een dictionary, of `None` als geen antwoord paste. Betere woorden voor een prompt veranderen alleen `app/prompts.py`, en de versie gaat omhoog; een andere server verandert alleen `app/client.py`. Het schema staat naast zijn template, omdat ze bij elkaar horen: een template die om andere keys vraagt, heeft een ander schema nodig. En de regel over JSON staat in de system prompt, niet in elke template, omdat hij voor elke taak geldt: de titel vraagt ook om JSON.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_meaning():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        ### Een controle van de vorm ziet de betekenis niet

        Het derde bericht zegt dat de leesclub *on Sunday at 3 in the afternoon* bijeenkomt: uur 15. Met de regel pasten alle {k} antwoorden bij de eerste poging bij het schema, en alle {k} gaven uur {hour}:
        """.format(k=len(_f["rule"]["hours"]), hour=_f["rule"]["hours"][0])),
        mo.Html(C.book_picture(L.lang)),
        mo.md(r"""
        `3` is een getal van 0 tot en met 23, dus `problems` vindt niets. Die controleert de vorm van een antwoord, niet of de waarden kloppen. Een zin in de template kan helpen, zoals een die zegt dat het uur in de 24-uursnotatie staat; dat is een nieuwe versie van de template. Of de waarden kloppen, weet je alleen door antwoorden te vergelijken met antwoorden waarvan je weet dat ze goed zijn, en dat doet een latere les.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: je assistent, goed geprompt

    1. **De woorden in `app/prompts.py`:** `SYSTEM` met de regels voor elke taak, en `TEMPLATES` met voor elke taak de `"version"` van de template, zijn `"text"` met het veld `{text}`, en het `"schema"` van het antwoord.
    2. **De eerste berichten:** `messages_for(name, text)` geeft de system prompt terug als `{"role": "system", ...}`, en daarna de template van `name`, ingevuld met `.format(text=text)`, als bericht van de gebruiker.
    3. **Vragen in `app/answers.py`:** `ask_json(url, name, text, tries=3)` begint met `messages_for(name, text)` en het schema van de template. Elke poging: `chat`, dan `problems`. Geen problemen: geef `json.loads(answer)` terug. Wel problemen: voeg het antwoord toe als bericht van de assistant en een bericht met de problemen als bericht van de gebruiker, en probeer opnieuw.
    4. **Geen antwoord paste:** geef na de laatste poging `None` terug, zodat de rest van je app kan zeggen dat hij het bericht niet kon lezen, in plaats van te stoppen met een fout.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.findings()["worked"]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste run van de course met de regel, seed 0, op het bericht over voetbal: `ask_json(url, "event", "{text}")`.

    1. **De woorden:** de template `"event"`, versie 1, met zijn schema: `what` en `day` strings, `hour` een getal.
    2. **De eerste berichten:** de system prompt met de regel over JSON, en de template ingevuld met het bericht, die eindigt met een lege regel en *{text}*
    3. **Vragen:** `chat` stuurt de twee berichten; Qwen3 antwoordde `{answer}`.
    4. **Controleren:** `problems` vindt niets, dus `ask_json` geeft na {tries} poging `{value}` terug. Een agenda kan het op dinsdag om 19:00 zetten.
    """.format(text=C.FOOTBALL, answer=_w["text"], value=repr(_w["value"]), tries=_w["tries"]))
    return


@app.cell(hide_code=True)
def text_prompts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De prompts van je assistent

        In het vak staan de woorden van je assistent: `SYSTEM`, en `TEMPLATES` met twee taken, de afspraak en een titel. Lange strings staan over twee regels, zoals in [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl): twee strings naast elkaar tussen haakjes zijn voor Python één string. Schrijf `messages_for(name, text)`: die geeft een lijst met twee berichten terug, de system prompt, en de template van `name` ingevuld met `text`, als bericht van de gebruiker. Laat `SYSTEM` en `TEMPLATES` voorlopig zoals ze zijn; de check vergelijkt met die van de course. Als het goed is, bewaart de check het hele vak in je project, als `work/projects/app/prompts.py`. Daarna is het bestand van jou om te veranderen; houd de twee taken erin, want oefening 2 vraagt om allebei.
        """),
        L.task("Schrijf `messages_for`: neem de template die `name` noemt, en geef het bericht van het systeem en de ingevulde template als bericht van de gebruiker terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompts():
    box_prompts = L.code_box("messages_for", lines=30)
    box_prompts
    return (box_prompts,)


@app.cell(hide_code=True)
def check_prompts(box_prompts):
    result_prompts = L.check("messages_for", box_prompts.value)
    result_prompts
    return (result_prompts,)


@app.cell(hide_code=True)
def hints_prompts():
    L.hints(
        "messages_for",
        [
            "Twee stappen: zoek de ingang van de template op zijn naam op, en maak een lijst van twee dictionaries.",
            "`template = TEMPLATES[name]` zoekt de ingang op; de string met `{text}` erin staat onder een van zijn keys, en `.format(text=text)` vult hem in.",
            "`return [{\"role\": \"system\", \"content\": SYSTEM},` en op de volgende regel `{\"role\": \"user\", \"content\": ___}]`.",
        ],
        why="""```python
def messages_for(name, text):
    template = TEMPLATES[name]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": template["text"].format(text=text)}]
```

De system prompt komt eerst, zodat het model zijn regels leest vóór de taak.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. JSON die je assistent zelf controleert

        `ask_json(url, name, text, tries=3)` vraagt met je prompts om de JSON van een taak, via je `chat`, en controleert elk antwoord met je `problems`. Het begin maakt al de eerste berichten, neemt het schema van de template, en vraagt in een loop. Vul de rest in: zoek na elk antwoord de problemen erin. Zijn er geen, geef dan terug wat `json.loads` ervan maakt. Anders voeg je het antwoord toe als bericht van de assistant en een bericht met de problemen als bericht van de gebruiker, zoals *Fix these problems: the answer is not JSON. Answer with the whole JSON again.*, en probeer je het opnieuw. Past geen enkele poging, dan geeft hij `None` terug. De check roept hem aan tegen een stand-in waarvan het eerste antwoord op het bericht over voetbal in een codeblok staat. Als hij goed is, bewaart de check hem in je project, als `work/projects/app/answers.py`.
        """),
        L.task("Na `chat`: zoek de problemen, geef de JSON terug als er geen zijn, en voeg anders het antwoord en de problemen toe aan `messages`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_json", lines=17)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask, result_prompts):
    result_prompts  # opnieuw uitvoeren als oefening 1 je prompts bewaart
    result_ask = L.check("ask_json", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_json",
        [
            "Het is de loop van je `ask_json` in [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl), met `chat` in plaats van `ask`.",
            "`found = problems(answer, schema)`; als `found == []`, geef je `json.loads(answer)` terug.",
            "Anders: `fix = \"Fix these problems: \" + ___ + \". Answer with the whole JSON again.\"`, en `messages = messages + [...]` met het antwoord als bericht van de assistant en `fix` als bericht van de gebruiker.",
        ],
        why="""```python
    for attempt in range(tries):
        answer = chat(url, messages)
        found = problems(answer, schema)
        if found == []:
            return json.loads(answer)
        fix = "Fix these problems: " + "; ".join(found) + ". Answer with the whole JSON again."
        messages = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]
    return None
```

`messages_for` maakt bij elke aanroep een nieuwe lijst, en `+` maakt er bij elke poging een nieuwe.""",
    )
    return


@app.cell(hide_code=True)
def text_new_version():
    mo.md(r"""
    ### Oefening 3. Een betere template
    """)
    return


@app.cell(hide_code=True)
def choice_new_version_cell():
    choice_new_version = L.choice(
        "new-version",
        question="Om uur 15 te krijgen voor de leesclub, voeg je een zin over de 24-uursnotatie toe aan de template van de afspraak, in `app/prompts.py`. Wat verander je nog meer?",
        options={
            "nothing": "Niets: de oude woorden zijn weg, dus niets kan ze door elkaar halen",
            "client": "`app/client.py`, zodat die de nieuwe template stuurt",
            "version": "De `\"version\"` van de template, van 1 naar 2",
        },
        correct="version",
        why={
            "version": "Nieuwe woorden zijn een nieuwe versie. Antwoorden van versie 1 kwamen van de oude woorden, en de volgende van versie 2, dus je kunt ze vergelijken; de demo hieronder laat bij elk antwoord de versie zien.",
            "nothing": "Antwoorden op de oude woorden zijn er nog, in je app of zijn log. Hoe onderscheid je die zonder nieuwe versie van de antwoorden op de nieuwe?",
            "client": "De client stuurt de berichten die hij krijgt: `messages_for` leest de template elke keer uit `app/prompts.py`. Welk bestand bevat de woorden, en wat hoort bij elke verandering eraan?",
        },
    )
    choice_new_version
    return (choice_new_version,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Wat je project nu kan

    De cel neemt `ask_json` uit je project, `work/projects/app/answers.py`, met je prompts en je client, en vraagt Qwen3-0.6B op de eigen server van de course om de afspraak in een bericht van jou, als JSON. Je ziet elke poging en de problemen erin, en de afspraak als kaartje, zoals een agenda in je app het zou kunnen laten zien. Je `app/prompts.py` wordt gebruikt zoals hij is: verander daar een template en verhoog de versie, dan gebruikt de demo de nieuwe woorden. Schrijf je bericht in het Engels, omdat Qwen3 Engels het best schrijft.
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
def message_box_cell():
    message_box = mo.ui.text_area(value="Can we have lunch on Thursday at 12? Kim", label="Een bericht met een afspraak", full_width=True, rows=2)
    send_button = mo.ui.run_button(label="Vraag met je ask_json")
    mo.vstack([message_box, send_button])
    return message_box, send_button


@app.cell(hide_code=True)
def demo_event(message_box, result_ask, result_prompts, send_button):
    result_prompts  # opnieuw uitvoeren als een oefening je module bewaart
    _ask, _why = C.learner_ask_json() if result_ask.state == "correct" else (None, None)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif result_ask.state != "correct":
        _out = mo.md("*Maak eerst oefening 2 af: dan staat je `ask_json` in je project, en gebruikt deze cel hem.*")
    elif _ask is None:
        _out = mo.md("*Je `ask_json` draait niet zoals je project hem nu heeft: {why}*".format(why=_why))
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not message_box.value.strip():
        _out = mo.md("*Schrijf eerst een bericht in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `ask_json` vraagt; Qwen3 schrijft …*"))
        _words = {"title": "Wat je assistent van het bericht maakte", "prompt": "Template event, versie {version}; requests gestuurd: {n}; {seconds} seconden.",
                  "try": "Poging {n}:", "fits": "past bij het schema", "found": "problemen: {p}",
                  "gave": "Je ask_json gaf terug:", "none": "Geen poging paste, dus je ask_json gaf None terug: je app kan zeggen dat hij het bericht niet kon lezen.",
                  "card": "In je agenda", "yours": "Alle modules waren van jou.", "course": "De versie van de course stond in voor: {files}.", "failed": "Je ask_json stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De system prompt van [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) draagt nu een regel voor elke taak van je assistent; de templates met een versie van [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl) staan in een eigen bestand; en de controle en de tweede poging van [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl) lopen via de client van [Project: je assistent antwoordt](/course/open/build-project-call?lang=nl). Je project komt terug in *Project: je assistent leest je documenten*, waar hij antwoordt uit je eigen documenten en zijn bron noemt. Eerst laat de volgende les zien waarom een prompt die bij het ene model werkt, bij een ander kan mislukken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_version, result_ask, result_prompts):
    L.wrap_up(result_prompts, result_ask, choice_new_version)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
