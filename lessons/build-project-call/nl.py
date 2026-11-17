# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-project-call. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-call", "nl")
    C = L.checks   # de runs van de course met en zonder het gesprek, de stand-ins, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en één over [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_content():
    quick_content = L.choice(
        "quick-content",
        question="`answer = json.loads(response.read())` bevat de JSON die de API van een model terugstuurde. Waar staat de tekst van het antwoord?",
        options={
            "usage": "`answer[\"usage\"]`",
            "content": "`answer[\"choices\"][0][\"message\"][\"content\"]`",
            "text": "`answer[\"text\"]`",
        },
        correct="content",
        why={
            "content": "`choices` is een lijst antwoorden; de eerste heeft de `message`, en de `content` daarvan is de tekst.",
            "usage": "`usage` telt de tokens die erin gingen en eruit kwamen. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "text": "Het antwoord heeft bovenaan geen key `\"text\"`. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_content
    return (quick_content,)


@app.cell(hide_code=True)
def quick_check_module():
    quick_module = L.choice(
        "quick-module",
        question="Waar vindt `from tools import stats` in de course zijn module?",
        options={
            "projects": "In `work/projects/tools/stats.py`",
            "lessons": "In de map van de les die hem maakte",
            "python": "Tussen de modules die bij Python horen",
        },
        correct="projects",
        why={
            "projects": "Je packages staan in `work/projects/`, en daar kijkt Python als je importeert, in elke les.",
            "lessons": "Lessen kunnen code bewaren, maar niet naast zichzelf. Zie [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
            "python": "`tools` is een package van jezelf, niet een van Python. Zie [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
        },
    )
    quick_module
    return (quick_module,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een assistent beantwoordt niet één vraag; hij voert een gesprek. Maar een model onthoudt niets tussen twee requests, zoals je zag in [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl): wat het van het gesprek weet, is wat het request meedraagt. Je programma houdt het gesprek dus bij, en stuurt het helemaal mee, elke beurt.

        Met deze les begint je project: een eigen assistent, in het package `app`. Vandaag krijgt de assistent zijn eerste module, de client: één functie die met de API praat. Latere stappen geven hem een goede prompt, en documenten om te lezen.

        **Een voorbeeld uit het dagelijks leven.** Je belt een helpdesk, en krijgt elke keer iemand anders. Begin je met "het gaat om mijn bestelling van vorige week, die kapot aankwam", dan kunnen ze je verder helpen. Zeg je alleen "en, wat nu?", dan niet. Dat is alleen een vergelijking; hieronder zie je het gebeuren met Qwen3.
        """),
        draw.project_steps(["Je assistent antwoordt", "Je assistent, goed geprompt", "Je assistent leest je documenten"], 0,
                           here_label="← vandaag", title="Waar je project staat: je assistent"),
    ])
    return


@app.cell(hide_code=True)
def predict_name():
    guess_name = L.guess(
        "name",
        question="De course zei tegen Qwen3-0.6B *My name is Sam* en vroeg om een idee voor het eten. Daarna vroeg hij *What is my name?* in een nieuw request, één keer met de eerste vraag en het antwoord erin, en één keer alleen. Elk acht runs. Hoe vaak noemde het antwoord Sam?",
        options={
            "both": "Bijna altijd, allebei",
            "with": "8 van de 8 met het gesprek, 0 van de 8 alleen",
            "neither": "Zelden, allebei: het model is te klein",
        },
        reveal=f"**8 van de 8 met het gesprek, 0 van de 8 alleen.** Zonder gesprek zei Qwen3 dat het geen mens is, of een language model zonder naam: het request bevatte geen naam. Met het gesprek noemde het elke keer Sam, maar {C.findings()['own']} van de 8 antwoordden *I am Sam.* of *My name is Sam.*, alsof het model Sam was, en geen enkele zei *Your name is Sam.*: een klein model haalt door elkaar wie wat zei.",
    )
    guess_name
    return (guess_name,)


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **API client** | het deel van een programma dat met een API praat: een functie die neemt wat je wilt vragen en het antwoord teruggeeft, zodat de rest van het programma nooit een URL, HTTP of JSON ziet |

    ### Eén functie tussen je programma en het model

    Je `ask` uit [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl) stuurde één vraag. De client stuurt een heel gesprek: de lijst `messages` die de rest van je programma bijhoudt. Alles van HTTP blijft erbinnen: stap je over op een andere server, of op een gehoste API met een key, dan verander je de client, niet de rest.

    Het gesprek groeit met elke beurt. In de runs van de course had het eerste request {C.findings()['tokens_first']} input tokens, en het tweede, met de eerste vraag en het antwoord erin, {C.findings()['tokens_second']}. Elke beurt stuurt alles ervoor opnieuw, en een gehoste API telt die tokens elke keer.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een client, en een gesprek

    1. **De client:** `chat(url, messages, max_tokens=200)` zet `messages` en `max_tokens` in de body, verstuurt hem zoals je `ask`, en geeft de tekst van het antwoord terug. De `=200` geeft `max_tokens` een standaardwaarde: `chat(url, messages)` vraagt om 200 tokens, en `chat(url, messages, 4)` om 4.
    2. **Het gesprek:** één lijst `messages`, één keer gemaakt.
    3. **Elke beurt:** voeg het bericht van de gebruiker toe, `{"role": "user", "content": text}`, en roep `chat(url, messages)` aan.
    4. **Na elk antwoord:** voeg het toe als `{"role": "assistant", "content": answer}`, zodat het volgende request het meedraagt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste run van de course, seed 0, met de hand:

    ```python
    messages = [{"role": "user", "content": "My name is Sam. Suggest a dinner for tonight, in one sentence."}]
    answer = chat(url, messages)
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": "What is my name?"})
    chat(url, messages)
    ```

    1. **Eerste beurt:** er gaat één bericht uit; het antwoord is *Sam's dinner could be a cozy meal with a classic blend of flavors, such as a classic steak and a side salad.*
    2. **Bewaar het:** het antwoord gaat als bericht van de assistant in `messages`.
    3. **Tweede beurt:** de vraag gaat er ook in, dus het tweede request draagt drie berichten.
    4. **Het antwoord:** *I am Sam.* De naam is er, omdat het request hem meedroeg; wie wie is, haalt een model van deze grootte nog door elkaar.
    """)
    return


@app.cell(hide_code=True)
def text_chat():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De client

        Schrijf `chat(url, messages, max_tokens=200)`: hij stuurt de hele lijst `messages`, met `max_tokens`, naar `url + "/chat/completions"`, en geeft de tekst van het antwoord terug. Begin bij je `ask`. De check roept hem aan met een gesprek van drie berichten en met een kleine `max_tokens`, tegen een stand-in van de server. Als hij goed is, bewaart de check hem in je project, als `work/projects/app/client.py`.
        """),
        L.task("Schrijf `chat`: de hele `messages` en `max_tokens` in de body, en de tekst van het antwoord terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chat():
    box_chat = L.code_box("chat", lines=11)
    box_chat
    return (box_chat,)


@app.cell(hide_code=True)
def check_chat(box_chat):
    result_chat = L.check("chat", box_chat.value)
    result_chat
    return (result_chat,)


@app.cell(hide_code=True)
def hints_chat():
    L.hints(
        "chat",
        [
            "Het is je `ask`, met twee veranderingen in de body: wat er in `\"messages\"` komt, en in `\"max_tokens\"`.",
            "`body = {\"model\": \"qwen3-0.6b\", \"messages\": messages, \"max_tokens\": max_tokens}`; de rest is zoals in `ask`.",
            "Na de body de `Request`, `urlopen` en `json.loads` van `ask`, en aan het eind `return answer[\"choices\"][___][\"message\"][\"content\"]`.",
        ],
        why="""```python
def chat(url, messages, max_tokens=200):
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

De lijst gaat zoals hij is in de body: het hele gesprek, elke keer.""",
    )
    return


@app.cell(hide_code=True)
def text_talk():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een gesprek van meer beurten

        `talk(url, said)` voert een gesprek: voor elke `text` in `said`, de beurt van de gebruiker, vraagt hij je `chat`, en hij geeft de lijst van de antwoorden terug. Hij importeert `chat` uit je project. De check geeft hem drie beurten; de laatste, *What do I like?*, is alleen te beantwoorden met de eerdere beurten in het request. Het plaatje laat elk request zien dat je `talk` stuurde.
        """),
        L.task("Vul de loop in: voeg de beurt van de gebruiker toe, vraag `chat`, en voeg het antwoord toe aan `messages` en aan `answers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_talk():
    box_talk = L.code_box("talk", lines=10)
    box_talk
    return (box_talk,)


@app.cell(hide_code=True)
def check_talk(box_talk):
    result_talk = L.check("talk", box_talk.value)
    result_talk
    return (result_talk,)


@app.cell(hide_code=True)
def hints_talk():
    L.hints(
        "talk",
        [
            "Stap 3 en 4 van het recept, in de loop: vier regels.",
            "`messages.append({\"role\": \"user\", \"content\": text})`, en dan `answer = chat(url, messages)`.",
            "Dan `messages.append({\"role\": ___, \"content\": answer})`, en `answers.append(answer)`.",
        ],
        why="""```python
def talk(url, said):
    messages = []
    answers = []
    for text in said:
        messages.append({"role": "user", "content": text})
        answer = chat(url, messages)
        messages.append({"role": "assistant", "content": answer})
        answers.append(answer)
    return answers
```

De lijst wordt één keer gemaakt, vóór de loop, en groeit elke beurt met twee berichten: het derde request draagt er vijf.""",
    )
    return


@app.cell(hide_code=True)
def text_why_all():
    mo.md(r"""
    ### Oefening 3. Waarom alles meesturen?
    """)
    return


@app.cell(hide_code=True)
def choice_why_all_cell():
    choice_why_all = L.choice(
        "why-all",
        question="Je assistent stuurt de eerste vraag en het antwoord mee met de tweede vraag. Waarom?",
        options={
            "faster": "Zodat het model sneller antwoordt",
            "memory": "Omdat het model niets onthoudt tussen requests: het kent het gesprek alleen uit wat het request meedraagt",
            "server": "Omdat de server het gesprek na een uur weggooit",
        },
        correct="memory",
        why={
            "memory": "Elk request begint bij niets; het gesprek bestaat alleen in je programma, dat het meestuurt. Daarom kost ook elke beurt meer tokens dan de vorige.",
            "faster": "Juist andersom: meer tokens lezen duurt langer. Wat zou het model zonder die tokens van de eerste beurt weten?",
            "server": "De server bewaarde het nooit: het model leest elk request op zichzelf. Waar leeft het gesprek dan?",
        },
    )
    choice_why_all
    return (choice_why_all,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Praat met Qwen3 via je client

    De cel neemt `chat` uit je project, `work/projects/app/client.py`, en voert een gesprek met Qwen3-0.6B op de server van de course. Elke beurt stuurt alles ervoor mee. *Opnieuw beginnen* maakt het gesprek leeg.
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
def conversation_state():
    get_talk, set_talk = mo.state([])
    return get_talk, set_talk


@app.cell(hide_code=True)
def turn_box_cell():
    turn_box = mo.ui.text(value="My name is Sam. Suggest a dinner for tonight, in one sentence.", label="Jouw beurt", full_width=True)
    send_button = mo.ui.run_button(label="Verstuur")
    again_button = mo.ui.run_button(label="Opnieuw beginnen")
    mo.vstack([turn_box, mo.hstack([send_button, again_button], justify="start")])
    return again_button, send_button, turn_box


@app.cell(hide_code=True)
def demo_turn(again_button, get_talk, result_chat, send_button, set_talk, turn_box):
    _chat = C.learner_chat() if result_chat.state == "correct" else None
    if again_button.value:
        set_talk([])
        _out = mo.md("*Het gesprek is weer leeg.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _chat is None:
        _out = mo.md("*Maak eerst oefening 1 af: dan staat je `chat` in je project, en gebruikt deze cel hem.*")
    elif not send_button.value:
        _out = mo.md("*Druk op Verstuur om het gesprek te beginnen.*")
    elif not turn_box.value.strip():
        _out = mo.md("*Schrijf eerst je beurt in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `chat` stuurt het gesprek; Qwen3 schrijft …*"))
        _words = {"title": "Je gesprek met Qwen3", "done": "{seconds} seconden; het request droeg {n} berichten.",
                  "failed": "Je chat stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _after, _html = C.demo_turn(_chat, get_talk(), turn_box.value, _words, L.lang)
        set_talk(_after)
        _out = mo.Html(_html)
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De client is je `ask` uit [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met het gesprek van [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl) erin, bewaard in je project zoals in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl). Elke beurt stuurt meer tokens, en die telde [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl). Je project komt terug in [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl), dat hem een system prompt en antwoorden als JSON geeft. Eerst laten de volgende lessen het antwoord binnenkomen terwijl het geschreven wordt, en gaan ze om met een server die nee zegt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why_all, result_chat, result_talk):
    L.wrap_up(result_chat, result_talk, choice_why_all)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
