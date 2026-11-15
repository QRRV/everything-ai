# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-sampling-api. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-sampling-api", "nl")
    C = L.checks   # de opgenomen runs van de course, de stand-in van de checks, en de demo met je ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en één over [Top-k en top-p sampling](/course/open/lm-top-k-top-p?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_finish():
    quick_finish = L.choice(
        "quick-finish",
        question="Een antwoord van de API heeft `\"finish_reason\": \"length\"`. Wat zegt dat?",
        options={
            "limit": "Het bereikte `max_tokens` en is afgekapt",
            "done": "Het model beëindigde het antwoord zelf",
            "long": "De vraag was te lang, en het model las hem niet helemaal",
        },
        correct="limit",
        why={
            "limit": "`\"length\"` betekent dat het antwoord het hoogste aantal tokens bereikte dat het request toestond; `\"stop\"` betekent dat het op een andere manier eindigde.",
            "done": "Dan zou er `\"stop\"` staan. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "long": "`finish_reason` gaat over het antwoord, niet over de vraag. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_finish
    return (quick_finish,)


@app.cell(hide_code=True)
def quick_check_top_p():
    quick_top_p = L.choice(
        "quick-top-p",
        question="De kansen op het volgende token zijn `[0.5, 0.3, 0.15, 0.05]`. Welke tokens houdt top-p met p = 0,75 over?",
        options={
            "all": "Alle vier, elk met een kleiner deel",
            "two": "De eerste twee",
            "three": "De eerste drie",
        },
        correct="two",
        why={
            "two": "0,5 alleen is minder dan 0,75; 0,5 + 0,3 = 0,8 haalt het. Top-p houdt zo min mogelijk tokens over die samen minstens kans p hebben.",
            "all": "Top-p snijdt weg: de tokens na de tokens die het nodig heeft, krijgen kans 0. Zie [Top-k en top-p sampling](/course/open/lm-top-k-top-p?lang=nl).",
            "three": "Tel vanaf de bovenste: na hoeveel tokens is de som minstens 0,75? Zie [Top-k en top-p sampling](/course/open/lm-top-k-top-p?lang=nl).",
        },
    )
    quick_top_p
    return (quick_top_p,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je weet hoe een model zijn volgende token kiest: bij een temperature, uit de tokens die top-p overhoudt. Via de meeste API's kies je dat per request. Een test van je app heeft elke keer hetzelfde antwoord nodig; een knop die namen voorstelt, elke keer een nieuw; en een antwoord dat maar één regel hoeft te geven, hoort na die regel te stoppen, want elk token dat het schrijft kost tijd, en bij een gehoste API geld. Dat gaat allemaal in de body van het request, naast de berichten.

    **Een voorbeeld uit het dagelijks leven.** Een koffieautomaat op het werk heeft knoppen: sterker of milder, een kleine beker of een grote. Het is elke keer dezelfde automaat, maar jij bepaalt per beker hoe de koffie eruit komt. Dat is maar een vergelijking; hieronder zie je de instellingen van de API van een model, en wat elke instelling met de antwoorden van Qwen3 deed.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | `"temperature"` | in de body: de temperature waarbij de server samplet; `0` is greedy decoding, elke keer hetzelfde antwoord; de server van de course neemt een getal van 0 tot en met 2 |
    | `"top_p"` | in de body: de p van top-p sampling, van 0 tot en met 1 |
    | `"seed"` | in de body: een geheel getal waar de toevalstrekking van de server begint; dezelfde seed met hetzelfde request geeft weer hetzelfde antwoord |
    | **max tokens** | `"max_tokens"`: het hoogste aantal tokens dat het antwoord mag hebben; een antwoord dat het bereikt, wordt afgekapt, met `finish_reason` `"length"` |
    | **stop sequence** | `"stop"`: een tekst, of een lijst van hoogstens vier, waarbij de server het antwoord beëindigt; de stop sequence zelf staat niet in het antwoord, en `finish_reason` is `"stop"` |

    ### Dezelfde vraag, drie instellingen

    De course vroeg Qwen3 via zijn eigen server om een naam voor een bakkerij, acht keer bij elk van drie instellingen, met de seeds 0 tot en met 7. De vraag is Engels, omdat Qwen3 Engels het best schrijft:
    """)
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Bij temperature 0 gaf elke seed dezelfde naam: greedy decoding neemt elke keer het waarschijnlijkste token, dus er is geen trekking die een seed kan veranderen. Bij 0,7 met top_p 0,8, wat de makers van Qwen3 voor dit model aanraden, gaven de acht runs {advised} verschillende namen. Bij 1,5 met top_p 1,0 werden het er {hot}, gedurfdere, en {sentences} daarvan was een zin in plaats van alleen de naam, tot `max_tokens` hem stopte: gedurfder samplen geeft meer variatie, en meer ruimte voor een antwoord dat de opdracht loslaat.

        De course vroeg de seeds 0, 1 en 2 bij 0,7 nog een keer, en kreeg dezelfde drie namen terug: `Bakerbloom`, `Bakery` en `smallbakery`. Een seed maakt een gesampled antwoord dus herhaalbaar: het antwoord komt nog steeds uit een trekking, maar elke keer uit dezelfde. Dat geldt voor hetzelfde model op dezelfde computer en software: op de jouwe kan seed 0 een andere naam geven dan hier.
        """.format(advised=_f["advised"], hot=_f["hot"], sentences=_f["sentences"])),
        L.watch_out("Een request zonder `\"temperature\"` krijgt die van de server. De server van de course samplet dan zoals de makers van Qwen3 aanraden, bij 0,7 met top_p 0,8, en houdt daarnaast altijd alleen de 20 waarschijnlijkste tokens over (top-k), waar de API van OpenAI geen instelling voor heeft. Een gehoste API heeft zijn eigen standaardwaarden, en niet elke API neemt bij elk model een seed, of zelfs een temperature; de API's die een seed nemen, beloven alleen dat dezelfde seed meestal hetzelfde antwoord geeft (voorbeelden in november 2026)."),
    ])
    return


@app.cell(hide_code=True)
def text_ends():
    mo.md(r"""
    ### Waar een antwoord eindigt

    Een antwoord eindigt op een van drie manieren: het model beëindigt het zelf, het bereikt `max_tokens`, of het bereikt een stop sequence. De course vroeg Qwen3 om drie soorten fruit, één per regel, op drie manieren:
    """)
    return


@app.cell(hide_code=True)
def picture_ends():
    mo.Html(C.ends_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ends_after():
    _f = C.findings()
    mo.md(r"""
    Met `"stop": ["\n"]` eindigt het antwoord bij de eerste regelovergang, die niet in het antwoord staat, en `finish_reason` is `"stop"`, net als wanneer het model een antwoord zelf beëindigt. Het model stopte daar met schrijven: {first} tokens in plaats van {lines}. Met `"max_tokens": 3` wordt het antwoord na {short} tokens afgekapt, midden in wat het model van plan was, en `finish_reason` zegt `"length"`. Allebei korter; alleen de stop sequence eindigt waar jij wilt. Lees dus `finish_reason` voordat je een antwoord gebruikt: `"length"` betekent dat het einde kan ontbreken.
    """.format(first=_f["first"]["tokens"], lines=_f["lines"]["tokens"], short=_f["short"]["tokens"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: bepaal hoe het model samplet

    1. **Hoeveel het varieert:** `"temperature"`: `0` voor elke keer hetzelfde antwoord, ongeveer `0.7` voor wat de makers van het model aanraden, hoger voor gedurfdere antwoorden die zich minder vaak aan de opdracht houden. `"top_p"` snijdt de onwaarschijnlijke tokens weg.
    2. **Weer hetzelfde antwoord:** een vaste `"seed"`, met hetzelfde request.
    3. **Waar het eindigt:** `"max_tokens"` als grens aan lengte en kosten, en `"stop"` voor een tekst waar het antwoord moet eindigen.
    4. **Kijk hoe het eindigde:** `answer["choices"][0]["finish_reason"]`: `"stop"`, of `"length"` voor een afgekapt antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een test van een app kijkt of de client een antwoord goed leest. De test moet elke keer hetzelfde antwoord krijgen, maar wel uit de sampling die de app echt gebruikt. Welke body?

    ```python
    body = {"model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Suggest a name for a small bakery. Answer with the name only."}],
            "max_tokens": 12, "temperature": 0.7, "top_p": 0.8, "seed": 0}
    ```

    1. **Hoeveel het varieert:** `0.7` en `0.8`, de instellingen van de app zelf.
    2. **Weer hetzelfde antwoord:** `"seed": 0`. In de runs van de course gaf seed 0 `Bakerbloom`, de eerste keer en de tweede keer.
    3. **Waar het eindigt:** `"max_tokens": 12` is ruim genoeg voor een naam.
    4. **Hoe het eindigde:** `finish_reason` was `"stop"`: het model beëindigde de naam zelf.

    Met `"temperature": 0` zou de test ook elke keer hetzelfde antwoord krijgen, maar dan uit greedy decoding, en dat doet de app niet.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Temperature en seed

        `ask(url, question, temperature, seed)` stuurt al een vraag. Laat hem de `temperature` en de `seed` meesturen die hij krijgt. De check roept je `ask` aan tegen een stand-in van de server, één keer met een temperature van 0, en kijkt wat je request bij zich had.
        """),
        L.task("Voeg `\"temperature\"` en `\"seed\"` toe aan de body, met de waarden die `ask` krijgt."),
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
            "De instellingen gaan in de body, naast `\"model\"`, `\"messages\"` en `\"max_tokens\"`, zoals in het uitgewerkte voorbeeld.",
            "Elke instelling is een key met het argument als waarde: `\"temperature\": temperature`.",
            "De body eindigt op `\"max_tokens\": 12, \"temperature\": temperature, \"seed\": seed}`.",
        ],
        why="""```python
body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 12,
        "temperature": temperature, "seed": seed}
```

De argumenten zelf gaan in de body, dus een temperature van 0 gaat ook mee.""",
    )
    return


@app.cell(hide_code=True)
def text_first():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén regel, en waarom het eindigde

        `first_line(url, question)` hoort alleen de eerste regel van het antwoord terug te geven, en waarom het antwoord eindigde. Laat de server het antwoord beëindigen bij de eerste regelovergang, en geef de tekst en de `finish_reason` samen terug. De check stelt één vraag waarvan het antwoord drie regels heeft, en één waarvan het antwoord geen regelovergang heeft en doorloopt tot 20 tokens.
        """),
        L.task("Voeg een stop sequence met `\"\\n\"` toe aan de body, en geef de tekst en de `finish_reason` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first():
    box_first = L.code_box("first_line", lines=12)
    box_first
    return (box_first,)


@app.cell(hide_code=True)
def check_first(box_first):
    result_first = L.check("first_line", box_first.value)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_first():
    L.hints(
        "first_line",
        [
            "Twee veranderingen: één key in de body, en een laatste regel die twee dingen teruggeeft.",
            "In de body: `\"stop\": [\"\\n\"]`. De `finish_reason` staat in de choice, naast het bericht: `answer[\"choices\"][0][\"finish_reason\"]`.",
            "Neem `choice = answer[\"choices\"][0]`, en geef `choice[\"message\"][\"content\"]` en `choice[\"finish_reason\"]` terug, met een komma ertussen.",
        ],
        why="""```python
def first_line(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 20,
            "stop": ["\\n"]}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    choice = answer["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]
```

De server beëindigt het antwoord bij de regelovergang; `"length"` zou betekenen dat het 20 tokens bereikte voordat er een kwam.""",
    )
    return


@app.cell(hide_code=True)
def text_names():
    mo.md(r"""
    ### Oefening 3. Tien namen
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="Een knop in je app vraagt de API om een naam voor een nieuw café, tien keer met dezelfde vraag. Je wilt tien verschillende namen. Welke instellingen?",
        options={
            "same-seed": "`\"temperature\": 0.9`, en elke keer precies dezelfde `\"seed\"`",
            "greedy": "`\"temperature\": 0`, en elke keer een nieuwe `\"seed\"`",
            "new-seed": "`\"temperature\": 0.9`, en elke keer een nieuwe `\"seed\"`",
        },
        correct="new-seed",
        why={
            "new-seed": "Door het samplen kunnen de namen verschillen, en een nieuwe seed geeft elke keer een nieuwe trekking.",
            "same-seed": "Dezelfde seed met hetzelfde request geeft hetzelfde antwoord: tien keer dezelfde naam. Wat moet er van request tot request veranderen?",
            "greedy": "Bij temperature 0 is er geen trekking, dus verandert de seed niets: bij 0 gaf elke seed de bakkerij dezelfde naam.",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag Qwen3 drie keer

    Zodra oefening 1 goed is, kan je `ask` met de echte server praten: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter, en je functie stelt je vraag drie keer, met de seeds 0, 1 en 2, bij de temperature die jij kiest.
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
    question_box = mo.ui.text(value=C.NAME, label="Je vraag", full_width=True)
    temperature_slider = mo.ui.slider(0, 2, step=0.1, value=0.7, label="temperature", show_value=True)
    send_button = mo.ui.run_button(label="Vraag drie keer met je ask")
    mo.vstack([question_box, temperature_slider, send_button])
    return question_box, send_button, temperature_slider


@app.cell(hide_code=True)
def demo_answer(box_ask, question_box, send_button, temperature_slider):
    _ask = C.learner_ask(box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _ask is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo vraagt met je eigen `ask`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `ask` heeft de vraag verstuurd; de server laadt Qwen3 en antwoordt drie keer …*"))
        _words = {"title": "Wat er terugkwam bij je ask, drie keer", "seed": "seed",
                  "done": "Temperature {t}: {different} verschillende antwoorden, in {seconds} seconden samen.",
                  "failed": "Je ask stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_ask, question_box.value, temperature_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De temperature van [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl) en de p van [Top-k en top-p sampling](/course/open/lm-top-k-top-p?lang=nl) zijn nu twee keys in de body van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en een seed maakt een gesampled antwoord herhaalbaar, zoals hij de dobbelstenen van [Code uit modules gebruiken](/course/open/py-imports?lang=nl) herhaalbaar maakte. In de volgende les, [System prompts en response prefill](/course/open/build-system-prompt?lang=nl), vertel je het model hoe het zich een heel gesprek lang gedraagt, en schrijf je zelf de eerste woorden van zijn antwoord. Een latere les stelt met opzet dezelfde vraag een paar keer, en neemt het antwoord dat het vaakst terugkomt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, result_ask, result_first):
    L.wrap_up(result_ask, result_first, choice_names)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
