# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-self-consistency. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-self-consistency", "nl")
    C = L.checks   # de opgenomen runs van de course, de checks, en de demo met je ask_many


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Instellingen voor sampling via de API](/course/open/build-sampling-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="Je stuurt hetzelfde request twee keer, met temperature 0,7 en dezelfde seed. Wat komt er terug?",
        options={
            "same": "Twee keer hetzelfde antwoord, omdat de seed het toeval vastlegt",
            "differ": "Twee verschillende antwoorden, omdat een temperature boven 0 elk token samplet",
            "error": "Een fout: een seed werkt alleen bij temperature 0",
        },
        correct="same",
        why={
            "same": "De seed legt de willekeurige trekkingen vast, dus hetzelfde request met dezelfde seed samplet dezelfde tokens. Voor verschillende antwoorden heeft elk request een eigen seed nodig.",
            "differ": "De temperature laat het model samplen, maar de seed legt de trekkingen vast: dezelfde seed geeft hetzelfde antwoord. Zie [Instellingen voor sampling via de API](/course/open/build-sampling-api?lang=nl).",
            "error": "Een seed werkt bij elke temperature; bij 0 verandert hij niets, omdat greedy decoding niets trekt. Zie [Instellingen voor sampling via de API](/course/open/build-sampling-api?lang=nl).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model dat een som stap voor stap uitrekent, kan zich in één stap vergissen, en dan is het antwoord aan het eind fout, hoe goed de rest ook was. Met sampling gaat elke run een net iets andere weg. De vergissingen komen op verschillende foute getallen uit, terwijl de runs die goed redeneren allemaal op hetzelfde goede getal uitkomen. Dezelfde vraag een paar keer stellen en het getal nemen dat het vaakst terugkomt, is dus vaker goed dan één antwoord. Dat kost een aanroep per antwoord, dus je wilt weten hoeveel het oplevert.

    **Een voorbeeld uit het dagelijks leven.** Als drie mensen dezelfde stapel munten tellen en twee van hen op 85 euro uitkomen, vertrouw je 85 meer dan de 79 van de derde; hun vergissingen komen niet snel op hetzelfde getal uit. Dat is maar een vergelijking; hieronder meet je hoe vaak de stemming over de antwoorden van Qwen3 goed is.
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
    | **majority voting** | van een paar antwoorden het antwoord nemen dat het vaakst voorkomt |
    | **self-consistency** | dezelfde vraag een paar keer stellen met een temperature boven 0, elke keer met een eigen seed, en het eindantwoord nemen dat het vaakst voorkomt: majority voting over chains of thought |

    ### Twaalf sommen, elk negen runs

    De course gaf Qwen3 {questions} sommen in woorden, elk {runs} keer met een eigen seed, bij de temperature 0,7 van de server, met deze prompt om elke vraag:

    ```text
    {steps}
    ```

    Uit elk antwoord haalt een kleine functie, `answer_of`, het getal na *Answer:*. Elke rij is een som, met het getal van elke run, het goede getal, en het getal dat het vaakst kwam:
    """.format(questions=_f["questions"], runs=_f["runs"], steps=C.STEPS))
    return


@app.cell(hide_code=True)
def picture_grid():
    mo.Html(C.grid_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_grid():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Van alle {total} antwoorden waren er {right} goed. Eén antwoord per som had gemiddeld {k1} van de {questions} sommen goed. Het getal dat het vaakst kwam in drie antwoorden had er {k3} goed, in vijf {k5}, en in alle negen {k9}. Bij 1, 3 en 5 is dat het gemiddelde over elke manier om er zoveel uit de negen runs te kiezen, dus de getallen hangen niet af van welke runs het eerst kwamen:
        """.format(total=_f["runs"] * _f["questions"], right=_f["all_right"], questions=_f["questions"],
                   k1=C.number(_f["per_k"][1], L.lang, 1), k3=C.number(_f["per_k"][3], L.lang, 1),
                   k5=C.number(_f["per_k"][5], L.lang, 1), k9=C.number(_f["per_k"][9], L.lang, 0))),
        mo.Html(C.vote_picture(L.lang)),
        mo.md(r"""
        Een stemming kan geen vergissing herstellen die de meeste runs op dezelfde manier maken. Bij som {s}, over de fiets van Lisa, zeiden {c} van de negen runs {v}: zoveel weken kost 200 euro bij 15 euro per week als je bij nul begint, zonder de 65 euro die Lisa al heeft. Het goede antwoord is {r}, en geen enkele run gaf het. Een stemming vindt het antwoord waar de meeste runs het over eens zijn, en dat is alleen het goede als de goede weg ook de meest gekozen weg is.
        """.format(s=_f["stuck"][0] + 1, c=_f["stuck_vote"][0][1], v=_f["stuck_vote"][0][0], r=C.QUESTIONS[_f["stuck"][0]][1])),
        L.watch_out("Een stemming heeft antwoorden nodig die je precies kunt vergelijken, zoals een getal of de letter van een keuze. Twee verhalen of twee e-mails zijn nooit dezelfde tekst, dus die kun je niet tellen; en elk antwoord is een aanroep waar je op wacht en voor betaalt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: self-consistency

    1. **Vraag om een redenering en een eindantwoord dat je kunt vergelijken,** zoals een getal op de laatste regel.
    2. **Stel dezelfde vraag `n` keer,** met een temperature boven 0 en elke keer een andere seed, zodat de runs verschillen.
    3. **Haal het eindantwoord uit elke run,** en laat de runs zonder eindantwoord weg.
    4. **Tel, en neem het antwoord dat het vaakst voorkomt;** bij gelijkspel het antwoord dat eerst kwam.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _i = _f["example"]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Som {k} van de twaalf: *{question}*

    1. **Redenering en eindantwoord:** de prompt vraagt om stap voor stap te denken en te eindigen met *Answer:* en een getal.
    2. **Dezelfde vraag negen keer:** seeds 0 tot en met 8, bij temperature 0,7.
    3. **De eindantwoorden:** `answer_of` haalde {numbers} uit de negen runs.
    4. **Tel:** {counts}. Het meest gegeven getal is {vote}, en het goede antwoord is {right}. Eén antwoord alleen was {alone} van de negen keer goed.
    """.format(k=_i + 1, question=C.QUESTIONS[_i][0], numbers=", ".join(str(n) for n in C.numbers(_i)),
               counts=", ".join("{n}: {c}×".format(n=n, c=c) for n, c in _f["example_counts"]),
               vote=C.majority(C.numbers(_i)), right=C.QUESTIONS[_i][1], alone=_f["right"][_i]))
    return


@app.cell(hide_code=True)
def text_majority():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het antwoord dat het vaakst voorkomt

        Schrijf `majority(answers)`, die het antwoord teruggeeft dat het vaakst voorkomt in de lijst `answers`. Een antwoord zonder getal is `None`; dat is geen stem, dus laat het weg. `x is not None` is `True` voor elke waarde van `x` die niet `None` is. Bij gelijkspel wint het antwoord dat eerst kwam, en zonder stemmen geeft de functie `None` terug. Het vak telt al in een dictionary, maar geeft het eerste antwoord terug.
        """),
        L.task("Laat `None` weg, en geef het antwoord met de meeste stemmen terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=14)
    box_majority
    return (box_majority,)


@app.cell(hide_code=True)
def check_majority(box_majority):
    result_majority = L.check("majority", box_majority.value)
    result_majority
    return (result_majority,)


@app.cell(hide_code=True)
def hints_majority():
    L.hints(
        "majority",
        [
            "Twee stappen: eerst tellen, zonder `None`; dan door de tellingen lopen en het antwoord onthouden met de meeste stemmen tot nu toe.",
            "Zet het tellen onder `if answer is not None:`. Begin daarna met `best = None`, en loop door `counts`: een antwoord neemt de plaats van `best` in als `best` nog `None` is, of als het meer stemmen heeft dan `best`.",
            "Na het tellen: `best = None`, dan `for answer in counts:` met `if best is None or counts[answer] > counts[best]:` en `best = answer` eronder. Aan het eind `return best`.",
        ],
        why="""```python
def majority(answers):
    counts = {}
    for answer in answers:
        if answer is not None:
            if answer in counts:
                counts[answer] = counts[answer] + 1
            else:
                counts[answer] = 1
    best = None
    for answer in counts:
        if best is None or counts[answer] > counts[best]:
            best = answer
    return best
```

Een dictionary houdt zijn keys in de volgorde waarin ze erin kwamen, dus wie door `counts` loopt, komt de antwoorden tegen in de volgorde van de lijst. Omdat een later antwoord alleen wint met meer stemmen, `>` en niet `>=`, wint bij gelijkspel het antwoord dat eerst kwam.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dezelfde vraag, `n` keer

        Het vak vraagt één keer, met seed 0, en geeft `majority` van dat ene antwoord terug. Verander `ask_many(url, question, n)` zo dat het `n` keer vraagt, elke keer met een andere seed, en `majority` van alle getallen teruggeeft. `answer_of` en je `majority` zijn gegeven. De check draait je functie tegen een stand-in die elke seed op zijn eigen manier beantwoordt.
        """),
        L.task("Vraag `n` keer, elke keer met een andere seed, en geef `majority` van de getallen terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_many", lines=18)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask_many", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_many",
        [
            "Alles van `messages = ...` tot en met `answers.append(...)` moet `n` keer gebeuren; alleen de seed verandert elke keer.",
            "Zet die regels in een loop, `for i in range(n):`, vier spaties verder naar rechts, en gebruik `i` als seed.",
            "`for i in range(n):` onder `answers = []`, met de regels tot en met `answers.append(...)` eronder, en `\"seed\": i` in de body. `return majority(answers)` blijft na de loop staan.",
        ],
        why="""```python
import json
import urllib.request


def ask_many(url, question, n):
    answers = []
    for i in range(n):
        messages = [{"role": "user", "content": question}]
        body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 300,
                "temperature": 0.7, "seed": i}
        data = json.dumps(body).encode()
        request = urllib.request.Request(url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            answer = json.loads(response.read())
        answers.append(answer_of(answer["choices"][0]["message"]["content"]))
    return majority(answers)
```

Elke seed geeft een eigen run, en dezelfde seed geeft dezelfde run opnieuw: dezelfde vraag met dezelfde `n` geeft dus dezelfde stemming, zodat een uitkomst herhaalbaar is.""",
    )
    return


@app.cell(hide_code=True)
def text_worth():
    mo.md(r"""
    ### Oefening 3. Wanneer het de moeite waard is
    """)
    return


@app.cell(hide_code=True)
def choice_worth_cell():
    choice_worth = L.choice(
        "worth",
        question="Waar verdient vijf keer vragen en stemmen zijn vijf aanroepen terug?",
        options={
            "story": "Bij een kort verhaal: neem de tekst die het vaakst terugkwam",
            "number": "Bij een getal dat moet kloppen, zoals een bedrag op een offerte",
            "chat": "Bij elk antwoord in een chatapp, omdat vijf antwoorden altijd meer weten dan één",
        },
        correct="number",
        why={
            "number": "Een getal kun je precies vergelijken, en een fout getal kost meer dan de vier extra aanroepen.",
            "story": "Vijf runs van een verhaal zijn vijf verschillende teksten, dus geen tekst komt twee keer voor: er valt niets te tellen. Een stemming heeft antwoorden nodig die je precies kunt vergelijken.",
            "chat": "Vijf aanroepen per antwoord is vijf keer het wachten en de tokens, en een chatantwoord is vrije tekst die je niet kunt tellen. Bewaar de stemming voor antwoorden die je kunt vergelijken en die moeten kloppen.",
        },
    )
    choice_worth
    return (choice_worth,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 en 2 goed zijn, kun je met het echte model stemmen over een eigen som: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo stelt je vraag met je `ask_many`, in de prompt van deze les, en laat het getal van elk antwoord en de stemming zien. Een antwoord kan een paar honderd tokens kosten, dus vijf antwoorden kunnen een minuut of langer duren. Je computer is niet die van de course, dus je antwoorden kunnen verschillen. Schrijf je som in het Engels, omdat Qwen3 Engels het best schrijft.
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
def question_cell():
    question_box = mo.ui.text_area(value=C.QUESTIONS[-1][0], label="Je som", full_width=True, rows=3)
    n_slider = mo.ui.slider(start=3, stop=9, step=2, value=5, label="Antwoorden", show_value=True)
    vote_button = mo.ui.run_button(label="Vraag met je ask_many")
    mo.vstack([question_box, n_slider, vote_button])
    return n_slider, question_box, vote_button


@app.cell(hide_code=True)
def demo_answer(box_ask, box_majority, n_slider, question_box, vote_button):
    _fn = C.learner_ask_many(box_majority.value, box_ask.value)
    _question = question_box.value.strip()
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo vraagt met je eigen `ask_many` en `majority`.*")
    elif not vote_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not _question:
        _out = mo.md("*Schrijf eerst een som in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag gaat {n} keer de deur uit; de server laadt Qwen3 en antwoordt …*".format(n=n_slider.value)))
        _words = {"title": "Je som, {n} keer".format(n=n_slider.value),
                  "run": "Antwoord {k}: {n}",
                  "vote": "Je ask_many gaf terug: {got}, in {seconds} seconden.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, _question, n_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De temperature en de seed van [Instellingen voor sampling via de API](/course/open/build-sampling-api?lang=nl) zijn waardoor de runs verschillen, en waardoor ze herhaalbaar zijn. [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl) kijkt naar wat een request meeneemt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_worth, result_ask, result_majority):
    L.wrap_up(result_majority, result_ask, choice_worth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
