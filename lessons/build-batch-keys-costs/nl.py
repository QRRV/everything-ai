# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-batch-keys-costs. Route: course/route/lessen/bouwen.yaml. Checks, de stand-ins en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-batch-keys-costs", "nl")
    C = L.checks   # de stand-ins die een key willen, de batch van het uitgewerkte voorbeeld, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_401():
    quick_401 = L.choice(
        "quick-401",
        question="Een aanroep krijgt status 401: er kwam geen geldige key mee met het request. Is dat nog een poging waard?",
        options={
            "wait": "Ja, na even wachten",
            "no": "Nee: hetzelfde request krijgt hetzelfde antwoord",
            "now": "Ja, meteen",
        },
        correct="no",
        why={
            "no": "Een ontbrekende of verkeerde key blijft ontbreken of verkeerd, hoe vaak je het request ook stuurt: geef de fout door, en zet de key goed.",
            "wait": "Wachten helpt een drukke server, geen fout request. Zie [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl).",
            "now": "Zou hetzelfde request, met dezelfde key, een ander antwoord krijgen? Zie [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl).",
        },
    )
    quick_401
    return (quick_401,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="Met `base` 2: hoeveel seconden wacht `time.sleep(base * 2 ** attempt)` als `attempt` 2 is?",
        options={
            "four": "4",
            "six": "6",
            "eight": "8",
        },
        correct="eight",
        why={
            "eight": "`2 ** 2` is 4, en 2 keer 4 is 8: de wachttijden zijn 2, 4 en 8 seconden.",
            "four": "Dat is alleen `2 ** 2`. Keer wat? Zie [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl).",
            "six": "`**` is een macht, geen keer. Zie [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De server van de course draait op je eigen computer: geen key, geen rekening. Een gehoste API staat open voor het hele internet. Hij wil bij elk request een key, om te weten wie het vraagt en wie betaalt, en hij rekent per token. Drie gewoontes houden dat in de hand: de key blijft uit je code, veel requests die mogen wachten gaan als één batch, en je telt wat elke aanroep kost.

    **Een voorbeeld uit het dagelijks leven.** De sleutel van een deelauto: wie hem heeft, kan rijden, en jij betaalt. Je plakt hem niet op het raam, en je leent hem niet uit aan vreemden. Dat is maar een vergelijking; hieronder zie je waar een echte key heen gaat.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **API key** | een geheime tekst, zoals `sk-…`, die zegt wie een request stuurt, en dus wie betaalt; hij gaat mee in de header `Authorization: Bearer` en de key |
    | **environment variable** | een naam met een waarde die een programma van de computer krijgt als het start, buiten zijn code; in een terminal zet `export MODEL_API_KEY=sk-…` er een |
    | `os.environ` | de environment variables in Python, als dictionary: `os.environ["MODEL_API_KEY"]` geeft de waarde, na `import os` |
    | **batch API** | een manier om veel requests tegelijk te sturen, in één bestand, voor antwoorden later, vaak tegen een lagere prijs |

    ### Waar de key heen gaat
    """)
    return


@app.cell(hide_code=True)
def picture_key():
    draw.flow(
        [("De terminal", ["voordat de course start:", "export MODEL_API_KEY=sk-…"]),
         ("Je code", ["import os", "os.environ[\"MODEL_API_KEY\"]"]),
         ("De server", ["controleert de key,", "telt de tokens"])],
        ["de omgeving", "Authorization:\nBearer sk-…"], ["", "het antwoord,\nmet usage"],
        title="De key: uit de omgeving, door je code, naar de server")
    return


@app.cell(hide_code=True)
def text_key():
    mo.vstack([
        mo.md(r"""
        De key staat nooit in je code: de code vraagt hem aan de omgeving, op het moment dat hij draait. Zo kun je je code delen, en op GitHub zetten, zonder je key; wie hem draait, zet een eigen key. De server leest de header, en weigert een request zonder geldige key met 401.
        """),
        L.watch_out("Schrijf nooit een key in je code of in een notebook. Code belandt op GitHub, notebooks worden gedeeld, en programma's doorzoeken de hele dag openbare code op keys. Is een key uitgelekt: verwijder hem meteen bij de aanbieder, en maak een nieuwe."),
    ])
    return


@app.cell(hide_code=True)
def text_batch():
    mo.md(r"""
    ### Veel requests tegelijk: een batch

    Samenvattingen van 5.000 reviews, een voor een, zijn 5.000 aanroepen: de rate limit houdt je tegen, en elke aanroep kost de volle prijs. Een batch API neemt ze allemaal in één bestand, werkt ze af wanneer het de aanbieder uitkomt, en zet de antwoorden voor je in een bestand, vaak binnen uren in plaats van seconden. In ruil daarvoor vragen veel aanbieders een lagere prijs. Het bestand heeft één regel JSON per request: een eigen id, `custom_id`, en het request zelf, dezelfde body die je al stuurt. In het bestand met de antwoorden heeft elk antwoord dezelfde `custom_id`, zodat je weet bij welke vraag het hoort.

    ### De rekening

    Elk antwoord telt zijn tokens in `usage`. Met een prijs per miljoen tokens kost een aanroep `prompt_tokens * price_in / 1_000_000 + completion_tokens * price_out / 1_000_000`; `1_000_000` is een miljoen, en Python slaat de liggende streepjes over. De aanroep van de course in [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl) had 19 tokens erin en 13 eruit. Tegen verzonnen prijzen van 3 dollar per miljoen erin en 15 eruit is dat 19 × 3 + 13 × 15 = 252 miljoenste dollar: bijna niets. Maar een app met tienduizend gebruikers doet zulke aanroepen de hele dag, en de tokens die eruit komen kosten het meest. Wie na elke aanroep `usage` optelt, houdt de rekening in zicht.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: keys, batches en de rekening

    1. **Houd de key uit de code:** zet hem in de omgeving, met `export MODEL_API_KEY=sk-…` in de terminal voordat je de course start.
    2. **Lees hem als de code draait:** `import os`, en dan `os.environ["MODEL_API_KEY"]`.
    3. **Stuur hem mee:** de header `"Authorization": "Bearer " + key`, met een spatie na `Bearer`.
    4. **Tel elke aanroep:** tel `usage` keer de prijs per miljoen op bij een totaal.
    5. **Veel requests die mogen wachten:** schrijf één regel JSON per request, met een `custom_id`, en stuur ze als één batch.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De regels van een batchbestand, voor twee vragen:

    ```python
    import json

    questions = ["Name a color.", "Name a fruit."]
    lines = []
    for i in range(len(questions)):
        request = {"custom_id": f"q{i}", "method": "POST", "url": "/v1/chat/completions",
                   "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": questions[i]}], "max_tokens": 50}}
        lines.append(json.dumps(request))
    for line in lines:
        print(line)
    ```

    ```
    {"custom_id": "q0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Name a color."}], "max_tokens": 50}}
    {"custom_id": "q1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Name a fruit."}], "max_tokens": 50}}
    ```

    1. **Eén request per vraag:** `custom_id` is `q0` en `q1`, gemaakt van de plek in de lijst, zodat elk antwoord zijn vraag terugvindt.
    2. **Waar het heen gaat:** `"method"` en `"url"` zeggen voor welk endpoint elk request is; `"body"` is de body die je zelf zou sturen.
    3. **Eén regel elk:** `json.dumps` schrijft de dictionary als JSON op één regel, zonder regeleinden.
    4. **Het bestand:** deze regels, bewaard als bestand dat eindigt op `.jsonl`, *JSON lines*, upload je naar een batch API. Ze volgen het formaat voor batches van OpenAI, een voorbeeld in november 2026; andere aanbieders gebruiken bijna dezelfde regels.
    """)
    return


@app.cell(hide_code=True)
def text_key_exercise():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De key uit de omgeving

        Deze `ask` verstuurt een vraag zonder key, en de server wil er een. Stuur de key mee, uit de environment variable `MODEL_API_KEY`. De check roept je `ask` aan tegen twee stand-ins, elk met een eigen key, en zet tijdens de aanroep elke key in `MODEL_API_KEY`; het plaatje onder de check laat het request zien dat je code stuurde, met de key verborgen.
        """),
        L.task("Lees de key uit de environment variable `MODEL_API_KEY`, en stuur hem mee in de header `Authorization`, na `Bearer ` en een spatie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    box_key = L.code_box("key", lines=13)
    box_key
    return (box_key,)


@app.cell(hide_code=True)
def check_key(box_key):
    result_key = L.check("key", box_key.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "key",
        [
            "Stap 2 en 3 van het recept: lees de key, en stuur hem in een header.",
            "`import os` bovenaan; `os.environ` bevat de environment variables, als dictionary. De header heet `\"Authorization\"`, en zijn waarde begint met `\"Bearer \"`.",
            "`headers = {\"Content-Type\": \"application/json\", \"Authorization\": \"Bearer \" + ___}`.",
        ],
        why="""```python
import json
import os
import urllib.request


def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
    request = urllib.request.Request(url + "/chat/completions", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

De code noemt alleen de naam van de variabele; de key zelf komt uit de omgeving als de code draait.""",
    )
    return


@app.cell(hide_code=True)
def text_bill():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De rekening

        Schrijf `bill(usages, price_in, price_out)`: `usages` is een lijst met de `usage` van aanroepen, elk een dictionary met `"prompt_tokens"` en `"completion_tokens"`; de prijzen zijn in dollar per miljoen tokens. Geef terug wat alle aanroepen samen kosten. Voor de twee aanroepen in het voorbeeld, tegen 3 en 15 dollar per miljoen, is dat 0,0135 dollar:

        ```python
        bill([{"prompt_tokens": 1200, "completion_tokens": 300}, {"prompt_tokens": 800, "completion_tokens": 200}], 3, 15)
        ```
        """),
        L.task("Maak `bill` af: tel voor elke aanroep zijn tokens erin tegen `price_in` en zijn tokens eruit tegen `price_out` op bij `total`, per miljoen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bill():
    box_bill = L.code_box("bill", lines=7)
    box_bill
    return (box_bill,)


@app.cell(hide_code=True)
def check_bill(box_bill):
    result_bill = L.check("bill", box_bill.value)
    result_bill
    return (result_bill,)


@app.cell(hide_code=True)
def hints_bill():
    L.hints(
        "bill",
        [
            "Een `for`-loop over `usages`; elke aanroep komt bij `total`.",
            "Eén aanroep kost zijn `\"prompt_tokens\"` keer `price_in`, plus zijn `\"completion_tokens\"` keer `price_out`; de prijzen zijn per miljoen tokens, dus deel elk door `1_000_000`.",
            "In de loop, de tokens erin: `total = total + ___ * price_in / 1_000_000`; een tweede regel zoals deze doet hetzelfde voor de tokens eruit.",
        ],
        why="""```python
def bill(usages, price_in, price_out):
    total = 0
    for usage in usages:
        total = total + usage["prompt_tokens"] * price_in / 1_000_000
        total = total + usage["completion_tokens"] * price_out / 1_000_000
    return total
```

Twee prijzen, omdat tokens erin en tokens eruit verschillend kosten, en een som over elke aanroep.""",
    )
    return


@app.cell(hide_code=True)
def text_batch_choice():
    mo.md(r"""
    ### Oefening 3. Een klus voor een batch
    """)
    return


@app.cell(hide_code=True)
def choice_batch_cell():
    choice_batch = L.choice(
        "batch",
        question="Bij welke van deze klussen past een batch het best?",
        options={
            "reviews": "Samenvattingen van 5.000 productreviews, voor een rapport volgende week",
            "chat": "De antwoorden in een chat, terwijl iemand erop wacht",
            "now": "Eén vraag waarop je nu meteen antwoord wilt",
        },
        correct="reviews",
        why={
            "reviews": "Veel requests, en niemand wacht erop: ze mogen uren later komen, misschien tegen een lagere prijs, en de rate limit staat niet in de weg.",
            "chat": "Een batch antwoordt binnen uren. Kan iemand in de chat zo lang wachten?",
            "now": "Eén request wint niets bij een batch, en het antwoord zou uren later komen. Welke klus heeft veel requests, en tijd?",
        },
    )
    choice_batch
    return (choice_batch,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De key gaat mee met hetzelfde request als in [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en de rate limit uit [Als een aanroep mislukt: fouten, limieten en opnieuw proberen](/course/open/build-errors-retries?lang=nl) telt per key. De tokens die `usage` telt, zijn die uit [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl). De rekening hangt ook af van welk model je vraagt: de volgende les, [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl), stuurt makkelijke vragen naar een goedkoper model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_batch, result_bill, result_key):
    L.wrap_up(result_key, result_bill, choice_batch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
