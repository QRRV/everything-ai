# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-errors-retries. Route: course/route/lessen/bouwen.yaml. Checks, de stand-ins en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-errors-retries", "nl")
    C = L.checks   # de stand-ins die eerst nee zeggen, en de plaatjes van de pogingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en één over [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_urlopen():
    quick_urlopen = L.choice(
        "quick-urlopen",
        question="Wat doet `urllib.request.urlopen(request)` met een request dat `data=` heeft?",
        options={
            "get": "Het haalt het adres op met een GET, en laat de data weg",
            "file": "Het schrijft de data naar een bestand op je computer",
            "post": "Het stuurt de data met een POST, en geeft de response terug",
        },
        correct="post",
        why={
            "post": "Met `data=` is het request een POST: de body gaat naar de server, en `response.read()` leest wat er terugkomt.",
            "get": "Een request zonder `data=` is een GET. Wat verandert er met `data=`? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "file": "`urlopen` praat over het netwerk met een server. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_urlopen
    return (quick_urlopen,)


@app.cell(hide_code=True)
def quick_check_try():
    quick_try = L.choice(
        "quick-try",
        question="Wat laat dit programma zien?\n\n```python\ntry:\n    print(int(\"7\"))\n    print(int(\"seven\"))\nexcept ValueError:\n    print(\"no number\")\nprint(\"on\")\n```",
        options={
            "traceback": "`7`, en dan een traceback",
            "all": "`7`, `no number` en `on`",
            "skip": "`no number` en `on`",
        },
        correct="all",
        why={
            "all": "`int(\"7\")` lukt en laat 7 zien; `int(\"seven\")` stopt met een `ValueError`, dus springt Python naar de `except`, en daarna gaat het programma verder.",
            "traceback": "De `ValueError` gebeurt in de `try`, en de `except` noemt precies die soort. Zie [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).",
            "skip": "De regels in de `try` draaien een voor een, tot er een mislukt. Welke mislukt er? Zie [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl).",
        },
    )
    quick_try
    return (quick_try,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De API van een model zegt af en toe nee: te veel requests in één minuut, een server die het druk heeft, een key die niet klopt. Code die bij de eerste fout stopt, maakt een app die zonder goede reden kapotgaat. Code die het meteen opnieuw probeert, steeds weer, maakt het probleem van de server erger. Goede code leest waarom de aanroep mislukte, probeert alleen opnieuw wat een moment later kan lukken, en wacht elke keer langer.

    **Een voorbeeld uit het dagelijks leven.** Je belt een winkel en de lijn is bezet. Je probeert het na een minuut opnieuw, en daarna na een paar minuten meer, niet elke seconde. En als het nummer niet bestaat, helpt opnieuw bellen helemaal niet. Dat is maar een vergelijking; hieronder zie je wat een server zegt, en hoe lang je wacht.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **HTTP-statuscode** | een getal in elke response dat zegt hoe het ging: `200` is goed, een status in de 400-reeks zegt dat het request niet klopt, een vanaf 500 dat de server een probleem heeft |
    | **rate limit** | hoeveel requests of tokens een key per minuut mag sturen; daarboven antwoordt de server `429`, vaak met een header `Retry-After`: het aantal seconden dat je wacht |
    | **timeout** | hoe lang je hooguit op de server wacht: `urlopen(request, timeout=30)` stopt met een `TimeoutError` als de server 30 seconden niets laat horen; die 30 seconden tellen voor elke wachtpauze, niet voor het hele antwoord, dus een trage stream kan langer duren; zonder `timeout=` kan het eeuwig wachten |
    | **retry** | hetzelfde request opnieuw sturen nadat het mislukte |
    | **exponential backoff** | retries met wachttijden die verdubbelen: `base`, dan twee keer zo lang, dan vier keer, dus `base * 2 ** attempt` |
    | `urllib.error.HTTPError` | de fout die urllib geeft bij een status vanaf 400: `e.code` is de status, `e.headers` de headers, `e.read()` de body |

    Dit zijn de statussen die de API van een model het vaakst geeft:

    | Status | Wat het betekent |
    |---|---|
    | `400` | het request is niet geldig, zoals een body zonder `"messages"` |
    | `401` | er kwam geen geldige key mee met het request |
    | `403` | de key mag dit niet |
    | `404` | dat adres bestaat niet, of dat model niet |
    | `429` | te veel requests: de rate limit is bereikt |
    | `500` | er ging iets mis in de server |
    | `503` | de server is bezet, of even uit de lucht |

    Welke zijn nog een poging waard? Vraag je af: zou hetzelfde request een moment later een ander antwoord krijgen?

    ### Wachttijden die verdubbelen

    Een server die 429 zei, daarna twee keer 503, en toen antwoordde, met `base` 1 seconde:
    """)
    return


@app.cell(hide_code=True)
def picture_waits():
    mo.Html(C.waits_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_waits():
    _w = C.all_waits()
    mo.vstack([
        mo.md(rf"""
        De eerste wachttijd is 1 seconde, dan 2, dan 4: `base * 2 ** attempt`, met `attempt` 0, 1 en 2. Waarom niet meteen opnieuw? Als duizenden programma's op hetzelfde moment een 503 krijgen, en ze allemaal direct opnieuw vragen, krijgt de server dezelfde vloed weer. Wachttijden die verdubbelen spreiden de retries, en geven de server ruimte. Ze groeien ook snel: tien pogingen zouden samen 1 + 2 + … + {_w['last']:g} = {_w['total']:g} seconden wachten, dus je stelt ook een aantal pogingen in, en na de laatste geef je de fout door.
        """),
        L.watch_out("Een timeout stopt met een `TimeoutError`, niet met een `HTTPError`; een timeout tijdens het verbinden komt als een `urllib.error.URLError`, met de `TimeoutError` erin. Allebei zijn ze nog een poging waard: vang ze op met `except TimeoutError:` en `except urllib.error.URLError:` onder de eerste."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: opnieuw proberen, met backoff

    1. **Stel een timeout in:** `urlopen(request, timeout=30)`.
    2. **Probeer in een loop:** `for attempt in range(tries):`, met `try:` om de aanroep, en `return` de tekst als het lukt.
    3. **Vang de fout op:** `except urllib.error.HTTPError as e:`. De `as e` geeft de fout een naam, zodat je zijn status kunt lezen, `e.code`.
    4. **Beslis:** als de status niet verandert, of dit de laatste poging was, geef de fout dan door met `raise`. In een `except` geeft `raise` op zichzelf dezelfde fout opnieuw, voor de code die de jouwe aanriep.
    5. **Wacht, en ga nog een rondje:** `time.sleep(base * 2 ** attempt)`; `time.sleep(s)` wacht `s` seconden. Als de server `Retry-After` stuurt, wacht dan zo lang.

    In november 2026 doet het package `openai` dit voor je: het probeert het zelf een paar keer opnieuw, met backoff, en `max_retries` stelt in hoe vaak.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Wat zegt een server die het druk heeft? Hier is `url` het adres van de stand-in van de course, die het eerste request met 429 beantwoordt.

    ```python
    import json
    import urllib.error
    import urllib.request

    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(e.code, e.headers["Retry-After"])
        print(json.loads(e.read())["error"]["message"])
    ```

    ```
    429 2
    Too many requests: wait a moment and try again.
    ```

    1. **Het request:** hetzelfde als in [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), nu met een timeout van 30 seconden.
    2. **De fout:** de server antwoordt met status 429, dus `urlopen` stopt met een `HTTPError`, en Python springt naar de `except`.
    3. **De status en de header:** `e.code` is `429`, en de header `Retry-After` zegt `2`: wacht twee seconden.
    4. **De body:** ook een fout heeft een body, JSON met de eigen woorden van de server onder `"error"`; `e.read()` geeft die als bytes, zoals `response.read()` dat doet.
    """)
    return


@app.cell(hide_code=True)
def text_should_retry():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Nog een poging waard?

        Schrijf `should_retry(status)`: `True` als hetzelfde request een moment later een ander antwoord kan krijgen, `False` als het dezelfde fout krijgt, hoe vaak je het ook stuurt. De check probeert de statussen uit de tabel, en nog een paar vanaf 500.
        """),
        L.task("Maak `should_retry` af: geef `True` voor een status die nog een poging waard is, en anders `False`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_should_retry():
    box_retry = L.code_box("should_retry", lines=4)
    box_retry
    return (box_retry,)


@app.cell(hide_code=True)
def check_should_retry(box_retry):
    result_retry = L.check("should_retry", box_retry.value)
    result_retry
    return (result_retry,)


@app.cell(hide_code=True)
def hints_should_retry():
    L.hints(
        "should_retry",
        [
            "Vraag je bij elke status in de tabel af: klopt het request niet, of heeft de server het nu te druk, of een probleem?",
            "Bij 429 kan het na even wachten goed gaan, en bij een probleem in de server ook: elke status vanaf 500. De andere statussen in de 400-reeks blijven zoals ze zijn.",
            "Eén regel: `return`, dan `status == 429`, dan `or`, dan een vergelijking die waar is vanaf 500.",
        ],
        why="""```python
def should_retry(status):
    return status == 429 or status >= 500
```

Een foute body, key of een fout adres blijft fout; te veel requests en een drukke server kunnen zo over zijn.""",
    )
    return


@app.cell(hide_code=True)
def text_ask_again():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Opnieuw proberen, en elke keer langer wachten

        `ask_again(url, question, tries, base)` verstuurt een vraag, hooguit `tries` keer. De loop en de `try` staan er al; vul in wat er onder `except` gebeurt: geef de fout door als `should_retry` nee zegt, of als het de laatste poging was, en wacht anders `base * 2 ** attempt` seconden. Hij gebruikt je `should_retry` uit oefening 1. De check draait je functie tegen drie stand-ins: een die 429 en 503 zegt en dan antwoordt, een die een verkeerde key krijgt, en een die uit de lucht blijft. Het plaatje onder de check laat je pogingen op een tijdlijn zien.
        """),
        L.task("Onder `except`: geef de fout door als hij niet verandert of dit de laatste poging was; wacht anders, `base` seconden na de eerste poging en twee keer zo lang na elke volgende."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask_again():
    box_again = L.code_box("ask_again", lines=18)
    box_again
    return (box_again,)


@app.cell(hide_code=True)
def check_ask_again(box_again, box_retry):
    result_again = L.check("ask_again", box_again.value, C.again_inputs(box_retry.value))
    result_again
    return (result_again,)


@app.cell(hide_code=True)
def hints_ask_again():
    L.hints(
        "ask_again",
        [
            "Stap 4 en 5 van het recept: beslis eerst of je stopt, en wacht dan.",
            "`if not should_retry(e.code) or attempt == tries - 1:` en daaronder, verder ingesprongen, `raise`.",
            "Na die `if`, even ver ingesprongen: `time.sleep(base * 2 ** attempt)`. Daarna gaat de loop vanzelf verder.",
        ],
        why="""```python
def ask_again(url, question, tries, base):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.loads(response.read())
            return answer["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if not should_retry(e.code) or attempt == tries - 1:
                raise
            time.sleep(base * 2 ** attempt)
```

`raise` geeft een fout door die niet verandert, of de laatste; anders wacht de functie, en stuurt de loop het request opnieuw, elke keer na een wachttijd die twee keer zo lang is.""",
    )
    return


@app.cell(hide_code=True)
def text_crowd():
    mo.md(r"""
    ### Oefening 3. Iedereen tegelijk
    """)
    return


@app.cell(hide_code=True)
def choice_crowd_cell():
    choice_crowd = L.choice(
        "crowd",
        question="Een drukke server antwoordt op hetzelfde moment 503 aan tienduizend apps. Elke app probeert het meteen opnieuw, steeds weer, zonder te wachten. Wat gebeurt er?",
        options={
            "worse": "De server krijgt nog meer requests dan eerst, en blijft overbelast",
            "fine": "De server haalt het in, want elke app vraagt nog maar één keer",
            "faster": "De apps krijgen hun antwoord eerder, omdat ze blijven vragen",
        },
        correct="worse",
        why={
            "worse": "Elke 503 brengt meteen een nieuw request, dus de vloed blijft. Met wachttijden die verdubbelen spreiden de retries zich over de tijd, en krijgt de server ruimte om het in te halen.",
            "fine": "Zonder te wachten vraagt elke app steeds opnieuw, zo snel als hij kan. Wat doet dat met het aantal requests?",
            "faster": "Een overbelaste server wordt met elk extra request trager. Wat zou hem ruimte geven?",
        },
    )
    choice_crowd
    return (choice_crowd,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De fouten zijn wat de server uit [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl) zegt als er iets mis is, opgevangen met `try` en `except` zoals in [Fouten opvangen en zelf een fout geven](/course/open/py-exceptions?lang=nl). De rate limit van een gehoste API telt requests en tokens per minuut, per key: de volgende les, [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl), brengt de key erbij, en veel requests tegelijk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_crowd, result_again, result_retry):
    L.wrap_up(result_retry, result_again, choice_crowd)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
