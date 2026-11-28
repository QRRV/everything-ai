# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-serving-api. Route: course/route/lessen/systemen.yaml. Checks, het vraagmodel van de course, de server en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-serving-api", "nl")
    C = L.checks   # het vraagmodel van de course, de verzoeken van de les, de server en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl) en [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_online():
    quick_online = L.choice(
        "quick-online",
        question="Wanneer maakt het model een voorspelling bij online inference?",
        options={
            "night": "Eén keer per nacht, voor alle leden tegelijk, opgeslagen voor de app",
            "right": "Op het moment dat een verzoek binnenkomt, uit de invoer van dat moment",
            "event": "Na elke gebeurtenis van een stream, zonder dat iemand erom vraagt",
            "phone": "Alleen op de eigen telefoon van de gebruiker, zodat er helemaal geen netwerk nodig is",
        },
        correct="right",
        why={
            "right": "Online inference maakt de voorspelling als er een verzoek binnenkomt, terwijl iemand erop wacht. Deze les bouwt het deel dat wacht: een server die elk verzoek aanneemt en beantwoordt.",
            "night": "Alles vooraf, één keer per nacht, is batch inference. Wanneer maakt online inference de voorspelling? Zie [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl).",
            "event": "Een voorspelling na elke gebeurtenis, zonder verzoek, is streaming inference. Wat zet online inference in gang? Zie [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl).",
            "phone": "Op de telefoon zegt *waar* het model draait, niet *wanneer*. Wanneer maakt online inference de voorspelling? Zie [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl).",
        },
    )
    quick_online
    return (quick_online,)


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="Een server laadt een model één keer, bij het starten. Wat heeft de server nodig naast model.safetensors?",
        options={
            "nothing": "Niets meer: het bestand met de weights bevat het hele netwerk",
            "right": "config.json, met de class van het netwerk en de schaling van de input",
            "data": "De trainingsdata, om bij elk verzoek de meest gelijke uren op te zoeken",
            "notebook": "Het notebook waarin het model getraind is, om bij elke start alle cellen opnieuw uit te voeren",
        },
        correct="right",
        why={
            "right": "De weights alleen zeggen niet welke Python-class het model is, of hoe een input geschaald wordt: config.json zegt dat wel, met de class en de instellingen, en met het gemiddelde en de standaardafwijking van elke feature. Een server leest beide één keer, bij het starten, en houdt het model in het geheugen; elk verzoek voert het model daarna alleen uit.",
            "nothing": "Een safetensors-bestand bevat tensors met een naam en een vorm. Zegt het bestand welke Python-class er gebouwd moet worden, of hoe een input geschaald wordt? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
            "data": "Het model leerde van de data. Heeft het model de data zelf nog nodig om te antwoorden? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
            "notebook": "Een pakket bestaat juist zodat niemand het model opnieuw hoeft te trainen. Wat zit er naast de weights in? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een getraind model in een map is nuttig voor één soort programma: een Python-programma dat het model laadt. Alleen zijn de programma's die de voorspellingen nodig hebben vaak andere. Bij een fietsdeelbedrijf zoals Capital Bikeshare in Washington, D.C. wil een planner voor de vrachtwagens weten hoeveel fietsen er om 17:00 verhuurd worden, wil een dashboard de getallen voor de hele dag, en wil de app er één voor de kaart. Andere teams schreven deze programma's, misschien in andere talen, en de programma's draaien op andere computers. Een PyTorch-model laden kunnen deze programma's niet, en dat hoort ook niet nodig te zijn.

    Het gebruikelijke antwoord: zet het model achter een API. Een klein programma, een server, houdt het model geladen en wacht. Elk programma stuurt de server een verzoek via HTTP, de taal van het web, met de input als JSON; de server controleert het verzoek, voert het model uit, en stuurt het antwoord terug als JSON, met een getal dat zegt hoe het ging. De andere programma's hoeven alleen te weten wat er heen moet, en wat er terugkomt.

    **Een voorbeeld uit het dagelijks leven.** Op het gemeentehuis lever je een ingevuld formulier in bij het juiste loket. Een medewerker controleert het formulier eerst: met een leeg veld krijg je het formulier terug, met een briefje erbij dat zegt welk veld; bij het verkeerde loket word je naar een ander loket gestuurd; met een goed formulier krijg je je antwoord, zonder te weten hoe het gemeentehuis vanbinnen werkt. Het formulier en de loketten zijn de afspraak tussen jou en het gemeentehuis, en elke bezoeker gebruikt dezelfde.
    """)
    return


@app.cell(hide_code=True)
def predict_midnight():
    guess_midnight = L.guess(
        "midnight",
        question="Een model voorspelt hoeveel fietsen er in een uur verhuurd worden, uit het uur en 10 features van de dag. De app van een collega schrijft middernacht als uur 24 in plaats van 0, en stuurt dat meteen naar het model, zonder controle. Wat gebeurt er, denk je?",
        options={
            "next": "Het model antwoordt voor 0:00, het begin van de nieuwe dag",
            "zero": "Het model antwoordt 0 fietsen, want het zag uur 24 nooit",
            "last": "Het model antwoordt voor 23:00, het laatste uur dat het kent",
            "right": "De code van het model stopt met een fout",
        },
        reveal="**De code van het model stopt met een fout.** Het uur gaat het model in als 24 one-hot-kolommen, één voor elk uur van 0 tot en met 23, zoals in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl). Uur 24 heeft geen kolom, dus stopt `one_hot` van PyTorch met *Class values must be smaller than num_classes*. Een server die zo'n verzoek aan het model doorgeeft, antwoordt met status 500, *Internal Server Error*: alsof de server stukging, terwijl het verzoek fout was. Een ontbrekend veld, of het woord \"warm\" als temperatuur, laat de code ook stoppen. In deze les controleert de server elk verzoek voordat het model het verzoek ziet.",
    )
    guess_midnight
    return (guess_midnight,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Model serving.** **Model serving** is een model als dienst draaien: een programma houdt het model geladen en beantwoordt verzoeken om voorspellingen, het ene na het andere, voor elk programma dat erom vraagt. Zo doet een model online inference (zie [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl)): een voorspelling op het moment van elk verzoek.

        **Webserver.** Dat programma is een **webserver**: een programma dat op verzoeken wacht en elk verzoek beantwoordt. De server luistert op een *poort*, een getal dat de programma's op één computer uit elkaar houdt, zoals 8000. Het adres 127.0.0.1, ook localhost genoemd, is de computer zelf: een server die daar luistert, neemt alleen verzoeken aan van programma's op dezelfde computer.

        **HTTP-request.** Een **HTTP-request** heeft een methode, zoals GET om iets op te halen of POST om iets te sturen; een pad, zoals /predict, dat zegt waar het verzoek voor is; headers, korte regels zoals Content-Type: application/json; en een body met de data: hier de input van het model als JSON. Een pad dat een server beantwoordt, met de methode erbij, is een **endpoint**: POST /predict voor een voorspelling, en GET /health om te zien of de server draait. De eerste regel van een verzoek bevat de methode, het pad en de versie van HTTP, zoals `HTTP/1.1`; de server van deze les antwoordt in de oudere versie `HTTP/1.0` en sluit de verbinding na elk antwoord.

        **Statuscode.** Elk antwoord begint met een **HTTP-statuscode**: een getal van drie cijfers dat zegt hoe het ging, nog voordat de client de body leest. Het eerste cijfer zegt wie aan zet is: een 2 voor succes, een 4 voor een verzoek dat de client moet aanpassen, een 5 voor een server die erop stukging, en dan kan hetzelfde verzoek later wel werken.

        | Code | Naam | Betekent |
        |---|---|---|
        | 200 | OK | het ging goed, en de body bevat het antwoord |
        | 400 | Bad Request | het verzoek is fout: er ontbreekt een veld, of een waarde kan niet kloppen |
        | 404 | Not Found | de server heeft zo'n pad niet |
        | 405 | Method Not Allowed | het pad bestaat, maar niet met deze methode, zoals een GET naar /predict |
        | 500 | Internal Server Error | de server ging stuk op het verzoek, bijvoorbeeld omdat de code stopte met een fout |

        **API.** De endpoints samen, met wat elk endpoint aanneemt en teruggeeft, zijn de **API** van de dienst: de afspraak tussen de server en elk programma dat de dienst gebruikt. Een client die zich eraan houdt, hoeft niets te weten van PyTorch, of van het model.

        **De handler.** Op de server krijgt één functie elk verzoek, en deze functie beslist over het antwoord: de handler. De handler kijkt naar het pad en de methode, controleert de input, voert het model uit, en maakt van de output waarden die JSON kan bevatten. In deze les schrijf je de handler, `handle`, en de course voert de handler uit in een echte server op je eigen computer: `ThreadingHTTPServer` uit `http.server` van Python, op 127.0.0.1. Daarachter zit een klein neural network dat voor één uur voorspelt hoeveel fietsen Capital Bikeshare verhuurt, uit het uur en 10 features van de dag en het weer: het model van *Project: een model van de vraag, bijgehouden en ingepakt*, dat de course hier opnieuw traint.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Op grote schaal.** **FastAPI** is een Python-library die het werk rond de handler doet. Een decorator, `@app.post("/predict")` boven een functie, maakt van de functie een endpoint. FastAPI leest de JSON, controleert de JSON aan de hand van de velden die je opgeeft, antwoordt 422 als er een veld ontbreekt of van de verkeerde soort is, een 4xx-code die specifieker is dan 400, en schrijft wat de functie teruggeeft als JSON. De documentatie van Python waarschuwt dat `http.server` niet bedoeld is voor productie: op je eigen computer is het prima, en een dienst voor echte gebruikers draait op een framework zoals FastAPI. Tussen de programma's van één bedrijf stuurt gRPC verzoeken als compacte bytes in plaats van JSON-tekst. En Triton Inference Server van NVIDIA draait veel modellen tegelijk op GPU's, en zet verzoeken die vlak na elkaar binnenkomen in één batch. Het idee blijft hetzelfde: een verzoek komt binnen, wordt gecontroleerd, gaat door het model, en er gaat een antwoord uit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **model serving** | | een model draaien als dienst voor verzoeken om voorspellingen | |
        | **webserver** | | een programma dat op een poort wacht op verzoeken via HTTP, en elk verzoek beantwoordt | `ThreadingHTTPServer` |
        | **HTTP-request** | | een bericht aan een server: een methode, een pad, headers en een body | `POST /predict` |
        | **endpoint** | | een pad dat een server beantwoordt, met de methode erbij | `GET /health` |
        | **HTTP-statuscode** | | het getal van drie cijfers aan het begin van een antwoord, dat zegt hoe het ging | `200`, `404` |
        | **API** | A P I | de endpoints van een dienst, met wat elk endpoint aanneemt en teruggeeft | |
        | **FastAPI** | fast A P I | een Python-library om een API te bouwen, waarin een decorator van een functie een endpoint maakt | `@app.post("/predict")` |
        """),
        L.watch_out("Status 200 betekent niet dat de voorspelling klopt: het betekent dat de server kon antwoorden. Een temperatuur van 26,2, in graden, waar de data 0,64 heeft, krijgt ook status 200, met een getal dat ver ernaast zit. Het onderwerp van de volgende les is wat er met een input moet gebeuren voordat het model de input ziet."),
        L.watch_out("Een pad moet letter voor letter kloppen: /Predict, met een hoofdletter P, of /predict/, met een schuine streep aan het eind, is een ander pad, en krijgt 404, tenzij de server iets anders zegt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model achter een API zetten:

    1. Laad het model één keer, bij het starten van de server, en houd het model in het geheugen: niet opnieuw bij elk verzoek.
    2. Kies de endpoints, met een methode en een pad voor elk endpoint: POST /predict voor een voorspelling, GET /health om te zien of de server draait.
    3. Controleer elk verzoek voordat het model het verzoek ziet: een ontbrekend veld, een waarde die geen getal is, een uur dat het model niet kent. Beantwoord zo'n verzoek met 400 en wat er mis is.
    4. Voer het model uit, en maak van de output waarden die JSON kan bevatten: een heel aantal fietsen, geen tensor.
    5. Stuur de statuscode mee met het antwoord als JSON, en houd de server op 127.0.0.1 zolang je bouwt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.start(C.handle)` start de server van de course op 127.0.0.1, op een vrije poort, met de eigen `handle` van de course. `C.send(server, method, path, body)` stuurt de server een verzoek, zoals elke client dat doet, met `body` als JSON, en geeft de statuscode en de JSON van het antwoord terug. `C.DAY[8]` is de body voor 8:00 op donderdag 14 juli 2011, een heldere zomerdag: het uur en de 10 features van dat uur, zoals in de data. De cel start de server met `with`, stuurt drie verzoeken, en de server stopt weer als de regels onder `with` klaar zijn, ook als een van de regels met een fout stopt. Voer de cel uit.
    """)
    return


@app.cell
def example_server():
    with C.start(C.handle) as _server:
        print("luistert op", _server.address, "poort", _server.port)
        print(C.send(_server, "GET", "/health"))
        print(C.send(_server, "POST", "/predict", C.DAY[8]))
        print(C.send(_server, "POST", "/forecast", C.DAY[8]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De server luistert op 127.0.0.1, op een poort die de computer gaf, dus is het getal elke keer anders. GET /health krijgt 200 en `{"status": "ok"}`. POST /predict met de 11 getallen van 8:00 krijgt 200 en `{"bikes": 431}`: op deze ochtend werden er 486 fietsen verhuurd. /forecast is geen endpoint van deze server, dus krijgt het verzoek 404, met wat er mis is. Alles ging via HTTP op je eigen computer, en de server is weer gestopt. In Oefening 1 schrijf je de regel om van de output van het model deze 431 te maken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Van een tensor naar JSON

        `answer(request)` krijgt de body van een verzoek als dict, met `hr` en de 10 features, en geeft een dict terug: de dict die de server als JSON stuurt. `expect_one(request)` voert het model van de course uit op dat uur en geeft `pred`: een tensor zonder assen, met het aantal fietsen dat het model verwacht. Maar json kan geen tensor schrijven, en geen vrachtwagen kan 431,2 fietsen brengen. Vervang de `...` zodat `bikes` het getal in `pred` is als gewoon Python-getal, afgerond op een heel aantal fietsen.

        De check roept `answer(C.DAY[8])` aan, waar het model 431,2 fietsen verwacht. Dan 10:00, 17:00 en 12:00 van dezelfde dag, en 17:00 op 7 december 2011, in lichte regen.
        """),
        L.task("Maak van `bikes` een heel aantal fietsen, als gewoon Python-getal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=4)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "answer",
        [
            "Stap 4 van het recept: maak van de output een waarde die JSON kan bevatten. `pred` is een tensor, en `round` rondt alleen een gewoon Python-getal af: haal eerst het getal uit `pred`, rond het getal dan af, en zet het in `bikes`.",
            "In [Tensors in PyTorch](/course/open/torch-tensors?lang=nl) gaf een tensor zonder assen het getal als Python-getal met een van de methodes van de tensor, voor een `print` of een lijst met losses. Welke methode was dat?",
            "`    bikes = round(pred.___)`",
        ],
        why="`bikes = round(pred.item())`: `pred.item()` haalt het getal uit de tensor als gewoon Python-float, 431,2 om 8:00, en `round` maakt er het hele getal 431 van. Dat kan json schrijven, en een programma in elke andere taal kan het lezen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_handle():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een verzoek dat het model niet aankan

        `handle(method, path, request)` krijgt de methode, het pad en de body van een verzoek, en geeft twee dingen terug: de statuscode en de dict die de server als JSON stuurt. De eerste regels beantwoorden `/health`, geven 404 bij een pad dat de server niet heeft, en 405 bij een andere methode dan POST. `problem(request)` geeft `""` voor een goed verzoek, en anders een korte tekst die zegt wat er mis is: `trouble`. `answer(request)` is de versie van de course uit Oefening 1. Vervang de `...` door twee regels: als `trouble` niet leeg is, geef dan de statuscode voor een fout verzoek terug en `{"error": trouble}`, zodat het model het verzoek nooit ziet.

        De check roept `handle("POST", "/predict", C.HOUR_24)` aan: de body van middernacht, geschreven als uur 24. Dan een goed verzoek, een verzoek zonder temp, een met "warm" als temp, een GET naar /predict, een POST naar /forecast en een GET naar /health.
        """),
        L.task("Antwoord met een fout als `trouble` niet leeg is, nog voordat het model het verzoek ziet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_handle():
    box_handle = L.code_box("handle", lines=11)
    box_handle
    return (box_handle,)


@app.cell(hide_code=True)
def check_handle(box_handle):
    result_handle = L.check("handle", box_handle.value)
    result_handle
    return (result_handle,)


@app.cell(hide_code=True)
def hints_handle():
    L.hints(
        "handle",
        [
            "Stap 3 van het recept. `trouble` is `\"\"` voor een goed verzoek. Bij elke andere tekst geeft `handle` meteen iets terug, met een statuscode en `{\"error\": trouble}`, en komt de functie nooit bij de laatste regel.",
            "Het eerste cijfer van een statuscode zegt wie aan zet is: een 4 voor een verzoek dat de client moet aanpassen, een 5 voor een server die stukging. Welke code in de tabel van het idee zegt: fout verzoek?",
            "`        return ___, {\"error\": trouble}`",
        ],
        why="`if trouble != \"\":` en dan `return 400, {\"error\": trouble}`: een fout verzoek krijgt 400, met wat er mis is, en het model ziet het verzoek nooit. Uur 24 krijgt *hr must be a whole number from 0 to 23, not 24*, en de client weet wat er anders moet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_codes():
    mo.md(r"""
    ### Oefening 3. Welke statuscode?

    Acht verzoeken aan de server van deze les, met POST /predict en GET /health als endpoints. Zet elk verzoek onder de statuscode die het antwoord moet hebben.
    """)
    return


@app.cell(hide_code=True)
def sort_codes_cell():
    sort_codes = L.sort("codes")
    sort_codes
    return (sort_codes,)


@app.cell(hide_code=True)
def check_codes(sort_codes):
    result_codes = L.check("codes", sort_codes)
    result_codes
    return (result_codes,)


@app.cell(hide_code=True)
def hints_codes():
    L.hints(
        "codes",
        [
            "Stap 3 van het recept, en de tabel van het idee. Vraag bij elk verzoek: ging het goed, is de body fout, is het pad fout, of ging de server stuk op een goed verzoek?",
            "Twee verzoeken gaan goed, en twee hebben een body waar het model niets mee kan. Welke twee vragen om een pad dat de server niet heeft?",
            "200: spits en \\_\\_\\_. 400: warm en \\_\\_\\_. 404: verwachting en \\_\\_\\_. 500: een bug en \\_\\_\\_.",
        ],
        why="**200:** de spits en /health, want er is niets mis. **400:** \"warm\" en geen luchtvochtigheid, want de client moet het verzoek aanpassen. **404:** /forecast en /Predict, want de server heeft zo'n pad niet, letter voor letter. **500:** de bug en het volle geheugen, want het verzoek was goed, en de server ging erop stuk.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reach():
    mo.md(r"""
    ### Oefening 4. Wie kan erbij?

    Je server luistert op 127.0.0.1, poort 8000. Op je wifi heeft je computer ook het adres 192.168.1.20, en een collega op dezelfde wifi stuurt een verzoek naar 192.168.1.20, poort 8000.
    """)
    return


@app.cell(hide_code=True)
def choice_reach_cell():
    choice_reach = L.choice(
        "reach",
        question="Wat gebeurt er?",
        options={
            "answer": "De collega krijgt de fietsen: beide computers zitten op dezelfde wifi",
            "right": "Niets antwoordt: 127.0.0.1 neemt alleen verzoeken aan van je eigen computer",
            "404": "De collega krijgt status 404: het endpoint bestaat alleen op 127.0.0.1",
            "router": "De collega krijgt de fietsen, iets later dan jij, via de router van het netwerk",
        },
        correct="right",
        why={
            "right": "Een server op 127.0.0.1 is onbereikbaar voor elke andere computer, ook op dezelfde wifi: geen programma op 192.168.1.20, poort 8000, neemt het verzoek aan, dus meldt het programma van de collega dat de verbinding geweigerd is, en komt er helemaal geen HTTP-antwoord. Zo wil je het hebben zolang je bouwt. Een server voor anderen luistert op het adres van het netwerk, en heeft dan meer nodig: een sleutel per client, versleuteling met HTTPS, en een grens aan het aantal verzoeken.",
            "answer": "De server luistert alleen op 127.0.0.1. Welke computers kunnen een verzoek naar dat adres sturen?",
            "404": "Een 404 komt van een server die het verzoek kreeg. Krijgt een server dit verzoek?",
            "router": "De router brengt het verzoek naar 192.168.1.20. Luistert de server op dat adres?",
        },
    )
    choice_reach
    return (choice_reach,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je server, op je eigen computer

    De demo start een echte server op 127.0.0.1, met je eigen `handle` zodra Oefening 2 klopt, en tot dan met de handle van de course. Kies een methode, typ een pad, en verander de JSON van de body: de server krijgt je verzoek via HTTP, en de afbeelding laat zien wat de server kreeg, wat de server terugstuurde, en hoe lang de client wachtte. Alles blijft op je computer, en de server stopt weer na elk verzoek.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_method = mo.ui.dropdown(options=["POST", "GET"], value="POST", label="Methode")
    demo_path = mo.ui.text(value="/predict", label="Pad")
    demo_body = mo.ui.text_area(value=C.DEMO_BODY, label="Body (JSON)", rows=7, full_width=True)
    mo.vstack([
        mo.hstack([demo_method, demo_path], justify="start"),
        demo_body,
        L.task("Vraag om uur 17, laat temp weg, schrijf \"warm\" als temp, stuur een GET, en probeer uur 4 en een temp van `26.2`. Wat krijgt elk verzoek?"),
    ])
    return demo_body, demo_method, demo_path


@app.cell(hide_code=True)
def demo_send(box_handle, demo_body, demo_method, demo_path, result_handle):
    _fn = C.mine(box_handle.value, result_handle.state == "correct")
    mo.Html(C.exchange_picture(demo_method.value, demo_path.value, demo_body.value, _fn, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Uur 17 krijgt 497 fietsen: de rest van de body beschrijft nog 8:00, en met het weer van 17:00 zelf verwacht het model 537; er werden er 595 verhuurd. Zonder temp, of met "warm", antwoordt de server 400 met wat er mis is, en het model ziet het verzoek nooit; een GET naar /predict krijgt 405. Uur 4 krijgt −3 fietsen: de server geeft door wat het model zegt, ook een getal onder 0. En een temp van 26,2, in graden in plaats van de 0,64 van de data, krijgt status 200 en 4.855 fietsen in plaats van 431: het verzoek was goed gevormd, dus hield niets het verzoek tegen.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Batch, online, streaming of op het apparaat](/course/open/sys-deployment-patterns?lang=nl) maakte online inference een voorspelling op het moment van elk verzoek. Deze les bouwde dat moment: een server met endpoints, een controle voordat het model iets ziet, en een antwoord in JSON met een statuscode, zoals FastAPI en Triton dat op grote schaal doen. Een echte dienst laadt een pakket zoals dat van [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl) één keer, bij het starten. In de volgende les, *Voor en na het model: het serving-pad*, krijgt de input dezelfde voorbereiding als bij de training, en wordt de output een beslissing: geen antwoord onder 0 fietsen meer, en geen 4.855 voor een temperatuur in graden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_reach, result_answer, result_codes, result_handle):
    L.wrap_up(result_answer, result_handle, result_codes, choice_reach)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
