# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-monitoring. Route: course/route/lessen/systemen.yaml. Checks, de log van de dag, de regels en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-monitoring", "nl")
    C = L.checks   # de log van één dag van de service, de p99 van elke minuut, de regels en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) en [Je model achter een API](/course/open/sys-serving-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="Voor 200 ritten naar school geeft `np.quantile(times, 0.9)` het getal 41. Wat zegt dat?",
        options={
            "mean": "De ritten duurden gemiddeld 41 minuten",
            "right": "Negen op de tien ritten duurden 41 minuten of korter",
            "longest": "De langste rit duurde 41 minuten, en alle andere ritten korter",
            "ninety": "90 van de ritten duurden 41 minuten",
        },
        correct="right",
        why={
            "right": "Het 0,9-kwantiel is de waarde met 9 van de 10 waarden eronder: 180 van de 200 ritten duurden 41 minuten of korter, en 20 duurden langer. In deze les beschrijft hetzelfde soort getal de traagste verzoeken van een service.",
            "mean": "Een gemiddelde zegt waar het midden ligt, omhooggetrokken door een paar lange ritten. Wat zegt een kwantiel? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "longest": "De langste rit is wat `np.max` geeft. Welk deel van de ritten ligt onder het 0,9-kwantiel? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "ninety": "0,9 is een deel van de ritten, geen aantal. Welk deel van de 200 ritten ligt onder 41? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def quick_check_500():
    quick_500 = L.choice(
        "quick-500",
        question="Een server antwoordt op een verzoek om een voorspelling met status 500. Wat zegt dat?",
        options={
            "bad-request": "Het verzoek was fout, bijvoorbeeld door een ontbrekend veld",
            "right": "De server ging stuk op het verzoek, dat best goed kan zijn geweest",
            "bikes": "Het model verwacht 500 fietsen voor dat uur",
            "slow": "Het antwoord duurde 500 milliseconden, langer dan mocht, en de client gaf het op",
        },
        correct="right",
        why={
            "right": "500 is *Internal Server Error*: de server ging stuk tijdens het afhandelen van het verzoek, bijvoorbeeld omdat de code stopte met een fout of het geheugen vol raakte. In deze les is het deel van de antwoorden met 500 een van de getallen die een service in de gaten houdt.",
            "bad-request": "Een fout verzoek, zoals een verzoek met een ontbrekend veld, krijgt een code vanaf 400. Welke kant krijgt de schuld bij een code vanaf 500? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
            "bikes": "De statuscode zegt hoe het verzoek ging, niet wat het model zei. Wat zegt 500? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
            "slow": "Een statuscode is geen tijd. Wat zegt 500 over de server? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
        },
    )
    quick_500
    return (quick_500,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model in productie draait dag en nacht, en niemand leest elk verzoek. In [Je model achter een API](/course/open/sys-serving-api?lang=nl) beantwoordde de service van het project je eigen verzoeken op je eigen computer. In productie beantwoordt dezelfde service tienduizenden verzoeken per dag, en sommige gaan mis op een manier die niemand ziet: een server die twee seconden stilstaat, een back-up waardoor elk antwoord traag wordt, een nieuwe versie van een app die een temperatuur in graden stuurt. Zonder getallen hoor je het van de gebruikers van de app, als je het al hoort.

    In het boek *Site Reliability Engineering* noemen engineers van Google vier golden signals om bij elke service in de gaten te houden: latency, traffic, errors en saturation, oftewel hoe vol de server zit. Zulke getallen heten metrics: getallen die steeds opnieuw gemeten worden terwijl de service draait. Een model heeft meer nodig. Een model kan snel antwoorden, met status 200, en toch ernaast zitten, dus verdienen ook de eigen getallen van het model aandacht: wat het model antwoordt, en wat erin gaat.

    **Een voorbeeld uit het dagelijks leven.** Een busmaatschappij zegt dat de bussen gemiddeld 2 minuten te laat rijden. Dat klinkt prima. Maar als 1 op de 100 bussen 40 minuten te laat is, zit iemand met twee busritten per dag ongeveer elke zeven weken in zo'n bus, en zo'n rit vergeet je niet. Het gemiddelde zegt niets over zo'n bus; alleen een getal over de traagste bussen doet dat.
    """)
    return


@app.cell(hide_code=True)
def predict_tail():
    guess_tail = L.guess(
        "tail",
        question="Op 14 juli 2011 beantwoordde de service van het project 59.912 verzoeken, in gemiddeld 28,0 ms. Gesorteerd op latency: bij hoeveel ms begint de traagste 1 op de 100 verzoeken, denk je?",
        options={
            "close": "Ongeveer 30 ms, net boven het gemiddelde",
            "half": "Ongeveer 45 ms, anderhalf keer het gemiddelde",
            "right": "Ongeveer 160 ms, bijna zes keer het gemiddelde",
            "slowest": "Ongeveer 1.600 ms, zo lang als het traagste verzoek",
        },
        reveal="**Ongeveer 160 ms, bijna zes keer het gemiddelde.** Gesorteerd op latency begint de traagste 1 op de 100 verzoeken bij 159 ms: 599 verzoeken duurden langer. De helft van alle verzoeken duurde korter dan 22,6 ms, en bijna 19 op de 20 korter dan 40 ms. Het gemiddelde van 28,0 ms ligt dicht bij de snelle meerderheid en verbergt de trage verzoeken: een server die om 09:12 twee seconden stilstond, een back-up waardoor elk antwoord van 14:00 tot 14:25 traag was, en af en toe één los traag verzoek. In deze les vind je de trage verzoeken in de log, en bepaal je wanneer iemand moet kijken.",
    )
    guess_tail
    return (guess_tail,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Model monitoring.** Een draaiend model houd je in de gaten met twee soorten getallen. Getallen over de *service* zeggen of er antwoord komt, en hoe snel: de vier golden signals. **Latency** is de tijd tussen een verzoek en het antwoord, hier in milliseconden (ms). *Traffic* of verkeer is het aantal verzoeken per minuut. *Errors* of fouten zijn de antwoorden met een statuscode vanaf 500, waarbij de server stukging. *Saturation* zegt hoe vol de server zit: het geheugen en de processor. Getallen over het *model* zeggen of de antwoorden deugen: wat het model antwoordt, uur na uur; wat erin gaat, naast wat er bij de training in ging; en, zodra bekend is hoeveel fietsen er in dat uur echt verhuurd werden, hoe ver het model ernaast zat. Beide in de gaten houden heet **model monitoring**.

        **Structured logging.** Elk getal begint bij een log. Een regel zoals *Verzoek voor 9:00 duurde 30 ms, alles goed* leest makkelijk voor een mens; een programma moet raden waar elk getal staat. Met **structured logging** schrijft een service per verzoek één regel als JSON-object, met vaste velden:

        ```
        {"time": "09:00:02.418", "path": "/predict", "status": 200, "ms": 29.7, "temp": 0.68, "bikes": 223}
        ```

        `json.loads` maakt van zo'n regel weer een dict, zodat een programma de verzoeken van een minuut kan tellen, de latency van elk verzoek kan pakken, of de antwoorden per uur kan groeperen. De regel bevat wat erin ging en wat eruit kwam, dus houdt de log ook het model in de gaten, en niet alleen de server.

        **De dag van deze les.** De log van deze les beslaat één dag van de service van het project, donderdag 14 juli 2011, gemaakt door de course: elke keer dat iemand de app van het fietsdeelbedrijf opent, vraagt de app aan de service hoe druk het uur wordt. De uren en het weer komen uit de fietsdata, en de antwoorden komen van het model van de course; de verzoeken en de latency van elk verzoek zijn verzonnen, met drie dingen die misgingen.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Tail latency.** De afbeelding toont de 59.912 verzoeken van de dag. De meeste duurden 15 tot 40 ms, en het gemiddelde is 28,0 ms. Maar een dunne staart loopt ver door naar rechts: verzoeken die tien of vijftig keer zo lang duurden als de mediaan. Het gemiddelde verbergt de trage verzoeken, want een paar trage verzoeken tussen duizenden snelle verschuiven het gemiddelde nauwelijks. Een service beschrijft de latency dus met kwantielen, zoals in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl). De *p50* is de mediaan: de helft van de verzoeken was sneller. De *p99* is het 99e percentiel: de latency met 99 van de 100 verzoeken eronder. De **tail latency** is de latency van de traagste verzoeken, boven de p99. Dat doet er meer toe dan het lijkt. Heeft een pagina 100 verzoeken aan een service nodig, dan is de pagina pas klaar na het traagste verzoek. Is 1 op de 100 verzoeken traag, dan wachten 63 van de 100 zulke pagina's op minstens één traag verzoek.

        **Alertregel.** Niemand kijkt de hele dag naar een grafiek. Een **alertregel** zegt wanneer iemand moet kijken: een getal, een grens, en hoe lang het getal erboven moet blijven, zoals *de p99 boven 300 ms gedurende 5 minuten achter elkaar*. De tijd doet er net zoveel toe als de grens. Gaat een regel af op elke piek, zoals de pauze van twee seconden om 09:12, dan roept de regel iemand voor niets op; na een paar keer kijkt niemand meer, en dat heet *alert fatigue*. Wacht een regel te lang, of ligt de grens te hoog, dan loopt een echt probleem een uur door voordat iemand het weet. In Oefening 2 schrijf je de voorwaarde, en in de demo stel je de regel af op deze dag.

        **Op grote schaal.** Prometheus haalt bij veel servers getallen op, zoals hoeveel verzoeken er in elk stuk van de schaal van de latency vielen, en schat daaruit met `histogram_quantile` de p99 per minuut. Een alertregel in Prometheus heeft een `for`-clausule, zoals `for: 5m`: de minuten van Oefening 2. Grafana tekent de grafieken, en OpenTelemetry is een veelgebruikte manier waarop een service de logs en getallen naar zulke tools stuurt. Tools voor model monitoring voegen de getallen van het model toe: wat erin gaat, wat eruit komt, en hoe ver het model ernaast zat.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **model monitoring** | | een model in gebruik in de gaten houden met getallen over de service en over het model, en een alert als er iets misgaat | |
        | **structured logging** | | per verzoek één regel schrijven met vaste velden, zoals een JSON-object, zodat een programma de log kan lezen | `json.dumps(...)`, `json.loads(line)` |
        | **tail latency** | | de latency van de traagste verzoeken, zoals de verzoeken boven het 99e percentiel | |
        | p50, p99 | p vijftig, p negenennegentig | de latency met de helft van de verzoeken eronder, of 99 van de 100 | |
        | **alertregel** | | een getal, een grens, en hoe lang het getal erboven moet blijven voordat iemand wordt opgeroepen | |
        """),
        L.watch_out("Getallen over de service laten niet zien dat een model ernaast zit. Een verzoek met een temperatuur in graden krijgt status 200, even snel als altijd, en een antwoord van duizenden fietsen. Houd ook in de gaten wat erin gaat en wat eruit komt."),
        L.watch_out("Een p99 over een paar verzoeken ligt dicht bij het traagste verzoek: bij 40 verzoeken in een minuut bepaalt één traag verzoek de p99. Nog een reden waarom een alert moet wachten op een getal dat hoog blijft."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een draaiend model in de gaten houden:

    1. Log elk verzoek als één gestructureerde regel: de tijd, het pad, de status, de latency, de inputs en het antwoord.
    2. Bereken per minuut de getallen over de service: de verzoeken, het deel met een fout, en de p50 en p99 van de latency; en houd het geheugen en de processor van de server in de gaten.
    3. Bereken per uur de getallen over het model: het gemiddelde antwoord, de inputs naast de inputs van de training, en, zodra bekend is wat er echt gebeurde, hoe ver het model ernaast zat.
    4. Teken elk getal in een grafiek over de tijd, naast de gebruikelijke waarden.
    5. Schrijf een alertregel voor elk getal waarbij iemand moet ingrijpen: een grens, en hoe lang het getal erboven moet blijven.
    6. Probeer elke regel uit op een eerdere dag: vangt de regel de echte problemen, en hoe vaak gaat de regel af voor niets?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.at(time)` geeft de logregels van één minuut als een list van teksten: `C.at("09:00")` bevat de verzoeken van 09:00:00 tot 09:00:59. De cel print de eerste regel van 09:00, en daarna voor 09:00 en 09:12 hoeveel verzoeken er binnenkwamen, het gemiddelde en de mediaan van de latency, en het traagste verzoek. Voer de cel uit.
    """)
    return


@app.cell
def example_log():
    import json
    import numpy as np

    print(C.at("09:00")[0])
    for _time in ("09:00", "09:12"):
        _ms = [json.loads(_line)["ms"] for _line in C.at(_time)]
        print(_time, len(_ms), "verzoeken, gemiddeld", round(float(np.mean(_ms)), 1),
              "mediaan", round(float(np.median(_ms)), 1), "traagste", max(_ms))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de rustige minuut van 09:00 kwamen 49 verzoeken binnen, en het gemiddelde van 24,6 ms ligt dicht bij de mediaan van 22,2 ms. Om 09:12 stond de server twee seconden stil, en één verzoek wachtte 1.579,6 ms. Dat ene verzoek trekt het gemiddelde omhoog tot 70,4 ms, zoals één lange rit een gemiddelde omhoogtrekt, terwijl de mediaan nauwelijks beweegt: 23,5 ms. Geen van beide zegt hoe traag de trage verzoeken waren. In Oefening 1 schrijf je het getal dat dat wel zegt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tail():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De traagste 1 op de 100

        `tail(lines)` krijgt `lines`, de logregels van één minuut, als een list van teksten. Elke `line` is één JSON-object: `json.loads` maakt er een dict van, en `ms` verzamelt de latency van elk verzoek, uit het veld `"ms"`. `p50` is de mediaan, uit `np.median`, en `float` maakt er een gewoon Python-getal van; `round` rondt beide getallen af op hele milliseconden in de dict die de functie teruggeeft. Vervang de `...` zodat `p99` de latency is met 99 van de 100 verzoeken van de minuut eronder.

        De check roept `tail(C.at("09:00"))` aan, en daarna de minuten van 09:12, 14:05, 03:00 en 17:30.
        """),
        L.task("Maak van `p99` de latency met 99 van de 100 verzoeken van de minuut eronder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tail():
    box_tail = L.code_box("tail", lines=5)
    box_tail
    return (box_tail,)


@app.cell(hide_code=True)
def check_tail(box_tail):
    result_tail = L.check("tail", box_tail.value)
    result_tail
    return (result_tail,)


@app.cell(hide_code=True)
def hints_tail():
    L.hints(
        "tail",
        [
            "Stap 2 van het recept: de p99 is een kwantiel van `ms`, zoals in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl). `np.quantile` krijgt de waarden en het deel daarvan dat eronder moet liggen, en `float` maakt van de uitkomst een gewoon Python-getal. De uitkomst gaat in `p99`.",
            "Een kwantiel krijgt het deel als een getal van 0 tot 1, niet als percentage: de mediaan is het 0,5-kwantiel. Welk getal is het deel 99 van de 100?",
            "`    p99 = float(np.quantile(ms, ___))`",
        ],
        why="`p99 = float(np.quantile(ms, 0.99))`: de latency met 99 van de 100 verzoeken van de minuut eronder. Om 09:00 is dat 55 ms, tweeënhalf keer de mediaan van 22 ms; om 09:12 trekt één verzoek van 1.579,6 ms de p99 omhoog tot 1.064 ms.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_alert():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Niet op één piek

        `alert(p99s, limit, minutes)` krijgt `p99s`, de p99 van elke minuut op volgorde, een grens `limit` in milliseconden, en een aantal minuten `minutes`. De loop gaat over elke minuut `i` vanaf `minutes - 1`, met `range` tot `len(p99s)`, en `last` bevat de p99 van de laatste `minutes` minuten, tot en met minuut `i`. De functie geeft de eerste minuut terug waarop de alert afgaat, of `None` als de alert nooit afgaat. Vervang de `...` zodat de alert alleen afgaat als elke minuut in `last` boven `limit` ligt.

        De check roept `alert([120, 950, 130, 340, 360, 380, 410, 390], 300, 5)` aan: één piek van 950 in minuut 1, en vanaf minuut 3 een getal dat boven 300 blijft. Daarna `C.P99`, de p99 van elke minuut van de dag, met drie regels, en een paar korte lists.
        """),
        L.task("Laat de alert alleen afgaan als elke minuut in `last` boven `limit` ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_alert():
    box_alert = L.code_box("alert", lines=6)
    box_alert
    return (box_alert,)


@app.cell(hide_code=True)
def check_alert(box_alert):
    result_alert = L.check("alert", box_alert.value)
    result_alert
    return (result_alert,)


@app.cell(hide_code=True)
def hints_alert():
    L.hints(
        "alert",
        [
            "Stap 5 van het recept: de alert gaat alleen af als het getal in elk van de laatste minuten boven de grens bleef, dus houdt één minuut van `last` op of onder `limit` de alert stil. Welk ene getal van `last` zegt of alle getallen boven `limit` liggen?",
            "Ligt het kleinste getal van `last` boven `limit`, dan geldt dat voor elk getal van `last`. Welke Python-functie geeft het kleinste getal van een list?",
            "`        if ___ > limit:`",
        ],
        why="`if min(last) > limit:`: de kleinste p99 van de laatste minuten ligt alleen boven de grens als elke p99 erboven ligt. De piek van 950 tussen minuten van 120 en 130 laat de alert niet afgaan; bij minuut 7 liggen de laatste 5 minuten allemaal boven 300, en gaat de alert af.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.md(r"""
    ### Oefening 3. Snel en toch fout

    Om 16:00 gaat een nieuwe versie van de app van het fietsdeelbedrijf live, voor één op de vijf gebruikers. Niemand merkt iets: de app krijgt nog steeds snel antwoord. Toch is er iets mis met wat de nieuwe versie stuurt.
    """)
    return


@app.cell(hide_code=True)
def choice_model_cell():
    choice_model = L.choice(
        "model",
        question="Welk getal in de log laat zien dat er iets mis is?",
        options={
            "tail": "De p99 van de latency van elke minuut",
            "errors": "Het deel van de antwoorden met status 400 of 500 in elke minuut",
            "median": "De mediaan van `bikes` per uur, naast andere dagen",
            "right": "Het gemiddelde van `bikes` per uur, naast andere dagen",
        },
        correct="right",
        why={
            "right": "De nieuwe versie stuurt temp in graden: 31,2 om 16:00, waar de data 0,76 heeft. Voor de service is zo'n verzoek goed gevormd: de service antwoordt met status 200, even snel als altijd, met meer dan 5.000 fietsen. Vanaf 16:00 springt het gemiddelde van `bikes` per uur naar 1.311, en om 17:00 naar 1.537, waar de oude versie 289 en 537 krijgt. De mediaan blijft 289 en 537, want vier op de vijf verzoeken komen nog van de oude versie: hier is het gemiddelde het getal om in de gaten te houden, want één op de vijf antwoorden is tien keer te hoog of meer, en dat trekt het gemiddelde ver omhoog. De p99 en de fouten houden de service in de gaten, en de service doet het prima.",
            "tail": "De nieuwe versie is even snel als de oude. Zou een trage minuut het laten zien?",
            "errors": "Een verzoek met een temperatuur in graden krijgt status 200: het getal is goed gevormd. Zou zo'n verzoek als fout tellen?",
            "median": "De mediaan blijft 289 om 16:00 en 537 om 17:00, de antwoorden van de oude versie: vier op de vijf verzoeken komen daar nog vandaan, en de mediaan is het middelste antwoord. Welk getal beweegt als één op de vijf antwoorden veel te hoog is?",
        },
    )
    choice_model
    return (choice_model,)


@app.cell(hide_code=True)
def text_exercise_watch():
    mo.md(r"""
    ### Oefening 4. De service of het model?

    Een team kan acht getallen in de gaten houden voor de service van het project. Zet elk getal bij de vraag waarop het een antwoord geeft: komt er antwoord van de service, en hoe snel, of deugen de antwoorden van het model?
    """)
    return


@app.cell(hide_code=True)
def sort_watch_cell():
    sort_watch = L.sort("watch")
    sort_watch
    return (sort_watch,)


@app.cell(hide_code=True)
def check_watch(sort_watch):
    result_watch = L.check("watch", sort_watch)
    result_watch
    return (result_watch,)


@app.cell(hide_code=True)
def hints_watch():
    L.hints(
        "watch",
        [
            "Vraag bij elk getal: verandert het getal als het model iets anders antwoordt, of als er andere getallen in gaan, terwijl de server even snel en gezond blijft?",
            "Vier getallen gaan over de server: hoeveel verzoeken er binnenkomen, hoeveel er stukgaan, hoe traag de verzoeken zijn, en hoe vol de server zit. Welke vier gaan over wat het model krijgt en antwoordt?",
            "Service: verkeer, fouten, \\_\\_\\_ en \\_\\_\\_. Model: antwoorden, nullen, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**De service:** verkeer, fouten, tail latency en geheugen, want deze vier zeggen of er antwoord komt, en hoe snel. **Het model:** antwoorden, nullen, inputs en hoe ver ernaast, want deze vier zeggen of de antwoorden deugen. De nieuwe versie van de app om 16:00 laat zich alleen in de tweede groep zien.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De alert afstellen

    Kies een grens voor de p99 en een aantal minuten achter elkaar. De afbeelding toont de p99 van elke minuut van 14 juli, de grens, en bovenaan een stip voor elke alert: een stip tijdens de vertraging van 14:00 tot 14:25, of een stip voor niets.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_limit = mo.ui.slider(start=100, stop=600, step=50, value=300, label="Grens in ms", show_value=True)
    demo_minutes = mo.ui.slider(start=1, stop=10, step=1, value=1, label="Minuten achter elkaar", show_value=True)
    mo.vstack([
        demo_limit,
        demo_minutes,
        L.task("Houd de grens op 300 ms en verhoog het aantal minuten vanaf 1: vanaf hoeveel minuten is de piek van 09:12 geen alert meer? Probeer daarna 150 ms met 1 minuut, en 600 ms met 10 minuten. Wat gaat er telkens mis?"),
    ])
    return demo_limit, demo_minutes


@app.cell(hide_code=True)
def demo_picture(demo_limit, demo_minutes):
    mo.Html(C.demo_picture(demo_limit.value, demo_minutes.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij 300 ms en 1 minuut gaat de regel twee keer af: om 09:12, voor een pauze die na twee seconden al voorbij was, en om 14:00. Vanaf 2 minuten laat alleen de vertraging de regel afgaan: om 14:01 bij 2 minuten, en een minuut later voor elke minuut meer. Bij 150 ms en 1 minuut gaat de regel 58 keer af, 57 keer voor niets, verspreid over de hele dag: in een minuut met enkele tientallen verzoeken is één traag verzoek genoeg. Zo begint alert fatigue. Bij 600 ms en 10 minuten gaat de regel nooit af, hoewel de verzoeken 25 minuten lang traag waren. De regel van Oefening 2 (300 ms gedurende 5 minuten) gaat één keer af, om 14:04.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je model achter een API](/course/open/sys-serving-api?lang=nl) beantwoordde de service elk verzoek met een statuscode; in deze les schrijft de service voor elk verzoek een gestructureerde regel, en maak je van de log getallen. De mediaan en de kwantielen van [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) maakten van de latency van elke minuut een p50 en een p99, en een alertregel bepaalde wanneer iemand moet kijken. *Voor en na het model: het serving-pad* raadde een log aan van de inputs van een service; de nieuwe versie van de app om 16:00 laat zien waarom. In de volgende les, *Hoe ML-systemen falen*, krijgen zulke stille fouten een naam, naast de crashes die de getallen over de service wel laten zien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_model, result_alert, result_tail, result_watch):
    L.wrap_up(result_tail, result_alert, choice_model, result_watch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
