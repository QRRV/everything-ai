# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-project-service. Route: course/route/lessen/systemen.yaml. Checks, de service van de course, de log van 2012, de demo die je service laat draaien, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-service", "nl")
    C = L.checks   # de service van de course, de tabellen van de checks, de server van de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voor en na het model: het serving-pad](/course/open/sys-serving-path?lang=nl) en [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_skew():
    quick_skew = L.choice(
        "quick-skew",
        question="Een service schaalt de inputs van elk verzoek met het gemiddelde en de standaardafwijking van de dag zelf, terwijl het model leerde met de getallen van de trainingsuren. Hoe heet dat?",
        options={
            "drift": "Data drift: de wereld stuurt de service andere inputs dan de inputs waarvan het model leerde",
            "right": "Training-serving skew: de training maakt de inputs op de ene manier, het serven op een andere",
            "rule": "Een bedrijfsregel: een keuze van het bedrijf na het model, zoals wanneer de ploeg op pad gaat",
            "overfit": "Overfitting: het model leerde de trainingsuren uit het hoofd, en gaat de mist in op nieuwe uren",
        },
        correct="right",
        why={
            "right": "Het model leerde van inputs die geschaald waren met het gemiddelde en de standaardafwijking van de training. Met andere getallen geeft een service het model andere inputs voor hetzelfde uur, dus raken de antwoorden ernaast: met de schaal van de batch van elke dag gemiddeld 73 fietsen per uur ernaast in plaats van 34.",
            "drift": "Drift is een verandering in de wereld: de uren zouden binnenkomen met ander weer of andere fietsers. Hier blijft de wereld gelijk, en maakt de service de inputs op een andere manier: hoe heet dat? Zie [Voor en na het model: het serving-pad](/course/open/sys-serving-path?lang=nl).",
            "rule": "Een bedrijfsregel komt na het model en maakt van een getal een beslissing, zoals de ploeg laten komen vanaf 400 fietsen. Dit gebeurt in de stap naar het model toe: hoe noemt de course inputs die in de training op de ene manier gemaakt worden en bij het serven op een andere? Zie [Voor en na het model: het serving-pad](/course/open/sys-serving-path?lang=nl).",
            "overfit": "Overfitting gaat over wat het model leerde. Hier blijft het model hetzelfde, en worden de inputs op een andere manier gemaakt dan in de training: hoe heet dat? Zie [Voor en na het model: het serving-pad](/course/open/sys-serving-path?lang=nl).",
        },
    )
    quick_skew
    return (quick_skew,)


@app.cell(hide_code=True)
def quick_check_psi():
    quick_psi = L.choice(
        "quick-psi",
        question="De PSI van de temperatuur in maart 2012, tegen maart in de trainingsuren van 2011, is 1,15. Wat zegt dat?",
        options={
            "error": "Het model zit er in maart 2012 115% verder naast dan op de validatiedagen van 2011",
            "share": "1,15% van de uren van maart 2012 had een temperatuur die in de trainingsuren nooit voorkwam",
            "right": "De temperaturen van maart 2012 vallen in andere bins dan in de training: ver boven 0,25",
            "year": "Nog niets: een PSI zegt pas iets als je de PSI over een heel jaar data meet",
        },
        correct="right",
        why={
            "right": "Boven 0,25 is een verdeling flink verschoven. Maart 2012 was veel warmer dan maart 2011: de helft van de uren lag in de bovenste bin van maart in de training, waar ongeveer een tiende hoort.",
            "error": "De PSI vergelijkt de aandelen van de waarden in de bins van de training: er zit geen antwoord van het model in en geen verhuurde fiets. Wat zegt de PSI over de temperaturen zelf? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
            "share": "De PSI is geen aandeel van de uren: de PSI telt over tien bins op hoe ver het aandeel nu van het aandeel in de training ligt. Wat zegt 1,15, ver boven 0,25, over de temperaturen? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
            "year": "De PSI werkt op elke twee reeksen waarden, zoals één maand tegen dezelfde maand van de training: de les deed dat met maart. Wat zegt 1,15 over de temperaturen? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
        },
    )
    quick_psi
    return (quick_psi,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je model van de vraag is ingepakt en getest. Nu moet het model aan het werk: een service beantwoordt elk verzoek, op elk uur, maandenlang. Dan kunnen er twee dingen misgaan zonder dat iemand het merkt. De service kan traag worden, en alleen een verslag van elk verzoek laat zien wanneer, en hoe traag. En de wereld kan wegschuiven van de uren waarvan het model leerde: in 2012 fietsten er veel meer mensen dan in 2011. Een team wil weten wanneer het model opnieuw moet trainen, en wil dat weten uit getallen, niet uit een klacht.

        Dus schrijft je service in deze stap van je project één regel per verzoek, met hoe lang het antwoord duurde, en lezen twee checks deze regels: een driftcheck van wat er binnenkwam, maand na maand, en, zodra de echte verhuurde fietsen bekend zijn, hoe ver het model ernaast zat. Een regel op deze getallen zegt wanneer het model opnieuw moet trainen.

        **Een voorbeeld uit het dagelijks leven.** Een bakkerij bakt elke ochtend wat de voorspelling zegt, en schrijft op wat er gebakken en verkocht is. De bakker ziet 's ochtends of het weer veranderde. Pas 's avonds blijkt of de voorspelling goed was, als de verkoop van de dag binnen is.
        """),
        draw.project_steps(["Een model van de vraag, bijgehouden en ingepakt", "Tests en een model card", "Je model als service, bewaakt op drift"], 2,
                           here_label="← vandaag", title="Waar je project staat: ops"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_months():
    guess_months = L.guess(
        "months",
        question="Je driftcheck vergelijkt elke maand van 2012 met dezelfde maand van 2011. De temperaturen van januari 2012 liggen ver van januari 2011, een PSI van 0,87; de temperaturen van juli 2012 liggen dicht bij juli 2011, een PSI van 0,06. In welke van de twee maanden zit je model er verder naast, denk je?",
        options={
            "january": "Januari, veruit, want het weer van januari ligt het verst van wat het model leerde",
            "same": "In beide ongeveer even ver, want het model krijgt het weer als een van de inputs",
            "right": "Juli, duidelijk, hoewel het weer dicht bij het weer van juli 2011 ligt",
            "neither": "Geen van beide: beide blijven dicht bij de 53,04 van de validatiedagen",
        },
        reveal="**Juli.** Een RMSE van 119,4 fietsen per uur in juli 2012, tegen 77,6 in januari: 2,25 keer de 53,04 van de validatiedagen, tegen 1,46. Het weer van juli 2012 leek op het weer van juli 2011, maar er werden 274 fietsen per uur verhuurd in plaats van 190, en het model, dat het uur en het weer ziet, bleef antwoorden zoals in 2011. Drift in de inputs zegt dat het weer verschoof, niet hoe ver het model ernaast zit. Oefening 3 meet dat, met de echte verhuurde fietsen.",
    )
    guess_months
    return (guess_months,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén regel per verzoek.** `time.perf_counter()` geeft de tijd op een klok die in seconden telt, vanaf een eigen moment: één stand alleen zegt niets, maar het verschil tussen twee standen is de tijd die verstreek, in seconden. Lees de klok vlak voor het antwoord en vlak erna, en je hebt de latency van het antwoord; een log geeft de latency in milliseconden, duizendsten van een seconde. De service schrijft de latency in één regel JSON per verzoek, met het antwoord en het verzoek zelf, zoals de structured logging van [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl) doet: `{"ms": 0.4, "bikes": 431, "request": {...}}`. Een bestand met één JSON-object op elke regel heet JSON Lines, vaak met de extensie `.jsonl`, zoals `runs.jsonl` van je project al is. Elke nieuwe regel komt achteraan in het bestand, dus gaat er niets verloren van wat er al staat, en een programma kan het bestand regel voor regel lezen, ook terwijl de service verder schrijft.

        **Drift, maand na maand.** De log bevat elk verzoek, dus ook de inputs die het model echt kreeg. Je driftcheck leest de inputs per maand en vergelijkt elke maand met dezelfde maand van de trainingsuren, met de PSI van [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl): een maand tegen een heel jaar meet vooral het seizoen. De course heeft deze PSI in je project gezet, als `ops/psi.py`: `psi_of(then, now)` maakt tien bins uit de waarden `then` en geeft de PSI van de waarden `now`. De check werkt op drie inputs, de temperatuur, de vochtigheid en de wind, en op de antwoorden van het model, de kolom `bikes`, zodra een maand in de log staat.

        **De echte verhuurde fietsen komen later.** Hoe ver een antwoord ernaast zat blijkt pas als het uur voorbij is en de fietsen geteld zijn, en een team krijgt de tellingen vaak in één keer, een dag of een maand later. Je `rmse_per` uit [Project: tests en een model card](/course/open/sys-project-tested-model?lang=nl) meet de RMSE van een maand zodra de verhuurde fietsen van de maand binnen zijn. Tegen welk getal? Het pakket heeft een eigen getal: `val_rmse` in `config.json`, de RMSE op de validatiedagen. Op nieuwe uren zit een model er altijd iets verder naast, dus laat een regel wat ruimte: train opnieuw als een maand er meer dan `LIMIT` keer zo ver naast zit, hier 1,5 keer.

        **Een waarschuwing, en een beslissing.** De driftcheck is er meteen, maar zegt dat de data verschoven zijn, niet dat het model fout zit. De fout telt, maar komt laat binnen. Een service houdt beide in de gaten: drift als vroege waarschuwing, en de fout van elke maand als de beslissing om opnieuw te trainen.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De bovenste rij is één verzoek: `serve` meet de tijd rond het model en geeft het antwoord terug. Elk verzoek laat één regel achter in de log. Vanuit de log vergelijkt de driftcheck meteen elke maand met dezelfde maand van de training; de fout van een maand wacht op de verhuurde fietsen.

        ### Nieuwe woorden en tekens

        | Woord of symbool | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **latency** | | hoe lang een service doet over één antwoord | `ms` |
        | **milliseconde** (ms) | | een duizendste van een seconde | |
        | `time.perf_counter()` | perf counter | een klok in seconden om te meten hoe lang iets duurt: alleen het verschil tussen twee standen betekent iets | `start = time.perf_counter()` |
        | **JSON Lines** | jeesson lains | een bestand met één JSON-object op elke regel, vaak met de extensie `.jsonl` | `requests.jsonl` |
        | `LIMIT` | | in je project: hoeveel keer zo ver als op de validatiedagen een maand ernaast mag zitten voordat het model opnieuw traint | `LIMIT = 1.5` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model als service, bewaakt op drift:

    1. Lees de klok vlak voor en vlak na elk antwoord, en schrijf één regel JSON per verzoek, met de latency, het antwoord en het verzoek.
    2. Lees de log per maand, en vergelijk elke maand met dezelfde maand van de training data: de PSI van elke input en van de antwoorden.
    3. Meet de RMSE van een maand zodra de echte uitkomsten van de maand binnen zijn, en zet de RMSE tegenover de RMSE van het pakket op de validatiedagen.
    4. Train opnieuw als een maand er meer dan `LIMIT` keer zo ver naast zit; lees drift van de inputs als waarschuwing, niet als de beslissing.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Er komt een verzoek binnen. De klok staat vlak voor het antwoord op 5.031,2140 seconden en vlak erna op 5.031,2146 seconden: er verstreek 0,0006 seconde, en dat is 0,6 milliseconde. De regel in de log zegt `"ms": 0.6`.

    Een pakket met een validatie-RMSE van 40,0 fietsen per uur, en een `LIMIT` van 1,5: de grens is 1,5 × 40,0 = 60,0. De eerste maand zit er 52,3 fietsen per uur naast, de tweede 72,4. De eerste blijft onder 60,0: nog geen training, hoewel de maand er 1,31 keer zo ver naast zit als op de validatiedagen. De tweede komt erboven, op 1,81 keer: de regel noemt de tweede maand.
    """)
    return


@app.cell(hide_code=True)
def text_request():
    mo.md(r"""
    ### Een verzoek, en de klok

    `C.REQUEST` is de body van een verzoek voor 8.00 uur op dinsdag 10 april 2012: het uur en de 10 features, zoals in [Je model achter een API](/course/open/sys-serving-api?lang=nl). In dat uur werden 692 fietsen verhuurd. `C.answer` is de service van de course: het antwoord van het verpakte model van de course, in hele fietsen, nooit minder dan 0. Rond het antwoord leest de cel twee keer `time.perf_counter()`, als `first` en `second`. Voer de cel uit, en kijk naar het antwoord, naast de 692 fietsen van dat uur, en naar de twee standen.
    """)
    return


@app.cell
def example_request():
    import time

    request = C.REQUEST                  # dinsdag 10 april 2012 om 8.00 uur, als de body van een verzoek
    print(request)
    first = time.perf_counter()          # een stand van de klok, in seconden
    body = C.answer(request)             # wat het model van de course antwoordt
    second = time.perf_counter()         # een tweede stand, vlak na het antwoord
    print(body)
    print(first)
    print(second)
    return


@app.cell(hide_code=True)
def text_exercise_serve():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén regel per verzoek

        `serve(answer, request, log)` beantwoordt één verzoek en schrijft één regel naar de log. `answer` is het model van de service: `answer` krijgt een verzoek en geeft een dict terug zoals `{"bikes": 431}`. `request` is de body van het verzoek, een dict met `hr` en de 10 features. `log` is het open logbestand: `log.write` zet tekst achteraan. `start` leest de klok met `time.perf_counter()` vlak voor het antwoord, `body` bevat het antwoord, en `ms` hoe lang het antwoord duurde. `line` is de regel voor de log: een dict met `"ms"`, afgerond op drie decimalen met `round`, de fietsen van het antwoord, en het verzoek. `json.dumps` schrijft de dict als één regel tekst, en `"\n"` sluit de regel af. De functie eindigt met `return body`, het antwoord dat de server terugstuurt. `import json` en `import time` staan er voor de functie.

        Vervang de `...` door hoe lang het antwoord duurde, in milliseconden.

        De check roept `serve(C.answer, C.REQUEST, log)` aan, met een eigen klok die alleen loopt terwijl het model antwoordt: `C.answer` doet er 0,4 ms over. In andere gevallen doet `C.slow_answer` er 1,5 seconde over, en is `C.EVENING` dezelfde dinsdag om 17.00 uur. Onder de check beantwoordt je `serve` de 24 uren van 14 juli 2011, met de echte klok van je computer. Klopt het, dan bewaart de check je functie als `work/projects/ops/service.py`.
        """),
        L.task("Vervang de `...` door hoe lang het antwoord duurde, in milliseconden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_serve():
    box_serve = L.code_box("serve", lines=12)
    box_serve
    return (box_serve,)


@app.cell(hide_code=True)
def check_serve(box_serve):
    result_serve = L.check("serve", box_serve.value)
    result_serve
    return (result_serve,)


@app.cell(hide_code=True)
def hints_serve():
    L.hints(
        "serve",
        [
            "`ms` is hoe lang `answer` duurde, uit twee standen van de klok: `start`, gelezen vlak voor `answer`, en een nieuwe stand van `time.perf_counter()`, vlak erna. Het verschil tussen de twee is in seconden.",
            "De latere stand min de eerdere geeft een tijd boven 0. Er zitten 1.000 milliseconden in een seconde: wat maakt dat van een tijd in seconden?",
            "`    ms = (___ - start) * 1000`",
        ],
        why="`ms = (time.perf_counter() - start) * 1000`: de stand na het antwoord min de stand van vlak voor het antwoord, in seconden, keer 1.000 voor milliseconden. Voor het model van de course een paar tiende milliseconde per antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_tables():
    mo.md(r"""
    ### De log als tabel

    Je driftcheck leest de log als tabel, een rij per verzoek: de velden, en het antwoord als de kolom `bikes`. `C.served()` geeft de log van de service van de course voor elk verzoek van 2012, op deze manier gelezen. `C.trained()` geeft de trainingsuren van 2011 als dezelfde tabel, met de antwoorden van de course als `bikes`, en de verhuurde fietsen als `cnt`. `temp` is de temperatuur gedeeld door 41, `hum` de vochtigheid van 0 tot 1, en `windspeed` de wind gedeeld door 67, zoals in de data. Voer de cel uit.
    """)
    return


@app.cell
def example_tables():
    train = C.trained()     # de trainingsuren van 2011, met de antwoorden van de course als de kolom bikes
    served = C.served()     # elk verzoek van 2012 in de log van de service van de course, als tabel
    print(len(train), "training hours,", len(served), "requests")
    print(served[["mnth", "hr", "temp", "hum", "windspeed", "bikes"]].head(3))
    return


@app.cell(hide_code=True)
def text_exercise_drift():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Drift, maand na maand

        `drift_by_month(train, served, column)` geeft de PSI van elke maand van de log, voor één kolom, tegen dezelfde maand van de trainingsuren. `train` is de tabel van de trainingsuren, `served` de tabel van de log, en `column` de naam van een kolom, zoals `"temp"`. `table` begint leeg. De loop gaat één keer over elke maand in de log, op volgorde, met `sorted(set(served["mnth"]))`. `now` bevat de waarden van `column` in de maand van de log, als NumPy-array, met `.to_numpy()`. `then` hoort hetzelfde te bevatten voor de trainingsuren. `psi_of(then, now)` geeft de PSI, en `table` bewaart de PSI onder `int(month)`. De functie eindigt met `return table`. De import van `psi_of` uit `ops.psi` staat er voor de functie.

        Vervang de `...` door de waarden van `column` in dezelfde maand van de trainingsuren.

        De check roept `drift_by_month(train, served, "temp")` aan, met de tabellen van de cel hierboven; daarna voor de antwoorden, de vochtigheid, en de wind van een log met alleen januari tot en met maart. Onder de check geeft je functie de PSI van elke maand voor de temperatuur en voor de antwoorden. Klopt het, dan bewaart de check je functie als `work/projects/ops/drift.py`.
        """),
        L.task("Vervang de `...` door de waarden van `column` in dezelfde maand van de trainingsuren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_drift():
    box_drift = L.code_box("drift", lines=11)
    box_drift
    return (box_drift,)


@app.cell(hide_code=True)
def check_drift(box_drift):
    result_drift = L.check("drift", box_drift.value)
    result_drift
    return (result_drift,)


@app.cell(hide_code=True)
def hints_drift():
    L.hints(
        "drift",
        [
            "`now` bevat de waarden van `column` in de maand, uit de log, `served`. `then` heeft hetzelfde nodig uit de tabel waarvan het model leerde: `train`. Ook `train` heeft de kolom `\"mnth\"`, dus kiest `month` de rijen, en maakt `.to_numpy()` van de waarden een array.",
            "Een mask kiest rijen: een vergelijking van de kolom `\"mnth\"` van `train` met `month` is True voor de trainingsuren van de maand. Zet de vergelijking tussen de vierkante haken van `train`, en neem dan de kolom zoals `now` het doet.",
            "`        then = train[train[\"mnth\"] == ___][column].to_numpy()`",
        ],
        why="`then = train[train[\"mnth\"] == month][column].to_numpy()`: de waarden van `column` in de trainingsuren van dezelfde maand. Tegen dezelfde maand ligt de temperatuur van 2012 boven 0,25 in januari, februari, maart, mei en november; tegen de hele training zou elke maand erboven liggen.",
    )
    return


@app.cell(hide_code=True)
def text_rentals():
    mo.md(r"""
    ### De verhuurde fietsen komen binnen

    De verhuurde fietsen van 2012 zijn nu binnen: `read_hours(2012)` uit je project geeft elk uur van 2012 met de verhuurde fietsen als `cnt`. `C.answers()` geeft wat de service van de course voor elk van deze uren antwoordde, als tensor, en `C.config()` de `config.json` van het pakket van de course, met `val_rmse`, de RMSE op de validatiedagen. Voer de cel uit.
    """)
    return


@app.cell
def example_rentals():
    from ops.demand import read_hours

    hours = read_hours(2012)     # elk uur van 2012, met de verhuurde fietsen als cnt
    pred = C.answers()           # wat de service van de course voor elk van deze uren antwoordde
    config = C.config()          # config.json van het pakket van de course, als dict
    print(len(hours), "hours,", len(pred), "answers")
    print("version:", config["version"], "  val_rmse:", config["val_rmse"])
    return


@app.cell(hide_code=True)
def text_exercise_retrain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wanneer opnieuw trainen

        `train_again(hours, pred, config)` noemt de maanden waarin het model opnieuw moet trainen. `hours` bevat de uren waarvan de verhuurde fietsen binnen zijn, `pred` wat de service voor elk van deze uren antwoordde, en `config` de dict van `config.json` uit het pakket. `LIMIT` zegt hoeveel keer zo ver als op de validatiedagen een maand ernaast mag zitten: 1,5. `table` komt uit `rmse_per` van je project, per maand: een dict met `"hours"` en `"rmse"` voor elke maand. `months` begint leeg, en de loop gaat over de maanden van `table`, op volgorde. De test na `if` beslist of `months.append(month)` de maand toevoegt. De functie eindigt met `return months`. De import van `rmse_per` uit `ops.report` staat er voor de functie.

        Vervang de `...` door de test: is de RMSE van de maand meer dan `LIMIT` keer de RMSE van het pakket op de validatiedagen?

        De check roept `train_again(hours, pred, config)` aan, met de cel hierboven; daarna met een andere config, met alleen januari tot en met maart, en met een paar eigen uren. Onder de check zie je de RMSE van elke maand en de maanden die je functie noemt. Klopt het, dan bewaart de check je functie als `work/projects/ops/retrain.py`.
        """),
        L.task("Vervang de `...` door de test van een maand tegen `LIMIT` keer de validatie-RMSE van het pakket."),
    ])
    return


@app.cell(hide_code=True)
def exercise_retrain():
    box_retrain = L.code_box("retrain", lines=13)
    box_retrain
    return (box_retrain,)


@app.cell(hide_code=True)
def check_retrain(box_retrain):
    result_retrain = L.check("retrain", box_retrain.value)
    result_retrain
    return (result_retrain,)


@app.cell(hide_code=True)
def hints_retrain():
    L.hints(
        "retrain",
        [
            "`table[month]` is een dict met twee keys: welke van de twee bevat hoe ver het model ernaast zat in de maand? De RMSE van het pakket op de validatiedagen staat in `config`, onder de key `\"val_rmse\"`.",
            "De test is True voor een maand die er te ver naast zit: de RMSE boven de grens, en de grens is `LIMIT` keer de validatie-RMSE. Schrijf de grens met namen, zodat de grens meeverandert met elk nieuw pakket, en laat een maand precies op de grens niet meetellen.",
            "`        if table[month][\"rmse\"] > ___:`",
        ],
        why="`if table[month][\"rmse\"] > LIMIT * config[\"val_rmse\"]:`: een maand telt als de maand er meer dan 1,5 keer zo ver naast zit als op de validatiedagen, 79,56 fietsen per uur voor het pakket van de course. Februari 2012, met 89,5, is de eerste.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Oefening 4. De volgende stap

    In februari 2012 heeft je model een RMSE van 89,5 fietsen per uur, 1,69 keer de 53,04, en op 1 maart, als de verhuurde fietsen van februari binnen zijn, noemt je `train_again` februari.
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next",
        question="Wat moet je project nu doen?",
        options={
            "limit": "`LIMIT` verhogen van 1,5 naar 3, zodat de regel alleen de maanden noemt waarin het model er echt ver naast zit",
            "february": "Het model meteen vervangen door een model dat alleen op de uren van februari 2012 leerde, de nieuwste data",
            "right": "Een nieuwe versie trainen op 2011 en de maanden van 2012 tot nu, testen, en serven als de versie beter is",
            "add": "Het model houden zoals het is, en elk antwoord verdubbelen, want januari en februari 2012 hadden 2,2 keer de verhuur van 2011",
        },
        correct="right",
        why={
            "right": "De regel zegt dat het model niet meer past, en je project heeft al elke stap om dat te herstellen: train een versie op data waarin de verandering zit, voer de tests uit en schrijf de card, en laat de nieuwe versie pas serven als de nieuwe versie er op de laatste uren minder ver naast zit dan de versie die nu draait. Een loop die dit zelf doet, steeds opnieuw, noemt een latere les MLOps.",
            "limit": "Over heel 2012 zou een grens van 3 alleen september en oktober noemen, terwijl het model er van maart tot en met november meer dan twee keer zo ver naast zit als op de validatiedagen. Een grens die je verhoogt tot de regel stil is, verbergt het probleem: wat zegt de regel over het model?",
            "february": "Eén wintermaand heeft geen zomeruren en maar 692 uren in totaal: een model met alleen deze uren weet niets van wat 2011 het oude model leerde. Welke data bevatten zowel wat het model wist als wat er veranderde?",
            "add": "Achteraf bekeken had juli 2012 1,4 keer de verhuur per uur van juli 2011, geen 2,2: één factor voor elke maand zit in de zomer ver ernaast. En met verdubbelde antwoorden is het model een nieuwe versie die niemand testte. Wat doet je project met een nieuwe versie voordat de versie gaat serven?",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_demo(result_drift, result_retrain, result_serve):
    result_serve, result_drift, result_retrain   # opnieuw gemaakt zodra een check je delen bewaart
    mo.md(r"""
    ### Je service, een jaar aan verzoeken

    De knop start je service op 127.0.0.1, met de server van [Je model achter een API](/course/open/sys-serving-api?lang=nl), je pakket uit `work/projects/ops/model` en je `serve`, en stuurt de service elk uur van 2012 als verzoek, maand na maand: 8.734 verzoeken. Elk verzoek laat een regel achter in `work/projects/ops/requests.jsonl`. Na elke maand vergelijkt je `drift_by_month` de maand in de log met dezelfde maand van de training, komen de verhuurde fietsen van de maand binnen, en zegt je `train_again` of het model opnieuw moet trainen. De grafiek laat de RMSE van elke maand zien; met streepjes de RMSE van je pakket op de validatiedagen, en `LIMIT` keer de validatie-RMSE. Ernaast staat wat de service in de maand zag. Het deel van de course neemt de plaats in van elk deel van jou dat ontbreekt of nog niet doet wat de check vraagt. Elke maand duurt een seconde of twee.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Start je service en stuur 2012")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_drift, result_retrain, result_serve, run_button):
    result_serve, result_drift, result_retrain   # opnieuw uitgevoerd met je bestanden zodra een check de bestanden bewaart
    if not run_button.value:
        _out = mo.md("*Druk op de knop om je service te starten en de verzoeken van 2012 te sturen.*")
    else:
        _w = C.live_words(L.lang)
        _now = {}

        def _month(_m, _run):
            if not _now:   # vanaf de eerste maand: de lijnen van streepjes, en wat de service zag naast de grafiek
                for _points, _label in C.references(_run, L.lang):
                    _chart.reference(_points, _label, role="ink-2")
                _chart.watch(lambda: C.month_picture(_now["run"], _now["m"], L.lang))
            _now["run"], _now["m"] = _run, _m
            _chart.add(_m, _run["rmse"][_m], C.progress_line(_run, _m, L.lang))

        _chart = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, 12,
                        role="loss", low=0, window=1, xticks=C.MONTH_TICKS)
        _chart.line(_w["start"])
        _run = C.demo(show=_month)
        if not _run["months"]:
            _out = _chart.stopped(C.summary(_run, L.lang))
        else:
            _out = mo.vstack([
                L.explore("rewind", "maand", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=_run["months"][-1], values=_run["months"], role="",
                          task="Ga terug naar januari, en dan maand voor maand vooruit: in welke maanden komt de driftcheck boven 0,25, en in welke noemt je train_again de maand?"),
                mo.md(C.summary(_run, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Met de bestanden en het pakket van de course noemt je `train_again` februari tot en met december: in februari heeft het model een RMSE van 89,5 fietsen per uur, en van maart tot en met november meer dan twee keer zo ver als op de validatiedagen, tot 165,6 in september. De driftcheck van de inputs kwam in zeven maanden boven 0,25, januari tot en met mei, september en november, en volgde de fout niet. Januari, met na maart de grootste drift van de temperatuur, had de kleinste fout. Juni, juli, augustus, oktober en december, zonder input boven 0,25, zaten er 1,84 tot 3,10 keer zo ver naast. De antwoorden van het model bleven in elke maand onder 0,1: de antwoorden hangen het meest af van het uur van de dag, dus verschoof zelfs een warmere maand de antwoorden weinig. Wat het meest veranderde was het aantal fietsers, en alleen de verhuurde fietsen laten dat zien, zodra de tellingen binnen zijn.

        Op de computer van de course mat `serve` ongeveer 0,3 ms per antwoord, en wachtte de client ongeveer 1,5 ms per verzoek: de rest van het wachten is de verbinding, de JSON en de thread van de server, buiten `serve`. Een latency in een log is altijd de latency van het stuk tussen de twee standen van de klok.
        """),
        L.watch_out("Meet met `time.perf_counter()`, niet met `time.time()`. De klok van `time.time()` volgt de tijd van de dag, en een computer kan de klok vooruit of achteruit zetten terwijl een verzoek loopt, zoals bij het gelijkzetten met een tijdserver: één latency in de log komt dan onder 0 uit, of veel te groot."),
    ])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De server van [Je model achter een API](/course/open/sys-serving-api?lang=nl) antwoordt nu met je eigen pakket, en elk verzoek laat een gestructureerde regel achter in de log, zoals in [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl). De PSI van [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl) draait maand na maand op wat je service binnenkreeg, en je `rmse_per` uit [Project: tests en een model card](/course/open/sys-project-tested-model?lang=nl) meet elke maand zodra de verhuurde fietsen binnen zijn. Je project *ops* heeft drie bestanden meer, `service.py`, `drift.py` en `retrain.py`, en een log, `requests.jsonl`. In de volgende les, *De computer van een ander: de cloud*, zou je service kunnen draaien op een computer die je huurt in plaats van je eigen computer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_next, result_drift, result_retrain, result_serve):
    L.wrap_up(result_serve, result_drift, result_retrain, choice_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
