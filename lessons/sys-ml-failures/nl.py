# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-ml-failures. Route: course/route/lessen/systemen.yaml. Checks, het model, de dagen van 2011 en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-ml-failures", "nl")
    C = L.checks   # het model van de course voor het project, de antwoorden voor elk uur van 2011, de verzoeken en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="Een alertregel zegt: de p99 boven 300 ms gedurende 3 minuten achter elkaar. De p99 van de laatste vijf minuten is 120, 950, 130, 320 en 330 ms. Gaat de regel af bij de laatste minuut?",
        options={
            "spike": "Ja: de minuut van 950 ms ligt ver boven 300",
            "most": "Ja: drie van de vijf minuten liggen boven 300, zoveel als de regel vraagt",
            "right": "Nee: een van de laatste 3 minuten, met 130 ms, ligt onder 300",
            "five": "Nee: een regel kijkt altijd naar 5 minuten, niet naar 3",
        },
        correct="right",
        why={
            "right": "De regel kijkt naar de laatste 3 minuten: 130, 320 en 330 ms. Eén daarvan ligt onder 300, dus blijft de regel stil; ligt de volgende minuut ook boven 300, dan gaat de regel af. Zulke regels houden de service in de gaten. In deze les houden andere getallen het model in de gaten.",
            "spike": "Eén minuut van 950 ms is een piek. Gaat de regel af op één minuut, of op minuten achter elkaar? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
            "most": "De drie minuten boven 300 liggen niet achter elkaar: 130 ligt ertussen. Naar welke minuten kijkt de regel? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
            "five": "Het aantal minuten hoort bij de regel, en deze regel zegt 3. Naar welke 3 minuten kijkt de regel? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Een service logt elk verzoek als een regel JSON met de velden `time`, `status`, `ms`, `temp` en `bikes`. Welke vraag kan een programma beantwoorden met alleen de log?",
        options={
            "rented": "Hoeveel fietsen er om 16:00 echt verhuurd werden",
            "right": "Hoeveel antwoorden boven 5.000 fietsen de service om 16:00 gaf",
            "warm": "Hoe warm het om 16:00 echt was, wat de app ook stuurde",
            "who": "Welke persoon het traagste verzoek van de dag stuurde, en vanaf waar",
        },
        correct="right",
        why={
            "right": "Elke regel bevat het antwoord onder `bikes`, dus kan een programma per minuut de antwoorden boven 5.000 tellen. In de log staat niet wat er echt verhuurd werd, en ook niet hoe warm het echt was: het eerste is pas later bekend, en `temp` is wat de app stuurde, goed of fout. Om dat verschil draait deze les.",
            "rented": "De log bevat wat de service antwoordde, niet wat er in het uur gebeurde. Welk veld bevat het antwoord? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
            "warm": "`temp` is wat de app stuurde, en dat kan fout zijn, zoals graden in plaats van de schaal van de data. Welke vraag heeft alleen de velden van de log nodig? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
            "who": "De velden bevatten geen persoon en geen plaats. Welke vraag heeft alleen `time`, `status`, `ms`, `temp` en `bikes` nodig? Zie [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een programma dat stukgaat laat dat meestal merken. Een server zonder vrij geheugen stopt, een ontbrekend bestand geeft een fout, een back-up maakt elk antwoord traag, en de getallen uit [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl) springen: fouten, verkeer, latency. In een talk op de conferentie OpML in 2020 liepen Daniel Papasian en Todd Underwood van Google tien jaar aan storingen van één grote ML-pipeline door, en de meeste storingen bleken helemaal niet over ML te gaan, maar over de vele machines en stappen eromheen.

    Een ML-systeem kan ook ongemerkt falen. Het systeem blijft antwoorden, met status 200 en even snel als altijd, alleen slechter: een input stroomopwaarts gaat stuk, er komt een dag die op niets uit de training lijkt, of de wereld schuift weg van de data waarvan het model leerde. Geen foutmelding zegt het je; alleen de eigen getallen van het model doen dat.

    **Een voorbeeld uit het dagelijks leven.** Een brandstofmeter die op half vol blijft hangen geeft geen waarschuwing. De auto rijdt door, de naald ziet er normaal uit, en niets piept, tot de tank leeg is. Alleen een tweede getal laat het zien: de kilometers sinds de laatste keer tanken.
    """)
    return


@app.cell(hide_code=True)
def predict_heat():
    guess_heat = L.guess(
        "heat",
        question="Op zaterdag 23 juli 2011 trof een hittegolf Washington: 's middags 38,5 graden volgens de data, en er werden 3.285 fietsen verhuurd. Het model van het project leerde uit 2011 dat warmere dagen meer fietsers brengen. Hoeveel fietsen voorspelde het model voor de hele dag?",
        options={
            "half": "Ongeveer 1.600: in zulke hitte de helft",
            "close": "Ongeveer 3.300: dicht bij wat er verhuurd werd",
            "right": "Ongeveer 6.800: meer dan twee keer zoveel",
            "none": "Geen: het model stopt met een fout, want 38,5 graden ligt buiten de training",
        },
        reveal="**Ongeveer 6.800: meer dan twee keer zoveel.** Het model antwoordde 6.824 fietsen voor de hele dag, terwijl er 3.285 verhuurd werden: 147,5 fietsen per uur ernaast, bijna zes keer zoveel als op een gewone dag. In 2011 brachten warmere dagen meer fietsers, tot op zekere hoogte; bij 38,5 graden bleven veel mensen uit de zon. Het model zag te weinig van zulke uren om dat te leren: alleen 21 van de 6.912 uren uit de training waren 37 graden of warmer, en 23 juli is zelfs een van de trainingsdagen. En 38,5 graden ligt binnen wat de training zag, dus hield niets de verzoeken tegen: de service beantwoordde elk uur met status 200, even snel als altijd.",
    )
    guess_heat
    return (guess_heat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Twee manieren om te falen.** Een *softwarefout* laat de service stukgaan of vertraagt de service: een crash, een ontbrekend bestand, een vol geheugen, een server met te veel werk. De service antwoordt met een fout, te laat, of helemaal niet, en de getallen over de service laten het zien: fouten, verkeer, de p99. Een ML-systeem kan ook op een eigen manier falen. De afbeelding toont beide, voor hetzelfde uur.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Silent failure.** Een **silent failure** is een storing waarbij het systeem blijft antwoorden, met status 200 en even snel als altijd, terwijl de antwoorden slechter worden. Aan de service verandert niets, dus ziet een softwaremonitor niets. Er zijn vier soorten oorzaken:

        - *Een input gaat stuk of verandert stroomopwaarts*, in een stap die de inputs voor het model maakt. Op 10 maart 2011 stuurde de vochtigheidssensor 22 uur lang 0, terwijl het regende. Een app stuurt graden, een stap stuurt ineens km/u, of een stap zet lokale tijd om naar UTC. Het verzoek is goed gevormd; alleen de betekenis veranderde.
        - *Een edge case*: een dag die op niets uit de training lijkt.
        - *De wereld schuift weg van de training data*, zoals veel meer fietsers een jaar later. Dat is de volgende les.
        - *Het model verandert de eigen data*, zoals een aanbevelingssysteem dat alleen laat zien wat al populair is. Dat is de les daarna.

        **Edge case.** Een **edge case** is een input aan de rand van wat het model kent: zeldzaam in de training data, of afwezig, zoals een hittegolf, een orkaan of een feestdag. Het model antwoordt toch, en het antwoord kan ver ernaast zitten: het model past wat het op gewone dagen leerde toe op een dag die niet gewoon is. In 2011 brachten warmere dagen meer fietsers, dus verwacht het model bij 38,5 graden nog meer. Alleen 21 van de 6.912 uren uit de training waren 37 graden of warmer, en alleen 5 van de 292 trainingsdagen waren feestdagen.

        **Twee checks vanaf de kant van het model.** Twee getallen van het model vangen veel silent failures. Het eerste getal vergelijkt elk verzoek met de training: ligt elke input binnen het bereik dat de training zag? Het werkt meteen, bij het eerste verzoek, en vangt een eenheid die veranderde: 31,2 graden ligt ver boven de 0,96 uit de training. De hitte van 23 juli komt erdoor, want 38,5 graden ligt binnen het bereik. Het tweede getal wacht tot bekend is wat er echt gebeurde: zodra het uur voorbij is en de verhuurde fietsen geteld zijn, hoe ver zat het model ernaast, vergeleken met een gewone dag? Het vangt alles waardoor de antwoorden slechter worden, alleen pas achteraf. Je schrijft beide, in Oefening 1 en 2.

        **Op grote schaal.** Libraries zoals TensorFlow Data Validation houden een schema van de training data bij, met het type van elke feature en de waarden of het bereik dat de feature mag hebben, en controleren elke nieuwe batch daartegen: je eerste check, voor alle features tegelijk. Tools voor model monitoring vergelijken hoe ver een model ernaast zit met eerdere weken, zodra bekend is wat er echt gebeurde. En teams schrijven elke storing op, met de oorzaak en de check die de storing eerder had gevangen, in een verslag dat een *postmortem* heet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **silent failure** | | een storing waarbij een systeem blijft antwoorden, snel en met status 200, terwijl de antwoorden slechter worden, zodat alleen de eigen getallen van het model het laten zien | |
        | **edge case** | | een input die zeldzaam is of ontbreekt in de training data, zoals een hittegolf of een feestdag, waarop een model ver ernaast kan zitten | |
        | stroomopwaarts (upstream) | | in de stappen die de inputs van het model maken, zoals een sensor, een weerdienst of een app | |
        """),
        L.watch_out("Ook de training data kan dezelfde storing bevatten. 10 maart 2011 is een van de trainingsdagen, dus begint de vochtigheid in de training al bij 0, en laat een check tegen het bereik van de training de kapotte sensor door."),
        L.watch_out("Je weet pas hoe ver het model ernaast zat als bekend is wat er echt gebeurde: hier een uur later, als de verhuurde fietsen geteld zijn. Bij veel modellen duurt dat veel langer, zoals bij de vraag of een lening wordt terugbetaald, of het gebeurt nooit, bij een lening die geweigerd werd."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Uitzoeken hoe een ML-systeem kan falen:

    1. Zet op een rij wat het systeem nodig heeft: de server en de code, het bestand van het model, elke input en waar de input vandaan komt, en de wereld waarvan het model leerde.
    2. Houd de service in de gaten met de eigen getallen van de service, zoals fouten, verkeer en de p99: deze getallen vangen de luide storingen.
    3. Controleer elk verzoek tegen de training: ligt elke input binnen het bereik dat de training zag?
    4. Bereken per dag hoe ver het model ernaast zat, vergeleken met een gewone dag, zodra bekend is wat er echt gebeurde.
    5. Bekijk de slechtste dagen één voor één: een kapotte input, een edge case, of een wereld die veranderde?
    6. Schrijf elke storing op, met de oorzaak en de check die de storing eerder had gevangen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.SENSOR` is het verzoek dat de service kreeg voor 8:00 op 10 maart 2011: een dict met het uur en de tien features. `C.answer(request)` geeft wat de service antwoordt, de fietsen van het model, in hele fietsen en nooit onder 0, en `C.rented(day, hour)` de verhuurde fietsen, met `C.SENSOR_DAY` voor 10 maart. De cel print het verzoek, het antwoord en de verhuurde fietsen, en dan het antwoord met de vochtigheid van andere regenachtige uren, 0,88. Voer de cel uit.
    """)
    return


@app.cell
def example_sensor():
    print(dict(C.SENSOR))
    print("antwoord", C.answer(C.SENSOR), "verhuurd", C.rented(C.SENSOR_DAY, 8))
    _fixed = dict(C.SENSOR)
    _fixed["hum"] = 0.88                 # de vochtigheid van andere regenachtige uren
    print("antwoord met hum 0,88", C.answer(_fixed))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Om 8:00 op 10 maart regende het (`weathersit` 3), stuurde de vochtigheidssensor 0, en antwoordde het model 226 fietsen, terwijl er 44 verhuurd werden. Met een vochtigheid van 0,88, zoals in andere regenachtige uren, antwoordt het model 190: nog steeds te veel, want regen houdt meer mensen thuis dan het model verwacht, maar wel 36 minder. Over de hele dag antwoordde het model 2.137 fietsen, met 0,88 waren het er 1.795, en verhuurd werden er 623. De service beantwoordde elk uur met status 200, even snel als altijd. In Oefening 1 schrijf je een check die elk verzoek bekijkt voordat het model dat doet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Buiten wat de training zag

        `unseen(request, low, high)` krijgt een `request`, een dict met het uur en de tien features, en twee dicts met een getal voor elk van `temp`, `atemp`, `hum` en `windspeed`: `low`, de laagste waarde in de uren van de training, en `high`, de hoogste. De loop gaat over elke `name` in `low`. De functie geeft de `name` terug van de eerste input buiten wat de training zag, of `None` als elke input binnen de training ligt. Vervang de `...` zodat de voorwaarde klopt als de waarde van de input onder de laagste waarde uit de training ligt of boven de hoogste.

        De check roept `unseen(C.NEW_APP, C.LOW, C.HIGH)` aan: het verzoek van de nieuwe app voor 16:00 op 14 juli, met de temperatuur in graden. Dan een gewone ochtend, een heet uur uit 2012, een uur van de hitte van 23 juli, de wind in km/u, een sensor die −1 stuurt als er geen meting is, een ochtend kouder dan elk uur uit de training, een ochtend zonder wind, en een regenachtige ochtend met een vochtigheid van 1.
        """),
        L.task("Laat de voorwaarde kloppen als de input onder de laagste waarde uit de training ligt of boven de hoogste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unseen():
    box_unseen = L.code_box("unseen", lines=5)
    box_unseen
    return (box_unseen,)


@app.cell(hide_code=True)
def check_unseen(box_unseen):
    result_unseen = L.check("unseen", box_unseen.value)
    result_unseen
    return (result_unseen,)


@app.cell(hide_code=True)
def hints_unseen():
    L.hints(
        "unseen",
        [
            "Stap 3 van het recept: `request[name]` is de waarde van de input in het verzoek, `low[name]` de laagste waarde uit de training en `high[name]` de hoogste. De eerste helft, `request[name] < low[name]`, vangt een waarde onder het bereik. Welke vergelijking vangt een waarde erboven?",
            "Een waarde ligt erbuiten als een van beide helften klopt, dus verbindt `or` de twee helften. De tweede helft vergelijkt `request[name]` met `high[name]`. Met welk teken?",
            "`        if request[name] < low[name] or ___:`",
        ],
        why="`if request[name] < low[name] or request[name] > high[name]:`: de waarde ligt onder de laagste waarde uit de training of boven de hoogste. De 31,2 van de nieuwe app ligt ver boven de 0,96 uit de training, dus geeft `unseen` `'temp'`; een gewone ochtend geeft `None`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoe ver ernaast, per dag

        Zodra een uur voorbij is, is bekend hoeveel fietsen er verhuurd werden, en kan het antwoord van het model ernaast gelegd worden. `off(predicted, rented)` krijgt twee lists met een getal voor elk uur van een dag: `predicted`, de antwoorden van de service, en `rented`, de verhuurde fietsen. De loop gaat over elk uur `h`, met `range` tot `len(rented)`, en `total` telt op hoe ver het model ernaast zat; de functie deelt `total` door het aantal uren. Vervang de `...` zodat `total` groeit met hoe ver het model er in uur `h` naast zat, omhoog of omlaag.

        De check roept `off([120, 80, 30], [100, 95, 30])` aan: 20 te veel, 15 te weinig, en precies goed. Dan de dagen 14 juli, 23 juli en 10 maart, en twee korte lists.
        """),
        L.task("Laat `total` groeien met hoe ver het model er in uur `h` naast zat, te hoog of te laag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_off():
    box_off = L.code_box("off", lines=5)
    box_off
    return (box_off,)


@app.cell(hide_code=True)
def check_off(box_off):
    result_off = L.check("off", box_off.value)
    result_off
    return (result_off,)


@app.cell(hide_code=True)
def hints_off():
    L.hints(
        "off",
        [
            "Stap 4 van het recept: in uur `h` antwoordde het model `predicted[h]` en werden er `rented[h]` fietsen verhuurd. Het verschil is positief als het model te hoog zat en negatief als het model te laag zat. De regel telt een getal op bij `total`; welk getal?",
            "Te hoog en te laag moeten allebei meetellen: `abs` maakt van een negatief getal een positief getal, zoals `abs(-15)` 15 geeft. Welk verschil zet je tussen de haakjes?",
            "`        total = total + ___`",
        ],
        why="`total = total + abs(predicted[h] - rented[h])`: hoe ver het model er in uur `h` naast zat, te hoog of te laag. Voor de korte lists: (20 + 15 + 0) / 3 = 11,7 fietsen per uur. Zonder `abs` heffen 20 te veel en 15 te weinig elkaar grotendeels op, tot 1,7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Oefening 3. Welke check ziet het eerst?

    Op donderdag 14 juli 2011 begint een stap stroomopwaarts `windspeed` in km/u te sturen, terwijl de data en de training km/u gedeeld door 67 bevatten. Om 8:00 bevat het verzoek 24,0 in plaats van 0,36. De service beantwoordt elk verzoek met status 200, even snel als altijd.
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="Welke check ziet het eerst?",
        options={
            "tail": "De p99 van de latency van elke minuut",
            "errors": "Het aandeel antwoorden met status 400 of 500 per minuut",
            "off": "Je `off` op elke dag",
            "right": "Je `unseen` op elk verzoek",
        },
        correct="right",
        why={
            "right": "`unseen` geeft `'windspeed'` bij het allereerste verzoek: 24,0 ligt ver boven de 0,85 uit de training. Ook `off` ziet het, alleen pas als de uren voorbij zijn: het model antwoordt 17.202 fietsen voor de hele dag, terwijl er 5.084 verhuurd werden, 505 per uur ernaast in plaats van 26. De p99 en de fouten houden de service in de gaten, en met de service is niets mis.",
            "tail": "Het model beantwoordt een verzoek met 24,0 even snel als een verzoek met 0,36. Zou een trage minuut het laten zien?",
            "errors": "24,0 is een goed gevormd getal, beantwoord met status 200. Telt dat als fout?",
            "off": "`off` heeft de verhuurde fietsen nodig, en het aantal verhuurde fietsen is pas bekend als elk uur voorbij is. Welke check bekijkt het verzoek zelf, voordat het model antwoordt?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_failures():
    mo.md(r"""
    ### Oefening 4. Luid of stil?

    Acht storingen die de service van het project kunnen overkomen. Bepaal voor elke storing waaraan je de storing ziet: aan een softwaremonitor, met de getallen over de service, of alleen aan de eigen getallen van het model.
    """)
    return


@app.cell(hide_code=True)
def sort_failures_cell():
    sort_failures = L.sort("failures")
    sort_failures
    return (sort_failures,)


@app.cell(hide_code=True)
def check_failures(sort_failures):
    result_failures = L.check("failures", sort_failures)
    result_failures
    return (result_failures,)


@app.cell(hide_code=True)
def hints_failures():
    L.hints(
        "failures",
        [
            "Vraag bij elke storing: beantwoordt de server nog elk verzoek, snel en met status 200?",
            "Vier storingen laten de service stukgaan of vertragen de service. Bij de andere vier zijn de verzoeken goed gevormd en komen de antwoorden snel: alleen het aantal fietsen in het antwoord klopt niet.",
            "Monitor: geen geheugen, een ontbrekend pakket, \\_\\_\\_ en \\_\\_\\_. Model: een kapotte sensor, graden, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Een softwaremonitor:** geen geheugen, een ontbrekend pakket, een back-up en een ontbrekend bestand, want de service gaat stuk of wordt trager. **Alleen de getallen van het model:** een kapotte sensor, graden, een hittegolf en een andere klok, want elk verzoek krijgt een snel antwoord met status 200, alleen een fout antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Zes dagen uit 2011

    Kies een dag. De afbeelding toont de verhuurde fietsen in elk uur, en wat de service antwoordde: het model van het project, in hele fietsen, nooit onder 0.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_day = mo.ui.dropdown(
        options={
            "Donderdag 14 juli: een gewone dag": "usual",
            "Donderdag 10 maart: de kapotte sensor": "sensor",
            "Zaterdag 23 juli: de hittegolf": "heat",
            "Zaterdag 27 augustus: orkaan Irene": "irene",
            "Maandag 4 juli: Independence Day": "july4",
            "Vrijdag 25 november: na Thanksgiving": "thanksgiving",
        },
        value="Donderdag 14 juli: een gewone dag",
        label="Dag",
    )
    mo.vstack([
        demo_day,
        L.task("Begin met de gewone dag, en bekijk dan de andere vijf. Op welke dagen antwoordt het model te veel, en op welke te weinig? Welke dagen had je `unseen` gevangen?"),
    ])
    return (demo_day,)


@app.cell(hide_code=True)
def demo_picture(demo_day):
    mo.Html(C.demo_picture(demo_day.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Op 14 juli volgt het model de spitsuren en zit het er 25,7 fietsen per uur naast. Op 10 maart antwoordt het model veel te veel: regen, en de kapotte sensor erbovenop. Op 23 juli hield de hitte mensen thuis, en antwoordt het model meer dan twee keer zoveel als er verhuurd werd. Op 27 augustus bereikte orkaan Irene Washington: weer antwoordt het model meer dan twee keer zoveel, en de data stopt om 17:00. Op 4 juli, Independence Day, antwoordt het model te weinig: er fietsten meer mensen dan het model voor een feestdag verwacht, en de training bevat alleen 5 feestdagen. Op 25 november, de dag na Thanksgiving, antwoordt het model te veel: in de data een werkdag als alle andere, terwijl veel mensen vrij waren. Je `unseen` vangt geen van de zes dagen, want elke input ligt binnen de training; alleen hoe ver het model ernaast zat laat de dagen zien.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl) vingen de getallen over de service een pauze en een vertraging, en liet het gemiddelde van de antwoorden zien dat een nieuwe app graden stuurde; in deze les hebben zulke storingen een naam, en twee checks vanaf de kant van het model. De edge cases van deze les zijn losse dagen. In de volgende les, *Als de wereld verandert: drift*, schuift de wereld zelf weg van 2011: in 2012 fietsen veel meer mensen, en raakt het model achterop.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_failures, result_off, result_unseen):
    L.wrap_up(result_unseen, result_off, choice_first, result_failures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
